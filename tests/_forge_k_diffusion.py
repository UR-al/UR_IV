"""팩 Extra Samplers 패리티 테스트 공용 — Forge 의 k-diffusion 과 sam-extra 모듈을 앱 venv 에서 읽기 전용으로 돌린다.

* ``forge_k_sampling()`` — Forge Neo 의 ``modules_forge/packages/k_diffusion/sampling.py`` 를 그 파일 텍스트
  그대로 실행한 모듈(캐시). Forge 의 나머지에서 가져오는 두 함수(``backend.patcher.base.
  set_model_options_post_cfg_function``, ``k_diffusion.utils.append_dims``)는 Forge 파일에서 AST 로 잘라
  쓰고, 앱 venv 에 없는 ``torchsde`` 는 부르면 실패하는 대역으로 둔다(이 테스트들은 SDE 경로마다 결정적
  ``noise_sampler`` 를 넘기므로 Brownian tree 를 만들지 않는다). Forge 는 ``_sam_extra_ext.FORGE_ROOT``
  (``SAM3_FORGE_ROOT`` 가 있으면 그 경로만, 없으면 확장 두 단계 위)에서 찾는다. 바이트코드는 쓰지 않는다 —
  사용자의 Forge 설치에 ``__pycache__`` 를 만들지 않는다.
* ``installed_k_sampling(module)`` — sam-extra 가 부를 때 ``importlib.import_module("k_diffusion.sampling")``
  으로 찾는 모듈을 잠시 ``module`` 로 둔다.
* ``import_ext("sam3ext.extra_samplers.er_sde")`` — 확장(``EXT_ROOT``)의 모듈을 import 한다(바이트코드 쓰기 끔:
  오라클 작업 트리·설치본에 ``__pycache__`` 를 남기지 않는다). sam-extra(GPL-3.0)는 **테스트 오라클로만**
  쓰고 팩에는 들어가지 않는다.
* 장난감 모델 — Forge 배치(``model.inner_model.predictor``)와 ComfyUI 배치(``inner_model.model_patcher.
  get_model_object("model_sampling")``), 결정적 잡음(``ReplayNoise``).

torch 는 모듈 최상위에서 import 하지 않는다(``tests/_optional_deps`` 규칙) — 함수 안에서만.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

from tests._sam_extra_ext import EXT_ROOT, FORGE_ROOT, FORGE_ROOT_ENV

K_DIR = Path("modules_forge") / "packages" / "k_diffusion"
SAMPLING = K_DIR / "sampling.py"
_PACKAGE = "_aistudio_forge_k_diffusion"
_CACHE: dict = {}


def forge_available() -> bool:
    return FORGE_ROOT is not None and (FORGE_ROOT / SAMPLING).is_file()


def require_forge() -> Path:
    """Forge 체크아웃 — 없으면 SkipTest(확장 강제 모드에서도: Forge 는 확장이 아니라 그 호스트다)."""
    if not forge_available():
        raise unittest.SkipTest(
            f"Forge 의 {SAMPLING.as_posix()} 를 찾지 못함 — FORGE_ROOT={FORGE_ROOT} "
            f"(확장 작업 트리를 오라클로 쓰면 {FORGE_ROOT_ENV} 로 Forge 를 지정)")
    return FORGE_ROOT


@contextlib.contextmanager
def stub_modules(stubs: dict):
    """``sys.modules`` 의 지정한 이름만 바꿨다가 되돌린다."""
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def forge_definition(rel_path: str | Path, name: str) -> str:
    """Forge 파일의 최상위 함수 ``name`` 소스(데코레이터 포함)."""
    path = require_forge() / rel_path
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            decorators = "".join(f"@{ast.get_source_segment(source, d)}\n" for d in node.decorator_list)
            return decorators + ast.get_source_segment(source, node)
    raise LookupError(f"{name} not found in {rel_path}")


def _torchsde_stand_in() -> types.ModuleType:
    module = types.ModuleType("torchsde")

    def BrownianTree(*_args, **_kwargs):
        raise RuntimeError("torchsde is not installed in the app venv; pass a noise_sampler")

    module.BrownianTree = BrownianTree
    module.BrownianInterval = BrownianTree
    return module


def forge_k_sampling():
    """Forge Neo 의 ``k_diffusion.sampling`` 모듈(Forge 파일 텍스트를 실행, 캐시)."""
    if "sampling" in _CACHE:
        return _CACHE["sampling"]
    root = require_forge()
    base_ns: dict = {}
    exec(compile(forge_definition("backend/patcher/base.py", "set_model_options_post_cfg_function"),
                 "backend/patcher/base.py", "exec"), base_ns)
    utils_ns: dict = {}
    exec(compile(forge_definition(K_DIR / "utils.py", "append_dims"), "k_diffusion/utils.py", "exec"), utils_ns)

    backend = types.ModuleType("backend")
    backend.__path__ = []
    patcher = types.ModuleType("backend.patcher")
    patcher.__path__ = []
    base = types.ModuleType("backend.patcher.base")
    base.set_model_options_post_cfg_function = base_ns["set_model_options_post_cfg_function"]
    package = types.ModuleType(_PACKAGE)
    package.__path__ = []
    utils = types.ModuleType(_PACKAGE + ".utils")
    utils.append_dims = utils_ns["append_dims"]
    package.utils = utils
    stubs = {"backend": backend, "backend.patcher": patcher, "backend.patcher.base": base,
             _PACKAGE: package, _PACKAGE + ".utils": utils}
    if importlib.util.find_spec("torchsde") is None:
        stubs["torchsde"] = _torchsde_stand_in()
    path = root / SAMPLING
    module = types.ModuleType(_PACKAGE + ".sampling")
    module.__file__ = str(path)
    module.__package__ = _PACKAGE
    with stub_modules(stubs):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    _CACHE["sampling"] = module
    return module


@contextlib.contextmanager
def installed_k_sampling(module):
    """sam-extra 가 호출 시점에 찾는 ``k_diffusion.sampling`` 을 ``module`` 로."""
    package = types.ModuleType("k_diffusion")
    package.__path__ = []
    package.sampling = module
    with stub_modules({"k_diffusion": package, "k_diffusion.sampling": module}):
        yield module


def import_ext(name: str):
    """확장(EXT_ROOT)의 모듈 ``name`` — 바이트코드를 쓰지 않고 import 한다."""
    if EXT_ROOT is None:
        raise unittest.SkipTest("sam-extra 확장 미설치")
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(name)
    finally:
        sys.dont_write_bytecode = previous


def ext_source(relative: str) -> str:
    return (EXT_ROOT / relative).read_text(encoding="utf-8")


def ext_literals(relative: str) -> dict:
    """확장 파일의 모듈 수준 ``NAME = <리터럴>`` 표(앱 SEMANTIC_PINS 와 같은 읽기)."""
    values = {}
    for node in ast.parse(ext_source(relative)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def load_ext_file(relative: str, name: str):
    """torch 없이 읽을 수 있는 확장 파일 하나(상대 import 없음)를 패키지 초기화 없이 실행한 모듈."""
    path = EXT_ROOT / relative
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module   # dataclasses 가 모듈을 찾는다
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# ---------------------------------------------------------------------------
# 장난감 모델 (torch 는 함수 안에서)
# ---------------------------------------------------------------------------


class Predictor:
    """Forge 의 예측기 대역: ``prediction_type`` 과 ``percent_to_sigma``(flow 는 shift 식, eps 는 로그 선형)."""

    def __init__(self, flow: bool, shift: float = 3.0, sigma_min: float = 0.0292, sigma_max: float = 14.6146,
                 percent_to_sigma: bool = True):
        self.prediction_type = "const" if flow else "eps"
        self.flow = flow
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if not percent_to_sigma:
            self.percent_to_sigma = None

    def percent_to_sigma(self, percent):  # noqa: E301 - replaced by None above when absent
        import math

        if percent <= 0.0:
            return 1.0 if self.flow else 999999999.9
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent
        if self.flow:
            return self.shift * t / (1 + (self.shift - 1) * t)
        return math.exp((math.log(self.sigma_max) - math.log(self.sigma_min)) * t + math.log(self.sigma_min))


class ComfyPatcher:
    """ComfyUI 배치: ``inner_model.model_patcher.get_model_object("model_sampling")``."""

    def __init__(self, sampling):
        self.sampling = sampling

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self.sampling


class ToyModel:
    """결정적 원소별 x0 예측기. ``calls`` 에 (σ, x 모양, model_options) 를 남긴다."""

    def __init__(self, sampling, comfy_layout: bool = False, bias: float = 0.05):
        inner = types.SimpleNamespace()
        if comfy_layout:
            inner.model_patcher = ComfyPatcher(sampling)
        else:
            inner.predictor = sampling
        self.inner_model = inner
        self.bias = bias
        self.calls = []

    def __call__(self, x, sigma, **extra_args):
        import torch

        self.calls.append((float(sigma.reshape(-1)[0]), tuple(x.shape), extra_args.get("model_options")))
        s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
        return torch.tanh(x) * 0.8 / (1.0 + s) + self.bias * s - 0.03 * x * x


class ReplayNoise:
    """결정적 잡음 샘플러: 같은 seed 면 같은 순서로 같은 값. ``draws`` 에 (σ, σ_next) 를 남긴다."""

    def __init__(self, like, seed: int = 1234):
        import torch

        self.like = like
        self.generator = torch.Generator().manual_seed(seed)
        self.draws = []

    def __call__(self, sigma, sigma_next):
        import torch

        self.draws.append((float(sigma), float(sigma_next)))
        return torch.randn(self.like.shape, generator=self.generator, dtype=torch.float64).to(self.like.dtype)
