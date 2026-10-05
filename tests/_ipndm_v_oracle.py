"""IPNDM_V 대조 공용 — 팩 ``ipndm_v_fixed`` 와 비교할 sam-extra 함수, Anima Linear Quadratic 28 스텝 σ 목록, 등속 탐침.

sam-extra 0.33.1 은 ``sam3ext/extra_samplers/ipndm_deis.py`` 의 ``coeff4`` 줄 끝을 ``* h_n_1 / h_n_2`` 에서
``* h_n_2 / h_n_3`` 로 고쳤다(상류 zju-pi ``ipndm_v_sampler`` 와 그것을 옮긴 ComfyUI ``sample_ipndm_v`` 의 오타 — 3계 차분의
마지막 항은 ``-q·h_{n-2}/h_{n-3}``). 오타가 있으면 4차 계수의 합이 1 이 아니어서(``1 + q·temp2·(h_{n-2}/h_{n-3} −
h_{n-1}/h_{n-2})``) Linear Quadratic 28 스텝의 16번째 스텝에서 약 −174 가 되고 그림이 녹색 잡음이 된다.

``ipndm_v_oracle(module)`` — 오라클(``EXT_ROOT``)의 파일이 고친 줄을 가졌으면 설치된 ``sample_ipndm_v`` 그대로. 레지스트리
기준(``core.sam_extra_contract.EXT_VERSION_AUDITED``)이 아직 0.33.1 보다 낮고 오라클이 오타 줄을 가졌으면(동기화 전 설치된
0.33.0), 그 파일 원문에서 **그 한 줄만** 0.33.1 의 줄로 바꾼 사본을 오라클 패키지 안에서 실행한 ``sample_ipndm_v`` —
나머지는 오라클의 코드 그대로다. 기준이 0.33.1 이상인데 오타 줄이면, 또는 두 줄이 다 없거나 다 있으면 실패한다(확장이 그
함수를 바꿨다 — 팩을 다시 맞춰야 한다).

torch 는 모듈 최상위에서 import 하지 않는다(``tests/_optional_deps`` 규칙).
"""
from __future__ import annotations

from tests._forge_k_diffusion import ext_source

IPNDM_DEIS = "sam3ext/extra_samplers/ipndm_deis.py"
TYPO_LINE = "coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2"
FIXED_LINE = "coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_2 / h_n_3"
FIX_VERSION = (0, 33, 1)

# ComfyUI calculate_sigmas(ModelSamplingDiscreteFlow shift 3 + CONST, "linear_quadratic", 28), float32 — Anima 의 10-05 GPU
# 확인과 같은 목록(scratchpad exposure/impl/gpu/part3/sigmas_cpu_lq.json "lq28"; 실제 ComfyUI 와의 일치는
# tests/test_comfy_extra_samplers_runtime.py 가 본다). 16번째 스텝(인덱스 15)이 σ 0.968 → 0.952.
LQ28 = (
    1.0, 0.9982143044471741, 0.9964285492897034, 0.9946428537368774, 0.9928571581840515, 0.9910714030265808,
    0.9892857074737549, 0.987500011920929, 0.9857142567634583, 0.9839285612106323, 0.9821428656578064,
    0.9803571701049805, 0.9785714149475098, 0.9767857193946838, 0.9750000238418579, 0.968367338180542,
    0.9520407915115356, 0.9260203838348389, 0.8903061151504517, 0.844897985458374, 0.7897959351539612,
    0.7250000238418579, 0.6505101919174194, 0.5663265585899353, 0.4724489748477936, 0.3688775599002838,
    0.25561225414276123, 0.13265305757522583, 0.0,
)


def _version(text: str) -> tuple:
    return tuple(int(part) for part in str(text).split(".")[:3])


def ipndm_v_oracle(module) -> tuple:
    """``(sample_ipndm_v, how)`` — ``how`` 는 'installed'(0.33.1 이상) 또는 'one-line fix of the installed file'."""
    from core import sam_extra_contract

    source = ext_source(IPNDM_DEIS).replace("\r\n", "\n")
    fixed, typo = source.count(FIXED_LINE), source.count(TYPO_LINE)
    if (fixed, typo) == (1, 0):
        return module.sample_ipndm_v, "installed"
    audited = _version(sam_extra_contract.EXT_VERSION_AUDITED)
    if (fixed, typo) == (0, 1) and audited < FIX_VERSION:
        namespace = dict(module.__dict__)
        namespace["__name__"] = module.__name__ + "_fixed_by_test"
        exec(compile(source.replace(TYPO_LINE, FIXED_LINE), module.__file__, "exec"), namespace)
        return namespace["sample_ipndm_v"], "one-line fix of the installed file"
    raise AssertionError(
        f"{IPNDM_DEIS}: 고친 coeff4 줄 {fixed}개, 오타 줄 {typo}개 (레지스트리 기준 {sam_extra_contract.EXT_VERSION_AUDITED}) — "
        "sam-extra 의 IPNDM_V 가 0.33.1 의 한 줄 수정과 다르다. 팩 extra_samplers_ipndm 을 다시 맞춰라")


def with_line(module, line: str):
    """오라클 파일의 coeff4 줄을 ``line`` 으로 둔 ``sample_ipndm_v``(오타 판·고친 판을 실제 코드로 돌린다)."""
    source = ext_source(IPNDM_DEIS).replace("\r\n", "\n")
    current = FIXED_LINE if FIXED_LINE in source else TYPO_LINE
    namespace = dict(module.__dict__)
    namespace["__name__"] = module.__name__ + "_line_by_test"
    exec(compile(source.replace(current, line), module.__file__, "exec"), namespace)
    return namespace["sample_ipndm_v"]


def upstream_ipndm_v_weights(h_n, h_n_1, h_n_2, h_n_3, *, fixed: bool):
    """zju-pi/ComfyUI 의 4차 계수 넷(``fixed`` 면 sam-extra 0.33.1 의 coeff4) — 테스트용 기준식(상류와 같은 연산 순서)."""
    q = h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))
    temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
    temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
        * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
    return ((2 + h_n / h_n_1) / 2 + temp1 + temp2,
            -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + h_n_1 / h_n_2 + q) * temp2,
            temp1 * h_n_1 / h_n_2 + (h_n_1 / h_n_2 + q * (1 + h_n_2 / h_n_3)) * temp2,
            -temp2 * q * h_n_2 / h_n_3 if fixed else -temp2 * q * h_n_1 / h_n_2)


class ConstantVelocity:
    """등속 탐침: ``D(x, σ) = x − σ·v`` 라 ``d = v`` 이고 정확한 한 스텝은 ``x + (σ' − σ)·v`` 다. 모델이 본 latent 로 스텝마다
    실제로 쓰인 계수의 합 ``(x_{i+1} − x_i) / ((σ_{i+1} − σ_i)·v)`` 를 잰다(정확하면 1). float64. IPNDM_V 는 모델의
    다른 속성을 읽지 않는다."""

    def __init__(self, shape=(1, 4, 3, 5), seed: int = 0):
        import torch

        generator = torch.Generator().manual_seed(seed)
        self.v = torch.randn(shape, generator=generator, dtype=torch.float64)
        self.x0 = torch.randn(shape, generator=generator, dtype=torch.float64)
        self.seen = []

    def __call__(self, x, sigma, **_extra_args):
        self.seen.append(x.clone())
        return x - sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype) * self.v

    def step_sums(self, sigmas, final) -> list:
        """스텝마다 계수의 합(마지막 스텝은 ``final`` — 샘플러가 돌려준 latent)."""
        states = [*self.seen, final]
        return [float(((states[i + 1] - states[i]) / ((sigmas[i + 1] - sigmas[i]).double() * self.v)).mean())
                for i in range(len(states) - 1)]
