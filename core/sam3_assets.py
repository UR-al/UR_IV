"""SAM3 토크나이저 자산(BPE vocab) 찾기 — Qt·torch 없이.

`sam3` pip 패키지는 `bpe_simple_vocab_16e6.txt.gz` 를 함께 설치하지 않는다(model_builder 는
패키지 옆 ``../assets`` 를 보지만 wheel 에 없다). 이 파일은 Forge SAM3 확장
(``extensions/forge_sam3_extension/assets``)이 동봉하고, 그 밖에는 gated 저장소 facebook/sam3 에서만
받을 수 있다. 예전에는 ``C:\\sd-webui-forge-neo`` 한 곳만 보고 곧바로 다운로드로 넘어가서, Forge
classic 만 있거나 facebook/sam3 접근 승인이 없는 PC 에서는 편집기 SAM3 가 아예 뜨지 않았다.

찾는 순서: 편집기 모델 폴더 → 알려진 Forge 설치의 SAM3 확장 자산 → sam3 패키지 옆 assets →
예전에 받아 둔 캐시. 모두 없을 때만 호출자가 Hugging Face 다운로드를 시도한다.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

BPE_VOCAB_NAME = "bpe_simple_vocab_16e6.txt.gz"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDITOR_MODELS_DIR = PROJECT_ROOT / "Editor_models"
DOWNLOAD_CACHE_DIR = PROJECT_ROOT / "image_cache" / "sam3_assets"
# sam-extra 저장소를 앱의 '확장 저장소 설치'나 git clone 으로 받으면 폴더 이름이 저장소 이름(sam-extra)이 되고,
# 예전 안내대로 받으면 forge_sam3_extension 이다. 두 이름을 먼저 보고, 그 밖의 확장도 assets 에 파일이 있으면 쓴다.
SAM3_EXTENSION_DIR_NAMES = ("forge_sam3_extension", "sam-extra")


def _read_forge_runtime(config_path: Path) -> Mapping[str, object]:
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    forge = ((data or {}).get("engines") or {}).get("forge") if isinstance(data, Mapping) else None
    return forge if isinstance(forge, Mapping) else {}


def _runtime_forge_roots(config_path: Path, runtime_root: Path) -> list[Path]:
    """앱 런타임 설정(backend_runtime.json)의 Forge 설치 루트 — 연결한 기존 설치 또는 관리형 릴리스."""
    forge = _read_forge_runtime(config_path)
    roots: list[Path] = []
    existing = str(forge.get("existingRoot") or "").strip()
    if existing:
        roots.append(Path(existing))
    release = str(forge.get("release") or "").strip()
    if release:
        roots.append(runtime_root / "forge" / "releases" / release / "source")
    return roots


def _runtime_forge_extension_roots(config_path: Path, runtime_root: Path) -> list[Path]:
    """앱이 띄우는 Forge 가 실제로 읽는 확장 폴더.

    앱은 Forge 를 ``--data-dir <runtime>/forge/data`` 로 띄우고, ``data/extensions`` 를 설정한 확장 폴더
    (없으면 관리형 기본값 ``shared/extensions``)로 잇는다(core.backend_runtime._ensure_extension_mount).
    관리형 설치의 확장은 releases/…/source/extensions 가 아니라 여기에 있다.
    """
    forge = _read_forge_runtime(config_path)
    roots: list[Path] = []
    configured = str(forge.get("extensionDir") or "").strip()
    if configured:
        roots.append(Path(configured))
    roots.append(runtime_root / "forge" / "data" / "extensions")
    roots.append(runtime_root / "forge" / "shared" / "extensions")
    return roots


def forge_install_roots(environ: Mapping[str, str] | None = None) -> list[Path]:
    """SAM3 확장이 있을 수 있는 Forge 설치 루트 후보(존재 여부는 보지 않음)."""
    env = os.environ if environ is None else environ
    roots: list[Path] = []
    try:
        from core.backend_runtime import CONFIG_PATH, _default_runtime_root
        roots.extend(_runtime_forge_roots(Path(CONFIG_PATH), _default_runtime_root()))
    except Exception:   # 런타임 모듈을 못 읽어도 나머지 후보로 찾는다
        pass
    models_root = str(env.get("FORGE_MODELS_ROOT", "") or "").strip()
    if models_root:
        roots.append(Path(models_root).parent)
    try:
        from core.forge_modules import FORGE_ROOT_CANDIDATES
        roots.extend(Path(p).parent for p in FORGE_ROOT_CANDIDATES)
    except Exception:
        roots.extend((Path(r"C:\sd-webui-forge-neo"), Path(r"C:\sd-webui-forge-classic")))
    return roots


def forge_extension_roots(environ: Mapping[str, str] | None = None) -> list[Path]:
    """SAM3 확장이 있을 수 있는 Forge ``extensions`` 폴더 후보 — 앱이 쓰는 확장 폴더가 먼저."""
    roots: list[Path] = []
    try:
        from core.backend_runtime import CONFIG_PATH, _default_runtime_root
        roots.extend(_runtime_forge_extension_roots(Path(CONFIG_PATH), _default_runtime_root()))
    except Exception:
        pass
    roots.extend(Path(root) / "extensions" for root in forge_install_roots(environ))
    return roots


def _extension_asset_candidates(extensions_dir: Path) -> list[Path]:
    """한 ``extensions`` 폴더 안의 후보 — 알려진 SAM3 확장 이름 먼저, 그다음 assets 에 파일이 있는 다른 확장."""
    known = [extensions_dir / name / "assets" / BPE_VOCAB_NAME for name in SAM3_EXTENSION_DIR_NAMES]
    try:
        others = sorted(
            p for p in extensions_dir.glob(f"*/assets/{BPE_VOCAB_NAME}")
            if p.parent.parent.name not in SAM3_EXTENSION_DIR_NAMES
        )
    except OSError:
        others = []
    return known + others


def _sam3_package_assets() -> list[Path]:
    try:
        import importlib.util
        spec = importlib.util.find_spec("sam3")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.origin:
        return []
    # model_builder 가 기본으로 보는 위치(sam3/../assets)와 패키지 안 assets
    package_dir = Path(spec.origin).parent
    return [package_dir.parent / "assets" / BPE_VOCAB_NAME, package_dir / "assets" / BPE_VOCAB_NAME]


def _cached_downloads(cache_dir: Path) -> list[Path]:
    """예전에 hf_hub_download(cache_dir=…)로 받아 둔 파일(snapshots/<rev>/<name>)."""
    try:
        return sorted(cache_dir.glob(f"**/{BPE_VOCAB_NAME}"))
    except OSError:
        return []


def bpe_vocab_candidates(
    *,
    editor_models_dir: Path = EDITOR_MODELS_DIR,
    extension_roots: Iterable[Path] | None = None,
    package_assets: Iterable[Path] | None = None,
    cache_dir: Path = DOWNLOAD_CACHE_DIR,
) -> list[Path]:
    """로컬 후보 경로(우선순위 순, 중복 제거). 존재 여부는 `find_local_bpe_vocab` 가 본다.

    ``extension_roots`` 는 Forge 의 ``extensions`` 폴더들이다(기본: `forge_extension_roots`).
    """
    roots = forge_extension_roots() if extension_roots is None else list(extension_roots)
    assets = _sam3_package_assets() if package_assets is None else list(package_assets)
    ordered = [Path(editor_models_dir) / BPE_VOCAB_NAME]
    for root in roots:
        ordered += _extension_asset_candidates(Path(root))
    ordered += [Path(p) for p in assets]
    ordered += _cached_downloads(Path(cache_dir))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in ordered:
        key = os.path.normcase(os.path.abspath(str(path)))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_local_bpe_vocab(candidates: Iterable[Path] | None = None) -> str:
    """존재하는 첫 로컬 BPE vocab 경로, 없으면 ''."""
    for path in (bpe_vocab_candidates() if candidates is None else candidates):
        try:
            if Path(path).is_file():
                return str(path)
        except OSError:
            continue
    return ""
