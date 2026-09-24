"""
danbooru_optimized parquet 자동 다운로드.

레포에는 .gitignore로 제외되어 있어 git clone만으로는 받을 수 없으므로
Hugging Face Datasets에서 누락된 경우에만 가져온다.
앱 시작 전(core/app_startup.prepare_application)에 호출하는 헬퍼.

무엇을 받을지는 git 이 추적하는 dataset_manifest.json 이 정한다. manifest 의 artifact 가
없거나 size_bytes 가 다르면(= 저장소를 pull 해 manifest 가 새 릴리스를 가리키는데
parquet 은 옛것) **그 path 만** 받는다. 예전엔 parquet 이 하나라도 있으면 통째로 건너뛰어,
manifest 갱신 뒤 Search 는 검증 실패로 막히고 Event 는 옛 데이터를 조용히 썼다.
기동 때 sha256 은 계산하지 않는다(수 GB 해싱) — 경로·크기만 본다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if not __package__:
    # `python core\fetch_data.py` 로 직접 실행(데이터 수동 복구 경로)하면 sys.path[0] 이
    # core/ 라 `core` 패키지를 못 찾는다 — 프로젝트 루트를 앞에 넣는다.
    # (`python -m core.fetch_data` 나 앱 기동 경로 `from core.fetch_data import` 는 해당 없음)
    _root = str(PROJECT_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)

from core.dataset_artifacts import (  # noqa: E402 — 위 경로 보정 뒤에 import 해야 한다
    manifest_artifacts,
    manifest_revision,
    read_manifest,
    stale_artifacts,
)

REPO_ID = "UR-AR/UR_IV"
REPO_TYPE = "dataset"
DATA_DIR = PROJECT_ROOT / "danbooru_optimized"
PATTERNS = ["*.parquet"]
# manifest 밖이지만 앱이 읽는 parquet (utils/tag_data.TagData — 자동완성·분류 사전)
EXTRA_REQUIRED = ("tags_dictionary.parquet",)


def _needs_download() -> bool:
    if not DATA_DIR.exists():
        return True
    return not any(DATA_DIR.rglob("*.parquet"))


def _missing_extras() -> list[str]:
    return [name for name in EXTRA_REQUIRED if not (DATA_DIR / name).is_file()]


def _download_targets() -> tuple[list[str] | None, dict | None]:
    """받을 경로 목록과 manifest. 목록이 None 이면 manifest 가 없어 예전 규칙(전부)을 쓴다."""
    manifest = read_manifest(DATA_DIR)
    if manifest is None or not manifest_artifacts(manifest):
        return None, None
    return stale_artifacts(manifest, DATA_DIR) + _missing_extras(), manifest


def _try_install_with_uv() -> bool:
    import shutil
    if not shutil.which("uv"):
        return False
    try:
        subprocess.check_call(
            [
                "uv", "pip", "install", "--python", sys.executable,
                "-U", "huggingface_hub",
            ]
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _ensure_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass
    print("[fetch_data] huggingface_hub 미설치 -> 자동 설치 중...", flush=True)
    if _try_install_with_uv():
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"]
    )


def _download(patterns: list[str], revision: str | None = None) -> bool:
    """Hugging Face 에서 `patterns` 에 맞는 파일만 DATA_DIR 로 받는다. 성공 여부."""
    print(f"[fetch_data] repo : {REPO_ID} ({REPO_TYPE})"
          + (f" @ {revision}" if revision else ""), flush=True)
    print(f"[fetch_data] dest : {DATA_DIR}", flush=True)
    print(f"[fetch_data] downloading from Hugging Face (may take a while)...", flush=True)

    try:
        _ensure_hf_hub()
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except Exception as e:
        print(f"[fetch_data] huggingface_hub unavailable: {e}", flush=True)
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(DATA_DIR),
            allow_patterns=list(patterns),
            **kwargs,
        )
    except HfHubHTTPError as e:
        print(f"[fetch_data] HTTP error: {e}", flush=True)
        print(f"[fetch_data] If repo is private, run `hf auth login` first.", flush=True)
        return False
    except Exception as e:
        print(f"[fetch_data] download failed: {e}", flush=True)
        return False
    return True


def ensure_data() -> int:
    targets, manifest = _download_targets()

    if targets is None:
        # manifest 가 없는 구형 체크아웃 — 예전처럼 parquet 이 하나도 없을 때만 전부 받는다
        if not _needs_download():
            print(f"[fetch_data] OK - {DATA_DIR.name}/ exists (skip download)", flush=True)
            return 0
        print(f"[fetch_data] data missing", flush=True)
        if not _download(PATTERNS):
            return 1
        n = sum(1 for _ in DATA_DIR.rglob("*.parquet"))
        print(f"[fetch_data] done - {n} parquet files", flush=True)
        return 0

    if not targets:
        print(f"[fetch_data] OK - {DATA_DIR.name}/ matches {DATA_DIR.name}/dataset_manifest.json "
              f"(skip download)", flush=True)
        return 0

    # 처음 설치인지, 일부만 없거나 낡은 것인지. 첫 설치는 **parquet 이 하나도 없을 때**만이다 —
    # 현재 manifest 경로가 하나도 없어도 tags_dictionary 나 옛 릴리스 이름의 shard 가 있으면
    # 쓰던 데이터가 있는 것이다. 그걸 첫 설치로 보면 오프라인에서 받기 실패 → 1 → 기동이
    # 막혔다(예전 규칙은 그 경우 받기를 건너뛰고 앱을 띄웠다).
    fresh_install = _needs_download()
    label = manifest.get("dataset_label") if isinstance(manifest, dict) else None
    print(f"[fetch_data] {'data missing' if fresh_install else 'stale or missing files'}"
          + (f" (release {label})" if label else "") + ":", flush=True)
    for relpath in targets:
        print(f"[fetch_data]   - {relpath}", flush=True)

    ok = _download(targets, manifest_revision(manifest))
    remaining, _manifest = _download_targets()
    remaining = remaining or []
    if ok and not remaining:
        print(f"[fetch_data] done - {len(targets)} file(s) updated in {DATA_DIR}", flush=True)
        return 0

    for relpath in remaining:
        print(f"[fetch_data] WARNING still missing or size-mismatched: {DATA_DIR / relpath}",
              flush=True)
    if fresh_install and not ok:
        # parquet 이 아예 없던 첫 설치는 예전과 같이 실패로 알린다
        return 1
    # 이미 쓰던 데이터가 있으면 앱은 띄운다 — 1 을 돌려주면 new_main_ui/web_main_ui 가
    # 기동 자체를 막는다. 어긋난 Search shard 는 Search 가, Event shard 는 Event 적재가 알린다.
    print("[fetch_data] WARNING continuing with the existing local data; "
          "Search/Event will report the mismatch until the files are updated.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(ensure_data())
