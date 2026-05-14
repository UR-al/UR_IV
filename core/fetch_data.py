"""
danbooru_optimized parquet 자동 다운로드.

레포에는 .gitignore로 제외되어 있어 git clone만으로는 받을 수 없으므로
Hugging Face Datasets에서 누락된 경우에만 가져온다.
앱 시작 전 .bat에서 호출하기 위한 헬퍼.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ID = "UR-AR/UR_IV"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "danbooru_optimized"
PATTERNS = ["*.parquet"]


def _needs_download() -> bool:
    if not DATA_DIR.exists():
        return True
    return not any(DATA_DIR.rglob("*.parquet"))


def _try_install_with_uv() -> bool:
    import shutil
    if not shutil.which("uv"):
        return False
    try:
        subprocess.check_call(["uv", "pip", "install", "-U", "huggingface_hub"])
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


def ensure_data() -> int:
    if not _needs_download():
        print(f"[fetch_data] OK - {DATA_DIR.name}/ exists (skip download)", flush=True)
        return 0

    print(f"[fetch_data] data missing", flush=True)
    print(f"[fetch_data] repo : {REPO_ID} ({REPO_TYPE})", flush=True)
    print(f"[fetch_data] dest : {DATA_DIR}", flush=True)
    print(f"[fetch_data] downloading from Hugging Face (may take a while)...", flush=True)

    _ensure_hf_hub()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(DATA_DIR),
            allow_patterns=PATTERNS,
        )
    except HfHubHTTPError as e:
        print(f"[fetch_data] HTTP error: {e}", flush=True)
        print(f"[fetch_data] If repo is private, run `hf auth login` first.", flush=True)
        return 1
    except Exception as e:
        print(f"[fetch_data] download failed: {e}", flush=True)
        return 1

    n = sum(1 for _ in DATA_DIR.rglob("*.parquet"))
    print(f"[fetch_data] done - {n} parquet files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(ensure_data())
