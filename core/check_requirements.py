"""
requirements.txt를 빠르게 검사하여 미설치 패키지만 설치.

검사: importlib.metadata (pip resolve보다 훨씬 빠름)
설치: uv가 있으면 `uv pip install`, 없으면 `python -m pip install` 폴백.
앱 시작 전 .bat에서 호출하기 위한 헬퍼.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # 3.7 이하 호환 (실제로는 사용 안 함)
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REQ_FILE = os.path.join(_PROJECT_ROOT, "requirements.txt")
_SPEC_SPLIT = re.compile(r"[<>=!~\s;]")


def _parse_requirements(path: str) -> list[tuple[str, str]]:
    """requirements.txt → [(package_name, full_spec), ...]
    주석/빈줄/`-r` 같은 옵션 제외. URL/Git 설치는 그대로 spec으로 보존.
    """
    items: list[tuple[str, str]] = []
    if not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("-r", "-c", "--", "-e ")):
                # editable / nested requirements는 처리 범위 밖
                continue
            if line.startswith(("http://", "https://", "git+")):
                # URL 설치는 패키지명을 추출할 수 없으므로 건너뜀 (수동 설치 대상)
                continue

            head = _SPEC_SPLIT.split(line, 1)[0].strip()
            if not head:
                continue
            items.append((head, line))
    return items


def find_missing(req_path: str = _REQ_FILE) -> list[str]:
    """누락된 패키지의 spec 문자열 목록 반환 (e.g. ['Pillow>=9.0.0', ...])"""
    missing: list[str] = []
    for name, spec in _parse_requirements(req_path):
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(spec)
    return missing


def _uv_executable() -> str | None:
    """시스템에서 uv 실행 파일 경로를 찾음. 없으면 None."""
    return shutil.which("uv")


def install_missing(missing: list[str], prefer_uv: bool = True) -> int:
    """누락 패키지를 개별 설치.
    한 패키지 실패가 다른 패키지 설치를 막지 않도록 하나씩 설치.
    실패 패키지가 있어도 0 반환 (앱은 lazy import + bbox 폴백으로 동작).
    """
    if not missing:
        return 0
    print(f"[deps] Installing {len(missing)} missing package(s):")
    for spec in missing:
        print(f"  - {spec}")

    uv = _uv_executable() if prefer_uv else None
    if uv:
        # 활성화된 venv를 명시적으로 지정 (sys.executable이 venv의 python을 가리킴)
        print(f"[deps] Using uv ({uv})")
        base_cmd = [uv, "pip", "install", "--python", sys.executable]
    else:
        print("[deps] uv not found — falling back to pip")
        base_cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]

    # 1단계: 일괄 설치 시도 (의존성 그래프를 한 번에 해결 — 가장 빠름)
    print(f"[deps] Bulk install attempt...")
    rc = subprocess.call(base_cmd + missing)
    if rc == 0:
        print("[deps] All packages installed successfully.")
        return 0

    # 2단계: 일괄 설치 실패 시 개별 재시도 (어떤 패키지가 문제인지 격리)
    print(f"[deps] Bulk install failed (rc={rc}) — retrying one-by-one...")
    failed: list[str] = []
    for spec in missing:
        # 이미 설치되었을 수도 있으니 재확인
        head = _SPEC_SPLIT.split(spec, 1)[0].strip()
        try:
            version(head)
            continue  # 1단계에서 일부 성공했을 수 있음
        except PackageNotFoundError:
            pass
        rc1 = subprocess.call(base_cmd + [spec])
        if rc1 != 0:
            failed.append(spec)
            print(f"[deps]   FAIL: {spec}")
        else:
            print(f"[deps]   OK  : {spec}")

    if failed:
        print(f"[deps] WARNING — {len(failed)} package(s) failed to install:")
        for spec in failed:
            print(f"  - {spec}")
        print("[deps] App will continue (lazy imports / fallbacks may apply).")

    return 0  # 앱 실행은 계속 — import 실패 시 각 모듈이 알아서 폴백


def main() -> int:
    missing = find_missing()
    if not missing:
        print("[deps] All requirements satisfied.")
        return 0
    rc = install_missing(missing)
    if rc != 0:
        print(f"[deps] pip install exit code: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
