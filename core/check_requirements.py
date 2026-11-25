"""
requirements.txt를 빠르게 검사하여 누락되거나 버전이 맞지 않는 패키지를 설치.

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

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:  # pip is bootstrapped by every launcher before this script.
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement  # type: ignore


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REQ_FILE = os.path.join(_PROJECT_ROOT, "requirements.txt")
_SPEC_SPLIT = re.compile(r"[<>=!~\s;]")

# CUDA torch (SAM3는 GPU 전제 — CPU torch는 미지원)
# 시스템 CUDA 버전이 다르면 cu121 / cu126 등으로 변경
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
_TORCH_CUDA_TAG = "+cu"  # 메타데이터의 local version에 포함되는 식별자
_TORCH_PACKAGES = ("torch", "torchvision")


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
    """누락되었거나 version spec을 만족하지 않는 requirement를 반환."""
    missing: list[str] = []
    for name, spec in _parse_requirements(req_path):
        if not _requirement_satisfied(name, spec):
            missing.append(spec)
    return missing


def _requirement_satisfied(name: str, spec: str) -> bool:
    try:
        requirement = Requirement(spec)
    except InvalidRequirement:
        return False
    if requirement.marker is not None and not requirement.marker.evaluate():
        return True
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        return False
    return not requirement.specifier or requirement.specifier.contains(installed, prereleases=True)


def _uv_executable() -> str | None:
    """시스템에서 uv 실행 파일 경로를 찾음. 없으면 None."""
    return shutil.which("uv")


def install_missing(missing: list[str], prefer_uv: bool = True) -> int:
    """누락 패키지를 개별 설치.
    한 패키지 실패가 다른 패키지 설치를 막지 않도록 하나씩 설치.
    하나라도 끝까지 설치되지 않으면 non-zero를 반환한다.
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
        print("[deps] uv not found - falling back to pip")
        base_cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]

    # 1단계: 일괄 설치 시도 (의존성 그래프를 한 번에 해결 — 가장 빠름)
    print(f"[deps] Bulk install attempt...")
    rc = subprocess.call(base_cmd + missing)
    if rc == 0:
        print("[deps] All packages installed successfully.")
        return 0

    # 2단계: 일괄 설치 실패 시 개별 재시도 (어떤 패키지가 문제인지 격리)
    print(f"[deps] Bulk install failed (rc={rc}) - retrying one-by-one...")
    failed: list[str] = []
    for spec in missing:
        # 일괄 설치에서 이미 충족되었을 수도 있으니 버전 조건까지 재확인
        head = _SPEC_SPLIT.split(spec, 1)[0].strip()
        if _requirement_satisfied(head, spec):
            continue
        rc1 = subprocess.call(base_cmd + [spec])
        if rc1 != 0:
            failed.append(spec)
            print(f"[deps]   FAIL: {spec}")
        else:
            print(f"[deps]   OK  : {spec}")

    if failed:
        print(f"[deps] ERROR - {len(failed)} package(s) failed to install:")
        for spec in failed:
            print(f"  - {spec}")
        print("[deps] App startup aborted. Fix the install error and run again.")
        return 1

    return 0


def _torch_is_cuda_build() -> bool:
    """torch 메타데이터에 CUDA local-version 태그(`+cuXXX`)가 있는지 확인.
    torch import 없이 빠르게 판정.
    """
    try:
        v = version("torch")
    except PackageNotFoundError:
        return False
    return _TORCH_CUDA_TAG in v


def ensure_cuda_torch(prefer_uv: bool = True) -> bool:
    """torch가 CUDA 빌드인지 보장. CPU 빌드 또는 미설치면 PyTorch CUDA index에서 재설치.
    SAM3는 GPU 전제이므로 CPU torch는 받지 않음.
    """
    if _torch_is_cuda_build():
        try:
            v = version("torch")
        except PackageNotFoundError:
            v = "?"
        print(f"[deps] torch CUDA build detected ({v})")
        return True

    try:
        v = version("torch")
        print(f"[deps] torch is CPU/unknown build ({v}) - reinstalling with CUDA from {_TORCH_CUDA_INDEX}")
    except PackageNotFoundError:
        print(f"[deps] torch not installed - installing CUDA build from {_TORCH_CUDA_INDEX}")

    uv = _uv_executable() if prefer_uv else None
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "--reinstall",
               "--index-url", _TORCH_CUDA_INDEX, *_TORCH_PACKAGES]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall",
               "--disable-pip-version-check",
               "--index-url", _TORCH_CUDA_INDEX, *_TORCH_PACKAGES]

    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[deps] WARNING - CUDA torch install failed (rc={rc}). "
              f"시스템 CUDA가 없거나 인덱스 URL을 시스템 CUDA 버전에 맞게 변경 필요.")
        return False

    # 재확인 (subprocess 후 메타데이터는 최신 상태)
    if _torch_is_cuda_build():
        try:
            v = version("torch")
        except PackageNotFoundError:
            v = "?"
        print(f"[deps] torch CUDA installed: {v}")
        return True
    print("[deps] WARNING - install 성공했으나 CUDA 태그가 없음. 인덱스 URL 확인 필요.")
    return False


def main() -> int:
    # 1) GPU torch 우선 확보 (SAM3 전제 조건)
    torch_ready = ensure_cuda_torch()

    # 2) requirements.txt의 나머지 패키지 누락 검사 + 설치
    missing = find_missing()
    install_rc = install_missing(missing) if missing else 0

    # 설치 도구의 성공 코드만 믿지 않고 실제 메타데이터를 다시 검사한다.
    remaining = find_missing()
    if remaining:
        print(f"[deps] ERROR - {len(remaining)} requirement(s) are still missing:")
        for spec in remaining:
            print(f"  - {spec}")
        return 1

    if install_rc != 0:
        print(f"[deps] pip install exit code: {install_rc}")
        return install_rc

    if not torch_ready:
        print("[deps] ERROR - a CUDA-enabled torch installation is required.")
        return 1

    print("[deps] All requirements satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
