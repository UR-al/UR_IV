#!/usr/bin/env python
"""Claude Code PostToolUse 훅 — Python 파일을 Edit/Write 했을 때만 회귀 테스트 실행.

Claude Code 가 stdin 으로 tool 정보(JSON)를 준다. .py 수정일 때만 테스트를 돌리고,
실패하면 stderr + exit 2 로 Claude 에게 피드백한다(편집 자체를 되돌리진 않음).
.claude/settings.json 의 PostToolUse 훅에서 호출된다.

두 가지를 지킨다:
  1) **venv 인터프리터로 실행.** sys.executable(시스템 python)로 돌리면 pandas/PIL/
     PyQt6/requests 가 없어 ModuleNotFoundError 가 대량으로 터지고, 그 가짜 실패가 매 편집마다
     Claude 에게 피드백돼 턴을 낭비한다. (경로 탐색은 run_tests.venv_python 과 공유)
  2) **기본은 --quick.** run_tests.SLOW_MODULES(느린 통합 테스트)는 빼고, 편집한 파일이
     그 모듈이 직접 검증하는 소스(또는 그 테스트 파일 자체)일 때만 **해당 모듈만** 더해 돌린다
     (`--quick --include <모듈>`). torch 표시 테스트(tests/_optional_deps.requires_torch)도
     quick 에선 빠지고, run_tests.TORCH_TEST_SOURCES 나 표시가 든 테스트 파일을 고쳤을 때만
     `--with-torch` 로 더한다(표시는 run_tests.uses_torch_mark — 규칙 테스트와 같은 판단).
     그래도 quick 이 torch 를 올리면 run_tests 가 1 로 끝나고 stderr 끝에 처음 import 한 위치를
     적으므로, 아래 실패 피드백으로 그대로 전달된다. 소요 시간 실측치는 run_tests.SLOW_MODULES 주석이 단일 출처.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tools/ -> repo root

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _utf8_stderr() -> None:
    """Windows 기본 stderr 는 cp949(+backslashreplace) — 한글·기호가 깨져 Claude 가
    피드백을 못 읽는다. 훅의 존재 이유가 사라지므로 UTF-8 로 고정한다.
    (import 부작용이 되지 않게 main() 에서만 호출 — 테스트가 이 모듈을 import 한다)"""
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # 파이프가 아닌 드문 경우
        pass


def _interpreter() -> str:
    """저장소 venv 를 우선 사용한다(run_tests.venv_python 재사용). 없으면 현재 인터프리터."""
    try:
        from run_tests import venv_python
        path = venv_python(ROOT)
    except Exception:  # run_tests 를 못 읽으면 직접 탐색
        path = next((p for p in (
            os.path.join(ROOT, "venv", "Scripts", "python.exe"),  # Windows
            os.path.join(ROOT, "venv", "bin", "python"),          # POSIX
        ) if os.path.isfile(p)), None)
    return path or sys.executable


def _repo_relpath(file_path: str):
    """편집된 파일을 저장소 기준 상대경로(슬래시)로. 저장소 밖이면 None."""
    try:
        rel = os.path.relpath(os.path.abspath(file_path), ROOT)
    except ValueError:  # 다른 드라이브
        return None
    if rel.startswith(".."):
        return None
    return rel.replace(os.sep, "/")


def runner_args(rel: str):
    """편집 파일에 맞는 run_tests.py 인자와 사람이 읽을 범위 라벨.

    - 느린 모듈과 무관: ["--quick"], "quick"
    - 느린 모듈의 직접 소스/테스트 파일: ["--quick", "--include", m, ...], "quick+m"
    - torch 테스트의 직접 소스/표시된 테스트 파일: [..., "--with-torch"], "...+torch"
    - run_tests 를 못 읽어 판단 불가: [] (전체), "전체" — 안전 쪽으로 폴백
    """
    try:
        from run_tests import slow_modules_for, torch_tests_for
    except Exception:
        return [], "전체"
    extra = sorted(slow_modules_for(rel))
    args = ["--quick"]
    for module in extra:
        args += ["--include", module]
    label = "quick" + "".join(f"+{m.rsplit('.', 1)[-1]}" for m in extra)
    if torch_tests_for(rel, ROOT):
        args.append("--with-torch")
        label += "+torch"
    return args, label


def main() -> int:
    _utf8_stderr()
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    fp = ((data.get("tool_input") or {}).get("file_path") or "")
    if not str(fp).endswith(".py"):
        return 0  # 파이썬 파일이 아니면 스킵 (Vue/MD/JSON 등)

    rel = _repo_relpath(str(fp))
    if rel is None or rel.startswith(("venv/", ".claude/")):
        return 0  # 저장소 밖이거나 가상환경/설정 파일이면 스킵

    runner = os.path.join(ROOT, "run_tests.py")
    if not os.path.exists(runner):
        return 0

    extra_args, scope = runner_args(rel)
    argv = [_interpreter(), runner] + extra_args
    env = dict(os.environ, PYTHONIOENCODING="utf-8")  # 자식도 UTF-8 로 출력
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           cwd=ROOT, env=env, timeout=180)
    except subprocess.TimeoutExpired:
        sys.stderr.write("⚠ run_tests.py 180초 초과 — 테스트가 멈춤(행) 상태일 수 있음.\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"⚠ run_tests.py 실행 실패: {exc}\n")
        return 2

    if r.returncode == 0:
        return 0

    sys.stderr.write(
        f"⚠ run_tests.py({scope}) 실패 — 방금 수정한 {rel} 이 회귀를 일으켰을 수 있음. 확인 필요:\n")
    # unittest 는 stderr 로 쓴다 — 실패 트레이스백이 있는 쪽에 예산을 더 준다.
    sys.stderr.write((r.stderr or "")[-3000:])
    tail = (r.stdout or "").strip()
    if tail:
        sys.stderr.write("\n[stdout]\n" + tail[-800:])
    return 2  # PostToolUse: exit 2 = stderr 를 Claude 에게 피드백


if __name__ == "__main__":
    sys.exit(main())
