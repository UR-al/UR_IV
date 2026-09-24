"""앱 재시작 — 지금 프로세스가 완전히 끝난 뒤 같은 명령으로 다시 띄운다(Qt 없음).

예전 숨은 SettingsTab 의 재시작은 새 프로세스를 먼저 띄우고 곧바로 os._exit 했다. 새 앱이
고정 경로 web_profile(QWebEngine 영속 저장소)을 옛 앱과 동시에 잡을 수 있었다. 이제 작은 대기
프로세스(``python -m core.app_restart --wait-pid <pid> --cwd <dir> -- <명령…>``)를 먼저 띄우고,
그 프로세스가 지금 PID 의 종료를 확인한 뒤 앱을 실행한다.

호출 순서(ui/settings_data_actions.restart_app): 대기 프로세스 → ``_quit_app``(설정 저장 —
가져온 백업이 있으면 건너뜀 — 과 워커 정리 뒤 os._exit).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

WAIT_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 0.25

_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_CONSOLE = 0x00000010


class RestartError(RuntimeError):
    """재시작을 준비하지 못했다(앱은 계속 돈다)."""


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def relaunch_command(executable: str | None = None, argv: Sequence[str] | None = None) -> list[str]:
    """지금 앱을 띄운 명령 — [python, 스크립트 절대경로, 인자…]."""
    exe = executable or sys.executable
    args = list(sys.argv if argv is None else argv)
    if not exe:
        raise RestartError('파이썬 실행 파일을 찾지 못했습니다')
    if not args or not str(args[0]).strip():
        raise RestartError('앱 시작 스크립트를 알 수 없습니다')
    script = os.path.abspath(str(args[0]))
    if not os.path.isfile(script):
        raise RestartError(f'앱 시작 스크립트가 없습니다: {script}')
    return [exe, script, *[str(a) for a in args[1:]]]


def waiter_command(pid: int, command: Sequence[str], cwd: str,
                   executable: str | None = None) -> list[str]:
    if not command:
        raise RestartError('다시 실행할 명령이 비어 있습니다')
    return [executable or sys.executable, '-m', 'core.app_restart',
            '--wait-pid', str(int(pid)), '--cwd', str(cwd), '--', *[str(c) for c in command]]


def _detached_kwargs(*, show_console: bool) -> dict:
    kwargs: dict = {'close_fds': True, 'stdin': subprocess.DEVNULL}
    if os.name == 'nt':
        flags = _CREATE_NEW_PROCESS_GROUP | (_CREATE_NEW_CONSOLE if show_console else _CREATE_NO_WINDOW)
        kwargs['creationflags'] = flags
    else:
        kwargs['start_new_session'] = True
    if not show_console:
        kwargs['stdout'] = subprocess.DEVNULL
        kwargs['stderr'] = subprocess.DEVNULL
    return kwargs


def spawn_restart_waiter(pid: int | None = None, command: Sequence[str] | None = None, *,
                         cwd: str | None = None,
                         popen: Callable[..., object] = subprocess.Popen) -> object:
    """대기 프로세스를 띄운다. 실패하면 RestartError — 호출자는 앱을 끄지 않아야 한다."""
    target_cwd = cwd or os.getcwd()
    cmd = waiter_command(pid or os.getpid(), command or relaunch_command(), target_cwd)
    try:
        return popen(cmd, cwd=_project_root(), **_detached_kwargs(show_console=False))
    except OSError as exc:
        raise RestartError(f'재시작 도우미를 실행하지 못했습니다: {exc}') from exc


def _default_exists(pid: int) -> bool:
    from core.app_instance import process_exists

    return process_exists(pid)


def _default_launch(command: Sequence[str], cwd: str) -> None:
    subprocess.Popen(list(command), cwd=cwd or None, **_detached_kwargs(show_console=True))


def wait_then_launch(pid: int, command: Sequence[str], cwd: str, *,
                     exists: Callable[[int], bool] = _default_exists,
                     launch: Callable[[Sequence[str], str], None] = _default_launch,
                     sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic,
                     timeout: float = WAIT_TIMEOUT_SECONDS) -> bool:
    """pid 가 끝나면 command 를 실행한다. 시간 안에 안 끝나면 실행하지 않고 False."""
    deadline = clock() + timeout
    while exists(pid):
        if clock() >= deadline:
            return False
        sleep(POLL_SECONDS)
    launch(command, cwd)
    return True


def parse_args(argv: Sequence[str]) -> tuple[int, str, list[str]]:
    args = list(argv)
    try:
        split = args.index('--')
    except ValueError as exc:
        raise RestartError('명령 구분자(--)가 없습니다') from exc
    head, command = args[:split], args[split + 1:]
    options = dict(zip(head[0::2], head[1::2]))
    try:
        pid = int(options['--wait-pid'])
    except (KeyError, ValueError) as exc:
        raise RestartError('--wait-pid 가 필요합니다') from exc
    if not command:
        raise RestartError('다시 실행할 명령이 비어 있습니다')
    return pid, options.get('--cwd', ''), command


def main(argv: Sequence[str] | None = None) -> int:
    try:
        pid, cwd, command = parse_args(sys.argv[1:] if argv is None else argv)
    except RestartError:
        return 2
    return 0 if wait_then_launch(pid, command, cwd) else 1


if __name__ == '__main__':
    raise SystemExit(main())
