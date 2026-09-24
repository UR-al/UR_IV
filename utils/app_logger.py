# utils/app_logger.py
"""
앱 전역 로깅 시스템

- 파일(app.log, 10MB × 5 로테이션)은 DEBUG, 콘솔은 INFO. 앱 모듈 25여 개가 자체 DEBUG 를 쓰므로
  루트·핸들러 레벨은 올리지 않는다.
- 서드파티 소음만 줄인다: PIL 은 PNG 를 열 때마다 STREAM DEBUG 3줄, urllib3 는 연결마다 DEBUG 를
  남겨 app.log 의 73~97% 를 차지했다(진단할 줄을 찾기 어렵고 PNG 헤더 열기도 수십 배 느려졌다).
- 파일 줄에는 날짜를 넣는다 — 로테이션된 수개월치 백업에서 날짜를 구분할 수 있게.
- ``AISTUDIO_LOG_FILE`` 환경변수로 로그 파일 경로를 바꿀 수 있다. tests/__init__.py 가 임시 경로를
  지정해, 테스트(자동 훅 포함)가 mock 으로 만든 가짜 ERROR 를 운영 app.log 에 섞지 않게 한다.
"""
import os
import logging
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_LOG_DIR, 'app.log')
LOG_FILE_ENV = 'AISTUDIO_LOG_FILE'
_INITIALIZED = False

_LOG_FORMAT = '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
FILE_DATEFMT = '%Y-%m-%d %H:%M:%S'
CONSOLE_DATEFMT = '%H:%M:%S'

# 서드파티 로거 레벨 — 루트(DEBUG)를 그대로 두고 이 로거들만 올린다.
THIRD_PARTY_LOG_LEVELS = {
    'PIL': logging.INFO,        # PngImagePlugin/TiffImagePlugin/Image 의 청크 단위 DEBUG
    'urllib3': logging.WARNING,  # connectionpool 의 요청마다 DEBUG ("Starting new HTTP connection")
}


def resolve_log_file(environ=None) -> str:
    """로그 파일 경로 — ``AISTUDIO_LOG_FILE`` 이 있으면 그 경로(절대 경로로), 없으면 저장소의 app.log."""
    env = os.environ if environ is None else environ
    override = str(env.get(LOG_FILE_ENV) or '').strip()
    return os.path.abspath(override) if override else _LOG_FILE


def quiet_third_party_loggers() -> None:
    """PIL·urllib3 처럼 DEBUG 로 로그를 뒤덮는 서드파티 로거의 레벨만 올린다."""
    for name, level in THIRD_PARTY_LOG_LEVELS.items():
        logging.getLogger(name).setLevel(level)


def build_file_handler(log_file: str):
    """app.log 파일 핸들러(10MB 로테이션, 최대 5개 백업). 파일을 열지 못하면 안내 한 줄을 찍고 None.

    파일을 **곧바로** 연다(``delay`` 없음). 지연 열기면 생성자는 성공하고 첫 기록 때 열기가 실패해,
    폴백(콘솔만) 대신 기록마다 '--- Logging error --- … PermissionError' 트레이스백이 stderr 를 덮었다.
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5,
                                 encoding='utf-8', delay=False)
    except OSError as exc:
        print(f"[Log] 로그 파일을 열지 못해 콘솔에만 기록합니다 ({log_file}): {exc}")
        return None
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=FILE_DATEFMT))
    return fh


def configure_root(root: logging.Logger, log_file: str) -> None:
    """``root`` 에 레벨·서드파티 소음 억제·파일(열리면)·콘솔 핸들러를 붙인다."""
    root.setLevel(logging.DEBUG)
    quiet_third_party_loggers()

    # 파일 핸들러. 열지 못해도(권한·잠금·디렉터리 경로) 콘솔 로깅으로 계속 뜬다.
    fh = build_file_handler(log_file)
    if fh is not None:
        root.addHandler(fh)

    # 콘솔 핸들러 — 콘솔은 짧은 시각만
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=CONSOLE_DATEFMT))
    root.addHandler(ch)


def _init_root():
    """루트 로거 초기화 (한 번만 실행)"""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    configure_root(logging.getLogger(), resolve_log_file())


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 반환"""
    _init_root()
    return logging.getLogger(name)
