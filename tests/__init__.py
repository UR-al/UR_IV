# tests 패키지 — python run_tests.py 로 전체 실행
import os
import tempfile

# 테스트(자동 훅 포함)가 운영 app.log 에 쓰지 않게 한다 — config 를 import 하면 utils.app_logger 가
# 루트 로거에 파일 핸들러를 붙이는데, mock 으로 만든 PermissionError 같은 가짜 ERROR 트레이스백이
# 실제 오류처럼 섞였다. tests.* 모듈보다 이 패키지가 먼저 import 되므로 여기서 정하면 충분하다.
# (이미 지정돼 있으면 존중한다 — 하위 프로세스 테스트도 이 환경변수를 물려받는다.)
os.environ.setdefault(
    "AISTUDIO_LOG_FILE",
    os.path.join(tempfile.gettempdir(), "aistudio_tests", "app.log"),
)
