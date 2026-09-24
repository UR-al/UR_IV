import os

import warnings

from PIL import Image
# 대형 업스케일 결과(8K+)는 PIL 기본 한도(~178MP)를 넘으므로 상한을 올리되,
# 무제한(None) 대신 유한 상한으로 손상/폭주 이미지의 OOM은 차단.
Image.MAX_IMAGE_PIXELS = 1_000_000_000  # 1기가픽셀 (32K×32K급)
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

from PyQt6.QtGui import QImageReader
QImageReader.setAllocationLimit(4096)  # MB 단위 — 16K RGBA 직전까지 허용 (기본 256MB는 8K RGBA도 초과)

# QtWebEngine은 QApplication 생성 전에 import 필요
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
from core.storage_paths import config_file, storage_paths, user_data_file

# --- [설정 상수] ---
# 기본 Forge/WebUI 주소. 실제 값은 설정 복원(ui/generator_settings.py)이 덮어쓴다.
WEBUI_API_URL = "http://127.0.0.1:7860"

COMFYUI_API_URL = "http://127.0.0.1:8188"
COMFYUI_WORKFLOW_PATH = ""
COMFYUI_WORKFLOW_IMG2IMG_PATH = ""

from utils.app_logger import get_logger as _get_logger
_logger = _get_logger('config')
# 고정 기본값이라 진단 가치가 낮다 — config 를 import 하는 모든 프로세스(테스트 하위 프로세스 포함)가
# INFO 로 남겨 app.log 에 같은 줄이 수백 번 쌓였다. 실제 주소는 설정 복원·연결 로그가 남긴다.
_logger.debug("기본 WebUI API 주소(설정 복원 전): %s", WEBUI_API_URL)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'generated_images')

# ── 애플리케이션 소유 저장 경계 ──
# config = 설정/경로, user_data = 프리셋·즐겨찾기 같은 사용자 작성 데이터.
# 위치 결정과 레거시 이동은 traversal-safe 중앙 모듈 한 곳에서 담당한다.

USER_DATA_DIR = str(storage_paths.user_data_dir)
os.makedirs(USER_DATA_DIR, exist_ok=True)
# prompt_presets.json 은 뺐다 — 유일한 소비자(utils.prompt_preset·PresetPreviewDialog)가 사라져
# 옮겨 둘 이유가 없다. 생성 프리셋은 presets/ 폴더(core.generation_presets)가 담당한다.
_USER_DATA_FILES = (
    'character_presets.json', 'event_gen_settings.json', 'favorite_tags.json',
    'favorites.json', 'prompt_history.json',
    'search_tab_settings.json',
)
for _n in _USER_DATA_FILES:
    try:
        user_data_file(_n, legacy_paths=_n)
    except OSError as _exc:
        _logger.warning("사용자 데이터 이전 실패 (%s): %s", _n, _exc)


def user_data_path(name: str) -> str:
    """user_data/ 내 파일 경로 반환 (사용자 상태 JSON 통합용)."""
    return str(user_data_file(name))


PROMPT_SETTINGS_FILE = str(config_file(
    'prompt_settings.json',
    legacy_paths=('user_data/prompt_settings.json', 'prompt_settings.json'),
))
CACHE_DIR = os.path.join(CURRENT_DIR, 'image_cache')
# 히스토리·갤러리·즐겨찾기 카드 썸네일 캐시(core.thumb_cache: 샤딩된 sha1(normpath@폭)).
# 옛 image_cache/thumbs 에는 은퇴한 PyQt 갤러리가 만든 레거시 썸네일(아무도 읽지 않음)이 이름만으로는
# 구분할 수 없게 섞여 있어 새 폴더를 쓴다. 옛 폴더는 앱 시작 뒤 백그라운드에서 한 번 정리한다
# (현재 형식은 새 폴더로 옮기고 나머지는 지움 — core.legacy_thumb_cache).
THUMB_DIR = os.path.join(CACHE_DIR, 'thumbs_v2')
LEGACY_THUMB_DIR = os.path.join(CACHE_DIR, 'thumbs')
os.makedirs(THUMB_DIR, exist_ok=True)
# (photodata.sqlite — 은퇴한 PyQt 갤러리 탭의 EXIF 캐시 DB — 는 더 이상 읽지도 쓰지도 않는다.
#  사용자 폴더에 남은 파일은 앱이 지우지 않는다.)
FAVORITES_FILE = user_data_path('favorites.json')
# ★★★ Danbooru 데이터셋(Search·Event·태그 사전) ★★★
# 경로의 단일 출처는 core.fetch_data.DATA_DIR(기동 때 받아 두는 폴더 그 자체)다. 설정 파일로 덮지 않는다 —
# 예전엔 숨은 설정 탭이 저장 때마다 절대경로(parquet_dir)를 prompt_settings.json 에 적고 불러올 때 그대로
# 덮어써, 설치 폴더를 옮기거나 다른 설치본의 백업을 가져오면 검색·이벤트가 옛 경로를 봤다(audit #178).
# 의존 방향은 config → fetch_data 다(fetch_data 는 무거운 config 보다 먼저 돈다 — tests/test_app_startup.py).
from core.fetch_data import DATA_DIR as _DATASET_DIR

# 검색용 Parquet
PARQUET_DIR = str(_DATASET_DIR)

# 이벤트 생성용 Parquet (parent_id 포함)
EVENT_PARQUET_DIR = os.path.join(PARQUET_DIR, 'danbooru_sorted')
