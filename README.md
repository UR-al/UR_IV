# UR_IV / AI Studio Pro

PyQt6와 Vue 3를 결합한 하이브리드 AI 이미지 생성 스튜디오입니다. Python 백엔드가 이미지 생성, 프롬프트 처리, 태그 데이터, 갤러리/업스케일/마스킹 작업을 담당하고, Vue SPA가 QWebEngineView 안에서 메인 UI를 렌더링합니다. Python과 Vue는 QWebChannel 브릿지로 실시간 통신합니다.

## 주요 기능

- Text-to-Image, Image-to-Image, Inpaint 워크플로우
- WebUI / ComfyUI 백엔드 연동
- 프롬프트 블록 모드와 일반 텍스트 모드
- Danbooru 기반 태그 검색, 분류, 위키 그룹, 제외 프롬프트 규칙
- LoRA Stack, 글로벌 태그 가중치, 조건부 프롬프트
- History, Gallery, EXIF, Queue 패널
- YOLO + SAM 기반 자동 검출/정밀 마스킹
- 배경 제거, 알파 매팅, 업스케일 작업
- VRAM 상태 표시와 생성 진행률 동기화
- 내장 브라우저와 백엔드 관리 탭

## 구조

```text
QMainWindow
└── QStackedWidget
    ├── index 0: QWebEngineView (Vue SPA)
    ├── index 1: BrowserTab
    └── index 2: BackendUITab

Vue SPA
├── TabBar
├── PromptPanel / Studio Tools
├── Parameters / Hires.fix / ADetailer / LoRA Stack
├── router-view
├── History / EXIF
├── QueuePanel
└── VRAM gauge
```

## 요구 사항

- Windows 10/11 권장
- Python 3.10 이상 권장
- Node.js 20 이상 권장
- NVIDIA GPU + CUDA Torch 빌드 권장
- WebUI 또는 ComfyUI 백엔드

Python 패키지는 `requirements.txt`를 기준으로 설치됩니다. Torch CUDA 빌드는 `core/check_requirements.py`가 별도 PyTorch CUDA 인덱스를 사용해 확인/설치합니다.

## 설치

```powershell
git clone https://github.com/UR-al/UR_IV.git
cd UR_IV

python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

프론트엔드 의존성:

```powershell
cd frontend
npm install
npm run build
cd ..
```

## 실행

가장 간단한 실행 방법:

```powershell
.\new_run_main_ui.bat
```

직접 실행:

```powershell
.\venv\Scripts\activate
python core\check_requirements.py
python new_main_ui.py
```

## 개발

Vue UI를 수정한 뒤에는 반드시 빌드합니다.

```powershell
cd frontend
npm run build
```

주요 개발 파일:

- `new_main_ui.py`: 앱 진입점
- `ui/vue_bridge.py`: Python/Vue QWebChannel 브릿지
- `ui/generator_main.py`: 메인 윈도우와 Vue 액션 처리
- `ui/generator_generation.py`: 이미지 생성 로직
- `ui/generator_prompts.py`: 프롬프트 처리와 제외 필터
- `frontend/src/App.vue`: Vue SPA 레이아웃
- `frontend/src/bridge.js`: QWebChannel 초기화

## 설정 파일

- `config/ui_prefs.json`: UI 설정
- `config/tab_defaults.json`: 탭별 기본값
- `config/cond_rules.json`: 조건부 프롬프트 규칙
- `config/global_weights.json`: 글로벌 태그 가중치
- `config/default_excludes.txt`: 기본 제외 프롬프트

사용자별 설정, 캐시, 생성 이미지, 모델 파일, 대용량 데이터베이스는 `.gitignore`에 따라 저장소에 포함하지 않습니다.

## 제외 프롬프트 문법

| 문법 | 의미 |
| --- | --- |
| `단어` | 포함 제외 |
| `*단어` | 완전일치 제외 |
| `_단어` | 접미 제외 |
| `단어_` | 접두 제외 |
| `_단어_` | 포함 제외 |
| `~단어` | 완전일치 유지 |
| `~_단어` | 접미 유지 |
| `~단어_` | 접두 유지 |
| `~_단어_` | 포함 유지 |

## 참고

- 모델 파일(`*.pt`, `*.safetensors` 등)은 저장소에 포함하지 않습니다.
- 로컬 캐시와 생성 결과는 `image_cache/`, `web_cache/`, `generated_images/` 등에 생성됩니다.
- 내장 브라우저의 외부 사이트 콘솔 로그는 앱 로그를 어지럽히지 않도록 기본적으로 억제됩니다.
