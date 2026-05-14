# UR_IV / AI Studio Pro

PyQt6와 Vue 3를 결합한 하이브리드 AI 이미지 생성 스튜디오입니다.
Python 백엔드가 이미지 생성, 프롬프트 처리, 태그 데이터, 갤러리/업스케일/마스킹 작업을 담당하고, Vue SPA가 QWebEngineView 안에서 메인 UI를 렌더링합니다. 두 계층은 `QWebChannel` 브릿지로 실시간 통신합니다.

> Vanilla PyQt UI 대신 Vue SPA를 채택해 복잡한 위젯/패널 레이아웃, 드래그, 가상 스크롤, 애니메이션 등을 웹 기술로 처리하면서, 무거운 작업(이미지 생성, 마스킹, DB 검색)은 Python에 맡깁니다.

---

## 목차
- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [요구 사항](#요구-사항)
- [설치](#설치)
- [실행](#실행)
- [백엔드 연결 (WebUI / ComfyUI)](#백엔드-연결-webui--comfyui)
- [데이터셋 (Hugging Face)](#데이터셋-hugging-face)
- [디렉토리 구조](#디렉토리-구조)
- [설정 파일](#설정-파일)
- [제외 프롬프트 문법](#제외-프롬프트-문법)
- [개발](#개발)
- [문제 해결](#문제-해결)

---

## 주요 기능

### 생성 워크플로우
- **T2I / I2I / Inpaint** 탭 — WebUI/ComfyUI 어느 쪽이든 같은 UI에서 사용
- **Forge Neo 확장 기능 호환** — ADetailer, Hires.fix, NegPiP, Refiner, FreeU 등을 `alwayson_scripts` 방식으로 그대로 전달
- **Queue 패널** — 다중 생성 요청을 큐잉, 실시간 진행률 표시
- **Batch Run** — 와일드카드/시드 탐색/A·B 테스트를 묶어서 일괄 실행

### 프롬프트
- **블록 모드** ↔ **텍스트 모드** 전환 (PromptPanel)
- **제외 프롬프트 9종 문법** (포함/완전일치/접두/접미 × 제외/유지)
- **조건부 프롬프트 규칙** — 키워드 매칭으로 프롬프트 자동 추가/제거
- **글로벌 태그 가중치** — 자주 쓰는 태그의 default 가중치 저장
- **LoRA Stack** — 트리거 워드/가중치/노트를 함께 관리
- **Wildcard / Preset / Weight / A·B Test** — 좌측 Studio Tools

### 태그 & 검색
- **Danbooru 기반 태그 DB** (`tags_db/` parquet 89개) — 카테고리별 분류, 자동 분류기
- **검색 탭** — `danbooru_optimized/` parquet으로 수백만 게시물 빠른 검색
- **이벤트 생성 탭** — `danbooru_optimized/danbooru_sorted/`의 parent_id 포함 데이터로 이벤트/시리즈 자동 추적

### 후처리
- **YOLO + SAM 자동 마스킹** — 얼굴/눈/입/손 등 정밀 검출
- **Auto Censor** — Auto / Mobile / SAM3 / Off 4단계 선택
- **배경 제거** (rembg + pymatting 알파 매팅)
- **업스케일** 탭, **이미지 비교 슬라이더** (Before/After)

### 기타
- **History 패널** (5개 페이지네이션) + EXIF 3탭
- **VRAM 게이지** 실시간 표시
- **내장 브라우저 탭** (Civitai/WebUI 등 외부 페이지)
- **백엔드 관리 탭** — WebUI/ComfyUI 시작/중지/로그
- **Editor 탭** — 워터마크, 리사이즈, 포맷 변환
- **Gallery 탭** — 폴더 단위 썸네일, 메타데이터 보기

---

## 아키텍처

```text
QMainWindow
└── QStackedWidget (_main_stack)
    ├── index 0: QWebEngineView (Vue SPA — 모든 메인 탭)
    ├── index 1: BrowserTab (내장 브라우저)
    └── index 2: BackendUITab (WebUI/ComfyUI 관리)

Vue SPA (frontend/)
├── App.vue
│   ├── TabBar (알약형, localStorage 순서 저장)
│   ├── left-panel  : PromptPanel + Studio Tools (T2I/I2I/Inpaint)
│   ├── content     : router-view + keep-alive
│   ├── right-panel : History + EXIF
│   └── 확장 패널   : Parameters / Hires.fix / ADetailer / NegPiP / LoRA Stack
├── QueuePanel  (하단 — 실시간 동기화)
└── VRAM gauge  (최하단)

Python ↔ Vue
└── QWebChannel (ui/vue_bridge.py ↔ frontend/src/bridge.js)
    └── WidgetProxy (ui/widget_proxies.py)
        Vue v-model 키 == Python proxy widget_id
```

---

## 요구 사항

| 항목 | 권장 |
|------|------|
| OS | Windows 10/11 |
| Python | 3.10 ~ 3.12 |
| Node.js | 20 LTS 이상 |
| GPU | NVIDIA + CUDA 12.x (SAM3는 GPU 전제) |
| VRAM | 6 GB+ (SAM3는 12 GB+ 권장) |
| 디스크 | 30 GB+ (데이터셋 + 모델) |
| 백엔드 | Forge Neo 또는 ComfyUI |

CPU 전용 환경도 일부 기능은 동작하지만 SAM3 / YOLO / ADetailer는 사실상 사용할 수 없습니다.

---

## 설치

### 1. 클론

```powershell
git clone https://github.com/UR-al/UR_IV.git
cd UR_IV
```

### 2. Python 가상환경

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

`uv`(권장)가 설치돼 있으면 `uv venv` / `uv pip install`을 자동으로 사용합니다.

### 3. 의존성 (자동)

`new_run_main_ui.bat` 실행 시 `core/check_requirements.py`가 누락된 패키지를 감지해 자동 설치합니다. 수동 설치:

```powershell
pip install -r requirements.txt
```

PyTorch CUDA 빌드는 `core/check_requirements.py`의 `ensure_cuda_torch()`가 `https://download.pytorch.org/whl/cu128` 에서 설치/재설치합니다. 시스템 CUDA가 다르면 해당 파일의 `_TORCH_CUDA_INDEX`를 수정하세요 (예: `cu121`, `cu126`).

### 4. 프론트엔드 빌드

```powershell
cd frontend
npm install
npm run build
cd ..
```

### 5. 데이터셋 (자동)

Danbooru 메타데이터 parquet은 용량이 커서 저장소에 포함하지 않습니다. **최초 실행 시 자동으로 Hugging Face에서 다운로드**되므로 사용자 조작은 필요 없습니다. 자세한 내용은 [데이터셋](#데이터셋-hugging-face) 섹션 참조.

### 6. 모델 파일 (수동)

SAM3, YOLO 모델은 직접 받아 배치합니다.

```text
Editor_models/
├── sam3.pt              (Meta SAM3 — Hugging Face에서 받기)
├── mobile_sam.pt        (선택, MobileSAM)
└── yolov8*.pt           (Ultralytics 모델)
```

---

## 실행

가장 간단한 실행:

```powershell
.\new_run_main_ui.bat
```

이 bat은 순서대로:
1. venv 활성화
2. `core/check_requirements.py` — 의존성 + CUDA torch 보장
3. `core/fetch_data.py` — `danbooru_optimized/` 누락 시 HF에서 다운로드
4. `new_main_ui.py` — 메인 윈도우 실행

직접 실행:

```powershell
.\venv\Scripts\activate
python core\check_requirements.py
python core\fetch_data.py
python new_main_ui.py
```

---

## 백엔드 연결 (WebUI / ComfyUI)

`config.py`에서 백엔드 URL을 설정합니다.

```python
USER_INPUT_URL = "http://127.0.0.1:7860/?__theme=dark"   # Forge Neo / A1111
COMFYUI_API_URL = "http://127.0.0.1:8188"
COMFYUI_WORKFLOW_PATH = ""                                # T2I 워크플로우 JSON
COMFYUI_WORKFLOW_IMG2IMG_PATH = ""                        # I2I 워크플로우 JSON
```

Forge Neo는 확장 기능(ADetailer, Hires.fix, NegPiP 등)을 별도 img2img 호출이 아니라 메인 요청의 `alwayson_scripts`로 전달하는 방식을 따릅니다. 기존 A1111 API와 호환됩니다.

ComfyUI를 사용하려면 워크플로우 JSON을 미리 export해 위 경로에 지정하세요.

---

## 데이터셋 (Hugging Face)

`config.py`에 정의된 두 폴더는 `git clone`에 포함되지 않습니다.

| 경로 | 용도 |
|------|------|
| `danbooru_optimized/*.parquet` | 검색 탭 — 게시물/태그 메타데이터 |
| `danbooru_optimized/danbooru_sorted/*.parquet` | 이벤트 생성 탭 — parent_id 포함 시리즈 |

`new_run_main_ui.bat` 실행 시 [`UR-AR/UR_IV`](https://huggingface.co/datasets/UR-AR/UR_IV) Dataset에서 누락된 parquet을 자동으로 다운로드합니다 (`core/fetch_data.py`).

수동 다운로드:

```powershell
pip install -U huggingface_hub hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER=1
hf download UR-AR/UR_IV --repo-type=dataset --local-dir danbooru_optimized --include "*.parquet"
```

`tags_db/` 내부의 89개 parquet은 별도 다운로드 불필요 — git에 포함돼 있습니다.

---

## 디렉토리 구조

```text
UR_IV/
├── new_main_ui.py              앱 진입점
├── new_run_main_ui.bat         실행 스크립트 (자동 의존성/데이터 체크)
├── config.py                   백엔드 URL, 경로 상수
├── requirements.txt
│
├── ui/                         PyQt 메인 윈도우, 시그널/슬롯
│   ├── vue_bridge.py           QWebChannel 브릿지
│   ├── generator_main.py       메인 윈도우
│   ├── generator_generation.py 이미지 생성 로직
│   ├── generator_prompts.py    프롬프트 + 제외 필터
│   ├── generator_settings.py   설정 저장/로드
│   ├── generator_actions.py    Vue 액션 핸들러
│   ├── generator_ui_setup.py   UI 초기화 + 프록시
│   └── widget_proxies.py       PyQt 위젯 인터페이스 프록시
│
├── core/                       독립 모듈 (UI 의존성 없음)
│   ├── check_requirements.py   의존성 자동 설치
│   ├── fetch_data.py           HF 데이터셋 자동 다운로드
│   ├── error_handler.py        전역 에러 코드 (E001~E999)
│   ├── tag_classifier.py
│   ├── tag_matcher.py
│   ├── ollama_client.py        Ollama REST 래퍼
│   ├── sam_refiner.py          YOLO+SAM 정밀 마스킹
│   ├── edge_refiner.py         배경 제거 알파 매팅
│   ├── event_data_loader.py    이벤트 탭 parquet 로더
│   ├── gen_stats.py
│   ├── database.py
│   └── image_utils.py
│
├── tabs/                       탭별 모듈
│   ├── search_tab.py / event_gen_tab.py
│   ├── i2i_tab.py / inpaint_tab.py / batch_tab.py
│   ├── gallery_tab.py / pnginfo_tab.py / xyz_plot_tab.py
│   ├── upscale_tab.py / editor_tab.py / settings_tab.py
│   ├── browser_tab.py / backend_ui_tab.py
│   └── editor/                 Editor 서브 모듈
│
├── backends/                   WebUI / ComfyUI 어댑터
├── workers/                    QThread 워커
├── widgets/                    커스텀 PyQt 위젯
│
├── frontend/                   Vue 3 SPA 소스
│   ├── src/
│   │   ├── App.vue
│   │   ├── bridge.js           QWebChannel 초기화
│   │   ├── components/
│   │   └── stores/widgetStore.js
│   └── package.json
├── frontend_dist/              빌드 결과
│
├── tags_db/                    Danbooru 태그 카테고리 parquet (89개)
├── danbooru_optimized/         [자동 다운로드] 검색/이벤트 parquet
├── Editor_models/              [수동] SAM3/YOLO 모델
├── wildcards/
│
├── config/                     사용자/탭별 설정 JSON
└── assets/
```

---

## 설정 파일

| 파일 | 역할 |
|------|------|
| `config/ui_prefs.json` | UI 상태 (블록모드, 메타데이터 패널 등) |
| `config/tab_defaults.json` | 탭별 기본 파라미터 |
| `config/cond_rules.json` | 조건부 프롬프트 규칙 |
| `config/global_weights.json` | 글로벌 태그 가중치 |
| `config/default_excludes.txt` | 기본 제외 프롬프트 (12 카테고리) |
| `config/gallery_last_folder.txt` | Gallery 마지막 폴더 |

사용자별 개인 설정 파일(`prompt_settings.json`, `favorites.json`, `yolo_model_config.json` 등)과 캐시(`image_cache/`, `web_cache/`), 생성 결과(`generated_images/`)는 `.gitignore`로 제외됩니다.

---

## 제외 프롬프트 문법

| 문법 | 의미 |
|------|------|
| `단어` | 포함 제외 |
| `*단어` | 완전일치 제외 |
| `_단어` | 접미 제외 |
| `단어_` | 접두 제외 |
| `_단어_` | 포함 제외 |
| `~단어` | 완전일치 유지 |
| `~_단어` | 접미 유지 |
| `~단어_` | 접두 유지 |
| `~_단어_` | 포함 유지 |

예시 — `default_excludes.txt`:
```text
*lowres
*bad_anatomy
hand_
_quality_
~good_quality
```

---

## 개발

### Vue 수정 후 빌드 필수

```powershell
cd frontend
npm run build
```

### Widget ID 매핑

```text
Vue: widgets.character_input       <->  Python: LineEditProxy(b, 'character_input')
Vue: widgets.model_combo           <->  Python: ComboBoxProxy(b, 'model_combo')
Vue: widgets.total_prompt_display  <->  Python: TextEditProxy(b, 'total_prompt_display')
```

Vue의 `v-model` 키와 Python proxy `widget_id`는 **반드시 일치**해야 합니다.

### 주요 시그널 (Python → Vue)

```text
imageGenerated, generationStarted, generationError, generationProgress
editorImageLoaded, i2iImageLoaded, inpaintImageLoaded
widgetValueChanged, widgetPropertyChanged, batchUpdate
searchResultsReady, eventSearchResults, searchStatus
loraInserted, yoloModelUpdated, compareImageLoaded
vramUpdated, ollamaResult, condRulesLoaded
queueUpdated, queueItemAdded, queueCompleted
showNotification, uiPrefsLoaded, globalWeightsLoaded
batchFilesSelected, seedExploreResult, tabChanged
```

### 탭 간 이미지 전송

`tabChanged` 시그널 발행 → 100ms 후 이미지 시그널 발행 순서로 처리해야 Vue 측 라우팅과 위젯 초기화가 끝난 뒤 이미지가 도착합니다.

### 코딩 규칙

- "최소한의 연결만" 같은 타협 금지 — 항상 완전한 구현
- PyQt 위젯을 직접 사용하지 말고 `WidgetProxy` 사용
- 에러는 `core/error_handler.py` (E001~E999) 사용
- 한글 커밋 메시지: `feat:` / `fix:` / `refactor:` / `docs:`

---

## 문제 해결

### `[fetch_data] HTTP 오류` 또는 401/403
Hugging Face 레포가 private이거나 토큰이 없는 경우입니다. `hf auth login` 후 재실행하거나 데이터셋을 public으로 설정하세요.

### CUDA torch가 자동으로 안 깔리거나 CPU 빌드가 깔림
`core/check_requirements.py`의 `_TORCH_CUDA_INDEX`가 시스템 CUDA와 다를 수 있습니다. `cu121`, `cu126`, `cu128` 중 자신의 환경에 맞게 수정.

### `QtWebEngineProcess.exe` 충돌 / 흰 화면
`frontend_dist/`가 비어 있으면 발생합니다. `cd frontend && npm run build` 후 재실행. 그래도 안 되면 `tempdir/AIStudioPro_*` 캐시 폴더 삭제.

### Forge Neo 확장이 안 먹힘
확장 파라미터는 별도 img2img 호출이 아니라 메인 요청의 `alwayson_scripts`로 전송돼야 합니다 (Forge Neo 정책). `ui/generator_generation.py` 참고.

### SAM3 메모리 부족
Auto Censor 설정에서 **SAM3 → Mobile** 또는 **Auto**로 변경하세요.

---

## 참고

- 모델 파일(`*.pt`, `*.safetensors` 등)은 저장소에 포함하지 않습니다.
- 로컬 캐시와 생성 결과는 `image_cache/`, `web_cache/`, `generated_images/` 등에 생성됩니다.
- 내장 브라우저의 외부 사이트 콘솔 로그는 앱 로그를 어지럽히지 않도록 기본적으로 억제됩니다.

## 크레딧

- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic) — 백엔드 호환 기준
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Segment Anything 3](https://github.com/facebookresearch/sam3) (Meta)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [rembg](https://github.com/danielgatis/rembg) + [pymatting](https://github.com/pymatting/pymatting)
- Danbooru 태그 데이터셋: [UR-AR/UR_IV](https://huggingface.co/datasets/UR-AR/UR_IV)
