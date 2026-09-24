# AGENTS.md - AI Studio Pro 개발 지침

## 프로젝트 개요
PyQt6 + Vue 3 SPA 하이브리드 AI 이미지 생성 애플리케이션.
QWebEngineView에서 Vue SPA를 렌더링하고, QWebChannel로 Python↔Vue 통신.

---

## ⚠️ 필수 규칙 (검증·빌드 — 여기부터 읽을 것)
Codex 는 이 파일만 자동으로 읽는다. CLAUDE.md 와 같은 규칙을 여기에도 둔다(한쪽을 고치면 둘 다).
- 모든 git / npm build 는 **메인 저장소 루트**에서. 워크트리(`.codex/worktrees/...` 등)에서 빌드·커밋하면 유실된다.
- Python 은 **반드시 venv**: `venv\Scripts\python.exe run_tests.py` (PATH 첫 `python` 은 의존성 없는 3.10 →
  가짜 ImportError 수십 개. run_tests.py 는 venv 로 스스로 재실행하지만 py_compile 등은 아님).
  편집 중 빠른 확인은 `--quick`(느린 통합 테스트·torch 표시 테스트 제외), 커밋 전엔 전체.
  테스트 모듈 최상위에서 `import torch` 금지 — `tests/_optional_deps` 의 `@requires_torch` + `load_torch()`/
  `bind_torch(globals())` 로 지연 import(`tests/test_optional_deps.py` 가 검사).
- `frontend/src/` 수정 후: `cd frontend` → `npm run test`(vitest) · `npm run test:node`(`src/studio/*.test.mjs`) ·
  `npm run type-check`(vue-tsc **0 errors 유지**) → `npm run build`, 그리고 `frontend_dist` 도 같이 커밋.
- 브리지 계약: `tests/test_bridge_contract.py` 가 액션·이벤트·슬롯·시그널 이름을 양방향으로 검사한다.
  새 액션은 Python 핸들러 + `frontend/src/types/bridge.d.ts`(ActionName, 필요하면 ActionPayloads) + 호출부를 같이.
  웹 모드 공개 목록(`web_main_ui._WEB_METHODS/_WEB_SIGNALS`)은 `tests/test_web_mode_security.py` 도 본다.
- 파일은 BOM 없는 UTF-8, `open()`/`subprocess` 에 `encoding='utf-8'`. `.bat`/`.cmd` 는 CRLF(.gitattributes).
- 런타임 데이터 커밋 금지: `config/cond_rules.json` · `config/char_global_prefs.json` ·
  `cache/session/session_backup.json`. API 키/시크릿 커밋 금지.

---

## 🏗️ 아키텍처 (v2.2.0+)

```
QMainWindow
└── QStackedWidget (_main_stack)
    ├── index 0: QWebEngineView (Vue SPA — 모든 탭)
    ├── index 1: BrowserTab (Web)
    └── index 2: BackendUITab (Backend)

Vue SPA 내부:
App.vue
├── NavRail (왼쪽 세로 탭 레일 — 순서는 Settings 에서 드래그, localStorage 저장)
├── main (flex)
│   ├── left-panel (T2I/I2I/Inpaint만)
│   │   ├── PromptPanel (블록 모드 지원)
│   │   └── Studio Tools (SAVE/PRESET/WEIGHT/WILDCARD/A/B TEST)
│   ├── 반달 화살표 → 확장 패널 오버레이
│   │   ├── Parameters (Resolution/Sampler/Scheduler/Steps/CFG/Seed)
│   │   ├── Hires.fix / ADetailer / NegPiP
│   │   ├── 조건부 프롬프트 → Search 탭에서 관리
│   │   └── LoRA Stack
│   ├── content (router-view + keep-alive)
│   └── right-panel (History 5개 페이지네이션 + EXIF 3탭)
├── QueuePanel (하단, 실시간 동기화)
├── VRAM 게이지 (최하단)
└── Toast 알림 시스템
```

---

## 📁 핵심 파일

### Python 백엔드
| 파일 | 역할 |
|------|------|
| `ui/vue_bridge.py` | QWebChannel 브릿지 — 모든 시그널/슬롯 (큰 파일: grep 후 부분 읽기) |
| `ui/generator_main.py` | 메인 윈도우 + _handle_vue_action (큰 파일: grep 후 부분 읽기) |
| `ui/*_actions.py` | 기능별 액션 믹스인(creator/chat/relight/xyz …) — `_handle_*_action` |
| `ui/generator_ui_setup.py` | UI 초기화 + 프록시 위젯 |
| `ui/generator_generation.py` | 이미지 생성 로직 |
| `ui/generator_prompts.py` | 프롬프트 처리 + 제외 필터 (9종 문법) |
| `ui/generator_settings.py` | 설정 저장/로드 |
| `ui/generator_actions.py` | 시그널 연결 + 액션 핸들러 |
| `ui/widget_proxies.py` | PyQt 위젯 인터페이스 프록시 |
| `core/tag_classifier.py` | tags_db 기반 태그 분류 |
| `core/tag_database.py` | manifest 기반 태그 자산 경로·스키마·그룹/implication 로더 |
| `core/ollama_client.py` | Ollama REST API 래퍼 |
| `core/sam_refiner.py` | YOLO+SAM 정밀 마스킹 |
| `core/edge_refiner.py` | 배경 제거 알파 매팅 |
| `core/error_handler.py` | 전역 에러 코드(`ERROR_CODES` — 사용 중: E010/E020/E030/E040/E050/E100) · 콘솔은 원문, UI 토스트는 `sanitize_for_ui`(절대 경로 가림) |

### Vue 프론트엔드
| 파일 | 역할 |
|------|------|
| `frontend/src/App.vue` | 전체 레이아웃 + 확장 패널 + 매니저 모달들 |
| `frontend/src/components/PromptPanel.vue` | T2I 프롬프트 입력 (블록 모드/텍스트 모드) |
| `frontend/src/components/TagBlockField.vue` | 범용 태그 블록 컴포넌트 |
| `frontend/src/components/NavRail.vue` | 왼쪽 세로 탭 레일 (Ctrl+Tab 이동) |
| `frontend/src/types/bridge.d.ts` | 브리지 계약 타입 — ActionName/BackendEvent/ActionPayloads |
| `frontend/src/components/CustomSelect.vue` | 커스텀 드롭다운 |
| `frontend/src/components/CompareSlider.vue` | Before/After 비교 슬라이더 |
| `frontend/src/components/QueuePanel.vue` | 대기열 (실시간 동기화) |
| `frontend/src/stores/widgetStore.js` | 위젯 상태 저장소 + useWidgetStore |
| `frontend/src/bridge.js` | QWebChannel 초기화 |

### 설정 파일
| 파일 | 역할 |
|------|------|
| `config/ui_prefs.json` | UI 설정 (블록모드, 메타데이터 패널 등) |
| `config/tab_defaults.json` | 탭별 기본값 |
| `config/cond_rules.json` | 조건부 프롬프트 규칙 |
| `config/global_weights.json` | 글로벌 태그 가중치 |
| `config/gallery_last_folder.txt` | (레거시) 옛 Gallery 마지막 폴더 — 지금은 `ui_prefs.galleryFolder`. 비어 있을 때 이 PC에 실제로 있는 폴더만 1회 흡수 |
| `tags_db/manifest.json` | 태그 데이터 파일 경로·형식·필수 컬럼 단일 소스 |

---

## 🔧 개발 규칙

### 빌드
```bash
cd frontend && npm run build  # Vue 수정 후 필수
```

### 커밋
- 한글 커밋 메시지: `feat:` / `fix:` / `refactor:` / `docs:`
- 파일 수정 완료 후 commit & push

### 코딩 규칙
- **"최소한의 연결만" 같은 타협 절대 금지** — 항상 완전한 구현
- PyQt 위젯을 직접 사용하지 않음 — WidgetProxy 시스템 사용
- Vue v-model 키는 Python proxy widget_id와 동일해야 함
  (ex: `widgets.character_input` ↔ `LineEditProxy(b, 'character_input')`)
- 탭 간 이미지 전송: `tabChanged` 먼저 → 100ms 후 이미지 시그널
- 에러 발생 시 `core/error_handler.py` 사용

### Widget ID 매핑
```
Vue: widgets.character_input  ↔  Python: LineEditProxy(b, 'character_input')
Vue: widgets.model_combo      ↔  Python: ComboBoxProxy(b, 'model_combo')
Vue: widgets.total_prompt_display ↔ Python: TextEditProxy(b, 'total_prompt_display')
```

---

## 📋 제외 프롬프트 문법 (9종)

| 제외 | 예외 |
|------|------|
| `단어` 포함 제외 | `~단어` 완전일치 유지 |
| `*단어` 완전일치 제외 | `~_단어` 접미 유지 |
| `_단어` 접미 제외 | `~단어_` 접두 유지 |
| `단어_` 접두 제외 | `~_단어_` 포함 유지 |
| `_단어_` 포함 제외 | |

---

## 🔑 주요 시그널 (Python → Vue)

```
imageGenerated, generationStarted, generationError, generationProgress
editorImageLoaded, i2iImageLoaded, inpaintImageLoaded (인페인트 전용), pngInfoImageLoaded (PNG Info 열기 전용)
widgetValueChanged, widgetPropertyChanged, batchUpdate
searchResultsReady, eventSearchResults, searchStatus
eventLoadStatus (Event 데이터 적재 문구 — Search 의 searchStatus 와 분리)
loraStackLoaded, yoloModelUpdated, compareImageLoaded
vramUpdated, ollamaResult, condRulesLoaded
queueUpdated, queueItemAdded, queueCompleted
showNotification, uiPrefsLoaded, globalWeightsLoaded
batchFilesSelected, batchJobState (일괄 처리·업스케일 진행), tabChanged
(전체 목록의 단일 출처: ui/vue_bridge.py 의 pyqtSignal / frontend/src/types/bridge.d.ts 의 BackendEvent)
```

---

## 🤖 Gemini CLI 활용
```bash
gemini chat "질문" --no-stream
gemini chat "@파일경로 분석해줘" --no-stream
```

---

## 📦 의존성
단일 출처는 `requirements.txt`(앱 시작 시 `core/app_startup` → `core/check_requirements.py` 가 누락분 자동 설치,
Python 3.10/3.11 — 검증 venv 3.11.9).
CUDA torch/torchvision 은 `core/check_requirements.py` 가 별도 인덱스에서 설치한다.
```
# 선택: MobileSAM — pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```
