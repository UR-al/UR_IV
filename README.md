# UR_IV / AI Studio Pro

PyQt6 + Vue 3 SPA 하이브리드 AI 이미지 생성 스튜디오. **무거운 작업은 Python**(생성·검색·마스킹), **UI는 Vue**(애니메이션·드래그·가상화)가 담당하고 `QWebChannel`로 실시간 통신.

> Stable Diffusion WebUI(A1111/Forge) 또는 ComfyUI 백엔드에 연결해서 사용. Danbooru 태그 데이터셋 검색, 자동 검열(YOLO+SAM3), 인페인트, 업스케일, ADetailer 배치까지 한 앱에서.

---

## ✨ 핵심 기능

- **하이브리드 아키텍처**: PyQt6 윈도우 + QWebEngineView 안에서 Vue 3 SPA 렌더
- **다중 백엔드**: WebUI(A1111/Forge), ComfyUI 동시 등록 및 즉시 전환
- **Danbooru 데이터셋 검색**: manifest로 검증하는 `2026_07` 단일 활성 릴리스(G/S/Q/E 4개 shard), AND/OR 토글, 페이지네이션
- **AI 자동 검열**: YOLO 검출 + SAM3 텍스트 프롬프트 정밀 마스킹 + exclude 프롬프트
- **워크플로우 프로파일**: 모델/VAE/TE/해상도 등 한 묶음으로 저장·복원
- **프롬프트 Undo/Redo**: Ctrl+Z/Y, 워크플로우 토큰 카운터, 자동완성
- **생성 큐**: 일시정지·우선순위·다중 삭제·ETA
- **Generation API Gateway**: 인증된 외부 요청 수신 + 승인된 Forge/WebUI·ComfyUI target 중계 + A1111 생성 API 호환 subset
- **단축키**: Ctrl+G 생성, Ctrl+Tab 탭 이동, Ctrl+F 설정 검색, ESC 패널 닫기

---

## 🗂️ 탭 가이드

| 탭 | 한 줄 설명 |
|---|---|
| **T2I** | 텍스트→이미지. 좌측 프롬프트 입력(블록/텍스트 모드), 반달 화살표로 확장 패널(해상도·샘플러·LoRA·Hires.fix·ADetailer) |
| **I2I** | 이미지→이미지. 이미지 업로드/드래그, 리사이즈 모드, denoising, seed |
| **Inpaint** | 마스크 그리기로 부분 재생성. 브러시/박스/올가미, 자석 올가미, 마스크 Undo |
| **Editor** | 비-생성 편집. 브러시/스탬프/지우개, **YOLO+SAM3 자동 검열**(exclude 프롬프트), 배경 제거(rembg+알파매팅), 크롭/리사이즈, 색감, 워터마크, 5분 자동저장 |
| **Search** | Danbooru `2026_07` 단일 활성 데이터셋 검색. **G/S/Q/E 등급 필터**, **AND/OR 결합 모드**, `[A\|B]` OR 그룹, `*word` 정확매칭, 결과 정렬·페이지네이션·50만 cap |
| **Event Gen** | 이벤트·시리즈 단위 검색 → 멀티스텝 프롬프트 생성 |
| **XYZ Plot** | 파라미터 그리드 비교 (steps×CFG×sampler 등). 결과 CSV 내보내기 |
| **Batch** | 일괄 처리. 좌측 설정 / 우측 썸네일 그리드. BATCH/UPSCALE/ADETAILER/SAM3 서브탭 (ADetailer 배치는 ETA 표시) |
| **Gallery** | 폴더 기반 갤러리. 무한스크롤, EXIF 검색, 메타 사이드바, 컨텍스트 메뉴(T2I/I2I/Inpaint/Editor 전송) |
| **Favorites** | 즐겨찾기 그리드. EXIF 배치 검색, 뷰어에서 항목별 복사 + 전송 카드 그리드 |
| **PNG Info** | EXIF/PNG chunks 메타데이터 표시. PROMPT/NEGATIVE/PARAMS 항목별 복사 버튼. Compare 서브탭에서 두 이미지 슬라이더 비교 + 파라미터 diff + GIF 내보내기 |
| **Settings** | 백엔드 runtime·모델 경로, Generation API/원격 target, 단축키, 탭 기본값, 와일드카드, Ollama. **Ctrl+F로 검색** |
| **Web** | 내장 웹브라우저 (참고용) |
| **Backend** | 백엔드 자체 UI 임베드 (디버그) |

탭은 왼쪽 세로 **탭 레일**(NavRail)에 있고 **Ctrl+Tab / Ctrl+Shift+Tab** 으로 이동한다. 탭 순서는 Settings 에서
드래그로 바꾼다. **VRAM 게이지**(하단 고정) 클릭으로 모델 unload 요청.

---

## 🏗️ 아키텍처

```
QMainWindow
└── QStackedWidget (_main_stack)
    ├── 0: QWebEngineView (Vue SPA — 대부분 탭)
    ├── 1: BrowserTab (Web)
    └── 2: BackendUITab (Backend)

Vue SPA
├── App.vue  (좌측 패널 + 확장 오버레이 + 모달 매니저들)
├── PromptPanel.vue (블록 모드 / 텍스트 모드 / Undo·Redo)
├── router.js  (T2I = components/ImageViewer, 나머지 탭 = views/*View.vue 지연 로드)
├── views/{I2I, Inpaint, Editor, Search, EventGen, XYZPlot, Batch, Gallery, Favorites, PngInfo, Settings, CreatorStudio, Chat}
├── components/{NavRail, QueuePanel, ImageViewer, CompareSlider, CustomSelect, TagBlockField, ...}
└── stores/widgetStore.js  (state + requestAction IPC)
(Web·Backend 탭은 Vue 뷰가 아니라 위 QStackedWidget 의 네이티브 위젯이다.)

Python ↔ Vue
└── ui/vue_bridge.py (QWebChannel) — signals: imageGenerated, queueUpdated, searchResultsReady, vramUpdated, showNotification...
                                       slots:   searchDanbooru, editorProcess, getImageExif...
```

핵심 파일은 [`CLAUDE.md`](CLAUDE.md) 참고.

---

## 📦 요구 사항

- Python **3.10–3.11** (런처 `new_run_main_ui.bat` 가 둘 다 허용, 검증 venv 3.11.9)
- Node.js 20+ (프론트엔드 빌드용)
- Windows 10/11 권장 (PyQt6 + QWebEngineView)
- 백엔드: Stable Diffusion WebUI / Forge 또는 ComfyUI (별도 실행)

### Python 패키지
목록의 단일 출처는 [`requirements.txt`](requirements.txt) 다(앱 시작 시 `core/check_requirements.py` 가 누락분을 자동 설치).
CUDA torch/torchvision 은 별도 인덱스라 `core/check_requirements.py` 가 설치한다.
SAM3(`sam3` + Windows 용 `triton-windows`)와 VRAM 표시용 `nvidia-ml-py` 도 여기에 포함된다.

### 선택 (자동 검열)
```bash
# MobileSAM (bbox 기반, 가볍고 빠름) — requirements.txt 밖의 선택 설치
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# SAM3 를 수동으로 다시 깔 때 (timm/einops/iopath/huggingface_hub 는 sam3 가 끌어온다.
# triton 은 sam3 가 선언하지 않았지만 sam3/model/edt.py 가 무조건 import 한다)
pip install sam3 triton-windows
```

YOLO 모델 (`.pt`)과 SAM/SAM3 체크포인트는 `editor_models/` 디렉토리에 넣으면 자동 감지.

---

## 🚀 설치 / 실행

```bash
# 1) 의존성 설치
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..

# 2) 백엔드 실행 (별도 터미널)
#    Forge:  webui.bat --api          (포트 7860)
#    ComfyUI: python main.py          (포트 8188)

# 3) 앱 실행 (venv 활성화 + 의존성 체크 + 데이터 준비 포함)
new_run_main_ui.bat
# 또는 수동 실행 (venv 활성화 후 — `venv\Scripts\activate`):
python core\check_requirements.py && python -m core.fetch_data && python new_main_ui.py
```

설정 → RUNTIMES / ENGINES에서 앱 관리형 또는 기존 설치 백엔드를 연결할 수 있습니다.
외부 프로그램에서 작업을 보내거나 다른 Forge/ComfyUI로 중계하려면 설정 → NETWORK의
Generation API Gateway를 사용하세요. 전체 호출 규격과 PowerShell 예시는
[`docs/generation_api.md`](docs/generation_api.md)에 있습니다.

### Vue 수정 시
```bash
cd frontend && npm run build
```
> Vue 코드는 빌드해야 반영됨. Python 코드는 앱 재시작 필요.

---

## 📊 데이터셋 (Search 탭)

Search는 `dataset_manifest.json`이 지정하고 검증하는 `2026_07` 릴리스만 사용합니다.
릴리스 선택이나 구형 파일 fallback 없이, G/S/Q/E 등급별 4개 shard를 불러옵니다.

```
danbooru_optimized/
├── dataset_manifest.json
├── danbooru_2026_07_g.parquet
├── danbooru_2026_07_s.parquet
├── danbooru_2026_07_q.parquet
├── danbooru_2026_07_e.parquet
└── legacy_search_tags_before_2026_07.csv
```

`legacy_search_tags_before_2026_07.csv`는 삭제 전 구형 3세대(`2025`, `2026`,
`2026_06`)의 합집합에는 있었지만 `2026_07`에는 없는 고유 태그만 보존합니다.
열은 `tag`, `legacy_categories`, `legacy_releases`입니다. 데이터 생성·검증 및
구형 태그 archive 재생성 방법은 [`danbooru_optimized/README.md`](danbooru_optimized/README.md)를 참고하세요.

### 검색 문법

| 문법 | 의미 | 예 |
|---|---|---|
| `word` | 포함 매칭 | `girl` → `1girl`, `multiple girls` 모두 매칭 |
| `*word` | 완전 일치 | `*1girl` → 정확히 `1girl` 태그만 |
| `_word` | 접미 | `_hair` → `short hair`, `long hair` |
| `word_` | 접두 | `hair_` → `hair ornament`, `hair ribbon` |
| `_word_` | 명시적 포함 | `word`와 동일 |
| `[A\|B\|C]` | OR 그룹 (모드 무관) | `[blue_hair\|red_hair]` |
| `[A\|B\|]` | **필드 와일드카드** (마지막 `\|` 다음 빈칸) | `copyright: [tots\|alc\|]` → tots 또는 alc 또는 **무관**. 중간 `\|\|`는 무시(오타 보호) |
| `[A,B,C]` | AND 그룹 (모드 무관) | `[1girl,blue_hair]` |
| `,` | AND 또는 OR (모드 따라) | AND 모드: `a,b`=`a∩b` / OR 모드: `a,b`=`a∪b` |

**AND/OR 모드 토글**: 필드 간 결합 + 같은 필드 내 콤마 토큰 모두 적용.
- AND: `character:ike + copyright:arknights` → Ike이면서 Arknights 작품 (좁은 교집합)
- OR: 같은 입력 → Ike 또는 Arknights 캐릭터 누구든 (합집합)

**제외(Exclude)** 입력란도 같은 문법 사용. 결과에서 빼냄.

---

## 🎨 제외 프롬프트 문법 (T2I/I2I/Inpaint 프롬프트 처리)

| 제외 | 예외 (유지) |
|---|---|
| `단어` 포함 제외 | `~단어` 완전일치 유지 |
| `*단어` 완전일치 제외 | `~_단어` 접미 유지 |
| `_단어` 접미 제외 | `~단어_` 접두 유지 |
| `단어_` 접두 제외 | `~_단어_` 포함 유지 |
| `_단어_` 포함 제외 |  |

총 9종. `config/default_excludes.txt`는 카테고리별로 정리한 **참고용 목록**이다 — 앱이 자동으로
읽거나 적용하지 않으니, 필요한 줄을 제외 칸에 직접 붙여 넣어 쓴다.

---

## ⌨️ 단축키

| 키 | 동작 |
|---|---|
| `Ctrl+G` | 이미지 생성 |
| `Ctrl+S` | 설정 저장 |
| `Ctrl+Tab` / `Ctrl+Shift+Tab` | 다음/이전 탭 |
| `Ctrl+F` | 설정 검색 (Settings 탭 활성 시) |
| `Esc` | 모달/확장 패널 닫기 |
| `F5` | 히스토리 새로고침 |

탭별 단축키는 Settings → HOTKEYS 참고. `Ctrl+Z/Y`는 현재 활성 탭의 컨텍스트에서 동작 (Editor=편집 Undo, T2I/I2I/Inpaint=프롬프트 Undo).

---

## 📁 디렉토리 구조

```
.
├── new_main_ui.py            # 진입점 (new_run_main_ui.bat로 실행)
├── new_run_main_ui.bat       # 실행 스크립트 (venv + 의존성 체크 + 데이터 fetch)
├── ui/                       # PyQt 메인 윈도우 + Vue Bridge
├── backends/                 # WebUI / ComfyUI / Forge 클라이언트
├── core/                     # 공용 로직 (큐, 파이프라인, SAM/edge refiner...)
├── workers/                  # QThread 워커 (생성/검색/갤러리/SAM3/업스케일)
├── widgets/                  # PyQt 위젯 (Queue Panel 등 일부 잔존)
├── tabs/                     # 일부 탭의 PyQt 보조 위젯
├── utils/                    # 헬퍼 (테마, 캐릭터 특성, 디바이스 등)
├── frontend/
│   └── src/
│       ├── App.vue
│       ├── views/            # 각 탭별 Vue 컴포넌트
│       ├── components/       # 공용 UI 컴포넌트
│       ├── stores/           # widgetStore (state + IPC)
│       └── bridge.js         # QWebChannel 초기화
├── frontend_dist/            # Vue 빌드 산출물 (커밋됨, QWebEngineView가 로드)
├── config/                   # 앱 설정·경로·UI 상태 (설정의 기준 위치)
├── user_data/                # 사용자 프리셋·통계·API 비밀값·관리형 백엔드
├── cache/                    # 검색 결과·대기열·세션 복구 캐시 (삭제 가능)
├── logs/                     # 크래시·진단 기록
├── editor_models/            # YOLO + SAM 체크포인트 (gitignored)
├── danbooru_optimized/       # Danbooru parquet (gitignored)
└── CLAUDE.md                 # 개발 지침 (Claude Code 에이전트용)
```

---

## 🔧 개발 규칙

- **Vue 수정 후**: `cd frontend && npm run build` 필수
- **검증**: 백엔드 `venv\Scripts\python.exe run_tests.py`(시스템 python 이면 venv 로 재실행),
  프론트 `cd frontend && npm run test && npm run test:node && npm run type-check`(0 errors 유지)
- **PyQt 위젯 직접 사용 금지** — `WidgetProxy` 시스템 사용
- **Vue v-model 키 = Python widget_id**: 예) `widgets.character_input` ↔ `LineEditProxy(b, 'character_input')`
- **탭 간 이미지 전송**: `tabChanged` 먼저 emit → 100ms 후 이미지 시그널
- **에러 처리**: `core/error_handler.py` 전역 코드 시스템 (`ERROR_CODES` 표 — 사용 중인 코드는 E010·E020·E030·E040·E050·E100). 콘솔에는 원문, UI 로 가는 문구는 `sanitize_for_ui` 로 절대 경로를 가린다
- **커밋 메시지**: 한국어 `feat:` / `fix:` / `refactor:` / `docs:` / `chore:`

---

## 🤖 AI 코딩 도구

- [Claude Code](https://claude.com/claude-code): `CLAUDE.md` 참조해서 자동 개발
- Gemini CLI: `gemini chat "@파일경로 분석해줘" --no-stream`
- 협업 가이드: [`AGENTS.md`](AGENTS.md)

---

## 라이선스

MIT
