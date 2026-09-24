# UR_IV — Claude Code 작업 가이드

PyQt6(백엔드) + Vue 3 SPA(프론트) 데스크탑 AI 이미지 생성기. (내부명 AI Studio Pro)

## ⚠️ 작업 전 필수 (이걸 안 지켜서 실수 많았음)
- **모든 git / npm build 는 이 파일이 있는 메인 저장소 루트에서** 한다.
  `.claude/worktrees/...` 경로에서 빌드/커밋하면 변경이 유실되거나 dist가 안 맞는다.
- **Vue(`frontend/src/`) 수정 후엔 반드시 `cd frontend && npm run build`** — dist를 안 만들면
  앱은 옛 화면을 보여준다. `frontend_dist`도 같이 커밋.
- **Python 수정 후엔 `venv\Scripts\python.exe run_tests.py`** 로 회귀 검증. (자동 훅으로도 돈다.)
  ⚠ 시스템 `python`(PATH 첫 항목은 3.10)엔 pandas/PIL/PyQt6 가 없어 **가짜 ImportError 수십 개**가 난다 —
  반드시 venv. `run_tests.py` 는 venv 밖에서 불리면 venv 로 스스로 재실행하지만(`URIV_TESTS_REEXEC=1` 로 끔),
  `py_compile` 등 다른 명령엔 그런 장치가 없다.
- **`frontend/src/utils/` 의 순수 로직을 고쳤으면 `cd frontend && npm run test`** (vitest).
  히스토그램·커브·그리기 도구·도구 레지스트리가 여기서 검증된다.
- API 키/시크릿/토큰은 절대 커밋 금지. 노출되면 재발급 안내.

## 검증 / 배포
- `/verify` — venv 테스트(전체) + py_compile + (프론트 수정 시) vitest · node --test · type-check · 빌드.
  **커밋 안 함**. (`.claude/commands/` 와 Codex 용 `.agents/skills/source-command-*` 는 로컬 전용 사본이라
  둘을 같이 고친다.)
- `/ship` — 위 검증 전부 통과 → 빌드 → 커밋(한국어 conventional) → 푸시.
- 수동: `venv\Scripts\python.exe run_tests.py` (pytest 불필요, 표준 unittest).
- 프론트 순수 로직: `cd frontend && npm run test` (vitest, `src/**/*.test.ts` 만).
  `src/studio/*.test.mjs` 는 node:test 라 vitest 범위 밖이다 — `npm run test:node` 로 돈다.

## 아키텍처
- 프론트: Vue 3 SPA in QWebEngineView — `frontend/src/`
- 백엔드: PyQt6 (QMainWindow + QWebChannel) — `ui/`, `core/`, `backends/`
- 브리지: `ui/vue_bridge.py`(Python @pyqtSlot/Signal) ↔ `frontend/src/bridge.js`
- 위젯 프록시: `ui/widget_proxies.py` — Vue `storeWidgets.<id>` ↔ 프록시 widget_id
- 상태 저장소: `frontend/src/stores/widgetStore.js`

## 작업 방식 — 작은 단일책임 파일 선호 (중요)
- **거대 파일보다 작은 파일이 낫다.** 부분만 정확히 읽고 고칠 수 있고, 버그가 격리되며,
  전체를 안 읽어도 돼 빠르고 실수가 적다. (이게 App.vue 분할·utils/core/tabs 분리의 이유)
- 큰 파일(App.vue, `ui/generator_main.py` 등)을 만질 땐 **먼저 grep으로 위치를 찾고
  Read는 offset/limit로 해당 부분만** — 통째로 로드하지 말 것. (줄 수는 적지 않는다 — 분할하면서 금방 낡는다.)
- App.vue 에서 이미 떼어낸 것 — 거기서 찾지 말 것:
  파라미터 열 카드 `components/params/*`(ADetailer 슬롯은 `AdSlotFields.vue`) ·
  매니저 모달 `components/managers/*`(표시만; 상태·백엔드 리스너는 `usePresetManager`·`useWildcardManager`·
  `useGlobalWeights` 같은 모듈 싱글턴 composable) · 토스트 `composables/useToasts.ts`+`ToastLayer.vue` ·
  전역 단축키 `utils/appShortcuts.ts` · 모달 ESC/↑↓ 게이팅 `utils/modalStack.ts`+`composables/useModalLayer.ts`
  (**새 모달은 setup 에서 `useModalLayer()` 를 부른다** — `App.modalLayer.test.ts` 가 `*Modal.vue` 전부를 검사).
- **새 응집 기능은 거대 파일에 더하지 말고 별도 모듈/composable로 만든다.**
  - 프론트: `frontend/src/composables/use*.js` (예: `useLoraStack`/`useHighRes`/`useRatingFilter`).
    App.vue는 `<script setup>` 최상위 destructure로 노출 — **템플릿이 쓰는 이름 전부 destructure**
    해야 함(누락 시 빌드는 통과해도 런타임에 깨짐). 공유 헬퍼(`saveUiPrefs`/`addToast`)는 주입.
  - 백엔드: 순수 로직은 Qt에서 분리해 `core/`·`utils/`로 (테스트도 같이).
- 분할/추출은 **동작 보존(위치만 이동)** 원칙 + 추출마다 `npm run build`/`run_tests.py`/스모크.

## 브리지 계약 (불일치 = 버그 주원인)
- 액션: Vue `requestAction(name, payload)` / `action(name)` → `generator_main._handle_vue_action`
  의 `action == 'name'` 또는 `action in ('a','b')`
- 이벤트: Python `vue_bridge.<signal>.emit(json)` → Vue `onBackendEvent(name, cb)`
- 페이로드 형태를 양쪽이 똑같이 맞춰야 함 (예전 버그: 조건식 `target`(문자열) vs `tags`(리스트))
- **회귀 가드**: `tests/test_bridge_contract.py` — 양방향 정적 검증(AST).
  - 정방향: 프론트 액션/이벤트 이름에 Python 핸들러/시그널이 있는지.
  - 역방향: Python 에만 남은 액션·@pyqtSlot·시그널·`_WEB_METHODS`/`_WEB_SIGNALS` 이름은 실패.
    Python 이 직접 쓰는 것은 `PYTHON_INTERNAL`, 정리 예정 사문은 `PENDING`(사유 필수)에 적는다.
    정리하면 목록에서도 지운다(낡은 항목도 실패).
  - 이름 emit(`_creator_emit("X")`·`getattr(bridge, "X")`)의 X 가 실제 시그널인지.
  - 핸들러의 `action` 비교 대상은 리터럴·모듈/클래스 상수·함수 안 지역 표(한 번 대입)로 둔다 —
    정적으로 못 읽으면 테스트가 실패한다.

## 프론트 타입 (점진 TS 도입 ②)
- 툴: `typescript` + `vue-tsc` 설치됨. `frontend/tsconfig.json`(allowJs, checkJs:false —
  기존 .js는 느슨, 새 .ts/lang="ts"만 엄격). **Vite 빌드는 esbuild라 타입검사 안 함** → 타입검사는
  `cd frontend && npm run type-check`(= `vue-tsc --noEmit`). **현재 0 errors가 베이스라인**.
- `frontend/src/types/bridge.d.ts` — 브리지 계약 타입(`ActionName`/`BackendEvent` + 페이로드).
  `requestAction` JSDoc 이 `<K extends ActionName>` 제네릭이라, `ActionPayloads` 맵에 적힌 액션은 TS
  호출부(뷰의 `action()` 래퍼 포함)에서 **페이로드 모양까지** type-check 된다(맵에 없으면 `object`,
  `.js` 호출부는 검사 밖). 이벤트는 수신부가 `JSON.parse(json) as XxxEvent` 로 연결한다.
  `types/bridgePayloads.test-d.ts` 가 `@ts-expect-error` 로 이 강제가 살아 있는지 지킨다.
  이름 정합성 **강제는 `tests/test_bridge_contract.py`**.
- **점진 전환**: 컴포넌트마다 `<script setup lang="ts">` + 타입 기반 props/emits로 전환,
  전환 후 `npm run type-check`(0 유지) + 런타임 스모크. 첫 예시: `components/ToggleSwitch.vue`
  (`withDefaults(defineProps<{...}>(), {...})` 패턴). 한 번에 갈아엎지 말 것.

## 커밋 규칙
- 한국어 conventional: feat/fix/refactor/test/chore/docs
- 메시지 끝에 `Co-Authored-By: Claude ...` 트레일러
- 런타임 데이터 커밋 금지: `config/cond_rules.json`, `config/char_global_prefs.json`,
  `cache/session/session_backup.json`(옛 위치 `config/session_backup.json`) (gitignore 처리됨)

## 테스트
- 백엔드: `tests/` (표준 unittest), 실행 `venv\Scripts\python.exe run_tests.py` — **venv 필수**
- 프론트: 소스 옆에 `*.test.ts` (vitest). 커버: 커브 LUT(파이썬 `core/curves.py` 와 같은
  golden 값으로 두 구현이 갈라지지 않게) / 히스토그램 / 플러드 필 / 도구 단축키
- `--quick` = `run_tests.SLOW_MODULES`(소켓·서브프로세스·스레드 대기를 쓰는 통합 테스트) 제외.
  PostToolUse 훅은 `.py` 편집마다 `--quick` 을 돌고, 편집한 파일이 느린 모듈이 **직접** 검증하는
  소스(또는 그 테스트 파일)일 때만 **그 모듈만** `--include` 로 더한다.
  느린 모듈을 빼도 나머지 모듈의 긴 꼬리가 남아 quick 도 즉시 끝나지는 않는다.
  소요 시간·테스트 수 실측치는 **여기 적지 않는다**(복붙한 옛 수치가 몇 배로 낡은 적이 있다).
  단일 출처는 `run_tests.py` 의 SLOW_MODULES 주석(실측 날짜 포함)이고,
  `venv\Scripts\python.exe run_tests.py --durations 15` 로 모듈별 시간·discovery(import) 시간을 다시 잰다.
- **torch 는 테스트 모듈 최상위에서 import 하지 않는다**(discovery 가 quick 이 거를 모듈도 import 한다).
  torch 가 필요한 테스트 클래스·메서드엔 `tests/_optional_deps.requires_torch` 를 붙이고, torch 는
  `load_torch()`(테스트 안) 또는 표시된 클래스 `setUpClass` 의 `bind_torch(globals())` 로 지연 import 한다.
  `--quick` 은 표시된 테스트를 거르고(`--with-torch` 로 되살림), 훅은 `run_tests.TORCH_TEST_SOURCES`
  (comfy_custom_nodes/ 등)나 표시가 든 테스트 파일을 고칠 때만 `--with-torch` 를 더한다.
  규칙은 `tests/test_optional_deps.py` 가 AST 로 지킨다. 소스의 전이 import 로 torch 가 올라와도
  run_tests 가 **실패로 끝낸다**(discovery 는 모든 모드, 실행 중엔 `--quick`) — stderr 끝의
  '처음 import 한 곳'을 보고 그 소스의 torch import 를 함수 안으로 옮긴다.
- 순수 로직은 Qt에서 분리해 테스트 추가 (예: `core/resolution_guard.py`)
- 커버: 조건식 / 캐릭터 분류 / NL 누출제거 / ANIMA 해상도캡
