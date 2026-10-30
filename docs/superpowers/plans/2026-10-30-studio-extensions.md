# 스튜디오 확장기능·이미지 확대 Implementation Plan

> **For agentic workers:** superpowers:subagent-driven-development 로 태스크별 실행.

**Goal:** 스튜디오 제어판의 t2i/i2i 탭 아래·엔진룸 버튼 위에 sam-extra 확장기능 묶음을 넣고, 결과 이미지를 클릭하면 Forge 갤러리처럼 크게 보이게 한다.

**Architecture:** 설정은 패널 상태 `ext`(확장마다 `{on, 값들}`)에 저장한다. 확장 표(`web/ai_studio/ext_schema.json`)가 단일 출처다 — Python(`studio_state.normalize_state`)과 JS(`studio_logic`, `panel_view`)가 같은 파일을 읽는다. 엔진룸에 확장 노드를 앱 컴파일러와 같은 순서로 넣고, 꺼진 확장은 `stageModes`가 BYPASS(4)로 둬서 ComfyUI가 검증하지 않게 한다. 설정 칸은 표에서 자동으로 그린다.

**Spec:** 사용자 요청(2026-10-30): 확장은 다 넣되 **ADetailer·캐릭터 레퍼런스 제외**, 위치는 탭 밑·엔진룸 위. 결과 이미지 클릭 = 확대.

## Global Constraints

- 기존 계획의 constraints 파일(R1–R20, 커밋 규칙 R1 포함)을 그대로 따른다. 대상 ComfyUI 0.43.1 / 프런트엔드 1.57.4.
- 팩은 `core`를 import할 수 없다. 앱 스펙을 쓰려면 생성기가 JSON으로 뽑아 팩에 넣는다.
- 동작·순서의 기준은 앱 컴파일러 `core/comfy_workflow_compiler.py` (`_add_default_model_stack`, `_add_extra_samplers`, `_add_anima_guidance`, `_add_detail_daemon`, `_add_speed`, `_add_colorcraft`, `_add_hires`, `_add_image_extensions`, `_add_degrid`)이다. 앱 쪽 동작은 바꾸지 않는다.
- GPU 사용 금지, 사용자가 켜 둔 ComfyUI(8188)를 끄거나 재시작하지 않는다.
- 화면 글자는 짧게. 기존 팩 파일은 작게 유지(한 파일 ~400줄 넘으면 나눈다).

## Review Focus

1. 예전 워크플로(확장 노드 없음)를 열어도 오류 없이 동작한다 — 없는 역할은 건너뛰고, 그 확장 스위치는 회색 + 이유(NegPiP와 같은 방식).
2. 저장된 패널 상태에 `ext`가 없거나 값이 이상해도 기본값으로 열린다.
3. 꺼진 확장은 그래프에서 BYPASS라, 모델 파일이 없는 COMBO가 있어도 생성이 막히지 않는다.
4. 확대 창: 이미지가 없을 때 클릭해도 안 열리고, Esc·바깥 클릭·재클릭으로 닫힌다.
5. 켠 확장의 설정이 XYZ 칸·갤러리 후처리 실행에도 그대로 적용된다.

---

### Task 1: 결과 이미지 확대(라이트박스)

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/studio_lightbox.js`
- Modify: `web/ai_studio/result_view.js`, `web/ai_studio/studio.css`
- Test: `comfy_custom_nodes/ai_studio_forge_parity/js_tests/studio_lightbox.test.mjs`

**Interfaces:**
- Produces: `openLightbox(items: Array<{url: string}>, index: number, doc = document) -> { close(), show(i) }` — `document.body`에 고정 전체화면 오버레이를 붙인다(캔버스 변환 밖). 이미지는 화면에 맞춰 최대로(비율 유지), 여러 장이면 ←/→ 키와 좌우 버튼으로 넘긴다. Esc·배경 클릭·이미지 재클릭으로 닫고 키 리스너를 지운다. `nextIndex(i, delta, n)`(순환)은 순수 함수로 내보낸다.

- [ ] **Step 1: 실패 테스트** — `nextIndex` 순환(마지막 다음 = 0, 0 이전 = n-1), 가짜 document로 열기·Esc 닫기·리스너 제거, 이미지 없으면 `openLightbox([], 0)`가 아무것도 만들지 않음.
- [ ] **Step 2: 실패 확인** — `cd frontend && npm run test:pack`
- [ ] **Step 3: 구현** — `result_view.js`의 큰 이미지 클릭으로 연다. 미리보기 탭은 지금 보이는 이미지부터(갤러리 전체를 넘겨 보기), XYZ 격자 탭은 격자 이미지 1장. 진행 중 미리보기(`b_preview`)에서는 열지 않는다. 커서 `zoom-in`.
- [ ] **Step 4: 통과 확인** — `npm run test:pack`
- [ ] **Step 5: 커밋**

### Task 2: 확장 표·상태·JS 로직·워크플로 그래프

**Files:**
- Create: `tools/build_studio_ext_schema.py`(앱 스펙 → JSON), `web/ai_studio/ext_schema.json`, `web/ai_studio/studio_ext.js`(순수 로직)
- Modify: `studio_state.py`, `web/ai_studio/studio_logic.js`, `studio_roles.js`, `tools/build_anima_studio_workflow.py`, `example_workflows/AI Studio Anima.json`, golden fixture, README
- Test: `tests/test_studio_ext_schema.py`, `tests/test_studio_state.py`, `tests/test_studio_workflow_json.py`, `js_tests/studio_ext.test.mjs`

**Interfaces:**
- `ext_schema.json`: `{"extensions": [{"key", "label", "group", "role", "class", "form": "json"|"widgets", "on_key"?, "fields": [{"key", "label", "kind": "bool"|"int"|"float"|"choice"|"text", "default", "min"?, "max"?, "choices"?}]}]}` 표시 순서대로. `group`은 `가이던스 / 디테일 / 샘플러·VAE / SAM3`.
- 확장 목록: 가이던스 스위트(`ForgeNeoAnimaGuidanceSuite`, `core/anima_guidance.PERTURBATION_SPEC`의 필드 전체 — PAG/SEG/SLG/S²/APG/ADG/SMC/CWM/DCW/RDC/DAVE/CNS/Mod/TSR/MG/HiGS/HiFlow), Optimal Scale(`OPTIMAL_SCALE_SPEC`), Skim CFG(`SKIMMED_SPEC`), Detail Daemon(`DETAIL_DAEMON_SPEC`), SPEED(`ForgeNeoAnimaSpeed`), ER SDE 값(`ForgeNeoExtraSamplerSettings`), LoRA 블록 가중치(`ForgeNeoLoraBlockWeight`, LoRA 하나), VAE 2x(`ForgeNeoAnimaVAE2x`), VAE DeGrid(`ForgeNeoAnimaVAEDeGrid`), PiD(`ForgeNeoAnimaPiD`), SAM3 Refine(`ForgeNeoSAM3Refine`), SAM3 타일 복구(`ForgeNeoSAM3TileRepair`). 별도 노드 `ForgeNeoAnimaSafePAG`·`ForgeNeoAnimaDAVE`·`ForgeNeoAnimaModGuidance`·`ForgeNeoDCWCWMSMC`는 스위트가 같은 기능을 하므로 넣지 않는다. **Colorcraft**: 앱의 `core/colorcraft.build_chain`을 팩에 옮겨야 하면 이번에 넣지 않고 보고서에 이유를 적는다. 이 판단은 구현자가 코드를 읽고 한다.
- `studio_state`: `DEFAULT_STATE["ext"]` = 표의 확장마다 `{"on": False, 필드 기본값…}`, `normalize_state`는 표 기준으로 형 변환·범위 자르기, 모르는 키 버림. 표는 `web/ai_studio/ext_schema.json`을 `Path(__file__)`로 읽는다.
- `studio_ext.js`: `extInputs(state, role, extras) -> object`(그 확장 노드에 넣을 입력; `form: "json"`이면 `settings_json`을 컴파일러와 같게 구성 — 스위트는 `cfg_scale`, S² 켜면 `guid_s2_seed`, HiFlow는 t2i+Hires일 때만, DD는 `cfg_scale_override`), `extModes(state, roleIds) -> {role: 0|4}`.
- `studio_logic`: `applyStateToPrompt`·`stageModes`가 위 함수를 쓴다(`stageModes`는 꺼진 확장 role을 4로). 역할이 없으면 건너뛴다.
- 그래프: 앱 컴파일러 순서대로. 메인 패스 모델 체인은 LoRA→NegPiP→DoRA→ER SDE→(③ 프롬프트 뒤) Optimal Scale→스위트→Skim CFG→DD→SPEED→샘플러. 디테일러·Hires가 어떤 모델을 받는지도 컴파일러와 같게 한다(SPEED·Colorcraft는 디테일러에 안 감). VAE 2x는 디코드 VAE 앞, DeGrid는 최종 이미지 뒤, Refine·타일 복구는 SAM3 단계. 새 줄 제목은 "④ 확장"으로 하고 뒤 줄 번호를 민다. `test_links_flow_one_way`·`test_engine_rows_top_to_bottom`을 지킨다. 후처리 줄(R19)은 바꾸지 않되, 후처리 실행은 메인 패스와 같은 모델을 받는다.

- [ ] **Step 1: 실패 테스트** — 스키마: 앱 스펙의 키 수·기본값과 JSON이 일치(`core` import 가능한 venv 테스트), ADetailer·캐릭터 레퍼런스 없음. 상태: `ext` 없음/이상한 값 → 기본값. JS: 스위트 켬 → `settings_json`에 `cfg_scale`, HiFlow는 i2i에서 꺼짐, 꺼진 확장 role = 4, 없는 role 건너뜀. 그래프: 모든 role 한 번, 모델 체인 순서, 링크 한 방향.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — `venv\Scripts\python.exe tools\build_studio_ext_schema.py`, `tools\build_anima_studio_workflow.py`로 파일 생성
- [ ] **Step 4: 통과 확인** — `npm run test:pack`, `venv\Scripts\python.exe run_tests.py` 전체
- [ ] **Step 5: 커밋**

### Task 3: 제어판 "확장기능" 칸

**Files:**
- Create: `web/ai_studio/ext_view.js`
- Modify: `web/ai_studio/panel_view.js`, `web/ai_studio/studio.css`, `web/ai_studio/studio_main.js`
- Test: `js_tests/ext_view.test.mjs`

**Interfaces:**
- Consumes: Task 2 `ext_schema.json`, `extModes`.
- Produces: `renderExtensions(state, { set, hasRole, fetchChoices }) -> HTMLElement` — 그룹 "확장기능"을 `tabs(s)`와 `engineBar()` 사이에 넣는다. 확장마다 한 줄(켜기 스위치·이름·펼침 ▾), 펼치면 필드를 표대로 자동 생성(bool=스위치, int/float=숫자 칸, choice=선택, text=입력). `choices`가 비어 있는 COMBO(모델 파일 등)는 `/object_info/<class>`에서 읽는다. 스위트처럼 필드가 많은 확장은 표의 소제목(`section`)으로 묶어 접는다. 켠 확장 수를 그룹 제목 옆에 표시. 그래프에 그 role이 없으면 스위치 회색 + 이유(NegPiP의 `NEGPIP_OLD_GRAPH`와 같은 문구 방식). 펼침 상태는 저장하지 않는다(뷰 안에서만).

- [ ] **Step 1: 실패 테스트** — 가짜 DOM으로: 켜기 누르면 `state.ext[key].on` 토글, 필드 입력이 범위로 잘림, role 없으면 스위치 비활성.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — 시안 C4의 카드·스위치 색을 그대로 쓴다.
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋**

### Task 4: 설치·CPU 확인

**Files:** 없음(발견된 결함만 수정)

- [ ] **Step 1:** 전체 검증(`run_tests.py`, `npm run test:pack`).
- [ ] **Step 2:** 사용자의 ComfyUI가 켜져 있으면 팩 폴더 교체가 막히므로, 바뀐 파일만 제자리에 덮어쓰고 `.aistudio-owned.json`을 갱신한다(앞선 NegPiP 설치와 같은 방식).
- [ ] **Step 3:** CPU 전용 ComfyUI(8199, `--cpu`, `CUDA_VISIBLE_DEVICES=-1`)와 헤드리스 Chrome으로: 새 워크플로 열기 → 빠진 노드 0, 확장 칸 표시, 켜고 끄기, `app.graphToPrompt()`에 켠 노드만 포함, 이미지 클릭 확대 동작. 스크린샷을 워크스페이스에 저장. 큐에는 넣지 않는다.
- [ ] **Step 4:** 커밋(수정이 있으면).
