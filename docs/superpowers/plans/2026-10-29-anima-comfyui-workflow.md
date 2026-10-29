# Anima ComfyUI 스튜디오 워크플로 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자 ComfyUI 안에 Forge(WebUI)처럼 쓰는 Anima 생성 화면(제어판·t2i/i2i 탭·결과 갤러리와 후처리 버튼·Forge Neo 동작 그대로의 XYZ)을 앱 팩에 만든다.

**Architecture:** 화면(제어판·결과·XYZ)은 앱 팩의 프런트 확장(JS)이 그리는 **UI 전용 노드 3개**의 DOM 위젯이다. 실제 생성은 엔진룸의 일반 노드
그래프(대부분 기존 `ForgeNeo*` 노드)가 하고, JS는 생성 직전에 제어판 상태(`studio_state` JSON)를 **역할(role)이 붙은 엔진 노드의 입력값**에
써 넣은 뒤 큐에 넣는다(제어판→엔진 사이에 선이 없다). XYZ·값 문법·격자 그리기·Forge 생성 정보는 **Python 순수 모듈 + 팩 서버 라우트**로
두어 unittest로 고정하고, JS는 얇게 유지한다.

**Tech Stack:** Python 3.13(ComfyUI 내장)·3.10/venv(앱 테스트) unittest, PIL; ComfyUI 0.36.0 + frontend 1.52.7 확장 API(`window.comfyAPI`),
ES 모듈 JS + `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-29-anima-comfyui-workflow-design.md` · 화면 확정안: https://claude.ai/artifact/3gfEnw1YFovVVwV8QLyccm (C4)
· 조사 메모(이 세션 scratchpad `plan_recon\*.md`): `forge_xyz_behavior.md`(Forge XYZ 동작 명세), `reuse_nodes.md`, `pack_integration.md`,
`comfy_frontend_api.md`, `sam31_loading.md` — 구현자는 해당 태스크에서 이 메모를 먼저 읽는다.

## Global Constraints

- 모든 새 코드는 `comfy_custom_nodes/ai_studio_forge_parity/` 안(팩, MIT). 새 Python 모듈 이름은 `studio_*` / `xyz_*` (금지 접두어: `speed_`,
  `colorcraft_`, `extra_samplers`, `extra_schedulers`).
- Forge `scripts/xyz_grid.py`·`modules/images.py`는 AGPL-3.0: **코드 복사 금지**, 동작만 따라 새로 쓴다(`TestPackCarriesNoForgeCode` 유사도 <0.80 유지).
  sam-extra 자체 코드는 MIT 재사용 가능, 그 밖의 제3자 GPL 금지.
- 테스트 모듈 최상위에서 torch import 금지(`tests/_optional_deps.requires_torch`). ComfyUI는 `mock.patch.dict(sys.modules)`/`SimpleNamespace`로 흉내.
- Python 수정 후 `venv\Scripts\python.exe run_tests.py`, 프런트(팩 JS) 수정 후 `cd frontend && npm run test:pack`. 커밋·푸시는 **사용자가 시킬 때만**.
- 화면 글자는 짧게("t2i <> i2i" 식). 버튼 이름: 폴더 · → i2i · → 인페인트 · 업스케일 · Hires · SAM3 · 같은 시드(이 순서).
- 기본값: sampler `exp_heun_2_x0_sde`(표기 "EXP Heun 2 x0 SDE"), scheduler `flow_cosmos_dynamic`("Flow Cosmos Dynamic"), steps 32, cfg 3.0,
  shift 3.0, 크기 832×1216, Hires 1.5배 · denoise 0.4 · steps 16 · upscaler `4x-UltraSharp.pth`(없으면 `latent:bicubic`), i2i denoise 0.55,
  mask blur 4, SAM3 prompt `face`, SAM3 denoise 0.4.
- GPU를 쓰는 확인(실제 생성)은 **사용자에게 먼저 묻고 VRAM 사용량을 밝힌다**(학습이 자주 돈다).
- 파일은 작게(한 책임), 새 노드는 README "### Custom ComfyUI workflows only"에 등록.

## Review Focus

1. XYZ 도중 한 칸이 실패(OOM 등)하거나 정지를 누름 → Forge처럼 그 칸(과 남은 칸)은 검은 칸으로, 나머지로 격자는 완성된다. (Task 7 grid 테스트, Task 11 JS 테스트)
2. Prompt S/R의 찾을 글이 프롬프트·네거티브·목록 LoRA 태그 어디에도 없음 → 큐에 넣기 전에 Forge 문구로 막는다. (Task 6 테스트)
3. 격자 라벨에 한글 등 Roboto에 없는 글자 → 네모가 아니라 한글 글꼴로 그려지고 줄바꿈된다. (Task 7 테스트)
4. 예전 워크플로에 저장된 제어판 상태에 키가 빠졌거나 값이 이상함(문자 steps, 없는 샘플러) → 기본값으로 메워 열린다. (Task 2 테스트)
5. 세션 기록이 없는 이미지(디스크에서 연 것)에 갤러리 후처리 버튼 → 이미지의 Forge 생성 정보로 프롬프트·시드를 채우고, 그것도 없으면 짧은 안내만. (Task 11 테스트)

---

### Task 1: 팩에 웹 확장·라우트 뼈대 + 설치기 지문

**Files:**
- Modify: `comfy_custom_nodes/ai_studio_forge_parity/__init__.py` (`WEB_DIRECTORY = "./web"`, 라우트 등록 호출)
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_routes.py` (라우트 등록만; 핸들러는 뒤 태스크가 추가)
- Create: `comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/studio_main.js` (빈 `registerExtension({name:"AIStudio.Studio"})`)
- Modify: `core/comfy_node_pack.py` (`_PACK_SUFFIXES`에 `.js`, `.css` 추가 — 재시작 목록에는 넣지 않음)
- Test: `tests/test_studio_routes.py`, `tests/test_comfy_node_pack.py`(기존 가드 갱신)

**Interfaces:**
- Produces: `studio_routes.register_routes() -> bool` (idempotent; `sys.modules.get("server")`가 없으면 False — `h3_cache_nodes._register_routes` 패턴),
  `studio_routes.ROUTE_PREFIX = "/aistudio/studio"`, `studio_routes.add_handler(method: str, path: str, handler)` (뒤 태스크가 핸들러를 끼움).

- [ ] **Step 1: 실패 테스트** — `test_register_routes_without_server_is_noop`(server 없음 → False, 예외 없음), `test_register_routes_is_idempotent`
  (가짜 `server.PromptServer.instance.routes`에 두 번 호출 → 경로 1번만 등록), `test_pack_exports_web_directory`(`pack.WEB_DIRECTORY == "./web"`),
  `test_js_change_changes_fingerprint`(설치기 지문 함수에 `.js`만 다른 두 폴더 → 지문 다름, `restart_required`는 False).
- [ ] **Step 2: 실행해 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_routes -v` → FAIL(모듈 없음)
- [ ] **Step 3: 구현** — 위 Interfaces 그대로. `WEB_DIRECTORY` 아래에는 `*.js`만 두고 테스트 파일은 두지 않는다(ComfyUI가 `**/*.js`를 다 읽음).
- [ ] **Step 4: 통과 확인** — 같은 명령 PASS, `venv\Scripts\python.exe run_tests.py` 전체 PASS
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 2: 제어판 상태 스키마 `studio_state`

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_state.py`
- Test: `tests/test_studio_state.py`

**Interfaces:**
- Produces: `DEFAULT_STATE: dict` (Global Constraints의 기본값; 키: `version=1, tab("t2i"|"i2i"), i2i_mode("img2img"|"inpaint"), prompt, negative,
  unet, te, vae, sampler, scheduler, steps, cfg, shift, seed(-1 허용), width, height, batch, loras:[{on,name,strength}], forge_like:{master,noise,
  prompt_syntax,dora,attention}, hires:{on,upscaler,scale,steps,denoise}, upscale:{on,model,scale}, sam3:{on,checkpoint,prompt,denoise},
  i2i:{image,denoise,mask_blur,inpaint_area("masked"|"whole")}`);
  `normalize_state(raw: object) -> dict` (모르는 키 버림, 빠진 키 기본값, 형 변환 실패 → 기본값, 범위 밖 → 잘라냄: steps 1..150, cfg 0..30, shift 0..100,
  width/height 64..4096 step 8, batch 1..16, strength −10..10);
  `effective_prompt(state: dict) -> str` (켜진 목록 LoRA를 `<lora:이름:세기>` 태그로 프롬프트 끝에 붙인 글 — 세기는 `repr` 대신 `f"{v:g}"`).

- [ ] **Step 1: 실패 테스트** — `test_defaults_match_user_forge_settings`(sampler `exp_heun_2_x0_sde`, scheduler `flow_cosmos_dynamic`, steps 32,
  cfg 3.0, shift 3.0, width 832, height 1216, hires scale 1.5/denoise 0.4/steps 16), `test_missing_and_bad_values_fall_back`
  (`{"steps":"abc","cfg":99,"foo":1}` → steps 32, cfg 30.0, `foo` 없음), `test_effective_prompt_appends_enabled_loras`
  (prompt `"1boy"`, loras `[{on:true,name:"a",strength:0.9},{on:false,name:"b",strength:1}]` → `"1boy <lora:a:0.9>"`).
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_state -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 3: Forge LoRA 태그 규칙 + `AIStudioLoraStack` 노드

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_lora_tags.py` (순수: Forge 규칙)
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_nodes.py` (이 태스크에서 `AIStudioLoraStack` 하나; 뒤 태스크가 노드를 더함)
- Modify: `__init__.py`(노드 맵 병합), `README.md`, `tests/test_comfy_node_pack.py`(노드 수 44 → 45, `CUSTOM_WORKFLOW_NODES`)
- Test: `tests/test_studio_lora_tags.py`, `tests/test_studio_nodes_lora.py`

**Interfaces:**
- Consumes: Task 2 `effective_prompt`.
- Produces: `parse_lora_tags(prompt: str, *negatives: str) -> tuple[list[LoraTag], list[str]]` — `LoraTag(name: str, strength_model: float,
  strength_clip: float)`; 동작은 `core/comfy_workflow_compiler.py:496 parse_lora_tags`와 **같은 결과**(태그 제거, `<lora:n:a:b>` = clip a / model b,
  `<lora:n:a>`/`<lora:n>` = 둘 다 a / 1.0, 닫히지 않은 `<x:` 가 삼킨 태그는 떼기만, `<lora :x>`는 글자 그대로).
  노드 `AIStudioLoraStack`: 입력 `model MODEL, clip CLIP, positive STRING(multiline), negative STRING(multiline)`, 출력
  `("MODEL","CLIP","STRING","STRING") = (model, clip, positive, negative)` — 태그를 떼고 `ForgeNeoAnimaLoraLoader`의 함수로 순서대로 건다.
  같은 이름이 여러 번이면 마지막 태그 세기 한 번만. 파일을 못 찾으면 Forge처럼 경고 한 줄 남기고 건너뛴다.

- [ ] **Step 1: 실패 테스트** — `test_matches_app_compiler_rules`: 같은 입력 묶음(최소 12개: 기본, a:b 두 값, 값 없음, 공백 콜론, 닫히지 않은 `<3:`,
  hypernet 태그, 네거티브의 태그, 한글 이름, 소수점 `0.75`, 음수 `-0.5`, 같은 LoRA 두 번, 태그 사이 쉼표)에 대해 팩 함수와
  `core.comfy_workflow_compiler.parse_lora_tags`의 (이름·세기 목록, 남은 문장) 이 같다(골든 대조, 커브 LUT 테스트 방식).
  `test_stack_applies_last_strength_once`(가짜 로더로 `"<lora:a:1> x <lora:a:0.8>"` → 로더 호출 1번, strength 0.8, 문장 `" x "` 정리 결과 `"x"`),
  `test_missing_lora_is_skipped_with_warning`.
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_lora_tags tests.test_studio_nodes_lora -v`
- [ ] **Step 3: 구현** — 노드는 로더 인스턴스를 노드 객체에 캐시해 같은 큐에서 재사용(`ForgeNeoAnimaLoraLoader`의 상태 캐시를 살림).
- [ ] **Step 4: 통과 확인** — 위 명령 + `run_tests.py` 전체
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 4: Forge 생성 정보(`parameters`)와 `AIStudioSaveForge`, `AIStudioUpscaleImage`

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_infotext.py`
- Modify: `studio_nodes.py` (+2 노드), `README.md`, `tests/test_comfy_node_pack.py`(45 → 47)
- Test: `tests/test_studio_infotext.py`, `tests/test_studio_nodes_save_upscale.py`

**Interfaces:**
- Consumes: Task 2 `normalize_state`.
- Produces: `forge_sampler_label(name: str) -> str`, `forge_scheduler_label(name: str) -> str` (comfy 이름 → Forge 표기; 표는 앱 컴파일러
  `_FORGE_SAMPLER_ALIASES`/`_FORGE_SCHEDULER_ALIASES`의 역방향 + `exp_heun_2_x0_sde`→`EXP Heun 2 x0 SDE`; 모르면 그대로);
  `build_infotext(state: dict, *, seed: int, extra: dict[str,str] | None = None) -> str` (A1111 형식: 1줄 프롬프트, `Negative prompt: …`,
  마지막 줄 `Steps: 32, Sampler: …, Schedule type: …, CFG scale: 3, Shift: 3, Seed: N, Size: 832x1216, Model: <unet stem>, VAE: <file>, …`;
  i2i면 `Denoising strength`, Hires 켜면 `Hires upscale`, `Hires steps`, `Hires upscaler`, `Denoising strength`(hires) — 키 이름·순서는
  Forge Neo가 txt2img에서 쓰는 순서; `extra`는 끝에 덧붙임(XYZ용)).
  노드 `AIStudioSaveForge`(OUTPUT_NODE): 입력 `images IMAGE, studio_state STRING, seed INT, filename_prefix STRING="AIStudio/anima"`,
  PNG에 `parameters`(위 infotext) + ComfyUI `prompt`/`workflow` 청크, UI 결과 `{"images":[{filename,subfolder,type}]}` 반환.
  노드 `AIStudioUpscaleImage`: 입력 `image IMAGE, enabled BOOLEAN, model_name COMBO(upscale_models), scale_by FLOAT 2.0 [1,8]`, 출력 `IMAGE`
  — 꺼지면 그대로, 켜면 모델 업스케일 후 원본×scale_by 로 맞춤(core `UpscaleModelLoader`/`ImageUpscaleWithModel` 함수 재사용).

- [ ] **Step 1: 실패 테스트** — `test_infotext_txt2img_matches_forge_layout`(기본 상태·seed 42 →
  마지막 줄이 정확히 `Steps: 32, Sampler: EXP Heun 2 x0 SDE, Schedule type: Flow Cosmos Dynamic, CFG scale: 3, Shift: 3, Seed: 42, Size: 832x1216, Model: …`
  로 시작), `test_infotext_hires_keys`(hires 켬 → `Hires upscale: 1.5, Hires steps: 16, Hires upscaler: 4x-UltraSharp` 포함),
  `test_app_reader_reads_our_parameters`(만든 PNG를 `core.image_metadata.read_metadata_for_ui`(또는 `extract_from_pil`)로 읽어 prompt·seed·steps 일치),
  `test_upscale_disabled_passthrough`(가짜 image 그대로 반환).
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_infotext tests.test_studio_nodes_save_upscale -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 위 + 전체
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 5: XYZ 값 문법과 축 목록

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/xyz_values.py`, `xyz_axes.py`
- Test: `tests/test_xyz_values.py`, `tests/test_xyz_axes.py`

**Interfaces:**
- Produces: `split_values(text: str) -> list[str]` (CSV + 줄바꿈, `skipinitialspace`, 따옴표, strip);
  `parse_axis_values(kind: Literal["int","float","str","choice","permutations","size","nothing"], text: str) -> list` (메모 §2 그대로);
  `AxisSpec(id: str, label: str, kind: str, field: str, cost: float, fmt: Literal["label_value","value","basename","joined","empty"], choices: str | None)`;
  `AXES: tuple[AxisSpec, ...]` 순서 = Forge 표시 순서. v1 축(라벨은 Forge와 글자까지 같게):
  `Nothing, Seed, Steps, Hires steps, Size, CFG Scale, Shift, Prompt S/R, Prompt order, Sampler, Hires sampler, Schedule type, Checkpoint name,
  VAE, Denoising, Hires upscaler, [Modules] Text Encoder, [DoRA] Inference mode, [SAM3] Enable, [SAM3] Checkpoint, [SAM3] Detect Prompt,
  [SAM3] Denoising Strength, [SAM3] Steps, [SAM3] CFG Scale`; cost: Checkpoint 2.0, Text Encoder 0.9, SAM3 Checkpoint 0.9, DoRA 0.8, VAE 0.7,
  Prompt S/R 0.1, Prompt order 0.1, 나머지 0. `field`는 Task 2 상태의 경로(예: `"hires.steps"`). `choices`는 Task 8 라우트 종류 이름
  (`samplers, schedulers, unets, vaes, tes, upscalers, sam3, dora_modes`).
  **결정(Claude):** 범위 결과는 Forge와 같게(정수 음수 스텝에서 끝값 빠짐, `0.1-0.3 (+0.1)` → 4개 포함), 단 `1e-3`은 범위로 읽지 않고 숫자 하나로.

- [ ] **Step 1: 실패 테스트** — 메모 §2 예시를 그대로: `1-5`→[1,2,3,4,5]; `1-10 (+3)`→[1,4,7,10]; `0-9 (+4)`→[0,4,8]; `-2-2`→[-2,-1,0,1,2];
  `10-1 (-3)`→[10,7,4]; `5-1`→[]; `1-10 [5]`→[1,3,5,7,10]; `0-100 [3]`→[0,50,100]; `10-0 [3]`→[10,5,0]; float `1-2 (+0.25)`→[1.0,1.25,1.5,1.75,2.0];
  `1-0 (-0.25)`→[1.0,0.75,0.5,0.25,0.0]; `0.1-0.3 (+0.1)`→4개; `0-1 [5]`→[0.0,0.25,0.5,0.75,1.0]; `1e-3`(float)→[0.001];
  str `a,,b`→['a','','b']; `a, "b, c"`→['a','b, c']; 줄바꿈 `"a\nb"`→['a','b']; int 축 빈칸 건너뜀; permutations `a, b, c`→ 6개 순열(itertools 순서);
  `test_axis_labels_match_forge`(위 라벨 목록 고정), `test_every_axis_field_exists_in_default_state`.
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_xyz_values tests.test_xyz_axes -v`
- [ ] **Step 3: 구현** — 정규식은 메모 §2의 네 개(점은 `\.`로 이스케이프).
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 6: XYZ 계획기 (칸 목록·순서·시드·S/R·라벨)

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/xyz_plan.py`
- Test: `tests/test_xyz_plan.py`

**Interfaces:**
- Consumes: Task 2 `normalize_state`·`effective_prompt`, Task 5 `AXES`·`parse_axis_values`.
- Produces: `plan_xyz(state: dict, axes: list[dict], options: dict, *, rng: random.Random, families: dict[str,str]) -> XyzPlan`;
  입력 `axes = [{"axis": "<AxisSpec.id>", "text": "<값 글>"}] * 3`(X,Y,Z), `options = {"draw_legend":True,"keep_minus1":False,"vary_x":False,
  "vary_y":False,"vary_z":False,"include_sub_images":False,"include_sub_grids":False,"margin":0,"row_count":0}`;
  `XyzPlan(cells: list[Cell], order: list[int], labels: dict[str,list[str]], fixed: dict[str,list[int]], infotext_extra: dict[str,str],
  errors: list[str])`, `Cell(ix:int, iy:int, iz:int, state: dict)`. `families`는 unet 파일 → `"2.9b"|"base"|"3.8b"` (Task 8이 채움).
  규칙(메모 §2·§3): 칸 상태의 프롬프트는 `effective_prompt` 후 S/R(대소문자 구분, 프롬프트+네거티브, X→Y→Z 순), 그 칸의 `loras`는 비움
  (태그로 옮겨졌으므로); 찾을 글이 없으면 `errors=['Prompt S/R did not find "<s>" in either prompt or negative prompt']`, 칸 0개.
  시드: keep_minus1 아니면 상태 seed −1 → `rng`로 하나 고정(모든 칸 공유), Seed 축의 −1은 칸마다 새로; vary는 메모 §3 식. 순서: cost 규칙(메모 §3).
  체크포인트 축 값들의 계열이 섞이면 `errors=["2.9B·base와 3.8B는 한 격자에 섞을 수 없음"]`. 격자 크기 가드: round(nx·ny·nz·W·H/1e6) ≥ 100 → 오류.
  `infotext_extra`: `Script: X/Y/Z plot`, `X Type`, `X Values`(원문), `Fixed X Values`(시드 축) …(메모 §4 끝).

- [ ] **Step 1: 실패 테스트** — `test_lora_epoch_by_strength_12_cells`(prompt `"1boy <lora:na-e10:1>"`; X=Prompt S/R `na-e10, na-e14, na-e18, na-e20`,
  Y=Prompt S/R `:1>, :0.9>, :1.1>` → 셀 12개, (ix=1,iy=2) 프롬프트 `"1boy <lora:na-e14:1.1>"`), `test_sr_searches_list_loras`(loras 목록에만 a →
  S/R 동작, 칸 상태의 loras==[]), `test_sr_missing_aborts`, `test_seed_fixed_once`(seed −1, rng seed 0 → 모든 칸 같은 값),
  `test_seed_axis_minus1_each_random`, `test_vary_seeds`(vary_x, nx 3 → base, base+1, base+2), `test_cost_order_checkpoint_outer`
  (X=Checkpoint, Y=CFG → 순서가 x 바깥, 메모 예시와 같음), `test_mixed_families_rejected`, `test_labels_format`(`CFG Scale: 4.0`,
  Prompt S/R은 값만, Checkpoint는 basename, Nothing은 빈 글), `test_grid_size_guard`.
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_xyz_plan -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 7: XYZ 격자 그리기

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/xyz_grid_draw.py`
- Create: `comfy_custom_nodes/ai_studio_forge_parity/fonts/Roboto-Regular.ttf` (Forge `modules/Roboto-Regular.ttf`에서 복사 — 글꼴 자체 라이선스
  Apache-2.0) + `LICENSES/Roboto-Apache-2.0.txt`, `THIRD_PARTY_NOTICES.md` 항목, `tests/test_comfy_node_pack.py`의 LICENSES 수 18 → 19
- Test: `tests/test_xyz_grid_draw.py`

**Interfaces:**
- Consumes: Task 6 `XyzPlan.labels`.
- Produces: `draw_xyz(cells: dict[tuple[int,int,int], PIL.Image.Image | None], nx:int, ny:int, nz:int, labels: dict[str,list[str]], *,
  draw_legend: bool, margin: int, row_count: int) -> tuple[PIL.Image.Image, list[PIL.Image.Image]]` (메인 격자, z별 하위 격자).
  메모 §4 그대로: 칸 상자 = 최대 w×h, 작은 그림 가운데, 배경 `#ffffff`, 글 `#000000`, 글꼴 크기 `(cell_w+cell_h)//25`, 줄 간격 `fs//2`,
  왼쪽 여백 `cell_w*3//4`(Y 글이 다 비면 0), 위 여백(X 머리 높이+2·줄 간격), 단어 단위 줄바꿈 + 넘치는 한 단어는 1px씩 줄임, margin은 X/Y 칸 사이만,
  Z 하위 격자는 한 줄로 나란히 + Z 제목(margin 0), Row Count는 X만 있을 때(0=auto → int(sqrt(nx))). `None` 칸 = 첫 칸 크기의 검은 그림.
  글꼴: 한 줄의 글자가 모두 Roboto에 있으면 Roboto, 아니면 `C:\Windows\Fonts\malgun.ttf` → `NotoSansKR-VF.ttf` → PIL 기본 순.

- [ ] **Step 1: 실패 테스트** — `test_size_with_legend`(64×64 칸 4×3, X/Y 라벨 있음, margin 0 → 너비 64*4 + 64*3//4, 높이 64*3 + 위 여백>0),
  `test_no_legend_no_padding`(크기 = 64*4 × 64*3), `test_margin_between_cells_only`(margin 10 → 너비 +30, 높이 +20, 바깥엔 없음),
  `test_failed_cell_black`(None 칸의 가운데 픽셀 (0,0,0)), `test_z_subgrids_side_by_side`(nz 2 → 메인 너비 ≥ 하위 격자 2개 합),
  `test_row_count_x_only`(nx 9, Y/Z 없음, row_count 0 → 3줄), `test_korean_label_uses_korean_font`(라벨 "세기: 0.9" → 고른 글꼴 경로가 malgun/Noto).
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_xyz_grid_draw -v`
- [ ] **Step 3: 구현** — Forge 코드를 보지 말고 메모 §4 설명만으로 작성.
- [ ] **Step 4: 통과 확인** — 위 + 전체(특히 `TestPackCarriesNoForgeCode`)
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 8: 스튜디오 라우트 (선택지 · XYZ 계획/격자 · 폴더 열기 · 업로드 복사)

**Files:**
- Modify: `studio_routes.py`
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_choices.py` (순수 + `folder_paths` 읽기)
- Test: `tests/test_studio_choices.py`, `tests/test_studio_routes.py`(확장)

**Interfaces:**
- Consumes: Task 5·6·7, Task 4 `build_infotext`.
- Produces (모두 JSON, 실패 시 `{"error": "..."}` + 400):
  - `GET /aistudio/studio/choices?kind=<samplers|schedulers|unets|vaes|tes|loras|upscalers|sam3|dora_modes>` → `{"items":[…]}`;
    `sam3`는 ComfyUI `models/sam3` + `extra_model_paths.yaml`의 Forge base `models/sam3`에서 `sam3` 접두 파일만, 값은 절대 경로, 표시는 파일 이름.
    `unets`는 `{"items":[…], "families":{파일:"2.9b"|"base"|"3.8b"}}`(파일 이름 규칙: `3.8b`/`38` 포함 → 3.8b, `base` 포함 → base, 그 외 2.9b).
  - `POST /aistudio/studio/xyz/plan` 본문 `{state, axes, options}` → `XyzPlan`을 JSON으로(시드는 `random.Random()`).
  - `POST /aistudio/studio/xyz/grid` 본문 `{nx,ny,nz,labels,options,cells:[{ix,iy,iz,image:{filename,subfolder,type}|null}],infotext}` →
    `{"grid":{filename,subfolder:"AIStudio/xyz",type:"output"},"subgrids":[…]}` (`folder_paths`로 읽고 `xyz_grid_00001.png` 식 저장, `parameters`=infotext).
  - `POST /aistudio/studio/open_output` → Windows `os.startfile(output 폴더)`, `{"path": …}`.
  - `POST /aistudio/studio/to_input` 본문 `{filename,subfolder,type}` → output/temp 이미지를 input 폴더로 복사(이름 충돌 시 `_1`), `{"name": …}`
    (갤러리 → i2i/인페인트·후처리 입력용).
- 핸들러 본문은 순수 함수(`handle_plan(body) -> (status, dict)` 등)로 두고 aiohttp 래퍼는 얇게.

- [ ] **Step 1: 실패 테스트** — `test_sam3_choices_filter_prefix`(임시 폴더에 `sam3.pt`, `sam3.1_multiplex_fp16.safetensors`, `other.json` → 앞 둘만),
  `test_unet_families`, `test_plan_handler_roundtrip`(Task 6 예시 본문 → 셀 12개 JSON), `test_grid_handler_writes_png_with_parameters`
  (가짜 folder_paths 임시 폴더, 64×64 그림 2개 → 저장된 PNG의 `parameters` 청크 = 보낸 infotext), `test_to_input_copies_without_overwrite`,
  `test_plan_handler_bad_body_400`.
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_choices tests.test_studio_routes -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 위 + 전체
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 9: "Forge처럼" 맞추기 — 프롬프트 편집 문법, 노이즈, DoRA, 어텐션

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/studio_prompt_schedule.py` (순수), `studio_forge_parity.md`(조사 결과 표, 팩 문서)
- Modify: `studio_nodes.py` (+`AIStudioPromptEncode`), `README.md`, `tests/test_comfy_node_pack.py`(47 → 48)
- Test: `tests/test_studio_prompt_schedule.py`, `tests/test_studio_nodes_prompt.py`

**Interfaces:**
- Produces: `schedule_prompt(text: str, steps: int) -> list[tuple[int, str]]` — Forge 프롬프트 편집 결과: (끝 스텝, 그 구간 프롬프트) 목록.
  `[a:b:w]`(w<1 → 스텝 비율, w≥1 → 스텝 수), `[a::w]`, `[:b:w]`, `[a|b]`(스텝마다 번갈아), 중첩, `\[` 이스케이프 — Forge Neo
  `modules/prompt_parser.py`의 **동작**만 따름(코드 복사 금지). 노드 `AIStudioPromptEncode`: 입력 `clip CLIP, text STRING, steps INT,
  forge_syntax BOOLEAN=True`, 출력 `CONDITIONING` — 꺼지면 `CLIPTextEncode`와 같음, 켜면 구간마다 인코딩하고 `ConditioningSetTimestepRange`로
  묶어 합침(구간 1개면 그대로). 가중치 `(x:1.2)`는 ComfyUI Anima 토크나이저가 이미 Forge처럼 읽으므로 손대지 않는다.
- 조사 산출물(`studio_forge_parity.md`): ① Forge Neo의 초기 노이즈(랜덤 소스 기본값·ENSD·SDE 단계 노이즈)와 ComfyUI `prepare_noise` 비교,
  같게 할 수 있으면 `AIStudioPromptEncode`처럼 작은 노드/옵션으로, 못 하면 이유 ② BREAK가 Forge의 Anima(Qwen3) 경로에서 하는 일과 우리 처리
  ③ DoRA는 `ForgeNeoDoraInferenceMode`의 `"Forge/Comfy (stock)"`(=Forge 기본)으로 맞춤 ④ 어텐션은 ComfyUI 실행 옵션이라 실행 중에 못 바꿈 →
  제어판에서 회색 + 이유. 표의 "가능/불가"가 Task 10 제어판 스위치의 활성 여부를 정한다.

- [ ] **Step 1: 실패 테스트** — `[cat:dog:0.5]`·steps 20 → `[(10,"cat"),(20,"dog")]`; `[cat:dog:5]` → `[(5,"cat"),(20,"dog")]`;
  `[cat::0.25]` steps 8 → `[(2,"cat"),(8,"")]`; `[cat|dog]` steps 4 → `[(1,"cat"),(2,"dog"),(3,"cat"),(4,"dog")]`; 문법 없음 → `[(steps, text)]`;
  `\[x\]` → 글자 그대로; 노드 `test_disabled_equals_plain_encode`(가짜 clip 호출 1번), `test_two_segments_set_ranges`(가짜 cond 2개, 범위 0–0.5/0.5–1).
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_prompt_schedule tests.test_studio_nodes_prompt -v`
- [ ] **Step 3: 구현 + 조사** — 노이즈·BREAK는 Forge Neo 소스를 **읽기만** 하고(GPU 없이), CPU에서 작은 크기(64×64 latent) 노이즈 텐서를 비교할 수
  있으면 비교(테스트는 `@requires_torch`).
- [ ] **Step 4: 통과 확인** — 위 + 전체
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 10: UI 전용 노드 3개 + 역할 지도 + JS 순수 로직

**Files:**
- Modify: `studio_nodes.py` (+`AIStudioPanel`, `AIStudioResultView`, `AIStudioXYZView`: 입력은 숨김 STRING 위젯 하나(`state`/`view`/`xyz`), 출력 없음,
  OUTPUT_NODE 아님 — 그래프에 있어도 실행되지 않음; +`AIStudioSwitch`: 입력 `select INT 0..1`, `a "*"`(lazy), `b "*"`(lazy), 출력 `"*"` —
  `check_lazy_status`가 고른 쪽 이름만 돌려줘 안 쓰는 모델 경로(2.9B/base ↔ 3.8B)는 실행되지 않음), `README.md`, `tests/test_comfy_node_pack.py`(48 → 52)
- Create: `web/ai_studio/studio_roles.js` (역할 상수), `web/ai_studio/studio_logic.js` (window 안 쓰는 순수 함수)
- Create: `comfy_custom_nodes/ai_studio_forge_parity/js_tests/studio_logic.test.mjs`
- Modify: `frontend/package.json` (`"test:pack": "node --test \"../comfy_custom_nodes/ai_studio_forge_parity/js_tests/*.test.mjs\""`),
  `.claude/commands/verify.md`와 `.agents/skills/source-command-verify/` 사본(둘 다)에 `npm run test:pack` 추가
- Test: 위 `.test.mjs`, `tests/test_studio_nodes_ui.py`

**Interfaces:**
- Produces (JS, `studio_logic.js`):
  - `ROLES`(studio_roles.js): `unet, te, vae, unet38, qwen35, pos_encode38, neg_encode38, family_switch_model, family_switch_clip,
    family_switch_pos, family_switch_neg, shift, sched, lora_stack, dora, pos_encode, neg_encode, latent, i2i_image, sampler, hires, decode,
    upscale, sam3_mask, sam3_detailer, save, post_image, post_upscale, post_encode, post_hires, post_decode, post_sam3_mask, post_sam3_detailer, post_save`
    — 워크플로 노드는 `properties.aistudio_role`로 찾는다.
  - `applyStateToPrompt(apiPrompt: object, roleIds: Record<string,string>, state: object) -> object` (깊은 복사 후 입력 덮어쓰기:
    unet→`unet_name`, te→`clip_name`, vae→`vae_name`, shift→`shift`, lora_stack→`positive`/`negative`(effective prompt는 Python이 아니라
    여기서도 같은 규칙으로 붙임), sampler→`seed,steps,cfg,sampler_name,scheduler,denoise`, latent→`mode,width,height,batch_size,mask_blur`,
    pos_encode/neg_encode→`steps,forge_syntax`, family_switch_*→`select`(state.unet 계열이 3.8b면 1, 아니면 0), unet38→`unet_name`,
    i2i_image→`image`, hires→`enabled,scale_by,upscale_method,steps,denoise,seed,shift`, upscale→`enabled,model_name,scale_by`,
    sam3_*→`enabled,checkpoint,prompt,denoise`, dora→`enabled,mode`, save→`studio_state`(JSON 문자열),`seed`).
  - `stageModes(state) -> Record<role, 0|4>`: 쓰지 않는 단계(post_* 전체 등)를 4(BYPASS)로.
  - `postPrompt(apiPrompt, roleIds, record, action: "upscale"|"hires"|"sam3") -> {prompt, outputNodeIds:[roleIds.post_save]}`
    (record = 그 이미지를 만든 상태 + 이미지 ref; 해당 단계만 enabled).
  - `recordFromParameters(text: string) -> object|null` (A1111 `parameters` → 상태 일부: prompt, negative, seed, steps, cfg, size; 실패 시 null).
  - `xyzCellPrompts(basePrompt, roleIds, plan) -> Array<{cellIndex, prompt}>` (plan.order 순서).

- [ ] **Step 1: 실패 테스트(.test.mjs)** — `applyStateToPrompt sets sampler inputs`(기본 상태 → sampler 노드 입력 `steps 32, cfg 3, sampler_name "exp_heun_2_x0_sde"`),
  `does not mutate input`, `tab i2i sets latent mode img2img`, `inpaint sets mode inpaint`, `postPrompt hires enables only post_hires`,
  `recordFromParameters parses forge text`(Task 4 예시 글 → seed 42, steps 32), `recordFromParameters returns null on junk`,
  `xyzCellPrompts follows plan order`, `3.8b unet selects switch 1`. Python: `test_ui_nodes_have_no_outputs_and_not_output_nodes`,
  `test_switch_lazy_requests_only_selected`(select 1 → `check_lazy_status` == `["b"]`, 반환값 = b).
- [ ] **Step 2: 실패 확인** — `cd frontend && npm run test:pack` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `npm run test:pack` PASS, `run_tests.py` 전체 PASS
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 11: 화면(DOM) — 제어판 · 결과/갤러리/진행 · XYZ 패널 · 생성/정지

**Files:**
- Create: `web/ai_studio/panel_view.js`, `result_view.js`, `xyz_view.js`, `run_controller.js`, `studio.css`(JS가 `<link>`로 붙임)
- Modify: `web/ai_studio/studio_main.js` (`nodeCreated`에서 세 UI 노드에 `addDOMWidget` — 크기 고정·`resizable=false`·`hideOnZoom:false`)
- Test: `js_tests/run_controller.test.mjs` (컨트롤러의 상태 기계만, DOM 없이)

**Interfaces:**
- Consumes: Task 10 `studio_logic.js`, Task 8 라우트, ComfyUI `window.comfyAPI.app/api`(메모 `comfy_frontend_api.md`).
- Produces: 화면은 시안 C4 그대로(제어판 790px: 프롬프트 한 줄 전체 → [모델·샘플링 | LoRA | 후처리] → 탭 `t2i`/`i2i` → 엔진룸 펼치기/접기 버튼;
  결과 680px: 이미지 높이 꽉·원래 비율, 탭 미리보기/XYZ 격자, 진행(스텝 n/N·%·남은 시간), 갤러리 가운데 정렬, 아이콘 버튼 7개(순서·이름은
  Global Constraints); XYZ 패널: 축 3개(종류 드롭다운·값 여러 줄·목록형 축은 고르기 버튼), Forge 옵션 체크박스·여백·Row Count, 축 바꾸기, 예상 장수).
  `RunController`: `generate()`(단일: `applyStateToPrompt` → `api.queuePrompt`), `generateXyz()`(라우트 plan → 칸마다 큐 → `executed`로 이미지 모음 →
  라우트 grid → 결과 탭), `stop()`(`api.interrupt()` + 이 컨트롤러가 넣은 대기 항목 `api.deleteItem('queue', id)` → 남은 칸 null),
  `runPost(action, galleryItem)`(Task 10 `postPrompt`, 기록 없으면 `recordFromParameters`, 둘 다 없으면 짧은 안내), 진행은 `progress`/`executing`/
  `b_preview` 이벤트. 생성 버튼은 실행 중 "정지"로 바뀜. Ctrl+Enter는 core `Comfy.QueuePrompt`를 그대로 두고 `beforeQueued`(또는 동등 훅)에서
  `applyStateToPrompt`가 적용되게 한다. "→ i2i/→ 인페인트": `/to_input` 복사 → `i2i_image` 위젯 값 설정 → 탭 전환(생성 안 함), 인페인트는
  i2i 노드를 선택하고 `Comfy.MaskEditor.OpenMaskEditor` 실행. "폴더": `/open_output`. 엔진룸 접기: 엔진 그룹 안 노드 `collapse()` + 그룹 `resizeTo`.
  "Forge처럼" 스위치: Task 9 표에서 불가 항목은 회색 + 마우스 올리면 이유.

- [ ] **Step 1: 실패 테스트** — `xyz run collects images in plan order`(가짜 api: queue → executed 이벤트 순서 뒤섞임 → cells가 plan 위치에 맞게),
  `stop marks remaining cells null`, `failed cell becomes null`(execution_error), `post without record uses parameters`, `post without anything returns notice`.
- [ ] **Step 2: 실패 확인** — `cd frontend && npm run test:pack`
- [ ] **Step 3: 구현** — 버튼은 `<button>` + `aria-label`, 아이콘은 인라인 SVG(시안 C4와 같은 모양), 색은 시안 값.
- [ ] **Step 4: 통과 확인** — `npm run test:pack` PASS
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 12: SAM3.1 연결

**Files:**
- Modify: `comfy_custom_nodes/ai_studio_forge_parity/sam3_nodes.py` (`_invoke_sam3_loader` 뒤 `bundle["dtype"]`을 요청한 precision으로 되돌림 —
  easy-sam3가 파일 이름 "fp16"을 보고 fp16으로 바꾸는 것을 덮음)
- Test: `tests/test_studio_sam31.py`

**Interfaces:**
- Consumes: Task 8 `sam3` 선택지(절대 경로).
- Produces: `ForgeNeoSAM3Mask`/`Detailer`가 `sam3.1_multiplex_fp16.safetensors` 절대 경로로 동작(로딩 방식은 SAM3와 같음: `detector.*` 키 재배치,
  `strict=False`, 3.1 전용 키는 무시 — 메모 `sam31_loading.md`).

- [ ] **Step 1: 실패 테스트** — `test_precision_restored_after_loader`(가짜 easy-sam3 로더가 dtype fp16을 넣어도 요청 bf16이 남음),
  `test_sam31_path_accepted`(체크포인트 인자에 3.1 절대 경로 → 로더에 그대로 전달).
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_sam31 -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 위 + 전체
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 13: 워크플로 생성기와 예제 워크플로

**Files:**
- Create: `tools/build_anima_studio_workflow.py` (실행하면 아래 JSON을 다시 씀)
- Create: `comfy_custom_nodes/ai_studio_forge_parity/example_workflows/AI Studio Anima.json`
- Test: `tests/test_studio_workflow_json.py`

**Interfaces:**
- Consumes: Task 10 역할 목록, 모든 노드 클래스 이름.
- Produces: ComfyUI UI 워크플로(JSON v0.4). 배치(캔버스 좌표, 확대 1 기준): `AIStudioPanel` (0,0) 790×960, `AIStudioResultView` (806,0) 680×960,
  `AIStudioXYZView` (1502,0) 400×960. 엔진룸: y≥1040부터 그룹 ①모델 ②LoRA ③프롬프트 ④잠재·샘플링 ⑤Hires ⑥업스케일 ⑦SAM3 ⑧저장
  ⑨후처리(갤러리 버튼) 를 **위에서 아래로** 한 줄씩, 각 줄 안 노드는 왼→오. 노드마다 `properties.aistudio_role`. 연결은 Task 10 역할 표대로:
  unet/te/vae 로더 → `ForgeNeoModelSamplingShift` → `ForgeNeoExtraSchedulers` → `AIStudioLoraStack` → `ForgeNeoDoraInferenceMode` →
  `AIStudioPromptEncode`×2 → `ForgeNeoLatentInput`(i2i는 `LoadImage` role `i2i_image`) → `ForgeNeoKSamplerCNS` → `ForgeNeoHiresFix` →
  (①모델 줄 아래 ①b 3.8B 줄: `ForgeNeoAnima38V2Loader`(unet38) · `ForgeNeoAnimaQwen35Loader`(qwen35) · `ForgeNeoAnima38V2Prompt`×2 — 2.9B/base 경로와
  3.8B 경로는 `AIStudioSwitch` 4개(family_switch_model/clip/pos/neg)에서 하나로 합쳐져 샘플러로 간다; 3.8B 경로에는 Forge 프롬프트 편집 문법을
  적용하지 않는다 — 팩 3.8B 프롬프트 노드 동작 그대로) →
  `VAEDecode` → `AIStudioUpscaleImage` → `ForgeNeoSAM3Mask` → `ForgeNeoSAM3Detailer` → `AIStudioSaveForge`; 후처리 줄은 `LoadImage(post_image)` →
  `AIStudioUpscaleImage` → `VAEEncode` → `ForgeNeoHiresFix` → `VAEDecode` → `ForgeNeoSAM3Mask` → `ForgeNeoSAM3Detailer` → `AIStudioSaveForge`
  (모델·조건은 위 줄에서 받음). 위젯 기본값 = Task 2 `DEFAULT_STATE`.

- [ ] **Step 1: 실패 테스트** — `test_every_role_once`, `test_node_types_known`(팩 `NODE_CLASS_MAPPINGS` + core 이름 목록
  `UNETLoader, CLIPLoader, VAELoader, LoadImage, VAEDecode, VAEEncode`), `test_links_consistent`(링크 id·슬롯·형 일치),
  `test_both_families_reach_sampler`(2.9B 경로와 3.8B 경로가 모두 switch를 거쳐 sampler의 model/positive/negative에 닿음),
  `test_engine_rows_top_to_bottom`(그룹 ①~⑨의 y가 커지는 순서, 같은 줄 노드 x 증가), `test_links_flow_one_way`(엔진룸 링크는 항상 같은 줄 왼→오
  또는 위 줄 → 아래 줄), `test_ui_nodes_layout`(세 UI 노드 위치·크기 위 값), `test_builder_reproduces_file`(생성기 출력 == 저장된 JSON).
- [ ] **Step 2: 실패 확인** — `venv\Scripts\python.exe -m unittest tests.test_studio_workflow_json -v`
- [ ] **Step 3: 구현** — `venv\Scripts\python.exe tools\build_anima_studio_workflow.py`로 JSON 생성
- [ ] **Step 4: 통과 확인** — 위 + 전체
- [ ] **Step 5: (사용자 승인 시) 커밋**

### Task 14: 팩 1.10.0 마무리 · 설치 · 확인

**Files:**
- Modify: `__init__.py` `__version__ = "1.10.0"`, `core/comfy_node_pack.py` `PACK_VERSION = "1.10.0"`(+변경 주석), `README.md` "## 1.10.0 changes",
  `THIRD_PARTY_NOTICES.md`, 버전을 박아 둔 테스트(`test_comfy_moss_spectrum.py:341`, `test_comfy_dora_infer_mode.py:1060`) 갱신

- [ ] **Step 1: 전체 검증** — `/verify` (venv `run_tests.py` 전체, 바뀐 Python `py_compile`, `cd frontend && npm run test && npm run test:node && npm run test:pack
  && npm run type-check && npm run build`) → 모두 통과
- [ ] **Step 2: 설치** — 앱 설치기(`core.comfy_node_pack.install_bundled_node_pack`)로 `C:\ComfyUI_windows_portable\ComfyUI\custom_nodes`에 설치 →
  `.aistudio-owned.json` 버전 1.10.0 확인. 사용자 ComfyUI가 떠 있으면 재시작은 **사용자에게 묻는다**.
- [ ] **Step 3: GPU 없이 확인** — CPU 전용 ComfyUI(`CUDA_VISIBLE_DEVICES=-1`, `--cpu`, 다른 포트 예: 8199)로 띄우고 헤드리스 Chrome(CDP)으로:
  예제 워크플로 열기 → 빠진 노드 0, 세 UI 노드 DOM 표시, `/aistudio/studio/choices?kind=sam3`에 3.1 포함, XYZ plan 라우트 12칸, 화면 스크린샷을
  시안 C4와 나란히 보고. (메모리 `reference_cdp_real_app_verification` 방식)
- [ ] **Step 4: GPU 확인 (사용자 허락 후, VRAM 약 12–20GB 예상을 밝힌다)** — 사용자 ComfyUI에서: t2i 1장, Hires/업스케일/SAM3(3과 3.1) 각각,
  i2i·인페인트 1장, 갤러리 버튼 7개, XYZ(LoRA 에폭 4 × 세기 3 = 12칸 격자), "Forge처럼" 켠 상태로 Forge와 같은 시드·설정 1장씩 나란히 비교 시트.
- [ ] **Step 5: 사용자에게 결과 보고 + (사용자 승인 시) 커밋·푸시**
