# UR_IV × sam-extra v0.35.1 계약 감사 (2026-10-25)

작업 트리: `/home/user/UR_IV-samx` (브랜치 `cloud/samx-0351-audit`, origin/main `5ace02c` 에서 분기).
확장 소스: `git clone --branch v0.35.1 --depth 1 https://github.com/UR-al/sam-extra` →
`/tmp/claude-0/-home-user/2ac51b6c-7b0c-51be-94a7-3ade1227a478/scratchpad/samx/sam-extra-0.35.1`
(태그 v0.35.1 = master 머지 `e29cfa1748372fa6fbb4e769fcee3801b146e25a`, `sam3ext/__version__.py` = `0.35.1`,
비교용으로 태그 v0.34.1 = `c3d6f89…`(레지스트리의 이전 EXT_COMMIT_AUDITED)도 받아 `git diff v0.34.1 v0.35.1` 로 봤다 — 읽기 전용).
라이브 Forge 는 없다. 설치된 확장은 `AISTUDIO_FORGE_EXTENSION_DIR` 로 이 클론을 가리켜 `AISTUDIO_REQUIRE_FORGE_EXT=1` 강제 모드로 돌렸다.

커밋: `cloud/samx-0351-audit` 에 둘(푸시하지 않음):
- `1e54372` feat: sam-extra 0.35.1 동기화 — TSM LoRA 보류 분류·가벼운 LoRA 목록 핀·픽스처 지연 표시(fixture_pending) — `core/sam_extra_contract.py`·`core/alwayson_propagation.py`·`tests/test_sam_extra_contract.py`
- `197ac62` docs: sam-extra 0.35.1 동기화 기록 — fixture_pending 규칙·Forge 2.31 XYZ 축·DoRA 오프셋 패치 메모 — `docs/sam_extra_sync.md`

## 1. 바뀐 것 (파일:줄)

### core/sam_extra_contract.py (레지스트리)
- `:40` `EXT_VERSION_AUDITED = "0.35.1"` (0.34.1 →).
- `:178-196` 감사 머리 주석에 2026-10-25 v0.35.0/v0.35.1 단락(무엇이 바뀌었고 어떻게 분류했는지).
- `:197` `EXT_COMMIT_AUDITED = "e29cfa17…"` (태그 v0.35.1 머지 커밋).
- `:198-202` **새 상수 `EXT_FIXTURE_VERSION = "0.34.1"`** — 픽스처를 받은 확장 버전. 감사 버전보다 뒤처진 상태를 명시한다(아래 3절).
- `:21-24` 모듈 설명에 `optional` 사용 현황(지금 없음)과 `fixture_pending` 규칙 단락.
- `:226`, `:241-242`, `:248` `_script(... fixture_pending: str = "")` — 픽스처에 아직 없는 이유를 적는 칸(테스트가 읽는다).
- `:987-1005` **SCRIPTS['TSM LoRA']** 신규 — `scripts/lora_tsm.py`, `positional_or_dict`, live_argc 6, `ui_return`/`arg_names` =
  (enabled, lora, strength, stage, pooling, lokr_merge_cache), runtime_choices (1,), `shape=""` + `fixture_pending=…`,
  classification `deferred(HOLD, …)` — 앱에 TSM 카드·블록 없음.
- `:1224-1225` XYZ 머리 주석 집계(170개), `:1311-1315` **XYZ_AXES['[TSM]']** 신규 — labels (Router pooling, Stage, Strength), `deferred("P18", …)`.
- `:1171-1183` ROUTES['GET /sam3-lora/lite'] — `optional=True` 제거, 주석에 응답 모양·401/503 폴백 메모.
- `:1398-1401` MODULES['scripts/lora_manager.py'] 사유에 lite 라우트 등록 추가, **MODULES['scripts/lora_tsm.py']** 신규 `deferred(HOLD, …)`.
- `:1554-1558` MODULES['sam3ext/dora_infer_mode.py'] gaps 에 v0.35.1 오프셋 패치 수정과 팩 1.9.0 사본 미반영 메모.
- `:1562-1565` MODULES['sam3ext/lora_lite.py'] — `optional=True` 제거, 핀 참조 메모.
- `:1682-1684` MODULES['sam3ext/unload_registry.py'] 사유에 TSM LoRA 훅, `:1687-1688` **MODULES['sam3ext/tsm_lora/']** 신규 `deferred(HOLD, …)`.
- `:2570-2582` **SEMANTIC_PINS** `lora_lite_path`(`LORA_LITE_PATH` = `/sam3-lora/lite` ↔ `backends.webui_backend:_LORA_LITE_PATH`),
  `lora_lite_version`(`LITE_VERSION` = 1 ↔ `backends.webui_backend:_LORA_LITE_VERSION`), `lora_lite_tag_frequency_top`(`TAG_FREQUENCY_TOP` = 32, 앱 상수 없음 — 앱은 앞 5개만 씀).

### core/alwayson_propagation.py (앱 코드 — 이번 감사의 유일한 앱 코드 변경)
- `:263-265` `NEVER["TSM LoRA"]` — `tests/test_alwayson_propagation.ClassificationTests` 가 레지스트리의 모든 제목을 PROPAGATION/NEVER 중 한 곳에 요구한다.
  앱이 TSM 블록을 만들지 않으므로 NEVER(사유 포함). 페이로드·UI 변화 없음.

### tests/test_sam_extra_contract.py
- `:16-17` 모듈 설명에 픽스처 지연 규칙.
- `:152-153` `_version_tuple` 도우미.
- `:299-315` `TestRegistry.test_fixture_pending_marks_match_the_fixture_lag` — `fixture_pending` ⇔ (`EXT_FIXTURE_VERSION` ≠ `EXT_VERSION_AUDITED`),
  표시된 스크립트는 script_info=True·shape ''·ui_return 튜플이어야 하고, 뒤처진 판은 버전 순서가 맞아야 한다.
- `:317-328` `TestRegistry.test_lora_lite_is_no_longer_optional_and_pinned` — optional 제거·핀 셋·앱 참조·`_TRIGGERS_FROM_TAGS(5) ≤ 32`.
- `:417-424` `test_fixture_is_for_audited_version` — 픽스처 `ext_version` 을 `EXT_FIXTURE_VERSION` 과 비교(이전엔 EXT_VERSION_AUDITED).
- `:426-430` `test_fixture_scripts_match_registry` — `fixture_pending` 스크립트를 등록 집합에서 뺀다.
- `:432-436` `test_fixture_pending_scripts_are_not_in_the_fixture_yet` — 표시된 스크립트가 픽스처에 나타나면 실패(낡은 표시).
- `:446`, `:460` `test_live_arg_counts`·`test_arg_shapes` — `fixture_pending` 스크립트 건너뜀.

### docs/sam_extra_sync.md
- `:18` 범위 단락 집계(스크립트 18개/픽스처 17개, XYZ 170개, lite 핀).
- `:39` `optional` 규칙 갱신(지금 쓰는 항목 없음, lite 핀 걸었음), `:40` **`fixture_pending` 규칙** 신규.
- `:46` 체크리스트 1번에 v0.35.1 커밋, `:60` 2번에 'Forge 도 저장 응답도 없을 때' 절차, `:92` 5번에 `EXT_FIXTURE_VERSION` 갱신.
- `:217-228` **'sam-extra 0.35.1 동기화(2026-10-25)' 절** — TSM LoRA·lite·2.31 XYZ 축·DoRA·검증·확인 못 한 것.

앱 코드(backends/core 런타임·프론트)는 바꾸지 않았다 — 확장 계약 중 앱이 쓰는 것(라우트·옵션·Sam3Args·XYZ 라벨·infotext 키·핀 상수)이 0.34.1 → 0.35.1 에서 하나도 달라지지 않았기 때문이다. 프론트 변경이 없어 npm 빌드·frontend_dist 변경도 없다.

## 2. v0.35.1 에서 달라진 확장 항목과 앱 영향

`git diff --stat v0.34.1 v0.35.1 -- scripts sam3ext preload.py install.py metadata.ini javascript` = 25 파일(+4901/−68). 계약 테스트(강제 모드) 첫 실행의 실패는 정확히 넷 — 버전 게이트, 새 스크립트 제목 'TSM LoRA', 새 모듈 단위 `scripts/lora_tsm.py`·`sam3ext/tsm_lora/`, 새 XYZ 라벨 `[TSM]` 셋. 그 밖(옵션 키·라우트·플래그·Gradio 이름 엔드포인트·Sam3Args 필드/기본값/범위·process() 가 읽는 state 키·모든 의미 핀·소스 ↔ 픽스처 ui() 비교)은 0.35.1 소스에서 그대로 통과했고 스캐너의 unresolved 도 없었다.

| 항목 | v0.35.x | 앱 영향 / 처리 |
|---|---|---|
| **TSM LoRA**(v0.35.0, 실험) — always-on 'TSM LoRA', `scripts/lora_tsm.py`, 인자 6(`enabled, lora, strength, stage, pooling, lokr_merge_cache`; dict 또는 위치 인자, `coerce_args`), 강도 −2~3 step 0.05, 단계/풀링 한국어 라디오, XYZ `[TSM] Stage`(auto·1·2·off)·`Router pooling`(sample·batch)·`Strength` cost 0.2, infotext `TSM LoRA`·`TSM LoRA info`·`TSM stage`·`TSM pooling`·`TSM LoKr merge cache`, 새 패키지 `sam3ext/tsm_lora/`(8 모듈), `unload_registry` 훅 `tsm_lora`, `layout_lanes` 자리 `lora-tsm` | 앱에 카드·블록 없음 → SCRIPTS·MODULES `deferred(HOLD)`, XYZ `P18`, 보조 패스 `NEVER`. **동작 영향**: 확장이 `networks.load_lora_for_models` 를 감싸므로 사용자가 프롬프트 `<lora:…>` 에 TSM 파일을 쓰면 앱 요청에서도 적용된다(전에는 Forge 가 'LoRA mismatch' 로 건너뜀) — 앱이 바꿀 것은 없고 TSM 파일 없는 생성은 그대로(CHANGELOG). TSM 파일은 앱 LoRA 목록(lite/full)에도 보통 LoRA 로 보인다. |
| **가벼운 LoRA 목록 `GET /sam3-lora/lite`**(v0.35.0) — `sam3ext/lora_lite.py`, `scripts/lora_manager.py` 가 app_started 에서 등록, 로그인 가드만(X-SAM3-Notebook 없음), 503(내장 LoRA 지원 없음)/401(로그인 없음) | 앱 `5ace02c` 가 이미 쓴다(`backends/webui_backend.get_loras` — lite 먼저, 실패 시 `/sdapi/v1/loras`). 설치본에 들어왔으므로 문서 규칙대로 `optional=True` 제거 + 핀 `LORA_LITE_PATH`·`LITE_VERSION`·`TAG_FREQUENCY_TOP`. 앱 상수 `_LORA_LITE_PATH`·`_LORA_LITE_VERSION` 과 값이 같음(핀 테스트). |
| **XYZ `[Modules] Text Encoder`·바꿔 끼운 `VAE` 축 — Forge Neo 2.31 대응**(v0.35.1 머리 변경) — `scripts/modules_xyz_text_encoder.py`(diffstat 168줄), `sam3ext/xyz_text_encoder.py`·`xyz_text_encoder_cell.py`·`xyz_vae.py`·`xyz_vae_cell.py`: 2.31 의 기본 `VAE`·`Checkpoint name` 축이 공유 `override_settings` 에 남긴 목록·체크포인트 때문에 앞 칸의 VAE 로 그리거나 그리드가 멈추던 것을 고침; Qwen3.5 9B(MOSS) 자리 추가 | **축 라벨·자리·값 후보·cost 그대로**(CHANGELOG 명시, `test_xyz_axis_labels` 통과). 앱이 아는 라벨 `core/xyz_capabilities.FORGE_TEXT_ENCODER_AXIS = "[Modules] Text Encoder"` 그대로. 앱 XYZ 는 Forge XYZ 스크립트를 부르지 않고(코드에 `xyz_grid`/`x_type` 사용 없음) 칸마다 따로 생성하며 모듈은 `/sdapi/v1/options` 로 바꾸므로 **영향 없음**. 앱이 보내는 XYZ 관련 축 라벨은 이것 하나(제외 목록용)뿐이다. Forge 2.31 자체의 변화(새 `VAE ("state_dict" only)` 축으로 축 번호 밀림, Checkpoint cost 2.0)도 앱과 무관. |
| **DoRA 추론 방식 — 오프셋 패치 DoRA 누락 수정**(v0.35.1) — `sam3ext/dora_infer_mode.py` `_is_partial_view`, `decompose(..., return_fp32=False)` 새 kwarg, `_can_return_fp32` 가 `_base` 대신 저장소 시작 위치·크기로 뷰 판정 | 인자·모드 키·라벨·infotext 그대로 → 앱 `core/dora_infer_mode` 변화 없음. **단, 앱 노드 팩 1.9.0 의 옮긴 사본** `comfy_custom_nodes/ai_studio_forge_parity/dora_infer_mode.py:288` 은 여전히 `getattr(weight, "_base", None) is None` 로만 뷰를 가린다 — ComfyUI 도 `torch.inference_mode` 안에서 패치하면 합쳐진 qkv 등 오프셋 패치의 DoRA 가 같은 식으로 빠질 수 있다(미확인). 코드는 고치지 않고 MODULES gaps 에 적었다(열린 질문). `tests/test_comfy_dora_infer_mode.py` 의 뷰 대조는 inference_mode 밖에서 만든 뷰(`_base` 있음)라 0.35.1 확장과 여전히 일치한다. |
| MCP 서버 `moss`·`qwen21` 프리셋, `mcp_server/sam_extra_mcp/forgeneo/*` | 스캐너·레지스트리 단위 밖(Forge 가 불러오지 않는 별도 uv 프로젝트). 영향 없음. |
| `sam3ext/ui_vae_degrid.py` `HELP_EXTRAS` 문구(Forge 2.31 부터 Forge 자체 DeGrid 기본 끔) | 도움말 문자열 — 핀 없음, 영향 없음. |
| `scripts/!sam3.py`·`scripts/anima_safe_pag.py`·`scripts/dora_infer_mode.py`·`sam3ext/notebook_store.py`·`xyz_vae_cell.py` | 주석·docstring 만. |
| `preload.py`·`install.py`·`metadata.ini`·`javascript/*` | diff 없음. |

## 3. 확인 못 한 것

1. **픽스처(`tests/fixtures/sam_extra_script_info.json`)** — 라이브 `GET /sdapi/v1/script-info` 가 필요해 0.35.1 로 못 받았다(0.34.1, 2026-10-13 저장본 그대로). 그래서
   - 'TSM LoRA' 의 script-info `shape` 해시·라이브 기본값·모드 항목 비교(`test_live_arg_counts`·`test_arg_shapes`)와 `test_fixture_scripts_match_registry` 의 양방향 비교가 이 스크립트에 대해 없다. 대신 소스 AST 로 인자 6개·ui() 반환 순서·ARG_NAMES·라벨·기본값·범위·선택지를 읽어 레지스트리에 적었고(`test_ui_return_order`·`test_arg_names`·`test_registered_ui_returns_have_live_arg_count`), Forge 의 실제 script-info 형식(값 타입·runtime 선택지)은 받아야 안다.
   - 이 상태를 레지스트리가 `EXT_FIXTURE_VERSION = "0.34.1"` 과 `fixture_pending` 으로 명시하고, 테스트가 (a) 픽스처 버전은 EXT_FIXTURE_VERSION 과 같아야 하고, (b) 두 버전이 같아지면 `fixture_pending` 이 하나도 남으면 안 되고, (c) 표시된 스크립트가 픽스처에 나타나면 실패하도록 지킨다. **소유자 작업**: Forge 를 sam-extra 0.35.1 로 다시 띄운 뒤 `venv\Scripts\python.exe tools\refresh_sam_extra_fixture.py` → `fixture_pending` 제거 → `test_arg_shapes` 가 알려 주는 해시로 `shape` 채움 → `EXT_FIXTURE_VERSION = "0.35.1"`.
   - 기존 17개 스크립트는 0.35.1 소스의 ui() 와 0.34.1 픽스처가 라벨·기본값·범위·step·선택지까지 모두 같았다(`test_fixture_matches_installed_source` 통과) — CHANGELOG 의 '기존 인자 그대로' 와 일치.
2. **GPU 대조(체크리스트 7번)** — 하지 않았다(GPU 없음). 픽셀이 바뀌는 항목: DoRA 오프셋 패치 수정(해당 LoRA 를 LyCORIS·Forge fp32 모드로 쓴 생성만), Forge 2.31 자체 변화.
3. **팩 1.9.0 DoRA 사본의 inference_mode 뷰 누락 여부** — ComfyUI 런타임이 없어 재현하지 못했다(위 표).
4. **Forge 체크아웃이 필요한 오라클 테스트**(`forge_root()` — k_diffusion·sd_schedulers·`tests/_forge_k_diffusion.py` 등)와 **ComfyUI 런타임 테스트**(`AISTUDIO_COMFY_TEST_ROOT`) — 이 환경에 없어 skip.
5. **사용자 PC 의 설치본** — 설치된 sam-extra 가 0.35.1 이 아니면 `test_version_gate` 가 실패한다(확장을 올리면 끝). 이전 동기화와 같은 동작.
6. **프론트** — 변경이 없어 vitest/type-check/build 를 돌리지 않았다(`frontend/src` 에 0.34.1 하드코딩은 얼굴 crop 기능 버전 `FACE_CROP_GROWTH_VERSION` 뿐이고 그 의미는 그대로).

## 4. 테스트 결과

- 계약 테스트 첫 실행(수정 전, 시스템 python3 3.13 — 이 모듈은 순수 의존성만): `Ran 75 tests … FAILED (failures=4)` — 위 2절의 네 가지(버전 게이트·TSM 제목·모듈 2·XYZ 라벨 3).
- 수정 후, venv 강제 모드(`AISTUDIO_FORGE_EXTENSION_DIR=<클론> AISTUDIO_REQUIRE_FORGE_EXT=1 QT_QPA_PLATFORM=offscreen`):
  `tests.test_sam_extra_contract tests.test_alwayson_propagation tests.test_webui_lora_lite` → `Ran 149 tests … OK`.
  `tests.test_sam3_args tests.test_anima_guidance tests.test_sam_extra_notices tests.test_tile_repair_request` → 시스템 python 에서는 PyQt6 import 오류 8개만(확장 소스 검사는 모두 통과), venv 전체 실행에 포함.
- **전체 `venv/bin/python run_tests.py`(--quick 아님), 클론을 설치된 확장으로(`AISTUDIO_FORGE_EXTENSION_DIR` + `AISTUDIO_REQUIRE_FORGE_EXT=1`)**:
  `Ran 6622 tests in 174.5s — FAILED (failures=21, errors=7, skipped=151)`, EXIT 1 (로그 `run2_with_ext.log`).
  실패·오류 28건은 **모두 이 변경과 무관**하다: 같은 venv 로 pristine `5ace02c`(`git archive` 로 뺀 사본 `baseline-5ace02c/`)에서 같은 테스트 id 를 다시 돌리면 같은 28건이 그대로 실패한다(`patches/compare_baseline.py` — backend_runtime 밖 22건: baseline `FAILED (failures=18, errors=4)` 로 전부 재현, backend_runtime 6건: 아래 별도 확인). 원인은 Linux 환경이다 — Windows 경로 기대(`C:\sd-webui-forge-neo`, 시스템 폴더 규칙, `user_data/models/Stable-diffusion`), 대소문자 구분 파일 시스템(`Face_Yolov8n.PT`), `file://` 퍼센트 경로 규칙, 한글 글꼴 없음(두부 글리프), 파일 이름 길이(이모지 json), torch 2.14 CPU 의 골든 허용 오차(상관 2.2e-6 > 2e-6), 강제 모드에서 Forge 체크아웃이 없어 `test_preset_choice.InstalledForgeScheduleLabelTests` 가 `self.fail`(`<ext>/../../modules/sd_schedulers.py` 없음), `test_comfy_node_pack.TestSamExtraSharedCodeIsListed.test_every_near_copy_is_named_next_to_its_module`(팩 함수 4개 — extra_schedulers `_refuse_flat_step`·`_refuse_unusable_flow_cosmos`·`laplace`, guidance_s2 `draw_blocks` — 가 설치된 확장과 0.80 이상 닮았는데 NOTICES 에 이름이 없다 — 해당 확장 함수는 0.34.1 → 0.35.1 diff 에 없어 판과 무관, baseline 도 같은 실패). 계약 관련 모듈(`test_sam_extra_contract`·`test_alwayson_propagation`·`test_webui_lora_lite`·`test_sam3_args`·`test_anima_guidance`·`test_sam_extra_notices`·`test_tile_repair_request`·`test_comfy_dora_infer_mode`·`test_dora_infer_mode`)의 FAIL/ERROR 는 0.
  (참고: libEGL 설치 전 첫 실행은 `Ran 5611 … FAILED (failures=19, errors=178, skipped=153)` — 오류 164건이 `libEGL.so.1` 없음과 그에 따른 QtWebEngine 초기화 실패, 테스트 모듈 74개가 import 단계에서 빠졌다. `apt-get install libegl1` 뒤 다시 돌린 것이 위 결과다.)
- **전체 `venv/bin/python run_tests.py`, 확장 포인터 없이(skip 동작 확인)**: `Ran 6709 tests in 199.9s — FAILED (failures=18, errors=7, skipped=514)`, EXIT 1 (로그 `run2_without_ext.log`). 건너뜀 514 중 439 가 `sam-extra 확장 미설치 — 찾아본 곳: <앱>/user_data/managed_backends/forge/data/extensions/forge_sam3_extension, …/sam-extra, …/shared/extensions/…, extensions/forge_sam3_extension, extensions/sam-extra (다른 위치는 AISTUDIO_FORGE_EXTENSION_DIR 로 지정)` — 설치 위치 후보를 다 적은 이유가 그대로 나온다. 계약 테스트(`test_sam_extra_contract`·`test_alwayson_propagation`·`test_webui_lora_lite`)는 ok 77 · skipped 16(설치된 확장 AST 검사 클래스) · 실패 0 — 레지스트리 형식·픽스처·`fixture_pending` 일관성 테스트는 확장 없이도 돈다. 실패·오류 25건(id 24개 — `test_install_source_switches_compute_the_result_snapshot_once` 는 FAIL 과 ERROR 둘)은 클론을 가리킨 실행과 같은 환경 실패 집합이고(강제 모드 전용 `test_preset_choice` 1건과 `test_comfy_node_pack` 근사 복사 1건, `test_sam3_face_crop.InstalledExtensionTests` 1건은 확장이 없어 빠짐), 24개 id 를 pristine `5ace02c` 에서 같은 venv 로 다시 돌리면 `FAILED (failures=18, errors=7)` 로 전부 재현된다 — **회귀 후보 0**.
- skip 사유 요약(클론을 가리킨 실행 151건): `Set AISTUDIO_COMFY_TEST_ROOT …`(실제 ComfyUI 런타임 계층) 22+16+14+12+9+8+8+4+3+2+2+1 = 101, `Forge 의 modules_forge/packages/k_diffusion/sampling.py 를 찾지 못함 — FORGE_ROOT=<클론의 두 단계 위>`(Forge 체크아웃 없음, `SAM3_FORGE_ROOT` 로 지정 가능) 13, Windows 전용 동작(경로 표기·시스템 폴더·정션·Job Object·named mutex·SO_REUSEADDR·대소문자 NTFS 기대값·읽기 전용 삭제) 약 20, 글꼴 없음(일본 한자·Arial) 2, GPU/NVML 없음 1, PyAV 없는 경로 1, 대소문자 가리지 않는 FS 1. 확장 없이 돌린 실행은 위에 439(`sam-extra 확장 미설치`)가 더해지고 k_diffusion 오라클 skip 이 3 으로 준다(`FORGE_ROOT=None`).
- `venv/bin/python -m py_compile core/sam_extra_contract.py core/alwayson_propagation.py tests/test_sam_extra_contract.py` 통과.

## 5. venv 설정 메모 (Linux, 이 컨테이너)

- 시스템 `python3` 는 3.13.17 이고 pandas/PIL/PyQt6 가 없다. `python3 -m venv /home/user/UR_IV-samx/venv` → `venv/bin/pip install --upgrade pip wheel` → `venv/bin/pip install -r requirements.txt` (약 10분, exit 0). `venv/` 는 .gitignore 에 있다.
- 설치된 주요 판: PyQt6 6.11.0 · PyQt6-WebEngine 6.11.0 · pandas 3.1.0 · Pillow 13.0.0 · torch 2.14.1+cu130(ultralytics 의존성으로 PyPI CPU/CUDA 휠 — CUDA 라이브러리까지 내려와 용량이 크다) · opencv 5.0.0 · onnxruntime 1.31.0 · sam3 0.1.4 · rembg 2.0.86 · av 19.0.2. 모두 import 됨(torch 는 CPU 로 돈다 — `CUDA is not available` 경고).
- `run_tests.py` 는 POSIX 경로 `venv/bin/python` 도 찾는다(`venv_python()`); 시스템 python 으로 불러도 venv 로 재실행된다. 여기서는 `venv/bin/python run_tests.py` 로 직접 불렀다. PyQt6 는 `QT_QPA_PLATFORM=offscreen` 이 필요하다.
- 확장 위치는 `AISTUDIO_FORGE_EXTENSION_DIR` 하나만 지정하면 그 경로만 본다(`core/sam_extra_scan.find_installed_extension`). 강제 모드 `AISTUDIO_REQUIRE_FORGE_EXT=1`. Forge 체크아웃을 가리키려면 `SAM3_FORGE_ROOT`(없어서 오라클 테스트 skip).

## 6. 열린 질문 (소유자)

1. 팩 1.9.0 `ForgeNeoDoraInferenceMode` 의 `_can_return_fp32`(`comfy_custom_nodes/ai_studio_forge_parity/dora_infer_mode.py:288`)를 sam-extra 0.35.1 처럼 `_is_partial_view`(저장소 기준) + '늘 제자리에도 쓰기'로 고칠지 — ComfyUI 가 LoRA 패치를 `torch.inference_mode` 안에서 하는지에 따라 합쳐진 qkv·diffusers q/k/v·gate_up 의 DoRA 가 빠질 수 있다. 고치면 팩 버전 올림 + `tests/test_comfy_dora_infer_mode.py` 뷰 대조를 inference_mode 안 뷰로 확장.
2. TSM LoRA 를 앱에 노출할지(현재 HOLD) — 노출하면 `_script` 의 app_spec·카드·블록·`alwayson_propagation` 행·XYZ 축을 함께 만든다(docs/sam_extra_sync.md '새 always-on 스크립트를 앱에 노출할 때').
3. 픽스처 갱신 — Forge 0.35.1 을 띄울 수 있을 때 위 3-1 절차.
