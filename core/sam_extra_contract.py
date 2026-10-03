"""sam-extra(forge_sam3_extension) 계약 레지스트리 — 앱이 추적해야 하는 확장의 모든 노출면.

순수 데이터(Qt·네트워크·확장 import 없음). 확장이 드러내는 항목은 하나도 빠짐없이 셋 중 하나로 분류한다.

- ``mapped``   앱이 쓴다. ``app`` 에 ``"파일:심볼"`` 을 적는다(심볼 없이 파일만 적어도 된다).
               계약 테스트가 그 파일과 심볼이 앱에 실제로 있는지 본다 — 앱 코드를 옮기면 여기도 고친다.
               ``gaps`` 에는 알려진 빈틈을 작업 패키지 id 와 함께 적는다(gap matrix P1-P21).
- ``ignored``  앱과 무관하다(Forge 화면 전용, 해당 없음). ``reason`` 필수.
- ``deferred`` 아직 앱에 없다. ``package``(P1-P21, 보류는 HOLD)와 ``reason`` 필수.

`tests/test_sam_extra_contract.py` 가 설치된 확장(AST, `core.sam_extra_scan`)과 저장된 script-info
픽스처(`tests/fixtures/sam_extra_script_info.json`, `tools/refresh_sam_extra_fixture.py`)를 이 표와
양방향으로 비교한다. 분류되지 않은 새 항목도, 등록됐는데 사라진 항목도 실패다. 확장이 업데이트되면
`docs/sam_extra_sync.md` 체크리스트를 따른다.

``optional=True`` 는 병렬로 들어오는 중인 항목에만 쓴다 — 확장에 아직 없거나 (mapped 면) 앱 참조가 아직
없어도 통과한다. 새 항목을 조용히 넘기는 뜻은 아니다(분류되지 않은 항목은 여전히 실패). 양쪽이 자리 잡으면
표시를 지운다 — 지금은 쓰는 항목이 없다(공유 메모 라우트·파일은 양쪽이 자리 잡아 표시를 지웠다).

AST 로 읽는 것과 못 읽는 것: 설치된 확장 소스에서 ui() 컴포넌트의 라벨·기본값·범위·step·선택지를 읽어
픽스처와 비교한다(소스가 바뀌었는데 픽스처가 낡으면 실패). 읽지 못하는 칸은 ``UI_UNREAD`` 에 사유와 함께 적는다.
gap matrix 5-(a) 가운데 아직 없는 검사: 5번 중 ``wire_tipo`` 입력 순서·``MODES``/``LENGTHS``·
``REFERENCE_ARG_KEYS``(P19·P20 이 Gradio 방식을 고를 때), 6번 앱 내부 커버리지(SAM3_KEYS → 설정·프록시·저장·
Vue 바인딩, 위치 spec 키 → components/guidance/*Section.vue), 7번 Comfy 검사(mapped 제목마다 컴파일러가 쓰거나 unsupported
선언). SEMANTIC_PINS 의 Comfy 미러는 핀마다 손으로 건다(``*_comfy`` — 팩 상수를 ``app`` 으로 보는 두 번째 핀, DD·DeGrid) — 빠진
미러를 자동으로 찾지는 않는다. Comfy 참조(``comfy=``)는 파일·심볼이 있는지만 본다. UI_ONLY_FEATURES 는
등록된 함수가 그 파일에 있는지만 보는 한 방향 검사다(새 Gradio 이름 엔드포인트는 잡지 않는다).
"""
from __future__ import annotations

from types import MappingProxyType

EXT_VERSION_AUDITED = "0.32.0"
# 감사 시점 HEAD — 원본 동등성 작업(861ac02..dd18876), 변경 기록(80d2dce), LoRA Manager 경로 인증(a2114b5), DAVE+DD 우회 토글(8878b9e),
# API 원본 기준 SAM3 sam3_source_image(3955d42 — scripts/!sam3.py 만, SAM3_REQUEST_ONLY_KEYS·SEMANTIC_PINS sam3_source_*),
# 원본으로 돌 때 Forge img2img 색 보정 끄기(0059da8 — scripts/!sam3.py 만, 새 폴백 이유 'color correction', 계약 키·상수 그대로)까지.
# 작업 트리는 깨끗했다. v0.30.0 은 아직 릴리스 전이라 같은 버전 문자열 안에서 코드가 바뀌었다(818b8fe 도 0.30.0).
# 2026-09-30 이어서: VAE DeGrid(3a74dd8..1dd1a98 — SCRIPTS·OPTIONS·MODULES 를 HOLD 로 분류, 앱 노출은 사용자 결정 대기),
# 새 Forge 텍스트 엔진 대응(3e35f1e — sam3ext/anima38 내부, 인자·infotext 계약 그대로), NegPiP 내장(0f2a98a — SCRIPTS["NegPiP"]
# mapped·script_info=False, sam3ext/negpip/ ignored; 3.8B 가 Forge 표준 infotext "Emphasis" 를 엔진 규칙대로 남긴다),
# NegPiP 내부 수정(395854b — 제목·인자 0개·always-on·파일 이름 그대로, 앱 계약 변화 없음)까지.
# 2026-10-01 VAE DeGrid 앱 노출(확장은 그대로 395854b): SCRIPTS·MODULES 를 HOLD → mapped(Extras 판은 ignored — API 없음),
# 옵션 셋은 P10 라디오 덮어쓰기(OPTIONS mapped), 상수 계약은 SEMANTIC_PINS degrid_*(앱·Comfy 팩 거울).
# 2026-10-02 3d5d26d(Forge 2.29.2 Anima 엔진 속성 이름 대응 — 인자·infotext 계약 그대로) 위의 fe2a4e7:
# v0.30 디테일 묶음(PAG 인자 62→91 — S²·Adaptive SMC·TSR·Momentum·HiGS·HiFlow, 앱은 고정 칸으로 확장 기본값만 보냄)과
# 2026-10-02 검토 제안 편입(새 스크립트 'Anima Optimal Scale', 옵션 sam3_guidance_pag_cosine_envelope 기본 끔·
# sam3_builtin_negpip_enabled 기본 켬 — 셋 다 HOLD). 픽스처는 같은 날 Forge 2.29.2(46365871)를 이 트리로 다시 띄워
# 라이브로 받았다 — 학습 중에 소스 AST 로 먼저 맞춘 새 칸(PAG 62-90·Optimal Scale)과 값이 모두 같았다.
# 같은 날 v0.30.0 으로 릴리스했다 — master 089333b 는 fe2a4e7 을 합친 머지 커밋이고 트리가 같다.
# 2026-10-03 v0.30.1(991c45b, master 머지 1c49f89 — 트리 같음): [VERIFY] MG·HiGS 적용 횟수를 패스마다 새로 센다
# (sam3ext/guidance/runtime.py reset_pass 가 새 HistoryState). 인자·infotext·옵션·제목 그대로 — 픽스처는 고친 코드로 띄운
# Forge 2.29.2 에서 커밋 전에 다시 받았다(기록된 커밋은 그때 HEAD fe2a4e7, 버전은 로컬 0.30.1). scripts 부분은 0.30.0 때와 같았다.
# 2026-10-03 v0.31.0(6adeb83, master 머지 a65c4ae — 트리 같음): 새 기능 일곱 —
# Colorcraft(always-on 'Colorcraft (sam-extra)' 579 인자(v0.32.0 에서 67 — 아래)·옵션 둘·XYZ 14), Anima SPEED(14 인자·옵션 둘·XYZ 6), Extra Schedulers
# (스케줄러 6개 + 'Extra Schedulers (sam-extra)' 5 인자·XYZ 4), Extra Samplers(샘플러 5개 + 'Extra Samplers' 2 인자·XYZ 2),
# 진행 막대(옵션 sam3_progress_* 10·GET /sam-extra/progress), MCP 권한 스위치(옵션 sam3_mcp_allow_* 4), 그리고 앱의 구도 ·
# 카메라 칸을 Forge 로 옮긴 것(옵션 sam3_composition_panel·scripts/composition_camera.py·javascript/composition_*.js — 화면 전용,
# 앱은 자기 원본을 쓴다). 앱 노출 요청이 없어 모두
# deferred(HOLD) 또는 ignored(사유)이고 앱 UI·페이로드는 그대로다(새 always-on 넷은 core/alwayson_propagation NEVER). 새 샘플러·
# 스케줄러는 Forge 목록(/sdapi/v1/samplers·schedulers)으로 들어와 앱 Forge 콤보에 라이브로 보이며(블록 없이 확장 기본값으로 돈다),
# ComfyUI 에서는 KSampler 에 없는 이름이라 컴파일 오류('지원하지 않는 값')다 — SCRIPTS 두 항목의 api_note. mcp_server/ 는 Forge 가
# 불러오지 않는 별도 uv 프로젝트라 스캐너의 모듈 단위 밖이다(Forge 쪽은 옵션 넷을 등록하는 scripts/mcp_settings.py 뿐). 옵션 넷은
# 도우미 함수·루프로 등록돼 스캐너 option_infos 가 도우미를 따라가게 보강했다. 기존 스크립트 변경(Detail Daemon·Safe PAG 의 σ
# 오프셋 도우미를 sam3ext/guidance/sigmas.py 로 옮김, layout_lanes 자리)은 인자·infotext·옵션·제목 그대로다. 새 스크립트 8항목·
# shape 는 처음에 Forge 정지 중 설치된 소스의 ui() 를 가짜 gradio 4.40 + Forge create_script_ui_inner 규칙으로 돌려 계산해
# 두었고(같은 방법이 기존 11개 스크립트의 라이브 항목을 값 타입까지 재현), 미커밋 트리로 띄운 Forge 의 라이브 값과 모든 칸·
# 타입이 같았다. 픽스처는 커밋 6adeb83(버전 0.31.0)으로 다시 띄운 Forge 2.29.2(7860)에서 릴리스 전에 다시 받았다
# (captured_at 2026-10-03T04:57:50Z, 스크립트 16개). 기존 스크립트의 인자 모양은 그대로다.
# 2026-10-03 v0.32.0(b816661, master 머지 252e24d — 트리 같음): Colorcraft 공유 편집기 — 'Colorcraft (sam-extra)' 인자
# 579 → 67(panel_state.ARG_NAMES — enabled·masking·숨은 state·debug·debug_step·ref·편집기 61칸). 계산·infotext·XYZ·옵션은 그대로이고
# (CHANGELOG '결과 같음' — CPU 비트 대조와 실제 Forge GPU 대조: v0.31.0 패널 기준 4개의 픽셀·PNG 파일 md5 가 같음), API 의 위치 인자
# 579개 형식만 더 읽지 않는다(호환 깨짐 — 앱은 Colorcraft 를 보내지 않으므로 영향 없음, HOLD 그대로). 새 파일:
# sam3ext/colorcraft/panel_state.py(기존 'sam3ext/colorcraft/' 단위)와 javascript/colorcraft_editor.js·colorcraft_schema.js(MODULES
# ignored N6). 진행 막대(표시 버그 4)·MCP 서버(mcp_server/ — 스캐너 단위 밖)·CI 수정은 옵션·라우트·인자 계약이 그대로다("새 설정은
# 없습니다"). 확장 tests/ 의 새 파일도 스캐너 단위 밖이다. 픽스처는 커밋 b816661(버전 0.32.0)을 불러온 Forge 2.29.2(7860)에서
# 릴리스 직후 읽기 전용 GET 으로 다시 받았다(captured_at 2026-10-03T11:03:06+00:00). Colorcraft 두 항목(txt2img·img2img) 말고는 0.31.0 픽스처와 같다.
EXT_COMMIT_AUDITED = "252e24d"

MAPPED, IGNORED, DEFERRED = "mapped", "ignored", "deferred"
STATUSES = (MAPPED, IGNORED, DEFERRED)
HOLD = "HOLD"   # gap matrix '보류' 표 (가치 2 이하이거나 선행 조건이 필요)


def mapped(*app: str, note: str = "", gaps: tuple = (), comfy: tuple = (), optional: bool = False) -> dict:
    return {"status": MAPPED, "app": tuple(app), "note": note, "gaps": tuple(gaps),
            "comfy": tuple(comfy), "optional": optional}


def ignored(reason: str, *, package: str = "", optional: bool = False) -> dict:
    return {"status": IGNORED, "reason": reason, "package": package, "optional": optional}


def deferred(package: str, reason: str) -> dict:
    return {"status": DEFERRED, "package": package, "reason": reason}


def _script(*, file: str, form: str, live_argc: int, shape: str, classification: dict,
            ui_return=None, api_reads: str | None = "", arg_names: tuple | None = None,
            runtime_choices: tuple = (), app_title: str = "", app_spec: tuple | None = None,
            app_arg_names: str = "", api_note: str = "", script_info: bool = True) -> dict:
    """스크립트 항목.

    form        dict / positional / positional_or_dict / none (API 가 받는 모양)
    live_argc   script-info 인자 수(txt2img·img2img 모두). PAG 62 = v0.21.3+, 57 = v0.21.2 빌드.
    ui_return   ui() 반환 변수 이름 순서. "app_spec" 이면 앱 spec 키에서 접두사를 뺀 순서와 같아야 한다.
                None = 동적 목록(SAM3 는 sam3_ui() 가 만든다).
    api_reads   API 경로가 _arg(N)/args[N] 로 읽는 인덱스('0-10,12-61'). None = 위치로 읽지 않음(dict).
    shape       script-info 인자 모양 해시(core.sam_extra_diff.shape_hash). 라벨·기본값·범위·선택지가 바뀌면 깨진다.
    runtime_choices  실행 시점 목록이라 해시·비교에서 선택지와 기본값을 빼는 인덱스(파일 목록 등).
    script_info False = ui() 가 None 을 돌려줘 Forge 가 api_info 를 만들지 않는다(modules/scripts.py
                create_script_ui_inner) — script-info·픽스처에 없고 shape 도 없다. 제목으로는 여전히 alwayson_scripts 에
                받는다(api.py script_name_to_index 가 title() 로 찾는다). 계약 테스트는 ui() 가 None 인지를 AST 로 본다.
    """
    return {"file": file, "form": form, "live_argc": live_argc, "shape": shape, "ui_return": ui_return,
            "api_reads": api_reads, "arg_names": arg_names, "runtime_choices": tuple(runtime_choices),
            "app_title": app_title, "app_spec": app_spec, "app_arg_names": app_arg_names,
            "api_note": api_note, "script_info": script_info, **classification}


# ── always-on 스크립트 (제목 = alwayson_scripts 키, script-info 에 15개(0.30.1 라이브 11 + 미커밋 4) + ui() None 인 NegPiP) ──
SCRIPTS = MappingProxyType({
    "SAM3 Mask": _script(
        file="scripts/!sam3.py", form="dict", live_argc=2, shape="9702cc627a8c",
        ui_return=None, api_reads=None, app_title="core.sam3_args:SCRIPT_SAM3",
        api_note="args=[state dict]. process() 는 dict 첫 인자에서 sam3_enable/enabled 를 읽고, 정해진 "
                 "49개 키만 골라 Sam3Args 에 넘긴다 — 모르는 키는 오류 없이 버려진다(나-4). "
                 "script-info args[1].value 는 50개 키(sam3_enable 포함). sam3_cn_module/sam3_cn_model 은 "
                 "Forge ControlNet 이 대소문자까지 그대로 찾는다(supported_preprocessors[name]·"
                 "controlnet_filename_dict[name]) — 앱은 기능 스냅샷의 /controlnet/module_list·model_list "
                 "표기로 맞추고, 모르면 CN_MODULES 정적 폴백('None')을 쓴다(P3). 단독 SAM3·Refine 은 state 에 "
                 "요청 전용 키 sam3_source_image='init'(SAM3_REQUEST_ONLY_KEYS)를 더한다 — process() 가 Sam3Args 밖에서 "
                 "읽고 postprocess_image 가 init 이미지로 검출·인페인트한 뒤 infotext 'SAM3 Source' 로 알린다.",
        classification=mapped(
            "core/sam3_args.py:SAM3_SPEC", "core/sam3_args.py:build_state",
            "core/sam3_args.py:CN_MODULES", "core/sam3_cn_names.py:normalize_state",
            "ui/generator_generation.py:_build_sam3_settings",
            "backends/webui_backend.py:_build_sam3_script_state",
            "frontend/src/components/params/Sam3MaskCard.vue",
            "frontend/src/components/Sam3ControlNetPanel.vue",
            "frontend/src/utils/sam3ControlNet.ts",
            # 결과 infotext 'SAM3 Error'·'SAM3 Enable' → 토스트, 단독 SAM3·Refine 은 실패로 알림(P4)
            "core/sam_extra_notices.py:KEY_SAM3_ERROR", "core/sam_extra_notices.py:standalone_sam3_failure",
            # 단독 SAM3·Refine 의 원본 기준 요청(denoise 0 부모 패스의 VAE 왕복 드리프트) → 'SAM3 Source' 결과 알림
            "core/sam3_args.py:with_init_source", "backends/webui_backend.py:with_init_source",
            "core/sam_extra_notices.py:KEY_SAM3_SOURCE", "core/sam_extra_notices.py:_sam3_source_notices",
            comfy=("core/comfy_workflow_compiler.py:compile_sam3_mask_only",
                   "core/comfy_workflow_compiler.py:_sam3_pass_stack",
                   "core/comfy_workflow_compiler.py:_postprocess_stack_loras",
                   "comfy_custom_nodes/ai_studio_forge_parity/sam3_nodes.py:ForgeNeoSAM3Detailer"),
            note="LoRA: 확장 p2 는 process_images 라 자기 인페인트 프롬프트(생성 안에서 비면 메인 프롬프트 — "
                 "copy_prompt)의 LoRA 만 걸고 메인 LoRA 는 따라오지 않는다. Comfy 도 같은 목록이면 메인 스택을, "
                 "다르면 LoRA 앞에서 가른 분기(LoRA→NegPiP→조건→가이던스→DD)를 SAM3 노드에 준다(ADetailer 슬롯도 "
                 "같은 규칙 — _adetailer_pass_stack). 단독 후처리는 쓰는 패스(ADetailer 슬롯·SAM3) 중 메인 목록을 "
                 "쓰는 것이 없으면 스택 자체를 첫 패스의 목록으로 만든다(쓰지 않는 메인 LoRA 는 풀지 않는다). 태그만 "
                 "적은 프롬프트는 비지 않은 글이라 메인 프롬프트로 채우지 않고 태그를 뗀 빈 글을 인코딩한다. positive "
                 "글은 Forge parse_prompt 처럼 모든 추가 네트워크 태그(<lyco:…>·<hypernet:…>, 닫히지 않은 <이름: 이 "
                 "삼킨 <lora:…> 포함)를 떼고 이름이 정확히 lora 인 것만 건다(_positive_extra_networks). 네거티브의 "
                 "태그는 Forge 가 파싱하지 않아 글자 그대로 인코딩된다 — 노드 negative_prompt 도 그대로. 강도는 "
                 "Forge 순서다: <lora:이름:TE:UNet>·te=/unet= (_lora_spec)",
            gaps=("P14: BatchView SAM3 필드·inpaint W/H 폴백 1024",))),
    "Anima Perturbation Guidance": _script(
        # shape: Attn Scale 최대 15→100(원본 scale 0~100 — wave B PAG-F), DCW·CWM·RDC·CNS 기본값·범위와 숨은
        # RDC 스위치 True(원본 입력 — wave C DCW-F·CNS-F). 픽스처는 둘 다 Forge 정지 중 소스 AST 로 맞췄다.
        # 2026-10-02 v0.30 디테일 묶음 62-90 append — 앱 설정(PERTURBATION_SPEC, 범위는 확장 슬라이더). 픽스처는 라이브
        # (Forge 2.29.2)
        file="scripts/anima_safe_pag.py", form="positional", live_argc=91, shape="6fcc5c61ac90",
        ui_return="app_spec", api_reads="0-10,12-90", runtime_choices=(45,),
        app_title="core.anima_guidance:SCRIPT_PERTURBATION",
        app_spec=("core.anima_guidance:PERTURBATION_SPEC", "guid_"),
        api_note="idx11(auto_decay)은 visible=False 자리 유지용이라 API 가 읽지 않는다. idx45(CLIP-L)는 실행 "
                 "시점 목록. live_argc 57 은 v0.21.2 빌드 — 뒤 5개가 잘리고 idx56 'Auto' 가 SMC 를 켠다(나-5, P5). "
                 "62-90 은 v0.30 디테일 묶음(SLG mode·S² 5칸·SMC controller·Adaptive α/λ·TSR 3칸·Momentum 6칸·HiGS 7칸·"
                 "HiFlow 4칸) — 62개 빌드에 보내면 Forge 가 넘친 인자를 버려 켠 디테일 기능만 빠진다(앱이 경고 — "
                 "anima_guidance.detail_suite_note).",
        classification=mapped(
            "core/anima_guidance.py:PERTURBATION_SPEC", "core/anima_guidance.py:build_alwayson",
            "ui/generator_generation.py:_apply_postprocess_chain",
            # 샘플링 블록 빌더(메인 체인·보조 패스 봉투 공용)와 보조 패스 전달 규칙(P7)
            "ui/sampling_blocks.py:build_sampling_blocks", "core/alwayson_propagation.py:PROPAGATION",
            # 칸은 AnimaGuidancePanel 이 아니라 기능별 섹션에 있다(P0-B 분할) — 섹션마다 켜기 키가 바인딩돼 있는지
            "frontend/src/components/guidance/PagSection.vue:guid_enabled",
            "frontend/src/components/guidance/ApgSection.vue:guid_apg_enabled",
            "frontend/src/components/guidance/CwmSmcSection.vue:guid_cwm_enabled",
            "frontend/src/components/guidance/CwmSmcSection.vue:guid_smc_enabled",
            "frontend/src/components/guidance/DcwSection.vue:guid_dcw_enabled",
            "frontend/src/components/guidance/RdcSection.vue:guid_rdc_enabled",
            "frontend/src/components/guidance/DaveSection.vue:guid_dave_enabled",
            "frontend/src/components/guidance/CnsSection.vue:guid_cns_enabled",
            "frontend/src/components/guidance/AdgSection.vue:guid_adg_enabled",
            "frontend/src/components/guidance/ModulationSection.vue:guid_mod_enabled",
            # v0.30 디테일 묶음(62-90) — 칸 범위는 tests/test_guidance_detail_inputs.py 가 픽스처와 대조한다
            "frontend/src/components/guidance/PagSection.vue:slgModes",
            "frontend/src/components/guidance/CwmSmcSection.vue:smcModes",
            "frontend/src/components/guidance/DetailStagesSection.vue:guid_tsr_enabled",
            "frontend/src/components/guidance/DetailStagesSection.vue:guid_mg_enabled",
            "frontend/src/components/guidance/DetailStagesSection.vue:guid_higs_enabled",
            "frontend/src/components/guidance/DetailStagesSection.vue:guid_hiflow_enabled",
            "core/anima_guidance.py:detail_suite_note",
            # 'Anima Perturbation Guidance' 누락(v0.30 OOM 폴백)·CFG≈1 의 APG 알림(P4, 다-2 — SMC/CWM 은 원본처럼 CFG 1 에서도 돈다)
            "core/sam_extra_notices.py:KEY_PAG", "core/sam_extra_notices.py:requested_features",
            comfy=("core/comfy_workflow_compiler.py:_add_anima_guidance",
                   "comfy_custom_nodes/ai_studio_forge_parity/guidance.py:ForgeNeoAnimaGuidanceSuite"),
            note="Comfy 스위트는 sam-extra _post_cfg 순서로 건다: CFG 단계(SMC→APG→CWM, sampler_cfg_function) → "
                 "PAG(원본 노드)/SEG/SLG → DCW/RDC. ADG 는 PAG 앞(원본 PAG 의 previous_calc)이고, ADG 가 건너뛴 "
                 "스텝은 cond 그대로(PAG/SEG/SLG 항 없음 — SEG/SLG 약한 패스도 안 돈다, APG 모멘텀만 비움, "
                 "SMC e_prev 유지)에 DCW 만 적용한다(sam-extra has_pert = not adg_skipped). CFG≈1 에서는 "
                 "SMC/CWM 이 원본 DCW(+a) 처럼 돌고 APG 만 건너뛴다. CNS 는 guidance_cns.apply_cns 래퍼 "
                 "(tests/test_comfy_guidance_suite.py). 확장은 DCW-F 전 빌드에서 CFG≈1 의 SMC/CWM 도 건너뛰고 ADG "
                 "스텝의 DCW 를 생략한다. v0.30 디테일 묶음(팩 1.6.0): Adaptive SMC 는 CFG 단계의 SMC 를, S² 는 SLG "
                 "약한 패스의 블록을 바꾸고, PAG/SEG/SLG 와 DCW 사이에 HiFlow→MG→HiGS→TSR, DCW 뒤에 HiFlow 기록 — 확장 "
                 "식과 같은 텐서로 대조한다(tests/test_comfy_detail_parity.py). HiFlow 는 txt2img Hires.fix 에서만 켠다",
            gaps=("P16: 붙여 넣은 infotext 의 디테일 묶음('Anima TSR'·'Anima Momentum Guidance'·'Anima HiGS'·"
                  "'Anima HiFlow'·S2·smc=Adaptive)을 앱이 되살리지 않는다(가이던스 공통 — infotext 가져오기 없음)",
                  "P15: guid_cfg_mode 활성 키, SLG 배지, CLIP-L 없음 경고",
                  "P17(호스트 차이): PAG 와 SEG/SLG 를 함께 켜고 rescale > 0 이면 확장은 두 항을 더해 한 번 rescale "
                  "하고(_apply_perturbation), 팩은 원본 PAG post-CFG 가 자기 항을, 이어서 SEG/SLG post-CFG 가 받은 "
                  "결과를 기준으로 자기 항을 rescale 한다 — 원본 PAG 노드를 고치지 않으므로 남는다(rescale 0 이면 "
                  "같다, 계획 §2.4)",
                  "P17(호스트 차이): APG 의 PAG rescale 자동 끄기 — 확장은 CFG≈1 로 APG 를 건너뛴 스텝에서 rescale "
                  "을 살리고(_apply_perturbation 의 apg_governs), 팩은 패치 때 정한 rescale 하나를 원본 PAG 노드에 "
                  "넘기므로(CFG 는 샘플링 때만 안다) APG 가 켜져 있으면 CFG 1 에서도 0 이다",
                  "P17(호스트 차이): Skimmed CFG + PAG — 팩의 원본 PAG 는 원본 Skimmed pre-CFG 가 깎은 "
                  "cond_denoised 로 PAG 항을 만들고, 확장은 제자리 덮어쓰기를 피하려 잡아 둔 cond_raw(깎기 전)를 "
                  "쓴다(계획 §2.2 #7)",
                  "P17(호스트 차이): 앞 노드가 sampler_cfg_function 을 걸었으면 팩은 DCW(+a) 처럼 CWM/SMC 와 함께 "
                  "APG 도 건너뛴다(팩 APG 는 그 자리에 들어가므로 앞 노드의 결과를 버리지 않는다). 확장은 "
                  "SMC/CWM 만 비키고 APG 가 그 결과를 대체한다. 팩은 APG 가 안 걸렸으므로 PAG rescale 자동 "
                  "끄기도 하지 않는다(확장이 APG 를 건너뛴 CFG≈1 스텝에서 rescale 을 살리는 것과 같은 규칙)",
                  "P17: SEG/SLG 적용 구간 — 확장은 스텝 비율(_percent_in_range, 한 스텝 늦은 _pct_now), 팩은 PAG 의 "
                  "σ창(guidance_pag._patch_seg_slg_guidance). SEG/SLG 는 원본이 없어(계획 §0.3 9번) 기준을 정해야 "
                  "한다",
                  "P17(호스트 차이): ADG 시작점 — 확장은 스텝 비율(_adg_should_skip 의 _pct_now), 팩은 현재 σ 를 "
                  "percent_to_sigma 로 되돌린 진행률(_patch_adaptive_guidance). ADG 는 원본이 없다"))),
    "Anima Skimmed CFG": _script(
        file="scripts/anima_skimmed_cfg.py", form="positional", live_argc=7, shape="f387535ca4a9",
        ui_return="app_spec", api_reads="0-6",
        app_title="core.anima_guidance:SCRIPT_SKIMMED_CFG",
        app_spec=("core.anima_guidance:SKIMMED_SPEC", "skim_"),
        classification=mapped(
            "core/anima_guidance.py:SKIMMED_SPEC", "ui/generator_generation.py:_apply_postprocess_chain",
            "ui/sampling_blocks.py:build_sampling_blocks", "core/alwayson_propagation.py:PROPAGATION",
            "frontend/src/components/guidance/SkimSection.vue:skim_enabled",
            comfy=("comfy_custom_nodes/ai_studio_forge_parity/guidance.py:ForgeNeoSkimmedCFG",))),
    "Anima Detail Daemon": _script(
        file="scripts/anima_detail_daemon.py", form="positional", live_argc=14, shape="1c9d71b3f1f6",
        ui_return="app_spec", api_reads="0,2-9,11,13",
        app_title="core.anima_guidance:SCRIPT_DETAIL_DAEMON",
        app_spec=("core.anima_guidance:DETAIL_DAEMON_SPEC", "dd_"),
        api_note="값은 원본 노드(Jonseed/ComfyUI-Detail-Daemon) 단위 그대로다 — 엔진이 스케줄 전체(amount 와 "
                 "start/end offset 의 선형 결합)에 ×0.1(_SIGMA_SCALE)×p.cfg_scale 을 곱한다. UI 범위는 노드와 같은 "
                 "±5(SEMANTIC_PINS) 이고 API(init_script_args)는 범위를 검사하지 않는다. 앱은 변환 없이 그대로 "
                 "보내되 노드 범위(amount ±5, offset ±1)로 자른다. arg1(preset)·arg10(multiplier)·arg12(cfg_couple)은 원본에 없는 숨은 자리라 읽지 않고, "
                 "앱은 중립값('Custom'·1.0·True)을 보낸다. arg13 Hires Pass(muerrilla, 기본 끔 = base 패스만)는 맨 "
                 "뒤 append — 13개 빌드는 잘라 버리고 모든 패스에 적용한다(v0.21.2 옛 의미는 지원하지 않는다).",
        classification=mapped(
            "core/anima_guidance.py:DETAIL_DAEMON_SPEC", "core/anima_guidance.py:DD_PRESET_NEUTRAL",
            "core/anima_guidance.py:detail_daemon_hires_note",
            "core/sam_extra_capabilities.py:DD_HIRES_INDEX",
            "ui/generator_generation.py:_apply_postprocess_chain",
            "ui/sampling_blocks.py:build_sampling_blocks", "core/alwayson_propagation.py:PROPAGATION",
            "frontend/src/components/guidance/DetailDaemonSection.vue:dd_enabled",
            comfy=("core/comfy_workflow_compiler.py:_add_detail_daemon",
                   "comfy_custom_nodes/ai_studio_forge_parity/guidance.py:ForgeNeoAnimaDetailDaemon",
                   "comfy_custom_nodes/ai_studio_forge_parity/guidance_dd.py:get_dd_schedule",
                   "comfy_custom_nodes/ai_studio_forge_parity/guidance_dd.py:detail_daemon_sampler"),
            note="앱은 원본 노드 단위 그대로 보낸다(변환·프리셋·저장값 이전 없음). Comfy 컴파일러도 값을 그대로 넘기고 "
                 "cfg 는 base cfg_scale 을 노드의 cfg_scale_override 로 준다. 팩 노드는 원본 노드(Jonseed)의 스케줄·σ 조회"
                 "(보간)·샘플러 래퍼를 그대로 복사해 SAMPLER_SAMPLE 래퍼로 건다(×0.1×cfg, 하한 1e-6 만, 프리셋 없음). "
                 "Hires Pass 는 컴파일러가 패스를 고른다(base 또는 hires 한 곳 — muerrilla). base 샘플러가 DPM "
                 "adaptive/HeunPP2 면 넣지 않는다. 생성 안의 디테일러도 muerrilla·확장과 같다: ADetailer 는 마지막 본 "
                 "패스의 모델을 이어받고(muerrilla 콜백은 postprocess 에서야 풀리고 ADetailer i2i 는 DD 를 다시 돌리지 "
                 "않는다 — 확장은 _DD['on'] 이 남는다), SAM3 패스는 Hires Pass 가 꺼져 있을 때 자기 cfg·샘플러로 다시 "
                 "건다(확장 SAM3 p2 가 DD 스크립트를 다시 돌린다). 단독 후처리도 Forge 와 같게 보조 패스 전달(P7): "
                 "Hires Pass 가 꺼져 있을 때만 전달하고(core/alwayson_propagation), 컴파일러는 부모 img2img 를 base "
                 "패스로 보고 ADetailer 에 DD 모델을, SAM3 디테일러는 자기 cfg·샘플러로 다시 판정한다",
            gaps=("P17: Hires 체크포인트 오버라이드(hr_checkpoint_name)면 ForgeNeoHiresFix 가 모델을 새로 읽어 hires "
                  "패스의 DD(와 다른 모델 패치)가 빠진다 — Forge 는 hires 패스에도 건다",
                  "P17(호스트 차이, 확장과 공통): muerrilla 는 ADetailer 패스에서도 본 패스에서 만든 스케줄을 디테일러의 "
                  "호출 카운터로 읽는다(detail_daemon.py:281-289) — 확장과 팩은 원본 노드처럼 디테일러 샘플러의 σ 목록으로 "
                  "새로 만든다",
                  "P17(호스트 차이): Comfy SAM3 디테일러의 restore_face 는 Impact FaceDetailer 샘플링이라 DD 모델을 같이 쓴다 "
                  "— Forge 의 restore_faces 는 확산 샘플링이 아니다(GFPGAN/CodeFormer)"))),
    "Anima 3.8B (Qwen3.5 / v2)": _script(
        file="scripts/anima_3_8b.py", form="positional_or_dict", live_argc=6, shape="13dc88989e84",
        ui_return=("enabled_component", "adapter", "strength", "negative", "negative_strength", "bypass"),
        arg_names=("enabled", "adapter", "strength", "negative", "negative_strength", "bypass"),
        runtime_choices=(1,), app_title="core.anima38:SCRIPT_NAME", app_arg_names="core.anima38:ARG_NAMES",
        api_note="위치 인자든 dict 한 개든 받는다(ARG_NAMES 로 zip, 빠진 키는 ARG_DEFAULTS). v2 번들은 "
                 "블록이 없어도 자동으로 켜진다. arg1(어댑터)는 실행 시점 목록.",
        classification=mapped(
            "core/anima38.py:ARG_NAMES", "core/anima38.py:parse_args",
            # 블록 판정(모델 종류별 effective 규칙·출처)과 앱 기본값(부정 커넥터 켬, v1 끔 — 결정 D1=B)
            "core/anima38.py:build_block", "core/anima38.py:plan", "core/anima38.py:APP_T2I_DEFAULTS",
            "core/anima_model_kind.py:classify", "core/anima_model_kind.py:header_kind",
            # 샘플링 블록 기여자(메인 t2i·i2i·보조 패스 봉투 공용)와 전달 규칙(앱 기본값은 모르면 SKIP — critic A2,
            # 보조 패스는 실제 모델이 같을 때만 — A7 model_bound)
            "ui/anima38_ui.py:contribute", "ui/anima38_ui.py:prewarm_model_kinds", "ui/sampling_blocks.py:CONTRIBUTORS",
            "core/alwayson_propagation.py:PROPAGATION", "core/alwayson_propagation.py:drop_model_bound",
            "ui/hand_reconstruction_actions.py:_sampling_scripts",
            "core/sam_extra_notices.py:KEY_ANIMA38_STATUS",   # 'Anima38: off: …' 알림(P4)·비 Anima 힌트(P9)
            "frontend/src/components/params/Anima38Card.vue",
            "frontend/src/utils/anima38Card.ts",
            comfy=("core/comfy_workflow_compiler.py:_resolve_anima38_plan",
                   "comfy_custom_nodes/ai_studio_forge_parity/anima38_nodes.py:ForgeNeoAnima38V2Prompt",
                   # 카드의 v1 어댑터 선택지 = object_info ForgeNeoAnimaQwen35Prompt.adapter_name(P9 리뷰 2) — 팩이
                   # 못 찾을 때 넣는 자리표시자 이름은 빼고, v1 을 켜면 설치 안내 오류로 멈춘다(P9 리뷰 2차 1)
                   "core/anima38.py:comfy_adapter_choices", "core/anima38.py:comfy_adapter_is_placeholder",
                   "ui/anima38_ui.py:push_comfy_adapters"),
            note="dict 한 개(6키)로 보낸다. 모델 종류(체크포인트 헤더 — core/anima_model_kind)마다 블록이 없을 때와 결과가 "
                 "달라질 때만 보낸다(effective): v2 = 부정 커넥터·Bypass, 비 번들 Anima = v1 켬, 비 Anima = 보내지 않음, "
                 "모름 = 사용자가 바꾼 값만. 앱 기본값은 사용자 Forge ui-config txt2img 의 부정 커넥터 켬(KNOWN_DIFFS) — "
                 "v1 아코디언 켬(:5443)은 따르지 않는다(D1=B). I2I·인페인트는 카드 토글(기본 끔 = img2img 탭 모두 끔). "
                 "ComfyUI 컴파일러도 같은 블록을 읽고, v1 은 Forge 처럼 enabled 일 때만 켜고 어댑터는 카드 값만 쓴다(모듈 쌍 "
                 "자동 켜짐 없음 — Comfy 카드 선택지는 object_info adapter_name). "
                 "켰는데 Qwen3.5·어댑터 파일이 없으면 Comfy 는 컴파일 오류로 멈추고 Forge 는 'off: …' 로 순정 계속(차이)",
            gaps=("P16: 결과 infotext 'Anima38 …' 붙여 넣기(core/anima38.settings_from_infotext 는 준비됨)",
                  "P18: XYZ 체크포인트 축의 셀별 모델 종류 — 지금은 T2I 콤보 모델 기준으로 블록을 만든다",
                  "P16: Comfy 결과에는 'Anima38' infotext 가 없다(C5)"))),
    "DoRA Inference Mode": _script(
        file="scripts/dora_infer_mode.py", form="positional_or_dict", live_argc=5, shape="027fc702edeb",
        ui_return=("enabled", "mode", "inserted", "weak_strength", "weak_scope"),
        arg_names=("enabled", "mode", "inserted", "weak_strength", "weak_scope"),
        app_title="core.dora_infer_mode:SCRIPT_NAME", app_arg_names="core.dora_infer_mode:ARG_NAMES",
        api_note="위치 또는 dict. 라벨이 한국어라 dict(키 값 lycoris/forge_fp32/forge/no_magnitude, "
                 "keep/additive/skip/weak, attn/attn_mlp/all)로 보내야 한다. weak_strength 는 UI 0-1, API 0-2. "
                 "빠지거나 모르는 mode 는 순정이 아니라 LyCORIS 로 읽는다(coerce_args) — 앱은 mode 를 늘 싣는다.",
        classification=mapped(
            "core/dora_infer_mode.py:SCRIPT_NAME", "core/dora_infer_mode.py:plan",
            "core/dora_infer_mode.py:parse_settings",
            # 샘플링 블록 기여자(메인 t2i·i2i·보조 패스 봉투 공용)와 전달 규칙(모르면 늘 SKIP — critic A2)
            "ui/dora_infer_mode_ui.py:contribute", "ui/sampling_blocks.py:CONTRIBUTORS",
            "core/alwayson_propagation.py:PROPAGATION",
            # 결과 infotext 'DoRA mode'·'DoRA inserted' 가 없으면 훅 폴백 경고(scripts/dora_infer_mode.py:414-419)
            "core/sam_extra_notices.py:CODE_DORA_NOT_APPLIED",
            "frontend/src/components/params/DoraModeCard.vue",
            "frontend/src/utils/doraMode.ts",
            note="dict 로 보낸다. 실효 순정(꺼짐 또는 forge+keep)이면 블록을 빼고(결과가 같다), 기능 스냅샷이 스크립트를 "
                 "확인했을 때만 보낸다. 앱 기본값은 사용자 Forge ui-config 의 txt2img 값(켬·LyCORIS — KNOWN_DIFFS), "
                 "I2I·인페인트는 카드 토글(기본 끔 = img2img 아코디언 꺼짐). ComfyUI 는 블록을 만들지 않는다(순정 공식·"
                 "그대로 복제와 같다 — 사용자가 바꾼 값이면 LoRA 생성에 정보 알림)",
            gaps=("HOLD: Comfy 는 순정·그대로 복제만 — 팩 노드(P8-C)가 있어야 LyCORIS·끼워 넣은 블록 정책을 맞춘다",
                  "P16: 결과 infotext 'DoRA mode'·'DoRA inserted' 붙여 넣기(core/dora_infer_mode.from_infotext 는 준비됨)",
                  "P18: [DoRA] XYZ 축"))),
    "Anima VAE 2x (spacepxl decoder)": _script(
        file="scripts/anima_vae_2x.py", form="positional", live_argc=5, shape="e9431e0d6245",
        ui_return=("enabled", "vae_file", "mode", "blur_sigma", "renorm"), api_reads="0-4",
        runtime_choices=(1,),
        api_note="위치 인자 list 만 받는다 — dict 로 보내면 조용히 아무 일도 하지 않는다. arg1 은 파일 목록.",
        classification=deferred(HOLD, "M9 보류 — 12채널 VAE 가 설치돼 있지 않다. Comfy 노드는 있으나 "
                                      "컴파일러에 연결 안 됨")),
    # 확장 3a74dd8..1dd1a98(0.30.0 안) — NAFNet 잔차로 Anima(Qwen·Wan VAE) 격자 무늬를 지운다. 모든 후처리 뒤·저장 직전
    # (postprocess_image_after_composite) 이미지마다 한 번, SAM3·ADetailer 내부 패스(_sam3_inner·_ad_inner)에서는 돌지 않는다.
    # 2026-10-01 앱 노출(보류 → mapped): 카드·샘플링 블록 기여자·최종 이미지 전달 규칙·결과 알림·Forge 옵션 셋(P10)·
    # ComfyUI 팩 1.5.0 노드(원본 대조 골든). 상수 계약은 SEMANTIC_PINS degrid_*.
    "Anima VAE DeGrid (NAFNet)": _script(
        file="scripts/anima_vae_degrid.py", form="positional_or_dict", live_argc=5, shape="41e41dac20fc",
        runtime_choices=(1,), app_title="core.vae_degrid:SCRIPT_NAME",
        api_note="위치 인자 [enabled, model, mode, strength, tile] 또는 그 키의 dict 한 개(sam3ext/ui_vae_degrid.py "
                 "ARG_NAMES·coerce_args — 스크립트 모듈 밖이라 arg_names 는 AST 로 안 읽힌다, 핀 degrid_arg_names). ui() 가 "
                 "build_controls 의 튜플을 list() 로 돌려줘 반환 순서도 정적으로 못 읽는다. 뒤 인자는 빼도 기본값. "
                 "arg1(모델)은 models/ESRGAN·models/DeGrid 의 실행 시점 목록 — 비우거나 'None'/'auto' 면 modelspec.version "
                 "이 가장 높은 파일. 모르는 mode 는 Full, 강도 0-1.5(NaN 은 1), 타일 0 또는 128-4096(1-127 은 128).",
        classification=mapped(
            "core/vae_degrid.py:SCRIPT_NAME", "core/vae_degrid.py:ARG_NAMES", "core/vae_degrid.py:parse_settings",
            "core/vae_degrid.py:as_block", "core/vae_degrid.py:plan", "core/vae_degrid.py:live_models",
            "core/vae_degrid.py:from_infotext",
            # 샘플링 블록 기여자(메인 t2i·i2i·보조 패스 봉투 공용 — 보조 패스에는 건너가지 않는다)와 최종 이미지 전달 규칙
            "ui/vae_degrid_ui.py:contribute", "ui/vae_degrid_ui.py:apply_saved_settings", "ui/sampling_blocks.py:CONTRIBUTORS",
            "core/alwayson_propagation.py:TITLE_DEGRID", "core/alwayson_propagation.py:final_image_titles",
            # 기능 스냅샷(스크립트·모델 목록 — 'None' 자리 표시는 빼고 degrid_no_model 진단)
            "core/sam_extra_capabilities.py:TITLE_DEGRID",
            # 결과 infotext 'Anima DeGrid error'(이미지마다 n/m 장)·보냈는데 흔적 없음 → 알림(P4 규칙)
            "core/sam_extra_notices.py:CODE_DEGRID_ERROR", "core/sam_extra_notices.py:CODE_DEGRID_NOT_APPLIED",
            "core/sam_extra_notices.py:degrid_result_notices",
            # 결과 infotext 6키를 확장 그룹으로(PNG Info·갤러리·히스토리)
            "core/image_metadata.py:_EXTENSION_MARKERS",
            "frontend/src/components/params/VaeDegridCard.vue",
            "frontend/src/utils/vaeDegrid.ts:cardStatus",
            # 업스케일러 칸(Hires.fix·배치 업스케일)의 'DeGrid 모델은 업스케일러가 아님' 경고 — 두 백엔드 목록
            "frontend/src/composables/useUpscalerDegridWarning.ts:useUpscalerDegridWarning",
            comfy=("comfy_custom_nodes/ai_studio_forge_parity/degrid_nodes.py:ForgeNeoAnimaVAEDeGrid",
                   "core/comfy_workflow_compiler.py:_add_degrid", "core/comfy_workflow_compiler.py:_degrid_state",
                   "core/comfy_degrid_report.py:result_notices", "core/comfy_metadata.py:_degrid_parameters",
                   "core/comfy_workflow_controls.py:feature_preflight", "core/vae_degrid.py:comfy_model_choices",
                   "ui/vae_degrid_ui.py:push_comfy_models", "ui/vae_degrid_ui.py:refresh_comfy_models"),
            note="dict 한 개(5키)로 보낸다 — Forge 가 캐시한 위치 기본값과 무관하다. 메인 요청만: T2I·대기열·XYZ·시드 탐색·"
                 "채팅·만화 컷, I2I·인페인트·채팅 편집은 카드 토글(기본 끔 — Forge img2img 아코디언도 꺼짐). Refine·단독/배치 "
                 "SAM3·단독/배치 ADetailer·손 재구성에는 보내지 않는다(PROPAGATION passes=() — 최종 이미지 블록). 사용자가 켠 "
                 "블록이라 기능 스냅샷이 없다고 하면 빼고 block_not_sent 알림, 모르면 보낸다(main_retry 없음 — 없는 Forge 는 "
                 "422 안내). 모델 기본값 ''(자동). Krea2 는 보내지 않는다. 실패는 HTTP 오류가 아니라 이미지마다 'Anima DeGrid "
                 "error' 로만 남아(이미지는 DeGrid 없이 저장) 결과 알림으로 띄운다. Forge 옵션 셋(장치·정밀도·VRAM 상주)은 "
                 "OPTIONS 의 P10 덮어쓰기(Forge 전용). ComfyUI: 팩 노드 ForgeNeoAnimaVAEDeGrid(1.5.0+)를 Save/Preview 바로 "
                 "앞(_add_image_extensions 뒤)에 한 번 — 장치·정밀도·상주는 확장 기본값 auto·fp32·끔(COMFY_OPTIONS), "
                 "forge_quantize 켬(Forge 8비트 저장과 같은 바이트, 원본 대조 골든 tests/test_comfy_degrid_origin.py). 모델이 "
                 "없거나 노드가 없으면(옛 팩·입력 계약 불일치) 노드를 빼고 생성하며 컴파일 경고 → 알림. 노드 리포트"
                 "(ui.ai_studio_degrid)는 Forge infotext 와 같은 규칙으로 알림이 된다",
            gaps=("P16: 결과 infotext 'Anima DeGrid …' 붙여 넣기(core/vae_degrid.from_infotext 는 준비됨)",
                  "HOLD: 이미 만든 이미지의 DeGrid(Extras 탭 판)는 API 가 없다 — MODULES scripts/anima_vae_degrid_extras.py",
                  "HOLD: Forge script-info 의 모델 목록은 Forge 시작 때 스냅샷 — 생성은 파일을 다시 찾으므로 목록에 없는 "
                  "모델도 보내고 로그만 남긴다(결과 infotext 가 정답)",
                  "P16: ComfyUI PNG 메타데이터는 요청한 DeGrid 값을 적는다 — 노드가 건너뛴 이미지에도 남는다(Forge 는 실패하면 "
                  "성공 키 대신 'Anima DeGrid error' 만 남긴다). 실행 결과는 알림으로만 보인다",
                  "HOLD: 사용자의 포터블 ComfyUI extra_model_paths.yaml 은 upscale_models 만 Forge models/ESRGAN 으로 비춘다 — "
                  "Forge models/DeGrid 의 파일은 ComfyUI/models/degrid(팩이 등록하는 폴더)에 따로 두거나 경로를 더해야 "
                  "보인다(관리형 ComfyUI 는 앱이 degrid 카테고리를 적는다 — core/backend_runtime.MODEL_PATH_CATEGORIES)",
                  "HOLD: 모델 내려받기 카탈로그 항목이 없다 — NAFNet 파일은 사용자가 직접 넣는다"))),
    # 2026-10-02 검토 제안 편입(확장 미커밋) — CFG-Zero* 의 optimized-scale 만(zero-init 제외)인 Anima·CFG > 1 전용 실험
    # post-CFG. 끄면 자기 콜백만 뗀다. Skimmed CFG·다른 CFG 함수·앞선 post-CFG 보정이 있으면 건너뛰고 status 에 남긴다.
    "Anima Optimal Scale": _script(
        file="scripts/anima_cfg_optimal_scale.py", form="positional", live_argc=4, shape="b38ff7c09159",
        ui_return="app_spec", api_reads="0-3",
        app_title="core.anima_guidance:SCRIPT_OPTIMAL_SCALE",
        app_spec=("core.anima_guidance:OPTIMAL_SCALE_SPEC", "ocfg_"),
        api_note="위치 인자 [enabled, blend(0-1, 기본 0.25), start, end(%, percent_to_sigma 창)]. infotext "
                 "'Anima Optimal Scale'(blend·start·end·zero_init=omitted)·'Anima Optimal Scale status'(적용·건너뜀 수).",
        classification=mapped(
            "core/anima_guidance.py:OPTIMAL_SCALE_SPEC", "ui/generator_generation.py:_apply_postprocess_chain",
            "ui/sampling_blocks.py:build_sampling_blocks", "core/alwayson_propagation.py:PROPAGATION",
            "frontend/src/components/guidance/OptimalScaleSection.vue:ocfg_enabled",
            comfy=("core/comfy_workflow_compiler.py:_add_anima_guidance",
                   "comfy_custom_nodes/ai_studio_forge_parity/guidance_optimal_scale.py:ForgeNeoAnimaOptimalScale"),
            note="실험 기능(기본 끔, 2026-10-02 검토 제안) — Anima 화질 효과는 확인 전. Skimmed CFG 와 함께면 건너뛰고, "
                 "SMC·APG·CWM 이 CFG 기반을 다시 만들면 보정이 남지 않는다(확장 _apply_cfg_base 가 cond/uncond 에서 다시 "
                 "계산). 보조 패스에는 Skimmed CFG 와 같은 규칙으로 간다. Comfy 는 팩 1.6.0 노드를 스위트 앞에 걸어 "
                 "post-CFG 순서가 같고, SMC·APG·CWM(sampler_cfg_function)이면 노드가 건너뛴다 — 결과는 Forge 와 같다 "
                 "(tests/test_comfy_detail_parity.py·test_comfy_detail_compiler.py)",
            gaps=("P16: 붙여 넣은 infotext 'Anima Optimal Scale' 을 앱이 되살리지 않는다(가이던스 공통)",))),
    # 2026-10-03 미커밋 작업 트리(CHANGELOG v0.31.0) — 넷 다 HOLD(앱 노출 요청 없음). 앱은 블록을 만들지 않고
    # (core/alwayson_propagation NEVER), Forge 는 블록이 없으면 ui() 기본값(전부 끔·확장 기본값)으로 돌린다.
    # v0.32.0 공유 편집기 — 인자 579 → 67. 계산·infotext·XYZ·옵션 그대로, HOLD 그대로.
    "Colorcraft (sam-extra)": _script(
        file="scripts/colorcraft.py", form="positional_or_dict", live_argc=67, shape="456003a0acc3",
        api_note="위치 인자 67개(sam3ext/colorcraft/panel_state.ARG_NAMES — enabled·masking·state(숨은 Textbox: \"\" 또는 "
                 "JSON {\"v\":1,\"rev\":…, 경로: 값}, 기본값은 뺀다)·debug·debug_step·ref(숨은 Textbox '<수정자>|<마스크>|<rev>', "
                 "기본 'I|M1|0')·편집기 61칸(수정자 44·마스크 10·조합 7). 확장은 이 67개를 내부 형식이라 바뀔 수 있다고 적는다). "
                 "생성은 state 를 읽고, ref 의 rev 가 state 의 rev 와 같을 때만 ref 가 가리키는 수정자·마스크에 편집기 값을 "
                 "얹는다(panel_state.config_from_script_args). API 는 첫 인자 하나(compact — infotext 'SAM Extra Colorcraft' "
                 "문자열이나 인자 경로 dict {'enabled': …, 'I.exposure': …}, spec.is_compact_arg)나 v0.31.0 의 579개 목록을 한 겹 "
                 "감싼 [[…]] 로 보낸다. 감싸지 않은 579개는 Forge 가 67개로 잘라 넘기고 확장이 옛 형식으로 알아봐 거절한다(이미지는 "
                 "Colorcraft 없이, status 'not applied: v0.31.0 positional arguments (579 values) are no longer read - …'; state "
                 "칸이 state 가 아니면 'not applied: unreadable panel state (…)'). [enabled]·[enabled, masking] 은 예전과 같은 "
                 "뜻(나머지는 Forge 가 ui() 기본값으로 채운다). ui() 가 spec 표로 컨트롤을 만들어(colorcraft/ui.build) 반환 순서·"
                 "컴포넌트를 정적으로 못 읽는다 — 소스↔픽스처 비교 밖. 선택지는 모두 고정 목록(조합 편집기 Mask A/B 는 script-info "
                 "에 전체 목록, 조합별 좁히기는 브라우저). 붙여 넣기 필드는 65개(인자 58 — debug·debug_step·조합 편집기 7칸 빼고 — "
                 "와 선택 줄·요약 같은 화면 7)이고 조합 편집기에는 놓이지 않는다. infotext 'SAM Extra Colorcraft'·'SAM Extra "
                 "Colorcraft status'(원본·포크의 'Colorcraft' 키도 붙여 넣기로 읽는다).",
        classification=deferred(HOLD, "샘플링 중 latent 색 보정(muerrilla/ComfyUI-Colorcraft 이식, 기본 끔) — 확장도 화질·VRAM 을 "
                                      "GPU 로 확인하지 않았다(CHANGELOG). 앱 노출 요청 없음 — 앱이 만들지 않는다(core/"
                                      "alwayson_propagation NEVER). Comfy 쪽은 원본 노드 팩이 따로 있고 앱 컴파일러는 넣지 않는다")),
    "Anima SPEED": _script(
        file="scripts/anima_speed.py", form="positional_or_dict", live_argc=14, shape="c6889a2ed8d6",
        ui_return=("enabled", "mode", "threshold", "preset", "scales", "delta", "divisor", "manual", "adaptive",
                   "transform", "spectrum_a", "spectrum_beta", "seed", "hires"),
        api_note="위치 인자 14개(sam3ext/speed/forge_host.py ARG_NAMES 순서 — 스크립트 모듈 밖이라 arg_names 는 AST 로 안 "
                 "읽힌다) 또는 dict 한 개(ARG_NAMES·infotext 'Anima SPEED' 키, coerce_settings). 뒤 인자는 빼도 기본값. "
                 "idx10-12 는 gr.Number(script-info 의 step 1 은 Gradio 기본값 — UI_UNREAD). infotext 'Anima SPEED'·"
                 "'Anima SPEED status'·'Anima SPEED img2img rescale'(옵션). XYZ 는 p 속성(_anima_speed_xyz)으로 덮어쓴다.",
        classification=deferred(HOLD, "실험 기능(SPEED — 초반 스텝을 DCT 저해상도로, 기본 끔, 이미지가 달라진다). 확장도 Anima "
                                      "화질·속도를 GPU 로 확인하지 않았다(CHANGELOG). 앱 노출 요청 없음 — 앱이 만들지 않는다"
                                      "(core/alwayson_propagation NEVER)")),
    "Extra Schedulers (sam-extra)": _script(
        file="scripts/anima_extra_schedulers.py", form="positional_or_dict", live_argc=5, shape="c3552551b4b0",
        api_note="위치 인자 [custom_mode, custom_expression, custom_sigmas, laplace_mu, laplace_beta] 또는 그 키의 dict 한 개"
                 "(sam3ext/ui_extra_schedulers.py ARG_NAMES·coerce_args — 스크립트 모듈 밖). ui() 가 build_controls 의 튜플을 "
                 "list() 로 돌려줘 반환 순서·컴포넌트를 정적으로 못 읽는다. 값은 생성의 Schedule type(·Hires schedule type)이 "
                 "이 확장의 custom·Laplace 일 때만 쓰인다. 스케줄러 6개(name cosine·cosine_exponential·phi·laplace·"
                 "karras_dynamic·custom, label Cosine·CosineExponential blend·Phi·Laplace·Karras Dynamic·custom)는 불러올 때 "
                 "Forge 목록에 등록돼 /sdapi/v1/schedulers 로 나온다 — 앱 Forge 스케줄러 콤보는 라이브 name 목록(backends/"
                 "webui_backend.py get_info)이라 그대로 고를 수 있고, 이 블록 없이 확장 기본값(식 'M * (m / M) ** x', μ 0·β "
                 "0.5)으로 돈다. ComfyUI 는 KSampler scheduler 선택지에 없는 이름이라 컴파일 오류('지원하지 않는 값', "
                 "core/comfy_workflow_compiler._runtime_sampler_values) — 별칭을 두지 않는다(Comfy LaplaceScheduler 는 "
                 "KSampler 가 아닌 SIGMAS 노드). infotext 'Custom scheduler expression'·'Custom scheduler sigmas'·"
                 "'Laplace mu'·'Laplace beta'.",
        classification=deferred(HOLD, "custom 식·시그마 목록·Laplace μ/β 칸 — 앱 노출 요청 없음. 앱이 만들지 않는다"
                                      "(core/alwayson_propagation NEVER)")),
    "Extra Samplers": _script(
        file="scripts/anima_extra_samplers.py", form="positional", live_argc=2, shape="9429dfbd04a7",
        ui_return=("max_stage", "eta"),
        api_note="위치 인자 [max_stage(1-3, 기본 3), eta(0-10, 기본 1.0)] — 둘 다 생략 가능(sam3ext/extra_samplers/params.py "
                 "settings_from_args — 스크립트 클래스 밖이라 api_reads 는 비어 있다). ER SDE (Reverse-time)·(ODE) 만 읽는다. "
                 "샘플러 5개(ER SDE (Reverse-time)·ER SDE (ODE)·DPM++ 4M SDE·Euler Dy CFG++·Euler SMEA Dy CFG++)는 불러올 때 "
                 "Forge 목록에 등록돼 /sdapi/v1/samplers 로 나온다 — 앱 Forge 샘플러 콤보는 라이브 목록이라 그대로 고를 수 "
                 "있고, 이 블록 없이 확장 기본값(3·1.0)으로 돈다. ComfyUI 는 KSampler sampler_name 에 없는 이름이라 컴파일 "
                 "오류('지원하지 않는 값') — 'ER SDE (Reverse-time)'·'(ODE)' 는 Comfy SamplerER_SDE 노드의 다른 잡음 척도라 "
                 "KSampler er_sde(Forge 'ER SDE')로 잇지 않는다. infotext 'ER SDE max stage'·'ER SDE eta'(기본값과 다를 때)·"
                 "'Extra Samplers status'.",
        classification=deferred(HOLD, "ER SDE max stage·eta 칸 — 앱 노출 요청 없음. 앱이 만들지 않는다"
                                      "(core/alwayson_propagation NEVER)")),
    "SAM Extra Anima sparse LoRA": _script(
        file="scripts/anima_lora_blocks.py", form="none", live_argc=0, shape="97d170e1550e", ui_return=(),
        classification=ignored("N7 — 인자 0개인 자동 훅이라 페이로드가 필요 없다. 옵션은 OPTIONS 의 "
                               "sam3_anima_sparse_lora_forge_guess(P10). 남기는 infotext 는 MODULES 의 "
                               "scripts/anima_lora_blocks.py 가 매핑한다(P4 정보 알림)")),
    "Anima Reference PoC (shape logger)": _script(
        file="scripts/anima_ref_poc.py", form="positional", live_argc=3, shape="5c0043a8e939",
        ui_return=("enabled", "do_concat", "guidance_diagnostics"), api_reads="0-2",
        classification=ignored("N9 — 디버그용 shape logger. guidance_diagnostics(arg2)만 개발자 토글 후보 "
                               "(사용자 가치 1)")),
    "SAM3 LoRA Manager bridge": _script(
        file="scripts/lora_manager.py", form="none", live_argc=0, shape="97d170e1550e", ui_return=(),
        classification=ignored("N8 — alwayson 인자 0개인 숨은 Gradio 브리지. 앱은 ROUTES 의 /sam3-lora/* 를 "
                               "쓴다")),
    # 2026-09-30 sd-forge-negpip(상류 0585496, AGPL-3.0) 편입 — 파일 이름·제목·인자 0개·always-on 이 독립 확장과 같다.
    "NegPiP": _script(
        file="scripts/negpip.py", form="none", live_argc=0, shape="", ui_return=None, script_info=False,
        app_title="core.alwayson_propagation:TITLE_NEGPIP",
        api_note="ui() 가 None 이라 인자 0개이고 script-info 에 나오지 않는다(독립 확장 때도 같았다). 앱이 보내는 "
                 "{'args': [True]} 는 효과가 없고 always-on 이라 늘 돈다. ADetailer 는 파일 이름(stem) 'negpip' 으로 자기 "
                 "패스에 넣는다(ad_script_names 기본값).",
        classification=mapped(
            "core/alwayson_propagation.py:TITLE_NEGPIP", "ui/generator_generation.py:apply_alwayson_extensions",
            "ui/sampling_blocks.py:_negpip", "core/sam_extra_notices.py:OTHER_EXTENSIONS",
            comfy=("core/comfy_workflow_compiler.py:_add_negpip",),
            note="Forge 는 NegPiP 체크와 상관없이 always-on 으로 돈다(프롬프트에 음수 가중치가 있을 때만 켜짐). 보조 패스 "
                 "전달 규칙은 core/alwayson_propagation PROPAGATION[TITLE_NEGPIP](Comfy 만). 기능 스냅샷 밖이라 알림은 "
                 "OTHER_EXTENSIONS 의 'negpip' 이 맡는다")),
})

# ── SAM3 state 계약 (dict 형태라 위치 대신 키로 맞춘다) ─────────────────────────────
# Sam3Args 에는 없지만 process() 가 state 에서 따로 읽는 활성화 플래그.
SAM3_ACTIVATION_KEYS = ("sam3_enable", "enabled")
# 요청 전용 키 — Sam3Args(extra=forbid) 밖에서 process() 가 state 로 읽는다(scripts/!sam3.py SOURCE_STATE_KEY).
# 단독 SAM3·Refine(backends/webui_backend — core/sam3_args.with_init_source)만 보내고 build_state(생성 안 SAM3)는
# 보내지 않는다: 생성 안에서는 부모 출력이 곧 결과라 init 이미지로 돌리면 안 된다. 이 키를 모르는 예전 확장도 state 에서
# 정해진 키만 골라 Sam3Args 에 넘기므로 오류 없이 무시한다(결과에 'SAM3 Source' 가 없어 앱이 정보 알림).
SAM3_REQUEST_ONLY_KEYS = ("sam3_source_image",)
SAM3_ENABLE_LABEL = "Enable SAM3"      # script-info args[0].label
SAM3_LIVE_STATE_KEYS = 50              # script-info args[1].value 키 수 = Sam3Args 49 + sam3_enable

# ── Forge 옵션 (shared.opts.add_option, 38개 — 0.30.1 라이브 20 + 2026-10-03 미커밋 18) ─────────────────
# 주의(나-7): override_settings 에 모르는 키가 있으면 Forge classic 은 KeyError 로 요청 전체를 실패시킨다.
# P10: 요청마다 덮어쓰는 11개(core/forge_override_settings.SPECS — 체크박스는 bool, 라디오는 선택지 문자열) — 기본은 'Forge 설정 따름'(키를 보내지 않음, D3), 키마다
# 기능 스냅샷의 has_option 이 True 일 때만 보내고, 거절(500 KeyError·설정 잠금)되면 앱 키를 빼고 한 번 더 보낸다.
_P10_APP = ("core/forge_override_settings.py:SPECS", "core/forge_override_settings.py:merge_into_payload",
            "backends/webui_backend.py:_forge_option_parts", "frontend/src/components/ForgeOptionOverridesSettings.vue")


def _p10(key_const: str, note: str) -> dict:
    return mapped(*_P10_APP, f"core/forge_override_settings.py:{key_const}", note=note,
                  gaps=("P16: 결과 infotext 붙여 넣기로 이 옵션을 한 번만 덮어쓰기(Forge 'Override settings' 드롭다운, "
                        "P10 연구의 P10b)는 아직 없다",))


# 2026-10-03 미커밋 작업 트리 — 진행 막대(sd-webui-smooth-progress 편입)·MCP 서버(forgeneo-mcp 편입)의 설정. 둘 다 생성과 무관하다.
_PROGRESS_BAR = ignored("진행 막대 설정(sd-webui-smooth-progress 편입, 기본 끔) — Forge 웹 화면(javascript/progress_bar.js)만 "
                        "onOptionsChanged 로 읽는다. 생성·결과와 무관하고 앱은 자체 진행 표시를 쓴다(라우트는 ROUTES "
                        "'GET /sam-extra/progress')")
_MCP_PERMISSION = ignored("MCP 서버 권한 스위치(기본: 생성만 켬) — Forge 밖 uv 프로젝트 mcp_server/ 가 도구 호출마다 Forge "
                          "config.json 에서 다시 읽는다(scripts/mcp_settings.py 는 등록만). Forge 생성·결과와 무관하고 앱은 MCP "
                          "서버를 쓰지 않는다")

OPTIONS = MappingProxyType({
    "sam3_unload_keep_in_ram": _p10(
        "OPT_UNLOAD_KEEP_IN_RAM",
        "S5 — SAM3 를 CPU RAM 에 보관(3.4 GB, 결과 같음). onchange 가 있는 유일한 옵션: 요청 적용 때는 콜백이 돌지 않고 "
        "복원 때 돈다(끔이면 즉시 버림, processing.py:813·:846). 그래서 앱 '끔'은 SAM3 unload_after 요청이면 그 요청의 "
        "언로드에서 RAM 을 비우고, Forge 끔에 앱 '켬'은 한 요청 안의 여러 장에서만 효과가 있다(카드 경고)"),
    "sam3_ipa_duplicate_policy": deferred("P20", "R3 — 캐릭터 레퍼런스 IP-Adapter 삽입 블록 정책"),
    "sam3_anima38_keep_resident": _p10(
        "OPT_KEEP_RESIDENT", "M2 — TE·Qwen3.5·커넥터 VRAM 최대 6-8GB 상주(결과 같음). 같은 GPU 학습이면 '끔' 권장(카드 문구)"),
    "sam3_anima38_connector_fp32": _p10("OPT_CONNECTOR_FP32", "M2 — 커넥터 fp32 상주(VRAM +1.5GB, 결과 같음)"),
    "sam3_anima38_connector_run_cache": _p10("OPT_CONNECTOR_RUN_CACHE", "M2 — 커넥터 실행 캐시(결과 같음)"),
    "sam3_anima38_reference_ipa": deferred("P20", "R3 — 3.8B 에서 레퍼런스 IP-Adapter"),
    "sam3_anima_sparse_lora_forge_guess": _p10(
        "OPT_SPARSE_FORGE_GUESS", "M5 — sparse LoRA 순정 추측 변환(결과가 달라짐). 값이 Forge 와 다르면 요청이 바뀔 때마다 "
                                  "LoRA 를 다시 합친다(scripts/anima_lora_blocks.py process)"),
    "sam3_guidance_pag_prefix_dedup": _p10(
        "OPT_PREFIX_DEDUP", "G16 — 결과가 아주 미세하게 다름(infotext 'Anima PAG prefix dedup'). v0.30 이전 결과를 비트 "
                            "단위로 재현할 때 '끔'"),
    "sam3_guidance_seg_separable_blur": _p10(
        "OPT_SEG_SEPARABLE", "G16 — 결과가 아주 미세하게 다름(infotext 'Anima SEG separable blur')"),
    "sam3_guidance_dave_pre_dd_sigma": _p10(
        "OPT_DAVE_PRE_DD", "DAVE+Detail Daemon 우회(기본 켬, 결과가 달라짐 — infotext 'Anima DAVE pre-DD sigma'). 끄면 원본 "
                           "노드 조합처럼 DAVE 가 모든 스텝에 걸려 무너진다. 앱 Comfy 팩은 같은 기본값(guid_dave_pre_dd, 팩 "
                           "1.4.1)"),
    # 2026-10-02 검토 제안 편입(v0.30.0) — 둘 다 P10 덮어쓰기(기본 = Forge 설정 따름)
    "sam3_guidance_pag_cosine_envelope": _p10(
        "OPT_PAG_COSINE", "PAG 강도 곡선(자체 실험, 기본 끔, 결과가 달라짐 — PAG σ 창 양끝 0·가운데 1 인 sin² 곡선을 PAG "
                          "항에만 곱함, infotext 'Anima PAG cosine envelope'·'Anima PAG envelope status'). 논문 기법이 "
                          "아니고 화질 미검증. Forge 전용: 팩의 PAG 는 원본 노드라 곡선이 없다"),
    "sam3_builtin_negpip_enabled": _p10(
        "OPT_BUILTIN_NEGPIP", "내장 NegPiP 스위치(기본 켬 = 예전 자동 적용, 끄면 infotext 'SAM Extra NegPiP enabled: "
                              "False'). 끄면 음수 가중치를 순정 Forge 가 처리한다. Forge 전용: ComfyUI 는 앱 NegPiP 칸이 "
                              "ForgeNeoNegPip 노드를 정한다"),
    # VAE DeGrid 옵션 셋 — Forge 전용(ComfyUI 노드는 확장 기본값 auto·fp32·끔, core/vae_degrid.COMFY_OPTIONS). 라디오
    # 둘은 P10 이 선택지 값 문자열로 보낸다(ForgeOptionSpec.choices — 계약 테스트가 설치된 소스의 선택지와 대조)
    "sam3_degrid_device": _p10(
        "OPT_DEGRID_DEVICE", "VAE DeGrid 계산 장치(Radio auto/cpu, 결과 같음). cpu 는 VRAM 을 쓰지 않는다 — 같은 GPU 로 "
                             "학습 중일 때(한 장에 몇 초 더). Forge 전용: ComfyUI 노드는 auto"),
    "sam3_degrid_gpu_precision": _p10(
        "OPT_DEGRID_GPU_PRECISION", "VAE DeGrid GPU 정밀도(Radio fp32 기본/fp16 autocast — 결과가 아주 미세하게 다름). "
                                    "결과 기록 'Anima DeGrid precision' 은 런타임이 쓰는 값이지 OptionInfo infotext 가 "
                                    "아니다(스펙 infotext=''). 옛 키 sam3_degrid_precision 은 확장이 읽지 않는다. Forge "
                                    "전용: ComfyUI 노드는 fp32"),
    "sam3_degrid_keep_loaded": _p10(
        "OPT_DEGRID_KEEP_LOADED", "VAE DeGrid 모델 VRAM 상주(Checkbox, 약 117 MB, 결과 같음). Forge 전용: ComfyUI "
                                  "노드는 끔(이미지마다 올렸다가 내림)"),
    # 2026-10-03 미커밋 작업 트리(CHANGELOG v0.31.0) — 결과를 바꾸는 둘은 스크립트와 함께 HOLD, 로그·화면·MCP 는 ignored
    "sam3_colorcraft_pre_dd_sigma": deferred(
        HOLD, "Colorcraft + Detail Daemon: 스케줄 위치를 DD 가 줄이기 전 σ 로 찾기(기본 켬, 결과가 달라짐 — infotext 'SAM "
              "Extra Colorcraft pre-DD sigma', 실제로 바뀐 생성에만). Colorcraft 를 켠 생성에만 쓰인다 — "
              "SCRIPTS['Colorcraft (sam-extra)'] 와 함께 보류"),
    "sam3_colorcraft_log": ignored("콘솔 로그(패스마다 σ 목록·탭별 스케줄 값, 기본 끔, 결과 같음)"),
    "sam3_speed_img2img_rescale": deferred(
        HOLD, "SPEED 의 img2img·Hires 저해상도 시작 latent 를 flow 형태로 맞추기(기본 켬, 결과가 달라짐 — infotext 'Anima SPEED "
              "img2img rescale', 끄면 원본 노드 동작). SPEED 를 켠 생성에만 쓰인다 — SCRIPTS['Anima SPEED'] 와 함께 보류"),
    "sam3_speed_log": ignored("콘솔 로그([AnimaSPEED] 전환 계획·결과, 기본 켬, 결과 같음)"),
    "sam3_progress_enabled": _PROGRESS_BAR,
    "sam3_progress_smoothness": _PROGRESS_BAR,
    "sam3_progress_text_format": _PROGRESS_BAR,
    "sam3_progress_text_align": _PROGRESS_BAR,
    "sam3_progress_after_finish": _PROGRESS_BAR,
    "sam3_progress_fade_seconds": _PROGRESS_BAR,
    "sam3_progress_interrupt_style": _PROGRESS_BAR,
    "sam3_progress_height": _PROGRESS_BAR,
    "sam3_progress_color": _PROGRESS_BAR,
    "sam3_progress_custom_color": _PROGRESS_BAR,
    "sam3_mcp_allow_generate": _MCP_PERMISSION,
    "sam3_mcp_allow_model_switch": _MCP_PERMISSION,
    "sam3_mcp_allow_interrupt": _MCP_PERMISSION,
    "sam3_mcp_allow_download": _MCP_PERMISSION,
    "sam3_appearance_theme": ignored("N3 — Forge 화면 테마. 앱은 자체 디자인 토큰을 쓴다"),
    "sam3_composition_panel": ignored("Forge 화면의 구도 · 카메라 칸 표시(UI 를 만들 때만 읽음) — 앱 CompositionControl 을 Forge 로 "
                                      "옮긴 것이라 앱은 자기 원본을 쓴다. 생성·인자와 무관"),
    "sam3_layout_sections": ignored("N4 — txt2img 섹션 CSS 재배치, 인자 순서와 무관"),
    "sam3_fast_dropdown_visible_choices": ignored("N3 — Forge 빠른 드롭다운 표시 개수"),
    "sam3_lora_manager_tab_mode": ignored("N2 — Forge extra-networks 탭 배치"),
    "sam3_lora_manager_port": ignored("N2 — 포트는 /sam3-lora/spawn 응답 URL 에 들어 있다"),
})

# ── FastAPI 라우트 ("METHOD /path") ─────────────────────────────────────────────
ROUTES = MappingProxyType({
    # LoRA Manager 라우트 두 개도 Notebook 과 같은 헤더(X-SAM3-Notebook: 1)·로그인 의존성 뒤에 있다(없으면 403).
    "GET /sam3-lora/spawn": mapped(
        "backends/webui_backend.py:get_lora_manager_url", "frontend/src/components/LoraManagerModal.vue",
        note="프로세스를 띄우는 부작용이 있다 — 사용자가 매니저를 열 때만 부른다",
        gaps=("P6: 'starting' 폴링, iframe postMessage(sam3-add-lora) 브리지, 원격 경고",)),
    "GET /sam3-lora/config": mapped(
        "core/sam_extra_capabilities.py:EP_LORA_CONFIG",
        note="부작용 없는 존재 확인 — 런타임 기능 스냅샷(P5)이 쓴다"),
    "GET /sam3-notebook": deferred(HOLD, "T7 Notebook 프리셋 저장소 — 앱 프리셋과 겹친다(보류). "
                                         "메모는 /sam3-notebook/memos 로 따로 동기화한다"),
    "PUT /sam3-notebook": deferred(HOLD, "T7 Notebook 프리셋 저장소 — 앱 프리셋과 겹친다(보류)"),
    # 공유 메모 계약(schema_version 1, 헤더 X-SAM3-Notebook: 1). 200 = 동기화 가능, 404 = 옛 확장.
    "GET /sam3-notebook/memos": mapped(
        "core/forge_memo_client.py:MEMO_API_PATH", "core/forge_memo_client.py:list_memos",
        note="?include_deleted=1 이면 삭제 표시도 준다. 스키마·한도는 SEMANTIC_PINS 의 memo_*"),
    "PUT /sam3-notebook/memos/{memo_id}": mapped(
        "core/forge_memo_client.py:put_memo",
        note="base_updated_at 이 저장본보다 오래되면 409 + 저장된 메모"),
    "DELETE /sam3-notebook/memos/{memo_id}": mapped(
        "core/forge_memo_client.py:delete_memo", note="삭제 표시(tombstone). 모르는 id 는 404"),
    # Anima Tile & Repair JSON 라우트(T11, 원본 동등성 TR-API) — sam3ext/tile_repair_api.py. Notebook 과 같은
    # 헤더(X-SAM3-Notebook: 1)·Gradio 로그인 의존성. 기능 스냅샷은 POST 라우트에 GET → 405 로 존재를 본다.
    "POST /sam-extra/tile-repair": mapped(
        "core/forge_tile_repair_client.py:TILE_REPAIR_API_PATH", "core/forge_tile_repair_client.py:run",
        "core/tile_repair_request.py:build_route_body", "core/tile_repair_request.py:ROUTE_KEYS",
        "ui/tile_repair_actions.py:TileRepairActionsMixin", "frontend/src/components/TileRepairPanel.vue",
        "frontend/src/composables/useTileRepair.ts", "core/sam_extra_capabilities.py:EP_TILE_REPAIR",
        note="패널 Tile-Repair 모드와 같은 run_tile_repair·run_exclusive·기본값(job 은 라우트 전용 "
             "sam3_route_tile_repair — 패널 ⏹ 와 서로 멈추지 않는다). 원본은 받은 PNG/JPEG/WebP 그대로. 모르는 키는 "
             "400 — 키 목록 ROUTE_KEYS 는 tests/test_tile_repair_request.py 가 설치된 확장 소스와 대조한다",
        gaps=("HOLD: LoRA 4슬롯·PiD Upscale 모드·갤러리 삽입은 패널 전용(라우트에 없다)",)),
    "GET /sam-extra/tile-repair/options": mapped(
        "core/forge_tile_repair_client.py:TILE_REPAIR_OPTIONS_PATH", "core/forge_tile_repair_client.py:options",
        note="3채널 Anima LLLite·DiT·TE·VAE 선택지와 패널 기본값·범위(카드 드롭다운)"),
    "POST /sam-extra/tile-repair/stop": mapped(
        "core/forge_tile_repair_client.py:TILE_REPAIR_STOP_PATH", "core/forge_tile_repair_client.py:stop",
        note="라우트 요청만 멈춘다 — 도착 순간부터(큐 대기 중이면 큐를 받자마자 interrupted, 실행 중이면 "
             "stop_if_job(sam3_route_tile_repair)). 큐를 쥔 txt2img·패널 Tile-Repair 는 건드리지 않는다"),
    # 진행 막대(2026-10-03 미커밋) — sam3ext/progress_api.py. 다른 sam-extra 라우트와 같은 헤더(X-SAM3-Notebook: 1)·로그인
    # 의존성, Cache-Control no-store, OpenAPI 밖. ?id_task= 가 없으면 busy·queue_size·server_time 만 준다.
    "GET /sam-extra/progress": deferred(
        HOLD, "작업 하나(?id_task=)의 Forge 진행률(progressapi 식)·작업 전체 ETA(Hires·배치 패스, 패스 종류별 스텝 평균)·대기열 "
              "위치 — Forge 진행 막대(javascript/progress_bar.js)용이다. 앱은 /sdapi/v1/progress 를 폴링하고 요청마다 "
              "force_task_id 를 붙이므로(core/webui_cancel) 같은 id 로 물으면 작업 전체 ETA 를 받을 수 있다 — 앱 진행 "
              "표시 개선 후보, 요청 전 보류"),
})

# ── XYZ 축 (접두어별 159개 — 0.30.1 라이브 133: [SAM3] 37, [Anima …] 80, [Anima Skim] 7, [Detail Daemon] 5, [DoRA] 4;
#    2026-10-03 미커밋 26: [Colorcraft] 14, [Anima SPEED] 6, [Extra Schedulers (sam-extra)] 4, [Extra Samplers] 2) ──
_XYZ_GUIDANCE = deferred("P18", "G17 — 앱 XYZ 가 가이던스 spec 키·인덱스를 바꿔 가며 돌릴 수 없다")
_XYZ_HOLD = deferred(HOLD, "스크립트가 보류(HOLD) — 앱이 그 블록을 만들지 않으므로 축으로 바꿀 값도 없다. 노출하면 P18(앱 XYZ) 대상")
XYZ_AXES = MappingProxyType({
    "[SAM3]": {**deferred("P18", "S9 — 앱 XYZ 가 SAM3 state 키를 바꿔 가며 돌릴 수 없다"), "labels": (
        "CFG Scale", "CN Enable", "CN Guidance End", "CN Guidance Start", "CN Model", "CN Module",
        "CN Override External", "CN Weight", "Checkpoint", "Denoising Strength", "Detect Prompt", "Device",
        "Enable", "Exclude Prompt", "Inpaint Height", "Inpaint Only Masked", "Inpaint Padding",
        "Inpaint Prompt", "Inpaint Width", "Inpainting Fill", "Mask Blur", "Mask Dilation", "Mask Hull",
        "Mask Mode", "Mask Outline Expand", "Mode", "Negative Prompt", "Noise Multiplier",
        "Prompt S/R (SAM3 inpaint and main prompt)", "Prompt S/R (SAM3 inpaint)", "Restore Face", "Sampler",
        "Scheduler", "Seed", "Steps", "Threshold", "Unload After")},
    "[Anima Pert]": {**_XYZ_GUIDANCE, "labels": (
        "Attn Block Indices", "Attn Head Indices", "Attn Method", "Attn Scale", "Enable", "End Percent",
        "Legacy Perturbation Strength", "Legacy Soft/Approx", "Perturbation Strength", "Rescale",
        "Rescale Mode", "SEG Blur Sigma", "SLG Block Indices", "SLG Enable", "SLG Mode", "SLG Scale",
        "Start Percent")},
    "[Anima APG]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Eta", "Momentum", "Norm Threshold")},
    "[Anima AdaptiveG]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Keep Every", "Skip After")},
    "[Anima CFG]": {**_XYZ_GUIDANCE, "labels": ("Base Mode", "Experimental Stack")},
    "[Anima CNS]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Gamma Power", "Gamma Scale", "Strength")},
    "[Anima CWM]": {**_XYZ_GUIDANCE, "labels": ("Alpha High", "Alpha Low", "Enable")},
    "[Anima DAVE]": {**_XYZ_GUIDANCE, "labels": ("Block Indices", "Enable", "Strength", "Tau")},
    "[Anima DCW]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Lambda High", "Lambda Low")},
    "[Anima Mod]": {**_XYZ_GUIDANCE, "labels": ("Direction Weight", "Enable", "End Block", "Start Block")},
    "[Anima RDC]": {**_XYZ_GUIDANCE, "labels": ("Alpha HH", "Alpha LL", "Enable", "Tau")},
    "[Anima SMC]": {**_XYZ_GUIDANCE, "labels": (
        "Adaptive Alpha", "Adaptive Lambda", "Controller", "Enable", "K", "Lambda", "Preset")},
    # v0.30 디테일 묶음(2026-10-02, 확장 미커밋) — 앱은 이 칸들을 고정 칸으로만 보낸다(PERTURBATION_SPEC 62-90)
    "[Anima S2]": {**_XYZ_GUIDANCE, "labels": ("Drop Ratio", "Eligible Blocks", "End", "Scale", "Start")},
    "[Anima TSR]": {**_XYZ_GUIDANCE, "labels": ("Enable", "K", "Sigma")},
    "[Anima MG]": {**_XYZ_GUIDANCE, "labels": ("Alpha", "Beta", "Enable", "Normalize", "Window Max", "Window Min")},
    "[Anima HiGS]": {**_XYZ_GUIDANCE, "labels": (
        "Cutoff", "Enable", "Eta", "History Alpha", "T Max", "T Min", "Weight")},
    "[Anima HiFlow]": {**_XYZ_GUIDANCE, "labels": ("Alpha", "Beta", "Cutoff", "Enable")},
    "[Anima Skim]": {**_XYZ_GUIDANCE, "labels": (
        "Disable Flipping Filter", "Enable", "End", "Flip At", "Full Skim Negative", "Skimming CFG", "Start")},
    "[Detail Daemon]": {**deferred("P18", "G17 — 앱 XYZ 가 가이던스 spec 키를 바꿔 가며 돌릴 수 없다. 'Amount' "
                                          "축 값은 확장 arg2 를 대신하고 앱 dd_amount 와 같은 노드 단위다(변환 없음)"),
                        "labels": ("Amount", "Bias", "Enable", "End", "Start")},
    "[DoRA]": {**deferred("P18", "M8 — DoRA 축은 값이 바뀔 때마다 LoRA 를 다시 합친다(바깥 루프에 둔다)"),
               "labels": ("Inference mode", "Inserted blocks", "Weak copy scope", "Weak copy strength")},
    # 2026-10-03 미커밋 작업 트리 — 넷 다 p 속성으로 값을 덮어쓴다(Colorcraft 는 탭 I 와 전체 켜기만)
    "[Colorcraft]": {**_XYZ_HOLD, "labels": (
        "Chroma Contrast", "Clarity", "Contrast", "Enable", "End", "Exposure", "Saturation", "Sharpness", "Start",
        "Strength", "Temperature", "Tint", "Tone Compression", "Vibrance")},
    "[Anima SPEED]": {**_XYZ_HOLD, "labels": ("Delta", "Enable", "Manual sigma", "Mode", "Scale", "Sigma divisor")},
    "[Extra Schedulers (sam-extra)]": {**_XYZ_HOLD, "labels": (
        "Custom expression", "Custom sigma list", "Laplace beta", "Laplace mu")},
    "[Extra Samplers]": {**_XYZ_HOLD, "labels": ("ER SDE eta", "ER SDE max stage")},
})

# ── Gradio 전용 기능 (이름 엔드포인트 = 함수 이름) — 앱이 기대면 안 되는 것 ──────────────
# 갤러리 선택 상태와 긴 위치 인자(47·31·296·39개 …)를 쓰고 함수 이름만 바뀌어도 조용히 깨진다.
# 필요한 기능은 앱이 다시 만들거나 확장에 JSON 라우트를 추가한다(열린 질문 1).
UI_ONLY_FEATURES = MappingProxyType({
    "/handle_refine_click": {"file": "sam3ext/ui_refine.py", **ignored(
        "Refine 패널 버튼(입력 47개). 앱은 alwayson SAM3 로 자체 Refine 을 돌린다", package="P12")},
    "/handle_anima_click": {"file": "sam3ext/ui_anima.py", **ignored(
        "T11 Tile-Repair/PiD 패널 버튼(입력 30개·갤러리 상태). 앱은 이 이름 엔드포인트 대신 ROUTES 의 "
        "POST /sam-extra/tile-repair 를 쓴다. PiD Upscale 모드는 패널 전용(보류)", package=HOLD)},
    "/sam3_quick": {"file": "sam3ext/quick_button.py", **deferred(
        "P11", "S13 SAM3 퀵 버튼(txt2img 폼 전체 296개) — 앱이 img2img+SAM3 블록으로 다시 만든다")},
    "/preview_reference_layout": {"file": "sam3ext/ui_anima_reference.py", **deferred(
        "P20", "캐릭터 레퍼런스 배치 미리보기(39개)")},
    "/load_selected_reference": {"file": "sam3ext/ui_anima_reference.py", **deferred(
        "P20", "선택 이미지를 레퍼런스로 불러오기(2개)")},
    "/handle_anima_reference_click": {"file": "sam3ext/ui_anima_reference.py", **deferred(
        "P20", "R1/R2 캐릭터 레퍼런스 생성(39개) — POST /sam-extra/reference 제안")},
    "/_apply_regional_preset": {"file": "scripts/!sam3.py", **deferred(
        "P12", "Regional Swap 프리셋 — 앱 RefinePanel 버튼으로 값만 옮긴다")},
    "/expand": {"file": "sam3ext/ui_tipo.py", **deferred(
        "P19", "T1 TIPO 확장(9개) — session_hash 두 번 호출 대신 확장 라우트 제안")},
    "/handle_apply": {"file": "sam3ext/ui_tipo.py", **deferred("P19", "T1 TIPO 결과 적용(1개)")},
})

# ── 모듈 파일 단위 (scripts/*.py, sam3ext/*.py, sam3ext/<하위 패키지>/, javascript/*.js, 루트 *.py) ──
_N1 = "N1 — Forge 내부 직렬화·콜백 보호. 앱은 HTTP 로 요청하고 Forge API 큐가 직렬화한다"
_N6 = "N6 — DOM 전용 JavaScript"
MODULES = MappingProxyType({
    # 루트
    "install.py": deferred("P21", "T9 — 앱의 pip install -r 이 install.py 의 버전 고정 정책을 거치지 않는다"),
    "preload.py": deferred("P21", "T8 — --sam3-no-* 플래그(CLI_FLAGS)를 앱이 넘기지 않는다"),
    "guidance_diagnostics.py": ignored("N9 — 개발자용 가이던스 검증 로그"),
    # scripts/
    "scripts/!sam3.py": mapped("core/sam3_args.py:SCRIPT_SAM3", note="SCRIPTS['SAM3 Mask']"),
    "scripts/anima_safe_pag.py": mapped("core/anima_guidance.py:SCRIPT_PERTURBATION"),
    "scripts/anima_skimmed_cfg.py": mapped("core/anima_guidance.py:SCRIPT_SKIMMED_CFG"),
    "scripts/anima_tile_repair_api.py": mapped(
        "core/forge_tile_repair_client.py:TILE_REPAIR_API_PATH",
        note="스크립트 클래스 없이 on_app_started 로 Tile & Repair 라우트만 등록한다(ROUTES 참고)"),
    "scripts/anima_detail_daemon.py": mapped("core/anima_guidance.py:SCRIPT_DETAIL_DAEMON"),
    "scripts/anima_3_8b.py": mapped("core/anima38.py:SCRIPT_NAME", "core/anima38.py:DEFAULT_ADAPTER",
                                    note="SCRIPTS['Anima 3.8B (Qwen3.5 / v2)'] — 기본 어댑터 이름·ARG_DEFAULTS 는 "
                                         "SEMANTIC_PINS anima38_default_adapter·anima38_arg_defaults"),
    "scripts/dora_infer_mode.py": mapped("core/dora_infer_mode.py:SCRIPT_NAME", "core/dora_infer_mode.py:MODE_LABELS",
                                         note="SCRIPTS['DoRA Inference Mode'] — 라벨·infotext 'DoRA inserted' 는 "
                                              "SEMANTIC_PINS dora_*_labels·dora_infotext_insert_key"),
    "scripts/anima_vae_2x.py": deferred(HOLD, "SCRIPTS['Anima VAE 2x (spacepxl decoder)'] — M9 보류"),
    "scripts/anima_cfg_optimal_scale.py": mapped("core/anima_guidance.py:OPTIMAL_SCALE_SPEC",
                                                 note="SCRIPTS['Anima Optimal Scale'] — 가이던스 패널의 Optimal Scale 그룹"),
    "scripts/anima_vae_degrid.py": mapped(
        "core/vae_degrid.py:SCRIPT_NAME", "core/forge_override_settings.py:OPT_DEGRID_DEVICE",
        note="SCRIPTS['Anima VAE DeGrid (NAFNet)'] — 옵션 sam3_degrid_* 셋도 여기서 등록한다(OPTIONS, P10 Forge 전용). "
             "postprocess_image_after_composite 에서 이미지마다 한 번, SAM3·ADetailer 내부 패스에서는 돌지 않는다"),
    "scripts/anima_vae_degrid_extras.py": ignored(
        "Extras 탭 전용(ScriptPostprocessing, 'Anima VAE DeGrid (NAFNet, Extras)') — Forge extras API(/sdapi/v1/extra-"
        "single-image)는 Upscale·GFPGAN·CodeFormer 인자만 만들어(modules/postprocessing.py run_extras) 앱이 부를 API 가 없다. 이미 만든 이미지의 DeGrid 는 "
        "확장 라우트가 생길 때(HOLD). Settings 에서 메인 탭에도 켜면 이 이름으로 always-on 이 하나 더 생겨 앱 블록과 두 번 "
        "걸린다 — 카드 도움말이 알린다", package=HOLD),
    "scripts/anima_lora_blocks.py": mapped(
        "core/sam_extra_notices.py:KEY_SPARSE_LORA_GUESS",
        note="N7 — 자동 훅(인자 0개)이라 페이로드는 없다. 이 스크립트가 남기는 infotext 'Anima sparse LoRA'(추측 변환)만 "
             "결과 정보 알림으로 읽는다(P4, SEMANTIC_PINS anima_sparse_lora_infotext_key)",
        gaps=("M4: 토글이 꺼져 건너뛴 부분 LoRA 는 gr.Warning 만 남아 API 응답에 흔적이 없다 — 앱은 알 수 없다. "
              "확장이 infotext 나 응답에 건너뜀을 남기기 전까지 보류",)),
    "scripts/anima_ref_poc.py": ignored("N9 — 디버그"),
    "scripts/lora_manager.py": ignored("N8 — 숨은 Gradio 브리지와 옵션 두 개(N2)"),
    "scripts/appearance_theme.py": ignored("N3 — Forge 화면 테마 옵션"),
    "scripts/negpip.py": mapped("core/alwayson_propagation.py:TITLE_NEGPIP",
                                note="SCRIPTS['NegPiP'] (sd-forge-negpip 편입). 내장 스위치 옵션 sam3_builtin_negpip_enabled 도 "
                                     "여기서 등록한다(OPTIONS — P10 덮어쓰기, Forge 전용)"),
    # 2026-10-03 미커밋 작업 트리(CHANGELOG v0.31.0)
    "scripts/colorcraft.py": deferred(HOLD, "SCRIPTS['Colorcraft (sam-extra)'] — 옵션 sam3_colorcraft_* 둘·XYZ [Colorcraft] 도 "
                                            "여기서 등록한다"),
    "scripts/anima_speed.py": deferred(HOLD, "SCRIPTS['Anima SPEED'] — 옵션 sam3_speed_* 둘·XYZ [Anima SPEED] 도 여기서 등록한다"),
    "scripts/anima_extra_schedulers.py": deferred(
        HOLD, "SCRIPTS['Extra Schedulers (sam-extra)'] — 불러올 때 스케줄러 6개를 Forge 목록에 등록한다(앱 Forge 스케줄러 콤보에 "
              "라이브로 보인다 — 등록 자체는 앱 코드가 필요 없다). XYZ [Extra Schedulers (sam-extra)] 도 여기서"),
    "scripts/anima_extra_samplers.py": deferred(
        HOLD, "SCRIPTS['Extra Samplers'] — 불러올 때 샘플러 5개를 Forge 목록에 등록한다(앱 Forge 샘플러 콤보에 라이브로 "
              "보인다 — 등록 자체는 앱 코드가 필요 없다). XYZ [Extra Samplers] 도 여기서"),
    "scripts/appearance_progress_bar.py": ignored(
        "진행 막대 — 설정 sam3_progress_* 와 GET /sam-extra/progress 등록만(스크립트 클래스·생성 훅 없음)"),
    "scripts/mcp_settings.py": ignored(
        "MCP 권한 스위치 sam3_mcp_allow_* 넷만 등록(라우트·UI·생성 훅 없음, mcp_server 패키지를 불러오지 않는다). "
        "mcp_server/ 자체는 Forge 밖 uv 프로젝트라 모듈 단위 밖"),
    "scripts/composition_camera.py": ignored(
        "구도 · 카메라 — 스타일 줄 아래 빈 gr.HTML 자리와 옵션 sam3_composition_panel 등록만(스크립트 클래스·생성 훅 없음). "
        "앱 frontend/src/components/CompositionControl.vue 를 Forge 로 옮긴 것"),
    # sam3ext/
    "sam3ext/__init__.py": ignored("패키지 지연 import — 공개 이름(SAM3_NAME, Sam3Args)은 SCRIPTS 가 본다"),
    "sam3ext/__version__.py": mapped("core/sam_extra_contract.py:EXT_VERSION_AUDITED",
                                     note="버전 게이트(계약 테스트)"),
    "sam3ext/args.py": mapped("core/sam3_args.py:SAM3_SPEC", note="Sam3Args 필드·Literal·_NUMERIC_BOUNDS"),
    "sam3ext/core.py": mapped("core/forge_modules.py:resolve_sam3_checkpoint",
                              "core/sam_extra_notices.py:sam3_checkpoint_notice",
                              note="SAM3_NAME, 체크포인트 스캔(sam3*), 옵션 sam3_unload_keep_in_ram. 로컬에 없는 기본값 "
                                   "sam3.pt 는 생성 중 HF facebook/sam3(3.4 GB)에서 받는다 — 앱은 생성 전에 경고한다(P4, 다-5)",
                              gaps=("P21: --sam3-no-huggingface 를 넘기지 않아 자동 다운로드 자체는 막지 못한다",)),
    "sam3ext/inpaint_core.py": mapped("core/sam3_args.py:SAM3_SPEC", "core/sam3_cn_names.py:name_warnings",
                                      note="SAM3 인페인트 패스 — state 키로만 조절한다(build_i2i). "
                                           "inject_controlnet_unit 은 패스마다 models/sam3 를 다시 스캔해 "
                                           "controlnet_filename_dict 에만 더한다(controlnet_names 는 그대로) — "
                                           "연결 때 /controlnet/model_list 에 없는 모델도 찾을 수 있어 앱은 목록 밖 "
                                           "모델을 막지 않고 경고만 한다(P3)"),
    "sam3ext/sam3_cn_lllite.py": mapped("core/sam3_cn_names.py:lllite_module_override",
                                        "core/sam3_cn_names.py:guard_lllite_module",
                                        "frontend/src/utils/sam3ControlNet.ts",
                                        note="SAM3 CN 유닛의 Anima LLLite 전처리기 가드(TR-SAM3) — 원본(kohya)처럼 "
                                             "사용자가 준 제어 이미지: Tile & Repair 는 module None, 그 밖의 Anima "
                                             "LLLite(3채널 lineart·canny·depth, 4채널 인페인트)는 inpaint_* 만 None. "
                                             "확장은 safetensors 헤더(lllite_dit 키·lllite.cond_in_channels·"
                                             "modelspec.title)로, 앱은 같은 이름 규칙('anima' 필수)으로 판별한다 "
                                             "(tests/test_sam3_cn_names.py 가 설치된 모듈과 대조)"),
    "sam3ext/ui.py": mapped("frontend/src/components/params/Sam3MaskCard.vue",
                            "core/sam3_cn_names.py:live_lists",
                            note="SAM3 Gradio 패널 — 앱은 Sam3MaskCard.vue 로 같은 state 를 만든다. "
                                 "_controlnet_model_choices 가 models/sam3 의 LLLite(anima-lllite-inpainting-v2 등)를 "
                                 "Forge CN 목록에 등록해 /controlnet/model_list 에 나온다 — 앱 CN 드롭다운은 그 라이브 "
                                 "목록을 쓰되(P3) UI 빌드 때 목록이라 직접 입력·다시 받기(↻)도 둔다. "
                                 "_default_cn_module 선호 순서(inpaint_only 먼저)는 SAM3_SPEC 기본값과 같다",
                            gaps=("P21: 관리형 Forge(--data-dir)는 models/sam3 LLLite CN 을 찾지 못한다(다-6)",)),
    "sam3ext/ui_refine.py": mapped("frontend/src/components/RefinePanel.vue", "core/refine_prompt.py",
                                   gaps=("P12: 기본값 7개 차이와 SD 모델 오버라이드·시드 가져오기",)),
    "sam3ext/quick_button.py": deferred("P11", "S13 SAM3 퀵 버튼"),
    "sam3ext/ui_anima.py": mapped("frontend/src/components/TileRepairPanel.vue",
                                  "core/tile_repair_request.py:DEFAULT_PROMPT",
                                  note="T11 Tile-Repair 패널 — 앱 카드는 같은 기본값·범위를 쓴다(원본 sd-scripts·kohya "
                                       "노드). 버튼은 UI_ONLY_FEATURES['/handle_anima_click'] 대신 라우트로 간다",
                                  gaps=("HOLD: PiD Upscale 모드·LoRA 4슬롯은 앱에 없다",)),
    "sam3ext/anima_core.py": mapped("core/forge_tile_repair_client.py:ForgeTileRepairClient",
                                    note="T11 Tile-Repair 코어 — POST /sam-extra/tile-repair 가 run_tile_repair 를 부른다",
                                    gaps=("HOLD: run_pid_upscale(PiD Upscale)은 라우트가 없다",)),
    "sam3ext/tile_repair_api.py": mapped("core/forge_tile_repair_client.py:ForgeTileRepairClient",
                                         "core/tile_repair_request.py:ROUTE_KEYS",
                                         note="Tile & Repair JSON 라우트 3개(ROUTES 참고)"),
    "sam3ext/ui_anima_reference.py": deferred("P20", "R1/R2 캐릭터 레퍼런스 패널"),
    "sam3ext/anima_reference_core.py": deferred("P20", "R1 이어붙이기 레퍼런스 코어"),
    "sam3ext/anima_reference_recipe.py": deferred("P20", "R1 레퍼런스 레시피"),
    "sam3ext/anima_reference_runner.py": deferred("P20", "R1/R2 레퍼런스 실행기"),
    "sam3ext/anima_ipa/": deferred("P20", "R2 IP-Adapter(SigLIP2 → DiT K/V)"),
    "sam3ext/ui_tipo.py": deferred("P19", "T1 TIPO 마술봉"),
    "sam3ext/tipo/": deferred("P19", "T1/T2 TIPO 런타임·모델"),
    "sam3ext/vae_degrid.py": mapped(
        "core/vae_degrid.py:MODES", "core/vae_degrid.py:coerce_strength", "core/vae_degrid.py:coerce_tile",
        "comfy_custom_nodes/ai_studio_forge_parity/degrid_math.py:PAD_MULTIPLE",
        note="NAFNet 잔차·모드·강도·타일·판정 문턱 — 앱은 정규화 규칙을 거울로 두고(core/vae_degrid), Comfy 팩은 같은 계산을 "
             "따로 쓴다(degrid_math·degrid_runner, 확장 코드를 실행한 골든으로 대조). 상수는 SEMANTIC_PINS degrid_*"),
    "sam3ext/vae_degrid_models.py": mapped(
        "core/vae_degrid.py:NONE_NAME", "core/vae_degrid.py:MODEL_FOLDERS", "core/vae_degrid.py:resolve_name",
        "comfy_custom_nodes/ai_studio_forge_parity/degrid_files.py",
        note="모델 찾기(models/ESRGAN·models/DeGrid, NAFNet 키 검사, modelspec.version 순 자동, 겹치면 '폴더/이름') — 앱은 "
             "목록의 'None' 자리 표시를 빼고 이름 규칙만 쓴다. Comfy 팩은 같은 규칙으로 upscale_models·degrid 를 읽는다"
             "(헤더만, torch.load 없음). 핀 degrid_none_name·degrid_model_folders·degrid_nafnet_keys"),
    "sam3ext/vae_degrid_runtime.py": mapped(
        "core/forge_override_settings.py:OPT_DEGRID_DEVICE", "core/forge_override_settings.py:OPT_DEGRID_GPU_PRECISION",
        "core/forge_override_settings.py:OPT_DEGRID_KEEP_LOADED", "core/vae_degrid.py:EXTENSION_OPTION_DEFAULTS",
        note="장치·정밀도·Forge 메모리 관리 — 앱은 옵션 셋을 P10 덮어쓰기로만 다루고(Forge 전용), ComfyUI 노드는 확장 "
             "기본값(auto·fp32·끔)으로 같은 계산을 한다. 옛 키 sam3_degrid_precision 은 확장이 읽지 않아 앱도 보내지 않는다. "
             "핀 degrid_opt_*·degrid_device_*·degrid_precision_*·degrid_default_*"),
    "sam3ext/ui_vae_degrid.py": mapped(
        "core/vae_degrid.py:ARG_NAMES", "core/vae_degrid.py:MODE_CHOICES", "core/vae_degrid.py:KEY_ERROR",
        "core/vae_degrid.py:EXTRAS_TITLE",
        note="인자 해석(ARG_NAMES·coerce_args)·infotext 'Anima DeGrid …'·Gradio 컨트롤 — 앱은 dict 키·라벨·infotext 키를 "
             "같은 값으로 둔다(핀 degrid_arg_names·degrid_mode_choices·degrid_key_*·degrid_title·degrid_extras_title)"),
    "sam3ext/negpip/": ignored("편입한 NegPiP 런타임(SD1/SDXL·Anima 훅, 마스크, 독립 확장과의 공존 가드) — 앱은 "
                               "SCRIPTS['NegPiP'] 제목만 쓴다. 앱이 읽는 infotext·옵션·라우트가 없다"),
    "sam3ext/anima38/": mapped("core/anima38.py:parse_args", "core/anima_model_kind.py:BUNDLE_ARCHITECTURE",
                               "core/anima_model_kind.py:V1_ADAPTER_ARCHITECTURE",
                               note="Qwen3.5 커넥터 런타임 — 앱은 제목·인자와 번들·v1 어댑터 판별 메타데이터(files.py, "
                                    "SEMANTIC_PINS anima38_bundle_*·anima38_v1_architecture)만 안다"),
    "sam3ext/guidance/": mapped("core/anima_guidance.py:PERTURBATION_SPEC",
                                note="SMC_PRESET_NAMES 등 PAG 선택지의 출처",
                                gaps=("P17: 남은 Comfy 차이는 'Anima Perturbation Guidance' 항목의 P17 gaps",)),
    "sam3ext/anima_lora_blocks.py": mapped(
        "core/dora_infer_mode.py:INSERT_POLICIES", "core/dora_infer_mode.py:normalize_insert",
        "core/dora_infer_mode.py:WEAK_SCOPES",
        note="M4 — Forge 쪽 28/40/52 블록 자동 리맵은 앱 페이로드가 없다. 앱은 DoRA 끼워 넣은 블록 정책·약한 복사 "
             "키와 기본 강도만 쓴다(SEMANTIC_PINS dora_insert_policies·dora_weak_scopes·dora_weak_default)",
        gaps=("P17: Comfy sparse 규칙 차이",)),
    "sam3ext/dora_infer_mode.py": mapped(
        "core/dora_infer_mode.py:EXTENSION_MODES", "core/dora_infer_mode.py:normalize_mode",
        note="DoRA 병합 공식 — 앱은 모드 키·별칭 정규화·infotext 'DoRA mode' 만 쓴다(SEMANTIC_PINS dora_modes·"
             "dora_infotext_*)",
        gaps=("HOLD: Comfy 는 순정 공식만(P8-C 팩 노드)",)),
    "sam3ext/lora_manager_core.py": mapped("backends/webui_backend.py:get_lora_manager_url",
                                           note="_BRIDGE_JS 메시지 모양은 SEMANTIC_PINS 의 lora_bridge_message",
                                           gaps=("P6: _BRIDGE_JS 'sam3-add-lora' 메시지를 앱이 받지 않는다",)),
    "sam3ext/notebook_store.py": deferred(HOLD, "T7 Notebook 프리셋 저장소"),
    "sam3ext/notebook_memos.py": mapped("core/forge_memo_client.py:ForgeMemoClient",
                                        note="공유 메모 저장소(memos.json). 스키마·한도는 SEMANTIC_PINS 의 memo_*"),
    "sam3ext/coerce.py": ignored(_N1),
    "sam3ext/forge_exclusive.py": ignored(_N1),
    "sam3ext/sampler_load_guard.py": ignored(_N1),
    "sam3ext/layout_lanes.py": ignored("N4 — txt2img 섹션 레이아웃"),
    "sam3ext/panel_container.py": ignored("N5 — 선택 이미지 도크 컨테이너"),
    "sam3ext/ui_dock.py": ignored("N5 — 선택 이미지 도크(안의 기능은 각 항목에서 다룬다)"),
    # 2026-10-03 미커밋 작업 트리(CHANGELOG v0.31.0)
    "sam3ext/colorcraft/": deferred(HOLD, "Colorcraft 런타임 — 인자 표(spec: arg_names·compact 인자·infotext)·엔진·마스크·"
                                          "Debug·Gradio 패널, 색 벡터 data/*.safetensors 셋(krea2·zimage·flux2) 포함. v0.32.0 "
                                          "패널 인자(panel_state: 67개 ARG_NAMES·숨은 state·편집기 겹치기·옛 579개 거절·붙여 "
                                          "넣기·colorcraft_schema.js 생성)"),
    "sam3ext/speed/": deferred(HOLD, "SPEED 런타임 — ARG_NAMES·coerce_settings·infotext(forge_host), 전환 계획(schedule)·"
                                     "스펙트럼 변환(spectral)·샘플러 래퍼(runner)"),
    "sam3ext/extra_schedulers/": deferred(HOLD, "스케줄러 6개·안전 식 계산기(AST 화이트리스트)·시그마 목록 보간·Forge 등록 — "
                                                "앱은 custom 식·Laplace 값을 보낼 수 없다(스케줄러 이름은 Forge 목록으로 고른다)"),
    "sam3ext/extra_samplers/": deferred(HOLD, "샘플러 5개(ER SDE 둘·DPM++ 4M SDE·Euler Dy CFG++ 둘)·ER SDE 값·Forge 등록 — 앱은 "
                                              "ER SDE 값을 보낼 수 없다(샘플러 이름은 Forge 목록으로 고른다)"),
    "sam3ext/ui_extra_schedulers.py": deferred(HOLD, "Extra Schedulers 인자(ARG_NAMES·coerce_args)·infotext 키·붙여 넣기·"
                                                     "Gradio 컨트롤 — SCRIPTS['Extra Schedulers (sam-extra)']"),
    "sam3ext/progress_api.py": deferred(HOLD, "ROUTES['GET /sam-extra/progress'] — 진행률·작업 전체 ETA 계산과 라우트 등록"),
    # javascript/
    "javascript/appearance_theme.js": ignored(_N6),
    "javascript/lora_manager.js": ignored(_N6 + " — 폴링·브리지 로직은 P6 참고", package="P6"),
    "javascript/notebook.js": ignored(_N6 + " — Notebook 화면(T7)"),
    "javascript/notebook_memo.js": ignored(_N6 + " — Forge Notebook 패널의 메모 탭. 저장소 계약은 ROUTES 의 "
                                           "/sam3-notebook/memos 가 본다"),
    "javascript/notebook_lanes.js": ignored(_N6 + " — N4 섹션 레이아웃"),
    "javascript/tipo_device.js": ignored(_N6 + " — TIPO 장치 표시(P19)", package="P19"),
    "javascript/progress_bar.js": ignored(_N6 + " — 진행 막대 화면(라우트 계약은 ROUTES 의 /sam-extra/progress)"),
    "javascript/colorcraft_sliders.js": ignored(_N6 + " — Colorcraft 패널 슬라이더 화면"),
    # v0.32.0 Colorcraft 공유 편집기 — 브라우저 전용(js 전용 이벤트, 서버 요청 없음)
    "javascript/colorcraft_editor.js": ignored(_N6 + " — Colorcraft 공유 편집기 화면(항목 고르기·Reset·●/○ 표시·요약 줄, "
                                                "js 전용 이벤트로 state·ref·편집기를 함께 갱신, notebook.js 빠른 드롭다운 맞추기)"),
    "javascript/colorcraft_schema.js": ignored(_N6 + " — 편집기 필드 표(window.samextraColorcraftSchema). sam3ext/colorcraft/"
                                                "panel_state.py 가 spec.py 에서 생성한다"),
    "javascript/composition_prompt.js": ignored(_N6 + " — 앱 frontend/src/utils/compositionPrompt.ts 의 이식(태그 계산). 확장 "
                                                  "tests/_origin_composition_prompt/ 가 앱 원본을 SHA-256 으로 고정해 대조한다"),
    "javascript/composition_ui.js": ignored(_N6 + " — 구도 · 카메라 칸 화면(앱 CompositionControl.vue 이식)"),
})

# ── preload.py 명령줄 플래그 ─────────────────────────────────────────────────
CLI_FLAGS = MappingProxyType({
    "--sam3-no-huggingface": deferred("P21", "T8 — 앱이 넘기지 않는다. 없으면 생성 중 HF 자동 다운로드가 가능"),
    "--sam3-no-auto-install": deferred("P21", "T8 — install.py 는 argv 대신 COMMANDLINE_ARGS·"
                                              "SAM3_NO_AUTO_INSTALL 환경 변수를 읽는다(나-6)"),
})

# ── 의미 핀: 인자 모양은 그대로인데 의미가 바뀌는 경우를 잡는다 ─────────────────────────
# source=fixture: script-info 인자 필드, source=ast: 확장 모듈 상수.
# ast 핀은 ``expected``(값이 같아야 한다) 또는 ``patterns``(문자열 상수에 정규식이 모두 있어야 한다) 중 하나.
# ``app`` 이 있으면 그 앱 상수('모듈:이름')도 ``expected`` 와 같아야 한다(확장 없이도 도는 레지스트리 테스트).
# VAE DeGrid 상수 계약 — 앱(core/vae_degrid·core/forge_override_settings)과 Comfy 팩(degrid_math·degrid_nodes·degrid_runner)이
# 같은 값을 하드코딩한다. 핀 하나에 앱 상수 하나라(``app``), 둘 다 가진 값은 '<id>' 와 '<id>_comfy' 두 핀이 된다.
_DEGRID_UI = "sam3ext/ui_vae_degrid.py"
_DEGRID_CORE = "sam3ext/vae_degrid.py"
_DEGRID_MODELS = "sam3ext/vae_degrid_models.py"
_DEGRID_RUNTIME = "sam3ext/vae_degrid_runtime.py"
_DEGRID_TITLE = "Anima VAE DeGrid (NAFNet)"
_APP_DEGRID = "core.vae_degrid"
_PACK = "comfy_custom_nodes.ai_studio_forge_parity"


def _degrid_pins(pin_id: str, file: str, name: str, expected, meaning: str, *, app: str = "",
                 comfy: str = "") -> tuple:
    """확장 ``file:name`` 핀 — ``app``/``comfy`` 는 '모듈:이름'(각각 핀 하나, 비우면 확장 값만 본다)."""
    base = {"source": "ast", "file": file, "name": name, "expected": expected, "meaning": meaning}
    pins = [{"id": pin_id, **base, **({"app": app} if app else {})}]
    if comfy:
        pins.append({"id": f"{pin_id}_comfy", **base, "app": comfy,
                     "meaning": meaning + " — Comfy 팩 노드(ForgeNeoAnimaVAEDeGrid)도 같은 값이어야 Forge 와 같은 결과"})
    return tuple(pins)


_DEGRID_INFOTEXT = (
    ("model", "KEY_MODEL", "Anima DeGrid model", "쓴 모델(성공 때만) — 붙여 넣기(from_infotext)는 이 키가 있어야 켠다"),
    ("mode", "KEY_MODE", "Anima DeGrid mode", "모드 이름(MODE_LABELS 값)"),
    ("strength", "KEY_STRENGTH", "Anima DeGrid strength", "강도(format_strength)"),
    ("tile", "KEY_TILE", "Anima DeGrid tile", "실제로 쓴 타일(OOM 으로 줄였으면 줄인 값)"),
    ("precision", "KEY_PRECISION", "Anima DeGrid precision", "정밀도 기록(fp32·fp16-autocast) — 설정이라 붙여 넣지 않는다"),
    ("error", "KEY_ERROR", "Anima DeGrid error",
     "실패 이유(쉼표가 들어 있어 따옴표로 묶일 수 있다) — 이 키만 남으면 그 이미지는 DeGrid 없이 저장됐다. 앱 결과 "
     "알림(CODE_DEGRID_ERROR)이 이 키를 읽는다"),
)
_DEGRID_GUARDS = (
    ("IMAGE_LIKE_MIN_ABS_MEAN", 2 / 255), ("IMAGE_LIKE_CORRELATION", 0.9), ("IMAGE_LIKE_ABS_MEAN", 25 / 255),
    ("IMAGE_LIKE_LARGE_CORRELATION", 0.5), ("IMAGE_LIKE_DC_MEAN", 25 / 255), ("IMAGE_LIKE_DC_SIGN", 0.8),
    ("IMAGE_LIKE_DC_RATIO", 0.5), ("RESIDUAL_BLOWUP_ABS_MEAN", 100 / 255),
)

_DEGRID_PINS = (
    *_degrid_pins("degrid_title", _DEGRID_UI, "TITLE", _DEGRID_TITLE,
                  "alwayson_scripts 키 — 앱 블록·기능 스냅샷·전달 규칙·알림 표의 제목이 모두 이 상수 하나에서 나온다",
                  app=f"{_APP_DEGRID}:SCRIPT_NAME"),
    *_degrid_pins("degrid_extras_title", _DEGRID_UI, "EXTRAS_TITLE", "Anima VAE DeGrid (NAFNet, Extras)",
                  "Extras 판 제목 — 메인 탭에도 켜면 이 이름의 always-on 이 하나 더 생긴다(앱은 보내지 않는다)",
                  app=f"{_APP_DEGRID}:EXTRAS_TITLE"),
    *_degrid_pins("degrid_arg_names", _DEGRID_UI, "ARG_NAMES", ("enabled", "model", "mode", "strength", "tile"),
                  "API dict 키(뒤에만 덧붙인다) — 앱은 dict 한 개로 보내므로 이름이 바뀌면 값이 조용히 기본값이 된다",
                  app=f"{_APP_DEGRID}:ARG_NAMES"),
    *_degrid_pins("degrid_mode_choices", _DEGRID_UI, "MODE_CHOICES",
                  ("Full (전체)", "Dark Pixels Mainly (어두운 점 위주)", "Bright Pixels Mainly (밝은 점 위주)"),
                  "카드 라벨·순서 — script-info 기본값과 위치 인자로 보낸 라벨을 앱이 모드 키로 읽는다(normalize_mode)",
                  app=f"{_APP_DEGRID}:MODE_CHOICES"),
    *(pin for short, const, value, meaning in _DEGRID_INFOTEXT
      for pin in _degrid_pins(f"degrid_key_{short}", _DEGRID_UI, const, value, "결과 infotext 키: " + meaning,
                              app=f"{_APP_DEGRID}:{const}")),
    *_degrid_pins("degrid_tile_step", _DEGRID_UI, "TILE_SLIDER_STEP", 128,
                  "타일 슬라이더 단위(0 과 128 단위) — 카드 입력 step", app=f"{_APP_DEGRID}:TILE_STEP"),
    *_degrid_pins("degrid_modes", _DEGRID_CORE, "MODES", ("full", "dark", "bright"),
                  "API mode 키 — 모르는 값은 Full 로 읽힌다(coerce_args)",
                  app=f"{_APP_DEGRID}:MODES", comfy=f"{_PACK}.degrid_math:MODES"),
    *_degrid_pins("degrid_mode_labels", _DEGRID_CORE, "MODE_LABELS",
                  {"full": "Full", "dark": "Dark Pixels Mainly", "bright": "Bright Pixels Mainly"},
                  "infotext 'Anima DeGrid mode' 값 = 노드 팩 이름 — Comfy 노드 mode 선택지·메타데이터도 이 이름",
                  app=f"{_APP_DEGRID}:MODE_LABELS", comfy=f"{_PACK}.degrid_math:MODE_LABELS"),
    *_degrid_pins("degrid_default_mode", _DEGRID_CORE, "DEFAULT_MODE", "full", "모드 기본값",
                  app=f"{_APP_DEGRID}:DEFAULT_MODE", comfy=f"{_PACK}.degrid_math:DEFAULT_MODE"),
    *_degrid_pins("degrid_default_strength", _DEGRID_CORE, "DEFAULT_STRENGTH", 1.0,
                  "강도 기본값(NaN·읽을 수 없는 값도 이 값)",
                  app=f"{_APP_DEGRID}:DEFAULT_STRENGTH", comfy=f"{_PACK}.degrid_math:DEFAULT_STRENGTH"),
    *_degrid_pins("degrid_strength_min", _DEGRID_CORE, "STRENGTH_MIN", 0.0, "강도 하한(자르기) — 0 이면 원본 그대로",
                  app=f"{_APP_DEGRID}:STRENGTH_MIN", comfy=f"{_PACK}.degrid_math:STRENGTH_MIN"),
    *_degrid_pins("degrid_strength_max", _DEGRID_CORE, "STRENGTH_MAX", 1.5, "강도 상한(자르기)",
                  app=f"{_APP_DEGRID}:STRENGTH_MAX", comfy=f"{_PACK}.degrid_math:STRENGTH_MAX"),
    *_degrid_pins("degrid_default_tile", _DEGRID_CORE, "DEFAULT_TILE", 512, "타일 기본값(노드 팩 nafnet_node.py 값)",
                  app=f"{_APP_DEGRID}:DEFAULT_TILE", comfy=f"{_PACK}.degrid_math:DEFAULT_TILE"),
    *_degrid_pins("degrid_tile_overlap", _DEGRID_CORE, "TILE_OVERLAP", 32, "타일 겹침 — 바뀌면 타일 경계의 결과가 달라진다",
                  app=f"{_APP_DEGRID}:TILE_OVERLAP", comfy=f"{_PACK}.degrid_math:TILE_OVERLAP"),
    *_degrid_pins("degrid_min_tile", _DEGRID_CORE, "MIN_TILE", 128,
                  "타일 하한(1-127 은 128, OOM 으로 줄이다 이 아래면 포기)",
                  app=f"{_APP_DEGRID}:MIN_TILE", comfy=f"{_PACK}.degrid_math:MIN_TILE"),
    *_degrid_pins("degrid_max_tile", _DEGRID_CORE, "MAX_TILE", 4096, "타일 상한(API·붙여 넣기 범위)",
                  app=f"{_APP_DEGRID}:MAX_TILE", comfy=f"{_PACK}.degrid_math:MAX_TILE"),
    *_degrid_pins("degrid_pad_multiple", _DEGRID_CORE, "PAD_MULTIPLE", 16, "NAFNet 입력 reflect 패딩 배수",
                  comfy=f"{_PACK}.degrid_math:PAD_MULTIPLE"),
    *(pin for name, value in _DEGRID_GUARDS
      for pin in _degrid_pins(f"degrid_guard_{name.lower()}", _DEGRID_CORE, name, value,
                              "잔차가 아닌 모델(이미지를 내는 업스케일러)·터진 출력을 건너뛰는 판정 문턱 — 'not a DeGrid "
                              "residual model'·'output blew up' 오류가 같은 이미지에서 나야 한다",
                              comfy=f"{_PACK}.degrid_math:{name}")),
    *_degrid_pins("degrid_none_name", _DEGRID_MODELS, "NONE_NAME", "None",
                  "모델이 하나도 없을 때 목록의 자리 표시 — 앱은 어디서나 이 값을 모델로 치지 않는다(degrid_no_model 진단)",
                  app=f"{_APP_DEGRID}:NONE_NAME"),
    *_degrid_pins("degrid_model_folders", _DEGRID_MODELS, "MODEL_FOLDERS", ("ESRGAN", "DeGrid"),
                  "모델 폴더와 이름이 겹칠 때 붙는 접두('ESRGAN/…'·'DeGrid/…') — Comfy 이름 ↔ Forge 이름 변환이 쓴다",
                  app=f"{_APP_DEGRID}:MODEL_FOLDERS"),
    *_degrid_pins("degrid_model_extensions", _DEGRID_MODELS, "MODEL_EXTENSIONS", (".safetensors", ".pth", ".pt"),
                  "모델 파일 확장자", comfy=f"{_PACK}.degrid_math:MODEL_EXTENSIONS"),
    *_degrid_pins("degrid_nafnet_keys", _DEGRID_MODELS, "NAFNET_KEYS", (
        "intro.weight", "ending.weight", "ups.0.0.weight", "downs.0.weight", "middle_blks.0.beta",
        "middle_blks.0.gamma", "middle_blks.0.conv1.weight", "middle_blks.0.conv2.weight",
        "middle_blks.0.conv3.weight", "middle_blks.0.sca.1.weight", "middle_blks.0.conv4.weight",
        "middle_blks.0.conv5.weight", "middle_blks.0.norm1.weight", "middle_blks.0.norm2.weight",
        "encoders.0.0.beta", "encoders.0.0.gamma", "decoders.0.0.beta", "decoders.0.0.gamma"),
                  "NAFNet 판별 키 — 목록에 오르는 파일(=모델 선택지·자동 선택)이 같아야 한다",
                  comfy=f"{_PACK}.degrid_math:NAFNET_KEYS"),
    *_degrid_pins("degrid_opt_device", _DEGRID_RUNTIME, "OPT_DEVICE", "sam3_degrid_device",
                  "이미지마다 읽는 장치 옵션 — 이름이 바뀌면 앱 P10 덮어쓰기가 조용히 아무 일도 하지 않는다",
                  app="core.forge_override_settings:OPT_DEGRID_DEVICE"),
    *_degrid_pins("degrid_opt_precision", _DEGRID_RUNTIME, "OPT_PRECISION", "sam3_degrid_gpu_precision",
                  "이미지마다 읽는 GPU 정밀도 옵션(옛 키 sam3_degrid_precision 은 읽지 않는다)",
                  app="core.forge_override_settings:OPT_DEGRID_GPU_PRECISION"),
    *_degrid_pins("degrid_opt_keep_loaded", _DEGRID_RUNTIME, "OPT_KEEP_LOADED", "sam3_degrid_keep_loaded",
                  "실행 뒤 읽는 VRAM 상주 옵션", app="core.forge_override_settings:OPT_DEGRID_KEEP_LOADED"),
    *_degrid_pins("degrid_device_auto", _DEGRID_RUNTIME, "DEVICE_AUTO", "auto", "장치 라디오 값(P10 이 이 문자열을 보낸다)",
                  app=f"{_APP_DEGRID}:DEVICE_AUTO", comfy=f"{_PACK}.degrid_nodes:DEVICE_AUTO"),
    *_degrid_pins("degrid_device_cpu", _DEGRID_RUNTIME, "DEVICE_CPU", "cpu", "장치 라디오 값 — VRAM 을 쓰지 않는다",
                  app=f"{_APP_DEGRID}:DEVICE_CPU", comfy=f"{_PACK}.degrid_nodes:DEVICE_CPU"),
    *_degrid_pins("degrid_precision_fp32", _DEGRID_RUNTIME, "PRECISION_FP32", "fp32", "정밀도 라디오 값(기본)",
                  app=f"{_APP_DEGRID}:PRECISION_FP32", comfy=f"{_PACK}.degrid_runner:PRECISION_FP32"),
    *_degrid_pins("degrid_precision_fp16", _DEGRID_RUNTIME, "PRECISION_FP16", "fp16",
                  "정밀도 라디오 값 — fp16 autocast(결과가 아주 미세하게 다름)",
                  app=f"{_APP_DEGRID}:PRECISION_FP16", comfy=f"{_PACK}.degrid_runner:PRECISION_FP16"),
    *_degrid_pins("degrid_default_device", _DEGRID_RUNTIME, "DEFAULT_DEVICE", "auto",
                  "옵션 기본값 — ComfyUI 노드가 쓰는 값(COMFY_OPTIONS, Forge 설정은 ComfyUI 에 해당 없음)",
                  app=f"{_APP_DEGRID}:DEFAULT_DEVICE"),
    *_degrid_pins("degrid_default_precision", _DEGRID_RUNTIME, "DEFAULT_PRECISION", "fp32",
                  "옵션 기본값 — ComfyUI 노드가 쓰는 값(COMFY_OPTIONS)", app=f"{_APP_DEGRID}:DEFAULT_PRECISION"),
    *_degrid_pins("degrid_default_keep_loaded", _DEGRID_RUNTIME, "DEFAULT_KEEP_LOADED", False,
                  "옵션 기본값 — ComfyUI 노드가 쓰는 값(COMFY_OPTIONS)", app=f"{_APP_DEGRID}:DEFAULT_KEEP_LOADED"),
    *_degrid_pins("degrid_bytes_per_pixel", _DEGRID_RUNTIME, "BYTES_PER_PIXEL", 1536,
                  "GPU 여유 메모리로 첫 타일 크기를 고르는 추정치 — 바뀌면 쓴 타일(infotext 'Anima DeGrid tile')이 달라진다",
                  comfy=f"{_PACK}.degrid_nodes:BYTES_PER_PIXEL"),
    # script-info(픽스처) — 카드 입력 범위·step 이 Forge UI 와 같다
    {"id": "degrid_ui_strength_step", "source": "fixture", "script": _DEGRID_TITLE, "index": 3,
     "field": "step", "expected": 0.05, "app": f"{_APP_DEGRID}:STRENGTH_STEP",
     "meaning": "강도 슬라이더 step — 카드(frontend/src/utils/vaeDegrid.ts STRENGTH_RANGE)도 같다"},
    {"id": "degrid_ui_strength_max", "source": "fixture", "script": _DEGRID_TITLE, "index": 3,
     "field": "maximum", "expected": 1.5, "app": f"{_APP_DEGRID}:STRENGTH_MAX",
     "meaning": "강도 슬라이더 최대 = API 자르기 상한(STRENGTH_MAX)"},
    {"id": "degrid_ui_tile_max", "source": "fixture", "script": _DEGRID_TITLE, "index": 4,
     "field": "maximum", "expected": 4096, "app": f"{_APP_DEGRID}:MAX_TILE",
     "meaning": "타일 슬라이더 최대 = API·붙여 넣기 범위(MAX_TILE)"},
    {"id": "degrid_ui_tile_min", "source": "fixture", "script": _DEGRID_TITLE, "index": 4,
     "field": "minimum", "expected": 0,
     "meaning": "타일 슬라이더 최소 0 = 나누지 않음 — 카드 TILE_RANGE.min 도 0(그 위는 128 단위)"},
)

SEMANTIC_PINS = (
    {"id": "dd_amount_max", "source": "fixture", "script": "Anima Detail Daemon", "index": 2,
     "field": "maximum", "expected": 5.0,
     "meaning": "원본 노드 detail_amount 범위 ±5(Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:326) — "
                "앱 DETAIL_DAEMON_SPEC(DD_AMOUNT_MAX)도 같은 범위라 KNOWN_DIFFS 가 없다. 앱은 값을 변환 없이 이 "
                "범위로 잘라 보내고(노드는 범위 밖 값을 받지 않는다) 이 값으로 아무것도 판정하지 않는다"},
    {"id": "dd_amount_min", "source": "fixture", "script": "Anima Detail Daemon", "index": 2,
     "field": "minimum", "expected": -5.0, "meaning": "dd_amount_max 와 같다(노드 min -5)"},
    {"id": "dd_sigma_scale", "source": "ast", "file": "scripts/anima_detail_daemon.py", "name": "_SIGMA_SCALE",
     "expected": 0.1,
     "meaning": "엔진이 amount·offset 스케줄에 곱하는 원본 고정 배율(노드 detail_daemon_node.py:169, :294, muerrilla "
                "detail_daemon.py:244) — 앱은 노드 단위 그대로 보내므로 이 값에 기대어 변환하지 않는다"},
    {"id": "dd_sigma_scale_comfy", "source": "ast", "file": "scripts/anima_detail_daemon.py", "name": "_SIGMA_SCALE",
     "expected": 0.1, "app": "comfy_custom_nodes.ai_studio_forge_parity.guidance:DD_SIGMA_SCALE",
     "meaning": "Comfy 팩 노드(ForgeNeoAnimaDetailDaemon, 원본 노드 detail_daemon_sampler 복사)가 스케줄 전체에 "
                "늘 곱하는 배율 — 같아야 한다. 앱 컴파일러는 노드 값을 변환 없이 넘긴다"},
    {"id": "pag_smc_presets", "source": "ast", "file": "sam3ext/guidance/cwm_smc.py", "name": "SMC_PRESET_NAMES",
     "expected": ("Off", "Auto", "SD1.5 / SD2", "SDXL", "SD3 / SD3.5", "Flux", "Qwen-Image", "Cosmos / Wan",
                  "Custom"),
     "meaning": "PAG idx56 선택지는 이 목록의 [1:] — 'Off' 가 빠져 있어 끄는 것은 idx57 마스터 토글이다"},
    {"id": "anima38_arg_defaults", "source": "ast", "file": "scripts/anima_3_8b.py", "name": "ARG_DEFAULTS",
     "expected": {"enabled": False, "adapter": "Anima-3.8B-expanded_adapter.safetensors", "strength": 1.0,
                  "negative": False, "negative_strength": 1.0, "bypass": False},
     "app": "core.anima38:ARG_DEFAULTS",
     "meaning": "dict 로 보낼 때 빠진 키를 채우는 값 = 블록이 없을 때의 동작 — core/anima38.py DEFAULT_SETTINGS 와 같아야 "
                "한다. 앱은 이 값과 결과가 같은 요청에는 블록을 싣지 않는다(effective, P9)"},
    # Anima 3.8B(P9) — 앱이 같은 값을 하드코딩한다. 바뀌면 모델 종류 판정·기본 어댑터가 조용히 어긋난다.
    {"id": "anima38_default_adapter", "source": "ast", "file": "scripts/anima_3_8b.py", "name": "DEFAULT_ADAPTER",
     "expected": "Anima-3.8B-expanded_adapter.safetensors", "app": "core.anima38:DEFAULT_ADAPTER",
     "meaning": "v1 어댑터 기본 이름 — 카드의 기본값·선택지 폴백(확장 _adapter_choices 는 아무것도 없어도 이 이름 하나)"},
    {"id": "anima38_bundle_architecture", "source": "ast", "file": "sam3ext/anima38/files.py",
     "name": "BUNDLE_ARCHITECTURE", "expected": "anima_3_8b_semantic_connector_v2_bundle",
     "app": "core.anima_model_kind:BUNDLE_ARCHITECTURE",
     "meaning": "v2 번들 체크포인트 메타데이터 architecture — 앱은 헤더만 읽어 같은 규칙(bundle_metadata)으로 v2 를 가린다"},
    {"id": "anima38_bundle_format", "source": "ast", "file": "sam3ext/anima38/files.py", "name": "BUNDLE_FORMAT",
     "expected": "1", "app": "core.anima_model_kind:BUNDLE_FORMAT",
     "meaning": "v2 번들 메타데이터 anima_v2_bundle_format — architecture 와 함께 맞아야 v2(자동 켜짐)"},
    {"id": "anima38_v1_architecture", "source": "ast", "file": "sam3ext/anima38/files.py", "name": "ARCHITECTURE",
     "expected": "anima_progressive_qwen35_cross_adapter_v1", "app": "core.anima_model_kind:V1_ADAPTER_ARCHITECTURE",
     "meaning": "v1 어댑터 architecture — infotext 'Anima38 architecture' 가 이 값이면 v1 을 켰던 이미지(붙여 넣기 "
                "settings_from_infotext)"},
    # 결과·생성 전 알림(P4) — core/sam_extra_notices.py 가 같은 값을 읽는다. 바뀌면 알림이 조용히 멈춘다.
    {"id": "anima38_status_key", "source": "ast", "file": "scripts/anima_3_8b.py", "name": "STATUS_KEY",
     "expected": "Anima38", "app": "core.sam_extra_notices:KEY_ANIMA38_STATUS",
     "meaning": "3.8B 상태 infotext 키('v2 bundle'·'v1 adapter'·'bypass'·'off: 이유') — 'off' 면 순정 Anima 로 생성됐다"},
    # 단독 SAM3·Refine 원본 기준 요청 — core/sam3_args·core/sam_extra_notices 가 같은 값을 하드코딩한다. 바뀌면 요청이 조용히
    # 무시되거나(키·값) 알림이 틀린다(infotext 키·값 — 키가 없으면 '지원 안 함' 정보 알림).
    {"id": "sam3_source_state_key", "source": "ast", "file": "scripts/!sam3.py", "name": "SOURCE_STATE_KEY",
     "expected": "sam3_source_image", "app": "core.sam3_args:SOURCE_STATE_KEY",
     "meaning": "SAM3 state 의 원본 기준 요청 키 — process() 가 Sam3Args 밖에서 읽는다(SAM3_REQUEST_ONLY_KEYS)"},
    {"id": "sam3_source_init", "source": "ast", "file": "scripts/!sam3.py", "name": "SOURCE_INIT",
     "expected": "init", "app": "core.sam3_args:SOURCE_INIT",
     "meaning": "요청 값(공백·대소문자 무시) — 이 값이면 init 이미지로 검출·인페인트, 다른 값은 예전처럼 부모 출력"},
    {"id": "sam3_source_infotext_key", "source": "ast", "file": "scripts/!sam3.py", "name": "INFOTEXT_SOURCE",
     "expected": "SAM3 Source", "app": "core.sam_extra_notices:KEY_SAM3_SOURCE",
     "meaning": "요청했을 때만 남는 infotext 키 — 없으면 요청을 모르는 확장(정보 알림 CODE_SAM3_SOURCE_UNSUPPORTED)"},
    {"id": "sam3_source_note_init", "source": "ast", "file": "scripts/!sam3.py", "name": "SOURCE_NOTE_INIT",
     "expected": "init image", "app": "core.sam_extra_notices:SAM3_SOURCE_INIT_NOTE",
     "meaning": "'SAM3 Source' 값 — init 이미지로 돌았다(마스크 밖이 원본 그대로). 다른 값은 경고"},
    {"id": "sam3_source_note_output", "source": "ast", "file": "scripts/!sam3.py", "name": "SOURCE_NOTE_OUTPUT",
     "expected": "output", "app": "core.sam_extra_notices:SAM3_SOURCE_OUTPUT_NOTE",
     "meaning": "'SAM3 Source: output (<이유>)' 의 머리 — 요청했지만 조건이 안 맞아 부모 출력으로 돌았다(경고 "
                "CODE_SAM3_SOURCE_FALLBACK 이 괄호 안 이유를 보인다)"},
    {"id": "sam3_hf_checkpoint_name", "source": "ast", "file": "sam3ext/core.py", "name": "HF_CHECKPOINT_NAME",
     "expected": "sam3.pt", "app": "core.sam_extra_notices:HF_CHECKPOINT_NAME",
     "meaning": "로컬에 없으면 생성 중 Hugging Face 에서 받는 기본 체크포인트 이름 — 앱이 생성 전에 경고한다(다-5)"},
    {"id": "sam3_hf_checkpoint_repo", "source": "ast", "file": "sam3ext/core.py", "name": "HF_CHECKPOINT_REPO",
     "expected": "facebook/sam3", "app": "core.sam_extra_notices:HF_CHECKPOINT_REPO",
     "meaning": "자동 다운로드 저장소(gated, 약 3.4 GB) — 경고 문구와 'SAM3 Error' 힌트가 이 이름을 쓴다"},
    {"id": "anima_sparse_lora_infotext_key", "source": "ast", "file": "sam3ext/anima_lora_blocks.py",
     "name": "INFOTEXT_SPARSE_GUESS_KEY", "expected": "Anima sparse LoRA",
     "app": "core.sam_extra_notices:KEY_SPARSE_LORA_GUESS",
     "meaning": "부분 LoRA 를 순정 추측 변환으로 로드한 생성의 infotext 키('Forge guess (<파일> 28->40, …)') — "
                "앱은 정보 알림으로 띄운다(P4, M4 일부). 건너뛴 부분 LoRA 는 gr.Warning 만 남아 API 로 알 수 없다"},
    # 공유 메모 계약 — 앱 core/memo_store.py 가 같은 값을 하드코딩한다.
    {"id": "memo_schema_version", "source": "ast", "file": "sam3ext/notebook_memos.py",
     "name": "MEMO_SCHEMA_VERSION", "expected": 1, "app": "core.memo_store:SCHEMA_VERSION",
     "meaning": "memos.json schema_version. 앱은 더 높은 버전을 읽으면 읽기 전용으로 둔다 — 올라가면 병합 규칙부터 맞춘다"},
    {"id": "memo_max_memos", "source": "ast", "file": "sam3ext/notebook_memos.py", "name": "MAX_MEMOS",
     "expected": 500, "app": "core.memo_store:MAX_MEMOS",
     "meaning": "삭제 표시 포함 최대 메모 수(오래된 삭제 표시부터 정리) — 앱과 다르면 한쪽이 정리한 삭제 표시가 되살아난다"},
    {"id": "memo_max_title", "source": "ast", "file": "sam3ext/notebook_memos.py",
     "name": "MAX_MEMO_TITLE_LENGTH", "expected": 120, "app": "core.memo_store:MAX_TITLE_LENGTH",
     "meaning": "제목 최대 길이 — 앱이 더 길게 받으면 PUT 이 거부된다"},
    {"id": "memo_max_text", "source": "ast", "file": "sam3ext/notebook_memos.py",
     "name": "MAX_MEMO_TEXT_LENGTH", "expected": 100_000, "app": "core.memo_store:MAX_TEXT_LENGTH",
     "meaning": "본문 최대 길이 — 앱이 더 길게 받으면 PUT 이 거부된다"},
    # LoRA Manager iframe → 부모 창 postMessage (P6 가 이 모양을 파싱한다).
    {"id": "lora_bridge_message", "source": "ast", "file": "sam3ext/lora_manager_core.py", "name": "_BRIDGE_JS",
     "patterns": (
         r'postMessage\(\s*\{\s*type:\s*"sam3-add-lora",\s*text:\s*syntax\s*\}',
         r'\{\s*type:\s*"sam3-add-lora",\s*text:\s*texts\.join\(", "\),\s*replace:\s*!!replace\s*\}',
         r'"<lora:"\s*\+\s*name\s*\+\s*":"\s*\+\s*strength\s*\+\s*":"\s*\+\s*clip\s*\+\s*">"',
         r'"<lora:"\s*\+\s*name\s*\+\s*":"\s*\+\s*strength\s*\+\s*">"',
     ),
     "meaning": "메시지 {type:'sam3-add-lora', text:'<lora:이름:강도[:clip]>, …', replace?:bool} — P6 의 "
                "loraManagerMessage 파서가 이 필드 이름·구분자(', ')·태그 형식을 믿는다. 이름은 LoRA Manager 의 것이고 "
                "Forge 는 둘째 값을 텍스트 인코더, 셋째 값을 UNet 강도로 읽는다(ComfyUI 컴파일러도 같다 — _lora_spec)"},
    # DoRA 추론 방식(P8) — core/dora_infer_mode.py 가 같은 키·라벨·infotext 를 하드코딩한다. 모양(인자 수·라벨 해시)이
    # 그대로여도 키 이름·순서·기본 강도가 바뀌면 앱이 보낸 dict 가 조용히 다른 값(LyCORIS·keep·attn)으로 읽힌다.
    {"id": "dora_modes", "source": "ast", "file": "sam3ext/dora_infer_mode.py", "name": "MODES",
     "expected": ("forge", "forge_fp32", "lycoris", "no_magnitude"), "app": "core.dora_infer_mode:EXTENSION_MODES",
     "meaning": "API dict 의 mode 키 — 모르는 값은 LyCORIS 로 읽힌다(coerce_args). 'forge' = 순정(블록 없음과 같음)"},
    {"id": "dora_insert_policies", "source": "ast", "file": "sam3ext/anima_lora_blocks.py",
     "name": "DUPLICATE_POLICIES", "expected": ("keep", "additive", "skip", "weak"),
     "app": "core.dora_infer_mode:INSERT_POLICIES",
     "meaning": "API dict 의 inserted 키(정규화는 이 순서의 부분 문자열) — 모르는 값은 keep(그대로 복제)"},
    {"id": "dora_weak_scopes", "source": "ast", "file": "sam3ext/anima_lora_blocks.py", "name": "WEAK_SCOPES",
     "expected": ("attn", "attn_mlp", "all"), "app": "core.dora_infer_mode:WEAK_SCOPES",
     "meaning": "약한 복사 범위 키 — 모르는 값은 attn"},
    {"id": "dora_weak_default", "source": "ast", "file": "sam3ext/anima_lora_blocks.py",
     "name": "DEFAULT_WEAK_STRENGTH", "expected": 0.12, "app": "core.dora_infer_mode:DEFAULT_WEAK_STRENGTH",
     "meaning": "약한 복사 강도 기본값(브리지 권장 0.08~0.18) — 숫자가 아닌 값도 이 값이 된다"},
    {"id": "dora_infotext_mode_key", "source": "ast", "file": "sam3ext/dora_infer_mode.py", "name": "INFOTEXT_KEY",
     "expected": "DoRA mode", "app": "core.dora_infer_mode:INFOTEXT_MODE_KEY",
     "meaning": "순정이 아닌 방식으로 합친 요청의 infotext 키 — 앱은 보냈는데 없으면 훅 폴백 경고(CODE_DORA_NOT_APPLIED)"},
    {"id": "dora_infotext_values", "source": "ast", "file": "sam3ext/dora_infer_mode.py", "name": "INFOTEXT_VALUES",
     "expected": {"forge_fp32": "Forge fp32", "lycoris": "LyCORIS", "no_magnitude": "No magnitude"},
     "app": "core.dora_infer_mode:INFOTEXT_VALUES",
     "meaning": "'DoRA mode' 값 — 붙여 넣기(from_infotext, P16)가 이 값을 모드 키로 되읽는다"},
    {"id": "dora_infotext_insert_key", "source": "ast", "file": "scripts/dora_infer_mode.py",
     "name": "INFOTEXT_DUP_KEY", "expected": "DoRA inserted", "app": "core.dora_infer_mode:INFOTEXT_INSERT_KEY",
     "meaning": "그대로 복제가 아닌 정책의 infotext 키(값 additive / skip / weak 0.12 attn)"},
    {"id": "dora_mode_labels", "source": "ast", "file": "scripts/dora_infer_mode.py", "name": "MODE_CHOICES",
     "expected": ("LyCORIS (학습과 동일 · fp32)", "Forge/Comfy 공식 · fp32", "Forge/Comfy (순정)",
                  "DoRA 끔 (크기 보정 없이 ΔW만 · 일반 LoKr처럼)"),
     "app": "core.dora_infer_mode:MODE_LABELS",
     "meaning": "카드 표시 라벨(상류 이름 그대로)·순서 — frontend/src/utils/doraMode.ts MODE_OPTIONS 가 같은 값이다"},
    {"id": "dora_insert_labels", "source": "ast", "file": "scripts/dora_infer_mode.py", "name": "DUP_CHOICES",
     "expected": ("그대로 복제 (순정 · Forge 기본)", "덧셈형 (끼워 넣은 블록만 DoRA 크기 보정 끔)",
                  "넣지 않음 (원래 블록에만 · 모든 LoRA)", "약한 복사 (브리지식 · 강도·범위 조절)"),
     "app": "core.dora_infer_mode:INSERT_LABELS",
     "meaning": "끼워 넣은 블록 라벨·순서 — 앱 정규화(normalize_insert)가 이 라벨을 서로 다른 키로 읽어야 한다"},
    {"id": "dora_scope_labels", "source": "ast", "file": "scripts/dora_infer_mode.py", "name": "SCOPE_CHOICES",
     "expected": ("어텐션만 (브리지 기본)", "어텐션+MLP", "전체 (모듈레이션·노름 포함)"),
     "app": "core.dora_infer_mode:SCOPE_LABELS",
     "meaning": "약한 복사 범위 라벨·순서"},
    # Forge 옵션 덮어쓰기(P10) — 요청 중에 옵션을 **읽는** 쪽 상수. 등록 키(OPTIONS·add_option)만 같고 읽는 이름이 바뀌면
    # 앱이 보낸 override 가 조용히 아무 일도 하지 않는다. 기본값·infotext·onchange·컴포넌트는 option_infos 테스트가 본다.
    {"id": "opt_anima38_keep_resident", "source": "ast", "file": "sam3ext/anima38/runtime.py",
     "name": "OPT_KEEP_RESIDENT", "expected": "sam3_anima38_keep_resident",
     "app": "core.forge_override_settings:OPT_KEEP_RESIDENT",
     "meaning": "인코딩 직후·생성 끝 restore 가 읽는 상주 옵션 — 앱 '끔'이면 생성 뒤 VRAM 6-8GB 를 반납한다"},
    {"id": "opt_anima38_connector_fp32", "source": "ast", "file": "sam3ext/anima38/connector_fp32.py",
     "name": "OPT_CONNECTOR_FP32", "expected": "sam3_anima38_connector_fp32",
     "app": "core.forge_override_settings:OPT_CONNECTOR_FP32",
     "meaning": "커넥터 설치 때 읽는 fp32 상주 옵션"},
    {"id": "opt_anima38_connector_run_cache", "source": "ast", "file": "sam3ext/anima38/connector_cache.py",
     "name": "OPT_CONNECTOR_RUN_CACHE", "expected": "sam3_anima38_connector_run_cache",
     "app": "core.forge_override_settings:OPT_CONNECTOR_RUN_CACHE",
     "meaning": "커넥터 호출마다 읽는 실행 캐시 옵션"},
    {"id": "opt_sam3_keep_in_ram", "source": "ast", "file": "sam3ext/core.py", "name": "OPT_UNLOAD_KEEP_IN_RAM",
     "expected": "sam3_unload_keep_in_ram", "app": "core.forge_override_settings:OPT_UNLOAD_KEEP_IN_RAM",
     "meaning": "'Unload after' 마다 읽는 RAM 보관 옵션(onchange 는 복원 때만 돈다)"},
    {"id": "opt_guidance_prefix_dedup", "source": "ast", "file": "scripts/anima_safe_pag.py", "name": "OPT_PREFIX_DEDUP",
     "expected": "sam3_guidance_pag_prefix_dedup", "app": "core.forge_override_settings:OPT_PREFIX_DEDUP",
     "meaning": "가이던스 패스마다 읽는 PAG prefix dedup 옵션"},
    {"id": "opt_guidance_seg_separable", "source": "ast", "file": "scripts/anima_safe_pag.py",
     "name": "OPT_SEG_SEPARABLE", "expected": "sam3_guidance_seg_separable_blur",
     "app": "core.forge_override_settings:OPT_SEG_SEPARABLE",
     "meaning": "가이던스 패스마다 읽는 SEG 1D 분리 블러 옵션"},
    {"id": "opt_guidance_dave_pre_dd", "source": "ast", "file": "scripts/anima_safe_pag.py", "name": "OPT_DAVE_PRE_DD",
     "expected": "sam3_guidance_dave_pre_dd_sigma", "app": "core.forge_override_settings:OPT_DAVE_PRE_DD",
     "meaning": "DAVE 를 붙일 때마다 읽는 DAVE+Detail Daemon 우회 옵션(8878b9e)"},
    {"id": "opt_sparse_forge_guess", "source": "ast", "file": "sam3ext/anima_lora_blocks.py",
     "name": "OPT_SPARSE_FORGE_GUESS", "expected": "sam3_anima_sparse_lora_forge_guess",
     "app": "core.forge_override_settings:OPT_SPARSE_FORGE_GUESS",
     "meaning": "LoRA 로드 때 읽는 부분 LoRA 추측 변환 옵션(value is True 로만 켜진다 — 앱은 JSON bool 만 보낸다)"},
    {"id": "opt_infotext_prefix_dedup", "source": "ast", "file": "scripts/anima_safe_pag.py",
     "name": "INFOTEXT_PREFIX_DEDUP", "expected": "Anima PAG prefix dedup",
     "app": "core.forge_override_settings:INFOTEXT_PREFIX_DEDUP",
     "meaning": "dedup 실제 적용 값이 남는 infotext 이름 — 강제한 값이 적용됐는지 결과로 확인할 수 있다"},
    {"id": "opt_infotext_seg_separable", "source": "ast", "file": "scripts/anima_safe_pag.py",
     "name": "INFOTEXT_SEG_SEPARABLE", "expected": "Anima SEG separable blur",
     "app": "core.forge_override_settings:INFOTEXT_SEG_SEPARABLE",
     "meaning": "SEG 분리 블러 실제 적용 값이 남는 infotext 이름"},
    {"id": "opt_infotext_dave_pre_dd", "source": "ast", "file": "scripts/anima_safe_pag.py",
     "name": "INFOTEXT_DAVE_PRE_DD", "expected": "Anima DAVE pre-DD sigma",
     "app": "core.forge_override_settings:INFOTEXT_DAVE_PRE_DD",
     "meaning": "DAVE+Detail Daemon 우회 infotext 이름"},
    {"id": "opt_infotext_sparse_guess", "source": "ast", "file": "sam3ext/anima_lora_blocks.py",
     "name": "INFOTEXT_SPARSE_GUESS_KEY", "expected": "Anima sparse LoRA",
     "app": "core.forge_override_settings:INFOTEXT_SPARSE_GUESS",
     "meaning": "추측 변환한 생성에만 남는 infotext 이름(붙여 넣으면 이 옵션을 켜는 덮어쓰기가 된다)"},
    {"id": "opt_guidance_pag_cosine", "source": "ast", "file": "scripts/anima_safe_pag.py", "name": "OPT_PAG_COSINE",
     "expected": "sam3_guidance_pag_cosine_envelope", "app": "core.forge_override_settings:OPT_PAG_COSINE",
     "meaning": "가이던스 패스마다 읽는 PAG 강도 곡선 옵션(2026-10-02 검토 제안)"},
    {"id": "opt_infotext_pag_cosine", "source": "ast", "file": "scripts/anima_safe_pag.py",
     "name": "INFOTEXT_PAG_COSINE", "expected": "Anima PAG cosine envelope",
     "app": "core.forge_override_settings:INFOTEXT_PAG_COSINE",
     "meaning": "곡선을 켠 PAG 생성에 남는 infotext 이름"},
    {"id": "opt_builtin_negpip", "source": "ast", "file": "scripts/negpip.py", "name": "OPT_BUILTIN_NEGPIP",
     "expected": "sam3_builtin_negpip_enabled", "app": "core.forge_override_settings:OPT_BUILTIN_NEGPIP",
     "meaning": "process_batch 마다 읽는 내장 NegPiP 스위치(끄면 음수 가중치를 순정 Forge 가 처리)"},
    {"id": "opt_infotext_builtin_negpip", "source": "ast", "file": "scripts/negpip.py",
     "name": "INFOTEXT_BUILTIN_NEGPIP", "expected": "SAM Extra NegPiP enabled",
     "app": "core.forge_override_settings:INFOTEXT_BUILTIN_NEGPIP",
     "meaning": "스위치를 끈 생성에 남는 infotext 이름"},
    # VAE DeGrid — 위 _DEGRID_PINS(앱·Comfy 팩 거울)
    *_DEGRID_PINS,
)

# ── ui() 컴포넌트에서 AST 로 못 읽는 칸 (script, index) → 필드와 사유 ─────────────────────
# 픽스처에는 값이 있는데 정적으로 풀 수 없어 소스↔픽스처 비교에서 빠지는 칸. 실행 시점 목록(runtime_choices 의
# value·choices)은 여기 적지 않는다. 새로 못 읽게 된 칸도, 이제 읽히는 칸도 실패한다(스캐너가 조용히 비교를
# 건너뛰지 않게) — 스캐너를 보강하거나 사유와 함께 적는다.
UI_UNREAD = MappingProxyType({
    ("Anima 3.8B (Qwen3.5 / v2)", 0): {
        "fields": ("label", "value"),
        "reason": "`accordion = InputAccordion if … else None` 별칭을 거쳐 부른다 — 스캐너가 호출 대상을 모른다"},
    ("Anima Perturbation Guidance", 11): {
        "fields": ("label",),
        "reason": "auto_decay — visible=False 자리 유지용 체크박스라 label 키워드가 없다(Gradio 기본값)"},
    ("DoRA Inference Mode", 0): {
        "fields": ("label",),
        "reason": "InputAccordion 과 구버전 폴백 gr.Checkbox('Enable') 의 라벨이 달라 한쪽을 고를 수 없다"},
    # Anima SPEED 의 gr.Number 셋(Spectrum A·beta, 노이즈 시드) — step 키워드가 없어 script-info 의 step 1 은 Gradio 4 기본값
    # (gradio/components/number.py step=1)이다. 스캐너는 Gradio 기본값을 흉내 내지 않는다(sam_extra_scan.ui_component).
    ("Anima SPEED", 10): {"fields": ("step",), "reason": "gr.Number(Spectrum A) — step 은 Gradio 기본값 1"},
    ("Anima SPEED", 11): {"fields": ("step",), "reason": "gr.Number(Spectrum beta) — step 은 Gradio 기본값 1"},
    ("Anima SPEED", 12): {"fields": ("step",), "reason": "gr.Number(precision=0, 노이즈 시드) — step 은 Gradio 기본값 1"},
})

# ── 앱 spec 과 라이브 값의 알려진 차이 (script, key, field) ────────────────────────────
# kind=intended: 앱이 일부러 다르게 둔다. kind=gap: 작업 패키지가 고칠 빈틈.
# source=fixture: script-info 픽스처와 비교해 나온 차이, source=ast: 확장 소스(Sam3Args 기본값·_NUMERIC_BOUNDS)와의
# 차이. 두 곳에서 모두 나오는 차이는 ``("fixture", "ast")`` 처럼 튜플로 적는다.
# 여기 없는 차이는 실패, 여기 있는데 더는 차이가 없으면(고쳐졌으면) 역시 실패 — 이 표에서 지운다.
KNOWN_DIFFS = MappingProxyType({
    ("SAM3 Mask", "sam3_device", "default"): {
        "app": "cuda", "live": "auto", "kind": "intended", "package": "", "source": ("fixture", "ast"),
        "reason": "앱은 예전부터 cuda 고정 — 동작 보존(core/sam3_args.py 주석)"},
    ("SAM3 Mask", "sam3_steps", "maximum"): {
        "app": 1000, "live": None, "kind": "intended", "package": "", "source": "ast",
        "reason": "확장은 상한 없음(PositiveInt). 앱은 위젯 범위로 한 번 더 자른다"},
    ("SAM3 Mask", "sam3_cfg_scale", "maximum"): {
        "app": 100.0, "live": None, "kind": "intended", "package": "", "source": "ast",
        "reason": "확장은 상한 없음(NonNegativeFloat). 앱은 위젯 범위로 한 번 더 자른다"},
    # 원본 동등성 wave C: DCW·CWM·RDC tau·CNS 의 기본값·범위는 확장 ui()(DCW-F·CNS-F)와 앱 spec(APP-SPEC)이 둘 다
    # 원본 노드 입력이 돼 차이가 없다. 남는 것은 원본에 없는 RDC 스위치(58)의 기본값뿐이다.
    ("Anima Perturbation Guidance", "rdc_enabled", "default"): {
        "app": False, "live": True, "kind": "intended", "package": "", "source": "fixture",
        "reason": "원본 RDC 는 스위치가 없다(tau 0 = 끔, DCW 안에서만 — origin: namemechan/ComfyUI-DCW@66aaf9dd:"
                  "dcw_node.py:757-815, :855-856). 확장은 58 칸을 숨은 True(끄는 쪽 veto 로만 읽음)로 두고, 앱은 "
                  "저장된 사용자 스위치라 기본 False 로 남기되 보내는 값은 _derive_rdc 가 원본 켜짐(스위치·DCW·"
                  "tau > 0)으로 맞춘다 — 두 기본 모두 tau 0 이라 켜지지 않는다"},
    # DoRA 앱 기본값(P8) — API 기본값은 코드 값(아코디언 꺼짐)이고, 사용자 Forge UI 는 ui-config 로 txt2img 를 켠다(다-3).
    # tests/test_sam_extra_contract.py test_dora_live_defaults_match_app 가 쓴다(core.dora_infer_mode.APP_DEFAULTS).
    # Anima 3.8B 앱 기본값(P9) — 사용자 Forge ui-config txt2img 의 부정 커넥터 켬(:5452). v1 아코디언 켬(:5443)은
    # 결정 D1=B 로 따르지 않는다(비 번들 Anima 결과 보존). tests/test_sam_extra_contract.py
    # test_anima38_live_defaults_vs_app_t2i_defaults 가 쓴다(core.anima38.APP_T2I_DEFAULTS).
    ("Anima 3.8B (Qwen3.5 / v2)", "negative", "default"): {
        "app": True, "live": False, "kind": "intended", "package": "", "source": "fixture",
        "reason": "사용자 Forge ui-config.json txt2img 부정 커넥터 켬(:5452) — API 는 블록이 없으면 끔이라 3.8B v2 요청에 "
                  "블록을 싣는다. img2img 는 모두 끔(:5460-5475)이라 I2I·인페인트 토글 기본 끔"},
    ("DoRA Inference Mode", "enabled", "default"): {
        "app": True, "live": False, "kind": "intended", "package": "", "source": "fixture",
        "reason": "사용자 Forge ui-config.json txt2img 아코디언 켬(:5669) — API 는 블록이 없으면 순정이라 앱이 늘 켠 블록을 "
                  "싣는다. img2img 는 꺼짐(:5675)이라 I2I·인페인트 토글 기본 끔"},
})


# ── 조회 도우미 ──────────────────────────────────────────────────────────────
def axis_label_map() -> dict[str, str]:
    """전체 XYZ 축 라벨 → 접두어."""
    return {f"{prefix} {suffix}": prefix for prefix, group in XYZ_AXES.items() for suffix in group["labels"]}


def diff_sources(entry: dict) -> tuple:
    """KNOWN_DIFFS 항목의 source 를 튜플로."""
    source = entry.get("source")
    return tuple(source) if isinstance(source, (tuple, list)) else (source,)


def classified_sections() -> dict[str, MappingProxyType]:
    """분류(status)가 붙은 표 전부 — 계약 테스트의 형식 검사용."""
    return {"SCRIPTS": SCRIPTS, "OPTIONS": OPTIONS, "ROUTES": ROUTES, "XYZ_AXES": XYZ_AXES,
            "UI_ONLY_FEATURES": UI_ONLY_FEATURES, "MODULES": MODULES, "CLI_FLAGS": CLI_FLAGS}


def optional_keys(section: MappingProxyType) -> frozenset:
    return frozenset(key for key, entry in section.items() if entry.get("optional"))
