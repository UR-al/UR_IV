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
선언, SEMANTIC_PINS 의 Comfy 미러). Comfy 참조(``comfy=``)는 파일·심볼이 있는지만 본다. UI_ONLY_FEATURES 는
등록된 함수가 그 파일에 있는지만 보는 한 방향 검사다(새 Gradio 이름 엔드포인트는 잡지 않는다).
"""
from __future__ import annotations

from types import MappingProxyType

EXT_VERSION_AUDITED = "0.30.0"
# 감사 시점 HEAD — 원본 동등성 작업(861ac02..dd18876), 변경 기록(80d2dce), LoRA Manager 경로 인증(a2114b5), DAVE+DD 우회 토글(8878b9e)까지.
# 작업 트리는 깨끗했다. v0.30.0 은 아직 릴리스 전이라 같은 버전 문자열 안에서 코드가 바뀌었다(818b8fe 도 0.30.0).
EXT_COMMIT_AUDITED = "8878b9e"

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
            app_arg_names: str = "", api_note: str = "") -> dict:
    """스크립트 항목.

    form        dict / positional / positional_or_dict / none (API 가 받는 모양)
    live_argc   script-info 인자 수(txt2img·img2img 모두). PAG 62 = v0.21.3+, 57 = v0.21.2 빌드.
    ui_return   ui() 반환 변수 이름 순서. "app_spec" 이면 앱 spec 키에서 접두사를 뺀 순서와 같아야 한다.
                None = 동적 목록(SAM3 는 sam3_ui() 가 만든다).
    api_reads   API 경로가 _arg(N)/args[N] 로 읽는 인덱스('0-10,12-61'). None = 위치로 읽지 않음(dict).
    shape       script-info 인자 모양 해시(core.sam_extra_diff.shape_hash). 라벨·기본값·범위·선택지가 바뀌면 깨진다.
    runtime_choices  실행 시점 목록이라 해시·비교에서 선택지와 기본값을 빼는 인덱스(파일 목록 등).
    """
    return {"file": file, "form": form, "live_argc": live_argc, "shape": shape, "ui_return": ui_return,
            "api_reads": api_reads, "arg_names": arg_names, "runtime_choices": tuple(runtime_choices),
            "app_title": app_title, "app_spec": app_spec, "app_arg_names": app_arg_names,
            "api_note": api_note, **classification}


# ── always-on 스크립트 (제목 = alwayson_scripts 키, 라이브 10개) ─────────────────────
SCRIPTS = MappingProxyType({
    "SAM3 Mask": _script(
        file="scripts/!sam3.py", form="dict", live_argc=2, shape="9702cc627a8c",
        ui_return=None, api_reads=None, app_title="core.sam3_args:SCRIPT_SAM3",
        api_note="args=[state dict]. process() 는 dict 첫 인자에서 sam3_enable/enabled 를 읽고, 정해진 "
                 "49개 키만 골라 Sam3Args 에 넘긴다 — 모르는 키는 오류 없이 버려진다(나-4). "
                 "script-info args[1].value 는 50개 키(sam3_enable 포함). sam3_cn_module/sam3_cn_model 은 "
                 "Forge ControlNet 이 대소문자까지 그대로 찾는다(supported_preprocessors[name]·"
                 "controlnet_filename_dict[name]) — 앱은 기능 스냅샷의 /controlnet/module_list·model_list "
                 "표기로 맞추고, 모르면 CN_MODULES 정적 폴백('None')을 쓴다(P3).",
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
            comfy=("core/comfy_workflow_compiler.py:compile_sam3_mask_only",
                   "comfy_custom_nodes/ai_studio_forge_parity/sam3_nodes.py:ForgeNeoSAM3Detailer"),
            gaps=("P14: BatchView SAM3 필드·inpaint W/H 폴백 1024",))),
    "Anima Perturbation Guidance": _script(
        # shape: Attn Scale 최대 15→100(원본 scale 0~100 — wave B PAG-F), DCW·CWM·RDC·CNS 기본값·범위와 숨은
        # RDC 스위치 True(원본 입력 — wave C DCW-F·CNS-F). 픽스처는 둘 다 Forge 정지 중 소스 AST 로 맞췄다
        file="scripts/anima_safe_pag.py", form="positional", live_argc=62, shape="4e714c4bdfa5",
        ui_return="app_spec", api_reads="0-10,12-61", runtime_choices=(45,),
        app_title="core.anima_guidance:SCRIPT_PERTURBATION",
        app_spec=("core.anima_guidance:PERTURBATION_SPEC", "guid_"),
        api_note="idx11(auto_decay)은 visible=False 자리 유지용이라 API 가 읽지 않는다. idx45(CLIP-L)는 실행 "
                 "시점 목록. live_argc 57 은 v0.21.2 빌드 — 뒤 5개가 잘리고 idx56 'Auto' 가 SMC 를 켠다(나-5, P5).",
        classification=mapped(
            "core/anima_guidance.py:PERTURBATION_SPEC", "core/anima_guidance.py:build_alwayson",
            "ui/generator_generation.py:_apply_postprocess_chain",
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
                 "스텝의 DCW 를 생략한다",
            gaps=("P15: guid_cfg_mode 활성 키, SLG 배지, CLIP-L 없음 경고",
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
                 "건다(확장 SAM3 p2 가 DD 스크립트를 다시 돌린다). 단독 후처리에는 없다(Forge 단독 요청에 DD 인자 없음)",
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
            "ui/hand_reconstruction_actions.py:anima38",
            "core/sam_extra_notices.py:KEY_ANIMA38_STATUS",   # 'Anima38: off: …' 알림(P4)
            comfy=("core/comfy_workflow_compiler.py:_resolve_anima38_plan",
                   "comfy_custom_nodes/ai_studio_forge_parity/anima38_nodes.py:ForgeNeoAnima38V2Prompt"),
            note="앱은 인자 파싱·hand repair 전달·Comfy 컴파일에만 쓰고 Forge 페이로드에는 블록을 만들지 않는다",
            gaps=("P9: Bypass·부정 커넥터·v1 어댑터 UI 와 Forge 페이로드가 없다",))),
    "DoRA Inference Mode": _script(
        file="scripts/dora_infer_mode.py", form="positional_or_dict", live_argc=5, shape="027fc702edeb",
        ui_return=("enabled", "mode", "inserted", "weak_strength", "weak_scope"),
        arg_names=("enabled", "mode", "inserted", "weak_strength", "weak_scope"),
        api_note="위치 또는 dict. 라벨이 한국어라 dict(키 값 lycoris/forge_fp32/forge/no_magnitude, "
                 "keep/additive/skip/weak, attn/attn_mlp/all)로 보내야 한다. weak_strength 는 UI 0-1, API 0-2.",
        classification=deferred(
            "P8", "UI·페이로드가 없다. Forge 전역 상태라 앱 요청마다 stock 으로 돌아가 사용자 Forge 결과"
                  "(LyCORIS)와 달라진다 (M6, M7)")),
    "Anima VAE 2x (spacepxl decoder)": _script(
        file="scripts/anima_vae_2x.py", form="positional", live_argc=5, shape="e9431e0d6245",
        ui_return=("enabled", "vae_file", "mode", "blur_sigma", "renorm"), api_reads="0-4",
        runtime_choices=(1,),
        api_note="위치 인자 list 만 받는다 — dict 로 보내면 조용히 아무 일도 하지 않는다. arg1 은 파일 목록.",
        classification=deferred(HOLD, "M9 보류 — 12채널 VAE 가 설치돼 있지 않다. Comfy 노드는 있으나 "
                                      "컴파일러에 연결 안 됨")),
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
})

# ── SAM3 state 계약 (dict 형태라 위치 대신 키로 맞춘다) ─────────────────────────────
# Sam3Args 에는 없지만 process() 가 state 에서 따로 읽는 활성화 플래그.
SAM3_ACTIVATION_KEYS = ("sam3_enable", "enabled")
SAM3_ENABLE_LABEL = "Enable SAM3"      # script-info args[0].label
SAM3_LIVE_STATE_KEYS = 50              # script-info args[1].value 키 수 = Sam3Args 49 + sam3_enable

# ── Forge 옵션 (shared.opts.add_option, 라이브 14개) ───────────────────────────────
# 주의(나-7): override_settings 에 모르는 키가 있으면 Forge classic 은 KeyError 로 요청 전체를 실패시킨다.
OPTIONS = MappingProxyType({
    "sam3_unload_keep_in_ram": deferred("P10", "S5 — SAM3 를 CPU RAM 에 보관(3.4 GB). override 는 "
                                               "run_callbacks=False 라 이미 올라간 모델은 안 내려간다"),
    "sam3_ipa_duplicate_policy": deferred("P20", "R3 — 캐릭터 레퍼런스 IP-Adapter 삽입 블록 정책"),
    "sam3_anima38_keep_resident": deferred("P10", "M2 — Qwen3.5 커넥터 VRAM 6-8GB 상주(학습과 같은 GPU)"),
    "sam3_anima38_connector_fp32": deferred("P10", "M2 — 커넥터 fp32"),
    "sam3_anima38_connector_run_cache": deferred("P10", "M2 — 커넥터 실행 캐시"),
    "sam3_anima38_reference_ipa": deferred("P20", "R3 — 3.8B 에서 레퍼런스 IP-Adapter"),
    "sam3_anima_sparse_lora_forge_guess": deferred("P10", "M5 — sparse LoRA 순정 추측 변환(요청별 토글)"),
    "sam3_guidance_pag_prefix_dedup": deferred("P10", "G16 — v0.30 이전 결과를 비트 단위로 재현할 때만"),
    "sam3_guidance_seg_separable_blur": deferred("P10", "G16 — v0.30 이전 결과를 비트 단위로 재현할 때만"),
    "sam3_guidance_dave_pre_dd_sigma": deferred(
        "P10", "DAVE+Detail Daemon 우회(기본 켬). 끄면 원본 노드 조합처럼 DAVE 가 모든 스텝에 걸린다 — "
               "앱 Comfy 팩은 같은 기본값(guid_dave_pre_dd, 팩 1.4.1)"),
    "sam3_appearance_theme": ignored("N3 — Forge 화면 테마. 앱은 자체 디자인 토큰을 쓴다"),
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
})

# ── XYZ 축 (접두어별, 라이브 104개: [SAM3] 37, [Anima …] 51, [Anima Skim] 7, [Detail Daemon] 5, [DoRA] 4) ──
_XYZ_GUIDANCE = deferred("P18", "G17 — 앱 XYZ 가 가이던스 spec 키·인덱스를 바꿔 가며 돌릴 수 없다")
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
        "Rescale Mode", "SEG Blur Sigma", "SLG Block Indices", "SLG Enable", "SLG Scale", "Start Percent")},
    "[Anima APG]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Eta", "Momentum", "Norm Threshold")},
    "[Anima AdaptiveG]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Keep Every", "Skip After")},
    "[Anima CFG]": {**_XYZ_GUIDANCE, "labels": ("Base Mode", "Experimental Stack")},
    "[Anima CNS]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Gamma Power", "Gamma Scale", "Strength")},
    "[Anima CWM]": {**_XYZ_GUIDANCE, "labels": ("Alpha High", "Alpha Low", "Enable")},
    "[Anima DAVE]": {**_XYZ_GUIDANCE, "labels": ("Block Indices", "Enable", "Strength", "Tau")},
    "[Anima DCW]": {**_XYZ_GUIDANCE, "labels": ("Enable", "Lambda High", "Lambda Low")},
    "[Anima Mod]": {**_XYZ_GUIDANCE, "labels": ("Direction Weight", "Enable", "End Block", "Start Block")},
    "[Anima RDC]": {**_XYZ_GUIDANCE, "labels": ("Alpha HH", "Alpha LL", "Enable", "Tau")},
    "[Anima SMC]": {**_XYZ_GUIDANCE, "labels": ("Enable", "K", "Lambda", "Preset")},
    "[Anima Skim]": {**_XYZ_GUIDANCE, "labels": (
        "Disable Flipping Filter", "Enable", "End", "Flip At", "Full Skim Negative", "Skimming CFG", "Start")},
    "[Detail Daemon]": {**deferred("P18", "G17 — 앱 XYZ 가 가이던스 spec 키를 바꿔 가며 돌릴 수 없다. 'Amount' "
                                          "축 값은 확장 arg2 를 대신하고 앱 dd_amount 와 같은 노드 단위다(변환 없음)"),
                        "labels": ("Amount", "Bias", "Enable", "End", "Start")},
    "[DoRA]": {**deferred("P18", "M8 — DoRA 축은 값이 바뀔 때마다 LoRA 를 다시 합친다(바깥 루프에 둔다)"),
               "labels": ("Inference mode", "Inserted blocks", "Weak copy scope", "Weak copy strength")},
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
    "scripts/anima_3_8b.py": mapped("core/anima38.py:SCRIPT_NAME", gaps=("P9",)),
    "scripts/dora_infer_mode.py": deferred("P8", "SCRIPTS['DoRA Inference Mode']"),
    "scripts/anima_vae_2x.py": deferred(HOLD, "SCRIPTS['Anima VAE 2x (spacepxl decoder)'] — M9 보류"),
    "scripts/anima_lora_blocks.py": mapped(
        "core/sam_extra_notices.py:KEY_SPARSE_LORA_GUESS",
        note="N7 — 자동 훅(인자 0개)이라 페이로드는 없다. 이 스크립트가 남기는 infotext 'Anima sparse LoRA'(추측 변환)만 "
             "결과 정보 알림으로 읽는다(P4, SEMANTIC_PINS anima_sparse_lora_infotext_key)",
        gaps=("M4: 토글이 꺼져 건너뛴 부분 LoRA 는 gr.Warning 만 남아 API 응답에 흔적이 없다 — 앱은 알 수 없다. "
              "확장이 infotext 나 응답에 건너뜀을 남기기 전까지 보류",)),
    "scripts/anima_ref_poc.py": ignored("N9 — 디버그"),
    "scripts/lora_manager.py": ignored("N8 — 숨은 Gradio 브리지와 옵션 두 개(N2)"),
    "scripts/appearance_theme.py": ignored("N3 — Forge 화면 테마 옵션"),
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
    "sam3ext/anima38/": mapped("core/anima38.py:parse_args", gaps=("P9",),
                               note="Qwen3.5 커넥터 런타임 — 앱은 제목·인자만 안다"),
    "sam3ext/guidance/": mapped("core/anima_guidance.py:PERTURBATION_SPEC",
                                note="SMC_PRESET_NAMES 등 PAG 선택지의 출처",
                                gaps=("P17: 남은 Comfy 차이는 'Anima Perturbation Guidance' 항목의 P17 gaps",)),
    "sam3ext/anima_lora_blocks.py": ignored("M4 — Forge 쪽 28/40/52 블록 자동 리맵, 앱 페이로드 없음. "
                                            "Comfy sparse 규칙 차이는 P17", package="P17"),
    "sam3ext/dora_infer_mode.py": deferred("P8", "DoRA 병합 공식"),
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
    # javascript/
    "javascript/appearance_theme.js": ignored(_N6),
    "javascript/lora_manager.js": ignored(_N6 + " — 폴링·브리지 로직은 P6 참고", package="P6"),
    "javascript/notebook.js": ignored(_N6 + " — Notebook 화면(T7)"),
    "javascript/notebook_memo.js": ignored(_N6 + " — Forge Notebook 패널의 메모 탭. 저장소 계약은 ROUTES 의 "
                                           "/sam3-notebook/memos 가 본다"),
    "javascript/notebook_lanes.js": ignored(_N6 + " — N4 섹션 레이아웃"),
    "javascript/tipo_device.js": ignored(_N6 + " — TIPO 장치 표시(P19)", package="P19"),
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
     "meaning": "dict 로 보낼 때 빠진 키를 채우는 값 — core/anima38.py DEFAULT_SETTINGS 와 같아야 한다(P9)"},
    # 결과·생성 전 알림(P4) — core/sam_extra_notices.py 가 같은 값을 읽는다. 바뀌면 알림이 조용히 멈춘다.
    {"id": "anima38_status_key", "source": "ast", "file": "scripts/anima_3_8b.py", "name": "STATUS_KEY",
     "expected": "Anima38", "app": "core.sam_extra_notices:KEY_ANIMA38_STATUS",
     "meaning": "3.8B 상태 infotext 키('v2 bundle'·'v1 adapter'·'bypass'·'off: 이유') — 'off' 면 순정 Anima 로 생성됐다"},
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
                "loraManagerMessage 파서가 이 필드 이름·구분자(', ')·태그 형식을 믿는다"},
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
