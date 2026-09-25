# core/sam3_args.py
"""SAM3 Mask 확장 인자 빌더 — 순수 함수(테스트 가능), Qt 의존 없음.

지금까지 SAM3 state dict를 만드는 코드가 두 군데에 거의 똑같이 복사돼 있었다
(`ui/generator_generation.py::_build_sam3_settings`, `backends/webui_backend.py::
_build_sam3_script_state`). 한쪽만 고치면 t2i와 배치 결과가 갈리므로 여기로 합친다.

확장(`sam3ext/args.py::Sam3Args`)은 `extra=Extra.forbid`이라 **모르는 키가 하나라도
있으면 검증이 통째로 실패하고 SAM3가 조용히 꺼진다.** 그래서 스펙에 있는 키만 내보낸다.

`sam3_enable`/`enabled`은 Sam3Args에 없는 키지만, 확장 `process()`가 Sam3Args에
넘기기 전에 별도로 읽는 활성화 플래그다(`!sam3.py`의 state.get("sam3_enable", ...)).
따라서 state dict에는 넣되 스펙(SAM3_SPEC)과는 분리해 둔다.
"""

# 확장 scripts/!sam3.py 의 title() = SAM3_NAME
SCRIPT_SAM3 = "SAM3 Mask"

_FALSEY = ('0', 'false', 'no', 'off', 'none', '')

_B, _F, _I, _C, _T = 'bool', 'float', 'int', 'choice', 'text'


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in _FALSEY


def _as_float(value, default: float, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float('inf'), float('-inf')):
        return default
    return max(lo, min(hi, out))


def _as_int(value, default: int, lo: int, hi: int) -> int:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def _as_choice(value, default: str, choices: tuple) -> str:
    raw = str(value if value is not None else '').strip().lower()
    for choice in choices:
        if choice.lower() == raw:
            return choice
    return default


def _as_text(value, default: str) -> str:
    return default if value is None else str(value)


# ── 스펙 ────────────────────────────────────────────────────────────────────
# sam3ext/args.py::Sam3Args 와 1:1. 기본값은 확장 **UI** 기준으로 맞춘다.
# (확장의 API 경로 `_xyz_or(...)` 기본값과 다른 항목이 있어서 명시 전송이 중요하다:
#   - sam3_unload_after : UI True / API False → 안 보내면 인페인트 내내 3.5GB 상주 → OOM
#   - sam3_inpainting_fill : UI 'original' / API 'latent noise')
_INT_MAX = 1 << 30

SAM3_SPEC = (
    ('sam3_mode',                       _C, 'Inpaint',    ('Mask only', 'Inpaint')),
    ('sam3_mask_mode',                  _C, 'Individual', ('Combined', 'Individual')),
    ('sam3_prompt',                     _T, 'face',       None),
    ('sam3_exclude_prompt',             _T, '',           None),
    ('sam3_inpaint_prompt',             _T, '',           None),
    ('sam3_negative_prompt',            _T, '',           None),
    ('sam3_threshold',                  _F, 0.4,          (0.0, 1.0)),
    ('sam3_mask_dilation',              _I, 0,            (0, _INT_MAX)),
    ('sam3_mask_hull',                  _B, False,        None),
    ('sam3_mask_outline_px',            _I, 0,            (0, _INT_MAX)),
    ('sam3_checkpoint',                 _T, 'sam3.pt',    None),
    # 확장 UI 기본은 'auto'지만 앱은 예전부터 'cuda' 고정 — 동작 보존을 위해 유지.
    ('sam3_device',                     _T, 'cuda',       None),
    ('sam3_mask_blur',                  _I, 4,            (0, _INT_MAX)),
    ('sam3_denoising_strength',         _F, 0.4,          (0.0, 1.0)),
    ('sam3_inpainting_fill',            _C, 'original',
     ('fill', 'original', 'latent noise', 'latent nothing')),
    ('sam3_inpaint_only_masked',        _B, True,         None),
    ('sam3_inpaint_only_masked_padding', _I, 32,          (0, _INT_MAX)),
    ('sam3_use_inpaint_width_height',   _B, False,        None),
    ('sam3_inpaint_width',              _I, 512,          (64, 8192)),
    ('sam3_inpaint_height',             _I, 512,          (64, 8192)),
    ('sam3_use_steps',                  _B, False,        None),
    ('sam3_steps',                      _I, 28,           (1, 1000)),
    ('sam3_use_cfg_scale',              _B, False,        None),
    ('sam3_cfg_scale',                  _F, 7.0,          (0.0, 100.0)),
    ('sam3_use_sampler',                _B, False,        None),
    ('sam3_sampler',                    _T, 'Use same sampler',   None),
    ('sam3_use_scheduler',              _B, False,        None),
    ('sam3_scheduler',                  _T, 'Use same scheduler', None),
    ('sam3_use_seed',                   _B, False,        None),
    ('sam3_seed',                       _I, -1,           (-1, 2 ** 32 - 1)),
    ('sam3_use_noise_multiplier',       _B, False,        None),
    ('sam3_noise_multiplier',           _F, 1.0,          (0.0, 2.0)),
    ('sam3_restore_face',               _B, False,        None),
    ('sam3_preview_overlay',            _B, False,        None),
    ('sam3_save_artifacts',             _B, True,         None),
    ('sam3_unload_after',               _B, True,         None),
    # ── ControlNet 주입 (sam3_mode == 'Inpaint' + sd_forge_controlnet 로드 시에만 유효)
    ('sam3_cn_enable',                  _B, False,        None),
    ('sam3_cn_override_external',       _B, False,        None),
    ('sam3_cn_model',                   _T, 'None',       None),
    ('sam3_cn_module',                  _T, 'inpaint_only', None),   # Anima LLLite 모델이면 build_state 가 'None' 으로 (sam3_cn_names.guard_lllite_module)
    ('sam3_cn_weight',                  _F, 1.0,          (0.0, 2.0)),
    ('sam3_cn_guidance_start',          _F, 0.0,          (0.0, 1.0)),
    ('sam3_cn_guidance_end',            _F, 1.0,          (0.0, 1.0)),
    ('sam3_cn_pixel_perfect',           _B, True,         None),
    ('sam3_cn_control_mode',            _C, 'Balanced',
     ('Balanced', 'My prompt is more important', 'ControlNet is more important')),
    ('sam3_cn_resize_mode',             _C, 'Crop and Resize',
     ('Just Resize', 'Crop and Resize', 'Resize and Fill')),
    ('sam3_cn_processor_res',           _I, 512,          (0, _INT_MAX)),
    ('sam3_cn_threshold_a',             _F, -1.0,         (-1e9, 1e9)),
    ('sam3_cn_threshold_b',             _F, -1.0,         (-1e9, 1e9)),
)

SAM3_KEYS = tuple(key for key, _k, _d, _e in SAM3_SPEC)

# CN 전처리기(module) **정적 폴백** 목록 — 연결된 Forge 의 라이브 목록(/controlnet/module_list,
# 기능 스냅샷)을 모를 때만 쓴다(core/sam3_cn_names.py). Forge 는 supported_preprocessors[name] 로
# 대소문자까지 그대로 찾으므로 '없음'은 반드시 'None' 이다 — 예전 소문자 'none' 은 KeyError 로
# SAM3 패스를 실패시켰다(P3). tests/test_sam3_cn_names.py 가 녹화된 라이브 목록과 대조한다.
CN_MODULES = (
    'inpaint_only', 'inpaint_only+lama', 'inpaint_global_harmonious',
    'tile_resample', 'tile_colorfix', 'tile_colorfix+sharp',
    'depth_midas', 'depth_zoe', 'depth_anything',
    'openpose', 'openpose_full', 'openpose_hand',
    'lineart_realistic', 'lineart_anime', 'lineart_coarse',
    'canny', 'softedge_hed', 'scribble_pidinet', 'None',
)


def _coerce(kind, value, default, extra):
    if kind == _B:
        return _as_bool(value, default)
    if kind == _F:
        return _as_float(value, default, extra[0], extra[1])
    if kind == _I:
        return _as_int(value, default, extra[0], extra[1])
    if kind == _C:
        return _as_choice(value, default, extra)
    return _as_text(value, default)


def default_settings() -> dict:
    return {key: default for key, _k, default, _e in SAM3_SPEC}


def build_state(settings=None, *, prompt: str = '', negative_prompt: str = '',
                capabilities=None) -> dict:
    """SAM3 alwayson state dict 생성.

    prompt/negative_prompt: 인페인트 프롬프트가 비어 있을 때 쓸 폴백(부모 생성 프롬프트).
    확장도 빈 문자열이면 부모 프롬프트를 쓰지만, 배치/Refine처럼 부모 프롬프트 자체가
    비어 있는 경로가 있어서 여기서 명시적으로 채운다.

    capabilities: sam-extra 기능 스냅샷(``SamExtraCapabilities`` | None). ControlNet 전처리기·모델
    이름의 대소문자를 연결된 Forge 의 라이브 목록에 맞춘다(모르면 정적 목록 + 경고 한 번).
    네트워크는 쓰지 않는다 — core/sam3_cn_names.py.
    """
    settings = settings if isinstance(settings, dict) else {}
    state = {key: _coerce(kind, settings.get(key), default, extra)
             for key, kind, default, extra in SAM3_SPEC}

    from core.sam3_cn_names import guard_lllite_module, normalize_state as _normalize_cn_names
    _normalize_cn_names(state, capabilities)
    # Anima LLLite 는 원본처럼 사용자가 준 제어 이미지를 받는다 — Tile & Repair 면 전처리기 None, 그 밖의
    # Anima LLLite 는 inpaint_* 만 None(lineart·canny 는 그대로). 확장 inject_controlnet_unit 과 같은 규칙
    # (저장값·기본 inpaint_only 보정).
    guard_lllite_module(state)

    # detect prompt는 비면 확장이 'face'로 되돌리므로 여기서도 동일하게 보정
    if not str(state['sam3_prompt']).strip():
        state['sam3_prompt'] = 'face'

    if not str(state['sam3_inpaint_prompt']).strip() and prompt:
        state['sam3_inpaint_prompt'] = prompt
    if not str(state['sam3_negative_prompt']).strip() and negative_prompt:
        state['sam3_negative_prompt'] = negative_prompt

    # 체크포인트는 절대 경로로 — 관리형 Forge(--data-dir) 의 확장은 data/models 만 보는데
    # 파일은 사용자의 Forge models/sam3 에 있다. 절대 경로는 확장이 그대로 쓴다.
    try:
        from core.forge_modules import resolve_sam3_checkpoint
        state['sam3_checkpoint'] = resolve_sam3_checkpoint(state['sam3_checkpoint'])
    except Exception:
        pass

    # 활성화 플래그 — Sam3Args 스키마 밖의 키. 확장 process()가 따로 읽는다.
    state['sam3_enable'] = True
    state['enabled'] = True
    return state


def build_alwayson(settings=None, *, prompt: str = '', negative_prompt: str = '',
                   capabilities=None) -> dict:
    """{"SAM3 Mask": {"args": [state]}} 조각 생성."""
    return {SCRIPT_SAM3: {"args": [
        build_state(settings, prompt=prompt, negative_prompt=negative_prompt,
                    capabilities=capabilities)]}}


def apply_to_payload(payload: dict, settings=None, *, capabilities=None) -> dict:
    """payload에 SAM3를 병합. 프롬프트 폴백은 payload에서 읽는다."""
    if not isinstance(payload, dict):
        return payload
    scripts = payload.setdefault('alwayson_scripts', {})
    if not isinstance(scripts, dict):
        return payload
    scripts.setdefault(SCRIPT_SAM3, build_alwayson(
        settings,
        prompt=str(payload.get('prompt', '') or ''),
        negative_prompt=str(payload.get('negative_prompt', '') or ''),
        capabilities=capabilities,
    )[SCRIPT_SAM3])
    return payload
