# core/anima_guidance.py
"""Anima Guidance Suite — alwayson_scripts 인자 빌더. 순수 함수(테스트 가능), Qt 의존 없음.

sam-extra(forge_sam3_extension v0.21.2+)의 세 스크립트는 `process_before_every_sampling`에서
`args[i]`를 **위치(index)로만** 읽는다(`_arg(i, default)`). 따라서 순서가 곧 계약이고,
한 칸만 밀려도 조용히 엉뚱한 값이 들어간다. SAM3 Mask처럼 dict를 받아주지 않는다.

그래서 이 모듈은 스펙을 선언적으로 두고 그 순서대로만 배열을 만든다.
스펙 순서 = 확장 `ui()`의 `return [...]` 순서이며, `tests/test_anima_guidance.py`가
개수·순서·기본값을 고정한다. 확장을 업데이트하면 그 테스트가 먼저 깨지게 하는 게 목적.

주의: 확장은 **인덱스 안정성을 위해 새 인자를 뒤에 append**한다(주석으로 명시됨).
      중간에 끼워넣지 말 것.

Detail Daemon 은 원본 노드(Jonseed/ComfyUI-Detail-Daemon) 단위 그대로 저장하고 그대로 보낸다 —
변환·프리셋 없음. 자리만 남은 칸(고정 칸)은 항상 중립값을 보낸다(DETAIL_DAEMON_SPEC 주석).
"""

# 확장 스크립트 title() 값 = alwayson_scripts 키
SCRIPT_PERTURBATION = "Anima Perturbation Guidance"   # scripts/anima_safe_pag.py
SCRIPT_SKIMMED_CFG = "Anima Skimmed CFG"              # scripts/anima_skimmed_cfg.py
SCRIPT_DETAIL_DAEMON = "Anima Detail Daemon"          # scripts/anima_detail_daemon.py


# ── 값 강제 변환 ────────────────────────────────────────────────────────────
# 설정값은 Vue 위젯 프록시에서 오므로 대부분 문자열('true' / '0.75')이다.
# 확장이 잘못된 타입을 만나면 조용히 폴백하거나 전체가 죽으므로 여기서 확실히 맞춘다.

_FALSEY = ('0', 'false', 'no', 'off', 'none', '')


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
    if out != out or out in (float('inf'), float('-inf')):  # NaN/Inf
        return default
    return max(lo, min(hi, out))


def _as_int(value, default: int, lo: int, hi: int) -> int:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def _as_choice(value, default: str, choices: tuple) -> str:
    """대소문자/공백 무시하고 매칭, 실패 시 default. 반환은 항상 확장이 아는 표기."""
    raw = str(value if value is not None else '').strip().lower()
    for choice in choices:
        if choice.lower() == raw:
            return choice
    return default


def _as_text(value, default: str) -> str:
    if value is None:
        return default
    return str(value)


# ── 스펙 ────────────────────────────────────────────────────────────────────
# (key, kind, default, extra) — extra: 숫자는 (lo, hi), choice는 선택지 튜플
_B, _F, _I, _C, _T = 'bool', 'float', 'int', 'choice', 'text'
# 고정 칸: 위치 계약 때문에 자리는 지키지만 사용자 설정이 아니다 — 저장값을 읽지 않고 항상 default 를
# 보낸다. 위젯 프록시·default_settings()·Forge 가져오기에서 빠진다.
FIXED = _X = 'fixed'

# scripts/anima_safe_pag.py — ui() return 순서 (v0.21.2+ 기준 62개)
PERTURBATION_SPEC = (
    # 0-4 : PAG/SEG 본체
    ('guid_enabled',            _B, False,  None),
    ('guid_attn_method',        _C, 'PAG',  ('PAG', 'SEG', 'None')),
    # 원본 노드 scale 0~100 (iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201)
    ('guid_scale',              _F, 4.0,    (0.0, 100.0)),
    ('guid_legacy_strength',    _F, 0.75,   (0.0, 1.0)),
    ('guid_block_indices',      _T, '18',   None),
    # 5-7 : SLG
    ('guid_slg_on',             _B, False,  None),
    ('guid_slg_scale',          _F, 3.0,    (0.0, 15.0)),
    ('guid_slg_blocks',         _T, '18',   None),
    # 8-11 : 공통 스케줄
    ('guid_start_percent',      _F, 0.0,    (0.0, 1.0)),
    ('guid_end_percent',        _F, 0.7,    (0.0, 1.0)),
    ('guid_rescale',            _F, 0.20,   (0.0, 1.0)),
    ('guid_auto_decay',         _B, False,  None),   # 확장에서 visible=False (자리 유지용)
    # 12-16 : APG
    ('guid_apg_enabled',        _B, False,  None),
    ('guid_apg_eta',            _F, 0.0,    (-10.0, 10.0)),
    ('guid_apg_norm',           _F, 15.0,   (0.0, 50.0)),
    ('guid_apg_momentum',       _F, 0.0,    (-1.0, 1.0)),
    ('guid_apg_autooff',        _B, True,   None),
    # 17-19 : Adaptive Guidance
    ('guid_adg_enabled',        _B, False,  None),
    ('guid_adg_start',          _F, 0.5,    (0.0, 1.0)),
    ('guid_adg_interval',       _I, 0,      (0, 10)),
    # 20-21 : legacy attn / SEG sigma
    ('guid_legacy_attn',        _B, False,  None),
    ('guid_seg_sigma',          _F, 100.0,  (0.0, 10000.0)),
    # 22-23 : legacy CFG base 라디오
    ('guid_cfg_mode',           _C, 'Preserve incoming',
     ('Preserve incoming', 'APG', 'CWM', 'SMC', 'SMC + CWM')),
    ('guid_experimental_stack', _B, False,  None),
    # 24-27 : CWM / SMC 파라미터 — 원본 노드 입력 그대로(숫자만 옮김, GPL 코드는 복사하지 않는다):
    #   alpha_l/alpha_h 기본 0 범위 −1~2, smc_lambda 6 범위 0.5~30, smc_k 0.1 범위 0~5
    #   (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:675-706, :732-755).
    ('guid_cwm_alpha_low',      _F, 0.0,    (-1.0, 2.0)),
    ('guid_cwm_alpha_high',     _F, 0.0,    (-1.0, 2.0)),
    ('guid_smc_lambda',         _F, 6.0,    (0.5, 30.0)),
    ('guid_smc_k',              _F, 0.10,   (0.0, 5.0)),
    # 28-30 : DCW — lambda_l 0.05 범위 ±0.5, lambda_h 0.01 범위 ±0.3
    #   (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:637-668). 켜기 토글 기본 끔은 호스트 차이
    #   (항상 붙는 스크립트라 기본이 중립이어야 한다 — 원본 노드는 넣으면 켜진다).
    ('guid_dcw_enabled',        _B, False,  None),
    ('guid_dcw_lambda_low',     _F, 0.05,   (-0.5, 0.5)),
    ('guid_dcw_lambda_high',    _F, 0.01,   (-0.3, 0.3)),
    # 31-34 : DAVE — strength 0.30, tau 0.10 (둘 다 0~1). 블록 칸은 원본 마스크(dave_alpha.npz = 블록 8~18)
    #   대신이라 빈 칸은 '8-18' 로 보낸다(_BLANK_MEANS_DEFAULT) — 확장 옛 빌드는 빈 칸을 {18} 로 읽었다
    #   (origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:113-149).
    ('guid_dave_enabled',       _B, False,  None),
    ('guid_dave_strength',      _F, 0.30,   (0.0, 1.0)),
    ('guid_dave_tau',           _F, 0.10,   (0.0, 1.0)),
    ('guid_dave_blocks',        _T, '8-18', None),
    # 35-38 : CNS — strength 1.0 (0~1), gamma_power 0.5 (0.1~2), gamma_scale 2.0 (0.1~25)
    #   (origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:396-437 — 숫자만).
    ('guid_cns_enabled',        _B, False,  None),
    ('guid_cns_strength',       _F, 1.0,    (0.0, 1.0)),
    ('guid_cns_gamma_power',    _F, 0.5,    (0.1, 2.0)),
    ('guid_cns_gamma_scale',    _F, 2.0,    (0.1, 25.0)),
    # 39-41 : v0.13 append (구버전 인덱스 보존 목적)
    ('guid_official_strength',  _F, 0.75,   (0.0, 1.0)),
    ('guid_head_indices',       _T, '',     None),
    ('guid_rescale_mode',       _C, 'full', ('full', 'partial')),
    # 42-43 : SMC/CWM 독립 토글 append
    ('guid_smc_enabled',        _B, False,  None),
    ('guid_cwm_enabled',        _B, False,  None),
    # 44-55 : v0.20 Modulation Guidance append
    ('guid_mod_enabled',        _B, False,  None),
    ('guid_mod_clip_model',     _T, '',     None),
    ('guid_mod_weight',         _F, 3.0,    (-20.0, 20.0)),
    ('guid_mod_start_layer',    _I, 0,      (0, 63)),
    ('guid_mod_end_layer',      _I, -1,     (-1, 63)),
    ('guid_mod_base_source',    _C, 'Main positive', ('Main positive', 'Custom')),
    ('guid_mod_base_prompt',    _T, '',     None),
    ('guid_mod_positive_prompt', _T, 'masterpiece, best quality, highres', None),
    ('guid_mod_negative_source', _C, 'Main negative', ('Main negative', 'Custom')),
    ('guid_mod_negative_prompt', _T, 'worst quality, low quality', None),
    ('guid_mod_adapter_mode',   _C, 'Auto-download official',
     ('Auto-download official', 'Local file')),
    ('guid_mod_adapter_path',   _T, '',     None),
    # 56 : v0.21.2 이후 SMC preset append
    ('guid_smc_preset',         _C, 'Auto',
     ('Auto', 'SD1.5 / SD2', 'SDXL', 'SD3 / SD3.5', 'Flux',
      'Qwen-Image', 'Cosmos / Wan', 'Custom')),
    # 57-61 : explicit SMC master + RDC append. 이전 56개 인덱스는 그대로 유지한다.
    # RDC 는 원본에 켜기 스위치가 없다: rdc_tau 0(기본) = 끔, > 0 = 켬, 그리고 DCW 훅 안에서 돌아 dcw_enabled 가
    #   꺼지면 같이 꺼진다. 범위 tau 0~0.5, alpha_ll 0.03 (0~0.3), alpha_hh 0 (0~0.1)
    #   (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:757-815, :855-856 — 숫자·뜻만). 앱의 Enable RDC 스위치는
    #   저장된 사용자 값이라 남기되, 보내는 값은 원본 뜻으로 맞춘다(_derive_rdc — 켜짐 = 스위치·DCW·tau > 0).
    ('guid_smc_master_enabled', _B, False,  None),
    ('guid_rdc_enabled',        _B, False,  None),
    ('guid_rdc_tau',            _F, 0.0,    (0.0, 0.5)),
    ('guid_rdc_alpha_ll',       _F, 0.03,   (0.0, 0.3)),
    ('guid_rdc_alpha_hh',       _F, 0.0,    (0.0, 0.1)),
)

# 빈 문자열이면 스펙 기본값을 보내는 텍스트 칸 — DAVE 블록 칸은 원본 기본 마스크(블록 8~18)를 대신한다.
# 팩(guidance_dave.py: ``str(blocks or '').strip() or '8-18'``)과 같은 규칙.
_BLANK_MEANS_DEFAULT = frozenset({'guid_dave_blocks'})

# scripts/anima_skimmed_cfg.py — ui() return 순서 (7개)
# 원본 CFG_Skimming_Single_Scale_Pre_CFG 입력: skimming_cfg 7.0 (0~10 step .5), start 0 / end 1 / flip_at 0
#   (모두 0~1 step .01) (origin: Extraltodeus/Skimmed_CFG@d8300583:skimmed_CFG.py:5-6, :93-133).
# skimming_cfg 하한 −1 은 호스트 차이 — 원본 래퍼 노드(Clean Skim·Timed flip, :204-281)가 넘기는 −1(= 현재 CFG)을
#   스크립트 하나로 대신하기 때문이다. start/end/flip 은 σ 기준 %(원본 percent_to_sigma)이고 앱은 값을 그대로 보낸다.
SKIMMED_SPEC = (
    ('skim_enabled',                 _B, False, None),
    ('skim_skimming_cfg',            _F, 7.0,   (-1.0, 10.0)),
    ('skim_full_skim_negative',      _B, False, None),
    ('skim_disable_flipping_filter', _B, False, None),
    ('skim_start_percent',           _F, 0.0,   (0.0, 1.0)),
    ('skim_end_percent',             _F, 1.0,   (0.0, 1.0)),
    ('skim_flip_at',                 _F, 0.0,   (0.0, 1.0)),
)

# scripts/anima_detail_daemon.py — ui() return 순서 (14개)
# 값은 원본 노드 단위 그대로 저장하고 그대로 보낸다 — 앱은 변환하지 않는다(엔진이 ×0.1×cfg 를 곱한다).
# 범위·기본값 = origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:320-369
#   (DetailDaemonSamplerNode INPUT_TYPES: detail_amount ±5 step .01 기본 0.1, start .2, end .8, bias .5,
#    exponent 1 (0~10), start/end_offset 0 (±1), fade 0 (0~1), smooth True).
# Hires Pass(arg13, 맨 뒤 append)는 노드에 없는 Forge 전용 칸 — 기본 끔 = base 패스만, 켜면 hires 패스만
#   (origin: muerrilla/sd-webui-detail-daemon@19479998:scripts/detail_daemon.py:104, :276).
# 고정 칸(_X): 원본에 없는 손잡이라 자리만 지키고 늘 중립값을 보낸다 — preset(arg1, 확장이 읽지 않는 숨은 칸),
#   multiplier(arg10) 1.0, cfg_couple(arg12) True(= 원본의 ×cfg). 저장된 옛 dd_preset 등은 읽지 않는다.
# 범위는 다른 칸처럼 노드 범위로 자른다(amount ±5, offset ±1 — detail_daemon_node.py:326, :346, :350). 노드는
#   그 밖의 값을 받지 않는다(ComfyUI 입력 검사가 거부하고 숫자 위젯은 min/max 로 자른다). 옛 저장값(실효 ±1 을
#   노드 단위로 그대로 둔 것)은 모두 이 범위 안이라 잘리는 저장값은 없다.
DD_AMOUNT_MAX = 5.0          # 노드 detail_amount min/max
DD_PRESET_NEUTRAL = 'Custom'  # 확장 arg1 숨은 Textbox 의 값(읽지 않는 자리)
DETAIL_DAEMON_SPEC = (
    ('dd_enabled',      _B, False,    None),
    ('dd_preset',       _X, DD_PRESET_NEUTRAL, None),
    ('dd_amount',       _F, 0.10,     (-DD_AMOUNT_MAX, DD_AMOUNT_MAX)),
    ('dd_start',        _F, 0.2,      (0.0, 1.0)),
    ('dd_end',          _F, 0.8,      (0.0, 1.0)),
    ('dd_bias',         _F, 0.5,      (0.0, 1.0)),
    ('dd_exponent',     _F, 1.0,      (0.0, 10.0)),
    ('dd_start_offset', _F, 0.0,      (-1.0, 1.0)),
    ('dd_end_offset',   _F, 0.0,      (-1.0, 1.0)),
    ('dd_fade',         _F, 0.0,      (0.0, 1.0)),
    ('dd_multiplier',   _X, 1.0,      None),
    ('dd_smooth',       _B, True,     None),
    ('dd_cfg_couple',   _X, True,     None),
    ('dd_hires',        _B, False,    None),
)
DD_HIRES_INDEX = [key for key, *_rest in DETAIL_DAEMON_SPEC].index('dd_hires')   # 13

# 뒤에 append 된 칸이라 없는 빌드도 앞 칸의 위치 계약은 같은 스크립트 → 처음 append 된 인덱스.
# Forge 가져오기는 이보다 짧으면 거부하고, 이 인덱스부터 빠진 칸은 기본값으로 둔다(meta 에 알린다).
# Hires Pass 는 13개 인자 빌드(sam-extra v0.30, 4045adb)에 없다 — 그 빌드의 앞 13칸은 지금과 같은 뜻이다.
_APPEND_ONLY_FROM = {
    SCRIPT_DETAIL_DAEMON: DD_HIRES_INDEX,
}

SPECS = {
    SCRIPT_PERTURBATION: PERTURBATION_SPEC,
    SCRIPT_SKIMMED_CFG: SKIMMED_SPEC,
    SCRIPT_DETAIL_DAEMON: DETAIL_DAEMON_SPEC,
}

# 스크립트를 페이로드에 넣을지 결정하는 마스터 토글.
# 전부 꺼져 있으면 alwayson_scripts에 아예 넣지 않는다 — 확장을 건드리지 않아야
# "전부 끄면 Forge 결과 그대로"가 보장되고, 불필요한 hook 설치도 피한다.
_ACTIVATION_KEYS = {
    # guid_rdc_enabled 는 활성 키가 아니다: 원본 RDC 는 DCW 안에서만 돌아(_derive_rdc) 혼자서는 아무것도 켜지 못하고,
    # 켤 수 있는 조건(guid_dcw_enabled)은 이미 키다. 원본 동등성 확장(DCW-F)은 58 칸을 숨은 True 로 내놓아
    # 'Forge에서 가져오기' 뒤 늘 True 이므로, 키로 두면 모든 생성(Anima 가 아닌 체크포인트 포함)에 중립 인자뿐인
    # 스크립트가 붙고 sam-extra 가 없는 Forge 에서는 켜지 않은 기능 때문에 422 가 난다.
    # 이 표는 Vue 거울(frontend/src/utils/samExtraCapabilities.ts)과 같아야 한다.
    SCRIPT_PERTURBATION: (
        'guid_enabled', 'guid_slg_on', 'guid_apg_enabled', 'guid_adg_enabled',
        'guid_smc_enabled', 'guid_smc_master_enabled', 'guid_cwm_enabled',
        'guid_dcw_enabled',
        'guid_dave_enabled', 'guid_cns_enabled', 'guid_mod_enabled',
        'guid_experimental_stack',
    ),
    SCRIPT_SKIMMED_CFG: ('skim_enabled',),
    SCRIPT_DETAIL_DAEMON: ('dd_enabled',),
}


def _coerce(kind, value, default, extra, key=None):
    if kind == _X:
        return default
    if key in _BLANK_MEANS_DEFAULT and not str(value if value is not None else '').strip():
        return default
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
    """사용자 설정 키의 기본값 dict — UI 초기화/리셋·위젯 프록시용. 고정 칸(FIXED)은 빠진다."""
    out = {}
    for spec in SPECS.values():
        for key, kind, default, _extra in spec:
            if kind != _X:
                out[key] = default
    return out


def parse_forge_script_info(payload) -> tuple[dict, dict]:
    """Forge ``/sdapi/v1/script-info`` 응답에서 지원하는 Anima 값을 가져온다.

    Forge는 txt2img/img2img 항목을 각각 반환할 수 있으므로 txt2img를 우선한다.
    확장의 인자는 위치 계약이므로 앱 스펙보다 짧은 배열은 잘못 매핑하지 않고
    즉시 거부한다. 단 맨 뒤에 append 된 칸(_APPEND_ONLY_FROM — Detail Daemon Hires Pass)만
    빠진 빌드는 앞 칸의 뜻이 같으므로 있는 칸만 가져오고 빠진 칸은 앱 기본값으로 둔다
    (meta ``missing_trailing_args`` 에 키를 알린다). 반대로 새 Forge 버전이 뒤에
    append한 인자는 기존 인덱스를 보존하므로 무시하되 개수를 메타데이터로 알린다.
    """
    if not isinstance(payload, list):
        raise ValueError('Forge script-info 응답이 배열이 아닙니다.')

    by_name = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name') or '').strip().casefold()
        if name:
            by_name.setdefault(name, []).append(entry)

    settings = {}
    imported_scripts = []
    missing_scripts = []
    missing_trailing_args = []
    ignored_trailing_args = 0

    for title, spec in SPECS.items():
        candidates = by_name.get(title.casefold(), [])
        if not candidates:
            missing_scripts.append(title)
            continue

        # Forge 응답은 보통 txt2img(false), img2img(true) 순이지만 순서에
        # 의존하지 않는다. 문자열 boolean도 기존 coercion 규칙으로 처리한다.
        entry = min(candidates, key=lambda item: _as_bool(item.get('is_img2img'), False))
        args = entry.get('args')
        if not isinstance(args, list):
            raise ValueError(f'{title}: args가 배열이 아닙니다.')
        required = _APPEND_ONLY_FROM.get(title, len(spec))
        if len(args) < required:
            raise ValueError(
                f'{title}: Forge 인자 수 {len(args)}개가 앱 계약 {required}개보다 짧습니다.'
            )

        for index, (key, kind, default, extra) in enumerate(spec):
            if kind == _X:
                continue
            if index >= len(args):
                # append 전 빌드라 없는 뒤 칸 — 그 빌드에는 이 설정이 없으니 앱 기본값
                settings[key] = default
                missing_trailing_args.append(key)
                continue
            item = args[index]
            raw = item.get('value', default) if isinstance(item, dict) else item
            settings[key] = _coerce(kind, raw, default, extra, key)

        imported_scripts.append(title)
        ignored_trailing_args += max(0, len(args) - len(spec))

    if not imported_scripts:
        raise ValueError('Forge에서 지원되는 Anima 스크립트를 찾지 못했습니다.')

    return settings, {
        'imported_scripts': imported_scripts,
        'missing_scripts': missing_scripts,
        'missing_trailing_args': missing_trailing_args,
        'ignored_trailing_args': ignored_trailing_args,
    }


_PAG_INDEX = {key: index for index, (key, *_rest) in enumerate(PERTURBATION_SPEC)}


def _rdc_on(dcw_enabled: bool, rdc_enabled: bool, tau: float) -> bool:
    """원본 RDC 켜짐 = DCW 켜짐 · tau > 0 (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:757-760, :855-856 뜻 —
    tau 0 이면 끔, RDC 는 DCW 보정 안에서만 돈다). 앱의 Enable RDC 스위치(저장된 사용자 값)도 켜져 있어야 한다."""
    return bool(dcw_enabled) and bool(rdc_enabled) and tau > 0.0


def _derive_rdc(args: list) -> list:
    """보내는 RDC 칸(58 스위치, 59 tau)을 원본 뜻으로 맞춘다 — 확장 빌드마다 같은 그림이 나오게.

    - 옛 빌드(원본 동등성 전)는 58 스위치만 보고 DCW 와 무관하게 켠다 → 58 에 원본 켜짐을 보낸다.
    - 원본 동등성 빌드(DCW-F)는 ``dcw_enabled and tau > 0`` 로 켜고 58 은 끄는 쪽(veto)으로만 읽는다(자기 화면은
      숨은 True). 꺼짐이면 tau 도 0 으로 보내 58 을 읽지 않는 호출 경로에서도 같은 답이 나오게 한다
      (사용자가 스위치를 끈 채 저장한 tau — 예전 기본 0.15 — 가 DCW 를 켰을 때 RDC 를 켜지 않게).
    저장된 값은 건드리지 않는다. Comfy 컴파일러도 이 배열을 읽어 같은 값을 팩 스위트에 넘긴다."""
    i_on, i_tau = _PAG_INDEX['guid_rdc_enabled'], _PAG_INDEX['guid_rdc_tau']
    on = _rdc_on(args[_PAG_INDEX['guid_dcw_enabled']], args[i_on], args[i_tau])
    args[i_on] = on
    if not on:
        args[i_tau] = 0.0
    return args


def build_args(script_title: str, settings=None) -> list:
    """스크립트 하나의 위치 인자 배열 생성. 스펙에 없는 키는 무시되고 고정 칸은 늘 중립값이다.

    Perturbation 의 RDC 칸은 원본 켜짐 규칙으로 맞춰 보낸다(_derive_rdc)."""
    spec = SPECS.get(script_title)
    if spec is None:
        raise KeyError(f"알 수 없는 Anima 스크립트: {script_title!r}")
    settings = settings if isinstance(settings, dict) else {}
    args = [_coerce(kind, settings.get(key), default, extra, key)
            for key, kind, default, extra in spec]
    if spec is PERTURBATION_SPEC:
        _derive_rdc(args)
    return args


def is_script_active(script_title: str, settings=None) -> bool:
    """마스터 토글 중 하나라도 켜져 있는지."""
    settings = settings if isinstance(settings, dict) else {}
    return any(_as_bool(settings.get(key), False)
               for key in _ACTIVATION_KEYS.get(script_title, ()))


def build_alwayson(settings=None) -> dict:
    """켜져 있는 Anima 스크립트만 골라 alwayson_scripts 조각을 만든다.

    반환 예: {"Anima Skimmed CFG": {"args": [True, 7.0, False, False, 0.0, 1.0, 0.0]}}
    전부 꺼져 있으면 빈 dict.
    """
    settings = settings if isinstance(settings, dict) else {}
    out = {}
    for title in SPECS:
        if is_script_active(title, settings):
            out[title] = {"args": build_args(title, settings)}
    return out


def apply_to_payload(payload: dict, settings=None) -> dict:
    """payload['alwayson_scripts']에 Anima 스크립트를 병합하고 payload를 반환.

    이미 같은 키가 있으면 덮어쓰지 않는다(호출자가 명시 지정한 값 우선).
    """
    if not isinstance(payload, dict):
        return payload
    block = build_alwayson(settings)
    if not block:
        return payload
    scripts = payload.setdefault('alwayson_scripts', {})
    if not isinstance(scripts, dict):
        return payload
    for title, args in block.items():
        scripts.setdefault(title, args)
    return payload


def describe_active(settings=None) -> str:
    """로그/토스트용 짧은 요약. 예: 'PAG(4.0) + Skimmed CFG + Detail Daemon(0.1)'"""
    settings = settings if isinstance(settings, dict) else {}
    parts = []
    if _as_bool(settings.get('guid_enabled'), False):
        method = _as_choice(settings.get('guid_attn_method'), 'PAG', ('PAG', 'SEG', 'None'))
        if method != 'None':
            parts.append(f"{method}({_as_float(settings.get('guid_scale'), 4.0, 0.0, 100.0):g})")
    for key, label in (
        ('guid_slg_on', 'SLG'), ('guid_apg_enabled', 'APG'),
        ('guid_adg_enabled', 'Adaptive'),
    ):
        if _as_bool(settings.get(key), False):
            parts.append(label)
    if any(_as_bool(settings.get(key), False) for key in (
        'guid_smc_master_enabled', 'guid_smc_enabled',
    )):
        parts.append('SMC')
    # CWM·DCW·RDC·DAVE 는 원본이 실제로 거는 조건으로 적는다(보내는 값 = build_args 로 판정):
    #   CWM = 켜짐 · alpha 하나라도 ≠ 0, DCW = 켜짐 · (lambda ≠ 0 또는 RDC), RDC = _derive_rdc
    #   (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:855-858 — 뜻만),
    #   DAVE = 켜짐 · strength > 1e-3 (origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:163-166).
    args = build_args(SCRIPT_PERTURBATION, settings)

    def arg(key):
        return args[_PAG_INDEX[key]]

    rdc_on = bool(arg('guid_rdc_enabled'))
    if arg('guid_cwm_enabled') and (arg('guid_cwm_alpha_low') != 0.0 or arg('guid_cwm_alpha_high') != 0.0):
        parts.append('CWM')
    if arg('guid_dcw_enabled') and (arg('guid_dcw_lambda_low') != 0.0 or arg('guid_dcw_lambda_high') != 0.0
                                    or rdc_on):
        parts.append('DCW')
    if rdc_on:
        parts.append('RDC')
    if arg('guid_dave_enabled') and arg('guid_dave_strength') > 1e-3:
        parts.append('DAVE')
    for key, label in (
        ('guid_cns_enabled', 'CNS'),
        ('guid_mod_enabled', 'Modulation'), ('skim_enabled', 'Skimmed CFG'),
    ):
        if _as_bool(settings.get(key), False):
            parts.append(label)
    if _as_bool(settings.get('dd_enabled'), False):
        amount = _as_float(settings.get('dd_amount'), 0.10, -DD_AMOUNT_MAX, DD_AMOUNT_MAX)
        hires = ', Hires' if _as_bool(settings.get('dd_hires'), False) else ''
        parts.append(f"Detail Daemon({amount:g}{hires})")
    return ' + '.join(parts)


DD_HIRES_NOTE_OLD_EXTENSION = (
    'Detail Daemon: 연결된 sam-extra 가 Hires Pass(인자 13)를 모릅니다 — 이 확장은 base·hires 패스 '
    '모두에 적용합니다. 확장을 업데이트하세요.')


def detail_daemon_hires_note(settings=None, capabilities=None, *, comfyui=False):
    """Hires Pass 를 켰는데 그대로 적용되지 않을 백엔드면 경고 문구, 아니면 None.

    - Forge: 연결된 확장이 arg 13 을 모르면(Hires Pass 추가 전 빌드) 그 인자를 잘라 버리고 모든 패스에 적용한다 —
      사용자가 원한 'hires 패스만'(muerrilla detail_daemon.py:276) 이 아니다. ``capabilities`` 는
      SamExtraCapabilities 또는 None(모르면 경고하지 않는다).
    - ComfyUI(``comfyui=True``): 경고 없음 — 컴파일러가 dd_hires 로 패스를 고른다(DD-C, comfy_workflow_compiler
      ``_add_detail_daemon``). 남아 있는 Forge 기능 스냅샷은 이 백엔드와 무관하므로 보지 않는다.
    """
    settings = settings if isinstance(settings, dict) else {}
    if comfyui or not (is_script_active(SCRIPT_DETAIL_DAEMON, settings)
                       and _as_bool(settings.get('dd_hires'), False)):
        return None
    if getattr(capabilities, 'detail_daemon_hires', None) is not False:
        return None
    return DD_HIRES_NOTE_OLD_EXTENSION
