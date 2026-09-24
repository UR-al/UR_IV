"""T2I SAM3 ControlNet 13필드 — 위젯·저장 키·확장 인자 대응표 (Qt 비의존 순수 로직).

``ui/generator_ui_setup.py`` 가 ``_sam3_cn_*`` 프록시 13개를 만들고
``_build_sam3_settings`` 가 읽어 alwayson_scripts 로 보내지만, 예전엔 Vue 어디에도 이
widget id 를 바인딩한 곳이 없어 T2I SAM3 는 늘 CN off·기본값으로 나갔다. 게다가 설정
저장/복원(``_get_sam3_settings``/``_set_sam3_settings``)에도 cn 키가 없어서, UI 만 연결하면
재시작마다 값이 초기화됐다.

이 모듈이 13필드의 단일 표다:
  * 위젯 dict 키(``cn_enable``) = 저장 키 = ``'_sam3_' + 키`` 가 widget id
  * 확장 인자 키는 ``'sam3_' + 키`` (``core.sam3_args.SAM3_SPEC``)
  * 기본값·선택지는 SAM3_SPEC 에서 가져온다(두 벌로 갈라지지 않게)
  * 전처리기(module) 목록은 ``sam3_args.CN_MODULES`` 하나 — 프록시 items 로 Vue 에 보낸다
"""
from __future__ import annotations

from typing import Mapping

from core import sam3_args

# (위젯/저장 키, 프록시 종류) — 순서는 확장 SAM3 > ControlNet 아코디언과 같다.
CN_FIELDS: tuple[tuple[str, str], ...] = (
    ('cn_enable', 'check'),
    ('cn_override_external', 'check'),
    ('cn_model', 'text'),     # 설치 모델에 따라 달라 선택지 없는 자유 입력 (LineEditProxy)
    ('cn_module', 'combo'),
    ('cn_weight', 'text'),
    ('cn_guidance_start', 'text'),
    ('cn_guidance_end', 'text'),
    ('cn_pixel_perfect', 'check'),
    ('cn_control_mode', 'combo'),
    ('cn_resize_mode', 'combo'),
    ('cn_processor_res', 'text'),
    ('cn_threshold_a', 'text'),
    ('cn_threshold_b', 'text'),
)
CN_KEYS: tuple[str, ...] = tuple(key for key, _kind in CN_FIELDS)

_SPEC = {key: (kind, default, extra) for key, kind, default, extra in sam3_args.SAM3_SPEC}


def spec_key(key: str) -> str:
    """위젯/저장 키 → 확장 인자 키 (cn_weight → sam3_cn_weight)."""
    return f'sam3_{key}'


def widget_id(key: str) -> str:
    """위젯/저장 키 → Vue widget id (cn_weight → _sam3_cn_weight)."""
    return f'_sam3_{key}'


def _as_text(value) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float) and value.is_integer():
        return f'{value:.1f}'
    return str(value)


def default_values() -> dict:
    """저장 키 → 기본값(체크는 bool, 나머지는 위젯이 들고 있는 문자열)."""
    out = {}
    for key, kind in CN_FIELDS:
        default = _SPEC[spec_key(key)][1]
        out[key] = bool(default) if kind == 'check' else _as_text(default)
    return out


def choice_items() -> dict:
    """콤보 위젯의 선택지 — Vue 는 ``getProperty(id, 'items')`` 로 읽는다.

    cn_model 은 설치된 ControlNet 모델에 따라 달라 선택지를 두지 않는다(자유 입력).
    """
    return {
        'cn_module': list(sam3_args.CN_MODULES),
        'cn_control_mode': list(_SPEC[spec_key('cn_control_mode')][2]),
        'cn_resize_mode': list(_SPEC[spec_key('cn_resize_mode')][2]),
    }


def _read(proxy, kind: str):
    if kind == 'check':
        return bool(proxy.isChecked())
    if kind == 'combo':
        return proxy.currentText()
    return proxy.text()


def read_settings(widgets: Mapping) -> dict:
    """프록시 → 저장 dict (``_get_sam3_settings`` 에 합친다)."""
    out = {}
    for key, kind in CN_FIELDS:
        proxy = widgets.get(key)
        if proxy is not None:
            out[key] = _read(proxy, kind)
    return out


def apply_settings(widgets: Mapping, settings: Mapping | None) -> None:
    """저장 dict → 프록시. 없는 키(예전 저장 파일)는 확장 기본값으로 채운다."""
    saved = settings if isinstance(settings, Mapping) else {}
    defaults = default_values()
    for key, kind in CN_FIELDS:
        proxy = widgets.get(key)
        if proxy is None:
            continue
        value = saved.get(key, defaults[key])
        if kind == 'check':
            if isinstance(value, str):
                value = value.strip().lower() in ('true', '1', 'yes', 'on')
            proxy.setChecked(bool(value))
        else:
            proxy.setText(defaults[key] if value is None else _as_text(value))


def init_widgets(widgets: Mapping) -> None:
    """콤보 선택지를 먼저 채운 뒤 기본값을 넣는다(선택지보다 값이 먼저 가면 인덱스가 어긋난다)."""
    for key, items in choice_items().items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, 'addItems'):
            proxy.addItems(items)
    apply_settings(widgets, None)
