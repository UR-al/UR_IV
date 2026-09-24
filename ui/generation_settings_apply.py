"""프롬프트·생성 설정을 위젯(프록시)에 적용 — load_settings 와 프리셋 불러오기가 같이 쓴다.

예전 프리셋 불러오기(_apply_settings_dict)는 load_settings 와 다른 작은 매핑(15키)을 따로 두고
모델을 exact-match setCurrentText 로 골라, Forge 에서 저장한 'x.safetensors [hash]' 가
ComfyUI 에서는 조용히 무시되면서도 '로드됨' 이 떴다(audit #154). 이제 한 함수가
match_checkpoint 로 모델을 맞추고, 프리셋 경로(only_present=True)는 파일에 있는 키만 바꾸며
맞출 수 없던 모델을 경고 목록으로 돌려준다.

load_settings(only_present=False)는 예전처럼 없는 키를 기본값으로 채운다.

콤보 규칙(:func:`choose_combo`) — 목록이 **이미 찼으면** 항목에서 index 를 찾아 setCurrentIndex
하고, 못 찾으면 현재 선택을 그대로 두고(Vue 에 아무것도 보내지 않음) 경고한다. 예전엔 샘플러·
스케줄러·VAE 를 ComboBoxProxy.setText 로 넣어, 목록에 없는 값이면 Vue 는 프리셋 값을 보여 주고
Python 은 이전 선택으로 생성했다. setText 의 fallback(목록 도착 뒤 복원)은 목록이 **비어 있을
때만** 쓴다 — 시작 시 load_settings 가 API 목록보다 먼저 도는 경우다.
TE 칩은 Vue 가 지금 보여 주는 선택지(VueBridge.last_widget_property)로 거르고, 뺀 파일을 경고한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _text(value: Any) -> str:
    return '' if value is None else str(value)


def _combo_items(combo) -> list[str]:
    items = getattr(combo, '_items', None)
    if items is not None:
        return [str(item) for item in items]
    try:
        return [combo.itemText(i) for i in range(combo.count())]
    except Exception:
        return []


def match_checkpoint_index(value: str, items: Sequence[str]) -> int:
    """해시 유무·경로·대소문자가 달라도 같은 파일(core.model_names.match_checkpoint) — 모델·VAE."""
    from core.model_names import match_checkpoint

    return match_checkpoint(value, list(items))


def match_sampler_index(value: str, items: Sequence[str]) -> int:
    from core.preset_choice import match_sampler

    return match_sampler(value, items)


def match_scheduler_index(value: str, items: Sequence[str]) -> int:
    from core.preset_choice import match_scheduler

    return match_scheduler(value, items)


def match_plain_index(value: str, items: Sequence[str]) -> int:
    from core.preset_choice import match_choice

    return match_choice(value, items)


def choose_combo(combo, value: str, match: Callable[[str, Sequence[str]], int], *,
                 defer_when_empty: bool) -> bool | None:
    """``value`` 를 콤보에서 고른다 — True(골랐거나 fallback 으로 미룸) / False(목록에 없어 현재 선택
    유지) / None(목록이 비어 있고 미루지 않음).

    목록이 찬 콤보에는 setText 를 쓰지 않는다: ComboBoxProxy.setText 는 없는 값을 Vue 에 보내면서
    Python index 는 그대로 둬, 화면과 생성 요청이 서로 다른 값을 가리켰다.
    """
    items = _combo_items(combo)
    if not items:
        if defer_when_empty:
            combo.setText(value)
            return True
        return None
    index = match(value, items)
    if index < 0:
        return False
    combo.setCurrentIndex(index)
    return True


def combo_miss_message(label: str, value: str, combo, chosen: bool | None) -> str:
    """choose_combo 가 False/None 일 때의 경고 문구."""
    if chosen is None:
        return f"{label} '{value}'은(는) 백엔드 목록이 아직 없어 적용하지 못했습니다"
    kept = ''
    try:
        kept = _text(combo.currentText())
    except Exception:
        pass
    message = f"{label} '{value}'을(를) 현재 백엔드 목록에서 찾지 못했습니다"
    return f"{message} — '{kept}' 유지" if kept else message


def apply_combo_value(combo, value: str, label: str, match: Callable[[str, Sequence[str]], int],
                      warnings: list[str], *, defer_when_empty: bool,
                      unavailable: list[str] | None = None) -> None:
    """빈 값은 건너뛰고, 목록에 없는 값은 ``warnings`` 에 문구를 더한다.

    목록이 비어 적용하지 못한 콤보는 ``unavailable`` 이 있으면 거기에 이름만 모은다 — 연결 전
    프리셋 불러오기의 경고를 항목마다가 아니라 한 줄로 알리려고.
    """
    if not value or combo is None:
        return
    chosen = choose_combo(combo, value, match, defer_when_empty=defer_when_empty)
    if chosen is None and unavailable is not None:
        unavailable.append(label)
    elif chosen is not True:
        warnings.append(combo_miss_message(label, value, combo, chosen))


def unavailable_message(labels: Sequence[str]) -> str:
    return (f"백엔드 목록이 아직 없어 {', '.join(labels)}을(를) 적용하지 못했습니다"
            " — 백엔드에 연결한 뒤 다시 불러오세요")


def _te_choices(host) -> list[str] | None:
    """Vue 가 지금 보여 주는 TE 칩 선택지. 모르면 None — 연결 때 on_webui_info_loaded 가 다시 거른다.

    연결이 끊겨 빈 목록을 보낸 상태(_backend_connected=False)도 '모름'이다 — 빈 목록으로 거르면
    다음 연결에서 쓸 수 있는 TE 까지 지워진다.
    """
    getter = getattr(getattr(host, 'vue_bridge', None), 'last_widget_property', None)
    if not callable(getter):
        return None
    try:
        items = getter('te_main_input', 'items')
    except Exception:
        return None
    if not isinstance(items, list):
        return None
    names = [str(item) for item in items if str(item or '')]
    if not names and not getattr(host, '_backend_connected', False):
        return None
    return names


def apply_prompt_settings(host, settings: Mapping[str, Any], *, only_present: bool = False) -> None:
    """캐릭터 수·캐릭터·작품·작가·선행/본문/후행/네거티브·로컬 제외 칸."""
    fields = (
        ('char_count', 'char_count_input', False),
        ('character', 'character_input', False),
        ('copyright', 'copyright_input', False),
        ('artist', 'artist_input', True),
        ('prefix_prompt', 'prefix_prompt_text', True),
        ('main_prompt', 'main_prompt_text', True),
        ('suffix_prompt', 'suffix_prompt_text', True),
        ('negative_prompt', 'neg_prompt_text', True),
        ('exclude_prompt_local', 'exclude_prompt_local_input', True),
    )
    for key, attr, plain in fields:
        if only_present and key not in settings:
            continue
        widget = getattr(host, attr, None)
        if widget is None:
            continue
        value = _text(settings.get(key, ''))
        if plain and hasattr(widget, 'setPlainText'):
            widget.setPlainText(value)
        else:
            widget.setText(value)


#: (설정 키, 위젯 속성, 베이스 템플릿 속성) — 랜덤·검색 적용·자동화 사이클(apply_prompt_from_data)은
#: 매번 선행/후행/네거티브 칸을 ``base_*`` 템플릿으로 되돌린다(조건부 태그 누적 방지·와일드카드 보존).
BASE_PROMPT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ('prefix_prompt', 'prefix_prompt_text', 'base_prefix_prompt'),
    ('suffix_prompt', 'suffix_prompt_text', 'base_suffix_prompt'),
    ('negative_prompt', 'neg_prompt_text', 'base_neg_prompt'),
)


def sync_base_prompts(host, settings: Mapping[str, Any]) -> None:
    """``settings`` 에 있는 선행/후행/네거티브 키만 방금 적용한 칸의 글을 ``base_*`` 템플릿으로 삼는다.

    프리셋 불러오기는 is_programmatic_change 를 세운 채 칸을 바꾸므로 on_base_prompts_changed 가
    템플릿을 갱신하지 않는다 — 그대로 두면 다음 랜덤 프롬프트·자동화가 프리셋 이전 네거티브·선행으로
    되돌렸다. on_base_prompts_changed 처럼 세 칸을 다 복사하지 않는 이유: 프리셋에 없는 칸엔 지난
    사이클의 조건부 태그·해석된 와일드카드가 들어 있을 수 있어, 그 글이 템플릿으로 굳으면 안 된다.
    """
    for key, attr, base_attr in BASE_PROMPT_FIELDS:
        if key not in settings or not hasattr(host, base_attr):
            continue
        widget = getattr(host, attr, None)
        if widget is not None:
            setattr(host, base_attr, widget.toPlainText())


def preset_prompt_templates(host, settings: Mapping[str, Any]) -> dict[str, Any]:
    """프리셋 저장용 설정 — 선행/후행/네거티브는 칸의 글 대신 ``base_*`` 템플릿.

    랜덤·자동화 사이클(apply_prompt_from_data)을 한 번 돌면 세 칸엔 해석된 와일드카드와 조건부
    태그가 들어간다. 그 글을 프리셋으로 저장하면 불러올 때 :func:`sync_base_prompts` 가 템플릿으로
    굳혀, 와일드카드는 더 이상 새로 뽑히지 않고 조건부 태그는 조건이 맞지 않아도 남았다. 템플릿은
    사용자가 칸을 고칠 때마다 따라가므로(on_base_prompts_changed) 사용자가 입력한 글 그대로다.
    템플릿 속성이 없는 호스트는 칸의 글 그대로.
    """
    out = dict(settings)
    for key, _attr, base_attr in BASE_PROMPT_FIELDS:
        if key in out and hasattr(host, base_attr):
            out[key] = _text(getattr(host, base_attr))
    return out


def apply_generation_settings(host, settings: Mapping[str, Any], *, only_present: bool = False) -> list[str]:
    """모델·VAE/TE·샘플러·스텝·해상도·Hires·ADetailer·SAM3·Anima 가이던스. 경고 목록을 돌려준다."""
    warnings: list[str] = []
    unavailable: list[str] = []   # 목록이 비어 적용 못 한 콤보 — 마지막에 한 줄로 알린다

    def has(key: str) -> bool:
        return not only_present or key in settings

    def line(attr: str, key: str, default: str, sync: bool = False) -> None:
        if not has(key):
            return
        widget = getattr(host, attr)
        widget.setText(_text(settings.get(key, default)))
        if sync:
            host._sync_slider(widget)

    # 모델 — Forge 는 "[hash]" 를 붙이고 ComfyUI 는 안 붙인다. 목록이 비었으면 적용하지 않는다
    # (연결 때 restore_backend_combos 가 디스크 값을 고른다 — 예전 load_settings 와 같음).
    if has('model'):
        apply_combo_value(host.model_combo, _text(settings.get('model', '')), '모델',
                          match_checkpoint_index, warnings, defer_when_empty=False, unavailable=unavailable)

    # VAE — 목록이 비어 있으면 setText fallback 으로 기억했다가 addItems 때 복원한다(시작 시).
    # 목록이 찼으면 같은 파일(해시·경로 무시)을 고르고, 없으면 현재 선택 유지 + 경고.
    if hasattr(host, 'vae_main_combo') and has('vae_main'):
        apply_combo_value(host.vae_main_combo, _text(settings.get('vae_main', '')), 'VAE',
                          match_checkpoint_index, warnings, defer_when_empty=True)
    # TE — 지금 선택지에 없는 파일은 forge_additional_modules 로 가지 않게 뺀다(연결 시 필터와 같은 규칙).
    if hasattr(host, 'te_main_input') and has('te_main'):
        te_text = _text(settings.get('te_main', ''))
        choices = _te_choices(host) if te_text else None
        if choices is not None:
            from core.main_module_choices import resolve_te_selection

            te_text, dropped = resolve_te_selection(te_text, choices)
            if dropped:
                names = ', '.join(f"'{name}'" for name in dropped)
                warnings.append(f"텍스트 인코더 {names}을(를) 현재 목록에서 찾지 못해 뺐습니다")
        host.te_main_input.setText(te_text)

    # 샘플러/스케줄러 — 목록 fetch 보다 먼저 오면(시작 시) fallback 으로 복원, 목록이 찼으면
    # Forge 표기 ↔ ComfyUI 이름 별칭까지 맞춰 고르고, 없으면 현재 선택 유지 + 경고.
    for attr, key, label, match in (('sampler_combo', 'sampler', '샘플러', match_sampler_index),
                                    ('scheduler_combo', 'scheduler', '스케줄러', match_scheduler_index)):
        if has(key):
            apply_combo_value(getattr(host, attr), _text(settings.get(key, '')), label, match,
                              warnings, defer_when_empty=True)

    line('steps_input', 'steps', '25', sync=True)
    line('cfg_input', 'cfg', '7.0', sync=True)
    line('shift_input', 'shift', '0', sync=True)
    line('seed_input', 'seed', '-1')
    line('width_input', 'width', '1024')
    line('height_input', 'height', '1024')

    # 랜덤 해상도
    if has('random_res_enabled'):
        host.random_res_check.setChecked(bool(settings.get('random_res_enabled', False)))
    if 'random_resolutions' in settings and isinstance(settings['random_resolutions'], list):
        host.random_resolutions = settings['random_resolutions']

    # Hires.fix
    if has('hires_enabled'):
        host.hires_options_group.setChecked(bool(settings.get('hires_enabled', False)))
    # Hires 콤보는 목록이 비었으면 적용하지 않는다(연결 때 restore_backend_combos 가 디스크 값을 고른다).
    if has('hires_upscaler'):
        apply_combo_value(host.upscaler_combo, _text(settings.get('hires_upscaler', '')), '업스케일러',
                          match_plain_index, warnings, defer_when_empty=False, unavailable=unavailable)
    line('hires_steps_input', 'hires_steps', '0', sync=True)
    line('hires_denoising_input', 'hires_denoising', '0.4', sync=True)
    line('hires_scale_input', 'hires_scale', '2.0', sync=True)
    line('hires_cfg_input', 'hires_cfg', '0', sync=True)
    if has('hires_checkpoint'):
        apply_combo_value(host.hires_checkpoint_combo, _text(settings.get('hires_checkpoint', '')),
                          'Hires 체크포인트', match_checkpoint_index, warnings, defer_when_empty=False,
                          unavailable=unavailable)
    if has('hires_sampler'):
        apply_combo_value(host.hires_sampler_combo, _text(settings.get('hires_sampler', '')),
                          'Hires 샘플러', match_sampler_index, warnings, defer_when_empty=False,
                          unavailable=unavailable)
    if has('hires_scheduler'):
        apply_combo_value(host.hires_scheduler_combo, _text(settings.get('hires_scheduler', '')),
                          'Hires 스케줄러', match_scheduler_index, warnings, defer_when_empty=False,
                          unavailable=unavailable)
    if has('hires_prompt'):
        host.hires_prompt_text.setPlainText(_text(settings.get('hires_prompt', '')))
    if has('hires_neg_prompt'):
        host.hires_neg_prompt_text.setPlainText(_text(settings.get('hires_neg_prompt', '')))

    # ADetailer
    if has('adetailer_enabled'):
        host.adetailer_group.setChecked(bool(settings.get('adetailer_enabled', False)))
    if has('adetailer_slot1_enabled'):
        host.ad_slot1_group.setChecked(bool(settings.get('adetailer_slot1_enabled', False)))
    if has('adetailer_slot2_enabled'):
        host.ad_slot2_group.setChecked(bool(settings.get('adetailer_slot2_enabled', False)))
    # 슬롯 콤보는 목록이 비었으면 fallback 으로 미룬다(_set_slot_settings) — 연결 전 저장이 그 값을
    # 지키고 연결 때 restore_backend_combos 가 고른다. 그래서 '목록 없음' 목록에 넣지 않는다.
    for slot, attr in (('1', 's1_widgets'), ('2', 's2_widgets')):
        slot_settings = settings.get(f'adetailer_slot{slot}')
        if isinstance(slot_settings, Mapping):
            misses = host._set_slot_settings(getattr(host, attr), slot_settings) or []
            warnings.extend(f'ADetailer {slot}: {miss}' for miss in misses)

    # SAM3
    if hasattr(host, 'sam3_group') and has('sam3_enabled'):
        host.sam3_group.setChecked(bool(settings.get('sam3_enabled', False)))
    if hasattr(host, 'sam3_widgets') and isinstance(settings.get('sam3_settings'), Mapping):
        misses = host._set_sam3_settings(host.sam3_widgets, settings['sam3_settings']) or []
        warnings.extend(f'SAM3: {miss}' for miss in misses)

    # Anima 가이던스
    if hasattr(host, 'anima_guidance_widgets') and isinstance(settings.get('anima_guidance_settings'), Mapping):
        host._set_anima_guidance_settings(host.anima_guidance_widgets, settings['anima_guidance_settings'])

    if unavailable:
        warnings.append(unavailable_message(unavailable))
    return warnings
