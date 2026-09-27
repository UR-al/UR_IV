# ui/aux_pass_snapshot.py
"""보조 패스(Refine·단독/배치 SAM3·단독/배치 ADetailer)에 메인 생성 설정을 싣는다 — GUI 스레드에서만(P7).

- ``attach_sampling_snapshot(host, settings)``  클릭한 순간의 T2I 패널 샘플링 블록(NegPiP·가이던스·Anima38·DoRA)을
  봉투(core/alwayson_propagation.envelope)로 settings **사본**에 싣는다. 워커는 settings 를 그대로 백엔드에 넘기고,
  백엔드가 요청 직전에 패스·백엔드 규칙과 기능 스냅샷으로 걸러 넣는다(WebUIBackend._propagate,
  ComfyUIBackend._standalone_detail). 봉투 키는 Forge/Comfy 요청 JSON 으로 나가지 않는다.
  출처는 '클릭한 순간의 T2I 패널'이다 — Forge 생성 안의 SAM3 패스·🎯 퀵 버튼과 같은 의미(확장 Refine 패널은
  img2img 탭 초기값이라 다르다). 위젯은 이 스레드에서만 읽고, 봉투는 JSON 원시값의 깊은 복사라 워커와 공유하는
  가변 상태가 없다. 배치는 클릭 때 한 번 찍어 모든 장이 같은 블록을 받는다.
- ``inherit_refine_prompts(host, settings)``  Refine 이 메인 프롬프트를 T2I 에서 물려받을 때 LoRA 스택도 함께 붙인다
  (사용자 결정 D2 — Forge Refine 패널은 LoRA 태그가 든 t2i 프롬프트를 물려받는다. 붙이지 않으면 전달한 DoRA 가
  효과가 없다). 스택은 T2I 페이로드와 같은 규칙(ui/generator_generation.py 의 append_lora_stack_to_prompt)이고
  Krea2 는 붙이지 않는다. Forge·Comfy 가 같은 settings 를 쓴다.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core import alwayson_propagation as ap

logger = logging.getLogger(__name__)

SOURCE_T2I_PANEL = "t2i_panel"


def _is_krea2(host) -> bool:
    probe = getattr(host, '_is_krea2_generation', None)
    try:
        return bool(probe()) if callable(probe) else False
    except Exception:
        return False


def capture_sampling_envelope(host) -> dict:
    """지금 T2I 패널의 샘플링 블록 봉투. Krea2 면 빈 봉투(Forge alwayson 경로가 아니다).

    게이트는 하지 않는다 — 요청 직전의 새 스냅샷으로 백엔드가 한다. 기여자 안내 알림(P8·P9)은 보내는 곳인 여기서
    띄운다(클릭 = 보내기).
    """
    if _is_krea2(host):
        return ap.envelope({}, source=SOURCE_T2I_PANEL, backend=ap.BACKEND_KREA2)
    from ui.sampling_blocks import build_sampling_blocks
    result = build_sampling_blocks(host, ap.TARGET_AUX)
    context = result.context
    if result.notices:
        try:
            from core.sam_extra_notices import PRE_GENERATION_NOTICE_TTL_S
            from ui.sam_extra_notices_ui import show_notices
            show_notices(host, result.notices, ttl=PRE_GENERATION_NOTICE_TTL_S)
        except Exception:
            logger.debug("보조 패스 봉투 알림 실패(무시)", exc_info=True)
    return ap.envelope(result.blocks, source=SOURCE_T2I_PANEL,
                       model=context.model if context else "",
                       backend=context.backend if context else "",
                       provenance=result.provenance)


def attach_sampling_snapshot(host, settings: Optional[Mapping]) -> dict:
    """settings 사본에 봉투(``SETTINGS_KEY``)를 싣는다. 원본은 바꾸지 않는다. 실패하면 봉투 없이(= P7 전 동작)."""
    out = dict(settings) if isinstance(settings, Mapping) else {}
    try:
        out[ap.SETTINGS_KEY] = capture_sampling_envelope(host)
    except Exception:
        out.pop(ap.SETTINGS_KEY, None)
        logger.warning("보조 패스에 메인 생성 설정을 싣지 못했습니다(전달 없이 진행)", exc_info=True)
    return out


def _plain_text(widget: Any) -> str:
    return widget.toPlainText() if widget is not None and hasattr(widget, 'toPlainText') else ''


def inherit_refine_prompts(host, settings: Optional[Mapping]) -> dict:
    """Refine settings 사본 — 메인 프롬프트·네거티브를 보내지 않았으면 지금 T2I 값을 물려받는다
    (확장 Refine 패널의 'Inherit main t2i prompt' 기본 ON 과 같다). 프롬프트를 물려받을 때만 LoRA 스택을 붙인다(D2)."""
    out = dict(settings) if isinstance(settings, Mapping) else {}
    if not out.get('main_prompt') and hasattr(host, 'total_prompt_display'):
        try:
            prompt = _plain_text(host.total_prompt_display)
            if not _is_krea2(host):
                from core.lora_stack import UNIT_MULTIPLIER, append_lora_stack_to_prompt
                prompt = append_lora_stack_to_prompt(
                    prompt, getattr(host, '_vue_lora_entries', None) or [], unit=UNIT_MULTIPLIER)
            out['main_prompt'] = prompt
        except Exception:
            logger.debug("Refine 메인 프롬프트 상속 실패(무시)", exc_info=True)
    if not out.get('main_negative') and hasattr(host, 'neg_prompt_text'):
        try:
            out['main_negative'] = _plain_text(host.neg_prompt_text)
        except Exception:
            pass
    return out


__all__ = ["SOURCE_T2I_PANEL", "attach_sampling_snapshot", "capture_sampling_envelope", "inherit_refine_prompts"]
