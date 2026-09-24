# ui/image_metadata_actions.py
"""Gallery·Favorites·PNG Info 메타데이터 액션의 호스트 헬퍼.

``GeneratorMainUI._handle_vue_action`` 이 부르는 모듈 함수들이다. 메서드가 아니라 함수로 둔
이유: tests/test_comfy_metadata_actions.py 의 하네스가 ``_handle_vue_action`` 만 클래스
본문에서 빌려 쓰므로, 새 메서드를 더하면 하네스에 없는 속성이 된다. 파싱·판정 규칙은
core.metadata_actions / core.image_metadata 에만 있다.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from core.metadata_actions import (
    NO_GENERATION_INFO_MESSAGE,
    build_generation_item,
    resolve_action_metadata,
)


def metadata_for_action(host: Any, payload: Any, clean_path: Callable[[str], str]) -> Optional[dict]:
    """액션 페이로드 → getImageExif 모양의 메타데이터(없으면 None)."""
    return resolve_action_metadata(
        payload, host.vue_bridge.getImageExif, clean_path=clean_path, is_file=os.path.isfile)


def generation_defaults(host: Any) -> dict:
    """메타데이터에 값이 없을 때 쓰는 현재 T2I 위젯 값."""
    def read(name: str, getter: str = "text") -> str:
        widget = getattr(host, name, None)
        try:
            return str(getattr(widget, getter)() or "") if widget is not None else ""
        except Exception:
            return ""

    return {
        "sampler_name": read("sampler_combo", "currentText"),
        "scheduler": read("scheduler_combo", "currentText"),
        "steps": read("steps_input") or 20,
        "cfg_scale": read("cfg_input") or 7,
        "width": read("width_input") or 1024,
        "height": read("height_input") or 1024,
    }


def queue_item_from_metadata(host: Any, info: dict, *, preserve_seed: bool = False) -> Optional[dict]:
    return build_generation_item(info, generation_defaults(host), preserve_seed=preserve_seed)


def start_generation_from_metadata(host: Any, info: dict) -> bool:
    """메타데이터 그대로(seed 포함) T2I 위젯에 적용하고 바로 생성한다.

    ComfyUI·WebUI 가 같은 경로다 — 모델/LoRA 는 메타데이터에서 불러오지 않고 현재 값을
    유지하며, 프롬프트는 prefix/suffix 를 다시 붙이지 않은 '완성된 전체 프롬프트'로 적용한다
    (큐 항목 적용과 같은 규칙).
    """
    item = host._build_queue_payload_from_exif(info, preserve_seed=True)
    if not item:
        host.vue_bridge.showNotification.emit('warning', NO_GENERATION_INFO_MESSAGE)
        return False
    host.main_prompt_text.setPlainText(item['prompt'])
    host._apply_payload_to_ui(item)
    host.start_generation()
    return True


# 메타 이식 대상 — 대화상자 필터와 경로 검증이 같은 목록을 쓴다. 예전에는 필터가 *.tif 를
# 보여 주는데 safe_input_path 기본 목록엔 '.tif' 가 없어 고르면 항상 '열 수 없습니다'였다.
TRANSPLANT_TARGET_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
TRANSPLANT_TARGET_FILTER = f"Images ({' '.join('*' + ext for ext in TRANSPLANT_TARGET_EXTS)});;All Files (*)"


def transplant_metadata_action(host: Any, payload: Any, clean_path: Callable[[str], str],
                               *, pick_target=None, pick_output=None) -> Optional[str]:
    """PNG Info '메타 이식' — 보고 있는 이미지의 생성 메타를 다른 이미지에 박아 새 PNG 로 저장.

    외부 편집기에서 메타가 지워진 사본에 원본의 프롬프트/파라미터(ComfyUI 그래프 포함)를
    되돌리는 용도다. 대상 픽셀은 그대로 두고 결과는 새 파일(기본 ``<이름>_withmeta.png``)로
    쓴다. ``pick_target``/``pick_output`` 은 테스트용 대화상자 대체. 저장한 경로(없으면 None).
    """
    from core.image_metadata import embed_to_file, extract_from_file, suggest_transplant_output
    from core.path_safety import safe_input_path

    notify = host.vue_bridge.showNotification.emit
    payload = payload if isinstance(payload, dict) else {}
    source = safe_input_path(clean_path(str(payload.get('path') or '')))
    if not source:
        notify('warning', '먼저 메타데이터를 가져올 이미지를 여세요.')
        return None
    meta = extract_from_file(source)
    if not meta.has_any():
        notify('warning', '이 이미지에는 옮길 생성 메타데이터가 없습니다.')
        return None

    if pick_target is None or pick_output is None:
        from PyQt6.QtWidgets import QFileDialog

        def pick_target(start_dir):
            return QFileDialog.getOpenFileName(
                host, "메타를 이식할 대상 이미지 선택", start_dir, TRANSPLANT_TARGET_FILTER)[0]

        def pick_output(suggested):
            return QFileDialog.getSaveFileName(host, "결과 PNG 저장 위치", suggested, "PNG (*.png)")[0]

    target_raw = pick_target(os.path.dirname(source))
    if not target_raw:
        return None
    target = safe_input_path(str(target_raw), allowed_exts=frozenset(TRANSPLANT_TARGET_EXTS))
    if not target:
        notify('error', '대상 이미지를 열 수 없습니다.')
        return None
    out_path = pick_output(suggest_transplant_output(target))
    if not out_path:
        return None
    out_path = str(out_path)
    if not out_path.lower().endswith('.png'):
        out_path += '.png'
    if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(source)):
        notify('error', '메타를 가져온 원본 이미지 자리에는 저장할 수 없습니다.')
        return None
    include_workflow = payload.get('include_workflow', True) is not False
    if not embed_to_file(target, meta, out_path, include_workflow=include_workflow):
        notify('error', '메타 이식에 실패했습니다. 로그를 확인하세요.')
        return None
    detail = '' if not (meta.raw_workflow or meta.raw_prompt) else (
        ' (ComfyUI 워크플로 포함)' if include_workflow else ' (ComfyUI 워크플로 제외)')
    notify('success', f'메타 이식 완료: {os.path.basename(out_path)}{detail}')
    return out_path
