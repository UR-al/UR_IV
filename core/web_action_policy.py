# core/web_action_policy.py
"""웹 모드 onAction 권한 정책 — 어떤 Vue 액션을 웹 클라이언트에게 거부할지.

웹 모드(run_WEB_gui.bat)는 ``onAction`` 을 통째로 공개한다. 예전 거부 목록은
``show_api_manager`` 하나뿐이라, 세션 토큰을 가진 클라이언트가

  - ``open_url`` 로 호스트에서 ``webbrowser.open``(= Windows ``os.startfile``)을 부르고,
  - 웹에선 열리지도 않는 백엔드 게이트 액션(probe/select_backend, pick_comfy_workflow)으로
    포트를 두드리거나 백엔드 설정을 영구히 바꾸고,
  - 호스트 데스크톱에 파일 대화상자를 띄울 수 있었다(원격 사용자에겐 무반응).

여기서 두 등급으로 나눈다.
  - ``WEB_BLOCKED_ACTIONS``: 웹 모드면 항상 거부(호스트 권한·설정 변경).
  - ``DESKTOP_DIALOG_ACTIONS``: 호스트 PC 에 네이티브 창을 띄우는 액션 — **원격** 웹 모드
    (바인드 주소가 loopback 이 아님)에서만 거부한다. loopback 웹 모드는 브라우저와
    대화상자가 같은 화면에 뜨므로 그대로 쓸 수 있다.

프론트 ``frontend/src/utils/hostDialogs.ts`` 의 목록이 이 집합과 같아야 한다
(tests/test_web_action_policy.py 가 대조한다). Qt 를 모르는 순수 모듈이다.
"""
from __future__ import annotations

import ipaddress
from collections.abc import Mapping

BACKEND_GATE_MESSAGE = (
    "웹 모드에서는 백엔드 연결 확인·선택을 할 수 없습니다 — 호스트 PC 의 앱에서 설정하세요."
)

COPY_IMAGE_MESSAGE = (
    "웹 모드에서는 호스트 PC 클립보드에 이미지를 복사하지 않습니다 — 이 브라우저에서 복사하세요."
)

#: 웹 모드(loopback 포함)에서 항상 거부하는 액션 → 사용자에게 보여 줄 이유.
WEB_BLOCKED_ACTIONS: Mapping[str, str] = {
    "show_api_manager": "웹 모드에서는 데스크톱 백엔드 관리 창을 열 수 없습니다.",
    # 백엔드 게이트는 웹 모드에서 열리지 않는다(generator_webui._can_use_backend_gate).
    "probe_backend": BACKEND_GATE_MESSAGE,
    "select_backend": BACKEND_GATE_MESSAGE,
    "pick_comfy_workflow": BACKEND_GATE_MESSAGE,
    # 링크는 브라우저가 스스로 연다(frontend/src/utils/externalUrl.ts).
    "open_url": "웹 모드에서는 호스트 PC 에서 링크를 열 수 없습니다 — 이 브라우저에서 직접 여세요.",
    # 설정 복원은 호스트 설정 파일을 덮어쓰고 앱을 재시작한다 — 호스트 PC 의 앱에서만.
    "settings_import": "웹 모드에서는 설정 백업을 가져올 수 없습니다 — 호스트 PC 의 앱에서 복원하세요.",
    "restart_app": "웹 모드에서는 호스트 PC 의 앱을 재시작할 수 없습니다.",
    # 이미지 복사는 브라우저가 자기 클립보드에 한다(frontend/src/utils/clipboardImageCopy.ts).
    # 예전엔 원격 클라이언트의 복사 버튼이 호스트 PC 사용자의 클립보드를 덮어쓰고 '복사됨'으로 답했다
    # — 텍스트 복사(VueBridge.copyTextToClipboard)와 같은 규칙.
    "copy_to_clipboard": COPY_IMAGE_MESSAGE,
}

HOST_DIALOG_MESSAGE = (
    "원격 웹 모드에서는 호스트 PC 의 파일 대화상자·창을 열 수 없습니다 — 호스트 PC 에서 작업하세요."
)

#: 호스트 데스크톱에 네이티브 대화상자/창을 띄우는 액션(원격 웹 모드에서만 거부).
DESKTOP_DIALOG_ACTIONS: frozenset[str] = frozenset({
    # 에디터
    "editor_open_file", "editor_save_as", "editor_load_watermark_image", "editor_add_yolo_model",
    # 배치·업스케일·ADetailer
    "open_batch_files", "open_upscale_files", "open_ad_files", "open_ad_folder",
    # 캡션
    "caption_pick_files", "caption_pick_folder", "caption_pick_outdir", "caption_pick_caformer_dir",
    # 갤러리·PNG Info(메타 이식: 대상 선택·저장 위치)·비교
    "gallery_open_folder", "open_png_info_file", "pnginfo_transplant_meta", "open_compare_image",
    # 검색·이벤트 내보내기/가져오기
    "export_search_results", "import_search_results", "export_event_results", "import_event_results",
    # 프롬프트 히스토리(QMenu)
    "show_prompt_history",
    # 설정 백업 내보내기·생성/캐릭터 프리셋 공유(ui/settings_data_actions.py)
    "settings_export", "presets_export", "presets_import",
    "character_presets_export", "character_presets_import",
    # Creator·대화
    "creator_select_media", "chat_export",
})

#: 브리지에서 막지 않고 **분기가 직접** 원격 웹 모드를 처리하는 대화상자 액션.
#: 에디터 저장은 프론트가 editorSaveResult 를 기다리며 '저장 중' 상태를 잡는다 — 브리지가
#: 조용히 거부하면 그 상태가 풀리지 않는다. 그래서 generator_main 의 저장 분기가
#: 대화상자 대신 오류 결과(editorSaveResult ok=false)를 돌려준다.
#:   - editor_save: 대화상자를 띄우지 않는다(비파괴 저장 — 원본 옆 _edited 사본, 임시 원본이면
#:     기본 출력 폴더). 같은 처리기를 공유하므로 분기 처리 목록에는 남겨 둔다.
#:   - editor_save_as: 항상 묻는다(DESKTOP_DIALOG_ACTIONS 에도 있어 프론트 버튼은 꺼진다).
BRANCH_HANDLED_DIALOG_ACTIONS: frozenset[str] = frozenset({"editor_save", "editor_save_as"})


def normalize_action_name(action: object) -> str:
    return str(action or "").strip().lower()


def is_loopback_bind_host(host: object) -> bool:
    """웹 서버 바인드 주소가 이 PC 전용(loopback)인지. 0.0.0.0/:: 과 LAN 주소는 원격이다."""
    value = str(host or "").strip().strip("[]").lower().rstrip(".")
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def host_dialogs_blocked(*, web_mode: bool, remote: bool) -> bool:
    """호스트 네이티브 대화상자를 띄우면 안 되는 상태인지(= 원격 웹 모드)."""
    return bool(web_mode) and bool(remote)


def web_action_denial(action: object, *, web_mode: bool, remote: bool) -> str | None:
    """거부 사유(사용자 메시지) 또는 None(허용)."""
    if not web_mode:
        return None
    name = normalize_action_name(action)
    reason = WEB_BLOCKED_ACTIONS.get(name)
    if reason:
        return reason
    if remote and name in DESKTOP_DIALOG_ACTIONS and name not in BRANCH_HANDLED_DIALOG_ACTIONS:
        return HOST_DIALOG_MESSAGE
    return None
