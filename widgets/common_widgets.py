# widgets/common_widgets.py
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import Qt

# 예전 WheelEventFilter(설치된 적 없음)·AutomationWidget(Vue 자동화 패널로 대체)·
# SettingsDialog(생성처 없음)는 지웠다 — P13c 고아 정리.
# ResolutionItemWidget(PyQt 랜덤 해상도 편집기 — Vue 편집기로 대체)·NoScrollDoubleSpinBox(숨은
# SettingsTab 전용)는 지웠다 — P13d(audit #175·#178).
# NoScrollSpinBox(batch_tab·mosaic_panel·event_gen_tab·settings_tab 전용)·FlowLayout(gallery_tab 전용)은
# 사용처가 모두 은퇴해 import 0건이 되어 지웠다 — P13 정리(audit #178 후속).


class NoScrollComboBox(QComboBox):
    """스크롤 방지 콤보박스"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()
