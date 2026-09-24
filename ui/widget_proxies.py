# ui/widget_proxies.py
"""
WidgetProxy — PyQt 위젯 인터페이스를 모방하지만 실제 데이터는 Vue와 동기화.
generator_settings.py, generator_actions.py 등 기존 코드가 수정 없이 동작.
"""
from PyQt6.QtCore import QObject, pyqtSignal


class _ProxyBase(QObject):
    """모든 프록시의 공통 no-op 메서드.

    표시 여부(setVisible/hide/show)는 두지 않는다 — 보이고 숨기는 것은 Vue 화면의 일이고, 예전
    no-op setVisible 에 연결된 배선은 아무 일도 하지 않으면서 죽은 분기를 숨겼다(audit #175).
    """
    def installEventFilter(self, f): pass
    def setStyleSheet(self, s): pass
    def setToolTip(self, t): pass
    def setFixedWidth(self, w): pass
    def setFixedHeight(self, h): pass
    def setFixedSize(self, *a): pass
    def setMinimumHeight(self, h): pass
    def setMaximumHeight(self, h): pass
    def setMinimumWidth(self, w): pass
    def setSizePolicy(self, *a): pass
    def setFont(self, f): pass
    def setCursor(self, c): pass
    def setMenu(self, m): pass
    def setObjectName(self, n): pass
    def setEnabled(self, e): pass
    def isEnabled(self): return True


class LineEditProxy(_ProxyBase):
    """QLineEdit 호환 프록시"""
    textChanged = pyqtSignal(str)
    editingFinished = pyqtSignal()

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._value = ""
        self._enabled = True
        self._placeholder = ""
        bridge._register_proxy(widget_id, self)

    def text(self) -> str:
        return self._value

    def setText(self, value: str):
        if self._value != value:
            self._value = value
            self._bridge.pushWidgetValue(self._id, value)
            self.textChanged.emit(value)

    def clear(self):
        self.setText("")

    def setPlaceholderText(self, text: str):
        self._placeholder = text
        self._bridge.pushWidgetProperty(self._id, "placeholder", text)

    def setEnabled(self, enabled: bool):
        self._enabled = enabled
        self._bridge.pushWidgetProperty(self._id, "enabled", enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def setFixedWidth(self, w): pass
    def setFixedHeight(self, h): pass
    def setAlignment(self, a): pass
    def setStyleSheet(self, s): pass
    def setToolTip(self, t): pass
    def installEventFilter(self, f): pass

    def _on_vue_changed(self, value: str):
        """Vue에서 값 변경 시 호출 (브릿지 경유)"""
        if self._value != value:
            self._value = value
            self.textChanged.emit(value)


class TextEditProxy(_ProxyBase):
    """QTextEdit 호환 프록시"""
    textChanged = pyqtSignal()

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._value = ""
        self._enabled = True
        bridge._register_proxy(widget_id, self)

    def toPlainText(self) -> str:
        return self._value

    def setPlainText(self, value: str):
        if self._value != value:
            self._value = value
            self._bridge.pushWidgetValue(self._id, value)
            self.textChanged.emit()

    def clear(self):
        self.setPlainText("")

    def setPlaceholderText(self, text: str):
        self._bridge.pushWidgetProperty(self._id, "placeholder", text)

    def setMinimumHeight(self, h): pass
    def setMaximumHeight(self, h): pass
    def setEnabled(self, enabled: bool):
        self._enabled = enabled
        self._bridge.pushWidgetProperty(self._id, "enabled", enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def setReadOnly(self, ro): pass
    def setStyleSheet(self, s): pass
    def setToolTip(self, t): pass
    def installEventFilter(self, f): pass

    def _on_vue_changed(self, value: str):
        if self._value != value:
            self._value = value
            self.textChanged.emit()


class ComboBoxProxy(_ProxyBase):
    """QComboBox 호환 프록시

    선택 상태 세 가지:
    - ``_index``: 지금 목록에서 고른 항목.
    - ``_fallback_text``: 목록에 없어 아직 고르지 못한 값(목록 도착 전 설정값·Vue 값). 실제 항목을
      새로 고르면 지운다 — 남겨 두면 부팅 때의 설정값이 사용자가 나중에 고른 값보다 오래 살아남아,
      연결 오류로 목록이 비었다가 다시 연결될 때 옛 값으로 되돌아갔다(감사 #29 후속).
    - ``_last_selection``: clear() 직전에 고른 항목. currentText() 로는 내보내지 않는다 — 비운
      콤보의 옛 이름이 다른 백엔드 생성 요청으로 흘러가면 안 된다. 연결 때 콤보를 다시 채우는
      ui/combo_restore.py 만 :meth:`preservedText` 로 읽어, 새 목록에 그 항목이 있을 때만 고른다.
    """
    currentTextChanged = pyqtSignal(str)
    currentIndexChanged = pyqtSignal(int)

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._items = []
        self._index = -1
        self._last_selection = ''
        self._enabled = True
        bridge._register_proxy(widget_id, self)

    def addItems(self, items: list):
        self._items = list(items)
        self._bridge.pushWidgetProperty(self._id, "items", self._items)
        # fallback text가 새 items에 있으면 index 복원
        fb = getattr(self, '_fallback_text', '')
        if fb and fb in self._items:
            self._index = self._items.index(fb)
            self._bridge.pushWidgetValue(self._id, self._items[self._index])
        elif self._items and self._index < 0:
            # 저장값이 목록에 없으면 첫 항목 — Vue 에도 알려야 한다. 안 알리면 화면은 '선택…' 인데
            # Python 은 첫 항목으로 생성해 엉뚱한 모델(예: krea2)로 요청이 나간다.
            self._index = 0
            self._bridge.pushWidgetValue(self._id, self._items[0])

    def addItem(self, item: str):
        self._items.append(item)
        self._bridge.pushWidgetProperty(self._id, "items", self._items)

    def clear(self):
        if 0 <= self._index < len(self._items):
            self._last_selection = self._items[self._index]
        self._items = []
        self._index = -1
        self._bridge.pushWidgetProperty(self._id, "items", [])

    def currentIndex(self) -> int:
        return self._index

    def setCurrentText(self, text: str):
        if text in self._items:
            idx = self._items.index(text)
            self.setCurrentIndex(idx)

    def setCurrentIndex(self, idx: int):
        if 0 <= idx < len(self._items):
            self._fallback_text = ''   # 실제 항목을 골랐다 — 미뤄 둔 값은 이제 낡았다
        if self._index != idx and 0 <= idx < len(self._items):
            self._index = idx
            # 인덱스가 아닌 텍스트를 Vue로 전송
            self._bridge.pushWidgetValue(self._id, self._items[idx])
            self.currentTextChanged.emit(self.currentText())
            self.currentIndexChanged.emit(idx)

    def findText(self, text: str) -> int:
        return self._items.index(text) if text in self._items else -1

    def count(self) -> int:
        return len(self._items)

    def itemText(self, idx: int) -> str:
        return self._items[idx] if 0 <= idx < len(self._items) else ""

    def text(self) -> str:
        """호환성: settings에서 .text() 호출 시"""
        return self.currentText()

    def setText(self, value: str):
        """호환성: settings에서 .setText() 호출 시"""
        if value in self._items:
            self.setCurrentText(value)
        else:
            # items 로드 전이면 fallback에 저장 + Vue로 push
            self._fallback_text = value
            self._bridge.pushWidgetValue(self._id, value)

    def setEnabled(self, enabled: bool):
        self._enabled = enabled
    def isEnabled(self) -> bool:
        return self._enabled
    def setStyleSheet(self, s): pass
    def setFixedWidth(self, w): pass
    def setFixedSize(self, *a): pass
    def setToolTip(self, t): pass
    def installEventFilter(self, f): pass

    def _on_vue_changed(self, value: str):
        try:
            idx = int(value)
        except ValueError:
            idx = self._items.index(value) if value in self._items else -1
        if idx >= 0 and idx != self._index and idx < len(self._items):
            self._index = idx
            self._fallback_text = ''   # 사용자가 실제 항목을 골랐다 — 미뤄 둔 값은 낡았다
            self.currentTextChanged.emit(self.currentText())
            self.currentIndexChanged.emit(idx)
        elif idx < 0 and value:
            # items에 없는 값이라도 저장 (설정 로드 시 items보다 값이 먼저 올 수 있음)
            self._fallback_text = value

    def currentText(self) -> str:
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        # fallback: items 로드 전에 설정된 텍스트 반환
        return getattr(self, '_fallback_text', '')

    def preservedText(self) -> str:
        """목록을 다시 채울 때 되살릴 선택 — 지금 항목 → 아직 못 고른 값(fallback) → 비우기 직전 항목.

        fallback 은 실제 항목을 고를 때마다 지워지므로, 남아 있다면 마지막 선택보다 새 값이다.
        """
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return getattr(self, '_fallback_text', '') or self._last_selection


class CheckBoxProxy(_ProxyBase):
    """QCheckBox 호환 프록시"""
    toggled = pyqtSignal(bool)
    stateChanged = pyqtSignal(int)

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._checked = False
        bridge._register_proxy(widget_id, self)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self._bridge.pushWidgetValue(self._id, "true" if checked else "false")
            self.toggled.emit(checked)
            self.stateChanged.emit(2 if checked else 0)

    def setStyleSheet(self, s): pass
    def setToolTip(self, t): pass
    def setEnabled(self, e): pass

    def _on_vue_changed(self, value: str):
        checked = value.lower() in ("true", "1")
        if self._checked != checked:
            self._checked = checked
            self.toggled.emit(checked)
            self.stateChanged.emit(2 if checked else 0)


class ButtonProxy(_ProxyBase):
    """QPushButton 호환 프록시"""
    clicked = pyqtSignal()
    toggled = pyqtSignal(bool)

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._text = ""
        self._enabled = True
        self._checked = False
        self._checkable = False
        bridge._register_proxy(widget_id, self)

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        self._text = text
        self._bridge.pushWidgetProperty(self._id, "text", text)

    def setEnabled(self, enabled: bool):
        self._enabled = enabled
        self._bridge.pushWidgetProperty(self._id, "enabled", enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self._bridge.pushWidgetProperty(self._id, "checked", checked)
            if self._checkable:
                self.toggled.emit(checked)

    def setCheckable(self, v: bool):
        self._checkable = v

    def setObjectName(self, n): pass
    def setFixedHeight(self, h): pass
    def setFixedSize(self, *a): pass
    def setFixedWidth(self, w): pass
    def setStyleSheet(self, s): pass
    def setToolTip(self, t): pass
    def setCursor(self, c): pass
    def setMenu(self, m): pass
    def setSizePolicy(self, *a): pass
    def setMinimumWidth(self, w): pass
    def setFont(self, f): pass
    def setMinimumHeight(self, h): pass

    def _on_vue_changed(self, value: str):
        if value == "click":
            self.clicked.emit()
        elif value in ("true", "false"):
            checked = value == "true"
            if self._checked != checked:
                self._checked = checked
                self.toggled.emit(checked)


class GroupBoxProxy(_ProxyBase):
    """QGroupBox 호환 프록시 (체크 가능)"""
    toggled = pyqtSignal(bool)

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._checked = False
        bridge._register_proxy(widget_id, self)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self._bridge.pushWidgetValue(self._id, "true" if checked else "false")
            self.toggled.emit(checked)

    def setCheckable(self, v): pass
    def setStyleSheet(self, s): pass

    def _on_vue_changed(self, value: str):
        checked = value.lower() in ("true", "1")
        if self._checked != checked:
            self._checked = checked
            self.toggled.emit(checked)


class SliderProxy(_ProxyBase):
    """Steps/CFG 등 슬라이더+입력 쌍 프록시 (NumericSlider 패턴 호환)"""
    textChanged = pyqtSignal(str)
    editingFinished = pyqtSignal()

    def __init__(self, bridge, widget_id: str, multiplier: int = 1, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._value = "0"
        self._multiplier = multiplier
        self._slider = self  # 자기 자신 (호환용)
        bridge._register_proxy(widget_id, self)

    def text(self) -> str:
        return self._value

    def setText(self, value: str):
        if self._value != value:
            self._value = value
            self._bridge.pushWidgetValue(self._id, value)
            self.textChanged.emit(value)

    def value(self) -> int:
        try:
            return int(float(self._value) * self._multiplier)
        except ValueError:
            return 0

    def setValue(self, v: int):
        self._value = str(v / self._multiplier) if self._multiplier != 1 else str(v)
        self._bridge.pushWidgetValue(self._id, self._value)

    def setFixedWidth(self, w): pass
    def setAlignment(self, a): pass
    def installEventFilter(self, f): pass

    def _on_vue_changed(self, value: str):
        if self._value != value:
            self._value = value
            self.textChanged.emit(value)
