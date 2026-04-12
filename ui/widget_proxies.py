# ui/widget_proxies.py
"""
WidgetProxy — PyQt 위젯 인터페이스를 모방하지만 실제 데이터는 Vue와 동기화.
generator_settings.py, generator_actions.py 등 기존 코드가 수정 없이 동작.
"""
from PyQt6.QtCore import QObject, pyqtSignal


class _ProxyBase(QObject):
    """모든 프록시의 공통 no-op 메서드"""
    def setVisible(self, v): pass
    def hide(self): pass
    def show(self): pass
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
    """QTextEdit / TagInputWidget 호환 프록시"""
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

    # QTextEdit document() 호환
    class _DummyLayout:
        def documentSize(self):
            from PyQt6.QtCore import QSizeF
            return QSizeF(100, 60)
        def blockBoundingRect(self, *a):
            from PyQt6.QtCore import QRectF
            return QRectF(0, 0, 100, 20)

    class _DummyBlock:
        def isValid(self): return False

    class _DummyDocument:
        class _Signal:
            def connect(self, *a): pass
        contentsChanged = _Signal()
        def documentLayout(self):
            return TextEditProxy._DummyLayout()
        def setDocumentMargin(self, m): pass
        def firstBlock(self):
            return TextEditProxy._DummyBlock()
        def blockCount(self): return 1

    def document(self):
        return TextEditProxy._DummyDocument()

    def _on_vue_changed(self, value: str):
        if self._value != value:
            self._value = value
            self.textChanged.emit()


class ComboBoxProxy(_ProxyBase):
    """QComboBox 호환 프록시"""
    currentTextChanged = pyqtSignal(str)
    currentIndexChanged = pyqtSignal(int)

    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._items = []
        self._index = -1
        self._enabled = True
        bridge._register_proxy(widget_id, self)

    def addItems(self, items: list):
        self._items = list(items)
        self._bridge.pushWidgetProperty(self._id, "items", self._items)
        # fallback text가 새 items에 있으면 index 복원
        fb = getattr(self, '_fallback_text', '')
        if fb and fb in self._items:
            self._index = self._items.index(fb)
        elif self._items and self._index < 0:
            self._index = 0

    def addItem(self, item: str):
        self._items.append(item)
        self._bridge.pushWidgetProperty(self._id, "items", self._items)

    def clear(self):
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
    def setVisible(self, v): pass
    def hide(self): pass
    def show(self): pass

    def _on_vue_changed(self, value: str):
        if self._value != value:
            self._value = value
            self.textChanged.emit(value)


class LoraProxy(_ProxyBase):
    """LoraActivePanel 호환 프록시"""
    def __init__(self, bridge, widget_id: str, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._id = widget_id
        self._loras = [] # list of {name, weight, enabled}
        bridge._register_proxy(widget_id, self)

    def get_active_lora_text(self) -> str:
        """활성화된 LoRA들을 <lora:name:weight> 형식의 텍스트로 반환"""
        # Vue에서 관리되는 LoRA 목록을 바탕으로 텍스트 구성
        # (실제 데이터는 Vue에서 전달받아야 하므로 일단 빈 문자열 또는 캐시된 값 반환)
        active_list = [f"<lora:{l['name']}:{l['weight']}>" for l in self._loras if l.get('enabled', True)]
        return ", ".join(active_list)

    def set_loras(self, loras: list):
        self._loras = loras
        self._bridge.pushWidgetProperty(self._id, "loras", loras)

    def _on_vue_changed(self, value: str):
        """Vue에서 LoRA 목록 변경 시 (JSON 형태)"""
        try:
            import json
            self._loras = json.loads(value)
        except Exception:
            pass
