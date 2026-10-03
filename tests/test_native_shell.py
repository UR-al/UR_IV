"""Small software-only Qt probes; never create a WebEngine view or start the app."""
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NativeShellTests(unittest.TestCase):
    def qt_probe(self, body):
        prelude = '''
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QStackedWidget, QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QDialog
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QTimer
from unittest.mock import patch
from pathlib import Path
import tempfile
import sys, types
# Load the real helper without ui/__init__ eagerly importing the whole app.
ui_package = types.ModuleType('ui')
ui_package.__path__ = [str(Path('ui').resolve())]
sys.modules['ui'] = ui_package
from utils.theme_manager import ThemeManager
from core.theme_presets import contrast_ratio
app = QApplication([])
theme = ThemeManager()
'''
        result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', textwrap.dedent(prelude) + textwrap.dedent(body)],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=15,
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8',
                                     'QT_QPA_PLATFORM': 'offscreen', 'QT_OPENGL': 'software'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exit_confirmation_matches_current_theme_each_time_it_opens(self):
        self.qt_probe('''
import ast
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from ui.native_dialogs import apply_native_shell_theme
tree = ast.parse(Path('ui/generator_main.py').read_text(encoding='utf-8'))
method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'closeEvent')
scope = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), 'exit-confirmation', 'exec'), scope)
window = QMainWindow()
quit_calls, errors, observed = [], [], []
window._quit_app = lambda: quit_calls.append(True)
cases = [
    {'theme': 'light'}, {'theme': 'dark'}, {'theme': 'default'}, {'theme': 'light'},
    {'theme': 'light', 'themeOverrides': {'accent': '#123456'}},
]
responses = ('no', 'escape', 'close', 'yes')
def inspect_and_respond():
    box = next(w for w in app.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible())
    try:
        colors = theme.get_colors()
        background = box.palette().color(QPalette.ColorRole.Window).name()
        observed.append(background)
        assert background == colors['bg_primary'].lower(), (prefs, background, colors['bg_primary'])
        assert box.parentWidget() is window
        assert box.defaultButton() is box.button(QMessageBox.StandardButton.No)
        for label in box.findChildren(QLabel):
            if label.text():
                foreground = label.palette().color(QPalette.ColorRole.WindowText).name()
                assert contrast_ratio(foreground, background) >= 4.5, (foreground, background)
        for standard in (QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No):
            button = box.button(standard)
            foreground = button.palette().color(QPalette.ColorRole.ButtonText).name()
            fill = button.palette().color(QPalette.ColorRole.Button).name()
            assert contrast_ratio(foreground, fill) >= 4.5, (standard, foreground, fill)
    except Exception as exc:
        errors.append(repr(exc))
    finally:
        if response == 'escape':
            QTest.keyClick(box, Qt.Key.Key_Escape)
        elif response == 'close':
            box.close()
        else:
            standard = QMessageBox.StandardButton.Yes if response == 'yes' else QMessageBox.StandardButton.No
            box.button(standard).click()
with patch('utils.theme_manager.get_theme_manager', return_value=theme):
    for prefs in cases:
        theme.apply_prefs(prefs)
        apply_native_shell_theme(window, theme)
        for response in responses:
            previous_quit_count = len(quit_calls)
            QTimer.singleShot(0, inspect_and_respond)
            event = QCloseEvent()
            scope['closeEvent'](window, event)
            assert event.isAccepted() == (response == 'yes'), response
            assert len(quit_calls) - previous_quit_count == int(response == 'yes'), response
assert len(observed) == len(cases) * len(responses), observed
assert len(quit_calls) == len(cases), quit_calls
assert not errors, errors
''')

    def test_live_theme_change_fixes_existing_stack_child_contrast(self):
        self.qt_probe('''
import ast
tree = ast.parse(Path('ui/generator_main.py').read_text(encoding='utf-8'))
method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_apply_theme_prefs')
scope = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), 'theme-method', 'exec'), scope)
window = QMainWindow()
window._main_stack = QStackedWidget(window)
window._main_stack.setStyleSheet('background: #0d0d0d;')
panel = QWidget()
button = QPushButton('AI Studio', panel)
window._main_stack.addWidget(panel)
window.setCentralWidget(window._main_stack)
window.setStyleSheet('QMainWindow { background: #0d0d0d; }')
with patch('utils.theme_manager.get_theme_manager', return_value=theme):
    for name in ('light', 'dark', 'light'):
        scope['_apply_theme_prefs'](window, {'theme': name})
        window.show()
        app.processEvents()
        foreground = button.palette().color(QPalette.ColorRole.ButtonText).name()
        background = button.palette().color(QPalette.ColorRole.Button).name()
        assert contrast_ratio(foreground, background) >= 4.5, (name, foreground, background)
window.close()
''')

    def test_directory_picker_opens_at_existing_path_with_stable_parent_and_can_cancel(self):
        self.qt_probe('''
from ui.native_dialogs import select_directory, start_directory
theme.apply_prefs({'theme': 'light'})
parent = QMainWindow()
child = QWidget(parent)
seen = []
with tempfile.TemporaryDirectory() as temp:
    folder = Path(temp)
    assert start_directory(str(folder / 'missing' / 'nested')) == str(folder.resolve())
    def inspect_and_finish(accepted):
        dialog = next(w for w in app.topLevelWidgets() if isinstance(w, QFileDialog) and w.isVisible())
        seen.append(dialog.windowTitle())
        assert dialog.parentWidget() is parent
        assert dialog.testOption(QFileDialog.Option.DontUseNativeDialog)
        assert dialog.fileMode() == QFileDialog.FileMode.Directory
        assert Path(dialog.directory().absolutePath()) == folder
        assert theme.get_colors()['text_primary'] in dialog.styleSheet()
        dialog.done(QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected)
    with patch('ui.native_dialogs.get_theme_manager', return_value=theme):
        QTimer.singleShot(0, lambda: inspect_and_finish(True))
        result = select_directory(child, 'Forge Neo installed path', str(folder))
        assert result == str(folder.resolve()), result
        QTimer.singleShot(0, lambda: inspect_and_finish(False))
        assert select_directory(child, 'Forge Neo extension path', str(folder)) is None
assert len(seen) == 2
parent.close()
''')

    def test_existing_browser_and_backend_chrome_recolors_without_page_reload(self):
        self.qt_probe('''
import ast
from types import SimpleNamespace
backgrounds = []
page = SimpleNamespace(setBackgroundColor=lambda color: backgrounds.append(color.name()))
web = SimpleNamespace(page=lambda: page)
with patch('utils.theme_manager.get_theme_manager', return_value=theme):
    for source in ('tabs/browser_tab.py', 'tabs/backend_ui_tab.py'):
        tree = ast.parse(Path(source).read_text(encoding='utf-8'))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'apply_theme')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), source, 'exec'), scope)
        panel = QWidget()
        panel.web_view = panel._web_view = web
        panel._current_url = 'http://already-loaded.example'
        panel._status_label = QLabel('Backend', panel)
        panel._url_display = QLineEdit(panel)
        panel._url_display.setReadOnly(True)
        for name in ('dark', 'light', 'dark'):
            theme.apply_prefs({'theme': name})
            scope['apply_theme'](panel)
            panel.show()
            app.processEvents()
            colors = theme.get_colors()
            if source.endswith('backend_ui_tab.py'):
                assert panel._status_label.palette().color(QPalette.ColorRole.WindowText).name() == colors['text_secondary'].lower()
                assert panel._url_display.palette().color(QPalette.ColorRole.Text).name() == colors['text_secondary'].lower()
            assert contrast_ratio(colors['text_secondary'], colors['bg_input']) >= 4.5
        panel.close()
assert len(backgrounds) == 6
''')

    def test_javascript_confirmation_is_themed_plain_text_and_defaults_to_cancel(self):
        self.qt_probe('''
from ui.native_dialogs import ThemedWebDialogs
from PyQt6.QtCore import QUrl, Qt
theme.apply_prefs({'theme': 'light'})
parent = QMainWindow()
class Page(ThemedWebDialogs):
    def parent(self): return parent
errors = []
def cancel_message():
    box = next(w for w in app.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible())
    try:
        assert box.parentWidget() is parent
        assert box.text() == '<b>Delete?</b>'
        assert box.textFormat() == Qt.TextFormat.PlainText
        assert box.defaultButton() is box.button(QMessageBox.StandardButton.Cancel)
        label = next(w for w in box.findChildren(QLabel) if w.text() == '<b>Delete?</b>')
        fg = label.palette().color(QPalette.ColorRole.WindowText).name()
        label_background = label.palette().color(QPalette.ColorRole.Window)
        bg = (label_background if label_background.alpha() else box.palette().color(QPalette.ColorRole.Window)).name()
        assert contrast_ratio(fg, bg) >= 4.5, (fg, bg)
    except Exception as exc:
        errors.append(repr(exc))
    finally:
        box.button(QMessageBox.StandardButton.Cancel).click()
with patch('ui.native_dialogs.get_theme_manager', return_value=theme):
    QTimer.singleShot(0, cancel_message)
    assert Page().javaScriptConfirm(QUrl('file:///index.html'), '<b>Delete?</b>') is False
assert not errors, errors
parent.close()
''')


if __name__ == '__main__':
    unittest.main()
