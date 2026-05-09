# tabs/editor/mosaic_panel.py
import os
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QFileDialog, QMessageBox, QSizePolicy, QLabel, QButtonGroup, QFrame,
    QDialog, QFormLayout, QCheckBox, QGridLayout, QStackedWidget
)
from widgets.common_widgets import NoScrollSpinBox
from PyQt6.QtCore import Qt
from widgets.sliders import NumericSlider
from utils.theme_manager import get_color


# YOLO 모델 경로 설정 파일
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EDITOR_MODELS_DIR = os.path.join(_PROJECT_ROOT, "editor_models")
_YOLO_CONFIG_PATH = os.path.join(_EDITOR_MODELS_DIR, "yolo_config.json")

# editor_models 디렉토리 자동 생성
os.makedirs(_EDITOR_MODELS_DIR, exist_ok=True)


# SAM 계열 모델 파일명 키워드 (YOLO 검출용으로 잘못 로드되는 것을 방지)
_SAM_KEYWORDS = (
    'mobile_sam', 'fastsam', 'fast_sam',
    'sam_vit', 'sam_hq', 'sam2', 'sam3',
)


def _is_sam_file(fname: str) -> bool:
    """파일명이 SAM 계열인지 판정"""
    fl = os.path.basename(fname).lower()
    return any(k in fl for k in _SAM_KEYWORDS)


def _load_yolo_model_paths() -> list:
    """저장된 YOLO 모델 경로 목록 불러오기 (editor_models/ 기준)
    SAM 계열(mobile_sam / FastSAM / sam_vit*)은 검출 모델이 아니므로 제외.
    """
    paths = []
    try:
        # 1. config에서 등록된 경로 로드
        if os.path.exists(_YOLO_CONFIG_PATH):
            with open(_YOLO_CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                paths = data.get('model_paths', [])
                if not paths:
                    old = data.get('model_path', '')
                    paths = [old] if old else []

        # 2. editor_models/ 디렉토리 내 .pt/.onnx 파일 자동 감지
        for f in os.listdir(_EDITOR_MODELS_DIR):
            fp = os.path.join(_EDITOR_MODELS_DIR, f)
            if f.lower().endswith(('.pt', '.onnx', '.safetensors')) and fp not in paths:
                paths.append(fp)
    except Exception:
        pass

    # 존재하는 경로 + SAM 계열 제외
    return [p for p in paths if os.path.exists(p) and not _is_sam_file(p)]


def _save_yolo_model_paths(paths: list):
    """YOLO 모델 경로 목록 저장 (editor_models/yolo_config.json)"""
    try:
        os.makedirs(_EDITOR_MODELS_DIR, exist_ok=True)
        with open(_YOLO_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump({'model_paths': paths}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_editor_models_dir() -> str:
    """editor_models 디렉토리 경로 반환"""
    return _EDITOR_MODELS_DIR


class ResizeDialog(QDialog):
    """이미지 리사이즈 다이얼로그"""

    def __init__(self, current_w: int, current_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이미지 리사이즈")
        self.setFixedSize(300, 180)

        self._aspect_ratio = current_w / current_h if current_h > 0 else 1.0
        self._updating = False

        layout = QVBoxLayout(self)

        info_label = QLabel(f"현재 크기: {current_w} × {current_h}")
        info_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 12px;")
        layout.addWidget(info_label)

        form = QFormLayout()
        self.spin_w = NoScrollSpinBox()
        self.spin_w.setRange(1, 65536)
        self.spin_w.setValue(current_w)

        self.spin_h = NoScrollSpinBox()
        self.spin_h.setRange(1, 65536)
        self.spin_h.setValue(current_h)

        form.addRow("폭 (W):", self.spin_w)
        form.addRow("높이 (H):", self.spin_h)
        layout.addLayout(form)

        self.chk_ratio = QCheckBox("비율 유지")
        self.chk_ratio.setChecked(True)
        layout.addWidget(self.chk_ratio)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("확인")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("취소")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.spin_w.valueChanged.connect(self._on_w_changed)
        self.spin_h.valueChanged.connect(self._on_h_changed)

    def _on_w_changed(self, val: int):
        if self._updating or not self.chk_ratio.isChecked():
            return
        self._updating = True
        self.spin_h.setValue(max(1, int(round(val / self._aspect_ratio))))
        self._updating = False

    def _on_h_changed(self, val: int):
        if self._updating or not self.chk_ratio.isChecked():
            return
        self._updating = True
        self.spin_w.setValue(max(1, int(round(val * self._aspect_ratio))))
        self._updating = False

    def get_size(self) -> tuple:
        return self.spin_w.value(), self.spin_h.value()


class MosaicPanel(QWidget):
    """모자이크/검열 도구 패널 - 편집 도구 영역"""

    def __init__(self, parent_editor=None):
        super().__init__()
        self.parent_editor = parent_editor
        self.image_label = None
        self.eraser_mode_restore = False
        self.lasso_magnetic_mode = False
        self._yolo_model_paths = _load_yolo_model_paths()

        self._init_ui()

    def set_image_label(self, label):
        """InteractiveLabel 참조 연결"""
        self.image_label = label
        self.slider_tool_size.valueChanged.connect(
            lambda v: self.image_label.set_brush_size(v)
        )

    def _init_ui(self):
        """UI 초기화"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(6)

        # ── 도구 선택 ──
        tool_header = QLabel("도구")
        tool_header.setStyleSheet(
            f"color: {get_color('text_muted')}; font-size: 18px; font-weight: bold; padding: 2px 2px;"
        )
        main_layout.addWidget(tool_header)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        _tool_btn_style = f"""
            QPushButton {{
                background-color: {get_color('bg_button')}; border: 1px solid {get_color('border')};
                border-radius: 6px; color: {get_color('text_secondary')};
                font-size: 13px; font-weight: bold;
                text-align: left; padding-left: 12px;
            }}
            QPushButton:checked {{
                background-color: {get_color('accent')}; color: white;
                border: 1px solid {get_color('accent')};
            }}
            QPushButton:hover {{ border: 1px solid {get_color('text_muted')}; background-color: {get_color('bg_button_hover')}; }}
            QPushButton:checked:hover {{ background-color: {get_color('accent')}; }}
        """

        def create_tool_btn(text, id_val):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(_tool_btn_style)
            self.tool_group.addButton(btn, id_val)
            main_layout.addWidget(btn)
            return btn

        self.btn_tool_box = create_tool_btn("🔲  사각형 선택", 0)
        self.btn_tool_lasso = create_tool_btn("➰  올가미 선택", 1)
        self.btn_tool_brush = create_tool_btn("🖌️  브러쉬", 2)
        self.btn_tool_eraser = create_tool_btn("🧹  지우기", 3)
        self.btn_tool_box.setChecked(True)

        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setStyleSheet(f"color: {get_color('border')};")
        main_layout.addWidget(line1)

        # ── 효과 선택 ──
        effect_header = QLabel("효과")
        effect_header.setStyleSheet(
            f"color: {get_color('text_muted')}; font-size: 18px; font-weight: bold; padding: 2px 2px;"
        )
        main_layout.addWidget(effect_header)

        self.effect_group = QButtonGroup(self)
        self.effect_group.setExclusive(True)

        _effect_btn_style = f"""
            QPushButton {{
                background-color: {get_color('bg_button')}; border: 1px solid {get_color('border')};
                border-radius: 6px; color: {get_color('text_secondary')};
                font-size: 13px; font-weight: bold;
                text-align: left; padding-left: 12px;
            }}
            QPushButton:checked {{
                background-color: {get_color('accent')}; color: white;
                border: 1px solid {get_color('accent')};
            }}
            QPushButton:hover {{ border: 1px solid {get_color('text_muted')}; background-color: {get_color('bg_button_hover')}; }}
            QPushButton:checked:hover {{ background-color: {get_color('accent')}; }}
        """

        def create_effect_btn(text, id_val):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(_effect_btn_style)
            self.effect_group.addButton(btn, id_val)
            main_layout.addWidget(btn)
            return btn

        self.btn_effect_mosaic = create_effect_btn("⬛  모자이크", 0)
        self.btn_effect_censor = create_effect_btn("➖  검은띠 (Bar)", 1)
        self.btn_effect_blur = create_effect_btn("💧  블러 (Blur)", 2)
        self.btn_effect_mosaic.setChecked(True)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet(f"color: {get_color('border')};")
        main_layout.addWidget(line2)

        # ── 자동 검열 (YOLO) ──
        auto_censor_widget = QWidget()
        ac_layout = QVBoxLayout(auto_censor_widget)
        ac_layout.setContentsMargins(0, 0, 0, 0)
        ac_layout.setSpacing(4)

        self.yolo_model_label = QLabel("모델 없음")
        self.yolo_model_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 12px;")
        self.yolo_model_label.setWordWrap(True)
        ac_layout.addWidget(self.yolo_model_label)

        model_btn_row = QHBoxLayout()
        model_btn_row.setSpacing(4)

        btn_add_model = QPushButton("+ 모델")
        btn_add_model.setFixedHeight(35)
        btn_add_model.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_add_model.setStyleSheet(f"""
            QPushButton {{ background-color: {get_color('bg_button_hover')}; border: 1px solid {get_color('border')};
                          border-radius: 4px; color: {get_color('text_primary')}; font-size: 13px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {get_color('border')}; }}
        """)
        btn_add_model.clicked.connect(self._add_yolo_model)

        btn_clear_model = QPushButton("초기화")
        btn_clear_model.setFixedHeight(35)
        btn_clear_model.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_clear_model.setStyleSheet(f"""
            QPushButton {{ background-color: {get_color('bg_button_hover')}; border: 1px solid {get_color('border')};
                          border-radius: 4px; color: {get_color('text_primary')}; font-size: 13px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {get_color('border')}; }}
        """)
        btn_clear_model.clicked.connect(self._clear_yolo_models)

        model_btn_row.addWidget(btn_add_model)
        model_btn_row.addWidget(btn_clear_model)
        ac_layout.addLayout(model_btn_row)

        self.slider_detect_conf = NumericSlider("신뢰도", 1, 100, 25)
        ac_layout.addWidget(self.slider_detect_conf)

        self.btn_auto_censor = QPushButton("🛡️ 자동 검열")
        self.btn_auto_censor.setFixedHeight(35)
        self.btn_auto_censor.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_auto_censor.setStyleSheet(f"""
            QPushButton {{
                background-color: #8B0000; color: white;
                border: 1px solid #B22222; border-radius: 4px;
                font-size: 13px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #A52A2A; }}
            QPushButton:disabled {{ background-color: {get_color('bg_button_hover')}; color: {get_color('text_muted')}; }}
        """)
        self.btn_auto_censor.setToolTip("YOLO 모델로 감지 후 즉시 검열 적용")
        ac_layout.addWidget(self.btn_auto_censor)

        self.btn_auto_detect = QPushButton("🔍 감지만 (마스크)")
        self.btn_auto_detect.setFixedHeight(35)
        self.btn_auto_detect.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_auto_detect.setStyleSheet(f"""
            QPushButton {{
                background-color: {get_color('bg_button_hover')}; color: {get_color('text_primary')};
                border: 1px solid {get_color('border')}; border-radius: 4px;
                font-size: 13px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {get_color('border')}; }}
            QPushButton:disabled {{ background-color: {get_color('bg_secondary')}; color: {get_color('text_muted')}; }}
        """)
        self.btn_auto_detect.setToolTip("감지 영역을 마스크로만 표시 (수동 적용)")
        ac_layout.addWidget(self.btn_auto_detect)

        self.auto_detect_status = QLabel("")
        self.auto_detect_status.setStyleSheet(f"color: {get_color('text_muted')}; font-size: 10px;")
        self.auto_detect_status.setFixedHeight(18)
        ac_layout.addWidget(self.auto_detect_status)

        main_layout.addWidget(auto_censor_widget)

        self._update_model_label()

        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        line3.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line3)

        # ── 액션 위젯 ──
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        # 도구 크기 영역 — QStackedWidget으로 높이 고정 (널뛰기 방지)
        self.slider_tool_size = NumericSlider("도구 크기", 1, 300, 20)

        self.bar_size_container = QWidget()
        bar_size_layout = QVBoxLayout(self.bar_size_container)
        bar_size_layout.setContentsMargins(0, 0, 0, 0)
        bar_size_layout.setSpacing(6)
        self.slider_bar_w = NumericSlider("너비(W)", 1, 500, 50)
        self.slider_bar_h = NumericSlider("높이(H)", 1, 500, 20)
        bar_size_layout.addWidget(self.slider_bar_w)
        bar_size_layout.addWidget(self.slider_bar_h)

        # 단일 슬라이더 페이지: 위에 슬라이더 + 아래 여백으로 높이 맞춤
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(0)
        single_layout.addWidget(self.slider_tool_size)
        single_layout.addStretch(1)

        self.size_stack = QStackedWidget()
        self.size_stack.addWidget(single_page)         # index 0: 도구 크기
        self.size_stack.addWidget(self.bar_size_container)  # index 1: 너비/높이
        # 높이를 bar 모드(2슬라이더) 기준으로 고정
        self.size_stack.setFixedHeight(
            self.slider_bar_w.sizeHint().height() +
            self.slider_bar_h.sizeHint().height() + 6
        )

        self.slider_strength = NumericSlider("효과 강도", 1, 100, 15)
        self.slider_feather = NumericSlider("페더링", 0, 50, 0)

        action_layout.addWidget(self.size_stack)
        action_layout.addWidget(self.slider_strength)
        action_layout.addWidget(self.slider_feather)

        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("✨ 적용 (Enter)")
        self.btn_apply.setStyleSheet(
            f"background-color: {get_color('accent')}; font-weight: bold; "
            "color: white; border-radius: 6px; font-size: 14px;"
        )
        self.btn_apply.setFixedHeight(40)
        self.btn_apply.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_apply.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_cancel_sel = QPushButton("❌ 선택 취소 (Esc)")
        self.btn_cancel_sel.setStyleSheet(
            f"background-color: {get_color('bg_button_hover')}; color: {get_color('text_secondary')}; "
            "border-radius: 6px; font-size: 13px;"
        )
        self.btn_cancel_sel.setFixedHeight(40)
        self.btn_cancel_sel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_cancel_sel.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        btn_layout.addWidget(self.btn_apply, 2)
        btn_layout.addWidget(self.btn_cancel_sel, 1)
        action_layout.addLayout(btn_layout)

        # 크롭/리사이즈 버튼
        crop_resize_layout = QHBoxLayout()
        self.btn_crop = QPushButton("✂️ 크롭")
        self.btn_crop.setFixedHeight(35)
        self.btn_crop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_crop.setStyleSheet(
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 13px; font-weight: bold;"
        )

        self.btn_resize = QPushButton("↔️ 리사이즈")
        self.btn_resize.setFixedHeight(35)
        self.btn_resize.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_resize.setStyleSheet(
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 13px; font-weight: bold;"
        )

        self.btn_perspective = QPushButton("📐 원근보정")
        self.btn_perspective.setFixedHeight(35)
        self.btn_perspective.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_perspective.setStyleSheet(
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 13px; font-weight: bold;"
        )

        crop_resize_layout.addWidget(self.btn_crop)
        crop_resize_layout.addWidget(self.btn_resize)
        crop_resize_layout.addWidget(self.btn_perspective)
        action_layout.addLayout(crop_resize_layout)

        # 회전/뒤집기 버튼
        _tf_style = (
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 13px; font-weight: bold;"
        )

        rotate_layout = QHBoxLayout()
        self.btn_rotate_cw = QPushButton("↻ 90°")
        self.btn_rotate_cw.setFixedHeight(35)
        self.btn_rotate_cw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_rotate_cw.setStyleSheet(_tf_style)
        self.btn_rotate_ccw = QPushButton("↺ 90°")
        self.btn_rotate_ccw.setFixedHeight(35)
        self.btn_rotate_ccw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_rotate_ccw.setStyleSheet(_tf_style)
        rotate_layout.addWidget(self.btn_rotate_cw)
        rotate_layout.addWidget(self.btn_rotate_ccw)
        action_layout.addLayout(rotate_layout)

        flip_layout = QHBoxLayout()
        self.btn_flip_h = QPushButton("↔ 좌우반전")
        self.btn_flip_h.setFixedHeight(35)
        self.btn_flip_h.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_flip_h.setStyleSheet(_tf_style)
        self.btn_flip_v = QPushButton("↕ 상하반전")
        self.btn_flip_v.setFixedHeight(35)
        self.btn_flip_v.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_flip_v.setStyleSheet(_tf_style)
        flip_layout.addWidget(self.btn_flip_h)
        flip_layout.addWidget(self.btn_flip_v)
        action_layout.addLayout(flip_layout)

        # 배경 제거
        line_bg = QFrame()
        line_bg.setFrameShape(QFrame.Shape.HLine)
        line_bg.setStyleSheet(f"color: {get_color('border')};")
        action_layout.addWidget(line_bg)

        from widgets.common_widgets import NoScrollComboBox

        bg_model_layout = QHBoxLayout()
        bg_model_layout.setSpacing(4)

        bg_model_label = QLabel("모델:")
        bg_model_label.setFixedWidth(35)
        bg_model_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 12px;")
        bg_model_layout.addWidget(bg_model_label)

        _BG_MODELS = [
            ("u2net",            "범용 (기본)"),
            ("isnet-anime",      "애니/일러스트"),
            ("isnet-general-use", "범용 개선판"),
            ("silueta",          "경량 빠름"),
        ]

        self.bg_model_combo = NoScrollComboBox()
        for model_id, desc in _BG_MODELS:
            self.bg_model_combo.addItem(f"{model_id}  —  {desc}", model_id)
        self.bg_model_combo.setFixedHeight(30)
        self.bg_model_combo.setStyleSheet(
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 12px; padding: 2px 6px;"
        )
        bg_model_layout.addWidget(self.bg_model_combo)

        bg_color_label = QLabel("배경:")
        bg_color_label.setFixedWidth(35)
        bg_color_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 12px;")
        bg_model_layout.addWidget(bg_color_label)

        self.bg_color_combo = NoScrollComboBox()
        self.bg_color_combo.addItems(["흰색", "검은색"])
        self.bg_color_combo.setFixedHeight(30)
        self.bg_color_combo.setStyleSheet(
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 12px; padding: 2px 6px;"
        )
        bg_model_layout.addWidget(self.bg_color_combo)

        action_layout.addLayout(bg_model_layout)

        self.btn_remove_bg = QPushButton("🖼️ 배경 제거")
        self.btn_remove_bg.setFixedHeight(36)
        self.btn_remove_bg.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_remove_bg.setStyleSheet(
            f"background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; "
            "border-radius: 4px; font-size: 13px; font-weight: bold;"
        )
        action_layout.addWidget(self.btn_remove_bg)

        main_layout.addWidget(action_widget)

        # ── 시그널 연결 ──
        self.tool_group.buttonClicked.connect(self.on_tool_group_clicked)
        self.effect_group.buttonClicked.connect(self.on_effect_changed)
        self.on_tool_group_clicked(self.btn_tool_box)

    # ── 도구 관련 메서드 ──

    def on_tool_group_clicked(self, btn):
        """도구 선택 이벤트"""
        id_val = self.tool_group.id(btn)

        if self.image_label:
            if id_val == 0:
                self.image_label.set_tool('box')
            elif id_val == 1:
                if self.image_label.current_tool == 'lasso':
                    self.lasso_magnetic_mode = not self.lasso_magnetic_mode
                self.image_label.set_tool('lasso')
                self.image_label.magnetic_lasso = self.lasso_magnetic_mode
                self.update_lasso_visual()
            elif id_val == 2:
                self.image_label.set_tool('brush')
            elif id_val == 3:
                if self.image_label.current_tool == 'eraser':
                    self.eraser_mode_restore = not self.eraser_mode_restore
                self.image_label.set_tool('eraser')
                self.image_label.eraser_restores_image = self.eraser_mode_restore
                self.update_eraser_visual()

        _default_style = f"""
            QPushButton {{
                background-color: {get_color('bg_button')}; border: 1px solid {get_color('border')};
                border-radius: 6px; color: {get_color('text_secondary')};
                font-size: 13px; font-weight: bold;
                text-align: left; padding-left: 12px;
            }}
            QPushButton:checked {{
                background-color: {get_color('accent')}; color: white;
                border: 1px solid {get_color('accent')};
            }}
            QPushButton:hover {{ border: 1px solid {get_color('text_muted')}; background-color: {get_color('bg_button_hover')}; }}
            QPushButton:checked:hover {{ background-color: {get_color('accent')}; }}
        """
        if id_val != 3:
            self.btn_tool_eraser.setText("🧹  지우기")
            self.btn_tool_eraser.setStyleSheet(_default_style)
        if id_val != 1:
            self.btn_tool_lasso.setText("➰  올가미 선택")
            self.btn_tool_lasso.setStyleSheet(_default_style)

        self.update_ui_state()
        if self.image_label:
            self.image_label.setFocus()

    def update_eraser_visual(self):
        """지우개 비주얼 업데이트"""
        if self.eraser_mode_restore:
            self.btn_tool_eraser.setText("✨  모자이크 지우기")
            self.btn_tool_eraser.setStyleSheet(f"""
                QPushButton {{
                    background-color: {get_color('bg_button')}; border: 1px solid #e67e22;
                    border-radius: 6px; color: #e67e22;
                    font-size: 13px; font-weight: bold;
                    text-align: left; padding-left: 12px;
                }}
                QPushButton:checked {{
                    background-color: #e67e22; color: white;
                    border: 1px solid #d35400;
                }}
                QPushButton:hover {{ border: 1px solid #d35400; }}
            """)
        else:
            self.btn_tool_eraser.setText("🧹  지우기")
            self.btn_tool_eraser.setStyleSheet(f"""
                QPushButton {{
                    background-color: {get_color('bg_button')}; border: 1px solid {get_color('border')};
                    border-radius: 6px; color: {get_color('text_secondary')};
                    font-size: 13px; font-weight: bold;
                    text-align: left; padding-left: 12px;
                }}
                QPushButton:checked {{
                    background-color: {get_color('accent')}; color: white;
                    border: 1px solid {get_color('accent')};
                }}
                QPushButton:hover {{ border: 1px solid {get_color('text_muted')}; background-color: {get_color('bg_button_hover')}; }}
                QPushButton:checked:hover {{ background-color: {get_color('accent')}; }}
            """)

    def update_lasso_visual(self):
        """올가미 비주얼 업데이트 (일반 ↔ 자석)"""
        if self.lasso_magnetic_mode:
            self.btn_tool_lasso.setText("🧲  자석 올가미")
            self.btn_tool_lasso.setStyleSheet(f"""
                QPushButton {{
                    background-color: {get_color('bg_button')}; border: 1px solid #2ecc71;
                    border-radius: 6px; color: #2ecc71;
                    font-size: 13px; font-weight: bold;
                    text-align: left; padding-left: 12px;
                }}
                QPushButton:checked {{
                    background-color: #2ecc71; color: white;
                    border: 1px solid #27ae60;
                }}
                QPushButton:hover {{ border: 1px solid #27ae60; }}
            """)
        else:
            self.btn_tool_lasso.setText("➰  올가미 선택")
            self.btn_tool_lasso.setStyleSheet(f"""
                QPushButton {{
                    background-color: {get_color('bg_button')}; border: 1px solid {get_color('border')};
                    border-radius: 6px; color: {get_color('text_secondary')};
                    font-size: 13px; font-weight: bold;
                    text-align: left; padding-left: 12px;
                }}
                QPushButton:checked {{
                    background-color: {get_color('accent')}; color: white;
                    border: 1px solid {get_color('accent')};
                }}
                QPushButton:hover {{ border: 1px solid {get_color('text_muted')}; background-color: {get_color('bg_button_hover')}; }}
                QPushButton:checked:hover {{ background-color: {get_color('accent')}; }}
            """)

    def on_effect_changed(self):
        """효과 변경 이벤트"""
        self.update_ui_state()
        if self.image_label:
            self.image_label.setFocus()

    def update_ui_state(self):
        """UI 상태 업데이트"""
        tool_id = self.tool_group.checkedId()
        effect_id = self.effect_group.checkedId()

        is_bar_mode = (tool_id == 0 and effect_id == 1)

        if is_bar_mode:
            self.size_stack.setCurrentIndex(1)
            self.slider_strength.setLabel("찍힘 간격")
        else:
            self.size_stack.setCurrentIndex(0)
            self.slider_tool_size.setEnabled(tool_id >= 2)
            self.slider_strength.setLabel("효과 강도")

        self.slider_strength.setEnabled(True)

    def apply_defaults(self, defaults: dict):
        """설정 탭에서 가져온 기본값을 적용"""
        if not defaults:
            return
        self.slider_tool_size.setValue(defaults.get('tool_size', 20))
        self.slider_strength.setValue(defaults.get('effect_strength', 15))
        self.slider_bar_w.setValue(defaults.get('bar_w', 50))
        self.slider_bar_h.setValue(defaults.get('bar_h', 20))
        self.slider_detect_conf.setValue(defaults.get('detect_conf', 25))

        self.lasso_magnetic_mode = defaults.get('magnetic_lasso', False)
        if self.image_label:
            self.image_label.magnetic_lasso = self.lasso_magnetic_mode
        self.update_lasso_visual()

    # ── YOLO 모델 관리 ──

    def _add_yolo_model(self):
        """YOLO 모델 파일 추가"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "YOLO 모델 선택", "",
            "YOLO Model (*.pt *.onnx);;All Files (*)"
        )
        if paths:
            for p in paths:
                if p not in self._yolo_model_paths:
                    self._yolo_model_paths.append(p)
            _save_yolo_model_paths(self._yolo_model_paths)
            self._update_model_label()

    def _clear_yolo_models(self):
        """모델 목록 초기화"""
        self._yolo_model_paths.clear()
        _save_yolo_model_paths([])
        self._update_model_label()

    def _update_model_label(self):
        """모델 목록 라벨 갱신"""
        if not self._yolo_model_paths:
            self.yolo_model_label.setText("모델 없음")
            return
        names = [os.path.basename(p) for p in self._yolo_model_paths]
        self.yolo_model_label.setText(", ".join(names))

    def validate_yolo_models(self) -> list:
        """YOLO 모델 경로 목록 검증"""
        if not self._yolo_model_paths:
            QMessageBox.warning(
                self, "알림",
                "YOLO 모델을 추가하세요.\n"
                "'+ 모델' 버튼으로 .pt 파일을 선택하세요."
            )
            return []
        valid = []
        for p in self._yolo_model_paths:
            if os.path.isfile(p):
                valid.append(p)
        if not valid:
            QMessageBox.warning(self, "알림", "유효한 모델 파일이 없습니다.")
            return []
        return valid

    def reset_detect_buttons(self):
        """감지/검열 버튼 상태 복원"""
        self.btn_auto_censor.setEnabled(True)
        self.btn_auto_censor.setText("🛡️ 자동 검열")
        self.btn_auto_detect.setEnabled(True)
        self.btn_auto_detect.setText("🔍 감지만 (마스크)")
