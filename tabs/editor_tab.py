# tabs/editor_tab.py
import os
import time
import random
import base64
import cv2
import numpy as np
from io import BytesIO
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget,
    QFileDialog, QMessageBox, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
from widgets.interactive_label import InteractiveLabel
from utils.shortcut_manager import get_shortcut_manager
from tabs.editor.mosaic_panel import MosaicPanel, ResizeDialog
from tabs.editor.color_panel import ColorAdjustPanel
from tabs.editor.watermark_panel import WatermarkPanel
from tabs.editor.move_panel import MovePanel
from workers.generation_worker import Img2ImgFlowWorker
from config import OUTPUT_DIR
from utils.theme_manager import get_color


class YOLODetectWorker(QThread):
    """복수 YOLO 모델로 NSFW 영역 감지"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, cv_image, model_paths: list, conf: float = 0.25):
        super().__init__()
        self.cv_image = cv_image
        self.model_paths = model_paths
        self.conf = conf

    def run(self):
        try:
            from ultralytics import YOLO

            h, w = self.cv_image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            all_bboxes = []

            for model_path in self.model_paths:
                model = YOLO(model_path)
                results = model(self.cv_image, conf=self.conf, verbose=False)
                if not results or len(results) == 0:
                    continue
                result = results[0]
                if result.masks is not None:
                    for mask_tensor in result.masks.data:
                        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                        combined_mask = cv2.bitwise_or(combined_mask, mask_np)
                elif result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes.xyxy.cpu().numpy():
                        all_bboxes.append([int(v) for v in box[:4]])

            if all_bboxes:
                sam_mask = self._refine_with_sam(all_bboxes, h, w)
                combined_mask = cv2.bitwise_or(combined_mask, sam_mask)

            found = cv2.countNonZero(combined_mask) > 0
            self.finished.emit(combined_mask if found else None)

        except ImportError:
            self.error.emit("ultralytics가 설치되지 않았습니다.\npip install ultralytics 로 설치하세요.")
        except Exception as e:
            self.error.emit(str(e))

    def _refine_with_sam(self, bboxes, h, w):
        """SAM으로 바운딩 박스 → 정밀 마스크 변환"""
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            from ultralytics import SAM
            sam_model = SAM("mobile_sam.pt")
            results = sam_model(self.cv_image, bboxes=bboxes, verbose=False)
            if results and results[0].masks is not None:
                for mask_tensor in results[0].masks.data:
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                    if mask_np.shape != (h, w):
                        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, mask_np)
                return mask
        except Exception:
            pass
        for box in bboxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return mask


class MosaicEditor(QWidget):
    """편집기 메인 탭 - 공통 캔버스 + 하단 서브탭"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._detect_worker = None
        self._removal_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # ── 상단 툴바 (공유) ──
        top_toolbar = QHBoxLayout()
        self.btn_load_pc = QPushButton("📂 PC에서 열기")
        self.btn_save = QPushButton("💾 저장")
        self.btn_undo = QPushButton("↩ UNDO")
        self.btn_redo = QPushButton("↪ REDO")
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        self.btn_load_pc.setMinimumWidth(100)
        self.btn_save.setMinimumWidth(60)
        for btn in [self.btn_load_pc, self.btn_save, self.btn_undo, self.btn_redo]:
            btn.setFixedHeight(38)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            top_toolbar.addWidget(btn)
        layout.addLayout(top_toolbar)

        # ── 이미지 캔버스 (공유) ──
        self.image_container = QWidget()
        self.image_container.setStyleSheet(f"background-color: {get_color('bg_secondary')}; border-radius: 8px;")
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = InteractiveLabel(self)
        container_layout.addWidget(self.image_label)
        layout.addWidget(self.image_container, stretch=8)

        # ── 하단 서브탭 (버튼 + 스택) ──
        from PyQt6.QtWidgets import QStackedWidget, QButtonGroup, QFrame
        self.bottom_tabs_container = QWidget()
        bt_layout = QVBoxLayout(self.bottom_tabs_container)
        bt_layout.setContentsMargins(0, 0, 0, 0)
        bt_layout.setSpacing(0)

        # 버튼 행
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 4, 4, 4)
        btn_row.setSpacing(3)

        _TAB_BTN = f"""
            QPushButton {{
                background: {get_color('bg_primary')}; color: {get_color('text_muted')}; padding: 6px 4px;
                border: none; border-radius: 4px;
                font-size: 12px; font-weight: bold;
            }}
            QPushButton:hover {{ color: {get_color('text_secondary')}; background: {get_color('bg_tertiary')}; }}
            QPushButton:checked {{
                color: {get_color('text_primary')}; background: {get_color('bg_input')};
                border-bottom: 2px solid #5865F2;
            }}
        """
        self._subtab_buttons = []
        tab_names = ["🔲 모자이크", "🎨 색감", "🔧 고급색감", "💧 워터마크", "✏️ 그리기", "✂️ 이동"]
        for name in tab_names:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(_TAB_BTN)
            btn_row.addWidget(btn)
            self._subtab_buttons.append(btn)
        self._subtab_buttons[0].setChecked(True)

        bt_layout.addLayout(btn_row)

        # 구분선
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {get_color('border')};")
        bt_layout.addWidget(sep)

        # 스택 위젯
        self._subtab_stack = QStackedWidget()

        # 모자이크 패널
        self.mosaic_panel = MosaicPanel(self)
        self.mosaic_panel.set_image_label(self.image_label)
        self._subtab_stack.addWidget(self.mosaic_panel)

        # 색감 조절 패널
        self.color_panel = ColorAdjustPanel(self)
        self._subtab_stack.addWidget(self.color_panel)

        # 고급 색감 패널
        from tabs.editor.advanced_color_panel import AdvancedColorPanel
        self.adv_color_panel = AdvancedColorPanel(self)
        self._subtab_stack.addWidget(self.adv_color_panel)

        # 워터마크 패널
        self.watermark_panel = WatermarkPanel(self)
        self._subtab_stack.addWidget(self.watermark_panel)

        # 그리기 패널
        from tabs.editor.draw_panel import DrawPanel
        self.draw_panel = DrawPanel(self)
        self._subtab_stack.addWidget(self.draw_panel)

        # 이동 패널
        self.move_panel = MovePanel(self)
        self._subtab_stack.addWidget(self.move_panel)

        bt_layout.addWidget(self._subtab_stack, 1)

        # 버튼 클릭 → 스택 전환
        for i, btn in enumerate(self._subtab_buttons):
            btn.clicked.connect(lambda checked, idx=i: self._switch_subtab(idx))

        # 호환성: bottom_tabs 참조를 유지 (왼쪽 패널에서 사용)
        self.bottom_tabs = self.bottom_tabs_container

        # ── 시그널 연결 ──
        self._connect_signals()
        self.setAcceptDrops(True)

    # ── parent_editor 프록시 프로퍼티 (InteractiveLabel 호환) ──

    @property
    def effect_group(self):
        return self.mosaic_panel.effect_group

    @property
    def slider_bar_w(self):
        return self.mosaic_panel.slider_bar_w

    @property
    def slider_bar_h(self):
        return self.mosaic_panel.slider_bar_h

    @property
    def slider_strength(self):
        return self.mosaic_panel.slider_strength

    def _connect_signals(self):
        """시그널/슬롯 연결"""
        # 상단 툴바
        self.btn_load_pc.clicked.connect(self.load_from_pc)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_undo.clicked.connect(self.image_label.undo)
        self.btn_redo.clicked.connect(self.image_label.redo)
        self.image_label.state_changed.connect(self.update_undo_redo_buttons)

        # 모자이크 패널
        self.mosaic_panel.btn_apply.clicked.connect(self.apply_effect_to_image)
        self.mosaic_panel.btn_cancel_sel.clicked.connect(self.image_label.clear_selection)
        self.mosaic_panel.btn_crop.clicked.connect(self.crop_image)
        self.mosaic_panel.btn_resize.clicked.connect(self.resize_image)
        self.mosaic_panel.btn_auto_censor.clicked.connect(self._on_auto_censor)
        self.mosaic_panel.btn_auto_detect.clicked.connect(self._on_auto_detect)
        self.mosaic_panel.btn_rotate_cw.clicked.connect(self._rotate_cw)
        self.mosaic_panel.btn_rotate_ccw.clicked.connect(self._rotate_ccw)
        self.mosaic_panel.btn_flip_h.clicked.connect(self._flip_horizontal)
        self.mosaic_panel.btn_flip_v.clicked.connect(self._flip_vertical)
        self.mosaic_panel.btn_remove_bg.clicked.connect(self._on_remove_bg)
        self.mosaic_panel.btn_perspective.clicked.connect(self._on_perspective)

        # 색감 조절 패널
        self.color_panel.adjustment_changed.connect(
            lambda b, c, s: self.image_label.set_adjustment_preview(b, c, s)
        )
        self.color_panel.apply_requested.connect(
            lambda b, c, s: self.image_label.apply_adjustment(b, c, s)
        )
        self.color_panel.reset_requested.connect(
            lambda: self.image_label.clear_adjustment_preview()
        )
        self.color_panel.filter_apply_requested.connect(self._apply_filter_preset)
        self.color_panel.filter_preview_requested.connect(self._preview_filter_preset)
        self.color_panel.filter_cancel_requested.connect(self._cancel_filter_preview)
        self.color_panel.auto_correct_requested.connect(self._apply_auto_correct)

        # 고급 색감 패널
        self.adv_color_panel.adjustment_preview.connect(self._on_adv_color_preview)
        self.adv_color_panel.apply_requested.connect(self._on_adv_color_apply)
        self.adv_color_panel.reset_requested.connect(
            lambda: self.image_label.clear_adjustment_preview()
        )

        # 워터마크 패널
        self.watermark_panel.text_watermark_requested.connect(self._apply_text_watermark)
        self.watermark_panel.image_watermark_requested.connect(self._apply_image_watermark)
        self.watermark_panel.preview_requested.connect(self._update_wm_preview)
        self.watermark_panel.preview_cleared.connect(self._clear_wm_preview)
        self.image_label.wm_position_changed.connect(self.watermark_panel.set_position_from_image)
        self.watermark_panel.clamp_changed.connect(self.image_label.set_wm_clamp)
        self.image_label.wm_scale_changed.connect(self._on_wm_scale_changed)
        self.image_label.wm_resize_finished.connect(self._on_wm_resize_finished)

        # 그리기 패널 — 스포이트 색상 연결
        self.image_label.color_picked.connect(self.draw_panel.set_color_from_bgr)
        self.draw_panel.btn_flatten.clicked.connect(self._on_flatten_layer)
        self.draw_panel.btn_heal_apply.clicked.connect(self._on_heal_apply)

        # 이동 패널
        self.move_panel.btn_start_move.clicked.connect(self._on_start_move)
        self.move_panel.btn_confirm.clicked.connect(self._on_confirm_move)
        self.move_panel.btn_cancel.clicked.connect(self._on_cancel_move)
        self.move_panel.btn_undo_move.clicked.connect(self._on_undo_move)
        self._move_snapshot = None
        self.move_panel.btn_send_inpaint.clicked.connect(self._on_send_to_inpaint)
        self.move_panel.slider_rotation.valueChanged.connect(
            lambda v: self._on_move_transform_changed()
        )
        self.move_panel.slider_scale.valueChanged.connect(
            lambda v: self._on_move_transform_changed()
        )

        # 서브탭 전환은 _switch_subtab에서 처리

    # ── 공통 이벤트 ──

    def dragEnterEvent(self, event):
        """드래그 앤 드롭 지원"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """파일 드롭 처리"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self.load_image(f)
                break

    def update_undo_redo_buttons(self, can_undo: bool, can_redo: bool):
        """Undo/Redo 버튼 상태 업데이트"""
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)

    def keyPressEvent(self, event):
        """에디터 단축키 처리"""
        sm = get_shortcut_manager()

        if sm.match(event, 'apply_effect'):
            self.apply_effect_to_image()
            return
        elif sm.match(event, 'cancel_selection'):
            self.image_label.clear_selection()
            return
        elif sm.match(event, 'undo'):
            self.image_label.undo()
            return
        elif sm.match(event, 'redo'):
            self.image_label.redo()
            return
        elif sm.match(event, 'tool_box'):
            self.mosaic_panel.btn_tool_box.setChecked(True)
            self.mosaic_panel.on_tool_group_clicked(self.mosaic_panel.btn_tool_box)
            return
        elif sm.match(event, 'tool_lasso'):
            self.mosaic_panel.btn_tool_lasso.setChecked(True)
            self.mosaic_panel.on_tool_group_clicked(self.mosaic_panel.btn_tool_lasso)
            return
        elif sm.match(event, 'tool_brush'):
            self.mosaic_panel.btn_tool_brush.setChecked(True)
            self.mosaic_panel.on_tool_group_clicked(self.mosaic_panel.btn_tool_brush)
            return
        elif sm.match(event, 'tool_eraser'):
            self.mosaic_panel.btn_tool_eraser.setChecked(True)
            self.mosaic_panel.on_tool_group_clicked(self.mosaic_panel.btn_tool_eraser)
            return

        super().keyPressEvent(event)

    def apply_defaults(self, defaults: dict):
        """설정 탭에서 가져온 기본값을 적용"""
        if not defaults:
            return
        self.mosaic_panel.apply_defaults(defaults)

        # interactive_label 내부 파라미터
        self.image_label._snap_radius = defaults.get('snap_radius', 12)
        self.image_label._canny_low = defaults.get('canny_low', 50)
        self.image_label._canny_high = defaults.get('canny_high', 150)
        self.image_label._smooth_factor = defaults.get('smooth_factor', 0.008)
        self.image_label._rotation_step = defaults.get('rotation_step', 5)
        self.image_label._undo_limit = defaults.get('undo_limit', 20)
        self.image_label._edge_map_dirty = True

    # ── 이미지 로드/저장 ──

    def load_from_pc(self):
        """PC에서 이미지 열기"""
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 열기", "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        """이미지 로드"""
        stream = open(path, "rb")
        bytes_data = bytearray(stream.read())
        stream.close()

        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        cv_img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

        if cv_img is None:
            QMessageBox.warning(self, "오류", "이미지를 불러올 수 없습니다.")
            return

        self.image_label.set_image(cv_img)
        self.btn_load_pc.setText(f"📂 {os.path.basename(path)}")
        self.image_label.setFocus()

    def save_image(self):
        """이미지 저장"""
        if self.image_label.cv_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "이미지 저장", "edited.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            ext = os.path.splitext(path)[1]
            result, n = cv2.imencode(ext, self.image_label.cv_image)
            if result:
                with open(path, mode='wb') as f:
                    f.write(n)
                QMessageBox.information(self, "저장 완료", "저장되었습니다.")
        self.image_label.setFocus()

    # ── 효과 적용 ──

    def apply_effect_to_image(self):
        """모자이크/블러/검은띠 효과 적용"""
        mask = self.image_label.get_current_mask()
        if mask is None or cv2.countNonZero(mask) == 0:
            return

        self.image_label.push_undo_stack()
        img = self.image_label.display_base_image

        try:
            effect_img = img.copy()
            strength = self.mosaic_panel.slider_strength.value()
            current_effect_id = self.mosaic_panel.effect_group.checkedId()

            if current_effect_id == 0:  # Mosaic
                s = max(1, strength)
                h, w = img.shape[:2]
                if w // s > 0 and h // s > 0:
                    small = cv2.resize(effect_img, (w // s, h // s), interpolation=cv2.INTER_LINEAR)
                    effect_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            elif current_effect_id == 1:  # Censor
                effect_img[:] = (0, 0, 0)
            elif current_effect_id == 2:  # Blur
                s = max(1, strength)
                k = s if s % 2 == 1 else s + 1
                effect_img = cv2.GaussianBlur(effect_img, (k, k), 0)

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # 페더링 적용
            feather = self.mosaic_panel.slider_feather.value()
            if feather > 0:
                k = feather * 2 + 1
                binary_mask = cv2.GaussianBlur(binary_mask, (k, k), 0)

            # 알파 블렌딩 (페더링 시 부드러운 경계)
            alpha = binary_mask.astype(np.float32) / 255.0
            alpha_3 = np.stack([alpha] * 3, axis=-1)
            dst = (effect_img.astype(np.float32) * alpha_3 +
                   img.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)

            self.image_label.update_image_keep_view(dst)
            self.image_label.clear_selection()

        except Exception as e:
            print(f"효과 적용 오류: {e}")

        self.image_label.setFocus()

    # ── 크롭 / 리사이즈 ──

    def crop_image(self):
        """크롭 대화상자를 통한 이미지 크롭"""
        if self.image_label.display_base_image is None:
            QMessageBox.warning(self, "알림", "이미지를 먼저 로드하세요.")
            return
        from tabs.editor.crop_dialog import CropDialog
        dlg = CropDialog(self.image_label.display_base_image, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            x, y, w, h = dlg.get_crop_rect()
            if w <= 0 or h <= 0:
                return
            self.image_label.push_undo_stack()
            self.image_label.crop_to_selection(x, y, w, h)
        self.image_label.setFocus()

    def resize_image(self):
        """리사이즈 다이얼로그 표시 및 적용"""
        if self.image_label.display_base_image is None:
            QMessageBox.warning(self, "알림", "이미지를 먼저 로드하세요.")
            return
        img_h, img_w = self.image_label.display_base_image.shape[:2]
        dlg = ResizeDialog(img_w, img_h, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_w, new_h = dlg.get_size()
            if new_w == img_w and new_h == img_h:
                return
            self.image_label.push_undo_stack()
            self.image_label.resize_image(new_w, new_h)
        self.image_label.setFocus()

    # ── 회전 / 뒤집기 ──

    def _rotate_cw(self):
        """90도 시계방향 회전"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        self.image_label.rotate_90_cw()
        self.image_label.setFocus()

    def _rotate_ccw(self):
        """90도 반시계방향 회전"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        self.image_label.rotate_90_ccw()
        self.image_label.setFocus()

    def _flip_horizontal(self):
        """좌우 반전"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        self.image_label.flip_horizontal()
        self.image_label.setFocus()

    def _flip_vertical(self):
        """상하 반전"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        self.image_label.flip_vertical()
        self.image_label.setFocus()

    # ── 원근 보정 ──

    def _on_perspective(self):
        """원근 보정 다이얼로그 표시 및 적용"""
        if self.image_label.display_base_image is None:
            QMessageBox.warning(self, "알림", "이미지를 먼저 로드하세요.")
            return
        from tabs.editor.perspective_dialog import PerspectiveDialog
        dlg = PerspectiveDialog(self.image_label.display_base_image, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            result = dlg.get_result()
            if result is not None:
                self.image_label.push_undo_stack()
                self.image_label.update_image_keep_view(result)
        self.image_label.setFocus()

    # ── 배경 제거 ──

    def _on_remove_bg(self):
        """배경 제거 시작"""
        if self.image_label.display_base_image is None:
            QMessageBox.warning(self, "알림", "이미지를 먼저 로드하세요.")
            return

        self.mosaic_panel.btn_remove_bg.setEnabled(False)
        self.mosaic_panel.btn_remove_bg.setText("제거 중...")

        from tabs.editor.removal_worker import RemovalWorker
        model_name = self.mosaic_panel.bg_model_combo.currentData()
        self._removal_worker = RemovalWorker(
            self.image_label.display_base_image.copy(), model_name
        )
        self._removal_worker.finished.connect(self._on_remove_bg_finished)
        self._removal_worker.error.connect(self._on_remove_bg_error)
        self._removal_worker.start()

    def _on_remove_bg_finished(self, bgra: np.ndarray):
        """배경 제거 완료 — 선택 다이얼로그"""
        self._bgra_result = bgra

        msg = QMessageBox(self)
        msg.setWindowTitle("배경 제거 완료")
        msg.setText("배경 제거가 완료되었습니다.\n어떻게 처리할까요?")
        btn_canvas = msg.addButton("캔버스에 적용", QMessageBox.ButtonRole.AcceptRole)
        btn_save = msg.addButton("투명 PNG 저장", QMessageBox.ButtonRole.ActionRole)
        btn_both = msg.addButton("둘 다", QMessageBox.ButtonRole.YesRole)
        msg.addButton("취소", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        do_canvas = clicked in (btn_canvas, btn_both)
        do_save = clicked in (btn_save, btn_both)

        if do_save:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "투명 PNG 저장", "transparent.png", "PNG (*.png)"
            )
            if save_path:
                cv2.imwrite(save_path, bgra)

        if do_canvas:
            alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
            bg_choice = self.mosaic_panel.bg_color_combo.currentText()
            bg_val = 255 if bg_choice == "흰색" else 0
            bg = np.full_like(bgra[:, :, :3], bg_val, dtype=np.float32)
            fg = bgra[:, :, :3].astype(np.float32)
            result = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)

            self.image_label.push_undo_stack()
            self.image_label.update_image_keep_view(result)

        # 버튼 복원
        self.mosaic_panel.btn_remove_bg.setEnabled(True)
        self.mosaic_panel.btn_remove_bg.setText("🖼️ 배경 제거")
        self.image_label.setFocus()

    def _on_remove_bg_error(self, msg: str):
        """배경 제거 에러"""
        self.mosaic_panel.btn_remove_bg.setEnabled(True)
        self.mosaic_panel.btn_remove_bg.setText("🖼️ 배경 제거")
        QMessageBox.warning(self, "배경 제거 실패", msg)

    # ── 필터 프리셋 ──

    def _blend_filter(self, img: np.ndarray, filter_name: str,
                      strength: int) -> np.ndarray:
        """필터 적용 + 강도 블렌딩"""
        filtered = ColorAdjustPanel.apply_filter(img, filter_name)
        if strength >= 100:
            return filtered
        alpha = strength / 100.0
        return cv2.addWeighted(filtered, alpha, img, 1.0 - alpha, 0)

    def _preview_filter_preset(self, filter_name: str, strength: int):
        """필터 프리셋 실시간 프리뷰 (display_base_image를 건드리지 않음)"""
        if self.image_label.display_base_image is None:
            return
        blended = self._blend_filter(
            self.image_label.display_base_image, filter_name, strength
        )
        self.image_label.cv_image = blended
        self.image_label.update()

    def _cancel_filter_preview(self):
        """필터 프리뷰 취소 → 원본 복원"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.cv_image = self.image_label.display_base_image.copy()
        self.image_label.update()

    def _apply_filter_preset(self, filter_name: str, strength: int = 100):
        """필터 프리셋 확정 적용"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        result = self._blend_filter(
            self.image_label.display_base_image, filter_name, strength
        )
        self.image_label.update_image_keep_view(result)
        self.image_label.setFocus()

    # ── 고급 색감 ──

    def _on_adv_color_preview(self, adjusted_img: np.ndarray):
        """고급 색감 조정 프리뷰"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.cv_image = adjusted_img.copy()
        self.image_label.update()

    def _on_adv_color_apply(self, adjusted_img: np.ndarray):
        """고급 색감 조정 적용"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        self.image_label.update_image_keep_view(adjusted_img)
        # 히스토그램 갱신
        self.adv_color_panel.set_source_image(self.image_label.display_base_image)
        self.image_label.setFocus()

    def _apply_auto_correct(self):
        """자동 색감 보정"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.push_undo_stack()
        corrected = ColorAdjustPanel.auto_correct(self.image_label.display_base_image)
        self.image_label.update_image_keep_view(corrected)
        self.image_label.setFocus()

    # ── 워터마크 ──

    def _switch_subtab(self, index: int):
        """서브탭 버튼 클릭 시 스택 전환 + 모드 토글"""
        self._subtab_stack.setCurrentIndex(index)
        for i, btn in enumerate(self._subtab_buttons):
            btn.setChecked(i == index)

        current_widget = self._subtab_stack.widget(index)

        is_wm_tab = (current_widget == self.watermark_panel)
        self.image_label.set_wm_mode(is_wm_tab)
        self.watermark_panel.set_preview_active(is_wm_tab)

        # 고급 색감 탭 전환 시 히스토그램 갱신
        is_adv_color_tab = (current_widget == self.adv_color_panel)
        if is_adv_color_tab and self.image_label.display_base_image is not None:
            self.adv_color_panel.set_source_image(self.image_label.display_base_image)

        is_draw_tab = (current_widget == self.draw_panel)
        self.image_label.set_draw_mode(is_draw_tab)
        if is_draw_tab:
            self.draw_panel._sync_to_label()

        is_move_tab = (current_widget == self.move_panel)
        self.image_label.set_move_mode(is_move_tab)

    def _on_wm_scale_changed(self, ratio: float):
        """캔버스에서 워터마크 크기 드래그 조절 (ratio는 시작점 대비 배율)"""
        # 리사이즈 시작 시 기준 값 기록
        if not hasattr(self, '_wm_resize_base'):
            self._wm_resize_base = None
        if self._wm_resize_base is None:
            atype = self.watermark_panel._active_type()
            if atype == 'text':
                self._wm_resize_base = ('text', self.watermark_panel.slider_font_size.value())
            elif atype == 'image':
                self._wm_resize_base = ('image', self.watermark_panel.slider_img_scale.value())
            else:
                return

        btype, base_val = self._wm_resize_base
        if btype == 'text':
            new_val = max(8, min(200, int(base_val * ratio)))
            self.watermark_panel.slider_font_size.setValue(new_val)
        elif btype == 'image':
            new_val = max(10, min(500, int(base_val * ratio)))
            self.watermark_panel.slider_img_scale.setValue(new_val)

    def _on_wm_resize_finished(self):
        """리사이즈 드래그 종료 → 기준 값 초기화"""
        self._wm_resize_base = None

    def _update_wm_preview(self, config: dict):
        """워터마크 미리보기 오버레이 갱신"""
        if self.image_label.display_base_image is None:
            return
        img = self.image_label.display_base_image
        h, w = img.shape[:2]
        x_pct = config.get('x_pct', 50)
        y_pct = config.get('y_pct', 50)

        if config.get('type') == 'text':
            overlay = self._render_text_overlay(img, config)
        elif config.get('type') == 'image':
            overlay = self._render_image_overlay(config)
        else:
            return
        if overlay is not None:
            self.image_label.set_wm_overlay(overlay, x_pct, y_pct)

    def _render_text_overlay(self, img, config: dict):
        """텍스트 워터마크를 BGRA 오버레이로 렌더링 (미리보기용)"""
        import cv2
        import numpy as np
        from tabs.editor.watermark_panel import WatermarkPanel

        text = config['text']
        font_size = config['font_size']
        font_family = config.get('font_family', '')
        color = config['color']
        opacity = config['opacity']
        rotation = config['rotation']

        # PIL로 텍스트 렌더링
        text_bgra = WatermarkPanel._render_text_pil(text, font_family, font_size, color)

        # 알파에 opacity 적용
        text_bgra[:, :, 3] = (text_bgra[:, :, 3].astype(np.float32) * opacity).astype(np.uint8)

        if rotation != 0:
            rh, rw = text_bgra.shape[:2]
            center = (rw // 2, rh // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            cos_v = np.abs(M[0, 0])
            sin_v = np.abs(M[0, 1])
            nw = int(rh * sin_v + rw * cos_v)
            nh = int(rh * cos_v + rw * sin_v)
            M[0, 2] += (nw / 2) - center[0]
            M[1, 2] += (nh / 2) - center[1]
            text_bgra = cv2.warpAffine(text_bgra, M, (nw, nh), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return text_bgra

    def _render_image_overlay(self, config: dict):
        """이미지 워터마크를 BGRA 오버레이로 렌더링 (미리보기용)"""
        import cv2
        import numpy as np

        wm_img = cv2.imread(config['image_path'], cv2.IMREAD_UNCHANGED)
        if wm_img is None:
            return None
        scale_val = config['scale']
        opacity = config['opacity']

        wm_h, wm_w = wm_img.shape[:2]
        new_w = max(1, int(wm_w * scale_val))
        new_h = max(1, int(wm_h * scale_val))
        wm_img = cv2.resize(wm_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        if wm_img.shape[2] == 4:
            wm_img[:, :, 3] = (wm_img[:, :, 3].astype(np.float32) * opacity).astype(np.uint8)
        else:
            alpha = np.full((new_h, new_w, 1), int(255 * opacity), dtype=np.uint8)
            wm_img = np.concatenate([wm_img, alpha], axis=2)
        return wm_img

    def _clear_wm_preview(self):
        """워터마크 미리보기 해제"""
        self.image_label.clear_wm_overlay()

    def _apply_text_watermark(self, config: dict):
        """텍스트 워터마크 적용"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.clear_wm_overlay()
        self.image_label.push_undo_stack()
        result = WatermarkPanel.render_text_watermark(self.image_label.display_base_image, config)
        self.image_label.update_image_keep_view(result)
        self.image_label.setFocus()

    def _apply_image_watermark(self, config: dict):
        """이미지 워터마크 적용"""
        if self.image_label.display_base_image is None:
            return
        self.image_label.clear_wm_overlay()
        self.image_label.push_undo_stack()
        result = WatermarkPanel.render_image_watermark(self.image_label.display_base_image, config)
        self.image_label.update_image_keep_view(result)
        self.image_label.setFocus()

    # ── 자동 검열 (YOLO) ──

    def _start_yolo_worker(self, on_finished):
        """YOLO 감지 워커 시작"""
        if self.image_label.display_base_image is None:
            QMessageBox.warning(self, "알림", "이미지를 먼저 로드하세요.")
            return False

        model_paths = self.mosaic_panel.validate_yolo_models()
        if not model_paths:
            return False

        self.mosaic_panel.btn_auto_censor.setEnabled(False)
        self.mosaic_panel.btn_auto_censor.setText("검열 중...")
        self.mosaic_panel.btn_auto_detect.setEnabled(False)
        self.mosaic_panel.btn_auto_detect.setText("감지 중...")
        self.mosaic_panel.auto_detect_status.setText("")

        conf = self.mosaic_panel.slider_detect_conf.value() / 100.0
        self._detect_worker = YOLODetectWorker(
            self.image_label.display_base_image.copy(), model_paths, conf
        )
        self._detect_worker.finished.connect(on_finished)
        self._detect_worker.error.connect(self._on_detect_error)
        self._detect_worker.start()
        return True

    def _on_auto_detect(self):
        """YOLO 감지 → 마스크만 표시"""
        self._start_yolo_worker(self._on_detect_mask_finished)

    def _on_detect_mask_finished(self, mask):
        """감지 완료 → 마스크만 표시"""
        self.mosaic_panel.reset_detect_buttons()

        if mask is None:
            self.mosaic_panel.auto_detect_status.setText("감지 결과 없음")
            QMessageBox.information(self, "결과", "감지된 영역이 없습니다.")
            return

        img = self.image_label.display_base_image
        img_h, img_w = img.shape[:2]
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if self.image_label.selection_mask is None or \
                self.image_label.selection_mask.shape != (img_h, img_w):
            self.image_label.selection_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        self.image_label.selection_mask = cv2.bitwise_or(self.image_label.selection_mask, mask)
        self.mosaic_panel.auto_detect_status.setText("마스크 표시됨")
        self.image_label.update()
        self.image_label.setFocus()

        QMessageBox.information(
            self, "자동 감지 완료",
            "감지 영역이 마스킹되었습니다.\n\n"
            "빨간 영역을 확인하고 브러시/지우기로 수정한 뒤\n"
            "'적용' 버튼을 눌러 효과를 적용하세요."
        )

    def _on_auto_censor(self):
        """YOLO 감지 → 즉시 검열 적용"""
        self._start_yolo_worker(self._on_detect_censor_finished)

    def _on_detect_censor_finished(self, mask):
        """감지 완료 → 바로 효과 적용"""
        self.mosaic_panel.reset_detect_buttons()

        if mask is None:
            self.mosaic_panel.auto_detect_status.setText("감지 결과 없음")
            QMessageBox.information(self, "결과", "감지된 영역이 없습니다.")
            return

        try:
            self.image_label.push_undo_stack()
            img = self.image_label.display_base_image
            img_h, img_w = img.shape[:2]
            effect_img = img.copy()

            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            strength = self.mosaic_panel.slider_strength.value()
            current_effect_id = self.mosaic_panel.effect_group.checkedId()

            if current_effect_id == 0:
                s = max(1, strength)
                if img_w // s > 0 and img_h // s > 0:
                    small = cv2.resize(effect_img, (img_w // s, img_h // s), interpolation=cv2.INTER_LINEAR)
                    effect_img = cv2.resize(small, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            elif current_effect_id == 1:
                effect_img[:] = (0, 0, 0)
            elif current_effect_id == 2:
                s = max(1, strength)
                k = s if s % 2 == 1 else s + 1
                effect_img = cv2.GaussianBlur(effect_img, (k, k), 0)

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            inv_mask = cv2.bitwise_not(binary_mask)
            img_bg = cv2.bitwise_and(img, img, mask=inv_mask)
            img_fg = cv2.bitwise_and(effect_img, effect_img, mask=binary_mask)
            dst = cv2.add(img_bg, img_fg)

            self.image_label.update_image_keep_view(dst)
            self.image_label.clear_selection()
            self.mosaic_panel.auto_detect_status.setText("검열 적용 완료")

            QMessageBox.information(self, "자동 검열 완료", "감지된 영역에 검열 효과가 적용되었습니다.")
        except Exception as e:
            self.mosaic_panel.auto_detect_status.setText("적용 오류")
            QMessageBox.critical(self, "오류", f"검열 적용 실패:\n{e}")

    def _on_detect_error(self, error_msg: str):
        """감지 오류"""
        self.mosaic_panel.reset_detect_buttons()
        self.mosaic_panel.auto_detect_status.setText("오류 발생")

        if "ultralytics" in error_msg.lower() or "No module" in error_msg:
            QMessageBox.critical(
                self, "오류",
                f"ultralytics가 설치되지 않았습니다.\n"
                f"pip install ultralytics 로 설치하세요.\n\n{error_msg}"
            )
        else:
            QMessageBox.critical(self, "오류", f"자동 감지 실패:\n{error_msg}")

    # ── 힐링 브러시 ──

    def _on_heal_apply(self):
        """힐링 브러시 적용"""
        self.image_label.apply_heal()
        self.image_label.setFocus()

    # ── 레이어 ──

    def _on_flatten_layer(self):
        """현재 블렌딩 상태를 pristine에 병합"""
        if self.image_label.display_base_image is not None:
            self.image_label.push_undo_stack()
            self.image_label.pristine_image = self.image_label.display_base_image.copy()
            self.draw_panel.slider_layer_opacity.setValue(100)

    # ── 이동 ──

    def _on_start_move(self):
        """이동 시작"""
        if self.image_label.display_base_image is None:
            self.move_panel.update_status("이미지를 먼저 로드하세요")
            return
        # 이동 전 스냅샷 저장
        self._move_snapshot = self.image_label.display_base_image.copy()
        fill_color = self.move_panel.fill_combo.currentData()
        ok = self.image_label.start_move(fill_color)
        if ok:
            self.move_panel.set_moving_state(True)
            self.move_panel.update_status("드래그로 영역을 이동하세요")
        else:
            self._move_snapshot = None
            self.move_panel.update_status("마스킹을 먼저 해주세요")

    def _on_move_transform_changed(self):
        """이동 중 회전/크기 슬라이더 변경"""
        self.image_label._move_rotation = self.move_panel.slider_rotation.value()
        self.image_label._move_scale = self.move_panel.slider_scale.value() / 100.0
        self.image_label.update()

    def _on_confirm_move(self):
        """이동 확정"""
        self.image_label.confirm_move()
        self.move_panel.set_confirmed_state()
        self.move_panel.slider_rotation.setValue(0)
        self.move_panel.slider_scale.setValue(100)
        self.move_panel.btn_undo_move.setEnabled(self._move_snapshot is not None)
        self.move_panel.update_status("이동 완료! Inpaint 전송 가능 (되돌리기 가능)")

    def _on_undo_move(self):
        """이동 되돌리기 — 이동 시작 전 이미지로 복원"""
        if self._move_snapshot is None:
            return
        self.image_label.display_base_image = self._move_snapshot
        self.image_label._mask_layer = None
        self.image_label.update()
        self._move_snapshot = None
        self.move_panel.btn_undo_move.setEnabled(False)
        self.move_panel.btn_send_inpaint.setEnabled(False)
        self.move_panel.update_status("이동이 되돌려졌습니다")

    def _on_cancel_move(self):
        """이동 취소"""
        self.image_label.cancel_move()
        self.move_panel.slider_rotation.setValue(0)
        self.move_panel.slider_scale.setValue(100)
        self.move_panel.set_moving_state(False)
        self.move_panel.btn_start_move.setEnabled(True)
        self.move_panel.btn_confirm.setEnabled(False)
        self.move_panel.btn_cancel.setEnabled(False)
        self.move_panel.update_status("이동이 취소되었습니다")

    def _on_send_to_inpaint(self):
        """현재 이미지 + 구멍 마스크로 바로 인페인트 실행"""
        if self.image_label.display_base_image is None:
            return

        hole_mask = self.image_label.get_move_hole_mask()
        if hole_mask is None:
            QMessageBox.warning(self, "경고", "마스크가 없습니다. 이동 후 확정해주세요.")
            return

        # 이미지를 base64로 변환
        img = self.image_label.display_base_image
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 마스크를 base64로 변환 (흰색=마스크 영역)
        mask_bw = np.zeros((h, w, 3), dtype=np.uint8)
        mask_bw[hole_mask > 0] = [255, 255, 255]
        mask_pil = Image.fromarray(mask_bw)
        mask_buffer = BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')

        # 프롬프트 가져오기
        prompt = self.move_panel.prompt_text.toPlainText().strip()
        neg_prompt = self.move_panel.neg_prompt_text.toPlainText().strip()
        main_window = self.window()
        if not prompt and main_window and hasattr(main_window, 'total_prompt_display'):
            prompt = main_window.total_prompt_display.toPlainText()
        if not neg_prompt and main_window and hasattr(main_window, 'neg_prompt_text'):
            neg_prompt = main_window.neg_prompt_text.toPlainText()

        payload = {
            "init_images": [img_b64],
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "denoising_strength": 0.75,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
            "mask_blur": 4,
            "inpainting_mask_invert": 0,
            "resize_mode": 0,
            "steps": 20,
            "cfg_scale": 7.0,
            "seed": -1,
            "width": w,
            "height": h,
            "send_images": True,
            "save_images": True,
            "alwayson_scripts": {},
        }

        model_name = ""
        if main_window and hasattr(main_window, 'model_combo'):
            model_name = main_window.model_combo.currentText()

        self.move_panel.btn_send_inpaint.setText("⏳ 생성 중...")
        self.move_panel.btn_send_inpaint.setEnabled(False)
        self.move_panel.update_status("🎨 인페인트 생성 중...")

        # 이전 워커 정리
        if hasattr(self, '_inpaint_worker') and self._inpaint_worker and self._inpaint_worker.isRunning():
            self._inpaint_worker.disconnect()
            self._inpaint_worker.quit()
            self._inpaint_worker.wait(2000)
        self._inpaint_worker = Img2ImgFlowWorker(model_name, payload)
        self._inpaint_worker.finished.connect(self._on_inpaint_finished)
        self._inpaint_worker.start()

    def _on_inpaint_finished(self, result, gen_info):
        """인페인트 생성 완료"""
        self.move_panel.btn_send_inpaint.setText("🎨  인페인트")
        self.move_panel.btn_send_inpaint.setEnabled(True)

        if isinstance(result, bytes):
            # 결과를 에디터 캔버스에 적용
            nparr = np.frombuffer(result, np.uint8)
            new_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if new_img is not None:
                self.image_label.display_base_image = new_img
                self.image_label._last_hole_mask = None
                self.image_label.update()
                self.image_label.push_undo_stack()
                self.move_panel.update_status("✅ 인페인트 완료")

                # 파일 저장
                filename = f"inpaint_{int(time.time())}_{random.randint(100, 999)}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(result)

                main_window = self.window()
                if main_window and hasattr(main_window, 'add_image_to_gallery'):
                    main_window.add_image_to_gallery(filepath)
        else:
            self.move_panel.update_status(f"❌ 인페인트 실패: {result}")
