# tabs/inpaint_tab.py
"""
Inpaint 탭
- 이미지 위에 마스크를 그려서 부분 재생성
- 브러시/지우개 도구로 마스크 편집
"""
import os
import time
import random
import base64
from io import BytesIO

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QLineEdit, QGroupBox, QFileDialog, QMessageBox,
    QFrame, QSlider, QCheckBox, QScrollArea
)
from widgets.common_widgets import NoScrollComboBox
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import (
    QPixmap, QFont, QImage, QPainter, QPen, QColor, QBrush, QCursor
)
from PIL import Image

from config import OUTPUT_DIR
from utils.app_logger import get_logger
from workers.generation_worker import Img2ImgFlowWorker
from utils.theme_manager import get_color


logger = get_logger('inpaint')


class MaskCanvas(QLabel):
    """마스크 드로잉 캔버스"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet(f"background-color: {get_color('bg_primary')}; border-radius: 8px;")

        self.source_pixmap = None
        self.mask_image = None  # QImage (ARGB)
        self.is_drawing = False
        self.last_point = None
        self.brush_size = 30
        self.is_eraser = False

    def set_image(self, pixmap: QPixmap):
        """소스 이미지 설정"""
        self.source_pixmap = pixmap
        # 마스크 초기화 (투명)
        self.mask_image = QImage(
            pixmap.width(), pixmap.height(), QImage.Format.Format_ARGB32
        )
        self.mask_image.fill(QColor(0, 0, 0, 0))
        self._update_display()

    def clear_mask(self):
        """마스크 초기화"""
        if self.mask_image:
            self.mask_image.fill(QColor(0, 0, 0, 0))
            self._update_display()

    def set_mask_from_numpy(self, mask_np):
        """numpy uint8 마스크를 빨간 반투명 QImage로 변환하여 설정"""
        import numpy as np
        h, w = mask_np.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask_np > 0] = [255, 0, 0, 128]
        q_img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self.mask_image = q_img.convertToFormat(QImage.Format.Format_ARGB32).copy()
        self._update_display()

    def invert_mask(self):
        """마스크 반전"""
        if not self.mask_image:
            return
        for y in range(self.mask_image.height()):
            for x in range(self.mask_image.width()):
                c = self.mask_image.pixelColor(x, y)
                if c.alpha() > 0:
                    self.mask_image.setPixelColor(x, y, QColor(0, 0, 0, 0))
                else:
                    self.mask_image.setPixelColor(x, y, QColor(255, 0, 0, 128))
        self._update_display()

    def get_mask_base64(self) -> str:
        """마스크를 base64로 반환 (흰색=마스크 영역, 검은색=유지 영역)"""
        if not self.mask_image:
            return ""
        # 흑백 마스크 생성
        bw = QImage(
            self.mask_image.width(), self.mask_image.height(),
            QImage.Format.Format_RGB32
        )
        bw.fill(QColor(0, 0, 0))

        painter = QPainter(bw)
        for y in range(self.mask_image.height()):
            for x in range(self.mask_image.width()):
                if self.mask_image.pixelColor(x, y).alpha() > 0:
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawPoint(x, y)
        painter.end()

        buffer = BytesIO()
        bw_pil = Image.fromqimage(bw)
        bw_pil.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _update_display(self):
        """소스 이미지 + 마스크 오버레이 표시"""
        if not self.source_pixmap:
            return

        display = self.source_pixmap.copy()
        painter = QPainter(display)
        painter.setOpacity(0.5)
        painter.drawImage(0, 0, self.mask_image)
        painter.end()

        self.setPixmap(display.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _map_to_image(self, pos):
        """위젯 좌표 → 이미지 좌표 변환"""
        if not self.source_pixmap or not self.pixmap():
            return None

        displayed = self.pixmap()
        # 표시 영역의 오프셋 계산
        x_offset = (self.width() - displayed.width()) // 2
        y_offset = (self.height() - displayed.height()) // 2

        img_x = pos.x() - x_offset
        img_y = pos.y() - y_offset

        if img_x < 0 or img_y < 0 or img_x >= displayed.width() or img_y >= displayed.height():
            return None

        # 스케일 계산
        scale_x = self.source_pixmap.width() / displayed.width()
        scale_y = self.source_pixmap.height() / displayed.height()

        return QPoint(int(img_x * scale_x), int(img_y * scale_y))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.mask_image:
            self.is_drawing = True
            mapped = self._map_to_image(event.pos())
            if mapped:
                self.last_point = mapped
                self._draw_at(mapped)

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.mask_image:
            mapped = self._map_to_image(event.pos())
            if mapped:
                self._draw_line(self.last_point, mapped)
                self.last_point = mapped

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            self.last_point = None

    def _draw_at(self, point):
        """한 점에 브러시 적용"""
        painter = QPainter(self.mask_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.is_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 0, 0, 128)))

        painter.drawEllipse(point, self.brush_size // 2, self.brush_size // 2)
        painter.end()
        self._update_display()

    def _draw_line(self, start, end):
        """두 점 사이 선 그리기"""
        if not start or not end:
            return

        painter = QPainter(self.mask_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.is_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            pen = QPen(QColor(0, 0, 0, 0), self.brush_size,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        else:
            pen = QPen(QColor(255, 0, 0, 128), self.brush_size,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

        painter.setPen(pen)
        painter.drawLine(start, end)
        painter.end()
        self._update_display()


class InpaintTab(QWidget):
    """Inpaint 탭"""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.current_image_path = None
        self.current_base64 = None
        self.gen_worker = None
        self.setAcceptDrops(True)
        self._setup_ui()

    def _apply_extensions(self, payload):
        """메인 창의 확장 설정(NegPiP/ADetailer/SAM3/Anima)을 payload에 반영.

        수동 마스크 인페인트 위에 SAM3가 얹히면 '사람이 칠한 영역 → 인페인트 →
        SAM3가 얼굴 재손질' 순서로 도는데, 이는 Forge Neo UI에서 i2i 탭의
        SAM3 아코디언을 켠 것과 동일한 동작이다.
        """
        mw = self.main_window
        if mw is None or not hasattr(mw, 'apply_alwayson_extensions'):
            return
        try:
            mw.apply_alwayson_extensions(payload)
        except Exception as e:
            logger.warning("Inpaint 확장 적용 실패 (확장 없이 진행): %s", e)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 왼쪽: 도구 패널 (left_stack에 삽입됨) ---
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 0, 10, 0)
        left_layout.setSpacing(8)

        title = QLabel("Inpaint")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {get_color('accent')};")
        left_layout.addWidget(title)

        # 이미지 로드
        load_group = QGroupBox("이미지")
        load_layout = QVBoxLayout(load_group)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("📂 열기")
        self.btn_open.setFixedHeight(38)
        self.btn_open.clicked.connect(self._open_image)
        btn_row.addWidget(self.btn_open)

        self.btn_from_t2i = QPushButton("📋 T2I 결과")
        self.btn_from_t2i.setFixedHeight(38)
        self.btn_from_t2i.clicked.connect(self._paste_from_t2i)
        btn_row.addWidget(self.btn_from_t2i)

        load_layout.addLayout(btn_row)
        left_layout.addWidget(load_group)

        # 브러시 설정
        brush_group = QGroupBox("브러시 도구")
        brush_layout = QVBoxLayout(brush_group)

        tool_row = QHBoxLayout()
        self.btn_brush = QPushButton("🖌️ 브러시")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setChecked(True)
        self.btn_brush.clicked.connect(lambda: self._set_tool(False))
        self.btn_brush.setFixedHeight(38)
        tool_row.addWidget(self.btn_brush)

        self.btn_eraser = QPushButton("🧹 지우개")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.clicked.connect(lambda: self._set_tool(True))
        self.btn_eraser.setFixedHeight(38)
        tool_row.addWidget(self.btn_eraser)
        brush_layout.addLayout(tool_row)

        # 브러시 크기
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("크기:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.brush_slider.wheelEvent = lambda event: event.ignore()
        self.brush_slider.setRange(5, 100)
        self.brush_slider.setValue(30)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        size_row.addWidget(self.brush_slider)
        self.brush_size_label = QLabel("30")
        self.brush_size_label.setFixedWidth(30)
        size_row.addWidget(self.brush_size_label)
        brush_layout.addLayout(size_row)

        # 마스크 조작
        mask_row = QHBoxLayout()
        self.btn_clear_mask = QPushButton("🗑️ 마스크 초기화")
        self.btn_clear_mask.clicked.connect(self._clear_mask)
        mask_row.addWidget(self.btn_clear_mask)

        self.btn_invert_mask = QPushButton("🔄 마스크 반전")
        self.btn_invert_mask.clicked.connect(self._invert_mask)
        mask_row.addWidget(self.btn_invert_mask)
        brush_layout.addLayout(mask_row)

        left_layout.addWidget(brush_group)

        # 프롬프트
        prompt_group = QGroupBox("프롬프트")
        prompt_layout = QVBoxLayout(prompt_group)

        prompt_layout.addWidget(QLabel("프롬프트:"))
        self.prompt_text = QTextEdit()
        self.prompt_text.setFixedHeight(50)
        self.prompt_text.setPlaceholderText("인페인트할 내용")
        prompt_layout.addWidget(self.prompt_text)

        prompt_layout.addWidget(QLabel("네거티브:"))
        self.neg_prompt_text = QTextEdit()
        self.neg_prompt_text.setFixedHeight(35)
        prompt_layout.addWidget(self.neg_prompt_text)

        left_layout.addWidget(prompt_group)

        # 인페인트 설정
        settings_group = QGroupBox("설정")
        settings_layout = QVBoxLayout(settings_group)

        # 디노이징
        d_row = QHBoxLayout()
        d_row.addWidget(QLabel("디노이징:"))
        self.denoise_input = QLineEdit("0.75")
        self.denoise_input.setFixedWidth(60)
        d_row.addWidget(self.denoise_input)
        d_row.addStretch()
        settings_layout.addLayout(d_row)

        # 인페인트 채우기
        fill_row = QHBoxLayout()
        fill_row.addWidget(QLabel("채우기:"))
        self.fill_combo = NoScrollComboBox()
        self.fill_combo.addItems(["채우기", "원본", "잠재 노이즈", "잠재 빈값"])
        self.fill_combo.setCurrentIndex(1)
        fill_row.addWidget(self.fill_combo)
        settings_layout.addLayout(fill_row)

        # 마스크 블러
        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("마스크 블러:"))
        self.mask_blur_input = QLineEdit("4")
        self.mask_blur_input.setFixedWidth(40)
        blur_row.addWidget(self.mask_blur_input)
        blur_row.addStretch()
        settings_layout.addLayout(blur_row)

        # 원본 해상도 인페인트
        self.chk_full_res = QCheckBox("원본 해상도로 인페인트")
        self.chk_full_res.setChecked(True)
        settings_layout.addWidget(self.chk_full_res)

        # 패딩
        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("패딩:"))
        self.padding_input = QLineEdit("32")
        self.padding_input.setFixedWidth(50)
        pad_row.addWidget(self.padding_input)
        pad_row.addStretch()
        settings_layout.addLayout(pad_row)

        # 스텝/CFG/시드
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("스텝:"))
        self.steps_input = QLineEdit("20")
        self.steps_input.setFixedWidth(40)
        param_row.addWidget(self.steps_input)
        param_row.addWidget(QLabel("CFG:"))
        self.cfg_input = QLineEdit("7.0")
        self.cfg_input.setFixedWidth(40)
        param_row.addWidget(self.cfg_input)
        settings_layout.addLayout(param_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("시드:"))
        self.seed_input = QLineEdit("-1")
        self.seed_input.setFixedWidth(80)
        seed_row.addWidget(self.seed_input)
        seed_row.addStretch()
        settings_layout.addLayout(seed_row)

        left_layout.addWidget(settings_group)

        # 생성 버튼
        self.btn_generate = QPushButton("🎨 인페인트 생성")
        self.btn_generate.setFixedHeight(50)
        self.btn_generate.setStyleSheet(f"""
            QPushButton {{
                background-color: #27ae60; color: white;
                font-weight: bold; font-size: 14px; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #2ecc71; }}
            QPushButton:disabled {{ background-color: {get_color('bg_primary')}; color: {get_color('text_muted')}; }}
        """)
        self.btn_generate.clicked.connect(self._on_generate)
        left_layout.addWidget(self.btn_generate)

        # 대기열 추가 버튼
        self.btn_add_queue = QPushButton("📋 대기열에 추가")
        self.btn_add_queue.setFixedHeight(40)
        self.btn_add_queue.setStyleSheet("""
            QPushButton {
                background-color: #E67E22; color: white;
                font-weight: bold; font-size: 13px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #D35400; }
        """)
        self.btn_add_queue.clicked.connect(self._on_add_to_queue)
        left_layout.addWidget(self.btn_add_queue)

        left_layout.addStretch()

        # --- 중앙: 마스크 캔버스 ---
        self.canvas = MaskCanvas()

        # --- 오른쪽: 결과 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 10, 0)

        result_title = QLabel("결과")
        result_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        result_title.setStyleSheet(f"color: {get_color('text_primary')};")
        right_layout.addWidget(result_title)

        self.result_label = QLabel("인페인트 결과")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet(
            f"background-color: {get_color('bg_primary')}; border-radius: 8px; color: {get_color('text_muted')};"
        )
        self.result_label.setMinimumHeight(300)
        self.result_label.setScaledContents(False)
        right_layout.addWidget(self.result_label, 1)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(60)
        self.info_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.info_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        right_layout.addWidget(self.info_text)

        # 스크롤 영역에 왼쪽 패널 설정
        self.left_scroll.setWidget(left_panel)

        # 왼쪽 패널은 left_stack에서 관리 — 여기서는 캔버스+결과만 배치
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(right_panel, 1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self._load_image(f)
                break

    def _set_tool(self, is_eraser):
        self.canvas.is_eraser = is_eraser
        self.btn_brush.setChecked(not is_eraser)
        self.btn_eraser.setChecked(is_eraser)

    def _on_brush_size_changed(self, value):
        self.canvas.brush_size = value
        self.brush_size_label.setText(str(value))

    def _clear_mask(self):
        self.canvas.clear_mask()

    def _invert_mask(self):
        self.canvas.invert_mask()

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 열기", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._load_image(path)

    def _paste_from_t2i(self):
        if self.main_window and hasattr(self.main_window, 'current_image_path'):
            path = self.main_window.current_image_path
            if path and os.path.exists(path):
                self._load_image(path)
                if hasattr(self.main_window, 'total_prompt_display'):
                    self.prompt_text.setPlainText(
                        self.main_window.total_prompt_display.toPlainText()
                    )
                if hasattr(self.main_window, 'neg_prompt_text'):
                    self.neg_prompt_text.setPlainText(
                        self.main_window.neg_prompt_text.toPlainText()
                    )
            else:
                QMessageBox.warning(self, "경고", "T2I 탭에 생성된 이미지가 없습니다.")

    def _load_image(self, path):
        self.current_image_path = path

        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.canvas.set_image(pixmap)

        # Base64 인코딩
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        self.current_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    def set_image_and_mask(self, pixmap: QPixmap, mask_np=None):
        """에디터에서 이미지 + 마스크를 직접 전달받아 설정"""
        if pixmap.isNull():
            return
        self.canvas.set_image(pixmap)

        # QPixmap → base64 인코딩
        buffer = BytesIO()
        q_img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = q_img.width(), q_img.height()
        ptr = q_img.bits()
        ptr.setsize(w * h * 3)
        img = Image.frombytes('RGB', (w, h), bytes(ptr))
        img.save(buffer, format='PNG')
        self.current_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 마스크 설정
        if mask_np is not None:
            self.canvas.set_mask_from_numpy(mask_np)

    def load_from_payload(self, payload: dict):
        """PngInfo 탭에서 전달받은 payload 적용"""
        if 'image_path' in payload and os.path.exists(payload['image_path']):
            self._load_image(payload['image_path'])

        if 'init_images' in payload and payload['init_images']:
            self.current_base64 = payload['init_images'][0]

        self.prompt_text.setPlainText(payload.get('prompt', ''))
        self.neg_prompt_text.setPlainText(payload.get('negative_prompt', ''))
        self.denoise_input.setText(str(payload.get('denoising_strength', 0.75)))
        self.steps_input.setText(str(payload.get('steps', 20)))
        self.cfg_input.setText(str(payload.get('cfg_scale', 7.0)))
        self.seed_input.setText(str(payload.get('seed', -1)))

    def generate_from_payload(self, payload: dict):
        """Vue InpaintView 페이로드로 생성 (image + mask + 설정)."""
        img = payload.get('image') or ''
        if isinstance(img, str) and img.startswith('data:'):
            img = img.split(',', 1)[-1]
        if img:
            self.current_base64 = img
            if payload.get('image_path'):
                self.current_image_path = payload['image_path']
        elif payload.get('image_path') and os.path.exists(payload['image_path']):
            self._load_image(payload['image_path'])
        mask = payload.get('mask') or ''
        if isinstance(mask, str) and mask.startswith('data:'):
            mask = mask.split(',', 1)[-1]
        self._vue_mask = mask or None
        self.prompt_text.setPlainText(payload.get('prompt', '') or '')
        if payload.get('negative_prompt') is not None:
            self.neg_prompt_text.setPlainText(payload.get('negative_prompt', '') or '')
        self.denoise_input.setText(str(payload.get('denoising', 0.75)))
        if not self.current_base64:
            if self.main_window and hasattr(self.main_window, 'vue_bridge'):
                self.main_window.vue_bridge.showNotification.emit('error', 'Inpaint: 입력 이미지가 없습니다')
            return
        if not self._vue_mask:
            if self.main_window and hasattr(self.main_window, 'vue_bridge'):
                self.main_window.vue_bridge.showNotification.emit('error', 'Inpaint: 마스크를 그려주세요')
            return
        self._on_generate()

    def _on_generate(self):
        if not self.current_base64:
            QMessageBox.warning(self, "경고", "먼저 이미지를 로드하세요.")
            return

        mask_b64 = getattr(self, '_vue_mask', None) or self.canvas.get_mask_base64()
        if not mask_b64:
            QMessageBox.warning(self, "경고", "마스크를 그려주세요.")
            return

        prompt = self.prompt_text.toPlainText().strip()
        neg_prompt = self.neg_prompt_text.toPlainText().strip()
        if not prompt and self.main_window and hasattr(self.main_window, 'total_prompt_display'):
            prompt = self.main_window.total_prompt_display.toPlainText()
        if not neg_prompt and self.main_window and hasattr(self.main_window, 'neg_prompt_text'):
            neg_prompt = self.main_window.neg_prompt_text.toPlainText()

        # 원본 이미지 크기
        img = Image.open(self.current_image_path) if self.current_image_path else None
        w = img.width if img else 1024
        h = img.height if img else 1024

        payload = {
            "init_images": [self.current_base64],
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "denoising_strength": float(self.denoise_input.text()),
            "inpainting_fill": self.fill_combo.currentIndex(),
            "inpaint_full_res": self.chk_full_res.isChecked(),
            "inpaint_full_res_padding": int(self.padding_input.text()),
            "mask_blur": int(self.mask_blur_input.text()),
            "inpainting_mask_invert": 0,
            "resize_mode": 0,
            "steps": int(self.steps_input.text()),
            "cfg_scale": float(self.cfg_input.text()),
            "seed": int(self.seed_input.text()),
            "width": w,
            "height": h,
            "send_images": True,
            "save_images": True,
            "alwayson_scripts": {},
        }

        # 확장(NegPiP/ADetailer/SAM3/Anima Guidance) — Forge Neo와 동일하게 적용
        self._apply_extensions(payload)

        model_name = ""
        if self.main_window and hasattr(self.main_window, 'model_combo'):
            model_name = self.main_window.model_combo.currentText()

        self.btn_generate.setText("⏳ 생성 중...")
        self.btn_generate.setEnabled(False)
        self.result_label.setText("🎨 인페인트 생성 중...")

        # 이전 워커 정리
        if hasattr(self, 'gen_worker') and self.gen_worker and self.gen_worker.isRunning():
            self.gen_worker.disconnect()
            self.gen_worker.quit()
            self.gen_worker.wait(2000)
        self.gen_worker = Img2ImgFlowWorker(model_name, payload)
        self.gen_worker.finished.connect(self._on_generation_finished)
        self.gen_worker.start()

    def _on_add_to_queue(self):
        """현재 설정을 대기열에 추가"""
        prompt = self.prompt_text.toPlainText().strip()
        neg_prompt = self.neg_prompt_text.toPlainText().strip()
        if not prompt and self.main_window and hasattr(self.main_window, 'total_prompt_display'):
            prompt = self.main_window.total_prompt_display.toPlainText()
        if not neg_prompt and self.main_window and hasattr(self.main_window, 'neg_prompt_text'):
            neg_prompt = self.main_window.neg_prompt_text.toPlainText()

        payload = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "steps": int(self.steps_input.text()),
            "cfg_scale": float(self.cfg_input.text()),
            "seed": int(self.seed_input.text()),
            "width": 1024,
            "height": 1024,
            "send_images": True,
            "save_images": True,
        }

        if self.main_window and hasattr(self.main_window, 'queue_panel'):
            self.main_window.queue_panel.add_single_item(payload)
            if hasattr(self.main_window, 'show_status'):
                self.main_window.show_status("📋 Inpaint 설정이 대기열에 추가되었습니다.")

    def _on_generation_finished(self, result, gen_info):
        self.btn_generate.setText("🎨 인페인트 생성")
        self.btn_generate.setEnabled(True)
        self._vue_mask = None   # Vue 마스크 1회용 — 다음 생성에 재사용 방지

        if isinstance(result, bytes):
            info = gen_info if isinstance(gen_info, dict) else {}
            # 이 탭은 화면에 없다(결과는 Vue 히스토리가 보여 준다) — 예전처럼 GUI 스레드에서 결과를
            # 풀 디코드·스무스 축소해 보이지 않는 result_label 에 그리지 않는다.
            self.result_label.setText("✅ 인페인트 생성 완료")

            filename = f"inpaint_{int(time.time())}_{random.randint(100, 999)}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(result)

            self.info_text.setPlainText(
                f"저장: {filepath}\n"
                f"Seed: {info.get('seed', '?')}"
            )

            # Vue 히스토리로만 보낸다 — 해상도는 결과 헤더(요청 크기는 폴백, core.result_image).
            # (숨은 PyQt 갤러리 add_image_to_gallery 는 은퇴했다 — 풀해상도 QPixmap 을 최대 100장 들고 있었다)
            if self.main_window and hasattr(self.main_window, 'vue_bridge'):
                try:
                    from core.result_image import result_image_size
                    width, height = result_image_size(result, info)
                    self.main_window.vue_bridge.send_image(
                        filepath, width, height, info.get('seed', -1))
                    self.main_window.vue_bridge.showNotification.emit('success', '인페인트 생성 완료')
                except Exception:
                    pass
        else:
            self.result_label.setText(f"❌ 실패\n{result}")
            self.info_text.setPlainText(f"오류: {result}")
            if self.main_window and hasattr(self.main_window, 'vue_bridge'):
                self.main_window.vue_bridge.showNotification.emit('error', f'인페인트 실패: {result}')
