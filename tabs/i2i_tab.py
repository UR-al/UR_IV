# tabs/i2i_tab.py
"""
Image-to-Image (img2img) 탭
- 이미지 입력 + 프롬프트 → 변형 이미지 생성
- Denoising Strength, Resize Mode 지원
"""
import os
import time
import random
import base64
from io import BytesIO

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QLineEdit, QGroupBox, QComboBox, QFileDialog, QMessageBox,
    QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont
from PIL import Image

from config import OUTPUT_DIR, WEBUI_API_URL
from workers.generation_worker import Img2ImgFlowWorker
from utils.theme_manager import get_theme_manager, get_color


class Img2ImgTab(QWidget):
    """img2img 탭"""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.current_image_path = None
        self.current_base64 = None
        self.gen_worker = None
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 왼쪽: 입력 이미지 + 설정 (left_stack에 삽입됨) ---
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(8)

        title = QLabel("Image-to-Image")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {get_color('accent')};")
        left_layout.addWidget(title)

        # 이미지 입력 영역
        img_group = QGroupBox("입력 이미지")
        img_layout = QVBoxLayout(img_group)

        self.input_image_label = QLabel("이미지를 드래그하거나\n더블클릭 또는 '열기' 버튼을 누르세요.")
        self.input_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c = get_theme_manager().get_colors()
        self.input_image_label.setStyleSheet(
            f"border: 2px dashed {c['border']}; border-radius: 10px; color: {c['text_muted']}; min-height: 200px;"
        )
        self.input_image_label.setMinimumHeight(200)
        self.input_image_label.mouseDoubleClickEvent = lambda e: self._open_image()
        img_layout.addWidget(self.input_image_label)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("📂 이미지 열기")
        self.btn_open.setFixedHeight(38)
        self.btn_open.setMinimumWidth(110)
        self.btn_open.clicked.connect(self._open_image)
        btn_row.addWidget(self.btn_open)

        self.btn_paste_from_t2i = QPushButton("📋 T2I 결과 가져오기")
        self.btn_paste_from_t2i.setFixedHeight(38)
        self.btn_paste_from_t2i.setMinimumWidth(150)
        self.btn_paste_from_t2i.clicked.connect(self._paste_from_t2i)
        btn_row.addWidget(self.btn_paste_from_t2i)

        img_layout.addLayout(btn_row)
        left_layout.addWidget(img_group)

        # 프롬프트
        prompt_group = QGroupBox("프롬프트")
        prompt_layout = QVBoxLayout(prompt_group)

        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_text = QTextEdit()
        self.prompt_text.setFixedHeight(60)
        self.prompt_text.setPlaceholderText("프롬프트 입력 (비우면 T2I 프롬프트 사용)")
        prompt_layout.addWidget(self.prompt_text)

        prompt_layout.addWidget(QLabel("Negative Prompt:"))
        self.neg_prompt_text = QTextEdit()
        self.neg_prompt_text.setFixedHeight(40)
        self.neg_prompt_text.setPlaceholderText("네거티브 프롬프트")
        prompt_layout.addWidget(self.neg_prompt_text)

        left_layout.addWidget(prompt_group)

        # 설정
        settings_group = QGroupBox("생성 설정")
        settings_layout = QVBoxLayout(settings_group)

        # Denoising Strength
        denoise_row = QHBoxLayout()
        denoise_row.addWidget(QLabel("Denoising Strength:"))
        self.denoise_input = QLineEdit("0.75")
        self.denoise_input.setFixedWidth(80)
        self.denoise_input.setToolTip("0.0 (변화 없음) ~ 1.0 (완전 새로 생성)")
        denoise_row.addWidget(self.denoise_input)
        denoise_row.addStretch()
        settings_layout.addLayout(denoise_row)

        # Resize Mode
        resize_row = QHBoxLayout()
        resize_row.addWidget(QLabel("Resize Mode:"))
        self.resize_combo = QComboBox()
        self.resize_combo.addItems([
            "Just resize",
            "Crop and resize",
            "Resize and fill",
            "Just resize (latent upscale)"
        ])
        resize_row.addWidget(self.resize_combo)
        resize_row.addStretch()
        settings_layout.addLayout(resize_row)

        # 해상도
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width:"))
        self.width_input = QLineEdit("1024")
        self.width_input.setFixedWidth(70)
        size_row.addWidget(self.width_input)
        size_row.addWidget(QLabel("Height:"))
        self.height_input = QLineEdit("1024")
        self.height_input.setFixedWidth(70)
        size_row.addWidget(self.height_input)
        size_row.addStretch()
        settings_layout.addLayout(size_row)

        # Steps / CFG
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Steps:"))
        self.steps_input = QLineEdit("20")
        self.steps_input.setFixedWidth(50)
        param_row.addWidget(self.steps_input)
        param_row.addWidget(QLabel("CFG:"))
        self.cfg_input = QLineEdit("7.0")
        self.cfg_input.setFixedWidth(50)
        param_row.addWidget(self.cfg_input)
        param_row.addWidget(QLabel("Seed:"))
        self.seed_input = QLineEdit("-1")
        self.seed_input.setFixedWidth(80)
        param_row.addWidget(self.seed_input)
        param_row.addStretch()
        settings_layout.addLayout(param_row)

        left_layout.addWidget(settings_group)

        # 생성 버튼
        self.btn_generate = QPushButton("🎨 img2img 생성")
        self.btn_generate.setFixedHeight(50)
        self.btn_generate.setStyleSheet(f"""
            QPushButton {{
                background-color: #5865F2; color: white;
                font-weight: bold; font-size: 14px; border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #4752C4; }}
            QPushButton:disabled {{ background-color: {c['disabled_bg']}; color: {c['disabled_text']}; }}
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

        # --- 오른쪽: 결과 이미지 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_title = QLabel("생성 결과")
        result_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        result_title.setStyleSheet(f"color: {c['text_primary']};")
        right_layout.addWidget(result_title)

        self.result_label = QLabel("생성된 이미지가 여기에 표시됩니다.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet(
            f"background-color: {c['bg_secondary']}; border-radius: 8px; color: {c['text_muted']};"
        )
        self.result_label.setMinimumSize(400, 400)
        right_layout.addWidget(self.result_label, 1)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(80)
        self.info_text.setPlaceholderText("생성 정보")
        right_layout.addWidget(self.info_text)

        # 스크롤 영역에 왼쪽 패널 설정
        self.left_scroll.setWidget(left_panel)

        # 왼쪽 패널은 left_stack에서 관리 — 여기서는 결과 영역만 배치
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

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 열기", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._load_image(path)

    def _paste_from_t2i(self):
        """T2I 탭의 현재 결과 이미지 가져오기"""
        if self.main_window and hasattr(self.main_window, 'current_image_path'):
            path = self.main_window.current_image_path
            if path and os.path.exists(path):
                self._load_image(path)
                # 프롬프트도 가져오기
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
        """이미지 로드 및 base64 인코딩"""
        self.current_image_path = path

        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.input_image_label.setPixmap(
                pixmap.scaled(
                    self.input_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            self.input_image_label.setStyleSheet("border: none;")

        # Base64 인코딩
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        self.current_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 원본 해상도 표시
        self.width_input.setText(str(img.width))
        self.height_input.setText(str(img.height))

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
        self.width_input.setText(str(payload.get('width', 1024)))
        self.height_input.setText(str(payload.get('height', 1024)))

    def _on_generate(self):
        """img2img 생성 시작"""
        if not self.current_base64:
            QMessageBox.warning(self, "경고", "먼저 입력 이미지를 로드하세요.")
            return

        # 프롬프트 (비어있으면 T2I에서 가져오기)
        prompt = self.prompt_text.toPlainText().strip()
        neg_prompt = self.neg_prompt_text.toPlainText().strip()
        if not prompt and self.main_window and hasattr(self.main_window, 'total_prompt_display'):
            prompt = self.main_window.total_prompt_display.toPlainText()
        if not neg_prompt and self.main_window and hasattr(self.main_window, 'neg_prompt_text'):
            neg_prompt = self.main_window.neg_prompt_text.toPlainText()

        payload = {
            "init_images": [self.current_base64],
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "denoising_strength": float(self.denoise_input.text()),
            "resize_mode": self.resize_combo.currentIndex(),
            "steps": int(self.steps_input.text()),
            "cfg_scale": float(self.cfg_input.text()),
            "seed": int(self.seed_input.text()),
            "width": int(self.width_input.text()),
            "height": int(self.height_input.text()),
            "send_images": True,
            "save_images": True,
            "alwayson_scripts": {},
        }

        # 모델
        model_name = ""
        if self.main_window and hasattr(self.main_window, 'model_combo'):
            model_name = self.main_window.model_combo.currentText()

        self.btn_generate.setText("⏳ 생성 중...")
        self.btn_generate.setEnabled(False)
        self.result_label.setText("🎨 img2img 생성 중...\n\n잠시만 기다려주세요.")

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
            "width": int(self.width_input.text()),
            "height": int(self.height_input.text()),
            "send_images": True,
            "save_images": True,
        }

        if self.main_window and hasattr(self.main_window, 'queue_panel'):
            self.main_window.queue_panel.add_single_item(payload)
            if hasattr(self.main_window, 'show_status'):
                self.main_window.show_status("📋 I2I 설정이 대기열에 추가되었습니다.")

    def _on_generation_finished(self, result, gen_info):
        """생성 완료"""
        self.btn_generate.setText("🎨 img2img 생성")
        self.btn_generate.setEnabled(True)

        if isinstance(result, bytes):
            pixmap = QPixmap()
            pixmap.loadFromData(result)
            self.result_label.setPixmap(
                pixmap.scaled(
                    self.result_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

            # 저장
            filename = f"i2i_{int(time.time())}_{random.randint(100, 999)}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(result)

            self.info_text.setPlainText(
                f"저장: {filepath}\n"
                f"Seed: {gen_info.get('seed', '?')}, "
                f"Steps: {gen_info.get('steps', '?')}, "
                f"CFG: {gen_info.get('cfg_scale', '?')}"
            )

            # 메인 윈도우 갤러리에도 추가
            if self.main_window and hasattr(self.main_window, 'add_image_to_gallery'):
                self.main_window.add_image_to_gallery(filepath)
        else:
            self.result_label.setText(f"❌ 생성 실패\n\n{result}")
            self.info_text.setPlainText(f"오류: {result}")
