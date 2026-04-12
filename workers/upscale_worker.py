# workers/upscale_worker.py
import os
import base64
from PyQt6.QtCore import QThread, pyqtSignal
from backends import get_backend


def _build_adetailer_slot(model: str, confidence: float = 0.3, denoise: float = 0.4, prompt: str = '') -> dict:
    """ADetailer 슬롯 딕셔너리 (공식 REST API 스펙)"""
    return {
        "ad_model": model,
        "ad_model_classes": "",
        "ad_tab_enable": True,
        "ad_prompt": prompt,
        "ad_negative_prompt": "",
        "ad_confidence": confidence,
        "ad_mask_filter_method": "Area",
        "ad_mask_k": 0,
        "ad_mask_min_ratio": 0.0,
        "ad_mask_max_ratio": 1.0,
        "ad_dilate_erode": 4,
        "ad_x_offset": 0,
        "ad_y_offset": 0,
        "ad_mask_merge_invert": "None",
        "ad_mask_blur": 4,
        "ad_denoising_strength": denoise,
        "ad_inpaint_only_masked": True,
        "ad_inpaint_only_masked_padding": 32,
        "ad_use_inpaint_width_height": False,
        "ad_inpaint_width": 512,
        "ad_inpaint_height": 512,
        "ad_use_steps": False,
        "ad_steps": 28,
        "ad_use_cfg_scale": False,
        "ad_cfg_scale": 7.0,
        "ad_use_checkpoint": False,
        "ad_checkpoint": None,
        "ad_use_vae": False,
        "ad_vae": None,
        "ad_use_sampler": False,
        "ad_sampler": "DPM++ 2M Karras",
        "ad_scheduler": "Use same scheduler",
        "ad_use_noise_multiplier": False,
        "ad_noise_multiplier": 1.0,
        "ad_use_clip_skip": False,
        "ad_clip_skip": 1,
        "ad_restore_face": False,
        "ad_controlnet_model": "None",
        "ad_controlnet_module": "None",
        "ad_controlnet_weight": 1.0,
        "ad_controlnet_guidance_start": 0.0,
        "ad_controlnet_guidance_end": 1.0,
    }


def _build_empty_adetailer_slot() -> dict:
    """빈 ADetailer 슬롯 (하위 호환)"""
    return _build_adetailer_slot(model="None")


def _image_to_base64(image_path: str) -> str:
    """이미지 파일을 base64 문자열로 변환"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _save_base64_image(b64_data: str, output_path: str):
    """base64 데이터를 이미지 파일로 저장"""
    img_bytes = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(img_bytes)


class BatchUpscaleWorker(QThread):
    """배치 업스케일/ADetailer 워커"""
    single_finished = pyqtSignal(int, bool, str)  # index, success, message
    all_finished = pyqtSignal()
    progress = pyqtSignal(int, int)  # current, total

    def __init__(self, image_paths: list, settings: dict):
        """
        settings = {
            'mode': 'upscale_only' | 'adetailer_only' | 'both',
            'upscaler_name': str,
            'scale_mode': 'factor' | 'size',
            'scale_factor': float,
            'target_width': int,
            'target_height': int,
            'ad_model': str,
            'ad_confidence': float,
            'ad_denoise': float,
            'ad_prompt': str,
            'output_folder': str,
        }
        """
        super().__init__()
        self.image_paths = image_paths
        self.settings = settings
        self._stop_requested = False

    def run(self):
        """배치 처리 실행"""
        total = len(self.image_paths)

        for i, path in enumerate(self.image_paths):
            if self._stop_requested:
                break

            self.progress.emit(i, total)
            try:
                b64_image = _image_to_base64(path)
                result_b64 = b64_image
                mode = self.settings['mode']
                backend = get_backend()

                # 업스케일
                if mode in ('upscale_only', 'both'):
                    result_b64 = backend.upscale(result_b64, self.settings)

                # ADetailer
                if mode in ('adetailer_only', 'both'):
                    result_b64 = backend.adetailer(result_b64, self.settings)

                # 저장
                output_folder = self.settings['output_folder']
                basename = os.path.splitext(os.path.basename(path))[0]
                suffix = "_upscaled" if mode == 'upscale_only' else "_ad" if mode == 'adetailer_only' else "_upscaled_ad"
                output_path = os.path.join(output_folder, f"{basename}{suffix}.png")
                _save_base64_image(result_b64, output_path)

                self.single_finished.emit(i, True, os.path.basename(output_path))

            except Exception as e:
                self.single_finished.emit(i, False, str(e))

        self.progress.emit(total, total)
        self.all_finished.emit()

    def request_stop(self):
        """처리 중지 요청"""
        self._stop_requested = True
