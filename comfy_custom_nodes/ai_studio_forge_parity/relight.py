"""Independent, deterministic 2.5D relighting; no model downloads or global state.

IMAGE arrays use RGB [0,1]. Depth uses white=near; normal maps encode XYZ in RGB
with +Y pointing up and +Z toward the camera. Without geometry, luminance is
only an explicitly approximate height field, not an estimated physical depth.
"""
from __future__ import annotations

import math
import numpy as np


DEFAULTS = {"azimuth": -35., "elevation": 45., "strength": 0.5, "ambient": 0.6,
            "depth_scale": 1., "shadow_strength": 0.2, "shadow_length": 32.,
            "shadow_softness": 3., "shadow_cleanup": 0.}
RANGES = {"azimuth": (-180., 180.), "elevation": (5., 85.), "strength": (0., 1.),
          "ambient": (0.1, 1.), "depth_scale": (0.1, 4.), "shadow_strength": (0., 1.),
          "shadow_length": (0., 128.), "shadow_softness": (0., 24.), "shadow_cleanup": (0., 0.7)}


def settings_from(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("조명 설정은 객체여야 합니다.")
    result = {}
    for key, default in DEFAULTS.items():
        value = raw.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key}: 숫자를 입력하세요.")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: 숫자를 입력하세요.") from exc
        low, high = RANGES[key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key}: {low}~{high} 범위를 확인하세요.")
        result[key] = value
    return result


def _image(value, label, shape=None):
    data = np.asarray(value, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3 or data.shape[-1] not in (1, 3, 4) or min(data.shape[:2]) < 2:
        raise ValueError(f"{label}: H×W×1/3/4 이미지가 필요합니다 (최소 2×2).")
    if shape and data.shape[:2] != shape:
        raise ValueError(f"{label}: 원본과 같은 해상도의 맵이 필요합니다.")
    if data.shape[0] * data.shape[1] > 16_777_216 or not np.isfinite(data).all():
        raise ValueError(f"{label}: 이미지 크기 또는 유한한 픽셀 값을 확인하세요.")
    if (data < 0).any() or (data > 1).any():
        raise ValueError(f"{label}: 픽셀 범위는 0~1이어야 합니다.")
    return data


def _blur(value, radius):
    if radius <= 0:
        return value.copy()
    # OpenCV is an existing app dependency; no animation or inference library.
    import cv2
    # C 연속 배열로 넘긴다 — OpenCV 5.0 은 비연속 배열(전치·채널 뷰)을 출력 배열로 받지 않는다(mask_ops.make_overlay).
    return cv2.GaussianBlur(np.ascontiguousarray(value), (0, 0), float(radius), borderType=cv2.BORDER_REFLECT_101)


def relight_image(image, *, depth=None, normals=None, mask=None, settings=None):
    source = _image(image, "원본")
    if source.shape[-1] < 3:
        raise ValueError("원본은 RGB 또는 RGBA여야 합니다.")
    cfg = settings_from(settings)
    shape = source.shape[:2]
    rgb = source[..., :3]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    height = (_image(depth, "깊이", shape)[..., 0] if depth is not None
              else _blur(luminance, 3.))
    geometry = "normal" if normals is not None else "depth" if depth is not None else "luminance-approximation"
    if normals is not None:
        normal_map = _image(normals, "노멀", shape)
        if normal_map.shape[-1] < 3:
            raise ValueError("노멀은 RGB XYZ 맵이어야 합니다.")
        normal = normal_map[..., :3] * 2 - 1
    else:
        dy, dx = np.gradient(height)
        scale = min(shape) * 0.08 * cfg["depth_scale"]
        normal = np.stack((-dx * scale, dy * scale, np.ones(shape, dtype=np.float32)), axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, np.maximum(norm, 1e-6))
    normal = np.where(norm < 1e-6, np.array([0., 0., 1.], dtype=np.float32), normal)
    az, el = math.radians(cfg["azimuth"]), math.radians(cfg["elevation"])
    light = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)], dtype=np.float32)
    diffuse = np.clip(normal @ light, 0, 1)
    light_map = cfg["ambient"] + (1 - cfg["ambient"]) * diffuse
    neutral = cfg["ambient"] + (1 - cfg["ambient"]) * light[2]
    shadow = np.zeros(shape, dtype=np.float32)
    # Height-field ray marching is bounded and never wraps pixels at edges.
    # Only a supplied depth map justifies cast shadows; brightness is not depth.
    if depth is not None and cfg["shadow_length"] > 0 and cfg["shadow_strength"] > 0:
        h, w = shape
        for distance in np.linspace(1, cfg["shadow_length"], 16):
            ox, oy = round(math.sin(az) * distance), round(-math.cos(az) * distance)
            if abs(ox) >= w or abs(oy) >= h:
                continue
            x0, x1 = max(0, -ox), min(w, w - ox)
            y0, y1 = max(0, -oy), min(h, h - oy)
            rise = distance * math.tan(el) / (min(shape) * cfg["depth_scale"])
            blocked = height[y0+oy:y1+oy, x0+ox:x1+ox] > height[y0:y1, x0:x1] + rise
            shadow[y0:y1, x0:x1] = np.maximum(shadow[y0:y1, x0:x1], blocked)
        shadow = _blur(shadow, cfg["shadow_softness"])
    gain = light_map / max(float(neutral), 0.1) * (1 - cfg["shadow_strength"] * shadow)
    cleanup = np.maximum(_blur(luminance, 12.) - luminance, 0) * cfg["shadow_cleanup"]
    lit = np.clip((rgb + cleanup[..., None]) * gain[..., None], 0, 1)
    coverage = np.ones(shape, dtype=np.float32)
    if mask is not None:
        coverage = _image(mask, "마스크", shape)[..., 0]
    if source.shape[-1] == 4:
        coverage = coverage * source[..., 3]
    mix = (cfg["strength"] * coverage)[..., None]
    result = source.copy()
    result[..., :3] = np.where(mix == 0, rgb, rgb * (1 - mix) + lit * mix)
    return {"image": result, "light": np.repeat(light_map[..., None], 3, axis=-1),
            "normals": (normal + 1) * .5, "shadow": shadow, "geometry": geometry}


class AIStudioRelight:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"image": ("IMAGE",)}
        required.update({key: ("FLOAT", {"default": value, "min": RANGES[key][0], "max": RANGES[key][1]})
                         for key, value in DEFAULTS.items()})
        return {"required": required, "optional": {"depth": ("IMAGE",), "normals": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "light_map", "normal_preview", "shadow")
    FUNCTION = "apply"
    CATEGORY = "AI Studio/Experimental"

    def apply(self, image, depth=None, normals=None, mask=None, **settings):
        import torch
        if image.ndim != 4 or not 1 <= image.shape[0] <= 32:
            raise ValueError("IMAGE는 1~32개 배치의 BHWC 텐서여야 합니다.")
        # Four diagnostic/result outputs coexist. Bound the entire batch, not
        # each image separately, before allocating CPU or GPU result buffers.
        if int(image.shape[0]) * int(image.shape[1]) * int(image.shape[2]) > 4_194_304:
            raise ValueError("조명 노드의 전체 배치는 4 MP 이하로 나누세요 (예: 2048×2048 1장 / 1024×1024 4장).")
        for label, value in (("depth", depth), ("normals", normals), ("mask", mask)):
            if value is not None and value.shape[0] not in (1, image.shape[0]):
                raise ValueError(f"{label}: 맵 배치는 1개 또는 원본 배치와 같아야 합니다.")
        def array(value, index):
            if value is None:
                return None
            return value[0 if value.shape[0] == 1 else index].detach().float().cpu().numpy()
        outputs = [[], [], [], []]
        for index in range(image.shape[0]):
            import comfy.model_management
            comfy.model_management.throw_exception_if_processing_interrupted()
            result = relight_image(array(image, index), depth=array(depth, index),
                                   normals=array(normals, index), mask=array(mask, index), settings=settings)
            for collection, key in zip(outputs, ("image", "light", "normals", "shadow")):
                collection.append(torch.from_numpy(result[key].copy()))
        return tuple(torch.stack(values).to(image.device, image.dtype) for values in outputs)


NODE_CLASS_MAPPINGS = {"AIStudioRelight": AIStudioRelight}
