"""Anima Tile & Repair 요청 만들기 — Vue ``tile_repair_run`` 페이로드 → sam-extra 라우트 본문. Qt·네트워크 없음.

라우트: sam-extra ``POST /sam-extra/tile-repair`` (확장 sam3ext/tile_repair_api.py). 확장은 모르는 키를 400 으로
거절하므로 여기 키 이름이 곧 계약이다(``ROUTE_KEYS`` — tests/test_tile_repair_request.py 가 확장 소스와 대조).

기본값·범위는 원본 그대로다(확장 패널 = 라우트 기본값과 같다):
  - kohya-ss/sd-scripts@690ea7f9 anima_minimal_inference_control_net_lllite.py:127-134 — negative "", 50 steps,
    CFG 3.5, flow shift 5.0 / :167-170 — lllite_multiplier 1.0
  - kohya-ss/ComfyUI-Anima-LLLite@b7495bd8 nodes.py:130 — strength −10..10 step .01
  - 짧은 변 1024(256..4096, 32 배수) — 원본 비율 유지(ComfyUI ResizeImagesByShorterEdge), 확장 패널과 같다
모델 칸(model·dit·text_encoder·vae)은 비워 두면 보내지 않는다 — 확장이 자기 기본값(최신 Tile & Repair v20,
Qwen3 0.6B TE, Qwen-Image VAE, Forge 현재 DiT)을 쓴다.

원본 이미지는 로컬 파일 경로(I2I 원본) 또는 업로드 data URL 로 받아 **받은 바이트 그대로**(PNG·JPEG·WebP) base64 로
보낸다. 라우트가 세 형식을 다 받고, 확장(run_tile_repair)과 sd-scripts 가 PIL 로 열어 ``convert("RGB")`` 하므로 PNG 로
다시 쓸 이유가 없다 — 다시 쓰면 64 MB 아래의 JPEG 가 64 MB 넘는 PNG 가 되어 Forge 에서 413 이 나고, CMYK JPEG 는
PNG 로 저장조차 안 된다. 앱은 단일 정지 이미지인지·크기·픽셀 한도(확장과 같은 값)만 미리 확인한다.
출력 크기도 확장과 같은 식(``output_size``)으로 미리 재서 64 MP 를 넘으면 올리기 전에 거절한다 — 가는 원본은
비율을 따라 수십억 픽셀이 될 수 있고, 확장도 같은 한도에서 400 으로 거절한다.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from PIL import Image

from core.local_image_io import decode_image_data_url, open_still_raster

# 확장 패널 프롬프트 기본값(sam3ext/ui_anima.py build_anima_panel 의 positive) = 라우트 기본값.
DEFAULT_PROMPT = (
    "repair the low-quality anime image, reduce blur and compression artifacts, "
    "preserve the original composition"
)
DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "prompt": DEFAULT_PROMPT,
    "negative_prompt": "",
    "steps": 50,
    "cfg_scale": 3.5,
    "flow_shift": 5.0,
    "multiplier": 1.0,
    "short_side": 1024,
    "seed": -1,
    "unload_forge_before": True,
})
RANGES: Mapping[str, tuple] = MappingProxyType({
    "steps": (1, 150),
    "cfg_scale": (0.0, 20.0),
    "flow_shift": (0.0, 30.0),
    "multiplier": (-10.0, 10.0),
    "short_side": (256, 4096),
})
_INTEGER_FIELDS = frozenset({"steps", "short_side"})
MODEL_FIELDS = ("model", "dit", "text_encoder", "vae")
#: 확장 라우트가 받는 키 전부 — 이 밖의 키는 확장이 400 으로 거절한다.
ROUTE_KEYS = frozenset({"image", *DEFAULTS, *MODEL_FIELDS})

MAX_SOURCE_BYTES = 64 * 1024 * 1024       # 확장 MAX_IMAGE_BYTES 와 같다
MAX_SOURCE_PIXELS = 64 * 1024 * 1024      # 확장 MAX_SOURCE_PIXELS 와 같다(8192²)
MAX_OUTPUT_PIXELS = MAX_SOURCE_PIXELS     # 확장 MAX_OUTPUT_PIXELS 와 같다
SIZE_MULTIPLE = 32                        # 확장 anima_core.TILE_REPAIR_SIZE_MULTIPLE
MIN_SIDE = 256                            # 확장 anima_core.TILE_REPAIR_MIN_SIDE
MAX_TEXT_CHARS = 20_000
MAX_SEED = 2**63 - 1
SOURCE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_LABEL = "Tile & Repair 원본"


def _number(settings: Mapping[str, Any], key: str):
    value = settings.get(key, DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Tile & Repair: {key} 는 숫자여야 합니다")
    if key in _INTEGER_FIELDS:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"Tile & Repair: {key} 는 정수여야 합니다")
            value = int(value)
    else:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Tile & Repair: {key} 가 올바르지 않습니다")
    low, high = RANGES[key]
    if not low <= value <= high:
        raise ValueError(f"Tile & Repair: {key} 는 {low}~{high} 범위여야 합니다 (받은 값 {value})")
    return value


def _text(settings: Mapping[str, Any], key: str) -> str:
    value = settings.get(key, DEFAULTS[key])
    if not isinstance(value, str):
        raise ValueError(f"Tile & Repair: {key} 는 문자열이어야 합니다")
    if len(value) > MAX_TEXT_CHARS:
        raise ValueError(f"Tile & Repair: {key} 가 {MAX_TEXT_CHARS}자를 넘습니다")
    return value


def normalize_settings(settings: Any) -> dict:
    """Vue 설정 → 라우트 본문(이미지 제외). 범위를 벗어나면 ValueError(사용자에게 보여도 되는 문구)."""
    settings = settings if isinstance(settings, Mapping) else {}
    body: dict = {key: _number(settings, key) for key in RANGES}
    prompt = _text(settings, "prompt")
    if not prompt.strip():
        raise ValueError("Tile & Repair: 프롬프트가 비어 있습니다")
    body["prompt"] = prompt
    body["negative_prompt"] = _text(settings, "negative_prompt")
    seed = settings.get("seed", DEFAULTS["seed"])
    if isinstance(seed, float) and seed.is_integer():
        seed = int(seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or not -1 <= seed <= MAX_SEED:
        raise ValueError("Tile & Repair: 시드는 -1(랜덤) 또는 0 이상의 정수여야 합니다")
    body["seed"] = seed
    unload = settings.get("unload_forge_before", DEFAULTS["unload_forge_before"])
    if not isinstance(unload, bool):
        raise ValueError("Tile & Repair: unload_forge_before 는 true/false 여야 합니다")
    body["unload_forge_before"] = unload
    for key in MODEL_FIELDS:
        value = settings.get(key)
        if value is None or value == "":
            continue   # 확장 기본값
        if not isinstance(value, str) or len(value) > 512:
            raise ValueError(f"Tile & Repair: {key} 이름이 올바르지 않습니다")
        body[key] = value
    return body


def _checked_still(data: bytes) -> bytes:
    """단일 정지 PNG/JPEG/WebP 이고 확장 한도(64 MB · 64 MP) 안이면 **같은 바이트**를 돌려준다."""
    with open_still_raster(data, _LABEL, max_bytes=MAX_SOURCE_BYTES, max_pixels=MAX_SOURCE_PIXELS):
        return data


def load_source_image(payload: Mapping[str, Any]) -> bytes:
    """``image_path``(로컬 파일) 또는 ``image``(data URL) → 원본 이미지 바이트. 경로가 있으면 경로가 이긴다."""
    raw_path = payload.get("image_path")
    if isinstance(raw_path, str) and raw_path.strip():
        from core.path_safety import safe_input_path

        path = safe_input_path(raw_path.strip(), allowed_exts=SOURCE_EXTS)
        if path is None:
            raise ValueError(f"{_LABEL}: 파일을 읽을 수 없습니다 (PNG/JPEG/WebP 파일만)")
        size = Path(path).stat().st_size
        if size > MAX_SOURCE_BYTES:
            raise ValueError(f"{_LABEL}: 64 MB 이하의 이미지를 사용하세요")
        return _checked_still(Path(path).read_bytes())
    data_url = payload.get("image")
    if isinstance(data_url, str) and data_url:
        return _checked_still(decode_image_data_url(data_url, _LABEL, max_bytes=MAX_SOURCE_BYTES))
    raise ValueError(f"{_LABEL}: 이미지를 먼저 올리세요")


def output_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """확장 ``anima_core.tile_repair_size`` 와 같은 출력 ``(너비, 높이)``.

    짧은 변이 ``short_side`` 가 되고 긴 변은 원본 비율을 따른다(``int(긴 변 × short_side / 짧은 변)``).
    두 변은 32 배수로 내리되 256 아래로는 내리지 않는다.
    """
    w, h, edge = int(width), int(height), int(short_side)
    if w <= 0 or h <= 0:
        raise ValueError(f"{_LABEL}: 크기가 올바르지 않습니다 ({w}x{h})")
    if w < h:
        new_w, new_h = edge, int(h * (edge / w))
    else:
        new_h, new_w = edge, int(w * (edge / h))
    return (max(MIN_SIDE, new_w // SIZE_MULTIPLE * SIZE_MULTIPLE),
            max(MIN_SIDE, new_h // SIZE_MULTIPLE * SIZE_MULTIPLE))


def check_output_size(data: bytes, short_side: int) -> tuple[int, int]:
    """원본 바이트의 출력 크기가 64 MP 이하면 그 크기를 돌려준다. 넘으면 ValueError(사용자 문구)."""
    with Image.open(io.BytesIO(data)) as opened:      # 헤더만 읽는다
        width, height = opened.size
    out_w, out_h = output_size(width, height, short_side)
    if out_w * out_h > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"Tile & Repair: 결과가 {out_w}x{out_h} 가 됩니다 — {width}x{height} 원본의 비율을 짧은 변 "
            f"{short_side} 로 늘리면 64 MP 를 넘습니다. 짧은 변을 줄이거나 이미지를 잘라서 쓰세요"
        )
    return out_w, out_h


def build_route_body(payload: Mapping[str, Any]) -> dict:
    """Vue ``tile_repair_run`` 페이로드 → ``POST /sam-extra/tile-repair`` 본문."""
    body = normalize_settings(payload.get("settings"))
    source = load_source_image(payload)
    check_output_size(source, body["short_side"])
    body["image"] = base64.b64encode(source).decode("ascii")
    return body


__all__ = [
    "DEFAULTS", "DEFAULT_PROMPT", "MODEL_FIELDS", "RANGES", "ROUTE_KEYS", "build_route_body",
    "check_output_size", "load_source_image", "normalize_settings", "output_size",
]
