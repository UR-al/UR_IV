"""Local-only, non-destructive adapter for the experimental relight node.

No input paths, URLs, models, GPU allocations or network access are accepted.
The one cached result may only be exported by its exact preview request ID.

Upload decoding, still-raster checks, PNG metadata preservation and exclusive
export are shared with hand reconstruction via core.local_image_io, so both
experimental editors apply one validation policy (strict MIME, fail on oversized
metadata instead of silently dropping it, RGB-only ICC).
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps

# comfy_custom_nodes 의 relight 노드는 render_relight_preview 안에서 import 한다. 여기서
# 최상단 import 하면 노드 팩 패키지 __init__ 전체(13개 모듈)가 앱 기동 경로에 올라, 노드 팩
# 한 파일의 SyntaxError·ImportError·노드 ID 충돌이 앱 전체 기동 실패로 번진다. 지연 import 면
# 그런 고장은 조명 미리보기 한 건의 오류 이벤트로 끝난다(워커의 try/except).
from core.local_image_io import (capture_png_metadata, decode_image_data_url, encode_png,
                                 export_exclusive_png, open_still_raster, png_data_url)

# Limits stay module globals and are read at call time (tests patch them).
MAX_PIXELS = 16_777_216
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_CHARS = 128 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")


def decode_relight_image(value, label, *, shape=None, normal=False, metadata=True):
    """data URL → (float32 픽셀, 메타데이터). 깊이·노멀·마스크 맵은 metadata=False —
    결과 PNG에 쓰지 않는 맵의 메타데이터 때문에 거부되지 않게 한다."""
    data = decode_image_data_url(value, label, max_bytes=MAX_FILE_BYTES)
    with open_still_raster(data, label, max_bytes=MAX_FILE_BYTES, max_pixels=MAX_PIXELS) as opened:
        if normal and opened.mode not in {"RGB", "RGBA"}:
            raise ValueError("노멀: RGB로 XYZ가 인코딩된 맵이 필요합니다.")
        # load() first: PNG text chunks after IDAT are only visible afterwards.
        opened.load()
        image = ImageOps.exif_transpose(opened)
        if shape is not None and image.size != shape:
            raise ValueError(f"{label}: 원본과 같은 해상도 {shape[0]}×{shape[1]} 맵을 사용하세요. 자동 리사이즈하지 않습니다.")
        # Preserve source transparency, including indexed PNG transparency.
        mode = "RGBA" if "A" in image.getbands() or "transparency" in opened.info else "RGB"
        pixels = np.asarray(image.convert(mode), dtype=np.float32) / 255.
        details = (capture_png_metadata(image, limit=MAX_METADATA_BYTES, source_mode=opened.mode)
                   if metadata else {"text": {}, "icc": None, "exif": None})
    return pixels, {**details, "sha256": hashlib.sha256(data).hexdigest()}


def encode_relight_png(pixels, *, metadata=None, diagnostic=False):
    image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    if diagnostic:
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return encode_png(image, metadata)


def render_relight_preview(request):
    from comfy_custom_nodes.ai_studio_forge_parity.relight import relight_image, settings_from

    values = [request.get(key, "") for key in ("image", "depth", "normals", "mask")]
    if any(not isinstance(value, str) for value in values) or sum(map(len, values)) > MAX_REQUEST_CHARS:
        raise ValueError("이미지·맵의 총 전송 크기는 128 MB 이하로 제한됩니다.")
    settings = settings_from(request.get("settings"))
    image, metadata = decode_relight_image(values[0], "원본")
    shape = (image.shape[1], image.shape[0])
    maps = {}
    for key, label, value in zip(("depth", "normals", "mask"), ("깊이", "노멀", "마스크"), values[1:]):
        if value:
            pixels, _ = decode_relight_image(value, label, shape=shape, normal=key == "normals", metadata=False)
            maps[key] = pixels if key == "normals" else pixels[..., 0]
    result = relight_image(image, settings=settings, **maps)
    provenance = {
        "tool": "AI Studio Pro experimental relight", "version": 1,
        "source_sha256": metadata["sha256"], "settings": settings,
        "geometry": result["geometry"],
        "cast_shadows": bool("depth" in maps and settings["shadow_strength"] > 0 and settings["shadow_length"] > 0),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata["text"]["ai_studio_relight"] = json.dumps(provenance, ensure_ascii=False)
    png = encode_relight_png(result["image"], metadata=metadata)
    diagnostics = {key: png_data_url(encode_relight_png(result[key], diagnostic=True))
                   for key in ("light", "normals", "shadow")}
    return {"png": png, "width": shape[0], "height": shape[1], "geometry": result["geometry"],
            "diagnostics": diagnostics, "sourceSha256": metadata["sha256"]}


def export_relight_png(data, output_root):
    # Explicit O_EXCL creation: no original or previous export can be replaced.
    # datetime/secrets are resolved here at call time so tests can patch this module.
    return export_exclusive_png(
        data, output_root, subdir="relight", prefix="relight",
        now=lambda: datetime.now(), token=lambda: secrets.token_hex(6),
        outside_message="조명 결과 폴더가 앱 출력 폴더 밖을 가리킵니다.",
        exhausted_message="새 결과 파일 이름을 만들지 못했습니다. 다시 저장하세요.",
    )


class RelightActionsMixin:
    def _relight_emit(self, event):
        if getattr(self, "_relight_closed", False):
            return
        signal = getattr(getattr(self, "vue_bridge", None), "relightEvent", None)
        if signal is not None:
            try:
                signal.emit(json.dumps(event, ensure_ascii=False))
            except RuntimeError:
                pass  # The Qt bridge may have been destroyed while CPU work ended.

    def _relight_error(self, event, exc):
        from core.error_handler import handle_error, sanitize_for_ui
        handle_error("E040", "실험 조명 편집", exc, notify=False)
        self._relight_emit({**event, "ok": False, "error": sanitize_for_ui(str(exc), 600)})

    def _ensure_relight_runtime(self):
        if not hasattr(self, "_relight_lock"):
            self._relight_lock = threading.RLock()
            self._relight_job = None
            self._relight_preview = None

    def _handle_relight_action(self, action, payload):
        if action in ("relight_preview", "relight_export", "relight_cancel"):
            return self._run_relight_action(action, payload)
        return False

    def _shutdown_relight(self):
        self._ensure_relight_runtime()
        with self._relight_lock:
            self._relight_closed = True
            self._relight_preview = None
            if self._relight_job:
                self._relight_job["cancel"].set()

    def _run_relight_action(self, action, payload):
        if getattr(self, "_relight_closed", False):
            return True
        request = dict(payload) if isinstance(payload, dict) else {}
        request_id = request.get("requestId", "")
        event = {"requestId": request_id if isinstance(request_id, str) else "", "action": action, "ok": False}
        try:
            if getattr(self, "web_mode", False):
                raise ValueError("실험 조명 편집은 로컬 앱에서만 사용할 수 있습니다.")
            if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
                raise ValueError("작업 식별자가 올바르지 않습니다. 패널을 다시 열어 주세요.")
            self._ensure_relight_runtime()
            with self._relight_lock:
                if action == "relight_cancel":
                    if self._relight_job and self._relight_job["requestId"] == request_id:
                        self._relight_job["cancel"].set()
                    if self._relight_preview and self._relight_preview["requestId"] == request_id:
                        self._relight_preview = None
                    self._relight_emit({**event, "ok": True, "canceled": True})
                    return True
                if self._relight_job:
                    if self._relight_job["requestId"] == request_id:
                        return True  # Idempotent transport redelivery, not a second CPU job.
                    raise ValueError("조명 작업 하나가 실행 중입니다. 완료된 뒤 다시 시도하세요.")
                if action == "relight_export":
                    cached = self._relight_preview
                    if not cached or cached["requestId"] != request.get("previewRequestId"):
                        raise ValueError("이 미리보기는 더 이상 유효하지 않습니다. 다시 미리보기를 만드세요.")
                    cached_png = cached["png"]
                    from config import OUTPUT_DIR
                    output_root = OUTPUT_DIR
                else:
                    self._relight_preview = None
                    cached_png = output_root = None
                job = {"requestId": request_id, "cancel": threading.Event()}
                self._relight_job = job
        except Exception as exc:
            self._relight_error(event, exc)
            return True

        def work():
            try:
                if job["cancel"].is_set():
                    with self._relight_lock:
                        if self._relight_job is job:
                            self._relight_job = None
                    self._relight_emit({**event, "canceled": True, "error": "조명 작업을 취소했습니다."})
                    return
                if action == "relight_preview":
                    result = render_relight_preview(request)
                    output = {**event, "ok": True, "image": png_data_url(result["png"]),
                              **{key: value for key, value in result.items() if key != "png"}}
                    with self._relight_lock:
                        if job["cancel"].is_set() or getattr(self, "_relight_closed", False):
                            if self._relight_job is job:
                                self._relight_job = None
                            self._relight_emit({**event, "canceled": True, "error": "변경된 입력의 이전 미리보기를 폐기했습니다."})
                            return
                        self._relight_preview = {"requestId": request_id, "png": result["png"]}
                        self._relight_job = None
                        self._relight_emit(output)
                elif not job["cancel"].is_set():
                    path = export_relight_png(cached_png, output_root)
                    with self._relight_lock:
                        self._relight_job = None
                        self._relight_emit({**event, "ok": True, "path": path})
            except Exception as exc:
                with self._relight_lock:
                    if self._relight_job is job:
                        self._relight_job = None
                self._relight_error(event, exc)
            finally:
                with self._relight_lock:
                    if self._relight_job is job:
                        self._relight_job = None

        try:
            thread = threading.Thread(target=work, daemon=True, name="ai-studio-relight")
            thread.start()
        except Exception as exc:
            with self._relight_lock:
                self._relight_job = None
            self._relight_error(event, exc)
        return True
