"""Face detection for SAM3 face crop on ComfyUI: ultralytics YOLO box models, as sam-extra and ADetailer use them.

Port of sam-extra's ``sam3ext/face_detect.py`` (v0.34.0; sam-extra's own code, MIT for this pack, see
``THIRD_PARTY_NOTICES.md``). The call is the same: ``ultralytics.YOLO(<file>)`` -> ``model(image, conf=…, device=…)``
and the ``boxes.xyxy`` / ``boxes.conf`` of the first result. The device rule is the same: ``"cpu"`` when SAM3 runs on
the CPU, otherwise ``""`` (ultralytics picks the current CUDA device; a device number would make ultralytics call
``torch.cuda.set_device`` and change the process default). Nothing is downloaded and nothing is cached: the model is
read for each node run and released after it.

Where the model comes from (sam-extra reads Forge ``models/adetailer``):

* an absolute path is used as it is (the app sends the file of its Forge ``models/adetailer``, like the SAM3
  checkpoint);
* a bare name is looked up in ComfyUI's ``ultralytics_bbox`` and ``ultralytics`` folders when another pack (Impact
  Subpack) registered them, then in ``models/ultralytics/bbox``, ``models/ultralytics`` and ``models/adetailer``.

ultralytics is not a ComfyUI dependency. ``detector_status`` tells (without importing it) whether this ComfyUI can
import it; ``ForgeNeoSAM3Mask`` publishes that in its informational ``face_crop_detector`` COMBO and the app refuses a
face-crop graph without it. This module imports neither torch nor ultralytics when it is loaded.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "face_yolov8n.pt"
DEFAULT_CONFIDENCE = 0.3
MODEL_SUFFIX = ".pt"
# The informational ``face_crop_detector`` COMBO of ForgeNeoSAM3Mask (core/sam3_face_crop.COMFY_DETECTOR_* in the app).
DETECTOR_INPUT = "face_crop_detector"
DETECTOR_READY = "ultralytics"
DETECTOR_MISSING = "(ultralytics not installed)"
# Folder registrations of other packs (Impact Subpack), then folders under ComfyUI's models directory.
FOLDER_KEYS = ("ultralytics_bbox", "ultralytics")
MODEL_SUBDIRS = (("ultralytics", "bbox"), ("ultralytics",), ("adetailer",))


class FaceDetectorUnavailable(RuntimeError):
    """The face detector cannot run (no model file, no ultralytics, a name ultralytics would download); the message
    says what to do."""


@dataclass
class FaceDetection:
    """Detector output: image-coordinate boxes (x0, y0, x1, y1), their confidences and the model file read."""

    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    model_path: str = ""


def detector_status() -> str:
    """``DETECTOR_READY`` when this Python can import ultralytics (found, not imported), else ``DETECTOR_MISSING``."""
    try:
        found = importlib.util.find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        found = False
    return DETECTOR_READY if found else DETECTOR_MISSING


def detector_input() -> tuple[list[str], dict[str, Any]]:
    """Spec of the informational ``DETECTOR_INPUT`` COMBO: this ComfyUI's ``detector_status`` as its only choice."""
    status = detector_status()
    return ([status], {
        "default": status,
        "tooltip": "Face detector for face crop in this ComfyUI (read by AI Studio; leave as it is). "
                   f"{DETECTOR_MISSING} = install ultralytics in ComfyUI's Python and restart ComfyUI.",
    })


def _comfy_folders() -> list[Path]:
    try:
        import folder_paths  # type: ignore
    except Exception:
        return []
    out: list[Path] = []
    for key in FOLDER_KEYS:
        try:
            out.extend(Path(path) for path in folder_paths.get_folder_paths(key))
        except Exception:
            continue
    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        out.extend(Path(models_dir, *parts) for parts in MODEL_SUBDIRS)
    return out


def model_dirs() -> list[Path]:
    """The folders searched for a bare model name (existing ones, in order, without repeats)."""
    seen: set[str] = set()
    out: list[Path] = []
    for folder in _comfy_folders():
        key = os.path.normcase(str(folder))
        if key in seen or not folder.is_dir():
            continue
        seen.add(key)
        out.append(folder)
    return out


def _is_box_model(path: Path) -> bool:
    # YOLO-World (``*-world*``) needs class names of its own; this box-only feature leaves it out.
    return path.is_file() and path.suffix.lower() == MODEL_SUFFIX and "-world" not in path.stem.lower()


def _scan(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.rglob(f"*{MODEL_SUFFIX}") if _is_box_model(p)), key=lambda p: p.name.lower())
    except OSError:
        return []


def resolve_face_model(name: str) -> Path | None:
    """Model name (or absolute path) -> an existing file. None when not found; nothing is downloaded."""
    value = str(name or "").strip() or DEFAULT_MODEL
    path = Path(value)
    if path.is_absolute():
        return path if _is_box_model(path) else None
    for folder in model_dirs():
        direct = folder / value
        if _is_box_model(direct):
            return direct
        for candidate in _scan(folder):
            if candidate.name.lower() == path.name.lower():
                return candidate
    return None


def missing_model_message(name: str) -> str:
    where = " or ".join(str(d) for d in model_dirs()) or "<ComfyUI>/models/ultralytics/bbox"
    return (
        f"face crop: face detector model {name!r} not found — put it in {where} or pass its absolute path "
        "(the app sends the file of Forge models/adetailer; it is never downloaded automatically)."
    )


def _guard_no_download(path: Path) -> None:
    """ultralytics renames 'yolov5s.pt' to 'yolov5su.pt' and downloads it when missing: stop for such a name."""
    try:
        from ultralytics.utils.checks import check_yolov5u_filename  # type: ignore
    except Exception:
        return
    try:
        renamed = str(check_yolov5u_filename(str(path), verbose=False))
    except TypeError:   # a version without the verbose argument
        renamed = str(check_yolov5u_filename(str(path)))
    if renamed != str(path) and not Path(renamed).exists():
        raise FaceDetectorUnavailable(
            f"face crop: ultralytics would rename {path.name!r} to {Path(renamed).name!r} and download it — "
            "rename the file (e.g. add 'u' as ultralytics expects) or pick another model."
        )


def ultralytics_device(sam3_device: Any) -> str:
    """SAM3 device -> ultralytics ``device``: "cpu" for the CPU, otherwise "" (the current CUDA device, as ADetailer)."""
    return "cpu" if str(sam3_device or "").strip().lower() == "cpu" else ""


def _to_list(value: Any) -> list:
    if value is None:
        return []
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(value, attr):
            value = getattr(value, attr)()
    return list(value.tolist()) if hasattr(value, "tolist") else list(value)


def detect_faces(image, model_name: str, confidence: float, device: Any) -> FaceDetection:
    """Face boxes in ``image`` (PIL RGB). ``FaceDetectorUnavailable`` without the model file or ultralytics."""
    return detect_faces_many([image], model_name, confidence, device)[0]


def _detection(pred: Any, path: Path) -> FaceDetection:
    if not pred:
        return FaceDetection(model_path=str(path))
    boxes = getattr(pred[0], "boxes", None)
    if boxes is None:
        return FaceDetection(model_path=str(path))
    xyxy = [[float(v) for v in row] for row in _to_list(boxes.xyxy)]
    scores = [float(v) for v in _to_list(boxes.conf)]
    if len(scores) != len(xyxy):
        scores = (scores + [float("nan")] * len(xyxy))[: len(xyxy)]
    return FaceDetection(boxes=xyxy, scores=scores, model_path=str(path))


def detect_faces_many(images, model_name: str, confidence: float, device: Any) -> list[FaceDetection]:
    """``detect_faces`` for each picture of an IMAGE batch with one model load (sam-extra reads the model for each
    image it is called with; the same file gives the same boxes)."""
    path = resolve_face_model(model_name)
    if path is None:
        raise FaceDetectorUnavailable(missing_model_message(str(model_name or DEFAULT_MODEL)))
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise FaceDetectorUnavailable(
            "face crop: the 'ultralytics' package is not installed in ComfyUI's Python — install it there "
            "(e.g. python_embeded\\python.exe -m pip install ultralytics) and restart ComfyUI, or generate on Forge."
        ) from exc
    _guard_no_download(path)
    model = YOLO(str(path))
    try:
        preds = [
            model(image, conf=float(confidence), device=ultralytics_device(device), verbose=False)
            for image in images
        ]
    finally:
        del model
    return [_detection(pred, path) for pred in preds]
