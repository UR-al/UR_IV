"""Single-image SAM face mask prototype for sam_extra experiments.

This script is intentionally local-first:
- uses the existing mobile_sam.pt checkpoint in Editor_models
- targets the sample image in sam_extra by default
- saves mask/overlay outputs into sam_extra\outputs

The initial prompt set is tuned for:
00200-20260412_102159_828965.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "sam_extra" / "00200-20260412_102159_828965.png"
DEFAULT_MODEL = ROOT / "Editor_models" / "mobile_sam.pt"
OUTPUT_DIR = ROOT / "sam_extra" / "outputs"


DEFAULT_CONFIG = {
    "box_rel": [0.285, 0.055, 0.655, 0.375],
    "positive_points_rel": [
        [0.455, 0.120],  # hair center
        [0.458, 0.175],  # forehead
        [0.425, 0.210],  # left eye / brow
        [0.500, 0.212],  # right eye / brow
        [0.463, 0.250],  # nose bridge
        [0.462, 0.292],  # mouth / chin center
        [0.448, 0.325],  # jaw left
        [0.492, 0.325],  # jaw right
    ],
    "negative_points_rel": [
        [0.338, 0.250],  # left background
        [0.585, 0.255],  # right background
        [0.455, 0.370],  # neck / collar
        [0.370, 0.342],  # left shirt
        [0.550, 0.345],  # right shirt
    ],
}


def _load_predictor(model_path: Path):
    import torch

    mobile_error = None
    SamPredictor = None
    sam_model_registry = None
    model_type = "vit_t"

    try:
        from mobile_sam import sam_model_registry, SamPredictor
        model_type = "vit_t"
    except ImportError as exc:
        mobile_error = exc

    if SamPredictor is None:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            model_type = "vit_t" if "mobile" in model_path.name.lower() else "vit_b"
        except ImportError as exc:
            raise RuntimeError(
                "Neither mobile_sam nor segment_anything is importable. "
                f"mobile_sam={mobile_error!r}, segment_anything={exc!r}"
            ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(model_path))
    sam.to(device)
    return SamPredictor(sam)


def _rel_to_abs(rel_points: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = []
    for x_rel, y_rel in rel_points:
        pts.append([int(round(x_rel * width)), int(round(y_rel * height))])
    return np.asarray(pts, dtype=np.float32)


def _box_from_rel(box_rel: list[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box_rel
    return np.asarray(
        [
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        ],
        dtype=np.float32,
    )


def _largest_mask_near_center(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(np.uint8)


def _postprocess_mask(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    roi = np.zeros_like(mask, dtype=np.uint8)
    roi[max(y1 - 24, 0):min(y2 + 16, mask.shape[0]), max(x1 - 16, 0):min(x2 + 16, mask.shape[1])] = 1
    cleaned = (mask > 0).astype(np.uint8) * roi
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return cleaned * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def run(image_path: Path, model_path: Path, config: dict):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {model_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = image.shape[:2]
    predictor = _load_predictor(model_path)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    box = _box_from_rel(config["box_rel"], w, h)
    positive = _rel_to_abs(config["positive_points_rel"], w, h)
    negative = _rel_to_abs(config["negative_points_rel"], w, h)
    point_coords = np.vstack([positive, negative]).astype(np.float32)
    point_labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    raw_mask = _largest_mask_near_center(masks, scores)
    final_mask = _postprocess_mask(raw_mask, box)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    mask_path = OUTPUT_DIR / f"{stem}_face_mask.png"
    overlay_path = OUTPUT_DIR / f"{stem}_face_overlay.png"
    prompt_path = OUTPUT_DIR / f"{stem}_face_prompt.json"

    cv2.imwrite(str(mask_path), final_mask)

    overlay = image.copy()
    overlay[final_mask > 0] = (overlay[final_mask > 0] * 0.35 + np.array([30, 210, 255]) * 0.65).astype(np.uint8)
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 220, 80), 2)
    for x, y in positive.astype(int):
        cv2.circle(overlay, (x, y), 6, (0, 220, 0), -1)
    for x, y in negative.astype(int):
        cv2.circle(overlay, (x, y), 6, (0, 0, 220), -1)
    cv2.imwrite(str(overlay_path), overlay)

    prompt_dump = {
        "image": str(image_path),
        "model": str(model_path),
        "box_abs": box.astype(int).tolist(),
        "positive_points_abs": positive.astype(int).tolist(),
        "negative_points_abs": negative.astype(int).tolist(),
        "scores": [float(s) for s in scores],
    }
    prompt_path.write_text(json.dumps(prompt_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "prompt": str(prompt_path),
            "scores": [float(s) for s in scores],
        },
        ensure_ascii=False,
        indent=2,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run(args.image, args.model, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
