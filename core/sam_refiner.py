# core/sam_refiner.py
"""
SAM (Segment Anything Model) 기반 정밀 마스킹
YOLO가 찾은 bbox를 SAM에 전달하여 픽셀 단위 마스크 생성

지원 모델:
- MobileSAM (mobile_sam.pt) — 경량, 빠름
- FastSAM (FastSAM-s.pt) — YOLO 기반, 가벼움
- SAM (sam_vit_b.pt) — 원본, 무거움

editor_models/ 디렉토리에 모델 파일을 넣으면 자동 감지
"""
import os
import numpy as np
import cv2


def find_sam_model(models_dir: str) -> tuple:
    """editor_models/에서 SAM 모델 자동 감지 → (path, type)
    우선순위: SAM3 > MobileSAM > FastSAM > SAM v1
    - SAM3는 .pt만 허용 (.safetensors는 SAM3 코드가 거부함)
    - SAM3는 text-prompted segmentation으로 정확도 최상
    """
    if not os.path.isdir(models_dir):
        return None, None

    priority = [
        ('sam3',        'sam3'),         # 최우선 — Meta SAM 3 (text-prompted)
        ('mobile_sam',  'mobile_sam'),
        ('FastSAM',     'fast_sam'),
        ('sam_vit_b',   'sam'),
        ('sam_vit_l',   'sam'),
        ('sam_vit_h',   'sam'),
    ]

    yolo_keywords = ['yolo', 'nsfw', 'detect', 'censor']

    def _ext_ok(fname: str, sam_type: str) -> bool:
        # SAM3는 .pt만, 나머지는 .pt/.pth/.onnx 허용
        fl = fname.lower()
        if sam_type == 'sam3':
            return fl.endswith('.pt')
        return fl.endswith(('.pt', '.pth', '.onnx'))

    files = sorted(os.listdir(models_dir))

    # 1차: priority 순서대로 매칭 (priority가 outer loop여야 우선순위가 보장됨)
    for keyword, sam_type in priority:
        kw = keyword.lower()
        for fname in files:
            flow = fname.lower()
            if any(yk in flow for yk in yolo_keywords):
                continue
            if kw in flow and _ext_ok(fname, sam_type):
                return os.path.join(models_dir, fname), sam_type

    # 2차: 'sam'이 포함된 파일 폴백
    for fname in sorted(os.listdir(models_dir)):
        flow = fname.lower()
        if not flow.endswith(('.pt', '.pth', '.onnx')):
            continue
        if any(yk in flow for yk in yolo_keywords):
            continue
        if 'sam' in flow:
            if 'sam3' in flow and flow.endswith('.pt'):
                sam_type = 'sam3'
            elif 'mobile' in flow:
                sam_type = 'mobile_sam'
            else:
                sam_type = 'sam'
            return os.path.join(models_dir, fname), sam_type

    return None, None


def _build_sam3_prompt(yolo_model_paths: list) -> str:
    """YOLO 모델 파일명에서 SAM3용 텍스트 프롬프트 추출.
    예: ['penis.pt', 'pussy.pt'] → 'penis, pussy'
    """
    if not yolo_model_paths:
        return 'person'
    tokens = []
    for p in yolo_model_paths:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        # 일반적인 비-의미 토큰 제거
        for junk in ('yolo', 'nsfw', 'detect', 'censor', '_v8', '_v9', '_v10', 'best', 'final'):
            stem = stem.replace(junk, '')
        # 구분자 정리
        stem = stem.replace('_', ' ').replace('-', ' ').strip()
        if stem and stem not in tokens:
            tokens.append(stem)
    return ', '.join(tokens) if tokens else 'person'


def refine_boxes_with_sam(image: np.ndarray, boxes: list, models_dir: str,
                          sam_model_path: str = None, sam_type: str = None,
                          text_prompt: str = None,
                          yolo_model_paths: list = None) -> np.ndarray:
    """
    YOLO bbox 목록을 SAM으로 정밀 마스킹

    Args:
        image: BGR numpy 이미지
        boxes: [(x1, y1, x2, y2), ...] YOLO 검출 박스
        models_dir: editor_models/ 경로
        sam_model_path: SAM 모델 파일 경로 (None이면 자동 감지)
        sam_type: 'sam3', 'mobile_sam', 'fast_sam', 'sam'
        text_prompt: SAM3 전용. None이면 yolo_model_paths에서 자동 생성
        yolo_model_paths: SAM3 텍스트 프롬프트 자동 생성에 사용

    Returns:
        combined_mask: uint8 마스크 (0 or 255)
    """
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # SAM 모델 찾기
    if sam_model_path is None:
        sam_model_path, sam_type = find_sam_model(models_dir)

    if sam_model_path is None or not os.path.exists(sam_model_path):
        print(f"[SAM] No SAM model found in {models_dir}")
        print(f"[SAM] Files: {os.listdir(models_dir) if os.path.isdir(models_dir) else 'dir not found'}")
        print("[SAM] Falling back to bbox mask")
        for (x1, y1, x2, y2) in boxes:
            combined_mask[y1:y2, x1:x2] = 255
        return combined_mask

    print(f"[SAM] Found model: {sam_type} → {sam_model_path}")
    if sam_type == 'sam3':
        prompt = text_prompt or _build_sam3_prompt(yolo_model_paths or [])
        print(f"[SAM3] Text prompt: '{prompt}', YOLO boxes: {len(boxes)}")
    else:
        print(f"[SAM] Processing {len(boxes)} boxes...")

    if sam_type != 'sam3' and not boxes:
        # SAM3 외에는 bbox 없으면 빈 마스크
        return combined_mask

    try:
        if sam_type == 'sam3':
            return _refine_with_sam3(
                image, boxes, sam_model_path, combined_mask,
                text_prompt=(text_prompt or _build_sam3_prompt(yolo_model_paths or [])),
            )
        if sam_type == 'fast_sam':
            return _refine_with_fastsam(image, boxes, sam_model_path, combined_mask)
        return _refine_with_sam(image, boxes, sam_model_path, sam_type, combined_mask)
    except Exception as e:
        import traceback
        print(f"[SAM] Error: {e}")
        traceback.print_exc()
        for (x1, y1, x2, y2) in boxes:
            combined_mask[y1:y2, x1:x2] = 255
        return combined_mask


def _refine_with_sam(image: np.ndarray, boxes: list, model_path: str,
                     sam_type: str, mask: np.ndarray) -> np.ndarray:
    """MobileSAM / SAM으로 정밀 마스킹"""
    import torch

    # SAM 라이브러리 로드 (여러 패키지 시도)
    SamPredictor = None
    sam_model_registry = None
    model_type = 'vit_b'

    _mobile_err = None
    if sam_type == 'mobile_sam':
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            model_type = 'vit_t'
            print("[SAM] Using mobile_sam package (vit_t)")
        except ImportError as ie:
            _mobile_err = ie
            print(f"[SAM] mobile_sam import failed: {ie}")

    if SamPredictor is None:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            if 'vit_h' in model_path.lower(): model_type = 'vit_h'
            elif 'vit_l' in model_path.lower(): model_type = 'vit_l'
            elif 'vit_t' in model_path.lower() or 'mobile' in model_path.lower(): model_type = 'vit_t'
            else: model_type = 'vit_b'
            print(f"[SAM] Using segment_anything package ({model_type})")
        except ImportError as ie2:
            print(f"[SAM] Neither mobile_sam nor segment_anything installed")
            print(f"[SAM] mobile_sam error: {_mobile_err}")
            print(f"[SAM] segment_anything error: {ie2}")
            for (x1, y1, x2, y2) in boxes:
                mask[y1:y2, x1:x2] = 255
            return mask

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    # RGB로 변환 후 이미지 설정
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    for (x1, y1, x2, y2) in boxes:
        box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
        # 가장 높은 스코어 마스크 선택
        best_idx = np.argmax(scores)
        seg_mask = masks[best_idx].astype(np.uint8) * 255
        mask = np.maximum(mask, seg_mask)

    return mask


def _refine_with_fastsam(image: np.ndarray, boxes: list,
                         model_path: str, mask: np.ndarray) -> np.ndarray:
    """FastSAM으로 정밀 마스킹"""
    from ultralytics import YOLO as FastSAMModel
    h, w = image.shape[:2]

    model = FastSAMModel(model_path)
    results = model(image, retina_masks=True, conf=0.4, iou=0.7)

    if not results or results[0].masks is None:
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    all_masks = results[0].masks.data.cpu().numpy()

    for (bx1, by1, bx2, by2) in boxes:
        best_iou = 0
        best_mask = None

        for seg in all_masks:
            seg_resized = cv2.resize(seg.astype(np.float32), (w, h))
            # bbox 영역과의 IoU 계산
            box_mask = np.zeros((h, w), dtype=np.float32)
            box_mask[by1:by2, bx1:bx2] = 1.0
            intersection = (seg_resized > 0.5) & (box_mask > 0.5)
            union = (seg_resized > 0.5) | (box_mask > 0.5)
            iou = intersection.sum() / max(union.sum(), 1)

            if iou > best_iou:
                best_iou = iou
                best_mask = seg_resized

        if best_mask is not None and best_iou > 0.1:
            mask[best_mask > 0.5] = 255
        else:
            # fallback: bbox만
            mask[by1:by2, bx1:bx2] = 255

    return mask


# ── SAM3 (Meta Segment Anything 3) ────────────────────────────────────
# 텍스트 프롬프트 기반 segmentation. bbox prompt가 아닌 자연어로 동작.
# Forge SAM3 확장과 동일한 sam3 패키지를 사용.
# 필요: pip install sam3 timm einops huggingface_hub iopath

_SAM3_BUNDLE_CACHE = {}  # checkpoint_path → (model, processor, device)
_SAM3_BPE_PATH = None


def _find_sam3_bpe_vocab() -> str:
    """SAM3가 사용하는 BPE vocab 파일 경로 찾기.
    1) Forge SAM3 확장 캐시 사용
    2) huggingface_hub로 다운로드
    """
    global _SAM3_BPE_PATH
    if _SAM3_BPE_PATH and os.path.exists(_SAM3_BPE_PATH):
        return _SAM3_BPE_PATH

    # Forge 확장 자산에서 우선 찾기
    forge_candidate = os.path.join(
        'C:\\sd-webui-forge-neo', 'extensions', 'forge_sam3_extension',
        'assets', 'bpe_simple_vocab_16e6.txt.gz',
    )
    if os.path.exists(forge_candidate):
        _SAM3_BPE_PATH = forge_candidate
        return forge_candidate

    # 프로젝트 캐시 디렉토리에 다운로드
    try:
        from huggingface_hub import hf_hub_download
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'image_cache', 'sam3_assets',
        )
        os.makedirs(cache_dir, exist_ok=True)
        path = hf_hub_download(
            repo_id='facebook/sam3',
            filename='bpe_simple_vocab_16e6.txt.gz',
            cache_dir=cache_dir,
        )
        _SAM3_BPE_PATH = path
        return path
    except Exception as e:
        print(f"[SAM3] BPE vocab download failed: {e}")
        return ''


def _refine_with_sam3(image: np.ndarray, boxes: list, model_path: str,
                      mask: np.ndarray, text_prompt: str = 'person') -> np.ndarray:
    """SAM3로 정밀 마스킹.
    SAM3는 text-prompted (bbox 입력 불가). YOLO bbox는 출력 마스크 필터링에만 사용.
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as ie:
        print(f"[SAM3] sam3 패키지 미설치 — bbox 폴백.")
        print(f"[SAM3]   설치: pip install sam3 timm einops huggingface_hub iopath")
        print(f"[SAM3]   ImportError: {ie}")
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    import torch

    bpe_path = _find_sam3_bpe_vocab()
    if not bpe_path:
        print("[SAM3] BPE vocab을 찾을 수 없음 — bbox 폴백")
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_key = f"{model_path}|{device}"

    if cache_key in _SAM3_BUNDLE_CACHE:
        sam3_model, processor = _SAM3_BUNDLE_CACHE[cache_key]
    else:
        print(f"[SAM3] Loading model: {os.path.basename(model_path)} on {device}...")
        sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=model_path,
            load_from_HF=False,
        )
        processor = Sam3Processor(sam3_model, device=device)
        _SAM3_BUNDLE_CACHE[cache_key] = (sam3_model, processor)

    processor.set_confidence_threshold(0.5)

    # PIL로 변환
    from PIL import Image as PILImage
    pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if device == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            state = processor.set_image(pil)
            state = processor.set_text_prompt(prompt=text_prompt, state=state)
    else:
        state = processor.set_image(pil)
        state = processor.set_text_prompt(prompt=text_prompt, state=state)

    masks_arr = state.get('masks', None)
    if masks_arr is None:
        print("[SAM3] no masks returned")
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    # tensor → numpy
    if hasattr(masks_arr, 'detach'):
        masks_arr = masks_arr.detach().float().cpu().numpy()
    masks_arr = np.asarray(masks_arr)
    if masks_arr.size == 0:
        print(f"[SAM3] no detections for prompt '{text_prompt}' — bbox 폴백")
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    # (N,H,W) 정규화
    if masks_arr.ndim == 2:
        masks_arr = masks_arr[None, ...]
    elif masks_arr.ndim > 3:
        masks_arr = masks_arr.reshape((-1, masks_arr.shape[-2], masks_arr.shape[-1]))

    h, w = image.shape[:2]
    sam3_masks = []
    for m in masks_arr:
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        sam3_masks.append((m > 0.5).astype(bool))

    if not sam3_masks:
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    # YOLO bbox가 있으면 IoU로 필터링 — SAM3 마스크 중 YOLO 영역과 겹치는 것만 채택
    if boxes:
        kept = 0
        for (bx1, by1, bx2, by2) in boxes:
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            best_iou, best_m = 0.0, None
            for m_bool in sam3_masks:
                inter = m_bool[by1:by2, bx1:bx2].sum()
                m_area = m_bool.sum()
                union = m_area + box_area - inter
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou = iou
                    best_m = m_bool
            if best_m is not None and best_iou > 0.05:
                # bbox 주변으로 마스크 제한 (Forge의 _restrict_mask_to_box와 유사)
                pad = max(8, int(round(max(bx2 - bx1, by2 - by1) * 0.08)))
                px1 = max(0, bx1 - pad); py1 = max(0, by1 - pad)
                px2 = min(w, bx2 + pad); py2 = min(h, by2 + pad)
                clipped = np.zeros_like(best_m, dtype=bool)
                clipped[py1:py2, px1:px2] = best_m[py1:py2, px1:px2]
                mask[clipped] = 255
                kept += 1
            else:
                # 매칭 실패 시 bbox만 사용
                mask[by1:by2, bx1:bx2] = 255
        print(f"[SAM3] kept {kept}/{len(boxes)} masks (IoU≥0.05)")
    else:
        # bbox 없으면 모든 SAM3 마스크 OR
        union = np.zeros((h, w), dtype=bool)
        for m_bool in sam3_masks:
            union |= m_bool
        mask[union] = 255
        print(f"[SAM3] no YOLO bbox — used all {len(sam3_masks)} SAM3 masks")

    return mask
