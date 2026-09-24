# core/sam_refiner.py
"""
SAM (Segment Anything Model) 기반 정밀 마스킹
YOLO가 찾은 bbox를 SAM에 전달하여 픽셀 단위 마스크 생성

지원 모델:
- MobileSAM (mobile_sam.pt) — 경량, 빠름
- FastSAM (FastSAM-s.pt) — YOLO 기반, 가벼움
- SAM (sam_vit_b.pt) — 원본, 무거움

editor_models/ 디렉토리에 모델 파일을 넣으면 자동 감지

정직한 폴백 계약
- 모델 파일이 있어도 실행 패키지가 없으면(예: mobile_sam.pt만 있고 mobile_sam 미설치)
  정밀화를 '한 척' 하지 않는다. resolve_sam_model이 사용 불가를 알려 주고, 세션당 1회
  경고 토스트를 띄운 뒤 호출자의 YOLO 마스크를 그대로 쓴다.
- auto는 패키지가 없다고 SAM3(3.45GB VRAM)로 몰래 승격하지 않는다 (b83190abe 결정 유지).
"""
import importlib.util
import os
import re
import threading
from typing import NamedTuple, Optional

import numpy as np
import cv2

# 진단 로그는 절대 예외를 내지 않는다 — stdout이 파이프·파일이면 cp949라 print()가
# '—' 같은 문자에서 UnicodeEncodeError를 내고, except 블록 안이면 알림·VRAM 반납을 건너뛴다.
from core.safe_print import safe_print as _log


_AUTO_PRIORITY = [
    # Default 'auto' priority — MobileSAM first (가볍고 이전부터 안정 동작)
    # SAM3가 항상 더 좋은 마스크를 주지는 않으므로 사용자가 명시 선택하지 않으면 보수적
    ('mobile_sam', 'mobile_sam'),
    ('sam3',       'sam3'),
    ('FastSAM',    'fast_sam'),
    ('sam_vit_b',  'sam'),
    ('sam_vit_l',  'sam'),
    ('sam_vit_h',  'sam'),
]

# 사용자가 명시 지정 시 해당 타입의 파일만 검색
_TYPE_KEYWORDS = {
    'sam3':       [('sam3',       'sam3')],
    'mobile_sam': [('mobile_sam', 'mobile_sam'), ('mobile',     'mobile_sam')],
    'fast_sam':   [('FastSAM',    'fast_sam'),   ('fastsam',    'fast_sam')],
    'sam':        [('sam_vit_b',  'sam'),        ('sam_vit_l',  'sam'),       ('sam_vit_h', 'sam')],
}

_YOLO_KEYWORDS = ('yolo', 'nsfw', 'detect', 'censor')

# 모델 유형 → 실행에 필요한 파이썬 패키지.
# segment_anything에는 MobileSAM의 vit_t가 없어 mobile_sam을 대신할 수 없다.
_RUNTIME_PACKAGES = {
    'mobile_sam': 'mobile_sam',
    'sam':        'segment_anything',
    'fast_sam':   'ultralytics',
    'sam3':       'sam3',
}
_TYPE_LABELS = {'mobile_sam': 'MobileSAM', 'sam': 'SAM', 'fast_sam': 'FastSAM', 'sam3': 'SAM3'}


class SamResolution(NamedTuple):
    """모델 선택 결과. 파일은 있지만 실행할 수 없으면 path/sam_type이 None이고
    missing_type/missing_package가 이유를 담는다."""
    path: Optional[str]
    sam_type: Optional[str]
    missing_type: Optional[str] = None
    missing_package: Optional[str] = None


class SamUnavailableError(ImportError):
    """SAM 실행 패키지·자산을 불러올 수 없음 — 정밀화 없이 호출자의 마스크를 써야 한다."""

    def __init__(self, sam_type: str, package: str, detail: str = ''):
        self.sam_type = sam_type
        self.package = package
        self.detail = str(detail or '')
        super().__init__(sam_unavailable_message(sam_type, package))


def missing_sam_runtime(sam_type: str) -> Optional[str]:
    """sam_type 실행 패키지가 없으면 그 패키지 이름, 있으면 None. import는 하지 않는다."""
    package = _RUNTIME_PACKAGES.get(sam_type)
    if not package:
        return None
    try:
        found = importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        found = False
    return None if found else package


def sam_unavailable_message(sam_type: str, package: str) -> str:
    label = _TYPE_LABELS.get(sam_type, str(sam_type))
    return (f"{label} 실행에 필요한 '{package}'을(를) 불러올 수 없어 "
            f"SAM 정밀화 없이 YOLO 마스크(bbox)를 사용합니다.")


_UNAVAILABLE_NOTICES = set()
_UNAVAILABLE_LOCK = threading.Lock()


def notify_sam_unavailable(notify, sam_type: str, package: str, detail: str = '') -> bool:
    """정밀화 불가를 알린다 — 사용자 토스트는 (유형, 패키지)별로 세션당 1회.

    로그는 매번 한 줄만 남긴다(예전에는 클릭마다 4줄 + '✓ Refined' 거짓 로그).
    반환: 이번 호출에서 토스트를 띄웠는지.
    """
    message = sam_unavailable_message(sam_type, package)
    _log(f"[SAM] {message}" + (f" ({detail})" if detail else ''))
    if notify is None:
        return False
    key = (sam_type, package)
    with _UNAVAILABLE_LOCK:
        if key in _UNAVAILABLE_NOTICES:
            return False
        _UNAVAILABLE_NOTICES.add(key)
    try:
        notify('warning', message)
    except Exception:
        pass
    return True


def _reset_unavailable_notices():
    """테스트용 — 세션당 1회 기록을 비운다."""
    with _UNAVAILABLE_LOCK:
        _UNAVAILABLE_NOTICES.clear()


def resolve_sam_model(models_dir: str, prefer_type: str = 'auto') -> SamResolution:
    """editor_models/에서 SAM 모델을 고르고 실행 가능 여부까지 판정한다.

    가장 우선인 **파일**을 고른 뒤 그 유형의 실행 패키지를 확인한다. 패키지가 없으면
    다음 우선순위(특히 SAM3 3.45GB)로 넘어가지 않고 '사용 불가'를 돌려준다 —
    사용자가 둔 모델 대신 다른 무거운 모델을 몰래 쓰지 않기 위해서다.
    """
    path, sam_type = _find_sam_file(models_dir, prefer_type)
    if path is None:
        return SamResolution(None, None)
    package = missing_sam_runtime(sam_type)
    if package:
        return SamResolution(None, None, sam_type, package)
    return SamResolution(path, sam_type)


def find_sam_model(models_dir: str, prefer_type: str = 'auto') -> tuple:
    """editor_models/에서 **실행 가능한** SAM 모델 감지 → (path, type)

    prefer_type:
      - 'auto'        : MobileSAM > SAM3 > FastSAM > SAM v1 우선순위로 파일 선택
      - 'sam3'        : SAM3 (.pt) 만 검색 — 텍스트 프롬프트 기반
      - 'mobile_sam'  : MobileSAM (.pt) 만 검색 — bbox 프롬프트 기반
      - 'fast_sam'    : FastSAM 만 검색
      - 'sam'         : SAM v1 만 검색
      - 'off'         : (None, None) 반환 — SAM 정밀화 건너뜀

    고른 파일의 실행 패키지가 없으면 (None, None) — 이유가 필요하면 resolve_sam_model.
    SAM3는 .pt만 허용 (.safetensors는 sam3 패키지가 거부).
    """
    resolution = resolve_sam_model(models_dir, prefer_type)
    return resolution.path, resolution.sam_type


def _find_sam_file(models_dir: str, prefer_type: str = 'auto') -> tuple:
    """파일 이름만으로 우선순위 최상위 SAM 모델 → (path, type). 실행 가능 여부는 보지 않는다."""
    if prefer_type == 'off':
        return None, None
    if not os.path.isdir(models_dir):
        return None, None

    if prefer_type == 'auto':
        priority = _AUTO_PRIORITY
    else:
        priority = _TYPE_KEYWORDS.get(prefer_type, _AUTO_PRIORITY)

    def _ext_ok(fname: str, sam_type: str) -> bool:
        fl = fname.lower()
        if sam_type == 'sam3':
            return fl.endswith('.pt')
        return fl.endswith(('.pt', '.pth', '.onnx'))

    files = sorted(os.listdir(models_dir))

    # priority가 outer loop여야 우선순위 보장
    for keyword, sam_type in priority:
        kw = keyword.lower()
        for fname in files:
            flow = fname.lower()
            if any(yk in flow for yk in _YOLO_KEYWORDS):
                continue
            if kw in flow and _ext_ok(fname, sam_type):
                return os.path.join(models_dir, fname), sam_type

    # 폴백 (auto 전용): 'sam'이 포함된 파일
    if prefer_type == 'auto':
        for fname in files:
            flow = fname.lower()
            if not flow.endswith(('.pt', '.pth', '.onnx')):
                continue
            if any(yk in flow for yk in _YOLO_KEYWORDS):
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
                          yolo_model_paths: list = None,
                          exclude_prompt: str = None,
                          notify=None) -> np.ndarray:
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
        exclude_prompt: SAM3 전용. 마스크에서 제외할 영역을 텍스트로 지정
            예: 'face' → 얼굴 영역을 검출해서 최종 마스크에서 빼기
        notify: 사용자 알림 콜백 — notify(level: 'info'|'warning'|'error', message: str)
            SAM3 exclude 안전장치 발동, 실행 패키지 없음(세션당 1회), 정밀화 실패 등

    Returns:
        uint8 마스크 (0 or 255). SAM이 아예 돌지 못했으면(모델·패키지·자산 없음, 실행 오류)
        **빈 마스크**를 돌려준다 — 호출자가 가진 YOLO 마스크(bbox/seg)를 그대로 쓰게 한다.
        예전에는 이때 bbox를 채워 정상값처럼 돌려줘 'Refined' 성공으로 보였다.
    """
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # SAM 모델 찾기
    if sam_model_path is None:
        resolution = resolve_sam_model(models_dir)
        if resolution.missing_package:
            notify_sam_unavailable(notify, resolution.missing_type, resolution.missing_package)
            return combined_mask
        sam_model_path, sam_type = resolution.path, resolution.sam_type

    if sam_model_path is None or not os.path.exists(sam_model_path):
        _log(f"[SAM] No SAM model found in {models_dir} - SAM 정밀화 없이 호출자 마스크 사용")
        return combined_mask

    _log(f"[SAM] Found model: {sam_type} → {sam_model_path}")
    if sam_type == 'sam3':
        prompt = text_prompt or _build_sam3_prompt(yolo_model_paths or [])
        _log(f"[SAM3] Text prompt: '{prompt}', YOLO boxes: {len(boxes)}")
    else:
        _log(f"[SAM] Processing {len(boxes)} boxes...")

    if sam_type != 'sam3' and not boxes:
        # SAM3 외에는 bbox 없으면 빈 마스크
        return combined_mask

    result = None
    release_after = False
    try:
        if sam_type == 'sam3':
            # 예전에는 여기서 try/finally로 매번 _unload_sam3_bundle()을 불러
            # 바로 위의 _SAM3_BUNDLE_CACHE를 무력화했다 → 클릭마다 3.45GB 재로딩(수십 초).
            # 이제는 유휴 캐시(core/model_cache.SAM3_CACHE)가 관리한다: 연속 작업은
            # 캐시 히트, 손을 떼면 90초 뒤 reaper 스레드가, 생성 시작 직전에는
            # resource_coordinator가 VRAM을 반납한다(추론 중이면 끝나는 즉시 — lease).
            result = _refine_with_sam3(
                image, boxes, sam_model_path, combined_mask,
                text_prompt=(text_prompt or _build_sam3_prompt(yolo_model_paths or [])),
                exclude_prompt=exclude_prompt,
                notify=notify,
            )
        elif sam_type == 'fast_sam':
            result = _refine_with_fastsam(image, boxes, sam_model_path, combined_mask)
            release_after = True
        else:
            result = _refine_with_sam(image, boxes, sam_model_path, sam_type, combined_mask)
            release_after = True
    except SamUnavailableError as unavailable:
        # 모델을 올리기 전에 멈췄다 — 돌려줄 VRAM이 없다(gc/empty_cache 비용 없음)
        notify_sam_unavailable(notify, unavailable.sam_type, unavailable.package, unavailable.detail)
        result = None
    except Exception as e:
        # 정리 표시를 먼저 한다 — 아래 로그·알림이 어떻게 되든 VRAM 반납은 한다.
        # 실패한 호출은 유형과 무관하게 반납한다: MobileSAM/FastSAM은 올리다 만 모델,
        # SAM3는 OOM 등으로 남은 활성값과, lease가 끝날 때 반납 예약돼 있었지만 예외
        # 프레임이 번들을 쥐고 있어 그때는 empty_cache가 돌려주지 못한 블록.
        result = None
        release_after = True
        import traceback
        _log("[SAM] Error:", e, "- SAM 정밀화 없이 호출자 마스크 사용")
        traceback.print_exc()
        if notify:
            try:
                notify('error', f'SAM 정밀화 실패: {e}')
            except Exception:
                pass
    if release_after:
        # MobileSAM/SAM v1/FastSAM은 호출마다 새로 올린다. 여기서는 그 함수의 지역 참조와
        # 예외 프레임(except 블록이 끝나며 해제)이 모두 끝났으므로 CUDA 캐시를 드라이버로
        # 돌려줄 수 있다. 성공한 SAM3 호출은 유휴 캐시가 관리하므로 여기서 부르지 않는다.
        from core.model_cache import release_torch_memory
        release_torch_memory()
    return result if result is not None else np.zeros((h, w), dtype=np.uint8)


def _import_sam_runtime(sam_type: str, model_path: str):
    """(sam_model_registry, SamPredictor, model_type). 패키지가 없으면 SamUnavailableError.

    MobileSAM(vit_t)은 mobile_sam 패키지만 실행할 수 있다 — segment_anything 레지스트리에는
    vit_t가 없어 예전의 '대체 시도'는 KeyError로 끝났다.
    """
    if sam_type == 'mobile_sam':
        try:
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError as ie:
            raise SamUnavailableError('mobile_sam', 'mobile_sam', str(ie)) from ie
        return sam_model_registry, SamPredictor, 'vit_t'
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as ie:
        raise SamUnavailableError('sam', 'segment_anything', str(ie)) from ie
    lowered = model_path.lower()
    model_type = 'vit_h' if 'vit_h' in lowered else 'vit_l' if 'vit_l' in lowered else 'vit_b'
    return sam_model_registry, SamPredictor, model_type


def _refine_with_sam(image: np.ndarray, boxes: list, model_path: str,
                     sam_type: str, mask: np.ndarray) -> np.ndarray:
    """MobileSAM / SAM으로 정밀 마스킹"""
    sam_model_registry, SamPredictor, model_type = _import_sam_runtime(sam_type, model_path)
    _log(f"[SAM] Using {'mobile_sam' if sam_type == 'mobile_sam' else 'segment_anything'} ({model_type})")
    import torch

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
    try:
        from ultralytics import YOLO as FastSAMModel
    except ImportError as ie:
        raise SamUnavailableError('fast_sam', 'ultralytics', str(ie)) from ie
    h, w = image.shape[:2]

    model = FastSAMModel(model_path)
    results = model(image, retina_masks=True, conf=0.4, iou=0.7)

    if not results or results[0].masks is None:
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    segments = _segment_masks(results[0].masks.data.cpu().numpy(), w, h)
    for (bx1, by1, bx2, by2) in boxes:
        best_mask, best_iou = _best_box_match(segments, (bx1, by1, bx2, by2))
        if best_mask is not None and best_iou > 0.1:
            mask[best_mask] = 255
        else:
            # fallback: bbox만
            mask[by1:by2, bx1:bx2] = 255

    return mask


def _segment_masks(raw_masks, w: int, h: int) -> list:
    """세그먼트 마스크를 원본 해상도 bool로 **한 번만** 만들고 면적을 함께 계산한다.

    예전 FastSAM 루프는 박스 × 마스크마다 전체 해상도 resize와 zeros를 새로 만들었다.
    반환: [(bool_mask, area), ...]
    """
    segments = []
    for seg in raw_masks:
        seg = np.asarray(seg, dtype=np.float32)
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h))
        seg_bool = seg > 0.5
        segments.append((seg_bool, int(seg_bool.sum())))
    return segments


def _best_box_match(segments: list, box: tuple):
    """bbox와 IoU가 가장 큰 세그먼트 → (bool_mask | None, iou).

    교집합은 bbox 슬라이스만 세고, 합집합은 면적 공식(|A|+|B|-|A∩B|)으로 구한다.
    전체 해상도 bbox 마스크를 만들던 예전 계산과 값이 같다.
    """
    bx1, by1, bx2, by2 = box
    best_iou, best_mask = 0.0, None
    for seg_bool, area in segments:
        region = seg_bool[by1:by2, bx1:bx2]
        inter = int(region.sum())
        union = area + int(region.size) - inter
        iou = inter / max(union, 1)
        if iou > best_iou:
            best_iou, best_mask = iou, seg_bool
    return best_mask, best_iou


# ── SAM3 (Meta Segment Anything 3) ────────────────────────────────────
# 텍스트 프롬프트 기반 segmentation. bbox prompt가 아닌 자연어로 동작.
# Forge SAM3 확장과 동일한 sam3 패키지를 사용.
# 필요: pip install sam3 triton-windows  (timm/einops/iopath/huggingface_hub 는 sam3 가 끌어온다.
#   triton 은 sam3 메타데이터에 없지만 sam3/model/edt.py 가 무조건 import 한다 — requirements.txt 참고)

#
# SAM3 번들(~3.5GB) 수명: core/model_cache.SAM3_CACHE가 관리한다.
#   - 연속 클릭은 캐시 히트
#   - 90초 유휴 → reaper 스레드가 반납 (IdleModelCache auto_sweep)
#   - 생성 시작 직전 → core.resource_coordinator가 release_for_generation()으로 반납
#     (Forge 후처리 워커 SAM3·ADetailer·Refine·업스케일도 작업 직전에 같은 함수를 부른다)
#   - VRAM 게이지 수동 언로드 → model_cache.clear_all()
#   - 추론 중(SAM3_CACHE.lease)에 위 반납 요청이 오면 그 추론이 끝나는 즉시 반납

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
        _log(f"[SAM3] BPE vocab download failed: {e}")
        return ''


def _refine_with_sam3(image: np.ndarray, boxes: list, model_path: str,
                      mask: np.ndarray, text_prompt: str = 'person',
                      exclude_prompt: str = None, notify=None) -> np.ndarray:
    """SAM3로 정밀 마스킹.
    SAM3는 text-prompted (bbox 입력 불가). YOLO bbox는 출력 마스크 필터링에만 사용.
    exclude_prompt가 주어지면 해당 영역(예: 'face')을 SAM3로 찾아서 최종 마스크에서 뺀다.
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as ie:
        # sam3 패키지는 있어도 의존성(triton 등)이 빠지면 여기서 실패한다.
        _log("[SAM3]   설치: pip install sam3 timm einops huggingface_hub iopath triton-windows")
        raise SamUnavailableError('sam3', 'sam3', str(ie)) from ie

    import torch

    bpe_path = _find_sam3_bpe_vocab()
    if not bpe_path:
        raise SamUnavailableError('sam3', 'bpe_simple_vocab_16e6.txt.gz',
                                  'Forge 확장 자산과 huggingface_hub 다운로드 모두 실패')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_key = f"{model_path}|{device}"

    def _load_bundle(_key):
        _log(f"[SAM3] Loading model: {os.path.basename(model_path)} on {device}...")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=model_path,
            load_from_HF=False,
        )
        return (model, Sam3Processor(model, device=device))

    # 유휴 캐시 — 연속 검출은 히트, 손 떼고 90초 지나면 VRAM 반납.
    # lease 동안에는 reaper·생성 시작(release_for_generation)·용량 초과가 추론 중인 번들을
    # 빼 가지 않고 '반납 예약'만 한다 → lease가 끝날 때 해제(on_evict + empty_cache).
    # 번들 참조는 _sam3_segment 프레임에만 둔다: lease가 끝나는 시점에 이 함수가 모델을
    # 쥐고 있으면 empty_cache가 블록을 돌려주지 못한다(3.4GB가 reserved로 남는다).
    from core.model_cache import SAM3_CACHE
    with SAM3_CACHE.lease(cache_key, _load_bundle) as bundle:
        try:
            return _sam3_segment(bundle, device, image, boxes, mask, text_prompt,
                                 exclude_prompt, notify)
        finally:
            bundle = None


def _sam3_segment(bundle, device: str, image: np.ndarray, boxes: list, mask: np.ndarray,
                  text_prompt: str, exclude_prompt: str = None, notify=None) -> np.ndarray:
    """빌린 SAM3 번들로 main 패스 + exclude 패스. 번들 참조는 이 프레임이 끝나면 사라진다."""
    import torch

    _sam3_model, processor = bundle
    bundle = None

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
        _log("[SAM3] no masks returned")
        for (x1, y1, x2, y2) in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    # tensor → numpy
    if hasattr(masks_arr, 'detach'):
        masks_arr = masks_arr.detach().float().cpu().numpy()
    masks_arr = np.asarray(masks_arr)
    if masks_arr.size == 0:
        _log(f"[SAM3] no detections for prompt '{text_prompt}' — bbox 폴백")
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
        # 마스크 면적은 박스와 무관 — 박스마다 전체 해상도로 다시 세지 않는다
        sam3_areas = [int(m_bool.sum()) for m_bool in sam3_masks]
        for (bx1, by1, bx2, by2) in boxes:
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            best_iou, best_m = 0.0, None
            for m_bool, m_area in zip(sam3_masks, sam3_areas):
                inter = m_bool[by1:by2, bx1:bx2].sum()
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
        _log(f"[SAM3] kept {kept}/{len(boxes)} masks (IoU≥0.05)")
    else:
        # bbox 없으면 모든 SAM3 마스크 OR
        union = np.zeros((h, w), dtype=bool)
        for m_bool in sam3_masks:
            union |= m_bool
        mask[union] = 255
        _log(f"[SAM3] no YOLO bbox — used all {len(sam3_masks)} SAM3 masks")

    # ── 제외 프롬프트 처리 ──
    # 사용자가 'face' 같은 텍스트로 마스크에서 빼고 싶은 영역을 지정한 경우
    # SAM3를 한 번 더 돌려 해당 영역 검출 → mask에서 0으로 빼기
    #
    # Forge SAM3와 동일하게 콤마/세미콜론/파이프로 토큰 분할 → 각각 별도 패스
    # (SAM3 set_text_prompt는 단일 개념에 최적화되어 있어 다중 토큰은 결과가 불안)
    #
    # 안전장치: 제외가 main mask의 80% 이상을 지워버리면 의도와 다를 가능성이
    # 매우 높으므로 적용을 취소 (mask 보존). 그렇지 않으면 vue_bridge에서
    # sam_mask.any()가 False가 되어 YOLO bbox로 폴백되어버림 — 사용자가
    # "exclude 넣으니까 bbox만 나옴" 증상을 보게 됨.
    if exclude_prompt and exclude_prompt.strip():
        excl_text = exclude_prompt.strip()
        excl_tokens = [t.strip() for t in re.split(r'[,;|\n]', excl_text) if t.strip()]
        if not excl_tokens:
            excl_tokens = [excl_text]
        _log(f"[SAM3] exclude tokens ({len(excl_tokens)}): {excl_tokens}")

        excl_union = np.zeros((h, w), dtype=bool)
        total_n = 0

        # 최적화: 이미지 인코딩(ViT)은 main pass의 set_image 결과를 그대로 재사용한다.
        # set_text_prompt는 backbone_out의 텍스트 키만 update하고 _forward_grounding이
        # masks/boxes/scores를 덮어쓸 뿐이라, 같은 이미지를 다시 인코딩할 이유가 없다.
        # main 마스크는 위에서 이미 numpy로 복사해 두었다(masks_arr).
        # 입력은 해상도와 무관하게 1008²로 리사이즈되므로 절감은 클릭당 ViT forward 1회.
        _excl_base_state = state

        for tok in excl_tokens:
            try:
                # base_state 공유 — set_text_prompt가 text_outputs만 in-place 갱신
                if device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        excl_state = processor.set_text_prompt(prompt=tok, state=_excl_base_state)
                else:
                    excl_state = processor.set_text_prompt(prompt=tok, state=_excl_base_state)

                excl_arr = excl_state.get('masks', None) if isinstance(excl_state, dict) else None
                if excl_arr is None:
                    _log(f"[SAM3]   exclude '{tok}': no masks returned")
                    continue
                if hasattr(excl_arr, 'detach'):
                    excl_arr = excl_arr.detach().float().cpu().numpy()
                excl_arr = np.asarray(excl_arr)
                if excl_arr.size == 0:
                    _log(f"[SAM3]   exclude '{tok}': no detections")
                    continue
                if excl_arr.ndim == 2:
                    excl_arr = excl_arr[None, ...]
                elif excl_arr.ndim > 3:
                    excl_arr = excl_arr.reshape((-1, excl_arr.shape[-2], excl_arr.shape[-1]))

                n_here = 0
                for em in excl_arr:
                    if em.shape != (h, w):
                        em = cv2.resize(em.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    em_bool = (em > 0.5)
                    if em_bool.any():
                        excl_union |= em_bool
                        n_here += 1
                total_n += n_here
                _log(f"[SAM3]   exclude '{tok}': {n_here} masks ({int(excl_union.sum())} cum px)")
            except Exception as e:
                import traceback
                _log(f"[SAM3]   exclude '{tok}' failed: {e}")
                traceback.print_exc()
                continue  # 다음 토큰 시도 — 한 토큰 실패가 main 패스 결과를 망치지 않게

        if total_n > 0:
            # 검출 누락 보강용 1px dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            excl_u8 = excl_union.astype(np.uint8) * 255
            excl_u8 = cv2.dilate(excl_u8, kernel, iterations=1)
            excl_bool = excl_u8 > 0

            before_px = int((mask > 0).sum())
            if before_px == 0:
                _log(f"[SAM3] exclude '{excl_text}': main mask가 비어있음 — skip")
            else:
                overlap_px = int(((mask > 0) & excl_bool).sum())
                kill_ratio = overlap_px / before_px
                # 80% 이상 지워버리면 의도와 다를 가능성 — 적용 취소
                if kill_ratio > 0.8:
                    msg = (f"제외 프롬프트 '{excl_text}'이(가) 마스크의 "
                           f"{kill_ratio*100:.0f}% 를 제거 — 너무 과해서 적용 취소됨. "
                           f"더 구체적인 단어로 시도해보세요 (예: 'face' → 'eyes')")
                    _log(f"[SAM3] exclude '{excl_text}' would remove "
                          f"{overlap_px}/{before_px} px ({kill_ratio*100:.1f}%) "
                          f"— main mask 거의 전체 — 적용 취소 (의도와 다를 가능성 높음)")
                    if notify:
                        try: notify('warning', msg)
                        except Exception: pass
                else:
                    mask[excl_bool] = 0
                    after_px = int((mask > 0).sum())
                    _log(f"[SAM3] excluded '{excl_text}': {total_n} masks total, "
                          f"removed {before_px - after_px} px "
                          f"({(before_px - after_px) / max(before_px, 1) * 100:.1f}% of main mask)")
        else:
            _log(f"[SAM3] exclude '{excl_text}': 모든 토큰이 검출 없음 — main mask 유지")

    return mask
