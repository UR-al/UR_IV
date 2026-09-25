"""Vue I2IView 페이로드 → img2img 요청 (Qt 비의존 순수 로직).

예전에는 Vue 페이로드를 숨은 레거시 PyQt ``Img2ImgTab`` 위젯에 넣었다가 다시 읽었다
(``generate_from_payload`` → ``_on_generate``). 그 왕복 때문에
  1) 탭이 이전 요청의 입력 이미지(``current_base64``)를 계속 들고 있어, 이미지가 빠졌거나
     경로 파일이 사라진 요청도 오류 없이 **이전 이미지**로 생성됐고,
  2) seed 칸의 숫자가 아닌 글('abc'·'')이 ``int()`` 예외로 요청 전체를 멈췄고,
  3) data URL 입력은 검증 없이 그대로 보내 깨진 이미지도 백엔드 왕복 뒤에야 실패했다.
여기서는 페이로드만 보고 요청 전체를 만든다 — 인페인트(``core/inpaint_payload``)와 같은 방식.

규칙(예전 경로와 같다):
  * 입력 이미지 — data URL/base64 가 있으면 그것, 없으면 ``image_path``(검증된 경로)의 원본 바이트.
    둘 다 ``core/image_payload`` 규칙으로 보낸다(알파·방향 태그 등만 RGB PNG 로 다시 인코딩).
    경로는 갤러리가 보여 주는 이미지 형식 전부를 받는다(``input_image_exts`` — .tif·.apng·.avif 포함;
    예전 탭은 ``os.path.exists`` 만 봤다). 못 여는 이유(없음·막힌 형식·시스템 폴더)는 따로 알린다.
  * 재인코딩(PNG 압축 — 2048² RGBA 에 0.4초+)은 GUI 스레드에서 하지 않는다: 여기서는 헤더만 읽어
    크기·통과 여부를 정하고(``probe_image_bytes``), 압축은 생성 워커가 백엔드 호출 직전에
    ``I2IRequest.prepare_payload`` 로 한다.
  * 크기 — 비었거나 잘못된 변은 보낼 이미지의 크기(#120, ``resolve_target_size``).
  * 프롬프트·네거티브가 비었으면 T2I 의 값(main_prompt/main_negative).
  * seed — 비었거나 글·음수면 -1(랜덤). family 상한(``seed_max_for_family`` — Krea2 2^32-1,
    그 밖 uint64)을 넘으면 오류로 알린다(잘라 쓰면 다른 seed 로 조용히 생성된다).
  * Krea2 아이덴티티 편집 — ``_generation_family='krea2'`` + ``krea2_fidelity``, 참조 이미지가 있으면
    ``krea2_reference_image``(원본 바이트 그대로 — 형식 판정·업로드는 core/krea2_generation).

Vue 칸 ↔ A1111/Forge 필드:
  denoising → ``denoising_strength`` · cfg → ``cfg_scale`` ·
  resize_mode 0..3 → ``resize_mode`` (그대로 늘리기 / 잘라서 맞추기 / 여백 채우기 / Latent 리사이즈)
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Callable, Mapping, Optional

from core.generation_family import seed_max_for_family
from core.image_payload import (
    MAX_INPUT_BYTES,
    MAX_TARGET_DIMENSION,
    ImagePayloadError,
    decode_base64_image,
    encode_image_bytes,
    image_size_from_bytes,
    probe_image_bytes,
    resolve_target_size,
    strip_data_url,
)
from core.request_numbers import DEFAULT_SEED, SeedRangeError, clamped_float, clamped_int, parse_seed_checked

LABEL = "I2I"
REFERENCE_LABEL = "I2I 아이덴티티 참조"

FAMILY_STANDARD = "standard"
FAMILY_KREA2 = "krea2"

# 기본값 — 숨은 탭의 입력칸 초기값과 같다(Vue 가 칸을 보내지 않았을 때만 쓴다).
DEFAULT_DENOISING = 0.75
DEFAULT_RESIZE_MODE = 0
DEFAULT_STEPS = 20
DEFAULT_CFG = 7.0
DEFAULT_FIDELITY = 4.0

RESIZE_MODE_COUNT = 4
MAX_STEPS = 150
MAX_CFG = 30.0
# Krea2 편집 워크플로(core/creator_workflows 의 krea2_edit)가 받는 범위 = I2IView 슬라이더 범위
FIDELITY_RANGE = (0.5, 12.0)


@dataclass(frozen=True)
class I2IRequest:
    payload: dict          # 백엔드 img2img payload (alwayson_scripts 는 아직 비어 있다)
    family: str            # FAMILY_STANDARD | FAMILY_KREA2
    source_path: str       # 경로로 받았으면 검증된 절대 경로, data URL 이면 ''
    width: int             # 요청 크기(결과 크기는 백엔드가 정한다)
    height: int
    # 다시 인코딩해야 하는 입력의 원본 바이트 — 이때 payload['init_images'] 는 비어 있고
    # 워커 스레드의 prepare_payload 가 채운다. 원본 그대로 보내는 입력이면 None.
    pending_image: Optional[bytes] = field(default=None, repr=False)

    @property
    def needs_prepare(self) -> bool:
        return self.pending_image is not None

    def prepare_payload(self, payload: dict) -> None:
        """백엔드 호출 직전(생성 워커 스레드)에 부른다 — 알파·방향 태그·BMP/GIF/TIFF 입력을 RGB PNG 로.

        실패는 ``ImagePayloadError``(사용자 문구). 원본 그대로 보내는 입력이면 아무것도 하지 않는다.
        """
        if self.pending_image is None:
            return
        encoded = encode_image_bytes(self.pending_image, label=LABEL)
        payload["init_images"] = [encoded.b64]


def input_image_exts() -> frozenset[str]:
    """경로 입력으로 받는 확장자 = 갤러리가 보여 주는 정지/애니메이션 이미지 전부.

    send_to_i2i 는 갤러리·히스토리의 파일을 ``os.path.exists`` 만 보고 넘긴다. 예전엔 여기서
    ``safe_input_path`` 기본 목록(.tiff 는 있고 .tif·.apng·.avif 는 없다)을 써서, 보내진 .tif 가
    '파일을 찾을 수 없습니다'로 실패했다(ui/image_metadata_actions 의 TRANSPLANT_TARGET_EXTS 와 같은 함정).
    읽을 수 없는 내용은 이미지 헤더 판정이 '이미지를 읽을 수 없습니다'로 알린다.
    """
    from core.thumb_cache import THUMB_SOURCE_EXTS
    return THUMB_SOURCE_EXTS


def _path_problem(reason: str, raw: str, subject: str) -> str:
    """core.path_safety.check_input_path 의 실패 이유 → 사용자 문구(라벨 뒤에 붙는다).

    ``subject`` 는 '없음' 문구 앞에 붙는 말('입력 이미지 ' 등) — '파일을 찾을 수 없습니다'는
    정말 없을 때만 쓴다. 예전에는 막힌 확장자·시스템 폴더도 이 문구였다.
    """
    from core import path_safety
    if reason == path_safety.INPUT_MISSING:
        return f"{subject}파일을 찾을 수 없습니다"
    if reason == path_safety.INPUT_BLOCKED_EXT:
        suffix = PurePath(path_safety.strip_file_url(raw)).suffix.lower()
        return f"지원하지 않는 파일 형식입니다 ({suffix or '확장자 없음'})"
    if reason == path_safety.INPUT_FORBIDDEN:
        return "시스템 폴더의 파일은 열 수 없습니다"
    if reason == path_safety.INPUT_NOT_FILE:
        return "이미지 파일이 아닙니다"
    return "경로를 읽을 수 없습니다"


def _resolve(raw: str, label: str, resolve_path: Optional[Callable[[str], Optional[str]]],
             subject: str = "") -> str:
    """검증된 절대 경로. 실패하면 이유별 ``ImagePayloadError``.

    ``resolve_path`` 는 테스트용 대체(None 을 돌려주면 '찾을 수 없음').
    """
    if resolve_path is not None:
        resolved = resolve_path(raw)
        if not resolved:
            raise ImagePayloadError(f"{label}: {subject}파일을 찾을 수 없습니다")
        return resolved
    from core.path_safety import INPUT_MISSING, check_input_path
    resolved, reason = check_input_path(raw, allowed_exts=input_image_exts())
    if not resolved:
        raise ImagePayloadError(f"{label}: {_path_problem(reason or INPUT_MISSING, raw, subject)}")
    return resolved


def generation_family(value) -> str:
    """Vue ``generation_family`` → 'krea2' 또는 'standard'(그 밖의 값 전부)."""
    family = str(value or FAMILY_STANDARD).strip().lower()
    return FAMILY_KREA2 if family == FAMILY_KREA2 else FAMILY_STANDARD


def resize_mode(value) -> int:
    """Vue '크기 조정' 인덱스 → 0..3. 비었거나(null) 범위 밖·글이면 0(그대로 늘리기)."""
    mode = clamped_int(value, DEFAULT_RESIZE_MODE, -1, RESIZE_MODE_COUNT)
    return mode if 0 <= mode < RESIZE_MODE_COUNT else DEFAULT_RESIZE_MODE


def _read_limited(path: str, label: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ImagePayloadError(f"{label}: 파일을 열 수 없습니다.") from exc
    if len(data) > MAX_INPUT_BYTES:
        raise ImagePayloadError(f"{label}: 이미지 파일이 너무 큽니다.")
    return data


def _reference_image(data: Mapping, resolve_path: Optional[Callable[[str], Optional[str]]]) -> str:
    """Krea2 아이덴티티 참조 → base64('' 이면 참조 없음). 원본 바이트를 그대로 보낸다(예전과 같다).

    예전 경로는 참조 파일을 못 찾으면 알리지 않고 참조 없이 생성했다 — 고른 참조가 빠진 결과가
    조용히 나오지 않게 오류로 알린다. 깨진 참조로 백엔드 왕복을 하지 않도록 헤더를 먼저 읽는다.
    """
    inline = strip_data_url(data.get("reference_image"))
    if inline:
        image_size_from_bytes(decode_base64_image(inline, label=REFERENCE_LABEL), label=REFERENCE_LABEL)
        return inline
    raw_path = str(data.get("reference_path") or "").strip()
    if not raw_path:
        return ""
    resolved = _resolve(raw_path, REFERENCE_LABEL, resolve_path)
    raw = _read_limited(resolved, REFERENCE_LABEL)
    image_size_from_bytes(raw, label=REFERENCE_LABEL)
    return base64.b64encode(raw).decode("ascii")


def _input_image(data: Mapping, resolve_path) -> tuple[bytes, str]:
    """(입력 이미지 원본 바이트, 검증된 경로 — data URL 이면 '')."""
    image_field = strip_data_url(data.get("image"))
    if image_field:
        # 파일 선택·드롭·조명 편집 결과 — 경로가 없다.
        return decode_base64_image(image_field, label=LABEL), ""
    raw_path = str(data.get("image_path") or "").strip()
    if raw_path:
        # 갤러리·히스토리 전송 — 원본 바이트(예전 _load_image 의 메인 스레드 재인코딩 ~250ms 없음)
        resolved = _resolve(raw_path, LABEL, resolve_path, subject="입력 이미지 ")
        return _read_limited(resolved, LABEL), resolved
    raise ImagePayloadError(f"{LABEL}: 입력 이미지가 없습니다")


def parse_i2i_seed(value, family: str) -> int:
    """seed 칸 → 요청 seed. family 상한을 넘으면 ``ImagePayloadError`` — 조용히 자르지 않는다."""
    maximum = seed_max_for_family(family)
    try:
        return parse_seed_checked(value, maximum)
    except SeedRangeError as exc:
        raise ImagePayloadError(
            f"{LABEL}: 시드는 -1(랜덤) 또는 0~{maximum} 이어야 합니다 (받은 값: {exc.seed})") from exc


def build_i2i_request(
    vue_payload: Mapping,
    *,
    main_prompt: str = "",
    main_negative: str = "",
    resolve_path: Optional[Callable[[str], Optional[str]]] = None,
) -> I2IRequest:
    """I2IView 의 ``generate_i2i`` 페이로드 → 백엔드 요청.

    실패는 사용자에게 보여 줄 문구를 담은 ``ImagePayloadError`` 로 올린다. 입력 이미지를 다시
    인코딩해야 하면 ``init_images`` 는 비어 있다 — 워커가 ``prepare_payload`` 로 채운다.
    ``resolve_path`` 는 테스트용 경로 판정 대체(기본은 ``check_input_path`` + ``input_image_exts``).
    """
    data = vue_payload if isinstance(vue_payload, Mapping) else {}
    family = generation_family(data.get("generation_family"))

    raw_image, source_path = _input_image(data, resolve_path)
    # 헤더만 읽는다 — PNG 재압축(알파·방향 태그·BMP/GIF/TIFF)은 워커 스레드의 prepare_payload 몫
    probe = probe_image_bytes(raw_image, label=LABEL)
    if probe.passthrough:
        init_images = [base64.b64encode(raw_image).decode("ascii")]
        pending = None
    else:
        init_images = []
        pending = bytes(raw_image)

    reference = _reference_image(data, resolve_path) if family == FAMILY_KREA2 else ""
    seed = parse_i2i_seed(data.get("seed", DEFAULT_SEED), family)

    width, height = resolve_target_size(data.get("width"), data.get("height"),
                                        (probe.width, probe.height))
    # 이미지 한 변이 요청 한도보다 크면 resolve_target_size 가 정하지 못한다 — 한도로 자른다.
    width = width or min(probe.width, MAX_TARGET_DIMENSION)
    height = height or min(probe.height, MAX_TARGET_DIMENSION)

    prompt = str(data.get("prompt") or "").strip() or str(main_prompt or "")
    negative = str(data.get("negative_prompt") or "").strip() or str(main_negative or "")

    payload = {
        "init_images": init_images,
        "prompt": prompt,
        "negative_prompt": negative,
        "denoising_strength": clamped_float(data.get("denoising"), DEFAULT_DENOISING, 0.0, 1.0),
        "resize_mode": resize_mode(data.get("resize_mode")),
        "steps": clamped_int(data.get("steps"), DEFAULT_STEPS, 1, MAX_STEPS),
        "cfg_scale": clamped_float(data.get("cfg"), DEFAULT_CFG, 0.0, MAX_CFG),
        "seed": seed,
        "width": width,
        "height": height,
        "send_images": True,
        "save_images": True,
        "alwayson_scripts": {},
    }
    if family == FAMILY_KREA2:
        low, high = FIDELITY_RANGE
        payload["_generation_family"] = FAMILY_KREA2
        payload["krea2_fidelity"] = clamped_float(data.get("fidelity"), DEFAULT_FIDELITY, low, high)
        if reference:
            payload["krea2_reference_image"] = reference
    return I2IRequest(payload=payload, family=family, source_path=source_path,
                      width=width, height=height, pending_image=pending)


__all__ = [
    "DEFAULT_CFG",
    "DEFAULT_DENOISING",
    "DEFAULT_FIDELITY",
    "DEFAULT_RESIZE_MODE",
    "DEFAULT_STEPS",
    "FAMILY_KREA2",
    "FAMILY_STANDARD",
    "FIDELITY_RANGE",
    "I2IRequest",
    "build_i2i_request",
    "generation_family",
    "input_image_exts",
    "parse_i2i_seed",
    "resize_mode",
]
