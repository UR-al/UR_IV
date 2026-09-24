# core/compare_gif.py
"""PNG Info 비교(Compare) 탭의 Before/After 전환 GIF — Qt 비의존.

예전엔 동기 슬롯(exportCompareGif)이 GUI 스레드에서 풀해상도 LANCZOS 2회 + 16프레임 blend +
optimize GIF 저장을 해 창이 수 초씩 멈췄고, RGBA(배경 제거 결과)와 RGB 를 섞으면
``Image.blend`` 가 ``ValueError('images do not match')`` 로 실패했다. 지금은
VueBridge.requestCompareGif 가 워커 스레드에서 이 모듈을 부르고 compareGifReady 로 알린다.

- 먼저 EXIF Orientation 을 적용한다 — 비교 슬라이더의 ``<img>``(Chromium 기본
  ``image-orientation: from-image``)와 같은 방향. 예전엔 폰·카메라 JPEG/WebP 가
  옆으로 누운 채 나오고, 똑바른 이미지와 섞으면 공통 크기까지 어긋나 찌그러졌다.
  방향은 Chromium 처럼 **진짜 EXIF 만** 본다(``core.cv_io.image_exif_orientation`` — 에디터와 같은 규칙).
  XMP ``tiff:Orientation`` 까지 따르는 Pillow ``exif_transpose`` 만 쓰면 XMP 에만 방향이 적힌 이미지가
  슬라이더와 달리 GIF 에서만 돌아가고, 똑바른 짝과 섞이면 공통 크기가 찌그러졌다(Codex S5 #4).
  깨진 EXIF 만 무시하고, 픽셀 디코드 실패(손상 PNG 등)는 예외로 올린다 — 반쯤 디코드된 GIF 를 쓰지 않는다.
- 두 이미지를 RGB 로 맞춘 뒤(투명 부분은 흰 배경에 합성), 작은 쪽 크기로 통일한다.
- 긴 변이 ``MAX_SIDE`` 를 넘으면 비율을 지켜 줄인다 — GIF(256색)에서 풀해상도는 품질 이득 없이
  시간·용량만 든다.
- 프레임: 0→1 로 9장, 되돌아오며 7장(양 끝 중복 없이) = 16장.
- 파일은 새 이름으로만 쓴다(core.output_files.write_new_file) — 같은 초의 두 번째 내보내기가
  첫 GIF 를 덮어쓰지 않는다.
"""
from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Optional

MAX_SIDE = 1024
BLEND_STEPS = 8
MIN_DURATION_MS = 20
MAX_DURATION_MS = 10_000


def _to_rgb(image):
    """RGBA/LA/P(투명) → 흰 배경 합성 RGB, 그 밖의 모드 → RGB."""
    from PIL import Image

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB") if image.mode != "RGB" else image


def _upright(image):
    """EXIF Orientation 을 제자리(in_place)로 적용한 같은 이미지. ``.info``(투명색)는 유지된다.

    디코드는 ``try`` 밖에서 먼저 끝낸다 — ``exif_transpose`` 가 맨 처음 ``load()`` 를 부르는데,
    디코더 오류(IDAT zlib 손상 등, err_code<0)면 Pillow 가 ``tile`` 을 비운 뒤 던지므로 여기서
    삼키면 뒤의 ``convert()`` 가 반쯤 디코드된(아래가 검정인) 픽셀로 조용히 통과했다.
    그래서 넓은 ``except`` 는 EXIF 읽기·적용만 감싼다: EXIF 가 깨져 방향을 읽을 수 없으면 저장된
    그대로 쓴다(예전 동작). 제자리 적용이라 태그 없는 흔한 입력에 풀해상도 사본을 하나 더 들지 않는다.

    ``exif_transpose`` 는 **진짜 EXIF 방향**(``core.cv_io.image_exif_orientation``)이 1 이 아닐 때만 부른다
    — 그땐 Pillow 도 같은 EXIF 값을 쓴다(Pillow 는 EXIF 에 방향 태그가 없을 때만 XMP 를 본다). XMP 에만
    방향이 있으면 슬라이더의 ``<img>`` 처럼 저장된 그대로 둔다. 방향은 ``load()`` 뒤에 읽는다 — TIFF 는
    load 가 이미 세우고 태그를 지우고, PNG 의 IDAT 뒤 eXIf 는 load 가 ``info['exif']`` 로 채운다.
    단 TIFF 의 load 는 XMP 에만 적힌 방향으로도 돌았다 — load 전에 ``suppress_xmp_orientation`` 으로 막는다
    (Codex S5 #1-b). 방향 5~8 무압축 TIFF 의 mmap 뒤섞임은 여는 쪽(``core.cv_io.open_pil_image``)이 막는다.
    """
    from PIL import ImageOps

    from core.cv_io import image_exif_orientation, suppress_xmp_orientation

    suppress_xmp_orientation(image)   # TIFF load_end 의 exif_transpose 가 XMP 방향을 따르지 않게(load 전)
    image.load()   # 디코드 실패는 여기서 그대로 올라간다 → 브리지가 {requestId, error} 로 답한다
    try:
        if image_exif_orientation(image) != 1:
            ImageOps.exif_transpose(image, in_place=True)
    except Exception:
        pass
    return image


def target_size(size_a: tuple[int, int], size_b: tuple[int, int], max_side: int = MAX_SIDE) -> tuple[int, int]:
    """두 이미지의 공통 크기 — 작은 쪽에 맞추고 긴 변을 max_side 이하로(비율 유지, 최소 1px)."""
    width = max(1, min(int(size_a[0]), int(size_b[0])))
    height = max(1, min(int(size_a[1]), int(size_b[1])))
    longest = max(width, height)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return width, height


def blend_alphas(steps: int = BLEND_STEPS) -> list[float]:
    """0→1(steps+1 장) 뒤 1→0 으로 되돌아오는 블렌드 계수 — 양 끝은 한 번씩만."""
    forward = [i / steps for i in range(steps + 1)]
    backward = [i / steps for i in range(steps - 1, 0, -1)]
    return forward + backward


def build_compare_gif(before_path: str, after_path: str, duration_ms: int, loops: int,
                      *, max_side: int = MAX_SIDE) -> tuple[bytes, int]:
    """두 이미지 경로 → (GIF 바이트, 프레임 수)."""
    from PIL import Image

    from core.cv_io import open_pil_image

    duration = max(MIN_DURATION_MS, min(int(duration_ms or 0), MAX_DURATION_MS))
    loop = max(0, int(loops or 0))
    # 경로로 열지 않는다 — 방향 5~8 무압축 TIFF 가 Pillow mmap 지름길에서 뒤섞였다(Codex S5 #1-a)
    with open_pil_image(before_path) as opened_a, open_pil_image(after_path) as opened_b:
        image_a = _to_rgb(_upright(opened_a))
        image_b = _to_rgb(_upright(opened_b))
        size = target_size(image_a.size, image_b.size, max_side)
        resample = Image.Resampling.LANCZOS
        if image_a.size != size:
            image_a = image_a.resize(size, resample)
        if image_b.size != size:
            image_b = image_b.resize(size, resample)
        frames = [Image.blend(image_a, image_b, alpha) for alpha in blend_alphas()]
    buffer = BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=loop, optimize=True)
    return buffer.getvalue(), len(frames)


def export_compare_gif(before_path: str, after_path: str, duration_ms: int, loops: int,
                       out_dir: str, *, now: Optional[float] = None) -> dict:
    """GIF 를 만들어 ``out_dir`` 에 새 파일로 쓴다 → ``{"path", "frames"}`` (경로는 슬래시)."""
    from core.output_files import write_new_file

    data, frames = build_compare_gif(before_path, after_path, duration_ms, loops)
    os.makedirs(out_dir, exist_ok=True)
    stamp = int(time.time() if now is None else now)
    out_path = write_new_file(os.path.join(out_dir, f"compare_{stamp}.gif"), data)
    return {"path": str(out_path).replace("\\", "/"), "frames": frames}


__all__ = ["MAX_SIDE", "blend_alphas", "build_compare_gif", "export_compare_gif", "target_size"]
