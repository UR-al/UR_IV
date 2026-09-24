"""Vue 일괄 처리(``start_batch``: 리사이즈/포맷 변환) — 순수 로직 (Qt 비의존).

예전 Vue 배치는 숨은 레거시 BatchTab → ``tabs/editor/batch_worker.BatchWorker`` 로 갔다(둘 다 지금은 삭제됨).
출력 폴더 기본값이 ``generated_images`` 자체였고 ``<원본이름><확장자>`` 에 충돌 검사 없이
cv2 로 다시 인코딩해 썼다. 그래서 생성 이미지를 리사이즈(확장자 유지)하거나 PNG→PNG 로
바꾸면 **원본이 제자리에서 덮어써지고** PNG 생성 파라미터까지 사라졌다.

이제 규칙:
* 결과는 ``<출력 폴더>/batch`` 에 **새 파일로만** 쓴다(원본·기존 결과를 절대 덮어쓰지 않음).
  이름은 리사이즈 ``<원본>_<W>x<H>.<확장자>``, 포맷 변환 ``<원본>.<새 확장자>`` 이고,
  이미 있으면 ``_2``, ``_3`` … 을 붙인다.
* 원본 메타데이터(PNG 텍스트 — parameters/workflow 등, EXIF, ICC)를 결과에 싣는다.
  다른 포맷으로 바꾸면 parameters 를 WebUI 호환 EXIF UserComment 로 옮긴다
  (``core.editor_save.encode_image`` 와 같은 규칙).
* 알파는 보존한다(JPEG 만 흰 바탕 합성). EXIF 방향은 화면처럼 적용해 세운다.
* 경로에 한글이 섞여도 되게 입출력은 전부 Python open/PIL 로 한다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from core.editor_save import (
    EditorSaveError,
    encode_image,
    format_for_path,
    load_image_array,
    read_source_metadata,
)
from core.output_files import same_file_path, write_new_file

logger = logging.getLogger(__name__)

BATCH_SUBDIR = 'batch'
MAX_DIMENSION = 16384
FORMAT_EXTENSIONS = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}
_FORMAT_ALIASES = {'PNG': 'PNG', 'JPG': 'JPEG', 'JPEG': 'JPEG', 'WEBP': 'WEBP'}
OPERATIONS = ('resize', 'format')


class BatchJobError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 배치 설정/처리 오류."""


@dataclass(frozen=True)
class BatchJob:
    operation: str            # 'resize' | 'format'
    width: int = 0            # resize 전용
    height: int = 0           # resize 전용
    target_format: str = ''   # format 전용: 'PNG' | 'JPEG' | 'WEBP'


def _dimension(value, label: str) -> int:
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        raise BatchJobError(f'배치: {label}를 숫자로 입력하세요') from None
    if not 1 <= number <= MAX_DIMENSION:
        raise BatchJobError(f'배치: {label}는 1~{MAX_DIMENSION} 사이여야 합니다')
    return number


def parse_batch_job(payload: Mapping) -> BatchJob:
    """Vue BatchView 페이로드 ``{operation, settings:{width,height,format}}`` → BatchJob."""
    data = payload if isinstance(payload, Mapping) else {}
    settings = data.get('settings') if isinstance(data.get('settings'), Mapping) else {}
    operation = str(data.get('operation') or 'resize').strip().lower()
    if operation not in OPERATIONS:
        raise BatchJobError(f'배치: 지원하지 않는 작업입니다 ({operation})')
    if operation == 'format':
        raw = str(settings.get('format') or 'PNG').strip().lstrip('.').upper()
        fmt = _FORMAT_ALIASES.get(raw)
        if fmt is None:
            raise BatchJobError(f'배치: 지원하지 않는 출력 포맷입니다 ({raw})')
        return BatchJob('format', target_format=fmt)
    return BatchJob(
        'resize',
        width=_dimension(settings.get('width', 1024), '너비'),
        height=_dimension(settings.get('height', 1024), '높이'),
    )


def batch_output_dir(output_root: str) -> str:
    return os.path.join(output_root, BATCH_SUBDIR)


def output_format(source_path: str, job: BatchJob) -> str:
    """결과 포맷 — 포맷 변환이면 고른 포맷, 리사이즈면 원본 포맷(쓸 수 없는 BMP 등은 PNG)."""
    if job.operation == 'format':
        return job.target_format
    return format_for_path(source_path) or 'PNG'


def planned_output_name(source_path: str, job: BatchJob) -> str:
    """충돌 처리 전 결과 파일 이름."""
    stem = os.path.splitext(os.path.basename(source_path))[0] or 'image'
    fmt = output_format(source_path, job)
    if job.operation == 'resize':
        source_ext = os.path.splitext(source_path)[1].lower()
        ext = source_ext if format_for_path(source_path) else FORMAT_EXTENSIONS[fmt]
        return f'{stem}_{job.width}x{job.height}{ext}'
    return f'{stem}{FORMAT_EXTENSIONS[fmt]}'


def _resize(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image
    mode = 'RGBA' if pixels.ndim == 3 and pixels.shape[2] == 4 else 'RGB'
    image = Image.fromarray(pixels, mode)
    # RGBA 는 PIL 이 premultiplied 로 리샘플링한다 — 투명 가장자리에 검은 테가 생기지 않는다.
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8).copy()


def display_name(path: str) -> str:
    """사용자 문구에 넣을 파일 이름 — 전체 경로(호스트 폴더 구조)는 원격 웹 클라이언트에도
    가므로 이름만 쓴다."""
    return os.path.basename(str(path or '').replace('\\', '/').rstrip('/')) or 'image'


def process_batch_file(source_path: str, output_dir: str, job: BatchJob) -> str:
    """파일 하나를 처리해 ``output_dir`` 에 새 파일로 쓰고, 쓴 경로를 돌려준다.

    실패는 배치 문구 + 파일 이름만 담은 ``BatchJobError`` 로 올린다. 아래 단계의 예외 문구
    (PIL 의 ``cannot identify image file 'C:\\…'``, 에디터용 '편집 이미지를…')는 그대로
    토스트에 실리면 호스트 경로가 드러나고 말도 맞지 않아 로그에만 남긴다.
    """
    name = display_name(source_path)
    try:
        pixels = load_image_array(source_path)
    except EditorSaveError as exc:
        logger.warning('batch: cannot read %s: %s', source_path, exc)
        raise BatchJobError(f'배치: 이미지를 읽을 수 없습니다 ({name})') from exc
    if job.operation == 'resize':
        try:
            pixels = _resize(pixels, job.width, job.height)
        except (MemoryError, OSError, ValueError) as exc:
            logger.warning('batch: resize failed for %s: %r', source_path, exc)
            raise BatchJobError(
                f'배치: {job.width}×{job.height} 로 크기를 바꾸지 못했습니다 ({name})') from exc
    meta = read_source_metadata(source_path)
    fmt = output_format(source_path, job)
    try:
        data = encode_image(pixels, fmt, meta)
    except (EditorSaveError, MemoryError, OSError, ValueError) as exc:
        logger.warning('batch: encode %s failed for %s: %r', fmt, source_path, exc)
        raise BatchJobError(f'배치: {fmt} 로 저장하지 못했습니다 ({name})') from exc
    target = os.path.join(output_dir, planned_output_name(source_path, job))
    try:
        written = write_new_file(target, data)
    except OSError as exc:
        logger.warning('batch: cannot write %s: %s', target, exc)
        reason = f': {exc.strerror}' if getattr(exc, 'strerror', None) else ''
        raise BatchJobError(f'배치: 결과 파일을 쓸 수 없습니다 ({name}){reason}') from exc
    if same_file_path(written, source_path):   # write_new_file 은 기존 파일을 피하므로 불가능해야 한다
        raise BatchJobError('배치: 원본 경로에는 쓸 수 없습니다')
    return written


def _default_input_resolver(raw: str):
    from core.path_safety import safe_input_path
    return safe_input_path(raw)


def batch_item_result(path, output_dir: str, job: BatchJob, *, resolve_path=None) -> tuple[bool, str]:
    """워커가 파일 하나마다 부르는 진입점 — 예외를 내지 않고 ``(성공, 결과 경로|사용자 문구)``.

    실패 문구는 전부 ``sanitize_for_ui`` 를 거친다: 이 문구가 JobTracker.first_error 를 거쳐
    최종 토스트(원격 웹 클라이언트 포함)로 가므로 호스트 절대 경로가 실리면 안 된다.
    """
    from core.error_handler import sanitize_for_ui
    name = display_name(path)
    try:
        source = (resolve_path or _default_input_resolver)(path)
        if not source:
            raise BatchJobError(f'배치: 입력 파일을 찾을 수 없거나 허용되지 않는 경로입니다 ({name})')
        written = process_batch_file(source, output_dir, job)
        return True, written.replace('\\', '/')
    except BatchJobError as exc:
        return False, sanitize_for_ui(str(exc)) or f'배치: 처리하지 못했습니다 ({name})'
    except Exception as exc:   # 예상 못 한 실패 — 원문은 로그에만
        logger.warning('batch image failed for %s: %r', path, exc)
        detail = sanitize_for_ui(str(exc), max_len=100)
        return False, f'배치: 처리하지 못했습니다 ({name})' + (f' — {detail}' if detail else '')
