"""
Image Metadata — PNG / JPEG / WebP 생성 메타데이터 추출 + 이식.

NAIA 2.0 ``tabs/png_info_tab.py`` 패턴 참고 (NAI Comment / Stealth PNG 제외).

지원 포맷:
- **PNG**: ``tEXt`` 청크의 ``parameters`` (WebUI) / ``workflow`` / ``prompt`` (ComfyUI)
- **JPEG**: EXIF ``UserComment`` (WebUI 호환), ``ImageDescription``
- **WebP**: EXIF/XMP (PIL이 PNG 텍스트 청크와 비슷하게 노출)

목적:
1. 추출 — 이미지에서 생성 파라미터를 dict로
2. 이식 — 한 이미지의 메타를 다른 이미지에 복사 (PNG Info '메타 이식' 액션)
3. 수정 — Gallery 'EXIF 저장'이 프롬프트만 바꾸고 파라미터 꼬리와 다른 PNG 청크는 바이트 그대로 보존

A1111 infotext(``parameters``)의 **유일한** 파서/직렬화기다. UI 액션·워커·브리지가
각자 ``split('\\nNegative prompt: ')`` 를 하던 복제본은 모두 이 모듈을 부른다.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from core.infotext import (  # A1111 infotext 문법 — 이 모듈 이름으로도 다시 내보낸다
    format_infotext,
    format_parameters_line,
    parse_parameters_text,
    quote_param_value,
    split_infotext,
    unquote_param_value,
)
from utils.app_logger import get_logger

_logger = get_logger("image_metadata")

# 메타 이식 결과(RGB/RGBA PNG)에 원본 ICC 프로파일을 그대로 옮겨도 되는 원본 모드
_RGB_PROFILE_MODES = frozenset({"RGB", "RGBA", "RGBX", "RGBa", "P", "PA"})
# 생성 메타 텍스트 키. PNG 헤더(IDAT 앞)가 이것들로 **완결된 기록**이면 뒤쪽 텍스트 청크를 훑지 않는다
# (``_header_record_complete``). 예전엔 하나만 있어도 건너뛰어, 헤더에 workflow 만 있고 prompt(API
# 그래프)가 IEND 앞에 붙은 PNG 는 프롬프트가 비고 '모호함'이 됐다(Codex S5 #3).
_GENERATION_TEXT_KEYS = frozenset({"parameters", "prompt", "workflow"})
_COMFY_TEXT_KEYS = frozenset({"prompt", "workflow"})


def _header_record_complete(info: dict) -> bool:
    """헤더 텍스트만으로 생성 기록이 완결됐는지 — 그러면 뒤쪽 청크(IDAT 수백 개 너머)를 훑지 않는다.

    * ``parameters`` (WebUI·Forge) — 소스·프롬프트를 이것이 정하므로 뒤쪽 무엇도 결과를 바꾸지 못한다.
      뒤쪽에 따로 붙은 Comfy 그래프는 원문 표시·이식에만 쓰였을 것이라, 흔한 Forge PNG 마다
      파일을 한 번 더 여는 비용(폴더 검색에서 파일당 ~0.2ms)을 들이지 않는다.
    * ``prompt`` + ``workflow`` — ComfyUI SaveImage 기본(뒤쪽 parameters 는 아래에서 어차피 버린다).
    그 밖(헤더에 prompt 나 workflow 하나만)은 뒤쪽이 빠진 키를 채울 수 있어 훑는다.
    """
    return bool(info.get("parameters")) or _COMFY_TEXT_KEYS.issubset(info)


class MetadataSource:
    WEBUI = "webui"
    COMFYUI = "comfyui"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    """추출된 생성 메타.

    포맷 간 차이를 흡수하는 정규화된 표현.
    원본 raw 텍스트도 ``raw_parameters`` / ``raw_workflow``에 보존.
    """
    source: str = MetadataSource.UNKNOWN

    # 정규화된 필드 — WebUI Parameters 파싱 결과
    prompt: str = ""
    negative_prompt: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)  # steps, cfg, sampler 등

    # ComfyUI 원본
    workflow: Optional[dict] = None  # ComfyUI workflow JSON (graph format)
    prompt_graph: Optional[dict] = None  # ComfyUI prompt JSON (API format)

    # 보존용 원본
    raw_parameters: str = ""  # WebUI Parameters 텍스트 그대로
    raw_workflow: str = ""    # ComfyUI workflow JSON 그대로
    raw_prompt: str = ""      # ComfyUI prompt JSON 그대로
    # WebUI 파라미터 꼬리 원문(``Steps: …`` 줄부터 끝까지, Template 줄 포함).
    # 'Parameters 복사'·EXIF 저장이 따옴표·순서를 잃지 않게 재조립하지 않고 그대로 쓴다.
    parameters_text: str = ""
    metadata_warnings: list[str] = field(default_factory=list)
    prompt_candidates: list[dict] = field(default_factory=list)
    comfy_parse_complete: bool = False

    def has_any(self) -> bool:
        return bool(
            self.prompt or self.negative_prompt or self.parameters
            or self.workflow or self.prompt_graph or self.raw_parameters or self.raw_prompt or self.raw_workflow
        )


# ─────────────────────────────────────────────
# 추출
# ─────────────────────────────────────────────

def extract_from_file(path: Union[str, Path]) -> ImageMetadata:
    """파일에서 메타데이터 추출. 실패해도 빈 ``ImageMetadata`` 반환."""
    p = Path(path)
    if not p.is_file():
        return ImageMetadata()
    try:
        from PIL import Image
        with Image.open(p) as img:
            return extract_from_pil(img)
    except Exception:
        _logger.exception(f"메타 추출 실패: {p}")
        return ImageMetadata()


def png_trailing_text(img) -> dict:
    """PNG 의 IDAT 뒤 텍스트 청크 {키: 값}. PNG 가 아니거나 못 읽으면 빈 dict.

    Pillow 는 열 때 IDAT 앞 청크만 ``img.info`` 에 담고, 뒤쪽 청크는 ``img.text`` 로 픽셀을
    전부 디코드해야 준다(일부 편집기·청크 도구가 메타를 IEND 앞에 붙인다). 여기서는 열린
    파일을 청크 머리만 훑어 읽고(core.png_chunks) 파일 위치를 되돌린다 — 픽셀 디코드 없음.
    이미 load() 한 이미지(파일이 닫힘)는 ``img.text`` 가 픽셀 디코드 없이 전부 준다.
    """
    return _png_trailing_text_and_exif(img)[0]


# PNG eXIf 를 아직 훑지 않았다는 표시(None 은 '훑었는데 없음')
_UNSCANNED: Any = object()


def _png_trailing_text_and_exif(img) -> tuple[dict, Optional[bytes]]:
    """(IDAT 뒤 텍스트 청크, eXIf 청크 데이터) — 청크 머리 한 번 훑기로. PNG 가 아니면 ({}, None).

    파일에서 연 이미지는 같은 파일을 **버퍼 없는** 핸들로 따로 열어 훑는다 — Pillow 의 버퍼 핸들로
    청크마다 seek + read(8) 하면 매번 8KB 버퍼를 다시 채워, IDAT 수천 개짜리 PNG 에서 파일 크기
    이상을 읽었다. 따로 못 열면(스트림에서 연 이미지 등) Pillow 핸들로 훑고 위치를 되돌린다.
    이미 load() 한 이미지(파일이 닫힘)는 ``img.text`` 가 픽셀 디코드 없이 전부 주고, EXIF 는
    ``getexif()`` 가 다시 디코드하지 않으므로 찾지 않는다(None).
    """
    if getattr(img, "format", None) != "PNG":
        return {}, None
    fp = getattr(img, "fp", None)
    if fp is None:
        try:
            return {str(k): str(v) for k, v in (getattr(img, "text", None) or {}).items()}, None
        except Exception:
            return {}, None
    import io
    from core.png_chunks import read_text_and_exif
    filename = getattr(img, "filename", None)
    if isinstance(filename, str) and filename and isinstance(fp, io.BufferedReader):
        try:
            with open(filename, "rb", buffering=0) as unbuffered:
                # 같은 파일인지 확인한다(상대 경로·그사이 교체된 파일이면 Pillow 가 연 헤더와 다른 파일이다)
                if os.path.sameopenfile(unbuffered.fileno(), fp.fileno()):
                    return read_text_and_exif(unbuffered, after_idat=True)
        except Exception:
            pass
        # 아래에서 Pillow 핸들로
    try:
        position = fp.tell()
    except Exception:
        return {}, None
    try:
        return read_text_and_exif(fp, after_idat=True)
    except Exception:
        return {}, None
    finally:
        try:
            fp.seek(position)
        except Exception:
            pass


def extract_from_pil(img, *, extra_text: Optional[dict] = None) -> ImageMetadata:
    """PIL Image 객체에서 추출.

    ``extra_text`` 는 ``img.info`` 에 없는 텍스트 청크를 보충한다(``img.info`` 값이 우선).
    생략하면 PNG 의 IDAT 뒤 텍스트 청크(``png_trailing_text``)를 쓴다 — Gallery·PNG Info·
    검색·메타 이식·'EXIF 저장'·배치 워커가 같은 파일에서 같은 메타를 본다.
    A1111 infotext 가 ``Description`` 청크에만 있으면(일부 도구) 파라미터 줄이 있을 때만 쓴다.
    """
    meta = ImageMetadata()

    raw_info = dict(getattr(img, "info", None) or {})
    png_exif: Any = _UNSCANNED
    if extra_text is None:
        # 헤더가 완결된 기록이면 그것이 이긴다 — 뒤쪽 청크(큰 PNG 는 IDAT 수백 개)를 훑지 않는다.
        # 훑을 때는 2) 의 EXIF(eXIf)도 같이 찾는다 — 청크 머리를 두 번 훑지 않는다.
        if _header_record_complete(raw_info):
            extra_text = {}
        else:
            extra_text, png_exif = _png_trailing_text_and_exif(img)
            if _GENERATION_TEXT_KEYS.intersection(raw_info):
                # 헤더가 반쪽 Comfy 기록(prompt 나 workflow 하나)이면 뒤쪽은 **빠진 키만** 채운다(키마다
                # 헤더 우선 — 아래 setdefault). 뒤쪽 parameters 는 버린다: 헤더 기록이 완결됐을 때(빠른 길)는
                # 보지도 않는 뒤쪽 A1111 청크가, 헤더가 반쪽일 때만 소스를 WEBUI 로 뒤집으면 안 된다.
                extra_text = {k: v for k, v in extra_text.items() if k != "parameters"}
    for key, value in (extra_text or {}).items():
        raw_info.setdefault(key, value)

    # 1) PNG tEXt / iTXt
    if raw_info:
        # WebUI: "parameters" 키
        if raw_info.get("parameters"):
            meta.raw_parameters = str(raw_info["parameters"])
            meta.source = MetadataSource.WEBUI
            _parse_webui_parameters(meta.raw_parameters, meta)

        # ComfyUI: "workflow" (graph) / "prompt" (API)
        if "workflow" in raw_info:
            meta.raw_workflow = str(raw_info["workflow"])
            try:
                value = _read_graph_json(meta.raw_workflow)
                meta.workflow = value if isinstance(value, dict) else None
                if meta.source == MetadataSource.UNKNOWN:
                    meta.source = MetadataSource.COMFYUI
            except (ValueError, RecursionError):
                meta.metadata_warnings.append("workflow JSON이 손상되었거나 크기 제한을 초과했습니다. 원문은 보존했습니다.")

        if "prompt" in raw_info:
            # Display WebUI fields first, but never discard embedded Comfy JSON.
            meta.raw_prompt = str(raw_info["prompt"])
            try:
                value = _read_graph_json(meta.raw_prompt)
                meta.prompt_graph = value if isinstance(value, dict) else None
                if meta.source == MetadataSource.UNKNOWN:
                    meta.source = MetadataSource.COMFYUI
            except (ValueError, RecursionError):
                meta.metadata_warnings.append("prompt JSON이 손상되었거나 크기 제한을 초과했습니다. 원문은 보존했습니다.")

    # 2) JPEG/WebP — EXIF UserComment
    if meta.source == MetadataSource.UNKNOWN:
        comment = _read_exif_user_comment(img, png_exif)
        if comment:
            meta.raw_parameters = comment
            meta.source = MetadataSource.WEBUI
            _parse_webui_parameters(comment, meta)

    # 3) 일부 도구는 A1111 형식 infotext 를 Description 청크에 쓴다 — 파라미터 줄이 있을 때만 믿는다.
    description = raw_info.get("Description")
    if not meta.has_any() and isinstance(description, str):
        parsed = parse_infotext(description)
        if parsed.parameters:
            meta = parsed

    if meta.source == MetadataSource.COMFYUI:
        from core.comfy_metadata import parse_comfy_metadata
        parsed = parse_comfy_metadata(meta.prompt_graph, meta.workflow)
        meta.prompt = parsed["prompt"]
        meta.negative_prompt = parsed["negative_prompt"]
        meta.parameters = parsed["parameters"]
        meta.metadata_warnings.extend(parsed["warnings"])
        meta.prompt_candidates = parsed["candidates"]
        meta.comfy_parse_complete = parsed["complete"]

    return meta


def _read_graph_json(raw: str):
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Metadata graph exceeds 8 MiB")
    return json.loads(raw)


def read_metadata_for_ui(path: Union[str, Path]) -> dict:
    """Gallery/PNG Info data, preserving the existing bridge field names.

    Raw graph JSON is for display/copy only; ``parameters`` and ``params_line``
    contain the conservative normalized settings usable by existing UI actions.
    """
    from PIL import Image
    file_path = Path(path)
    with Image.open(file_path) as image:
        meta = extract_from_pil(image)
        size = f"{image.width} × {image.height}"
    return {**metadata_ui_fields(meta),
            "path": str(file_path).replace("\\", "/"), "filename": file_path.name, "size": size}


def metadata_ui_fields(meta: ImageMetadata) -> dict:
    """브리지 필드(경로·크기 제외). 파일 없이 infotext 문자열만 받은 레거시 페이로드도 같은 모양."""
    can_apply = _can_apply(meta)
    return {"source": meta.source, "raw": meta.raw_parameters or meta.raw_prompt or meta.raw_workflow,
            "prompt": meta.prompt, "negative": meta.negative_prompt,
            "parameters": dict(meta.parameters), "params_line": parameters_line(meta),
            # 표시용 그룹 — 따옴표를 아는 dict 에서 만든다(표시 문자열을 다시 파싱하지 않는다).
            "params": group_parameters(meta.parameters) if meta.parameters else None,
            "raw_prompt": meta.raw_prompt, "raw_workflow": meta.raw_workflow,
            "metadata_warnings": list(meta.metadata_warnings), "prompt_candidates": list(meta.prompt_candidates),
            "can_apply": can_apply, "metadata_ambiguous": meta.source == MetadataSource.COMFYUI and not can_apply}


def _can_apply(meta: ImageMetadata) -> bool:
    """프롬프트를 UI·워커에 자동 적용해도 되는가.

    ComfyUI 는 모든 샘플러 후보가 같은 (prompt, negative) 로 확정될 때만, WebUI 는
    무엇이든 읽혔으면 True. PNG Info·Gallery·History·배치 워커가 같은 판정을 쓴다.
    """
    if meta.source == MetadataSource.COMFYUI:
        candidates = meta.prompt_candidates
        known = bool(candidates) and all(item["positive_known"] and item["negative_known"] for item in candidates)
        same_prompts = len({(item["prompt"], item["negative"]) for item in candidates}) <= 1
        return bool(meta.comfy_parse_complete and known and same_prompts)
    return bool(meta.prompt or meta.negative_prompt or meta.parameters)


def parameters_line(meta: ImageMetadata) -> str:
    """'Parameters' 표시·복사 문자열.

    WebUI 는 원문 꼬리를 그대로(따옴표·순서·Template 줄 보존), ComfyUI 처럼 원문이 없는
    소스는 A1111 규칙(쉼표·콜론·줄바꿈이 든 값은 JSON 따옴표)으로 직렬화한다.
    """
    if meta.parameters_text:
        return meta.parameters_text
    return format_parameters_line(meta.parameters)


# 워커(SAM3/ADetailer 배치 'EXIF 프롬프트 사용')가 결과와 함께 알리는 경고 문구
EXIF_WARNING_MISSING = "이미지에 생성 메타데이터가 없어 EXIF 프롬프트를 적용하지 못했습니다."
EXIF_WARNING_UNREADABLE = "이미지 메타데이터를 읽지 못해 EXIF 프롬프트를 적용하지 못했습니다."
EXIF_WARNING_AMBIGUOUS = "ComfyUI 프롬프트를 확정할 수 없어 EXIF 프롬프트를 적용하지 않았습니다."
EXIF_WARNING_NO_POSITIVE = "메타데이터에 긍정 프롬프트가 없어 네거티브만 적용했습니다."


def read_applicable_prompts(path: Union[str, Path]) -> tuple[str, str, str]:
    """배치 워커용 — (prompt, negative, warning).

    읽기는 UI·검색과 같은 ``extract_from_pil`` 이다 — PNG 의 IDAT 뒤 텍스트 청크와
    Description 청크의 infotext 까지 같은 규칙으로 본다. JPEG/WebP 는 EXIF UserComment 를 읽는다.
    ComfyUI 그래프가 여러 갈래이거나 해석하지 못한 노드가 있으면 추측하지 않고 빈 프롬프트 +
    경고를 돌려준다(UI 액션과 같은 규칙).
    """
    file_path = Path(path)
    if not file_path.is_file():
        return "", "", EXIF_WARNING_UNREADABLE
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            meta = extract_from_pil(img)
    except Exception:
        _logger.warning("EXIF 프롬프트 읽기 실패: %s", file_path.name, exc_info=True)
        return "", "", EXIF_WARNING_UNREADABLE
    if not meta.has_any():
        return "", "", EXIF_WARNING_MISSING
    if not _can_apply(meta):
        if meta.source == MetadataSource.COMFYUI:
            return "", "", EXIF_WARNING_AMBIGUOUS
        return "", "", EXIF_WARNING_MISSING
    if not meta.prompt:
        return "", meta.negative_prompt, (EXIF_WARNING_NO_POSITIVE if meta.negative_prompt else EXIF_WARNING_MISSING)
    return meta.prompt, meta.negative_prompt, ""


def _exif_for_user_comment(img, png_exif: Any = _UNSCANNED):
    """UserComment 를 읽을 EXIF(없으면 None). PNG 는 픽셀을 디코드하지 않는다.

    PNG 의 Pillow ``getexif()`` 는 헤더에 eXIf 가 없으면 ``load()`` 로 픽셀을 전부 디코드한다 —
    메타 없는 큰 PNG(27MB 실측 43ms)에서 GUI 스레드 슬롯 getImageExif·saveImagePrompt 와 웹 facade 가
    그만큼 멈췄다. 그런 PNG 는 청크 머리를 훑어 찾은 eXIf 를 쓴다(``png_exif`` — ``extract_from_pil``
    이 뒤쪽 텍스트와 함께 이미 찾았으면 그것, 아니면 여기서 훑는다). 헤더에 eXIf 가 있거나 이미
    load() 한 이미지(파일이 닫힘)는 ``getexif()`` 가 디코드하지 않으므로 그대로 쓴다.
    """
    if not hasattr(img, "getexif"):
        return None
    info = getattr(img, "info", None) or {}
    if getattr(img, "format", None) != "PNG" or info.get("exif") or getattr(img, "fp", None) is None:
        return img.getexif()
    raw = _png_trailing_text_and_exif(img)[1] if png_exif is _UNSCANNED else png_exif
    if not raw:
        return None
    from PIL import Image
    exif = Image.Exif()
    exif.load(bytes(raw))
    return exif


def _read_exif_user_comment(img, png_exif: Any = _UNSCANNED) -> str:
    """EXIF UserComment(0x9286) 또는 ImageDescription(0x010E) 읽기.

    ``png_exif``: PNG 에서 이미 훑어 찾은 eXIf 데이터(None = 없음) — 생략하면 필요할 때 훑는다.
    """
    try:
        exif = _exif_for_user_comment(img, png_exif)
        if not exif:
            return ""
        # UserComment
        v = exif.get(0x9286)
        if v is None and hasattr(exif, "get_ifd"):
            v = exif.get_ifd(0x8769).get(0x9286)
        if isinstance(v, (bytes, bytearray)):
            # WebUI는 보통 UTF-16/UTF-8 prefix 후 텍스트
            try:
                if v[:8] == b"UNICODE\0":
                    return v[8:].decode("utf-16-be", errors="replace").rstrip("\x00")
                if v[:8] == b"ASCII\0\0\0":
                    return v[8:].decode("ascii", errors="replace").rstrip("\x00")
                return v.decode("utf-8", errors="replace").rstrip("\x00")
            except Exception:
                return ""
        if isinstance(v, str):
            return v
        # ImageDescription
        d = exif.get(0x010E)
        if isinstance(d, str):
            return d
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────
# A1111 infotext 파서 / 직렬화기 (유일본) — 문법은 core/infotext.py
# ─────────────────────────────────────────────
# 줄 분류·쌍 분리는 모두 선형이다(점 키·따옴표 없는 JSON 값·여러 줄 Template 포함).
# 기존 호출부(vue_bridge·metadata_actions·테스트)는 이 모듈 이름으로 부른다(맨 위 import 로 다시 내보냄).


def parse_infotext(text: str) -> ImageMetadata:
    """A1111 infotext 문자열 하나를 ImageMetadata 로(source=webui)."""
    meta = ImageMetadata(source=MetadataSource.WEBUI, raw_parameters=str(text or ""))
    _parse_webui_parameters(meta.raw_parameters, meta)
    return meta


def _parse_webui_parameters(text: str, meta: ImageMetadata) -> None:
    """WebUI ``parameters`` 텍스트 파싱."""
    if not text:
        return
    prompt, negative, tail = split_infotext(text)
    meta.prompt = prompt
    meta.negative_prompt = negative
    meta.parameters_text = tail
    meta.parameters = parse_parameters_text(tail) if tail else {}


# 표시용 그룹 — PNG Info·Gallery·History 가 같은 줄 구성으로 보여 준다.
_GROUP_GENERATION = ("Steps", "Sampler", "Schedule type")
_GROUP_CORE = ("CFG scale", "Seed", "Size")
_GROUP_MODEL = ("Model", "Model hash", "VAE", "Clip skip")
_EXTENSION_MARKERS = ("adetailer", "sam3", "negpip", "controlnet", "ad_", "tiled")


def group_parameters(params: dict) -> dict[str, str]:
    """파라미터 dict → 표시 그룹(generation/core/model/hires/extensions/other).

    값은 A1111 규칙으로 다시 따옴표쳐서 한 줄로 이어도 모호하지 않다.
    """
    remaining = dict(params or {})

    def take(keys) -> str:
        parts = []
        for key in keys:
            if key in remaining:
                parts.append(f"{key}: {quote_param_value(remaining.pop(key))}")
        return ", ".join(parts)

    groups = {"generation": take(_GROUP_GENERATION), "core": take(_GROUP_CORE), "model": take(_GROUP_MODEL)}
    groups["hires"] = take([
        key for key in list(remaining)
        if key.lower().startswith("hires") or key.lower().startswith("hr ") or key == "Denoising strength"
    ])
    groups["extensions"] = take([
        key for key in list(remaining) if any(marker in key.lower() for marker in _EXTENSION_MARKERS)
    ])
    groups["other"] = take(list(remaining))
    return groups


# ─────────────────────────────────────────────
# 수정 (Gallery 'EXIF 저장')
# ─────────────────────────────────────────────

def replace_prompt_in_parameters(raw: str, prompt: str, negative: str) -> str:
    """infotext 의 프롬프트/네거티브만 바꾸고 파라미터 꼬리(Template 줄 포함)는 원문 그대로 둔다."""
    _old_prompt, _old_negative, tail = split_infotext(raw)
    return format_infotext(str(prompt or "").strip(), str(negative or "").strip(), tail)


def rewrite_png_parameters(path: Union[str, Path], text: str) -> None:
    """PNG 의 ``parameters`` 텍스트 청크만 교체해 원자적으로 저장한다.

    - 청크 단위로 고친다(core.png_chunks) — 픽셀을 다시 인코딩하지 않으므로 APNG 프레임·
      시간·반복(acTL/fcTL/fdAT), 16비트 깊이, gAMA·sRGB·cHRM·sBIT·bKGD·tIME·iCCP·eXIf·pHYs·
      tRNS, 다른 텍스트 청크(IDAT 뒤 청크 포함)가 **바이트 그대로** 남는다.
      예전 Pillow 재저장은 APNG 를 첫 프레임으로, 16비트를 8비트로 만들고 gAMA 등을 버렸다.
    - 같은 이름의 parameters 청크(tEXt/zTXt/iTXt, IDAT 뒤 것 포함)는 새 청크 하나로 합친다.
    - 같은 폴더 임시 파일에 쓴 뒤 os.replace — 실패하면 원본은 그대로다.
    실패는 예외로 알린다(PNG 가 아니거나 구조가 깨졌으면 ValueError — 호출자가 사용자 메시지로 바꾼다).
    """
    from core.png_chunks import replace_text_chunk

    target = Path(path)
    updated = replace_text_chunk(target.read_bytes(), "parameters", text)
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".exif_", suffix=".png", dir=str(target.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write(updated)
        os.replace(tmp_path, target)
        tmp_path = ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
# 이식
# ─────────────────────────────────────────────

def transplant(src: Union[str, Path],
               dst: Union[str, Path],
               out: Union[str, Path],
               include_workflow: bool = True) -> bool:
    """``src`` 이미지의 메타를 추출해서 ``dst`` 이미지에 이식 → ``out``에 저장.

    PNG로 저장 (원본 dst 포맷 무관 — 메타 손실 없도록).

    :param include_workflow: ComfyUI workflow/prompt 그래프까지 복사할지
    """
    meta = extract_from_file(src)
    if not meta.has_any():
        _logger.warning(f"이식 실패: src에 메타 없음: {src}")
        return False
    return embed_to_file(dst, meta, out, include_workflow=include_workflow)


def embed_to_file(src_image: Union[str, Path],
                  meta: ImageMetadata,
                  out: Union[str, Path],
                  include_workflow: bool = True) -> bool:
    """이미지에 메타를 박아 ``out``에 PNG로 저장.

    대상 픽셀은 그대로 둔다 — JPEG 회전 태그는 픽셀에 반영하고(PNG 에는 회전 태그를 쓰지
    않으므로), 투명도·ICC 프로파일·dpi 를 유지한다. 같은 폴더 임시 파일에 쓴 뒤 교체하므로
    ``out`` 이 대상 파일 자신이어도 중간에 깨진 파일이 남지 않는다.
    """
    tmp_path = ""
    try:
        from PIL import Image, ImageOps, PngImagePlugin
        with Image.open(src_image) as img:
            # 결과는 RGB(A) PNG 다 — CMYK(TIFF·JPEG)·그레이 원본의 ICC 는 RGB 픽셀에 붙이면
            # 색이 틀어지는 잘못된 조합이라 RGB 계열 원본의 프로파일만 옮긴다.
            icc_profile = img.info.get("icc_profile") if img.mode in _RGB_PROFILE_MODES else None
            dpi = img.info.get("dpi")
            has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
            upright = ImageOps.exif_transpose(img)
            img_copy = upright.convert("RGBA" if has_alpha else "RGB")
            # convert() 는 info 를 복사하고 PNG 저장은 info 의 icc_profile 을 그대로 쓴다 — 아래
            # save_kwargs 로만 넣도록 비운다.
            img_copy.info.pop("icc_profile", None)

        pnginfo = PngImagePlugin.PngInfo()

        # WebUI parameters
        # A normalized Comfy prompt is not an A1111 generation record. Adding
        # a synthetic parameters chunk would change source precedence on read.
        params_text = meta.raw_parameters or (
            "" if meta.source == MetadataSource.COMFYUI and include_workflow else _format_parameters(meta))
        if params_text:
            pnginfo.add_text("parameters", params_text)

        # ComfyUI workflow / prompt
        if include_workflow:
            if meta.raw_workflow:
                pnginfo.add_text("workflow", meta.raw_workflow)
            elif meta.workflow is not None:
                pnginfo.add_text("workflow", json.dumps(meta.workflow, ensure_ascii=False))
            if meta.raw_prompt:
                pnginfo.add_text("prompt", meta.raw_prompt)
            elif meta.prompt_graph is not None:
                pnginfo.add_text("prompt", json.dumps(meta.prompt_graph, ensure_ascii=False))

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"pnginfo": pnginfo}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        if dpi:
            save_kwargs["dpi"] = dpi
        fd, tmp_path = tempfile.mkstemp(prefix=".meta_", suffix=".png", dir=str(out_path.parent))
        os.close(fd)
        img_copy.save(tmp_path, format="PNG", **save_kwargs)
        os.replace(tmp_path, out_path)
        tmp_path = ""
        return True
    except Exception:
        _logger.exception(f"메타 이식 실패: {out}")
        return False
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def suggest_transplant_output(target: Union[str, Path]) -> str:
    """메타 이식 결과 기본 경로 — 대상 옆 ``<이름>_withmeta.png`` (이미 있으면 _2, _3 …)."""
    target_path = Path(target)
    base = target_path.with_name(f"{target_path.stem}_withmeta.png")
    candidate = base
    index = 2
    while candidate.exists():
        candidate = base.with_name(f"{target_path.stem}_withmeta_{index}.png")
        index += 1
    return str(candidate)


def _format_parameters(meta: ImageMetadata) -> str:
    """raw_parameters가 없을 때 정규화 필드로 WebUI 형식 재구성 (A1111 따옴표 규칙)."""
    if not (meta.prompt or meta.negative_prompt or meta.parameters):
        return ""
    return format_infotext(meta.prompt, meta.negative_prompt, format_parameters_line(meta.parameters))
