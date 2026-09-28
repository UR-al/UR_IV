# backends/webui_backend.py
"""WebUI (A1111/Forge) 백엔드 구현"""
import base64
import binascii
import copy
import io
import json
import logging
import threading
import time
import requests
from dataclasses import replace
from typing import Callable, Dict, Optional, Any
from PIL import Image, ImageOps

from backends.base import (
    AbstractBackend, BackendInfo, GenerationResult, MediaArtifact,
    ProgressCallback,
)
from core.http_retry import get_with_retry

logger = logging.getLogger(__name__)

_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}
_MAX_RESULT_ARTIFACTS = 64
_MAX_RESULT_BYTES = 256 * 1024 * 1024
_MAX_RESPONSE_BYTES = 384 * 1024 * 1024
# get_info 의 sd-models — 5xx(모델 로딩 중)면 이만큼 쉬고 딱 한 번 더 묻는다.
_TRANSIENT_STATUS = (500, 502, 503, 504)
_SD_MODELS_RETRY_PAUSE_S = 1.0


def _get_retrying_once(url: str, *, headers, timeout):
    """GET — 응답 지연(ReadTimeout)·5xx 만 **정확히 한 번** 더 시도한다(총 최대 2회).

    연결 거부·connect 타임아웃(``ConnectTimeout`` 도 ``ConnectionError``)은 곧바로 올린다 —
    꺼진 백엔드를 1+2+4초 백오프로 기다리면 '연결 실패' 통보만 7~15초 늦는다. 두 번째 시도의
    실패도 그대로 올린다. 공유 http_retry(get_with_retry) 는 retries+1 회를 시도하므로 여기에는
    쓰지 않는다(첫 요청 + retries=1 이면 3회였다).
    """
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise
    except requests.exceptions.Timeout as exc:
        logger.warning("GET %s 응답 지연(%s) — 한 번 더 시도", url, type(exc).__name__)
        return requests.get(url, headers=headers, timeout=timeout)
    if response.status_code in _TRANSIENT_STATUS:
        logger.warning("GET %s → %d — %.1f초 뒤 한 번 더 시도",
                       url, response.status_code, _SD_MODELS_RETRY_PAUSE_S)
        time.sleep(_SD_MODELS_RETRY_PAUSE_S)
        return requests.get(url, headers=headers, timeout=timeout)
    return response


def _bounded_response_json(response):
    """Parse a real streaming response without accepting an unbounded body."""

    headers = getattr(response, "headers", {}) or {}
    try:
        declared_length = int(headers.get("Content-Length", "0") or 0)
    except (TypeError, ValueError):
        declared_length = 0
    if declared_length > _MAX_RESPONSE_BYTES:
        raise ValueError("WebUI 응답이 허용된 384MiB를 초과합니다.")
    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        # Lightweight test doubles and legacy response wrappers.
        return response.json()
    body = bytearray()
    for chunk in iterator(chunk_size=1024 * 1024):
        if not chunk:
            continue
        if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
            raise ValueError("WebUI 응답이 허용된 384MiB를 초과합니다.")
        body.extend(chunk)
    return json.loads(body.decode("utf-8"))


def _decode_webui_image(value: str, index: int) -> MediaArtifact:
    """Decode one A1111 image field without trusting a data-URI MIME value."""
    if not isinstance(value, str):
        raise ValueError(f"images[{index}]가 base64 문자열이 아닙니다.")

    encoded = value.strip()
    if encoded[:5].lower() == "data:":
        header, separator, encoded = encoded.partition(",")
        if not separator:
            raise ValueError(f"images[{index}]의 data URI가 올바르지 않습니다.")
        media_parts = header[5:].split(";")
        candidate = media_parts[0].strip().lower()
        parameters = {part.strip().lower() for part in media_parts[1:]}
        if "base64" not in parameters:
            raise ValueError(f"images[{index}]의 data URI는 base64 형식이어야 합니다.")
        if candidate:
            if candidate not in _IMAGE_EXTENSIONS:
                raise ValueError(f"images[{index}]의 MIME 형식이 올바르지 않습니다.")

    compact = "".join(encoded.split())
    if not compact:
        raise ValueError(f"images[{index}]가 비어 있습니다.")
    compact += "=" * (-len(compact) % 4)
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"images[{index}]의 base64 데이터가 올바르지 않습니다.") from exc
    if not data:
        raise ValueError(f"images[{index}]의 이미지 데이터가 비어 있습니다.")

    detected_mime = None
    is_animated = False
    try:
        with Image.open(io.BytesIO(data)) as image:
            detected_mime = Image.MIME.get(image.format)
            is_animated = bool(getattr(image, "is_animated", False))
    except (OSError, ValueError) as exc:
        raise ValueError(f"images[{index}]가 지원되는 raster 이미지가 아닙니다.") from exc

    if detected_mime not in _IMAGE_EXTENSIONS:
        raise ValueError(f"images[{index}]의 이미지 형식을 지원하지 않습니다.")
    mime = detected_mime
    extension = _IMAGE_EXTENSIONS.get(mime, ".img")
    return MediaArtifact(
        kind="animated" if is_animated or mime == "image/gif" else "image",
        data=data,
        filename=f"image_{index + 1:03d}{extension}",
        mime=mime,
        metadata={"source_index": index},
    )


def _parse_generation_info(raw_info) -> Dict:
    """Forge 응답 'info'(JSON 문자열 또는 dict) → dict. 읽을 수 없으면 {'raw_info': …}."""
    if isinstance(raw_info, dict):
        return copy.deepcopy(raw_info)
    if isinstance(raw_info, str):
        try:
            parsed = json.loads(raw_info) if raw_info else {}
        except json.JSONDecodeError:
            return {"raw_info": raw_info}
        return parsed if isinstance(parsed, dict) else {"raw_info": parsed}
    return {"raw_info": raw_info}


def _read_error_body(response):
    """거절 응답의 본문(JSON, 아니면 text)을 읽고 응답을 닫는다."""
    try:
        return response.json()
    except Exception:
        return getattr(response, "text", "")
    finally:
        close_response = getattr(response, "close", None)
        if callable(close_response):
            close_response()


def _read_retryable(response, parts=None) -> tuple:
    """Forge 가 거절했나 → (상태, 본문). 422(스크립트 거절 — 설명·재시도 판정)는 늘, 500 은 앱이 Forge 옵션을 넣었을
    때만(``parts.option_keys`` — P10 KeyError·설정 잠금 판정) 본문을 읽고 응답을 닫는다. 그 밖은 (None, None) 이고 응답은
    그대로다(raise_for_status·본문 읽기는 호출자 — 앱 옵션이 없는 500 은 지금과 같은 오류)."""
    status = getattr(response, "status_code", None)
    if status == 422 or (status == 500 and parts is not None and getattr(parts, "option_keys", ())):
        return status, _read_error_body(response)
    return None, None


def _cancelled(cancel_check: Optional[Callable[[], bool]]) -> bool:
    """중지 요청이 왔나 — ``cancel_check`` 가 없으면 False."""
    return cancel_check is not None and bool(cancel_check())


def _explain_rejection(body, payload: Dict) -> str:
    """Forge 가 요청을 거절(HTTP 422 — 확장 스크립트 없음 등)했으면 어느 확장·기능 때문인지 설명(P4, core/sam_extra_notices)."""
    from core.sam_extra_notices import explain_rejected_request
    message = explain_rejected_request(422, body, payload)
    logger.warning("WebUI 요청 거절(422): %s", message)
    return message


def _extension_notices(api_url: str, info: Dict, payload: Dict, *, propagated_titles=()) -> list:
    """Forge info + 보낸 요청 → sam-extra 결과 알림(SAM3 Error·Anima38 off·PAG 누락). 실패하면 [].

    ``propagated_titles``: 보조 패스가 메인 생성 설정에서 전달한 제목 — 그 기능 알림에 '(전달됨)' 을 붙인다(P7)."""
    try:
        from core.sam_extra_notices import result_notices
        from core.sam_extra_probe import peek_capabilities
        return result_notices(info, payload, capabilities=peek_capabilities(api_url),   # 캐시만 — HTTP 없음
                              propagated_titles=propagated_titles)
    except Exception:
        logger.debug("sam-extra 결과 알림 계산 실패(무시)", exc_info=True)
        return []


def _take_envelope(settings):
    """워커 settings → (봉투를 뺀 사본, 메인 생성 샘플링 블록 봉투|None) — core/alwayson_propagation."""
    from core.alwayson_propagation import take_envelope
    return take_envelope(settings)


def _forge_init_size(image_b64: str) -> tuple:
    """Forge 가 API init 이미지로 쓰는 (width, height) — 단독 경로 payload 의 width/height.

    Forge 는 base64 init 이미지를 decode_base64_to_image → images.read → fix_image 로 읽고, fix_image 가
    ``ImageOps.exif_transpose`` 로 EXIF 방향부터 적용한다(실패하면 그대로 쓴다). 워커는 파일 바이트를 그대로 보내므로
    방향 적용 전 크기를 보내면, 방향 태그로 세운 사진(휴대폰·카메라 JPEG)을 부모 패스가 가로·세로 바뀐 크기로 늘려
    이미지 전체가 찌그러지고 원본 기준 SAM3 도 'size WxH != HxW' 로 폴백한다. 그래서 Forge 와 같게 세워서 잰다.
    """
    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as image:
        try:
            upright = ImageOps.exif_transpose(image)
        except Exception:
            upright = image          # Forge fix_image 도 방향 적용이 실패하면 그대로 쓴다
        return upright.size


class WebUIBackend(AbstractBackend):
    """Stable Diffusion WebUI API 백엔드"""

    def __init__(self, api_url: str):
        super().__init__(api_url)
        self._generation_state_lock = threading.Lock()
        self._generation_inflight = False
        # force_task_id → interrupt 요청 이벤트. interrupt() 는 여기에 표시만 하고, 요청별
        # 감시 스레드가 '우리 작업이 active 일 때만' 전역 interrupt 를 보낸다 (core/webui_cancel.py).
        self._inflight_interrupts: Dict[str, threading.Event] = {}
        # /internal/progress 가 없는 서버(--nowebui)면 False — 매 폴링마다 404 를 다시 치지 않는다.
        self._task_progress_supported: Optional[bool] = None

    @staticmethod
    def _setting_number(settings: Dict, key: str, default, cast):
        """settings 의 숫자 값 — 비었거나 잘못된 값이면 기본값(빈 문자열 → ValueError 방지)."""
        value = settings.get(key, default)
        try:
            return cast(float(value)) if cast is int else cast(value)
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _build_postprocess_payload(image_b64: str, settings: Dict, *, prompt: str, negative_prompt: str) -> Dict:
        """단독 ADetailer img2img payload.

        입력 원본 크기와 샘플링 파라미터(steps/cfg/seed/sampler/scheduler)를 명시한다 —
        ADetailer 의 skip_img2img 는 이 값들을 ``_ad_orig`` 로 보관해 인페인트 패스에 쓰므로,
        안 보내면 Forge API 기본값(steps 50·기본 sampler)이 그대로 들어간다.
        (예전엔 ``_postprocess_base_payload`` 를 복사·정리하는 코드가 있었지만 그 값을 채우는
        호출자가 없어 늘 빈 dict 였다.)
        """
        init_width, init_height = _forge_init_size(image_b64)   # Forge 가 EXIF 방향을 적용한 크기

        number = WebUIBackend._setting_number
        payload = {
            "init_images": [image_b64],
            "resize_mode": 0,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "send_images": True,
            "save_images": False,
            "width": init_width,
            "height": init_height,
            # skip_img2img 면 부모 패스(1스텝·128×128) 결과는 버려지므로 값 자체는 결과에 영향이 없다.
            "denoising_strength": number(settings, 'denoising_strength', 0.1, float),
            "steps": number(settings, 'steps', 28, int),
            "cfg_scale": number(settings, 'cfg_scale', 7.0, float),
            "seed": number(settings, 'seed', -1, int),
            "alwayson_scripts": {},
        }
        sampler = str(settings.get('sampler') or '').strip()
        if sampler and sampler != 'Use same sampler':
            payload["sampler_name"] = sampler
        scheduler = str(settings.get('scheduler') or '').strip()
        if scheduler and scheduler != 'Use same scheduler':
            payload["scheduler"] = scheduler
        return payload

    @staticmethod
    def _build_sam3_script_state(settings: Dict, capabilities=None) -> Dict:
        """SAM3 alwayson state — core/sam3_args로 일원화 (t2i 경로와 동일 로직).

        예전에는 이 함수와 ui/generator_generation.py에 거의 같은 dict가 복사돼 있어
        한쪽만 고치면 t2i와 배치 결과가 갈렸다. 기본값·범위·ControlNet 필드는 전부
        core/sam3_args.SAM3_SPEC 한 곳에서 관리한다.

        ★ sam3_unload_after: 확장 UI 기본은 True지만 API 경로는 `_xyz_or(..., False)`라
          명시하지 않으면 인페인트 내내 SAM3(~3.5GB)가 상주해 16GB GPU에서 OOM 난다.
          SAM3_SPEC의 기본값이 True이므로 명시 전송된다.

        capabilities: sam-extra 기능 스냅샷 — CN 전처리기·모델 이름을 라이브 목록 표기로 맞춘다(P3).
        """
        from core import sam3_args
        # 예전 호출자가 denoising_strength만 넘기던 경우 호환
        if 'sam3_denoising_strength' not in settings and 'denoising_strength' in settings:
            settings = dict(settings)
            settings['sam3_denoising_strength'] = settings['denoising_strength']
        return sam3_args.build_state(
            settings,
            prompt=str(settings.get('prompt', '') or ''),
            negative_prompt=str(settings.get('negative_prompt', '') or ''),
            capabilities=capabilities,
        )

    def _sam_extra_snapshot(self):
        """이 WebUI 의 sam-extra 기능 스냅샷 — 연결 때 받아 둔 캐시만 본다(네트워크 없음). 없으면 None."""
        try:
            from core.sam_extra_probe import peek_capabilities
            return peek_capabilities(self.api_url)
        except Exception as exc:
            logger.warning("sam-extra 스냅샷 조회 실패(무시): %s", exc)
            return None

    def _run_img2img_postprocess(self, image_b64: str, payload: Dict, *, parts=None, extra_notices=(),
                                 cancel_check: Optional[Callable[[], bool]] = None, context: str = "") -> str:
        """단독 ADetailer·SAM3·Refine 의 img2img. 결과는 base64 str 이다 — Forge info 와 sam-extra 알림
        ('SAM3 Error' 등)을 붙인 ``NoticedImage`` 라 워커가 ``notices_of(result)`` 로 볼 수 있다(P4).

        ``parts``(core/forge_optional_parts.OptionalParts): 앱이 스스로 덧붙인 부분 — P7 전달 블록과, 여기서 싣는 앱의
        Forge 옵션 덮어쓰기(P10 ``_forge_option_parts``, 기본 'Forge 설정 따름'이면 없음). Forge 가 그중 하나를 거절하면
        (422 제목·500 옵션 KeyError·설정 잠금) 그것만 빼고 다시 보낸다(부분이 줄어드는 동안만, 시도 수 상한
        ``max_attempts``). 다시 보내기 전에 ``cancel_check`` 를 본다 — 스냅샷 새로고침 앞과 **뒤** 둘 다(새로고침은 블로킹
        GET 여러 개라 그 사이에 중지할 수 있다. 이 경로는 Forge 를 끊지 않으므로 다시 보내면 중지 뒤에도 끝까지 돈다 —
        P10 검토 1). 취소됐으면 원래 거절로 실패한다. 재시도 때는 기능 스냅샷 캐시를 버리지 않고 이 워커에서 새로 받는다
        (다음 요청이 새 스냅샷으로 게이트한다 — critic A1).
        ``extra_notices``: 요청 전에 생긴 알림(전달하지 않은 블록 등). ``context``: 알림 문구의 작업 이름(보조 패스 id).
        """
        from core.forge_optional_parts import OptionalParts, max_attempts, plan_retry
        parts = parts if parts is not None else OptionalParts()
        notices = list(extra_notices or ())
        payload, option_keys, option_notices = self._forge_option_parts(payload)
        if option_keys:
            parts = replace(parts, option_keys=option_keys)
        notices.extend(option_notices)
        attempts = max_attempts(parts)
        for attempt in range(1, attempts + 1):
            response = requests.post(
                f'{self.api_url}/sdapi/v1/img2img',
                json=payload, headers=_HEADERS, timeout=600
            )
            status, body = _read_retryable(response, parts)
            if status is not None:
                retry = plan_retry(status, body, payload, parts) if attempt < attempts else None
                self._remember_frozen(retry)
                if retry is not None and not _cancelled(cancel_check):
                    retry_notices = self._optional_parts_retry_notices(retry, context)   # 스냅샷 새로고침(블로킹 GET)
                    if not _cancelled(cancel_check):   # 새로고침 중에 중지됐으면 다시 보내지 않는다(B10)
                        notices.extend(retry_notices)
                        payload, parts = retry.payload, retry.remaining
                        continue
                if status == 422:
                    raise RuntimeError(_explain_rejection(body, payload))
            response.raise_for_status()
            r = response.json()
            if 'images' in r and r['images']:
                image = r['images'][-1]
                if not isinstance(image, str):
                    return image
                from core.sam_extra_notices import NoticedImage
                info = _parse_generation_info(r.get('info', {}))
                return NoticedImage(image, info, [
                    *_extension_notices(self.api_url, info, payload, propagated_titles=parts.titles), *notices])
            raise RuntimeError("img2img 후처리 API 응답에 이미지가 없습니다.")
        raise RuntimeError("img2img 후처리 재시도 한도를 넘었습니다.")   # 도달하지 않는다(마지막 시도는 재시도하지 않음)

    def _optional_parts_retry_notices(self, retry, context: str, *, main: bool = False, provenance=None) -> list:
        """거절된 부분을 빼고 다시 보낼 때: 알림 + 기능 스냅샷 새로 받기(이 워커 스레드에서, 캐시 무효화 없이 — critic A1).

        ``main``: 메인 생성(``_generate``) — 블록 출처(``provenance`` — 요청의 비공개 키)가 앱 기본값이면 정보
        (``app_block_retried_notice``), 사용자 값·모르면 경고(``block_retried_notice``, P10 검토 2). 아니면 보조 패스
        ``context``. 재시도 알림 코드는 모두 ``REFRESH_CAPABILITIES_CODES`` 라 GUI 가 띄울 때 GUI 스냅샷도 다시 받는다."""
        from core import sam_extra_notices as sn
        from core.alwayson_propagation import aux_label
        from core.forge_optional_parts import REASON_FROZEN
        logger.warning("Forge 가 앱이 덧붙인 부분을 거절 — 빼고 다시 보냄: %s", ", ".join(retry.removed))
        # 설정 잠금은 Forge 시작 인자라 스냅샷에 드러나지 않는다 — 새로 받아도 소용없다(기억은 _remember_frozen, P10 검토 3)
        if retry.reason != REASON_FROZEN:
            try:
                from core.sam_extra_probe import get_capabilities
                get_capabilities(self.api_url, refresh=True)   # 워커 전용(블로킹 GET). 배치의 다음 장은 새 스냅샷으로 게이트
            except Exception:
                logger.debug("sam-extra 스냅샷 새로고침 실패(무시)", exc_info=True)
        where = "" if main else aux_label(context)
        if retry.code == sn.CODE_FORGE_OPTION_REJECTED:
            from core.forge_override_settings import label_of
            frozen = retry.reason == REASON_FROZEN
            # 잠금이면 거절된 것 = Forge 가 이름을 댄(기억할) 키. 함께 뺀 나머지는 이번 한 번만 빠졌다(P10 검토 4)
            named = retry.frozen if frozen else retry.removed
            also = [label_of(key) for key in retry.removed if key not in named]
            return [sn.forge_option_rejected_notice([label_of(key) for key in named], frozen=frozen, where=where,
                                                    also=also)]
        if retry.code != sn.CODE_PROPAGATION_RETRIED:
            return []
        if main:
            from core.alwayson_propagation import PROVENANCE_APP_DEFAULT, provenance_of
            return [sn.app_block_retried_notice(title) if provenance_of(provenance, title) == PROVENANCE_APP_DEFAULT
                    else sn.block_retried_notice(title) for title in retry.removed]
        return [sn.propagation_retried_notice(title, where) for title in retry.removed]

    def _remember_frozen(self, retry) -> None:
        """Forge 가 설정 잠금(--freeze-settings…)으로 앱 옵션을 거절했으면 이 주소에 기억한다 — 다음 요청부터 보내지 않는다.

        잠금은 스냅샷을 새로 받아도 드러나지 않아(옵션은 '있다'로 남는다) 기억이 없으면 요청마다 거절·재시도를
        되풀이한다(P10 검토 3). 기억하는 것은 Forge 가 잠겼다고 이름을 댄 키(``retry.frozen``)뿐이다 — 섹션·키 잠금은
        일부만 잠그므로 재시도에서 함께 뺀 나머지는 다음 요청부터 다시 보낸다(전역 잠금만 앱 키 전부, P10 검토 4).
        취소로 다시 보내지 않더라도 기억한다 — Forge 의 사실이다. 백엔드 변경·재연결과 수동 새로고침이 잊게 한다
        (ui/sam_extra_capabilities_actions)."""
        from core.forge_optional_parts import REASON_FROZEN
        if retry is None or retry.reason != REASON_FROZEN or not retry.frozen:
            return
        from core.forge_override_settings import remember_frozen_options
        remember_frozen_options(self.api_url, retry.frozen)
        logger.warning("Forge 설정 잠금 — 앞으로 이 Forge 에는 보내지 않음: %s", ", ".join(retry.frozen))

    def _forge_option_parts(self, payload: Dict) -> tuple:
        """앱의 Forge 옵션 덮어쓰기(P10, core/forge_override_settings)를 요청에 싣는다 → (요청 payload, 넣은 키, 알림).

        설정(GUI 가 밀어 넣은 ``forgeOptionOverrides``)이 비면 — 기본 'Forge 설정 따름'(D3) — payload 객체를 그대로
        돌려준다: 요청은 P10 전과 바이트 단위로 같다. 키마다 요청 직전의 스냅샷(peek — 캐시만, HTTP 없음)이 '있다'고 한
        것만 넣는다(모르는 키는 요청 전체가 500). 이 Forge 가 설정 잠금으로 거절했던 키(``_remember_frozen``)는 넣지 않는다.
        알림(없음·확인 못 함·잠금)은 결과 info 에 실린다."""
        try:
            from core import forge_override_settings as fos
            overrides = fos.forge_option_overrides_setting()
            if not overrides:
                return payload, (), []
            plan = fos.plan_overrides(overrides, self._sam_extra_snapshot(),
                                      frozen=fos.frozen_options(self.api_url))
            merged, sent = fos.merge_into_payload(payload, plan)
            if sent:
                logger.info("Forge 옵션 덮어쓰기(이 요청만): %s",
                            ", ".join(f"{key}={merged['override_settings'][key]}" for key in sent))
            return merged, sent, fos.plan_notices(plan)
        except Exception:
            logger.debug("Forge 옵션 덮어쓰기 계산 실패(무시하고 그대로 보냄)", exc_info=True)
            return payload, (), []

    def _propagate(self, payload: Dict, envelope: Dict, aux: str, settings: Dict) -> tuple:
        """메인 생성의 샘플링 블록 봉투를 이 보조 요청에 싣는다 → (OptionalParts, 알림).

        봉투는 클릭한 순간의 T2I 패널(ui/aux_pass_snapshot). 여기서 패스·백엔드 규칙과 조건(DD Hires Pass,
        SAM3 Mask only)을 적용하고, 요청 직전의 스냅샷(peek — 캐시만)으로 게이트한 뒤 setdefault 로 넣는다 —
        보조 경로 자신의 SAM3 Mask·ADetailer 는 덮지 않는다. 봉투 키는 요청 JSON 에 들어가지 않는다.
        """
        from core import alwayson_propagation as ap
        from core.forge_optional_parts import OptionalParts
        blocks = ap.blocks_for(envelope, aux, backend=ap.BACKEND_WEBUI, aux_settings=settings)
        if ap.has_model_bound(blocks):
            # (critic A7) Anima38 블록은 클릭한 순간의 T2I 모델 종류로 만들었다. 보조 요청은 체크포인트를 지정하지 않아
            # Forge 에 지금 걸린 체크포인트로 돈다 — 그것이 봉투 모델과 같을 때만 보낸다(모르거나 다르면 빼고 정보 로그).
            actual = self._active_checkpoint()
            blocks, dropped = ap.drop_model_bound(blocks, envelope_model=envelope.get("model"), actual_model=actual)
            if dropped:
                logger.info("%s: 실제 모델(%s)이 메인 생성 모델(%s)과 같다고 확인하지 못해 전달하지 않음 — %s",
                            ap.aux_label(aux), actual or "모름", envelope.get("model") or "모름", ", ".join(dropped))
        provenance = ap.envelope_provenance(envelope)
        result = ap.gate(blocks, self._sam_extra_snapshot(), img2img=True, provenance=provenance)
        inserted = ap.apply(payload, result.kept)
        if inserted:
            logger.info("%s: 메인 생성 설정 전달 — %s", ap.aux_label(aux), ", ".join(inserted))
        notices = ap.gate_notices(result, provenance, img2img=True, aux=aux)
        return OptionalParts(titles=inserted), tuple(notices)

    def _active_checkpoint(self) -> str:
        """Forge 에 지금 걸린 체크포인트(``options.sd_model_checkpoint``) — 보조 요청(Refine·SAM3·ADetailer)은 체크포인트를
        지정하지 않아 이 모델로 돈다(critic A7). 메인 생성 전에 ``_switch_model_if_needed`` 가 부르는 것과 같은 GET 이고,
        봉투에 모델에 묶인 블록(Anima38)이 있을 때만 부른다(워커 스레드). 실패하면 '' (모름 → 그 블록은 빠진다)."""
        try:
            response = requests.get(url=f'{self.api_url}/sdapi/v1/options', headers=_HEADERS, timeout=10)
            response.raise_for_status()
            options = response.json()
            return str(options.get('sd_model_checkpoint') or '') if isinstance(options, dict) else ''
        except Exception as e:
            logger.info("Forge 현재 체크포인트 확인 실패(모델에 묶인 전달 블록은 빼고 보냄): %s", e)
            return ''

    def _run_aux_postprocess(self, image_b64: str, payload: Dict, envelope, aux: str, settings: Dict,
                             cancel_check: Optional[Callable[[], bool]]) -> str:
        """보조 패스 공통 송신. 봉투도 취소 확인도 없으면 예전과 같은 2-인자 호출(테스트 더블 호환)."""
        if envelope is None and cancel_check is None:
            return self._run_img2img_postprocess(image_b64, payload)
        parts, notices = self._propagate(payload, envelope, aux, settings) if envelope is not None else (None, ())
        return self._run_img2img_postprocess(image_b64, payload, parts=parts, extra_notices=notices,
                                             cancel_check=cancel_check, context=aux)

    def get_lora_manager_url(self) -> Dict:
        """sam-extra의 임베드된 LoRA Manager 주소를 얻는다.

        확장이 Forge FastAPI에 등록해 둔 라우트(`sam3ext/lora_manager_core.py`):
            GET /sam3-lora/spawn  → {"url", "port", "status", "message"}
        서버가 안 떠 있으면 이 호출이 띄운다(lazy spawn). 확장이 없으면 404 → 안내 메시지.
        메모·Tile & Repair 라우트처럼 같은 출처 헤더(X-SAM3-Notebook: 1)가 없으면 403 이다 — 헤더를
        보지 않는 옛 확장에 보내도 무해하다. 401·403(로그인·헤더 거절)은 URL 을 싣지 않은 안내로 바꾼다.

        반환: {'url': str, 'status': str, 'message': str}
        """
        from core.sam_extra_capabilities import NOTEBOOK_HEADERS
        try:
            r = requests.get(f'{self.api_url}/sam3-lora/spawn',
                             headers={**_HEADERS, **NOTEBOOK_HEADERS}, timeout=20)
            if r.status_code == 404:
                return {'url': '', 'status': 'missing',
                        'message': 'sam-extra 확장의 LoRA Manager를 찾을 수 없습니다 '
                                   '(확장 미설치이거나 v0.9.0 미만)'}
            if r.status_code in (401, 403):
                return {'url': '', 'status': 'refused',
                        'message': 'Forge 가 LoRA Manager 요청을 거부했습니다 — 로그인 설정(--gradio-auth·--api-auth)이나 '
                                   'sam-extra 버전을 확인하세요'}
            r.raise_for_status()
            data = r.json()
            return {
                'url': str(data.get('url') or ''),
                'status': str(data.get('status') or ''),
                'message': str(data.get('message') or ''),
            }
        except Exception as e:
            logger.warning("LoRA Manager URL 조회 실패: %s", e)
            return {'url': '', 'status': 'error', 'message': str(e)}

    def get_backend_type(self) -> str:
        return "webui"

    def interrupt(self):
        """진행 중인 **이 어댑터의** 생성 중단 요청 (best-effort, 실패 무시).

        전역 ``/sdapi/v1/interrupt`` 를 바로 보내지 않는다 — 우리 요청이 외부 Forge 작업 뒤에
        줄 서 있으면 남의 작업을 끊기 때문이다. 요청별 감시 스레드가 우리 작업이 실제로
        돌기 시작한(active) 뒤에만 보낸다. 진행 중인 요청이 없으면 아무것도 하지 않는다."""
        with self._generation_state_lock:
            pending = list(self._inflight_interrupts.values())
        for requested in pending:
            requested.set()

    def _task_state(self, task_id: str) -> str:
        """``POST /internal/progress`` 로 우리 작업의 상태 (core.webui_cancel.TASK_*).

        '엔드포인트 없음'(404/405, 처음부터 모르는 응답 모양 → TASK_UNKNOWN, 예전 방식 폴백)과
        '일시 실패'(타임아웃·연결 끊김·5xx, 정상 응답을 받은 뒤의 이상한 응답 → TASK_TRANSIENT)를
        나눈다. 예전엔 한 번의 일시 실패도 UNKNOWN 이라, 우리 요청이 외부 작업 뒤에 줄 서 있는
        동안 전역 interrupt 가 나가 남의 작업을 끊었다. TRANSIENT 처리는 TaskInterruptPolicy.
        """
        from core.webui_cancel import TASK_TRANSIENT, TASK_UNKNOWN, parse_task_state
        if self._task_progress_supported is False:
            return TASK_UNKNOWN
        try:
            response = requests.post(
                f'{self.api_url}/internal/progress',
                json={'id_task': task_id, 'id_live_preview': -1, 'live_preview': False},
                headers=_HEADERS, timeout=3,
            )
            if response.status_code in (404, 405):
                self._task_progress_supported = False
                return TASK_UNKNOWN
            response.raise_for_status()
            state = parse_task_state(response.json())
        except Exception as e:
            logger.debug("task progress 조회 실패(일시): %s", e)
            return TASK_TRANSIENT
        if state != TASK_UNKNOWN:
            self._task_progress_supported = True
            return state
        # 200 인데 모르는 모양 — 전에 정상 응답을 받았으면 일시 이상, 아니면 지원 안 하는 서버
        return TASK_TRANSIENT if self._task_progress_supported else TASK_UNKNOWN

    def _watch_cancellation(self, task_id: str, stop_event: threading.Event,
                            interrupt_requested: threading.Event,
                            cancel_check: Optional[Callable[[], bool]],
                            unseen_grace: float) -> None:
        """취소(cancel_check 또는 interrupt())가 오면, 우리 작업이 도는 동안 interrupt 를 반복한다.

        Forge 는 ``state.begin()`` 에서 interrupted 플래그를 지우므로 시작 전에 보낸 한 번은
        사라진다 — active 가 된 뒤에 보내고, HTTP 가 끝날 때까지 반복한다. queued(남의 작업이
        도는 중)·completed 에는 보내지 않는다. 판단은 core.webui_cancel.TaskInterruptPolicy.
        ``unseen_grace`` 는 본문 크기에 맞춘 unseen 유예(core.webui_cancel.unseen_grace_seconds).
        """
        from core.webui_cancel import TaskInterruptPolicy
        while not stop_event.wait(0.05):
            if interrupt_requested.is_set():
                break
            try:
                if cancel_check is not None and cancel_check():
                    break
            except Exception:
                logger.debug("cancel_check 실패(무시)", exc_info=True)
        else:
            return   # 취소 없이 요청이 끝났다
        policy = TaskInterruptPolicy(grace_seconds=unseen_grace)
        while not stop_event.is_set():
            state = self._task_state(task_id)
            if stop_event.is_set():
                return
            if policy.should_interrupt(state, time.monotonic()):
                self._post_interrupt()
            if stop_event.wait(0.25):
                return

    def _post_interrupt(self):
        try:
            requests.post(f'{self.api_url}/sdapi/v1/interrupt',
                          headers=_HEADERS, timeout=5)
            logger.info("interrupt 요청 전송")
        except Exception as e:
            logger.debug(f"interrupt 실패(무시): {e}")

    def test_connection(self) -> bool:
        """WebUI 연결 상태 확인"""
        try:
            r = requests.get(
                f'{self.api_url}/sdapi/v1/options',
                timeout=3
            )
            r.raise_for_status()
            return True
        except Exception:
            return False

    def get_info(self) -> BackendInfo:
        """WebUI API에서 모델, 샘플러 등 정보 가져오기"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from core.backend_probe import request_timeout
        headers = {"accept": "application/json"}
        # 루프백이면 connect 만 0.3초 — 꺼진 로컬 백엔드를 약 2~4초 대신 곧바로 알아챈다.
        timeout = request_timeout(self.api_url, 5)
        info = BackendInfo()

        # 모델 목록 (필수). 연결 자체가 거부되면(백엔드가 꺼짐) 재시도하지 않는다 —
        # 1+2+4초 백오프는 '연결 실패' 통보만 7~15초 늦췄다. 응답 지연(ReadTimeout)이나
        # 5xx(모델 로딩 중)만 딱 한 번 더 시도한다(_get_retrying_once, 총 최대 2회).
        # 공유 http_retry 정책은 건드리지 않는다.
        url = f'{self.api_url}/sdapi/v1/sd-models'
        res = _get_retrying_once(url, headers=headers, timeout=timeout)
        res.raise_for_status()
        sd_models = res.json()
        if isinstance(sd_models, list):
            info.models = [m.get('title', '') for m in sd_models]
            info.checkpoints = ["Use same checkpoint"] + [
                m.get('model_name', '') for m in sd_models
            ]

        def _fetch(endpoint):
            response = get_with_retry(
                f'{self.api_url}{endpoint}',
                headers=headers, timeout=timeout, retries=2,
            )
            # 404(구 A1111 의 /schedulers 등)·5xx 본문을 목록으로 오인하지 않는다.
            response.raise_for_status()
            return response.json()

        # 나머지 5개 병렬 호출
        tasks = {
            'samplers': '/sdapi/v1/samplers',
            'schedulers': '/sdapi/v1/schedulers',
            'upscalers': '/sdapi/v1/upscalers',
            'vae': '/sdapi/v1/sd-vae',
            'options': '/sdapi/v1/options',
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_fetch, ep): name for name, ep in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    r = future.result()
                    if name == 'samplers' and isinstance(r, list):
                        info.samplers = [s.get('name', '') for s in r]
                    elif name == 'schedulers' and isinstance(r, list):
                        info.schedulers = [s.get('name', '') for s in r] or ["Automatic"]
                    elif name == 'upscalers' and isinstance(r, list):
                        info.upscalers = [u.get('name', '') for u in r]
                    elif name == 'vae' and isinstance(r, list):
                        info.vae = ["Use same VAE"] + [v.get('model_name', '') for v in r]
                    elif name == 'options':
                        info.options = r
                except Exception as e:
                    logger.warning("get_info endpoint '%s' failed: %s", name, e)

        # 구 A1111 은 /sdapi/v1/schedulers 가 없다 — 빈 콤보 대신 WebUI 기본값을 둔다.
        if not info.schedulers:
            info.schedulers = ["Automatic"]
        return info

    def get_system_stats(self) -> dict:
        """GPU/VRAM 상태 조회"""
        try:
            r = requests.get(f'{self.api_url}/sdapi/v1/memory', timeout=3)
            if r.status_code == 200:
                data = r.json()
                cuda = data.get('cuda', {})
                sys_info = cuda.get('system', {})
                return {
                    'vram_used': sys_info.get('used', 0),
                    'vram_total': sys_info.get('total', 0),
                    'vram_free': sys_info.get('free', 0),
                }
        except Exception as e:
            logger.warning("get_system_stats failed: %s", e)
        return {}

    def get_loras(self) -> list:
        """WebUI LoRA 목록 반환 (트리거 워드 포함)"""
        try:
            r = requests.get(
                f'{self.api_url}/sdapi/v1/loras',
                headers=_HEADERS, timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                result = []
                for item in data:
                    lora = {
                        'name': item.get('name', ''),
                        'alias': item.get('alias', item.get('name', '')),
                        'path': item.get('path', ''),
                        'trigger_words': [],
                    }
                    metadata = item.get('metadata', {})
                    if metadata:
                        # Method 1: activation text (CivitAI / user-defined)
                        act_text = (metadata.get('activation text', '')
                                    or metadata.get('ss_activation_text', ''))
                        if act_text:
                            lora['trigger_words'] = [
                                t.strip() for t in act_text.split(',') if t.strip()
                            ][:8]
                        # Method 2: ss_tag_frequency (trained tags — fallback)
                        if not lora['trigger_words']:
                            tag_freq = metadata.get('ss_tag_frequency', {})
                            if isinstance(tag_freq, str):
                                try:
                                    import json as _json
                                    tag_freq = _json.loads(tag_freq)
                                except Exception:
                                    tag_freq = {}
                            if isinstance(tag_freq, dict):
                                all_tags = {}
                                for ds_tags in tag_freq.values():
                                    if isinstance(ds_tags, dict):
                                        all_tags.update(ds_tags)
                                if all_tags:
                                    sorted_tags = sorted(
                                        all_tags.items(),
                                        key=lambda x: x[1], reverse=True
                                    )
                                    lora['trigger_words'] = [
                                        t[0] for t in sorted_tags[:5]
                                    ]
                    result.append(lora)
                return result
        except Exception as e:
            logger.warning("get_loras failed: %s", e)
        return []

    @staticmethod
    def _module_names(values) -> list[str]:
        """옵션의 전체 경로든 요청의 파일명이든 파일명만 — 비교는 이걸로 한다."""
        names = []
        for value in values or []:
            if isinstance(value, str) and value.strip():
                names.append(value.replace('\\', '/').rsplit('/', 1)[-1].strip())
        return sorted(names)

    def _desired_forge_modules(self, current_options: dict, modules) -> Optional[list[str]]:
        """옵션에 넣어야 할 모듈 목록. 바꿀 필요가 없으면 None.

        Forge classic 은 요청 본문의 ``forge_additional_modules`` 를 **읽지 않는다** —
        ``shared.opts.forge_additional_modules`` 만 본다(processing.py / main_entry.py).
        그래서 본문에만 넣으면 Anima 처럼 VAE·TE 가 따로인 모델은
        "You do not have VAE state dict!" 로 500 이 난다. 옵션 API 는 파일명을 받아
        ``modules_change`` 로 경로에 매핑하고 로딩 파라미터를 갱신·저장한다.
        A1111 처럼 이 옵션이 없는 백엔드는 건드리지 않는다.
        """
        if modules is None or 'forge_additional_modules' not in current_options:
            return None
        wanted = self._module_names(modules)
        if wanted == self._module_names(current_options.get('forge_additional_modules')):
            return None
        return wanted

    def _switch_model_if_needed(self, model_name: str, modules=None):
        """필요 시 모델 전환 + VAE/TE 모듈 동기화 (한 번의 옵션 POST)."""
        if not model_name and modules is None:
            return
        current_response = requests.get(
            url=f'{self.api_url}/sdapi/v1/options',
            headers=_HEADERS, timeout=10
        )
        current_response.raise_for_status()
        current_options = current_response.json()

        changes: Dict[str, Any] = {}
        # 모듈을 체크포인트보다 먼저 — Forge 가 순서대로 처리하고 마지막에 한 번 로딩 파라미터를 갱신한다
        wanted = self._desired_forge_modules(current_options, modules)
        if wanted is not None:
            changes['forge_additional_modules'] = wanted
        if model_name and current_options.get('sd_model_checkpoint') != model_name:
            changes['sd_model_checkpoint'] = model_name
        if not changes:
            return
        if 'forge_additional_modules' in changes:
            logger.info("Forge modules → %s", changes['forge_additional_modules'] or '(없음)')
        switch_response = requests.post(
            url=f'{self.api_url}/sdapi/v1/options',
            json=changes,
            headers=_HEADERS, timeout=60
        )
        switch_response.raise_for_status()

    def refresh_loras(self) -> bool:
        """Forge 가 LoRA 폴더를 다시 스캔하게 한다(POST /sdapi/v1/refresh-loras).

        메모리는 정리하지 않는다 — 새로 넣은 LoRA 파일을 목록에 보이게 할 뿐이다. LoRA 매니저의
        '목록 다시 스캔'(mode='force')에서만 부른다: requestLoras 워커 스레드
        (ui.lora_catalog_cache.load_catalog_json)가 목록보다 먼저 보낸다 — Forge 가 파일을 전부 다시
        읽어 오래 걸릴 수 있으니 GUI 스레드(동기 슬롯)에서 부르지 않는다. 예전엔 대기열 정기 정리
        (cleanup_models)가 VRAM 정리인 줄 알고 매번 불렀다.
        """
        try:
            response = requests.post(
                url=f'{self.api_url}/sdapi/v1/refresh-loras',
                headers=_HEADERS, timeout=20,
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.warning("refresh-loras failed: %s", e)
            return False

    def unload_models(self) -> bool:
        """VRAM 게이지 수동 언로드 — Forge 에선 체크포인트 언로드가 VRAM 을 실제로 회수하는 유일한 API 다.

        예전 cleanup_models(full_reload=True) 는 여기에 LoRA 재스캔과 ``options``
        ``{'memmon_poll_rate': 8}`` POST 를 붙였는데, 둘 다 메모리를 정리하지 않고 options POST 는
        사용자의 Forge 설정(기본 5)을 영구히 8 로 바꾸며 config.json 을 매번 다시 썼다.
        """
        return self.unload_checkpoint()

    def unload_checkpoint(self) -> bool:
        """체크포인트를 내린다(POST /sdapi/v1/unload-checkpoint — Forge 가 모델을 내리고
        캐시를 비운 뒤 gc 한다). 다음 생성 때 Forge 가 알아서 다시 올린다.

        '생성 후 모델 언로드' · 대기열 정기 정리 · VRAM 게이지 수동 언로드가 모두 이것을 쓴다.
        엔드포인트가 없는 옛 버전(404)이나 서버 오류면 False — 성공으로 보고하지 않는다.

        타임아웃은 공유 상수 — 후처리 작업이 이 언로드(hold)를 기다리는 시간
        (BACKEND_JOB_UNLOAD_WAIT_SECONDS)이 이 요청의 상한보다 길도록 한 곳에서 맞춘다.
        """
        from core.resource_coordinator import UNLOAD_HTTP_TIMEOUT_SECONDS
        try:
            response = requests.post(
                url=f'{self.api_url}/sdapi/v1/unload-checkpoint',
                headers=_HEADERS, timeout=UNLOAD_HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            logger.info("[unload] checkpoint unloaded")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning("[unload] unload-checkpoint failed: %s", e)
            return False

    @staticmethod
    def _progress_preview(response: Any) -> Optional[str]:
        """/sdapi/v1/progress 의 current_image — 접두사 없는 base64. 없거나 이상하면 None.

        Forge 는 live preview 가 켜져 있으면(기본) 몇 스텝마다 중간 그림을 base64 로 준다.
        예전엔 이걸 버려서, 이미지를 보고 있는 동안엔 생성 중인지 화면에서 알 수 없었다.
        """
        if not isinstance(response, dict):
            return None
        raw = response.get('current_image')
        if not isinstance(raw, str) or not raw:
            return None
        if raw.startswith('data:'):
            raw = raw.split(',', 1)[-1]
        return raw if len(raw) > 64 else None

    def _start_progress_polling(self, callback: Optional[ProgressCallback],
                                stop_event: threading.Event):
        """별도 스레드에서 /sdapi/v1/progress 폴링"""
        if callback is None:
            return
        last_preview = None
        while not stop_event.is_set():
            try:
                r = requests.get(
                    f'{self.api_url}/sdapi/v1/progress',
                    timeout=3
                ).json()
                step = r.get('state', {}).get('sampling_step', 0)
                total = r.get('state', {}).get('sampling_steps', 0)
                progress_val = r.get('progress', 0)
                # 라이브 프리뷰 — 같은 그림을 매 폴링마다 다시 보내지 않는다 (base64 수십~수백 KB)
                preview = self._progress_preview(r)
                if preview == last_preview:
                    preview = None
                elif preview is not None:
                    last_preview = preview
                if total > 0:
                    callback(step, total, preview)
                elif progress_val > 0:
                    callback(int(progress_val * 100), 100, preview)
            except Exception as e:
                # 진행률 폴링은 반복 호출 — 노이즈 방지 위해 debug 로그만
                logger.debug("progress poll failed: %s", e)
            stop_event.wait(0.5)

    def _generate(self, endpoint: str, model_name: str, payload: Dict,
                  progress_callback: Optional[ProgressCallback] = None,
                  cancel_check: Optional[Callable[[], bool]] = None) -> GenerationResult:
        """txt2img / img2img 공통 생성 로직.

        앱이 스스로 덧붙인 부분 — Forge 설정에 맞춰 넣는 블록(DoRA·Anima38, ``main_retry_titles``)과 Forge 옵션 덮어쓰기
        (P10 ``_forge_option_parts``) — 을 Forge 가 거절하면(422 스크립트 없음·500 KeyError·설정 잠금) 그것만 빼고 다시
        보낸다(``core/forge_optional_parts.plan_retry``, 부분이 줄어드는 동안만). 시도마다 새 task id(``_generate_once``),
        다시 보내기 전에 취소 확인(critic B10), 기능 스냅샷은 캐시를 버리지 않고 이 워커에서 새로 받는다(critic A1).
        사용자가 패널에서 켠 가이던스 블록의 422 는 지금처럼 설명과 함께 실패한다. 모델 전환은 첫 시도 전에 한 번.

        DoRA·Anima38 422 알림은 요청의 비공개 블록 출처(``core/alwayson_propagation.PROVENANCE_KEY`` — 메인 체인이 앱 기본값
        블록만 적는다)로 가른다: 앱 기본값 = 정보(``app_block_retried``), 사용자 값·출처 모름 = 경고(``block_retried``,
        P10 검토 2 — A6). 그 키는 여기서 떼어 Forge 요청에는 싣지 않는다(키가 없으면 payload 객체 그대로).
        """
        try:
            from core.alwayson_propagation import take_provenance
            payload, provenance = take_provenance(payload)
            if cancel_check and cancel_check():
                return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")
            self._switch_model_if_needed(model_name, payload.get('forge_additional_modules'))
            if cancel_check and cancel_check():
                return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")

            # NOTE: 이전에 SAM3+LoRA 감지 시 사전 unload-checkpoint 호출 했었음.
            # 그러나 sam3_unload_after=True 추가로 sam-extra가 검출 후 알아서
            # 정리하므로 사전 unload는 불필요 + 메모리 단편화로 가용 VRAM
            # 4GB 정도 손실시킴 (사용자 로그 비교로 확인). 제거.

            from core.alwayson_propagation import main_retry_titles
            from core.forge_optional_parts import OptionalParts, max_attempts, plan_retry
            request_base, option_keys, notices = self._forge_option_parts(payload)
            parts = OptionalParts(titles=main_retry_titles(request_base), option_keys=option_keys)
            attempts = max_attempts(parts)
            for attempt in range(1, attempts + 1):
                result, rejection = self._generate_once(endpoint, request_base, progress_callback, cancel_check,
                                                        parts=parts, notices=notices)
                if rejection is None:
                    return result
                status, body, response = rejection
                retry = plan_retry(status, body, request_base, parts) if attempt < attempts else None
                self._remember_frozen(retry)
                if retry is None:
                    if status == 422:   # 어느 확장·기능인지 설명(P4)
                        return GenerationResult(success=False, error=_explain_rejection(body, request_base))
                    response.raise_for_status()   # 앱 옵션 탓이 아닌 500 — 지금과 같은 오류(아래 except)
                    return GenerationResult(success=False, error=f"API 요청 실패: HTTP {status}")
                if cancel_check and cancel_check():
                    return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")
                notices = [*notices, *self._optional_parts_retry_notices(retry, "", main=True, provenance=provenance)]
                request_base, parts = retry.payload, retry.remaining
            return GenerationResult(success=False, error="생성 재시도 한도를 넘었습니다.")   # 도달하지 않는다

        except requests.exceptions.RequestException as e:
            return GenerationResult(success=False, error=f"API 요청 실패: {e}")
        except Exception as e:
            return GenerationResult(success=False, error=f"생성 중 오류: {e}")

    def _generate_once(self, endpoint: str, payload: Dict,
                       progress_callback: Optional[ProgressCallback],
                       cancel_check: Optional[Callable[[], bool]], *,
                       parts=None, notices=()) -> tuple:
        """한 번 보낸다 → ``(결과, None)``, Forge 가 거절했으면 ``(None, (상태, 본문, 응답))``(``_read_retryable``).

        시도마다 새 force_task_id·취소 감시 스레드를 쓴다 — 거절된 요청은 Forge 가 start_task 만 하고 finish_task 를
        하지 않았다(modules/api/api.py) — 같은 id 를 다시 쓰면 취소 감시가 헷갈린다. ``notices``: 요청 전·재시도에서 생긴
        알림 — 결과 info 에 sam-extra 결과 알림 뒤로 붙인다. 네트워크·응답 예외는 호출자(``_generate``)가 받는다.
        """
        # 요청마다 force_task_id 를 붙여 취소 시 '우리 작업'의 상태를 볼 수 있게 한다
        # (core/webui_cancel.py). save_images 는 Forge 숨은 폴더 중복 저장 방지 정책으로
        # 확정한다 — 사용자가 설정에서 Forge 쪽 저장을 켰을 때만 요청값을 따른다.
        # 설정은 GUI 가 메모리로 밀어 넣은 값만 읽는다(이 워커 스레드가 ui_prefs.json 을
        # 열면 GUI 의 os.replace 저장이 Windows 에서 PermissionError 로 실패했다).
        from core.forge_output_policy import apply_save_policy, forge_save_outputs_setting
        from core.webui_cancel import approx_payload_bytes, unseen_grace_seconds, with_task_id
        request_payload, task_id = with_task_id(
            apply_save_policy(payload, forge_save_outputs_setting()))
        # 큰 본문(init 이미지·마스크)은 업로드·파싱이 끝나야 큐에 보인다 — unseen 유예를 늘린다.
        unseen_grace = unseen_grace_seconds(approx_payload_bytes(request_payload))

        # 진행률 폴링 시작
        stop_event = threading.Event()
        interrupt_requested = threading.Event()
        with self._generation_state_lock:
            if cancel_check and cancel_check():
                return GenerationResult(success=False, error="사용자가 작업을 취소했습니다"), None
            self._generation_inflight = True
            self._inflight_interrupts[task_id] = interrupt_requested
        if progress_callback:
            poll_thread = threading.Thread(
                target=self._start_progress_polling,
                args=(progress_callback, stop_event),
                daemon=True
            )
            poll_thread.start()
        # cancel_check 또는 interrupt() 가 오면 우리 작업이 active 일 때만 전역 interrupt 반복.
        threading.Thread(
            target=self._watch_cancellation,
            args=(task_id, stop_event, interrupt_requested, cancel_check, unseen_grace),
            name="webui-cancel-watch",
            daemon=True,
        ).start()

        try:
            response = requests.post(
                url=f'{self.api_url}{endpoint}',
                json=request_payload, headers=_HEADERS, timeout=600, stream=True
            )
            status, body = _read_retryable(response, parts)   # 422 → 설명·재시도 판정, 앱 옵션의 500 → 재시도 판정
            if status is not None:
                return None, (status, body, response)
            response.raise_for_status()
        finally:
            with self._generation_state_lock:
                self._inflight_interrupts.pop(task_id, None)
                self._generation_inflight = bool(self._inflight_interrupts)
            stop_event.set()

        try:
            r = _bounded_response_json(response)
        finally:
            close_response = getattr(response, "close", None)
            if callable(close_response):
                close_response()
        if 'images' in r and r['images']:
            image_values = r['images']
            if not isinstance(image_values, list) or len(image_values) > _MAX_RESULT_ARTIFACTS:
                raise ValueError(f"WebUI 이미지 결과는 최대 {_MAX_RESULT_ARTIFACTS}개까지 허용됩니다.")
            artifacts = []
            total_bytes = 0
            for index, value in enumerate(image_values):
                artifact = _decode_webui_image(value, index)
                total_bytes += len(artifact.data or b"")
                if total_bytes > _MAX_RESULT_BYTES:
                    raise ValueError("WebUI 이미지 결과 총 용량이 256MiB를 초과합니다.")
                artifacts.append(artifact)
            raw_info = r.get('info', {})
            if isinstance(raw_info, dict):
                generation_info = copy.deepcopy(raw_info)
            elif isinstance(raw_info, str):
                try:
                    parsed_info = json.loads(raw_info) if raw_info else {}
                except json.JSONDecodeError:
                    parsed_info = {"raw_info": raw_info}
                generation_info = (
                    parsed_info if isinstance(parsed_info, dict)
                    else {"raw_info": parsed_info}
                )
            else:
                generation_info = {"raw_info": raw_info}
            generation_info['artifact_count'] = len(artifacts)
            # sam-extra 결과 알림(SAM3 Error·Anima38 off·PAG 누락) + 요청 전·재시도 알림(P10) — UI 가 토스트로 띄운다(P4)
            result_notices = [*_extension_notices(self.api_url, generation_info, payload), *(notices or ())]
            if result_notices:
                from core.sam_extra_notices import INFO_KEY, notices_to_dicts
                generation_info[INFO_KEY] = notices_to_dicts(result_notices)
            return GenerationResult(
                success=True,
                image_data=artifacts[0].data,
                info=generation_info,
                artifacts=artifacts,
            ), None
        return GenerationResult(
            success=False,
            error=f"API 응답에 이미지가 없습니다: {r.get('detail', '알 수 없는 오류')}"
        ), None

    def txt2img(self, model_name: str, payload: Dict,
                progress_callback: Optional[ProgressCallback] = None,
                cancel_check: Optional[Callable[[], bool]] = None) -> GenerationResult:
        """텍스트→이미지 생성"""
        return self._generate(
            '/sdapi/v1/txt2img', model_name, payload, progress_callback, cancel_check
        )

    def img2img(self, model_name: str, payload: Dict,
                progress_callback: Optional[ProgressCallback] = None,
                cancel_check: Optional[Callable[[], bool]] = None) -> GenerationResult:
        """이미지→이미지 생성"""
        return self._generate(
            '/sdapi/v1/img2img', model_name, payload, progress_callback, cancel_check
        )

    def upscale(self, image_b64: str, settings: Dict) -> str:
        """extra-single-image API로 업스케일"""
        payload = {
            "image": image_b64,
            "resize_mode": 0 if settings.get('scale_mode') == 'factor' else 1,
            "upscaling_resize": settings.get('scale_factor', 2),
            "upscaling_resize_w": settings.get('target_width', 1024),
            "upscaling_resize_h": settings.get('target_height', 1024),
            "upscaler_1": settings.get('upscaler_name', 'Lanczos'),
        }
        response = requests.post(
            f'{self.api_url}/sdapi/v1/extra-single-image',
            json=payload, headers=_HEADERS, timeout=600
        )
        response.raise_for_status()
        r = response.json()
        if 'image' in r and r['image']:
            return r['image']
        raise RuntimeError("업스케일 API 응답에 이미지가 없습니다.")

    def adetailer(self, image_b64: str, settings: Dict, *,
                  cancel_check: Optional[Callable[[], bool]] = None) -> str:
        """단독/배치 ADetailer — 확장의 공식 ``skip_img2img`` 로 부모 재확산 없이 보정한다.

        확장 인자는 ``[enable, skip_img2img, slot...]`` 이다(aadetailer ui.py 의 components 순서).
        예전엔 ``[True, False, slot]`` 이라 skip 이 꺼져 있었다 — 부모 img2img 가 denoise 0.1·
        빈 프롬프트·API 기본 steps 로 **원본 전체를 한 번 더 재확산**했고(얼굴이 없어도 드리프트,
        그 재확산본이 저장됨), 이미지마다 풀해상도 샘플링과 VAE 왕복이 한 번씩 더 들었다.

        skip_img2img=True 면 확장이 부모 패스를 1스텝·128×128 로 줄이고 원래 steps/sampler/
        width/height 를 ``_ad_orig`` 에 보관해 인페인트 패스에 쓰며, 입력은 init 이미지 원본이다.
        그래서 그 값들을 sam3()/refine() 처럼 명시 전송한다(``_build_postprocess_payload``).
        t2i/i2i 안의 ADetailer(generator_generation)는 부모 생성 자체가 목적이라 False 가 맞다.

        settings 에 메인 생성 샘플링 블록 봉투가 있으면 ADetailer 블록 **뒤에** 전달한다(P7, ``_propagate``).
        """
        settings, envelope = _take_envelope(settings)
        adetailer_args = settings.get('adetailer_args')
        if not adetailer_args:
            # 슬롯 본문·기본값은 core/adetailer_args 한 벌 (예전엔 workers 레이어를 역참조했다).
            from core.adetailer_args import slot_from_settings

            adetailer_args = [True, True, slot_from_settings(settings)]
        # 호출자가 완성된 adetailer_args 를 넘기면(현재 없음) 그 skip 값을 그대로 존중한다.

        payload = self._build_postprocess_payload(
            image_b64,
            settings,
            prompt=settings.get('ad_prompt', ''),
            negative_prompt=settings.get('ad_negative', ''),
        )
        payload["alwayson_scripts"]["ADetailer"] = {"args": adetailer_args}
        from core.alwayson_propagation import AUX_ADETAILER
        return self._run_aux_postprocess(image_b64, payload, envelope, AUX_ADETAILER, settings, cancel_check)

    def refine(self, image_b64: str, settings: Dict, *,
               cancel_check: Optional[Callable[[], bool]] = None) -> str:
        """SAM3 Refine — 기존 이미지를 Target/Replacement로 재손질.

        sam-extra 워크플로 2를 앱에서 구현한 것. 확장의 Refine 패널은 Gradio 전용이라
        HTTP로 부를 수 없어서, 같은 결과가 나오도록 여기서 payload를 만든다.

        핵심 (sam3() 와 같은 규칙):
          · Refine은 마스크 영역만 건드려야 하므로 부모 i2i는 denoise 0으로 통과시키고
            SAM3 인페인트만 일하게 한다. (예전 sam3()는 denoise 0.1 부모 패스로 **이미지
            전체를 한 번 재확산**했다 — 지금은 sam3()도 denoise 0.)
          · denoise 0 이어도 부모 패스는 VAE 인코드·디코드를 거쳐 픽셀이 조금씩 바뀐다(드리프트) — SAM3 가 그 출력을
            쓰면 마스크 밖도 원본과 달라진다. 그래서 SAM3 state 에 원본 기준 요청(``sam3_args.with_init_source``)을 실어
            확장이 init 이미지로 검출·인페인트하게 한다. 지원하지 않는 확장·조건이 안 맞아 출력으로 돌았으면 결과 알림
            (infotext 'SAM3 Source', core/sam_extra_notices)으로 알린다.
          · steps/cfg/sampler/seed를 명시해 Forge 현재 UI 값에 좌우되지 않게 한다.
          · 메인 생성 샘플링 블록 봉투가 있으면 SAM3 Mask 뒤에 전달한다(P7 — 생성 안 SAM3 패스와 같게).
        """
        from core.refine_prompt import build_refine_prompts

        settings, envelope = _take_envelope(settings)
        prompts = build_refine_prompts(
            main_prompt=settings.get('main_prompt', ''),
            main_negative=settings.get('main_negative', ''),
            target=settings.get('target', ''),
            replacement=settings.get('replacement', ''),
            negative=settings.get('negative', ''),
            inherit_main=bool(settings.get('inherit_main', True)),
            inherit_negative=bool(settings.get('inherit_negative', True)),
        )

        sam3_settings = dict(settings)
        sam3_settings['sam3_prompt'] = settings.get('target') or 'face'
        sam3_settings['sam3_inpaint_prompt'] = prompts['prompt']
        sam3_settings['sam3_negative_prompt'] = prompts['negative_prompt']
        sam3_settings['sam3_mode'] = 'Inpaint'
        from core.sam3_args import with_init_source
        sam3_state = with_init_source(self._build_sam3_script_state(sam3_settings, self._sam_extra_snapshot()))
        # Forge 가 EXIF 방향을 적용한 크기 — 다르면 부모 패스가 늘려 찌그러지고 원본 기준 SAM3 도 'size' 로 폴백한다
        init_width, init_height = _forge_init_size(image_b64)

        payload = {
            "init_images": [image_b64],
            "prompt": prompts['prompt'],
            "negative_prompt": prompts['negative_prompt'],
            # 부모 i2i는 재확산하지 않게 denoise 0 — 그래도 VAE 왕복으로 픽셀이 조금 바뀌므로 SAM3 는 그 출력이 아니라
            # init 이미지를 쓴다(sam3_state 의 원본 기준 요청). 실제 작업은 SAM3 인페인트 패스가 한다.
            "denoising_strength": 0.0,
            "resize_mode": 0,
            "width": init_width,
            "height": init_height,
            "steps": int(settings.get('steps', 28)),
            "cfg_scale": float(settings.get('cfg_scale', 7.0)),
            "seed": int(settings.get('seed', -1)),
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {"SAM3 Mask": {"args": [sam3_state]}},
        }
        sampler = str(settings.get('sampler') or '').strip()
        if sampler and sampler != 'Use same sampler':
            payload["sampler_name"] = sampler
        scheduler = str(settings.get('scheduler') or '').strip()
        if scheduler and scheduler != 'Use same scheduler':
            payload["scheduler"] = scheduler

        logger.info("Refine: target=%r → prompt=%r", settings.get('target'), prompts['prompt'])
        from core.alwayson_propagation import AUX_REFINE
        return self._run_aux_postprocess(image_b64, payload, envelope, AUX_REFINE, sam3_settings, cancel_check)

    def sam3(self, image_b64: str, settings: Dict, *,
             cancel_check: Optional[Callable[[], bool]] = None) -> str:
        """img2img + SAM3 확장으로 마스킹/인페인트 (배치/단독 실행 경로).

        예전에는 `_build_postprocess_payload`를 썼는데 거기 기본 denoising_strength가
        0.1이라 **이미지 전체가 한 번 재확산**됐다. `_postprocess_base_payload`를 채우는
        호출자도 없어 steps/cfg/sampler가 전부 Forge 현재 UI 값에 좌우돼 재현도 안 됐다.

        이제 부모 i2i는 denoise 0으로 통과시키고, 실제 작업은 SAM3 인페인트 패스가
        전담한다. 샘플링 파라미터도 명시 전송한다. 다만 denoise 0 이어도 부모 패스는 VAE
        인코드·디코드를 거쳐 픽셀이 조금씩 바뀐다 — SAM3 가 그 출력을 검출·인페인트하면 마스크
        밖도 원본과 달라진다(드리프트). 그래서 SAM3 state 에 원본 기준 요청
        (``sam3_args.with_init_source``)을 실어 확장이 init 이미지를 쓰게 한다. 확장이 모르거나
        조건이 안 맞아 출력으로 돌았으면 결과 알림(infotext 'SAM3 Source', core/sam_extra_notices)이 알린다.

        메인 생성 샘플링 블록 봉투가 있으면 SAM3 Mask 뒤에 전달한다(P7). 'Mask only' 는 인페인트 패스가 없어
        전달하지 않는다(core/alwayson_propagation.blocks_for).
        """
        settings, envelope = _take_envelope(settings)
        from core.sam3_args import with_init_source
        sam3_state = with_init_source(self._build_sam3_script_state(settings, self._sam_extra_snapshot()))
        # Forge 가 EXIF 방향을 적용한 크기 — 다르면 부모 패스가 늘려 찌그러지고 원본 기준 SAM3 도 'size' 로 폴백한다
        init_width, init_height = _forge_init_size(image_b64)

        prompt = sam3_state.get('sam3_inpaint_prompt', '') or settings.get('prompt', '')
        negative = sam3_state.get('sam3_negative_prompt', '') or settings.get('negative_prompt', '')

        payload = {
            "init_images": [image_b64],
            "prompt": prompt,
            "negative_prompt": negative,
            "denoising_strength": 0.0,
            "resize_mode": 0,
            "width": init_width,
            "height": init_height,
            "steps": int(settings.get('steps', 28)),
            "cfg_scale": float(settings.get('cfg_scale', 7.0)),
            "seed": int(settings.get('seed', -1)),
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {"SAM3 Mask": {"args": [sam3_state]}},
        }
        sampler = str(settings.get('sampler') or '').strip()
        if sampler and sampler != 'Use same sampler':
            payload["sampler_name"] = sampler
        scheduler = str(settings.get('scheduler') or '').strip()
        if scheduler and scheduler != 'Use same scheduler':
            payload["scheduler"] = scheduler
        from core.alwayson_propagation import AUX_SAM3
        return self._run_aux_postprocess(image_b64, payload, envelope, AUX_SAM3, settings, cancel_check)
