"""sam-extra Anima Tile & Repair 라우트 HTTP 클라이언트 — Qt 를 모르는 순수 모듈.

확장 sam3ext/tile_repair_api.py 의 계약:
  GET  /sam-extra/tile-repair/options → {"version","available","models","default_model","dit","text_encoder",
                                          "vae","defaults","ranges","increments"}
  POST /sam-extra/tile-repair  {image(base64 PNG/JPEG/WebP), prompt, negative_prompt, steps, cfg_scale, flow_shift,
                                multiplier, short_side, seed, [model, dit, text_encoder, vae], unload_forge_before}
       → {"version","interrupted","image"(base64 PNG, infotext 는 parameters 청크),"info","seed","width","height","model"}
       · 400 값 오류 · 413 너무 큼 · 422 확장 쪽 준비 문제(TE/VAE/DiT·모델 없음) · 503 벤더 없음
  POST /sam-extra/tile-repair/stop → {"stopped": bool} — 이 라우트의 요청만 멈춘다. 도착한 순간부터(Forge 큐를
       기다리는 중이면 큐를 받자마자 아무것도 올리지 않고 interrupted 로 답한다) — txt2img 도, 패널 Tile-Repair 도
       건드리지 않는다. false = 멈출 요청이 없었다(아직 가는 중이거나 이미 끝났다)
모든 요청에 Notebook 라우트와 같은 ``X-SAM3-Notebook: 1`` 헤더가 필요하다(없으면 403). 라우트가 없으면 404 —
옛 확장이다(:class:`TileRepairUnavailable`). 앱의 WebUI 백엔드처럼 인증 헤더는 보내지 않는다.

예외 문구는 사용자에게 그대로 보여도 된다 — URL 은 싣지 않고, 확장의 ``detail`` 은 200자까지만 붙인다.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.backend_probe import request_timeout

TILE_REPAIR_API_PATH = "/sam-extra/tile-repair"
TILE_REPAIR_OPTIONS_PATH = "/sam-extra/tile-repair/options"
TILE_REPAIR_STOP_PATH = "/sam-extra/tile-repair/stop"
OPTIONS_TIMEOUT = 15.0
STOP_TIMEOUT = 5.0
# 첫 실행은 Anima DiT·TE·VAE 를 디스크에서 올리고(수십 초) 50스텝을 돈다. 다른 생성이 큐를 쥐고 있으면
# 그만큼 더 기다린다(확장은 Forge Generate 와 같은 queue_lock 을 쓴다).
RUN_TIMEOUT = 1800.0
_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "X-SAM3-Notebook": "1",
}
_DETAIL_LIMIT = 200

Requester = Callable[..., Any]


class TileRepairError(RuntimeError):
    """Tile & Repair 요청 실패의 공통 부모. 문구는 사용자에게 그대로 보여도 된다."""


class TileRepairUnavailable(TileRepairError):
    """연결된 Forge 의 sam-extra 에 Tile & Repair 라우트가 없다(옛 확장)."""


@dataclass(frozen=True)
class TileRepairResult:
    png: bytes
    info: str
    seed: Optional[int]
    width: Optional[int]
    height: Optional[int]
    model: str
    interrupted: bool = False


def normalize_base_url(api_url: Any) -> str:
    return str(api_url or "").strip().rstrip("/")


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return str(detail)[:_DETAIL_LIMIT] if isinstance(detail, (str, int, float)) else ""


def _optional_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class ForgeTileRepairClient:
    def __init__(self, api_url: Any, *, request: Optional[Requester] = None) -> None:
        self.base_url = normalize_base_url(api_url)
        if not self.base_url:
            raise ValueError("Forge 주소가 비어 있습니다")
        self._request = request

    # ── 전송 ──
    def _send(self, method: str, path: str, *, timeout: float, body: Optional[dict] = None) -> Any:
        request = self._request
        if request is None:
            import requests

            request = requests.request
        kwargs: dict = {"headers": dict(_HEADERS), "timeout": request_timeout(self.base_url, timeout)}
        if body is not None:
            kwargs["json"] = body
        try:
            return request(method, f"{self.base_url}{path}", **kwargs)
        except Exception as exc:
            raise TileRepairError(f"Forge 에 연결하지 못했습니다 ({type(exc).__name__})") from exc

    @staticmethod
    def _json(response: Any) -> dict:
        try:
            body = response.json()
        except Exception as exc:
            raise TileRepairError("Forge Tile & Repair 응답 형식이 올바르지 않습니다") from exc
        if not isinstance(body, dict):
            raise TileRepairError("Forge Tile & Repair 응답 형식이 올바르지 않습니다")
        return body

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        status = getattr(response, "status_code", None)
        if status == 200:
            return
        detail = _detail(response)
        suffix = f": {detail}" if detail else ""
        if status in (404, 405):
            raise TileRepairUnavailable("연결된 Forge 의 sam-extra 가 Tile & Repair 라우트를 지원하지 않습니다 — "
                                        "확장을 업데이트하세요")
        if status in (401, 403):
            raise TileRepairError("Forge 가 Tile & Repair 요청을 거부했습니다 — 로그인 설정(--gradio-auth·--api-auth)이나 "
                                  "sam-extra 버전을 확인하세요")
        if status == 413:
            raise TileRepairError("원본 이미지가 너무 큽니다 (Forge 한도 64 MB · 64 MP)")
        if status == 400:
            raise TileRepairError("Forge 가 Tile & Repair 설정을 받지 않았습니다" + suffix)
        if status == 422:
            raise TileRepairError("Forge Tile & Repair 준비가 안 됐습니다" + suffix)
        if status == 503:
            raise TileRepairError("Forge 에 Anima 벤더(sd-scripts)가 없습니다 — sam-extra install.py 를 다시 "
                                  "실행하세요" + suffix)
        raise TileRepairError(f"Forge Tile & Repair 가 실패했습니다 (HTTP {status})" + suffix)

    # ── 라우트 ──
    def options(self) -> dict:
        """모델 선택지·기본값·범위. 라우트가 없으면 :class:`TileRepairUnavailable`."""
        response = self._send("GET", TILE_REPAIR_OPTIONS_PATH, timeout=OPTIONS_TIMEOUT)
        self._raise_for_status(response)
        body = self._json(response)
        if not isinstance(body.get("models"), list) or not isinstance(body.get("defaults"), dict):
            raise TileRepairError("Forge Tile & Repair 선택지 형식이 올바르지 않습니다")
        return body

    def run(self, body: dict) -> TileRepairResult:
        """한 번 복원한다. ⏹ 로 멈췄으면 ``interrupted=True`` 인 빈 결과."""
        response = self._send("POST", TILE_REPAIR_API_PATH, timeout=RUN_TIMEOUT, body=body)
        self._raise_for_status(response)
        payload = self._json(response)
        model = str(payload.get("model") or "")
        if payload.get("interrupted") is True:
            return TileRepairResult(b"", "", None, None, None, model, interrupted=True)
        image = payload.get("image")
        try:
            png = base64.b64decode(image, validate=True) if isinstance(image, str) else b""
        except (binascii.Error, ValueError):
            png = b""
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise TileRepairError("Forge Tile & Repair 결과 이미지가 올바르지 않습니다")
        info = payload.get("info")
        return TileRepairResult(
            png=png,
            info=info if isinstance(info, str) else "",
            seed=_optional_int(payload.get("seed")),
            width=_optional_int(payload.get("width")),
            height=_optional_int(payload.get("height")),
            model=model,
        )

    def stop(self) -> bool:
        """Tile & Repair 작업을 멈춘다. 멈춘 작업이 있었으면 True."""
        response = self._send("POST", TILE_REPAIR_STOP_PATH, timeout=STOP_TIMEOUT)
        self._raise_for_status(response)
        return self._json(response).get("stopped") is True


__all__ = [
    "ForgeTileRepairClient", "RUN_TIMEOUT", "TILE_REPAIR_API_PATH", "TILE_REPAIR_OPTIONS_PATH",
    "TILE_REPAIR_STOP_PATH", "TileRepairError", "TileRepairResult", "TileRepairUnavailable",
    "normalize_base_url",
]
