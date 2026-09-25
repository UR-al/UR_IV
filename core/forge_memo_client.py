"""sam-extra 확장의 메모 라우트(/sam3-notebook/memos) HTTP 클라이언트 — Qt 를 모르는 순수 모듈.

계약(공유 메모 계약):
  GET    /sam3-notebook/memos[?include_deleted=1] → {"revision", "memos"}
  PUT    /sam3-notebook/memos/{id}  {"title","text","updated_at","base_updated_at"}
         → {"revision","memo"} · 409 {"detail","memo"}(서버 것이 더 새것)
  DELETE /sam3-notebook/memos/{id}[?base_updated_at=] → {"revision","memo"(삭제 표시)} · 모르는 id 면 404
         · base 뒤에 고친 메모면 409 {"detail","memo"}(계약에 더한 선택 쿼리 — 모르는 확장은 무시)
모든 요청에 Notebook 라우트와 같은 ``X-SAM3-Notebook: 1`` 헤더가 필요하다(없으면 403).
GET 이 404 면 메모 라우트가 없는 옛 확장이다(:class:`MemoRoutesUnavailable`).

앱의 WebUI 백엔드(backends/webui_backend.py)는 인증 헤더를 쓰지 않는다 — 여기서도 JSON 헤더와
Notebook 헤더만 보낸다. Forge 를 ``--gradio-auth`` 로 띄웠거나 ``--api``·``--nowebui`` 와 ``--api-auth`` 로
띄웠으면 라우트가 401 을 돌려주고, 그건 사용자에게 보여 줄 오류(:class:`MemoRemoteError`)가 된다.
예외 문구에 URL·서버 원문을 싣지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import quote

from core.backend_probe import request_timeout
from core.memo_store import MEMO_ID_RE, sanitize_memo

MEMO_API_PATH = "/sam3-notebook/memos"
DEFAULT_TIMEOUT = 5.0
_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "X-SAM3-Notebook": "1",
}
_DETAIL_LIMIT = 200

Requester = Callable[..., Any]


class ForgeMemoError(RuntimeError):
    """메모 동기화 요청 실패의 공통 부모. 문구는 사용자에게 그대로 보여도 된다."""


class MemoRoutesUnavailable(ForgeMemoError):
    """연결된 Forge 의 sam-extra 에 메모 라우트가 없다(옛 확장)."""


class MemoRemoteError(ForgeMemoError):
    """연결 실패·거부·서버 오류·형식 오류."""


class MemoRejectedError(MemoRemoteError):
    """서버가 이 메모 하나를 받지 않았다(형식·크기·개수 한도) — 다른 메모 동기화는 계속할 수 있다."""


class MemoNotFoundError(ForgeMemoError):
    """DELETE 대상 id 를 서버가 모른다."""


class MemoConflictError(ForgeMemoError):
    """PUT 의 base_updated_at 보다 서버 것이 새것 — ``memo`` 는 서버에 저장된 메모."""

    def __init__(self, message: str, memo: Optional[dict]) -> None:
        super().__init__(message)
        self.memo = memo


@dataclass(frozen=True)
class RemoteMemos:
    revision: int
    memos: list


def normalize_base_url(api_url: Any) -> str:
    return str(api_url or "").strip().rstrip("/")


def _revision(body: dict) -> int:
    value = body.get("revision")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return str(detail)[:_DETAIL_LIMIT] if isinstance(detail, (str, int, float)) else ""


class ForgeMemoClient:
    def __init__(self, api_url: Any, *, request: Optional[Requester] = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = normalize_base_url(api_url)
        if not self.base_url:
            raise ValueError("Forge 주소가 비어 있습니다")
        self._request = request
        self._timeout = request_timeout(self.base_url, timeout)

    # ── 전송 ──
    def _send(self, method: str, path: str, *, params: Optional[dict] = None,
              body: Optional[dict] = None) -> Any:
        request = self._request
        if request is None:
            import requests

            request = requests.request
        kwargs: dict = {"headers": dict(_HEADERS), "timeout": self._timeout}
        if params:
            kwargs["params"] = params
        if body is not None:
            kwargs["json"] = body
        try:
            return request(method, f"{self.base_url}{path}", **kwargs)
        except Exception as exc:
            raise MemoRemoteError("Forge 에 연결하지 못했습니다 — 메모는 이 PC 에 저장했고 "
                                  "다시 연결되면 동기화합니다") from exc

    @staticmethod
    def _json(response: Any) -> dict:
        try:
            body = response.json()
        except Exception as exc:
            raise MemoRemoteError("Forge 메모 응답 형식이 올바르지 않습니다") from exc
        if not isinstance(body, dict):
            raise MemoRemoteError("Forge 메모 응답 형식이 올바르지 않습니다")
        return body

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        status = getattr(response, "status_code", None)
        if status == 200:
            return
        if status in (401, 403):
            raise MemoRemoteError("Forge 가 메모 요청을 거부했습니다 — 로그인 설정(--gradio-auth·--api-auth)이나 "
                                  "sam-extra 버전을 확인하세요")
        if status == 413:
            raise MemoRejectedError("Forge 메모 저장소 한도(메모 수 또는 크기)를 넘어 받지 않았습니다")
        if status in (400, 422):
            detail = _detail(response)
            raise MemoRejectedError("Forge 가 메모를 받지 않았습니다(형식 오류)"
                                    + (f": {detail}" if detail else ""))
        if status == 409:
            # 메모 없는 409 = 서버 저장소가 더 새 schema(확장 MemoSchemaError)
            raise MemoRemoteError("Forge 의 메모 저장소가 더 새 버전 형식입니다 — 앱을 업데이트하세요")
        raise MemoRemoteError(f"Forge 메모 요청이 실패했습니다 (HTTP {status})")

    @staticmethod
    def _memo_path(memo_id: str) -> str:
        if not isinstance(memo_id, str) or not MEMO_ID_RE.fullmatch(memo_id):
            raise ValueError("메모 id 형식이 올바르지 않습니다")
        return f"{MEMO_API_PATH}/{quote(memo_id, safe='')}"

    def _memo_result(self, response: Any, memo_id: str) -> tuple[int, dict]:
        body = self._json(response)
        memo = sanitize_memo(body.get("memo"))
        if memo is None or memo["id"] != memo_id:
            raise MemoRemoteError("Forge 메모 응답 형식이 올바르지 않습니다")
        return _revision(body), memo

    # ── 라우트 ──
    def list_memos(self, *, include_deleted: bool = True) -> RemoteMemos:
        response = self._send("GET", MEMO_API_PATH,
                              params={"include_deleted": "1"} if include_deleted else None)
        if getattr(response, "status_code", None) == 404:
            raise MemoRoutesUnavailable("연결된 Forge 의 sam-extra 가 메모 동기화를 지원하지 않습니다 — "
                                        "확장을 업데이트하세요")
        self._raise_for_status(response)
        body = self._json(response)
        raw = body.get("memos")
        if not isinstance(raw, list):
            raise MemoRemoteError("Forge 메모 응답 형식이 올바르지 않습니다")
        memos, seen = [], set()
        for item in raw:
            memo = sanitize_memo(item)
            if memo is None or memo["id"] in seen:
                continue
            seen.add(memo["id"])
            memos.append(memo)
        return RemoteMemos(_revision(body), memos)

    def probe(self) -> bool:
        """메모 라우트가 있으면 True, 옛 확장(404)이면 False. 연결 실패는 MemoRemoteError."""
        try:
            self.list_memos(include_deleted=False)
        except MemoRoutesUnavailable:
            return False
        return True

    def put_memo(self, memo_id: str, *, title: str, text: str,
                 updated_at: Optional[str] = None,
                 base_updated_at: Optional[str] = None,
                 created_at: Optional[str] = None) -> tuple[int, dict]:
        """``created_at`` 은 계약 밖의 선택 필드 — 확장은 새 메모일 때만 쓰고 모르면 무시한다."""
        body: dict = {"title": title, "text": text, "base_updated_at": base_updated_at or None}
        if updated_at:
            body["updated_at"] = updated_at
        if created_at:
            body["created_at"] = created_at
        response = self._send("PUT", self._memo_path(memo_id), body=body)
        status = getattr(response, "status_code", None)
        if status == 409:
            payload = self._json(response)
            stored = sanitize_memo(payload.get("memo"))
            if stored is not None:
                raise MemoConflictError("Forge 에 더 새 메모가 있습니다", stored)
        if status == 404:
            raise MemoRoutesUnavailable("연결된 Forge 의 sam-extra 가 메모 동기화를 지원하지 않습니다 — "
                                        "확장을 업데이트하세요")
        self._raise_for_status(response)
        return self._memo_result(response, memo_id)

    def delete_memo(self, memo_id: str, *, base_updated_at: Optional[str] = None) -> tuple[int, dict]:
        """``base_updated_at``(선택) = 지우기로 정할 때 본 서버 updated_at. 그 뒤 다른 곳에서 고쳤으면
        확장이 409 로 거절한다(:class:`MemoConflictError` — 계약에 더한 선택 쿼리)."""
        response = self._send("DELETE", self._memo_path(memo_id),
                              params={"base_updated_at": base_updated_at} if base_updated_at else None)
        status = getattr(response, "status_code", None)
        if status == 404:
            raise MemoNotFoundError("Forge 에 없는 메모입니다")
        if status == 409:
            stored = sanitize_memo(self._json(response).get("memo"))
            if stored is not None:
                raise MemoConflictError("Forge 에 더 새 메모가 있습니다", stored)
        self._raise_for_status(response)
        return self._memo_result(response, memo_id)


__all__ = [
    "DEFAULT_TIMEOUT", "ForgeMemoClient", "ForgeMemoError", "MEMO_API_PATH", "MemoConflictError",
    "MemoNotFoundError", "MemoRejectedError", "MemoRemoteError", "MemoRoutesUnavailable", "RemoteMemos",
    "normalize_base_url",
]
