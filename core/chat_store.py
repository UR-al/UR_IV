"""대화 탭의 스레드 저장소 — config/chat_threads.json.

왜 localStorage 가 아니라 파일이냐: 대화에 이미지(base64)가 붙으면 수 MB 가 금방 된다.
localStorage 는 5MB 안팎에서 조용히 실패하고, 그러면 대화가 사라진 채로 앱은 멀쩡히
돈다 — 이 저장소의 단골 실패 방식이다. 파일이면 크기 제한이 없고 웹 모드에서도 같은
경로로 산다.

Qt 를 모른다. 순수 함수라 tests/test_chat_feature.py 가 그대로 돌린다.
"""
from __future__ import annotations

import os
import base64
import json
import math
import re
from pathlib import Path
from typing import Any

from utils.atomic_json import atomic_write_json, load_json_safe

#: 스레드 수 상한 — GemmaStudio 와 같은 100. 그 위는 오래된 것부터 버린다.
MAX_THREADS = 100
#: 메시지 하나의 본문 상한(글자). 모델 출력이 폭주해도 파일이 무한히 크지 않게.
MAX_CONTENT_CHARS = 200_000
#: 스레드 하나에 붙는 이미지 총량 상한(base64 글자 수 ≈ 바이트). 넘치면 오래된 이미지부터 뗀다.
MAX_IMAGE_CHARS_PER_THREAD = 6_000_000


def default_path() -> Path:
    try:
        from core.storage_paths import config_file  # type: ignore
        return Path(config_file("chat_threads.json"))
    except Exception:
        return Path("config") / "chat_threads.json"


def _clean_message(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role", "") or "")
    if role not in {"user", "assistant", "system"}:
        return None
    content = str(raw.get("content", "") or "")
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    out: dict[str, Any] = {
        "id": str(raw.get("id", "") or ""),
        "role": role,
        "content": content,
        "createdAt": int(raw.get("createdAt", 0) or 0),
    }
    images = raw.get("images")
    if isinstance(images, list):
        kept = [str(x) for x in images if isinstance(x, str) and x]
        if kept:
            out["images"] = kept
    if raw.get("error"):
        out["error"] = str(raw["error"])[:2000]
    if raw.get("thinking"):
        out["thinking"] = str(raw["thinking"])[:MAX_CONTENT_CHARS]
    for key in ("evalCount", "durationMs"):
        try:
            if raw.get(key) is not None:
                out[key] = int(raw[key])
        except (TypeError, ValueError):
            pass
    if raw.get("doneReason"):
        out["doneReason"] = str(raw["doneReason"])[:40]
    if raw.get("model"):
        out["model"] = str(raw["model"])[:200]
    if raw.get('structured') is True:
        out['structured'] = True
    artifacts = []
    raw_artifacts = raw.get('artifacts')
    for item in (raw_artifacts[:32] if isinstance(raw_artifacts, list) else []):
        if not isinstance(item, dict) or item.get('kind') not in {'image', 'animated', 'video', 'audio'}:
            continue
        path = str(item.get('path') or '')[:4000]
        if not path or re.match(r'^(?:https?|data|blob|javascript|file):', path, re.I):
            continue
        artifacts.append({'kind': item['kind'], 'path': path,
                          'filename': str(item.get('filename') or '')[:200],
                          'mime': str(item.get('mime') or '')[:100]})
    if artifacts:
        out['artifacts'] = artifacts
    request = raw.get('generationRequest')
    if isinstance(request, dict) and request.get('mode') in {'auto', 'chat', 'image', 'video'}:
        cleaned = {'mode': request['mode'], 'family': 'krea2' if request.get('family') == 'krea2' else 'current',
                   'hadImage': bool(request.get('hadImage'))}
        for key, default, low, high in [('duration', 5, 1, 15), ('denoise', .65, .01, 1)]:
            try:
                value = float(request.get(key, default))
                cleaned[key] = max(low, min(high, value)) if math.isfinite(value) else default
            except (TypeError, ValueError):
                cleaned[key] = default
        out['generationRequest'] = cleaned
    generation = raw.get('generation')
    if isinstance(generation, dict) and generation.get('kind') in {'image', 'video'}:
        out['generation'] = {'kind': generation['kind'], 'phase': str(generation.get('phase') or '')[:40],
                             'message': str(generation.get('message') or '')[:1000]}
    return out


def _clean_thread(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    thread_id = str(raw.get("id", "") or "")
    if not thread_id:
        return None
    messages = [m for m in (_clean_message(x) for x in (raw.get("messages") or [])) if m]
    # 이미지 총량 — 오래된 메시지의 이미지부터 뗀다
    budget = MAX_IMAGE_CHARS_PER_THREAD
    for m in reversed(messages):
        imgs = m.get("images")
        if not imgs:
            continue
        size = sum(len(x) for x in imgs)
        if size <= budget:
            budget -= size
        else:
            m.pop("images", None)
            m["imagesDropped"] = True
    return {
        "id": thread_id,
        "title": str(raw.get("title", "") or "")[:200],
        "model": str(raw.get("model", "") or "")[:200],
        "createdAt": int(raw.get("createdAt", 0) or 0),
        "updatedAt": int(raw.get("updatedAt", 0) or 0),
        "messages": messages,
    }


def normalise_threads(raw: Any) -> list[dict[str, Any]]:
    """저장 전·후에 같은 모양으로. 최근 것이 앞에, 상한 초과는 뒤에서 버린다."""
    items = [t for t in (_clean_thread(x) for x in (raw or [])) if t] if isinstance(raw, list) else []
    items.sort(key=lambda t: t.get("updatedAt", 0), reverse=True)
    return items[:MAX_THREADS]


class ChatStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_path()

    def load(self) -> list[dict[str, Any]]:
        raw = load_json_safe(str(self.path), {})
        threads = raw.get("threads") if isinstance(raw, dict) else raw
        return normalise_threads(threads)

    def save(self, threads: Any) -> list[dict[str, Any]]:
        cleaned = normalise_threads(threads)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self.path), {"version": 1, "threads": cleaned})
        return cleaned


def strip_data_url(value: str) -> str:
    """`data:image/png;base64,....` → base64 본문. Ollama 는 접두사 없는 base64 를 받는다."""
    text = str(value or "")
    if text.startswith("data:"):
        comma = text.find(",")
        if comma >= 0:
            return text[comma + 1:]
    return text


OPTION_FLOATS = ("temperature", "top_p")
OPTION_INTS = ("num_predict", "num_ctx")


def clean_options(raw) -> dict:
    """Vue 가 보낸 생성 옵션 중 Ollama 에 넘길 것만, 값이 말이 되는 것만.

    temperature/top_p: 0 이상 유한 실수. num_predict: -1(제한 없음) 또는 양수. num_ctx: 양수.
    그 밖의 키(num_gpu 등)는 버린다 — 화면에 없는 손잡이가 몰래 들어오지 않게.
    """
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for key in OPTION_FLOATS:
        if key in raw:
            try:
                value = float(raw[key])
            except (TypeError, ValueError):
                continue
            if value == value and value >= 0 and value != float("inf"):
                out[key] = value
    for key in OPTION_INTS:
        if key in raw:
            try:
                value = int(raw[key])
            except (TypeError, ValueError):
                continue
            if value > 0 or (key == "num_predict" and value == -1):
                out[key] = value
    return out


_BASE64_BODY = re.compile(r"[A-Za-z0-9+/]+={0,2}")


def looks_like_image_path(value: str) -> bool:
    """이미지 항목이 파일 **경로**인지 — `data:` URL 과 맨 base64 본문은 아니다.

    '/' 포함 여부만 보면 안 된다: JPEG base64 는 '/9j/' 로 시작하고 본문에도 '/' 가 흔하다
    (os.path.isabs 도 '/9j/…' 를 절대경로로 오판한다). base64 알파벳만으로 된 문자열은
    본문으로 본다 — 그래서 `build_ollama_messages` 가 접두사를 뗀 뒤에 불러도 안전하다.
    """
    if not isinstance(value, str) or not value or value.startswith("data:"):
        return False
    if _BASE64_BODY.fullmatch(value):
        return False
    return "/" in value or "\\" in value


def inline_image_paths(messages, *, max_bytes=20_000_000):
    """히스토리·갤러리 카드를 끌어다 놓으면 이미지가 **경로**로 온다 — 파일을 읽어 base64 로 바꾼다.

    `data:` URL 과 맨 base64 는 그대로 둔다(`build_ollama_messages` 가 접두사를 뗀다).
    없거나 너무 큰 파일은 조용히 뺀다 — 모델에게 깨진 이미지를 주느니 안 주는 편이 낫다.
    파일 I/O 라 GUI 스레드가 아니라 ChatWorker.run 에서, 최근 턴만 남긴 뒤에 부른다.
    """
    out = []
    for msg in messages or []:
        if not isinstance(msg, dict) or not isinstance(msg.get("images"), list):
            out.append(msg)
            continue
        fixed = []
        for img in msg["images"]:
            if not isinstance(img, str) or not img:
                continue
            if not looks_like_image_path(img):
                fixed.append(img)
                continue
            try:
                if not os.path.isfile(img):
                    continue
                # 이 base64 는 호출자가 고른 URL(Ollama/LM Studio)로 나간다 — 웹의 다른 파일
                # 읽기 경로와 같은 검증(이미지 확장자 화이트리스트 + 시스템 폴더 차단)을 거친다.
                # 없으면 토큰 보유 웹 클라이언트가 config JSON 같은 임의 파일을 외부로 빼낼 수 있다.
                from core.path_safety import safe_input_path
                safe = safe_input_path(img)
                if not safe or os.path.getsize(safe) > max_bytes:
                    continue
                with open(safe, "rb") as fh:
                    fixed.append(base64.b64encode(fh.read()).decode("ascii"))
            except OSError:
                continue
        copy = dict(msg)
        if fixed:
            copy["images"] = fixed
        else:
            copy.pop("images", None)
        out.append(copy)
    return out


def build_ollama_messages(messages: list[dict[str, Any]], system_prompt: str = "", max_turns: int = 24) -> list[dict[str, Any]]:
    """Vue 가 보낸 대화 → Ollama /api/chat 메시지 배열.

    system 은 맨 앞에 하나. 최근 ``max_turns`` 개만 보낸다 — 문맥창을 넘기면 모델이
    앞부분을 조용히 버리는데, 그러면 어디까지 기억하는지 알 수 없다. 자르는 쪽이 낫다.
    """
    out: list[dict[str, Any]] = []
    system = str(system_prompt or "").strip()
    if system:
        out.append({"role": "system", "content": system})
    turns = [m for m in messages if isinstance(m, dict) and m.get("role") in {"user", "assistant"}]
    for m in turns[-max_turns:]:
        item: dict[str, Any] = {"role": m["role"], "content": str(m.get("content", "") or "")}
        images = [strip_data_url(x) for x in (m.get("images") or []) if isinstance(x, str) and x]
        if images:
            item["images"] = images
        out.append(item)
    return out
