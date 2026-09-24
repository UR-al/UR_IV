"""Per-backend snapshot of ComfyUI ``/object_info`` for graph compilation.

``/object_info`` is the full node/resource schema (1.8-9 MB).  Compiling one
image used to fetch and parse it once for the generation and once more for
every post-processing pass.  The compiler only needs a *recent* document:

* the snapshot belongs to one backend instance and one API URL (a new URL,
  e.g. after a managed restart, is a miss);
* it is refreshed by every live fetch (connect/``get_info``), dropped when the
  node pack is installed or the managed ComfyUI restarts, and expires after a
  short TTL;
* a compile error while using a cached document is retried once with a fresh
  one by the backend, so a model/LoRA added a moment ago is never rejected.

Live callers (workflow inspector, XYZ preflight, Krea2, ``get_info``) keep
using the uncached fetch.

Generation API profile jobs build a new backend per job, so an instance
snapshot never hits there; ``EXTERNAL_OBJECT_INFO`` keeps one snapshot per
endpoint for them (same TTL and one-shot fresh retry, bounded number of URLs).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

DEFAULT_TTL_SECONDS = 60.0
DEFAULT_MAX_URLS = 4


class ObjectInfoCache:
    def __init__(
        self,
        *,
        ttl: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = max(0.0, float(ttl))
        self._clock = clock
        self._lock = threading.Lock()
        self._url = ""
        self._data: Optional[dict] = None
        self._stored_at = 0.0

    @staticmethod
    def _key(url: str) -> str:
        return str(url or "").strip().rstrip("/")

    def store(self, url: str, data: dict) -> None:
        """Remember a freshly fetched document for ``url``."""
        with self._lock:
            self._url = self._key(url)
            self._data = data
            self._stored_at = self._clock()

    def invalidate(self) -> None:
        with self._lock:
            self._url = ""
            self._data = None
            self._stored_at = 0.0

    def peek(self, url: str) -> Optional[dict]:
        """The snapshot for ``url`` if it is still fresh, else ``None``."""
        with self._lock:
            if self._data is None or self._url != self._key(url):
                return None
            if self._clock() - self._stored_at > self._ttl:
                # Expired: release the multi-MB document instead of holding it.
                self._url, self._data, self._stored_at = "", None, 0.0
                return None
            return self._data

    def expired(self) -> bool:
        """True when nothing fresh is held (empty, invalidated or past the TTL)."""
        with self._lock:
            if self._data is None:
                return True
            if self._clock() - self._stored_at > self._ttl:
                self._url, self._data, self._stored_at = "", None, 0.0
                return True
            return False

    def get(
        self, url: str, fetch: Callable[[], Any], *, fresh: bool = False,
    ) -> tuple[Any, bool]:
        """Return ``(document, from_cache)``; ``fetch`` runs on a miss.

        ``fetch`` is expected to store its own result (the backend's live
        ``get_object_info`` does); a fetch that does not is stored here.
        """
        if not fresh:
            cached = self.peek(url)
            if cached is not None:
                return cached, True
        data = fetch()
        if isinstance(data, dict) and self.peek(url) is not data:
            self.store(url, data)
        return data, False


class ObjectInfoCacheRegistry:
    """One ``ObjectInfoCache`` per API URL, shared by short-lived backends.

    At most ``max_urls`` endpoints are kept (least recently used first out)
    and expired snapshots are dropped whenever the registry is used, so a
    long-running process does not hold schemas of endpoints it stopped using.
    """

    def __init__(
        self,
        *,
        max_urls: int = DEFAULT_MAX_URLS,
        ttl: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_urls = max(1, int(max_urls))
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._caches: "OrderedDict[str, ObjectInfoCache]" = OrderedDict()

    def for_url(self, url: str) -> ObjectInfoCache:
        key = ObjectInfoCache._key(url)
        with self._lock:
            cache = self._caches.pop(key, None)
            for other_key in [k for k, other in self._caches.items() if other.expired()]:
                del self._caches[other_key]
            if cache is None:
                cache = ObjectInfoCache(ttl=self._ttl, clock=self._clock)
            self._caches[key] = cache
            while len(self._caches) > self._max_urls:
                self._caches.popitem(last=False)
            return cache

    def invalidate(self, url: Optional[str] = None) -> None:
        """Drop ``url``'s snapshot, or every snapshot when ``url`` is None."""
        with self._lock:
            if url is None:
                self._caches.clear()
            else:
                self._caches.pop(ObjectInfoCache._key(url), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._caches)


# Generation API ComfyUI profile jobs (ComfyUIBackend.generate_workflow).
EXTERNAL_OBJECT_INFO = ObjectInfoCacheRegistry()


__all__ = [
    "DEFAULT_MAX_URLS", "DEFAULT_TTL_SECONDS", "EXTERNAL_OBJECT_INFO",
    "ObjectInfoCache", "ObjectInfoCacheRegistry",
]
