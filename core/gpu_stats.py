"""GPU VRAM 사용량 — **장치 전체(모든 프로세스)** 기준.

왜 따로 있나: 하단 VRAM 바가 Forge 의 `/sdapi/v1/memory` 를 30초마다 읽었는데, 그 값은 Forge
자신이 잡은 메모리(체크포인트 ≈12GB)라 Ollama 가 27GB 를 올려도 12.1 에서 꿈쩍하지 않았다.
사용자가 기대하는 건 작업 관리자의 숫자 — 카드 전체가 얼마나 찼는가 — 이므로 NVML 로 직접 잰다.

우선순위: pynvml(NVML, 마이크로초) → `nvidia-smi` 프로세스 → None(호출자가 백엔드 값으로 대체).
Qt 에 의존하지 않는다 — tests/test_gpu_stats.py 가 가짜 NVML·가짜 nvidia-smi 로 검증한다.

호출 주기는 1초(generator_main 의 VRAM 타이머)다. 그래서 느린 폴백은 따로 묶는다.
- NVML 이 한 번 실패해도(TDR·드라이버 리셋) 영구 포기하지 않고 60초 뒤 다시 초기화한다.
- 그동안의 `nvidia-smi` 결과(실패 포함)는 5초간 재사용한다 — 매초 프로세스를 띄우지 않는다.
  Windows 에서는 CREATE_NO_WINDOW 로 콘솔 창이 깜빡이지 않게 한다.
- 백엔드 HTTP 폴백(read_backend_vram)도 5초에 한 번만 보낸다.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from typing import Any, Callable

NVML_RETRY_SECONDS = 60.0
SMI_CACHE_SECONDS = 5.0
BACKEND_FALLBACK_SECONDS = 5.0

_lock = threading.Lock()
_nvml_handle: Any = None
_nvml_initialised = False
_nvml_retry_at = 0.0          # monotonic — 이 시각 전에는 NVML 을 다시 두드리지 않는다
_smi_cache: tuple[Any, float, dict | None] | None = None      # (run, 시각, 결과)
_backend_cache: tuple[float, dict | None] | None = None       # (시각, 결과)


def _read_nvml(nvml: Any, now: float) -> dict | None:
    """pynvml 로 0번 장치의 used/total(bytes). 초기화는 한 번만, 실패하면 60초 뒤 재시도."""
    global _nvml_handle, _nvml_initialised, _nvml_retry_at
    if nvml is None or now < _nvml_retry_at:
        return None
    try:
        if _nvml_handle is None:
            nvml.nvmlInit()
            _nvml_initialised = True
            _nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        used, total = int(info.used), int(info.total)
        if total <= 0:
            return None
        return {"vram_used": used, "vram_total": total, "vram_free": max(0, total - used), "source": "nvml"}
    except Exception:
        _nvml_handle = None
        if _nvml_initialised:
            # 다음 재시도의 nvmlInit 이 참조 카운트만 쌓지 않게 정리한다(실패해도 무시).
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
            _nvml_initialised = False
        _nvml_retry_at = now + NVML_RETRY_SECONDS
        return None


def _no_window_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if sys.platform == "win32" else 0


def _read_nvidia_smi(run: Callable[..., Any], now: float) -> dict | None:
    """`nvidia-smi --query-gpu=memory.used,memory.total` (MiB) — NVML 이 없을 때. 5초 캐시."""
    global _smi_cache
    cached = _smi_cache
    if cached is not None and cached[0] is run and now - cached[1] < SMI_CACHE_SECONDS:
        return dict(cached[2]) if cached[2] else None
    result = _query_nvidia_smi(run)
    _smi_cache = (run, now, result)
    return dict(result) if result else None


def _query_nvidia_smi(run: Callable[..., Any]) -> dict | None:
    try:
        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 3}
        flags = _no_window_flags()
        if flags:
            kwargs["creationflags"] = flags
        proc = run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            **kwargs,
        )
        line = (proc.stdout or "").strip().splitlines()
        if proc.returncode != 0 or not line:
            return None
        used_mib, total_mib = (float(x.strip()) for x in line[0].split(",")[:2])
        if total_mib <= 0:
            return None
        used, total = int(used_mib * 1024 * 1024), int(total_mib * 1024 * 1024)
        return {"vram_used": used, "vram_total": total, "vram_free": max(0, total - used), "source": "nvidia-smi"}
    except Exception:
        return None


def _import_nvml() -> Any:
    try:
        import pynvml  # noqa: WPS433 — 선택 의존성
        return pynvml
    except Exception:
        return None


def read_vram(
    *,
    nvml: Any = "auto",
    run: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
) -> dict | None:
    """장치 전체 VRAM {vram_used, vram_total, vram_free, source} 또는 None."""
    module = _import_nvml() if nvml == "auto" else nvml
    with _lock:
        now = clock()
        return _read_nvml(module, now) or _read_nvidia_smi(run, now)


def read_backend_vram(
    fetch: Callable[[], dict | None],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict | None:
    """GPU 툴이 전혀 없을 때만 쓰는 백엔드 /memory 값 — HTTP 는 5초에 한 번만 보낸다.

    VRAM 바는 1초마다 값을 다시 emit 한다(새로 붙은 웹 클라이언트도 바를 채워야 한다).
    그 사이에는 마지막 값을 그대로 돌려준다. 결과에는 ``source='backend'`` 가 붙는다.
    """
    global _backend_cache
    now = clock()
    with _lock:
        cached = _backend_cache
        if cached is not None and now - cached[0] < BACKEND_FALLBACK_SECONDS:
            return dict(cached[1]) if cached[1] else None
    try:
        stats = fetch()
    except Exception:
        stats = None
    result = dict(stats, source="backend") if stats else None
    with _lock:
        _backend_cache = (now, result)
    return dict(result) if result else None


def reset_cache() -> None:
    """테스트용 — NVML 핸들·재시도 시각·폴백 캐시 초기화."""
    global _nvml_handle, _nvml_initialised, _nvml_retry_at, _smi_cache, _backend_cache
    with _lock:
        _nvml_handle = None
        _nvml_initialised = False
        _nvml_retry_at = 0.0
        _smi_cache = None
        _backend_cache = None
