"""하단 VRAM 바의 데이터 — 장치 전체 사용량을 NVML → nvidia-smi 순으로 읽는다."""
from __future__ import annotations

import types
import unittest.mock
import unittest

from core import gpu_stats


class _MemInfo:
    def __init__(self, used, total):
        self.used, self.total = used, total


def _fake_nvml(used, total, *, calls):
    mod = types.SimpleNamespace()
    mod.nvmlInit = lambda: calls.append("init")
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"
    mod.nvmlDeviceGetMemoryInfo = lambda h: (calls.append("mem"), _MemInfo(used, total))[1]
    return mod


def _fake_run(stdout, returncode=0):
    def run(cmd, **kw):
        assert cmd[0] == "nvidia-smi" and "--query-gpu=memory.used,memory.total" in cmd
        return types.SimpleNamespace(stdout=stdout, returncode=returncode)
    return run


class GpuStatsTests(unittest.TestCase):
    def setUp(self):
        gpu_stats.reset_cache()

    def test_nvml_is_device_wide_and_initialised_once(self):
        calls = []
        nvml = _fake_nvml(31_177 * 2**20, 32_607 * 2**20, calls=calls)
        first = gpu_stats.read_vram(nvml=nvml, run=_fake_run(""))
        second = gpu_stats.read_vram(nvml=nvml, run=_fake_run(""))
        self.assertEqual(first["source"], "nvml")
        self.assertEqual(first["vram_total"], 32_607 * 2**20)
        self.assertEqual(first["vram_used"], 31_177 * 2**20)
        self.assertEqual(first["vram_free"], (32_607 - 31_177) * 2**20)
        self.assertEqual(second["vram_used"], first["vram_used"])
        self.assertEqual(calls.count("init"), 1, "핸들은 한 번만 만든다 — 5초마다 nvmlInit 하지 않는다")
        self.assertEqual(calls.count("mem"), 2)

    def test_nvidia_smi_fallback_parses_mib(self):
        out = gpu_stats.read_vram(nvml=None, run=_fake_run("12345, 32607\n"))
        self.assertEqual(out["source"], "nvidia-smi")
        self.assertEqual(out["vram_used"], 12345 * 2**20)
        self.assertEqual(out["vram_total"], 32607 * 2**20)

    def test_broken_nvml_falls_back_and_is_not_retried(self):
        calls = []
        nvml = types.SimpleNamespace(
            nvmlInit=lambda: calls.append("init") or (_ for _ in ()).throw(RuntimeError("no driver")),
            nvmlDeviceGetHandleByIndex=lambda i: "h", nvmlDeviceGetMemoryInfo=lambda h: _MemInfo(1, 2),
        )
        out = gpu_stats.read_vram(nvml=nvml, run=_fake_run("100, 200"))
        self.assertEqual(out["source"], "nvidia-smi")
        gpu_stats.read_vram(nvml=nvml, run=_fake_run("100, 200"))
        self.assertEqual(calls.count("init"), 1, "고장난 NVML 은 다시 두드리지 않는다")

    def test_broken_nvml_recovers_after_the_retry_window(self):
        calls = []
        state = {"fail": True}

        def init():
            calls.append("init")
            if state["fail"]:
                raise RuntimeError("TDR")

        nvml = types.SimpleNamespace(
            nvmlInit=init,
            nvmlShutdown=lambda: calls.append("shutdown"),
            nvmlDeviceGetHandleByIndex=lambda i: "h",
            nvmlDeviceGetMemoryInfo=lambda h: _MemInfo(10, 20),
        )
        now = [1000.0]
        clock = lambda: now[0]
        run = _fake_run("100, 200")
        self.assertEqual(gpu_stats.read_vram(nvml=nvml, run=run, clock=clock)["source"], "nvidia-smi")
        now[0] += gpu_stats.NVML_RETRY_SECONDS - 1
        gpu_stats.read_vram(nvml=nvml, run=run, clock=clock)
        self.assertEqual(calls.count("init"), 1, "재시도 창 안에서는 NVML 을 다시 두드리지 않는다")
        state["fail"] = False
        now[0] += 2
        out = gpu_stats.read_vram(nvml=nvml, run=run, clock=clock)
        self.assertEqual(out["source"], "nvml", "드라이버가 돌아오면 NVML 로 복구한다")
        self.assertEqual(calls.count("init"), 2)

    def test_failure_after_successful_init_shuts_nvml_down_before_retrying(self):
        calls = []
        state = {"mem_fail": True}

        def memory(_handle):
            if state["mem_fail"]:
                raise RuntimeError("GPU lost")
            return _MemInfo(1, 4)

        nvml = types.SimpleNamespace(
            nvmlInit=lambda: calls.append("init"),
            nvmlShutdown=lambda: calls.append("shutdown"),
            nvmlDeviceGetHandleByIndex=lambda i: "h",
            nvmlDeviceGetMemoryInfo=memory,
        )
        now = [0.0]
        gpu_stats.read_vram(nvml=nvml, run=_fake_run("", returncode=1), clock=lambda: now[0])
        self.assertEqual(calls, ["init", "shutdown"])

    def test_nvidia_smi_is_spawned_at_most_once_per_cache_window(self):
        spawned = []

        def run(cmd, **kw):
            spawned.append(kw)
            return types.SimpleNamespace(stdout="100, 200", returncode=0)

        now = [50.0]
        clock = lambda: now[0]
        for _ in range(5):   # 1초 폴링 5번
            self.assertEqual(gpu_stats.read_vram(nvml=None, run=run, clock=clock)["source"], "nvidia-smi")
            now[0] += 1.0
        self.assertEqual(len(spawned), 1)
        now[0] += gpu_stats.SMI_CACHE_SECONDS
        gpu_stats.read_vram(nvml=None, run=run, clock=clock)
        self.assertEqual(len(spawned), 2)

    def test_failed_nvidia_smi_is_cached_too(self):
        spawned = []

        def run(cmd, **kw):
            spawned.append(cmd)
            raise FileNotFoundError("nvidia-smi")

        now = [0.0]
        for _ in range(3):
            self.assertIsNone(gpu_stats.read_vram(nvml=None, run=run, clock=lambda: now[0]))
            now[0] += 1.0
        self.assertEqual(len(spawned), 1)

    def test_nvidia_smi_hides_the_console_window_on_windows(self):
        seen = {}

        def run(cmd, **kw):
            seen.update(kw)
            return types.SimpleNamespace(stdout="1, 2", returncode=0)

        with unittest.mock.patch.object(gpu_stats.sys, "platform", "win32"), \
                unittest.mock.patch.object(gpu_stats.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True):
            gpu_stats.read_vram(nvml=None, run=run)
        self.assertEqual(seen.get("creationflags"), 0x08000000)

    def test_backend_fallback_http_is_rate_limited(self):
        fetched = []

        def fetch():
            fetched.append(1)
            return {"vram_used": 1, "vram_total": 2, "vram_free": 1}

        now = [10.0]
        clock = lambda: now[0]
        for _ in range(4):
            out = gpu_stats.read_backend_vram(fetch, clock=clock)
            self.assertEqual(out["source"], "backend")
            now[0] += 1.0
        self.assertEqual(len(fetched), 1, "1초 폴링마다 백엔드 HTTP 를 보내지 않는다")
        now[0] += gpu_stats.BACKEND_FALLBACK_SECONDS
        gpu_stats.read_backend_vram(fetch, clock=clock)
        self.assertEqual(len(fetched), 2)
        self.assertIsNone(gpu_stats.read_backend_vram(lambda: None, clock=lambda: 10_000.0))

    def test_nothing_available_returns_none(self):
        self.assertIsNone(gpu_stats.read_vram(nvml=None, run=_fake_run("", returncode=1)))
        self.assertIsNone(gpu_stats.read_vram(nvml=None, run=_fake_run("garbage")))

    def test_real_nvml_if_present_reports_sane_numbers(self):
        """실제 GPU 가 있으면 총량이 양수이고 사용량이 총량을 넘지 않는다 — 없으면 None 이어도 통과."""
        out = gpu_stats.read_vram()
        if out is None:
            self.skipTest("GPU/NVML 없음")
        self.assertGreater(out["vram_total"], 0)
        self.assertLessEqual(out["vram_used"], out["vram_total"])


if __name__ == "__main__":
    unittest.main()
