"""core.backend_probe — 시작 게이트 probe 는 병렬로, 루프백은 connect 만 짧게."""
from __future__ import annotations

import socket
import threading
import time
import unittest

from core.backend_probe import (
    HEALTH_PATHS,
    LOOPBACK_CONNECT_TIMEOUT,
    probe_backends,
    probe_targets,
    probe_url,
)


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _RecordingGet:
    def __init__(self, ok_urls=(), delay=0.0):
        self.ok_urls = set(ok_urls)
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, timeout=None):
        with self.lock:
            self.calls.append((url, timeout))
        if self.delay:
            time.sleep(self.delay)
        if url in self.ok_urls:
            return _Response(200)
        raise ConnectionError("refused")


class ProbeTargetTests(unittest.TestCase):
    def test_localhost_is_checked_on_both_loopback_families(self):
        targets, loopback = probe_targets("http://localhost:7860/")
        self.assertTrue(loopback)
        self.assertEqual(["http://127.0.0.1:7860/", "http://[::1]:7860/"], targets)

    def test_numeric_loopback_is_kept_and_remote_is_not_loopback(self):
        self.assertEqual((["http://127.0.0.1:8188"], True), probe_targets("http://127.0.0.1:8188"))
        self.assertEqual((["http://[::1]:8188"], True), probe_targets("http://[::1]:8188"))
        self.assertEqual((["http://192.168.0.10:7860"], False), probe_targets("http://192.168.0.10:7860"))
        self.assertEqual(([], False), probe_targets("  "))

    def test_https_localhost_keeps_its_host_for_tls_verification(self):
        """예전: https://localhost 도 127.0.0.1/::1 로 바꿔 localhost 전용 인증서가 IP 불일치로 실패."""
        self.assertEqual((["https://localhost:7860/"], True), probe_targets("https://localhost:7860/"))
        self.assertEqual((["HTTPS://LOCALHOST:7860"], True), probe_targets("HTTPS://LOCALHOST:7860"))
        # 대문자 http 는 여전히 두 루프백으로 나눈다(scheme 은 대소문자 무관).
        self.assertEqual(
            (["http://127.0.0.1:7860", "http://[::1]:7860"], True), probe_targets("HTTP://LOCALHOST:7860")
        )


class ProbeUrlTests(unittest.TestCase):
    def test_https_localhost_is_probed_on_the_original_host_with_short_connect(self):
        get = _RecordingGet(ok_urls={"https://localhost:8188/system_stats"})
        self.assertTrue(probe_url("https://localhost:8188", "/system_stats", get=get))
        self.assertEqual([("https://localhost:8188/system_stats", (LOOPBACK_CONNECT_TIMEOUT, 2.0))], get.calls)

    def test_loopback_uses_short_connect_timeout_but_keeps_read_timeout(self):
        get = _RecordingGet(ok_urls={"http://127.0.0.1:7860/sdapi/v1/samplers"})
        self.assertTrue(probe_url("http://127.0.0.1:7860", "/sdapi/v1/samplers", get=get))
        self.assertEqual([("http://127.0.0.1:7860/sdapi/v1/samplers", (LOOPBACK_CONNECT_TIMEOUT, 2.0))], get.calls)

    def test_remote_urls_keep_the_full_timeout(self):
        get = _RecordingGet()
        self.assertFalse(probe_url("http://10.0.0.5:7860", "/x", get=get))
        self.assertEqual([("http://10.0.0.5:7860/x", 2.0)], get.calls)

    def test_localhost_succeeds_when_only_ipv6_answers(self):
        get = _RecordingGet(ok_urls={"http://[::1]:8188/system_stats"})
        self.assertTrue(probe_url("http://localhost:8188", "/system_stats", get=get))

    def test_non_200_and_empty_urls_are_failures(self):
        self.assertFalse(probe_url("", "/x", get=_RecordingGet()))
        self.assertFalse(probe_url("http://127.0.0.1:1", "/x", get=lambda url, timeout=None: _Response(500)))

    def test_closed_real_loopback_port_is_rejected_quickly(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        started = time.monotonic()
        self.assertFalse(probe_url(f"http://localhost:{port}", "/system_stats"))
        # 예전: Windows 에서 localhost 닫힌 포트는 약 4초(::1 → 127.0.0.1 순차 2초씩).
        self.assertLess(time.monotonic() - started, 1.5)


class ProbeBackendsTests(unittest.TestCase):
    def test_backends_are_probed_in_parallel_with_engine_health_paths(self):
        get = _RecordingGet(ok_urls={"http://10.0.0.2:8188" + HEALTH_PATHS["comfyui"]}, delay=0.3)
        started = time.monotonic()
        found = probe_backends({"forge": "http://10.0.0.1:7860", "comfyui": "http://10.0.0.2:8188"}, get=get)
        elapsed = time.monotonic() - started
        self.assertEqual({"forge": False, "comfyui": True}, found)
        self.assertLess(elapsed, 0.55, "두 probe 가 순차로 돌면 0.6초 이상 걸린다")
        self.assertEqual(
            {"http://10.0.0.1:7860/sdapi/v1/samplers", "http://10.0.0.2:8188/system_stats"},
            {url for url, _timeout in get.calls},
        )

    def test_runtime_definitions_share_the_same_health_paths(self):
        from core.backend_runtime import ENGINE_DEFINITIONS

        for engine, definition in ENGINE_DEFINITIONS.items():
            with self.subTest(engine=engine):
                self.assertEqual(HEALTH_PATHS[engine], definition.health_path)


class GatePairTests(unittest.TestCase):
    def test_gate_pair_maps_engines_and_swallows_errors(self):
        from unittest import mock

        from ui.generator_webui import WebUIMixin

        with mock.patch("core.backend_probe.probe_backends", return_value={"forge": True, "comfyui": False}) as probe:
            self.assertEqual((True, False), WebUIMixin._probe_pair("http://a", "http://b"))
        probe.assert_called_once_with({"forge": "http://a", "comfyui": "http://b"})
        with mock.patch("core.backend_probe.probe_backends", side_effect=RuntimeError("boom")):
            self.assertEqual((False, False), WebUIMixin._probe_pair("http://a", "http://b"))

    def test_async_pair_hands_result_to_the_scheduler_and_drops_stale_checks(self):
        from unittest import mock

        from ui.generator_webui import WebUIMixin

        def blocking_schedule(ms, fn):
            time.sleep(ms / 1000.0)
            fn()

        delivered = []
        with mock.patch.object(WebUIMixin, "_probe_pair", return_value=(False, True)):
            WebUIMixin._probe_pair_async(
                "http://a", "http://b", is_current=lambda: True,
                on_result=lambda *result: delivered.append(result), schedule=blocking_schedule,
            )
            self.assertEqual([(False, True)], delivered)
            WebUIMixin._probe_pair_async(
                "http://a", "http://b", is_current=lambda: False,
                on_result=lambda *result: delivered.append(result), schedule=blocking_schedule,
            )
        self.assertEqual([(False, True)], delivered, "늦게 끝난 옛 검사는 버린다")

    def test_gate_and_emergency_dialog_share_one_probe_path(self):
        """예전 순차 probe(_quick_test)는 두 경로가 _probe_pair_async 로 옮긴 뒤 호출자가 없었다."""
        import inspect

        from ui import generator_webui
        from ui.generator_webui import WebUIMixin

        self.assertFalse(hasattr(WebUIMixin, "_quick_test"))
        source = inspect.getsource(generator_webui)
        self.assertNotIn("_quick_test", source)
        self.assertIn("_probe_pair_async(", inspect.getsource(WebUIMixin._probe_backends_async))


if __name__ == "__main__":
    unittest.main()
