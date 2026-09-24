"""core.port_probe — 관리형 백엔드 포트 선택이 이미 쓰이는 포트를 비었다고 오판하지 않는지.

회귀: Windows 에서 SO_REUSEADDR 로 bind 를 시험해, SO_REUSEADDR 로 리슨 중인 Generation API
(기본 17860 = Forge 기본 포트)의 포트를 '비어 있음'으로 판정했다.
"""
from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

from core.port_probe import port_available

WINDOWS = os.name == "nt"


def _listener(host: str, *, reuse: bool = False) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if reuse:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen()
    return sock


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class PortProbeTests(unittest.TestCase):
    def test_free_port_is_available(self):
        self.assertTrue(port_available("127.0.0.1", _free_port()))

    def test_default_loopback_listener_is_busy(self):
        with _listener("127.0.0.1") as sock:
            self.assertFalse(port_available("127.0.0.1", sock.getsockname()[1]))

    @unittest.skipUnless(WINDOWS, "Windows SO_REUSEADDR 포트 공유 의미론")
    def test_reuseaddr_listener_is_busy_on_windows(self):
        with _listener("127.0.0.1", reuse=True) as sock:
            port = sock.getsockname()[1]
            # 예전 판정(SO_REUSEADDR bind)은 여기서 True 였다.
            self.assertFalse(port_available("127.0.0.1", port))

    @unittest.skipUnless(WINDOWS, "Windows 는 특정 주소 bind 가 와일드카드 리스너와 겹쳐도 성공한다")
    def test_wildcard_listener_blocks_loopback_choice_on_windows(self):
        with _listener("0.0.0.0") as sock:
            self.assertFalse(port_available("127.0.0.1", sock.getsockname()[1]))

    def test_runtime_adapter_uses_the_shared_probe(self):
        from core.backend_runtime import LocalRuntimeAdapter

        adapter = LocalRuntimeAdapter()
        with _listener("127.0.0.1", reuse=WINDOWS) as sock:
            self.assertFalse(adapter.port_available("127.0.0.1", sock.getsockname()[1]))
        self.assertTrue(adapter.port_available("127.0.0.1", _free_port()))


class GenerationApiServerPortTests(unittest.TestCase):
    def test_running_generation_api_port_is_not_offered_to_managed_backends(self):
        from core.generation_api import GenerationApiManager
        from core.resource_coordinator import GenerationResourceCoordinator

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = GenerationApiManager(
                config_path=root / "generation_api.json",
                storage_root=root / "results",
                target_factory=lambda profile: None,
                coordinator=GenerationResourceCoordinator(),
            )
            try:
                port = _free_port()
                manager.save_config({
                    "enabled": True, "bindHost": "127.0.0.1", "port": port,
                    "token": "port-probe-token-0123456789", "targets": [],
                })
                manager.start(persist_enabled=False)
                self.assertFalse(port_available("127.0.0.1", port))
                if WINDOWS:
                    # 배타 bind 라 다른 SO_REUSEADDR 소켓이 같은 포트를 가로채지 못한다.
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as thief:
                        thief.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        with self.assertRaises(OSError):
                            thief.bind(("127.0.0.1", port))
                manager.stop(persist_enabled=False)
                self.assertTrue(port_available("127.0.0.1", port))
            finally:
                manager.shutdown()


class GenerationApiReservedPortTests(unittest.TestCase):
    """core.generation_api_port — 관리형 엔진 포트 선택이 켜 둔 API 포트를 bind 전에도 피한다."""

    def test_enabled_listen_port(self):
        from core.generation_api_port import DEFAULT_PORT, enabled_listen_port

        cases = [
            ({"enabled": True, "port": 17860}, 17860),
            ({"enabled": True, "port": "18001"}, 18001),
            ({"enabled": True}, DEFAULT_PORT),          # 매니저도 포트가 없으면 기본값으로 뜬다
            ({"enabled": False, "port": 17860}, None),  # 꺼 둔 설정은 포트를 쓰지 않는다
            ({"port": 17860}, None),
            ({"enabled": True, "port": 80}, None),      # 매니저가 거부하고 기본값(꺼짐)으로 뜬다
            ({"enabled": True, "port": 70000}, None),
            ({"enabled": True, "port": "abc"}, None),
            ({"enabled": True, "port": True}, None),
            ([], None),
            (None, None),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(enabled_listen_port(raw), expected)

    def test_reserved_ports_reads_the_saved_config(self):
        from core.generation_api_port import reserved_ports

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "generation_api.json"
            self.assertEqual(reserved_ports(path), frozenset(), "설정 파일이 없으면 예약 없음")
            path.write_text('{"enabled": true, "port": 17860}', encoding="utf-8")
            self.assertEqual(reserved_ports(path), frozenset({17860}))
            path.write_text('{"enabled": false, "port": 17860}', encoding="utf-8")
            self.assertEqual(reserved_ports(path), frozenset())
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(reserved_ports(path), frozenset())
            self.assertTrue(path.exists(), "읽기 전용 — 깨진 파일을 옮기거나 지우지 않는다")

    def test_generation_api_shares_the_default_port_and_config_path(self):
        import core.generation_api as generation_api
        import core.generation_api_port as generation_api_port

        self.assertEqual(generation_api.DEFAULT_PORT, generation_api_port.DEFAULT_PORT)
        project_root = Path(generation_api.__file__).resolve().parent.parent
        self.assertEqual(
            generation_api_port.default_config_path(),
            project_root / "user_data" / "generation_api.json",
        )

    def test_saved_enabled_api_config_is_reserved_end_to_end(self):
        """GenerationApiManager 가 저장한 설정 그대로 예약된다(키 이름·형태 계약)."""
        from core.generation_api import GenerationApiManager
        from core.generation_api_port import reserved_ports
        from core.resource_coordinator import GenerationResourceCoordinator

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = GenerationApiManager(
                config_path=root / "generation_api.json",
                storage_root=root / "results",
                target_factory=lambda profile: None,
                coordinator=GenerationResourceCoordinator(),
            )
            try:
                manager.save_config({
                    "enabled": True, "bindHost": "127.0.0.1", "port": 17861,
                    "token": "port-probe-token-0123456789", "targets": [],
                })
                self.assertEqual(reserved_ports(root / "generation_api.json"), frozenset({17861}))
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
