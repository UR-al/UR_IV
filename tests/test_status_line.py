"""show_status 의 실제 표시 경로 — core.status_message 페이로드와 ui.status_line 전달.

Vue 전환 뒤 show_status 는 모든 속성이 no-op 인 더미 라벨(status_message_label = _D())에만 써서,
79곳의 진행·완료·실패 문구(클립보드 복사·설정 저장·모델 언로드 실패 등)가 화면 어디에도 나오지
않고 호출마다 쓸모없는 5초 QTimer 만 생겼다. 지금은 statusMessage 시그널로 Vue 하단 계기 스트립에
한 줄을 보이고, 놓치면 안 되는 결과만 호출처가 notify_user 로 토스트를 따로 띄운다.
"""
from __future__ import annotations

import io
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from core.status_message import (
    DEFAULT_TIMEOUT_MS,
    MAX_STATUS_CHARS,
    MAX_TIMEOUT_MS,
    build_status_payload,
    infer_status_level,
)

try:
    # ui.generator_main(→ config)은 QtWebEngine 을 끌어오므로 QCoreApplication 보다 먼저 import 해야 한다.
    from ui.generator_main import GeneratorMainUI
    from ui.vue_bridge import VueBridge
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - PyQt6 없는 환경
    GeneratorMainUI = VueBridge = None
    _IMPORT_ERROR = exc


class InferStatusLevelTests(unittest.TestCase):
    def test_levels_from_the_message_markers(self):
        cases = {
            "❌ 설정 불러오기 실패: x": "error",
            "Model unload failed.": "error",
            "완료 12장 — 실패 3건": "error",       # 실패가 완료보다 우선
            "⚠ 덱이 비었습니다": "warning",
            "✅ 설정이 저장되었습니다.": "success",
            "이미지 생성 완료!": "success",
            "🎨 생성 중... 3/20 steps": "info",
            "": "info",
        }
        for text, level in cases.items():
            with self.subTest(text=text):
                self.assertEqual(infer_status_level(text), level)


class BuildStatusPayloadTests(unittest.TestCase):
    def test_payload_shape_and_single_line(self):
        payload = build_status_payload("  첫 줄\n  둘째   줄 ", 3000, now_ms=1234)
        self.assertEqual(payload, {"text": "첫 줄 둘째 줄", "level": "info", "timeoutMs": 3000, "at": 1234})

    def test_blank_messages_send_nothing(self):
        for message in ("", "   \n\t", None):
            with self.subTest(message=message):
                self.assertIsNone(build_status_payload(message))

    def test_paths_are_masked_and_urls_kept(self):
        payload = build_status_payload(r"Export failed: C:\Users\me\out\x.parquet", now_ms=1)
        self.assertEqual(payload["text"], "Export failed: [path]")
        self.assertEqual(payload["level"], "error")
        url = build_status_payload("❌ Forge 연결 실패: http://127.0.0.1:7860/sdapi/v1/options", now_ms=1)
        self.assertIn("http://127.0.0.1:7860/sdapi/v1/options", url["text"])

    def test_length_is_limited(self):
        payload = build_status_payload("x" * 1000, now_ms=1)
        self.assertEqual(len(payload["text"]), MAX_STATUS_CHARS)

    def test_timeouts_are_normalised(self):
        cases = {0: 0, -5: 0, 2000: 2000, 10 ** 9: MAX_TIMEOUT_MS, "abc": DEFAULT_TIMEOUT_MS, None: DEFAULT_TIMEOUT_MS}
        for given, expected in cases.items():
            with self.subTest(given=given):
                self.assertEqual(build_status_payload("m", given, now_ms=1)["timeoutMs"], expected)

    def test_explicit_level_wins_and_unknown_level_falls_back_to_inference(self):
        self.assertEqual(build_status_payload("✅ ok", level="warning", now_ms=1)["level"], "warning")
        self.assertEqual(build_status_payload("✅ ok", level="loud", now_ms=1)["level"], "success")

    def test_at_defaults_to_now(self):
        before = int(time.time() * 1000)
        at = build_status_payload("m")["at"]
        self.assertGreaterEqual(at, before)
        self.assertLessEqual(at, int(time.time() * 1000))


class PublishStatusWithoutQtTests(unittest.TestCase):
    def test_only_problems_are_logged_with_the_original_text(self):
        """진행·완료 문구(스텝마다 한 줄)는 로그를 덮지 않고, 경고·오류만 원문(경로 포함)으로 남는다."""
        from ui.status_line import publish_status

        with self.assertLogs("status", level="DEBUG") as logs:
            publish_status(SimpleNamespace(), "🎨 생성 중... 3/20 steps")
            publish_status(SimpleNamespace(), "✅ 저장")
            publish_status(SimpleNamespace(), "   ")      # 빈 문구는 아무것도 하지 않는다
            publish_status(SimpleNamespace(), r"Export failed: C:\out\x.parquet")
            publish_status(SimpleNamespace(), "덱 확인", level="warning")
        self.assertEqual(logs.output, [
            r"WARNING:status:[Status] Export failed: C:\out\x.parquet",
            "WARNING:status:[Status] 덱 확인",
        ])

    def test_console_encoding_errors_never_reach_the_caller(self):
        """show_status 는 except 블록·생성 경로에서도 불린다 — cp949 stdout 의 이모지가 호출자를 깨면 안 된다."""
        import codecs
        import logging
        from ui.status_line import publish_status

        self.assertRaises(UnicodeEncodeError, codecs.encode, "❌", "cp949")
        cp949 = io.TextIOWrapper(io.BytesIO(), encoding="cp949", errors="strict")
        handler = logging.StreamHandler(cp949)   # cp949 로 리다이렉트된 콘솔 핸들러 흉내
        status_logger = logging.getLogger("status")
        status_logger.addHandler(handler)
        self.addCleanup(status_logger.removeHandler, handler)
        # logging 의 handleError 도 같은 stderr 에 이모지가 든 repr 을 쓰다 UnicodeEncodeError 를 낸다
        with mock.patch("sys.stdout", cp949), mock.patch("sys.stderr", cp949):
            publish_status(SimpleNamespace(), "❌ 🎨 실패")
            publish_status(SimpleNamespace(), "✅ 🎨 완료")

    def test_notify_user_masks_paths_and_normalises_level(self):
        from ui.status_line import notify_user

        toasts = []
        host = SimpleNamespace(vue_bridge=SimpleNamespace(
            showNotification=SimpleNamespace(emit=lambda level, text: toasts.append((level, text)))))
        notify_user(host, "error", r"검색 결과 불러오기 실패: D:\data\x.parquet")
        notify_user(host, "loud", "정보")
        notify_user(host, "info", "")                      # 빈 문구는 토스트하지 않는다
        notify_user(SimpleNamespace(), "error", "브리지 없음")   # 조용히 넘어간다
        self.assertEqual(toasts, [("error", "검색 결과 불러오기 실패: [path]"), ("info", "정보")])


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class PublishStatusToVueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtCore import QCoreApplication

        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.bridge = VueBridge()
        self.received = []
        self.bridge.statusMessage.connect(self.received.append)
        self.host = SimpleNamespace(vue_bridge=self.bridge)

    def test_gui_thread_emits_directly_and_keeps_the_last_line_for_late_clients(self):
        from ui.status_line import publish_status

        self.assertEqual(self.bridge.getStatusMessage(), "{}")
        with mock.patch("sys.stdout", io.StringIO()):
            publish_status(self.host, "📋 Copied to clipboard.", 2000)
        self.assertEqual(len(self.received), 1)
        payload = json.loads(self.received[0])
        self.assertEqual((payload["text"], payload["level"], payload["timeoutMs"]),
                         ("📋 Copied to clipboard.", "info", 2000))
        self.assertEqual(json.loads(self.bridge.getStatusMessage()), payload)

    def test_show_status_keeps_its_one_argument_signature_and_reaches_vue(self):
        """테스트 스텁(show_status(self, message))과 같은 호출 모양으로 계기 스트립에 닿는다."""
        with mock.patch("sys.stdout", io.StringIO()):
            GeneratorMainUI.show_status(self.host, "Model unload failed.")
            GeneratorMainUI.show_status(self.host, "❌ 설정 불러오기 실패: x", 0)
        first, second = (json.loads(item) for item in self.received)
        self.assertEqual((first["level"], first["timeoutMs"]), ("error", DEFAULT_TIMEOUT_MS))
        self.assertEqual(second["timeoutMs"], 0, "timeout_ms=0 은 다음 문구까지 유지")

    def test_worker_thread_is_relayed_to_the_gui_thread(self):
        """QWebChannel 에 등록된 bridge 시그널은 GUI 스레드에서만 emit 한다 — 워커는 중계 객체로 넘긴다."""
        from PyQt6.QtCore import QThread
        from ui.status_line import publish_status

        main_thread = QThread.currentThread()
        delivered_on = []
        self.bridge.statusMessage.connect(lambda _json: delivered_on.append(QThread.currentThread() is main_thread))
        errors = []

        def work():
            try:
                with mock.patch("sys.stdout", io.StringIO()):
                    publish_status(self.host, "워커에서 보낸 문구")
            except Exception as exc:  # pragma: no cover - 실패 진단용
                errors.append(exc)

        worker = threading.Thread(target=work)
        worker.start()
        worker.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(self.received, [], "워커 스레드에서 bridge 시그널을 직접 emit 하지 않는다")
        deadline = time.monotonic() + 5
        while not self.received and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(json.loads(self.received[0])["text"], "워커에서 보낸 문구")
        self.assertEqual(delivered_on, [True])
        self.assertEqual(json.loads(self.bridge.getStatusMessage())["text"], "워커에서 보낸 문구")


if __name__ == "__main__":
    unittest.main()
