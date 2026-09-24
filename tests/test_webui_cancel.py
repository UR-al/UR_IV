"""WebUI(Forge) 취소 — 우리 작업이 active 일 때만 전역 interrupt, 모델 전환 중 취소 보존.

감사 #31: GUI 워커가 cancel_check 를 넘기지 않아 체크포인트 전환 중·발송 직후의 취소가
사라졌다. cancel_check 를 넘기되, 반복 전역 interrupt 가 외부 Forge 클라이언트의 작업을
끊지 않도록 force_task_id + /internal/progress 로 '우리 작업이 도는 동안'에만 보낸다.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import unittest
from unittest import mock

from PIL import Image

from backends.webui_backend import WebUIBackend
from core.webui_cancel import (
    TASK_ACTIVE, TASK_COMPLETED, TASK_QUEUED, TASK_TRANSIENT, TASK_UNKNOWN, TASK_UNSEEN,
    UNSEEN_GRACE_MAX_SECONDS, UNSEEN_GRACE_SECONDS,
    TaskInterruptPolicy, approx_payload_bytes, parse_task_state, unseen_grace_seconds, with_task_id,
)


def _png_b64() -> str:
    out = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeForge:
    """requests.post 대역 — 생성 요청은 interrupt 가 올 때까지(또는 타임아웃) 막힌다."""

    def __init__(self, states, *, progress_status=200):
        self.states = list(states)
        self.progress_status = progress_status
        self.generation_json = None
        self.progress_bodies = []
        self.interrupts = []          # 각 interrupt 시점에 보고된 마지막 상태
        self.last_state = None
        self.released = threading.Event()
        self.generation_started = threading.Event()

    def post(self, url, json=None, **_kwargs):
        if url.endswith("/sdapi/v1/txt2img"):
            self.generation_json = dict(json or {})
            self.generation_started.set()
            self.released.wait(5)
            return _Response({"images": [_png_b64()], "info": "{}"})
        if url.endswith("/internal/progress"):
            self.progress_bodies.append(dict(json or {}))
            if self.progress_status != 200:
                return _Response({}, status_code=self.progress_status)
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if state == "error":              # 타임아웃·연결 끊김
                raise ConnectionError("progress poll reset")
            if state == "http500":
                return _Response({}, status_code=500)
            self.last_state = state
            return _Response({
                "active": state == "active", "queued": state == "queued",
                "completed": state == "completed",
            })
        if url.endswith("/sdapi/v1/interrupt"):
            self.interrupts.append(self.last_state)
            self.released.set()
            return _Response({})
        raise AssertionError(f"unexpected POST {url}")


def _backend() -> WebUIBackend:
    backend = WebUIBackend("http://127.0.0.1:7860")
    backend._switch_model_if_needed = lambda *_a, **_k: None
    return backend


class TaskStateTests(unittest.TestCase):
    def test_parse_task_state(self):
        self.assertEqual(parse_task_state({"active": True, "queued": False, "completed": False}), TASK_ACTIVE)
        self.assertEqual(parse_task_state({"active": False, "queued": True, "completed": False}), TASK_QUEUED)
        self.assertEqual(parse_task_state({"active": False, "queued": False, "completed": True}), TASK_COMPLETED)
        self.assertEqual(parse_task_state({"active": False, "queued": False, "completed": False}), TASK_UNSEEN)
        self.assertEqual(parse_task_state({"detail": "Not Found"}), TASK_UNKNOWN)
        self.assertEqual(parse_task_state(None), TASK_UNKNOWN)

    def test_policy_interrupts_only_our_running_job(self):
        policy = TaskInterruptPolicy(grace_seconds=3)
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 0.1))   # 아직 큐 등록 전
        self.assertFalse(policy.should_interrupt(TASK_QUEUED, 0.5))   # 남의 작업이 도는 중
        self.assertTrue(policy.should_interrupt(TASK_ACTIVE, 1.0))
        self.assertFalse(policy.should_interrupt(TASK_COMPLETED, 1.5))
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 10))    # 봤다가 사라짐 → 대상 없음

    def test_policy_falls_back_for_servers_without_task_tracking(self):
        self.assertTrue(TaskInterruptPolicy().should_interrupt(TASK_UNKNOWN, 0))
        policy = TaskInterruptPolicy(grace_seconds=3)
        # 유예는 '처음 unseen 을 본 시각'부터 (now 는 단조 시계 값)
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 100.0))
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 102.9))
        self.assertTrue(policy.should_interrupt(TASK_UNSEEN, 103.0))  # force_task_id 무시하는 옛 서버

    def test_unseen_grace_starts_at_first_poll_not_at_dispatch(self):
        # 취소가 발송 30초 뒤에 와도, 첫 unseen 부터 유예를 다시 잰다(발송 시각은 본문 업로드 전)
        policy = TaskInterruptPolicy(grace_seconds=3)
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 5000.0))
        self.assertFalse(policy.should_interrupt(TASK_TRANSIENT, 5001.0))
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 5002.0))
        self.assertFalse(policy.should_interrupt(TASK_QUEUED, 5004.0))  # 큐에 보였다 → 옛 서버 아님
        self.assertFalse(policy.should_interrupt(TASK_UNSEEN, 5100.0))

    def test_single_transient_failure_never_interrupts_a_queued_job(self):
        # 남의 작업 뒤에 줄 서 있는 동안 조회가 한 번(또는 계속) 실패해도 전역 interrupt 금지
        policy = TaskInterruptPolicy(grace_seconds=3, transient_limit=5)
        self.assertFalse(policy.should_interrupt(TASK_QUEUED, 1.0))
        for tick in range(20):
            self.assertFalse(policy.should_interrupt(TASK_TRANSIENT, 2.0 + tick))
        self.assertTrue(policy.should_interrupt(TASK_ACTIVE, 30.0))
        # 우리 작업이 돌던 중의 일시 실패 → 계속 보낸다(HTTP 가 살아 있으니 아직 우리 작업)
        self.assertTrue(policy.should_interrupt(TASK_TRANSIENT, 30.5))
        self.assertFalse(policy.should_interrupt(TASK_COMPLETED, 31.0))
        self.assertFalse(policy.should_interrupt(TASK_TRANSIENT, 31.5))

    def test_never_answered_endpoint_falls_back_only_after_consecutive_failures(self):
        policy = TaskInterruptPolicy(grace_seconds=3, transient_limit=3)
        self.assertFalse(policy.should_interrupt(TASK_TRANSIENT, 0.0))
        self.assertFalse(policy.should_interrupt(TASK_TRANSIENT, 0.3))
        self.assertTrue(policy.should_interrupt(TASK_TRANSIENT, 0.6))
        self.assertTrue(policy.should_interrupt(TASK_TRANSIENT, 0.9))
        # 한 번이라도 응답을 받으면 그 상태를 따르고, 실패 횟수는 다시 센다
        fresh = TaskInterruptPolicy(grace_seconds=3, transient_limit=3)
        fresh.should_interrupt(TASK_TRANSIENT, 0.0)
        fresh.should_interrupt(TASK_TRANSIENT, 0.3)
        self.assertFalse(fresh.should_interrupt(TASK_UNSEEN, 0.6))
        self.assertEqual(fresh.transient_failures, 0)
        self.assertFalse(fresh.should_interrupt(TASK_TRANSIENT, 0.9))

    def test_unseen_grace_grows_with_request_body(self):
        self.assertEqual(unseen_grace_seconds(0), UNSEEN_GRACE_SECONDS)
        eight_mb = 8 * 1024 * 1024
        self.assertAlmostEqual(unseen_grace_seconds(eight_mb), UNSEEN_GRACE_SECONDS + 4.0)
        self.assertEqual(unseen_grace_seconds(10 ** 12), UNSEEN_GRACE_MAX_SECONDS)
        payload = {"prompt": "abc", "init_images": ["x" * 1000, "y" * 500],
                   "alwayson_scripts": {"cn": {"args": [{"image": "z" * 250}, 1, None]}}}
        size = approx_payload_bytes(payload)
        self.assertGreaterEqual(size, 1000 + 500 + 250 + 3)
        self.assertLess(size, 2000)

    def test_with_task_id_keeps_caller_id_and_does_not_mutate(self):
        source = {"prompt": "x"}
        out, task_id = with_task_id(source)
        self.assertNotIn("force_task_id", source)
        self.assertEqual(out["force_task_id"], task_id)
        self.assertTrue(task_id.startswith("task("))
        out2, task_id2 = with_task_id({"force_task_id": "task(client-1)"})
        self.assertEqual((out2["force_task_id"], task_id2), ("task(client-1)", "task(client-1)"))


class WebUICancelIntegrationTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "core.forge_output_policy.forge_save_outputs_setting", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, forge, cancel_after_start=True, *, use_interrupt=False):
        backend = _backend()
        cancelled = threading.Event()
        results = []
        with mock.patch("backends.webui_backend.requests.post", side_effect=forge.post):
            thread = threading.Thread(target=lambda: results.append(backend.txt2img(
                "", {"prompt": "p", "save_images": True},
                cancel_check=None if use_interrupt else cancelled.is_set)))
            thread.start()
            self.assertTrue(forge.generation_started.wait(3))
            if cancel_after_start:
                if use_interrupt:
                    backend.interrupt()
                else:
                    cancelled.set()
            thread.join(8)
        self.assertFalse(thread.is_alive())
        return backend, results[0]

    def test_queued_behind_external_job_waits_until_ours_is_active(self):
        forge = _FakeForge(["queued", "queued", "active"])
        _backend_obj, _result = self._run(forge)
        self.assertEqual(forge.interrupts[:1], ["active"])   # 남의 작업(queued 동안)은 끊지 않았다
        task_id = forge.generation_json["force_task_id"]
        self.assertTrue(all(body["id_task"] == task_id for body in forge.progress_bodies))

    def test_interrupt_method_is_targeted_too(self):
        forge = _FakeForge(["queued", "active"])
        backend, _result = self._run(forge, use_interrupt=True)
        self.assertEqual(forge.interrupts[:1], ["active"])
        self.assertFalse(backend._generation_inflight)
        self.assertEqual(backend._inflight_interrupts, {})

    def test_server_without_progress_endpoint_uses_legacy_global_interrupt(self):
        forge = _FakeForge(["active"], progress_status=404)
        backend, _result = self._run(forge)
        self.assertTrue(forge.interrupts)
        self.assertIs(backend._task_progress_supported, False)
        self.assertEqual(len(forge.progress_bodies), 1)   # 404 뒤로는 다시 묻지 않는다

    def test_no_cancel_means_no_progress_queries_or_interrupts(self):
        forge = _FakeForge(["active"])
        forge.released.set()
        _backend_obj, result = self._run(forge, cancel_after_start=False)
        self.assertTrue(result.success)
        self.assertEqual(forge.progress_bodies, [])
        self.assertEqual(forge.interrupts, [])

    def test_cancel_during_model_switch_never_dispatches(self):
        backend = WebUIBackend("http://127.0.0.1:7860")
        cancelled = threading.Event()
        backend._switch_model_if_needed = lambda *_a, **_k: cancelled.set()   # 전환 중 취소
        with mock.patch("backends.webui_backend.requests.post") as post:
            result = backend.txt2img("model.safetensors", {"prompt": "p"}, cancel_check=cancelled.is_set)
        self.assertFalse(result.success)
        self.assertIn("취소", result.error)
        post.assert_not_called()

    def test_transient_poll_failure_while_queued_never_interrupts_the_external_job(self):
        # 외부 작업 뒤에 줄 선 동안 /internal/progress 가 연결 끊김·500 으로 실패해도 전역
        # interrupt 금지 — 예전엔 한 번의 실패가 UNKNOWN 으로 취급돼 남의 작업을 끊었다
        forge = _FakeForge(["queued", "error", "http500", "error", "queued", "http500", "active"])
        backend, _result = self._run(forge)
        self.assertEqual(forge.interrupts[:1], ["active"])
        self.assertIs(backend._task_progress_supported, True)   # 일시 실패로 지원 여부를 잃지 않는다

    def test_large_body_widens_the_unseen_grace(self):
        forge = _FakeForge(["active"])
        backend = _backend()
        captured = []
        real_policy = TaskInterruptPolicy

        def capture(*args, **kwargs):
            policy = real_policy(*args, **kwargs)
            captured.append(policy.grace_seconds)
            return policy

        cancelled = threading.Event()
        # 큰 base64 필드(ControlNet 이미지 등) — 가짜 서버는 txt2img 만 받으므로 txt2img 로 보낸다
        big = {"prompt": "p", "alwayson_scripts": {"cn": {"args": [{"image": "x" * (6 * 1024 * 1024)}]}}}
        results = []
        with mock.patch("backends.webui_backend.requests.post", side_effect=forge.post), \
             mock.patch("core.webui_cancel.TaskInterruptPolicy", side_effect=capture):
            thread = threading.Thread(target=lambda: results.append(
                backend.txt2img("", big, cancel_check=cancelled.is_set)))
            thread.start()
            self.assertTrue(forge.generation_started.wait(3))
            cancelled.set()
            thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(captured), 1)
        self.assertAlmostEqual(captured[0], unseen_grace_seconds(approx_payload_bytes(big)), places=3)
        self.assertGreater(captured[0], UNSEEN_GRACE_SECONDS + 2.9)


class TaskStateProbeTests(unittest.TestCase):
    """_task_state — '엔드포인트 없음'(UNKNOWN)과 '일시 실패'(TRANSIENT)를 나눈다."""

    def _state(self, backend, response=None, error=None):
        def post(*_a, **_k):
            if error is not None:
                raise error
            return response
        with mock.patch("backends.webui_backend.requests.post", side_effect=post):
            return backend._task_state("task(x)")

    def test_404_means_unsupported_and_is_remembered(self):
        backend = _backend()
        self.assertEqual(self._state(backend, _Response({}, status_code=404)), TASK_UNKNOWN)
        self.assertIs(backend._task_progress_supported, False)
        self.assertEqual(self._state(backend, error=AssertionError("must not poll again")), TASK_UNKNOWN)

    def test_failures_are_transient_before_and_after_a_good_answer(self):
        backend = _backend()
        self.assertEqual(self._state(backend, error=ConnectionError("reset")), TASK_TRANSIENT)
        self.assertIsNone(backend._task_progress_supported)
        ok = _Response({"active": False, "queued": True, "completed": False})
        self.assertEqual(self._state(backend, ok), TASK_QUEUED)
        self.assertIs(backend._task_progress_supported, True)
        self.assertEqual(self._state(backend, error=TimeoutError("timeout")), TASK_TRANSIENT)
        self.assertEqual(self._state(backend, _Response({}, status_code=502)), TASK_TRANSIENT)
        # 정상 응답을 받은 뒤의 이상한 모양은 일시 이상
        self.assertEqual(self._state(backend, _Response({"detail": "busy"})), TASK_TRANSIENT)
        self.assertIs(backend._task_progress_supported, True)

    def test_unknown_shape_from_a_never_supported_server_is_unsupported(self):
        backend = _backend()
        self.assertEqual(self._state(backend, _Response({"detail": "Not Found"})), TASK_UNKNOWN)


class ForgeSavePolicyTests(unittest.TestCase):
    def _posted_json(self, requested, enabled):
        forge = _FakeForge(["active"])
        forge.released.set()
        backend = _backend()
        payload = {"prompt": "p"}
        if requested is not None:
            payload["save_images"] = requested
        with mock.patch("backends.webui_backend.requests.post", side_effect=forge.post), \
             mock.patch("core.forge_output_policy.forge_save_outputs_setting", return_value=enabled):
            result = backend.txt2img("", payload)
        self.assertTrue(result.success)
        self.assertEqual(payload.get("save_images"), requested)   # 호출부 payload 는 그대로
        return forge.generation_json

    def test_forge_does_not_duplicate_outputs_by_default(self):
        self.assertIs(self._posted_json(True, False)["save_images"], False)

    def test_user_opt_in_keeps_requested_forge_save(self):
        self.assertIs(self._posted_json(True, True)["save_images"], True)

    def test_paths_that_ask_for_no_save_stay_unsaved_even_with_opt_in(self):
        self.assertIs(self._posted_json(False, True)["save_images"], False)
        self.assertIs(self._posted_json(None, True)["save_images"], False)


if __name__ == "__main__":
    unittest.main()
