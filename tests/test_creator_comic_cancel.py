"""Creator Comic 컷 생성 — 모델 전환 중 누른 취소가 사라지지 않는다 (감사 #31 같은 결함군).

예전엔 Comic 컷 T2I 가 cancel_check 없이 backend.txt2img 를 불렀다. WebUI 가 체크포인트를
바꾸는 동안(_switch_model_if_needed) 취소하면 interrupt() 는 등록된 요청이 없어 아무것도 안 했고,
요청은 그대로 발송돼 끝까지 돈 뒤 결과만 버려졌다.
"""
from __future__ import annotations

import json
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from backends.base import GenerationResult
from backends.webui_backend import WebUIBackend
from ui.creator_actions import CreatorActionsMixin


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(json.loads(value))


class _Bridge:
    def __init__(self):
        for name in ("creatorResult", "creatorProgress", "creatorStateChanged", "comicDocumentChanged"):
            setattr(self, name, _Signal())


class _ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class _Panel:
    def __init__(self, panel_id):
        self.id = panel_id
        self.image_path = ""


class _ComicDocument:
    def __init__(self, count):
        self.panels = [_Panel(f"panel-{i + 1}") for i in range(count)]

    def to_dict(self):
        return {"panels": [{"id": p.id, "imagePath": p.image_path} for p in self.panels]}


def _actions(document):
    actions = CreatorActionsMixin()
    actions.vue_bridge = _Bridge()
    actions._creator_state_lock = threading.RLock()
    actions._creator_job_local = threading.local()
    actions._creator_running = False
    actions._creator_mode = ""
    actions._creator_active_job = None
    actions._creator_cancel_event = threading.Event()
    actions._creator_coordinator = SimpleNamespace(reserve=lambda *_a, **_k: nullcontext())
    actions._creator_should_unload_ollama = lambda: False
    actions._comic_studio = lambda: SimpleNamespace(normalize=lambda _d: document, save=lambda d: d)
    actions.written = []
    actions._creator_write_bytes = lambda data, name, feature: actions.written.append(name) or f"out/{name}"
    return actions


def _panel_payloads(document):
    return [{"prompt": f"p{i}", "negative_prompt": "", "seed": i} for i, _ in enumerate(document.panels)]


class ComicCancelTests(unittest.TestCase):
    def _run(self, actions, mode, target, payload, backend):
        with mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread), \
             mock.patch("backends.get_backend", return_value=backend), \
             mock.patch("core.comic_studio.panel_generation_payloads", side_effect=_panel_payloads):
            actions._creator_run_thread(mode, target, payload)
        return actions.vue_bridge.creatorResult.values[-1]

    def _switching_webui(self, actions):
        backend = WebUIBackend("http://127.0.0.1:7860")
        # 체크포인트 전환 중에 사용자가 Creator 취소를 누른다
        backend._switch_model_if_needed = lambda *_a, **_k: actions._creator_cancel({})
        return backend

    def test_cancel_during_model_switch_never_dispatches_single_panel(self):
        document = _ComicDocument(2)
        actions = _actions(document)
        backend = self._switching_webui(actions)
        with mock.patch("backends.webui_backend.requests.post") as post:
            result = self._run(actions, "comic_panel", actions._comic_generate_panel,
                               {"panelId": "panel-2", "model": "model.safetensors"}, backend)
        post.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertTrue(result["canceled"])
        self.assertEqual(actions.written, [])

    def test_cancel_during_model_switch_stops_the_whole_comic_batch(self):
        document = _ComicDocument(3)
        actions = _actions(document)
        backend = self._switching_webui(actions)
        with mock.patch("backends.webui_backend.requests.post") as post:
            result = self._run(actions, "comic_generate", actions._comic_generate_all,
                               {"model": "model.safetensors"}, backend)
        post.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertTrue(result["canceled"])
        self.assertEqual(actions.written, [])

    def test_interrupted_partial_panel_is_not_saved(self):
        # 발송 뒤 취소 → Forge 가 끊긴 부분 이미지를 success 로 돌려줘도 컷으로 저장하지 않는다
        document = _ComicDocument(2)
        actions = _actions(document)
        seen_checks = []

        class _Backend:
            def txt2img(self, model, payload, progress_callback=None, cancel_check=None):
                seen_checks.append(cancel_check)
                actions._creator_cancel_event.set()
                return GenerationResult(success=True, image_data=b"partial")

            def interrupt(self):
                pass

        result = self._run(actions, "comic_generate", actions._comic_generate_all,
                           {"model": "m"}, _Backend())
        self.assertFalse(result["ok"])
        self.assertTrue(result["canceled"])
        self.assertEqual(actions.written, [])
        self.assertEqual(len(seen_checks), 1)
        self.assertTrue(callable(seen_checks[0]) and seen_checks[0]())

    def test_legacy_adapter_without_cancel_check_still_generates(self):
        document = _ComicDocument(1)
        actions = _actions(document)
        calls = []

        class _LegacyBackend:
            def txt2img(self, model, payload, progress_callback=None):
                calls.append((model, payload["prompt"]))
                return GenerationResult(success=True, image_data=b"png")

            def interrupt(self):
                pass

        result = self._run(actions, "comic_panel", actions._comic_generate_panel,
                           {"panelId": "panel-1", "model": "m"}, _LegacyBackend())
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, [("m", "p0")])
        self.assertEqual(actions.written, ["comic_panel_1.png"])


if __name__ == "__main__":
    unittest.main()
