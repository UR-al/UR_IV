"""'시드 탐색' — 완성 payload 동결 + subseed 보존 + 대기열 소비 경로 회귀."""
import random
import unittest
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core.seed_explore import (
    MARKER, SeedExploreError, build_seed_explore_jobs, parse_base_seed,
    prepare_seed_explore_generation, variation_strengths,
)

_BASE = {
    "prompt": "1girl, smile, <lora:style:0.70>",
    "negative_prompt": "lowres",
    "seed": -1, "steps": 28, "cfg_scale": 5.0, "width": 832, "height": 1216,
    "enable_hr": True, "hr_scale": 1.5,
    "forge_additional_modules": ["vae.safetensors"],
    "alwayson_scripts": {"ADetailer": {"args": [True, {"ad_model": "face.pt"}]}},
}


class PureRuleTests(unittest.TestCase):
    def test_parse_base_seed(self):
        rng = random.Random(1)
        self.assertEqual(parse_base_seed("123"), 123)
        self.assertEqual(parse_base_seed(42), 42)
        self.assertEqual(parse_base_seed(2 ** 32 - 1), 2 ** 32 - 1)
        for bad in ("", "abc", None, -1, "-1"):
            seed = parse_base_seed(bad, rng)
            self.assertTrue(0 <= seed <= 2 ** 32 - 1, bad)

    def test_seed_outside_the_generation_range_is_refused_not_replaced(self):
        # 예전엔 조용히 새 랜덤 시드로 바꿔 이미지와 무관한 시드를 탐색했다
        from core.payload_validator import SEED_MAX as VALIDATOR_MAX
        for big in (2 ** 32, 2 ** 33, str(2 ** 64 - 1)):
            with self.assertRaisesRegex(SeedExploreError, str(VALIDATOR_MAX)):
                parse_base_seed(big, random.Random(1))

    def test_strengths_are_distinct_and_start_at_the_original(self):
        s = variation_strengths(9)
        self.assertEqual(s[0], 0.0)
        self.assertEqual(len(set(s)), 9)
        self.assertEqual(max(s), 0.4)

    def test_webui_jobs_keep_the_full_payload_and_vary_subseed(self):
        jobs = build_seed_explore_jobs(_BASE, base_seed=100, model="m.safetensors",
                                       backend_id="b1", supports_subseed=True)
        self.assertEqual(len(jobs), 9)
        self.assertEqual({j["seed"] for j in jobs}, {100})
        self.assertEqual(len({j["subseed"] for j in jobs}), 9)
        self.assertEqual([j["subseed_strength"] for j in jobs], variation_strengths(9))
        for job in jobs:
            self.assertEqual(job["alwayson_scripts"], _BASE["alwayson_scripts"])
            self.assertTrue(job["enable_hr"])
            self.assertIn("<lora:style:0.70>", job["prompt"])
        jobs[0]["alwayson_scripts"]["ADetailer"]["args"][1]["ad_model"] = "changed"
        self.assertEqual(_BASE["alwayson_scripts"]["ADetailer"]["args"][1]["ad_model"], "face.pt")
        self.assertEqual(jobs[1]["alwayson_scripts"]["ADetailer"]["args"][1]["ad_model"], "face.pt")

    def test_comfy_jobs_use_neighbouring_seeds(self):
        jobs = build_seed_explore_jobs({**_BASE, "subseed": 5}, base_seed=2 ** 32 - 2, model="m",
                                       backend_id="c1", supports_subseed=False)
        self.assertEqual([j["seed"] for j in jobs[:3]], [2 ** 32 - 2, 2 ** 32 - 1, 0])
        self.assertTrue(all("subseed" not in j and "subseed_strength" not in j for j in jobs))

    def test_prepare_strips_queue_keys_and_checks_the_backend(self):
        [job] = build_seed_explore_jobs(_BASE, base_seed=7, model="m.safetensors",
                                        backend_id="b1", supports_subseed=True, count=1)
        item = {**job, "id": 3, "group_id": "g"}
        payload, model = prepare_seed_explore_generation(item, "b1")
        self.assertEqual(model, "m.safetensors")
        self.assertNotIn(MARKER, payload)
        self.assertNotIn("id", payload)
        self.assertEqual(payload["subseed_strength"], 0.0)
        with self.assertRaises(SeedExploreError):
            prepare_seed_explore_generation(item, "other")

    def test_prepare_prefers_the_comfy_model_snapshot(self):
        [job] = build_seed_explore_jobs({**_BASE, "_comfy_model_snapshot": "comfy.safetensors",
                                         "_comfy_workflow_snapshot": {"k": 1}},
                                        base_seed=7, model="ui.safetensors", backend_id="c1",
                                        supports_subseed=False, count=1)
        payload, model = prepare_seed_explore_generation(job, "c1")
        self.assertEqual(model, "comfy.safetensors")
        self.assertNotIn("_comfy_model_snapshot", payload)
        self.assertEqual(payload["_comfy_workflow_snapshot"], {"k": 1})


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _Owner:
    def __init__(self, *, running=False, automating=False, busy=False):
        self.vue_bridge = SimpleNamespace(showNotification=_Signal())
        self.queue_panel = SimpleNamespace(items=[])
        self.queue_panel.add_single_item = self.queue_panel.items.append
        self.queue_manager = SimpleNamespace(is_running=running, total_count=4, started=0)
        self.queue_manager.start = lambda: setattr(self.queue_manager, "started",
                                                   self.queue_manager.started + 1)
        self.is_automating = automating
        self.gen_worker = SimpleNamespace(isRunning=lambda: busy, _result_emitted=False)
        self.model_combo = SimpleNamespace(currentText=lambda: "anima.safetensors")
        self.statuses = []
        self.seed_input_writes = 0

    def show_status(self, message, *_a):
        self.statuses.append(message)

    def _is_krea2_generation(self):
        return False

    def _build_generation_payload(self, *, snapshot=False, **_k):
        assert snapshot, "UI 위젯을 건드리지 않는 스냅숏이어야 한다"
        return {**_BASE, "_chat_deferred_prompt": {"wildcards": False}}, None


class StartSeedExploreTests(unittest.TestCase):
    def start(self, owner, seed="123", kind=BackendType.WEBUI):
        from ui.seed_explore_actions import start_seed_explore
        backend = SimpleNamespace(api_url="http://127.0.0.1:7860")
        with mock.patch("backends.get_backend", return_value=backend), \
             mock.patch("backends.get_backend_type", return_value=kind):
            return start_seed_explore(owner, {"seed": seed})

    def test_queues_nine_frozen_full_payloads_with_subseed(self):
        owner = _Owner()
        self.assertTrue(self.start(owner))
        items = owner.queue_panel.items
        self.assertEqual(len(items), 9)
        self.assertEqual({i["seed"] for i in items}, {123})
        self.assertEqual(len({i["subseed_strength"] for i in items}), 9)
        self.assertTrue(all(i["alwayson_scripts"]["ADetailer"] for i in items))
        self.assertTrue(all("_chat_deferred_prompt" not in i for i in items))   # 프롬프트는 한 번만 해석
        self.assertEqual(owner.queue_manager.started, 1)

    def test_joins_a_running_queue_without_restarting_it(self):
        owner = _Owner(running=True)
        self.assertTrue(self.start(owner))
        self.assertEqual(owner.queue_manager.started, 0)
        self.assertEqual(owner.queue_manager.total_count, 13)

    def test_refuses_during_a_manual_generation(self):
        owner = _Owner(busy=True)
        self.assertFalse(self.start(owner))
        self.assertEqual(owner.queue_panel.items, [])
        self.assertEqual(owner.vue_bridge.showNotification.calls[-1][0], "warning")

    def test_during_automation_queues_without_starting_the_queue_manager(self):
        # 자동화 '큐 우선' 경로가 다음 장 전에 먼저 낸다(동결 payload 그대로 — ui.queue_item_dispatch).
        # 대기열 매니저를 시작하면 자동화의 생성과 워커를 다툰다.
        owner = _Owner(automating=True, busy=True)
        self.assertTrue(self.start(owner))
        self.assertEqual(len(owner.queue_panel.items), 9)
        self.assertEqual(owner.queue_manager.started, 0)
        self.assertEqual(owner.queue_manager.total_count, 4)
        self.assertIn("자동화", owner.vue_bridge.showNotification.calls[-1][1])

    def test_out_of_range_seed_is_reported_and_nothing_is_queued(self):
        owner = _Owner()
        self.assertFalse(self.start(owner, seed=str(2 ** 40)))
        self.assertEqual(owner.queue_panel.items, [])
        self.assertEqual(owner.queue_manager.started, 0)
        kind, message = owner.vue_bridge.showNotification.calls[-1]
        self.assertEqual(kind, "warning")
        self.assertIn(str(2 ** 40), message)

    def test_comfy_uses_neighbouring_seeds(self):
        owner = _Owner()
        self.assertTrue(self.start(owner, seed="10", kind=BackendType.COMFYUI))
        self.assertEqual([i["seed"] for i in owner.queue_panel.items], list(range(10, 19)))
        self.assertTrue(all("subseed" not in i for i in owner.queue_panel.items))


class QueueDispatchTests(unittest.TestCase):
    def test_queued_item_is_sent_as_frozen_payload(self):
        from ui.generator_main import GeneratorMainUI
        backend = SimpleNamespace(api_url="http://127.0.0.1:7860")
        from core.xyz_capabilities import backend_identity
        [job] = build_seed_explore_jobs(_BASE, base_seed=9, model="anima.safetensors",
                                        backend_id=backend_identity("webui", backend),
                                        supports_subseed=True, count=2)[1:]
        host = SimpleNamespace(start_generation=mock.Mock(return_value=True),
                               queue_manager=mock.Mock(), _abort_generation=mock.Mock(),
                               _apply_payload_to_ui=mock.Mock())
        with mock.patch("backends.get_backend", return_value=backend), \
             mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            GeneratorMainUI._on_generation_requested(host, {**job, "id": 1})
        host._apply_payload_to_ui.assert_not_called()          # seed 칸을 건드리지 않는다
        kwargs = host.start_generation.call_args.kwargs
        self.assertEqual(kwargs["payload_override"]["subseed_strength"], 0.05)
        self.assertEqual(kwargs["model_override"], "anima.safetensors")
        self.assertIs(kwargs["backend_override"], backend)

    def test_backend_change_pauses_the_queue(self):
        from ui.generator_main import GeneratorMainUI
        [job] = build_seed_explore_jobs(_BASE, base_seed=9, model="m", backend_id="elsewhere",
                                        supports_subseed=True, count=1)
        host = SimpleNamespace(start_generation=mock.Mock(), queue_manager=mock.Mock(),
                               _abort_generation=mock.Mock())
        with mock.patch("backends.get_backend", return_value=SimpleNamespace(api_url="http://x")), \
             mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            GeneratorMainUI._on_generation_requested(host, job)
        host.start_generation.assert_not_called()
        host.queue_manager.pause.assert_called_once()
        host._abort_generation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
