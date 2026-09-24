import threading
import unittest
from unittest import mock

from core.resource_coordinator import (
    GenerationResourceCoordinator,
    ResourceBusyError,
    ResourceTransitionError,
    get_generation_coordinator,
    release_in_process_vision_models,
)


class ResourceCoordinatorTests(unittest.TestCase):
    def test_unloads_before_running_and_returns_to_idle(self):
        calls = []
        coordinator = GenerationResourceCoordinator(
            unload_llm=lambda: calls.append("unload") or True,
            on_state=lambda state: calls.append(state.phase),
        )
        with coordinator.reserve("h3") as state:
            self.assertEqual(state.phase, "running")
            self.assertTrue(state.llm_unloaded)
        self.assertEqual(coordinator.state.phase, "idle")
        self.assertEqual(calls[:3], ["preparing", "unload", "running"])

    def test_does_not_invoke_unsupplied_external_process_hooks(self):
        coordinator = GenerationResourceCoordinator()
        with coordinator.reserve("krea", unload_llm=False):
            self.assertEqual(coordinator.state.owner, "krea")

    def test_rejects_concurrent_generation(self):
        coordinator = GenerationResourceCoordinator()
        entered = threading.Event()
        release = threading.Event()

        def hold():
            with coordinator.reserve("first", unload_llm=False):
                entered.set()
                release.wait(2)

        thread = threading.Thread(target=hold)
        thread.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaises(ResourceBusyError):
            with coordinator.reserve("second", unload_llm=False):
                pass
        release.set()
        thread.join(2)

    def test_failed_unload_releases_lease(self):
        coordinator = GenerationResourceCoordinator(unload_llm=lambda: False)
        with self.assertRaises(ResourceTransitionError):
            with coordinator.reserve("first"):
                pass
        with coordinator.reserve("second", unload_llm=False):
            self.assertEqual(coordinator.state.owner, "second")

    def test_process_wide_accessor_keeps_one_shared_lease(self):
        first = get_generation_coordinator()
        second = get_generation_coordinator()
        self.assertIs(first, second)

    def test_hooks_can_be_configured_after_coordinator_creation(self):
        calls = []
        coordinator = GenerationResourceCoordinator()
        coordinator.configure(
            unload_llm=lambda: calls.append("unload") or True,
            on_state=lambda state: calls.append(state.phase),
        )
        with coordinator.reserve("creator"):
            pass
        self.assertIn("unload", calls)
        self.assertEqual(calls[-1], "idle")


class BeforeGenerationHookTests(unittest.TestCase):
    """생성 직전 앱이 쥔 비전 모델(SAM3 번들) 반납 — 16GB OOM 회귀 방지."""

    def test_hook_runs_after_lease_before_llm_unload_and_running(self):
        calls = []
        coordinator = GenerationResourceCoordinator(
            unload_llm=lambda: calls.append("unload") or True,
            on_state=lambda state: calls.append(state.phase),
            before_generation=lambda: calls.append("release"),
        )
        with coordinator.reserve("txt2img"):
            calls.append("body")
        self.assertEqual(calls[:5], ["preparing", "release", "unload", "running", "body"])

    def test_hook_failure_never_blocks_generation_or_strands_lease(self):
        def boom():
            raise RuntimeError("release failed")

        coordinator = GenerationResourceCoordinator(before_generation=boom)
        with coordinator.reserve("first", unload_llm=False) as state:
            self.assertEqual(state.phase, "running")
        with coordinator.reserve("second", unload_llm=False):
            self.assertEqual(coordinator.state.owner, "second")

    def test_hook_is_not_called_when_lease_is_busy(self):
        calls = []
        coordinator = GenerationResourceCoordinator(before_generation=lambda: calls.append("release"))
        with coordinator.reserve("first", unload_llm=False):
            calls.clear()
            with self.assertRaises(ResourceBusyError):
                with coordinator.reserve("second", unload_llm=False):
                    pass
            self.assertEqual(calls, [], "리스를 못 잡은 요청이 다른 작업의 모델을 내리면 안 된다")

    def test_hook_can_be_configured_later(self):
        calls = []
        coordinator = GenerationResourceCoordinator()
        coordinator.configure(before_generation=lambda: calls.append("release"))
        with coordinator.reserve("late", unload_llm=False):
            pass
        self.assertEqual(calls, ["release"])

    def test_shared_coordinator_releases_editor_sam3_cache(self):
        with mock.patch("core.model_cache.release_for_generation", return_value=1) as release:
            with get_generation_coordinator().reserve("shared-hook-test", unload_llm=False, timeout=1):
                pass
        release.assert_called_once_with()

    def test_shared_hook_frees_a_cached_bundle_through_the_real_cache(self):
        from core import model_cache

        cache = model_cache.IdleModelCache("sam3-test", idle_seconds=90.0, max_items=1)
        cache.get("sam3.pt|cuda", lambda _key: object())
        with mock.patch.object(model_cache, "SAM3_CACHE", cache):
            release_in_process_vision_models()
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
