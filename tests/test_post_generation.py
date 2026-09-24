"""core.post_generation — 생성 후 언로드는 설정이 켜져 있고 연속 작업이 끝났을 때만."""
import unittest

from core.post_generation import (
    PREF_KEY,
    should_unload_after_generation,
    unload_after_generation_enabled,
    unload_backend_models,
)


class ShouldUnloadTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(unload_after_generation_enabled({}))
        self.assertFalse(unload_after_generation_enabled(None))
        self.assertFalse(should_unload_after_generation({}, automating=False, queue_running=False))

    def test_on_only_after_the_last_job(self):
        prefs = {PREF_KEY: True}
        self.assertTrue(should_unload_after_generation(prefs, automating=False, queue_running=False))
        self.assertFalse(should_unload_after_generation(prefs, automating=True, queue_running=False), "자동화가 이어서 만든다")
        self.assertFalse(should_unload_after_generation(prefs, automating=False, queue_running=True), "대기열이 이어서 만든다")
        self.assertFalse(should_unload_after_generation(prefs, automating=False, queue_running=False, worker_running=True), "샘플링 중엔 안 내린다")
        self.assertFalse(should_unload_after_generation(prefs, automating=False, queue_running=False,
                                                        generation_active=True),
                         "gen_worker 밖의 생성(인페인트·I2I·채팅 등)이 GPU 리스를 쥐고 있으면 안 내린다")

    def test_truthy_strings_from_json_are_respected(self):
        self.assertTrue(unload_after_generation_enabled({PREF_KEY: True}))
        self.assertFalse(unload_after_generation_enabled({PREF_KEY: False}))


class UnloadBackendTests(unittest.TestCase):
    def test_prefers_light_unload_then_full_then_plain_and_never_raises(self):
        calls = []

        class Full:
            def unload_checkpoint(self): calls.append("checkpoint"); return True
            def unload_models(self): calls.append("models"); return True

        class Legacy:
            def unload_models(self): calls.append("models"); return True
            def unload(self): calls.append("plain"); return True

        class Plain:
            def unload(self): calls.append("plain"); return True

        class Broken:
            def unload_checkpoint(self): raise RuntimeError("down")

        self.assertTrue(unload_backend_models(Full()))
        self.assertTrue(unload_backend_models(Legacy()))
        self.assertTrue(unload_backend_models(Plain()))
        self.assertEqual(calls, ["checkpoint", "models", "plain"], "가벼운 unload_checkpoint 를 먼저 쓴다")
        self.assertFalse(unload_backend_models(Broken()))
        self.assertFalse(unload_backend_models(None))
        self.assertFalse(unload_backend_models(object()))


if __name__ == "__main__":
    unittest.main()
