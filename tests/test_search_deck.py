"""자동화 덱 재구성 단일 규칙(core.search_deck) + 뽑기·저장 경로 회귀."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import search_deck as sd
from ui.generator_actions import ActionsMixin
from ui.generator_main import GeneratorMainUI
from ui.generator_prompts import PromptHandlingMixin


def _rows(*ratings):
    return [{"general": f"t{i}", "rating": r} for i, r in enumerate(ratings)]


class RatingRuleTests(unittest.TestCase):
    def test_unknown_ratings_pass_and_full_names_are_understood(self):
        gs = {"g", "s"}
        self.assertTrue(sd.rating_ok({"rating": ""}, gs))
        self.assertTrue(sd.rating_ok({"rating": "nan"}, gs))
        self.assertTrue(sd.rating_ok({}, gs))
        self.assertTrue(sd.rating_ok({"rating": "General"}, gs))
        self.assertFalse(sd.rating_ok({"rating": "explicit"}, gs))
        self.assertFalse(sd.rating_ok({"rating": "q"}, gs))

    def test_normalize_ratings(self):
        self.assertEqual(sd.normalize_ratings(None), sd.ALL_RATINGS)
        self.assertEqual(sd.normalize_ratings(["G", "sensitive", "x"]), frozenset({"g", "s"}))
        self.assertEqual(sd.normalize_ratings([]), frozenset())

    def test_build_deck_filters_then_shuffles_without_touching_the_pool(self):
        pool = _rows("g", "e", "s", "", "q")
        deck = sd.build_deck(pool, {"g", "s"}, random.Random(1))
        self.assertEqual(sorted(r["general"] for r in deck), ["t0", "t2", "t3"])
        self.assertEqual(len(pool), 5)

    def test_refilter_keeps_progress(self):
        pool = _rows("g", "g", "q", "q", "e")
        deck = [pool[1], pool[3]]           # t0·t2 는 이미 뽑았다
        self.assertIs(sd.refilter_deck(deck, pool, {"g", "q"}, {"q", "g"}), deck)
        self.assertEqual(sd.refilter_deck(deck, pool, {"g", "q"}, {"g"}), [pool[1]])
        widened = sd.refilter_deck(deck, pool, {"g", "q"}, {"g", "q", "e"}, random.Random(2))
        self.assertEqual(sorted(r["general"] for r in widened), ["t1", "t3", "t4"])   # t0·t2 안 돌아온다


class _Owner:
    def __init__(self, pool, ratings=None):
        self.filtered_results = pool
        self.shuffled_prompt_deck = []
        if ratings is not None:
            self._rating_filter = set(ratings)
        self.saves = 0
        self.emits = 0

    def _save_deck_state(self):
        self.saves += 1

    def _emit_auto_status(self):
        self.emits += 1


class OwnerHelperTests(unittest.TestCase):
    def test_refill_applies_the_rating_filter_saves_and_emits_once(self):
        owner = _Owner(_rows("g", "e", "s"), {"g", "s"})
        deck = sd.refill_owner_deck(owner, rng=random.Random(3))
        self.assertIs(owner.shuffled_prompt_deck, deck)
        self.assertEqual(sorted(r["rating"] for r in deck), ["g", "s"])
        self.assertEqual((owner.saves, owner.emits), (1, 1))
        self.assertEqual(owner._deck_built_ratings, frozenset({"g", "s"}))

    def test_refill_tolerates_owners_without_hooks(self):
        class Bare:
            filtered_results = _rows("g")
        bare = Bare()
        self.assertEqual(len(sd.refill_owner_deck(bare)), 1)

    def test_same_filter_again_does_not_reshuffle(self):
        owner = _Owner(_rows("g", "s", "q", "e"), {"g", "s"})
        sd.refill_owner_deck(owner, rng=random.Random(4))
        owner.shuffled_prompt_deck.pop()    # 한 장 뽑음
        deck_before = owner.shuffled_prompt_deck
        saves = owner.saves
        self.assertFalse(sd.set_owner_rating_filter(owner, ["s", "g"]))
        self.assertIs(owner.shuffled_prompt_deck, deck_before)
        self.assertEqual(owner.saves, saves)

    def test_filter_change_narrows_or_widens_and_persists(self):
        pool = _rows("g", "s", "q")
        owner = _Owner(pool, {"g", "s"})
        sd.refill_owner_deck(owner, rng=random.Random(5))
        owner.shuffled_prompt_deck = [pool[1]]           # g 는 이미 뽑았다
        self.assertTrue(sd.set_owner_rating_filter(owner, ["g", "s", "q"], rng=random.Random(6)))
        self.assertEqual(sorted(r["rating"] for r in owner.shuffled_prompt_deck), ["q", "s"])
        self.assertEqual(owner._rating_filter, {"g", "s", "q"})
        self.assertTrue(sd.set_owner_rating_filter(owner, ["q"]))
        self.assertEqual(owner.shuffled_prompt_deck, [pool[2]])
        self.assertEqual(owner.saves, 3)

    def test_unknown_build_filter_only_narrows(self):
        pool = _rows("g", "e")
        owner = _Owner(pool)
        owner.shuffled_prompt_deck = list(pool)     # 옛 덱 파일 — 어떤 필터였는지 모름
        self.assertTrue(sd.set_owner_rating_filter(owner, ["g"]))
        self.assertEqual(owner.shuffled_prompt_deck, [pool[0]])

    def test_pool_size_counts_only_eligible_rows_and_caches(self):
        owner = _Owner(_rows("g", "e", "s", ""), {"g"})
        self.assertEqual(sd.deck_pool_size(owner), 2)
        with mock.patch.object(sd, "eligible_rows", side_effect=AssertionError("cached")):
            self.assertEqual(sd.deck_pool_size(owner), 2)
        owner._rating_filter = {"g", "s", "e"}
        self.assertEqual(sd.deck_pool_size(owner), 4)

    def test_prompt_bundle_keeps_resolution_for_auto_res(self):
        bundle = sd.prompt_bundle_from_row({
            "general": "NaN", "character": "miku", "copyright": None, "artist": "x",
            "rating": "g", "image_width": 832.0, "image_height": "1216",
            "cond_positive": [{"condition": "a"}],
        })
        self.assertEqual(bundle, {
            "copyright": "", "character": "miku", "artist": "x", "general": "",
            "rating": "g", "image_width": 832, "image_height": 1216,
        })


class ThrottleTests(unittest.TestCase):
    def test_saves_every_n_draws_after_interval_or_when_empty(self):
        now = [0.0]
        t = sd.DeckSaveThrottle(every=3, interval=60, clock=lambda: now[0])
        self.assertFalse(t.note_draw(10))
        self.assertFalse(t.note_draw(9))
        self.assertTrue(t.note_draw(8))
        t.mark_saved()
        self.assertFalse(t.dirty)
        self.assertTrue(t.note_draw(0))          # 덱이 비면 바로
        t.mark_saved()
        now[0] = 61.0
        self.assertTrue(t.note_draw(5))          # 시간이 지나면
        self.assertTrue(t.dirty)


class _DeckHost(PromptHandlingMixin, ActionsMixin):
    """apply_random_prompt · 덱 저장/복원만 도는 메인 윈도우 대역."""

    def __init__(self, path, pool, ratings=("g", "s")):
        self._path = path
        self.filtered_results = pool
        self.shuffled_prompt_deck = []
        self._rating_filter = set(ratings)
        self._search_snapshot_id = "a" * 32
        self.is_automating = False
        self.applied = []
        self.statuses = []
        self.stops = []

        class _Btn:
            def setText(self, _t):
                pass
        self.btn_random_prompt = _Btn()

    def _deck_state_path(self):
        return str(self._path)

    def apply_prompt_from_data(self, bundle, *a, **k):
        self.applied.append(bundle)

    def show_status(self, message, *_a):
        self.statuses.append(message)

    def _stop_automation(self, message=None):
        self.is_automating = False
        self.stops.append(message)


class DrawPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "last_deck.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_eligible_rows_stops_automation_instead_of_index_error(self):
        host = _DeckHost(self.path, _rows("e", "q"))
        host.is_automating = True
        self.assertFalse(host.apply_random_prompt())
        self.assertEqual(host.stops, [sd.NO_ELIGIBLE_MESSAGE])
        self.assertEqual(host.applied, [])

    def test_manual_draw_with_no_eligible_rows_warns(self):
        host = _DeckHost(self.path, _rows("e"))
        with mock.patch("ui.generator_prompts.QMessageBox") as box:
            self.assertFalse(host.apply_random_prompt())
        box.warning.assert_called_once()

    def test_automation_refill_does_not_open_a_modal(self):
        host = _DeckHost(self.path, _rows("g", "s"))
        host.is_automating = True
        with mock.patch("ui.generator_prompts.QMessageBox") as box:
            self.assertTrue(host.apply_random_prompt())
        box.information.assert_not_called()
        self.assertEqual(len(host.applied), 1)

    def test_draws_are_saved_in_batches_and_flushed(self):
        host = _DeckHost(self.path, _rows(*("g" * 30)))
        host.is_automating = True
        with mock.patch.object(ActionsMixin, "_save_deck_state", autospec=True,
                               side_effect=ActionsMixin._save_deck_state) as save:
            for _ in range(5):
                host.apply_random_prompt()
            self.assertEqual(save.call_count, 1)          # 첫 리필 저장만 — 뽑기 5번은 모아 둔다
            host._flush_deck_state()
            self.assertEqual(save.call_count, 2)
            host._flush_deck_state()
            self.assertEqual(save.call_count, 2)          # 더 쓸 게 없다
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(stored["remaining"]), 25)
        self.assertEqual(stored["ratings"], ["g", "s"])

    def test_saved_filter_round_trips_and_old_files_still_restore(self):
        pool = _rows("g", "s", "q")
        host = _DeckHost(self.path, pool)
        sd.refill_owner_deck(host, emit=False)
        restored = _DeckHost(self.path, pool)
        self.assertTrue(restored._restore_deck_state())
        self.assertEqual(restored._deck_built_ratings, frozenset({"g", "s"}))

        self.path.write_text(json.dumps({"schema_version": 1, "snapshot_id": "a" * 32,
                                         "pool_size": 3, "remaining": [2]}), encoding="utf-8")
        legacy = _DeckHost(self.path, pool)
        self.assertTrue(legacy._restore_deck_state())
        self.assertIsNone(legacy._deck_built_ratings)
        self.assertEqual(legacy.shuffled_prompt_deck, [pool[2]])


class RatingFilterActionTests(unittest.TestCase):
    def test_repeated_set_rating_filter_keeps_deck_progress(self):
        from ui.model_download_actions import ModelDownloadActionsMixin

        class Subject(ModelDownloadActionsMixin):
            _handle_creator_action = staticmethod(lambda *_: False)
            _handle_chat_action = staticmethod(lambda *_: False)

            def __init__(self):
                self.filtered_results = _rows("g", "s", "q")
                self._rating_filter = {"g", "s"}
                self.saves = 0

            def _save_deck_state(self):
                self.saves += 1

        subject = Subject()
        sd.refill_owner_deck(subject, emit=False)
        subject.shuffled_prompt_deck.pop()
        progress = list(subject.shuffled_prompt_deck)
        for _ in range(3):   # Vue 마운트·웹 탭마다 같은 필터가 다시 온다
            GeneratorMainUI._handle_vue_action(subject, "set_rating_filter", {"ratings": ["g", "s"]})
        self.assertEqual(subject.shuffled_prompt_deck, progress)
        self.assertEqual(subject.saves, 1)


if __name__ == "__main__":
    unittest.main()
