"""core.search_session — Search 런타임 스냅숏(덱·lineage·필터 base) 규칙."""
from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from core.search_session import (
    BASE_MISMATCH_MESSAGE,
    LINEAGE_MISMATCH_MESSAGE,
    load_runtime_base,
    publish_snapshot,
    resolve_prompt_deck_update,
    restore_runtime_from_disk,
    runtime_base_rows,
    runtime_lineage,
    runtime_snapshot_is_current,
    serve_runtime_base,
    share_base_rows,
)

IDENTITY = {'label': '2026_07', 'fingerprint': 'f' * 64}
SNAPSHOT = 'a' * 32
LINEAGE = {**IDENTITY, 'snapshot_id': SNAPSHOT}


class _Owner:
    """메인 윈도우의 Search 런타임 속성만 가진 가짜."""

    def __init__(self) -> None:
        self.filtered_results = []
        self.shuffled_prompt_deck = []
        self.deck_restores = 0
        self.restore_deck_result = False
        self.deck_saves = 0

    def _restore_deck_state(self) -> bool:
        self.deck_restores += 1
        return self.restore_deck_result

    def _save_deck_state(self) -> None:
        self.deck_saves += 1


class _Store:
    """SearchResultStore 의 필요한 면만 흉내 낸다(호출 기록 포함)."""

    def __init__(self, *, active=None, full=None, snapshot=SNAPSHOT, identity=IDENTITY,
                 current_identity=IDENTITY) -> None:
        self._active = active if active is not None else []
        self._full = full if full is not None else []
        self._snapshot = snapshot
        self._identity = identity
        self._current = current_identity
        self.last_error = None
        self.last_snapshot_id = None
        self.last_dataset_identity = None
        self.calls = []

    def dataset_info(self):
        self.calls.append('dataset_info')
        return dict(self._current)

    def load_active(self):
        self.calls.append('load_active')
        self.last_snapshot_id = self._snapshot
        self.last_dataset_identity = dict(self._identity) if self._snapshot else None
        return self._active

    def load_full(self, *, expected_snapshot_id=None):
        self.calls.append(('load_full', expected_snapshot_id))
        if expected_snapshot_id != self._snapshot:
            self.last_error = 'different snapshot'
            return []
        return self._full


def _published_owner(base, active=None):
    owner = _Owner()
    publish_snapshot(
        owner,
        active=active if active is not None else base,
        base=base,
        identity=dict(IDENTITY),
        snapshot_id=SNAPSHOT,
    )
    return owner


class DeckUpdateTests(unittest.TestCase):
    def test_index_payload_resolves_rows_from_the_runtime_base(self) -> None:
        base = [{'general': 'a'}, {'general': 'b'}, {'general': 'c'}]
        owner = _published_owner(base)

        update = resolve_prompt_deck_update(
            owner, {'indices': [2, 0], 'base_size': 3, 'lineage': LINEAGE}
        )

        self.assertIsNone(update.error)
        self.assertEqual(len(update.rows), 2)
        self.assertIs(update.rows[0], base[2])
        self.assertIs(update.rows[1], base[0])

    def test_empty_index_payload_is_an_empty_deck(self) -> None:
        owner = _published_owner([{'general': 'a'}])
        update = resolve_prompt_deck_update(
            owner, {'indices': [], 'base_size': 1, 'lineage': LINEAGE}
        )
        self.assertEqual(update.rows, [])
        self.assertIsNone(update.error)

    def test_index_payload_is_rejected_when_the_base_differs(self) -> None:
        owner = _published_owner([{'general': 'a'}, {'general': 'b'}])
        for payload in (
            {'indices': [0], 'base_size': 3, 'lineage': LINEAGE},
            {'indices': [2], 'base_size': 2, 'lineage': LINEAGE},
            {'indices': [-1], 'base_size': 2, 'lineage': LINEAGE},
            {'indices': [True], 'base_size': 2, 'lineage': LINEAGE},
            {'indices': ['0'], 'base_size': 2, 'lineage': LINEAGE},
            {'indices': 'nope', 'base_size': 2, 'lineage': LINEAGE},
            {'indices': [0], 'base_size': '2', 'lineage': LINEAGE},
        ):
            with self.subTest(payload=payload):
                update = resolve_prompt_deck_update(owner, payload)
                self.assertIsNone(update.rows)
                self.assertEqual(update.error, BASE_MISMATCH_MESSAGE)

    def test_index_payload_needs_a_base_of_the_current_snapshot(self) -> None:
        owner = _published_owner([{'general': 'a'}])
        owner._search_snapshot_id = 'b' * 32   # 다른 스냅숏으로 넘어감(base 는 옛 것)
        lineage = {**IDENTITY, 'snapshot_id': 'b' * 32}
        self.assertIsNone(runtime_base_rows(owner))
        update = resolve_prompt_deck_update(
            owner, {'indices': [0], 'base_size': 1, 'lineage': lineage}
        )
        self.assertEqual(update.error, BASE_MISMATCH_MESSAGE)

    def test_stale_lineage_is_rejected_before_anything_else(self) -> None:
        owner = _published_owner([{'general': 'a'}])
        stale = {**LINEAGE, 'snapshot_id': 'c' * 32}
        for payload in (
            {'indices': [0], 'base_size': 1, 'lineage': stale},
            {'results': [{'general': 'x'}], 'lineage': stale},
        ):
            update = resolve_prompt_deck_update(owner, payload)
            self.assertIsNone(update.rows)
            self.assertEqual(update.error, LINEAGE_MISMATCH_MESSAGE)

    def test_legacy_row_payload_is_still_accepted(self) -> None:
        owner = _published_owner([{'general': 'a'}])
        rows = [{'general': 'z'}]
        update = resolve_prompt_deck_update(owner, {'results': rows, 'lineage': LINEAGE})
        self.assertIs(update.rows, rows)

    def test_payload_without_rows_or_indices_is_ignored_silently(self) -> None:
        owner = _published_owner([{'general': 'a'}])
        for payload in ({}, {'lineage': LINEAGE}, {'results': 'x', 'lineage': LINEAGE}, None, []):
            update = resolve_prompt_deck_update(owner, payload)
            self.assertIsNone(update.rows)
            self.assertIsNone(update.error)

    def test_runtime_lineage_shape(self) -> None:
        owner = _Owner()
        self.assertIsNone(runtime_lineage(owner))
        owner._search_dataset_identity = dict(IDENTITY)
        self.assertEqual(runtime_lineage(owner), {**IDENTITY, 'snapshot_id': None})
        owner._search_snapshot_id = SNAPSHOT
        self.assertEqual(runtime_lineage(owner), LINEAGE)


class RuntimeRestoreTests(unittest.TestCase):
    def test_disk_restore_sets_runtime_state_without_loading_the_base(self) -> None:
        active = [{'general': 'a'}, {'general': 'b'}]
        store = _Store(active=active, full=active + [{'general': 'c'}])
        owner = _Owner()

        rows = restore_runtime_from_disk(owner, store)

        self.assertIs(rows, active)
        self.assertIs(owner.filtered_results, active)
        self.assertEqual(owner._search_snapshot_id, SNAPSHOT)
        self.assertEqual(owner._search_dataset_identity, IDENTITY)
        self.assertEqual(sorted(r['general'] for r in owner.shuffled_prompt_deck), ['a', 'b'])
        self.assertEqual(owner.deck_restores, 1)
        self.assertIsNone(runtime_base_rows(owner))
        self.assertEqual(store.calls, ['load_active'])   # full 은 시작 때 읽지 않는다

    def test_disk_restore_keeps_saved_deck_progress(self) -> None:
        store = _Store(active=[{'general': 'a'}])
        owner = _Owner()
        owner.restore_deck_result = True
        owner.shuffled_prompt_deck = ['kept']

        restore_runtime_from_disk(owner, store)

        self.assertEqual(owner.shuffled_prompt_deck, ['kept'])

    def test_disk_restore_without_snapshot_leaves_runtime_alone(self) -> None:
        store = _Store(active=[], snapshot=None)
        owner = _Owner()
        owner.filtered_results = ['existing']

        self.assertEqual(restore_runtime_from_disk(owner, store), [])
        self.assertEqual(owner.filtered_results, ['existing'])
        self.assertIsNone(runtime_lineage(owner))

    def test_runtime_snapshot_currency_checks_dataset_identity(self) -> None:
        owner = _published_owner([{'general': 'a'}])
        self.assertTrue(runtime_snapshot_is_current(owner, _Store()))
        changed = _Store(current_identity={'label': '2026_07', 'fingerprint': 'e' * 64})
        self.assertFalse(runtime_snapshot_is_current(owner, changed))
        self.assertFalse(runtime_snapshot_is_current(_Owner(), _Store()))
        self.assertFalse(runtime_snapshot_is_current(None, _Store()))

        class Broken(_Store):
            def dataset_info(self):
                raise ValueError('manifest missing')

        self.assertFalse(runtime_snapshot_is_current(owner, Broken()))

    def test_base_comes_from_memory_when_the_snapshot_owns_one(self) -> None:
        base = [{'general': 'a'}, {'general': 'b'}]
        owner = _published_owner(base, active=[base[0]])
        store = _Store()

        self.assertIs(load_runtime_base(owner, store), base)
        self.assertNotIn('load_active', store.calls)
        self.assertFalse(any(isinstance(c, tuple) for c in store.calls))

    def test_base_is_read_once_from_full_by_snapshot_id(self) -> None:
        full = [{'general': 'a'}, {'general': 'b'}, {'general': 'c'}]
        store = _Store(active=[{'general': 'a'}], full=full)
        owner = _Owner()
        restore_runtime_from_disk(owner, store)

        first = load_runtime_base(owner, store)
        second = load_runtime_base(owner, store)

        self.assertIs(first, full)
        self.assertIs(second, full)
        self.assertEqual(
            [c for c in store.calls if isinstance(c, tuple)],
            [('load_full', SNAPSHOT)],
        )
        # 이제 인덱스 필터가 이 base 로 풀린다
        update = resolve_prompt_deck_update(
            owner, {'indices': [2], 'base_size': 3, 'lineage': LINEAGE}
        )
        self.assertIs(update.rows[0], full[2])

    def test_base_falls_back_to_active_when_full_is_missing(self) -> None:
        active = [{'general': 'only-active'}]
        store = _Store(active=active, full=[])
        owner = _Owner()
        restore_runtime_from_disk(owner, store)

        base = load_runtime_base(owner, store)

        self.assertEqual(base, active)
        self.assertIsNot(base, owner.filtered_results)   # 이후 필터가 active 를 바꿔도 base 유지

    def test_unreadable_full_cache_falls_back_to_active_like_vue(self) -> None:
        active = [{'general': 'a'}]
        store = _Store(active=active)
        owner = _Owner()
        restore_runtime_from_disk(owner, store)

        def broken(**_kwargs):
            raise OSError('disk error')

        store.load_full = broken
        with redirect_stdout(StringIO()):
            base = load_runtime_base(owner, store)

        self.assertEqual(base, active)
        self.assertIs(runtime_base_rows(owner), base)

    def test_no_runtime_snapshot_means_no_runtime_base(self) -> None:
        self.assertIsNone(load_runtime_base(_Owner(), _Store()))
        self.assertIsNone(load_runtime_base(SimpleNamespace(), _Store()))


def _rows(*names):
    """디스크에서 따로 파싱한 것처럼 매번 새 dict 를 만든다(값은 같고 객체는 다름)."""
    return [{'general': name, 'rating': 'g'} for name in names]


class ShareBaseRowsTests(unittest.TestCase):
    def test_equal_rows_share_the_base_list_itself(self) -> None:
        base = _rows('a', 'b', 'c')
        self.assertIs(share_base_rows(_rows('a', 'b', 'c'), base), base)

    def test_ordered_subset_maps_to_base_row_objects(self) -> None:
        base = _rows('a', 'b', 'c', 'd')
        shared = share_base_rows(_rows('b', 'd'), base)
        self.assertEqual(len(shared), 2)
        self.assertIs(shared[0], base[1])
        self.assertIs(shared[1], base[3])

    def test_duplicate_rows_keep_their_positions(self) -> None:
        base = _rows('a', 'a', 'b')
        shared = share_base_rows(_rows('a', 'a'), base)
        self.assertIs(shared[0], base[0])
        self.assertIs(shared[1], base[1])

    def test_unprovable_rows_are_left_alone(self) -> None:
        base = _rows('a', 'b', 'c')
        for rows in (
            _rows('c', 'a'),                 # 순서가 다름
            _rows('b', 'a', 'c'),            # 같은 길이 순열
            _rows('a', 'z'),                 # base 에 없는 행
            _rows('a', 'b', 'c', 'd'),       # base 보다 김
            [{'general': 'a', 'rating': 'g', 'extra': 1}],
        ):
            with self.subTest(rows=rows):
                self.assertIsNone(share_base_rows(rows, base))
        nan_base = [{'score': float('nan')}]
        self.assertIsNone(share_base_rows([{'score': float('nan')}], nan_base))
        self.assertIsNone(share_base_rows('x', base))


class RuntimeBaseMemoryTests(unittest.TestCase):
    """재시작 복원 뒤 active 와 full 은 따로 파싱된 두 벌 — base 기록 때 한 벌로 합친다."""

    def _restored_owner(self, active, full):
        owner = _Owner()
        restore_runtime_from_disk(owner, _Store(active=active, full=full))
        return owner

    def test_unfiltered_restore_keeps_a_single_list(self) -> None:
        full = _rows('a', 'b', 'c')
        owner = self._restored_owner(_rows('a', 'b', 'c'), full)
        deck_before = [row['general'] for row in owner.shuffled_prompt_deck]

        base = load_runtime_base(owner, _Store(active=[], full=full))

        self.assertIs(base, full)
        self.assertIs(owner.filtered_results, full)   # 두 번째 복사본 없음
        self.assertTrue(all(any(row is b for b in full) for row in owner.shuffled_prompt_deck))
        self.assertEqual([row['general'] for row in owner.shuffled_prompt_deck], deck_before)
        self.assertEqual(owner.deck_saves, 1)   # 새 풀 목록 기준으로 덱 저장(id 캐시) 갱신

    def test_filtered_restore_shares_row_objects_without_moving_them(self) -> None:
        full = _rows('a', 'b', 'c', 'd')
        owner = self._restored_owner(_rows('b', 'd'), full)
        owner.shuffled_prompt_deck = [owner.filtered_results[1]]   # 'd' 하나 남음

        load_runtime_base(owner, _Store(active=[], full=full))

        self.assertEqual(owner.filtered_results, _rows('b', 'd'))   # 내용·순서 그대로
        self.assertIs(owner.filtered_results[0], full[1])
        self.assertIs(owner.filtered_results[1], full[3])
        self.assertEqual(len(owner.shuffled_prompt_deck), 1)
        self.assertIs(owner.shuffled_prompt_deck[0], full[3])
        self.assertIs(runtime_base_rows(owner), full)

    def test_active_that_is_not_a_subset_keeps_its_own_rows(self) -> None:
        full = _rows('a', 'b', 'c')
        active = _rows('c', 'a')
        owner = self._restored_owner(active, full)

        load_runtime_base(owner, _Store(active=[], full=full))

        self.assertIs(owner.filtered_results, active)
        self.assertEqual(owner.deck_saves, 0)
        self.assertIs(runtime_base_rows(owner), full)

    def test_index_filter_after_sharing_uses_the_same_objects(self) -> None:
        full = _rows('a', 'b', 'c')
        owner = self._restored_owner(_rows('a', 'b', 'c'), full)
        load_runtime_base(owner, _Store(active=[], full=full))

        update = resolve_prompt_deck_update(
            owner, {'indices': [1], 'base_size': 3, 'lineage': LINEAGE}
        )
        self.assertIs(update.rows[0], full[1])


class LazyBaseReloadTests(unittest.TestCase):
    """loadFullResults 가 base 를 기록하지 못했어도 lineage 가 맞으면 같은 규칙으로 복구."""

    def test_missing_base_is_loaded_lazily_for_a_matching_lineage(self) -> None:
        full = _rows('a', 'b', 'c')
        owner = _Owner()
        restore_runtime_from_disk(owner, _Store(active=_rows('a', 'c'), full=full))
        self.assertIsNone(runtime_base_rows(owner))
        store = _Store(active=[], full=full)

        update = resolve_prompt_deck_update(
            owner,
            {'indices': [2], 'base_size': 3, 'lineage': LINEAGE},
            store_factory=lambda: store,
        )

        self.assertIsNone(update.error)
        self.assertIs(update.rows[0], full[2])
        self.assertIs(runtime_base_rows(owner), full)
        self.assertEqual([c for c in store.calls if isinstance(c, tuple)], [('load_full', SNAPSHOT)])

    def test_lazy_base_still_guards_the_base_size(self) -> None:
        owner = _Owner()
        restore_runtime_from_disk(owner, _Store(active=_rows('a'), full=_rows('a', 'b')))
        store = _Store(active=[], full=_rows('a', 'b'))

        update = resolve_prompt_deck_update(
            owner,
            {'indices': [0], 'base_size': 1, 'lineage': LINEAGE},   # Vue base 는 active(1행)
            store_factory=lambda: store,
        )

        self.assertIsNone(update.rows)
        self.assertEqual(update.error, BASE_MISMATCH_MESSAGE)

    def test_stale_lineage_never_triggers_a_disk_read(self) -> None:
        owner = _Owner()
        restore_runtime_from_disk(owner, _Store(active=_rows('a')))

        def factory():
            raise AssertionError('stale filter must not read the cache')

        update = resolve_prompt_deck_update(
            owner,
            {'indices': [0], 'base_size': 1, 'lineage': {**LINEAGE, 'snapshot_id': 'c' * 32}},
            store_factory=factory,
        )
        self.assertEqual(update.error, LINEAGE_MISMATCH_MESSAGE)

    def test_store_failure_is_a_base_mismatch_not_an_exception(self) -> None:
        owner = _Owner()
        restore_runtime_from_disk(owner, _Store(active=_rows('a')))

        def factory():
            raise OSError('cache dir unavailable')

        with redirect_stdout(StringIO()):
            update = resolve_prompt_deck_update(
                owner,
                {'indices': [0], 'base_size': 1, 'lineage': LINEAGE},
                store_factory=factory,
            )
        self.assertEqual(update.error, BASE_MISMATCH_MESSAGE)


class _FallbackStore:
    """owner 스냅숏이 '현재'가 아닐 때(loadFullResults 폴백) 디스크 응답을 흉내 낸다."""

    def __init__(self, *, full=None, active=None, snapshot=SNAPSHOT, current=None) -> None:
        self._full = full if full is not None else []
        self._active = active if active is not None else []
        self._snapshot = snapshot
        self._current = current
        self.last_error = None
        self.last_snapshot_id = None

    def dataset_info(self):
        if self._current is None:
            raise ValueError('manifest unreadable')
        return dict(self._current)

    def load_full(self, *, expected_snapshot_id=None):
        self.last_snapshot_id = self._snapshot if self._full else None
        return self._full

    def load_active(self):
        self.last_snapshot_id = self._snapshot if self._active else None
        return self._active


class ServeRuntimeBaseTests(unittest.TestCase):
    """loadFullResults 가 돌려준 배열 = Python 이 기록하는 base (폴백 경로 포함)."""

    def test_current_snapshot_serves_and_records_the_runtime_base(self) -> None:
        full = _rows('a', 'b')
        owner = _Owner()
        restore_runtime_from_disk(owner, _Store(active=_rows('a'), full=full))

        served = serve_runtime_base(owner, _Store(active=[], full=full))

        self.assertIs(served, full)
        self.assertIs(runtime_base_rows(owner), full)

    def test_empty_fallback_records_the_active_rows_vue_uses_as_base(self) -> None:
        owner = _published_owner(None, active=_rows('a', 'b'))
        self.assertIsNone(runtime_base_rows(owner))
        store = _FallbackStore(current={'label': '2026_07', 'fingerprint': 'e' * 64})

        with redirect_stdout(StringIO()):
            served = serve_runtime_base(owner, store)

        self.assertEqual(served, [])
        # Vue: full=[] → base = loadLast 응답(= filtered_results)
        self.assertEqual(runtime_base_rows(owner), _rows('a', 'b'))
        update = resolve_prompt_deck_update(
            owner, {'indices': [1], 'base_size': 2, 'lineage': LINEAGE}
        )
        self.assertIs(update.rows[0], owner.filtered_results[1])

    def test_fallback_rows_of_the_owner_snapshot_become_the_base(self) -> None:
        owner = _published_owner(None, active=_rows('b'))
        full = _rows('a', 'b')
        store = _FallbackStore(full=full)   # dataset_info 가 일시적으로 실패

        served = serve_runtime_base(owner, store)

        self.assertIs(served, full)
        self.assertIs(runtime_base_rows(owner), full)
        self.assertIs(owner.filtered_results[0], full[1])   # 한 벌로 합쳐짐

    def test_fallback_rows_of_another_snapshot_are_not_recorded(self) -> None:
        owner = _published_owner(None, active=_rows('b'))
        store = _FallbackStore(full=_rows('x', 'y'), snapshot='d' * 32)

        served = serve_runtime_base(owner, store)

        self.assertEqual(served, _rows('x', 'y'))
        self.assertIsNone(runtime_base_rows(owner))

    def test_fallback_without_owner_snapshot_just_serves_disk(self) -> None:
        store = _FallbackStore(active=_rows('a'))
        self.assertEqual(serve_runtime_base(None, store), _rows('a'))
        owner = _Owner()
        self.assertEqual(serve_runtime_base(owner, store), _rows('a'))
        self.assertIsNone(runtime_base_rows(owner))


if __name__ == '__main__':
    unittest.main()
