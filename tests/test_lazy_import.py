"""utils.lazy_import — 첫 속성 접근 때만 실제 모듈을 불러오는 대리 모듈."""
from __future__ import annotations

import sys
import threading
import types
import unittest
from unittest import mock

from utils.lazy_import import LazyModule, lazy_module


class _FakeModuleFactory:
    """import_module 을 대신해 호출 횟수를 센다."""

    def __init__(self):
        self.calls = 0
        self.module = types.ModuleType("fake_heavy")
        self.module.CONST = 7
        self.module.double = lambda x: x * 2
        self.module.error = type("error", (Exception,), {})

    def __call__(self, name):
        self.calls += 1
        return self.module


class LazyModuleTests(unittest.TestCase):
    def test_nothing_is_imported_until_first_attribute_access(self):
        factory = _FakeModuleFactory()
        with mock.patch("utils.lazy_import.importlib.import_module", factory):
            proxy = lazy_module("fake_heavy")
            self.assertEqual(factory.calls, 0)
            self.assertFalse(proxy.is_loaded)
            self.assertIn("not loaded", repr(proxy))
            self.assertEqual(proxy.CONST, 7)
            self.assertEqual(proxy.double(4), 8)
            self.assertTrue(proxy.is_loaded)
            self.assertEqual(factory.calls, 1, "로드한 모듈은 캐시한다")

    def test_exception_classes_work_in_except_clauses(self):
        factory = _FakeModuleFactory()
        with mock.patch("utils.lazy_import.importlib.import_module", factory):
            proxy = lazy_module("fake_heavy")
            try:
                raise factory.module.error("boom")
            except proxy.error as exc:   # except 절에서 평가될 때 로드
                self.assertEqual(str(exc), "boom")

    def test_attribute_writes_and_patches_go_to_the_real_module(self):
        factory = _FakeModuleFactory()
        with mock.patch("utils.lazy_import.importlib.import_module", factory):
            proxy = lazy_module("fake_heavy")
            with mock.patch.object(proxy, "double", return_value=99):
                self.assertEqual(proxy.double(1), 99)
                self.assertEqual(factory.module.double(1), 99)
            self.assertEqual(proxy.double(3), 6, "패치가 끝나면 원래 속성으로 돌아온다")
            proxy.NEW = 1
            self.assertEqual(factory.module.NEW, 1)
            del proxy.NEW
            self.assertFalse(hasattr(factory.module, "NEW"))

    def test_missing_module_raises_on_first_use_not_at_creation(self):
        proxy = lazy_module("definitely_not_a_real_module_xyz")
        with self.assertRaises(ModuleNotFoundError):
            proxy.anything

    def test_real_module_round_trip_and_concurrent_first_access(self):
        proxy = LazyModule("json")
        results = []

        def use():
            results.append(proxy.dumps({"a": 1}))

        threads = [threading.Thread(target=use) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, ['{"a": 1}'] * 8)
        self.assertIs(proxy._load(), sys.modules["json"])
        self.assertIn("dumps", dir(proxy))


if __name__ == "__main__":
    unittest.main()
