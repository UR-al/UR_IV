"""cancel_check 를 받을 수 있는 어댑터에만 넘기는 공용 헬퍼 (감사 #31 — 판정 로직 단일화)."""
from __future__ import annotations

import unittest
from unittest import mock

from core.cancellable_call import accepts_keyword, call_with_optional_cancel


class CancellableCallTests(unittest.TestCase):
    def test_accepts_keyword_detects_named_and_var_keyword(self):
        def named(model, payload, progress_callback=None, cancel_check=None):
            return None

        def kwargs_only(*args, **kwargs):
            return None

        def legacy(model, payload, progress_callback=None):
            return None

        self.assertTrue(accepts_keyword(named, "cancel_check"))
        self.assertTrue(accepts_keyword(kwargs_only, "cancel_check"))
        self.assertFalse(accepts_keyword(legacy, "cancel_check"))
        self.assertFalse(accepts_keyword(object(), "cancel_check"))   # 시그니처 없음 → False

    def test_call_passes_cancel_check_only_when_supported(self):
        seen = {}

        def modern(model, payload, progress_callback=None, cancel_check=None):
            seen["modern"] = (model, payload, progress_callback, cancel_check)
            return "ok"

        def legacy(model, payload, progress_callback=None):
            seen["legacy"] = (model, payload, progress_callback)
            return "legacy"

        check = lambda: False
        self.assertEqual(call_with_optional_cancel(modern, "m", {}, progress_callback=print,
                                                   cancel_check=check), "ok")
        self.assertIs(seen["modern"][3], check)
        self.assertEqual(call_with_optional_cancel(legacy, "m", {}, progress_callback=print,
                                                   cancel_check=check), "legacy")
        self.assertIs(seen["legacy"][2], print)

    def test_none_cancel_check_is_not_forwarded(self):
        method = mock.Mock(return_value=1)
        call_with_optional_cancel(method, "a", cancel_check=None)
        method.assert_called_once_with("a")

    def test_generation_api_keeps_its_own_cancel_exception(self):
        from core.generation_api import GenerationApiManager, GenerationConflictError
        invoke = GenerationApiManager._invoke_cancellable
        with self.assertRaises(GenerationConflictError):
            invoke(mock.Mock(), "m", cancel_check=lambda: True)
        method = mock.Mock(return_value="done")
        self.assertEqual(invoke(method, "m", cancel_check=lambda: False), "done")
        self.assertIn("cancel_check", method.call_args.kwargs)   # Mock 은 **kwargs 를 받는다


if __name__ == "__main__":
    unittest.main()
