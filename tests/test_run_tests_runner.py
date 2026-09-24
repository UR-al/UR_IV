"""run_tests.py · tools/hook_run_tests.py 의 순수 로직 회귀 테스트.

- venv 자동 재실행 판단(시스템 python 으로 돌리면 가짜 ImportError 수십 개)과 main() 의 재실행 경로
- SLOW_MODULES dict → 편집 파일별 느린 모듈 선택(훅의 --quick --include)
- torch 표시 테스트: quick 필터 · 훅의 --with-torch 판단(속성 형태 표시 포함) · import 위치 추적
  (ImportTripwire) · torch 가 새면 main() 이 1 로 끝나고 훅이 그 위치를 전달하는지
- --durations 집계기(모듈별 실행 · discovery import)
- 소요 시간 실측치의 단일 출처(SLOW_MODULES 주석) — 문서·docstring 에 복붙 금지
"""
import contextlib
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types
import unittest
from unittest import mock

import run_tests
from tools import hook_run_tests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_case(module_name):
    """__module__ 만 다른 가짜 TestCase 인스턴스(느린 모듈 필터 검증용)."""
    cls = type("FakeCase", (unittest.TestCase,), {"test_ok": lambda self: None})
    cls.__module__ = module_name
    return cls("test_ok")


class SlowModuleRegistryTests(unittest.TestCase):
    def test_sources_are_derived_from_the_module_map(self):
        union = set()
        for sources in run_tests.SLOW_MODULES.values():
            union |= set(sources)
        self.assertEqual(run_tests.SLOW_MODULE_SOURCES, frozenset(union))

    def test_every_slow_module_and_source_exists(self):
        # 파일이 옮겨지거나 지워졌는데 목록만 남으면 훅이 조용히 느린 모듈을 안 돌린다.
        for module, sources in run_tests.SLOW_MODULES.items():
            test_file = os.path.join(ROOT, *module.split(".")) + ".py"
            self.assertTrue(os.path.isfile(test_file), f"느린 테스트 모듈 없음: {module}")
            self.assertTrue(sources, f"{module} 의 직접 소스가 비어 있음")
            for rel in sources:
                self.assertTrue(os.path.isfile(os.path.join(ROOT, *rel.split("/"))),
                                f"{module} 의 소스 경로가 없음: {rel}")

    def test_source_edit_selects_only_matching_slow_modules(self):
        hits = run_tests.slow_modules_for("core/generation_api.py")
        self.assertIn("tests.test_generation_api", hits)
        self.assertIn("tests.test_generation_api_remote_e2e", hits)
        self.assertNotIn("tests.test_backend_runtime", hits)
        self.assertEqual(run_tests.slow_modules_for("core/backend_runtime.py"),
                         frozenset({"tests.test_backend_runtime"}))

    def test_slow_test_file_itself_selects_its_module(self):
        self.assertEqual(run_tests.slow_modules_for("tests/test_backend_runtime.py"),
                         frozenset({"tests.test_backend_runtime"}))

    def test_windows_and_dot_prefixed_paths_are_normalised(self):
        self.assertEqual(run_tests.slow_modules_for("core\\backend_runtime.py"),
                         frozenset({"tests.test_backend_runtime"}))
        self.assertEqual(run_tests.slow_modules_for("./core/backend_runtime.py"),
                         frozenset({"tests.test_backend_runtime"}))

    def test_unrelated_edit_needs_no_slow_module(self):
        self.assertEqual(run_tests.slow_modules_for("core/tag_matcher.py"), frozenset())
        self.assertEqual(run_tests.slow_modules_for("tests/test_tag_matcher.py"), frozenset())
        self.assertEqual(run_tests.slow_modules_for(""), frozenset())

    def test_quick_filter_drops_slow_modules_except_included(self):
        slow = sorted(run_tests.SLOW_MODULES)[0]
        other_slow = sorted(run_tests.SLOW_MODULES)[1]
        suite = unittest.TestSuite([
            unittest.TestSuite([_fake_case("tests.test_fast_thing")]),
            _fake_case(slow),
            _fake_case(other_slow),
        ])
        kept = [type(t).__module__ for t in run_tests._iter_tests(run_tests._without_slow(suite))]
        self.assertEqual(kept, ["tests.test_fast_thing"])
        kept = [type(t).__module__
                for t in run_tests._iter_tests(run_tests._without_slow(suite, keep={slow}))]
        self.assertEqual(kept, ["tests.test_fast_thing", slow])

    def test_include_accepts_only_known_slow_modules(self):
        module = sorted(run_tests.SLOW_MODULES)[0]
        args = run_tests._parse_args(["--quick", "--include", module, "--durations", "5"])
        self.assertTrue(args.quick)
        self.assertEqual(args.include, [module])
        self.assertEqual(args.durations, 5)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            run_tests._parse_args(["--quick", "--include", "tests.test_not_slow"])
        self.assertIn("tests.test_not_slow", err.getvalue())


class VenvReexecTests(unittest.TestCase):
    def _make_venv(self, root):
        scripts = os.path.join(root, "venv", "Scripts")
        os.makedirs(scripts)
        python = os.path.join(scripts, "python.exe")
        with open(python, "wb"):
            pass
        return python

    def test_venv_python_is_none_without_a_venv(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(run_tests.venv_python(root))

    def test_venv_python_finds_the_windows_interpreter(self):
        with tempfile.TemporaryDirectory() as root:
            python = self._make_venv(root)
            self.assertEqual(run_tests.venv_python(root), python)

    def test_system_python_is_reexecuted_with_the_venv(self):
        with tempfile.TemporaryDirectory() as root:
            python = self._make_venv(root)
            command = run_tests.reexec_command(
                ["--quick"], root=root, prefix=os.path.join(root, "system-python"), environ={})
            self.assertEqual(command,
                             [python, os.path.join(root, "run_tests.py"), "--quick"])

    def test_already_inside_the_venv_runs_in_place(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_venv(root)
            self.assertIsNone(run_tests.reexec_command(
                [], root=root, prefix=os.path.join(root, "venv"), environ={}))

    def test_reexec_marker_prevents_a_loop(self):
        with tempfile.TemporaryDirectory() as root:
            self._make_venv(root)
            self.assertIsNone(run_tests.reexec_command(
                [], root=root, prefix=os.path.join(root, "elsewhere"),
                environ={run_tests.REEXEC_ENV: "1"}))

    def test_missing_venv_runs_with_the_current_interpreter(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(run_tests.reexec_command(
                [], root=root, prefix=os.path.join(root, "elsewhere"), environ={}))


class DurationTableTests(unittest.TestCase):
    def test_accumulates_and_ranks_slowest_first(self):
        table = run_tests.DurationTable()
        table.add("tests.a", 0.5, count=1)
        table.add("tests.b", 2.0, count=1)
        table.add("tests.a", 1.0, count=2)
        table.add("tests.c", -3.0)  # 음수(시계 역행)는 0 으로
        self.assertEqual(table.top(2), [("tests.b", 2.0, 1), ("tests.a", 1.5, 3)])
        self.assertAlmostEqual(table.total(), 3.5)
        self.assertEqual(table.top(0)[-1], ("tests.c", 0.0, 0))

    def test_ties_are_ordered_by_name_for_stable_output(self):
        table = run_tests.DurationTable()
        table.add("tests.z", 1.0)
        table.add("tests.a", 1.0)
        self.assertEqual([row[0] for row in table.top(5)], ["tests.a", "tests.z"])

    def test_format_lists_share_and_counts(self):
        table = run_tests.DurationTable()
        table.add("tests.a", 3.0, count=4)
        table.add("tests.b", 1.0, count=1)
        text = table.format("실행", 1)
        self.assertIn("합계 4.00s", text)
        self.assertIn("75.0%", text)
        self.assertIn("tests.a (4 tests)", text)
        self.assertNotIn("tests.b", text)
        self.assertNotIn("tests)", table.format("import", 5, show_counts=False))


class HookScopeTests(unittest.TestCase):
    def test_unrelated_edit_runs_plain_quick(self):
        self.assertEqual(hook_run_tests.runner_args("core/tag_matcher.py"), (["--quick"], "quick"))

    def test_slow_source_edit_adds_only_its_modules(self):
        args, label = hook_run_tests.runner_args("core/backend_runtime.py")
        self.assertEqual(args, ["--quick", "--include", "tests.test_backend_runtime"])
        self.assertEqual(label, "quick+test_backend_runtime")

    def test_hook_arguments_are_accepted_by_the_runner(self):
        args, _label = hook_run_tests.runner_args("core/generation_api.py")
        parsed = run_tests._parse_args(args)
        self.assertTrue(parsed.quick)
        self.assertEqual(sorted(parsed.include),
                         sorted(run_tests.slow_modules_for("core/generation_api.py")))

    def test_hook_interpreter_prefers_the_repo_venv(self):
        expected = run_tests.venv_python(ROOT)
        if expected is None:
            self.skipTest("저장소 venv 없음")
        self.assertEqual(hook_run_tests._interpreter(), expected)


class TimingSingleSourceTests(unittest.TestCase):
    """소요 시간 실측치는 run_tests.SLOW_MODULES 주석에만 적는다.

    CLAUDE.md 에 복붙한 '2.3초'가 테스트 수가 몇 배로 는 뒤에도 남아 있었다(감사 168). 문서와
    docstring 은 수치 없이 SLOW_MODULES 주석과 `--durations` 만 가리키게 한다.
    """
    # '39~43초' · '2.3초' · '21 초' · '9.3s' 같은 시간 표기(개수 '2,900개' · '--durations 15' 는 제외)
    _TIMING = re.compile(r"\d+(?:\.\d+)?\s*(?:~\s*\d+(?:\.\d+)?\s*)?(?:초|s\b)")

    def test_pattern_catches_the_old_drift_but_not_counts(self):
        self.assertTrue(self._TIMING.search("전체 12.6초 중 10.6초"))
        self.assertTrue(self._TIMING.search("quick 21~26초 + 3초"))
        self.assertTrue(self._TIMING.search("Ran 1284 tests in 9.31s"))
        self.assertIsNone(self._TIMING.search("약 2,900개 · run_tests.py --durations 15 · 5종"))

    def _claude_md_test_section(self):
        with open(os.path.join(ROOT, "CLAUDE.md"), encoding="utf-8") as f:
            text = f.read()
        match = re.search(r"^## 테스트[ \t]*$(.*?)(?=^## |\Z)", text, re.M | re.S)
        self.assertIsNotNone(match, "CLAUDE.md 에 '## 테스트' 절이 없다")
        return match.group(1)

    def test_claude_md_points_to_the_single_source_without_copying_numbers(self):
        section = self._claude_md_test_section()
        self.assertEqual(self._TIMING.findall(section), [],
                         "CLAUDE.md '## 테스트' 절에 소요 시간 수치가 있다 — run_tests.SLOW_MODULES 주석에만 적어라")
        self.assertIn("SLOW_MODULES", section)
        self.assertIn("--durations", section)

    def test_runner_and_hook_docstrings_carry_no_timing_numbers(self):
        for module in (run_tests, hook_run_tests):
            with self.subTest(module.__name__):
                self.assertEqual(self._TIMING.findall(module.__doc__ or ""), [])
                self.assertIn("SLOW_MODULES", module.__doc__ or "")

    def test_agent_guides_carry_no_timing_numbers(self):
        """Codex(AGENTS.md)·Copilot 안내도 같은 규칙 — 수치 대신 --durations 를 가리킨다."""
        for rel in ("AGENTS.md", os.path.join(".github", "copilot-instructions.md")):
            with self.subTest(rel), open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                self.assertEqual(self._TIMING.findall(f.read()), [])


def _marked_case(name, *, class_mark=False, method_mark=False, module="tests.test_fast_thing"):
    """REQUIRES_ATTR 표시만 붙인 가짜 TestCase(진짜 torch·_optional_deps 없이 필터 검증)."""
    def test_ok(self):
        return None

    if method_mark:
        setattr(test_ok, run_tests.REQUIRES_ATTR, frozenset({run_tests.TORCH}))
    cls = type(name, (unittest.TestCase,), {"test_ok": test_ok})
    cls.__module__ = module
    if class_mark:
        setattr(cls, run_tests.REQUIRES_ATTR, frozenset({run_tests.TORCH}))
    return cls("test_ok")


class TorchMarkedTestSelectionTests(unittest.TestCase):
    """--quick 은 torch 표시 테스트를 걸러 torch 를 올리지 않는다 — 훅은 관련 편집에만 되살린다."""

    def test_required_deps_reads_class_and_method_marks(self):
        self.assertEqual(run_tests.required_deps(_marked_case("Plain")), frozenset())
        self.assertEqual(run_tests.required_deps(_marked_case("Cls", class_mark=True)),
                         frozenset({run_tests.TORCH}))
        self.assertEqual(run_tests.required_deps(_marked_case("Meth", method_mark=True)),
                         frozenset({run_tests.TORCH}))

    def test_quick_filter_drops_marked_tests_only_when_asked(self):
        plain = _marked_case("Plain")
        by_class = _marked_case("ByClass", class_mark=True)
        by_method = _marked_case("ByMethod", method_mark=True)
        suite = unittest.TestSuite([plain, unittest.TestSuite([by_class, by_method])])

        def kept(**kwargs):
            return [type(t).__name__ for t in run_tests._iter_tests(run_tests._without_slow(suite, **kwargs))]

        self.assertEqual(kept(skip_deps={run_tests.TORCH}), ["Plain"])
        self.assertEqual(kept(), ["Plain", "ByClass", "ByMethod"])  # --with-torch

    def test_slow_and_torch_filters_combine(self):
        slow = sorted(run_tests.SLOW_MODULES)[0]
        suite = unittest.TestSuite([
            _marked_case("SlowPlain", module=slow),
            _marked_case("SlowTorch", class_mark=True, module=slow),
            _marked_case("FastPlain"),
        ])
        kept = [type(t).__name__ for t in run_tests._iter_tests(
            run_tests._without_slow(suite, keep={slow}, skip_deps={run_tests.TORCH}))]
        self.assertEqual(kept, ["SlowPlain", "FastPlain"])

    def test_with_torch_flag_is_parsed(self):
        self.assertTrue(run_tests._parse_args(["--quick", "--with-torch"]).with_torch)
        self.assertFalse(run_tests._parse_args(["--quick"]).with_torch)

    def test_torch_sources_select_torch_tests(self):
        for rel in ("comfy_custom_nodes/ai_studio_forge_parity/sam3_nodes.py",
                    "comfy_custom_nodes\\ai_studio_forge_parity\\mask_ops.py",
                    "./comfy_custom_nodes/ai_studio_forge_parity/h3_cache_nodes.py",
                    "core/creator_workflows.py",
                    run_tests.TORCH_MARK_HELPER):
            with self.subTest(rel):
                self.assertTrue(run_tests.torch_tests_for(rel))
        for rel in ("comfy_custom_nodes_backup/x.py", "core/creator_workflows_extra.py",
                    "core/tag_matcher.py", "", None):
            with self.subTest(rel):
                self.assertFalse(run_tests.torch_tests_for(rel))

    def test_marked_test_file_selects_torch_tests_by_reading_it(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "tests"))
            for name, body in (("test_marked.py", "@requires_torch\nclass T: pass\n"),
                               ("test_marked_method.py", "class T:\n    @requires_torch\n    def test_a(self): pass\n"),
                               # 속성 형태 — tests/test_optional_deps 규칙과 run_tests.required_deps 가 표시로 본다
                               ("test_marked_attr.py", "@_optional_deps.requires_torch\nclass T: pass\n"),
                               ("test_marked_attr_method.py",
                                "class T:\n    @_optional_deps.requires_torch\n    def test_a(self): pass\n"),
                               # 편집 중 문법이 깨진 파일 — 줄 모양으로 판단(표시가 보이면 torch 쪽)
                               ("test_marked_broken.py", "@_optional_deps.requires_torch\nclass T(:\n"),
                               ("test_mentions.py", '"""requires_torch 규칙을 설명만 한다."""\n'
                                                    "X = 'requires_torch'\n"),
                               # docstring 속 예시 줄은 데코레이터가 아니다(줄 모양만 보면 걸린다)
                               ("test_docstring_example.py", '"""예:\n\n    @requires_torch\n    class T: ...\n"""\n'),
                               ("test_plain.py", "class T: pass\n")):
                with open(os.path.join(root, "tests", name), "w", encoding="utf-8") as f:
                    f.write(body)
            for name in ("test_marked", "test_marked_method", "test_marked_attr", "test_marked_attr_method",
                         "test_marked_broken"):
                with self.subTest(name):
                    self.assertTrue(run_tests.torch_tests_for(f"tests/{name}.py", root))
            # 이름만 언급하는 파일(규칙·러너 테스트)을 고쳤다고 torch 를 올리지 않는다.
            for name in ("test_mentions", "test_docstring_example", "test_plain", "test_missing"):
                with self.subTest(name):
                    self.assertFalse(run_tests.torch_tests_for(f"tests/{name}.py", root))

    def test_decorator_names_cover_bare_attribute_and_call_forms(self):
        import ast

        tree = ast.parse("@a\n@m.b\n@c()\n@m.n.d(1)\ndef f(): pass\n")
        self.assertEqual(run_tests.decorator_names(tree.body[0]), {"a", "b", "c", "d"})
        self.assertEqual(run_tests.decorator_names(tree), set())  # 데코레이터가 없는 노드

    def test_real_marked_modules_are_detected_and_rule_tests_are_not(self):
        self.assertTrue(run_tests.torch_tests_for("tests/test_comfy_sam3_nodes.py"))
        self.assertTrue(run_tests.torch_tests_for("tests/test_h3_conditioning_cache.py"))
        self.assertFalse(run_tests.torch_tests_for("tests/test_run_tests_runner.py"))

    def test_every_torch_source_exists(self):
        for source in run_tests.TORCH_TEST_SOURCES:
            path = os.path.join(ROOT, *source.rstrip("/").split("/"))
            with self.subTest(source):
                self.assertTrue(os.path.isdir(path) if source.endswith("/") else os.path.isfile(path))
        self.assertTrue(os.path.isfile(os.path.join(ROOT, *run_tests.TORCH_MARK_HELPER.split("/"))))

    def test_hook_adds_with_torch_only_for_torch_edits(self):
        args, label = hook_run_tests.runner_args("comfy_custom_nodes/ai_studio_forge_parity/mask_ops.py")
        self.assertEqual((args, label), (["--quick", "--with-torch"], "quick+torch"))
        self.assertTrue(run_tests._parse_args(args).with_torch)
        self.assertNotIn("--with-torch", hook_run_tests.runner_args("core/tag_matcher.py")[0])

    def test_hook_combines_slow_module_and_torch_scopes(self):
        with mock.patch.object(run_tests, "torch_tests_for", return_value=True):
            args, label = hook_run_tests.runner_args("core/backend_runtime.py")
        self.assertEqual(args, ["--quick", "--include", "tests.test_backend_runtime", "--with-torch"])
        self.assertEqual(label, "quick+test_backend_runtime+torch")


class ImportTripwireTests(unittest.TestCase):
    """--quick 이 torch 를 그래도 올리면 '어디서' 올렸는지 알려 준다(관찰만 — import 는 막지 않는다)."""

    _NAME = "uriv_tripwire_probe_mod"

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        with open(os.path.join(self._dir.name, self._NAME + ".py"), "w", encoding="utf-8") as f:
            f.write("VALUE = 1\n")
        self.tripwire = run_tests.ImportTripwire(self._NAME)
        patches = (mock.patch.object(sys, "meta_path", [self.tripwire, *sys.meta_path]),
                   mock.patch.object(sys, "path", [self._dir.name, *sys.path]),
                   mock.patch.dict(sys.modules))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        sys.modules.pop(self._NAME, None)
        importlib.invalidate_caches()

    def test_nothing_to_report_until_the_module_is_really_imported(self):
        self.assertIsNone(self.tripwire.report())
        importlib.util.find_spec(self._NAME)  # 설치 확인(import 아님)은 보고하지 않는다
        self.assertIsNone(self.tripwire.report())

    def test_report_names_the_repo_frame_that_imported_it(self):
        module = importlib.import_module(self._NAME)  # 이 파일(저장소 안)에서 import
        self.assertEqual(module.VALUE, 1)
        report = self.tripwire.report()
        self.assertIsNotNone(report)
        self.assertIn(self._NAME, report)
        self.assertIn("tests/test_run_tests_runner.py:", report)
        self.assertIn("requires_torch", report)

    def test_frames_outside_the_repo_are_summarised(self):
        tripwire = run_tests.ImportTripwire(self._NAME, root=self._dir.name)
        self.assertEqual(tripwire._repo_frames([]), "(저장소 밖)")

    def test_venv_frames_are_skipped_so_our_code_stays_visible(self):
        """서드파티(venv)를 거쳐 올라오면 마지막 프레임이 전부 site-packages 라 우리 코드가 잘렸다."""
        def frame(*parts, line):
            return traceback.FrameSummary(os.path.join(run_tests.ROOT, *parts), line, "f", lookup_line=False)

        stack = [frame("tests", "test_x.py", line=3), frame("core", "y.py", line=7)]
        stack += [frame("venv", "Lib", "site-packages", "lib3p", f"m{i}.py", line=i) for i in range(6)]
        self.assertEqual(self.tripwire._repo_frames(stack), "tests/test_x.py:3 -> core/y.py:7")

    def test_find_spec_while_another_module_imports_is_not_an_import(self):
        """회귀: discovery 가 테스트 모듈을 import 하는 중 클래스 본문의 @requires_torch 가
        torch_installed → importlib.util.find_spec 을 부른다 — 스택 바깥의 _find_and_load 는
        다른 모듈 것이므로 torch import 로 치면 안 된다."""
        outer = "uriv_tripwire_outer_mod"
        with open(os.path.join(self._dir.name, outer + ".py"), "w", encoding="utf-8") as f:
            f.write(f"import importlib.util\nSPEC = importlib.util.find_spec({self._NAME!r})\n")
        sys.modules.pop(outer, None)
        importlib.invalidate_caches()
        self.assertIsNotNone(importlib.import_module(outer).SPEC)
        self.assertNotIn(self._NAME, sys.modules)
        self.assertIsNone(self.tripwire.report())

    def test_import_undone_by_patch_dict_is_still_reported(self):
        """mock.patch.dict(sys.modules) 가 끝나며 모듈을 도로 빼도 올렸던 사실(과 위치)은 남는다."""
        with mock.patch.dict(sys.modules):
            importlib.import_module(self._NAME)
        self.assertNotIn(self._NAME, sys.modules)
        report = self.tripwire.report("discovery")
        self.assertIsNotNone(report)
        self.assertIn("discovery", report)
        self.assertIn("tests/test_run_tests_runner.py:", report)

    def test_module_put_straight_into_sys_modules_is_reported_without_location(self):
        sys.modules[self._NAME] = types.ModuleType(self._NAME)
        report = self.tripwire.report()
        self.assertIsNotNone(report)
        self.assertIn("알 수 없음", report)

    def test_first_import_location_is_kept(self):
        importlib.import_module(self._NAME)
        first = self.tripwire.where
        sys.modules.pop(self._NAME)
        importlib.import_module(self._NAME)
        self.assertEqual(self.tripwire.where, first)


class MainTorchGuardTests(unittest.TestCase):
    """main() 은 torch 가 새면 테스트가 다 통과해도 1 로 끝낸다 — 경고만 쓰면 훅(종료코드만 본다)과
    /verify 가 못 본다(감사 #168 후속). torch 대신 가짜 모듈 이름으로 같은 경로를 돈다."""

    _NAME = "uriv_main_guard_probe_mod"

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        with open(os.path.join(self._dir.name, self._NAME + ".py"), "w", encoding="utf-8") as f:
            f.write("VALUE = 1\n")
        patches = (mock.patch.object(run_tests, "TORCH", self._NAME),
                   mock.patch.object(run_tests, "reexec_command", return_value=None),
                   mock.patch.object(sys, "meta_path", list(sys.meta_path)),
                   mock.patch.object(sys, "path", [self._dir.name, *sys.path]),
                   mock.patch.dict(sys.modules),
                   mock.patch.dict(os.environ, {run_tests.SKIP_DEPS_ENV: "outer"}))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        sys.modules.pop(self._NAME, None)
        importlib.invalidate_caches()
        self.seen_skip = []

    def _case(self, *, imports=False, fails=False):
        name, seen = self._NAME, self.seen_skip

        def test_probe(case):
            seen.append(run_tests.active_skip_deps())
            if imports:
                importlib.import_module(name)  # 표시 없는 테스트가 (전이) import 하는 자리
            if fails:
                case.fail("일부러 실패")

        cls = type("ProbeCase", (unittest.TestCase,), {"test_probe": test_probe})
        cls.__module__ = "tests.test_fast_thing"
        return cls("test_probe")

    def _main(self, argv, case, *, discovery_imports=False):
        sys.modules.pop(self._NAME, None)
        name = self._NAME

        def discover(*_args, **_kwargs):
            if discovery_imports:
                importlib.import_module(name)  # 테스트 모듈 import 가 끌어오는 자리
            return unittest.TestSuite([case])

        with mock.patch.object(run_tests.unittest.TestLoader, "discover", side_effect=discover), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = run_tests.main(argv)
        self.assertEqual(os.environ.get(run_tests.SKIP_DEPS_ENV), "outer", "실행 뒤 환경변수를 되돌려야 한다")
        self.assertNotIn(run_tests.ImportTripwire, [type(f) for f in sys.meta_path])
        return code, err.getvalue()

    def test_quick_run_that_imports_it_fails_with_the_location(self):
        code, err = self._main(["--quick"], self._case(imports=True))
        self.assertEqual(code, 1)
        self.assertIn(f"--quick 실행 중에 {self._NAME} 가 import 됐다", err)
        self.assertIn("tests/test_run_tests_runner.py:", err)
        self.assertTrue(err.rstrip().endswith("import 하는 모든 테스트가 torch 를 올린다)."),
                        "훅은 stderr 꼬리만 보여 준다 — 실패 사유가 맨 끝에 와야 한다")

    def test_quick_run_without_the_import_passes_and_publishes_the_mode(self):
        code, err = self._main(["--quick"], self._case())
        self.assertEqual(code, 0, err)
        self.assertEqual(self.seen_skip, [frozenset({self._NAME})])
        self.assertNotIn("[run_tests] 실패", err)

    def test_discovery_import_fails_in_every_mode(self):
        for argv in (["--quick"], [], ["--quick", "--with-torch"]):
            with self.subTest(argv=argv):
                code, err = self._main(argv, self._case(), discovery_imports=True)
                self.assertEqual(code, 1)
                self.assertIn(f"discovery(테스트 모듈 import) 중에 {self._NAME} 가 import 됐다", err)
                self.assertIn("tests/test_run_tests_runner.py:", err)

    def test_runs_that_keep_torch_tests_may_import_it_while_running(self):
        for argv in ([], ["--quick", "--with-torch"]):
            with self.subTest(argv=argv):
                self.seen_skip.clear()
                code, err = self._main(argv, self._case(imports=True))
                self.assertEqual(code, 0, err)
                self.assertEqual(self.seen_skip, [frozenset()])

    def test_a_failing_test_still_returns_one(self):
        code, err = self._main(["--quick"], self._case(fails=True))
        self.assertEqual(code, 1)
        self.assertIn("일부러 실패", err)
        self.assertNotIn("[run_tests] 실패", err)

    def test_active_skip_deps_parses_the_published_list(self):
        self.assertEqual(run_tests.active_skip_deps({}), frozenset())
        self.assertEqual(run_tests.active_skip_deps({run_tests.SKIP_DEPS_ENV: ""}), frozenset())
        self.assertEqual(run_tests.active_skip_deps({run_tests.SKIP_DEPS_ENV: " torch , x ,"}),
                         frozenset({"torch", "x"}))


class HookFeedbackTests(unittest.TestCase):
    """러너가 1 을 돌려주면 훅이 stderr 꼬리(= torch 실패 사유·위치)를 Claude 에게 넘긴다(exit 2)."""

    def _hook(self, returncode, stderr=""):
        payload = json.dumps({"tool_input": {"file_path": os.path.join(ROOT, "core", "tag_matcher.py")}})
        completed = subprocess.CompletedProcess(["python"], returncode, stdout="", stderr=stderr)
        with mock.patch.object(hook_run_tests.subprocess, "run", return_value=completed) as run, \
                mock.patch.object(sys, "stdin", io.StringIO(payload)), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = hook_run_tests.main()
        return code, err.getvalue(), run

    def test_runner_failure_reaches_claude_with_the_import_location(self):
        tail = ("Ran 3 tests\n\nOK\n\n[run_tests] 실패: --quick 실행 중에 torch 가 import 됐다 — ... "
                "처음 import 한 곳: tests/test_x.py:3 -> core/y.py:7\n")
        code, err, run = self._hook(1, tail)
        self.assertEqual(code, 2)
        self.assertIn("처음 import 한 곳: tests/test_x.py:3 -> core/y.py:7", err)
        self.assertIn("--quick", run.call_args[0][0])

    def test_passing_run_stays_silent(self):
        code, err, _run = self._hook(0)
        self.assertEqual((code, err), (0, ""))


class MainReexecTests(unittest.TestCase):
    """main() 은 venv 밖이면 테스트를 한 줄도 돌리지 않고 venv 로 넘긴다(인자·종료코드 그대로)."""

    def test_reexec_passes_arguments_marker_and_exit_code_through(self):
        command = ["C:/repo/venv/Scripts/python.exe", "C:/repo/run_tests.py", "--quick", "--durations", "3"]
        with mock.patch.object(run_tests, "reexec_command", return_value=command) as build, \
                mock.patch.object(run_tests.subprocess, "call", return_value=7) as call, \
                mock.patch.object(run_tests.unittest.TestLoader, "discover") as discover, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = run_tests.main(["--quick", "--durations", "3"])
        self.assertEqual(code, 7)
        build.assert_called_once_with(["--quick", "--durations", "3"])
        discover.assert_not_called()
        (called,), kwargs = call.call_args
        self.assertEqual(called, command)
        self.assertEqual(kwargs["cwd"], run_tests.ROOT)
        self.assertEqual(kwargs["env"][run_tests.REEXEC_ENV], "1")
        self.assertIn("재실행", err.getvalue())


class DurationCollectorTests(unittest.TestCase):
    def test_timing_result_charges_each_test_to_its_module(self):
        table = run_tests.DurationTable()
        with mock.patch.object(run_tests._TimingResult, "table", table):
            runner = unittest.TextTestRunner(stream=io.StringIO(), resultclass=run_tests._TimingResult)
            runner.run(unittest.TestSuite([_fake_case("tests.test_a"), _fake_case("tests.test_a"),
                                           _fake_case("tests.test_b")]))
        self.assertEqual({row[0]: row[2] for row in table.top(0)}, {"tests.test_a": 2, "tests.test_b": 1})

    def test_timed_loader_records_module_import_time(self):
        loader = run_tests._TimedLoader()
        loader.table = run_tests.DurationTable()
        loader._get_module_from_name("json")
        self.assertEqual([row[0] for row in loader.table.top(0)], ["json"])


if __name__ == "__main__":
    unittest.main()
