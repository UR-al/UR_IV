"""tests/_optional_deps(torch 표시·지연 import) 와 그 규칙의 회귀 테스트.

감사 #168: --quick(훅)이 torch 를 올리지 않게 하려면 두 가지가 동시에 지켜져야 한다.
  1) torch 를 쓰는 테스트는 @requires_torch 로 표시(run_tests --quick 이 거른다)
  2) 테스트 모듈이 최상위에서 torch 를 import 하지 않는다(discover 는 거른 모듈도 import 한다)
여기서는 헬퍼 동작(진짜 torch 를 올리지 않고 가짜 모듈로)과, tests/test_*.py 전체가 규칙을
지키는지(AST)와, 표시된 테스트 모듈을 import 해도 torch 가 올라오지 않는지를 본다.

이 모듈 자신도 quick 에서 돈다 — 규칙 검사가 quick 을 도로 느리게 만들지 않게, AST 는 'torch' 가
나오는 파일만 파싱하고, import 검사는 torch 가 아직 없는 프로세스(quick·단독 실행)에서는 하위
프로세스 없이 그 자리에서 증명한다. quick(run_tests.active_skip_deps 에 torch)인데 torch 가 이미
있으면 하위 프로세스로 빠지지 않고 실패한다 — 표시 없는 테스트의 전이 import 가 새는 경로다.
표시 데코레이터 판단은 훅과 같은 run_tests.decorator_names 를 쓴다(속성 형태 포함).
"""
import ast
import importlib
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import run_tests
from tests import _optional_deps

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
_SELF = os.path.basename(__file__)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_LOADERS = frozenset({"load_torch", "bind_torch"})
# unittest 테스트 클래스로 보는 부모 이름(같은 모듈의 테스트 클래스를 상속해도 테스트 클래스다).
_TEST_BASES = frozenset({"TestCase", "IsolatedAsyncioTestCase"})


def _test_sources():
    """{파일 이름: 소스} — discover 대상(tests/test_*.py) 중 'torch' 가 나오는 파일.

    torch 를 import 하거나 load_torch/bind_torch/requires_torch 를 쓰려면 소스에 'torch' 가 있어야
    하므로, 나머지 수백 개 파일은 파싱할 필요가 없다(전부 파싱하면 quick 에 1초 넘게 더해진다).
    """
    out = {}
    for name in sorted(os.listdir(TESTS_DIR)):
        if name.startswith("test_") and name.endswith(".py"):
            with open(os.path.join(TESTS_DIR, name), encoding="utf-8") as f:
                text = f.read()
            if run_tests.TORCH in text:
                out[name] = text
    return out


def _is_torch_import(node) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == "torch" or a.name.startswith("torch.") for a in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return node.level == 0 and (module == "torch" or module.startswith("torch."))
    return False


def _import_time_torch_imports(tree):
    """함수 밖(모듈·클래스 본문 — import 시점에 실행되는 곳)의 torch import 줄 번호."""
    lines, stack = [], [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                continue
            if _is_torch_import(child):
                lines.append(child.lineno)
            stack.append(child)
    return sorted(lines)


def _called_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _binds_name_locally(fn, name) -> bool:
    """함수(중첩 포함) 안에서 name 을 지역 이름으로 묶나 — 매개변수·대입·for/with 대상·import 별칭.

    `global name` 이 있으면 모듈 전역을 쓰는 것이므로 지역이 아니다.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Global) and name in node.names:
            return False
    for node in ast.walk(fn):
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                return True
    return False


def _loads_torch(fn, loaders, *, global_torch=False) -> bool:
    """함수(중첩 포함)가 torch 를 쓰나.

    - torch 를 import 하거나 torch 로더(load_torch·bind_torch·그것을 부르는 헬퍼)를 부른다.
    - global_torch(모듈이 bind_torch 로 전역 ``torch`` 를 채우는 모양)이면, 지역으로 묶지 않은
      이름 ``torch`` 를 읽는 것도 torch 사용이다 — 표시 없는 테스트가 그 전역을 읽으면 quick 에선
      None 이고, 전체 실행에선 표시된 클래스가 먼저 돌았는지에 따라 결과가 갈린다.
    """
    for node in ast.walk(fn):
        if _is_torch_import(node):
            return True
        if isinstance(node, ast.Call) and _called_name(node) in loaders:
            return True
    if global_torch and not _binds_name_locally(fn, run_tests.TORCH):
        return any(isinstance(node, ast.Name) and node.id == run_tests.TORCH
                   and isinstance(node.ctx, ast.Load) for node in ast.walk(fn))
    return False


def _torch_import_check_mode():
    """표시된 모듈 import 검사를 어떻게 할지.

    - "quick": 지금 run_tests 가 torch 표시 테스트를 거른 실행이다 — torch 가 올라와 있으면 그 자체가 회귀.
    - "here": torch 가 아직 없다(단독 실행 등) — 그 자리에서 import 해 보면 된다.
    - "subprocess": 전체 실행처럼 torch 테스트가 이미 torch 를 올렸다 — 새 인터프리터에서 본다.
    """
    if run_tests.TORCH in run_tests.active_skip_deps():
        return "quick"
    if run_tests.TORCH not in sys.modules:
        return "here"
    return "subprocess"


# 표시 데코레이터를 알아보는 규칙은 훅(run_tests.torch_tests_for → uses_torch_mark)과 **같은 함수**를
# 쓴다 — 여기서만 속성 형태(@_optional_deps.requires_torch)를 받으면 훅이 그 파일 편집에 torch 테스트를
# 빼고 돌려 표시한 테스트가 영영 안 돈다(감사 #168 후속).
_decorator_names = run_tests.decorator_names


def _base_names(cls):
    names = set()
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _methods(cls):
    return [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _test_class_names(classes):
    """unittest 테스트 클래스 — TestCase 계열(같은 모듈 상속 포함)이거나 test* 메서드가 있다."""
    names = {cls.name for cls in classes if any(m.name.startswith("test") for m in _methods(cls))}
    grown = True
    while grown:
        grown = False
        for cls in classes:
            if cls.name not in names and _base_names(cls) & (_TEST_BASES | names):
                names.add(cls.name)
                grown = True
    return names


def _unmarked_torch_users(tree):
    """torch 를 쓰는데 @requires_torch 표시가 없는 곳 — [(줄, 설명)].

    - 모듈 헬퍼 함수와 테스트가 아닌 헬퍼 클래스(가짜 모델 등)가 torch 를 쓰면 그 이름도 로더로
      친다(고정점) — 표시 없는 테스트가 그 헬퍼를 부르거나 헬퍼 클래스를 만들면 잡힌다.
    - 모듈이 bind_torch 로 전역 torch 를 채우면 전역 ``torch`` 참조도 torch 사용이다(_loads_torch).
    - 테스트 클래스는 자기 표시 또는 같은 모듈의 표시된 부모 클래스로 표시된 것으로 본다.
    - setUpModule 은 torch 를 쓰면 안 된다(같은 모듈의 torch 없는 테스트가 돌 때도 실행된다).
    """
    global_torch = any(isinstance(node, ast.Call) and _called_name(node) == "bind_torch"
                       for node in ast.walk(tree))
    loaders = set(_LOADERS)
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    marked_classes = set()
    for cls in classes:  # 부모는 자식보다 먼저 정의된다 — 한 번 훑으면 상속 표시까지 모인다
        if run_tests.TORCH_MARK in _decorator_names(cls) or _base_names(cls) & marked_classes:
            marked_classes.add(cls.name)
    # 표시된 클래스는 (test* 메서드가 없는 공통 부모여도) 테스트 클래스 쪽이다 — 헬퍼로 치지 않는다.
    test_classes = _test_class_names(classes) | marked_classes
    helpers = [(fn.name, [fn]) for fn in functions if fn.name not in ("setUpModule", "tearDownModule")]
    helpers += [(cls.name, _methods(cls)) for cls in classes if cls.name not in test_classes]
    grown = True
    while grown:
        grown = False
        for name, bodies in helpers:
            if name not in loaders and any(
                    _loads_torch(body, loaders, global_torch=global_torch) for body in bodies):
                loaders.add(name)
                grown = True
    problems = []
    for fn in functions:
        if fn.name in ("setUpModule", "tearDownModule") and _loads_torch(
                fn, loaders, global_torch=global_torch):
            problems.append((fn.lineno, f"{fn.name} 가 torch 를 올린다"))
    for cls in classes:
        if cls.name not in test_classes or cls.name in marked_classes:
            continue
        for method in _methods(cls):
            if run_tests.TORCH_MARK in _decorator_names(method):
                continue
            if _loads_torch(method, loaders, global_torch=global_torch):
                problems.append((method.lineno, f"{cls.name}.{method.name} 가 표시 없이 torch 를 쓴다"))
    return problems


class TorchImportDisciplineTests(unittest.TestCase):
    """tests/test_*.py 전체가 규칙을 지키는지 — 새 테스트가 최상위 `import torch` 로 되돌리지 않게."""

    @classmethod
    def setUpClass(cls):
        cls.sources = _test_sources()
        cls.trees = {name: ast.parse(text, filename=name) for name, text in cls.sources.items()}

    def test_no_test_module_imports_torch_at_import_time(self):
        offenders = {name: lines for name, tree in self.trees.items()
                     if (lines := _import_time_torch_imports(tree))}
        self.assertEqual(
            offenders, {},
            "테스트 모듈 최상위(또는 클래스 본문)에서 torch 를 import 한다 — discover 는 --quick 이 거를 "
            "모듈도 import 하므로 훅이 매번 torch 를 올린다. tests/_optional_deps 의 load_torch/"
            "bind_torch 로 지연 import 하라.")

    def test_tests_that_load_torch_are_marked(self):
        # 이 파일은 헬퍼(load_torch·bind_torch)를 가짜 torch 로 시험한다 — 진짜 torch 를 올리지 않는다.
        offenders = {name: problems for name, tree in self.trees.items()
                     if name != _SELF and (problems := _unmarked_torch_users(tree))}
        self.assertEqual(
            offenders, {},
            "torch 를 쓰는 테스트에 @requires_torch 표시가 없다 — --quick 이 거르지 못해 torch 를 "
            "올리고(전역 torch 를 읽으면 quick 에선 None), torch 없는 환경에선 skip 대신 ImportError 가 난다.")

    def test_source_filter_keeps_every_torch_user(self):
        """'torch' 가 없는 파일만 건너뛴다 — 표시된 모듈과 가짜 torch 를 쓰는 모듈은 검사 대상에 남는다."""
        for name in ("test_comfy_sam3_nodes.py", "test_forge_parity_mask_pag.py", "test_h3_conditioning_cache.py",
                     _SELF):
            self.assertIn(name, self.sources)
        self.assertNotIn("test_tag_matcher.py", self.sources)

    def test_rules_catch_the_old_patterns(self):
        """규칙 자체의 회귀 — 예전 모양(최상위 import · try-import · 표시 없는 지연 import)을 잡는다."""
        cases = {
            "import torch\n": ([1], []),
            "try:\n    import torch\nexcept ImportError:\n    torch = None\n": ([2], []),
            "import torch.nn.functional as F\n": ([1], []),
            "class T:\n    import torch\n": ([2], []),
            # 표시 없는 테스트 메서드의 지연 import — 메서드 줄(3)을 가리킨다.
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        import torch\n":
                ([], [3]),
            "def _t():\n    return load_torch()\nclass T:\n    def test_a(self):\n        _t()\n":
                ([], [4]),
            "def setUpModule():\n    bind_torch(globals())\n": ([], [1]),
            # bind_torch 로 채우는 전역 torch 를 표시 없는 테스트가 읽는다(quick 에선 None).
            ("torch = None\n"
             "@requires_torch\n"
             "class A(unittest.TestCase):\n"
             "    @classmethod\n"
             "    def setUpClass(cls):\n"
             "        bind_torch(globals())\n"
             "class B(unittest.TestCase):\n"
             "    def test_b(self):\n"
             "        torch.zeros(1)\n"): ([], [8]),
            # 전역 torch 를 쓰는 가짜 모델 클래스를 헬퍼 함수를 거쳐 표시 없는 테스트가 만든다.
            ("torch = None\n"
             "class _Fake:\n"
             "    def __init__(self):\n"
             "        self.device = torch.device('meta')\n"
             "def _make():\n"
             "    return _Fake()\n"
             "@requires_torch\n"
             "class A(unittest.TestCase):\n"
             "    @classmethod\n"
             "    def setUpClass(cls):\n"
             "        bind_torch(globals())\n"
             "    def test_a(self):\n"
             "        _make()\n"
             "class B(unittest.TestCase):\n"
             "    def test_b(self):\n"
             "        _make()\n"): ([], [15]),
        }
        for source, (import_lines, problem_lines) in cases.items():
            with self.subTest(source=source):
                tree = ast.parse(source)
                self.assertEqual(_import_time_torch_imports(tree), import_lines)
                self.assertEqual([line for line, _why in _unmarked_torch_users(tree)], problem_lines)

    def test_rules_accept_the_sanctioned_patterns(self):
        source = (
            "torch = None\n"
            "@requires_torch\n"
            "class A:\n"
            "    @classmethod\n"
            "    def setUpClass(cls):\n"
            "        bind_torch(globals())\n"
            "class B(A):\n"
            "    def test_b(self):\n"
            "        import torch\n"
            "    def test_global(self):\n"
            "        torch.zeros(1)\n"
            "class C(unittest.TestCase):\n"
            "    @requires_torch\n"
            "    def test_c(self):\n"
            "        torch = load_torch()\n"
            "    def test_fake(self):\n"
            "        with mock.patch.dict(sys.modules, {'torch': object()}):\n"
            "            pass\n"
            "    def test_local_fake(self):\n"
            "        torch = types.ModuleType('torch')\n"
            "        torch.long = 'int64'\n"
            "class _Fake:\n"
            "    def __init__(self):\n"
            "        self.device = torch.device('meta')\n"
            "@requires_torch\n"
            "class D(unittest.TestCase):\n"
            "    def test_d(self):\n"
            "        _Fake()\n"
        )
        tree = ast.parse(source)
        self.assertEqual(_import_time_torch_imports(tree), [])
        self.assertEqual(_unmarked_torch_users(tree), [])

    def test_marked_test_modules_do_not_import_torch_when_imported(self):
        """표시가 든 테스트 모듈(과 그 소스)을 import 해도 torch 가 올라오지 않아야 quick 이 빠르다.

        --quick 실행(run_tests 가 torch 를 거른다고 알린다)에서는 torch 가 **이미** 올라와 있는 것
        자체가 회귀다 — 표시된 테스트는 다 걸러졌으니 표시 없는 테스트나 그 소스(전이 import 포함)가
        올렸다. 예전엔 이때 하위 프로세스로 빠져 표시된 모듈만 다시 import 해 보고 통과했다.
        torch 가 아직 없는 프로세스(quick · 이 파일 단독 실행)에서는 그 자리에서 import 해 보고
        증명한다(discover 가 이미 import 했으면 공짜다). 전체 실행처럼 torch 테스트가 이미 torch 를
        올린 프로세스에서는 섞이지 않게 새 인터프리터에서 확인한다.
        """
        marked = sorted(name[:-3] for name, text in self.sources.items()
                        if run_tests.uses_torch_mark(text) and name != _SELF)
        self.assertTrue(marked, "requires_torch 로 표시한 테스트 모듈을 찾지 못했다(탐지가 깨졌다)")
        mode = _torch_import_check_mode()
        if mode == "quick":
            self.assertNotIn(
                run_tests.TORCH, sys.modules,
                "--quick 인데 torch 가 이미 올라와 있다 — 표시 없는 테스트나 그 소스가 (전이) import 했다. "
                "run_tests 가 끝에 적는 '처음 import 한 곳'을 보라.")
        if mode in ("quick", "here"):
            for name in marked:
                importlib.import_module("tests." + name)
                self.assertNotIn(run_tests.TORCH, sys.modules,
                                 f"tests.{name} 을 import 하자 torch 가 올라왔다(최상위·전이 import)")
            return
        script = (
            "import importlib, sys\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            f"for name in {marked!r}:\n"
            "    importlib.import_module('tests.' + name)\n"
            "print('torch' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False",
                         f"표시된 테스트 모듈 import 가 torch 를 올렸다: {marked}")

    def test_quick_run_fails_when_torch_is_already_loaded(self):
        """리뷰 재현(감사 #168 후속): quick 인데 torch 가 이미 sys.modules 에 있으면 위 검사가 실패한다.

        예전엔 'torch 가 이미 있다 → 하위 프로세스에서 표시된 모듈만 import' 로 빠져 통과했다.
        """
        fake = types.ModuleType(run_tests.TORCH)
        result = unittest.TestResult()
        with mock.patch.dict(os.environ, {run_tests.SKIP_DEPS_ENV: run_tests.TORCH}), \
                mock.patch.dict(sys.modules, {run_tests.TORCH: fake}):
            self.assertEqual(_torch_import_check_mode(), "quick")
            type(self)("test_marked_test_modules_do_not_import_torch_when_imported").run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 1)
        self.assertIn("--quick 인데 torch 가 이미 올라와 있다", result.failures[0][1])

    def test_check_mode_outside_quick_keeps_the_in_process_and_subprocess_paths(self):
        fake = types.ModuleType(run_tests.TORCH)
        # 전체·--with-torch 실행: run_tests 가 빈 목록을 알린다 → torch 가 있으면 새 인터프리터에서 본다.
        with mock.patch.dict(os.environ, {run_tests.SKIP_DEPS_ENV: ""}), \
                mock.patch.dict(sys.modules, {run_tests.TORCH: fake}):
            self.assertEqual(_torch_import_check_mode(), "subprocess")
        # run_tests 밖(단독 실행)이고 torch 가 아직 없다 → 그 자리에서.
        with mock.patch.dict(os.environ), mock.patch.dict(sys.modules):
            os.environ.pop(run_tests.SKIP_DEPS_ENV, None)
            sys.modules.pop(run_tests.TORCH, None)
            self.assertEqual(_torch_import_check_mode(), "here")

    def test_hook_and_rule_recognise_the_same_marks(self):
        """훅(run_tests.uses_torch_mark)과 이 규칙(_unmarked_torch_users)이 같은 표시 형태를 받는다.

        규칙만 속성 형태(@_optional_deps.requires_torch)를 받으면, 그 파일을 고쳐도 훅이 --with-torch
        없이 돌아 방금 고친 torch 테스트가 빠진다(리뷰 지적).
        """
        test_body = "    def test_a(self):\n        load_torch()\n"
        cases = (
            "@requires_torch\nclass T(unittest.TestCase):\n" + test_body,
            "@_optional_deps.requires_torch\nclass T(unittest.TestCase):\n" + test_body,
            "class T(unittest.TestCase):\n    @_optional_deps.requires_torch\n" + test_body,
            "class T(unittest.TestCase):\n    @requires_torch\n" + test_body,
            "class T(unittest.TestCase):\n" + test_body,                               # 표시 없음
            '"""예:\n\n    @requires_torch\n"""\nclass T(unittest.TestCase):\n' + test_body,  # 문자열 속 언급
        )
        for source in cases:
            with self.subTest(source=source):
                accepted = _unmarked_torch_users(ast.parse(source)) == []
                self.assertEqual(run_tests.uses_torch_mark(source), accepted)
        self.assertEqual(sum(run_tests.uses_torch_mark(source) for source in cases), 4)


class RequiresTorchTests(unittest.TestCase):
    """헬퍼 동작 — 진짜 torch 는 올리지 않는다(설치 여부·import 는 가짜로 대체)."""

    def _fresh_class(self):
        return type("Case", (unittest.TestCase,), {"test_x": lambda self: None})

    def test_marks_class_and_method_for_the_runner(self):
        with mock.patch.object(_optional_deps, "torch_installed", return_value=True):
            cls = _optional_deps.requires_torch(self._fresh_class())
            method = _optional_deps.requires_torch(lambda self: None)
        self.assertEqual(getattr(cls, run_tests.REQUIRES_ATTR), frozenset({"torch"}))
        self.assertEqual(getattr(method, run_tests.REQUIRES_ATTR), frozenset({"torch"}))
        self.assertFalse(getattr(cls, "__unittest_skip__", False))
        self.assertEqual(run_tests.required_deps(cls("test_x")), frozenset({"torch"}))

    def test_skips_when_torch_is_not_installed_and_keeps_the_mark(self):
        with mock.patch.object(_optional_deps, "torch_installed", return_value=False):
            cls = _optional_deps.requires_torch(self._fresh_class())
            Holder = type("Holder", (unittest.TestCase,), {
                "test_m": _optional_deps.requires_torch(lambda self: None)})
        self.assertTrue(cls.__unittest_skip__)
        self.assertIn("torch", cls.__unittest_skip_why__)
        # skip 래퍼(functools.wraps)로 바뀐 메서드에서도 표시를 읽어 quick 이 거른다.
        self.assertEqual(run_tests.required_deps(Holder("test_m")), frozenset({"torch"}))
        result = unittest.TestResult()
        Holder("test_m").run(result)
        self.assertEqual(len(result.skipped), 1)

    def test_subclasses_inherit_the_mark(self):
        with mock.patch.object(_optional_deps, "torch_installed", return_value=True):
            base = _optional_deps.requires_torch(self._fresh_class())
        child = type("Child", (base,), {"test_y": lambda self: None})
        self.assertEqual(run_tests.required_deps(child("test_y")), frozenset({"torch"}))

    def test_load_torch_skips_when_missing_and_imports_lazily_when_present(self):
        with mock.patch.object(_optional_deps, "torch_installed", return_value=False):
            with self.assertRaises(unittest.SkipTest):
                _optional_deps.load_torch()
        fake = types.ModuleType("torch")
        with mock.patch.object(_optional_deps, "torch_installed", return_value=True), \
                mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertIs(_optional_deps.load_torch(), fake)
            namespace = {"torch": None}
            self.assertIs(_optional_deps.bind_torch(namespace), fake)
            self.assertIs(namespace["torch"], fake)

    def test_broken_torch_install_is_an_error_not_a_skip(self):
        """설치돼 있는데 import 가 깨지면(DLL 등) skip 으로 숨기지 않는다."""
        with mock.patch.object(_optional_deps, "torch_installed", return_value=True), \
                mock.patch.dict(sys.modules, {"torch": None}):  # None = import 가 ImportError
            with self.assertRaises(ImportError) as caught:
                _optional_deps.load_torch()
        self.assertNotIsInstance(caught.exception, unittest.SkipTest)

    def test_installed_check_does_not_import_torch(self):
        _optional_deps.torch_installed.cache_clear()
        self.addCleanup(_optional_deps.torch_installed.cache_clear)
        with mock.patch.object(_optional_deps.importlib.util, "find_spec", return_value=None) as find:
            self.assertFalse(_optional_deps.torch_installed())
        find.assert_called_once_with("torch")
        with mock.patch.object(_optional_deps.importlib.util, "find_spec", side_effect=ValueError):
            _optional_deps.torch_installed.cache_clear()
            self.assertFalse(_optional_deps.torch_installed())


if __name__ == "__main__":
    unittest.main()
