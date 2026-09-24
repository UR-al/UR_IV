#!/usr/bin/env python
"""테스트 실행기 — 추가 설치 없이 표준 라이브러리(unittest)로 tests/ 전체 실행.

사용법 (Windows):
    venv\\Scripts\\python.exe run_tests.py             # 전체 — /verify · /ship 이 쓰는 경로
    venv\\Scripts\\python.exe run_tests.py --quick     # SLOW_MODULES 제외 — PostToolUse 훅용
    venv\\Scripts\\python.exe run_tests.py --quick --include tests.test_backend_runtime
                                                      # quick + 지정한 느린 모듈만 추가
    venv\\Scripts\\python.exe run_tests.py --quick --with-torch
                                                      # quick + torch 표시 테스트(TORCH_TEST_SOURCES 참고)
    venv\\Scripts\\python.exe run_tests.py --durations 15
                                                      # 모듈별 소요 시간 상위 15개 + discovery 시간
시간 수치는 아래 SLOW_MODULES 주석(실측 날짜 포함)에만 적는다 — 여기 복붙하면 또 낡는다.

--quick 은 두 가지를 거른다: 느린 통합 테스트 모듈(SLOW_MODULES)과, 무거운 선택 의존성 torch 가
필요하다고 표시된 테스트(tests/_optional_deps.requires_torch — 모듈이 아니라 클래스·메서드 단위).
torch 테스트 파일은 torch 를 지연 import 하므로, 걸러지면 quick 은 torch 를 아예 올리지 않는다.
그래도 torch 가 올라오면(표시 없는 테스트가 부르는 소스의 전이 import 등) **실패로 끝낸다** —
discovery(모든 모드)나 --quick 실행 중 torch 를 처음 import 한 위치를 stderr 에 적는다(ImportTripwire).

성공하면 종료코드 0, 테스트가 하나라도 실패하거나 위 torch 검사에 걸리면 1.
(pytest 불필요 — 비개발자도 한 줄로 검증 가능)

⚠ venv 의존성(pandas/PIL/PyQt6/requests/torch)이 필요하다. 시스템 python(PATH 의 첫 python 은
   의존성 없는 3.10)으로 실행하면 main() 이 저장소 venv 인터프리터로 **자동 재실행**한다.
   재실행을 끄려면 URIV_TESTS_REEXEC=1 을 설정한다(venv 가 없으면 현재 인터프리터로 그대로 돈다).
"""
import argparse
import ast
import importlib.abc
import os
import re
import subprocess
import sys
import time
import traceback
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))

# venv 재실행이 이미 일어났음을 자식에게 알리는 환경변수(무한 재실행 방지 겸 수동 opt-out).
REEXEC_ENV = "URIV_TESTS_REEXEC"

# 느린 통합 테스트 → 그 테스트가 **직접** 검증하는 소스(손으로 적은 집합).
# --quick 에서 제외하고, 훅은 편집한 파일이 여기 소스와 맞는 모듈만 quick 에 더해 돌린다.
# 전이 import 그래프를 자동 계산하지 않는 이유: 공용 모듈(config/path_safety 등)이 대거 딸려 와
# 거의 모든 편집이 느린 모듈을 끌어들이게 된다. 전이 의존성(launch_args/storage_paths/
# forge_modules 등)은 각자 전용 quick 테스트가 있다.
#
# 넣는 기준: 소켓·하위 프로세스·스레드 대기처럼 **기다리느라** 1초 안팎 이상 들고, 직접 검증하는
# 소스를 손으로 적을 수 있는 모듈. 새 후보는 `--durations` 로 재고 근거와 함께 여기 추가한다.
#
# 실측 2026-09-24 (`run_tests.py --durations`, 병렬 작업 부하가 있어 ±30% 흔들림):
#   전체 3,440개 45.5~47.6초(벽시계 47.1초) · discovery 0.8초 · 모듈별: generation_api 6.4초,
#   backend_runtime 4.9~5.8초, webui_cancel 2.7초, model_cache 1.6초, generation_api_remote_e2e
#   1.1초, webui_modules_sync 1.0초, web_tabs 0.6초(예전 1.2초 — 하위 프로세스라 흔들려 그대로 둔다).
#   이 7개(약 18초)를 빼면 quick 은 3,188개 24.1~25.5초(벽시계 25.3~26.8초) — torch 표시 테스트도
#   빠져 torch 를 아예 올리지 않는다(올리면 실패로 끝난다 — ImportTripwire). 같은 트리에서
#   --with-torch 를 켜면(= torch 를 최상위에서 import 하던 예전 quick 에 해당) 3,250개 27.4초(벽시계
#   28.9초): torch import 약 1초 + torch 텐서 테스트. 나머지는 테스트 모듈 250여 개의 긴 꼬리다.
#   quick 상위(1.5~1.8초) test_tag_database_assets · test_ui_prefs_single_source · test_cv_io 는
#   기다리는 게 아니라 저장소·태그 자산 전체를 훑는 정적 가드다 — 직접 소스가 '저장소 전체'라 여기
#   넣으면 훅이 영영 안 돌린다. 그다음은 모두 0.9초 미만이다.
SLOW_MODULES = {
    # 실제 HTTP 소켓 서버 + 스레드 대기
    "tests.test_generation_api": frozenset({
        "core/generation_api.py",
        "core/resource_coordinator.py",
        "backends/base.py",
    }),
    # 원격 클라이언트 ↔ 서버 E2E (소켓 + 스레드)
    "tests.test_generation_api_remote_e2e": frozenset({
        "core/generation_api.py",
        "core/resource_coordinator.py",
    }),
    # subprocess 스폰 · 프로세스 가드 · 포트 예약
    "tests.test_backend_runtime": frozenset({
        "core/backend_runtime.py",
        "core/generation_api_port.py",
        "core/process_guard.py",
    }),
    # 가짜 Forge HTTP 서버 + 폴링 스레드 대기(단일 테스트 최대 1.6초)
    "tests.test_webui_cancel": frozenset({
        "core/webui_cancel.py",
        "backends/webui_backend.py",
    }),
    # offscreen QApplication 하위 프로세스 4개
    "tests.test_web_tabs": frozenset({
        "tabs/browser_tab.py",
        "tabs/backend_ui_tab.py",
    }),
    # 실제 reaper 스레드 대기 — 만료되지 않음을 보이는 대기(0.3·0.4초)는 매번 끝까지 기다린다
    "tests.test_model_cache": frozenset({
        "core/model_cache.py",
    }),
    # 진행률 폴링 스레드 대기(단일 테스트가 모듈 시간의 거의 전부)
    "tests.test_webui_modules_sync": frozenset({
        "backends/webui_backend.py",
    }),
}

# 위 느린 모듈이 커버하는 소스 전체(SLOW_MODULES 에서 파생 — 단일 진실 원천).
SLOW_MODULE_SOURCES = frozenset().union(*SLOW_MODULES.values())

# ── 무거운 선택 의존성(torch) 테스트 ──────────────────────────────────────────
# torch 는 import 만으로 느린 모듈 하나만큼 든다(실측은 위 SLOW_MODULES 주석). 그래서 torch 가
# 필요한 테스트는 모듈이 아니라 **클래스·메서드 단위로 표시**하고(tests/_optional_deps.requires_torch →
# 아래 속성), 테스트 파일은 torch 를 지연 import 한다. --quick 은 표시된 테스트를 걸러 torch 를
# 아예 올리지 않고, 훅은 아래 소스나 표시가 든 테스트 파일을 고쳤을 때만 --with-torch 로 되살린다.
REQUIRES_ATTR = "__uriv_requires__"   # 클래스·테스트 함수에 붙는 frozenset({"torch"})
TORCH = "torch"
# 이번 실행이 거른 선택 의존성(쉼표 구분, --quick 이면 "torch")을 테스트에 알리는 환경변수 — main() 이
# 실행 동안만 둔다. 모듈 속성이 아닌 이유: `python run_tests.py` 의 main 은 __main__ 모듈이고, 테스트가
# `import run_tests` 로 받는 건 별개의 모듈 객체라 속성을 바꿔도 테스트에서 안 보인다.
SKIP_DEPS_ENV = "URIV_TESTS_SKIP_DEPS"


def active_skip_deps(environ=None) -> frozenset:
    """지금 도는 run_tests 실행이 거른 선택 의존성(--quick 이면 {"torch"}). run_tests 밖이면 빈 집합.

    tests/test_optional_deps 가 읽는다 — quick 에선 torch 가 올라와 있는 것 자체가 회귀다.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(SKIP_DEPS_ENV) or ""
    return frozenset(part.strip() for part in raw.split(",") if part.strip())
# torch 테스트가 **직접** 검증하는 소스 — '/' 로 끝나면 폴더 접두, 아니면 파일 하나.
TORCH_TEST_SOURCES = (
    "comfy_custom_nodes/",            # sam3·mask_ops·guidance·generation·h3 캐시·anima_lora 노드
    "core/creator_workflows.py",      # h3 캐시 노드 텐서 테스트가 build() 결과로 돈다
)
# 표시를 담당하는 테스트 헬퍼(여기를 고치면 torch 테스트 전체를 다시 본다).
TORCH_MARK_HELPER = "tests/_optional_deps.py"
# 표시 데코레이터 이름 — `@requires_torch` 와 속성 형태 `@_optional_deps.requires_torch` 둘 다 표시다.
TORCH_MARK = "requires_torch"
# 파싱이 안 되는(편집 중 깨진) 테스트 파일에서만 쓰는 줄 모양 판단 — 위 두 형태를 같이 받는다.
_TORCH_MARK_USE = re.compile(r"^[ \t]*@(?:[A-Za-z_][\w.]*\.)?requires_torch\b", re.M)
# 저장소 안이지만 우리 코드가 아닌 곳(서드파티 패키지) — torch 를 처음 import 한 위치를 보고할 때 건너뛴다.
_VENV_DIR = "venv"


def decorator_names(node) -> set:
    """AST 클래스·함수 노드의 데코레이터 이름 — `@x` · `@m.x` · `@x(...)` 모두 마지막 이름 x.

    torch 표시를 알아보는 **단일 규칙**이다: 훅(torch_tests_for)과 규칙 테스트
    (tests/test_optional_deps)가 같이 쓴다 — 둘이 갈라지면 한쪽만 표시로 보는 형태가 생긴다.
    """
    names = set()
    for deco in getattr(node, "decorator_list", ()):
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def uses_torch_mark(source: str) -> bool:
    """소스에 torch 표시(TORCH_MARK 데코레이터)가 붙은 클래스·함수가 있나.

    AST 로 본다 — 문자열·docstring 안에서 이름만 언급하는 파일(규칙·러너 테스트)은 아니다.
    편집 중이라 파싱이 안 되면 줄 모양(_TORCH_MARK_USE)으로 판단한다(표시가 보이면 torch 쪽으로).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _TORCH_MARK_USE.search(source) is not None
    return any(TORCH_MARK in decorator_names(node) for node in ast.walk(tree)
               if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))


def _normalize_rel(rel_path) -> str:
    """훅이 넘기는 경로를 저장소 상대경로(슬래시)로 — Windows 구분자·'./' 접두 제거."""
    rel = str(rel_path or "").replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def torch_tests_for(rel_path: str, root: str = ROOT) -> bool:
    """저장소 상대경로 파일을 고쳤을 때 quick 에 torch 표시 테스트를 더해야 하나.

    - TORCH_TEST_SOURCES 의 파일이거나 그 폴더 아래
    - 표시 헬퍼(tests/_optional_deps.py) 자체
    - `@requires_torch`(속성 형태 `@_optional_deps.requires_torch` 포함)로 표시한 테스트가 든
      tests/*.py — 편집한 테스트 파일을 uses_torch_mark 로 읽어 판단한다(목록을 따로 두지 않는다.
      이름만 언급하는 파일은 아니다 — 괜히 torch 를 올리지 않게)
    """
    rel = _normalize_rel(rel_path)
    if not rel:
        return False
    for source in TORCH_TEST_SOURCES:
        if rel == source or (source.endswith("/") and rel.startswith(source)):
            return True
    if rel == TORCH_MARK_HELPER:
        return True
    if rel.startswith("tests/") and rel.endswith(".py"):
        try:
            with open(os.path.join(root, *rel.split("/")), encoding="utf-8", errors="replace") as f:
                return uses_torch_mark(f.read())
        except OSError:
            return False
    return False


def slow_modules_for(rel_path: str) -> frozenset:
    """저장소 상대경로(슬래시) 파일을 고쳤을 때 quick 에 더해 돌려야 할 느린 모듈.

    - 느린 모듈이 직접 검증하는 소스면 그 모듈들
    - 느린 테스트 모듈 파일 자체면 그 모듈
    - 그 밖에는 빈 집합(quick 만으로 충분)
    """
    rel = _normalize_rel(rel_path)
    hits = {module for module, sources in SLOW_MODULES.items() if rel in sources}
    if rel.endswith(".py"):
        module = rel[:-3].replace("/", ".")  # tests/test_x.py -> tests.test_x
        if module in SLOW_MODULES:
            hits.add(module)
    return frozenset(hits)


def venv_python(root: str = ROOT):
    """저장소 venv 인터프리터 경로. 없으면 None (훅과 재실행이 공유한다)."""
    candidates = (
        os.path.join(root, "venv", "Scripts", "python.exe"),  # Windows
        os.path.join(root, "venv", "bin", "python"),          # POSIX
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def reexec_command(argv, *, root: str = ROOT, prefix=None, environ=None):
    """venv 밖 인터프리터로 실행됐으면 venv 로 재실행할 명령을, 아니면 None.

    None 인 경우: 이미 재실행된 자식(REEXEC_ENV 설정) / 이미 venv 안 / venv 가 없음.
    """
    environ = os.environ if environ is None else environ
    if environ.get(REEXEC_ENV):
        return None
    prefix = sys.prefix if prefix is None else prefix
    if _same_path(prefix, os.path.join(root, "venv")):
        return None
    python = venv_python(root)
    if python is None:
        return None
    return [python, os.path.join(root, "run_tests.py"), *list(argv)]


class DurationTable:
    """모듈별 소요 시간 누적기(순수 로직 — 테스트 가능하게 결과 객체와 분리)."""

    def __init__(self):
        self.seconds = {}
        self.counts = {}

    def add(self, module: str, seconds: float, *, count: int = 0) -> None:
        self.seconds[module] = self.seconds.get(module, 0.0) + max(0.0, float(seconds))
        self.counts[module] = self.counts.get(module, 0) + int(count)

    def total(self) -> float:
        return sum(self.seconds.values())

    def top(self, limit: int):
        """(모듈, 초, 테스트 수) 를 느린 순으로. 동률은 이름순(출력 안정)."""
        rows = sorted(self.seconds.items(), key=lambda kv: (-kv[1], kv[0]))
        if limit and limit > 0:
            rows = rows[:limit]
        return [(module, secs, self.counts.get(module, 0)) for module, secs in rows]

    def format(self, title: str, limit: int, *, show_counts: bool = True) -> str:
        total = self.total()
        lines = [f"[durations] {title}: 합계 {total:.2f}s"]
        for module, secs, count in self.top(limit):
            share = (secs / total * 100.0) if total > 0 else 0.0
            suffix = f" ({count} tests)" if show_counts else ""
            lines.append(f"  {secs:8.2f}s {share:5.1f}%  {module}{suffix}")
        return "\n".join(lines)


def _test_module_name(test) -> str:
    return type(test).__module__ or "?"


class _TimingResult(unittest.TextTestResult):
    """테스트 경계 사이의 벽시계 시간을 모듈별로 합산한다.

    startTest 까지의 간격(직전 클래스 tearDownClass + 이번 setUpModule/setUpClass)은
    이번 테스트 모듈에 귀속한다 — fixture 비용도 모듈 비용으로 보이게.
    """

    table = None  # main() 이 주입

    def startTestRun(self):
        super().startTestRun()
        self._mark = time.perf_counter()
        self._last_module = None

    def stopTest(self, test):
        now = time.perf_counter()
        module = _test_module_name(test)
        self.table.add(module, now - self._mark, count=1)
        self._mark = now
        self._last_module = module
        super().stopTest(test)

    def stopTestRun(self):
        # 마지막 tearDownClass/tearDownModule 은 마지막 모듈 몫
        if self._last_module is not None:
            self.table.add(self._last_module, time.perf_counter() - self._mark)
        super().stopTestRun()


class _TimedLoader(unittest.TestLoader):
    """discover 가 테스트 모듈을 import 하는 시간을 모듈별로 잰다(torch 등 무거운 import 추적)."""

    table = None  # main() 이 주입

    def _get_module_from_name(self, name):  # unittest.TestLoader 내부 훅(3.x 공통)
        started = time.perf_counter()
        try:
            return super()._get_module_from_name(name)
        finally:
            if self.table is not None:
                self.table.add(name, time.perf_counter() - started)


def _iter_tests(suite):
    """중첩된 TestSuite 를 평탄화해 개별 TestCase 를 내놓는다."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def required_deps(test) -> frozenset:
    """테스트가 요구하는 무거운 선택 의존성 — 클래스 표시와 테스트 메서드 표시의 합.

    표시는 tests/_optional_deps.requires_torch 가 REQUIRES_ATTR 로 붙인다(skip 래퍼도
    functools.wraps 로 속성을 옮기므로 skip 된 메서드에서도 읽힌다).
    """
    deps = set(getattr(type(test), REQUIRES_ATTR, None) or ())
    method_name = getattr(test, "_testMethodName", None)
    if method_name:
        method = getattr(type(test), method_name, None)
        deps.update(getattr(method, REQUIRES_ATTR, None) or ())
    return frozenset(deps)


def _without_slow(suite, keep=(), skip_deps=()):
    """SLOW_MODULES 소속 테스트를 걷어낸다(keep 에 든 느린 모듈은 남긴다).

    skip_deps 가 주어지면 그 의존성(예: {"torch"})이 필요하다고 표시된 테스트도 걷어낸다.
    import 실패로 생긴 _FailedTest 는 모듈이 unittest.loader 라 그대로 남는다 —
    느린 모듈이 깨져도 quick 모드에서 조용히 묻히지 않는다.
    (discover 한 **뒤에** 거른다 — 모듈 import 실패가 quick 에서도 보이게 하려는 의도다.)
    """
    keep = frozenset(keep)
    skip_deps = frozenset(skip_deps)
    kept = unittest.TestSuite()
    for test in _iter_tests(suite):
        module = type(test).__module__
        if module in SLOW_MODULES and module not in keep:
            continue
        if skip_deps and required_deps(test) & skip_deps:
            continue
        kept.addTest(test)
    return kept


def _importing(frame) -> bool:
    """이 find_spec 호출이 **실제 import**(import 문·importlib.import_module·__import__)에서 왔나.

    finder 를 부르는 건 import 시스템의 _find_spec 이다. 그 _find_spec 을 부른 쪽이
    _find_and_load(_unlocked) 면 실제 import, importlib.util.find_spec(설치 여부만 확인 — 모듈을
    실행하지 않는다)이면 아니다. **바로 위 호출자만** 본다 — 스택 더 바깥의 _find_and_load 는 다른
    모듈의 import 일 수 있다(예: discovery 가 테스트 모듈을 import 하는 중에 클래스 본문의
    @requires_torch 가 torch_installed → find_spec 을 부른다).
    """
    while frame is not None and frame.f_code.co_name == "_find_spec":
        frame = frame.f_back
    if frame is None:
        return False
    code = frame.f_code
    return code.co_name in ("_find_and_load_unlocked", "_find_and_load") and "importlib" in code.co_filename


class ImportTripwire(importlib.abc.MetaPathFinder):
    """걸러야 할 무거운 모듈(torch)을 누가 **처음** import 하는지 잡는다.

    관찰만 한다(find_spec 이 None 을 돌려 import 는 평소대로 진행). 실제 import(_importing)일 때만
    위치를 적고(설치 확인용 importlib.util.find_spec 은 무시) 첫 기록을 지킨다. 기록이 있거나
    sys.modules 에 올라와 있으면 import 된 것으로 본다:
    - 기록만 있다: mock.patch.dict(sys.modules) 가 끝나며 올라온 torch 를 도로 뺐다(느려진 건 그대로다).
    - sys.modules 에만 있다: finder 를 거치지 않고 누가 직접 넣었다(위치는 알 수 없다).
    main() 은 discovery(모든 모드)와 --quick 실행에서 이것을 **실패**로 친다 — 경고만 쓰면 훅과
    /verify 가 종료코드만 보므로 아무도 못 본다(감사 #168 후속).
    """

    def __init__(self, name: str, root: str = ROOT):
        self.name = name
        self.root = os.path.normcase(os.path.abspath(root))
        self.where = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name and self.where is None and _importing(sys._getframe(1)):
            self.where = self._repo_frames(traceback.extract_stack()[:-1])
        return None

    def imported(self) -> bool:
        return self.where is not None or self.name in sys.modules

    def _repo_frames(self, stack) -> str:
        """저장소의 우리 코드(run_tests 자신·venv 제외) 프레임을 바깥→안 순서로 짧게. 없으면 '(저장소 밖)'.

        venv(서드파티)를 빼는 이유: 소스가 서드파티를 거쳐 torch 를 올리면 마지막 몇 프레임이 전부
        site-packages 라 정작 고칠 우리 코드 위치가 잘려 나간다.
        """
        this_file = os.path.normcase(os.path.abspath(__file__))
        venv = os.path.join(self.root, _VENV_DIR) + os.sep
        frames = []
        for frame in stack:
            if frame.filename.startswith("<"):  # <frozen importlib._bootstrap> 등
                continue
            filename = os.path.normcase(os.path.abspath(frame.filename))
            if filename == this_file or not filename.startswith(self.root + os.sep) or filename.startswith(venv):
                continue
            rel = os.path.relpath(filename, self.root).replace(os.sep, "/")
            frames.append(f"{rel}:{frame.lineno}")
        return " -> ".join(frames[-4:]) if frames else "(저장소 밖)"

    def report(self, stage: str = "--quick 실행"):
        """import 됐으면 실패 문구, 아니면 None. stage 는 어느 단계에서 봤는지(discovery·--quick 실행)."""
        if not self.imported():
            return None
        return (f"[run_tests] 실패: {stage} 중에 {self.name} 가 import 됐다 — --quick(PostToolUse 훅)이 "
                f".py 편집마다 그만큼 느려진다. 처음 import 한 곳: "
                f"{self.where or '(알 수 없음 — finder 를 거치지 않고 sys.modules 에 들어왔다)'}\n"
                f"  테스트라면 tests/_optional_deps.requires_torch 로 표시하고, torch 는 테스트 안(load_torch)"
                f"이나 표시된 클래스의 setUpClass(bind_torch)에서 지연 import 한다. 소스라면 torch 를 쓰는 "
                f"함수 안에서 import 한다(최상위 import 는 그 소스를 import 하는 모든 테스트가 torch 를 올린다).\n")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="run_tests.py",
        description="tests/ 를 표준 unittest 로 실행한다.")
    parser.add_argument("--quick", action="store_true",
                        help="SLOW_MODULES(느린 통합 테스트)와 torch 표시 테스트를 제외한다")
    parser.add_argument("--include", action="append", default=[], metavar="MODULE",
                        help="--quick 이어도 이 느린 모듈은 돌린다(반복 가능)")
    parser.add_argument("--with-torch", action="store_true",
                        help="--quick 이어도 torch 표시 테스트(tests/_optional_deps.requires_torch)를 돌린다")
    parser.add_argument("--durations", type=int, default=None, metavar="N",
                        help="모듈별 실행 시간 상위 N 개와 discovery(import) 시간을 출력한다")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.include) - set(SLOW_MODULES))
    if unknown:
        parser.error(f"--include 는 SLOW_MODULES 만 받는다: {', '.join(unknown)}")
    return args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    command = reexec_command(argv)
    if command is not None:
        # 시스템 python 은 pandas/PyQt6/torch 가 없어 가짜 ImportError 수십 개를 낸다.
        sys.stderr.write(
            f"[run_tests] venv 밖 인터프리터({sys.executable}) -> {command[0]} 로 재실행합니다.\n")
        sys.stderr.flush()
        return subprocess.call(command, cwd=ROOT, env={**os.environ, REEXEC_ENV: "1"})

    args = _parse_args(argv)
    durations = args.durations is not None
    skip_deps = frozenset({TORCH}) if args.quick and not args.with_torch else frozenset()

    sys.path.insert(0, ROOT)
    # torch 를 누가 처음 올리는지 지켜본다(관찰만). discovery 는 모든 모드에서 — 테스트 모듈 import 가
    # torch 를 올리면 quick 도 매번 올린다(/verify 의 전체 실행도 여기서 잡는다). 실행 중에는 torch
    # 테스트를 거르는 --quick 에서만 — 전체·--with-torch 에선 표시된 테스트가 torch 를 올리는 게 정상이다.
    tripwire = None
    if TORCH not in sys.modules:
        tripwire = ImportTripwire(TORCH)
        sys.meta_path.insert(0, tripwire)
    previous_skip_env = os.environ.get(SKIP_DEPS_ENV)
    os.environ[SKIP_DEPS_ENV] = ",".join(sorted(skip_deps))
    failures = []
    try:
        import_table = DurationTable() if durations else None
        loader = _TimedLoader() if durations else unittest.TestLoader()
        if durations:
            loader.table = import_table
        discover_started = time.perf_counter()
        suite = loader.discover(os.path.join(ROOT, "tests"),
                                pattern="test_*.py", top_level_dir=ROOT)
        discover_seconds = time.perf_counter() - discover_started
        if tripwire is not None:
            leak = tripwire.report("discovery(테스트 모듈 import)")
            if leak or TORCH not in skip_deps:
                if leak:
                    failures.append(leak)
                if tripwire in sys.meta_path:
                    sys.meta_path.remove(tripwire)
                tripwire = None  # 이미 실패했거나, 이 실행은 torch 테스트를 돌린다
        if args.quick:
            suite = _without_slow(suite, keep=args.include, skip_deps=skip_deps)

        # quick 은 훅이 호출 → 실패 시 stderr 꼬리가 잘리지 않게 출력을 짧게(dots).
        runner = unittest.TextTestRunner(verbosity=1 if args.quick else 2)
        run_table = None
        if durations:
            run_table = DurationTable()
            _TimingResult.table = run_table
            runner.resultclass = _TimingResult
        result = runner.run(suite)
    finally:
        if tripwire is not None and tripwire in sys.meta_path:
            sys.meta_path.remove(tripwire)
        if previous_skip_env is None:
            os.environ.pop(SKIP_DEPS_ENV, None)
        else:
            os.environ[SKIP_DEPS_ENV] = previous_skip_env

    if tripwire is not None:  # --quick 에서 discovery 는 통과 → 실행 중에 올렸나
        leak = tripwire.report("--quick 실행")
        if leak:
            failures.append(leak)
    if durations:
        limit = max(0, args.durations)
        report = [
            f"[durations] discovery {discover_seconds:.2f}s "
            f"(테스트 모듈 import: 무거운 import 는 처음 import 한 모듈에 몰린다)",
            import_table.format("import 상위", limit, show_counts=False),
            run_table.format("모듈별 실행 상위(fixture 포함)", limit),
        ]
        sys.stderr.write("\n" + "\n".join(report) + "\n")
    # 훅은 stderr 꼬리만 보여 준다 — 실패 사유를 맨 끝에 둔다.
    for failure in failures:
        sys.stderr.write("\n" + failure)
    return 0 if result.wasSuccessful() and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
