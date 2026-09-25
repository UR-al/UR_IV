"""sam-extra(forge_sam3_extension) 소스를 AST 로만 읽는 스캐너 — import·torch·Qt·Forge 없이.

계약 레지스트리(`core.sam_extra_contract`)와 비교할 '확장이 실제로 드러내는 것'을 뽑는다:
always-on 스크립트 제목과 ui() 반환 순서, ui() 컴포넌트의 라벨·기본값·범위·step·선택지,
ARG_NAMES/ARG_DEFAULTS, API 경로에서 위치로 읽는 인덱스, Forge 옵션 키, FastAPI 라우트, XYZ 축 라벨,
SAM3 process() 가 state 에서 읽는 키, Sam3Args 필드·기본값·어노테이션 범위, 모듈 파일 집합,
preload 명령줄 플래그, 함수 이름(Gradio 이름 엔드포인트가 함수 이름을 그대로 쓴다).
확장 코드를 실행하지 않으므로 GPU·모델 로드와 무관하다.

상수는 끝까지 따라간다: 리터럴, 모듈 상수, ``from X import Y as Z``(패키지 ``__init__`` 의 지연 import
포함), ``모듈별칭.상수``, f-string·문자열 덧셈, ``CONST[1:]``·``CONST[0]``, 함수 안 지역 대입,
``for a, b in ((..), ..)`` 루프 변수. 따라가지 못한 것은 `ExtensionSource.unresolved` 에 파일:줄로 남긴다 —
계약 테스트가 이를 분류되지 않은 새 항목으로 보고한다(조용히 빠지지 않게). ui() 컴포넌트 필드는 예외로
풀린 것만 담고, 픽스처에는 있는데 못 읽은 칸은 계약 테스트가 레지스트리 ``UI_UNREAD`` 와 맞춰 본다.
"""
from __future__ import annotations

import ast
import itertools
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

EXTENSION_DIR_ENV = "AISTUDIO_FORGE_EXTENSION_DIR"
REQUIRE_EXTENSION_ENV = "AISTUDIO_REQUIRE_FORGE_EXT"

_ROUTE_METHOD_CALLS = {"get": "GET", "put": "PUT", "post": "POST", "delete": "DELETE", "patch": "PATCH"}
_AXIS_CALLS = {"AxisOption", "AxisOptionImg2Img", "AxisOptionTxt2Img"}
_MAX_DEPTH = 24

# script-info 인자 항목과 같은 필드. ui() 컴포넌트에서 AST 로 읽어 픽스처와 비교한다.
UI_FIELDS = ("label", "value", "minimum", "maximum", "step", "choices")
# 값을 갖는 Gradio 컴포넌트 → 위치 인자 순서(Gradio 3/4 공통 앞부분, 나머지는 키워드 전용).
_UI_COMPONENT_ARGS = {
    "Checkbox": ("value",),
    "Slider": ("minimum", "maximum", "value"),
    "Radio": ("choices", "value"),
    "Dropdown": ("choices", "value"),
    "CheckboxGroup": ("choices", "value"),
    "Textbox": ("value",),
    "Number": ("value",),
    "ColorPicker": ("value",),
    "InputAccordion": ("value",),   # Forge modules.ui_components — 체크박스처럼 값(bool)을 준다
}
# 값이 없는 레이아웃 블록. ``with gr.Accordion(...) as x`` 폴백 뒤에 x 를 다시 대입하는 패턴에서 뺀다.
_UI_CONTAINERS = {"Accordion", "Row", "Column", "Group", "Tab", "Tabs", "TabItem", "Box", "Blocks",
                  "FormRow", "FormColumn", "FormGroup"}
_CONTAINER = object()
# pydantic 제약 타입 → (하한, 상한). gt=0 인 정수는 하한 1 로 적는다(_NUMERIC_BOUNDS 와 같은 표기).
_PYDANTIC_NUMERIC_TYPES = {
    "NonNegativeInt": (0, None), "NonNegativeFloat": (0.0, None), "PositiveInt": (1, None),
    "PositiveFloat": (0.0, None), "NonPositiveInt": (None, 0), "NonPositiveFloat": (None, 0.0),
    "NegativeInt": (None, -1), "NegativeFloat": (None, 0.0),
}


# ── 설치 위치 ───────────────────────────────────────────────────────────────
def is_extension_dir(path) -> bool:
    """sam-extra 작업 사본처럼 보이는 폴더인가(scripts/ 와 sam3ext/ 가 둘 다 있다)."""
    try:
        base = Path(path)
        return (base / "scripts").is_dir() and (base / "sam3ext").is_dir()
    except OSError:
        return False


def extension_dir_candidates(environ: Mapping[str, str] | None = None) -> list[Path]:
    """찾는 순서: 환경 변수 → 앱이 쓰는 Forge 확장 폴더(backend_runtime.json) → 알려진 Forge 설치.

    폴더 이름은 `core.sam3_assets.SAM3_EXTENSION_DIR_NAMES`(forge_sam3_extension, sam-extra) 둘 다 본다.
    환경 변수는 확장 루트나 그 아래 scripts 폴더 어느 쪽이어도 된다(예전 가드와 같은 규칙).
    """
    env = os.environ if environ is None else environ
    ordered: list[Path] = []
    configured = configured_extension_dir(env)
    if configured is not None:
        ordered.append(configured)
    from core.sam3_assets import SAM3_EXTENSION_DIR_NAMES, forge_extension_roots
    for root in forge_extension_roots(env):
        ordered.extend(Path(root) / name for name in SAM3_EXTENSION_DIR_NAMES)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in ordered:
        key = os.path.normcase(os.path.abspath(str(path)))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def configured_extension_dir(environ: Mapping[str, str] | None = None) -> Path | None:
    """환경 변수로 지정한 확장 루트(scripts 폴더를 줬으면 그 부모). 지정하지 않았으면 None."""
    env = os.environ if environ is None else environ
    configured = str(env.get(EXTENSION_DIR_ENV, "") or "").strip()
    if not configured:
        return None
    path = Path(configured)
    return path.parent if path.name == "scripts" else path


def find_installed_extension(environ: Mapping[str, str] | None = None) -> Path | None:
    """설치된 sam-extra 확장 루트, 없으면 None.

    환경 변수로 지정했으면 그 경로만 본다 — 오타가 난 경로가 조용히 다른 설치로 바뀌어 엉뚱한 사본을
    검사하지 않게 한다(확장 폴더가 아니면 None → 가드 테스트는 skip, 강제 모드면 실패).
    """
    configured = configured_extension_dir(environ)
    if configured is not None:
        return configured if is_extension_dir(configured) else None
    return next((path for path in extension_dir_candidates(environ) if is_extension_dir(path)), None)


# ── 결과 모양 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScriptInfo:
    """always-on 스크립트 하나(제목 = alwayson_scripts 키)."""

    title: str
    file: str                          # 확장 루트 기준 posix 경로
    cls: str
    # ui() 가 돌려주는 변수 이름 순서 — 두 모드(txt2img·img2img)가 같을 때만. 리스트가 아니거나(동적)
    # 모드마다 다르면 None. 모드별 계약 비교는 ui_returns 로 한다.
    ui_return: tuple | None
    arg_names: tuple | None = None     # 모듈 상수 ARG_NAMES
    arg_defaults: dict | None = None   # 모듈 상수 ARG_DEFAULTS
    api_reads: frozenset = field(default_factory=frozenset)  # _arg(N, …) / args[N] 로 읽는 인덱스
    # ui_return 순서대로 gr.* 컴포넌트의 {필드: 값}(UI_FIELDS 중 정적으로 풀린 것만). ui_return 이 None 이면 None.
    ui_components: tuple | None = None
    # 모드별 {False: txt2img, True: img2img} — ui() 의 모든 return 갈래를 is_img2img 로 풀어 읽는다(_ui_returns).
    ui_returns: dict | None = None
    ui_components_by_mode: dict | None = None


# ── 소스 해석 ────────────────────────────────────────────────────────────────
class _Unresolved(Exception):
    pass


class ExtensionSource:
    """확장 작업 사본 하나를 AST 로 읽는다. 파일마다 한 번만 파싱한다."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.unresolved: list[str] = []
        self._trees: dict[Path, ast.Module | None] = {}
        self._parents: dict[Path, dict[ast.AST, ast.AST]] = {}
        self._bindings: dict[Path, dict[str, ast.AST]] = {}
        self._imports: dict[Path, dict[str, tuple[str, str | None]]] = {}

    # 파일 ─────────────────────────────────────────────────────────────────
    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def script_files(self) -> list[Path]:
        return sorted((self.root / "scripts").glob("*.py"))

    def python_files(self) -> list[Path]:
        files = list(self.script_files())
        files += sorted(p for p in (self.root / "sam3ext").rglob("*.py") if "__pycache__" not in p.parts)
        return files

    def tree(self, path: Path) -> ast.Module | None:
        path = Path(path)
        if path not in self._trees:
            try:
                with open(path, encoding="utf-8") as fh:
                    self._trees[path] = ast.parse(fh.read(), filename=str(path))
            except (OSError, SyntaxError, ValueError) as exc:
                self.unresolved.append(f"{self._safe_rel(path)}: 파싱 실패 ({type(exc).__name__}: {exc})")
                self._trees[path] = None
        return self._trees[path]

    def _safe_rel(self, path: Path) -> str:
        try:
            return self.rel(path)
        except ValueError:
            return str(path)

    def parents(self, path: Path) -> dict[ast.AST, ast.AST]:
        if path not in self._parents:
            mapping: dict[ast.AST, ast.AST] = {}
            tree = self.tree(path)
            if tree is not None:
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        mapping[child] = node
            self._parents[path] = mapping
        return self._parents[path]

    # 모듈 이름 ─────────────────────────────────────────────────────────────
    def module_file(self, dotted: str) -> Path | None:
        if not dotted:
            return None
        base = self.root.joinpath(*dotted.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def _package_of(self, path: Path) -> str:
        parts = list(Path(self._safe_rel(path)).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            return ".".join(parts[:-1])
        return ".".join(parts[:-1])

    def _absolute_module(self, path: Path, module: str | None, level: int) -> str:
        if not level:
            return module or ""
        package = self._package_of(path).split(".") if self._package_of(path) else []
        if level > 1:
            package = package[: len(package) - (level - 1)]
        return ".".join([*package, *([module] if module else [])])

    # 이름 바인딩 ───────────────────────────────────────────────────────────
    def bindings(self, path: Path) -> dict[str, ast.AST]:
        """모듈 수준 대입(if/try/with 안 포함, 함수·클래스 본문 제외). 마지막 대입이 이긴다."""
        if path not in self._bindings:
            found: dict[str, ast.AST] = {}
            tree = self.tree(path)

            def visit(stmts: Iterable[ast.stmt]) -> None:
                for stmt in stmts:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                found[target.id] = stmt.value
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
                        found[stmt.target.id] = stmt.value
                    elif isinstance(stmt, (ast.If, ast.With, ast.For, ast.While)):
                        visit(stmt.body)
                        visit(getattr(stmt, "orelse", []))
                    elif isinstance(stmt, ast.Try):
                        visit(stmt.body)
                        for handler in stmt.handlers:
                            visit(handler.body)
                        visit(stmt.orelse)
                        visit(stmt.finalbody)

            if tree is not None:
                visit(tree.body)
            self._bindings[path] = found
        return self._bindings[path]

    def imports(self, path: Path) -> dict[str, tuple[str, str | None]]:
        """별칭 → (절대 모듈, 가져온 이름 또는 None=모듈 자체). 함수 안 지연 import 도 포함(모듈 수준 우선)."""
        if path not in self._imports:
            found: dict[str, tuple[str, str | None]] = {}
            tree = self.tree(path)
            if tree is not None:
                top = set(map(id, tree.body))
                nodes = sorted(
                    (n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
                    key=lambda n: (id(n) not in top, getattr(n, "lineno", 0)),
                )
                for node in nodes:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.asname:
                                found.setdefault(alias.asname, (alias.name, None))
                            else:
                                head = alias.name.split(".")[0]
                                found.setdefault(head, (head, None))
                    else:
                        module = self._absolute_module(path, node.module, node.level)
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            found.setdefault(alias.asname or alias.name, (module, alias.name))
            self._imports[path] = found
        return self._imports[path]

    # 값 풀기 ───────────────────────────────────────────────────────────────
    def values(self, path: Path, node: ast.AST, *, scope: ast.AST | None = None,
               env: Mapping[str, ast.AST] | None = None, depth: int = 0) -> list:
        """노드가 가질 수 있는 파이썬 값 목록(루프 변수면 여러 개). 못 풀면 빈 목록."""
        try:
            return self._values(Path(path), node, scope, dict(env or {}), depth)
        except (_Unresolved, RecursionError):
            return []

    def strings(self, path: Path, node: ast.AST, **kwargs) -> list[str]:
        return [v for v in self.values(path, node, **kwargs) if isinstance(v, str)]

    def single(self, path: Path, node: ast.AST, **kwargs):
        found = self.values(path, node, **kwargs)
        if len(found) != 1:
            raise _Unresolved(ast.dump(node)[:80])
        return found[0]

    def _values(self, path: Path, node: ast.AST, scope, env, depth) -> list:
        if depth > _MAX_DEPTH:
            raise _Unresolved("depth")
        nxt = depth + 1
        if isinstance(node, ast.Constant):
            return [node.value]
        if isinstance(node, ast.JoinedStr):
            parts: list[list[str]] = []
            for part in node.values:
                if isinstance(part, ast.Constant):
                    parts.append([str(part.value)])
                elif isinstance(part, ast.FormattedValue) and part.format_spec is None:
                    sub = [str(v) for v in self._values(path, part.value, scope, env, nxt)]
                    if not sub:
                        raise _Unresolved("f-string")
                    parts.append(sub)
                else:
                    raise _Unresolved("f-string spec")
            return ["".join(combo) for combo in itertools.product(*parts)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._values(path, node.left, scope, env, nxt)
            right = self._values(path, node.right, scope, env, nxt)
            return [a + b for a, b in itertools.product(left, right)]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = self._values(path, node.operand, scope, env, nxt)
            return [-v if isinstance(node.op, ast.USub) else v for v in inner if isinstance(v, (int, float))]
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            items = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    items.extend(self._one(path, elt.value, scope, env, nxt))
                else:
                    items.append(self._one(path, elt, scope, env, nxt))
            return [tuple(items) if not isinstance(node, ast.Set) else frozenset(items)]
        if isinstance(node, ast.Dict):
            out: dict = {}
            for key, value in zip(node.keys, node.values):
                if key is None:  # **other
                    other = self._one(path, value, scope, env, nxt)
                    if not isinstance(other, dict):
                        raise _Unresolved("dict unpack")
                    out.update(other)
                else:
                    out[self._one(path, key, scope, env, nxt)] = self._one(path, value, scope, env, nxt)
            return [out]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "keys" \
                and not node.args:
            base = self._one(path, node.func.value, scope, env, nxt)
            if isinstance(base, dict):
                return [tuple(base.keys())]
            raise _Unresolved("keys()")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("list", "tuple") and len(node.args) == 1:
            return [tuple(self._one(path, node.args[0], scope, env, nxt))]
        if isinstance(node, ast.Subscript):   # CONST[1:], CONST[0], TABLE["key"]
            base = self._one(path, node.value, scope, env, nxt)
            if isinstance(node.slice, ast.Slice):
                bounds = [None if part is None else self._one(path, part, scope, env, nxt)
                          for part in (node.slice.lower, node.slice.upper, node.slice.step)]
                if isinstance(base, (tuple, str)) and all(b is None or type(b) is int for b in bounds):
                    return [base[slice(*bounds)]]
                raise _Unresolved("slice")
            key = self._one(path, node.slice, scope, env, nxt)
            try:
                return [base[key]]
            except (KeyError, IndexError, TypeError) as exc:
                raise _Unresolved("subscript") from exc
        if isinstance(node, ast.Name):
            return self._name(path, node.id, scope, env, nxt)
        if isinstance(node, ast.Attribute):
            module = self._module_of_expr(path, node.value)
            if module is not None:
                target = self.module_file(module)
                if target is not None:
                    return self._name(target, node.attr, None, {}, nxt)
            raise _Unresolved("attribute")
        raise _Unresolved(type(node).__name__)

    def _one(self, path, node, scope, env, depth):
        found = self._values(path, node, scope, env, depth)
        if len(found) != 1:
            raise _Unresolved("ambiguous")
        return found[0]

    def _name(self, path: Path, name: str, scope, env, depth) -> list:
        if name in env:
            return self._values(path, env[name], scope, {}, depth)
        if scope is not None:
            local = [
                n.value for n in _walk_scope(scope)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
            ]
            if local:
                out: list = []
                for value in local:
                    out.extend(self._values(path, value, scope, {}, depth))
                return out
            loop = self._loop_values(path, name, scope, depth)
            if loop is not None:
                return loop
        bindings = self.bindings(path)
        if name in bindings:
            return self._values(path, bindings[name], None, {}, depth)
        imported = self.imports(path).get(name)
        if imported is not None:
            module, attr = imported
            if attr is None:
                raise _Unresolved("module object")
            submodule = self.module_file(f"{module}.{attr}")
            if submodule is not None:
                raise _Unresolved("submodule object")
            target = self.module_file(module)
            if target is not None:
                return self._name(target, attr, None, {}, depth)
        raise _Unresolved(f"name {name}")

    def _loop_values(self, path: Path, name: str, scope, depth) -> list | None:
        for node in _walk_scope(scope):
            if not isinstance(node, ast.For):
                continue
            for env in self._for_envs(path, node, depth):
                if name in env:
                    out: list = []
                    for env2 in self._for_envs(path, node, depth):
                        out.extend(self._values(path, env2[name], None, {}, depth + 1))
                    return out
        return None

    def _for_envs(self, path: Path, loop: ast.For, depth) -> list[dict[str, ast.AST]]:
        """``for a, b in ((x, y), (z, w))`` → [{a: x, b: y}, {a: z, b: w}]. 풀 수 없으면 []."""
        iterable = loop.iter
        if isinstance(iterable, ast.Name):
            iterable = self.bindings(path).get(iterable.id, iterable)
        if not isinstance(iterable, (ast.Tuple, ast.List)):
            return []
        envs: list[dict[str, ast.AST]] = []
        for item in iterable.elts:
            if isinstance(loop.target, ast.Name):
                envs.append({loop.target.id: item})
            elif isinstance(loop.target, ast.Tuple) and isinstance(item, (ast.Tuple, ast.List)) \
                    and len(item.elts) == len(loop.target.elts):
                env = {}
                for tgt, val in zip(loop.target.elts, item.elts):
                    if isinstance(tgt, ast.Name):
                        env[tgt.id] = val
                envs.append(env)
            else:
                return []
        return envs

    def _module_of_expr(self, path: Path, node: ast.AST) -> str | None:
        """``alias`` 또는 ``a.b.c`` 가 확장 안의 모듈을 가리키면 그 절대 이름."""
        if isinstance(node, ast.Name):
            imported = self.imports(path).get(node.id)
            if imported is None:
                return None
            module, attr = imported
            dotted = module if attr is None else f"{module}.{attr}"
            return dotted if self.module_file(dotted) is not None else None
        if isinstance(node, ast.Attribute):
            base = self._module_of_expr(path, node.value)
            if base is None:
                return None
            dotted = f"{base}.{node.attr}"
            return dotted if self.module_file(dotted) is not None else None
        return None

    def _call_envs(self, path: Path, call: ast.Call) -> list[dict[str, ast.AST]]:
        """호출을 감싼 가장 안쪽 for 루프(리터럴 반복)의 반복별 바인딩. 없으면 [{}]."""
        parents = self.parents(path)
        node = parents.get(call)
        while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            if isinstance(node, ast.For):
                envs = self._for_envs(path, node, 0)
                if envs:
                    return envs
            node = parents.get(node)
        return [{}]

    def enclosing_function(self, path: Path, node: ast.AST) -> ast.AST | None:
        parents = self.parents(path)
        cur = parents.get(node)
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cur = parents.get(cur)
        return cur

    def _resolve_arg(self, path: Path, call: ast.Call, index: int, keyword: str, what: str,
                     *, required_prefix: str = "", report: bool = True) -> list[tuple[str, dict[str, ast.AST]]]:
        """호출의 index 번째 인자(또는 keyword)를 문자열로 — 루프면 반복마다.

        실패는 unresolved 에 남긴다. ``report=False`` 는 라우트인지 확실하지 않은 호출(``app.get(x)``)용 —
        풀지 못하면 라우트가 아니라고 보고 조용히 넘어간다.
        """
        node = call.args[index] if len(call.args) > index else next(
            (kw.value for kw in call.keywords if kw.arg == keyword), None)
        where = f"{self.rel(path)}:{call.lineno}"
        if node is None:
            if report:
                self.unresolved.append(f"{where}: {what} 인자 없음")
            return []
        scope = self.enclosing_function(path, call)
        out: list[tuple[str, dict[str, ast.AST]]] = []
        for env in self._call_envs(path, call):
            found = self.strings(path, node, scope=scope, env=env)
            if not found:
                if report:
                    self.unresolved.append(f"{where}: {what} 을(를) 정적으로 풀지 못함 ({ast.unparse(node)[:60]})")
                return []
            out.extend((value, env) for value in found if value.startswith(required_prefix))
        return out

    # ── 추출 ─────────────────────────────────────────────────────────────
    def version(self) -> str | None:
        path = self.root / "sam3ext" / "__version__.py"
        if not path.is_file():
            return None
        node = self.bindings(path).get("__version__")
        found = self.strings(path, node) if node is not None else []
        return found[0] if found else None

    def module_constant(self, rel_file: str, name: str):
        """확장 파일의 모듈 상수 값(풀 수 없으면 KeyError)."""
        path = self.root / rel_file
        node = self.bindings(path).get(name)
        if node is None:
            raise KeyError(f"{rel_file}:{name} 없음")
        found = self.values(path, node)
        if len(found) != 1:
            raise KeyError(f"{rel_file}:{name} 을(를) 정적으로 풀지 못함")
        return found[0]

    def scripts(self) -> dict[str, ScriptInfo]:
        """AlwaysVisible 스크립트 제목 → ScriptInfo."""
        found: dict[str, ScriptInfo] = {}
        for path in self.script_files():
            tree = self.tree(path)
            if tree is None:
                continue
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                methods = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                show = methods.get("show")
                if show is None or not any(
                    isinstance(r, ast.Return) and _is_always_visible(r.value) for r in _walk_scope(show)
                ):
                    continue
                title_fn = methods.get("title")
                titles = []
                if title_fn is not None:
                    for ret in (r for r in _walk_scope(title_fn) if isinstance(r, ast.Return) and r.value):
                        titles.extend(self.strings(path, ret.value, scope=title_fn))
                if len(set(titles)) != 1:
                    self.unresolved.append(f"{self.rel(path)}:{cls.lineno}: {cls.name}.title() 을 풀지 못함")
                    continue
                arg_names = self._module_value(path, "ARG_NAMES")
                arg_defaults = self._module_value(path, "ARG_DEFAULTS")
                ui_fn = methods.get("ui")
                ui_returns, conflicts = _ui_returns(ui_fn)
                if conflicts:
                    modes = "·".join("img2img" if mode else "txt2img" for mode in sorted(conflicts))
                    lines = sorted({line for found_lines in conflicts.values() for line in found_lines})
                    self.unresolved.append(
                        f"{self.rel(path)}:{ui_fn.lineno}: {cls.name}.ui() 가 {modes} 에서 조건에 따라 다른 목록을 "
                        f"돌려준다(return 줄 {', '.join(map(str, lines))}) — 위치 인자 계약을 하나로 정할 수 없다")
                by_mode = {mode: None if names is None else tuple(
                    self.ui_component(path, ui_fn, name) for name in names) for mode, names in ui_returns.items()}
                common = ui_returns[False] == ui_returns[True]
                found[titles[0]] = ScriptInfo(
                    title=titles[0],
                    file=self.rel(path),
                    cls=cls.name,
                    ui_return=ui_returns[False] if common else None,
                    arg_names=tuple(arg_names) if isinstance(arg_names, tuple) else None,
                    arg_defaults=arg_defaults if isinstance(arg_defaults, dict) else None,
                    api_reads=frozenset(_positional_reads(cls)),
                    ui_components=by_mode[False] if common else None,
                    ui_returns=ui_returns,
                    ui_components_by_mode=by_mode,
                )
        return found

    # ui() 컴포넌트 ───────────────────────────────────────────────────────
    def ui_component(self, path: Path, ui_fn: ast.AST, name: str) -> dict:
        """ui() 안에서 ``name`` 에 묶이는 gr.* 컴포넌트의 {필드: 값}. 정적으로 확실한 필드만 담는다.

        ``name = gr.Slider(...)`` 와 ``with InputAccordion(...) as name`` (문맥이 지역 변수여도) 를 따라간다.
        후보가 여럿이면(구버전 Gradio 폴백 분기) 모든 후보가 같은 값을 주는 필드만 남긴다. 모르는 호출이
        하나라도 섞이면 아무 필드도 믿지 않는다(빈 dict) — 비교에서 조용히 틀린 값을 쓰지 않게.
        실행 시점 값(파일 목록 등)은 풀리지 않아 빠진다. 키워드가 없는 필드는 Gradio 기본값을 흉내 내지 않고 뺀다.
        """
        merged: dict | None = None
        for node in self._component_nodes(ui_fn, name):
            fields = self._component_fields(path, ui_fn, node)
            if fields is _CONTAINER:
                continue
            if fields is None:
                return {}
            merged = dict(fields) if merged is None else {
                key: value for key, value in merged.items() if key in fields and fields[key] == value}
        return merged or {}

    def _component_nodes(self, ui_fn: ast.AST, name: str) -> list[ast.AST]:
        nodes: list[ast.AST] = []
        for node in _walk_scope(ui_fn):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                nodes.extend(self._local_producers(ui_fn, node.value, 0))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == name:
                        nodes.extend(self._local_producers(ui_fn, item.context_expr, 0))
        return nodes

    def _local_producers(self, ui_fn: ast.AST, expr: ast.AST, depth: int) -> list[ast.AST]:
        """``context = X(...)`` 처럼 지역 변수를 거치면 그 대입값들로 바꾼다(if/else 분기 모두)."""
        if not isinstance(expr, ast.Name) or depth > 4:
            return [expr]
        values = [n.value for n in _walk_scope(ui_fn)
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == expr.id for t in n.targets)]
        if not values:
            return [expr]
        out: list[ast.AST] = []
        for value in values:
            out.extend(self._local_producers(ui_fn, value, depth + 1))
        return out

    def _component_fields(self, path: Path, ui_fn: ast.AST, node: ast.AST):
        """컴포넌트 호출의 필드. 레이아웃 블록이면 _CONTAINER, 모르는 것(함수 호출·별칭 등)이면 None."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        kind = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if kind in _UI_CONTAINERS:
            return _CONTAINER
        params = _UI_COMPONENT_ARGS.get(kind)
        if params is None or any(isinstance(a, ast.Starred) for a in node.args) \
                or any(kw.arg is None for kw in node.keywords):
            return None
        bound = dict(zip(params, node.args))
        bound.update((kw.arg, kw.value) for kw in node.keywords if kw.arg in UI_FIELDS)
        fields: dict = {}
        for key in UI_FIELDS:
            if key in bound:
                found = self.values(path, bound[key], scope=ui_fn)
                if len(found) == 1:
                    fields[key] = found[0]
        return fields

    def _module_value(self, path: Path, name: str):
        node = self.bindings(path).get(name)
        if node is None:
            return None
        found = self.values(path, node)
        return found[0] if len(found) == 1 else None

    def option_keys(self) -> dict[str, str]:
        """``shared.opts.add_option(KEY, …)`` 키 → 파일."""
        found: dict[str, str] = {}
        for path, call in self._calls(lambda name: name == "add_option"):
            for key, _env in self._resolve_arg(path, call, 0, "key", "옵션 키"):
                found[key] = self.rel(path)
        return found

    def routes(self) -> dict[str, str]:
        """FastAPI 라우트 ``"METHOD /path"`` → 파일 (add_api_route / add_route / app.get 등)."""
        found: dict[str, str] = {}
        for path, call in self._calls(lambda name: name in ("add_api_route", "add_route", "api_route")):
            scope = self.enclosing_function(path, call)
            for route, env in self._resolve_arg(path, call, 0, "path", "라우트 경로", required_prefix="/"):
                methods_node = next((kw.value for kw in call.keywords if kw.arg == "methods"), None)
                methods = ("GET",)
                if methods_node is not None:
                    resolved = self.values(path, methods_node, scope=scope, env=env)
                    if len(resolved) != 1 or not isinstance(resolved[0], tuple):
                        self.unresolved.append(f"{self.rel(path)}:{call.lineno}: 라우트 methods 를 풀지 못함")
                        continue
                    methods = resolved[0]
                for method in methods:
                    found[f"{str(method).upper()} {route}"] = self.rel(path)
        for path, call in self._calls(lambda name: name in _ROUTE_METHOD_CALLS, attr_only=True):
            base = call.func.value
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            if not (base_name.endswith("app") or base_name in ("router", "api")):
                continue
            for route, _env in self._resolve_arg(path, call, 0, "path", "라우트 경로", required_prefix="/",
                                                 report=False):
                found[f"{_ROUTE_METHOD_CALLS[call.func.attr]} {route}"] = self.rel(path)
        return found

    def axis_labels(self) -> dict[str, str]:
        """XYZ ``AxisOption(label, …)`` 라벨 → 파일."""
        found: dict[str, str] = {}
        for path, call in self._calls(lambda name: name in _AXIS_CALLS):
            for label, _env in self._resolve_arg(path, call, 0, "label", "XYZ 축 라벨"):
                found[label] = self.rel(path)
        return found

    def cli_flags(self) -> dict[str, str]:
        """preload.py 의 ``parser.add_argument("--flag")`` → 파일."""
        path = self.root / "preload.py"
        found: dict[str, str] = {}
        if not path.is_file() or self.tree(path) is None:
            return found
        for call in (n for n in ast.walk(self.tree(path)) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument":
                for flag, _env in self._resolve_arg(path, call, 0, "", "명령줄 플래그", required_prefix="--"):
                    found[flag] = self.rel(path)
        return found

    def function_names(self) -> dict[str, set[str]]:
        """함수 이름 → 정의된 파일들 (Gradio 이름 엔드포인트 = 함수 이름)."""
        found: dict[str, set[str]] = {}
        for path in self.python_files():
            tree = self.tree(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.setdefault(node.name, set()).add(self.rel(path))
        return found

    def module_units(self) -> set[str]:
        """분류 단위: scripts/*.py, sam3ext/*.py, sam3ext/<하위 패키지>/, javascript/*.js, 루트 *.py."""
        units: set[str] = set()
        units.update(f"scripts/{p.name}" for p in self.script_files())
        sam3ext = self.root / "sam3ext"
        if sam3ext.is_dir():
            for child in sam3ext.iterdir():
                if child.is_file() and child.suffix == ".py":
                    units.add(f"sam3ext/{child.name}")
                elif child.is_dir() and child.name != "__pycache__" and any(child.rglob("*.py")):
                    units.add(f"sam3ext/{child.name}/")
        units.update(f"javascript/{p.name}" for p in sorted((self.root / "javascript").glob("*.js")))
        units.update(p.name for p in self.root.glob("*.py"))
        return units

    def dict_keys_assigned(self, rel_file: str, function: str, variable: str) -> frozenset:
        """``function`` 안에서 ``variable = {...}`` 로 만드는 dict 의 문자열 키."""
        path = self.root / rel_file
        tree = self.tree(path)
        keys: set[str] = set()
        if tree is None:
            return frozenset()
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function):
            for node in _walk_scope(fn):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(
                        isinstance(t, ast.Name) and t.id == variable for t in node.targets):
                    keys.update(k.value for k in node.value.keys
                                if isinstance(k, ast.Constant) and isinstance(k.value, str))
        return frozenset(keys)

    def state_reads(self, rel_file: str, function: str, state_var: str, readers: Iterable[str]) -> frozenset:
        """``function`` 안에서 ``state_var.get("k")`` 또는 ``reader("k")`` 로 읽는 키."""
        path = self.root / rel_file
        tree = self.tree(path)
        reader_names = set(readers)
        keys: set[str] = set()
        if tree is None:
            return frozenset()
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function):
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                first = node.args[0]
                if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "get" \
                        and isinstance(func.value, ast.Name) and func.value.id == state_var:
                    keys.add(first.value)
                elif isinstance(func, ast.Name) and func.id in reader_names:
                    keys.add(first.value)
        return frozenset(keys)

    def sam3args(self) -> dict:
        """sam3ext/args.py 의 Sam3Args: 필드 순서, Literal 선택지, 기본값, 어노테이션 범위, _NUMERIC_BOUNDS.

        ``constraints`` 는 pydantic 이 실제로 거부하는 범위(``confloat(ge=, le=)``, ``NonNegativeInt`` 등)
        ``{필드: (하한|None, 상한|None)}`` 이다. ``_NUMERIC_BOUNDS`` 는 그 앞에서 값을 잘라 주는 표라서 둘이
        어긋나면(한쪽만 고침) 잘린 값이 다시 거부돼 SAM3 가 꺼진다. 기본값을 풀지 못한 필드는 unresolved.
        """
        path = self.root / "sam3ext" / "args.py"
        tree = self.tree(path)
        fields: list[str] = []
        literals: dict[str, tuple] = {}
        defaults: dict = {}
        constraints: dict[str, tuple] = {}
        if tree is not None:
            cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Sam3Args"), None)
            for stmt in (cls.body if cls else []):
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    fields.append(name)
                    ann = stmt.annotation
                    if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) \
                            and ann.value.id == "Literal":
                        inner = ann.slice
                        elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
                        literals[name] = tuple(e.value for e in elts if isinstance(e, ast.Constant))
                    constraint = self._numeric_constraint(path, ann)
                    if constraint is not None:
                        constraints[name] = constraint
                    if stmt.value is not None:
                        found = self.values(path, stmt.value)
                        if len(found) == 1:
                            defaults[name] = found[0]
                        else:
                            self.unresolved.append(f"{self.rel(path)}:{stmt.lineno}: Sam3Args.{name} 기본값을 "
                                                   "정적으로 풀지 못함")
        bounds = self._module_value(path, "_NUMERIC_BOUNDS") if path.is_file() else None
        return {"fields": tuple(fields), "literals": literals, "defaults": defaults,
                "constraints": constraints, "numeric_bounds": bounds if isinstance(bounds, dict) else {}}

    def _numeric_constraint(self, path: Path, ann: ast.AST) -> tuple | None:
        """pydantic 숫자 제약 어노테이션 → (하한, 상한). 제약이 아니면 None."""
        if isinstance(ann, ast.Name) and ann.id in _PYDANTIC_NUMERIC_TYPES:
            return _PYDANTIC_NUMERIC_TYPES[ann.id]
        if isinstance(ann, ast.Call) and isinstance(ann.func, ast.Name) and ann.func.id in ("confloat", "conint"):
            low = high = None
            for kw in ann.keywords:
                if kw.arg in ("ge", "gt", "le", "lt"):
                    found = self.values(path, kw.value)
                    if len(found) != 1:
                        self.unresolved.append(f"{self.rel(path)}:{ann.lineno}: {ann.func.id}({kw.arg}=…) 를 "
                                               "정적으로 풀지 못함")
                        return None
                    if kw.arg in ("ge", "gt"):
                        low = found[0]
                    else:
                        high = found[0]
            return (low, high)
        return None

    def changelog_versions(self) -> list[tuple[str, str]]:
        """CHANGELOG.md 의 ``## vX.Y.Z — 제목`` 목록(위에서부터 = 최신 순)."""
        path = self.root / "CHANGELOG.md"
        out: list[tuple[str, str]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    match = re.match(r"^##\s+\[?v?(\d+\.\d+\.\d+)\]?\s*(.*)$", line.rstrip())
                    if match:
                        out.append((match.group(1), match.group(2).strip(" —-")))
        except OSError:
            pass
        return out

    def _calls(self, want, *, attr_only: bool = False):
        for path in self.python_files():
            tree = self.tree(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and want(func.attr):
                    yield path, node
                elif not attr_only and isinstance(func, ast.Name) and want(func.id):
                    yield path, node


# ── 작은 AST 도우미 ─────────────────────────────────────────────────────────
def _walk_scope(fn: ast.AST):
    """함수 본문을 걷되 안쪽 함수·클래스·람다 본문으로는 들어가지 않는다."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _is_always_visible(node: ast.AST | None) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "AlwaysVisible") or (
        isinstance(node, ast.Name) and node.id == "AlwaysVisible")


UI_MODES = (False, True)   # ui(self, is_img2img) 의 두 호출 — txt2img, img2img


def _ui_returns(fn: ast.AST | None) -> tuple[dict, dict]:
    """ui() 가 모드별로 돌려주는 이름 순서 {False: txt2img, True: img2img} 와 모호한 모드 {모드: return 줄들}.

    마지막 return 하나만 보면 ``if is_img2img: return [...]`` 같은 조기 반환의 인자 증감을 놓친다. 그래서
    모드마다 닿을 수 있는 return 을 모두 모은다. ``is_img2img`` 조건(``not``·``is``/``==``·and/or)은 모드별로
    풀고, 모르는 조건은 두 갈래를 다 본다. 닿는 return 이 모두 같은 리스트/튜플이면 그 이름들, 리스트가 아닌
    것이 섞이면 None(동적). 같은 모드에서 서로 다른 리스트가 닿으면 None 이고 두 번째 dict 에 남긴다 —
    호출부가 unresolved 로 올려 계약 검사가 조용히 통과하지 않게 한다.
    """
    names: dict = {mode: None for mode in UI_MODES}
    conflicts: dict = {}
    if fn is None:
        return names, conflicts
    params = [*fn.args.posonlyargs, *fn.args.args]
    param = params[1].arg if len(params) > 1 else None
    for mode in UI_MODES:
        returns, _ends = _reachable_returns(fn.body, param, mode)
        if not returns or any(not isinstance(r.value, (ast.List, ast.Tuple)) for r in returns):
            continue
        found = {_list_names(r.value) for r in returns}
        if len(found) == 1:
            names[mode] = found.pop()
        else:
            conflicts[mode] = sorted({r.lineno for r in returns})
    return names, conflicts


def _list_names(node: ast.List | ast.Tuple) -> tuple:
    return tuple(el.id if isinstance(el, ast.Name) else ast.unparse(el) for el in node.elts)


def _reachable_returns(stmts: list, param: str | None, mode: bool) -> tuple[list, bool]:
    """문장 목록에서 ``param == mode`` 일 때 닿을 수 있는 return 들과, 목록이 반드시 return 으로 끝나는지.

    if 는 모드로 풀리면 한 갈래만, 아니면 두 갈래 다. with 는 본문을 그대로 따른다. 반복문·try·match 는
    모든 본문의 return 을 모으되 반드시 끝난다고 보지 않는다(넉넉하게 — 모호하면 unresolved 로 드러난다).
    안쪽 함수·클래스의 return 은 ui() 의 반환이 아니다.
    """
    found: list = []
    for stmt in stmts:
        if isinstance(stmt, ast.Return):
            found.append(stmt)
            return found, True
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If):
            truth = _mode_truth(stmt.test, param, mode)
            branches = [stmt.body] if truth is True else [stmt.orelse] if truth is False else [stmt.body, stmt.orelse]
            results = [_reachable_returns(branch, param, mode) for branch in branches]
            for returns, _ends in results:
                found.extend(returns)
            if all(ends for _returns, ends in results):
                return found, True
            continue
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            returns, ends = _reachable_returns(stmt.body, param, mode)
            found.extend(returns)
            if ends:
                return found, True
            continue
        for block in _child_blocks(stmt):
            found.extend(_reachable_returns(block, param, mode)[0])
    return found, False


def _child_blocks(stmt: ast.stmt) -> list:
    """반복문·try·match 같은 복합문의 안쪽 문장 목록들."""
    blocks = [getattr(stmt, name) for name in ("body", "orelse", "finalbody") if isinstance(getattr(stmt, name, None), list)]
    blocks += [handler.body for handler in getattr(stmt, "handlers", ())]
    blocks += [case.body for case in getattr(stmt, "cases", ())]
    return blocks


def _mode_truth(test: ast.AST, param: str | None, mode: bool) -> bool | None:
    """조건식이 ``param == mode`` 일 때 참/거짓인지. 모드만으로 정해지지 않으면 None."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.Name) and param is not None and test.id == param:
        return mode
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _mode_truth(test.operand, param, mode)
        return None if inner is None else not inner
    if isinstance(test, ast.BoolOp):
        values = [_mode_truth(v, param, mode) for v in test.values]
        decisive = isinstance(test.op, ast.Or)   # or 는 참 하나로, and 는 거짓 하나로 정해진다
        if decisive in values:
            return decisive
        return None if None in values else not decisive
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and param is not None:
        left, right = test.left, test.comparators[0]
        if isinstance(right, ast.Name) and right.id == param:
            left, right = right, left
        if isinstance(left, ast.Name) and left.id == param and isinstance(right, ast.Constant) \
                and isinstance(right.value, bool):
            op = test.ops[0]
            if isinstance(op, (ast.Is, ast.Eq)):
                return mode == right.value
            if isinstance(op, (ast.IsNot, ast.NotEq)):
                return mode != right.value
    return None


def _positional_reads(cls: ast.ClassDef) -> set[int]:
    """스크립트 클래스 안에서 ``_arg(N, …)`` 또는 ``args[N]`` 으로 읽는 위치 인덱스."""
    reads: set[int] = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_arg" \
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            reads.add(node.args[0].value)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "args" \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            reads.add(node.slice.value)
    return reads


def format_index_set(indices: Iterable[int]) -> str:
    """{0,1,2,4} → '0-2,4' (레지스트리에 짧게 적기용)."""
    items = sorted(set(indices))
    parts: list[str] = []
    start = prev = None
    for index in items:
        if start is None:
            start = prev = index
        elif index == prev + 1:
            prev = index
        else:
            parts.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = index
    if start is not None:
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def parse_index_set(text: str) -> frozenset:
    """'0-2,4' → {0,1,2,4}. 빈 문자열은 빈 집합."""
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in str(text or "").split(","))):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return frozenset(out)
