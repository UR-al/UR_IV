"""브리지 계약 정합성 테스트 — 프론트(Vue) ↔ 백엔드(PyQt) 이름 불일치 회귀 방지.

'브리지 계약 불일치 = 버그 주원인' (CLAUDE.md). 예전 실제 버그:
  - 프론트가 requestAction('X')/action('X')를 부르는데 Python에 핸들러(action=='X')가 없음
  - 프론트가 onBackendEvent('Y')를 듣는데 Python에 pyqtSignal Y가 없음
  - 페이로드 키 불일치 (조건식 target vs tags) 등
이 테스트는 소스를 정적 분석해 위 '이름' 불일치를 컴파일 타임처럼 잡는다.
(Qt 비의존 — 순수 텍스트/AST 파싱.)

검사 방향:
  - 정방향(FE → PY): 프론트가 부르는 액션·듣는 이벤트에 Python 핸들러·시그널이 있어야 한다.
  - 역방향(PY → FE): Python 에만 남은 액션·슬롯·시그널·웹 화이트리스트 이름은 사문이다.
    PYTHON_INTERNAL(Python 이 직접 쓰는 것) 또는 PENDING(정리 예정, 사유 필수)에 있어야 한다.
  - 이름 기반 emit: ``self._creator_emit("X", ...)`` · ``getattr(bridge, "X", None)`` 의 X 는
    실제 VueBridge 시그널이어야 한다(오타면 이벤트가 조용히 사라진다). X 가 리터럴이 아니면
    조건식·상수·한 번만 대입한 지역 변수·매개변수를 그대로 넘기는 래퍼까지 따라가고, 그래도 못 읽으면
    테스트가 실패한다(액션 표와 같은 정책 — 조용히 검사에서 빠지지 않게).

Python 액션 이름은 AST 로 뽑는다 — 디스패처 함수(`_handle_*action*` 등) 안에서 Name ``action`` 과의
``==``/``!=``/``in``/``not in`` 비교만 센다. 조건식 규칙·런타임 이벤트 함수의 지역변수 ``action``
(add/remove/start/stop …)은 핸들러로 오인하지 않는다. ``action in _SOME_TABLE``/``action not in
handlers`` 처럼 모듈·클래스 상수나 함수 안 지역 표(한 번만 대입)를 가리키면 그 문자열(dict 면 키)을
따라가 읽는다 — 해석 못 하면 테스트가 실패한다(조용히 검사에서 빠지지 않게).
"""
import ast
import os
import posixpath
import re
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT = os.path.join(ROOT, 'frontend', 'src')
UI = os.path.join(ROOT, 'ui')

# 프론트가 부르지만 Python 핸들러가 없어도 되는 예외(동적/특수). 비어 있으면 전부 매칭 요구.
ACTION_ALLOWLIST: set = set()
# 프론트가 듣지만 pyqtSignal이 없어도 되는 예외.
EVENT_ALLOWLIST: set = set()

# 프론트가 안 쓰지만 Python 이 직접 쓰는 브리지 이름(정상). (종류, 이름) → 사유.
PYTHON_INTERNAL: dict = {
    ("slot", "_apply_adetailer_models_json"):
        "vue_bridge 가 adetailerModelsReady 를 이 슬롯에 connect 한다(Qt 스레드 경계 넘기기)",
    ("signal", "backendRuntimeEvent"):
        "generator_main 과 studio_qwebchannel(DesktopNativeHost)이 Python 에서 connect/emit 한다",
}

# Python 에만 남은 사문 — 다음 정리 단계(P13c)에서 지운다. (종류, 이름) → 사유.
# 정리하면 여기서도 지워라(test_reverse_pending_entries_are_not_stale 가 알려 준다).
# 새 사문을 여기에 더하지 말 것 — 지우거나 프론트에 배선한다.
PENDING: dict = {
}

# 액션 이름을 담지만 '보내는 곳'이 아닌 프론트 파일 — 역방향 검사의 호출 근거에서 뺀다.
_NON_CALLSITE_FRONTEND = {
    'frontend/src/utils/hostDialogs.ts':
        "Python core/web_action_policy.DESKTOP_DIALOG_ACTIONS 의 거울 목록(웹 모드 버튼 잠금용)",
}

# 액션 디스패처로 보는 함수: 이름이 _handle_*action* 이거나 아래 명시 목록(중첩 def 포함).
_HANDLER_NAME = re.compile(r"^_?handle_\w*action\w*$")
_EXTRA_HANDLERS = frozenset({"_model_download_dispatch", "_run_relight_action"})

_RQ = re.compile(r"requestAction\(\s*['\"]([a-zA-Z_]\w*)['\"]")
# requestAction(cond ? 'a' : 'b', …) — 조건식으로 고르는 두 이름을 모두 호출로 읽는다.
_RQ_TERNARY = re.compile(
    r"requestAction\(\s*[^'\"(),?]+\?\s*(['\"])([a-zA-Z_]\w*)\1\s*:\s*(['\"])([a-zA-Z_]\w*)\3")
# 뷰의 액션 래퍼 — action('X') · quickAction('X', …)(첫 인자가 ActionName). 'requestAction'(대문자 A)/
# 'foo.action'(메서드)·runAppUpdateAction 같은 다른 이름은 제외한다(브리지 액션이 아니다).
_AW = re.compile(r"(?<![A-Za-z_.$])(?:action|quickAction)\(\s*['\"]([a-zA-Z_]\w*)['\"]")
_EV = re.compile(r"onBackendEvent\(\s*['\"]([a-zA-Z_]\w*)['\"]")
# 프론트가 시그널에 직접 붙는 경우(widgetStore: backend.widgetValueChanged.connect(...))
_CONNECT = re.compile(r"\.([a-zA-Z_]\w*)\.connect\(")
_FE_LITERAL = re.compile(r"""(['"`])([A-Za-z_]\w*)\1""")
_TS_LITERAL = re.compile(r"['\"]([a-zA-Z_]\w*)['\"]")


def _read_tree(folder, exts):
    """{저장소 상대경로(슬래시): 텍스트}."""
    out = {}
    for dirpath, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.endswith(exts):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, encoding='utf-8') as f:
                        out[os.path.relpath(path, ROOT).replace(os.sep, '/')] = f.read()
                except OSError:
                    pass
    return out


def _frontend_actions(texts):
    names = set()
    for t in texts:
        names.update(_RQ.findall(t))
        names.update(_AW.findall(t))
        for match in _RQ_TERNARY.finditer(t):
            names.update((match.group(2), match.group(4)))
    return names


def _frontend_events(texts):
    names = set()
    for t in texts:
        names.update(_EV.findall(t))
    return names


# ── Python AST ────────────────────────────────────────────────────────────

_COLLECTION_CALLS = frozenset({'frozenset', 'set', 'tuple', 'list', 'dict', 'MappingProxyType'})


def _string_items(node, constants):
    """비교 대상 노드 → 문자열 집합(dict 는 키). 정적으로 해석 못 하면 None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [_string_items(e, constants) for e in node.elts]
        return None if any(i is None for i in items) else set().union(*items)
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):  # {**other}
            return None
        items = [_string_items(k, constants) for k in node.keys]
        return None if any(i is None for i in items) else set().union(*items)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _COLLECTION_CALLS and len(node.args) == 1 and not node.keywords):
        return _string_items(node.args[0], constants)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id in ('self', 'cls')):
        return constants.get(node.attr)
    return None


def _collect_constants(tree):
    """모듈/클래스 최상위의 `NAME = <문자열 컬렉션>` 을 {NAME: 문자열 집합} 으로."""
    constants = {}
    bodies = [tree.body] + [n.body for n in tree.body if isinstance(n, ast.ClassDef)]
    for body in bodies:
        for stmt in body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                target, value = stmt.target, stmt.value
            else:
                continue
            if isinstance(target, ast.Name):
                items = _string_items(value, constants)
                if items is not None:
                    constants[target.id] = items
    return constants


def _is_action_name(node):
    return isinstance(node, ast.Name) and node.id == 'action'


def _is_action_dispatcher(fn):
    return _HANDLER_NAME.match(fn.name) is not None or fn.name in _EXTRA_HANDLERS


def _local_constants(fn, constants):
    """디스패처 함수 안의 `handlers = {'x': self._x, ...}` 같은 지역 표도 따라간다.

    한 번만, 값을 아는 대입으로 묶인 이름만 받는다. 재대입·for/with/except·튜플 언패킹·매개변수로
    묶였거나 값을 해석 못 하는 지역 이름은 같은 이름의 모듈/클래스 상수를 **가린다**(표에서 지운다) —
    가려진 상수로 읽으면 엉뚱한 표의 이름을 핸들러로 착각한다. 가려진 비교는 unresolved 로 남는다.
    """
    local = dict(constants)
    for name in _bound_params(fn):
        local.pop(name, None)
    for name, values in _local_bindings(fn).items():
        items = None
        if len(values) == 1 and values[0] is not None:
            items = _string_items(values[0], local)
        if items is None:
            local.pop(name, None)
        else:
            local[name] = items
    return local


def _parse_sources(sources):
    """{상대경로: 텍스트} → {상대경로: AST}. 큰 파일이 많아 한 번만 파싱해 돌려 쓴다."""
    return {rel: ast.parse(text, filename=rel) for rel, text in sources.items()}


def _python_action_table(sources, trees=None):
    """디스패처 함수의 `action` 비교에서 액션 이름을 뽑는다.

    반환: ({액션: {"파일:함수", ...}}, [해석 못 한 비교 위치]).
    """
    names, unresolved = {}, []
    trees = trees if trees is not None else _parse_sources(sources)
    for rel, tree in trees.items():
        module_constants = _collect_constants(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not _is_action_dispatcher(fn):
                continue
            constants = _local_constants(fn, module_constants)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Compare):
                    continue
                operands = [node.left, *node.comparators]
                for index, op in enumerate(node.ops):
                    left, right = operands[index], operands[index + 1]
                    if isinstance(op, (ast.Eq, ast.NotEq)):
                        other = right if _is_action_name(left) else left if _is_action_name(right) else None
                    elif isinstance(op, (ast.In, ast.NotIn)):
                        other = right if _is_action_name(left) else None
                    else:
                        other = None
                    if other is None:
                        continue
                    values = _string_items(other, constants)
                    if values is None:
                        unresolved.append(f"{rel}:{node.lineno} ({fn.name})")
                        continue
                    for value in values:
                        names.setdefault(value, set()).add(f"{rel}:{fn.name}")
    return names, unresolved


def _bridge_members(text):
    """vue_bridge.py 의 VueBridge 클래스에서 (시그널, 슬롯, 전체 멤버 이름)."""
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VueBridge')
    signals, slots, members = set(), set(), set()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    members.add(target.id)
                    value = stmt.value
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                            and value.func.id == 'pyqtSignal'):
                        signals.add(target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.add(stmt.name)
            for deco in stmt.decorator_list:
                func = deco.func if isinstance(deco, ast.Call) else deco
                if isinstance(func, ast.Name) and func.id == 'pyqtSlot':
                    slots.add(stmt.name)
    return signals, slots, members


def _module_string_set(path, name):
    """모듈 최상위 상수(frozenset({...}) 등)를 import 없이 읽는다(Qt 비의존)."""
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    constants = _collect_constants(tree)
    if name not in constants:
        raise AssertionError(f"{os.path.basename(path)} 에서 {name} 상수를 정적으로 읽지 못했습니다")
    return constants[name]


_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _positional_params(fn):
    """위치 인자 이름(메서드면 self/cls 제외) — 헬퍼 호출의 인자 위치와 맞춘다."""
    params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    return params[1:] if params[:1] in (['self'], ['cls']) else params


def _bound_params(scope):
    """def/lambda 가 매개변수로 묶는 이름 전부(*args·**kwargs·키워드 전용 포함)."""
    args = scope.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    names.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
    return names


def _local_bindings(fn):
    """def 안에서 대입되는 이름 → [값 노드 | None].

    None 은 값을 정적으로 모르는 대입(for·with·+=·except·튜플 언패킹). 중첩 def 의 대입도 함께
    세므로 보수적이다 — 대입이 둘 이상이거나 None 이 섞이면 해석 불가로 친다.
    """
    bindings = {}

    def bind(target, value):
        if isinstance(target, ast.Name):
            bindings.setdefault(target.id, []).append(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                bind(elt, None)
        elif isinstance(target, ast.Starred):
            bind(target.value, None)

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.AugAssign):
            bind(node.target, None)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind(node.target, None)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(node.optional_vars, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bindings.setdefault(node.name, []).append(None)
    return bindings


def _scoped_calls(tree):
    """모든 호출 → (호출 노드, 둘러싼 def/lambda 체인 — 바깥→안).

    generator_main 의 긴 elif 사슬은 AST 가 매우 깊어 재귀 대신 스택으로 돈다.
    """
    stack, calls = [(tree, ())], []
    while stack:
        node, chain = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                calls.append((child, chain))
            inner = chain + (child,) if isinstance(child, (*_FUNCTION_SCOPES, ast.Lambda)) else chain
            stack.append((child, inner))
    return calls


def _call_name(call):
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def _emit_name_arg(call, call_name, helpers):
    """이름 emit 호출이면 (종류, 시그널 이름 인자 노드 | None=인자 없음·*args), 아니면 None."""
    if call_name == 'getattr':
        if len(call.args) >= 2 and 'bridge' in ast.unparse(call.args[0]):
            return 'getattr', call.args[1]
        return None
    if call_name not in helpers:
        return None
    index, param = helpers[call_name]
    if index is not None:
        if any(isinstance(a, ast.Starred) for a in call.args[:index + 1]):
            return call_name, None  # *args — 어느 위치가 이름인지 정적으로 모른다
        if len(call.args) > index:
            return call_name, call.args[index]
    for kw in call.keywords:
        if kw.arg == param:
            return call_name, kw.value
    return call_name, None  # 인자 누락(또는 **kwargs 로만 전달)


def _resolve_emit_name(node, chain, module_constants, bindings_of, seen=frozenset()):
    """시그널 이름 인자 → ('values', {이름}) | ('wrapper', (def, 위치|None, 인자명)) | ('unresolved', None).

    리터럴·조건식 리터럴·모듈/클래스 상수·한 번만 대입된 지역 변수(바깥 def 의 클로저 포함)를 따라간다.
    감싼 def 의 매개변수를 그대로 넘기면 그 def 가 새 '이름 emit 헬퍼'(래퍼)다 — 호출처에서 검사한다.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return 'values', {node.value}
    if isinstance(node, ast.IfExp):
        results = [_resolve_emit_name(branch, chain, module_constants, bindings_of, seen)
                   for branch in (node.body, node.orelse)]
        if all(kind == 'values' for kind, _value in results):
            return 'values', results[0][1] | results[1][1]
        return 'unresolved', None
    if isinstance(node, ast.Name):
        for depth in range(len(chain) - 1, -1, -1):
            scope = chain[depth]
            if node.id in _bound_params(scope):
                if isinstance(scope, _FUNCTION_SCOPES):
                    positional = _positional_params(scope)
                    if node.id in positional:
                        return 'wrapper', (scope, positional.index(node.id), node.id)
                    if node.id in {a.arg for a in scope.args.kwonlyargs}:
                        return 'wrapper', (scope, None, node.id)
                return 'unresolved', None  # lambda 인자·*args·**kwargs — 호출처를 이름으로 못 찾는다
            if isinstance(scope, _FUNCTION_SCOPES):
                values = bindings_of(scope).get(node.id)
                if values is not None:
                    key = (scope, node.id)
                    if len(values) != 1 or values[0] is None or key in seen:
                        return 'unresolved', None
                    return _resolve_emit_name(values[0], chain[:depth + 1], module_constants,
                                              bindings_of, seen | {key})
    values = _string_items(node, module_constants)
    return ('values', values) if values is not None else ('unresolved', None)


def _name_dispatch_emits(sources, trees=None):
    """이름으로 시그널을 찾아 emit 하는 곳 → (found, helpers, unresolved).

    - found: {시그널 이름: [(위치, 'getattr' | 헬퍼 이름)]}
    - helpers: {이름 emit 헬퍼 def 이름: (인자 위치 | None=키워드 전용, 인자 이름)}
    - unresolved: 이름을 정적으로 읽지 못한 호출 위치 목록

    1) `getattr(<bridge 식>, X, ...)` — 공개 이름(밑줄 없음)만 found 에 넣는다.
    2) 매개변수로 `getattr(<bridge 식>, param)` 을 하는 '이름 emit 헬퍼'(`_creator_emit`, `_xyz_emit` …)와
       자기 매개변수를 그대로 헬퍼에 넘기는 래퍼 def 까지 고정점으로 찾아, 그 호출의 이름 인자.
    이름 인자를 해석 못 하면(함수 호출 결과·루프 변수·lambda 인자·*args·인자 누락 …) unresolved 에
    남긴다 — 액션 표(_python_action_table)와 같은 정책: 오타가 조용히 검사에서 빠지지 않게 한다.
    """
    trees = trees if trees is not None else _parse_sources(sources)
    defined = {fn.name for tree in trees.values() for fn in ast.walk(tree)
               if isinstance(fn, _FUNCTION_SCOPES)}
    candidates = []
    for rel, tree in trees.items():
        constants = _collect_constants(tree)
        for call, chain in _scoped_calls(tree):
            name = _call_name(call)
            if name == 'getattr' or name in defined:
                candidates.append((rel, constants, call, chain, name))
    binding_cache = {}

    def bindings_of(fn):
        if fn not in binding_cache:
            binding_cache[fn] = _local_bindings(fn)
        return binding_cache[fn]

    helpers = {}
    while True:  # 래퍼가 새 헬퍼를 만들면 그 호출처까지 다시 본다(헬퍼는 늘기만 해 반드시 끝난다)
        found, unresolved, grown = {}, [], False
        for rel, constants, call, chain, name in candidates:
            sink = _emit_name_arg(call, name, helpers)
            if sink is None:
                continue
            kind, arg = sink
            where = f"{rel}:{call.lineno}"
            if arg is None:
                unresolved.append(f"{where} ({kind}: 이름 인자 없음 또는 *args)")
                continue
            result, value = _resolve_emit_name(arg, chain, constants, bindings_of)
            if result == 'wrapper':
                fn, index, param = value
                if fn.name not in helpers:
                    helpers[fn.name] = (index, param)
                    grown = True
                continue
            if result == 'unresolved':
                unresolved.append(f"{where} ({kind}: {ast.unparse(arg)[:80]})")
                continue
            for signal in sorted(value):
                if kind == 'getattr' and signal.startswith('_'):
                    continue  # 브리지의 내부 속성(_proxies 등) 조회 — 시그널이 아니다
                found.setdefault(signal, []).append((where, kind))
        if not grown:
            return found, helpers, unresolved


# ── 프론트 런타임 소스 ─────────────────────────────────────────────────────

def _is_runtime_frontend(rel):
    """타입 선언·테스트(타입 수준 *.test-d.ts 포함)는 호출 근거가 아니다(런타임에 실행되지 않음)."""
    name = rel.rsplit('/', 1)[-1]
    return not (name.endswith('.d.ts') or '.test.' in name or '.test-d.' in name or '.spec.' in name)


# 앱 진입점(frontend/index.html → /src/main.js). 뷰는 여기서 import 사슬(router.js 의 lazy import 포함)로
# 닿아야 번들에 들고 실행된다.
_FRONT_ENTRY = 'frontend/src/main.js'
# 상대 경로 import 지정자 — `import x from '…'`·`export … from '…'`·`import '…'`(부수효과)·`import('…')`.
# 이 저장소엔 경로 별칭(@/…)이 없다(vite.config.js). Vite 쿼리(`?raw` 등)는 group 3 으로 따로 받는다.
_IMPORT_SPEC = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])(\.{1,2}/[^'"?]+)(\?[^'"]*)?\1""")
# 모듈을 실행하지 않고 원문·URL 만 가져오는 쿼리 — 이렇게만 가리키는 뷰는 실행되지 않는다.
_NON_EXECUTING_QUERIES = frozenset({'raw', 'url'})
_IMPORT_SUFFIXES = ('', '.js', '.ts', '.vue', '.mjs', '/index.js', '/index.ts')

# ── 실행되는 import 만 남기기 ──────────────────────────────────────────────
# 원문에 정규식을 바로 돌리면 주석·문자열·HTML 주석·타입 전용 import 속의 지정자까지 '실행 경로'로
# 셌다 — 주석 처리한 라우트나 `import type` 으로만 남은 뷰가 살아 있는 것으로 보여 고아 뷰 검사와
# 그 뷰의 브리지 호출(역방향 검사의 근거)이 거짓 통과했다. 아래는 길이를 보존하며(줄 번호 유지)
# 실행되지 않는 부분을 공백으로 지운다. Qt·외부 의존 없이 순수 텍스트로만.

# 문자열이 import 지정자인 자리 — 바로 앞(주석을 지운 코드)이 `from` · `import` · `import(` 로 끝난다.
_SPEC_CONTEXT = re.compile(r"(?<![\w$.])(?:from|import\s*\(|import)\s*$")
# `/` 앞이 이 키워드면 나눗셈이 아니라 정규식 리터럴이다.
_REGEX_PREFIX_KEYWORDS = frozenset({
    'return', 'typeof', 'case', 'do', 'else', 'in', 'of', 'new', 'delete', 'void', 'throw',
    'instanceof', 'yield', 'await'})
# .vue 에서 import 가 있을 수 있는 곳은 <script> 블록뿐 — 템플릿·<style>·<!-- --> 는 지운다.
# 주석이 먼저 걸리면 그 안의 <script> 는 건너뛴다(왼쪽부터 짝을 찾는다).
_VUE_SCRIPT_OR_COMMENT = re.compile(r"<!--.*?-->|<script\b[^>]*>(.*?)</script\s*>", re.S | re.I)
# TS 가 지우는 타입 전용 import/export(`import type X from` · `import type { A } from` ·
# `import type * as N from` · `export type { A } from` · `export type * from`).
# `import type from './x'` 는 이름이 type 인 기본 import(값)라 걸리지 않는다.
_TYPE_ONLY_STATEMENT = re.compile(
    r"""(?<![\w$.])(?:import\s+type\s+(?:\{[^}]*\}|\*\s*as\s+[\w$]+|[\w$]+)"""
    r"""|export\s+type\s+(?:\{[^}]*\}|\*(?:\s*as\s+[\w$]+)?))\s*from\s*(['"])[^'"\n]*\1""")
# `import { … } from` / `export { … } from` — 항목이 전부 `type X` 면 TS 가 문장째 지운다.
_NAMED_CLAUSE = re.compile(r"""(?<![\w$.])(?:import|export)\s*\{([^}]*)\}\s*from\s*(['"])[^'"\n]*\2""")
# 선언만 하고 쓰지 않는 lazy 라우트 — `const X = () => import('./views/X.vue')`. 라우트 항목을 지우고
# 이 줄만 남기면 번들러가 쓰이지 않는 화살표 함수째 버려 뷰가 실행되지 않는다.
_LAZY_IMPORT_CONST = re.compile(
    r"""(?<![\w$.])(export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"""
    r"""\(\s*\)\s*=>\s*import\s*\(\s*(['"])[^'"\n]*\3\s*\)""")
# .vue 템플릿이 <script setup> 바인딩을 쓰는 자리 — `{{ … }}` 보간, 바인딩 속성(`:x` · `v-bind:x` · `@x` ·
# `#slot` · `v-if` 등 `v-*`)의 값, 같은 이름 줄임(`:loadMore` = `:loadMore="loadMore"`), 컴포넌트 태그.
# 템플릿 밖(<script>·<style>·<!-- -->)은 먼저 지운다. 평문 텍스트·일반 속성 값은 사용이 아니다.
_VUE_NON_TEMPLATE = re.compile(r"<!--.*?-->|<(script|style)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_VUE_INTERPOLATION = re.compile(r"\{\{(.*?)\}\}", re.S)
_VUE_BOUND_VALUE = re.compile(r"""(?<![\w-])(?:[:@#]|v-)[^\s=/>"']*\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_VUE_BOUND_SHORTHAND = re.compile(r"(?<![\w-])(?::|v-bind:)([A-Za-z_$][\w$-]*)(?![\w$-])(?!\s*=)")
_VUE_TAG = re.compile(r"<([A-Za-z][\w.-]*)")


def _blank(chars, start, end):
    """chars[start:end] 를 공백으로 — 줄바꿈은 남겨 줄 번호와 길이를 보존한다."""
    for k in range(max(0, start), min(end, len(chars))):
        if chars[k] not in '\r\n':
            chars[k] = ' '


def _regex_literal_allowed(chars, index):
    """index 의 `/` 가 정규식 리터럴의 시작인가(아니면 나눗셈) — 앞의 의미 있는 토큰으로 가른다."""
    k = index - 1
    while k >= 0 and chars[k].isspace():
        k -= 1
    if k < 0:
        return True
    ch = chars[k]
    if ch.isalnum() or ch in '_$':
        start = k
        while start > 0 and (chars[start - 1].isalnum() or chars[start - 1] in '_$'):
            start -= 1
        return ''.join(chars[start:k + 1]) in _REGEX_PREFIX_KEYWORDS
    return ch not in ')]\'"`'


def _strip_js_non_code(text, keep_specifiers=True):
    """JS/TS 에서 주석을 지우고 문자열·템플릿·정규식 리터럴의 내용을 지운다(import 지정자 문자열은 남긴다).

    `//` 가 든 URL 문자열을 주석으로 오인하지 않고, 주석 뒤 같은 줄의 import 는 남는다.
    템플릿 리터럴의 `${ … }` 안은 코드로 본다(중첩 템플릿 포함). 길이·줄바꿈을 보존한다.
    ``keep_specifiers=False`` 면 import 지정자 문자열의 내용도 지운다 — 식별자 사용을 셀 때
    `'./views/I2IView.vue'` 속 파일 이름을 상수 `I2IView` 의 사용으로 세지 않게.
    """
    chars = list(text)
    n = len(text)
    expr_depths = []      # 열린 `${` 마다 그 안의 `{` 깊이
    in_template = False
    i = 0
    while i < n:
        ch = text[i]
        if in_template:
            if ch == '\\':
                _blank(chars, i, i + 2)
                i += 2
            elif ch == '`':
                in_template = False
                i += 1
            elif text.startswith('${', i):
                expr_depths.append(0)
                in_template = False
                i += 2
            else:
                _blank(chars, i, i + 1)
                i += 1
            continue
        if text.startswith('//', i):
            end = text.find('\n', i)
            end = n if end < 0 else end
            _blank(chars, i, end)
            i = end
        elif text.startswith('/*', i):
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            _blank(chars, i, end)
            i = end
        elif ch in '\'"':
            j = i + 1
            while j < n and text[j] != ch and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            j = min(j, n)
            if not keep_specifiers or not _SPEC_CONTEXT.search(''.join(chars[max(0, i - 80):i])):
                _blank(chars, i + 1, j)
            i = j + 1
        elif ch == '`':
            in_template = True
            i += 1
        elif ch == '{' and expr_depths:
            expr_depths[-1] += 1
            i += 1
        elif ch == '}' and expr_depths:
            if expr_depths[-1] == 0:
                expr_depths.pop()
                in_template = True
            else:
                expr_depths[-1] -= 1
            i += 1
        elif ch == '/' and _regex_literal_allowed(chars, i):
            j, in_class = i + 1, False
            while j < n and text[j] != '\n':
                c = text[j]
                if c == '\\':
                    j += 2
                    continue
                if c == '[':
                    in_class = True
                elif c == ']':
                    in_class = False
                elif c == '/' and not in_class:
                    break
                j += 1
            _blank(chars, i + 1, j)
            i = j + 1
        else:
            i += 1
    return ''.join(chars)


def _is_type_only_specifier(item):
    """named import/export 항목 하나가 `type X`(·`type X as Y`)인가. `type`·`type as X` 는 값 이름이다."""
    tokens = item.split()
    return bool(tokens) and tokens[0] == 'type' and (
        len(tokens) == 2 or (len(tokens) == 4 and tokens[2] == 'as'))


def _camelize(name):
    """케밥 이름 → camelCase(Vue 가 `<lazy-card>` · `:load-more` 를 `LazyCard` · `loadMore` 로 찾는 규칙)."""
    return re.sub(r'-([A-Za-z0-9])', lambda m: m.group(1).upper(), name)


def _vue_template_use_text(text):
    """.vue 템플릿에서 스크립트 바인딩을 쓰는 자리만 모은 텍스트(문자열·주석 내용은 지움).

    `<script setup>` 의 lazy 상수를 템플릿에서만 부르는 경우(`@click="load()"`)를 '쓰지 않는 상수'로
    오인하지 않게 한다. 사용으로 치는 곳은 `{{ … }}` 보간, 바인딩 속성 값, 같은 이름 줄임, 태그 이름
    (원문·camelCase·PascalCase)뿐 — 평문 텍스트·일반 속성 값·HTML 주석 속 이름은 사용이 아니다.
    """
    outside = _VUE_NON_TEMPLATE.sub(' ', text)
    exprs = [m.group(1) for m in _VUE_INTERPOLATION.finditer(outside)]
    exprs += [m.group(1) if m.group(1) is not None else m.group(2) for m in _VUE_BOUND_VALUE.finditer(outside)]
    names = [_camelize(m.group(1)) for m in _VUE_BOUND_SHORTHAND.finditer(outside)]
    for m in _VUE_TAG.finditer(outside):
        tag = m.group(1).split('.')[0]
        camel = _camelize(tag)
        names += [tag, camel, camel[:1].upper() + camel[1:]]
    return '\n'.join([_strip_js_non_code(expr, keep_specifiers=False) for expr in exprs] + names)


def _executable_import_text(rel, text):
    """rel 파일에서 **실행되는** import 지정자만 남긴 텍스트(길이 보존).

    지우는 것: .vue 의 <script> 밖(템플릿·<!-- --> 등), 줄·블록 주석, import 지정자가 아닌 문자열·
    템플릿·정규식 리터럴의 내용, TS 가 지우는 타입 전용 import/export(`import type …` ·
    `import { type A, type B } from` · `export type … from`), 선언만 하고 파일 안 어디서도 쓰지 않는
    lazy import 상수(`const X = () => import('…')` — export 하면 쓰는 것으로 본다).
    '사용'은 선언 밖의 코드(모든 문자열 내용을 지운 텍스트 — import 지정자 속 같은 이름의 파일 이름은
    사용이 아니다)와, .vue 면 템플릿의 바인딩 자리(:func:`_vue_template_use_text`)에서 센다.
    남는 것: `import { type A, B } from` 같은 섞인 import, 주석 뒤 같은 줄의 import 등 실제 실행 경로.
    알려진 한계: TS 의 '값으로 쓰지 않는 import 생략'(타입으로만 쓰는 일반 import)과, 쓰지 않는
    lazy 상수를 다른 쓰지 않는 상수만 가리키는 사슬은 보지 않는다(실행되는 것으로 센다).
    """
    template_uses = ''
    if rel.endswith('.vue'):
        template_uses = _vue_template_use_text(text)
        chars = [c if c in '\r\n' else ' ' for c in text]
        for match in _VUE_SCRIPT_OR_COMMENT.finditer(text):
            if match.group(1) is not None:
                start, end = match.span(1)
                chars[start:end] = text[start:end]
        text = ''.join(chars)
    code = _strip_js_non_code(text)
    chars = list(code)
    for match in _TYPE_ONLY_STATEMENT.finditer(code):
        _blank(chars, *match.span())
    for match in _NAMED_CLAUSE.finditer(code):
        items = [item for item in match.group(1).split(',') if item.strip()]
        if items and all(_is_type_only_specifier(item) for item in items):
            _blank(chars, *match.span())
    code = ''.join(chars)
    # 사용을 셀 텍스트 — 지정자 문자열까지 지운다. 예전엔 `code`(지정자 보존)에서 세어
    # `const I2IView = () => import('./views/I2IView.vue')` 가 지정자 속 이름으로 늘 '쓰임'이 됐다.
    names_text = _strip_js_non_code(text, keep_specifiers=False)
    for match in _LAZY_IMPORT_CONST.finditer(code):
        if match.group(1):
            continue
        start, end = match.span()
        outside = names_text[:start] + ' ' * (end - start) + names_text[end:] + '\n' + template_uses
        if not re.search(r"(?<![\w$.])" + re.escape(match.group(2)) + r"(?![\w$])", outside):
            _blank(chars, start, end)
    return ''.join(chars)


def _resolve_frontend_import(rel, spec, front_sources):
    """상대 지정자 → front_sources 키. 읽지 않는 파일(.css 등)·타입 선언만 있는 경로면 None."""
    base = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
    candidates = [base + suffix for suffix in _IMPORT_SUFFIXES]
    if base.endswith('.js'):
        candidates.append(base[:-3] + '.ts')  # TS ESM 관례: './x.js' 가 x.ts 를 가리킨다
    return next((c for c in candidates if c in front_sources), None)


def _reachable_frontend(front_sources, entry=_FRONT_ENTRY):
    """진입점에서 상대 import 사슬로 닿는 런타임 파일. 테스트·타입 선언은 따라가지 않는다.

    각 파일에서 실행되는 import 만 따라간다(:func:`_executable_import_text` — 주석·문자열·HTML 주석·
    타입 전용 import·쓰지 않는 lazy 상수는 근거가 아니다).
    """
    seen, stack = set(), [entry]
    while stack:
        rel = stack.pop()
        if rel in seen or rel not in front_sources or not _is_runtime_frontend(rel):
            continue
        seen.add(rel)
        for match in _IMPORT_SPEC.finditer(_executable_import_text(rel, front_sources[rel])):
            query = (match.group(3) or '').lstrip('?')
            if query.split('&')[0] in _NON_EXECUTING_QUERIES:
                continue
            target = _resolve_frontend_import(rel, match.group(2), front_sources)
            if target is not None:
                stack.append(target)
    return seen


def _orphan_views(front_sources):
    """앱 진입점에서 import 사슬로 닿지 않는 views/*.vue — 라우트가 없어 실행되지 않는다.

    예전엔 뷰 이름 문자열만 찾아서 뷰 옆 테스트(`ChatView.test.ts` 의 `./ChatView.vue?raw`)·주석·
    역시 고아인 다른 파일이 가리켜도 살아 있는 뷰로 오인했다 — 실제 import 그래프로 본다(감사 #83).
    그래프의 간선은 실행되는 import 뿐이다 — 줄·블록·HTML 주석, 문자열 리터럴, `import type`·
    `import { type X }` 같은 타입 전용 import, 쓰지 않는 lazy 라우트 상수가 가리키는 뷰는 고아다
    (:func:`_executable_import_text`). 예전엔 원문 정규식이라 이것들도 간선으로 셌다.
    """
    reachable = _reachable_frontend(front_sources)
    return {rel for rel in front_sources
            if rel.startswith('frontend/src/views/') and rel.endswith('.vue') and rel not in reachable}


def _frontend_runtime_texts(front_sources):
    orphans = _orphan_views(front_sources)
    return [text for rel, text in front_sources.items()
            if _is_runtime_frontend(rel) and rel not in orphans and rel not in _NON_CALLSITE_FRONTEND]


def _frontend_literals(texts):
    names = set()
    for text in texts:
        names.update(m.group(2) for m in _FE_LITERAL.finditer(text))
    return names


_MEMBER_ACCESS = re.compile(r"\.([A-Za-z_]\w*)")
_FE_EXPORTED = re.compile(r"\bexport\s+(?:async\s+)?(?:function\*?|const|let)\s+([A-Za-z_]\w*)")
# 프론트가 브리지 객체를 담는 변수 이름(실측: backend/bk/_bk/_backend/b/host).
_BRIDGE_RECEIVER = r"(?<![\w$.])(?:backend|_backend|bk|_bk|b|host|bridge)\??\."


def _frontend_used_slot_names(slots, texts):
    """프론트가 부르는 슬롯 — `<아무것>.X` 멤버 접근이나 따옴표 리터럴 'X'.

    test_web_mode_security 와 같은 기준이되, 프론트가 **같은 이름을 스스로 export** 하는 경우
    (widgetStore 의 requestAction 등)는 `widgetStore.requestAction` 이 슬롯 호출로 오인되므로
    브리지 변수(`backend.X` 등)를 거친 접근이나 리터럴만 근거로 친다.
    """
    literals = _frontend_literals(texts)
    members, exported = set(), set()
    for text in texts:
        members.update(_MEMBER_ACCESS.findall(text))
        exported.update(_FE_EXPORTED.findall(text))
    joined = "\n".join(texts)
    used = set()
    for name in slots:
        if name in literals:
            used.add(name)
        elif name in exported:
            if re.search(_BRIDGE_RECEIVER + re.escape(name) + r"\b", joined):
                used.add(name)
        elif name in members:
            used.add(name)
    return used


def _typescript_union(text, name):
    """bridge.d.ts의 문자열 literal union만 추출한다."""
    match = re.search(
        rf"export\s+type\s+{re.escape(name)}\s*=\s*(.*?)"
        rf"(?=\n\s*(?:/\*\*|export\s+(?:type|interface))|\Z)",
        text,
        re.S,
    )
    if not match:
        return set()
    return set(_TS_LITERAL.findall(match.group(1)))


class TestBridgeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        front_sources = _read_tree(FRONT, ('.vue', '.js', '.ts', '.mjs'))
        # 정방향 검사는 예전과 같은 범위(.vue/.js/.ts 전체) — .mjs 는 역방향 근거로만 쓴다.
        cls.front = [text for rel, text in front_sources.items() if not rel.endswith('.mjs')]
        cls.front_runtime = _frontend_runtime_texts(front_sources)
        cls.orphan_views = _orphan_views(front_sources)
        cls.view_files = {rel for rel in front_sources
                          if rel.startswith('frontend/src/views/') and rel.endswith('.vue')}
        cls.reachable_frontend = _reachable_frontend(front_sources)
        cls.ui_sources = _read_tree(UI, ('.py',))
        cls.ui_trees = _parse_sources(cls.ui_sources)  # 큰 파일이 많다 — 한 번만 파싱
        cls.action_table, cls.unresolved_actions = _python_action_table(cls.ui_sources, cls.ui_trees)
        cls.emits, cls.emit_helpers, cls.unresolved_emits = _name_dispatch_emits(
            cls.ui_sources, cls.ui_trees)
        with open(os.path.join(UI, 'vue_bridge.py'), encoding='utf-8') as f:
            signals, slots, members = _bridge_members(f.read())
        cls.signals, cls.slots, cls.bridge_members = sorted(signals), slots, members
        cls.front_literals = _frontend_literals(cls.front_runtime)
        cls.used_slots = _frontend_used_slot_names(cls.slots, cls.front_runtime)
        listened = _frontend_events(cls.front_runtime)
        for text in cls.front_runtime:
            listened.update(_CONNECT.findall(text))
        cls.listened_signals = listened & set(cls.signals)
        web_main = os.path.join(ROOT, 'web_main_ui.py')
        cls.web_methods = _module_string_set(web_main, '_WEB_METHODS')
        cls.web_signals = _module_string_set(web_main, '_WEB_SIGNALS')
        with open(os.path.join(FRONT, 'types', 'bridge.d.ts'), encoding='utf-8') as f:
            cls.bridge_types = f.read()
        cls.typed_actions = _typescript_union(cls.bridge_types, 'ActionName')
        cls.typed_events = _typescript_union(cls.bridge_types, 'BackendEvent')

    def _frontend_used_slots(self):
        return self.used_slots

    def _frontend_listened_signals(self):
        return self.listened_signals

    def _allowed(self, kind):
        return ({name for (k, name) in PYTHON_INTERNAL if k == kind}
                | {name for (k, name) in PENDING if k == kind})

    def test_extraction_not_empty(self):
        # 안전장치 — 추출 0건이면 경로/정규식이 깨진 것(거짓 통과 방지)
        self.assertGreater(len(_frontend_actions(self.front)), 20)
        self.assertGreater(len(self.action_table), 20)
        self.assertGreater(len(self.signals), 20)
        self.assertGreater(len(self.slots), 20)
        self.assertGreater(len(self.web_methods), 20)
        self.assertGreater(len(self.web_signals), 20)
        self.assertGreater(len(_frontend_events(self.front)), 20)
        self.assertGreater(len(self.typed_actions), 20)
        self.assertGreater(len(self.typed_events), 20)
        self.assertGreater(len(self.emits), 5)
        self.assertTrue({'_creator_emit', '_xyz_emit'} <= set(self.emit_helpers),
                        f"이름 emit 헬퍼 자동 탐지가 깨졌습니다: {sorted(self.emit_helpers)}")

    def test_python_action_extraction_is_fully_resolved(self):
        """`action in <상수>` 를 못 따라가면 그 액션들이 계약 검사에서 조용히 빠진다."""
        self.assertEqual(
            self.unresolved_actions, [],
            "디스패처의 action 비교 대상을 정적으로 해석하지 못했습니다 — 리터럴이나 "
            "모듈/클래스 상수(tuple/set/frozenset/dict)로 두세요.")

    def test_local_action_variables_are_not_mistaken_for_handlers(self):
        """조건식 규칙·런타임 이벤트의 지역변수 action(start/stop/add …)은 핸들러가 아니다."""
        for rel, tree in self.ui_trees.items():
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef) and fn.name == '_on_backend_runtime_event':
                    self.assertFalse(_is_action_dispatcher(fn), rel)
        self.assertNotIn('stop', self.action_table)
        self.assertNotIn('start', self.action_table)

    def test_dispatcher_gates_are_seen(self):
        """`action not in {...}` 게이트와 `action in <dict 상수>` 디스패치도 액션으로 인식한다."""
        for name in ('hand_reconstruction_generate', 'hand_reconstruction_export',
                     'hand_reconstruction_cancel', 'creator_h3_cache_clear', 'comic_export_living'):
            self.assertIn(name, self.action_table)

    def test_every_frontend_action_has_python_handler(self):
        emitted = _frontend_actions(self.front)
        handled = set(self.action_table)
        missing = emitted - handled - ACTION_ALLOWLIST
        self.assertEqual(
            missing, set(),
            f"프론트가 호출하지만 Python 핸들러(action=='X' / in (...))가 없는 액션: {sorted(missing)}")

    def test_every_frontend_event_has_signal(self):
        listened = _frontend_events(self.front)
        signals = set(self.signals)
        missing = listened - signals - EVENT_ALLOWLIST
        self.assertEqual(
            missing, set(),
            f"프론트가 onBackendEvent로 듣지만 pyqtSignal이 없는 이벤트: {sorted(missing)}")

    def test_bridge_types_are_strict_and_cover_frontend_usage(self):
        self.assertNotIn(
            '(string & {})',
            self.bridge_types,
            '알 수 없는 bridge 이름을 허용하면 TypeScript 계약 드리프트를 잡을 수 없습니다.',
        )

        missing_actions = _frontend_actions(self.front) - self.typed_actions
        missing_events = _frontend_events(self.front) - self.typed_events
        self.assertEqual(
            missing_actions,
            set(),
            f"frontend 사용 액션이 ActionName union에 없습니다: {sorted(missing_actions)}",
        )
        self.assertEqual(
            missing_events,
            set(),
            f"frontend 사용 이벤트가 BackendEvent union에 없습니다: {sorted(missing_events)}",
        )

    def test_typed_names_exist_in_python_contract(self):
        handled = set(self.action_table)
        signals = set(self.signals)
        self.assertEqual(
            self.typed_actions - handled - ACTION_ALLOWLIST,
            set(),
            'ActionName union에 Python 핸들러가 없는 이름이 있습니다.',
        )
        self.assertEqual(
            self.typed_events - signals - EVENT_ALLOWLIST,
            set(),
            'BackendEvent union에 Python signal이 없는 이름이 있습니다.',
        )

    # ── 역방향: Python 에만 남은 이름 ────────────────────────────────────────

    def test_every_python_action_is_sent_by_the_frontend(self):
        """핸들러는 있는데 프론트 런타임 어디에도 그 이름이 없으면 사문 액션이다.

        래퍼(`runAction('x')` 등)를 거치는 호출도 잡도록 프론트의 모든 문자열 리터럴과 대조한다.
        bridge.d.ts·테스트·라우트 없는 고아 뷰는 호출 근거에서 뺀다.
        """
        literals = self.front_literals
        py_only = {name for name in self.action_table if name not in literals}
        unexpected = py_only - self._allowed('action')
        self.assertEqual(
            unexpected, set(),
            "Python 핸들러만 있고 프론트가 보내지 않는 액션: "
            + ", ".join(f"{n} ({', '.join(sorted(self.action_table[n]))})" for n in sorted(unexpected))
            + " — 핸들러를 지우거나, 미룰 사유를 PENDING 에 적어라.")

    def test_every_bridge_slot_is_called_by_the_frontend(self):
        unused = self.slots - self._frontend_used_slots() - self._allowed('slot')
        self.assertEqual(
            unused, set(),
            f"프론트가 부르지 않는 @pyqtSlot: {sorted(unused)} — 지우거나 PENDING/PYTHON_INTERNAL 에 사유를 적어라.")

    def test_every_bridge_signal_is_listened_to(self):
        unused = set(self.signals) - self._frontend_listened_signals() - self._allowed('signal')
        self.assertEqual(
            unused, set(),
            f"프론트가 듣지 않는 pyqtSignal: {sorted(unused)} — 지우거나 PENDING/PYTHON_INTERNAL 에 사유를 적어라.")

    def test_web_whitelists_expose_only_frontend_used_names(self):
        """_WEB_METHODS/_WEB_SIGNALS 에 프론트가 안 쓰는 이름이 남으면 웹 공격면만 넓힌다."""
        pending_slots = {name for (kind, name) in PENDING if kind == 'slot'}
        stale_methods = self.web_methods - self._frontend_used_slots() - pending_slots
        self.assertEqual(
            stale_methods, set(),
            f"_WEB_METHODS 에 프론트가 부르지 않는 슬롯: {sorted(stale_methods)}")
        pending_signals = {name for (kind, name) in PENDING if kind == 'signal'}
        stale_signals = self.web_signals - self._frontend_listened_signals() - pending_signals
        self.assertEqual(
            stale_signals, set(),
            f"_WEB_SIGNALS 에 프론트가 듣지 않는 시그널: {sorted(stale_signals)}")

    def test_reverse_pending_entries_are_not_stale(self):
        """정리가 끝났는데 PENDING/PYTHON_INTERNAL 에 남아 있으면 알려 준다 — 목록이 낡지 않게."""
        literals = self.front_literals
        present = {
            'action': {n for n in self.action_table if n not in literals},
            'slot': self.slots - self._frontend_used_slots(),
            'signal': set(self.signals) - self._frontend_listened_signals(),
        }
        stale = sorted(
            key for key in list(PENDING) + list(PYTHON_INTERNAL)
            if key[1] not in present.get(key[0], set()))
        self.assertEqual(stale, [], "이미 정리됐거나 프론트가 쓰기 시작한 항목이 목록에 남아 있다. 지워라.")
        self.assertTrue(all(reason.strip() for reason in PENDING.values()))
        self.assertTrue(set(PENDING).isdisjoint(PYTHON_INTERNAL))

    def test_every_view_is_reachable_from_the_app_entry(self):
        """라우트·import 가 빠진 뷰는 번들에 들지 않는 고아다 — 한 건도 두지 않는다.

        감사 #83: BackendView/WebView 가 라우터에서 빠진 뒤 두 달간 타입 검사·계약 테스트·스윕
        커밋의 수정 대상으로만 남았다. 새 뷰는 router.js 에 배선하고, 안 쓰게 된 뷰는 지운다.
        판별기 자체가 공허하지 않은지는 test_orphan_view_detection_follows_the_import_graph 가 지킨다.
        """
        self.assertEqual(
            sorted(self.orphan_views), [],
            "main.js 에서 import 사슬(router.js 의 lazy import 포함)로 닿지 않는 뷰 — 라우트에 배선하거나 지워라")
        # 실제 트리에서도 그래프가 뷰까지 이어지는지(뷰 목록·진입점이 비어 공허하게 통과하지 않게)
        self.assertIn('frontend/src/views/SettingsView.vue', self.view_files)
        self.assertTrue(self.view_files <= self.reachable_frontend)

    def test_orphan_view_detection_follows_the_import_graph(self):
        """이름 문자열이 아니라 실행 경로로 본다 — 테스트·?raw·타입 선언·고아끼리의 참조는 근거가 아니다.
        주석·문자열·타입 전용 import 는 test_non_executing_import_text_is_not_an_edge 가 지킨다."""
        sources = {
            'frontend/src/main.js': "import './style.css'\nimport router from './router.js'\n",
            'frontend/src/router.js': (
                "// 지운 CommentedView.vue 자리 — views/CommentedView 는 더 이상 라우트가 없다\n"
                "const Live = () => import('./views/LiveView.vue')\n"
                "export const routes = [{ path: '/live', component: Live }]\n"
                "export { helper } from './utils/helper.js'\n"),
            # TS ESM 관례('./helper.js' → helper.ts)와 멀티라인 import 도 따라간다
            'frontend/src/utils/helper.ts': (
                "import {\n  default as Nested,\n} from '../views/NestedView.vue'\n"
                "export const helper = Nested\n"),
            'frontend/src/views/LiveView.vue': (
                "<script setup>\nimport Panel from '../components/Panel.vue'\n</script>\n"),
            'frontend/src/components/Panel.vue': (
                "<script setup>\nimport rawSource from '../views/RawView.vue?raw'\n</script>\n"),
            'frontend/src/views/NestedView.vue': "",
            'frontend/src/views/RawView.vue': "",
            'frontend/src/views/TestedView.vue': "",
            'frontend/src/views/TestedView.test.ts': "import source from './TestedView.vue?raw'\n",
            'frontend/src/views/TypedView.vue': "",
            'frontend/src/types/views.d.ts': "import type TypedView from '../views/TypedView.vue'\n",
            'frontend/src/views/CommentedView.vue': "",
            'frontend/src/views/DeadView.vue': "<script setup>\nimport Chained from './ChainedView.vue'\n</script>\n",
            'frontend/src/views/ChainedView.vue': "",
        }
        self.assertEqual(
            _orphan_views(sources),
            {f'frontend/src/views/{name}.vue'
             for name in ('RawView', 'TestedView', 'TypedView', 'CommentedView', 'DeadView', 'ChainedView')})
        reachable = _reachable_frontend(sources)
        self.assertTrue({'frontend/src/utils/helper.ts', 'frontend/src/components/Panel.vue',
                         'frontend/src/views/LiveView.vue', 'frontend/src/views/NestedView.vue'} <= reachable)

    def test_non_executing_import_text_is_not_an_edge(self):
        """주석·문자열·HTML 주석·타입 전용 import·쓰지 않는 lazy 상수로만 가리키는 뷰는 고아다.

        예전엔 원문에 정규식을 돌려 이것들도 실행 경로로 셌다 — 라우트를 주석 처리하거나 `import type`
        으로만 남긴 뷰가 살아 있는 것으로 보였고, 그 뷰의 requestAction 이 역방향 검사의 호출 근거로 남아
        사문 Python 핸들러까지 거짓 통과했다. 섞인 import·주석/URL/정규식/나눗셈 뒤의 import 는 남는다.
        """
        sources = {
            'frontend/src/main.js': (
                "import router from './router.js'\n"
                "import './utils/types.ts'\n"
                "import Shell from './components/Shell.vue'\n"),
            'frontend/src/router.js': (
                "// const C1 = () => import('./views/LineCommentView.vue')\n"
                "/* { path: '/b', component: () => import('./views/BlockCommentView.vue') } */\n"
                "// const Dead = () => import('./views/DeadCommentView.vue')\n"
                "const hint = \"see import('./views/StringView.vue')\"\n"
                "const doc = `from './views/TemplateStringView.vue'`\n"
                "const Unrouted = () => import('./views/UnroutedView.vue')\n"
                "const Live = () => import('./views/LiveView.vue')\n"
                "export const Exported = () => import('./views/ExportedLazyView.vue')\n"
                "export const routes = [\n"
                "  { path: '/live', component: Live },\n"
                "  { path: '/a', component: /* lazy */ () => import('./views/AfterCommentView.vue') },\n"
                "]\n"
                "const u = 'http://x'; routes.push({ component: () => import('./views/UrlLineView.vue') })\n"
                "const re = /['\"\\/]+/; routes.push({ component: () => import('./views/AfterRegexView.vue') })\n"
                "const half = routes.length / 2; routes.push({ component: () => import('./views/AfterDivisionView.vue') })\n"
                "const label = `${(await import('./views/TemplateExprView.vue')).default.name}`\n"),
            'frontend/src/utils/types.ts': (
                "import type T from '../views/TypeOnlyView.vue'\n"
                "import { type P } from '../views/InlineTypeView.vue'\n"
                "import { type P1, type P2 as Q } from '../views/InlineTypesView.vue'\n"
                "export type { R } from '../views/ExportTypeView.vue'\n"
                "import type * as NS from '../views/NamespaceTypeView.vue'\n"
                "import { type A, Real } from '../views/MixedView.vue'\n"
                "import X, { type B } from '../views/DefaultPlusTypeView.vue'\n"
                "import type from '../views/TypeNamedView.vue'\n"
                "export const used = [Real, X, type]\n"),
            'frontend/src/components/Shell.vue': (
                "<template>\n"
                "  <!-- import X from '../views/TemplateCommentView.vue' -->\n"
                "  <div>import Y from '../views/TemplateTextView.vue'</div>\n"
                "  <!-- <script>import Z from '../views/HtmlCommentScriptView.vue'</script> -->\n"
                "</template>\n"
                "<script setup lang=\"ts\">\n"
                "// import Y from '../views/ScriptCommentView.vue'\n"
                "import Kept from '../views/ScriptKeptView.vue'\n"
                "</script>\n"),
            'frontend/src/views/LiveView.vue': "<script setup>\nrequestAction('live_only_action')\n</script>\n",
            'frontend/src/views/DeadCommentView.vue': (
                "<script setup>\nrequestAction('dead_only_action')\n</script>\n"),
        }
        orphans_expected = {
            'LineCommentView', 'BlockCommentView', 'DeadCommentView', 'StringView', 'TemplateStringView',
            'UnroutedView', 'TypeOnlyView', 'InlineTypeView', 'InlineTypesView', 'ExportTypeView',
            'NamespaceTypeView', 'TemplateCommentView', 'TemplateTextView', 'HtmlCommentScriptView',
            'ScriptCommentView',
        }
        reachable_expected = {
            'LiveView', 'ExportedLazyView', 'AfterCommentView', 'UrlLineView', 'AfterRegexView',
            'AfterDivisionView', 'TemplateExprView', 'MixedView', 'DefaultPlusTypeView', 'TypeNamedView',
            'ScriptKeptView',
        }
        for name in orphans_expected | reachable_expected:
            sources.setdefault(f'frontend/src/views/{name}.vue', "")
        self.assertEqual(
            _orphan_views(sources), {f'frontend/src/views/{name}.vue' for name in orphans_expected})
        reachable = _reachable_frontend(sources)
        self.assertTrue({f'frontend/src/views/{name}.vue' for name in reachable_expected} <= reachable)
        # 고아로 본 뷰의 호출은 역방향 검사의 근거에서 빠진다(살아 있는 뷰의 호출은 남는다)
        actions = _frontend_actions(_frontend_runtime_texts(sources))
        self.assertIn('live_only_action', actions)
        self.assertNotIn('dead_only_action', actions)

    def test_unused_lazy_const_named_like_its_file_is_not_an_edge(self):
        """router.js 관례(상수 이름 = 파일 이름, `const I2IView = () => import('./views/I2IView.vue')`)여도
        라우트 항목을 지우고 상수만 남기면 고아다.

        예전엔 사용 횟수를 지정자 문자열을 남긴 텍스트에서 세어 `'./views/I2IView.vue'` 속 `I2IView` 가
        사용으로 잡혔다(uses=2) — 고아 검사와 그 뷰의 requestAction(역방향 검사의 호출 근거)이 거짓
        통과했다. 다른 import 지정자·문자열·주석 속 같은 이름도 사용이 아니다.
        """
        sources = {
            'frontend/src/main.js': "import router from './router.js'\n",
            'frontend/src/router.js': (
                "import './styles/ChatView.css'\n"
                "const I2IView = () => import('./views/I2IView.vue')\n"
                "const ChatView = () => import('./views/ChatView.vue')\n"
                "const GalleryView = () => import('./views/GalleryView.vue')\n"
                "const note = 'I2IView'\n"
                "const routes = [\n"
                "  { path: '/gallery', component: GalleryView },\n"
                "  // { path: '/i2i', component: I2IView },\n"
                "]\n"
                "export default routes\n"),
            'frontend/src/views/I2IView.vue': "<script setup>\nrequestAction('i2i_dead_action')\n</script>\n",
            'frontend/src/views/ChatView.vue': "",
            'frontend/src/views/GalleryView.vue': (
                "<script setup>\nrequestAction('gallery_live_action')\n</script>\n"),
        }
        self.assertEqual(_orphan_views(sources),
                         {'frontend/src/views/I2IView.vue', 'frontend/src/views/ChatView.vue'})
        actions = _frontend_actions(_frontend_runtime_texts(sources))
        self.assertIn('gallery_live_action', actions)
        self.assertNotIn('i2i_dead_action', actions)
        # 리뷰 재현 — 상수 한 줄과 빈 라우트 표뿐인 router.js
        probe = {
            'frontend/src/main.js': "import router from './router.js'\n",
            'frontend/src/router.js': (
                "const I2IView = () => import('./views/I2IView.vue')\nexport const routes = []\n"),
            'frontend/src/views/I2IView.vue': "",
        }
        self.assertEqual(_orphan_views(probe), {'frontend/src/views/I2IView.vue'})

    def test_lazy_const_used_only_from_the_vue_template_is_an_edge(self):
        """`<script setup>` 의 lazy 상수를 템플릿에서만 불러도(`@click="openHelp()"` 등) 쓰이는 것이다.

        예전엔 <script> 밖을 지운 뒤 사용을 세어 이런 상수를 지웠다 — 가리키는 뷰가 거짓 고아가 되고
        그 파일의 호출 근거도 빠졌다. 바인딩 속성 값·보간·같은 이름 줄임·컴포넌트 태그는 사용이고,
        평문 텍스트·일반 속성 값·HTML 주석·표현식 속 문자열의 같은 이름은 사용이 아니다.
        """
        panel = (
            "<template>\n"
            "  <button @click=\"openHelp()\">?</button>\n"
            "  <chart-box :loader=\"loadChart\" />\n"
            "  <p v-if=\"ready\">{{ describe(loadStats) }}</p>\n"
            "  <lazy-card />\n"
            "  <Wrap :load-more />\n"
            "  <!-- <b @click=\"loadCommented()\">x</b> -->\n"
            "  <p title=\"loadAttr\">loadProse</p>\n"
            "  <i @click=\"go('loadQuoted')\">q</i>\n"
            "</template>\n"
            "<script setup>\n"
            "const openHelp = () => import('../views/HelpView.vue')\n"
            "const loadChart = () => import('../views/ChartView.vue')\n"
            "const loadStats = () => import('../views/StatsView.vue')\n"
            "const LazyCard = () => import('../views/CardView.vue')\n"
            "const loadMore = () => import('../views/MoreView.vue')\n"
            "const loadCommented = () => import('../views/CommentedView.vue')\n"
            "const loadAttr = () => import('../views/AttrView.vue')\n"
            "const loadProse = () => import('../views/ProseView.vue')\n"
            "const loadQuoted = () => import('../views/QuotedView.vue')\n"
            "</script>\n")
        sources = {'frontend/src/main.js': "import Panel from './components/Panel.vue'\n",
                   'frontend/src/components/Panel.vue': panel}
        live = {'HelpView', 'ChartView', 'StatsView', 'CardView', 'MoreView'}
        dead = {'CommentedView', 'AttrView', 'ProseView', 'QuotedView'}
        for name in live | dead:
            sources[f'frontend/src/views/{name}.vue'] = ""
        self.assertEqual(_orphan_views(sources), {f'frontend/src/views/{name}.vue' for name in dead})
        # 리뷰 재현 — 템플릿에서만 쓰는 상수의 지정자가 남는다
        vue = ("<template><button @click=\"load()\">x</button></template>\n"
               "<script setup>\nconst load = () => import('./x.js')\n</script>\n")
        out = _executable_import_text('frontend/src/a.vue', vue)
        self.assertEqual(len(out), len(vue))
        self.assertEqual([m.group(2) for m in _IMPORT_SPEC.finditer(out)], ['./x.js'])

    def test_executable_import_text_keeps_offsets_and_real_specifiers(self):
        """지운 자리는 같은 길이의 공백 — 줄 번호가 그대로고, 실제 지정자는 글자 그대로 남는다."""
        text = ("// c './a.js'\nimport a from './a.js' // tail\n"
                "const s = 'x\\'y'; /* ./b */ export * from \"./b.js\"\n")
        out = _executable_import_text('frontend/src/x.js', text)
        self.assertEqual(len(out), len(text))
        self.assertEqual(out.count('\n'), text.count('\n'))
        self.assertEqual([m.group(2) for m in _IMPORT_SPEC.finditer(out)], ['./a.js', './b.js'])
        vue = "<template><!-- x --></template>\n<script>\nimport a from './a.js'\n</script>\n<style>a{}</style>\n"
        out = _executable_import_text('frontend/src/x.vue', vue)
        self.assertEqual(len(out), len(vue))
        self.assertEqual([m.group(2) for m in _IMPORT_SPEC.finditer(out)], ['./a.js'])
        self.assertEqual(out.splitlines()[2], "import a from './a.js'")

    # ── 이름 기반 emit ─────────────────────────────────────────────────────

    def test_name_dispatched_emits_target_real_signals(self):
        """`_creator_emit("X")`·`getattr(bridge, "X", None)` 는 오타여도 조용히 버려진다."""
        emits = self.emits
        unknown = {name: sites for name, sites in emits.items() if name not in self.bridge_members}
        self.assertEqual(
            unknown, {},
            f"VueBridge 에 없는 이름으로 emit/getattr 한다(오타면 이벤트가 사라진다): {unknown}")
        # 이름 emit 헬퍼(_creator_emit 등)로 넘기는 이름은 반드시 시그널이어야 .emit 이 성립한다.
        not_signals = {name: sites for name, sites in emits.items()
                       if name not in self.signals
                       and any(kind != 'getattr' for _site, kind in sites)}
        self.assertEqual(not_signals, {},
                         f"이름 emit 헬퍼에 시그널이 아닌 이름을 넘긴다: {not_signals}")

    def test_name_dispatched_emit_names_are_fully_resolved(self):
        """이름 인자를 정적으로 못 읽으면 그 emit 은 오타 검사에서 조용히 빠진다(액션 표와 같은 정책)."""
        self.assertEqual(
            self.unresolved_emits, [],
            "이름 emit(`_creator_emit(X)`·`getattr(bridge, X)`)의 X 를 정적으로 해석하지 못했습니다 — "
            "리터럴·조건식 리터럴·모듈/클래스 상수·한 번만 대입한 지역 변수로 두거나, "
            "매개변수를 그대로 넘기는 래퍼 def 로 만드세요.")

    def test_show_toast_payload_includes_warning(self):
        match = re.search(
            r"interface\s+ShowToastPayload\s*\{(.*?)\}",
            self.bridge_types,
            re.S,
        )
        self.assertIsNotNone(match)
        self.assertIn("'warning'", match.group(1))


_FAKE_EMIT_HELPER = '''
import json

MODULE_SIG = "creatorCacheEvent"
_RELAY_ATTR = "_status_line_relay"


class Mixin:
    CLASS_SIG = "creatorProgress"

    def _creator_emit(self, signal_name, payload=None):
        bridge = getattr(self, "vue_bridge", None)
        signal = getattr(bridge, signal_name, None)
        if signal is not None:
            signal.emit(json.dumps(payload))
'''


class NameDispatchEmitExtractionTests(unittest.TestCase):
    """_name_dispatch_emits 자체의 회귀 — 해석 못 한 이름 인자는 조용히 빠지지 않고 unresolved 로 남는다."""

    def _extract(self, methods):
        body = textwrap.indent(textwrap.dedent(methods), '    ')
        return _name_dispatch_emits({'ui/fake.py': _FAKE_EMIT_HELPER + body})

    def _assert_single_unresolved(self, methods, needle):
        found, _helpers, unresolved = self._extract(methods)
        self.assertEqual(found, {})
        self.assertEqual(len(unresolved), 1, unresolved)
        self.assertTrue(unresolved[0].startswith('ui/fake.py:'), unresolved)
        self.assertIn(needle, unresolved[0])

    def test_helper_is_found_by_its_getattr_parameter(self):
        found, helpers, unresolved = self._extract('''
            def a(self):
                self._creator_emit("creatorResult", {})
        ''')
        self.assertEqual(helpers, {'_creator_emit': (0, 'signal_name')})
        self.assertEqual(list(found), ['creatorResult'])
        self.assertEqual(found['creatorResult'][0][1], '_creator_emit')
        self.assertEqual(unresolved, [])

    def test_conditional_local_name_resolves_both_branches(self):
        """리뷰 사례: 지역 변수로 고른 이름도 두 갈래를 다 읽어 오타(creatorProgres)를 잡는다."""
        found, _helpers, unresolved = self._extract('''
            def a(self, ok):
                signal = 'creatorResult' if ok else 'creatorProgres'
                self._creator_emit(signal, {})

            def b(self, ok):
                self._creator_emit('creatorStateChanged' if ok else 'creatorMediaSelected')
        ''')
        self.assertEqual(set(found), {'creatorResult', 'creatorProgres',
                                      'creatorStateChanged', 'creatorMediaSelected'})
        self.assertEqual(unresolved, [])

    def test_constants_keywords_and_closures_resolve(self):
        found, _helpers, unresolved = self._extract('''
            def a(self):
                self._creator_emit(self.CLASS_SIG, {})
                self._creator_emit(MODULE_SIG)
                self._creator_emit(payload={}, signal_name="creatorResult")

            def b(self):
                sig = "comicStoryboardReady"
                run_later(lambda: self._creator_emit(sig, {}))
        ''')
        self.assertEqual(set(found), {'creatorProgress', 'creatorCacheEvent',
                                      'creatorResult', 'comicStoryboardReady'})
        self.assertEqual(unresolved, [])

    def test_wrappers_that_pass_their_parameter_through_become_helpers(self):
        found, helpers, unresolved = self._extract('''
            def _emit_later(self, sig, payload):
                run_later(lambda: self._creator_emit(sig, payload))

            def _emit_named(self, payload, *, sig):
                self._emit_later(sig, payload)

            def a(self):
                self._emit_later("creatorResult", {})
                self._emit_named({}, sig="xyzPlotEvent")
        ''')
        self.assertEqual(helpers['_emit_later'], (0, 'sig'))
        self.assertEqual(helpers['_emit_named'], (None, 'sig'))
        self.assertEqual(found['creatorResult'], [(found['creatorResult'][0][0], '_emit_later')])
        self.assertEqual(found['xyzPlotEvent'][0][1], '_emit_named')
        self.assertEqual(unresolved, [])

    def test_unreadable_names_are_reported_not_skipped(self):
        cases = {
            'call result': ('''
                def a(self):
                    self._creator_emit(self._pick(), {})
            ''', 'self._pick()'),
            'loop variable': ('''
                def a(self, names):
                    for name in names:
                        self._creator_emit(name, {})
            ''', 'name'),
            'reassigned local': ('''
                def a(self):
                    sig = "creatorResult"
                    sig = "creatorProgress"
                    self._creator_emit(sig)
            ''', 'sig'),
            'lambda parameter': ('''
                def a(self):
                    run_later(lambda sig: self._creator_emit(sig))
            ''', 'sig'),
            'f-string': ('''
                def a(self, kind):
                    self._creator_emit(f"creator{kind}")
            ''', 'creator'),
            'imported name': ('''
                def a(self):
                    from core.signals import CREATOR_SIG
                    self._creator_emit(CREATOR_SIG)
            ''', 'CREATOR_SIG'),
            'star args': ('''
                def a(self, args):
                    self._creator_emit(*args)
            ''', '*args'),
            'missing argument': ('''
                def a(self):
                    self._creator_emit(payload={})
            ''', '이름 인자 없음'),
            'non-literal getattr': ('''
                def a(self):
                    getattr(self.vue_bridge, self._pick(), None)
            ''', 'getattr'),
        }
        for label, (methods, needle) in cases.items():
            with self.subTest(label):
                self._assert_single_unresolved(methods, needle)

    def test_private_bridge_attributes_are_not_signals(self):
        found, _helpers, unresolved = self._extract('''
            def a(self):
                getattr(self.vue_bridge, _RELAY_ATTR, None)
                getattr(self.vue_bridge, "_proxies", {})
                getattr(self.vue_bridge, "creatorResult", None)
        ''')
        self.assertEqual(list(found), ['creatorResult'])
        self.assertEqual(found['creatorResult'][0][1], 'getattr')
        self.assertEqual(unresolved, [])

    def test_deeply_nested_elif_chains_do_not_hit_the_recursion_limit(self):
        branches = "".join(f"    elif action == 'a{i}':\n        pass\n" for i in range(1500))
        source = ("def _handle_vue_action(self, action):\n    if action == 'x':\n        pass\n"
                  + branches + "    self._creator_emit('creatorResult')\n")
        found, _helpers, unresolved = _name_dispatch_emits(
            {'ui/fake.py': _FAKE_EMIT_HELPER, 'ui/deep.py': source})
        self.assertEqual(list(found), ['creatorResult'])
        self.assertEqual(unresolved, [])


class PythonActionTableExtractionTests(unittest.TestCase):
    """_python_action_table 자체의 회귀 — 오인(false positive)도 누락(false negative)도 없게."""

    def _table(self, source):
        return _python_action_table({'ui/fake.py': textwrap.dedent(source)})

    def test_only_dispatcher_functions_count(self):
        names, unresolved = self._table('''
            def _on_backend_runtime_event(self, action):
                if action == 'start':
                    pass

            def apply_rules(rule):
                action = rule.get('action')
                if action == 'add':
                    pass

            def _handle_vue_action(self, action, payload):
                if action == 'generate':
                    pass
                elif 'swap_resolution' == action:
                    pass
        ''')
        self.assertEqual(set(names), {'generate', 'swap_resolution'})
        self.assertEqual(unresolved, [])

    def test_gates_tables_and_nested_workers_are_seen(self):
        names, unresolved = self._table('''
            _RELIGHT = ('relight_preview', 'relight_export')

            class Mixin:
                _CHAT = frozenset({'chat_send', 'chat_stop'})

                def _handle_x_action(self, action, payload):
                    handlers = {'creator_generate': self._g, 'creator_cancel': self._c}
                    if action not in handlers and action not in self._CHAT:
                        return False
                    if action in _RELIGHT:
                        pass

                    def work():
                        if action == 'hand_reconstruction_export':
                            pass
                    return work
        ''')
        self.assertEqual(set(names), {'creator_generate', 'creator_cancel', 'chat_send', 'chat_stop',
                                      'relight_preview', 'relight_export', 'hand_reconstruction_export'})
        self.assertEqual(unresolved, [])

    def test_shadowed_or_unknown_tables_are_reported_not_misread(self):
        """지역 이름이 같은 이름의 모듈 상수를 가리면 그 상수로 읽지 않는다 — unresolved 로 남긴다."""
        cases = {
            'reassigned local': '''
                def _handle_x_action(self, action):
                    handlers = {'a_one': 1}
                    handlers = {'a_two': 2}
                    if action in handlers:
                        pass
            ''',
            'shadowing call result': '''
                TABLE = ('module_only',)

                def _handle_x_action(self, action):
                    TABLE = self._build_table()
                    if action in TABLE:
                        pass
            ''',
            'shadowing loop variable': '''
                TABLE = ('module_only',)

                def _handle_x_action(self, action, tables):
                    for TABLE in tables:
                        if action in TABLE:
                            pass
            ''',
            'shadowing parameter': '''
                TABLE = ('module_only',)

                def _handle_x_action(self, action, TABLE):
                    if action in TABLE:
                        pass
            ''',
        }
        for label, source in cases.items():
            with self.subTest(label):
                names, unresolved = self._table(source)
                self.assertEqual(names, {})
                self.assertEqual(len(unresolved), 1, unresolved)
                self.assertTrue(unresolved[0].startswith('ui/fake.py:'), unresolved)


class FrontendActionExtractionTests(unittest.TestCase):
    def test_wrappers_and_ternaries_are_read_as_calls(self):
        text = textwrap.dedent('''
            requestAction('generate')
            action("save_ui_prefs", {})
            quickAction('add_favorite', img)
            requestAction(save ? 'comfy_compatibility_save_baseline' : 'comfy_compatibility_refresh', {})
            requestAction(name, payload)
            runAppUpdateAction('install')
            store.action('not_a_bridge_call')
        ''')
        self.assertEqual(_frontend_actions([text]), {
            'generate', 'save_ui_prefs', 'add_favorite',
            'comfy_compatibility_save_baseline', 'comfy_compatibility_refresh',
        })


if __name__ == "__main__":
    unittest.main()
