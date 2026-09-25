"""sam-extra(forge_sam3_extension) 계약 테스트 — 확장의 모든 노출면이 레지스트리에 분류돼 있는가.

세 겹으로 본다. 레지스트리는 `core/sam_extra_contract.py`.

1) 레지스트리 형식 (확장 없이도 돈다) — 항목마다 mapped/ignored/deferred 가 있고, mapped 의
   앱 참조(파일:심볼)가 실제로 있고, 앱 상수(제목·ARG_NAMES·기본값)가 레지스트리와 같다.
2) 픽스처 (확장 없이도 돈다) — `tests/fixtures/sam_extra_script_info.json`(라이브 GET
   /sdapi/v1/script-info 저장본, `tools/refresh_sam_extra_fixture.py`)과 비교: 스크립트 집합, 인자 수,
   인자 모양 해시, 앱 spec 의 기본값·범위·선택지, 의미 핀. AST 로는 못 읽는 실행 시점 값의 출처다.
3) 설치된 확장 (AST) — 제목, ui() 순서, ARG_NAMES, 위치 읽기 인덱스, 옵션 키, 라우트, XYZ 축,
   명령줄 플래그, Gradio 전용 함수, 모듈 파일, Sam3Args 필드·기본값·어노테이션 범위와 process() 가 읽는 키,
   의미 핀(모듈 상수·_BRIDGE_JS 메시지), 버전 게이트. 그리고 **소스 ↔ 픽스처**: ui() 컴포넌트의 라벨·기본값·
   범위·step·선택지를 AST 로 읽어 픽스처와 비교한다. 2) 는 픽스처만 보므로, 이 검사가 없으면 확장 소스의
   범위·기본값 변경이 Forge 를 다시 시작해 픽스처를 갱신할 때까지 통과한다.
   확장이 없으면 이유를 남기고 skip, AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패(`tests/_sam_extra_ext.py`).

실패 메시지의 '새 항목'은 레지스트리에 분류하고, '사라진 항목'은 앱 쪽 사용처를 정리한 뒤 지운다.
아직 없는 검사(gap matrix 5-(a) 5·6·7 일부)는 `core/sam_extra_contract.py` 모듈 설명에 적어 두었다.
절차: docs/sam_extra_sync.md
"""
from __future__ import annotations

import dataclasses
import importlib
import io
import json
import re
import tokenize
import unittest
from pathlib import Path

from core import sam_extra_contract as reg
from core.sam_extra_diff import (
    compare_defaults,
    compare_positional_spec,
    compare_ui_components,
    entries_by_title,
    same_value,
    shape_hash,
)
from core.sam_extra_scan import ExtensionSource, format_index_set, parse_index_set
from tests._sam_extra_ext import EXT_ROOT, requires_extension

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "sam_extra_script_info.json"
SYNC_DOC = "docs/sam_extra_sync.md"
_PACKAGE_RE = re.compile(r"^(P\d{1,2}|HOLD)$")


# ── 공용 도우미 ─────────────────────────────────────────────────────────────
def _set_mismatch(what: str, registered, found, optional=frozenset()) -> str:
    """양방향 비교 메시지. 차이가 없으면 ''."""
    new = sorted(set(found) - set(registered))
    removed = sorted(set(registered) - set(found) - set(optional))
    if not new and not removed:
        return ""
    lines = [f"\n[{what}] 레지스트리(core/sam_extra_contract.py)와 설치된 확장이 다릅니다."]
    if new:
        lines.append("  새 항목 — mapped / ignored(사유) / deferred(패키지·사유) 중 하나로 분류하라:")
        lines += [f"    + {item!r}" for item in new]
    if removed:
        lines.append("  사라진 항목 — 앱 쪽 사용처(UI·페이로드·저장값)를 정리하고 레지스트리에서 지워라:")
        lines += [f"    - {item!r}" for item in removed]
    lines.append(f"  절차: {SYNC_DOC}")
    return "\n".join(lines)


def _load(ref: str):
    """'pkg.module:ATTR' → 값."""
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)


def _python_identifiers(text: str) -> set[str]:
    """파이썬 소스의 식별자(NAME 토큰) — 주석·문자열·독스트링 속 단어는 빠진다."""
    return {tok.string for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type == tokenize.NAME}


def _ref_problem(ref: str) -> str:
    """'경로:심볼' 앱 참조가 실제로 있는지. 문제가 없으면 ''.

    .py 는 심볼이 코드 식별자로 있어야 한다(주석에만 남은 이름은 인정하지 않는다). .vue/.ts 등은 단어 검색.
    """
    path, _, symbol = ref.partition(":")
    target = REPO / path
    if not target.is_file():
        return f"{ref}: 파일이 없다"
    if symbol:
        text = target.read_text(encoding="utf-8")
        if target.suffix == ".py":
            found = symbol in _python_identifiers(text)
        else:
            found = re.search(rf"(?<![\w]){re.escape(symbol)}(?![\w])", text) is not None
        if not found:
            return f"{ref}: 파일에 '{symbol}' 이 (코드로) 없다"
    return ""


def _spec_names(title: str) -> list[str]:
    spec_ref, prefix = reg.SCRIPTS[title]["app_spec"]
    return [key[len(prefix):] for key, _k, _d, _e in _load(spec_ref)]


def _known(source: str, title: str | None = None, fields=None) -> dict:
    return {key: value for key, value in reg.KNOWN_DIFFS.items()
            if source in reg.diff_sources(value) and (title is None or key[0] == title)
            and (fields is None or key[2] in fields)}


def _diff_problems(observed: dict, known: dict) -> list[str]:
    """observed/known: {(script, key, field): (app, live)} / KNOWN_DIFFS 부분집합."""
    problems = []
    for key, (app, live) in sorted(observed.items(), key=str):
        entry = known.get(key)
        if entry is None:
            problems.append(f"  새 차이 {key}: 앱={app!r} 확장={live!r} — spec 을 고치거나 KNOWN_DIFFS 에 사유와 함께 적어라")
        elif not (same_value(entry["app"], app) and same_value(entry["live"], live)):
            problems.append(f"  값이 바뀐 차이 {key}: 기록 앱={entry['app']!r}/확장={entry['live']!r}, "
                            f"지금 앱={app!r}/확장={live!r}")
    for key in sorted(set(known) - set(observed), key=str):
        problems.append(f"  더는 차이가 없다 {key} ({known[key].get('package') or known[key]['kind']}) — "
                        "KNOWN_DIFFS 에서 지워라")
    return problems


def _arg_table(args) -> str:
    rows = []
    for index, arg in enumerate(args):
        rows.append(f"    [{index}] {arg.get('label')!r} value={arg.get('value')!r} "
                    f"min={arg.get('minimum')!r} max={arg.get('maximum')!r} step={arg.get('step')!r} "
                    f"choices={arg.get('choices')!r}")
    return "\n".join(rows)


# ── 1) 레지스트리 형식 ─────────────────────────────────────────────────────────
class TestRegistry(unittest.TestCase):
    def test_every_entry_is_classified(self):
        problems = []
        for section, table in reg.classified_sections().items():
            for key, entry in table.items():
                where = f"{section}[{key!r}]"
                status = entry.get("status")
                if status not in reg.STATUSES:
                    problems.append(f"{where}: status={status!r}")
                elif status == reg.MAPPED:
                    if not entry.get("app") or not all(isinstance(r, str) and r for r in entry["app"]):
                        problems.append(f"{where}: mapped 인데 앱 참조(app)가 없다")
                elif status == reg.IGNORED:
                    if not str(entry.get("reason") or "").strip():
                        problems.append(f"{where}: ignored 인데 사유가 없다")
                    if entry.get("package") and not _PACKAGE_RE.match(entry["package"]):
                        problems.append(f"{where}: package={entry['package']!r}")
                elif status == reg.DEFERRED:
                    if not _PACKAGE_RE.match(str(entry.get("package") or "")):
                        problems.append(f"{where}: deferred 는 작업 패키지 id(P1-P21 또는 HOLD)가 필요하다")
                    if not str(entry.get("reason") or "").strip():
                        problems.append(f"{where}: deferred 인데 사유가 없다")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_mapped_app_references_exist(self):
        problems = []
        for section, table in reg.classified_sections().items():
            for key, entry in table.items():
                if entry.get("status") != reg.MAPPED:
                    continue
                for ref in (*entry["app"], *entry.get("comfy", ())):
                    problem = _ref_problem(ref)
                    if problem and not entry.get("optional"):
                        problems.append(f"{section}[{key!r}] {problem}")
        self.assertEqual(problems, [], "\n앱 코드를 옮겼으면 레지스트리의 app/comfy 참조도 고쳐라:\n"
                         + "\n".join(problems))

    def test_python_reference_needs_code_identifier(self):
        """주석·문자열·독스트링에만 남은 이름은 앱 참조로 인정하지 않는다."""
        text = "# put_memo 는 주석\n'''put_memo 독스트링'''\nPATH = 'put_memo'\n\ndef list_memos():\n    pass\n"
        names = _python_identifiers(text)
        self.assertIn("list_memos", names)
        self.assertNotIn("put_memo", names)

    def test_app_title_constants_match(self):
        for title, entry in reg.SCRIPTS.items():
            if entry["app_title"]:
                with self.subTest(script=title):
                    self.assertEqual(_load(entry["app_title"]), title)

    def test_app_positional_specs_have_live_arg_count(self):
        for title, entry in reg.SCRIPTS.items():
            if entry["app_spec"]:
                with self.subTest(script=title):
                    self.assertEqual(len(_load(entry["app_spec"][0])), entry["live_argc"])

    def test_app_arg_names_match_registry(self):
        for title, entry in reg.SCRIPTS.items():
            if entry["app_arg_names"]:
                with self.subTest(script=title):
                    self.assertEqual(tuple(_load(entry["app_arg_names"])), entry["arg_names"])

    def test_registered_ui_returns_have_live_arg_count(self):
        for title, entry in reg.SCRIPTS.items():
            if isinstance(entry["ui_return"], tuple):
                with self.subTest(script=title):
                    self.assertEqual(len(entry["ui_return"]), entry["live_argc"])

    def test_anima38_app_defaults_follow_pinned_extension_defaults(self):
        from core.anima38 import DEFAULT_SETTINGS
        pin = next(p for p in reg.SEMANTIC_PINS if p["id"] == "anima38_arg_defaults")
        self.assertEqual(dataclasses.asdict(DEFAULT_SETTINGS), pin["expected"])

    def test_app_smc_preset_choices_follow_pinned_extension_names(self):
        from core.anima_guidance import PERTURBATION_SPEC
        pin = next(p for p in reg.SEMANTIC_PINS if p["id"] == "pag_smc_presets")
        app = next(extra for key, _k, _d, extra in PERTURBATION_SPEC if key == "guid_smc_preset")
        self.assertEqual(tuple(app), tuple(pin["expected"][1:]))

    def test_known_diffs_are_well_formed(self):
        for (title, key, field), entry in reg.KNOWN_DIFFS.items():
            with self.subTest(diff=(title, key, field)):
                self.assertIn(title, reg.SCRIPTS)
                self.assertIn(entry["kind"], ("intended", "gap"))
                sources = reg.diff_sources(entry)
                self.assertTrue(sources and set(sources) <= {"fixture", "ast"}, sources)
                self.assertTrue(entry["reason"].strip())
                if entry["kind"] == "gap":
                    self.assertRegex(entry["package"], _PACKAGE_RE)

    def test_semantic_pins_are_well_formed(self):
        ids = [pin["id"] for pin in reg.SEMANTIC_PINS]
        self.assertEqual(len(ids), len(set(ids)))
        for pin in reg.SEMANTIC_PINS:
            with self.subTest(pin=pin["id"]):
                self.assertTrue(pin["meaning"].strip())
                if pin["source"] == "fixture":
                    self.assertIn(pin["script"], reg.SCRIPTS)
                    self.assertIn("expected", pin)
                else:
                    self.assertEqual(pin["source"], "ast")
                    self.assertTrue(pin["file"] and pin["name"])
                    self.assertNotEqual("expected" in pin, "patterns" in pin, "expected 와 patterns 중 하나만")
                    for pattern in pin.get("patterns", ()):
                        re.compile(pattern)
                if "app" in pin:
                    self.assertIn("expected", pin)

    def test_app_constants_follow_pinned_extension_constants(self):
        """공유 계약 값(메모 스키마·한도 등)을 앱이 하드코딩한 곳 — 확장 핀과 같아야 한다."""
        for pin in (p for p in reg.SEMANTIC_PINS if "app" in p):
            with self.subTest(pin=pin["id"]):
                self.assertEqual(_load(pin["app"]), pin["expected"],
                                 f"\n{pin['app']} ≠ 확장 {pin['file']}:{pin['name']} 핀 {pin['expected']!r}.\n"
                                 f"  뜻: {pin['meaning']}")

    def test_ui_unread_is_well_formed(self):
        for (title, index), entry in reg.UI_UNREAD.items():
            with self.subTest(cell=(title, index)):
                self.assertIn(title, reg.SCRIPTS)
                self.assertIsInstance(index, int)
                self.assertTrue(entry["fields"] and set(entry["fields"]) <= {
                    "label", "value", "minimum", "maximum", "step", "choices"})
                self.assertTrue(entry["reason"].strip())

    def test_xyz_labels_are_unique_and_prefixed(self):
        for prefix, group in reg.XYZ_AXES.items():
            with self.subTest(prefix=prefix):
                self.assertRegex(prefix, r"^\[[^\]]+\]$")
                self.assertEqual(len(group["labels"]), len(set(group["labels"])))

    def test_index_set_roundtrip(self):
        self.assertEqual(format_index_set({0, 1, 2, 4, 6, 7}), "0-2,4,6-7")
        self.assertEqual(parse_index_set("0-2,4,6-7"), frozenset({0, 1, 2, 4, 6, 7}))
        self.assertEqual(parse_index_set(""), frozenset())
        for entry in reg.SCRIPTS.values():
            if entry["api_reads"]:
                self.assertEqual(format_index_set(parse_index_set(entry["api_reads"])), entry["api_reads"])


# ── 2) 라이브 script-info 픽스처 ─────────────────────────────────────────────────
class TestScriptInfoFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as fh:
            cls.fixture = json.load(fh)
        cls.by_title = entries_by_title(cls.fixture.get("scripts") or [])

    def _modes(self, title: str) -> dict:
        modes = self.by_title.get(title.lower())
        self.assertIsNotNone(modes, f"픽스처에 '{title}' 이 없다 — tools/refresh_sam_extra_fixture.py")
        return modes

    def test_fixture_is_for_audited_version(self):
        version = self.fixture.get("ext_version")
        if version is None:
            self.skipTest("픽스처에 로컬 확장 버전이 없다(확장이 없는 PC 에서 받음)")
        self.assertEqual(version, reg.EXT_VERSION_AUDITED,
                         f"\n픽스처는 확장 {version} 에서 받았고 레지스트리는 {reg.EXT_VERSION_AUDITED} 기준이다. "
                         f"{SYNC_DOC} 의 순서대로 둘을 맞춰라.")

    def test_fixture_scripts_match_registry(self):
        registered = {title.lower() for title in reg.SCRIPTS}
        message = _set_mismatch("script-info 스크립트(소문자 제목)", registered, set(self.by_title))
        self.assertFalse(message, message)

    def test_live_arg_counts(self):
        for title, entry in reg.SCRIPTS.items():
            with self.subTest(script=title):
                modes = self._modes(title)
                self.assertEqual(set(modes), {False, True}, "txt2img·img2img 항목이 모두 있어야 한다")
                for is_img2img, args in modes.items():
                    self.assertEqual(
                        len(args), entry["live_argc"],
                        f"\n'{title}' ({'img2img' if is_img2img else 'txt2img'}) 인자 수가 바뀌었다: "
                        f"{entry['live_argc']} → {len(args)}. {entry.get('api_note', '')}\n{_arg_table(args)}")

    def test_arg_shapes(self):
        for title, entry in reg.SCRIPTS.items():
            with self.subTest(script=title):
                for is_img2img, args in self._modes(title).items():
                    current = shape_hash(args, entry["runtime_choices"])
                    self.assertEqual(
                        current, entry["shape"],
                        f"\n'{title}' ({'img2img' if is_img2img else 'txt2img'}) 인자의 라벨·기본값·범위·선택지가 "
                        f"바뀌었다. git diff tests/fixtures/sam_extra_script_info.json 과 확장 CHANGELOG 로 "
                        f"의미를 확인하고, 앱 spec·KNOWN_DIFFS·SEMANTIC_PINS 를 고친 뒤 shape 를 "
                        f"'{current}' 로 바꿔라.\n{_arg_table(args)}")

    def test_app_positional_specs_match_live_values(self):
        observed = {}
        for title, entry in reg.SCRIPTS.items():
            if not entry["app_spec"]:
                continue
            spec_ref, prefix = entry["app_spec"]
            for args in self._modes(title).values():
                for key, field, app, live in compare_positional_spec(
                        _load(spec_ref), args, prefix=prefix, runtime_indices=entry["runtime_choices"]):
                    observed[(title, key, field)] = (app, live)
        known = {k: v for k, v in _known("fixture").items() if reg.SCRIPTS[k[0]]["app_spec"]}
        problems = _diff_problems(observed, known)
        self.assertEqual(problems, [], "\n앱 spec 과 라이브 기본값·범위·선택지:\n" + "\n".join(problems))

    def test_sam3_live_state_matches_app_spec(self):
        from core.sam3_args import SAM3_KEYS, default_settings
        title = "SAM3 Mask"
        observed = {}
        for is_img2img, args in self._modes(title).items():
            with self.subTest(img2img=is_img2img):
                self.assertEqual(args[0].get("label"), reg.SAM3_ENABLE_LABEL)
                state = args[1].get("value")
                self.assertIsInstance(state, dict, "args[1].value 가 state dict 여야 한다(dict API)")
                self.assertEqual(len(state), reg.SAM3_LIVE_STATE_KEYS)
                self.assertEqual(set(state) - {"sam3_enable"}, set(SAM3_KEYS),
                                 "script-info 기본 state 키 ≠ core/sam3_args.py SAM3_KEYS")
                for key, field, app, live in compare_defaults(default_settings(), state, skip=("sam3_enable",)):
                    observed[(title, key, field)] = (app, live)
        problems = _diff_problems(observed, _known("fixture", title))
        self.assertEqual(problems, [], "\nSAM3 기본 state:\n" + "\n".join(problems))

    def test_anima38_live_defaults_match_app(self):
        from core.anima38 import ARG_NAMES, DEFAULT_SETTINGS
        entry = reg.SCRIPTS["Anima 3.8B (Qwen3.5 / v2)"]
        app = dataclasses.asdict(DEFAULT_SETTINGS)
        for args in self._modes("Anima 3.8B (Qwen3.5 / v2)").values():
            for index, name in enumerate(ARG_NAMES):
                if index in entry["runtime_choices"]:
                    continue
                self.assertTrue(same_value(app[name], args[index].get("value")),
                                f"{name}: 앱 {app[name]!r} / 확장 {args[index].get('value')!r}")

    def test_semantic_pins_on_live_values(self):
        for pin in (p for p in reg.SEMANTIC_PINS if p["source"] == "fixture"):
            with self.subTest(pin=pin["id"]):
                for args in self._modes(pin["script"]).values():
                    value = args[pin["index"]].get(pin["field"])
                    self.assertTrue(same_value(pin["expected"], value),
                                    f"\n의미 핀 {pin['id']}: 기대 {pin['expected']!r}, 라이브 {value!r}.\n"
                                    f"  뜻: {pin['meaning']}")


# ── 3) 설치된 확장 (AST) ────────────────────────────────────────────────────────
@requires_extension
class TestInstalledExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = ExtensionSource(EXT_ROOT)
        cls.src = src
        cls.version = src.version()
        cls.scripts = src.scripts()
        cls.options = src.option_keys()
        cls.routes = src.routes()
        cls.axes = src.axis_labels()
        cls.cli = src.cli_flags()
        cls.units = src.module_units()
        cls.functions = src.function_names()
        cls.sam3args = src.sam3args()
        sam3_file = reg.SCRIPTS["SAM3 Mask"]["file"]
        cls.payload_keys = src.dict_keys_assigned(sam3_file, "process", "payload")
        cls.state_reads = src.state_reads(sam3_file, "process", "state", ("_xyz_or", "_bool_or"))
        cls.unresolved = list(src.unresolved)

    def _assert_sets(self, what, registered, found, optional=frozenset()):
        message = _set_mismatch(what, registered, found, optional)
        self.assertFalse(message, message)

    def test_everything_was_resolved_statically(self):
        self.assertEqual(self.unresolved, [],
                         "\n확장 소스에서 정적으로 풀지 못한 항목(새 패턴일 수 있다 — core/sam_extra_scan.py 를 "
                         "보강하거나 레지스트리에 분류하라):\n  " + "\n  ".join(self.unresolved))

    def test_version_gate(self):
        if self.version == reg.EXT_VERSION_AUDITED:
            return
        newer = []
        for version, heading in self.src.changelog_versions():
            if version == reg.EXT_VERSION_AUDITED:
                break
            newer.append(f"  v{version} — {heading}")
        self.fail(f"\n설치된 확장 {self.version} ≠ 레지스트리 기준 {reg.EXT_VERSION_AUDITED}. "
                  f"인자 모양이 같아도 의미가 바뀌었을 수 있다(예: Detail Daemon ×0.1). 아래 CHANGELOG 를 읽고 "
                  f"{SYNC_DOC} 대로 확인한 뒤 EXT_VERSION_AUDITED 를 올려라:\n" + "\n".join(newer or ["  (CHANGELOG 없음)"]))

    def test_script_titles(self):
        self._assert_sets("always-on 스크립트 제목", reg.SCRIPTS, self.scripts)

    def test_script_files(self):
        for title, info in self.scripts.items():
            if title in reg.SCRIPTS:
                with self.subTest(script=title):
                    self.assertEqual(info.file, reg.SCRIPTS[title]["file"])

    def test_ui_return_order(self):
        for title, entry in reg.SCRIPTS.items():
            info = self.scripts.get(title)
            if info is None:
                continue
            expected = _spec_names(title) if entry["ui_return"] == "app_spec" else entry["ui_return"]
            expected = None if expected is None else list(expected)
            # 모드마다 따로 — ui() 가 is_img2img 로 갈라 한 모드에서만 인자를 더하거나 빼도 잡는다
            for is_img2img, names in sorted(info.ui_returns.items()):
                mode = "img2img" if is_img2img else "txt2img"
                with self.subTest(script=title, mode=mode):
                    live = None if names is None else list(names)
                    self.assertEqual(
                        live, expected,
                        f"\n{info.file} 의 ui() 반환 순서가 바뀌었다({mode} 위치 인자 계약).\n"
                        f"확장: {live}\n기대: {expected}")

    def test_arg_names(self):
        for title, entry in reg.SCRIPTS.items():
            info = self.scripts.get(title)
            if info is not None:
                with self.subTest(script=title):
                    self.assertEqual(info.arg_names, entry["arg_names"],
                                     f"\n{info.file} 의 ARG_NAMES 가 레지스트리와 다르다(dict API 키 계약)")

    def test_positional_reads(self):
        for title, entry in reg.SCRIPTS.items():
            info = self.scripts.get(title)
            if info is None:
                continue
            expected = frozenset() if entry["api_reads"] is None else parse_index_set(entry["api_reads"])
            with self.subTest(script=title):
                self.assertEqual(
                    format_index_set(info.api_reads), format_index_set(expected),
                    f"\n{info.file} 가 API 경로에서 위치로 읽는 인덱스가 바뀌었다. {entry.get('api_note', '')}")

    def test_option_keys(self):
        self._assert_sets("Forge 옵션 키 (shared.opts.add_option)", reg.OPTIONS, self.options)

    def test_routes(self):
        self._assert_sets("FastAPI 라우트", reg.ROUTES, self.routes, reg.optional_keys(reg.ROUTES))

    def test_xyz_axis_labels(self):
        self._assert_sets("XYZ 축 라벨", reg.axis_label_map(), self.axes)

    def test_cli_flags(self):
        self._assert_sets("preload.py 명령줄 플래그", reg.CLI_FLAGS, self.cli)

    def test_module_units(self):
        self._assert_sets("모듈 파일 단위", reg.MODULES, self.units, reg.optional_keys(reg.MODULES))

    def test_ui_only_endpoint_functions_exist(self):
        """Gradio 이름 엔드포인트 = 함수 이름. 이름이 바뀌면 조용히 깨지므로 등록된 함수가 그 파일에 있어야 한다."""
        problems = []
        for endpoint, entry in reg.UI_ONLY_FEATURES.items():
            files = self.functions.get(endpoint.lstrip("/"), set())
            if entry["file"] not in files:
                problems.append(f"  {endpoint}: {entry['file']} 에 함수 '{endpoint.lstrip('/')}' 가 없다 "
                                f"(지금 정의된 곳: {sorted(files) or '없음'})")
        self.assertEqual(problems, [], "\nGradio 전용 기능(UI_ONLY_FEATURES):\n" + "\n".join(problems))

    def test_sam3args_fields_match_app_spec(self):
        from core.sam3_args import SAM3_KEYS
        self._assert_sets("Sam3Args 필드 ↔ core/sam3_args.py SAM3_KEYS", SAM3_KEYS, self.sam3args["fields"])

    def test_sam3args_literal_choices_match_app_spec(self):
        from core.sam3_args import SAM3_SPEC
        app = {key: extra for key, kind, _d, extra in SAM3_SPEC if kind == "choice"}
        self.assertEqual(set(app), set(self.sam3args["literals"]),
                         "Literal 필드 집합 ≠ 앱 choice 필드 집합")
        for key, choices in self.sam3args["literals"].items():
            with self.subTest(field=key):
                self.assertEqual(set(app[key]), set(choices),
                                 f"{key}: 새 선택지는 앱에서 조용히 기본값으로 되돌아간다")

    def test_sam3args_numeric_bounds_match_app_spec(self):
        from core.sam3_args import _INT_MAX, SAM3_SPEC
        app = {key: extra for key, kind, _d, extra in SAM3_SPEC if kind in ("float", "int")}
        observed = {}
        for key, (low, high, _integer) in self.sam3args["numeric_bounds"].items():
            if key not in app:
                observed[("SAM3 Mask", key, "missing_app")] = (None, (low, high))
                continue
            app_low, app_high = app[key]
            app_high = None if app_high == _INT_MAX else app_high
            if not same_value(app_low, low):
                observed[("SAM3 Mask", key, "minimum")] = (app_low, low)
            if not (app_high is None and high is None) and not same_value(app_high, high):
                observed[("SAM3 Mask", key, "maximum")] = (app_high, high)
        problems = _diff_problems(observed, _known("ast", "SAM3 Mask", ("minimum", "maximum", "missing_app")))
        self.assertEqual(problems, [], "\nSam3Args._NUMERIC_BOUNDS 와 앱 범위:\n" + "\n".join(problems))

    def test_sam3args_constraints_match_clamp_table_and_accept_app_range(self):
        """pydantic 제약(confloat/NonNegativeInt …)이 실제로 거부하는 범위.

        _NUMERIC_BOUNDS 는 그 앞에서 값을 잘라 주는 표다. 한쪽만 바뀌면(예: confloat le=1.0 → 0.9) 잘린 값이
        다시 거부돼 그 생성에서 SAM3 가 꺼진다. 앱 범위는 제약 안에 있어야 한다.
        """
        from core.sam3_args import _INT_MAX, SAM3_SPEC
        app = {key: extra for key, kind, _d, extra in SAM3_SPEC if kind in ("float", "int")}
        constraints = self.sam3args["constraints"]
        bounds = self.sam3args["numeric_bounds"]
        problems = []
        for key in sorted(set(bounds) - set(constraints)):
            problems.append(f"  {key}: _NUMERIC_BOUNDS 에는 있는데 어노테이션에 숫자 제약이 없다")
        for key, (low, high) in sorted(constraints.items()):
            if key in bounds:
                clamp_low, clamp_high, _integer = bounds[key]
                if not (same_value(low, clamp_low) and (high is clamp_high is None or same_value(high, clamp_high))):
                    problems.append(f"  {key}: 어노테이션 ({low}, {high}) ≠ _NUMERIC_BOUNDS ({clamp_low}, {clamp_high})")
            if key not in app:
                problems.append(f"  {key}: 앱 SAM3_SPEC 에 숫자 범위가 없다")
                continue
            app_low, app_high = app[key]
            app_high = None if app_high == _INT_MAX else app_high
            if low is not None and app_low < low:
                problems.append(f"  {key}: 앱 하한 {app_low} < 확장 제약 {low}")
            if high is not None and (app_high is None or app_high > high):
                problems.append(f"  {key}: 앱 상한 {app_high} > 확장 제약 {high}")
        self.assertEqual(problems, [], "\nSam3Args 어노테이션 범위:\n" + "\n".join(problems))

    def test_sam3args_defaults_match_app_defaults(self):
        """Sam3Args 기본값 = 앱 default_settings(). 필드 집합은 test_sam3args_fields_match_app_spec 가 본다."""
        from core.sam3_args import default_settings
        observed = {("SAM3 Mask", key, field): (app, live)
                    for key, field, app, live in compare_defaults(default_settings(), self.sam3args["defaults"])
                    if field == "default"}
        problems = _diff_problems(observed, _known("ast", "SAM3 Mask", ("default",)))
        self.assertEqual(problems, [], "\nSam3Args 기본값과 앱 기본값:\n" + "\n".join(problems))

    def test_sam3_process_reads_every_key_the_app_sends(self):
        """나-4: process() 는 정해진 키만 골라 Sam3Args 에 넘긴다 — 확장이 키 이름을 바꾸면 앱 값이 조용히 버려진다."""
        from core.sam3_args import build_state
        sent = set(build_state({}))
        dropped = sorted(sent - self.state_reads)
        unsent = sorted(self.state_reads - sent)
        self.assertEqual(dropped, [], f"앱이 보내지만 process() 가 읽지 않는 키(조용히 버려짐): {dropped}")
        self.assertEqual(unsent, [], f"process() 가 state 에서 읽지만 앱이 보내지 않는 키: {unsent}")
        self.assertEqual(set(self.payload_keys), set(self.sam3args["fields"]),
                         "process() 가 Sam3Args 에 넘기는 키 ≠ Sam3Args 필드")
        self.assertEqual(sorted(self.state_reads - set(self.payload_keys)), sorted(reg.SAM3_ACTIVATION_KEYS),
                         "Sam3Args 밖에서 읽는 활성화 플래그가 바뀌었다")

    def test_semantic_pins_on_source_constants(self):
        for pin in (p for p in reg.SEMANTIC_PINS if p["source"] == "ast"):
            with self.subTest(pin=pin["id"]):
                try:
                    value = self.src.module_constant(pin["file"], pin["name"])
                except KeyError as exc:
                    self.fail(f"의미 핀 {pin['id']}: {exc}")
                if "patterns" in pin:
                    self.assertIsInstance(value, str)
                    missing = [p for p in pin["patterns"] if not re.search(p, value)]
                    self.assertEqual(missing, [],
                                     f"\n의미 핀 {pin['id']} ({pin['file']}:{pin['name']}) 에서 사라진 모양:\n  "
                                     + "\n  ".join(missing) + f"\n  뜻: {pin['meaning']}")
                    continue
                self.assertEqual(value, pin["expected"],
                                 f"\n의미 핀 {pin['id']} ({pin['file']}:{pin['name']}) 가 바뀌었다.\n  뜻: {pin['meaning']}")

    @staticmethod
    def _fixture_by_title() -> dict:
        with open(FIXTURE, encoding="utf-8") as fh:
            return entries_by_title(json.load(fh).get("scripts") or [])

    def test_fixture_matches_installed_source(self):
        """픽스처 = 설치된 소스의 ui(): 인자 수와 컴포넌트 라벨·기본값·범위·step·선택지(AST).

        다르면 확장 소스가 바뀌었는데 픽스처가 옛 값이다(또는 Forge 가 옛 코드를 띄웠다). 픽스처 기준 테스트
        (shape 해시·앱 spec 비교·의미 핀)는 픽스처만 보므로 이 검사가 소스 변경을 --quick 에서 잡는다.
        """
        by_title = self._fixture_by_title()
        refresh = ("Forge 를 다시 시작한 뒤 tools/refresh_sam_extra_fixture.py 로 픽스처를 갱신하라. 그러면 "
                   "shape 해시·앱 spec 비교(test_arg_shapes 등)가 앱에서 무엇을 고칠지 알려 준다")
        problems = []
        for title, info in sorted(self.scripts.items()):
            if title.lower() not in by_title or title not in reg.SCRIPTS:
                continue
            runtime = reg.SCRIPTS[title]["runtime_choices"]
            for is_img2img, args in sorted(by_title[title.lower()].items()):
                mode = "img2img" if is_img2img else "txt2img"
                names = info.ui_returns.get(bool(is_img2img))   # 픽스처와 같은 모드의 ui() 반환
                if names is None:
                    continue   # 동적 반환 — 순서·모호함은 test_ui_return_order·test_everything_was_resolved_statically
                if len(args) != len(names):
                    problems.append(f"  {title} ({mode}): 인자 수 소스 {len(names)} / 픽스처 {len(args)}")
                    continue
                components = info.ui_components_by_mode[bool(is_img2img)]
                diffs, _unread = compare_ui_components(components, args, runtime_indices=runtime)
                for index, field, source, live in diffs:
                    problems.append(f"  {title} ({mode}) [{index}] {names[index]}.{field}: "
                                    f"소스 {source!r} / 픽스처 {live!r}")
        self.assertEqual(problems, [], f"\n설치된 확장 소스가 픽스처와 다르다 — {refresh}.\n" + "\n".join(problems))

    def test_ui_components_are_read_statically(self):
        """픽스처에 값이 있는 칸은 AST 로도 읽혀야 비교된다. 못 읽는 칸은 UI_UNREAD 에 사유와 함께 적는다."""
        by_title = self._fixture_by_title()
        observed: dict = {}
        for title, info in self.scripts.items():
            if title.lower() not in by_title or title not in reg.SCRIPTS:
                continue
            for is_img2img, args in by_title[title.lower()].items():
                components = info.ui_components_by_mode.get(bool(is_img2img))
                if components is None or len(args) != len(components):
                    continue   # 인자 수는 test_fixture_matches_installed_source 가 본다
                _diffs, unread = compare_ui_components(
                    components, args, runtime_indices=reg.SCRIPTS[title]["runtime_choices"])
                for index, fields in unread.items():
                    observed.setdefault((title, index), set()).update(fields)
        problems = []
        for cell in sorted(set(observed) | set(reg.UI_UNREAD), key=str):
            registered = set(reg.UI_UNREAD.get(cell, {}).get("fields", ()))
            found = observed.get(cell, set())
            if found != registered:
                problems.append(f"  {cell}: 못 읽는 필드 지금 {sorted(found)} / UI_UNREAD {sorted(registered)}")
        self.assertEqual(problems, [],
                         "\nui() 컴포넌트를 AST 로 읽지 못하는 칸이 바뀌었다 — core/sam_extra_scan.py 를 보강하거나 "
                         "UI_UNREAD 를 사유와 함께 고쳐라(이제 읽히면 지운다):\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
