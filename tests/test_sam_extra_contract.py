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

import ast
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


def _state_reads_by_constant(src: ExtensionSource, rel_file: str, function: str, state_var: str) -> frozenset:
    """``function`` 안에서 ``state_var.get(모듈 상수)`` 로 읽는 키 — ``ExtensionSource.state_reads`` 는 문자열
    리터럴만 본다. 요청 전용 키(SAM3_REQUEST_ONLY_KEYS)는 확장이 상수(SOURCE_STATE_KEY)로 읽는다."""
    tree = src.tree(src.root / rel_file)
    keys: set[str] = set()
    for fn in (n for n in ast.walk(tree or ast.Module(body=[], type_ignores=[]))
               if isinstance(n, ast.FunctionDef) and n.name == function):
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Name)):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "get"
                    and isinstance(func.value, ast.Name) and func.value.id == state_var):
                try:
                    value = src.module_constant(rel_file, node.args[0].id)
                except KeyError:
                    continue   # 지역 이름(예: _xyz_or 의 state_key 인자) — 문자열 리터럴 읽기는 state_reads 가 본다
                if isinstance(value, str):
                    keys.add(value)
    return frozenset(keys)


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
            where = (f"{pin['file']}:{pin['name']}" if pin["source"] == "ast"
                     else f"script-info '{pin['script']}'[{pin['index']}].{pin['field']}")
            with self.subTest(pin=pin["id"]):
                self.assertEqual(_load(pin["app"]), pin["expected"],
                                 f"\n{pin['app']} ≠ 확장 {where} 핀 {pin['expected']!r}.\n"
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


    def test_degrid_is_mapped_for_main_requests_with_comfy_parity(self):
        """VAE DeGrid 앱 노출(보류 → mapped): 스크립트·옵션 셋·모듈 6개가 모두 분류됐고 DEFERRED 로 남은 것이 없다.
        Extras 판은 API 가 없어 ignored(HOLD 사유). 앱 제목 상수·Comfy 노드·컴파일러 참조가 레지스트리에 있다."""
        from core import vae_degrid as vdg
        entry = reg.SCRIPTS[vdg.SCRIPT_NAME]
        self.assertEqual(entry["status"], reg.MAPPED)
        self.assertEqual(entry["app_title"], "core.vae_degrid:SCRIPT_NAME")
        self.assertEqual((entry["form"], entry["live_argc"], entry["runtime_choices"]), ("positional_or_dict", 5, (1,)))
        self.assertIsNone(entry["arg_names"], "ARG_NAMES 는 스크립트 모듈 밖(sam3ext/ui_vae_degrid.py) — 핀 degrid_arg_names")
        for ref in ("ui/vae_degrid_ui.py:contribute", "ui/sampling_blocks.py:CONTRIBUTORS",
                    "core/alwayson_propagation.py:TITLE_DEGRID", "core/sam_extra_notices.py:CODE_DEGRID_ERROR",
                    "frontend/src/components/params/VaeDegridCard.vue"):
            self.assertIn(ref, entry["app"])
        self.assertIn(f"comfy_custom_nodes/ai_studio_forge_parity/degrid_nodes.py:{vdg.COMFY_NODE_CLASS}", entry["comfy"])
        self.assertIn("core/comfy_workflow_compiler.py:_add_degrid", entry["comfy"])
        self.assertIn("ui/vae_degrid_ui.py:push_comfy_models", entry["comfy"])
        self.assertTrue(any(gap.startswith("P16") and "from_infotext" in gap for gap in entry["gaps"]))
        degrid_rows = {f"{section}[{key}]": row for section, table in reg.classified_sections().items()
                       for key, row in table.items() if "degrid" in key.lower()}
        self.assertEqual(len(degrid_rows), 1 + 3 + 6, sorted(degrid_rows))
        self.assertEqual(sorted(k for k, row in degrid_rows.items() if row["status"] == reg.DEFERRED), [])
        extras = reg.MODULES["scripts/anima_vae_degrid_extras.py"]
        self.assertEqual((extras["status"], extras["package"]), (reg.IGNORED, reg.HOLD))
        self.assertNotIn(vdg.EXTRAS_TITLE, reg.SCRIPTS, "Extras 판은 always-on 스크립트가 아니다(ScriptPostprocessing)")

    def test_degrid_pins_mirror_every_shared_constant(self):
        """앱과 Comfy 팩이 확장 값을 하드코딩한 곳마다 핀이 있다 — 앱 상수(core.vae_degrid·P10 옵션 키)와 팩 상수
        (degrid_math·degrid_nodes·degrid_runner)가 핀 하나씩 갖는다. 핀이 빠지면 확장이 바뀌어도 한쪽만 조용히 어긋난다."""
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS if pin["id"].startswith("degrid_")}
        apps = {pin["app"] for pin in pins.values() if "app" in pin}
        core = ("SCRIPT_NAME", "EXTRAS_TITLE", "ARG_NAMES", "MODE_CHOICES", "MODES", "MODE_LABELS", "DEFAULT_MODE",
                "DEFAULT_STRENGTH", "STRENGTH_MIN", "STRENGTH_MAX", "STRENGTH_STEP", "DEFAULT_TILE", "TILE_OVERLAP",
                "MIN_TILE", "MAX_TILE", "TILE_STEP", "NONE_NAME", "MODEL_FOLDERS", "KEY_MODEL", "KEY_MODE",
                "KEY_STRENGTH", "KEY_TILE", "KEY_PRECISION", "KEY_ERROR", "DEVICE_AUTO", "DEVICE_CPU",
                "PRECISION_FP32", "PRECISION_FP16", "DEFAULT_DEVICE", "DEFAULT_PRECISION", "DEFAULT_KEEP_LOADED")
        pack = {"degrid_math": ("MODES", "MODE_LABELS", "DEFAULT_MODE", "DEFAULT_STRENGTH", "STRENGTH_MIN",
                                "STRENGTH_MAX", "DEFAULT_TILE", "TILE_OVERLAP", "MIN_TILE", "MAX_TILE", "PAD_MULTIPLE",
                                "NAFNET_KEYS", "MODEL_EXTENSIONS", "IMAGE_LIKE_MIN_ABS_MEAN", "IMAGE_LIKE_CORRELATION",
                                "IMAGE_LIKE_ABS_MEAN", "IMAGE_LIKE_LARGE_CORRELATION", "IMAGE_LIKE_DC_MEAN",
                                "IMAGE_LIKE_DC_SIGN", "IMAGE_LIKE_DC_RATIO", "RESIDUAL_BLOWUP_ABS_MEAN"),
                "degrid_nodes": ("DEVICE_AUTO", "DEVICE_CPU", "BYTES_PER_PIXEL"),
                "degrid_runner": ("PRECISION_FP32", "PRECISION_FP16")}
        wanted = {f"core.vae_degrid:{name}" for name in core}
        wanted |= {f"core.forge_override_settings:{name}" for name in (
            "OPT_DEGRID_DEVICE", "OPT_DEGRID_GPU_PRECISION", "OPT_DEGRID_KEEP_LOADED")}
        wanted |= {f"comfy_custom_nodes.ai_studio_forge_parity.{module}:{name}"
                   for module, names in pack.items() for name in names}
        self.assertEqual(sorted(wanted - apps), [], "핀이 없는 앱·팩 상수")
        # 확장 파일 4개 모두 — 런타임 장치·정밀도 이름까지(P10 이 보내는 문자열)
        self.assertEqual({pin["file"] for pin in pins.values() if pin["source"] == "ast"},
                         {"sam3ext/ui_vae_degrid.py", "sam3ext/vae_degrid.py", "sam3ext/vae_degrid_models.py",
                          "sam3ext/vae_degrid_runtime.py"})
        # P10 스펙의 라디오 선택지·기본값도 같은 핀 값에서 나온다(앱 상수 → SPECS)
        from core import forge_override_settings as fos
        device = fos.SPEC_BY_KEY[fos.OPT_DEGRID_DEVICE]
        precision = fos.SPEC_BY_KEY[fos.OPT_DEGRID_GPU_PRECISION]
        keep = fos.SPEC_BY_KEY[fos.OPT_DEGRID_KEEP_LOADED]
        self.assertEqual(device.values, (pins["degrid_device_auto"]["expected"], pins["degrid_device_cpu"]["expected"]))
        self.assertEqual(precision.values, (pins["degrid_precision_fp32"]["expected"],
                                            pins["degrid_precision_fp16"]["expected"]))
        self.assertEqual((device.default, precision.default, keep.default),
                         (pins["degrid_default_device"]["expected"], pins["degrid_default_precision"]["expected"],
                          pins["degrid_default_keep_loaded"]["expected"]))
        self.assertEqual(precision.infotext, "", "확장 OptionInfo 에 infotext 가 없다 — 'Anima DeGrid precision' 은 런타임 기록")

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
        registered = {title.lower() for title, entry in reg.SCRIPTS.items() if entry["script_info"]}
        message = _set_mismatch("script-info 스크립트(소문자 제목)", registered, set(self.by_title))
        self.assertFalse(message, message)

    def test_scripts_outside_script_info_have_no_live_entry(self):
        """script_info=False(ui() None — Forge 가 api_info 를 만들지 않는다)는 script-info 에 나올 수 없다."""
        for title, entry in reg.SCRIPTS.items():
            if not entry["script_info"]:
                with self.subTest(script=title):
                    self.assertNotIn(title.lower(), self.by_title)
                    self.assertEqual((entry["live_argc"], entry["shape"], entry["ui_return"]), (0, "", None))

    def test_live_arg_counts(self):
        for title, entry in reg.SCRIPTS.items():
            if not entry["script_info"]:
                continue
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
            if not entry["script_info"]:
                continue
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

    def test_anima38_live_defaults_vs_app_t2i_defaults(self):
        """Anima 3.8B 앱 기본값(APP_T2I_DEFAULTS — 사용자 Forge ui-config txt2img)과 라이브 txt2img 기본값의 차이는
        KNOWN_DIFFS 에 적힌 것(부정 커넥터 켬)뿐이다. v1 enabled 는 결정 D1=B 로 라이브와 같다(끔). img2img 는 앱이
        블록을 보내지 않는 기본(토글 끔)이라 라이브 기본값 그대로다(test_anima38_live_defaults_match_app)."""
        from core.anima38 import APP_T2I_DEFAULTS, ARG_NAMES, V1_ENABLED_DEFAULT
        title = "Anima 3.8B (Qwen3.5 / v2)"
        entry = reg.SCRIPTS[title]
        app = dataclasses.asdict(APP_T2I_DEFAULTS)
        observed = {}
        args = self._modes(title)[False]
        for index, name in enumerate(ARG_NAMES):
            if index in entry["runtime_choices"]:
                continue
            live = args[index].get("value")
            if not same_value(app[name], live):
                observed[(title, name, "default")] = (app[name], live)
        self.assertFalse(V1_ENABLED_DEFAULT)
        problems = _diff_problems(observed, _known("fixture", title))
        self.assertEqual(problems, [], "\nAnima 3.8B 앱 기본값과 라이브 txt2img 기본값:\n" + "\n".join(problems))

    def test_dora_live_defaults_match_app(self):
        """DoRA 라이브 기본값(t2i·i2i arg0-4)을 앱 규칙으로 읽으면 EXTENSION_DEFAULTS 이고, 앱 기본값(APP_DEFAULTS —
        사용자 Forge ui-config)과의 차이는 KNOWN_DIFFS 에 적힌 것(txt2img 켬)뿐이다."""
        from core import dora_infer_mode as dim
        title = "DoRA Inference Mode"
        observed = {}
        for is_img2img, args in self._modes(title).items():
            with self.subTest(img2img=is_img2img):
                live = dict(zip(dim.ARG_NAMES, (arg.get("value") for arg in args)))
                parsed = dim.DoraSettings(
                    enabled=bool(live["enabled"]), mode=dim.normalize_mode(live["mode"]),
                    inserted=dim.normalize_insert(live["inserted"]),
                    weak_strength=dim.clamp_strength(live["weak_strength"]),
                    weak_scope=dim.normalize_scope(live["weak_scope"]))
                self.assertEqual(parsed, dim.EXTENSION_DEFAULTS)
                for name in dim.ARG_NAMES:
                    app, value = getattr(dim.APP_DEFAULTS, name), getattr(parsed, name)
                    if not same_value(app, value):
                        observed[(title, name, "default")] = (app, live[name])
        problems = _diff_problems(observed, _known("fixture", title))
        self.assertEqual(problems, [], "\nDoRA 앱 기본값과 라이브 기본값:\n" + "\n".join(problems))

    def test_dora_live_choices_normalize_to_app_keys_in_extension_order(self):
        """라이브 선택지 라벨 → 서로 다른 앱 키(확장 라디오 순서). 모르는 라벨이 생기면 앱이 그 값을 못 고른다."""
        from core import dora_infer_mode as dim
        cases = ((1, dim.normalize_mode, dim.MODE_OPTIONS), (2, dim.normalize_insert, dim.INSERT_OPTIONS),
                 (4, dim.normalize_scope, dim.SCOPE_OPTIONS))
        for args in self._modes("DoRA Inference Mode").values():
            for index, normalize, options in cases:
                with self.subTest(index=index):
                    labels = args[index].get("choices")
                    self.assertEqual(tuple(labels), tuple(label for _k, label in options))
                    self.assertEqual([normalize(label) for label in labels], [key for key, _l in options])

    def test_degrid_live_defaults_match_app(self):
        """VAE DeGrid 라이브 기본값(t2i·i2i arg0-4)을 앱 규칙으로 읽으면 EXTENSION_DEFAULTS 이고(모델 arg1 은 실행 시점
        목록 — 앱 기본값은 '' = 자동, 확장도 목록의 첫 파일), 앱 기본값(APP_DEFAULTS — 사용자 Forge ui-config 도 꺼짐·
        Full·1·512)과의 차이는 KNOWN_DIFFS 에 적힌 것뿐이다(지금은 없다 — 앱 기본값 출처 표시도 없다)."""
        from core import vae_degrid as vdg
        title = vdg.SCRIPT_NAME
        runtime = reg.SCRIPTS[title]["runtime_choices"]
        observed = {}
        for is_img2img, args in self._modes(title).items():
            with self.subTest(img2img=is_img2img):
                live = dict(zip(vdg.ARG_NAMES, (arg.get("value") for arg in args)))
                self.assertIsInstance(live["enabled"], bool)
                self.assertIsNotNone(vdg.normalize_mode(live["mode"]), f"모르는 모드 라벨 {live['mode']!r}")
                parsed = vdg.DegridSettings(
                    enabled=live["enabled"], model=vdg.AUTO, mode=vdg.coerce_mode(live["mode"]),
                    strength=vdg.coerce_strength(live["strength"]), tile=vdg.coerce_tile(live["tile"]))
                self.assertEqual(parsed, vdg.EXTENSION_DEFAULTS)
                self.assertEqual(vdg.ARG_NAMES.index("model"), 1)
                self.assertEqual(runtime, (vdg.ARG_NAMES.index("model"),))
                for name in vdg.ARG_NAMES:
                    if vdg.ARG_NAMES.index(name) in runtime:
                        continue
                    app, value = getattr(vdg.APP_DEFAULTS, name), getattr(parsed, name)
                    if not same_value(app, value):
                        observed[(title, name, "default")] = (app, live[name])
        self.assertFalse(vdg.APP_DEFAULTS.apply_img2img, "I2I·인페인트 토글 기본 끔(D4)")
        problems = _diff_problems(observed, _known("fixture", title))
        self.assertEqual(problems, [], "\nVAE DeGrid 앱 기본값과 라이브 기본값:\n" + "\n".join(problems))

    def test_degrid_live_mode_choices_normalize_to_app_keys_in_extension_order(self):
        """라이브 모드 라벨(카드 MODE_CHOICES) → 서로 다른 앱 키(확장 라디오 순서 = MODES). 모르는 라벨이 생기면 위치
        인자 라벨이나 붙여 넣은 값이 조용히 Full 이 된다. 모델 선택지에는 'None' 자리 표시가 아닌 실제 이름만 있다."""
        from core import vae_degrid as vdg
        for args in self._modes(vdg.SCRIPT_NAME).values():
            labels = tuple(args[vdg.ARG_NAMES.index("mode")].get("choices") or ())
            self.assertEqual(labels, vdg.MODE_CHOICES)
            self.assertEqual([vdg.normalize_mode(label) for label in labels], list(vdg.MODES))
            self.assertEqual([vdg.normalize_mode(vdg.MODE_LABELS[key]) for key in vdg.MODES], list(vdg.MODES))
            models = args[vdg.ARG_NAMES.index("model")].get("choices") or []
            self.assertTrue(models, "픽스처를 받은 Forge 에 DeGrid 모델이 있었다(없으면 ['None'])")
            self.assertNotIn(vdg.NONE_NAME, models)

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
        cls.state_reads |= _state_reads_by_constant(src, sam3_file, "process", "state")
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

    def test_scripts_outside_script_info_return_none_from_ui(self):
        """script_info=False 항목은 ui() 의 모든 return 이 None 이어야 한다 — 목록을 돌려주면 script-info 에 나와
        인자 계약이 생긴다(레지스트리를 script_info=True 로 옮기고 픽스처를 갱신한다)."""
        for title, entry in reg.SCRIPTS.items():
            info = self.scripts.get(title)
            if entry["script_info"] or info is None:
                continue
            with self.subTest(script=title):
                tree = ast.parse((EXT_ROOT / info.file).read_text(encoding="utf-8"))
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == info.cls)
                ui = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "ui")
                returns = [n for n in ast.walk(ui) if isinstance(n, ast.Return)]
                self.assertTrue(returns, f"{info.file} {info.cls}.ui() 에 return 이 없다")
                for ret in returns:
                    self.assertTrue(ret.value is None or (isinstance(ret.value, ast.Constant) and ret.value.value is None),
                                    f"{info.file}:{ret.lineno} ui() 가 None 이 아닌 값을 돌려준다")

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

    def test_p10_option_specs_match_installed_source(self):
        """P10 — 요청마다 덮어쓰는 옵션의 기본값·infotext·컴포넌트(체크박스 = bool, 라디오 = 선택지 문자열)·라디오 선택지
        값(순서까지)이 설치된 소스와 같고, onchange 콜백이 있는 것은 SAM3 RAM 보관 하나다. override 는 적용 때 콜백을
        돌리지 않으므로(processing.py:813) 다른 옵션에 콜백이 생기면 '요청마다 덮어써도 된다'는 가정과 카드 경고를 다시
        봐야 한다. 라디오 값이 바뀌면 앱이 보낸 값을 확장이 모르는 값으로 읽는다(조용히 다르게 돈다)."""
        from core.forge_override_settings import SPECS
        infos = self.src.option_infos()
        self.assertEqual(set(infos), set(self.options), "option_infos 가 add_option 을 모두 읽지 못했다")
        problems = []
        for spec in SPECS:
            info = infos.get(spec.key)
            if info is None:
                problems.append(f"  {spec.key}: 설치된 확장에 없다")
                continue
            checks = [("default", spec.default), ("infotext", spec.infotext), ("component", spec.component)]
            if spec.choices:
                checks.append(("choices", spec.values))
            for field, want in checks:
                if info[field] != want or type(info[field]) is not type(want):
                    problems.append(f"  {spec.key}.{field}: 앱 {want!r} / 확장 {info[field]!r} ({info['file']})")
            if not spec.choices and info["choices"] != ():
                problems.append(f"  {spec.key}.choices: 앱 체크박스 / 확장 선택지 {info['choices']!r} ({info['file']})")
            if info["onchange"] != spec.onchange:
                problems.append(f"  {spec.key}.onchange: 앱 {spec.onchange!r} / 확장 {info['onchange']!r}")
        self.assertEqual(problems, [], "\ncore/forge_override_settings.SPECS 가 설치된 확장 옵션과 다르다:\n"
                         + "\n".join(problems))

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
        """나-4: process() 는 정해진 키만 골라 Sam3Args 에 넘긴다 — 확장이 키 이름을 바꾸면 앱 값이 조용히 버려진다.

        요청 전용 키(``SAM3_REQUEST_ONLY_KEYS`` — 단독 SAM3·Refine 의 원본 기준 요청)는 생성 안 SAM3(build_state)가
        보내지 않고 단독 경로(``with_init_source``)만 더한다. process() 는 그 키를 Sam3Args 밖에서 읽어야 한다
        (extra=forbid 라 Sam3Args 에 넣으면 검증이 통째로 실패해 SAM3 가 꺼진다).
        """
        from core.sam3_args import build_state, with_init_source
        request_only = set(reg.SAM3_REQUEST_ONLY_KEYS)
        sent = set(build_state({}))
        standalone = set(with_init_source(build_state({})))
        dropped = sorted(sent - self.state_reads)
        unsent = sorted(self.state_reads - sent - request_only)
        self.assertEqual(dropped, [], f"앱이 보내지만 process() 가 읽지 않는 키(조용히 버려짐): {dropped}")
        self.assertEqual(unsent, [], f"process() 가 state 에서 읽지만 앱이 보내지 않는 키: {unsent}")
        self.assertEqual(sorted(sent & request_only), [], "생성 안 SAM3(build_state)는 요청 전용 키를 보내지 않는다")
        self.assertEqual(sorted(standalone - sent), sorted(request_only), "단독 경로가 더하는 키 = 요청 전용 키")
        self.assertEqual(sorted(request_only - self.state_reads), [],
                         "앱이 단독 경로에서 보내지만 process() 가 읽지 않는 요청 전용 키(조용히 무시됨)")
        self.assertEqual(set(self.payload_keys), set(self.sam3args["fields"]),
                         "process() 가 Sam3Args 에 넘기는 키 ≠ Sam3Args 필드")
        self.assertEqual(sorted(request_only & set(self.payload_keys)), [],
                         "요청 전용 키를 Sam3Args 에 넘기면(extra=forbid) 검증이 실패해 SAM3 가 꺼진다")
        self.assertEqual(sorted(self.state_reads - set(self.payload_keys) - request_only),
                         sorted(reg.SAM3_ACTIVATION_KEYS), "Sam3Args 밖에서 읽는 활성화 플래그가 바뀌었다")

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


class ExtraSchedulersRegistryTests(unittest.TestCase):
    """Extra Schedulers 앱 노출(계획 A-XS3, 보류 → mapped) — 확장 없이 도는 레지스트리 검사."""

    def test_mapped_with_the_comfy_wiring(self):
        from core import extra_schedulers as xs
        entry = reg.SCRIPTS[xs.SCRIPT_NAME]
        self.assertEqual(entry["status"], reg.MAPPED)
        self.assertEqual(entry["app_title"], "core.extra_schedulers:SCRIPT_NAME")
        self.assertEqual((entry["form"], entry["live_argc"], entry["shape"]), ("positional_or_dict", 8, "7c291b1a040a"))
        self.assertIsNone(entry["arg_names"], "ARG_NAMES 는 스크립트 모듈 밖(sam3ext/ui_extra_schedulers.py) — 핀 xsched_arg_names")
        for ref in ("core/extra_schedulers.py:plan", "core/extra_schedulers.py:from_infotext",
                    "core/alwayson_propagation.py:TITLE_EXTRA_SCHEDULERS", "core/sam_extra_notices.py:CODE_XSCHED_NOT_APPLIED",
                    "core/image_metadata.py:_EXTENSION_MARKERS"):
            self.assertIn(ref, entry["app"])
        # 계획 C6 — ComfyUI(앱 노드 팩 1.7.0)에 연결됐다: comfy 참조가 있고 '팩 1.7.0 단계 전' 틈은 없다
        for ref in ("comfy_custom_nodes/ai_studio_forge_parity/extra_schedulers.py:ForgeNeoExtraSchedulers",
                    "comfy_custom_nodes/ai_studio_forge_parity/extra_schedulers.py:install_extra_schedulers",
                    "core/comfy_workflow_compiler.py:_add_extra_schedulers",
                    "core/comfy_workflow_compiler.py:_check_extra_schedulers_reach",
                    "core/comfy_workflow_compiler.py:_check_pack_owner",
                    "core/comfy_metadata.py:_extra_schedulers_parameters",
                    "core/comfy_workflow_controls.py:feature_preflight"):
            self.assertIn(ref, entry["comfy"])
        self.assertFalse([gap for gap in entry["gaps"] if "단계 전" in gap or "comfy_pending" in gap], entry["gaps"])
        self.assertTrue(any('ρ 7·모델의 σ 범위' in gap for gap in entry["gaps"]), 'ρ 7·모델의 σ 범위')
        # 계획 C7 — 식·목록 미리 검사가 연결됐다(오류 줄·생성 전 경고): 'C7' 틈은 없다
        for ref in ("core/extra_schedulers.py:check_text", "ui/extra_schedulers_ui.py:watch_text_errors",
                    "core/sam_extra_notices.py:CODE_XSCHED_INVALID_TEXT"):
            self.assertIn(ref, entry["app"])
        self.assertFalse([f"{key}: {gap}" for key in (xs.SCRIPT_NAME,) for gap in entry["gaps"] if gap.startswith("C7")])
        self.assertFalse([gap for gap in reg.MODULES["sam3ext/extra_schedulers/"].get("gaps", ()) if gap.startswith("C7")])
        self.assertTrue(any('model_sampling 사본' in gap for gap in entry["gaps"]), 'model_sampling 사본')
        self.assertTrue(any('SamplerCustom·BasicScheduler' in gap for gap in entry["gaps"]), 'SamplerCustom·BasicScheduler')
        self.assertTrue(any(gap.startswith("P18") for gap in entry["gaps"]))
        rows = {f"{section}[{key}]": row for section, table in reg.classified_sections().items()
                for key, row in table.items() if "extra_schedulers" in key.lower() or "extra schedulers" in key.lower()}
        self.assertEqual(len(rows), 1 + 1 + 3, sorted(rows))   # SCRIPTS·XYZ·MODULES 셋
        self.assertEqual(sorted(k for k, row in rows.items() if row.get("package") == reg.HOLD), [])
        xyz = reg.XYZ_AXES["[Extra Schedulers (sam-extra)]"]
        self.assertEqual((xyz["status"], xyz["package"]), (reg.DEFERRED, "P18"))

    def test_generate_from_metadata_restore_is_registered(self):
        """계획 A-R1(X3b) — '이 설정으로 생성'이 스케줄러와 그 값을 함께 되살리는 곳이 행에 있다."""
        from core import extra_schedulers as xs
        entry = reg.SCRIPTS[xs.SCRIPT_NAME]
        for ref in ("core/extra_schedulers.py:restore_from_infotext", "ui/extra_schedulers_ui.py:restore_from_metadata",
                    "ui/image_metadata_actions.py:restore_sampling_values", "ui/extra_schedulers_ui.py:apply_infotext"):
            self.assertIn(ref, entry["app"])

    def test_forge_500_scheduler_error_reaches_the_user_and_the_gap_is_gone(self):
        """GPU 확인 F1: Forge 500 본문의 'Extra Schedulers: …' 가 생성 오류 문구가 된다 — 그 호출 지점이 행에 있고 '아직
        붙지 않는다' 틈은 지웠다. 보조 패스 봉투의 다시 판정(drop_unread)도 행에 있다."""
        from core import extra_schedulers as xs
        entry = reg.SCRIPTS[xs.SCRIPT_NAME]
        for ref in ("backends/webui_backend.py:_server_error_line", "backends/webui_backend.py:_scheduler_error_line",
                    "core/sam_extra_notices.py:explain_scheduler_error", "core/alwayson_propagation.py:drop_unread"):
            self.assertIn(ref, entry["app"])
        self.assertFalse([gap for gap in entry["gaps"] if "explain_scheduler_error" in gap or "HTTP 500 본문" in gap],
                         entry["gaps"])

    def test_pins_mirror_every_shared_app_constant(self):
        """앱이 확장 값을 하드코딩한 곳마다 핀이 있다 — 빠지면 확장이 바뀌어도 앱만 조용히 어긋난다."""
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS if pin["id"].startswith("xsched_")}
        apps = {pin["app"] for pin in pins.values() if "app" in pin}
        names = ("SCRIPT_NAME", "ARG_NAMES", "MODE_LABELS", "MODES", "MODE_EXPRESSION", "MODE_SIGMAS",
                 "DEFAULT_EXPRESSION", "DEFAULT_SIGMAS", "KEY_CUSTOM_EXPRESSION", "KEY_CUSTOM_SIGMAS", "KEY_LAPLACE_MU",
                 "KEY_LAPLACE_BETA", "INFOTEXT_KEYS", "USE_SAME_SCHEDULER", "LABEL_LAPLACE", "LABEL_CUSTOM",
                 "README_LABELS", "LAPLACE_MU_MIN", "LAPLACE_MU_MAX", "LAPLACE_MU_STEP", "LAPLACE_MU_DEFAULT",
                 "LAPLACE_BETA_MIN", "LAPLACE_BETA_MAX", "LAPLACE_BETA_STEP", "LAPLACE_BETA_DEFAULT",
                 # v0.33.0 — Flow Cosmos 세 값(D19)·새 스케줄러 셋의 라벨·리터럴
                 "KEY_FLOW_COSMOS_RHO", "KEY_FLOW_COSMOS_SIGMA_MAX", "KEY_FLOW_COSMOS_SIGMA_MIN",
                 *(f"FLOW_COSMOS_{what}_{bound}" for what in ("RHO", "SIGMA_MAX", "SIGMA_MIN")
                   for bound in ("MIN", "MAX", "STEP", "DEFAULT")),
                 "LABEL_REACT_DYNSF", "REACT_DYNSF_FACTOR", "LABEL_FLOW_COSMOS_RHO7", "LABEL_FLOW_COSMOS_DYNAMIC",
                 "FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_RHO")
        self.assertEqual(sorted({f"core.extra_schedulers:{name}" for name in names} - apps), [], "핀이 없는 앱 상수")
        self.assertEqual({pin["file"] for pin in pins.values() if pin["source"] == "ast"},
                         {"sam3ext/ui_extra_schedulers.py", "sam3ext/extra_schedulers/settings.py",
                          "sam3ext/extra_schedulers/registry.py", "sam3ext/extra_schedulers/schedulers.py",
                          # 계획 C0 — 팩만 하드코딩하는 식 계산기·시그마 목록 한도(앱 상수 없음, '_comfy' 쌍둥이)
                          "sam3ext/extra_schedulers/expression.py", "sam3ext/extra_schedulers/sigma_list.py"})
        self.assertEqual({(pin["index"], pin["field"]) for pin in pins.values() if pin["source"] == "fixture"},
                         {(index, field) for index in (3, 4, 5, 6, 7) for field in ("minimum", "maximum", "step", "value")})
        logs = {pin["name"] for pin in pins.values() if pin["source"] == "ast" and pin["name"].endswith("_NOT_FLOW_LOG")}
        self.assertEqual(logs, {"FLOW_COSMOS_NOT_FLOW_LOG", "FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG"})


class ExtraSamplersV033RegistryTests(unittest.TestCase):
    """sam-extra v0.33.0 Extra Samplers 앱 노출(계획 A-XSa2, 보류 → mapped) — 확장 없이 도는 레지스트리 검사: 행은 인자 5개의
    위치 spec(app_spec — 픽스처·ui() 와 비교), ComfyUI 는 팩 1.7.0 연결(계획 C6), 앱 core/extra_samplers 가 하드코딩한 확장
    값마다 핀(PLAN_amend_v033 §4 + 결과 알림이 읽는 Dy 상태 줄)."""

    def test_mapped_with_the_comfy_wiring(self):
        from core import extra_samplers as xsamp
        entry = reg.SCRIPTS[xsamp.SCRIPT_NAME]
        self.assertEqual(entry["status"], reg.MAPPED)
        self.assertEqual((entry["form"], entry["live_argc"], entry["shape"]), ("positional", 5, "3ee8b4de008d"))
        self.assertEqual(entry["app_title"], "core.extra_samplers:SCRIPT_NAME")
        self.assertEqual(entry["app_spec"], ("core.extra_samplers:ARG_SPEC", xsamp.SPEC_PREFIX))
        self.assertEqual(entry["ui_return"], "app_spec")
        self.assertEqual(tuple(key[len(xsamp.SPEC_PREFIX):] for key, *_ in xsamp.ARG_SPEC), xsamp.ARG_NAMES)
        self.assertEqual(entry["api_reads"], "", "params.settings_from_args 는 스크립트 클래스 밖 — 인덱스를 AST 로 못 읽는다")
        for ref in ("core/extra_samplers.py:plan", "core/extra_samplers.py:as_block", "core/extra_samplers.py:from_infotext",
                    "core/extra_samplers.py:parse_status", "core/alwayson_propagation.py:TITLE_EXTRA_SAMPLERS",
                    "core/sam_extra_capabilities.py:POSITIONAL_SPECS",
                    "core/sam_extra_notices.py:CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED",
                    "core/sam_extra_notices.py:CODE_EXTRA_SAMPLERS_WINDOW_SKIPPED",
                    "core/sam_extra_notices.py:CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED",
                    "core/image_metadata.py:_EXTENSION_MARKERS",
                    # A-XSa3 — 기여자·저장·붙여 넣기
                    "ui/extra_samplers_ui.py:contribute", "ui/extra_samplers_ui.py:apply_saved_settings",
                    "ui/extra_samplers_ui.py:apply_infotext", "ui/sampling_blocks.py:CONTRIBUTORS",
                    # A-XSa5 — 파라미터 카드의 ER SDE 칸(TS 거울)
                    "frontend/src/components/params/ExtraSamplersPanel.vue",
                    "frontend/src/utils/extraSamplers.ts:passesUsing",
                    "frontend/src/utils/extraSamplers.ts:windowUnsupported"):
            self.assertIn(ref, entry["app"])
        # 계획 C6 — ComfyUI(앱 노드 팩 1.7.0)에 연결됐다: comfy 참조가 있고 '팩 1.7.0 단계 전' 틈은 없다
        for ref in ("comfy_custom_nodes/ai_studio_forge_parity/extra_samplers_nodes.py:ForgeNeoExtraSamplerSettings",
                    "comfy_custom_nodes/ai_studio_forge_parity/extra_samplers_register.py:install_extra_samplers",
                    "core/comfy_workflow_compiler.py:_add_extra_samplers",
                    "core/comfy_workflow_compiler.py:_check_pack_sampler_names",
                    "core/comfy_workflow_compiler.py:_check_pack_owner",
                    "core/comfy_metadata.py:_extra_samplers_parameters",
                    "core/comfy_workflow_controls.py:feature_preflight"):
            self.assertIn(ref, entry["comfy"])
        self.assertFalse([gap for gap in entry["gaps"] if "단계 전" in gap or "comfy_pending" in gap], entry["gaps"])
        self.assertTrue(any('Restart (flow): Forge 는 재시작 구간에서 프롬프트 편집 스텝 카운터를 되감는다' in gap for gap in entry["gaps"]), 'Restart (flow): Forge 는 재시작 구간에서 프롬프트 편집 스텝 카운터를 되감는다')
        self.assertTrue(any('UniPC bh2: 같은 알고리즘, 비트 동일은 주장하지 않음' in gap for gap in entry["gaps"]), 'UniPC bh2: 같은 알고리즘, 비트 동일은 주장하지 않음')
        self.assertTrue(any('XSa-b' in gap for gap in entry["gaps"]), 'XSa-b')
        self.assertTrue(any('XSa-d' in gap for gap in entry["gaps"]), 'XSa-d')
        # GPU part 2 O3 — ADetailer 만 ER SDE 인 생성의 값은 infotext 에 없어 되살릴 수 없다
        self.assertTrue(any('ADetailer 만 ER SDE' in gap and '되살릴 수 없다' in gap for gap in entry["gaps"]), 'O3')
        self.assertIn("core/alwayson_propagation.py:drop_unread", entry["app"])
        self.assertTrue(any(gap.startswith("P18") for gap in entry["gaps"]))
        rows = {f"{section}[{key}]": row for section, table in reg.classified_sections().items()
                for key, row in table.items() if "extra_samplers" in key.lower() or "extra samplers" in key.lower()}
        self.assertEqual(sorted(rows), ["MODULES[sam3ext/extra_samplers/]", "MODULES[scripts/anima_extra_samplers.py]",
                                        "SCRIPTS[Extra Samplers]", "XYZ_AXES[[Extra Samplers]]"])
        self.assertEqual(sorted(k for k, row in rows.items() if row.get("package") == reg.HOLD), [])
        self.assertEqual(sorted(k for k, row in rows.items() if row["status"] == reg.MAPPED), [
            "MODULES[sam3ext/extra_samplers/]", "MODULES[scripts/anima_extra_samplers.py]", "SCRIPTS[Extra Samplers]"])
        xyz = reg.XYZ_AXES[xsamp.XYZ_PREFIX]
        self.assertEqual(xyz["labels"], tuple(sorted(axis[len(xsamp.XYZ_PREFIX) + 1:] for axis in xsamp.XYZ_AXES)))
        self.assertEqual((xyz["status"], xyz["package"]), (reg.DEFERRED, "P18"))

    def test_generate_from_metadata_restore_is_registered(self):
        """계획 A-R1(X3b) — '이 설정으로 생성'이 샘플러와 ER SDE 값을 함께 되살리는 곳이 행에 있다."""
        from core import extra_samplers as xsamp
        entry = reg.SCRIPTS[xsamp.SCRIPT_NAME]
        for ref in ("core/extra_samplers.py:restore_from_infotext", "ui/extra_samplers_ui.py:restore_from_metadata",
                    "ui/image_metadata_actions.py:restore_sampling_values", "ui/extra_samplers_ui.py:apply_infotext"):
            self.assertIn(ref, entry["app"])

    def test_pins_mirror_every_shared_app_constant(self):
        from core import extra_samplers as xsamp
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS if pin["id"].startswith("extra_samplers_")}
        apps = {pin["app"] for pin in pins.values() if "app" in pin}
        labels = tuple(name for name in dir(xsamp) if name.startswith("LABEL_") and isinstance(getattr(xsamp, name), str)
                       and getattr(xsamp, name) in xsamp.LABELS)
        self.assertEqual(len(labels), 18)
        names = (*labels, "SCRIPT_NAME", "XYZ_PREFIX", "ARG_NAMES", "DEFAULT_MAX_STAGE", "MAX_STAGE_MIN", "MAX_STAGE_MAX",
                 "DEFAULT_ETA", "ETA_MIN", "ETA_MAX", "ETA_STEP", "DEFAULT_NOISE_WINDOW", "DEFAULT_NOISE_START",
                 "DEFAULT_NOISE_END", "NOISE_PERCENT_MIN", "NOISE_PERCENT_MAX", "NOISE_PERCENT_STEP", "KEY_MAX_STAGE",
                 "KEY_ETA", "KEY_NOISE_WINDOW", "NOISE_WINDOW_OFF", "WINDOW_SKIPPED_STATUS", "STATUS_KEY", "RESTART_S_MIN",
                 "RESTART_S_MAX", "RESTART_MIN_STEPS", "RESTART_TIMES", "RESTART_TIMES_LONG", "RESTART_LONG_FROM",
                 "UD10_HISTORY_WEIGHT", "UD10_ZERO_WEIGHT", "UD10_UNCOND_HISTORY_WEIGHT", "IPNDM_MAX_ORDER", "DEIS_MAX_ORDER",
                 "DEIS_MODE",
                 # A-XSa2 — 결과 알림(CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED)이 읽는 Dy 상태 줄 머리·이유
                 "DY_SKIPPED_PREFIX", "DY_REASON_SPECTRUM", "DY_REASON_WAN", "DY_REASON_PID", "DY_REASON_EXTRA_CONCAT")
        self.assertEqual(sorted({f"core.extra_samplers:{name}" for name in names} - apps), [], "핀이 없는 앱 상수")
        self.assertEqual(tuple(pins[f"extra_samplers_reason_{short}"]["app"] for short in
                               ("spectrum", "wan", "pid", "extra_concat")),
                         tuple(f"core.extra_samplers:DY_REASON_{name}" for name in ("SPECTRUM", "WAN", "PID", "EXTRA_CONCAT")))
        self.assertEqual(len(xsamp.DY_REASONS), 4)
        self.assertEqual({pin["file"] for pin in pins.values() if pin["source"] == "ast"},
                         {"scripts/anima_extra_samplers.py", "sam3ext/extra_samplers/params.py",
                          "sam3ext/extra_samplers/registry.py", "sam3ext/extra_samplers/substep_guard.py",
                          "sam3ext/extra_samplers/restart_flow.py", "sam3ext/extra_samplers/cfgpp_ud10_ab.py",
                          "sam3ext/extra_samplers/ipndm_deis.py",
                          "sam3ext/extra_samplers/euler_dy.py"})   # 계획 C0 — Dy 규칙(팩만, '_comfy' 쌍둥이)
        self.assertEqual({(pin["index"], pin["field"]) for pin in pins.values() if pin["source"] == "fixture"},
                         {(2, "value")} | {(index, field) for index in (3, 4) for field in ("minimum", "maximum", "step", "value")})

    def test_new_screen_only_files_are_ignored_with_a_reason(self):
        """v0.33.0 의 화면 전용 파일(목록 계열 순서·'기본값 저장' 버튼)은 앱이 따라 할 것이 없다 — ignored + 사유."""
        for key in ("scripts/list_order.py", "sam3ext/list_order.py", "scripts/layout_save_defaults.py",
                    "sam3ext/save_defaults.py", "sam3ext/ui_save_defaults.py", "javascript/notebook_save_defaults.js"):
            with self.subTest(module=key):
                row = reg.MODULES[key]
                self.assertEqual(row["status"], reg.IGNORED)
                self.assertTrue(row["reason"].strip())
                self.assertFalse(row.get("optional"))


class AnimaSpeedRegistryTests(unittest.TestCase):
    """Anima SPEED 앱 노출(계획 A-SP3, 보류 → mapped) — 확장 없이 도는 레지스트리 검사."""

    def test_mapped_main_only_with_the_comfy_wiring(self):
        from core import anima_speed as sp
        entry = reg.SCRIPTS[sp.SCRIPT_NAME]
        self.assertEqual(entry["status"], reg.MAPPED)
        self.assertEqual(entry["app_title"], "core.anima_speed:SCRIPT_NAME")
        self.assertEqual((entry["form"], entry["live_argc"], entry["shape"]), ("positional_or_dict", 14, "c6889a2ed8d6"))
        self.assertIsNone(entry["arg_names"], "ARG_NAMES 는 스크립트 모듈 밖(sam3ext/speed/forge_host.py) — 핀 speed_arg_names")
        for ref in ("core/anima_speed.py:plan", "core/anima_speed.py:as_block", "core/anima_speed.py:from_infotext",
                    "core/alwayson_propagation.py:TITLE_SPEED", "core/alwayson_propagation.py:main_only_titles",
                    "core/sam_extra_notices.py:CODE_SPEED_NOT_APPLIED", "core/sam_extra_notices.py:speed_result_notices",
                    "core/image_metadata.py:_EXTENSION_MARKERS"):
            self.assertIn(ref, entry["app"])
        # 계획 C6 — ComfyUI(앱 노드 팩 1.7.0)에 연결됐다: comfy 참조가 있고 '팩 1.7.0 단계 전' 틈은 없다
        for ref in ("comfy_custom_nodes/ai_studio_forge_parity/speed_nodes.py:ForgeNeoAnimaSpeed",
                    "comfy_custom_nodes/ai_studio_forge_parity/speed_runner.py:run_speed",
                    "core/comfy_workflow_compiler.py:_add_speed",
                    "core/comfy_workflow_compiler.py:_speed_plan",
                    "core/comfy_metadata.py:_speed_parameters",
                    "core/comfy_workflow_controls.py:feature_preflight",
                    "core/comfy_compatibility.py:RECIPES"):
            self.assertIn(ref, entry["comfy"])
        self.assertFalse([gap for gap in entry["gaps"] if "단계 전" in gap or "comfy_pending" in gap], entry["gaps"])
        self.assertTrue(any("'Anima SPEED status'" in gap for gap in entry["gaps"]), "'Anima SPEED status'")
        self.assertTrue(any('hires= 는 Hires 노드를 따른다' in gap for gap in entry["gaps"]), 'hires= 는 Hires 노드를 따른다')
        self.assertTrue(any('노드가 받은 기본값' in gap for gap in entry["gaps"]), '노드가 받은 기본값')
        self.assertTrue(any('build_plan 거절' in gap for gap in entry["gaps"]), 'build_plan 거절')
        self.assertTrue(any(gap.startswith("P18") for gap in entry["gaps"]))
        # 붙여 넣기는 가져오기(액션 sam_extra_import — 계획 A-CC12a/b)로 연결됐다: P16 틈은 닫혔다
        self.assertFalse(any(gap.startswith("P16") for gap in entry["gaps"]), entry["gaps"])
        for ref in ("ui/sam_extra_import.py:import_infotext", "frontend/src/views/PngInfoView.vue",
                    "frontend/src/views/GalleryView.vue"):
            self.assertIn(ref, entry["app"])
        rows = {f"{section}[{key}]": row for section, table in reg.classified_sections().items()
                for key, row in table.items() if "anima speed" in key.lower() or "anima_speed" in key.lower()
                or key.startswith("sam3ext/speed/")}
        self.assertEqual(sorted(rows), ["MODULES[sam3ext/speed/]", "MODULES[scripts/anima_speed.py]",
                                        "SCRIPTS[Anima SPEED]", "XYZ_AXES[[Anima SPEED]]"])
        self.assertEqual(sorted(k for k, row in rows.items() if row.get("package") == reg.HOLD), [])
        xyz = reg.XYZ_AXES[sp.XYZ_PREFIX]
        self.assertEqual((xyz["status"], xyz["package"]), (reg.DEFERRED, "P18"))
        self.assertEqual(reg.OPTIONS[sp.OPT_IMG2IMG_RESCALE]["status"], reg.MAPPED)     # P10 행(F2)
        self.assertEqual(reg.OPTIONS[sp.OPT_LOG]["status"], reg.IGNORED)

    def test_pins_mirror_every_shared_app_constant(self):
        """앱이 확장 값을 하드코딩한 곳마다 핀이 있다 — 빠지면 확장이 바뀌어도 앱만 조용히 어긋난다."""
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS if pin["id"].startswith("speed_")}
        apps = {pin["app"] for pin in pins.values() if "app" in pin}
        names = ("SCRIPT_NAME", "XYZ_PREFIX", "ARG_NAMES", "DEFAULTS", "ARG_ALIASES", "INFOTEXT_KEY", "STATUS_KEY",
                 "INFOTEXT_IMG2IMG_RESCALE", "OPT_LOG", "UNSUPPORTED_SAMPLERS", "MODES", "THRESHOLDS", "TRANSFORMS",
                 "PRESET_NAMES", "PRESETS", "REFERENCE_LATENT", "CUSTOM_DEFAULT_A", "CUSTOM_DEFAULT_BETA",
                 "DELTA_MIN", "DELTA_MAX", "DELTA_STEP", "DIVISOR_MIN", "DIVISOR_MAX", "DIVISOR_STEP")
        self.assertEqual(sorted({f"core.anima_speed:{name}" for name in names} - apps), [], "핀이 없는 앱 상수")
        self.assertEqual({pin["file"] for pin in pins.values() if pin["source"] == "ast"},
                         {"scripts/anima_speed.py", "sam3ext/speed/forge_host.py", "sam3ext/speed/schedule.py",
                          "sam3ext/speed/runner.py"})   # 계획 C0 — 런너 시드 오프셋(팩만, '_comfy' 쌍둥이)
        self.assertEqual({(pin["index"], pin["field"]) for pin in pins.values() if pin["source"] == "fixture"},
                         {(index, field) for index in (5, 6) for field in ("minimum", "maximum", "step")})


class ColorcraftRegistryTests(unittest.TestCase):
    """Colorcraft 앱 노출(계획 A-CC5, 보류 → mapped) — 확장 없이 도는 레지스트리 검사."""

    def test_mapped_main_only_with_the_comfy_wiring(self):
        from core import colorcraft as cc
        entry = reg.SCRIPTS[cc.SCRIPT_NAME]
        self.assertEqual(entry["status"], reg.MAPPED)
        self.assertEqual(entry["app_title"], "core.colorcraft:SCRIPT_NAME")
        self.assertEqual((entry["form"], entry["live_argc"], entry["shape"]), ("positional_or_dict", 67, "456003a0acc3"))
        self.assertIsNone(entry["app_spec"], "위치 인자 spec 이 없다 — 첫 인자 하나(compact 문자열)만 보낸다")
        for ref in ("core/colorcraft.py:to_infotext", "core/colorcraft.py:plan", "core/colorcraft.py:from_infotext",
                    "core/colorcraft.py:build_chain", "core/alwayson_propagation.py:TITLE_COLORCRAFT",
                    "core/alwayson_propagation.py:main_only_titles", "core/sam_extra_capabilities.py:TITLE_COLORCRAFT",
                    "ui/colorcraft_ui.py:contribute", "ui/colorcraft_ui.py:apply_saved_settings",
                    "ui/sampling_blocks.py:CONTRIBUTORS", "core/sam_extra_notices.py:CODE_COLORCRAFT_NOT_APPLIED",
                    "core/sam_extra_notices.py:colorcraft_result_notices", "core/image_metadata.py:_EXTENSION_MARKERS"):
            self.assertIn(ref, entry["app"])
        # 계획 C6 — ComfyUI(앱 노드 팩 1.7.0)에 연결됐다: comfy 참조가 있고 '팩 1.7.0 단계 전' 틈은 없다
        for ref in ("comfy_custom_nodes/ai_studio_forge_parity/colorcraft_nodes.py:ForgeNeoColorcraft",
                    "comfy_custom_nodes/ai_studio_forge_parity/colorcraft_engine.py:apply_chain",
                    "core/comfy_workflow_compiler.py:_add_colorcraft",
                    "core/comfy_metadata.py:_colorcraft_parameters",
                    "core/comfy_workflow_controls.py:feature_preflight",
                    "core/comfy_compatibility.py:RECIPES"):
            self.assertIn(ref, entry["comfy"])
        self.assertFalse([gap for gap in entry["gaps"] if "단계 전" in gap or "comfy_pending" in gap], entry["gaps"])
        self.assertTrue(any("'SAM Extra Colorcraft status'" in gap for gap in entry["gaps"]), "'SAM Extra Colorcraft status'")
        self.assertTrue(any('디코드 VAE' in gap for gap in entry["gaps"]), '디코드 VAE')
        for package in ("P18", "HOLD"):
            self.assertTrue(any(gap.startswith(package) for gap in entry["gaps"]), package)
        self.assertFalse(any(gap.startswith("P16") for gap in entry["gaps"]), entry["gaps"])     # 가져오기로 연결됐다
        for ref in ("ui/sam_extra_import.py:import_infotext", "ui/colorcraft_ui.py:apply_config",
                    "frontend/src/utils/samExtraImport.ts:importSamExtra", "frontend/src/views/PngInfoView.vue",
                    "frontend/src/views/GalleryView.vue", "frontend/src/components/params/ColorcraftCard.vue"):
            self.assertIn(ref, entry["app"])
        rows = {f"{section}[{key}]": row for section, table in reg.classified_sections().items()
                for key, row in table.items() if "colorcraft" in key.lower()}
        self.assertEqual(sorted(rows), [
            "MODULES[javascript/colorcraft_editor.js]", "MODULES[javascript/colorcraft_schema.js]",
            "MODULES[javascript/colorcraft_sliders.js]", "MODULES[sam3ext/colorcraft/]", "MODULES[scripts/colorcraft.py]",
            "OPTIONS[sam3_colorcraft_log]", "OPTIONS[sam3_colorcraft_pre_dd_sigma]", "SCRIPTS[Colorcraft (sam-extra)]",
            "XYZ_AXES[[Colorcraft]]"])
        self.assertEqual(sorted(k for k, row in rows.items() if row.get("package") == reg.HOLD), [])
        self.assertEqual(sorted(k for k, row in rows.items() if row["status"] == reg.MAPPED), [
            "MODULES[sam3ext/colorcraft/]", "MODULES[scripts/colorcraft.py]", "OPTIONS[sam3_colorcraft_pre_dd_sigma]",
            "SCRIPTS[Colorcraft (sam-extra)]"])
        xyz = reg.XYZ_AXES[cc.XYZ_PREFIX]
        self.assertEqual((xyz["status"], xyz["package"]), (reg.DEFERRED, "P18"))
        self.assertEqual(len(xyz["labels"]), 14)
        self.assertEqual(reg.OPTIONS[cc.OPT_PRE_DD]["status"], reg.MAPPED)      # P10 행(F2)

    def test_pins_mirror_every_shared_app_constant(self):
        """앱이 확장 값을 하드코딩한 곳마다 핀이 있다 — 빠지면 확장이 바뀌어도 앱만 조용히 어긋난다."""
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS if pin["id"].startswith("colorcraft_")}
        apps = {pin["app"] for pin in pins.values() if "app" in pin}
        names = ("SCRIPT_NAME", "XYZ_PREFIX", "INFOTEXT_KEY", "STATUS_KEY", "FOREIGN_KEY", "VERSION", "MODIFIER_COUNT",
                 "MASK_COUNT", "COMBO_COUNT", "MODIFIER_TAGS", "KINDS", "ADVANCED_KINDS", "KIND_LABELS", "KIND_GROUPS",
                 "KIND_GATES", "MASK_NONE", "MASK_AXIS_OPTIONS", "MASK_MODE_OPTIONS", "COMBO_OP_OPTIONS",
                 "COLOR_SHIFT_MODE_OPTIONS", "CHROMA_PLANE_OPTIONS", "GROUP_ORDER")
        self.assertEqual(sorted({f"core.colorcraft:{name}" for name in names} - apps), [], "핀이 없는 앱 상수")
        self.assertEqual({pin["file"] for pin in pins.values()},
                         {"scripts/colorcraft.py", "sam3ext/colorcraft/spec.py", "sam3ext/colorcraft/engine.py",
                          "sam3ext/colorcraft/masking.py", "sam3ext/colorcraft/panel_state.py",
                          # 계획 C0 — 기저 표·앵커 캐시(팩만, '_comfy' 쌍둥이)
                          "sam3ext/colorcraft/basis.py", "sam3ext/colorcraft/hook.py"})
        self.assertEqual({pin["source"] for pin in pins.values()}, {"ast"})
        options = {pin["id"]: pin["app"] for pin in reg.SEMANTIC_PINS if pin["id"].startswith("opt_")}
        self.assertEqual(options["opt_colorcraft_pre_dd"], "core.forge_override_settings:OPT_COLORCRAFT_PRE_DD")
        self.assertEqual(options["opt_infotext_colorcraft_pre_dd"],
                         "core.forge_override_settings:INFOTEXT_COLORCRAFT_PRE_DD")

class ComfyWiringRegistryTests(unittest.TestCase):
    """계획 C6 — 네 기능의 MODULES 행(계산 모듈)도 팩 1.7.0 의 다시 구현을 comfy 로 가리키고, 어느 행에도 '팩 1.7.0 단계
    전'·'comfy_pending' 틈이 남지 않는다."""

    def test_module_rows_point_at_the_pack(self):
        expected = {
            "sam3ext/colorcraft/": "comfy_custom_nodes/ai_studio_forge_parity/colorcraft_engine.py:apply_chain",
            "sam3ext/speed/": "comfy_custom_nodes/ai_studio_forge_parity/speed_runner.py:run_speed",
            "sam3ext/extra_schedulers/": "comfy_custom_nodes/ai_studio_forge_parity/extra_schedulers_math.py:compile_expression",
            "sam3ext/extra_samplers/": "comfy_custom_nodes/ai_studio_forge_parity/extra_samplers_register.py:install_extra_samplers",
        }
        for key, ref in expected.items():
            with self.subTest(module=key):
                self.assertEqual(reg.MODULES[key]["status"], reg.MAPPED)
                self.assertIn(ref, reg.MODULES[key]["comfy"])
        stale = [f"{section}[{key}]" for section, table in reg.classified_sections().items()
                 for key, row in table.items() for gap in row.get("gaps", ())
                 if "comfy_pending" in gap or "팩 1.7.0 단계 전" in gap or "C0 에서 들어온다" in gap]
        self.assertEqual(stale, [])


class ComfyPackMirrorPinTests(unittest.TestCase):
    """계획 C0 — 본 트리에 들어온 팩 1.7.0 이 확장 값을 하드코딩한 곳마다 '<id>_comfy' 쌍둥이 핀이 있고, 쌍둥이는 짝 핀
    (확장 값)과 같은 대상을 본다. 값 비교 자체는 test_app_constants_follow_pinned_extension_constants(팩 import)와
    test_semantic_pins_on_source_constants·test_semantic_pins_on_live_values(설치된 확장)가 한다."""

    PACK = "comfy_custom_nodes.ai_studio_forge_parity."

    def test_every_comfy_twin_mirrors_its_extension_pin(self):
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS}
        twins = {pin_id: pin for pin_id, pin in pins.items() if pin_id.endswith("_comfy")}
        self.assertGreater(len(twins), 100)
        keys = ("source", "file", "name", "expected", "script", "index", "field")
        for pin_id, twin in sorted(twins.items()):
            with self.subTest(pin=pin_id):
                self.assertTrue(twin["app"].startswith(self.PACK), twin["app"])
                base = pins.get(pin_id[:-len("_comfy")])
                self.assertIsNotNone(base, "짝 핀(확장 값)이 없다")
                self.assertEqual({key: base.get(key) for key in keys}, {key: twin.get(key) for key in keys})
                self.assertFalse(base.get("app", "").startswith(self.PACK), "짝 핀의 app 은 앱 상수(또는 없음)")

    def test_shared_markers_are_one_set_in_the_pack(self):
        """계획 R4 — 보조 스텝 표시·스텝 오프셋 키워드·다시 내놓은 σ 표시는 확장 파일 셋의 상수 하나씩이고, 팩은
        guidance_common 에 한 번만 둔다(기능별 중복 핀 colorcraft_substep_marker·speed_step_offset_kwarg_comfy 는 없다)."""
        pins = {pin["id"]: pin for pin in reg.SEMANTIC_PINS}
        self.assertEqual({pin_id: (pins[pin_id]["file"], pins[pin_id]["name"], pins[f"{pin_id}_comfy"]["app"])
                          for pin_id in ("shared_substep_marker", "shared_step_offset_kwarg", "shared_republished_attr")}, {
            "shared_substep_marker": ("sam3ext/extra_samplers/common.py", "SUBSTEP_MARKER",
                                      f"{self.PACK}guidance_common:SUBSTEP_MARKER"),
            "shared_step_offset_kwarg": ("sam3ext/speed/runner.py", "STEP_OFFSET_KWARG",
                                         f"{self.PACK}guidance_common:STEP_OFFSET_KWARG"),
            "shared_republished_attr": ("sam3ext/guidance/sigmas.py", "REPUBLISHED_FROM",
                                        f"{self.PACK}guidance_common:REPUBLISHED_FROM")})
        for duplicate in ("colorcraft_substep_marker", "speed_step_offset_kwarg_comfy", "extra_samplers_substep_marker",
                          "extra_samplers_step_offset_kwarg"):
            self.assertNotIn(duplicate, pins)

    def test_every_twin_guards_a_constant_the_pack_reads(self):
        """쌍둥이가 지키는 팩 상수는 팩 어딘가가 실제로 읽어야 한다(정의·``__all__`` 문자열만 있으면 확장 값이 바뀔 때 핀이
        죽은 상수를 고치게 보내고, 팩이 실제로 쓰는 하드코딩 문자열은 아무도 보지 않는다 — GC 검토: speed_nodes.KEY 는
        읽는 곳이 없었다. 팩은 infotext 를 쓰지 않고 ComfyUI PNG 키는 앱 상수 핀 speed_infotext_key 가 지킨다)."""
        pack_dir = REPO / "comfy_custom_nodes" / "ai_studio_forge_parity"
        trees = [ast.parse(path.read_text(encoding="utf-8")) for path in sorted(pack_dir.rglob("*.py"))
                 if "vendor" not in path.relative_to(pack_dir).parts]
        read: set = set()
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    read.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    read.add(node.attr)
                elif isinstance(node, ast.ImportFrom):
                    read.update(alias.name for alias in node.names)
        unread = sorted(pin["app"] for pin in reg.SEMANTIC_PINS if pin["id"].endswith("_comfy")
                        and pin["app"].split(":", 1)[1].split(".")[0] not in read)
        self.assertEqual(unread, [], "팩이 읽지 않는 상수를 지키는 쌍둥이 — 상수를 지우고 쌍둥이도 빼거나 팩이 쓰게 하라")

    def test_pack_constants_that_mirror_the_extension_are_pinned(self):
        """팩이 확장 값을 그대로 하드코딩한 상수마다 쌍둥이 핀 — 빠지면 확장이 바뀌어도 팩만 조용히 어긋난다."""
        apps = {pin["app"] for pin in reg.SEMANTIC_PINS if pin["id"].endswith("_comfy")}
        cosmos = tuple(f"FLOW_COSMOS_{what}_{bound}" for what in ("RHO", "SIGMA_MAX", "SIGMA_MIN")
                       for bound in ("MIN", "MAX", "STEP", "DEFAULT"))
        wanted = {
            "guidance_common": ("SUBSTEP_MARKER", "STEP_OFFSET_KWARG", "REPUBLISHED_FROM"),
            "extra_schedulers_math": (
                "ARG_NAMES", "MODES", "MODE_EXPRESSION", "MODE_SIGMAS", "DEFAULT_EXPRESSION", "DEFAULT_SIGMAS",
                "LAPLACE_MU_MIN", "LAPLACE_MU_MAX", "LAPLACE_MU_DEFAULT", "LAPLACE_BETA_MIN", "LAPLACE_BETA_MAX",
                "LAPLACE_BETA_DEFAULT", *cosmos, "REACT_DYNSF_FACTOR", "FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN",
                "FLOW_COSMOS_RHO", "FLOW_COSMOS_NOT_FLOW_LOG", "FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG",
                "FLOW_COSMOS_VALUES_SOURCE", "MAX_LENGTH", "MAX_NODES", "MAX_DEPTH", "MAX_EXPONENT", "MAX_CALL_ARGS",
                "VARIABLES", "MAX_TEXT_LENGTH", "MAX_VALUES", "KARRAS_DYNAMIC_RHO", "LAPLACE_EPSILON", "RHO_SOURCE",
                "ZERO_LENGTH_STEP"),
            "extra_samplers_math": (
                "ARG_NAMES", "DEFAULT_MAX_STAGE", "MAX_STAGE_MIN", "MAX_STAGE_MAX", "DEFAULT_ETA", "ETA_MIN", "ETA_MAX",
                "ETA_STEP", "DEFAULT_NOISE_WINDOW", "DEFAULT_NOISE_START", "DEFAULT_NOISE_END", "NOISE_PERCENT_MIN",
                "NOISE_PERCENT_MAX", "NOISE_PERCENT_STEP"),
            "extra_samplers_restart": ("RESTART_S_MIN", "RESTART_S_MAX", "RESTART_MIN_STEPS", "RESTART_TIMES",
                                       "RESTART_TIMES_LONG", "RESTART_LONG_FROM"),
            "extra_samplers_dy": ("SMEA_SCALE", "RESIZE_MODE", "CHURN_MIN", "CHURN_MAX"),
            "extra_samplers_ipndm": ("IPNDM_MAX_ORDER",),
            "speed_nodes": ("ARG_NAMES", "DEFAULTS", "ARG_ALIASES", "BROWNIAN_STAGE_SEED_STRIDE"),
            "speed_schedule": ("MODES", "THRESHOLDS", "TRANSFORMS", "PRESET_NAMES", "PRESETS", "REFERENCE_LATENT",
                               "CUSTOM_DEFAULT_A", "CUSTOM_DEFAULT_BETA", "MAX_COARSE_FRACTION", "RESPACE_PATCH"),
            "speed_runner": ("EXPANSION_SEED_STRIDE", "RESPACE_SEED_OFFSET", "INIT_NOISE_SEED_OFFSET"),
            "colorcraft_engine": ("KINDS", "ADVANCED_KINDS", "NEUTRAL_COLOR"),
            "colorcraft_host": ("MASK_AXES", "MASK_MODES", "COMBINE_OPERATIONS", "ANCHOR_CACHE_LIMIT"),
            "colorcraft_basis": ("LATENT_FORMAT_TO_FAMILY", "MODEL_DEV_DEFAULTS", "VAE_DOWNSCALE_FACTORS",
                                 "VAE_DOWNSCALE_FACTOR"),
        }
        self.assertEqual(sorted({f"{self.PACK}{module}:{name}" for module, names in wanted.items() for name in names}
                                - apps), [], "쌍둥이 핀이 없는 팩 상수")
        # 팩 노드 입력은 그 상수에서 나온다 — 노드 이름 순서·범위·기본값이 핀이 지키는 값과 같다
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_math as xsm, extra_schedulers_math as xsc
        from comfy_custom_nodes.ai_studio_forge_parity.extra_samplers_nodes import ForgeNeoExtraSamplerSettings
        from comfy_custom_nodes.ai_studio_forge_parity.extra_schedulers import ForgeNeoExtraSchedulers
        samplers = ForgeNeoExtraSamplerSettings.INPUT_TYPES()["required"]
        self.assertEqual(tuple(name for name in samplers if name != "model"), xsm.ARG_NAMES)
        self.assertEqual(samplers["eta"][1]["min"], xsm.ETA_MIN)
        self.assertEqual((samplers["noise_start"][1]["max"], samplers["noise_end"][1]["default"]),
                         (xsm.NOISE_PERCENT_MAX, xsm.DEFAULT_NOISE_END))
        schedulers = ForgeNeoExtraSchedulers.INPUT_TYPES()["required"]
        self.assertEqual(tuple(name for name in schedulers if name != "model"), xsc.ARG_NAMES)
        self.assertEqual((schedulers["flow_cosmos_sigma_min"][1]["min"], schedulers["flow_cosmos_rho"][1]["default"]),
                         (xsc.FLOW_COSMOS_SIGMA_MIN_MIN, xsc.FLOW_COSMOS_RHO_DEFAULT))


if __name__ == "__main__":
    unittest.main()
