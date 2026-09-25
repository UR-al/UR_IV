"""core.sam_extra_scan 단위 테스트 — 작은 가짜 확장 트리로 AST 추출 규칙을 고정한다.

설치된 sam-extra 가 없는 PC 에서도 스캐너가 어떤 패턴을 따라가는지(상수·지연 import·f-string·루프 변수)
확인할 수 있게 한다. 실제 확장과의 비교는 tests/test_sam_extra_contract.py 가 한다.
"""
from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from core.sam_extra_diff import compare_ui_components
from core.sam_extra_scan import (
    ExtensionSource,
    extension_dir_candidates,
    find_installed_extension,
    format_index_set,
    is_extension_dir,
    parse_index_set,
)

_FILES = {
    "sam3ext/__init__.py": '''
        from .__version__ import __version__

        def __getattr__(name):
            if name == "TOOL_NAME":
                from .core import TOOL_NAME
                return TOOL_NAME
            raise AttributeError(name)
    ''',
    "sam3ext/__version__.py": '__version__ = "9.9.9"\n',
    "sam3ext/core.py": '''
        TOOL_NAME = "Fake Tool"
        OPT_KEEP = "fake_keep"
        BASE_PATH = "/fake-notes"
    ''',
    "sam3ext/routes.py": '''
        from .core import BASE_PATH

        ITEM_PATH = f"{BASE_PATH}/items"

        def register(app):
            item = f"{ITEM_PATH}/{{item_id}}"
            app.add_api_route(ITEM_PATH, lambda: None, methods=["GET"])
            app.add_api_route(item, lambda: None, methods=["PUT", "DELETE"])
            app.add_api_route(BASE_PATH + "/ping", lambda: None)
            cache = {}
            cache.get("/not-a-route")
    ''',
    "sam3ext/tools/__init__.py": "",
    "sam3ext/tools/helper.py": "X = 1\n",
    "sam3ext/args.py": '''
        from typing import Literal
        from pydantic import BaseModel, NonNegativeInt, confloat

        LIMIT = 1.0
        _NUMERIC_BOUNDS = {"sam3_threshold": (0.0, 1.0, False), "sam3_blur": (0, None, True)}

        class Sam3Args(BaseModel):
            sam3_mode: Literal["A", "B"] = "A"
            sam3_threshold: confloat(ge=0.0, le=LIMIT) = 0.4
            sam3_blur: NonNegativeInt = 4
            sam3_name: str = "x"
    ''',
    "scripts/fake.py": '''
        from modules import scripts, shared
        import sam3ext.core as core_mod
        from sam3ext import TOOL_NAME

        ARG_NAMES = ("enabled", "mode")
        ARG_DEFAULTS = {"enabled": False, "mode": core_mod.OPT_KEEP}
        LABEL_B = "[Fake] Mode"
        MODES = ("Off", "Keep", "Drop")

        def on_ui_settings():
            shared.opts.add_option(core_mod.OPT_KEEP, shared.OptionInfo(True, "keep"))
            shared.opts.add_option("fake_" + "literal", shared.OptionInfo(True, "lit"))

        def axes(xyz_grid):
            xyz_grid.AxisOption("[Fake] Enable", str, None)
            for label, setter in (("[Fake] Strength", None), (LABEL_B, None)):
                xyz_grid.AxisOption(label, float, setter)

        def handle_fake_click(*args):
            return None

        class FakeScript(scripts.Script):
            def title(self):
                return TOOL_NAME

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, is_img2img):
                if is_img2img:
                    return []
                label = "Fake enable"
                if InputAccordion is not None:
                    context = InputAccordion(False, label=label)
                else:
                    context = gr.Accordion(label, open=False)
                with context as enabled_box:
                    if InputAccordion is None:
                        enabled_box = gr.Checkbox(label="Enable", value=False)
                    mode = gr.Radio(label="Mode", choices=list(MODES[1:]), value=MODES[1])
                    strength = gr.Slider(-2.0, 2.0, 0.5, step=0.1, label=f"{LABEL_B} strength")
                    files = gr.Dropdown(label="Files", choices=_list_files(), value="None")
                    hidden = gr.Checkbox(value=False, visible=False)
                    other = make_widget(label="Other")
                return [enabled_box, mode, strength, files, hidden, other]

            def process(self, p, *args):
                def _arg(i, default):
                    return args[i] if len(args) > i else default
                state = dict(args[0]) if args else {}
                payload = {"a": _arg(0, 1), "b": _arg(2, 2)}
                state.get("sam3_enable", False)
                _xyz_or("mode")
                return payload

        class NotAlwaysOn(scripts.Script):
            def title(self):
                return "Selectable"

            def show(self, is_img2img):
                return True
    ''',
    "scripts/bad.py": '''
        from modules import shared

        def settings(key):
            shared.opts.add_option(key, None)
    ''',
    "preload.py": '''
        def preload(parser):
            parser.add_argument("--fake-flag", action="store_true")
    ''',
    "javascript/fake.js": "// dom\n",
    "install.py": "",
    "CHANGELOG.md": "# Changelog\n\n## v9.9.9 — 새 것\n\n### 세부\n\n## v9.9.8 — 옛 것\n",
}


class TestFakeExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "forge_sam3_extension"
        for rel, text in _FILES.items():
            target = cls.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        cls.src = ExtensionSource(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_extension_dir_detection(self):
        self.assertTrue(is_extension_dir(self.root))
        self.assertFalse(is_extension_dir(self.root / "scripts"))

    def test_environment_override_accepts_root_or_scripts_dir(self):
        for configured in (self.root, self.root / "scripts"):
            with self.subTest(configured=str(configured)):
                env = {"AISTUDIO_FORGE_EXTENSION_DIR": str(configured)}
                self.assertEqual(extension_dir_candidates(env)[0], self.root)
                self.assertEqual(find_installed_extension(env), self.root)

    def test_invalid_environment_override_does_not_fall_back(self):
        """오타 난 경로가 조용히 다른 설치(알려진 Forge 폴더)로 바뀌면 엉뚱한 사본을 검사한다 — None 이어야 한다."""
        env = {"AISTUDIO_FORGE_EXTENSION_DIR": str(self.root.parent / "typo" / "forge_sam3_extension")}
        self.assertIsNone(find_installed_extension(env))

    def test_version_and_changelog(self):
        self.assertEqual(self.src.version(), "9.9.9")
        self.assertEqual(self.src.changelog_versions(), [("9.9.9", "새 것"), ("9.9.8", "옛 것")])

    def test_script_title_follows_lazy_package_import(self):
        scripts = self.src.scripts()
        self.assertEqual(list(scripts), ["Fake Tool"])  # show() 가 AlwaysVisible 이 아닌 스크립트는 뺀다
        info = scripts["Fake Tool"]
        self.assertEqual(info.file, "scripts/fake.py")
        # ui() 는 img2img 에서 [] 를 먼저 돌려준다 — 모드마다 따로 읽고, 두 모드가 달라 공통 계약(ui_return)은 없다
        self.assertEqual(info.ui_returns, {False: ("enabled_box", "mode", "strength", "files", "hidden", "other"),
                                           True: ()})
        self.assertIsNone(info.ui_return)
        self.assertEqual(info.arg_names, ("enabled", "mode"))
        self.assertEqual(info.arg_defaults, {"enabled": False, "mode": "fake_keep"})
        self.assertEqual(format_index_set(info.api_reads), "0,2")

    def test_routes_follow_constants_fstrings_and_locals(self):
        self.assertEqual(self.src.routes(), {
            "GET /fake-notes/items": "sam3ext/routes.py",
            "PUT /fake-notes/items/{item_id}": "sam3ext/routes.py",
            "DELETE /fake-notes/items/{item_id}": "sam3ext/routes.py",
            "GET /fake-notes/ping": "sam3ext/routes.py",
        })

    def test_options_and_unresolved_keys_are_reported(self):
        self.assertEqual(self.src.option_keys(), {"fake_keep": "scripts/fake.py",
                                                  "fake_literal": "scripts/fake.py"})
        self.assertTrue(any(item.startswith("scripts/bad.py:") for item in self.src.unresolved),
                        self.src.unresolved)

    def test_axis_labels_follow_loop_variables(self):
        self.assertEqual(set(self.src.axis_labels()), {"[Fake] Enable", "[Fake] Strength", "[Fake] Mode"})

    def test_cli_flags_functions_and_units(self):
        self.assertEqual(self.src.cli_flags(), {"--fake-flag": "preload.py"})
        self.assertIn("scripts/fake.py", self.src.function_names()["handle_fake_click"])
        self.assertEqual(self.src.module_units(), {
            "scripts/fake.py", "scripts/bad.py", "sam3ext/__init__.py", "sam3ext/__version__.py",
            "sam3ext/core.py", "sam3ext/routes.py", "sam3ext/args.py", "sam3ext/tools/", "javascript/fake.js",
            "preload.py", "install.py",
        })

    def test_process_payload_and_state_reads(self):
        self.assertEqual(self.src.dict_keys_assigned("scripts/fake.py", "process", "payload"), {"a", "b"})
        self.assertEqual(self.src.state_reads("scripts/fake.py", "process", "state", ("_xyz_or",)),
                         {"sam3_enable", "mode"})

    def test_ui_components_follow_constants_locals_and_fallbacks(self):
        info = self.src.scripts()["Fake Tool"]
        self.assertIsNone(info.ui_components)          # 모드마다 반환이 달라 공통 컴포넌트 목록이 없다
        self.assertEqual(info.ui_components_by_mode[True], ())
        components = info.ui_components_by_mode[False]
        self.assertEqual(components, (
            {"value": False},        # InputAccordion(label=지역 변수) 와 폴백 Checkbox('Enable') — 라벨이 달라 뺀다
            {"label": "Mode", "choices": ("Keep", "Drop"), "value": "Keep"},   # 상수 슬라이스·인덱스
            {"label": "[Fake] Mode strength", "minimum": -2.0, "maximum": 2.0, "value": 0.5, "step": 0.1},
            {"label": "Files", "value": "None"},          # choices 는 실행 시점 호출 — 모른다
            {"value": False},                             # label 키워드가 없다 — Gradio 기본값을 흉내 내지 않는다
            {},                                           # 모르는 함수가 만든 컴포넌트 — 아무 필드도 믿지 않는다
        ))

    def test_sam3args_defaults_and_constraints(self):
        info = self.src.sam3args()
        self.assertEqual(info["fields"], ("sam3_mode", "sam3_threshold", "sam3_blur", "sam3_name"))
        self.assertEqual(info["literals"], {"sam3_mode": ("A", "B")})
        self.assertEqual(info["defaults"], {"sam3_mode": "A", "sam3_threshold": 0.4, "sam3_blur": 4, "sam3_name": "x"})
        self.assertEqual(info["constraints"], {"sam3_threshold": (0.0, 1.0), "sam3_blur": (0, None)})
        self.assertEqual(info["numeric_bounds"], {"sam3_threshold": (0.0, 1.0, False), "sam3_blur": (0, None, True)})

    def test_module_constant(self):
        self.assertEqual(self.src.module_constant("sam3ext/core.py", "BASE_PATH"), "/fake-notes")
        with self.assertRaises(KeyError):
            self.src.module_constant("sam3ext/core.py", "MISSING")


_PROBE_SCRIPT = '''
from modules import scripts


class Probe(scripts.Script):
    def title(self):
        return "Probe"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
@BODY@
'''


class TestUiReturnPerMode(unittest.TestCase):
    """ui() 의 모든 return 갈래를 모드별로 읽는다 — 마지막 return 만 보면 한 모드의 인자 증감을 놓친다."""

    TXT, IMG = False, True
    A = 'a = gr.Checkbox(label="A", value=False)'
    B = 'b = gr.Slider(label="B", minimum=0.0, maximum=1.0, value=0.5)'

    def _scan(self, body: str):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "forge_sam3_extension"
            (root / "sam3ext").mkdir(parents=True)
            (root / "scripts").mkdir()
            code = _PROBE_SCRIPT.replace("@BODY@", textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8))
            (root / "scripts" / "probe.py").write_text(code, encoding="utf-8")
            src = ExtensionSource(root)
            return src.scripts()["Probe"], list(src.unresolved)

    def test_img2img_early_return_that_drops_an_arg_is_its_own_contract(self):
        info, unresolved = self._scan(f"""
            {self.A}
            {self.B}
            hires = gr.Checkbox(label="Hires", value=False)
            if is_img2img:
                return [a, b]
            return [a, b, hires]
        """)
        self.assertEqual(info.ui_returns, {self.TXT: ("a", "b", "hires"), self.IMG: ("a", "b")})
        self.assertIsNone(info.ui_return)        # 한 계약으로 비교할 수 없다 — 모드별로 비교해야 한다
        self.assertIsNone(info.ui_components)
        self.assertEqual(info.ui_components_by_mode[self.IMG], (
            {"label": "A", "value": False}, {"label": "B", "minimum": 0.0, "maximum": 1.0, "value": 0.5}))
        self.assertEqual(len(info.ui_components_by_mode[self.TXT]), 3)
        self.assertEqual(unresolved, [])

    def test_mode_condition_forms_inside_with_and_else(self):
        for cond in ("not is_img2img", "is_img2img is False", "is_img2img == False", "is_img2img != True",
                     "is_img2img is not True", "False == is_img2img", "not is_img2img and True",
                     "not (is_img2img or False)"):
            with self.subTest(cond=cond):
                info, unresolved = self._scan(f"""
                    with gr.Accordion("x", open=False):
                        {self.A}
                        if {cond}:
                            extra = gr.Checkbox(label="Extra", value=True)
                            return [a, extra]
                        else:
                            return [a]
                """)
                self.assertEqual(info.ui_returns, {self.TXT: ("a", "extra"), self.IMG: ("a",)})
                self.assertEqual(unresolved, [])

    def test_same_list_on_every_branch_is_one_contract(self):
        info, unresolved = self._scan("""
            if is_img2img:
                return []
            with gr.Group(visible=False):
                btn = gr.Button(value="x")
            return []
        """)
        self.assertEqual(info.ui_returns, {self.TXT: (), self.IMG: ()})
        self.assertEqual(info.ui_return, ())
        self.assertEqual(info.ui_components, ())
        self.assertEqual(unresolved, [])

    def test_unknown_condition_with_different_lists_is_unresolved(self):
        info, unresolved = self._scan(f"""
            {self.A}
            if HAS_EXTRA:
                return [a, a]
            return [a]
        """)
        self.assertEqual(info.ui_returns, {self.TXT: None, self.IMG: None})
        self.assertIsNone(info.ui_return)
        self.assertEqual(len(unresolved), 1, unresolved)
        self.assertIn("scripts/probe.py", unresolved[0])
        self.assertIn("Probe.ui()", unresolved[0])

    def test_only_the_ambiguous_mode_is_unresolved(self):
        info, unresolved = self._scan(f"""
            {self.A}
            {self.B}
            if is_img2img and HAS_EXTRA:
                return [a]
            return [a, b]
        """)
        self.assertEqual(info.ui_returns, {self.TXT: ("a", "b"), self.IMG: None})
        self.assertEqual(len(unresolved), 1, unresolved)
        self.assertIn("img2img", unresolved[0])
        self.assertNotIn("txt2img", unresolved[0])

    def test_code_after_a_fully_returning_if_is_unreachable(self):
        info, unresolved = self._scan(f"""
            {self.A}
            {self.B}
            if HAS_EXTRA:
                return [a]
            else:
                return [a]
            return [a, b]
        """)
        self.assertEqual(info.ui_return, ("a",))
        self.assertEqual(unresolved, [])

    def test_dynamic_return_and_nested_function_returns(self):
        info, unresolved = self._scan("""
            components = build(is_img2img)
            return components
        """)
        self.assertEqual(info.ui_returns, {self.TXT: None, self.IMG: None})
        self.assertEqual(unresolved, [])     # 동적 반환은 레지스트리의 ui_return=None 과 비교한다(모호함이 아니다)
        info, unresolved = self._scan(f"""
            def helper():
                return [1, 2, 3]
            {self.A}
            return [a]
        """)
        self.assertEqual(info.ui_return, ("a",))
        self.assertEqual(unresolved, [])


class TestUiComponentDiff(unittest.TestCase):
    """compare_ui_components — AST 컴포넌트 필드 ↔ script-info 인자."""

    LIVE = [
        {"label": "Enable", "value": False, "minimum": None, "maximum": None, "step": None, "choices": None},
        {"label": "Mode", "value": "Keep", "minimum": None, "maximum": None, "step": None,
         "choices": [["Keep", "Keep"], ["Drop", "Drop"]]},
        {"label": "Amount", "value": 0.1, "minimum": -5.0, "maximum": 5.0, "step": 0.01, "choices": None},
        {"label": "Files", "value": "a.safetensors", "minimum": None, "maximum": None, "step": None,
         "choices": ["None", "a.safetensors"]},
    ]

    def test_same_values_including_choice_pairs_and_json_lists(self):
        components = ({"label": "Enable", "value": False}, {"label": "Mode", "value": "Keep", "choices": ("Keep", "Drop")},
                      {"label": "Amount", "value": 0.1, "minimum": -5.0, "maximum": 5, "step": 0.01},
                      {"label": "Files"})
        diffs, unread = compare_ui_components(components, self.LIVE, runtime_indices=(3,))
        self.assertEqual(diffs, [])
        self.assertEqual(unread, {})   # 실행 시점 인덱스의 value·choices 는 요구하지 않는다

    def test_changed_range_and_unread_fields_are_reported(self):
        components = ({"value": False}, {"label": "Mode", "value": "Keep", "choices": ("Keep",)},
                      {"label": "Amount", "value": 0.1, "minimum": -3.0, "maximum": 3.0, "step": 0.01}, {"label": "Files"})
        diffs, unread = compare_ui_components(components, self.LIVE)
        self.assertEqual(diffs, [(1, "choices", ("Keep",), [["Keep", "Keep"], ["Drop", "Drop"]]),
                                 (2, "minimum", -3.0, -5.0), (2, "maximum", 3.0, 5.0)])
        self.assertEqual(unread, {0: ("label",), 3: ("value", "choices")})


class TestIndexSets(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(format_index_set([5, 0, 1, 2, 9]), "0-2,5,9")
        self.assertEqual(parse_index_set("0-2,5,9"), frozenset({0, 1, 2, 5, 9}))
        self.assertEqual(format_index_set([]), "")


if __name__ == "__main__":
    unittest.main()
