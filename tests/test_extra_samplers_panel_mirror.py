"""Extra Samplers ER SDE 칸(frontend/src/utils/extraSamplers.ts · components/params/ExtraSamplersPanel.vue)은
core/extra_samplers 의 거울이다.

위젯 키·앱 기본값(문자열과 값)·차수/η/잡음 구간 범위(= 확장 슬라이더 — script-info 픽스처 0~4번과도 직접 대조: 슬라이더 넷과
체크 칸 하나)·샘플러 18개의 표기/ComfyUI 이름/별칭/종류·값을 읽는 묶음·'같은 샘플러'·패스 판정 위젯 id·스냅샷 키·옛 sam-extra
문구를 파이썬 표와 대조한다(하나라도 다르면 칸이 파이썬 프록시와 다른 값을 쓰거나, 블록 값을 읽는 패스와 다른 때에 보인다).
이름 풀이·패스 판정·칸 값 정규화와 요약·잡음 구간 판정은 두 구현이 같은 골든 파일(utils/extraSamplers.cases.json — 파이썬이
만든 기대값)을 읽는다 — 여기서는 파이썬 쪽, vitest 가 TS 쪽.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unittest
from pathlib import Path

from core import extra_samplers as xsa
from core.sam_extra_capabilities import FEATURE_FLAGS, TITLE_EXTRA_SAMPLERS, SamExtraCapabilities

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "frontend" / "src"
TS = SRC / "utils" / "extraSamplers.ts"
CASES = SRC / "utils" / "extraSamplers.cases.json"
FIXTURE = REPO / "tests" / "fixtures" / "sam_extra_script_info.json"
PANEL = SRC / "components" / "params" / "ExtraSamplersPanel.vue"
BASIC = SRC / "components" / "params" / "ParamsBasicCard.vue"
APP = SRC / "App.vue"


def _block(text: str, head: str) -> str:
    match = re.search(re.escape(head) + r"(.*?)\n[}\]]", text, re.S)
    if match is None:
        raise AssertionError(f"{head!r} 블록을 찾지 못했다")
    return match.group(1)


def _range(text: str, name: str) -> dict:
    match = re.search(rf"export const {name} = \{{ min: (-?[\d.]+), max: (-?[\d.]+), step: (-?[\d.]+) \}} as const",
                      text)
    if match is None:
        raise AssertionError(f"{name} 를 찾지 못했다")
    return {key: float(value) for key, value in zip(("min", "max", "step"), match.groups())}


def _list(text: str, name: str) -> tuple:
    match = re.search(rf"export const {name}: readonly \w+\[\] = \[([^\]]*)\]", text)
    if match is None:
        raise AssertionError(f"{name} 를 찾지 못했다")
    return tuple(re.findall(r"'([^']+)'", match.group(1)))


def _literal(token: str):
    if token in ("true", "false"):
        return token == "true"
    return float(token)


class ExtraSamplersPanelMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.cases = json.loads(CASES.read_text(encoding="utf-8"))

    def test_widget_ids_and_defaults(self):
        ids = dict(re.findall(r"^\s+(\w+): '(_xsamp_\w+)',$", _block(self.ts, "export const WIDGET_IDS"), re.M))
        self.assertEqual(ids, {key: xsa.widget_id(key) for key in xsa.WIDGET_KEYS})
        self.assertEqual(list(ids), list(xsa.WIDGET_KEYS))
        defaults = dict(re.findall(r"(\w+): '([^']*)'", _block(self.ts, "export const DEFAULTS: Readonly<XsampValues> = {")))
        self.assertEqual(defaults, xsa.widget_values(xsa.APP_DEFAULTS))
        settings = dict(re.findall(r"(\w+): (-?[\d.]+|true|false)",
                                   _block(self.ts, "export const DEFAULT_SETTINGS: Readonly<SamplerSettings> = {")))
        self.assertEqual({key: _literal(value) for key, value in settings.items()}, dataclasses.asdict(xsa.APP_DEFAULTS))
        self.assertIn(f"export const ARG_COUNT = {len(xsa.ARG_NAMES)}\n", self.ts)
        self.assertIn(f"export const SCRIPT_TITLE = '{TITLE_EXTRA_SAMPLERS}'\n", self.ts)

    def test_ranges_follow_python_and_the_live_sliders(self):
        stage = {"min": xsa.MAX_STAGE_MIN, "max": xsa.MAX_STAGE_MAX, "step": xsa.MAX_STAGE_STEP}
        eta = {"min": xsa.ETA_MIN, "max": xsa.ETA_MAX, "step": xsa.ETA_STEP}
        noise = {"min": xsa.NOISE_PERCENT_MIN, "max": xsa.NOISE_PERCENT_MAX, "step": xsa.NOISE_PERCENT_STEP}
        self.assertEqual(_range(self.ts, "STAGE_RANGE"), stage)
        self.assertEqual(_range(self.ts, "ETA_RANGE"), eta)
        self.assertEqual(_range(self.ts, "NOISE_RANGE"), noise)
        self.assertEqual(_list(self.ts, "STAGE_OPTIONS"),
                         tuple(str(n) for n in range(xsa.MAX_STAGE_MIN, xsa.MAX_STAGE_MAX + 1, xsa.MAX_STAGE_STEP)))
        # 확장 ui() 슬라이더 넷과 체크 칸 하나(sam-extra v0.33.0 script-info — txt2img·img2img 둘 다)
        scripts = json.loads(FIXTURE.read_text(encoding="utf-8"))["scripts"]
        entries = [s for s in scripts if s["name"].lower() == xsa.SCRIPT_NAME.lower()]
        self.assertEqual({s["is_img2img"] for s in entries}, {False, True})
        expected = ((stage, xsa.DEFAULT_MAX_STAGE), (eta, xsa.DEFAULT_ETA), (None, xsa.DEFAULT_NOISE_WINDOW),
                    (noise, xsa.DEFAULT_NOISE_START), (noise, xsa.DEFAULT_NOISE_END))
        for entry in entries:
            self.assertEqual(len(entry["args"]), len(xsa.ARG_NAMES))
            for index, (bounds, value) in enumerate(expected):
                arg = entry["args"][index]
                with self.subTest(img2img=entry["is_img2img"], index=index):
                    self.assertEqual(arg["value"], value)
                    self.assertIs(type(arg["value"]), type(value), "체크 칸은 bool, 차수는 int, 나머지는 float")
                    got = {"min": arg["minimum"], "max": arg["maximum"], "step": arg["step"]}
                    self.assertEqual(got, bounds or {"min": None, "max": None, "step": None})

    def test_sampler_table(self):
        rows = tuple(re.findall(r"\{ label: '([^']+)', name: '(\w+)', kind: '(\w*)'(?:, alias: '(\w+)')? \}",
                                _block(self.ts, "export const SAMPLERS: readonly SamplerRow[] = [")))
        expected = tuple((label, aliases[0], xsa.sampler_kind(label), aliases[1] if len(aliases) > 1 else "")
                         for label, aliases in xsa.ALIASES.items())
        self.assertEqual(rows, expected)
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(len(aliases) <= 2 for aliases in xsa.ALIASES.values()), "TS 줄은 별칭 둘까지만 담는다")
        self.assertEqual(_list(self.ts, "ER_SDE_LABELS"), xsa.ER_SDE_LABELS)
        self.assertEqual(_list(self.ts, "ETA_LABELS"), xsa.ETA_LABELS)
        self.assertEqual(xsa.WINDOW_LABELS, xsa.ETA_LABELS)
        self.assertEqual(_list(self.ts, "CFG_PP_LABELS"), xsa.CFG_PP_LABELS)
        self.assertEqual(_list(self.ts, "FLOW_ODE_LABELS"), xsa.FLOW_ODE_LABELS)
        kinds = re.search(r"export type SamplerKind = (.*)\n", self.ts).group(1)
        self.assertEqual(set(re.findall(r"'(\w*)'", kinds)),
                         {xsa.KIND_ER_SDE, xsa.KIND_DY, xsa.KIND_SMEA, xsa.KIND_CFG_PP, xsa.KIND_RESTART, xsa.KIND_OTHER})
        self.assertIn(f"export const USE_SAME_SAMPLER = '{xsa.USE_SAME_SAMPLER}'", self.ts)

    def test_pass_widgets_are_the_python_ids_in_order(self):
        block = _block(self.ts, "export const PASS_WIDGETS = {")
        self.assertEqual(tuple(re.findall(r"'((?:_|[a-z])[\w]*)'", block)), xsa.pass_widget_ids())
        self.assertIn("const MASK_ONLY = 'mask only'", self.ts)

    def test_golden_cases_are_what_python_gives(self):
        for item in self.cases["resolve"]:
            with self.subTest(text=item["text"]):
                self.assertEqual(xsa.resolve_sampler(item["text"]), item["label"])
                self.assertEqual(xsa.sampler_kind(item["text"]), item["kind"])
        self.assertEqual({item["label"] for item in self.cases["resolve"]} - {None}, set(xsa.LABELS),
                         "모든 표기를 한 번 이상 다룬다")
        for item in self.cases["passes"]:
            with self.subTest(case=item["name"]):
                got = [[use.pass_id, use.sampler] for use in xsa.passes_from_widgets(item["widgets"])]
                self.assertEqual(got, item["passes"])
        self.assertGreaterEqual(len(self.cases["passes"]), 20)
        samplers = self.cases["describe_samplers"]
        for item in self.cases["settings"]:
            with self.subTest(settings=item["name"]):
                settings = xsa.parse_settings(item["values"])     # 프록시 글자 → 보낼 값(ui/extra_samplers_ui)
                self.assertEqual(dataclasses.asdict(settings), item["settings"])
                self.assertIs(settings.window_empty, item["window_empty"])
                self.assertEqual({key: xsa.describe(settings, samplers[key]) for key in item["describe"]},
                                 item["describe"])
        self.assertGreaterEqual(len(self.cases["settings"]), 40)
        on, off = xsa.parse_settings({"noise_window": "1"}), xsa.parse_settings({"noise_window": "0"})
        on_eta0 = xsa.parse_settings({"noise_window": "1", "eta": "0"})
        for item in self.cases["window"]:
            with self.subTest(window=item["name"]):
                caps = SamExtraCapabilities(status=item["caps"]["status"], scripts=item["caps"]["scripts"])
                self.assertIs(caps.known, item["caps"]["known"])
                self.assertIs(xsa.window_supported(caps), item["supported"])
                self.assertIs(xsa.window_unsupported(on, caps), item["unsupported_when_on"])
                self.assertIs(xsa.window_unsupported(off, caps), item["unsupported_when_off"])
                self.assertIs(xsa.window_unsupported(on_eta0, caps), item["unsupported_when_on_eta0"])
        self.assertEqual({item["supported"] for item in self.cases["window"]}, {True, False, None})
        # FIX-forge(claude:forge-notices): 구간을 버리는 옛 sam-extra 여도 η 0 이면 경고하지 않는다 — 그 차이가 사례에 있다
        self.assertTrue(any(item["unsupported_when_on"] and not item["unsupported_when_on_eta0"]
                            for item in self.cases["window"]))

    def test_texts_and_feature_flag_shared_with_python(self):
        self.assertIn(f"export const WINDOW_UNSUPPORTED_TEXT = '{xsa.WINDOW_UNSUPPORTED_TEXT}'\n", self.ts)
        # Hires CFG++ 경고(FIX-forge, codex:forge): Hires CFG 칸이 끔이면 Forge 하이레스 패스는 hr_cfg 기본값으로 돈다
        from core.sam_extra_notices import FORGE_DEFAULT_HR_CFG
        self.assertIn(f"export const FORGE_DEFAULT_HR_CFG = {FORGE_DEFAULT_HR_CFG:g}\n", self.ts)
        self.assertIn("export const HIRES_CFG_WIDGET = 'hires_cfg_input'\n", self.ts)
        hires_card = (SRC / "components" / "params" / "HiresFixCard.vue").read_text(encoding="utf-8")
        self.assertIn('v-model="storeWidgets.hires_cfg_input"', hires_card)
        generation = (REPO / "ui" / "generator_generation.py").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertIn("hr_cfg = _widget_float(self.hires_cfg_input, 0.0)\n", generation)
        self.assertIn('if hr_cfg > 0:\n                hr_payload["hr_cfg"] = hr_cfg', generation)
        # IPNDM 힌트(GPU part 2 O2)를 보이는 표기는 파이썬 표기
        self.assertEqual(_list(self.ts, "IPNDM_HINT_LABELS"), (xsa.LABEL_IPNDM, xsa.LABEL_IPNDM_V, xsa.LABEL_UNIPC_BH2))
        self.assertIn("extra_samplers", FEATURE_FLAGS)
        self.assertIn("caps.features?.extra_samplers !== true", self.ts)

    def test_placement_under_the_sampler_scheduler_row_before_extra_schedulers(self):
        """계획 R8: 파라미터 카드의 Sampler/Scheduler 줄 바로 아래, Extra Schedulers 값 칸 앞(줄의 왼쪽→오른쪽 순서). App.vue 를
        거치지 않는다(스토어를 직접 읽는다), 새 브리지 액션 없음, 칸은 TS 표의 키·범위로만 그린다."""
        basic = BASIC.read_text(encoding="utf-8")
        row = basic.index('<CustomSelect v-model="storeWidgets.scheduler_combo"')
        panel = basic.index("<ExtraSamplersPanel />")
        schedulers = basic.index("<ExtraSchedulerFields />")
        self.assertLess(row, panel)
        self.assertLess(panel, schedulers)
        self.assertEqual(basic[row:panel].count("<div"), 0, "Sampler/Scheduler 줄 바로 다음이어야 한다")
        self.assertEqual(basic[panel:schedulers].count("<"), basic[panel:schedulers].count("<!--") + 1,
                         "Extra Schedulers 값 칸 바로 앞이어야 한다")
        self.assertEqual(basic.count("<ExtraSamplersPanel"), 1)
        self.assertIn("import ExtraSamplersPanel from './ExtraSamplersPanel.vue'", basic)
        self.assertNotIn("ExtraSamplersPanel", APP.read_text(encoding="utf-8"))
        panel_src = PANEL.read_text(encoding="utf-8")
        for name in ("WIDGET_IDS", "STAGE_OPTIONS", "ETA_RANGE", "NOISE_RANGE", "passesUsing", "usesEta", "windowEmpty",
                     "windowUnsupported", "WINDOW_UNSUPPORTED_TEXT", "samplerHints", "cardNote", "isDefault"):
            self.assertIn(name, panel_src)
        self.assertNotIn("requestAction", panel_src, "새 브리지 액션 없음 — 값은 위젯 스토어로만")
        for literal in ('min="', 'max="', 'step="'):
            self.assertNotIn(f" {literal}", panel_src, "범위는 TS 표(ETA_RANGE·NOISE_RANGE)로만 — 손으로 적은 숫자 없음")


if __name__ == "__main__":
    unittest.main()
