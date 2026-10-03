"""Extra Schedulers 값 칸(frontend/src/utils/extraSchedulers.ts · components/params/ExtraSchedulerFields.vue)은
core/extra_schedulers 의 거울이다.

위젯 키·앱 기본값·μ/β 범위(= 확장 슬라이더, 레지스트리 픽스처 핀 xsched_ui_*)·모드 키·스케줄러 이름/라벨/README 별칭·
값을 읽는 스케줄러·flow 에 맞지 않는 스케줄러·'같은 스케줄러'·패스 판정 위젯 id 를 파이썬 표와 대조한다(하나라도 다르면
칸이 파이썬 프록시와 다른 값을 쓰거나, 블록을 보내는 패스와 다른 때에 보인다). 이름 풀이와 패스 판정은 두 구현이 같은
골든 파일(utils/extraSchedulers.cases.json)을 읽는다 — 여기서는 파이썬 쪽, vitest 가 TS 쪽.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from core import extra_schedulers as xs
from core.sam_extra_capabilities import FEATURE_FLAGS

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "frontend" / "src"
TS = SRC / "utils" / "extraSchedulers.ts"
CASES = SRC / "utils" / "extraSchedulers.cases.json"
FIELDS = SRC / "components" / "params" / "ExtraSchedulerFields.vue"
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


class ExtraSchedulersCardMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.cases = json.loads(CASES.read_text(encoding="utf-8"))

    def test_widget_ids_and_defaults(self):
        ids = dict(re.findall(r"^\s+(\w+): '(_xsched_\w+)',$", _block(self.ts, "export const WIDGET_IDS"), re.M))
        self.assertEqual(ids, {key: xs.widget_id(key) for key in xs.WIDGET_KEYS})
        self.assertEqual(list(ids), list(xs.WIDGET_KEYS))
        defaults = dict(re.findall(r"(\w+): '([^']*)'", _block(self.ts, "export const DEFAULTS: Readonly<XschedValues> = {")))
        self.assertEqual(defaults, xs.widget_values(xs.APP_DEFAULTS))

    def test_ranges_and_modes(self):
        self.assertEqual(_range(self.ts, "MU_RANGE"),
                         {"min": xs.LAPLACE_MU_MIN, "max": xs.LAPLACE_MU_MAX, "step": xs.LAPLACE_MU_STEP})
        self.assertEqual(_range(self.ts, "BETA_RANGE"),
                         {"min": xs.LAPLACE_BETA_MIN, "max": xs.LAPLACE_BETA_MAX, "step": xs.LAPLACE_BETA_STEP})
        modes = tuple(re.findall(r"\{ key: '(\w+)', label: '[^']+' \}", _block(self.ts, "export const MODE_OPTIONS")))
        self.assertEqual(modes, xs.MODES)
        words = re.search(r"\['sigmas', 'sigma list', 'sigma_list', 'list', 'sigma'\]\.includes", self.ts)
        self.assertIsNotNone(words, "readValues 의 시그마 목록 모드 표기가 파이썬 coerce_mode 와 같아야 한다")
        for word in ("sigmas", "sigma list", "sigma_list", "list", "sigma"):
            self.assertEqual(xs.coerce_mode(word), xs.MODE_SIGMAS)

    def test_scheduler_table(self):
        rows = tuple(re.findall(r"\{ name: '(\w+)', label: '([^']+)', alias: '([^']+)' \}",
                                _block(self.ts, "export const SCHEDULERS: readonly SchedulerRow[] = [")))
        self.assertEqual(rows, tuple((name, xs.SCHEDULER_LABELS[name], xs.README_LABELS[name])
                                     for name in xs.SCHEDULER_NAMES))
        self.assertEqual(_list(self.ts, "VALUE_SCHEDULERS"), xs.VALUE_SCHEDULERS)
        self.assertEqual(_list(self.ts, "FLOW_UNSUITABLE"), xs.FLOW_UNSUITABLE)
        self.assertIn(f"export const USE_SAME_SCHEDULER = '{xs.USE_SAME_SCHEDULER}'", self.ts)

    def test_pass_widgets_are_the_python_ids_in_order(self):
        block = _block(self.ts, "export const PASS_WIDGETS = {")
        self.assertEqual(tuple(re.findall(r"'((?:_|[a-z])[\w]*)'", block)), xs.pass_widget_ids())
        self.assertIn("const MASK_ONLY = 'mask only'", self.ts)

    def test_golden_cases(self):
        for item in self.cases["resolve"]:
            with self.subTest(text=item["text"]):
                self.assertEqual(xs.resolve_scheduler(item["text"]), item["name"])
        for item in self.cases["passes"]:
            with self.subTest(case=item["name"]):
                got = [[use.pass_id, use.scheduler] for use in xs.passes_from_widgets(item["widgets"])]
                self.assertEqual(got, item["passes"])
        self.assertGreaterEqual(len(self.cases["passes"]), 15)

    def test_old_forge_note_reads_the_feature_flag(self):
        self.assertIn("extra_schedulers", FEATURE_FLAGS)
        self.assertIn("caps.features?.extra_schedulers !== true", self.ts)

    def test_placement_under_the_sampler_scheduler_row(self):
        """계획 R8: 파라미터 카드의 Sampler/Scheduler 줄 바로 아래(App.vue 를 거치지 않는다 — 스토어를 직접 읽는다)."""
        basic = BASIC.read_text(encoding="utf-8")
        row = basic.index('<CustomSelect v-model="storeWidgets.scheduler_combo"')
        fields = basic.index("<ExtraSchedulerFields />")
        steps = basic.index("<label>스텝</label>")
        self.assertLess(row, fields)
        self.assertLess(fields, steps)
        self.assertEqual(basic[row:fields].count('<div class="ext-row">'), 0, "Sampler/Scheduler 줄 바로 다음이어야 한다")
        self.assertIn("import ExtraSchedulerFields from './ExtraSchedulerFields.vue'", basic)
        self.assertNotIn("ExtraSchedulerFields", APP.read_text(encoding="utf-8"))
        fields_src = FIELDS.read_text(encoding="utf-8")
        for name in ("WIDGET_IDS", "MU_RANGE", "BETA_RANGE", "MODE_OPTIONS", "passesUsing", "ERROR_PROPERTY"):
            self.assertIn(name, fields_src)
        self.assertNotIn("requestAction", fields_src, "새 브리지 액션 없음 — 값은 위젯 스토어로만")


if __name__ == "__main__":
    unittest.main()
