"""sam-extra 가져오기(ui/sam_extra_import — 액션 ``sam_extra_import {text, only?}``, 계획 X3c).

붙여 넣은 글(이미지 infotext 전체·``Key: value`` 줄·맨 값)에서 Colorcraft(자기·업스트림·aoleg 포크 형식 — core/colorcraft 의
붙여 넣기 규칙, 설치된 확장과의 대조는 tests/test_colorcraft_spec_parity.py)와 Anima SPEED(요약 — core/anima_speed
붙여 넣기 규칙) 값을 찾아 카드 위젯에 쓰고 토스트 하나로 알린다. 글에 그 기능의 키가 없으면 그 카드는 그대로 둔다.
Qt 없이: 위젯은 가짜 프록시, 토스트는 가짜 시그널.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from core import anima_speed as asp
from core import colorcraft as cc
from tests._colorcraft_corpus import FORK_TEXTS, UPSTREAM_TEXTS
from tests.test_anima_speed_ui import speed_widgets
from tests.test_colorcraft_settings import card_state, cc_widgets
from ui import anima_speed_ui as aui
from ui import colorcraft_ui as cui
from ui import sam_extra_import as imp

REPO = Path(__file__).resolve().parent.parent
OWN = "v1;mods=I,III;masking=1;masks=M1;combos=;I.kind=Luma;I.mask=M1;I.exposure=0.3;III.kind=Punch;III.pass=Hires"
SPEED = ("mode=respace; threshold=neo_shift; preset=flux; scales=0.25,0.5,1; delta=0.02; divisor=1.1; manual=0.7; "
         "adaptive=False; transform=fft; seed=7; hires=True")
INFOTEXT = ("1girl, solo\nNegative prompt: worst quality\nSteps: 28, Sampler: Euler, Seed: 7, Size: 832x1216, "
            f"SAM Extra Colorcraft: {json.dumps(OWN)}, SAM Extra Colorcraft status: \"base: krea2 (Wan21); mods I\", "
            f"Anima SPEED: {json.dumps(SPEED)}, Version: f2")


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


def host(**colorcraft):
    return SimpleNamespace(colorcraft_widgets=cc_widgets(**colorcraft), anima_speed_widgets=speed_widgets(),
                           vue_bridge=SimpleNamespace(showNotification=_Signal()))


class ParseTests(unittest.TestCase):
    def test_whole_infotext_finds_both(self):
        plan = imp.parse_import(INFOTEXT)
        self.assertEqual(plan.found, ("colorcraft", "speed"))
        self.assertEqual(plan.colorcraft, cc.parse_own(OWN))
        self.assertEqual(plan.speed[asp.INFOTEXT_KEY], SPEED)
        only = imp.parse_import(INFOTEXT, "colorcraft")
        self.assertEqual((only.found, only.asked), (("colorcraft",), ("colorcraft",)))
        self.assertEqual(imp.parse_import(INFOTEXT, "speed").found, ("speed",))

    def test_key_lines_keep_commas_and_semicolons(self):
        plan = imp.parse_import(f"SAM Extra Colorcraft: {OWN}\nAnima SPEED: {SPEED}")
        self.assertEqual(plan.colorcraft, cc.parse_own(OWN))
        self.assertEqual(plan.speed[asp.INFOTEXT_KEY], SPEED)                # 'scales=0.25,0.5,1' 이 잘리지 않는다
        quoted = imp.parse_import(f"Anima SPEED: {json.dumps(SPEED)}")
        self.assertEqual(quoted.speed[asp.INFOTEXT_KEY], SPEED)
        for text in (*UPSTREAM_TEXTS[:3], *FORK_TEXTS[1:4]):
            with self.subTest(text=text[:40]):
                self.assertEqual(imp.parse_import(f"Colorcraft: {text}").colorcraft,
                                 cc.from_infotext({cc.FOREIGN_KEY: text}))

    def test_bare_values_use_the_extension_format_order(self):
        self.assertEqual(imp.parse_import(OWN).colorcraft, cc.parse_own(OWN))
        self.assertEqual(imp.parse_import(UPSTREAM_TEXTS[0]).colorcraft, cc.from_value(UPSTREAM_TEXTS[0]))
        self.assertEqual(imp.parse_import(FORK_TEXTS[2]).colorcraft, cc.parse_fork(FORK_TEXTS[2]))
        bare_speed = imp.parse_import(SPEED)
        self.assertEqual((bare_speed.found, bare_speed.speed), (("speed",), {asp.INFOTEXT_KEY: SPEED}))
        self.assertEqual(imp.parse_import(SPEED, "colorcraft").found, ())
        self.assertEqual(imp.parse_import(OWN, "speed").found, ())

    def test_nothing_and_errors(self):
        for text in ("", "   ", None, "a cat\nSteps: 20, Sampler: Euler", "hello: world", "mode is not a key",
                     "MODE(I)"):
            with self.subTest(text=text):
                self.assertEqual(imp.parse_import(text).found, ())
        with self.assertRaises(ValueError):
            imp.parse_import(OWN, "dora")
        huge = OWN + ";" + "x" * (imp.MAX_TEXT * 2)
        self.assertIsNotNone(imp.parse_import(huge).colorcraft)                  # 한도 뒤는 읽지 않는다

    def test_feature_keys_are_the_core_constants(self):
        self.assertEqual(imp.FEATURE_KEYS, {"colorcraft": (cc.INFOTEXT_KEY, cc.FOREIGN_KEY),
                                            "speed": (asp.INFOTEXT_KEY,)})


class ImportTests(unittest.TestCase):
    def test_both_cards_come_back_and_the_app_toggles_stay(self):
        h = host(img2img="true")
        h.anima_speed_widgets["apply_img2img"].setText("true")
        plan = imp.import_infotext(h, {"text": INFOTEXT})
        self.assertEqual(plan.found, ("colorcraft", "speed"))
        self.assertEqual(cui.get_settings(h), {"enabled": "true", "masking": "true", "img2img": "true",
                                               "state": cc.dump_state(cc.parse_own(OWN))})
        speed = aui.current_settings(h)
        self.assertEqual((speed.enabled, speed.mode, speed.preset, speed.seed, speed.hires, speed.apply_img2img),
                         (True, "respace", "flux", 7, True, True))
        [(level, message)] = h.vue_bridge.showNotification.calls
        self.assertEqual(level, "success")
        self.assertIn("Colorcraft(켠 수정자: I Luma ➜ M1, III Punch (Hires) · 쓰는 마스크: M1)", message)
        self.assertIn("Anima SPEED(respace", message)

    def test_only_one_feature_leaves_the_other_card_alone(self):
        h = host(paths={"II.contrast": 1.0}, enabled="true")
        before = cui.get_settings(h)
        imp.import_infotext(h, {"text": INFOTEXT, "only": "speed"})
        self.assertEqual(cui.get_settings(h), before)
        self.assertTrue(aui.current_settings(h).enabled)
        h2 = host()
        imp.import_infotext(h2, {"text": f"SAM Extra Colorcraft: {OWN}"})        # SPEED 키가 없으면 SPEED 는 그대로
        self.assertEqual(aui.current_settings(h2), asp.APP_DEFAULTS)
        self.assertEqual(cui.get_settings(h2)["enabled"], "true")

    def test_nothing_found_warns_once(self):
        h = host()
        imp.import_infotext(h, {"text": "a cat\nSteps: 20"})
        imp.import_infotext(h, {"text": SPEED, "only": "colorcraft"})
        imp.import_infotext(h, {"text": OWN, "only": "nope"})
        imp.import_infotext(h, "not a payload")
        levels = [level for level, _message in h.vue_bridge.showNotification.calls]
        self.assertEqual(levels, ["warning"] * 4)
        messages = [message for _level, message in h.vue_bridge.showNotification.calls]
        self.assertEqual(messages[0], imp.NOTHING_MESSAGE)
        self.assertTrue(messages[1].startswith("가져올 Colorcraft 값이 없습니다"))
        self.assertIn("nope", messages[2])
        self.assertEqual(cui.get_settings(h)["state"], card_state())
        self.assertEqual(aui.current_settings(h), asp.APP_DEFAULTS)

    def test_hosts_without_cards_import_nothing(self):
        bare = SimpleNamespace(vue_bridge=SimpleNamespace(showNotification=_Signal()))
        imp.import_infotext(bare, {"text": INFOTEXT})
        self.assertEqual([level for level, _m in bare.vue_bridge.showNotification.calls], ["warning"])
        imp.import_infotext(SimpleNamespace(), {"text": INFOTEXT})               # 브리지도 없으면 조용히


class WiringTests(unittest.TestCase):
    def test_dispatcher_routes_the_action(self):
        source = (REPO / "ui" / "generator_main.py").read_text(encoding="utf-8")
        branch = source[source.index("elif action == 'sam_extra_import':"):][:300]
        self.assertIn("from ui.sam_extra_import import import_infotext", branch)
        self.assertIn("(self, payload)", branch)

    def test_frontend_keys_mirror_python(self):
        ts = (REPO / "frontend" / "src" / "utils" / "samExtraImport.ts").read_text(encoding="utf-8")
        block = ts[ts.index("export const IMPORT_KEYS"):]
        block = block[:block.index("}")]
        keys = {name: tuple(re.findall(r"'([^']+)'", values)) for name, values in re.findall(r"(\w+): \[([^\]]*)\]", block)}
        self.assertEqual(keys, imp.FEATURE_KEYS)
        self.assertIn("requestAction('sam_extra_import', payload)", ts)
        types = (REPO / "frontend" / "src" / "types" / "bridge.d.ts").read_text(encoding="utf-8")
        self.assertIn("'sam_extra_import'", types)
        self.assertIn("sam_extra_import: SamExtraImportPayload", types)
        self.assertIn("export type SamExtraImportFeature = 'colorcraft' | 'speed'", types)
        self.assertEqual(imp.FEATURES, ("colorcraft", "speed"))

    def test_entry_points_use_the_generic_action(self):
        """카드의 붙여 넣기 칸(Colorcraft 만)과 PNG Info·갤러리 단추(이미지에 있는 기능 전부)가 같은 액션을 쓴다."""
        src = REPO / "frontend" / "src"
        card = (src / "components" / "params" / "ColorcraftCard.vue").read_text(encoding="utf-8")
        self.assertIn("importSamExtra(pasteText.value, 'colorcraft')", card)
        for view in ("PngInfoView.vue", "GalleryView.vue"):
            with self.subTest(view=view):
                text = (src / "views" / view).read_text(encoding="utf-8")
                self.assertIn("from '../utils/samExtraImport'", text)
                self.assertIn('v-if="importable.length"', text)
                self.assertIn("importSamExtra(importText(", text)


if __name__ == "__main__":
    unittest.main()
