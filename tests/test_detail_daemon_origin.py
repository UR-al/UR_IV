"""Detail Daemon — 앱이 보내는 값이 원본 노드와 같은 σ 배율을 내는가 (parity_plan §1, DD-A).

기준(사용자 결정): Jonseed/ComfyUI-Detail-Daemon 노드 값을 **그대로** 쓴다. 앱은 노드 단위를 변환 없이 보내고
(프리셋·÷0.1·저장값 이전 없음) 엔진이 ×0.1×cfg 를 곱한다. 노드에 없는 Forge 전용 Hires Pass 는
muerrilla/sd-webui-detail-daemon 을 따른다(기본 끔 = base 패스만, 인자 13 으로 맨 뒤 append).

아래 ``node_make_schedule`` · ``node_sigma_factor`` · ``NODE_INPUTS`` 는 원본(MIT)에서 그대로 옮겼다 — 각 origin 줄 참고.
  ComfyUI-Detail-Daemon  Copyright (c) 2024 Jonseed — MIT License
  sd-webui-detail-daemon Copyright (c) 2024 Sahand Ahmadian — MIT License
황금값은 원본 클론의 코드를 그대로 돌려 얻었다(세션 스크래치 dd_a/golden.py — 원본 두 make_schedule 이 30가지 경우에서
차이 0 인 것도 거기서 확인했다).
numpy 만 쓴다 — torch·ComfyUI·확장을 import 하지 않는다.
"""
from __future__ import annotations

import unittest

import numpy as np

from core import anima_guidance
from core.anima_guidance import (
    DETAIL_DAEMON_SPEC, SCRIPT_DETAIL_DAEMON, build_alwayson, build_args, default_settings,
    describe_active, detail_daemon_hires_note, parse_forge_script_info,
)

ORIGIN_NODE = "Jonseed/ComfyUI-Detail-Daemon@3394e44afea04ed0188fb37b21f0d9952469766b:detail_daemon_node.py"
ORIGIN_FORGE = "muerrilla/sd-webui-detail-daemon@19479998340831d7804fca8efd3f262b54b6373f:scripts/detail_daemon.py"


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:25-67 (make_detail_daemon_schedule, 원문 그대로)
def node_make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


def node_sigma_factor(schedule_value, cfg_scale):
    """σ 에 곱하는 값.

    origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:294-295
        dd_adjustment = get_dd_schedule(...) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
    (스케줄 σ 와 정확히 같은 σ 에서 get_dd_schedule 은 dd_schedule[idx] 를 돌려준다 — :243-250.
     같은 식 ×0.1×cfg: :169-176 DetailDaemonGraphSigmasNode, muerrilla detail_daemon.py:244, :304)
    """
    dd_adjustment = schedule_value * 0.1
    return max(1e-06, 1.0 - dd_adjustment * cfg_scale)


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:320-369 (DetailDaemonSamplerNode INPUT_TYPES)
# 이름: (default, min, max, step)
NODE_INPUTS = {
    "detail_amount": (0.1, -5.0, 5.0, 0.01),
    "start": (0.2, 0.0, 1.0, 0.01),
    "end": (0.8, 0.0, 1.0, 0.01),
    "bias": (0.5, 0.0, 1.0, 0.01),
    "exponent": (1.0, 0.0, 10.0, 0.05),
    "start_offset": (0.0, -1.0, 1.0, 0.01),
    "end_offset": (0.0, -1.0, 1.0, 0.01),
    "fade": (0.0, 0.0, 1.0, 0.05),
}
NODE_SMOOTH_DEFAULT = True   # :356 "smooth": ("BOOLEAN", {"default": True})
# origin: muerrilla/sd-webui-detail-daemon@19479998:scripts/detail_daemon.py:104
#   gr_hires = gr.Checkbox(label="Hires Pass", value=False, ...)
FORGE_HIRES_DEFAULT = False

# 노드 입력 이름 → 앱 설정 키 (값은 그대로 — 단위 변환이 없다)
NODE_TO_APP = {
    "detail_amount": "dd_amount", "start": "dd_start", "end": "dd_end", "bias": "dd_bias",
    "exponent": "dd_exponent", "start_offset": "dd_start_offset", "end_offset": "dd_end_offset",
    "fade": "dd_fade", "smooth": "dd_smooth",
}
# 확장 ui() 위치 (DETAIL_DAEMON_SPEC 순서) — 노드 입력이 실리는 칸
ARG = {key: index for index, (key, *_rest) in enumerate(DETAIL_DAEMON_SPEC)}


def forge_pass_runs(hires_checkbox, is_hires_pass):
    """origin: muerrilla detail_daemon.py:276-277 — ``if daemon['hires'] != self.is_hires_pass: continue``"""
    return hires_checkbox == is_hires_pass


def node_factors(node, steps, cfg):
    """노드에 같은 값을 넣었을 때 스텝마다 σ 배율."""
    schedule = node_make_schedule(
        steps, node["start"], node["end"], node["bias"], node["detail_amount"], node["exponent"],
        node["start_offset"], node["end_offset"], node["fade"], node["smooth"])
    return [node_sigma_factor(float(value), cfg) for value in schedule]


def payload_factors(args, steps, cfg):
    """앱 페이로드(14개 위치 인자)를 계약대로 읽어 같은 원본 식으로 계산한 σ 배율.

    고정 칸(arg10 multiplier, arg12 cfg_couple)도 식에 넣는다 — 앱이 중립값(1.0/True)을 보내야 노드와 같다.
    """
    schedule = node_make_schedule(
        steps, args[ARG["dd_start"]], args[ARG["dd_end"]], args[ARG["dd_bias"]], args[ARG["dd_amount"]],
        args[ARG["dd_exponent"]], args[ARG["dd_start_offset"]], args[ARG["dd_end_offset"]],
        args[ARG["dd_fade"]], args[ARG["dd_smooth"]])
    multiplier = float(args[ARG["dd_multiplier"]])
    coupled_cfg = cfg if args[ARG["dd_cfg_couple"]] else 1.0
    return [node_sigma_factor(float(value) * multiplier, coupled_cfg) for value in schedule]


def _node(**over):
    values = {name: spec[0] for name, spec in NODE_INPUTS.items()}
    values["smooth"] = NODE_SMOOTH_DEFAULT
    values.update(over)
    return values


def _app_settings(node, **extra):
    return {"dd_enabled": True, **{NODE_TO_APP[name]: value for name, value in node.items()}, **extra}


# 대표 설정 — 노드 기본값, 양/음 amount, offset·bias·exponent·fade·smooth 끔, ±1 밖 amount
REPRESENTATIVE = (
    _node(),
    _node(detail_amount=0.25),
    _node(detail_amount=1.0, start_offset=0.05),
    _node(detail_amount=-0.5, start=0.1, end=0.9, bias=0.3, exponent=2.0, start_offset=0.1,
          end_offset=-0.2, fade=0.25, smooth=False),
    _node(detail_amount=3.5, exponent=0.5, start_offset=0.4, end_offset=-0.6, fade=0.1),
    _node(detail_amount=0.0, start_offset=0.2, end_offset=-0.1),
    _node(detail_amount=-5.0, start=0.0, end=1.0, bias=1.0),
    _node(detail_amount=5.0),
)


class NodeDefaultsTests(unittest.TestCase):
    """앱 스펙의 기본값·범위 = 원본 노드 INPUT_TYPES."""

    def test_spec_defaults_and_ranges_are_the_node_inputs(self):
        spec = {key: (default, extra) for key, _kind, default, extra in DETAIL_DAEMON_SPEC}
        for name, (default, lo, hi, _step) in NODE_INPUTS.items():
            with self.subTest(node_input=name):
                app_default, app_range = spec[NODE_TO_APP[name]]
                self.assertEqual(app_default, default)
                self.assertEqual(app_range, (lo, hi))
        self.assertIs(spec["dd_smooth"][0], NODE_SMOOTH_DEFAULT)
        self.assertEqual(anima_guidance.DD_AMOUNT_MAX, NODE_INPUTS["detail_amount"][2])

    def test_hires_is_a_new_trailing_arg_with_the_forge_default(self):
        self.assertEqual(len(DETAIL_DAEMON_SPEC), 14)
        self.assertEqual(ARG["dd_hires"], 13)
        self.assertIs(default_settings()["dd_hires"], FORGE_HIRES_DEFAULT)
        from core.sam_extra_capabilities import DD_HIRES_INDEX
        self.assertEqual(DD_HIRES_INDEX, 13)

    def test_user_settings_have_no_presets_or_extra_knobs(self):
        defaults = default_settings()
        for key in ("dd_preset", "dd_multiplier", "dd_cfg_couple"):
            self.assertNotIn(key, defaults, "원본에 없는 칸은 사용자 설정이 아니다")
        dd_keys = [key for key in defaults if key.startswith("dd_")]
        self.assertEqual(dd_keys, ["dd_enabled", "dd_amount", "dd_start", "dd_end", "dd_bias", "dd_exponent",
                                   "dd_start_offset", "dd_end_offset", "dd_fade", "dd_smooth", "dd_hires"])
        self.assertEqual(sorted(dd_keys[1:-1]), sorted(NODE_TO_APP.values()), "노드 입력 전부 + Hires Pass")
        self.assertEqual(defaults["dd_amount"], 0.10)


class PayloadMatchesNodeTests(unittest.TestCase):
    """같은 숫자를 앱과 노드에 넣으면 스텝마다 같은 σ 배율."""

    def test_payload_carries_node_values_unchanged(self):
        for node in REPRESENTATIVE:
            with self.subTest(node=node):
                args = build_args(SCRIPT_DETAIL_DAEMON, _app_settings(node))
                self.assertEqual(len(args), 14)
                for name, key in NODE_TO_APP.items():
                    self.assertEqual(args[ARG[key]], node[name], name)
                # 고정 칸은 중립, Hires Pass 는 기본 끔
                self.assertEqual(args[ARG["dd_preset"]], anima_guidance.DD_PRESET_NEUTRAL)
                self.assertEqual(args[ARG["dd_multiplier"]], 1.0)
                self.assertIs(args[ARG["dd_cfg_couple"]], True)
                self.assertIs(args[ARG["dd_hires"]], False)

    def test_sigma_factors_equal_the_node(self):
        for node in REPRESENTATIVE:
            for steps, cfg in ((20, 4.5), (28, 5.0), (30, 7.0), (7, 1.0), (1, 3.0)):
                with self.subTest(node=node, steps=steps, cfg=cfg):
                    args = build_args(SCRIPT_DETAIL_DAEMON, _app_settings(node))
                    np.testing.assert_array_equal(payload_factors(args, steps, cfg),
                                                  node_factors(node, steps, cfg))

    def test_string_values_from_vue_give_the_same_factors(self):
        # Vue 프록시는 문자열을 보낸다 — 변환 뒤에도 노드와 같다
        node = _node(detail_amount=0.35, start_offset=-0.05, exponent=1.5, smooth=False)
        settings = {key: ("true" if value is True else "false" if value is False else str(value))
                    for key, value in _app_settings(node).items()}
        args = build_args(SCRIPT_DETAIL_DAEMON, settings)
        np.testing.assert_array_equal(payload_factors(args, 28, 5.0), node_factors(node, 28, 5.0))

    def test_golden_factors_from_the_original(self):
        # 원본 코드(make_detail_daemon_schedule + max(1e-06, 1 - s×0.1×cfg))를 스크래치에서 돌린 값
        args = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True})
        factors = payload_factors(args, 28, 5.0)
        self.assertAlmostEqual(min(factors), 0.95, places=12)      # plan §1.0: amount 0.10, cfg 5, 28 스텝
        self.assertEqual(int(np.argmin(factors)), 14)
        for index, want in ((0, 1.0), (6, 0.99849231552), (11, 0.9625), (18, 0.975), (21, 0.998096988313),
                            (22, 1.0)):
            self.assertAlmostEqual(factors[index], want, places=11, msg=index)
        args = build_args(SCRIPT_DETAIL_DAEMON, _app_settings(_node(detail_amount=1.0, start_offset=0.05)))
        factors = payload_factors(args, 20, 4.5)
        for index, want in ((0, 0.9775), (6, 0.870625), (10, 0.55), (12, 0.705471176266), (19, 1.0)):
            self.assertAlmostEqual(factors[index], want, places=11, msg=index)
        # 원본에는 0.05~3.0 클램프가 없다 — 음수 amount 는 σ 를 1 위로 올리고, 큰 amount 는 1e-06 바닥까지 간다
        negative = _node(detail_amount=-0.5, start=0.1, end=0.9, bias=0.3, exponent=2.0, start_offset=0.1,
                         end_offset=-0.2, fade=0.25, smooth=False)
        factors = payload_factors(build_args(SCRIPT_DETAIL_DAEMON, _app_settings(negative)), 20, 7.0)
        self.assertAlmostEqual(factors[6], 1.2625, places=12)
        self.assertAlmostEqual(factors[-1], 1.105, places=12)
        strong = payload_factors(build_args(SCRIPT_DETAIL_DAEMON, _app_settings(_node(detail_amount=5.0))), 28, 7.0)
        self.assertEqual(min(strong), 1e-06)

    def test_forge_hires_pass_semantics(self):
        # 기본(끔) = base 패스만, 켜면 hires 패스만 — 앱은 그 체크박스 값을 인자 13 에 싣는다
        off = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True})
        on = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True, "dd_hires": "true"})
        self.assertEqual(on[:13], off[:13])
        for args, base, hires in ((off, True, False), (on, False, True)):
            self.assertEqual(forge_pass_runs(args[ARG["dd_hires"]], is_hires_pass=False), base)
            self.assertEqual(forge_pass_runs(args[ARG["dd_hires"]], is_hires_pass=True), hires)


class SavedValueTests(unittest.TestCase):
    """저장값은 그대로 — 옛 프리셋·배율 키는 조용히 무시, 이전(×10) 없음."""

    def test_old_preset_and_knob_keys_are_ignored(self):
        saved = {"dd_enabled": "true", "dd_preset": "Strong", "dd_amount": "0.1",
                 "dd_multiplier": "1.7", "dd_cfg_couple": "false"}
        args = build_args(SCRIPT_DETAIL_DAEMON, saved)
        self.assertEqual(args[ARG["dd_amount"]], 0.1, "프리셋이 amount 를 덮어쓰지 않는다")
        self.assertEqual(args[ARG["dd_preset"]], "Custom")
        self.assertEqual(args[ARG["dd_multiplier"]], 1.0)
        self.assertIs(args[ARG["dd_cfg_couple"]], True)
        self.assertEqual(describe_active(saved), "Detail Daemon(0.1)")

    def test_values_are_sent_as_saved_without_migration(self):
        # 노드 범위 안의 저장값은 변환·이전 없이 그대로 간다
        for raw, sent in (("0.3", 0.3), (0.10, 0.10), ("2.5", 2.5), (-0.012, -0.012), ("5", 5.0), (-5, -5.0)):
            with self.subTest(raw=raw):
                args = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True, "dd_amount": raw})
                self.assertEqual(args[ARG["dd_amount"]], sent)
        args = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True, "dd_start_offset": "0.05",
                                                 "dd_end_offset": "-1"})
        self.assertEqual(args[ARG["dd_start_offset"]:ARG["dd_end_offset"] + 1], [0.05, -1.0])

    def test_amount_and_offsets_keep_the_node_ranges(self):
        # 노드는 범위 밖 값을 받지 않는다 — 숫자 위젯이 min/max 로 자르고, 그래도 들어오면 입력 검사가 거부한다.
        # origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:326 (detail_amount min -5 / max 5),
        #         :346 (start_offset min -1 / max 1), :350 (end_offset min -1 / max 1)
        # 앱은 다른 칸(start/end/bias/exponent/fade)처럼 그 범위로 자른다 — 예: 저장·프리셋 JSON 의 amount 8 이
        # Forge 에 8 로 가서 창 전체가 1e-06 바닥에 붙는 일이 없게.
        lo, hi = NODE_INPUTS["detail_amount"][1:3]
        for raw, sent in (("7", hi), (-12.5, lo), (8, hi), ("5.0001", hi)):
            with self.subTest(raw=raw):
                args = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True, "dd_amount": raw})
                self.assertEqual(args[ARG["dd_amount"]], sent)
        for name in ("start_offset", "end_offset"):
            lo, hi = NODE_INPUTS[name][1:3]
            key = NODE_TO_APP[name]
            for raw, sent in ((3, hi), ("-2", lo), (1.5, hi)):
                with self.subTest(offset=name, raw=raw):
                    self.assertEqual(build_args(SCRIPT_DETAIL_DAEMON, {key: raw})[ARG[key]], sent)
        # 요약도 보내는 값과 같다
        self.assertEqual(describe_active({"dd_enabled": True, "dd_amount": 8}), "Detail Daemon(5)")

    def test_non_finite_or_garbage_falls_back_to_the_node_default(self):
        for raw in ("abc", "", None, "nan", float("inf"), "-inf"):
            with self.subTest(raw=raw):
                args = build_args(SCRIPT_DETAIL_DAEMON, {"dd_enabled": True, "dd_amount": raw,
                                                         "dd_start_offset": raw})
                self.assertEqual(args[ARG["dd_amount"]], 0.10)
                self.assertEqual(args[ARG["dd_start_offset"]], 0.0)

    def test_schedule_shape_values_keep_the_node_ranges(self):
        args = build_args(SCRIPT_DETAIL_DAEMON, {"dd_start": -1, "dd_end": 3, "dd_bias": 2, "dd_exponent": 99,
                                                 "dd_fade": -0.5})
        self.assertEqual(args[ARG["dd_start"]:ARG["dd_fade"] + 1], [0.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0])

    def test_saved_settings_load_ignores_old_keys(self):
        from ui.generator_settings import SettingsMixin

        class _Proxy:
            def __init__(self):
                self.value = ""

            def text(self):
                return self.value

            def setText(self, value):
                self.value = value

        widgets = {key: _Proxy() for key in default_settings()}
        SettingsMixin()._set_anima_guidance_settings(
            widgets, {"dd_enabled": True, "dd_preset": "Strong", "dd_amount": 0.3, "dd_multiplier": 1.5})
        self.assertNotIn("dd_preset", widgets)
        self.assertEqual(widgets["dd_amount"].text(), "0.3")
        self.assertEqual(widgets["dd_hires"].text(), "false")

    def test_describe_active(self):
        self.assertEqual(describe_active({"dd_enabled": True}), "Detail Daemon(0.1)")
        self.assertEqual(describe_active({"dd_enabled": True, "dd_amount": "0.35", "dd_hires": "true"}),
                         "Detail Daemon(0.35, Hires)")

    def test_alwayson_block(self):
        block = build_alwayson({"dd_enabled": True, "dd_amount": 0.4})
        self.assertEqual(block[SCRIPT_DETAIL_DAEMON]["args"][:3], [True, "Custom", 0.4])


class ForgeImportTests(unittest.TestCase):
    """"Forge에서 가져오기" — 확장 값(노드 단위)을 그대로 저장값으로."""

    @staticmethod
    def _entry(values):
        args = [{"value": default} for _key, _kind, default, _extra in DETAIL_DAEMON_SPEC]
        for key, value in values.items():
            args[ARG[key]]["value"] = value
        return {"name": "anima detail daemon", "is_img2img": False, "args": args}

    def test_imports_node_values_without_conversion(self):
        entry = self._entry({"dd_enabled": True, "dd_preset": "Custom", "dd_amount": 2.5,
                             "dd_start_offset": 0.5, "dd_end_offset": -0.3, "dd_hires": True})
        entry["args"][ARG["dd_amount"]].update(minimum=-5.0, maximum=5.0)
        settings, meta = parse_forge_script_info([entry])
        self.assertEqual(settings["dd_amount"], 2.5)
        self.assertEqual((settings["dd_start_offset"], settings["dd_end_offset"]), (0.5, -0.3))
        self.assertIs(settings["dd_hires"], True)
        for key in ("dd_preset", "dd_multiplier", "dd_cfg_couple"):
            self.assertNotIn(key, settings)
        self.assertNotIn("detail_daemon_scale", meta)
        # 다시 보내면 같은 인자
        self.assertEqual(build_args(SCRIPT_DETAIL_DAEMON, settings), [e["value"] for e in entry["args"]])

    @staticmethod
    def _other(title, spec, values):
        args = [{"value": default} for _key, _kind, default, _extra in spec]
        keys = [key for key, *_rest in spec]
        for key, value in values.items():
            args[keys.index(key)]["value"] = value
        return {"name": title.lower(), "is_img2img": False, "args": args}

    def test_extension_without_hires_arg_imports_the_first_13(self):
        # Hires Pass(arg13)는 맨 뒤 append — 13개 빌드(sam-extra v0.30, 4045adb)의 앞 13칸은 뜻이 같다.
        # 그 빌드에서도 가져오기가 통째로 실패하지 않고(PAG·Skimmed 도 함께 잃지 않고) 빠진 칸만 기본값으로 둔다.
        dd = self._entry({"dd_enabled": True, "dd_amount": 0.35, "dd_end_offset": -0.2})
        dd["args"] = dd["args"][:ARG["dd_hires"]]
        payload = [
            self._other(anima_guidance.SCRIPT_PERTURBATION, anima_guidance.PERTURBATION_SPEC,
                        {"guid_enabled": True, "guid_scale": 5.5}),
            self._other(anima_guidance.SCRIPT_SKIMMED_CFG, anima_guidance.SKIMMED_SPEC,
                        {"skim_enabled": True, "skim_skimming_cfg": 3.0}),
            dd,
        ]
        settings, meta = parse_forge_script_info(payload)
        self.assertEqual(meta["imported_scripts"], list(anima_guidance.SPECS))
        self.assertEqual((settings["guid_enabled"], settings["guid_scale"]), (True, 5.5))
        self.assertEqual((settings["skim_enabled"], settings["skim_skimming_cfg"]), (True, 3.0))
        self.assertEqual((settings["dd_enabled"], settings["dd_amount"], settings["dd_end_offset"]),
                         (True, 0.35, -0.2))
        self.assertIs(settings["dd_hires"], FORGE_HIRES_DEFAULT)
        self.assertEqual(meta["missing_trailing_args"], ["dd_hires"])
        self.assertEqual(meta["ignored_trailing_args"], 0)

    def test_full_extension_reports_no_missing_trailing_args(self):
        settings, meta = parse_forge_script_info([self._entry({"dd_hires": True})])
        self.assertIs(settings["dd_hires"], True)
        self.assertEqual(meta["missing_trailing_args"], [])

    def test_array_short_inside_the_positional_contract_is_rejected(self):
        # append 칸 앞(arg0-12)이 빠진 배열은 위치가 어긋날 수 있으니 전과 같이 거부한다
        entry = self._entry({})
        entry["args"] = entry["args"][:ARG["dd_hires"] - 1]
        with self.assertRaisesRegex(ValueError, "인자 수 12개가 앱 계약 13개"):
            parse_forge_script_info([entry])
        pag = self._other(anima_guidance.SCRIPT_PERTURBATION, anima_guidance.PERTURBATION_SPEC, {})
        pag["args"].pop()
        with self.assertRaisesRegex(ValueError, "Perturbation Guidance: Forge 인자 수 61개"):
            parse_forge_script_info([pag])


class HiresSupportNoteTests(unittest.TestCase):
    class _Caps:
        def __init__(self, hires):
            self.detail_daemon_hires = hires

    def test_warns_only_when_hires_is_on_and_the_extension_lacks_arg_13(self):
        on = {"dd_enabled": True, "dd_hires": "true"}
        self.assertIn("Hires Pass", detail_daemon_hires_note(on, self._Caps(False)))
        for caps in (self._Caps(True), self._Caps(None), None):
            self.assertIsNone(detail_daemon_hires_note(on, caps))
        self.assertIsNone(detail_daemon_hires_note({"dd_enabled": True}, self._Caps(False)))
        self.assertIsNone(detail_daemon_hires_note({"dd_hires": True}, self._Caps(False)))

    def test_comfyui_follows_hires_pass_so_there_is_no_note(self):
        # DD-C 뒤로 컴파일러가 dd_hires 로 패스를 고른다(test_comfy_detail_daemon) — 경고 없음. 남아 있는 Forge
        # 기능 스냅샷(인자 13 없는 빌드)은 ComfyUI 와 무관하니 보지 않는다.
        on = {"dd_enabled": True, "dd_hires": "true"}
        for caps in (None, self._Caps(None), self._Caps(True), self._Caps(False)):
            self.assertIsNone(detail_daemon_hires_note(on, caps, comfyui=True))
        self.assertFalse(hasattr(anima_guidance, "DD_HIRES_NOTE_COMFYUI"))

    def test_generation_logs_the_note_for_the_active_backend(self):
        from unittest import mock

        import backends
        from ui.generator_generation import GenerationMixin

        class _Unchecked:
            def isChecked(self):
                return False

        class _Host(GenerationMixin):
            adetailer_group = _Unchecked()
            sam3_group = None
            sam_extra_capabilities = None

            def _build_anima_settings(self):
                return {"dd_enabled": True, "dd_hires": True}

        old_forge = self._Caps(False)   # 인자 13 없는 Forge 빌드의 스냅샷(ComfyUI 로 바꾼 뒤 남아 있어도 무관)
        for backend, caps, note in ((backends.BackendType.COMFYUI, old_forge, None),
                                    (backends.BackendType.COMFYUI, None, None),
                                    (backends.BackendType.WEBUI, None, None),
                                    (backends.BackendType.WEBUI, old_forge,
                                     anima_guidance.DD_HIRES_NOTE_OLD_EXTENSION)):
            with self.subTest(backend=backend, caps=caps), mock.patch.object(backends, "get_backend_type",
                                                                             return_value=backend):
                payload = {}
                host = _Host()
                host.sam_extra_capabilities = caps
                with self.assertLogs("generation", "INFO") as logs:
                    host._apply_postprocess_chain(payload)
                warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
                self.assertEqual(warnings, [note] if note else [])
                self.assertIs(payload["alwayson_scripts"][SCRIPT_DETAIL_DAEMON]["args"][ARG["dd_hires"]], True)


class VueContractTests(unittest.TestCase):
    """DD-A 가 바꾼 파이썬 쪽 계약(기능 스냅샷 키·위젯 프록시)을 Vue 가 그대로 쓰는가."""

    FRONT = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "frontend", "src")

    def _read(self, *parts):
        import os

        with open(os.path.join(self.FRONT, *parts), encoding="utf-8") as f:
            return f.read()

    def test_capabilities_event_type_has_the_python_keys(self):
        import re

        from core.sam_extra_capabilities import unknown_capabilities

        text = self._read("types", "bridge.d.ts")
        body = re.search(r"export interface SamExtraCapabilitiesEvent \{(.*?)\n\}", text, re.S).group(1)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        depth, keys = 0, []
        for line in body.splitlines():
            match = re.match(r"\s*(\w+)\??\s*:", line)
            if depth == 0 and match:
                keys.append(match.group(1))
            depth += line.count("{") - line.count("}")
        self.assertEqual(sorted(keys), sorted(unknown_capabilities().to_dict()))
        self.assertNotIn("detail_daemon_scale", keys)

    def test_guidance_sections_use_only_user_setting_keys(self):
        # 섹션이 쓰는 위젯 id(_ + 스펙 키)는 프록시가 있는 키여야 한다 — 없으면 vue_bridge 가 조용히 버린다
        import glob
        import os
        import re

        user_keys = set(default_settings())
        paths = glob.glob(os.path.join(self.FRONT, "components", "guidance", "*Section.vue"))
        self.assertTrue(any(p.endswith("DetailDaemonSection.vue") for p in paths))
        for path in paths:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            used = set(re.findall(r"\bw\._(\w+)", text)) | set(re.findall(r"\b(?:b|setB)\('(\w+)'", text))
            with self.subTest(section=os.path.basename(path)):
                self.assertTrue(used)
                self.assertEqual(sorted(used - user_keys), [])

    def test_detail_daemon_section_binds_every_node_input_and_hires(self):
        import re

        text = self._read("components", "guidance", "DetailDaemonSection.vue")
        used = set(re.findall(r"\bw\._(dd_\w+)", text)) | set(re.findall(r"\b(?:b|setB)\('(dd_\w+)'", text))
        self.assertEqual(used, {key for key in default_settings() if key.startswith("dd_")})
        # 입력 범위·단위 = 노드 INPUT_TYPES (origin: detail_daemon_node.py:324-355)
        for name, (_default, lo, hi, step) in NODE_INPUTS.items():
            key = NODE_TO_APP[name]
            with self.subTest(node_input=name):
                tag = re.search(rf"<input\b[^>]*w\._{key}\b[^>]*>", text, re.S)
                self.assertIsNotNone(tag, key)
                attrs = dict(re.findall(r'\b(step|min|max)="([^"]+)"', tag.group(0)))
                self.assertEqual({k: float(v) for k, v in attrs.items()}, {"step": step, "min": lo, "max": hi})


if __name__ == "__main__":
    unittest.main()
