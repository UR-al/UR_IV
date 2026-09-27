"""샘플링 블록 제목별 전달 규칙(P7) — core/alwayson_propagation (순수)."""
import copy
import unittest
from types import SimpleNamespace

from core import alwayson_propagation as ap
from core import anima38, anima_guidance, sam3_args
from core import sam_extra_contract as reg
from core import sam_extra_notices as sn
from core.sam_extra_capabilities import SamExtraCapabilities, _freeze

PAG, SKIM, DD = (anima_guidance.SCRIPT_PERTURBATION, anima_guidance.SCRIPT_SKIMMED_CFG,
                 anima_guidance.SCRIPT_DETAIL_DAEMON)
A38, DORA, NEGPIP = anima38.SCRIPT_NAME, ap.TITLE_DORA, ap.TITLE_NEGPIP
SAMPLING = (NEGPIP, PAG, SKIM, DD, A38, DORA)


def dd_block(hires=False, argc=14):
    args = anima_guidance.build_args(DD, {"dd_enabled": True, "dd_hires": hires})
    return {"args": args[:argc]}


def all_blocks(**overrides):
    blocks = {NEGPIP: {"args": [True]}, PAG: {"args": [True, 4.0]}, SKIM: {"args": [True, 7.0]},
              DD: dd_block(), A38: {"args": [{"negative": True}]}, DORA: {"args": [{"enabled": True}]},
              "SAM3 Mask": {"args": [{"sam3_prompt": "face"}]}, "ADetailer": {"args": [True, True, {}]}}
    blocks.update(overrides)
    return blocks


def caps(present=(), img2img=None, *, status="ok"):
    """known 스냅샷 — present(txt2img)·img2img 목록. img2img None 이면 present 와 같다."""
    img2img = present if img2img is None else img2img
    titles = {t.lower() for t in (*present, *img2img)} | {t.lower() for t in (PAG, SKIM, DD, A38, DORA)}
    scripts = {t: {"present": t in {p.lower() for p in present}, "img2img": t in {p.lower() for p in img2img}}
               for t in titles}
    return SamExtraCapabilities(status=status, installed=True, scripts=_freeze(scripts))


class ClassificationTests(unittest.TestCase):
    def test_t1_every_registry_script_and_negpip_is_classified_exactly_once(self):
        titles = set(reg.SCRIPTS) | {NEGPIP, "ADetailer"}   # 앱이 내는 확장 밖 제목
        for title in titles:
            with self.subTest(title=title):
                self.assertEqual((title in ap.PROPAGATION) + (title in ap.NEVER), 1)
        self.assertEqual(set(ap.PROPAGATION) | set(ap.NEVER), titles)
        for title, rule in ap.PROPAGATION.items():
            self.assertTrue(rule.reason.strip(), title)
            self.assertTrue(rule.passes <= set(ap.AUX_PASSES))
            self.assertTrue(set(rule.when_unknown) == set(ap.PROVENANCES))
            self.assertTrue(set(rule.when_unknown.values()) <= {ap.SEND, ap.SKIP})
        for title, reason in ap.NEVER.items():
            self.assertTrue(reason.strip(), title)

    def test_package_marks_until_the_producer_lands(self):
        # P8(DoRA)·P9(Anima38) 기여자가 모두 등록됐다 — 생산자를 기다리는 행이 없다
        self.assertEqual({t for t, r in ap.PROPAGATION.items() if r.package}, set())
        from ui import sampling_blocks as sb
        from ui.anima38_ui import contribute as anima38_contribute
        from ui.dora_infer_mode_ui import contribute as dora_contribute
        self.assertIn(anima38_contribute, sb.CONTRIBUTORS)
        self.assertIn(dora_contribute, sb.CONTRIBUTORS)

    def test_only_anima38_is_bound_to_the_model(self):
        """(A7) 블록 내용이 T2I 모델 종류로 정해지는 제목만 model_bound — 보조 패스가 실제 모델과 대조한다."""
        self.assertEqual({t for t, r in ap.PROPAGATION.items() if r.model_bound}, {A38})

    def test_drop_model_bound_keeps_anima38_only_for_the_same_checkpoint(self):
        blocks = {PAG: {"args": [1]}, A38: {"args": [{"negative": True}]}, DORA: {"args": [{}]}}
        same = ap.drop_model_bound(blocks, envelope_model="sub/Anima-3.8B-v1.1.safetensors [abcdef12]",
                                   actual_model="Anima-3.8B-v1.1.safetensors")
        self.assertEqual(same, (blocks, ()))
        # (P9 리뷰 1) Forge 'Show filenames without folder' 의 short_title·name_for_extra 별칭도 같은 모델이다
        for actual in ("Anima-3.8B-v1.1 [abcdef12]", "Anima-3.8B-v1.1"):
            with self.subTest(short_title=actual):
                self.assertEqual(ap.drop_model_bound(blocks, envelope_model="Anima-3.8B-v1.1.safetensors [abcdef12]",
                                                     actual_model=actual), (blocks, ()))
        for envelope_model, actual in (("Anima-3.8B-v1.1.safetensors", "Anima-2.9B.safetensors"),   # 다름
                                       ("Anima-3.8B-v1.1.safetensors", ""),                          # 실제 모름
                                       ("", "Anima-3.8B-v1.1.safetensors"),                          # 봉투 모름
                                       (None, None)):
            with self.subTest(envelope=envelope_model, actual=actual):
                kept, dropped = ap.drop_model_bound(blocks, envelope_model=envelope_model, actual_model=actual)
                self.assertEqual(list(kept), [PAG, DORA])
                self.assertEqual(dropped, (A38,))
        self.assertEqual(list(blocks), [PAG, A38, DORA])                                    # 입력 불변
        self.assertTrue(ap.has_model_bound({"anima 3.8b (qwen3.5 / v2)": {}}))
        self.assertFalse(ap.has_model_bound({PAG: {}, DORA: {}}))

    def test_dora_title_is_the_single_app_constant(self):
        """(B13) 제목 사본은 core.dora_infer_mode.SCRIPT_NAME 하나 — 전달 표·기능 스냅샷·알림 표가 같은 값을 쓴다."""
        from core import dora_infer_mode, sam_extra_capabilities as caps
        self.assertIs(ap.TITLE_DORA, dora_infer_mode.SCRIPT_NAME)
        self.assertEqual(caps.TITLE_DORA, dora_infer_mode.SCRIPT_NAME.lower())
        self.assertIs(sn._TITLE_DORA, dora_infer_mode.SCRIPT_NAME)

    def test_dora_title_copies_are_derived_from_script_name_in_the_source(self):
        """(B13) 값 비교로는 같은 문자열 사본('dora inference mode')을 못 잡는다(소문자 사본은 assertIs 도 못 쓴다) —
        세 모듈의 대입식이 SCRIPT_NAME 에서 나오는지, 제목 문자열 리터럴이 없는지 AST 로 본다."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        expected = {
            "core/sam_extra_capabilities.py": ("TITLE_DORA", "dora_infer_mode.SCRIPT_NAME.lower()"),
            "core/alwayson_propagation.py": ("TITLE_DORA", "dora_infer_mode.SCRIPT_NAME"),
            "core/sam_extra_notices.py": ("_TITLE_DORA", "dora_infer_mode.SCRIPT_NAME"),
        }
        for rel, (name, source) in expected.items():
            with self.subTest(module=rel):
                tree = ast.parse((root / rel).read_text(encoding="utf-8"))
                values = [ast.unparse(node.value) for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
                self.assertEqual(values, [source])
                literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                            and isinstance(node.value, str) and node.value.strip().lower() == "dora inference mode"]
                self.assertEqual(literals, [])

    def test_dora_is_skipped_when_unknown_for_both_provenances(self):
        """(A2) 모를 때는 사용자 값이어도 보내지 않는다(메인 생성엔 422 재시도가 없다)."""
        rule = ap.PROPAGATION[DORA]
        self.assertEqual(rule.unknown_action(ap.PROVENANCE_USER), ap.SKIP)
        self.assertEqual(rule.unknown_action(ap.PROVENANCE_APP_DEFAULT), ap.SKIP)
        self.assertEqual(rule.backends, frozenset({ap.BACKEND_WEBUI}))

    def test_dora_title_matches_the_registry_and_notice_table(self):
        self.assertIn(DORA, reg.SCRIPTS)
        self.assertIn(DORA.lower(), sn.SAM_EXTRA_FEATURES)

    def test_t2_pass_and_backend_table(self):
        webui, comfy = ap.BACKEND_WEBUI, ap.BACKEND_COMFY
        expected = {
            # (제목, 백엔드) → 보내는 보조 패스
            (NEGPIP, webui): {ap.AUX_HAND},                           # A3: Forge 보조 패스에는 없음(손 재구성만 메인 경로)
            (NEGPIP, comfy): set(ap.AUX_PASSES),
            (DORA, webui): set(ap.AUX_PASSES),
            (DORA, comfy): set(),                                    # Comfy 는 만들지 않는다(P8)
        }
        for title in (PAG, SKIM, DD, A38):
            expected[(title, webui)] = set(ap.AUX_PASSES)
            expected[(title, comfy)] = set(ap.AUX_PASSES)
        for (title, backend), passes in expected.items():
            for aux in ap.AUX_PASSES:
                with self.subTest(title=title, backend=backend, aux=aux):
                    self.assertEqual(ap.PROPAGATION[title].allows(aux, backend), aux in passes)
                    got = ap.blocks_for(all_blocks(), aux, backend=backend)
                    self.assertEqual(title in got, aux in passes)
        for title in ap.NEVER:
            for aux in ap.AUX_PASSES:
                self.assertNotIn(title, ap.blocks_for(all_blocks(), aux, backend=webui))

    def test_unknown_pass_is_rejected(self):
        with self.assertRaises(ValueError):
            ap.blocks_for(all_blocks(), "upscale", backend=ap.BACKEND_WEBUI)


class BlocksForTests(unittest.TestCase):
    def test_t3_titles_are_canonical_and_unknown_titles_dropped(self):
        source = {"anima perturbation guidance": {"args": [1]}, "  ANIMA SKIMMED CFG ": {"args": [2]},
                  "sam3 mask": {"args": [3]}, "Adetailer": {"args": [4]}, "Mystery Script": {"args": [5]}}
        got = ap.blocks_for(source, ap.AUX_REFINE, backend=ap.BACKEND_WEBUI)
        self.assertEqual(list(got), [PAG, SKIM])
        self.assertEqual(ap.canonical_title("dora inference mode"), DORA)
        self.assertEqual(ap.canonical_title("sam3 mask"), "SAM3 Mask")
        self.assertIsNone(ap.canonical_title("Mystery Script"))

    def test_t4_deep_copies(self):
        source = all_blocks()
        before = copy.deepcopy(source)
        got = ap.blocks_for(source, ap.AUX_REFINE, backend=ap.BACKEND_WEBUI)
        got[PAG]["args"].append("changed")
        env = ap.envelope(source, source="t2i_panel")
        env["blocks"][SKIM]["args"].append("changed")
        self.assertEqual(source, before)

    def test_t5_dd_only_without_hires_pass_and_short_args_count_as_off(self):
        for aux in ap.AUX_PASSES:
            with self.subTest(aux=aux):
                self.assertIn(DD, ap.blocks_for({DD: dd_block(hires=False)}, aux, backend=ap.BACKEND_WEBUI))
                self.assertNotIn(DD, ap.blocks_for({DD: dd_block(hires=True)}, aux, backend=ap.BACKEND_WEBUI))
                self.assertIn(DD, ap.blocks_for({DD: dd_block(hires=True, argc=13)}, aux, backend=ap.BACKEND_COMFY))
                self.assertIn(DD, ap.blocks_for({DD: {"args": [True, 0.75]}}, aux, backend=ap.BACKEND_WEBUI))
        # 문자열 불리언도 켬으로 본다
        block = dd_block()
        block["args"][anima_guidance.DD_HIRES_INDEX] = "true"
        self.assertEqual(ap.blocks_for({DD: block}, ap.AUX_SAM3, backend=ap.BACKEND_WEBUI), {})

    def test_t5_sam3_mask_only_gets_nothing_by_the_sam3_args_normalisation(self):
        for mode in ("Mask only", "mask only", "  MASK ONLY "):
            with self.subTest(mode=mode):
                self.assertEqual(sam3_args.mode_of({"sam3_mode": mode}), sam3_args.MODE_MASK_ONLY)
                self.assertEqual(ap.blocks_for(all_blocks(), ap.AUX_SAM3, backend=ap.BACKEND_WEBUI,
                                               aux_settings={"sam3_mode": mode}), {})
        for mode in ("Inpaint", "", None, "mask-only"):
            self.assertTrue(ap.blocks_for(all_blocks(), ap.AUX_SAM3, backend=ap.BACKEND_WEBUI,
                                          aux_settings={"sam3_mode": mode}))
        # Mask only 는 sam3 패스만의 규칙
        self.assertTrue(ap.blocks_for(all_blocks(), ap.AUX_REFINE, backend=ap.BACKEND_WEBUI,
                                      aux_settings={"sam3_mode": "Mask only"}))

    def test_order_follows_the_source(self):
        got = ap.blocks_for(all_blocks(), ap.AUX_REFINE, backend=ap.BACKEND_COMFY)
        self.assertEqual(list(got), [NEGPIP, PAG, SKIM, DD, A38])


class GateTests(unittest.TestCase):
    blocks = {NEGPIP: {"args": [True]}, PAG: {"args": [1]}, DD: {"args": [2]}, A38: {"args": [3]},
              DORA: {"args": [4]}}

    def test_t6_unknown_snapshots_follow_the_rule_and_provenance(self):
        unknown_caps = (None, SamExtraCapabilities(), SamExtraCapabilities(status="error"),
                        SimpleNamespace(detail_daemon_hires=False),        # tests/test_detail_daemon_origin 의 _Caps
                        SimpleNamespace(known=True))                       # script 가 없는 더블
        for capabilities in unknown_caps:
            with self.subTest(caps=capabilities):
                user = ap.gate(self.blocks, capabilities, img2img=True)
                self.assertEqual(list(user.kept), [NEGPIP, PAG, DD, A38])   # DoRA 는 모르면 늘 보내지 않는다
                self.assertEqual(user.dropped_unknown, (DORA,))
                default = ap.gate(self.blocks, capabilities, img2img=True,
                                  provenance={A38: ap.PROVENANCE_APP_DEFAULT, DORA: ap.PROVENANCE_APP_DEFAULT})
                self.assertEqual(list(default.kept), [NEGPIP, PAG, DD])
                self.assertEqual(default.dropped_unknown, (A38, DORA))
                self.assertEqual(default.dropped_missing, ())

    def test_t6_known_snapshot_drops_missing_and_uses_the_right_list(self):
        capabilities = caps(present=(PAG, DD, A38, DORA), img2img=(PAG, A38, DORA))   # DD 는 txt2img 에만
        t2i = ap.gate(self.blocks, capabilities, img2img=False)
        self.assertEqual(list(t2i.kept), [NEGPIP, PAG, DD, A38, DORA])
        i2i = ap.gate(self.blocks, capabilities, img2img=True)
        self.assertEqual(list(i2i.kept), [NEGPIP, PAG, A38, DORA])
        self.assertEqual(i2i.dropped_missing, (DD,))
        # 확장이 아예 없다 — NegPiP(확장 밖)만 남는다
        none = ap.gate(self.blocks, caps(present=()), img2img=True)
        self.assertEqual(list(none.kept), [NEGPIP])
        self.assertEqual(set(none.dropped_missing), {PAG, DD, A38, DORA})

    def test_case_insensitive_titles_and_non_sampling_titles_pass_through(self):
        blocks = {"anima perturbation guidance": {"args": [1]}, "SAM3 Mask": {"args": []}, "Other": {}}
        result = ap.gate(blocks, caps(present=()), img2img=True)
        self.assertEqual(list(result.kept), ["SAM3 Mask", "Other"])
        self.assertEqual(result.dropped_missing, ("anima perturbation guidance",))

    def test_script_available(self):
        capabilities = caps(present=(PAG,), img2img=())
        self.assertIs(ap.script_available(capabilities, PAG, img2img=False), True)
        self.assertIs(ap.script_available(capabilities, PAG, img2img=True), False)
        self.assertIsNone(ap.script_available(None, PAG, img2img=True))
        self.assertIsNone(ap.script_available(SamExtraCapabilities(), PAG, img2img=True))
        broken = SimpleNamespace(known=True, script=lambda _t: (_ for _ in ()).throw(RuntimeError()))
        self.assertIsNone(ap.script_available(broken, PAG, img2img=True))

    def test_drop_missing_only_drops_known_missing_sampling_blocks(self):
        payload = {"alwayson_scripts": {PAG: {"args": [1]}, DORA: {"args": [2]}, "SAM3 Mask": {"args": []},
                                        NEGPIP: {"args": [True]}}}
        self.assertEqual(ap.drop_missing(payload, None, img2img=False), ())
        self.assertEqual(set(payload["alwayson_scripts"]), {PAG, DORA, "SAM3 Mask", NEGPIP})   # 모르면 그대로
        self.assertEqual(ap.drop_missing(payload, caps(present=(DORA,)), img2img=False), (PAG,))
        self.assertEqual(list(payload["alwayson_scripts"]), [DORA, "SAM3 Mask", NEGPIP])       # 이미지 패스는 그대로
        self.assertEqual(ap.drop_missing({"prompt": "x"}, None, img2img=False), ())


class GateNoticeTests(unittest.TestCase):
    def test_user_values_warn_app_defaults_stay_quiet(self):
        result = ap.GateResult({}, dropped_missing=(PAG, A38), dropped_unknown=(DORA,))
        provenance = {A38: ap.PROVENANCE_APP_DEFAULT}
        with self.assertLogs("core.alwayson_propagation", "INFO"):
            main = ap.gate_notices(result, provenance, img2img=False)
        self.assertEqual([(n.code, n.detail) for n in main],
                         [(sn.CODE_BLOCK_NOT_SENT, f"{PAG}@txt2img"), (sn.CODE_BLOCK_DEFERRED, DORA)])
        self.assertEqual(main[0].level, sn.LEVEL_WARNING)
        self.assertEqual(main[1].level, sn.LEVEL_INFO)
        aux = ap.gate_notices(result, {A38: ap.PROVENANCE_APP_DEFAULT, DORA: ap.PROVENANCE_APP_DEFAULT},
                              img2img=True, aux=ap.AUX_REFINE)
        self.assertEqual([n.code for n in aux], [sn.CODE_PROPAGATION_DROPPED])
        self.assertIn("Refine", aux[0].message)


class EnvelopeTests(unittest.TestCase):
    def test_envelope_round_trip_and_take_does_not_mutate(self):
        blocks = {**all_blocks(), "Mystery": {"args": []}}
        env = ap.envelope(blocks, source="t2i_panel", model="m.safetensors", backend="webui",
                          provenance={A38: ap.PROVENANCE_APP_DEFAULT})
        self.assertEqual(env["version"], ap.ENVELOPE_VERSION)
        self.assertEqual(list(env["blocks"]), list(SAMPLING))            # PROPAGATION 제목만
        self.assertEqual(env["provenance"][A38], ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(env["provenance"][PAG], ap.PROVENANCE_USER)
        settings = {"target": "face", ap.SETTINGS_KEY: env}
        before = copy.deepcopy(settings)
        clean, taken = ap.take_envelope(settings)
        self.assertEqual(settings, before)
        self.assertNotIn(ap.SETTINGS_KEY, clean)
        self.assertEqual(taken, env)
        self.assertEqual(ap.envelope_provenance(taken)[A38], ap.PROVENANCE_APP_DEFAULT)

    def test_malformed_envelope_is_dropped_but_key_removed(self):
        for bad in (None, {}, {"version": 99, "blocks": {}}, {"version": 1, "blocks": []}, "x"):
            with self.subTest(bad=bad):
                clean, env = ap.take_envelope({"a": 1, ap.SETTINGS_KEY: bad})
                self.assertEqual(clean, {"a": 1})
                self.assertIsNone(env)
        self.assertEqual(ap.take_envelope(None), ({}, None))

    def test_t7_apply_is_setdefault_and_never_overwrites_image_passes(self):
        own = {"args": [{"sam3_prompt": "hand"}]}
        payload = {"alwayson_scripts": {"SAM3 Mask": own, "anima perturbation guidance": {"args": ["mine"]}}}
        inserted = ap.apply(payload, {"SAM3 Mask": {"args": ["t2i"]}, PAG: {"args": ["t2i"]},
                                      SKIM: {"args": [1]}})
        self.assertEqual(inserted, (SKIM,))
        self.assertIs(payload["alwayson_scripts"]["SAM3 Mask"], own)
        self.assertEqual(payload["alwayson_scripts"]["anima perturbation guidance"], {"args": ["mine"]})
        self.assertEqual(list(payload["alwayson_scripts"]), ["SAM3 Mask", "anima perturbation guidance", SKIM])
        fresh = {}
        self.assertEqual(ap.apply(fresh, {PAG: {"args": [1]}}), (PAG,))
        self.assertEqual(fresh, {"alwayson_scripts": {PAG: {"args": [1]}}})


class RequestProvenanceKeyTests(unittest.TestCase):
    """(P10 검토 2) 요청 페이로드의 비공개 출처 키 — 백엔드가 떼어 쓴다. 키가 없으면 같은 객체(바이트 동일)."""

    def test_take_provenance_strips_a_copy_and_leaves_the_input(self):
        plain = {"prompt": "p", "alwayson_scripts": {DORA: {"args": [1]}}}
        same, provenance = ap.take_provenance(plain)
        self.assertIs(same, plain)
        self.assertEqual(provenance, {})
        marked = {**plain, ap.PROVENANCE_KEY: {DORA: ap.PROVENANCE_APP_DEFAULT}}
        stripped, provenance = ap.take_provenance(marked)
        self.assertEqual(stripped, plain)
        self.assertEqual(provenance, {DORA: ap.PROVENANCE_APP_DEFAULT})
        self.assertIn(ap.PROVENANCE_KEY, marked)
        self.assertEqual(ap.take_provenance({**plain, ap.PROVENANCE_KEY: "junk"}), (plain, {}))
        self.assertEqual(ap.take_provenance(None), (None, {}))

    def test_provenance_of_defaults_to_user_and_ignores_case(self):
        provenance = {DORA: ap.PROVENANCE_APP_DEFAULT}
        self.assertEqual(ap.provenance_of(provenance, DORA), ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(ap.provenance_of(provenance, f"  {DORA.upper()} "), ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(ap.provenance_of(provenance, A38), ap.PROVENANCE_USER)
        self.assertEqual(ap.provenance_of(None, DORA), ap.PROVENANCE_USER)

    def test_mark_provenance_writes_app_default_titles_that_are_still_sent(self):
        payload = {"alwayson_scripts": {PAG: {}, DORA: {}}, ap.PROVENANCE_KEY: {"stale": ap.PROVENANCE_APP_DEFAULT}}
        ap.mark_provenance(payload, {DORA: ap.PROVENANCE_APP_DEFAULT, A38: ap.PROVENANCE_APP_DEFAULT,
                                     PAG: ap.PROVENANCE_USER})
        self.assertEqual(payload[ap.PROVENANCE_KEY], {DORA: ap.PROVENANCE_APP_DEFAULT})
        ap.mark_provenance(payload, {PAG: ap.PROVENANCE_USER})
        self.assertNotIn(ap.PROVENANCE_KEY, payload)


class MainRetryTitleTests(unittest.TestCase):
    """P10(critic A2) — 메인 생성에서 422 면 빼고 다시 보낼 제목은 앱이 Forge 설정에 맞춰 넣는 블록뿐이다."""

    def test_only_app_supplied_blocks_are_retried_in_the_main_generation(self):
        self.assertEqual({title for title, rule in ap.PROPAGATION.items() if rule.main_retry}, {A38, DORA})

    def test_titles_are_read_from_the_payload_as_spelled(self):
        payload = {"alwayson_scripts": {PAG: {"args": [1]}, "dora inference mode": {"args": [{}]},
                                        A38: {"args": [{}]}, "SAM3 Mask": {"args": [{}]}, "Unknown": {}}}
        self.assertEqual(ap.main_retry_titles(payload), ("dora inference mode", A38))
        for bad in (None, {}, {"alwayson_scripts": None}, {"alwayson_scripts": [DORA]}):
            with self.subTest(payload=bad):
                self.assertEqual(ap.main_retry_titles(bad), ())


if __name__ == "__main__":
    unittest.main()
