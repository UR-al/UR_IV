"""ADetailer 슬롯의 LoRA — ComfyUI 에서도 Forge 처럼 그 슬롯 프롬프트의 LoRA 만 건다(사용자 결정 2026-09-27 "Forge와 똑같이").

Forge(읽기 전용 참조): aadetailer-neoforge scripts/!adetailer.py 는 슬롯마다 p2 로 process_images 를 돈다(:1048-1078).
process_images 는 배치마다 forge_objects_original(LoRA 없음)로 되돌리고 p2 positive 의 태그만 건다(modules/processing.py
:947, :967-970). p2 프롬프트 = get_prompt(:441-466 → _get_prompt :307-327): ad_prompt 를 [SEP] 로 나누고, 빈 칸은 메인
프롬프트(태그 포함), [PROMPT] 는 메인 프롬프트로 바꾸며, ad_copy_main_loras(기본 False — adetailer/args.py:63)면 메인
프롬프트의 LoRA 토큰을 붙인다. 네거티브의 태그는 걸리지 않고 글자 그대로다.

예전 Comfy: ADetailer 노드는 메인 model/clip(메인 LoRA)을 받고, ad_prompt 를 그대로 Impact FaceDetailer wildcard 로
넘겨 Impact 가 그 태그를 받은 모델 위에 core LoraLoader(ComfyUI 순서 model·clip)로 더 걸었다
(ComfyUI-Impact-Pack modules/impact/wildcards.py process_with_loras, core.py enhance_detail). 그래서 태그 없는 ad_prompt
는 메인 LoRA 를 그대로 받고, 태그가 있으면 메인 + 그 태그, 메인 프롬프트와 같은 태그는 두 번 걸렸다.

컴파일러(core/comfy_workflow_compiler.py ``_adetailer_pass_stack``): 목록이 메인과 같으면 메인 스택 그대로(그래프 불변 —
Impact 에 넘기는 글에서 태그만 뗀다), 다르면 LoRA 앞에서 가른 분기(그 LoRA → NegPiP → 조건 → 가이던스 → 마지막 본 패스의
Detail Daemon). 같은 목록의 슬롯은 분기 하나를 같이 쓴다. 단독 ADetailer 는 쓰는 패스가 메인 목록을 쓰지 않으면 스택
자체를 첫 패스의 목록으로 만든다(``_postprocess_stack_loras``).
"""
import copy
import json
import re
import unittest
from unittest import mock

from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core import anima_guidance, sam3_args
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests import test_comfy_backend_regressions as regressions
from tests.test_comfy_anima38_compiler import _anima_capabilities, _modules, V2_MODEL
from tests.test_comfy_sam3_lora_branch import (
    ALICE, INK, _LORA_CLASSES, _controlled_workflow, _guidance_scripts, _live_node_ids, _main_stack_loras,
    _prompt_texts, _upstream_loras,
)
from tests.test_comfy_workflow_compiler import _capabilities, _custom_workflow

_png = regressions._png

_AD_TITLE = "(ADetailer LoRA branch)"
_SAMPLERS = {"ForgeNeoKSamplerCNS", "KSampler"}
_IMPACT_LORA_RE = re.compile(r"<lora:([^>]+)>")          # Impact wildcards.py:845 extract_lora_values


def _slot(**values):
    return {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt", **values}


def _payload(prompt, slots, *, negative="bad anatomy", scripts=None, anima=False, sam3=None, **extra):
    alwayson = copy.deepcopy(scripts or {})
    alwayson["ADetailer"] = {"args": [True, False, *[_slot(**slot) for slot in slots]]}
    if sam3 is not None:
        alwayson.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam3}))
    payload = {"prompt": prompt, "negative_prompt": negative, "cfg_scale": 5.0, "seed": 7,
               "alwayson_scripts": alwayson, **extra}
    if anima:
        payload["forge_additional_modules"] = _modules()
    return payload


def _compile(path, payload, *, anima=False, capabilities=None, workflow=None, controls=None):
    compiler = ComfyWorkflowCompiler(capabilities or (_anima_capabilities() if anima else _capabilities()))
    model = V2_MODEL if anima else "checkpoint.safetensors"
    if path == "in-generation":
        return compiler.compile("txt2img", model, payload)
    if path == "custom workflow":
        return compiler.compile("txt2img", model, payload, workflow=workflow or _custom_workflow(),
                                workflow_controls=controls)
    return compiler.compile_postprocess(model, payload, uploaded_image="in.png")


def _identity_stack(_self, _graph, _slot, _index, _model, clip, positive, negative, node_model, _base):
    """이 규칙 전의 동작 — ADetailer 노드는 늘 메인 스택(마지막 본 패스의 모델)을 받았다."""
    return node_model, clip, positive, negative


def _ad_nodes(graph):
    return [node["inputs"] for node in graph.values() if node["class_type"] == "ForgeNeoADetailer"]


def _settings(inputs):
    return json.loads(inputs["settings_json"])


def _sampler(graph):
    [inputs] = [node["inputs"] for node in graph.values() if node["class_type"] in _SAMPLERS]
    return inputs


def _titles(graph, ids):
    return [str((graph[node_id].get("_meta") or {}).get("title") or "") for node_id in ids]


class AdetailerLoraBranchTests(unittest.TestCase):
    PATHS = ("in-generation", "custom workflow", "standalone")

    def assert_adetailer_loras(self, graph, expected, *, main=None):
        """모든 ADetailer 노드의 model·clip·positive·negative 위 LoRA 로더 = expected, Impact 에 넘기는 글에는
        LoRA 태그가 없다(Impact 가 더 걸지 않는다). main 이 있으면 메인 샘플러가 그 목록이고 분기 노드를 읽지 않는다."""
        nodes = _ad_nodes(graph)
        self.assertTrue(nodes)
        for inputs in nodes:
            for key in ("model", "clip", "positive", "negative"):
                self.assertEqual(_upstream_loras(graph, inputs[key]), sorted(expected), key)
            self.assertEqual(_IMPACT_LORA_RE.findall(_settings(inputs)["prompt"]), [])
        if main is not None:
            sampler = _sampler(graph)
            links = (sampler["model"], sampler["positive"], sampler["negative"])
            self.assertEqual(_upstream_loras(graph, *links), sorted(main))
            ids = ComfyWorkflowCompiler._upstream_node_ids(graph, links)
            self.assertFalse([title for title in _titles(graph, ids) if "LoRA branch" in title])

    # ---- 같은 목록 → 그래프 그대로(태그만 뗀다) ------------------------------------------------------

    def test_default_flows_keep_the_main_stack_and_the_graph(self):
        """기본 흐름은 메인 스택 그대로다 — 이 규칙 전(_identity_stack)과 바이트 단위로 같은 그래프이고, 달라지는 것은
        ad_prompt 에 태그가 있을 때 Impact 에 넘기는 글에서 그 태그를 뗀 것뿐이다(예전에는 Impact 가 메인 위에 한 번 더
        걸었다 — 이중 적용). 앱은 빈 ad_prompt 를 채우지 않는다(ui/generator_generation.py _build_adetailer_slot)."""
        main = "portrait, <lora:ink:0.8>"
        cases = {
            "empty ad_prompt = main prompt": (main, [{}], None),
            "two empty slots": (main, [{}, {"ad_negative_prompt": "lowres"}], None),
            "tagless ad_prompt, main without LoRA": ("portrait", [{"ad_prompt": "detailed face"}], None),
            "same tag as main": (main, [{"ad_prompt": "face, <lora:ink:0.8>"}], "face, "),
            "same file, full path": (main, [{"ad_prompt": "face, <lora:styles/ink.safetensors:0.8>"}], "face, "),
            "[PROMPT] brings the main prompt": (main, [{"ad_prompt": "[PROMPT], detailed face"}], None),
            "ad_copy_main_loras on a tagless prompt": (main, [{"ad_prompt": "face", "ad_copy_main_loras": True}], None),
        }
        for anima in (False, True):
            for path in self.PATHS:
                if path == "custom workflow" and anima:
                    continue
                for label, (prompt, slots, stripped) in cases.items():
                    with self.subTest(anima=anima, path=path, case=label):
                        payload = _payload(prompt, slots, scripts=_guidance_scripts(), anima=anima)
                        graph = _compile(path, copy.deepcopy(payload), anima=anima)
                        with mock.patch.object(ComfyWorkflowCompiler, "_adetailer_pass_stack", _identity_stack), \
                                mock.patch.object(ComfyWorkflowCompiler, "_postprocess_stack_loras", _main_stack_loras):
                            before = _compile(path, copy.deepcopy(payload), anima=anima)
                        self.assertEqual(graph, before)
                        # 스택 선택 밖에서 바뀐 것은 Impact 글뿐이다: 태그가 있던 슬롯만 그 태그를 뗐다(예전: Impact 가
                        # 메인 스택 위에 같은 LoRA 를 한 번 더 걸었다).
                        for slot, node in zip(slots, _ad_nodes(graph)):
                            raw = slot.get("ad_prompt", "")
                            self.assertEqual(_settings(node)["prompt"], raw if stripped is None else stripped)
                            self.assertEqual(len(_IMPACT_LORA_RE.findall(raw)), 0 if stripped is None else 1)
                        self.assertNotIn(_AD_TITLE, json.dumps(graph))
                        expected = [(INK, 0.8, 0.8)] if "lora" in prompt else []
                        self.assert_adetailer_loras(graph, expected)
                        if path != "standalone":
                            self.assertEqual(_sampler(graph)["positive"], _ad_nodes(graph)[0]["positive"])

    # ---- 다른 목록 → 분기 ------------------------------------------------------------------------

    def test_a_different_list_feeds_only_that_list_to_the_adetailer_node(self):
        """태그 없는 ad_prompt 는 LoRA 없음(예전: 메인 LoRA), 태그가 있으면 그 태그만(예전: 메인 + 그 태그). 강도는
        Forge 순서(<lora:n:TE:UNet>). 메인 샘플러는 메인 LoRA 그대로다."""
        cases = (("tagless", "face", []),
                 ("own tag", "face, <lora:alice:0.7>", [(ALICE, 0.7, 0.7)]),
                 ("same LoRA, other weights", "face, <lora:ink:0.3:0.2>", [(INK, 0.2, 0.3)]))
        for anima in (False, True):
            for path in self.PATHS:
                if path == "custom workflow" and anima:
                    continue
                for label, ad_prompt, expected in cases:
                    with self.subTest(anima=anima, path=path, case=label):
                        graph = _compile(path, _payload("portrait, <lora:ink:0.8>", [{"ad_prompt": ad_prompt}],
                                                        anima=anima), anima=anima)
                        main = [(INK, 0.8, 0.8)] if path != "standalone" else None
                        self.assert_adetailer_loras(graph, expected, main=main)
                        [node] = _ad_nodes(graph)
                        self.assertEqual(_settings(node)["prompt"], _IMPACT_LORA_RE.sub("", ad_prompt))
                        self.assertEqual(_settings(node)["ad_prompt"], ad_prompt)
                        if expected:
                            lora_id, clip_index = node["clip"]
                            self.assertIn(graph[lora_id]["class_type"], _LORA_CLASSES)
                            self.assertEqual(clip_index, 1)
                            self.assertEqual(graph[lora_id]["class_type"],
                                             "ForgeNeoAnimaLoraLoader" if anima else "LoraLoader")
                        if path == "standalone":
                            # 스택 자체가 슬롯 목록이다 — 메인 LoRA 는 그래프 어디에도 없다
                            self.assertEqual(sorted((n["inputs"]["lora_name"], n["inputs"]["strength_model"],
                                                     n["inputs"]["strength_clip"]) for n in graph.values()
                                                    if n["class_type"] in _LORA_CLASSES), sorted(expected))

    def test_a_tag_only_ad_prompt_encodes_its_empty_text_not_the_main_prompt(self):
        """Forge: '<lora:alice:0.7>' 는 비지 않은 ad_prompt 라 메인 프롬프트로 채우지 않고 빈 글을 그 LoRA 로 인코딩한다.
        Impact 는 빈 wildcard 이면 positive 입력을 쓰므로, 컴파일러가 Impact 와 같은 인코딩(CLIPTextEncode + 슬롯 clip)
        으로 빈 글을 준다 — 분기·메인과 같은 목록 모두."""
        cases = (("branch", "<lora:alice:0.7>", [(ALICE, 0.7, 0.7)]),
                 ("same list as main", "<lora:ink:0.8>", [(INK, 0.8, 0.8)]),
                 ("uppercase: removed, not loaded", "<LoRA:alice:0.7>", []))
        for path in self.PATHS:
            for label, ad_prompt, expected in cases:
                with self.subTest(path=path, case=label):
                    graph = _compile(path, _payload("portrait, <lora:ink:0.8>", [{"ad_prompt": ad_prompt}]))
                    [node] = _ad_nodes(graph)
                    self.assertEqual(_settings(node)["prompt"], "")
                    encoder = graph[node["positive"][0]]
                    self.assertEqual(encoder["class_type"], "CLIPTextEncode")
                    self.assertEqual(encoder["inputs"], {"clip": node["clip"], "text": ""})
                    self.assertNotIn("portrait", _prompt_texts(graph, node["positive"]))
                    self.assert_adetailer_loras(graph, expected)

    def test_an_empty_ad_prompt_inherits_main_prompt_loras_but_not_negative_ones(self):
        """빈 ad_prompt = 메인 프롬프트(태그 포함). 네거티브의 LoRA 는 Forge 가 걸지 않는다 — 메인 Comfy 경로는 걸지만
        (기존 규약, 그대로) ADetailer 는 메인 프롬프트의 LoRA 만 받는다. 단독이면 스택 자체가 그 목록이다."""
        for path in self.PATHS:
            with self.subTest(path=path):
                graph = _compile(path, _payload("portrait, <lora:ink:0.8>", [{}], negative="bad, <lora:alice:0.4>"))
                main = [(ALICE, 0.4, 0.4), (INK, 0.8, 0.8)] if path != "standalone" else None
                self.assert_adetailer_loras(graph, [(INK, 0.8, 0.8)], main=main)
                if path == "standalone":
                    self.assertEqual([n["inputs"]["lora_name"] for n in graph.values()
                                      if n["class_type"] in _LORA_CLASSES], [INK])

    # ---- 분기 위의 NegPiP·가이던스·Detail Daemon ---------------------------------------------------

    def test_branch_rebuilds_negpip_guidance_and_the_last_pass_detail_daemon(self):
        """분기도 메인과 같은 단계: LoRA → NegPiP → 가이던스 스위트(→ Skimmed) → 마지막 본 패스의 DD(같은 설정·base cfg —
        슬롯 cfg 가 아니다). 스위트의 clip·positive·negative(Modulation 기준)도 분기에서 다시 만든다."""
        for anima in (False, True):
            with self.subTest(anima=anima):
                payload = _payload("portrait, <lora:ink:0.8>", [{
                    "ad_prompt": "face, <lora:alice:0.7>", "ad_use_cfg_scale": True, "ad_cfg_scale": 3.5,
                }], scripts=_guidance_scripts(modulation=True), anima=anima)
                graph = _compile("in-generation", payload, anima=anima)
                [node] = _ad_nodes(graph)
                dd_id, _ = node["model"]
                self.assertEqual(graph[dd_id]["class_type"], "ForgeNeoAnimaDetailDaemon")
                [main_dd] = [n["inputs"] for n in graph.values()
                             if n["class_type"] == "ForgeNeoAnimaDetailDaemon" and _AD_TITLE not in n["_meta"]["title"]]
                self.assertEqual({k: v for k, v in graph[dd_id]["inputs"].items() if k != "model"},
                                 {k: v for k, v in main_dd.items() if k != "model"})
                self.assertEqual(graph[dd_id]["inputs"]["cfg_scale_override"], 5.0)
                chain, link = [], graph[dd_id]["inputs"]["model"]
                while graph[link[0]]["class_type"] not in {"CheckpointLoaderSimple", "ForgeNeoAnima38V2Loader"}:
                    chain.append(graph[link[0]]["class_type"])
                    link = graph[link[0]]["inputs"]["model"]
                lora = "ForgeNeoAnimaLoraLoader" if anima else "LoraLoader"
                self.assertEqual(chain, ["ForgeNeoSkimmedCFG", "ForgeNeoAnimaGuidanceSuite", "ForgeNeoNegPip", lora])
                [suite_id] = [node_id for node_id in ComfyWorkflowCompiler._upstream_node_ids(graph, [node["model"]])
                              if graph[node_id]["class_type"] == "ForgeNeoAnimaGuidanceSuite"]
                suite = graph[suite_id]["inputs"]
                self.assertEqual(_upstream_loras(graph, suite["clip"], suite["positive"], suite["negative"]),
                                 [(ALICE, 0.7, 0.7)])
                self.assert_adetailer_loras(graph, [(ALICE, 0.7, 0.7)], main=[(INK, 0.8, 0.8)])
                branch_ids = [node_id for node_id, n in graph.items() if _AD_TITLE in (n.get("_meta") or {}).get("title", "")]
                self.assertTrue(branch_ids)
                self.assertEqual(set(branch_ids) - _live_node_ids(graph), set())

    def test_the_branch_detail_daemon_follows_the_last_pass(self):
        """ADetailer DD 규칙(_add_detail_daemon): 슬롯은 마지막 본 패스의 모델을 받는다. Hires 를 돌렸고 Hires Pass 가
        꺼져 있으면 마지막 패스에 DD 가 없어 분기에도 없고, Hires Pass 가 켜져 있으면 hires DD 를 분기에 다시 만든다."""
        for dd_hires, expect_dd in ((False, False), (True, True)):
            with self.subTest(dd_hires=dd_hires):
                scripts = _guidance_scripts()
                scripts[anima_guidance.SCRIPT_DETAIL_DAEMON] = {"args": anima_guidance.build_args(
                    anima_guidance.SCRIPT_DETAIL_DAEMON, {"dd_enabled": True, "dd_hires": dd_hires})}
                for ad_prompt, expected in (("", [(INK, 0.8, 0.8)]), ("face", [])):
                    graph = _compile("in-generation", _payload("portrait, <lora:ink:0.8>", [{"ad_prompt": ad_prompt}],
                                                               scripts=scripts, enable_hr=True, hr_scale=1.5))
                    [node] = _ad_nodes(graph)
                    top = graph[node["model"][0]]
                    self.assertEqual(top["class_type"] == "ForgeNeoAnimaDetailDaemon", expect_dd, ad_prompt)
                    if expect_dd:
                        self.assertIn("hires pass", top["_meta"]["title"])
                        self.assertEqual(_AD_TITLE in top["_meta"]["title"], ad_prompt == "face")
                    self.assertEqual(_upstream_loras(graph, node["model"]), expected)

    def test_slots_with_the_same_list_share_one_branch(self):
        """같은 목록의 슬롯은 분기(LoRA 로더·DD) 하나를 같이 쓰고, 목록이 다르면 분기가 따로다. 같은 목록의 SAM3 패스는
        자기(SAM3) 분기를 쓴다 — SAM3 그래프는 ADetailer 가 없을 때와 같은 모양이다."""
        graph = _compile("in-generation", _payload(
            "portrait, <lora:ink:0.8>",
            [{"ad_prompt": "face, <lora:alice:0.7>"}, {"ad_prompt": "hand, <lora:alice:0.7>"},
             {"ad_prompt": "eyes"}, {}],
            scripts=_guidance_scripts(), sam3={"sam3_inpaint_prompt": "face, <lora:alice:0.7>"}))
        first, second, third, fourth = _ad_nodes(graph)
        self.assertEqual((first["model"], first["clip"]), (second["model"], second["clip"]))
        self.assertNotEqual(first["clip"], third["clip"])
        self.assertEqual(_upstream_loras(graph, third["model"], third["clip"]), [])
        self.assertEqual(_upstream_loras(graph, fourth["model"], fourth["clip"]), [(INK, 0.8, 0.8)])
        ad_loras = [n for n in graph.values() if n["class_type"] in _LORA_CLASSES and _AD_TITLE in n["_meta"]["title"]]
        self.assertEqual([n["inputs"]["lora_name"] for n in ad_loras], [ALICE])
        self.assertEqual(len([n for n in graph.values() if n["class_type"] == "ForgeNeoAnimaDetailDaemon"
                              and _AD_TITLE in n["_meta"]["title"]]), 2)          # alice 분기·LoRA 없는 분기
        [sam3] = [n["inputs"] for n in graph.values() if n["class_type"] == "ForgeNeoSAM3Detailer"]
        self.assertEqual(_upstream_loras(graph, sam3["clip"]), [(ALICE, 0.7, 0.7)])
        self.assertIn("(SAM3 LoRA branch)", graph[sam3["clip"][0]]["_meta"]["title"])
        alone = _compile("in-generation", _payload(
            "portrait, <lora:ink:0.8>", [], scripts=_guidance_scripts(),
            sam3={"sam3_inpaint_prompt": "face, <lora:alice:0.7>"}))
        self.assertEqual(sorted(_titles(graph, graph)), sorted(
            [*_titles(alone, alone), *[t for t in _titles(graph, graph) if _AD_TITLE in t or "ADetailer slot" in t]]))

    # ---- Forge 프롬프트 규칙: ad_copy_main_loras·[SEP]·[PROMPT]·[SKIP] ------------------------------

    def test_ad_copy_main_loras_appends_the_main_prompt_lora_tokens(self):
        """Forge 기본은 끔(args.py:63) — 앱은 이 키를 보내지 않는다. 켜진 슬롯이 오면 Forge 처럼 메인 프롬프트의
        ``<…:숫자>`` 토큰(!adetailer.py LORA_RE — ``<lora:x>`` 는 아니다) 중 없는 것을 붙인다."""
        main = "portrait, <lora:ink:0.8>, <lora:alice>"
        cases = (("off (Forge default)", {"ad_prompt": "face, <lora:alice:0.5>"}, [(ALICE, 0.5, 0.5)]),
                 ("on", {"ad_prompt": "face, <lora:alice:0.5>", "ad_copy_main_loras": True},
                  [(ALICE, 0.5, 0.5), (INK, 0.8, 0.8)]),
                 ("on, token already there", {"ad_prompt": "face, <lora:ink:0.8>", "ad_copy_main_loras": "true"},
                  [(INK, 0.8, 0.8)]),
                 ("on, empty prompt = main prompt", {"ad_copy_main_loras": True},
                  [(INK, 0.8, 0.8), (ALICE, 1.0, 1.0)]))
        for path in self.PATHS:
            for label, slot, expected in cases:
                with self.subTest(path=path, case=label):
                    graph = _compile(path, _payload(main, [slot]))
                    self.assert_adetailer_loras(graph, expected)

    def test_sep_parts_must_load_one_list(self):
        """[SEP] 칸마다 p2 가 따로 돌아 LoRA 가 칸마다 다를 수 있다. ComfyUI ADetailer 노드는 모델 하나를 받으므로
        목록이 다르면 컴파일 오류, 같으면 그 목록. [SKIP] 칸은 돌지 않아 세지 않고, 빈 칸은 메인 프롬프트다."""
        main = "portrait, <lora:ink:0.8>"
        ok = (("same tag in every part", "face <lora:alice:0.7> [SEP] hand <lora:alice:0.7>", [(ALICE, 0.7, 0.7)]),
              ("skipped part", "face <lora:alice:0.7> [SEP] [SKIP]", [(ALICE, 0.7, 0.7)]),
              ("empty part = main prompt", "[SEP] hand <lora:ink:0.8>", [(INK, 0.8, 0.8)]),
              ("[PROMPT] = main prompt", "[PROMPT], face", [(INK, 0.8, 0.8)]))
        for path in self.PATHS:
            for label, ad_prompt, expected in ok:
                with self.subTest(path=path, case=label):
                    self.assert_adetailer_loras(_compile(path, _payload(main, [{"ad_prompt": ad_prompt}])), expected)
            for ad_prompt in ("face <lora:alice:0.7> [SEP] hand", "[SEP] hand", "face [SEP] hand <lora:ink:0.5>"):
                with self.subTest(path=path, error=ad_prompt):
                    with self.assertRaisesRegex(WorkflowCompileError, r"ADetailer 슬롯 1의 \[SEP\]"):
                        _compile(path, _payload(main, [{"ad_prompt": ad_prompt}]))

    def test_forge_prompt_grammar_edges(self):
        """!adetailer.py 의 문법 끝자락(검토 무력화 L1·L4·L5·L6 이 살아남던 곳 — 모두 Forge 기준 구현과 같은 답):
        (a) 공백만 적은 ad_prompt 는 빈 글이 아니다(_get_prompt :316 ``if not prompts[n]``) — 메인 프롬프트를 쓰지 않고
            LoRA 없이 돈다. Impact 에는 그 공백이 그대로 간다.
        (b) [SEP] 는 둘레 공백째 나눈다(:314 ``\\s*\\[SEP\\]\\s*``) — 'face <lora:ink:0.8> [SEP] ' 의 둘째 칸은 빈 글이라
            메인 프롬프트, 두 칸 모두 ink 라 메인 스택 그대로다(오류가 아니다).
        (c) [SKIP] 은 칸 전체가 [SKIP] 일 때만 건너뛴다(:1056 ``^\\s*\\[SKIP\\]\\s*$``) — 'face [SKIP]' 는 돈다.
        (d) [SKIP] 판정은 ad_copy_main_loras 가 붙인 뒤다(get_prompt :452 → :1056) — '[SKIP]' 에 메인 LoRA 가 붙어
            '[SKIP], <lora:ink:0.8>' 가 되어 돈다. 목록은 ink 뿐이라(네거티브 alice 는 아니다) 분기다."""
        main = "portrait, <lora:ink:0.8>"
        cases = (
            ("a whitespace-only prompt", {"ad_prompt": "  "}, "bad anatomy", [], True, "  "),
            ("[SEP] with surrounding spaces", {"ad_prompt": "face <lora:ink:0.8> [SEP] "}, "bad anatomy",
             [(INK, 0.8, 0.8)], False, "face  [SEP] "),
            ("[SKIP] inside a part", {"ad_prompt": "face [SKIP]"}, "bad anatomy", [], True, "face [SKIP]"),
            ("[SKIP] after ad_copy_main_loras", {"ad_prompt": "[SKIP]", "ad_copy_main_loras": True},
             "bad, <lora:alice:0.4>", [(INK, 0.8, 0.8)], True, "[SKIP]"),
        )
        for path in self.PATHS:
            for label, slot, negative, expected, branched, impact in cases:
                with self.subTest(path=path, case=label):
                    graph = _compile(path, _payload(main, [slot], negative=negative))
                    self.assert_adetailer_loras(graph, expected)
                    [node] = _ad_nodes(graph)
                    self.assertEqual(_settings(node)["prompt"], impact)
                    if path == "standalone":
                        continue          # 단독은 스택 자체가 그 슬롯 목록이라(_postprocess_stack_loras) 분기가 없다
                    self.assertEqual(_AD_TITLE in json.dumps(graph), branched)
                    main_loras = [(INK, 0.8, 0.8)] + ([(ALICE, 0.4, 0.4)] if "alice" in negative else [])
                    self.assertEqual(_upstream_loras(graph, _sampler(graph)["model"]), sorted(main_loras))

    def test_prompt_placeholder_keeps_negative_loras_out(self):
        """[PROMPT] 는 메인 프롬프트(태그 포함)로 바뀐다 — 네거티브 LoRA 가 있는 메인 스택과는 다른 목록이라 분기다."""
        graph = _compile("in-generation", _payload("portrait, <lora:ink:0.8>", [{"ad_prompt": "[PROMPT], face"}],
                                                   negative="bad, <lora:alice:0.4>"))
        self.assert_adetailer_loras(graph, [(INK, 0.8, 0.8)], main=[(ALICE, 0.4, 0.4), (INK, 0.8, 0.8)])
        self.assertIn(_AD_TITLE, json.dumps(graph))

    # ---- 네거티브·없는 LoRA ---------------------------------------------------------------------

    def test_an_ad_negative_lora_tag_is_literal_text_and_never_loaded(self):
        """Forge 는 네거티브를 파싱하지 않는다 — ad_negative_prompt 의 태그는 걸리지도 떼어지지도 않고 글자 그대로
        인코딩된다. 인코더 clip 은 슬롯 clip(분기면 분기 clip)이다."""
        for path in self.PATHS:
            for ad_prompt, expected in (("", [(INK, 0.8, 0.8)]), ("face", [])):
                with self.subTest(path=path, ad_prompt=ad_prompt):
                    graph = _compile(path, _payload("portrait, <lora:ink:0.8>", [{
                        "ad_prompt": ad_prompt, "ad_negative_prompt": "lowres, <lora:alice:0.4>"}]))
                    [node] = _ad_nodes(graph)
                    encoder = graph[node["negative"][0]]
                    self.assertEqual(encoder["inputs"], {"clip": node["clip"], "text": "lowres, <lora:alice:0.4>"})
                    self.assertEqual(_upstream_loras(graph, node["negative"]), expected)
                    self.assertNotIn(ALICE, [n["inputs"]["lora_name"] for n in graph.values()
                                             if n["class_type"] in _LORA_CLASSES])

    def test_a_missing_adetailer_lora_fails_like_a_missing_main_lora(self):
        with self.assertRaises(WorkflowCompileError) as main_error:
            _compile("in-generation", _payload("portrait, <lora:missing:1>", [{}]))
        self.assertIn("lora_name", str(main_error.exception))
        for path in self.PATHS:
            with self.subTest(path=path):
                with self.assertRaises(WorkflowCompileError) as slot_error:
                    _compile(path, _payload("portrait", [{"ad_prompt": "face, <lora:missing:1>"}]))
                self.assertEqual(str(slot_error.exception), str(main_error.exception))

    def test_standalone_passes_never_resolve_a_main_lora_no_pass_loads(self):
        """단독 후처리: 쓰는 패스(ADetailer 슬롯·SAM3) 중 메인 목록(프롬프트 + 네거티브)을 쓰는 것이 없으면 스택을 첫
        패스의 목록으로 만든다 — 없는 메인 LoRA 도 오류가 아니고, 출력에 닿지 않는 메인 스택도 없다. 다른 목록의 패스는
        가른다(SAM3 분기). 한 슬롯이라도 메인 목록을 쓰면 예전처럼 푼다."""
        kept = _LORA_CLASSES | {"ForgeNeoNegPip", "CLIPTextEncode", "ForgeNeoAnimaGuidanceSuite", "ForgeNeoSkimmedCFG"}
        cases = (("slot only", [{"ad_prompt": "face, <lora:alice:0.6>"}], None, [(ALICE, 0.6, 0.6)], None),
                 ("two slots", [{"ad_prompt": "face"}, {"ad_prompt": "hand, <lora:alice:0.6>"}], None, [], None),
                 ("slot + SAM3", [{"ad_prompt": "face, <lora:alice:0.6>"}], {"sam3_inpaint_prompt": "eyes"},
                  [(ALICE, 0.6, 0.6)], []))
        for label, slots, sam3, first, sam3_expected in cases:
            with self.subTest(label):
                graph = _compile("standalone", _payload("portrait, <lora:missing:1>", slots, sam3=sam3,
                                                        negative="bad, <lora:ink:0.4>",
                                                        scripts=_guidance_scripts(modulation=True)))
                nodes = _ad_nodes(graph)
                self.assertEqual(_upstream_loras(graph, nodes[0]["model"], nodes[0]["clip"]), first)
                self.assertNotIn("missing", json.dumps(graph))
                self.assertNotIn(INK, json.dumps(graph))
                live = _live_node_ids(graph)
                self.assertEqual([(node_id, node["class_type"]) for node_id, node in graph.items()
                                  if node["class_type"] in kept and node_id not in live], [])
                if sam3_expected is not None:
                    [sam3_node] = [n["inputs"] for n in graph.values() if n["class_type"] == "ForgeNeoSAM3Detailer"]
                    self.assertEqual(_upstream_loras(graph, sam3_node["model"], sam3_node["clip"]), sam3_expected)
        with self.subTest("an empty ad_prompt uses the main list, so it still resolves"):
            with self.assertRaisesRegex(WorkflowCompileError, "lora_name.*missing"):
                _compile("standalone", _payload("portrait, <lora:missing:1>",
                                                [{"ad_prompt": "face, <lora:alice:0.6>"}, {}]))

    # ---- 사용자 워크플로 상세 설정·실제 백엔드 ---------------------------------------------------------

    def test_workflow_controls_reach_the_conditioning_nodes_the_branch_copies(self):
        """분기가 다시 만든 조건 노드(사용자 워크플로 노드의 복제본)도 상세 설정 값을 받는다(_Graph.copied_from →
        apply_controls(copies=)). 같은 목록이면 복제본이 없다."""
        capabilities, workflow, binding = _controlled_workflow()
        for label, ad_prompt, copies in (("reuse", "", 1), ("branch", "face, <lora:alice:0.7>", 2)):
            with self.subTest(label):
                graph = _compile("custom workflow", _payload("portrait, <lora:ink:0.8>", [{"ad_prompt": ad_prompt}],
                                                             scripts=_guidance_scripts(modulation=True)),
                                 capabilities=capabilities, workflow=workflow, controls=binding)
                ranges = [n["inputs"] for n in graph.values() if n["class_type"] == "ConditioningSetTimestepRange"]
                strengths = [n["inputs"] for n in graph.values() if n["class_type"] == "ConditioningSetAreaStrength"]
                self.assertEqual([(i["start"], i["end"]) for i in ranges], [(0.0, 0.5)] * copies)
                self.assertEqual([i["strength"] for i in strengths], [0.3] * copies)
                self.assertEqual(_sampler(graph)["positive"], ["9", 0])
                [node] = _ad_nodes(graph)
                self.assertEqual(graph[node["negative"][0]]["inputs"]["strength"], 0.3)
                self.assertEqual(graph[node["positive"][0]]["inputs"]["end"], 0.5)

    def test_standalone_adetailer_through_the_backend_loads_the_prompt_lora_once(self):
        """단독/배치 ADetailer(EXIF 프롬프트 등): payload 프롬프트 = ad_prompt 라 메인 스택이 이미 그 LoRA 를 건다. 예전에는
        Impact 가 같은 태그를 그 위에 또 걸었다(이중 적용). 이제 로더는 한 번이고 Impact 글에는 태그가 없다. 저장된 T2I
        프롬프트의 LoRA 는 쓰지 않는다."""
        backend = regressions.TestStandaloneComfyContext._backend(self, {})
        backend._last_generation_context["payload"]["prompt"] = "portrait, <lora:ink:0.8>"
        backend.adetailer(_png(), {"ad_model": "face_yolov8n.pt", "ad_prompt": "1girl, face, <lora:alice:0.6>",
                                   "ad_negative": "bad"})
        graph = backend._queue_and_wait.call_args.args[0]
        self.assertEqual([n["inputs"]["lora_name"] for n in graph.values() if n["class_type"] in _LORA_CLASSES], [ALICE])
        self.assert_adetailer_loras(graph, [(ALICE, 0.6, 0.6)])
        [node] = _ad_nodes(graph)
        self.assertEqual(_settings(node)["prompt"], "1girl, face, ")
        self.assertNotIn(_AD_TITLE, json.dumps(graph))

    def test_standalone_adetailer_with_an_empty_prompt_never_reads_the_saved_generation(self):
        """Forge 단독/배치 ADetailer 는 img2img 프롬프트 = ad_prompt, 네거티브 = ad_negative(webui_backend.adetailer →
        _build_postprocess_payload) — 빈 ad_prompt 슬롯은 그 빈 메인 프롬프트를 물려받아(!adetailer.py _get_prompt)
        LoRA 없이 '' 를 인코딩한다. 저장된 마지막 T2I 프롬프트(태그 포함)·네거티브는 쓰지 않는다. BatchView 기본값
        (adPrompt '', EXIF 끔)이 이 경우다. SAM3·Refine 은 저장된 프롬프트를 계속 쓴다(sam3_args.build_state)."""
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        backend._last_generation_context = {"model_name": "checkpoint.safetensors", "payload": {
            "prompt": "portrait, <lora:ink:0.8>", "negative_prompt": "bad, <lora:alice:0.4>", "alwayson_scripts": {}}}
        backend._workflow_compiler = mock.Mock(return_value=ComfyWorkflowCompiler(_capabilities()))
        backend._upload_image = mock.Mock(return_value="source.png")
        backend._queue_and_wait = mock.Mock(return_value=GenerationResult(success=True, image_data=b"x"))
        for label, settings in (("single/batch default", {"ad_model": "face_yolov8n.pt", "ad_prompt": ""}),
                                ("no prompt keys at all", {"ad_model": "face_yolov8n.pt"})):
            with self.subTest(label):
                backend.adetailer(_png(), settings)
                graph = backend._queue_and_wait.call_args.args[0]
                self.assertEqual([n for n in graph.values() if n["class_type"] in _LORA_CLASSES], [])
                [node] = _ad_nodes(graph)
                self.assertEqual(_settings(node)["prompt"], "")
                for key in ("positive", "negative"):
                    encoder = graph[node[key][0]]
                    self.assertEqual(encoder["class_type"], "CLIPTextEncode")
                    self.assertEqual(encoder["inputs"]["text"], "")
                self.assertNotIn("portrait", json.dumps(graph))
        with self.subTest("SAM3 keeps the saved-prompt fallback"):
            backend.sam3(_png(), {"sam3_prompt": "face"})
            graph = backend._queue_and_wait.call_args.args[0]
            [sam3] = [n["inputs"] for n in graph.values() if n["class_type"] == "ForgeNeoSAM3Detailer"]
            self.assertEqual(sam3["inpaint_prompt"], "portrait, ")
            self.assertEqual(_upstream_loras(graph, sam3["model"], sam3["clip"]), [(INK, 0.8, 0.8)])


if __name__ == "__main__":
    unittest.main()
