"""SAM3 패스의 LoRA — ComfyUI 에서도 Forge 처럼 그 패스 프롬프트의 LoRA 만 건다(사용자 결정 2026-09-27 "Forge와 똑같이").

Forge 는 process_images 배치마다 forge_objects 를 forge_objects_original(LoRA 없음)로 되돌리고 그 배치의 positive
프롬프트에서만 LoRA 를 건다(modules/processing.py:947, :967 → extra_networks.parse_prompts). SAM3 p2 도
process_images 라 자기 인페인트 프롬프트(생성 안에서 비면 메인 프롬프트 — copy_prompt)의 LoRA 만 받고, 메인 LoRA 는
따라오지 않으며, 네거티브의 ``<lora:>`` 는 걸리지 않고 글자 그대로 인코딩된다.

컴파일러(core/comfy_workflow_compiler.py ``_sam3_pass_stack``): 목록이 메인과 같으면 메인 스택 그대로(그래프 불변),
다르면 LoRA 앞에서 갈라 그 패스 LoRA → NegPiP → 조건 → 가이던스 → (SAM3 패스 규칙의) Detail Daemon 분기를 만든다.
검토 반영: 태그만 적은 인페인트 프롬프트는 뗀 빈 글을 인코딩한다(메인 프롬프트로 채우지 않는다), 분기가 복제한 사용자
워크플로 조건 노드도 상세 설정 값을 받는다, ADetailer 없는 단독 SAM3/Refine 은 스택 자체를 그 패스 목록으로 만든다
(``_postprocess_stack_loras`` — 쓰지 않는 메인 LoRA 를 풀지 않는다).
"""
import copy
import json
import unittest
from unittest import mock

from core import anima_guidance, sam3_args
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_controls import describe_controls
from tests import test_comfy_backend_regressions as regressions
from tests.test_comfy_anima38_compiler import _anima_capabilities, _modules, V2_MODEL
from tests.test_comfy_workflow_compiler import _capabilities, _custom_workflow

_png = regressions._png

INK = "styles/ink.safetensors"
ALICE = "characters/alice.safetensors"
_LORA_CLASSES = {"LoraLoader", "ForgeNeoAnimaLoraLoader"}
_SAM3_CLASSES = {"ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine"}
_SAMPLERS = {"ForgeNeoKSamplerCNS", "KSampler"}
_BRANCH_TITLE = "(SAM3 LoRA branch)"


def _identity_stack(_self, _graph, _state, model, clip, positive, negative, _base):
    """이 규칙 전의 동작 — SAM3 패스는 늘 메인 스택을 받았다."""
    return model, clip, positive, negative


def _main_stack_loras(_self, _payload, loras, _prompt_loras, **_kwargs):
    """이 규칙 전의 동작 — 단독 후처리 스택은 늘 payload 프롬프트 + 네거티브의 LoRA 로 만들었다.
    (ADetailer LoRA 분기) 컴파일러가 main_prompt= 도 넘긴다 — 옛 동작은 읽지 않는다."""
    return loras


def _guidance_scripts(*, modulation=False):
    guidance = anima_guidance.default_settings()
    guidance.update({"guid_enabled": True, "guid_attn_method": "PAG", "guid_mod_enabled": modulation})
    scripts = anima_guidance.build_alwayson(guidance)
    scripts["NegPiP"] = {"args": [True]}
    scripts[anima_guidance.SCRIPT_SKIMMED_CFG] = {"args": anima_guidance.build_args(
        anima_guidance.SCRIPT_SKIMMED_CFG, {"skim_enabled": True})}
    scripts[anima_guidance.SCRIPT_DETAIL_DAEMON] = {"args": anima_guidance.build_args(
        anima_guidance.SCRIPT_DETAIL_DAEMON, {"dd_enabled": True})}
    return scripts


def _payload(prompt, sam3, *, negative="bad anatomy", scripts=None, anima=False, **extra):
    alwayson = copy.deepcopy(scripts or {})
    alwayson.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam3}))
    payload = {"prompt": prompt, "negative_prompt": negative, "cfg_scale": 5.0, "seed": 7,
               "alwayson_scripts": alwayson, **extra}
    if anima:
        payload["forge_additional_modules"] = _modules()
    return payload


def _compile(path, payload, *, anima=False):
    compiler = ComfyWorkflowCompiler(_anima_capabilities() if anima else _capabilities())
    model = V2_MODEL if anima else "checkpoint.safetensors"
    if path == "in-generation":
        return compiler.compile("txt2img", model, payload)
    if path == "custom workflow":
        return compiler.compile("txt2img", model, payload, workflow=_custom_workflow())
    detailer = "ForgeNeoSAM3Refine" if path == "refine" else "ForgeNeoSAM3Detailer"
    return compiler.compile_postprocess(model, payload, uploaded_image="in.png", sam3_detailer_class=detailer)


def _upstream_loras(graph, *links):
    ids = ComfyWorkflowCompiler._upstream_node_ids(graph, links)
    return sorted(
        (graph[node_id]["inputs"]["lora_name"], graph[node_id]["inputs"]["strength_model"],
         graph[node_id]["inputs"]["strength_clip"])
        for node_id in ids if graph[node_id]["class_type"] in _LORA_CLASSES
    )


def _sam3_nodes(graph):
    return [node["inputs"] for node in graph.values() if node["class_type"] in _SAM3_CLASSES]


def _sampler(graph):
    [inputs] = [node["inputs"] for node in graph.values() if node["class_type"] in _SAMPLERS]
    return inputs


def _below_detail_daemon(graph, link):
    while graph[link[0]]["class_type"] == "ForgeNeoAnimaDetailDaemon":
        link = graph[link[0]]["inputs"]["model"]
    return link


def _live_node_ids(graph):
    """출력 노드와 그 위 노드 — ComfyUI 는 출력에 닿지 않는 노드를 실행하지 않는다."""
    outputs = [node_id for node_id, node in graph.items() if node["class_type"] in {"SaveImage", "PreviewImage"}]
    return set(outputs) | ComfyWorkflowCompiler._upstream_node_ids(
        graph, [link for node_id in outputs for link in graph[node_id]["inputs"].values()])


def _prompt_texts(graph, link):
    return {graph[node_id]["inputs"].get("text", graph[node_id]["inputs"].get("prompt"))
            for node_id in ComfyWorkflowCompiler._upstream_node_ids(graph, [link])}


def _controlled_workflow():
    """조건 체인에 상세 설정(워크플로 컨트롤)을 건 노드가 있는 사용자 워크플로와 그 capability·바인딩.

    positive: CLIPTextEncode 2 → ConditioningSetTimestepRange 9(end 0.5 로 덮어씀) → 샘플러,
    negative: CLIPTextEncode 3 → ConditioningSetAreaStrength 10(strength 0.3 으로 덮어씀) → 샘플러."""
    capabilities = _capabilities()
    capabilities["ConditioningSetTimestepRange"] = {"input": {"required": {
        "conditioning": ["CONDITIONING", {}],
        "start": ["FLOAT", {"min": 0.0, "max": 1.0}], "end": ["FLOAT", {"min": 0.0, "max": 1.0}],
    }}}
    capabilities["ConditioningSetAreaStrength"] = {"input": {"required": {
        "conditioning": ["CONDITIONING", {}], "strength": ["FLOAT", {"min": 0.0, "max": 10.0}],
    }}}
    workflow = _custom_workflow()
    workflow["9"] = {"class_type": "ConditioningSetTimestepRange",
                     "inputs": {"conditioning": ["2", 0], "start": 0.0, "end": 1.0}}
    workflow["10"] = {"class_type": "ConditioningSetAreaStrength",
                      "inputs": {"conditioning": ["3", 0], "strength": 1.0}}
    workflow["5"]["inputs"].update(positive=["9", 0], negative=["10", 0])
    schema = describe_controls(workflow, capabilities)
    binding = {"workflowFingerprint": schema["workflowFingerprint"],
               "schemaFingerprint": schema["schemaFingerprint"], "overrides": [
                   {"nodeId": "9", "name": "end", "classType": "ConditioningSetTimestepRange", "value": 0.5},
                   {"nodeId": "10", "name": "strength", "classType": "ConditioningSetAreaStrength", "value": 0.3},
               ]}
    return capabilities, workflow, binding


class Sam3LoraBranchTests(unittest.TestCase):
    PATHS = ("in-generation", "custom workflow", "standalone sam3", "refine")

    def assert_sam3_loras(self, graph, expected, *, main=None):
        """SAM3 노드의 model·clip·positive·negative 위 LoRA 로더 = expected(메인 샘플러는 main)."""
        nodes = _sam3_nodes(graph)
        self.assertTrue(nodes)
        for inputs in nodes:
            for key in ("model", "clip", "positive", "negative"):
                self.assertEqual(_upstream_loras(graph, inputs[key]), sorted(expected), key)
        if main is not None:
            sampler = _sampler(graph)
            self.assertEqual(_upstream_loras(graph, sampler["model"], sampler["positive"],
                                             sampler["negative"]), sorted(main))

    # ---- 같은 목록 → 그래프 그대로 --------------------------------------------------------

    def test_equal_lora_list_leaves_the_graph_exactly_as_before(self):
        """기본 흐름(빈 인페인트 프롬프트·앱이 채운 메인 프롬프트·같은 태그의 다른 글·같은 파일의 다른 철자)은 SAM3 패스가
        메인 스택을 그대로 받는다 — 이 규칙 전(_identity_stack)과 바이트 단위로 같은 그래프다."""
        main = "portrait, <lora:ink:0.8>"
        prompts = {"empty": "", "app fill = main prompt": main,
                   "same tags, other text": "detailed face, <lora:ink:0.8>",
                   "same file, full path": "face, <lora:styles/ink.safetensors:0.8>"}
        for anima in (False, True):
            for path in self.PATHS:
                if path == "custom workflow" and anima:
                    continue
                for label, text in prompts.items():
                    with self.subTest(anima=anima, path=path, sam3=label):
                        payload = _payload(main, {"sam3_inpaint_prompt": text}, scripts=_guidance_scripts(),
                                           anima=anima, _comfy_detail_passes=(
                                               ["eyes"] if path in {"in-generation", "custom workflow"} else []))
                        graph = _compile(path, copy.deepcopy(payload), anima=anima)
                        with mock.patch.object(ComfyWorkflowCompiler, "_sam3_pass_stack", _identity_stack), \
                                mock.patch.object(ComfyWorkflowCompiler, "_postprocess_stack_loras", _main_stack_loras):
                            before = _compile(path, copy.deepcopy(payload), anima=anima)
                        self.assertEqual(graph, before)
                        self.assertNotIn(_BRANCH_TITLE, json.dumps(graph))
                        self.assertEqual(len([n for n in graph.values() if n["class_type"] in _LORA_CLASSES]), 1)
                        self.assert_sam3_loras(graph, [(INK, 0.8, 0.8)])
                        if path in {"in-generation", "custom workflow"}:
                            sampler = _sampler(graph)
                            for inputs in _sam3_nodes(graph):
                                self.assertEqual(_below_detail_daemon(graph, inputs["model"]),
                                                 _below_detail_daemon(graph, sampler["model"]))
                                self.assertEqual((inputs["positive"], inputs["negative"]),
                                                 (sampler["positive"], sampler["negative"]))

    # ---- 다른 목록 → 분기 ----------------------------------------------------------------

    def test_a_different_lora_feeds_only_that_lora_to_the_sam3_node(self):
        """SAM3 프롬프트의 LoRA 만 SAM3 노드의 model·clip(·조건) 위에 있다. 메인 샘플러는 메인 LoRA 그대로."""
        for anima in (False, True):
            for path in self.PATHS:
                if path == "custom workflow" and anima:
                    continue
                with self.subTest(anima=anima, path=path):
                    graph = _compile(path, _payload("portrait, <lora:ink:0.8>",
                                                    {"sam3_inpaint_prompt": "face, <lora:alice:0.7>"},
                                                    anima=anima), anima=anima)
                    main = [(INK, 0.8, 0.8)] if path in {"in-generation", "custom workflow"} else None
                    self.assert_sam3_loras(graph, [(ALICE, 0.7, 0.7)], main=main)
                    [sam3] = _sam3_nodes(graph)
                    self.assertEqual(sam3["inpaint_prompt"], "face, ")        # 태그만 뗀다(P7)
                    lora_id, clip_index = sam3["clip"]
                    self.assertEqual(graph[lora_id]["inputs"]["lora_name"], ALICE)   # 노드 clip = 분기 clip
                    self.assertEqual(clip_index, 1)

    def test_a_tagless_sam3_prompt_gets_no_lora_while_main_keeps_its_style_lora(self):
        for path in self.PATHS:
            with self.subTest(path=path):
                graph = _compile(path, _payload("portrait, <lora:ink:0.8>", {"sam3_inpaint_prompt": "face"}))
                main = [(INK, 0.8, 0.8)] if path in {"in-generation", "custom workflow"} else None
                self.assert_sam3_loras(graph, [], main=main)
                [sam3] = _sam3_nodes(graph)
                self.assertNotIn("LoraLoader", {graph[node_id]["class_type"] for node_id in
                                                ComfyWorkflowCompiler._upstream_node_ids(graph, [sam3["model"]])})

    def test_the_same_lora_at_another_weight_uses_only_the_sam3_weight(self):
        # <lora:ink:0.3:0.2> 는 Forge 순서로 텍스트 인코더 0.3·UNet 0.2 → (이름, strength_model 0.2, strength_clip 0.3).
        # (예전 기대값 (0.3, 0.2) 는 첫 값을 model 로 읽었다.)
        for path in self.PATHS:
            with self.subTest(path=path):
                graph = _compile(path, _payload("portrait, <lora:ink:0.8>",
                                                {"sam3_inpaint_prompt": "face, <lora:ink:0.3:0.2>"}))
                main = [(INK, 0.8, 0.8)] if path in {"in-generation", "custom workflow"} else None
                self.assert_sam3_loras(graph, [(INK, 0.2, 0.3)], main=main)

    def test_an_empty_sam3_prompt_inherits_the_main_prompt_loras_but_not_negative_ones(self):
        """Forge copy_prompt: 빈 인페인트 프롬프트 = 메인 프롬프트(태그 포함). 네거티브의 LoRA 는 Forge 가 어디서도 걸지
        않는다 — 메인 Comfy 경로는 걸지만(기존 규약, 그대로) SAM3 패스는 메인 프롬프트의 LoRA 만 받는다."""
        payload = _payload("portrait, <lora:ink:0.8>", {"sam3_inpaint_prompt": ""},
                           negative="bad anatomy, <lora:alice:0.4>")
        graph = _compile("in-generation", payload)
        self.assert_sam3_loras(graph, [(INK, 0.8, 0.8)], main=[(ALICE, 0.4, 0.4), (INK, 0.8, 0.8)])

    def test_refine_through_the_backend_drops_a_lora_inherited_into_the_negative(self):
        """Refine(D2 로 메인 프롬프트·네거티브를 물려받음): payload 프롬프트 = 인페인트 프롬프트라 기본은 메인 스택 그대로.
        물려받은 네거티브에 LoRA 태그가 있으면 Forge p2 는 그것을 걸지 않는다 — 스택에는 프롬프트 LoRA 만.
        (검토 3) 단독 후처리 스택 자체가 SAM3 패스 목록이라 분기 없이 그래프 어디에도 네거티브 LoRA 가 없다(예전 기대값은
        메인 스택에 걸고 분기에서 뺐다)."""
        for negative in ("bad", "bad, <lora:alice:0.4>"):
            with self.subTest(negative=negative):
                backend = regressions.TestStandaloneComfyContext._backend(self, {})
                backend.refine(_png(), {"target": "shirt", "replacement": "blue shirt",
                                        "main_prompt": "1girl, <lora:ink:0.8>", "main_negative": negative})
                graph = backend._queue_and_wait.call_args.args[0]
                self.assert_sam3_loras(graph, [(INK, 0.8, 0.8)])
                self.assertNotIn(_BRANCH_TITLE, json.dumps(graph))
                self.assertEqual([node["inputs"]["lora_name"] for node in graph.values()
                                  if node["class_type"] in _LORA_CLASSES], [INK])

    def test_standalone_sam3_through_the_backend_uses_its_own_prompt_loras(self):
        """단독 SAM3: 인페인트 프롬프트가 비면 저장된 T2I 프롬프트(LoRA 포함)를 물려받아 메인 스택 그대로, 사용자가 적은
        프롬프트면 그 LoRA 만. (검토 3) 스택 자체를 그 목록으로 만들어 분기가 없다(예전 기대값은 분기였다)."""
        for sam3, expected in (({}, [(INK, 0.8, 0.8)]),
                               ({"sam3_inpaint_prompt": "face, <lora:alice:0.6>"}, [(ALICE, 0.6, 0.6)]),
                               ({"sam3_inpaint_prompt": "face"}, [])):
            with self.subTest(sam3=sam3):
                backend = regressions.TestStandaloneComfyContext._backend(self, {})
                backend._last_generation_context["payload"]["prompt"] = "portrait, <lora:ink:0.8>"
                backend.sam3(_png(), {"sam3_prompt": "face", **sam3})
                graph = backend._queue_and_wait.call_args.args[0]
                self.assert_sam3_loras(graph, expected)
                self.assertNotIn(_BRANCH_TITLE, json.dumps(graph))
                self.assertEqual(sorted((node["inputs"]["lora_name"], node["inputs"]["strength_model"],
                                         node["inputs"]["strength_clip"]) for node in graph.values()
                                        if node["class_type"] in _LORA_CLASSES), sorted(expected))

    # ---- 분기 위의 가이던스·NegPiP·Detail Daemon ------------------------------------------------

    def test_guidance_negpip_and_detail_daemon_stay_on_the_branch(self):
        """분기도 메인과 같은 단계: LoRA → NegPiP → 가이던스 스위트(→ Skimmed) → SAM3 패스 DD(그 패스 cfg). 스위트의
        clip·positive·negative(Modulation 기준)도 분기에서 다시 만든다 — 메인 LoRA 가 그 길로 섞이지 않는다.
        순차 보정 패스는 같은 분기를 다시 쓰고(분기 LoRA 로더 하나) DD 만 패스마다 건다."""
        for anima in (False, True):
            with self.subTest(anima=anima):
                payload = _payload("portrait, <lora:ink:0.8>", {
                    "sam3_inpaint_prompt": "face, <lora:alice:0.7>",
                    "sam3_use_cfg_scale": True, "sam3_cfg_scale": 3.5,
                }, scripts=_guidance_scripts(modulation=True), anima=anima, _comfy_detail_passes=["eyes"])
                graph = _compile("in-generation", payload, anima=anima)
                nodes = _sam3_nodes(graph)
                self.assertEqual(len(nodes), 2)
                for inputs in nodes:
                    dd_id, _ = inputs["model"]
                    self.assertEqual(graph[dd_id]["class_type"], "ForgeNeoAnimaDetailDaemon")
                    self.assertEqual(graph[dd_id]["inputs"]["cfg_scale_override"], 3.5)
                    chain, link = [], graph[dd_id]["inputs"]["model"]
                    while graph[link[0]]["class_type"] not in {"CheckpointLoaderSimple", "ForgeNeoAnima38V2Loader"}:
                        chain.append(graph[link[0]]["class_type"])
                        link = graph[link[0]]["inputs"]["model"]
                    lora = "ForgeNeoAnimaLoraLoader" if anima else "LoraLoader"
                    self.assertEqual(chain, ["ForgeNeoSkimmedCFG", "ForgeNeoAnimaGuidanceSuite", "ForgeNeoNegPip", lora])
                    [suite_id] = [node_id for node_id in ComfyWorkflowCompiler._upstream_node_ids(graph, [inputs["model"]])
                                  if graph[node_id]["class_type"] == "ForgeNeoAnimaGuidanceSuite"]
                    suite = graph[suite_id]["inputs"]
                    self.assertEqual(_upstream_loras(graph, suite["clip"], suite["positive"], suite["negative"]),
                                     [(ALICE, 0.7, 0.7)])
                    self.assertEqual(json.loads(suite["settings_json"])["guid_mod_enabled"], True)
                self.assertNotEqual(nodes[0]["model"], nodes[1]["model"])            # DD 는 패스마다
                self.assertEqual(nodes[0]["clip"], nodes[1]["clip"])                 # 분기는 하나
                self.assertEqual(len([n for n in graph.values() if n["class_type"] in _LORA_CLASSES]), 2)
                self.assert_sam3_loras(graph, [(ALICE, 0.7, 0.7)], main=[(INK, 0.8, 0.8)])
                suites = [n for n in graph.values() if n["class_type"] == "ForgeNeoAnimaGuidanceSuite"]
                self.assertEqual(len(suites), 2)                                       # 메인 + 분기
                self.assertEqual(suites[0]["inputs"]["settings_json"], suites[1]["inputs"]["settings_json"])

    # ---- 네거티브·없는 LoRA ---------------------------------------------------------------

    def test_a_negative_lora_tag_is_literal_text_and_never_loaded(self):
        """Forge 는 네거티브 프롬프트를 파싱하지 않는다(parse_prompts 는 p.prompts·hr_prompts 만) — 태그는 걸리지도
        떼어지지도 않고 글자 그대로 인코딩된다. SAM3 노드의 negative_prompt 도 그대로다."""
        for path in self.PATHS:
            with self.subTest(path=path):
                graph = _compile(path, _payload("portrait", {"sam3_inpaint_prompt": "face",
                                                             "sam3_negative_prompt": "lowres, <lora:alice:0.4>"}))
                [sam3] = _sam3_nodes(graph)
                self.assertEqual(sam3["negative_prompt"], "lowres, <lora:alice:0.4>")
                self.assertEqual([n for n in graph.values() if n["class_type"] in _LORA_CLASSES], [])

    def test_a_missing_sam3_lora_fails_like_a_missing_main_lora(self):
        """SAM3 프롬프트의 없는 LoRA 는 메인 패스의 없는 LoRA 와 같은 컴파일 오류다(같은 _resolve_choice).
        (검토 3) 단독 SAM3/Refine 은 메인 프롬프트의 LoRA 를 걸지 않으므로 그 LoRA 가 없어도 오류가 아니다 —
        예전 기대값은 네 경로 모두 메인 쪽 오류를 요구했다(test_standalone_passes_never_resolve_main_loras_they_do_not_load)."""
        with self.assertRaises(WorkflowCompileError) as main_error:
            _compile("in-generation", _payload("portrait, <lora:missing:1>", {"sam3_inpaint_prompt": "face"}))
        self.assertIn("lora_name", str(main_error.exception))
        for path in self.PATHS:
            with self.subTest(path=path):
                if path in {"in-generation", "custom workflow"}:
                    with self.assertRaises(WorkflowCompileError) as path_main_error:
                        _compile(path, _payload("portrait, <lora:missing:1>", {"sam3_inpaint_prompt": "face"}))
                    self.assertEqual(str(path_main_error.exception), str(main_error.exception))
                with self.assertRaises(WorkflowCompileError) as sam3_error:
                    _compile(path, _payload("portrait", {"sam3_inpaint_prompt": "face, <lora:missing:1>"}))
                self.assertEqual(str(sam3_error.exception), str(main_error.exception))

    # ---- 검토 반영(fix round) ------------------------------------------------------------

    def test_a_tag_only_sam3_prompt_encodes_its_empty_text_not_the_main_prompt(self):
        """(검토 1) 태그만 적은 인페인트 프롬프트: Forge copy_prompt 는 비지 않은 글이라 메인 프롬프트로 채우지 않고
        (sam3ext/inpaint_core.py copy_prompt, scripts/!sam3.py), process_images 가 태그를 떼어 빈 글 + 그 LoRA 로 돈다
        (modules/processing.py parse_extra_network_prompts → setup_conds). 노드는 글이 비면 positive 입력(메인 프롬프트
        조건)을 쓰므로, 컴파일러가 노드와 같은 인코딩(CLIPTextEncode + 그 패스 clip)으로 뗀 글을 인코딩해 positive 로
        준다 — 분기(다른 LoRA)·메인과 같은 목록(재사용) 모두."""
        main = "portrait, <lora:ink:0.8>"
        cases = (("branch", "<lora:alice:0.7>", [(ALICE, 0.7, 0.7)]),
                 ("same list as main", "<lora:ink:0.8>", [(INK, 0.8, 0.8)]),
                 ("two tags, whitespace only between", " <lora:alice:0.7> <lora:ink:0.3>\n",
                  [(ALICE, 0.7, 0.7), (INK, 0.3, 0.3)]))
        for anima in (False, True):
            for path in self.PATHS:
                if path == "custom workflow" and anima:
                    continue
                for label, text, expected in cases:
                    with self.subTest(anima=anima, path=path, sam3=label):
                        graph = _compile(path, _payload(main, {"sam3_inpaint_prompt": text}, anima=anima), anima=anima)
                        [sam3] = _sam3_nodes(graph)
                        self.assertEqual(sam3["inpaint_prompt"].strip(), "")
                        encoder = graph[sam3["positive"][0]]
                        self.assertEqual(encoder["class_type"], "CLIPTextEncode")
                        self.assertEqual(encoder["inputs"], {"clip": sam3["clip"], "text": sam3["inpaint_prompt"]})
                        self.assertNotIn("portrait", _prompt_texts(graph, sam3["positive"]))
                        self.assert_sam3_loras(graph, expected)

    def test_workflow_controls_reach_the_conditioning_nodes_the_branch_copies(self):
        """(검토 2·4) 사용자 워크플로의 상세 설정은 컴파일 뒤(apply_controls) 원본 노드 id 에 걸린다. 분기가 다시 만든
        조건 노드(복제본)도 같은 워크플로 노드이므로 같은 값을 받아야 한다 — 분기 가이던스 스위트의 Modulation 기준 조건과,
        글이 빈 SAM3 네거티브가 쓰는 negative 입력이 복제본을 읽는다. 재사용(빈 인페인트 프롬프트)이면 복제본이 없다."""
        capabilities, workflow, binding = _controlled_workflow()
        for label, text, copies in (("reuse", "", 1), ("tag only", "<lora:alice:0.7>", 2),
                                    ("branch", "face, <lora:alice:0.7>", 2)):
            with self.subTest(label):
                payload = _payload("portrait, <lora:ink:0.8>", {"sam3_inpaint_prompt": text},
                                   scripts=_guidance_scripts(modulation=True))
                graph = ComfyWorkflowCompiler(capabilities).compile(
                    "txt2img", "checkpoint.safetensors", payload,
                    workflow=workflow, workflow_controls=binding,
                )
                ranges = [node["inputs"] for node in graph.values()
                          if node["class_type"] == "ConditioningSetTimestepRange"]
                strengths = [node["inputs"] for node in graph.values()
                             if node["class_type"] == "ConditioningSetAreaStrength"]
                self.assertEqual((len(ranges), len(strengths)), (copies, copies))
                self.assertEqual([(inputs["start"], inputs["end"]) for inputs in ranges], [(0.0, 0.5)] * copies)
                self.assertEqual([inputs["strength"] for inputs in strengths], [0.3] * copies)
                self.assertEqual(_sampler(graph)["positive"], ["9", 0])
                [sam3] = _sam3_nodes(graph)
                self.assertEqual(sam3["negative_prompt"], "")
                negative = graph[sam3["negative"][0]]
                self.assertEqual((negative["class_type"], negative["inputs"]["strength"]),
                                 ("ConditioningSetAreaStrength", 0.3))
                suites = [node["inputs"] for node in graph.values()
                          if node["class_type"] == "ForgeNeoAnimaGuidanceSuite"]
                self.assertEqual(len(suites), copies)
                for suite in suites:
                    self.assertEqual(graph[suite["positive"][0]]["inputs"]["end"], 0.5)
                    self.assertEqual(graph[suite["negative"][0]]["inputs"]["strength"], 0.3)

    def test_standalone_passes_never_resolve_main_loras_they_do_not_load(self):
        """(검토 3) 단독 SAM3/Refine(ADetailer 없음): 모델을 쓰는 곳이 SAM3 패스뿐이라 스택을 그 패스의 LoRA 목록으로
        만든다 — Forge 단독 p2 는 저장된 메인 프롬프트·네거티브의 LoRA 를 읽지 않는다(inpaint_core.py 단독 경로,
        processing.py forge_objects_original). 그래서 없는 메인 LoRA 도 오류가 아니고, 분기도 출력에 닿지 않는 메인
        스택(LoRA·NegPiP·인코더·스위트)도 없다. ADetailer 가 같이 있고 그 슬롯이 메인 목록을 쓰면(빈 ad_prompt =
        메인 프롬프트) 예전처럼 푼다 — 슬롯도 자기 목록을 쓰면 풀지 않는다(test_comfy_adetailer_lora_branch)."""
        kept = _LORA_CLASSES | {"ForgeNeoNegPip", "CLIPTextEncode", "ForgeNeoAnimaGuidanceSuite", "ForgeNeoSkimmedCFG"}
        for path in ("standalone sam3", "refine"):
            for text, expected in (("face, <lora:alice:0.6>", [(ALICE, 0.6, 0.6)]), ("face", []),
                                   ("<lora:alice:0.6>", [(ALICE, 0.6, 0.6)])):
                with self.subTest(path=path, sam3=text):
                    graph = _compile(path, _payload("portrait, <lora:missing:1>", {"sam3_inpaint_prompt": text},
                                                    negative="bad, <lora:ink:0.4>",
                                                    scripts=_guidance_scripts(modulation=True)))
                    self.assert_sam3_loras(graph, expected)
                    self.assertNotIn(_BRANCH_TITLE, json.dumps(graph))
                    live = _live_node_ids(graph)
                    self.assertEqual([(node_id, node["class_type"]) for node_id, node in graph.items()
                                      if node["class_type"] in kept and node_id not in live], [])
        with self.subTest("backend sam3, saved main prompt names a missing LoRA"):
            backend = regressions.TestStandaloneComfyContext._backend(self, {})
            backend._last_generation_context["payload"]["prompt"] = "portrait, <lora:missing:1>"
            backend.sam3(_png(), {"sam3_prompt": "face", "sam3_inpaint_prompt": "face, <lora:alice:0.6>"})
            self.assert_sam3_loras(backend._queue_and_wait.call_args.args[0], [(ALICE, 0.6, 0.6)])
        with self.subTest("ADetailer with an empty ad_prompt uses the main list, so its LoRAs still resolve"):
            payload = _payload("portrait, <lora:missing:1>", {"sam3_inpaint_prompt": "face"})
            payload["alwayson_scripts"]["ADetailer"] = {"args": [True, False, {
                "ad_tab_enable": True, "ad_model": "face_yolov8n.pt"}]}
            with self.assertRaisesRegex(WorkflowCompileError, "lora_name.*missing"):
                _compile("standalone sam3", payload)


if __name__ == "__main__":
    unittest.main()
