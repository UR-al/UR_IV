"""sam-extra v0.30.0 디테일 가이던스·Optimal Scale 을 ComfyUI 그래프로 (팩 1.6.0).

- 디테일 스위치(TSR·MG·HiGS·HiFlow)는 혼자서도 가이던스 스위트를 붙인다(Forge 쪽 활성 키와 같다).
- HiFlow 는 txt2img + Hires.fix 에서만 켠다(확장 _hiflow_attach) — 메인 생성이든 사용자 워크플로든.
  나머지는 settings_json 에 False 로 보낸다.
- S² 를 켰을 때만 settings_json 에 생성 시드(guid_s2_seed)를 넣는다 — 샘플러가 받는 바로 그 시드.
- Optimal Scale 노드는 스위트 앞(모델 쪽)에 붙는다 — post-CFG 순서가 확장(Safe PAG 보다 먼저)과 같다.
- 1.6.0 표지(ForgeNeoAnimaOptimalScale)가 없는 옛 팩은 디테일 설정을 큐 전에 거부한다(1.5.0 스위트는 새 키를
  모르고 조용히 건너뛴다).

ComfyUI 에는 접속하지 않는다. Optimal Scale 노드의 입력 계약은 번들 노드의 INPUT_TYPES 그대로다.
"""
from __future__ import annotations

import json
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import guidance_optimal_scale
from core import anima_guidance
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests.test_comfy_workflow_compiler import _capabilities, _classes, _custom_workflow, _node

MARKER = "ForgeNeoAnimaOptimalScale"
SUITE = "ForgeNeoAnimaGuidanceSuite"
S2 = {"guid_enabled": True, "guid_slg_on": True, "guid_slg_mode": anima_guidance.SLG_MODE_S2}


def caps(*, stale: bool = False) -> dict:
    capabilities = _capabilities()
    if not stale:
        capabilities[MARKER] = {
            "input": guidance_optimal_scale.ForgeNeoAnimaOptimalScale.INPUT_TYPES(), "output": ["MODEL"],
        }
    return capabilities


def body(guidance: dict, **extra) -> dict:
    settings = anima_guidance.default_settings()
    settings.update(guidance)
    return {"prompt": "portrait", "negative_prompt": "bad", "seed": 1234,
            "alwayson_scripts": anima_guidance.build_alwayson(settings), **extra}


def compile_graph(guidance: dict, *, mode="txt2img", stale=False, workflow=None, **extra) -> dict:
    return ComfyWorkflowCompiler(caps(stale=stale)).compile(
        mode, "checkpoint.safetensors", body(guidance, **extra), workflow=workflow,
        uploaded_image="source.png" if mode != "txt2img" else "",
    )


def suites(graph: dict) -> list[dict]:
    return [json.loads(node["inputs"]["settings_json"])
            for node in graph.values() if isinstance(node, dict) and node.get("class_type") == SUITE]


class TestDetailSuiteCompilation(unittest.TestCase):
    def test_a_detail_stage_alone_attaches_the_suite(self):
        for key in ("guid_tsr_enabled", "guid_mg_enabled", "guid_higs_enabled"):
            with self.subTest(key=key):
                [settings] = suites(compile_graph({key: True}))
                self.assertIs(settings[key], True)
                self.assertFalse(settings["guid_enabled"])
        [settings] = suites(compile_graph({"guid_hiflow_enabled": True}, enable_hr=True, hr_scale=1.5))
        self.assertIs(settings["guid_hiflow_enabled"], True)
        self.assertEqual(suites(compile_graph({})), [])

    def test_hiflow_only_for_a_txt2img_hires_generation(self):
        hiflow = {"guid_hiflow_enabled": True, "guid_tsr_enabled": True}
        for label, kwargs, expected in (
            ("txt2img + Hires.fix", {"enable_hr": True, "hr_scale": 1.5}, True),
            ("txt2img", {}, False),
            ("img2img", {"mode": "img2img", "enable_hr": True, "hr_scale": 1.5}, False),
            ("custom workflow + Hires.fix", {"workflow": _custom_workflow(), "enable_hr": True, "hr_scale": 1.5}, True),
            ("custom workflow", {"workflow": _custom_workflow()}, False),
        ):
            with self.subTest(case=label):
                [settings] = suites(compile_graph(hiflow, **kwargs))
                self.assertIs(settings["guid_hiflow_enabled"], expected)
                self.assertIs(settings["guid_tsr_enabled"], True)

    def test_the_s2_seed_rides_only_with_s2_and_is_the_samplers(self):
        [settings] = suites(compile_graph(S2))
        self.assertEqual(settings["guid_s2_seed"], 1234)
        [settings] = suites(compile_graph({**S2, "guid_slg_mode": anima_guidance.SLG_MODE_FIXED}))
        self.assertNotIn("guid_s2_seed", settings)
        [settings] = suites(compile_graph({**S2, "guid_slg_on": False}))   # S² 는 SLG 의 방식 — SLG 를 끄면 없다
        self.assertNotIn("guid_s2_seed", settings)
        graph = compile_graph(S2, seed=-1)   # 무작위 시드는 컴파일 때 한 번 정해진다
        [settings] = suites(graph)
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(settings["guid_s2_seed"], sampler["inputs"]["seed"])
        self.assertGreaterEqual(settings["guid_s2_seed"], 0)

    def test_optimal_scale_sits_before_the_suite(self):
        graph = compile_graph({"guid_tsr_enabled": True, "ocfg_enabled": True, "ocfg_blend": 0.5,
                               "ocfg_start": 0.1, "ocfg_end": 0.8})
        optimal_id, optimal = _node(graph, MARKER)
        _suite_id, suite = _node(graph, SUITE)
        self.assertEqual(suite["inputs"]["model"], [optimal_id, 0])
        self.assertEqual(
            {key: optimal["inputs"][key] for key in ("enabled", "blend", "start_percent", "end_percent")},
            {"enabled": True, "blend": 0.5, "start_percent": 0.1, "end_percent": 0.8},
        )
        # 혼자 켜도 샘플러의 모델 체인에 들어간다
        graph = compile_graph({"ocfg_enabled": True})
        optimal_id, _optimal = _node(graph, MARKER)
        self.assertNotIn(SUITE, _classes(graph))
        links = [value for node in graph.values() if isinstance(node, dict)
                 for value in node.get("inputs", {}).values() if value == [optimal_id, 0]]
        self.assertTrue(links)
        self.assertNotIn(MARKER, _classes(compile_graph({"ocfg_enabled": False, "guid_tsr_enabled": True})))


class TestStalePackDetailSuite(unittest.TestCase):
    """A 1.5.0 suite takes the same inputs and ignores the new keys — the compiler refuses before queueing."""

    def test_the_marker_is_a_bundled_node(self):
        from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS
        from core import comfy_workflow_compiler

        self.assertEqual(comfy_workflow_compiler._DETAIL_SUITE_MARKER, MARKER)
        self.assertIn(MARKER, NODE_CLASS_MAPPINGS)

    def test_a_stale_pack_refuses_each_detail_feature(self):
        for label, guidance, extra in (
            ("S²", S2, {}),
            ("Adaptive SMC", {"guid_smc_enabled": True, "guid_smc_mode": anima_guidance.SMC_MODE_ADAPTIVE}, {}),
            ("TSR", {"guid_tsr_enabled": True}, {}),
            ("Momentum", {"guid_mg_enabled": True}, {}),
            ("HiGS", {"guid_higs_enabled": True}, {}),
            ("HiFlow", {"guid_hiflow_enabled": True}, {"enable_hr": True, "hr_scale": 1.5}),
        ):
            with self.subTest(feature=label), self.assertRaisesRegex(
                WorkflowCompileError, "디테일 가이던스.*1\\.6\\.0.*ForgeNeoAnimaOptimalScale",
            ):
                compile_graph(guidance, stale=True, **extra)

    def test_a_stale_pack_still_compiles_what_it_can_run(self):
        for label, guidance in (
            ("plain PAG", {"guid_enabled": True}),
            ("Fixed SLG", {**S2, "guid_slg_mode": anima_guidance.SLG_MODE_FIXED}),
            ("Unit-L2 SMC", {"guid_smc_enabled": True}),
            ("HiFlow without Hires.fix", {"guid_hiflow_enabled": True}),   # 꺼서 보내므로 옛 팩도 같은 그림
            ("TSR k 1", {"guid_tsr_enabled": True, "guid_tsr_k": 1.0}),
        ):
            with self.subTest(case=label):
                compile_graph(guidance, stale=True)

    def test_a_hand_built_suite_with_a_detail_key_is_refused_on_a_stale_pack(self):
        workflow = _custom_workflow()
        workflow["8"] = {"class_type": SUITE, "inputs": {
            "model": ["1", 0], "clip": ["1", 1], "positive": ["2", 0], "negative": ["3", 0],
            "enabled": True, "settings_json": json.dumps({"guid_higs_enabled": True}),
        }}
        workflow["5"]["inputs"]["model"] = ["8", 0]
        with self.assertRaisesRegex(WorkflowCompileError, "디테일 가이던스"):
            ComfyWorkflowCompiler(caps(stale=True)).compile("txt2img", "checkpoint.safetensors", {},
                                                            workflow=workflow)
        workflow["8"]["inputs"]["enabled"] = False
        ComfyWorkflowCompiler(caps(stale=True)).compile("txt2img", "checkpoint.safetensors", {}, workflow=workflow)
        ComfyWorkflowCompiler(caps()).compile("txt2img", "checkpoint.safetensors", {}, workflow=workflow)


if __name__ == "__main__":
    unittest.main()
