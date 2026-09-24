import unittest

from core.xyz_capabilities import comfy_capabilities


class XYZCapabilityTests(unittest.TestCase):
    def test_semantic_custom_graph_advertises_sr_and_compiles_both_replacements(self):
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        from core.xyz_capabilities import build_jobs
        from tests.test_comfy_anima38_compiler import (
            ADAPTER, V1_MODEL, V2_MODEL, _anima_capabilities, _modules, _script,
        )
        for model, enabled, node_type in (
            (V1_MODEL, True, "ForgeNeoAnimaQwen35Prompt"),
            (V2_MODEL, False, "ForgeNeoAnima38V2Prompt"),
        ):
            with self.subTest(node_type=node_type):
                schema = _anima_capabilities()
                compiler = ComfyWorkflowCompiler(schema)
                base = {
                    "prompt": "original", "negative_prompt": "old negative",
                    "forge_additional_modules": _modules(adapter=ADAPTER if enabled else ""),
                    "alwayson_scripts": _script(enabled, ADAPTER, 1, True, 1, False),
                }
                original = compiler.compile("txt2img", model, base)
                capability = comfy_capabilities(schema, workflow=original)
                axes = {axis["id"] for axis in capability["axes"]}
                self.assertIn("prompt_sr", axes)
                self.assertIn("negative_sr", axes)
                jobs = build_jobs(base, model, [
                    {"id": "prompt_sr", "search": "original", "values": ["updated"]},
                    {"id": "negative_sr", "search": "old negative", "values": ["new negative"]},
                ], capability)
                updated = compiler.compile("txt2img", model, jobs[0], workflow=original)
                self.assertEqual(
                    {node["inputs"]["prompt"] for node in updated.values() if node["class_type"] == node_type},
                    {"updated", "new negative"},
                )

    def test_comfy_axis_values_compile_into_the_real_sampler_graph(self):
        from core.xyz_capabilities import build_jobs
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        schema = {name: {"input": {"required": {}}} for name in (
            "KSampler", "ForgeNeoKSamplerCNS", "ForgeNeoLatentInput", "CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "VAEDecode", "SaveImage", "PreviewImage")}
        schema["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"] = [["model.safetensors"]]
        schema["KSampler"]["input"]["required"].update({
            "seed": ["INT", {"min": 0, "max": 2**32 - 1}], "steps": ["INT", {"min": 1, "max": 100}],
            "cfg": ["FLOAT", {"min": 0, "max": 30}], "sampler_name": [["euler", "dpmpp_2m"]], "scheduler": [["normal", "karras"]]})
        capability = comfy_capabilities(schema)
        jobs = build_jobs({"prompt": "synthetic", "width": 512, "height": 512, "seed": 123}, "model.safetensors",
            [{"id": "steps", "values": [31]}, {"id": "cfg_scale", "values": [4.5]}, {"id": "sampler_name", "values": ["dpmpp_2m"]}], capability)
        graph = ComfyWorkflowCompiler(schema).compile("txt2img", "model.safetensors", jobs[0])
        sampler = next(node["inputs"] for node in graph.values() if node["class_type"] == "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["steps"], 31)
        self.assertEqual(sampler["cfg"], 4.5)
        self.assertEqual(sampler["sampler_name"], "dpmpp_2m")
        self.assertEqual(sampler["seed"], 123)

    def test_unmappable_custom_sampler_does_not_advertise_other_ineffective_axes(self):
        schema = {"CLIPTextEncode": {"input": {"required": {"text": ["STRING"]}}},
                  "SamplerCustomAdvanced": {"input": {"required": {}}}}
        workflow = {"1": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
                    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}}}
        self.assertEqual(comfy_capabilities(schema, workflow=workflow)["axes"], [])

    def test_axis_values_reach_jobs_without_losing_base_extensions_or_sr_search(self):
        from core.xyz_capabilities import build_jobs
        capability = {"axes": [
            {"id": "prompt_sr", "label": "Prompt S/R", "type": "replace"},
            {"id": "steps", "label": "Steps", "type": "integer", "min": 1, "max": 50, "step": 1},
            {"id": "sampler_name", "label": "Sampler", "type": "choice", "choices": ["euler"]},
        ]}
        base = {"prompt": "red hair", "negative_prompt": "blur", "alwayson_scripts": {"SAM3": {"args": [True]}}, "enable_hr": True}
        jobs = build_jobs(base, "model.safetensors", [{"id": "prompt_sr", "search": "red", "values": ["red", "blue"]},
            {"id": "steps", "values": [20, 30]}], capability)
        self.assertEqual([j["prompt"] for j in jobs], ["red hair", "red hair", "blue hair", "blue hair"])
        self.assertEqual([j["steps"] for j in jobs], [20, 30, 20, 30])
        self.assertEqual(jobs[-1]["alwayson_scripts"]["SAM3"]["args"], [True])
        jobs[0]["alwayson_scripts"]["SAM3"]["args"][0] = False
        self.assertTrue(base["alwayson_scripts"]["SAM3"]["args"][0])
        with self.assertRaises(ValueError):
            build_jobs(base, "m", [{"id": "sampler_name", "values": ["invented"]}], capability)
        with self.assertRaises(ValueError):
            build_jobs(base, "m", [{"id": "steps", "values": [1.5]}], capability)

    def test_forge_uses_api_fields_and_lists_unimplemented_extension_axes_separately(self):
        from core.xyz_capabilities import forge_capabilities
        api = {"paths": {"/sdapi/v1/txt2img": {"post": {"requestBody": {"content": {"application/json": {
            "schema": {"$ref": "#/components/schemas/Txt2Img"}}}}}}},
            "components": {"schemas": {"Txt2Img": {"properties": {
                "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                "prompt": {"type": "string"}, "sampler_name": {"type": "string"},
                "scheduler": {"type": "string"}, "denoising_strength": {"type": "number"}}}}}}
        scripts = [{"name": "x/y/z plot", "is_img2img": False, "args": [{"label": "X type",
            "choices": ["Nothing", "Steps", "Sampler", "Schedule type", "PAG custom axis"]}]}]
        result = forge_capabilities(api, scripts=scripts, samplers=[{"name": "Euler"}], schedulers=[{"name": "normal", "label": "Normal"}])
        axes = {axis["id"]: axis for axis in result["axes"]}
        self.assertEqual(set(axes), {"steps", "prompt_sr", "sampler_name", "scheduler"})
        self.assertEqual(axes["scheduler"]["choices"], ["Normal"])
        self.assertEqual(result["unsupported"], ["PAG custom axis"])

    def test_comfy_lists_only_live_schema_fields_and_exact_choices(self):
        schema = {"KSampler": {"input": {"required": {
            "seed": ["INT", {"min": 0, "max": 9999}], "steps": ["INT", {"min": 1, "max": 50}],
            "cfg": ["FLOAT", {"min": 0, "max": 20}], "sampler_name": [["euler", "er_sde"]],
            "scheduler": [["normal", "beta57"]],
        }}}}
        result = comfy_capabilities(schema)
        axes = {axis["id"]: axis for axis in result["axes"]}
        self.assertEqual(set(axes), {"seed", "steps", "cfg_scale", "sampler_name", "scheduler"})
        self.assertEqual(axes["sampler_name"]["choices"], ["euler", "er_sde"])
        self.assertEqual(axes["steps"]["max"], 50)
        self.assertNotIn("denoising_strength", axes)

    def test_krea2_capabilities_skip_every_network_call(self):
        from unittest import mock
        from core.xyz_capabilities import KREA2_NOTE, MAX_JOBS, fetch_capabilities

        class _Backend:
            api_url = "http://127.0.0.1:8188/"

            def get_object_info(self):
                raise AssertionError("krea2 must not fetch /object_info")

            def _load_configured_workflow(self, _mode):
                raise AssertionError("krea2 must not load the txt2img workflow")

        with mock.patch("requests.get", side_effect=AssertionError("krea2 must not call Forge")):
            for kind in ("comfyui", "webui"):
                with self.subTest(kind=kind):
                    result = fetch_capabilities(_Backend(), kind, family="krea2")
                    self.assertEqual(result["axes"], [])
                    self.assertEqual(result["notes"], [KREA2_NOTE])
                    self.assertEqual(result["backend"], kind)
                    self.assertEqual(result["maxJobs"], MAX_JOBS)
                    self.assertTrue(result["capabilityId"])

    def test_unknown_backend_kind_is_still_rejected_before_krea2_shortcut(self):
        from core.xyz_capabilities import fetch_capabilities
        with self.assertRaises(ValueError):
            fetch_capabilities(object(), "invokeai", family="krea2")


if __name__ == "__main__":
    unittest.main()
