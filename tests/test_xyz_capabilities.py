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

    def test_text_encoder_axis_on_forge_and_comfy_uses_the_app_te_choices(self):
        """sam-extra 0.34.0 의 Forge XYZ '[Modules] Text Encoder' 와 같은 뜻 — 앱 XYZ 는 칸마다 따로 생성하므로 앱 축이 맡는다."""
        from core.xyz_capabilities import (FORGE_TEXT_ENCODER_AXIS, TEXT_ENCODER_CURRENT, TEXT_ENCODER_NOTE,
                                           forge_capabilities)
        api = {"paths": {"/sdapi/v1/txt2img": {"post": {"requestBody": {"content": {"application/json": {
            "schema": {"properties": {"steps": {"type": "integer"}}}}}}}}}}
        scripts = [{"name": "x/y/z plot", "is_img2img": False, "args": [{"label": "X type",
            "choices": ["Nothing", "Steps", "VAE", FORGE_TEXT_ENCODER_AXIS, "[SAM3] Face Crop"]}]}]
        tes = ["qwen_3_06b_base.safetensors", "sub/qwen_alt.safetensors", "qwen_3_06b_base.safetensors", " "]
        result = forge_capabilities(api, scripts=scripts, text_encoders=tes)
        axis = next(axis for axis in result["axes"] if axis["id"] == "text_encoder")
        self.assertEqual(axis["label"], "Text Encoder")
        self.assertEqual(axis["choices"], [TEXT_ENCODER_CURRENT, "qwen_3_06b_base.safetensors", "sub/qwen_alt.safetensors"])
        # 앱 축이 맡는 확장 축은 '실행하지 않는 축' 목록에서 빠진다 — VAE 축은 앱에 없다
        self.assertEqual(result["unsupported"], ["VAE", "[SAM3] Face Crop"])
        self.assertIn(TEXT_ENCODER_NOTE, result["notes"])
        # TE 선택지가 없으면 축도, 빠지는 확장 축도 없다
        bare = forge_capabilities(api, scripts=scripts)
        self.assertNotIn("text_encoder", {axis["id"] for axis in bare["axes"]})
        self.assertIn(FORGE_TEXT_ENCODER_AXIS, bare["unsupported"])
        # ComfyUI: 앱이 그래프를 만들 때만(사용자 워크플로는 자기 CLIP 로더를 쓴다)
        schema = {"KSampler": {"input": {"required": {"steps": ["INT", {"min": 1, "max": 50}]}}}}
        self.assertIn("text_encoder", {axis["id"] for axis in comfy_capabilities(schema, text_encoders=tes)["axes"]})
        workflow = {"1": {"class_type": "KSampler", "inputs": {}}}
        self.assertNotIn("text_encoder", {axis["id"] for axis in
                                         comfy_capabilities(schema, workflow=workflow, text_encoders=tes)["axes"]})
        self.assertEqual(comfy_capabilities(schema, family="krea2", text_encoders=tes)["axes"], [])

    def test_text_encoder_axis_replaces_only_the_te_entry_in_every_cell(self):
        """앱 VAE·TE 칩을 골랐으면 칸이 그 목록의 TE 칸을 바꾼 목록을 싣는다 — Forge·ComfyUI 같다(GPU 로 확인한 길 그대로).
        Forge 의 목록(forge_modules)은 보지 않는다 — TE 가 둘인 목록을 줘도 거절하지 않는다."""
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TEXT_ENCODER_CURRENT, build_jobs, text_encoder_axis
        axes = [text_encoder_axis(["te_a.safetensors", "te_b.safetensors"], source="t"),
                {"id": "steps", "label": "Steps", "type": "integer", "min": 1, "max": 50, "step": 1}]
        base = {"prompt": "p", "forge_additional_modules": ["vae.safetensors", "te_a.safetensors"],
                "enable_hr": True, "hr_additional_modules": ["Use same choices"]}
        for backend in ("webui", "comfyui"):
            with self.subTest(backend=backend):
                jobs = build_jobs(base, "m", [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "te_b.safetensors"]},
                                              {"id": "steps", "values": [10, 20]}], {"axes": axes, "backend": backend},
                                  text_encoders=["te_a.safetensors"],
                                  forge_modules=["te_a.safetensors", "te_b.safetensors"])
                self.assertEqual([job["forge_additional_modules"] for job in jobs], [
                    ["vae.safetensors", "te_a.safetensors"], ["vae.safetensors", "te_a.safetensors"],
                    ["vae.safetensors", "te_b.safetensors"], ["vae.safetensors", "te_b.safetensors"]])
                self.assertEqual(jobs[2]["_xyz_info"]["axes"], {"Text Encoder": "te_b.safetensors", "Steps": 10})
                self.assertFalse(any(FORGE_TE_CELL_KEY in job for job in jobs))
        self.assertEqual(base["forge_additional_modules"], ["vae.safetensors", "te_a.safetensors"])

    def test_comfy_cells_without_app_modules_name_the_te_or_the_built_in(self):
        """칩이 빈 ComfyUI 는 보통 생성도 체크포인트 내장 VAE·TE 다 — 칸은 [그 TE](CLIP 로더에 들어간다)를 더하고, 'Use current'
        칸은 빈 목록(내장)을 명시한다(예전과 같다). 1패스에 모듈이 생긴 칸의 Hires 는 'Use same choices'. 손으로 만든
        capability(backend 없음)도 같은 길이다."""
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TEXT_ENCODER_CURRENT, build_jobs, text_encoder_axis
        bare = {"prompt": "p", "enable_hr": True, "hr_additional_modules": []}
        for backend in ("comfyui", None):
            with self.subTest(backend=backend):
                capability = {"axes": [text_encoder_axis(["te_a.safetensors", "te_b.safetensors"], source="t")]}
                if backend:
                    capability["backend"] = backend
                jobs = build_jobs(bare, "m", [{"id": "text_encoder", "values": ["te_a.safetensors", TEXT_ENCODER_CURRENT]}],
                                  capability, text_encoders=[], forge_modules=["te_a.safetensors", "te_b.safetensors"])
                self.assertEqual([job["forge_additional_modules"] for job in jobs], [["te_a.safetensors"], []])
                self.assertEqual([job["hr_additional_modules"] for job in jobs], [["Use same choices"], []])
                self.assertFalse(any(FORGE_TE_CELL_KEY in job for job in jobs))

    def test_forge_cells_without_app_modules_carry_only_the_te_choice(self):
        """칩이 빈 Forge 의 보통 생성은 모듈 키 없이 Forge 드롭다운의 목록(예: qwen_image_vae + qwen_3_06b_base)을 쓴다 — 칸도
        목록을 싣지 않고 TE 선택만 싣는다(WebUI 백엔드가 Forge 목록에서 TE 만 바꾼다). 예전엔 [그 TE]·[] 를 실어 Forge 가
        VAE 를 잃었다("You do not have VAE state dict!")."""
        from core.xyz_capabilities import (FORGE_TE_CELL_KEY, TEXT_ENCODER_CURRENT, TRANSIENT_MODULES_KEY, build_jobs,
                                           text_encoder_axis)
        tes = ["qwen_3_06b_base.safetensors", "sub/qwen_alt.safetensors"]
        capability = {"axes": [text_encoder_axis(tes, source="t"),
                               {"id": "steps", "label": "Steps", "type": "integer", "min": 1, "max": 50, "step": 1}],
                      "backend": "webui"}
        bare = {"prompt": "p", "enable_hr": True, "hr_additional_modules": []}
        forge = ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"]
        jobs = build_jobs(bare, "anima", [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "sub/qwen_alt.safetensors"]}],
                          capability, text_encoders=[], forge_modules=forge)
        self.assertTrue(all("forge_additional_modules" not in job for job in jobs), "Forge keeps its own list")
        self.assertEqual([job[FORGE_TE_CELL_KEY] for job in jobs], [
            {"value": None, "encoders": tes}, {"value": "sub/qwen_alt.safetensors", "encoders": tes}])
        self.assertEqual([job[TRANSIENT_MODULES_KEY] for job in jobs], [True, True])
        # Hires: TE 를 고른 칸은 1패스 목록을 쓰고('Use same choices'), 'Use current' 칸은 보통 생성 그대로('Built-in')
        self.assertEqual([job["hr_additional_modules"] for job in jobs], [[], ["Use same choices"]])
        self.assertEqual(jobs[1]["_xyz_info"]["axes"], {"Text Encoder": "sub/qwen_alt.safetensors"})
        self.assertNotIn(FORGE_TE_CELL_KEY, bare)
        # 축 목록을 받을 때 Forge 목록을 못 읽었으면(None) 첫 칸 전 확인 없이 만든다 — 백엔드가 보낼 때 다시 본다
        jobs = build_jobs(bare, "anima", [{"id": "text_encoder", "values": ["sub/qwen_alt.safetensors"]}], capability)
        self.assertEqual(jobs[0][FORGE_TE_CELL_KEY]["value"], "sub/qwen_alt.safetensors")
        # Text Encoder 축이 없으면 칸에 아무것도 싣지 않는다(보통 생성과 같은 요청)
        jobs = build_jobs(bare, "anima", [{"id": "steps", "values": [10]}], capability, forge_modules=forge)
        self.assertFalse({FORGE_TE_CELL_KEY, TRANSIENT_MODULES_KEY, "forge_additional_modules"} & set(jobs[0]))

    def test_forge_te_cell_modules_replaces_only_the_te_in_forges_list(self):
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT, forge_te_cell_modules
        tes = ["qwen_3_06b_base.safetensors", "sub/qwen_alt.safetensors"]
        forge = ["C:/forge/models/VAE/qwen_image_vae.safetensors",
                 "C:\\forge\\models\\text_encoder\\qwen_3_06b_base.safetensors"]
        self.assertEqual(forge_te_cell_modules(forge, tes, "sub/qwen_alt.safetensors"),
                         ["qwen_image_vae.safetensors", "qwen_alt.safetensors"], "the VAE stays")
        for current in (None, TEXT_ENCODER_CURRENT):
            self.assertEqual(forge_te_cell_modules(forge, tes, current),
                             ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"], "'Use current' = Forge's list")
        # Forge 목록에 TE 가 없으면 더한다(체크포인트 내장 TE 를 대신 — VAE 그대로)
        self.assertEqual(forge_te_cell_modules(forge[:1], tes, "qwen_3_06b_base.safetensors"),
                         ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"])
        self.assertEqual(forge_te_cell_modules([], tes, "sub/qwen_alt.safetensors"), ["qwen_alt.safetensors"])
        # 파일 이름은 대소문자 없이 맞추고, TE 칩 선택지에 없는 파일(VAE·어댑터 …)은 TE 로 보지 않는다
        self.assertEqual(forge_te_cell_modules(["Qwen_3_06B_Base.safetensors", "adapter.safetensors"], tes,
                                               "sub/qwen_alt.safetensors"), ["qwen_alt.safetensors", "adapter.safetensors"])
        two = ["ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors"]
        with self.assertRaisesRegex(ValueError, "Text Encoder 축: Forge 에서 고른 .*TE 가 2개"):
            forge_te_cell_modules(two, ["clip_l.safetensors", "t5xxl.safetensors"], "clip_l.safetensors")
        self.assertEqual(forge_te_cell_modules(two, ["clip_l.safetensors", "t5xxl.safetensors"], TEXT_ENCODER_CURRENT), two)

    def test_forge_cells_refuse_two_encoders_in_forges_list_before_the_first_cell(self):
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TEXT_ENCODER_CURRENT, build_jobs, text_encoder_axis
        capability = {"axes": [text_encoder_axis(["clip_l.safetensors", "t5xxl.safetensors", "clip_g.safetensors"],
                                                 source="t")], "backend": "webui"}
        forge = ["ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors"]
        with self.assertRaisesRegex(ValueError, "TE 가 2개"):
            build_jobs({"prompt": "p"}, "flux", [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT,
                                                                                   "clip_g.safetensors"]}],
                       capability, forge_modules=forge)
        jobs = build_jobs({"prompt": "p"}, "flux", [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT]}],
                          capability, forge_modules=forge)
        self.assertEqual([job[FORGE_TE_CELL_KEY]["value"] for job in jobs], [None], "'Use current' only is fine")

    def test_text_encoder_axis_marks_its_cells_for_the_forge_module_restore(self):
        """Forge 옵션으로 바꾼 TE 를 끝나고 되돌리도록(webui_backend._switch_model_if_needed) TE 축 칸만 표시한다."""
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT, TRANSIENT_MODULES_KEY, build_jobs, text_encoder_axis
        capability = {"axes": [text_encoder_axis(["te_a.safetensors"], source="t"),
                               {"id": "steps", "label": "Steps", "type": "integer", "min": 1, "max": 50, "step": 1}]}
        base = {"prompt": "p", "forge_additional_modules": ["vae.safetensors"]}
        jobs = build_jobs(base, "m", [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "te_a.safetensors"]}],
                          capability, text_encoders=[])
        self.assertEqual([job.get(TRANSIENT_MODULES_KEY) for job in jobs], [True, True])
        jobs = build_jobs(base, "m", [{"id": "steps", "values": [10, 20]}], capability, text_encoders=[])
        self.assertEqual([TRANSIENT_MODULES_KEY in job for job in jobs], [False, False], "the chip's modules stay")
        self.assertNotIn(TRANSIENT_MODULES_KEY, base)

    def test_text_encoder_axis_refuses_two_encoders_but_allows_only_current(self):
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT, build_jobs, text_encoder_axis
        capability = {"axes": [text_encoder_axis(["clip_l.safetensors", "t5xxl.safetensors"], source="t")]}
        base = {"prompt": "p", "forge_additional_modules": ["clip_l.safetensors", "t5xxl.safetensors"]}
        with self.assertRaisesRegex(ValueError, "Text Encoder 축"):
            build_jobs(base, "m", [{"id": "text_encoder", "values": ["t5xxl.safetensors"]}], capability,
                       text_encoders=["clip_l.safetensors", "t5xxl.safetensors"])
        jobs = build_jobs(base, "m", [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT]}], capability,
                          text_encoders=["clip_l.safetensors", "t5xxl.safetensors"])
        self.assertEqual(jobs[0]["forge_additional_modules"], ["clip_l.safetensors", "t5xxl.safetensors"])
        with self.assertRaises(ValueError):
            build_jobs(base, "m", [{"id": "text_encoder", "values": ["unknown.safetensors"]}], capability)

    def test_comfy_text_encoder_cells_compile_into_the_clip_loader(self):
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        from core.xyz_capabilities import build_jobs
        schema = {name: {"input": {"required": {}}} for name in (
            "KSampler", "ForgeNeoKSamplerCNS", "ForgeNeoLatentInput", "CheckpointLoaderSimple", "CLIPLoader",
            "CLIPTextEncode", "EmptyLatentImage", "VAEDecode", "SaveImage", "PreviewImage")}
        schema["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"] = [["model.safetensors"]]
        schema["CLIPLoader"]["input"]["required"].update({"clip_name": [["te_a.safetensors", "te_b.safetensors"]],
                                                          "type": [["stable_diffusion", "sdxl"]]})
        schema["KSampler"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 50}]
        capability = comfy_capabilities(schema, text_encoders=["te_a.safetensors", "te_b.safetensors"])
        jobs = build_jobs({"prompt": "p", "width": 512, "height": 512, "seed": 1}, "model.safetensors",
                          [{"id": "text_encoder", "values": ["te_b.safetensors"]}], capability, text_encoders=[])
        graph = ComfyWorkflowCompiler(schema).compile("txt2img", "model.safetensors", jobs[0])
        loaders = [node["inputs"]["clip_name"] for node in graph.values() if node["class_type"] == "CLIPLoader"]
        self.assertEqual(loaders, ["te_b.safetensors"])

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
