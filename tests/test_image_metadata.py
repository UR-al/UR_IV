"""Image metadata contracts exercised through tiny PNG files, never model execution."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from core.image_metadata import extract_from_file


def graph_fixture():
    return {
        "99": {"class_type": "CLIPTextEncode", "inputs": {"text": "unused decoy"}},
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "shared/model.safetensors"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad anatomy", "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "a blue bird", "clip": ["1", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 640, "height": 832, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["8", 0], "latent_image": ["5", 0], "seed": 42, "steps": 28, "cfg": 6.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0]}},
    }


class ImageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.png"

    def png(self, **metadata):
        info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            info.add_text(key, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        Image.new("RGB", (8, 8), "blue").save(self.path, pnginfo=info)
        return self.path

    def test_comfy_api_follows_sampler_links_not_node_order_and_preserves_source(self):
        graph = graph_fixture()
        workflow = {"nodes": [], "links": [], "extra": {"label": "keep me"}}
        self.png(prompt=graph, workflow=workflow)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        meta = extract_from_file(self.path)
        self.assertEqual(meta.prompt, "a blue bird")
        self.assertEqual(meta.negative_prompt, "bad anatomy")
        self.assertEqual(meta.parameters["Seed"], 42)
        self.assertEqual(meta.parameters["Size"], "640x832")
        self.assertEqual(meta.parameters["Model"], "shared/model.safetensors")
        self.assertEqual(meta.prompt_graph, graph)
        self.assertEqual(meta.workflow, workflow)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_multiple_sampler_prompts_are_separate_not_arbitrarily_selected(self):
        graph = graph_fixture()
        graph["17"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "a red cat"}}
        graph["13"] = {"class_type": "KSampler", "inputs": {**graph["3"]["inputs"], "positive": ["17", 0], "seed": 43}}
        graph["19"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0]}}
        graph["20"] = {"class_type": "SaveImage", "inputs": {"images": ["19", 0]}}
        self.png(prompt=graph)
        meta = extract_from_file(self.path)
        self.assertEqual(meta.prompt, "")
        self.assertEqual(meta.negative_prompt, "bad anatomy")
        self.assertNotIn("Seed", meta.parameters)
        self.assertEqual({item["prompt"] for item in meta.prompt_candidates}, {"a blue bird", "a red cat"})
        self.assertTrue(meta.metadata_warnings)

    def test_sdxl_distinct_encoders_and_conditioning_zero_are_not_merged(self):
        graph = graph_fixture()
        graph["7"] = {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "global scene", "text_l": "local detail"}}
        graph["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
        graph["3"]["inputs"]["negative"] = ["6", 0]
        self.png(prompt=graph)
        meta = extract_from_file(self.path)
        self.assertEqual(meta.prompt, "")
        self.assertEqual(meta.negative_prompt, "")
        candidate = meta.prompt_candidates[0]
        self.assertEqual([part["text"] for part in candidate["positive_parts"]], ["global scene", "local detail"])
        self.assertFalse(candidate["positive_known"])
        self.assertTrue(candidate["negative_known"])
        self.assertTrue(meta.metadata_warnings)

    def test_sampler_custom_advanced_uses_guider_roles_and_noise_scheduler(self):
        graph = graph_fixture()
        graph["21"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["8", 0], "cfg": 4}}
        graph["22"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 99}}
        graph["23"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m"}}
        graph["24"] = {"class_type": "BasicScheduler", "inputs": {"steps": 18, "scheduler": "karras", "denoise": 0.7}}
        graph["3"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["21", 0], "noise": ["22", 0], "sampler": ["23", 0], "sigmas": ["24", 0], "latent_image": ["5", 0]}}
        self.png(prompt=graph)
        meta = extract_from_file(self.path)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("a blue bird", "bad anatomy"))
        self.assertEqual(meta.parameters["Seed"], 99)
        self.assertEqual(meta.parameters["Steps"], 18)
        self.assertEqual(meta.parameters["CFG scale"], 4)

    def test_workflow_only_png_reads_links_and_known_widget_layouts(self):
        nodes = [
            {"id": 1, "type": "CLIPTextEncode", "widgets_values": ["workflow positive"], "inputs": []},
            {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["workflow negative"], "inputs": []},
            {"id": 3, "type": "KSampler", "widgets_values": [314, "randomize", 24, 7, "euler", "normal", 1],
             "inputs": [{"name": "positive", "link": 11}, {"name": "negative", "link": 12}]},
            {"id": 4, "type": "VAEDecode", "inputs": [{"name": "samples", "link": 13}]},
            {"id": 5, "type": "SaveImage", "inputs": [{"name": "images", "link": 14}]},
        ]
        workflow = {"nodes": nodes, "links": [[11, 1, 0, 3, 0, "CONDITIONING"], [12, 2, 0, 3, 1, "CONDITIONING"], [13, 3, 0, 4, 0, "LATENT"], [14, 4, 0, 5, 0, "IMAGE"]]}
        self.png(workflow=workflow)
        meta = extract_from_file(self.path)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("workflow positive", "workflow negative"))
        self.assertEqual(meta.parameters["Seed"], 314)
        self.assertEqual(meta.parameters["Steps"], 24)
        self.assertEqual(meta.workflow, workflow)

    def test_ui_contract_keeps_raw_graphs_and_normalized_fields_separate(self):
        from core.image_metadata import read_metadata_for_ui
        graph = graph_fixture()
        self.png(prompt=graph, workflow={"nodes": [], "extra": {"keep": True}})
        data = read_metadata_for_ui(self.path)
        self.assertEqual(data["prompt"], "a blue bird")
        self.assertEqual(data["negative"], "bad anatomy")
        self.assertEqual(data["source"], "comfyui")
        self.assertEqual(data["parameters"]["Seed"], 42)
        self.assertIn("Steps: 28", data["params_line"])
        self.assertEqual(json.loads(data["raw_prompt"]), graph)
        self.assertTrue(data["raw_workflow"])
        self.assertTrue(data["can_apply"])

    def test_webui_priority_preserves_comfy_raw_and_negative_free_parameters(self):
        raw = "a WebUI cat\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 12"
        graph = graph_fixture()
        self.png(parameters=raw, prompt=graph)
        meta = extract_from_file(self.path)
        self.assertEqual(meta.prompt, "a WebUI cat")
        self.assertEqual(meta.parameters["Seed"], 12)
        self.assertEqual(meta.source, "webui")
        self.assertEqual(json.loads(meta.raw_prompt), graph)

    def test_controlnet_output_roles_and_linked_text_are_followed(self):
        graph = graph_fixture()
        graph["30"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "linked\npositive"}}
        graph["7"]["inputs"]["text"] = ["30", 0]
        graph["31"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["7", 0], "negative": ["8", 0]}}
        graph["3"]["inputs"].update(positive=["31", 0], negative=["31", 1])
        self.png(prompt=graph)
        meta = extract_from_file(self.path)
        self.assertEqual(meta.prompt, "linked\npositive")
        self.assertEqual(meta.negative_prompt, "bad anatomy")

    def test_unknown_custom_encoder_and_cycles_fail_closed(self):
        from core.image_metadata import read_metadata_for_ui
        for encoder in (
            {"class_type": "MysteryPromptExpansion", "inputs": {"text": "must not guess this", "conditioning": ["99", 0]}},
            {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["7", 0]}},
        ):
            graph = graph_fixture()
            graph["7"] = encoder
            self.png(prompt=graph)
            data = read_metadata_for_ui(self.path)
            self.assertEqual(data["prompt"], "")
            self.assertFalse(data["can_apply"])
            self.assertTrue(data["metadata_warnings"])
            self.assertEqual(json.loads(data["raw_prompt"]), graph)

    def test_malformed_graph_json_is_still_available_as_raw(self):
        from core.image_metadata import read_metadata_for_ui
        raw = '{"1": broken JSON'
        self.png(prompt=raw)
        data = read_metadata_for_ui(self.path)
        self.assertEqual(data["raw_prompt"], raw)
        self.assertEqual(data["prompt"], "")
        self.assertFalse(data["can_apply"])
        self.assertTrue(data["metadata_warnings"])

    def test_anima_semantic_nodes_preserve_positive_and_negative_roles(self):
        graph = graph_fixture()
        graph["7"] = {"class_type": "ForgeNeoAnima38V2Prompt", "inputs": {"prompt": "Anima positive"}}
        graph["8"] = {"class_type": "ForgeNeoAnimaQwen35Prompt", "inputs": {"prompt": "Anima negative"}}
        self.png(prompt=graph)
        meta = extract_from_file(self.path)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("Anima positive", "Anima negative"))

    def test_sdxl_matching_encoder_text_can_be_applied_without_duplication(self):
        from core.image_metadata import read_metadata_for_ui
        graph = graph_fixture()
        graph["7"] = {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "same scene", "text_l": "same scene"}}
        self.png(prompt=graph)
        data = read_metadata_for_ui(self.path)
        self.assertEqual(data["prompt"], "same scene")
        self.assertTrue(data["can_apply"])

    def test_comfy_transplant_keeps_original_graph_format_without_inventing_webui(self):
        from core.image_metadata import transplant
        graph = graph_fixture()
        self.png(prompt=graph)
        output = self.path.with_name("copy.png")
        self.assertTrue(transplant(self.path, self.path, output))
        meta = extract_from_file(output)
        self.assertEqual(meta.source, "comfyui")
        self.assertEqual(meta.prompt_graph, graph)
        self.assertEqual(meta.prompt, "a blue bird")

    def test_jpeg_webp_nested_exif_usercomment_remains_readable(self):
        raw = "EXIF positive\nNegative prompt: EXIF negative\nSteps: 30, Seed: 111"
        for suffix in (".jpg", ".webp"):
            path = self.path.with_suffix(suffix)
            exif = Image.Exif()
            exif[0x8769] = {0x9286: b"UNICODE\x00" + raw.encode("utf-16-be")}
            Image.new("RGB", (8, 8)).save(path, exif=exif)
            before = path.read_bytes()
            meta = extract_from_file(path)
            self.assertEqual((meta.prompt, meta.negative_prompt), ("EXIF positive", "EXIF negative"))
            self.assertEqual(meta.parameters["Seed"], 111)
            self.assertEqual(path.read_bytes(), before)

    def test_empty_webui_negative_does_not_swallow_parameter_line(self):
        self.png(parameters="a cat\nNegative prompt: \nSteps: 21, Seed: 222")
        meta = extract_from_file(self.path)
        self.assertEqual(meta.negative_prompt, "")
        self.assertEqual(meta.parameters["Seed"], 222)

    def test_basic_guider_h3_cache_node_fails_closed_without_opening_cache(self):
        from core.image_metadata import read_metadata_for_ui
        graph = graph_fixture()
        graph["6"] = {"class_type": "ForgeNeoH3ConditioningCacheLoad", "inputs": {"descriptor": '{"path":"C:/private/cache"}'}}
        graph["21"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["6", 0]}}
        graph["3"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["21", 0]}}
        self.png(prompt=graph)
        data = read_metadata_for_ui(self.path)
        self.assertFalse(data["can_apply"])
        self.assertTrue(data["metadata_warnings"])

    def test_many_shared_conditioning_branches_obey_budget(self):
        from core.image_metadata import read_metadata_for_ui
        graph = graph_fixture()
        for number in range(100, 124):
            previous = str(number - 1) if number > 100 else "7"
            graph[str(number)] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": [previous, 0], "conditioning_2": [previous, 0]}}
        graph["3"]["inputs"]["positive"] = ["123", 0]
        self.png(prompt=graph)
        data = read_metadata_for_ui(self.path)
        self.assertFalse(data["can_apply"])
        self.assertTrue(any("너무 많아" in warning for warning in data["metadata_warnings"]))

    def test_sampler_candidate_limit_cannot_turn_partial_read_into_safe_apply(self):
        from core.image_metadata import read_metadata_for_ui
        graph = graph_fixture()
        graph.pop("9")
        graph.pop("10")
        for number in range(1000, 1129):
            graph[str(number)] = {"class_type": "KSampler", "inputs": dict(graph["3"]["inputs"])}
        self.png(prompt=graph)
        data = read_metadata_for_ui(self.path)
        self.assertEqual(len(data["prompt_candidates"]), 128)
        self.assertFalse(data["can_apply"])
        self.assertTrue(data["metadata_ambiguous"])


def _insert_png_chunk_before_iend(path: Path, chunk_type: bytes, data: bytes) -> None:
    """IDAT 뒤(IEND 앞)에 청크를 끼운다 — 일부 편집기가 메타를 픽셀 뒤에 붙이는 모양."""
    import struct
    import zlib
    raw = path.read_bytes()
    iend = raw.rindex(b"IEND") - 4
    chunk = struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    path.write_bytes(raw[:iend] + chunk + raw[iend:])


class InfotextGoldenTests(unittest.TestCase):
    """A1111 infotext 파서 골든 — UI 액션·워커·브리지가 모두 이 결과를 쓴다."""

    def parse(self, text):
        from core.image_metadata import parse_infotext
        meta = parse_infotext(text)
        return meta.prompt, meta.negative_prompt, meta.parameters

    def test_negative_free_infotext_keeps_parameter_line_out_of_prompt(self):
        prompt, negative, params = self.parse("a cat, 1girl\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 123, Size: 832x1216")
        self.assertEqual((prompt, negative), ("a cat, 1girl", ""))
        self.assertEqual(params, {"Steps": 20, "Sampler": "Euler", "CFG scale": 7, "Seed": 123, "Size": "832x1216"})

    def test_empty_positive_starts_with_negative_prompt_line(self):
        prompt, negative, params = self.parse("Negative prompt: lowres\nSteps: 20, Sampler: Euler, CFG scale: 7")
        self.assertEqual((prompt, negative), ("", "lowres"))
        self.assertEqual(params["Sampler"], "Euler")

    def test_parameters_only_infotext_has_no_prompt(self):
        prompt, negative, params = self.parse("Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 5")
        self.assertEqual((prompt, negative), ("", ""))
        self.assertEqual(params["Seed"], 5)

    def test_crlf_and_multiline_sections(self):
        prompt, negative, params = self.parse("line one\r\nline two\r\nNegative prompt: n1\r\nn2\r\nSteps: 20, Seed: 1")
        self.assertEqual((prompt, negative), ("line one\nline two", "n1\nn2"))
        self.assertEqual(params, {"Steps": 20, "Seed": 1})

    def test_prompt_lines_that_look_like_pairs_stay_in_the_prompt(self):
        for text, expected in (
            ("Size: huge, masterpiece", "Size: huge, masterpiece"),
            ("masterpiece\nSize: huge, masterpiece\nSteps: 20, Seed: 1", "masterpiece\nSize: huge, masterpiece"),
            ("(masterpiece:1.2), (best quality:1.1), (1girl:1.3)\nSteps: 20, Sampler: Euler",
             "(masterpiece:1.2), (best quality:1.1), (1girl:1.3)"),
            ("hair: blue, eyes: red, outfit: dress\nSteps: 20, Sampler: Euler", "hair: blue, eyes: red, outfit: dress"),
            ("caption\nSteps: the staircase", "caption\nSteps: the staircase"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text)[0], expected)

    def test_parameter_line_without_steps_is_still_recognised(self):
        prompt, _negative, params = self.parse("a cat\nSampler: Euler a, CFG scale: 7")
        self.assertEqual(prompt, "a cat")
        self.assertEqual(params, {"Sampler": "Euler a", "CFG scale": 7})

    def test_quoted_values_are_unquoted_with_json_rules(self):
        _p, _n, params = self.parse(
            'a cat\nNegative prompt: bad\nSteps: 20, ADetailer prompt: "smile, blue eyes", '
            'Lora hashes: "a: 1, b: 2", Wild: "say \\"hi\\", ok", Seed: 3')
        self.assertEqual(params["ADetailer prompt"], "smile, blue eyes")
        self.assertEqual(params["Lora hashes"], "a: 1, b: 2")
        self.assertEqual(params["Wild"], 'say "hi", ok')
        self.assertEqual(params["Seed"], 3)
        self.assertNotIn("b", params)

    def test_template_tail_lines_are_parameters_not_negative(self):
        from core.image_metadata import split_infotext
        text = ("a {red|blue} cat\nNegative prompt: bad\nSteps: 20, Sampler: Euler, Seed: 1\n"
                "Template: a {red|blue} cat, sitting\nNegative Template: bad, worse")
        prompt, negative, params = self.parse(text)
        self.assertEqual((prompt, negative), ("a {red|blue} cat", "bad"))
        self.assertEqual(params["Template"], "a {red|blue} cat, sitting")
        self.assertEqual(params["Negative Template"], "bad, worse")
        self.assertTrue(split_infotext(text)[2].endswith("Negative Template: bad, worse"))

    def test_ui_params_line_is_the_raw_tail_and_groups_come_from_the_dict(self):
        from core.image_metadata import read_metadata_for_ui
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "quoted.png"
        tail = 'Steps: 20, Sampler: Euler, ADetailer prompt: "smile, blue eyes", Lora hashes: "a: 1, b: 2"'
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "a cat\n" + tail)
        Image.new("RGB", (8, 8)).save(path, pnginfo=info)
        data = read_metadata_for_ui(path)
        self.assertEqual(data["params_line"], tail)
        self.assertEqual(data["params"]["extensions"], 'ADetailer prompt: "smile, blue eyes"')
        self.assertEqual(data["params"]["other"], 'Lora hashes: "a: 1, b: 2"')

    def test_comfy_params_line_quotes_values_like_a1111(self):
        from core.image_metadata import format_parameters_line, parse_parameters_text
        params = {"Steps": 28, "Model": "a, b.safetensors", "Note": "x: y", "Skip": None}
        line = format_parameters_line(params)
        self.assertEqual(line, 'Steps: 28, Model: "a, b.safetensors", Note: "x: y"')
        self.assertEqual(parse_parameters_text(line), {"Steps": 28, "Model": "a, b.safetensors", "Note": "x: y"})

    def test_replace_prompt_keeps_the_parameter_tail_verbatim(self):
        from core.image_metadata import replace_prompt_in_parameters
        text = ('old\nNegative prompt: old neg\nSteps: 20, Lora hashes: "a: 1, b: 2"\n'
                "Template: a {red|blue} cat, sitting")
        self.assertEqual(
            replace_prompt_in_parameters(text, "new prompt", "new neg"),
            'new prompt\nNegative prompt: new neg\nSteps: 20, Lora hashes: "a: 1, b: 2"\nTemplate: a {red|blue} cat, sitting')
        # 파라미터만 있던 이미지에 Steps 줄이 두 번 들어가지 않는다
        self.assertEqual(replace_prompt_in_parameters("Steps: 20, Seed: 5", "", ""), "Steps: 20, Seed: 5")
        self.assertEqual(replace_prompt_in_parameters("a\nSteps: 20, Seed: 5", "b", ""), "b\nSteps: 20, Seed: 5")

    # ── 실제 A1111/Forge 줄 — 줄 전체를 'Key: value' 로 fullmatch 하던 파서가 놓친 모양들 ──

    def test_forge_neo_anima_line_with_dotted_keys_is_the_parameter_line(self):
        # 사용자 Forge Neo + SAM3 + Anima 3.8B 이미지의 줄 구조(이름·해시는 익명화)
        line = (
            r'Steps: 50, Sampler: ER SDE, Schedule type: Beta57 (RES4LYF), CFG scale: 5.5, Shift: 3.0, '
            r'Seed: 113, Size: 2048x1144, Model: Anima-3.8B-v1.1, Model hash: 0123456789, '
            r'Module 1: vae_a, Module 2: te_b, RNG: CPU, SAM3 Enable: True, SAM3 Prompt: "face, eyes, hair", '
            r'SAM3 Inpaint Prompt: "1girl, solo, <lora:style_a:0.85>", '
            r'SAM3 Checkpoint: "C:\\models\\sam3\\sam3.safetensors", SAM3 Threshold: 0.4, '
            r'Lora hashes: "style_a: 0x0000000001, style_b: 0x0000000002", '
            r'Anima 3.8B adapter: adapter_v2.safetensors, Anima 3.8B strength: 1.0, '
            r'Anima 3.8B architecture: semantic_connector_v2, Anima 3.8B bundle: Anima-3.8B-v1.1.safetensors, '
            r'NegPiP: True, Version: neo-2.29')
        for text, expected in (
            ("1girl, solo\nNegative prompt: lowres, bad hands\n" + line, ("1girl, solo", "lowres, bad hands")),
            ("1girl, solo\n" + line, ("1girl, solo", "")),
            ("Negative prompt: lowres\n" + line, ("", "lowres")),
        ):
            with self.subTest(text=text[:40]):
                prompt, negative, params = self.parse(text)
                self.assertEqual((prompt, negative), expected)
                self.assertEqual(params["Size"], "2048x1144")
                self.assertEqual(params["Schedule type"], "Beta57 (RES4LYF)")
                self.assertEqual(params["Anima 3.8B adapter"], "adapter_v2.safetensors")
                self.assertEqual(params["Anima 3.8B strength"], 1.0)
                self.assertEqual(params["Anima 3.8B bundle"], "Anima-3.8B-v1.1.safetensors")
                self.assertEqual(params["SAM3 Prompt"], "face, eyes, hair")
                self.assertEqual(params["SAM3 Checkpoint"], "C:\\models\\sam3\\sam3.safetensors")
                self.assertEqual(params["Lora hashes"], "style_a: 0x0000000001, style_b: 0x0000000002")
                self.assertEqual(params["Version"], "neo-2.29")
                self.assertNotIn("8B adapter", params)
                self.assertNotIn("style_b", params)

    def test_unquoted_json_values_are_one_parameter_and_do_not_leak(self):
        from core.image_metadata import group_parameters
        cases = {
            "Hashes": '{"vae": "a1b2", "model": "c3d4", "lora:x": "e5f6"}',
            "Civitai resources": '[{"type": "checkpoint", "modelVersionId": 1}, {"type": "lora", "weight": 0.8}]',
            "Tiled Diffusion": '{"Method": "MultiDiffusion", "Tile tile width": 96, "Note": "a, b: c"}',
        }
        for key, value in cases.items():
            for head in ("a cat\nNegative prompt: lowres\n", "a cat\n"):
                with self.subTest(key=key, negative=bool(head.count("\n") - 1)):
                    prompt, negative, params = self.parse(f"{head}Steps: 20, Sampler: Euler, {key}: {value}, Seed: 7")
                    self.assertEqual((prompt, negative), ("a cat", "lowres" if "Negative" in head else ""))
                    self.assertEqual(params[key], value)
                    self.assertEqual((params["Steps"], params["Sampler"], params["Seed"]), (20, "Euler", 7))
                    # 표시 그룹에도 이스케이프된 따옴표 없이 그대로 나온다
                    self.assertIn(f"{key}: {value}", "".join(group_parameters(params).values()))
        # Steps 없이 쓰는 도구의 줄도 점 키·JSON 값 때문에 탈락하지 않는다
        prompt, _negative, params = self.parse(
            'a cat\nSampler: Euler, CFG scale: 7, Anima 3.8B adapter: x.safetensors, Hashes: {"vae": "a"}')
        self.assertEqual(prompt, "a cat")
        self.assertEqual(params["Anima 3.8B adapter"], "x.safetensors")
        self.assertEqual(params["Hashes"], '{"vae": "a"}')

    def test_steps_line_stays_parameters_whatever_follows(self):
        base = "a cat\nNegative prompt: bad\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 1, Size: 512x512"
        for suffix, extra in (
            ("\nTemplate: a cat\nNegative Template:", {"Template": "a cat", "Negative Template": ""}),
            ("\nTemplate: a {red|\nblue} cat, sitting", {"Template": "a {red|\nblue} cat, sitting"}),
            ("\nTemplate: a, b: c, d: e", {"Template": "a, b: c, d: e"}),
            ("\nsome free text trailer", {}),
        ):
            for head in (base, base.replace("\nNegative prompt: bad", "")):
                with self.subTest(suffix=suffix, head=head[:30]):
                    prompt, negative, params = self.parse(head + suffix)
                    self.assertEqual((prompt, negative), ("a cat", "bad" if "Negative" in head else ""))
                    self.assertEqual((params["Steps"], params["Seed"], params["Size"]), (20, 1, "512x512"))
                    for key, value in extra.items():
                        self.assertEqual(params[key], value)

    def test_steps_less_parameter_block_and_pair_like_prompt_lines(self):
        # 알려진 키로 시작하는 Steps 없는 파라미터 줄 여러 개는 한 꼬리로 묶인다
        prompt, _negative, params = self.parse("a cat\nSampler: Euler, CFG scale: 7, Seed: 1\nSize: 512x512, Model: m")
        self.assertEqual(prompt, "a cat")
        self.assertEqual(params, {"Sampler": "Euler", "CFG scale": 7, "Seed": 1, "Size": "512x512", "Model": "m"})
        # 'Key: value' 만으로 된 프롬프트 줄은 Steps 없는 파라미터 줄 위에서도 프롬프트로 남는다
        prompt, _negative, params = self.parse("hair: blue, eyes: red, outfit: dress\nSampler: Euler, CFG scale: 7")
        self.assertEqual(prompt, "hair: blue, eyes: red, outfit: dress")
        self.assertEqual(params, {"Sampler": "Euler", "CFG scale": 7})
        # 값에 쌍이 든 Template 줄은 파라미터 줄로 오인하지 않는다
        prompt, _negative, params = self.parse("a cat\nSampler: Euler, CFG scale: 7\nTemplate: x: 1, y: 2, z: 3")
        self.assertEqual(prompt, "a cat")
        self.assertEqual(params["Template"], "x: 1, y: 2, z: 3")
        self.assertNotIn("y", params)

    def test_pasted_parameter_line_in_the_prompt_does_not_win(self):
        prompt, negative, params = self.parse("a cat\nSteps: 5, Seed: 9\nNegative prompt: bad\nSteps: 20, Seed: 1")
        self.assertEqual((prompt, negative), ("a cat\nSteps: 5, Seed: 9", "bad"))
        self.assertEqual(params, {"Steps": 20, "Seed": 1})
        # 네거티브 뒤에 파라미터 줄이 없으면 프롬프트 속 Steps 줄은 프롬프트로 남는다
        self.assertEqual(self.parse("a cat\nSteps: 5, Seed: 9\nNegative prompt: bad"),
                         ("a cat\nSteps: 5, Seed: 9", "bad", {}))

    def test_json_container_values_round_trip_without_quotes(self):
        from core.image_metadata import format_parameters_line, parse_parameters_text
        params = {"Steps": 20, "Hashes": '{"vae": "a", "model": "b"}', "Obj": {"a": 1, "b": [1, 2]},
                  "Text": "{not, json", "Multi": "[a,\nb]"}
        line = format_parameters_line(params)
        self.assertEqual(line, 'Steps: 20, Hashes: {"vae": "a", "model": "b"}, Obj: {"a": 1, "b": [1, 2]}, '
                               'Text: "{not, json", Multi: "[a,\\nb]"')
        self.assertEqual(parse_parameters_text(line), {
            "Steps": 20, "Hashes": '{"vae": "a", "model": "b"}', "Obj": '{"a": 1, "b": [1, 2]}',
            "Text": "{not, json", "Multi": "[a,\nb]"})

    def test_parsing_stays_linear_on_long_and_malformed_lines(self):
        """예전 fullmatch 정규식은 실패하는 줄에서 LoRA 해시 하나마다 두 배로 느려졌다(GUI 정지)."""
        import time
        from core.image_metadata import parse_infotext
        hashes = ", ".join(f"lora_{i}: 0x{i:010x}" for i in range(30))
        line = ('Steps: 20, Sampler: Euler, SAM3 Prompt: "face, hands", '
                f'Lora hashes: "{hashes}", ' + ", ".join(f"Lora {i}: h{i}" for i in range(30))
                + ', (junk:1.2), Anima 3.8B adapter: x.safetensors, Hashes: {"vae": "a"}')
        pairs = ", ".join(f"tag{i}: {i}" for i in range(5000))
        texts = {
            "hashes_and_failing_segment": "a cat\nNegative prompt: bad\n" + line,
            "steps_less_strict_check": "a cat\n" + line.replace("Steps: 20, ", ""),
            "long_pair_prompt_without_steps": pairs + ", (tail:1.1)",
            "unclosed_brackets": "a cat\nSteps: 1, " + "k: {a, " * 5000,
            "nested_brackets_without_separator": "a cat\nSteps: 1, " + "k: {a, " * 3000 + "}" * 3000 + "x",
            "many_quotes": "a cat\nSteps: 1, " + 'k: "x, ' * 5000,
        }
        for name, text in texts.items():
            with self.subTest(case=name):
                best = float("inf")
                for _ in range(3):
                    started = time.perf_counter()
                    parse_infotext(text)
                    best = min(best, time.perf_counter() - started)
                self.assertLess(best, 0.05, f"{name}: {best * 1000:.1f} ms")
        meta = parse_infotext(texts["hashes_and_failing_segment"])
        self.assertEqual((meta.prompt, meta.negative_prompt), ("a cat", "bad"))
        self.assertEqual(meta.parameters["Lora hashes"], hashes)
        self.assertEqual(meta.parameters["Lora 29"], "h29")
        self.assertEqual(meta.parameters["Anima 3.8B adapter"], "x.safetensors")


class PngParameterRewriteTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "saved.png"

    def test_rewrite_keeps_exif_dpi_other_chunks_and_trailing_text(self):
        from core.image_metadata import extract_from_file, rewrite_png_parameters
        exif = Image.Exif()
        exif[0x010F] = "CameraMaker"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "old\nSteps: 20, Seed: 1")
        info.add_text("workflow", '{"nodes": []}')
        Image.new("RGB", (8, 8), "red").save(self.path, pnginfo=info, exif=exif, dpi=(300, 300))
        _insert_png_chunk_before_iend(self.path, b"tEXt", b"Comment\x00after idat")

        rewrite_png_parameters(self.path, "new\nSteps: 20, Seed: 1")

        with Image.open(self.path) as img:
            self.assertEqual(img.getexif().get(0x010F), "CameraMaker")
            self.assertEqual(tuple(round(v) for v in img.info["dpi"]), (300, 300))
            self.assertEqual(img.text["workflow"], '{"nodes": []}')
            self.assertEqual(img.text["Comment"], "after idat")
            self.assertEqual(img.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(extract_from_file(self.path).prompt, "new")
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["saved.png"])

    def test_rewrite_refuses_non_png_and_leaves_file_untouched(self):
        from core.image_metadata import rewrite_png_parameters
        jpeg = self.path.with_suffix(".jpg")
        Image.new("RGB", (8, 8)).save(jpeg)
        before = jpeg.read_bytes()
        with self.assertRaises(ValueError):
            rewrite_png_parameters(jpeg, "x")
        self.assertEqual(jpeg.read_bytes(), before)

    @staticmethod
    def _chunks(path):
        from core.png_chunks import split_chunks
        data = Path(path).read_bytes()
        chunks, _end = split_chunks(data)
        return [(ctype, data[start:end]) for ctype, start, end in chunks]

    def test_rewrite_keeps_apng_frames_and_timing(self):
        # 예전 Pillow 재저장은 첫 프레임만 남기고 acTL/fcTL/fdAT 를 버렸다
        from core.image_metadata import extract_from_file, rewrite_png_parameters
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        frames = [Image.new("RGB", (6, 6), c) for c in colors]
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "old cat\nSteps: 20, Seed: 1")
        frames[0].save(self.path, save_all=True, append_images=frames[1:], duration=[100, 200, 300],
                       loop=0, pnginfo=info)

        rewrite_png_parameters(self.path, "new cat\nSteps: 20, Seed: 1")

        with Image.open(self.path) as img:
            self.assertTrue(img.is_animated)
            self.assertEqual(img.n_frames, 3)
            self.assertEqual(img.info["loop"], 0)
            for index, (color, duration) in enumerate(zip(colors, (100, 200, 300))):
                img.seek(index)
                self.assertEqual(img.convert("RGB").getpixel((0, 0)), color)
                self.assertEqual(img.info["duration"], duration)
        self.assertEqual(extract_from_file(self.path).prompt, "new cat")

    def test_rewrite_changes_only_the_parameters_chunk_of_a_16bit_png(self):
        # 16비트 깊이·gAMA·bKGD·tIME·IDAT 뒤 텍스트가 바이트 그대로 남는다
        import struct
        from core.image_metadata import rewrite_png_parameters
        info = PngImagePlugin.PngInfo()
        info.add(b"gAMA", struct.pack(">I", 45455))
        info.add(b"bKGD", struct.pack(">H", 1000))
        info.add(b"tIME", b"\x07\xea\x09\x18\x0c\x00\x00")
        info.add_text("parameters", "old\nSteps: 20, Seed: 1")
        Image.new("I;16", (5, 4), 40000).save(self.path, pnginfo=info)
        _insert_png_chunk_before_iend(self.path, b"tEXt", b"Comment\x00after idat")
        before = self._chunks(self.path)

        rewrite_png_parameters(self.path, "new\nSteps: 20, Seed: 1")

        after = self._chunks(self.path)
        strip = [c for c in before if not c[1][8:].startswith(b"parameters\x00")]
        self.assertEqual([c for c in after if not c[1][8:].startswith(b"parameters\x00")], strip)
        self.assertEqual(after[0][1][16], 16)   # IHDR 비트 깊이
        with Image.open(self.path) as img:
            self.assertEqual(img.info["parameters"], "new\nSteps: 20, Seed: 1")
            self.assertEqual(img.getpixel((0, 0)), 40000)

    def test_rewrite_non_latin1_prompt_round_trips(self):
        from core.image_metadata import extract_from_file, rewrite_png_parameters
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "old\nSteps: 20, Seed: 1")
        Image.new("RGB", (4, 4)).save(self.path, pnginfo=info)
        rewrite_png_parameters(self.path, "한글 고양이, 猫\nNegative prompt: 나쁨\nSteps: 20, Seed: 1")
        meta = extract_from_file(self.path)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("한글 고양이, 猫", "나쁨"))
        self.assertIn((b"iTXt", True), [(t, c[8:].startswith(b"parameters\x00")) for t, c in self._chunks(self.path)])

    def test_rewrite_replaces_duplicate_compressed_and_trailing_parameters(self):
        from core.image_metadata import extract_from_file, rewrite_png_parameters
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "zipped old\nSteps: 20, Seed: 1", zip=True)
        Image.new("RGB", (4, 4)).save(self.path, pnginfo=info)
        _insert_png_chunk_before_iend(self.path, b"tEXt", b"parameters\x00trailing old\nSteps: 20, Seed: 1")
        rewrite_png_parameters(self.path, "one\nSteps: 20, Seed: 1")
        params = [c for _t, c in self._chunks(self.path) if c[8:].startswith(b"parameters\x00")]
        self.assertEqual(len(params), 1)
        self.assertEqual(extract_from_file(self.path).prompt, "one")

    def test_rewrite_inserts_a_chunk_when_the_webui_text_came_from_exif(self):
        from core.image_metadata import MetadataSource, extract_from_file, rewrite_png_parameters
        exif = Image.Exif()
        exif[0x8769] = {0x9286: b"UNICODE\0" + "exif cat\nSteps: 20, Seed: 1".encode("utf-16-be")}
        Image.new("RGB", (4, 4)).save(self.path, exif=exif)
        self.assertEqual(extract_from_file(self.path).source, MetadataSource.WEBUI)
        rewrite_png_parameters(self.path, "chunk cat\nSteps: 20, Seed: 1")
        types = [t for t, _c in self._chunks(self.path)]
        self.assertLess(types.index(b"tEXt"), types.index(b"IDAT"))
        self.assertEqual(extract_from_file(self.path).prompt, "chunk cat")

    def test_rewrite_refuses_a_corrupt_png_and_leaves_it_untouched(self):
        from core.image_metadata import rewrite_png_parameters
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "old\nSteps: 20, Seed: 1")
        Image.new("RGB", (4, 4)).save(self.path, pnginfo=info)
        truncated = self.path.read_bytes()[:-20]
        self.path.write_bytes(truncated)
        with self.assertRaises(ValueError):
            rewrite_png_parameters(self.path, "x")
        self.assertEqual(self.path.read_bytes(), truncated)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["saved.png"])


class TrailingPngMetadataTests(unittest.TestCase):
    """IDAT 뒤(IEND 앞)에 붙은 메타 — UI·검색·이식·워커가 모두 같은 것을 본다."""

    TAIL = "tail cat\nNegative prompt: tail neg\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 8x8"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def plain(self, name):
        path = self.dir / name
        Image.new("RGB", (8, 8), "blue").save(path)
        return path

    def test_trailing_parameters_are_seen_everywhere(self):
        from core.image_metadata import (
            MetadataSource, extract_from_file, read_applicable_prompts, read_metadata_for_ui, transplant,
        )
        from core.metadata_search import metadata_search_text
        latin = self.plain("latin.png")
        _insert_png_chunk_before_iend(latin, b"tEXt", b"parameters\x00" + self.TAIL.encode("latin-1"))
        korean_tail = self.TAIL.replace("tail cat", "한글 고양이")
        korean = self.plain("korean.png")
        _insert_png_chunk_before_iend(korean, b"iTXt", b"parameters\x00\x00\x00\x00\x00" + korean_tail.encode("utf-8"))
        for path, prompt, raw in ((latin, "tail cat", self.TAIL), (korean, "한글 고양이", korean_tail)):
            with self.subTest(path=path.name):
                meta = extract_from_file(path)
                self.assertEqual((meta.source, meta.prompt, meta.negative_prompt),
                                 (MetadataSource.WEBUI, prompt, "tail neg"))
                ui = read_metadata_for_ui(path)
                self.assertEqual(ui["source"], "webui")
                self.assertEqual(ui["prompt"], prompt)
                self.assertEqual(ui["params_line"], "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 8x8")
                self.assertTrue(ui["can_apply"])
                self.assertIn(prompt, metadata_search_text(path))
                self.assertEqual(read_applicable_prompts(path), (prompt, "tail neg", ""))
                out = self.dir / f"{path.stem}_withmeta.png"
                self.assertTrue(transplant(path, self.plain(f"{path.stem}_target.png"), out))
                self.assertEqual(extract_from_file(out).raw_parameters, raw)

    def test_trailing_comfy_prompt_is_comfy_for_the_ui_too(self):
        from core.image_metadata import MetadataSource, extract_from_file, read_applicable_prompts
        path = self.plain("comfy.png")
        _insert_png_chunk_before_iend(path, b"tEXt", b"prompt\x00" + json.dumps(graph_fixture()).encode("latin-1"))
        meta = extract_from_file(path)
        self.assertEqual(meta.source, MetadataSource.COMFYUI)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("a blue bird", "bad anatomy"))
        self.assertEqual(read_applicable_prompts(path), ("a blue bird", "bad anatomy", ""))

    # ── 헤더 기록이 반쪽이면 뒤쪽이 빠진 키를 채운다 (Codex S5 #3) ──
    # 예전엔 헤더에 생성 키가 하나만 있어도 뒤쪽을 훑지 않아, 헤더 workflow + 뒤쪽 prompt 는 프롬프트가
    # 비고 '모호함'(워커는 적용 거부), 이식은 뒤쪽 그래프를 조용히 버렸다.

    UI_WORKFLOW = {"nodes": [], "links": [], "extra": {"label": "header"}}   # 프롬프트를 못 주는 UI 그래프

    def header_png(self, name, **text):
        path = self.dir / name
        info = PngImagePlugin.PngInfo()
        for key, value in text.items():
            info.add_text(key, value if isinstance(value, str) else json.dumps(value))
        Image.new("RGB", (8, 8), "blue").save(path, pnginfo=info)
        return path

    @staticmethod
    def trail(path, key, value):
        raw = value if isinstance(value, str) else json.dumps(value)
        _insert_png_chunk_before_iend(path, b"tEXt", key.encode("latin-1") + b"\x00" + raw.encode("latin-1"))

    def test_header_workflow_with_trailing_prompt_is_applicable_everywhere(self):
        from core.image_metadata import (
            MetadataSource, extract_from_file, read_applicable_prompts, read_metadata_for_ui, transplant,
        )
        from core.metadata_search import metadata_search_text
        path = self.header_png("split.png", workflow=self.UI_WORKFLOW)
        self.trail(path, "prompt", graph_fixture())
        meta = extract_from_file(path)
        self.assertEqual(meta.source, MetadataSource.COMFYUI)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("a blue bird", "bad anatomy"))
        self.assertTrue(meta.comfy_parse_complete)
        self.assertEqual(meta.prompt_graph, graph_fixture())
        self.assertEqual(json.loads(meta.raw_workflow), self.UI_WORKFLOW)   # 헤더 것
        self.assertEqual(read_applicable_prompts(path), ("a blue bird", "bad anatomy", ""))
        ui = read_metadata_for_ui(path)
        self.assertTrue(ui["can_apply"])
        self.assertFalse(ui["metadata_ambiguous"])
        self.assertIn("a blue bird", metadata_search_text(path))
        out = self.dir / "split_withmeta.png"
        self.assertTrue(transplant(path, self.plain("split_target.png"), out))
        copied = extract_from_file(out)
        self.assertEqual(json.loads(copied.raw_prompt), graph_fixture())
        self.assertEqual(json.loads(copied.raw_workflow), self.UI_WORKFLOW)

    def test_header_prompt_with_trailing_workflow_keeps_both(self):
        from core.image_metadata import extract_from_file, transplant
        trailing_workflow = {"nodes": [], "links": [], "extra": {"label": "trailing"}}
        path = self.header_png("prompt_head.png", prompt=graph_fixture())
        self.trail(path, "workflow", trailing_workflow)
        meta = extract_from_file(path)
        self.assertEqual(meta.prompt, "a blue bird")
        self.assertEqual(json.loads(meta.raw_workflow), trailing_workflow)
        out = self.dir / "prompt_head_withmeta.png"
        self.assertTrue(transplant(path, self.plain("prompt_head_target.png"), out))
        copied = extract_from_file(out)
        self.assertEqual(json.loads(copied.raw_prompt), graph_fixture())
        self.assertEqual(json.loads(copied.raw_workflow), trailing_workflow)

    def test_header_keys_win_over_trailing_copies_and_trailing_parameters_never_flip_the_source(self):
        from core.image_metadata import MetadataSource, extract_from_file
        # 헤더 workflow + 뒤쪽의 다른 workflow + 뒤쪽 prompt → workflow 는 헤더 것
        path = self.header_png("both.png", workflow=self.UI_WORKFLOW)
        self.trail(path, "workflow", {"nodes": [], "extra": {"label": "trailing"}})
        self.trail(path, "prompt", graph_fixture())
        meta = extract_from_file(path)
        self.assertEqual(json.loads(meta.raw_workflow), self.UI_WORKFLOW)
        self.assertEqual(meta.prompt, "a blue bird")
        # 헤더 Comfy 기록(반쪽이든 완결이든) + 뒤쪽 A1111 parameters → 소스는 ComfyUI 그대로
        for name, header in (("half.png", {"prompt": graph_fixture()}),
                             ("full.png", {"prompt": graph_fixture(), "workflow": self.UI_WORKFLOW})):
            with self.subTest(header=name):
                path = self.header_png(name, **header)
                self.trail(path, "parameters", self.TAIL)
                meta = extract_from_file(path)
                self.assertEqual(meta.source, MetadataSource.COMFYUI)
                self.assertEqual(meta.prompt, "a blue bird")
                self.assertEqual(meta.raw_parameters, "")

    def test_complete_header_records_do_not_scan_the_trailing_chunks(self):
        # 빠른 길 — Forge(parameters)·ComfyUI SaveImage(prompt+workflow) 헤더는 파일을 다시 열어 훑지 않는다
        from unittest import mock
        from core.image_metadata import extract_from_pil
        cases = (
            ("forge.png", {"parameters": self.TAIL}, "tail cat"),
            ("comfy.png", {"prompt": graph_fixture(), "workflow": self.UI_WORKFLOW}, "a blue bird"),
        )
        for name, header, prompt in cases:
            with self.subTest(name=name):
                path = self.header_png(name, **header)
                with mock.patch("core.image_metadata._png_trailing_text_and_exif",
                                side_effect=AssertionError("완결된 헤더인데 뒤쪽을 훑었다")), Image.open(path) as img:
                    self.assertEqual(extract_from_pil(img).prompt, prompt)

    def test_description_infotext_is_shared_by_ui_search_and_worker(self):
        from core.image_metadata import read_applicable_prompts, read_metadata_for_ui
        from core.metadata_search import metadata_search_text
        path = self.dir / "described.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("Description", self.TAIL)
        Image.new("RGB", (8, 8)).save(path, pnginfo=info)
        self.assertEqual(read_metadata_for_ui(path)["prompt"], "tail cat")
        self.assertIn("tail cat", metadata_search_text(path))
        self.assertEqual(read_applicable_prompts(path), ("tail cat", "tail neg", ""))
        # 파라미터 줄 없는 설명문은 생성 메타가 아니다
        plain = self.dir / "caption.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("Description", "a photo of my cat")
        Image.new("RGB", (8, 8)).save(plain, pnginfo=info)
        self.assertEqual(read_metadata_for_ui(plain)["source"], "unknown")

    def test_trailing_read_restores_the_file_position_for_pixel_decoding(self):
        from core.image_metadata import extract_from_pil
        path = self.plain("pixels.png")
        _insert_png_chunk_before_iend(path, b"tEXt", b"parameters\x00" + self.TAIL.encode("latin-1"))
        with Image.open(path) as img:
            self.assertEqual(extract_from_pil(img).prompt, "tail cat")
            self.assertEqual(img.getpixel((0, 0)), (0, 0, 255))   # 픽셀 디코드가 그대로 된다
            # 이미 load() 한 이미지(파일이 닫힘)도 같은 결과
            self.assertEqual(extract_from_pil(img).prompt, "tail cat")
        with Image.open(path) as img:
            self.assertEqual(extract_from_pil(img, extra_text={}).prompt, "")   # 명시적으로 끄면 헤더만

    # ── 메타 없는 PNG 에서 픽셀을 디코드하지 않는다 ──
    # PngImageFile.getexif() 는 헤더에 eXIf 가 없으면 load() 로 픽셀을 전부 디코드했다(GUI 스레드 슬롯).

    def _forbid_pixel_decode(self):
        from unittest import mock
        return mock.patch.object(PngImagePlugin.PngImageFile, "load",
                                 side_effect=AssertionError("메타 읽기가 픽셀을 디코드했다"))

    def _usercomment_exif_bytes(self, text):
        exif = Image.Exif()
        exif[0x8769] = {0x9286: b"UNICODE\0" + text.encode("utf-16-be")}
        return exif.tobytes()

    def test_metadata_less_png_is_read_without_decoding_pixels(self):
        from core.image_metadata import extract_from_pil, read_applicable_prompts, read_metadata_for_ui
        path = self.plain("nometa.png")
        with self._forbid_pixel_decode():
            ui = read_metadata_for_ui(path)
            self.assertEqual((ui["source"], ui["size"]), ("unknown", "8 × 8"))
            with Image.open(path) as img:
                self.assertFalse(extract_from_pil(img).has_any())
            self.assertEqual(read_applicable_prompts(path)[0], "")

    def test_png_exif_usercomment_is_read_before_or_after_idat_without_decoding(self):
        from core.image_metadata import MetadataSource, extract_from_file, extract_from_pil, read_metadata_for_ui
        raw = "exif cat\nNegative prompt: exif neg\nSteps: 20, Seed: 1"
        exif = self._usercomment_exif_bytes(raw)
        header = self.dir / "header_exif.png"
        Image.new("RGB", (8, 8), "blue").save(header, exif=exif)
        trailing = self.plain("trailing_exif.png")
        _insert_png_chunk_before_iend(trailing, b"eXIf", exif[6:] if exif.startswith(b"Exif\0\0") else exif)
        for path in (header, trailing):
            with self.subTest(path=path.name), self._forbid_pixel_decode():
                ui = read_metadata_for_ui(path)
                self.assertEqual((ui["source"], ui["prompt"], ui["negative"]), ("webui", "exif cat", "exif neg"))
                self.assertEqual(extract_from_file(path).source, MetadataSource.WEBUI)
        # 이미 load() 한 이미지(파일이 닫힘)도 같은 결과 — getexif() 가 다시 디코드하지 않는다
        with Image.open(trailing) as img:
            img.load()
            self.assertEqual(extract_from_pil(img).prompt, "exif cat")

    def test_trailing_walk_does_not_move_the_pillow_handle(self):
        # 파일에서 연 이미지는 버퍼 없는 핸들을 따로 열어 훑는다 — Pillow 핸들 위치·픽셀 디코드가 그대로
        from core.image_metadata import png_trailing_text
        path = self.plain("handle.png")
        _insert_png_chunk_before_iend(path, b"tEXt", b"parameters\x00" + self.TAIL.encode("latin-1"))
        with Image.open(path) as img:
            before = img.fp.tell()
            self.assertEqual(png_trailing_text(img), {"parameters": self.TAIL})
            self.assertEqual(img.fp.tell(), before)
            self.assertEqual(img.getpixel((0, 0)), (0, 0, 255))
        # 스트림에서 연 이미지는 그 스트림으로 훑고 위치를 되돌린다
        import io
        stream = io.BytesIO(path.read_bytes())
        with Image.open(stream) as img:
            before = stream.tell()
            self.assertEqual(png_trailing_text(img), {"parameters": self.TAIL})
            self.assertEqual(stream.tell(), before)
            self.assertEqual(img.getpixel((0, 0)), (0, 0, 255))


class MetadataTransplantTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def test_webui_transplant_copies_raw_infotext_and_keeps_target_pixels(self):
        from core.image_metadata import extract_from_file, transplant
        source = self.dir / "source.png"
        info = PngImagePlugin.PngInfo()
        raw = 'a cat\nNegative prompt: bad\nSteps: 20, Lora hashes: "a: 1, b: 2"'
        info.add_text("parameters", raw)
        Image.new("RGB", (4, 4), "blue").save(source, pnginfo=info)
        target = self.dir / "edited.png"
        Image.new("RGBA", (6, 5), (0, 255, 0, 128)).save(target, dpi=(144, 144))
        out = self.dir / "edited_withmeta.png"
        self.assertTrue(transplant(source, target, out))
        self.assertEqual(extract_from_file(out).raw_parameters, raw)
        with Image.open(out) as img:
            self.assertEqual((img.size, img.mode), ((6, 5), "RGBA"))
            self.assertEqual(img.getpixel((0, 0)), (0, 255, 0, 128))
            self.assertEqual(tuple(round(v) for v in img.info["dpi"]), (144, 144))

    def test_transplant_bakes_jpeg_orientation_into_png_pixels(self):
        from core.image_metadata import transplant
        source = self.dir / "source.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "a\nSteps: 20, Seed: 1")
        Image.new("RGB", (4, 4)).save(source, pnginfo=info)
        target = self.dir / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # 90° 회전 표시
        Image.new("RGB", (40, 20), "white").save(target, exif=exif)
        out = self.dir / "rotated_withmeta.png"
        self.assertTrue(transplant(source, target, out))
        with Image.open(out) as img:
            self.assertEqual(img.size, (20, 40))

    def test_suggested_output_never_overwrites_an_existing_file(self):
        from core.image_metadata import suggest_transplant_output
        target = self.dir / "photo.jpg"
        self.assertEqual(Path(suggest_transplant_output(target)).name, "photo_withmeta.png")
        (self.dir / "photo_withmeta.png").write_bytes(b"x")
        self.assertEqual(Path(suggest_transplant_output(target)).name, "photo_withmeta_2.png")

    def test_source_without_metadata_is_refused(self):
        from core.image_metadata import transplant
        source = self.dir / "plain.png"
        Image.new("RGB", (4, 4)).save(source)
        out = self.dir / "out.png"
        self.assertFalse(transplant(source, source, out))
        self.assertFalse(out.exists())

    def test_only_rgb_icc_profiles_follow_the_pixels_into_the_png(self):
        # CMYK TIFF 의 CMYK 프로파일을 RGB PNG 에 붙이면 색이 틀어진다 — RGB 계열 원본만 옮긴다
        from PIL import ImageCms
        from core.image_metadata import transplant
        source = self.dir / "source.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "a\nSteps: 20, Seed: 1")
        Image.new("RGB", (4, 4)).save(source, pnginfo=info)
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        cases = (
            ("rgb.tif", Image.new("RGB", (6, 4), "white"), True),
            ("cmyk.tif", Image.new("CMYK", (6, 4), (0, 0, 0, 0)), False),
            ("gray.tif", Image.new("L", (6, 4), 128), False),
        )
        for name, image, keeps in cases:
            with self.subTest(name=name):
                target = self.dir / name
                image.save(target, icc_profile=srgb)
                out = self.dir / f"{name}.png"
                self.assertTrue(transplant(source, target, out))
                with Image.open(out) as img:
                    self.assertEqual(img.mode, "RGB")
                    self.assertEqual(bool(img.info.get("icc_profile")), keeps)


if __name__ == "__main__":
    unittest.main()
