"""프롬프트 주석은 어느 길로 가도 백엔드 요청에 없다 — 공용 빌더, Forge(가짜 requests.post), ComfyUI(가짜 큐 —
컴파일된 그래프가 곧 Comfy PNG 메타데이터), Krea2 러너, 채팅 지연 훅, 프롬프트 기록.

같은 입력이면 Forge 요청 JSON 과 Comfy 그래프의 텍스트가 같다(주석을 백엔드가 갈라지기 전에 떼고, 두 입구에서 한 번 더).
GPU·네트워크 없음.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from backends import BackendType
from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core.prompt_comments import comment_spans, strip_request_comments
from tests.test_chat_generation_snapshot import ReadOnlyWidget, SnapshotHost
from tests.test_webui_forge_options import _ForgeCase, ok

MAIN = "1girl, // 메인 메모\nsolo, /* 끈 태그: red hair, */ blue eyes"
MAIN_SENT = "1girl,\nsolo, blue eyes"
NEG = "lowres /* x */, bad hands // n"
NEG_SENT = "lowres, bad hands"


def _png_b64(size=(16, 16)) -> str:
    out = io.BytesIO()
    Image.new("RGB", size, "gray").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _commented_payload() -> dict:
    return {
        "prompt": MAIN, "negative_prompt": NEG, "steps": 4, "seed": 5, "cfg_scale": 5.0,
        "enable_hr": True, "hr_scale": 1.5, "denoising_strength": 0.4,
        "hr_prompt": "detailed // hr 메모", "hr_negative_prompt": "/* off */ blurry",
        "alwayson_scripts": {
            "ADetailer": {"args": [True, False,
                                   {"ad_model": "face_yolov8n.pt", "ad_prompt": "smile // a", "ad_negative_prompt": ""},
                                   {"ad_model": "hand_yolov8n.pt", "ad_prompt": "hands", "ad_negative_prompt": "// n"}]},
        },
    }


def _strings(value):
    """중첩 dict/list 의 모든 문자열."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


class _CommentFree:
    def assert_no_comments(self, value, *, where=""):
        for text in _strings(value):
            for marker in ("메모", "// a", "// n", "/* x */", "/* off */", "끈 태그"):
                self.assertNotIn(marker, text, f"{where}: 주석이 남았다 — {text!r}")


# ── 공용 빌더(Forge·ComfyUI 공통) ───────────────────────────────────────────────

class _BuilderHost(SnapshotHost):
    def __init__(self, **values):
        super().__init__()
        self.random_res_check = ReadOnlyWidget(False)
        self.postprocess = {}
        for name, value in values.items():
            setattr(self, name, ReadOnlyWidget(value))


class SharedBuilderTests(_CommentFree, unittest.TestCase):
    def setUp(self):
        for patcher in (
            mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI),
            mock.patch("backends.get_backend", return_value=SimpleNamespace(api_url="http://127.0.0.1:7860")),
            mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def host(self, **values):
        return _BuilderHost(total_prompt_display=MAIN, neg_prompt_text=NEG,
                            hires_prompt_text="// 메모만", hires_neg_prompt_text="soft /* x */ blur", **values)

    def test_generation_strips_before_wildcards_and_hooks_and_again_before_the_lora_stack(self):
        seen = []

        def pipeline(text):
            seen.append(text)
            return text + (", $$extra$$" if text == MAIN_SENT else "")

        with mock.patch("ui.generator_generation.resolve_file_wildcards", side_effect=lambda t: t), \
                mock.patch("ui.generator_generation.process_wildcards", side_effect=lambda t: t), \
                mock.patch("core.standard_hooks.run_pipeline_on_text", side_effect=pipeline):
            payload, error = self.host()._build_generation_payload()
        self.assertIsNone(error)
        self.assertEqual(seen, [MAIN_SENT, NEG_SENT])            # 훅은 주석을 뗀 글자만 본다
        self.assertEqual(payload["prompt"], MAIN_SENT + ", $$extra$$, <lora:style:0.70>")
        self.assertEqual(payload["negative_prompt"], NEG_SENT)
        self.assertNotIn("hr_prompt", payload)                     # 주석만 있는 Hires 칸 = 빈 칸
        self.assertEqual(payload["hr_negative_prompt"], "soft blur")

    def test_chat_snapshot_and_prompt_override_are_stripped_too(self):
        model, payload = self.host()._chat_generation_snapshot("portrait // 채팅 메모")
        self.assertEqual(payload["prompt"], "portrait, <lora:style:0.70>")   # LoRA 가 주석에 먹히지 않는다
        self.assertEqual(payload["negative_prompt"], NEG_SENT)
        self.assert_no_comments(payload, where="chat snapshot")

    def test_comment_free_prompts_are_byte_identical(self):
        host = _BuilderHost(total_prompt_display="http://x.com/a, a//b, (c:1.2)", neg_prompt_text="  low  ",
                            hires_prompt_text=" hr ", hires_neg_prompt_text="")
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(payload["prompt"], "http://x.com/a, a//b, (c:1.2), <lora:style:0.70>")
        self.assertEqual(payload["negative_prompt"], "low")
        self.assertEqual(payload["hr_prompt"], "hr")

    def test_prompt_history_records_what_was_sent(self):
        host = SimpleNamespace(total_prompt_display=ReadOnlyWidget(MAIN), neg_prompt_text=ReadOnlyWidget(NEG))
        from ui.generator_generation import GenerationMixin
        with mock.patch("utils.prompt_history.add_entry") as add_entry:
            GenerationMixin._record_prompt_history(host)
        add_entry.assert_called_once_with(MAIN_SENT, NEG_SENT)


# ── Forge ─────────────────────────────────────────────────────────────────────

class ForgeRequestTests(_CommentFree, _ForgeCase):
    def test_txt2img_and_img2img_requests_carry_no_comments(self):
        for method in ("txt2img", "img2img"):
            with self.subTest(method=method):
                payload = _commented_payload()
                if method == "img2img":
                    payload["init_images"] = [_png_b64()]
                before = copy.deepcopy(payload)
                result = self.run_main(payload, [ok()], method=method)
                self.assertTrue(result.success, result.error)
                sent = self.sent[-1]
                self.assertEqual(payload, before, "호출자 payload 를 고쳤다")
                self.assertEqual((sent["prompt"], sent["negative_prompt"]), (MAIN_SENT, NEG_SENT))
                self.assertEqual((sent["hr_prompt"], sent["hr_negative_prompt"]), ("detailed", "blurry"))
                slots = sent["alwayson_scripts"]["ADetailer"]["args"]
                self.assertEqual([(s["ad_prompt"], s["ad_negative_prompt"]) for s in slots[2:]],
                                 [("smile", ""), ("hands", "")])
                self.assert_no_comments(sent, where=method)

    def test_comment_free_requests_are_byte_identical(self):
        payload = {"prompt": "http://x.com/a, a//b, ~/hair/~", "negative_prompt": "low", "steps": 4,
                   "alwayson_scripts": {"ADetailer": {"args": [True, False, {"ad_prompt": "face"}]}}}
        self.assertIs(strip_request_comments(payload), payload)
        self.run_main(copy.deepcopy(payload), [ok()])
        self.assertEqual({k: v for k, v in self.sent[-1].items() if k not in ("force_task_id", "save_images")},
                         {k: v for k, v in payload.items() if k != "save_images"})

    def test_standalone_adetailer_sam3_and_refine(self):
        self.run_aux("adetailer", {"ad_prompt": "smile // a", "ad_negative": "/* x */ bad"}, [ok()])
        sent = self.sent[-1]
        self.assertEqual((sent["prompt"], sent["negative_prompt"]), ("smile", "bad"))
        self.assertEqual(sent["alwayson_scripts"]["ADetailer"]["args"][2]["ad_prompt"], "smile")

        self.run_aux("sam3", {"sam3_prompt": "face // 검출", "sam3_inpaint_prompt": "// 메모만",
                              "prompt": "1girl // 메인", "negative_prompt": "low // n"}, [ok()])
        sent = self.sent[-1]
        state = sent["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual((state["sam3_prompt"], state["sam3_inpaint_prompt"]), ("face", "1girl"))
        self.assertEqual((sent["prompt"], sent["negative_prompt"]), ("1girl", "low"))

        self.run_aux("refine", {"target": "shirt // t", "replacement": "dress", "main_prompt": MAIN + ", shirt",
                                "main_negative": NEG}, [ok()])
        sent = self.sent[-1]
        self.assertEqual(sent["prompt"], "1girl, solo, blue eyes, dress")
        self.assertEqual(sent["alwayson_scripts"]["SAM3 Mask"]["args"][0]["sam3_prompt"], "shirt")
        self.assert_no_comments(self.sent, where="aux")


# ── ComfyUI ───────────────────────────────────────────────────────────────────

class _ComfyHarness:
    def comfy_backend(self):
        from tests.test_comfy_workflow_compiler import _capabilities
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="", img2img_workflow_path="")
        backend._preflight_bundled_node_pack = mock.Mock()

        def fake_fetch(_self=backend):
            data = _capabilities()
            _self._object_info_cache.store(_self.api_url, data)
            return data

        backend.get_object_info = fake_fetch
        backend._queue_and_wait = mock.Mock(side_effect=lambda *a, **k: GenerationResult(
            success=True, image_data=b"img", info={"prompt_id": "p", "node_outputs": {}}))
        backend._upload_image = mock.Mock(return_value="input_new.png")
        return backend

    @staticmethod
    def graph(backend) -> dict:
        return backend._queue_and_wait.call_args.args[0]

    @staticmethod
    def texts(graph, class_type, key="text"):
        return [node["inputs"][key] for node in graph.values() if node["class_type"] == class_type]


class ComfyRequestTests(_CommentFree, _ComfyHarness, unittest.TestCase):
    def test_txt2img_graph_and_saved_context_carry_no_comments(self):
        backend = self.comfy_backend()
        payload = _commented_payload()
        before = copy.deepcopy(payload)
        result = backend.txt2img("checkpoint.safetensors", payload)
        self.assertTrue(result.success, result.error)
        self.assertEqual(payload, before)
        graph = self.graph(backend)
        self.assertEqual(self.texts(graph, "CLIPTextEncode")[:2], [MAIN_SENT, NEG_SENT])
        hires = [n["inputs"] for n in graph.values() if n["class_type"] == "ForgeNeoHiresFix"][0]
        self.assertEqual((hires["positive_text"], hires["base_positive_text"]), ("detailed", MAIN_SENT))
        settings = [json.loads(s) for s in self.texts(graph, "ForgeNeoADetailer", "settings_json")]
        self.assertEqual([s["ad_prompt"] for s in settings], ["smile", "hands"])
        self.assert_no_comments(graph, where="comfy graph")
        self.assert_no_comments(backend._last_generation_context, where="saved context")

    def test_img2img_graph_carries_no_comments(self):
        backend = self.comfy_backend()
        payload = {**_commented_payload(), "init_images": [_png_b64()], "enable_hr": False}
        result = backend.img2img("checkpoint.safetensors", payload)
        self.assertTrue(result.success, result.error)
        graph = self.graph(backend)
        self.assertIn(MAIN_SENT, self.texts(graph, "CLIPTextEncode"))
        self.assert_no_comments(graph, where="comfy i2i graph")

    def test_standalone_adetailer_and_sam3_mask_only(self):
        backend = self.comfy_backend()
        backend._last_generation_context = {"model_name": "checkpoint.safetensors",
                                            "payload": {"prompt": MAIN, "negative_prompt": NEG, "alwayson_scripts": {}}}
        backend.adetailer(_png_b64(), {"ad_prompt": "smile // a", "ad_negative": "/* x */ bad"})
        settings = [json.loads(s) for s in self.texts(self.graph(backend), "ForgeNeoADetailer", "settings_json")]
        self.assertEqual([(s["ad_prompt"], s["ad_negative_prompt"]) for s in settings], [("smile", "bad")])
        self.assert_no_comments(self.graph(backend), where="comfy adetailer")

        backend.sam3(_png_b64(), {"sam3_mode": "Mask only", "sam3_prompt": "face // 검출"})
        self.assertEqual(self.texts(self.graph(backend), "ForgeNeoSAM3Mask", "prompt"), ["face"])

    def test_generation_api_workflows_are_stripped_before_mapping(self):
        backend = self.comfy_backend()
        seen = []

        def mapped(workflow, mode, model, payload, cancel_check=None):
            seen.append(copy.deepcopy(payload))
            return workflow, []

        backend._map_external_workflow = mapped
        backend.run_workflow = mock.Mock(return_value=GenerationResult(success=True, image_data=b"x", info={}))
        result = backend.generate_workflow("t2i", {"1": {"class_type": "KSampler", "inputs": {}}}, "m",
                                           {"prompt": MAIN, "negative_prompt": NEG, "seed": 1})
        self.assertTrue(result.success, result.error)
        self.assertEqual((seen[0]["prompt"], seen[0]["negative_prompt"]), (MAIN_SENT, NEG_SENT))


class ForgeComfyParityTests(_CommentFree, _ComfyHarness, _ForgeCase):
    def test_same_prompt_text_reaches_both_backends(self):
        payload = {"prompt": MAIN, "negative_prompt": NEG, "steps": 4, "seed": 5}
        self.run_main(copy.deepcopy(payload), [ok()])
        forge = self.sent[-1]
        backend = self.comfy_backend()
        backend.txt2img("checkpoint.safetensors", copy.deepcopy(payload))
        comfy = self.texts(self.graph(backend), "CLIPTextEncode")
        self.assertEqual(comfy[:2], [forge["prompt"], forge["negative_prompt"]])
        self.assertEqual(comment_spans(forge["prompt"]), [])


# ── Krea2 · 채팅 ──────────────────────────────────────────────────────────────

class Krea2AndChatTests(_CommentFree, unittest.TestCase):
    def test_krea2_runner_strips_its_prompt(self):
        from core.krea2_generation import run_krea2_generation
        from tests.test_krea2_generation import _FakeComfy
        backend = _FakeComfy()
        result = run_krea2_generation(backend, "t2i", {"prompt": "portrait // 메모\nstudio light", "seed": 1,
                                                       "width": 1024, "height": 1024})
        self.assertTrue(result.success, result.error)
        self.assertEqual(backend.workflow["4"]["inputs"]["text"], "portrait\nstudio light")

    def test_chat_deferred_hooks_see_and_return_stripped_text(self):
        from core.chat_generation import prepare_prompt_payload
        seen = []

        def pipeline(text):
            seen.append(text)
            return text + ", tail // 훅이 넣은 메모"

        with mock.patch("core.standard_hooks.run_pipeline_on_text", side_effect=pipeline):
            out = prepare_prompt_payload({"prompt": MAIN, "negative_prompt": NEG,
                                          "_chat_deferred_prompt": {"wildcards": False}})
        self.assertEqual(seen, [MAIN_SENT, NEG_SENT])
        self.assertEqual((out["prompt"], out["negative_prompt"]), (MAIN_SENT + ", tail", NEG_SENT + ", tail"))
        self.assertNotIn("_chat_deferred_prompt", out)


if __name__ == "__main__":
    unittest.main()
