"""메타데이터 액션 동등성 — 네거티브 없는 A1111 PNG 에서 모든 액션이 같은 결과를 내는지.

예전에는 Gallery 'T2I에서 사용'과 PNG Info '즉시 생성'이 raw 를 '\\nNegative prompt: '로
다시 split 해서, 네거티브 줄이 없는 이미지는 'Steps: …' 줄 전체가 프롬프트로 들어갔다.
이제 모든 액션이 core.image_metadata 파싱 결과(core.metadata_actions)를 쓴다.
작은 임시 파일과 가짜 화면/생성 경계만 쓴다 — 실제 생성·서버·GPU 없음.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, PngImagePlugin

from core.image_metadata import read_applicable_prompts
from core.metadata_actions import (
    NO_PROMPT_MESSAGE,
    READ_FAILED_MESSAGE,
    apply_block_reason,
    build_generation_item,
    prompt_transfer,
    resolve_action_metadata,
)
from core.metadata_search import SearchTextCache, metadata_search_text, search_texts_for
from ui.generator_main import GeneratorMainUI
from ui.image_metadata_actions import transplant_metadata_action
from ui.model_download_actions import ModelDownloadActionsMixin
from ui.vue_bridge import VueBridge

NEGATIVE_FREE = "a cat, 1girl\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 123, Size: 832x1216"
# Forge Neo 식 파라미터 줄 — 점 키(Anima 3.8B …), 따옴표 없는 JSON 값, 뒤따르는 꼬리 줄.
# 예전 fullmatch 파서는 이 줄을 파라미터로 못 알아봐 Steps 줄이 프롬프트로 샜다.
FORGE_NEO_LIKE = (
    'a cat, 1girl\nSteps: 30, Sampler: Euler, CFG scale: 7, Seed: 123, Size: 832x1216, '
    'Hashes: {"vae": "a", "model": "b"}, Anima 3.8B adapter: x.safetensors, Version: neo-2.29\n'
    'Template: a cat, 1girl\nNegative Template:')


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _Widget:
    def __init__(self, value):
        self.value = str(value)

    def text(self):
        return self.value

    toPlainText = text
    currentText = text

    def setText(self, value):
        self.value = str(value)

    setPlainText = setText
    setCurrentText = setText


class _Bridge:
    getImageExif = VueBridge.getImageExif
    saveImagePrompt = VueBridge.saveImagePrompt
    renameFile = VueBridge.renameFile
    requestImageSearchTexts = VueBridge.requestImageSearchTexts

    def __init__(self):
        self.showNotification = _Signal()
        self.imageSearchTextsReady = _Signal()


class _Host(ModelDownloadActionsMixin):
    _handle_vue_action = GeneratorMainUI._handle_vue_action
    _handle_immediate_generation_from_raw = GeneratorMainUI._handle_immediate_generation_from_raw
    _build_queue_payload_from_exif = GeneratorMainUI._build_queue_payload_from_exif
    _apply_payload_to_ui = GeneratorMainUI._apply_payload_to_ui

    def __init__(self):
        self.vue_bridge = _Bridge()
        self.started = []
        self.transferred = []
        self.queued = []
        self.queue_panel = SimpleNamespace(add_single_item=self.queued.append)
        for name, value in {
            "main_prompt_text": "existing positive", "total_prompt_display": "existing positive",
            "neg_prompt_text": "existing negative", "steps_input": 30, "cfg_input": 5,
            "seed_input": -1, "width_input": 1024, "height_input": 1024,
            "sampler_combo": "old sampler", "scheduler_combo": "old scheduler",
        }.items():
            setattr(self, name, _Widget(value))

    def _handle_chat_action(self, _action, _payload):
        return False

    def _handle_creator_action(self, _action, _payload):
        return False

    def handle_prompt_only_transfer(self, prompt, negative):
        self.transferred.append((prompt, negative))

    def start_generation(self):
        self.started.append({
            "prompt": self.total_prompt_display.toPlainText(), "negative": self.neg_prompt_text.toPlainText(),
            "size": (self.width_input.text(), self.height_input.text()),
            "sampler": self.sampler_combo.currentText(), "seed": self.seed_input.text(),
            "steps": self.steps_input.text(),
        })


class MetadataActionParityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.path = self.dir / "negative_free.png"
        self.write_png(self.path, NEGATIVE_FREE)
        self.host = _Host()
        self.info = json.loads(self.host.vue_bridge.getImageExif(str(self.path)))

    @staticmethod
    def write_png(path, parameters=None, **extra):
        info = PngImagePlugin.PngInfo()
        if parameters is not None:
            info.add_text("parameters", parameters)
        for key, value in extra.items():
            info.add_text(key, value)
        Image.new("RGB", (8, 8), "blue").save(path, pnginfo=info)

    def test_all_actions_agree_on_prompt_negative_and_size(self):
        host = self.host
        host._handle_vue_action("gallery_send_exif_to_t2i", {"path": str(self.path), "metadata": self.info, "exif": self.info["raw"]})
        host._handle_vue_action("gallery_send_exif_to_t2i", {"path": str(self.path), "exif": self.info["raw"]})  # 즐겨찾기/구 페이로드
        host._handle_vue_action("pnginfo_send_prompt", self.info)
        host._handle_vue_action("pull_prompt_from_image", {"path": str(self.path)})
        self.assertEqual(host.transferred, [("a cat, 1girl", "")] * 4)

        host._handle_vue_action("pnginfo_generate", self.info)
        host._handle_vue_action("add_image_to_queue", {"path": str(self.path)})
        self.assertEqual(host.started, [{
            "prompt": "a cat, 1girl", "negative": "", "size": ("832", "1216"),
            "sampler": "Euler", "seed": "123", "steps": "20",
        }])
        self.assertEqual(host.queued, [{
            "prompt": "a cat, 1girl", "negative_prompt": "", "sampler_name": "Euler",
            "scheduler": "old scheduler", "steps": 20, "cfg_scale": 7.0, "seed": -1,
            "width": 832, "height": 1216,
        }])

    def test_gallery_sends_the_prompt_edited_in_the_large_view(self):
        edited = dict(self.info, prompt="edited prompt")
        self.host._handle_vue_action("gallery_send_exif_to_t2i", {"path": str(self.path), "metadata": edited})
        self.assertEqual(self.host.transferred, [("edited prompt", "")])

    def test_legacy_raw_only_payloads_use_the_core_parser(self):
        self.host._handle_vue_action("gallery_send_exif_to_t2i", {"exif": NEGATIVE_FREE})
        self.host._handle_immediate_generation_from_raw(NEGATIVE_FREE)
        self.assertEqual(self.host.transferred, [("a cat, 1girl", "")])
        self.assertEqual(self.host.started[0]["prompt"], "a cat, 1girl")
        self.assertEqual(self.host.started[0]["size"], ("832", "1216"))

    def test_images_without_prompt_are_refused_with_a_notice(self):
        empty = self.dir / "params_only.png"
        self.write_png(empty, "Steps: 20, Sampler: Euler, CFG scale: 7")
        info = json.loads(self.host.vue_bridge.getImageExif(str(empty)))
        self.assertEqual((info["prompt"], info["negative"]), ("", ""))
        self.assertTrue(info["can_apply"])  # 파라미터는 읽혔다 — 프롬프트 전송만 막아야 한다
        for action, payload in (
            ("gallery_send_exif_to_t2i", {"path": str(empty), "metadata": info}),
            ("pnginfo_send_prompt", info),
            ("pull_prompt_from_image", {"path": str(empty)}),
            ("pnginfo_generate", info),
            ("add_image_to_queue", {"path": str(empty)}),
        ):
            with self.subTest(action=action):
                host = _Host()
                host._handle_vue_action(action, payload)
                self.assertEqual((host.transferred, host.started, host.queued), ([], [], []))
                self.assertTrue(host.vue_bridge.showNotification.calls)
                # T2I 프롬프트 칸은 그대로다
                self.assertEqual(host.main_prompt_text.text(), "existing positive")
                self.assertEqual(host.neg_prompt_text.text(), "existing negative")
        host = _Host()
        host._handle_vue_action("pnginfo_send_prompt", info)
        self.assertEqual(host.vue_bridge.showNotification.calls, [("warning", NO_PROMPT_MESSAGE)])

    def test_forge_neo_line_with_dotted_keys_and_json_values_agrees_everywhere(self):
        path = self.dir / "forge_neo.png"
        self.write_png(path, FORGE_NEO_LIKE)
        host = _Host()
        info = json.loads(host.vue_bridge.getImageExif(str(path)))
        self.assertEqual((info["prompt"], info["negative"]), ("a cat, 1girl", ""))
        self.assertEqual(info["parameters"]["Anima 3.8B adapter"], "x.safetensors")
        self.assertEqual(info["parameters"]["Hashes"], '{"vae": "a", "model": "b"}')
        host._handle_vue_action("gallery_send_exif_to_t2i", {"path": str(path), "metadata": info, "exif": info["raw"]})
        host._handle_vue_action("gallery_send_exif_to_t2i", {"path": str(path), "exif": info["raw"]})
        host._handle_vue_action("pnginfo_send_prompt", info)
        host._handle_vue_action("pull_prompt_from_image", {"path": str(path)})
        self.assertEqual(host.transferred, [("a cat, 1girl", "")] * 4)
        host._handle_vue_action("pnginfo_generate", info)
        host._handle_vue_action("add_image_to_queue", {"path": str(path)})
        self.assertEqual(host.started, [{
            "prompt": "a cat, 1girl", "negative": "", "size": ("832", "1216"),
            "sampler": "Euler", "seed": "123", "steps": "30",
        }])
        self.assertEqual(
            {key: host.queued[0][key] for key in ("prompt", "negative_prompt", "sampler_name", "steps", "width", "height")},
            {"prompt": "a cat, 1girl", "negative_prompt": "", "sampler_name": "Euler", "steps": 30,
             "width": 832, "height": 1216})
        # SAM3/ADetailer 배치 워커도 같은 결과
        self.assertEqual(read_applicable_prompts(path), ("a cat, 1girl", "", ""))


class MetadataActionRuleTests(unittest.TestCase):
    def test_resolve_prefers_metadata_then_path_then_raw(self):
        reads = []

        def reader(path):
            reads.append(path)
            return json.dumps({"source": "webui", "prompt": "from file"})

        meta = {"source": "webui", "prompt": "from view"}
        self.assertIs(resolve_action_metadata({"metadata": meta, "path": "x"}, reader, is_file=lambda _p: True), meta)
        self.assertEqual(resolve_action_metadata({"path": "x"}, reader, is_file=lambda _p: True)["prompt"], "from file")
        self.assertEqual(resolve_action_metadata({"raw": NEGATIVE_FREE}, reader)["prompt"], "a cat, 1girl")
        self.assertEqual(resolve_action_metadata({"path": "x"}, lambda _p: "not json", is_file=lambda _p: True),
                         {"error": READ_FAILED_MESSAGE})
        self.assertIsNone(resolve_action_metadata({}, reader))
        self.assertEqual(reads, ["x"])

    def test_block_reasons(self):
        self.assertEqual(apply_block_reason(None), READ_FAILED_MESSAGE)
        self.assertEqual(apply_block_reason({"source": "comfyui", "can_apply": False}, comfy_message="c"), "c")
        self.assertTrue(apply_block_reason({"source": "webui", "can_apply": False}))
        self.assertEqual(apply_block_reason({"source": "webui", "can_apply": True}), "")

    def test_prompt_transfer_refuses_params_only_and_passes_prompt_only_payloads(self):
        self.assertEqual(prompt_transfer({"source": "webui", "can_apply": True, "prompt": "", "negative": ""}),
                         ("", "", NO_PROMPT_MESSAGE))
        self.assertEqual(prompt_transfer({"source": "webui", "can_apply": True, "prompt": "", "negative": "n"}),
                         ("", "n", ""))
        self.assertEqual(prompt_transfer({"source": "comfyui", "can_apply": False, "prompt": "p"}, comfy_message="c"),
                         ("", "", "c"))
        self.assertEqual(prompt_transfer({"error": "x"}), ("", "", READ_FAILED_MESSAGE))
        self.assertEqual(prompt_transfer(None), ("", "", READ_FAILED_MESSAGE))
        # 이벤트 생성 탭은 source 없이 프롬프트만 보낸다
        self.assertEqual(prompt_transfer({"prompt": "step prompt", "negative": ""}), ("step prompt", "", ""))

    def test_build_item_uses_parsed_parameters_and_defaults(self):
        defaults = {"sampler_name": "d", "scheduler": "s", "steps": "30", "cfg_scale": "5", "width": "1024", "height": "768"}
        item = build_generation_item({"source": "webui", "prompt": "p", "negative": "n", "can_apply": True,
                                      "parameters": {"Steps": 12, "Seed": 9, "Size": "640x480", "Schedule type": "Karras"}},
                                     defaults, preserve_seed=True)
        self.assertEqual(item, {"prompt": "p", "negative_prompt": "n", "sampler_name": "d", "scheduler": "Karras",
                                "steps": 12, "cfg_scale": 5.0, "seed": 9, "width": 640, "height": 480})
        self.assertIsNone(build_generation_item({"source": "webui", "prompt": "", "can_apply": True}, defaults))


class MetadataTransplantActionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.source = self.dir / "source.png"
        MetadataActionParityTests.write_png(self.source, "a cat\nNegative prompt: bad\nSteps: 20, Seed: 1")
        self.target = self.dir / "edited.jpg"
        Image.new("RGB", (12, 10), "red").save(self.target)
        self.host = SimpleNamespace(vue_bridge=_Bridge())

    def test_transplant_writes_a_new_png_next_to_the_target(self):
        from core.image_metadata import extract_from_file
        suggested = []
        out = transplant_metadata_action(
            self.host, {"path": str(self.source)}, lambda p: p,
            pick_target=lambda _start: str(self.target),
            pick_output=lambda s: suggested.append(s) or s)
        self.assertEqual(Path(suggested[0]).name, "edited_withmeta.png")
        self.assertEqual(out, suggested[0])
        meta = extract_from_file(out)
        self.assertEqual((meta.prompt, meta.negative_prompt), ("a cat", "bad"))
        self.assertEqual(self.host.vue_bridge.showNotification.calls[-1][0], "success")

    def test_transplant_refuses_missing_metadata_and_overwriting_the_source(self):
        plain = self.dir / "plain.png"
        Image.new("RGB", (4, 4)).save(plain)
        self.assertIsNone(transplant_metadata_action(self.host, {"path": str(plain)}, lambda p: p,
                                                     pick_target=lambda _s: self.fail("no dialog"),
                                                     pick_output=lambda _s: self.fail("no dialog")))
        before = self.source.read_bytes()
        self.assertIsNone(transplant_metadata_action(self.host, {"path": str(self.source)}, lambda p: p,
                                                     pick_target=lambda _s: str(self.target),
                                                     pick_output=lambda _s: str(self.source)))
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual([c[0] for c in self.host.vue_bridge.showNotification.calls], ["warning", "error"])

    def test_cancelled_dialogs_do_nothing(self):
        self.assertIsNone(transplant_metadata_action(self.host, {"path": str(self.source)}, lambda p: p,
                                                     pick_target=lambda _s: "", pick_output=lambda _s: ""))
        self.assertEqual(self.host.vue_bridge.showNotification.calls, [])

    def test_every_extension_the_picker_offers_is_accepted(self):
        # 필터가 보여 주는 확장자(.tif 포함)를 고르면 실제로 이식된다 — 필터와 검증이 같은 목록
        from core.image_metadata import extract_from_file
        from ui.image_metadata_actions import TRANSPLANT_TARGET_EXTS, TRANSPLANT_TARGET_FILTER
        offered = TRANSPLANT_TARGET_FILTER.split(";;")[0]
        for ext in TRANSPLANT_TARGET_EXTS:
            self.assertIn(f"*{ext}", offered)
        for ext in (".tif", ".tiff", ".bmp"):
            with self.subTest(ext=ext):
                target = self.dir / f"scan_{ext[1:]}{ext}"
                Image.new("RGB", (6, 4), "green").save(target)
                out = transplant_metadata_action(
                    self.host, {"path": str(self.source)}, lambda p: p,
                    pick_target=lambda _s, t=target: str(t), pick_output=lambda s: s)
                self.assertIsNotNone(out, self.host.vue_bridge.showNotification.calls)
                self.assertEqual(Path(out).name, f"scan_{ext[1:]}_withmeta.png")
                self.assertEqual(extract_from_file(out).prompt, "a cat")
        # 목록 밖 확장자는 여전히 거부한다
        other = self.dir / "clip.mp4"
        other.write_bytes(b"x")
        self.assertIsNone(transplant_metadata_action(
            self.host, {"path": str(self.source)}, lambda p: p,
            pick_target=lambda _s: str(other), pick_output=lambda _s: self.fail("no dialog")))
        self.assertEqual(self.host.vue_bridge.showNotification.calls[-1], ("error", "대상 이미지를 열 수 없습니다."))


class GalleryBridgeSlotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.bridge = _Bridge()

    def test_save_prompt_keeps_the_parameter_tail_and_returns_fresh_metadata(self):
        path = self.dir / "a.png"
        tail = 'Steps: 20, Lora hashes: "a: 1, b: 2"\nTemplate: a {red|blue} cat, sitting'
        MetadataActionParityTests.write_png(path, "old\n" + tail)
        result = json.loads(self.bridge.saveImagePrompt(str(path), "new prompt", "new neg"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["info"]["prompt"], "new prompt")
        self.assertEqual(result["info"]["negative"], "new neg")
        self.assertEqual(result["info"]["params_line"], tail)
        self.assertEqual(result["info"]["raw"], "new prompt\nNegative prompt: new neg\n" + tail)

    def test_save_prompt_refuses_comfy_graph_and_non_png(self):
        from tests.test_image_metadata import graph_fixture
        comfy = self.dir / "comfy.png"
        MetadataActionParityTests.write_png(comfy, None, prompt=json.dumps(graph_fixture()))
        before = comfy.read_bytes()
        self.assertIn("error", json.loads(self.bridge.saveImagePrompt(str(comfy), "x", "")))
        self.assertEqual(comfy.read_bytes(), before)
        jpeg = self.dir / "a.jpg"
        Image.new("RGB", (4, 4)).save(jpeg)
        self.assertIn("error", json.loads(self.bridge.saveImagePrompt(str(jpeg), "x", "")))

    def test_save_prompt_keeps_apng_animation(self):
        # 예전엔 Pillow 재저장이 첫 프레임만 남기고도 '저장 완료'를 알렸다
        path = self.dir / "anim.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "old\nSteps: 20, Seed: 1")
        frames = [Image.new("RGB", (4, 4), c) for c in ("red", "blue")]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0, pnginfo=info)
        result = json.loads(self.bridge.saveImagePrompt(str(path), "new", ""))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["info"]["prompt"], "new")
        with Image.open(path) as img:
            self.assertEqual(img.n_frames, 2)
            img.seek(1)
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (0, 0, 255))

    def test_save_prompt_sees_metadata_written_after_the_pixels(self):
        from tests.test_image_metadata import _insert_png_chunk_before_iend, graph_fixture
        tail = "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 8x8"
        path = self.dir / "trailing.png"
        MetadataActionParityTests.write_png(path)
        _insert_png_chunk_before_iend(
            path, b"tEXt", b"parameters\x00" + ("tail cat\nNegative prompt: tail neg\n" + tail).encode("latin-1"))
        result = json.loads(self.bridge.saveImagePrompt(str(path), "edited cat", "edited neg"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["info"]["prompt"], "edited cat")
        self.assertEqual(result["info"]["params_line"], tail)   # 파라미터 꼬리를 잃지 않는다
        comfy = self.dir / "trailing_comfy.png"
        MetadataActionParityTests.write_png(comfy)
        _insert_png_chunk_before_iend(comfy, b"tEXt", b"prompt\x00" + json.dumps(graph_fixture()).encode("latin-1"))
        before = comfy.read_bytes()
        result = json.loads(self.bridge.saveImagePrompt(str(comfy), "x", ""))
        self.assertEqual(result.get("error"), "ComfyUI 워크플로 메타데이터는 수정하지 않습니다")
        self.assertEqual(comfy.read_bytes(), before)

    def test_rename_returns_the_real_new_path_for_all_gallery_media(self):
        video = self.dir / "clip.mp4"
        video.write_bytes(b"not really a video")
        result = json.loads(self.bridge.renameFile(str(video), "renamed clip"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["new_path"]).name, "renamed clip.mp4")
        self.assertTrue(Path(result["new_path"]).is_file())

        image = self.dir / "char.png"
        Image.new("RGB", (4, 4)).save(image)
        result = json.loads(self.bridge.renameFile(str(image), "char.v2"))
        self.assertEqual(Path(result["new_path"]).name, "char.v2.png")
        result = json.loads(self.bridge.renameFile(result["new_path"], "evil.:$DATA"))
        self.assertTrue(result["ok"], result)
        self.assertNotIn(":", Path(result["new_path"]).name)
        self.assertTrue(Path(result["new_path"]).is_file())

    def test_search_texts_are_answered_off_thread_with_the_token(self):
        path = self.dir / "s.png"
        MetadataActionParityTests.write_png(path, "Blue Bird\nNegative prompt: Lowres\nSteps: 20, Seed: 1")
        self.bridge.requestImageSearchTexts(json.dumps({"token": "t1", "paths": [str(path), str(self.dir / "gone.png")]}))
        deadline = time.monotonic() + 5
        while not self.bridge.imageSearchTextsReady.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        payload = json.loads(self.bridge.imageSearchTextsReady.calls[0][0])
        self.assertEqual(payload["token"], "t1")
        self.assertIn("blue bird", payload["texts"][str(path)])
        self.assertIn("lowres", payload["texts"][str(path)])
        self.assertEqual(payload["texts"][str(self.dir / "gone.png")], "")
        # 잘못된 요청도 같은 채널로 응답한다(프런트가 기다리다 멈추지 않게)
        self.bridge.requestImageSearchTexts("not json")
        deadline = time.monotonic() + 5
        while len(self.bridge.imageSearchTextsReady.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(json.loads(self.bridge.imageSearchTextsReady.calls[1][0])["texts"], {})


class SearchTextCacheTests(unittest.TestCase):
    def test_cache_is_invalidated_by_mtime_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.png"
            MetadataActionParityTests.write_png(path, "first\nSteps: 20, Seed: 1")
            cache = SearchTextCache(max_entries=1)
            reads = []

            def reader(p):
                reads.append(p)
                return metadata_search_text(p)

            self.assertIn("first", cache.get(str(path), reader))
            self.assertIn("first", cache.get(str(path), reader))
            self.assertEqual(len(reads), 1)
            MetadataActionParityTests.write_png(path, "second\nSteps: 20, Seed: 1")
            stat = path.stat()
            import os
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            self.assertIn("second", cache.get(str(path), reader))
            other = Path(temp) / "b.png"
            MetadataActionParityTests.write_png(other, "other\nSteps: 20, Seed: 1")
            cache.get(str(other), reader)
            self.assertEqual(len(cache), 1)

    def test_workflow_json_is_only_used_when_nothing_else_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "w.png"
            MetadataActionParityTests.write_png(path, "cat\nSteps: 20, Seed: 1", workflow='{"nodes": ["SECRET_NODE"]}')
            self.assertNotIn("secret_node", metadata_search_text(path))
            texts = search_texts_for([str(path), ""], is_allowed=lambda p: p)
            self.assertEqual(list(texts), [str(path)])


if __name__ == "__main__":
    unittest.main()
