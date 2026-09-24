"""Krea2/H3 그래프 준비 단일 소스 — Krea2 러너·Creator·채팅·생성 API 가 같은 규칙을 쓴다 (감사 #169).

- 노드 검사·선택지 해석: core/comfy_choice_resolver 한 벌 (krea2_generation 사본 삭제)
- /object_info: Creator 도 backend.get_object_info(재시도) 우선, 없는 fake 만 GET 폴백
- H3 turbo block cache 자동 판단: _creator_configure_h3_cache 안 → Living Comic 도 사용
- seed: Krea2 는 32비트(앱 replay 범위) — 생성 API 검증·Creator 난수 모두
- 채팅 Krea2: T2I 의 steps/cfg/sampler 를 넘기고 <lora:> 태그를 뗀다
- mode 별칭: creator_workflows.canonical_mode 한 벌
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from core import comfy_choice_resolver, krea2_generation
from core.comfy_choice_resolver import check_required_nodes, resolve_choices
from core.creator_workflows import CreatorWorkflowError, build, canonical_mode
from core.generation_family import KREA2_SEED_MAX, STANDARD_SEED_MAX, seed_max_for_family
from ui.creator_actions import CreatorActionsMixin


def _info(class_type, field, choices):
    return {class_type: {"input": {"required": {field: [list(choices), {}]}}}}


class ChoiceResolverTests(unittest.TestCase):
    def test_missing_nodes_are_reported_sorted(self):
        with self.assertRaisesRegex(RuntimeError, "A, Z"):
            check_required_nodes({"required_node_types": ["Z", "LoadImage", "A"]}, {"LoadImage"})
        check_required_nodes({"required_node_types": ["LoadImage"]}, {"LoadImage", "Other"})

    def test_exact_casefold_separator_match_then_unique_stem_then_error(self):
        built = {"workflow": {
            "1": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "krea2/ID.safetensors"}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.safetensors"}},
        }}
        info = {**_info("LoraLoaderModelOnly", "lora_name", ["Krea2\\id.safetensors"]),
                **_info("UpscaleModelLoader", "model_name", ["RealESRGAN_x4plus.pth", "other.pth"])}
        resolve_choices(built, info)
        self.assertEqual(built["workflow"]["1"]["inputs"]["lora_name"], "Krea2\\id.safetensors")
        self.assertEqual(built["workflow"]["2"]["inputs"]["model_name"], "RealESRGAN_x4plus.pth")

        ambiguous = {"workflow": {"1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "x.safetensors"}}}}
        with self.assertRaisesRegex(RuntimeError, "UpscaleModelLoader.model_name"):
            resolve_choices(ambiguous, _info("UpscaleModelLoader", "model_name", ["a/x.pth", "b/x.pt"]))

    def test_uploaded_load_image_and_non_choice_inputs_are_untouched(self):
        built = {"workflow": {
            "1": {"class_type": "LoadImage", "inputs": {"image": "fresh_upload.png"}},
            "2": {"class_type": "KSampler", "inputs": {"seed": 5, "sampler_name": "euler"}},
            "3": "not-a-node",
        }}
        info = {**_info("LoadImage", "image", ["old.png"]), "KSampler": {"input": {"required": {"seed": ["INT", {}]}}}}
        resolve_choices(built, info)
        self.assertEqual(built["workflow"]["1"]["inputs"]["image"], "fresh_upload.png")
        self.assertEqual(built["workflow"]["2"]["inputs"], {"seed": 5, "sampler_name": "euler"})

    def test_h3_cache_descriptor_models_are_resolved_and_reserialised(self):
        built = build("h3_t2v", {"prompt": "synthetic", "conditioning_cache": True})
        names = {}
        for stage in built["stages"]:
            for node in stage["workflow"].values():
                if node["class_type"] == "CLIPLoader":
                    names.setdefault("CLIPLoader", set()).add("shared\\" + node["inputs"]["clip_name"])
        resolve_choices(built, _info("CLIPLoader", "clip_name", sorted(names["CLIPLoader"])))
        descriptors = [json.loads(node["inputs"]["descriptor"]) for stage in built["stages"]
                       for node in stage["workflow"].values()
                       if node["class_type"].startswith("ForgeNeoH3ConditioningCache")]
        self.assertTrue(descriptors)
        dumped = json.dumps(descriptors, ensure_ascii=False)
        self.assertIn("shared\\\\", dumped)

    def test_krea2_runner_and_creator_share_the_single_resolver(self):
        self.assertFalse(hasattr(krea2_generation, "_resolve_comfy_choices"))
        self.assertFalse(hasattr(krea2_generation, "_check_required_nodes"))
        self.assertIs(krea2_generation.resolve_choices, comfy_choice_resolver.resolve_choices)
        with mock.patch("core.comfy_choice_resolver.resolve_choices") as resolver:
            CreatorActionsMixin._creator_resolve_comfy_choices({"workflow": {}}, {})
        resolver.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            CreatorActionsMixin._creator_check_nodes({"required_node_types": ["Missing"]}, set())


class CreatorObjectInfoTests(unittest.TestCase):
    def test_prefers_backend_get_object_info_without_forcing_cancel_check(self):
        calls = []

        class _Backend:
            api_url = "http://synthetic.invalid"

            def get_object_info(self):  # ComfyUIBackend 처럼 cancel_check 를 안 받는다
                calls.append("retrying-get")
                return {"Node": {}}

        with mock.patch("requests.get", side_effect=AssertionError("raw GET must not be used")):
            info = CreatorActionsMixin._creator_object_info(_Backend(), lambda: False)
        self.assertEqual(info, {"Node": {}})
        self.assertEqual(calls, ["retrying-get"])

    def test_non_dict_object_info_is_an_error(self):
        backend = SimpleNamespace(get_object_info=lambda: ["bad"])
        with self.assertRaisesRegex(RuntimeError, "object_info"):
            CreatorActionsMixin._creator_object_info(backend)

    def test_duck_typed_adapter_without_getter_falls_back_to_get(self):
        backend = SimpleNamespace(api_url="http://synthetic.invalid/")
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"Node": {}})
        with mock.patch("requests.get", return_value=response) as get:
            self.assertEqual(CreatorActionsMixin._creator_object_info(backend), {"Node": {}})
        self.assertEqual(get.call_args.args[0], "http://synthetic.invalid/object_info")


class H3BlockCacheTests(unittest.TestCase):
    def _configure(self, mode, params, available):
        actions = CreatorActionsMixin()
        actions._creator_prefs = lambda: {"h3ConditioningCacheEnabled": False}
        actions._creator_emit = lambda *_a, **_k: None
        actions._creator_configure_h3_cache(mode, params, available)
        return params

    def test_turbo_i2v_enables_block_cache_when_server_has_node_for_living_comic_too(self):
        params = self._configure("h3_i2v", {"prompt": "x"}, {"MiniMaxH3BlockCacheT8"})
        self.assertTrue(params["block_cache"])
        params = self._configure("h3_i2v", {"prompt": "x"}, set())
        self.assertFalse(params["block_cache"])

    def test_quality_mode_v2v_and_explicit_choice_are_respected(self):
        self.assertNotIn("block_cache", self._configure("h3_t2v", {"quality": "quality"}, {"MiniMaxH3BlockCacheT8"}))
        self.assertNotIn("block_cache", self._configure("h3_v2v", {}, {"MiniMaxH3BlockCacheT8"}))
        self.assertIs(self._configure("h3_t2v", {"block_cache": False}, {"MiniMaxH3BlockCacheT8"})["block_cache"], False)
        self.assertNotIn("block_cache", self._configure("h3_t2v", {"blockCache": False}, {"MiniMaxH3BlockCacheT8"}))

    def test_living_comic_animation_passes_server_capabilities_to_the_shared_configurator(self):
        from pathlib import Path
        import tempfile
        from contextlib import nullcontext
        from backends import BackendType

        seen = []
        actions = CreatorActionsMixin()
        actions._creator_cancel_event = __import__("threading").Event()
        actions._creator_should_unload_ollama = lambda: False
        actions._creator_reserve = lambda *_a, **_k: nullcontext()
        actions._creator_emit = lambda *_a, **_k: None
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "panel.png"
            image.write_bytes(b"png")
            panel = SimpleNamespace(image_path=str(image), motion_prompt="walk", text="", seed=-1, video_path="")
            document = SimpleNamespace(panels=[panel], to_dict=lambda: {})
            actions._comic_studio = lambda: SimpleNamespace(normalize=lambda _d: document, save=lambda d: d)
            actions._creator_configure_h3_cache = lambda mode, params, available: seen.append((mode, set(available)))
            actions._creator_run_workflow = mock.Mock(side_effect=RuntimeError("stop after configure"))
            backend = SimpleNamespace(upload_media=lambda *_a: "panel.png",
                                      get_object_info=lambda: {name: {} for name in
                                                               build("h3_i2v", {"prompt": "x", "input_image": "a.png"})["required_node_types"]}
                                      | {"MiniMaxH3BlockCacheT8": {}})
            with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI), \
                 mock.patch("backends.get_backend", return_value=backend):
                with self.assertRaisesRegex(RuntimeError, "stop after configure"):
                    actions._comic_animate_all({"document": {}})
        self.assertEqual(seen[0][0], "h3_i2v")
        self.assertIn("MiniMaxH3BlockCacheT8", seen[0][1])


class SeedRangeTests(unittest.TestCase):
    def test_family_seed_limits(self):
        self.assertEqual(KREA2_SEED_MAX, 0xFFFFFFFF)
        self.assertEqual(seed_max_for_family("krea2"), KREA2_SEED_MAX)
        self.assertEqual(seed_max_for_family("Krea2"), KREA2_SEED_MAX)
        self.assertEqual(seed_max_for_family("standard"), STANDARD_SEED_MAX)

    def test_generation_api_rejects_out_of_range_krea2_seed_at_validation(self):
        from core.generation_api import GenerationValidationError, _normalise_job_request

        config = {"targets": []}
        ok, _ = _normalise_job_request({"family": "krea2", "payload": {"prompt": "x", "seed": KREA2_SEED_MAX}}, config)
        self.assertEqual(ok["payload"]["seed"], KREA2_SEED_MAX)
        with self.assertRaisesRegex(GenerationValidationError, "seed"):
            _normalise_job_request({"family": "krea2", "payload": {"prompt": "x", "seed": KREA2_SEED_MAX + 1}}, config)
        standard, _ = _normalise_job_request({"payload": {"prompt": "x", "seed": 1 << 40}}, config)
        self.assertEqual(standard["payload"]["seed"], 1 << 40)

    def test_creator_random_seed_stays_in_app_replay_range_for_krea2(self):
        actions = CreatorActionsMixin()
        backend = SimpleNamespace(upload_media=mock.Mock())
        with mock.patch("ui.creator_actions.secrets.randbelow", return_value=KREA2_SEED_MAX) as below, \
             mock.patch("ui.creator_actions.secrets.randbits", side_effect=AssertionError("63-bit seed for krea2")):
            params = actions._creator_prepare_params(backend, {"mode": "krea2_t2i", "prompt": "x", "seed": -1})
        below.assert_called_once_with(KREA2_SEED_MAX + 1)
        self.assertEqual(params["seed"], KREA2_SEED_MAX)
        # 별칭(krea2 → krea2_edit)도 같은 범위
        with mock.patch("ui.creator_actions.secrets.randbelow", return_value=7):
            self.assertEqual(actions._creator_prepare_params(backend, {"mode": "krea2", "seed": -1})["seed"], 7)
        # H3 는 예전처럼 63비트
        with mock.patch("ui.creator_actions.secrets.randbits", return_value=1 << 50) as bits:
            self.assertEqual(actions._creator_prepare_params(backend, {"mode": "h3_t2v", "seed": -1})["seed"], 1 << 50)
        bits.assert_called_once_with(63)

    def test_creator_rejects_out_of_range_krea2_seed_before_uploading(self):
        import tempfile
        from pathlib import Path

        actions = CreatorActionsMixin()
        backend = SimpleNamespace(upload_media=mock.Mock())
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            source.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "4294967295"):
                actions._creator_prepare_params(backend, {"mode": "krea2_edit", "sourcePath": str(source),
                                                          "seed": KREA2_SEED_MAX + 1})
        backend.upload_media.assert_not_called()


class ModeAliasTests(unittest.TestCase):
    def test_canonical_mode_is_the_single_alias_table(self):
        self.assertEqual(canonical_mode("krea2"), "krea2_edit")
        self.assertEqual(canonical_mode("krea_edit"), "krea2_edit")
        self.assertEqual(canonical_mode("krea_hires"), "krea2_hires")
        self.assertEqual(canonical_mode("H3-T2V"), "h3_t2v")
        with self.assertRaises(CreatorWorkflowError):
            canonical_mode("nope")

    def test_media_inputs_follow_the_capability_table(self):
        """Creator 가 올리는 미디어 = 그 mode 그래프가 읽는 입력(아이덴티티는 V2V 전용)."""
        from core.creator_workflows import media_inputs

        self.assertEqual(media_inputs("h3_t2v"), frozenset())
        self.assertEqual(media_inputs("H3-I2V"), {"input_image"})
        self.assertEqual(media_inputs("h3_v2v"), {"input_video", "input_image"})
        self.assertEqual(media_inputs("krea2"), {"input_image", "reference_image"})
        self.assertEqual(media_inputs("krea2_hires"), {"input_image"})
        self.assertEqual(media_inputs("krea2_t2i"), frozenset())
        with self.assertRaises(CreatorWorkflowError):
            media_inputs("nope")

    def test_creator_generate_rejects_unknown_mode_before_uploading_or_querying(self):
        from backends import BackendType

        actions = CreatorActionsMixin()
        actions._creator_cancel_event = __import__("threading").Event()
        backend = SimpleNamespace(run_workflow=mock.Mock(), upload_media=mock.Mock(),
                                  get_object_info=mock.Mock(side_effect=AssertionError("queried")))
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI), \
             mock.patch("backends.get_backend", return_value=backend):
            with self.assertRaises(CreatorWorkflowError):
                actions._creator_generate({"mode": "bogus", "sourcePath": "C:/nowhere.png"})
        backend.upload_media.assert_not_called()

    def test_v2v_aliases_prepare_the_same_params_as_canonical_mode(self):
        """'h3-v2v'/'H3_V2V' 도 원본 영상을 input_video 로 올리고 오디오·<Video 1> 지시를 받는다."""
        import tempfile
        from pathlib import Path

        def prepare(mode):
            uploads = []
            backend = SimpleNamespace(upload_media=lambda data, name, mime: uploads.append(name) or f"up/{name}")
            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "motion.mp4"
                face = Path(tmp) / "face.png"
                video.write_bytes(b"video")
                face.write_bytes(b"image")
                params = CreatorActionsMixin()._creator_prepare_params(backend, {
                    "mode": mode, "sourcePath": str(video), "identityPath": str(face),
                    "prompt": "walk forward", "seed": 11, "includeAudio": True,
                })
            return params, uploads

        from core.comfy_upload_names import content_upload_name

        canonical, canonical_uploads = prepare("h3_v2v")
        # 업로드 이름은 역할 접두어 + 내용 해시(사용자 파일명·overwrite 로 서로 덮지 않게).
        self.assertEqual(canonical["input_video"], "up/" + content_upload_name(b"video", "creator_source", "mp4"))
        self.assertEqual(canonical["input_image"], "up/" + content_upload_name(b"image", "creator_identity", "png"))
        self.assertIs(canonical["include_reference_audio"], True)
        self.assertTrue(canonical["prompt"].startswith("Use <Picture 1>"))
        self.assertIn("Use <Video 1>", canonical["prompt"])
        self.assertIn("Use <Audio 1>", canonical["prompt"])
        for alias in ("h3-v2v", "H3_V2V", " H3-V2V "):
            with self.subTest(alias=alias):
                params, uploads = prepare(alias)
                self.assertEqual(uploads, canonical_uploads)
                for key in ("input_video", "input_image", "include_reference_audio", "generate_audio",
                            "prompt", "seed"):
                    self.assertEqual(params[key], canonical[key], key)
                # build() 가 정규 mode 로 그대로 받아야 한다(업로드 뒤 확장자 오류가 아니라)
                build(canonical_mode(alias), params)


class ChatCreatorPreparationTests(unittest.TestCase):
    def _prepare(self, snapshot, content="draw a cat"):
        from core.chat_generation import MediaGenerationJob, plan_chat_generation

        plan = plan_chat_generation({"generation": {"mode": "image", "family": "krea2"},
                                     "messages": [{"role": "user", "content": content}]})
        return MediaGenerationJob("chat-krea2", plan).prepare_creator(snapshot)

    def test_krea2_follows_t2i_sampling_and_strips_forge_lora_tags(self):
        prepared = self._prepare({"prompt": "cat, <lora:style:0.8>, sunset", "seed": 5, "width": 832,
                                  "height": 1216, "steps": 12, "cfg_scale": 1.5,
                                  "sampler_name": "DPM++ 2M SDE"})
        self.assertEqual(prepared["mode"], "krea2_t2i")
        self.assertEqual(prepared["prompt"], "cat, sunset")
        self.assertEqual((prepared["steps"], prepared["cfg"], prepared["sampler"]), (12, 1.5, "dpmpp_2m_sde"))
        self.assertEqual((prepared["width"], prepared["height"], prepared["seed"]), (832, 1216, 5))
        # Creator 그래프가 그대로 받는 값이어야 한다
        build(prepared["mode"], prepared)

    def test_plain_chat_request_without_t2i_snapshot_keeps_graph_defaults(self):
        prepared = self._prepare({"prompt": "a fox"})
        self.assertNotIn("steps", prepared)
        self.assertNotIn("cfg", prepared)
        self.assertNotIn("sampler", prepared)
        self.assertEqual(prepared["prompt"], "a fox")


if __name__ == "__main__":
    unittest.main()
