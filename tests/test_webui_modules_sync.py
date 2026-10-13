"""Forge classic 은 요청 본문의 forge_additional_modules 를 읽지 않는다 — 옵션으로 동기화해야 한다."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backends import webui_backend
from backends.webui_backend import WebUIBackend


class _Resp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


FORGE_OPTS = {
    "sd_model_checkpoint": "krea2_raw_bf16.safetensors",
    "forge_additional_modules": [],
}
ANIMA_MODULES = [
    "qwenImage_qwenImageVAE.safetensors",
    "Anima-3.8B-expanded_adapter.safetensors",
    "qwen35_4b.safetensors",
    "qwen_3_06b_base.safetensors",
]


class ForgeModuleSyncTests(unittest.TestCase):
    def setUp(self):
        self.backend = WebUIBackend("http://127.0.0.1:17860")
        webui_backend._MODULE_RESTORE.clear()
        self.addCleanup(webui_backend._MODULE_RESTORE.clear)

    def _run(self, options, model, modules, **kwargs):
        with patch("backends.webui_backend.requests.get", return_value=_Resp(options)), \
             patch("backends.webui_backend.requests.post", return_value=_Resp({})) as post:
            self.backend._switch_model_if_needed(model, modules, **kwargs)
        return post

    def test_anima_switch_sends_modules_and_checkpoint_in_one_post(self):
        """예전엔 체크포인트만 바꿔 Anima 가 VAE 없이 로드됐다 — 'You do not have VAE state dict!'."""
        post = self._run(FORGE_OPTS, "Anima-3.8B-v1.1.safetensors", ANIMA_MODULES)
        self.assertEqual(post.call_count, 1)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["sd_model_checkpoint"], "Anima-3.8B-v1.1.safetensors")
        self.assertEqual(body["forge_additional_modules"], sorted(ANIMA_MODULES))
        self.assertEqual(list(body), ["forge_additional_modules", "sd_model_checkpoint"], "모듈이 체크포인트보다 먼저")

    def test_same_modules_by_basename_do_not_repost(self):
        """옵션엔 전체 경로가, 요청엔 파일명이 온다 — 파일명이 같으면 다시 보내지 않는다(재로딩 방지)."""
        opts = dict(FORGE_OPTS, sd_model_checkpoint="Anima-3.8B-v1.1.safetensors",
                    forge_additional_modules=[f"C:\\\\sd-webui-forge-classic\\\\models\\\\text_encoder\\\\{n}" for n in ANIMA_MODULES])
        post = self._run(opts, "Anima-3.8B-v1.1.safetensors", list(reversed(ANIMA_MODULES)))
        self.assertEqual(post.call_count, 0)

    def test_empty_request_clears_leftover_modules(self):
        """모듈이 필요 없는 체크포인트로 가면 남아 있던 Anima 모듈을 비운다."""
        opts = dict(FORGE_OPTS, forge_additional_modules=["C:/x/models/VAE/qwen_image_vae.safetensors"])
        post = self._run(opts, "krea2_raw_bf16.safetensors", [])
        self.assertEqual(post.call_args.kwargs["json"], {"forge_additional_modules": []})

    def test_a1111_without_the_option_only_switches_checkpoint(self):
        post = self._run({"sd_model_checkpoint": "a.safetensors"}, "b.safetensors", ANIMA_MODULES)
        self.assertEqual(post.call_args.kwargs["json"], {"sd_model_checkpoint": "b.safetensors"})

    def test_no_modules_key_leaves_modules_alone(self):
        opts = dict(FORGE_OPTS, forge_additional_modules=["C:/x/vae.safetensors"])
        post = self._run(opts, "krea2_raw_bf16.safetensors", None)
        self.assertEqual(post.call_count, 0)

    def test_generate_passes_payload_modules(self):
        """_generate 가 본문의 모듈을 동기화 함수로 넘긴다 — 여기서 끊기면 위 테스트가 다 헛것이다."""
        seen = {}
        with patch.object(self.backend, "_switch_model_if_needed",
                          side_effect=lambda m, mods=None, transient=False, te_cell=None: seen.update(
                              model=m, modules=mods, transient=transient, te_cell=te_cell)), \
             patch("backends.webui_backend.requests.post", side_effect=RuntimeError("stop here")):
            result = self.backend._generate("/sdapi/v1/txt2img", "Anima-3.8B-v1.1.safetensors",
                                            {"prompt": "x", "forge_additional_modules": ANIMA_MODULES})
        self.assertFalse(result.success)
        self.assertEqual(seen, {"model": "Anima-3.8B-v1.1.safetensors", "modules": ANIMA_MODULES, "transient": False,
                                "te_cell": None})

    def test_generate_takes_the_xyz_marker_off_the_request(self):
        """XYZ Text Encoder 축 칸의 표시·TE 선택은 동기화 함수로만 가고 Forge 요청에는 실리지 않는다."""
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TRANSIENT_MODULES_KEY

        cell = {"value": "te_b.safetensors", "encoders": ["te_a.safetensors", "te_b.safetensors"]}
        for payload, expected in (
                ({"prompt": "x", "forge_additional_modules": ["te_b.safetensors"], TRANSIENT_MODULES_KEY: True},
                 {"transient": True, "te_cell": None}),
                ({"prompt": "x", TRANSIENT_MODULES_KEY: True, FORGE_TE_CELL_KEY: cell},
                 {"transient": True, "te_cell": cell})):
            seen, sent = {}, []
            with self.subTest(keys=sorted(payload)), \
                 patch.object(self.backend, "_switch_model_if_needed",
                              side_effect=lambda m, mods=None, transient=False, te_cell=None: seen.update(
                                  transient=transient, te_cell=te_cell)), \
                 patch("backends.webui_backend.requests.post",
                       side_effect=lambda *a, **k: sent.append(k.get("json")) or (_ for _ in ()).throw(RuntimeError("x"))):
                self.backend._generate("/sdapi/v1/txt2img", "m.safetensors", payload)
                self.assertEqual(seen, expected)
                self.assertTrue(sent)
                self.assertTrue(all(not {TRANSIENT_MODULES_KEY, FORGE_TE_CELL_KEY} & set(body)
                                    for body in sent if isinstance(body, dict)))
                self.assertIn(TRANSIENT_MODULES_KEY, payload, "the caller's payload is not changed")


class _ForgeOptions:
    """Forge /sdapi/v1/options: GET gives the stored options, POST changes (and keeps) them."""

    def __init__(self, modules):
        self.options = {"sd_model_checkpoint": "anima.safetensors",
                        "forge_additional_modules": [f"C:/forge/models/text_encoder/{m}" for m in modules]}
        self.posts = []

    def get(self, url, **kwargs):
        return _Resp(dict(self.options))

    def post(self, url, json=None, **kwargs):
        self.posts.append(json)
        self.options.update(json)
        return _Resp({})


class XyzTextEncoderRestoreTests(unittest.TestCase):
    """XYZ 'Text Encoder' 축은 칸마다 Forge 옵션(config.json 에도 저장)을 바꾼다 — 끝난 뒤 모듈 키 없이 오는 보통 생성은
    처음 목록으로 되돌린다(확장의 '[Modules] Text Encoder' 축처럼 남는 상태가 없다)."""

    def setUp(self):
        webui_backend._MODULE_RESTORE.clear()
        self.addCleanup(webui_backend._MODULE_RESTORE.clear)
        self.forge = _ForgeOptions(["clip_l_a.safetensors"])

    def _switch(self, modules, *, transient=False, backend=None):
        backend = backend or WebUIBackend("http://127.0.0.1:17860")
        with patch("backends.webui_backend.requests.get", side_effect=self.forge.get), \
             patch("backends.webui_backend.requests.post", side_effect=self.forge.post):
            backend._switch_model_if_needed("anima.safetensors", modules, transient=transient)

    def _names(self):
        return WebUIBackend._module_names(self.forge.options["forge_additional_modules"])

    def test_the_next_generation_without_modules_restores_the_list_before_the_run(self):
        self._switch([], transient=True)                          # 'Use current' with an empty chip
        self._switch(["clip_l_b.safetensors"], transient=True)    # the last cell
        self.assertEqual(self._names(), ["clip_l_b.safetensors"])
        self._switch(None, backend=WebUIBackend("http://127.0.0.1:17860"))   # normal T2I, empty chip (a new backend)
        self.assertEqual(self._names(), ["clip_l_a.safetensors"])
        self.assertEqual(self.forge.posts[-1], {"forge_additional_modules": ["clip_l_a.safetensors"]})
        posts = len(self.forge.posts)
        self._switch(None)
        self.assertEqual(len(self.forge.posts), posts, "restored once")

    def test_a_list_changed_elsewhere_after_the_run_is_kept(self):
        self._switch(["clip_l_b.safetensors"], transient=True)
        self.forge.options["forge_additional_modules"] = ["C:/forge/models/text_encoder/clip_l_c.safetensors"]
        self._switch(None)
        self.assertEqual(self._names(), ["clip_l_c.safetensors"], "the Forge UI's choice wins")
        self.assertEqual(webui_backend._MODULE_RESTORE, {})

    def test_a_request_with_its_own_modules_ends_the_restore(self):
        self._switch(["clip_l_b.safetensors"], transient=True)
        self._switch(["clip_l_d.safetensors"])                    # a normal request with the chip set
        self.assertEqual(self._names(), ["clip_l_d.safetensors"])
        self._switch(None)
        self.assertEqual(self._names(), ["clip_l_d.safetensors"])

    def test_without_a_run_nothing_is_restored(self):
        self._switch(["clip_l_b.safetensors"])                    # the chip, not the XYZ axis
        self._switch(None)
        self.assertEqual(self._names(), ["clip_l_b.safetensors"])

    def test_an_xyz_run_then_a_normal_generation_end_to_end(self):
        """build_jobs → 대기열 칸(_generate) → 칩이 빈 보통 생성(_generate): Forge 가 처음 목록으로 돌아온다.
        칩이 빈 Forge 칸은 Forge 의 목록에서 TE 만 바꾼다 — 'Use current' 칸은 Forge 의 목록 그대로다(예전엔 빈 목록)."""
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT, build_jobs, text_encoder_axis

        capability = {"axes": [text_encoder_axis(["clip_l_a.safetensors", "clip_l_b.safetensors"], source="t")],
                      "backend": "webui"}
        jobs = build_jobs({"prompt": "p"}, "anima.safetensors",
                          [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "clip_l_b.safetensors"]}],
                          capability, text_encoders=[], forge_modules=["clip_l_a.safetensors"])
        backend = WebUIBackend("http://127.0.0.1:17860")
        seen = []

        def generate(payload):
            payload = {k: v for k, v in payload.items() if not k.startswith("_xyz")}
            with patch.object(backend, "_generate_once", side_effect=RuntimeError("stop after the switch")), \
                 patch("backends.webui_backend.requests.get", side_effect=self.forge.get), \
                 patch("backends.webui_backend.requests.post", side_effect=self.forge.post):
                backend._generate("/sdapi/v1/txt2img", "anima.safetensors", payload)
            seen.append(self._names())

        for job in jobs:
            generate(job)
        generate({"prompt": "p"})
        self.assertEqual(seen, [["clip_l_a.safetensors"], ["clip_l_b.safetensors"], ["clip_l_a.safetensors"]])


ANIMA_VAE = "C:/forge/models/VAE/qwen_image_vae.safetensors"
ANIMA_TE = "C:\\forge\\models\\text_encoder\\qwen_3_06b_base.safetensors"
TE_CHOICES = ["qwen_3_06b_base.safetensors", "sub/qwen_alt.safetensors"]   # 앱 TE 칩 선택지(텍스트 인코더 폴더)


class XyzTextEncoderForgeListTests(unittest.TestCase):
    """앱 VAE·TE 칩이 빈 Text Encoder 축 칸 — 보통 생성처럼 Forge 드롭다운의 목록(Anima: VAE + TE)을 쓰고 TE 만 바꾼다.
    예전엔 칸이 [그 TE]·[] 를 옵션으로 보내 Forge 가 VAE 를 잃었다("You do not have VAE state dict!" 500)."""

    def setUp(self):
        webui_backend._MODULE_RESTORE.clear()
        self.addCleanup(webui_backend._MODULE_RESTORE.clear)
        self.forge = _ForgeOptions([])
        self.forge.options["forge_additional_modules"] = [ANIMA_VAE, ANIMA_TE]

    def _patched(self):
        return patch("backends.webui_backend.requests.get", side_effect=self.forge.get), \
            patch("backends.webui_backend.requests.post", side_effect=self.forge.post)

    def _cell(self, value, *, encoders=TE_CHOICES, backend=None):
        backend = backend or WebUIBackend("http://127.0.0.1:17860")
        get, post = self._patched()
        with get, post:
            backend._switch_model_if_needed("anima.safetensors", None, transient=True,
                                            te_cell={"value": value, "encoders": list(encoders)})

    def _normal(self):
        """칩이 빈 보통 생성 — 모듈 키 없음(새 백엔드 객체: 기록은 모듈 단위라 남는다)."""
        get, post = self._patched()
        with get, post:
            WebUIBackend("http://127.0.0.1:17860")._switch_model_if_needed("anima.safetensors", None)

    def _base(self):
        get, post = self._patched()
        with get, post:
            return WebUIBackend("http://127.0.0.1:17860").forge_module_base()

    def _names(self):
        return WebUIBackend._module_names(self.forge.options["forge_additional_modules"])

    def test_a_te_cell_keeps_forges_vae(self):
        self._cell("sub/qwen_alt.safetensors")
        self.assertEqual(self._names(), ["qwen_alt.safetensors", "qwen_image_vae.safetensors"])
        self.assertEqual(self.forge.posts, [{"forge_additional_modules": ["qwen_alt.safetensors",
                                                                          "qwen_image_vae.safetensors"]}])
        self.assertEqual(webui_backend._MODULE_RESTORE["http://127.0.0.1:17860"],
                         {"written": ["qwen_alt.safetensors", "qwen_image_vae.safetensors"],
                          "original": ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"]})

    def test_use_current_keeps_forges_list(self):
        self._cell(None)
        self.assertEqual(self.forge.posts, [], "Forge's list is the cell's list - nothing to change")
        self._cell("sub/qwen_alt.safetensors")
        self._cell(None)                                          # 'Use current' after a TE cell puts Forge's list back
        self.assertEqual(self._names(), ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"])
        self._normal()
        self.assertEqual(len(self.forge.posts), 2, "nothing left to restore")
        self.assertEqual(webui_backend._MODULE_RESTORE, {})

    def test_the_next_generation_without_modules_restores_forges_list(self):
        self._cell(None)
        self._cell("qwen_3_06b_base.safetensors")
        self._cell("sub/qwen_alt.safetensors")                    # the last cell
        with self.assertLogs("backends.webui_backend", level="INFO") as logs:
            self._normal()
        self.assertEqual(self._names(), ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"])
        self.assertEqual(self.forge.posts[-1], {"forge_additional_modules": ["qwen_3_06b_base.safetensors",
                                                                             "qwen_image_vae.safetensors"]})
        self.assertTrue(any("XYZ Text Encoder 축 전으로 되돌림" in line for line in logs.output), logs.output)
        posts = len(self.forge.posts)
        self._normal()
        self.assertEqual(len(self.forge.posts), posts, "restored once")

    def test_a_second_run_starts_from_the_list_before_the_first(self):
        """앞 XYZ 가 바꿔 둔 채(보통 생성 전) 다시 돌려도 보통 생성이 쓸 목록 = 첫 XYZ 전 목록이다."""
        self._cell("sub/qwen_alt.safetensors")                    # run 1 ends on the other TE
        self.assertEqual(self._base(), ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"])
        self._cell(None)                                          # run 2, 'Use current'
        self.assertEqual(self._names(), ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"])
        self._cell("sub/qwen_alt.safetensors")
        self.assertEqual(webui_backend._MODULE_RESTORE["http://127.0.0.1:17860"]["original"],
                         ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"])

    def test_a_list_changed_in_forge_between_runs_becomes_the_base(self):
        self._cell("sub/qwen_alt.safetensors")
        self.forge.options["forge_additional_modules"] = ["C:/forge/models/VAE/other_vae.safetensors", ANIMA_TE]
        self.assertEqual(self._base(), ["other_vae.safetensors", "qwen_3_06b_base.safetensors"])
        self._cell(None)
        self.assertEqual(self._names(), ["other_vae.safetensors", "qwen_3_06b_base.safetensors"], "the Forge UI's choice")
        self._cell("sub/qwen_alt.safetensors")
        self.assertEqual(self._names(), ["other_vae.safetensors", "qwen_alt.safetensors"])
        self._normal()
        self.assertEqual(self._names(), ["other_vae.safetensors", "qwen_3_06b_base.safetensors"])

    def test_no_te_in_forges_list_adds_the_choice(self):
        self.forge.options["forge_additional_modules"] = [ANIMA_VAE]
        self._cell("sub/qwen_alt.safetensors")
        self.assertEqual(self._names(), ["qwen_alt.safetensors", "qwen_image_vae.safetensors"])
        self._normal()
        self.assertEqual(self._names(), ["qwen_image_vae.safetensors"])

    def test_two_encoders_in_forges_list_refuse_before_any_change(self):
        self.forge.options["forge_additional_modules"] = ["C:/f/VAE/ae.safetensors", "C:/f/text_encoder/clip_l.safetensors",
                                                          "C:/f/text_encoder/t5xxl.safetensors"]
        encoders = ["clip_l.safetensors", "t5xxl.safetensors", "clip_g.safetensors"]
        with self.assertRaisesRegex(ValueError, "TE 가 2개"):
            self._cell("clip_g.safetensors", encoders=encoders)
        self.assertEqual((self.forge.posts, webui_backend._MODULE_RESTORE), ([], {}))
        self._cell(None, encoders=encoders)                       # 'Use current' only is fine
        self.assertEqual(self.forge.posts, [])
        # 대기열 칸(_generate)은 Forge 에 보내지 않고 이유와 함께 실패한다
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TRANSIENT_MODULES_KEY
        get, post = self._patched()
        backend = WebUIBackend("http://127.0.0.1:17860")
        with get, post, patch.object(backend, "_generate_once", side_effect=AssertionError("must not send")):
            result = backend._generate("/sdapi/v1/txt2img", "anima.safetensors", {
                "prompt": "p", TRANSIENT_MODULES_KEY: True,
                FORGE_TE_CELL_KEY: {"value": "clip_g.safetensors", "encoders": encoders}})
        self.assertFalse(result.success)
        self.assertIn("TE 가 2개", result.error)
        self.assertEqual(self.forge.posts, [])

    def test_a1111_without_the_option_ignores_the_cell(self):
        del self.forge.options["forge_additional_modules"]
        self._cell("sub/qwen_alt.safetensors")
        self.assertEqual((self.forge.posts, webui_backend._MODULE_RESTORE), ([], {}))
        self.assertIsNone(self._base())

    def test_an_anima_xyz_run_end_to_end(self):
        """build_jobs(칩이 빈 Forge) → 대기열 칸(_generate) → 칩이 빈 보통 생성: 칸마다 VAE 가 남고 TE 만 바뀌며, Forge 요청에는
        모듈 키도 비공개 키도 없다(Forge classic 은 옵션만 본다)."""
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT, build_jobs, text_encoder_axis

        capability = {"axes": [text_encoder_axis(TE_CHOICES, source="t")], "backend": "webui"}
        jobs = build_jobs({"prompt": "p"}, "anima.safetensors",
                          [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "sub/qwen_alt.safetensors"]}],
                          capability, text_encoders=[], forge_modules=self._base())
        backend = WebUIBackend("http://127.0.0.1:17860")
        seen, requests_sent = [], []

        def generate(payload):
            payload = {k: v for k, v in payload.items() if not k.startswith("_xyz")}
            get, post = self._patched()
            with get, post, patch.object(backend, "_generate_once",
                                         side_effect=lambda _e, body, *a, **k: requests_sent.append(body) or
                                         (_ for _ in ()).throw(RuntimeError("stop after the switch"))):
                backend._generate("/sdapi/v1/txt2img", "anima.safetensors", payload)
            seen.append(self._names())

        for job in jobs:
            generate(job)
        generate({"prompt": "p"})
        self.assertEqual(seen, [["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"],
                                ["qwen_alt.safetensors", "qwen_image_vae.safetensors"],
                                ["qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"]])
        self.assertEqual(len(requests_sent), 3)
        self.assertTrue(all(not any(key.startswith("_forge") or key == "forge_additional_modules" for key in body)
                            for body in requests_sent), requests_sent)


class ProgressPreviewTests(unittest.TestCase):
    """/progress 의 current_image 는 접두사 없는 base64 로, 없으면 None 으로."""

    def test_plain_and_data_url_forms(self):
        b64 = "A" * 100
        self.assertEqual(WebUIBackend._progress_preview({"current_image": b64}), b64)
        self.assertEqual(WebUIBackend._progress_preview({"current_image": "data:image/png;base64," + b64}), b64)

    def test_missing_or_tiny_is_none(self):
        self.assertIsNone(WebUIBackend._progress_preview({"current_image": None}))
        self.assertIsNone(WebUIBackend._progress_preview({"current_image": ""}))
        self.assertIsNone(WebUIBackend._progress_preview({"current_image": "short"}))
        self.assertIsNone(WebUIBackend._progress_preview("not a dict"))

    def test_polling_forwards_a_preview_only_when_it_changes(self):
        import threading
        backend = WebUIBackend("http://127.0.0.1:17860")
        b64 = "B" * 100
        responses = [
            {"state": {"sampling_step": 1, "sampling_steps": 4}, "current_image": b64},
            {"state": {"sampling_step": 2, "sampling_steps": 4}, "current_image": b64},
            {"state": {"sampling_step": 3, "sampling_steps": 4}, "current_image": "C" * 100},
        ]
        seen = []
        stop = threading.Event()

        def fake_get(url, timeout=3):
            data = responses.pop(0)
            if not responses:
                stop.set()
            return _Resp(data)

        with patch("backends.webui_backend.requests.get", side_effect=fake_get):
            backend._start_progress_polling(lambda step, total, preview: seen.append((step, preview)), stop)
        self.assertEqual(seen, [(1, b64), (2, None), (3, "C" * 100)])


if __name__ == "__main__":
    unittest.main()
