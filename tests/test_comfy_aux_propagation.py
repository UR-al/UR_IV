"""ComfyUI 단독 ADetailer·SAM3·Refine — 봉투가 저장 문맥의 샘플링 블록을 교체한다(P7 T11, T10)."""
import copy
import json
import unittest
from unittest import mock

from core import alwayson_propagation as ap
from core import anima38, anima_guidance
from tests.test_alwayson_propagation import dd_block
from tests import test_comfy_backend_regressions as regressions

_png = regressions._png

PAG = anima_guidance.SCRIPT_PERTURBATION


def pag_block():
    return {"args": anima_guidance.build_args(PAG, {"guid_enabled": True})}


def envelope(blocks):
    return ap.envelope(blocks, source="t2i_panel", backend=ap.BACKEND_COMFY)


class ComfyEnvelopeTests(unittest.TestCase):
    _backend = regressions.TestStandaloneComfyContext._backend   # 같은 하네스(클래스는 가져오지 않는다 — 중복 실행 방지)

    def _compiled_payload(self, backend, settings, kind):
        captured = {}

        def compile_graph(build):
            compiler = mock.Mock()

            def compile_postprocess(model_name, payload, **kwargs):
                captured["payload"] = copy.deepcopy(payload)
                return {}

            compiler.compile_postprocess.side_effect = compile_postprocess
            return build(compiler)

        with mock.patch.object(backend, "_compile_graph", side_effect=compile_graph):
            getattr(backend, kind)(_png(), settings)
        return captured["payload"]

    def test_t11_envelope_replaces_saved_sampling_blocks(self):
        for kind in ("adetailer", "sam3", "refine"):
            with self.subTest(kind=kind):
                backend = self._backend({PAG: pag_block(), "NegPiP": {"args": [True]},
                                         "SAM3 Mask": {"args": [{"sam3_prompt": "old"}]}})
                before = copy.deepcopy(backend._last_generation_context)
                settings = {"sam3_prompt": "face", ap.SETTINGS_KEY: envelope({
                    anima_guidance.SCRIPT_DETAIL_DAEMON: dd_block(), "NegPiP": {"args": [True]}})}
                payload = self._compiled_payload(backend, settings, kind)
                scripts = payload["alwayson_scripts"]
                self.assertNotIn(PAG, scripts)                                  # 저장 문맥의 PAG 는 교체돼 빠진다
                self.assertIn(anima_guidance.SCRIPT_DETAIL_DAEMON, scripts)     # 봉투의 DD
                self.assertIn("NegPiP", scripts)                                # Comfy 는 NegPiP 을 받는다
                self.assertNotIn(ap.SETTINGS_KEY, json.dumps(payload))          # T10
                self.assertEqual(backend._last_generation_context, before)

    def test_without_envelope_the_saved_context_is_used(self):
        backend = self._backend({PAG: pag_block()})
        payload = self._compiled_payload(backend, {"sam3_prompt": "face"}, "sam3")
        self.assertIn(PAG, payload["alwayson_scripts"])

    def test_explicit_settings_scripts_win_over_the_envelope(self):
        backend = self._backend({PAG: pag_block()})
        explicit = {"args": [{"bypass": True}]}
        settings = {"alwayson_scripts": {anima38.SCRIPT_NAME: explicit},
                    ap.SETTINGS_KEY: envelope({anima38.SCRIPT_NAME: {"args": [{"negative": True}]}})}
        before = copy.deepcopy(settings)
        payload = self._compiled_payload(backend, settings, "adetailer")
        self.assertEqual(payload["alwayson_scripts"][anima38.SCRIPT_NAME], explicit)
        self.assertNotIn(PAG, payload["alwayson_scripts"])
        self.assertEqual(settings, before)

    def test_a7_anima38_follows_the_saved_context_model(self):
        """(critic A7) Comfy 단독 후처리는 저장된 문맥의 모델로 돈다 — 봉투(T2I 콤보) 모델과 같을 때만 Anima38 을 넣고,
        다르거나 모르면 뺀다(저장 문맥의 Anima38 도 교체 규칙으로 이미 빠진다 — 다른 모델로 만든 값을 섞지 않는다)."""
        a38 = {"args": [{"negative": True}]}
        for label, model, expected in (("같은 모델(폴더·해시 차이)", f"diffusion/{regressions.V2_MODEL} [abcdef12]", True),
                                       ("다른 모델", "Anima-2.9B-preview-v1.safetensors", False),
                                       ("모름", "", False)):
            with self.subTest(label):
                backend = self._backend({anima38.SCRIPT_NAME: {"args": [{"bypass": True}]}})
                settings = {ap.SETTINGS_KEY: ap.envelope({anima38.SCRIPT_NAME: a38, PAG: pag_block()},
                                                         source="t2i_panel", backend=ap.BACKEND_COMFY, model=model)}
                if expected:
                    payload = self._compiled_payload(backend, settings, "sam3")
                    self.assertEqual(payload["alwayson_scripts"][anima38.SCRIPT_NAME], a38)
                else:
                    with self.assertLogs("comfyui", "INFO") as logs:
                        payload = self._compiled_payload(backend, settings, "sam3")
                    self.assertNotIn(anima38.SCRIPT_NAME, payload["alwayson_scripts"])
                    self.assertTrue(any(anima38.SCRIPT_NAME in line for line in logs.output))
                self.assertIn(PAG, payload["alwayson_scripts"])                  # 다른 전달 블록은 그대로

    def test_dd_hires_is_not_propagated_and_empty_envelope_clears_guidance(self):
        backend = self._backend({PAG: pag_block()})
        settings = {ap.SETTINGS_KEY: envelope({anima_guidance.SCRIPT_DETAIL_DAEMON: dd_block(hires=True)})}
        payload = self._compiled_payload(backend, settings, "adetailer")
        self.assertEqual(set(payload["alwayson_scripts"]), {"ADetailer"})

    def test_lora_syntax_loads_the_lora_but_never_reaches_the_sam3_prompt_text(self):
        """(P7-R1) D2 로 Refine 이 물려받은 LoRA 스택, 단독 SAM3 가 물려받은 T2I 프롬프트의 LoRA 태그 —
        로더 체인은 LoRA 를 걸고, SAM3 노드의 inpaint_prompt(노드 안에서 CLIPTextEncode 그대로)에는 ``<lora:>``
        문법이 남지 않는다(Forge SAM3 p2 는 process_images 가 positive 의 추가 네트워크 태그를 뗀다).
        (SAM3 LoRA 분기) 네거티브의 태그는 Forge 가 파싱하지 않아 걸리지도 떼어지지도 않는다 — 글자 그대로 남고 로드되지
        않는다(예전 기대값은 네거티브에서도 뗐다)."""
        from types import SimpleNamespace

        from ui.aux_pass_snapshot import inherit_refine_prompts

        def text(value):
            return SimpleNamespace(toPlainText=lambda: value)

        host = SimpleNamespace(total_prompt_display=text("1girl, solo, red shirt"), neg_prompt_text=text("bad"),
                               _vue_lora_entries=[{"name": "ink", "weight": 0.8, "enabled": True}],
                               _is_krea2_generation=lambda: False)
        refine = inherit_refine_prompts(host, {"target": "shirt", "replacement": "blue shirt"})
        self.assertIn("<lora:ink:0.80>", refine["main_prompt"])                  # D2 전제
        cases = (("refine", "ForgeNeoSAM3Refine", refine, "blue shirt"),
                 ("sam3", "ForgeNeoSAM3Detailer",
                  {"sam3_prompt": "face", "sam3_negative_prompt": "lowres, <lora:ink:0.3>"}, "portrait"))
        for kind, node_class, settings, expected in cases:
            with self.subTest(kind=kind):
                backend = self._backend({})
                backend._last_generation_context["payload"]["prompt"] = "portrait, <lora:ink:0.8>"
                getattr(backend, kind)(_png(), settings)
                graph = backend._queue_and_wait.call_args.args[0]
                loras = [node["inputs"] for node in graph.values()
                         if node["class_type"] == "ForgeNeoAnimaLoraLoader"]
                self.assertEqual([(lora["lora_name"], lora["strength_model"]) for lora in loras],
                                 [("styles/ink.safetensors", 0.8)])
                [detail] = [node["inputs"] for node in graph.values() if node["class_type"] == node_class]
                self.assertNotIn("<lora", detail["inpaint_prompt"].casefold())
                self.assertEqual(detail["negative_prompt"], settings.get("sam3_negative_prompt",
                                                                         detail["negative_prompt"]))
                self.assertIn(expected, detail["inpaint_prompt"])

    def test_sam3_prompt_text_is_kept_byte_identical_except_the_lora_tags(self):
        """(P7 검토 R2-2) R1 수정은 ``<lora:>`` 문법만 뗀다. LoRA 태그가 없는 SAM3 프롬프트는 노드까지 바이트 그대로
        간다(쉼표·줄바꿈·앞뒤 공백을 다듬으면 CLIP 토큰이 P7 전·Forge 와 달라진다). 태그가 있으면 Forge
        ``modules/extra_networks.py`` parse_prompt 처럼 태그만 떼고 구분자는 남긴다. 단독 SAM3(사용자 프롬프트 ·
        기본값 = 저장된 T2I 프롬프트)·Refine·생성 안 SAM3 가 모두 같은 노드 입력을 지난다.
        (SAM3 LoRA 분기) 떼는 것은 인페인트(positive) 프롬프트뿐이다 — Forge 는 네거티브를 파싱하지 않아 태그가 글자 그대로
        인코딩되므로 negative_prompt 는 태그까지 그대로다(예전 기대값 '\\nlowres, ' 는 네거티브에서도 뗐다)."""
        from core import sam3_args
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        from core.refine_prompt import build_refine_prompts
        from tests.test_comfy_workflow_compiler import _capabilities

        inpaint, negative = "detailed face, sharp eyes, ", "\nlowres, , blurry,\n"
        # Refine 은 build_refine_prompts 가 스스로 다듬는다 — 남는 차이는 D2 로 물려받은 LoRA 스택 앞의 구분자
        refine_settings = {"target": "shirt", "replacement": "blue shirt", "negative": "wrinkles",
                           "main_prompt": "1girl, solo, <lora:ink:0.80>", "main_negative": "bad anatomy"}
        refine = build_refine_prompts(main_prompt=refine_settings["main_prompt"],
                                      main_negative=refine_settings["main_negative"],
                                      target=refine_settings["target"], replacement=refine_settings["replacement"],
                                      negative=refine_settings["negative"], inherit_main=True, inherit_negative=True)
        self.assertEqual(refine["prompt"], "1girl, solo, <lora:ink:0.80>")      # 전제 — Forge p2 는 '1girl, solo, '
        saved_prompt, saved_negative = "masterpiece, , 1girl,\n", " bad anatomy, "
        cases = (
            ("sam3 user prompt", "sam3", "ForgeNeoSAM3Detailer",
             {"sam3_prompt": "face", "sam3_inpaint_prompt": inpaint, "sam3_negative_prompt": negative},
             (inpaint, negative)),
            ("sam3 default = saved T2I prompt", "sam3", "ForgeNeoSAM3Detailer", {"sam3_prompt": "face"},
             (saved_prompt, saved_negative)),
            ("refine (inherited LoRA stack)", "refine", "ForgeNeoSAM3Refine", refine_settings,
             ("1girl, solo, ", refine["negative_prompt"])),
            ("tag only, separators kept (Forge parse_prompt); negative untouched", "sam3", "ForgeNeoSAM3Detailer",
             {"sam3_prompt": "face", "sam3_inpaint_prompt": "a, <lora:ink:0.3>, b ",
              "sam3_negative_prompt": "<LORA:ink:0.5>\nlowres, "},
             ("a, , b ", "<LORA:ink:0.5>\nlowres, ")),
        )
        for label, kind, node_class, settings, expected in cases:
            with self.subTest(label):
                backend = self._backend({})
                backend._last_generation_context["payload"].update(prompt=saved_prompt,
                                                                   negative_prompt=saved_negative)
                getattr(backend, kind)(_png(), dict(settings))
                graph = backend._queue_and_wait.call_args.args[0]
                [detail] = [node["inputs"] for node in graph.values() if node["class_type"] == node_class]
                self.assertEqual((detail["inpaint_prompt"], detail["negative_prompt"]), expected)
        with self.subTest("in-generation SAM3"):
            scripts = sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face",
                                                "sam3_inpaint_prompt": inpaint, "sam3_negative_prompt": negative})
            graph = ComfyWorkflowCompiler(_capabilities()).compile(
                "txt2img", "checkpoint.safetensors", {"prompt": "portrait", "alwayson_scripts": scripts})
            [detail] = [node["inputs"] for node in graph.values() if node["class_type"] == "ForgeNeoSAM3Detailer"]
            self.assertEqual((detail["inpaint_prompt"], detail["negative_prompt"]), (inpaint, negative))

    def test_real_compile_gets_the_dd_node(self):
        """G7-2 의 그래프 부분(GPU 없이): 봉투의 DD 가 단독 ADetailer 그래프에 노드로 들어간다."""
        backend = self._backend({})
        backend._standalone_detail(_png(), {ap.SETTINGS_KEY: envelope({
            anima_guidance.SCRIPT_DETAIL_DAEMON: dd_block()})}, "adetailer")
        graph = backend._queue_and_wait.call_args.args[0]
        classes = [node["class_type"] for node in graph.values()]
        self.assertIn("ForgeNeoAnimaDetailDaemon", classes)


if __name__ == "__main__":
    unittest.main()
