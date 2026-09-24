"""ADetailer model names are resolved to Impact detector choices before /prompt (감사 #118)."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests.test_comfy_workflow_compiler import _capabilities, _choice, _node


def _payload(*models):
    slots = [{"ad_tab_enable": True, "ad_model": model} for model in models]
    return {"alwayson_scripts": {"ADetailer": {"args": [True, False, *slots]}}}


def _detector_capabilities(*choices):
    capabilities = _capabilities()
    capabilities["UltralyticsDetectorProvider"]["input"]["required"]["model_name"] = _choice(*choices)
    return capabilities


def _slot_model(graph, index=0):
    _id, node = _node(graph, "ForgeNeoADetailer", index)
    return json.loads(node["inputs"]["settings_json"])["model_name"]


class AdetailerModelResolutionTests(unittest.TestCase):
    def test_forge_names_map_to_bbox_and_seg_models_prefer_segm(self):
        compiler = ComfyWorkflowCompiler(_detector_capabilities(
            "bbox/face_yolov8n.pt", "bbox/person_yolov8n-seg.pt",
            "segm/person_yolov8n-seg.pt", "bbox/hands/hand_yolov8n.pt",
        ))
        graph = compiler.compile(
            "txt2img", "checkpoint.safetensors",
            _payload("face_yolov8n.pt", "person_yolov8n-seg.pt", "hand_yolov8n.pt"),
        )
        self.assertEqual(
            [_slot_model(graph, i) for i in range(3)],
            ["bbox/face_yolov8n.pt", "segm/person_yolov8n-seg.pt", "bbox/hands/hand_yolov8n.pt"],
        )

    def test_seg_model_only_in_bbox_folder_stays_bbox_and_stem_is_accepted(self):
        compiler = ComfyWorkflowCompiler(_detector_capabilities("bbox/person_yolov8n-seg.pt"))
        graph = compiler.compile("txt2img", "checkpoint.safetensors", _payload("person_yolov8n-seg"))
        self.assertEqual(_slot_model(graph), "bbox/person_yolov8n-seg.pt")

    def test_prefixed_names_must_exist_exactly(self):
        compiler = ComfyWorkflowCompiler(_detector_capabilities("bbox/face_yolov8n.pt"))
        graph = compiler.compile("txt2img", "checkpoint.safetensors", _payload("BBOX/face_yolov8n.pt"))
        self.assertEqual(_slot_model(graph), "bbox/face_yolov8n.pt")
        with self.assertRaisesRegex(WorkflowCompileError, "segm/face_yolov8n.pt"):
            compiler.compile("txt2img", "checkpoint.safetensors", _payload("segm/face_yolov8n.pt"))

    def test_mediapipe_and_missing_models_fail_before_queueing(self):
        compiler = ComfyWorkflowCompiler(_detector_capabilities("bbox/face_yolov8n.pt"))
        for name in ("mediapipe_face_full", "mediapipe_face_short", "hand_yolov9c.pt"):
            with self.subTest(name=name), self.assertRaisesRegex(WorkflowCompileError, "ADetailer 슬롯 1"):
                compiler.compile("txt2img", "checkpoint.safetensors", _payload(name))

    def test_missing_impact_providers_fail_before_queueing(self):
        for missing in ("UltralyticsDetectorProvider", "FaceDetailer"):
            capabilities = _capabilities()
            del capabilities[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(WorkflowCompileError, missing):
                ComfyWorkflowCompiler(capabilities).compile(
                    "txt2img", "checkpoint.safetensors", _payload("face_yolov8n.pt"),
                )

    def test_postprocess_uses_the_same_resolution(self):
        graph = ComfyWorkflowCompiler(_detector_capabilities("segm/person_yolov8n-seg.pt")).compile_postprocess(
            "checkpoint.safetensors", _payload("person_yolov8n-seg.pt"), uploaded_image="source.png",
        )
        self.assertEqual(_slot_model(graph), "segm/person_yolov8n-seg.pt")

    def test_none_model_slots_are_skipped_like_forge(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", _payload("None", "face_yolov8n.pt"),
        )
        nodes = [node for node in graph.values() if node["class_type"] == "ForgeNeoADetailer"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(_slot_model(graph), "bbox/face_yolov8n.pt")

    def test_offline_compile_keeps_the_forge_name_for_the_node_default(self):
        graph = ComfyWorkflowCompiler().compile("txt2img", "model.safetensors", _payload("face_yolov8n.pt"))
        self.assertEqual(_slot_model(graph), "face_yolov8n.pt")


class AdetailerNodeModelNameTests(unittest.TestCase):
    def test_node_passes_a_resolved_segm_choice_through_unchanged(self):
        image = mock.Mock(shape=(1, 8, 8, 3), dtype="float32", device="cpu")
        seen = {}

        class Segm:
            def detect(self, *args, **kwargs):
                return None

        class FakeMask:
            def __gt__(self, _other):
                return self

            def any(self, **_kwargs):
                return self

            def sum(self):
                return self

            def item(self):
                return 1

        segm = Segm()

        def invoke(node_type, **kwargs):
            if node_type == "UltralyticsDetectorProvider":
                seen["model"] = kwargs["args"][0]
                return object(), segm
            if node_type == "FaceDetailer":
                seen["segm"] = kwargs["kwargs"]["segm_detector_opt"]
                return image, None, None, FakeMask()
            raise AssertionError(node_type)

        with (
            mock.patch.object(generation, "require_torch", return_value=object()),
            mock.patch.object(generation, "_image_tensor", return_value=image),
            mock.patch.object(generation, "invoke_provider", side_effect=invoke),
        ):
            generation.ForgeNeoADetailer().detail(
                image, object(), object(), object(), object(), object(), True,
                json.dumps({"ad_model": "person_yolov8n-seg.pt", "model_name": "segm/person_yolov8n-seg.pt"}),
            )
        self.assertEqual(seen["model"], "segm/person_yolov8n-seg.pt")
        self.assertIs(seen["segm"], segm)


if __name__ == "__main__":
    unittest.main()
