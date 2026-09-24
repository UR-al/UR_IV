from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import mask_ops
from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes
from tests._optional_deps import bind_torch, requires_torch

# torch 는 지연 import 한다(tests/_optional_deps). 최상위 import 는 discovery 마다 — 즉 .py 편집마다
# 도는 --quick 훅마다 — torch 로드 시간을 더했다. 아래 헬퍼·가짜 모델이 모듈 전역 torch 를 쓰므로
# torch 텐서 테스트 클래스(TestComfySam3Nodes)의 setUpClass 가 채운다.
torch = None


def _image(batch=1, height=12, width=14, value=0.0):
    return torch.full((batch, height, width, 3), value, dtype=torch.float32)


class _FakeModel:
    def __init__(self):
        self.targets = []

    def to(self, target):
        self.targets.append(str(target))
        return self


class _DeviceModel(_FakeModel):
    """parameters() 가 현재 장치를 보고하는 모델 — 캐시 적중 시 '원래 장치'(home)로의 복귀를
    CPU 오프로드와 구분해 검증한다. 'meta' 를 GPU 대신 쓴다(실제 GPU 불필요)."""

    def __init__(self, device="meta"):
        super().__init__()
        self.device = torch.device(device)
        self.fail_on: set[str] = set()

    def parameters(self):
        yield torch.empty(0, device=self.device)

    def to(self, target):
        self.targets.append(str(target))
        if str(target) in self.fail_on:
            raise RuntimeError("CUDA out of memory (simulated)")
        self.device = torch.device(target)
        return self


class _FakeSegmenter:
    calls = []

    @classmethod
    def execute(cls, sam3_model, images, prompt, threshold=0.3,
                keep_model_loaded=False, add_background="none", detection_limit=-1,
                coordinates_positive=None, coordinates_negative=None, bboxes=None,
                mask=None):
        cls.calls.append((prompt, threshold, detection_limit, keep_model_loaded))
        output = torch.zeros((images.shape[0], images.shape[1], images.shape[2]))
        locations = {"face": (1, 1), "eyes": (2, 2), "hand": (7, 7), "protect": (1, 1)}
        y, x = locations[prompt]
        output[:, y:y + 2, x:x + 2] = 1
        return output, images, output[:, None], [[[x, y, x + 2, y + 2]]], [[0.9]]


class TestComfySam3NodePureContracts(unittest.TestCase):
    """torch 없이 도는 계약·산술 — --quick(훅)에서도 돈다(워크플로 컴파일러가 기대는 노드 계약 포함)."""

    def test_prompt_grammar_preserves_or_and_sequential_groups(self):
        self.assertEqual(
            mask_ops.split_prompt_groups(" face, eyes | hair / hand; fingers\narm "),
            [["face", "eyes", "hair"], ["hand", "fingers", "arm"]],
        )
        self.assertEqual(mask_ops.split_prompt_groups(" / , ; \n"), [])

    def test_unload_hook_is_a_no_op_outside_comfyui_and_installed_by_the_pack(self):
        with mock.patch.dict(sys.modules, {"comfy": None, "comfy.model_management": None}):
            self.assertFalse(sam3_nodes.install_unload_release_hook())
        self.assertFalse(sam3_nodes.install_unload_release_hook(SimpleNamespace()))
        # 팩 __init__ 이 등록 시점에 훅을 건다(ComfyUI 밖에서는 위처럼 아무 일도 안 한다).
        import comfy_custom_nodes.ai_studio_forge_parity as pack

        self.assertIs(pack._install_sam3_unload_release_hook, sam3_nodes.install_unload_release_hook)

    def test_enabled_detection_fails_clearly_without_easy_provider(self):
        with mock.patch.object(sam3_nodes, "_node_mappings", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "Easy SAM3.*not installed/loaded"):
                sam3_nodes._resolve_easy_node(
                    sam3_nodes._EASY_SEGMENTATION_KEYS, "image segmentation"
                )

    def test_override_external_control_copies_and_strips_metadata(self):
        tensor = object()
        source = [[tensor, {"control": "old", "control_apply_to_uncond": True, "pooled_output": "keep"}]]
        result = sam3_nodes._without_existing_control(source)
        self.assertIs(result[0][0], tensor)
        self.assertEqual(result[0][1], {"pooled_output": "keep"})
        self.assertIn("control", source[0][1])

    def test_expand_crop_region_matches_forge_integer_arithmetic(self):
        # Forge modules/masking.py 의 expand_crop_region 을 그대로 옮긴 값들.
        cases = (
            # 128x32 크롭 → 512x512 처리: 세로로 넓혀 128x128
            (((100, 200, 228, 232), 512, 512, 1024, 1024), (100, 152, 228, 280)),
            # 아래 가장자리에 닿으면 위로 밀어 올린다
            (((10, 990, 138, 1022), 512, 512, 1024, 1024), (10, 896, 138, 1024)),
            # 이미지가 모자라면 이미지 전체 높이로 잘린다
            (((0, 0, 200, 50), 512, 512, 200, 100), (0, 0, 200, 100)),
            # 세로로 긴 크롭은 가로로 넓힌다 (2:3 처리 크기)
            (((50, 50, 82, 178), 512, 768, 1024, 1024), (24, 50, 109, 178)),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(mask_ops.expand_crop_region(*arguments), expected)
        with self.assertRaises(ValueError):
            mask_ops.expand_crop_region((5, 5, 5, 9), 512, 512, 64, 64)

    def test_fit_sample_size_never_stretches_the_crop(self):
        self.assertEqual(mask_ops.fit_sample_size(14, 21, 64, 96), (64, 96))
        # 이미지 경계 때문에 종횡비를 못 맞춘 크롭은 비율을 유지한 채 맞춘다
        self.assertEqual(mask_ops.fit_sample_size(200, 100, 512, 512), (512, 256))
        self.assertEqual(mask_ops.fit_sample_size(2000, 1000, 1024, 1024), (1024, 512))

    def test_node_contracts_are_stable_for_workflow_compiler(self):
        self.assertEqual(
            sam3_nodes.ForgeNeoSAM3Mask.RETURN_NAMES,
            ("selected_mask", "combined_mask", "individual_masks", "overlay", "boxes", "scores", "artifacts_json"),
        )
        self.assertEqual(
            sam3_nodes.ForgeNeoSAM3Detailer.RETURN_NAMES,
            ("image", "applied_mask", "report_json"),
        )
        self.assertTrue(
            {"sam3_model", "manual_mask"}
            <= set(sam3_nodes.ForgeNeoSAM3Mask.INPUT_TYPES()["optional"])
        )
        self.assertTrue(
            {
                "model", "clip", "vae", "positive", "negative", "inpaint_prompt",
                "controlnet_enable", "controlnet_override_external",
                "controlnet_settings_json", "restore_face", "restore_face_settings_json",
            }
            <= set(sam3_nodes.ForgeNeoSAM3Detailer.INPUT_TYPES()["required"])
        )
        self.assertTrue(
            {"control_net", "control_image", "face_detector"}
            <= set(sam3_nodes.ForgeNeoSAM3Detailer.INPUT_TYPES()["optional"])
        )


@requires_torch
class TestComfySam3Nodes(unittest.TestCase):
    """텐서 동작 — torch 필요. --quick 은 이 클래스를 거르고(torch 를 올리지 않는다) 전체 실행과
    훅의 --with-torch(comfy_custom_nodes/ 편집 등)에서 돈다."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def test_ensure_mask_resizes_and_broadcasts_without_batch_guessing(self):
        source = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = mask_ops.ensure_mask(source, height=4, width=6, batch=2)
        self.assertEqual(result.shape, (2, 4, 6))
        self.assertTrue(torch.equal(result[0], result[1]))
        with self.assertRaisesRegex(ValueError, "does not match IMAGE batch"):
            mask_ops.ensure_mask(torch.zeros((3, 4, 6)), batch=2)

    def test_bicubic_overshoot_is_clamped_not_divided_by_255(self):
        # 흰 배경 선화를 bicubic 으로 키우면 max 가 ~1.2 가 된다. 예전 'max>1 이면 /255'
        # 휴리스틱은 이 값을 max 0.005 의 거의 검은 이미지로 만들었다.
        line_art = torch.ones((1, 8, 8, 3))
        line_art[:, 3:5, :, :] = 0.0
        upscaled = torch.nn.functional.interpolate(
            line_art.permute(0, 3, 1, 2), scale_factor=1.5, mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        self.assertGreater(float(upscaled.max()), 1.05)
        normalized = mask_ops.ensure_image(upscaled)
        self.assertAlmostEqual(float(normalized.max()), 1.0, places=6)
        self.assertGreater(float(normalized.mean()), 0.5)
        mask = mask_ops.ensure_mask(upscaled[..., 0])
        self.assertAlmostEqual(float(mask.max()), 1.0, places=6)
        self.assertGreater(float(mask.mean()), 0.5)

        # Detailer 가 disabled/빈 마스크일 때 돌려주는 image_value 도 어두워지지 않는다.
        output, _, _ = sam3_nodes.ForgeNeoSAM3Detailer().detail(
            upscaled, torch.zeros((1, 12, 12)), object(), object(), object(),
            object(), object(), enabled=False,
        )
        self.assertTrue(torch.allclose(output, upscaled.clamp(0.0, 1.0)))

    def test_integer_and_bool_inputs_are_scaled_by_dtype(self):
        byte_image = torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)
        byte_image[0, 0, 0] = 0
        self.assertTrue(torch.equal(
            mask_ops.ensure_image(byte_image), byte_image.float() / 255.0,
        ))
        # 값이 1 이하인 uint8 도 byte 스케일이다 (max 로 추측하지 않는다).
        dark_bytes = torch.ones((1, 2, 2, 3), dtype=torch.uint8)
        self.assertAlmostEqual(float(mask_ops.ensure_image(dark_bytes).max()), 1 / 255, places=6)

        bool_mask = torch.tensor([[[True, False], [False, True]]])
        self.assertTrue(torch.equal(mask_ops.ensure_mask(bool_mask), bool_mask.float()))
        int_rgba_mask = torch.zeros((1, 2, 2, 4), dtype=torch.int32)
        int_rgba_mask[..., 3] = 255
        self.assertTrue(torch.equal(
            mask_ops.ensure_mask(int_rgba_mask), torch.ones((1, 2, 2)),
        ))
        # 0..255 스케일로 들어온 float 데이터는 여전히 인식한다.
        float_bytes = torch.full((1, 2, 2), 255.0)
        self.assertTrue(torch.equal(mask_ops.ensure_mask(float_bytes), torch.ones((1, 2, 2))))

    def test_pack_bicubic_resize_output_stays_in_unit_range(self):
        from comfy_custom_nodes.ai_studio_forge_parity import generation

        line_art = torch.ones((1, 8, 8, 3))
        line_art[:, 3:5, :, :] = 0.0
        for fit in ("stretch", "crop", "contain"):
            with self.subTest(fit=fit):
                resized = generation.resize_image(line_art, 13, 12, fit)
                self.assertLessEqual(float(resized.max()), 1.0)
                self.assertGreaterEqual(float(resized.min()), 0.0)

    def test_refine_order_matches_forge_extension(self):
        calls = []

        def stage(name):
            def apply(value, *args, **kwargs):
                calls.append(name)
                return value
            return apply

        with (
            mock.patch.object(mask_ops, "convex_hull", side_effect=stage("hull")),
            mock.patch.object(mask_ops, "edge_aware_outline", side_effect=stage("outline")),
            mock.patch.object(mask_ops, "dilate", side_effect=stage("dilation")),
        ):
            mask_ops.refine_generated_mask(
                torch.zeros((1, 4, 4)), _image(height=4, width=4),
                use_convex_hull=True, outline_pixels=2, dilation_pixels=3,
            )
        self.assertEqual(calls, ["hull", "outline", "dilation"])

    def test_real_convex_hull_dilation_and_blur_preserve_shape(self):
        source = torch.zeros((1, 9, 9))
        source[0, 2, 2:7] = 1
        source[0, 2:7, 2] = 1
        hull = mask_ops.convex_hull(source)
        expanded = mask_ops.dilate(hull, 1)
        blurred = mask_ops.gaussian_blur(expanded, 2)
        self.assertEqual(blurred.shape, source.shape)
        self.assertGreater(hull.sum().item(), source.sum().item())
        self.assertGreater(expanded.sum().item(), hull.sum().item())
        self.assertGreater(blurred[0, 0, 0].item(), 0.0)
        self.assertLessEqual(blurred.max().item(), 1.0)

    def test_intersection_falls_back_to_manual_per_image(self):
        generated = torch.zeros((2, 6, 6))
        generated[0, 1:4, 1:4] = 1
        manual = torch.zeros((2, 6, 6))
        manual[0, 2:5, 2:5] = 1
        manual[1, 4:6, 4:6] = 1
        selected = mask_ops.select_mask_groups(
            [generated], manual, "intersection",
            reference=_image(batch=2, height=6, width=6),
        )[0]
        self.assertEqual(selected[0].sum().item(), 4)
        self.assertTrue(torch.equal(selected[1], manual[1]))

    def test_exclusion_then_invert_keeps_combined_semantics(self):
        first = torch.zeros((1, 5, 5))
        second = torch.zeros((1, 5, 5))
        protected = torch.zeros((1, 5, 5))
        first[:, 1:3, 1:3] = 1
        second[:, 3:5, 3:5] = 1
        protected[:, 1, 1] = 1
        groups = mask_ops.subtract_exclusion([first, second], protected)
        combined, individual = mask_ops.finish_masks(
            groups, _image(height=5, width=5), blur_pixels=0, invert=True
        )
        self.assertEqual(individual.shape, (2, 5, 5))
        self.assertEqual(combined[0, 1, 1].item(), 1)
        self.assertEqual(combined[0, 2, 2].item(), 0)
        self.assertEqual(combined[0, 4, 4].item(), 0)

    def test_empty_detection_is_not_inverted_into_a_full_image_mask(self):
        source = _image(height=5, width=7)
        combined, individual = mask_ops.finish_masks(
            [], source, blur_pixels=0, invert=True
        )
        self.assertEqual(combined.sum().item(), 0)
        self.assertEqual(individual.sum().item(), 0)

        combined, individual = mask_ops.finish_masks(
            [torch.zeros((1, 5, 7))], source, blur_pixels=0, invert=True
        )
        self.assertEqual(combined.sum().item(), 0)
        self.assertEqual(individual.sum().item(), 0)

    def test_mask_node_runs_or_groups_sequential_groups_and_exclusion(self):
        _FakeSegmenter.calls = []
        with mock.patch.object(sam3_nodes, "_resolve_easy_node", return_value=_FakeSegmenter):
            result = sam3_nodes.ForgeNeoSAM3Mask().segment(
                _image(height=10, width=10), prompt="face, eyes / hand",
                exclude_prompt="protect", mask_mode="Individual",
                mask_source="generated", threshold=0.55, detection_limit=3,
                mask_blur=0, save_artifacts=False, unload_after=False,
                sam3_model={"model": _FakeModel(), "device": "cpu", "segmentor": "image"},
            )
        selected, combined, individual, overlay, boxes, scores, report_json = result
        self.assertEqual([call[0] for call in _FakeSegmenter.calls], ["face", "eyes", "hand", "protect"])
        self.assertTrue(all(call[1:3] == (0.55, 3) for call in _FakeSegmenter.calls))
        self.assertEqual(selected.shape, individual.shape)
        self.assertEqual(selected.shape, (2, 10, 10))
        self.assertEqual(combined.shape, (1, 10, 10))
        self.assertEqual(combined[0, 1, 1].item(), 0)
        self.assertEqual(combined[0, 3, 3].item(), 1)
        self.assertEqual(overlay.shape, (1, 10, 10, 3))
        self.assertEqual(boxes, [[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0], [7.0, 7.0, 9.0, 9.0]])
        self.assertEqual(scores, [0.9, 0.9, 0.9])
        self.assertEqual(json.loads(report_json)["prompt_groups"], [["face", "eyes"], ["hand"]])

    def _cached_segment(self, checkpoint, **kwargs):
        # Comfy 의 folder_paths 스텁: 절대 경로 체크포인트 등록/복원 계약만 흉내 낸다.
        folder_paths = SimpleNamespace(
            folder_names_and_paths={"sam3": ([], {".pt"})},
            get_full_path=lambda folder, name: None,
        )
        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}), mock.patch.object(
            sam3_nodes, "_resolve_easy_node",
            side_effect=lambda keys, purpose: (
                self._loader if keys is sam3_nodes._EASY_LOADER_KEYS else _FakeSegmenter
            ),
        ):
            return sam3_nodes.ForgeNeoSAM3Mask().segment(
                _image(height=10, width=10), prompt="face", mask_blur=0,
                save_artifacts=False, checkpoint=str(checkpoint), device="cpu",
                precision="fp32", **kwargs,
            )

    def _make_loader(self, model_factory=_FakeModel):
        test = self

        class Loader:
            loads = []

            @classmethod
            def execute(cls, model, segmentor, device, precision):
                bundle = {"model": model_factory(), "device": device, "segmentor": segmentor}
                cls.loads.append((model, bundle))
                return (bundle,)

        test._loader = Loader
        return Loader

    def test_node_loaded_sam3_bundle_is_reused_from_cpu_between_runs(self):
        sam3_nodes.clear_sam3_cache()
        self.addCleanup(sam3_nodes.clear_sam3_cache)
        # 모델이 로드된 장치(home)는 'meta' — CPU 오프로드 대상과 달라야 복귀를 구분할 수 있다.
        loader = self._make_loader(_DeviceModel)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"weights")
            first = json.loads(self._cached_segment(checkpoint)[-1])
            second = json.loads(self._cached_segment(checkpoint)[-1])
            self.assertEqual(len(loader.loads), 1)
            self.assertEqual((first["model_cache"], second["model_cache"]), ("miss", "hit"))
            model = loader.loads[0][1]["model"]
            # unload_after: 사용 뒤 CPU 로 내려 두고(miss), 다음 실행은 원래 장치(meta)로 되돌린
            # 뒤 쓰고 다시 CPU 로 내린다(hit). 복귀가 오프로드 장치로 가거나 빠지면 여기서 걸린다.
            self.assertEqual(model.targets, ["cpu", "meta", "cpu"])
            self.assertEqual(model.device, torch.device("cpu"))

            # 체크포인트 파일이 바뀌면 메모리 사본을 쓰지 않는다.
            checkpoint.write_bytes(b"replaced weights")
            third = json.loads(self._cached_segment(checkpoint)[-1])
            self.assertEqual(third["model_cache"], "miss")
            self.assertEqual(len(loader.loads), 2)

            # cache_model=False 는 매번 로드하고 보관본도 비운다.
            disabled = json.loads(self._cached_segment(checkpoint, cache_model=False)[-1])
            self.assertEqual(disabled["model_cache"], "disabled")
            self.assertEqual(len(loader.loads), 3)
            again = json.loads(self._cached_segment(checkpoint)[-1])
            self.assertEqual(again["model_cache"], "miss")
            self.assertEqual(len(loader.loads), 4)

    def test_unload_after_false_keeps_cached_bundle_on_device(self):
        sam3_nodes.clear_sam3_cache()
        self.addCleanup(sam3_nodes.clear_sam3_cache)
        loader = self._make_loader(_DeviceModel)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"weights")
            report = json.loads(self._cached_segment(checkpoint, unload_after=False)[-1])
            self._cached_segment(checkpoint, unload_after=False)
        self.assertEqual(len(loader.loads), 1)
        self.assertFalse(report["unloaded"])
        # 오프로드 없이 같은 장치(meta)로 되돌리는 no-op 만 있다.
        model = loader.loads[0][1]["model"]
        self.assertEqual(model.targets, ["meta"])
        self.assertEqual(model.device, torch.device("meta"))

    def test_failed_restore_on_cache_hit_drops_the_half_moved_bundle(self):
        sam3_nodes.clear_sam3_cache()
        self.addCleanup(sam3_nodes.clear_sam3_cache)
        loader = self._make_loader(_DeviceModel)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"weights")
            self._cached_segment(checkpoint)  # miss → CPU 에 보관
            model = loader.loads[0][1]["model"]
            model.fail_on = {"meta"}  # 원래 장치로 되돌리다 OOM
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self._cached_segment(checkpoint)
            # 반쯤 옮겨진 번들은 캐시에 남지 않고, 해제 전에 CPU 로 내려 VRAM 을 비운다.
            self.assertEqual(sam3_nodes._SAM3_CACHE, {})
            self.assertEqual(model.targets, ["cpu", "meta", "cpu"])
            # 다음 실행은 믿을 수 없는 사본 대신 새로 로드한다.
            report = json.loads(self._cached_segment(checkpoint)[-1])
        self.assertEqual(report["model_cache"], "miss")
        self.assertEqual(len(loader.loads), 2)

    def test_comfy_unload_all_models_also_releases_the_cached_bundle(self):
        sam3_nodes.clear_sam3_cache()
        self.addCleanup(sam3_nodes.clear_sam3_cache)
        calls = []

        def unload_all_models():
            calls.append("unload")

        management = SimpleNamespace(unload_all_models=unload_all_models)
        self.assertTrue(sam3_nodes.install_unload_release_hook(management))
        # 두 번 감싸지 않는다(팩 재로드·중복 호출).
        self.assertFalse(sam3_nodes.install_unload_release_hook(management))
        self.assertEqual(management.unload_all_models.__wrapped__, unload_all_models)
        loader = self._make_loader(_DeviceModel)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"weights")
            self._cached_segment(checkpoint)
            self.assertTrue(sam3_nodes._SAM3_CACHE)
            # ComfyUI /free(앱의 '생성 후 모델 언로드')가 부르는 경로.
            management.unload_all_models()
            self.assertEqual(calls, ["unload"])
            self.assertEqual(sam3_nodes._SAM3_CACHE, {})
            report = json.loads(self._cached_segment(checkpoint)[-1])
        self.assertEqual(report["model_cache"], "miss")
        self.assertEqual(len(loader.loads), 2)

    def test_unload_hook_releases_even_when_comfy_unload_raises(self):
        sam3_nodes.clear_sam3_cache()
        self.addCleanup(sam3_nodes.clear_sam3_cache)

        def failing_unload():
            raise RuntimeError("comfy unload failed")

        management = SimpleNamespace(unload_all_models=failing_unload)
        sam3_nodes.install_unload_release_hook(management)
        self._make_loader(_DeviceModel)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"weights")
            self._cached_segment(checkpoint)
        with self.assertRaisesRegex(RuntimeError, "comfy unload failed"):
            management.unload_all_models()
        self.assertEqual(sam3_nodes._SAM3_CACHE, {})

    def test_clear_reports_whether_a_bundle_was_released(self):
        sam3_nodes.clear_sam3_cache()
        self.addCleanup(sam3_nodes.clear_sam3_cache)
        self.assertFalse(sam3_nodes.clear_sam3_cache())
        self._make_loader(_DeviceModel)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"weights")
            self._cached_segment(checkpoint)
        self.assertTrue(sam3_nodes.clear_sam3_cache())
        self.assertFalse(sam3_nodes.clear_sam3_cache())

    def test_external_sam3_model_is_never_offloaded_by_the_mask_node(self):
        external = {"model": _FakeModel(), "device": "cuda", "segmentor": "image"}
        with mock.patch.object(sam3_nodes, "_resolve_easy_node", return_value=_FakeSegmenter):
            result = sam3_nodes.ForgeNeoSAM3Mask().segment(
                _image(height=10, width=10), prompt="face", mask_blur=0,
                save_artifacts=False, unload_after=True, sam3_model=external,
            )
        self.assertEqual(external["model"].targets, [])
        report = json.loads(result[-1])
        self.assertFalse(report["unloaded"])
        self.assertEqual(report["model_cache"], "external")

    def test_easy_sam3_adapter_invokes_each_batch_image_independently(self):
        class UnevenProvider:
            calls = []

            @classmethod
            def execute(cls, sam3_model, images, prompt, **kwargs):
                cls.calls.append(tuple(images.shape))
                self_mask = torch.zeros((1, images.shape[1], images.shape[2]))
                if float(images.mean()) > 0.5:
                    self_mask[:, 2:4, 3:6] = 1
                    boxes, scores = [[[[3, 2, 6, 4]]]], [[[0.75]]]
                else:
                    # Current Easy-SAM3 no-detection sentinel.
                    boxes, scores = [[[[0, 0, 0, 0]]]], [[[0.0]]]
                return self_mask, images, self_mask[:, None], boxes, scores

        batch = torch.cat([
            _image(height=8, width=9, value=0.0),
            _image(height=8, width=9, value=1.0),
        ])
        detected, boxes, scores = sam3_nodes._detect_token(
            UnevenProvider, object(), batch, "face", 0.4, -1
        )
        self.assertEqual(UnevenProvider.calls, [(1, 8, 9, 3), (1, 8, 9, 3)])
        self.assertEqual(tuple(detected.shape), (2, 8, 9))
        self.assertEqual(detected[0].sum().item(), 0)
        self.assertEqual(detected[1].sum().item(), 6)
        self.assertEqual(boxes, [[3.0, 2.0, 6.0, 4.0]])
        self.assertEqual(scores, [0.75])

    def test_manual_mask_mode_does_not_require_easy_sam3(self):
        manual = torch.zeros((1, 8, 8))
        manual[:, 2:6, 2:6] = 1
        with mock.patch.object(
            sam3_nodes, "_resolve_easy_node", side_effect=AssertionError("provider should not load")
        ):
            result = sam3_nodes.ForgeNeoSAM3Mask().segment(
                _image(height=8, width=8), prompt="", mask_source="manual",
                manual_mask=manual, mask_blur=0, save_artifacts=False,
            )
        self.assertTrue(torch.equal(result[0], manual))
        self.assertEqual(json.loads(result[-1])["mask_source"], "manual")

    def test_unknown_controlnet_module_and_unknown_settings_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown ControlNet module"):
            sam3_nodes._prepare_control_hint(
                _image(), torch.ones((1, 12, 14)), "not_a_real_module", 512, -1, -1
            )
        with self.assertRaisesRegex(ValueError, "Unknown controlnet_settings_json keys"):
            sam3_nodes._settings_object(
                '{"typo": true}', sam3_nodes._CONTROLNET_EXTRA_DEFAULTS,
                "controlnet_settings_json",
            )

    def test_controlnet_control_priority_uses_advanced_provider_and_keeps_negative_uncontrolled(self):
        calls = []

        class FakeApply:
            @classmethod
            def apply_controlnet(cls, **kwargs):
                calls.append(kwargs)
                return "conditioned-positive", "conditioned-negative"

        with mock.patch.object(
            sam3_nodes, "_node_mappings",
            return_value={"ControlNetApplyAdvanced": FakeApply},
        ):
            positive, negative, report = sam3_nodes._apply_controlnet(
                "positive", "negative", object(), _image(), 1.0, 0.1, 0.9,
                object(), "ControlNet is more important",
            )
        self.assertEqual(positive, "conditioned-positive")
        self.assertEqual(negative, "negative")
        self.assertEqual(calls[0]["negative"], [])
        self.assertEqual(calls[0]["strength"], 0.825)
        self.assertEqual(report["translation"], "forge_positive_soft_negative_zero")

    def test_zero_controlnet_strength_is_a_provider_free_no_op(self):
        with mock.patch.object(
            sam3_nodes, "_node_mappings", side_effect=AssertionError("provider must not load")
        ):
            positive, negative, report = sam3_nodes._apply_controlnet(
                "positive", "negative", object(), _image(), 0.0, 0.0, 1.0,
                object(), "Balanced",
            )
        self.assertEqual((positive, negative), ("positive", "negative"))
        self.assertEqual(report["translation"], "disabled_zero_strength")
        self.assertEqual(report["effective_strength"], 0.0)

    def test_missing_or_semantically_different_control_preprocessor_fails_fast(self):
        with mock.patch.object(sam3_nodes, "_node_mappings", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "refusing to substitute"):
                sam3_nodes._prepare_control_hint(
                    _image(), torch.ones((1, 12, 14)), "canny", 512, -1, -1
                )
        with self.assertRaisesRegex(RuntimeError, "tile_colorfix.*unsupported"):
            sam3_nodes._prepare_control_hint(
                _image(), torch.ones((1, 12, 14)), "tile_colorfix", 512, -1, -1
            )

    def test_vae_padding_uses_reported_ratio_and_unpads_exactly(self):
        class RatioVAE:
            @staticmethod
            def spacial_compression_encode():
                return 8

        source = torch.linspace(0.0, 1.0, 7 * 13 * 3).reshape(1, 7, 13, 3)
        mask = torch.zeros((1, 7, 13))
        mask[:, 1:6, 2:11] = 1
        padded, padded_mask, padding = sam3_nodes._pad_for_vae(
            source, mask, RatioVAE()
        )
        self.assertEqual(tuple(padded.shape), (1, 8, 16, 3))
        self.assertEqual(tuple(padded_mask.shape), (1, 8, 16))
        self.assertEqual(padding, (1, 2, 0, 1))
        self.assertTrue(torch.equal(sam3_nodes._unpad_image(padded, padding), source))
        self.assertEqual(padded_mask.sum().item(), mask.sum().item())

    def test_four_fill_modes_keep_distinct_latent_semantics(self):
        class IdentityVAE:
            @staticmethod
            def encode(pixels):
                return pixels.movedim(-1, 1).clone()

        image = torch.linspace(0.0, 1.0, 8 * 8 * 3).reshape(1, 8, 8, 3)
        mask = torch.zeros((1, 8, 8))
        mask[:, 2:6, 2:6] = 1
        values = {
            mode: sam3_nodes._vae_encode_for_inpaint(
                IdentityVAE(), image, mask, 0, fill_mode=mode, seed=123
            )["samples"]
            for mode in ("fill", "original", "latent noise", "latent nothing")
        }
        latent_mask = mask.unsqueeze(1).bool()
        self.assertFalse(torch.equal(values["fill"], values["original"]))
        self.assertTrue(torch.equal(
            values["latent nothing"].masked_select(latent_mask.expand_as(values["latent nothing"])),
            torch.zeros_like(values["latent nothing"].masked_select(latent_mask.expand_as(values["latent nothing"]))),
        ))
        self.assertFalse(torch.equal(values["latent noise"], values["latent nothing"]))
        repeated = sam3_nodes._vae_encode_for_inpaint(
            IdentityVAE(), image, mask, 0, fill_mode="latent noise", seed=123
        )["samples"]
        self.assertTrue(torch.equal(values["latent noise"], repeated))

    def test_artifact_output_writes_all_declared_files(self):
        combined = torch.zeros((1, 5, 6))
        combined[:, 1:4, 2:5] = 1
        individuals = torch.cat([combined, torch.zeros_like(combined)], dim=0)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = mask_ops.save_mask_artifacts(
                temp_dir, combined=combined, individuals=individuals,
                overlay=mask_ops.make_overlay(_image(height=5, width=6), combined),
                prompt="face / hand", seed=17, metadata={"device": "cpu"},
            )
            paths = [result["combined_mask"], result["overlay"], result["metadata"], *result["individual_masks"]]
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_detailer_sequentially_inpaints_individual_masks(self):
        encoded_shapes = []
        sampled = []

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            encoded_shapes.append((pixels.shape, mask.shape, grow_mask_by))
            return {
                "samples": torch.zeros((pixels.shape[0], 4, pixels.shape[1] // 8, pixels.shape[2] // 8)),
                "test_image_shape": tuple(pixels.shape),
            }

        def fake_sample(model, seed, steps, cfg, sampler_name, scheduler,
                        positive, negative, latent, denoise, noise_multiplier):
            sampled.append((seed, steps, cfg, sampler_name, scheduler, denoise, noise_multiplier))
            return latent

        def fake_decode(vae, latent):
            return torch.ones(latent["test_image_shape"])

        masks = torch.zeros((2, 16, 16))
        masks[0, 2:5, 2:5] = 1
        masks[1, 10:13, 10:13] = 1
        with (
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=fake_sample),
            mock.patch.object(sam3_nodes, "_vae_decode", side_effect=fake_decode),
            mock.patch.object(sam3_nodes, "_encode_prompt", side_effect=lambda clip, text: ("encoded", text)),
        ):
            output, applied, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                _image(height=16, width=16), masks, object(), object(), object(),
                "positive-conditioning", "negative-conditioning",
                inpaint_prompt="detail prompt", negative_prompt="",
                mask_mode="Individual", seed=100, steps=9, cfg=4.5,
                sampler_name="euler", scheduler="normal", denoise=0.35,
                noise_multiplier=1.2, fill_mode="original", only_masked=True,
                mask_padding=1, use_custom_size=False, grow_mask_by=4,
            )
        self.assertEqual(len(encoded_shapes), 2)
        self.assertEqual(len(sampled), 2)
        self.assertEqual([item[0] for item in sampled], [100, 100])
        self.assertTrue(all(item[-1] == 1.2 for item in sampled))
        self.assertTrue(output[0, 3, 3].eq(1).all())
        self.assertTrue(output[0, 11, 11].eq(1).all())
        self.assertTrue(output[0, 0, 0].eq(0).all())
        self.assertEqual(applied.sum().item(), 18)
        self.assertEqual(
            [item["status"] for item in json.loads(report_json)["passes"]],
            ["sampled", "sampled"],
        )

    def test_detailer_pads_odd_full_frame_for_vae_then_unpads_output(self):
        class RatioVAE:
            encoded_shape = None

            @staticmethod
            def spacial_compression_encode():
                return 8

            def encode(self, pixels):
                self.encoded_shape = tuple(pixels.shape)
                return torch.zeros((pixels.shape[0], 4, pixels.shape[1] // 8, pixels.shape[2] // 8))

        vae = RatioVAE()
        source = _image(height=7, width=13)
        mask = torch.ones((1, 7, 13))
        with (
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(
                sam3_nodes, "_vae_decode",
                side_effect=lambda _vae, latent: torch.ones((1, 8, 16, 3)),
            ),
        ):
            output, _, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                source, mask, object(), object(), vae, object(), object(),
                only_masked=False, grow_mask_by=0,
            )
        self.assertEqual(vae.encoded_shape, (1, 8, 16, 3))
        self.assertEqual(tuple(output.shape), (1, 7, 13, 3))
        pass_report = json.loads(report_json)["passes"][0]
        self.assertEqual(pass_report["sample_size"], [13, 7])
        self.assertEqual(pass_report["vae_sample_size"], [16, 8])
        self.assertEqual(pass_report["vae_padding"], [1, 2, 0, 1])

    def _recording_detailer(self, **kwargs):
        encoded = []
        image_height, image_width = kwargs.pop("image_hw", (256, 256))

        def fake_encode(vae, pixels, mask, grow_mask_by, **encode_kwargs):
            encoded.append((tuple(pixels.shape), tuple(mask.shape)))
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape)}

        with (
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(
                sam3_nodes, "_vae_decode",
                side_effect=lambda vae, latent: torch.ones(latent["shape"]),
            ),
        ):
            output, _, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                _image(height=image_height, width=image_width), kwargs.pop("mask"), object(), object(),
                object(), object(), object(), **kwargs,
            )
        return output, encoded, json.loads(report_json)

    def test_whole_image_custom_size_samples_at_exactly_that_size_like_forge(self):
        # Forge 전체 이미지(inpaint_full_res=False)는 resize_mode 0 "Just Resize" 로
        # 정확히 p.width/p.height 에서 샘플링한다 — only-masked 용 종횡비 맞춤(fit_sample_size)을
        # 쓰면 256x128 이미지 + 128x128 요청이 128x64 로 줄어 요청 해상도보다 낮게 돌았다.
        mask = torch.ones((1, 128, 256))
        output, encoded, report = self._recording_detailer(
            mask=mask, image_hw=(128, 256), only_masked=False, grow_mask_by=0,
            use_custom_size=True, custom_width=128, custom_height=128,
        )
        self.assertEqual(encoded, [((1, 128, 128, 3), (1, 128, 128))])
        pass_report = report["passes"][0]
        self.assertEqual(pass_report["sample_size"], [128, 128])
        self.assertEqual(pass_report["crop"], [0, 0, 256, 128])
        self.assertEqual(report["processing_size"], [128, 128])
        # 결과는 원본 크기로 되돌린다(Forge 와 달리 출력 크기는 바꾸지 않는다).
        self.assertEqual(tuple(output.shape), (1, 128, 256, 3))
        self.assertTrue(torch.allclose(output, torch.ones_like(output), atol=1e-5))

    def test_whole_image_target_size_with_other_aspect_is_not_shrunk(self):
        mask = torch.ones((1, 256, 256))
        output, encoded, report = self._recording_detailer(
            mask=mask, only_masked=False, grow_mask_by=0,
            target_width=64, target_height=96,
        )
        self.assertEqual(encoded, [((1, 96, 64, 3), (1, 96, 64))])
        self.assertEqual(report["passes"][0]["sample_size"], [64, 96])
        self.assertEqual(report["passes"][0]["crop"], [0, 0, 256, 256])
        self.assertEqual(tuple(output.shape), (1, 256, 256, 3))

    def _controlnet_geometry_detail(self, source, mask, *, module="inpaint_only",
                                    settings=None, **kwargs):
        """ControlNet 을 켠 detail() 의 (VAE 입력 픽셀, VAE 입력 마스크, 힌트, 리포트).

        인코드·샘플·디코드와 ControlNet 로드·적용만 가짜로 두고, 힌트 맞춤(_fit_control_image)과
        inpaint_only 힌트(마스크 칸 → -1)는 실제 코드가 만든다. ``settings`` 는
        _CONTROLNET_EXTRA_DEFAULTS 위에 덮어쓸 값, ``module="none"`` 이면 힌트가 맞춘 이미지 그대로다.
        """
        encoded = []
        encoded_masks = []
        hints = []

        def fake_encode(vae, pixels, mask, grow_mask_by, **encode_kwargs):
            encoded.append(pixels.clone())
            encoded_masks.append(mask.clone())
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape)}

        def fake_apply(positive, negative, control_net, hint, *args):
            hints.append(hint.clone())
            return positive, negative, {"mode": "Balanced"}

        with (
            mock.patch.object(sam3_nodes, "_load_controlnet", return_value=(object(), "test-cn")),
            mock.patch.object(sam3_nodes, "_apply_controlnet", side_effect=fake_apply),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(
                sam3_nodes, "_vae_decode",
                side_effect=lambda vae, latent: torch.ones(latent["shape"]),
            ),
        ):
            _output, _applied, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                source, mask, object(), object(), object(), object(), object(),
                controlnet_enable=True, controlnet_module=module,
                controlnet_settings_json=json.dumps(
                    {**sam3_nodes._CONTROLNET_EXTRA_DEFAULTS, **(settings or {})}
                ),
                **kwargs,
            )
        return encoded, encoded_masks, hints, json.loads(report_json)

    def test_whole_image_controlnet_hint_is_stretched_like_the_latent_input(self):
        # 전체 이미지 + 다른 종횡비 요청: 잠재 입력은 요청 크기로 '늘려'(Just Resize) 샘플링되므로
        # 같은 이미지에서 뽑는 힌트도 똑같이 늘려야 한다. 설정값(기본 Crop and Resize)대로 가운데를
        # 자르면 256x128 의 왼쪽 1/4 표식이 잠재 입력엔 32열, 힌트엔 0열로 어긋났다.
        # Forge 도 유닛 이미지가 없으면 p.resize_mode(ADetailer/SAM3 = 0 Just Resize)로 맞춘다.
        source = _image(height=128, width=256)
        source[:, :, :64, :] = 1.0
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 192:] = 1.0
        encoded, encoded_masks, hints, report = self._controlnet_geometry_detail(
            source, mask, only_masked=False, grow_mask_by=0,
            use_custom_size=True, custom_width=128, custom_height=128,
        )
        self.assertEqual(len(encoded), 1)
        self.assertEqual(len(hints), 1)
        latent_input, latent_mask, hint = encoded[0], encoded_masks[0], hints[0]
        self.assertEqual(tuple(latent_input.shape), (1, 128, 128, 3))
        self.assertEqual(tuple(hint.shape), (1, 128, 128, 3))
        latent_marker = latent_input[0, :, :, 0] > 0.5
        self.assertEqual(int(latent_marker[0].sum()), 32)
        self.assertTrue(torch.equal(hint[0, :, :, 0] > 0.5, latent_marker))
        # inpaint_only 가 -1 로 칠한 칸도 잠재 입력의 (늘린) 마스크와 같은 자리다.
        self.assertTrue(torch.equal(hint[0, :, :, 0] < -0.5, latent_mask[0] > 0.5))
        control = report["passes"][0]["controlnet"]
        self.assertEqual(control["resize_mode"], "Just Resize")
        self.assertEqual(control["resize_mode_setting"], "Crop and Resize")
        self.assertEqual(control["hint_source"], "input")
        # pixel perfect 도 실제로 쓴 모드로 잰다: max(128/128, 128/256) * min(128, 256)
        self.assertEqual(control["processor_resolution"], 128)

    def test_explicit_control_image_keeps_its_configured_resize_mode(self):
        # 따로 연결한 control_image 는 Forge 유닛 이미지처럼 설정한 resize_mode 로 맞춘다.
        source = _image(height=128, width=256)
        control_image = _image(height=128, width=256)
        control_image[:, :, :64, :] = 1.0
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 192:] = 1.0
        _encoded, _masks, hints, report = self._controlnet_geometry_detail(
            source, mask, only_masked=False, grow_mask_by=0,
            use_custom_size=True, custom_width=128, custom_height=128,
            control_image=control_image,
        )
        control = report["passes"][0]["controlnet"]
        self.assertEqual(control["resize_mode"], "Crop and Resize")
        self.assertEqual(control["resize_mode_setting"], "Crop and Resize")
        self.assertEqual(control["hint_source"], "control_image")
        # Crop and Resize 는 가운데 128열만 남겨 왼쪽 1/4 표식이 빠진다.
        self.assertEqual(int((hints[0][0, 0, :, 0] > 0.5).sum()), 0)
        self.assertEqual(control["control_image_resize_mode"], "Crop and Resize")

    def test_whole_image_control_image_of_another_aspect_is_not_prestretched(self):
        # Forge 전체 이미지는 유닛 이미지를 그대로 unit.resize_mode 로 p.width/p.height 에 맞춘다.
        # 예전엔 먼저 입력 크기(128x256)로 늘린 뒤 Crop and Resize 해 정사각 control 의 왼쪽
        # 32열 표식이 가운데 크롭에 잘려 0열이 됐다.
        source = _image(height=128, width=256)
        control_image = _image(height=128, width=128)
        control_image[:, :, :32, :] = 1.0
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 192:] = 1.0
        _encoded, _masks, hints, report = self._controlnet_geometry_detail(
            source, mask, only_masked=False, grow_mask_by=0,
            use_custom_size=True, custom_width=128, custom_height=128,
            control_image=control_image,
        )
        hint = hints[0]
        self.assertEqual(tuple(hint.shape), (1, 128, 128, 3))
        self.assertEqual(int((hint[0, 0, :, 0] > 0.5).sum()), 32)
        # inpaint_only 가 -1 로 칠한 마스크 열(96~) 밖은 원본 control 을 직접 맞춘 것과 같다.
        expected = sam3_nodes._fit_control_image(control_image, 128, 128, "Crop and Resize")
        self.assertTrue(torch.allclose(hint[:, :, :96, :], expected[:, :, :96, :]))
        control = report["passes"][0]["controlnet"]
        self.assertEqual(control["resize_mode"], "Crop and Resize")
        self.assertEqual(control["control_image_resize_mode"], "Crop and Resize")
        # pixel perfect 도 늘리지 않은 control 로 잰다: max(128/128, 128/128) * 128
        self.assertEqual(control["processor_resolution"], 128)

    def test_whole_image_control_image_honours_the_configured_mode_at_image_size(self):
        # sample 크기 == 입력 크기여도 설정한 모드가 살아 있어야 한다 — 예전엔 입력 크기로 먼저
        # 늘려 둔 탓에 Crop and Resize 가 조용히 Just Resize 처럼 됐다.
        source = _image(height=128, width=256)
        control_image = _image(height=128, width=128)
        control_image[:, :16, :, :] = 1.0
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 192:] = 1.0
        for mode, marked_rows in (("Crop and Resize", 0), ("Just Resize", 16)):
            with self.subTest(mode=mode):
                _encoded, _masks, hints, report = self._controlnet_geometry_detail(
                    source, mask, only_masked=False, grow_mask_by=0,
                    control_image=control_image, module="none", settings={"resize_mode": mode},
                )
                hint = hints[0]
                self.assertEqual(tuple(hint.shape), (1, 128, 256, 3))
                self.assertEqual(int((hint[0, :, 0, 0] > 0.5).sum()), marked_rows)
                self.assertEqual(report["passes"][0]["controlnet"]["resize_mode"], mode)

    def test_only_masked_control_image_is_fitted_to_the_image_before_cropping(self):
        # Forge only-masked: 유닛 이미지를 unit.resize_mode 로 입력 크기에 맞춘 뒤 크롭한다.
        # 예전엔 입력 크기로 늘려(Just Resize) 정사각 표식이 크롭 힌트에서 32x64 로 찌그러졌다.
        source = _image(height=128, width=256)
        control_image = _image(height=128, width=128)
        control_image[:, 48:80, 48:80, :] = 1.0
        mask = torch.zeros((1, 128, 256))
        mask[:, 56:72, 120:136] = 1.0
        _encoded, _masks, hints, report = self._controlnet_geometry_detail(
            source, mask, only_masked=True, mask_padding=32, grow_mask_by=0,
            control_image=control_image, module="none",
        )
        hint = hints[0]
        pass_report = report["passes"][0]
        self.assertEqual(pass_report["crop"], [88, 24, 168, 104])
        self.assertEqual(tuple(hint.shape), (1, 80, 80, 3))
        marker = hint[0, :, :, 0] > 0.5
        rows = torch.nonzero(marker.any(dim=1)).flatten()
        cols = torch.nonzero(marker.any(dim=0)).flatten()
        height = int(rows.max() - rows.min() + 1)
        width = int(cols.max() - cols.min() + 1)
        self.assertEqual((height, width), (64, 64))
        # 입력 크기에 Crop and Resize 로 맞춘 control 의 같은 크롭 그대로다.
        canvas = sam3_nodes._fit_control_image(control_image, 128, 256, "Crop and Resize")
        self.assertTrue(torch.allclose(hint, canvas[:, 24:104, 88:168, :]))
        control = pass_report["controlnet"]
        self.assertEqual(control["resize_mode"], "Just Resize")
        self.assertEqual(control["control_image_resize_mode"], "Crop and Resize")
        self.assertEqual(control["hint_source"], "control_image")

    @staticmethod
    def _edge_marked_control():
        """128x128 control: 가운데 0.5, 맨 왼쪽 열 0.25, 맨 오른쪽 열 0.75."""
        control_image = _image(height=128, width=128, value=0.5)
        control_image[:, :, 0, :] = 0.25
        control_image[:, :, -1, :] = 0.75
        return control_image

    def test_fit_control_image_fills_resize_and_fill_bands_like_forge(self):
        control_image = self._edge_marked_control()
        # A1111 images.resize_image(2): 가장자리 열(가로 띠면 행)을 띠 끝까지 늘린다.
        edge = sam3_nodes._fit_control_image(
            control_image, 128, 256, "Resize and Fill", fill="edge",
        )
        self.assertEqual(tuple(edge.shape), (1, 128, 256, 3))
        self.assertTrue(torch.equal(edge[:, :, :64, :], torch.full((1, 128, 64, 3), 0.25)))
        self.assertTrue(torch.equal(edge[:, :, 192:, :], torch.full((1, 128, 64, 3), 0.75)))
        self.assertTrue(torch.allclose(edge[:, :, 64:192, :], control_image))
        tall = sam3_nodes._fit_control_image(
            control_image, 256, 128, "Resize and Fill", fill="edge",
        )
        self.assertTrue(torch.allclose(tall[:, :64, :, :], control_image[:, :1].expand(-1, 64, -1, -1)))
        self.assertTrue(torch.allclose(tall[:, 192:, :, :], control_image[:, -1:].expand(-1, 64, -1, -1)))

        # ControlNet crop_and_resize_image(OUTER_FIT): 원본 테두리의 채널별 np.median —
        # 짝수 개면 가운데 두 값의 평균이다(torch.median 이면 0.2 가 됐다).
        import numpy as np

        two = torch.tensor([[[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                             [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]]], dtype=torch.float32)
        detected_map = two[0].numpy()
        borders = np.concatenate(
            [detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]],
            axis=0,
        )
        forge_colour = torch.from_numpy(np.median(borders, axis=0))
        median = sam3_nodes._fit_control_image(two, 2, 4, "Resize and Fill")
        self.assertTrue(torch.allclose(forge_colour, torch.full((3,), 0.4)))
        for column in (0, 3):
            self.assertTrue(torch.allclose(median[0, :, column, :], forge_colour.expand(2, 3)))
        self.assertTrue(torch.allclose(median[:, :, 1:3, :], two))
        # 검정 띠는 어느 Forge 경로에도 없다 — 모르는 fill 은 조용히 넘기지 않는다.
        with self.assertRaisesRegex(ValueError, "fill"):
            sam3_nodes._fit_control_image(two, 2, 4, "Resize and Fill", fill="black")

    def test_resize_and_fill_band_mask_marks_only_the_bands(self):
        band = sam3_nodes._resize_and_fill_band_mask(_image(height=128, width=128), 128, 256)
        self.assertEqual(tuple(band.shape), (1, 128, 256))
        self.assertEqual(float(band[:, :, :64].min()), 1.0)
        self.assertEqual(float(band[:, :, 192:].min()), 1.0)
        self.assertEqual(float(band[:, :, 64:192].max()), 0.0)
        # VAE 패딩(left, right, top, bottom)은 힌트처럼 가장자리를 복제한다.
        padded = sam3_nodes._resize_and_fill_band_mask(
            _image(height=128, width=128), 128, 256, (1, 2, 3, 0),
        )
        self.assertEqual(tuple(padded.shape), (1, 131, 259))
        self.assertEqual(float(padded[:, :, :65].min()), 1.0)
        self.assertEqual(float(padded[:, :, 193:].min()), 1.0)
        self.assertEqual(float(padded[:, :, 65:193].max()), 0.0)
        # 종횡비가 같으면 띠가 없다.
        self.assertEqual(
            float(sam3_nodes._resize_and_fill_band_mask(_image(height=64, width=64), 128, 128).sum()),
            0.0,
        )

    def test_only_masked_resize_and_fill_crop_in_a_band_gets_edge_content(self):
        # 감사 S1-comfy#1: only-masked 는 유닛 이미지를 A1111 resize_image(2)로 입력 크기에 맞춘다 —
        # 띠는 가장자리 열을 늘린 것이다. 예전엔 검정 캔버스라 띠에 걸린 크롭 힌트가 0 이었다.
        source = _image(height=128, width=256)
        control_image = self._edge_marked_control()
        mask = torch.zeros((1, 128, 256))
        mask[:, 56:72, 8:24] = 1.0
        _encoded, _masks, hints, report = self._controlnet_geometry_detail(
            source, mask, only_masked=True, mask_padding=4, grow_mask_by=0,
            control_image=control_image, module="none",
            settings={"resize_mode": "Resize and Fill"},
        )
        pass_report = report["passes"][0]
        self.assertEqual(pass_report["crop"], [4, 52, 28, 76])
        hint = hints[0]
        self.assertEqual(tuple(hint.shape), (1, 24, 24, 3))
        self.assertTrue(torch.allclose(hint, torch.full_like(hint, 0.25)))
        canvas = sam3_nodes._fit_control_image(
            control_image, 128, 256, "Resize and Fill", fill="edge",
        )
        self.assertTrue(torch.allclose(hint, canvas[:, 52:76, 4:28, :]))
        self.assertEqual(
            pass_report["controlnet"]["control_image_resize_mode"], "Resize and Fill",
        )

    @staticmethod
    def _median_marked_control():
        """128x128 control 0.3, 윗줄 10칸만 0.9 — 테두리 중앙값은 0.3."""
        control_image = _image(height=128, width=128, value=0.3)
        control_image[:, 0, 10:20, :] = 0.9
        return control_image

    def test_whole_image_resize_and_fill_bands_use_the_border_median(self):
        # Forge 전체 이미지(crop_and_resize_image OUTER_FIT)는 띠를 테두리 중앙값으로 채운다 — 검정이 아니다.
        source = _image(height=128, width=256)
        control_image = self._median_marked_control()
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 120:136] = 1.0
        _encoded, _masks, hints, report = self._controlnet_geometry_detail(
            source, mask, only_masked=False, grow_mask_by=0,
            control_image=control_image, module="none",
            settings={"resize_mode": "Resize and Fill"},
        )
        hint = hints[0]
        self.assertEqual(tuple(hint.shape), (1, 128, 256, 3))
        self.assertTrue(torch.allclose(hint[:, :, :64, :], torch.full((1, 128, 64, 3), 0.3)))
        self.assertTrue(torch.allclose(hint[:, :, 192:, :], torch.full((1, 128, 64, 3), 0.3)))
        self.assertTrue(torch.allclose(
            hint, sam3_nodes._fit_control_image(control_image, 128, 256, "Resize and Fill"),
        ))
        self.assertEqual(report["passes"][0]["controlnet"]["resize_mode"], "Resize and Fill")

    def test_whole_image_inpaint_hint_marks_resize_and_fill_bands_for_generation(self):
        # Forge inpaint 전처리기: Resize and Fill 에선 유닛 마스크의 띠를 255 로 채운다 → 힌트 -1.
        # 잠재 입력의 마스크(무엇을 다시 그릴지)는 그대로다 — 띠는 힌트에만 들어간다.
        source = _image(height=128, width=256)
        control_image = self._median_marked_control()
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 120:136] = 1.0
        _encoded, encoded_masks, hints, _report = self._controlnet_geometry_detail(
            source, mask, only_masked=False, grow_mask_by=0,
            control_image=control_image, settings={"resize_mode": "Resize and Fill"},
        )
        hint = hints[0]
        self.assertTrue(torch.equal(hint[:, :, :64, :], torch.full((1, 128, 64, 3), -1.0)))
        self.assertTrue(torch.equal(hint[:, :, 192:, :], torch.full((1, 128, 64, 3), -1.0)))
        self.assertTrue(torch.equal(hint[:, :, 120:136, :], torch.full((1, 128, 16, 3), -1.0)))
        self.assertTrue(torch.allclose(hint[:, 1:, 64:120, :], torch.full((1, 127, 56, 3), 0.3)))
        latent_mask = encoded_masks[0]
        self.assertEqual(float(latent_mask[:, :, :120].sum()), 0.0)
        self.assertEqual(float(latent_mask[:, :, 136:].sum()), 0.0)
        # Crop and Resize 는 띠가 없으니 마스크 칸만 -1 이다.
        _encoded, _masks, hints, _report = self._controlnet_geometry_detail(
            source, mask, only_masked=False, grow_mask_by=0,
            control_image=control_image, settings={"resize_mode": "Crop and Resize"},
        )
        self.assertEqual(int((hints[0][0, :, :, 0] < -0.5).sum()), 128 * 16)

    def test_whole_image_control_image_is_preprocessed_before_it_is_fitted(self):
        # Forge 전체 이미지: preprocessor(유닛 이미지) → crop_and_resize_image(출력, 모드, h, w).
        # 전처리기는 맞추기 전 원본 control 을 받고, Resize and Fill 띠는 '출력' 테두리의 중앙값이다.
        seen = []

        class FakeCanny:
            FUNCTION = "execute"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)},
                        "optional": {"resolution": ("INT", {"default": 512})}}

            @classmethod
            def execute(cls, image, resolution):
                seen.append((tuple(image.shape), resolution))
                return (1.0 - image,)

        source = _image(height=128, width=256)
        control_image = self._median_marked_control()
        mask = torch.zeros((1, 128, 256))
        mask[:, :, 120:136] = 1.0
        for mode in ("Resize and Fill", "Crop and Resize", "Just Resize"):
            with self.subTest(mode=mode):
                seen.clear()
                with mock.patch.object(
                    sam3_nodes, "_node_mappings", return_value={"CannyEdgePreprocessor": FakeCanny},
                ):
                    _encoded, _masks, hints, report = self._controlnet_geometry_detail(
                        source, mask, only_masked=False, grow_mask_by=0,
                        control_image=control_image, module="canny",
                        settings={"resize_mode": mode},
                    )
                control = report["passes"][0]["controlnet"]
                self.assertEqual(seen, [((1, 128, 128, 3), control["processor_resolution"])])
                expected = sam3_nodes._fit_control_image(1.0 - control_image, 128, 256, mode)
                self.assertTrue(torch.allclose(hints[0], expected))
                self.assertEqual(control["preprocessor"], "CannyEdgePreprocessor")
                if mode == "Resize and Fill":
                    # 띠 = 출력 테두리 중앙값 1 - 0.3.
                    self.assertTrue(torch.allclose(
                        hints[0][:, :, :64, :], torch.full((1, 128, 64, 3), 0.7),
                    ))

    def test_only_masked_implicit_hint_also_uses_just_resize(self):
        mask = torch.zeros((1, 256, 256))
        mask[:, 100:110, 100:110] = 1
        _encoded, _masks, hints, report = self._controlnet_geometry_detail(
            _image(height=256, width=256), mask, mask_padding=2, only_masked=True,
            target_width=64, target_height=96,
        )
        self.assertEqual(tuple(hints[0].shape), (1, 96, 64, 3))
        control = report["passes"][0]["controlnet"]
        self.assertEqual(control["resize_mode"], "Just Resize")
        self.assertEqual(control["hint_source"], "input")

    def test_effective_control_resize_mode_rule(self):
        effective = sam3_nodes._effective_control_resize_mode
        for mode in ("Just Resize", "Crop and Resize", "Resize and Fill"):
            with self.subTest(mode=mode):
                self.assertEqual(effective(mode, explicit_control_image=True), mode)
                self.assertEqual(effective(mode, explicit_control_image=False), "Just Resize")
        # 입력 힌트에선 설정값을 안 쓰더라도 잘못된 값은 조용히 넘기지 않는다.
        for explicit in (True, False):
            with self.subTest(explicit=explicit), self.assertRaisesRegex(ValueError, "resize_mode"):
                effective("Stretch", explicit_control_image=explicit)

    def test_only_masked_samples_small_crop_at_target_size_like_forge(self):
        mask = torch.zeros((1, 256, 256))
        mask[:, 100:110, 100:110] = 1
        output, encoded, report = self._recording_detailer(
            mask=mask, mask_padding=2, only_masked=True,
            target_width=64, target_height=96,
        )
        # 14x14 패딩 크롭 → 2:3 로 넓혀 14x21 → 64x96 로 인코딩(예전: 16x16)
        self.assertEqual(encoded, [((1, 96, 64, 3), (1, 96, 64))])
        pass_report = report["passes"][0]
        self.assertEqual(pass_report["mask_crop"], [98, 98, 112, 112])
        self.assertEqual(pass_report["crop"], [98, 95, 112, 116])
        self.assertEqual(pass_report["sample_size"], [64, 96])
        self.assertEqual(report["processing_size"], [64, 96])
        # 합성은 마스크 안쪽만 바꾼다 (크롭 크기로 되돌리는 bicubic 은 float 오차가 있다)
        self.assertTrue(torch.allclose(output[0, 105, 105], torch.ones(3), atol=1e-5))
        self.assertTrue(output[0, 96, 105].eq(0).all())
        self.assertEqual(float(output[0, :90].sum()), 0.0)

    def test_custom_size_expands_crop_to_its_aspect_instead_of_stretching(self):
        mask = torch.zeros((1, 256, 256))
        mask[:, 100:110, 100:110] = 1
        _output, encoded, report = self._recording_detailer(
            mask=mask, mask_padding=0, only_masked=True,
            use_custom_size=True, custom_width=128, custom_height=64,
            target_width=512, target_height=512,
        )
        # 커스텀 크기가 target 보다 우선하고, 10x10 크롭은 2:1(20x10) 로 넓혀진다.
        self.assertEqual(report["passes"][0]["crop"], [95, 100, 115, 110])
        self.assertEqual(encoded, [((1, 64, 128, 3), (1, 64, 128))])

    def test_zero_target_keeps_crop_size_sampling_for_saved_workflows(self):
        mask = torch.zeros((1, 256, 256))
        mask[:, 100:110, 100:110] = 1
        _output, encoded, report = self._recording_detailer(
            mask=mask, mask_padding=3, only_masked=True,
        )
        self.assertEqual(report["passes"][0]["crop"], [97, 97, 113, 113])
        self.assertEqual(encoded, [((1, 16, 16, 3), (1, 16, 16))])
        self.assertIsNone(report["processing_size"])

    def test_invalid_target_size_is_rejected(self):
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:8, 4:8] = 1
        for width, height in ((32, 32), (512, 0), (0, 512), (9000, 512)):
            with self.subTest(size=(width, height)), self.assertRaisesRegex(ValueError, "target size"):
                sam3_nodes.ForgeNeoSAM3Detailer().detail(
                    _image(height=16, width=16), mask, object(), object(), object(),
                    object(), object(), target_width=width, target_height=height,
                )

    def test_detailer_executes_controlnet_and_restore_adapters(self):
        sampled_conditioning = []

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape)}

        def fake_sample(model, seed, steps, cfg, sampler_name, scheduler,
                        positive, negative, latent, denoise, noise_multiplier):
            sampled_conditioning.append((positive, negative))
            return latent

        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1
        restore_result = (_image(height=16, width=16, value=0.25), mask, {"provider": "fake-face"})
        with (
            mock.patch.object(sam3_nodes, "_load_controlnet", return_value=(object(), "cn.safetensors")),
            mock.patch.object(sam3_nodes, "_prepare_control_hint", return_value=(_image(height=8, width=8), "inpaint_hint")),
            mock.patch.object(sam3_nodes, "_apply_controlnet", return_value=("cn-positive", "cn-negative", {"mode": "Balanced"})),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=fake_sample),
            mock.patch.object(sam3_nodes, "_vae_decode", side_effect=lambda vae, latent: torch.ones(latent["shape"])),
            mock.patch.object(sam3_nodes, "_restore_faces", return_value=restore_result) as restore_mock,
        ):
            output, _, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                _image(height=16, width=16), mask, object(), object(), object(),
                "positive", "negative", mask_padding=0,
                controlnet_enable=True, controlnet_model_name="cn.safetensors",
                controlnet_settings_json=json.dumps(sam3_nodes._CONTROLNET_EXTRA_DEFAULTS),
                restore_face=True,
                restore_face_settings_json=json.dumps(sam3_nodes._RESTORE_FACE_DEFAULTS),
            )
        self.assertEqual(sampled_conditioning, [("cn-positive", "cn-negative")])
        restore_mock.assert_called_once()
        self.assertTrue(torch.allclose(output, torch.full_like(output, 0.25)))
        report = json.loads(report_json)
        self.assertTrue(report["controlnet_enabled"])
        self.assertEqual(report["restore_face"]["provider"], "fake-face")

    def test_zero_denoise_skips_prompt_vae_sampler_and_controlnet(self):
        source = _image(height=9, width=11, value=0.2)
        mask = torch.zeros((1, 9, 11))
        mask[:, 2:7, 3:8] = 1
        with (
            mock.patch.object(
                sam3_nodes, "_encode_prompt", side_effect=AssertionError("prompt must not encode")
            ),
            mock.patch.object(
                sam3_nodes, "_load_controlnet", side_effect=AssertionError("ControlNet must not load")
            ),
            mock.patch.object(
                sam3_nodes, "_vae_encode_for_inpaint", side_effect=AssertionError("VAE must not encode")
            ),
            mock.patch.object(
                sam3_nodes, "_sample_latent", side_effect=AssertionError("sampler must not run")
            ),
        ):
            output, applied, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                source, mask, object(), object(), object(), object(), object(),
                inpaint_prompt="unused", negative_prompt="unused", denoise=0.0,
                controlnet_enable=True, controlnet_strength=1.0,
            )
        self.assertTrue(torch.equal(output, source))
        self.assertTrue(torch.equal(applied, mask))
        report = json.loads(report_json)
        self.assertEqual(report["status"], "no_op_zero_denoise")
        self.assertFalse(report["controlnet_enabled"])
        self.assertEqual(report["controlnet_disabled_reason"], "zero_denoise")
        self.assertEqual(report["passes"][0]["status"], "no_op_zero_denoise")

    def test_zero_control_strength_skips_model_loading_but_still_inpaints(self):
        mask = torch.zeros((1, 8, 8))
        mask[:, 2:6, 2:6] = 1

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape)}

        with (
            mock.patch.object(
                sam3_nodes, "_load_controlnet", side_effect=AssertionError("ControlNet must not load")
            ),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(
                sam3_nodes, "_vae_decode",
                side_effect=lambda vae, latent: torch.ones(latent["shape"]),
            ),
        ):
            output, _, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                _image(height=8, width=8), mask, object(), object(), object(),
                object(), object(), mask_padding=0, controlnet_enable=True,
                controlnet_strength=0.0,
            )
        self.assertTrue(output[0, 3, 3].eq(1).all())
        report = json.loads(report_json)
        self.assertFalse(report["controlnet_enabled"])
        self.assertEqual(report["controlnet_disabled_reason"], "zero_strength")

    def test_implicit_control_source_tracks_current_sequential_result(self):
        control_means = []
        decode_values = iter((0.25, 0.75))

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            return {"samples": torch.zeros((1, 4, 2, 2)), "shape": tuple(pixels.shape)}

        def fake_control_hint(image, mask, *args):
            control_means.append(float(image.mean()))
            return image, "raw-test"

        def fake_decode(vae, latent):
            return torch.full(latent["shape"], next(decode_values))

        masks = torch.zeros((2, 16, 16))
        masks[0, 1:5, 1:5] = 1
        masks[1, 10:14, 10:14] = 1
        with (
            mock.patch.object(sam3_nodes, "_load_controlnet", return_value=(object(), "test-cn")),
            mock.patch.object(sam3_nodes, "_prepare_control_hint", side_effect=fake_control_hint),
            mock.patch.object(
                sam3_nodes, "_apply_controlnet",
                side_effect=lambda positive, negative, *args: (positive, negative, {"mode": "Balanced"}),
            ),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(sam3_nodes, "_vae_decode", side_effect=fake_decode),
        ):
            output, _, _ = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                _image(height=16, width=16), masks, object(), object(), object(),
                object(), object(), mask_mode="Individual", only_masked=False,
                controlnet_enable=True,
                controlnet_settings_json=json.dumps(sam3_nodes._CONTROLNET_EXTRA_DEFAULTS),
            )
        self.assertEqual(len(control_means), 2)
        self.assertEqual(control_means[0], 0.0)
        self.assertGreater(control_means[1], 0.0)
        self.assertTrue(output[0, 2, 2].eq(0.25).all())
        self.assertTrue(output[0, 11, 11].eq(0.75).all())

    def test_restore_face_uses_impact_provider_contract(self):
        detector = object()
        calls = []

        class FakeDetectorProvider:
            FUNCTION = "doit"

            @classmethod
            def doit(cls, model_name):
                return detector, object()

        class FakeFaceDetailer:
            FUNCTION = "doit"

            @classmethod
            def doit(cls, **kwargs):
                calls.append(kwargs)
                return kwargs["image"] + 0.1, [], [], torch.ones((1, 8, 8)), object(), []

        settings = dict(sam3_nodes._RESTORE_FACE_DEFAULTS)
        with mock.patch.object(
            sam3_nodes,
            "_node_mappings",
            return_value={
                "UltralyticsDetectorProvider": FakeDetectorProvider,
                "FaceDetailer": FakeFaceDetailer,
            },
        ):
            output, face_mask, report = sam3_nodes._restore_faces(
                _image(height=8, width=8), model=object(), clip=object(), vae=object(),
                positive=object(), negative=object(), sampler_name="euler",
                scheduler="normal", seed=4, steps=8, cfg=5.0,
                settings_json=json.dumps(settings),
            )
        self.assertAlmostEqual(output.mean().item(), 0.1, places=5)
        self.assertEqual(face_mask.sum().item(), 64)
        self.assertIs(calls[0]["bbox_detector"], detector)
        self.assertEqual(report["provider"], "Impact FaceDetailer")

    def test_detailer_reports_empty_detection_without_sampling(self):
        source = _image(height=8, width=8)
        output, applied, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
            source, torch.zeros((1, 8, 8)),
            object(), object(), object(), object(), object(),
        )
        self.assertTrue(torch.equal(output, source))
        self.assertEqual(applied.sum().item(), 0)
        self.assertEqual(json.loads(report_json)["status"], "no_detection")

    def test_refine_and_tile_repair_are_real_paths(self):
        self.assertTrue(issubclass(sam3_nodes.ForgeNeoSAM3Refine, sam3_nodes.ForgeNeoSAM3Detailer))
        self.assertEqual(sam3_nodes.ForgeNeoSAM3Refine.FUNCTION, "detail")
        self.assertEqual(sam3_nodes.ForgeNeoSAM3TileRepair.FUNCTION, "tile_repair")
        with self.assertRaisesRegex(ValueError, "Invalid SAM3 tile settings_json"):
            sam3_nodes.ForgeNeoSAM3TileRepair().tile_repair(
                _image(), torch.ones((1, 12, 14)), object(), object(), object(),
                object(), object(), settings_json="{bad json",
            )

    def test_refine_outputs_each_mask_from_the_unmodified_source(self):
        decode_values = iter((0.25, 0.75))

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape)}

        masks = torch.zeros((2, 8, 8))
        masks[0, 1:4, 1:4] = 1
        masks[1, 5:7, 5:7] = 1
        with (
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(
                sam3_nodes, "_vae_decode",
                side_effect=lambda vae, latent: torch.full(
                    latent["shape"], next(decode_values)
                ),
            ),
        ):
            output, applied, report_json = sam3_nodes.ForgeNeoSAM3Refine().detail(
                _image(height=8, width=8), masks, object(), object(), object(),
                object(), object(), mask_mode="Individual", mask_padding=0,
            )
        self.assertEqual(tuple(output.shape), (2, 8, 8, 3))
        self.assertTrue(output[0, 2, 2].eq(0.25).all())
        self.assertTrue(output[0, 5, 5].eq(0.0).all())
        self.assertTrue(output[1, 2, 2].eq(0.0).all())
        self.assertTrue(output[1, 5, 5].eq(0.75).all())
        self.assertEqual(applied.sum().item(), 13)
        report = json.loads(report_json)
        self.assertEqual(report["processing"], "independent_from_original")
        self.assertEqual(report["result_count"], 2)

    def test_refine_loads_a_named_controlnet_once_for_every_mask(self):
        loads = []
        control_models = []

        class FakeControlNetLoader:
            @classmethod
            def load_controlnet(cls, control_net_name):
                loads.append(control_net_name)
                return (object(),)

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape)}

        def fake_apply(positive, negative, control_net, *args):
            control_models.append(control_net)
            return positive, negative, {"mode": "Balanced"}

        masks = torch.zeros((3, 16, 16))
        masks[0, 1:4, 1:4] = 1
        masks[1, 6:9, 6:9] = 1
        masks[2, 11:14, 11:14] = 1
        with (
            mock.patch.object(
                sam3_nodes, "_node_mappings",
                return_value={"ControlNetLoader": FakeControlNetLoader},
            ),
            mock.patch.object(
                sam3_nodes, "_prepare_control_hint",
                side_effect=lambda image, *args: (image, "inpaint_hint"),
            ),
            mock.patch.object(sam3_nodes, "_apply_controlnet", side_effect=fake_apply),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda *args: args[8]),
            mock.patch.object(
                sam3_nodes, "_vae_decode",
                side_effect=lambda vae, latent: torch.ones(latent["shape"]),
            ),
        ):
            output, _, report_json = sam3_nodes.ForgeNeoSAM3Refine().detail(
                _image(height=16, width=16), masks, object(), object(), object(),
                object(), object(), mask_mode="Individual", mask_padding=0,
                controlnet_enable=True, controlnet_model_name="cn.safetensors",
                controlnet_settings_json=json.dumps(sam3_nodes._CONTROLNET_EXTRA_DEFAULTS),
            )
        self.assertEqual(loads, ["cn.safetensors"])
        self.assertEqual(len(control_models), 3)
        self.assertEqual(len({id(model) for model in control_models}), 1)
        self.assertEqual(tuple(output.shape), (3, 16, 16, 3))
        report = json.loads(report_json)
        self.assertEqual(
            [result["passes"][0]["controlnet"]["model"] for result in report["results"]],
            ["cn.safetensors"] * 3,
        )

    def test_tile_repair_rejects_forge_anima_options_instead_of_masquerading(self):
        with self.assertRaisesRegex(RuntimeError, "Anima Tile-Repair/PiD.*lllite_model"):
            sam3_nodes.ForgeNeoSAM3TileRepair().tile_repair(
                _image(), torch.ones((1, 12, 14)), object(), object(), object(),
                object(), object(), settings_json='{"lllite_model": "anima.safetensors"}',
            )


if __name__ == "__main__":
    unittest.main()
