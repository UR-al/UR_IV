"""Anima Tile & Repair 요청 만들기(core/tile_repair_request.py) — 원본 기본값·확장 라우트 계약·원본 이미지 읽기.

원본(원본 동등성 계획 §7, 패키지 TR-API):
  - kohya-ss/sd-scripts@690ea7f9 anima_minimal_inference_control_net_lllite.py argparse 기본값
  - kohya-ss/ComfyUI-Anima-LLLite@b7495bd8 nodes.py strength 범위
확장 라우트(sam3ext/tile_repair_api.py)는 모르는 키를 400 으로 거절하므로, 설치된 확장 소스(AST)와 키·기본값·범위를
대조한다. 프론트 거울(frontend/src/utils/tileRepair.ts)도 같은 값인지 본다. 네트워크·Forge·torch 없음.
"""
from __future__ import annotations

import ast
import base64
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from core import tile_repair_request as trr
from core.tile_repair_request import (
    DEFAULT_PROMPT, DEFAULTS, RANGES, ROUTE_KEYS, build_route_body, check_output_size, load_source_image,
    normalize_settings, output_size,
)
from tests._sam_extra_ext import EXT_ROOT, requires_extension

ROOT = Path(__file__).resolve().parents[1]


def image_bytes(fmt="PNG", size=(20, 12)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buffer, format=fmt)
    return buffer.getvalue()


def data_url(fmt="PNG") -> str:
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[fmt]
    return f"data:image/{mime};base64," + base64.b64encode(image_bytes(fmt)).decode()


class OriginDefaultsTests(unittest.TestCase):
    def test_defaults_are_the_originals(self):
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134
        self.assertEqual(DEFAULTS["negative_prompt"], "")      # --negative_prompt default ""
        self.assertEqual(DEFAULTS["steps"], 50)                # --infer_steps 50
        self.assertEqual(DEFAULTS["cfg_scale"], 3.5)           # --guidance_scale 3.5
        self.assertEqual(DEFAULTS["flow_shift"], 5.0)          # --flow_shift 5.0
        # origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:167-170
        self.assertEqual(DEFAULTS["multiplier"], 1.0)          # --lllite_multiplier 1.0
        # origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:130
        #   "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        self.assertEqual(RANGES["multiplier"], (-10.0, 10.0))
        # 확장 패널: 짧은 변 1024(원본 비율 유지), 시드 -1, 실행 전 Forge 내리기
        self.assertEqual((DEFAULTS["short_side"], DEFAULTS["seed"], DEFAULTS["unload_forge_before"]),
                         (1024, -1, True))

    def test_frontend_mirror_has_the_same_values(self):
        text = (ROOT / "frontend" / "src" / "utils" / "tileRepair.ts").read_text(encoding="utf-8")
        prompt = re.search(r"TILE_REPAIR_DEFAULT_PROMPT =\s*'([^']*)'", text).group(1)
        self.assertEqual(prompt, DEFAULT_PROMPT)
        ranges = dict(re.findall(r"^\s+(\w+): \[(-?[\d.]+), (-?[\d.]+)\],$", text, re.M) and
                      [(k, (float(a), float(b))) for k, a, b in
                       re.findall(r"^\s+(\w+): \[(-?[\d.]+), (-?[\d.]+)\],$", text, re.M)])
        self.assertEqual(ranges, {k: (float(a), float(b)) for k, (a, b) in RANGES.items()})
        body = re.search(r"export function defaultTileRepairSettings\(\)[^{]*\{\s*return \{(.*?)\n  \}", text, re.S)
        defaults = dict(re.findall(r"(\w+): (-?[\d.]+|true|false),", body.group(1)))
        for key in ("steps", "cfg_scale", "flow_shift", "multiplier", "short_side", "seed"):
            with self.subTest(key=key):
                self.assertEqual(float(defaults[key]), float(DEFAULTS[key]))
        self.assertEqual(defaults["unload_forge_before"], "true")
        self.assertIn("negative_prompt: ''", body.group(1))


def _extension_constants() -> dict:
    """설치된 확장 sam3ext/tile_repair_api.py 의 모듈 상수(DEFAULT_PROMPT·DEFAULTS·RANGES·REQUEST_KEYS)."""
    tree = ast.parse((EXT_ROOT / "sam3ext" / "tile_repair_api.py").read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        value = getattr(node, "value", None)
        if not isinstance(target, ast.Name) or value is None:
            continue
        if isinstance(value, ast.Call) and value.args:   # MappingProxyType({...}) / frozenset({...})
            value = value.args[0]
        if target.id == "DEFAULTS":
            found[target.id] = {ast.literal_eval(k): (DEFAULT_PROMPT if isinstance(v, ast.Name) else ast.literal_eval(v))
                                for k, v in zip(value.keys, value.values)}
            continue
        if target.id in ("DEFAULT_PROMPT", "RANGES", "REQUEST_KEYS", "TILE_REPAIR_API_PATH",
                         "TILE_REPAIR_OPTIONS_PATH", "TILE_REPAIR_STOP_PATH", "MAX_IMAGE_BYTES", "MAX_SOURCE_PIXELS",
                         "MAX_OUTPUT_PIXELS"):
            try:
                found[target.id] = ast.literal_eval(value)
            except ValueError:
                found[target.id] = ast.unparse(value)
    return found


def _extension_tile_repair_size():
    """설치된 확장 sam3ext/anima_core.py 의 ``tile_repair_size`` — 함수와 두 상수만 떼어 실행한다(torch 없이)."""
    tree = ast.parse((EXT_ROOT / "sam3ext" / "anima_core.py").read_text(encoding="utf-8"))
    wanted = {"TILE_REPAIR_SIZE_MULTIPLE", "TILE_REPAIR_MIN_SIDE", "tile_repair_size"}
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets))
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "anima_core.py", "exec"), namespace)  # noqa: S102
    return namespace


@requires_extension
class ExtensionRouteContractTests(unittest.TestCase):
    """앱이 보내는 본문 = 확장 라우트가 받는 본문(키·기본값·범위)."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _extension_constants()

    def test_route_keys_match(self):
        self.assertEqual(ROUTE_KEYS, set(self.ext["REQUEST_KEYS"]))

    def test_defaults_ranges_and_prompt_match(self):
        self.assertEqual(self.ext["DEFAULT_PROMPT"], DEFAULT_PROMPT)
        self.assertEqual(self.ext["DEFAULTS"], dict(DEFAULTS))
        self.assertEqual({k: tuple(v) for k, v in self.ext["RANGES"].items()}, dict(RANGES))

    def test_paths_and_limits_match_the_client(self):
        from core import forge_tile_repair_client as client
        self.assertEqual(self.ext["TILE_REPAIR_API_PATH"], client.TILE_REPAIR_API_PATH)
        self.assertEqual(self.ext["TILE_REPAIR_OPTIONS_PATH"], client.TILE_REPAIR_OPTIONS_PATH)
        self.assertEqual(self.ext["TILE_REPAIR_STOP_PATH"], client.TILE_REPAIR_STOP_PATH)
        self.assertEqual(eval(self.ext["MAX_IMAGE_BYTES"]), trr.MAX_SOURCE_BYTES)      # noqa: S307 - 상수식
        self.assertEqual(eval(self.ext["MAX_SOURCE_PIXELS"]), trr.MAX_SOURCE_PIXELS)  # noqa: S307
        source_pixels = eval(self.ext["MAX_SOURCE_PIXELS"])  # noqa: S307
        self.assertEqual(eval(str(self.ext["MAX_OUTPUT_PIXELS"]), {"MAX_SOURCE_PIXELS": source_pixels}),  # noqa: S307
                         trr.MAX_OUTPUT_PIXELS)

    def test_output_size_is_the_extension_formula(self):
        ext = _extension_tile_repair_size()
        self.assertEqual((ext["TILE_REPAIR_SIZE_MULTIPLE"], ext["TILE_REPAIR_MIN_SIDE"]),
                         (trr.SIZE_MULTIPLE, trr.MIN_SIDE))
        sides = (2, 3, 31, 257, 640, 832, 1080, 1215, 1920, 4097, 8192)
        for width in sides:
            for height in sides:
                for short_side in (256, 1000, 1024, 4096):
                    with self.subTest(size=(width, height), short_side=short_side):
                        self.assertEqual(output_size(width, height, short_side),
                                         ext["tile_repair_size"](width, height, short_side))


class NormalizeSettingsTests(unittest.TestCase):
    def test_empty_settings_give_the_defaults_and_no_model_fields(self):
        body = normalize_settings({})
        self.assertEqual(body, {key: DEFAULTS[key] for key in DEFAULTS})
        self.assertTrue(set(body) <= ROUTE_KEYS)

    def test_values_pass_through_and_empty_models_are_left_to_the_extension(self):
        body = normalize_settings({
            "model": "animaTileRepair_v10.safetensors", "dit": "", "text_encoder": None,
            "vae": "qwen_image_vae.safetensors", "prompt": "fix", "negative_prompt": "blur",
            "steps": 30.0, "cfg_scale": 5, "flow_shift": 3, "multiplier": -10, "short_side": 768,
            "seed": 42.0, "unload_forge_before": False,
        })
        self.assertEqual(body["model"], "animaTileRepair_v10.safetensors")
        self.assertEqual(body["vae"], "qwen_image_vae.safetensors")
        self.assertNotIn("dit", body)
        self.assertNotIn("text_encoder", body)
        self.assertEqual((body["steps"], body["cfg_scale"], body["multiplier"], body["seed"]), (30, 5.0, -10.0, 42))
        self.assertIsInstance(body["steps"], int)
        self.assertIs(body["unload_forge_before"], False)

    def test_out_of_range_and_wrong_types_are_refused(self):
        cases = [
            {"multiplier": 10.5}, {"multiplier": -10.01}, {"steps": 0}, {"steps": 151}, {"steps": 2.5},
            {"cfg_scale": "3"}, {"cfg_scale": True}, {"short_side": 255}, {"flow_shift": float("nan")},
            {"prompt": "   "}, {"prompt": 3}, {"seed": -3}, {"seed": "1"}, {"unload_forge_before": 1},
            {"model": 5},
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    normalize_settings(settings)


class OutputSizeTests(unittest.TestCase):
    def test_short_side_follows_the_slider_and_long_side_the_aspect(self):
        self.assertEqual(output_size(1000, 1500, 1024), (1024, 1536))
        self.assertEqual(output_size(1920, 1080, 1024), (1792, 1024))   # 1820 → 32 배수
        self.assertEqual(output_size(512, 512, 256), (256, 256))
        self.assertEqual(output_size(100, 20, 256), (1280, 256))

    def test_thin_source_is_refused_before_upload(self):
        # 2x8192 원본(16 KP)도 짧은 변 1024 면 1024x4194304 — 확장이 400 으로 거절하는 크기다.
        data = image_bytes(size=(2, 8192))
        with self.assertRaises(ValueError) as caught:
            check_output_size(data, 1024)
        self.assertIn("1024x4194304", str(caught.exception))
        with self.assertRaises(ValueError):
            build_route_body({"image": "data:image/png;base64," + base64.b64encode(data).decode()})

    def test_a_smaller_short_side_gets_the_same_source_through(self):
        data = image_bytes(size=(2, 8192))
        # 짧은 변 256: 256x1048576 = 256 MP — 여전히 넘는다. 32x8192 원본은 256 이면 256x65536 = 16 MP.
        with self.assertRaises(ValueError):
            check_output_size(data, 256)
        self.assertEqual(check_output_size(image_bytes(size=(32, 8192)), 256), (256, 65536))

    def test_exactly_64_mp_passes_and_one_step_more_does_not(self):
        # 1024x4096 을 짧은 변 4096 으로: 4096x16384 = 64 MP 정확히. 4104 면 16416 이라 넘는다.
        self.assertEqual(check_output_size(image_bytes(size=(1024, 4096)), 4096), (4096, 16384))
        self.assertEqual(4096 * 16384, trr.MAX_OUTPUT_PIXELS)
        with self.assertRaises(ValueError):
            check_output_size(image_bytes(size=(1024, 4104)), 4096)


class LoadSourceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def _write(self, name: str, data: bytes) -> str:
        path = self.root / name
        path.write_bytes(data)
        return str(path)

    def test_png_jpeg_and_webp_files_are_sent_byte_for_byte(self):
        # 라우트가 세 형식을 다 받고 run_tile_repair 가 PIL 로 열어 convert("RGB") 한다 — PNG 로 다시 쓰지 않는다.
        for fmt, name in (("PNG", "원본 a.png"), ("JPEG", "a.jpg"), ("WEBP", "a.webp")):
            with self.subTest(fmt=fmt):
                data = image_bytes(fmt)
                self.assertEqual(load_source_image({"image_path": self._write(name, data)}), data)

    def test_cmyk_jpeg_is_sent_too(self):
        # PNG 로는 저장조차 안 되는 모드 — 확장이 convert("RGB") 한다.
        buffer = io.BytesIO()
        Image.new("CMYK", (20, 12), (10, 20, 30, 40)).save(buffer, format="JPEG")
        data = buffer.getvalue()
        self.assertEqual(load_source_image({"image_path": self._write("cmyk.jpg", data)}), data)

    def test_a_jpeg_under_the_limit_is_never_inflated_past_it(self):
        # 예전: JPEG 를 PNG 로 다시 써서 64 MB 아래 사진이 64 MB 넘는 PNG 가 되고 Forge 가 413 을 냈다.
        buffer = io.BytesIO()
        Image.effect_noise((256, 256), 64).convert("RGB").save(buffer, format="JPEG", quality=30)
        data = buffer.getvalue()
        path = self._write("noise.jpg", data)
        with mock.patch.object(trr, "MAX_SOURCE_BYTES", len(data)):
            body = build_route_body({"image_path": path})
        self.assertEqual(base64.b64decode(body["image"]), data)

    def test_file_url_path_is_accepted(self):
        data = image_bytes()
        path = self._write("b.png", data)
        self.assertEqual(load_source_image({"image_path": "file:///" + path.replace(os.sep, "/")}), data)

    def test_path_wins_over_data_url_and_data_url_works_alone(self):
        path = self._write("c.png", image_bytes(size=(30, 30)))
        with Image.open(io.BytesIO(load_source_image({"image_path": path, "image": data_url()}))) as image:
            self.assertEqual(image.size, (30, 30))
        webp = load_source_image({"image_path": "", "image": data_url("WEBP")})
        self.assertEqual(webp, image_bytes("WEBP"))

    def test_animated_or_too_many_pixels_are_refused(self):
        frames = [Image.new("RGB", (8, 8), color) for color in ((255, 0, 0), (0, 255, 0))]
        buffer = io.BytesIO()
        frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:])
        with self.assertRaises(ValueError):
            load_source_image({"image_path": self._write("anim.webp", buffer.getvalue())})
        with mock.patch.object(trr, "MAX_SOURCE_PIXELS", 100):
            with self.assertRaises(ValueError):
                load_source_image({"image_path": self._write("g.png", image_bytes())})

    def test_refusals(self):
        cases = {
            "nothing": {},
            "missing file": {"image_path": str(self.root / "none.png")},
            "wrong extension": {"image_path": self._write("d.txt", image_bytes())},
            "not an image": {"image_path": self._write("e.png", b"not an image")},
            "external url": {"image": "https://example.com/a.png"},
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    load_source_image(payload)

    def test_too_large_file_is_refused_before_reading(self):
        path = self._write("big.png", image_bytes())
        with mock.patch.object(trr, "MAX_SOURCE_BYTES", 10):
            with self.assertRaises(ValueError):
                load_source_image({"image_path": path})

    def test_route_body_has_only_route_keys(self):
        body = build_route_body({"image_path": self._write("f.png", image_bytes()), "settings": {"seed": 5}})
        self.assertTrue(set(body) <= ROUTE_KEYS)
        self.assertEqual(base64.b64decode(body["image"]), image_bytes())
        self.assertEqual(body["seed"], 5)


if __name__ == "__main__":
    unittest.main()
