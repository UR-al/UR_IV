import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import forge_modules


class TestForgeModelPaths(unittest.TestCase):
    def _make_dirs(self, root: Path) -> dict[str, Path]:
        paths = {
            "checkpoint_dir": root / "checkpoints",
            "lora_dir": root / "loras",
            "vae_dir": root / "vae",
            "text_encoder_dir": root / "text_encoder",
        }
        for path in paths.values():
            path.mkdir(parents=True)
        return paths

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "forge_model_paths.json"
            paths = self._make_dirs(root)

            effective = forge_modules.save_forge_paths(
                {key: str(path) for key, path in paths.items()},
                config_path=config,
                environ={},
            )

            saved = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 1)
            self.assertEqual(saved["paths"], {key: str(path) for key, path in paths.items()})
            self.assertEqual(effective, paths)
            self.assertEqual(
                forge_modules.get_forge_paths(config_path=config, environ={}),
                paths,
            )

    def test_no_detected_forge_creates_app_owned_shared_model_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            project_root = Path(temp) / "project"
            missing = Path(temp) / "missing"

            with (
                patch.object(forge_modules, "PROJECT_ROOT", project_root),
                patch.object(forge_modules, "LEGACY_ROOT_FILE", missing / "legacy.txt"),
                patch.object(
                    forge_modules,
                    "FORGE_ROOT_CANDIDATES",
                    (missing / "forge-neo", missing / "forge-classic"),
                ),
            ):
                root = forge_modules.get_forge_root(environ={})
                paths = forge_modules.get_app_model_paths()

            self.assertEqual(root, (project_root / "user_data" / "models").resolve())
            self.assertEqual(
                {key: path.name for key, path in paths.items()},
                {
                    "checkpoints": "Stable-diffusion",
                    "diffusion_models": "diffusion_models",
                    "loras": "Lora",
                    "vae": "VAE",
                    "text_encoders": "text_encoder",
                    "upscale_models": "upscale_models",
                },
            )
            self.assertTrue(all(path.is_dir() for path in paths.values()))
            upscale = paths["upscale_models"]
            self.assertEqual(upscale, root / "upscale_models")
            self.assertEqual(list(upscale.iterdir()), [])
            # Creating the fallback again must never replace existing weights.
            weight = upscale / "existing-upscaler.pth"
            weight.write_bytes(b"existing model")
            recreated = forge_modules.ensure_app_model_layout(root)
            self.assertEqual(recreated, paths)
            self.assertEqual(weight.read_bytes(), b"existing model")

    def test_detected_forge_keeps_priority_over_app_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            project_root = Path(temp) / "project"
            detected = Path(temp) / "forge" / "models"
            detected.mkdir(parents=True)

            with (
                patch.object(forge_modules, "PROJECT_ROOT", project_root),
                patch.object(
                    forge_modules,
                    "LEGACY_ROOT_FILE",
                    Path(temp) / "missing-legacy.txt",
                ),
                patch.object(forge_modules, "FORGE_ROOT_CANDIDATES", (detected,)),
            ):
                root = forge_modules.get_forge_root(environ={})

            self.assertEqual(root, detected.resolve())
            self.assertFalse((project_root / "user_data" / "models").exists())

    def test_invalid_path_rejects_whole_save_and_preserves_existing_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "forge_model_paths.json"
            paths = self._make_dirs(root)
            payload = {key: str(path) for key, path in paths.items()}
            forge_modules.save_forge_paths(payload, config_path=config, environ={})
            before = config.read_bytes()

            payload["vae_dir"] = str(root / "missing")
            with self.assertRaises(forge_modules.ForgePathError) as caught:
                forge_modules.save_forge_paths(payload, config_path=config, environ={})

            self.assertIn("vae_dir", caught.exception.errors)
            self.assertEqual(config.read_bytes(), before)

    def test_relative_path_and_file_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file_path = root / "model.safetensors"
            file_path.write_bytes(b"x")

            with self.assertRaises(forge_modules.ForgePathError) as caught:
                forge_modules.validate_forge_paths({
                    "checkpoint_dir": "relative/models",
                    "vae_dir": str(file_path),
                })

            self.assertIn("checkpoint_dir", caught.exception.errors)
            self.assertIn("vae_dir", caught.exception.errors)

    def test_environment_override_wins_without_becoming_persistent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "forge_model_paths.json"
            original = self._make_dirs(root / "original")
            replacement = self._make_dirs(root / "replacement")
            env_lora = root / "env-lora"
            env_lora.mkdir()
            forge_modules.save_forge_paths(
                {key: str(path) for key, path in original.items()},
                config_path=config,
                environ={},
            )

            effective = forge_modules.save_forge_paths(
                {key: str(path) for key, path in replacement.items()},
                config_path=config,
                environ={"FORGE_LORA_DIR": str(env_lora)},
            )

            saved = json.loads(config.read_text(encoding="utf-8"))["paths"]
            self.assertEqual(saved["lora_dir"], str(original["lora_dir"]))
            self.assertEqual(effective["lora_dir"], env_lora)
            self.assertEqual(saved["checkpoint_dir"], str(replacement["checkpoint_dir"]))

    def test_scanners_use_category_extensions_and_nested_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self._make_dirs(root)
            nested = paths["checkpoint_dir"] / "nested"
            nested.mkdir()
            (nested / "model.safetensors").write_bytes(b"")
            (paths["checkpoint_dir"] / "quant.gguf").write_bytes(b"")
            (paths["checkpoint_dir"] / "sidecar.vae.safetensors").write_bytes(b"")
            (paths["checkpoint_dir"] / "ignore.pt").write_bytes(b"")

            (paths["lora_dir"] / "style.pt").write_bytes(b"")
            (paths["lora_dir"] / "style.gguf").write_bytes(b"")
            (paths["vae_dir"] / "vae.sft").write_bytes(b"")
            (paths["text_encoder_dir"] / "clip.bin").write_bytes(b"")

            self.assertEqual(
                forge_modules._checkpoint_files(paths["checkpoint_dir"]),
                ["quant.gguf", "nested/model.safetensors"],
            )
            self.assertEqual(forge_modules._lora_files(paths["lora_dir"]), ["style.pt"])
            with patch.object(forge_modules, "get_forge_paths", return_value=paths):
                self.assertEqual(forge_modules.list_vae_files(), ["vae.sft"])
                self.assertEqual(forge_modules.list_te_files(), ["clip.bin"])

    def test_path_state_counts_use_the_same_scanners(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self._make_dirs(root)
            (paths["checkpoint_dir"] / "model.safetensors").write_bytes(b"")
            (paths["checkpoint_dir"] / "sidecar.vae.safetensors").write_bytes(b"")
            (paths["lora_dir"] / "style.safetensors").write_bytes(b"")
            (paths["lora_dir"] / "style.gguf").write_bytes(b"")
            (paths["vae_dir"] / "a" ).mkdir()
            (paths["vae_dir"] / "a" / "vae.sft").write_bytes(b"")
            (paths["vae_dir"] / "vae.sft").write_bytes(b"")

            with patch.object(forge_modules, "get_forge_paths", return_value=paths),                     patch.object(forge_modules, "get_default_forge_paths", return_value=paths):
                state = forge_modules.get_forge_path_state()

            counts = {key: entry["count"] for key, entry in state["entries"].items()}
            self.assertEqual(counts["checkpoint_dir"], 1)  # .vae. 사이드카 제외
            self.assertEqual(counts["lora_dir"], 1)        # LoRA 확장자만
            self.assertEqual(counts["vae_dir"], 1)         # 파일명 기준 중복 제거
            self.assertEqual(counts["text_encoder_dir"], 0)


if __name__ == "__main__":
    unittest.main()
