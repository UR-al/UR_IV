import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# YOLO 목록 로직은 core/yolo_models.py 에만 있다(옛 tabs/editor/mosaic_panel 은 은퇴해 삭제됨).
from core import yolo_models


def _patched(root: Path):
    models = root / "Editor_models"
    return patch.multiple(
        yolo_models,
        _PROJECT_ROOT=str(root),
        _EDITOR_MODELS_DIR=str(models),
        _YOLO_CONFIG_PATH=str(models / "yolo_config.json"),
    )


class TestPortableYoloModelPaths(unittest.TestCase):
    def _patched_paths(self, root: Path):
        return _patched(root)

    def test_relative_config_resolves_on_current_project(self):
        with tempfile.TemporaryDirectory() as temp, self._patched_paths(Path(temp)):
            root = Path(temp)
            models = root / "Editor_models"
            models.mkdir()
            model = models / "detector.pt"
            model.write_bytes(b"model")
            (models / "yolo_config.json").write_text(
                json.dumps({"model_paths": ["Editor_models/detector.pt"]}),
                encoding="utf-8",
            )

            self.assertEqual(yolo_models.load_model_paths(), [str(model)])

    def test_old_absolute_path_migrates_by_filename(self):
        with tempfile.TemporaryDirectory() as temp, self._patched_paths(Path(temp)):
            root = Path(temp)
            models = root / "Editor_models"
            models.mkdir()
            model = models / "detector.pt"
            model.write_bytes(b"model")
            (models / "yolo_config.json").write_text(
                json.dumps({"model_paths": [r"X:\Users\Legacy\App\Editor_models\detector.pt"]}),
                encoding="utf-8",
            )

            self.assertEqual(yolo_models.load_model_paths(), [str(model)])

    def test_project_model_is_saved_as_forward_slash_relative_path(self):
        with tempfile.TemporaryDirectory() as temp, self._patched_paths(Path(temp)):
            root = Path(temp)
            models = root / "Editor_models"
            models.mkdir()
            model = models / "detector.pt"
            model.write_bytes(b"model")

            copies = []
            result = yolo_models.add_models([str(model)], copy=lambda *a: copies.append(a))

            saved = json.loads((models / "yolo_config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["model_paths"], ["Editor_models/detector.pt"])
            self.assertEqual(copies, [], "Editor_models 안의 파일은 복사하지 않는다")
            self.assertEqual(result, [str(model)])

    def test_yolo_helpers_live_only_in_core(self):
        """숨은 PyQt 에디터(tabs/editor/mosaic_panel 재-export shim)는 은퇴했다 — 호출처는 core 를 쓴다."""
        repo = Path(__file__).resolve().parents[1]
        self.assertFalse((repo / "tabs" / "editor").exists())
        for rel in ("ui/vue_bridge.py", "ui/generator_main.py"):
            source = (repo / rel).read_text(encoding="utf-8")
            for legacy in ("_load_yolo_model_paths", "_save_yolo_model_paths", "_is_sam_file",
                           "mosaic_panel", "refresh_yolo_models"):
                self.assertNotIn(legacy, source, f"{rel} 가 은퇴한 YOLO 헬퍼 {legacy} 를 쓴다")
        self.assertFalse(hasattr(yolo_models, "save_model_paths"),
                         "레거시 호환용 save_model_paths 는 호출처가 없어 지웠다 — add_models/clear_models 를 쓸 것")


class TestYoloModelClearAndAdd(unittest.TestCase):
    """'YOLO 모델 초기화'가 자동 감지 때문에 되살아나던 문제(audit #44)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.models = self.root / "Editor_models"
        self.models.mkdir()
        for name in ("penis.pt", "pussy.pt", "mobile_sam.pt", "sam3.safetensors"):
            (self.models / name).write_bytes(b"w")
        (self.models / "yolo_config.json").write_text(
            json.dumps({"model_paths": ["Editor_models/penis.pt"]}), encoding="utf-8")
        patcher = _patched(self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _config(self):
        return json.loads((self.models / "yolo_config.json").read_text(encoding="utf-8"))

    def test_auto_detects_folder_models_without_sam(self):
        names = [Path(p).name for p in yolo_models.load_model_paths()]
        self.assertEqual(names, ["penis.pt", "pussy.pt"])
        self.assertEqual(yolo_models.model_label(), "penis.pt, pussy.pt")

    def test_clear_really_clears_including_auto_detected(self):
        self.assertEqual(yolo_models.clear_models(), [])
        self.assertEqual(yolo_models.load_model_paths(), [])
        self.assertEqual(yolo_models.model_label(), yolo_models.NO_MODEL_LABEL)
        cfg = self._config()
        self.assertEqual(cfg["model_paths"], [])
        self.assertEqual(sorted(cfg["disabled_models"]), ["Editor_models/penis.pt", "Editor_models/pussy.pt"])

    def test_new_file_after_clear_is_still_detected(self):
        yolo_models.clear_models()
        (self.models / "face.onnx").write_bytes(b"w")
        self.assertEqual([Path(p).name for p in yolo_models.load_model_paths()], ["face.onnx"])

    def test_add_reenables_a_cleared_model_and_does_not_persist_autodetected(self):
        yolo_models.clear_models()
        result = yolo_models.add_models([str(self.models / "pussy.pt")])
        self.assertEqual([Path(p).name for p in result], ["pussy.pt"])
        cfg = self._config()
        self.assertEqual(cfg["model_paths"], ["Editor_models/pussy.pt"])
        self.assertEqual(cfg["disabled_models"], ["Editor_models/penis.pt"])

    def test_add_external_copies_once_and_reuses_same_named_file(self):
        outside = self.root / "downloads"
        outside.mkdir()
        (outside / "hand.pt").write_bytes(b"new")
        (outside / "penis.pt").write_bytes(b"other")
        copies = []

        def fake_copy(src, dst):
            copies.append((Path(src).name, Path(dst).parent))
            Path(dst).write_bytes(Path(src).read_bytes())

        yolo_models.clear_models()
        result = yolo_models.add_models([str(outside / "hand.pt"), str(outside / "penis.pt")], copy=fake_copy)
        self.assertEqual(copies, [("hand.pt", self.models)])
        self.assertEqual(sorted(Path(p).name for p in result), ["hand.pt", "penis.pt"])
        self.assertTrue(all(Path(p).parent == self.models for p in result), "외부 경로가 중복 로드되면 안 된다")
        self.assertEqual(sorted(self._config()["model_paths"]), ["Editor_models/hand.pt", "Editor_models/penis.pt"])

    def test_broken_config_falls_back_to_folder_scan(self):
        (self.models / "yolo_config.json").write_text("{not json", encoding="utf-8")
        self.assertEqual([Path(p).name for p in yolo_models.load_model_paths()], ["penis.pt", "pussy.pt"])


if __name__ == "__main__":
    unittest.main()
