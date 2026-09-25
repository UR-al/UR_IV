"""core.sam3_assets — SAM3 BPE vocab 를 로컬(Forge classic 포함)에서 먼저 찾는지."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import sam3_assets
from core.sam3_assets import BPE_VOCAB_NAME, bpe_vocab_candidates, find_local_bpe_vocab


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


class Sam3BpeVocabLookupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.editor = self.root / "Editor_models"
        self.neo = self.root / "sd-webui-forge-neo"
        self.classic = self.root / "sd-webui-forge-classic"
        self.cache = self.root / "image_cache" / "sam3_assets"

    def tearDown(self):
        self._tmp.cleanup()

    def _candidates(self, **overrides):
        kwargs = dict(editor_models_dir=self.editor,
                      extension_roots=[self.neo / "extensions", self.classic / "extensions"],
                      package_assets=[], cache_dir=self.cache)
        kwargs.update(overrides)
        return bpe_vocab_candidates(**kwargs)

    def _asset(self, forge_root: Path, folder: str = "forge_sam3_extension") -> Path:
        return forge_root / "extensions" / folder / "assets" / BPE_VOCAB_NAME

    def test_forge_classic_extension_asset_is_found_without_forge_neo(self):
        # 예전 조회는 C:\sd-webui-forge-neo 만 봐서, classic 만 있는 PC 는 gated 다운로드로 넘어가 실패했다
        classic_asset = _touch(self._asset(self.classic))
        self.assertEqual(find_local_bpe_vocab(self._candidates()), str(classic_asset))

    def test_repo_named_install_and_other_extension_folders_are_found(self):
        # 앱의 '확장 저장소 설치'나 git clone 은 저장소 이름(sam-extra)을 폴더 이름으로 쓴다
        repo_named = _touch(self._asset(self.classic, "sam-extra"))
        self.assertEqual(find_local_bpe_vocab(self._candidates()), str(repo_named))
        repo_named.unlink()
        renamed = _touch(self._asset(self.classic, "my-sam3-fork"))
        self.assertEqual(find_local_bpe_vocab(self._candidates()), str(renamed))

    def test_order_editor_models_then_forge_then_package_then_cached_download(self):
        package = self.root / "site" / "assets" / BPE_VOCAB_NAME
        cached = self.cache / "models--facebook--sam3" / "snapshots" / "abc" / BPE_VOCAB_NAME
        _touch(cached)
        order = self._candidates(package_assets=[package])
        self.assertEqual(order, [
            self.editor / BPE_VOCAB_NAME,
            self._asset(self.neo),
            self._asset(self.neo, "sam-extra"),
            self._asset(self.classic),
            self._asset(self.classic, "sam-extra"),
            package,
            cached,
        ])
        # 존재하는 것 중 가장 앞의 것
        self.assertEqual(find_local_bpe_vocab(order), str(cached))
        _touch(self._asset(self.classic))
        self.assertEqual(find_local_bpe_vocab(order), str(self._asset(self.classic)))
        _touch(self.editor / BPE_VOCAB_NAME)
        self.assertEqual(find_local_bpe_vocab(order), str(self.editor / BPE_VOCAB_NAME))

    def test_duplicate_roots_are_listed_once_and_nothing_found_is_empty(self):
        order = self._candidates(extension_roots=[self.classic / "extensions", self.classic / "extensions"])
        self.assertEqual(order.count(self._asset(self.classic)), 1)
        self.assertEqual(find_local_bpe_vocab(order), "")

    def test_app_managed_forge_extension_folders_come_first(self):
        # 앱은 Forge 를 --data-dir <runtime>/forge/data 로 띄우고 data/extensions 를 설정한 확장 폴더
        # (없으면 shared/extensions)로 잇는다 — 관리형 설치의 확장은 releases/…/source 가 아니라 여기에 있다
        config = self.root / "backend_runtime.json"
        runtime_root = self.root / "managed_backends"
        custom = self.root / "my_extensions"
        config.write_text(json.dumps({"engines": {"forge": {"extensionDir": str(custom)}}}), encoding="utf-8")
        self.assertEqual(sam3_assets._runtime_forge_extension_roots(config, runtime_root), [
            custom,
            runtime_root / "forge" / "data" / "extensions",
            runtime_root / "forge" / "shared" / "extensions",
        ])
        config.write_text(json.dumps({"engines": {"forge": {}}}), encoding="utf-8")
        self.assertEqual(sam3_assets._runtime_forge_extension_roots(config, runtime_root), [
            runtime_root / "forge" / "data" / "extensions",
            runtime_root / "forge" / "shared" / "extensions",
        ])
        with mock.patch.object(sam3_assets, "_runtime_forge_extension_roots", return_value=[custom]), \
                mock.patch.object(sam3_assets, "forge_install_roots", return_value=[self.classic]):
            self.assertEqual(sam3_assets.forge_extension_roots(), [custom, self.classic / "extensions"])

    def test_runtime_config_linked_install_and_managed_release_are_roots(self):
        config = self.root / "backend_runtime.json"
        runtime_root = self.root / "managed_backends"
        config.write_text(json.dumps({"engines": {"forge": {
            "sourceMode": "existing", "existingRoot": str(self.classic), "release": "r1"}}}),
            encoding="utf-8")
        self.assertEqual(sam3_assets._runtime_forge_roots(config, runtime_root), [
            self.classic, runtime_root / "forge" / "releases" / "r1" / "source"])
        config.write_text("{broken", encoding="utf-8")
        self.assertEqual(sam3_assets._runtime_forge_roots(config, runtime_root), [])

    def test_forge_install_roots_include_both_default_installs_and_env_root(self):
        roots = sam3_assets.forge_install_roots({"FORGE_MODELS_ROOT": str(self.root / "custom" / "models")})
        self.assertIn(self.root / "custom", roots)
        self.assertIn(Path(r"C:\sd-webui-forge-neo"), roots)
        self.assertIn(Path(r"C:\sd-webui-forge-classic"), roots)


class SamRefinerUsesLocalVocabTests(unittest.TestCase):
    def setUp(self):
        from core import sam_refiner
        self.sam_refiner = sam_refiner
        self._saved = sam_refiner._SAM3_BPE_PATH
        sam_refiner._SAM3_BPE_PATH = None

    def tearDown(self):
        self.sam_refiner._SAM3_BPE_PATH = self._saved

    def test_local_vocab_skips_the_gated_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = _touch(Path(tmp) / BPE_VOCAB_NAME)
            with mock.patch.object(sam3_assets, "find_local_bpe_vocab", return_value=str(local)), \
                    mock.patch("huggingface_hub.hf_hub_download") as download:
                self.assertEqual(self.sam_refiner._find_sam3_bpe_vocab(), str(local))
            download.assert_not_called()

    def test_download_is_the_fallback_when_nothing_is_local(self):
        with mock.patch.object(sam3_assets, "find_local_bpe_vocab", return_value=""), \
                mock.patch("huggingface_hub.hf_hub_download", return_value="C:/dl/bpe.gz") as download, \
                mock.patch("os.makedirs"):
            self.assertEqual(self.sam_refiner._find_sam3_bpe_vocab(), "C:/dl/bpe.gz")
        self.assertEqual(download.call_args.kwargs["repo_id"], "facebook/sam3")
        self.assertEqual(download.call_args.kwargs["filename"], BPE_VOCAB_NAME)


if __name__ == "__main__":
    unittest.main()
