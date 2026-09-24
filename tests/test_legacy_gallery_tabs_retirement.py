"""회귀 방지: 숨은 레거시 PyQt 탭 4종(EventGen·XYZ Plot·PNG Info·Gallery)과 그 사슬을 다시 만들지 않는다.

네 탭은 생성 직후 setParent(None) 으로 떼어져 한 번도 보이지 않았고, 라이브 코드는 이들의
시그널에 connect 만 했다(시그널은 숨은 버튼·썸네일에서만 emit). 그래서 탭 자체와
  - GalleryMixin(ui/generator_gallery.py: 숨은 썸네일 스트립·즐겨찾기 스트립·우클릭 메뉴),
  - 전용 위젯(thumbnail·image_viewer·stats_panel·folder_organizer·param_diff_dialog·
    similar_group_dialog), 워커(gallery_worker), core/database(두 번째 sqlite 연결), exifread,
  - 연결 전용 핸들러(handle_immediate_generation·_gallery_send_to_*·_handle_send_to_i2i/inpaint·
    load_base_prompt_to_event)
가 전부 사문이었다. 생성마다 돌던 헛작업(QPixmap 풀 디코드 + 축소, 평면 썸네일, 숨은
ThumbnailItem, generation_data)도 같이 없앴다(tests/test_process_new_image.py).

Vue 가 쓰는 것은 남는다 — receive_event_scenarios·handle_prompt_only_transfer·
_handle_immediate_generation_from_raw·event_data_load_worker·migrate_legacy_gallery_folder 호출.
search_tab 은퇴(tests/test_legacy_ui_retirement.py)와 같은 방식이다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RETIRED_FILES = (
    "tabs/event_gen_tab.py",
    "tabs/xyz_plot_tab.py",
    "tabs/pnginfo_tab.py",
    "tabs/gallery_tab.py",
    "ui/generator_gallery.py",
    "widgets/thumbnail.py",
    "widgets/image_viewer.py",
    "widgets/stats_panel.py",
    "widgets/folder_organizer.py",
    "widgets/param_diff_dialog.py",
    "widgets/similar_group_dialog.py",
    "workers/gallery_worker.py",
    "core/database.py",
)

RETIRED_MODULES = tuple(path[:-3].replace("/", ".") for path in RETIRED_FILES)

# 앱 코드 — tests/ 는 이 은퇴를 설명·검증하느라 이름을 쓴다
_APP_DIRS = ("ui", "tabs", "widgets", "workers", "core", "utils", "backends")


def _app_sources():
    for folder in _APP_DIRS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            yield path
    for path in sorted(ROOT.glob("*.py")):
        yield path


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _identifiers(source: str) -> set[str]:
    """코드가 실제로 쓰는 이름 — 주석·독스트링(설명)은 빼고 hasattr(self, 'x') 같은 문자열 이름은 넣는다."""
    import ast
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.update(node.name.split("."))
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(node.module.split("."))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
            names.add(node.value)
    return names


class LegacyGalleryTabsRetirementTests(unittest.TestCase):
    def test_retired_files_are_gone(self):
        for rel in RETIRED_FILES:
            with self.subTest(file=rel):
                self.assertFalse((ROOT / rel).exists())
        # 파일은 비우되 지우지 않는다 — 없으면 tabs 가 namespace package 가 된다
        self.assertTrue((ROOT / "tabs" / "__init__.py").exists())

    def test_no_app_module_imports_a_retired_module(self):
        pattern = re.compile(
            r"^\s*(?:from|import)\s+(" + "|".join(re.escape(m) for m in RETIRED_MODULES) + r")\b",
            re.MULTILINE,
        )
        offenders = []
        for path in _app_sources():
            match = pattern.search(path.read_text(encoding="utf-8"))
            if match:
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(1)}")
        self.assertEqual(offenders, [])

    def test_hidden_tabs_are_not_created_or_wired(self):
        setup = _identifiers(_read("ui/generator_ui_setup.py"))
        for name in ("EventGenTab", "XYZPlotTab", "PngInfoTab", "GalleryTab",
                     "event_gen_tab", "xyz_plot_tab", "png_info_tab", "gallery_tab",
                     "gallery_items", "gallery_layout", "btn_add_favorite", "exif_display",
                     "customContextMenuRequested"):
            with self.subTest(name=name):
                self.assertNotIn(name, setup)
        dead = {
            "event_gen_tab", "xyz_plot_tab", "png_info_tab", "gallery_tab", "GalleryMixin", "generator_gallery",
            "handle_immediate_generation",
            "_gallery_send_to_editor", "_gallery_send_to_i2i", "_gallery_send_to_inpaint",
            "_gallery_send_to_upscale", "_gallery_send_to_queue", "_gallery_send_to_compare",
            "_handle_send_to_i2i", "_handle_send_to_inpaint",
            "load_base_prompt_to_event", "setup_viewer_context_menu", "add_image_to_gallery",
            "_create_thumbnail", "generation_data", "_pending_xyz_info", "add_result_image",
            "_on_xyz_add_to_queue", "_on_xyz_start_generation",
            "add_to_favorites", "refresh_favorites", "clear_all_favorites", "gallery_items",
        }
        offenders = []
        for path in sorted((ROOT / "ui").glob("*.py")):
            found = sorted(dead & _identifiers(path.read_text(encoding="utf-8")))
            offenders.extend(f"{path.name}: {name}" for name in found)
        self.assertEqual(offenders, [])

    def test_hidden_i2i_and_inpaint_tabs_no_longer_feed_the_invisible_gallery(self):
        for rel in ("tabs/i2i_tab.py", "tabs/inpaint_tab.py", "ui/inpaint_actions.py"):
            with self.subTest(file=rel):
                names = _identifiers(_read(rel))
                self.assertNotIn("add_image_to_gallery", names)
                self.assertIn("result_image_size", names)

    def test_settings_restore_keeps_only_the_live_gallery_folder_migration(self):
        settings = _read("ui/generator_settings.py")
        self.assertNotIn("gallery_tab", settings)
        self.assertIn("migrate_legacy_gallery_folder(settings)", settings)

    def test_legacy_thumbnail_and_metadata_helpers_are_gone(self):
        image_utils = _read("core/image_utils.py")
        for name in ("def get_thumb_path", "def read_exif", "def exif_for_display", "def normalize_path"):
            with self.subTest(name=name):
                self.assertNotIn(name, image_utils)
        base = _read("ui/generator_base.py")
        self.assertNotIn("_create_thumbnail", base)
        self.assertNotIn("gallery_items", base)
        self.assertNotIn("generation_data", base)
        config_source = _read("config.py")
        self.assertIsNone(re.search(r"^DB_FILE\s*=", config_source, re.MULTILINE))

    def test_exifread_is_no_longer_a_dependency(self):
        from core.check_requirements import _parse_requirements
        names = {name.lower() for name, _spec in _parse_requirements(str(ROOT / "requirements.txt"))}
        self.assertNotIn("exifread", names)
        offenders = [str(path.relative_to(ROOT)) for path in _app_sources()
                     if re.search(r"^\s*import exifread|^\s*from exifread", path.read_text(encoding="utf-8"), re.M)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
