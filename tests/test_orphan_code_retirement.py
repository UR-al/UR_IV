"""회귀 방지(P13c): 호출자 없던 고아 모듈·사문 헬퍼를 되살리지 않는다.

감사 #83·#86·#93·#95·#134 — 라우트 없는 Vue 뷰, 연결된 적 없는 다운로더 초안, import 만 되던
위젯, 도입 때부터 부르는 곳이 없던 공개 헬퍼, Vite 스캐폴드 자산을 지웠다. 이름만 비슷한 살아 있는
짝(ModelDownloadManager·get_character_preset_full·escape_parentheses·TagIntelligence.copyright_of
·requestGalleryImages …)은 그대로 쓰인다.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

RETIRED_PATHS = (
    # #83 라우터에서 빠진 뒤 어디서도 import 되지 않던 뷰 — Web/Backend 탭은 네이티브 PyQt 위젯이다
    "frontend/src/views/BackendView.vue",
    "frontend/src/views/WebView.vue",
    # #86 core/model_downloads.ModelDownloadManager 가 대체한 초안
    "core/artifact_manager.py",
    "tests/test_artifact_manager.py",
    # #134 고아 위젯·유틸
    "widgets/preset_preview_dialog.py",
    "widgets/search_preview.py",
    "widgets/tag_weight_editor.py",
    "widgets/favorite_tags.py",
    "widgets/tag_input.py",
    "widgets/image_viewer.py",
    "ui/generator_search.py",
    "utils/feature_extractor.py",
    "utils/prompt_preset.py",
    "utils/xyz_plot.py",
    "assets/icons/check.svg",
    # #95 npm create vite 템플릿 잔재
    "frontend/src/assets",
    "frontend/public",
)


class RetiredFilesTests(unittest.TestCase):
    def test_retired_files_stay_deleted(self):
        present = [rel for rel in RETIRED_PATHS if (ROOT / rel).exists()]
        self.assertEqual(present, [])

    def test_frontend_readme_is_not_the_vite_template(self):
        text = (ROOT / "frontend" / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("This template should help get you started", text)
        for command in ("npm run build", "npm run test", "npm run test:node", "npm run type-check"):
            self.assertIn(command, text)

    def test_default_excludes_is_documented_as_reference_only(self):
        # 코드가 읽지 않는 참고 목록이다 — 앱 기능처럼 안내하지 않는다(파일은 사용자 자료라 남긴다)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        line = next(l for l in readme.splitlines() if "default_excludes.txt" in l)
        self.assertIn("참고용", line)

    def test_prompt_presets_json_is_no_longer_migrated(self):
        tree = ast.parse((ROOT / "config.py").read_text(encoding="utf-8"))
        names = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "_USER_DATA_FILES" for t in node.targets):
                names = {elt.value for elt in node.value.elts}
        self.assertIsNotNone(names)
        self.assertNotIn("prompt_presets.json", names)
        # 레거시 데이터 이전 목록은 유지한다(탭은 죽었어도 사용자 파일은 옮긴다)
        self.assertTrue({"event_gen_settings.json", "favorite_tags.json"} <= names)


class RetiredHelpersTests(unittest.TestCase):
    def test_common_widgets_drops_never_used_classes(self):
        # 모듈 자체가 은퇴했다 — 마지막 NoScrollComboBox 의 사용처(숨은 InpaintTab)가 사라졌다
        # (tests/test_legacy_i2i_inpaint_upscale_retirement.py). 되살아나면 옛 클래스는 없어야 한다.
        path = ROOT / "widgets" / "common_widgets.py"
        if path.exists():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
            for name in ("WheelEventFilter", "AutomationWidget", "SettingsDialog"):
                self.assertNotIn(name, classes)
        # 생성 창 기반·UI 구성은 더 이상 이 모듈에서 위의 클래스를 가져오지 않는다
        for rel in ("ui/generator_base.py", "ui/generator_ui_setup.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for name in ("WheelEventFilter", "AutomationWidget", "FavoriteTagsBar", "TagInputWidget"):
                self.assertNotIn(name, text, f"{rel}: {name}")

    def test_character_feature_lookup_does_not_keep_unread_profile_maps(self):
        from utils import character_features

        for name in ("lookup_multiple", "lookup_multiple_split", "get_copyright", "get_gender"):
            self.assertFalse(hasattr(character_features.CharacterFeatureLookup, name), name)
        with mock.patch.object(character_features, "get_tag_database", return_value=object()):
            lookup = character_features.CharacterFeatureLookup()
        # 3만여 프로필마다 채우고 아무도 읽지 않던 사전(약 10MB)
        self.assertFalse(hasattr(lookup, "_copyright"))
        self.assertFalse(hasattr(lookup, "_gender"))

    def test_unused_public_helpers_stay_removed(self):
        from core import path_safety, theme_presets
        from core.ui_state_manager import UIStateManager
        from utils import character_presets, prompt_cleaner
        from utils.tag_completer import TagCompleter

        self.assertFalse(hasattr(theme_presets, "preset_options"))
        for name in ("unregister", "registered_names", "get_cached", "set_cached"):
            self.assertFalse(hasattr(UIStateManager, name), name)
        self.assertFalse(hasattr(path_safety, "safe_output_file"))
        self.assertFalse(hasattr(TagCompleter, "get_all_tags"))
        self.assertFalse(hasattr(character_presets, "get_character_preset"))
        self.assertTrue(callable(character_presets.get_character_preset_full))
        for name in ("clean_prompt", "unescape_parentheses"):
            self.assertFalse(hasattr(prompt_cleaner, name), name)
        for name in ("get_options", "unescape_parentheses"):
            self.assertFalse(hasattr(prompt_cleaner.PromptCleaner, name), name)
        self.assertTrue(callable(prompt_cleaner.escape_parentheses))

    def test_caformer_tagger_drops_uncalled_helpers_but_result_keeps_tag_text(self):
        from core.image_captioning import CAFormerTagger, CaptionResult

        self.assertFalse(hasattr(CAFormerTagger, "tag_text"))
        self.assertFalse(hasattr(CAFormerTagger, "clear_session_cache"))
        # 이름이 같은 CaptionResult.tag_text 는 살아 있는 property(CaptionResult.text 가 쓴다)
        self.assertIsInstance(CaptionResult.__dict__["tag_text"], property)


if __name__ == "__main__":
    unittest.main()
