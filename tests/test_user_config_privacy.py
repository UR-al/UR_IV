"""사용자 작성 설정 파일이 저장소에 휩쓸려 올라가지 않고 백업에는 들어가는지.

config/instruction_presets.json(대화/보조/스키마 지침 프리셋)과 config/global_weights.json
(글로벌 태그 가중치)이 .gitignore 에 없어서 광범위한 ``git add`` 로 공개 원격에 개인
프롬프트가 올라갈 수 있었고, 백업 목록에도 지침 프리셋이 빠져 있었다.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

from core.settings_backup import CONFIG_EXPORT_FILES, CONFIG_IMPORT_FILES

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIVATE_CONFIG_FILES = ("instruction_presets.json", "global_weights.json")


def _gitignore_rules() -> set[str]:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


class UserConfigPrivacyTests(unittest.TestCase):
    def test_private_config_files_are_gitignored(self):
        rules = _gitignore_rules()
        for name in PRIVATE_CONFIG_FILES:
            with self.subTest(name=name):
                self.assertIn(f"config/{name}", rules)

    def test_private_config_files_are_backed_up_and_restorable(self):
        for name in PRIVATE_CONFIG_FILES:
            with self.subTest(name=name):
                self.assertIn(name, CONFIG_EXPORT_FILES)
                self.assertIn(name, CONFIG_IMPORT_FILES)

    def test_instruction_presets_live_in_the_config_boundary(self):
        source = (ROOT / "core" / "instruction_presets.py").read_text(encoding="utf-8")
        self.assertIn("config_file('instruction_presets.json')", source)

    def test_global_weights_path_goes_through_config_file(self):
        """os.path.join(__file__ …) 로 따로 조립하면 저장 경계·백업 목록과 어긋나기 쉽다.

        저장은 generator_main 의 save_global_weights, 로드는 부팅 pull(vue_bridge.getInitialConfig —
        Qt·웹 모드 공통, 감사 #107)이다. 둘 다 config_file 을 거쳐야 한다.
        """
        tree = ast.parse((ROOT / "ui" / "generator_main.py").read_text(encoding="utf-8"))
        joined = []
        via_config_file = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            literals = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "global_weights.json" not in literals:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "config_file":
                via_config_file += 1
            else:
                joined.append(name)
        self.assertEqual(joined, [])
        self.assertGreaterEqual(via_config_file, 1, "저장(save_global_weights)은 config_file 을 써야 한다")

        bridge_source = (ROOT / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        loader = next(
            node for node in ast.walk(ast.parse(bridge_source))
            if isinstance(node, ast.FunctionDef) and node.name == "getInitialConfig"
        )
        body = ast.get_source_segment(bridge_source, loader) or ""
        self.assertIn("'global_weights.json'", body)
        self.assertIn("config_file(name)", body, "로드(getInitialConfig)도 config_file 을 써야 한다")
        self.assertNotIn("os.path.join", body)


if __name__ == "__main__":
    unittest.main()
