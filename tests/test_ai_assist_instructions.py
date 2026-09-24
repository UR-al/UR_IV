"""AI assist instruction contracts; temporary config and fake HTTP only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.ai_assist_instructions import FEATURES, compose_system_prompt, normalize_instructions
from core.ollama_client import OllamaClient, SYSTEM_PROMPTS


class InstructionCompositionTests(unittest.TestCase):
    def test_empty_instructions_leave_every_existing_system_prompt_byte_identical(self):
        normalized = normalize_instructions(None)
        self.assertEqual(normalized["common"], "")
        self.assertEqual(len(normalized["features"]), 9)
        for mode, base in SYSTEM_PROMPTS.items():
            with self.subTest(mode=mode):
                self.assertEqual(compose_system_prompt(base, mode, normalized), base)
        self.assertEqual(compose_system_prompt("caption base", "nl_caption", normalized,
                                              feature="auto_nl"), "caption base")

    def test_common_and_target_content_override_defaults_without_changing_output_contract(self):
        value = {"common": "Keep the named character.", "features": {
            "expand": "Do not add quality tags.", "negative": "Unrelated negative rule.",
        }}
        system = compose_system_prompt("Existing tag-only default", "expand", value)
        self.assertIn("Keep the named character.", system)
        self.assertIn("Do not add quality tags.", system)
        self.assertNotIn("Unrelated negative rule.", system)
        self.assertLess(system.index("Keep the named character."), system.index("Do not add quality tags."))
        self.assertIn("take precedence over conflicting default content rules", system)
        self.assertIn("Output ONLY comma-separated tags", system)

    def test_auto_caption_inherits_common_and_caption_once_without_other_features(self):
        value = {"common": "COMMON_SENTINEL", "features": {
            "nl_caption": "CAPTION_SENTINEL", "auto_nl": "AUTO_SENTINEL", "creative": "OTHER_SENTINEL",
        }}
        system = compose_system_prompt("base", "nl_caption", value, feature="auto_nl")
        for text in ("COMMON_SENTINEL", "CAPTION_SENTINEL", "AUTO_SENTINEL"):
            self.assertEqual(system.count(text), 1)
        self.assertNotIn("OTHER_SENTINEL", system)
        manual = compose_system_prompt("base", "nl_caption", value)
        self.assertIn("CAPTION_SENTINEL", manual)
        self.assertNotIn("AUTO_SENTINEL", manual)
        self.assertEqual(compose_system_prompt("base", "chat", value), "base")
        self.assertEqual(compose_system_prompt("base", "auto_nl", value), "base")
        self.assertEqual(compose_system_prompt("base", "expand", value, feature="negative"), "base")

    def test_each_feature_receives_only_its_rules_and_keeps_its_output_family(self):
        value = {"features": {name: f"RULE_{name}_END" for name in FEATURES}}
        contracts = {"expand": "comma-separated tags", "suggest": "comma-separated tags",
                     "nl2tags": "comma-separated tags", "negative": "NEGATIVE tags",
                     "nl_caption": "English caption", "nl_scene": "English scene description",
                     "translate": "translated text", "creative": "Resolution: WIDTHxHEIGHT"}
        for mode, contract in contracts.items():
            with self.subTest(mode=mode):
                system = compose_system_prompt("base", mode, value)
                self.assertIn(contract, system)
                self.assertIn(f"RULE_{mode}_END", system)
                for other in FEATURES:
                    if other != mode:
                        self.assertNotIn(f"RULE_{other}_END", system)
        self.assertEqual(compose_system_prompt("base", "expand", {
            "common": " \n\t", "features": {"expand": " "}}), "base")


class InstructionSchemaTests(unittest.TestCase):
    def test_strict_save_schema_rejects_invalid_types_lengths_and_unknown_keys(self):
        invalid = [None, [], "text", {"common": 1}, {"features": []}, {"unknown": "text"},
                   {"features": {"unknown": "text"}}, {"features": {"expand": None}},
                   {"common": "x" * 8001}, {"features": {"auto_nl": "x" * 8001}}]
        for value in invalid:
            with self.subTest(value=str(value)[:60]), self.assertRaises(ValueError):
                normalize_instructions(value, strict=True)
        value = {"common": " ", "features": {"expand": "x" * 8000}}
        normalized = normalize_instructions(value, strict=True)
        self.assertEqual(normalized["common"], " ")
        self.assertEqual(normalized["features"]["expand"], "x" * 8000)
        self.assertEqual(tuple(normalized["features"]), FEATURES)
        value["features"]["expand"] = "changed"
        self.assertEqual(normalized["features"]["expand"], "x" * 8000)

    def test_tolerant_read_ignores_unknown_invalid_fields_and_bounds_large_text(self):
        normalized = normalize_instructions({"common": "x" * 8001, "features": {
            "expand": 3, "suggest": "유사 태그", "unknown": "ignored", "auto_nl": "y" * 8001,
        }})
        self.assertEqual(normalized["common"], "x" * 8000)
        self.assertEqual(normalized["features"]["expand"], "")
        self.assertEqual(normalized["features"]["suggest"], "유사 태그")
        self.assertEqual(normalized["features"]["auto_nl"], "y" * 8000)
        self.assertNotIn("unknown", normalized["features"])


class InstructionPersistenceTests(unittest.TestCase):
    def test_save_merges_only_instructions_and_load_is_read_only(self):
        from core.ai_assist_instructions import load_instructions, save_instructions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_prefs.json"
            existing = {"schema_version": 1, "theme": "light", "modelPaths": ["shared/model"],
                        "iconAnimationStyle": "claude", "loraStack": [{"name": "user-lora"}]}
            path.write_text(json.dumps(existing), encoding="utf-8")
            saved = save_instructions({"common": "사용자 규칙", "features": {"expand": "확장 규칙"}}, path)
            disk = json.loads(path.read_text(encoding="utf-8"))
            for key, value in existing.items():
                self.assertEqual(disk[key], value)
            self.assertEqual(disk["aiAssistInstructions"], saved)
            before = path.read_bytes()
            self.assertEqual(load_instructions(path), saved)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(item.name for item in path.parent.iterdir()), ["ui_prefs.json"])
            saved["features"]["expand"] = "caller changed"
            self.assertEqual(load_instructions(path)["features"]["expand"], "확장 규칙")

    def test_corrupt_existing_preferences_are_read_tolerantly_but_never_overwritten(self):
        from core.ai_assist_instructions import load_instructions, save_instructions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_prefs.json"
            for raw in (b'{"secret": not valid JSON', b'[]', b'\xff\xfe\xff'):
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    self.assertEqual(load_instructions(path), normalize_instructions(None))
                    with self.assertRaises(RuntimeError):
                        save_instructions({"common": "new instructions"}, path)
                    self.assertEqual(path.read_bytes(), raw)
            path.unlink()
            self.assertEqual(load_instructions(path), normalize_instructions(None))
            self.assertFalse(path.exists())

    def test_invalid_instruction_save_and_publish_failure_preserve_existing_preferences(self):
        from core.ai_assist_instructions import save_instructions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_prefs.json"
            original = b'{"schema_version": 1, "otherSetting": "keep me"}'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                save_instructions({"features": {"expand": "x" * 8001}}, path)
            self.assertEqual(path.read_bytes(), original)
            with patch("core.config_migration.os.replace", side_effect=PermissionError("synthetic write failure")):
                with self.assertRaises(PermissionError):
                    save_instructions({"common": "new content"}, path)
            self.assertEqual(path.read_bytes(), original)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class InstructionHttpTests(unittest.TestCase):
    def setUp(self):
        # 지침 합성만 본다 — /api/show 능력 조회(think 제어)는 tests/test_ollama_think.py 가 검증.
        # 조회 요청이 끼면 side_effect 응답 순서가 밀리므로 '능력 모름'으로 고정한다.
        thinking = patch.object(OllamaClient, "thinking_mode", return_value="unknown")
        thinking.start()
        self.addCleanup(thinking.stop)

    def test_all_three_enhance_fallbacks_receive_the_same_composed_system_once(self):
        value = {"common": "COMMON_SENTINEL", "features": {
            "nl_caption": "CAPTION_SENTINEL", "auto_nl": "AUTO_SENTINEL", "negative": "OTHER_SENTINEL",
        }}
        responses = [_Response({"message": {}}), _Response({"message": {}}),
                     _Response({"response": "A bird is perched. Blue wings are visible."})]
        with patch("core.ollama_client.requests.post", side_effect=responses) as post:
            result = OllamaClient("http://unused.local", "fixture-model").enhance(
                "blue bird", "nl_caption", instructions=value, instruction_feature="auto_nl")
        self.assertEqual(result, "A bird is perched. Blue wings are visible.")
        calls = post.call_args_list
        self.assertEqual(len(calls), 3)
        first = calls[0].kwargs["json"]["messages"][0]["content"]
        merged = calls[1].kwargs["json"]["messages"][0]["content"]
        generated = calls[2].kwargs["json"]["system"]
        self.assertEqual(first, generated)
        self.assertTrue(merged.startswith(first + "\n\n"))
        for system in (first, merged, generated):
            for token in ("COMMON_SENTINEL", "CAPTION_SENTINEL", "AUTO_SENTINEL"):
                self.assertEqual(system.count(token), 1)
            self.assertNotIn("OTHER_SENTINEL", system)

    def test_blank_legacy_calls_and_unknown_mode_send_exact_existing_default(self):
        for mode in ("expand", "unknown-mode"):
            with self.subTest(mode=mode), patch("core.ollama_client.requests.post", return_value=_Response({
                "message": {"content": "blue_hair, red_eyes"},
            })) as post:
                client = OllamaClient("http://unused.local", "fixture-model")
                if mode == "expand":
                    result = client.enhance("input", mode, "extra input")
                else:
                    result = client.enhance("input", mode, instructions={"common": "DO_NOT_INJECT"})
                self.assertEqual(result, "blue_hair, red_eyes")
                self.assertEqual(post.call_args.kwargs["json"]["messages"][0]["content"], SYSTEM_PROMPTS["expand"])

    def test_translation_preserves_existing_plain_text_response_contract(self):
        with patch("core.ollama_client.requests.post", return_value=_Response({
            "message": {"content": "파란 새가 보입니다."},
        })) as post:
            result = OllamaClient("http://unused.local").enhance("A blue bird is visible.", "translate",
                instructions={"features": {"translate": "Keep the original meaning."}})
        self.assertEqual(result, "파란 새가 보입니다.")
        self.assertIn("Keep the original meaning.", post.call_args.kwargs["json"]["messages"][0]["content"])


class InstructionWorkerTests(unittest.TestCase):
    def test_worker_uses_creation_time_settings_snapshot_even_after_preferences_change(self):
        from core.ai_assist_instructions import save_instructions
        from workers.ollama_worker import OllamaWorker
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_prefs.json"
            with patch("core.ai_assist_instructions.config_file", return_value=path):
                save_instructions({"common": "BEFORE_SENTINEL", "features": {"auto_nl": "AUTO_SENTINEL"}})
                worker = OllamaWorker("http://unused.local", "fixture-model", "blue bird", "nl_caption",
                                      "", None, instruction_feature="auto_nl")
                save_instructions({"common": "AFTER_SENTINEL"})
                results, errors = [], []
                worker.finished.connect(results.append)
                worker.error.connect(errors.append)
                with patch("core.ollama_client.requests.post", return_value=_Response({
                    "message": {"content": "A blue bird is perched. Its wings are folded."},
                })) as post:
                    worker.run()
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(results[0])["mode"], "nl_caption")
            system = post.call_args.kwargs["json"]["messages"][0]["content"]
            self.assertIn("BEFORE_SENTINEL", system)
            self.assertIn("AUTO_SENTINEL", system)
            self.assertNotIn("AFTER_SENTINEL", system)

    def test_explicit_worker_instructions_are_detached_and_do_not_read_personal_settings(self):
        from workers.ollama_worker import OllamaWorker
        value = {"common": "ORIGINAL_SENTINEL", "features": {"expand": "FEATURE_SENTINEL"}}
        with patch("core.ai_assist_instructions.config_file", side_effect=AssertionError("must not read config")):
            worker = OllamaWorker("http://unused.local", "fixture-model", "blue bird", "expand", "", None,
                                  instructions=value)
            value["common"] = "MUTATED_SENTINEL"
            value["features"]["expand"] = "MUTATED_FEATURE"
            results = []
            worker.finished.connect(results.append)
            with patch("core.ollama_client.requests.post", return_value=_Response({
                "message": {"content": "blue_bird, folded_wings"},
            })) as post:
                worker.run()
        self.assertEqual(json.loads(results[0]), {"tags": "blue_bird, folded_wings", "mode": "expand"})
        system = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("ORIGINAL_SENTINEL", system)
        self.assertIn("FEATURE_SENTINEL", system)
        self.assertNotIn("MUTATED", system)


if __name__ == "__main__":
    unittest.main()
