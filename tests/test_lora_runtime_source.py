"""생성 LoRA 단일 소스(_vue_lora_entries, 배율) — 부팅 복원 단위와 Vue 전송 계약.

감사 #2: 생성 payload 가 '활성 LoRA 가 있을 때만' 갱신되는 _vue_lora_text 미러를 읽어,
LoRA 를 전부 끄거나 지우면 부팅/직전 값의 LoRA 가 계속 붙었다.
감사 #28: 부팅 복원이 ui_prefs 의 정수 퍼센트를 배율 필드에 그대로 넣어 단위가 섞였다.
"""
from __future__ import annotations

import os
import re
import unittest
from types import SimpleNamespace

from core.lora_stack import UNIT_MULTIPLIER, append_lora_stack_to_prompt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class LoraRuntimeRestoreTests(unittest.TestCase):
    def _restore(self, prefs):
        from ui.generator_main import GeneratorMainUI
        host = SimpleNamespace(_apply_anima_guard_prefs=lambda _prefs: None)
        GeneratorMainUI._restore_runtime_prefs(host, prefs)
        return host

    def test_boot_restore_stores_multiplier_entries(self):
        host = self._restore({"loraStack": [
            {"name": "a", "weight": 85, "enabled": True, "triggerWords": []},
            {"name": "b", "weight": 100, "enabled": False, "triggerWords": []},
        ]})
        self.assertEqual([e["weight"] for e in host._vue_lora_entries], [0.85, 1.0])
        self.assertFalse(hasattr(host, "_vue_lora_text"))   # 미러는 더 이상 없다
        self.assertEqual(
            append_lora_stack_to_prompt("1girl", host._vue_lora_entries, unit=UNIT_MULTIPLIER),
            "1girl, <lora:a:0.85>",
        )

    def test_boot_restore_ignores_non_list_stack(self):
        host = self._restore({"loraStack": "broken"})
        self.assertFalse(hasattr(host, "_vue_lora_entries"))


class LoraBridgeContractTests(unittest.TestCase):
    def test_generate_always_resends_the_whole_stack(self):
        app = _read("frontend", "src", "App.vue")
        body = re.search(r"function _doGenerateNow\(\)\s*\{(.*?)\n\}", app, re.S)
        self.assertIsNotNone(body)
        self.assertIn("syncLoraStack()", body.group(1))
        # 빈 스택이면 건너뛰던 조건부 전송이 돌아오면 안 된다
        self.assertNotIn("if (loraText)", body.group(1))

    def test_lora_text_mirror_action_is_gone_on_both_sides(self):
        for parts in (("frontend", "src", "App.vue"), ("frontend", "src", "types", "bridge.d.ts"),
                      ("ui", "generator_main.py"), ("ui", "generator_generation.py")):
            # 주석 설명은 허용 — 액션 이름 리터럴(호출·핸들러·타입)만 없어야 한다
            self.assertNotRegex(_read(*parts), r"['\"]set_lora_text['\"]", parts)
        self.assertNotIn("_vue_lora_text", _read("ui", "generator_generation.py"))

    def test_vue_converts_units_only_through_lora_units(self):
        composable = _read("frontend", "src", "composables", "useLoraStack.js")
        self.assertIn("toBridgeLoraEntries(loraStack)", composable)
        self.assertIn("fromBridgeLoraEntries(", composable)
        # 프로파일 적용 결과를 ui_prefs 에 영속(안 하면 재시작 시 옛 스택으로 돌아갔다)
        handler = composable.split("onBackendEvent('loraStackLoaded'", 1)[1]
        self.assertIn("_saveLoraStack()", handler.split("})", 1)[0])

    def test_prefs_restore_pushes_stack_and_mount_does_not(self):
        # 웹 모드: ui_prefs 는 getInitialConfig 응답으로 늦게 오고 Python 은 새 클라이언트마다
        # _restore_runtime_prefs 를 돌리지 않는다. 마운트 시점의 낡은 localStorage 스택을 보내면
        # 공유 _vue_lora_entries 가 덮였고, 복원 뒤에 다시 보내지 않아 그대로 남았다.
        composable = _read("frontend", "src", "composables", "useLoraStack.js")
        body = re.search(r"function restoreFromPrefs\(prefs\)\s*\{(.*?)\n  \}", composable, re.S)
        self.assertIsNotNone(body)
        after_restore = body.group(1).split("nextTick(", 1)
        self.assertEqual(len(after_restore), 2, "복원 플래그가 풀린 뒤(nextTick) 전송해야 한다")
        self.assertRegex(after_restore[1], r"syncLoraStack\(\)\s*\n\s*\}\)\s*$")

        app = _read("frontend", "src", "App.vue")
        mounted = app.split("onMounted(async () => {", 1)
        self.assertEqual(len(mounted), 2)
        tail = mounted[1].split("\n</script>", 1)[0]
        code_lines = [ln for ln in tail.splitlines() if not ln.strip().startswith("//")]
        self.assertNotIn("syncLoraStack()", "\n".join(code_lines))


if __name__ == "__main__":
    unittest.main()
