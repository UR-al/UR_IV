"""워크플로 프로파일 LoRA 단위 — collect → save → load → apply 왕복과 옛(version 1) 파일 호환.

감사 #28: _vue_lora_entries 에 퍼센트(부팅 복원)와 배율(set_lora_stack)이 섞여 저장됐고,
적용 시 Vue 가 ×100 을 한 번 더 해 <lora:x:95.00> 처럼 100배로 생성됐다.

저장 충돌: 이름 규칙상 같은 파일('Flux.' · 'flux' → Flux.json)은 확인받은 요청만 덮는다.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core import workflow_profiles as wp
from core.lora_stack import UNIT_MULTIPLIER, append_lora_stack_to_prompt
from ui.generator_main import GeneratorMainUI
from ui.model_download_actions import ModelDownloadActionsMixin


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, value):
        self.emitted.append(value)


class _Bridge:
    def __init__(self):
        self.loraStackLoaded = _Signal()


class _Text:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


class _Host:
    def __init__(self):
        self.vue_bridge = _Bridge()
        self.steps_input = _Text("28")
        self._vue_lora_entries = []


# config/profiles/ANIMA.json(version 1) 의 실제 LoRA 값 — 퍼센트로 저장돼 있다
_LEGACY_ANIMA = {
    "name": "ANIMA", "created_at": 1780805350, "version": 1,
    "fields": {"steps_input": "32"},
    "lora_stack": [
        {"name": "ANIMAV10_Lokr_@enast_243648", "weight": 95, "enabled": True, "triggerWords": []},
        {"name": "ANIMAV10_Lokr_@o6nine", "weight": 10, "enabled": True, "triggerWords": []},
        {"name": "v2ANIMAV10_Lokr_@saushi", "weight": 100, "enabled": True, "triggerWords": []},
        {"name": "ANIMAV10_Lokr_@msolush", "weight": 100, "enabled": True, "triggerWords": []},
        {"name": "v3ANIMAV10_Lokr_@isb47", "weight": 30, "enabled": True, "triggerWords": ["solo"]},
    ],
}


class WorkflowProfileLoraUnitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(wp, "_profiles_dir", return_value=Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_raw(self, data):
        path = Path(self._tmp.name) / f"{data['name']}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_legacy_percent_profile_is_normalized_on_apply_without_touching_file(self):
        path = self._write_raw(_LEGACY_ANIMA)
        before = path.read_bytes()
        host = _Host()
        result = wp.apply_profile_to_host(wp.load_profile("ANIMA"), host)
        self.assertIn("lora_stack", result["applied"])
        self.assertEqual([e["weight"] for e in host._vue_lora_entries], [0.95, 0.1, 1.0, 1.0, 0.3])
        pushed = json.loads(host.vue_bridge.loraStackLoaded.emitted[-1])
        self.assertEqual([e["weight"] for e in pushed], [0.95, 0.1, 1.0, 1.0, 0.3])
        self.assertEqual(pushed[4]["triggerWords"], ["solo"])
        prompt = append_lora_stack_to_prompt("1girl", host._vue_lora_entries, unit=UNIT_MULTIPLIER)
        self.assertIn("<lora:ANIMAV10_Lokr_@enast_243648:0.95>", prompt)
        self.assertNotIn("95.00", prompt)
        self.assertEqual(path.read_bytes(), before)   # 읽기 호환만 — 사용자 파일은 그대로

    def test_legacy_multiplier_profile_is_kept(self):
        self._write_raw({"name": "old", "version": 1, "fields": {},
                         "lora_stack": [{"name": "a", "weight": 0.8}, {"name": "b", "weight": 2.5}]})
        host = _Host()
        wp.apply_profile_to_host(wp.load_profile("old"), host)
        self.assertEqual([e["weight"] for e in host._vue_lora_entries], [0.8, 2.5])

    def test_collect_save_load_apply_round_trip_is_multiplier_version_2(self):
        source = _Host()
        source._vue_lora_entries = [
            {"name": "a", "weight": 0.65, "enabled": True, "triggerWords": []},
            {"name": "b", "weight": 0.05, "enabled": False, "triggerWords": ["tw"]},
        ]
        snap = wp.collect_from_host(source)
        self.assertTrue(wp.save_profile("round", snap["fields"], snap["lora_stack"]))
        stored = json.loads((Path(self._tmp.name) / "round.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["version"], wp.PROFILE_VERSION)
        self.assertEqual([e["weight"] for e in stored["lora_stack"]], [0.65, 0.05])

        target = _Host()
        wp.apply_profile_to_host(wp.load_profile("round"), target)
        self.assertEqual(target._vue_lora_entries, snap["lora_stack"])
        self.assertEqual(target.steps_input.value, "28")

    def test_version_2_small_weights_are_never_rescaled(self):
        # version 2 는 단위가 확정 — |w|>3 휴리스틱을 적용하지 않는다
        data = {"version": 2, "lora_stack": [{"name": "a", "weight": 0.02}]}
        self.assertEqual(wp.profile_lora_entries(data)[0]["weight"], 0.02)

    def test_profile_without_lora_stack_clears_stack(self):
        host = _Host()
        host._vue_lora_entries = [{"name": "keep?", "weight": 1.0, "enabled": True, "triggerWords": []}]
        wp.apply_profile_to_host({"version": 2, "fields": {}}, host)
        self.assertEqual(host._vue_lora_entries, [])
        self.assertEqual(json.loads(host.vue_bridge.loraStackLoaded.emitted[-1]), [])


# 입력 이름 → (기록되는 이름 = profile_saved_name, 실제 파일 이름 = _path 의 stem).
# frontend/src/utils/profileFileName.test.ts 가 같은 표로 Vue 쪽 사본(profileSavedName · profileFileKey)을
# 지킨다 — 한쪽 규칙만 바뀌면 둘 중 하나가 깨진다. _path 는 저장 때 이미 정규화된 이름을 한 번 더
# 정규화한다(64자에서 잘리며 드러난 끝의 점·공백, 문자를 지워 드러난 앞 공백은 두 번째에 지워진다 —
# 기록되는 이름과 파일 이름이 다른 행이 그 경우다).
PROFILE_NAME_GOLDEN = [
    ("Flux", "Flux", "Flux"),
    ("  Flux.", "Flux", "Flux"),
    ("FLUX..", "FLUX", "FLUX"),
    ("Fl/ux", "Flux", "Flux"),
    ("Flux?", "Flux", "Flux"),
    ('a\\b:c*d"e<f>g|h', "abcdefgh", "abcdefgh"),
    ("tab\there", "tabhere", "tabhere"),
    ("con", "_con", "_con"),
    ("CON.json", "_CON.json", "_CON.json"),
    ("lpt9.x", "_lpt9.x", "_lpt9.x"),
    ("console", "console", "console"),
    ("???", "프로파일", "프로파일"),
    ("   ", "프로파일", "프로파일"),
    (". .", "프로파일", "프로파일"),
    ("A" * 64 + "x", "A" * 64, "A" * 64),
    ("A" * 63 + ". b", "A" * 63 + ".", "A" * 63),
    ("A" * 63 + " b", "A" * 63 + " ", "A" * 63),
    ("/ Flux", " Flux", "Flux"),
    ("/ con", " con", "_con"),
    ("한글 프로파일", "한글 프로파일", "한글 프로파일"),
    ("\U0001F600" * 70, "\U0001F600" * 64, "\U0001F600" * 64),
    ("con" + "x" * 70, "con" + "x" * 61, "con" + "x" * 61),
]


class ProfileSaveConflictTests(unittest.TestCase):
    """규칙상 같은 파일을 가리키는 이름이 확인 없이 기존 프로파일을 덮던 버그(S2-appvue-split#1).

    Vue 는 목록과 글자 그대로만 비교했고 save_profile 은 기본이 덮어쓰기라, 'Flux.' · 'flux' ·
    'Fl/ux' · 64자 뒤만 다른 이름이 기존 Flux.json(체크포인트 · 프롬프트 · LoRA 스택)을 조용히 바꿨다.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(wp, "_profiles_dir", return_value=Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _save(self, name, model, **kw):
        return wp.save_profile(name, {"model_combo": model}, [], **kw)

    def _stored(self):
        return {p.name: json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(Path(self._tmp.name).glob("*.json"))}

    def test_saved_and_file_names_match_the_shared_golden_table(self):
        for typed, saved, stem in PROFILE_NAME_GOLDEN:
            self.assertEqual(wp.profile_saved_name(typed), saved, repr(typed))
            self.assertEqual(wp._path(saved).stem, stem, repr(typed))
            self.assertEqual(wp._saved_path(typed).stem, stem, repr(typed))

    def test_conflict_check_looks_at_the_file_save_writes_for_every_golden_name(self):
        # find_conflicting_profile 이 _path(typed)(규칙 한 번)를 보던 때 '/ Flux' · 'A'*63 + '. b' 는
        # '충돌 없음' 이 나온 뒤 save_profile(규칙 두 번 = 실제 파일)이 거절돼, 경고·새 목록 없이
        # '저장 실패' 만 떴다(S2-appvue-split#1-a). 행마다 빈 폴더에서 저장 → 같은 이름 재저장을 본다.
        for i, (typed, saved, stem) in enumerate(PROFILE_NAME_GOLDEN):
            folder = Path(self._tmp.name) / f"row{i}"
            folder.mkdir()
            with self.subTest(typed=typed), mock.patch.object(wp, "_profiles_dir", return_value=folder):
                self.assertIsNone(wp.find_conflicting_profile(typed))
                self.assertTrue(self._save(typed, "ORIGINAL"))
                self.assertEqual([p.stem for p in folder.glob("*.json")], [stem])
                self.assertEqual(wp.find_conflicting_profile(typed), saved)
                self.assertFalse(self._save(typed, "NEW"))
                # 목록에 보이는 이름(기록된 이름)으로도 같은 파일 — 불러오기가 방금 것을 돌려준다
                self.assertEqual(wp.find_conflicting_profile(saved), saved)
                self.assertEqual(wp.load_profile(saved)["fields"]["model_combo"], "ORIGINAL")

    def test_names_that_map_to_an_existing_file_are_refused_by_default(self):
        self.assertTrue(self._save("Flux", "ORIGINAL"))
        variants = ["Flux", "Flux ", "Flux.", "FLUX..", "Fl/ux", "Flux?", "F:lux", 'Flux"',
                    "/ Flux", ": Flux.", "|  Flux"]   # 문자를 지워 드러난 앞 공백 — 두 번째 규칙에서 지워진다
        if os.path.normcase("A") == os.path.normcase("a"):   # 대소문자를 가리지 않는 파일 시스템(Windows)
            variants += ["flux", "FLUX.."]
        for typed in variants:
            self.assertEqual(wp.find_conflicting_profile(typed), "Flux", typed)
            self.assertFalse(self._save(typed, "NEW"), typed)
        stored = self._stored()
        self.assertEqual(list(stored), ["Flux.json"])
        self.assertEqual(stored["Flux.json"]["fields"]["model_combo"], "ORIGINAL")

    def test_names_that_differ_only_past_64_characters_collide(self):
        first, second = "A" * 64 + "y", "A" * 64 + "x"
        self.assertTrue(self._save(first, "ORIGINAL"))
        self.assertEqual(wp.find_conflicting_profile(second), "A" * 64)
        self.assertFalse(self._save(second, "NEW"))
        self.assertEqual(self._stored()["A" * 64 + ".json"]["fields"]["model_combo"], "ORIGINAL")

    def test_a_cut_that_leaves_a_trailing_dot_or_space_finds_the_existing_file(self):
        # 'A'*63 + '. b' 는 64자에서 잘려 'A'*63 + '.' 로 기록되고 파일은 'A'*63 이다
        self.assertTrue(self._save("A" * 63, "ORIGINAL"))
        for typed in ("A" * 63 + ". b", "A" * 63 + " b", "A" * 63 + ". .x"):
            self.assertEqual(wp.find_conflicting_profile(typed), "A" * 63, typed)
            self.assertFalse(self._save(typed, "NEW"), typed)
        stored = self._stored()
        self.assertEqual(list(stored), ["A" * 63 + ".json"])
        self.assertEqual(stored["A" * 63 + ".json"]["fields"]["model_combo"], "ORIGINAL")

    def test_rename_onto_a_name_that_maps_to_an_existing_file_is_refused(self):
        # 예전엔 '/ Flux' 가 ' Flux.json' 으로 써져 통과했고, 그 이름(' Flux')으로 불러오기·삭제하면
        # 다른 프로파일 Flux.json 을 건드렸다
        self.assertTrue(self._save("Flux", "ORIGINAL"))
        self.assertTrue(self._save("Other", "OTHER"))
        for new in ("/ Flux", ": Flux.", "Flux."):
            self.assertFalse(wp.rename_profile("Other", new), new)
        stored = self._stored()
        self.assertEqual(list(stored), ["Flux.json", "Other.json"])
        self.assertEqual(stored["Flux.json"]["fields"]["model_combo"], "ORIGINAL")
        self.assertEqual(stored["Other.json"]["fields"]["model_combo"], "OTHER")

    def test_renamed_profile_is_the_file_its_listed_name_loads_and_deletes(self):
        self.assertTrue(self._save("Other", "OTHER"))
        self.assertTrue(wp.rename_profile("Other", "/ Pony"))
        self.assertEqual(list(self._stored()), ["Pony.json"])
        listed = [p["name"] for p in wp.list_profiles()]
        self.assertEqual(listed, [" Pony"])   # save_profile('/ Pony') 과 같은 기록 이름
        self.assertEqual(wp.load_profile(listed[0])["fields"]["model_combo"], "OTHER")
        self.assertEqual(wp.find_conflicting_profile("/ Pony"), " Pony")
        self.assertTrue(wp.delete_profile(listed[0]))
        self.assertEqual(self._stored(), {})

    def test_confirmed_overwrite_replaces_the_existing_file(self):
        self.assertTrue(self._save("Flux", "ORIGINAL"))
        self.assertTrue(self._save("Flux.", "NEW", overwrite=True))
        stored = self._stored()
        self.assertEqual(list(stored), ["Flux.json"])
        self.assertEqual(stored["Flux.json"]["fields"]["model_combo"], "NEW")
        self.assertEqual(stored["Flux.json"]["name"], "Flux")

    def test_a_new_name_has_no_conflict(self):
        self.assertTrue(self._save("Flux", "ORIGINAL"))
        self.assertIsNone(wp.find_conflicting_profile("Flux 2"))
        self.assertTrue(self._save("Flux 2", "NEW"))
        self.assertEqual(list(self._stored()), ["Flux 2.json", "Flux.json"])

    def test_conflict_reports_the_stored_display_name(self):
        # 옛 파일 — 파일 이름과 name 이 다를 수 있다(목록은 name 을 보여 준다)
        (Path(self._tmp.name) / "anima.json").write_text(
            json.dumps({"name": "ANIMA Pony", "fields": {}}, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(wp.find_conflicting_profile("anima."), "ANIMA Pony")
        (Path(self._tmp.name) / "broken.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(wp.find_conflicting_profile("broken"), "broken")


class _Notify:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


class _ActionHarness(ModelDownloadActionsMixin):
    _handle_vue_action = GeneratorMainUI._handle_vue_action
    _send_workflow_profiles_list = GeneratorMainUI._send_workflow_profiles_list

    def __init__(self, model):
        self.vue_bridge = SimpleNamespace(showNotification=_Notify(), workflowProfilesList=_Notify(),
                                          loraStackLoaded=_Signal())
        self.model_combo = _Text(model)
        self._vue_lora_entries = []

    def _handle_chat_action(self, _action, _payload):
        return False

    def _handle_creator_action(self, _action, _payload):
        return False


class WorkflowProfileSaveActionTests(unittest.TestCase):
    """workflow_profile_save 핸들러 — 확인받은 요청(overwrite: true)만 기존 파일을 덮는다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(wp, "_profiles_dir", return_value=Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertTrue(wp.save_profile("Flux", {"model_combo": "ORIGINAL"}, []))

    def _model(self):
        return json.loads((Path(self._tmp.name) / "Flux.json").read_text(encoding="utf-8"))["fields"]["model_combo"]

    def test_unconfirmed_collision_is_refused_with_a_warning_and_a_fresh_list(self):
        # '/ Flux' — 규칙을 두 번 거쳐야 Flux.json 이 되는 이름(예전엔 '저장 실패' 만 뜨고 목록이 안 왔다)
        for payload in ({"name": "Flux."}, {"name": "Fl/ux", "overwrite": False},
                        {"name": "Flux?", "overwrite": "true"}, {"name": "Flux ", "overwrite": 1},
                        {"name": "/ Flux"}, {"name": ": Flux.", "overwrite": False}):
            host = _ActionHarness("NEW")
            host._handle_vue_action("workflow_profile_save", payload)
            self.assertEqual(self._model(), "ORIGINAL", payload)
            self.assertEqual(host.vue_bridge.showNotification.emitted,
                             [("warning", "같은 파일 이름의 프로파일 'Flux'이(가) 이미 있어 저장하지 않았습니다")], payload)
            names = [p["name"] for p in json.loads(host.vue_bridge.workflowProfilesList.emitted[-1][0])]
            self.assertEqual(names, ["Flux"])

    def test_confirmed_overwrite_saves_and_names_what_was_stored(self):
        host = _ActionHarness("NEW")
        host._handle_vue_action("workflow_profile_save", {"name": "Flux.", "overwrite": True})
        self.assertEqual(self._model(), "NEW")
        self.assertEqual(host.vue_bridge.showNotification.emitted, [("success", "프로파일 저장: Flux")])

    def test_new_name_saves_under_the_sanitized_name(self):
        host = _ActionHarness("OTHER")
        host._handle_vue_action("workflow_profile_save", {"name": "SD/XL?", "overwrite": False})
        stored = json.loads((Path(self._tmp.name) / "SDXL.json").read_text(encoding="utf-8"))
        self.assertEqual((stored["name"], stored["fields"]["model_combo"]), ("SDXL", "OTHER"))
        self.assertEqual(host.vue_bridge.showNotification.emitted, [("success", "프로파일 저장: SDXL")])
        self.assertEqual(self._model(), "ORIGINAL")


if __name__ == "__main__":
    unittest.main()
