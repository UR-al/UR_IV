"""The LoRA axis of the X/Y/Z plot: which columns a block makes, and what a column does to a cell's prompt."""
import random
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import xyz_lora as xl
from comfy_custom_nodes.ai_studio_forge_parity.xyz_plan import plan_xyz

ENAA = "@enaa97_2.9b_lokr"
RAI = "@rai11ly_came_fp32"


def run_files(stem: str, last: int = 25, folder: str = "LoKr") -> list[str]:
    numbered = [f"{folder}/{stem}-{epoch:06d}.safetensors" for epoch in range(1, last)]
    return [*numbered, f"{folder}/{stem}.safetensors"]


FILES = [*run_files(ENAA), *run_files(RAI), *run_files("@enaa97_a"), "LoRA/style.safetensors"]


def row(lora="", strength=None, trigger="", epoch_mode="off", epoch=None):
    return {"lora": lora, "strength": strength, "trigger": trigger, "epoch_mode": epoch_mode, "epoch": epoch}


def block(*rows, find="", after=False):
    return {"find": find, "after": after, "rows": list(rows)}


def columns(settings, prompt=f"best, 1boy, @enaa97, solo <lora:{ENAA}-000014:1>", negative="", files=FILES, read_epoch=None):
    return xl.lora_columns(settings, prompt, negative, lambda: files, read_epoch)


class EpochSeriesTests(unittest.TestCase):
    def test_the_file_without_a_number_is_the_last_epoch(self):
        series = xl.epoch_series(f"LoKr/{ENAA}-000014.safetensors", FILES)
        self.assertEqual([item.number for item in series], list(range(1, 26)))
        self.assertEqual(series[13].entry, f"LoKr/{ENAA}-000014.safetensors")
        self.assertEqual(series[-1].entry, f"LoKr/{ENAA}.safetensors")
        self.assertTrue(series[-1].last)
        self.assertFalse(series[13].last)

    def test_the_last_epoch_follows_the_saving_interval(self):
        files = [f"a-{epoch:06d}.safetensors" for epoch in (2, 4, 6)] + ["a.safetensors"]
        self.assertEqual([item.number for item in xl.epoch_series("a.safetensors", files)], [2, 4, 6, 8])

    def test_the_file_can_say_which_epoch_it_is(self):
        files = ["a-000014.safetensors", "a.safetensors"]
        series = xl.epoch_series("a.safetensors", files, lambda entry: 25 if entry == "a.safetensors" else None)
        self.assertEqual([item.number for item in series], [14, 25])

    def test_a_file_that_says_nothing_useful_counts_by_the_interval(self):
        files = [f"a-{epoch:06d}.safetensors" for epoch in (2, 4, 6)] + ["a.safetensors"]
        for told in (None, 3, 6):
            with self.subTest(told=told):
                series = xl.epoch_series("a.safetensors", files, lambda entry: told)
                self.assertEqual(series[-1].number, 8)

    def test_a_run_of_one_file(self):
        self.assertEqual(xl.epoch_series("a.safetensors", ["a.safetensors"])[0].number, 1)
        self.assertEqual(xl.epoch_series("a.safetensors", ["a.safetensors"], lambda entry: 30)[0].number, 30)

    def test_another_run_and_another_folder_stay_out(self):
        files = ["x/a-000001.safetensors", "x/a.safetensors", "y/a-000002.safetensors", "x/a_b-000003.safetensors"]
        self.assertEqual([item.entry for item in xl.epoch_series("x/a.safetensors", files)], ["x/a-000001.safetensors", "x/a.safetensors"])

    def test_a_run_name_that_ends_like_a_number_keeps_its_last_file(self):
        files = ["d-251031-000001.safetensors", "d-251031-000002.safetensors", "d-251031.safetensors"]
        series = xl.epoch_series(files[0], files)
        self.assertEqual([item.entry for item in series], files)
        self.assertTrue(series[-1].last)

    def test_a_file_without_epochs_is_a_series_of_one(self):
        series = xl.epoch_series("LoRA/style.safetensors", FILES)
        self.assertEqual([item.entry for item in series], ["LoRA/style.safetensors"])

    def test_a_run_whose_last_epoch_is_gone_has_no_last(self):
        series = xl.epoch_series("a-000002.safetensors", ["a-000001.safetensors", "a-000002.safetensors"])
        self.assertEqual([(item.number, item.last) for item in series], [(1, False), (2, False)])


class ColumnTests(unittest.TestCase):
    def test_each_row_is_a_column_when_no_epoch_is_set(self):
        made = columns(block(row(f"{ENAA}-000014"), row(RAI, 0.8), row("style")))
        self.assertEqual([column.name for column in made], [f"{ENAA}-000014", RAI, "style"])
        self.assertEqual([column.strength for column in made], [None, 0.8, None])

    def test_last_epoch_runs_from_the_selected_file_to_the_one_without_a_number(self):
        made = columns(block(row(f"{ENAA}-000001", epoch_mode="last")), prompt=f"x <lora:{ENAA}-000001:1>")
        self.assertEqual(len(made), 25)
        self.assertEqual(made[0].name, f"{ENAA}-000001")
        self.assertEqual(made[-1].name, ENAA)

    def test_direct_runs_to_that_number(self):
        made = columns(block(row(f"{ENAA}-000014", epoch_mode="direct", epoch=20)))
        self.assertEqual([column.label for column in made], ["e14", "e15", "e16", "e17", "e18", "e19", "e20"])

    def test_direct_at_the_last_number_takes_the_file_without_a_number(self):
        made = columns(block(row(f"{ENAA}-000023", epoch_mode="direct", epoch=25)), prompt=f"x <lora:{ENAA}-000023:1>")
        self.assertEqual([column.label for column in made], ["e23", "e24", "last"])
        self.assertEqual(made[-1].name, ENAA)

    def test_direct_past_the_last_epoch_stops_the_plan(self):
        with self.assertRaises(xl.LoraAxisError) as caught:
            columns(block(row(f"{ENAA}-000024", epoch_mode="direct", epoch=26)), prompt=f"x <lora:{ENAA}-000024:1>")
        self.assertIn("마지막은 25", caught.exception.args[0])
        self.assertEqual(len(columns(block(row(f"{ENAA}-000024", epoch_mode="direct", epoch=25)),
                                     prompt=f"x <lora:{ENAA}-000024:1>")), 2)

    def test_direct_reaches_the_last_file_by_the_number_it_holds(self):
        cases = (
            (["a-000014", "a"], 25, 25, ["e14", "last"]),      # a pruned run
            (["a-000014", "a-000020", "a"], 25, 25, ["e14", "e20", "last"]),
            (["a-000014", "a-000018", "a-000020", "a"], 25, 25, ["e14", "e18", "e20", "last"]),  # an interval the gaps misread
        )
        for stems, told, target, labels in cases:
            with self.subTest(stems=stems):
                files = [f"{stem}.safetensors" for stem in stems]
                made = columns(block(row("a-000014", epoch_mode="direct", epoch=target)), prompt="x <lora:a-000014:1>",
                               files=files, read_epoch=lambda entry: told)
                self.assertEqual([column.label for column in made], labels)

    def test_a_saving_interval_ends_at_the_real_last_epoch(self):
        files = [f"a-{epoch:06d}.safetensors" for epoch in (2, 4, 6, 8)] + ["a.safetensors"]
        made = columns(block(row("a-000006", epoch_mode="direct", epoch=9)), prompt="x <lora:a-000006:1>", files=files,
                       read_epoch=lambda entry: 9)
        self.assertEqual([column.label for column in made], ["e6", "e8", "last"])
        with self.assertRaises(xl.LoraAxisError) as caught:
            columns(block(row("a-000006", epoch_mode="direct", epoch=10)), prompt="x <lora:a-000006:1>", files=files,
                    read_epoch=lambda entry: 9)
        self.assertIn("마지막은 9", caught.exception.args[0])

    def test_only_a_direct_row_asks_a_file_for_its_epoch(self):
        asked = []
        def read_epoch(entry):
            asked.append(entry)
            return None

        columns(block(row(f"{ENAA}-000024", epoch_mode="last"), row(RAI)), prompt=f"x <lora:{ENAA}-000024:1>", read_epoch=read_epoch)
        self.assertEqual(asked, [])
        columns(block(row(f"{ENAA}-000024", epoch_mode="direct", epoch=25)), prompt=f"x <lora:{ENAA}-000024:1>", read_epoch=read_epoch)
        self.assertEqual(asked, [f"LoKr/{ENAA}.safetensors"])

    def test_direct_with_a_bad_number_stops_the_plan(self):
        for bad in (0, -3, 2.5, "x", 10 ** 400, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(xl.LoraAxisError) as caught:
                columns(block(row(f"{ENAA}-000014", epoch_mode="direct", epoch=bad)))
            self.assertIn("에폭 번호", caught.exception.args[0])

    def test_direct_before_the_selected_epoch_stops_the_plan(self):
        with self.assertRaises(xl.LoraAxisError) as caught:
            columns(block(row(ENAA, epoch_mode="direct", epoch=20)), prompt=f"x <lora:{ENAA}:1>")
        self.assertIn("에폭을 바꿔", caught.exception.args[0])
        self.assertIn("20", caught.exception.args[0])

    def test_an_empty_direct_box_changes_nothing(self):
        for empty in (None, ""):
            made = columns(block(row(f"{ENAA}-000014", epoch_mode="direct", epoch=empty)))
            self.assertEqual([column.name for column in made], [f"{ENAA}-000014"])

    def test_a_row_expands_on_its_own(self):
        made = columns(block(row(f"{ENAA}-000024", epoch_mode="last"), row(f"{RAI}-000023", epoch_mode="last")),
                       prompt=f"x <lora:{ENAA}-000024:1>")
        self.assertEqual([column.name for column in made], [f"{ENAA}-000024", ENAA, f"{RAI}-000023", f"{RAI}-000024", RAI])

    def test_labels_name_the_file_when_runs_differ_and_the_strength_when_it_varies(self):
        made = columns(block(row(f"{ENAA}-000014", 1.0), row(RAI, 0.8)))
        self.assertEqual([column.label for column in made], [f"{ENAA}-000014 (1)", f"{RAI} (0.8)"])

    def test_a_file_without_a_number_is_called_last_only_among_numbered_files(self):
        self.assertEqual([c.label for c in columns(block(row("style")), prompt="x <lora:style:1>")], ["style"])
        self.assertEqual([c.label for c in columns(block(row(f"{ENAA}-000024"), row(ENAA)), prompt=f"x <lora:{ENAA}-000024:1>")],
                         ["e24", "last"])

    def test_columns_that_differ_by_their_trigger_alone_say_so(self):
        settings = block(row(f"{ENAA}-000014"), row(f"{ENAA}-000014", trigger="@rai11ly"), row(RAI), find="@enaa97")
        self.assertEqual([column.label for column in columns(settings)][:2], [f"{ENAA}-000014 / @enaa97", f"{ENAA}-000014 / @rai11ly"])

    def test_a_run_is_one_folder(self):
        files = [f"{folder}/a-{epoch:06d}.safetensors" for folder in ("LoKr", "TSM") for epoch in (1, 2)]
        files += ["LoKr/a.safetensors", "TSM/a.safetensors"]
        prompt = "x <lora:TSM/a-000001:1>"
        made = columns(block(row("TSM/a-000001"), row("LoKr/a-000001")), prompt=prompt, files=files)
        self.assertEqual([column.label for column in made], ["TSM/a-000001", "LoKr/a-000001"])
        made = columns(block(row("TSM/a-000001"), row("LoKr/a-000002")), prompt=prompt, files=files)
        self.assertEqual([column.label for column in made], ["a-000001", "a-000002"])

    def test_case_does_not_split_a_run(self):
        files = ["A-000001.safetensors", "a-000002.safetensors", "A.safetensors"]
        made = columns(block(row("a-000001", epoch_mode="last")), prompt="x <lora:a-000001:1>", files=files)
        self.assertEqual([column.label for column in made], ["e1", "e2", "last"])

    def test_a_same_run_strength_comparison_shows_the_strength(self):
        made = columns(block(row(f"{ENAA}-000014", 1.0), row(f"{ENAA}-000014", 0.5)))
        self.assertEqual([column.label for column in made], ["e14 (1)", "e14 (0.5)"])


class RefusalTests(unittest.TestCase):
    def refused(self, settings, **options):
        with self.assertRaises(xl.LoraAxisError) as caught:
            columns(settings, **options)
        return " ".join(caught.exception.args)

    def test_no_rows(self):
        self.assertIn("LoRA", self.refused(block()))

    def test_no_base_lora(self):
        self.assertIn("기준", self.refused(block(row(""))))

    def test_the_base_lora_must_be_in_the_prompt_exactly(self):
        self.assertIn("-00001", self.refused(block(row(f"{ENAA}-00001"))))

    def test_a_row_without_a_lora(self):
        self.assertIn("2번", self.refused(block(row(f"{ENAA}-000014"), row(""))))

    def test_a_tag_the_parser_does_not_load_is_not_the_base(self):
        for prompt in ("<b:foo x <lora:style:1>", "x <LORA:style:1>"):
            with self.subTest(prompt=prompt):
                self.assertIn("style", self.refused(block(row("style"), row(RAI)), prompt=prompt))

    def test_a_row_whose_file_is_gone_is_named(self):
        self.assertIn("2번: gone 파일", self.refused(block(row(f"{ENAA}-000014"), row("gone"))))

    def test_a_name_a_tag_cannot_hold_is_refused(self):
        files = ["base.safetensors", "a=b.safetensors"]
        self.assertIn("a=b", self.refused(block(row("base"), row("a=b")), prompt="x <lora:base:1>", files=files))

    def test_rows_of_the_wrong_type(self):
        for rows in (5, True, "x", None, {}):
            with self.subTest(rows=rows):
                self.assertIn("줄이 없습니다", self.refused({"rows": rows}))

    def test_a_find_text_that_is_not_there(self):
        self.assertIn("@nobody", self.refused(block(row(f"{ENAA}-000014"), row(RAI, trigger="@x"), find="@nobody")))

    def test_a_find_text_is_checked_only_when_a_row_uses_it(self):
        self.assertEqual(len(columns(block(row(f"{ENAA}-000014"), row(RAI), find="@nobody"))), 2)

    def test_the_find_text_must_be_in_the_positive_prompt(self):
        for after in (False, True):
            with self.subTest(after=after):
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@x"), find="watermark", after=after)
                self.assertIn("watermark", self.refused(settings, negative="watermark"))

    def test_the_find_text_inside_a_lora_name_does_not_count(self):
        name = "@enaa97_2.9b-000014"
        settings = block(row(name), row(RAI, trigger="@r"), find="@enaa97")
        self.assertIn("@enaa97", self.refused(settings, prompt=f"best, 1boy <lora:{name}:1>"))

    def test_a_dash_while_inserting_needs_no_find_text(self):
        self.assertEqual(len(columns(block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@nobody", after=True))), 2)

    def test_a_missing_file_is_named(self):
        self.assertIn("gone-000001", self.refused(block(row("gone-000001", epoch_mode="last")), prompt="x <lora:gone-000001:1>"))


class FileListTests(unittest.TestCase):
    def test_windows_separators(self):
        files = [f"LoKr\\run\\a-{epoch:06d}.safetensors" for epoch in (1, 2)] + ["LoKr\\run\\a.safetensors"]
        made = columns(block(row("a-000001", epoch_mode="last")), prompt="x <lora:a-000001:1>", files=files)
        self.assertEqual([column.name for column in made], ["a-000001", "a-000002", "a"])

    def test_a_name_used_in_two_folders_keeps_its_folder(self):
        files = ["old/a-000001.safetensors", "new/a-000001.safetensors", "new/a-000002.safetensors", "new/a.safetensors"]
        made = columns(block(row("new/a-000001", epoch_mode="last")), prompt="x <lora:new/a-000001:1>", files=files)
        self.assertEqual([column.name for column in made], ["new/a-000001", "a-000002", "a"])


class ApplyTests(unittest.TestCase):
    PROMPT = f"best, 1boy, @enaa97, solo <lora:{ENAA}-000014:1> <lora:style:0.7>"

    def applied(self, settings, index=1, prompt=None, negative=""):
        state = {"prompt": prompt or self.PROMPT, "negative": negative}
        made = xl.lora_columns(settings, state["prompt"], state["negative"], lambda: FILES)
        xl.apply_column(state, settings, made[index])
        return state

    def test_only_the_named_tag_changes(self):
        settings = block(row(f"{ENAA}-000014"), row(f"{ENAA}-000015"))
        self.assertEqual(self.applied(settings)["prompt"], f"best, 1boy, @enaa97, solo <lora:{ENAA}-000015:1> <lora:style:0.7>")

    def test_a_longer_name_is_not_this_tag(self):
        prompt = f"x <lora:{ENAA}-0000140:1> <lora:{ENAA}-000014:1>"
        settings = block(row(f"{ENAA}-000014"), row(RAI))
        self.assertEqual(self.applied(settings, prompt=prompt)["prompt"], f"x <lora:{ENAA}-0000140:1> <lora:{RAI}:1>")

    def test_the_strength_of_the_row_goes_into_the_tag(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, 0.8))
        self.assertIn(f"<lora:{RAI}:0.8>", self.applied(settings)["prompt"])

    def test_the_base_strength_applies_to_rows_without_their_own(self):
        settings = block(row(f"{ENAA}-000014", 0.6), row(RAI))
        self.assertIn(f"<lora:{RAI}:0.6>", self.applied(settings)["prompt"])

    def test_without_a_strength_the_tag_keeps_its_own_numbers(self):
        prompt = f"x <lora:{ENAA}-000014:0.5:0.7>"
        settings = block(row(f"{ENAA}-000014"), row(RAI))
        self.assertEqual(self.applied(settings, prompt=prompt)["prompt"], f"x <lora:{RAI}:0.5:0.7>")

    def test_the_base_column_can_change_only_its_strength(self):
        settings = block(row(f"{ENAA}-000014", 0.5), row(RAI))
        self.assertIn(f"<lora:{ENAA}-000014:0.5>", self.applied(settings, index=0)["prompt"])

    def test_a_tag_in_the_negative_changes_too(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI))
        self.assertIn(f"<lora:{RAI}:1>", self.applied(settings, prompt="x", negative=f"<lora:{ENAA}-000014:1>")["negative"])

    def test_a_name_after_named_arguments_is_the_same_tag(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI))
        self.assertIn(f"<lora:te=0.5:{RAI}>", self.applied(settings, prompt=f"x <lora:te=0.5:{ENAA}-000014>")["prompt"])

    def test_a_strength_replaces_named_arguments(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, 0.8))
        self.assertIn(f"<lora:{RAI}:0.8>", self.applied(settings, prompt=f"x <lora:te=0.5:{ENAA}-000014>")["prompt"])

    def test_a_tag_the_parser_does_not_load_stays(self):
        settings = block(row("style"), row(RAI))
        self.assertEqual(self.applied(settings, prompt="x <LORA:style:1> <lora:style:0.5>")["prompt"], f"x <LORA:style:1> <lora:{RAI}:0.5>")

    def test_a_trigger_that_makes_a_tag_stops_the_plan(self):
        cases = (("<3", "@enaa97", "best, (@enaa97:1.2), solo"), (">_<", "sm", "best, (smile:1.2), solo"),
                 ("a:b", "@enaa97", "best, <@enaa97>, solo"), ("<lora:style:1>", "@enaa97", "best, @enaa97, solo"))
        for trigger, find, prompt in cases:
            with self.subTest(trigger=trigger), self.assertRaises(xl.LoraAxisError) as caught:
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger=trigger), find=find)
                self.applied(settings, prompt=f"{prompt} <lora:{ENAA}-000014:1>")
            self.assertIn(trigger, caught.exception.args[0])

    def test_a_trigger_with_a_bracket_that_breaks_nothing_is_fine(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="<3"), find="solo", after=True)
        self.assertIn("solo, <3", self.applied(settings)["prompt"])

    def test_a_tag_with_spaces_around_its_name_is_the_same_tag(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI))
        self.assertIn(f"<lora:{RAI}:0.5>", self.applied(settings, prompt=f"x <lora: {ENAA}-000014 :0.5>")["prompt"])

    def test_an_empty_trigger_keeps_the_prompt(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI), find="@enaa97")
        self.assertIn("1boy, @enaa97, solo", self.applied(settings)["prompt"])

    def test_a_new_trigger_replaces_the_find_text(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@rai11ly"), find="@enaa97")
        self.assertIn("1boy, @rai11ly, solo", self.applied(settings)["prompt"])

    def test_replacing_a_trigger_leaves_every_lora_tag_alone(self):
        prompt = "1boy, @enaa97, solo <lora:@enaa97_a-000001:1> <lora:@enaa97_b:0.5>"
        settings = block(row("@enaa97_a-000001"), row("@enaa97_a-000002", trigger="@rai11ly"), find="@enaa97")
        self.assertEqual(self.applied(settings, prompt=prompt)["prompt"],
                         "1boy, @rai11ly, solo <lora:@enaa97_a-000002:1> <lora:@enaa97_b:0.5>")

    def test_removing_a_trigger_leaves_every_lora_tag_alone(self):
        prompt = "1boy, @enaa97, solo <lora:@enaa97_a-000001:1>"
        settings = block(row("@enaa97_a-000001"), row("@enaa97_a-000002", trigger="-"), find="@enaa97")
        self.assertEqual(self.applied(settings, prompt=prompt)["prompt"], "1boy, solo <lora:@enaa97_a-000002:1>")

    def test_inserting_skips_a_lora_name_holding_the_word(self):
        prompt = "<lora:@enaa97_a-000001:1> 1boy, @enaa97, solo"
        settings = block(row("@enaa97_a-000001"), row("@enaa97_a-000002", trigger="@r"), find="@enaa97", after=True)
        self.assertEqual(self.applied(settings, prompt=prompt)["prompt"], "<lora:@enaa97_a-000002:1> 1boy, @enaa97, @r, solo")

    def test_removing_part_of_an_item_keeps_its_neighbours_apart(self):
        cases = (
            ("best, @enaa97_v2, solo", "@enaa97", "best, _v2, solo"),
            ("best, @enaa97 style, solo", "@enaa97", "best,  style, solo"),
            ("best, by @enaa97, solo", "@enaa97", "best, by , solo"),
            ("1boy, solo", "boy", "1, solo"),
        )
        for prompt, find, wanted in cases:
            with self.subTest(prompt=prompt):
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find=find)
                state = self.applied(settings, prompt=f"{prompt} <lora:{ENAA}-000014:1>")
                self.assertEqual(state["prompt"], f"{wanted} <lora:{RAI}:1>")

    def test_removing_a_weighted_item_takes_all_of_it(self):
        for item in ("(@enaa97:1.2)", "( @enaa97 )", "((@enaa97))", "[@enaa97]", "(@enaa97:-0.5)"):
            with self.subTest(item=item):
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@enaa97")
                state = self.applied(settings, prompt=f"best, {item}, solo <lora:{ENAA}-000014:1>")
                self.assertEqual(state["prompt"], f"best, solo <lora:{RAI}:1>")

    def test_removing_the_first_or_last_item_keeps_the_space_around_a_tag(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@enaa97")
        self.assertEqual(self.applied(settings, prompt=f"best, @enaa97 <lora:{ENAA}-000014:1>")["prompt"], f"best <lora:{RAI}:1>")
        self.assertEqual(self.applied(settings, prompt=f"<lora:{ENAA}-000014:1> @enaa97, solo")["prompt"], f"<lora:{RAI}:1> solo")
        self.assertEqual(self.applied(settings, prompt=f"@enaa97 <lora:{ENAA}-000014:1>")["prompt"].strip(), f"<lora:{RAI}:1>")

    def test_removing_an_item_with_brackets_in_the_find_text(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="(@enaa97)")
        self.assertEqual(self.applied(settings, prompt=f"best, (@enaa97), solo <lora:{ENAA}-000014:1>")["prompt"], f"best, solo <lora:{RAI}:1>")

    def test_removing_a_word_from_a_group_keeps_the_group(self):
        cases = (
            ("x, (@a, b:1.1), y", "x, (b:1.1), y"),
            ("x, (b, @a:1.1), y", "x, (b:1.1), y"),
            ("x, (b, @a), y", "x, (b), y"),
            ("x, ((@a, b)), y", "x, ((b)), y"),
            ("x, [@a, b], y", "x, [b], y"),
            ("x, ((@a:1.1), b), y", "x, (b), y"),
        )
        for prompt, wanted in cases:
            with self.subTest(prompt=prompt):
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@a")
                state = self.applied(settings, prompt=f"{prompt} <lora:{ENAA}-000014:1>", negative=prompt)
                self.assertEqual(state["prompt"], f"{wanted} <lora:{RAI}:1>")
                self.assertEqual(state["negative"], wanted)

    def test_removing_a_weighted_word_leaves_no_weight_behind(self):
        cases = (("best, (@a:1.2) style, solo", "best,  style, solo"), ("best,\n(@a:1.2)\nsolo", "best,\n\nsolo"))
        for prompt, wanted in cases:
            with self.subTest(prompt=prompt):
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@a")
                self.assertEqual(self.applied(settings, prompt=f"{prompt} <lora:{ENAA}-000014:1>")["prompt"], f"{wanted} <lora:{RAI}:1>")

    def test_removing_a_word_said_twice(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@a")
        self.assertEqual(self.applied(settings, prompt=f"@a, @a, x, @a <lora:{ENAA}-000014:1>")["prompt"], f"x <lora:{RAI}:1>")

    def test_removing_a_find_text_that_holds_a_comma(self):
        prompt = f"best, blue hair, long hair, solo <lora:{ENAA}-000014:1>"
        cases = (("blue hair, long hair", "best, solo"), ("blue hair,", "best,  long hair, solo"),
                 (", solo", "best, blue hair, long hair"))
        for find, wanted in cases:
            with self.subTest(find=find):
                settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find=find)
                self.assertEqual(self.applied(settings, prompt=prompt)["prompt"], f"{wanted} <lora:{RAI}:1>")

    def test_a_removed_trigger_takes_a_tight_comma_with_it(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@enaa97")
        self.assertTrue(self.applied(settings, prompt=f"a,@enaa97,b <lora:{ENAA}-000014:1>")["prompt"].startswith("a,b "))

    def test_after_stops_at_the_end_of_the_line(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="1boy", after=True)
        self.assertTrue(self.applied(settings, prompt=f"1boy\nsolo, x <lora:{ENAA}-000014:1>")["prompt"].startswith("1boy, @r\nsolo"))

    def test_a_cell_that_lost_the_find_text_stops_the_plan(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="@enaa97", after=True)
        made = columns(settings)
        with self.assertRaises(xl.LoraAxisError):
            xl.apply_column({"prompt": f"gone <lora:{ENAA}-000014:1>", "negative": ""}, settings, made[1])

    def test_a_cell_that_lost_the_tag_stops_the_plan(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI))
        made = columns(settings)
        with self.assertRaises(xl.LoraAxisError):
            xl.apply_column({"prompt": "gone", "negative": ""}, settings, made[1])

    def test_a_negative_tag_that_is_no_lora_is_plain_text(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="@enaa97")
        state = self.applied(settings, negative="<bad:@enaa97>, lowres <lora:style:1>")
        self.assertEqual(state["negative"], "<bad:@r>, lowres <lora:style:1>")

    def test_the_base_column_keeps_the_trigger(self):
        settings = block(row(f"{ENAA}-000014", trigger="@ignored"), row(RAI, trigger="@rai11ly"), find="@enaa97")
        self.assertIn("1boy, @enaa97, solo", self.applied(settings, index=0)["prompt"])

    def test_a_dash_removes_the_find_text_with_its_comma(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="@enaa97")
        self.assertIn("best, 1boy, solo", self.applied(settings)["prompt"])
        first = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="best")
        self.assertTrue(self.applied(first)["prompt"].startswith("1boy, @enaa97"))

    def test_after_inserts_behind_the_whole_tag_holding_the_word(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@rai11ly"), find="boy", after=True)
        self.assertIn("best, 1boy, @rai11ly, @enaa97, solo", self.applied(settings)["prompt"])

    def test_after_goes_behind_an_item_that_goes_on_past_a_tag(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="1boy", after=True)
        self.assertEqual(self.applied(settings, prompt=f"best, 1boy <lora:{ENAA}-000014:1> solo, smile")["prompt"],
                         f"best, 1boy <lora:{RAI}:1> solo, @r, smile")
        self.assertEqual(self.applied(settings, prompt=f"best, 1boy <lora:{ENAA}-000014:1>, solo")["prompt"],
                         f"best, 1boy, @r <lora:{RAI}:1>, solo")

    def test_after_stops_before_a_windows_line_end(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="1boy", after=True)
        self.assertTrue(self.applied(settings, prompt=f"1boy\r\nsolo <lora:{ENAA}-000014:1>")["prompt"].startswith("1boy, @r\r\n"))

    def test_after_the_last_tag_stays_before_the_lora_tags(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@rai11ly"), find="solo", after=True)
        self.assertIn(f"solo, @rai11ly <lora:{RAI}:1>", self.applied(settings)["prompt"])

    def test_after_touches_only_the_first_place_and_only_the_positive_prompt(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="1boy", after=True)
        state = self.applied(settings, prompt=f"1boy, 1boy <lora:{ENAA}-000014:1>", negative="1boy")
        self.assertTrue(state["prompt"].startswith("1boy, @r, 1boy"))
        self.assertEqual(state["negative"], "1boy")

    def test_a_dash_while_inserting_does_nothing(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="1boy", after=True)
        self.assertIn("best, 1boy, @enaa97", self.applied(settings)["prompt"])

    def test_no_find_text_puts_the_trigger_in_front(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@rai11ly"))
        self.assertTrue(self.applied(settings)["prompt"].startswith("@rai11ly, best, 1boy"))

    def test_no_find_text_and_a_dash_does_nothing(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="-"))
        self.assertTrue(self.applied(settings)["prompt"].startswith("best, 1boy, @enaa97"))


class PreviewTests(unittest.TestCase):
    PROMPT = f"best, 1boy, @enaa97, solo <lora:{ENAA}-000014:0.7>"

    def preview(self, settings, prompt=PROMPT, negative=""):
        made = columns(settings, prompt=prompt, negative=negative)
        return [xl.column_preview(column, settings, prompt, negative) for column in made]

    def test_the_trigger_text_says_what_a_column_does(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI), row(RAI, trigger="@r"), row(RAI, trigger="-"), find="@enaa97")
        self.assertEqual([item["trigger"] for item in self.preview(settings)], ["@enaa97", "@enaa97", "@r", "지움"])

    def test_inserting_is_marked_with_a_plus(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="1boy", after=True)
        self.assertEqual([item["trigger"] for item in self.preview(settings)], ["없음", "+ @r"])
        settings = block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"))
        self.assertEqual([item["trigger"] for item in self.preview(settings)], ["없음", "+ @r"])

    def test_it_marks_the_columns_that_change_the_prompt(self):
        settings = block(row(f"{ENAA}-000014"), row(RAI), row(RAI, trigger="@r"), find="@enaa97")
        self.assertEqual([item["changed"] for item in self.preview(settings)], [False, False, True])

    def test_a_dash_that_does_nothing_is_not_marked(self):
        for settings in (block(row(f"{ENAA}-000014"), row(RAI, trigger="-"), find="1boy", after=True),
                         block(row(f"{ENAA}-000014"), row(RAI, trigger="-"))):
            with self.subTest(find=settings["find"]):
                self.assertEqual([item["changed"] for item in self.preview(settings)], [False, False])

    def test_it_lists_the_label_file_and_strength(self):
        item = self.preview(block(row(f"{ENAA}-000014", 0.8)))[0]
        self.assertEqual((item["label"], item["name"], item["strength"]), ("e14", f"{ENAA}-000014", "0.8"))

    def test_a_kept_strength_reads_the_tag(self):
        item = self.preview(block(row(f"{ENAA}-000014")))[0]
        self.assertEqual(item["strength"], "0.7")

    def test_a_kept_strength_reads_a_tag_in_the_negative(self):
        item = self.preview(block(row("style")), prompt="x", negative="<lora:style:0.3>")[0]
        self.assertEqual(item["strength"], "0.3")

    def test_a_kept_strength_is_the_one_that_loads(self):
        item = self.preview(block(row("style")), prompt="x <lora:style:0.3> <lora:style:0.6>")[0]
        self.assertEqual(item["strength"], "0.6")


class PlanTests(unittest.TestCase):
    STATE = {"prompt": "best, 1boy, @enaa97, solo", "negative": "", "unet": "a.safetensors",
             "loras": [{"on": True, "name": f"{ENAA}-000014", "strength": 1.0}]}

    def plan(self, settings, state=None, files=FILES):
        axes = [{"axis": "lora", "text": "", "lora": settings}]
        return plan_xyz(state or self.STATE, axes, {}, rng=random.Random(1), families={"a.safetensors": "2.9b"}, loras=files)

    def test_a_plan_of_the_panel_list_rewrites_each_cell(self):
        plan = self.plan(block(row(f"{ENAA}-000014"), row(RAI, trigger="@rai11ly"), find="@enaa97"))
        self.assertEqual(plan.errors, [])
        prompts = [cell.state["prompt"] for cell in plan.cells]
        self.assertEqual(prompts[0], f"best, 1boy, @enaa97, solo <lora:{ENAA}-000014:1>")
        self.assertEqual(prompts[1], f"best, 1boy, @rai11ly, solo <lora:{RAI}:1>")

    def test_epochs_make_the_columns_and_the_labels(self):
        plan = self.plan(block(row(f"{ENAA}-000014", epoch_mode="direct", epoch=16)))
        self.assertEqual(plan.labels["x"], ["e14", "e15", "e16"])
        self.assertEqual(len(plan.cells), 3)
        self.assertEqual(plan.infotext_extra["X Type"], "LoRA")
        self.assertEqual(plan.infotext_extra["X Values"], "e14, e15, e16")

    def test_the_plan_lists_the_columns_for_the_panel(self):
        plan = self.plan(block(row(f"{ENAA}-000014"), row(RAI)))
        self.assertEqual([item["name"] for item in plan.lora_preview["x"]], [f"{ENAA}-000014", RAI])

    def test_two_axes_cannot_take_the_same_tag(self):
        axes = [{"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row(RAI))},
                {"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row("style"))}]
        plan = plan_xyz(self.STATE, axes, {}, rng=random.Random(1), families={}, loras=FILES)
        self.assertEqual(plan.cells, [])
        self.assertIn("두 축", plan.errors[0])

    def test_a_refusal_says_nothing_is_made(self):
        plan = self.plan(block(row("nope")))
        self.assertEqual(plan.errors[-1], xl.NOT_MADE)

    def test_every_refusal_with_a_lora_axis_says_nothing_is_made(self):
        axes = [{"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row(RAI))}]
        gone = {**self.STATE, "prompt": "best <lora:gone:1>"}
        self.assertEqual(plan_xyz(gone, axes, {}, rng=random.Random(1), families={}, loras=FILES).errors[-1], xl.NOT_MADE)
        steps = [*axes, {"axis": "steps", "text": "x"}]
        self.assertEqual(plan_xyz(self.STATE, steps, {}, rng=random.Random(1), families={}, loras=FILES).errors[-1], xl.NOT_MADE)

    def test_a_plan_without_a_lora_axis_has_no_such_line(self):
        plan = plan_xyz(self.STATE, [{"axis": "steps", "text": "x"}], {}, rng=random.Random(1), families={}, loras=FILES)
        self.assertTrue(plan.errors)
        self.assertNotIn(xl.NOT_MADE, plan.errors)

    def test_bad_values_never_raise(self):
        for bad in (row(f"{ENAA}-000014", strength=10 ** 400), row(f"{ENAA}-000014", epoch_mode="direct", epoch=10 ** 400)):
            with self.subTest(bad=bad):
                self.plan(block(bad))
        for settings in ({"rows": 5}, {"rows": True}, {"rows": ["x"]}, {"find": 5, "after": "x", "rows": [row(f"{ENAA}-000014")]}):
            with self.subTest(settings=settings):
                self.plan(settings)

    def test_a_swap_onto_a_tag_another_axis_needs_stops_the_plan(self):
        state = {**self.STATE, "prompt": "x <lora:style:0.5>"}
        first = {"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row("style"))}
        second = {"axis": "lora", "text": "", "lora": block(row("style"), row(RAI))}
        for axes in ([first, second], [second, first]):
            plan = plan_xyz(state, axes, {}, rng=random.Random(1), families={}, loras=FILES)
            self.assertEqual(plan.cells, [])
            self.assertIn("이미", plan.errors[0])

    def test_the_planner_asks_the_file_for_its_epoch(self):
        files = ["a-000014.safetensors", "a.safetensors"]
        axes = [{"axis": "lora", "text": "", "lora": block(row("a-000014", epoch_mode="direct", epoch=25))}]
        state = {"prompt": "x <lora:a-000014:1>"}
        plan = plan_xyz(state, axes, {}, rng=random.Random(1), families={}, loras=files, lora_epoch=lambda entry: 25)
        self.assertEqual(plan.labels["x"], ["e14", "last"])

    def test_a_prompt_sr_axis_that_removes_the_find_text_stops_the_plan(self):
        axes = [{"axis": "prompt_sr", "text": "@enaa97, @nobody"},
                {"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row(RAI, trigger="@r"), find="@enaa97", after=True)}]
        plan = plan_xyz(self.STATE, axes, {}, rng=random.Random(1), families={}, loras=FILES)
        self.assertEqual(plan.cells, [])
        self.assertEqual(plan.errors[-1], xl.NOT_MADE)

    def test_the_order_puts_the_lora_axis_outside_cheap_axes(self):
        axes = [{"axis": "steps", "text": "10, 20"}, {"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row(RAI))}]
        plan = plan_xyz(self.STATE, axes, {}, rng=random.Random(1), families={}, loras=FILES)
        self.assertEqual(plan.order, [0, 1, 2, 3])

    def test_a_name_the_prompt_already_has_stops_the_plan(self):
        for state in ({**self.STATE, "prompt": "x <lora:style:0.5>"}, {**self.STATE, "negative": "<lora:style:1>"}):
            for name in ("style", "STYLE", "LoRA/style", "style.safetensors"):
                with self.subTest(state=state["prompt"], name=name):
                    plan = self.plan(block(row(f"{ENAA}-000014"), row(name)), state=state)
                    self.assertEqual(plan.cells, [])
                    self.assertIn("이미", plan.errors[0])

    def test_a_name_spelled_another_way_stops_an_epoch_row_too(self):
        files = [f"LoKr\\a-{epoch:06d}.safetensors" for epoch in (1, 2)] + ["LoKr\\a.safetensors"]
        axes = [{"axis": "lora", "text": "", "lora": block(row("a-000001", epoch_mode="last"))}]
        state = {"prompt": "x <lora:a-000001:1> <lora:LoKr/a-000002:0.5>"}
        plan = plan_xyz(state, axes, {}, rng=random.Random(1), families={}, loras=files)
        self.assertEqual(plan.cells, [])
        self.assertIn("a-000002", plan.errors[0])

    def test_the_base_column_is_no_collision(self):
        plan = self.plan(block(row(f"{ENAA}-000014"), row(RAI)))
        self.assertEqual(plan.errors, [])

    def test_a_tag_the_panel_wrote_twice_is_no_collision(self):
        plan = self.plan(block(row("style"), row(RAI)), state={"prompt": "<lora:style:1> x <lora:style:0.5>"})
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.cells[1].state["prompt"], f"<lora:{RAI}:1> x <lora:{RAI}:0.5>")

    def test_a_later_axis_that_brings_the_column_back_stops_the_plan(self):
        state = {"prompt": f"<lora:{ENAA}-000014:1>, best, solo"}
        lora = {"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row(RAI, 0.9))}
        replace = {"axis": "prompt_sr", "text": f"solo, <lora:{RAI}:0.3>"}
        for axes in ([lora, replace], [replace, lora]):
            plan = plan_xyz(state, axes, {}, rng=random.Random(1), families={}, loras=FILES)
            self.assertEqual(plan.cells, [])
            self.assertIn("이미", plan.errors[0])

    def test_a_refused_axis_comes_back_as_errors(self):
        plan = self.plan(block(row(ENAA, epoch_mode="direct", epoch=20)),
                         state={**self.STATE, "loras": [{"on": True, "name": ENAA, "strength": 1.0}]})
        self.assertEqual(plan.cells, [])
        self.assertIn("에폭을 바꿔", plan.errors[0])

    def test_an_axis_id_of_the_wrong_type_is_an_error_not_a_crash(self):
        for axis in (["lora"], {"a": 1}):
            with self.subTest(axis=axis):
                plan = plan_xyz(self.STATE, [{"axis": axis, "text": ""}], {}, rng=random.Random(1), families={}, loras=FILES)
                self.assertTrue(plan.errors)

    def test_a_missing_settings_block_is_an_error_not_a_crash(self):
        plan = plan_xyz(self.STATE, [{"axis": "lora", "text": ""}], {}, rng=random.Random(1), families={}, loras=FILES)
        self.assertEqual(plan.cells, [])
        self.assertTrue(plan.errors)

    def test_it_works_with_another_axis(self):
        axes = [{"axis": "lora", "text": "", "lora": block(row(f"{ENAA}-000014"), row(RAI))}, {"axis": "steps", "text": "10, 20"}]
        plan = plan_xyz(self.STATE, axes, {}, rng=random.Random(1), families={}, loras=FILES)
        self.assertEqual(len(plan.cells), 4)
        self.assertEqual([cell.state["steps"] for cell in plan.cells], [10, 10, 20, 20])


if __name__ == "__main__":
    unittest.main()
