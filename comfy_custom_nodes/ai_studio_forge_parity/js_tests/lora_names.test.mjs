// node:test for the LoRA picker's pure name helpers (web/ai_studio/lora_names.js). The golden table is shared with the Python side.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  appendTrigger,
  familiesShown,
  familyOfModelFolder,
  familyOfRecipe,
  familyVisible,
  matchesQuery,
  middleEllipsis,
  previewTargets,
  parseLoraPath,
  shortLoraName,
} from "../web/ai_studio/lora_names.js";

const GOLDEN = JSON.parse(readFileSync(new URL("./fixtures/lora_names_golden.json", import.meta.url), "utf8"));
const BJUN = (n) => `LoKr\\묵밥 (@Bjunbae)\\@bjunbae_2.9b_lokr_came_b10_2e06_fp16\\@bjunbae_2.9b_lokr_came_b10_2e06_fp16-0000${n}.safetensors`;
const GT_FINAL = "LoKr\\GT_DunE\\@gt_dune_2.9b_lokr_came_b10_2e06_fp16\\@gt_dune_2.9b_lokr_came_b10_2e06_fp16.safetensors";

test("golden: every entry parses to its expected fields and label", () => {
  assert.ok(GOLDEN.length >= 30);
  assert.ok(GOLDEN.filter(({ expect }) => "modelFamily" in expect).length >= 16);
  for (const { entry, expect } of GOLDEN) {
    const p = parseLoraPath(entry);
    const got = {
      group: p.group,
      run: p.run,
      runShort: p.runShort,
      epoch: p.epoch,
      final: p.final,
      downloaded: p.downloaded,
      label: shortLoraName(entry),
    };
    // rows for the 11-16 [model/]algo/... layout also pin the model folder and the run key
    if ("model" in expect) got.model = p.model;
    if ("runKey" in expect) got.runKey = p.runKey;
    // the family the model folder alone gives (= lora_index.model_folder_family)
    if ("modelFamily" in expect) got.modelFamily = familyOfModelFolder(p.model);
    // rows with non-ASCII whitespace pin the trim / \s semantics the Python side mirrors
    if ("groupLabel" in expect) got.groupLabel = p.groupLabel;
    if ("recipe" in expect) got.recipe = p.recipe;
    assert.deepStrictEqual(got, expect, entry);
  }
});

test("epoch files and the final file of one run share a runKey, in both layouts", () => {
  assert.equal(parseLoraPath(BJUN(14)).runKey, parseLoraPath(BJUN(15)).runKey);
  assert.equal(parseLoraPath(BJUN(14)).runKey, "LoKr/묵밥 (@Bjunbae)/@bjunbae_2.9b_lokr_came_b10_2e06_fp16");
  const old1 = parseLoraPath("LoKr\\묵밥 (@Bjunbae)\\@bjunbae_3.8b_anima-000001.safetensors");
  const old2 = parseLoraPath("LoKr/묵밥 (@Bjunbae)/@bjunbae_3.8b_anima.safetensors");
  assert.equal(old1.runKey, old2.runKey);
  assert.deepEqual(parseLoraPath(GT_FINAL).recipe, ["2.9b", "lokr", "came", "b10", "2e06"]);
  assert.equal(parseLoraPath(GT_FINAL).groupLabel, "GT_DunE");
  assert.equal(parseLoraPath(BJUN(14)).groupLabel, "묵밥");
});

test("model folder layout: each run folder of a group is its own runKey (11-16 D:\\LoRA\\ANIMA\\...)", () => {
  // the bug: ANIMA was read as the algo, LoKr as the group and the group folder as the run -> one card, every epoch 3x
  const g = "ANIMA\\LoKr\\\u0eca (@9an9ta)\\";
  const runs = ["@na-eun_2.9b_lokr_came_b10_2e06", "@na-eun_2.9b_lokr_came_b10_2e06_fp16", "@na-eun_2.9b_lokr_came_b10_2e06_fp16_v2"];
  const keys = new Set(runs.flatMap((r) => [1, 2].map((n) => parseLoraPath(`${g}${r}\\${r}-00000${n}.safetensors`).runKey)));
  assert.equal(keys.size, 3);
  const p = parseLoraPath(`${g}${runs[0]}\\${runs[0]}-000002.safetensors`);
  assert.deepEqual([p.model, p.algo, p.group, p.run, p.epoch], ["ANIMA", "LoKr", "\u0eca (@9an9ta)", runs[0], 2]);
  // the same algo/group/run under two model folders stays apart; the old layout keeps its key
  assert.notEqual(parseLoraPath("ANIMA/LoKr/g/r/r-000001.safetensors").runKey, parseLoraPath("MOSS/LoKr/g/r/r-000001.safetensors").runKey);
  assert.equal(parseLoraPath("LoKr/g/r/r-000001.safetensors").runKey, "LoKr/g/r");
  assert.equal(parseLoraPath("LoKr/g/r/r-000001.safetensors").model, "");
  // previewTargets keeps the three runs apart too
  const entries = runs.map((r) => `${g}${r}\\${r}-000001.safetensors`);
  const text = runs.map((r) => `<lora:${r}-000001:1>`).join(" ");
  assert.equal(previewTargets(text, entries).length, 3);
});

test("final label uses num_epochs when given", () => {
  assert.equal(shortLoraName(GT_FINAL, { num_epochs: 15 }), "GT_DunE · @gt_dune fp16 · 최종 e15");
  assert.equal(shortLoraName(BJUN(14), { num_epochs: 15 }), "묵밥 · @bjunbae fp16 · e14");
});

test("middleEllipsis keeps the epoch tail", () => {
  const text = "아주 긴 그룹 이름 · @some_very_long_trigger_name fp16 extra words · e14";
  assert.ok(text.length >= 60);
  const cut = middleEllipsis(text, 30);
  assert.ok(cut.endsWith(" · e14"), cut);
  assert.ok(cut.length <= 30, cut);
  assert.ok(cut.includes("…"), cut);
  assert.equal(middleEllipsis("짧은 이름 · e3", 30), "짧은 이름 · e3");
  assert.ok(middleEllipsis(`${"x".repeat(50)} · 최종 e15`, 30).endsWith(" · 최종 e15"));
});

test("matchesQuery", () => {
  const e14 = { entry: BJUN(14) };
  const e15 = { entry: BJUN(15) };
  assert.equal(matchesQuery(e14, "bjun 14"), true);
  assert.equal(matchesQuery(e15, "bjun 14"), false);
  assert.equal(matchesQuery(e14, "bjun e14"), true);
  assert.equal(matchesQuery(e15, "bjun e14"), false);
  assert.equal(matchesQuery(e14, "BJUN"), true);
  assert.equal(matchesQuery(e14, "1"), false);
  assert.equal(matchesQuery(e14, "묵밥"), true);
  assert.equal(matchesQuery(e14, " _ "), true);
  assert.equal(matchesQuery(e14, "nothing"), false);
  assert.equal(matchesQuery({ entry: GT_FINAL, trigger: "@gt dune" }, "@gt dune"), true);
  assert.equal(matchesQuery({ entry: "LoRA\\HUSI\\HUSI.safetensors", trigger: "@gt dune" }, "@gt dune"), true);
  assert.equal(matchesQuery({ entry: "LoRA\\HUSI\\HUSI.safetensors" }, "@gt dune"), false);
  assert.equal(matchesQuery({ entry: GT_FINAL }, "gt_dune"), true);
  // a final file has no "-0000NN" suffix: the index epoch (its header) answers the number word
  assert.equal(matchesQuery({ entry: GT_FINAL }, "gt 15"), false);
  assert.equal(matchesQuery({ entry: GT_FINAL, epoch: 15 }, "gt 15"), true);
  assert.equal(matchesQuery({ entry: GT_FINAL, epoch: 15 }, "gt e14"), false);
  assert.equal(matchesQuery({ entry: BJUN(14), epoch: null }, "bjun 14"), true); // no index epoch: the path's
});

test("matchesQuery leaves the model folder out: \"anima\" does not hit every run under ANIMA (11-16 review item 3)", () => {
  const run = (model, group, name, n) => ({ entry: `${model}\\LoKr\\${group}\\${name}\\${name}-00000${n}.safetensors` });
  const plainRun = run("ANIMA", "@enaa97", "@enaa97_2.9b_lokr_came", 3);
  assert.equal(matchesQuery(plainRun, "anima"), false);
  assert.equal(matchesQuery(run("MOSS", "@enaa97", "@enaa97_lokr", 2), "moss"), false);
  assert.equal(matchesQuery(run("ANIMA", "@enaa97", "@enaa97_3.8b_anima", 1), "anima"), true, "the run name says anima");
  assert.equal(matchesQuery(run("MOSS", "@enaa97", "@enaa97_moss", 2), "moss 2"), true);
  assert.equal(matchesQuery({ entry: "ANIMA\\LoRA\\_downloaded\\anima-rl\\anima-rl-v0.1.safetensors" }, "anima rl"), true, "group / file");
  assert.equal(matchesQuery({ ...plainRun, trigger: "@anima_fan" }, "anima"), true, "the trigger");
  // the rest of the path is searched as before: algorithm, _downloaded, group, run, file, epoch
  assert.equal(matchesQuery(plainRun, "lokr enaa97 came 3"), true);
  assert.equal(matchesQuery({ entry: "SDXL\\LoRA\\_downloaded\\Komano\\Komano_Civ.safetensors" }, "downloaded komano"), true);
  assert.equal(matchesQuery({ entry: "SDXL\\LoRA\\_downloaded\\Komano\\Komano_Civ.safetensors" }, "sdxl"), false);
  // without a model folder the first folder is the algorithm and stays in
  assert.equal(matchesQuery({ entry: "LoKr\\g\\r\\r-000001.safetensors" }, "lokr g r 1"), true);
  assert.equal(matchesQuery({ entry: "LoRA\\LoKr\\x-000003.safetensors" }, "lora lokr x"), true);
});

test("familyOfRecipe", () => {
  assert.equal(familyOfRecipe(["2.9b", "lokr"]), "2.9b");
  assert.equal(familyOfRecipe(["lokr", "3.8b"]), "3.8b");
  assert.equal(familyOfRecipe(["anima"]), null);
  assert.equal(familyOfRecipe([]), null);
});

test("familyVisible", () => {
  assert.equal(familyVisible(null, "2.9b", false), true);
  assert.equal(familyVisible("2.9b", "2.9b", false), true);
  assert.equal(familyVisible("3.8b", "2.9b", false), false);
  assert.equal(familyVisible("base", "2.9b", false), false);
  assert.equal(familyVisible("base", "2.9b", true), true);
});

test("familyOfModelFolder: ANIMA / MOSS / KREA2 / SDXL, case-blind and trimmed; _unsorted and other folders unknown", () => {
  assert.deepEqual(["ANIMA", "MOSS", "KREA2", "SDXL", "Sdxl", " KREA2　", "﻿MOSS"].map(familyOfModelFolder),
    ["anima", "moss", "krea2", "sdxl", "sdxl", "krea2", "moss"]);
  for (const other of ["_unsorted", "", null, undefined, "FLUX", "LoKr", "constructor", "toString", "__proto__"]) {
    assert.equal(familyOfModelFolder(other), null, String(other));
  }
});

test("familyVisible with the model-folder families: an Anima checkpoint hides MOSS / KREA2 / SDXL, a MOSS one all but MOSS", () => {
  for (const checkpoint of ["2.9b", "3.8b", "base"]) {
    for (const family of ["moss", "krea2", "sdxl"]) {
      assert.equal(familyVisible(family, checkpoint, false), false, `${family} on ${checkpoint}`);
      assert.equal(familyVisible(family, checkpoint, true), true, `${family} on ${checkpoint}, 모든 모델`);
    }
    assert.equal(familyVisible("anima", checkpoint, false), true, `an ANIMA folder without a size stays on ${checkpoint}`);
    assert.equal(familyVisible(null, checkpoint, false), true);
  }
  for (const family of ["2.9b", "3.8b", "base", "anima", "krea2", "sdxl"]) {
    assert.equal(familyVisible(family, "moss", false), false, `${family} on MOSS`);
    assert.equal(familyVisible(family, "moss", true), true);
  }
  assert.equal(familyVisible("moss", "moss", false), true);
  assert.equal(familyVisible(null, "moss", false), true, "_unsorted (unknown) stays");
  assert.equal(familyVisible("anima", "krea2", false), false);
  assert.equal(familyVisible("krea2", "krea2", false), true);
});

test("familiesShown names what familyVisible lets through besides the unknown ones", () => {
  assert.deepEqual(familiesShown("2.9b"), ["2.9b", "anima"]);
  assert.deepEqual(familiesShown("base"), ["base", "anima"]);
  assert.deepEqual(familiesShown("moss"), ["moss"]);
  const all = ["2.9b", "3.8b", "base", "anima", "moss", "krea2", "sdxl"];
  for (const checkpoint of ["2.9b", "3.8b", "base", "moss", "krea2"]) {
    assert.deepEqual(all.filter((family) => familyVisible(family, checkpoint, false)), familiesShown(checkpoint), checkpoint);
  }
});

test("appendTrigger", () => {
  assert.equal(appendTrigger("", "@gt dune"), "@gt dune");
  assert.equal(appendTrigger("a, b", "@gt dune"), "a, b, @gt dune");
  assert.equal(appendTrigger("a, @gt dune", "@gt dune"), "a, @gt dune");
  assert.equal(appendTrigger("a,", "@gt dune"), "a, @gt dune");
  assert.equal(appendTrigger("a, b", ""), "a, b");
});

test("previewTargets: epochs of one run give one target, unknown names are left out", () => {
  const entries = [
    "LoKr\\묵밥 (@bjunbae)\\mukbap_2.9b_lokr_came_b10\\mukbap_2.9b_lokr_came_b10-000014.safetensors",
    "LoKr\\묵밥 (@bjunbae)\\mukbap_2.9b_lokr_came_b10\\mukbap_2.9b_lokr_came_b10-000015.safetensors",
    "LoKr/other (@x)/other_run/other_run.safetensors",
  ];
  const text = "1boy, solo <lora:mukbap_2.9b_lokr_came_b10-000014:0.8> <lora:mukbap_2.9b_lokr_came_b10-000015:0.5> <lora:gone:1>\nSteps: 20";
  const targets = previewTargets(text, entries);
  assert.equal(targets.length, 1);
  assert.equal(targets[0].runKey, parseLoraPath(entries[0]).runKey);
  assert.ok(targets[0].label.length > 0 && !/ · e\d+$/.test(targets[0].label));
});

test("previewTargets: folder prefixes in the tag, several runs in tag order, none", () => {
  const entries = ["LoKr/a (@a)/run_a/run_a-000003.safetensors", "LoKr/b (@b)/run_b/run_b.safetensors"];
  const targets = previewTargets("x <lora:LoKr\\b (@b)\\run_b:1> <lora:run_a-000003:0.7>", entries);
  assert.deepEqual(targets.map((t) => t.runKey), [parseLoraPath(entries[1]).runKey, parseLoraPath(entries[0]).runKey]);
  assert.deepEqual(previewTargets("no tags here", entries), []);
  assert.deepEqual(previewTargets("", entries), []);
  assert.deepEqual(previewTargets(null, entries), []);
});
