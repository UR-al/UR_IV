// node:test for the LoRA picker's pure name helpers (web/ai_studio/lora_names.js). The golden table is shared with the Python side.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  appendTrigger,
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
