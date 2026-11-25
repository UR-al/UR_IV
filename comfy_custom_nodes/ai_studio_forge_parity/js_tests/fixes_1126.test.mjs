// node:test for 11-26 review fixes (B-22 trim consistency, B-23 labelSuffixes stage-2 readAs).
import assert from "node:assert/strict";
import { test } from "node:test";

import { familyOfModelFolder, matchesQuery, parseLoraPath } from "../web/ai_studio/lora_names.js";
import { labelSuffixes } from "../web/ai_studio/lora_browser_state.js";

test("B-22: isModelFolder and isAlgoFolder trim whitespace and BOM consistently with familyOfModelFolder", () => {
  const cases = [
    ["\ufeffMOSS\\OFT\\g\\r\\r-000001.safetensors", "\ufeffMOSS", "OFT", "g", "r", "moss"],
    [" SDXL \\styles\\x.safetensors", " SDXL ", "styles", "", "x", "sdxl"],
    ["\u3000KREA2\u3000\\OFT\\g\\r.safetensors", "\u3000KREA2\u3000", "OFT", "g", "r", "krea2"],
    ["FLUX\\\ufeffLoKr\\g\\r\\r-000002.safetensors", "FLUX", "\ufeffLoKr", "g", "r", null],
    [" LoRA \\LoKr\\x.safetensors", "", " LoRA ", "LoKr", "x", null],
  ];
  for (const [entry, model, algo, group, run, family] of cases) {
    const p = parseLoraPath(entry);
    assert.deepEqual([p.model, p.algo, p.group, p.run], [model, algo, group, run], entry);
    assert.equal(familyOfModelFolder(p.model), family, entry);
  }
  // searchPath leaves out the trimmed model folder too
  assert.equal(matchesQuery({ entry: " SDXL \\styles\\x.safetensors" }, "sdxl"), false);
  assert.equal(matchesQuery({ entry: " SDXL \\styles\\x.safetensors" }, "styles x"), true);
});

test("B-23: labelSuffixes stage 2 compares labels with readAs so case/_ collisions still get disambiguated", () => {
  // Different groupLabel: stage 1 has no unique token and same model; stage 2 must still see "snapref_v2a" == "snapref v2a"
  assert.deepEqual(labelSuffixes([
    { label: "snapref_v2a", run: "snapref_v2a", model: "ANIMA", groupLabel: "g1", runKey: "ANIMA/LoKr/g1/snapref_v2a" },
    { label: "snapref v2a", run: "snapref v2a", model: "ANIMA", groupLabel: "g2", runKey: "ANIMA/LoKr/g2/snapref v2a" },
  ]), ["g1", "g2"]);

  // Case difference only in label + same group + same run spelling -> keyDifference disambiguates by algo folder
  assert.deepEqual(labelSuffixes([
    { label: "Snapref V2a", run: "snapref_v2a", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/snapref_v2a" },
    { label: "snapref_v2a", run: "snapref_v2a", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoRA/g/snapref_v2a" },
  ]), ["LoKr", "LoRA"]);

  // Same group, only "_" vs " " in run name -> run name distinguishes without duplicating via keyDifference
  assert.deepEqual(labelSuffixes([
    { label: "g · x y", run: "x y", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x y" },
    { label: "g · x y", run: "x_y", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x_y" },
  ]), ["x y", "x_y"]);
});
