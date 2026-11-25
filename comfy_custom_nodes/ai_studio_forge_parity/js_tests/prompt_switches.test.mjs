// node:test for the prompt card's on/off switches in the pure logic (web/ai_studio/studio_logic.js, studio_settings.js):
// every box and the negative has one; off, the text stays and the run leaves it out. Python twin: studio_state.py
// (tests/test_studio_prompt_switches.py); the golden table is shared with tests/test_studio_logic_golden.py.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  BOX_SWITCHES, DEFAULT_STATE, NEGATIVE_SWITCH, PROMPT_SECTIONS, PROMPT_SWITCHES, addPromptTrigger, applyStateToPrompt, assembledPrompt,
  effectiveNegative,
  effectivePrompt, hasNegativeWeights, hasPromptWeights, loraTagNames, mayHaveWildcards, missingLoras, postPrompt, postRunState, withDefaults,
  xyzCellPrompts,
} from "../web/ai_studio/studio_logic.js";
import { pulledState } from "../web/ai_studio/studio_settings.js";

const golden = JSON.parse(readFileSync(new URL("./fixtures/studio_logic_golden.json", import.meta.url), "utf-8"));

const SWITCHES = ["prompt_prefix_on", "prompt_char_count_on", "prompt_character_on", "prompt_copyright_on", "prompt_artist_on", "prompt_on", "prompt_suffix_on", "negative_on"];
const BOXES = { prompt_prefix: "pre", prompt_char_count: "1boy", prompt_character: "hero", prompt_copyright: "game", prompt_artist: "@a", prompt: "main", prompt_suffix: "post" };
const WHOLE = "pre, 1boy, hero, game, @a, main, post";
const ALL_OFF = Object.fromEntries(SWITCHES.map((key) => [key, false]));
const ALL_ON = Object.fromEntries(SWITCHES.map((key) => [key, true]));
const API = { 1: { class_type: "AIStudioLoraStack", inputs: {} } };
const ROLES = { lora_stack: "1" };
const stack = (state) => applyStateToPrompt(API, ROLES, { seed: 5, ...state })[1].inputs;

test("every box and the negative have a switch, named as in studio_state.py", () => {
  assert.deepEqual(PROMPT_SWITCHES, Object.fromEntries(PROMPT_SECTIONS.map((key) => [key, `${key}_on`])));
  assert.equal(NEGATIVE_SWITCH, "negative_on");
  assert.deepEqual(Object.keys(BOX_SWITCHES), [...PROMPT_SECTIONS, "negative"]); // the card's rows, top to bottom
  assert.deepEqual(Object.values(BOX_SWITCHES), SWITCHES);
});

test("each switch defaults on, and only a boolean turns it off", () => {
  for (const key of SWITCHES) {
    assert.equal(DEFAULT_STATE[key], true, key);
    assert.equal(withDefaults({})[key], true, key); // a panel saved before the switches
    assert.equal(withDefaults({ [key]: false })[key], false, key);
    for (const other of ["no", 0, null, "false"]) assert.equal(withDefaults({ [key]: other })[key], true, key);
  }
});

test("effectiveNegative matches Python's effective_negative", () => {
  assert.ok(golden.effective_negative.length >= 6);
  for (const { state, expected } of golden.effective_negative) {
    assert.equal(effectiveNegative(withDefaults(state)), expected, JSON.stringify(state));
  }
});

test("each switch takes its own box out of the prompt and leaves the text in the state", () => {
  for (const [key, text] of Object.entries(BOXES)) {
    const wanted = WHOLE.split(", ").filter((part) => part !== text).join(", ");
    const state = { ...BOXES, [PROMPT_SWITCHES[key]]: false };
    assert.equal(assembledPrompt(state), wanted, key);
    assert.equal(withDefaults(state)[key], text, key);
  }
  assert.equal(assembledPrompt({ ...BOXES, negative_on: false }), WHOLE); // the negative's switch is the negative's alone
});

test("the main box off leaves the others; alone, nothing is sent", () => {
  assert.equal(assembledPrompt({ ...BOXES, prompt_on: false }), "pre, 1boy, hero, game, @a, post");
  assert.equal(assembledPrompt({ prompt: "1boy, \n", prompt_on: false }), "");
  assert.equal(assembledPrompt({ ...BOXES, ...ALL_OFF }), "");
  assert.equal(effectivePrompt({ prompt: "1boy <lora:in_box:1>", prompt_on: false, loras: [{ on: true, name: "l", strength: 0.5 }] }), "<lora:l:0.5>");
});

test("a state without the switches builds the run of a state with all of them on, byte for byte", () => {
  for (const state of [{ ...BOXES, negative: "bad hands, \n" }, { prompt: "1boy, \n", negative: "lowres" }, { prompt: "1boy,", prompt_artist: "@a," }, {}]) {
    assert.deepEqual(applyStateToPrompt(API, ROLES, { seed: 5, ...state }), applyStateToPrompt(API, ROLES, { seed: 5, ...state, ...ALL_ON }), JSON.stringify(state));
  }
  assert.deepEqual(stack({ ...BOXES, negative: "bad hands, \n" }), { positive: WHOLE, negative: "bad hands, \n" });
  assert.deepEqual(stack({ prompt: "1boy, \n", negative: "lowres" }), { positive: "1boy, \n", negative: "lowres" });
});

test("the negative off builds a run with an empty negative; its text stays in the saved state", () => {
  const state = { ...BOXES, negative: "bad hands", negative_on: false };
  assert.deepEqual(stack(state), { positive: WHOLE, negative: "" });
  assert.deepEqual(stack({ ...state, negative_on: true }), { positive: WHOLE, negative: "bad hands" });
  const api = { ...API, 2: { class_type: "AIStudioSaveImage", inputs: {} } };
  const saved = JSON.parse(applyStateToPrompt(api, { ...ROLES, save: "2" }, { seed: 5, ...state })[2].inputs.studio_state);
  assert.deepEqual([saved.negative, saved.negative_on], ["bad hands", false]); // the image's infotext is built from this
});

test("each switched-off box is left out of the built run", () => {
  assert.equal(stack({ ...BOXES, prompt_character_on: false, prompt_copyright_on: false }).positive, "pre, 1boy, @a, main, post");
  assert.equal(stack({ ...BOXES, prompt_on: false }).positive, "pre, 1boy, hero, game, @a, post");
  assert.deepEqual(stack({ ...BOXES, negative: "bad", ...ALL_OFF }), { positive: "", negative: "" });
});

test("what a run checks leaves a switched-off box and negative out: LoRA tags, weights, wildcards", () => {
  const off = { prompt: "1boy", prompt_character: "(bad:-1.0), __wild__, <lora:c:1>", negative: "(worse:-1.2), {a|b}, <lora:n:1>", prompt_character_on: false, negative_on: false };
  assert.deepEqual(loraTagNames(off), []);
  assert.deepEqual(missingLoras(off, ["x.safetensors"]), []);
  assert.equal(hasNegativeWeights(off), false);
  assert.equal(hasPromptWeights(off), false);
  assert.equal(mayHaveWildcards(off), false);
  const on = { ...off, prompt_character_on: true, negative_on: true };
  assert.deepEqual(loraTagNames(on), ["c", "n"]);
  assert.deepEqual(missingLoras(on, ["x.safetensors"]), ["c", "n"]);
  assert.equal(hasNegativeWeights({ ...off, negative_on: true }), true);
  assert.equal(hasPromptWeights({ ...off, negative_on: true }), true);
  assert.equal(mayHaveWildcards({ ...off, negative_on: true }), true);
  assert.equal(mayHaveWildcards({ prompt: "__wild__", prompt_on: false }), false);
});

test("missingLoras matches Python's missing_loras with switches off", () => {
  const cases = golden.missing_loras.filter(({ state }) => SWITCHES.some((key) => key in state));
  assert.ok(cases.length >= 2);
  for (const { state, available, missing } of cases) assert.deepEqual(missingLoras(state, available), missing, JSON.stringify(state));
});

test("an XYZ cell runs what the plan rendered", () => {
  const cell = { ...structuredClone(DEFAULT_STATE), seed: 1, prompt: "pre, 1boy, post", negative: "" }; // xyz_plan: a negative that is off is "" in the cell
  const [{ prompt }] = xyzCellPrompts(API, ROLES, { cells: [{ state: cell }], order: [0] });
  assert.deepEqual(prompt[1].inputs, { positive: "pre, 1boy, post", negative: "" });
});

test("a gallery run reads a record without the switches as made with everything in", () => {
  const panel = { prompt: "now", negative: "panel negative", ...ALL_OFF };
  const old = postRunState({ ...BOXES, negative: "bad hands" }, panel, "upscale"); // a record from before the switches
  assert.equal(effectivePrompt(old), WHOLE);
  assert.equal(effectiveNegative(old), "bad hands");
  for (const key of SWITCHES) assert.equal(old[key], true, key);
  const fromText = postRunState({ prompt: "1girl", negative: "lowres", seed: 3, steps: 20 }, panel, "hires"); // recordFromParameters
  assert.deepEqual([effectivePrompt(fromText), effectiveNegative(fromText)], ["1girl", "lowres"]);
});

test("a gallery run keeps the record's own switches, whatever the panel says", () => {
  const record = { ...BOXES, negative: "bad hands", prompt_artist_on: false, negative_on: false };
  const merged = postRunState(record, { prompt: "now", negative: "panel negative" }, "upscale");
  assert.equal(effectivePrompt(merged), "pre, 1boy, hero, game, main, post");
  assert.equal(effectiveNegative(merged), "");
  const api = { ...API, 2: { class_type: "LoadImage", inputs: {} }, 3: { class_type: "AIStudioSaveImage", inputs: {} } };
  const { prompt } = postPrompt(api, { ...ROLES, post_image: "2", post_save: "3" }, { ...record, seed: 9, image: "a.png" }, "upscale");
  assert.deepEqual(prompt[1].inputs, { positive: "pre, 1boy, hero, game, main, post", negative: "" });
});

// ---------------------------------------------------------------------------------------------- 설정 가져오기

const current = (changes = {}) => withDefaults({ unet: "a.safetensors", ...changes });

test("설정 가져오기: a studio image brings its switches back; a panel saved without them comes back all on", () => {
  const saved = { ...BOXES, negative: "bad hands", prompt_character_on: false, prompt_on: false, negative_on: false };
  const { state } = pulledState(current(), { panel: saved });
  assert.deepEqual(SWITCHES.filter((key) => state[key] === false), ["prompt_character_on", "prompt_on", "negative_on"]);
  assert.deepEqual([state.prompt_character, state.prompt, state.negative], ["hero", "main", "bad hands"]); // the text too
  const old = pulledState(current(ALL_OFF), { panel: { ...BOXES, negative: "bad hands" } }).state; // made before the switches
  for (const key of SWITCHES) assert.equal(old[key], true, key);
  assert.deepEqual([effectivePrompt(old), effectiveNegative(old)], [WHOLE, "bad hands"]);
});

test("설정 가져오기: an infotext's prompt and negative come back switched on", () => {
  // what /infotext/parse answers (studio_infotext_read.parse_infotext): the boxes empty, the two texts on
  const settings = { prompt_prefix: "", prompt_char_count: "", prompt_character: "", prompt_copyright: "", prompt_artist: "", prompt_suffix: "", prompt: "1girl", prompt_on: true, negative: "lowres", negative_on: true, loras: [], steps: 20 };
  const { state } = pulledState(current({ ...BOXES, negative: "panel", ...ALL_OFF }), { settings, model: "", modules: [] });
  assert.deepEqual([effectivePrompt(state), effectiveNegative(state)], ["1girl", "lowres"]);
  assert.deepEqual([state.prompt_on, state.negative_on, state.prompt_artist_on], [true, true, false]); // an emptied box keeps the panel's switch
});

test("addPromptTrigger: in front of the main box unless the run has it; a main box that is off comes back on", () => {
  const put = (state) => { addPromptTrigger(state, "TRG"); return [state.prompt, state.prompt_on, assembledPrompt(state)]; };
  assert.deepEqual(put({ prompt: "" }), ["TRG", true, "TRG"]);
  assert.deepEqual(put({ prompt: "1girl", prompt_on: true }), ["TRG 1girl", true, "TRG 1girl"]);
  assert.deepEqual(put({ prompt: "1girl", prompt_on: false, prompt_character: "hero" }), ["TRG 1girl", true, "hero, TRG 1girl"]);
  assert.deepEqual(put({ prompt: "TRG 1girl", prompt_on: false, prompt_character: "hero" }), ["TRG 1girl", true, "hero, TRG 1girl"]); // not twice
  assert.deepEqual(put({ prompt: "1girl", prompt_on: true, prompt_suffix: "TRG" }), ["1girl", true, "1girl, TRG"]); // another box that is on has it
  assert.deepEqual(put({ prompt: "1girl", prompt_suffix: "TRG", prompt_suffix_on: false }), ["TRG 1girl", true, "TRG 1girl"]); // that box is off
  const untouched = { prompt: "a TRG b", prompt_on: true, negative_on: false };
  addPromptTrigger(untouched, "TRG");
  assert.deepEqual(untouched, { prompt: "a TRG b", prompt_on: true, negative_on: false });
  const pressedTwice = { prompt: "1girl", prompt_on: false };
  addPromptTrigger(pressedTwice, "TRG");
  addPromptTrigger(pressedTwice, "TRG");
  assert.deepEqual(pressedTwice, { prompt: "TRG 1girl", prompt_on: true });
});
