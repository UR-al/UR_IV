// node:test for the control panel's prompt card (web/ai_studio/panel_view.js): the on/off switch left of every box's name,
// the positive / negative colours, and the label hook the preset list attaches to. Drawn into a tiny fake document.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { installFakeDom, mount } from "./fake_dom.mjs";
import { PanelView } from "../web/ai_studio/panel_view.js";
import { renderReference } from "../web/ai_studio/reference_view.js";
import { BOX_SWITCHES, applyStateToPrompt, assembledPrompt, withDefaults } from "../web/ai_studio/studio_logic.js";

installFakeDom();

const KEYS = ["prompt_prefix", "prompt_char_count", "prompt_character", "prompt_copyright", "prompt_artist", "prompt", "prompt_suffix", "negative"];
const LABELS = ["선행 고정", "인물 수", "캐릭터", "작품", "작가", "프롬프트", "후행 고정", "네거티브"];
const TEXTS = { prompt_prefix: "pre", prompt_char_count: "1boy", prompt_character: "hero", prompt_copyright: "game", prompt_artist: "@a", prompt: "main", prompt_suffix: "post", negative: "bad hands" };

function panelView(state = {}, extra = {}) {
  const ctx = {
    choices: async () => ({ items: [] }),
    engineRoom: { collapsed: () => true, toggle() {} },
    i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
    hasRole: () => true,
    fetchChoices: async () => ({}),
    ...extra,
  };
  return new PanelView({ value: JSON.stringify(state) }, ctx);
}

const card = (view) => view.element.byClass("ais-prompts")[0];
const rows = (view) => card(view).byClass("ais-row");
const rowOf = (view, key) => rows(view).find((row) => row.attrs["data-box"] === key);
const switches = (view) => card(view).byClass("ais-box-switch");
const switchOf = (view, key) => rowOf(view, key).byClass("ais-box-switch")[0];
const labelOf = (view, key) => rowOf(view, key).byClass("ais-box-label")[0];
const boxOf = (view, key) => rowOf(view, key).byTag("textarea")[0];
const isOn = (button) => button.attrs["aria-checked"] === "true" && button.classList.contains("on") && !button.classList.contains("off");
const dimmed = (view, key) => rowOf(view, key).classList.contains("ais-off");
const stepPill = (view, text) => view.element.byClass("ais-step-switches")[0].byClass("ais-pill").find((button) => button.text === text);

test("the card has one switch a box, left of its name, in the order the boxes join and the negative last", () => {
  const view = panelView(TEXTS);
  assert.deepEqual(rows(view).map((row) => row.attrs["data-box"]), KEYS);
  assert.deepEqual(switches(view).map((button) => button.attrs["data-switch"]), KEYS.map((key) => BOX_SWITCHES[key]));
  assert.equal(switches(view).length, 8);
  for (const [index, key] of KEYS.entries()) {
    const label = rowOf(view, key).byClass("ais-label")[0];
    assert.deepEqual(label.children, [switchOf(view, key), labelOf(view, key)], key); // the switch, then the name
    assert.equal(label.text, LABELS[index]); // the switch has no text of its own
    const button = switchOf(view, key);
    assert.equal(button.tag, "button");
    assert.equal(button.attrs.role, "switch");
    assert.equal(isOn(button), true, key);
    assert.match(button.attrs.title, /넣기\/빼기/, key);
    assert.match(button.attrs["aria-label"], new RegExp(LABELS[index]), key);
    assert.equal(boxOf(view, key).value, TEXTS[key]);
    assert.equal(dimmed(view, key), false);
  }
});

test("the seven positive rows and the negative row are marked for their colours", () => {
  const view = panelView();
  assert.deepEqual(rows(view).map((row) => (row.classList.contains("ais-pos") ? "pos" : row.classList.contains("ais-neg") ? "neg" : "")),
    ["pos", "pos", "pos", "pos", "pos", "pos", "pos", "neg"]);
});

test("a switch turns its box off and on: saved, the row dimmed, the text kept", () => {
  const view = panelView(TEXTS);
  for (const key of KEYS) {
    const flag = BOX_SWITCHES[key];
    switchOf(view, key).fire("click");
    assert.equal(view.state[flag], false, key);
    assert.equal(JSON.parse(view.widget.value)[flag], false, key); // saved with the workflow
    assert.equal(view.getState()[flag], false, key); // and what a run reads
    assert.equal(isOn(switchOf(view, key)), false, key);
    assert.equal(switchOf(view, key).attrs["aria-checked"], "false", key);
    assert.equal(dimmed(view, key), true, key);
    assert.match(rowOf(view, key).attrs.title, /^꺼짐/, key);
    assert.equal(boxOf(view, key).value, TEXTS[key], key);
    assert.equal(view.state[key], TEXTS[key], key);
    assert.deepEqual(KEYS.filter((other) => dimmed(view, other)), [key]); // its own row only
    switchOf(view, key).fire("click");
    assert.equal(view.state[flag], true, key);
    assert.equal(isOn(switchOf(view, key)), true, key);
    assert.equal(dimmed(view, key), false, key);
    assert.equal(rowOf(view, key).attrs.title, undefined, key);
  }
});

test("a box that is off can still be typed in", () => {
  const view = panelView({ ...TEXTS, prompt_character_on: false });
  const box = boxOf(view, "prompt_character");
  assert.equal(box.disabled, false);
  box.value = "another";
  box.fire("input");
  assert.deepEqual([view.state.prompt_character, view.state.prompt_character_on], ["another", false]);
});

test("a switch keeps the keyboard focus across the redraw of its own toggle", () => {
  const view = panelView(TEXTS);
  try {
    for (const key of KEYS) {
      for (const on of [false, true]) { // Space twice: off, then on again
        const button = switchOf(view, key);
        button.focus();
        button.fire("click");
        assert.notEqual(switchOf(view, key), button, key); // drawn again
        assert.equal(globalThis.document.activeElement, switchOf(view, key), key);
        assert.equal(isOn(switchOf(view, key)), on, key);
      }
    }
  } finally {
    globalThis.document.activeElement = null;
  }
});

test("the 스텝 row's 선행 / 후행 and the card's switches are the same two flags", () => {
  const view = panelView(TEXTS);
  try {
    for (const [text, key] of [["선행", "prompt_prefix"], ["후행", "prompt_suffix"]]) {
      const pillOn = () => stepPill(view, text).attrs["aria-pressed"] === "true";
      switchOf(view, key).fire("click");
      assert.deepEqual([pillOn(), isOn(switchOf(view, key)), dimmed(view, key)], [false, false, true], text);
      stepPill(view, text).fire("click");
      assert.deepEqual([pillOn(), isOn(switchOf(view, key)), dimmed(view, key), view.state[`${key}_on`]], [true, true, false, true], text);
      const pill = stepPill(view, text);
      pill.focus();
      pill.fire("click");
      assert.deepEqual([pillOn(), isOn(switchOf(view, key)), dimmed(view, key)], [false, false, true], text);
      assert.equal(globalThis.document.activeElement, stepPill(view, text), text); // the one that was pressed keeps the focus
      switchOf(view, key).fire("click");
    }
  } finally {
    globalThis.document.activeElement = null;
  }
});

test("the negative off builds a run with an empty negative; on again, the text is back", () => {
  const view = panelView(TEXTS);
  const api = { 1: { class_type: "AIStudioLoraStack", inputs: {} } };
  const built = () => applyStateToPrompt(api, { lora_stack: "1" }, { ...view.getState(), seed: 5 })[1].inputs;
  assert.deepEqual(built(), { positive: "pre, 1boy, hero, game, @a, main, post", negative: "bad hands" });
  switchOf(view, "negative").fire("click");
  assert.deepEqual(built(), { positive: "pre, 1boy, hero, game, @a, main, post", negative: "" });
  switchOf(view, "prompt_character").fire("click");
  switchOf(view, "prompt_copyright").fire("click");
  assert.deepEqual(built(), { positive: "pre, 1boy, @a, main, post", negative: "" });
  switchOf(view, "negative").fire("click");
  assert.deepEqual(built(), { positive: "pre, 1boy, @a, main, post", negative: "bad hands" });
});

test("a panel saved before the switches loads with every box on", () => {
  const view = panelView(TEXTS); // no *_on key at all
  assert.deepEqual(KEYS.filter((key) => !isOn(switchOf(view, key)) || dimmed(view, key)), []);
  for (const flag of Object.values(BOX_SWITCHES)) assert.equal(view.getState()[flag], true, flag);
  view.widget.value = JSON.stringify({ ...TEXTS, prompt_artist_on: "no", negative_on: 0 }); // only a real boolean switches off
  view.load();
  assert.deepEqual(KEYS.filter((key) => dimmed(view, key)), []);
});

// ---------------------------------------------------------------------------------------------- the name: a hook for the preset list

test("the name is its own element with data-section, apart from the switch", () => {
  const view = panelView(TEXTS);
  assert.deepEqual(KEYS.map((key) => labelOf(view, key).attrs["data-section"]), KEYS);
  assert.deepEqual(KEYS.map((key) => labelOf(view, key).text), LABELS);
  for (const key of KEYS) {
    assert.equal(labelOf(view, key).tag, "span", key); // nothing to open yet: not a button that does nothing
    labelOf(view, key).fire("click");
    assert.equal(view.state[BOX_SWITCHES[key]], true, key); // the name is not the switch
  }
});

test("with ctx.boxLabel the name is a button that hands over its box and itself, and keeps the focus over a redraw", () => {
  const calls = [];
  const view = panelView(TEXTS, { boxLabel: (key, anchor) => calls.push([key, anchor]) });
  try {
    for (const key of KEYS) {
      const name = labelOf(view, key);
      assert.equal(name.tag, "button", key);
      assert.equal(name.attrs.type, "button", key);
      assert.equal(name.attrs["data-section"], key);
      name.focus();
      name.fire("click");
      assert.deepEqual(calls.at(-1), [key, name]);
      assert.equal(view.state[BOX_SWITCHES[key]], true, key);
      view.set((state) => { state[key] = "from a preset"; }, true); // what a picked preset does
      assert.equal(labelOf(view, key), name, key); // the element that was handed over is still the box's name
      assert.equal(globalThis.document.activeElement, name, key);
      assert.equal(boxOf(view, key).value, "from a preset");
    }
    assert.equal(calls.length, 8);
  } finally {
    globalThis.document.activeElement = null;
  }
});

// The preset list keeps the name it opened under (its anchor) for as long as it is open: it reads the name's place, treats
// a press on it as its own, marks it aria-expanded and gives the focus back to it. So a redraw must not swap the element.
test("a name is one element for the panel's life: every redraw keeps it in the page, with what was set on it", () => {
  for (const extra of [{}, { boxLabel() {} }]) {
    const view = panelView(TEXTS, extra);
    mount(view.element);
    const names = KEYS.map((key) => labelOf(view, key));
    for (const name of names) name.setAttribute("aria-expanded", "true"); // what the open list sets on its anchor
    const redraws = [
      () => view.set((state) => { state.negative = "from a preset"; }, true),
      () => view.render(),
      () => switchOf(view, "prompt_artist").fire("click"),
      () => { view.widget.value = JSON.stringify({ ...TEXTS, prompt: "loaded" }); view.load(); },
    ];
    for (const [index, redraw] of redraws.entries()) {
      redraw();
      assert.deepEqual(KEYS.map((key) => labelOf(view, key)), names, `redraw ${index}`);
      assert.equal(card(view).byClass("ais-box-label").length, 8, `redraw ${index}`);
      assert.ok(names.every((name) => name.isConnected && name.attrs["aria-expanded"] === "true"), `redraw ${index}`);
      // the off look is the row's, not the name's: nothing on the kept element goes stale
      assert.equal(dimmed(view, "prompt_artist"), index === 2, `redraw ${index}`);
    }
    assert.equal(boxOf(view, "prompt").value, "loaded");
  }
});

// ---------------------------------------------------------------------------------------------- 트리거 넣기 (레퍼런스 카드, Krea2 QuadView)

/** Presses the reference card's "트리거 넣기" `times` times on a Krea2 QuadView state; returns the state. */
function pressQuadviewTrigger(changes, times = 1) {
  const state = withDefaults({ unet: "krea2_raw.safetensors", reference: { on: true, image: "a.png", krea2: { file: "krea2/QuadView_krea2_v1.safetensors" } }, ...changes });
  const view = { set: (mutate) => mutate(state), hasRole: () => true, family: "krea2", memory: { open: new Set(["reference"]), lists: {} }, rerender() {}, loras: [], image: { url: () => "" }, image2: { url: () => "" } };
  for (let press = 0; press < times; press += 1) {
    renderReference(state, view).all((element) => element.tag === "button" && element.text === "트리거 넣기")[0].fire("click");
  }
  return state;
}

test("트리거 넣기 with the 프롬프트 box off: the caption goes in once and the box is switched on, so the run has it", () => {
  const caption = pressQuadviewTrigger({ prompt: "" }).prompt;
  assert.ok(caption.length > 20);
  for (const times of [1, 2, 3]) {
    const state = pressQuadviewTrigger({ prompt: "1girl", prompt_on: false, prompt_character: "hero" }, times);
    assert.equal(state.prompt, `${caption} 1girl`, `${times} presses`); // never twice
    assert.equal(state.prompt_on, true);
    assert.equal(assembledPrompt(state), `hero, ${caption} 1girl`);
  }
  const held = pressQuadviewTrigger({ prompt: `${caption}, 1girl`, prompt_on: false }, 2); // already in the box that is off
  assert.deepEqual([held.prompt, held.prompt_on], [`${caption}, 1girl`, true]);
});

test("트리거 넣기 counts only the boxes the run uses, and leaves the other switches alone", () => {
  const caption = pressQuadviewTrigger({ prompt: "" }).prompt;
  const off = pressQuadviewTrigger({ prompt: "1girl", prompt_prefix: caption, prompt_prefix_on: false }, 2); // held by a box that is off
  assert.deepEqual([off.prompt, off.prompt_prefix, off.prompt_prefix_on, off.prompt_on], [`${caption} 1girl`, caption, false, true]);
  const on = pressQuadviewTrigger({ prompt: "1girl", prompt_prefix: caption, negative_on: false, prompt_artist_on: false }); // the run has it
  assert.deepEqual([on.prompt, on.prompt_on, on.negative_on, on.prompt_artist_on], ["1girl", true, false, false]);
});

// ---------------------------------------------------------------------------------------------- colours (studio.css)

const CSS = readFileSync(new URL("../web/ai_studio/studio.css", import.meta.url), "utf-8");
const token = (name) => new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})\\b`).exec(CSS)?.[1];
const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + 0.05) / (Math.min(luminance(a), luminance(b)) + 0.05);
const hue = (hex) => {
  const [r, g, b] = [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16) / 255);
  const max = Math.max(r, g, b);
  const span = max - Math.min(r, g, b);
  const turn = max === r ? ((g - b) / span) % 6 : max === g ? (b - r) / span + 2 : (r - g) / span + 4;
  return (turn * 60 + 360) % 360;
};
const CARD = "#2b2b2b"; // .ais-node's background, where the names sit

test("the positive and the negative names are two clearly different hues, both readable on the card", () => {
  const [positive, negative, off] = ["--ais-pos", "--ais-neg", "--ais-box-off"].map(token);
  assert.ok(positive && negative && off, "studio.css defines --ais-pos, --ais-neg and --ais-box-off");
  assert.equal(/\.ais-node\s*\{[^}]*background:\s*#2b2b2b/.test(CSS), true); // the colour the contrast is measured against
  for (const [name, colour] of [["positive", positive], ["negative", negative]]) {
    assert.ok(contrast(colour, CARD) >= 4.5, `${name} ${colour}: ${contrast(colour, CARD).toFixed(2)}:1 on the card`);
  }
  assert.ok(contrast(off, CARD) >= 3, `off ${off}: ${contrast(off, CARD).toFixed(2)}:1`);
  const apart = Math.abs(hue(positive) - hue(negative));
  assert.ok(Math.min(apart, 360 - apart) >= 90, `hues ${hue(positive).toFixed(0)}° and ${hue(negative).toFixed(0)}°`);
  assert.ok(hue(positive) > 90 && hue(positive) < 180, "positive: a green"); // green-ish
  assert.ok(hue(negative) < 40 || hue(negative) > 340, "negative: a red or orange"); // red / orange-ish
});

test("the colours and the off look are wired to the card's rows, in the ais-* scope", () => {
  assert.match(CSS, /\.ais-prompts \.ais-row\.ais-pos \.ais-box-label\s*\{[^}]*color:\s*var\(--ais-pos\)/);
  assert.match(CSS, /\.ais-prompts \.ais-row\.ais-neg \.ais-box-label\s*\{[^}]*color:\s*var\(--ais-neg\)/);
  assert.match(CSS, /\.ais-prompts \.ais-row\.ais-off \.ais-box-label\s*\{[^}]*color:\s*var\(--ais-box-off\)[^}]*line-through/);
  assert.match(CSS, /\.ais-box-switch:focus-visible\s*\{[^}]*outline/);
  const selectors = CSS.replace(/\/\*[\s\S]*?\*\//g, "").split("}").map((rule) => rule.split("{")[0].trim()).filter((selector) => /box-(switch|label)/.test(selector));
  assert.ok(selectors.length >= 6);
  for (const selector of selectors) assert.match(selector, /^\.ais-/, selector);
});
