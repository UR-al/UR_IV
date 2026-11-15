// node:test for the small line under a prompt box's name on the control panel (web/ai_studio/panel_view.js): the name of the
// preset the box holds, "저장 안 됨" for a text no preset has, nothing for an empty box — and the one table of box names the
// panel and the presets share (studio_logic.js BOX_LABELS). Drawn into the tiny fake document.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { installFakeDom, mount } from "./fake_dom.mjs";
import { PanelView } from "../web/ai_studio/panel_view.js";
import { PRESET_LABELS, PRESET_SECTIONS, labelMark } from "../web/ai_studio/preset_logic.js";
import { BOX_LABELS, BOX_SWITCHES } from "../web/ai_studio/studio_logic.js";

installFakeDom();

const KEYS = Object.keys(BOX_SWITCHES);
const LABELS = ["선행 고정", "인물 수", "캐릭터", "작품", "작가", "프롬프트", "후행 고정", "네거티브"];
const web = (name) => readFileSync(new URL(`../web/ai_studio/${name}`, import.meta.url), "utf-8");
const BASE_CTX = {
  choices: async () => ({ items: [] }),
  engineRoom: { collapsed: () => true, toggle() {} },
  i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
  hasRole: () => true,
  fetchChoices: async () => ({}),
};

/** A panel whose presets are `presets` ({box: [{name, text}]}); `ready: false` = the presets were not read yet. */
function panelView(state = {}, presets = {}, { ready = true } = {}) {
  const env = { presets, ready };
  const ctx = {
    ...BASE_CTX,
    boxLabel() {},
    boxPreset: (key, text) => (env.ready ? (env.presets[key] || []).find((preset) => preset.text === text.trim()) || null : undefined),
  };
  env.view = new PanelView({ value: JSON.stringify(state) }, ctx);
  mount(env.view.element);
  return env;
}

const rowOf = (view, key) => view.element.byClass("ais-prompts")[0].byClass("ais-row").find((row) => row.attrs["data-box"] === key);
const markOf = (view, key) => rowOf(view, key).byClass("ais-box-mark")[0];
const nameOf = (view, key) => rowOf(view, key).byClass("ais-box-label")[0];
const boxOf = (view, key) => rowOf(view, key).byTag("textarea")[0];
const shown = (mark) => (mark.hidden ? null : [mark.text, mark.attrs["data-mark"]]);

test("the box names are one table: the panel and the presets read studio_logic's BOX_LABELS", () => {
  assert.deepEqual(Object.keys(BOX_LABELS), KEYS);
  assert.deepEqual(Object.values(BOX_LABELS), LABELS);
  assert.equal(Object.isFrozen(BOX_LABELS), true);
  assert.equal(PRESET_LABELS, BOX_LABELS);
  assert.deepEqual(Object.keys(BOX_LABELS), [...PRESET_SECTIONS]);
  const { view } = panelView();
  assert.deepEqual(KEYS.map((key) => nameOf(view, key).text), LABELS);
  assert.deepEqual(KEYS.map((key) => boxOf(view, key).attrs["aria-label"]), LABELS);
  // written out once: neither the panel nor the presets keep a copy that could drift
  for (const file of ["panel_view.js", "preset_logic.js"]) assert.doesNotMatch(web(file), /:\s*"(후행 고정|인물 수|네거티브)"/, file);
  assert.match(web("studio_logic.js"), /prompt_suffix:\s*"후행 고정"/);
});

test("labelMark: the preset's name, 저장 안 됨 for a text no preset has, nothing for an empty box", () => {
  assert.deepEqual(labelMark({ name: "Anima 기본" }, "worst quality"), { kind: "match", text: "Anima 기본", title: "프리셋 ‘Anima 기본’과 같음" });
  assert.deepEqual(labelMark(null, "my text"), { kind: "unsaved", text: "저장 안 됨", title: "같은 내용의 프리셋이 없음 (이름을 눌러 저장)" });
  for (const empty of ["", " , \n", null, undefined]) assert.deepEqual(labelMark(null, empty), { kind: "empty", text: "", title: "" }, JSON.stringify(empty));
  assert.deepEqual(labelMark(undefined, "my text"), { kind: "empty", text: "", title: "" }); // the presets were not read: nothing to say
});

test("without ctx.boxPreset the card has no mark at all", () => {
  const view = new PanelView({ value: "{}" }, BASE_CTX);
  assert.equal(view.element.byClass("ais-box-mark").length, 0);
  assert.doesNotThrow(() => view.refreshBoxMarks());
});

test("every box and the negative show what they hold under the name: preset, 저장 안 됨, or nothing", () => {
  const presets = { prompt_artist: [{ name: "Ena", text: "@enaa97" }], negative: [{ name: "Anima 기본", text: "worst quality" }] };
  const { view } = panelView({ prompt_artist: "@enaa97", prompt: "1girl", negative: "worst quality" }, presets);
  assert.equal(view.element.byClass("ais-box-mark").length, 8);
  for (const key of KEYS) {
    const label = rowOf(view, key).byClass("ais-label")[0];
    assert.deepEqual(label.children, [rowOf(view, key).byClass("ais-box-switch")[0], nameOf(view, key), markOf(view, key)], key); // under the name
    assert.equal(markOf(view, key).tag, "span", key);
    assert.equal(markOf(view, key).attrs["data-for"], key);
    assert.equal(nameOf(view, key).text, BOX_LABELS[key], key); // the name itself never changes
  }
  assert.deepEqual(shown(markOf(view, "prompt_artist")), ["Ena", "match"]);
  assert.match(markOf(view, "prompt_artist").title, /Ena/);
  assert.deepEqual(shown(markOf(view, "negative")), ["Anima 기본", "match"]);
  assert.deepEqual(shown(markOf(view, "prompt")), ["저장 안 됨", "unsaved"]);
  for (const key of ["prompt_prefix", "prompt_char_count", "prompt_character", "prompt_copyright", "prompt_suffix"]) {
    assert.equal(shown(markOf(view, key)), null, key);
    assert.equal(markOf(view, key).text, "", key);
  }
});

test("before the presets are read no box says 저장 안 됨; refreshBoxMarks fills the marks in place once they are", () => {
  const env = panelView({ prompt: "1girl", negative: "worst quality" }, { negative: [{ name: "Anima 기본", text: "worst quality" }] }, { ready: false });
  assert.deepEqual(KEYS.map((key) => shown(markOf(env.view, key))), KEYS.map(() => null));
  const marks = KEYS.map((key) => markOf(env.view, key));
  const scroll = env.view.element.byClass("ais-scroll")[0];
  env.ready = true;
  env.view.refreshBoxMarks();
  assert.deepEqual(shown(markOf(env.view, "negative")), ["Anima 기본", "match"]);
  assert.deepEqual(shown(markOf(env.view, "prompt")), ["저장 안 됨", "unsaved"]);
  assert.deepEqual(KEYS.map((key) => markOf(env.view, key)), marks);
  assert.equal(env.view.element.byClass("ais-scroll")[0], scroll); // the panel was not drawn again
});

test("typing in a box changes its mark at once, without drawing the panel again", () => {
  const { view } = panelView({ negative: "worst quality" }, { negative: [{ name: "Anima 기본", text: "worst quality" }] });
  const box = boxOf(view, "negative");
  box.focus();
  box.value = "worst quality, lowres";
  box.fire("input");
  assert.equal(view.state.negative, "worst quality, lowres");
  assert.deepEqual(shown(markOf(view, "negative")), ["저장 안 됨", "unsaved"]);
  assert.equal(boxOf(view, "negative"), box); // the same box: the caret stays
  assert.equal(globalThis.document.activeElement, box);
  box.value = "";
  box.fire("input");
  assert.equal(shown(markOf(view, "negative")), null);
  box.value = "worst quality";
  box.fire("input");
  assert.deepEqual(shown(markOf(view, "negative")), ["Anima 기본", "match"]);
  globalThis.document.activeElement = null;
});

test("a redraw and a box switched off keep the mark; the off look stays the row's", () => {
  const { view } = panelView({ prompt_artist: "@enaa97", prompt_artist_on: false }, { prompt_artist: [{ name: "Ena", text: "@enaa97" }] });
  assert.equal(rowOf(view, "prompt_artist").classList.contains("ais-off"), true);
  assert.deepEqual(shown(markOf(view, "prompt_artist")), ["Ena", "match"]);
  view.set((state) => { state.prompt_artist = "@other"; }, true); // what a picked preset does
  assert.deepEqual(shown(markOf(view, "prompt_artist")), ["저장 안 됨", "unsaved"]);
  assert.equal(nameOf(view, "prompt_artist").className, "ais-box-label"); // nothing of the state is set on the name
  view.widget.value = JSON.stringify({ prompt_artist: "@enaa97" });
  view.load();
  assert.deepEqual(shown(markOf(view, "prompt_artist")), ["Ena", "match"]);
  assert.equal(view.element.byClass("ais-box-mark").length, 8);
});

test("the mark's look: small, under the name, cut with … when long, greyed with a box that is off", () => {
  const css = web("studio.css");
  assert.match(css, /\.ais-prompts \.ais-box-mark\s*\{[^}]*grid-column:\s*2[^}]*\}/);
  assert.match(css, /\.ais-prompts \.ais-box-mark\s*\{[^}]*text-overflow:\s*ellipsis/);
  assert.match(css, /\.ais-prompts \.ais-box-mark\[hidden\]\s*\{[^}]*display:\s*none/);
  assert.match(css, /\.ais-prompts \.ais-row\.ais-off \.ais-box-mark\s*\{[^}]*color:\s*var\(--ais-box-off\)/);
  const selectors = css.replace(/\/\*[\s\S]*?\*\//g, "").split("}").map((rule) => rule.split("{")[0].trim()).filter((selector) => /box-mark/.test(selector));
  assert.ok(selectors.length >= 4);
  for (const selector of selectors) assert.match(selector, /^\.ais-/, selector);
});
