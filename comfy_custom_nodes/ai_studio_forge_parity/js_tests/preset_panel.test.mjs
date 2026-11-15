// node:test for the presets' wiring into the control panel (web/ai_studio/preset_panel.js, used by studio_main.js): a box's
// name opens the quick list, "프리셋 관리…" the window, and the line under every name follows the box and the store.
// Uses the preset tests' fake document and fake preset routes (preset_fakes.mjs). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { PanelView } from "../web/ai_studio/panel_view.js";
import { wirePresets } from "../web/ai_studio/preset_panel.js";
import { createPresetTools } from "../web/ai_studio/preset_tools.js";
import { BOX_SWITCHES } from "../web/ai_studio/studio_logic.js";
import { fakeServer, installFakeDom, settle } from "./preset_fakes.mjs";

const KEYS = Object.keys(BOX_SWITCHES);
const BASE_CTX = {
  choices: async () => ({ items: [] }),
  engineRoom: { collapsed: () => true, toggle() {} },
  i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
  hasRole: () => true,
  fetchChoices: async () => ({}),
};
let hooks = null; // the wiring a test left behind: its list is closed before the next test

/** A panel in the page wired the way studio_main.js does it; `load: false` leaves the presets unread. */
async function setup(state = { negative: "my negative" }, { load = true, fail = null } = {}) {
  hooks?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  server.seed("negative", "Anima 기본", "worst quality, low quality");
  server.seed("prompt_artist", "Ena", "@enaa97");
  server.fail = fail;
  const env = { doc, server, notices: [], view: null };
  hooks = wirePresets(createPresetTools(server.route), { panel: () => env.view, notice: (text) => env.notices.push(text) });
  env.hooks = hooks;
  env.view = new PanelView({ value: JSON.stringify(state) }, { ...BASE_CTX, boxLabel: hooks.boxLabel, boxPreset: hooks.boxPreset });
  doc.body.append(env.view.element);
  if (load) hooks.load();
  await settle();
  env.name = (key) => env.view.element.querySelector(`.ais-box-label[data-section="${key}"]`);
  env.mark = (key) => env.view.element.querySelector(`.ais-box-mark[data-for="${key}"]`);
  env.shown = (key) => (env.mark(key).hidden ? null : env.mark(key).textContent);
  env.popup = () => doc.body.querySelector(".ais-pq");
  return env;
}

test("the presets are read once for the panel, and the marks under the names appear when they arrive", async () => {
  const env = await setup({ negative: "worst quality, low quality", prompt: "1girl" }, { load: false });
  assert.deepEqual(env.server.calls, []); // nothing is read at wiring
  assert.deepEqual(KEYS.map(env.shown), KEYS.map(() => null));
  env.hooks.load();
  env.hooks.load(); // a second panel node, a reloaded workflow
  await settle();
  assert.deepEqual(env.server.calls, [["/presets"]]);
  assert.equal(env.shown("negative"), "Anima 기본");
  assert.equal(env.shown("prompt"), "저장 안 됨");
  assert.equal(env.shown("prompt_artist"), null);
});

test("all eight names are buttons: a press opens that box's quick list, and a picked preset fills the box and its mark", async () => {
  const env = await setup();
  for (const key of KEYS) assert.equal(env.name(key).tag, "button", key);
  assert.equal(env.shown("negative"), "저장 안 됨");
  env.name("negative").click();
  await settle();
  assert.match(env.popup().getAttribute("aria-label"), /네거티브/);
  env.popup().querySelector(".ais-pq-pick").click();
  await settle();
  assert.equal(env.view.state.negative, "worst quality, low quality");
  assert.equal(env.view.element.querySelector('[data-box="negative"]').querySelector("textarea").value, "worst quality, low quality");
  assert.equal(env.shown("negative"), "Anima 기본");
  assert.equal(env.popup(), null);
  assert.equal(env.notices.length, 1, "what was done goes to the studio's notice line");
  for (const key of KEYS.filter((box) => box !== "negative")) { // every positive box opens its own list
    env.name(key).click();
    await settle();
    assert.equal(env.popup().querySelector(".ais-pq-title").textContent, `${env.name(key).textContent} 프리셋`, key);
  }
});

test("'+ 지금 내용 저장' turns 저장 안 됨 into the new preset's name, with no redraw of the panel", async () => {
  const env = await setup({ prompt_character: "hero, cape" });
  const scroll = env.view.element.querySelector(".ais-scroll");
  assert.equal(env.shown("prompt_character"), "저장 안 됨");
  env.name("prompt_character").click();
  await settle();
  env.popup().querySelector(".ais-pq-save").click();
  await settle();
  assert.equal(env.server.doc.sections.prompt_character.length, 1);
  assert.equal(env.shown("prompt_character"), env.server.doc.sections.prompt_character[0].name);
  assert.equal(env.view.element.querySelector(".ais-scroll"), scroll);
});

test("'프리셋 관리…' opens the big window on that box, working on the panel's boxes", async () => {
  const env = await setup({ prompt_artist: "@someone" });
  env.name("prompt_artist").click();
  await settle();
  env.popup().querySelector(".ais-pq-manage").click();
  await settle();
  const win = env.doc.body.one("ais-pp");
  assert.equal(win.querySelector("h1").textContent, "작가 프리셋");
  win.one("ais-pp-card").one("ais-pp-replace").click();
  await settle();
  assert.equal(env.view.state.prompt_artist, "@enaa97");
  assert.equal(env.shown("prompt_artist"), "Ena");
  win.one("ais-pp-icon").click();
});

test("with no panel on the graph the hooks do nothing and do not throw", async () => {
  const env = await setup();
  env.view = null;
  assert.doesNotThrow(() => env.hooks.boxLabel("negative", env.doc.createElement("button")));
  assert.equal(env.hooks.boxPreset("negative", "worst quality, low quality").name, "Anima 기본");
  await settle();
});

test("a failed read leaves the marks empty and warns once; the next panel asks again", async () => {
  const warned = [];
  const warn = console.warn;
  console.warn = (...args) => warned.push(args.join(" "));
  try {
    const env = await setup({ negative: "worst quality, low quality" }, { fail: "500 Internal Server Error" });
    assert.equal(env.shown("negative"), null);
    assert.equal(warned.length, 1);
    assert.match(warned[0], /preset/i);
    env.server.fail = null;
    env.hooks.load();
    await settle();
    assert.equal(env.shown("negative"), "Anima 기본");
  } finally {
    console.warn = warn;
  }
});

// Seen in a real browser: the first list was drawn before preset.css arrived (static, about 550px tall), so it was put
// "above" its name by that height, at the top of the window, and stayed there.
test("the lists' stylesheet is asked for when the panel is made, before the first press on a name", async () => {
  const env = await setup({}, { load: false });
  assert.equal(env.doc.getElementById("aistudio-preset-css"), null);
  env.hooks.load();
  const link = env.doc.getElementById("aistudio-preset-css");
  assert.equal(link.rel, "stylesheet");
  assert.match(link.href, /preset\.css$/);
  env.hooks.load();
  assert.equal(env.doc.head.all((el) => el.id === "aistudio-preset-css").length, 1);
});

test("a list drawn before its stylesheet arrived is placed again when it does", async () => {
  const env = await setup();
  const link = env.doc.getElementById("aistudio-preset-css"); // asked for, not here yet (no `sheet`)
  env.name("negative").rect = { left: 40, top: 300, right: 100, bottom: 316, width: 60, height: 16 };
  env.name("negative").click();
  await settle();
  assert.equal(env.popup().style.top, "320px"); // worked out from what the page looked like without the rules
  env.name("negative").rect = { left: 40, top: 500, right: 100, bottom: 516, width: 60, height: 16 }; // the rules change the layout
  link.fire("load");
  assert.equal(env.popup().style.top, "520px", "placed again");
  link.sheet = {}; // the stylesheet is in: a later list is placed at opening and waits for nothing
  env.name("negative").click();
  env.name("negative").click();
  await settle();
  assert.equal(env.popup().style.top, "520px");
  assert.equal((link.listeners.load || []).length, 1, "no new wait once the stylesheet is there");
});

test("studio_main.js makes the tools once and hands the panel the two hooks", () => {
  const main = readFileSync(new URL("../web/ai_studio/studio_main.js", import.meta.url), "utf-8");
  assert.equal(main.match(/createPresetTools\(/g)?.length, 1);
  assert.match(main, /wirePresets\(createPresetTools\(route\),\s*\{\s*panel:\s*\(\)\s*=>\s*viewOf\(PANEL\),\s*notice\s*\}\)/);
  assert.match(main, /boxLabel:\s*presets\.boxLabel/);
  assert.match(main, /boxPreset:\s*presets\.boxPreset/);
  assert.match(main, /presets\.load\(\)/);
});
