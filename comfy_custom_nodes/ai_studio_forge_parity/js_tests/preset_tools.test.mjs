// node:test with a fake document (preset_fakes.mjs): the presets' one entry point for the studio
// (web/ai_studio/preset_tools.js) — the store, the quick list and the window tied together. Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

import { PRESET_LABELS, PRESET_SECTIONS } from "../web/ai_studio/preset_logic.js";
import { createPresetTools } from "../web/ai_studio/preset_tools.js";
import { fakeServer, installFakeDom, settle } from "./preset_fakes.mjs";

function setup() {
  const doc = installFakeDom();
  const server = fakeServer();
  server.seed("prompt_artist", "Ena", "@enaa97");
  const tools = createPresetTools(server.route);
  const anchor = doc.createElement("button");
  doc.body.append(anchor);
  const texts = { prompt_artist: "@enaa97, ", negative: "" };
  const io = { getText: (key) => texts[key] ?? "", setText: (key, text) => { texts[key] = text; } };
  return { doc, server, tools, anchor, texts, io };
}

test("the tools carry the boxes, their labels and a store that reads the route once", async () => {
  const { server, tools } = setup();
  assert.equal(tools.sections, PRESET_SECTIONS);
  assert.equal(tools.labels, PRESET_LABELS);
  assert.equal(tools.match("prompt_artist", "@enaa97"), null, "nothing is known before the first read");
  let told = 0;
  const off = tools.subscribe(() => { told += 1; });
  await Promise.all([tools.load(), tools.load()]);
  assert.deepEqual(server.calls, [["/presets"]]);
  assert.equal(told, 1);
  assert.equal(tools.match("prompt_artist", " @enaa97,\n").name, "Ena");
  assert.equal(tools.match("negative", "x"), null);
  off();
});

test("the quick list's '프리셋 관리…' opens the window on that box with the same getText / setText", async () => {
  const { doc, tools, anchor, texts, io } = setup();
  const notices = [];
  const quick = tools.openQuick(anchor, "prompt_artist", { ...io, onNotice: (message) => notices.push(message) });
  await settle();
  assert.equal(quick.key, "prompt_artist");
  assert.equal(doc.body.one("ais-pq").one("ais-pq-dot").classList.contains("on"), true);
  doc.body.one("ais-pq").one("ais-pq-manage").click();
  assert.equal(doc.body.querySelector(".ais-pq"), null);
  const win = doc.body.one("ais-pp");
  assert.equal(win.querySelector("h1").textContent, "작가 프리셋");
  win.one("ais-pp-clear").click();
  assert.equal(texts.prompt_artist, "", "the window writes through the panel's setText");
  win.one("ais-pp-card").one("ais-pp-replace").click();
  assert.equal(texts.prompt_artist, "@enaa97");
  win.one("ais-pp-icon").click();
});

test("over a panel that redraws, the window opened from the list gives the focus to the label that is there", async () => {
  const { doc, tools, io } = setup();
  const panel = doc.createElement("div");
  doc.body.append(panel);
  const draw = () => {
    const label = doc.createElement("button");
    label.className = "ais-box-label";
    label.setAttribute("data-section", "prompt_artist");
    panel.replaceChildren(label);
    return label;
  };
  const first = draw();
  tools.openQuick(first, "prompt_artist", io);
  await settle();
  doc.body.one("ais-pq").one("ais-pq-manage").click();
  const win = doc.body.one("ais-pp");
  const now = draw(); // a 바꾸기 made the panel redraw
  win.one("ais-pp-icon").click();
  assert.equal(doc.activeElement, now);

  const quick = tools.openQuick(() => panel.querySelector(".ais-box-label"), "prompt_artist", io); // or a function
  assert.equal(quick.key, "prompt_artist");
  quick.close();
});

test("a caller's own onManage and onClose are used when given", async () => {
  const { doc, tools, anchor, io } = setup();
  const managed = [];
  tools.openQuick(anchor, "negative", { ...io, onManage: (key) => managed.push(key) });
  doc.body.one("ais-pq").one("ais-pq-manage").click();
  assert.deepEqual(managed, ["negative"]);
  assert.equal(doc.body.querySelector(".ais-pp"), null);
  let closed = 0;
  const handle = tools.openWindow("negative", { ...io, onClose: () => { closed += 1; } });
  handle.close();
  assert.equal(closed, 1);
});

// ---- the preset modules as ComfyUI serves them
const WEB = new URL("../web/ai_studio/", import.meta.url);
const presetFiles = readdirSync(WEB).filter((name) => name.startsWith("preset"));

test("every preset module is a pure ES module that imports only its neighbours", () => {
  const scripts = presetFiles.filter((name) => name.endsWith(".js"));
  assert.deepEqual(scripts.sort(), ["preset_api.js", "preset_dom.js", "preset_logic.js", "preset_panel.js", "preset_quick.js", "preset_tools.js", "preset_window.js", "preset_window_view.js"]);
  for (const name of scripts) {
    const source = readFileSync(new URL(name, WEB), "utf-8");
    assert.match(source, /Pure ES module: nothing runs at import/, name);
    for (const [, spec] of source.matchAll(/from\s+"([^"]+)"/g)) assert.match(spec, /^\.\/[\w]+\.js$/, `${name} imports ${spec}`);
  }
});

test("the stylesheet styles only the preset classes (nothing of the panel or the LoRA window)", () => {
  assert.ok(presetFiles.includes("preset.css"));
  const css = readFileSync(new URL("preset.css", WEB), "utf-8").replace(/\/\*[\s\S]*?\*\//g, "");
  const classes = [...css.matchAll(/\.([a-zA-Z][\w-]*)/g)].map((match) => match[1]).filter((name) => name.startsWith("ais-"));
  assert.ok(classes.length > 100);
  assert.deepEqual([...new Set(classes.filter((name) => !/^ais-p[pq](-|$)/.test(name)))], []);
  for (const selector of css.split("}").map((rule) => rule.split("{")[0].trim()).filter((s) => s && !s.startsWith("@"))) {
    for (const part of selector.split(",")) assert.match(part.trim(), /^\.ais-p[pq]/, `"${part.trim()}" is not scoped to the presets`);
  }
});

test("every class the preset modules put on an element has a rule or is a state / hook the tests name", () => {
  const css = readFileSync(new URL("preset.css", WEB), "utf-8");
  const hooks = new Set(["ais-pp-up", "ais-pp-down", "ais-pp-gridsec"]); // styled through .ais-pp-btn / layout only
  const used = new Set();
  for (const name of presetFiles.filter((file) => file.endsWith(".js"))) {
    for (const [, cls] of readFileSync(new URL(name, WEB), "utf-8").matchAll(/\b(ais-p[pq](?:-[a-z0-9]+)*)\b/g)) used.add(cls);
  }
  const missing = [...used].filter((cls) => !hooks.has(cls) && !css.includes(`.${cls}`));
  assert.deepEqual(missing, []);
});
