// node:test for the XYZ LoRA axis' pure helpers (web/ai_studio/xyz_lora_view.js) and how the panel keeps its block; the
// DOM part is checked in a browser. Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { addPickedRows, baseItems, epochText, fileItems, loraWithDefaults, newLoraBlock, refillFiles, refaceFiles, rowFaceOf } from "../web/ai_studio/xyz_lora_view.js";
import { optionItems } from "../web/ai_studio/studio_dom.js";
import { XYZ_AXES, XyzView, xyzWithDefaults } from "../web/ai_studio/xyz_view.js";

const EMPTY_ROW = { lora: "", strength: null, trigger: "", epoch_mode: "off", epoch: null };

test("a new block has the base row, on the LoRA it is given", () => {
  assert.deepEqual(newLoraBlock("a-000014"), { find: "", after: false, rows: [{ ...EMPTY_ROW, lora: "a-000014" }] });
  assert.deepEqual(newLoraBlock(), { find: "", after: false, rows: [EMPTY_ROW] });
});

test("loraWithDefaults keeps a good block", () => {
  const block = {
    find: "@enaa97",
    after: true,
    rows: [
      { lora: "a-000014", strength: 0.8, trigger: "", epoch_mode: "last", epoch: null },
      { lora: "b", strength: null, trigger: "@b", epoch_mode: "direct", epoch: 20 },
    ],
  };
  assert.deepEqual(loraWithDefaults(block), block);
});

test("loraWithDefaults repairs a broken block and never loses the base row", () => {
  assert.deepEqual(loraWithDefaults(null), newLoraBlock());
  assert.deepEqual(loraWithDefaults({ rows: [] }), newLoraBlock());
  assert.deepEqual(loraWithDefaults({ find: 5, after: "yes", rows: ["x", { lora: 1, strength: "0.5", epoch_mode: "soon", epoch: Infinity, extra: 1 }] }), {
    find: "",
    after: false,
    rows: [EMPTY_ROW, EMPTY_ROW],
  });
});

test("fileItems lists file stems, like the panel's LoRA tags", () => {
  assert.deepEqual(fileItems(["LoKr\\RAIY\\a-000001.safetensors", "style.safetensors"]), [
    { value: "a-000001", label: "a-000001" },
    { value: "style", label: "style" },
  ]);
  assert.deepEqual(fileItems(undefined), []);
});

test("fileItems keeps the folder of a name two files share, so each file can be picked", () => {
  assert.deepEqual(fileItems(["LoKr\\a-000001.safetensors", "TSM\\a-000001.safetensors", "TSM\\b.safetensors"]), [
    { value: "LoKr/a-000001", label: "LoKr/a-000001" },
    { value: "TSM/a-000001", label: "TSM/a-000001" },
    { value: "b", label: "b" },
  ]);
});

test("fileItems counts names the way the tag lookup does, ignoring case", () => {
  assert.deepEqual(fileItems(["A\\foo.safetensors", "B\\FOO.safetensors"]).map((item) => item.value), ["A/foo", "B/FOO"]);
});

test("optionItems marks a gone value only when the list is known, and a computed list always is", () => {
  assert.deepEqual(optionItems([], "a"), [{ value: "a", label: "a" }]);
  assert.deepEqual(optionItems([], "a", { loaded: true }), [{ value: "a", label: "a (없음)" }]);
  assert.deepEqual(optionItems(["b"], "a").at(-1), { value: "a", label: "a (없음)" });
});

test("baseItems are the LoRAs a run loads: list LoRAs that are on, prompt tags and negative tags", () => {
  const state = {
    prompt: "1boy <lora:tagged:0.5>",
    negative: "<lora:neg:1>",
    loras: [{ on: true, name: "listed", strength: 1 }, { on: false, name: "off", strength: 1 }],
  };
  assert.deepEqual(baseItems(state).map((item) => item.value), ["tagged", "listed", "neg"]);
  assert.deepEqual(baseItems({}), []);
});

test("epochText gives the number a narrow drop-down cuts off", () => {
  assert.equal(epochText("@enaa97_2.9b_lokr_came_b10_2e06-000014"), "에폭 14");
  assert.equal(epochText("@enaa97_2.9b_lokr_came_b10_2e06"), "에폭");
  assert.equal(epochText("run-14"), "에폭");
  assert.equal(epochText("a-000014.safetensors"), "에폭 14");
});

test("LoRA is an axis without a value list", () => {
  assert.deepEqual({ ...XYZ_AXES.find((axis) => axis.id === "lora") }, { id: "lora", label: "LoRA", kind: "lora", choices: null, values: [], group: "" });
});

test("xyzWithDefaults keeps the block of a LoRA axis, also while another axis is picked", () => {
  const block = newLoraBlock("a");
  const saved = xyzWithDefaults({ axes: [{ axis: "lora", text: "", lora: block }, { axis: "steps", text: "1", lora: block }] });
  assert.deepEqual(saved.axes[0], { axis: "lora", text: "", lora: block });
  assert.deepEqual(saved.axes[1], { axis: "steps", text: "1", lora: block });
  assert.deepEqual(saved.axes[2], { axis: "nothing", text: "" });
});

test("a LoRA axis saved without a block gets one", () => {
  assert.deepEqual(xyzWithDefaults({ axes: [{ axis: "lora", text: "" }] }).axes[0].lora, newLoraBlock());
});

function bareView(settings, answer, state = {}) {
  const view = Object.create(XyzView.prototype);
  view.settings = xyzWithDefaults(settings);
  view.choices = {};
  view.typed = new Set();
  view.loraColumns = {};
  view.renders = 0;
  view.render = () => { view.renders += 1; view.baseLoras = baseItems(view.ctx.getState()); };
  view.element = { querySelectorAll: () => [] };
  view.ctx = { getState: () => state, plan: async () => answer, choices: async () => ({ items: ["a.safetensors"] }) };
  return view;
}

async function settle(t, view) {
  view.refreshEstimate();
  t.mock.timers.tick(1000);
  for (let i = 0; i < 3; i++) await Promise.resolve();
}

test("the plan's column list is kept, and dropped with an error or while XYZ is off", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const columns = [{ label: "e14", name: "a-000014", strength: "1", trigger: "없음", changed: false }];
  const settings = { on: true, axes: [{ axis: "lora", text: "" }] };

  const good = bareView(settings, { errors: [], labels: { x: ["e14"] }, lora_preview: { x: columns } });
  await settle(t, good);
  assert.deepEqual(good.loraColumns, { x: columns });
  assert.equal(good.estimate, "1장");

  const bad = bareView(settings, { errors: ["기준: 에폭 20", "생성하지 않습니다."], labels: {}, lora_preview: {} });
  bad.loraColumns = { x: columns };
  await settle(t, bad);
  assert.deepEqual(bad.loraColumns, {});
  assert.equal(bad.error, "기준: 에폭 20\n생성하지 않습니다.");

  const off = bareView({ ...settings, on: false }, {});
  off.loraColumns = { x: columns };
  off.refreshEstimate();
  assert.deepEqual(off.loraColumns, {});
});

test("a late answer to an older request does not replace the newer one", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const answers = [];
  const view = bareView({ on: true }, {});
  view.ctx.plan = () => new Promise((ok) => { answers.push(ok); });
  view.refreshEstimate();
  t.mock.timers.tick(1000);
  view.refreshEstimate();
  t.mock.timers.tick(1000);
  answers[1]({ errors: [], labels: { x: ["new"] } });
  for (let i = 0; i < 3; i++) await Promise.resolve();
  answers[0]({ errors: ["old"], labels: {} });
  for (let i = 0; i < 3; i++) await Promise.resolve();
  assert.equal(view.error, "");
  assert.equal(view.estimate, "1장");
});

test("the base row's list follows the panel, and only a change redraws", () => {
  const state = { prompt: "<lora:a:1>", loras: [] };
  const view = bareView({ axes: [{ axis: "lora", text: "" }] }, {}, state);
  view.render();
  view.renders = 0;
  view.syncBaseLoras();
  assert.equal(view.renders, 0);
  state.prompt = "<lora:b:1>";
  view.syncBaseLoras();
  assert.equal(view.renders, 1);
  assert.deepEqual(view.baseLoras.map((item) => item.value), ["b"]);

  const none = bareView({}, {}, state);
  none.syncBaseLoras();
  assert.equal(none.renders, 0);
});

test("the file list is read only for a LoRA axis, and arriving it redraws nothing", async () => {
  const none = bareView({}, {});
  await none.loadLoraFiles();
  assert.equal(none.choices.loras, undefined);

  const view = bareView({ axes: [{ axis: "lora", text: "" }] }, {});
  view.render();
  view.renders = 0;
  await view.loadLoraFiles();
  assert.deepEqual(view.choices.loras, ["a.safetensors"]);
  assert.equal(view.renders, 0);
});

test("the file list goes into the drawn drop-downs, keeping their picks", () => {
  const made = [];
  const box = { value: "a", replaceChildren(...options) { made.push(...options); } };
  globalThis.document = { createElement: () => ({ value: "", title: "", append() {} }) };
  globalThis.Node = class {};
  try {
    refillFiles({ querySelectorAll: () => [box] }, [{ value: "b", label: "b" }]);
  } finally {
    delete globalThis.document;
    delete globalThis.Node;
  }
  assert.deepEqual(made.map((option) => option.value), ["", "b", "a"]);
  assert.equal(box.value, "a");
});

test("a panel that gained a LoRA while the list loaded is still seen", async () => {
  const state = { prompt: "<lora:a:1>" };
  const view = bareView({ axes: [{ axis: "lora", text: "" }] }, {}, state);
  view.render();
  state.prompt = "<lora:a:1> <lora:b:1>";
  view.renders = 0;
  await view.loadLoraFiles();
  assert.equal(view.renders, 1);
});

test("swapping axes takes a list axis' 직접 입력 along; picking a type ends only that axis' own", () => {
  const view = bareView({ axes: [{ axis: "sampler", text: "" }, { axis: "lora", text: "" }, { axis: "scheduler", text: "" }] }, {});
  view.widget = null;
  view.typed = new Set([0]);
  view.swapAxes(0, 1);
  assert.deepEqual([...view.typed], [1]);
  assert.deepEqual(view.settings.axes.map((entry) => entry.axis), ["lora", "sampler", "scheduler"]);
  view.typed.add(2);
  view.pickAxis(2, "vae");
  assert.deepEqual([...view.typed], [1]);
  view.swapAxes(1, 2);
  assert.deepEqual([...view.typed], [2]);
});

test("picking the LoRA axis starts a block on the panel's first LoRA and keeps an existing one", async () => {
  const view = bareView({}, {}, { prompt: "<lora:a:1> <lora:b:1>" });
  view.widget = null;
  view.pickAxis(0, "lora");
  assert.deepEqual(view.settings.axes[0].lora, newLoraBlock("a"));
  view.settings.axes[0].lora.find = "@a";
  view.pickAxis(0, "steps");
  view.pickAxis(0, "lora");
  assert.equal(view.settings.axes[0].lora.find, "@a");
});

const block3 = () => ({ find: "@a", after: true, rows: [{ ...EMPTY_ROW, lora: "base" }, { ...EMPTY_ROW, lora: "x-000001", trigger: "@x" }] });

test("addPickedRows fills the empty last row first", () => {
  const block = block3();
  block.rows.push({ ...EMPTY_ROW });
  const out = addPickedRows(block, [{ name: "r-000010" }, { name: "r-000020" }, { name: "r-000030" }]);
  assert.equal(out, block);
  assert.deepEqual(block.rows.map((row) => row.lora), ["base", "x-000001", "r-000010", "r-000020", "r-000030"]);
  assert.deepEqual(block.rows[3], { ...EMPTY_ROW, lora: "r-000020" });
});

test("addPickedRows keeps base row and existing rows", () => {
  const block = block3();
  const before = JSON.parse(JSON.stringify(block));
  addPickedRows(block, [{ name: "r-000010" }, { name: "r-000020" }]);
  assert.deepEqual(block.rows.slice(0, 2), before.rows);
  assert.deepEqual(block.rows.slice(2).map((row) => row.lora), ["r-000010", "r-000020"]);
  assert.equal(block.find, "@a");
  assert.equal(block.after, true);
  // a base row that is still empty is not filled: only rows after it are
  const lone = newLoraBlock();
  addPickedRows(lone, [{ name: "q" }]);
  assert.deepEqual(lone.rows.map((row) => row.lora), ["", "q"]);
  assert.equal(addPickedRows(block3(), []).rows.length, 2);
});

test("picked names are stems (same as before)", () => {
  const block = block3();
  addPickedRows(block, [{ name: fileItems(["LoKr\\run\\r-000010.safetensors"])[0].value }]);
  assert.equal(block.rows.at(-1).lora, "r-000010");
  assert.deepEqual(block.rows.at(-1), { ...EMPTY_ROW, lora: "r-000010" });
});

test("rowFaceOf shows the short name of the row's file, also a folder-form name", () => {
  const entries = ["LoKr\\run\\r-000010.safetensors", "A\\same.safetensors", "B\\same.safetensors"];
  assert.equal(rowFaceOf("r-000010", entries).missing, false);
  assert.equal(rowFaceOf("r-000010", entries).title, entries[0]);
  assert.equal(rowFaceOf("B/same", entries).title, entries[2]);
  assert.equal(rowFaceOf("", entries).text, "[LoRA]");
  assert.equal(rowFaceOf("gone", entries).missing, true);
  assert.equal(rowFaceOf("gone", []).missing, false);
});

test("refaceFiles updates the drawn row buttons in place when the list arrives", () => {
  const classes = new Set();
  const text = { textContent: "x" };
  const button = {
    title: "",
    getAttribute: () => "r-000010",
    classList: { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)) },
    querySelector: () => text,
  };
  const root = { querySelectorAll: () => [button] };
  refaceFiles(root, ["run\\r-000010.safetensors"]);
  assert.equal(button.title, "run\\r-000010.safetensors");
  assert.equal(classes.has("missing"), false);
  refaceFiles(root, ["other.safetensors"]);
  assert.equal(classes.has("missing"), true);
  assert.match(text.textContent, /없음/);
});
