// node:test for the control panel's "확장기능" group (web/ai_studio/ext_view.js) with a tiny fake document (no DOM in node).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { EXT_SCHEMA } from "../web/ai_studio/ext_schema.js";
import { dropBlockedExt, extMemory, renderExtensions } from "../web/ai_studio/ext_view.js";
import { PanelView } from "../web/ai_studio/panel_view.js";
import { withDefaults } from "../web/ai_studio/studio_logic.js";

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.value = "";
    this.scrollTop = 0;
    this.parent = null;
  }
  append(...nodes) {
    for (const node of nodes) {
      if (typeof node !== "string") node.parent = this;
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) {
    this.children = [];
    this.append(...nodes);
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this); }
  fire(type) {
    for (const fn of this.listeners[type] || []) fn({ target: this });
  }
  get text() {
    return this.children.map((child) => (typeof child === "string" ? child : child.text)).join("");
  }
  /** Every descendant (itself included) the test function accepts, in document order. */
  all(accept) {
    const own = accept(this) ? [this] : [];
    return own.concat(this.children.flatMap((child) => (typeof child === "string" ? [] : child.all(accept))));
  }
  byClass(name) {
    return this.all((element) => element.className.split(" ").includes(name));
  }
  querySelector(selector) {
    return this.byClass(selector.slice(1))[0] ?? null;
  }
  querySelectorAll() {
    return []; // growPrompts: no layout here
  }
}

globalThis.document = { createElement: (tag) => new FakeElement(tag) };
globalThis.Node = FakeElement;

const ITEMS = EXT_SCHEMA.extensions;
const item = (key) => ITEMS.find((entry) => entry.key === key);
const flush = () => new Promise((done) => setTimeout(done));

/** The group over a live state: `set` applies the mutation, `redraw()` draws again with the same view memory. */
function panel({ state = withDefaults({}), hasRole, fetchChoices, memory = extMemory() } = {}) {
  const view = { state, memory, sets: [], draws: 0, element: null };
  const ctx = {
    set: (mutate, rerender) => {
      mutate(state);
      view.sets.push(Boolean(rerender));
      if (rerender) view.redraw();
    },
    hasRole,
    fetchChoices,
    memory,
    rerender: () => view.redraw(),
  };
  view.redraw = () => {
    view.draws += 1;
    view.element = renderExtensions(state, ctx);
    return view.element;
  };
  view.redraw();
  return view;
}

const rowOf = (view, key) => view.element.byClass("ais-ext").find((row) => row.attrs["data-ext"] === key);
const switchOf = (view, key) => rowOf(view, key).byClass("ais-pill")[0];
const caretOf = (view, key) => rowOf(view, key).byClass("ais-caret")[0];
const fieldOf = (view, key, fieldKey) => rowOf(view, key).all((element) => element.attrs["data-field"] === fieldKey)[0];
const expand = (view, key) => caretOf(view, key).fire("click");

test("one row per extension, grouped by the table's groups, nothing unfolded", () => {
  const view = panel();
  assert.deepEqual(view.element.byClass("ais-ext").map((row) => row.attrs["data-ext"]).sort(), ITEMS.map((entry) => entry.key).sort());
  assert.deepEqual(view.element.byClass("ais-ext-group").map((card) => card.byClass("ais-nt")[0].text), ["가이던스", "디테일", "샘플러·VAE", "SAM3"]);
  assert.equal(view.element.byClass("ais-ext-fields").length, 0);
  assert.equal(view.element.byClass("ais-gh")[0].text, "확장기능");
  assert.match(rowOf(view, "guidance").text, /가이던스/);
});

test("the switch toggles state.ext[key].on and the header counts what is on", () => {
  const view = panel();
  switchOf(view, "skim").fire("click");
  assert.equal(view.state.ext.skim.on, true);
  assert.deepEqual(view.sets, [true]);
  assert.equal(switchOf(view, "skim").attrs["aria-pressed"], "true");
  assert.equal(view.element.byClass("ais-gh")[0].text, "확장기능1개 켬");
  switchOf(view, "detail_daemon").fire("click");
  assert.equal(view.element.byClass("ais-ext-count")[0].text, "2개 켬");
  switchOf(view, "skim").fire("click");
  assert.equal(view.state.ext.skim.on, false);
  assert.equal(view.element.byClass("ais-ext-count")[0].text, "1개 켬");
});

test("a role the graph lacks greys the switch, says why and is not counted", () => {
  const state = withDefaults({ ext: { skim: { on: true }, detail_daemon: { on: true } } });
  const view = panel({ state, hasRole: (role) => role !== "ext_skim" });
  const button = switchOf(view, "skim");
  assert.equal(button.attrs["aria-disabled"], "true");
  assert.equal(button.attrs["aria-pressed"], "false");
  assert.match(button.attrs.title, /불가: 이 워크플로에는 Skim CFG 노드가 없음/);
  button.fire("click");
  assert.equal(view.sets.length, 0);
  assert.equal(view.element.byClass("ais-ext-count")[0].text, "1개 켬");
  assert.equal(switchOf(view, "detail_daemon").attrs["aria-disabled"], undefined);
});

test("without hasRole every switch works (the role check is optional)", () => {
  const view = panel();
  assert.equal(switchOf(view, "skim").attrs["aria-disabled"], undefined);
});

test("dropBlockedExt turns off what the graph cannot run", () => {
  const ext = withDefaults({ ext: { skim: { on: true }, detail_daemon: { on: true } } }).ext;
  dropBlockedExt(ext, (role) => role !== "ext_skim");
  assert.deepEqual([ext.skim.on, ext.detail_daemon.on], [false, true]);
  const kept = withDefaults({ ext: { skim: { on: true } } }).ext;
  dropBlockedExt(kept, undefined);
  assert.equal(kept.skim.on, true);
});

test("unfolding draws the fields by kind", () => {
  const view = panel();
  expand(view, "speed");
  const kinds = Object.fromEntries(item("speed").fields.map((field) => [field.kind, field]));
  const tag = (kind) => fieldOf(view, "speed", kinds[kind].key).tag;
  assert.equal(tag("bool"), "button");
  assert.equal(tag("int"), "input");
  assert.equal(tag("float"), "input");
  assert.equal(tag("choice"), "select");
  assert.equal(tag("text"), "input");
  assert.equal(fieldOf(view, "speed", kinds.float.key).attrs.type, "number");
  assert.equal(rowOf(view, "speed").byClass("ais-label").length, item("speed").fields.length);
  assert.equal(view.element.byClass("ais-ext-wide").length, 1); // its card takes the whole width
  expand(view, "speed");
  assert.equal(view.element.byClass("ais-ext-fields").length, 0);
});

test("a number is clamped to the table's range, and the box shows the clamped value", () => {
  const view = panel();
  expand(view, "skim");
  const field = item("skim").fields.find((entry) => entry.kind === "float");
  const box = fieldOf(view, "skim", field.key);
  box.value = String(field.max + 1000);
  box.fire("change");
  assert.equal(view.state.ext.skim[field.key], field.max);
  assert.equal(box.value, String(field.max));
  box.value = String(field.min - 1000);
  box.fire("change");
  assert.equal(view.state.ext.skim[field.key], field.min);
  box.value = "";
  box.fire("change"); // an empty entry is ignored, not zero
  assert.equal(view.state.ext.skim[field.key], field.min);
});

test("an int field drops the fraction", () => {
  const view = panel();
  expand(view, "er_sde");
  const field = item("er_sde").fields.find((entry) => entry.kind === "int");
  const box = fieldOf(view, "er_sde", field.key);
  box.value = String(field.min + 1.7);
  box.fire("change");
  assert.equal(view.state.ext.er_sde[field.key], field.min + 1);
  assert.equal(box.value, String(field.min + 1));
});

test("bool, choice and text fields write into state.ext", () => {
  const view = panel();
  expand(view, "speed");
  const fields = item("speed").fields;
  const bool = fields.find((entry) => entry.kind === "bool");
  fieldOf(view, "speed", bool.key).fire("click");
  assert.equal(view.state.ext.speed[bool.key], !bool.default);
  const choice = fieldOf(view, "speed", "mode");
  choice.value = "respace";
  choice.fire("change");
  assert.equal(view.state.ext.speed.mode, "respace");
  const text = fields.find((entry) => entry.kind === "text");
  const input = fieldOf(view, "speed", text.key);
  input.value = "1/2";
  input.fire("change");
  assert.equal(view.state.ext.speed[text.key], "1/2");
});

test("the suite folds into its sections and keeps what is unfolded across a redraw", () => {
  const memory = extMemory();
  const view = panel({ memory });
  expand(view, "guidance");
  const heads = view.element.byClass("ais-ext-sec");
  const names = [...new Set(item("guidance").fields.map((field) => field.section))];
  assert.equal(names.length, 16);
  assert.deepEqual(heads.map((head) => head.text.replace(/^[▾▴] /, "")), names);
  assert.equal(view.element.byClass("ais-ext-fields").length, 0);
  heads[1].fire("click");
  const opened = view.element.byClass("ais-ext-fields");
  assert.equal(opened.length, 1);
  assert.equal(opened[0].byClass("ais-label").length, item("guidance").fields.filter((field) => field.section === names[1]).length);
  switchOf(view, "guidance").fire("click"); // a redraw from the state keeps the unfolded rows
  assert.equal(view.element.byClass("ais-ext-fields").length, 1);
  assert.equal(panel({ memory }).element.byClass("ais-ext-fields").length, 1);
});

test("a section with a live switch is marked while folded", () => {
  const state = withDefaults({ ext: { guidance: { guid_apg_enabled: true } } });
  const view = panel({ state });
  expand(view, "guidance");
  const marked = view.element.byClass("ais-ext-sec").filter((head) => head.className.split(" ").includes("lit"));
  assert.deepEqual(marked.map((head) => head.text.replace(/^[▾▴] /, "")), ["APG"]);
});

const objectInfo = (name, items) => ({ required: { [name]: [items, {}] } });

test("a file list is read when its extension is first unfolded, once per class", async () => {
  const asked = [];
  const view = panel({ fetchChoices: async (name) => { asked.push(name); return objectInfo("lora_name", ["a.safetensors", "b.safetensors"]); } });
  assert.deepEqual(asked, []); // nothing is read while folded
  expand(view, "lora_block");
  assert.deepEqual(asked, ["ForgeNeoLoraBlockWeight"]);
  await flush();
  const options = () => fieldOf(view, "lora_block", "lora_name").children.map((option) => option.attrs.title ?? option.text);
  assert.deepEqual(options(), ["[LoRA]", "a.safetensors", "b.safetensors"]);
  expand(view, "lora_block");
  expand(view, "lora_block");
  await flush();
  assert.deepEqual(asked, ["ForgeNeoLoraBlockWeight"]);
  assert.deepEqual(options(), ["[LoRA]", "a.safetensors", "b.safetensors"]);
});

test("an extension without a file list never asks", async () => {
  const asked = [];
  const view = panel({ fetchChoices: async (name) => { asked.push(name); return {}; } });
  expand(view, "skim");
  await flush();
  assert.deepEqual(asked, []);
});

test("a switched-on extension reads its list too (the run notices need it)", async () => {
  const asked = [];
  const state = withDefaults({ ext: { degrid: { on: true } } });
  const memory = extMemory();
  panel({ state, memory, fetchChoices: async (name) => { asked.push(name); return objectInfo("model_name", ["auto"]); } });
  assert.deepEqual(asked, ["ForgeNeoAnimaVAEDeGrid"]);
  await flush();
  assert.deepEqual(memory.lists.get("ForgeNeoAnimaVAEDeGrid"), objectInfo("model_name", ["auto"]));
});

test("a failed read shows the current value only and is tried again on the next unfold", async (t) => {
  t.mock.method(console, "warn", () => {});
  let fail = true;
  const state = withDefaults({ ext: { pid: { ckpt_name: "kept.safetensors" } } });
  const view = panel({ state, fetchChoices: async () => {
    if (fail) throw new Error("500");
    return objectInfo("ckpt_name", ["new.safetensors"]);
  } });
  expand(view, "pid");
  await flush();
  const options = () => fieldOf(view, "pid", "ckpt_name").children.map((option) => option.attrs.title ?? option.text);
  assert.deepEqual(options(), ["[체크포인트]", "kept.safetensors"]);
  assert.equal(fieldOf(view, "pid", "ckpt_name").value, "kept.safetensors");
  assert.equal(console.warn.mock.callCount(), 1);
  fail = false;
  expand(view, "pid");
  expand(view, "pid");
  await flush();
  assert.deepEqual(options(), ["[체크포인트]", "new.safetensors", "kept.safetensors (없음)"]);
});

// ---------------------------------------------------------------------------------------------- PanelView

function panelView(state = {}, hasRole = () => true, fetchChoices = async () => ({})) {
  const ctx = {
    choices: async () => ({ items: [] }),
    engineRoom: { collapsed: () => true, toggle() {} },
    i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
    hasRole,
    fetchChoices,
  };
  return new PanelView({ value: JSON.stringify(state) }, ctx);
}

test("the group sits between the tabs and the engine-room bar", () => {
  const view = panelView();
  const classes = view.element.byClass("ais-scroll")[0].children.map((child) => child.className);
  assert.equal(classes.length, 4);
  assert.match(classes[1], /ais-tabs-wrap/);
  assert.match(classes[2], /ais-ext-root/);
  assert.match(classes[3], /ais-engine/);
});

test("the engine-room label names the rows up to ⑩", () => {
  assert.match(panelView().element.byClass("ais-engine")[0].text, /①~⑩/);
});

test("a redraw keeps the scroll position and what is unfolded", async () => {
  const view = panelView();
  view.element.byClass("ais-scroll")[0].scrollTop = 420;
  view.element.byClass("ais-ext").find((row) => row.attrs["data-ext"] === "skim").byClass("ais-caret")[0].fire("click");
  await flush();
  assert.equal(view.element.byClass("ais-scroll")[0].scrollTop, 420);
  assert.equal(view.element.byClass("ais-ext-fields").length, 1);
  view.element.byClass("ais-ext").find((row) => row.attrs["data-ext"] === "skim").byClass("ais-pill")[0].fire("click");
  assert.equal(view.state.ext.skim.on, true);
  assert.equal(view.element.byClass("ais-scroll")[0].scrollTop, 420);
  assert.equal(view.element.byClass("ais-ext-fields").length, 1);
});

test("getState leaves out extensions the graph cannot run, and saves their switch as it was", () => {
  const saved = { ext: { skim: { on: true }, detail_daemon: { on: true } } };
  const view = panelView(saved, (role) => role !== "ext_skim");
  const state = view.getState();
  assert.deepEqual([state.ext.skim.on, state.ext.detail_daemon.on], [false, true]);
  assert.equal(view.state.ext.skim.on, true);
});

// ---------------------------------------------------------------------------------------------- prompt tag autocomplete wiring

function withBody(run) {
  const doc = globalThis.document;
  const saved = [doc.body, doc.addEventListener, doc.removeEventListener];
  doc.body = new FakeElement("body");
  doc.addEventListener = doc.removeEventListener = () => {}; // the popup's click-away listener
  return Promise.resolve(run(doc.body)).finally(() => { [doc.body, doc.addEventListener, doc.removeEventListener] = saved; });
}

function panelWithTags(tags, state = {}) {
  const ctx = {
    choices: async () => ({ items: [] }),
    engineRoom: { collapsed: () => true, toggle() {} },
    i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
    hasRole: () => true,
    fetchChoices: async () => ({}),
    tags,
  };
  return new PanelView({ value: JSON.stringify(state) }, ctx);
}

const later = (ms) => new Promise((done) => setTimeout(done, ms));
const boxRect = () => ({ left: 10, top: 10, bottom: 60, right: 300, width: 290, height: 50 });
const boxes = (view, ...names) => names.map((name) => Object.assign(view.element.byClass(name)[0], { getBoundingClientRect: boxRect }));

test("autocomplete: typing in the prompt asks for tags and still saves the text; the negative box is wired too", () => withBody(async (body) => {
  const asked = [];
  const view = panelWithTags(async (query, limit) => { asked.push([query, limit]); return [{ name: "long_hair", category: 0, count: 5 }]; });
  await later(5); // the choice lists arrive and redraw the panel once; type into the boxes that stay
  const [prompt, negative] = boxes(view, "ais-prompt", "ais-negative");
  prompt.value = "1girl, lo";
  prompt.fire("input");
  assert.equal(view.state.prompt, "1girl, lo"); // the panel's own handler ran
  await later(220);
  assert.deepEqual(asked, [["lo", 12]]);
  assert.equal(body.children.length, 1); // the popup is on document.body
  assert.equal(view.autocomplete.isOpen(), true);
  negative.value = "bl";
  negative.fire("input");
  assert.equal(view.state.negative, "bl");
  await later(220);
  assert.deepEqual(asked.at(-1), ["bl", 12]);
  view.dispose();
  assert.equal(body.children.length, 0);
}));

test("autocomplete: <lora: in the prompt lists the panel's LoRA files; a redraw closes the popup", () => withBody(async () => {
  const asked = [];
  const view = panelWithTags(async (query) => { asked.push(query); return []; });
  await later(5);
  view.choices.loras = ["meow/meow_e6.safetensors"];
  const [prompt] = boxes(view, "ais-prompt");
  prompt.value = "<lora:me";
  prompt.fire("input");
  assert.equal(view.autocomplete.isOpen(), true);
  assert.deepEqual(asked, []); // answered from the list, no request
  view.render();
  assert.equal(view.autocomplete.isOpen(), false);
}));

test("autocomplete: a panel without a tags route simply has no autocomplete", () => {
  const view = panelView();
  assert.equal(view.autocomplete, null);
});
