// node:test for the control panel's "확장기능" group (web/ai_studio/ext_view.js) with a tiny fake document (no DOM in node).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { EXT_SCHEMA } from "../web/ai_studio/ext_schema.js";
import { dropBlockedExt, extMemory, renderExtensions } from "../web/ai_studio/ext_view.js";
import { referenceBlock, renderReference } from "../web/ai_studio/reference_view.js";
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
    for (const fn of this.listeners[type] || []) fn({ target: this, stopPropagation() {} });
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

// ---------------------------------------------------------------------------------------------- 후처리 card, Hires card

const LISTS = {
  samplers: ["euler", "heun"], schedulers: ["simple", "karras"], vaes: ["a.safetensors"], sam3: [{ value: "C:/sam3.pt", label: "sam3.pt" }],
  upscalers: ["latent:bicubic", "Lanczos", "4x-UltraSharp.pth", "2x.pth"], controlnets: ["lllite.safetensors"],
  face_models: [{ value: "C:/face_yolov8n.pt", label: "face_yolov8n.pt" }],
};

async function listedPanel(state = {}, extra = {}) {
  const ctx = {
    ...extra,
    choices: async (kind) => ({ items: LISTS[kind] || [] }),
    engineRoom: { collapsed: () => true, toggle() {} },
    i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
    hasRole: () => true,
    fetchChoices: async () => ({}),
  };
  const view = new PanelView({ value: JSON.stringify(state) }, ctx);
  await flush();
  return view;
}

const caret = (view, label) => view.element.all((element) => element.attrs["aria-label"] === `${label} 세부 설정`)[0];
const section = (view, name) => view.element.byClass("ais-ext-sec").find((button) => button.text.includes(name));
const field = (view, key) => view.element.all((element) => element.attrs["data-field"] === key)[0];
const rowLabelled = (view, label) => view.element.byClass("ais-row").find((row) => row.byClass("ais-label")[0]?.text === label);
const postNode = (view) => view.element.byClass("ais-node").find((node) => node.byClass("ais-nt")[0].text.startsWith("후처리"));
const hiresNode = (view) => view.element.byClass("ais-node").find((node) => node.byClass("ais-nt")[0].text === "Hires fix");

test("the SAM3 caret unfolds its sections in the card", async () => {
  const view = await listedPanel();
  assert.equal(postNode(view).byClass("ais-ext-sec").length, 0);
  caret(view, "SAM3").fire("click");
  await flush();
  assert.deepEqual(postNode(view).byClass("ais-ext-sec").map((button) => button.text.slice(2)), ["찾기", "마스크", "얼굴", "인페인트", "샘플링", "ControlNet", "실행"]);
  caret(view, "SAM3").fire("click");
  await flush();
  assert.equal(postNode(view).byClass("ais-ext-sec").length, 0);
});

test("the common cards and the extension groups stack one per line", async () => {
  const view = await listedPanel();
  const stacks = view.element.byClass("ais-stack");
  assert.equal(stacks.length, 2);
  assert.deepEqual(stacks[0].children.map((node) => node.byClass("ais-nt")[0].children[0].text), ["프롬프트", "모델 · 샘플링", "LoRA", "후처리 (켜면 바로 적용)"]);
  assert.match(stacks[1].parent.className, /ais-ext-root/);
  assert.equal(view.element.byClass("ais-grid3").length, 0);
});

const seedButton = (view, label) => view.element.all((element) => element.attrs["aria-label"] === label)[0];

test("♻ takes the picked image's seed, 랜덤 puts -1 back", async () => {
  const view = await listedPanel({ seed: 5 }, { pickedSeed: async () => 1234 });
  assert.equal(seedButton(view, "현재 이미지 시드 가져오기").text, "♻");
  seedButton(view, "현재 이미지 시드 가져오기").fire("click");
  await flush();
  assert.equal(view.state.seed, 1234);
  seedButton(view, "랜덤 시드").fire("click");
  assert.equal(view.state.seed, -1);
});

test("♻ without a picked image leaves the seed", async () => {
  const view = await listedPanel({ seed: 5 }, { pickedSeed: async () => null });
  seedButton(view, "현재 이미지 시드 가져오기").fire("click");
  await flush();
  assert.equal(view.state.seed, 5);
});

test("a typed seed counts before the box is left: Ctrl+Enter comes first", async () => {
  const view = await listedPanel();
  const box = rowLabelled(view, "시드").byClass("ais-num")[0];
  box.value = "777";
  box.fire("input");
  assert.equal(view.state.seed, 777);
});

test("SAM3's prompt takes two lines and says its grammar", async () => {
  const view = await listedPanel({ sam3: { on: true } });
  const box = postNode(view).byClass("ais-negative")[0];
  assert.equal(box.tag, "textarea");
  box.value = "eyes, hair / hand";
  box.fire("input");
  assert.equal(view.state.sam3.prompt, "eyes, hair / hand");
  box.value = "";
  box.fire("input");
  assert.equal(view.state.sam3.prompt, "face");
  assert.match(postNode(view).byClass("ais-note")[0].text, /쉼표.*슬래시.*eyes, hair \/ hand/);
});

test("the gallery's SAM3 settings are reachable with the switch off", async () => {
  const view = await listedPanel();
  assert.equal(rowLabelled(view, "찾을 것"), undefined);
  caret(view, "SAM3").fire("click");
  await flush();
  assert.ok(rowLabelled(view, "찾을 것"));
});

test("a SAM3 field writes the sam3 section, clamped by the table", async () => {
  const view = await listedPanel({ sam3: { on: true } });
  caret(view, "SAM3").fire("click");
  await flush();
  section(view, "찾기").fire("click");
  await flush();
  const threshold = field(view, "threshold");
  threshold.value = "7";
  threshold.fire("change");
  assert.equal(view.state.sam3.threshold, 1);
  assert.equal(threshold.value, "1");
  assert.equal(JSON.parse(view.widget.value).sam3.threshold, 1);
});

test("a section with a changed value is lit", async () => {
  const view = await listedPanel({ sam3: { threshold: 0.2 } });
  caret(view, "SAM3").fire("click");
  await flush();
  assert.match(section(view, "찾기").className, /lit/);
  assert.doesNotMatch(section(view, "마스크").className, /lit/);
});

test("the sampling picks say what empty means and list the route's items", async () => {
  const view = await listedPanel();
  caret(view, "SAM3").fire("click");
  await flush();
  section(view, "샘플링").fire("click");
  await flush();
  const labels = field(view, "sampler").children.map((option) => option.text);
  assert.deepEqual(labels, ["메인과 같음", "euler", "heun"]);
  const sampler = field(view, "sampler");
  sampler.value = "heun";
  sampler.fire("change");
  assert.equal(view.state.sam3.sampler, "heun");
});

test("file picks with objects show the name and store the path", async () => {
  const view = await listedPanel();
  caret(view, "SAM3").fire("click");
  await flush();
  section(view, "얼굴").fire("click");
  await flush();
  assert.deepEqual(field(view, "face_crop_model").children.map((option) => [option.value, option.text]),
    [["", "[얼굴 모델]"], ["C:/face_yolov8n.pt", "face_yolov8n.pt"]]);
});

test("the upscale caret shows Scale by's row or Scale to's rows", async () => {
  const view = await listedPanel({ upscale: { on: true } });
  caret(view, "업스케일").fire("click");
  await flush();
  assert.ok(rowLabelled(view, "최대 변"));
  const mode = rowLabelled(view, "방식").all((element) => element.tag === "select")[0];
  mode.value = "Scale to";
  mode.fire("change");
  await flush();
  assert.equal(view.state.upscale.mode, "Scale to");
  assert.equal(rowLabelled(view, "최대 변"), undefined);
  assert.equal(postNode(view).byClass("ais-row").find((row) => row.byClass("ais-label")[0]?.text === "배율"), undefined);
  const pick = postNode(view).byClass("ais-row").find((row) => row.byClass("ais-label")[0]?.text === "크기");
  assert.ok(pick);
  const before = [view.state.upscale.width, view.state.upscale.height];
  view.state.upscale.width = 2048;
  pick.byClass("ais-mini")[0].fire("click");
  assert.deepEqual([view.state.upscale.width, view.state.upscale.height], [before[1], 2048]);
});

test("picking an upscaler sets the factor in its name, and leaves latent methods out", async () => {
  const view = await listedPanel({ upscale: { on: true } });
  const model = postNode(view).all((element) => element.tag === "select")[0];
  assert.deepEqual(model.children.map((option) => option.text), ["[업스케일 모델]", "Lanczos", "4x-UltraSharp.pth", "2x.pth"]);
  model.value = "4x-UltraSharp.pth";
  model.fire("change");
  assert.deepEqual([view.state.upscale.model, view.state.upscale.scale], ["4x-UltraSharp.pth", 4]);
  const again = postNode(view).all((element) => element.tag === "select")[0];
  again.value = "Lanczos";
  again.fire("change");
  assert.deepEqual([view.state.upscale.model, view.state.upscale.scale], ["Lanczos", 4]);
});

test("the Hires card shows its result size and follows the size boxes without a redraw", async () => {
  const view = await listedPanel({ width: 832, height: 1216, hires: { scale: 1.5 } });
  const size = hiresNode(view).byClass("ais-hires-size")[0];
  assert.equal(size.text, "1280 × 1856");
  const width = rowLabelled(view, "폭").byClass("ais-num")[0];
  width.value = "1024";
  width.fire("change");
  assert.equal(view.state.hires.resize_width, 1024);
  assert.equal(hiresNode(view).byClass("ais-hires-size")[0], size);
  assert.equal(size.text, "1024 × 1472");
});

test("the Hires card writes its new fields", async () => {
  const view = await listedPanel();
  const scheduler = hiresNode(view).byClass("ais-row").find((row) => row.byClass("ais-label")[0]?.text === "스케줄러");
  const pick = scheduler.all((element) => element.tag === "select")[0];
  pick.value = "karras";
  pick.fire("change");
  assert.equal(view.state.hires.scheduler, "karras");
  const prompt = hiresNode(view).byClass("ais-negative")[0];
  prompt.value = "detail";
  prompt.fire("input");
  assert.equal(view.state.hires.prompt, "detail");
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
/** The prompt card's small boxes in order: 선행 고정 ... 작가, 후행 고정, 네거티브. */
const promptBoxes = (view) => view.element.byClass("ais-negative").slice(0, 7).map((box) => Object.assign(box, { getBoundingClientRect: boxRect }));

test("autocomplete: typing in the prompt asks for tags and still saves the text; the negative box is wired too", () => withBody(async (body) => {
  const asked = [];
  const view = panelWithTags(async (query, limit) => { asked.push([query, limit]); return [{ name: "long_hair", category: 0, count: 5 }]; });
  await later(5); // the choice lists arrive and redraw the panel once; type into the boxes that stay
  const [prompt] = boxes(view, "ais-prompt");
  const negative = promptBoxes(view)[6];
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

test("the prompt card lists its boxes in the order they join, each with its name", () => withBody(async () => {
  const view = panelWithTags(async () => []);
  await later(5);
  const card = view.element.byClass("ais-prompt")[0].parent.parent.parent;
  const names = card.byClass("ais-label").map((label) => label.text);
  assert.deepEqual(names, ["선행 고정", "인물 수", "캐릭터", "작품", "작가", "프롬프트", "후행 고정", "네거티브"]);
  const boxes = card.all((element) => element.tag === "textarea");
  assert.deepEqual(boxes.map((box) => box.attrs["aria-label"]), names);
  view.dispose();
}));

test("every prompt box is saved and autocompleted like the prompt", () => withBody(async () => {
  const asked = [];
  const view = panelWithTags(async (query) => { asked.push(query); return []; });
  await later(5);
  const keys = ["prompt_prefix", "prompt_char_count", "prompt_character", "prompt_copyright", "prompt_artist", "prompt_suffix"];
  const boxes = promptBoxes(view);
  for (const [index, key] of keys.entries()) {
    boxes[index].value = `tag${index}`;
    boxes[index].fire("input");
    assert.equal(view.state[key], `tag${index}`);
    await later(220);
  }
  assert.deepEqual(asked, keys.map((_, index) => `tag${index}`));
  view.dispose();
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

test("a family greys the extensions it cannot run and turns them off for the run", () => {
  const ext = withDefaults({}).ext;
  for (const key of ["guidance", "skim", "vae_2x"]) ext[key].on = true;
  dropBlockedExt(ext, null, "moss");
  assert.deepEqual([ext.guidance.on, ext.skim.on, ext.vae_2x.on], [false, true, false]);
  const krea2 = withDefaults({}).ext;
  for (const key of ["guidance", "skim", "vae_2x"]) krea2[key].on = true;
  dropBlockedExt(krea2, null, "krea2");
  assert.deepEqual([krea2.guidance.on, krea2.skim.on, krea2.vae_2x.on], [false, true, true]);
  const anima = withDefaults({}).ext;
  anima.guidance.on = true;
  dropBlockedExt(anima, null);
  assert.equal(anima.guidance.on, true);
});

function referenceCard(reference = {}, { family = "anima", hasRole = () => true } = {}) {
  const state = withDefaults({ reference: { on: true, image: "one.png", ...reference } });
  const sets = [];
  const view = {
    set: (mutate) => { mutate(state); sets.push(state.reference); },
    hasRole,
    family,
    memory: { open: new Set(["reference"]), lists: {} },
    rerender: () => {},
    loras: ["snapref_v2a_pre2.safetensors", "snapref_v2a_pre2__40b.safetensors"],
    image: { url: (value) => `/one/${value}`, upload: async () => ({ name: "up1.png" }) },
    image2: { url: (value) => `/two/${value}`, upload: async () => ({ name: "up2.png" }) },
  };
  return { card: renderReference(state, view), state, sets };
}

const labelsOf = (card) => card.byClass("ais-label").map((label) => label.text);
const imageBoxes = (card) => card.byClass("ais-i2i-box");

test("충실도 shows with SnapRef's guidance off: the node ignores it with guidance on", () => {
  assert.ok(!labelsOf(referenceCard({ snapref: { guidance_on: true } }).card).includes("충실도"));
  assert.ok(labelsOf(referenceCard({ snapref: { guidance_on: false } }).card).includes("충실도"));
});

test("SnapRef has a second picture box, Ref Adapter and Krea2 do not", () => {
  assert.equal(imageBoxes(referenceCard().card).length, 2);
  assert.equal(imageBoxes(referenceCard({ method: "ref_adapter" }).card).length, 1);
  assert.equal(imageBoxes(referenceCard({}, { family: "krea2" }).card).length, 1);
});

test("a workflow without the second image node has one box", () => {
  const { card } = referenceCard({}, { hasRole: (role) => role !== "reference_image_2" });
  assert.equal(imageBoxes(card).length, 1);
});

test("the weights show once a second picture is in", () => {
  assert.ok(!labelsOf(referenceCard().card).includes("비중 2"));
  const labels = labelsOf(referenceCard({ image_2: "two.png" }).card);
  assert.ok(labels.includes("비중 1") && labels.includes("비중 2"));
});

test("the second box shows its own picture and clears without touching the first", () => {
  const { card, state } = referenceCard({ image_2: "two.png" });
  assert.equal(imageBoxes(card)[1].all((element) => element.tag === "img")[0].attrs.src, "/two/two.png");
  card.byClass("ais-ref-clear")[0].fire("click");
  assert.equal(state.reference.image_2, "");
  assert.equal(state.reference.image, "one.png");
});

test("a reference runs on Anima and Krea2, MOSS has none", () => {
  assert.equal(referenceBlock("anima", "snapref", null), "");
  assert.equal(referenceBlock("krea2", "snapref", null), "");
  assert.match(referenceBlock("moss", "snapref", null), /MOSS/);
});

function pressQuadviewTrigger(changes) {
  const state = withDefaults({ unet: "krea2_raw.safetensors", reference: { on: true, image: "a.png", krea2: { file: "krea2/QuadView_krea2_v1.safetensors" } }, ...changes });
  const view = { set: (mutate) => mutate(state), hasRole: () => true, family: "krea2", memory: { open: new Set(["reference"]), lists: {} }, rerender() {}, loras: [], image: { url: () => "" }, image2: { url: () => "" } };
  renderReference(state, view).all((element) => element.tag === "button" && element.text === "트리거 넣기")[0].fire("click");
  return state;
}

test("트리거 넣기 adds the caption once, wherever the prompt card already holds it", () => {
  const caption = pressQuadviewTrigger({ prompt: "" }).prompt;
  assert.ok(caption.length > 20);
  assert.equal(pressQuadviewTrigger({ prompt: "1girl" }).prompt, `${caption} 1girl`);
  for (const key of ["prompt", "prompt_prefix", "prompt_artist", "prompt_suffix"]) {
    const given = { prompt: "1girl", [key]: key === "prompt" ? `${caption}, 1girl` : caption };
    assert.equal(pressQuadviewTrigger(given).prompt, given.prompt, key);
  }
});
