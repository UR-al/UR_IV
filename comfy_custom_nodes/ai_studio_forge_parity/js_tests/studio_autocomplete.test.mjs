// node:test for the prompt tag autocomplete (web/ai_studio/studio_autocomplete.js) with a tiny fake document (no DOM in node).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  Suggester, caretPoint, createAutocomplete, formatCount, formatTag, insertTag, loraQuery, matchLoras, narrow, tagCandidate,
  tokenAtCaret,
} from "../web/ai_studio/studio_autocomplete.js";

// ----------------------------------------------------------------------------------------------- pure text helpers

const tokenOf = (text, caret = text.length) => tokenAtCaret(text, caret);

test("tokenAtCaret: the tag being typed, bounded by commas and newlines", () => {
  assert.deepEqual(tokenOf("1girl, lo"), { start: 7, end: 9, query: "lo" });
  assert.deepEqual(tokenOf("lo"), { start: 0, end: 2, query: "lo" }); // first token
  assert.deepEqual(tokenOf("1girl,lo"), { start: 6, end: 8, query: "lo" });
  assert.deepEqual(tokenOf("1girl, solo\nlong ha"), { start: 12, end: 19, query: "long ha" });
  assert.deepEqual(tokenOf(""), { start: 0, end: 0, query: "" });
  assert.deepEqual(tokenOf("1girl, "), { start: 7, end: 7, query: "" });
  assert.deepEqual(tokenOf("1girl, lo", 999), { start: 7, end: 9, query: "lo" }); // caret beyond the text
});

test("tokenAtCaret: inside a weight group the ( [ and the :1.2) are not part of the tag", () => {
  assert.deepEqual(tokenOf("(blue_hair:1.2)", 7), { start: 1, end: 10, query: "blue_h" });
  assert.deepEqual(tokenOf("1girl, (bl"), { start: 8, end: 10, query: "bl" });
  assert.deepEqual(tokenOf("[bl"), { start: 1, end: 3, query: "bl" });
  assert.deepEqual(tokenOf("((bl"), { start: 2, end: 4, query: "bl" });
  assert.deepEqual(tokenOf("(blue_hair)", 5), { start: 1, end: 10, query: "blue" });
  // a caret in the weight is not editing the tag
  assert.deepEqual(tokenOf("(blue_hair:1.2)", 13), { start: 13, end: 13, query: "" });
  // without an opening bracket nothing is stripped: a name may end in ")" or look like ":3"
  assert.deepEqual(tokenOf("ganyu_(genshin_impact)"), { start: 0, end: 22, query: "ganyu_(genshin_impact)" });
  assert.deepEqual(tokenOf(":3"), { start: 0, end: 2, query: ":3" });
});

test("tokenAtCaret: middle of a tag replaces the whole word, spaces around are kept out", () => {
  assert.deepEqual(tokenOf("1girl,   long_h  , solo", 15), { start: 9, end: 15, query: "long_h" });
  assert.deepEqual(tokenOf("1girl, long_hair, solo", 13), { start: 7, end: 16, query: "long_h" });
  assert.deepEqual(tokenOf("lo   ", 2), { start: 0, end: 2, query: "lo" });
});

test("formatTag: underscores become spaces (not in emoticons), parentheses are escaped", () => {
  assert.equal(formatTag("long_hair"), "long hair");
  assert.equal(formatTag("ganyu_(genshin_impact)"), "ganyu \\(genshin impact\\)");
  for (const emoticon of ["^_^", "o_o", ">_<", "0_0", "x_x", "+_+"]) assert.equal(formatTag(emoticon), emoticon);
  assert.equal(formatTag("v_v"), "v_v");
  assert.equal(formatTag("sleeves_past_wrists"), "sleeves past wrists");
  assert.equal(formatTag(":)"), ":\\)");
});

test("insertTag: replaces the token and adds ', ' at the end of the text", () => {
  const text = "1girl, lo";
  assert.deepEqual(insertTag(text, tokenOf(text), "long_hair"), { text: "1girl, long hair, ", caret: 18 });
  assert.deepEqual(insertTag("lo", tokenOf("lo"), "long_hair"), { text: "long hair, ", caret: 11 });
  assert.deepEqual(insertTag("a,lo", tokenOf("a,lo"), "long_hair"), { text: "a, long hair, ", caret: 14 }); // space after the comma
  assert.deepEqual(insertTag("lo  ", tokenOf("lo  ", 2), "long_hair"), { text: "long hair, ", caret: 11 });
});

test("insertTag: escapes parentheses and keeps the rest of the prompt", () => {
  const text = "1girl, ganyu";
  assert.equal(insertTag(text, tokenOf(text), "ganyu_(genshin_impact)").text, "1girl, ganyu \\(genshin impact\\), ");
  // in the middle: the comma after it is already there, so none is added and the caret sits right after the tag
  const middle = "1girl, lo, solo";
  assert.deepEqual(insertTag(middle, tokenOf(middle, 9), "long_hair"), { text: "1girl, long hair, solo", caret: 16 });
  const lines = "lo\nsolo";
  assert.deepEqual(insertTag(lines, tokenOf(lines, 2), "long_hair"), { text: "long hair\nsolo", caret: 9 });
});

test("insertTag: a weight around the tag survives", () => {
  const text = "(blue_h:1.2)";
  const result = insertTag(text, tokenOf(text, 7), "blue_hair");
  assert.deepEqual(result, { text: "(blue hair:1.2)", caret: 10 });
  assert.equal(insertTag("[bl]", tokenOf("[bl]", 3), "blue_hair").text, "[blue hair]");
});

test("insertTag: verbatim inserts the text as given", () => {
  const text = "1girl, <lora:mu";
  assert.deepEqual(insertTag(text, tokenOf(text), "<lora:muruko_e6:1>", { verbatim: true }), { text: "1girl, <lora:muruko_e6:1>, ", caret: 27 });
  const closed = "<lora:mu>, solo";
  assert.equal(insertTag(closed, tokenOf(closed, 8), "<lora:muruko:1>", { verbatim: true }).text, "<lora:muruko:1>, solo");
});

// ----------------------------------------------------------------------------------------------- LoRA branch

const LORAS = [
  "묵밥 (@Bjunbae)/ep10/bjunbae_e10.safetensors",
  "meow/meow_e6.safetensors",
  "meow/old/meow_e6.safetensors", // same stem in another folder
  "enaa97 V2/enaa97_v2-e8.safetensors",
  "plain.safetensors",
];

test("loraQuery: only text after '<lora:' counts", () => {
  assert.equal(loraQuery("<lora:"), "");
  assert.equal(loraQuery("<lora:me"), "me");
  assert.equal(loraQuery("<LoRA:me"), "me");
  assert.equal(loraQuery("<lora:me:0.8"), null); // already past the name
  assert.equal(loraQuery("<lora:me>"), null);
  assert.equal(loraQuery("lora:me"), null);
  assert.equal(loraQuery("long"), null);
});

test("matchLoras: by stem, stem prefix before contains before folder, deduplicated, inserted as <lora:stem:1>", () => {
  const stems = (query) => matchLoras(LORAS, query).map((item) => item.name);
  assert.deepEqual(stems("meow"), ["meow_e6"]); // one entry for the two folders
  assert.deepEqual(stems("e6"), ["meow_e6"]);
  assert.deepEqual(stems("ENAA"), ["enaa97_v2-e8"]);
  assert.deepEqual(stems("묵밥"), ["bjunbae_e10"]); // folder match
  assert.deepEqual(stems("e"), ["enaa97_v2-e8", "bjunbae_e10", "meow_e6"]); // prefix first, then contains by name
  assert.equal(stems("").length, 4); // right after "<lora:" every LoRA is offered
  assert.deepEqual(stems("zzz"), []);
  const item = matchLoras(LORAS, "bjun")[0];
  assert.equal(item.insert, "<lora:bjunbae_e10:1>");
  assert.equal(item.verbatim, true);
  assert.equal(item.hint, "묵밥 (@Bjunbae)/ep10");
  assert.equal(matchLoras(LORAS, "", 2).length, 2);
  assert.deepEqual(matchLoras(undefined, "a"), []);
});

// ----------------------------------------------------------------------------------------------- candidates

test("tagCandidate / formatCount / narrow", () => {
  const tag = tagCandidate({ name: "1girl", category: 0, count: 8359716, alias: "1girls", ko: "소녀" });
  assert.equal(tag.hint, "← 1girls · 소녀");
  assert.equal(tag.insert, "1girl");
  assert.equal(tag.verbatim, false);
  assert.equal(tagCandidate({ name: "x", category: "4", count: "5" }).category, 4);
  assert.deepEqual([8359716, 56789, 1234, 999, 0, 999999, 12].map(formatCount), ["8.4M", "57k", "1.2k", "999", "0", "1M", "12"]);

  const rows = [
    tagCandidate({ name: "long_hair", category: 0, count: 5, ko: "긴 머리" }),
    tagCandidate({ name: "blue_hair", category: 0, count: 4 }),
    tagCandidate({ name: "ponytail", category: 0, count: 3, alias: "long_ponytail" }),
  ];
  assert.deepEqual(narrow(rows, "long_h").map((row) => row.name), ["long_hair"]);
  assert.deepEqual(narrow(rows, "long").map((row) => row.name), ["long_hair", "ponytail"]); // alias counts
  assert.deepEqual(narrow(rows, "긴").map((row) => row.name), ["long_hair"]);
  assert.deepEqual(narrow(rows, ""), []);
  assert.deepEqual(narrow(rows, "<lora:x"), []);
  const loras = matchLoras(LORAS, "");
  assert.deepEqual(narrow(loras, "<lora:meow").map((row) => row.name), ["meow_e6"]);
  assert.deepEqual(narrow(loras, "meow"), []);
});

// ----------------------------------------------------------------------------------------------- requests

function fakeTimers() {
  const timers = new Map();
  let next = 1;
  return {
    setTimer: (run, ms) => { timers.set(next, { run, ms }); return next++; },
    clearTimer: (id) => { timers.delete(id); },
    pending: () => [...timers.values()],
    fire() { const [id, timer] = [...timers].pop(); timers.delete(id); timer.run(); },
  };
}
const tick = () => new Promise((done) => setImmediate(done));

function deferredFetch() {
  const calls = [];
  const fetchTags = (query, limit) => new Promise((resolve, reject) => calls.push({ query, limit, resolve, reject }));
  return { calls, fetchTags };
}

test("Suggester: 150ms debounce, one request for a burst of typing", async () => {
  const timers = fakeTimers();
  const { calls, fetchTags } = deferredFetch();
  const answers = [];
  const suggester = new Suggester({ fetchTags, onItems: (rows, query) => answers.push([query, rows.map((row) => row.name)]), ...timers });
  suggester.ask("lo");
  suggester.ask("lon");
  suggester.ask("long");
  assert.equal(timers.pending().length, 1);
  assert.equal(timers.pending()[0].ms, 150);
  timers.fire();
  await tick();
  assert.deepEqual(calls.map((call) => [call.query, call.limit]), [["long", 12]]);
  calls[0].resolve([{ name: "long_hair", category: 0, count: 5 }]);
  await tick();
  assert.deepEqual(answers, [["long", ["long_hair"]]]);
});

test("Suggester: an answer for an older query is dropped", async () => {
  const timers = fakeTimers();
  const { calls, fetchTags } = deferredFetch();
  const answers = [];
  const suggester = new Suggester({ fetchTags, onItems: (rows, query) => answers.push(query), ...timers });
  suggester.ask("lo");
  timers.fire();
  await tick();
  suggester.ask("l"); // too short: nothing is asked, the list is emptied
  suggester.ask("lon");
  timers.fire();
  await tick();
  assert.equal(calls.length, 2);
  calls[1].resolve([{ name: "long_hair", category: 0, count: 1 }]); // the newer answer arrives first
  await tick();
  calls[0].resolve([{ name: "lollipop", category: 0, count: 9 }]); // the stale one afterwards
  await tick();
  assert.deepEqual(answers, ["l", "lon"]);
});

test("Suggester: cancel drops a pending timer and an answer on its way", async () => {
  const timers = fakeTimers();
  const { calls, fetchTags } = deferredFetch();
  const answers = [];
  const suggester = new Suggester({ fetchTags, onItems: (rows) => answers.push(rows), ...timers });
  suggester.ask("lo");
  suggester.cancel();
  assert.equal(timers.pending().length, 0);
  suggester.ask("lo");
  timers.fire();
  await tick();
  suggester.cancel();
  calls[0].resolve([{ name: "long_hair", category: 0, count: 1 }]);
  await tick();
  assert.deepEqual(answers, []);
});

test("Suggester: minimum length (2, or 1 for Hangul), other extra-network tags, failures", async () => {
  const timers = fakeTimers();
  const { calls, fetchTags } = deferredFetch();
  const answers = [];
  const suggester = new Suggester({ fetchTags, onItems: (rows, query) => answers.push([query, rows.length]), ...timers });
  suggester.ask("a");
  suggester.ask("<hypernet:x");
  suggester.ask("   ");
  assert.equal(timers.pending().length, 0);
  assert.deepEqual(answers, [["a", 0], ["<hypernet:x", 0], ["   ", 0]]);
  suggester.ask("장");
  assert.equal(timers.pending().length, 1);
  timers.fire();
  await tick();
  assert.equal(calls[0].query, "장");
  calls[0].reject(new Error("offline"));
  await tick();
  assert.deepEqual(answers.at(-1), ["장", 0]); // a failed request just shows nothing
});

test("Suggester: a LoRA query is answered at once from the panel's list, without the server", () => {
  const timers = fakeTimers();
  const { calls, fetchTags } = deferredFetch();
  const answers = [];
  const suggester = new Suggester({ fetchTags, loras: () => LORAS, onItems: (rows, query) => answers.push([query, rows.map((row) => row.insert)]), ...timers });
  suggester.ask("<lora:bjun");
  assert.deepEqual(answers, [["<lora:bjun", ["<lora:bjunbae_e10:1>"]]]);
  assert.equal(calls.length, 0);
  assert.equal(timers.pending().length, 0);
});

// ----------------------------------------------------------------------------------------------- popup and keys (fake DOM)

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.textContent = "";
    this.parent = null;
    this.hidden = false;
  }
  append(...nodes) { nodes.forEach((node) => { node.parent = this; this.children.push(node); }); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== fn); }
  contains(node) { for (let at = node; at; at = at.parent) if (at === this) return true; return false; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this); this.parent = null; }
  fire(type, event = {}) {
    const e = { type, target: this, defaultPrevented: false, stopped: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; }, ...event };
    for (const fn of this.listeners[type] || []) fn(e);
    return e;
  }
}

class FakeTextarea extends FakeElement {
  constructor(doc) {
    super("textarea");
    this.doc = doc;
    this.value = "";
    this.selectionStart = 0;
  }
  setSelectionRange(start) { this.selectionStart = start; }
  dispatchEvent(event) { this.fire(event.type, { isEvent: true }); return true; }
  getBoundingClientRect() { return { left: 100, top: 50, bottom: 150, right: 400, width: 300, height: 100 }; }
  type(text) { this.value = text; this.selectionStart = text.length; return this.fire("input"); }
  press(key, extra = {}) { return this.fire("keydown", { key, ...extra }); }
}

function setup(options = {}) {
  const doc = new FakeElement("document");
  doc.body = new FakeElement("body");
  doc.createElement = (tag) => new FakeElement(tag);
  const win = { Event: class { constructor(type) { this.type = type; } }, innerWidth: 1200, innerHeight: 800 };
  const timers = fakeTimers();
  const { calls, fetchTags } = deferredFetch();
  const autocomplete = createAutocomplete({ fetchTags, loras: () => LORAS, doc, win, ...timers, ...options });
  const box = new FakeTextarea(doc);
  const inputs = [];
  box.addEventListener("input", (event) => { if (event.isEvent) inputs.push(box.value); });
  autocomplete.attach(box);
  const popup = () => doc.body.children[0];
  const rows = () => popup().children.map((row) => row.children[0].textContent);
  const answer = async (items, call = calls.at(-1)) => { call.resolve(items); await tick(); };
  const typeAndShow = async (text, items) => {
    box.type(text);
    timers.fire();
    await tick();
    await answer(items);
  };
  return { doc, win, box, autocomplete, timers, calls, popup, rows, answer, typeAndShow, inputs };
}

const LONG = [
  { name: "long_hair", category: 0, count: 6168790, ko: "긴 머리" },
  { name: "long_sleeves", category: 0, count: 2000000 },
  { name: "longing", category: 0, count: 50 },
];

test("popup: typing asks after the debounce and shows name, label, count; the first row is selected", async () => {
  const t = setup();
  await t.typeAndShow("1girl, lo", LONG);
  assert.equal(t.autocomplete.isOpen(), true);
  assert.equal(t.popup().parent, t.doc.body); // on document.body, not inside the node
  assert.equal(Number(t.popup().style.zIndex) > 1000, true);
  assert.deepEqual(t.rows(), ["long_hair", "long_sleeves", "longing"]);
  assert.equal(t.popup().children[0].children[1].textContent, "긴 머리");
  assert.equal(t.popup().children[0].children[2].textContent, "6.2M");
  assert.match(t.popup().children[0].className, /sel/);
  assert.match(t.popup().children[0].children[0].className, /ais-ac-c0/);
});

test("popup: Tab and Enter accept the selected row, ArrowDown/Up move (wrapping), Esc closes", async () => {
  for (const key of ["Tab", "Enter"]) {
    const t = setup();
    await t.typeAndShow("1girl, lo", LONG);
    const down = t.box.press("ArrowDown");
    assert.equal(down.defaultPrevented, true);
    assert.match(t.popup().children[1].className, /sel/);
    t.box.press("ArrowUp");
    t.box.press("ArrowUp"); // wraps to the last row
    assert.match(t.popup().children[2].className, /sel/);
    const accepted = t.box.press(key);
    assert.equal(accepted.defaultPrevented, true);
    assert.equal(t.box.value, "1girl, longing, ");
    assert.equal(t.box.selectionStart, t.box.value.length);
    assert.deepEqual(t.inputs, ["1girl, longing, "]); // the panel hears about the change
    assert.equal(t.autocomplete.isOpen(), false);
  }
  const t = setup();
  await t.typeAndShow("lo", LONG);
  const esc = t.box.press("Escape");
  assert.equal(esc.defaultPrevented && esc.stopped, true);
  assert.equal(t.autocomplete.isOpen(), false);
  assert.equal(t.box.value, "lo");
});

test("keys keep their normal meaning when the popup is closed (Enter, Tab, Ctrl+Enter, Esc)", async () => {
  const t = setup();
  for (const key of ["Enter", "Tab", "Escape", "ArrowDown"]) {
    const event = t.box.press(key);
    assert.equal(event.defaultPrevented || event.stopped, false, key);
  }
  t.box.type("a"); // too short: still closed
  assert.equal(t.box.press("Enter").defaultPrevented, false);
  t.box.type("lo"); // asked but no answer yet
  assert.equal(t.box.press("Enter").defaultPrevented, false);
  t.timers.fire();
  await tick();
  await t.answer(LONG);
  assert.equal(t.autocomplete.isOpen(), true);
  // open: Ctrl+Enter / Shift+Enter / Alt+Tab are not ours (Ctrl+Enter must still queue a generation)
  for (const extra of [{ ctrlKey: true }, { metaKey: true }, { shiftKey: true }, { altKey: true }]) {
    const event = t.box.press("Enter", extra);
    assert.equal(event.defaultPrevented || event.stopped, false, JSON.stringify(extra));
    assert.equal(t.box.value, "lo");
  }
  assert.equal(t.box.press("Tab", { shiftKey: true }).defaultPrevented, false);
  assert.equal(t.autocomplete.isOpen(), true);
});

test("Enter while an IME is composing does not accept", async () => {
  const t = setup();
  await t.typeAndShow("장", [{ name: "long_hair", category: 0, count: 1, ko: "장발" }]);
  assert.equal(t.autocomplete.isOpen(), true);
  assert.equal(t.box.press("Enter", { isComposing: true }).defaultPrevented, false);
  assert.equal(t.box.press("Enter", { keyCode: 229 }).defaultPrevented, false);
  assert.equal(t.box.value, "장");
  assert.equal(t.box.press("Enter").defaultPrevented, true);
  assert.equal(t.box.value, "long hair, ");
});

test("popup closes on blur, click in the box, caret keys, and a click outside (not inside the popup)", async () => {
  const t = setup();
  const open = async () => { await t.typeAndShow("lo", LONG); assert.equal(t.autocomplete.isOpen(), true); };
  await open();
  t.box.fire("blur");
  assert.equal(t.autocomplete.isOpen(), false);
  await open();
  t.box.fire("click");
  assert.equal(t.autocomplete.isOpen(), false);
  for (const key of ["ArrowLeft", "ArrowRight", "Home", "End"]) {
    await open();
    t.box.press(key);
    assert.equal(t.autocomplete.isOpen(), false, key);
  }
  await open();
  t.doc.fire("pointerdown", { target: t.popup().children[0] }); // inside the popup
  assert.equal(t.autocomplete.isOpen(), true);
  t.doc.fire("pointerdown", { target: new FakeElement("div") }); // elsewhere
  assert.equal(t.autocomplete.isOpen(), false);
  assert.equal((t.doc.listeners.pointerdown || []).length, 0); // the document listener is gone with the popup
});

test("a late or stale answer never opens the popup or lets Tab take another tag", async () => {
  const t = setup();
  t.box.type("lo");
  t.timers.fire();
  await tick();
  const first = t.calls[0];
  t.box.fire("blur"); // closed before the answer arrived
  await t.answer(LONG, first);
  assert.equal(t.autocomplete.isOpen(), false);

  await t.typeAndShow("lo", LONG);
  // typing on while the popup shows old rows: rows that do not fit the new query go at once
  t.box.type("lon");
  assert.deepEqual(t.rows(), ["long_hair", "long_sleeves", "longing"]);
  t.box.type("long_h");
  assert.deepEqual(t.rows(), ["long_hair"]);
  t.box.type("blue");
  assert.equal(t.autocomplete.isOpen(), false);
  assert.equal(t.box.press("Tab").defaultPrevented, false);
  t.timers.fire();
  await tick();
  await t.answer([{ name: "blue_hair", category: 0, count: 3 }]);
  assert.deepEqual(t.rows(), ["blue_hair"]);

  // the text changed without an input event (should not happen) -> accept refuses and closes
  t.box.value = "changed";
  const tab = t.box.press("Tab");
  assert.equal(tab.defaultPrevented, false);
  assert.equal(t.box.value, "changed");
  assert.equal(t.autocomplete.isOpen(), false);
});

test("accepting inside the prompt edits only the tag at the caret", async () => {
  const t = setup();
  t.box.type("1girl, lo, solo");
  t.box.selectionStart = 9;
  t.box.fire("input");
  t.timers.fire();
  await tick();
  assert.equal(t.calls.at(-1).query, "lo");
  await t.answer(LONG);
  t.box.press("Enter");
  assert.equal(t.box.value, "1girl, long hair, solo");
  assert.equal(t.box.selectionStart, 16);
});

test("clicking a row accepts it and keeps the focus in the box", async () => {
  const t = setup();
  await t.typeAndShow("lo", LONG);
  const row = t.popup().children[1];
  const mousedown = row.fire("mousedown");
  assert.equal(mousedown.defaultPrevented, true);
  assert.equal(t.box.value, "long sleeves, ");
  assert.equal(t.autocomplete.isOpen(), false);
  const background = new FakeElement("div");
  assert.equal(t.popup().fire("mousedown").defaultPrevented, true); // clicking the popup's padding does not blur the box
  assert.equal(background.parent, null);
});

test("hovering a row selects it", async () => {
  const t = setup();
  await t.typeAndShow("lo", LONG);
  t.popup().children[2].fire("mousemove");
  assert.match(t.popup().children[2].className, /sel/);
  assert.doesNotMatch(t.popup().children[0].className, /sel/);
  t.box.press("Enter");
  assert.equal(t.box.value, "longing, ");
});

test("after '<lora:' the panel's LoRA list is used and <lora:stem:1> is inserted (no server request)", async () => {
  const t = setup();
  t.box.type("1girl, <lora:bjun");
  assert.equal(t.timers.pending().length, 0);
  assert.equal(t.calls.length, 0);
  assert.equal(t.autocomplete.isOpen(), true);
  assert.deepEqual(t.rows(), ["bjunbae_e10"]);
  t.box.press("Tab");
  assert.equal(t.box.value, "1girl, <lora:bjunbae_e10:1>, ");
  // the whole LoRA list right after the colon, narrowed while typing on
  t.box.type("<lora:");
  assert.equal(t.rows().length, 4);
  t.box.type("<lora:meo");
  assert.deepEqual(t.rows(), ["meow_e6"]);
});

test("a pending request is dropped when the box loses its popup owner; dispose removes the popup", async () => {
  const t = setup();
  await t.typeAndShow("lo", LONG);
  t.autocomplete.dispose();
  assert.equal(t.doc.body.children.length, 0);
  assert.equal(t.autocomplete.isOpen(), false);
  t.box.type("long"); // a box rendered earlier is still wired but does nothing now
  assert.equal(t.timers.pending().length, 0);
});

test("attach tolerates a missing box or a document without a body (tests, other hosts)", () => {
  const autocomplete = createAutocomplete({ fetchTags: async () => [], doc: { createElement() {} }, win: {} });
  assert.doesNotThrow(() => autocomplete.attach(null));
  assert.doesNotThrow(() => autocomplete.attach({ addEventListener() {} }));
  assert.doesNotThrow(() => autocomplete.close());
});

test("module import has no side effects", async () => {
  const before = Object.keys(globalThis).sort().join();
  await import("../web/ai_studio/studio_autocomplete.js?again");
  assert.equal(Object.keys(globalThis).sort().join(), before);
});

// ----------------------------------------------------------------------------------------------- caret position

test("caretPoint: measures the text up to the caret in an invisible copy and maps it to the screen", () => {
  const made = [];
  const doc = {
    createElement: (tag) => {
      const element = new FakeElement(tag);
      element.offsetLeft = 0;
      element.offsetTop = 0;
      made.push(element);
      return element;
    },
    body: new FakeElement("body"),
  };
  // the mirror's marker sits after 40px of text on its second line (lineHeight 20, padding top 4)
  const origCreate = doc.createElement;
  doc.createElement = (tag) => {
    const element = origCreate(tag);
    if (tag === "span") { element.offsetLeft = 40; element.offsetTop = 24; }
    return element;
  };
  const win = { getComputedStyle: () => ({ fontSize: "13px", lineHeight: "20px", borderLeftWidth: "1px", borderTopWidth: "1px" }) };
  const box = new FakeTextarea(doc);
  box.value = "hello world";
  box.scrollLeft = 0;
  box.scrollTop = 0;
  box.offsetWidth = 300; // rect.width is 300: scale 1
  const point = caretPoint(box, 5, doc, win);
  assert.deepEqual(point, { left: 100 + 41, top: 50 + 25, bottom: 50 + 25 + 20 });
  assert.equal(made.length, 2);
  assert.equal(doc.body.children.length, 0); // the copy was removed again
  box.offsetWidth = 150; // the canvas shows the node at 2x
  assert.deepEqual(caretPoint(box, 5, doc, win), { left: 100 + 82, top: 50 + 50, bottom: 50 + 50 + 40 });
  assert.equal(caretPoint(box, 5, doc, {}), null); // cannot measure -> the popup goes under the box
});

test("without a measurable caret the popup sits under the box", async () => {
  const t = setup();
  await t.typeAndShow("lo", LONG); // the fake window has no getComputedStyle
  assert.equal(t.popup().style.left, "100px");
  assert.equal(t.popup().style.top, "152px"); // box bottom (150) + 2
});
