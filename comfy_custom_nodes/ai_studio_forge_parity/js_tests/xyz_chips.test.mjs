// node:test for the XYZ panel's value blocks (web/ai_studio/xyz_chips.js): the value box read the way xyz_values.py reads it
// (golden table shared with tests/test_xyz_values_golden.py) and the block field drawn into a tiny fake document.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { installFakeDom, mount } from "./fake_dom.mjs";
import { chipField, csvField, joinValues, knownValue, menuItems, moveValue, splitValues, stripValue } from "../web/ai_studio/xyz_chips.js";
import { optionItems } from "../web/ai_studio/studio_dom.js";

const GOLDEN = JSON.parse(readFileSync(new URL("./fixtures/xyz_values_golden.json", import.meta.url), "utf8"));

test("golden: splitValues reads a value box as xyz_values.split_values does", () => {
  assert.ok(GOLDEN.length >= 40);
  for (const row of GOLDEN) assert.deepEqual(splitValues(row.text), row.fields, JSON.stringify(row.text));
});

test("golden: joinValues writes one value a line, quoted when it needs it, and reads back the same", () => {
  const joined = GOLDEN.filter((row) => row.joined);
  assert.ok(joined.length >= 5);
  for (const row of joined) assert.equal(joinValues(row.fields), row.text);
  for (const row of GOLDEN) assert.deepEqual(splitValues(joinValues(row.fields)), row.fields, JSON.stringify(row.text));
});

test("csvField quotes a comma, a quote, outer spaces, a line break and the empty value", () => {
  assert.equal(csvField("plain.safetensors"), "plain.safetensors");
  assert.equal(csvField("a, b"), '"a, b"');
  assert.equal(csvField('say "hi"'), '"say ""hi"""');
  assert.equal(csvField(" a"), '" a"');
  assert.equal(csvField("a\nb"), '"a\nb"');
  assert.equal(csvField(""), '""');
  assert.equal(joinValues([]), "");
});

test("a value is stripped the way Python strips it, not the way trim() does: a BOM stays, U+001F and U+0085 go", () => {
  const BOM = String.fromCharCode(0xfeff);
  const US = String.fromCharCode(0x1f);
  const NEL = String.fromCharCode(0x85);
  assert.equal(stripValue(` ${BOM}euler${US}${NEL}\t`), `${BOM}euler`);
  assert.deepEqual(splitValues(`${BOM}euler`), [`${BOM}euler`]);
  assert.deepEqual(splitValues(`euler${US}, lcm${NEL}`), ["euler", "lcm"]);
  // the planner does not know "<BOM>euler": the block must be marked, not shown as a plain "euler"
  const items = optionItems(["euler", "sage"]);
  assert.equal(knownValue({ id: "sampler", kind: "choice" }, splitValues(`${BOM}euler`)[0], items), false);
  assert.equal(knownValue({ id: "attention", kind: "choice" }, `${BOM}Sage`, items), false);
  assert.equal(knownValue({ id: "sampler", kind: "choice" }, splitValues(`euler${US}`)[0], items), true);
  assert.equal(csvField(`${BOM}a`), `${BOM}a`, "no outer space for the planner: not quoted");
  assert.equal(csvField(`a${US}`), `"a${US}"`);
});

test("knownValue follows the planner: exact names, a checkpoint by a part of its name, attention ignoring case, bool words", () => {
  const items = (...names) => optionItems(names);
  const axis = (id, kind = "choice") => ({ id, kind });
  assert.equal(knownValue(axis("sampler"), "euler", items("euler", "lcm")), true);
  assert.equal(knownValue(axis("sampler"), "Euler", items("euler", "lcm")), false);
  assert.equal(knownValue(axis("sampler"), "", items("euler")), false);
  assert.equal(knownValue(axis("sampler"), "anything", null), true, "a list not read yet marks nothing");
  assert.equal(knownValue(axis("checkpoint"), "anima-3.8b", items("sub\\anima-3.8b-fp16.safetensors")), true);
  assert.equal(knownValue(axis("checkpoint"), "gone.safetensors", items("sub\\anima-3.8b-fp16.safetensors")), false);
  assert.equal(knownValue(axis("attention"), "Sage", items("launch", "sage")), true);
  assert.equal(knownValue(axis("attention"), "xformers", items("launch", "sage")), false);
  const modes = items("LyCORIS fp32 (as trained)", "Forge/Comfy fp32", "Forge/Comfy (stock)");
  assert.equal(knownValue(axis("dora_mode"), "Forge/Comfy fp32", modes), true);
  assert.equal(knownValue(axis("dora_mode"), "forge fp32", modes), true, "sam-extra's label");
  assert.equal(knownValue(axis("dora_mode"), "forge", modes), true, "the label up to its bracket");
  assert.equal(knownValue(axis("dora_mode"), "forge_fp32", modes), true, "a mode key is left to the planner");
  assert.equal(knownValue(axis("dora_mode"), "no such mode", modes), false);
  for (const word of ["True", "false", "ON", "0", "yes"]) assert.equal(knownValue(axis("sam3_enable", "bool"), word, items("True", "False")), true);
  assert.equal(knownValue(axis("sam3_enable", "bool"), "maybe", items("True", "False")), false);
});

test("menuItems lists what is not chosen yet, narrowed by every word typed (case ignored)", () => {
  const items = optionItems(["euler", "euler_ancestral", "dpmpp_2m", "DPMPP_SDE"]);
  const values = (list) => list.map((item) => item.value);
  assert.deepEqual(values(menuItems(items, ["euler"], "")), ["euler_ancestral", "dpmpp_2m", "DPMPP_SDE"]);
  assert.deepEqual(values(menuItems(items, [], "dpm")), ["dpmpp_2m", "DPMPP_SDE"]);
  assert.deepEqual(values(menuItems(items, [], " eul  anc ")), ["euler_ancestral"]);
  assert.deepEqual(values(menuItems(items, ["dpmpp_2m"], "DPM")), ["DPMPP_SDE"]);
});

test("moveValue moves one value and leaves a move off the ends alone", () => {
  assert.deepEqual(moveValue(["a", "b", "c"], 0, 2), ["b", "c", "a"]);
  assert.deepEqual(moveValue(["a", "b", "c"], 2, 1), ["a", "c", "b"]);
  assert.deepEqual(moveValue(["a", "b", "c"], 0, -1), ["a", "b", "c"]);
  assert.deepEqual(moveValue(["a", "b", "c"], 2, 3), ["a", "b", "c"]);
});

// ---------------------------------------------------------------------------------------------- the drawn field

const SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "lcm"];

function field(values, { names = SAMPLERS, loaded = true } = {}) {
  installFakeDom();
  const store = { values: [...values], writes: 0, opened: 0, names, loaded };
  const made = chipField({
    name: "X",
    scrollKey: "chips-0-sampler",
    read: () => store.values,
    write: (next) => { store.values = next; store.writes += 1; },
    list: () => ({ items: optionItems(store.names), loaded: store.loaded }),
    known: (value) => !store.loaded || store.names.includes(value),
    opened: () => { store.opened += 1; },
  });
  mount(made.el);
  const el = made.el;
  return {
    store, made, el,
    box: el.byClass("ais-chips")[0],
    search: el.byClass("ais-chip-search")[0],
    menu: el.byClass("ais-chip-menu")[0],
    chips: () => el.byClass("ais-chip"),
    texts: () => el.byClass("ais-chip-text").map((chip) => chip.textContent),
    options: () => el.byClass("ais-chip-opt").map((option) => option.textContent),
    button: (text) => el.byTag("button").find((button) => button.textContent === text),
  };
}

test("the chosen values are blocks in order, each with its ✕; a value the list does not have is marked", () => {
  const f = field(["lcm", "gone_sampler", "euler", ""]);
  assert.deepEqual(f.texts(), ["lcm", "gone_sampler", "euler", "(빈 값)"]);
  assert.deepEqual(f.chips().map((chip) => chip.classList.contains("warn")), [false, true, false, true]);
  assert.equal(f.chips()[1].attrs.title, "gone_sampler — 목록에 없음");
  assert.equal(f.chips().every((chip) => chip.byClass("ais-chip-x").length === 1), true);
  assert.equal(f.box.attrs["data-scroll"], "chips-0-sampler");
  assert.equal(f.menu.hidden, true);
});

test("an empty field says so and its 모두 지우기 is greyed; nothing is marked while the list is not read", () => {
  const f = field([]);
  assert.equal(f.chips().length, 0);
  assert.equal(f.el.byClass("ais-chip-none").length, 1);
  assert.equal(f.button("모두 지우기").disabled, true);
  const loading = field(["whatever"], { names: [], loaded: false });
  assert.equal(loading.chips()[0].classList.contains("warn"), false);
  assert.equal(loading.button("전부 넣기").disabled, true);
  loading.search.fire("click");
  assert.equal(loading.el.byClass("ais-chip-empty")[0].textContent, "목록 읽는 중…");
});

test("✕ takes one block out in place: the box is the same element, its scroll stays and the focus goes to the next ✕", () => {
  const f = field(["euler", "lcm", "dpmpp_2m"]);
  f.box.scrollTop = 40;
  const cross = f.chips()[1].byClass("ais-chip-x")[0];
  cross.focus();
  cross.fire("click");
  assert.deepEqual(f.store.values, ["euler", "dpmpp_2m"]);
  assert.deepEqual(f.texts(), ["euler", "dpmpp_2m"]);
  assert.equal(f.el.byClass("ais-chips")[0], f.box);
  assert.equal(f.box.scrollTop, 40);
  assert.equal(document.activeElement, f.chips()[1].byClass("ais-chip-x")[0], "the block that took its place");
  document.activeElement.fire("click"); // the last block
  assert.equal(document.activeElement, f.chips()[0].byClass("ais-chip-x")[0], "the one before");
  document.activeElement.fire("click");
  assert.deepEqual(f.store.values, []);
  assert.equal(document.activeElement, f.search, "no block left: the add box");
  assert.equal(f.menu.hidden, true, "focusing the add box alone does not open the list");
});

test("a click on the add box opens the list of values not chosen yet; a click on one appends it and the list stays open", () => {
  const f = field(["lcm"]);
  f.search.focus();
  f.search.fire("click");
  assert.equal(f.menu.hidden, false);
  assert.equal(f.store.opened, 1);
  assert.deepEqual(f.options(), ["euler", "euler_ancestral", "dpmpp_2m"]);
  assert.equal(f.menu.fire("mousedown").defaultPrevented, true, "a press on the list must not take the focus from the box");
  f.box.scrollHeight = 300;
  f.el.byClass("ais-chip-opt")[1].fire("click");
  assert.deepEqual(f.store.values, ["lcm", "euler_ancestral"]);
  assert.deepEqual(f.texts(), ["lcm", "euler_ancestral"]);
  assert.equal(f.box.scrollTop, 300, "the new block is shown");
  assert.equal(f.menu.hidden, false);
  assert.deepEqual(f.options(), ["euler", "dpmpp_2m"]);
  assert.ok(document.activeElement === f.search, "the focus went to the add box");
  f.search.blur();
  assert.equal(f.menu.hidden, true, "leaving the box closes the list");
});

test("typing narrows the list; ↓ ↑ move the mark, Enter adds the marked value, Esc closes, ⌫ in the empty box takes the last block", () => {
  const f = field(["lcm"]);
  f.search.focus();
  f.search.value = "eul";
  f.search.fire("input");
  assert.equal(f.menu.hidden, false);
  assert.deepEqual(f.options(), ["euler", "euler_ancestral"]);
  const marked = () => f.el.byClass("ais-chip-opt").findIndex((option) => option.classList.contains("cur"));
  assert.equal(marked(), 0);
  assert.equal(f.search.fire("keydown", { key: "ArrowDown" }).defaultPrevented, true);
  assert.equal(marked(), 1);
  f.search.fire("keydown", { key: "ArrowDown" });
  assert.equal(marked(), 1, "stops at the end");
  f.search.fire("keydown", { key: "ArrowUp" });
  assert.equal(marked(), 0);
  const enter = f.search.fire("keydown", { key: "Enter" });
  assert.equal(enter.defaultPrevented && enter.stopped, true);
  assert.deepEqual(f.store.values, ["lcm", "euler"]);
  assert.deepEqual(f.options(), ["euler_ancestral"], "the box keeps its text, so Enter again takes the next match");
  assert.equal(f.search.selected, true, "typing now starts a new search");
  const esc = f.search.fire("keydown", { key: "Escape" });
  assert.equal(esc.stopped, true);
  assert.equal(f.menu.hidden, true);
  assert.equal(f.search.fire("keydown", { key: "Escape" }).stopped, false, "a closed list leaves Esc alone");
  f.search.fire("keydown", { key: "Backspace" });
  assert.deepEqual(f.store.values, ["lcm", "euler"], "the box still has text");
  f.search.value = "";
  f.search.fire("keydown", { key: "Backspace", repeat: true });
  assert.deepEqual(f.store.values, ["lcm", "euler"], "⌫ held to empty the box stops at the blocks");
  f.search.fire("keydown", { key: "Backspace" });
  assert.deepEqual(f.store.values, ["lcm"]);
  f.search.fire("keydown", { key: "ArrowDown" });
  assert.equal(f.menu.hidden, false, "↓ opens the list");
  f.search.fire("keydown", { key: "Enter", isComposing: true });
  assert.deepEqual(f.store.values, ["lcm"], "Enter of the Korean IME is not ours");
});

test("the list says why it is empty", () => {
  const all = field(SAMPLERS);
  all.search.fire("click");
  assert.equal(all.el.byClass("ais-chip-empty")[0].textContent, "모든 값을 넣었어요");
  const none = field([]);
  none.search.value = "zzz";
  none.search.fire("input");
  assert.equal(none.el.byClass("ais-chip-empty")[0].textContent, "맞는 값이 없어요");
  const noList = field([], { names: [] });
  noList.search.fire("click");
  assert.equal(noList.el.byClass("ais-chip-empty")[0].textContent, "목록 없음");
});

test("전부 넣기 puts every value of the list in list order, 모두 지우기 empties the field", () => {
  const f = field(["lcm", "gone"]);
  f.button("전부 넣기").fire("click");
  assert.deepEqual(f.store.values, SAMPLERS);
  assert.deepEqual(f.texts(), SAMPLERS);
  assert.equal(f.button("전부 넣기").disabled, true, "nothing left to put in");
  f.button("모두 지우기").fire("click");
  assert.deepEqual(f.store.values, []);
  assert.equal(f.chips().length, 0);
  assert.equal(f.button("전부 넣기").disabled, false);
});

test("a button that greys itself hands the focus to the add box, not to the page (mouse and keyboard alike)", () => {
  const f = field(["lcm", "gone"]);
  const all = f.button("전부 넣기");
  all.focus(); // a click focuses the button first; Tab + Enter fires the same click
  all.fire("click");
  assert.equal(all.disabled, true);
  assert.ok(document.activeElement === f.search, "전부 넣기 greyed itself");
  assert.equal(f.menu.hidden, true, "the list does not open by itself");
  const clear = f.button("모두 지우기");
  clear.focus();
  clear.fire("click");
  assert.equal(clear.disabled, true);
  assert.ok(document.activeElement === f.search, "모두 지우기 greyed itself");
});

test("a list that changes under a focused 전부 넣기 (refresh) does not drop the focus to the page either", () => {
  const f = field(["euler", "lcm"], { names: ["euler", "lcm", "later"] });
  const all = f.button("전부 넣기");
  all.focus();
  f.store.names = ["euler", "lcm"]; // the stored values are now the whole list
  f.made.refresh();
  assert.equal(all.disabled, true);
  assert.ok(document.activeElement === f.search, "the focus went to the add box");

  const usable = field(["lcm"]);
  const fill = usable.button("전부 넣기");
  fill.focus();
  usable.made.refresh();
  assert.equal(fill.disabled, false);
  assert.ok(document.activeElement === fill, "a button that stays usable keeps the focus");
});

test("a block moves with Alt+←/→ on its ✕ (the focus follows it) or by dragging it onto another", () => {
  const f = field(["euler", "lcm", "dpmpp_2m"]);
  const cross = (index) => f.chips()[index].byClass("ais-chip-x")[0];
  cross(0).focus();
  const key = cross(0).fire("keydown", { key: "ArrowRight", altKey: true });
  assert.equal(key.defaultPrevented && key.stopped, true);
  assert.deepEqual(f.store.values, ["lcm", "euler", "dpmpp_2m"]);
  assert.equal(document.activeElement, cross(1));
  cross(1).fire("keydown", { key: "ArrowLeft", altKey: true });
  assert.deepEqual(f.store.values, ["euler", "lcm", "dpmpp_2m"]);
  assert.equal(document.activeElement, cross(0));
  cross(0).fire("keydown", { key: "ArrowLeft", altKey: true });
  assert.deepEqual(f.store.values, ["euler", "lcm", "dpmpp_2m"], "already first");
  assert.equal(cross(0).fire("keydown", { key: "ArrowRight" }).defaultPrevented, false, "without Alt the key is not ours");

  const data = { setData() {}, effectAllowed: "" };
  assert.equal(f.chips()[0].draggable, true);
  f.chips()[2].fire("dragstart", { dataTransfer: data });
  assert.equal(f.chips()[0].fire("dragover", { dataTransfer: data }).defaultPrevented, true);
  const drop = f.chips()[0].fire("drop", { dataTransfer: data });
  assert.equal(drop.defaultPrevented && drop.stopped, true);
  assert.deepEqual(f.store.values, ["dpmpp_2m", "euler", "lcm"]);
  assert.equal(f.chips()[1].fire("dragover", { dataTransfer: data }).defaultPrevented, false, "nothing of ours is being dragged");
});

test("Delete on a block's ✕ takes it out; refresh() redraws from the stored values and a new list", () => {
  const f = field(["euler", "new_sampler"]);
  assert.equal(f.chips()[1].classList.contains("warn"), true);
  f.store.names = [...SAMPLERS, "new_sampler"];
  f.made.refresh();
  assert.equal(f.chips()[1].classList.contains("warn"), false);
  const held = f.chips()[0].byClass("ais-chip-x")[0].fire("keydown", { key: "Delete", repeat: true });
  assert.deepEqual(f.store.values, ["euler", "new_sampler"], "a held key takes no more blocks");
  assert.equal(held.stopped, true, "but it still never reaches ComfyUI (Delete deletes the selected nodes there)");
  f.chips()[0].byClass("ais-chip-x")[0].fire("keydown", { key: "Delete" });
  assert.deepEqual(f.store.values, ["new_sampler"]);
  f.store.values = ["lcm", "euler"]; // changed from outside (typed in the text box)
  f.made.refresh();
  assert.deepEqual(f.texts(), ["lcm", "euler"]);
});
