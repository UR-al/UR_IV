// node:test for the XYZ panel's text blocks (web/ai_studio/xyz_blocks.js): the value box of Prompt S/R and Prompt order drawn
// as one editable block a value. The blocks are a view of the axis' value text, kept in the CSV form xyz_values.py reads
// (golden table shared with tests/test_xyz_values_golden.py), so a block may hold commas, quotes, spaces and line breaks.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { installFakeDom, mount } from "./fake_dom.mjs";
import { BLOCK_AXES, blockField, blockValues } from "../web/ai_studio/xyz_blocks.js";
import { joinValues, splitValues, stripValue } from "../web/ai_studio/xyz_chips.js";

const GOLDEN = JSON.parse(readFileSync(new URL("./fixtures/xyz_values_golden.json", import.meta.url), "utf8"));

/** A field over a value box (`store.text`), read and written the way xyz_view.js does it. */
function field(text = "", axis = "prompt_sr") {
  installFakeDom();
  const store = { text, writes: 0 };
  const made = blockField({
    name: "X",
    scrollKey: `blocks-0-${axis}`,
    texts: BLOCK_AXES[axis],
    read: () => splitValues(store.text),
    write: (values) => { store.text = joinValues(values); store.writes += 1; },
  });
  mount(made.el);
  const el = made.el;
  return {
    store, made, el,
    box: el.byClass("ais-blks")[0],
    blocks: () => el.byClass("ais-blk"),
    areas: () => el.byClass("ais-blk-text"),
    values: () => el.byClass("ais-blk-text").map((area) => area.value),
    tags: () => el.byClass("ais-blk-tag").map((tag) => tag.textContent),
    add: el.byClass("ais-blk-add")[0],
    button: (index, name) => el.byClass("ais-blk")[index].byClass(`ais-blk-${name}`)[0],
    type(index, value) {
      const area = el.byClass("ais-blk-text")[index];
      area.focus();
      area.value = value;
      area.fire("input");
      return area;
    },
    key: (target, key, extra = {}) => target.fire("keydown", { key, altKey: false, shiftKey: false, ctrlKey: false, metaKey: false, ...extra }),
  };
}

test("only Prompt S/R and Prompt order are block axes; the first S/R block is the text to find", () => {
  assert.deepEqual(Object.keys(BLOCK_AXES), ["prompt_sr", "prompt_order"]);
  assert.deepEqual([0, 1, 2].map(BLOCK_AXES.prompt_sr.tag), ["찾을 글", "바꿀 글", "바꿀 글"]);
  assert.deepEqual([0, 1, 2].map(BLOCK_AXES.prompt_order.tag), ["조각 1", "조각 2", "조각 3"]);
  assert.equal(Object.isFrozen(BLOCK_AXES), true);
});

test("one block a value, in order; a value with commas, quotes, spaces or a line break is still one block", () => {
  const f = field('"blue hair, long hair"\n"say ""hi"", ok"\n"1girl,\nsmile", plain words');
  assert.deepEqual(f.values(), ["blue hair, long hair", 'say "hi", ok', "1girl,\nsmile", "plain words"]);
  assert.deepEqual(f.tags(), ["찾을 글", "바꿀 글", "바꿀 글", "바꿀 글"]);
  assert.equal(f.blocks()[0].classList.contains("find"), true);
  assert.equal(f.blocks()[1].classList.contains("find"), false);
  assert.deepEqual(f.areas().map((area) => area.attrs["aria-label"]), ["X 찾을 글", "X 바꿀 글 1", "X 바꿀 글 2", "X 바꿀 글 3"]);
  assert.equal(f.box.attrs["data-scroll"], "blocks-0-prompt_sr");
  assert.equal(f.box.attrs.role, "list");
  assert.equal(f.store.writes, 0, "drawing writes nothing");
});

test("Forge's one-line form with quotes shows as the same blocks", () => {
  assert.deepEqual(field('"blue hair, long hair", "red hair, short hair"').values(), ["blue hair, long hair", "red hair, short hair"]);
  assert.deepEqual(field("blue hair, long hair").values(), ["blue hair", "long hair"], "without quotes a comma splits, as the planner reads it");
});

test("golden: the blocks show what the planner reads, and what they write the planner reads back the same", () => {
  for (const row of GOLDEN) {
    const f = field(row.text);
    assert.deepEqual(f.values(), row.fields.length ? row.fields : [""], JSON.stringify(row.text));
    assert.equal(f.store.writes, 0);
    const written = blockValues(row.fields);
    assert.deepEqual(splitValues(joinValues(written)), written, JSON.stringify(row.text));
  }
});

test("an empty value box shows one empty 찾을 글 block and writes nothing; it has nothing to take out or move", () => {
  const f = field("");
  assert.deepEqual(f.values(), [""]);
  assert.deepEqual(f.tags(), ["찾을 글"]);
  assert.equal(f.store.writes, 0);
  assert.deepEqual(["x", "up", "down"].map((name) => f.button(0, name).disabled), [true, true, true]);
  assert.match(f.areas()[0].attrs.placeholder, /프롬프트/);
});

test("typing in a block writes the value box in CSV without a redraw: the block, the caret and the scroll stay", () => {
  const f = field("a\nb");
  f.box.scrollTop = 30;
  const area = f.type(1, "red hair, short hair");
  assert.equal(f.store.text, 'a\n"red hair, short hair"');
  assert.equal(f.areas()[1], area, "the same box");
  assert.equal(document.activeElement, area);
  assert.equal(f.box.scrollTop, 30);
  f.type(0, 'say "hi"');
  assert.equal(f.store.text, '"say ""hi"""\n"red hair, short hair"');
  f.type(0, "blue ");
  assert.equal(f.store.text, 'blue\n"red hair, short hair"', "the planner strips a value's ends: the box says the same");
  assert.equal(f.areas()[0].value, "blue ", "while typing the block keeps what was typed");
  f.type(0, "1girl,\nsmile");
  assert.deepEqual(splitValues(f.store.text), ["1girl,\nsmile", "red hair, short hair"]);
});

test("typing in the block of an empty box makes the first value; emptying the only block empties the box", () => {
  const f = field("");
  f.type(0, "blue hair, long hair");
  assert.equal(f.store.text, '"blue hair, long hair"');
  f.type(0, "");
  assert.equal(f.store.text, "", "no values: not one empty value");
  assert.deepEqual(blockValues([""]), []);
  assert.deepEqual(blockValues(["", ""]), ["", ""]);
  assert.deepEqual(blockValues([" a ", "b\n"]), ["a", "b"]);
});

// the list box bringing the block into view: xyz_blocks_reveal.test.mjs (this file's document has no layout)
test("+ 블록 adds an empty block at the end and takes the focus there", () => {
  const f = field("blue hair");
  f.add.focus();
  f.add.fire("click");
  assert.equal(f.store.text, 'blue hair\n""');
  assert.deepEqual(f.values(), ["blue hair", ""]);
  assert.equal(document.activeElement, f.areas()[1]);
  assert.equal(f.el.byClass("ais-blks")[0], f.box, "the list box is the same element");
  assert.match(f.areas()[1].attrs.placeholder, /빈 값/);
  f.type(1, "red hair, bob cut");
  assert.equal(f.store.text, 'blue hair\n"red hair, bob cut"');
  const empty = field("");
  empty.add.fire("click");
  assert.equal(empty.store.text, '""\n""', "two blocks are two values; the planner then asks for the text to find");
  assert.deepEqual(empty.tags(), ["찾을 글", "바꿀 글"]);
});

test("Enter in a block opens a new one right after it; Shift+Enter, Ctrl+Enter and an IME's Enter are left alone", () => {
  const f = field("a\nb");
  const first = f.areas()[0];
  first.focus();
  const event = f.key(first, "Enter");
  assert.equal(event.defaultPrevented, true);
  assert.equal(event.stopped, true);
  assert.deepEqual(f.values(), ["a", "", "b"]);
  assert.equal(f.store.text, 'a\n""\nb');
  assert.equal(document.activeElement, f.areas()[1]);
  for (const extra of [{ shiftKey: true }, { ctrlKey: true }, { metaKey: true }, { isComposing: true }, { keyCode: 229 }]) {
    const left = f.key(f.areas()[2], "Enter", extra);
    assert.equal(left.defaultPrevented, false, JSON.stringify(extra)); // a line break in the value, ComfyUI's run, the IME's own
    assert.equal(left.stopped, false, JSON.stringify(extra));
  }
  assert.deepEqual(f.values(), ["a", "", "b"]);
});

test("✕ takes a block out; the focus goes to the block that took its place, at the end to the one before, then to the empty block", () => {
  const f = field("a\nb\nc");
  f.box.scrollTop = 12;
  f.button(1, "x").focus();
  f.button(1, "x").fire("click");
  assert.equal(f.store.text, "a\nc");
  assert.deepEqual(f.values(), ["a", "c"]);
  assert.deepEqual(f.tags(), ["찾을 글", "바꿀 글"]);
  assert.equal(document.activeElement, f.button(1, "x"));
  assert.equal(f.box.scrollTop, 12);
  document.activeElement.fire("click");
  assert.equal(document.activeElement, f.button(0, "x"));
  document.activeElement.fire("click");
  assert.equal(f.store.text, "");
  assert.deepEqual(f.values(), [""]);
  assert.equal(document.activeElement, f.areas()[0], "nothing left to take out: the empty block's text");
  const first = field("a\nb");
  first.button(0, "x").fire("click");
  assert.deepEqual(first.tags(), ["찾을 글"], "the next value becomes the text to find");
  assert.equal(first.store.text, "b");
});

test("↑ ↓ move a block and the focus follows it; the ends are greyed", () => {
  const f = field("a\nb\nc");
  assert.deepEqual(f.blocks().map((_, index) => [f.button(index, "up").disabled, f.button(index, "down").disabled]), [[true, false], [false, false], [false, true]]);
  f.button(2, "up").focus();
  f.button(2, "up").fire("click");
  assert.equal(f.store.text, "a\nc\nb");
  assert.equal(document.activeElement, f.button(1, "up"), "the same block, the same button");
  document.activeElement.fire("click");
  assert.equal(f.store.text, "c\na\nb");
  assert.equal(document.activeElement, f.button(0, "down"), "↑ is greyed at the top: the block's other arrow, not the page");
  document.activeElement.fire("click");
  assert.equal(f.store.text, "a\nc\nb");
  assert.equal(document.activeElement, f.button(1, "down"));
  assert.deepEqual(f.tags(), ["찾을 글", "바꿀 글", "바꿀 글"]);
});

test("Alt+↑ / Alt+↓ in a block's text move it; the focus and the caret stay in that text", () => {
  const f = field("a\nbcd\ne");
  const area = f.areas()[1];
  area.focus();
  area.selectionStart = 2;
  area.selectionEnd = 3;
  const up = f.key(area, "ArrowUp", { altKey: true });
  assert.equal(up.defaultPrevented, true);
  assert.equal(up.stopped, true);
  assert.equal(f.store.text, "bcd\na\ne");
  const moved = f.areas()[0];
  assert.equal(document.activeElement, moved);
  assert.deepEqual([moved.selectionStart, moved.selectionEnd], [2, 3], "the caret is where it was");
  f.key(moved, "ArrowUp", { altKey: true }); // already first: nothing
  assert.equal(f.store.text, "bcd\na\ne");
  f.key(moved, "ArrowDown", { altKey: true });
  assert.equal(f.store.text, "a\nbcd\ne");
  assert.equal(document.activeElement, f.areas()[1]);
  assert.equal(f.key(f.areas()[1], "ArrowDown").defaultPrevented, false, "a plain arrow is the caret's");
});

test("what was typed and not yet redrawn is what moves, takes out and adds work on", () => {
  const f = field("a\nb");
  f.type(1, "red hair, ");
  f.add.fire("click");
  assert.deepEqual(f.values(), ["a", "red hair,", ""], "a redraw shows the value as the planner reads it");
  assert.equal(f.store.text, 'a\n"red hair,"\n""');
});

test("Delete and ⌫ on a block's buttons never reach ComfyUI (it would delete the selected nodes); on ✕ they take the block out", () => {
  const f = field("a\nb\nc");
  for (const key of ["Delete", "Backspace"]) {
    const event = f.key(f.button(0, "down"), key);
    assert.equal(event.stopped && event.defaultPrevented, true, key);
    assert.equal(f.key(f.add, key).stopped, true, key);
  }
  assert.equal(f.store.text, "a\nb\nc");
  f.button(1, "x").focus();
  const held = f.key(f.button(1, "x"), "Delete", { repeat: true });
  assert.equal(held.stopped, true);
  assert.equal(f.store.text, "a\nb\nc", "a held key does not run through the blocks");
  f.key(f.button(1, "x"), "Delete");
  assert.equal(f.store.text, "a\nc");
  assert.equal(f.key(f.areas()[0], "Backspace").stopped, false, "in a text the key is the text's");
});

test("refresh() redraws from the value box (a change from outside) into the same list box", () => {
  const f = field("a");
  f.box.scrollTop = 8;
  f.store.text = '"x, y"\nz';
  f.made.refresh();
  assert.deepEqual(f.values(), ["x, y", "z"]);
  assert.equal(f.el.byClass("ais-blks")[0], f.box);
  assert.equal(f.box.scrollTop, 8);
  assert.equal(f.store.writes, 0);
  assert.equal(typeof f.made.grow, "function");
  assert.doesNotThrow(() => f.made.grow());
});

test("Prompt order: every block is a piece, numbered; the X axis' box is the tall one", () => {
  const f = field('"blue hair, long hair"\nsmile', "prompt_order");
  assert.deepEqual(f.tags(), ["조각 1", "조각 2"]);
  assert.equal(f.blocks()[0].classList.contains("find"), false);
  assert.deepEqual(f.areas().map((area) => area.attrs["aria-label"]), ["X 조각 1", "X 조각 2"]);
  installFakeDom();
  const tall = blockField({ name: "X", tall: true, scrollKey: "k", texts: BLOCK_AXES.prompt_sr, read: () => [], write() {} });
  assert.equal(tall.el.byClass("ais-blks")[0].classList.contains("tall"), true);
});

test("a block stripped the way the planner strips it: a BOM stays in the value", () => {
  const BOM = String.fromCharCode(0xfeff);
  assert.deepEqual(blockValues([`${BOM}a `, "\tb"]), [`${BOM}a`, "b"]);
  assert.equal(stripValue(`${BOM}a `), `${BOM}a`);
});

test("xyz_blocks.css: every rule is ais-* scoped; the list grows to a limit and then scrolls, never following a redraw", () => {
  const css = readFileSync(new URL("../web/ai_studio/xyz_blocks.css", import.meta.url), "utf8");
  const selectors = css.replace(/\/\*[\s\S]*?\*\//g, "").split("}").map((rule) => rule.split("{")[0].trim()).filter(Boolean);
  assert.ok(selectors.length >= 10);
  for (const selector of selectors) for (const part of selector.split(",")) assert.match(part.trim(), /^\.ais-/, selector);
  assert.match(css, /\.ais-blks\s*\{[^}]*max-height:[^}]*overflow-y:\s*auto[^}]*overflow-anchor:\s*none/);
});
