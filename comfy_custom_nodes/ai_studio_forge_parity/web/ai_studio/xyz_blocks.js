// Anima studio: the value box of an XYZ axis whose values are free text (Prompt S/R, Prompt order), drawn as text blocks:
// one editable block a value, in order — "+ 블록" adds one, ✕ takes one out, ↑ ↓ move it. A block may hold commas, quotes,
// spaces and line breaks and is still ONE value, so Prompt S/R can swap several tags at once.
// The blocks are only a view: the axis keeps its value text (one value a line, CSV quoted — xyz_chips.js joinValues), which
// xyz_values.py reads as before; a value box written by hand, Forge's `"a, b", "c, d"` included, shows as the same blocks.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { stripValue } from "./xyz_chips.js";

const CSS_ID = "ai-studio-xyz-blocks-css";
const LIST_PAD = 5; // xyz_blocks.css: .ais-blks padding

/**
 * The axes drawn as text blocks, by axis id: `tag(index)` = a block's name, `find` = the first block is set apart (Prompt
 * S/R: the text to search for), `hint` = the line over the blocks, `first` / `rest` = the placeholders, `typed` = the
 * tooltip of the axis' 직접 입력 switch (the value box as text).
 */
export const BLOCK_AXES = Object.freeze({
  prompt_sr: Object.freeze({
    tag: (index) => (index === 0 ? "찾을 글" : "바꿀 글"),
    label: (index) => (index === 0 ? "찾을 글" : `바꿀 글 ${index}`),
    find: true,
    hint: "첫 블록을 찾아 블록마다 바꿈 · 쉼표도 한 블록",
    first: "프롬프트에 적힌 그대로 (쉼표 포함 여러 태그도)",
    rest: "(빈 값 = 찾은 글 지우기)",
    typed: '블록 대신 글자로 고치기 (한 줄에 하나, 쉼표로도 나눔. 쉼표가 든 값은 "a, b" 처럼 따옴표로)',
  }),
  prompt_order: Object.freeze({
    tag: (index) => `조각 ${index + 1}`,
    label: (index) => `조각 ${index + 1}`,
    find: false,
    hint: "조각들을 모든 순서로 · 쉼표도 한 블록",
    first: "프롬프트에 적힌 그대로",
    rest: "프롬프트에 적힌 그대로",
    typed: '블록 대신 글자로 고치기 (쉼표로 나눔. 쉼표가 든 조각은 "a, b" 처럼 따옴표로)',
  }),
});

/**
 * The values the blocks store: each stripped the way the planner strips it (so the value box says what will run), and a
 * lone empty block is no value at all — the box is then empty, as before anything was typed.
 */
export function blockValues(values) {
  const out = values.map(stripValue);
  return out.length === 1 && out[0] === "" ? [] : out;
}

/**
 * The scrollTop that brings a block (`top`, `height` inside the list's content) wholly into a list `view` high that is
 * scrolled to `scroll`, `pad` clear of the edge. Null = stay: the block shows, or nothing is laid out. A block higher than
 * the list shows its top, unless the list is already somewhere inside it.
 */
export function revealScrollTop({ top, height, view, scroll, pad = 0 }) {
  if (![top, height, view, scroll].every(Number.isFinite) || view <= 0) return null;
  const upper = Math.max(0, top - pad); // the furthest scroll that still shows the block's top
  const lower = Math.max(0, top + height + pad - view); // the least scroll that shows its bottom
  if (lower > upper) return scroll < upper || scroll > lower ? upper : null;
  if (scroll > upper) return upper;
  return scroll < lower ? lower : null;
}

/** Loads xyz_blocks.css once (the way xyz_chips.js loads its own). */
export function addBlockStyles(doc = globalThis.document) {
  if (!doc?.head || doc.getElementById(CSS_ID)) return;
  const link = doc.createElement("link");
  link.id = CSS_ID;
  link.rel = "stylesheet";
  link.href = new URL("./xyz_blocks.css", import.meta.url).href;
  doc.head.append(link);
}

/** Makes a block's text as high as its lines (CSS sets the floor), keeping every scroll box around it where it is. */
function fit(area) {
  const boxes = [];
  for (let node = area.parentElement; node; node = node.parentElement) {
    if (node.getAttribute?.("data-scroll") != null) boxes.push([node, node.scrollTop]); // the list box, the panel's column
  }
  area.style.height = "auto";
  const height = area.scrollHeight;
  area.style.height = height ? `${height + area.offsetHeight - area.clientHeight}px` : "";
  for (const [node, top] of boxes) if (node.scrollTop !== top) node.scrollTop = top;
}

/**
 * The block field of one axis. `name` = the axis letter (labels), `tall` = the X axis' height, `scrollKey` = the list box's
 * `data-scroll` key (studio_dom.js keepScroll), `texts` = the axis' entry of BLOCK_AXES; `read() -> values` (the value box
 * split as the planner splits it), `write(values)` stores them. Typing writes without a redraw (the block and its caret
 * stay); adding, taking out and moving redraw the blocks inside the same list box and put the focus on the block that was
 * acted on. The list box scrolls just enough to show that block, also while it is typed in; the focus itself never
 * scrolls, so the panel's column stays. Returns `{el, refresh(), grow()}`: `refresh()` redraws from `read()` after a change from outside,
 * `grow()` fits the blocks' heights again (call it once the field is on the page and when its width changes).
 */
export function blockField({ name, tall = false, scrollKey, texts, read, write }) {
  let values = []; // what the blocks show: as typed, until the next redraw
  const box = h("div", { class: `ais-blks${tall ? " tall" : ""}`, role: "list", "aria-label": `${name} 값`, "data-scroll": scrollKey });
  const addBtn = h("button", {
    type: "button", class: "ais-mini ais-blk-add", title: "빈 블록을 끝에 더하기 (블록 안에서 Enter 는 그 블록 바로 뒤에)",
    onClick: () => add(values.length), onKeydown: (event) => buttonKey(event),
  }, "+ 블록");
  const el = h("div", { class: "ais-blockfield" }, h("div", { class: "ais-blk-head" }, h("span", { class: "ais-blk-hint" }, texts.hint), addBtn), box);

  const rows = () => [...box.querySelectorAll(".ais-blk")];
  const part = (index, cls) => rows()[index]?.querySelector(`.${cls}`) || null;
  /** The focus on block `index`'s control `cls`; a greyed one hands it to the first of `others` that is not, at last to the text. */
  function focusPart(index, cls, ...others) {
    const target = [cls, ...others, "ais-blk-text"].map((item) => part(index, item)).find((item) => item && !item.disabled);
    target?.focus({ preventScroll: true });
    reveal(index);
    return target;
  }
  /** Scrolls the list box, and nothing else, so that block `index` shows. */
  function reveal(index) {
    const row = rows()[index];
    if (!row) return;
    const top = row.offsetParent === box ? row.offsetTop : row.offsetTop - box.offsetTop - box.clientTop;
    const to = revealScrollTop({ top, height: row.offsetHeight, view: box.clientHeight, scroll: box.scrollTop, pad: LIST_PAD });
    if (to !== null) box.scrollTop = to;
  }
  const commit = () => write(blockValues(values));

  function add(at) {
    values.splice(at, 0, "");
    commit();
    draw();
    focusPart(at, "ais-blk-text");
  }
  function remove(index) {
    values.splice(index, 1);
    commit();
    draw();
    focusPart(Math.min(index, values.length - 1), "ais-blk-x");
  }
  /** Moves block `from` to `to`; returns false when there is nowhere to go. */
  function move(from, to) {
    if (to < 0 || to >= values.length) return false;
    values.splice(to, 0, ...values.splice(from, 1));
    commit();
    draw();
    return true;
  }

  /** A button's keys: Delete / ⌫ would be ComfyUI's "delete the selected nodes"; on ✕ they take the block out. */
  function buttonKey(event, index = -1) {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    event.preventDefault();
    event.stopPropagation();
    if (index >= 0 && !event.repeat) remove(index); // a held key would run through the blocks
  }
  function textKey(event, index) {
    if (event.isComposing || event.keyCode === 229) return; // Korean IME composing: Enter belongs to it
    const { key } = event;
    if (event.altKey && (key === "ArrowUp" || key === "ArrowDown")) {
      const { selectionStart, selectionEnd } = event.target;
      const to = index + (key === "ArrowDown" ? 1 : -1);
      if (move(index, to)) {
        const area = focusPart(to, "ais-blk-text");
        if (area && Number.isInteger(selectionStart)) {
          if (typeof area.setSelectionRange === "function") area.setSelectionRange(selectionStart, selectionEnd);
          else Object.assign(area, { selectionStart, selectionEnd });
        }
      }
    } else if (key === "Enter" && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) add(index + 1);
    else return; // Shift+Enter: a line break in the value; Ctrl+Enter: ComfyUI's run
    event.preventDefault();
    event.stopPropagation();
  }

  function block(value, index) {
    const label = texts.label(index);
    const area = h("textarea", {
      class: "ais-f ais-blk-text", rows: 1, value, "aria-label": `${name} ${label}`,
      placeholder: index === 0 ? texts.first : texts.rest,
      onInput: (event) => {
        values[index] = event.target.value;
        commit();
        fit(event.target);
        reveal(index); // a text that grew by a line at the list's bottom edge would be typed under the fold
      },
      onKeydown: (event) => textKey(event, index),
    });
    area.spellcheck = false; // h() skips false props
    const button = (cls, text, word, title, off, onClick, keyIndex = -1) => {
      const made = h("button", { type: "button", class: `ais-blk-btn ${cls}`, "aria-label": `${label} ${word}`, title, onClick, onKeydown: (event) => buttonKey(event, keyIndex) }, text);
      made.disabled = off;
      return made;
    };
    const lone = values.length === 1 && value === ""; // the empty block of an empty box: nothing to take out
    return h("div", { class: `ais-blk${texts.find && index === 0 ? " find" : ""}`, role: "listitem" },
      h("span", { class: "ais-blk-tag", title: texts.find && index === 0 ? "프롬프트·네거티브에서 찾을 글. 이 글 그대로가 첫 칸(원본)" : undefined }, texts.tag(index)),
      area,
      h("span", { class: "ais-blk-btns" },
        button("ais-blk-up", "↑", "위로", "위로 (글 안에서는 Alt+↑)", index === 0, () => { if (move(index, index - 1)) focusPart(index - 1, "ais-blk-up", "ais-blk-down"); }),
        button("ais-blk-down", "↓", "아래로", "아래로 (글 안에서는 Alt+↓)", index === values.length - 1, () => { if (move(index, index + 1)) focusPart(index + 1, "ais-blk-down", "ais-blk-up"); }),
        button("ais-blk-x", "✕", "빼기", "블록 빼기", lone, () => remove(index), index)));
  }

  function grow() {
    box.querySelectorAll(".ais-blk-text").forEach(fit);
  }
  function draw() {
    const top = box.scrollTop;
    values = read();
    if (!values.length) values = [""]; // an empty box still shows where to type
    box.replaceChildren(...values.map(block));
    grow();
    if (box.scrollTop !== top) box.scrollTop = top;
  }

  draw();
  return { el, refresh: draw, grow };
}
