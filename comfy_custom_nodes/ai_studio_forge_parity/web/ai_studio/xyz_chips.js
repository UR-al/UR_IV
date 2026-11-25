// Anima studio: the value box of an XYZ axis whose values come from a list (sampler, checkpoint, VAE, a switch ...), drawn as
// blocks: the chosen values in order, each with its ✕, and an add box that lists the values not chosen yet. The blocks are
// only a view: the axis keeps its value text (one value a line, CSV quoted), which xyz_values.py reads as before.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";

const CSS_ID = "ai-studio-xyz-chips-css";
const BOOL_WORDS = new Set(["true", "1", "on", "yes", "false", "0", "off", "no"]); // xyz_values.py _TRUE / _FALSE
const BLANK = "(빈 값)";
const MENU_MIN = 110; // px: the shortest the open list is made (about four values)
const MENU_GAP = 6; // px kept free under the open list
// What Python's str.strip() takes off (str.isspace). Not String.trim(): that also takes U+FEFF (a BOM) and leaves
// U+001C..U+001F and U+0085, so a block would show a value the planner reads differently.
const PY_SPACE = String.raw`\t-\r\x1c- \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000`;
const OUTER_SPACE = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "g");

/** `text` without its outer whitespace, as Python's `str.strip()` does it (the planner strips every value so). */
export const stripValue = (text) => String(text).replace(OUTER_SPACE, "");

/**
 * A value box split on commas and line breaks: the twin of xyz_values.py `split_values` (csv.reader with
 * skipinitialspace, every field stripped the way str.strip() does); js_tests/fixtures/xyz_values_golden.json pins the two
 * together. Quotes protect commas and line breaks, `""` in quotes is a quote, an empty line has no value, an empty field
 * between commas is kept.
 */
export function splitValues(text) {
  const fields = [];
  let field = "";
  let state = "record"; // record (line start) | field (field start) | plain | quoted | quote (a quote inside quotes)
  const save = (next) => {
    fields.push(stripValue(field));
    field = "";
    state = next;
  };
  for (const c of String(text ?? "")) {
    const eol = c === "\n" || c === "\r";
    if (state === "record") {
      if (eol) continue;
      state = "field";
    }
    if (state === "field") {
      if (eol) save("record");
      else if (c === '"') state = "quoted";
      else if (c === ",") save("field");
      else if (c !== " ") {
        field += c;
        state = "plain";
      }
    } else if (state === "plain") {
      if (eol) save("record");
      else if (c === ",") save("field");
      else field += c;
    } else if (state === "quoted") {
      if (c === '"') state = "quote";
      else field += c;
    } else if (c === '"') {
      field += c;
      state = "quoted";
    } else if (c === ",") save("field");
    else if (eol) save("record");
    else {
      field += c;
      state = "plain";
    }
  }
  if (state !== "record") save("record");
  return fields;
}

/** A value as one line of the value box (quoted when it is empty or has a comma, quote, line break or outer spaces). */
export function csvField(value) {
  return value === "" || /[",\r\n]/.test(value) || value !== stripValue(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

/** The value box of these values: one a line, as the panel has always written a picked value. */
export const joinValues = (values) => values.map(csvField).join("\n");

/**
 * Whether the planner will take `value` on this axis (xyz_plan.py `_resolve`): a name of the list; a checkpoint also by a
 * part of its name; attention ignoring case; a DoRA mode by sam-extra's label (a mode key is left to the planner); a switch
 * by its word. `items` = `[{value, label}]`, null while the list is not read (nothing is marked then). Only a hint for the
 * block's look: the planner's own error still shows under the panel.
 */
export function knownValue(axis, value, items) {
  if (axis.kind === "bool") return BOOL_WORDS.has(value.toLowerCase());
  if (!items) return true;
  if (items.some((item) => item.value === value)) return true;
  const low = stripValue(value).toLowerCase();
  if (axis.id === "checkpoint") return value !== "" && items.some((item) => item.value.includes(value));
  if (axis.id === "attention") return items.some((item) => item.value === low);
  if (axis.id === "dora_mode") {
    const plain = (label) => label.replace("/Comfy", "").toLowerCase();
    return /^[a-z0-9_]+$/.test(value) || items.some((item) => plain(item.value) === low || plain(item.value).startsWith(`${low} (`));
  }
  return false;
}

/** The list's values not chosen yet whose label has every word typed in the add box (case ignored). */
export function menuItems(items, values, query) {
  const chosen = new Set(values);
  const words = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
  return items.filter((item) => !chosen.has(item.value) && words.every((word) => item.label.toLowerCase().includes(word)));
}

/** `values` with the one at `from` moved to `to`; a move off the ends changes nothing. */
export function moveValue(values, from, to) {
  const next = [...values];
  if (from === to || from < 0 || from >= next.length || to < 0 || to >= next.length) return next;
  next.splice(to, 0, ...next.splice(from, 1));
  return next;
}

/** Loads xyz_chips.css once (the way studio_main.js loads studio.css). */
export function addChipStyles(doc = globalThis.document) {
  if (!doc?.head || doc.getElementById(CSS_ID)) return;
  const link = doc.createElement("link");
  link.id = CSS_ID;
  link.rel = "stylesheet";
  link.href = new URL("./xyz_chips.css", import.meta.url).href;
  doc.head.append(link);
}

/**
 * Keeps the open list inside the column the field is in: a list that would reach past the column's lower edge is made
 * shorter; only when fewer than MENU_MIN px are left is the column scrolled instead (that moves the add box under the
 * pointer, so it is the last resort). Never `scrollIntoView`: it would also scroll the panel's clipped root.
 */
function fitInColumn(field, menu) {
  menu.style.maxHeight = "";
  const column = field.closest?.("[data-scroll]");
  if (!column || typeof menu.getBoundingClientRect !== "function") return;
  const frame = column.getBoundingClientRect();
  const scale = frame.height / column.offsetHeight || 1; // a ComfyUI DOM widget is drawn scaled
  const rect = menu.getBoundingClientRect();
  const over = (rect.bottom - frame.bottom) / scale + MENU_GAP;
  if (over <= 0) return;
  const room = (frame.bottom - rect.top) / scale - MENU_GAP;
  if (room >= MENU_MIN) menu.style.maxHeight = `${Math.floor(room)}px`;
  else column.scrollTop += Math.ceil(over);
}

/**
 * The block field of one axis. `name` = the axis letter (labels), `tall` = the X axis' height, `scrollKey` = the box's
 * `data-scroll` key (studio_dom.js keepScroll); `read() -> values`, `write(values)` store them (the field redraws itself,
 * in place: the box, its scroll and the focus stay); `list() -> {items: [{value, label}], loaded}`; `known(value)`;
 * `opened()` runs when the list opens (the caller may read the list again). Returns `{el, refresh()}`; `refresh()` redraws
 * from `read()` and `list()` after a change from outside.
 */
export function chipField({ name, tall = false, scrollKey, read, write, list, known, opened }) {
  let cursor = 0;
  let shown = [];
  let dragFrom = -1;
  const box = h("div", { class: `ais-chips${tall ? " tall" : ""}`, role: "list", "aria-label": `${name} 값`, "data-scroll": scrollKey });
  const search = h("input", {
    type: "text", class: "ais-f ais-grow ais-chip-search", placeholder: "값 넣기 (검색)", autocomplete: "off",
    role: "combobox", "aria-expanded": "false", "aria-label": `${name} 값 넣기`,
  });
  search.spellcheck = false; // h() skips false props
  const allBtn = h("button", { type: "button", class: "ais-mini", title: "목록의 값을 목록 순서대로 전부 넣기", onClick: () => fill() }, "전부 넣기");
  const clearBtn = h("button", { type: "button", class: "ais-mini", title: "넣은 값을 모두 빼기", onClick: () => change([]) }, "모두 지우기");
  const menu = h("div", { class: "ais-chip-menu", role: "listbox", hidden: true });
  const el = h("div", { class: "ais-chipfield" }, box, h("div", { class: "ais-chip-add" }, search, allBtn, clearBtn, menu));

  const labelOf = (value, items) => (value === "" ? BLANK : items.find((item) => item.value === value)?.label ?? value);
  const crosses = () => [...box.querySelectorAll(".ais-chip-x")];
  /** The focus on block `index`'s ✕; with no block left, on the add box (else it would drop to the page under the panel). */
  const focusChip = (index) => (crosses()[index] || search).focus({ preventScroll: true });

  /**
   * Switches a button off or on. One that is switched off while it has the focus (전부 넣기 / 모두 지우기 grey themselves
   * with their own click) first hands the focus to the add box: the browser would drop it to the page, where the next key
   * is a ComfyUI shortcut and Tab starts over.
   */
  function grey(button, off) {
    if (off && !button.disabled && button.ownerDocument.activeElement === button) search.focus({ preventScroll: true });
    button.disabled = off;
  }
  function change(values) {
    write(values);
    draw();
  }
  function fill() {
    change(list().items.map((item) => item.value));
  }
  function remove(index) {
    const values = read().filter((_, at) => at !== index);
    change(values);
    focusChip(Math.min(index, values.length - 1));
  }
  function move(from, to, focus = false) {
    const values = read();
    if (to < 0 || to >= values.length || from === to) return;
    change(moveValue(values, from, to));
    if (focus) focusChip(to);
  }
  function add(value, byKey = false) {
    change([...read(), value]);
    box.scrollTop = box.scrollHeight; // the new block is the last one
    if (byKey) search.select(); // the next key starts a new search; Enter again takes the next match
  }

  function chipKey(event, index) {
    const { key } = event;
    if (event.altKey && (key === "ArrowLeft" || key === "ArrowRight")) move(index, index + (key === "ArrowRight" ? 1 : -1), true);
    else if (!event.altKey && (key === "Delete" || key === "Backspace")) {
      if (!event.repeat) remove(index); // a held key would run through the blocks: the focus moves on to the next one
    } else return;
    event.preventDefault(); // Alt+← is also the browser's "back"
    event.stopPropagation(); // and Delete / ⌫ are ComfyUI's "delete the selected nodes"
  }
  /** Dragging a block: `to` = the block it is dropped on (-1 = the box itself: to the end). */
  function dragTarget(to) {
    const over = (event) => {
      if (dragFrom < 0) return;
      event.preventDefault();
      event.stopPropagation();
    };
    const drop = (event) => {
      if (dragFrom < 0) return;
      over(event);
      const from = dragFrom;
      dragFrom = -1;
      move(from, to < 0 ? read().length - 1 : to);
    };
    return { onDragover: over, onDrop: drop };
  }
  function chip(value, index, items) {
    const text = labelOf(value, items);
    const ok = known(value);
    return h("span", {
      class: `ais-chip${ok ? "" : " warn"}`, role: "listitem", draggable: true, title: ok ? value : `${value || "빈 값"} — 목록에 없음`,
      onDragstart: (event) => {
        dragFrom = index;
        event.stopPropagation();
        if (event.dataTransfer) {
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", value);
        }
      },
      onDragend: () => { dragFrom = -1; },
      ...dragTarget(index),
    },
    h("span", { class: "ais-chip-text" }, text),
    h("button", { type: "button", class: "ais-chip-x", "aria-label": `빼기: ${text}`, title: "빼기 · Alt+←/→ 또는 끌어서 순서 바꾸기",
      onClick: () => remove(index), onKeydown: (event) => chipKey(event, index) }, "✕"));
  }

  function draw() {
    const values = read();
    const { items, loaded } = list();
    box.replaceChildren(...(values.length
      ? values.map((value, index) => chip(value, index, items))
      : [h("span", { class: "ais-chip-none" }, "넣은 값 없음 — 아래에서 넣기")]));
    grey(allBtn, !loaded || !items.length || (values.length === items.length && items.every((item, index) => item.value === values[index])));
    grey(clearBtn, !values.length);
    if (!menu.hidden) drawMenu();
  }

  function drawMenu() {
    const values = read();
    const { items, loaded } = list();
    shown = menuItems(items, values, search.value);
    cursor = Math.max(0, Math.min(cursor, shown.length - 1));
    const why = !loaded ? "목록 읽는 중…" : !items.length ? "목록 없음" : menuItems(items, values, "").length ? "맞는 값이 없어요" : "모든 값을 넣었어요";
    menu.replaceChildren(...(shown.length
      ? shown.map((item, index) => h("button", {
        type: "button", class: `ais-chip-opt${index === cursor ? " cur" : ""}`, role: "option", tabindex: "-1", title: item.value,
        onClick: () => add(item.value),
      }, item.label))
      : [h("span", { class: "ais-chip-empty" }, why)]));
  }
  function setCursor(next) {
    const options = menu.querySelectorAll(".ais-chip-opt");
    cursor = Math.max(0, Math.min(next, options.length - 1));
    options.forEach((option, index) => option.classList.toggle("cur", index === cursor));
    const at = options[cursor];
    if (!at?.offsetHeight) return;
    if (at.offsetTop < menu.scrollTop) menu.scrollTop = at.offsetTop;
    else if (at.offsetTop + at.offsetHeight > menu.scrollTop + menu.clientHeight) menu.scrollTop = at.offsetTop + at.offsetHeight - menu.clientHeight;
  }
  function openMenu() {
    if (!menu.hidden) return;
    menu.hidden = false;
    search.setAttribute("aria-expanded", "true");
    cursor = 0;
    opened?.();
    drawMenu();
    fitInColumn(el, menu);
  }
  function closeMenu() {
    if (menu.hidden) return;
    menu.hidden = true;
    search.setAttribute("aria-expanded", "false");
    menu.replaceChildren();
  }

  function searchKey(event) {
    if (event.isComposing || event.keyCode === 229) return; // Korean IME composing: Enter / arrows belong to it
    const { key } = event;
    const open = !menu.hidden;
    if (key === "ArrowDown" || key === "ArrowUp") {
      if (open) setCursor(cursor + (key === "ArrowDown" ? 1 : -1));
      else openMenu();
    } else if (key === "Enter" && open && shown.length) add(shown[cursor].value, true);
    else if (key === "Escape" && open) closeMenu();
    else if (key === "Backspace" && search.value === "" && read().length) {
      if (!event.repeat) change(read().slice(0, -1)); // ⌫ held to empty the box must not go on into the blocks
    } else return;
    event.preventDefault();
    event.stopPropagation(); // not a ComfyUI shortcut
  }
  search.addEventListener("click", openMenu); // a click, not the focus: ✕ on the last block moves the focus here
  search.addEventListener("input", () => {
    cursor = 0;
    if (menu.hidden) openMenu();
    else drawMenu();
  });
  search.addEventListener("keydown", searchKey);
  search.addEventListener("blur", closeMenu);
  menu.addEventListener("mousedown", (event) => event.preventDefault()); // a press on the list keeps the focus in the add box
  const onBox = dragTarget(-1);
  box.addEventListener("dragover", (event) => { if (event.target === box) onBox.onDragover(event); });
  box.addEventListener("drop", (event) => { if (event.target === box) onBox.onDrop(event); });

  draw();
  return { el, refresh: draw };
}
