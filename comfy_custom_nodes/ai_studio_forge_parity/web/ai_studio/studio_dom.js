// Anima studio: tiny DOM helpers and the C4 design's inline SVG icons, shared by the three views.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).

/** `h("div", {class, style, onClick, value, title, ...}, ...children)`; false / null / undefined props and children are skipped. */
export function h(tag, props = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") element.className = value;
    else if (key === "style") element.style.cssText = value;
    else if (key.startsWith("on") && typeof value === "function") element.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in element && !key.includes("-")) element[key] = value;
    else element.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : String(child));
  }
  return element;
}

const SVG_HEAD = 'width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ececec" stroke-width="2"';
const ICON_SVG = {
  folder: `<svg ${SVG_HEAD} stroke-linejoin="round" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>`,
  i2i: `<svg ${SVG_HEAD} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="12" height="13" rx="2"/><path d="M3 15l4-4 4 4M15 11h6M18 8l3 3-3 3"/></svg>`,
  inpaint: `<svg ${SVG_HEAD} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 4l5 5-9 9H6v-5z"/><path d="M4 21h8"/></svg>`,
  upscale: `<svg ${SVG_HEAD} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 14v6h6M20 10V4h-6M4 20l7-7M20 4l-7 7"/></svg>`,
  hires: `<svg ${SVG_HEAD} stroke-linejoin="round" aria-hidden="true"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 16l.7 1.8 1.8.7-1.8.7L19 21l-.7-1.8-1.8-.7 1.8-.7z"/></svg>`,
  sam3: `<svg ${SVG_HEAD} stroke-linecap="round" aria-hidden="true"><path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/><circle cx="12" cy="12" r="3.5"/></svg>`,
  lora_preview: `<svg ${SVG_HEAD} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M3 17l5-4 4 3 3-2 6 4M17 8h4M19 6v4"/></svg>`,
  seed: `<svg ${SVG_HEAD} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14-4.5M4 4v4h4M4 13a8 8 0 0 0 14 4.5M20 20v-4h-4"/></svg>`,
};

/** An `<svg>` element of the C4 icon set (static markup, no user text). */
export function icon(name) {
  const template = document.createElement("template");
  template.innerHTML = ICON_SVG[name];
  return template.content.firstElementChild;
}

/**
 * A 켬/끔 pill button (`.on` / `.off` of the design). `disabled` greys it with aria-disabled instead of the
 * `disabled` attribute, so the mouse-over `title` (the reason) still shows. `focusKey` names it across a redraw of
 * the panel (PanelView.render gives the focus back by it), so a keyboard toggle can be pressed again.
 */
export function pill(on, onToggle, { on: onText = "켬", off: offText = "끔", title, disabled, focusKey } = {}) {
  return h(
    "button",
    {
      type: "button",
      class: `ais-pill ${on ? "on" : "off"} ${disabled ? "dis" : ""}`,
      title,
      "aria-disabled": disabled ? "true" : undefined,
      "aria-pressed": on ? "true" : "false",
      "data-focus-key": focusKey,
      onClick: disabled ? undefined : onToggle,
    },
    on ? onText : offText,
  );
}

/** `[{value, label}]` from route items (plain strings or `{value, label}`); the current value stays listed even if gone. */
// Grow a textarea to fit its text (CSS min-height sets the floor), keeping the panel's scroll position.
export function autoGrow(el) {
  const box = el.closest?.(".ais-scroll");
  const top = box?.scrollTop;
  el.style.height = "auto";
  const height = el.scrollHeight;
  el.style.height = height ? `${height + el.offsetHeight - el.clientHeight}px` : "";
  if (box) box.scrollTop = top;
}

/**
 * Draws `root` again with `draw()` and gives its scroll boxes back where they were. A redraw builds every box anew, so each
 * would start at the top: a box marked `data-scroll="<key>"` takes the offsets of the old box with the same key (put the
 * content's identity in the key, so other content starts at the top). `only`: the keys to keep (new content: the rest start at 0).
 */
export function keepScroll(root, draw, { only } = {}) {
  const boxes = () => [...root.querySelectorAll("[data-scroll]")];
  const keyOf = (box) => box.getAttribute("data-scroll");
  const saved = new Map(boxes().filter((box) => !only || only.includes(keyOf(box))).map((box) => [keyOf(box), [box.scrollTop, box.scrollLeft]]));
  draw();
  for (const box of boxes()) { // document order: an outer box before the boxes inside it
    const [top, left] = saved.get(keyOf(box)) || [0, 0];
    if (top) box.scrollTop = top;
    if (left) box.scrollLeft = left;
  }
}

// ComfyUI's shortcuts for the canvas (frontend 1.58 defaults: delete, delete, fit view, pin, collapse, bypass, mute, select
// all, paste with links, zoom), named as ComfyUI names its combos. It runs them on the selected nodes when the pressed
// element is inside the canvas container and is not a text box. A node's DOM widget is inside it and a click in the widget
// selects the node, so Delete on a focused button of a panel would delete that panel. (result_view.js keeps its own copy.)
export const CANVAS_KEYS = new Set(["delete", "backspace", ".", "p", "alt+c", "ctrl+b", "ctrl+m", "ctrl+a", "ctrl+shift+v", "alt+=", "alt+shift++", "alt++", "alt+-"]);
const TEXT_TAGS = new Set(["INPUT", "TEXTAREA"]); // ComfyUI leaves the plain keys to these; a list (<select>) is not one

/** Whether `event` is a canvas shortcut pressed on something that is not a text box (a button, a switch, a list, ...). */
export function isCanvasKey(event) {
  const target = event?.target;
  if (target && (TEXT_TAGS.has(String(target.tagName || "").toUpperCase()) || target.isContentEditable === true)) return false;
  const combo = `${event.ctrlKey || event.metaKey ? "ctrl+" : ""}${event.altKey ? "alt+" : ""}${event.shiftKey ? "shift+" : ""}${String(event.key ?? "").toLowerCase()}`;
  return CANVAS_KEYS.has(combo);
}

/**
 * Keeps the canvas's shortcuts pressed inside `element` (a panel's root) from reaching ComfyUI. Only the way out is closed:
 * the key is not prevented (P in a list still jumps to an option), the listeners inside `element` have already run, text
 * boxes keep every key, and ComfyUI's other shortcuts (Ctrl+Enter, ...) pass. Returns `element`.
 */
export function guardCanvasKeys(element) {
  element.addEventListener("keydown", (event) => {
    if (isCanvasKey(event)) event.stopPropagation();
  });
  return element;
}

export function optionItems(items, current, { empty, loaded = false } = {}) {
  const list = (items || []).map((item) => (typeof item === "string" ? { value: item, label: item } : item));
  const known = loaded || list.length > 0; // an empty list is still loading (or the route failed): no "(없음)" mark
  if (empty !== undefined) list.unshift({ value: "", label: empty });
  if (current !== undefined && current !== null && !list.some((item) => item.value === current)) {
    list.push({ value: current, label: current === "" ? "-" : known ? `${current} (없음)` : current });
  }
  return list;
}

const optionTag = (item) => h("option", { value: item.value, title: item.label }, item.label);

/** `<option>`s; neighbours with the same `group` share an `<optgroup>`. */
export function optionTags(items) {
  const tags = [];
  for (const item of items) {
    const last = tags[tags.length - 1];
    if (!item.group) tags.push(optionTag(item));
    else if (last?.label === item.group) last.appendChild(optionTag(item));
    else tags.push(h("optgroup", { label: item.group }, optionTag(item)));
  }
  return tags;
}

export function select(items, value, onChange, props = {}) {
  const element = h("select", { class: "ais-f", ...props, onChange: (event) => onChange(event.target.value) }, optionTags(items));
  element.value = value;
  return element;
}

/** A number box; `onChange` gets a finite number (an empty or bad entry is ignored). `live`: on every key, not only when it is left. */
export function numberBox(value, onChange, { min, max, step = "any", width, title, live = false, disabled = false } = {}) {
  const write = (event) => {
    const number = Number(event.target.value);
    if (event.target.value !== "" && Number.isFinite(number)) onChange(number);
  };
  return h("input", {
    type: "number",
    class: "ais-f ais-num",
    value: String(value),
    min,
    max,
    step,
    title,
    disabled: disabled || undefined,
    style: width ? `width:${width}px` : undefined,
    onChange: write,
    onInput: live ? write : undefined,
  });
}

export const card = (title, head, body, extraClass = "") =>
  h("div", { class: `ais-node ${extraClass}` }, h("div", { class: "ais-nt" }, h("span", {}, title), head), h("div", { class: "ais-nb" }, body));

export const row = (label, ...controls) => h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, label), h("span", { class: "ais-ctl" }, controls));

/** `{filename, subfolder, type}` from a LoadImage value such as `sub/name.png [output]` (default type input). */
export function refFromValue(value) {
  const match = /^(.*?)(?: \[(input|output|temp)\])?$/.exec(String(value || "").trim());
  const path = match[1];
  const cut = path.lastIndexOf("/");
  return { filename: path.slice(cut + 1), subfolder: cut < 0 ? "" : path.slice(0, cut), type: match[2] || "input" };
}

export const refKey = (ref) => `${ref.type}/${ref.subfolder || ""}/${ref.filename}`;
