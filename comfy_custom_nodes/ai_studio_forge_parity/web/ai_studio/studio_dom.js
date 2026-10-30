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
 * `disabled` attribute, so the mouse-over `title` (the reason) still shows.
 */
export function pill(on, onToggle, { on: onText = "켬", off: offText = "끔", title, disabled } = {}) {
  return h(
    "button",
    {
      type: "button",
      class: `ais-pill ${on ? "on" : "off"} ${disabled ? "dis" : ""}`,
      title,
      "aria-disabled": disabled ? "true" : undefined,
      "aria-pressed": on ? "true" : "false",
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

export function optionItems(items, current, { empty } = {}) {
  const list = (items || []).map((item) => (typeof item === "string" ? { value: item, label: item } : item));
  const known = list.length > 0; // an empty list is still loading (or the route failed): no "(없음)" mark
  if (empty !== undefined) list.unshift({ value: "", label: empty });
  if (current !== undefined && current !== null && !list.some((item) => item.value === current)) {
    list.push({ value: current, label: current === "" ? "-" : known ? `${current} (없음)` : current });
  }
  return list;
}

export function select(items, value, onChange, props = {}) {
  const element = h(
    "select",
    { class: "ais-f", ...props, onChange: (event) => onChange(event.target.value) },
    items.map((item) => h("option", { value: item.value, title: item.label }, item.label)),
  );
  element.value = value;
  return element;
}

/** A number box; `onChange` gets a finite number (an empty or bad entry is ignored). */
export function numberBox(value, onChange, { min, max, step = "any", width, title } = {}) {
  return h("input", {
    type: "number",
    class: "ais-f ais-num",
    value: String(value),
    min,
    max,
    step,
    title,
    style: width ? `width:${width}px` : undefined,
    onChange: (event) => {
      const number = Number(event.target.value);
      if (event.target.value !== "" && Number.isFinite(number)) onChange(number);
    },
  });
}

/** `{filename, subfolder, type}` from a LoadImage value such as `sub/name.png [output]` (default type input). */
export function refFromValue(value) {
  const match = /^(.*?)(?: \[(input|output|temp)\])?$/.exec(String(value || "").trim());
  const path = match[1];
  const cut = path.lastIndexOf("/");
  return { filename: path.slice(cut + 1), subfolder: cut < 0 ? "" : path.slice(0, cut), type: match[2] || "input" };
}

export const refKey = (ref) => `${ref.type}/${ref.subfolder || ""}/${ref.filename}`;
