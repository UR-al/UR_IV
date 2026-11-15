// Prompt-box presets: the small DOM pieces the quick list (preset_quick.js) and the preset window (preset_window.js) share —
// their stylesheet (preset.css, added once), static SVG icons and the Tab trap of a dialog.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).

const CSS_ID = "aistudio-preset-css";
const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Adds preset.css to the page once (the way studio_main.js adds studio.css); returns its `<link>`. */
export function addPresetStylesheet() {
  const added = document.getElementById(CSS_ID);
  if (added) return added;
  const link = document.createElement("link");
  link.id = CSS_ID;
  link.rel = "stylesheet";
  link.href = new URL("./preset.css", import.meta.url).href;
  document.head.append(link);
  return link;
}

/**
 * Runs `fn` when preset.css arrives, if it has not yet. A popup drawn before that is laid out without its rules (in the
 * page's flow, far taller), so the place worked out from its size is wrong: it must be placed again.
 */
export function afterPresetStyles(fn) {
  const link = addPresetStylesheet();
  if (!link.sheet) link.addEventListener("load", () => fn(), { once: true });
}

const SVG = (size, body) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">${body}</svg>`;
const ICONS = {
  search: SVG(15, '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>'),
  close: SVG(14, '<path d="M6 6l12 12M18 6L6 18"/>'),
};

/** An `<svg>` of the static markup above (no user text). */
export function presetIcon(name) {
  const template = document.createElement("template");
  template.innerHTML = ICONS[name];
  return template.content.firstElementChild;
}

/** Tab / Shift+Tab stay inside `dialog`: from its last control to the first and back. */
export function trapTab(dialog, event) {
  const list = [...dialog.querySelectorAll(FOCUSABLE)].filter((el) => el.getClientRects().length > 0);
  if (!list.length) return;
  const active = dialog.ownerDocument.activeElement;
  const edge = event.shiftKey ? active === list[0] || active === dialog : active === list[list.length - 1];
  if (!edge && dialog.contains(active)) return;
  event.preventDefault();
  (event.shiftKey ? list[list.length - 1] : list[0]).focus();
}

/** `err` as a short text for a message line. */
export const errText = (err) => String(err?.message || err);
