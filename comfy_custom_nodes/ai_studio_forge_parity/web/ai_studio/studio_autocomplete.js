// Anima studio: tag autocomplete for the prompt / negative text boxes (like Forge's tagcomplete). One instance per panel:
// `createAutocomplete({fetchTags, loras})` -> `attach(textarea)` after each render, `close()`, `dispose()`.
// Typing asks `GET /aistudio/studio/tags` (through `fetchTags(query, limit)`, 150ms debounce, a numbered request so a late answer
// for an older query is dropped); after `<lora:` the panel's own LoRA list is searched instead and `<lora:stem:1>` is inserted.
// The popup lives on document.body (the canvas transforms everything inside a node's DOM widget) at the caret's line.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).

const HANGUL = /[ㄱ-ㆎ가-힣]/;
const SEPARATOR = /[,\n]/;
const WEIGHT_SUFFIX = /:-?\d*\.?\d+$/;
const LORA_QUERY = /^<lora:([^:>]*)$/i;
const RESULT_LIMIT = 12;
const POPUP_Z = 99990; // above the DOM widgets, below the lightbox (100000)

const isSpace = (ch) => ch === " " || ch === "\t" || ch === "\r" || ch === "　";
const clampCaret = (text, caret) => Math.max(0, Math.min(text.length, Number.isFinite(caret) ? caret : text.length));
const normalize = (text) => String(text ?? "").toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ");

export const hasHangul = (text) => HANGUL.test(text || "");
/** Letters needed before suggestions start: 2, or 1 for Hangul (one syllable already narrows a lot). */
export const minQueryLength = (query) => (hasHangul(query) ? 1 : 2);

/**
 * The tag being typed at `caret`: `{start, end, query}`. A tag is bounded by `,` and newlines; `start` skips spaces and the
 * `(` / `[` that open a weight, `end` is where the tag name stops (a `:1.2)` weight suffix, closing brackets and trailing
 * spaces are not part of it), `query` is the text from `start` to the caret. A caret outside the name (in the weight) gives
 * an empty query at the caret.
 */
export function tokenAtCaret(text, caret) {
  const src = String(text ?? "");
  const pos = clampCaret(src, caret);
  let start = pos;
  while (start > 0 && !SEPARATOR.test(src[start - 1])) start--;
  let stop = pos;
  while (stop < src.length && !SEPARATOR.test(src[stop])) stop++;
  let openers = 0;
  while (start < pos && (isSpace(src[start]) || src[start] === "(" || src[start] === "[")) {
    if (!isSpace(src[start])) openers++;
    start++;
  }
  while (stop > start && isSpace(src[stop - 1])) stop--;
  if (openers > 0) {
    let closers = 0;
    while (closers < openers && stop > start && (src[stop - 1] === ")" || src[stop - 1] === "]")) {
      stop--;
      closers++;
    }
    const weight = closers ? WEIGHT_SUFFIX.exec(src.slice(start, stop)) : null; // "(blue_hair:1.2)": the ":1.2" and ")" are not the name
    if (weight) stop -= weight[0].length;
  }
  if (pos > stop) return { start: pos, end: pos, query: "" };
  return { start, end: stop, query: src.slice(start, pos) };
}

// Face / eye emoticon tags (o_o, ^_^, >_<): every part between the underscores is one character; they keep their "_".
const isEmoticon = (name) => name.includes("_") && name.split("_").every((part) => part.length <= 1);

/** A tag name as it is written in a prompt: `_` as a space (except emoticons), `(` `)` escaped so they are not weights. */
export function formatTag(name) {
  const spaced = isEmoticon(name) ? name : name.replace(/_/g, " ");
  return spaced.replace(/[()]/g, "\\$&");
}

/**
 * `text` with `token` (from tokenAtCaret) replaced by `tag`, and the caret after it: `{text, caret}`. At the end of the text
 * `, ` is added so the next tag can be typed at once; before a `,` / newline / weight suffix nothing is added. A tag right
 * after a comma gets a space first. `verbatim` inserts `tag` as given (a `<lora:...>` tag).
 */
export function insertTag(text, token, tag, { verbatim = false } = {}) {
  const src = String(text ?? "");
  const start = Math.max(0, Math.min(src.length, token.start));
  const end = Math.max(start, Math.min(src.length, token.end));
  const name = verbatim ? tag : formatTag(tag);
  let before = src.slice(0, start);
  const after = src.slice(end);
  if (before.endsWith(",")) before += " ";
  const rest = after.replace(/^[ \t\r　]+/, "");
  if (!rest) {
    const next = `${before}${name}, `;
    return { text: next, caret: next.length };
  }
  const caret = before.length + name.length;
  if (SEPARATOR.test(rest[0]) || ":)]".includes(rest[0])) return { text: before + name + after, caret };
  return { text: `${before}${name}, ${rest}`, caret: caret + 2 };
}

// ------------------------------------------------------------------------------------------------- candidates

/** The LoRA name typed after `<lora:` in a token's query (`""` right after the colon), or null when it is not a lora tag. */
export function loraQuery(query) {
  const match = LORA_QUERY.exec(String(query ?? ""));
  return match ? match[1] : null;
}

const stemOf = (entry) => entry.replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "");
const folderOf = (entry) => entry.replace(/\\/g, "/").split("/").slice(0, -1).join("/");

/** LoRA candidates for `name` from the ComfyUI file names: stem prefix, then stem contains, then folder contains. */
export function matchLoras(entries, name, limit = RESULT_LIMIT) {
  const wanted = normalize(name).trim();
  const seen = new Set();
  const ranked = [];
  for (const entry of entries || []) {
    const stem = stemOf(String(entry));
    if (!stem || seen.has(stem)) continue;
    const key = normalize(stem);
    const tier = !wanted || key.startsWith(wanted) ? 0 : key.includes(wanted) ? 1 : normalize(folderOf(String(entry))).includes(wanted) ? 2 : -1;
    if (tier < 0) continue;
    seen.add(stem);
    ranked.push({ tier, stem, folder: folderOf(String(entry)) });
  }
  ranked.sort((a, b) => a.tier - b.tier || a.stem.localeCompare(b.stem));
  return ranked.slice(0, limit).map(({ stem, folder }) => ({
    kind: "lora", name: stem, hint: folder, count: 0, category: "lora", insert: `<lora:${stem}:1>`, verbatim: true,
  }));
}

/** A popup row from a `/tags` item `{name, category, count, alias?, ko?}`. */
export function tagCandidate(item) {
  const hint = [item.alias ? `← ${item.alias}` : "", item.ko || ""].filter(Boolean).join(" · ");
  return {
    kind: "tag", name: item.name, alias: item.alias || "", ko: item.ko || "", hint,
    count: Number(item.count) || 0, category: Number(item.category) || 0, insert: item.name, verbatim: false,
  };
}

/** The rows that still match a longer `query` (what is on screen while the answer for it is on its way). */
export function narrow(candidates, query) {
  const lora = loraQuery(query);
  const wanted = normalize(query).trim();
  return candidates.filter((item) => {
    if (item.kind === "lora") return lora !== null && normalize(item.name).includes(normalize(lora).trim());
    if (lora !== null || !wanted) return false;
    if (hasHangul(wanted)) return normalize(item.ko).includes(wanted);
    return normalize(item.name).includes(wanted) || (item.alias && normalize(item.alias).includes(wanted));
  });
}

/** 1234 -> "1.2k", 56789 -> "57k", 1234567 -> "1.2M" (the popup shows how common a tag is). */
export function formatCount(count) {
  const n = Number(count) || 0;
  const short = (value, unit) => `${value < 10 ? value.toFixed(1).replace(/\.0$/, "") : Math.round(value)}${unit}`;
  if (n >= 999500) return short(n / 1e6, "M");
  if (n >= 1e3) return short(n / 1e3, "k");
  return String(n);
}

// ------------------------------------------------------------------------------------------------- requests

/**
 * Turns "the query changed" into candidates, debounced and with stale answers dropped. `ask(query)` ends in exactly one
 * `onItems(candidates, query)` unless a newer `ask` / `cancel` came first. A LoRA query is answered at once from `loras()`.
 */
export class Suggester {
  constructor({ fetchTags, loras = () => [], onItems, delay = 150, limit = RESULT_LIMIT, setTimer = (run, ms) => setTimeout(run, ms), clearTimer = (id) => clearTimeout(id) }) {
    Object.assign(this, { fetchTags, loras, onItems, delay, limit, setTimer, clearTimer });
    this.seq = 0;
    this.timer = null;
  }

  /** Forget the pending request and any answer still on its way. */
  cancel() {
    this.seq++;
    if (this.timer !== null) this.clearTimer(this.timer);
    this.timer = null;
  }

  ask(query) {
    this.cancel();
    const seq = this.seq;
    const lora = loraQuery(query);
    if (lora !== null) return this.onItems(matchLoras(this.loras(), lora, this.limit), query);
    const text = String(query ?? "").trim();
    if (text.startsWith("<") || text.length < minQueryLength(text) || !this.fetchTags) return this.onItems([], query);
    this.timer = this.setTimer(() => {
      this.timer = null;
      Promise.resolve()
        .then(() => this.fetchTags(text, this.limit))
        .then((items) => (Array.isArray(items) ? items : []), () => [])
        .then((items) => {
          if (seq === this.seq) this.onItems(items.map(tagCandidate), query);
        });
    }, this.delay);
  }
}

// ------------------------------------------------------------------------------------------------- caret position

const MIRRORED_STYLES = [
  "direction", "boxSizing", "width", "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
  "borderStyle", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "fontStyle", "fontVariant", "fontWeight",
  "fontStretch", "fontSize", "fontFamily", "lineHeight", "textAlign", "textTransform", "textIndent", "letterSpacing",
  "wordSpacing", "tabSize", "wordBreak", "overflowWrap",
];

/**
 * Where the caret is on screen: `{left, top, bottom}` (its line) in viewport pixels, measured by laying the text up to the caret out in
 * an invisible copy of the box. Null when that cannot be measured (the popup then goes under the box).
 */
export function caretPoint(box, pos, doc, win) {
  try {
    const style = win.getComputedStyle(box);
    const mirror = doc.createElement("div");
    for (const name of MIRRORED_STYLES) mirror.style[name] = style[name];
    Object.assign(mirror.style, { position: "absolute", left: "-9999px", top: "0", visibility: "hidden", whiteSpace: "pre-wrap", overflow: "hidden", height: "auto" });
    mirror.textContent = box.value.slice(0, pos);
    const mark = doc.createElement("span");
    mark.textContent = box.value.slice(pos) || ".";
    mirror.append(mark);
    doc.body.append(mirror);
    const fontSize = parseFloat(style.fontSize) || 13;
    const lineHeight = parseFloat(style.lineHeight) || fontSize * 1.3;
    const markLeft = mark.offsetLeft + (parseFloat(style.borderLeftWidth) || 0);
    const markTop = mark.offsetTop + (parseFloat(style.borderTopWidth) || 0);
    mirror.remove();
    const rect = box.getBoundingClientRect();
    const scale = box.offsetWidth ? rect.width / box.offsetWidth : 1; // the canvas may scale the node
    if (!Number.isFinite(scale) || !Number.isFinite(markTop)) return null;
    const top = rect.top + (markTop - box.scrollTop) * scale;
    return { left: rect.left + (markLeft - box.scrollLeft) * scale, top, bottom: top + lineHeight * scale };
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------------------------------------- popup + keys

/**
 * `fetchTags(query, limit) -> Promise<[{name, category, count, alias?, ko?}]>`, `loras() -> [ComfyUI LoRA file names]`.
 * Keys while the popup is open: ↑ ↓ move, Tab / Enter accept (only then: otherwise they keep their meaning, and Ctrl+Enter
 * still queues), Esc closes. Clicking outside, blur, moving the caret and a re-render close it.
 */
export function createAutocomplete({ fetchTags, loras, delay = 150, doc = globalThis.document, win = globalThis.window, setTimer, clearTimer } = {}) {
  let popup = null;
  let owner = null; // the text box the shown rows belong to
  let asked = null; // {text, start, query} when the shown (or awaited) rows were asked for
  let items = [];
  let index = 0;
  let outsideBound = false; // the document-wide pointerdown listener is on while the popup is open
  let disposed = false;

  const suggester = new Suggester({ fetchTags, loras, delay, setTimer, clearTimer, onItems: (rows, query) => show(rows, query) });
  const isOpen = () => items.length > 0 && popup !== null && !popup.hidden;

  function make(tag, className, text) {
    const element = doc.createElement(tag);
    element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function ensurePopup() {
    if (popup) return popup;
    popup = make("div", "ais-ac");
    popup.setAttribute("role", "listbox");
    popup.style.zIndex = String(POPUP_Z);
    popup.hidden = true;
    popup.addEventListener("mousedown", (event) => event.preventDefault()); // keep the focus in the text box
    doc.body.append(popup);
    return popup;
  }

  function select(next) {
    index = ((next % items.length) + items.length) % items.length;
    Array.from(popup.children).forEach((row, at) => {
      row.className = at === index ? "ais-ac-row sel" : "ais-ac-row";
      row.setAttribute("aria-selected", at === index ? "true" : "false");
    });
    popup.children[index]?.scrollIntoView?.({ block: "nearest" });
  }

  function render() {
    popup.replaceChildren(...items.map((item, at) => {
      const row = make("div", "ais-ac-row");
      row.setAttribute("role", "option");
      row.append(
        make("span", `ais-ac-name ais-ac-c${item.category}`, item.name),
        make("span", "ais-ac-hint", item.hint),
        make("span", "ais-ac-count", item.count ? formatCount(item.count) : ""),
      );
      row.addEventListener("mousedown", (event) => {
        event.preventDefault();
        accept(at);
      });
      row.addEventListener("mousemove", () => { if (at !== index) select(at); });
      return row;
    }));
    select(Math.min(index, items.length - 1));
  }

  function place() {
    const rect = owner.getBoundingClientRect();
    const caret = caretPoint(owner, owner.selectionStart ?? owner.value.length, doc, win);
    const left = caret ? caret.left : rect.left;
    const top = (caret ? caret.bottom : rect.bottom) + 2;
    const width = popup.offsetWidth || 0;
    const height = popup.offsetHeight || 0;
    const viewWidth = win?.innerWidth || 0;
    const viewHeight = win?.innerHeight || 0;
    popup.style.left = `${Math.max(4, viewWidth ? Math.min(left, viewWidth - width - 8) : left)}px`;
    const above = (caret ? caret.top : rect.top) - height - 2; // no room below: over the line instead
    popup.style.top = `${viewHeight && top + height > viewHeight && above >= 0 ? above : top}px`;
  }

  function show(rows, query) {
    if (!owner || !asked || asked.query !== query || rows.length === 0) return hide();
    items = rows;
    ensurePopup();
    popup.hidden = false;
    index = 0;
    render();
    place();
    if (!outsideBound) {
      doc.addEventListener("pointerdown", onOutside, true);
      outsideBound = true;
    }
    owner.setAttribute?.("aria-expanded", "true");
  }

  function hide() {
    items = [];
    index = 0;
    if (popup) popup.hidden = true;
    if (outsideBound) {
      doc.removeEventListener("pointerdown", onOutside, true);
      outsideBound = false;
    }
    owner?.setAttribute?.("aria-expanded", "false");
  }

  function close() {
    suggester.cancel();
    hide();
    owner = null;
    asked = null;
  }

  function accept(at = index) {
    const item = items[at];
    const box = owner;
    if (!item || !box || !asked || box.value !== asked.text) return false;
    const token = tokenAtCaret(box.value, box.selectionStart);
    if (token.start !== asked.start) return false;
    const next = insertTag(box.value, token, item.insert, { verbatim: item.verbatim });
    box.value = next.text;
    box.setSelectionRange(next.caret, next.caret);
    const EventClass = win?.Event || globalThis.Event;
    box.dispatchEvent(new EventClass("input", { bubbles: true })); // the panel saves the text and grows the box
    close(); // after the event: whatever our own input handler just asked for is dropped
    return true;
  }

  function onOutside(event) {
    if (event.target === owner || popup?.contains?.(event.target)) return;
    close();
  }

  function onInput(event) {
    if (disposed) return;
    const box = event.target;
    if (owner !== box) hide();
    owner = box;
    const token = tokenAtCaret(box.value, box.selectionStart);
    asked = { text: box.value, start: token.start, query: token.query };
    if (items.length) {
      // what is on screen answers the old query: keep only the rows that fit the new one, so Tab / Enter cannot take a stale tag
      items = narrow(items, token.query);
      if (items.length) render();
      else hide();
    }
    suggester.ask(token.query);
  }

  const CARET_KEYS = ["Home", "End", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"];

  function onKeyDown(event) {
    if (!isOpen() || event.isComposing || event.keyCode === 229) return;
    const plain = !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey;
    const consume = () => { event.preventDefault(); event.stopPropagation(); };
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!plain) return;
      consume();
      select(index + (event.key === "ArrowDown" ? 1 : -1));
    } else if (event.key === "Tab" || event.key === "Enter") {
      if (!plain) return;
      if (accept()) consume();
      else close();
    } else if (event.key === "Escape") {
      consume();
      close();
    } else if (CARET_KEYS.includes(event.key)) {
      close();
    }
  }

  return {
    /** Wire a (freshly rendered) text box. */
    attach(box) {
      if (!box?.addEventListener || !doc?.body || disposed) return;
      box.setAttribute?.("autocomplete", "off");
      box.setAttribute?.("aria-autocomplete", "list");
      box.addEventListener("input", onInput);
      box.addEventListener("keydown", onKeyDown);
      box.addEventListener("blur", close);
      box.addEventListener("click", close);
    },
    close,
    isOpen,
    /** Remove the popup (the panel node was deleted); text boxes still attached stop asking. */
    dispose() {
      disposed = true;
      close();
      popup?.remove();
      popup = null;
    },
  };
}
