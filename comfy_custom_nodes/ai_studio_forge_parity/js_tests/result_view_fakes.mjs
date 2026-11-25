// Test helpers for the result column's tests (not a test file itself: the runner takes *.test.mjs): a fake document that
// bubbles events, moves the focus and lays the gallery strip out, and a ResultView set up on it. Importing it installs
// `ResizeObserver`, `Node` and `window`; `setup()` installs a new `document` each time.
import { ResultView } from "../web/ai_studio/result_view.js";

// The strip as studio.css draws it: 62px thumbnails, 8px apart, 12px of padding at both ends.
export const THUMB = 62;
export const GAP = 8;
export const PAD = 12;
export const layout = { on: true, view: 300, parent: "strip" }; // `parent`: what a thumbnail's offsetLeft is measured from
export const STRIP_AT = 40; // the strip's own offsetLeft when both are measured from the node ("root")
export const stripWidth = (count) => Math.max(layout.view, 2 * PAD + count * THUMB + Math.max(0, count - 1) * GAP);
export const thumbLeft = (index) => PAD + index * (THUMB + GAP);

export class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parent = null;
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.textContent = "";
    this.scrollLeft = 0;
    this.scrolls = []; // scrollTo() calls: {left, behavior, from}
    this.focused = []; // focus() options
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node instanceof FakeElement) node.parent = this;
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) {
    for (const child of this.children) if (child instanceof FakeElement) child.parent = null;
    this.children = [];
    this.append(...nodes);
  }
  /** Leaving the page with the focus inside drops the focus to the page, as in a browser. */
  remove() {
    const doc = globalThis.document;
    if (this.isConnected && doc.activeElement && this.contains(doc.activeElement)) doc.activeElement = doc.body;
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }
  get isConnected() { return globalThis.document.body.contains(this); }
  contains(node) {
    for (let at = node; at; at = at.parent) if (at === this) return true;
    return false;
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  /** Dispatch with bubbling up the parents (until a listener calls stopPropagation); returns the event. */
  fire(type, extra = {}) {
    const event = {
      type, target: this, defaultPrevented: false, stopped: false,
      preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this.stopped = true; },
      ...extra,
    };
    for (let node = this; node && !event.stopped; node = node.parent) for (const fn of node.listeners[type] || []) fn(event);
    return event;
  }
  focus(options) {
    this.focused.push(options);
    globalThis.document.activeElement = this;
  }
  scrollTo(options) {
    this.scrolls.push({ ...options, from: this.scrollLeft });
    this.scrollLeft = options.left;
  }
  scrollIntoView() {
    throw new Error("scrollIntoView can move the page or the panel: the strip is scrolled by scrollLeft only");
  }
  has(name) { return this.className.split(" ").includes(name); }
  // --- layout: only the strip and its thumbnails have one
  get index() { return this.parent ? this.parent.children.indexOf(this) : -1; }
  get offsetParent() {
    if (!layout.on) return null;
    if (this.has("ais-thumb")) return layout.parent === "strip" ? this.parent : this.parent?.parent ?? null;
    return this.parent;
  }
  get offsetLeft() {
    if (!layout.on) return 0;
    if (this.has("ais-thumb")) return thumbLeft(this.index) + (layout.parent === "strip" ? 0 : STRIP_AT);
    return this.has("ais-gallery") ? STRIP_AT : 0;
  }
  get offsetWidth() { return layout.on && this.has("ais-thumb") ? THUMB : 0; }
  get clientWidth() { return layout.on && this.has("ais-gallery") ? layout.view : 0; }
  get scrollWidth() { return layout.on && this.has("ais-gallery") ? stripWidth(this.children.length) : 0; }
  get content() { // icon(): a <template>'s parsed markup
    return { firstElementChild: new FakeElement("svg") };
  }
  get text() {
    return this.children.map((child) => (typeof child === "string" ? child : child.text)).join("");
  }
  all(accept = () => true) {
    const own = accept(this) ? [this] : [];
    return own.concat(this.children.flatMap((child) => (typeof child === "string" ? [] : child.all(accept))));
  }
  byClass(name) { return this.all((element) => element.has(name)); }
}

export function fakeDocument() {
  const doc = {
    activeElement: null,
    body: new FakeElement("body"),
    head: new FakeElement("head"),
    listeners: [],
    createElement: (tag) => new FakeElement(tag),
    getElementById: (id) => doc.head.children.find((child) => child.id === id) || null,
    addEventListener: (type, fn, capture) => doc.listeners.push({ type, fn, capture }),
    removeEventListener: (type, fn, capture) => {
      doc.listeners = doc.listeners.filter((l) => !(l.type === type && l.fn === fn && l.capture === capture));
    },
  };
  return doc;
}

export const observers = [];
globalThis.ResizeObserver = class {
  constructor(callback) {
    this.callback = callback;
    this.targets = [];
    observers.push(this);
  }
  observe(target) { this.targets.push(target); }
  disconnect() { this.targets = []; }
};
globalThis.Node = FakeElement;
globalThis.window = { listeners: [], addEventListener: (type) => globalThis.window.listeners.push(type) };

export const ref = (filename) => ({ filename, subfolder: "", type: "output" });
export const item = (filename) => ({ image: ref(filename), record: null });
export const names = (count) => Array.from({ length: count }, (_, i) => `${i}.png`);
/** A saved view of `count` images, `selected` picked. */
export const saved = (count, selected = 0, more = {}) => ({ gallery: names(count).map(ref), selected, ...more });

export function setup(state = {}, { laidOut = true, parent = "strip", view = 300 } = {}) {
  Object.assign(layout, { on: laidOut, parent, view });
  globalThis.document = fakeDocument();
  globalThis.matchMedia = () => ({ matches: false });
  observers.length = 0;
  const widget = { value: JSON.stringify(state) };
  const picks = [];
  const result = new ResultView(widget, {
    imageUrl: (image) => `/view/${image.filename}`,
    action: () => {},
    selected: (image) => picks.push(image && image.filename),
  });
  globalThis.document.body.append(result.element);
  return { view: result, widget, picks };
}

export const strip = (view) => view.element.byClass("ais-gallery")[0];
export const thumbsOf = (view) => view.element.byClass("ais-thumb");
export const picked = (view) => view.element.byClass("sel").map((thumb) => thumb.attrs.title);
export const stageImage = (view) => view.element.byClass("ais-stage-img")[0]?.attrs.src;
export const pillNamed = (view, text) => view.element.byClass("ais-pill").find((pill) => pill.text === text);
/** Press `key` with the focus on `target` (the strip unless given). */
export const press = (view, key, target = strip(view), extra = {}) => target.fire("keydown", { key, ...extra });
export const images = (view, list, replace = true) => view.onStudioEvent("images", { items: list.map(item), replace });
export const start = (view) => view.onStudioEvent("running", { running: true, kind: "single" });

/** Press `key` on the page, as the enlarged view hears it (its listener is on document, capturing). */
export function pageKey(key) {
  const event = { type: "keydown", key, defaultPrevented: false, stopped: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; } };
  for (const l of [...globalThis.document.listeners]) if (l.type === "keydown") l.fn(event);
  return event;
}
/** The enlarged view on the page (undefined when closed) and its parts. */
export const lightbox = () => globalThis.document.body.byClass("ais-lightbox")[0];
export const lightboxImage = () => lightbox()?.byClass("ais-lightbox-img")[0]?.attrs.src;
/** Click the big image: it opens the enlarged view. */
export const zoom = (view) => view.element.byClass("ais-stage-img")[0].fire("click");
/**
 * A mouse click as a browser does it: the press moves the focus first (to the nearest button or tabindex element under
 * it, or to the page when there is none), then the click is heard.
 */
export function mouseClick(element) {
  let taker = element;
  while (taker && taker.tag !== "button" && !("tabindex" in taker.attrs)) taker = taker.parent;
  globalThis.document.activeElement = taker || globalThis.document.body;
  return element.fire("click");
}
