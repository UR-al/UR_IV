// Test helpers for the prompt-box preset tests (not a test file itself): a small fake document — node has no DOM — and an
// in-memory fake of the preset routes (preset_routes.py). Only what the preset modules use is here.
import { PRESET_SECTIONS } from "../web/ai_studio/preset_logic.js";

const ZERO_RECT = { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
const PROPS = ["id", "title", "type", "value", "placeholder", "disabled", "hidden", "readOnly", "rows", "spellcheck", "autocomplete", "textContent"];

function parseCompound(text) {
  const parts = { tag: null, classes: [], attrs: [], nots: [] };
  let rest = text.trim();
  const take = (regex, use) => {
    const match = regex.exec(rest);
    if (!match) return false;
    use(match);
    rest = rest.slice(match[0].length);
    return true;
  };
  while (rest) {
    const ok = take(/^:not\(([^)]+)\)/, (m) => parts.nots.push(parseCompound(m[1])))
      || take(/^\.([\w-]+)/, (m) => parts.classes.push(m[1]))
      || take(/^\[([\w-]+)(?:="([^"]*)")?\]/, (m) => parts.attrs.push([m[1], m[2]]))
      || take(/^[a-zA-Z][\w-]*/, (m) => { parts.tag = m[0].toLowerCase(); });
    if (!ok) throw new Error(`fake DOM: unsupported selector "${text}"`);
  }
  return parts;
}

function matchesCompound(el, parts) {
  if (parts.tag && el.tag !== parts.tag) return false;
  if (!parts.classes.every((name) => el.classList.contains(name))) return false;
  for (const [name, value] of parts.attrs) {
    const got = el.getAttribute(name);
    if (got === null || (value !== undefined && got !== value)) return false;
  }
  return !parts.nots.some((not) => matchesCompound(el, not));
}

export class FakeElement {
  constructor(tag) {
    this.tag = String(tag).toLowerCase();
    this.children = [];
    this.parent = null;
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.style = { cssText: "" };
    this.rect = null;
    this.offsetWidth = 0;
    this.offsetHeight = 0;
    this.clientWidth = 0;
    this.scrollTop = 0;
    for (const name of PROPS) if (name !== "textContent") this[name] = name === "disabled" || name === "hidden" || name === "readOnly" ? false : "";
    const el = this;
    this.classList = {
      contains: (name) => el.className.split(/\s+/).includes(name),
      add: (name) => { if (!el.classList.contains(name)) el.className = `${el.className} ${name}`.trim(); },
      remove: (name) => { el.className = el.className.split(/\s+/).filter((n) => n && n !== name).join(" "); },
      toggle: (name, force) => {
        const on = force === undefined ? !el.classList.contains(name) : Boolean(force);
        if (on) el.classList.add(name); else el.classList.remove(name);
        return on;
      },
    };
  }

  get ownerDocument() { return globalThis.document; }
  get parentElement() { return this.parent; }
  get isConnected() {
    for (let node = this; node; node = node.parent) if (node === globalThis.document.body || node === globalThis.document.head) return true;
    return false;
  }
  get textContent() { return this.children.map((child) => (child instanceof FakeElement ? child.textContent : child)).join(""); }
  set textContent(value) {
    this.#drop(this.children);
    this.children = value === "" || value == null ? [] : [String(value)];
  }
  set innerHTML(markup) { this.content = { firstElementChild: new FakeElement("svg") }; this.content.firstElementChild.markup = markup; }
  get firstElementChild() { return this.children.find((child) => child instanceof FakeElement) || null; }

  #drop(nodes) {
    const doc = globalThis.document;
    for (const node of nodes) {
      if (!(node instanceof FakeElement)) continue;
      if (doc.activeElement && node.contains(doc.activeElement)) doc.activeElement = doc.body; // a removed focused node blurs
      node.parent = null;
    }
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node instanceof FakeElement) {
        if (node.parent) node.parent.children = node.parent.children.filter((child) => child !== node);
        node.parent = this;
      }
      this.children.push(node);
    }
  }
  appendChild(node) { this.append(node); return node; }
  replaceChildren(...nodes) {
    const old = this.children.filter((child) => !nodes.includes(child));
    this.children = [];
    this.#drop(old);
    this.append(...nodes);
  }
  remove() {
    if (!this.parent) return;
    const parent = this.parent;
    parent.children = parent.children.filter((child) => child !== this);
    parent.#drop([this]);
  }
  contains(node) {
    for (let at = node; at; at = at.parent) if (at === this) return true;
    return false;
  }

  setAttribute(name, value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; }
  getAttribute(name) {
    if (name in this.attrs) return this.attrs[name];
    if (name === "class") return this.className || null;
    if ((name === "disabled" || name === "hidden") && this[name]) return "";
    if (name === "id" || name === "type" || name === "title") return this[name] || null;
    return null;
  }
  hasAttribute(name) { return this.getAttribute(name) !== null; }

  matches(selector) { return selector.split(",").some((part) => matchesCompound(this, parseCompound(part))); }
  closest(selector) {
    for (let node = this; node; node = node.parent) if (node.matches(selector)) return node;
    return null;
  }
  all(accept = () => true) {
    const found = [];
    for (const child of this.children) {
      if (!(child instanceof FakeElement)) continue;
      if (accept(child)) found.push(child);
      found.push(...child.all(accept));
    }
    return found;
  }
  querySelectorAll(selector) { return this.all((el) => el.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  /** Test shorthand: the one element with this class (throws when there is none or more than one). */
  one(name) {
    const found = this.querySelectorAll(`.${name}`);
    if (found.length !== 1) throw new Error(`expected one .${name}, found ${found.length}`);
    return found[0];
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== fn); }
  /** Dispatches like a browser: the document's capture listeners, the target, its ancestors, the document. */
  fire(type, init = {}) {
    const doc = globalThis.document;
    const event = {
      type, target: this, defaultPrevented: false, stopped: false, key: undefined, shiftKey: false, ctrlKey: false, metaKey: false, ...init,
      preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this.stopped = true; },
    };
    const run = (list) => {
      for (const fn of [...list]) {
        if (event.stopped) return;
        fn(event);
      }
    };
    run(doc.listenersOf(type, true));
    for (let node = this; node && !event.stopped; node = node.parent) run(node.listeners[type] || []);
    if (!event.stopped) run(doc.listenersOf(type, false));
    return event;
  }
  click() { return this.disabled ? null : this.fire("click"); }
  /** Test shorthand: set the value as typing does. */
  typeIn(text) { this.value = text; return this.fire("input"); }
  focus() { if (this.isConnected && !this.disabled) globalThis.document.activeElement = this; }
  blur() { if (globalThis.document.activeElement === this) globalThis.document.activeElement = globalThis.document.body; }
  select() {}
  setSelectionRange() {}
  scrollIntoView() {}
  getBoundingClientRect() { return (this.isConnected && this.rect) || ZERO_RECT; } // off the page: all zeros, as in a browser
  getClientRects() {
    for (let node = this; node; node = node.parent) if (node.hidden) return [];
    return this.isConnected ? [ZERO_RECT] : [];
  }
}

/** Installs `document`, `Node` and a window size on globalThis (a fresh document each call); returns the document. */
export function installFakeDom({ width = 1280, height = 800 } = {}) {
  const listeners = [];
  const doc = {
    createElement: (tag) => new FakeElement(tag),
    activeElement: null,
    addEventListener: (type, fn, capture = false) => listeners.push({ type, fn, capture: capture === true || capture?.capture === true }),
    removeEventListener: (type, fn) => {
      const at = listeners.findIndex((item) => item.type === type && item.fn === fn);
      if (at >= 0) listeners.splice(at, 1);
    },
    listenersOf: (type, capture) => listeners.filter((item) => item.type === type && item.capture === capture).map((item) => item.fn),
    listenerCount: () => listeners.length,
    getElementById: (id) => [doc.head, doc.body].flatMap((root) => root.all((el) => el.id === id))[0] || null,
    querySelectorAll: (selector) => doc.body.querySelectorAll(selector),
    querySelector: (selector) => doc.body.querySelector(selector),
  };
  globalThis.document = doc;
  globalThis.Node = FakeElement;
  doc.head = new FakeElement("head");
  doc.body = new FakeElement("body");
  doc.activeElement = doc.body;
  globalThis.innerWidth = width;
  globalThis.innerHeight = height;
  return doc;
}

export const emptyDoc = () => ({ version: 1, sections: Object.fromEntries(PRESET_SECTIONS.map((key) => [key, []])), prefs: { close_after_insert: false } });
export const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
/** Lets every pending promise chain (store queue, window actions) finish. */
export const settle = async () => { for (let i = 0; i < 5; i += 1) await tick(); };

/**
 * A fake `route(path, body)` over an in-memory document (`server.doc`): `calls` lists the requests, `fail` (a message) makes
 * the next answers throw, `broken` makes a read say the file could not be parsed, `hold(server)` keeps one answer back until
 * the returned function is called. `seed(key, name, text)` puts a preset in without a request (as another tab would).
 */
export function fakeServer() {
  const server = { doc: emptyDoc(), calls: [], fail: null, gates: [], seq: 0, broken: false };
  const GONE = "프리셋을 찾지 못했어요 (다른 창에서 지웠을 수 있어요)";
  const limits = { name: 60, text: 20000, count: 200 };
  const answer = (body) => {
    if (server.fail) throw new Error(server.fail);
    const doc = server.doc;
    if (!body) return { doc: structuredClone(doc), limits, broken: server.broken === true };
    const list = doc.sections[body.section];
    let extra = {};
    if (body.op === "add") {
      const id = `id${++server.seq}`;
      list.push({ id, name: body.name, text: body.text, created: 1000 + server.seq, used: 0 });
      extra = { id };
    } else if (body.op === "update") {
      const preset = list.find((p) => p.id === body.id);
      if (!preset) throw new Error(GONE);
      if (body.name !== undefined) preset.name = body.name;
      if (body.text !== undefined) preset.text = body.text;
    } else if (body.op === "delete") {
      const at = list.findIndex((p) => p.id === body.id);
      if (at >= 0) list.splice(at, 1);
    } else if (body.op === "move") {
      const at = list.findIndex((p) => p.id === body.id);
      if (at < 0) throw new Error(GONE);
      list.splice(Math.max(0, Math.min(list.length - 1, at + body.delta)), 0, list.splice(at, 1)[0]);
    } else if (body.op === "use") {
      const preset = list.find((p) => p.id === body.id);
      if (preset) preset.used = 4242;
    } else if (body.op === "insert") {
      if (!list.some((p) => p.id === body.item.id)) list.splice(Math.max(0, Math.min(list.length, body.index)), 0, structuredClone(body.item));
    } else if (body.op === "prefs") Object.assign(doc.prefs, body.prefs);
    else throw new Error(`op must be one of: add, update, delete, move, use, insert, prefs (got ${body.op})`);
    return { doc: structuredClone(doc), limits, ...extra };
  };
  server.route = (path, body) => {
    server.calls.push(body === undefined ? [path] : [path, body]);
    const gate = server.gates.shift();
    return (gate || Promise.resolve()).then(() => answer(body));
  };
  server.seed = (key, name, text, used = 0) => {
    const id = `id${++server.seq}`;
    server.doc.sections[key].push({ id, name, text, created: 1000 + server.seq, used });
    return id;
  };
  server.ops = () => server.calls.filter((call) => call[1]).map((call) => call[1]);
  return server;
}

export function hold(server) {
  let open;
  server.gates.push(new Promise((resolve) => { open = resolve; }));
  return open;
}
