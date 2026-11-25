// A tiny fake document for the pack's node:test files (node has no DOM): enough of an element for studio_dom.js `h()` and the
// views drawn with it. A new element starts scrolled to the top, as in a browser, and `focus()` moves `document.activeElement`.
// Not a test file itself (the runner takes *.test.mjs): import it first, it installs `document` and `Node`.

export class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.parentElement = null;
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.value = "";
    this.hidden = false;
    this.off = false;
    this.draggable = false;
    this.scrollTop = 0; // a new box starts at the top
    this.scrollLeft = 0;
    this.scrollHeight = 0;
    this.clientWidth = 0; // nothing is laid out
    this.clientHeight = 0;
  }
  get disabled() { return this.off; }
  /** A control switched off while it has the focus loses it, as in Chrome: the focus drops to the page. */
  set disabled(off) {
    this.off = Boolean(off);
    if (this.off) this.blur();
  }
  get tagName() { return this.tag.toUpperCase(); }
  get ownerDocument() { return globalThis.document; }
  get isConnected() { return this.parentElement ? this.parentElement.isConnected : this.root === true; }
  get classList() {
    const names = () => this.className.split(" ").filter(Boolean);
    const write = (list) => { this.className = list.join(" "); };
    return {
      contains: (name) => names().includes(name),
      add: (name) => { if (!names().includes(name)) write([...names(), name]); },
      remove: (name) => write(names().filter((item) => item !== name)),
      toggle: (name, on = !names().includes(name)) => (on ? this.classList.add(name) : this.classList.remove(name)),
    };
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node instanceof FakeElement) {
        node.remove();
        node.parentElement = this;
      }
      this.children.push(node);
    }
  }
  remove() {
    const parent = this.parentElement;
    if (!parent) return;
    parent.children = parent.children.filter((child) => child !== this);
    this.parentElement = null;
  }
  replaceChildren(...nodes) {
    for (const child of this.children) if (child instanceof FakeElement) child.parentElement = null;
    this.children = [];
    this.append(...nodes);
  }
  contains(node) {
    for (let at = node; at; at = at.parentElement) if (at === this) return true;
    return false;
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  /** Runs the listeners with an event object; returns it (`defaultPrevented`, `stopped`). */
  fire(type, extra = {}) {
    const event = {
      type, target: this, defaultPrevented: false, stopped: false,
      preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this.stopped = true; },
      ...extra,
    };
    for (const fn of this.listeners[type] || []) fn(event);
    return event;
  }
  focus() {
    const before = globalThis.document.activeElement;
    if (before === this) return;
    globalThis.document.activeElement = this;
    if (before instanceof FakeElement) before.fire("blur");
    this.fire("focus");
  }
  blur() {
    if (globalThis.document.activeElement !== this) return;
    globalThis.document.activeElement = null;
    this.fire("blur");
  }
  select() { this.selected = true; }
  get textContent() {
    return this.children.map((child) => (typeof child === "string" ? child : child.textContent)).join("");
  }
  set textContent(text) { this.replaceChildren(...(text === "" ? [] : [String(text)])); }
  get text() { return this.textContent; }
  all(accept) {
    const own = accept(this) ? [this] : [];
    return own.concat(this.children.flatMap((child) => (child instanceof FakeElement ? child.all(accept) : [])));
  }
  byClass(name) { return this.all((element) => element.classList.contains(name)); }
  byTag(tag) { return this.all((element) => element.tag === tag); }
  matches(selector) {
    const attr = /^\[([\w-]+)(?:=["']?([^"'\]]*)["']?)?\]$/.exec(selector);
    if (attr) return attr[1] in this.attrs && (attr[2] === undefined || this.attrs[attr[1]] === attr[2]);
    return selector.startsWith(".") ? this.classList.contains(selector.slice(1)) : this.tag === selector;
  }
  querySelectorAll(selector) { return this.all((element) => element !== this && element.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
  closest(selector) {
    for (let at = this; at; at = at.parentElement) if (at.matches(selector)) return at;
    return null;
  }
}

/** A `<template>`: its markup is not parsed, `content.firstElementChild` is one empty element (the views only place an icon). */
function fakeTemplate() {
  return { innerHTML: "", content: { firstElementChild: new FakeElement("svg") } };
}

/** Installs a fresh fake `document` with an empty `body` (and `Node`); returns it. */
export function installFakeDom() {
  const body = mount(new FakeElement("body"));
  globalThis.document = {
    createElement: (tag) => (tag === "template" ? fakeTemplate() : new FakeElement(tag)),
    activeElement: null,
    body,
    querySelector: (selector) => body.querySelector(selector),
  };
  globalThis.Node = FakeElement;
  return globalThis.document;
}

/** Marks `element` as being in the page (its descendants are then `isConnected`). */
export function mount(element) {
  element.root = true;
  return element;
}

export const flush = () => new Promise((done) => setTimeout(done));
