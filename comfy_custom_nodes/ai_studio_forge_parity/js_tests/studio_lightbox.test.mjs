// node:test for the result lightbox (web/ai_studio/studio_lightbox.js) with a tiny fake document (no DOM in node).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { nextIndex, openLightbox } from "../web/ai_studio/studio_lightbox.js";

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.textContent = "";
    this.parent = null;
  }
  append(...nodes) {
    for (const node of nodes) {
      node.parent = this;
      this.children.push(node);
    }
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }
  /** Dispatch with bubbling up the parent chain (until a handler calls stopPropagation). */
  fire(type, event = {}) {
    const e = { target: this, stopPropagation() { this.stopped = true; }, preventDefault() { this.prevented = true; }, ...event };
    for (let node = this; node && !e.stopped; node = node.parent) for (const fn of node.listeners[type] || []) fn(e);
    return e;
  }
  find(className) {
    if (this.className.split(" ").includes(className)) return this;
    for (const child of this.children) {
      const found = child.find(className);
      if (found) return found;
    }
    return null;
  }
}

function fakeDocument() {
  const doc = {
    body: new FakeElement("body"),
    listeners: [],
    createElement: (tag) => new FakeElement(tag),
    addEventListener: (type, fn, capture) => doc.listeners.push({ type, fn, capture }),
    removeEventListener: (type, fn, capture) => {
      doc.listeners = doc.listeners.filter((l) => !(l.type === type && l.fn === fn && l.capture === capture));
    },
    key(key) {
      const event = { key, preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; } };
      for (const l of [...doc.listeners]) if (l.type === "keydown") l.fn(event);
      return event;
    },
  };
  return doc;
}

const ITEMS = [{ url: "/a.png" }, { url: "/b.png" }, { url: "/c.png" }];

test("nextIndex wraps around in both directions", () => {
  assert.equal(nextIndex(2, 1, 3), 0);
  assert.equal(nextIndex(0, -1, 3), 2);
  assert.equal(nextIndex(1, 1, 3), 2);
  assert.equal(nextIndex(0, 0, 1), 0);
  assert.equal(nextIndex(0, 1, 0), 0);
});

test("no images: nothing is created", () => {
  const doc = fakeDocument();
  const box = openLightbox([], 0, doc);
  assert.equal(doc.body.children.length, 0);
  assert.equal(doc.listeners.length, 0);
  box.close();
  box.show(1);
});

test("opens on document.body at the given image and Esc closes it and removes the key listener", () => {
  const doc = fakeDocument();
  openLightbox(ITEMS, 1, doc);
  assert.equal(doc.body.children.length, 1);
  assert.equal(doc.body.children[0].find("ais-lightbox-img").attrs.src, "/b.png");
  assert.equal(doc.listeners.length, 1);
  doc.key("Escape");
  assert.equal(doc.body.children.length, 0);
  assert.equal(doc.listeners.length, 0);
});

test("arrow keys and buttons page through the images, wrapping", () => {
  const doc = fakeDocument();
  const box = openLightbox(ITEMS, 2, doc);
  const img = doc.body.children[0].find("ais-lightbox-img");
  doc.key("ArrowRight");
  assert.equal(img.attrs.src, "/a.png");
  assert.equal(doc.key("ArrowLeft").prevented, true);
  assert.equal(img.attrs.src, "/c.png");
  const next = doc.body.children[0].find("ais-lightbox-next");
  const click = next.fire("click");
  assert.equal(click.stopped, true);
  assert.equal(doc.body.children.length, 1); // a button click must not bubble to the backdrop's close handler
  assert.equal(img.attrs.src, "/a.png");
  doc.body.children[0].find("ais-lightbox-prev").fire("click");
  assert.equal(img.attrs.src, "/c.png");
  box.show(1);
  assert.equal(img.attrs.src, "/b.png");
  assert.equal(doc.body.children.length, 1);
});

test("backdrop click and clicking the image close it; one image has no arrows", () => {
  const doc = fakeDocument();
  openLightbox(ITEMS, 0, doc);
  const root = doc.body.children[0];
  root.fire("click", { target: root });
  assert.equal(doc.body.children.length, 0);
  assert.equal(doc.listeners.length, 0);

  openLightbox([{ url: "/only.png" }], 0, doc);
  const single = doc.body.children[0];
  assert.equal(single.find("ais-lightbox-next"), null);
  doc.key("ArrowRight"); // ignored, still the same image
  assert.equal(single.find("ais-lightbox-img").attrs.src, "/only.png");
  const img = single.find("ais-lightbox-img");
  img.fire("click");
  assert.equal(doc.body.children.length, 0);
});

test("close() is idempotent, show() after close does nothing, and a second open replaces the first", () => {
  const doc = fakeDocument();
  const first = openLightbox(ITEMS, 0, doc);
  openLightbox(ITEMS, 1, doc);
  assert.equal(doc.body.children.length, 1);
  assert.equal(doc.listeners.length, 1);
  first.close(); // already replaced: must not remove the new one's listener
  const second = doc.body.children[0];
  assert.equal(doc.listeners.length, 1);
  second.fire("click", { target: second });
  second.fire("click", { target: second });
  assert.equal(doc.body.children.length, 0);
});
