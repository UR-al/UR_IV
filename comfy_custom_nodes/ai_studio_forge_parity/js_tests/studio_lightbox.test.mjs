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
  /** Leaving the page with the focus inside drops the focus to the page, as in a browser. */
  remove() {
    const doc = this.doc;
    if (doc && this.contains(doc.activeElement)) doc.activeElement = doc.body;
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }
  contains(node) {
    for (let at = node; at; at = at.parent) if (at === this) return true;
    return false;
  }
  get doc() {
    let top = this;
    while (top.parent) top = top.parent;
    return top.ownerDoc || null;
  }
  get isConnected() { return this.doc !== null; }
  focus(options) {
    (this.focused ||= []).push(options);
    if (this.doc) this.doc.activeElement = this;
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
    /** A focusable element on the page (what opened the viewer). */
    add(tag = "div") {
      const element = new FakeElement(tag);
      doc.body.append(element);
      return element;
    },
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
  doc.body.ownerDoc = doc;
  doc.activeElement = doc.body;
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

// ---------------------------------------------------------------------------------------------- onShow

/** Opens with an `onShow` that writes down every call as `[item, index]`. */
function openHeard(items, index, doc) {
  const heard = [];
  const box = openLightbox(items, index, doc, { onShow: (item, at) => heard.push([item, at]) });
  return { box, heard };
}

test("onShow hears every move to another image, by key, button or show(): the item as it was passed and its place", () => {
  const doc = fakeDocument();
  const items = [{ url: "/a.png", ref: "A" }, { url: "/b.png", ref: "B" }, { url: "/c.png", ref: "C" }];
  const { box, heard } = openHeard(items, 1, doc);
  assert.deepEqual(heard, []); // opening is not a move: the caller knows what it opened
  doc.key("ArrowRight");
  doc.key("ArrowRight"); // wraps
  doc.body.children[0].find("ais-lightbox-prev").fire("click");
  doc.body.children[0].find("ais-lightbox-next").fire("click");
  box.show(1);
  assert.deepEqual(heard.map(([, at]) => at), [2, 0, 2, 0, 1]);
  assert.deepEqual(heard.map(([item]) => item.ref), ["C", "A", "C", "A", "B"]);
  assert.equal(heard[0][0], items[2]); // the very object
});

test("onShow is told after the image changed, so it sees the viewer as the user does", () => {
  const doc = fakeDocument();
  const seen = [];
  openLightbox(ITEMS, 0, doc, { onShow: (item) => seen.push([item.url, doc.body.children[0].find("ais-lightbox-img").attrs.src]) });
  doc.key("ArrowRight");
  assert.deepEqual(seen, [["/b.png", "/b.png"]]);
});

test("onShow is not called when the image stays the same, nor after closing", () => {
  const doc = fakeDocument();
  const { box, heard } = openHeard(ITEMS, 1, doc);
  box.show(1);
  box.show(4); // the same place in the ring
  doc.key("Escape");
  box.show(2);
  doc.key("ArrowRight");
  assert.deepEqual(heard, []);

  const one = openHeard([{ url: "/only.png" }], 0, doc);
  doc.key("ArrowRight");
  one.box.show(3);
  assert.deepEqual(one.heard, []);
});

test("items without a url are not shown: onShow counts among the shown ones", () => {
  const doc = fakeDocument();
  const items = [{ url: "/a.png" }, null, { url: "" }, { url: "/d.png", ref: "D" }];
  const { heard } = openHeard(items, 0, doc);
  doc.key("ArrowRight");
  assert.deepEqual(heard, [[items[3], 1]]);
  assert.equal(doc.body.children[0].find("ais-lightbox-count").textContent, "2 / 2");
});

test("a second viewer's moves go to its own onShow only", () => {
  const doc = fakeDocument();
  const first = openHeard(ITEMS, 0, doc);
  const second = openHeard(ITEMS, 0, doc); // closes the first
  doc.key("ArrowRight");
  first.box.show(2);
  assert.deepEqual(first.heard, []);
  assert.deepEqual(second.heard.map(([, at]) => at), [1]);
});

// ---------------------------------------------------------------------------------------------- the focus

const viewer = (doc) => doc.body.children.find((child) => child.className === "ais-lightbox");
/**
 * A mouse click as a browser does it: the press moves the focus first (to the button under it, or to the page when
 * nothing under it takes the focus), then the click is heard.
 */
function mouseClick(doc, element) {
  let taker = element;
  while (taker && taker.tag !== "button" && taker.tag !== "input") taker = taker.parent;
  doc.activeElement = taker || doc.body;
  return element.fire("click");
}

test("closing gives the focus back to what held it when the viewer took it (a clicked side button)", () => {
  const doc = fakeDocument();
  const opener = doc.add();
  opener.focus();
  opener.focused.length = 0;
  openLightbox(ITEMS, 0, doc);
  assert.equal(doc.activeElement, opener); // opening takes no focus
  const next = viewer(doc).find("ais-lightbox-next");
  next.focus(); // a click on it
  next.fire("click");
  doc.key("Escape");
  assert.equal(doc.activeElement, opener);
  assert.deepEqual(opener.focused, [{ preventScroll: true }]); // the page does not move
});

test("closing leaves the focus alone when the viewer never had it, or when it went somewhere else", () => {
  const doc = fakeDocument();
  const opener = doc.add();
  opener.focus();
  opener.focused.length = 0;
  openLightbox(ITEMS, 0, doc);
  doc.key("ArrowRight");
  doc.key("Escape");
  assert.deepEqual(opener.focused, []);

  const box = openLightbox(ITEMS, 0, doc);
  viewer(doc).find("ais-lightbox-next").focus();
  const other = doc.add("input");
  other.focus(); // a script, another window of the page
  box.close();
  assert.equal(doc.activeElement, other);
  assert.deepEqual(opener.focused, []);
});

test("an opener that left the page, or no focus at opening: closing focuses nothing", () => {
  const doc = fakeDocument();
  const opener = doc.add();
  opener.focus();
  opener.focused.length = 0;
  openLightbox(ITEMS, 0, doc);
  viewer(doc).find("ais-lightbox-prev").focus();
  opener.remove(); // the node was redrawn or removed meanwhile
  doc.key("Escape");
  assert.deepEqual(opener.focused, []);
  assert.equal(doc.activeElement, doc.body);

  openLightbox(ITEMS, 0, doc); // the focus is on the page itself
  viewer(doc).find("ais-lightbox-prev").focus();
  doc.key("Escape");
  assert.equal(doc.activeElement, doc.body);
  assert.equal(doc.body.focused, undefined);
});

test("a mouse close (the backdrop or the image) gives the focus back too: the press had dropped it to the page", () => {
  const doc = fakeDocument();
  const opener = doc.add();
  opener.focus();
  opener.focused.length = 0;
  openLightbox(ITEMS, 0, doc);
  doc.key("ArrowRight");
  mouseClick(doc, viewer(doc)); // the dark backdrop
  assert.equal(viewer(doc), undefined);
  assert.equal(doc.activeElement, opener);
  assert.deepEqual(opener.focused, [{ preventScroll: true }]);

  opener.focused.length = 0;
  openLightbox(ITEMS, 0, doc);
  mouseClick(doc, viewer(doc).find("ais-lightbox-next")); // the focus goes to the button
  assert.equal(doc.activeElement, viewer(doc).find("ais-lightbox-next"));
  mouseClick(doc, viewer(doc).find("ais-lightbox-img")); // the enlarged image closes as well
  assert.equal(viewer(doc), undefined);
  assert.equal(doc.activeElement, opener);
  assert.deepEqual(opener.focused, [{ preventScroll: true }]);

  opener.focused.length = 0;
  const box = openLightbox(ITEMS, 0, doc);
  doc.activeElement = null; // a page that reports no focus at all
  box.close();
  assert.equal(doc.activeElement, opener);
  assert.deepEqual(opener.focused, [{ preventScroll: true }]);
});

test("a mouse close focuses nothing when nothing held the focus at opening, or when the opener left the page", () => {
  const doc = fakeDocument();
  openLightbox(ITEMS, 0, doc); // the focus is on the page itself
  mouseClick(doc, viewer(doc));
  assert.equal(doc.activeElement, doc.body);
  assert.equal(doc.body.focused, undefined);

  const opener = doc.add();
  opener.focus();
  opener.focused.length = 0;
  openLightbox(ITEMS, 0, doc);
  opener.remove();
  mouseClick(doc, viewer(doc).find("ais-lightbox-img"));
  assert.deepEqual(opener.focused, []);
  assert.equal(doc.activeElement, doc.body);
});
