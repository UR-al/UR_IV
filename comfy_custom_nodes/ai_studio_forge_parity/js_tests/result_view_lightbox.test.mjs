// node:test for the result gallery following the enlarged view (web/ai_studio/result_view.js + studio_lightbox.js): paging
// there picks the same image in the gallery, as a click on its thumbnail would. The fake document is result_view_fakes.mjs.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { ResultView } from "../web/ai_studio/result_view.js";
import { openLightbox } from "../web/ai_studio/studio_lightbox.js";
import { images, item, lightbox, lightboxImage, mouseClick, pageKey, picked, press, ref, saved, setup, stageImage, strip, thumbsOf, zoom } from "./result_view_fakes.mjs";

const sideButton = (side) => lightbox().byClass(`ais-lightbox-${side}`)[0];
const tabOf = (widget) => JSON.parse(widget.value).tab;

// ---------------------------------------------------------------------------------------------- the pick follows

test("→ in the enlarged view picks that image in the gallery too, exactly as a click on its thumbnail does", () => {
  const { view, widget, picks } = setup(saved(5, 1));
  zoom(view);
  assert.equal(lightboxImage(), "/view/1.png");
  pageKey("ArrowRight");
  assert.equal(lightboxImage(), "/view/2.png");
  assert.ok(lightbox(), "the viewer stays open");
  assert.deepEqual(picked(view), ["2.png"]);
  assert.equal(stageImage(view), "/view/2.png");
  assert.equal(view.selectedItem().image.filename, "2.png"); // what 같은 시드 / → i2i / → 인페인트 use
  assert.deepEqual(picks, ["2.png"]); // the EXIF panel

  const clicked = setup(saved(5, 1));
  thumbsOf(clicked.view)[2].fire("click");
  assert.equal(widget.value, clicked.widget.value);
  assert.deepEqual(picks, clicked.picks);
});

test("the side buttons do the same, and the ring's wrap takes the gallery from its last image to its first", () => {
  const { view, picks } = setup(saved(3, 1));
  zoom(view);
  sideButton("next").fire("click");
  assert.deepEqual(picked(view), ["2.png"]);
  sideButton("next").fire("click"); // 3 / 3 -> 1 / 3
  assert.deepEqual(picked(view), ["0.png"]);
  sideButton("prev").fire("click");
  sideButton("prev").fire("click");
  assert.deepEqual(picked(view), ["1.png"]);
  assert.deepEqual(picks, ["2.png", "0.png", "2.png", "1.png"]);
});

test("the image a record was kept for comes with it: 같은 시드 gets the seed of the image paged to", () => {
  const { view } = setup();
  view.onStudioEvent("images", { items: [0, 1, 2].map((n) => ({ image: ref(`${n}.png`), record: { seed: 100 + n } })), replace: true });
  zoom(view);
  pageKey("ArrowLeft"); // wraps to the last
  assert.deepEqual(view.selectedItem(), { image: ref("2.png"), record: { seed: 102 } });
});

test("opening and closing without paging changes nothing: nothing is saved or told again", () => {
  const { view, widget, picks } = setup(saved(5, 3));
  const before = widget.value;
  zoom(view);
  pageKey("Escape");
  assert.equal(lightbox(), undefined);
  assert.equal(widget.value, before);
  assert.deepEqual(picks, []);
  assert.deepEqual(strip(view).scrolls, []);
});

test("after closing, the gallery's own keys go on from the image the enlarged view ended on", () => {
  const { view } = setup(saved(6, 0));
  zoom(view);
  pageKey("ArrowRight");
  pageKey("ArrowRight");
  pageKey("ArrowRight");
  pageKey("Escape");
  assert.deepEqual(picked(view), ["3.png"]);
  press(view, "ArrowRight", view.element);
  assert.deepEqual(picked(view), ["4.png"]);
  zoom(view); // opens where the gallery is now
  assert.equal(lightboxImage(), "/view/4.png");
});

// ---------------------------------------------------------------------------------------------- centring and focus

test("the paged-to thumbnail glides to the middle of the strip, by the strip's own scroll only", () => {
  const { view } = setup(saved(10, 4));
  strip(view).scrolls.length = 0;
  zoom(view);
  pageKey("ArrowRight"); // FakeElement.scrollIntoView throws: it would move the page
  assert.deepEqual(strip(view).scrolls, [{ left: 243, behavior: "smooth", from: 173 }]);
  pageKey("ArrowLeft");
  pageKey("ArrowLeft");
  assert.equal(strip(view).scrollLeft, 103); // 3.png: 222 + 31 - 150
});

test("with reduced motion asked for, the thumbnail jumps to the middle instead", () => {
  const { view } = setup(saved(10, 4));
  globalThis.matchMedia = (query) => ({ matches: query.includes("prefers-reduced-motion") });
  zoom(view);
  pageKey("ArrowRight");
  assert.equal(strip(view).scrollLeft, 243);
  assert.deepEqual(strip(view).scrolls, []);
});

test("paging never takes the focus from where it is; a thumbnail that holds it hands it to the picked one", () => {
  const { view } = setup(saved(5, 1));
  view.element.focus(); // a click on the big image leaves the focus on the node
  zoom(view);
  pageKey("ArrowRight");
  assert.equal(globalThis.document.activeElement, view.element);
  assert.deepEqual(thumbsOf(view).flatMap((thumb) => thumb.focused), []);

  thumbsOf(view)[2].focus();
  pageKey("ArrowRight");
  assert.equal(globalThis.document.activeElement, thumbsOf(view)[3]);
  assert.deepEqual(thumbsOf(view)[3].focused, [{ preventScroll: true }]);

  const outside = setup(saved(5, 1));
  zoom(outside.view); // the focus is nowhere in the node
  pageKey("ArrowRight");
  assert.equal(globalThis.document.activeElement, null);
});

test("paged with the side buttons (a click puts the focus on one): closing gives the focus back to the node, its keys work at once", () => {
  const { view } = setup(saved(6, 0));
  view.element.focus(); // the click on the big image
  zoom(view);
  sideButton("next").focus();
  sideButton("next").fire("click");
  sideButton("next").fire("click");
  pageKey("Escape");
  assert.equal(globalThis.document.activeElement, view.element);
  assert.deepEqual(view.element.focused.at(-1), { preventScroll: true });
  press(view, "ArrowRight", globalThis.document.activeElement);
  assert.deepEqual(picked(view), ["3.png"]);
});

test("closed with the mouse (the backdrop, or the enlarged image after a side button): the node has the focus again, its keys work at once", () => {
  const { view } = setup(saved(6, 0));
  mouseClick(view.element.byClass("ais-stage-img")[0]); // focuses the node and opens the viewer
  assert.equal(globalThis.document.activeElement, view.element);
  pageKey("ArrowRight");
  mouseClick(lightbox()); // the press on the backdrop drops the focus to the page
  assert.equal(lightbox(), undefined);
  assert.equal(globalThis.document.activeElement, view.element);
  assert.deepEqual(view.element.focused.at(-1), { preventScroll: true });
  press(view, "ArrowRight", globalThis.document.activeElement);
  assert.deepEqual(picked(view), ["2.png"]);

  mouseClick(view.element.byClass("ais-stage-img")[0]);
  mouseClick(sideButton("next"));
  mouseClick(lightbox().byClass("ais-lightbox-img")[0]);
  assert.equal(lightbox(), undefined);
  assert.equal(globalThis.document.activeElement, view.element);
  press(view, "ArrowRight", globalThis.document.activeElement);
  assert.deepEqual(picked(view), ["4.png"]);
});

// ---------------------------------------------------------------------------------------------- a gallery that changed

test("new images that arrived meanwhile: the image is found by what it is, not by its place in the viewer", () => {
  const { view } = setup(saved(4, 1));
  zoom(view); // the viewer holds 0..3
  images(view, ["new-a.png", "new-b.png"], false); // in front: 0.png is the third now
  assert.deepEqual(picked(view), ["new-a.png"]);
  pageKey("ArrowRight"); // 2.png, the viewer's third, the gallery's fifth
  assert.equal(lightboxImage(), "/view/2.png");
  assert.deepEqual(picked(view), ["2.png"]);
  assert.equal(JSON.parse(view.widget.value).selected, 4);
});

test("an image that left the gallery (갤러리 자동 비우기) picks nothing: nothing is saved or told", () => {
  const { view, widget, picks } = setup(saved(4, 1, { auto_clear: true }));
  zoom(view);
  images(view, ["new.png"], true); // replaces the four
  const before = widget.value;
  picks.length = 0;
  pageKey("ArrowRight");
  assert.equal(lightboxImage(), "/view/2.png"); // the viewer itself still pages
  assert.equal(widget.value, before);
  assert.deepEqual(picked(view), ["new.png"]);
  assert.deepEqual(picks, []);
});

test("a workflow loaded into the node meanwhile: the viewer's image is looked up in the gallery as it is now", () => {
  const { view, widget } = setup(saved(4, 0));
  zoom(view);
  widget.value = JSON.stringify({ gallery: ["x.png", "3.png", "1.png"].map(ref), selected: 0 });
  view.load();
  pageKey("ArrowRight"); // 1.png
  assert.deepEqual(picked(view), ["1.png"]);
  pageKey("ArrowRight"); // 2.png: not there
  assert.deepEqual(picked(view), ["1.png"]);
  pageKey("ArrowRight"); // 3.png
  assert.deepEqual(picked(view), ["3.png"]);
});

// ---------------------------------------------------------------------------------------------- the XYZ grid

test("the XYZ grid opens alone: the gallery's pick and the tab stay as they are", () => {
  const { view, widget, picks } = setup(saved(3, 2, { grid: ref("grid.png"), tab: "grid" }));
  const before = widget.value;
  zoom(view);
  assert.equal(lightboxImage(), "/view/grid.png");
  pageKey("ArrowRight");
  pageKey("ArrowLeft");
  pageKey("Escape");
  assert.equal(widget.value, before);
  assert.equal(stageImage(view), "/view/grid.png");
  assert.deepEqual(picks, []);
});

test("a grid that arrived while the viewer was open: paging shows the picked image, like a click on its thumbnail", () => {
  const { view, widget } = setup(saved(3, 0));
  zoom(view);
  view.onStudioEvent("grid", { item: item("grid.png"), subgrids: [], replace: false });
  assert.equal(tabOf(widget), "grid");
  pageKey("ArrowRight"); // 1.png
  assert.equal(tabOf(widget), "preview");
  assert.deepEqual(picked(view), ["1.png"]);
  assert.equal(stageImage(view), "/view/1.png");
});

test("paging to the gallery's pick while the XYZ 격자 tab is up shows it too: the pick was not on screen", () => {
  const { view, widget, picks } = setup(saved(3, 0));
  zoom(view);
  widget.value = JSON.stringify(saved(3, 1, { grid: ref("grid.png"), tab: "grid" })); // undo, another workflow tab
  view.load();
  assert.equal(stageImage(view), "/view/grid.png");
  pageKey("ArrowRight"); // 1.png, the pick already
  assert.equal(tabOf(widget), "preview");
  assert.equal(stageImage(view), "/view/1.png");
  assert.deepEqual(picks, ["1.png"]);
});

// ---------------------------------------------------------------------------------------------- only its own viewer

test("a viewer somebody else opened (the LoRA picker, EXIF) moves no gallery, even on the gallery's own images", () => {
  const { view, widget, picks } = setup(saved(4, 1));
  const before = widget.value;
  openLightbox(view.view.gallery.map((image) => ({ url: `/view/${image.filename}`, ref: image })), 1);
  pageKey("ArrowRight");
  assert.equal(lightboxImage(), "/view/2.png");
  assert.equal(widget.value, before);
  assert.deepEqual(picks, []);
});

test("two result nodes: the viewer moves the gallery it was opened from", () => {
  const { view } = setup(saved(4, 1));
  const otherWidget = { value: JSON.stringify(saved(4, 1)) };
  const other = new ResultView(otherWidget, { imageUrl: (image) => `/view/${image.filename}`, action: () => {} });
  globalThis.document.body.append(other.element);
  const before = otherWidget.value;
  zoom(view);
  pageKey("ArrowRight");
  assert.deepEqual(picked(view), ["2.png"]);
  assert.equal(otherWidget.value, before);
  zoom(other); // replaces the first viewer
  pageKey("ArrowLeft");
  assert.deepEqual(picked(other), ["0.png"]);
  assert.deepEqual(picked(view), ["2.png"]);
});

test("a removed node's viewer pages on but writes nothing into the node any more", () => {
  const { view, widget, picks } = setup(saved(4, 1));
  zoom(view);
  view.dispose();
  const before = widget.value;
  pageKey("ArrowRight");
  assert.equal(lightboxImage(), "/view/2.png");
  assert.equal(widget.value, before);
  assert.deepEqual(picks, []);
});
