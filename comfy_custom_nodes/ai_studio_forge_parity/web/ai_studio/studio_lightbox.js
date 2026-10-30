// Anima studio: full-screen image viewer for the result column (like clicking a Forge gallery image). The overlay goes on
// document.body, outside the node's DOM widget, because the canvas applies CSS transforms to everything inside the widget.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).

let current = null; // the open viewer; only one at a time

/** The index `delta` steps from `i` in a ring of `n` images. */
export function nextIndex(i, delta, n) {
  if (!n) return 0;
  return (((i + delta) % n) + n) % n;
}

const NOOP = Object.freeze({ close() {}, show() {} });

/**
 * Open the viewer on `items[index]` (`[{url}]`): fit to screen, ←/→ and the side buttons page through, Esc / a click
 * on the backdrop or the image closes it. Returns `{close(), show(i)}`; an empty list opens nothing.
 */
export function openLightbox(items, index = 0, doc = document) {
  const urls = (items || []).map((item) => item && item.url).filter((url) => typeof url === "string" && url !== "");
  if (!urls.length) return NOOP;
  current?.close();

  let at = nextIndex(Number.isInteger(index) ? index : 0, 0, urls.length);
  let closed = false;
  const make = (tag, className, props = {}) => {
    const element = doc.createElement(tag);
    element.className = className;
    for (const [name, value] of Object.entries(props)) element.setAttribute(name, value);
    return element;
  };

  const root = make("div", "ais-lightbox", { role: "dialog", "aria-modal": "true", "aria-label": "이미지 확대" });
  const img = make("img", "ais-lightbox-img", { alt: "결과 이미지" });
  const counter = make("div", "ais-lightbox-count");
  root.append(img);

  const show = (i) => {
    if (closed) return;
    at = nextIndex(i, 0, urls.length);
    img.setAttribute("src", urls[at]);
    counter.textContent = `${at + 1} / ${urls.length}`;
  };
  const step = (delta) => show(nextIndex(at, delta, urls.length));

  const onKey = (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (urls.length > 1 && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault();
      event.stopPropagation();
      step(event.key === "ArrowLeft" ? -1 : 1);
    }
  };
  function close() {
    if (closed) return;
    closed = true;
    doc.removeEventListener("keydown", onKey, true);
    root.remove();
    if (current === handle) current = null;
  }

  // A click anywhere closes (backdrop or the image itself, it bubbles here); the arrow buttons stop their click.
  root.addEventListener("click", close);
  if (urls.length > 1) {
    const arrow = (side, label, delta) => {
      const button = make("button", `ais-lightbox-btn ais-lightbox-${side}`, { type: "button", "aria-label": label, title: label });
      button.textContent = delta < 0 ? "‹" : "›";
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        step(delta);
      });
      return button;
    };
    root.append(arrow("prev", "이전", -1), arrow("next", "다음", 1), counter);
  }

  doc.addEventListener("keydown", onKey, true);
  doc.body.append(root);
  show(at);
  const handle = { close, show };
  current = handle;
  return handle;
}
