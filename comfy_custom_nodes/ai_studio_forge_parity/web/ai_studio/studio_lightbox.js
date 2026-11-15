// Anima studio: full-screen image viewer for the result column (like clicking a Forge gallery image). The overlay goes on
// document.body, outside the node's DOM widget, because the canvas applies CSS transforms to everything inside the widget.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).

let current = null; // the open viewer; only one at a time

/** Whether the viewer is open now (the gallery's own ←/→ wait while it is). */
export const lightboxOpen = () => current !== null;

/** The index `delta` steps from `i` in a ring of `n` images. */
export function nextIndex(i, delta, n) {
  if (!n) return 0;
  return (((i + delta) % n) + n) % n;
}

const NOOP = Object.freeze({ close() {}, show() {} });

/**
 * Open the viewer on `items[index]` (`[{url}]`): fit to screen, ←/→ and the side buttons page through, Esc / a click
 * on the backdrop or the image closes it. Returns `{close(), show(i)}`; an empty list opens nothing.
 * `onShow(item, i)` (optional) hears every move to another image (a key, a side button, `show(i)`; not the opening): the
 * item as it was passed and its place among the shown ones. Without it the viewer tells nobody (the caller's choice).
 * Opening takes no focus. A click on a side button does, and the press of a mouse close drops it to the page: closing
 * then hands the focus back to what held it at opening, so the opener's own keys go on working. A focus that went
 * somewhere else meanwhile is left there.
 */
export function openLightbox(items, index = 0, doc = document, { onShow } = {}) {
  const shown = (items || []).filter((item) => item && typeof item.url === "string" && item.url !== "");
  const urls = shown.map((item) => item.url);
  if (!urls.length) return NOOP;
  current?.close();

  let at = nextIndex(Number.isInteger(index) ? index : 0, 0, urls.length);
  let closed = false;
  const opener = doc.activeElement ?? null;
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
    const before = at;
    at = nextIndex(i, 0, urls.length);
    img.setAttribute("src", urls[at]);
    counter.textContent = `${at + 1} / ${urls.length}`;
    if (at !== before) onShow?.(shown[at], at); // last: the viewer already shows it
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
    const held = doc.activeElement;
    // The viewer took the focus from the opener: it is on a side button (which goes away with the viewer), or the press
    // of a mouse close (the backdrop, the image: neither takes the focus) dropped it to the page.
    const taken = !held || held === doc.body || root.contains?.(held) === true;
    doc.removeEventListener("keydown", onKey, true);
    root.remove();
    if (current === handle) current = null;
    if (taken && opener && opener !== doc.body && opener.isConnected !== false) opener.focus?.({ preventScroll: true });
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
