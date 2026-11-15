// Anima studio: the result column (C4 middle, 680px) drawn into AIStudioResultView's DOM widget — image, 미리보기 /
// XYZ 격자 tabs, progress, gallery, the seven gallery buttons and 갤러리 자동 비우기. The gallery's image refs are kept as
// JSON in the node's hidden `view` widget; the settings each image was made with are kept for this session only.
// ←/→ (Home/End) pick the previous/next gallery image while the focus is somewhere in this node (a click in it puts it
// there): the key listener is on the node's own element, nothing on document or window. The picked thumbnail is kept in
// the middle of the strip (result_gallery.css has the focus styles). With the focus in the node, the canvas's own shortcuts
// (Delete, Backspace, ...) stay in it too: ComfyUI would run them on this node.
// The enlarged view (a click on the image) pages through the gallery; the pick follows it (`follow`).
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h, icon, pill, refKey } from "./studio_dom.js";
import { lightboxOpen, openLightbox } from "./studio_lightbox.js";

const GALLERY_LIMIT = 40;
const NOTICE_MS = 8000;
const AUTO_CLEAR = "갤러리 자동 비우기";
const GALLERY_CSS_ID = "aistudio-result-gallery-css";
const GALLERY_LABEL = "갤러리 (←/→ 로 이동)";
const TYPING_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);
// ComfyUI's shortcuts for the canvas (frontend 1.58 defaults: delete, delete, fit view, pin, collapse, bypass, mute, select
// all, paste with links, zoom). It runs them when the pressed element is inside the canvas container, and a node's DOM
// widget is: a click in this node selects it, so Delete would remove the node and its gallery.
const CANVAS_KEYS = new Set(["delete", "backspace", ".", "p", "alt+c", "ctrl+b", "ctrl+m", "ctrl+a", "ctrl+shift+v", "alt+=", "alt+shift++", "alt++", "alt+-"]);
// Forge's order under the gallery (Global Constraints): name -> label.
export const GALLERY_BUTTONS = Object.freeze([
  ["folder", "폴더"],
  ["i2i", "→ i2i"],
  ["inpaint", "→ 인페인트"],
  ["upscale", "업스케일"],
  ["hires", "Hires"],
  ["sam3", "SAM3"],
  ["seed", "같은 시드"],
  ["lora_preview", "LoRA 미리보기로"],
]);

const isRef = (ref) => ref && typeof ref === "object" && typeof ref.filename === "string" && ref.filename !== "";
const cleanRef = (ref) => ({ filename: ref.filename, subfolder: String(ref.subfolder || ""), type: ref.type || "output" });

function duration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  return total < 60 ? `${total}초` : `${Math.floor(total / 60)}분 ${total % 60}초`;
}

/**
 * Where a gallery key moves the pick among `count` images: the new index (the same one at an end: no wrap), or null
 * for a key that is not the gallery's.
 */
export function galleryKeyIndex(key, selected, count) {
  if (!(count > 0)) return null;
  if (key === "ArrowLeft") return Math.max(0, selected - 1);
  if (key === "ArrowRight") return Math.min(count - 1, selected + 1);
  if (key === "Home") return 0;
  if (key === "End") return count - 1;
  return null;
}

/**
 * The scrollLeft that puts a thumbnail (`left` and `width` inside the strip's content) in the middle of a strip `view`
 * wide whose content is `total` wide, clamped at both ends. Null while the strip is not laid out (not on the page yet).
 */
export function centredScrollLeft({ left, width, view, total }) {
  if (![left, width, view, total].every(Number.isFinite) || view <= 0) return null;
  return Math.round(Math.min(Math.max(0, total - view), Math.max(0, left + width / 2 - view / 2)));
}

/** A key event as ComfyUI names its combos: Ctrl (or ⌘), Alt, Shift, then the key, whatever its case. */
const comboOf = (event) =>
  `${event.ctrlKey || event.metaKey ? "ctrl+" : ""}${event.altKey ? "alt+" : ""}${event.shiftKey ? "shift+" : ""}${String(event.key ?? "").toLowerCase()}`;
const isTyping = (element) => !!element && (TYPING_TAGS.has(String(element.tagName || "").toUpperCase()) || element.isContentEditable === true);
const reducedMotion = () => globalThis.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true;

/** The gallery's focus styles, once per page (studio.css is studio_main.js's). */
function addGalleryStyles(doc = globalThis.document) {
  if (!doc?.head || doc.getElementById(GALLERY_CSS_ID)) return;
  const link = doc.createElement("link");
  link.id = GALLERY_CSS_ID;
  link.rel = "stylesheet";
  link.href = new URL("./result_gallery.css", import.meta.url).href;
  doc.head.append(link);
}

/** A finished run's time: tenths of a second are worth showing only under a minute (Forge's Time taken). */
const tookText = (ms) => (ms < 59950 ? `${(ms / 1000).toFixed(1)}초` : duration(ms / 1000));

export class ResultView {
  /** `ctx`: `imageUrl(ref, thumb?)`, `action(name, item)` (a gallery button), `changed()`. */
  constructor(widget, ctx) {
    this.widget = widget;
    this.ctx = ctx;
    this.records = new Map(); // refKey -> {state, seed} of images made in this session
    this.running = false;
    this.elapsed = null; // ms of the last run that made images
    this.progress = null;
    this.previewUrl = "";
    this.notice = "";
    this.view = this.read();
    this.uncentred = false; // the pick still has to be brought to the middle: the strip had no size when it was picked
    this.disposed = false; // the node is gone: an enlarged view it opened may still be on the page
    // tabindex -1: a click anywhere in the node (the image, an empty spot) leaves the focus in it, so its keys come here.
    this.element = h("div", { class: "ais-root ais-result", tabindex: "-1", onKeydown: (event) => this.onKey(event) });
    addGalleryStyles();
    if (typeof ResizeObserver === "function") {
      // The view is drawn before ComfyUI puts it on the page: centre the pick when the node first has a size.
      this.observer = new ResizeObserver(() => { if (this.uncentred) this.centre(); });
      this.observer.observe(this.element);
    }
    this.render();
    this.centre();
  }

  read() {
    let raw = {};
    try {
      raw = JSON.parse(this.widget?.value || "{}") || {};
    } catch {
      raw = {};
    }
    const gallery = (Array.isArray(raw.gallery) ? raw.gallery : []).filter(isRef).map(cleanRef).slice(0, GALLERY_LIMIT);
    const selected = Number.isInteger(raw.selected) && raw.selected >= 0 && raw.selected < gallery.length ? raw.selected : 0;
    return { tab: raw.tab === "grid" ? "grid" : "preview", gallery, selected, grid: isRef(raw.grid) ? cleanRef(raw.grid) : null, auto_clear: raw.auto_clear === true };
  }

  load() {
    this.view = this.read();
    this.render();
    this.centre();
  }

  save() {
    if (this.widget) this.widget.value = JSON.stringify(this.view);
    this.ctx.changed?.();
  }

  /** `{image, record}` of the picked gallery image (record null when it was not made in this session). */
  selectedItem() {
    const image = this.view.gallery[this.view.selected];
    return image ? { image, record: this.records.get(refKey(image)) || null } : null;
  }

  /**
   * New images go in front, in their own order; the first of them is picked. `replace` (RunController: a generation's
   * first images) empties the gallery first with 갤러리 자동 비우기 on, so a stopped or failed run leaves the last images (Forge).
   */
  add(items, replace = false) {
    const fresh = items.filter((item) => isRef(item.image));
    if (!fresh.length) return;
    for (const item of fresh) if (item.record) this.records.set(refKey(item.image), item.record);
    const keys = new Set(fresh.map((item) => refKey(item.image)));
    const rest = replace && this.view.auto_clear ? [] : this.view.gallery.filter((ref) => !keys.has(refKey(ref)));
    this.view.gallery = [...fresh.map((item) => cleanRef(item.image)), ...rest].slice(0, GALLERY_LIMIT);
    this.view.selected = 0;
    if (this.galleryEl) this.galleryEl.scrollLeft = 0; // the picked image is in front now
    this.save();
    this.announce();
  }

  /**
   * Picks gallery image `index` (a clicked thumbnail, ←/→): the image and everything that follows the pick (EXIF panel,
   * the gallery buttons) show it, and it glides to the middle of the strip. On the 미리보기 tab only the frames and the
   * image change, so the strip and its scroll stay as they are. A thumbnail that holds the focus hands it to the picked
   * one: Enter / Space on it would pick the old image again.
   */
  pick(index) {
    const redraw = this.view.tab !== "preview"; // the tab pills change too
    const active = globalThis.document?.activeElement;
    const onThumb = !!active && active !== this.galleryEl && this.galleryEl.contains?.(active) === true;
    this.view.selected = index;
    this.view.tab = "preview";
    this.save();
    if (redraw) this.render();
    else {
      Array.from(this.galleryEl.children).forEach((thumb, at) => { thumb.className = `ais-thumb ${at === index ? "sel" : ""}`; });
      this.renderStage();
    }
    const thumb = this.galleryEl.children[index];
    if (onThumb && thumb !== active) thumb?.focus?.({ preventScroll: true }); // centre() alone scrolls the strip
    this.centre(true);
    this.announce();
  }

  /**
   * The enlarged view this gallery opened moved to `ref`: the gallery picks it too, as a click on its thumbnail would (the
   * image, EXIF panel, gallery buttons, the saved pick, the thumbnail in the middle). The image is looked up by what it is,
   * not by its place in the viewer: the gallery may have changed since it was opened (new images in front, another
   * workflow). One that is not in the gallery any more picks nothing.
   */
  follow(ref) {
    if (this.disposed || !isRef(ref)) return;
    const key = refKey(ref);
    const index = this.view.gallery.findIndex((image) => refKey(image) === key);
    if (index < 0 || (index === this.view.selected && this.view.tab === "preview")) return; // gone, or already shown
    this.pick(index);
  }

  /**
   * A key pressed in the node. ←/→, Home/End: see `galleryKeyIndex`; the enlarged view has its own keys. The canvas's
   * shortcuts (`CANVAS_KEYS`) go no further than the node, the enlarged view open or not. Typing keeps all its keys.
   */
  onKey(event) {
    if (isTyping(event.target)) return;
    if (CANVAS_KEYS.has(comboOf(event))) {
      event.stopPropagation(); // not prevented: the browser's own part (none for Delete on a button) stays
      return;
    }
    if (event.defaultPrevented || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || lightboxOpen()) return;
    const index = galleryKeyIndex(event.key, this.view.selected, this.view.gallery.length);
    if (index === null) return;
    event.preventDefault(); // the browser would scroll the focused strip
    event.stopPropagation(); // the canvas does not get a key the gallery used
    if (index !== this.view.selected) this.pick(index);
  }

  /** Whether the focus is on this node or in it. */
  holdsFocus() {
    const active = globalThis.document?.activeElement;
    return !!active && this.element.contains?.(active) === true;
  }

  focusGallery() {
    this.galleryEl?.focus?.({ preventScroll: true }); // the page and the canvas stay where they are
  }

  /**
   * Scrolls the strip so the picked thumbnail is in its middle (the ends clamp): `smooth` for a pick by the user, at once
   * otherwise. Only scrollLeft of the strip itself, in layout pixels (the canvas zoom is a CSS transform, which
   * getBoundingClientRect would include), never scrollIntoView. A strip without a size yet is centred when it gets one.
   */
  centre(smooth = false) {
    const box = this.galleryEl;
    const thumb = box?.children?.[this.view.selected];
    if (!thumb) {
      this.uncentred = false;
      return;
    }
    const left = thumb.offsetParent === box ? thumb.offsetLeft : thumb.offsetLeft - box.offsetLeft;
    const target = centredScrollLeft({ left, width: thumb.offsetWidth, view: box.clientWidth, total: box.scrollWidth });
    this.uncentred = target === null;
    if (target === null) return;
    if (smooth && typeof box.scrollTo === "function" && !reducedMotion()) box.scrollTo({ left: target, behavior: "smooth" });
    else box.scrollLeft = target;
  }

  /** Tells the EXIF panel which image is picked now. */
  announce() {
    this.ctx.selected?.(this.view.gallery[this.view.selected] || null);
  }

  onStudioEvent(type, payload) {
    if (type === "running") {
      this.running = payload.running;
      this.elapsed = payload.elapsed ?? this.elapsed;
      this.progress = null;
      this.setPreview(null);
      this.render();
    } else if (type === "progress") {
      this.progress = payload;
      this.renderProgress();
    } else if (type === "preview") {
      this.setPreview(payload.blob);
      this.renderStage();
    } else if (type === "images") {
      this.view.tab = "preview";
      this.add(payload.items, payload.replace);
      this.render();
      this.centre();
    } else if (type === "grid") {
      this.view.grid = cleanRef(payload.item.image);
      this.view.tab = "grid";
      this.add([payload.item, ...(payload.subgrids || [])], payload.replace);
      this.render();
      this.centre();
    } else if (type === "notice") {
      this.showNotice(payload.text);
    }
  }

  setPreview(blob) {
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    this.previewUrl = blob ? URL.createObjectURL(blob) : "";
  }

  showNotice(text) {
    this.notice = text || "";
    clearTimeout(this.noticeTimer);
    if (this.notice) {
      this.noticeTimer = setTimeout(() => {
        this.notice = "";
        this.renderNotice();
      }, NOTICE_MS);
    }
    this.renderNotice();
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.noticeTimer);
    this.setPreview(null);
    this.observer?.disconnect();
  }

  // ------------------------------------------------------------------------------------------- drawing

  render() {
    const tab = (name, text) =>
      h("button", { type: "button", class: `ais-pill ${this.view.tab === name ? "on" : "off"}`, "aria-pressed": this.view.tab === name ? "true" : "false",
        onClick: () => { this.view.tab = name; this.save(); this.render(); } }, text);
    const scrolled = this.galleryEl?.scrollLeft ?? 0; // the gallery is drawn anew
    const held = this.holdsFocus(); // on a thumbnail it goes away with the old strip
    this.progressEl = h("div", { class: "ais-progress" });
    this.noticeEl = h("div", { class: "ais-notice", role: "status" });
    this.stage = h("div", { class: "ais-stage" });
    this.galleryEl = this.gallery();
    this.element.replaceChildren(
      h("div", { class: "ais-nt" },
        h("span", { class: "ais-head-ctl" }, h("span", {}, "결과"),
          this.running || this.elapsed === null ? null : h("span", { class: "ais-took", title: "생성을 누른 때부터 끝날 때까지" }, `걸린 시간 ${tookText(this.elapsed)}`)),
        h("span", { class: "ais-head-ctl" }, tab("preview", "미리보기"), tab("grid", "XYZ 격자"))),
      this.progressEl,
      this.noticeEl,
      this.stage,
      this.galleryEl,
      h("div", { class: "ais-buttons" }, GALLERY_BUTTONS.map(([name, label]) =>
        h("button", { type: "button", class: "ais-ib", "aria-label": label, title: label, onClick: () => this.ctx.action(name, this.selectedItem()) }, icon(name)))),
      h("div", { class: "ais-gallery-opts" },
        pill(this.view.auto_clear, () => { this.view.auto_clear = !this.view.auto_clear; this.save(); this.render(); }, {
          on: AUTO_CLEAR, off: AUTO_CLEAR, title: "켜면 Forge처럼 생성할 때마다 새 이미지로 바꿈 (파일은 그대로)" })),
    );
    this.galleryEl.scrollLeft = scrolled; // after it joined the page: an element outside it has nowhere to scroll
    if (held && !this.holdsFocus()) this.focusGallery(); // the keys keep working
    this.renderProgress();
    this.renderNotice();
    this.renderStage();
  }

  renderProgress() {
    if (!this.progressEl) return;
    this.progressEl.hidden = !this.running;
    const p = this.progress;
    if (!this.running) return;
    const cell = p?.cell ? `칸 ${p.cell.index} / ${p.cell.total} · ` : "";
    const step = p ? `스텝 ${p.value} / ${p.max} · ${p.percent}%` : "준비 중";
    const eta = p && p.eta !== null && p.eta !== undefined ? `남은 시간 약 ${duration(p.eta)}` : "";
    this.progressEl.replaceChildren(
      h("div", { class: "ais-progress-text" }, h("span", {}, cell + step), h("span", {}, eta)),
      h("div", { class: "ais-progress-bar" }, h("div", { style: `width:${p ? Math.min(100, p.percent) : 0}%` })),
    );
  }

  renderNotice() {
    if (!this.noticeEl) return;
    this.noticeEl.hidden = !this.notice;
    this.noticeEl.textContent = this.notice;
  }

  renderStage() {
    if (!this.stage) return;
    let src = "";
    let zoom = null; // what a click on the image opens big (`ref`: what `follow` picks); the live step preview opens nothing
    let text = this.running ? "생성 중 미리보기 (중간 단계)" : "선택한 이미지";
    if (this.running && this.previewUrl && this.view.tab === "preview") src = this.previewUrl;
    else if (this.view.tab === "grid") {
      if (this.view.grid) {
        src = this.ctx.imageUrl(this.view.grid);
        zoom = { items: [{ url: src }], index: 0 };
      } else text = "XYZ 격자 없음";
    } else if (this.view.gallery[this.view.selected]) {
      src = this.ctx.imageUrl(this.view.gallery[this.view.selected]);
      zoom = { items: this.view.gallery.map((ref) => ({ url: this.ctx.imageUrl(ref), ref })), index: this.view.selected };
    }
    const img = src
      ? h("img", { class: `ais-stage-img ${zoom ? "zoomable" : ""}`, src, alt: "결과 이미지", title: zoom ? "클릭하면 확대" : undefined,
        onClick: zoom ? () => openLightbox(zoom.items, zoom.index, undefined, { onShow: (item) => this.follow(item.ref) }) : undefined })
      : null;
    this.stage.replaceChildren(img || h("div", { class: "ais-stage-empty" }, text));
  }

  gallery() {
    return h("div", { class: "ais-gallery", tabindex: "0", "aria-label": GALLERY_LABEL }, this.view.gallery.map((ref, index) =>
      h("button", { type: "button", class: `ais-thumb ${index === this.view.selected ? "sel" : ""}`, title: ref.filename, "aria-label": ref.filename,
        onClick: () => {
          this.pick(index);
          if (!this.holdsFocus()) this.focusGallery(); // where a click does not focus a button (Safari)
        } },
      h("img", { src: this.ctx.imageUrl(ref, true), loading: "lazy", alt: "" }))));
  }
}
