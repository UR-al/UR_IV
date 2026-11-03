// Anima studio: the result column (C4 middle, 680px) drawn into AIStudioResultView's DOM widget — image, 미리보기 /
// XYZ 격자 tabs, progress, gallery, the seven gallery buttons and 갤러리 자동 비우기. The gallery's image refs are kept as
// JSON in the node's hidden `view` widget; the settings each image was made with are kept for this session only.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h, icon, pill, refKey } from "./studio_dom.js";
import { openLightbox } from "./studio_lightbox.js";

const GALLERY_LIMIT = 40;
const NOTICE_MS = 8000;
const AUTO_CLEAR = "갤러리 자동 비우기";
// Forge's order under the gallery (Global Constraints): name -> label.
export const GALLERY_BUTTONS = Object.freeze([
  ["folder", "폴더"],
  ["i2i", "→ i2i"],
  ["inpaint", "→ 인페인트"],
  ["upscale", "업스케일"],
  ["hires", "Hires"],
  ["sam3", "SAM3"],
  ["seed", "같은 시드"],
]);

const isRef = (ref) => ref && typeof ref === "object" && typeof ref.filename === "string" && ref.filename !== "";
const cleanRef = (ref) => ({ filename: ref.filename, subfolder: String(ref.subfolder || ""), type: ref.type || "output" });

function duration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  return total < 60 ? `${total}초` : `${Math.floor(total / 60)}분 ${total % 60}초`;
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
    this.element = h("div", { class: "ais-root ais-result" });
    this.render();
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
    } else if (type === "grid") {
      this.view.grid = cleanRef(payload.item.image);
      this.view.tab = "grid";
      this.add([payload.item, ...(payload.subgrids || [])], payload.replace);
      this.render();
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
    clearTimeout(this.noticeTimer);
    this.setPreview(null);
  }

  // ------------------------------------------------------------------------------------------- drawing

  render() {
    const tab = (name, text) =>
      h("button", { type: "button", class: `ais-pill ${this.view.tab === name ? "on" : "off"}`, "aria-pressed": this.view.tab === name ? "true" : "false",
        onClick: () => { this.view.tab = name; this.save(); this.render(); } }, text);
    const scrolled = this.galleryEl?.scrollLeft ?? 0; // the gallery is drawn anew
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
    let zoom = null; // what a click on the image opens big; the live step preview opens nothing
    let text = this.running ? "생성 중 미리보기 (중간 단계)" : "선택한 이미지";
    if (this.running && this.previewUrl && this.view.tab === "preview") src = this.previewUrl;
    else if (this.view.tab === "grid") {
      if (this.view.grid) {
        src = this.ctx.imageUrl(this.view.grid);
        zoom = { items: [{ url: src }], index: 0 };
      } else text = "XYZ 격자 없음";
    } else if (this.view.gallery[this.view.selected]) {
      src = this.ctx.imageUrl(this.view.gallery[this.view.selected]);
      zoom = { items: this.view.gallery.map((ref) => ({ url: this.ctx.imageUrl(ref) })), index: this.view.selected };
    }
    const img = src
      ? h("img", { class: `ais-stage-img ${zoom ? "zoomable" : ""}`, src, alt: "결과 이미지", title: zoom ? "클릭하면 확대" : undefined,
        onClick: zoom ? () => openLightbox(zoom.items, zoom.index) : undefined })
      : null;
    this.stage.replaceChildren(img || h("div", { class: "ais-stage-empty" }, text));
  }

  gallery() {
    return h("div", { class: "ais-gallery" }, this.view.gallery.map((ref, index) =>
      h("button", { type: "button", class: `ais-thumb ${index === this.view.selected ? "sel" : ""}`, title: ref.filename, "aria-label": ref.filename,
        onClick: () => { this.view.selected = index; this.view.tab = "preview"; this.save(); this.render(); this.announce(); } },
      h("img", { src: this.ctx.imageUrl(ref, true), loading: "lazy", alt: "" }))));
  }
}
