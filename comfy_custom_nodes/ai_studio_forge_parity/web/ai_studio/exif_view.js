// Anima studio: the EXIF panel next to XYZ 비교 (Forge's PNG Info): the gallery's picked image, or a file dropped on the
// panel or opened with its button, and the text that file carries. Nothing here is saved with the workflow.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { INFOTEXT_KEYS, readImageInfo } from "./studio_exif.js";
import { imageSettings } from "./studio_settings.js";

const FOLD_CHARS = 2000; // longer text (ComfyUI's prompt and workflow JSON) starts folded; the infotext never does
const MAX_SHOWN = 1024 * 1024; // a textarea takes seconds to lay out past a few MB and stops answering near 100 MB
const MESSAGES = Object.freeze({
  empty: "이미지를 여기에 놓거나 갤러리에서 고르기",
  noInfo: "이 이미지엔 생성 정보가 없음",
  noFormat: "PNG · JPEG · WebP 만 읽음",
  failed: "이미지를 읽지 못함",
});
const PULL_TITLE = "이 이미지를 만들 때의 제어판과 XYZ 설정을 그대로 되돌림 (탭과 i2i 이미지는 지금 그대로). Forge 이미지는 생성 정보에 적힌 것만";
const PULL_NONE = "가져올 생성 정보가 있는 이미지를 고르기";

const keepFromComfy = (event) => {
  event.preventDefault();
  event.stopPropagation(); // ComfyUI's document listener would open the file as a workflow
};

export class ExifView {
  /**
   * `ctx`: `imageUrl(ref)`, `readImage(ref)` (the file's bytes as an ArrayBuffer, null when the server has none),
   * `pullSettings(settings)` (settings: studio_settings.imageSettings).
   */
  constructor(widget, ctx) {
    this.ctx = ctx;
    this.shown = null; // {name, url, info}; info: null while reading, `{failed: true}` when that failed
    this.reading = 0; // only the newest read may draw
    this.fileUrl = ""; // the object URL of a dropped or opened file
    this.element = h("div", {
      class: "ais-root ais-exif",
      onDragover: keepFromComfy,
      onDrop: (event) => {
        keepFromComfy(event);
        const file = event.dataTransfer?.files?.[0];
        if (file) this.openFile(file);
      },
    });
    this.picker = h("input", {
      type: "file",
      accept: "image/png,image/jpeg,image/webp",
      hidden: true,
      onChange: (event) => {
        const file = event.target.files?.[0];
        event.target.value = ""; // the same file can be opened again
        if (file) this.openFile(file);
      },
    });
    this.render();
  }

  onStudioEvent(type, payload) {
    if (type === "selected") this.showImage(payload.image);
  }

  dispose() {
    this.reading += 1;
    this.setFileUrl("");
  }

  setFileUrl(url) {
    if (this.fileUrl) URL.revokeObjectURL(this.fileUrl);
    this.fileUrl = url;
  }

  showImage(ref) {
    this.setFileUrl("");
    if (!ref) {
      this.reading += 1;
      this.shown = null;
      this.render();
      return;
    }
    this.show(ref.filename, this.ctx.imageUrl(ref), () => this.ctx.readImage(ref));
  }

  openFile(file) {
    this.setFileUrl(URL.createObjectURL(file));
    this.show(file.name, this.fileUrl, () => file.arrayBuffer());
  }

  async show(name, url, read) {
    this.reading += 1;
    const reading = this.reading;
    this.shown = { name, url, info: null };
    this.render();
    let info;
    try {
      const buffer = await read();
      info = buffer ? await readImageInfo(buffer) : { failed: true };
    } catch {
      info = { failed: true };
    }
    if (reading !== this.reading) return;
    this.shown.info = info;
    this.shown.settings = info.entries ? imageSettings(info.entries) : null;
    this.render();
  }

  // ------------------------------------------------------------------------------------------- drawing

  /** Grey with aria-disabled, as `pill`: the mouse-over title still says why. */
  pullButton() {
    const settings = this.shown?.settings;
    return h("button", {
      type: "button",
      class: `ais-mini ${settings ? "" : "dis"}`,
      title: settings ? PULL_TITLE : PULL_NONE,
      "aria-disabled": settings ? undefined : "true",
      onClick: settings ? () => this.ctx.pullSettings(settings) : undefined,
    }, "설정 가져오기");
  }

  render() {
    this.element.replaceChildren(
      h("div", { class: "ais-nt" }, h("span", {}, "EXIF"),
        h("span", { class: "ais-head-ctl" }, this.picker,
          h("button", { type: "button", class: "ais-mini", onClick: () => this.picker.click() }, "파일 열기"),
          this.pullButton())),
      h("div", { class: "ais-exif-body" }, this.shown ? this.details(this.shown) : h("div", { class: "ais-note" }, MESSAGES.empty)),
    );
  }

  details({ name, url, info }) {
    const label = [name, info?.format].filter(Boolean).join(" · ");
    return [h("img", { class: "ais-exif-image", src: url, alt: "" }), h("div", { class: "ais-exif-name" }, label), ...this.body(info)];
  }

  body(info) {
    if (!info) return [];
    if (info.failed) return [h("div", { class: "ais-note" }, MESSAGES.failed)];
    if (!info.format) return [h("div", { class: "ais-note" }, MESSAGES.noFormat)];
    if (!info.entries.length) return [h("div", { class: "ais-note" }, MESSAGES.noInfo)];
    return info.entries.map(({ key, text }) => {
      const cut = text.length > MAX_SHOWN;
      const title = cut ? `${key} (앞부분만)` : key;
      const box = h("textarea", { class: "ais-f ais-exif-text", readOnly: true, value: cut ? text.slice(0, MAX_SHOWN) : text, "aria-label": key });
      return text.length > FOLD_CHARS && !INFOTEXT_KEYS.includes(key)
        ? h("details", { class: "ais-exif-fold" }, h("summary", {}, title), box)
        : h("div", { class: "ais-exif-entry" }, h("span", { class: "ais-label" }, title), box);
    });
  }
}
