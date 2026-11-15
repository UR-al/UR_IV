// Gallery button "LoRA 미리보기로": the LoRAs an image was made with (its <lora:..> tags) -> set the image as that run's
// preview. One target is set at once, several get a small list (Esc / outside click closes), none gives a notice.
// `deps`: {readText(ref) -> infotext|null, entries() -> Promise<string[]>, api: {setPreview}, notice(text)}.
import { previewTargets } from "./lora_names.js";
import { h } from "./studio_dom.js";

function chooser(targets, onPick) {
  return new Promise((resolve) => {
    const close = (target) => {
      menu.remove();
      document.removeEventListener("keydown", onKey, true);
      document.removeEventListener("pointerdown", onOutside, true);
      resolve(target);
    };
    const onKey = (event) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      close(null);
    };
    const onOutside = (event) => { if (!menu.contains(event.target)) close(null); };
    const menu = h("div", { class: "ais-lb-menu", role: "menu" },
      h("div", { class: "ais-lb-menu-title" }, "미리보기로 지정할 LoRA"),
      targets.map((target) => h("button", { type: "button", role: "menuitem", title: target.runKey, onClick: () => close(target) }, target.label)));
    document.body.append(menu);
    document.addEventListener("keydown", onKey, true);
    setTimeout(() => document.addEventListener("pointerdown", onOutside, true), 0);
    menu.querySelector("button")?.focus();
  });
}

export async function setImageAsLoraPreview(item, deps) {
  const text = await deps.readText(item.image);
  const targets = previewTargets(text, await deps.entries());
  if (!targets.length) return deps.notice("이 그림에 LoRA 가 없음");
  const target = targets.length === 1 ? targets[0] : await chooser(targets);
  if (!target) return null;
  await deps.api.setPreview(target.runKey, item.image);
  return deps.notice(`${target.label} 미리보기로 지정함`);
}
