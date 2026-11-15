// The control panel's LoRA rows <-> the LoRA picker window (lora_browser.js): row edits and the row button's face. Pure, no DOM.
// A row is saved as {on, name, strength} with `name` = the file stem (studio_logic.fileStem of the list entry), as before.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { middleEllipsis, shortLoraName } from "./lora_names.js";
import { fileStem } from "./studio_logic.js";

const LABEL_MAX = 40;
// Where the window's slider starts for a row strength: lora_browser_state clamps to 0~2 with two decimals.
const windowStart = (strength) => Math.min(2, Math.max(0, Number(strength.toFixed(2))));

/** The saved row of a window pick `{name, strength}`. */
export const pickedRow = (pick) => ({ on: true, name: String(pick?.name ?? ""), strength: Number.isFinite(pick?.strength) ? pick.strength : 1.0 });

/** Is `name` (a stem) one of the `/choices?kind=loras` entries? */
export const listedStem = (entries, name) => (entries || []).some((entry) => fileStem(String(entry)) === name);

/**
 * One "+ LoRA" window: `onPick(loras, pick)` appends a row and answers its index; `onUnpick(loras, index)` takes that row out
 * again, only while it still holds the picked name (the user may have changed the panel since), and answers whether it did.
 * Rows after a removed one move up one, so the indexes handed out move with them (the window shifts its own the same way).
 */
export function addSession() {
  let picked = new Map(); // row index -> picked name
  return {
    onPick(loras, pick) {
      const row = pickedRow(pick);
      const index = loras.push(row) - 1;
      picked.set(index, row.name);
      return index;
    },
    onUnpick(loras, index) {
      const name = picked.get(index);
      picked.delete(index);
      if (name === undefined || loras[index]?.name !== name) return false;
      loras.splice(index, 1);
      picked = new Map([...picked].map(([at, stem]) => [at > index ? at - 1 : at, stem]));
      return true;
    },
  };
}

/**
 * "바꾸기": row `index` takes the picked LoRA; its switch stays. The picked strength applies when the slider was moved; left where
 * the window started it, the row keeps its own value (one the window cannot show, e.g. -1 or 3, or more than two decimals).
 */
export function replaceRow(loras, index, pick) {
  const row = loras[index];
  if (!row) return false;
  row.name = String(pick?.name ?? "");
  const own = Number.isFinite(row.strength);
  if (Number.isFinite(pick?.strength) && !(own && pick.strength === windowStart(row.strength))) row.strength = pick.strength;
  return true;
}

/**
 * A row button's face `{text, title, missing}`: the short name (the full path on hover). A name the loaded list does not have
 * is marked "(없음)" as the old select did; an empty list is still loading (or the route failed), so nothing is marked then.
 */
export function rowFace(name, entries) {
  if (!name) return { text: "[LoRA]", title: "LoRA 고르기", missing: false };
  const entry = (entries || []).find((item) => fileStem(String(item)) === name);
  if (entry) return { text: middleEllipsis(shortLoraName(entry), LABEL_MAX), title: String(entry), missing: false };
  const missing = (entries || []).length > 0;
  const text = middleEllipsis(shortLoraName(`${name}.safetensors`), LABEL_MAX); // a stem: "2.9b" must not read as an extension
  return missing
    ? { text: `${text} (없음)`, title: `${name} (없음: 이 ComfyUI 의 LoRA 목록에 없음)`, missing: true }
    : { text, title: name, missing: false };
}
