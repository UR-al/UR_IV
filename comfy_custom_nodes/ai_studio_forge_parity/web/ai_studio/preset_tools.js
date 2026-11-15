// Prompt-box presets, the one thing the studio wires in: a store over the preset routes plus the two entry points — the quick
// list under a box's label and the preset window — already tied to that store and to each other ("프리셋 관리…" opens the window).
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { createPresetStore, presetApi } from "./preset_api.js";
import { PRESET_LABELS, PRESET_SECTIONS } from "./preset_logic.js";
import { anchorFinder, openPresetQuick } from "./preset_quick.js";
import { openPresetWindow } from "./preset_window.js";

/**
 * `route(path, body)` = studio_main's JSON helper. Returns
 * - `sections` / `labels`: the boxes that have presets (the prompt card's order, then `negative`) and their panel labels;
 * - `store`: the shared store (preset_api.js createPresetStore) — `load()`, `subscribe(fn)`, `list(key)`, `match(key, text)` …;
 * - `load()`: read the presets once (call it when the panel is made; resolves to the document, rejects when the route fails);
 * - `subscribe(fn)` -> unsubscribe: `fn()` after every load and change (redraw the labels' marks);
 * - `match(key, text)`: the preset whose text equals the box's (edges ignored, as the joined prompt ignores them) or null;
 * - `openQuick(anchor, key, {getText, setText, onNotice?, onManage?})`: the quick list under the box's label; null when that
 *   box's list was open (the label is a toggle). `anchor` is the label element — `ctx.boxLabel(key, thatButton)` hands it
 *   over — or a function giving the label that is on the page now. The panel may redraw (setText, a subscriber) while the
 *   list is open: a label that left the page is found again as `.ais-box-label[data-section="<key>"]` in the same panel
 *   (preset_quick.js anchorFinder). `getText(key)` reads the box, `setText(key, text)` writes it into the panel state,
 *   `onNotice(message)` gets what was done; "프리셋 관리…" opens the window unless `onManage(key)` is given;
 * - `openWindow(key, {getText, setText, onClose?, returnFocus?})`: the preset window on that box's tab (it can change every
 *   box, so `getText` / `setText` must take any of `sections`). `returnFocus()` gives the element to focus on closing when
 *   the one that had the focus at opening is gone (the list's own "프리셋 관리…" passes the box's label).
 */
export function createPresetTools(route) {
  const store = createPresetStore(presetApi(route));
  const openWindow = (key, { getText, setText, onClose, returnFocus } = {}) => openPresetWindow(key, { store, getText, setText, onClose, returnFocus });
  const openQuick = (anchor, key, { getText, setText, onNotice, onManage } = {}) => {
    const label = anchorFinder(anchor, key);
    return openPresetQuick(label, key, {
      store, getText, setText, onNotice,
      onManage: onManage || ((managed) => openWindow(managed, { getText, setText, returnFocus: label })),
    });
  };
  return {
    sections: PRESET_SECTIONS,
    labels: PRESET_LABELS,
    store,
    load: () => store.load(),
    subscribe: (fn) => store.subscribe(fn),
    match: (key, text) => store.match(key, text),
    openQuick,
    openWindow,
  };
}
