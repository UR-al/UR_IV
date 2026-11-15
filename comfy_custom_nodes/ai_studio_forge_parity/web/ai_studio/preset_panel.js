// Prompt-box presets on the control panel: what studio_main.js hands the panel (panel_view.js `ctx.boxLabel` / `ctx.boxPreset`)
// so that a box's name opens the quick list ("프리셋 관리…" in it opens the preset window) and the line under every name
// says which preset the box holds. All eight boxes, the negative too.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { addPresetStylesheet } from "./preset_dom.js";

/**
 * `tools` = preset_tools.js createPresetTools(route). `panel()` -> the control panel's view (PanelView) or null while the
 * graph has none; `notice(text)` shows what a list did on the studio's notice line. Returns
 * - `boxLabel(key, anchorEl)`: the panel's hook — the quick list under that name (the name is a toggle);
 * - `boxPreset(key, text)`: the panel's hook — the preset equal to the box's text, null when there is none, undefined while
 *   the presets are not read (the panel then says nothing under the name);
 * - `load()`: reads the presets once (call it when a panel is made; a failed read is asked again by the next call) and
 *   asks for the lists' stylesheet, so the first list a name opens is already styled;
 * - `close()`: closes the quick list that is open.
 * The panel's marks are filled again after every read and change of the store (`PanelView.refreshBoxMarks`: no panel redraw).
 */
export function wirePresets(tools, { panel, notice } = {}) {
  const io = {
    getText: (key) => panel()?.state[key] ?? "",
    setText: (key, text) => panel()?.set((state) => { state[key] = text; }, true),
    onNotice: notice,
  };
  let quick = null;
  let reading = false;
  tools.subscribe(() => panel()?.refreshBoxMarks());
  return {
    boxLabel(key, anchorEl) {
      if (!panel()) return;
      quick = tools.openQuick(anchorEl, key, io) || null;
    },
    boxPreset: (key, text) => (tools.store.loaded ? tools.match(key, text) : undefined),
    load() {
      addPresetStylesheet();
      if (reading || tools.store.loaded) return;
      reading = true;
      tools.load()
        .catch((error) => console.warn("[AI Studio] prompt presets not read:", error))
        .finally(() => { reading = false; });
    },
    close() {
      quick?.close();
      quick = null;
    },
  };
}
