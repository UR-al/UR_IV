// Anima studio: what the EXIF panel's 설정 가져오기 brings back. A studio image keeps the panel and the XYZ panel of its run in
// ComfyUI's `workflow` text, so they return as they were, even from an XYZ cell (the plot's base, not the cell). Any other image
// has its infotext, which studio_infotext_read.py reads through POST /infotext/parse.
// Pure ES module without side effects (ComfyUI also loads every .js under web/ as an extension).
import { INFOTEXT_KEYS } from "./studio_exif.js";
import { fileStem, missingLoras, pickCheckpoint, withDefaults } from "./studio_logic.js";

const PANEL = "AIStudioPanel";
const XYZ = "AIStudioXYZView";
const INFOTEXT_LINE = /^Steps: /m;
const HIRES_FILES = Object.freeze({ upscaler: "upscalers", vae: "vaes" });
const LISTED = Object.freeze({ unet: "unets", te: "tes", vae: "vaes", sampler: "samplers", scheduler: "schedulers" });

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

function parsed(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** The state a UI node saved in its hidden widget, null when `workflow` has none. */
function savedWidget(workflow, type) {
  const nodes = Array.isArray(workflow?.nodes) ? workflow.nodes : [];
  const value = parsed(nodes.find((node) => node?.type === type)?.widgets_values?.[0]);
  return isObject(value) && Object.keys(value).length ? value : null;
}

/** `{panel, xyz}` of a studio image, `{infotext}` of any other image with generation info, else null. */
export function imageSettings(entries) {
  const text = (key) => entries.find((entry) => entry.key === key)?.text;
  const workflow = parsed(text("workflow"));
  const panel = savedWidget(workflow, PANEL);
  if (panel) return { panel, xyz: savedWidget(workflow, XYZ) };
  const infotext = INFOTEXT_KEYS.map(text).find((value) => value && INFOTEXT_LINE.test(value));
  return infotext ? { infotext } : null;
}

const byName = (list, name) => (list || []).find((item) => item === name || fileStem(item) === name);

/** The saved panel, but for the tab and the i2i image: those are where the work is now, not settings. */
function wholeState(current, panel) {
  const state = withDefaults(panel);
  Object.assign(state, { tab: current.tab, i2i_mode: current.i2i_mode });
  state.i2i.image = current.i2i.image;
  return state;
}

/** The current panel with what an infotext names; the stems of `model`, `modules` and the hires files find their files. */
function namedState(current, { settings, model, modules = [] }, lists, missing) {
  const state = withDefaults(current);
  const found = (list, name) => byName(list, name) || void missing.push(name);
  const unet = model && found(lists.unets, model);
  if (unet) pickCheckpoint(state, unet, lists);
  for (const [key, value] of Object.entries(settings)) state[key] = isObject(value) ? { ...state[key], ...value } : value;
  for (const name of modules) {
    const vae = byName(lists.vaes, name);
    const te = vae ? "" : found(lists.tes, name);
    if (vae) state.vae = vae;
    if (te) state.te = te;
  }
  for (const [key, kind] of Object.entries(HIRES_FILES)) {
    const name = settings.hires?.[key];
    if (name) state.hires[key] = found(lists[kind], name) || current.hires[key];
  }
  return state;
}

/**
 * `{state, missing}`: `current` with `read`, which is `{panel}` (studio_settings.imageSettings) or `/infotext/parse`'s answer.
 * A file or sampler this ComfyUI lacks keeps the panel's pick and is named in `missing`; a list not read yet checks nothing.
 */
export function pulledState(current, read, lists = {}) {
  const missing = [];
  const state = read.panel ? wholeState(current, read.panel) : namedState(current, read, lists, missing);
  for (const [key, kind] of Object.entries(LISTED)) {
    if (!lists[kind]?.length || !state[key] || lists[kind].includes(state[key])) continue;
    missing.push(state[key]);
    state[key] = current[key];
  }
  missing.push(...missingLoras(state, lists.loras));
  return { state, missing: [...new Set(missing)] };
}
