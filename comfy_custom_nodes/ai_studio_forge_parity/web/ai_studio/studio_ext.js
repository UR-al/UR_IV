// Anima studio: pure logic for the extension table (ext_schema.js, generated from the app's specs).
// No `window`, no ComfyUI imports, no side effects (ComfyUI also loads this file as an extension).
// `normalizeExt` mirrors studio_state.py; the inputs and modes mirror core/comfy_workflow_compiler.py
// (_add_anima_guidance, _add_detail_daemon, _speed_plan, _add_speed, _add_degrid), pinned by the golden `ext_compiler`.
import { EXT_SCHEMA } from "./ext_schema.js";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const S2_MODE = "Stochastic (S²)";
// Detail Daemon refuses these samplers (muerrilla: "not supported"), as the compiler's _add_detail_daemon does.
const DD_UNSUPPORTED_SAMPLERS = new Set(["dpm_adaptive", "heunpp2"]);
const UNPICKED_FILES = new Set(["", "None", "(auto-download)"]);
const PICKED_FIELD = Object.freeze({ lora_block: "lora_name", vae_2x: "vae_name", pid: "ckpt_name" });
const POST_RUN_KEYS = new Set(["lora_block"]); // the only file pick on a gallery run's path (the post row is R19)

// Detail Daemon and SPEED have one node per pass (the compiler adds them per pass): the table's `role` is the main
// pass, these are the others.
const PASS_ROLES = Object.freeze({
  ext_dd_hires: "detail_daemon",
  ext_speed_hires: "speed",
  ext_dd_sam3: "detail_daemon",
});

const EXTENSIONS = EXT_SCHEMA.extensions;
const BY_KEY = new Map(EXTENSIONS.map((item) => [item.key, item]));
const KEY_OF_ROLE = new Map([...EXTENSIONS.map((item) => [item.role, item.key]), ...Object.entries(PASS_ROLES)]);

export const EXT_ROLES = Object.freeze([...KEY_OF_ROLE.keys()]);

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const NUMBER_TEXT = /^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$/;

function numberValue(field, value) {
  const number = typeof value === "string" && NUMBER_TEXT.test(value) ? Number(value) : value;
  if (typeof number !== "number" || !Number.isFinite(number)) return field.default;
  return Math.max(field.min, Math.min(field.max, field.kind === "int" ? Math.trunc(number) : number));
}

/** One field's value coerced and clamped by the table (`field` is a row of ext_schema.js). */
export function extFieldValue(field, value) {
  if (field.kind === "bool") return typeof value === "boolean" ? value : field.default;
  if (field.kind === "int" || field.kind === "float") return numberValue(field, value);
  if (typeof value !== "string") return field.default;
  return field.kind === "choice" && field.choices.length && !field.choices.includes(value) ? field.default : value;
}

/** studio_state._ext: every extension with `on` and its fields, coerced and clamped by the table; unknown keys dropped. */
export function normalizeExt(raw) {
  const source = isObject(raw) ? raw : {};
  const out = {};
  for (const item of EXTENSIONS) {
    const values = isObject(source[item.key]) ? source[item.key] : {};
    out[item.key] = { on: values.on === true };
    for (const field of item.fields) out[item.key][field.key] = extFieldValue(field, values[field.key]);
  }
  return out;
}

export const EXT_DEFAULTS = Object.freeze(normalizeExt({}));

const sortedJson = (settings) =>
  JSON.stringify(Object.fromEntries(Object.entries(settings).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))));

function fieldInputs(item, values) {
  return Object.fromEntries(item.fields.filter((field) => field.input).map((field) => [field.input, values[field.key]]));
}

const sam3Cfg = (s) => (s.sam3.cfg > 0 ? s.sam3.cfg : s.cfg);

/** What a SAM3 pass samples with: its own steps and cfg (0 = the main values), the main sampler, scheduler and seed. */
export function sam3PassInputs(s) {
  return {
    steps: s.sam3.steps > 0 ? s.sam3.steps : s.steps,
    cfg: sam3Cfg(s),
    sampler_name: s.sampler,
    scheduler: s.scheduler,
    seed: s.seed,
  };
}

/** The suite's values as core/anima_guidance.build_args sends them: RDC by the original's rule, DAVE's blank blocks. */
function suiteValues(ext) {
  const { on, ...values } = ext.guidance;
  values.guid_rdc_enabled = values.guid_dcw_enabled && values.guid_rdc_enabled && values.guid_rdc_tau > 0;
  if (!values.guid_rdc_enabled) values.guid_rdc_tau = 0;
  if (!values.guid_dave_blocks.trim()) values.guid_dave_blocks = EXT_DEFAULTS.guidance.guid_dave_blocks;
  return values;
}

const s2Runs = (values) => values.guid_enabled && values.guid_slg_on && values.guid_slg_mode === S2_MODE;

function suiteSettings(s, extras) {
  const values = suiteValues(s.ext);
  const settings = { ...values, cfg_scale: s.cfg };
  if (!(s.tab === "t2i" && s.hires.on && !extras.postAction)) settings.guid_hiflow_enabled = false;
  if (s2Runs(values)) settings.guid_s2_seed = s.seed; // only then, so other settings do not change with the seed
  return settings;
}

function ddInputs(s, item, cfg) {
  const { on, dd_hires: hiresOnly, ...values } = s.ext.detail_daemon;
  return { ...item.fixed, settings_json: sortedJson({ ...values, [item.on_key]: true }), cfg_scale_override: cfg };
}

const speedInputs = (s, item, label) => ({
  ...item.fixed,
  ...fieldInputs(item, s.ext.speed),
  pass_label: label,
  image_seed: s.seed,
});

function tileInputs(s, item) {
  const { on, ...values } = s.ext.sam3_tile;
  return { ...item.fixed, settings_json: sortedJson({ ...values, ...sam3PassInputs(s) }) };
}

const ROLE_INPUTS = Object.freeze({
  ext_guidance: (s, item, extras) => ({ ...item.fixed, settings_json: sortedJson(suiteSettings(s, extras)) }),
  ext_dd: (s, item) => ddInputs(s, item, s.cfg),
  ext_dd_hires: (s, item) => ddInputs(s, item, s.cfg), // Forge's Detail Daemon reads the base cfg on the hires pass too
  ext_dd_sam3: (s, item) => ddInputs(s, item, sam3Cfg(s)),
  ext_speed: (s, item) => speedInputs(s, item, s.tab === "t2i" ? "base" : "img2img"),
  ext_speed_hires: (s, item) => speedInputs(s, item, "hires"),
  ext_lora_block: (s, item) => ({ ...item.fixed, ...fieldInputs(item, s.ext.lora_block), seed: s.seed }),
  ext_sam3_refine: (s, item) => ({ ...item.fixed, ...fieldInputs(item, s.ext.sam3_refine), ...sam3PassInputs(s) }),
  ext_sam3_tile: tileInputs,
});

const complete = (state) => ({ ...state, ext: normalizeExt(state.ext) });

/**
 * The inputs of the extension node at `role` for a complete state with a concrete seed. `extras.postAction` is the
 * gallery post-processing action of this run, if any. {} for a role that is not an extension's.
 */
export function extInputs(state, role, extras = {}) {
  const key = KEY_OF_ROLE.get(role);
  if (key === undefined) return {};
  const item = BY_KEY.get(key);
  const s = complete(state);
  const build = ROLE_INPUTS[role];
  return build ? build(s, item, extras) : { ...item.fixed, ...fieldInputs(item, s.ext[key]) };
}

/** The CNS inputs of ForgeNeoKSamplerCNS / ForgeNeoHiresFix: CNS runs in the sampler, the suite only carries its values. */
export function extCnsInputs(state) {
  const { guidance } = normalizeExt(state.ext);
  const values = guidance.on ? guidance : EXT_DEFAULTS.guidance;
  return {
    cns_enabled: guidance.on && values.guid_cns_enabled,
    cns_strength: values.guid_cns_strength,
    cns_gamma_power: values.guid_cns_gamma_power,
    cns_gamma_scale: values.guid_cns_gamma_scale,
  };
}

/** Refine and tile repair use the SAM3 mask, so it must run even when the SAM3 detailer is off. */
export function extNeedsSam3Mask(state) {
  const { sam3_refine: refine, sam3_tile: tile } = normalizeExt(state.ext);
  return refine.on || tile.on;
}

/** SPEED and DeGrid: on an i2i run only with their own switch (the app's core/anima_speed.plan, core/vae_degrid.plan). */
function runsOnTab(s, key) {
  return s.tab === "t2i" || s.ext[key].apply_img2img;
}

function ddRuns(s, sampler, cfg) {
  return s.ext.detail_daemon.on && !DD_UNSUPPORTED_SAMPLERS.has(sampler) && cfg > 0;
}

function roleRuns(s, role, postAction) {
  const key = KEY_OF_ROLE.get(role);
  const ext = s.ext[key];
  const hiresRuns = postAction ? postAction === "hires" : s.hires.on && s.tab === "t2i"; // Forge's img2img has no hires
  const sam3Runs = postAction ? postAction === "sam3" : s.sam3.on || extNeedsSam3Mask(s);
  const speed = s.ext.speed;
  switch (role) {
    case "ext_guidance": {
      const values = suiteValues(s.ext);
      return ext.on && BY_KEY.get(key).active_keys.some((name) => values[name]);
    }
    case "ext_dd":
      return ddRuns(s, s.sampler, s.cfg) && !ext.dd_hires;
    case "ext_dd_hires":
      return ddRuns(s, s.sampler, s.cfg) && ext.dd_hires && hiresRuns;
    case "ext_dd_sam3":
      return ddRuns(s, s.sampler, sam3Cfg(s)) && !ext.dd_hires && sam3Runs;
    case "ext_speed": // the compiler leaves SPEED out of an inpaint pass (the mask is bound to the full-size grid)
      return speed.on && runsOnTab(s, "speed") && !(s.tab === "i2i" && s.i2i_mode === "inpaint");
    case "ext_speed_hires":
      return speed.on && speed.hires && hiresRuns;
    case "ext_degrid":
      return ext.on && runsOnTab(s, "degrid");
    default:
      return ext.on && !(key in PICKED_FIELD && UNPICKED_FILES.has(ext[PICKED_FIELD[key]]));
  }
}

/** A COMBO's items from the `input` of `/object_info/<class>` (both the list form and V3's COMBO form); [] when unknown. */
export function comboItems(inputs, name) {
  const spec = inputs?.required?.[name] ?? inputs?.optional?.[name];
  if (!Array.isArray(spec)) return [];
  if (Array.isArray(spec[0])) return spec[0];
  return spec[0] === "COMBO" && Array.isArray(spec[1]?.options) ? spec[1].options : [];
}

function fileMissing(s, key, lists) {
  if (key in PICKED_FIELD) return UNPICKED_FILES.has(s.ext[key][PICKED_FIELD[key]]);
  if (key !== "degrid" || !runsOnTab(s, key)) return false;
  const inputs = lists?.get(BY_KEY.get(key).class); // unknown until the panel has read it: no guess
  return Boolean(inputs) && comboItems(inputs, "model_name").every((name) => name === "auto");
}

/**
 * Labels of the switched-on extensions that will do nothing for want of a file: a pick left empty (roleRuns bypasses
 * them), DeGrid with no model at all (the node keeps the image and only reports it). `postAction`: as in extInputs.
 * `lists`: class -> `/object_info/<class>` input, what the panel has read so far.
 */
export function idleExtensions(state, postAction = null, lists = null) {
  const s = complete(state);
  return EXTENSIONS.filter((item) => s.ext[item.key].on && (!postAction || POST_RUN_KEYS.has(item.key)) && fileMissing(s, item.key, lists))
    .map((item) => item.label);
}

/**
 * role -> 0 (run) | 4 (bypass) for the extension nodes; with `roleIds`, only the roles the graph has. A bypassed
 * MODEL/VAE/IMAGE node passes its first input of that type through, so an extension that is off costs nothing and
 * its file lists are not validated. `extras.postAction`: as in extInputs.
 */
export function extModes(state, roleIds = null, extras = {}) {
  const s = complete(state);
  const roles = EXT_ROLES.filter((role) => !roleIds || roleIds[role] !== undefined);
  return Object.fromEntries(roles.map((role) => [role, roleRuns(s, role, extras.postAction) ? MODE_ALWAYS : MODE_BYPASS]));
}
