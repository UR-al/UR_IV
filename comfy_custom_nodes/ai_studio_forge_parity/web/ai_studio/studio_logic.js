// Anima studio: pure logic between the control-panel state and ComfyUI's API prompt.
// No `window`, no ComfyUI imports, no side effects (ComfyUI also loads this file as an extension).
// Mirrors studio_state.py (defaults, effective prompt), studio_choices.py (unet family) and reads
// studio_infotext.py's text; js_tests/fixtures/studio_logic_golden.json pins both sides.
import { EXT_DEFAULTS, EXT_ROLES, extCnsInputs, extInputs, extModes, extNeedsSam3Mask, normalizeExt, sam3PassInputs } from "./studio_ext.js";
import { ROLES } from "./studio_roles.js";

export const DORA_MODES = Object.freeze([
  "LyCORIS fp32 (as trained)",
  "Forge/Comfy fp32",
  "Forge/Comfy (stock)",
  "DoRA off (no magnitude)",
]);
const DORA_FORGE_MODE = "LyCORIS fp32 (as trained)"; // "Forge처럼" DoRA = sam-extra's LyCORIS mode, the Forge txt2img setting (stock overshoots adaLN rows)
const LATENT_FALLBACK_UPSCALER = "latent:bicubic";
const SEED_RANGE = 4294967294; // Forge: random.randrange(4294967294)
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

function deepFreeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

export const DEFAULT_STATE = deepFreeze({
  version: 1,
  tab: "t2i",
  i2i_mode: "img2img",
  prompt: "",
  negative: "",
  unet: "",
  te: "",
  vae: "",
  sampler: "exp_heun_2_x0_sde",
  scheduler: "flow_cosmos_dynamic",
  steps: 32,
  cfg: 3.0,
  shift: 3.0,
  seed: -1,
  width: 832,
  height: 1216,
  batch: 1,
  loras: [],
  negpip: true,
  dora_mode: DORA_MODES[0],
  forge_like: { master: true, noise: true, prompt_syntax: true, dora: true, attention: false },
  hires: { on: false, upscaler: "4x-UltraSharp.pth", sampler: "", scale: 1.5, steps: 16, denoise: 0.4 },
  upscale: { on: false, model: "", scale: 2.0 },
  sam3: { on: false, checkpoint: "", prompt: "face", denoise: 0.4, steps: 0, cfg: 0.0 },
  // inpaint_area: no node input takes it yet (ForgeNeoLatentInput inpaints the whole picture), so the panel shows
  // "전체" greyed out; the key stays so a saved state keeps its shape.
  i2i: { image: "", width: 832, height: 1216, denoise: 0.55, mask_blur: 4, inpaint_area: "whole" },
  ext: EXT_DEFAULTS, // the table in ext_schema.js: one {on, fields...} per extension
});

const SECTIONS = new Set(["forge_like", "hires", "upscale", "sam3", "i2i"]);
const POST_SECTIONS = Object.freeze({ upscale: "upscale", hires: "hires", sam3: "sam3" });
// Post runs route the image around the stages they do not use, so e.g. an upscale never takes a VAE round trip.
const POST_ROUTES = Object.freeze({
  upscale: [["post_sam3_detailer", "post_upscale"]],
  hires: [["post_upscale", "post_image"], ["post_sam3_detailer", "post_decode"]],
  sam3: [["post_decode", "post_image"]],
});

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

function normalizeLora(item) {
  const strength = typeof item.strength === "number" && Number.isFinite(item.strength) ? item.strength : 1.0;
  return {
    on: typeof item.on === "boolean" ? item.on : true,
    name: typeof item.name === "string" ? item.name : "",
    strength: Math.max(-10, Math.min(10, strength)),
  };
}

/** A complete copy of `state`: missing keys (an older saved state) come from DEFAULT_STATE, unknown keys are dropped. */
export function withDefaults(state) {
  const raw = isObject(state) ? state : {};
  const out = structuredClone(DEFAULT_STATE);
  for (const key of Object.keys(DEFAULT_STATE)) {
    const value = raw[key];
    if (value === undefined || value === null) continue;
    if (SECTIONS.has(key)) out[key] = { ...out[key], ...(isObject(value) ? value : {}) };
    else if (key === "loras") out.loras = Array.isArray(value) ? value.filter(isObject).map(normalizeLora) : [];
    else if (key === "ext") out.ext = normalizeExt(value);
    else out[key] = structuredClone(value);
  }
  return out;
}

/** Python's `f"{v:g}"` (6 significant digits, trailing zeros dropped, exponent below 1e-4 / from 1e6). */
function formatG(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "nan" : v > 0 ? "inf" : "-inf";
  if (v === 0) return Object.is(v, -0) ? "-0" : "0";
  const dropZeros = (text) => (text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text);
  const [mantissa, expText] = v.toExponential(5).split("e");
  const exp = Number(expText);
  if (exp < -4 || exp >= 6) {
    return `${dropZeros(mantissa)}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return dropZeros(v.toFixed(5 - exp));
}

/** studio_state.effective_prompt: the prompt plus `<lora:name:strength>` for enabled list LoRAs not already tagged (R3). */
export function effectivePrompt(state) {
  const prompt = state.prompt || "";
  const tagged = new Set([...prompt.matchAll(/<lora:([^:>]*)[:>]/g)].map((match) => match[1]));
  const tags = (state.loras || [])
    .filter((item) => item.on && item.name && !tagged.has(item.name))
    .map((item) => `<lora:${item.name}:${formatG(item.strength)}>`);
  return [...(prompt ? [prompt] : []), ...tags].join(" ");
}

/** `"2.9b" | "base" | "3.8b"`: the `/choices?kind=unets` families map when it knows the file, else the R5 name rule. */
export function unetFamily(name, families) {
  const file = typeof name === "string" ? name : "";
  if (isObject(families) && Object.prototype.hasOwnProperty.call(families, file)) return families[file];
  if (/(?<!\d)3[._-]?8b/i.test(file)) return "3.8b";
  if (file.toLowerCase().includes("base")) return "base";
  return "2.9b";
}

const SIZE_STEP = 8;
const SIZE_MIN = 64;
const SIZE_MAX = 4096;

/**
 * The generation size for an i2i / inpaint input image (Forge's send-to carries the image's size): each side rounded
 * to a multiple of 8, both shrunk together to fit 4096 so the aspect holds; null when the size is unknown.
 */
export function imageSize(width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!(Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0)) return null;
  const fit = Math.min(1, SIZE_MAX / Math.max(w, h)); // shrink both sides, or a big image would be cropped to fit
  const side = (value) => Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.round((value * fit) / SIZE_STEP) * SIZE_STEP));
  return { width: side(w), height: side(h) };
}

/**
 * The model file the panel picks when its list arrives and nothing is chosen yet (an empty pick would leave the
 * loader on whatever file sorts first): a 2.9B-family Anima UNET, the Qwen3 0.6B TE, the Qwen-Image / Anima VAE.
 */
export function defaultModel(kind, items, families) {
  const names = (Array.isArray(items) ? items : []).filter((item) => typeof item === "string");
  const find = (pattern) => names.find((name) => pattern.test(name)) || "";
  if (kind === "unets") return names.find((name) => /anima/i.test(name) && unetFamily(name, families) === "2.9b") || "";
  if (kind === "tes") return find(/qwen.*0[._-]?6b/i);
  if (kind === "vaes") return find(/qwen[_-]?image/i) || find(/anima/i);
  return "";
}

export const PICK_NOTICES = Object.freeze({
  models: "체크포인트·TE·VAE 고르기",
  upscale: "업스케일 모델 고르기",
});

/**
 * Why a run of `state` must not start (an empty pick would run the loader's first file), else "". `action` is a
 * gallery post-processing run ("upscale" needs only the upscale model), null for a generation.
 */
export function missingPicks(state, action = null) {
  const s = withDefaults(state);
  const upscale = action === "upscale" || (action === null && s.upscale.on);
  if (action !== "upscale" && (!s.unet || !s.te || !s.vae)) return PICK_NOTICES.models;
  if (upscale && !s.upscale.model) return PICK_NOTICES.upscale;
  return "";
}

const LORA_EXTENSIONS = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"];

function loraTagName(args) {
  const positional = args.split(":").filter((item) => item.split("=").length !== 2);
  return (positional[0] || "").trim();
}

/** Names of the `<lora:...>` tags a run loads (studio_lora_tags.parse_lora_tags: positive, then the negative's). */
export function loraTagNames(state) {
  const s = withDefaults(state);
  const names = [];
  for (const match of effectivePrompt(s).matchAll(/<(\w+):([^>]+)>/g)) if (match[1] === "lora") names.push(loraTagName(match[2]));
  for (const match of (s.negative || "").matchAll(/<(lora):([^>]+)>/gi)) if (match[1] === "lora") names.push(loraTagName(match[2]));
  return [...new Set(names.filter(Boolean))];
}

function stripLoraExtension(path) {
  const lowered = path.toLowerCase();
  const extension = LORA_EXTENSIONS.find((item) => lowered.endsWith(item));
  return extension ? path.slice(0, -extension.length) : path;
}

/** studio_lora_tags.resolve_lora_name: the name with or without extension, else the file stem in a subfolder. */
export function resolveLoraName(name, available) {
  const fold = (text) => stripLoraExtension(String(text).replace(/\\/g, "/")).toLowerCase();
  const wanted = fold(String(name || "").trim());
  if (!wanted) return null;
  const stems = (available || []).map((entry) => [fold(entry), entry]);
  return (stems.find(([stem]) => stem === wanted) || stems.find(([stem]) => stem.endsWith(`/${wanted}`)) || [])[1] ?? null;
}

/** The tag names of `state` that match no file of `available` (the LoRA node skips them). */
const NEGATIVE_WEIGHT = /:\s*-\s*\d[\d.]*\s*\)/;

/** True when the prompt or negative uses a negative attention weight `(x:-1.0)`, which only NegPiP makes sense of. */
export function hasNegativeWeights(state) {
  const s = withDefaults(state);
  return NEGATIVE_WEIGHT.test(s.prompt) || NEGATIVE_WEIGHT.test(s.negative);
}

export function missingLoras(state, available) {
  if (!Array.isArray(available)) return [];
  return loraTagNames(state).filter((name) => resolveLoraName(name, available) === null);
}

function concreteSeed(seed, rng) {
  const value = Number(seed);
  return Number.isInteger(value) && value >= 0 ? value : Math.floor(rng() * SEED_RANGE);
}

/** A choice the state may leave empty ("" = keep the workflow's own widget value). */
const choice = (value) => (typeof value === "string" && value !== "" ? value : undefined);

function hiresUpscaler(name, upscalers) {
  if (Array.isArray(upscalers) && !name.startsWith("latent:") && !upscalers.includes(name)) return LATENT_FALLBACK_UPSCALER;
  return name;
}

function effectiveDoraMode(s) {
  if (s.forge_like.master && s.forge_like.dora) return DORA_FORGE_MODE;
  return DORA_MODES.includes(s.dora_mode) ? s.dora_mode : DORA_MODES[0];
}

function hiresInputs(s, enabled, options) {
  return {
    enabled,
    scale_by: s.hires.scale,
    upscale_method: hiresUpscaler(s.hires.upscaler, options.upscalers),
    steps: s.hires.steps,
    denoise: s.hires.denoise,
    seed: s.seed,
    shift: s.shift,
    sampler_name: s.hires.sampler || s.sampler,
    scheduler: s.scheduler,
    cfg: s.cfg,
    base_steps: s.steps,
    forge_noise: s.forge_like.master && s.forge_like.noise,
    ...extCnsInputs(s),
  };
}

const upscaleInputs = (s, enabled) => ({ enabled, model_name: choice(s.upscale.model), scale_by: s.upscale.scale });

const sam3MaskInputs = (s, enabled) => ({ enabled, checkpoint: choice(s.sam3.checkpoint), prompt: s.sam3.prompt });

function sam3DetailerInputs(s, enabled) {
  return { enabled, denoise: s.sam3.denoise, ...sam3PassInputs(s) };
}

const saveInputs = (s) => ({ studio_state: JSON.stringify(s), seed: s.seed });

/** role -> {input: value} for the generation graph; `s` is complete and its seed concrete. */
function mainInputs(s, options) {
  const is38 = unetFamily(s.unet, options.families) === "3.8b";
  const isI2i = s.tab === "i2i";
  const select = { select: is38 ? 1 : 0 };
  const encode = { steps: s.steps, forge_syntax: s.forge_like.master && s.forge_like.prompt_syntax };
  return {
    unet: is38 ? {} : { unet_name: choice(s.unet) },
    unet38: is38 ? { model_name: choice(s.unet) } : {},
    te: { clip_name: choice(s.te) },
    vae: { vae_name: choice(s.vae) },
    family_switch_model: select,
    family_switch_pos: select,
    family_switch_neg: select,
    shift: { shift: s.shift },
    lora_stack: { positive: effectivePrompt(s), negative: s.negative },
    negpip: { enabled: Boolean(s.negpip) },
    dora: { enabled: true, mode: effectiveDoraMode(s) },
    pos_encode: encode,
    neg_encode: encode,
    latent: {
      mode: !isI2i ? "txt2img" : s.i2i_mode === "inpaint" ? "inpaint" : "img2img",
      width: (isI2i ? s.i2i : s).width,
      height: (isI2i ? s.i2i : s).height,
      batch_size: s.batch,
      mask_blur: s.i2i.mask_blur,
    },
    i2i_image: { image: choice(s.i2i.image) },
    sampler: {
      seed: s.seed,
      steps: s.steps,
      cfg: s.cfg,
      sampler_name: s.sampler,
      scheduler: s.scheduler,
      denoise: isI2i ? s.i2i.denoise : 1.0,
      forge_noise: s.forge_like.master && s.forge_like.noise,
      ...extCnsInputs(s),
    },
    hires: hiresInputs(s, s.hires.on && !isI2i, options), // Forge's img2img has no hires pass
    upscale: upscaleInputs(s, s.upscale.on),
    sam3_mask: sam3MaskInputs(s, s.sam3.on || extNeedsSam3Mask(s)),
    sam3_detailer: sam3DetailerInputs(s, s.sam3.on),
    save: saveInputs(s),
    ...Object.fromEntries(EXT_ROLES.map((role) => [role, extInputs(s, role, { postAction: options.postAction })])),
  };
}

/** Writes `table` into the prompt's role nodes; skips absent (bypassed) nodes, linked inputs and undefined values. */
function writeRoles(prompt, roleIds, table) {
  for (const [role, values] of Object.entries(table)) {
    const id = roleIds?.[role];
    const node = id === undefined || id === null ? undefined : prompt[String(id)];
    if (!node || !isObject(node.inputs)) continue;
    for (const [name, value] of Object.entries(values)) {
      if (value === undefined || Array.isArray(node.inputs[name])) continue;
      node.inputs[name] = value;
    }
  }
}

function prepared(state, rng) {
  const s = withDefaults(state);
  s.seed = concreteSeed(s.seed, rng || Math.random);
  return s;
}

/**
 * A deep copy of ComfyUI's API prompt (graphToPrompt `output`) with the panel state written into the role nodes.
 * `options`: `rng` (seed -1), `families` (/choices?kind=unets), `upscalers` (hires choices; a missing file -> latent:bicubic).
 * An extension whose role the graph lacks (an older workflow) is skipped.
 */
export function applyStateToPrompt(apiPrompt, roleIds, state, options = {}) {
  const s = prepared(state, options.rng);
  const out = structuredClone(apiPrompt);
  writeRoles(out, roleIds, mainInputs(s, options));
  return out;
}

/**
 * role -> 0 (run) | 4 (bypass): the unused family, the i2i input in t2i, SAM3 when off, post nodes unless a post run
 * (R7), and every extension that does not run (studio_ext.extModes).
 */
export function stageModes(state, postAction = null, options = {}) {
  const s = withDefaults(state);
  const bypass = new Set(
    unetFamily(s.unet, options.families) === "3.8b"
      ? ["unet", "pos_encode", "neg_encode"]
      : ["unet38", "qwen35", "pos_encode38", "neg_encode38"],
  );
  if (s.tab === "t2i") bypass.add("i2i_image");
  if (!s.sam3.on || s.ext.sam3_refine.on) bypass.add("sam3_detailer"); // R22: Refine replaces it, as in the app's post run
  if (!s.sam3.on && !extNeedsSam3Mask(s)) bypass.add("sam3_mask");
  if (!postAction) ROLES.filter((role) => role.startsWith("post_")).forEach((role) => bypass.add(role));
  const ext = extModes(s, null, { postAction });
  return Object.fromEntries(ROLES.map((role) => [role, bypass.has(role) ? MODE_BYPASS : ext[role] ?? MODE_ALWAYS]));
}

/**
 * The state a gallery post-processing run uses: the record's settings (the image's own prompt, seed, models …;
 * keys it lacks come from the current state) with the post-processing sections from the current state, only the
 * `action`'s stage on (spec §4). A record without `loras` (read from `parameters`) already has its tags in the prompt.
 */
export function postRunState(record, currentState, action) {
  const current = withDefaults(currentState);
  const generation = { ...(isObject(record) ? record : {}) };
  for (const section of Object.values(POST_SECTIONS)) delete generation[section];
  const merged = withDefaults({ ...current, loras: [], ...generation });
  for (const section of Object.values(POST_SECTIONS)) {
    merged[section] = { ...current[section], on: POST_SECTIONS[action] === section };
  }
  return merged;
}

/** A session image ref `{filename, subfolder, type}` (or a plain input name) -> LoadImage's value. */
function loadImageValue(ref) {
  if (typeof ref === "string") return ref.trim();
  if (!isObject(ref) || typeof ref.filename !== "string" || !ref.filename) return "";
  const path = ref.subfolder ? `${ref.subfolder}/${ref.filename}` : ref.filename;
  return ref.type && ref.type !== "input" ? `${path} [${ref.type}]` : path;
}

function routeAround(prompt, roleIds, action) {
  for (const [fromRole, toRole] of POST_ROUTES[action]) {
    const from = roleIds[fromRole];
    const to = roleIds[toRole];
    if (from === undefined || to === undefined || !prompt[String(from)] || !prompt[String(to)]) continue;
    for (const node of Object.values(prompt)) {
      for (const [name, value] of Object.entries(node.inputs || {})) {
        if (Array.isArray(value) && String(value[0]) === String(from) && value[1] === 0) node.inputs[name] = [String(to), 0];
      }
    }
  }
}

/**
 * Gallery post-processing (`"upscale" | "hires" | "sam3"`) of `record.image`: `{prompt, outputNodeIds}` where
 * outputNodeIds goes to `/prompt` as `partial_execution_targets`, so only the post row runs.
 */
export function postPrompt(apiPrompt, roleIds, record, action, currentState = record, options = {}) {
  if (!Object.prototype.hasOwnProperty.call(POST_SECTIONS, action)) throw new Error(`postPrompt: unknown action "${action}"`);
  const image = loadImageValue(record?.image);
  if (!image) throw new Error("postPrompt: the record has no image");
  const s = prepared(postRunState(record, currentState, action), options.rng);
  const out = structuredClone(apiPrompt);
  writeRoles(out, roleIds, mainInputs(s, { ...options, postAction: action }));
  writeRoles(out, roleIds, {
    post_image: { image },
    post_upscale: upscaleInputs(s, action === "upscale"),
    post_hires: hiresInputs(s, action === "hires", options),
    post_sam3_mask: sam3MaskInputs(s, action === "sam3"),
    post_sam3_detailer: sam3DetailerInputs(s, action === "sam3"),
    post_save: saveInputs(s),
  });
  routeAround(out, roleIds, action);
  return { prompt: out, outputNodeIds: [String(roleIds.post_save)] };
}

function infotextParams(line) {
  const params = {};
  for (const match of line.matchAll(/\s*([^,:"]+?):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)/g)) {
    let value = match[2].trim();
    if (value.startsWith('"')) {
      try {
        value = JSON.parse(value);
      } catch {
        // keep the raw text
      }
    }
    params[match[1].trim()] = value;
  }
  return params;
}

const intOf = (text) => (/^-?\d+$/.test(text ?? "") ? Number(text) : null);

/** A1111/Forge `parameters` text -> part of a state (prompt, negative, seed, steps, cfg, width, height); null if unreadable. */
export function recordFromParameters(text) {
  if (typeof text !== "string" || !text.trim()) return null;
  const lines = text.replace(/\r\n?/g, "\n").trim().split("\n");
  const params = infotextParams(lines.pop());
  const steps = intOf(params.Steps);
  const seed = intOf(params.Seed);
  if (steps === null || seed === null) return null;
  const negativeAt = lines.findIndex((line) => line.startsWith("Negative prompt:"));
  const promptLines = negativeAt < 0 ? lines : lines.slice(0, negativeAt);
  const negativeLines = negativeAt < 0 ? [] : [lines[negativeAt].slice("Negative prompt:".length), ...lines.slice(negativeAt + 1)];
  const record = { prompt: promptLines.join("\n").trim(), negative: negativeLines.join("\n").trim(), seed, steps };
  const cfg = Number.parseFloat(params["CFG scale"]);
  if (Number.isFinite(cfg)) record.cfg = cfg;
  const size = /^(\d+)x(\d+)$/.exec(params.Size ?? "");
  if (size) {
    record.width = Number(size[1]);
    record.height = Number(size[2]);
  }
  return record;
}

/** One prompt per XYZ cell (the `/xyz/plan` result), in `plan.order` (Forge's cost order). */
export function xyzCellPrompts(basePrompt, roleIds, plan, options = {}) {
  const cells = plan?.cells || [];
  return (plan?.order || []).map((cellIndex) => ({
    cellIndex,
    prompt: applyStateToPrompt(basePrompt, roleIds, cells[cellIndex].state, options),
  }));
}
