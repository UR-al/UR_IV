// Anima studio: pure logic between the control-panel state and ComfyUI's API prompt.
// No `window`, no ComfyUI imports, no side effects (ComfyUI also loads this file as an extension).
// Mirrors studio_state.py (defaults, effective prompt), studio_choices.py (unet family) and reads
// studio_infotext.py's text; js_tests/fixtures/studio_logic_golden.json pins both sides.
import { EXT_DEFAULTS, EXT_ROLES, extCnsInputs, extInputs, extModes, extNeedsSam3Mask, normalizeExt, sam3MaskOnly, sam3PassInputs } from "./studio_ext.js";
import { postDefaults, postInputs } from "./studio_post.js";
import { ROLES } from "./studio_roles.js";

export const DORA_MODES = Object.freeze([
  "LyCORIS fp32 (as trained)",
  "Forge/Comfy fp32",
  "Forge/Comfy (stock)",
  "DoRA off (no magnitude)",
]);
const DORA_FORGE_MODE = "LyCORIS fp32 (as trained)"; // "Forge처럼" DoRA = sam-extra's LyCORIS mode, the Forge txt2img setting (stock overshoots adaLN rows)
const LATENT_FALLBACK_UPSCALER = "latent:bicubic";
const HIRES_SAME_VAE = "Use same choices"; // ForgeNeoHiresFix's first vae_name choice
const NO_SECOND_UPSCALER = "None"; // AIStudioUpscaleImage's first model_name_2 choice
const IDLE_UPSCALER = "Lanczos"; // always in AIStudioUpscaleImage's list: a moved file of a switched-off stage must not stop a run
const CPU = "cpu";
const CN_RESIZE_MODE = "Crop and Resize"; // no effect without a control_image, which the studio never links
const INDIVIDUAL_MASKS = "Individual";
const SEED_RANGE = 4294967294; // Forge: random.randrange(4294967294)
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const MASK_OVERLAY_SLOT = 3; // ForgeNeoSAM3Mask's overlay output
const FORGE_ATTENTION = "pytorch"; // "Forge처럼": the user's Forge runs --use-pytorch-cross-attention
export const REFERENCE_METHODS = Object.freeze(["snapref", "ref_adapter"]);
export const REFERENCE_CORRECTIONS = Object.freeze(["apg", "rescale", "none"]);
export const KREA2_POSITIONS = Object.freeze(["aligned (trained)", "beside (right of target)"]);
export const INPAINT_FILLS = Object.freeze(["original", "fill", "latent noise", "latent nothing"]);

function deepFreeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

// The prompt card's boxes in the order they join (studio_state.PROMPT_SECTIONS).
export const PROMPT_SECTIONS = Object.freeze(["prompt_prefix", "prompt_char_count", "prompt_character", "prompt_copyright", "prompt_artist", "prompt", "prompt_suffix"]);
const PROMPT_BOXES = PROMPT_SECTIONS.filter((key) => key !== "prompt");
const EMPTY_PROMPT_BOXES = Object.fromEntries(PROMPT_BOXES.map((key) => [key, ""]));

export const DEFAULT_STATE = deepFreeze({
  version: 1,
  tab: "t2i",
  i2i_mode: "img2img",
  prompt: "",
  negative: "",
  ...EMPTY_PROMPT_BOXES,
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
  moss_negpip: false, // MOSS's own switch: on, its tokenizer reads a plain sentence's parentheses as weights
  attention: "launch", // what ComfyUI was started with, else a registered attention function
  moss_stream: true, // MOSS-Image only: the fp32 residual stream
  family_memory: {}, // family -> the model picks and sampling numbers it had when the checkpoint left it
  dora_mode: DORA_MODES[0],
  forge_like: { master: false, noise: true, prompt_syntax: true, dora: true, attention: true },
  // steps: 0 = the main steps; the rest of these three sections is the table in post_schema.js
  hires: { on: false, upscaler: "4x-UltraSharp.pth", sampler: "", scale: 1.5, steps: 16, denoise: 0.4, ...postDefaults("hires") },
  upscale: { on: false, model: "", scale: 2.0, ...postDefaults("upscale") },
  sam3: { on: false, checkpoint: "", prompt: "face", denoise: 0.4, ...postDefaults("sam3") },
  // The inpaint_detailer role runs every inpaint. inpaint_area "masked" is Forge's Only masked, padding belongs to it.
  i2i: { image: "", width: 832, height: 1216, denoise: 0.55, mask_blur: 4, inpaint_area: "whole", inpaint_padding: 32, inpaint_fill: INPAINT_FILLS[0] },
  reference: {
    on: false,
    image: "",
    image_2: "", // SnapRef only: another picture of the same character
    method: REFERENCE_METHODS[0], // Anima's two adapters; Krea2 has its own section
    megapixels: 1.0,
    snapref: { file: "", mode: "cs", strength: 0.0, fidelity: 1.0, weight_1: 1.0, weight_2: 1.0, guidance_on: true, guidance: 4.5, correction: "apg", ref_end: 1.0, caption: "" },
    ref_adapter: { file: "", strength: 1.0, guidance: 4.5, correction: "apg", ref_end: 1.0 },
    krea2: { file: "", strength: 1.0, grounding_px: 768, boost: 1.0, position: KREA2_POSITIONS[0] },
  },
  ext: EXT_DEFAULTS, // the table in ext_schema.js: one {on, fields...} per extension
  // What a run's wildcards were written as (studio_wildcards.py): the saved info, never the panel's boxes.
  prompt_template: "",
  negative_template: "",
});

const SECTIONS = new Set(["forge_like", "hires", "upscale", "sam3", "i2i"]);
const REFERENCE_SECTIONS = ["snapref", "ref_adapter", "krea2"];
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

function normalizeReference(raw) {
  const value = isObject(raw) ? raw : {};
  const out = { ...structuredClone(DEFAULT_STATE.reference), ...value };
  for (const key of REFERENCE_SECTIONS) out[key] = { ...DEFAULT_STATE.reference[key], ...(isObject(value[key]) ? value[key] : {}) };
  return out;
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
    else if (key === "reference") out.reference = normalizeReference(value);
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

const promptEdge = (text) => (typeof text === "string" ? text.replace(/^[ \t\r\n,]+|[ \t\r\n,]+$/g, "") : "");

/** studio_state.assembled_prompt: the boxes in PROMPT_SECTIONS order; with only the main box the prompt exactly as written. */
export function assembledPrompt(state) {
  if (!PROMPT_BOXES.some((key) => promptEdge(state[key]))) return state.prompt || "";
  return PROMPT_SECTIONS.map((key) => promptEdge(state[key])).filter(Boolean).join(", ");
}

/** studio_state.effective_prompt: the assembled prompt plus `<lora:name:strength>` for enabled list LoRAs not already tagged (R3). */
export function effectivePrompt(state) {
  const prompt = assembledPrompt(state);
  const tagged = new Set([...prompt.matchAll(/<lora:([^:>]*)[:>]/g)].map((match) => match[1]));
  const tags = (state.loras || [])
    .filter((item) => item.on && item.name && !tagged.has(item.name))
    .map((item) => `<lora:${item.name}:${formatG(item.strength)}>`);
  return [...(prompt ? [prompt] : []), ...tags].join(" ");
}

/** `"2.9b" | "base" | "3.8b" | "moss" | "krea2"`: the `/choices?kind=unets` families map when it knows the file, else the R5 name rule. */
export function unetFamily(name, families) {
  const file = typeof name === "string" ? name : "";
  if (isObject(families) && Object.prototype.hasOwnProperty.call(families, file)) return families[file];
  const lowered = file.toLowerCase();
  if (lowered.includes("moss")) return "moss";
  if (lowered.includes("krea")) return "krea2";
  if (/(?<!\d)3[._-]?8b/i.test(file)) return "3.8b";
  return lowered.includes("base") ? "base" : "2.9b";
}

/** `"anima" | "moss" | "krea2"`, by file name alone: which TE type, nodes and default settings a checkpoint runs on. */
export function modelFamily(name) {
  const family = unetFamily(name);
  return family === "moss" || family === "krea2" ? family : "anima";
}

// te / vae: the file picked when a family has no remembered pick. Numbers: the MOSS AI Studio workflow, comfyui-krea2edit's
// README (Turbo 8 steps CFG 1; Raw 20 steps CFG 3) and the core models' default shift (Krea2's is an exponent, mu).
const FAMILY_PROFILES = Object.freeze({
  anima: {
    clip_type: "stable_diffusion",
    te: /qwen.*0[._-]?6b/i,
    vae: /qwen[_-]?image/i,
    settings: { sampler: DEFAULT_STATE.sampler, scheduler: DEFAULT_STATE.scheduler, steps: DEFAULT_STATE.steps, cfg: DEFAULT_STATE.cfg, shift: DEFAULT_STATE.shift },
  },
  moss: {
    clip_type: "moss_image",
    te: /qwen3\.?5.*9b/i,
    vae: /moss_image_vae/i,
    settings: { sampler: "euler", scheduler: "sgm_uniform", steps: 50, cfg: 3.0, shift: 5.0 },
  },
  krea2: {
    clip_type: "krea2",
    te: /qwen3vl_4b/i,
    vae: /(^|[\\/])qwen_image_vae\.safetensors$/i,
    settings: { sampler: "euler", scheduler: "simple", steps: 20, cfg: 3.0, shift: 1.15 },
  },
});
const FAMILY_MEMORY_KEYS = ["te", "vae", "sampler", "scheduler", "steps", "cfg", "shift"];

const isKrea2Turbo = (unet) => modelFamily(unet) === "krea2" && /turbo/i.test(unet);
/** Turbo is distilled and takes other numbers than Raw, so it keeps its own picks. */
const memorySlot = (unet) => (isKrea2Turbo(unet) ? "krea2_turbo" : modelFamily(unet));

function familySettings(family, unet) {
  const settings = FAMILY_PROFILES[family].settings;
  return isKrea2Turbo(unet) ? { ...settings, steps: 8, cfg: 1.0 } : settings;
}

/**
 * Picks checkpoint `unet` in `state`. Leaving a family keeps its model picks and sampling numbers in `family_memory`;
 * entering one brings back its own, or its defaults with the first matching file of `lists` ({tes, vaes}).
 */
export function pickCheckpoint(state, unet, lists = {}) {
  const from = memorySlot(state.unet);
  const to = memorySlot(unet);
  const sameFamily = modelFamily(state.unet) === modelFamily(unet);
  const family = modelFamily(unet);
  state.unet = unet;
  if (from === to) return;
  state.family_memory = { ...state.family_memory, [from]: Object.fromEntries(FAMILY_MEMORY_KEYS.map((key) => [key, state[key]])) };
  Object.assign(state, familySettings(family, unet), sameFamily ? {} : { te: "", vae: "" }, state.family_memory[to]);
  if (!sameFamily && state.hires) state.hires = { ...state.hires, vae: "" };
  state.te ||= defaultModel("tes", lists.tes, null, family);
  state.vae ||= defaultModel("vaes", lists.vaes, null, family);
}

const SIZE_STEP = 8;
const MOSS_SIZE_STEP = 16;
const SIZE_MIN = 64;
const SIZE_MAX = 4096;

/** The size grid of `family`: MOSS latents are 16x, a size off that grid would change without notice. */
export const sizeStep = (family) => (family === "moss" ? MOSS_SIZE_STEP : SIZE_STEP);
const familySide = (value, family) => (family === "moss" ? Math.floor(value / MOSS_SIZE_STEP + 0.5) * MOSS_SIZE_STEP : value);

/**
 * The generation size for an i2i / inpaint input image (Forge's send-to carries the image's size): each side rounded
 * to a multiple of `step`, both shrunk together to fit 4096 so the aspect holds; null when the size is unknown.
 */
export function imageSize(width, height, step = SIZE_STEP) {
  const w = Number(width);
  const h = Number(height);
  if (!(Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0)) return null;
  const fit = Math.min(1, SIZE_MAX / Math.max(w, h)); // shrink both sides, or a big image would be cropped to fit
  const side = (value) => Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.round((value * fit) / step) * step));
  return { width: side(w), height: side(h) };
}

/**
 * The model file the panel picks when its list arrives and nothing is chosen yet (an empty pick would leave the
 * loader on whatever file sorts first): a 2.9B-family Anima UNET, the Qwen3 0.6B TE, the Qwen-Image / Anima VAE.
 */
export function defaultModel(kind, items, families, family = "anima") {
  const names = (Array.isArray(items) ? items : []).filter((item) => typeof item === "string");
  const find = (pattern) => names.find((name) => pattern.test(name)) || "";
  if (kind === "unets") return names.find((name) => /anima/i.test(name) && unetFamily(name, families) === "2.9b") || "";
  if (kind === "tes") return find(FAMILY_PROFILES[family].te);
  if (kind === "vaes") return find(FAMILY_PROFILES[family].vae) || (family === "anima" ? find(/anima/i) : "");
  return "";
}

/** The adapter or edit-LoRA file a reference run of `family` reads, "" where the family has no reference. */
export function referenceFile(state, family) {
  const { reference } = state;
  if (family === "krea2") return reference.krea2.file;
  return family === "anima" ? reference[reference.method].file : "";
}

const referenceRuns = (s, family) => s.reference.on && family !== "moss";

/** The guidance suite when its row is on, else null. */
const guidanceOn = (s) => (s.ext.guidance.on ? s.ext.guidance : null);

function snaprefPagClash(s, family) {
  const guidance = guidanceOn(s);
  return Boolean(guidance) && family === "anima" && s.reference.method === "snapref" && guidance.guid_enabled
    && guidance.guid_attn_method === "PAG" && Number(guidance.guid_scale) !== 0;
}

/** What the guidance suite loses under Ref Adapter: a v2 file runs the blocks itself, past these three patches. */
export function refAdapterDrops(state) {
  const s = withDefaults(state);
  const guidance = guidanceOn(s);
  if (!guidance || modelFamily(s.unet) !== "anima" || !s.reference.on || s.reference.method !== "ref_adapter") return [];
  const seg = guidance.guid_enabled && guidance.guid_attn_method === "SEG";
  return [guidance.guid_dave_enabled && "DAVE", guidance.guid_slg_on && "SLG", seg && "SEG"].filter(Boolean);
}

export const PICK_NOTICES = Object.freeze({
  models: "체크포인트·TE·VAE 고르기",
  upscale: "업스케일 모델 고르기",
  sam3: "SAM3 모델 고르기",
  faceModel: "얼굴 모델 고르기",
  controlnet: "ControlNet 모델 고르기",
  referenceImage: "레퍼런스 이미지 고르기",
  referenceFile: "레퍼런스 파일 고르기",
  snaprefPag: "SnapRef 는 PAG 와 같이 못 씀: Ref Adapter 로 바꾸거나 PAG 끄기",
});

/** A gallery run that loads a model: upscale and Mask only draw nothing. */
export function postDraws(state, action) {
  return action !== "upscale" && !(action === "sam3" && sam3MaskOnly(withDefaults(state)));
}

/** What a SAM3 run lacks, else "". ControlNet belongs to the detailer alone, which Refine replaces (R22). `action`: as in missingPicks. */
export function missingSam3Picks(state, action = null) {
  const s = withDefaults(state);
  if (action !== null && action !== "sam3") return "";
  if (action === "sam3" || s.sam3.on || extNeedsSam3Mask(s)) {
    if (!s.sam3.checkpoint) return PICK_NOTICES.sam3;
    if (s.sam3.face_crop && !s.sam3.face_crop_model) return PICK_NOTICES.faceModel;
  }
  const detailer = action === "sam3" || (s.sam3.on && !s.ext.sam3_refine.on);
  return detailer && !sam3MaskOnly(s) && s.sam3.cn_enable && !s.sam3.cn_model ? PICK_NOTICES.controlnet : "";
}

/**
 * Why a run of `state` must not start (an empty pick would run the loader's first file), else "". `action` is a
 * gallery post-processing run ("upscale" needs only the upscale model), null for a generation. `sam3: false` leaves the
 * SAM3 picks out: a plot's cells may bring them (run_controller.js checks the cells).
 */
export function missingPicks(state, action = null, { sam3 = true } = {}) {
  const s = withDefaults(state);
  const upscale = action === "upscale" || (action === null && s.upscale.on);
  const draws = postDraws(s, action);
  if (draws && (!s.unet || !s.te || !s.vae)) return PICK_NOTICES.models;
  if (upscale && !s.upscale.model) return PICK_NOTICES.upscale;
  const sam3Pick = sam3 ? missingSam3Picks(s, action) : "";
  if (sam3Pick) return sam3Pick;
  const family = modelFamily(s.unet);
  if (draws && referenceRuns(s, family)) {
    if (!s.reference.image) return PICK_NOTICES.referenceImage;
    if (!referenceFile(s, family)) return PICK_NOTICES.referenceFile;
    if (snaprefPagClash(s, family)) return PICK_NOTICES.snaprefPag;
  }
  return "";
}

const LORA_EXTENSIONS = [".safetensors", ".sft", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".pkl"];

/** A `/choices?kind=loras` entry as a `<lora:...>` tag names it. */
export const fileStem = (entry) => entry.replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "");

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
  const written = String(name || "").trim().replace(/\\/g, "/").toLowerCase();
  const wanted = [written, stripLoraExtension(written)].filter(Boolean);
  if (!wanted.length) return null;
  const stems = (available || []).map((entry) => [stripLoraExtension(String(entry).replace(/\\/g, "/")).trim().toLowerCase(), entry]);
  const found = stems.find(([stem]) => wanted.includes(stem)) || stems.find(([stem]) => wanted.some((item) => stem.endsWith(`/${item}`)));
  return found ? found[1] : null;
}

/** The tag names of `state` that match no file of `available` (the LoRA node skips them). */
const NEGATIVE_WEIGHT = /:\s*-\s*\d[\d.]*\s*\)/;

/** True when the prompt or negative uses a negative attention weight `(x:-1.0)`, which only NegPiP makes sense of. */
export function hasNegativeWeights(state) {
  const s = withDefaults(state);
  return NEGATIVE_WEIGHT.test(assembledPrompt(s)) || NEGATIVE_WEIGHT.test(s.negative);
}

const PROMPT_WEIGHT = /\((?:[^()]|\([^()]*\))*:\s*-?(?:\d+\.?\d*|\.\d+)\s*\)/; // a tag's own `(series)` may sit inside
const hasWeight = (text) => PROMPT_WEIGHT.test(text.replace(/\\[()]/g, ""));

/** The prompts of their own that a run's Hires and SAM3 passes encode (only passes that run). `action`: as in missingPicks. */
function passPrompts(s, action) {
  const hires = [s.hires.prompt, s.hires.negative];
  const detailer = [s.sam3.inpaint_prompt, s.sam3.negative_prompt];
  const draws = !sam3MaskOnly(s);
  if (action) return action === "hires" ? hires : action === "sam3" && draws ? detailer : [];
  const { sam3_refine: refine, sam3_tile: tile } = s.ext;
  const stopped = s.sam3.on && !draws; // Mask only stops Refine and tile too
  return [
    ...(s.hires.on && s.tab === "t2i" ? hires : []),
    ...(s.sam3.on && draws && !refine.on ? detailer : []),
    ...(refine.on && !stopped ? [refine.inpaint_prompt, refine.negative_prompt] : []),
    ...(tile.on && !stopped ? [tile.inpaint_prompt, tile.negative_prompt] : []),
  ].filter(Boolean);
}

/** True when a prompt has an explicit `(x:1.2)`: Krea2's encoder reads it as words, and MOSS's without NegPiP. `action`: as in missingPicks. */
export function hasPromptWeights(state, action = null) {
  const s = withDefaults(state);
  return [assembledPrompt(s), s.negative, ...passPrompts(s, action)].filter(Boolean).some(hasWeight);
}

/** True when a Hires or SAM3 pass has a prompt of its own on 3.8B: the node encodes it without the v2 connector. */
export function ownPromptOn38(state, families, action = null) {
  const s = withDefaults(state);
  return unetFamily(s.unet, families) === "3.8b" && passPrompts(s, action).some(Boolean);
}

/** NegPiP needs a text path that reads weights: ComfyUI-ppm patches Anima's and MOSS's, Krea2 has none. */
export const familyHasNegpip = (family) => family === "anima" || family === "moss";

/** The panel's NegPiP switch of `family`. */
export const negpipKey = (family) => (family === "moss" ? "moss_negpip" : "negpip");

/** True when this state's run reads NegPiP weights. */
export function negpipOn(state) {
  const family = modelFamily(state.unet);
  return familyHasNegpip(family) && Boolean(state[negpipKey(family)]);
}

const MAYBE_WILDCARD = /__|~\/|\{[^{}]*\|/;

/** True when the run may hold a wildcard, so its prompts should go to `/wildcards/expand`, which has the final word. */
export function mayHaveWildcards(state) {
  const s = withDefaults(state);
  return [assembledPrompt(s), s.negative, s.hires.prompt, s.hires.negative, s.sam3.inpaint_prompt, s.sam3.negative_prompt,
    s.ext.sam3_refine.inpaint_prompt, s.ext.sam3_refine.negative_prompt, s.ext.sam3_tile.inpaint_prompt, s.ext.sam3_tile.negative_prompt]
    .some((text) => MAYBE_WILDCARD.test(text || ""));
}

export function missingLoras(state, available) {
  if (!Array.isArray(available)) return [];
  return loraTagNames(state).filter((name) => resolveLoraName(name, available) === null);
}

/** Only a negative or unreadable seed is drawn at random: any other value stays what the panel says. */
function concreteSeed(seed, rng) {
  const value = Math.trunc(Number(seed));
  return Number.isFinite(value) && value >= 0 ? value : Math.floor(rng() * SEED_RANGE);
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

/** studio_state.effective_attention. */
export const effectiveAttention = (s) => (s.forge_like.master && s.forge_like.attention ? FORGE_ATTENTION : s.attention);

/** An inpaint run: the detailer draws it, from the original pixels as Forge does, and the main sampler does not run. */
export const inpaintByDetailer = (s) => s.tab === "i2i" && s.i2i_mode === "inpaint";

const inpaintDetailerInputs = (s, generation, seed) => ({
  enabled: true,
  mask_mode: "Combined",
  seed,
  steps: s.steps,
  cfg: s.cfg,
  sampler_name: s.sampler,
  scheduler: s.scheduler,
  denoise: s.i2i.denoise,
  fill_mode: s.i2i.inpaint_fill,
  only_masked: s.i2i.inpaint_area === "masked",
  mask_padding: s.i2i.inpaint_padding,
  mask_blur: s.i2i.mask_blur,
  target_width: generation.width,
  target_height: generation.height,
  batch_size: s.batch,
  tell_empty_mask: true,
});

function hiresInputs(s, enabled, options) {
  return {
    enabled,
    scale_by: s.hires.scale,
    upscale_method: hiresUpscaler(s.hires.upscaler, options.upscalers),
    steps: s.hires.steps,
    denoise: s.hires.denoise,
    seed: s.seed,
    shift: s.hires.shift > 0 ? s.hires.shift : s.shift,
    sampler_name: s.hires.sampler || s.sampler,
    scheduler: s.hires.scheduler || s.scheduler,
    cfg: s.hires.cfg > 0 ? s.hires.cfg : s.cfg,
    base_steps: s.steps,
    vae_name: (enabled && s.hires.vae) || HIRES_SAME_VAE,
    forge_noise: s.forge_like.master && s.forge_like.noise,
    ...postInputs("hires", s.hires),
    ...extCnsInputs(s),
  };
}

const upscaleInputs = (s, enabled) => ({
  enabled,
  model_name: enabled ? choice(s.upscale.model) : IDLE_UPSCALER,
  scale_by: s.upscale.scale,
  model_name_2: (enabled && s.upscale.visibility > 0 && s.upscale.model_2) || NO_SECOND_UPSCALER,
  ...postInputs("upscale", s.upscale),
});

/** Forge's Mask only: the row after the detailer reads the mask node's image output, so nothing loads a model for it. */
const sam3MaskRuns = (s) => s.sam3.on && sam3MaskOnly(s);

const sam3MaskInputs = (s, enabled) => ({
  enabled,
  checkpoint: choice(s.sam3.checkpoint),
  prompt: s.sam3.prompt,
  seed: s.seed, // the saved files' name and metadata
  precision: s.sam3.device === CPU ? "fp32" : s.sam3.precision, // Easy SAM3 runs fp32 only there
  cache_model: !s.sam3.unload_after || s.sam3.keep_in_ram, // the app compiler's rule
  plain_overlay: !s.sam3.overlay,
  ...postInputs("sam3", s.sam3, "mask"),
  ...(extNeedsSam3Mask(s) ? { mask_mode: INDIVIDUAL_MASKS } : {}), // Refine and tile choose for themselves from the groups
});

function sam3ControlInputs(s) {
  const { cn_start: start, cn_end: end } = s.sam3;
  return {
    controlnet_enable: s.sam3.cn_enable && start !== end, // an empty range does nothing in Forge, the node refuses one
    controlnet_start: Math.min(start, end), // Forge swaps a reversed pair, the node refuses one
    controlnet_end: Math.max(start, end),
    controlnet_settings_json: JSON.stringify({
      pixel_perfect: s.sam3.cn_pixel_perfect,
      override_external: false,
      control_mode: s.sam3.cn_control_mode,
      resize_mode: CN_RESIZE_MODE,
      threshold_a: s.sam3.cn_threshold_a,
      threshold_b: s.sam3.cn_threshold_b,
    }),
  };
}

/** `size`: Forge's `p.width/p.height`, what an "only masked" crop is drawn at. */
function sam3DetailerInputs(s, enabled, size) {
  return {
    ...postInputs("sam3", s.sam3, "detailer"),
    enabled: enabled && !sam3MaskOnly(s),
    denoise: s.sam3.denoise,
    ...size,
    ...sam3ControlInputs(s),
    ...sam3PassInputs(s),
  };
}

const saveInputs = (s) => ({ studio_state: JSON.stringify(s), seed: s.seed });

const animaRefInputs = ({ method, megapixels, ...sections }) => ({ method, megapixels, ...sections[method] });

/** The tab's size, as the latent gets it (MOSS's 16x grid). */
export function generationSize(state) {
  const size = state.tab === "i2i" ? state.i2i : state;
  const family = modelFamily(state.unet);
  return { width: familySide(size.width, family), height: familySide(size.height, family) };
}

/** role -> {input: value} for the generation graph; `s` is complete and its seed concrete. */
function mainInputs(s, options) {
  const is38 = unetFamily(s.unet, options.families) === "3.8b";
  const family = modelFamily(s.unet);
  const isI2i = s.tab === "i2i";
  const select = { select: is38 ? 1 : 0 };
  const size = isI2i ? s.i2i : s;
  const generation = generationSize(s);
  const encode = { steps: s.steps, forge_syntax: s.forge_like.master && s.forge_like.prompt_syntax };
  return {
    unet: is38 ? {} : { unet_name: choice(s.unet) },
    unet38: is38 ? { model_name: choice(s.unet) } : {},
    te: { clip_name: choice(s.te), type: FAMILY_PROFILES[family].clip_type },
    vae: { vae_name: choice(s.vae) },
    family_switch_model: select,
    family_switch_pos: select,
    family_switch_neg: select,
    shift: { shift: s.shift },
    lora_stack: { positive: effectivePrompt(s), negative: s.negative },
    negpip: { enabled: negpipOn(s) },
    dora: { enabled: true, mode: effectiveDoraMode(s) },
    attention: { mode: effectiveAttention(s) },
    pos_encode: encode,
    neg_encode: encode,
    reference_image: { image: choice(s.reference.image) },
    reference_image_2: { image: choice(s.reference.image_2) },
    anima_ref: animaRefInputs(s.reference),
    krea2_ref: { ...s.reference.krea2, width: size.width, height: size.height, ...encode },
    latent: {
      mode: !isI2i ? "txt2img" : s.i2i_mode === "inpaint" ? "inpaint" : "img2img",
      ...generation,
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
    inpaint_detailer: inpaintDetailerInputs(s, generation, s.seed),
    inpaint_switch: { select: inpaintByDetailer(s) ? 1 : 0 },
    hires: hiresInputs(s, s.hires.on && !isI2i, options), // Forge's img2img has no hires pass
    upscale: upscaleInputs(s, s.upscale.on),
    sam3_mask: sam3MaskInputs(s, s.sam3.on || extNeedsSam3Mask(s)),
    sam3_detailer: sam3DetailerInputs(s, s.sam3.on, { target_width: generation.width, target_height: generation.height }),
    save: saveInputs(s),
    ...Object.fromEntries(EXT_ROLES.map((role) => [role, extInputs(s, role, { postAction: options.postAction })])),
  };
}

/** Every input fed by slot 0 of node `from` takes slot `slot` of node `to` instead; nothing happens if either is absent. */
function relink(prompt, from, to, slot = 0) {
  if (from === undefined || to === undefined || !prompt[String(from)] || !prompt[String(to)]) return;
  for (const node of Object.values(prompt)) {
    for (const [name, value] of Object.entries(node.inputs || {})) {
      if (Array.isArray(value) && String(value[0]) === String(from) && value[1] === 0) node.inputs[name] = [String(to), slot];
    }
  }
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
  if (sam3MaskRuns(s)) relink(out, roleIds.sam3_detailer, roleIds.sam3_mask, MASK_OVERLAY_SLOT);
  return out;
}

/**
 * role -> 0 (run) | 4 (bypass): the unused family, the i2i input in t2i, SAM3 when off, post nodes unless a post run
 * (R7), and every extension that does not run (studio_ext.extModes).
 */
export function stageModes(state, postAction = null, options = {}) {
  const s = withDefaults(state);
  const family = modelFamily(s.unet);
  const bypass = new Set(
    unetFamily(s.unet, options.families) === "3.8b"
      ? ["unet", "pos_encode", "neg_encode"]
      : ["unet38", "qwen35", "pos_encode38", "neg_encode38"],
  );
  if (family !== "moss" || !s.moss_stream) bypass.add("moss_stream");
  const reference = referenceRuns(s, family);
  if (!reference) bypass.add("reference_image");
  if (!reference || family !== "anima" || s.reference.method !== "snapref" || !s.reference.image_2) bypass.add("reference_image_2");
  if (!reference || family !== "anima") bypass.add("anima_ref");
  if (!reference || family !== "krea2") bypass.add("krea2_ref");
  if (s.tab === "t2i") bypass.add("i2i_image");
  if (postAction || !inpaintByDetailer(s)) bypass.add("inpaint_detailer");
  // R22: Refine replaces it, as in the app's post run. Mask only keeps it in the prompt: relink needs its links, and then nothing reads it.
  if (!s.sam3.on || (s.ext.sam3_refine.on && !sam3MaskOnly(s))) bypass.add("sam3_detailer");
  if (!s.sam3.on && !extNeedsSam3Mask(s)) bypass.add("sam3_mask");
  if (!postAction) ROLES.filter((role) => role.startsWith("post_")).forEach((role) => bypass.add(role));
  const ext = extModes(s, null, { postAction });
  return Object.fromEntries(ROLES.map((role) => [role, bypass.has(role) ? MODE_BYPASS : ext[role] ?? MODE_ALWAYS]));
}

/**
 * The state a gallery post-processing run uses: the record's settings (the image's own prompt, seed, models …;
 * keys it lacks come from the current state) with the post-processing sections from the current state, only the
 * `action`'s stage on (spec §4). A record without `loras` or the prompt boxes (read from `parameters`) already has them
 * in the prompt.
 */
export function postRunState(record, currentState, action) {
  const current = withDefaults(currentState);
  const generation = { ...(isObject(record) ? record : {}) };
  for (const section of Object.values(POST_SECTIONS)) delete generation[section];
  const merged = withDefaults({ ...current, loras: [], ...EMPTY_PROMPT_BOXES, ...generation });
  for (const section of Object.values(POST_SECTIONS)) {
    merged[section] = { ...current[section], on: POST_SECTIONS[action] === section };
  }
  if (modelFamily(merged.unet) !== modelFamily(current.unet)) merged.hires.vae = ""; // a VAE of the panel's family, as pickCheckpoint
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
  for (const [fromRole, toRole] of POST_ROUTES[action]) relink(prompt, roleIds[fromRole], roleIds[toRole]);
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
    post_sam3_detailer: sam3DetailerInputs(s, action === "sam3", { target_image_size: true }),
    post_save: saveInputs(s),
  });
  routeAround(out, roleIds, action);
  if (action === "sam3" && sam3MaskRuns(s)) relink(out, roleIds.post_sam3_detailer, roleIds.post_sam3_mask, MASK_OVERLAY_SLOT);
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
  if (params.Model) record.model = params.Model; // not a state key: it only tells which family made the image
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
