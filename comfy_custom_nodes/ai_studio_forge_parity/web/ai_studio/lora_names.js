// Pure helpers for the LoRA picker: ComfyUI LoRA list entries (relative paths under the LoRA roots) -> short labels,
// run/epoch grouping keys, search matching and trigger-word text. No imports, no DOM. The golden table
// js_tests/fixtures/lora_names_golden.json is shared with the Python side (parse rules must stay identical).

const RECIPE_ALGOS = new Set(["lokr", "loha", "locon", "lora", "dora", "tsm", "full"]);
const RECIPE_OPTIMIZERS = new Set(["came", "adamw", "adamw8bit", "prodigy", "adafactor"]);
const SIZE_TOKEN = /^\d+(\.\d+)?b$/;
const BATCH_TOKEN = /^b\d+$/;
const LR_TOKEN = /^\d+e\d+$/;
const EPOCH_SUFFIX = /-(\d{6})$/;
const EPOCH_TAIL = /( · (?:e\d+|최종(?: e\d+)?))$/;

export function isRecipeToken(token) {
  const t = token.toLowerCase();
  return SIZE_TOKEN.test(t) || RECIPE_ALGOS.has(t) || RECIPE_OPTIMIZERS.has(t) || BATCH_TOKEN.test(t) || LR_TOKEN.test(t);
}

/** Folders and file stem of an entry; `\` and `/` both separate. */
function splitEntry(entry) {
  const parts = String(entry || "").split(/[\\/]+/).filter((p) => p !== "");
  const file = parts.pop() || "";
  return { dirs: parts, stem: file.replace(/\.[^.]+$/, "") };
}

const MODEL_FOLDERS = new Set(["anima", "moss", "sdxl", "krea2", "_unsorted"]); // D:\LoRA's top folders since 11-16 (= lora_index._MODEL_FOLDERS)
const isAlgoFolder = (name) => RECIPE_ALGOS.has(String(name).toLowerCase());
// A known model folder name, or a first folder that is not an algorithm folder followed by one that is (so a new algorithm
// folder, or one with a stray space/BOM from a hand move, under ANIMA/MOSS/... still keeps its runs apart).
const isModelFolder = (first, second) =>
  MODEL_FOLDERS.has(String(first).toLowerCase()) || (!isAlgoFolder(first) && isAlgoFolder(second));

/**
 * Where an entry sits: [model/]algo/group/[run folder/]file. A file straight under the group is an old-layout run named by
 * its stem without the epoch suffix. `runKey` is shared by all epoch files and the final file of one run. The optional model
 * folder (D:\LoRA\ANIMA\LoKr\... since 11-16) is a known model folder name, or a first folder that is not an algorithm
 * folder followed by one that is (`isModelFolder`); it goes into `runKey` so runs of two models never share a card.
 */
export function parseLoraPath(entry) {
  const { dirs: all, stem } = splitEntry(entry);
  const model = all.length >= 2 && isModelFolder(all[0], all[1]) ? all[0] : "";
  const dirs = model ? all.slice(1) : all;
  const algo = dirs[0] || "";
  const downloaded = dirs[1] === "_downloaded";
  const group = (downloaded ? dirs[2] : dirs[1]) || "";
  const below = dirs.slice(downloaded ? 3 : 2);
  const suffix = EPOCH_SUFFIX.exec(stem);
  const epoch = suffix ? parseInt(suffix[1], 10) : null;
  const run = below.length ? below[0] : stem.replace(EPOCH_SUFFIX, "");
  const final = below.length > 0 && stem === run;

  const tokens = run.split("_").map((t) => t.trim()).filter(Boolean);
  const recipe = tokens.filter(isRecipeToken);
  const firstRecipe = tokens.findIndex(isRecipeToken);
  // The name before the recipe keeps its underscores ("@gt_dune"); without any recipe token the words are plain text.
  const rest = tokens.filter((t, i) => i > firstRecipe && !isRecipeToken(t));
  const head = firstRecipe < 0 ? [] : tokens.slice(0, firstRecipe).join("_");
  const runShort = (firstRecipe < 0 ? tokens : [head, ...rest]).filter(Boolean).join(" ");

  const groupLabel = group.replace(/\s*\(@[^)]*\)\s*$/, "").trim() || group;
  return {
    model, algo, group, groupLabel, run, runShort, recipe, epoch, final, downloaded,
    runKey: (model ? [model, algo, group, run] : [algo, group, run]).join("/"),
  };
}

const plain = (text) => String(text).toLowerCase().replace(/_/g, " ");

/** "묵밥 · @bjunbae fp16 · e14" / "... · 최종" (+ " eNN" when the run's epoch count is known). */
export function shortLoraName(entry, meta = {}) {
  const p = parseLoraPath(entry);
  const body = p.runShort || p.run;
  const showGroup = p.groupLabel && !plain(body).startsWith(plain(p.groupLabel));
  const tail = p.final ? (meta.num_epochs ? `최종 e${meta.num_epochs}` : "최종") : p.epoch != null ? `e${p.epoch}` : "";
  return [showGroup ? p.groupLabel : "", body, tail].filter(Boolean).join(" · ");
}

/** Shortens the middle of `text` to `max` characters; the " · eNN" / " · 최종" tail always stays. */
export function middleEllipsis(text, max) {
  const chars = Array.from(String(text));
  if (chars.length <= max) return String(text);
  const tailMatch = EPOCH_TAIL.exec(text);
  const tail = tailMatch ? Array.from(tailMatch[1]) : [];
  const head = chars.slice(0, chars.length - tail.length);
  const room = Math.max(max - tail.length - 1, 2);
  const front = Math.ceil(room / 2);
  const back = Math.floor(room / 2);
  return head.slice(0, front).join("") + "…" + head.slice(head.length - back).join("") + tail.join("");
}

/** The folders and file name of an entry for the search, without the model folder (else "anima" hit every run under ANIMA). */
function searchPath(entry) {
  const parts = String(entry || "").split(/[\\/]+/).filter((p) => p !== ""); // = splitEntry: the model folder is parts[0]
  return (parseLoraPath(entry).model ? parts.slice(1) : parts).join(" ");
}

/**
 * Every query word must hit the entry path (its model folder left out) or trigger (case/underscore-blind); a number word
 * ("14", "e14") must equal the epoch: `item.epoch` when given (the index's epoch — a final file carries its header epoch),
 * else the path's "-0000NN" suffix.
 */
export function matchesQuery(item, query) {
  const words = plain(query).split(/[\s\\/]+/).filter(Boolean);
  if (!words.length) return true;
  const hay = plain(`${searchPath(item.entry)} ${item.trigger || ""}`).replace(/[\\/]+/g, " ");
  const epoch = item.epoch ?? parseLoraPath(item.entry).epoch;
  return words.every((w) => {
    const num = /^e?(\d+)$/.exec(w);
    return num ? epoch != null && parseInt(num[1], 10) === epoch : hay.includes(w);
  });
}

/** "2.9b" / "3.8b" from a recipe's size token, else null. */
export function familyOfRecipe(recipe) {
  const found = (recipe || []).find((t) => /^(2\.9b|3\.8b)$/i.test(t));
  return found ? found.toLowerCase() : null;
}

// D:\LoRA's model folders -> the family of a LoRA in them (= lora_index._MODEL_FAMILIES); "anima" = an Anima LoRA of unknown
// size (an ANIMA run whose name has no 2.9b / 3.8b). The server (lora_index.lora_family) takes MOSS / KREA2 / SDXL before the
// header's base model and the recipe size (the user's own sorting), ANIMA only after both.
const MODEL_FAMILIES = new Map([["anima", "anima"], ["moss", "moss"], ["krea2", "krea2"], ["sdxl", "sdxl"]]);
const ANIMA_CHECKPOINTS = new Set(["2.9b", "3.8b", "base"]); // studio_logic.unetFamily's Anima families

/** The family a model folder name stands for (case-blind, trimmed), or null: _unsorted and other folders say nothing. */
export function familyOfModelFolder(model) {
  return MODEL_FAMILIES.get(String(model ?? "").trim().toLowerCase()) ?? null;
}

/** The LoRA families a checkpoint family shows besides the unknown ones: its own, and "anima" too on an Anima checkpoint. */
export function familiesShown(current) {
  return ANIMA_CHECKPOINTS.has(current) ? [current, "anima"] : [current];
}

/**
 * A LoRA of an unknown family is always shown; a known one only for the current checkpoint family (`familiesShown`) unless
 * "show all": an Anima checkpoint hides MOSS / Krea2 / SDXL LoRAs, a MOSS one everything but MOSS.
 */
export function familyVisible(itemFamily, current, showAll) {
  return Boolean(showAll) || itemFamily == null || familiesShown(current).includes(itemFamily);
}

/** Adds `trigger` to a comma-separated prompt text unless it is already one of the items. */
export function appendTrigger(text, trigger) {
  const word = String(trigger || "").trim();
  const base = String(text || "");
  if (!word) return base;
  const items = base.split(",").map((s) => s.trim().toLowerCase());
  if (items.includes(word.toLowerCase())) return base;
  const trimmed = base.replace(/[\s,]+$/, "");
  return trimmed ? `${trimmed}, ${word}` : word;
}

const LORA_TAG = /<lora:([^<>]+?)(?::-?\d*\.?\d+(?::-?\d*\.?\d+)?)?>/gi;
const lastPart = (name) => String(name).split(/[\\/]+/).pop().trim();

/**
 * The runs an image's `<lora:NAME:STRENGTH>` tags point at: NAME (a stem, maybe with a folder prefix) is matched against the
 * list `entries` by stem; epochs of one run give one `{runKey, label}` (label = short run name without the epoch tail),
 * in tag order. Names that are not in the list are left out.
 */
export function previewTargets(infotext, entries) {
  const byStem = new Map();
  for (const entry of entries || []) {
    const stem = lastPart(entry).replace(/\.[^.]+$/, "");
    if (!byStem.has(stem)) byStem.set(stem, []);
    byStem.get(stem).push(entry);
  }
  const targets = new Map();
  for (const match of String(infotext || "").matchAll(LORA_TAG)) {
    for (const entry of byStem.get(lastPart(match[1])) || []) {
      const { runKey } = parseLoraPath(entry);
      if (!targets.has(runKey)) targets.set(runKey, { runKey, label: shortLoraName(entry).replace(EPOCH_TAIL, "") });
    }
  }
  return [...targets.values()];
}
