// node:test for the studio's pure JS (web/ai_studio/studio_logic.js, studio_roles.js).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  DEFAULT_STATE,
  PICK_NOTICES,
  applyStateToPrompt,
  defaultModel,
  effectivePrompt,
  imageSize,
  missingLoras,
  hasPromptWeights,
  missingPicks,
  modelFamily,
  pickCheckpoint,
  postPrompt,
  postRunState,
  recordFromParameters,
  refAdapterDrops,
  referenceFile,
  sizeStep,
  stageModes,
  unetFamily,
  withDefaults,
  xyzCellPrompts,
} from "../web/ai_studio/studio_logic.js";
import { autoGrow } from "../web/ai_studio/studio_dom.js";
import { ROLES, ROLE_PROPERTY, roleIdsFromNodes } from "../web/ai_studio/studio_roles.js";

const golden = JSON.parse(readFileSync(new URL("./fixtures/studio_logic_golden.json", import.meta.url), "utf-8"));

const fixedRng = () => 0.5; // -> seed 2147483647

// A small API prompt shaped like the studio workflow's (graphToPrompt output): role -> node.
function studioPrompt() {
  const nodes = {
    unet: ["UNETLoader", { unet_name: "workflow.safetensors", weight_dtype: "default" }],
    te: ["CLIPLoader", { clip_name: "qwen_3_06b_base.safetensors", type: "stable_diffusion", device: "default" }],
    vae: ["VAELoader", { vae_name: "workflow_vae.safetensors" }],
    unet38: ["ForgeNeoAnima38V2Loader", { model_name: "workflow38.safetensors" }],
    qwen35: ["ForgeNeoAnimaQwen35Loader", { qwen35_model: "qwen35_4b.safetensors" }],
    family_switch_model: ["AIStudioSwitch", { select: 0, a: ["unet", 0], b: ["unet38", 0] }],
    moss_stream: ["AIStudioMossStream", { model: ["family_switch_model", 0] }],
    shift: ["ForgeNeoModelSamplingShift", { model: ["moss_stream", 0], shift: 1.0 }],
    sched: ["ForgeNeoExtraSchedulers", { model: ["shift", 0], flow_cosmos_rho: 7.0 }],
    lora_stack: ["AIStudioLoraStack", { model: ["sched", 0], clip: ["te", 0], positive: "", negative: "" }],
    negpip: ["ForgeNeoNegPip", { model: ["lora_stack", 0], clip: ["lora_stack", 1], enabled: false }],
    dora: ["ForgeNeoDoraInferenceMode", { model: ["negpip", 0], enabled: false, mode: "DoRA off (no magnitude)" }],
    pos_encode: ["AIStudioPromptEncode", { clip: ["lora_stack", 1], text: ["lora_stack", 2], steps: 20, forge_syntax: false }],
    neg_encode: ["AIStudioPromptEncode", { clip: ["lora_stack", 1], text: ["lora_stack", 3], steps: 20, forge_syntax: false }],
    pos_encode38: ["ForgeNeoAnima38V2Prompt", { model: ["dora", 0], native_clip: ["lora_stack", 1], qwen35_clip: ["qwen35", 0], prompt: ["lora_stack", 2] }],
    neg_encode38: ["ForgeNeoAnima38V2Prompt", { model: ["dora", 0], native_clip: ["lora_stack", 1], qwen35_clip: ["qwen35", 0], prompt: ["lora_stack", 3] }],
    family_switch_pos: ["AIStudioSwitch", { select: 0, a: ["pos_encode", 0], b: ["pos_encode38", 0] }],
    family_switch_neg: ["AIStudioSwitch", { select: 0, a: ["neg_encode", 0], b: ["neg_encode38", 0] }],
    reference_image: ["LoadImage", { image: "example.png", upload: "image" }],
    reference_image_2: ["LoadImage", { image: "example.png", upload: "image" }],
    anima_ref: ["AIStudioAnimaReference", { model: ["dora", 0], clip: ["lora_stack", 1], vae: ["vae", 0], image: ["reference_image", 0], method: "snapref", file: "" }],
    krea2_ref: ["AIStudioKrea2Reference", { model: ["anima_ref", 0], positive: ["family_switch_pos", 0], negative: ["family_switch_neg", 0], file: "", width: 1024, height: 1024 }],
    i2i_image: ["LoadImage", { image: "example.png", upload: "image" }],
    latent: ["ForgeNeoLatentInput", { vae: ["vae", 0], mode: "txt2img", width: 1536, height: 1536, batch_size: 1, fit: "crop", mask_blur: 0, img2img_image: ["i2i_image", 0] }],
    sampler: ["ForgeNeoKSamplerCNS", { model: ["dora", 0], positive: ["family_switch_pos", 0], negative: ["family_switch_neg", 0], latent_image: ["latent", 0], seed: 0, steps: 30, cfg: 5.0, sampler_name: "euler", scheduler: "simple", denoise: 1.0, forge_noise: false }],
    hires: ["ForgeNeoHiresFix", { model: ["dora", 0], positive: ["family_switch_pos", 0], negative: ["family_switch_neg", 0], samples: ["sampler", 0], seed: -1, enabled: false, scale_by: 2.0, upscale_method: "latent:bislerp", steps: 0, cfg: 5.0, sampler_name: "euler", scheduler: "simple", denoise: 0.35, base_steps: 30, shift: 1.0, forge_noise: false }],
    decode: ["VAEDecode", { samples: ["hires", 0], vae: ["hires", 1] }],
    upscale: ["AIStudioUpscaleImage", { image: ["decode", 0], enabled: false, model_name: "workflow_up.pth", scale_by: 2.0 }],
    sam3_mask: ["ForgeNeoSAM3Mask", { image: ["upscale", 0], prompt: "eyes", checkpoint: "sam3.pt", enabled: true }],
    sam3_detailer: ["ForgeNeoSAM3Detailer", { image: ["upscale", 0], mask: ["sam3_mask", 0], denoise: 0.5, steps: 28, cfg: 7.0, sampler_name: "euler", scheduler: "simple", seed: 0, enabled: true }],
    save: ["AIStudioSaveForge", { images: ["sam3_detailer", 0], studio_state: "{}", seed: 0, filename_prefix: "AIStudio/anima" }],
    post_image: ["LoadImage", { image: "example.png", upload: "image" }],
    post_upscale: ["AIStudioUpscaleImage", { image: ["post_image", 0], enabled: true, model_name: "workflow_up.pth", scale_by: 2.0 }],
    post_encode: ["VAEEncode", { pixels: ["post_upscale", 0], vae: ["vae", 0] }],
    post_hires: ["ForgeNeoHiresFix", { model: ["dora", 0], positive: ["family_switch_pos", 0], negative: ["family_switch_neg", 0], samples: ["post_encode", 0], seed: -1, enabled: true, scale_by: 2.0, upscale_method: "latent:bislerp", steps: 0, cfg: 5.0, sampler_name: "euler", scheduler: "simple", denoise: 0.35, base_steps: 30, shift: 1.0, forge_noise: false }],
    post_decode: ["VAEDecode", { samples: ["post_hires", 0], vae: ["post_hires", 1] }],
    post_sam3_mask: ["ForgeNeoSAM3Mask", { image: ["post_decode", 0], prompt: "eyes", checkpoint: "sam3.pt", enabled: true }],
    post_sam3_detailer: ["ForgeNeoSAM3Detailer", { image: ["post_decode", 0], mask: ["post_sam3_mask", 0], denoise: 0.5, steps: 28, cfg: 7.0, sampler_name: "euler", scheduler: "simple", seed: 0, enabled: true }],
    post_save: ["AIStudioSaveForge", { images: ["post_sam3_detailer", 0], studio_state: "{}", seed: 0, filename_prefix: "AIStudio/anima" }],
    panel: ["AIStudioPanel", { state: "{}" }],
  };
  const roles = Object.keys(nodes);
  const roleIds = Object.fromEntries(roles.map((role, i) => [role, String(i + 1)]));
  const relink = (value) => (Array.isArray(value) ? [roleIds[value[0]], value[1]] : value);
  const prompt = {};
  for (const [role, [classType, inputs]] of Object.entries(nodes)) {
    const linked = Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, relink(v)]));
    prompt[roleIds[role]] = { class_type: classType, inputs: linked, _meta: { title: role } };
  }
  delete roleIds.panel;
  return { prompt, roleIds };
}

const inputs = (prompt, roleIds, role) => prompt[roleIds[role]].inputs;
const state = (overrides = {}) => ({ ...structuredClone(DEFAULT_STATE), ...overrides });

// ---------------------------------------------------------------------------------------------- roles

test("ROLES lists every engine role once, without family_switch_clip (R6)", () => {
  assert.equal(new Set(ROLES).size, ROLES.length);
  assert.equal(ROLES.length, 54);
  assert.ok(!ROLES.includes("family_switch_clip"));
  const { roleIds } = studioPrompt(); // a workflow without extension nodes: studio_ext.test.mjs covers those roles
  assert.deepEqual(ROLES.filter((role) => !role.startsWith("ext_")).sort(), Object.keys(roleIds).sort());
});

test("negpip follows the panel switch", () => {
  const prompt = { 9: { class_type: "ForgeNeoNegPip", inputs: { enabled: true } } };
  assert.equal(applyStateToPrompt(prompt, { negpip: "9" }, { ...DEFAULT_STATE, negpip: false })[9].inputs.enabled, false);
  assert.equal(applyStateToPrompt(prompt, { negpip: "9" }, DEFAULT_STATE)[9].inputs.enabled, true);
});

test("autoGrow fits the text and keeps the panel scroll position", () => {
  const box = { scrollTop: 40 };
  const el = { style: {}, scrollHeight: 120, offsetHeight: 122, clientHeight: 120, closest: () => box };
  autoGrow(el);
  assert.equal(el.style.height, "122px");
  assert.equal(box.scrollTop, 40);
});

test("roleIdsFromNodes reads properties.aistudio_role", () => {
  const nodes = [
    { id: 7, properties: { [ROLE_PROPERTY]: "sampler" } },
    { id: 9, properties: {} },
    { id: 12, properties: { [ROLE_PROPERTY]: "not_a_role" } },
    { id: 3 },
  ];
  assert.equal(ROLE_PROPERTY, "aistudio_role");
  assert.deepEqual(roleIdsFromNodes(nodes), { sampler: "7" });
});

// ---------------------------------------------------------------------------------------------- golden (shared with Python)

test("DEFAULT_STATE matches Python's DEFAULT_STATE", () => {
  assert.deepEqual(DEFAULT_STATE, golden.default_state);
});

test("effectivePrompt matches Python's effective_prompt", () => {
  for (const { state: raw, expected } of golden.effective_prompt) {
    assert.equal(effectivePrompt(withDefaults(raw)), expected, JSON.stringify(raw));
  }
});

test("unetFamily name rule matches Python's unet_family", () => {
  for (const { name, expected } of golden.unet_family) {
    assert.equal(unetFamily(name), expected, name);
  }
});

test("unetFamily prefers the families map from /choices", () => {
  assert.equal(unetFamily("bundle.safetensors", { "bundle.safetensors": "3.8b" }), "3.8b");
  assert.equal(unetFamily("anima-3.8b.safetensors", { "anima-3.8b.safetensors": "2.9b" }), "2.9b");
  assert.equal(unetFamily("x13.8b.safetensors"), "2.9b"); // R5 rule with the digit lookbehind
});

// ---------------------------------------------------------------------------------------------- applyStateToPrompt

test("applyStateToPrompt sets sampler inputs", () => {
  const { prompt, roleIds } = studioPrompt();
  const out = applyStateToPrompt(prompt, roleIds, state(), { rng: fixedRng });
  const sampler = inputs(out, roleIds, "sampler");
  assert.equal(sampler.steps, 32);
  assert.equal(sampler.cfg, 3);
  assert.equal(sampler.sampler_name, "exp_heun_2_x0_sde");
  assert.equal(sampler.scheduler, "flow_cosmos_dynamic");
  assert.equal(sampler.denoise, 1.0);
  assert.equal(sampler.forge_noise, true);
  assert.equal(sampler.seed, 2147483647);
  assert.deepEqual(sampler.model, [roleIds.dora, 0]);
});

test("applyStateToPrompt does not mutate input", () => {
  const { prompt, roleIds } = studioPrompt();
  const before = structuredClone(prompt);
  const s = state({ prompt: "1girl", loras: [{ on: true, name: "meow", strength: 1 }] });
  const sBefore = structuredClone(s);
  const out = applyStateToPrompt(prompt, roleIds, s, { rng: fixedRng });
  assert.deepEqual(prompt, before);
  assert.deepEqual(s, sBefore);
  assert.notEqual(out, prompt);
  assert.notEqual(out[roleIds.sampler].inputs, prompt[roleIds.sampler].inputs);
});

test("applyStateToPrompt maps models, shift, latent, prompts and save", () => {
  const { prompt, roleIds } = studioPrompt();
  const s = state({
    unet: "anima-preview2.safetensors", te: "qwen.safetensors", vae: "vae.safetensors", shift: 2.5, seed: 7,
    width: 1024, height: 1344, batch: 2, prompt: "1girl", negative: "lowres",
    loras: [{ on: true, name: "meow", strength: 0.9 }],
  });
  const out = applyStateToPrompt(prompt, roleIds, s);
  assert.equal(inputs(out, roleIds, "unet").unet_name, "anima-preview2.safetensors");
  assert.equal(inputs(out, roleIds, "te").clip_name, "qwen.safetensors");
  assert.equal(inputs(out, roleIds, "vae").vae_name, "vae.safetensors");
  assert.equal(inputs(out, roleIds, "unet38").model_name, "workflow38.safetensors");
  assert.equal(inputs(out, roleIds, "shift").shift, 2.5);
  assert.deepEqual(
    { ...inputs(out, roleIds, "latent"), vae: undefined, img2img_image: undefined },
    { vae: undefined, img2img_image: undefined, mode: "txt2img", width: 1024, height: 1344, batch_size: 2, fit: "crop", mask_blur: 4 },
  );
  const stack = inputs(out, roleIds, "lora_stack");
  assert.equal(stack.positive, "1girl <lora:meow:0.9>");
  assert.equal(stack.negative, "lowres");
  for (const role of ["pos_encode", "neg_encode"]) {
    const encode = inputs(out, roleIds, role);
    assert.equal(encode.steps, 32);
    assert.equal(encode.forge_syntax, true);
    assert.deepEqual(encode.text, prompt[roleIds[role]].inputs.text); // a link stays a link
  }
  for (const role of ["family_switch_model", "family_switch_pos", "family_switch_neg"]) {
    assert.equal(inputs(out, roleIds, role).select, 0);
  }
  const save = inputs(out, roleIds, "save");
  assert.equal(save.seed, 7);
  const saved = JSON.parse(save.studio_state);
  assert.equal(saved.seed, 7);
  assert.equal(saved.prompt, "1girl");
  assert.deepEqual(saved.loras, [{ on: true, name: "meow", strength: 0.9 }]);
});

test("applyStateToPrompt resolves seed -1 once and shares it", () => {
  const { prompt, roleIds } = studioPrompt();
  const out = applyStateToPrompt(prompt, roleIds, state({ hires: { ...DEFAULT_STATE.hires, on: true } }), { rng: () => 0.25 });
  const seed = inputs(out, roleIds, "sampler").seed;
  assert.equal(seed, Math.floor(0.25 * 4294967294));
  assert.equal(inputs(out, roleIds, "hires").seed, seed);
  assert.equal(inputs(out, roleIds, "sam3_detailer").seed, seed);
  assert.equal(inputs(out, roleIds, "save").seed, seed);
  assert.equal(JSON.parse(inputs(out, roleIds, "save").studio_state).seed, seed);
});

test("empty model choices keep the workflow's values", () => {
  const { prompt, roleIds } = studioPrompt();
  const out = applyStateToPrompt(prompt, roleIds, state(), { rng: fixedRng });
  assert.equal(inputs(out, roleIds, "unet").unet_name, "workflow.safetensors");
  assert.equal(inputs(out, roleIds, "vae").vae_name, "workflow_vae.safetensors");
  assert.equal(inputs(out, roleIds, "upscale").model_name, "workflow_up.pth");
  assert.equal(inputs(out, roleIds, "sam3_mask").checkpoint, "sam3.pt");
  assert.equal(inputs(out, roleIds, "i2i_image").image, "example.png");
});

test("roles missing from the prompt (bypassed) or from roleIds are skipped", () => {
  const { prompt, roleIds } = studioPrompt();
  delete prompt[roleIds.hires];
  const partialIds = { ...roleIds };
  delete partialIds.sampler;
  const out = applyStateToPrompt(prompt, partialIds, state(), { rng: fixedRng });
  assert.equal(out[roleIds.hires], undefined);
  assert.equal(inputs(out, roleIds, "sampler").steps, 30);
});

test("the i2i tab keeps its own size; t2i ignores it", () => {
  const { prompt, roleIds } = studioPrompt();
  const i2i = { ...DEFAULT_STATE.i2i, width: 1248, height: 1824 };
  const sent = inputs(applyStateToPrompt(prompt, roleIds, state({ tab: "i2i", i2i }), { rng: fixedRng }), roleIds, "latent");
  assert.deepEqual([sent.width, sent.height], [1248, 1824]);
  const main = inputs(applyStateToPrompt(prompt, roleIds, state({ tab: "t2i", i2i }), { rng: fixedRng }), roleIds, "latent");
  assert.deepEqual([main.width, main.height], [832, 1216]);
});

test("tab i2i sets latent mode img2img", () => {
  const { prompt, roleIds } = studioPrompt();
  const s = state({ tab: "i2i", i2i: { ...DEFAULT_STATE.i2i, image: "in.png", mask_blur: 8 }, hires: { ...DEFAULT_STATE.hires, on: true } });
  const out = applyStateToPrompt(prompt, roleIds, s, { rng: fixedRng });
  const latent = inputs(out, roleIds, "latent");
  assert.equal(latent.mode, "img2img");
  assert.equal(latent.mask_blur, 8);
  assert.equal(inputs(out, roleIds, "i2i_image").image, "in.png");
  assert.equal(inputs(out, roleIds, "sampler").denoise, 0.55);
  assert.equal(inputs(out, roleIds, "hires").enabled, false); // Forge img2img has no hires
});

test("inpaint sets mode inpaint", () => {
  const { prompt, roleIds } = studioPrompt();
  const out = applyStateToPrompt(prompt, roleIds, state({ tab: "i2i", i2i_mode: "inpaint" }), { rng: fixedRng });
  assert.equal(inputs(out, roleIds, "latent").mode, "inpaint");
});

test("3.8b unet selects switch 1", () => {
  const { prompt, roleIds } = studioPrompt();
  const out = applyStateToPrompt(prompt, roleIds, state({ unet: "anima_3.8b_v2.safetensors" }), { rng: fixedRng });
  for (const role of ["family_switch_model", "family_switch_pos", "family_switch_neg"]) {
    assert.equal(inputs(out, roleIds, role).select, 1, role);
  }
  assert.equal(inputs(out, roleIds, "unet38").model_name, "anima_3.8b_v2.safetensors");
  assert.equal(inputs(out, roleIds, "unet").unet_name, "workflow.safetensors");
  const viaMap = applyStateToPrompt(prompt, roleIds, state({ unet: "bundle.safetensors" }), {
    rng: fixedRng, families: { "bundle.safetensors": "3.8b" },
  });
  assert.equal(inputs(viaMap, roleIds, "family_switch_model").select, 1);
});

test("Forge-like switches: noise, prompt syntax and the effective DoRA mode (R2, R15)", () => {
  const { prompt, roleIds } = studioPrompt();
  const on = applyStateToPrompt(prompt, roleIds, state(), { rng: fixedRng });
  assert.deepEqual(inputs(on, roleIds, "dora").enabled, true);
  assert.equal(inputs(on, roleIds, "dora").mode, "LyCORIS fp32 (as trained)");
  assert.equal(inputs(on, roleIds, "hires").forge_noise, true);

  const masterOff = state({ dora_mode: "Forge/Comfy fp32", forge_like: { ...DEFAULT_STATE.forge_like, master: false } });
  const off = applyStateToPrompt(prompt, roleIds, masterOff, { rng: fixedRng });
  assert.equal(inputs(off, roleIds, "dora").enabled, true);
  assert.equal(inputs(off, roleIds, "dora").mode, "Forge/Comfy fp32");
  assert.equal(inputs(off, roleIds, "sampler").forge_noise, false);
  assert.equal(inputs(off, roleIds, "hires").forge_noise, false);
  assert.equal(inputs(off, roleIds, "pos_encode").forge_syntax, false);

  const partial = state({ forge_like: { ...DEFAULT_STATE.forge_like, noise: false, dora: false } });
  const p = applyStateToPrompt(prompt, roleIds, partial, { rng: fixedRng });
  assert.equal(inputs(p, roleIds, "sampler").forge_noise, false);
  assert.equal(inputs(p, roleIds, "neg_encode").forge_syntax, true);
  assert.equal(inputs(p, roleIds, "dora").mode, "LyCORIS fp32 (as trained)");
});

test("hires gets the main sampler settings unless hires.sampler is set (R8)", () => {
  const { prompt, roleIds } = studioPrompt();
  const s = state({ seed: 11, steps: 28, cfg: 4.5, shift: 2.0, hires: { ...DEFAULT_STATE.hires, on: true } });
  const out = applyStateToPrompt(prompt, roleIds, s);
  const hires = inputs(out, roleIds, "hires");
  assert.deepEqual(
    { enabled: hires.enabled, scale_by: hires.scale_by, upscale_method: hires.upscale_method, steps: hires.steps,
      denoise: hires.denoise, seed: hires.seed, shift: hires.shift, sampler_name: hires.sampler_name,
      scheduler: hires.scheduler, cfg: hires.cfg, base_steps: hires.base_steps },
    { enabled: true, scale_by: 1.5, upscale_method: "4x-UltraSharp.pth", steps: 16, denoise: 0.4, seed: 11, shift: 2.0,
      sampler_name: "exp_heun_2_x0_sde", scheduler: "flow_cosmos_dynamic", cfg: 4.5, base_steps: 28 },
  );
  const own = applyStateToPrompt(prompt, roleIds, state({ hires: { ...DEFAULT_STATE.hires, sampler: "euler" } }), { rng: fixedRng });
  assert.equal(inputs(own, roleIds, "hires").sampler_name, "euler");
  assert.equal(inputs(own, roleIds, "hires").enabled, false);
});

test("hires upscaler falls back to latent:bicubic when the file is not offered", () => {
  const { prompt, roleIds } = studioPrompt();
  const missing = applyStateToPrompt(prompt, roleIds, state(), { rng: fixedRng, upscalers: ["latent:bicubic", "2x.pth"] });
  assert.equal(inputs(missing, roleIds, "hires").upscale_method, "latent:bicubic");
  const present = applyStateToPrompt(prompt, roleIds, state(), { rng: fixedRng, upscalers: ["4x-UltraSharp.pth"] });
  assert.equal(inputs(present, roleIds, "hires").upscale_method, "4x-UltraSharp.pth");
});

test("upscale and SAM3 inputs; SAM3 steps/cfg 0 use the main values (R8)", () => {
  const { prompt, roleIds } = studioPrompt();
  const s = state({
    seed: 5, steps: 30, cfg: 3.5,
    upscale: { on: true, model: "2x.pth", scale: 1.5 },
    sam3: { on: true, checkpoint: "C:/m/sam3.1_multiplex_fp16.safetensors", prompt: "hand", denoise: 0.3, steps: 0, cfg: 0 },
  });
  const out = applyStateToPrompt(prompt, roleIds, s);
  assert.deepEqual(
    { ...inputs(out, roleIds, "upscale"), image: undefined },
    { image: undefined, enabled: true, model_name: "2x.pth", scale_by: 1.5 },
  );
  const mask = inputs(out, roleIds, "sam3_mask");
  assert.deepEqual([mask.enabled, mask.checkpoint, mask.prompt], [true, "C:/m/sam3.1_multiplex_fp16.safetensors", "hand"]);
  const detailer = inputs(out, roleIds, "sam3_detailer");
  assert.deepEqual(
    [detailer.enabled, detailer.denoise, detailer.steps, detailer.cfg, detailer.sampler_name, detailer.scheduler, detailer.seed],
    [true, 0.3, 30, 3.5, "exp_heun_2_x0_sde", "flow_cosmos_dynamic", 5],
  );
  const own = applyStateToPrompt(prompt, roleIds, state({ sam3: { ...s.sam3, steps: 12, cfg: 6 } }), { rng: fixedRng });
  assert.deepEqual([inputs(own, roleIds, "sam3_detailer").steps, inputs(own, roleIds, "sam3_detailer").cfg], [12, 6]);
});

test("applyStateToPrompt fills keys missing from an old saved state", () => {
  const { prompt, roleIds } = studioPrompt();
  const out = applyStateToPrompt(prompt, roleIds, { prompt: "x", hires: { on: true } }, { rng: fixedRng });
  assert.equal(inputs(out, roleIds, "sampler").steps, 32);
  assert.equal(inputs(out, roleIds, "hires").denoise, 0.4);
  assert.equal(inputs(out, roleIds, "hires").enabled, true);
});

// ---------------------------------------------------------------------------------------------- stageModes

test("stageModes bypasses the unused family, i2i input, SAM3 and post nodes (R7)", () => {
  const modes = stageModes(state({ unet: "anima-preview2.safetensors" }));
  assert.deepEqual(Object.keys(modes).sort(), [...ROLES].sort());
  const bypassed = Object.keys(modes).filter((role) => modes[role] === 4 && !role.startsWith("ext_")).sort();
  assert.deepEqual(bypassed, [
    "anima_ref", "i2i_image", "krea2_ref", "moss_stream", "neg_encode38", "pos_encode38", "post_decode", "post_encode",
    "post_hires", "post_image", "post_sam3_detailer", "post_sam3_mask", "post_save", "post_upscale", "qwen35", "reference_image", "reference_image_2",
    "sam3_detailer", "sam3_mask", "unet38",
  ]);
  assert.ok(Object.values(modes).every((mode) => mode === 0 || mode === 4));
});

test("stageModes for 3.8b, i2i with SAM3 on, and a post run", () => {
  const modes = stageModes(state({ unet: "anima_3.8b_v2.safetensors", tab: "i2i", sam3: { ...DEFAULT_STATE.sam3, on: true } }));
  for (const role of ["unet", "pos_encode", "neg_encode"]) assert.equal(modes[role], 4, role);
  for (const role of ["unet38", "qwen35", "pos_encode38", "neg_encode38", "te", "i2i_image", "sam3_mask", "sam3_detailer"]) {
    assert.equal(modes[role], 0, role);
  }
  const post = stageModes(state(), "hires");
  for (const role of ROLES.filter((r) => r.startsWith("post_"))) assert.equal(post[role], 0, role);
  const viaMap = stageModes(state({ unet: "b.safetensors" }), null, { families: { "b.safetensors": "3.8b" } });
  assert.equal(viaMap.unet, 4);
});

// ---------------------------------------------------------------------------------------------- postPrompt

const record = () => ({
  ...state({ prompt: "1girl <lora:meow:1>", negative: "lowres", seed: 42, steps: 30, cfg: 4, unet: "made-with.safetensors" }),
  image: "made.png",
});

test("postPrompt hires enables only post_hires", () => {
  const { prompt, roleIds } = studioPrompt();
  const current = state({ prompt: "other", seed: 9, hires: { ...DEFAULT_STATE.hires, scale: 2.0, denoise: 0.3 } });
  const { prompt: out, outputNodeIds } = postPrompt(prompt, roleIds, record(), "hires", current, { rng: fixedRng });
  assert.deepEqual(outputNodeIds, [roleIds.post_save]);
  assert.equal(inputs(out, roleIds, "post_image").image, "made.png");
  const hires = inputs(out, roleIds, "post_hires");
  assert.equal(hires.enabled, true);
  assert.equal(hires.scale_by, 2.0); // current post settings
  assert.equal(hires.denoise, 0.3);
  assert.equal(hires.seed, 42); // the record's seed
  assert.equal(hires.base_steps, 30);
  assert.equal(inputs(out, roleIds, "post_upscale").enabled, false);
  assert.equal(inputs(out, roleIds, "post_sam3_mask").enabled, false);
  assert.equal(inputs(out, roleIds, "post_sam3_detailer").enabled, false);
  assert.equal(inputs(out, roleIds, "lora_stack").positive, "1girl <lora:meow:1>"); // the record's prompt
  assert.equal(inputs(out, roleIds, "unet").unet_name, "made-with.safetensors");
  // Only the hires stage sits between the image and the save: upscale and SAM3 are routed around.
  assert.deepEqual(inputs(out, roleIds, "post_encode").pixels, [roleIds.post_image, 0]);
  assert.deepEqual(inputs(out, roleIds, "post_save").images, [roleIds.post_decode, 0]);
  const saved = JSON.parse(inputs(out, roleIds, "post_save").studio_state);
  assert.equal(saved.seed, 42);
  assert.equal(saved.hires.on, true);
  assert.equal(saved.hires.scale, 2.0);
  assert.equal(inputs(out, roleIds, "post_save").seed, 42);
});

test("postPrompt upscale and sam3 skip the VAE round trip", () => {
  const { prompt, roleIds } = studioPrompt();
  const current = state({ upscale: { on: false, model: "2x.pth", scale: 3 }, sam3: { ...DEFAULT_STATE.sam3, prompt: "hand" } });
  const up = postPrompt(prompt, roleIds, record(), "upscale", current).prompt;
  assert.deepEqual(
    { ...inputs(up, roleIds, "post_upscale"), image: undefined },
    { image: undefined, enabled: true, model_name: "2x.pth", scale_by: 3 },
  );
  assert.equal(inputs(up, roleIds, "post_hires").enabled, false);
  assert.deepEqual(inputs(up, roleIds, "post_save").images, [roleIds.post_upscale, 0]);

  const sam = postPrompt(prompt, roleIds, record(), "sam3", current).prompt;
  assert.equal(inputs(sam, roleIds, "post_sam3_mask").enabled, true);
  assert.equal(inputs(sam, roleIds, "post_sam3_mask").prompt, "hand");
  assert.equal(inputs(sam, roleIds, "post_sam3_detailer").enabled, true);
  assert.equal(inputs(sam, roleIds, "post_sam3_detailer").seed, 42);
  assert.equal(inputs(sam, roleIds, "post_hires").enabled, false);
  assert.equal(inputs(sam, roleIds, "post_upscale").enabled, false);
  assert.deepEqual(inputs(sam, roleIds, "post_sam3_mask").image, [roleIds.post_image, 0]);
  assert.deepEqual(inputs(sam, roleIds, "post_sam3_detailer").image, [roleIds.post_image, 0]);
  assert.deepEqual(inputs(sam, roleIds, "post_save").images, [roleIds.post_sam3_detailer, 0]);
});

test("postPrompt reads an output image ref as an annotated LoadImage value", () => {
  const { prompt, roleIds } = studioPrompt();
  const rec = { ...record(), image: { filename: "a_00001_.png", subfolder: "AIStudio", type: "output" } };
  const out = postPrompt(prompt, roleIds, rec, "upscale", state()).prompt;
  assert.equal(inputs(out, roleIds, "post_image").image, "AIStudio/a_00001_.png [output]");
  const input = postPrompt(prompt, roleIds, { ...rec, image: { filename: "b.png", subfolder: "", type: "input" } }, "upscale", state()).prompt;
  assert.equal(inputs(input, roleIds, "post_image").image, "b.png");
});

test("postPrompt refuses an unknown action or a record without an image", () => {
  const { prompt, roleIds } = studioPrompt();
  assert.throws(() => postPrompt(prompt, roleIds, record(), "same_seed", state()), /action/);
  assert.throws(() => postPrompt(prompt, roleIds, { ...record(), image: "" }, "hires", state()), /image/);
});

test("postRunState keeps the record's settings and takes post settings from the current state", () => {
  const fromParameters = { prompt: "1girl <lora:meow:1>", negative: "", seed: 42, steps: 32, cfg: 3, width: 832, height: 1216, image: "x.png" };
  const current = state({
    prompt: "now", unet: "current.safetensors", loras: [{ on: true, name: "other", strength: 1 }],
    upscale: { on: false, model: "2x.pth", scale: 2 },
  });
  const merged = postRunState(fromParameters, current, "upscale");
  assert.equal(merged.prompt, "1girl <lora:meow:1>");
  assert.equal(merged.seed, 42);
  assert.equal(merged.unet, "current.safetensors"); // not in the record
  assert.deepEqual(merged.loras, []); // the record's prompt already carries its LoRA tags
  assert.equal(merged.upscale.model, "2x.pth");
  assert.equal(merged.upscale.on, true);
  assert.equal(merged.hires.on, false);
  assert.equal(merged.sam3.on, false);
  assert.equal("image" in merged, false);
});

// ---------------------------------------------------------------------------------------------- recordFromParameters

test("recordFromParameters parses forge text", () => {
  const rec = recordFromParameters(golden.infotext.text);
  assert.deepEqual(rec, {
    prompt: "1girl, solo <lora:meow:1>",
    negative: "worst quality, lowres",
    seed: 42,
    steps: 32,
    cfg: 3,
    width: 832,
    height: 1216,
    model: "anima-preview2",
  });
});

test("recordFromParameters reads multi-line prompts, quoted values and no negative", () => {
  const text = 'line one\nline two\nSteps: 20, Sampler: "DPM++ 2M, Karras", CFG scale: 4.5, Seed: 123, Size: 512x768, Model: "a:b"';
  assert.deepEqual(recordFromParameters(text), {
    prompt: "line one\nline two", negative: "", seed: 123, steps: 20, cfg: 4.5, width: 512, height: 768, model: "a:b",
  });
  const multiNeg = "p\nNegative prompt: n1\nn2\nSteps: 10, Seed: 1";
  assert.deepEqual(recordFromParameters(multiNeg), { prompt: "p", negative: "n1\nn2", seed: 1, steps: 10 });
  assert.deepEqual(recordFromParameters("Steps: 10, Seed: 1\r\n"), { prompt: "", negative: "", seed: 1, steps: 10 });
});

test("recordFromParameters returns null on junk", () => {
  for (const junk of [null, undefined, 42, "", "   ", "hello world", "a\nb: c", "p\nSteps: x, Seed: 1", "p\nSteps: 20, Sampler: Euler"]) {
    assert.equal(recordFromParameters(junk), null, String(junk));
  }
});

// ---------------------------------------------------------------------------------------------- xyzCellPrompts

test("xyzCellPrompts follows plan order", () => {
  const { prompt, roleIds } = studioPrompt();
  const cell = (ix, cfg) => ({ ix, iy: 0, iz: 0, state: state({ seed: 77, cfg }) });
  const plan = { cells: [cell(0, 3), cell(1, 4), cell(2, 5)], order: [2, 0, 1], errors: [] };
  const prompts = xyzCellPrompts(prompt, roleIds, plan);
  assert.deepEqual(prompts.map((p) => p.cellIndex), [2, 0, 1]);
  assert.deepEqual(prompts.map((p) => inputs(p.prompt, roleIds, "sampler").cfg), [5, 3, 4]);
  assert.ok(prompts.every((p) => inputs(p.prompt, roleIds, "sampler").seed === 77));
  assert.notEqual(prompts[0].prompt, prompts[1].prompt);
});

// ---------------------------------------------------------------------------------------------- final fix wave

test("missingLoras matches Python's missing_loras (the planner's LoRA check)", () => {
  for (const testCase of golden.missing_loras) {
    assert.deepEqual(missingLoras(testCase.state, testCase.available), testCase.missing, JSON.stringify(testCase.state));
  }
  assert.deepEqual(missingLoras({ prompt: "<lora:x:1>" }, undefined), []); // list not loaded: no claim
});

test("defaultModel picks an Anima 2.9B UNET, the Qwen3 0.6B TE and the Qwen-Image VAE", () => {
  const unets = ["3d.safetensors", "anima-3.8b.safetensors", "anima-base.safetensors", "Anima-preview2.safetensors"];
  assert.equal(defaultModel("unets", unets, {}), "Anima-preview2.safetensors");
  assert.equal(defaultModel("unets", ["anima-x.safetensors"], { "anima-x.safetensors": "3.8b" }), "");
  assert.equal(defaultModel("tes", ["Anima-3.8B-expanded_adapter.safetensors", "t5.safetensors", "qwen_3_06b_base.safetensors"]),
    "qwen_3_06b_base.safetensors");
  assert.equal(defaultModel("tes", ["qwen3-0.6b.safetensors"]), "qwen3-0.6b.safetensors");
  assert.equal(defaultModel("vaes", ["ae.safetensors", "anima_vae.safetensors", "qwen_image_vae.safetensors"]), "qwen_image_vae.safetensors");
  assert.equal(defaultModel("vaes", ["ae.safetensors", "anima_vae.safetensors"]), "anima_vae.safetensors");
  assert.equal(defaultModel("vaes", ["ae.safetensors"]), "");
  assert.equal(defaultModel("loras", ["anima.safetensors"]), "");
  assert.equal(defaultModel("unets", undefined), "");
});

test("missingPicks refuses empty model picks, and 업스케일 without a model", () => {
  const picked = { unet: "a.safetensors", te: "t.safetensors", vae: "v.safetensors" };
  assert.equal(missingPicks(picked), "");
  assert.equal(missingPicks(DEFAULT_STATE), PICK_NOTICES.models);
  assert.equal(missingPicks({ ...picked, te: "" }), PICK_NOTICES.models);
  assert.equal(missingPicks({ ...picked, upscale: { on: true, model: "" } }), PICK_NOTICES.upscale);
  assert.equal(missingPicks({ ...picked, upscale: { on: true, model: "4x.pth" } }), "");
  assert.equal(missingPicks({ upscale: { model: "4x.pth" } }, "upscale"), ""); // a post upscale needs only its model
  assert.equal(missingPicks({ ...picked }, "upscale"), PICK_NOTICES.upscale);
  assert.equal(missingPicks({ ...picked, vae: "" }, "hires"), PICK_NOTICES.models);
});

test("imageSize: an i2i image's size, rounded to 8 and kept in 64..4096", () => {
  assert.deepEqual(imageSize(1248, 1824), { width: 1248, height: 1824 });
  assert.deepEqual(imageSize(1023, 1021), { width: 1024, height: 1024 });
  assert.deepEqual(imageSize(10, 9000), { width: 64, height: 4096 });
  assert.deepEqual(imageSize(4992, 7296), { width: 2800, height: 4096 }); // keeps the aspect, not 4096x4096
  assert.deepEqual(imageSize(8000, 4000), { width: 4096, height: 2048 });
  assert.equal(imageSize(undefined, 512), null);
  assert.equal(imageSize(0, 512), null);
  assert.equal(imageSize("x", 512), null);
});

test("i2i runs at its own size (the image's, once sent) and inpaint_area maps to no input", () => {
  const { prompt: api, roleIds } = studioPrompt();
  const base = { tab: "i2i", i2i_mode: "inpaint" };
  const prompt = (area) => applyStateToPrompt(api, roleIds, { ...base, i2i: { image: "a.png", width: 1248, height: 1824, inpaint_area: area } }, { rng: fixedRng });
  const latent = inputs(prompt("whole"), roleIds, "latent");
  assert.deepEqual([latent.mode, latent.width, latent.height], ["inpaint", 1248, 1824]);
  const withoutSavedState = (out) => {
    delete out[roleIds.save].inputs.studio_state; // the PNG's saved state is the only place the value shows up
    return out;
  };
  assert.deepEqual(withoutSavedState(prompt("masked")), withoutSavedState(prompt("whole"))); // the greyed setting changes nothing
  assert.equal(DEFAULT_STATE.i2i.inpaint_area, "whole"); // and the panel says so (Forge's default too)
});

// ---------------------------------------------------------------------------------------------- model families

const FAMILY_ROLES = {
  te: "1", negpip: "2", moss_stream: "3", reference_image: "4", anima_ref: "5", krea2_ref: "6", unet: "7", family_switch_model: "8",
  reference_image_2: "9",
};
const familyPrompt = () => Object.fromEntries(Object.values(FAMILY_ROLES).map((id) => [id, { inputs: {} }]));
const ready = (overrides) => state({ te: "te", vae: "vae", ...overrides });

test("modelFamily reads the checkpoint name", () => {
  assert.equal(modelFamily("moss_image_sft_bf16.safetensors"), "moss");
  assert.equal(modelFamily("sub/krea2Anime_v15.safetensors"), "krea2");
  for (const name of ["anima-2.9b.safetensors", "anima-base.safetensors", "Anima-3.8B.safetensors", ""]) assert.equal(modelFamily(name), "anima");
});

test("the TE type follows the family, and NegPiP runs on Anima only", () => {
  const out = (unet) => applyStateToPrompt(familyPrompt(), FAMILY_ROLES, ready({ unet, negpip: true }), { rng: fixedRng });
  assert.equal(out("anima.safetensors")[1].inputs.type, "stable_diffusion");
  assert.equal(out("moss_sft.safetensors")[1].inputs.type, "moss_image");
  assert.equal(out("krea2_raw.safetensors")[1].inputs.type, "krea2");
  assert.equal(out("anima.safetensors")[2].inputs.enabled, true);
  assert.equal(out("moss_sft.safetensors")[2].inputs.enabled, false);
  assert.equal(out("krea2_raw.safetensors")[2].inputs.enabled, false);
  assert.equal(out("moss_sft.safetensors")[7].inputs.unet_name, "moss_sft.safetensors"); // the core loader, as Anima's
  assert.equal(out("moss_sft.safetensors")[8].inputs.select, 0);
});

test("the fp32 stream runs on MOSS with its switch on", () => {
  const mode = (overrides) => stageModes(state(overrides)).moss_stream;
  assert.equal(mode({ unet: "moss_sft.safetensors" }), 0);
  assert.equal(mode({ unet: "moss_sft.safetensors", moss_stream: false }), 4);
  assert.equal(mode({ unet: "anima.safetensors" }), 4);
  assert.equal(mode({ unet: "krea2.safetensors" }), 4);
});

test("the reference runs its family's node only", () => {
  const reference = { ...structuredClone(DEFAULT_STATE.reference), on: true, image: "ref.png" };
  const modes = (unet, on = true) => {
    const { anima_ref, krea2_ref, reference_image } = stageModes(state({ unet, reference: { ...reference, on } }));
    return [anima_ref, krea2_ref, reference_image];
  };
  assert.deepEqual(modes("anima.safetensors"), [0, 4, 0]);
  assert.deepEqual(modes("krea2.safetensors"), [4, 0, 0]);
  assert.deepEqual(modes("moss.safetensors"), [4, 4, 4]);
  assert.deepEqual(modes("anima.safetensors", false), [4, 4, 4]);
});

test("the reference nodes get their method's settings and the tab's size", () => {
  const reference = structuredClone(DEFAULT_STATE.reference);
  reference.snapref = { ...reference.snapref, file: "snapref_v1a.safetensors", mode: "char", strength: 0.5 };
  reference.ref_adapter = { ...reference.ref_adapter, file: "anima_ref_adapter_v2_r64.safetensors", strength: 1.25 };
  reference.krea2 = { ...reference.krea2, file: "QuadView_krea2_v1.safetensors", grounding_px: 1024 };
  const out = (changes, overrides = {}) =>
    applyStateToPrompt(familyPrompt(), FAMILY_ROLES, ready({ reference: { ...reference, image: "ref.png", ...changes }, ...overrides }), { rng: fixedRng });
  const snapref = out({})[5].inputs;
  assert.equal(snapref.method, "snapref");
  assert.equal(snapref.file, "snapref_v1a.safetensors");
  assert.equal(snapref.mode, "char");
  assert.equal(snapref.strength, 0.5);
  assert.equal(snapref.guidance_on, true);
  assert.equal(snapref.on, undefined);
  const adapter = out({ method: "ref_adapter" })[5].inputs;
  assert.equal(adapter.file, "anima_ref_adapter_v2_r64.safetensors");
  assert.equal(adapter.strength, 1.25);
  assert.equal(adapter.caption, undefined);
  assert.equal(out({})[4].inputs.image, "ref.png");
  const i2i = { ...DEFAULT_STATE.i2i, width: 512, height: 640 };
  assert.deepEqual([out({})[6].inputs.width, out({})[6].inputs.height], [832, 1216]);
  const krea2 = out({}, { tab: "i2i", i2i })[6].inputs;
  assert.deepEqual([krea2.file, krea2.grounding_px, krea2.width, krea2.height], ["QuadView_krea2_v1.safetensors", 1024, 512, 640]);
});

test("the second reference runs only on Anima's SnapRef, with its own picture", () => {
  const reference = { ...structuredClone(DEFAULT_STATE.reference), on: true, image: "ref.png", image_2: "two.png" };
  const mode = (unet, changes = {}) => stageModes(state({ unet, reference: { ...reference, ...changes } })).reference_image_2;
  assert.equal(mode("anima.safetensors"), 0);
  assert.equal(mode("anima.safetensors", { image_2: "" }), 4);
  assert.equal(mode("anima.safetensors", { method: "ref_adapter" }), 4);
  assert.equal(mode("anima.safetensors", { on: false }), 4);
  assert.equal(mode("krea2.safetensors"), 4);
  assert.equal(mode("moss.safetensors"), 4);
});

test("the second picture, Fidelity and the weights reach their nodes", () => {
  const base = structuredClone(DEFAULT_STATE.reference);
  const reference = { ...base, on: true, image: "ref.png", image_2: "two.png",
    snapref: { ...base.snapref, file: "snapref_v2a_pre2.safetensors", fidelity: 0.7, weight_1: 0.8, weight_2: 0.4 } };
  const out = applyStateToPrompt(familyPrompt(), FAMILY_ROLES, ready({ unet: "anima.safetensors", reference }), { rng: fixedRng });
  const node = (role) => out[FAMILY_ROLES[role]].inputs;
  assert.equal(node("reference_image_2").image, "two.png");
  assert.deepEqual([node("anima_ref").fidelity, node("anima_ref").weight_1, node("anima_ref").weight_2], [0.7, 0.8, 0.4]);
});

test("a reference run needs its image and file", () => {
  const reference = (changes) => ({ ...structuredClone(DEFAULT_STATE.reference), on: true, ...changes });
  const picks = (unet, ref) => missingPicks(ready({ unet, reference: ref }));
  assert.equal(picks("anima.safetensors", reference({})), PICK_NOTICES.referenceImage);
  assert.equal(picks("anima.safetensors", reference({ image: "a.png" })), PICK_NOTICES.referenceFile);
  const snapref = reference({ image: "a.png", snapref: { ...DEFAULT_STATE.reference.snapref, file: "s.safetensors" } });
  assert.equal(picks("anima.safetensors", snapref), "");
  assert.equal(referenceFile({ reference: snapref }, "anima"), "s.safetensors");
  assert.equal(picks("anima.safetensors", { ...snapref, method: "ref_adapter" }), PICK_NOTICES.referenceFile); // each method has its own file
  assert.equal(picks("krea2.safetensors", snapref), PICK_NOTICES.referenceFile);
  assert.equal(picks("moss.safetensors", reference({})), ""); // MOSS has no reference: the switch does nothing
  assert.equal(missingPicks(ready({ unet: "anima.safetensors", reference: reference({}), upscale: { ...DEFAULT_STATE.upscale, model: "x" } }), "upscale"), "");
});

test("pickCheckpoint keeps each family's picks", () => {
  const lists = {
    tes: ["qwen_3_06b_base.safetensors", "qwen3.5_9b_moss_image_sft_bf16.safetensors", "qwen3vl_4b_bf16.safetensors"],
    vaes: ["moss_image_vae_fp32.safetensors", "qwen_image_vae.safetensors", "qwen_image_vae_HDR_fp32.safetensors"],
  };
  const s = state({ unet: "anima.safetensors", te: "qwen_3_06b_base.safetensors", vae: "my_vae.safetensors", steps: 28, cfg: 4 });
  pickCheckpoint(s, "moss_image_sft_bf16.safetensors", lists);
  assert.deepEqual([s.te, s.vae], ["qwen3.5_9b_moss_image_sft_bf16.safetensors", "moss_image_vae_fp32.safetensors"]);
  assert.deepEqual([s.sampler, s.scheduler, s.steps, s.cfg, s.shift], ["euler", "sgm_uniform", 50, 3, 5]);
  s.steps = 40;
  pickCheckpoint(s, "moss_image_base_bf16.safetensors", lists); // same family: nothing moves
  assert.equal(s.steps, 40);
  pickCheckpoint(s, "krea2_turbo_bf16.safetensors", lists);
  assert.deepEqual([s.te, s.vae], ["qwen3vl_4b_bf16.safetensors", "qwen_image_vae.safetensors"]);
  assert.deepEqual([s.sampler, s.scheduler, s.steps, s.cfg, s.shift], ["euler", "simple", 8, 1, 1.15]);
  pickCheckpoint(s, "anima.safetensors", lists);
  assert.deepEqual([s.te, s.vae, s.steps, s.cfg], ["qwen_3_06b_base.safetensors", "my_vae.safetensors", 28, 4]);
  pickCheckpoint(s, "moss_image_sft_bf16.safetensors", lists);
  assert.equal(s.steps, 40);
  const raw = state({ unet: "anima.safetensors" });
  pickCheckpoint(raw, "krea2_raw_bf16.safetensors", lists);
  assert.deepEqual([raw.steps, raw.cfg], [20, 3]);
});

test("Krea2 Raw and Turbo keep their own numbers and share the model picks", () => {
  const s = state({ unet: "krea2_raw_bf16.safetensors", te: "uncensored.safetensors", vae: "v.safetensors", steps: 24, cfg: 3.5 });
  pickCheckpoint(s, "krea2_turbo_bf16.safetensors");
  assert.deepEqual([s.steps, s.cfg, s.te, s.vae], [8, 1, "uncensored.safetensors", "v.safetensors"]);
  s.steps = 6;
  pickCheckpoint(s, "krea2_raw_bf16.safetensors");
  assert.deepEqual([s.steps, s.cfg], [24, 3.5]);
  pickCheckpoint(s, "krea2_turbo_bf16.safetensors");
  assert.equal(s.steps, 6);
});

test("withDefaults fills a reference saved with fewer keys", () => {
  const s = withDefaults({ reference: { on: true, snapref: { file: "a.safetensors" } } });
  assert.equal(s.reference.on, true);
  assert.equal(s.reference.snapref.file, "a.safetensors");
  assert.equal(s.reference.snapref.guidance, 4.5);
  assert.equal(s.reference.krea2.grounding_px, 768);
});

test("a family's default TE and VAE", () => {
  const vaes = ["moss_image_vae_fp32.safetensors", "qwen_image_vae_HDR_fp32.safetensors", "sub/qwen_image_vae.safetensors"];
  assert.equal(defaultModel("vaes", vaes, null, "moss"), "moss_image_vae_fp32.safetensors");
  assert.equal(defaultModel("vaes", vaes, null, "krea2"), "sub/qwen_image_vae.safetensors");
  assert.equal(defaultModel("tes", ["krea2UncensoredLLMCLIP_v10.safetensors"], null, "krea2"), "");
});

test("MOSS sizes snap to its 16x grid at run time and in the send-to size", () => {
  const roles = { latent: "1", krea2_ref: "2" };
  const prompt = { 1: { inputs: {} }, 2: { inputs: {} } };
  const out = (unet) => applyStateToPrompt(prompt, roles, ready({ unet, width: 856, height: 840 }), { rng: fixedRng })[1].inputs;
  assert.deepEqual([out("moss_sft.safetensors").width, out("moss_sft.safetensors").height], [864, 848]);
  assert.deepEqual([out("anima.safetensors").width, out("anima.safetensors").height], [856, 840]);
  assert.equal(sizeStep("moss"), 16);
  assert.equal(sizeStep("krea2"), 8);
  assert.deepEqual(imageSize(1000, 1000, sizeStep("moss")), { width: 1008, height: 1008 });
  assert.deepEqual(imageSize(1000, 1000), { width: 1000, height: 1000 });
});

test("Krea2's reference node gets the prompt editing settings", () => {
  const prompt = { 2: { inputs: {} } };
  const out = applyStateToPrompt(prompt, { krea2_ref: "2" }, ready({ unet: "krea2.safetensors", steps: 12 }), { rng: fixedRng })[2].inputs;
  assert.deepEqual([out.steps, out.forge_syntax], [12, true]);
});

test("SnapRef refuses PAG, Ref Adapter names what it loses", () => {
  const ext = structuredClone(DEFAULT_STATE.ext);
  Object.assign(ext.guidance, { on: true, guid_enabled: true, guid_attn_method: "PAG", guid_scale: 3, guid_dave_enabled: true, guid_slg_on: true });
  const reference = { ...structuredClone(DEFAULT_STATE.reference), on: true, image: "a.png" };
  reference.snapref.file = "s.safetensors";
  reference.ref_adapter.file = "a.safetensors";
  const anima = (ref, extState = ext) => ready({ unet: "anima-base.safetensors", reference: ref, ext: extState });
  assert.equal(missingPicks(anima(reference)), PICK_NOTICES.snaprefPag);
  assert.equal(missingPicks(anima({ ...reference, on: false })), "");
  const adapter = { ...reference, method: "ref_adapter" };
  assert.equal(missingPicks(anima(adapter)), "");
  assert.deepEqual(refAdapterDrops(anima(adapter)), ["DAVE", "SLG"]);
  const seg = structuredClone(ext);
  Object.assign(seg.guidance, { guid_attn_method: "SEG", guid_dave_enabled: false, guid_slg_on: false });
  assert.equal(missingPicks(anima(reference, seg)), "");
  assert.deepEqual(refAdapterDrops(anima(adapter, seg)), ["SEG"]);
  assert.deepEqual(refAdapterDrops(anima(reference, seg)), []); // SnapRef keeps SEG
  const parked = structuredClone(ext);
  parked.guidance.guid_scale = 0;
  assert.equal(missingPicks(anima(reference, parked)), ""); // PAG at scale 0 installs nothing
});

test("hasPromptWeights finds explicit weights, not plain parentheses", () => {
  assert.equal(hasPromptWeights(ready({ prompt: "(red hair:1.3), girl" })), true);
  assert.equal(hasPromptWeights(ready({ negative: "(blurry:0.7)" })), true);
  assert.equal(hasPromptWeights(ready({ prompt: "saber_(fate), \\(x:1.2\\)" })), false);
  assert.equal(hasPromptWeights(ready({ prompt: "(saber_(fate):1.3), (blue hair:.8)" })), true);
  assert.equal(hasPromptWeights(ready({ prompt: "(hatsune_miku_\\(vocaloid\\):1.2)" })), true);
  assert.equal(hasPromptWeights(ready({ prompt: "plain (word) 10:30" })), false);
});
