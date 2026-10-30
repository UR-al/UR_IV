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
  missingPicks,
  postPrompt,
  postRunState,
  recordFromParameters,
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
    shift: ["ForgeNeoModelSamplingShift", { model: ["family_switch_model", 0], shift: 1.0 }],
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
  assert.equal(ROLES.length, 49);
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
    "i2i_image", "neg_encode38", "pos_encode38", "post_decode", "post_encode", "post_hires", "post_image",
    "post_sam3_detailer", "post_sam3_mask", "post_save", "post_upscale", "qwen35", "sam3_detailer", "sam3_mask", "unet38",
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
  });
});

test("recordFromParameters reads multi-line prompts, quoted values and no negative", () => {
  const text = 'line one\nline two\nSteps: 20, Sampler: "DPM++ 2M, Karras", CFG scale: 4.5, Seed: 123, Size: 512x768, Model: "a:b"';
  assert.deepEqual(recordFromParameters(text), {
    prompt: "line one\nline two", negative: "", seed: 123, steps: 20, cfg: 4.5, width: 512, height: 768,
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
