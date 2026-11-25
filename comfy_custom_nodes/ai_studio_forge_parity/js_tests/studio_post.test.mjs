// node:test for the post-processing settings: the table (studio_post.js) and what studio_logic.js writes from it.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { POST_SCHEMA } from "../web/ai_studio/post_schema.js";
import { extInputs, extModes } from "../web/ai_studio/studio_ext.js";
import { hiresSize, postDefaults, postInputs, scaleFromName } from "../web/ai_studio/studio_post.js";
import { hiresSizeText } from "../web/ai_studio/post_view.js";
import {
  DEFAULT_STATE,
  PICK_NOTICES,
  applyStateToPrompt,
  generationSize,
  hasPromptWeights,
  missingPicks,
  missingSam3Picks,
  ownPromptOn38,
  pickCheckpoint,
  postPrompt,
  postRunState,
  stageModes,
} from "../web/ai_studio/studio_logic.js";

const ROLE_NODES = {
  hires: "ForgeNeoHiresFix", upscale: "AIStudioUpscaleImage", sam3_mask: "ForgeNeoSAM3Mask", sam3_detailer: "ForgeNeoSAM3Detailer",
  post_image: "LoadImage", post_upscale: "AIStudioUpscaleImage", post_hires: "ForgeNeoHiresFix", post_decode: "VAEDecode",
  post_sam3_mask: "ForgeNeoSAM3Mask", post_sam3_detailer: "ForgeNeoSAM3Detailer", post_save: "AIStudioSaveForge", save: "AIStudioSaveForge",
};
const roleIds = Object.fromEntries(Object.keys(ROLE_NODES).map((role, index) => [role, String(index + 1)]));

/** A small studio graph: the upscale feeds the mask and the detailer, the detailer the save, and the same again for the post row. */
function graph() {
  const nodes = Object.fromEntries(Object.entries(ROLE_NODES).map(([role, classType]) => [roleIds[role], { class_type: classType, inputs: {} }]));
  const node = (role) => nodes[roleIds[role]].inputs;
  Object.assign(node("sam3_mask"), { image: [roleIds.upscale, 0] });
  Object.assign(node("sam3_detailer"), { image: [roleIds.upscale, 0], mask: [roleIds.sam3_mask, 0] });
  Object.assign(node("save"), { images: [roleIds.sam3_detailer, 0] });
  Object.assign(node("post_upscale"), { image: [roleIds.post_image, 0] });
  Object.assign(node("post_decode"), { samples: [roleIds.post_hires, 0] });
  Object.assign(node("post_sam3_mask"), { image: [roleIds.post_decode, 0] });
  Object.assign(node("post_sam3_detailer"), { image: [roleIds.post_decode, 0], mask: [roleIds.post_sam3_mask, 0] });
  Object.assign(node("post_save"), { images: [roleIds.post_sam3_detailer, 0] });
  return nodes;
}

const state = (overrides = {}) => ({ ...structuredClone(DEFAULT_STATE), unet: "a.safetensors", te: "t", vae: "v", seed: 7, ...overrides });
const sam3 = (overrides) => ({ ...DEFAULT_STATE.sam3, on: true, checkpoint: "C:/sam3.pt", ...overrides });
const hires = (overrides) => ({ ...DEFAULT_STATE.hires, on: true, ...overrides });
const upscale = (overrides) => ({ ...DEFAULT_STATE.upscale, on: true, model: "4x.pth", ...overrides });
const run = (s, options) => applyStateToPrompt(graph(), roleIds, s, options);
const inputs = (prompt, role) => prompt[roleIds[role]].inputs;
const record = { prompt: "x", seed: 3, image: { filename: "a.png", type: "output" } };

test("only -1 draws a seed: any other value stays through a run", () => {
  const seedOf = (seed) => inputs(run(state({ seed, sam3: sam3(), hires: hires() }), { rng: () => 0.5 }), "sam3_detailer").seed;
  assert.equal(seedOf(123), 123);
  assert.equal(seedOf(0), 0);
  assert.equal(seedOf(12.7), 12);
  assert.equal(seedOf(-1), 2147483647);
  const s = state({ seed: 123 });
  run(s);
  assert.equal(s.seed, 123);
});

test("postDefaults gives every field of a section", () => {
  assert.deepEqual(Object.keys(postDefaults("upscale")), POST_SCHEMA.upscale.map((field) => field.key));
  assert.equal(postDefaults("sam3").threshold, 0.4);
});

test("postInputs keeps a node's inputs and leaves an empty pick to the workflow", () => {
  const values = { ...postDefaults("sam3"), threshold: 0.2, fill_mode: "fill" };
  const mask = postInputs("sam3", values, "mask");
  assert.equal(mask.threshold, 0.2);
  assert.equal(mask.mask_mode, "Individual");
  assert.equal("fill_mode" in mask, false);
  assert.equal("face_crop_model" in mask, false);
  const detailer = postInputs("sam3", values, "detailer");
  assert.equal(detailer.fill_mode, "fill");
  assert.equal(detailer.mask_mode, "Individual");
  assert.equal("threshold" in detailer, false);
  assert.equal(postInputs("sam3", { ...values, face_crop_model: "C:/f.pt" }, "mask").face_crop_model, "C:/f.pt");
});

test("hiresSize is the node's arithmetic", () => {
  const size = (width, height, overrides) => hiresSize(width, height, { scale: 1.5, resize_width: 0, resize_height: 0, size_step: 8, ...overrides });
  assert.deepEqual(size(832, 1216, { size_step: 64 }), { width: 1280, height: 1856 });
  assert.deepEqual(size(832, 1216), { width: 1248, height: 1824 });
  assert.deepEqual(size(832, 1216, { resize_width: 1024 }), { width: 1024, height: 1496 });
  assert.deepEqual(size(832, 1216, { resize_height: 1216 }), { width: 832, height: 1216 });
  assert.deepEqual(size(832, 1216, { resize_width: 1024, resize_height: 1024, size_step: 64 }), { width: 1024, height: 1024 });
  assert.deepEqual(size(8, 8, { scale: 1, size_step: 64 }), { width: 64, height: 64 });
});

test("hiresSize ends on the model's latent grid, as the node floors it", () => {
  const size = (overrides, grid) => hiresSize(832, 1216, { scale: 1.5, resize_width: 0, resize_height: 0, size_step: 8, ...overrides }, grid);
  assert.deepEqual(size({ resize_width: 1000, resize_height: 1000 }, 16), { width: 992, height: 992 });
  assert.deepEqual(size({ resize_width: 1000, resize_height: 1000 }), { width: 1000, height: 1000 });
  assert.deepEqual(size({ resize_width: 8, resize_height: 8 }, 16), { width: 16, height: 16 });
});

test("the Hires card's size text knows MOSS's grid", () => {
  const moss = state({ unet: "moss.safetensors", hires: { ...DEFAULT_STATE.hires, resize_width: 1000, resize_height: 1000, size_step: 8 } });
  assert.equal(hiresSizeText(moss), "992 × 992");
  assert.equal(hiresSizeText(state({ hires: { ...DEFAULT_STATE.hires, scale: 1.5 } })), "1280 × 1856");
});

test("scaleFromName reads the factor Forge reads", () => {
  assert.equal(scaleFromName("4x-UltraSharp.pth"), 4);
  assert.equal(scaleFromName("RealESRGAN_x2plus.pth"), 2);
  assert.equal(scaleFromName("Lanczos"), null);
});

test("SAM3 detection and mask settings reach the mask node, the inpaint ones the detailer", () => {
  const out = run(state({ sam3: sam3({ threshold: 0.25, exclude_prompt: "hand", mask_blur: 8, invert: true, mask_mode: "Combined", fill_mode: "latent noise", mask_padding: 64, only_masked: false, inpaint_prompt: "eyes", noise_multiplier: 1.2 }) }));
  const mask = inputs(out, "sam3_mask");
  assert.deepEqual([mask.threshold, mask.exclude_prompt, mask.mask_blur, mask.invert, mask.mask_mode], [0.25, "hand", 8, true, "Combined"]);
  const detailer = inputs(out, "sam3_detailer");
  assert.deepEqual([detailer.mask_mode, detailer.fill_mode, detailer.mask_padding, detailer.only_masked, detailer.inpaint_prompt, detailer.noise_multiplier],
    ["Combined", "latent noise", 64, false, "eyes", 1.2]);
});

test("a SAM3 pass takes its own sampling settings and otherwise the main ones", () => {
  const main = inputs(run(state({ steps: 30, cfg: 3.5, sampler: "euler", scheduler: "simple", sam3: sam3() })), "sam3_detailer");
  assert.deepEqual([main.steps, main.cfg, main.sampler_name, main.scheduler, main.seed], [30, 3.5, "euler", "simple", 7]);
  const own = inputs(run(state({ sam3: sam3({ steps: 12, cfg: 6, sampler: "heun", scheduler: "karras", seed: 99 }) })), "sam3_detailer");
  assert.deepEqual([own.steps, own.cfg, own.sampler_name, own.scheduler, own.seed], [12, 6, "heun", "karras", 99]);
});

test("the crop is drawn at the generation size, an i2i run's own, and MOSS's grid", () => {
  const size = (s) => [inputs(run(s), "sam3_detailer").target_width, inputs(run(s), "sam3_detailer").target_height];
  assert.deepEqual(size(state({ width: 832, height: 1216, sam3: sam3() })), [832, 1216]);
  assert.deepEqual(size(state({ tab: "i2i", i2i: { ...DEFAULT_STATE.i2i, width: 1024, height: 768 }, sam3: sam3() })), [1024, 768]);
  assert.deepEqual(size(state({ unet: "moss.safetensors", width: 840, height: 1200, sam3: sam3() })), [848, 1200]);
  assert.deepEqual(generationSize(state({ unet: "moss.safetensors", width: 840, height: 1200 })), { width: 848, height: 1200 });
});

test("a gallery SAM3 run lets the node use the image's size", () => {
  const { prompt } = postPrompt(graph(), roleIds, record, "sam3", state({ sam3: sam3() }), { rng: () => 0.5 });
  const detailer = inputs(prompt, "post_sam3_detailer");
  assert.equal(detailer.target_image_size, true);
  assert.equal(detailer.target_width, undefined);
  assert.equal(detailer.enabled, true);
});

test("Mask only reads the mask node's image output: the picture itself, or the overlay when asked", () => {
  const plain = state({ sam3: sam3({ mode: "Mask only" }) });
  assert.equal(stageModes(plain).sam3_detailer, 0); // bypassed, the prompt would hold no link to move
  assert.equal(stageModes(plain).sam3_mask, 0);
  const out = run(plain);
  assert.equal(inputs(out, "sam3_detailer").enabled, false);
  assert.equal(inputs(out, "sam3_mask").enabled, true);
  assert.equal(inputs(out, "sam3_mask").plain_overlay, true);
  assert.deepEqual(inputs(out, "save").images, [roleIds.sam3_mask, 3]);
  const overlay = run(state({ sam3: sam3({ mode: "Mask only", overlay: true }) }));
  assert.equal(inputs(overlay, "sam3_mask").plain_overlay, false);
  assert.deepEqual(inputs(overlay, "save").images, [roleIds.sam3_mask, 3]);
  assert.deepEqual(inputs(run(state({ sam3: sam3({ overlay: true }) })), "save").images, [roleIds.sam3_detailer, 0]);
});

test("Mask only draws nothing: Refine and tile stay out, the overlay still reaches the save", () => {
  const ext = {
    ...DEFAULT_STATE.ext,
    sam3_refine: { ...DEFAULT_STATE.ext.sam3_refine, on: true },
    sam3_tile: { ...DEFAULT_STATE.ext.sam3_tile, on: true },
  };
  const modes = stageModes(state({ ext, sam3: sam3({ mode: "Mask only", overlay: true }) }));
  assert.deepEqual([modes.ext_sam3_refine, modes.ext_sam3_tile, modes.sam3_detailer, modes.sam3_mask], [4, 4, 0, 0]);
  const painting = stageModes(state({ ext, sam3: sam3() }));
  assert.deepEqual([painting.ext_sam3_refine, painting.ext_sam3_tile, painting.sam3_detailer], [0, 0, 4]); // R22
  const alone = stageModes(state({ ext }));
  assert.deepEqual([alone.ext_sam3_refine, alone.ext_sam3_tile, alone.sam3_mask], [0, 0, 0]);
});

test("a gallery Mask only run does the same on the post row, overlay or not", () => {
  for (const overlay of [true, false]) {
    const current = state({ sam3: sam3({ mode: "Mask only", overlay }) });
    const { prompt } = postPrompt(graph(), roleIds, record, "sam3", current, { rng: () => 0.5 });
    assert.equal(inputs(prompt, "post_sam3_detailer").enabled, false);
    assert.equal(inputs(prompt, "post_sam3_mask").plain_overlay, !overlay);
    assert.deepEqual(inputs(prompt, "post_save").images, [roleIds.post_sam3_mask, 3]);
  }
});

test("a gallery Mask only run loads no model, so it asks for none", () => {
  const noModels = state({ unet: "", te: "", vae: "", sam3: sam3({ mode: "Mask only" }) });
  assert.equal(missingPicks(noModels, "sam3"), "");
  assert.equal(missingPicks({ ...noModels, sam3: sam3() }, "sam3"), PICK_NOTICES.models);
  assert.equal(missingPicks({ ...noModels, sam3: sam3({ mode: "Mask only", checkpoint: "" }) }, "sam3"), PICK_NOTICES.sam3);
  assert.equal(missingPicks(noModels), PICK_NOTICES.models); // a generation still draws the picture
});

test("a run of another family's image drops the panel's Hires VAE", () => {
  const panel = state({ unet: "a.safetensors", hires: { ...DEFAULT_STATE.hires, vae: "x.safetensors" } });
  assert.equal(postRunState(state({ unet: "moss.safetensors" }), panel, "hires").hires.vae, "");
  assert.equal(postRunState(state({ unet: "b.safetensors" }), panel, "hires").hires.vae, "x.safetensors");
  assert.equal(panel.hires.vae, "x.safetensors");
});

test("the mask node's run settings: cpu forces fp32, keep-in-RAM follows the app's rule", () => {
  const mask = (overrides) => inputs(run(state({ sam3: sam3(overrides) })), "sam3_mask");
  assert.deepEqual([mask({}).device, mask({}).precision, mask({}).cache_model], ["auto", "bf16", true]);
  assert.equal(mask({ device: "cpu" }).precision, "fp32");
  assert.equal(mask({ precision: "fp16" }).precision, "fp16");
  assert.equal(mask({ unload_after: true, keep_in_ram: false }).cache_model, false);
  assert.equal(mask({ unload_after: false, keep_in_ram: false }).cache_model, true);
  assert.equal(mask({}).seed, 7);
});

test("ControlNet: an empty range is a ControlNet that does nothing, as in Forge", () => {
  const detailer = inputs(run(state({ sam3: sam3({ cn_enable: true, cn_model: "x", cn_start: 0.5, cn_end: 0.5 }) })), "sam3_detailer");
  assert.equal(detailer.controlnet_enable, false);
});

test("ControlNet: a reversed range is swapped and the node's JSON carries the rest", () => {
  const detailer = inputs(run(state({ sam3: sam3({ cn_enable: true, cn_model: "lllite.safetensors", cn_strength: 0.8, cn_start: 0.9, cn_end: 0.1, cn_control_mode: "ControlNet is more important", cn_threshold_a: 100 }) })), "sam3_detailer");
  assert.deepEqual([detailer.controlnet_enable, detailer.controlnet_model_name, detailer.controlnet_strength], [true, "lllite.safetensors", 0.8]);
  assert.deepEqual([detailer.controlnet_start, detailer.controlnet_end], [0.1, 0.9]);
  assert.deepEqual(JSON.parse(detailer.controlnet_settings_json), {
    pixel_perfect: true, override_external: false, control_mode: "ControlNet is more important", resize_mode: "Crop and Resize", threshold_a: 100, threshold_b: -1,
  });
});

test("Hires takes its own CFG, Shift and scheduler, and 0 or empty keeps the main ones", () => {
  const main = inputs(run(state({ cfg: 3, shift: 2, scheduler: "simple", hires: hires() })), "hires");
  assert.deepEqual([main.cfg, main.shift, main.scheduler, main.vae_name], [3, 2, "simple", "Use same choices"]);
  const own = inputs(run(state({ hires: hires({ cfg: 5, shift: 4, scheduler: "karras", vae: "x.safetensors" }) })), "hires");
  assert.deepEqual([own.cfg, own.shift, own.scheduler, own.vae_name], [5, 4, "karras", "x.safetensors"]);
});

test("Hires writes its size, tiles and prompts; the gallery row gets the same", () => {
  const settings = { resize_width: 1024, resize_height: 1536, size_step: 8, tile_size: 256, tile_overlap: 16, prompt: "detail", negative: "blur" };
  const check = (node) => assert.deepEqual(
    [node.resize_width, node.resize_height, node.size_step, node.tile_size, node.tile_overlap, node.positive_text, node.negative_text],
    [1024, 1536, 8, 256, 16, "detail", "blur"],
  );
  check(inputs(run(state({ hires: hires(settings) })), "hires"));
  check(inputs(postPrompt(graph(), roleIds, record, "hires", state({ hires: hires(settings) }), { rng: () => 0.5 }).prompt, "post_hires"));
});

test("a second upscaler at no visibility is not named: it never runs", () => {
  const node = (overrides) => inputs(run(state({ upscale: upscale({ model_2: "gone.pth", ...overrides }) })), "upscale");
  assert.equal(node({}).model_name_2, "None");
  assert.equal(node({ visibility: 0.5 }).model_name_2, "gone.pth");
});

test("upscale writes Forge's Extras settings", () => {
  const node = inputs(run(state({ upscale: upscale({ mode: "Scale to", width: 2048, height: 1024, crop: false, max_side: 3000, model_2: "2x.pth", visibility: 0.5, color_correction: true, tile_size: 256, tile_overlap: 16 }) })), "upscale");
  assert.deepEqual(
    [node.resize_mode, node.width, node.height, node.crop, node.max_side_length, node.model_name_2, node.upscaler_2_visibility, node.color_correction, node.tile_size, node.tile_overlap],
    ["Scale to", 2048, 1024, false, 3000, "2x.pth", 0.5, true, 256, 16],
  );
  assert.equal(inputs(run(state({ upscale: upscale() })), "upscale").model_name_2, "None");
});

test("a switched-off stage never names a file: a moved file must not stop a run", () => {
  const out = run(state({
    hires: { ...DEFAULT_STATE.hires, vae: "gone.safetensors" },
    upscale: { ...DEFAULT_STATE.upscale, model: "gone.pth", model_2: "gone2.pth" },
  }));
  assert.equal(inputs(out, "hires").vae_name, "Use same choices");
  assert.equal(inputs(out, "upscale").model_name, "Lanczos");
  assert.equal(inputs(out, "upscale").model_name_2, "None");
  const running = run(state({ hires: hires({ vae: "a.safetensors" }), upscale: upscale({ model_2: "2x.pth", visibility: 0.5 }) }));
  assert.equal(inputs(running, "hires").vae_name, "a.safetensors");
  assert.deepEqual([inputs(running, "upscale").model_name, inputs(running, "upscale").model_name_2], ["4x.pth", "2x.pth"]);
});

test("a gallery run names only its own stage's files", () => {
  const current = state({ hires: { ...DEFAULT_STATE.hires, vae: "gone.safetensors" }, upscale: { ...DEFAULT_STATE.upscale, model: "4x.pth", model_2: "2x.pth", visibility: 0.5 } });
  const { prompt } = postPrompt(graph(), roleIds, record, "upscale", current, { rng: () => 0.5 });
  assert.deepEqual([inputs(prompt, "post_upscale").model_name, inputs(prompt, "post_upscale").model_name_2], ["4x.pth", "2x.pth"]);
  assert.equal(inputs(prompt, "post_hires").vae_name, "Use same choices");
  const hiresRun = postPrompt(graph(), roleIds, record, "hires", current, { rng: () => 0.5 }).prompt;
  assert.equal(inputs(hiresRun, "post_hires").vae_name, "gone.safetensors");
  assert.equal(inputs(hiresRun, "post_upscale").model_name, "Lanczos");
});

test("changing the model family drops the Hires VAE, and a checkpoint of the same family keeps it", () => {
  const picked = (unet) => {
    const s = state({ unet: "a.safetensors", hires: { ...DEFAULT_STATE.hires, vae: "x.safetensors" } });
    pickCheckpoint(s, unet);
    return s.hires.vae;
  };
  assert.equal(picked("moss.safetensors"), "");
  assert.equal(picked("b.safetensors"), "x.safetensors");
});

test("a SAM3 run needs its picks", () => {
  assert.equal(missingPicks(state({ sam3: { ...DEFAULT_STATE.sam3, on: true } })), PICK_NOTICES.sam3);
  assert.equal(missingPicks(state(), "sam3"), PICK_NOTICES.sam3);
  assert.equal(missingPicks(state({ sam3: sam3({ face_crop: true }) })), PICK_NOTICES.faceModel);
  assert.equal(missingPicks(state({ sam3: sam3({ cn_enable: true }) })), PICK_NOTICES.controlnet);
  assert.equal(missingPicks(state({ sam3: sam3() })), "");
  assert.equal(missingPicks(state()), "");
  assert.equal(missingPicks(state({ ext: { ...DEFAULT_STATE.ext, sam3_refine: { ...DEFAULT_STATE.ext.sam3_refine, on: true } } })), PICK_NOTICES.sam3);
});

test("ControlNet's pick matters only where the detailer draws", () => {
  const cn = sam3({ cn_enable: true });
  assert.equal(missingPicks(state({ sam3: { ...cn, mode: "Mask only" } })), "");
  assert.equal(missingPicks(state({ sam3: cn, ext: { ...DEFAULT_STATE.ext, sam3_refine: { ...DEFAULT_STATE.ext.sam3_refine, on: true } } })), "");
  assert.equal(missingPicks(state({ sam3: { ...cn, on: false } })), "");
  assert.equal(missingPicks(state({ sam3: { ...cn, on: false } }), "sam3"), PICK_NOTICES.controlnet);
});

test("a plot can bring the SAM3 file in its cells", () => {
  const panel = state({ sam3: { ...DEFAULT_STATE.sam3, on: true } });
  assert.equal(missingPicks(panel, null, { sam3: false }), "");
  assert.equal(missingSam3Picks(panel), PICK_NOTICES.sam3);
  assert.equal(missingPicks(state({ unet: "" }), null, { sam3: false }), PICK_NOTICES.models);
});

test("a pass prompt of its own is reported on 3.8B only", () => {
  const with38 = (overrides) => state({ unet: "anima_3.8b_v2.safetensors", ...overrides });
  assert.equal(ownPromptOn38(with38({ hires: hires({ prompt: "x" }) })), true);
  assert.equal(ownPromptOn38(with38({ sam3: sam3({ negative_prompt: "x" }) })), true);
  assert.equal(ownPromptOn38(with38({ hires: hires() })), false);
  assert.equal(ownPromptOn38(with38({ hires: { ...DEFAULT_STATE.hires, prompt: "x" } })), false);
  assert.equal(ownPromptOn38(state({ hires: hires({ prompt: "x" }) })), false);
});

test("the prompt notices look at the passes that run", () => {
  const refine = (on, prompt = "") => ({ ...DEFAULT_STATE.ext, sam3_refine: { ...DEFAULT_STATE.ext.sam3_refine, on, inpaint_prompt: prompt } });
  const with38 = (overrides) => state({ unet: "anima_3.8b_v2.safetensors", ...overrides });
  assert.equal(ownPromptOn38(with38({ tab: "i2i", hires: hires({ prompt: "x" }) })), false); // no Hires on i2i
  assert.equal(ownPromptOn38(with38({ sam3: sam3({ mode: "Mask only", inpaint_prompt: "x" }) })), false);
  assert.equal(ownPromptOn38(with38({ ext: refine(true), sam3: sam3({ inpaint_prompt: "x" }) })), false); // Refine replaces the detailer
  assert.equal(ownPromptOn38(with38({ ext: refine(true, "x") })), true);
  assert.equal(ownPromptOn38(with38({ ext: { ...DEFAULT_STATE.ext, sam3_tile: { ...DEFAULT_STATE.ext.sam3_tile, on: true, negative_prompt: "x" } } })), true);
  assert.equal(ownPromptOn38(with38({ ext: refine(true, "x"), sam3: sam3({ mode: "Mask only" }) })), false);
});

test("a gallery run's notice looks at its own pass", () => {
  const with38 = (overrides) => state({ unet: "anima_3.8b_v2.safetensors", ...overrides });
  const off = { ...DEFAULT_STATE.hires, prompt: "x" };
  assert.equal(ownPromptOn38(with38({ hires: off }), undefined, "hires"), true);
  assert.equal(ownPromptOn38(with38({ hires: off, sam3: { ...DEFAULT_STATE.sam3, inpaint_prompt: "" } }), undefined, "sam3"), false);
  assert.equal(ownPromptOn38(with38({ sam3: { ...DEFAULT_STATE.sam3, inpaint_prompt: "x" } }), undefined, "sam3"), true);
});

test("a weight in a pass prompt counts for MOSS's and Krea2's notice", () => {
  assert.equal(hasPromptWeights(state({ hires: hires({ prompt: "(eyes:1.2)" }) })), true);
  assert.equal(hasPromptWeights(state({ sam3: sam3({ inpaint_prompt: "(eyes:1.2)" }) })), true);
  assert.equal(hasPromptWeights(state({ hires: { ...DEFAULT_STATE.hires, prompt: "(eyes:1.2)" } })), false);
});

test("Refine draws at the image's size and samples with the SAM3 section", () => {
  const ext = { ...DEFAULT_STATE.ext, sam3_refine: { ...DEFAULT_STATE.ext.sam3_refine, on: true } };
  const refine = extInputs(state({ ext, sam3: sam3({ sampler: "heun", scheduler: "karras" }) }), "ext_sam3_refine");
  assert.deepEqual([refine.target_image_size, refine.sampler_name, refine.scheduler], [true, "heun", "karras"]);
});

test("Refine and tile read one mask per group, whatever the detailer combines", () => {
  const combined = sam3({ prompt: "face / hand", mask_mode: "Combined" });
  const mask = (ext) => inputs(run(state({ ext, sam3: combined })), "sam3_mask").mask_mode;
  assert.equal(mask(DEFAULT_STATE.ext), "Combined");
  assert.equal(mask({ ...DEFAULT_STATE.ext, sam3_refine: { ...DEFAULT_STATE.ext.sam3_refine, on: true } }), "Individual");
  assert.equal(mask({ ...DEFAULT_STATE.ext, sam3_tile: { ...DEFAULT_STATE.ext.sam3_tile, on: true } }), "Individual");
  const detailer = inputs(run(state({ ext: { ...DEFAULT_STATE.ext, sam3_tile: { ...DEFAULT_STATE.ext.sam3_tile, on: true } }, sam3: combined })), "sam3_detailer");
  assert.equal(detailer.mask_mode, "Combined");
});

test("Detail Daemon on the SAM3 pass judges the pass's own sampler", () => {
  const ext = { ...DEFAULT_STATE.ext, detail_daemon: { ...DEFAULT_STATE.ext.detail_daemon, on: true } };
  assert.equal(extModes(state({ ext, sam3: sam3() }), null).ext_dd_sam3, 0);
  assert.equal(extModes(state({ ext, sam3: sam3({ sampler: "dpm_adaptive" }) }), null).ext_dd_sam3, 4);
  assert.equal(extModes(state({ ext, sampler: "dpm_adaptive", sam3: sam3({ sampler: "euler" }) }), null).ext_dd_sam3, 0);
});
