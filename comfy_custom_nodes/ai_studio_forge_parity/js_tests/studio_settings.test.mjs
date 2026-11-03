// node:test for the EXIF panel's 설정 가져오기 (web/ai_studio/studio_settings.js).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { withDefaults } from "../web/ai_studio/studio_logic.js";
import { imageSettings, pulledState } from "../web/ai_studio/studio_settings.js";

const workflow = (nodes) => ({ key: "workflow", text: JSON.stringify({ nodes }) });
const node = (type, value) => ({ type, widgets_values: [JSON.stringify(value)] });
const LINE = "1girl\nSteps: 20, Seed: 5";

test("a studio image gives the panel and the XYZ panel of its run", () => {
  const panel = { prompt: "1girl", seed: -1 };
  const xyz = { on: true, axes: [] };
  const entries = [workflow([{ type: "KSampler", widgets_values: [1] }, node("AIStudioPanel", panel), node("AIStudioXYZView", xyz)])];
  assert.deepEqual(imageSettings(entries), { panel, xyz });
});

test("the panel wins over the infotext: it has what the text cannot say", () => {
  const entries = [{ key: "parameters", text: LINE }, workflow([node("AIStudioPanel", { prompt: "1girl" })])];
  assert.deepEqual(imageSettings(entries), { panel: { prompt: "1girl" }, xyz: null });
});

test("an image of another workflow with an infotext gives the text", () => {
  const entries = [{ key: "parameters", text: LINE }, workflow([{ type: "KSampler", widgets_values: [1] }])];
  assert.deepEqual(imageSettings(entries), { infotext: LINE });
});

test("Forge's JPEG and WebP keep the infotext in UserComment", () => {
  assert.deepEqual(imageSettings([{ key: "UserComment", text: LINE }]), { infotext: LINE });
});

test("nothing to pull: no text, other text, a broken workflow, an empty panel", () => {
  assert.equal(imageSettings([]), null);
  assert.equal(imageSettings([{ key: "parameters", text: "a comment" }]), null);
  assert.equal(imageSettings([{ key: "Software", text: LINE }]), null);
  assert.equal(imageSettings([{ key: "workflow", text: "{not json" }]), null);
  assert.equal(imageSettings([workflow([node("AIStudioPanel", {})])]), null);
  assert.equal(imageSettings([workflow([{ type: "AIStudioPanel", widgets_values: ["{not json"] }])]), null);
  assert.equal(imageSettings([workflow([{ type: "AIStudioPanel" }])]), null);
  assert.equal(imageSettings([{ key: "workflow", text: "null" }]), null);
});

const LISTS = {
  unets: ["Anima-2.9B.safetensors", "MOSS-Image.safetensors"],
  tes: ["qwen_0.6b.safetensors", "qwen3.5_9b.safetensors", "te_b.safetensors"],
  vaes: ["qwen_image_vae.safetensors", "moss_image_vae.safetensors"],
  samplers: ["euler", "er_sde", "exp_heun_2_x0_sde"],
  schedulers: ["simple", "beta57", "flow_cosmos_dynamic", "sgm_uniform"],
  upscalers: ["latent:bicubic", "sub/4x-UltraSharp.pth"],
  loras: ["meow.safetensors"],
};
const current = (changes = {}) => withDefaults({ unet: LISTS.unets[0], te: LISTS.tes[0], vae: LISTS.vaes[0], ...changes });

test("a panel comes back whole: seed, boxes, batch, sections", () => {
  const saved = {
    unet: LISTS.unets[0], te: LISTS.tes[0], vae: LISTS.vaes[0], seed: -1, batch: 4, prompt: "x", prompt_artist: "@a", steps: 24,
    hires: { on: true, scale: 2 }, ext: { guidance: { on: true } },
  };
  const { state, missing } = pulledState(current({ seed: 77, prompt: "old" }), { panel: saved }, LISTS);
  assert.deepEqual(missing, []);
  assert.equal(state.seed, -1);
  assert.equal(state.batch, 4);
  assert.equal(state.prompt, "x");
  assert.equal(state.prompt_artist, "@a");
  assert.equal(state.steps, 24);
  assert.equal(state.hires.scale, 2);
  assert.equal(state.hires.upscaler, "4x-UltraSharp.pth"); // an older panel's missing keys are the defaults
  assert.equal(state.ext.guidance.on, true);
});

test("the tab and the i2i image stay this panel's: they are where the work is, not settings", () => {
  const here = current({ tab: "i2i", i2i_mode: "inpaint" });
  here.i2i.image = "mine.png";
  const { state } = pulledState(here, { panel: { tab: "t2i", i2i_mode: "img2img", i2i: { image: "theirs.png", denoise: 0.3 } } }, LISTS);
  assert.equal(state.tab, "i2i");
  assert.equal(state.i2i_mode, "inpaint");
  assert.equal(state.i2i.image, "mine.png");
  assert.equal(state.i2i.denoise, 0.3);
});

test("a file this PC does not have keeps the panel's pick and is reported", () => {
  const saved = { unet: "gone.safetensors", te: LISTS.tes[1], vae: "gone_vae.safetensors", sampler: "gone_sampler", scheduler: "beta57", prompt: "<lora:meow:1> <lora:gone:1>" };
  const { state, missing } = pulledState(current(), { panel: saved }, LISTS);
  assert.equal(state.unet, LISTS.unets[0]);
  assert.equal(state.te, LISTS.tes[1]);
  assert.equal(state.vae, LISTS.vaes[0]);
  assert.equal(state.sampler, withDefaults({}).sampler);
  assert.equal(state.scheduler, "beta57");
  assert.deepEqual(missing, ["gone.safetensors", "gone_vae.safetensors", "gone_sampler", "gone"]);
});

test("lists not read yet check nothing", () => {
  const { state, missing } = pulledState(current(), { panel: { unet: "any.safetensors", prompt: "<lora:x:1>" } }, {});
  assert.equal(state.unet, "any.safetensors");
  assert.deepEqual(missing, []);
});

const infotext = (settings, extra = {}) => ({ settings: { hires: { on: false }, ...settings }, model: "", modules: [], ...extra });

test("an infotext changes what it says and leaves the rest of the panel", () => {
  const here = current({ seed: 77, prompt_artist: "@a", loras: [{ on: true, name: "meow", strength: 1 }], batch: 3 });
  here.sam3.on = true;
  const read = infotext({ prompt: "x <lora:meow:0.8>", negative: "y", steps: 20, sampler: "er_sde", scheduler: "beta57", cfg: 5.5, width: 1024, height: 768, loras: [], prompt_artist: "" });
  const { state, missing } = pulledState(here, read, LISTS);
  assert.deepEqual(missing, []);
  assert.equal(state.prompt, "x <lora:meow:0.8>");
  assert.equal(state.prompt_artist, "");
  assert.deepEqual(state.loras, []);
  assert.equal(state.sampler, "er_sde");
  assert.equal(state.seed, 77);
  assert.equal(state.batch, 3);
  assert.equal(state.sam3.on, true);
  assert.equal(here.prompt_artist, "@a"); // the panel's own object is left alone
});

test("an infotext's sections merge into the panel's", () => {
  const here = current();
  const { state } = pulledState(here, infotext({ i2i: { denoise: 0.3 }, hires: { on: true, scale: 2 } }), LISTS);
  assert.equal(state.i2i.denoise, 0.3);
  assert.equal(state.i2i.mask_blur, here.i2i.mask_blur);
  assert.equal(state.hires.on, true);
  assert.equal(state.hires.steps, here.hires.steps);
});

test("the infotext's Model finds its file and brings that family's picks, its own numbers on top", () => {
  const { state, missing } = pulledState(current(), infotext({ steps: 30 }, { model: "MOSS-Image" }), LISTS);
  assert.deepEqual(missing, []);
  assert.equal(state.unet, "MOSS-Image.safetensors");
  assert.equal(state.te, "qwen3.5_9b.safetensors");
  assert.equal(state.vae, "moss_image_vae.safetensors");
  assert.equal(state.sampler, "euler"); // MOSS's default
  assert.equal(state.steps, 30);
});

test("a Model this PC does not have is reported and the panel's stays", () => {
  const { state, missing } = pulledState(current(), infotext({}, { model: "gone" }), LISTS);
  assert.equal(state.unet, LISTS.unets[0]);
  assert.deepEqual(missing, ["gone"]);
});

test("Forge's modules are found in the VAE list, then the TE list", () => {
  const { state, missing } = pulledState(current(), infotext({}, { modules: ["moss_image_vae", "te_b", "gone"] }), LISTS);
  assert.equal(state.vae, "moss_image_vae.safetensors");
  assert.equal(state.te, "te_b.safetensors");
  assert.deepEqual(missing, ["gone"]);
});

test("hires files are found by name, a missing one keeps the panel's", () => {
  const here = current();
  here.hires.upscaler = "latent:bicubic";
  const found = pulledState(here, infotext({ hires: { on: true, upscaler: "4x-UltraSharp", vae: "moss_image_vae" } }), LISTS);
  assert.equal(found.state.hires.upscaler, "sub/4x-UltraSharp.pth");
  assert.equal(found.state.hires.vae, "moss_image_vae.safetensors");
  const lost = pulledState(here, infotext({ hires: { on: true, upscaler: "gone", vae: "" } }), LISTS);
  assert.equal(lost.state.hires.upscaler, "latent:bicubic");
  assert.equal(lost.state.hires.vae, "");
  assert.deepEqual(lost.missing, ["gone"]);
});

test("a latent hires upscaler matches as it is", () => {
  const { state } = pulledState(current(), infotext({ hires: { on: true, upscaler: "latent:bicubic" } }), LISTS);
  assert.equal(state.hires.upscaler, "latent:bicubic");
});

test("each missing name is said once", () => {
  const { missing } = pulledState(current(), { panel: { unet: "gone", vae: "gone" } }, LISTS);
  assert.deepEqual(missing, ["gone"]);
});
