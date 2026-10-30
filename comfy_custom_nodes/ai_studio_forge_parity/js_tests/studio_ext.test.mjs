// node:test for the studio's extension logic (web/ai_studio/studio_ext.js) and how studio_logic.js uses it.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { EXT_SCHEMA } from "../web/ai_studio/ext_schema.js";
import {
  EXT_DEFAULTS,
  EXT_ROLES,
  extCnsInputs,
  extInputs,
  extModes,
  normalizeExt,
} from "../web/ai_studio/studio_ext.js";
import {
  DEFAULT_STATE,
  applyStateToPrompt,
  postPrompt,
  postRunState,
  stageModes,
  withDefaults,
} from "../web/ai_studio/studio_logic.js";
import { ROLES } from "../web/ai_studio/studio_roles.js";

const golden = JSON.parse(readFileSync(new URL("./fixtures/studio_logic_golden.json", import.meta.url), "utf-8"));

const state = (overrides = {}) => withDefaults({ seed: 123, ...overrides });
const withExt = (key, values, overrides = {}) => state({ ...overrides, ext: { [key]: { on: true, ...values } } });
const fields = (key) => EXT_SCHEMA.extensions.find((item) => item.key === key).fields;
const settings = (inputs) => JSON.parse(inputs.settings_json);

function suiteBaseline() {
  return Object.fromEntries(fields("guidance").map((field) => [field.key, field.default]));
}

// ---------------------------------------------------------------------------------------------- state

test("every extension role is a workflow role, and the other way round", () => {
  for (const role of EXT_ROLES) assert.ok(ROLES.includes(role), role);
  assert.deepEqual(ROLES.filter((role) => role.startsWith("ext_")).sort(), [...EXT_ROLES].sort());
  for (const item of EXT_SCHEMA.extensions) assert.ok(EXT_ROLES.includes(item.role), item.key);
});

test("DEFAULT_STATE.ext is the table's defaults, all off", () => {
  assert.deepEqual(DEFAULT_STATE.ext, EXT_DEFAULTS);
  assert.deepEqual(Object.keys(DEFAULT_STATE.ext), EXT_SCHEMA.extensions.map((item) => item.key));
  for (const item of EXT_SCHEMA.extensions) {
    assert.deepEqual(DEFAULT_STATE.ext[item.key], {
      on: false,
      ...Object.fromEntries(item.fields.map((field) => [field.key, field.default])),
    });
  }
});

test("withDefaults fills a saved state without ext and resets junk", () => {
  for (const raw of [{}, { ext: null }, { ext: "junk" }, { ext: [] }, { ext: { skim: 5 } }]) {
    assert.deepEqual(withDefaults(raw).ext, DEFAULT_STATE.ext);
  }
});

test("normalizeExt coerces, clamps and drops unknown keys like normalize_state", () => {
  const out = normalizeExt({
    nope: { on: true },
    skim: { on: "yes", skim_skimming_cfg: "2.5", skim_flip_at: 9, skim_full_skim_negative: 1, zzz: 1 },
    guidance: { on: true, guid_attn_method: "SEG", guid_scale: NaN, guid_adg_interval: 99.7, guid_block_indices: 18, guid_mod_start_layer: true },
    speed: { mode: "nope", preset: "flux", seed: "-5", scales: "1/2,1" },
    degrid: { model: "my.safetensors", mode: "Full (전체)" },
  });
  assert.equal(out.nope, undefined);
  assert.equal(out.skim.on, false);
  assert.equal(out.skim.skim_skimming_cfg, 2.5);
  assert.equal(out.skim.skim_flip_at, 1);
  assert.equal(out.skim.skim_full_skim_negative, false);
  assert.equal(out.skim.zzz, undefined);
  assert.equal(out.guidance.on, true);
  assert.equal(out.guidance.guid_attn_method, "SEG");
  assert.equal(out.guidance.guid_scale, 4);
  assert.equal(out.guidance.guid_adg_interval, 10);
  assert.equal(out.guidance.guid_block_indices, "18");
  assert.equal(out.guidance.guid_mod_start_layer, 0);
  assert.equal(out.speed.mode, "transition");
  assert.equal(out.speed.preset, "flux");
  assert.equal(out.speed.seed, -1);
  assert.equal(out.speed.scales, "1/2,1");
  assert.equal(out.degrid.model, "my.safetensors");
  assert.equal(out.degrid.mode, "Full");
  assert.deepEqual(normalizeExt(out), out);
});

// ---------------------------------------------------------------------------------------------- compiler parity

for (const testCase of golden.ext_compiler) {
  test(`same as the app compiler: ${testCase.name}`, () => {
    const s = withDefaults(testCase.state);
    const modes = extModes(s, null);
    const { cns, ...expected } = testCase.expected;
    assert.deepEqual(extCnsInputs(s), cns);
    for (const role of EXT_ROLES.filter((item) => ["ext_lora_block", "ext_vae_2x", "ext_pid", "ext_degrid", "ext_sam3_refine", "ext_sam3_tile"].indexOf(item) < 0)) {
      const wanted = expected[role];
      assert.equal(modes[role], wanted ? 0 : 4, role);
      if (!wanted) continue;
      const inputs = { ...extInputs(s, role) };
      if ("settings_delta" in wanted) {
        const { settings_delta: delta, ...rest } = wanted;
        const { settings_json: text, ...others } = inputs;
        assert.deepEqual(others, rest, role);
        assert.deepEqual(JSON.parse(text), { ...suiteBaseline(), ...delta }, role);
      } else if ("settings_json" in wanted) {
        assert.deepEqual({ ...inputs, settings_json: settings(inputs) }, wanted, role);
      } else {
        assert.deepEqual(inputs, wanted, role);
      }
    }
  });
}

// ---------------------------------------------------------------------------------------------- inputs

test("the suite's settings_json carries cfg_scale and sorted keys", () => {
  const inputs = extInputs(withExt("guidance", { guid_enabled: true }, { cfg: 3.5 }), "ext_guidance");
  assert.equal(inputs.enabled, true);
  const parsed = settings(inputs);
  assert.equal(parsed.cfg_scale, 3.5);
  assert.deepEqual(Object.keys(parsed), Object.keys(parsed).sort());
});

test("HiFlow runs only on a t2i generation with Hires, and never on a post run", () => {
  const values = { guid_hiflow_enabled: true };
  const hires = { ...DEFAULT_STATE.hires, on: true };
  assert.equal(settings(extInputs(withExt("guidance", values, { hires }), "ext_guidance")).guid_hiflow_enabled, true);
  assert.equal(settings(extInputs(withExt("guidance", values), "ext_guidance")).guid_hiflow_enabled, false);
  assert.equal(settings(extInputs(withExt("guidance", values, { hires, tab: "i2i" }), "ext_guidance")).guid_hiflow_enabled, false);
  assert.equal(settings(extInputs(withExt("guidance", values, { hires }), "ext_guidance", { postAction: "hires" })).guid_hiflow_enabled, false);
});

test("S² gets the generation seed only when it really runs", () => {
  const s2 = { guid_enabled: true, guid_slg_on: true, guid_slg_mode: "Stochastic (S²)" };
  assert.equal(settings(extInputs(withExt("guidance", s2), "ext_guidance")).guid_s2_seed, 123);
  const off = { ...s2, guid_slg_mode: "Fixed" };
  assert.equal("guid_s2_seed" in settings(extInputs(withExt("guidance", off), "ext_guidance")), false);
  const noPag = { ...s2, guid_enabled: false };
  assert.equal("guid_s2_seed" in settings(extInputs(withExt("guidance", noPag), "ext_guidance")), false);
});

test("RDC follows the original's rule: DCW on, the switch on and tau above 0", () => {
  const read = (values) => settings(extInputs(withExt("guidance", values), "ext_guidance"));
  const on = read({ guid_dcw_enabled: true, guid_rdc_enabled: true, guid_rdc_tau: 0.1 });
  assert.deepEqual([on.guid_rdc_enabled, on.guid_rdc_tau], [true, 0.1]);
  const noDcw = read({ guid_enabled: true, guid_rdc_enabled: true, guid_rdc_tau: 0.1 });
  assert.deepEqual([noDcw.guid_rdc_enabled, noDcw.guid_rdc_tau], [false, 0]);
  const noSwitch = read({ guid_dcw_enabled: true, guid_rdc_tau: 0.15 });
  assert.deepEqual([noSwitch.guid_rdc_enabled, noSwitch.guid_rdc_tau], [false, 0]);
});

test("a blank DAVE block list means the original's default blocks", () => {
  assert.equal(settings(extInputs(withExt("guidance", { guid_dave_blocks: "  " }), "ext_guidance")).guid_dave_blocks, "8-18");
});

test("Detail Daemon: the node's keys only, and each pass's cfg", () => {
  const s = withExt("detail_daemon", { dd_amount: 0.3 }, { cfg: 3, sam3: { ...DEFAULT_STATE.sam3, on: true, cfg: 5 } });
  const base = extInputs(s, "ext_dd");
  assert.equal(base.cfg_scale_override, 3);
  assert.deepEqual(Object.keys(settings(base)), ["dd_amount", "dd_bias", "dd_enabled", "dd_end", "dd_end_offset", "dd_exponent", "dd_fade", "dd_smooth", "dd_start", "dd_start_offset"]);
  assert.equal(settings(base).dd_enabled, true);
  assert.equal(extInputs(s, "ext_dd_hires").cfg_scale_override, 3);
  assert.equal(extInputs(s, "ext_dd_sam3").cfg_scale_override, 5);
});

test("SPEED: pass label, seeds and the card's own switches stay out of the node", () => {
  const s = withExt("speed", { hires: true, apply_img2img: true }, { tab: "i2i" });
  const inputs = extInputs(s, "ext_speed");
  assert.equal(inputs.pass_label, "img2img");
  assert.equal(inputs.image_seed, 123);
  assert.equal(inputs.seed, -1);
  assert.equal(inputs.img2img_rescale, true);
  assert.equal("hires" in inputs, false);
  assert.equal("apply_img2img" in inputs, false);
  assert.equal(extInputs(s, "ext_speed_hires").pass_label, "hires");
  assert.equal(extInputs(state(), "ext_speed").pass_label, "base");
});

test("widget extensions write their node input names and fixed values", () => {
  assert.deepEqual(extInputs(withExt("degrid", { strength: 0.5, tile: 0 }), "ext_degrid"), {
    enabled: true, device: "auto", precision: "fp32", keep_loaded: false, forge_quantize: true,
    model_name: "auto", mode: "Full", strength: 0.5, tile: 0,
  });
  const block = extInputs(withExt("lora_block", { lora_name: "a.safetensors", A: 2 }), "ext_lora_block");
  assert.equal(block.lora_name, "a.safetensors");
  assert.equal(block.A, 2);
  assert.equal(block.seed, 123);
  assert.equal(block.enabled, true);
  const pid = extInputs(withExt("pid", { ckpt_name: "pid.safetensors" }), "ext_pid");
  assert.equal(pid.ckpt_name, "pid.safetensors");
  assert.deepEqual([pid.dtype, pid.tile_latent, pid.tile_overlap, pid.compile, pid.use_calib, pid.attention], ["bf16", 64, 16, false, false, "auto"]);
});

test("SAM3 Refine and tile repair sample with the SAM3 section's values", () => {
  const sam3 = { ...DEFAULT_STATE.sam3, steps: 20, cfg: 4 };
  const refine = extInputs(withExt("sam3_refine", { denoise: 0.3, mask_padding: 16 }, { sam3, sampler: "euler", scheduler: "simple" }), "ext_sam3_refine");
  assert.deepEqual([refine.steps, refine.cfg, refine.sampler_name, refine.scheduler, refine.seed, refine.denoise, refine.mask_padding, refine.enabled], [20, 4, "euler", "simple", 123, 0.3, 16, true]);
  const tile = extInputs(withExt("sam3_tile", { tile_width: 768 }, { sam3: DEFAULT_STATE.sam3, steps: 30, cfg: 3 }), "ext_sam3_tile");
  assert.equal(tile.enabled, true);
  const parsed = settings(tile);
  assert.deepEqual([parsed.tile_width, parsed.steps, parsed.cfg, parsed.seed], [768, 30, 3, 123]);
  assert.equal(parsed.sampler_name, DEFAULT_STATE.sampler);
});

test("an unknown role has no inputs", () => {
  assert.deepEqual(extInputs(state(), "sampler"), {});
});

// ---------------------------------------------------------------------------------------------- modes

test("every extension role is bypassed by default", () => {
  const modes = extModes(state(), null);
  assert.deepEqual(Object.keys(modes).sort(), [...EXT_ROLES].sort());
  assert.ok(Object.values(modes).every((mode) => mode === 4));
});

test("extModes only names the roles the graph has", () => {
  const modes = extModes(withExt("skim", {}), { ext_skim: "5", sampler: "9" });
  assert.deepEqual(modes, { ext_skim: 0 });
});

test("the suite stays out until something in it is on", () => {
  assert.equal(extModes(withExt("guidance", {}), null).ext_guidance, 4);
  assert.equal(extModes(withExt("guidance", { guid_cfg_mode: "APG" }), null).ext_guidance, 4);
  assert.equal(extModes(withExt("guidance", { guid_tsr_enabled: true }), null).ext_guidance, 0);
});

test("Detail Daemon picks one main pass and skips what the compiler skips", () => {
  const hires = { ...DEFAULT_STATE.hires, on: true };
  const sam3 = { ...DEFAULT_STATE.sam3, on: true };
  const base = extModes(withExt("detail_daemon", {}, { hires, sam3 }), null);
  assert.deepEqual([base.ext_dd, base.ext_dd_hires, base.ext_dd_sam3], [0, 4, 0]);
  const late = extModes(withExt("detail_daemon", { dd_hires: true }, { hires, sam3 }), null);
  assert.deepEqual([late.ext_dd, late.ext_dd_hires, late.ext_dd_sam3], [4, 0, 4]);
  const noHires = extModes(withExt("detail_daemon", { dd_hires: true }), null);
  assert.deepEqual([noHires.ext_dd, noHires.ext_dd_hires], [4, 4]);
  for (const sampler of ["dpm_adaptive", "heunpp2"]) assert.equal(extModes(withExt("detail_daemon", {}, { sampler }), null).ext_dd, 4);
  assert.equal(extModes(withExt("detail_daemon", {}, { cfg: 0 }), null).ext_dd, 4);
  const sam3Cfg = extModes(withExt("detail_daemon", {}, { cfg: 0, sam3: { ...sam3, cfg: 4 } }), null);
  assert.deepEqual([sam3Cfg.ext_dd, sam3Cfg.ext_dd_sam3], [4, 0]);
});

test("SPEED: main passes only, i2i with its switch, never on inpaint", () => {
  const hires = { ...DEFAULT_STATE.hires, on: true };
  const t2i = extModes(withExt("speed", {}, { hires }), null);
  assert.deepEqual([t2i.ext_speed, t2i.ext_speed_hires], [0, 4]);
  const both = extModes(withExt("speed", { hires: true }, { hires }), null);
  assert.deepEqual([both.ext_speed, both.ext_speed_hires], [0, 0]);
  assert.equal(extModes(withExt("speed", {}, { tab: "i2i" }), null).ext_speed, 4);
  assert.equal(extModes(withExt("speed", { apply_img2img: true }, { tab: "i2i" }), null).ext_speed, 0);
  assert.equal(extModes(withExt("speed", { apply_img2img: true }, { tab: "i2i", i2i_mode: "inpaint" }), null).ext_speed, 4);
});

test("DeGrid follows the i2i switch", () => {
  assert.equal(extModes(withExt("degrid", {}), null).ext_degrid, 0);
  assert.equal(extModes(withExt("degrid", {}, { tab: "i2i" }), null).ext_degrid, 4);
  assert.equal(extModes(withExt("degrid", { apply_img2img: true }, { tab: "i2i" }), null).ext_degrid, 0);
});

test("an extension that needs a file stays out until one is picked", () => {
  assert.equal(extModes(withExt("lora_block", {}), null).ext_lora_block, 4);
  assert.equal(extModes(withExt("lora_block", { lora_name: "a.safetensors" }), null).ext_lora_block, 0);
  assert.equal(extModes(withExt("vae_2x", {}), null).ext_vae_2x, 4);
  assert.equal(extModes(withExt("vae_2x", { vae_name: "None" }), null).ext_vae_2x, 4);
  assert.equal(extModes(withExt("vae_2x", { vae_name: "2x.safetensors" }), null).ext_vae_2x, 0);
  assert.equal(extModes(withExt("pid", {}), null).ext_pid, 4);
  assert.equal(extModes(withExt("pid", { ckpt_name: "(auto-download)" }), null).ext_pid, 4);
  assert.equal(extModes(withExt("pid", { ckpt_name: "pid.safetensors" }), null).ext_pid, 0);
});

test("a post run decides the passes by its action, not the state's tab", () => {
  const s = withExt("speed", { hires: true }, { tab: "i2i" });
  assert.equal(extModes(s, null, { postAction: "hires" }).ext_speed_hires, 0);
  assert.equal(extModes(s, null).ext_speed_hires, 4);
  const dd = withExt("detail_daemon", {});
  assert.equal(extModes(dd, null, { postAction: "sam3" }).ext_dd_sam3, 0);
  assert.equal(extModes(dd, null, { postAction: "hires" }).ext_dd_sam3, 4);
});

// ---------------------------------------------------------------------------------------------- studio_logic

const ROLE_INPUTS = {
  ext_er_sde: { model: ["1", 0], max_stage: 3, eta: 1 },
  ext_guidance: { model: ["1", 0], enabled: false, settings_json: "{}" },
  ext_skim: { model: ["1", 0], enabled: false, skimming_cfg: 7 },
  ext_dd: { model: ["1", 0], enabled: false, settings_json: "{}", cfg_scale_override: 0 },
  ext_dd_sam3: { model: ["1", 0], enabled: false, settings_json: "{}", cfg_scale_override: 0 },
  ext_speed_hires: { model: ["1", 0], enabled: false, pass_label: "base", image_seed: 0 },
  sam3_mask: { enabled: false, prompt: "x" },
  sampler: { cns_enabled: false, cns_strength: 0.5, seed: 0 },
  hires: { cns_enabled: false, seed: 0, enabled: false },
  post_image: { image: "x.png" },
  post_upscale: { enabled: false },
  post_hires: { cns_enabled: false, seed: 0, enabled: false },
  post_sam3_mask: { enabled: false },
  post_sam3_detailer: { enabled: false },
  post_save: { images: ["1", 0] },
  save: { studio_state: "{}", seed: 0 },
};

function prompt(roles = Object.keys(ROLE_INPUTS)) {
  const roleIds = Object.fromEntries(roles.map((role, index) => [role, String(index + 10)]));
  const api = Object.fromEntries(roles.map((role) => [roleIds[role], { class_type: role, inputs: structuredClone(ROLE_INPUTS[role]) }]));
  return { api, roleIds };
}

test("applyStateToPrompt writes extension inputs and keeps their links", () => {
  const { api, roleIds } = prompt();
  const s = state({ ext: { skim: { on: true, skim_skimming_cfg: 3 }, guidance: { on: true, guid_enabled: true } } });
  const out = applyStateToPrompt(api, roleIds, s);
  const skim = out[roleIds.ext_skim].inputs;
  assert.deepEqual(skim.model, ["1", 0]);
  assert.equal(skim.enabled, true);
  assert.equal(skim.skimming_cfg, 3);
  assert.equal(JSON.parse(out[roleIds.ext_guidance].inputs.settings_json).guid_enabled, true);
});

test("an older workflow without extension roles still applies", () => {
  const { api, roleIds } = prompt(["sampler", "save"]);
  const out = applyStateToPrompt(api, roleIds, withExt("speed", {}));
  assert.equal(out[roleIds.sampler].inputs.seed, 123);
  assert.deepEqual(Object.keys(out).sort(), Object.keys(api).sort());
});

test("CNS goes to the sampler and Hires, not only the suite", () => {
  const { api, roleIds } = prompt();
  const on = { guid_cns_enabled: true, guid_cns_strength: 0.7, guid_cns_gamma_power: 1, guid_cns_gamma_scale: 3 };
  const out = applyStateToPrompt(api, roleIds, withExt("guidance", on));
  for (const role of ["sampler", "hires"]) {
    const inputs = out[roleIds[role]].inputs;
    assert.deepEqual([inputs.cns_enabled, inputs.cns_strength, inputs.cns_gamma_power, inputs.cns_gamma_scale], [true, 0.7, 1, 3], role);
  }
  const off = applyStateToPrompt(api, roleIds, withExt("guidance", { guid_cns_enabled: false, guid_tsr_enabled: true }));
  assert.equal(off[roleIds.sampler].inputs.cns_enabled, false);
  const suiteOff = applyStateToPrompt(api, roleIds, state({ ext: { guidance: { on: false, guid_cns_enabled: true } } }));
  assert.equal(suiteOff[roleIds.sampler].inputs.cns_enabled, false);
});

test("stageModes bypasses switched-off extensions and runs switched-on ones", () => {
  const off = stageModes(state());
  for (const role of EXT_ROLES) assert.equal(off[role], 4, role);
  const on = stageModes(withExt("skim", {}));
  assert.equal(on.ext_skim, 0);
  assert.equal(on.ext_guidance, 4);
  assert.deepEqual(Object.keys(on).sort(), [...ROLES].sort());
});

test("Refine and tile repair keep the SAM3 mask running without the SAM3 detailer", () => {
  const modes = stageModes(withExt("sam3_refine", {}));
  assert.deepEqual([modes.ext_sam3_refine, modes.sam3_mask, modes.sam3_detailer], [0, 0, 4]);
  const { api, roleIds } = prompt();
  assert.equal(applyStateToPrompt(api, roleIds, withExt("sam3_tile", {}))[roleIds.sam3_mask].inputs.enabled, true);
  assert.equal(applyStateToPrompt(api, roleIds, state())[roleIds.sam3_mask].inputs.enabled, false);
});

test("a post run keeps the extensions of its record and turns HiFlow off", () => {
  const { api, roleIds } = prompt();
  const ext = { guidance: { on: true, guid_hiflow_enabled: true, guid_tsr_enabled: true }, speed: { on: true, hires: true } };
  const record = { ...state({ ext, hires: { ...DEFAULT_STATE.hires, on: true } }), image: "a.png" };
  const { prompt: out } = postPrompt(api, roleIds, record, "hires", state());
  assert.equal(JSON.parse(out[roleIds.ext_guidance].inputs.settings_json).guid_hiflow_enabled, false);
  assert.equal(out[roleIds.ext_speed_hires].inputs.pass_label, "hires");
  const runState = postRunState(record, state(), "hires");
  assert.equal(runState.ext.speed.on, true);
  assert.equal(stageModes(runState, "hires").ext_speed_hires, 0);
});

test("a record without ext uses the current panel's extensions", () => {
  const current = withExt("skim", {});
  const merged = postRunState({ prompt: "x", image: "a.png" }, current, "upscale");
  assert.equal(merged.ext.skim.on, true);
});
