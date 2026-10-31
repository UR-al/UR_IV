// Engine-room roles of the Anima studio workflow. Each engine node carries its role in
// `properties.aistudio_role`; the studio JS finds nodes by role, never by id or title.
// Pure ES module: no side effects (ComfyUI loads every .js under web/ as an extension).

export const ROLE_PROPERTY = "aistudio_role";

export const ROLES = Object.freeze([
  // ① models (2.9B / base) and ①b 3.8B; the switches pick one family (R6: the TE is shared, no clip switch)
  "unet", "te", "vae", "unet38", "qwen35", "family_switch_model", "moss_stream",
  "shift", "sched",
  // ② LoRA / DoRA
  "lora_stack", "ext_lora_block", "negpip", "dora", "ext_er_sde",
  // ③ prompts
  "pos_encode", "neg_encode", "pos_encode38", "neg_encode38", "family_switch_pos", "family_switch_neg",
  // ③b the reference image and the node of the family that uses it
  "reference_image", "anima_ref", "krea2_ref",
  // ④ extensions: the guidance stack, then each pass's Detail Daemon and SPEED
  "ext_optimal_scale", "ext_guidance", "ext_skim", "ext_dd", "ext_speed", "ext_dd_hires", "ext_speed_hires", "ext_dd_sam3",
  // ⑤ latent and sampling
  "latent", "i2i_image", "sampler",
  // ⑥ hires ⑦ upscale ⑧ SAM3 ⑨ save
  "hires", "ext_vae_2x", "decode", "ext_pid", "upscale", "sam3_mask", "sam3_detailer", "ext_sam3_refine", "ext_sam3_tile",
  "ext_degrid", "save",
  // ⑩ post-processing of a gallery image
  "post_image", "post_upscale", "post_encode", "post_hires", "post_decode", "post_sam3_mask", "post_sam3_detailer",
  "post_save",
]);

const KNOWN = new Set(ROLES);

/** `[{id, properties}]` (graph nodes or workflow JSON nodes) -> `{role: "id"}`; unknown roles are ignored. */
export function roleIdsFromNodes(nodes) {
  const ids = {};
  for (const node of nodes || []) {
    const role = node?.properties?.[ROLE_PROPERTY];
    if (KNOWN.has(role)) ids[role] = String(node.id);
  }
  return ids;
}
