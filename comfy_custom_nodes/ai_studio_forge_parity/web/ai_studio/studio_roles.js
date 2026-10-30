// Engine-room roles of the Anima studio workflow. Each engine node carries its role in
// `properties.aistudio_role`; the studio JS finds nodes by role, never by id or title.
// Pure ES module: no side effects (ComfyUI loads every .js under web/ as an extension).

export const ROLE_PROPERTY = "aistudio_role";

export const ROLES = Object.freeze([
  // ① models (2.9B / base) and ①b 3.8B; the switches pick one family (R6: the TE is shared, no clip switch)
  "unet", "te", "vae", "unet38", "qwen35", "family_switch_model",
  "shift", "sched",
  // ② LoRA / DoRA
  "lora_stack", "negpip", "dora",
  // ③ prompts
  "pos_encode", "neg_encode", "pos_encode38", "neg_encode38", "family_switch_pos", "family_switch_neg",
  // ④ latent and sampling
  "latent", "i2i_image", "sampler",
  // ⑤ hires ⑥ upscale ⑦ SAM3 ⑧ save
  "hires", "decode", "upscale", "sam3_mask", "sam3_detailer", "save",
  // ⑨ post-processing of a gallery image
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
