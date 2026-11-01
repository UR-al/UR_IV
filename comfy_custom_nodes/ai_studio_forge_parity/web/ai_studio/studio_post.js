// Anima studio: pure logic of the post-processing field table (post_schema.js, from studio_post_schema.py).
// No `window`, no ComfyUI imports, no side effects (ComfyUI also loads this file as an extension).
import { POST_SCHEMA } from "./post_schema.js";

const SCALE_DIGIT = /\d(?=[xX])|(?<=[xX])\d/;
const LATENT_GRID = 8;

/** `{key: default}` of one section of the table. */
export function postDefaults(section) {
  return Object.fromEntries(POST_SCHEMA[section].map((field) => [field.key, field.default]));
}

/** The node inputs of a section; `node` picks a SAM3 node's. A pick left empty stays out: the workflow's value stands. */
export function postInputs(section, values, node = null) {
  const fields = POST_SCHEMA[section].filter((field) => field.input && (!node || field.node === node || field.node === "both"));
  return Object.fromEntries(fields.filter((field) => !field.list || values[field.key]).map((field) => [field.input, values[field.key]]));
}

/** Forge `sRound`: the nearest multiple of `step`, halves up (studio_upscale.snap). */
const snap = (value, step) => Math.floor(value / step + 0.5) * step;

/** The size Hires ends at (studio_upscale.hires_size), for a first pass of `width` x `height`; `grid`: the latent's cell in pixels. */
export function hiresSize(width, height, hires, grid = LATENT_GRID) {
  const { resize_width: toWidth, resize_height: toHeight, size_step: step } = hires;
  let target = [width * hires.scale, height * hires.scale];
  if (toWidth && toHeight) target = [toWidth, toHeight];
  else if (toWidth) target = [toWidth, toWidth * (height / width)];
  else if (toHeight) target = [toHeight * (width / height), toHeight];
  const side = (value) => Math.max(grid, Math.floor(Math.max(step, snap(value, step)) / grid) * grid);
  return { width: side(target[0]), height: side(target[1]) };
}

/** The factor in an upscaler's file name (`4x-UltraSharp` -> 4), null when it has none. */
export function scaleFromName(name) {
  const match = SCALE_DIGIT.exec(String(name));
  return match ? Number(match[0]) : null;
}
