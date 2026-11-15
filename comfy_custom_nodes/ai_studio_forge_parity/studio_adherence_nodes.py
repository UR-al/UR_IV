"""Prompt adherence nodes of the Anima studio (pack 1.17.0): LoRA control, CFG schedule, word emphasis.

The rules are in ``studio_adherence``; this module is the ComfyUI side. ComfyUI and torch are imported inside the
functions so the app can import the node classes.

- ``AIStudioLoraControl`` sits after the LoRA nodes. It notes which patch entries the LoRAs added (against the MODEL
  before them), drops those of Anima's text path when asked, and, for "Base 먼저", leaves the split for
  ``ForgeNeoKSamplerCNS`` under ``model_options[LORA_SCHEDULE_KEY]``.
- Base 먼저 (``split_lora_sample``) runs two ComfyUI sampling runs, the first with the base MODEL and the second with
  the LoRAs, so ComfyUI loads and patches each MODEL the usual way. Both runs get the WHOLE schedule: what plans over
  the run (Detail Daemon, DAVE, S², CFG 스케줄, 단어 주목, the preview) sees one run. Only the innermost sampler
  function is cut: the first run samples steps ``0..split`` and keeps the latent, the second picks it up at ``split``
  with the step offset (``sam_extra_step_offset``, the Dy samplers) and the noise sampler of the first run, so the step
  noise goes on as one stream. Samplers that cannot be cut there (Restart, UniPC, DPM fast/adaptive; SPEED's list) are
  refused, as are SPEED and Spectrum.
- ``AIStudioCFGSchedule`` changes the guider's CFG per step (a ``PREDICT_NOISE`` wrapper): everything downstream (APG,
  Skimmed CFG, Optimal Scale, CFG 1 skipping the negative) sees the scheduled value as the run's CFG.
- ``AIStudioWordEmphasis`` adds ``log(multiplier)`` to the cross-attention logits of the listed words' tokens: the
  token positions come from each positive cond's T5 ids (an ``extra_conds`` object patch, the way ComfyUI-ppm's NegPiP
  reads its mask), the step window from an ``APPLY_MODEL`` wrapper, the cross-attention calls are marked by an
  ``attn2_patch`` (Cosmos calls it for cross-attention only, right before that call's attention) and an
  ``optimized_attention_override`` runs those calls through PyTorch SDPA with the offsets as an additive mask (every
  other call goes to the override or attention function that was there before).
"""
from __future__ import annotations

import importlib
import logging
import uuid
from typing import Any

from .compat import clone_model, require_torch
from .guidance_common import STEP_OFFSET_KWARG, declares_keyword, republished_from, sampler_base
from .studio_adherence import (
    CFG_CURVES,
    CROSS_ATTN_MODES,
    CROSS_ATTN_OFF,
    LORA_SCHEDULE_KEY,
    base_steps,
    drop_cross_attn,
    emphasis_logit,
    emphasis_phrases,
    find_positions,
    in_window,
    lora_spans,
    scheduled_cfg,
    step_progress,
    without_spans,
)

CATEGORY = "AI Studio/Studio"
LOGGER = logging.getLogger("ai_studio_forge_parity")
LOG_PREFIX = "[AI Studio]"

CFG_WRAPPER_KEY = "ai_studio_forge_parity.cfg_schedule"
EMPHASIS_WRAPPER_KEY = "ai_studio_forge_parity.word_emphasis"
EMPHASIS_COND_KEY = "ai_studio_emphasis_bias"  # model_conds key: [1, context length] logit offsets of one cond
EMPHASIS_OPTION_KEY = "ai_studio_emphasis"  # transformer_options key while a call is in the window
CROSS_PENDING = "cross_pending"  # set by the attn2 patch, taken by the next attention call

SPLIT_REFUSED = "LoRA Base 먼저 는 {what} 과 같이 못 씁니다: 둘 중 하나를 끄세요 (Base 0 %)."


def _wrapper_types():
    return importlib.import_module("comfy.patcher_extension").WrappersMP


def run_sigmas(transformer_options: dict) -> Any:
    """The run's own schedule: ``sample_sigmas``, or the list a re-published copy (SPEED) stands in for."""
    sigmas = (transformer_options or {}).get("sample_sigmas")
    origin = republished_from(sigmas)
    return origin if origin is not None else sigmas


def run_progress(sigma: Any, transformer_options: dict) -> float:
    return step_progress(sigma, run_sigmas(transformer_options))


# ---------------------------------------------------------------------------------------------- LoRA control

class AIStudioLoraControl:
    """What the LoRAs may touch and when: Base 먼저 (main sampler) and 교차 어텐션 제외 (every pass)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "base_model": ("MODEL", {"tooltip": "The MODEL before the LoRA nodes: what it lacks are the LoRA patches."}),
            "base_percent": ("INT", {"default": 0, "min": 0, "max": 100, "step": 5, "tooltip": (
                "The first N % of the main sampler's steps run without the LoRAs (Hires, SAM3 and post passes keep them).")}),
            "cross_attn": (list(CROSS_ATTN_MODES), {"tooltip": (
                "The LoRAs leave Anima's text path (cross-attention and the LLM adapter; +변조 also the cross-attention "
                "adaLN modulation) as the base model has it.")}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "LoRA Base 먼저 (the main sampler's first steps without the LoRAs) and LoRA 교차 어텐션 제외."

    def apply(self, model, base_model, base_percent=0, cross_attn=CROSS_ATTN_OFF):
        if cross_attn not in CROSS_ATTN_MODES:
            raise ValueError(f"cross_attn must be one of {', '.join(CROSS_ATTN_MODES)}: {cross_attn!r}")
        percent = int(base_percent)
        if percent <= 0 and cross_attn == CROSS_ATTN_OFF:
            return (model,)
        patched = clone_model(model, "LoRA control")
        spans = lora_spans(getattr(base_model, "patches", None) or {}, patched.patches)
        if cross_attn != CROSS_ATTN_OFF:
            patches, spans, keys = drop_cross_attn(patched.patches, spans, cross_attn)
            if keys:
                patched.patches.clear()
                patched.patches.update(patches)
                patched.patches_uuid = uuid.uuid4()  # ComfyUI re-merges only when this changes
            LOGGER.info("%s LoRA 교차 어텐션 제외 (%s): %d keys keep the base weights", LOG_PREFIX, cross_attn, keys)
        if percent > 0:
            if spans:
                options = dict(patched.model_options)
                options[LORA_SCHEDULE_KEY] = {"base_percent": percent, "spans": dict(spans)}
                patched.model_options = options
                LOGGER.info("%s LoRA Base 먼저: %d %% of the main steps without %d LoRA keys", LOG_PREFIX, percent, len(spans))
            else:
                LOGGER.info("%s LoRA Base 먼저: no LoRA patches on this MODEL, nothing to split", LOG_PREFIX)
        return (patched,)


def lora_schedule(model: Any) -> dict | None:
    """The split the LoRA control node left on ``model`` (``LORA_SCHEDULE_KEY``), or None."""
    schedule = (getattr(model, "model_options", None) or {}).get(LORA_SCHEDULE_KEY)
    return schedule if isinstance(schedule, dict) and schedule.get("spans") else None


def planned_split(model: Any, steps, latent: dict, *, warn: bool = True) -> tuple[dict | None, int]:
    """(schedule, base steps) of a main-sampler run: 0 = no split (none asked, rounded to nothing, a masked latent —
    the second run would blend the kept area from the first run's latent, not from the image)."""
    schedule = lora_schedule(model)
    if schedule is None:
        return None, 0
    split = base_steps(schedule["base_percent"], steps)
    if split and (latent or {}).get("noise_mask") is not None:
        if warn:
            LOGGER.warning("%s LoRA Base 먼저: not with a masked latent, sampled with the LoRAs throughout", LOG_PREFIX)
        split = 0
    return schedule, split


def base_model_of(model: Any, schedule: dict) -> Any:
    """A clone of ``model`` without the LoRA patch entries ``schedule`` names (the base pass of a split run)."""
    base = clone_model(model, "LoRA Base 먼저")
    patches = without_spans(base.patches, schedule["spans"])
    base.patches.clear()
    base.patches.update(patches)
    base.patches_uuid = uuid.uuid4()
    options = dict(base.model_options)
    options.pop(LORA_SCHEDULE_KEY, None)
    base.model_options = options
    return base


def has_wrapper(model: Any, key: str) -> bool:
    """Does ``model`` carry a wrapper under ``key`` (any wrapper type)?"""
    wrappers = getattr(model, "wrappers", None) or {}
    return any(key in (by_key or {}) for by_key in wrappers.values())


def _kd_module():
    return importlib.import_module("comfy.k_diffusion.sampling")


def _live(function):
    """The function to call now: a k-diffusion ``sample_*`` is looked up by name, so a wrapper that swaps the module's
    functions while it runs (CNS path B) reaches it, as it reaches SPEED's segments."""
    name = getattr(function, "__name__", "")
    kd = _kd_module()
    found = getattr(kd, name, None) if name.startswith("sample_") else None
    return found if callable(found) else function


def _run_noise_sampler(function, x, sigmas, extra_args) -> Any:
    """The noise sampler a plain run of ``function`` over ``sigmas`` would make itself: ComfyUI's seeded default stream,
    or the Brownian tree over the whole schedule for the samplers that use one (SPEED's ``BROWNIAN_SAMPLERS``)."""
    from .speed_nodes import BROWNIAN_SAMPLERS

    kd = _kd_module()
    seed = (extra_args or {}).get("seed")
    cpu = BROWNIAN_SAMPLERS.get(getattr(sampler_base(function), "__name__", ""))
    if cpu is not None:
        positive = sigmas[sigmas > 0]
        return kd.BrownianTreeNoiseSampler(x, positive.min(), sigmas.max(), seed=seed, cpu=cpu)
    return kd.default_noise_sampler(x, seed=seed)


def split_sampler_functions(function, split: int, total: int):
    """(first, second) stand-ins for the KSAMPLER's ``sampler_function`` of a split run.

    Both are called with the whole schedule (wrappers outside them, Detail Daemon and CNS, see one run). ``first``
    samples ``sigmas[:split + 1]`` and keeps the latent and the noise sampler; ``second`` ignores the ``x`` it is given,
    samples ``sigmas[split:]`` from the kept latent with the kept noise sampler, the step offset (when ``function``
    takes it) and the preview steps counted from ``split``."""
    state: dict[str, Any] = {}

    def _check(sigmas):
        if len(sigmas) - 1 != total:
            raise RuntimeError(f"LoRA Base 먼저: the sampler got {len(sigmas) - 1} steps where the split planned {total}")

    def first(model, x, sigmas, *args, **kwargs):
        _check(sigmas)
        live = _live(function)
        if kwargs.get("noise_sampler") is None and declares_keyword(live, "noise_sampler"):
            kwargs["noise_sampler"] = _run_noise_sampler(live, x, sigmas, kwargs.get("extra_args"))
        state["noise_sampler"] = kwargs.get("noise_sampler")
        state["x"] = live(model, x, sigmas[:split + 1], *args, **kwargs)
        return state["x"]

    def second(model, x, sigmas, *args, **kwargs):
        _check(sigmas)
        if "x" not in state:
            raise RuntimeError("LoRA Base 먼저: the base run left no latent")
        live = _live(function)
        if state.get("noise_sampler") is not None:
            kwargs["noise_sampler"] = state["noise_sampler"]
        if declares_keyword(live, STEP_OFFSET_KWARG):
            kwargs[STEP_OFFSET_KWARG] = split
        callback = kwargs.get("callback")
        if callback is not None:
            kwargs["callback"] = lambda info: callback({**info, "i": info["i"] + split})
        return live(model, state["x"], sigmas[split:], *args, **kwargs)

    return first, second


def split_lora_sample(model, schedule, split, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                      latent_image, denoise, *, callback=None, disable_pbar=False, seed=None):
    """``split`` steps with the base MODEL, then the rest with the LoRAs (see the module docstring)."""
    from .speed_nodes import UNSUPPORTED_SAMPLERS

    samplers = importlib.import_module("comfy.samplers")
    sample = importlib.import_module("comfy.sample")
    torch = require_torch()
    total, split = int(steps), int(split)
    inner = samplers.sampler_object(sampler_name)
    function = getattr(inner, "sampler_function", None)
    reason = UNSUPPORTED_SAMPLERS.get(getattr(sampler_base(function), "__name__", "")) if function else "no sampler function"
    if reason:
        raise RuntimeError(SPLIT_REFUSED.format(what=f"{sampler_name} ({reason})"))
    sigmas = samplers.KSampler(model, steps=total, device=model.load_device, sampler=sampler_name, scheduler=scheduler,
                               denoise=denoise, model_options=model.model_options).sigmas
    first, second = split_sampler_functions(function, split, total)
    options = dict(getattr(inner, "extra_options", None) or {})
    inpaint = dict(getattr(inner, "inpaint_options", None) or {})
    LOGGER.info("%s LoRA Base 먼저: steps 1-%d base, %d-%d with the LoRAs", LOG_PREFIX, split, split + 1, total)
    sample.sample_custom(base_model_of(model, schedule), noise, cfg, samplers.KSAMPLER(first, options, inpaint), sigmas,
                         positive, negative, latent_image, callback=callback, disable_pbar=disable_pbar, seed=seed)
    return sample.sample_custom(model, torch.zeros_like(noise), cfg, samplers.KSAMPLER(second, options, inpaint), sigmas,
                                positive, negative, latent_image, callback=callback, disable_pbar=disable_pbar, seed=seed)


# ---------------------------------------------------------------------------------------------- CFG schedule

def cfg_schedule_wrapper(start_cfg: float, end_percent: float, curve: str):
    """``WrappersMP.PREDICT_NOISE``: the guider's CFG for this step, put back after the call."""

    def predict_noise(executor, x, timestep, model_options=None, seed=None, *args, **kwargs):
        model_options = model_options if model_options is not None else {}
        guider = executor.class_obj
        base = guider.cfg
        progress = run_progress(timestep, model_options.get("transformer_options") or {})
        cfg = scheduled_cfg(progress, base, start_cfg, end_percent, curve)
        if cfg == base:
            return executor(x, timestep, model_options, seed, *args, **kwargs)
        guider.cfg = cfg
        try:
            return executor(x, timestep, model_options, seed, *args, **kwargs)
        finally:
            guider.cfg = base

    return predict_noise


class AIStudioCFGSchedule:
    """A higher CFG early in the main pass, falling to the sampler's CFG by ``end_percent`` % of the steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "start_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5,
                                    "tooltip": "The CFG at the first step."}),
            "end_percent": ("INT", {"default": 40, "min": 0, "max": 100, "step": 5,
                                    "tooltip": "From this % of the steps on, the sampler's own CFG."}),
            "curve": (list(CFG_CURVES), {"tooltip": "코사인 eases, 선형 goes straight, 계단 holds the start CFG."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "CFG schedule: start_cfg at the first step, the sampler's CFG from end_percent % of the steps on."

    def patch(self, model, start_cfg=5.0, end_percent=40, curve=CFG_CURVES[0]):
        if curve not in CFG_CURVES:
            raise ValueError(f"curve must be one of {', '.join(CFG_CURVES)}: {curve!r}")
        if int(end_percent) <= 0:
            return (model,)
        patched = clone_model(model, "CFG schedule")
        patched.add_wrapper_with_key(_wrapper_types().PREDICT_NOISE, CFG_WRAPPER_KEY,
                                     cfg_schedule_wrapper(float(start_cfg), int(end_percent), curve))
        return (patched,)


# ---------------------------------------------------------------------------------------------- word emphasis

_T5_TOKENIZER = None


def _t5_tokenizer():
    """ComfyUI's Anima T5 tokenizer (the one that builds the prompt's ``t5xxl_ids``), made once."""
    global _T5_TOKENIZER
    if _T5_TOKENIZER is None:
        _T5_TOKENIZER = importlib.import_module("comfy.text_encoders.anima").T5XXLTokenizer()
    return _T5_TOKENIZER


def phrase_token_ids(phrases: list[str], tokenizer: Any = None) -> list[list[int]]:
    """Each phrase's T5 ids as ComfyUI tokenizes it inside a prompt (word by word, without the end token)."""
    tokenizer = tokenizer or _t5_tokenizer()
    end = getattr(tokenizer, "end_token", None)
    pad = getattr(tokenizer, "pad_token", None)
    out = []
    for phrase in phrases:
        batches = tokenizer.tokenize_with_weights(phrase)
        ids = [int(pair[0]) for batch in batches for pair in batch]
        while ids and ids[-1] in (end, pad):
            ids.pop()
        if ids:
            out.append(ids)
    return out


def emphasis_extra_conds(previous, sequences: list[list[int]], logit: float):
    """``extra_conds`` that adds each cond's logit offsets (positive: the words' positions; negative: zeros, so both
    still batch together). The 3.8B connector's conds (``qwen35_gated_run``) build their context later, from other
    tokens: they get nothing."""

    def extra_conds(**kwargs):
        out = previous(**kwargs)
        cross = out.get("c_crossattn")
        ids = kwargs.get("t5xxl_ids")
        if cross is None or ids is None or kwargs.get("qwen35_gated_run") is not None:
            return out
        torch = require_torch()
        conds = importlib.import_module("comfy.conds")
        length = int(cross.cond.shape[1])
        bias = torch.zeros((1, length), dtype=torch.float32)
        if kwargs.get("prompt_type") == "positive":
            flat = ids.reshape(-1).tolist() if hasattr(ids, "reshape") else list(ids)
            positions = [position for position in find_positions(flat, sequences) if position < length]
            if positions:
                bias[0, positions] = logit
        out[EMPHASIS_COND_KEY] = conds.CONDRegular(bias)
        return out

    return extra_conds


def emphasis_apply_wrapper(start_percent: float, end_percent: float):
    """``WrappersMP.APPLY_MODEL``: takes the offsets out of the model's kwargs and, inside the step window, hands them to
    the attention override per batch row (and per cond / uncond chunk, for a wrapper that splits the batch)."""

    def apply_model(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        transformer_options = transformer_options if transformer_options is not None else {}
        bias = kwargs.pop(EMPHASIS_COND_KEY, None)
        if bias is not None and bool((bias != 0).any()) and in_window(run_progress(t, transformer_options),
                                                                      start_percent, end_percent):
            flags = list(transformer_options.get("cond_or_uncond") or [])
            chunks = {}
            if flags and bias.shape[0] % len(flags) == 0:
                rows = bias.shape[0] // len(flags)
                for index, flag in enumerate(flags):
                    chunks.setdefault(flag, bias[index * rows:(index + 1) * rows])
            transformer_options = {**transformer_options, EMPHASIS_OPTION_KEY: {"rows": bias, "chunks": chunks}}
        return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    return apply_model


def emphasis_attn2_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
    """Cosmos ``attn2_patch``: called in a cross-attention's ``compute_qkv`` only, right before its attention call. It
    marks that call for the override and changes nothing."""
    entry = (extra_options or {}).get(EMPHASIS_OPTION_KEY)
    if entry is not None:
        entry[CROSS_PENDING] = True
    return {"q": q, "k": k, "v": v, "pe": pe}


def _emphasis_rows(entry: dict, rows: int, options: dict):
    """The offsets for a call of ``rows`` batch rows: the whole batch's, or one cond / uncond chunk's."""
    bias = entry["rows"]
    if bias.shape[0] == rows:
        return bias
    flags = list(options.get("cond_or_uncond") or [])
    if len(flags) == 1:
        chunk = entry["chunks"].get(flags[0])
        if chunk is not None and chunk.shape[0] == rows:
            return chunk
    return None


def emphasis_attention_override(previous):
    """``optimized_attention_override``: Anima's cross-attention calls in the window run through PyTorch SDPA with the
    offsets as an additive mask; anything else goes to ``previous`` (the attention pick) or ComfyUI's function."""

    def override(func, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        entry = options.get(EMPHASIS_OPTION_KEY)
        if entry is not None and entry.pop(CROSS_PENDING, False) and len(args) >= 4 and kwargs.get("skip_reshape"):
            q, k = args[0], args[1]
            if getattr(q, "ndim", 0) == 4 and getattr(k, "ndim", 0) == 4:
                bias = _emphasis_rows(entry, q.shape[0], options)
                if bias is not None and k.shape[-2] == bias.shape[-1]:  # the keys are the text tokens
                    attention = importlib.import_module("comfy.ldm.modules.attention")
                    mask = bias.to(device=q.device, dtype=q.dtype).reshape(bias.shape[0], 1, 1, bias.shape[-1])
                    rest = {name: value for name, value in kwargs.items() if name != "mask"}
                    return attention.attention_pytorch(*args, mask=mask, **rest)
        if previous is not None:
            return previous(func, *args, **kwargs)
        return func(*args, **kwargs)

    return override


def _is_anima(model: Any) -> bool:
    inner = getattr(model, "model", None)
    return any(klass.__name__ == "Anima" for klass in type(inner).__mro__) if inner is not None else False


class AIStudioWordEmphasis:
    """More cross-attention on the listed words' tokens early in the main pass (Anima)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "words": ("STRING", {"multiline": True, "default": "",
                                 "tooltip": "Words or tags of the prompt, comma or line separated."}),
            "multiplier": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                                     "tooltip": "Their attention weight times this (below 1 = less)."}),
            "start_percent": ("INT", {"default": 0, "min": 0, "max": 100, "step": 5}),
            "end_percent": ("INT", {"default": 40, "min": 0, "max": 100, "step": 5}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Word emphasis: the listed words' tokens get `multiplier` times their cross-attention weight in a step window."

    def patch(self, model, words="", multiplier=2.0, start_percent=0, end_percent=40):
        phrases = emphasis_phrases(words)
        if not phrases or float(multiplier) == 1.0 or int(end_percent) <= int(start_percent):
            return (model,)
        if not _is_anima(model):
            LOGGER.info("%s 단어 주목: Anima only, left out for this model", LOG_PREFIX)
            return (model,)
        sequences = phrase_token_ids(phrases)
        if not sequences:
            return (model,)
        patched = clone_model(model, "Word emphasis")
        patched.add_object_patch("extra_conds", emphasis_extra_conds(
            patched.get_model_object("extra_conds"), sequences, emphasis_logit(multiplier)))
        patched.add_wrapper_with_key(_wrapper_types().APPLY_MODEL, EMPHASIS_WRAPPER_KEY,
                                     emphasis_apply_wrapper(int(start_percent), int(end_percent)))
        patched.set_model_attn2_patch(emphasis_attn2_patch)
        options = dict(patched.model_options)
        transformer = dict(options.get("transformer_options") or {})
        transformer["optimized_attention_override"] = emphasis_attention_override(
            transformer.get("optimized_attention_override"))
        options["transformer_options"] = transformer
        patched.model_options = options
        LOGGER.info("%s 단어 주목: %s x%g, %d-%d %%", LOG_PREFIX, ", ".join(phrases), float(multiplier),
                    int(start_percent), int(end_percent))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "AIStudioLoraControl": AIStudioLoraControl,
    "AIStudioCFGSchedule": AIStudioCFGSchedule,
    "AIStudioWordEmphasis": AIStudioWordEmphasis,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioLoraControl": "AI Studio LoRA Control (Base 먼저 · 교차 어텐션)",
    "AIStudioCFGSchedule": "AI Studio CFG Schedule",
    "AIStudioWordEmphasis": "AI Studio Word Emphasis",
}
