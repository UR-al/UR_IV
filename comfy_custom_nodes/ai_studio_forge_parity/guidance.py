"""Forge Neo guidance features expressed as composable ComfyUI MODEL nodes.

The tensor math mirrors the user's ``forge_sam3_extension`` implementation.
Imports are deliberately lazy so the app can validate contracts without
loading ComfyUI or torch.

Each feature lives in its own module: ``guidance_pag`` (PAG/SEG/SLG),
``guidance_dcw`` (DCW/RDC, CWM, SMC, APG), ``guidance_dave``, ``guidance_skim``,
``guidance_dd`` (Detail Daemon) and ``guidance_cns``, with shared helpers in
``guidance_common``. This module keeps the NegPiP and modulation provider
nodes, the Anima guidance suite that composes the features, and the node
registration. It re-exports the feature modules' names, so
``guidance.<name>`` keeps working for the workflow compiler, ``generation.py``,
the sam-extra contract and the tests.

The suite applies the features in sam-extra's order (``scripts/anima_safe_pag.py``
``_post_cfg``): the CFG stage (SMC -> APG -> CWM, one ``sampler_cfg_function``),
then PAG/SEG/SLG, then DCW/RDC, each post-CFG function appended after the one
before. Adaptive guidance is installed before PAG, and CNS is the MODEL's
sampler wrapper (``guidance_cns.apply_cns``).
"""

from __future__ import annotations

import functools
import math
from typing import Any

from .compat import clone_model, invoke_provider, provider

# Re-exports (see the module docstring) — also used by the suite below.
from .guidance_common import (  # noqa: F401
    CATEGORY,
    parse_indices,
    _as_bool,
    _haar_dwt,
    _haar_idwt,
    _json_settings,
    _model_blocks,
    _pad_even,
    _sampling_percent,
    _sampling_percent_for_model,
    _scalar_sigma,
    _setting,
    _sigma_norm,
    _transformer_options,
)
from .guidance_pag import (  # noqa: F401
    ForgeNeoAnimaSafePAG,
    apply_anima_safe_pag,
    normalize_rescale_mode,
    reject_unsupported_seg,
    _gaussian_blur_query,
    _patch_perturbation_guidance,
)
from .guidance_dcw import (  # noqa: F401
    APG_STATE_KEY,
    SMC_PRESET_CHOICES,
    apply_dcw,
    ForgeNeoDCWCWMSMC,
    patch_dcw,
    _apply_apg_error,
    _cwm_error,
    _detect_smc,
    _energy_weight,
    _project_guidance,
    _smc_error,
    _SMC_PRESETS,
)
from .guidance_dave import (  # noqa: F401
    apply_dave,
    ForgeNeoAnimaDAVE,
    _patch_anima_blocks,
)
from .guidance_skim import (  # noqa: F401
    ForgeNeoSkimmedCFG,
)
from .guidance_dd import (  # noqa: F401
    DD_SIGMA_SCALE,
    DD_WRAPPER_KEY,
    SETTING_KEYS as DD_SETTING_KEYS,
    detail_daemon_node_values,
    detail_daemon_schedule,
    ForgeNeoAnimaDetailDaemon,
)
from .guidance_cns import (  # noqa: F401
    CNS_DEFAULTS,
    apply_cns,
    color_noise_wavelet,
)


class ForgeNeoNegPip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "enabled": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, clip, enabled=True):
        if not enabled:
            return model, clip
        return invoke_provider(
            "CLIPNegPip",
            method="execute",
            feature="NegPiP",
            kwargs={"model": model, "clip": clip},
        )[:2]


class ForgeNeoAnimaModGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "quality_tags": ("STRING", {"default": "highres, best quality, score_7", "multiline": True}),
            "quality_negative": ("STRING", {"default": "score_1, score_2, score_3, worst quality, lowres", "multiline": True}),
            "weight": ("FLOAT", {"default": 3.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            "start_layer": ("INT", {"default": 8, "min": 0, "max": 255}),
            "end_layer": ("INT", {"default": 27, "min": 0, "max": 255}),
            "taper_layers": ("INT", {"default": 0, "min": 0, "max": 255}),
            "taper_scale": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
            "final_layer_weight": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0}),
            "adapter_name": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, clip, positive, negative, enabled=False, quality_tags="", quality_negative="", weight=3.0, start_layer=8, end_layer=27, taper_layers=0, taper_scale=0.25, final_layer_weight=0.0, adapter_name=""):
        if not enabled or float(weight) == 0:
            return (model,)
        instance = provider("AnimaModGuidance", feature="Anima modulation guidance")
        public = getattr(instance, "patch", None)
        globals_dict = getattr(public, "__globals__", {})
        exact = globals_dict.get("_apply_mod_guidance")
        if not callable(exact):
            raise RuntimeError(
                "Installed Spectrum provider lacks the exact layer-range modulation API; update comfyui-spectrum-ksampler."
            )
        return (exact(
            model, clip, positive, negative, adapter_name or None, quality_tags,
            quality_neg=quality_negative, w=float(weight),
            start_layer=int(start_layer), end_layer=int(end_layer),
            taper=int(taper_layers), taper_scale=float(taper_scale),
            final_w=float(final_layer_weight),
        ),)


# Per-run marker the suite's adaptive guidance writes on every model call and
# its CFG stage reads: {"skipped": True} when the call ran cond-only.
ADG_STEP_KEY = "_forge_neo_adg_step"


def _patch_adaptive_guidance(
    model: Any, *, start_percent: float, interval: int
):
    """Cond-only batches after ``start_percent`` (uncond := cond).

    On a skipped step every cond entry gets the cond prediction, including the
    extra entries a later ``sampler_calc_cond_batch_function`` appended. The
    suite installs this before PAG, so the original PAG node chains it as
    ``previous_calc`` (origin: iljung1106/comfyui-anima-safe-pag@905b0107
    :__init__.py:237, :255-269): its padded row is the cond prediction there,
    so the PAG term is 0. SEG/SLG would still run their weak pass, so the suite
    gates every perturbation post-CFG function on the marker below
    (``_skip_perturbation_on_adg_steps``). The step is then plain cond like
    sam-extra's ADG step (``_post_cfg``: ``has_pert = not adg_skipped and``).

    Every call records in ``model_options[ADG_STEP_KEY]`` whether it ran
    cond-only, for the suite's CFG stage (``_patch_cfg_stage``). The nested
    dict is copied into each sampling run's options and the original PAG node
    copies ``model_options`` shallowly (:285), so the calc function and the CFG
    function share it. ``disable_cfg1_optimization`` is not set: at CFG 1 there
    is no uncond pass to skip, and sam-extra's ADG does not force one either.
    """

    patched = clone_model(model, "Anima Adaptive Guidance")
    options = dict(getattr(patched, "model_options", {}) or {})
    if options.get("sampler_calc_cond_batch_function") is not None:
        raise RuntimeError(
            "Adaptive Guidance cannot replace an existing sampler_calc_cond_batch_function."
        )
    state: dict[str, Any] = {"sigma": None, "step": 0}

    def calculate(args):
        import comfy.samplers  # lazy: only reachable inside ComfyUI

        sigma = args.get("sigma", 1.0)
        current = _scalar_sigma(sigma)
        last = state["sigma"]
        if last is None or current > float(last) + 1e-6:
            state["step"] = 0
        elif abs(current - float(last)) > 1e-8:
            state["step"] += 1
        state["sigma"] = current
        conds = list(args.get("conds") or [])
        if not conds:
            raise RuntimeError("Adaptive Guidance received no sampler conditioning.")
        progress = _sampling_percent_for_model(patched, sigma)
        keep_interval = max(0, int(interval))
        skip_uncond = progress >= float(start_percent) and (
            keep_interval == 0 or state["step"] % keep_interval != 0
        )
        skipped = skip_uncond and len(conds) > 1 and conds[1] is not None
        marker = (args.get("model_options") or {}).get(ADG_STEP_KEY)
        if isinstance(marker, dict):
            marker["skipped"] = skipped
        if skipped:
            (conditional,) = comfy.samplers.calc_cond_batch(
                args["model"], [conds[0]], args["input"], sigma,
                args["model_options"],
            )
            return [conditional] * len(conds)
        return comfy.samplers.calc_cond_batch(
            args["model"], conds, args["input"], sigma, args["model_options"]
        )

    options["sampler_calc_cond_batch_function"] = calculate
    options[ADG_STEP_KEY] = {"skipped": False}
    patched.model_options = options
    return patched


def _adg_step_skipped(args: dict[str, Any]) -> bool:
    marker = (args.get("model_options") or {}).get(ADG_STEP_KEY)
    return isinstance(marker, dict) and bool(marker.get("skipped"))


def _skip_perturbation_on_adg_steps(before: Any, patched: Any):
    """No PAG/SEG/SLG on a step adaptive guidance ran cond-only.

    sam-extra adds no perturbation term on such a step and runs no weak pass
    (``scripts/anima_safe_pag.py`` ``_post_cfg``: ``has_pert = not adg_skipped
    and ...``; with interval 0 the window is also cut at the ADG start). Each
    post-CFG function ``_patch_perturbation_guidance`` appended to ``before``'s
    list is wrapped to return ``denoised`` on a skipped step, before SEG/SLG's
    weak ``calc_cond_batch``. The original PAG's term there is already 0 (see
    ``_patch_adaptive_guidance``), so the gate changes no PAG value. The
    wrappers keep the wrapped function's name (``functools.wraps``).
    """

    if patched is before:
        return patched
    kept = len((getattr(before, "model_options", {}) or {}).get("sampler_post_cfg_function") or [])
    functions = list(patched.model_options.get("sampler_post_cfg_function") or [])

    def gate(function):
        @functools.wraps(function)
        def perturbation_unless_adg_skipped(args):
            if _adg_step_skipped(args):
                return args["denoised"]
            return function(args)

        return perturbation_unless_adg_skipped

    patched.model_options["sampler_post_cfg_function"] = [
        *functions[:kept], *(gate(function) for function in functions[kept:])
    ]
    return patched


def _suite_smc_preset(value: Any) -> str:
    """The SMC preset once a suite switch has turned SMC on, read like sam-extra.

    An unknown name reads as Off (``sam3ext/guidance/cwm_smc.py``
    ``normalize_smc_preset``), and Off with SMC on means Custom, the two values
    (``scripts/anima_safe_pag.py`` ``effective_smc_preset``). Off, the original
    DCW(+a) default (dcw_node.py:713-716), is also the fallback.
    """

    text = str(value or "").strip().casefold()
    for choice in SMC_PRESET_CHOICES:
        if choice.casefold() == text and choice != "Off":
            return choice
    return "Custom"


def _plain_cfg_noise(args: dict[str, Any]) -> Any:
    scale = float(args.get("cond_scale", 7.0))
    return args["uncond"] + scale * (args["cond"] - args["uncond"])


def _patch_cfg_stage(
    model: Any,
    *,
    cwm_enabled: bool,
    alpha_low: float,
    alpha_high: float,
    smc_preset: str,
    smc_lambda: float,
    smc_k: float,
    apg_settings: dict[str, float] | None,
):
    """The CFG stage, SMC -> APG -> CWM, as DCW(+a)'s ``sampler_cfg_function``.

    ``guidance_dcw.patch_dcw`` installs the hook (SMC/CWM as the original,
    APG between them). Two sam-extra rules sit on top of it:

    - A call adaptive guidance ran cond-only (uncond := cond) returns the cond
      prediction: no SMC/APG/CWM, APG momentum cleared, SMC's ``e_prev`` kept
      (plan §3.3 F; sam-extra ``_post_cfg`` and ``reset_cfg_state``).
    - APG does not run at CFG ~ 1 (sam-extra ``_cfg_base_skip_reason``); SMC and
      CWM still do, as in the original. With APG alone
      ``disable_cfg1_optimization`` is not kept, so CFG 1 runs no uncond pass.

    An earlier node's ``sampler_cfg_function`` is left alone (DCW(+a) warns and
    skips CWM/SMC/APG).

    Returns ``(model, apg_installed)``: whether APG is in the installed hook,
    which gates the suite's PAG rescale auto-off.
    """

    common = dict(
        dcw_enabled=False, lambda_l=0.0, lambda_h=0.0,
        cwm_enabled=cwm_enabled, alpha_l=alpha_low, alpha_h=alpha_high,
        smc_preset=smc_preset, smc_lambda=smc_lambda, smc_k=smc_k,
        rdc_tau=0.0, rdc_alpha_ll=0.0, rdc_alpha_hh=0.0,
    )
    incoming = dict(getattr(model, "model_options", {}) or {})
    patched = patch_dcw(model, apg_settings=apg_settings, **common)
    if patched is model:
        return model, False
    options = patched.model_options
    hook = options.get("sampler_cfg_function")
    if hook is None or hook is incoming.get("sampler_cfg_function"):
        return patched, False

    without_apg = None
    if apg_settings is not None:
        smc_cwm = patch_dcw(model, apg_settings=None, **common)
        if smc_cwm is not model:
            without_apg = smc_cwm.model_options.get("sampler_cfg_function")
        elif "disable_cfg1_optimization" not in incoming:
            options.pop("disable_cfg1_optimization", None)

    def cfg_stage(args: dict[str, Any]):
        run_options = args.get("model_options") or {}
        marker = run_options.get(ADG_STEP_KEY)
        if isinstance(marker, dict) and marker.get("skipped"):
            apg_state = run_options.get(APG_STATE_KEY)
            if isinstance(apg_state, dict):
                apg_state.clear()
            return args["cond"]
        if apg_settings is not None and math.isclose(float(args.get("cond_scale", 7.0)), 1.0):
            return without_apg(args) if without_apg is not None else _plain_cfg_noise(args)
        return hook(args)

    options["sampler_cfg_function"] = cfg_stage
    return patched, apg_settings is not None


class ForgeNeoAnimaGuidanceSuite:
    """Compose the extension's 62-field Anima guidance payload on one MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "settings_json": ("STRING", {"default": "{}", "multiline": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, clip, positive, negative, enabled=False, settings_json="{}"):
        if not enabled:
            return (model,)
        settings = _json_settings(settings_json, "Anima guidance suite")

        attention_requested = _as_bool(_setting(settings, "guid_enabled", False))
        method_text = str(_setting(settings, "guid_attn_method", "PAG")).strip().casefold()
        method_map = {"pag": "pag", "seg": "seg", "none": None, "off": None, "": None}
        if method_text not in method_map:
            raise RuntimeError(f"Unsupported Anima attention guidance method: {method_text!r}.")
        attention_method = method_map[method_text] if attention_requested else None
        # PAG takes legacy (legacy strength) and head indices through the original
        # node; only SEG's legacy/head modes are rejected, before the model is touched.
        legacy_attn = _as_bool(_setting(settings, "guid_legacy_attn", False))
        head_indices = str(_setting(settings, "guid_head_indices", ""))
        reject_unsupported_seg(
            attention_method, legacy_attn=legacy_attn, head_indices=head_indices
        )

        slg_enabled = _as_bool(_setting(settings, "guid_slg_on", False))
        dave_enabled = _as_bool(_setting(settings, "guid_dave_enabled", False))
        current = _patch_anima_blocks(
            model,
            dave_enabled=dave_enabled,
            dave_blocks=str(_setting(settings, "guid_dave_blocks", "8-18")),
            dave_strength=float(_setting(settings, "guid_dave_strength", 0.3)),
            dave_tau=float(_setting(settings, "guid_dave_tau", 0.1)),
            dave_pre_dd=_as_bool(_setting(settings, "guid_dave_pre_dd", True)),
            slg_enabled=slg_enabled,
            slg_blocks=str(_setting(settings, "guid_slg_blocks", "18")),
        )

        mode_text = str(_setting(settings, "guid_cfg_mode", "Preserve incoming")).strip().casefold()
        mode_map = {
            "preserve incoming": "preserve", "preserve": "preserve",
            "apg": "apg", "cwm": "cwm", "smc": "smc",
            "smc + cwm": "smc+cwm", "smc+cwm": "smc+cwm",
        }
        if mode_text not in mode_map:
            raise RuntimeError(f"Unsupported Anima CFG mode: {mode_text!r}.")
        mode = mode_map[mode_text]
        stack = _as_bool(_setting(settings, "guid_experimental_stack", False))
        apg_enabled = (
            _as_bool(_setting(settings, "guid_apg_enabled", False))
            or mode == "apg" or stack
        )
        smc_enabled = (
            _as_bool(_setting(settings, "guid_smc_enabled", False))
            or _as_bool(_setting(settings, "guid_smc_master_enabled", False))
            or mode in {"smc", "smc+cwm"} or stack
        )
        cwm_enabled = (
            _as_bool(_setting(settings, "guid_cwm_enabled", False))
            or mode in {"cwm", "smc+cwm"} or stack
        )
        dcw_enabled = _as_bool(_setting(settings, "guid_dcw_enabled", False))
        # The original has no RDC switch (RDC is on when rdc_tau > 0 inside DCW;
        # origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:855-856). sam-extra
        # keeps arg 58 as a veto-only slot that defaults to True
        # (scripts/anima_safe_pag.py ``rdc_switch``), so only an explicit False
        # turns RDC off here.
        rdc_enabled = _as_bool(_setting(settings, "guid_rdc_enabled", True), True)

        # sam-extra's order (scripts/anima_safe_pag.py _post_cfg): the CFG stage
        # (SMC -> APG -> CWM), then PAG/SEG/SLG, then DCW/RDC. Fallbacks for
        # missing keys are the original nodes' defaults (DCW(+a): lambda
        # 0.05/0.01, alpha 0/0, SMC Off, rdc_tau 0; CNS: guidance_cns.CNS_DEFAULTS).
        apg_installed = False
        if apg_enabled or smc_enabled or cwm_enabled:
            current, apg_installed = _patch_cfg_stage(
                current,
                cwm_enabled=cwm_enabled,
                alpha_low=float(_setting(settings, "guid_cwm_alpha_low", 0.0)),
                alpha_high=float(_setting(settings, "guid_cwm_alpha_high", 0.0)),
                smc_preset=(
                    _suite_smc_preset(_setting(settings, "guid_smc_preset", "Off"))
                    if smc_enabled else "Off"
                ),
                smc_lambda=float(_setting(settings, "guid_smc_lambda", 6.0)),
                smc_k=float(_setting(settings, "guid_smc_k", 0.1)),
                apg_settings={
                    "eta": float(_setting(settings, "guid_apg_eta", 0.0)),
                    "norm_threshold": float(_setting(settings, "guid_apg_norm", 15.0)),
                    "momentum": float(_setting(settings, "guid_apg_momentum", 0.0)),
                } if apg_enabled else None,
            )

        # ADG before PAG: the original PAG node chains an existing
        # sampler_calc_cond_batch_function as previous_calc, and ADG would refuse
        # to replace PAG's. See _patch_adaptive_guidance.
        adg_enabled = _as_bool(_setting(settings, "guid_adg_enabled", False))
        if adg_enabled:
            current = _patch_adaptive_guidance(
                current,
                start_percent=float(_setting(settings, "guid_adg_start", 0.5)),
                interval=int(_setting(settings, "guid_adg_interval", 0)),
            )

        if attention_method or slg_enabled:
            rescale = float(_setting(settings, "guid_rescale", 0.20))
            # sam-extra turns the rescale off only while APG runs (``apg_governs``
            # in ``_apply_perturbation``): not when an earlier node's
            # sampler_cfg_function kept the CFG stage, and so APG, out.
            if apg_installed and _as_bool(_setting(settings, "guid_apg_autooff", True), True):
                rescale = 0.0
            before_perturbation = current
            current = _patch_perturbation_guidance(
                current,
                attention_method=attention_method,
                attention_scale=float(_setting(settings, "guid_scale", 4.0)),
                attention_blocks=str(_setting(settings, "guid_block_indices", "18")),
                attention_strength=float(
                    _setting(settings, "guid_official_strength", _setting(settings, "guid_legacy_strength", 0.75))
                ),
                seg_sigma=float(_setting(settings, "guid_seg_sigma", 100.0)),
                slg_enabled=slg_enabled,
                slg_scale=float(_setting(settings, "guid_slg_scale", 3.0)),
                start_percent=float(_setting(settings, "guid_start_percent", 0.0)),
                end_percent=float(_setting(settings, "guid_end_percent", 0.7)),
                rescale=rescale,
                rescale_mode=str(_setting(settings, "guid_rescale_mode", "full")),
                head_indices=head_indices,
                legacy_attn=legacy_attn,
                legacy_strength=float(_setting(settings, "guid_legacy_strength", 0.75)),
            )
            if adg_enabled:
                current = _skip_perturbation_on_adg_steps(before_perturbation, current)

        # DCW/RDC last: appended after the PAG/SEG/SLG post-CFG functions. As in
        # the original, RDC only runs inside DCW; the suite's RDC switch only vetoes it.
        if dcw_enabled:
            rdc_tau = float(_setting(settings, "guid_rdc_tau", 0.0)) if rdc_enabled else 0.0
            current = patch_dcw(
                current,
                dcw_enabled=True,
                lambda_l=float(_setting(settings, "guid_dcw_lambda_low", 0.05)),
                lambda_h=float(_setting(settings, "guid_dcw_lambda_high", 0.01)),
                cwm_enabled=False, alpha_l=0.0, alpha_h=0.0,
                smc_preset="Off", smc_lambda=6.0, smc_k=0.1,
                rdc_tau=rdc_tau,
                rdc_alpha_ll=float(_setting(settings, "guid_rdc_alpha_ll", 0.03)),
                rdc_alpha_hh=float(_setting(settings, "guid_rdc_alpha_hh", 0.0)),
            )

        if _as_bool(_setting(settings, "guid_mod_enabled", False)):
            mod_clip = clip
            clip_name = str(_setting(settings, "guid_mod_clip_model", "")).strip()
            if clip_name:
                mod_clip = invoke_provider(
                    "CLIPLoader", method="load_clip", feature="Anima modulation CLIP-L",
                    args=(clip_name, "stable_diffusion", "default"),
                )[0]
            mod_positive, mod_negative = positive, negative
            if str(_setting(settings, "guid_mod_base_source", "Main positive")).strip().casefold() == "custom":
                mod_positive = invoke_provider(
                    "CLIPTextEncode", method="encode", feature="Anima modulation base prompt",
                    args=(mod_clip, str(_setting(settings, "guid_mod_base_prompt", ""))),
                )[0]
            if str(_setting(settings, "guid_mod_negative_source", "Main negative")).strip().casefold() == "custom":
                mod_negative = invoke_provider(
                    "CLIPTextEncode", method="encode", feature="Anima modulation negative prompt",
                    args=(mod_clip, str(_setting(settings, "guid_mod_negative_prompt", ""))),
                )[0]
            adapter_mode = str(_setting(settings, "guid_mod_adapter_mode", "Auto-download official")).strip().casefold()
            adapter = ""
            if adapter_mode == "local file":
                adapter = str(_setting(settings, "guid_mod_adapter_path", "")).strip()
                if not adapter:
                    raise RuntimeError(
                        "Anima modulation is set to Local file but guid_mod_adapter_path is empty."
                    )
            elif adapter_mode != "auto-download official":
                raise RuntimeError(f"Unsupported modulation adapter mode: {adapter_mode!r}.")
            blocks = _model_blocks(current)
            end_layer = int(_setting(settings, "guid_mod_end_layer", -1))
            if end_layer < 0 and blocks:
                end_layer = len(blocks) - 1
            current = ForgeNeoAnimaModGuidance().patch(
                current, mod_clip, mod_positive, mod_negative, True,
                str(_setting(settings, "guid_mod_positive_prompt", "masterpiece, best quality, highres")),
                "",
                float(_setting(settings, "guid_mod_weight", 3.0)),
                int(_setting(settings, "guid_mod_start_layer", 0)),
                end_layer,
                0, 0.25, 0.0, adapter,
            )[0]

        if _as_bool(_setting(settings, "guid_cns_enabled", False)):
            # The MODEL's outermost SAMPLER_SAMPLE wrapper colours every step's
            # noise in every sampling run of this model (KSampler, Hires, DD).
            current = apply_cns(current, *(
                float(_setting(settings, f"guid_cns_{name}", CNS_DEFAULTS[name]))
                for name in ("strength", "gamma_power", "gamma_scale")
            ))
        return (current,)


NODE_CLASS_MAPPINGS = {
    "ForgeNeoNegPip": ForgeNeoNegPip,
    "ForgeNeoAnimaDAVE": ForgeNeoAnimaDAVE,
    "ForgeNeoAnimaModGuidance": ForgeNeoAnimaModGuidance,
    "ForgeNeoSkimmedCFG": ForgeNeoSkimmedCFG,
    "ForgeNeoAnimaSafePAG": ForgeNeoAnimaSafePAG,
    "ForgeNeoDCWCWMSMC": ForgeNeoDCWCWMSMC,
    "ForgeNeoAnimaGuidanceSuite": ForgeNeoAnimaGuidanceSuite,
    "ForgeNeoAnimaDetailDaemon": ForgeNeoAnimaDetailDaemon,
}
