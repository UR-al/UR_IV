"""Extra Samplers for ComfyUI (pack 1.7.0): sam-extra's sampler names in ComfyUI's KSampler list.

sam-extra v0.33.0 adds 18 samplers to Forge's list. Five of them ComfyUI already has under the same
name and the same maths (``CORE_SAMPLERS``: ``dpmpp_2m_sde_heun``, ``uni_pc_bh2``, ``cfgpp_ud10_ab`` from
ComfyUI v0.35.0, ``ipndm``, ``deis`` — the app only maps the Forge label to the name). The other thirteen
(``PACK_SAMPLERS``: twelve under sam-extra's first aliases, in its table order, and ``ipndm_v_fixed``) are
registered here, the way ``compat.install_forge_scheduler_support`` adds ``beta57``, so every pass the app
compiles (``ForgeNeoKSamplerCNS``, ``ForgeNeoHiresFix``, ADetailer through Impact, SAM3, a custom workflow's
stock ``KSampler``/``KSamplerSelect``) finds them by name. ``ipndm_v_fixed`` is sam-extra's "IPNDM_V" as of
0.33.1 (``extra_samplers_ipndm``): ComfyUI's ``ipndm_v`` — sam-extra's alias for that entry — keeps the upstream
typo in its fourth-order weight that sam-extra 0.33.1 fixed, and the pack never replaces a core name, so the
fixed sampler has a name of its own and the app maps "IPNDM_V" to it:

* the function goes into ``comfy.k_diffusion.sampling`` as ``sample_<name>`` (what
  ``comfy.samplers.ksampler`` looks up) — a thin ``functools.wraps`` wrapper tagged with the pack's owner
  (``compat.mark_owned``), so CNS's direct-sampler detection sees through it and finds its
  ``noise_sampler`` parameter;
* the name is appended to ``KSAMPLER_NAMES``, ``SAMPLER_NAMES`` and ``KSampler.SAMPLERS`` (each distinct
  list once), and ``dpmpp_4m_sde`` / ``dpmpp_3m_flow_ode`` to ``KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS``
  (Forge's ``discard_next_to_last_sigma`` option of those two entries); Impact's ``FaceDetailer`` (ADetailer,
  SAM3 restore face) keeps its own literal discard list, so those passes hand it ``impact_scheduler_func``;
* a name another provider already registered is never overwritten (``compat.register_names``): it is
  left alone, logged once and not reported as the pack's — the app refuses a graph that would run a
  different algorithm under it (e.g. Koishi-Star's own ComfyUI node registers ``euler_dy`` /
  ``euler_smea_dy`` with its always-churn rule).

The four Dy entries run through ``extra_samplers_dy_host.guarded`` (ComfyUI's sub-step inputs, the
sub-step guard). A name whose implementation module is not in this pack yet (``IMPLEMENTATIONS``) is
not registered and is listed by ``unavailable_names``. No torch or ComfyUI import at module level;
outside ComfyUI (the app's tests) registration does nothing.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Any, Callable

from .compat import NameRegistration, entry_owner, mark_owned, register_names

LOGGER = logging.getLogger("ai_studio_forge_parity")

OWNER = "extra_samplers"

# sam-extra v0.33.0 registry.SPECS order (first alias = the ComfyUI name), then the pack's own name for sam-extra
# 0.33.1's IPNDM_V (ComfyUI core's ``ipndm_v`` keeps the upstream typo sam-extra 0.33.1 fixed).
PACK_SAMPLERS = (
    "er_sde_reverse_time",
    "er_sde_ode",
    "er_sde_tunable",
    "dpmpp_4m_sde",
    "euler_dy_cfg_pp",
    "euler_smea_dy_cfg_pp",
    "euler_dy",
    "euler_smea_dy",
    "dpmpp_2m_flow_ode",
    "dpmpp_2m_heun_flow_ode",
    "dpmpp_3m_flow_ode",
    "restart_flow",
    "ipndm_v_fixed",
)
# sam-extra entries whose ComfyUI name is core's own (same maths; never registered by the pack).
CORE_SAMPLERS = ("dpmpp_2m_sde_heun", "uni_pc_bh2", "cfgpp_ud10_ab", "ipndm", "deis")
# Forge options discard_next_to_last_sigma (uni_pc_bh2 is already in ComfyUI's set).
DISCARD_PENULTIMATE_SIGMA = ("dpmpp_4m_sde", "dpmpp_3m_flow_ode")
# The Dy entries and whether they are CFG++ (their console line names the step kind).
DY_SAMPLERS = {"euler_dy_cfg_pp": True, "euler_smea_dy_cfg_pp": True, "euler_dy": False, "euler_smea_dy": False}
# Where each name's function lives in the pack: (module, attribute).
IMPLEMENTATIONS = {
    "er_sde_reverse_time": ("extra_samplers_math", "sample_er_sde_reverse_time"),
    "er_sde_ode": ("extra_samplers_math", "sample_er_sde_ode"),
    "er_sde_tunable": ("extra_samplers_math", "sample_er_sde_tunable"),
    "dpmpp_4m_sde": ("extra_samplers_dpmpp_4m", "sample_dpmpp_4m_sde"),
    "euler_dy_cfg_pp": ("extra_samplers_dy", "sample_euler_dy_cfg_pp"),
    "euler_smea_dy_cfg_pp": ("extra_samplers_dy", "sample_euler_smea_dy_cfg_pp"),
    "euler_dy": ("extra_samplers_dy", "sample_euler_dy"),
    "euler_smea_dy": ("extra_samplers_dy", "sample_euler_smea_dy"),
    "dpmpp_2m_flow_ode": ("extra_samplers_flow", "sample_dpmpp_2m_flow_ode"),
    "dpmpp_2m_heun_flow_ode": ("extra_samplers_flow", "sample_dpmpp_2m_heun_flow_ode"),
    "dpmpp_3m_flow_ode": ("extra_samplers_flow", "sample_dpmpp_3m_flow_ode"),
    "restart_flow": ("extra_samplers_restart", "sample_restart_flow"),
    "ipndm_v_fixed": ("extra_samplers_ipndm", "sample_ipndm_v_fixed"),
}


def implementation(name: str) -> Callable[..., Any] | None:
    """The pack's function for ``name``, or None while its module is not in the pack."""
    module_name, attribute = IMPLEMENTATIONS[name]
    try:
        module = importlib.import_module(f".{module_name}", __package__)
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.{module_name}":
            raise
        return None
    function = getattr(module, attribute, None)
    return function if callable(function) else None


def unavailable_names() -> tuple[str, ...]:
    """``PACK_SAMPLERS`` the pack cannot register yet (no implementation)."""
    return tuple(name for name in PACK_SAMPLERS if implementation(name) is None)


def sampler_entry(name: str) -> Callable[..., Any]:
    """The ``sample_<name>`` function the pack puts into ``comfy.k_diffusion.sampling``."""
    function = implementation(name)
    if function is None:
        raise LookupError(f"the pack has no implementation of the sampler {name!r} yet")
    if name in DY_SAMPLERS:
        from .extra_samplers_dy_host import guarded

        function = guarded(function, name, cfg_pp=DY_SAMPLERS[name])

    @functools.wraps(function)
    def entry(*args, **kwargs):
        return function(*args, **kwargs)

    entry.__name__ = entry.__qualname__ = "sample_" + name
    return mark_owned(entry, OWNER)


def _comfy_modules() -> tuple[Any, Any] | None:
    try:
        return (importlib.import_module("comfy.samplers"), importlib.import_module("comfy.k_diffusion.sampling"))
    except ModuleNotFoundError:
        return None


def install_extra_samplers(*, required: bool = False) -> NameRegistration:
    """Register ``PACK_SAMPLERS`` in ComfyUI (module docstring). Called once when the pack is imported;
    again is harmless. Outside ComfyUI it does nothing. Without ``required`` it never raises: a host it
    cannot register in is logged and nothing is owned (the app's owner check then names the problem)."""

    try:
        return _install(required)
    except Exception:
        if required:
            raise
        LOGGER.exception("AI Studio pack: Extra Samplers were not registered in ComfyUI.")
        return NameRegistration()


def _install(required: bool) -> NameRegistration:
    modules = _comfy_modules()
    if modules is None:
        if required:
            raise RuntimeError("Extra Samplers registration requires a ComfyUI host.")
        return NameRegistration()
    comfy_samplers, k_sampling = modules
    ksampler = getattr(comfy_samplers, "KSampler", None)
    available = [name for name in PACK_SAMPLERS if implementation(name) is not None]
    registration = register_names(
        k_sampling,
        {name: functools.partial(sampler_entry, name) for name in available},
        owner=OWNER,
        name_lists=(getattr(comfy_samplers, "KSAMPLER_NAMES", None), getattr(comfy_samplers, "SAMPLER_NAMES", None),
                    getattr(ksampler, "SAMPLERS", None)),
        prefix="sample_",
        label="sampler",
        required=required,
    )
    discard = getattr(ksampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS", None)
    if registration.published and isinstance(discard, set):
        discard.update(name for name in DISCARD_PENULTIMATE_SIGMA if name in registration.owned)
    return registration


def impact_scheduler_func(sampler_name: Any, scheduler: Any) -> Callable[..., Any] | None:
    """Impact Pack's ``scheduler_func_opt`` (``SCHEDULER_FUNC``) for a pack sampler that drops the penultimate sigma.

    ADetailer and the SAM3 restore-face pass run through Impact's ``FaceDetailer``, whose sigmas come from
    ``impact_sampling.calculate_sigmas``: its discard list is a literal (``dpm_2``, ``dpm_2_ancestral``,
    ``uni_pc``, ``uni_pc_bh2``) that never reads ``KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS``. For
    ``DISCARD_PENULTIMATE_SIGMA`` names in that set (the pack registered them) this returns the function
    ``separated_sample`` calls instead — ``scheduler_func(model, sampler_name, steps)``: the sigmas
    ``comfy.samplers.KSampler.calculate_sigmas`` gives for that set (the MODEL's ``model_sampling``, so the
    Extra Schedulers values count; one step more; the penultimate sigma dropped), as Forge's
    ``discard_next_to_last_sigma`` does on every pass, the ADetailer one included. Impact slices the denoise
    window from them as it does for its own discard samplers. Any other sampler -> None (Impact's own rule).
    """

    name = str(sampler_name)
    if name not in DISCARD_PENULTIMATE_SIGMA:
        return None
    modules = _comfy_modules()
    if modules is None:
        return None
    comfy_samplers = modules[0]
    discard = getattr(getattr(comfy_samplers, "KSampler", None), "DISCARD_PENULTIMATE_SIGMA_SAMPLERS", None)
    if not isinstance(discard, (set, frozenset)) or name not in discard:
        return None
    scheduler_name = str(scheduler)

    def discard_penultimate_sigmas(model: Any, _sampler_name: Any, steps: Any) -> Any:
        torch = importlib.import_module("torch")
        sigmas = comfy_samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler_name,
                                                 int(steps) + 1)
        return torch.cat([sigmas[:-2], sigmas[-1:]])

    discard_penultimate_sigmas.sampler_name, discard_penultimate_sigmas.scheduler = name, scheduler_name
    return discard_penultimate_sigmas


def owned_names() -> tuple[str, ...]:
    """The names whose ``sample_*`` function is the pack's and that KSampler offers — read live, so a
    provider that replaced a function after the pack's import is not reported as the pack's."""

    modules = _comfy_modules()
    if modules is None:
        return ()
    comfy_samplers, k_sampling = modules
    offered = getattr(getattr(comfy_samplers, "KSampler", None), "SAMPLERS", None)
    if not isinstance(offered, list):
        return ()
    return tuple(name for name in PACK_SAMPLERS
                 if entry_owner(getattr(k_sampling, "sample_" + name, None)) == OWNER and name in offered)
