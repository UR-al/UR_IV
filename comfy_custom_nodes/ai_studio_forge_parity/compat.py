"""Small lazy compatibility layer for the AI Studio ComfyUI node pack.

The app owns this node pack and also imports parts of it in its normal Python
test process, where the large ComfyUI runtime is not installed.  Import-time
registration therefore only probes Comfy's lightweight sampler registry; all
torch and model work remains behind runtime calls.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Iterable


BETA57_SCHEDULER = "beta57"


class MissingComfyProvider(RuntimeError):
    """Raised when an explicitly enabled feature has no runtime provider."""


def require_torch():
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on the host runtime
        raise RuntimeError(
            "This Forge-parity node needs torch and must run inside ComfyUI."
        ) from exc


def json_object(value: Any, feature: str) -> dict[str, Any]:
    """Parse a node's ``settings_json`` input into a dict (``{}`` when empty)."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{feature} settings_json must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{feature} settings_json must be a JSON object.")
    return parsed


def comfy_nodes():
    try:
        return importlib.import_module("nodes")
    except Exception as exc:  # pragma: no cover - depends on the host runtime
        raise RuntimeError(
            "ComfyUI's nodes module is unavailable. Install this pack under "
            "ComfyUI/custom_nodes and restart ComfyUI."
        ) from exc


def folder_paths_module():
    try:
        return importlib.import_module("folder_paths")
    except Exception as exc:  # pragma: no cover - depends on the host runtime
        raise RuntimeError("ComfyUI folder_paths is unavailable.") from exc


def node_mapping() -> dict[str, type]:
    mapping = getattr(comfy_nodes(), "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("ComfyUI NODE_CLASS_MAPPINGS is unavailable.")
    return mapping


def provider_class(name: str, *, feature: str | None = None) -> type:
    cls = node_mapping().get(name)
    if cls is None:
        label = feature or name
        raise MissingComfyProvider(
            f"{label} is enabled, but required ComfyUI provider node "
            f"{name!r} is not installed. Install/update the provider and "
            "restart ComfyUI."
        )
    return cls


def provider(name: str, *, feature: str | None = None) -> Any:
    return provider_class(name, feature=feature)()


def node_result(value: Any) -> tuple[Any, ...]:
    """Normalize legacy tuples, output dictionaries, and io.NodeOutput."""

    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    elif hasattr(value, "result"):
        result = getattr(value, "result")
        if result is not None:
            value = result
        elif hasattr(value, "args"):
            value = getattr(value, "args")
    elif hasattr(value, "args") and not isinstance(value, (tuple, list)):
        value = getattr(value, "args")

    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def invoke_provider(
    name: str,
    *,
    method: str | None = None,
    feature: str | None = None,
    args: Iterable[Any] = (),
    kwargs: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    instance = provider(name, feature=feature)
    function_name = method or getattr(instance, "FUNCTION", None)
    function = getattr(instance, function_name, None) if function_name else None
    if not callable(function):
        raise RuntimeError(
            f"Provider {name!r} does not expose the expected "
            f"{function_name or 'node execution'} method."
        )
    return node_result(function(*tuple(args), **(kwargs or {})))


def filename_choices(kind: str, *, fallback: str = "None") -> list[str]:
    """Return a non-empty Comfy combo list, including tests outside ComfyUI."""

    try:
        values = list(folder_paths_module().get_filename_list(kind))
    except Exception:
        values = []
    return values or [fallback]


def sampler_names() -> list[str]:
    try:
        comfy_samplers = importlib.import_module("comfy.samplers")
        return list(comfy_samplers.KSampler.SAMPLERS)
    except Exception:
        return ["euler"]


def scheduler_names() -> list[str]:
    try:
        comfy_samplers = importlib.import_module("comfy.samplers")
        return list(comfy_samplers.KSampler.SCHEDULERS)
    except Exception:
        return ["simple"]


def install_forge_scheduler_support(*, required: bool = False) -> bool:
    """Register Forge/RES4LYF's exact ``beta57`` schedule in ComfyUI.

    Forge's ``Beta57 (RES4LYF)`` is not Comfy's stock ``beta`` preset: both
    use the same beta-distribution PPF scheduler, but Beta57 pins alpha=0.5
    and beta=0.7.  Registering it in Comfy's scheduler table keeps the same
    meaning in KSampler, Hires, ADetailer, and SAM3 instead of approximating it
    with ``beta`` (whose stock parameters are 0.6/0.6).
    """

    try:
        comfy_samplers = importlib.import_module("comfy.samplers")
    except ModuleNotFoundError as exc:
        if required:
            raise RuntimeError(
                "Beta57 scheduler registration requires a ComfyUI host."
            ) from exc
        return False

    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    scheduler_handler = getattr(comfy_samplers, "SchedulerHandler", None)
    beta_scheduler = getattr(comfy_samplers, "beta_scheduler", None)
    if not isinstance(handlers, dict) or not callable(scheduler_handler):
        if required:
            raise RuntimeError(
                "This ComfyUI version has no extensible scheduler registry."
            )
        return False

    inserted_handler = False
    if BETA57_SCHEDULER not in handlers:
        if not callable(beta_scheduler):
            if required:
                raise RuntimeError(
                    "This ComfyUI version has no beta scheduler implementation."
                )
            return False

        def beta57_scheduler(model_sampling, steps):
            return beta_scheduler(
                model_sampling, steps, alpha=0.5, beta=0.7,
            )

        beta57_scheduler._ai_studio_res4lyf_beta57 = True
        handlers[BETA57_SCHEDULER] = scheduler_handler(beta57_scheduler)
        inserted_handler = True

    published_lists: list[list[str]] = []
    scheduler_names_value = getattr(comfy_samplers, "SCHEDULER_NAMES", None)
    if isinstance(scheduler_names_value, list):
        published_lists.append(scheduler_names_value)
    ksampler = getattr(comfy_samplers, "KSampler", None)
    ksampler_names = getattr(ksampler, "SCHEDULERS", None)
    if isinstance(ksampler_names, list) and all(
        ksampler_names is not values for values in published_lists
    ):
        published_lists.append(ksampler_names)
    if not published_lists:
        if required:
            raise RuntimeError(
                "This ComfyUI version does not publish scheduler choices."
            )
        if inserted_handler:
            handlers.pop(BETA57_SCHEDULER, None)
        return False
    for values in published_lists:
        if BETA57_SCHEDULER not in values:
            values.append(BETA57_SCHEDULER)
    return True


def clone_model(model: Any, feature: str) -> Any:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise RuntimeError(f"{feature} requires a ComfyUI MODEL input.")
    return clone()


def is_disabled_choice(value: Any) -> bool:
    return str(value or "").strip().casefold() in {
        "",
        "none",
        "off",
        "use current",
        "use same checkpoint",
        "use same choices",
        "(none)",
    }


# Inside ComfyUI this runs while custom nodes are imported, before /object_info
# is served.  In the desktop app's lightweight test process Comfy is absent and
# the registration is intentionally a no-op.
install_forge_scheduler_support(required=False)
