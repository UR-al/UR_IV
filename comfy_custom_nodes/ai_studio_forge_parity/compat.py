"""Small lazy compatibility layer for the AI Studio ComfyUI node pack.

The app owns this node pack and also imports parts of it in its normal Python
test process, where the large ComfyUI runtime is not installed.  Import-time
registration therefore only probes Comfy's lightweight sampler registry; all
torch and model work remains behind runtime calls.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import MutableMapping
from typing import Any, Callable, Iterable, Mapping, NamedTuple


LOGGER = logging.getLogger("ai_studio_forge_parity")

BETA57_SCHEDULER = "beta57"
# Tag on a function the pack put into a ComfyUI registry (a ``comfy.k_diffusion.sampling``
# ``sample_*`` attribute, a ``SchedulerHandler``'s function); value = the registering feature.
# A registered name whose entry lacks the pack's tag belongs to someone else (register_names).
OWNER_ATTR = "_ai_studio_forge_parity_owner"
# Optional COMBO input on a node whose feature registers names: its choices are the names the
# pack owns in this ComfyUI (the app reads them from /object_info and never sets the input).
OWNED_NAMES_INPUT = "owned_names"
NO_OWNED_NAMES = "(none)"
# comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE on hosts that have no such enum.
SAMPLER_SAMPLE_FALLBACK = "sampler_sample"


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


# --- names the pack registers in ComfyUI tables (pack 1.7.0) ----------------------------------
_MISSING = object()
_FOREIGN_LOGGED: set[tuple[str, str]] = set()


class NameRegistration(NamedTuple):
    """What ``register_names`` did, in the order of the requested names."""

    owned: tuple[str, ...] = ()      # the pack's entry is in place (put there now or earlier)
    inserted: tuple[str, ...] = ()   # put into the registry by this call
    foreign: tuple[str, ...] = ()    # held by another provider: left as it is, not published
    published: bool = False          # the owned names are in every choice list found


def mark_owned(function: Callable[..., Any], owner: str) -> Callable[..., Any]:
    """Tag ``function`` as registered by the pack feature ``owner`` (``OWNER_ATTR``)."""

    setattr(function, OWNER_ATTR, str(owner))
    return function


def own_attribute(value: Any, name: str, default: Any = None) -> Any:
    """``value``'s own attribute ``name``, else ``default``.

    Never a class attribute or one a mock invents on reading: the instance ``__dict__`` is
    asked, and plain ``getattr`` only for objects without one (a NamedTuple field such as
    ``SchedulerHandler.handler``, slots).
    """

    try:
        namespace = vars(value)
    except TypeError:
        return getattr(value, name, default)
    return namespace.get(name, default)


def entry_owner(entry: Any) -> str | None:
    """The pack feature that registered ``entry`` — a function or a ``SchedulerHandler`` — else None."""

    for candidate in (entry, own_attribute(entry, "handler")):
        if candidate is None:
            continue
        owner = own_attribute(candidate, OWNER_ATTR)
        if isinstance(owner, str):
            return owner
    return None


def _describe_entry(entry: Any) -> str:
    target = own_attribute(entry, "handler", entry)
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None) or type(target).__name__
    return f"{module}.{name}" if module else str(name)


class _Slots:
    """A registry addressed by name: a dict, or an object's ``<prefix><name>`` attributes."""

    def __init__(self, registry: Any, prefix: str):
        self.registry, self.prefix = registry, prefix
        self.is_mapping = isinstance(registry, MutableMapping)

    def get(self, name: str) -> Any:
        if self.is_mapping:
            return self.registry.get(name, _MISSING)
        return getattr(self.registry, self.prefix + name, _MISSING)

    def put(self, name: str, entry: Any) -> None:
        if self.is_mapping:
            self.registry[name] = entry
        else:
            setattr(self.registry, self.prefix + name, entry)

    def remove(self, name: str) -> None:
        if self.is_mapping:
            self.registry.pop(name, None)
        else:
            try:
                delattr(self.registry, self.prefix + name)
            except AttributeError:
                pass


def _choice_lists(candidates: Iterable[Any]) -> list[list[Any]]:
    lists: list[list[Any]] = []
    for values in candidates:
        if isinstance(values, list) and all(values is not known for known in lists):
            lists.append(values)
    return lists


def register_names(
    registry: Any,
    entries: Mapping[str, Callable[[], Any]],
    *,
    owner: str,
    name_lists: Iterable[Any] = (),
    prefix: str = "",
    label: str = "name",
    required: bool = False,
) -> NameRegistration:
    """Register the pack's ``entries`` in a ComfyUI table without ever replacing another provider's.

    ``registry`` is a dict keyed by name (``comfy.samplers.SCHEDULER_HANDLERS``) or an object whose
    ``<prefix><name>`` attributes are the entries (``comfy.k_diffusion.sampling`` with
    ``prefix="sample_"``). ``entries`` maps each name to a factory of its entry; the entry must carry
    ``owner`` (``mark_owned`` on the function, or on a ``SchedulerHandler``'s function). Per name, in
    order: a free slot gets the factory's entry; a slot holding ``owner``'s entry is kept as it is
    (registering twice changes nothing); a slot holding anything else is foreign — left untouched,
    logged once, and never published by the pack (the beta57 rule, ``install_forge_scheduler_support``,
    which this generalises with an owner check).

    ``name_lists`` are the published choice lists (``SCHEDULER_NAMES``, ``KSampler.SCHEDULERS``,
    ``KSAMPLER_NAMES`` …): every distinct ``list`` object gets each owned name appended once. When
    owned names exist but no candidate is a list, nothing can be offered: the names inserted by this
    call are taken out again and nothing is owned — or, with ``required``, ``RuntimeError``.
    """

    slots = _Slots(registry, prefix)
    owned: list[str] = []
    inserted: list[str] = []
    foreign: list[str] = []
    for name, factory in entries.items():
        current = slots.get(name)
        if current is _MISSING:
            entry = factory()
            if entry_owner(entry) != owner:
                raise TypeError(
                    f"The pack's {label} {name!r} entry is not tagged with owner {owner!r} (mark_owned)."
                )
            slots.put(name, entry)
            inserted.append(name)
            owned.append(name)
        elif entry_owner(current) == owner:
            owned.append(name)
        else:
            foreign.append(name)
            if (label, name) not in _FOREIGN_LOGGED:
                _FOREIGN_LOGGED.add((label, name))
                LOGGER.warning(
                    "AI Studio pack: ComfyUI %s %r is already provided by %s; the pack leaves it "
                    "as it is, and the app refuses graphs that need the pack's %r.",
                    label, name, _describe_entry(current), name,
                )
    lists = _choice_lists(name_lists)
    if owned and not lists:
        for name in inserted:
            slots.remove(name)
        if required:
            raise RuntimeError(
                f"This ComfyUI version does not publish {label} choices; the pack cannot add "
                f"{', '.join(owned)}."
            )
        return NameRegistration(foreign=tuple(foreign))
    for values in lists:
        for name in owned:
            if name not in values:
                values.append(name)
    return NameRegistration(
        owned=tuple(owned), inserted=tuple(inserted), foreign=tuple(foreign),
        published=bool(owned),
    )


def owned_names_input(names: Iterable[str], *, tooltip: str | None = None) -> tuple[list[str], dict[str, Any]]:
    """Spec of the optional ``OWNED_NAMES_INPUT`` COMBO: the names the pack owns, else ``(none)``.

    Informational: the app reads the choices from ``/object_info`` to refuse a graph whose sampler
    or scheduler name another provider owns (or an older pack lacks); nothing sets the input.
    """

    choices = list(dict.fromkeys(str(name) for name in names)) or [NO_OWNED_NAMES]
    return (choices, {
        "default": choices[0],
        "tooltip": tooltip or (
            "Names this node pack registered in ComfyUI (read by AI Studio; leave as it is). "
            f"{NO_OWNED_NAMES} = another provider owns them or ComfyUI has no such table."
        ),
    })


def sampler_sample_wrapper_type() -> str:
    """``comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE`` (``SAMPLER_SAMPLE_FALLBACK`` outside ComfyUI)."""

    try:
        patcher_extension = importlib.import_module("comfy.patcher_extension")
    except Exception:
        return SAMPLER_SAMPLE_FALLBACK
    wrappers = getattr(patcher_extension, "WrappersMP", None)
    return str(getattr(wrappers, "SAMPLER_SAMPLE", SAMPLER_SAMPLE_FALLBACK))


def make_first(patched: Any, wrapper_type: str, key: str) -> bool:
    """Move the wrappers registered under ``key`` to the front of ``patched.wrappers[wrapper_type]``.

    ``comfy.patcher_extension.get_all_wrappers`` lists the wrappers in the dict's order and
    ``WrapperExecutor`` calls the first one first (outermost). ``ModelPatcher.clone`` gives the
    clone its own per-type dicts, so the model that was cloned keeps its order. Returns whether
    ``key`` was found. Whoever calls it last is outermost: CNS re-asserts itself at every sampler
    node, so CNS stays outside the pack's other SAMPLER_SAMPLE wrappers.
    """

    wrappers = getattr(patched, "wrappers", None)
    registered = wrappers.get(wrapper_type) if isinstance(wrappers, dict) else None
    if not isinstance(registered, dict) or key not in registered:
        return False
    wrappers[wrapper_type] = {
        key: registered[key],
        **{name: items for name, items in registered.items() if name != key},
    }
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
