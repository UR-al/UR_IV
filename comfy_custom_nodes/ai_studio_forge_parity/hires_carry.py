"""What a Hires pass keeps of the generation's MODEL patches (pack 1.7.0).

Two ways a MODEL patch can be lost before the Hires pass samples, and what the pack does:

* **Checkpoint/UNET override.** ``ForgeNeoHiresFix`` loads a fresh MODEL for the pass
  (``generation.py``). Forge keeps a generation's always-on state on that pass, so each pack
  feature that patches the MODEL registers how to put its patch on the new one::

      def carry(source_model, target_model, target_vae):   # -> MODEL (or None = unchanged)
          ...
      hires_carry.register(carry, key="extra_schedulers")

  ``ForgeNeoHiresFix`` calls ``apply(model, active_model, active_vae)`` after the override and
  its flow shift; the carries run in registration order, each on the previous one's result.
  ``target_vae`` is the pass's VAE (an override VAE included) for features that encode with it.
  A failing carry fails the pass: an override that silently drops a feature makes a different
  image. Detail Daemon and the guidance suite do not register (a documented gap: the override
  MODEL samples without them, as before 1.7.0).
* **A VAE-only override.** With only ``vae_name`` overridden the pass samples its own MODEL, but
  Forge reloads the model with the new VAE for the Hires pass, so a feature that encodes with the
  pass's VAE (Colorcraft's colour anchors) does it with the override VAE there. Such a feature
  registers with ``uses_vae=True``; ``ForgeNeoHiresFix`` calls
  ``apply_vae_change(model, active_model, active_vae)`` after its flow shift, which runs only those
  carries — also when the MODEL to sample is the incoming one itself (Shift 0) — so each prepares
  its patch again with the override VAE in place of the one the base VAE built.
* **A new ``model_sampling``.** Some features carry values on the MODEL's ``model_sampling``
  object (Extra Schedulers: a scheduler handler receives nothing else). A node that installs a
  new ``model_sampling`` drops them. The pack's own flow shift (``generation._patch_flow_shift``:
  ``ForgeNeoModelSamplingShift``, ``ForgeNeoHiresFix`` on **every** pass with Shift > 0, Anima
  Tile & Repair) copies the attribute names registered with ``register_sampling_attr`` from the
  old ``model_sampling`` to the new one (``copy_sampling_attrs``). Third-party ModelSampling*
  nodes cannot be fixed this way; the app places the carrier node after them.

No torch or ComfyUI import: registration happens when the feature modules are imported.
"""

from __future__ import annotations

from typing import Any, Callable

from .compat import own_attribute


# carry(source_model, target_model, target_vae) -> MODEL or None
Carry = Callable[[Any, Any, Any], Any]

_CARRIES: dict[str, Carry] = {}
_VAE_CARRIES: dict[str, None] = {}      # keys of the carries that encode with the pass's VAE
_SAMPLING_ATTRS: dict[str, None] = {}
_ABSENT = object()


def register(carry: Carry, *, key: str | None = None, uses_vae: bool = False) -> Carry:
    """Register ``carry`` for Hires checkpoint overrides; returns it (usable as a decorator).

    ``key`` defaults to the function's module and qualified name. Registering a key again
    replaces its function in place (a re-imported module keeps its position in the order).
    ``uses_vae``: the carry builds its patch with ``target_vae`` — ``apply_vae_change`` runs it
    for a Hires pass whose only override is the VAE.
    """

    if not callable(carry):
        raise TypeError(f"hires_carry.register needs a callable, not {type(carry).__name__}")
    name = str(key or f"{carry.__module__}.{carry.__qualname__}")
    _CARRIES[name] = carry
    if uses_vae:
        _VAE_CARRIES[name] = None
    else:
        _VAE_CARRIES.pop(name, None)
    return carry


def unregister(key: str) -> None:
    _CARRIES.pop(str(key), None)
    _VAE_CARRIES.pop(str(key), None)


def registered() -> tuple[str, ...]:
    """Registered carry keys in the order ``apply`` runs them."""

    return tuple(_CARRIES)


def registered_vae() -> tuple[str, ...]:
    """The keys ``apply_vae_change`` runs (registered with ``uses_vae``), in registration order."""

    return tuple(key for key in _CARRIES if key in _VAE_CARRIES)


def apply(source_model: Any, target_model: Any, target_vae: Any = None) -> Any:
    """Put every registered feature's patch from ``source_model`` onto ``target_model``.

    Returns the MODEL to sample (``target_model`` itself when nothing carried or when the two
    are the same MODEL). A carry that raises stops the pass with its key in the message.
    """

    if target_model is source_model:
        return target_model
    return _run(tuple(_CARRIES.items()), source_model, target_model, target_vae,
                "Hi-res checkpoint override: could not carry {key!r} onto the override model - {exc}")


def apply_vae_change(source_model: Any, target_model: Any, target_vae: Any) -> Any:
    """A Hires pass whose only override is the VAE: the ``uses_vae`` carries prepare their patch again with it.

    ``source_model`` is the MODEL the pass received (its patches were built with the base VAE);
    ``target_model`` the MODEL it samples — ``source_model`` itself (Shift 0) or its flow-shifted
    clone. Unlike ``apply`` this runs when the two are the same object. Returns the MODEL to sample.
    """

    carries = tuple((key, _CARRIES[key]) for key in registered_vae())
    return _run(carries, source_model, target_model, target_vae,
                "Hi-res VAE override: could not prepare {key!r} again with the override VAE - {exc}")


def _run(carries, source_model: Any, target_model: Any, target_vae: Any, message: str) -> Any:
    current = target_model
    for key, carry in carries:
        try:
            result = carry(source_model, current, target_vae)
        except Exception as exc:
            raise RuntimeError(message.format(key=key, exc=exc)) from exc
        if result is not None:
            current = result
    return current


def register_sampling_attr(name: str) -> str:
    """Keep ``model_sampling`` attribute ``name`` when the pack's flow shift replaces the object."""

    text = str(name)
    if not text:
        raise ValueError("A model_sampling carrier attribute needs a name.")
    _SAMPLING_ATTRS[text] = None
    return text


def unregister_sampling_attr(name: str) -> None:
    _SAMPLING_ATTRS.pop(str(name), None)


def sampling_attrs() -> tuple[str, ...]:
    return tuple(_SAMPLING_ATTRS)


def copy_sampling_attrs(source_sampling: Any, model: Any) -> Any:
    """Copy the registered carrier attributes of ``source_sampling`` onto ``model``'s ``model_sampling``.

    ``source_sampling`` is the object a node just replaced; ``model`` is that node's output MODEL,
    whose new ``model_sampling`` belongs to it alone (``ModelSamplingSD3.patch`` builds one per
    call). Only the object's own attributes count (never class attributes). Returns ``model``.
    Nothing to keep, or the same object, changes nothing; a carrier with nowhere to go raises.
    """

    if source_sampling is None or not _SAMPLING_ATTRS:
        return model
    values = {}
    for name in _SAMPLING_ATTRS:
        value = own_attribute(source_sampling, name, _ABSENT)
        if value is not _ABSENT:
            values[name] = value
    if not values:
        return model
    get_model_object = getattr(model, "get_model_object", None)
    target = get_model_object("model_sampling") if callable(get_model_object) else None
    if target is None:
        raise RuntimeError(
            "The flow-shifted MODEL has no model_sampling to keep "
            f"{', '.join(sorted(values))} on."
        )
    if target is source_sampling:
        return model
    for name, value in values.items():
        setattr(target, name, value)
    return model
