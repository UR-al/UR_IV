r"""Pack 1.7.0 foundation shared by the four sam-extra ports.

* ``guidance_common`` coordination names and helpers (sub-step marker, step-offset keyword,
  re-published sigma lists, ``declares_keyword``) — the values and the rules are sam-extra's, so
  they are checked against the **installed** extension (AST only: the extension modules that hold
  them import torch or Forge). The pack's code is its own; the extension's functions are cut out
  of its source and run here only as the oracle.
* ``compat`` registration helpers (never-overwrite ``register_names``, ``owned_names_input``,
  ``make_first``, ``sampler_sample_wrapper_type``) with fake ComfyUI tables.
* ``guidance_cns`` ``_forge_neo_base`` so SPEED can see the real ``sample_*`` through CNS.
* ``hires_carry``: the Hires checkpoint-override carry registry, and the ``model_sampling``
  carrier copy in the pack's flow shift that keeps carried values on every Hires pass with
  Shift > 0 (PLAN R2/R2b) — with fake ModelPatchers.
* The Dy sub-steps-off flag ``ForgeNeoKSamplerCNS`` sets for ComfyUI Spectrum and the legacy
  SPEED provider only (PLAN R3).
* No pack module imports sam-extra (GPL): computation is re-implemented, never imported.

Everything above needs neither torch nor ComfyUI. One more class runs the flow-shift carrier
copy on the real ComfyUI runtime (``ModelPatcher``, ``ModelSamplingSD3``, ``common_upscale``) —
skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set; run it with the portable python, repo first
on ``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model
management loads; no server, no GPU)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_pack_foundation', '-v'])"
"""
from __future__ import annotations

import ast
import functools
import inspect
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # real-ComfyUI tier only: force --cpu before anything loads model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from comfy_custom_nodes.ai_studio_forge_parity import (  # noqa: E402
    compat,
    generation,
    guidance_cns,
    guidance_common,
    hires_carry,
)
from tests._optional_deps import bind_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import EXT_ROOT, requires_extension  # noqa: E402

torch = None  # filled by bind_torch() in the real-ComfyUI class

PACK_DIR = Path(__file__).resolve().parents[1] / "comfy_custom_nodes" / "ai_studio_forge_parity"


def _ext_constant(relative: str, name: str) -> Any:
    """A module-level literal of the installed extension, read by AST (nothing is imported)."""
    tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in item.targets
        ):
            return ast.literal_eval(item.value)
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == name:
            return ast.literal_eval(item.value)
    raise AssertionError(f"{relative} has no module-level {name}")


def _ext_functions(relative: str, *names: str, **namespace: Any) -> dict[str, Callable[..., Any]]:
    """Top-level functions cut out of an extension file and run alone (the file's imports are not)."""
    tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
    nodes = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name in names]
    missing = set(names) - {item.name for item in nodes}
    if missing:
        raise AssertionError(f"{relative} has no function(s) {sorted(missing)}")
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{relative}:{','.join(names)}", "exec"), scope)
    return {name: scope[name] for name in names}


class _Taggable:
    """A sigma-list stand-in that takes attributes like a torch tensor."""

    def __init__(self, label: str):
        self.label = label

    def __repr__(self):
        return f"<{self.label}>"


def _substep_cases() -> list[dict[str, Any]]:
    marker = guidance_common.SUBSTEP_MARKER
    return [
        {},
        {"model_options": {}},
        {"model_options": None, "c": None},
        {"model_options": {"transformer_options": {marker: "dy"}}},
        {"model_options": {"transformer_options": {marker: "smea"}}},
        {"model_options": {"transformer_options": {marker: ""}}},
        {"model_options": {"transformer_options": {marker: None}}},
        {"model_options": {"transformer_options": {"other": 1}}},
        # empty model options fall back to the wrapper's
        {"model_options": {"transformer_options": {}}, "c": {"transformer_options": {marker: "dy"}}},
        {"model_options": {}, "c": {"transformer_options": {marker: "smea"}}},
        {"c": {"transformer_options": {marker: "dy"}}},
        {"c": {"transformer_options": {}}},
        {"c": {}},
        # non-empty model options win even without the marker
        {"model_options": {"transformer_options": {"x": 1}}, "c": {"transformer_options": {marker: "dy"}}},
        {"model_options": {"transformer_options": {marker: "dy"}}, "c": {"transformer_options": {}}},
        {"model_options": {"transformer_options": {marker: 0}}, "c": {"transformer_options": {marker: 1}}},
    ]


def _signature_corpus() -> list[Callable[..., Any]]:
    offset = guidance_common.STEP_OFFSET_KWARG

    def plain(model, x, sigmas, extra_args=None, callback=None, disable=None):
        return x

    def kwargs_only(model, x, sigmas, **kwargs):
        return x

    namespace: dict[str, Any] = {}
    exec(
        f"def named(model, x, sigmas, extra_args=None, {offset}=0, **kwargs):\n    return x\n"
        f"def keyword_only(model, x, sigmas, *, {offset}=0):\n    return x\n"
        f"def positional_only({offset}, /, model, x):\n    return x\n"
        f"def var_positional(*{offset}):\n    return None\n"
        f"def var_keyword(**{offset}):\n    return None\n",
        namespace,
    )

    @functools.wraps(namespace["named"])
    def wrapped_named(*args, **kwargs):
        return namespace["named"](*args, **kwargs)

    return [plain, kwargs_only, *(namespace[name] for name in (
        "named", "keyword_only", "positional_only", "var_positional", "var_keyword",
    )), wrapped_named, len]


@requires_extension
class CoordinationNamesMatchInstalledExtensionTests(unittest.TestCase):
    """The pack writes and reads sam-extra's names with sam-extra's rules."""

    def test_names_are_the_extension_values(self):
        for relative, name in (
            ("sam3ext/extra_samplers/common.py", "SUBSTEP_MARKER"),
            ("sam3ext/speed/runner.py", "STEP_OFFSET_KWARG"),
            ("sam3ext/guidance/sigmas.py", "REPUBLISHED_FROM"),
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(guidance_common, name), _ext_constant(relative, name))
        # the guidance script's own fallback copy of the marker (used when the import fails)
        self.assertEqual(
            _ext_constant_in_try("scripts/anima_safe_pag.py", "EXTRA_SAMPLER_SUBSTEP"),
            guidance_common.SUBSTEP_MARKER,
        )

    def test_republished_tags_match_the_extension(self):
        ext = _ext_functions(
            "sam3ext/guidance/sigmas.py", "mark_republished", "republished_from",
            REPUBLISHED_FROM=guidance_common.REPUBLISHED_FROM,
        )
        packs = (guidance_common.mark_republished, guidance_common.republished_from)
        exts = (ext["mark_republished"], ext["republished_from"])
        results = []
        for mark, read in (packs, exts):
            run, copy, copy_of_copy = _Taggable("run"), _Taggable("copy"), _Taggable("copy2")
            frozen = (1.0, 0.5, 0.0)
            outcome = [
                mark(copy, run) is copy,
                read(copy) is run,
                mark(copy_of_copy, copy) is copy_of_copy,
                read(copy_of_copy) is run,          # a copy of a copy points at the run's list
                read(run) is None,
                mark(frozen, run) is frozen,        # takes no attribute: returned untagged
                read(frozen) is None,
                mark(_Taggable("x"), frozen) is not None,
            ]
            results.append(outcome)
        self.assertEqual(results[0], results[1])
        self.assertTrue(all(results[0]))

    def test_substep_lookup_matches_the_extension(self):
        ext = _ext_functions(
            "scripts/anima_safe_pag.py", "_is_extra_sampler_substep",
            EXTRA_SAMPLER_SUBSTEP=guidance_common.SUBSTEP_MARKER,
        )["_is_extra_sampler_substep"]
        for case in _substep_cases():
            with self.subTest(case=case):
                self.assertIs(guidance_common.is_extra_sampler_substep(case), ext(case))

    def test_declared_keyword_rule_matches_the_extension(self):
        ext = _ext_functions("sam3ext/speed/runner.py", "_declares", inspect=inspect)["_declares"]
        name = guidance_common.STEP_OFFSET_KWARG
        for function in _signature_corpus():
            with self.subTest(function=getattr(function, "__name__", function)):
                self.assertIs(guidance_common.declares_keyword(function, name), ext(function, name))


def _ext_constant_in_try(relative: str, name: str) -> Any:
    """``NAME = literal`` anywhere in the file (the guidance script sets its fallback in ``except``)."""
    tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
    for item in ast.walk(tree):
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in item.targets
        ) and isinstance(item.value, ast.Constant):
            return item.value.value
    raise AssertionError(f"{relative} has no literal {name}")


class CoordinationHelperTests(unittest.TestCase):
    """Pack-only behaviour of the helpers (no extension needed)."""

    def test_substep_marker_is_read_from_post_cfg_and_wrapper_args(self):
        marker = guidance_common.SUBSTEP_MARKER
        self.assertTrue(guidance_common.is_extra_sampler_substep(
            {"model_options": {"transformer_options": {marker: "dy"}}}))
        self.assertTrue(guidance_common.is_extra_sampler_substep(
            {"input": None, "c": {"transformer_options": {marker: "smea"}}}))
        self.assertFalse(guidance_common.is_extra_sampler_substep(
            {"model_options": {"transformer_options": {"forge_neo_pass": "base"}}}))
        for not_args in (None, [], "dy", ({marker: "dy"},)):
            with self.subTest(args=not_args):
                self.assertFalse(guidance_common.is_extra_sampler_substep(not_args))
        # a non-dict transformer_options never counts as marked
        self.assertFalse(guidance_common.is_extra_sampler_substep(
            {"model_options": {"transformer_options": [marker]}}))

    def test_sampler_base_follows_cns_and_functools_links(self):
        offset = guidance_common.STEP_OFFSET_KWARG
        namespace: dict[str, Any] = {}
        exec(f"def sample_dy(model, x, sigmas, extra_args=None, {offset}=0):\n    return x\n", namespace)
        sample_dy = namespace["sample_dy"]

        def sample_plain(model, x, sigmas, extra_args=None):
            return x

        cns = guidance_cns.cns_sampler_function(sample_dy, kd_module=SimpleNamespace())
        self.assertIs(getattr(cns, guidance_common.SAMPLER_BASE_ATTR), sample_dy)
        self.assertFalse(hasattr(cns, "__wrapped__"))   # Path B for a second CNS, as before
        self.assertIs(guidance_common.sampler_base(cns), sample_dy)
        self.assertTrue(guidance_common.declares_keyword(cns, offset))
        # stacked CNS, then a functools.wraps wrapper around it
        stacked = guidance_cns.cns_sampler_function(cns, kd_module=SimpleNamespace())

        @functools.wraps(stacked)
        def outer(*args, **kwargs):
            return stacked(*args, **kwargs)

        self.assertIs(guidance_common.sampler_base(outer), sample_dy)
        self.assertTrue(guidance_common.declares_keyword(outer, offset))
        plain_cns = guidance_cns.cns_sampler_function(sample_plain, kd_module=SimpleNamespace())
        self.assertFalse(guidance_common.declares_keyword(plain_cns, offset))
        # without the link the CNS function's own (**kwargs) signature is all there is
        self.assertFalse(guidance_common.declares_keyword(
            lambda model, x, sigmas, **kwargs: x, offset))

    def test_sampler_base_survives_cycles_and_attribute_inventing_objects(self):
        def first():
            return None

        def second():
            return None

        setattr(first, guidance_common.SAMPLER_BASE_ATTR, second)
        setattr(second, guidance_common.SAMPLER_BASE_ATTR, first)
        self.assertIs(guidance_common.sampler_base(first), second)
        invented = mock.Mock()   # every attribute read invents a new callable child
        self.assertIsNotNone(guidance_common.sampler_base(invented))
        with mock.patch.object(guidance_common.inspect, "signature", side_effect=ValueError("no")):
            self.assertFalse(guidance_common.declares_keyword(first, "x"))

    def test_republished_copy_points_at_the_first_list(self):
        run, copy, again = _Taggable("run"), _Taggable("copy"), _Taggable("again")
        guidance_common.mark_republished(copy, run)
        guidance_common.mark_republished(again, copy)
        self.assertIs(guidance_common.republished_from(again), run)
        self.assertIsNone(guidance_common.republished_from(run))
        self.assertEqual(
            getattr(copy, guidance_common.REPUBLISHED_FROM), run,
        )


class _Handler(NamedTuple):
    """comfy.samplers.SchedulerHandler's shape (a NamedTuple: no __dict__)."""

    handler: Callable[..., Any]
    use_ms: bool = True


def _scheduler_entry(owner: str, label: str):
    def factory():
        def schedule(model_sampling, steps):
            return (label, steps)

        return _Handler(compat.mark_owned(schedule, owner))

    return factory


def _sampler_entry(owner: str, label: str):
    def factory():
        def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return label

        return compat.mark_owned(sample, owner)

    return factory


class RegisterNamesTests(unittest.TestCase):
    def setUp(self):
        compat._FOREIGN_LOGGED.clear()
        self.addCleanup(compat._FOREIGN_LOGGED.clear)

    def _scheduler_tables(self, *, same_list: bool = False):
        names = ["simple", "karras"]
        handlers = {"simple": _Handler(lambda ms, steps: None), "karras": _Handler(lambda ms, steps: None)}
        ksampler_names = names if same_list else list(names)
        return handlers, names, SimpleNamespace(SCHEDULERS=ksampler_names)

    def test_free_names_are_inserted_and_published_once(self):
        handlers, names, ksampler = self._scheduler_tables()
        entries = {name: _scheduler_entry("xs", name) for name in ("cosine", "phi")}
        report = compat.register_names(
            handlers, entries, owner="xs", name_lists=(names, ksampler.SCHEDULERS), label="scheduler",
        )
        self.assertEqual(report, compat.NameRegistration(
            owned=("cosine", "phi"), inserted=("cosine", "phi"), foreign=(), published=True))
        self.assertEqual(names, ["simple", "karras", "cosine", "phi"])
        self.assertEqual(ksampler.SCHEDULERS, ["simple", "karras", "cosine", "phi"])
        self.assertEqual(handlers["phi"].handler("ms", 7), ("phi", 7))
        first = handlers["cosine"]
        again = compat.register_names(
            handlers, entries, owner="xs", name_lists=(names, ksampler.SCHEDULERS), label="scheduler",
        )
        self.assertEqual(again.owned, ("cosine", "phi"))
        self.assertEqual(again.inserted, ())
        self.assertIs(handlers["cosine"], first)      # the pack's own entry is kept, not rebuilt
        self.assertEqual(names.count("cosine"), 1)
        self.assertEqual(ksampler.SCHEDULERS.count("phi"), 1)

    def test_one_list_object_published_under_two_names_gets_each_name_once(self):
        handlers, names, ksampler = self._scheduler_tables(same_list=True)
        compat.register_names(
            handlers, {"laplace": _scheduler_entry("xs", "laplace")}, owner="xs",
            name_lists=(names, ksampler.SCHEDULERS, names),
        )
        self.assertIs(ksampler.SCHEDULERS, names)
        self.assertEqual(names.count("laplace"), 1)

    def test_a_foreign_entry_is_never_overwritten_nor_published(self):
        handlers, names, ksampler = self._scheduler_tables()
        foreign = _Handler(lambda ms, steps: "theirs")
        handlers["laplace"] = foreign
        other_owner = _scheduler_entry("someone_else", "custom")()
        handlers["custom"] = other_owner
        entries = {name: _scheduler_entry("xs", name) for name in ("laplace", "custom", "phi")}
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            report = compat.register_names(
                handlers, entries, owner="xs", name_lists=(names, ksampler.SCHEDULERS),
                label="scheduler",
            )
        self.assertEqual(report.owned, ("phi",))
        self.assertEqual(report.foreign, ("laplace", "custom"))
        self.assertIs(handlers["laplace"], foreign)
        self.assertIs(handlers["custom"], other_owner)
        self.assertNotIn("laplace", names)
        self.assertNotIn("custom", ksampler.SCHEDULERS)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("'laplace'", logs.output[0])
        # logged once per name
        with self.assertNoLogs("ai_studio_forge_parity", level="WARNING"):
            compat.register_names(
                handlers, entries, owner="xs", name_lists=(names, ksampler.SCHEDULERS),
                label="scheduler",
            )

    def test_module_attribute_registry_with_prefix(self):
        def sample_euler(model, x, sigmas):
            return "stock"

        kd = types.ModuleType("fake_kd")
        kd.sample_euler = sample_euler
        kd.sample_er_sde_reverse_time = lambda *a: "foreign"
        ksampler_names = ["euler"]
        sampler_names = ksampler_names + ["ddim"]
        entries = {
            name: _sampler_entry("xsa", name)
            for name in ("er_sde_reverse_time", "dpmpp_4m_sde", "euler")
        }
        with self.assertLogs("ai_studio_forge_parity", level="WARNING"):
            report = compat.register_names(
                kd, entries, owner="xsa", prefix="sample_",
                name_lists=(ksampler_names, sampler_names, SimpleNamespace(SAMPLERS=sampler_names).SAMPLERS),
                label="sampler",
            )
        self.assertEqual(report.owned, ("dpmpp_4m_sde",))
        self.assertEqual(report.foreign, ("er_sde_reverse_time", "euler"))
        self.assertIs(kd.sample_euler, sample_euler)
        self.assertEqual(kd.sample_dpmpp_4m_sde(None, None, None), "dpmpp_4m_sde")
        self.assertEqual(compat.entry_owner(kd.sample_dpmpp_4m_sde), "xsa")
        self.assertEqual(ksampler_names, ["euler", "dpmpp_4m_sde"])
        self.assertEqual(sampler_names, ["euler", "ddim", "dpmpp_4m_sde"])

    def test_without_a_choice_list_the_inserted_names_are_taken_out(self):
        handlers = {"simple": object()}
        report = compat.register_names(
            handlers, {"cosine": _scheduler_entry("xs", "cosine")}, owner="xs",
            name_lists=((), None, ("simple",)),
        )
        self.assertEqual(report, compat.NameRegistration())
        self.assertNotIn("cosine", handlers)
        with self.assertRaisesRegex(RuntimeError, "does not publish scheduler choices"):
            compat.register_names(
                handlers, {"cosine": _scheduler_entry("xs", "cosine")}, owner="xs",
                name_lists=(), label="scheduler", required=True,
            )
        self.assertNotIn("cosine", handlers)
        kd = types.ModuleType("fake_kd")
        compat.register_names(kd, {"x": _sampler_entry("xsa", "x")}, owner="xsa", prefix="sample_")
        self.assertFalse(hasattr(kd, "sample_x"))

    def test_only_foreign_names_need_no_choice_list(self):
        handlers = {"laplace": object()}
        with self.assertLogs("ai_studio_forge_parity", level="WARNING"):
            report = compat.register_names(
                handlers, {"laplace": _scheduler_entry("xs", "laplace")}, owner="xs", required=True,
            )
        self.assertEqual(report, compat.NameRegistration(foreign=("laplace",)))

    def test_an_untagged_factory_is_a_programming_error(self):
        with self.assertRaisesRegex(TypeError, "mark_owned"):
            compat.register_names(
                {}, {"cosine": lambda: _Handler(lambda ms, steps: None)}, owner="xs", name_lists=([],),
            )

    def test_entry_owner_reads_only_real_tags(self):
        self.assertIsNone(compat.entry_owner(mock.Mock()))
        self.assertIsNone(compat.entry_owner(SimpleNamespace(handler=mock.Mock())))
        self.assertIsNone(compat.entry_owner(object()))
        self.assertIsNone(compat.entry_owner(None))
        tagged = compat.mark_owned(lambda: None, "xs")
        self.assertEqual(compat.entry_owner(_Handler(tagged)), "xs")
        self.assertEqual(compat.entry_owner(SimpleNamespace(handler=tagged)), "xs")

    def test_wrapped_function_keeps_its_own_tag(self):
        inner = compat.mark_owned(lambda *a: None, "someone_else")

        @functools.wraps(inner)
        def outer(*args):
            return inner(*args)

        compat.mark_owned(outer, "xsa")
        self.assertEqual(compat.entry_owner(outer), "xsa")
        self.assertEqual(compat.entry_owner(inner), "someone_else")


class OwnedNamesInputTests(unittest.TestCase):
    def test_choices_default_and_none(self):
        choices, options = compat.owned_names_input(["cosine", "phi", "cosine"])
        self.assertEqual(choices, ["cosine", "phi"])
        self.assertEqual(options["default"], "cosine")
        self.assertIn("tooltip", options)
        self.assertEqual(compat.owned_names_input(()), ([compat.NO_OWNED_NAMES], {
            "default": compat.NO_OWNED_NAMES, "tooltip": compat.owned_names_input(())[1]["tooltip"],
        }))
        self.assertEqual(compat.owned_names_input((), tooltip="x")[1]["tooltip"], "x")
        self.assertEqual(compat.OWNED_NAMES_INPUT, "owned_names")
        self.assertTrue(compat.is_disabled_choice(compat.NO_OWNED_NAMES))


class _Patcher:
    """ModelPatcher's wrapper API (comfy/model_patcher.py add/remove_wrappers_with_key, clone)."""

    def __init__(self):
        self.wrappers: dict[str, dict[str, list[Any]]] = {}
        self.model_options: dict[str, Any] = {}

    def clone(self):
        clone = _Patcher()
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()}
                          for kind, keyed in self.wrappers.items()}
        clone.model_options = dict(self.model_options)
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


class WrapperOrderTests(unittest.TestCase):
    def test_make_first_moves_one_key_and_keeps_the_rest_in_order(self):
        model = _Patcher()
        for key in ("dd", "speed", "colorcraft"):
            model.add_wrapper_with_key("sampler_sample", key, key)
        clone = model.clone()
        self.assertTrue(compat.make_first(clone, "sampler_sample", "speed"))
        self.assertEqual(list(clone.wrappers["sampler_sample"]), ["speed", "dd", "colorcraft"])
        self.assertEqual(list(model.wrappers["sampler_sample"]), ["dd", "speed", "colorcraft"])
        self.assertFalse(compat.make_first(clone, "sampler_sample", "missing"))
        self.assertFalse(compat.make_first(clone, "other_type", "speed"))
        self.assertFalse(compat.make_first(object(), "sampler_sample", "speed"))

    def test_cns_applied_after_another_first_wrapper_is_still_outermost(self):
        model = _Patcher()
        model.add_wrapper_with_key("sampler_sample", "dd", "dd")
        model.add_wrapper_with_key("sampler_sample", "speed", "speed")
        compat.make_first(model, "sampler_sample", "speed")
        patched = guidance_cns.apply_cns(model)
        self.assertEqual(
            list(patched.wrappers["sampler_sample"]),
            [guidance_cns.CNS_WRAPPER_KEY, "speed", "dd"],
        )

    def test_sampler_sample_wrapper_type_reads_comfy_or_falls_back(self):
        with mock.patch.dict(sys.modules, {"comfy.patcher_extension": None}):
            self.assertEqual(compat.sampler_sample_wrapper_type(), compat.SAMPLER_SAMPLE_FALLBACK)
        stub = types.ModuleType("comfy.patcher_extension")
        stub.WrappersMP = SimpleNamespace(SAMPLER_SAMPLE="sampler_sample_v2")
        comfy_stub = types.ModuleType("comfy")
        comfy_stub.patcher_extension = stub
        with mock.patch.dict(sys.modules, {"comfy": comfy_stub, "comfy.patcher_extension": stub}):
            self.assertEqual(compat.sampler_sample_wrapper_type(), "sampler_sample_v2")
            self.assertEqual(guidance_cns._sampler_sample_wrapper_type(), "sampler_sample_v2")
        self.assertEqual(compat.SAMPLER_SAMPLE_FALLBACK, "sampler_sample")


class PackNeverImportsSamExtraTests(unittest.TestCase):
    """sam-extra is GPL-3.0: the pack re-implements, it never imports the extension."""

    def test_no_pack_module_imports_sam3ext_or_extension_scripts(self):
        offenders = []
        for path in sorted(PACK_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for item in ast.walk(tree):
                modules: list[str] = []
                if isinstance(item, ast.Import):
                    modules = [alias.name for alias in item.names]
                elif isinstance(item, ast.ImportFrom) and item.module and not item.level:
                    modules = [item.module]
                elif isinstance(item, ast.Call) and getattr(item.func, "attr", getattr(item.func, "id", "")) in (
                    "import_module", "__import__",
                ) and item.args and isinstance(item.args[0], ast.Constant):
                    modules = [str(item.args[0].value)]
                for module in modules:
                    if module.split(".")[0] in ("sam3ext", "forge_sam3_extension") or module.startswith("scripts."):
                        offenders.append(f"{path.relative_to(PACK_DIR)}:{item.lineno} {module}")
        self.assertEqual(offenders, [])


# --- PB0b: Hires carry, flow-shift carrier copy, Dy sub-steps flag ---------------------------
CARRIER = "_ai_studio_test_carrier"


class _Sampling:
    """A model_sampling stand-in (ModelSamplingDiscreteFlow's shift/multiplier)."""

    family = "class attribute: never carried"

    def __init__(self, shift: float = 1.0, multiplier: float = 1.0):
        self.shift, self.multiplier = shift, multiplier


class _SamplingPatcher:
    """ModelPatcher's clone / object patch / get_model_object / wrapper API (comfy/model_patcher.py)."""

    def __init__(self, sampling: Any = None, label: str = "base"):
        self.label = label
        self.base_sampling = sampling if sampling is not None else _Sampling()
        self.object_patches: dict[str, Any] = {}
        self.model_options: dict[str, Any] = {}
        self.wrappers: dict[str, dict[str, list[Any]]] = {}

    def clone(self):
        clone = _SamplingPatcher(self.base_sampling, self.label)
        clone.object_patches = dict(self.object_patches)
        clone.model_options = dict(self.model_options)
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()}
                          for kind, keyed in self.wrappers.items()}
        return clone

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name != "model_sampling":
            raise AttributeError(name)
        return self.object_patches.get(name, self.base_sampling)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def __repr__(self):
        return f"<patcher {self.label}>"


def _carry(model: _SamplingPatcher, settings: Any) -> _SamplingPatcher:
    """What a feature's carrier does (Extra Schedulers): a clone whose model_sampling copy holds the values."""
    patched = model.clone()
    sampling = model.get_model_object("model_sampling")
    copy = _Sampling(sampling.shift, sampling.multiplier)
    setattr(copy, CARRIER, settings)
    patched.add_object_patch("model_sampling", copy)
    return patched


class _Providers:
    """``generation.invoke_provider`` stand-in for the ComfyUI nodes HiresFix calls."""

    def __init__(self, *, checkpoint=None, unet=None):
        self.calls: list[tuple[str, tuple]] = []
        self.checkpoint, self.unet = checkpoint, unet

    def __call__(self, name, *, method, feature, args=(), kwargs=None):
        self.calls.append((name, tuple(args)))
        if name == "ModelSamplingSD3":     # comfy_extras/nodes_model_advanced.py ModelSamplingSD3.patch
            model, shift, multiplier = args
            patched = model.clone()
            patched.add_object_patch("model_sampling", _Sampling(shift, multiplier))
            return (patched,)
        if name == "CheckpointLoaderSimple":
            return self.checkpoint
        if name == "UNETLoader":
            return (self.unet,)
        if name == "CLIPTextEncode":
            return (f"encoded:{args[1]}",)
        raise AssertionError(f"unexpected provider {name}")


def _paths(*, checkpoint: bool = False, unet: bool = False):
    def get_full_path(kind, name):
        if kind == "checkpoints" and checkpoint:
            return f"C:/models/{name}"
        if kind == "diffusion_models" and unet:
            return f"C:/unet/{name}"
        return None

    return SimpleNamespace(get_full_path=get_full_path)


class _HiresHarness:
    """Runs ForgeNeoHiresFix.run with stubbed ComfyUI; returns the MODEL the pass samples."""

    def __init__(self, providers: _Providers, paths=None):
        self.providers, self.paths = providers, paths or _paths()

    def run(self, model, **inputs):
        comfy_stub = types.ModuleType("comfy")
        utils_stub = types.ModuleType("comfy.utils")
        utils_stub.common_upscale = lambda tensor, w, h, method, crop: ("upscaled", w, h)
        comfy_stub.utils = utils_stub
        samples = {"samples": SimpleNamespace(shape=(1, 16, 8, 10))}
        with (
            mock.patch.dict(sys.modules, {"comfy": comfy_stub, "comfy.utils": utils_stub}),
            mock.patch.object(generation, "invoke_provider", self.providers),
            mock.patch.object(generation, "folder_paths_module", return_value=self.paths),
            mock.patch.object(generation, "_common_sample",
                              side_effect=lambda m, *a, **k: ({"model": m, "args": a},)),
        ):
            output, vae = generation.ForgeNeoHiresFix().run(
                model, "positive", "negative", samples, seed=5, enabled=True, steps=10,
                base_vae="base-vae", base_clip="base-clip", **inputs,
            )
        return output["model"], vae, output["args"]


class _IsolatedCarryRegistry(unittest.TestCase):
    """Runs with an empty carry registry and one test carrier attribute; restores both after."""

    def setUp(self):
        carries = mock.patch.dict(hires_carry._CARRIES, clear=True)
        attrs = mock.patch.dict(hires_carry._SAMPLING_ATTRS, clear=True)
        carries.start()
        attrs.start()
        self.addCleanup(carries.stop)
        self.addCleanup(attrs.stop)
        hires_carry.register_sampling_attr(CARRIER)


class HiresCarryRegistryTests(_IsolatedCarryRegistry):
    def test_carries_run_in_registration_order_on_the_previous_result(self):
        seen = []

        def first(source, target, vae):
            seen.append(("first", source, target, vae))
            return f"{target}+first"

        def second(source, target, vae):
            seen.append(("second", source, target, vae))
            return None        # nothing to carry: the MODEL stays as it is

        @hires_carry.register
        def third(source, target, vae):
            seen.append(("third", source, target, vae))
            return f"{target}+third"

        hires_carry.register(first, key="first")
        hires_carry.register(second, key="second")
        hires_carry.register(first, key="first")     # again: keeps its place
        self.assertEqual(hires_carry.registered()[1:], ("first", "second"))
        self.assertTrue(hires_carry.registered()[0].endswith("third"))
        result = hires_carry.apply("base", "override", "vae2")
        self.assertEqual(result, "override+third+first")
        self.assertEqual(seen, [
            ("third", "base", "override", "vae2"),
            ("first", "base", "override+third", "vae2"),
            ("second", "base", "override+third+first", "vae2"),
        ])
        hires_carry.unregister("first")
        self.assertNotIn("first", hires_carry.registered())

    def test_same_model_and_empty_registry_change_nothing(self):
        model = object()
        self.assertIs(hires_carry.apply(model, model, None), model)
        other = object()
        self.assertIs(hires_carry.apply(model, other, None), other)
        hires_carry.register(lambda s, t, v: self.fail("must not run"), key="never")
        self.assertIs(hires_carry.apply(model, model, None), model)

    def test_a_failing_carry_fails_the_pass_with_its_key(self):
        def broken(source, target, vae):
            raise ValueError("no anchors")

        hires_carry.register(broken, key="colorcraft")
        with self.assertRaisesRegex(RuntimeError, "'colorcraft'.*no anchors") as caught:
            hires_carry.apply("base", "override", None)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        with self.assertRaises(TypeError):
            hires_carry.register("not callable")
        with self.assertRaises(ValueError):
            hires_carry.register_sampling_attr("")


class SamplingAttrCopyTests(_IsolatedCarryRegistry):
    def test_registered_own_attributes_reach_the_new_model_sampling(self):
        settings = {"laplace_mu": 0.4}
        source = _Sampling(3.0)
        setattr(source, CARRIER, settings)
        source.unregistered = "stays behind"
        model = _SamplingPatcher(label="shifted")
        model.add_object_patch("model_sampling", _Sampling(5.0))
        self.assertIs(hires_carry.copy_sampling_attrs(source, model), model)
        target = model.get_model_object("model_sampling")
        self.assertIs(getattr(target, CARRIER), settings)
        self.assertFalse(hasattr(target, "unregistered"))
        self.assertEqual(hires_carry.sampling_attrs(), (CARRIER,))

    def test_nothing_to_keep_changes_nothing(self):
        model = _SamplingPatcher()
        target = model.get_model_object("model_sampling")
        for source in (None, _Sampling(), mock.Mock(), SimpleNamespace(x=1)):
            with self.subTest(source=source):
                self.assertIs(hires_carry.copy_sampling_attrs(source, model), model)
                self.assertFalse(hasattr(target, CARRIER))
        # a class attribute is not the object's own value
        hires_carry.register_sampling_attr("family")
        self.assertIs(hires_carry.copy_sampling_attrs(_Sampling(), model), model)
        self.assertNotIn("family", vars(target))
        # the same object: already there
        setattr(target, CARRIER, "kept")
        self.assertIs(hires_carry.copy_sampling_attrs(target, model), model)
        self.assertEqual(getattr(target, CARRIER), "kept")

    def test_a_carrier_with_nowhere_to_go_raises(self):
        source = _Sampling()
        setattr(source, CARRIER, "values")
        with self.assertRaisesRegex(RuntimeError, CARRIER):
            hires_carry.copy_sampling_attrs(source, "not a model")


class FlowShiftKeepsCarrierTests(_IsolatedCarryRegistry):
    def test_flow_shift_keeps_the_carrier_and_the_multiplier(self):
        base = _SamplingPatcher(_Sampling(1.0, multiplier=1.0))
        carried = _carry(base, {"custom": "0.9,0.5,0"})
        providers = _Providers()
        with mock.patch.object(generation, "invoke_provider", providers):
            shifted = generation._patch_flow_shift(carried, 3.0)
        (name, args), = providers.calls
        self.assertEqual((name, args[1:]), ("ModelSamplingSD3", (3.0, 1.0)))
        sampling = shifted.get_model_object("model_sampling")
        self.assertIsNot(sampling, carried.get_model_object("model_sampling"))
        self.assertEqual(sampling.shift, 3.0)
        self.assertEqual(getattr(sampling, CARRIER), {"custom": "0.9,0.5,0"})
        # the model it was shifted from is untouched
        self.assertFalse(hasattr(base.get_model_object("model_sampling"), CARRIER))

    def test_without_registered_attributes_the_flow_shift_is_as_before(self):
        hires_carry.unregister_sampling_attr(CARRIER)
        carried = _carry(_SamplingPatcher(), "values")
        with mock.patch.object(generation, "invoke_provider", _Providers()):
            shifted = generation._patch_flow_shift(carried, 3.0)
        self.assertFalse(hasattr(shifted.get_model_object("model_sampling"), CARRIER))

    def test_hires_pass_with_shift_keeps_the_carrier_without_an_override(self):
        called = []
        hires_carry.register(lambda *a: called.append(a), key="spy")
        model = _carry(_SamplingPatcher(), {"laplace_beta": 0.7})
        sampled, vae, _ = _HiresHarness(_Providers()).run(model, shift=3.0)
        sampling = sampled.get_model_object("model_sampling")
        self.assertEqual(getattr(sampling, CARRIER), {"laplace_beta": 0.7})
        self.assertEqual(sampling.shift, 3.0)
        self.assertEqual(vae, "base-vae")
        self.assertEqual(called, [])              # no override: nothing to re-apply
        self.assertNotIn(guidance_common.DY_SUBSTEPS_OFF_KEY,
                         sampled.model_options.get("transformer_options", {}))

    def test_hires_pass_without_shift_samples_the_incoming_model(self):
        model = _carry(_SamplingPatcher(), "values")
        providers = _Providers()
        sampled, _, _ = _HiresHarness(providers).run(model, shift=0.0)
        self.assertIs(sampled, model)
        self.assertEqual(providers.calls, [])


class HiresOverrideCarryTests(_IsolatedCarryRegistry):
    def _register_feature_carry(self, record):
        def carry(source, target, vae):
            record.append((source, target, vae))
            settings = getattr(source.get_model_object("model_sampling"), CARRIER, None)
            return None if settings is None else _carry(target, settings)

        hires_carry.register(carry, key="extra_schedulers")

    def test_checkpoint_override_gets_the_patches_after_its_flow_shift(self):
        record = []
        self._register_feature_carry(record)
        model = _carry(_SamplingPatcher(label="base"), {"laplace_mu": 0.2})
        fresh = _SamplingPatcher(_Sampling(1.0, multiplier=1.0), label="override")
        providers = _Providers(checkpoint=(fresh, "override-clip", "override-vae"))
        sampled, vae, args = _HiresHarness(providers, _paths(checkpoint=True)).run(
            model, shift=3.0, checkpoint_name="other.safetensors",
        )
        (source, target, carry_vae), = record
        self.assertIs(source, model)
        self.assertEqual(target.label, "override")
        self.assertEqual(target.get_model_object("model_sampling").shift, 3.0)   # after the flow shift
        self.assertEqual(carry_vae, "override-vae")
        self.assertEqual(vae, "override-vae")
        sampling = sampled.get_model_object("model_sampling")
        self.assertEqual(getattr(sampling, CARRIER), {"laplace_mu": 0.2})
        self.assertEqual(sampling.shift, 3.0)
        self.assertEqual(sampled.label, "override")
        self.assertEqual([name for name, _ in providers.calls],
                         ["CheckpointLoaderSimple", "CLIPTextEncode", "CLIPTextEncode", "ModelSamplingSD3"])
        self.assertEqual(args[5:7], ("encoded:", "encoded:"))   # positive, negative

    def test_unet_override_without_shift_gets_the_patches(self):
        record = []
        self._register_feature_carry(record)
        model = _carry(_SamplingPatcher(label="base"), "values")
        fresh = _SamplingPatcher(label="unet")
        providers = _Providers(unet=fresh)
        sampled, vae, _ = _HiresHarness(providers, _paths(unet=True)).run(
            model, shift=0.0, checkpoint_name="other_unet.safetensors",
        )
        (source, target, carry_vae), = record
        self.assertIs(source, model)
        self.assertIs(target, fresh)
        self.assertEqual(carry_vae, "base-vae")
        self.assertEqual(vae, "base-vae")
        self.assertEqual(getattr(sampled.get_model_object("model_sampling"), CARRIER), "values")
        self.assertFalse(hasattr(fresh.get_model_object("model_sampling"), CARRIER))

    def test_a_failing_carry_stops_the_override_pass(self):
        hires_carry.register(lambda s, t, v: 1 / 0, key="speed")
        providers = _Providers(unet=_SamplingPatcher(label="unet"))
        with self.assertRaisesRegex(RuntimeError, "'speed'"):
            _HiresHarness(providers, _paths(unet=True)).run(
                _SamplingPatcher(), shift=0.0, checkpoint_name="other_unet.safetensors",
            )

    def test_carry_runs_before_cns_so_cns_stays_outermost(self):
        def carry(source, target, vae):
            patched = target.clone()
            patched.add_wrapper_with_key("sampler_sample", "speed", "speed-wrapper")
            compat.make_first(patched, "sampler_sample", "speed")
            return patched

        hires_carry.register(carry, key="speed")
        providers = _Providers(unet=_SamplingPatcher(label="unet"))
        sampled, _, _ = _HiresHarness(providers, _paths(unet=True)).run(
            _SamplingPatcher(), shift=0.0, checkpoint_name="other_unet.safetensors",
            cns_enabled=True,
        )
        self.assertEqual(list(sampled.wrappers["sampler_sample"]),
                         [guidance_cns.CNS_WRAPPER_KEY, "speed"])


class DySubstepsFlagTests(unittest.TestCase):
    """ForgeNeoKSamplerCNS marks ComfyUI Spectrum and the legacy SPEED provider only (PLAN R3)."""

    def _sample(self, model, **inputs):
        received = []

        def fake_provider(name, *, method, feature, args=(), kwargs=None):
            received.append((name, args[0]))
            return (args[0],) if name == "DiTSpectrumPatch" else ({"samples": "speed"},)

        with (
            mock.patch.object(generation, "invoke_provider", side_effect=fake_provider),
            mock.patch.object(generation, "_common_sample", side_effect=lambda m, *a, **k: (m,)),
        ):
            result = generation.ForgeNeoKSamplerCNS().sample(
                model, "positive", "negative", {"samples": None}, 1, 28, 5.0, "euler", "simple",
                **inputs,
            )
        return result, received

    def test_spectrum_turns_dy_substeps_off_on_its_own_clone(self):
        model = _SamplingPatcher()
        model.model_options["transformer_options"] = {"keep": 1}
        (sampled,), received = self._sample(model, spectrum_enabled=True)
        (name, patched), = received
        self.assertEqual(name, "DiTSpectrumPatch")
        self.assertIs(sampled, patched)
        self.assertIsNot(patched, model)
        self.assertEqual(patched.model_options["transformer_options"],
                         {"keep": 1, guidance_common.DY_SUBSTEPS_OFF_KEY: "Spectrum"})
        self.assertEqual(model.model_options["transformer_options"], {"keep": 1})

    def test_legacy_speed_provider_turns_dy_substeps_off(self):
        model = _SamplingPatcher()
        result, received = self._sample(model, speed_enabled=True)
        self.assertEqual(result, ({"samples": "speed"},))
        (name, speed_model), = received
        self.assertEqual(name, "SpectrumSPDKSampler")
        self.assertIsNot(speed_model, model)
        self.assertTrue(speed_model.model_options["transformer_options"][guidance_common.DY_SUBSTEPS_OFF_KEY])
        self.assertEqual(model.model_options, {})

    def test_plain_sampling_has_no_flag(self):
        model = _SamplingPatcher()
        (sampled,), received = self._sample(model)
        self.assertIs(sampled, model)
        self.assertEqual(received, [])
        self.assertEqual(model.model_options, {})

    def test_hires_fix_never_sets_the_flag(self):
        source = inspect.getsource(generation.ForgeNeoHiresFix)
        self.assertNotIn("DY_SUBSTEPS_OFF_KEY", source)

    def test_set_transformer_option_copies_the_option_dicts(self):
        shared = {"transformer_options": {"a": 1}, "other": [1]}
        model = SimpleNamespace(model_options=shared)
        self.assertIs(guidance_common.set_transformer_option(model, "b", 2), model)
        self.assertEqual(model.model_options["transformer_options"], {"a": 1, "b": 2})
        self.assertEqual(shared, {"transformer_options": {"a": 1}, "other": [1]})
        bare = SimpleNamespace()
        guidance_common.set_transformer_option(bare, "b", 2)
        self.assertEqual(bare.model_options, {"transformer_options": {"b": 2}})


@requires_torch
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run the flow-shift carrier on the real ComfyUI runtime (CPU)")
class FlowShiftCarrierOnRealComfyTests(_IsolatedCarryRegistry):
    """The carrier copy with ComfyUI's own ModelPatcher, ModelSamplingSD3 and common_upscale."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.model_patcher
        import comfy.model_sampling
        import comfy.samplers
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3

        cls.model_patcher, cls.model_sampling = comfy.model_patcher, comfy.model_sampling
        cls.samplers, cls.sd3 = comfy.samplers, ModelSamplingSD3
        cls.flow_class = type(
            "AnimaFlowSampling",
            (comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST), {},
        )

    def _patcher(self):
        flow_class = self.flow_class

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model_config = SimpleNamespace(sampling_settings={"shift": 1.0, "multiplier": 1.0})
                self.model_sampling = flow_class(self.model_config)
                self.weight = torch.nn.Parameter(torch.zeros(1))

        cpu = torch.device("cpu")
        return self.model_patcher.ModelPatcher(Tiny(), cpu, cpu)

    def _carry(self, patcher, settings):
        carried = patcher.clone()
        current = patcher.get_model_object("model_sampling")
        copy = self.flow_class(patcher.model.model_config)
        copy.set_parameters(shift=current.shift, multiplier=current.multiplier)
        setattr(copy, CARRIER, settings)
        carried.add_object_patch("model_sampling", copy)
        return carried

    def _providers(self):
        def provider(name, *, method, feature, args=(), kwargs=None):
            if name != "ModelSamplingSD3":
                raise AssertionError(f"unexpected provider {name}")
            return self.sd3().patch(*args)

        return provider

    def test_real_flow_shift_keeps_the_carrier(self):
        base = self._patcher()
        settings = {"laplace_mu": 0.3}
        carried = self._carry(base, settings)
        with mock.patch.object(generation, "invoke_provider", side_effect=self._providers()):
            shifted = generation._patch_flow_shift(carried, 3.0)
        sampling = shifted.get_model_object("model_sampling")
        self.assertIsNot(sampling, carried.get_model_object("model_sampling"))
        self.assertIsInstance(sampling, self.model_sampling.CONST)
        self.assertEqual((sampling.shift, sampling.multiplier), (3.0, 1.0))
        self.assertIs(getattr(sampling, CARRIER), settings)
        self.assertNotIn(CARRIER, vars(base.get_model_object("model_sampling")))
        # the copied attribute changes nothing about the schedule itself
        hires_carry.unregister_sampling_attr(CARRIER)
        with mock.patch.object(generation, "invoke_provider", side_effect=self._providers()):
            plain = generation._patch_flow_shift(base, 3.0)
        self.assertFalse(hasattr(plain.get_model_object("model_sampling"), CARRIER))
        self.assertTrue(torch.equal(
            self.samplers.calculate_sigmas(sampling, "simple", 6),
            self.samplers.calculate_sigmas(plain.get_model_object("model_sampling"), "simple", 6),
        ))

    def test_real_hires_pass_with_shift_samples_the_carrier(self):
        carried = self._carry(self._patcher(), "values")
        captured = []
        with (
            mock.patch.object(generation, "invoke_provider", side_effect=self._providers()),
            mock.patch.object(generation, "_common_sample",
                              side_effect=lambda m, *a, **k: (captured.append(m) or {"samples": a[7]},)),
        ):
            output, _vae = generation.ForgeNeoHiresFix().run(
                carried, "positive", "negative", {"samples": torch.zeros(1, 4, 8, 8)},
                seed=3, enabled=True, steps=4, shift=3.0, upscale_method="latent:nearest-exact",
            )
        (sampled,) = captured
        sampling = sampled.get_model_object("model_sampling")
        self.assertEqual(sampling.shift, 3.0)
        self.assertEqual(getattr(sampling, CARRIER), "values")
        self.assertEqual(tuple(output["samples"]["samples"].shape), (1, 4, 12, 12))


if __name__ == "__main__":
    unittest.main()
