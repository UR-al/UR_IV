r"""ComfyUI pack 1.7.0 Colorcraft = sam-extra v0.32.1 Colorcraft — same tensors, same numbers (CPU).

The pack vendors muerrilla/ComfyUI-Colorcraft@d28ac6a's MIT library files and basis vectors unchanged
(``vendor/comfyui_colorcraft/``, the fork's Flux2 vectors in ``vendor/comfyui_colorcraft_fork/``) and writes
its own engine/host/node from the MIT upstream and fork. The sam-extra extension (GPL-3.0) is never imported
by the pack: here the **installed** extension is only the oracle.

* ``VendorProvenanceTests`` (no torch, no extension): every vendored file has the pinned SHA-256, the
  manifests list them, the vector files are part of the pack fingerprint (a vectors-only change asks for a
  restart), git does not ignore them, and the vendored package imports nothing at pack import.
* ``ExtensionProvenanceTests`` (``@requires_extension``): the pins equal the hashes the extension pins for
  its own verbatim copies of the same upstream files (``tests/test_colorcraft_origin.py`` ``ORIGIN_SHA256``,
  read by AST) and the extension's copy text; the vector files equal the extension's ``data/`` bytes; the
  licence pins equal the extension's ``LICENSE_SHA256`` and its upstream/fork LICENSE files.
* ``EngineStructureTests`` (``@requires_extension``, no torch): ``colorcraft_engine``/``colorcraft_basis`` are
  the MIT upstream's code — the modifier loop equals upstream ``nodes.py``'s ``post_cfg_function`` loop after
  exactly the three documented substitutions, ``build_entries``/``sigma_to_value`` equal upstream's,
  ``load_basis``/``resolve_dev`` upstream ``basis.py``'s, ``family_for_latent_format``/``to_model_space`` the
  fork's ``core.py``'s (syntax trees; the upstream texts are read from the extension's pinned verbatim copies,
  their SHA-256 checked first); the tables equal upstream/fork and the extension's; both modules import
  without torch.
* ``EngineParityTests`` (``@requires_torch @requires_extension``): ``apply_chain`` == the extension's
  ``engine.apply_chain`` bit for bit (each side with the basis loaded from its own vector file, the same
  anchors) on seeded random stacks (all kinds, passes, gates, masks, combos), every mask axis x mode, every
  combo operation, leaf/combo shaping, 4-D and 5-D latents, krea2/zimage (16 channels) and flux2 (128
  channels, 16x factor), no family, float32/bfloat16/float16; ``build_entries`` schedules,
  ``needed_colors``, ``needs_basis``, ``schedule_values`` and ``sigma_to_value`` equal.
"""
from __future__ import annotations

import ast
import copy
import gc
import hashlib
import importlib
import importlib.util
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.comfy_node_pack import install_bundled_node_pack, node_pack_fingerprint, pack_files
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

torch = None  # filled by bind_torch() in the @requires_torch classes

REPO = Path(__file__).resolve().parents[1]
PACK_DIR = REPO / "comfy_custom_nodes" / "ai_studio_forge_parity"
VENDOR = PACK_DIR / "vendor" / "comfyui_colorcraft"
VENDOR_FORK = PACK_DIR / "vendor" / "comfyui_colorcraft_fork"

UPSTREAM_COMMIT = "d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf"
FORK_COMMIT = "f00066c63c9d8f96abc119cada3b51b36688fcac"

# SHA-256 of the vendored bytes (= upstream's files at the pinned commits; LF for the .py files).
VENDORED_SHA256 = {
    "vendor/comfyui_colorcraft/lib_colorcraft/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "vendor/comfyui_colorcraft/lib_colorcraft/vectors.py":
        "9f8a1dce3866a2c57cd912d89606d38a6e972b6d8a39009b18e0fd42f098666d",
    "vendor/comfyui_colorcraft/lib_colorcraft/masking.py":
        "ce584f9b0a6db5ab8e42560afbc8c1a092cb268553045bc345b36f318b9f2df8",
    "vendor/comfyui_colorcraft/lib_colorcraft/color.py":
        "b7df301fcadc685ad0a6a78ee3a2a4c657e6e33fcf9639ef7a776b238307f37f",
    "vendor/comfyui_colorcraft/lib_colorcraft/schedule.py":
        "bc1b8b98079e861d50b7a1f4ad045fbd216523e9ee0ca3fb4b624f88c64c2534",
    "vendor/comfyui_colorcraft/vectors/colorcraft-krea2.safetensors":
        "0612ace7de9857927fc843f008a64633a097ab260b2c9ec3dc76b64c86b46cf2",
    "vendor/comfyui_colorcraft/vectors/colorcraft-zimage.safetensors":
        "824313efd6046492927087b3745cdc7e83ef8cd5dd5858fe42ec86c9b693c3f4",
    "vendor/comfyui_colorcraft_fork/vectors/colorcraft-flux2.safetensors":
        "303e8a77d8fcba89162cbcc7b90a2e1094af5ff923b76be81b72a27c411bfacc",
}
VECTOR_FILES = tuple(rel for rel in VENDORED_SHA256 if rel.endswith(".safetensors"))
# The upstream / fork LICENSE texts (byte copies) that ship as the pack's LICENSES/ files.
LICENSE_SHA256 = {
    "LICENSES/ComfyUI-Colorcraft-MIT.txt": "b1a3a92dd9b78c9e99969106bac1ba7d84fbf337b489352d74e322402f77160c",
    "LICENSES/ComfyUI-Colorcraft-aoleg-MIT.txt": "d05d785c0378e5dd55a01ac7395049e08b6f9bd2762b113d749bdff41681f47b",
}
# PB-INT1 wrote LICENSES/ (and THIRD_PARTY_NOTICES.md) from the Colorcraft fragment, so the two files must exist
# with the pinned bytes. (While this was True the test asserted they were absent — a stale True failed.)
LICENSES_PENDING = False

# Our vendored path -> the extension's pinned copy (ORIGIN_SHA256 key) or data/ file.
EXT_ORIGIN_KEYS = {
    "vendor/comfyui_colorcraft/lib_colorcraft/__init__.py": "_origin_colorcraft/lib_colorcraft/__init__.py",
    "vendor/comfyui_colorcraft/lib_colorcraft/vectors.py": "_origin_colorcraft/lib_colorcraft/vectors.py",
    "vendor/comfyui_colorcraft/lib_colorcraft/masking.py": "_origin_colorcraft/lib_colorcraft/masking.py",
    "vendor/comfyui_colorcraft/lib_colorcraft/color.py": "_origin_colorcraft/lib_colorcraft/color.py",
    "vendor/comfyui_colorcraft/lib_colorcraft/schedule.py": "_origin_colorcraft/lib_colorcraft/schedule.py",
}
EXT_DATA = {
    "vendor/comfyui_colorcraft/vectors/colorcraft-krea2.safetensors": "colorcraft-krea2.safetensors",
    "vendor/comfyui_colorcraft/vectors/colorcraft-zimage.safetensors": "colorcraft-zimage.safetensors",
    "vendor/comfyui_colorcraft_fork/vectors/colorcraft-flux2.safetensors": "colorcraft-flux2.safetensors",
}
EXT_LICENSES = {
    "LICENSES/ComfyUI-Colorcraft-MIT.txt": "_origin_colorcraft/LICENSE",
    "LICENSES/ComfyUI-Colorcraft-aoleg-MIT.txt": "_origin_colorcraft_fork/LICENSE",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _comfy_importable() -> bool:
    """Is ComfyUI importable in this interpreter (the portable python of the real-ComfyUI tier)? There the pack's
    ``compat`` loads ComfyUI — and torch — when the pack is imported, as it does inside ComfyUI, so the app-venv
    rule "the pack imports without torch" is checked by the venv run only."""
    try:
        return importlib.util.find_spec("comfy") is not None
    except (ImportError, ValueError):
        return False


def _ext_table(name: str) -> dict:
    """A literal table of the extension's tests/test_colorcraft_origin.py, read by AST (never imported)."""
    tree = ast.parse((EXT_ROOT / "tests" / "test_colorcraft_origin.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in the extension's test_colorcraft_origin.py")


def _ext_origin_body(relative: str) -> str:
    """The upstream text of an extension test copy: everything after its ``# ---- upstream`` marker line."""
    text = (EXT_ROOT / "tests" / relative).read_text(encoding="utf-8")
    marker = next(line for line in text.splitlines(keepends=True) if line.startswith("# ---- upstream "))
    return text.split(marker, 1)[1]


class VendorProvenanceTests(unittest.TestCase):
    def test_vendored_files_have_the_pinned_bytes(self):
        for relative, digest in VENDORED_SHA256.items():
            with self.subTest(file=relative):
                self.assertEqual(_sha256((PACK_DIR / relative).read_bytes()), digest)

    def test_vendored_python_is_lf_without_bom(self):
        for relative in VENDORED_SHA256:
            if not relative.endswith(".py"):
                continue
            with self.subTest(file=relative):
                data = (PACK_DIR / relative).read_bytes()
                self.assertNotIn(b"\r", data)
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"))

    def test_vendor_directories_hold_only_the_listed_files_and_our_own(self):
        ours = {
            "vendor/comfyui_colorcraft/__init__.py", "vendor/comfyui_colorcraft/UPSTREAM.md",
            "vendor/comfyui_colorcraft/.gitattributes",
            "vendor/comfyui_colorcraft_fork/UPSTREAM.md", "vendor/comfyui_colorcraft_fork/.gitattributes",
        }
        found = {
            path.relative_to(PACK_DIR).as_posix()
            for directory in (VENDOR, VENDOR_FORK)
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(found, set(VENDORED_SHA256) | ours)

    def test_manifests_name_the_commits_and_every_file_with_its_hash(self):
        manifests = {"vendor/comfyui_colorcraft": (VENDOR / "UPSTREAM.md", UPSTREAM_COMMIT),
                     "vendor/comfyui_colorcraft_fork": (VENDOR_FORK / "UPSTREAM.md", FORK_COMMIT)}
        for prefix, (path, commit) in manifests.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(manifest=prefix):
                self.assertIn(commit, text)
                self.assertIn("MIT", text)
            for relative, digest in VENDORED_SHA256.items():
                if relative.startswith(prefix + "/"):
                    with self.subTest(manifest=prefix, file=relative):
                        self.assertIn(f"`{relative[len(prefix) + 1:]}`", text)
                        self.assertIn(digest, text)

    def test_vendored_package_imports_nothing_at_pack_import(self):
        """vendor/comfyui_colorcraft/__init__.py is ours and import-free (lib_colorcraft imports torch)."""
        tree = ast.parse((VENDOR / "__init__.py").read_text(encoding="utf-8"))
        self.assertFalse([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
        self.assertEqual(len(tree.body), 1)  # the docstring only

    def test_vector_files_are_fingerprinted_runtime_files(self):
        everything = {path.relative_to(PACK_DIR).as_posix() for path in pack_files(PACK_DIR)}
        runtime = {path.relative_to(PACK_DIR).as_posix() for path in pack_files(PACK_DIR, runtime_only=True)}
        for relative in VECTOR_FILES:
            with self.subTest(file=relative):
                self.assertIn(relative, everything)
                self.assertIn(relative, runtime)

    def test_a_vectors_only_change_refreshes_the_install_and_asks_for_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ai_studio_forge_parity"
            (source / "vendor" / "comfyui_colorcraft" / "vectors").mkdir(parents=True)
            (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            vector = source / "vendor" / "comfyui_colorcraft" / "vectors" / "colorcraft-krea2.safetensors"
            shutil.copyfile(PACK_DIR / VECTOR_FILES[0], vector)
            custom_nodes = root / "custom_nodes"
            custom_nodes.mkdir()
            first = install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(first.changed)
            installed = first.target / "vendor" / "comfyui_colorcraft" / "vectors" / vector.name
            self.assertEqual(installed.read_bytes(), vector.read_bytes())
            before = node_pack_fingerprint(source, runtime_only=True)

            vector.write_bytes(vector.read_bytes()[:-4] + b"\x00\x00\x80\x3f")
            self.assertNotEqual(node_pack_fingerprint(source, runtime_only=True), before)
            second = install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(second.changed)
            self.assertTrue(second.restart_required)
            self.assertEqual(installed.read_bytes(), vector.read_bytes())

    def test_git_does_not_ignore_the_vector_files(self):
        git = shutil.which("git")
        if git is None or not (REPO / ".git").exists():
            self.skipTest("git or a git checkout is not available")
        for relative in VECTOR_FILES:
            path = (PACK_DIR / relative).relative_to(REPO).as_posix()
            with self.subTest(file=relative):
                result = subprocess.run([git, "check-ignore", "--no-index", "-q", path], cwd=REPO,
                                        capture_output=True, encoding="utf-8")
                self.assertEqual(result.returncode, 1, f"{path} is ignored by .gitignore: {result.stderr}")
        # the negation is scoped: other .safetensors files in the pack stay ignored
        other = "comfy_custom_nodes/ai_studio_forge_parity/models/some-model.safetensors"
        result = subprocess.run([git, "check-ignore", "--no-index", "-q", other], cwd=REPO,
                                capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0)

    def test_git_treats_the_vector_files_as_binary(self):
        git = shutil.which("git")
        if git is None or not (REPO / ".git").exists():
            self.skipTest("git or a git checkout is not available")
        for relative in VECTOR_FILES:
            path = (PACK_DIR / relative).relative_to(REPO).as_posix()
            with self.subTest(file=relative):
                result = subprocess.run([git, "check-attr", "text", "diff", "--", path], cwd=REPO,
                                        capture_output=True, encoding="utf-8", check=True)
                self.assertIn(f"{path}: text: unset", result.stdout)
                self.assertIn(f"{path}: diff: unset", result.stdout)

    def test_licence_files(self):
        for relative, digest in LICENSE_SHA256.items():
            path = PACK_DIR / relative
            with self.subTest(file=relative, pending=LICENSES_PENDING):
                if LICENSES_PENDING:
                    self.assertFalse(path.exists(), "LICENSES file present: set LICENSES_PENDING = False")
                else:
                    text = path.read_text(encoding="utf-8")
                    self.assertEqual(_sha256(text.encode("utf-8")), digest)
                    self.assertIn("MIT License", text)


@requires_extension
class ExtensionProvenanceTests(unittest.TestCase):
    """The vendored bytes are the upstream files the extension pins for its own origin tests."""

    @classmethod
    def setUpClass(cls):
        cls.origin_pins = _ext_table("ORIGIN_SHA256")
        cls.license_pins = _ext_table("LICENSE_SHA256")

    def test_library_pins_equal_the_extensions_upstream_pins(self):
        for relative, key in EXT_ORIGIN_KEYS.items():
            with self.subTest(file=relative):
                self.assertEqual(VENDORED_SHA256[relative], self.origin_pins[key])

    def test_library_files_equal_the_extensions_verbatim_upstream_text(self):
        for relative, key in EXT_ORIGIN_KEYS.items():
            with self.subTest(file=relative):
                body = _ext_origin_body(key)
                self.assertEqual(_sha256(body.encode("utf-8")), self.origin_pins[key])
                self.assertEqual((PACK_DIR / relative).read_bytes(), body.encode("utf-8"))
                head = (EXT_ROOT / "tests" / key).read_text(encoding="utf-8").split("# ---- upstream", 1)[0]
                self.assertIn(UPSTREAM_COMMIT, head)

    def test_vector_files_equal_the_extensions_data_files(self):
        data = EXT_ROOT / "sam3ext" / "colorcraft" / "data"
        self.assertEqual({path.name for path in data.glob("*.safetensors")}, set(EXT_DATA.values()))
        for relative, name in EXT_DATA.items():
            with self.subTest(file=relative):
                self.assertEqual((PACK_DIR / relative).read_bytes(), (data / name).read_bytes())

    def test_licence_pins_equal_the_extensions_upstream_and_fork_licences(self):
        for relative, key in EXT_LICENSES.items():
            with self.subTest(file=relative):
                self.assertEqual(LICENSE_SHA256[relative], self.license_pins[key])
                text = (EXT_ROOT / "tests" / key).read_text(encoding="utf-8")
                self.assertEqual(_sha256(text.encode("utf-8")), LICENSE_SHA256[relative])


# --- PB-CC2: engine and basis ------------------------------------------------------------------------------

ORIGIN_NODES_KEY = "_origin_colorcraft/nodes.py"
ORIGIN_BASIS_KEY = "_origin_colorcraft/lib_colorcraft/basis.py"
FORK_CORE_KEY = "_origin_colorcraft_fork/lib_colorcraft/core.py"
FAMILY_CHANNELS = {"krea2": 16, "zimage": 16, "flux2": 128, None: 4}   # None: an SD 1.5/SDXL-like 4-channel latent


def _pinned_origin(relative: str) -> str:
    """Upstream text of an extension test copy, refused unless it hashes to the extension's own pin."""
    body = _ext_origin_body(relative)
    assert _sha256(body.encode("utf-8")) == _ext_table("ORIGIN_SHA256")[relative], relative
    return body


def _function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _dump(node) -> str:
    return ast.dump(node, include_attributes=False)


def _module_literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _pack_source(module: str) -> str:
    return (PACK_DIR / f"{module}.py").read_text(encoding="utf-8")


class _HostSubstitutions(ast.NodeTransformer):
    """Upstream's post-CFG loop with the three host substitutions of ``colorcraft_engine.apply_chain``:
    ``s = sigma_to_value(...)`` -> ``s = values[i]``; ``get_color_latent`` without ``latent_format``;
    ``VAE_DOWNSCALE_FACTOR`` -> ``vae_downscale_factor``."""

    def __init__(self):
        self.counts = {"values": 0, "anchor": 0, "factor": 0}

    def visit_Assign(self, node):
        self.generic_visit(node)
        call = node.value
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "sigma_to_value":
            node.value = ast.Subscript(ast.Name("values", ast.Load()), ast.Name("i", ast.Load()), ast.Load())
            self.counts["values"] += 1
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if getattr(node.func, "id", None) == "get_color_latent":
            kept = [arg for arg in node.args if getattr(arg, "id", None) != "latent_format"]
            self.counts["anchor"] += len(node.args) - len(kept)
            node.args = kept
        return node

    def visit_Name(self, node):
        if node.id == "VAE_DOWNSCALE_FACTOR":
            self.counts["factor"] += 1
            return ast.copy_location(ast.Name("vae_downscale_factor", node.ctx), node)
        return node


def _modifier_loop(function: ast.FunctionDef) -> ast.For:
    for node in ast.walk(function):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2
                and isinstance(node.target.elts[1], ast.Tuple)
                and [getattr(e, "id", None) for e in node.target.elts[1].elts]
                == ["schedule", "kind", "p", "mask_params"]):
            return node
    raise AssertionError("modifier loop not found")


def _entry_loop(function: ast.FunctionDef) -> ast.For:
    for node in ast.walk(function):
        if isinstance(node, ast.For) and getattr(node.target, "id", None) == "entry":
            return node
    raise AssertionError("per-entry schedule loop not found")


def _without_docstring(function: ast.FunctionDef) -> ast.FunctionDef:
    function = copy.deepcopy(function)
    if (function.body and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str)):
        function.body = function.body[1:]
    return function


@requires_extension
class EngineStructureTests(unittest.TestCase):
    """The pack's engine/basis are the MIT upstream's code (syntax trees), no torch at import."""

    @classmethod
    def setUpClass(cls):
        cls.nodes = _pinned_origin(ORIGIN_NODES_KEY)
        cls.basis = _pinned_origin(ORIGIN_BASIS_KEY)
        cls.fork = _pinned_origin(FORK_CORE_KEY)
        cls.engine_src = _pack_source("colorcraft_engine")
        cls.basis_src = _pack_source("colorcraft_basis")
        cls.ext_engine_src = (EXT_ROOT / "sam3ext" / "colorcraft" / "engine.py").read_text(encoding="utf-8")
        cls.ext_basis_src = (EXT_ROOT / "sam3ext" / "colorcraft" / "basis.py").read_text(encoding="utf-8")

    def test_modifier_loop_is_upstreams_after_exactly_three_substitutions(self):
        upstream = _function(self.nodes, "post_cfg_function")
        substitutions = _HostSubstitutions()
        theirs = substitutions.visit(copy.deepcopy(_modifier_loop(upstream)))
        ours = _modifier_loop(_function(self.engine_src, "apply_chain"))
        self.assertEqual(_dump(ours), _dump(theirs))
        # each substitution actually happened: one sigma lookup, five anchor look-ups (contrast twice, colour
        # shift three times), six mask gates
        self.assertEqual(substitutions.counts, {"values": 1, "anchor": 5, "factor": 6})

    def test_squeeze_and_unsqueeze_around_the_loop_are_upstreams(self):
        upstream = _function(self.nodes, "post_cfg_function")
        ours = _function(self.engine_src, "apply_chain")

        def statements(function):
            found = []
            for node in ast.walk(function):
                if isinstance(node, (ast.If, ast.Assign, ast.Return)) and "is_5d" in ast.unparse(node):
                    found.append(ast.unparse(node))
            return found

        self.assertEqual(statements(ours), statements(upstream))
        self.assertEqual(len(statements(ours)), 3)
        self.assertIn("return out", [ast.unparse(node) for node in ours.body])

    def test_build_entries_loop_is_upstreams(self):
        upstream = _entry_loop(_function(self.nodes, "wrapped_sampler_function"))
        theirs = ast.unparse(upstream).replace(
            "('start', 'end', 'bias', 'exponent', 'start_off', 'end_off', 'smooth')", "_SCHEDULE_KEYS")
        self.assertEqual(ast.unparse(_entry_loop(_function(self.engine_src, "build_entries"))), theirs)
        self.assertEqual(_module_literal(self.engine_src, "_SCHEDULE_KEYS"),
                         ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth"))

    def test_sigma_to_value_is_upstreams(self):
        self.assertEqual(_dump(_without_docstring(_function(self.engine_src, "sigma_to_value"))),
                         _dump(_without_docstring(_function(self.nodes, "sigma_to_value"))))

    def test_advanced_kinds_and_kinds_are_upstreams(self):
        upstream = _function(self.nodes, "wrapped_sampler_function")
        literal = next(node for node in ast.walk(upstream) if isinstance(node, ast.Assign)
                       and getattr(node.targets[0], "id", None) == "ADVANCED_KINDS")
        self.assertEqual(_module_literal(self.engine_src, "ADVANCED_KINDS"), ast.literal_eval(literal.value))
        made = set()
        for node in ast.parse(self.nodes).body:
            if isinstance(node, ast.ClassDef):
                for item in ast.walk(node):
                    if isinstance(item, ast.Dict):
                        keys = [k.value for k in item.keys if isinstance(k, ast.Constant)]
                        if "kind" in keys and isinstance(item.values[keys.index("kind")], ast.Constant):
                            made.add(item.values[keys.index("kind")].value)
        self.assertEqual(set(_module_literal(self.engine_src, "KINDS")), made)
        # the panel order the extension (and so the app's chain entries) use
        self.assertEqual(_module_literal(self.engine_src, "KINDS"), _module_literal(self.ext_engine_src, "KINDS"))
        self.assertEqual(_module_literal(self.engine_src, "NEUTRAL_COLOR"), (0.0, 0.0, 0.0, 0.0))

    def test_basis_functions_are_upstreams_and_the_forks(self):
        for name in ("load_basis", "resolve_dev"):
            with self.subTest(upstream=name):
                self.assertEqual(_dump(_function(self.basis_src, name)), _dump(_function(self.basis, name)))
        for name in ("family_for_latent_format", "to_model_space"):
            with self.subTest(fork=name):
                self.assertEqual(_dump(_function(self.basis_src, name)), _dump(_function(self.fork, name)))

    def test_basis_tables_are_upstreams_the_forks_and_the_extensions(self):
        ours = {name: _module_literal(self.basis_src, name) for name in (
            "BASIS_FAMILIES", "VAE_DOWNSCALE_FACTOR", "VAE_DOWNSCALE_FACTORS", "LATENT_FORMAT_TO_FAMILY",
            "MODEL_DEV_DEFAULTS")}
        upstream_dev = _module_literal(self.basis, "MODEL_DEV_DEFAULTS")
        fork_dev = _module_literal(self.fork, "MODEL_DEV_DEFAULTS")
        for family in ("krea2", "zimage"):
            self.assertEqual(ours["MODEL_DEV_DEFAULTS"][family], upstream_dev[family], family)
        self.assertEqual(ours["MODEL_DEV_DEFAULTS"]["flux2"], dict(fork_dev["flux2"], detail_scale=4.0))
        self.assertEqual(ours["VAE_DOWNSCALE_FACTOR"], _module_literal(self.basis, "VAE_DOWNSCALE_FACTOR"))
        for name in ("BASIS_FAMILIES", "VAE_DOWNSCALE_FACTORS", "LATENT_FORMAT_TO_FAMILY"):
            self.assertEqual(ours[name], _module_literal(self.fork, name), name)
        for name, value in ours.items():
            self.assertEqual(value, _module_literal(self.ext_basis_src, name), f"extension {name}")
        self.assertEqual(_module_literal(self.engine_src, "ADVANCED_KINDS"),
                         _module_literal(self.ext_engine_src, "ADVANCED_KINDS"))

    def test_the_vendored_vector_files_hold_every_family(self):
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_basis

        for family in colorcraft_basis.BASIS_FAMILIES:
            with self.subTest(family=family):
                path = Path(colorcraft_basis.vectors_dir(family)) / f"colorcraft-{family}.safetensors"
                self.assertTrue(path.is_file(), path)
                self.assertIn(path.relative_to(PACK_DIR).as_posix(), VECTOR_FILES)

    @unittest.skipIf(_comfy_importable(), "ComfyUI is importable here: the pack loads it (and torch) at import")
    def test_engine_and_basis_import_without_torch(self):
        for module in ("colorcraft_basis", "colorcraft_engine"):
            imported = set()
            for node in ast.parse(_pack_source(module)).body:  # module level only; functions import lazily
                if isinstance(node, ast.Import):
                    imported |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported.add("." * node.level + (node.module or ""))
            with self.subTest(module=module):
                self.assertLessEqual(imported, {"__future__", "os", ".colorcraft_basis"})
        code = ("import sys\n"
                "from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_basis, colorcraft_engine\n"
                "colorcraft_engine.needed_colors([{'kind': 'basic', 'params': {'contrast': 1.0}}])\n"
                "colorcraft_basis.vectors_dir('flux2')\n"
                "print(sorted(m for m in ('torch', 'safetensors') if m in sys.modules))\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, encoding="utf-8",
                                check=True)
        self.assertEqual(result.stdout.strip(), "[]", result.stderr)


# --- numbers (CPU, bit for bit) -----------------------------------------------------------------------------

def _ext_module(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def _random_value(field, rng: random.Random, spec):
    if field.kind == spec.FLOAT:
        roll = rng.random()
        if roll < 0.3:
            return field.default
        if roll < 0.38:
            return -0.0 if field.minimum < 0 else 0.0
        if roll < 0.43:  # out of range: the extension does not clamp typed values
            return round(field.maximum * 1.25 + 0.01, 2)
        value = rng.uniform(field.minimum, field.maximum)
        return round(value, 2) if rng.random() < 0.7 else round(value, 9)
    if field.kind == spec.BOOL:
        return rng.random() < 0.5
    return rng.choice(list(field.choices))


def _random_config(spec, rng: random.Random, *, masking=None):
    """A seeded random Colorcraft panel config (the extension's ``Config``): 1-4 active modifiers of any
    Type and Pass, random leaves and combos (combos reach leaves and earlier combos, some incomplete)."""
    config = spec.default_config()
    config.enabled = True
    config.masking = rng.random() < 0.6 if masking is None else masking
    for modifier in config.modifiers:
        modifier["active"] = False
    for index in rng.sample(range(spec.MODIFIER_COUNT), rng.randint(1, 4)):
        modifier = config.modifiers[index]
        for field in spec.MODIFIER_FIELDS:
            modifier[field.name] = _random_value(field, rng, spec)
        modifier["active"] = True
        if rng.random() < 0.5:  # a strong, wide schedule so most entries do something
            modifier["strength"], modifier["start"], modifier["end"] = rng.choice((1.0, -0.8, 0.6)), 0.0, 1.0
    for leaf in config.leaves:
        for field in spec.LEAF_FIELDS:
            leaf[field.name] = _random_value(field, rng, spec)
    for index, combo in enumerate(config.combos):
        for field in spec.COMBO_FIELDS:
            combo[field.name] = _random_value(field, rng, spec)
        refs = spec.combo_ref_choices(index)
        combo["mask_a"], combo["mask_b"] = rng.choice(refs[1:]), rng.choice(refs)
    return config


def _anchor_source(channels: int):
    """``get_color_latent(colour, device, dtype)``: one fixed pseudo-random model-space anchor per colour."""
    cache = {}

    def get_color_latent(colour, device, dtype):
        key = tuple(round(c, 4) for c in colour)
        if key not in cache:
            seed = zlib.crc32(repr(key).encode("utf-8"))
            cache[key] = torch.randn(channels, generator=torch.Generator().manual_seed(seed)) * 0.5
        return cache[key].to(device=device, dtype=dtype)

    return get_color_latent


def _latent(family: str, seed: int, *, five_d: bool, batch: int = 2, height: int = 13, width: int = 17,
            dtype=None):
    channels = FAMILY_CHANNELS.get(family, 16)
    shape = (batch, channels, 1, height, width) if five_d else (batch, channels, height, width)
    x0 = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    return x0 if dtype is None else x0.to(dtype)


def _flow_sigmas(steps: int, shift: float = 3.0):
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    sigmas = shift * t / (1 + (shift - 1) * t)
    return sigmas.to(torch.float32)


def _karras_sigmas(steps: int, sigma_min=0.03, sigma_max=14.6, rho=7.0):
    ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
    inv_min, inv_max = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sigmas = (inv_max + ramp * (inv_min - inv_max)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)]).to(torch.float32)


def _outcome(function, *args):
    """``(result, None)`` or ``(None, (error type, message))`` — for comparing two implementations' failures."""
    try:
        return function(*args), None
    except Exception as error:  # noqa: BLE001 - the caller compares both sides' outcomes
        return None, (type(error), str(error))


@requires_torch
@requires_extension
class EngineParityTests(unittest.TestCase):
    """``colorcraft_engine`` == the installed extension's ``engine`` on the same tensors (CPU, atol=0)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_basis, colorcraft_engine

        cls.pack, cls.pack_basis = colorcraft_engine, colorcraft_basis
        cls.ext = _ext_module("sam3ext.colorcraft.engine")
        cls.ext_basis = _ext_module("sam3ext.colorcraft.basis")
        cls.ext_schedule = _ext_module("sam3ext.colorcraft.schedule")
        cls.spec = _ext_module("sam3ext.colorcraft.spec")
        cls.bases = {}
        for family in cls.pack_basis.BASIS_FAMILIES:
            cls.bases[family] = (cls.pack_basis.basis_for(family),
                                 cls.ext_basis.load_basis(family, cls.ext_basis.VECTORS_DIR))

    def _same(self, ours, theirs, label):
        self.assertEqual(ours.dtype, theirs.dtype, label)
        self.assertEqual(ours.shape, theirs.shape, label)
        torch.testing.assert_close(ours, theirs, rtol=0, atol=0, equal_nan=True, msg=lambda m: f"{label}: {m}")

    def _cur_basis(self, family, side, x0):
        if family is None:
            return None
        basis = self.bases[family][side]
        return {k: v.to(device=x0.device, dtype=x0.dtype) for k, v in basis.items()}

    def _same_outcome(self, ours, theirs, label):
        """Both sides raised the same error (type and message), or neither did. True when both returned."""
        self.assertEqual(ours[1], theirs[1], f"{label}: different outcome")
        return ours[1] is None

    def _built(self, entries, steps, label):
        """``build_entries`` on both sides (an out-of-range typed schedule raises on both); None if it raised."""
        ours, theirs = _outcome(self.pack.build_entries, entries, steps), _outcome(self.ext.build_entries, entries, steps)
        if not self._same_outcome(ours, theirs, label):
            return None, None
        built, built_ext = ours[0], theirs[0]
        self.assertEqual(len(built), len(built_ext))
        for (s1, k1, p1, m1), (s2, k2, p2, m2) in zip(built, built_ext):
            self.assertEqual(s1.dtype, s2.dtype)
            self.assertEqual(s1.shape, s2.shape)
            self.assertTrue((s1 == s2).all(), label)
            self.assertEqual((k1, p1, m1), (k2, p2, m2))
        return built, built_ext

    def _both(self, entries, family, x0, values, steps=12, label=""):
        """Run pack and extension on the same chain; returns (ours, theirs) or (None, None) when both raised."""
        built, built_ext = self._built(entries, steps, label)
        if built is None:
            return None, None
        anchors = _anchor_source(x0.shape[1])
        ours = _outcome(self.pack.apply_chain, built, x0.clone(), values, family, self._cur_basis(family, 0, x0),
                        anchors, self.pack_basis.vae_downscale_factor(family))
        theirs = _outcome(self.ext.apply_chain, built_ext, x0.clone(), values, family,
                          self._cur_basis(family, 1, x0), anchors, self.ext_basis.vae_downscale_factor(family))
        if not self._same_outcome(ours, theirs, label):
            return None, None
        self._same(ours[0], theirs[0], label)
        return ours[0], theirs[0]

    def test_bases_load_the_same_vectors(self):
        for family, (ours, theirs) in self.bases.items():
            with self.subTest(family=family):
                self.assertIsNotNone(ours)
                self.assertEqual(sorted(ours), sorted(theirs))
                self.assertEqual(len(ours), 11)
                for key in ours:
                    self.assertEqual(ours[key].dtype, theirs[key].dtype)
                    self.assertTrue(torch.equal(ours[key], theirs[key]), key)
                    self.assertEqual(ours[key].shape, (FAMILY_CHANNELS[family],))
        self.assertIsNone(self.pack_basis.basis_for(None))
        self.assertIsNone(self.pack_basis.load_basis("krea2", str(PACK_DIR)))

    def test_random_stacks_on_every_family(self):
        rng = random.Random(20261004)
        moved = total = 0
        layouts = (("krea2", True), ("krea2", False), ("zimage", False), ("flux2", False))
        for case in range(96):
            config = _random_config(self.spec, rng)
            family, five_d = layouts[case % len(layouts)]
            hires = rng.random() < 0.4
            entries = self.spec.build_chain(config, hires).entries
            x0 = _latent(family, 100 + case, five_d=five_d)
            values = [rng.choice((0.0, 1.0, -0.6, 0.35, 0.123456789, -1.0, 0.8)) for _ in entries]
            with self.subTest(case=case, family=family, five_d=five_d):
                ours, _theirs = self._both(entries, family, x0, values, steps=rng.randint(2, 30),
                                           label=f"case {case} {family}")
                if ours is not None and entries:
                    total += 1
                    moved += not torch.equal(ours, x0)
        self.assertGreater(total, 60)
        self.assertGreater(moved, total * 2 // 3)  # the latents really moved (not two no-ops agreeing)

    def _masked_entries(self, kind, leaf_values, *, combo=None, extra=None):
        config = self.spec.default_config()
        config.enabled, config.masking = True, True
        modifier = config.modifiers[0]
        modifier.update({"active": True, "kind": self.spec.KIND_LABELS[kind], "start": 0.0, "end": 1.0,
                         "exposure": 0.7, "temperature": 1.2, "saturation": 0.5, "clarity": 1.5, "contrast": 0.9,
                         "temp_plus_tint": 0.8, "color_shift_amount": 0.6, "color_shift_red": 0.4,
                         "mask": "M1" if combo is None else "C1"})
        modifier.update(extra or {})
        config.leaves[0].update(leaf_values)
        if combo is not None:
            config.leaves[1].update({"mask_axis": "temperature", "mask_mode": "range", "mask_center": 0.2,
                                     "mask_width": 0.5, "mask_hardness": 2.0})
            config.combos[0].update(combo)
        entries = self.spec.build_chain(config, False).entries
        self.assertEqual(len(entries), 1)
        self.assertIsNotNone(entries[0]["mask"])
        return entries

    def test_every_mask_axis_and_mode_on_every_family(self):
        rng = random.Random(7)
        axes = self.spec.LEAF_BY_NAME["mask_axis"].choices
        modes = self.spec.LEAF_BY_NAME["mask_mode"].choices
        self.assertEqual(len(axes), 13)
        for family, five_d in (("krea2", True), ("zimage", False), ("flux2", False)):
            for axis in axes:
                for mode in modes:
                    shaping = rng.random() < 0.5
                    leaf = {"mask_axis": axis, "mask_mode": mode, "mask_center": round(rng.uniform(-1, 1), 2),
                            "mask_hardness": round(rng.uniform(0, 10), 1), "mask_width": round(rng.uniform(0, 2), 2),
                            "mask_strength": round(rng.uniform(0.2, 1), 2),
                            "blur": round(rng.uniform(0.1, 40), 1) if shaping else 0.0,
                            "spread": round(rng.uniform(-3, 3), 2) if shaping else 0.0,
                            "contrast": round(rng.uniform(-10, 10), 2) if shaping else 0.0,
                            "normalize": shaping and rng.random() < 0.5}
                    kind = rng.choice(("advanced", "luma", "chroma", "chroma_plus", "punch", "shift"))
                    entries = self._masked_entries(kind, leaf)
                    x0 = _latent(family, zlib.crc32(f"{family}{axis}{mode}".encode()), five_d=five_d)
                    with self.subTest(family=family, axis=axis, mode=mode, kind=kind):
                        self._both(entries, family, x0, [0.85], label=f"{family} {axis} {mode} {kind}")

    def test_every_combo_operation_with_shaping_and_nesting(self):
        leaf = {"mask_axis": "exposure", "mask_mode": "highs", "mask_hardness": 3.0, "blur": 6.5}
        for family, five_d in (("krea2", True), ("flux2", False), ("zimage", False)):
            for operation in self.spec.COMBO_OP_OPTIONS:
                for shaping in (False, True):
                    combo = {"mask_a": "M1", "mask_b": "M2", "operation": operation,
                             "blur": 12.0 if shaping else 0.0, "spread": -0.7 if shaping else 0.0,
                             "contrast": 3.5 if shaping else 0.0, "normalize": shaping}
                    entries = self._masked_entries("advanced", leaf, combo=combo)
                    x0 = _latent(family, len(operation) * 31 + shaping, five_d=five_d)
                    with self.subTest(family=family, operation=operation, shaping=shaping):
                        self._both(entries, family, x0, [-0.9], label=f"{family} {operation} {shaping}")

    def test_no_family_runs_only_the_vector_free_edits(self):
        rng = random.Random(11)
        for case in range(24):
            config = _random_config(self.spec, rng)
            entries = self.spec.build_chain(config, False).entries
            x0 = _latent("zimage", 900 + case, five_d=case % 2 == 0)
            values = [rng.choice((0.0, 1.0, -0.5, 0.25)) for _ in entries]
            with self.subTest(case=case):
                self._both(entries, None, x0, values, label=f"no family {case}")

    def test_half_precision_inputs(self):
        rng = random.Random(5)
        for case in range(16):
            config = _random_config(self.spec, rng)
            entries = self.spec.build_chain(config, False).entries
            family = ("krea2", "zimage", "flux2")[case % 3]
            dtype = (torch.bfloat16, torch.float16)[case % 2]
            x0 = _latent(family, 500 + case, five_d=family == "krea2", dtype=dtype)
            values = [rng.choice((1.0, -0.5, 0.3)) for _ in entries]
            with self.subTest(case=case, family=family, dtype=str(dtype)):
                self._both(entries, family, x0, values, label=f"half {case}")

    def test_all_zero_values_return_the_input(self):
        rng = random.Random(3)
        config = _random_config(self.spec, rng, masking=True)
        entries = self.spec.build_chain(config, False).entries + [
            {"kind": "luma", "params": {"exposure": 1.0, "tone_compression": 0.0},
             "schedule": {"strength": 1.0, "start": 0.0, "end": 1.0, "bias": 0.5, "exponent": 0.0,
                          "start_off": 0.0, "end_off": 0.0, "smooth": True}, "mask": None}]
        for five_d in (False, True):
            x0 = _latent("krea2", 1, five_d=five_d)
            built = self.pack.build_entries(entries, 10)
            out = self.pack.apply_chain(built, x0, [0.0] * len(built), "krea2", self._cur_basis("krea2", 0, x0),
                                        _anchor_source(16), 8)
            if five_d:
                self.assertTrue(torch.equal(out, x0))
                self.assertEqual(out.data_ptr(), x0.data_ptr())
            else:
                self.assertIs(out, x0)

    def test_chain_helpers_equal_the_extensions(self):
        rng = random.Random(99)
        errors = 0
        for case in range(200):
            config = _random_config(self.spec, rng)
            entries = self.spec.build_chain(config, rng.random() < 0.5).entries
            with self.subTest(case=case):
                self.assertEqual(self.pack.needed_colors(entries), self.ext.needed_colors(entries))
                self.assertEqual(self.pack.needs_basis(entries), self.ext.needs_basis(entries))
                steps = rng.randint(1, 40)
                built, built_ext = self._built(entries, steps, f"case {case}")
                if built is None:
                    errors += 1
                    continue
                self.assertEqual(self.pack.needs_basis(built), self.ext.needs_basis(built_ext))
                sigmas = _flow_sigmas(steps)
                for sigma in [float(s) for s in sigmas] + [0.5, 1.7, -0.1]:
                    self.assertEqual(self.pack.schedule_values(built, sigma, sigmas),
                                     self.ext.schedule_values(built_ext, sigma, sigmas))
        self.assertLess(errors, 20)  # out-of-range schedules raise on both sides; most stacks build

    def test_sigma_to_value_on_forge_and_comfy_lists(self):
        make_schedule = self.ext_schedule.make_schedule
        for steps in (1, 2, 3, 10, 28):
            schedule = make_schedule(steps, 0.1, 0.9, 0.4, 0.8, 1.2, 0.1, -0.2, True)
            for sigmas in (_flow_sigmas(steps), _flow_sigmas(steps, 1.0), _karras_sigmas(steps)):
                real = [float(s) for s in sigmas]
                probes = real + [(a + b) / 2 for a, b in zip(real, real[1:])] + [real[0] * 1.5, real[-2] / 3,
                                                                                    0.0, -1.0, 1e-9]
                for sigma in probes:
                    with self.subTest(steps=steps, sigma=sigma):
                        self.assertEqual(self.pack.sigma_to_value(sigma, sigmas, schedule),
                                         self.ext_schedule.sigma_to_value(sigma, sigmas, schedule))
        self.assertEqual(self.pack.sigma_to_value(0.5, _flow_sigmas(4), []), 0.0)

    def test_basis_helpers_equal_the_extensions(self):
        class Wan21:
            pass

        class Flux:
            pass

        class Flux2:
            pass

        class SDXL:
            pass

        for fmt in (Wan21, Flux, Flux2, SDXL, Wan21(), Flux2(), None):
            self.assertEqual(self.pack_basis.family_for_latent_format(fmt),
                             self.ext_basis.family_for_latent_format(fmt))
        for family in ("krea2", "zimage", "flux2", None, "sdxl"):
            self.assertEqual(self.pack_basis.vae_downscale_factor(family), self.ext_basis.vae_downscale_factor(family))
        rng = random.Random(1)
        for family in ("krea2", "zimage", "flux2"):
            for _ in range(20):
                params = {"recenter_override": rng.random() < 0.5, "recenter": rng.random(),
                          "max_chroma_override": rng.random() < 0.5, "max_chroma": rng.uniform(0, 10),
                          "chroma_plane_override": rng.random() < 0.5, "chroma_plane": rng.choice(("lab", "temp_tint"))}
                self.assertEqual(self.pack_basis.resolve_dev(params, family), self.ext_basis.resolve_dev(params, family))
                self.assertEqual(self.pack_basis.resolve_dev({}, family), self.ext_basis.resolve_dev({}, family))


# --- PB-CC3: host part 1 (settings_json, prepare, anchors) ---------------------------------------------------

def _schedule(**values):
    sched = {"strength": 1.0, "start": 0.0, "end": 1.0, "bias": 0.5, "exponent": 0.0, "start_off": 0.0,
             "end_off": 0.0, "smooth": True}
    sched.update(values)
    return sched


def _leaf(axis="exposure", mode="highs", **values):
    leaf = {"mask_axis": axis, "mask_mode": mode, "mask_center": 0.0, "mask_hardness": 1.0, "mask_width": 0.0,
            "mask_strength": 1.0}
    leaf.update(values)
    return leaf


def _settings(*entries, pass_name="base", **extra):
    data = {"v": 1, "pass": pass_name, "entries": list(entries), "infotext": "v1;mods=I"}
    data.update(extra)
    return data


def _luma(exposure=0.5, mask=None):
    return {"kind": "luma", "params": {"exposure": exposure, "tone_compression": 0.0}, "schedule": _schedule(),
            "mask": mask}


def _basic(contrast=0.4, amount=0.0):
    return {"kind": "basic", "params": {"contrast": contrast, "color_shift_amount": amount, "mode": "default",
                                        "red": 0.1, "green": 0.0, "blue": -0.2, "brightness": 0.0},
            "schedule": _schedule(), "mask": None}


class HostSettingsTests(unittest.TestCase):
    """``colorcraft_host.parse_settings``: the node's settings_json contract (no torch, no extension)."""

    @classmethod
    def setUpClass(cls):
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host

        cls.host = colorcraft_host

    def test_valid_settings_round_trip_with_numbers_as_floats(self):
        nested = {"operation": "subtract", "a": {"blur": 8, "spread": 0, "contrast": 0, "normalize": False,
                                                 "a": _leaf("hue", "range", mask_width=1)},
                  "b": _leaf("lab-a+b")}
        data = _settings(_luma(1, mask=nested), _basic(), pass_name="hires", tags=["I", "III"])
        parsed = self.host.parse_settings(json.dumps(data))
        self.assertEqual(parsed["pass"], "hires")
        self.assertEqual(parsed["tags"], ["I", "III"])
        self.assertEqual(parsed["infotext"], "v1;mods=I")
        self.assertEqual(parsed["entries"][0]["params"]["exposure"], 1.0)
        self.assertIsInstance(parsed["entries"][0]["params"]["exposure"], float)
        self.assertIsInstance(parsed["entries"][0]["mask"]["a"]["blur"], float)
        self.assertEqual(parsed["entries"], json.loads(json.dumps(data["entries"])))
        self.assertEqual(self.host.parse_settings(data), parsed)  # a dict is read the same way
        self.assertEqual(self.host.parse_settings(json.dumps(_settings()))["entries"], [])
        self.assertEqual(self.host.parse_settings(json.dumps(_settings(_luma(), _luma())))["tags"], ["#1", "#2"])

    def test_out_of_range_values_are_kept_like_the_extension(self):
        parsed = self.host.parse_settings(json.dumps(_settings(_luma(exposure=12.5))))
        self.assertEqual(parsed["entries"][0]["params"]["exposure"], 12.5)

    def test_malformed_settings_are_refused_with_a_reason(self):
        good = _settings(_luma(mask=_leaf()), _basic())

        def mutate(change):
            data = json.loads(json.dumps(good))
            change(data)
            return data

        cases = {
            "not json": "{",
            "not an object": "[1]",
            "version 2": mutate(lambda d: d.update(v=2)),
            "version true": mutate(lambda d: d.update(v=True)),
            "unknown key": mutate(lambda d: d.update(seed=3)),
            "pass": mutate(lambda d: d.update({"pass": "img2img"})),
            "entries not a list": mutate(lambda d: d.update(entries={})),
            "kind": mutate(lambda d: d["entries"][0].update(kind="tone")),
            "entry key missing": mutate(lambda d: d["entries"][0].pop("mask")),
            "param missing": mutate(lambda d: d["entries"][0]["params"].pop("tone_compression")),
            "param extra": mutate(lambda d: d["entries"][0]["params"].update(vibrance=0.1)),
            "bool for number": mutate(lambda d: d["entries"][0]["params"].update(exposure=True)),
            "string for number": mutate(lambda d: d["entries"][0]["params"].update(exposure="0.5")),
            "nan": json.dumps(good).replace('"exposure": 0.5', '"exposure": NaN'),
            "infinity": json.dumps(good).replace('"exposure": 0.5', '"exposure": Infinity'),
            "choice": mutate(lambda d: d["entries"][1]["params"].update(mode="hsv")),
            "schedule key": mutate(lambda d: d["entries"][0]["schedule"].pop("smooth")),
            "smooth not bool": mutate(lambda d: d["entries"][0]["schedule"].update(smooth=1)),
            "mask on basic": mutate(lambda d: d["entries"][1].update(mask=_leaf())),
            "mask axis": mutate(lambda d: d["entries"][0]["mask"].update(mask_axis="luma")),
            "mask mode": mutate(lambda d: d["entries"][0]["mask"].update(mask_mode="mids")),
            "mask leaf key": mutate(lambda d: d["entries"][0]["mask"].pop("mask_width")),
            "combine op": mutate(lambda d: d["entries"][0].update(mask={"operation": "nand", "a": _leaf(),
                                                                         "b": _leaf()})),
            "shaping key": mutate(lambda d: d["entries"][0].update(mask={"blur": 1.0, "a": _leaf()})),
            "infotext": mutate(lambda d: d.update(infotext=3)),
            "tags length": mutate(lambda d: d.update(tags=["I"])),
            "tags empty name": mutate(lambda d: d.update(tags=["I", ""])),
        }
        deep = _leaf()
        for _ in range(40):
            deep = {"operation": "and", "a": deep, "b": _leaf()}
        cases["too deep"] = _settings(_luma(mask=deep))
        for label, text in cases.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, "Colorcraft 설정을 읽지 못했습니다"):
                    self.host.parse_settings(text)

    def test_mask_axes_are_the_vendored_upstream_list(self):
        tree = ast.parse((VENDOR / "lib_colorcraft" / "masking.py").read_text(encoding="utf-8"))
        axes = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", None) == "MASK_AXIS_OPTIONS")
        self.assertEqual(list(self.host.MASK_AXES), axes)

    def test_noop_and_vector_free_rules(self):
        noop = self.host.params_are_noop
        advanced = {name: 0.0 for name in self.host.KIND_PARAMS["advanced"]}
        advanced.update(more_colors=False, color_shift=False, mode="default", chroma_plane="temp_tint",
                        recenter_override=False, max_chroma_override=False, chroma_plane_override=False,
                        lab_a=1.0, color_shift_amount=1.0, chroma_center=0.7, recenter=0.5)
        self.assertTrue(noop("advanced", advanced))          # gated amounts and chroma_center alone do nothing
        self.assertFalse(noop("advanced", dict(advanced, more_colors=True)))
        self.assertFalse(noop("advanced", dict(advanced, color_shift=True)))
        self.assertTrue(self.host.vector_free(_basic(contrast=0.3)))
        self.assertFalse(self.host.vector_free(_luma()))
        kept = self.host.without_anchor_edits([("I", _basic(contrast=0.3, amount=0.2)), ("II", _luma())])
        self.assertEqual([tag for tag, _entry in kept], ["II"])

    def test_anchor_cache_keeps_weak_references_and_a_bound(self):
        class Module:
            pass

        cache = self.host.AnchorCache(limit=3)
        modules = [Module() for _ in range(5)]
        for index, module in enumerate(modules):
            cache.put(module, (index,), f"anchor{index}")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get(modules[0], (0,)))           # least recently used out
        self.assertEqual(cache.get(modules[4], (4,)), "anchor4")
        self.assertIsNone(cache.get(modules[3], (4,)))           # another module never gets it
        self.assertEqual(len(cache), 2)                          # ... and the mismatch drops the entry
        cache.put(5, (9,), "x")                                  # not weakly referenceable: not cached
        self.assertIsNone(cache.get(5, (9,)))
        self.assertEqual(len(cache), 2)                          # entries of modules 2 and 3
        del modules[3], module
        gc.collect()
        cache.put(modules[0], (0,), "again")                     # a put purges the dead encoder's entry
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.get(modules[2], (2,)), "anchor2")
        self.assertEqual(cache.get(modules[0], (0,)), "again")


class _FakeModel:
    """A ComfyUI MODEL stand-in for ``prepare`` (``get_model_object('latent_format')``)."""

    def __init__(self, latent_format, *, getter=True):
        self._latent_format = latent_format
        if not getter:
            self.get_model_object = None
        self.model = SimpleNamespace(latent_format=latent_format)

    def get_model_object(self, name):
        if name != "latent_format":
            raise AttributeError(name)
        return self._latent_format


def _fake_latent_format(name, channels, rank):
    """A latent format class named like ComfyUI's/Forge's (the family is read off the name). Like the real ones,
    its per-channel buffers have the latent's own rank (Wan21: [1, C, 1, 1, 1]) and follow the input's dtype, so
    an anchor lifted with the wrong rank comes out the wrong size and a half-precision one stays half."""

    def process_in(self, latent):
        shape = (1, channels) + (1,) * rank
        shift = self.shift.view(shape).to(latent.dtype)
        return (latent - shift) * self.scale.view(shape).to(latent.dtype)

    generator = torch.Generator().manual_seed(zlib.crc32(name.encode("utf-8")))
    attrs = {"process_in": process_in, "latent_dimensions": rank,
             "shift": torch.randn(channels, generator=generator) * 0.1,
             "scale": 0.5 + torch.rand(channels, generator=generator)}
    return type(name, (), attrs)()


class _FakeVAE:
    """ComfyUI-VAE-shaped stand-in: ``encode([B, H, W, 3] in 0..1)`` -> a deterministic colour-dependent latent."""

    def __init__(self, channels=16, rank=2, dtype=None, seed=0, fail=False):
        self.first_stage_model = torch.nn.Identity()
        self.vae_dtype = dtype or torch.float32
        self.device = torch.device("cpu")
        self.patcher = SimpleNamespace(patches_uuid="uuid-1")
        self.latent_dim = rank
        generator = torch.Generator().manual_seed(seed)
        self.mix = torch.randn(channels, 3, generator=generator)
        self.bias = torch.randn(channels, generator=generator)
        self.fail = fail
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("fake VAE: out of memory")
        pixels = torch.nn.functional.avg_pool2d(image.movedim(-1, 1).float(), 8)
        latent = torch.tanh(torch.einsum("kc,bchw->bkhw", self.mix, pixels) + self.bias.view(1, -1, 1, 1))
        latent = latent * 1.7 + pixels.mean(dim=1, keepdim=True) ** 2
        latent = latent.to(self.vae_dtype)
        return latent.unsqueeze(2) if self.latent_dim == 3 else latent


class _FakeUnet:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})

    def clone(self):
        return _FakeUnet(self.model_options)


@requires_torch
@requires_extension
class HostPrepareParityTests(unittest.TestCase):
    """``prepare``/``encode_anchors`` = the extension hook's ``process``/``encode_anchors`` (CPU, exact)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_basis, colorcraft_host

        cls.host, cls.pack_basis = colorcraft_host, colorcraft_basis
        cls.hook = _ext_module("sam3ext.colorcraft.hook")
        cls.spec = _ext_module("sam3ext.colorcraft.spec")
        cls.panel = _ext_module("sam3ext.colorcraft.panel_state")
        cls.cpu = torch.device("cpu")
        cls.formats = {"krea2": _fake_latent_format("Wan21", 16, 3), "zimage": _fake_latent_format("Flux", 16, 2),
                       "flux2": _fake_latent_format("Flux2", 128, 2), None: _fake_latent_format("SDXL", 4, 2)}

    def setUp(self):
        self.host.ANCHORS.clear()
        self.hook.clear_anchor_cache()

    def _vae(self, family, **kwargs):
        channels = FAMILY_CHANNELS.get(family, 4)
        rank = 3 if family == "krea2" else 2
        return _FakeVAE(channels, rank, **kwargs)

    def test_contract_tables_equal_the_extensions(self):
        theirs = {kind: tuple(node for node, _panel in pairs) for kind, pairs in self.spec._KIND_PARAMS.items()}
        self.assertEqual(self.host.KIND_PARAMS, theirs)
        self.assertEqual(self.host.SCHEDULE_KEYS, self.spec._SCHEDULE_FIELDS)
        self.assertEqual(self.host.TAKES_MASK, self.spec.KIND_TAKES_MASK)
        self.assertEqual(list(self.host.MASK_MODES), self.spec.MASK_MODE_OPTIONS)
        self.assertEqual(list(self.host.COMBINE_OPERATIONS), self.spec.COMBO_OP_OPTIONS)
        self.assertEqual(list(self.host.MASK_AXES), list(_ext_module("sam3ext.colorcraft.masking").MASK_AXIS_OPTIONS))
        self.assertEqual(list(self.host.CHOICE_PARAMS["mode"]), self.spec.COLOR_SHIFT_MODE_OPTIONS)
        self.assertEqual(list(self.host.CHOICE_PARAMS["chroma_plane"]), self.spec.CHROMA_PLANE_OPTIONS)
        flags = {name for name, field_ in self.spec.MODIFIER_BY_NAME.items() if field_.kind == self.spec.BOOL}
        panel_to_node = {panel: node for pairs in self.spec._KIND_PARAMS.values() for node, panel in pairs}
        self.assertEqual(self.host.BOOL_PARAMS, {panel_to_node[name] for name in flags if name in panel_to_node})

    def test_every_extension_chain_passes_parse_settings_unchanged(self):
        rng = random.Random(41)
        seen = 0
        for case in range(150):
            config = _random_config(self.spec, rng)
            chain = self.spec.build_chain(config, rng.random() < 0.5)
            data = _settings(*chain.entries, tags=chain.tags)
            with self.subTest(case=case):
                parsed = self.host.parse_settings(json.dumps(data))
                self.assertEqual(parsed["entries"], chain.entries)
                self.assertEqual(parsed["tags"], chain.tags)
                seen += len(chain.entries)
        self.assertGreater(seen, 150)

    def test_noop_vector_free_and_anchor_edit_rules_equal_the_extensions(self):
        rng = random.Random(8)
        for case in range(300):
            config = _random_config(self.spec, rng)
            for modifier in config.modifiers:
                for name in rng.sample(list(self.spec.MODIFIER_BY_NAME), 12):
                    if self.spec.MODIFIER_BY_NAME[name].kind == self.spec.FLOAT:
                        modifier[name] = 0.0
            entries = self.spec.build_chain(config, False).entries
            for modifier in config.modifiers:
                kind = self.spec.kind_of(modifier)
                params = self.spec.params_of(modifier)
                self.assertEqual(self.host.params_are_noop(kind, params), self.spec.params_are_noop(kind, params))
            pairs = [(f"#{i}", entry) for i, entry in enumerate(entries)]
            with self.subTest(case=case):
                self.assertEqual([self.host.vector_free(entry) for entry in entries],
                                 [self.hook._vector_free(entry) for entry in entries])
                self.assertEqual(self.host.without_anchor_edits(pairs), self.hook._without_anchor_edits(pairs))

    def test_anchors_equal_the_extensions(self):
        colours = [(0.0, 0.0, 0.0, 0.0), (0.4, -0.2, 0.1, 0.05), (1.5, 2.0, -2.0, -1.0), (0.12344, 0.0, 0.0, 0.0),
                   (0.12341, 0.0, 0.0, 0.0), (-0.0, 0.3333333, 0.0, 0.0)]
        for family in ("krea2", "zimage", "flux2", None):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(family=family, dtype=str(dtype)):
                    self.setUp()
                    fmt = self.formats[family]
                    ours = self.host.encode_anchors(self._vae(family, dtype=dtype), fmt, colours, device=self.cpu)
                    theirs = self.hook.encode_anchors(self._vae(family, dtype=dtype), fmt, colours, {},
                                                      device=self.cpu)
                    self.assertEqual(list(ours), list(theirs))
                    self.assertEqual(len(ours), 5)               # 0.12344 and 0.12341 share upstream's key
                    for key in ours:
                        self.assertEqual(ours[key].dtype, torch.float32)
                        self.assertTrue(torch.equal(ours[key], theirs[key]), key)
        # no latent format / one without process_in: the VAE-space anchor as is
        for fmt in (None, SimpleNamespace()):
            ours = self.host.encode_anchors(self._vae("zimage"), fmt, colours[:2], device=self.cpu)
            theirs = self.hook.encode_anchors(self._vae("zimage"), fmt, colours[:2], {}, device=self.cpu)
            for key in ours:
                self.assertTrue(torch.equal(ours[key], theirs[key]))

    def test_anchor_cache_encodes_as_often_as_the_extensions(self):
        fmt = self.formats["krea2"]
        sides = {"pack": (self._vae("krea2"), lambda vae, colours: self.host.encode_anchors(
                     vae, fmt, colours, device=self.cpu)),
                 "ext": (self._vae("krea2"), lambda vae, colours: self.hook.encode_anchors(
                     vae, fmt, colours, {}, device=self.cpu))}
        calls = {}
        for side, (vae, encode) in sides.items():
            log = []
            encode(vae, [(0.0,) * 4, (0.5, 0.0, 0.0, 0.0)])
            log.append(vae.calls)
            encode(vae, [(0.0,) * 4, (0.5, 0.0, 0.0, 0.0), (0.0, 0.5, 0.0, 0.0)])   # one new colour
            log.append(vae.calls)
            vae.patcher.patches_uuid = "uuid-2"                                     # LoRA patches changed
            encode(vae, [(0.0,) * 4])
            log.append(vae.calls)
            vae.vae_dtype = torch.bfloat16                                          # another compute dtype
            encode(vae, [(0.0,) * 4])
            log.append(vae.calls)
            for index in range(40):                                                 # beyond the bound
                encode(vae, [(index / 100.0, 0.0, 0.0, 0.0)])
            encode(vae, [(0.0,) * 4])
            log.append(vae.calls)
            calls[side] = log
        self.assertEqual(calls["pack"], calls["ext"])
        self.assertEqual(calls["pack"][:4], [2, 3, 4, 5])
        self.assertEqual(self.host.ANCHOR_CACHE_LIMIT, self.hook.ANCHOR_CACHE_LIMIT)

    def _process(self, config, family, *, hires=False, vae=None, pre_dd=True):
        """The extension's ``process`` on a Forge-shaped stand-in ``p`` and the pack's ``prepare`` on the same
        chain: (Run or None, Prepared)."""
        text = self.spec.to_infotext(config)
        args = [text] + self.panel.default_args()[1:]
        fmt = self.formats[family]
        ext_vae = vae() if vae else self._vae(family)
        p = SimpleNamespace(sd_model=SimpleNamespace(model_config=SimpleNamespace(latent_format=fmt),
                                                     forge_objects=SimpleNamespace(vae=ext_vae, unet=_FakeUnet())),
                            is_hr_pass=hires, extra_generation_params={}, sampler=None)
        with mock.patch("sys.stderr", new=io.StringIO()):
            run = self.hook.process(p, args)
        chain = self.spec.build_chain(self.panel.config_from_script_args(args), hires)
        settings = self.host.parse_settings(json.dumps(_settings(*chain.entries, tags=chain.tags,
                                                                 pass_name="hires" if hires else "base",
                                                                 infotext=text)))
        prepared = self.host.prepare(_FakeModel(fmt), vae() if vae else self._vae(family), settings, pre_dd,
                                     device=self.cpu)
        return run, prepared, p

    def _assert_same_pass(self, run, prepared, p, label):
        if run is None:
            self.assertEqual(prepared.entries, [], label)
            status = (getattr(p, "_sam3_colorcraft", None) or {}).get("passes", {}).get(prepared.pass_name)
            if status is not None:  # the extension got as far as a status line: same notes
                theirs = [note for note in status.notes if not note.startswith("incomplete combo")]
                self.assertEqual(prepared.notes, theirs, label)
            return
        self.assertEqual(prepared.entries, run.entries, label)
        self.assertEqual(prepared.tags, run.tags, label)
        self.assertEqual(prepared.family, run.family, label)
        self.assertEqual(prepared.downscale, run.downscale, label)
        self.assertEqual(prepared.family_label, run.status.family, label)
        self.assertEqual(list(prepared.anchors), list(run.anchors), label)
        for key in prepared.anchors:
            self.assertTrue(torch.equal(prepared.anchors[key], run.anchors[key]), f"{label} {key}")
        theirs = [note for note in run.status.notes if not note.startswith("incomplete combo")]
        self.assertEqual(prepared.notes, theirs, label)

    def test_prepare_equals_the_extensions_process(self):
        rng = random.Random(1004)
        attached = 0
        for case in range(80):
            config = _random_config(self.spec, rng)
            family = ("krea2", "zimage", "flux2", None)[case % 4]
            hires = rng.random() < 0.4
            with self.subTest(case=case, family=family, hires=hires):
                self.setUp()
                run, prepared, p = self._process(config, family, hires=hires)
                self._assert_same_pass(run, prepared, p, f"case {case}")
                attached += run is not None
        self.assertGreater(attached, 30)

    def test_prepare_with_a_failing_vae_equals_the_extension(self):
        rng = random.Random(77)
        for case in range(24):
            config = _random_config(self.spec, rng)
            family = ("krea2", None, "zimage")[case % 3]
            with self.subTest(case=case, family=family):
                run, prepared, p = self._process(config, family, vae=lambda: self._vae(family, fail=True))
                self._assert_same_pass(run, prepared, p, f"failing VAE {case}")

    def test_prepare_with_a_missing_vector_file_equals_the_extension(self):
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch.object(self.pack_basis, "vectors_dir", lambda family: empty), \
                    mock.patch.object(self.hook.basis_mod, "VECTORS_DIR", empty), \
                    mock.patch.dict(self.hook._BASIS_CACHE, clear=True):
                for case in range(12):
                    config = _random_config(self.spec, rng)
                    with self.subTest(case=case):
                        run, prepared, p = self._process(config, "krea2")
                        self._assert_same_pass(run, prepared, p, f"missing vectors {case}")
                        if prepared.entries:
                            self.assertIn("colorcraft-krea2.safetensors missing: vector controls off", prepared.notes)

    def test_latent_format_is_read_from_the_model_object_or_the_inner_model(self):
        fmt = self.formats["zimage"]
        self.assertIs(self.host.model_latent_format(_FakeModel(fmt)), fmt)
        self.assertIs(self.host.model_latent_format(_FakeModel(fmt, getter=False)), fmt)

        class Broken:
            model = SimpleNamespace(latent_format=fmt)

            def get_model_object(self, name):
                raise AttributeError(name)

        self.assertIs(self.host.model_latent_format(Broken()), fmt)
        self.assertIsNone(self.host.model_latent_format(object()))


# --- PB-CC4: host part 2 (run state, SAMPLER_SAMPLE wrapper, post-CFG) ---------------------------------------

class _FakePatcher:
    """A ComfyUI ModelPatcher stand-in: options cloned like ``create_model_options_clone``, keyed wrappers."""

    def __init__(self, model_options=None, wrappers=None, latent_format=None):
        self.model_options = model_options if model_options is not None else {}
        self.wrappers = wrappers if wrappers is not None else {}
        self.latent_format = latent_format
        self.model = SimpleNamespace(latent_format=latent_format)

    def get_model_object(self, name):
        if name != "latent_format":
            raise AttributeError(name)
        return self.latent_format

    def clone(self):
        def copy_nested(value):
            if isinstance(value, dict):
                return {k: copy_nested(v) for k, v in value.items()}
            return list(value) if isinstance(value, list) else value

        wrappers = {kind: {key: list(fns) for key, fns in keyed.items()} for kind, keyed in self.wrappers.items()}
        return _FakePatcher(copy_nested(self.model_options), wrappers, self.latent_format)

    def add_wrapper_with_key(self, wrapper_type, key, function):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(function)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


class _Executor:
    """``WrapperExecutor`` stand-in: calling it runs ``body`` (the sampler) with the run's arguments."""

    def __init__(self, body):
        self.body = body
        self.class_obj = SimpleNamespace(sampler_function=None)

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)


def _evals(walked, rng, *, family, batch=2, hw=(40, 48), five_d=None, extra=()):
    """One sampling run's evaluations: every step's sigma and its midpoint (second-order samplers), in order."""
    five_d = family == "krea2" if five_d is None else five_d
    sigmas = [float(s) for s in walked[:-1]]
    probes = []
    for index, sigma in enumerate(sigmas):
        probes.append(sigma)
        if index + 1 < len(sigmas) and rng.random() < 0.5:
            probes.append((sigma + sigmas[index + 1]) / 2)
    out = []
    for index, sigma in enumerate(probes):
        seed = rng.randrange(1 << 30)
        out.append({"x0": _latent(family, seed, five_d=five_d, batch=batch, height=hw[0], width=hw[1]),
                    "sigma": sigma})
    for item in extra:
        out.insert(min(item.pop("at"), len(out)), item)
    return out


@requires_torch
@requires_extension
class HostRunParityTests(unittest.TestCase):
    """The pack's wrapper + post-CFG function grade every evaluation like the extension's Forge hook
    (``process`` + its post-CFG callback / ``run_evaluation``) on the same tensors (CPU, exact)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host, guidance_common

        cls.host, cls.common = colorcraft_host, guidance_common
        cls.hook = _ext_module("sam3ext.colorcraft.hook")
        cls.spec = _ext_module("sam3ext.colorcraft.spec")
        cls.panel = _ext_module("sam3ext.colorcraft.panel_state")
        cls.dave = _ext_module("sam3ext.guidance.dave_gate")
        cls.ext_sigmas = _ext_module("sam3ext.guidance.sigmas")
        cls.cpu = torch.device("cpu")
        cls.formats = {"krea2": _fake_latent_format("Wan21", 16, 3), "zimage": _fake_latent_format("Flux", 16, 2),
                       "flux2": _fake_latent_format("Flux2", 128, 2), None: _fake_latent_format("SDXL", 4, 2)}

    def setUp(self):
        self.host.ANCHORS.clear()
        self.hook.clear_anchor_cache()
        self.dave.note_pre_dd_sigma(None, None)
        self.addCleanup(self.dave.note_pre_dd_sigma, None, None)

    def _vae(self, family):
        return _FakeVAE(FAMILY_CHANNELS.get(family, 4), 3 if family == "krea2" else 2)

    def _setup(self, config, family, *, hires=False, pre_dd=True, hw=(40, 48)):
        """(ext callback, ext Run, pack ColorcraftPatch) for one pass of ``config``."""
        text = self.spec.to_infotext(config)
        args = [text] + self.panel.default_args()[1:]
        fmt = self.formats[family]
        unet = _FakeUnet()
        p = SimpleNamespace(sd_model=SimpleNamespace(model_config=SimpleNamespace(latent_format=fmt),
                                                     forge_objects=SimpleNamespace(vae=self._vae(family), unet=unet)),
                            is_hr_pass=hires, extra_generation_params={}, sampler=None)
        with mock.patch("sys.stderr", new=io.StringIO()):
            run = self.hook.process(p, args, full_hw=hw)
        chain = self.spec.build_chain(self.panel.config_from_script_args(args), hires)
        settings = self.host.parse_settings(json.dumps(_settings(*chain.entries, tags=chain.tags,
                                                                 pass_name="hires" if hires else "base")))
        prepared = self.host.prepare(_FakeModel(fmt), self._vae(family), settings, pre_dd, device=self.cpu)
        if run is None:
            self.assertEqual(prepared.entries, [])
            return None, None, None
        run.pre_dd = pre_dd
        callback = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"][-1]
        return callback, run, self.host.ColorcraftPatch(prepared, settings)

    def _run_both(self, callback, run, patch, full, offset, evals, *, hw=(40, 48), label=""):
        """Run ``evals`` through the extension callback (Forge's published ``full`` list) and through the pack's
        wrapper + post-CFG (ComfyUI's ``sample_sigmas`` = the walked tail); compare every result."""
        whole = full[offset:].clone()
        published_ext = {"list": full}
        transformer = {"sample_sigmas": whole}
        extra_args = {"model_options": {"transformer_options": transformer}, "seed": 1}
        noise = torch.zeros(1, 4, *hw)
        ours, theirs = [], []

        def sampler(guider, sigmas, extra_args_, callback_, noise_, *rest):
            for item in evals:
                batch = item["x0"].shape[0]
                if item.get("publish") == "speed":
                    patched_tail = whole.clone()
                    patched_tail[item["index"]] = item["value"]
                    transformer["sample_sigmas"] = self.common.mark_republished(patched_tail, whole)
                    patched_full = full.clone()
                    patched_full[offset + item["index"]] = item["value"]
                    published_ext["list"] = self.ext_sigmas.mark_republished(patched_full, full)
                elif item.get("publish") == "foreign":
                    transformer["sample_sigmas"] = whole.clone()
                    published_ext["list"] = full.clone()
                elif item.get("publish") == "restore":
                    transformer["sample_sigmas"] = whole
                    published_ext["list"] = full
                # per-image sigmas a little apart: both hosts look the schedule up by their max
                spread = torch.tensor([1.0 - 0.01 * i for i in range(batch)])
                sigma = item["sigma"] * spread
                model_sigma = item["scaled"] * spread if "scaled" in item else sigma
                if "scaled" in item:
                    self.dave.note_pre_dd_sigma(sigma, model_sigma)
                    transformer[self.common.PRE_DD_SIGMAS_KEY] = sigma
                else:
                    self.dave.note_pre_dd_sigma(None, None)
                    transformer.pop(self.common.PRE_DD_SIGMAS_KEY, None)
                ext_transformer = {"sampling_sigmas": published_ext["list"]}
                pack_options = extra_args_["model_options"]
                if item.get("substep"):
                    ext_transformer[self.common.SUBSTEP_MARKER] = "dy"
                    pack_options = dict(pack_options, transformer_options=dict(
                        pack_options["transformer_options"], **{self.common.SUBSTEP_MARKER: "dy"}))
                x_ext, x_pack = item["x0"].clone(), item["x0"].clone()
                theirs.append((x_ext, callback({"denoised": x_ext, "sigma": model_sigma,
                                                "model_options": {"transformer_options": ext_transformer}})))
                ours.append((x_pack, patch.post_cfg({"denoised": x_pack, "sigma": model_sigma,
                                                     "model_options": pack_options})))
            return "samples"

        self.assertEqual(patch.wrapper(_Executor(sampler), None, whole, extra_args, None, noise), "samples")
        self.assertEqual(len(ours), len(evals))
        for index, ((x_pack, out_pack), (x_ext, out_ext)) in enumerate(zip(ours, theirs)):
            where = f"{label} eval {index}"
            self.assertEqual(out_pack is x_pack, out_ext is x_ext, f"{where}: pass-through differs")
            self.assertEqual(out_pack.dtype, out_ext.dtype, where)
            torch.testing.assert_close(out_pack, out_ext, rtol=0, atol=0, equal_nan=True,
                                       msg=lambda m: f"{where}: {m}")
        report, status = patch.last_report, run.status
        for name in ("evals", "edited", "idle", "foreign", "substeps", "errors", "nonfinite"):
            self.assertEqual(getattr(report, name), getattr(status, name), f"{label}: {name}")
        if run.built is None and status.errors:
            # The schedules cannot be built (an out-of-range typed Start/End): every evaluation passes through
            # as an error on both hosts. The extension builds them at its first evaluation and names that
            # error, then the follow-on TypeError of its missing schedules; the pack names the cause each time.
            self.assertEqual(report.last_error, type(patch.last_state.setup_error).__name__, label)
        else:
            self.assertEqual(report.last_error, status.last_error, label)
        self.assertEqual(report.family, status.family, label)
        self.assertEqual(report.tags, status.tags, label)
        return report

    def test_random_passes_on_every_family(self):
        rng = random.Random(404)
        edited = 0
        for case in range(40):
            config = _random_config(self.spec, rng)
            family = ("krea2", "zimage", "flux2", None)[case % 4]
            with self.subTest(case=case, family=family):
                self.setUp()
                callback, run, patch = self._setup(config, family)
                if callback is None:
                    continue
                full = _flow_sigmas(rng.randint(4, 16))
                report = self._run_both(callback, run, patch, full, 0, _evals(full, rng, family=family),
                                        label=f"case {case}")
                edited += report.edited
        self.assertGreater(edited, 100)

    def _strong(self, *, hires=False, masked=True):
        config = self.spec.default_config()
        config.enabled, config.masking = True, masked
        config.modifiers[0].update({"active": True, "kind": "Advanced", "pass": "Both", "start": 0.1, "end": 0.9,
                                    "exposure": 0.6, "contrast": 0.8, "temperature": 1.1, "clarity": 1.5,
                                    "color_shift_amount": 0.5, "color_shift_red": 0.3,
                                    "mask": "M1" if masked else "none"})
        config.modifiers[1].update({"active": True, "kind": "Chroma", "pass": "Hires" if hires else "Base",
                                    "start": 0.0, "end": 1.0, "saturation": 0.5, "strength": -0.7})
        config.leaves[0].update({"mask_axis": "exposure", "mask_mode": "highs", "blur": 12.0, "mask_hardness": 2.0})
        return config

    def test_hires_tail_lists(self):
        rng = random.Random(9)
        for family in ("krea2", "zimage"):
            for offset in (3, 9):
                with self.subTest(family=family, offset=offset):
                    self.setUp()
                    callback, run, patch = self._setup(self._strong(hires=True), family, hires=True)
                    full = _flow_sigmas(20)
                    report = self._run_both(callback, run, patch, full, offset,
                                            _evals(full[offset:], rng, family=family), label=f"hires {offset}")
                    self.assertGreater(report.edited, 5)

    def test_detail_daemon_note_with_pre_dd_on_and_off(self):
        rng = random.Random(10)
        for pre_dd in (True, False):
            with self.subTest(pre_dd=pre_dd):
                self.setUp()
                callback, run, patch = self._setup(self._strong(), "krea2", pre_dd=pre_dd)
                full = _flow_sigmas(12)
                evals = _evals(full, rng, family="krea2")
                for item in evals[1:-1]:
                    item["scaled"] = item["sigma"] * 0.93
                report = self._run_both(callback, run, patch, full, 0, evals, label=f"pre_dd {pre_dd}")
                self.assertEqual(report.pre_dd > 0, pre_dd)

    def test_substeps_republished_and_foreign_lists(self):
        rng = random.Random(11)
        for family in ("krea2", "flux2"):
            with self.subTest(family=family):
                self.setUp()
                callback, run, patch = self._setup(self._strong(), family)
                full = _flow_sigmas(12)
                five_d = family == "krea2"

                def item(at, **values):
                    x0 = _latent(family, rng.randrange(1 << 30), five_d=five_d, height=40, width=48)
                    return dict(values, at=at, x0=x0)

                aligned = float(full[5]) * 0.97
                extra = [item(2, sigma=float(full[1]), substep=True),
                         item(7, sigma=aligned, publish="speed", index=5, value=aligned),
                         item(8, sigma=float(full[6])),
                         item(9, sigma=float(full[6]), publish="foreign"),
                         item(10, sigma=float(full[7]), publish="restore")]
                report = self._run_both(callback, run, patch, full, 0, _evals(full, rng, family=family, extra=extra),
                                        label=f"lists {family}")
                self.assertEqual((report.substeps, report.foreign), (1, 1))

    def test_half_precision_non_finite_and_errors(self):
        rng = random.Random(12)
        full = _flow_sigmas(8)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                self.setUp()
                callback, run, patch = self._setup(self._strong(), "zimage")
                evals = _evals(full, rng, family="zimage")
                for item in evals:
                    item["x0"] = item["x0"].to(dtype)
                self._run_both(callback, run, patch, full, 0, evals, label=f"half {dtype}")
        with self.subTest(case="non-finite"):
            self.setUp()
            config = self._strong(masked=False)
            config.modifiers[0].update({"strength": 1e38, "exposure": 4.0})
            callback, run, patch = self._setup(config, "zimage")
            report = self._run_both(callback, run, patch, full, 0, _evals(full, rng, family="zimage"),
                                    label="non-finite")
            self.assertGreater(report.nonfinite, 0)
        with self.subTest(case="missing anchor"):
            self.setUp()
            callback, run, patch = self._setup(self._strong(masked=False), "zimage")
            run.anchors.clear()
            patch.prepared.anchors.clear()
            report = self._run_both(callback, run, patch, full, 0, _evals(full, rng, family="zimage"),
                                    label="missing anchor")
            self.assertGreater(report.errors, 0)
            self.assertEqual(report.last_error, "KeyError")

    def test_coarse_grid_keeps_the_blur_in_image_pixels(self):
        rng = random.Random(13)
        for family in ("krea2", "flux2"):
            with self.subTest(family=family):
                self.setUp()
                callback, run, patch = self._setup(self._strong(), family, hw=(32, 40))
                full = _flow_sigmas(10)
                evals = _evals(full, rng, family=family, hw=(16, 20))      # SPEED's half-size stage
                evals += _evals(full, rng, family=family, hw=(32, 40))[-3:]
                report = self._run_both(callback, run, patch, full, 0, evals, hw=(32, 40), label=f"grid {family}")
                self.assertGreater(report.edited, 3)

    def test_numerically_empty_edits_count_as_idle(self):
        rng = random.Random(15)
        config = self.spec.default_config()
        config.enabled = True
        config.modifiers[0].update({"active": True, "kind": "Luma", "start": 0.0, "end": 1.0, "strength": 1e-300,
                                    "exposure": 1e-30})
        callback, run, patch = self._setup(config, "zimage")
        full = _flow_sigmas(6)
        report = self._run_both(callback, run, patch, full, 0, _evals(full, rng, family="zimage"), label="underflow")
        self.assertGreater(report.idle, 3)
        self.assertEqual(report.edited, 0)

    def test_idle_steps_return_the_same_object(self):
        rng = random.Random(14)
        config = self.spec.default_config()
        config.enabled = True
        config.modifiers[0].update({"active": True, "kind": "Luma", "start": 0.5, "end": 0.6, "exposure": 1.0})
        callback, run, patch = self._setup(config, "krea2")
        full = _flow_sigmas(10)
        report = self._run_both(callback, run, patch, full, 0, _evals(full, rng, family="krea2"), label="idle")
        self.assertGreater(report.idle, 3)
        self.assertGreater(report.edited, 0)


@requires_torch
class HostPatchTests(unittest.TestCase):
    """``apply_colorcraft`` on a ModelPatcher stand-in and the run bookkeeping (no extension)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host, compat

        cls.host, cls.compat = colorcraft_host, compat

    def _patch(self, family="zimage", tags=("I",)):
        settings = self.host.parse_settings(_settings(_luma(0.8), tags=list(tags)))
        prepared = self.host.prepare(_FakeModel(_fake_latent_format("Flux", 16, 2)), None, settings, True,
                                     device=torch.device("cpu"))
        return self.host.ColorcraftPatch(prepared, settings)

    def test_post_cfg_goes_last_and_replaces_an_earlier_colorcraft(self):
        def foreign(args):
            return args["denoised"]

        model = _FakePatcher({"sampler_post_cfg_function": [foreign]})
        first, second = self._patch(), self._patch()
        patched = self.host.apply_colorcraft(model, first)
        self.assertEqual(model.model_options["sampler_post_cfg_function"], [foreign])     # the input is untouched
        self.assertEqual(model.wrappers, {})
        self.assertEqual(patched.model_options["sampler_post_cfg_function"], [foreign, first.post_cfg])
        again = self.host.apply_colorcraft(patched, second)
        self.assertEqual(again.model_options["sampler_post_cfg_function"], [foreign, second.post_cfg])
        wrapper_type = self.compat.sampler_sample_wrapper_type()
        self.assertEqual(again.wrappers[wrapper_type][self.host.WRAPPER_KEY], [second.wrapper])
        self.assertIs(self.host.model_colorcraft_patch(again), second)
        self.assertIsNone(self.host.model_colorcraft_patch(model))
        with self.assertRaisesRegex(RuntimeError, "sampler wrappers"):
            self.host.apply_colorcraft(SimpleNamespace(clone=lambda: SimpleNamespace(model_options={})), first)

    def test_each_run_gets_its_own_state_and_a_summary_line(self):
        patch = self._patch()
        steps = []

        def run_once(sigmas):
            transformer = {"sample_sigmas": sigmas}
            extra_args = {"model_options": {"transformer_options": transformer}}

            def sampler(guider, sigmas_, extra_args_, *rest):
                state = patch.run_state({"model_options": extra_args_["model_options"]})
                steps.append(len(state.walked) - 1)
                x0 = torch.randn(1, 16, 8, 8)
                middle = sigmas_[(len(sigmas_) - 1) // 2:][:1]           # a step the schedule is not 0 at
                patch.post_cfg({"denoised": x0, "sigma": middle, "model_options": extra_args_["model_options"]})
                return x0

            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                patch.wrapper(_Executor(sampler), None, sigmas, extra_args, None, torch.zeros(1, 16, 8, 8))
            return logs.output

        first = run_once(_flow_sigmas(6))
        second = run_once(_flow_sigmas(9))
        self.assertEqual(steps, [6, 9])
        self.assertIn("[AI Studio Colorcraft] base: zimage (Flux); mods I; 1 evals, 1 edited", first[-1])
        self.assertIn("1 evals", second[-1])          # a fresh report per run

    def test_without_a_run_or_a_sigma_list_the_input_passes_through(self):
        patch = self._patch()
        x0 = torch.randn(1, 16, 4, 4)
        self.assertIs(patch.post_cfg({"denoised": x0, "sigma": torch.ones(1), "model_options": {}}), x0)
        self.assertIn("no sampling run: passed through", patch.orphan.notes)
        transformer = {}
        extra_args = {"model_options": {"transformer_options": transformer}}

        def sampler(guider, sigmas, extra_args_, *rest):
            return patch.post_cfg({"denoised": x0, "sigma": torch.ones(1), "model_options": extra_args_["model_options"]})

        self.assertIs(patch.wrapper(_Executor(sampler), None, _flow_sigmas(4), extra_args, None, None), x0)
        self.assertIn("no sigma list: passed through", patch.last_report.notes)

    def test_steps_at_zero_never_touch_the_basis(self):
        settings = self.host.parse_settings(_settings({"kind": "luma", "params": {"exposure": 1.0,
                                                                                  "tone_compression": 0.0},
                                                       "schedule": _schedule(start=0.5, end=0.6), "mask": None}))
        prepared = self.host.prepare(_FakeModel(_fake_latent_format("Flux", 16, 2)), None, settings, True,
                                     device=torch.device("cpu"))
        patch = self.host.ColorcraftPatch(prepared, settings)
        sigmas = _flow_sigmas(10)
        transformer = {"sample_sigmas": sigmas}
        extra_args = {"model_options": {"transformer_options": transformer}}
        x0 = torch.randn(1, 16, 8, 8)

        def sampler(guider, sigmas_, extra_args_, *rest):
            return patch.post_cfg({"denoised": x0, "sigma": sigmas_[:1], "model_options": extra_args_["model_options"]})

        with mock.patch.object(self.host, "basis_on", side_effect=AssertionError("basis loaded for a zero step")):
            self.assertIs(patch.wrapper(_Executor(sampler), None, sigmas, extra_args, None, x0), x0)
        self.assertEqual((patch.last_report.idle, patch.last_report.errors), (1, 0))

    def test_a_failing_setup_never_breaks_sampling(self):
        patch = self._patch()
        with mock.patch.object(self.host, "begin_run", side_effect=RuntimeError("boom")):
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                result = patch.wrapper(_Executor(lambda *a: "done"), None, _flow_sigmas(4), {}, None, None)
        self.assertEqual(result, "done")
        self.assertIn("not set up (RuntimeError: boom)", logs.output[0])


# --- PB-CC5: the node, the Hires carry ---------------------------------------------------------------------

class PackBehaviourTests(unittest.TestCase):
    """``ForgeNeoColorcraft`` without the extension: contract, no-op, refusals, patch placement, Hires carry."""

    @classmethod
    def setUpClass(cls):
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host, colorcraft_nodes, compat, hires_carry

        cls.host, cls.nodes, cls.compat, cls.carry = colorcraft_host, colorcraft_nodes, compat, hires_carry

    def test_node_contract(self):
        node = self.nodes.ForgeNeoColorcraft
        required = node.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["model", "vae", "settings_json", "contract", "pre_dd_sigma"])
        self.assertEqual(required["model"][0], "MODEL")
        self.assertEqual(required["vae"][0], "VAE")
        self.assertEqual(required["settings_json"][0], "STRING")
        self.assertEqual(required["contract"][0], ["v1"])
        self.assertEqual(required["pre_dd_sigma"][0], "BOOLEAN")
        self.assertIs(required["pre_dd_sigma"][1]["default"], True)
        self.assertEqual((node.RETURN_TYPES, node.FUNCTION), (("MODEL",), "patch"))
        self.assertEqual(self.nodes.NODE_CLASS_MAPPINGS, {"ForgeNeoColorcraft": node})
        self.assertEqual(self.nodes.NODE_DISPLAY_NAME_MAPPINGS, {"ForgeNeoColorcraft": "Colorcraft (Forge parity)"})
        self.assertIn(self.nodes.CARRY_KEY, self.carry.registered())

    def test_nothing_to_grade_returns_the_input_model(self):
        model = _FakePatcher(latent_format=type("Flux", (), {})())
        node = self.nodes.ForgeNeoColorcraft()
        with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):   # a pass with no tabs: silent
            self.assertIs(node.patch(model, None, json.dumps(_settings()))[0], model)
        # a vector edit on a model without a basis family (SDXL): nothing applies, no clone
        sdxl = _FakePatcher(latent_format=type("SDXL", (), {})())
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            self.assertIs(node.patch(sdxl, None, json.dumps(_settings(_luma())))[0], sdxl)
        self.assertIn("base pass: no basis (SDXL); vector controls off on this model", logs.output[-1])

    def test_bad_settings_and_an_unknown_contract_are_refused(self):
        model = _FakePatcher(latent_format=type("Flux", (), {})())
        node = self.nodes.ForgeNeoColorcraft()
        with self.assertRaisesRegex(ValueError, "Colorcraft 설정을 읽지 못했습니다"):
            node.patch(model, None, "{not json")
        with self.assertRaisesRegex(ValueError, "update the bundled node pack"):
            node.patch(model, None, json.dumps(_settings(_luma())), "v2")

    def test_the_patch_goes_last_on_a_clone(self):
        def earlier(args):
            return args["denoised"]

        model = _FakePatcher({"sampler_post_cfg_function": [earlier]}, latent_format=type("Flux", (), {})())
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            patched = self.nodes.ForgeNeoColorcraft().patch(model, None, json.dumps(_settings(_luma(), tags=["II"])),
                                                            "v1", False)[0]
        self.assertIsNot(patched, model)
        functions = patched.model_options["sampler_post_cfg_function"]
        self.assertIs(functions[0], earlier)
        self.assertTrue(self.host.is_owned(functions[1]))
        self.assertEqual(model.model_options["sampler_post_cfg_function"], [earlier])
        patch = self.host.model_colorcraft_patch(patched)
        self.assertFalse(patch.prepared.pre_dd)
        self.assertEqual(patch.settings["tags"], ["II"])
        wrapper_type = self.compat.sampler_sample_wrapper_type()
        self.assertEqual(patched.wrappers[wrapper_type][self.host.WRAPPER_KEY], [patch.wrapper])
        self.assertIn("base pass: zimage (Flux); II luma", logs.output[-1])

    @unittest.skipIf(_comfy_importable(), "ComfyUI is importable here: the pack loads it (and torch) at import")
    def test_pack_modules_import_without_torch(self):
        code = ("import sys\n"
                "from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host, colorcraft_nodes\n"
                "colorcraft_host.parse_settings('{\"v\": 1, \"pass\": \"base\", \"entries\": []}')\n"
                "print(sorted(m for m in ('torch', 'safetensors') if m in sys.modules))\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, encoding="utf-8",
                                check=True)
        self.assertEqual(result.stdout.strip(), "[]", result.stderr)


@requires_torch
class HiresCarryTests(unittest.TestCase):
    """A Hires checkpoint/UNET override keeps the hires pass's Colorcraft (``hires_carry``), prepared again for the
    override MODEL's latent format and with the override VAE (Forge re-attaches to the new UNet with its VAE)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host, colorcraft_nodes, hires_carry

        cls.host, cls.nodes, cls.carry = colorcraft_host, colorcraft_nodes, hires_carry

    def setUp(self):
        self.host.ANCHORS.clear()

    def _source(self, entries, vae):
        model = _FakePatcher(latent_format=_fake_latent_format("Wan21", 16, 3))
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            return self.nodes.ForgeNeoColorcraft().patch(model, vae, json.dumps(_settings(*entries,
                                                                                          pass_name="hires")))[0]

    def test_the_override_model_gets_the_pass_with_its_own_vae(self):
        base_vae, override_vae = _FakeVAE(16, 3), _FakeVAE(16, 2, seed=4)
        source = self._source([_basic(contrast=0.5, amount=0.3), _luma()], base_vae)
        encoded = base_vae.calls
        fmt = _fake_latent_format("Flux", 16, 2)
        target = _FakePatcher(latent_format=fmt)
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            carried = self.carry.apply(source, target, override_vae)
        self.assertIsNot(carried, target)
        self.assertEqual(base_vae.calls, encoded)
        self.assertEqual(override_vae.calls, 2)                          # the two anchors, with the override VAE
        patch, original = self.host.model_colorcraft_patch(carried), self.host.model_colorcraft_patch(source)
        self.assertEqual((patch.prepared.family, patch.prepared.pass_name), ("zimage", "hires"))
        self.assertEqual(patch.settings, original.settings)
        self.assertEqual(patch.prepared.pre_dd, original.prepared.pre_dd)
        expected = self.host.encode_anchors(_FakeVAE(16, 2, seed=4), fmt,
                                            [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, -0.2, 0.0)],
                                            device=torch.device("cpu"))
        self.assertEqual(list(patch.prepared.anchors), list(expected))
        for key in expected:
            self.assertTrue(torch.equal(patch.prepared.anchors[key], expected[key]))

    def test_nothing_carried_without_a_patch_or_when_nothing_applies(self):
        target = _FakePatcher(latent_format=_fake_latent_format("Flux", 16, 2))
        plain = _FakePatcher(latent_format=_fake_latent_format("Wan21", 16, 3))
        self.assertIs(self.carry.apply(plain, target, None), target)
        source = self._source([_luma()], _FakeVAE(16, 3))
        sdxl = _FakePatcher(latent_format=_fake_latent_format("SDXL", 4, 2))
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            self.assertIs(self.carry.apply(source, sdxl, None), sdxl)   # vector edits only: none on SDXL


if __name__ == "__main__":
    unittest.main()
