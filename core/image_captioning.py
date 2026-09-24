"""Local image tagging and natural-language caption inference.

This module deliberately contains no UI or output-file code.  It exposes a small
inference boundary used by the Batch/Caption view:

* :class:`CAFormerTagger` runs the local ``caformer_s18.dbv4-full`` ONNX model.
* :class:`ImageCaptioningEngine` dispatches CAFormer and Ollama vision models.
* :class:`CaptionResult` keeps tag/caption metadata while retaining a simple
  string-returning ``caption()`` API.

Heavy runtimes are imported and models are opened lazily so importing the app does
not allocate VRAM or make ``onnxruntime`` a hard dependency for Ollama-only use.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps

from core.ollama_client import DEFAULT_OLLAMA_URL, OllamaClient


CAFORMER_REPO_ID = "animetimm/caformer_s18.dbv4-full"
CAFORMER_CACHE_NAME = "models--animetimm--caformer_s18.dbv4-full"
CAFORMER_REQUIRED_FILES = ("model.onnx", "selected_tags.csv", "preprocess.json")

TORIIGATE_BF16_MODEL = "hf.co/DraconicDragon/ToriiGate-0.5-GGUF:BF16"
TORIIGATE_SHORT_FORMAT = (
    "The caption for image should be quite short without long purple prose and slop. "
    "Cover main objects and details."
)
TORIIGATE_PLAIN_TEXT_RULE = (
    "Write regular natural text only. Do not use JSON, Markdown headings, lists, "
    "or key-value fields."
)
TORIIGATE_FACTUAL_SYSTEM_PROMPT = (
    "You are an image captioning expert. Describe the user's picture according to the "
    "requested format and instructions. Report only directly visible facts. Do not infer "
    "or embellish mood, atmosphere, aesthetics, symbolism, intention, relationships, "
    "identity, or unseen context. Do not guess uncertain details; omit them. Never add a "
    "concluding evaluation about the overall style or describe the image as clean, "
    "minimalist, modern, or having a particular feel. Output only the requested "
    "natural-language caption."
)

CATEGORY_BY_ID = {0: "general", 4: "character", 9: "rating"}
DEFAULT_CATEGORY_THRESHOLDS = {
    "general": 0.35,
    "character": 0.43,
    "rating": 0.38,
}
CAFORMER_PAD_INTERPOLATION = Image.Resampling.BILINEAR
CAFORMER_RESIZE_INTERPOLATION = Image.Resampling.BICUBIC


class ImageCaptioningError(RuntimeError):
    """Base error for local image-captioning failures."""


class ModelDiscoveryError(FileNotFoundError, ImageCaptioningError):
    """Raised when a complete local model directory cannot be found."""


class ModelValidationError(ValueError, ImageCaptioningError):
    """Raised when model artifacts do not satisfy the local runtime contract."""


class RuntimeDependencyError(ImportError, ImageCaptioningError):
    """Raised when an optional inference runtime is not installed."""


@dataclass(frozen=True)
class TagPrediction:
    """One selected CAFormer tag and its post-sigmoid confidence."""

    name: str
    score: float
    category: str


@dataclass(frozen=True)
class CAFormerOptions:
    """Filtering options for CAFormer output.

    ``best`` uses each row's ``best_threshold`` from ``selected_tags.csv``.
    ``category`` uses the three editable category thresholds instead.
    ``from_mapping`` accepts both Python snake_case and Vue camelCase keys.
    """

    include_rating: bool = False
    include_characters: bool = True
    threshold_mode: str = "best"
    general_threshold: float = DEFAULT_CATEGORY_THRESHOLDS["general"]
    character_threshold: float = DEFAULT_CATEGORY_THRESHOLDS["character"]
    rating_threshold: float = DEFAULT_CATEGORY_THRESHOLDS["rating"]

    def __post_init__(self) -> None:
        mode = self.threshold_mode.strip().lower()
        if mode not in {"best", "category"}:
            raise ValueError("threshold_mode must be 'best' or 'category'")
        object.__setattr__(self, "threshold_mode", mode)
        for name in ("general_threshold", "character_threshold", "rating_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "CAFormerOptions":
        if values is None:
            return cls()
        if isinstance(values, cls):
            return values

        category_values = values.get("categoryThresholds", values.get("category_thresholds", {}))
        if not isinstance(category_values, Mapping):
            category_values = {}

        def pick(default: Any, *keys: str) -> Any:
            for key in keys:
                if key in values:
                    return values[key]
            return default

        return cls(
            include_rating=bool(pick(False, "includeRating", "include_rating")),
            include_characters=bool(pick(True, "includeCharacters", "include_characters")),
            threshold_mode=str(pick("best", "thresholdMode", "threshold_mode")),
            general_threshold=float(
                pick(
                    category_values.get("general", DEFAULT_CATEGORY_THRESHOLDS["general"]),
                    "generalThreshold",
                    "general_threshold",
                )
            ),
            character_threshold=float(
                pick(
                    category_values.get("character", DEFAULT_CATEGORY_THRESHOLDS["character"]),
                    "characterThreshold",
                    "character_threshold",
                )
            ),
            rating_threshold=float(
                pick(
                    category_values.get("rating", DEFAULT_CATEGORY_THRESHOLDS["rating"]),
                    "ratingThreshold",
                    "rating_threshold",
                )
            ),
        )

    def threshold_for(self, category: str) -> float:
        return {
            "general": self.general_threshold,
            "character": self.character_threshold,
            "rating": self.rating_threshold,
        }[category]


@dataclass(frozen=True)
class CaptionResult:
    """Structured result returned by :meth:`ImageCaptioningEngine.caption_result`."""

    mode: str
    tags: tuple[TagPrediction, ...] = field(default_factory=tuple)
    natural_caption: str = ""
    separator: str = "\n\n"

    @property
    def tag_text(self) -> str:
        return ", ".join(item.name for item in self.tags)

    @property
    def tags_by_category(self) -> dict[str, tuple[TagPrediction, ...]]:
        return {
            category: tuple(item for item in self.tags if item.category == category)
            for category in ("general", "character", "rating")
        }

    @property
    def text(self) -> str:
        if self.mode == "caformer":
            return self.tag_text
        if self.mode == "combined":
            parts = [part for part in (self.tag_text, self.natural_caption) if part]
            return self.separator.join(parts)
        return self.natural_caption


@dataclass(frozen=True)
class _TagRecord:
    name: str
    category: str
    best_threshold: float


class _OnnxInput(Protocol):
    name: str


class _OnnxOutput(Protocol):
    name: str
    shape: Sequence[Any]


class _OnnxSession(Protocol):
    def get_inputs(self) -> Sequence[_OnnxInput]: ...

    def get_outputs(self) -> Sequence[_OnnxOutput]: ...

    def run(self, output_names: Any, input_feed: Mapping[str, np.ndarray]) -> Sequence[Any]: ...


class _VisionClient(Protocol):
    model: str

    def caption_image(
        self,
        image_path: str,
        prompt: str = "",
        timeout: int = 180,
        system_prompt: str | None = None,
    ) -> str: ...

    def list_models(self) -> list[str]: ...


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        key = os.path.normcase(os.path.abspath(str(expanded)))
        if key not in seen:
            seen.add(key)
            result.append(expanded)
    return result


def default_hf_cache_roots() -> list[Path]:
    """Return likely Hugging Face hub cache roots without scanning broad disks."""

    roots: list[Path] = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]))
    if os.environ.get("HF_HOME"):
        roots.extend((Path(os.environ["HF_HOME"]), Path(os.environ["HF_HOME"]) / "hub"))
    roots.extend(
        (
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface",
        )
    )
    return _unique_paths(roots)


def missing_caformer_files(model_dir: str | os.PathLike[str]) -> tuple[str, ...]:
    folder = Path(model_dir)
    return tuple(name for name in CAFORMER_REQUIRED_FILES if not (folder / name).is_file())


def validate_caformer_model_dir(model_dir: str | os.PathLike[str]) -> Path:
    """Validate and return a resolved CAFormer model directory."""

    folder = Path(model_dir).expanduser()
    missing = missing_caformer_files(folder)
    if missing:
        joined = ", ".join(missing)
        raise ModelValidationError(
            f"CAFormer model folder is incomplete: {folder} (missing: {joined})"
        )
    return folder.resolve()


def _snapshot_candidates(root: Path) -> Iterator[Path]:
    """Yield only the narrow repository paths used by the HF hub cache."""

    yield root
    yield root / CAFORMER_CACHE_NAME
    repo_roots = (root / CAFORMER_CACHE_NAME, root / "hub" / CAFORMER_CACHE_NAME)
    for repo_root in repo_roots:
        yield repo_root
        snapshots = repo_root / "snapshots"
        if snapshots.is_dir():
            children = [child for child in snapshots.iterdir() if child.is_dir()]
            children.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            yield from children


def discover_caformer_model(
    model_dir: str | os.PathLike[str] | None = None,
    *,
    cache_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> Path:
    """Find a complete CAFormer folder explicitly or in the local HF cache."""

    if model_dir is not None and str(model_dir).strip():
        return validate_caformer_model_dir(model_dir)

    roots = (
        [Path(item) for item in cache_roots]
        if cache_roots is not None
        else default_hf_cache_roots()
    )
    searched: list[Path] = []
    for root in _unique_paths(roots):
        for candidate in _snapshot_candidates(root):
            searched.append(candidate)
            if not missing_caformer_files(candidate):
                return candidate.resolve()

    searched_text = "; ".join(str(path) for path in searched) or "(no cache roots)"
    required = ", ".join(CAFORMER_REQUIRED_FILES)
    raise ModelDiscoveryError(
        "CAFormer model was not found in the local Hugging Face cache. "
        f"A folder containing {required} is required. Searched: {searched_text}"
    )


def _rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def preprocess_caformer_image(image_path: str | os.PathLike[str]) -> np.ndarray:
    """Apply the model card's 512 white-pad and 384 ImageNet preprocessing."""

    try:
        with Image.open(image_path) as source:
            image = _rgb_on_white(ImageOps.exif_transpose(source))
    except Exception as exc:
        raise ImageCaptioningError(f"Unable to open image for CAFormer: {image_path}: {exc}") from exc

    # Fit without cropping, then pad to a fixed white square.  The subsequent
    # resize and center crop mirror preprocess.json (the crop is a no-op for a
    # square but is kept explicit to document the model contract).
    image.thumbnail((512, 512), CAFORMER_PAD_INTERPOLATION)
    padded = Image.new("RGB", (512, 512), "white")
    left = (512 - image.width) // 2
    top = (512 - image.height) // 2
    padded.paste(image, (left, top))
    resized = padded.resize((384, 384), CAFORMER_RESIZE_INTERPOLATION)
    cropped = ImageOps.fit(
        resized,
        (384, 384),
        method=CAFORMER_RESIZE_INTERPOLATION,
        centering=(0.5, 0.5),
    )

    array = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def _category_name(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"general", "character", "rating"}:
        return value
    try:
        category_id = int(float(value))
    except ValueError as exc:
        raise ModelValidationError(f"Unknown CAFormer tag category: {raw!r}") from exc
    try:
        return CATEGORY_BY_ID[category_id]
    except KeyError as exc:
        raise ModelValidationError(f"Unknown CAFormer tag category id: {category_id}") from exc


def _load_tag_records(csv_path: Path) -> tuple[_TagRecord, ...]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {field.strip() for field in (reader.fieldnames or []) if field}
            required = {"name", "category", "best_threshold"}
            if not required.issubset(fields):
                missing = ", ".join(sorted(required - fields))
                raise ModelValidationError(
                    f"selected_tags.csv is missing required columns: {missing}"
                )
            records: list[_TagRecord] = []
            for row_number, row in enumerate(reader, start=2):
                name = (row.get("name") or "").strip()
                if not name:
                    raise ModelValidationError(
                        f"selected_tags.csv row {row_number} has an empty tag name"
                    )
                category = _category_name(row.get("category") or "")
                raw_threshold = (row.get("best_threshold") or "").strip()
                try:
                    threshold = float(raw_threshold)
                except ValueError as exc:
                    raise ModelValidationError(
                        f"selected_tags.csv row {row_number} has an invalid best_threshold"
                    ) from exc
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ModelValidationError(
                        f"selected_tags.csv row {row_number} best_threshold must be between 0 and 1"
                    )
                records.append(_TagRecord(name, category, threshold))
    except ModelValidationError:
        raise
    except OSError as exc:
        raise ModelValidationError(f"Unable to read selected_tags.csv: {exc}") from exc
    if not records:
        raise ModelValidationError("selected_tags.csv does not contain any tags")
    return tuple(records)


def _is_probability_output(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in {"prediction", "predictions", "probability", "probabilities"}


def _scores_from_model_output(
    output: Any,
    *,
    expected_count: int,
    output_name: str,
    is_probability: bool,
) -> np.ndarray:
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size != expected_count:
        raise ModelValidationError(
            "CAFormer output/tag metadata mismatch: "
            f"output {output_name!r} has {values.size} values but "
            f"selected_tags.csv has {expected_count} rows"
        )
    if is_probability:
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ModelValidationError(
                f"CAFormer probability output {output_name!r} contains values outside 0..1"
            )
        return values
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _read_caformer_scores(
    session: _OnnxSession,
    input_feed: Mapping[str, np.ndarray],
    expected_count: int,
) -> np.ndarray:
    """Select CAFormer's class output without assuming ONNX output order.

    Current DBV4 exports expose ``embedding``, ``logits`` and ``prediction`` in
    that order.  ``prediction`` is already post-sigmoid, whereas ``logits`` is
    not.  Prefer a named logits tensor; older/custom exports fall back to an
    output whose flattened size equals the tag metadata row count.
    """

    try:
        output_infos = list(session.get_outputs())
    except (AttributeError, TypeError):
        output_infos = []

    for info in output_infos:
        name = str(getattr(info, "name", ""))
        if name.strip().lower() == "logits":
            outputs = session.run([name], input_feed)
            if not outputs:
                raise ImageCaptioningError("CAFormer ONNX logits output is empty")
            return _scores_from_model_output(
                outputs[0],
                expected_count=expected_count,
                output_name=name,
                is_probability=False,
            )

    outputs = list(session.run(None, input_feed))
    candidates: list[tuple[str, Any]] = []
    if output_infos and len(output_infos) == len(outputs):
        candidates = [
            (str(getattr(info, "name", f"output_{index}")), output)
            for index, (info, output) in enumerate(zip(output_infos, outputs, strict=True))
        ]
    else:
        candidates = [(f"output_{index}", output) for index, output in enumerate(outputs)]

    matching = [
        (name, output)
        for name, output in candidates
        if np.asarray(output).size == expected_count
    ]
    if not matching:
        sizes = ", ".join(
            f"{name}={np.asarray(output).size}" for name, output in candidates
        ) or "no outputs"
        raise ModelValidationError(
            "CAFormer output/tag metadata mismatch: no class output has "
            f"{expected_count} values ({sizes})"
        )

    # If both an unknown class tensor and an explicitly post-sigmoid tensor are
    # present, the named probability is the least ambiguous fallback.
    matching.sort(key=lambda item: 0 if _is_probability_output(item[0]) else 1)
    name, output = matching[0]
    return _scores_from_model_output(
        output,
        expected_count=expected_count,
        output_name=name,
        is_probability=_is_probability_output(name),
    )


class CAFormerTagger:
    """Lazy local ONNX runner for ``animetimm/caformer_s18.dbv4-full``."""

    _SESSION_CACHE: dict[tuple[str, tuple[str, ...]], _OnnxSession] = {}
    _SESSION_LOCK = threading.Lock()

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        cache_roots: Iterable[str | os.PathLike[str]] | None = None,
        providers: Sequence[str] | None = None,
        session_factory: Callable[[Path], _OnnxSession] | None = None,
    ) -> None:
        self._configured_model_dir = model_dir
        self._cache_roots = tuple(cache_roots) if cache_roots is not None else None
        self._providers = tuple(providers or ())
        self._session_factory = session_factory
        self._model_dir: Path | None = None
        self._records: tuple[_TagRecord, ...] | None = None
        self._session: _OnnxSession | None = None

    @property
    def model_dir(self) -> Path:
        if self._model_dir is None:
            self._model_dir = discover_caformer_model(
                self._configured_model_dir,
                cache_roots=self._cache_roots,
            )
        return self._model_dir

    @property
    def records(self) -> tuple[_TagRecord, ...]:
        if self._records is None:
            self._records = _load_tag_records(self.model_dir / "selected_tags.csv")
        return self._records

    def _create_runtime_session(self, model_path: Path) -> _OnnxSession:
        try:
            ort = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError(
                "CAFormer tagging requires onnxruntime. Install 'onnxruntime' for CPU "
                "or 'onnxruntime-gpu' for CUDA inference."
            ) from exc

        providers = list(self._providers)
        if not providers:
            available = set(ort.get_available_providers())
            providers = [
                name
                for name in (
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                )
                if name in available
            ]
        kwargs = {"providers": providers} if providers else {}
        return ort.InferenceSession(str(model_path), **kwargs)

    def _get_session(self) -> _OnnxSession:
        if self._session is not None:
            return self._session
        model_path = (self.model_dir / "model.onnx").resolve()
        if self._session_factory is not None:
            self._session = self._session_factory(model_path)
            return self._session

        key = (os.path.normcase(str(model_path)), self._providers)
        with self._SESSION_LOCK:
            session = self._SESSION_CACHE.get(key)
            if session is None:
                session = self._create_runtime_session(model_path)
                self._SESSION_CACHE[key] = session
        self._session = session
        return session

    def tag_image(
        self,
        image_path: str | os.PathLike[str],
        options: CAFormerOptions | Mapping[str, Any] | None = None,
    ) -> list[TagPrediction]:
        opts = options if isinstance(options, CAFormerOptions) else CAFormerOptions.from_mapping(options)
        tensor = preprocess_caformer_image(image_path)
        records = self.records
        session = self._get_session()
        inputs = session.get_inputs()
        if not inputs:
            raise ImageCaptioningError("CAFormer ONNX model does not expose an input tensor")
        scores = _read_caformer_scores(
            session,
            {inputs[0].name: tensor},
            expected_count=len(records),
        )

        selected: list[TagPrediction] = []
        for record, score_value in zip(records, scores, strict=True):
            if record.category == "character" and not opts.include_characters:
                continue
            if record.category == "rating" and not opts.include_rating:
                continue
            threshold = (
                record.best_threshold
                if opts.threshold_mode == "best"
                else opts.threshold_for(record.category)
            )
            score = float(score_value)
            if score >= threshold:
                selected.append(TagPrediction(record.name, score, record.category))
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected


def select_toriigate_model(models: Iterable[str]) -> str:
    """Choose the exact BF16 Ollama artifact first, then another ToriiGate model."""

    available = [str(model).strip() for model in models if str(model).strip()]
    exact = TORIIGATE_BF16_MODEL.casefold()
    for model in available:
        if model.casefold() == exact:
            return model
    torii_models = [model for model in available if "toriigate" in model.casefold()]
    for model in torii_models:
        if "bf16" in model.casefold():
            return model
    return torii_models[0] if torii_models else TORIIGATE_BF16_MODEL


def build_torii_prompt(
    *,
    tags: Sequence[TagPrediction] = (),
    user_instruction: str = "",
) -> str:
    """Build the model author's short-format request with safe output constraints."""

    sections = [
        "# Captioning format:\n"
        f"{TORIIGATE_SHORT_FORMAT}\n"
        f"{TORIIGATE_PLAIN_TEXT_RULE}"
    ]
    instruction = (user_instruction or "").strip()
    if instruction:
        sections.append(f"# Additional instructions:\n{instruction}")
    if tags:
        tag_text = " ".join(item.name for item in tags)
        sections.append(f"# Booru tags for the image\n[{tag_text}]")
    sections.append("# Characters on picture:\nAvoid guessing names for characters.")
    return "\n\n".join(sections)


def _json_string_leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        cleaned = " ".join(value.split()).strip()
        if cleaned:
            yield cleaned
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _json_string_leaves(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _json_string_leaves(child)


def _join_caption_fragments(fragments: Iterable[str]) -> str:
    sentences: list[str] = []
    for fragment in fragments:
        text = fragment.strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        sentences.append(text)
    return " ".join(sentences)


_SUBJECTIVE_TORII_SENTENCE = re.compile(
    r"(?:"
    r"\b(?:mood|atmosphere|aesthetics?|symbolism|sense\s+of|feeling\s+of)\b|"
    r"\b(?:overall\s+)?style\s+(?:is|appears|looks|feels)\b|"
    r"\b(?:abstract|artistic|clean|minimalist|modern|sophisticated)"
    r"(?:\s+\w+){0,2}\s+feel\b|"
    r"\b(?:evok(?:e|es|ed|ing)|convey(?:s|ed|ing)?|suggest(?:s|ed|ing)?)\b|"
    r"\bcreat(?:e|es|ed|ing)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:calm|serene|peaceful|cozy|mysterious|dramatic|sophisticated)\b"
    r")",
    flags=re.IGNORECASE,
)


def _strip_subjective_torii_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for sentence in sentences:
        # Preserve the visible fact before a common interpretive tail.
        sentence = re.sub(
            r",\s*(?:which\s+)?(?:enhanc(?:e|es|ed|ing)|emphasiz(?:e|es|ed|ing))"
            r"\b[^.!?]*([.!?])$",
            r"\1",
            sentence,
            flags=re.IGNORECASE,
        )
        if not _SUBJECTIVE_TORII_SENTENCE.search(sentence):
            kept.append(sentence)
    return " ".join(kept).strip() or text.strip()


def normalize_torii_caption(response: str) -> str:
    """Recover plain prose when ToriiGate ignores the format and emits JSON/fences."""

    text = (response or "").strip()
    if not text:
        return ""
    fenced = re.fullmatch(
        r"```(?:json|javascript|js)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()

    if text.startswith(("{", "[", '"')):
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        text = _join_caption_fragments(_json_string_leaves(decoded))
    return _strip_subjective_torii_sentences(text)


@contextmanager
def _limited_image_path(
    image_path: str | os.PathLike[str],
    max_pixels: int,
) -> Iterator[Path]:
    """Yield the source or a short-lived, aspect-preserving <= max_pixels JPEG.

    The ``yield`` must stay outside both the ``Image.open`` block and the
    ``except OSError`` guard: the caller runs the Ollama request inside this
    context, and its ConnectionError/TimeoutError are OSError subclasses that
    would otherwise be reported as "unable to open image"; an open handle would
    also lock a JPEG/WebP source (WinError 32 on move/delete) for the whole request.
    """

    source_path = Path(image_path)
    if max_pixels <= 0:
        raise ValueError("max_caption_pixels must be positive")
    image = None
    try:
        with Image.open(source_path) as source:
            width, height = source.size
            orientation = source.getexif().get(274, 1)
            needs_normalization = orientation not in (None, 1) or source.mode != "RGB"
            if width * height > max_pixels or needs_normalization:
                image = _rgb_on_white(ImageOps.exif_transpose(source))
    except OSError as exc:
        raise ImageCaptioningError(f"Unable to open image for captioning: {source_path}: {exc}") from exc

    if image is None:
        # Already small, upright RGB: send the original bytes (file handle closed).
        yield source_path
        return

    if image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / float(image.width * image.height))
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    with tempfile.TemporaryDirectory(prefix="image_viewer_caption_") as temp_dir:
        limited_path = Path(temp_dir) / "input.jpg"
        image.save(limited_path, format="JPEG", quality=95)
        yield limited_path


class ImageCaptioningEngine:
    """Dispatch CAFormer, generic Ollama, ToriiGate, or the combined pipeline."""

    def __init__(
        self,
        *,
        caformer_tagger: CAFormerTagger | None = None,
        caformer_model_dir: str | os.PathLike[str] | None = None,
        caformer_cache_roots: Iterable[str | os.PathLike[str]] | None = None,
        caformer_providers: Sequence[str] | None = None,
        caformer_session_factory: Callable[[Path], _OnnxSession] | None = None,
        ollama_client: _VisionClient | None = None,
        ollama_client_factory: Callable[..., _VisionClient] = OllamaClient,
        ollama_base_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str | None = None,
        torii_model: str | None = None,
        max_caption_pixels: int = 1_000_000,
    ) -> None:
        self._tagger = caformer_tagger
        self._caformer_model_dir = caformer_model_dir
        self._caformer_cache_roots = (
            tuple(caformer_cache_roots) if caformer_cache_roots is not None else None
        )
        self._caformer_providers = caformer_providers
        self._caformer_session_factory = caformer_session_factory
        self._injected_ollama_client = ollama_client
        self._ollama_client_factory = ollama_client_factory
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.torii_model = torii_model
        self.max_caption_pixels = max_caption_pixels

    @property
    def caformer_tagger(self) -> CAFormerTagger:
        if self._tagger is None:
            self._tagger = CAFormerTagger(
                self._caformer_model_dir,
                cache_roots=self._caformer_cache_roots,
                providers=self._caformer_providers,
                session_factory=self._caformer_session_factory,
            )
        return self._tagger

    @contextmanager
    def _ollama_client(
        self,
        *,
        model: str | None,
        base_url: str,
    ) -> Iterator[_VisionClient]:
        if self._injected_ollama_client is None:
            kwargs: dict[str, Any] = {"base_url": base_url}
            if model:
                kwargs["model"] = model
            yield self._ollama_client_factory(**kwargs)
            return

        client = self._injected_ollama_client
        old_model = getattr(client, "model", None)
        old_url = getattr(client, "base_url", None)
        if model:
            client.model = model
        if hasattr(client, "base_url"):
            client.base_url = base_url.rstrip("/")
        try:
            yield client
        finally:
            if old_model is not None:
                client.model = old_model
            if old_url is not None:
                client.base_url = old_url

    def _select_torii_model(self, base_url: str, configured: str | None) -> str:
        if configured:
            return configured
        if self.torii_model:
            return self.torii_model
        try:
            with self._ollama_client(model=TORIIGATE_BF16_MODEL, base_url=base_url) as client:
                return select_toriigate_model(client.list_models())
        except Exception:
            # Connection/capability errors should be reported by caption_image; model
            # discovery itself safely falls back to the requested exact artifact.
            return TORIIGATE_BF16_MODEL

    def _natural_caption(
        self,
        image_path: str | os.PathLike[str],
        *,
        model: str | None,
        base_url: str,
        prompt: str,
        timeout: int,
        normalize_torii: bool = False,
        system_prompt: str | None = None,
    ) -> str:
        with _limited_image_path(image_path, self.max_caption_pixels) as limited_path:
            with self._ollama_client(model=model, base_url=base_url) as client:
                call_kwargs: dict[str, Any] = {"prompt": prompt, "timeout": timeout}
                if system_prompt is not None:
                    call_kwargs["system_prompt"] = system_prompt
                caption = client.caption_image(str(limited_path), **call_kwargs).strip()
                if normalize_torii:
                    caption = normalize_torii_caption(caption).strip()
                if not caption:
                    raise ImageCaptioningError(
                        f"Ollama model {getattr(client, 'model', model)!r} returned an empty caption"
                    )
                return caption

    def caption_result(
        self,
        image_path: str | os.PathLike[str],
        mode: str = "ollama",
        *,
        caformer_options: CAFormerOptions | Mapping[str, Any] | None = None,
        prompt: str = "",
        separator: str = "\n\n",
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        timeout: int = 180,
    ) -> CaptionResult:
        selected_mode = (mode or "ollama").strip().lower()
        if selected_mode not in {"ollama", "caformer", "torii", "combined"}:
            raise ValueError(
                "caption mode must be one of: ollama, caformer, torii, combined"
            )
        base_url = (ollama_base_url or self.ollama_base_url).rstrip("/")

        if selected_mode == "caformer":
            tags = tuple(self.caformer_tagger.tag_image(image_path, caformer_options))
            return CaptionResult(selected_mode, tags=tags, separator=separator)

        if selected_mode == "ollama":
            model = ollama_model or self.ollama_model
            caption = self._natural_caption(
                image_path,
                model=model,
                base_url=base_url,
                prompt=(prompt or "Describe this image in detail.").strip(),
                timeout=timeout,
            )
            return CaptionResult(selected_mode, natural_caption=caption, separator=separator)

        if selected_mode == "torii":
            model = self._select_torii_model(base_url, ollama_model)
            caption = self._natural_caption(
                image_path,
                model=model,
                base_url=base_url,
                prompt=build_torii_prompt(user_instruction=prompt),
                timeout=timeout,
                normalize_torii=True,
                system_prompt=TORIIGATE_FACTUAL_SYSTEM_PROMPT,
            )
            return CaptionResult(selected_mode, natural_caption=caption, separator=separator)

        tags = tuple(self.caformer_tagger.tag_image(image_path, caformer_options))
        model = self._select_torii_model(base_url, ollama_model)
        caption = self._natural_caption(
            image_path,
            model=model,
            base_url=base_url,
            prompt=build_torii_prompt(tags=tags, user_instruction=prompt),
            timeout=timeout,
            normalize_torii=True,
            system_prompt=TORIIGATE_FACTUAL_SYSTEM_PROMPT,
        )
        return CaptionResult(
            selected_mode,
            tags=tags,
            natural_caption=caption,
            separator=separator,
        )

    def caption(
        self,
        image_path: str | os.PathLike[str],
        mode: str = "ollama",
        **kwargs: Any,
    ) -> str:
        """Return only the formatted text for callers that do not need metadata."""

        return self.caption_result(image_path, mode, **kwargs).text


__all__ = [
    "CAFORMER_REPO_ID",
    "CAFORMER_REQUIRED_FILES",
    "CAFORMER_PAD_INTERPOLATION",
    "CAFORMER_RESIZE_INTERPOLATION",
    "TORIIGATE_BF16_MODEL",
    "TORIIGATE_PLAIN_TEXT_RULE",
    "TORIIGATE_SHORT_FORMAT",
    "TORIIGATE_FACTUAL_SYSTEM_PROMPT",
    "CAFormerOptions",
    "CAFormerTagger",
    "CaptionResult",
    "ImageCaptioningEngine",
    "ImageCaptioningError",
    "ModelDiscoveryError",
    "ModelValidationError",
    "RuntimeDependencyError",
    "TagPrediction",
    "default_hf_cache_roots",
    "build_torii_prompt",
    "discover_caformer_model",
    "missing_caformer_files",
    "normalize_torii_caption",
    "preprocess_caformer_image",
    "select_toriigate_model",
    "validate_caformer_model_dir",
]
