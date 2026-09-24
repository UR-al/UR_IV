"""Payload-independent check: can the app generate with this API workflow?

The workflow picker used to call a graph "valid" when it merely contained a
sampler, a text encoder and an output, while generation (``_compile_custom``)
intentionally rejects several such graphs: more than one sampler,
``SamplerCustom*``, encoder/latent/decoder branches it cannot map.  Instead of
re-implementing those rules, this runs the real compiler offline with an empty
payload — every error it raises there is structural and would be raised for
any generation request — and reports it as a warning for the picker.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def describe_generation_structure(workflow: Mapping[str, Any]) -> dict:
    """``{'generation_blocker': str | None, 'model_selectable': bool}``.

    ``model_selectable`` is True when the sampler's model chain ends in a
    loader whose model the app writes from its model selector; otherwise the
    workflow keeps its own model (custom/GGUF loaders) and the selector is
    ignored for it.
    """
    from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError

    blocker: Optional[str] = None
    try:
        ComfyWorkflowCompiler(None).compile("txt2img", "", {}, workflow=dict(workflow))
    except WorkflowCompileError as exc:
        blocker = str(exc)
    except Exception as exc:  # malformed node/input shapes
        blocker = f"워크플로 구조를 해석할 수 없습니다: {exc}"

    model_selectable = False
    try:
        sampler_id = ComfyWorkflowCompiler._find_sampler(workflow)
        # A node whose "inputs" is null/a list is malformed, not a crash:
        # the picker (analyze_workflow) must always return a summary.
        inputs = ComfyWorkflowCompiler._node_inputs(workflow.get(sampler_id))
        model_selectable = bool(
            ComfyWorkflowCompiler._trace_model_loader(workflow, inputs.get("model"))
        )
    except WorkflowCompileError:
        model_selectable = False
    except Exception:  # malformed graph: the blocker above already explains it
        model_selectable = False
    return {"generation_blocker": blocker, "model_selectable": model_selectable}


__all__ = ["describe_generation_structure"]
