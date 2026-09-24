"""
ComfyUI Workflow Inspector — 워크플로우 JSON 호환성 자동 감지.

NAIA 2.0 ``core/comfyui_workflow_manager.py``의 3-state 분류 패턴 참고.

목적
----
사용자가 ComfyUI 워크플로우 JSON을 로드할 때, 해당 워크플로우가
**표준 체크포인트 로더**(UR_IV가 모델 콤보로 제어 가능)인지,
**커스텀/잠긴 로더**(GGUF/Unet 분리 등 — UI에서 모델 선택 비활성화 필요)인지
자동 판정.

추가로 sampler-centric 역추적으로 메인 모델 노드를 식별하고,
Sage Attention / TorchCompile / RescaleCFG 등 **패치 노드**는 통과시킨다.

3-state 분류
------------
- ``NATIVE_CHECKPOINT``  — 표준 ``CheckpointLoaderSimple`` 등 — UI 모델 콤보 활성
- ``NATIVE_UNET``        — ``UNETLoader`` (UNet/체크포인트 분리) — UI 모델 콤보 활성
- ``LOCKED_UNKNOWN``     — 커스텀 노드 (GGUF, NF4 등) — UI 모델 콤보 비활성

체크포인트 후보 ``class_type`` 목록은 ``CHECKPOINT_LOADERS`` 상수.
패치 노드는 ``PATCH_NODES`` (sampler 역추적 시 투과).

사용 예
-------
>>> inspector = WorkflowInspector(workflow_json)
>>> result = inspector.inspect()
>>> result.classification         # WorkflowClassification.NATIVE_CHECKPOINT
>>> result.model_node_id          # "4" (체크포인트 로더 노드 ID)
>>> result.model_param            # "ckpt_name"
>>> result.is_locked              # False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core import comfy_node_classes as _classes
from utils.app_logger import get_logger

_logger = get_logger("workflow_inspector")

# 노드 분류는 생성 컴파일러와 한 곳(core/comfy_node_classes.py)에서 공유한다 —
# 선택 화면이 '정상'이라 한 워크플로를 생성이 모르는 로더/샘플러로 거부하지 않도록.

# 표준 체크포인트 로더 (단일 노드에서 model+clip+vae 전부 반환)
NATIVE_CHECKPOINT_LOADERS: set[str] = set(_classes.CHECKPOINT_LOADER_NODES)

# UNet/모델만 반환 (clip/vae는 별도 노드) — KJNodes DiffusionModelLoaderKJ 포함
NATIVE_UNET_LOADERS: set[str] = set(_classes.UNET_LOADER_NODES)

# Sampler 노드 — 메인 모델 노드 역추적 시작점
SAMPLER_NODES: set[str] = set(_classes.SAMPLER_NODES)

TEXT_ENCODER_NODES = set(_classes.TEXT_ENCODER_NODES)
SAVE_NODES = set(_classes.IMAGE_SAVE_NODES)

# 모델 라인을 통과시키는 패치 노드 (model 입력이 있고 model 출력도 있음)
PATCH_NODES: set[str] = {
    # Attention 패치
    "PatchSageAttentionKJ",
    "SageAttention",
    "PatchModelAddDownscale",
    # 컴파일
    "TorchCompileModel",
    "TorchCompileModelKJ",
    "TorchCompileLoadWanVideo",
    # CFG / sampling 패치
    "RescaleCFG",
    "PerturbedAttention",
    "ModelSamplingSD3",
    "ModelSamplingFlux",
    "ModelSamplingDiscrete",
    "ModelSamplingContinuousEDM",
    # FreeU / 기타
    "FreeU",
    "FreeU_V2",
    "Inject Noise",
    # LoRA (model 통과)
    "LoraLoader",
    "LoraLoaderModelOnly",
    "Power Lora Loader (rgthree)",
    # Differential diffusion 등
    "DifferentialDiffusion",
    "ForgeNeoAnimaLoraLoader", "ForgeNeoAnimaLoraLoaderModelOnly",
    "ForgeNeoLoraBlockWeight", "ForgeNeoModelSamplingShift",
    "ForgeNeoNegPip", "ForgeNeoAnimaSafePAG", "ForgeNeoAnimaGuidanceSuite",
}


class WorkflowClassification(Enum):
    NATIVE_CHECKPOINT = "native_checkpoint"
    NATIVE_UNET = "native_unet"
    LOCKED_UNKNOWN = "locked_unknown"
    NO_SAMPLER = "no_sampler"  # 샘플러 노드 자체가 없음 (img2img/upscale 전용 등)


@dataclass
class InspectionResult:
    classification: WorkflowClassification = WorkflowClassification.LOCKED_UNKNOWN
    sampler_node_id: Optional[str] = None
    sampler_class: str = ""
    model_node_id: Optional[str] = None  # 모델 입력의 최종 origin
    model_class: str = ""
    model_param: str = ""  # 'ckpt_name' / 'unet_name' / 알 수 없으면 ""
    # 모델 라인에서 만난 패치 노드들 (디버깅/UI 표시용)
    patch_chain: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_locked(self) -> bool:
        """UI 모델 콤보를 비활성화해야 하는가."""
        return self.classification == WorkflowClassification.LOCKED_UNKNOWN


class WorkflowInspector:
    """ComfyUI 워크플로우 JSON 분석기.

    워크플로우 형식은 ComfyUI prompt JSON (``{node_id: {class_type, inputs, ...}}``).
    """

    # 모델 입력 키 후보 (노드별로 다를 수 있음)
    _MODEL_INPUT_KEYS = ("model", "MODEL", "unet", "diffusion_model")

    # 최대 역추적 깊이 (순환 방지)
    _MAX_DEPTH = 30

    def __init__(self, workflow: dict):
        if not isinstance(workflow, dict):
            raise TypeError(f"workflow must be dict, got {type(workflow).__name__}")
        self.workflow = workflow

    # ────────────────────────────────────────
    # Public
    # ────────────────────────────────────────

    def inspect(self) -> InspectionResult:
        result = InspectionResult()

        # 1) 메인 sampler 찾기
        sampler_id = self._find_main_sampler()
        if sampler_id is None:
            result.classification = WorkflowClassification.NO_SAMPLER
            result.notes.append("샘플러 노드를 찾지 못함")
            return result
        result.sampler_node_id = sampler_id
        result.sampler_class = self._class_of(sampler_id)

        # 2) sampler의 model 입력에서 역추적 (패치 노드 통과)
        terminal_id = self._trace_model_origin(sampler_id, result.patch_chain)
        if terminal_id is None:
            result.classification = WorkflowClassification.LOCKED_UNKNOWN
            result.notes.append("모델 입력을 추적하지 못함")
            return result
        result.model_node_id = terminal_id
        result.model_class = self._class_of(terminal_id)

        # 3) 분류
        if result.model_class in NATIVE_CHECKPOINT_LOADERS:
            result.classification = WorkflowClassification.NATIVE_CHECKPOINT
            result.model_param = "ckpt_name"
        elif result.model_class in NATIVE_UNET_LOADERS:
            result.classification = WorkflowClassification.NATIVE_UNET
            result.model_param = self._guess_unet_param(terminal_id)
        else:
            result.classification = WorkflowClassification.LOCKED_UNKNOWN
            result.notes.append(
                f"비표준 로더 '{result.model_class}' — UI 모델 선택 비활성 권장"
            )
        return result

    # ────────────────────────────────────────
    # 내부 헬퍼
    # ────────────────────────────────────────

    def _class_of(self, node_id: str) -> str:
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return ""
        return str(node.get("class_type", ""))

    def _inputs_of(self, node_id: str) -> dict:
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return {}
        inp = node.get("inputs")
        return inp if isinstance(inp, dict) else {}

    def _find_main_sampler(self) -> Optional[str]:
        """샘플러 노드들 중 메인 선택.

        규칙:
        1. 단일 sampler면 그것
        2. 여러 개면 ``latent``를 다른 sampler에서 받지 않는 것 (최종 출력측)
        3. 그래도 모르면 가장 ID가 큰 것 (편의상)
        """
        sampler_ids = [
            nid for nid in self.workflow
            if self._class_of(nid) in SAMPLER_NODES
        ]
        if not sampler_ids:
            return None
        if len(sampler_ids) == 1:
            return sampler_ids[0]

        # 다른 sampler의 latent_image로 들어가지 않는 sampler 후보
        sampler_set = set(sampler_ids)
        consumed_by_other_sampler: set[str] = set()
        for sid in sampler_ids:
            for key, val in self._inputs_of(sid).items():
                if key in ("latent_image", "samples") and self._is_link(val):
                    src_id = str(val[0])
                    if src_id in sampler_set:
                        consumed_by_other_sampler.add(src_id)
        terminals = [sid for sid in sampler_ids if sid not in consumed_by_other_sampler]
        if terminals:
            # 가장 ID가 큰 (보통 후속 단계) 선택
            return max(terminals, key=lambda s: self._id_sort_key(s))
        return max(sampler_ids, key=lambda s: self._id_sort_key(s))

    def _trace_model_origin(self,
                            start_id: str,
                            patch_chain: list[str],
                            depth: int = 0) -> Optional[str]:
        """``start_id``의 model 입력에서 시작해 패치 노드 통과 후 최종 origin 반환."""
        if depth > self._MAX_DEPTH:
            return None

        inputs = self._inputs_of(start_id)
        model_link = None
        for key in self._MODEL_INPUT_KEYS:
            if key in inputs and self._is_link(inputs[key]):
                model_link = inputs[key]
                break
        if model_link is None:
            return None

        src_id = str(model_link[0])
        src_class = self._class_of(src_id)

        if not src_class:
            return None

        # 패치 노드면 통과
        if src_class in PATCH_NODES:
            patch_chain.append(f"{src_class}({src_id})")
            return self._trace_model_origin(src_id, patch_chain, depth + 1)

        # 알려진 로더면 반환
        if src_class in NATIVE_CHECKPOINT_LOADERS or src_class in NATIVE_UNET_LOADERS:
            return src_id

        # 알려지지 않은 노드 — model 입력이 또 있으면 한 번 더 따라감
        for key in self._MODEL_INPUT_KEYS:
            if key in self._inputs_of(src_id):
                patch_chain.append(f"?{src_class}({src_id})")
                deeper = self._trace_model_origin(src_id, patch_chain, depth + 1)
                if deeper is not None:
                    return deeper
                break

        # 못 따라가면 src_id를 그대로 반환 (커스텀 로더로 분류됨)
        return src_id

    @staticmethod
    def _is_link(val) -> bool:
        """ComfyUI inputs의 link 형식: ``[node_id, slot_index]``."""
        return isinstance(val, list) and len(val) >= 2

    @staticmethod
    def _id_sort_key(nid: str):
        """ID가 숫자면 int, 아니면 문자열로 비교."""
        try:
            return (0, int(nid))
        except (ValueError, TypeError):
            return (1, nid)

    def _guess_unet_param(self, node_id: str) -> str:
        """UNETLoader류의 모델 파일명 입력 키 추측."""
        inputs = self._inputs_of(node_id)
        for key in ("unet_name", "model_name", "ckpt_name", "diffusion_model"):
            if key in inputs and not self._is_link(inputs[key]):
                return key
        return ""


# ─────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────

def inspect_workflow(workflow: dict) -> InspectionResult:
    """워크플로우 JSON 분석 — 한 줄 호출."""
    return WorkflowInspector(workflow).inspect()
