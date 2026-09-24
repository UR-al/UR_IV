# backends/base.py
"""백엔드 추상 인터페이스 및 공통 데이터 클래스"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable


@dataclass
class BackendInfo:
    """백엔드에서 가져온 서버 정보"""
    models: List[str] = field(default_factory=list)
    samplers: List[str] = field(default_factory=list)
    schedulers: List[str] = field(default_factory=list)
    upscalers: List[str] = field(default_factory=list)
    vae: List[str] = field(default_factory=lambda: ["Use same VAE"])
    checkpoints: List[str] = field(default_factory=list)
    options: Dict = field(default_factory=dict)


@dataclass
class MediaArtifact:
    """백엔드가 생성한 하나의 미디어 결과.

    ``data``와 ``path`` 중 적어도 하나를 채우는 것이 생산자 측 계약이다.
    ``kind``는 현재 ``image``, ``animated``, ``video``, ``audio``를 사용한다.
    원격 백엔드의 저장소 정보는 로컬 경로로 오인되지 않도록 ``metadata``에
    보존한다.
    """
    kind: str
    data: Optional[bytes] = None
    path: Optional[str] = None
    filename: Optional[str] = None
    mime: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """통합 생성 결과.

    ``image_data``는 기존 이미지 호출부를 위한 첫 정적/애니메이션 이미지
    호환 필드다. 새 호출부는 모든 결과가 보존되는 ``artifacts``를 사용한다.
    """
    success: bool
    image_data: Optional[bytes] = None
    info: Dict = field(default_factory=dict)
    error: Optional[str] = None
    artifacts: List[MediaArtifact] = field(default_factory=list)

    def __post_init__(self):
        if self.image_data is not None:
            return
        primary = next(
            (
                artifact for artifact in self.artifacts
                if artifact.kind in ("image", "animated")
                and artifact.data is not None
            ),
            None,
        )
        if primary is not None:
            self.image_data = primary.data


# Preview strings are raw base64 (Forge/Comfy); bytes remain valid for adapters.
ProgressCallback = Callable[[int, int, Optional[bytes | str]], None]


class AbstractBackend(ABC):
    """백엔드 추상 클래스"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    @abstractmethod
    def test_connection(self) -> bool:
        """연결 상태 확인"""
        ...

    @abstractmethod
    def get_info(self) -> BackendInfo:
        """서버 정보 (모델, 샘플러 등) 가져오기"""
        ...

    @abstractmethod
    def txt2img(self, model_name: str, payload: Dict,
                progress_callback: Optional[ProgressCallback] = None,
                cancel_check: Optional[Callable[[], bool]] = None) -> GenerationResult:
        """텍스트→이미지 생성.

        ``cancel_check`` 가 주어지면 True 를 돌려주는 즉시 협조적으로 멈춘다 — 모델 전환 뒤·
        발송 직전에 다시 확인하고, 이미 보낸 요청은 **이 어댑터의 작업만** 중단한다(다른
        클라이언트의 작업을 끊지 않는다). 취소되면 ``success=False`` 결과를 돌려준다.
        호출부는 ``core.cancellable_call.call_with_optional_cancel`` 로 넘긴다
        (duck-typed 옛 어댑터·fake 는 이 인자를 모를 수 있다).
        """
        ...

    @abstractmethod
    def img2img(self, model_name: str, payload: Dict,
                progress_callback: Optional[ProgressCallback] = None,
                cancel_check: Optional[Callable[[], bool]] = None) -> GenerationResult:
        """이미지→이미지 생성. ``cancel_check`` 계약은 :meth:`txt2img` 와 같다."""
        ...

    @abstractmethod
    def upscale(self, image_b64: str, settings: Dict) -> str:
        """이미지 업스케일. base64 결과 반환"""
        ...

    @abstractmethod
    def adetailer(self, image_b64: str, settings: Dict) -> str:
        """ADetailer 처리. base64 결과 반환"""
        ...

    @abstractmethod
    def sam3(self, image_b64: str, settings: Dict) -> str:
        """SAM3 처리. base64 결과 반환"""
        ...

    def refine(self, image_b64: str, settings: Dict) -> str:
        """SAM3 Refine (Target/Replacement 재손질). base64 결과 반환.

        기본 구현은 미지원 — Forge SAM3 확장이 있는 백엔드만 구현한다.
        """
        raise NotImplementedError(
            "이 백엔드는 SAM3 Refine을 지원하지 않습니다.\n"
            "Forge Neo WebUI 백엔드에서 사용하세요."
        )

    def interrupt(self):
        """진행 중 생성을 서버 측에서 중단 (best-effort).
        취소 플래그만으로는 이미 보낸 HTTP 요청이 끝까지 돌므로,
        실제 취소는 이 훅으로 백엔드에 전달한다. 기본 구현은 no-op."""
        return

    def get_loras(self) -> List[Dict]:
        """LoRA 목록 반환.

        각 항목: ``{'name': str, 'alias': str, 'path': str, 'trigger_words': list[str]}``
        (trigger_words 는 메타데이터가 없으면 빈 리스트).
        """
        return []

    def get_system_stats(self) -> Dict:
        """GPU/VRAM 상태 조회. 기본 구현은 빈 dict 반환"""
        return {}

    @abstractmethod
    def get_backend_type(self) -> str:
        """백엔드 이름 반환 ('webui' 또는 'comfyui')"""
        ...
