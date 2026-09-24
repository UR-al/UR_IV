# workers/ollama_worker.py
"""Ollama 비동기 Worker — UI 블로킹 방지"""
from PyQt6.QtCore import QThread, pyqtSignal


class OllamaWorker(QThread):
    finished = pyqtSignal(str)  # JSON {tags, mode}
    error = pyqtSignal(str)

    def __init__(self, base_url: str, model: str, tags: str, mode: str, extra_prompt: str = '', parent=None, *,
                 instruction_feature=None, instructions=None, resolve_installed: bool = False):
        super().__init__(parent)
        from core.ai_assist_instructions import load_instructions, normalize_instructions
        self._base_url = base_url
        self._model = model
        self._tags = tags
        self._mode = mode
        self._extra = extra_prompt
        # 설치 모델 대조(/api/tags)를 run() 안에서 — GUI 스레드에서 동기 HTTP 를 하지 않게.
        # 옵트인이라 모델을 직접 넘기는 테스트·호출은 GET 을 보내지 않는다.
        self._resolve_installed = bool(resolve_installed)
        # Snapshot at request creation, not after the worker starts. Settings
        # edits or mutations of a caller's dict cannot change an in-flight job.
        self._instructions = (load_instructions() if instructions is None
                              else normalize_instructions(instructions))
        self._instruction_feature = instruction_feature

    def run(self):
        try:
            import json
            from core.ollama_client import DEFAULT_OLLAMA_MODEL, OllamaClient, resolve_installed_model
            if self._resolve_installed:
                model = resolve_installed_model(self._base_url, self._model)
            else:
                model = (self._model or '').strip() or DEFAULT_OLLAMA_MODEL
            client = OllamaClient(self._base_url, model)
            result = client.enhance(self._tags, self._mode, self._extra,
                                    instructions=self._instructions,
                                    instruction_feature=self._instruction_feature)
            self.finished.emit(json.dumps({'tags': result, 'mode': self._mode}))
        except Exception as e:
            self.error.emit(str(e))


def detach_result_signals(worker) -> None:
    """교체되는 이전 워커의 결과 연결만 끊는다.

    진행 중인 HTTP 는 취소할 수 없고, run() 만 오버라이드한 QThread 에 quit() 은 효과가 없으며
    wait() 은 GUI 스레드만 막는다. 인자 없는 ``QObject.disconnect()`` 는 스레드 종료 뒤
    정리(:func:`release_when_done`) 연결까지 끊으므로 쓰지 않는다. 연결이 없으면 TypeError 가 난다.
    """
    if worker is None:
        return
    for signal in (worker.finished, worker.error):
        try:
            signal.disconnect()
        except (TypeError, RuntimeError):
            pass


def release_when_done(worker, owner, attr: str) -> None:
    """스레드가 실제로 끝나면(QThread 기본 finished) ``owner.attr`` 참조를 비우고 deleteLater.

    ``OllamaWorker.finished`` 는 run() 안에서 쏘는 결과 시그널이라 여기에 deleteLater 를 걸면
    스레드가 아직 도는 중에 지워질 수 있다 — 그래서 가려진 기본 시그널을 명시적으로 꺼내 쓴다.
    참조를 먼저 비워야 다음 요청의 isRunning() 이 지워진 객체를 건드리지 않는다.
    """
    def _cleanup():
        if getattr(owner, attr, None) is worker:
            setattr(owner, attr, None)
        worker.deleteLater()

    QThread.finished.__get__(worker, QThread).connect(_cleanup)
