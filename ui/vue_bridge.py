# ui/vue_bridge.py
"""
PyQt6 ↔ Vue 통신 브릿지 (QWebChannel)
위젯 프록시 값 동기화 + 액션 디스패치 + 이미지 생성 이벤트
"""
import json
import logging
import os
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from core.path_safety import safe_input_path as _normalize_vue_path  # noqa: F401

logger = logging.getLogger(__name__)

# 편집 임시본 보존 개수. EditorView의 MAX_UNDO(30)보다 넉넉해야 undo 히스토리가
# 참조하는 파일이 지워지지 않는다.
_EDITOR_TEMP_KEEP = 60
# 썸네일 캐시 상한 — 넘으면 오래된 것부터 정리 (언제든 재생성 가능한 캐시)
_THUMB_CACHE_MAX_BYTES = 300 * 1024 * 1024

_RUNTIME_ENGINE_TO_CORE = {
    'forge': 'forge',
    'forge_neo': 'forge',
    'comfyui': 'comfyui',
}
_RUNTIME_CORE_TO_ENGINE = {
    'forge': 'forge',
    'comfyui': 'comfyui',
}

# Gallery는 Creator Studio가 생성하는 정지 이미지, 애니메이션, 영상, 오디오를
# 같은 목록에서 다룬다. 확장자 판정은 한곳에 두어 동기/비동기 API가 어긋나지 않게 한다.
_GALLERY_MEDIA_EXTS = frozenset({
    # still / animated image
    '.png', '.jpg', '.jpeg', '.webp', '.gif', '.apng', '.bmp', '.tif', '.tiff', '.avif',
    # video
    '.mp4', '.webm', '.mov', '.mkv', '.m4v', '.avi', '.ogv',
    # audio
    '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.opus',
})


def _scan_gallery_media(target: str, recursive_roots=()) -> list[str]:
    """지원 미디어를 수정 시각 내림차순으로 반환하는 순수 스캔 경계."""
    entries = []

    def collect(folder: str) -> None:
        try:
            with os.scandir(folder) as scan:
                for entry in scan:
                    if not entry.is_file() or os.path.splitext(entry.name)[1].lower() not in _GALLERY_MEDIA_EXTS:
                        continue
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        mtime = 0
                    entries.append((mtime, entry.path.replace('\\', '/')))
        except OSError:
            return

    collect(target)
    for recursive_root in recursive_roots:
        if not os.path.isdir(recursive_root):
            continue
        for folder, _dirs, _files in os.walk(recursive_root):
            collect(folder)
    entries.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in entries]


class VueBridge(QObject):
    """Vue 프론트엔드와 통신하는 중앙 브릿지"""

    # ── Python → Vue 시그널 ──
    imageGenerated = pyqtSignal(str)
    generationStarted = pyqtSignal()
    generationError = pyqtSignal(str)

    editorImageLoaded = pyqtSignal(str)   # file path
    editorWatermarkImageLoaded = pyqtSignal(str)   # 워터마크로 쓸 이미지 파일 경로
    captionFilesSelected = pyqtSignal(str)  # JSON [path] — 캡션 대상 이미지
    captionProgress = pyqtSignal(str)       # JSON {clientToken,jobId,index,total,path,caption,error}
    captionDone = pyqtSignal(str)           # JSON {clientToken,jobId,total,ok,failed,skipped}
    captionOutDirSelected = pyqtSignal(str)  # 캡션 저장 폴더 경로
    captionModelDirSelected = pyqtSignal(str)  # CAFormer model.onnx 폴더
    captionRuntimeReady = pyqtSignal(str)    # JSON {caformer,torii,onnxruntime}
    i2iImageLoaded = pyqtSignal(str)     # file path
    galleryFolderLoaded = pyqtSignal(str)  # folder path
    inpaintImageLoaded = pyqtSignal(str)   # file path (PngInfo + InpaintView 공용)
    searchStatus = pyqtSignal(str)         # status message

    loraInserted = pyqtSignal(str)       # JSON {name, weight}
    loraStackLoaded = pyqtSignal(str)    # JSON [{name, weight, enabled, triggerWords}]
    yoloModelUpdated = pyqtSignal(str)   # model label text
    condRulesLoaded = pyqtSignal(str)    # JSON {positive, negative}
    batchFilesSelected = pyqtSignal(str) # JSON [paths]
    ollamaResult = pyqtSignal(str)       # JSON {tags, mode} or {error}
    genNlResult = pyqtSignal(str)        # JSON {tags, mode} or {error} — 생성 시 태그→자연어 전용 채널
    globalWeightsLoaded = pyqtSignal(str) # JSON [{tag, weight}]
    uiPrefsLoaded = pyqtSignal(str)      # JSON {tagBlockMode, ...}
    compareImageLoaded = pyqtSignal(str) # JSON {slot, path}
    galleryImagesReady = pyqtSignal(str) # JSON {folder, files}
    upscalersReady = pyqtSignal(str)      # JSON [name]
    ollamaModelsReady = pyqtSignal(str)   # JSON {url, models}
    chatToken = pyqtSignal(str)          # JSON {id, text} — 대화 탭 스트리밍 조각(모아 보냄)
    chatDone = pyqtSignal(str)           # JSON {id, ok, content, stopped, error?}
    chatThreads = pyqtSignal(str)        # JSON [threads] — 저장된 대화 목록
    chatModelInfo = pyqtSignal(str)      # capability metadata, never loads a model
    chatModelsReady = pyqtSignal(str)
    aiAssistInstructionsChanged = pyqtSignal(str)
    instructionPresetsChanged = pyqtSignal(str)
    xyzCapabilitiesReceived = pyqtSignal(str)
    xyzPlotEvent = pyqtSignal(str)
    chatGenerationEvent = pyqtSignal(str)  # JSON request-owned generation progress/media
    adetailerModelsReady = pyqtSignal(str) # JSON [name]
    queueItemAdded = pyqtSignal(str)     # JSON {prompt, ...}
    queueCompleted = pyqtSignal(str)     # JSON {total}
    showNotification = pyqtSignal(str, str)  # (type: success|error|info, message)
    adetailerResult = pyqtSignal(str)       # JSON {before, after, output_path} or {error}
    adetailerProgress = pyqtSignal(int, int) # (current, total)
    sam3Result = pyqtSignal(str)            # JSON {before, after, output_path} or {error}
    sam3Progress = pyqtSignal(int, int)     # (current, total)
    # Refine (sam-extra 워크플로 2) — JSON {before, after, prompt, negative_prompt} or {error}
    refineResult = pyqtSignal(str)
    # sam-extra 임베드 LoRA Manager 주소 — JSON {url, status, message}
    loraManagerUrlReady = pyqtSignal(str)
    eventSearchProgress = pyqtSignal(int, int) # (current, total)
    # JSON {running, count, waiting, wait_remaining_ms, wait_total_ms,
    #       deck_remaining, deck_total, deck_used, allow_duplicates,
    #       paused,   ← 일시정지 여부 (Vue 가 일시정지/재개 버튼 모양을 정한다)
    #       prompt}   ← 다음 생성에 나갈 프롬프트 전문. 없으면 ''.
    # prompt 로 무엇을 보내는지는 generator_actions._emit_auto_status 주석 참조 —
    # total_prompt_display(=API 로 그대로 나가는 문자열)이며, 생성 직전에 붙는
    # 파이프라인 훅·LoRA 꼬리·자연어 변환만 아직 반영되지 않는다.
    automationStatus = pyqtSignal(str)
    automationSettingsLoaded = pyqtSignal(str)  # JSON {mode, limit, repeat, delay, allowDupes, maxRetries} — PR 9 mode-aware
    instantWildcardsList = pyqtSignal(str)      # JSON [{name, lines: [...]}] — PR 8
    promptOrderLoaded = pyqtSignal(str)         # JSON [{key, label}] — 사용자 지정 섹션 순서
    workflowProfilesList = pyqtSignal(str)      # JSON [{name, created_at, model, vae}]
    comfyWorkflowEvent = pyqtSignal(str)
    comfyCompatibilityResult = pyqtSignal(str)
    relightEvent = pyqtSignal(str)
    handReconstructionEvent = pyqtSignal(str)
    eventImportResults = pyqtSignal(str)      # JSON event list

    # Creator Studio — 모든 payload는 JSON 문자열 하나로 유지해 Qt/WebSocket
    # transport가 동일한 interface를 사용한다.
    creatorStateChanged = pyqtSignal(str)
    creatorProgress = pyqtSignal(str)
    creatorResult = pyqtSignal(str)
    creatorMediaSelected = pyqtSignal(str)
    creatorCacheEvent = pyqtSignal(str)
    comicStoryboardReady = pyqtSignal(str)
    comicDocumentChanged = pyqtSignal(str)

    # 앱 관리형 Forge Neo / ComfyUI runtime. 설치처럼 긴 작업은 슬롯에서
    # operation id만 즉시 돌려주고 이 단일 JSON signal로 진행/완료를 보낸다.
    backendRuntimeEvent = pyqtSignal(str)
    modelDownloadEvent = pyqtSignal(str)

    # ── 시작 백엔드 게이트 (앱 창 위의 Vue 오버레이) ──
    # 왜 별도 채널인가: 예전엔 이 선택이 Vue보다 먼저 뜨는 별도 QDialog였다.
    # 그래서 시작 화면만 앱의 나머지와 다른 규칙(QSS)으로 살았고, 스플래시가
    # 뜨기도 전에 모달이 떠 "앱이 아직 안 켜졌는데 뭘 고르라는 거지"였다.
    # 이제 창을 먼저 띄우고 그 위에 오버레이를 얹는다 — 아래 4개가 그 배선이다.
    # (QDialog는 Vue가 못 뜬 경우의 비상 경로로 살아 있다.)
    backendSelectionRequired = pyqtSignal(str)  # JSON {webuiUrl, comfyUrl, workflowPath}
    backendProbeResult = pyqtSignal(str)        # JSON {webui: 'ok'|'fail', comfy: 'ok'|'fail'}
    backendSelected = pyqtSignal(str)           # JSON {ok: bool, error?: str}
    comfyWorkflowPicked = pyqtSignal(str)       # JSON {path, info}

    # ── 하단 계기 스트립이 읽는 백엔드 연결 상태 ──
    # 왜 게이트 4종과 따로인가: 게이트는 '고르기 전' 한 번의 사건이고 이건 앱이
    # 사는 내내 바뀌는 상태다 — URL 을 바꾸거나 연결이 끊겨도 같은 채널로 온다.
    # 받는 쪽이 마지막 값만 들고 있으면 되도록 항상 전체 상태를 보낸다(멱등).
    # JSON {
    #   kind: 'webui' | 'comfyui',
    #   label: 'Forge' | 'A1111' | 'ComfyUI' | 'WebUI',  # 사람이 읽는 이름
    #   url: 'http://127.0.0.1:7860',
    #   connected: bool,
    #   error?: str,                                     # connected=false 일 때만
    # }
    backendStatus = pyqtSignal(str)

    # 외부에 제공하는 생성 API 서버와 원격 WebUI/ComfyUI target 관리.
    # 민감한 token은 Web 모드에서 항상 제거된 snapshot만 전달한다.
    generationApiEvent = pyqtSignal(str)

    # 위젯 값/속성 동기화 (Python → Vue)
    widgetValueChanged = pyqtSignal(str, str)       # (widget_id, value)
    widgetPropertyChanged = pyqtSignal(str, str, str)  # (widget_id, prop, value_json)

    # 배치 업데이트 (설정 로드 시 한번에 전송)
    batchUpdate = pyqtSignal(str)  # JSON: {widget_id: value, ...}

    # 탭 전환
    tabChanged = pyqtSignal(str)  # tab_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proxies = {}  # widget_id → proxy 객체
        self._batch_mode = False
        self._batch_buffer = {}
        self._action_handler = None  # 액션 디스패처 (메인 윈도우에서 설정)
        self._async_lookup_inflight = set()
        self._async_lookup_lock = threading.Lock()
        self._backend_runtime_inflight = {}
        self._backend_runtime_lock = threading.Lock()
        self._generation_api_inflight = None
        self._generation_api_lock = threading.Lock()
        self._caption_job_lock = threading.Lock()
        # Caption signals are broadcast by QWebChannel.  Keep a small keyed
        # journal so the initiating Vue client can ignore another client's job
        # and recover state after a transient WebChannel reconnect.
        self._caption_state_lock = threading.Lock()
        self._caption_job_states = {}
        self.adetailerModelsReady.connect(self._apply_adetailer_models_json)

    def _run_async_lookup(self, key, loader, signal):
        """GUI 스레드를 막는 조회를 중복 없이 백그라운드에서 실행한다."""
        with self._async_lookup_lock:
            if key in self._async_lookup_inflight:
                return
            self._async_lookup_inflight.add(key)

        def _work():
            try:
                signal.emit(loader())
            except Exception as e:
                logger.warning("async lookup failed (%s): %s", key, e)
            finally:
                with self._async_lookup_lock:
                    self._async_lookup_inflight.discard(key)

        threading.Thread(target=_work, daemon=True, name=f"vue-{key}").start()

    def _register_proxy(self, widget_id: str, proxy):
        """위젯 프록시 등록 + 부모 설정 (GC 방지)"""
        self._proxies[widget_id] = proxy
        if hasattr(proxy, 'setParent') and proxy.parent() is None:
            proxy.setParent(self)

    def set_action_handler(self, handler):
        """액션 핸들러 설정 (메인 윈도우의 메서드를 디스패치)"""
        self._action_handler = handler

    # ── Python → Vue 데이터 전송 ──

    def pushWidgetValue(self, widget_id: str, value: str):
        """위젯 값을 Vue로 전송"""
        if self._batch_mode:
            self._batch_buffer[widget_id] = value
        else:
            self.widgetValueChanged.emit(widget_id, str(value))

    def pushWidgetProperty(self, widget_id: str, prop: str, value):
        """위젯 속성을 Vue로 전송"""
        self.widgetPropertyChanged.emit(widget_id, prop, json.dumps(value))

    def beginBatchUpdate(self):
        """배치 모드 시작 (load_settings 등에서 사용)"""
        self._batch_mode = True
        self._batch_buffer = {}

    def endBatchUpdate(self):
        """배치 모드 종료 — 버퍼의 모든 값을 한번에 전송"""
        self._batch_mode = False
        if self._batch_buffer:
            self.batchUpdate.emit(json.dumps(self._batch_buffer))
            self._batch_buffer = {}

    # ── 이미지 생성 이벤트 ──

    def send_image(self, path: str, width: int, height: int, seed: int):
        data = json.dumps({
            'path': path.replace('\\', '/'),
            'width': width, 'height': height, 'seed': seed,
        })
        self.imageGenerated.emit(data)

    def send_start(self):
        self.generationStarted.emit()

    # ── Vue → Python 슬롯 ──

    @pyqtSlot(str, str)
    @pyqtSlot(str, float)
    @pyqtSlot(str, bool)
    def onWidgetChanged(self, widget_id: str, value):
        """Vue에서 사용자가 위젯 값을 변경했을 때 (타입 무관 수용)"""
        proxy = self._proxies.get(widget_id)
        if proxy:
            previous = self.getWidgetValue(widget_id)
            # 문자열로 변환하여 전달
            proxy._on_vue_changed(str(value))
            preset_changed = getattr(self.parent(), '_comfy_quality_user_change', None)
            if callable(preset_changed):
                preset_changed(widget_id, previous, self.getWidgetValue(widget_id))

    @pyqtSlot(str, str)
    def onAction(self, action: str, payload_json: str):
        """Vue에서 버튼 클릭 등 액션 요청"""
        if self._backend_runtime_is_web_mode() and str(action or '').strip().lower() in {'show_api_manager'}:
            self.showNotification.emit('warning', '웹 모드에서는 데스크톱 백엔드 관리 창을 열 수 없습니다.')
            return
        if self._action_handler:
            try:
                # payload가 이미 dict인 경우와 JSON 문자열인 경우 모두 대응
                if isinstance(payload_json, str):
                    payload = json.loads(payload_json) if payload_json else {}
                else:
                    payload = payload_json
                self._action_handler(action, payload)
            except Exception as e:
                print(f"[VueBridge] Action error: {action} - {e}")

    @pyqtSlot(str, str)
    def requestAction(self, action: str, payload_json: str):
        """onAction의 별칭 - Vue에서 더 직관적으로 호출 가능하도록"""
        self.onAction(action, payload_json)

    @pyqtSlot(str, result=str)
    def getWidgetValue(self, widget_id: str) -> str:
        """Vue에서 위젯 값 동기 요청"""
        proxy = self._proxies.get(widget_id)
        if not proxy:
            return ""
        if hasattr(proxy, 'text'):
            return proxy.text()
        if hasattr(proxy, 'toPlainText'):
            return proxy.toPlainText()
        if hasattr(proxy, 'isChecked'):
            return "true" if proxy.isChecked() else "false"
        if hasattr(proxy, 'currentText'):
            return proxy.currentText()
        return ""

    @pyqtSlot(result=str)
    def getAllWidgetValues(self) -> str:
        """모든 위젯 값을 JSON으로 반환 (초기 로드용)"""
        result = {}
        for wid, proxy in self._proxies.items():
            if hasattr(proxy, 'text'):
                result[wid] = proxy.text()
            elif hasattr(proxy, 'toPlainText'):
                result[wid] = proxy.toPlainText()
            elif hasattr(proxy, 'isChecked'):
                result[wid] = "true" if proxy.isChecked() else "false"
            elif hasattr(proxy, 'currentText'):
                result[wid] = proxy.currentText()
        return json.dumps(result)

    # ── App-managed backend runtime ──

    def _backend_runtime_is_web_mode(self) -> bool:
        """로컬 파일/프로세스 변경 권한이 없는 Web host인지 확인."""
        return bool(getattr(self.parent(), 'web_mode', False))

    @staticmethod
    def _backend_runtime_engine(engine: str) -> tuple[str, str]:
        ui_engine = str(engine or '').strip().lower()
        core_engine = _RUNTIME_ENGINE_TO_CORE.get(ui_engine)
        if not core_engine:
            raise ValueError(f'지원하지 않는 백엔드 runtime입니다: {engine}')
        return _RUNTIME_CORE_TO_ENGINE[core_engine], core_engine

    def _backend_runtime_public_snapshot(self, raw=None) -> dict:
        """core snapshot을 Settings가 소비하는 안정된 JSON 계약으로 어댑트."""
        if raw is None:
            from core.backend_runtime import get_backend_runtime_manager
            raw = get_backend_runtime_manager().snapshot()
        raw = raw if isinstance(raw, dict) else {}
        runtimes = raw.get('engines') if isinstance(raw.get('engines'), dict) else {}
        active_core = _RUNTIME_ENGINE_TO_CORE.get(
            str(raw.get('activeEngine') or '').strip().lower(),
            '',
        )
        active_engine = _RUNTIME_CORE_TO_ENGINE.get(active_core, '')
        primary_core = _RUNTIME_ENGINE_TO_CORE.get(
            str(raw.get('primaryModelEngine') or '').strip().lower(),
            '',
        )
        primary_engine = _RUNTIME_CORE_TO_ENGINE.get(primary_core, '')
        with self._backend_runtime_lock:
            inflight = dict(self._backend_runtime_inflight)

        engines = {}
        for ui_engine, core_engine, name in (
            ('forge', 'forge', 'Forge Neo'),
            ('comfyui', 'comfyui', 'ComfyUI'),
        ):
            runtime = runtimes.get(core_engine)
            runtime = runtime if isinstance(runtime, dict) else {}
            running = bool(runtime.get('running', False))
            api_url = str(runtime.get('apiUrl') or '')
            update_available = bool(runtime.get('updateAvailable', False))
            extension_dir = str(runtime.get('extensionDir') or '')
            version = str(runtime.get('version') or '')
            remote_commit = str(runtime.get('remoteCommit') or '')
            engines[ui_engine] = {
                'engine': ui_engine,
                'kind': core_engine,
                'name': str(runtime.get('name') or name),
                'installed': bool(runtime.get('installed', False)),
                'running': running,
                'healthy': bool(runtime.get('healthy', running and bool(api_url))),
                'owned': bool(runtime.get('owned', False)),
                'busy': ui_engine in inflight or bool(runtime.get('busy', False)),
                'autoStart': bool(runtime.get('autoStart', False)),
                'active': bool(runtime.get('active', active_core == core_engine)),
                'ownership': 'managed',
                'sourceMode': str(runtime.get('sourceMode') or 'managed'),
                'existingRoot': str(runtime.get('existingRoot') or ''),
                'root': str(runtime.get('root') or ''),
                'installRoot': str(
                    runtime.get('installRoot') or runtime.get('sourceRoot') or runtime.get('root') or ''
                ),
                'sourceRoot': str(runtime.get('sourceRoot') or ''),
                'pythonPath': str(runtime.get('pythonPath') or ''),
                'dataRoot': str(runtime.get('dataRoot') or ''),
                'modelPaths': dict(runtime.get('modelPaths') or {}),
                'apiUrl': api_url,
                'port': runtime.get('port'),
                'version': version,
                'installedCommit': str(runtime.get('commit') or ''),
                'latestCommit': remote_commit,
                'updateAvailable': update_available,
                'updateStatus': str(runtime.get('updateStatus') or ('Update available' if update_available else 'Up to date')),
                'extensionDir': extension_dir,
                'defaultExtensionDir': str(runtime.get('defaultExtensionDir') or ''),
                'extensionDirExternal': bool(runtime.get('extensionDirExternal', False)),
                'extensionWritable': bool(
                    runtime.get('extensionWritable', runtime.get('installed', False))
                ),
                'extensions': list(runtime.get('extensions') or []),
                'status': 'running' if running else 'installed' if runtime.get('installed') else 'not_installed',
                'message': str(runtime.get('message') or ''),
                'logPath': str(runtime.get('logPath') or ''),
            }

        return {
            'ok': True,
            'nativeOperations': not self._backend_runtime_is_web_mode(),
            'active': {
                'engine': active_engine,
                'kind': active_core,
                'ownership': 'managed' if active_core else '',
                'autoStart': bool(
                    engines.get(active_engine, {}).get('autoStart', False)
                    if active_engine else False
                ),
            },
            'runtimeRoot': str(raw.get('runtimeRoot') or ''),
            'primaryModelEngine': primary_engine,
            'engines': engines,
        }

    @pyqtSlot(result=str)
    def getBackendRuntimeState(self) -> str:
        """앱 관리형 backend 상태 조회. Web facade에도 읽기 전용으로 허용."""
        try:
            return json.dumps(
                self._backend_runtime_public_snapshot(),
                ensure_ascii=False,
                default=str,
            )
        except Exception as exc:
            return json.dumps(
                {
                    'ok': False,
                    'nativeOperations': not self._backend_runtime_is_web_mode(),
                    'error': str(exc),
                    'engines': {},
                },
                ensure_ascii=False,
            )

    # ── Generation API gateway ──

    def _generation_api_public_snapshot(self) -> dict:
        from core.generation_api import get_generation_api_manager

        include_secret = not self._backend_runtime_is_web_mode()
        raw = get_generation_api_manager().snapshot(include_secret=include_secret)
        snapshot = dict(raw) if isinstance(raw, dict) else {}
        if not include_secret:
            # 코어 실현이 제거하더라도 bridge 경계에서 중첩 객체까지 한 번 더 차단한다.
            sensitive = {'token', 'apitoken', 'authorization', 'secret'}

            def redact(value):
                if isinstance(value, dict):
                    return {
                        key: redact(item)
                        for key, item in value.items()
                        if str(key).replace('_', '').lower() not in sensitive
                    }
                if isinstance(value, list):
                    return [redact(item) for item in value]
                return value

            snapshot = redact(snapshot)
            config = snapshot.get('config')
            if isinstance(config, dict):
                safe_targets = []
                for item in config.get('targets', []):
                    if not isinstance(item, dict):
                        continue
                    target = dict(item)
                    target['urlConfigured'] = bool(target.pop('url', ''))
                    target['workflowConfigured'] = bool(target.pop('workflowPath', ''))
                    target['img2imgWorkflowConfigured'] = bool(target.pop('img2imgWorkflowPath', ''))
                    safe_targets.append(target)
                config['targets'] = safe_targets
        snapshot['nativeOperations'] = include_secret
        return snapshot

    @pyqtSlot(result=str)
    def getGenerationApiState(self) -> str:
        """생성 API 서버/target 상태 조회.

        Web facade에서도 상태는 읽을 수 있지만 token과 변경 권한은 주지 않는다.
        """
        try:
            return json.dumps(
                {'ok': True, **self._generation_api_public_snapshot()},
                ensure_ascii=False,
                default=str,
            )
        except Exception as exc:
            return json.dumps({
                'ok': False,
                'nativeOperations': not self._backend_runtime_is_web_mode(),
                'error': str(exc),
            }, ensure_ascii=False)

    @pyqtSlot(str, str, result=str)
    def runGenerationApiOperation(self, action: str, payload_json: str) -> str:
        """생성 API 설정/실행 작업을 worker에 넘기고 즉시 operation id를 반환."""
        if self._backend_runtime_is_web_mode():
            return json.dumps({
                'ok': False,
                'accepted': False,
                'error': '웹 모드에서는 API 서버·token·원격 target 설정을 변경할 수 없습니다.',
            }, ensure_ascii=False)

        try:
            action = str(action or '').strip().lower()
            allowed = {'save_config', 'start', 'stop', 'rotate_token', 'test_target'}
            if action not in allowed:
                raise ValueError(f'지원하지 않는 생성 API 작업입니다: {action}')
            payload = json.loads(payload_json) if payload_json else {}
            if not isinstance(payload, dict):
                raise ValueError('payload는 JSON 객체여야 합니다')
        except Exception as exc:
            return json.dumps(
                {'ok': False, 'accepted': False, 'error': str(exc)},
                ensure_ascii=False,
            )

        import uuid
        operation_id = uuid.uuid4().hex
        with self._generation_api_lock:
            if self._generation_api_inflight:
                return json.dumps({
                    'ok': False,
                    'accepted': False,
                    'operationId': self._generation_api_inflight,
                    'error': '생성 API 설정 작업이 이미 진행 중입니다.',
                }, ensure_ascii=False)
            self._generation_api_inflight = operation_id

        threading.Thread(
            target=self._run_generation_api_operation,
            args=(action, payload, operation_id),
            daemon=True,
            name=f'generation-api-{action}',
        ).start()
        return json.dumps({
            'ok': True,
            'accepted': True,
            'action': action,
            'operationId': operation_id,
        }, ensure_ascii=False)

    def _emit_generation_api_event(self, payload: dict) -> None:
        try:
            self.generationApiEvent.emit(json.dumps(payload, ensure_ascii=False, default=str))
        except RuntimeError:
            pass

    def _run_generation_api_operation(
        self,
        action: str,
        payload: dict,
        operation_id: str,
    ) -> None:
        self._emit_generation_api_event({
            'type': 'started',
            'action': action,
            'operationId': operation_id,
            'message': f'생성 API {action} 작업을 시작했습니다.',
        })
        try:
            from core.generation_api import get_generation_api_manager

            manager = get_generation_api_manager()
            raw = manager.execute(action, dict(payload))
            result = dict(raw) if isinstance(raw, dict) else {'value': raw}
            ok = bool(result.get('ok', True))
            error = result.get('error') if not ok else None
        except Exception as exc:
            result = {'ok': False, 'error': str(exc)}
            ok = False
            error = str(exc)
        finally:
            with self._generation_api_lock:
                if self._generation_api_inflight == operation_id:
                    self._generation_api_inflight = None

        try:
            snapshot = self._generation_api_public_snapshot()
        except Exception as exc:
            snapshot = {
                'nativeOperations': True,
                'error': str(exc),
            }
        message = str(
            result.get('message')
            or (error.get('message') if isinstance(error, dict) else error)
            or ('작업이 완료되었습니다.' if ok else '작업에 실패했습니다.')
        )
        self._emit_generation_api_event({
            'type': 'completed' if ok else 'error',
            'action': action,
            'operationId': operation_id,
            'ok': ok,
            'message': message,
            'error': error,
            'result': result,
            'snapshot': snapshot,
        })

    @pyqtSlot(str, str, str, result=str)
    def runBackendRuntimeOperation(self, engine: str, action: str, payload_json: str) -> str:
        """runtime 변경 작업을 worker에서 시작하고 즉시 operation id를 반환."""
        if self._backend_runtime_is_web_mode():
            return json.dumps({
                'ok': False,
                'accepted': False,
                'error': '웹 모드에서는 로컬 백엔드 설치·실행을 변경할 수 없습니다.',
            }, ensure_ascii=False)

        try:
            ui_engine, core_engine = self._backend_runtime_engine(engine)
            action = str(action or '').strip().lower()
            allowed = {
                'install', 'update', 'check_update', 'start', 'stop', 'use',
                'set_auto_start', 'save_extension_dir', 'install_extension',
                'update_extension', 'check_extension', 'check_extensions',
                'set_install_root', 'use_managed_install',
                'set_primary_model_engine', 'set_extra_args', 'set_launch_options',
            }
            if action not in allowed:
                raise ValueError(f'지원하지 않는 runtime 작업입니다: {action}')
            payload = json.loads(payload_json) if payload_json else {}
            if not isinstance(payload, dict):
                raise ValueError('payload는 JSON 객체여야 합니다')
        except Exception as exc:
            return json.dumps(
                {'ok': False, 'accepted': False, 'error': str(exc)},
                ensure_ascii=False,
            )

        import uuid
        operation_id = uuid.uuid4().hex
        with self._backend_runtime_lock:
            current = self._backend_runtime_inflight.get(ui_engine)
            if current:
                return json.dumps({
                    'ok': False,
                    'accepted': False,
                    'operationId': current,
                    'error': f'{ui_engine} runtime 작업이 이미 진행 중입니다.',
                }, ensure_ascii=False)
            self._backend_runtime_inflight[ui_engine] = operation_id

        threading.Thread(
            target=self._run_backend_runtime_operation,
            args=(ui_engine, core_engine, action, payload, operation_id),
            daemon=True,
            name=f'backend-runtime-{ui_engine}-{action}',
        ).start()
        return json.dumps({
            'ok': True,
            'accepted': True,
            'engine': ui_engine,
            'action': action,
            'operationId': operation_id,
        }, ensure_ascii=False)

    def _emit_backend_runtime_event(self, payload: dict) -> None:
        try:
            self.backendRuntimeEvent.emit(json.dumps(payload, ensure_ascii=False, default=str))
        except RuntimeError:
            # 앱 종료 중 bridge QObject가 이미 정리된 경우 결과를 버린다.
            pass

    def _run_backend_runtime_operation(
        self,
        ui_engine: str,
        core_engine: str,
        action: str,
        payload: dict,
        operation_id: str,
    ) -> None:
        """Qt 비의존 runtime manager를 worker에서 실행하는 단일 경계."""
        self._emit_backend_runtime_event({
            'engine': ui_engine,
            'type': 'started',
            'action': action,
            'operationId': operation_id,
            'message': f'{ui_engine} {action} 작업을 시작했습니다.',
            'startup': bool(payload.get('startup', False)),
        })

        def progress(update):
            event = dict(update) if isinstance(update, dict) else {'message': str(update)}
            event.update({
                'engine': ui_engine,
                'type': 'progress',
                'action': action,
                'operationId': operation_id,
                'startup': bool(payload.get('startup', False)),
            })
            self._emit_backend_runtime_event(event)

        raw_result = None
        try:
            from core.backend_runtime import get_backend_runtime_manager
            manager = get_backend_runtime_manager()
            raw_result = manager.execute(
                core_engine,
                action,
                dict(payload),
                on_progress=progress,
            )
            if not isinstance(raw_result, dict):
                raw_result = {
                    'ok': False,
                    'engine': core_engine,
                    'action': action,
                    'error': 'runtime manager가 잘못된 결과를 반환했습니다.',
                }
        except Exception as exc:
            error = exc.as_dict() if hasattr(exc, 'as_dict') else str(exc)
            raw_result = {
                'ok': False,
                'engine': core_engine,
                'action': action,
                'error': error,
            }
        finally:
            with self._backend_runtime_lock:
                if self._backend_runtime_inflight.get(ui_engine) == operation_id:
                    self._backend_runtime_inflight.pop(ui_engine, None)

        try:
            from core.backend_runtime import get_backend_runtime_manager
            public = self._backend_runtime_public_snapshot(
                get_backend_runtime_manager().snapshot()
            )
        except Exception as exc:
            public = {
                'ok': False,
                'nativeOperations': True,
                'engines': {},
                'error': str(exc),
            }

        ok = bool(raw_result.get('ok', False))
        error = raw_result.get('error')
        if isinstance(error, dict):
            message = str(error.get('message') or error.get('error') or error)
        else:
            message = str(error or raw_result.get('message') or '')
        result = {
            key: value for key, value in raw_result.items()
            if key not in {'snapshot'}
        }
        event = {
            'engine': ui_engine,
            'type': 'completed' if ok else 'error',
            'action': action,
            'operationId': operation_id,
            'ok': ok,
            'result': result,
            'error': error if not ok else None,
            'message': message,
            'startup': bool(payload.get('startup', False)),
            'activate': bool(ok and raw_result.get('activate', False)),
            'state': public.get('engines', {}).get(ui_engine, {}),
            'snapshot': public,
        }
        self._emit_backend_runtime_event(event)

    @pyqtSlot(str, result=str)
    def selectBackendExtensionDirectory(self, engine: str) -> str:
        """기존 extensions/custom_nodes 폴더를 고르는 desktop-only 슬롯."""
        if self._backend_runtime_is_web_mode():
            return json.dumps({
                'ok': False,
                'error': '웹 모드에서는 로컬 폴더를 선택할 수 없습니다.',
            }, ensure_ascii=False)
        try:
            from ui.native_dialogs import select_directory

            ui_engine, _core_engine = self._backend_runtime_engine(engine)
            state = self._backend_runtime_public_snapshot()
            current = str(
                state.get('engines', {}).get(ui_engine, {}).get('extensionDir') or ''
            )
            selected = select_directory(
                self.parent(),
                '기존 extensions 폴더 선택' if ui_engine == 'forge' else '기존 custom_nodes 폴더 선택',
                current,
            )
            if not selected:
                return json.dumps({'ok': False, 'cancelled': True}, ensure_ascii=False)
            return json.dumps({'ok': True, 'path': selected}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def selectBackendInstallDirectory(self, engine: str) -> str:
        """Forge/Comfy 기존 설치 루트를 고르는 desktop-only 슬롯."""
        if self._backend_runtime_is_web_mode():
            return json.dumps({
                'ok': False,
                'error': '웹 모드에서는 로컬 설치 폴더를 선택할 수 없습니다.',
            }, ensure_ascii=False)
        try:
            from ui.native_dialogs import select_directory

            ui_engine, _core_engine = self._backend_runtime_engine(engine)
            state = self._backend_runtime_public_snapshot()
            runtime = state.get('engines', {}).get(ui_engine, {})
            current = str(
                runtime.get('existingRoot')
                or runtime.get('installRoot')
                or runtime.get('sourceRoot')
                or ''
            )
            title = (
                '기존 Forge Neo 설치 폴더 선택'
                if ui_engine == 'forge'
                else '기존 ComfyUI 또는 portable 폴더 선택'
            )
            selected = select_directory(self.parent(), title, current)
            if not selected:
                return json.dumps({'ok': False, 'cancelled': True}, ensure_ascii=False)
            return json.dumps({'ok': True, 'path': selected}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)

    # ── Editor ──

    editorResult = pyqtSignal(str)  # 에디터 비동기 처리 결과 JSON (path|mask_base64|error + operation, job_id)
    # 프리뷰 축소 한도 — 이보다 길면 줄여서 처리한다(1024면 대부분 20ms 내).
    _PREVIEW_MAX_EDGE = 1024

    @pyqtSlot(str, str, str, result=str)
    def editorProcess(self, image_path: str, operation: str, params_json: str) -> str:
        """에디터 이미지 처리 — 무거운 작업(YOLO/SAM/rembg/OpenCV)을 백그라운드 스레드로.

        동기 슬롯으로 GUI 스레드에서 전부 돌면 클릭마다 창 전체가 수~수십 초 멈춤
        (YOLO 로드+추론, SAM 체크포인트 로드, alpha_matting rembg).
        즉시 {'started': True, 'job_id': n}을 반환하고, 완료 시 editorResult
        시그널로 결과를 보낸다 (generateThumbnails / startCaptioning과 동일 패턴).
        """
        clean_path = _normalize_vue_path(image_path)
        if not clean_path:
            logger.warning("[Editor] invalid or forbidden path")
            return json.dumps({'error': '유효하지 않은 이미지 경로입니다'})

        # params가 객체로 올 수도 있고 JSON 문자열로 올 수도 있음
        try:
            if isinstance(params_json, str):
                params = json.loads(params_json) if params_json else {}
            else:
                params = params_json or {}
            if not isinstance(params, dict):
                params = {}
        except Exception as e:
            return json.dumps({'error': f'잘못된 파라미터: {e}'})

        self._editor_job_seq = getattr(self, '_editor_job_seq', 0) + 1
        job_id = self._editor_job_seq

        import threading

        def _work():
            result_json = self._editor_process_impl(clean_path, operation, params)
            try:
                payload = json.loads(result_json)
            except Exception:
                payload = {'error': '에디터 내부 오류'}
            payload['job_id'] = job_id
            payload['operation'] = operation
            try:
                # 비-Qt 스레드 emit은 queued connection이라 안전
                self.editorResult.emit(json.dumps(payload))
            except Exception:
                pass

        threading.Thread(target=_work, daemon=True).start()
        return json.dumps({'started': True, 'job_id': job_id})

    def _editor_process_impl(self, clean_path: str, operation: str, params: dict) -> str:
        """editorProcess 본체 — 워커 스레드에서 실행. 위젯 직접 접근 금지(시그널 emit만)."""
        try:
            import cv2
            import numpy as np

            # IMREAD_UNCHANGED — 기본 IMREAD_COLOR는 알파를 버린다.
            # 그래서 '배경 제거' 후 아무 편집이나 하면 투명도가 죽고 배경이 검게 됐다.
            img = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return json.dumps({'error': '이미지를 읽을 수 없습니다 (OpenCV)'})
            # 채널 정규화: 흑백 → BGR. BGRA는 그대로 두고 각 연산이 알파를 보존한다.
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 2:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

            # ── 실시간 프리뷰 ──────────────────────────────────────────────
            # 슬라이더를 움직이는 동안 '적용해야만 결과를 아는' 문제를 없앤다.
            # CSS 필터로 흉내 내면 백엔드 연산(HSV 채도, 레벨, 12종 필터)과 어긋나므로
            # 같은 코드로 축소본을 실제 처리해 돌려준다.
            #
            # 예전에는 마스크가 있으면 "좌표가 어긋난다"는 이유로 프리뷰를 막았다.
            # 그런데 바로 아래에서 마스크를 이미 `img` 크기에 맞춰 리사이즈하고 있어
            # 어긋날 일이 없다 — 그래서 모자이크·블러·검은띠도 프리뷰가 된다.
            # (축소본에서 INTER_NEAREST 로 줄이므로 가장자리가 한두 픽셀 흔들릴 수는
            #  있지만, 확정 결과는 원본 해상도에서 다시 계산한다.)
            is_preview = bool(params.get('preview'))
            if is_preview:
                _h, _w = img.shape[:2]
                _long = max(_h, _w)
                if _long > self._PREVIEW_MAX_EDGE:
                    _r = self._PREVIEW_MAX_EDGE / float(_long)
                    img = cv2.resize(img, (max(1, int(_w * _r)), max(1, int(_h * _r))),
                                     interpolation=cv2.INTER_AREA)

            # ── 마스크 처리 (base64 PNG → numpy) ──
            mask = None
            mask_b64 = params.get('mask_base64')
            if mask_b64:
                import base64
                from io import BytesIO
                from PIL import Image as PILImage
                header, b64data = mask_b64.split(',', 1) if ',' in mask_b64 else ('', mask_b64)
                mask_bytes = base64.b64decode(b64data)
                mask_pil = PILImage.open(BytesIO(mask_bytes)).convert('L')
                mask = np.array(mask_pil)
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 선택 영역 추출 (좌표 정수화)
            sel = params.get('selection')
            if sel:
                x1, y1 = int(float(sel.get('x', 0))), int(float(sel.get('y', 0)))
                x2 = x1 + int(float(sel.get('w', img.shape[1])))
                y2 = y1 + int(float(sel.get('h', img.shape[0])))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            else:
                x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]

            has_roi = x2 > x1 and y2 > y1

            # ── 마스크 기반 효과 (정밀 적용) ──
            def _apply_effect_with_mask(src, effect_mask, effect_type, strength_val):
                """마스크 영역에만 효과 적용. BGR/BGRA 모두 대응(알파 보존)."""
                from core.editor_ops import split_alpha, merge_alpha
                base, alpha_ch = split_alpha(src)
                result = base.copy()
                sel = effect_mask > 127
                if effect_type == 'mosaic':
                    s = max(2, strength_val)
                    h_i, w_i = base.shape[:2]
                    small = cv2.resize(base, (max(1, w_i // s), max(1, h_i // s)))
                    mosaic = cv2.resize(small, (w_i, h_i), interpolation=cv2.INTER_NEAREST)
                    result[sel] = mosaic[sel]
                elif effect_type == 'censor_bar':
                    result[sel] = 0
                elif effect_type == 'blur':
                    k = max(1, strength_val) | 1
                    blurred = cv2.GaussianBlur(base, (k, k), 0)
                    result[sel] = blurred[sel]
                return merge_alpha(result, alpha_ch)

            if operation in ('mosaic', 'censor_bar', 'blur'):
                strength_val = int(params.get('strength', 15))
                if mask is not None:
                    # 마스크 기반 정밀 적용
                    img = _apply_effect_with_mask(img, mask, operation, strength_val)
                elif has_roi:
                    # 사각형 영역 기반 적용 (fallback)
                    roi = img[y1:y2, x1:x2]
                    if operation == 'mosaic':
                        h_r, w_r = roi.shape[:2]
                        small = cv2.resize(roi, (max(1, w_r // max(2, strength_val)), max(1, h_r // max(2, strength_val))))
                        roi = cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
                    elif operation == 'censor_bar':
                        roi[:] = 0
                    elif operation == 'blur':
                        k = max(1, strength_val) | 1
                        roi = cv2.GaussianBlur(roi, (k, k), 0)
                    img[y1:y2, x1:x2] = roi

            elif operation in ('auto_censor', 'auto_detect'):
                # YOLO 기반 자동 검열 / 마스크만 감지
                try:
                    from tabs.editor.mosaic_panel import _load_yolo_model_paths, _is_sam_file
                    model_paths = _load_yolo_model_paths()
                    sam_choice = str(params.get('sam_model', 'auto')).lower()
                    detect_prompt = str(params.get('detect_prompt') or '').strip()

                    # SAM3는 '텍스트 프롬프트' 세그멘터다. 예전에는 YOLO 모델이 없으면
                    # 여기서 바로 에러를 냈고, YOLO 박스가 0개면 SAM3를 아예 건너뛰어서
                    # 사용자가 SAM3를 골라도 텍스트만으로는 절대 쓸 수 없었다.
                    sam3_standalone = (sam_choice == 'sam3' and bool(detect_prompt))
                    if not model_paths and not sam3_standalone:
                        return json.dumps({'error': 'YOLO 모델을 먼저 추가하세요 (+ADD .PT) '
                                                    '— 또는 SAM3를 선택하고 Detect Prompt를 입력하세요'})
                    conf = float(params.get('confidence', 0.25))
                    h_img, w_img = img.shape[:2]
                    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    detect_count = 0
                    yolo_boxes = []
                    has_seg_mask = False
                    loaded_names, failed = [], []

                    # 단일 패스: 모델당 1회만 로드 → mask + bbox 동시 수집
                    for mp in model_paths:
                        if not os.path.exists(mp):
                            failed.append((mp, 'not found'))
                            continue
                        if _is_sam_file(mp):
                            print(f"[YOLO] Skip SAM model (not a detector): {os.path.basename(mp)}")
                            continue
                        try:
                            # 유휴 캐시 — 예전에는 클릭마다 루프 안에서 새로 로딩했다
                            # (모델 3개면 클릭 1회에 3회 로딩)
                            from core.model_cache import YOLO_CACHE

                            def _load_yolo(path):
                                from ultralytics import YOLO
                                print(f"[YOLO] Loading {os.path.basename(path)}...")
                                return YOLO(path)

                            model = YOLO_CACHE.get(mp, _load_yolo)
                        except Exception as me:
                            print(f"[YOLO] Model load failed: {mp} — {me}")
                            failed.append((os.path.basename(mp), str(me)))
                            continue
                        loaded_names.append(os.path.basename(mp))
                        try:
                            results = model(img, conf=conf, verbose=False)
                        except Exception as ie:
                            print(f"[YOLO] Inference failed: {mp} — {ie}")
                            failed.append((os.path.basename(mp), f'inference: {ie}'))
                            continue
                        for r in results:
                            # 세그먼트 마스크 (성기 형태 정밀)
                            if r.masks is not None:
                                has_seg_mask = True
                                for m_tensor in r.masks.data:
                                    m_np = m_tensor.cpu().numpy().astype(np.float32)
                                    m_resized = cv2.resize(m_np, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                                    combined_mask[m_resized > 0.3] = 255
                                    detect_count += 1
                            # 박스 (SAM 정밀화 입력 + 마스크 폴백)
                            if r.boxes is not None:
                                for box in r.boxes.xyxy:
                                    bx1, by1, bx2, by2 = map(int, box.tolist())
                                    bx1, by1 = max(0, bx1), max(0, by1)
                                    bx2, by2 = min(w_img, bx2), min(h_img, by2)
                                    if bx2 > bx1 and by2 > by1:
                                        yolo_boxes.append((bx1, by1, bx2, by2))
                                        if r.masks is None:
                                            combined_mask[by1:by2, bx1:bx2] = 255
                                            detect_count += 1

                    if not loaded_names and not sam3_standalone:
                        # 등록된 모든 모델이 실패한 경우 명확한 에러
                        msg = '; '.join(f'{os.path.basename(n)}: {e}' for n, e in failed) or '모든 YOLO 모델 로드 실패'
                        try:
                            self.showNotification.emit('error', f'YOLO 모델 로드 실패 — {msg[:200]}')
                        except Exception:
                            pass
                        return json.dumps({'error': f'YOLO 모델 로드 실패 — {msg}'})

                    # 일부만 실패한 경우 경고
                    if failed:
                        fail_msg = ', '.join(os.path.basename(n) for n, _e in failed[:3])
                        if len(failed) > 3:
                            fail_msg += f' 외 {len(failed) - 3}개'
                        try:
                            self.showNotification.emit('warning', f'YOLO 일부 모델 실패: {fail_msg}')
                        except Exception:
                            pass

                    print(f"[YOLO] Loaded {loaded_names} → {detect_count} regions, {len(yolo_boxes)} boxes, seg_mask={has_seg_mask}")

                    # SAM 정밀 마스킹 — 사용자가 모델 선택 가능.
                    # SAM3 + detect prompt면 YOLO 박스가 없어도 단독으로 돈다.
                    if (yolo_boxes or sam3_standalone) and sam_choice != 'off':
                        try:
                            from core.sam_refiner import refine_boxes_with_sam, find_sam_model
                            from tabs.editor.mosaic_panel import get_editor_models_dir
                            models_dir = get_editor_models_dir()
                            sam_path, sam_type = find_sam_model(models_dir, prefer_type=sam_choice)
                            print(f"[SAM] choice={sam_choice}, models_dir={models_dir}, found={sam_path}, type={sam_type}, has_seg={has_seg_mask}")

                            if has_seg_mask and sam_type != 'sam3':
                                # YOLO seg 마스크가 이미 있고 SAM3가 아니면 정밀화 생략
                                print("[SAM] YOLO seg mask available, skipping SAM")
                            elif sam_path:
                                # SAM3 전용: 마스크에서 빼고 싶은 영역의 텍스트 프롬프트
                                #   예: 'face' → 얼굴 영역을 검출해서 최종 마스크에서 빼기
                                excl_prompt = params.get('exclude_prompt') or params.get('excludePrompt')
                                excl_prompt = str(excl_prompt).strip() if excl_prompt else None
                                if excl_prompt and sam_type == 'sam3':
                                    print(f"[SAM3] exclude prompt requested: '{excl_prompt}'")
                                if detect_prompt and sam_type == 'sam3':
                                    print(f"[SAM3] detect prompt: '{detect_prompt}'")
                                # 사용자 알림 콜백 — SAM3 exclude 안전장치 발동 시 토스트로 전달
                                def _sam_notify(level, message):
                                    try:
                                        self.showNotification.emit(level, message)
                                    except Exception:
                                        pass
                                sam_mask = refine_boxes_with_sam(
                                    img, yolo_boxes, models_dir,
                                    sam_model_path=sam_path, sam_type=sam_type,
                                    yolo_model_paths=model_paths,
                                    text_prompt=(detect_prompt or None),
                                    exclude_prompt=excl_prompt,
                                    notify=_sam_notify,
                                )
                                if sam_mask.any():
                                    combined_mask = sam_mask
                                    pixel_count = int(sam_mask.sum() / 255)
                                    print(f"[SAM] ✓ Refined mask applied ({sam_type}, {len(yolo_boxes)} boxes → {pixel_count} pixels)")
                                else:
                                    print("[SAM] No mask generated, using YOLO bbox")
                            else:
                                if sam_choice != 'auto':
                                    print(f"[SAM] '{sam_choice}' 모델이 editor_models/에 없음 — bbox 사용")
                                else:
                                    print(f"[SAM] No SAM model in {models_dir}, using YOLO bbox")
                        except ImportError as ie:
                            print(f"[SAM] Import error: {ie}")
                            try:
                                self.showNotification.emit('warning', f'SAM 라이브러리 미설치 — bbox 마스크 사용 ({type(ie).__name__})')
                            except Exception:
                                pass
                        except Exception as sam_e:
                            import traceback
                            print(f"[SAM] Error: {sam_e}")
                            traceback.print_exc()
                            try:
                                self.showNotification.emit('error', f'SAM 정밀화 실패: {sam_e}')
                            except Exception:
                                pass

                    if operation == 'auto_detect':
                        # MASK ONLY: 마스크를 base64로 반환 (적용 안함)
                        import base64
                        from io import BytesIO
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray(combined_mask)
                        buf = BytesIO()
                        mask_pil.save(buf, format='PNG')
                        mask_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                        return json.dumps({'mask_base64': mask_b64, 'detect_count': detect_count, 'path': clean_path})
                    else:
                        # AUTO CENSOR: 감지 + 모자이크 적용
                        if combined_mask.any():
                            # 마스크 약간 확장 (dilate)으로 경계 커버
                            kernel = np.ones((5, 5), np.uint8)
                            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
                            img = _apply_effect_with_mask(img, combined_mask, 'mosaic', 15)
                        else:
                            return json.dumps({'error': f'감지된 영역이 없습니다 (conf={conf})'})
                except Exception as e:
                    from core.error_handler import handle_error
                    handle_error('E100', 'Auto Censor', e)
                    return json.dumps({'error': f'[E100] Auto censor 실패: {e}'})

            elif operation == 'text_watermark':
                # 텍스트 워터마크
                from PIL import Image as PILImage, ImageDraw, ImageFont
                pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
                overlay = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                text = params.get('text', 'Watermark')
                font_size = int(params.get('fontSize', 36))
                opacity = float(params.get('opacity', 0.5))
                x_pct = float(params.get('xPct', 50))
                y_pct = float(params.get('yPct', 50))
                rotation = float(params.get('rotation', 0))
                try:
                    font_family = params.get('fontFamily', 'Arial')
                    font = ImageFont.truetype(font_family, font_size)
                except Exception:
                    font = ImageFont.load_default()
                alpha_val = int(opacity * 255)
                # 예전에는 흰색 고정이라 패널의 색상 선택이 결과에 반영되지 않았다.
                _hex = str(params.get('color', '#FFFFFF')).lstrip('#')
                try:
                    _r, _g, _b = int(_hex[0:2], 16), int(_hex[2:4], 16), int(_hex[4:6], 16)
                except Exception:
                    _r, _g, _b = 255, 255, 255
                color = (_r, _g, _b, alpha_val)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x = int(pil_img.width * x_pct / 100 - tw / 2)
                y = int(pil_img.height * y_pct / 100 - th / 2)

                if params.get('tile'):
                    # 타일 반복
                    for ty in range(-th, pil_img.height + th, th + 40):
                        for tx in range(-tw, pil_img.width + tw, tw + 40):
                            draw.text((tx, ty), text, fill=color, font=font)
                else:
                    draw.text((x, y), text, fill=color, font=font)

                if rotation != 0:
                    overlay = overlay.rotate(-rotation, expand=False, center=(pil_img.width // 2, pil_img.height // 2))
                result = PILImage.alpha_composite(pil_img, overlay)
                img = cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)

            elif operation == 'image_watermark':
                # 예전에는 무조건 에러를 반환하는 스텁이었다 — 패널·파일 다이얼로그는
                # 있는데 백엔드가 없어 end-to-end 로 죽어 있었다.
                wm_path = str(params.get('watermark_path', '') or '')
                if not wm_path or not os.path.isfile(wm_path):
                    return json.dumps({'error': '워터마크 이미지를 먼저 불러오세요'})
                from PIL import Image as PILImage
                try:
                    wm = PILImage.open(wm_path).convert('RGBA')
                except Exception as e:
                    return json.dumps({'error': f'워터마크 이미지를 열 수 없습니다: {e}'})

                base = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
                scale = max(1.0, float(params.get('scale', 100))) / 100.0
                new_w = max(1, int(wm.width * scale))
                new_h = max(1, int(wm.height * scale))
                if params.get('clamp', True):
                    # 원본 밖으로 나가지 않도록 축소
                    ratio = min(1.0, base.width / new_w, base.height / new_h)
                    new_w, new_h = max(1, int(new_w * ratio)), max(1, int(new_h * ratio))
                wm = wm.resize((new_w, new_h), PILImage.LANCZOS)

                opacity = max(0.0, min(1.0, float(params.get('opacity', 0.5))))
                if opacity < 1.0:
                    alpha = wm.getchannel('A').point(lambda v: int(v * opacity))
                    wm.putalpha(alpha)

                x = int(base.width * float(params.get('xPct', 50)) / 100 - new_w / 2)
                y = int(base.height * float(params.get('yPct', 50)) / 100 - new_h / 2)
                if params.get('clamp', True):
                    x = max(0, min(x, base.width - new_w))
                    y = max(0, min(y, base.height - new_h))

                overlay = PILImage.new('RGBA', base.size, (0, 0, 0, 0))
                overlay.paste(wm, (x, y), wm)
                result = PILImage.alpha_composite(base, overlay)
                img = cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)

            elif operation == 'rotate_cw':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif operation == 'rotate_ccw':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif operation == 'flip_h':
                img = cv2.flip(img, 1)
            elif operation == 'flip_v':
                img = cv2.flip(img, 0)
            elif operation == 'resize':
                w = max(1, int(params.get('width', img.shape[1])))
                h = max(1, int(params.get('height', img.shape[0])))
                # 축소는 INTER_AREA 가 맞다 — 기본 INTER_LINEAR 는 축소에서 계단이 생긴다.
                interp = cv2.INTER_AREA if (w < img.shape[1] or h < img.shape[0]) else cv2.INTER_LANCZOS4
                img = cv2.resize(img, (w, h), interpolation=interp)
            elif operation == 'crop' and has_roi:
                img = img[y1:y2, x1:x2]
            elif operation == 'remove_bg':
                try:
                    from rembg import remove
                    from PIL import Image as PILImage
                    quality = params.get('quality', 'balanced')
                    pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    rm_kwargs = {}
                    if quality in ('balanced', 'quality'):
                        rm_kwargs['alpha_matting'] = True
                        rm_kwargs['alpha_matting_foreground_threshold'] = 240 if quality == 'balanced' else 270
                        rm_kwargs['alpha_matting_background_threshold'] = 10 if quality == 'balanced' else 20
                        rm_kwargs['alpha_matting_erode_size'] = 10 if quality == 'balanced' else 15

                    result = remove(pil_img, **rm_kwargs)
                    img = cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)

                    # Quality 모드: 엣지 정제
                    if quality == 'quality':
                        try:
                            from core.edge_refiner import refine_alpha
                            img = refine_alpha(img)
                        except Exception as re:
                            print(f"[Editor] Edge refine skipped: {re}")
                except Exception as e:
                    return json.dumps({'error': f'배경 제거 실패: {e}'})
            
            elif operation == 'color_adjust':
                from core.editor_ops import split_alpha, merge_alpha
                _bgr, _alpha = split_alpha(img)
                b_val = params.get('brightness', 0)
                c_val = params.get('contrast', 0)
                s_val = params.get('saturation', 0)
                if b_val != 0: _bgr = cv2.convertScaleAbs(_bgr, alpha=1, beta=b_val)
                if c_val != 0:
                    factor = (100 + c_val) / 100.0
                    _bgr = cv2.convertScaleAbs(_bgr, alpha=factor, beta=0)
                if s_val != 0:
                    hsv = cv2.cvtColor(_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:,:,1] *= (100 + s_val) / 100.0
                    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                    _bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                img = merge_alpha(_bgr, _alpha)

            # ── 아래는 예전에 핸들러가 아예 없어서 '원본 재저장 = 무반응'이던 작업들 ──
            elif operation == 'auto_correct':
                from core.editor_ops import auto_correct
                img = auto_correct(img)

            elif operation == 'adv_color':
                from core.editor_ops import adv_color
                img = adv_color(
                    img,
                    black_point=params.get('blackPoint', 0),
                    white_point=params.get('whitePoint', 255),
                    gamma=params.get('gamma', 1.0),
                    temperature=params.get('temperature', 0),
                    tint=params.get('tint', 0),
                    curves=params.get('curves'),
                )

            elif operation == 'filter':
                from core.editor_ops import apply_filter
                try:
                    img = apply_filter(img, params.get('filter'),
                                       float(params.get('strength', 100)) / 100.0)
                except ValueError as fe:
                    return json.dumps({'error': str(fe)})

            elif operation == 'move_region':
                from core.editor_ops import move_region
                if mask is None:
                    return json.dumps({'error': '이동할 영역을 먼저 마스킹하세요'})
                img = move_region(
                    img, mask,
                    dx=float(params.get('dx', 0)), dy=float(params.get('dy', 0)),
                    rotation=float(params.get('rotation', 0)),
                    scale=float(params.get('scale', 100)),
                    fill_color=str(params.get('fillColor', 'black')),
                )

            elif operation == 'perspective':
                from core.editor_ops import perspective as _perspective
                corners = params.get('corners')
                if not corners:
                    return json.dumps({'error': '원근 보정: 꼭짓점 4개가 필요합니다'})
                try:
                    img = _perspective(img, corners,
                                       width=params.get('width'), height=params.get('height'))
                except ValueError as pe:
                    return json.dumps({'error': str(pe)})

            elif operation == 'restore':
                # 모자이크 지우개 — 효과 적용 '전' 이미지에서 마스크 영역 픽셀을 되가져온다.
                # 예전에는 프론트 캔버스에만 그려서 저장에 전혀 반영되지 않았다.
                if mask is None:
                    return json.dumps({'error': '복원할 영역이 없습니다'})
                src_path = _normalize_vue_path(params.get('source_path') or '')
                if not src_path or not os.path.exists(src_path):
                    return json.dumps({'error': '복원 원본 이미지를 찾을 수 없습니다'})
                src_img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                if src_img is None:
                    return json.dumps({'error': '복원 원본 이미지를 읽을 수 없습니다'})
                if src_img.ndim == 2:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                if src_img.shape[:2] != img.shape[:2]:
                    return json.dumps({'error': '복원 원본과 크기가 달라 되돌릴 수 없습니다'})
                # 채널 수를 맞춘 뒤 마스크 영역만 교체
                if src_img.shape[2] != img.shape[2]:
                    if img.shape[2] == 4 and src_img.shape[2] == 3:
                        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2BGRA)
                    elif img.shape[2] == 3 and src_img.shape[2] == 4:
                        src_img = src_img[:, :, :3]
                img[mask > 127] = src_img[mask > 127]

            elif operation == 'heal':
                from core.editor_ops import heal as _heal
                heal_b64 = params.get('mask_base64')
                if not heal_b64:
                    return json.dumps({'error': '복원할 자리를 먼저 칠하세요'})
                import base64 as _b64
                _raw = heal_b64.split(',', 1)[-1]
                _buf = np.frombuffer(_b64.b64decode(_raw), dtype=np.uint8)
                heal_mask = cv2.imdecode(_buf, cv2.IMREAD_UNCHANGED)
                if heal_mask is None:
                    return json.dumps({'error': '복원 마스크를 읽지 못했습니다'})
                img = _heal(img, heal_mask, int(params.get('radius', 3)))

            elif operation == 'flatten':
                from core.editor_ops import flatten as _flatten
                overlay = None
                overlay_b64 = params.get('overlay_base64')
                if overlay_b64:
                    import base64 as _b64
                    _raw = overlay_b64.split(',', 1)[-1]
                    _buf = np.frombuffer(_b64.b64decode(_raw), dtype=np.uint8)
                    overlay = cv2.imdecode(_buf, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    return json.dumps({'error': '병합할 드로잉 레이어가 없습니다'})
                img = _flatten(img, overlay, float(params.get('opacity', 100)) / 100.0)

            else:
                # 예전에는 모르는 op가 모든 elif를 통과해 원본을 그대로 다시 저장했다.
                # 사용자에겐 '눌렀는데 아무 일도 안 남 + undo 스택만 늘어남'이었다.
                return json.dumps({'error': f'지원하지 않는 편집 작업: {operation}'})

            if is_preview:
                # 파일을 만들지 않는다 — 프리뷰는 undo 스택에도 들어가면 안 된다.
                ok, buf = cv2.imencode('.png', img)
                if not ok:
                    return json.dumps({'error': '프리뷰 인코딩 실패'})
                import base64 as _b64p
                return json.dumps({
                    'preview': True,
                    'image_base64': 'data:image/png;base64,' + _b64p.b64encode(buf.tobytes()).decode('ascii'),
                    'width': img.shape[1], 'height': img.shape[0],
                })

            # 결과 저장 — uuid4로 파일명 충돌 제거.
            # (예전 f"edited_{int(time.time())}_{randint(100,999)}"는 같은 초에 1/900 확률로
            #  충돌해 직전 편집본을 덮어썼다.)
            import uuid
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'image_cache', 'editor_temp')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"edited_{uuid.uuid4().hex}.png")
            if not cv2.imwrite(out_path, img):
                return json.dumps({'error': '편집 결과 저장 실패 (디스크 공간/권한 확인)'})

            # 세션 임시본이 무한 누적되지 않게 정리 (실측 115개 256MB까지 쌓여 있었음)
            try:
                from core.cache_cleanup import prune_editor_temp
                prune_editor_temp(out_dir, keep=_EDITOR_TEMP_KEEP)
            except Exception:
                pass

            return json.dumps({'path': out_path.replace('\\', '/'), 'width': img.shape[1], 'height': img.shape[0]})
        except Exception as e:
            from core.error_handler import handle_error
            handle_error('E040', f'Editor: {operation}', e)
            return json.dumps({'error': f'[E040] {operation}: {e}'})

    # ── 갤러리 ──

    @pyqtSlot(result=str)
    def getLastGalleryFolder(self) -> str:
        """ui_prefs의 마지막 Gallery 폴더 경로 반환 (옛 txt는 1회 흡수)."""
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        prefs_path = os.path.join(base, 'config', 'ui_prefs.json')
        try:
            from core.config_migration import load_ui_prefs
            prefs = load_ui_prefs(prefs_path)
            saved = str(prefs.get('galleryFolder', '') or '').strip()
            if saved:
                return saved
            legacy = os.path.join(base, 'config', 'gallery_last_folder.txt')
            if os.path.exists(legacy):
                with open(legacy, 'r', encoding='utf-8') as f:
                    saved = f.read().strip()
                if saved:
                    self._save_gallery_folder(saved)
                    return saved
        except Exception as e:
            logger.warning("getLastGalleryFolder failed: %s", e)
        from config import OUTPUT_DIR
        return OUTPUT_DIR

    def _save_gallery_folder(self, folder: str):
        import os
        from core.config_migration import load_ui_prefs, save_ui_prefs
        prefs_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'ui_prefs.json',
        )
        prefs = load_ui_prefs(prefs_path)
        prefs['galleryFolder'] = str(folder or '').strip()
        save_ui_prefs(prefs_path, prefs)

    def _gallery_images_payload(self, folder: str) -> str:
        """scandir의 stat 캐시를 이용해 날짜순 미디어 목록을 만든다."""
        import os
        from config import OUTPUT_DIR
        target = folder if folder else OUTPUT_DIR
        if not os.path.isdir(target):
            return json.dumps({'folder': folder, 'files': []})
        try:
            creator_root = os.path.join(target, 'creator')
            recursive_roots = (
                (creator_root,)
                if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(OUTPUT_DIR))
                else ()
            )
            files = _scan_gallery_media(target, recursive_roots)
        except Exception as e:
            logger.warning("getGalleryImages failed (%s): %s", target, e)
            files = []
        return json.dumps({'folder': folder, 'files': files})

    @pyqtSlot(str, result=str)
    def getGalleryImages(self, folder: str) -> str:
        """하위호환 동기 API. 신규 Vue 코드는 requestGalleryImages를 사용한다."""
        try:
            return json.dumps(json.loads(self._gallery_images_payload(folder))['files'])
        except Exception:
            return '[]'

    @pyqtSlot()
    def requestLoraManagerUrl(self):
        """sam-extra 임베드 LoRA Manager 주소 조회 (백그라운드 — 서버를 띄우느라 몇 초 걸림)."""
        def _lookup():
            try:
                from backends import get_backend
                backend = get_backend()
                if not backend or not hasattr(backend, 'get_lora_manager_url'):
                    return json.dumps({
                        'url': '', 'status': 'unsupported',
                        'message': '이 백엔드는 LoRA Manager 임베드를 지원하지 않습니다 '
                                   '(Forge Neo + sam-extra 필요)',
                    })
                return json.dumps(backend.get_lora_manager_url())
            except Exception as e:
                return json.dumps({'url': '', 'status': 'error', 'message': str(e)})

        self._run_async_lookup('lora_manager_url', _lookup, self.loraManagerUrlReady)

    @pyqtSlot(str)
    def requestGalleryImages(self, folder: str):
        """폴더 스캔/정렬을 백그라운드에서 수행해 GUI 멈춤을 방지한다."""
        key = f"gallery:{folder or '<default>'}"
        self._run_async_lookup(
            key,
            lambda: self._gallery_images_payload(folder),
            self.galleryImagesReady,
        )

    @pyqtSlot(result=str)
    def getFavorites(self) -> str:
        """즐겨찾기 목록 반환"""
        import os
        from config import FAVORITES_FILE
        if os.path.exists(FAVORITES_FILE):
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                return f.read()
        return json.dumps([])

    thumbnailReady = pyqtSignal(str)   # JSON {path, thumb} — 썸네일 1건 생성/조회 완료 통지

    @pyqtSlot(str, int)
    def generateThumbnails(self, paths_json: str, width: int = 256):
        """주어진 이미지 경로들의 썸네일을 백그라운드 스레드에서 생성/캐싱하고,
        각 건마다 thumbnailReady 시그널로 통지 (GUI 스레드 블로킹 방지).
        캐시: image_cache/thumbs/<sha1>.jpg"""
        try:
            paths = json.loads(paths_json) if paths_json else []
        except Exception:
            return
        if not paths:
            return
        import threading

        def _work():
            import hashlib
            from core.cache_cleanup import ensure_shard_dir, shard_path
            base = os.path.dirname(os.path.dirname(__file__))
            thumb_dir = os.path.join(base, 'image_cache', 'thumbs')
            try:
                os.makedirs(thumb_dir, exist_ok=True)
            except Exception:
                return
            self._migrate_thumb_cache_once(thumb_dir)
            for p in paths:
                thumb_url = ''
                try:
                    norm = os.path.normpath(p)
                    h = hashlib.sha1(f"{norm}@{width}".encode('utf-8')).hexdigest()
                    # sha1 앞 2자리로 샤딩 — 평면 디렉터리에 10만 개가 쌓이면
                    # NTFS 조회 자체가 느려진다 (실측 96,641개)
                    tp = shard_path(thumb_dir, h, '.jpg')
                    ensure_shard_dir(tp)
                    if not os.path.exists(tp):
                        if os.path.exists(p):
                            from PIL import Image, ImageOps
                            im = Image.open(p)
                            try:
                                im = ImageOps.exif_transpose(im)
                            except Exception:
                                pass
                            im = im.convert('RGB')
                            im.thumbnail((width, width), Image.Resampling.LANCZOS)
                            im.save(tp, 'JPEG', quality=80)
                    if os.path.exists(tp):
                        thumb_url = 'file:///' + tp.replace('\\', '/')
                except Exception:
                    thumb_url = ''
                try:
                    self.thumbnailReady.emit(json.dumps({'path': p, 'thumb': thumb_url}))
                except Exception:
                    pass
        threading.Thread(target=_work, daemon=True).start()

    def _migrate_thumb_cache_once(self, thumb_dir: str):
        """평면 thumbs/ → 샤딩 구조 이관 + 용량 상한 정리. 프로세스당 1회.

        이관은 5000개씩 끊어서 하므로 앱이 멈추지 않는다. 남은 건 다음 실행에서
        이어서 처리된다(썸네일은 언제든 재생성 가능하므로 중간에 끊겨도 안전).
        """
        if getattr(self, '_thumb_cache_migrated', False):
            return
        self._thumb_cache_migrated = True
        try:
            from core.cache_cleanup import migrate_flat_to_sharded, prune_thumbs
            moved = migrate_flat_to_sharded(thumb_dir)
            if moved:
                logger.info("썸네일 캐시 샤딩 이관: %d개", moved)
            removed = prune_thumbs(thumb_dir, _THUMB_CACHE_MAX_BYTES)
            if removed:
                logger.info("썸네일 캐시 정리: %d개 삭제", removed)
        except Exception as e:
            logger.warning("썸네일 캐시 정리 실패 (무시): %s", e)

    searchResultsReady = pyqtSignal(str)   # JSON results
    searchResultLineage = pyqtSignal(str)  # JSON {label,fingerprint,snapshot_id}
    queueUpdated = pyqtSignal(str)         # JSON queue state
    eventSearchResults = pyqtSignal(str)   # JSON event results
    generationProgress = pyqtSignal(int, int)  # current, total steps
    generationPreview = pyqtSignal(str)   # 생성 중 중간 그림 — 접두사 없는 base64 (Forge current_image)

    @pyqtSlot(str)
    def searchDanbooru(self, query_json: str):
        """Danbooru parquet 검색
        query JSON 구조:
          ratings: ['g', 's', ...]
          queries: { character: '...', copyright: '...', ... }    # 포함 조건
          excludes: { character: '...', ... }                     # 제외 조건
          combine_mode: 'and' | 'or'  (필드 간 결합 — 기본 'and')
          활성 데이터셋은 danbooru_optimized/dataset_manifest.json의 단일 릴리스
        """
        try:
            if isinstance(query_json, str):
                q = json.loads(query_json)
            else:
                q = query_json
            ratings = q.get('ratings', ['g'])
            queries = q.get('queries', {})
            excludes = q.get('excludes', {})
            combine_mode = str(q.get('combine_mode', 'and')).lower()
            if combine_mode not in ('and', 'or'):
                combine_mode = 'and'

            from workers.search_worker import PandasSearchWorker
            from config import PARQUET_DIR

            # 결과 cap 비활성화 — 사용자가 "무제한" 모드 선택 시
            self._disable_result_cap = bool(q.get('disable_result_cap', False))

            # 이전 검색 워커 정리 — 덮어쓰기만 하면 stale 결과가 새 결과를 덮거나
            # 실행 중 QThread 파괴로 크래시 가능 (ollamaEnhance와 동일 패턴)
            prev = getattr(self, '_search_worker', None)
            if prev is not None and prev.isRunning():
                try:
                    prev.results_ready.disconnect(self._on_search_results)
                except TypeError:
                    pass
                try:
                    prev.status_update.disconnect(self._on_search_status)
                except TypeError:
                    pass
                prev.stop()
                if not prev.wait(1000):
                    # 아직 도는 중 — 참조를 보관해 가비지 파괴 크래시 방지, 종료 시 자동 제거
                    if not hasattr(self, '_stale_search_workers'):
                        self._stale_search_workers = []
                    self._stale_search_workers.append(prev)
                    prev.finished.connect(
                        lambda w=prev: self._stale_search_workers.remove(w)
                        if w in getattr(self, '_stale_search_workers', []) else None)

            self._search_worker = PandasSearchWorker(
                PARQUET_DIR, ratings, queries, excludes,
                combine_mode=combine_mode,
                result_cap=None if self._disable_result_cap else 500_000,
            )
            self._search_worker.results_ready.connect(self._on_search_results)
            self._search_worker.status_update.connect(self._on_search_status)
            self._search_worker.start()
            cap_note = " · cap OFF" if self._disable_result_cap else ""
            self.searchStatus.emit(
                f"검색 중... (최신 데이터 · {combine_mode.upper()}{cap_note})"
            )
        except Exception as e:
            self.searchResultsReady.emit(json.dumps({'error': str(e)}))

    def _on_search_status(self, message):
        """현재 Search worker의 진행/오류만 Vue에 전달한다."""
        sender = self.sender()
        if sender is not None and sender is not getattr(self, '_search_worker', None):
            return
        text = str(message)
        if (
            sender is not None
            and getattr(sender, '_bridge_result_rejected', False)
            and not text.lstrip().startswith('❌')
        ):
            return
        self.searchStatus.emit(text)

    @staticmethod
    def _search_result_lineage_json(dataset_identity, snapshot_id):
        """Build the additive lineage event without changing result JSON shape."""
        if not isinstance(dataset_identity, dict):
            raise ValueError('검색 결과 dataset identity가 올바르지 않습니다.')
        return json.dumps({
            'label': dataset_identity.get('label'),
            'fingerprint': dataset_identity.get('fingerprint'),
            'snapshot_id': snapshot_id,
        }, ensure_ascii=False, separators=(',', ':'))

    def _on_search_results(self, results, total_count):
        """검색 결과 수신 → Vue 전달 + Python filtered_results 업데이트

        BUG FIX: search_worker는 to_dict('records')로 list[dict]를 emit하지만
        과거 코드가 DataFrame 가정으로 hasattr(iterrows)만 체크 → list 무시 → 0건.
        list / DataFrame 양쪽 지원 + 컬럼명도 두 스키마 (tag_string_* / *) 호환.
        """
        # 순서 역전 차단 — 현재 워커가 아닌(이전 검색의) 늦은 시그널은 무시
        sender = self.sender()
        if sender is not None and sender is not getattr(self, '_search_worker', None):
            print("[Search] stale worker result ignored")
            return
        try:
            import random as _rnd
            import uuid as _uuid
            from core.search_result_store import SearchResultStore

            store = SearchResultStore()
            current_identity = store.dataset_info()
            worker_identity = getattr(sender, 'dataset_identity', None)
            if sender is not None:
                if not isinstance(worker_identity, dict):
                    sender._bridge_result_rejected = True
                    self.searchStatus.emit(
                        '❌ 검색 결과의 데이터셋 검증 정보가 없습니다.'
                    )
                    return
                if worker_identity != current_identity:
                    sender._bridge_result_rejected = True
                    self.searchStatus.emit(
                        '❌ 검색 중 데이터셋이 변경되어 결과를 폐기했습니다.'
                    )
                    return
            else:
                worker_identity = current_identity
            if sender is not None:
                sender._bridge_result_rejected = False
            snapshot_id = _uuid.uuid4().hex
            out = []

            def _pick(row, primary, alt):
                """tag_string_X 우선, 없으면 X — 양쪽 parquet 스키마 호환"""
                v = row.get(primary)
                if v is None or v == '':
                    v = row.get(alt, '')
                return str(v) if v is not None else ''

            def _dim(row, key):
                """image_width/height → int 또는 None (없으면 자동 해상도 폴백)"""
                v = row.get(key)
                try:
                    if v is None or v == '':
                        return None
                    iv = int(float(v))
                    return iv if iv > 0 else None
                except (ValueError, TypeError):
                    return None

            if isinstance(results, list):
                # 현재 워커 결과는 이 시점 이후 다른 소비자가 없으므로 기존 dict를
                # 정규화해 재사용한다. 수십만 행에서 동일 크기의 두 번째 list[dict]가
                # 동시에 존재하던 피크 메모리를 제거한다.
                out = results
                write_idx = 0
                for row in out:
                    if not isinstance(row, dict):
                        continue
                    copyright = _pick(row, 'tag_string_copyright', 'copyright')
                    character = _pick(row, 'tag_string_character', 'character')
                    artist = _pick(row, 'tag_string_artist', 'artist')
                    general = _pick(row, 'tag_string_general', 'general')
                    rating = str(row.get('rating') or '')
                    image_width = _dim(row, 'image_width')
                    image_height = _dim(row, 'image_height')
                    row.clear()
                    row['copyright'] = copyright
                    row['character'] = character
                    row['artist'] = artist
                    row['general'] = general
                    row['rating'] = rating
                    row['image_width'] = image_width
                    row['image_height'] = image_height
                    out[write_idx] = row
                    write_idx += 1
                if write_idx < len(out):
                    del out[write_idx:]
            elif hasattr(results, 'iterrows'):
                # 옛 형식 fallback: DataFrame
                for _, row in results.iterrows():
                    out.append({
                        'copyright': _pick(row, 'tag_string_copyright', 'copyright'),
                        'character': _pick(row, 'tag_string_character', 'character'),
                        'artist':    _pick(row, 'tag_string_artist',    'artist'),
                        'general':   _pick(row, 'tag_string_general',   'general'),
                        'rating':    str(row.get('rating') or ''),
                        'image_width':  _dim(row, 'image_width'),
                        'image_height': _dim(row, 'image_height'),
                    })
            else:
                print(f"[Search] _on_search_results: unknown type {type(results)}")

            print(f"[Search] _on_search_results: built {len(out):,} dicts from {type(results).__name__}")

            # 큰 결과셋 안전장치 — Vue에 너무 많이 보내면 JSON 직렬화/전송이 느려짐
            # 사용자가 "무제한" 토글로 끌 수 있음 (disable_result_cap)
            MAX_RESULTS_TO_VUE = 500_000
            disable_cap = getattr(self, '_disable_result_cap', False)
            if disable_cap:
                print(f"[Search] cap DISABLED — emitting all {len(out):,} rows (UI 느려질 수 있음)")
            elif len(out) > MAX_RESULTS_TO_VUE:
                print(f"[Search] capping {len(out):,} → {MAX_RESULTS_TO_VUE:,} (UI 부하 방지)")
                # 무작위 샘플링 (앞부분만 보내면 편향됨)
                _rnd.shuffle(out)
                out = out[:MAX_RESULTS_TO_VUE]

            # JSON 직렬화까지 끝낸 뒤 manifest identity를 다시 확인한다. 큰 결과는
            # 직렬화 자체도 오래 걸릴 수 있으므로 이 검증보다 앞에 두어야 한다.
            result_json = json.dumps(
                out, ensure_ascii=False, separators=(',', ':')
            )
            publish_identity = store.dataset_info()
            if worker_identity != publish_identity:
                if sender is not None:
                    sender._bridge_result_rejected = True
                self.searchStatus.emit(
                    '❌ 검색 결과 게시 전 데이터셋이 변경되어 결과를 폐기했습니다.'
                )
                return

            # 기존 searchResultsReady 배열 계약은 유지하고, 그 배열에 결합된
            # provenance를 additive signal로 먼저 전달한다.
            self.searchResultLineage.emit(
                self._search_result_lineage_json(worker_identity, snapshot_id)
            )
            self.searchResultsReady.emit(result_json)
            self.searchStatus.emit(f'{len(out):,}개 결과 (전체 {total_count:,}개)')

            # Python 메인 윈도우의 filtered_results도 업데이트 (랜덤 프롬프트용)
            main_win = self.parent()
            if main_win and hasattr(main_win, 'filtered_results'):
                main_win._search_dataset_identity = worker_identity
                main_win._search_snapshot_id = snapshot_id
                main_win.filtered_results = out
                main_win.shuffled_prompt_deck = out.copy()
                _rnd.shuffle(main_win.shuffled_prompt_deck)
                # 새 검색 → 덱 진행도 초기화 저장 (옛 진행도 덮어쓰기)
                if hasattr(main_win, '_save_deck_state'):
                    main_win._save_deck_state()

            # ── 디스크 영속(단일 쓰기 경로): 재시작 시 자동 복원 → 자동화 즉시 사용 ──
            #   새 검색이므로 active=full 동일(전체). manifest provenance를 검증하는
            #   backend cache만 영속 소스로 사용한다.
            #   쓰기 로직은 generator_main._persist_search_results 한 곳으로 통일(드리프트 방지).
            if main_win and hasattr(main_win, '_persist_search_results'):
                main_win._persist_search_results(
                    out,
                    full=out,
                    dataset_identity=worker_identity,
                    snapshot_id=snapshot_id,
                )
                print(f"[Search] saved {len(out):,} rows to disk (single write path)")
            else:
                # 폴백: 메인 윈도우 없음(개발/단독) — 직접 기록
                try:
                    store.save(
                        out,
                        full=out,
                        snapshot_id=snapshot_id,
                        expected_identity=worker_identity,
                    )
                except Exception as e:
                    print(f"[Search] disk backup failed: {e}")
        except Exception as e:
            if sender is not None:
                sender._bridge_result_rejected = True
            self.searchStatus.emit(f'❌ 검색 결과 처리 실패: {e}')
            self.searchResultsReady.emit(json.dumps({'error': str(e)}))

    @pyqtSlot(result=str)
    def getActiveSearchDataset(self) -> str:
        """현재 Search 경로의 manifest label/fingerprint를 반환한다."""
        try:
            from core.search_result_store import SearchResultStore
            return json.dumps(SearchResultStore().dataset_info(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @pyqtSlot(result=str)
    def loadLastSearchResults(self) -> str:
        """디스크에서 마지막 검색 결과 로드 (Vue가 onMounted에서 호출)
        Returns: JSON string of list[dict] (빈 경우 '[]')
        """
        try:
            from core.search_result_store import SearchResultStore
            store = SearchResultStore()
            parsed = store.load_active()
            if store.last_error:
                print(f"[Search] cache ignored: {store.last_error}")
            main_win = self.parent()
            if store.last_snapshot_id is not None:
                self.searchResultLineage.emit(
                    self._search_result_lineage_json(
                        store.last_dataset_identity,
                        store.last_snapshot_id,
                    )
                )
            if (store.last_snapshot_id is not None and main_win
                    and hasattr(main_win, 'filtered_results')):
                import random as _rnd
                main_win._search_snapshot_id = store.last_snapshot_id
                main_win._search_dataset_identity = store.last_dataset_identity
                main_win.filtered_results = parsed
                # 저장된 덱 진행도 복원 ('얼마나 뽑았는지' 유지).
                # 실패(파일 없음/풀 크기 변경)면 전체 셔플로 폴백.
                if not (hasattr(main_win, '_restore_deck_state')
                        and main_win._restore_deck_state()):
                    main_win.shuffled_prompt_deck = parsed.copy()
                    _rnd.shuffle(main_win.shuffled_prompt_deck)
                print(f"[Search] restored {len(parsed):,} rows from disk → filtered_results")
            return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            print(f"[Search] loadLastSearchResults failed: {e}")
        return '[]'

    @pyqtSlot(result=str)
    def loadFullResults(self) -> str:
        """필터 적용 '전' 전체 검색 셋을 디스크에서 로드 (Vue가 '필터 해제' 베이스로 사용).
        full cache 우선, 없으면 active cache로 폴백.
        (loadLastSearchResults와 달리 Python 덱/filtered_results는 건드리지 않음 —
        순수 조회.)"""
        try:
            from core.search_result_store import SearchResultStore
            store = SearchResultStore()
            parsed = store.load_full()
            if not parsed:
                parsed = store.load_active()
            if store.last_error:
                print(f"[Search] full cache ignored: {store.last_error}")
            return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            print(f"[Search] loadFullResults failed: {e}")
        return '[]'

    @pyqtSlot(result=str)
    def getUiPrefs(self) -> str:
        """ui_prefs.json 전체를 JSON 문자열로 반환 — Vue가 mount 시 능동 복원용.

        uiPrefsLoaded 이벤트는 앱 startup에 1회만 emit되므로, 라우터+keep-alive로
        늦게 mount되는 SearchView가 그 이벤트를 놓칠 수 있음. 또 QWebEngine 저장소
        경로가 PID 기반이라 재시작 시 localStorage가 비워지는데, ui_prefs.json은
        파일이라 재시작 후에도 남음 → 이 getter로 능동 복원하면 검색 입력이 보존됨.
        """
        try:
            import os
            from core.config_migration import load_ui_prefs
            prefs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'ui_prefs.json')
            return json.dumps(load_ui_prefs(prefs_path), ensure_ascii=False)
        except Exception as e:
            print(f"[UIPrefs] getUiPrefs failed: {e}")
            return '{}'

    @pyqtSlot(result=str)
    def getAiAssistInstructions(self) -> str:
        """AI helper instructions only; chat settings use their separate store."""
        try:
            from core.ai_assist_instructions import load_instructions
            return json.dumps({'ok': True, 'instructions': load_instructions()}, ensure_ascii=False)
        except Exception as exc:
            from core.error_handler import handle_error
            try:
                handle_error('E030', 'AI 어시스트 지침 불러오기', exc, notify=False)
            except Exception:
                pass  # A closed/legacy-encoded console must not swallow the reply.
            return json.dumps({'ok': False, 'error': 'AI 어시스트 지침을 불러오지 못했습니다.'}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def saveAiAssistInstructions(self, payload_json: str) -> str:
        """Acknowledge only after validation and an atomic settings-file save."""
        try:
            from core.ai_assist_instructions import save_instructions
            # Bound parsing as well as individual fields, including JSON escapes.
            if len(payload_json) > 600_000:
                raise ValueError('지침 데이터가 너무 큽니다. 각 입력란을 8,000자 이내로 작성하세요.')
            instructions = save_instructions(json.loads(payload_json))
            self.aiAssistInstructionsChanged.emit(json.dumps({'ok': True, 'instructions': instructions}, ensure_ascii=False))
            return json.dumps({'ok': True, 'instructions': instructions}, ensure_ascii=False)
        except (ValueError, TypeError):
            return json.dumps({'ok': False, 'error': '지침 형식이 올바르지 않습니다. 각 입력란을 8,000자 이내로 작성하세요.'}, ensure_ascii=False)
        except Exception as exc:
            from core.error_handler import handle_error
            try:
                handle_error('E030', 'AI 어시스트 지침 저장', exc, notify=False)
            except Exception:
                pass  # Still deliver a usable error if console logging itself fails.
            return json.dumps({'ok': False, 'error': 'AI 어시스트 지침을 저장하지 못했습니다. 설정 파일과 쓰기 권한을 확인하세요.'}, ensure_ascii=False)

    def _instruction_presets_request(self, operation, payload):
        try:
            from core.instruction_presets import list_presets, save_preset, delete_preset
            if not isinstance(payload, str) or len(payload) > 600_000:
                raise ValueError('프리셋 데이터가 너무 큽니다')
            data = {'scope': payload} if operation == 'list' else json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError('프리셋 요청 형식이 올바르지 않습니다')
            scope = data.get('scope')
            item = None
            if operation == 'save':
                item = save_preset(scope, data.get('name'), data.get('instructions'), preset_id=data.get('id'))
            elif operation == 'delete':
                delete_preset(scope, data.get('id'))
            reply = json.dumps({'ok': True, 'scope': scope, 'presets': list_presets(scope), 'preset': item}, ensure_ascii=False)
            if operation != 'list':
                self.instructionPresetsChanged.emit(reply)
            return reply
        except (ValueError, TypeError) as exc:
            return json.dumps({'ok': False, 'error': str(exc)[:500]}, ensure_ascii=False)
        except Exception as exc:
            from core.error_handler import handle_error
            try:
                handle_error('E030', '지침 프리셋 저장소', exc, notify=False)
            except Exception:
                pass
            return json.dumps({'ok': False, 'error': '프리셋을 처리하지 못했습니다. 파일과 쓰기 권한을 확인하세요.'}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def getInstructionPresets(self, scope):
        return self._instruction_presets_request('list', scope)

    @pyqtSlot(str, result=str)
    def saveInstructionPreset(self, payload):
        return self._instruction_presets_request('save', payload)

    @pyqtSlot(str, result=str)
    def deleteInstructionPreset(self, payload):
        return self._instruction_presets_request('delete', payload)

    def _refresh_forge_module_widgets(self) -> None:
        """저장된 Forge VAE/TE 경로를 현재 프록시 목록에 즉시 반영."""
        try:
            from core.forge_modules import list_te_files, list_vae_files

            vae_files = list_vae_files()
            vae_proxy = self._proxies.get('vae_main_combo')
            if vae_proxy is not None and hasattr(vae_proxy, 'addItems'):
                current = vae_proxy.currentText() if hasattr(vae_proxy, 'currentText') else ''
                merged = ['Use checkpoint default']
                # Settings 경로를 다시 스캔하는 흐름에서는 빈 폴더도 의도된 결과다.
                # 이전 로컬/API 목록을 섞으면 제거된 모듈이 계속 전송될 수 있으므로 교체한다.
                for name in vae_files:
                    if name and name not in ('Use same VAE', 'Use checkpoint default') and name not in merged:
                        merged.append(name)
                vae_proxy.clear()
                vae_proxy.addItems(merged)
                selected = current if current in merged else merged[0]
                if hasattr(vae_proxy, 'setCurrentText'):
                    vae_proxy.setCurrentText(selected)
                # addItems()가 index 0을 자동 선택해도 값 signal은 보내지 않으므로
                # 제거된 이전 VAE가 Vue 상태에 남지 않게 선택값을 명시 동기화한다.
                self.pushWidgetValue('vae_main_combo', selected)

            te_files = list_te_files()
            self.pushWidgetProperty('te_main_input', 'items', te_files)
            te_proxy = self._proxies.get('te_main_input')
            if te_proxy is not None and hasattr(te_proxy, 'text') and hasattr(te_proxy, 'setText'):
                current_te = [
                    item.strip() for item in (te_proxy.text() or '').split(',')
                    if item.strip()
                ]
                available = set(te_files)
                valid_te = [item for item in current_te if item in available]
                if valid_te != current_te:
                    te_proxy.setText(', '.join(valid_te))

            # 다음 LoRA Manager 열기/새로고침 때 Forge API 목록을 다시 받는다.
            from widgets.lora_manager import LoraManagerDialog
            LoraManagerDialog._lora_cache = []
            self._merged_lora_cache = None
        except Exception as exc:
            logger.warning("Forge module widget refresh failed: %s", exc)

    @staticmethod
    def _forge_model_paths_web_denial() -> str:
        return json.dumps({
            'ok': False,
            'error': '웹 모드에서는 로컬 Forge 모델 경로에 접근할 수 없습니다.',
        }, ensure_ascii=False)

    @pyqtSlot(result=str)
    def getForgeModelPaths(self) -> str:
        """Forge 체크포인트/LoRA/VAE/TE 디렉터리와 스캔 상태 반환."""
        if self._backend_runtime_is_web_mode():
            return self._forge_model_paths_web_denial()
        try:
            from core.forge_modules import get_forge_path_state
            return json.dumps({'ok': True, **get_forge_path_state()}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def selectForgeModelDirectory(self, key: str) -> str:
        """Settings의 BROWSE 버튼용 네이티브 폴더 선택기."""
        if self._backend_runtime_is_web_mode():
            return self._forge_model_paths_web_denial()
        try:
            from ui.native_dialogs import select_directory
            from core.forge_modules import FORGE_PATH_KEYS, get_forge_paths

            if key not in FORGE_PATH_KEYS:
                raise ValueError(f'지원하지 않는 Forge 경로 키: {key}')
            current = get_forge_paths()[key]
            selected = select_directory(
                self.parent(),
                'Forge Neo 모델 폴더 선택',
                str(current),
            )
            if not selected:
                return json.dumps({'ok': False, 'cancelled': True}, ensure_ascii=False)
            return json.dumps({'ok': True, 'key': key, 'path': selected}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def saveForgeModelPaths(self, payload_json: str) -> str:
        """네 Forge 모델 디렉터리를 전체 검증 후 원자적으로 저장."""
        if self._backend_runtime_is_web_mode():
            return self._forge_model_paths_web_denial()
        try:
            from core.forge_modules import get_forge_path_state, save_forge_paths

            payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
            save_forge_paths(payload)
            self._refresh_forge_module_widgets()
            self.showNotification.emit('success', 'Forge Neo 모델 경로를 저장했습니다')
            return json.dumps({'ok': True, **get_forge_path_state()}, ensure_ascii=False)
        except Exception as exc:
            errors = getattr(exc, 'errors', None)
            message = str(exc)
            self.showNotification.emit('error', f'Forge 경로 저장 실패: {message}')
            return json.dumps(
                {'ok': False, 'error': message, 'errors': errors or {}},
                ensure_ascii=False,
            )

    @pyqtSlot(result=str)
    def resetForgeModelPaths(self) -> str:
        """사용자 지정값을 지우고 자동 감지/default 경로로 복귀."""
        if self._backend_runtime_is_web_mode():
            return self._forge_model_paths_web_denial()
        try:
            from core.forge_modules import get_forge_path_state, reset_forge_paths

            reset_forge_paths()
            self._refresh_forge_module_widgets()
            self.showNotification.emit('success', 'Forge Neo 경로를 자동 감지 기본값으로 되돌렸습니다')
            return json.dumps({'ok': True, **get_forge_path_state()}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)

    @pyqtSlot(result=str)
    def refreshForgeModelPaths(self) -> str:
        """저장된 경로를 다시 스캔하고 VAE/TE 목록을 갱신."""
        if self._backend_runtime_is_web_mode():
            return self._forge_model_paths_web_denial()
        try:
            from core.forge_modules import get_forge_path_state

            self._refresh_forge_module_widgets()
            return json.dumps({'ok': True, **get_forge_path_state()}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def loadImageBase64(self, filepath: str) -> str:
        """이미지를 base64로 반환"""
        import base64, os
        clean = _normalize_vue_path(filepath)
        if not clean:
            return ''
        try:
            # base64는 원본보다 약 33% 커지므로 비정상적인 대용량 입력은 거절한다.
            if os.path.getsize(clean) > 64 * 1024 * 1024:
                logger.warning("loadImageBase64 blocked oversized image: %s", clean)
                return ''
        except OSError:
            return ''
        with open(clean, 'rb') as f:
            data = f.read()
        ext = os.path.splitext(clean)[1].lower()
        mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}.get(ext, 'image/png')
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    def _load_upscalers_json(self) -> str:
        try:
            from backends import get_backend
            backend = get_backend()
            if backend:
                import requests
                r = requests.get(f"{backend.api_url}/sdapi/v1/upscalers", timeout=5)
                if r.status_code == 200:
                    return json.dumps([u['name'] for u in r.json()])
        except Exception:
            pass
        return json.dumps([])

    @pyqtSlot(result=str)
    def getUpscalers(self) -> str:
        """하위호환 동기 API. 신규 Vue 코드는 requestUpscalers를 사용한다."""
        return self._load_upscalers_json()

    @pyqtSlot()
    def requestUpscalers(self):
        """업스케일러 목록을 백그라운드에서 조회한다."""
        self._run_async_lookup('upscalers', self._load_upscalers_json, self.upscalersReady)

    @pyqtSlot(str, str, result=str)
    def saveImageExif(self, filepath: str, new_params: str) -> str:
        """이미지의 PNG 메타데이터(parameters)를 수정하여 저장"""
        tmp_path = ''
        try:
            import os, tempfile
            from PIL import Image as PILImage
            from PIL.PngImagePlugin import PngInfo
            clean = _normalize_vue_path(filepath)
            if not clean:
                return json.dumps({'error': '파일을 찾을 수 없습니다'})
            if not clean.lower().endswith('.png'):
                return json.dumps({'error': 'PNG 파일만 메타데이터 수정 가능'})
            with PILImage.open(clean) as img:
                meta = PngInfo()
                meta.add_text("parameters", new_params)
                # 기존 메타데이터 중 parameters 외 보존
                for k, v in img.info.items():
                    if k != "parameters" and isinstance(v, str):
                        meta.add_text(k, v)
                fd, tmp_path = tempfile.mkstemp(prefix='.exif_', suffix='.png', dir=os.path.dirname(clean))
                os.close(fd)
                img.save(tmp_path, pnginfo=meta)
            os.replace(tmp_path, clean)
            return json.dumps({'ok': True})
        except Exception as e:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, str, result=str)
    def renameFile(self, filepath: str, new_name: str) -> str:
        """파일 이름 변경"""
        try:
            import os
            from core.file_naming import sanitize_filename
            clean = _normalize_vue_path(filepath)
            if not clean:
                return json.dumps({'error': '파일을 찾을 수 없습니다'})
            dir_path = os.path.dirname(clean)
            ext = os.path.splitext(clean)[1]
            # 구분자 제거 — 새 이름은 같은 디렉토리 안의 단일 파일명만 허용
            stem, new_ext = os.path.splitext(new_name)
            new_name = sanitize_filename(stem, fallback='renamed', max_len=128) + (new_ext or '')
            if not new_name.endswith(ext):
                new_name += ext
            new_path = os.path.join(dir_path, new_name)
            if os.path.normcase(new_path) != os.path.normcase(clean) and os.path.exists(new_path):
                return json.dumps({'error': '같은 이름의 파일이 이미 존재합니다'})
            os.rename(clean, new_path)
            return json.dumps({'ok': True, 'new_path': new_path.replace('\\', '/')})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, int, int, result=str)
    def getEdgeMap(self, image_path: str, canny_low: int, canny_high: int) -> str:
        """Canny edge detection → base64 PNG (자석 올가미용)"""
        try:
            import cv2, base64
            from io import BytesIO
            from PIL import Image as PILImage
            clean = _normalize_vue_path(image_path)
            if not clean:
                return ''
            img = cv2.imread(clean)
            if img is None:
                return ''
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, canny_low, canny_high)
            pil = PILImage.fromarray(edges)
            buf = BytesIO()
            pil.save(buf, format='PNG')
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception as e:
            logger.warning("getEdgeMap failed: %s", e)
            return ''

    @pyqtSlot(str, str, str)
    def ollamaEnhance(self, tags: str, mode: str, extra_json: str):
        """Ollama로 태그 강화 (비동기)"""
        try:
            # 이전 worker가 실행 중이면 정리
            if hasattr(self, '_ollama_worker') and self._ollama_worker and self._ollama_worker.isRunning():
                self._ollama_worker.disconnect()
                self._ollama_worker.quit()
                self._ollama_worker.wait(1000)
            extra = json.loads(extra_json) if extra_json else {}
            from workers.ollama_worker import OllamaWorker
            url = extra.get('url', 'http://localhost:11434')
            model = (extra.get('model') or '').strip()
            # 모델 검증 — 설치 목록과 대조해 없으면 첫 설치 모델로 대체.
            # (저장된 기본값 gemma3:4b 미설치, :latest 유무 등으로 모델 못 불러오던 문제 방지)
            try:
                from core.ollama_client import OllamaClient
                installed = OllamaClient(base_url=url).list_models()
                if installed:
                    def _b(s): return (s or '').split(':')[0].lower()
                    if not (model and any(m == model or _b(m) == _b(model) for m in installed)):
                        model = installed[0]
            except Exception:
                pass
            if not model:
                model = 'gemma3:4b'
            extra_prompt = extra.get('prompt', '')
            if mode == 'creative':
                # 창의 모드: 캐릭터의 실제 외견 태그를 DB에서 조회해 입력에 포함
                tags, extra_prompt = self._build_creative_input(tags, extra.get('character', ''))
            self._ollama_worker = OllamaWorker(url, model, tags, mode, extra_prompt, self)
            self._ollama_worker.finished.connect(lambda r: self.ollamaResult.emit(r))
            self._ollama_worker.error.connect(lambda e: self.ollamaResult.emit(json.dumps({'error': e})))
            self._ollama_worker.start()
        except Exception as e:
            self.ollamaResult.emit(json.dumps({'error': str(e)}))

    @pyqtSlot(str, str)
    def convertPromptToNl(self, text: str, extra_json: str):
        """생성 시 태그→자연어(nl_caption) 변환 — 전용 시그널 genNlResult로 결과 전달.
        PromptPanel의 ollamaResult 리스너와 충돌하지 않도록 별도 채널을 사용한다."""
        try:
            if hasattr(self, '_gennl_worker') and self._gennl_worker and self._gennl_worker.isRunning():
                self._gennl_worker.disconnect()
                self._gennl_worker.quit()
                self._gennl_worker.wait(1000)
            extra = json.loads(extra_json) if extra_json else {}
            from workers.ollama_worker import OllamaWorker
            url = extra.get('url', 'http://localhost:11434')
            model = (extra.get('model') or '').strip()
            # 모델 검증 (ollamaEnhance와 동일 — 미설치/별칭 문제 방지)
            try:
                from core.ollama_client import OllamaClient
                installed = OllamaClient(base_url=url).list_models()
                if installed:
                    def _b(s): return (s or '').split(':')[0].lower()
                    if not (model and any(m == model or _b(m) == _b(model) for m in installed)):
                        model = installed[0]
            except Exception:
                pass
            if not model:
                model = 'gemma3:4b'
            self._gennl_worker = OllamaWorker(
                url, model, text, 'nl_caption', '', self, instruction_feature='auto_nl',
            )
            self._gennl_worker.finished.connect(lambda r: self.genNlResult.emit(r))
            self._gennl_worker.error.connect(lambda e: self.genNlResult.emit(json.dumps({'error': e})))
            self._gennl_worker.start()
        except Exception as e:
            self.genNlResult.emit(json.dumps({'error': str(e)}))

    def _build_creative_input(self, hints: str, character: str):
        """창의 모드 입력 구성 — 캐릭터의 실제 외견 핵심 태그(우리 DB)를 함께 전달.
        태그 파일 전체를 LLM에 먹이는 대신, 해당 캐릭터의 검증된 외견 태그만 O(1) 조회해 주입."""
        character = (character or '').strip()
        hints = (hints or '').strip()
        feat = ''
        if character:
            first = character.split(',')[0].strip()
            # 1) 사용자가 저장한 프리셋(수정본) 우선 — danbooru로 고친 캐릭터 반영
            try:
                from utils.character_presets import get_character_preset_full
                preset = get_character_preset_full(first)
                if preset and (preset.get('extra_prompt') or '').strip():
                    feat = preset['extra_prompt'].strip()
            except Exception:
                pass
            # 2) 없으면 로컬 DB 핵심 특징
            if not feat:
                try:
                    from utils.character_features import get_character_features
                    core = get_character_features().lookup_core(first)
                    if core and core[0]:
                        feat = core[0]
                except Exception:
                    pass
        parts = []
        if character:
            parts.append(f"Character: {character}")
        if feat:
            parts.append(f"Canonical appearance tags (keep these accurate): {feat}")
        if hints:
            parts.append(f"Extra theme / hints (highest priority): {hints}")
        if not parts:
            parts.append("No specific character given — invent a fresh original anime character.")
        return "\n".join(parts), ''

    @pyqtSlot(str, str, result=str)
    def editorPasteImage(self, b64_data: str, mime_type: str) -> str:
        """클립보드 이미지를 임시 파일로 저장하고 경로 반환.

        Vue에서 navigator.clipboard.read()로 받은 base64 이미지 받음.
        """
        try:
            import base64
            import tempfile
            from pathlib import Path
            # mime_type 예: 'image/png', 'image/jpeg', ...
            ext = mime_type.split('/')[-1] if '/' in mime_type else 'png'
            if ext == 'jpeg':
                ext = 'jpg'
            raw = base64.b64decode(b64_data)
            # tempdir 경로 (앱 캐시 폴더 안에)
            tmp_dir = Path(tempfile.gettempdir()) / "AIStudioPro_editor"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            import time
            tmp_path = tmp_dir / f"clipboard_{int(time.time())}.{ext}"
            tmp_path.write_bytes(raw)
            return json.dumps({"path": str(tmp_path).replace('\\', '/')})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def editorAutoSave(self, path: str) -> str:
        """현재 편집 중인 파일을 임시 위치에 복사 → 크래시 복구용."""
        try:
            import shutil
            import time
            from pathlib import Path
            import tempfile
            clean = _normalize_vue_path(path)
            if not clean:
                return json.dumps({})
            src = Path(clean)
            tmp_dir = Path(tempfile.gettempdir()) / "AIStudioPro_editor"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            # 단일 복구 파일 (덮어쓰기)
            dst = tmp_dir / "_autosave_session.png"
            shutil.copy2(src, dst)
            # 원본 경로 메타 같이 저장
            meta = tmp_dir / "_autosave_session.meta.json"
            meta.write_text(
                json.dumps({"original": str(src), "saved_at": int(time.time())}),
                encoding="utf-8",
            )
            return json.dumps({"path": str(dst).replace('\\', '/')})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(result=str)
    def editorCheckAutoSave(self) -> str:
        """앱 시작 시 호출 — 미저장 복구본 있는지 확인."""
        try:
            import time
            from pathlib import Path
            import tempfile
            tmp_dir = Path(tempfile.gettempdir()) / "AIStudioPro_editor"
            dst = tmp_dir / "_autosave_session.png"
            meta = tmp_dir / "_autosave_session.meta.json"
            if not dst.is_file():
                return json.dumps({"exists": False})
            saved_at = 0
            original = ""
            if meta.is_file():
                try:
                    m = json.loads(meta.read_text(encoding="utf-8"))
                    saved_at = int(m.get("saved_at", 0))
                    original = m.get("original", "")
                except Exception:
                    pass
            age_min = max(1, int((time.time() - saved_at) // 60)) if saved_at else 0
            # 24시간 초과면 무시 (오래된 복구본은 자동 정리 권장)
            if saved_at and (time.time() - saved_at) > 86400:
                try:
                    dst.unlink()
                    if meta.exists():
                        meta.unlink()
                except OSError:
                    pass
                return json.dumps({"exists": False})
            return json.dumps({
                "exists": True,
                "path": str(dst).replace('\\', '/'),
                "basename": Path(original).name if original else dst.name,
                "age_minutes": age_min,
            })
        except Exception:
            return json.dumps({"exists": False})

    @pyqtSlot(result=str)
    def editorClearAutoSave(self) -> str:
        """복구본 폐기."""
        try:
            from pathlib import Path
            import tempfile
            tmp_dir = Path(tempfile.gettempdir()) / "AIStudioPro_editor"
            for fn in ("_autosave_session.png", "_autosave_session.meta.json"):
                p = tmp_dir / fn
                if p.exists():
                    p.unlink()
            return json.dumps({"cleared": True})
        except Exception:
            return json.dumps({"cleared": False})

    @pyqtSlot(str, result=str)
    def getFileInfo(self, path: str) -> str:
        """파일 기본 정보 반환 (포맷/용량)."""
        try:
            from pathlib import Path
            clean = _normalize_vue_path(path)
            if not clean:
                return json.dumps({})
            p = Path(clean)
            stat = p.stat()
            ext = p.suffix.lstrip('.').upper() or ''
            return json.dumps({"size": stat.st_size, "format": ext})
        except Exception:
            return json.dumps({})

    def _load_ollama_models_json(self, base_url: str = '') -> str:
        try:
            from core.ollama_client import OllamaClient
            url = base_url.strip() if base_url.strip() else 'http://localhost:11434'
            client = OllamaClient(base_url=url)
            models = client.list_models()
            return json.dumps(models)
        except Exception as e:
            print(f"[Ollama] ollamaListModels 오류: {e}")
            return json.dumps([])

    @pyqtSlot(str, result=str)
    def ollamaListModels(self, base_url: str = '') -> str:
        """하위호환 동기 API. 신규 Vue 코드는 requestOllamaModels를 사용한다."""
        return self._load_ollama_models_json(base_url)

    @pyqtSlot(str)
    def requestOllamaModels(self, base_url: str = ''):
        """Ollama 모델 목록을 백그라운드에서 조회한다."""
        url = base_url.strip() if base_url.strip() else 'http://localhost:11434'

        def _load():
            return json.dumps({
                'url': url,
                'models': json.loads(self._load_ollama_models_json(url)),
            })

        self._run_async_lookup(f'ollama-models:{url}', _load, self.ollamaModelsReady)

    @pyqtSlot(result=str)
    def getRandomResolutions(self) -> str:
        """랜덤 해상도 목록 반환"""
        try:
            gen = self.parent()
            if gen and hasattr(gen, 'random_resolutions'):
                return json.dumps(gen.random_resolutions)
        except Exception:
            pass
        return json.dumps([])

    @pyqtSlot(result=str)
    def getInitialConfig(self) -> str:
        """클라이언트별 초기 설정 응답.

        QWebChannel 시그널 재발행은 연결된 모든 브라우저에 방송되므로, 새 웹
        클라이언트 하나가 기존 클라이언트 상태까지 다시 덮어쓰지 않게 직접 반환한다.
        """
        root = os.path.dirname(os.path.dirname(__file__))

        def _load(name, default):
            try:
                path = os.path.join(root, 'config', name)
                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning("initial config load failed (%s): %s", name, e)
            return default

        try:
            ui_prefs = json.loads(self.getUiPrefs() or '{}')
        except Exception:
            ui_prefs = {}
        try:
            tab_defaults = json.loads(self.getTabDefaults() or '{}')
        except Exception:
            tab_defaults = {}
        return json.dumps({
            'uiPrefs': ui_prefs,
            'condRules': _load('cond_rules.json', {'positive': [], 'negative': []}),
            'globalWeights': _load('global_weights.json', []),
            'tabDefaults': tab_defaults,
        }, ensure_ascii=False)

    @pyqtSlot(result=str)
    def requestInitialConfig(self) -> str:
        """구버전 프론트 호환 별칭 — 더 이상 전역 시그널을 재발행하지 않는다."""
        return self.getInitialConfig()

    @pyqtSlot(result=str)
    def getGenStats(self) -> str:
        """생성 통계 요약 반환"""
        try:
            from core.gen_stats import get_gen_stats
            return json.dumps(get_gen_stats().get_summary())
        except Exception:
            return json.dumps({'total': 0})

    @pyqtSlot(result=str)
    def getWildcardTree(self) -> str:
        """wildcards/ 디렉토리의 파일 트리 + 내용 반환"""
        import os
        wc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wildcards')
        if not os.path.isdir(wc_dir):
            return json.dumps([])
        tree = []
        for f in sorted(os.listdir(wc_dir)):
            fp = os.path.join(wc_dir, f)
            if not f.endswith('.txt') or not os.path.isfile(fp):
                continue
            try:
                with open(fp, 'r', encoding='utf-8') as fh:
                    lines = [l.strip() for l in fh if l.strip() and not l.startswith('#')]
                tree.append({'name': f.replace('.txt', ''), 'file': f, 'tags': lines})
            except Exception:
                pass
        return json.dumps(tree)

    vramUpdated = pyqtSignal(str)  # JSON {used, total, pct}

    @pyqtSlot(result=str)
    def getPresetList(self) -> str:
        """프리셋 목록 반환"""
        import os
        preset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')
        os.makedirs(preset_dir, exist_ok=True)
        files = [f.replace('.json', '') for f in sorted(os.listdir(preset_dir)) if f.endswith('.json')]
        return json.dumps(files)

    @pyqtSlot(str, result=str)
    def getPresetData(self, name: str) -> str:
        """프리셋 데이터 반환"""
        import os
        from core.file_naming import sanitize_filename
        preset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')
        # 읽기 경로도 정규화 — 저장/삭제와 동일 규칙(traversal 차단, 라운드트립 일치)
        fp = os.path.join(preset_dir, f"{sanitize_filename(name, fallback='')}.json")
        try:
            if os.path.exists(fp):
                with open(fp, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning("getPreset failed (%s): %s", name, e)
        return '{}'

    # ══════════ 캐릭터 특징 프리셋 (Vue 모달) ══════════

    def _current_prompt_norm_tags(self) -> set:
        """현재 프롬프트(main/prefix/suffix/character)의 정규화 태그 집합 (중복 표시용)."""
        gen = self.parent()
        out: set = set()
        if not gen:
            return out
        for attr in ('main_prompt_text', 'prefix_prompt_text',
                     'suffix_prompt_text', 'character_input'):
            w = getattr(gen, attr, None)
            if w is None:
                continue
            if hasattr(w, 'toPlainText'):
                src = w.toPlainText()
            elif hasattr(w, 'text'):
                src = w.text()
            else:
                continue
            for t in src.split(","):
                n = t.strip().lower().replace("_", " ")
                if n:
                    out.add(n)
                    out.add(n.replace(r"\(", "(").replace(r"\)", ")"))
        return out

    @pyqtSlot(str, result=str)
    def searchCharacters(self, query: str) -> str:
        """캐릭터 이름 검색 → JSON [{key, count, hasPreset}] (2글자 미만은 빈 배열)."""
        try:
            q = (query or "").strip()
            if len(q) < 2:
                return json.dumps([])
            from utils.character_features import get_character_features
            from utils.character_presets import list_character_presets
            lookup = get_character_features()
            results = lookup.search(q, limit=80)
            saved = list_character_presets()
            out = []
            for orig_key, _features, count in results:
                norm = orig_key.strip().lower().replace("_", " ")
                out.append({
                    "key": orig_key,
                    "count": int(count or 0),
                    "hasPreset": norm in saved,
                })
            return json.dumps(out, ensure_ascii=False)
        except Exception as e:
            print(f"[CharPreset] searchCharacters 실패: {e}")
            return json.dumps([])

    @pyqtSlot(str, result=str)
    def getCharacterFeatures(self, name: str) -> str:
        """캐릭터 → 핵심/의상 특징 분리 + 저장된 커스텀/조건부 규칙.
        Returns JSON {name, count, core:[{tag,existing,costume}], costume:[...],
                      custom:[str], hasPreset, condRulesJson}
        """
        try:
            from utils.character_features import get_character_features
            from utils.character_presets import get_character_preset_full
            lookup = get_character_features()
            core = lookup.lookup_core(name)
            aux = lookup.lookup_aux(name)
            costume = lookup.lookup_costume(name)
            etc = lookup.lookup_etc(name)
            full = lookup.lookup(name)
            count = (core[1] if core else 0) or (full[1] if full else 0)

            existing = self._current_prompt_norm_tags()
            char_norm = name.strip().lower().replace("_", " ")

            from core.tag_intelligence import get_tag_intelligence
            ti = get_tag_intelligence()

            def _split(s):
                return [t.strip() for t in s.split(",") if t.strip()] if s else []

            core_tags = _split(core[0]) if core else []
            aux_tags = _split(aux[0]) if aux else []
            costume_tags = _split(costume[0]) if costume else []
            etc_tags = _split(etc[0]) if etc else []
            if not core_tags and not aux_tags and not costume_tags and not etc_tags and full:
                core_tags = _split(full[0])

            def _mk(tags, is_costume):
                items = []
                for t in tags:
                    norm = t.strip().lower().replace("_", " ")
                    if norm == char_norm:
                        continue
                    esc = norm.replace("(", r"\(").replace(")", r"\)")
                    item = {
                        "tag": t,
                        "existing": (norm in existing or esc in existing),
                        "costume": is_costume,
                    }
                    if is_costume:   # ④ 의상 부위(region) 태깅
                        r = ti.region_of(t)
                        item["region"] = r or "UNASSIGNED"
                        item["regionLabel"] = ti.region_label(r or "UNASSIGNED")
                    items.append(item)
                return items

            custom = []
            cond_json = ""
            has_preset = False
            preset = get_character_preset_full(name)
            if preset:
                has_preset = True
                cond_json = preset.get("cond_rules_json", "") or ""
                feature_norms = {
                    t.strip().lower().replace("_", " ")
                    for t in (core_tags + aux_tags + costume_tags + etc_tags)
                }
                for t in (preset.get("extra_prompt", "") or "").split(","):
                    tag = t.strip()
                    n = tag.lower().replace("_", " ")
                    if n and n not in feature_norms:
                        custom.append(tag)

            # ③ copyright(시리즈) + 자동추가 설정
            copyright_tag = ""
            auto_copy = True
            try:
                copyright_tag = ti.copyright_of(name) or ""
            except Exception:
                pass
            gen = self.parent()
            if gen is not None and hasattr(gen, "_get_ui_pref"):
                auto_copy = bool(gen._get_ui_pref("autoAddCopyright", True))

            return json.dumps({
                "name": name,
                "count": int(count or 0),
                "core": _mk(core_tags, False),
                "aux": _mk(aux_tags, False),
                "etc": _mk(etc_tags, False),
                "costume": _mk(costume_tags, True),
                "custom": custom,
                "hasPreset": has_preset,
                "condRulesJson": cond_json,
                "copyright": copyright_tag,
                "autoAddCopyright": auto_copy,
            }, ensure_ascii=False)
        except Exception as e:
            print(f"[CharPreset] getCharacterFeatures 실패: {e}")
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def getCharacterCopyright(self, name: str) -> str:
        """캐릭터 → copyright(시리즈) 태그. {copyright: str}."""
        try:
            from core.tag_intelligence import get_tag_intelligence
            return json.dumps(
                {"copyright": get_tag_intelligence().copyright_of(name) or ""},
                ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def fetchCharacterTagsOnline(self, name: str) -> str:
        """danbooru에서 캐릭터의 실제 공통 general 태그를 라이브 집계 (로컬 DB가 틀린/없는 캐릭터 보완).
        posts.json 표본의 tag_string_general 빈도 집계 → 상위 태그."""
        try:
            import requests
            from collections import Counter
            tag = (name or '').strip().lower().replace(' ', '_')
            if not tag:
                return json.dumps({"error": "캐릭터 이름 없음"})
            hdr = {"User-Agent": "UR_IV/1.0 (character tag lookup)"}
            posts = []
            for q in (f"{tag} solo", tag):
                try:
                    r = requests.get(
                        "https://danbooru.donmai.us/posts.json",
                        params={"tags": q, "limit": 100, "only": "tag_string_general"},
                        timeout=12, headers=hdr,
                    )
                    r.raise_for_status()
                    posts = r.json()
                    if isinstance(posts, list) and posts:
                        break
                except Exception:
                    continue
            if not isinstance(posts, list) or not posts:
                return json.dumps({"error": "danbooru 게시물 없음 (이름/철자 확인)"})
            cnt = Counter()
            n = 0
            for p in posts:
                g = p.get("tag_string_general") if isinstance(p, dict) else ""
                if not g:
                    continue
                n += 1
                for t in g.split():
                    cnt[t] += 1
            if not n:
                return json.dumps({"error": "태그 없음"})
            ranked = [t.replace("_", " ") for t, _c in cnt.most_common(60)]
            return json.dumps({"tags": ranked[:40], "sampled": n}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, str, result=str)
    def separateTags(self, prompt: str, categories_json: str) -> str:
        """⑤ 프롬프트를 카테고리별로 분리. categories=대상 카테고리 리스트.
        Returns {rest:'...', groups:{cat:[...]}, counts:{cat:n}}.
        (rest = 어떤 대상 카테고리에도 속하지 않은 태그 = 카테고리 제거 결과)"""
        try:
            from core.tag_intelligence import get_tag_intelligence
            cats = json.loads(categories_json) if categories_json else []
            tags = [t.strip() for t in (prompt or "").split(",") if t.strip()]
            res = get_tag_intelligence().split_by_categories(tags, cats)
            counts = {c: len(v) for c, v in res["groups"].items()}
            return json.dumps({
                "rest": ", ".join(res["rest"]),
                "groups": res["groups"],
                "counts": counts,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def pairColors(self, prompt: str) -> str:
        """② 분리된 단일 색상 단어를 바로 뒤 태그와 결합 (결합 결과가 실재 태그일 때만).
        Returns {result:'...', before:n, after:m, merged:k}."""
        try:
            from core.tag_intelligence import get_tag_intelligence
            tags = [t.strip() for t in (prompt or "").split(",") if t.strip()]
            paired = get_tag_intelligence().pair_colors(tags)
            return json.dumps({
                "result": ", ".join(paired),
                "before": len(tags),
                "after": len(paired),
                "merged": len(tags) - len(paired),
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def refineToSpecificTags(self, prompt: str) -> str:
        """덜 구체적인(상위) 태그 제거 — muscular+muscular male → muscular male,
        dress+blue dress → blue dress. Returns {result, before, after, removed:[...]}"""
        try:
            from core.tag_intelligence import get_tag_intelligence
            tags = [t.strip() for t in (prompt or "").split(",") if t.strip()]
            kept, removed = get_tag_intelligence().remove_redundant_subtags(tags)
            return json.dumps({
                "result": ", ".join(kept),
                "before": len(tags),
                "after": len(kept),
                "removed": removed,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def getLoras(self, mode: str = '') -> str:
        """활성 API 메타데이터 + 메인/보조 디스크 LoRA 카탈로그 반환.

        ``_lora_cache``에는 종전과 동일하게 백엔드의 raw 응답만 보관한다.
        출처/중복/사용 가능 여부를 합친 뷰 모델을 캐시에 덮어쓰지 않으므로
        ``mode='force'`` 이후에도 기존 PyQt LoRA 매니저 계약이 유지된다.
        """
        try:
            import hashlib

            from backends import get_backend_type
            from widgets.lora_manager import LoraManagerDialog

            loras = LoraManagerDialog._lora_cache
            backend_type = get_backend_type()
            backend_value = str(getattr(backend_type, 'value', backend_type) or '').casefold()
            active_engine = 'comfyui' if backend_value == 'comfyui' else 'forge'

            def cache_signature(items) -> str:
                try:
                    payload = json.dumps(
                        items or [], ensure_ascii=False, sort_keys=True, default=str,
                        separators=(',', ':'),
                    )
                except Exception:
                    payload = repr(items)
                return hashlib.sha256(payload.encode('utf-8', errors='replace')).hexdigest()

            raw_signature = cache_signature(loras)
            cached = getattr(self, '_merged_lora_cache', None)
            if (
                mode != 'force'
                and isinstance(cached, dict)
                and cached.get('activeEngine') == active_engine
                and cached.get('rawSignature') == raw_signature
                and isinstance(cached.get('json'), str)
            ):
                return cached['json']

            if mode == 'force' or not loras:
                from backends import get_backend
                b = get_backend()
                fresh = b.get_loras() if b else []
                # 빈 응답도 유효한 최신 상태다. 이전 캐시를 남기면 삭제된 LoRA가
                # 계속 사용 가능한 것처럼 보이므로 항상 교체한다.
                loras = list(fresh or [])
                LoraManagerDialog._lora_cache = loras
                raw_signature = cache_signature(loras)

            from core.model_inventory import get_model_inventory

            out = get_model_inventory(active_engine=active_engine).merge_loras(loras or [])
            encoded = json.dumps(out, ensure_ascii=False)
            if self is not None:
                self._merged_lora_cache = {
                    'activeEngine': active_engine,
                    'rawSignature': raw_signature,
                    'json': encoded,
                }
            return encoded
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def saveSession(self, payload_json: str) -> str:
        """세션 상태(탭/프롬프트 등)를 cache/session에 저장 (크래시 복구용).
        localStorage가 PID별로 초기화돼도 살아남도록 백엔드 파일에 보관."""
        try:
            from core.storage_paths import cache_file
            from utils.atomic_json import atomic_write_json
            path = cache_file(
                'session/session_backup.json',
                legacy_paths='config/session_backup.json',
            )
            payload = json.loads(payload_json or '{}')
            if not isinstance(payload, dict):
                raise ValueError('세션 payload는 JSON 객체여야 합니다')
            atomic_write_json(str(path), payload, indent=None)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(result=str)
    def getSession(self) -> str:
        """저장된 세션 상태 반환 (없으면 {})."""
        try:
            from core.storage_paths import cache_file
            path = cache_file(
                'session/session_backup.json',
                legacy_paths='config/session_backup.json',
            )
            if path.exists():
                with path.open(encoding='utf-8') as f:
                    return f.read() or '{}'
            return json.dumps({})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def getClothingRegions(self, tags_json: str) -> str:
        """④ 의류 태그를 부위(region)별로 그룹화 → [{region, label, tags:[...]}]."""
        try:
            from core.tag_intelligence import get_tag_intelligence
            tags = json.loads(tags_json) if tags_json else []
            groups = get_tag_intelligence().group_by_region(tags)
            return json.dumps({"groups": groups}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, str, str, result=str)
    def saveCharacterPreset(self, name: str, tags_json: str, cond_rules_json: str) -> str:
        """캐릭터 프리셋 저장 (선택된 태그 + 조건부 규칙 JSON)."""
        try:
            from utils.character_presets import save_character_preset
            tags = json.loads(tags_json) if tags_json else []
            combined = ", ".join(str(t).strip() for t in tags if str(t).strip())
            save_character_preset(name, combined, cond_rules_json or "")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def deleteCharacterPreset(self, name: str) -> str:
        """캐릭터 프리셋 삭제."""
        try:
            from utils.character_presets import delete_character_preset, has_preset
            if not has_preset(name):
                return json.dumps({"ok": False, "reason": "프리셋 없음"})
            delete_character_preset(name)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(result=str)
    def getCharGlobalPrefs(self) -> str:
        """캐릭터 프리셋 글로벌 설정 로드 (모든 캐릭터 공통).
        {categoryOff: [핵심/보조/의상/기타 중 OFF인 것], wordOff: [전역 제외 단어]}"""
        try:
            import os
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'config', 'char_global_prefs.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                return json.dumps({
                    "categoryOff": list(d.get("categoryOff", [])),
                    "wordOff": list(d.get("wordOff", [])),
                })
        except Exception:
            pass
        # 기본값: 기타 카테고리만 OFF
        return json.dumps({"categoryOff": ["etc"], "wordOff": []})

    @pyqtSlot(str, result=str)
    def saveCharGlobalPrefs(self, prefs_json: str) -> str:
        """캐릭터 프리셋 글로벌 설정 저장."""
        try:
            import os
            d = json.loads(prefs_json) if prefs_json else {}
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, 'char_global_prefs.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    "categoryOff": list(d.get("categoryOff", [])),
                    "wordOff": list(d.get("wordOff", [])),
                }, f, ensure_ascii=False, indent=2)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, result=str)
    def applyCharacterPreset(self, payload_json: str) -> str:
        """Vue 모달의 적용 결과를 프롬프트 위젯에 반영.
        payload: {character: str|'', tags: [str]}
        """
        try:
            gen = self.parent()
            if not gen or not hasattr(gen, '_apply_character_features_result'):
                return json.dumps({"error": "메인 윈도우 없음"})
            payload = json.loads(payload_json) if payload_json else {}
            gen._apply_character_features_result(
                (payload.get("character") or "").strip(),
                payload.get("tags", []) or [],
                add_copyright=payload.get("addCopyright"),
            )
            return json.dumps({"ok": True})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return json.dumps({"error": str(e)})

    @pyqtSlot(result=str)
    def getDeckCharacters(self) -> str:
        """현재 자동화 덱(또는 filtered_results)에 등장하는 캐릭터 정규화 집합 → JSON [str]."""
        try:
            gen = self.parent()
            deck = (getattr(gen, 'shuffled_prompt_deck', None)
                    or getattr(gen, 'filtered_results', None) or [])
            chars: set = set()
            for b in deck:
                if not isinstance(b, dict):
                    continue
                cval = b.get('character', '') or ''
                if not cval:
                    continue
                parts = cval.split(',') if ',' in cval else cval.split()
                for p in parts:
                    n = p.strip().lower().replace('_', ' ')
                    if n:
                        chars.add(n)
            return json.dumps(sorted(chars), ensure_ascii=False)
        except Exception as e:
            print(f"[CharPreset] getDeckCharacters 실패: {e}")
            return json.dumps([])

    @pyqtSlot(str, result=str)
    def submitABTest(self, payload_json: str) -> str:
        """A/B 테스트: prompt_a, prompt_b 를 같은 시드로 큐에 추가.
        payload: {prompt_a, prompt_b, negative, seed}
        """
        try:
            gen = self.parent()
            p = json.loads(payload_json) if payload_json else {}
            try:
                seed = int(p.get("seed", -1))
            except (ValueError, TypeError):
                seed = -1
            if seed <= 0:
                import random
                seed = random.randint(1, 2147483647)
            neg = p.get("negative", "") or ""
            added = 0
            for key in ("prompt_a", "prompt_b"):
                prm = (p.get(key) or "").strip()
                if not prm:
                    continue
                item = {"prompt": prm, "negative_prompt": neg, "seed": seed}
                if gen and hasattr(gen, "queue_panel"):
                    gen.queue_panel.add_single_item(item)
                    added += 1
            return json.dumps({"ok": True, "seed": seed, "added": added})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(result=str)
    def getCharFeatureOverride(self) -> str:
        """auto-remove override 설정 조회 → JSON {hair_length, eye_color}."""
        gen = self.parent()
        ov = getattr(gen, '_char_feature_override', None) or {}
        return json.dumps({
            "hair_length": bool(ov.get("hair_length")),
            "eye_color": bool(ov.get("eye_color")),
        })

    @pyqtSlot(str, result=str)
    def setCharFeatureOverride(self, payload_json: str) -> str:
        """auto-remove override 설정 저장."""
        try:
            gen = self.parent()
            p = json.loads(payload_json) if payload_json else {}
            if gen is not None:
                gen._char_feature_override = {
                    "hair_length": bool(p.get("hair_length")),
                    "eye_color": bool(p.get("eye_color")),
                }
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    def _wildcard_path(name: str) -> str:
        """와일드카드 이름 → wildcards/ 안의 안전한 .txt 경로 (탈출 차단)."""
        import os
        from core.file_naming import sanitize_filename
        wc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wildcards')
        os.makedirs(wc_dir, exist_ok=True)
        if name.endswith('.txt'):
            name = name[:-4]
        return os.path.join(wc_dir, sanitize_filename(name, fallback='wildcard') + '.txt')

    @pyqtSlot(str, str, result=str)
    def saveWildcard(self, filename: str, content: str) -> str:
        """와일드카드 파일 저장/수정"""
        try:
            with open(self._wildcard_path(filename), 'w', encoding='utf-8') as f:
                f.write(content)
            return json.dumps({'ok': True})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def deleteWildcard(self, filename: str) -> str:
        """와일드카드 파일 삭제"""
        import os
        try:
            fp = self._wildcard_path(filename)
            if os.path.exists(fp): os.remove(fp)
            return json.dumps({'ok': True})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, str, result=str)
    def renameWildcard(self, old_name: str, new_name: str) -> str:
        """와일드카드 파일 이름 변경"""
        import os
        try:
            old_fp = self._wildcard_path(old_name)
            new_fp = self._wildcard_path(new_name)
            if os.path.exists(old_fp): os.rename(old_fp, new_fp)
            return json.dumps({'ok': True})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def getExcludeMatches(self, rule: str) -> str:
        """제외 규칙에 매칭되는 태그 목록 반환 (tags_db 기반)"""
        try:
            rule = rule.strip()
            if not rule or rule.startswith('~'):
                return json.dumps([])

            # tags_db에서 모든 태그 수집
            if not hasattr(self, '_all_tags_set'):
                self._all_tags_set = set()
                from core.tag_database import TagAsset, get_tag_database
                database = get_tag_database()
                # 명시적으로 등록된 소형 태그 목록(선별 + 확장)
                for asset in (
                    TagAsset.CLOTHING_TAGS_CURATED,
                    TagAsset.CLOTHING_TAGS_EXTENDED,
                    TagAsset.APPEARANCE_TAGS_CURATED,
                    TagAsset.APPEARANCE_TAGS_EXTENDED,
                    TagAsset.COLOR_TERMS_CURATED,
                    TagAsset.COLOR_TERMS_EXTENDED,
                ):
                    try:
                        self._all_tags_set.update(
                            line.lower().replace(' ', '_')
                            for line in database.read_lines(asset)
                        )
                    except Exception as e:
                        logger.debug("tag lexicon read failed (%s): %s", asset.value, e)
                # 기존 KR_tags.parquet 첫 열 스캔이 제공하던 Search 일반 태그.
                try:
                    catalog = database.read_parquet(
                        TagAsset.KOREAN_TAG_CATALOG,
                        columns=['tag'],
                    )
                    self._all_tags_set.update(
                        str(tag).strip().lower().replace(' ', '_')
                        for tag in catalog['tag'].dropna()
                        if str(tag).strip()
                    )
                except Exception as e:
                    logger.warning("Korean tag catalog scan failed: %s", e)
                # 통합 Wiki 그룹의 실제 tag 열만 수집
                try:
                    self._all_tags_set.update(
                        tag.lower().replace(' ', '_')
                        for tag in database.all_group_tags()
                    )
                except Exception as e:
                    logger.warning("tag group scan failed: %s", e)
                # TagClassifier의 tag_to_category
                try:
                    from core.tag_classifier import TagClassifier
                    if not hasattr(self, '_tag_classifier'):
                        self._tag_classifier = TagClassifier()
                    self._all_tags_set.update(
                        tag.lower().replace(' ', '_')
                        for tag in self._tag_classifier.tag_to_category
                    )
                except Exception as e:
                    logger.debug("TagClassifier categories load failed: %s", e)
                # character/copyright/artist 사전도 추가
                try:
                    from core.tag_classifier import TagClassifier
                    if not hasattr(self, '_tag_classifier'):
                        self._tag_classifier = TagClassifier()
                    tc = self._tag_classifier
                    if hasattr(tc, 'characters'): self._all_tags_set.update(t.lower().replace(' ', '_') for t in tc.characters)
                    if hasattr(tc, 'copyrights'): self._all_tags_set.update(t.lower().replace(' ', '_') for t in tc.copyrights)
                    if hasattr(tc, 'artists'): self._all_tags_set.update(t.lower().replace(' ', '_') for t in tc.artists)
                except Exception as e:
                    logger.debug("TagClassifier name dicts load failed: %s", e)
                print(f"[Exclude] Tag DB loaded: {len(self._all_tags_set)} tags")

            # 규칙 매칭
            rule_lower = rule.lower().replace(' ', '_')
            matches = []
            if rule_lower.startswith('~'):
                matches = []
            elif rule_lower.startswith('*'):
                keyword = rule_lower[1:]
                matches = [t for t in self._all_tags_set if t == keyword]
            elif rule_lower.startswith('_') and rule_lower.endswith('_') and len(rule_lower) > 2:
                keyword = rule_lower[1:-1]
                matches = [t for t in self._all_tags_set if keyword in t]
            elif rule_lower.startswith('_'):
                keyword = rule_lower[1:]
                matches = [t for t in self._all_tags_set if t.endswith(keyword)]
            elif rule_lower.endswith('_'):
                keyword = rule_lower[:-1]
                matches = [t for t in self._all_tags_set if t.startswith(keyword)]
            else:
                matches = [t for t in self._all_tags_set if rule_lower in t]

            matches.sort()
            return json.dumps(matches)
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def deepCleanPrompt(self, prompt_json: str) -> str:
        """딥 프롬프트 클리너: 충돌 감지 + 중복 제거 + 최적 순서 재배치"""
        try:
            data = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
            tags = [t.strip() for t in data.get('prompt', '').split(',') if t.strip()]

            # 1. 중복 제거
            seen = set()
            unique = []
            for t in tags:
                tl = t.lower().replace(' ', '_')
                if tl not in seen:
                    seen.add(tl)
                    unique.append(t)

            # 2. 충돌 감지
            conflicts = []
            conflict_pairs = [
                (['black_hair', 'blonde_hair', 'brown_hair', 'red_hair', 'blue_hair', 'green_hair', 'white_hair', 'pink_hair', 'purple_hair', 'silver_hair', 'orange_hair', 'grey_hair'], '머리색'),
                (['blue_eyes', 'red_eyes', 'green_eyes', 'brown_eyes', 'yellow_eyes', 'purple_eyes', 'pink_eyes', 'grey_eyes', 'black_eyes', 'orange_eyes'], '눈색'),
                (['short_hair', 'long_hair', 'very_long_hair', 'medium_hair'], '머리 길이'),
                (['standing', 'sitting', 'lying', 'kneeling', 'squatting'], '포즈'),
                (['day', 'night', 'sunset', 'sunrise'], '시간'),
                (['indoors', 'outdoors'], '장소'),
            ]
            tag_lower = {t.lower().replace(' ', '_') for t in unique}
            for group, label in conflict_pairs:
                found = [t for t in group if t in tag_lower]
                if len(found) > 1:
                    conflicts.append({'group': label, 'tags': found})

            # 3. 최적 순서 재배치 (작가→캐릭터→품질→배경→포즈→의상→기타)
            quality_tags = {'masterpiece', 'best_quality', 'high_quality', 'absurdres', 'highres'}
            count_pattern = ['1girl', '2girls', '3girls', '1boy', '2boys', 'solo', 'multiple_girls', 'multiple_boys']

            ordered = {'count': [], 'quality': [], 'body': [], 'clothing': [], 'pose': [], 'bg': [], 'other': []}
            for t in unique:
                tl = t.lower().replace(' ', '_')
                if tl in quality_tags: ordered['quality'].append(t)
                elif any(tl == c for c in count_pattern): ordered['count'].append(t)
                else: ordered['other'].append(t)

            optimized = ordered['count'] + ordered['quality'] + ordered['body'] + ordered['clothing'] + ordered['pose'] + ordered['bg'] + ordered['other']

            removed_count = len(tags) - len(unique)
            return json.dumps({
                'optimized': ', '.join(optimized),
                'removed': removed_count,
                'conflicts': conflicts,
                'tag_count': len(optimized),
            })
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def getCharacterInsight(self, character: str) -> str:
        """캐릭터 대표 프롬프트 태그를 정식 Parquet 자산에서 반환."""
        try:
            # 캐시
            if not hasattr(self, '_char_desc_cache'):
                from core.tag_database import TagAsset, get_tag_database
                self._char_desc_cache = {}
                frame = get_tag_database().read_parquet(
                    TagAsset.CHARACTER_PROMPT_TAGS,
                    columns=['character', 'description'],
                )
                for name, desc in frame.itertuples(index=False, name=None):
                    normalized = str(name or '').lower().strip()
                    description = str(desc or '')
                    if normalized and description:
                        self._char_desc_cache[normalized] = description
                print(f"[CharInsight] Loaded {len(self._char_desc_cache)} characters")
            # 검색
            char_lower = character.lower().strip().replace(' ', '_')
            desc = self._char_desc_cache.get(char_lower, '')
            if not desc:
                for k, v in self._char_desc_cache.items():
                    if char_lower in k:
                        desc = v; break
            if desc:
                tags = [t.strip() for t in desc.split(',') if t.strip()]
                return json.dumps({'character': character, 'tags': tags, 'raw': desc})
            return json.dumps({'tags': [], 'raw': ''})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def classifyTags(self, tags_json: str) -> str:
        """태그 목록을 분류하여 카테고리별로 반환 (tags_db 기반)"""
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
            result = {}

            # TagClassifier 시도
            tc = None
            try:
                from core.tag_classifier import TagClassifier
                if not hasattr(self, '_tag_classifier'):
                    self._tag_classifier = TagClassifier()
                tc = self._tag_classifier
            except Exception:
                pass

            # fallback: 정식 태그 자산 기반 간이 분류
            if not hasattr(self, '_fallback_clothes'):
                self._fallback_clothes = set()
                self._fallback_sexual = set()
                try:
                    from core.tag_database import TagAsset, get_tag_database
                    database = get_tag_database()
                    self._fallback_clothes = {
                        line.lower().replace(' ', '_')
                        for line in database.read_lines(TagAsset.CLOTHING_TAGS_CURATED)
                    }
                    groups = database.load_tag_groups()
                    for group_name in (
                        'sex_acts', 'nudity', 'pussy', 'sexual_positions',
                        'sexual_attire', 'sex_objects',
                    ):
                        self._fallback_sexual.update(
                            tag.lower() for tag in groups.get(group_name, set())
                        )
                except Exception as e:
                    logger.debug("tag classification fallback load failed: %s", e)

            for tag in tags:
                t = tag.strip().lower().replace(' ', '_')
                if tc:
                    cat = tc.classify_tag(t)
                    result[tag] = cat
                else:
                    # fallback 분류
                    if t in self._fallback_sexual:
                        result[tag] = 'sexual'
                    elif t in self._fallback_clothes:
                        result[tag] = 'clothing'
                    elif any(kw in t for kw in ['breast', 'thigh', 'ass', 'navel', 'nipple', 'penis', 'pussy', 'anus']):
                        result[tag] = 'body_parts'
                    elif any(kw in t for kw in ['stand', 'sit', 'ly', 'kneel', 'squat', 'walk', 'run', 'jump', 'smile', 'blush', 'cry', 'open_mouth']):
                        result[tag] = 'pose'
                    elif any(kw in t for kw in ['outdoor', 'indoor', 'sky', 'night', 'beach', 'forest', 'city', 'school', 'water', 'snow']):
                        result[tag] = 'background'
                    elif any(kw in t for kw in ['sex', 'vaginal', 'anal', 'oral', 'cum', 'nude', 'naked', 'penetrat']):
                        result[tag] = 'sexual'
                    else:
                        result[tag] = 'general'
            return json.dumps(result)
        except Exception as e:
            from core.error_handler import handle_error
            handle_error('E050', 'ClassifyTags', e, notify=False)
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, str, int, int, result=str)
    def exportCompareGif(self, before_path: str, after_path: str, duration: int, loops: int) -> str:
        """Before/After 비교 GIF 생성"""
        try:
            from PIL import Image as PILImage
            import os, time

            clean_before = _normalize_vue_path(before_path)
            clean_after = _normalize_vue_path(after_path)
            if not clean_before or not clean_after:
                return json.dumps({'error': '비교 이미지 경로가 올바르지 않습니다'})
            img_a = PILImage.open(clean_before)
            img_b = PILImage.open(clean_after)

            # 크기 통일 (작은 쪽에 맞춤)
            w = min(img_a.width, img_b.width)
            h = min(img_a.height, img_b.height)
            img_a = img_a.resize((w, h), PILImage.LANCZOS)
            img_b = img_b.resize((w, h), PILImage.LANCZOS)

            # 중간 프레임 생성 (부드러운 전환)
            frames = []
            steps = 8
            for i in range(steps + 1):
                alpha = i / steps
                blended = PILImage.blend(img_a, img_b, alpha)
                frames.append(blended)
            # 역방향
            for i in range(steps - 1, 0, -1):
                alpha = i / steps
                blended = PILImage.blend(img_a, img_b, alpha)
                frames.append(blended)

            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gif')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"compare_{int(time.time())}.gif")

            frames[0].save(
                out_path, save_all=True, append_images=frames[1:],
                duration=duration, loop=loops, optimize=True
            )
            return json.dumps({'path': out_path.replace('\\', '/'), 'frames': len(frames)})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(result=str)
    def getTabDefaults(self) -> str:
        """tab_defaults.json 반환"""
        import os
        fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'tab_defaults.json')
        try:
            if os.path.exists(fp):
                with open(fp, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning("getTabDefaults failed: %s", e)
        return '{}'

    def _load_adetailer_models_json(self) -> str:
        models = ["face_yolov8n.pt", "hand_yolov8n.pt", "person_yolov8n-seg.pt",
                   "mediapipe_face_full", "mediapipe_face_short"]
        try:
            from backends import get_backend
            backend = get_backend()
            if backend:
                import requests
                r = requests.get(f"{backend.api_url}/adetailer/v1/ad_model", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    models = data if isinstance(data, list) else data.get('ad_model', [])
        except Exception:
            pass
        return json.dumps(models)

    @pyqtSlot(str)
    def _apply_adetailer_models_json(self, models_json: str):
        """worker 결과를 GUI 스레드에서 proxy 목록에 반영한다."""
        try:
            models = json.loads(models_json)
        except Exception:
            return
        for wid in ('_ad_s1_model', '_ad_s2_model'):
            proxy = self._proxies.get(wid)
            if proxy and hasattr(proxy, 'addItems'):
                proxy.addItems(models)

    @pyqtSlot(result=str)
    def getADetailerModels(self) -> str:
        """하위호환 동기 API. 신규 Vue 코드는 requestADetailerModels를 사용한다."""
        models_json = self._load_adetailer_models_json()
        self._apply_adetailer_models_json(models_json)
        return models_json

    @pyqtSlot()
    def requestADetailerModels(self):
        """ADetailer 모델 목록을 백그라운드에서 조회한다."""
        self._run_async_lookup(
            'adetailer-models',
            self._load_adetailer_models_json,
            self.adetailerModelsReady,
        )

    @pyqtSlot(result=str)
    def getYoloModelLabel(self) -> str:
        """YOLO 모델 라벨 반환 (editor_models/ 자동 감지 포함)"""
        try:
            import os
            from tabs.editor.mosaic_panel import _load_yolo_model_paths
            # _load_yolo_model_paths()가 editor_models/ 내 파일도 자동 감지
            paths = _load_yolo_model_paths()
            if paths:
                names = [os.path.basename(p) for p in paths]
                return ", ".join(names)
        except Exception:
            pass
        return "No Model Loaded"

    @pyqtSlot(result=str)
    def refreshYoloModels(self) -> str:
        """editor_models/ 재스캔 후 라벨 반환"""
        label = self.getYoloModelLabel()
        self.yoloModelUpdated.emit(label)
        return label

    @pyqtSlot(str, result=str)
    def getTagSuggestions(self, prefix: str) -> str:
        """태그 자동완성 후보 반환."""
        try:
            from utils.tag_completer import get_tag_completer
            completer = get_tag_completer()
            # 주의: TagCompleter.get_suggestions의 키워드는 max_count
            suggestions = completer.get_suggestions(prefix, max_count=10)
            return json.dumps(suggestions)
        except Exception as e:
            import traceback
            print(f"[getTagSuggestions] 오류: {e}")
            traceback.print_exc()
            return json.dumps([])

    @pyqtSlot(str, result=str)
    def generateXYZCombinations(self, axes_json: str) -> str:
        """XYZ 축 데이터로 조합 생성"""
        try:
            import itertools
            if isinstance(axes_json, str):
                axes = json.loads(axes_json)
            else:
                axes = axes_json
            if not axes:
                return json.dumps([])
            value_lists = [a.get('values', []) for a in axes]
            types = [a.get('type', '') for a in axes]
            combos = list(itertools.product(*value_lists))
            result = []
            for combo in combos:
                item = {}
                for i, val in enumerate(combo):
                    item[types[i]] = val
                result.append(item)
            return json.dumps({'combinations': result, 'count': len(result)})
        except Exception as e:
            return json.dumps({'error': str(e)})

    # ── 이미지 캡션 (CAFormer 태그 + ToriiGate/Ollama 자연어) ──
    @staticmethod
    def _caption_identifier(value, *, fallback: str = '') -> str:
        """Return a bounded identifier safe to echo through broadcast signals."""
        import re
        cleaned = re.sub(r'[^A-Za-z0-9_.-]', '', str(value or ''))[:128]
        return cleaned or fallback

    @staticmethod
    def _caption_job_key(client_token: str, job_id: str) -> str:
        return f'{client_token}\x1f{job_id}'

    def _begin_caption_job_state(self, payload: dict, total: int) -> None:
        """Create the reconnect journal entry before the worker thread starts."""
        import time
        key = self._caption_job_key(payload['clientToken'], payload['jobId'])
        state = {
            'status': 'running',
            'clientToken': payload['clientToken'],
            'jobId': payload['jobId'],
            'engine': payload['engine'],
            'total': int(total),
            'ok': 0,
            'succeeded': 0,
            'failed': 0,
            'skipped': 0,
            'processed': 0,
            'error': '',
            'items': [None] * int(total),
            'updatedAt': time.time(),
        }
        with self._caption_state_lock:
            self._caption_job_states[key] = state
            # Results can contain captions, so retain only a small recovery window.
            if len(self._caption_job_states) > 8:
                oldest = sorted(
                    self._caption_job_states,
                    key=lambda item: self._caption_job_states[item].get('updatedAt', 0),
                )
                for stale_key in oldest[:-8]:
                    self._caption_job_states.pop(stale_key, None)

    def _update_caption_job_state(
        self,
        payload: dict,
        *,
        item: dict | None = None,
        **updates,
    ) -> None:
        """Update counts/result journal without exposing mutable state across threads."""
        import time
        key = self._caption_job_key(payload['clientToken'], payload['jobId'])
        with self._caption_state_lock:
            state = self._caption_job_states.get(key)
            if state is None:
                return
            if item is not None:
                index = item.get('index')
                if isinstance(index, int) and 0 <= index < len(state['items']):
                    # Recovery needs the rendered text/status, not duplicate tag
                    # score metadata that can make large batch journals enormous.
                    state['items'][index] = {
                        key: item[key]
                        for key in (
                            'clientToken', 'jobId', 'index', 'total', 'path',
                            'caption', 'error', 'skipped', 'engine', 'txtPath',
                        )
                        if key in item
                    }
            state.update(updates)
            state['updatedAt'] = time.time()

    def _caption_txt_path(self, image_path: str, out_dir: str = '') -> str:
        """캡션 .txt 경로. out_dir 지정+유효 시 그 폴더에 {basename}.txt, 아니면 이미지 옆."""
        import os
        if out_dir and os.path.isdir(out_dir):
            base = os.path.splitext(os.path.basename(image_path))[0]
            return os.path.join(out_dir, base + '.txt')
        return os.path.splitext(image_path)[0] + '.txt'

    @staticmethod
    def _caption_mode(payload: dict) -> str:
        """신규 engine 키와 기존 Ollama 전용 payload를 함께 수용한다."""
        mode = str(payload.get('engine') or payload.get('mode') or 'ollama').strip().lower()
        if mode not in {'ollama', 'caformer', 'torii', 'combined'}:
            raise ValueError('지원하지 않는 캡션 처리 방식입니다')
        return mode

    @staticmethod
    def _caption_caformer_options(payload: dict) -> dict:
        return {
            'includeCharacters': bool(payload.get('includeCharacters', True)),
            'includeRating': bool(payload.get('includeRating', False)),
            'thresholdMode': (
                'best' if bool(payload.get('useBestThresholds', True))
                else str(payload.get('thresholdMode') or 'category')
            ),
            'generalThreshold': payload.get('generalThreshold', 0.35),
            'characterThreshold': payload.get('characterThreshold', 0.43),
            'ratingThreshold': payload.get('ratingThreshold', 0.38),
        }

    @staticmethod
    def _caption_result_payload(result) -> dict:
        return {
            'caption': result.text,
            'engine': result.mode,
            'naturalCaption': result.natural_caption,
            'tags': [
                {'name': item.name, 'score': round(float(item.score), 6), 'category': item.category}
                for item in result.tags
            ],
        }

    @staticmethod
    def _write_caption_atomic(path: str, text: str) -> None:
        """중단 시 반쪽짜리 sidecar를 남기지 않는 동일 폴더 원자 저장."""
        import os
        tmp = path + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8', newline='') as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _prepare_caption_payload(self, raw_payload: dict, *, batch: bool) -> tuple[dict, list[str]]:
        """외부 payload의 경로·모드·숫자를 검증하고 정규화한다."""
        import math
        import os
        import uuid
        from core.path_safety import safe_output_dir

        payload = dict(raw_payload or {})
        payload['clientToken'] = self._caption_identifier(
            payload.get('clientToken'), fallback='legacy')
        payload['jobId'] = self._caption_identifier(
            payload.get('jobId'), fallback=uuid.uuid4().hex)
        mode = self._caption_mode(payload)
        raw_files = payload.get('files') if batch else [payload.get('path')]
        if isinstance(raw_files, str):
            raw_files = [raw_files]
        files, seen = [], set()
        for raw in (raw_files or []):
            clean = _normalize_vue_path(str(raw or ''))
            if not clean:
                continue
            key = os.path.normcase(os.path.abspath(clean))
            if key not in seen:
                seen.add(key)
                files.append(clean)
        if not files:
            raise ValueError('대상 이미지가 없습니다')

        model = str(payload.get('model') or '').strip()
        if mode == 'ollama' and not model:
            raise ValueError('Ollama 비전 모델을 지정하세요')
        if mode in {'torii', 'combined'} and model and 'toriigate' not in model.casefold():
            raise ValueError('ToriiGate 처리에는 ToriiGate Ollama 모델을 지정하세요')

        payload['engine'] = mode
        payload['model'] = model
        payload['url'] = str(payload.get('url') or 'http://localhost:11434').strip().rstrip('/')
        payload['prompt'] = str(payload.get('prompt') or '')[:20000]
        def _bool_value(value, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, str):
                return value.strip().lower() not in {'', '0', 'false', 'no', 'off'}
            return bool(value)

        payload['save'] = _bool_value(payload.get('save'), True)
        payload['overwrite'] = _bool_value(payload.get('overwrite'), False)
        payload['includeCharacters'] = _bool_value(payload.get('includeCharacters'), True)
        payload['includeRating'] = _bool_value(payload.get('includeRating'), False)
        payload['useBestThresholds'] = _bool_value(payload.get('useBestThresholds'), True)
        payload['timeout'] = max(30, min(900, int(payload.get('timeout') or 300)))
        separator = str(payload.get('separator') or '\n\n').replace('\r', '')
        payload['separator'] = separator[:32] or '\n\n'
        payload['caformerModelDir'] = str(payload.get('caformerModelDir') or '').strip()

        if mode in {'caformer', 'combined'}:
            for key, default in (
                ('generalThreshold', 0.35),
                ('characterThreshold', 0.43),
                ('ratingThreshold', 0.38),
            ):
                try:
                    value = float(payload.get(key, default))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{key} 값이 올바르지 않습니다') from exc
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f'{key} 값은 0과 1 사이여야 합니다')
                payload[key] = value

        out_dir = str(payload.get('outDir') or '').strip()
        if out_dir and payload['save']:
            out_dir = safe_output_dir(out_dir, create=True)
        payload['outDir'] = out_dir

        if payload['save']:
            targets: dict[str, str] = {}
            for image_path in files:
                target = self._caption_txt_path(image_path, out_dir)
                key = os.path.normcase(os.path.abspath(target))
                previous = targets.get(key)
                if previous and os.path.normcase(previous) != os.path.normcase(image_path):
                    raise ValueError(
                        '서로 다른 동명 이미지가 같은 캡션 파일에 충돌합니다. '
                        '한 종류씩 처리하거나 이미지 이름을 구분하세요.'
                    )
                targets[key] = image_path
        return payload, files

    @staticmethod
    def _create_caption_engine(payload: dict):
        from core.image_captioning import ImageCaptioningEngine
        return ImageCaptioningEngine(
            caformer_model_dir=(payload.get('caformerModelDir') or None),
            # 태그 추론은 warm 0.1초 미만이라 CPU로 고정해 Forge/Comfy/Torii VRAM과
            # 경쟁하지 않는다. core API 자체는 명시 provider가 없으면 GPU도 지원한다.
            caformer_providers=('CPUExecutionProvider',),
            ollama_base_url=payload.get('url') or 'http://localhost:11434',
            ollama_model=(payload.get('model') or None),
            torii_model=(payload.get('model') or None),
            max_caption_pixels=1_000_000,
        )

    def _run_caption_inference(self, engine, image_path: str, payload: dict):
        return engine.caption_result(
            image_path,
            payload['engine'],
            caformer_options=self._caption_caformer_options(payload),
            prompt=payload.get('prompt', ''),
            separator=payload.get('separator', '\n\n'),
            ollama_model=(payload.get('model') or None),
            ollama_base_url=payload.get('url') or 'http://localhost:11434',
            timeout=payload.get('timeout', 300),
        )

    def _caption_runtime_snapshot(self, payload: dict) -> str:
        """디스크 복제나 다운로드 없이 현재 로컬 추론 준비 상태만 확인한다."""
        import importlib.util
        from core.error_handler import sanitize_for_ui
        from core.image_captioning import (
            TORIIGATE_BF16_MODEL,
            discover_caformer_model,
            select_toriigate_model,
        )
        from core.ollama_client import OllamaClient

        try:
            request_id = int(payload.get('requestId') or 0)
        except (TypeError, ValueError):
            request_id = 0
        result = {
            'clientToken': self._caption_identifier(payload.get('clientToken')),
            'requestId': request_id,
            'onnxruntime': importlib.util.find_spec('onnxruntime') is not None,
        }
        try:
            model_dir = discover_caformer_model(payload.get('caformerModelDir') or None)
            result['caformer'] = {
                'available': result['onnxruntime'],
                'modelDir': str(model_dir).replace('\\', '/'),
            }
            if not result['onnxruntime']:
                result['caformer']['error'] = 'onnxruntime가 설치되지 않았습니다.'
        except Exception as exc:
            logger.warning('CAFormer runtime discovery failed: %s', exc)
            result['caformer'] = {'available': False, 'modelDir': '', 'error': sanitize_for_ui(exc)}

        url = str(payload.get('url') or 'http://localhost:11434').rstrip('/')
        requested = str(payload.get('toriiModel') or '').strip()
        try:
            models = OllamaClient(url, TORIIGATE_BF16_MODEL).list_models()
            selected = requested or select_toriigate_model(models)
            matches = {model.casefold() for model in models}
            available = selected.casefold() in matches and 'toriigate' in selected.casefold()
            result['torii'] = {
                'available': available,
                'model': selected,
            }
            if not available:
                result['torii']['error'] = 'Ollama에서 ToriiGate 모델을 찾지 못했습니다.'
        except Exception as exc:
            logger.warning('ToriiGate runtime discovery failed: %s', exc)
            result['torii'] = {
                'available': False,
                'model': requested or TORIIGATE_BF16_MODEL,
                'error': sanitize_for_ui(exc),
            }
        return json.dumps(result, ensure_ascii=False)

    @pyqtSlot(str)
    def requestCaptionRuntime(self, payload_json: str):
        """CAFormer/Ollama 상태 확인은 GUI 스레드 밖에서 수행한다."""
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload['clientToken'] = self._caption_identifier(payload.get('clientToken'))
        import hashlib
        lookup_key = 'caption-runtime-' + hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()[:10]
        self._run_async_lookup(
            lookup_key,
            lambda: self._caption_runtime_snapshot(payload),
            self.captionRuntimeReady,
        )

    @pyqtSlot(str, result=str)
    def captionImage(self, payload_json: str) -> str:
        """하위 호환 단일 호출. 제품 UI는 startCaptionBatch의 비동기 1장 경로를 쓴다."""
        acquired = False
        try:
            from contextlib import nullcontext
            from core.resource_coordinator import get_generation_coordinator
            p, files = self._prepare_caption_payload(
                json.loads(payload_json) if payload_json else {}, batch=False)
            acquired = self._caption_job_lock.acquire(blocking=False)
            if not acquired:
                return json.dumps({'error': '다른 캡션 작업이 이미 실행 중입니다'}, ensure_ascii=False)
            lease = (
                get_generation_coordinator().reserve('caption', unload_llm=False, timeout=0)
                if p['engine'] != 'caformer' else nullcontext()
            )
            with lease:
                try:
                    result = self._run_caption_inference(
                        self._create_caption_engine(p), files[0], p)
                finally:
                    if p['engine'] != 'caformer' and (
                        p['engine'] in {'torii', 'combined'}
                        or bool(p.get('unloadAfter', False))
                    ):
                        try:
                            from core.image_captioning import TORIIGATE_BF16_MODEL
                            from core.ollama_client import OllamaClient
                            OllamaClient(
                                p.get('url') or 'http://localhost:11434',
                                p.get('model') or TORIIGATE_BF16_MODEL,
                            ).unload()
                        except Exception:
                            logger.debug('single caption Ollama unload failed', exc_info=True)
            cap = result.text
            if not cap.strip():
                raise RuntimeError('모델이 빈 캡션을 반환했습니다')
            txt = self._caption_txt_path(files[0], p.get('outDir', ''))
            saved = False
            if p.get('save', True):
                self._write_caption_atomic(txt, cap)
                saved = True
            response = self._caption_result_payload(result)
            response.update({'txtPath': txt.replace('\\', '/'), 'saved': saved})
            return json.dumps(response, ensure_ascii=False)
        except Exception as e:
            logger.exception('single caption failed')
            from core.error_handler import sanitize_for_ui
            return json.dumps({'error': sanitize_for_ui(e)}, ensure_ascii=False)
        finally:
            if acquired:
                self._caption_job_lock.release()

    @pyqtSlot(str, result=str)
    def startCaptionBatch(self, payload_json: str) -> str:
        """단일/여러 이미지의 CAFormer/ToriiGate 캡션을 백그라운드에서 직렬 실행."""
        job_locked = False
        thread_started = False
        p = None
        files = []
        try:
            p, files = self._prepare_caption_payload(
                json.loads(payload_json) if payload_json else {}, batch=True)
            if not self._caption_job_lock.acquire(blocking=False):
                return json.dumps({
                    'clientToken': p['clientToken'],
                    'jobId': p['jobId'],
                    'error': '다른 캡션 작업이 이미 실행 중입니다',
                }, ensure_ascii=False)
            job_locked = True
            self._begin_caption_job_state(p, len(files))

            def _emit(d, **state_updates):
                event = {
                    'clientToken': p['clientToken'],
                    'jobId': p['jobId'],
                }
                event.update(d)
                self._update_caption_job_state(p, item=event, **state_updates)
                try:
                    self.captionProgress.emit(json.dumps(event, ensure_ascii=False))
                except RuntimeError:
                    # The WebChannel can disappear while a daemon worker is still
                    # finishing.  The journal remains queryable after reconnect.
                    logger.debug('caption progress receiver is unavailable', exc_info=True)

            def _run():
                import os
                from contextlib import nullcontext
                from core.error_handler import sanitize_for_ui
                from core.resource_coordinator import get_generation_coordinator

                total = len(files)
                succeeded = failed = skipped = processed = 0
                first_error = ''
                job_error = ''
                try:
                    lease = (
                        get_generation_coordinator().reserve(
                            'caption', unload_llm=False, timeout=0)
                        if p['engine'] != 'caformer' else nullcontext()
                    )
                    with lease:
                        try:
                            engine = self._create_caption_engine(p)
                            for i, path in enumerate(files):
                                public_path = path.replace('\\', '/')
                                public_txt = ''
                                try:
                                    txt = self._caption_txt_path(path, p.get('outDir', ''))
                                    public_txt = txt.replace('\\', '/')
                                    if p['save'] and not p['overwrite'] and os.path.exists(txt):
                                        with open(txt, encoding='utf-8') as handle:
                                            existing = handle.read().strip()
                                        # 0바이트/공백 파일은 실패 중단 흔적일 수 있으므로 재생성한다.
                                        if existing:
                                            skipped += 1
                                            processed = i + 1
                                            _emit(
                                                {'index': i, 'total': total, 'path': public_path,
                                                 'txtPath': public_txt, 'caption': existing,
                                                 'skipped': True},
                                                status='running',
                                                ok=succeeded + skipped,
                                                succeeded=succeeded,
                                                failed=failed,
                                                skipped=skipped,
                                                processed=processed,
                                            )
                                            continue
                                    result = self._run_caption_inference(engine, path, p)
                                    caption = result.text
                                    if not caption.strip():
                                        raise RuntimeError('모델이 빈 캡션을 반환했습니다')
                                    if p['save']:
                                        self._write_caption_atomic(txt, caption)
                                    succeeded += 1
                                    processed = i + 1
                                    event = {
                                        'index': i,
                                        'total': total,
                                        'path': public_path,
                                        'txtPath': public_txt,
                                    }
                                    event.update(self._caption_result_payload(result))
                                    _emit(
                                        event,
                                        status='running',
                                        ok=succeeded + skipped,
                                        succeeded=succeeded,
                                        failed=failed,
                                        skipped=skipped,
                                        processed=processed,
                                    )
                                except Exception as exc:
                                    logger.exception('caption failed for %s', path)
                                    failed += 1
                                    processed = i + 1
                                    error = sanitize_for_ui(exc)
                                    if not first_error:
                                        first_error = error
                                    failed_event = {
                                        'index': i,
                                        'total': total,
                                        'path': public_path,
                                        'error': error,
                                    }
                                    if public_txt:
                                        failed_event['txtPath'] = public_txt
                                    _emit(
                                        failed_event,
                                        status='running',
                                        ok=succeeded + skipped,
                                        succeeded=succeeded,
                                        failed=failed,
                                        skipped=skipped,
                                        processed=processed,
                                        error=first_error,
                                    )
                        finally:
                            # Keep the shared generation lease until model cleanup
                            # completes. Otherwise a newly-started generation could
                            # acquire the lease and have its Ollama model unloaded.
                            if p['engine'] != 'caformer' and (
                                p['engine'] in {'torii', 'combined'}
                                or bool(p.get('unloadAfter', False))
                            ):
                                try:
                                    from core.image_captioning import TORIIGATE_BF16_MODEL
                                    from core.ollama_client import OllamaClient
                                    OllamaClient(
                                        p.get('url') or 'http://localhost:11434',
                                        p.get('model') or TORIIGATE_BF16_MODEL,
                                    ).unload()
                                except Exception:
                                    logger.debug('caption Ollama unload failed', exc_info=True)
                except Exception as exc:
                    logger.exception('caption job failed')
                    job_error = sanitize_for_ui(exc)
                    if not first_error:
                        first_error = job_error
                    for i in range(processed, total):
                        path = files[i]
                        failed += 1
                        processed = i + 1
                        _emit(
                            {'index': i, 'total': total, 'path': path.replace('\\', '/'),
                             'error': job_error},
                            status='running',
                            ok=succeeded + skipped,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=skipped,
                            processed=processed,
                            error=first_error,
                        )
                finally:
                    done_payload = {
                        'clientToken': p['clientToken'],
                        'jobId': p['jobId'],
                        'total': total,
                        # ok retains the legacy invariant ok + failed == total;
                        # succeeded/skipped make the two successful outcomes explicit.
                        'ok': succeeded + skipped,
                        'succeeded': succeeded,
                        'failed': failed,
                        'skipped': skipped,
                        'processed': processed,
                        'engine': p['engine'],
                        'error': job_error or first_error,
                    }
                    self._update_caption_job_state(p, status='done', **{
                        key: done_payload[key]
                        for key in (
                            'ok', 'succeeded', 'failed', 'skipped', 'processed', 'error'
                        )
                    })
                    try:
                        self.captionDone.emit(json.dumps(done_payload, ensure_ascii=False))
                    except RuntimeError:
                        logger.debug('caption completion receiver is unavailable', exc_info=True)
                    finally:
                        self._caption_job_lock.release()

            threading.Thread(
                target=_run,
                daemon=True,
                name='caption-inference',
            ).start()
            thread_started = True
            return json.dumps({
                'started': True,
                'clientToken': p['clientToken'],
                'jobId': p['jobId'],
                'total': len(files),
                'engine': p['engine'],
            })
        except Exception as e:
            if job_locked and not thread_started:
                self._caption_job_lock.release()
            if p is not None:
                error = str(e)
                try:
                    from core.error_handler import sanitize_for_ui
                    error = sanitize_for_ui(e)
                except Exception:
                    pass
                for index, path in enumerate(files):
                    event = {
                        'clientToken': p['clientToken'],
                        'jobId': p['jobId'],
                        'index': index,
                        'total': len(files),
                        'path': path.replace('\\', '/'),
                        'error': error,
                    }
                    self._update_caption_job_state(p, item=event)
                self._update_caption_job_state(
                    p,
                    status='done',
                    ok=0,
                    succeeded=0,
                    failed=len(files),
                    skipped=0,
                    processed=len(files),
                    error=error,
                )
            logger.exception('unable to start caption job')
            from core.error_handler import sanitize_for_ui
            response = {'error': sanitize_for_ui(e)}
            if p is not None:
                response.update({
                    'clientToken': p['clientToken'],
                    'jobId': p['jobId'],
                })
            return json.dumps(response, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def getCaptionJobStatus(self, payload_json: str = '') -> str:
        """Return a journaled caption job so Vue can recover missed Qt signals."""
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        client_token = self._caption_identifier(payload.get('clientToken'))
        job_id = self._caption_identifier(payload.get('jobId'))
        with self._caption_state_lock:
            if client_token and job_id:
                key = self._caption_job_key(client_token, job_id)
                state = self._caption_job_states.get(key)
                if state is not None:
                    return json.dumps(state, ensure_ascii=False)
        return json.dumps({
            'status': 'idle',
            'clientToken': client_token,
            'jobId': job_id,
        }, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def loadCaption(self, payload_or_path: str) -> str:
        """이미지 옆 또는 선택 출력 폴더의 .txt 사이드카를 안전하게 읽는다."""
        try:
            from core.path_safety import safe_output_dir
            try:
                payload = json.loads(payload_or_path)
            except (TypeError, json.JSONDecodeError):
                payload = {'path': payload_or_path}
            if not isinstance(payload, dict):
                payload = {'path': payload_or_path}
            path = _normalize_vue_path(str(payload.get('path') or ''))
            if not path:
                raise ValueError('허용되지 않은 이미지 경로입니다')
            out_dir = str(payload.get('outDir') or '').strip()
            if out_dir:
                out_dir = safe_output_dir(out_dir, create=False)
            txt = self._caption_txt_path(path, out_dir)
            public_txt = txt.replace('\\', '/')
            if os.path.exists(txt):
                with open(txt, encoding='utf-8') as f:
                    return json.dumps({
                        "caption": f.read().strip(),
                        "txtPath": public_txt,
                    }, ensure_ascii=False)
            return json.dumps({"caption": "", "txtPath": public_txt}, ensure_ascii=False)
        except Exception as e:
            logger.warning('caption sidecar load failed: %s', e)
            from core.error_handler import sanitize_for_ui
            return json.dumps({'error': sanitize_for_ui(e)}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def saveCaption(self, payload_json: str) -> str:
        """캡션을 .txt 사이드카로 저장. payload {path, caption, outDir}. → {ok, txtPath}."""
        save_locked = False
        try:
            from core.path_safety import safe_output_dir
            p = json.loads(payload_json) if payload_json else {}
            path = _normalize_vue_path(str(p.get('path') or ''))
            if not path:
                raise ValueError('허용되지 않은 이미지 경로입니다')
            save_locked = self._caption_job_lock.acquire(blocking=False)
            if not save_locked:
                raise RuntimeError('캡션 작업 중에는 수동 저장할 수 없습니다')
            out_dir = str(p.get('outDir') or '').strip()
            if out_dir:
                out_dir = safe_output_dir(out_dir, create=True)
            txt = self._caption_txt_path(path, out_dir)
            self._write_caption_atomic(txt, str(p.get('caption') or ''))
            return json.dumps({"ok": True, "txtPath": txt.replace('\\', '/')}, ensure_ascii=False)
        except Exception as e:
            logger.exception('caption sidecar save failed')
            from core.error_handler import sanitize_for_ui
            return json.dumps({'error': sanitize_for_ui(e)}, ensure_ascii=False)
        finally:
            if save_locked:
                self._caption_job_lock.release()

    @pyqtSlot(str, result=bool)
    def copyTextToClipboard(self, text: str) -> bool:
        """Copy on the desktop GUI thread; remote clients keep their own clipboard."""
        if self._backend_runtime_is_web_mode():
            return False
        try:
            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            if clipboard is None:
                return False
            clipboard.setText(text)
            return clipboard.text() == text
        except Exception:
            logger.warning('Desktop clipboard write failed', exc_info=True)
            return False

    @pyqtSlot(str, result=str)
    def getImageExif(self, filepath: str) -> str:
        """Shared A1111/Forge/Comfy metadata reader; never modifies the image."""
        try:
            from core.image_metadata import read_metadata_for_ui
            clean = _normalize_vue_path(filepath)
            if not clean:
                return json.dumps({'error': '파일을 찾을 수 없습니다', 'path': filepath})
            info = read_metadata_for_ui(clean)
            info['params'] = self._parse_params_line(info.get('params_line', ''))
            return json.dumps(info, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'error': str(e), 'path': filepath})

    def _parse_params_line(self, params_line: str) -> dict:
        """SD Parameter 라인을 구조화된 딕셔너리로 파싱"""
        import re
        result = {'generation': '', 'model': '', 'hires': '', 'extensions': '', 'other': ''}
        # 개별 파라미터 파싱 (Key: Value 형식)
        params = {}
        for m in re.finditer(r'([A-Za-z][A-Za-z0-9_ ]*?):\s*([^,]+?)(?:,\s*|$)', params_line):
            params[m.group(1).strip()] = m.group(2).strip()

        # Line 1: Steps + Sampler + Scheduler
        gen_parts = []
        for k in ['Steps', 'Sampler', 'Schedule type']:
            if k in params:
                gen_parts.append(f"{k}: {params.pop(k)}")
        result['generation'] = ', '.join(gen_parts)

        # Line 2: CFG, Seed, Size
        core_parts = []
        for k in ['CFG scale', 'Seed', 'Size']:
            if k in params:
                core_parts.append(f"{k}: {params.pop(k)}")
        result['core'] = ', '.join(core_parts)

        # Line 3: Model
        model_parts = []
        for k in ['Model', 'Model hash', 'VAE', 'Clip skip']:
            if k in params:
                model_parts.append(f"{k}: {params.pop(k)}")
        result['model'] = ', '.join(model_parts)

        # Line 4: Hires
        hires_parts = []
        for k in list(params.keys()):
            if k.lower().startswith('hires') or k.lower().startswith('hr ') or 'Denoising strength' == k:
                hires_parts.append(f"{k}: {params.pop(k)}")
        result['hires'] = ', '.join(hires_parts)

        # Line 5: Extensions (ADetailer, SAM3, NegPiP 등)
        ext_parts = []
        for k in list(params.keys()):
            kl = k.lower()
            if any(x in kl for x in ['adetailer', 'sam3', 'negpip', 'controlnet', 'ad_', 'tiled']):
                ext_parts.append(f"{k}: {params.pop(k)}")
        result['extensions'] = ', '.join(ext_parts)

        # 나머지
        other_parts = [f"{k}: {v}" for k, v in params.items()]
        result['other'] = ', '.join(other_parts)

        return result

