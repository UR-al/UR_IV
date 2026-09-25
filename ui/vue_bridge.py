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
from core.ollama_client import DEFAULT_OLLAMA_URL

logger = logging.getLogger(__name__)

# 편집 임시본 보존 개수. EditorView의 MAX_UNDO(30)보다 넉넉해야 undo 히스토리가
# 참조하는 파일이 지워지지 않는다.
_EDITOR_TEMP_KEEP = 60
# 썸네일 캐시 상한 — 넘으면 오래된 것부터 정리 (언제든 재생성 가능한 캐시)
_THUMB_CACHE_MAX_BYTES = 300 * 1024 * 1024

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
    return _scan_gallery_media_versions(target, recursive_roots)[0]


def _scan_gallery_media_versions(target: str, recursive_roots=()) -> tuple[list[str], list[str]]:
    """(경로 목록, 같은 순서의 내용 버전). 버전 = 원본 서명(mtime_ns·크기)의 16진 문자열, 못 읽으면 ''.

    카드 썸네일 URL 은 경로·폭만 담아, 원본을 덮어쓰면 브라우저가 같은 URL 의 옛 그림을 계속
    보여 줬다(Qt 페이지 이미지 캐시·웹 max-age). 프런트가 이 버전을 URL 에 붙인다
    (frontend/src/utils/mediaVersions.ts). scandir 의 stat 캐시라 추가 시스템 호출이 없다.
    """
    entries = []

    def collect(folder: str) -> None:
        try:
            with os.scandir(folder) as scan:
                for entry in scan:
                    if not entry.is_file() or os.path.splitext(entry.name)[1].lower() not in _GALLERY_MEDIA_EXTS:
                        continue
                    try:
                        stat = entry.stat()
                        mtime = stat.st_mtime
                        version = f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
                    except OSError:
                        mtime, version = 0, ''
                    entries.append((mtime, entry.path.replace('\\', '/'), version))
        except OSError:
            return

    collect(target)
    for recursive_root in recursive_roots:
        if not os.path.isdir(recursive_root):
            continue
        for folder, _dirs, _files in os.walk(recursive_root):
            collect(folder)
    entries.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path, _v in entries], [version for _, _p, version in entries]


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
    # Vue I2I 진행 — JSON {running, cancelling} (ui/i2i_actions.emit_job_state). I2IView 가
    # 실행 중 시작 버튼을 막고 취소 버튼(cancel_i2i)을 보여 준다.
    i2iJobState = pyqtSignal(str)
    galleryFolderLoaded = pyqtSignal(str)  # folder path
    # delete_image 결과 1건 — JSON {path(요청 원문), ok, removed, level, message}.
    # 프론트는 removed 가 참일 때만 갤러리·폴더 캐시·히스토리에서 뺀다(core/image_delete.py).
    imageDeleteResult = pyqtSignal(str)
    inpaintImageLoaded = pyqtSignal(str)   # file path — InpaintView 전용 (send_to_inpaint)
    # PNG Info '열기'로 고른 파일 — PngInfoView 전용. 예전엔 inpaintImageLoaded 를 같이 써서
    # keep-alive 로 살아 있는 InpaintView 의 원본·마스크·undo 가 PNG Info 열기마다 날아갔다.
    pngInfoImageLoaded = pyqtSignal(str)   # file path
    searchStatus = pyqtSignal(str)         # status message

    loraStackLoaded = pyqtSignal(str)    # JSON [{name, weight, enabled, triggerWords}]
    yoloModelUpdated = pyqtSignal(str)   # model label text
    condRulesLoaded = pyqtSignal(str)    # JSON {positive, negative}
    batchFilesSelected = pyqtSignal(str) # JSON [paths]
    # Vue 일괄 처리·업스케일 진행 — JSON {job, running, done, total, success, failed,
    # output_dir, stopped} (core/batch_job_state.py). 실행 중 시작 버튼을 막는 데 쓴다.
    batchJobState = pyqtSignal(str)
    ollamaResult = pyqtSignal(str)       # JSON {tags, mode} or {error}
    genNlResult = pyqtSignal(str)        # JSON {tags, mode} or {error} — 생성 시 태그→자연어 전용 채널
    globalWeightsLoaded = pyqtSignal(str) # JSON [{tag, weight}]
    uiPrefsLoaded = pyqtSignal(str)      # JSON {tagBlockMode, ...}
    compareImageLoaded = pyqtSignal(str) # JSON {slot, path}
    galleryImagesReady = pyqtSignal(str) # JSON {folder, files, versions(files 와 같은 순서의 원본 서명)}
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
    # show_status 한 줄 — 하단 계기 스트립 표시. JSON {text, level, timeoutMs} (core/status_message.py)
    statusMessage = pyqtSignal(str)
    # GUI 스레드를 막던 동기 슬롯의 비동기 짝 — 모두 JSON 에 요청 식별자(requestId)를 되돌려 준다.
    lorasReady = pyqtSignal(str)                 # {requestId, mode, loras:[...]} | {requestId, mode, error}
    characterTagsOnlineReady = pyqtSignal(str)   # {requestId, name, tags, sampled} | {requestId, name, error}
    compareGifReady = pyqtSignal(str)            # {requestId, path, frames} | {requestId, error}
    adetailerResult = pyqtSignal(str)       # JSON {before, after, output_path} or {error}
    adetailerProgress = pyqtSignal(int, int) # (current, total)
    sam3Result = pyqtSignal(str)            # JSON {before, after, output_path} or {error}
    sam3Progress = pyqtSignal(int, int)     # (current, total)
    # Refine (sam-extra 워크플로 2) — JSON {before, after, prompt, negative_prompt} or {error}
    refineResult = pyqtSignal(str)
    # sam-extra 임베드 LoRA Manager 주소 — JSON {url, status, message}
    loraManagerUrlReady = pyqtSignal(str)
    eventSearchProgress = pyqtSignal(int, int) # (current, total)
    # Event 데이터 적재 진행 문구 — Search 전용 searchStatus 와 분리(두 뷰가 keep-alive 로 공존)
    eventLoadStatus = pyqtSignal(str)
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

    # 앱 관리형 Forge Neo / ComfyUI runtime 이벤트(Python 내부 전용). Studio runtime.execute 작업의
    # 진행/완료를 DesktopNativeHost(ui/studio_qwebchannel.py)가 이 시그널로 넘기고
    # generator_main._on_backend_runtime_event 가 연결 전환을 처리한다. Vue 는 Studio journal 을 구독한다.
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
        # Python → Vue 위젯 속성(items·enabled·placeholder …)의 마지막 값 (widget_id → {prop: JSON}).
        # 속성은 push 로만 가서, Vue 페이지가 뜨기 전(_setup_ui 에서 채우는 SAM3 ControlNet
        # 선택지 등)이나 웹 클라이언트가 붙기 전에 보낸 값은 사라졌다 — getAllWidgetProperties
        # 가 이 표를 돌려줘 늦게 연결한 클라이언트도 같은 상태에서 시작한다.
        self._widget_properties = {}
        self._widget_properties_lock = threading.Lock()
        self._batch_mode = False
        self._batch_buffer = {}
        self._action_handler = None  # 액션 디스패처 (메인 윈도우에서 설정)
        self._async_lookup_inflight = set()
        self._async_lookup_lock = threading.Lock()
        self._caption_job_lock = threading.Lock()
        # Caption signals are broadcast by QWebChannel.  Keep a small keyed
        # journal so the initiating Vue client can ignore another client's job
        # and recover state after a transient WebChannel reconnect.
        self._caption_state_lock = threading.Lock()
        self._caption_job_states = {}
        # 호스트 대화상자가 승인한 캡션 저장 폴더(core.caption_out_dir) — 첫 사용 때 ui_prefs 에서 읽는다.
        self._caption_out_dir_approvals = None
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
        """위젯 속성을 Vue로 전송 — 마지막 값은 getAllWidgetProperties 용으로 남긴다."""
        value_json = json.dumps(value)
        with self._widget_properties_lock:
            self._widget_properties.setdefault(widget_id, {})[prop] = value_json
        self.widgetPropertyChanged.emit(widget_id, prop, value_json)

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
        # 웹 모드 권한 정책(core.web_action_policy): 호스트 권한·설정을 바꾸는 액션은 항상,
        # 호스트에 네이티브 대화상자를 띄우는 액션은 원격 웹 모드에서 핸들러 전에 거부한다.
        from core.web_action_policy import web_action_denial
        denial = web_action_denial(
            action,
            web_mode=self._backend_runtime_is_web_mode(),
            remote=self._web_clients_are_remote(),
        )
        if denial:
            self.showNotification.emit('warning', denial)
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

    @pyqtSlot(result=str)
    def getAllWidgetProperties(self) -> str:
        """지금까지 push 한 위젯 속성의 마지막 값 ``{widget_id: {prop: value}}`` (초기 로드·재연결용).

        widgetPropertyChanged 는 연결된 클라이언트에만 가서, 페이지 로드 전에 채운 콤보
        선택지(예: SAM3 ControlNet 전처리기 목록)가 Vue 에 닿지 않았다. QWebChannel 은
        응답과 시그널을 보낸 순서대로 전달하므로 이 스냅숏은 앞서 받은 push 보다 오래되지 않는다.
        """
        with self._widget_properties_lock:
            snapshot = {wid: dict(props) for wid, props in self._widget_properties.items()}
        return json.dumps({
            wid: {prop: json.loads(value_json) for prop, value_json in props.items()}
            for wid, props in snapshot.items()
        })

    def last_widget_property(self, widget_id: str, prop: str, default=None):
        """Python 이 마지막으로 push 한 위젯 속성 값(예: TE 칩 선택지 ``te_main_input.items``).

        한 번도 보낸 적 없으면 ``default``. 프리셋 적용(ui/generation_settings_apply.py)이 Vue 가
        지금 보여 주는 선택지와 같은 목록으로 값을 거른다. Python 전용 — @pyqtSlot 아님.
        """
        with self._widget_properties_lock:
            value_json = self._widget_properties.get(widget_id, {}).get(prop)
        if value_json is None:
            return default
        try:
            return json.loads(value_json)
        except ValueError:
            return default

    # ── 호스트 모드 판정 (앱 관리형 runtime·생성 API·모델 경로는 Studio Interface —
    #    core/studio_application.py — 가 담당한다. 레거시 설정 슬롯은 제거됨) ──

    def _backend_runtime_is_web_mode(self) -> bool:
        """로컬 파일/프로세스 변경 권한이 없는 Web host인지 확인."""
        return bool(getattr(self.parent(), 'web_mode', False))

    def _web_clients_are_remote(self) -> bool:
        """웹 모드가 loopback 밖(LAN)에 열려 있는지 — web_main_ui 가 window.web_remote 로 알린다.

        표시가 없으면 원격으로 본다(fail closed): 호스트 대화상자를 원격 사용자가 볼 수 없다.
        """
        return bool(getattr(self.parent(), 'web_remote', True))

    # ── Editor ──

    editorResult = pyqtSignal(str)  # 에디터 비동기 처리 결과 JSON (path|mask_base64|error + operation, job_id, doc_gen)
    # 에디터 저장/다른 이름으로 저장 결과 JSON (request_id, ok, path|cancelled|error, snapshot_path …)
    # — Vue 는 이 응답을 받은 뒤에만 '저장됨' 상태로 바꾼다 (ui/editor_save_actions.py)
    editorSaveResult = pyqtSignal(str)
    # 크래시 복구본(자동저장) 결과 — requestEditorAutoSave 짝.
    # {requestId, path, drawing} | {requestId, discarded: true} | {requestId, error}
    editorAutoSaveReady = pyqtSignal(str)
    # 프리뷰 축소 한도 — 이보다 길면 줄여서 처리한다(1024면 대부분 20ms 내).
    # 프론트 EditorView 의 PREVIEW_MAX_EDGE(마스크 축소)와 같은 값이어야 한다.
    _PREVIEW_MAX_EDGE = 1024
    # 편집 결과(undo 히스토리)와 복구본 작업 사본이 사는 폴더 — prune_editor_temp 가 정리한다.
    # 인스턴스 속성으로 덮어쓸 수 있게 둔다(테스트가 사용자 editor_temp 를 건드리지 않도록).
    _editor_temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'image_cache', 'editor_temp')

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
            # 요청한 문서의 세대를 모든 결과(이미지·마스크·오류·프리뷰)에 돌려준다 — 그사이 다른
            # 이미지를 열거나 에디터를 닫았으면 프론트가 옛 문서의 결과를 새 문서에 넣지 않고 버린다.
            doc_gen = params.get('doc_gen')
            if isinstance(doc_gen, (int, float)) and not isinstance(doc_gen, bool):
                payload['doc_gen'] = doc_gen
            if params.get('preview'):
                # 프론트가 걷은(무효화한) 프리뷰가 늦게 도착하면 버릴 수 있게 세대 토큰을 돌려준다.
                # 프리뷰 실패도 '프리뷰였다'고 표시해야 확정 작업 오류처럼 토스트를 띄우지 않는다.
                payload['preview_request'] = True
                if isinstance(params.get('preview_token'), (int, float)):
                    payload['preview_token'] = params['preview_token']
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
            from core.editor_preview import (
                PreviewSourceCache, encode_preview, imread_unchanged, imwrite as _imwrite_u,
                normalize_channels,
            )

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
            #
            # 프리뷰는 축소 소스를 캐시에서 사본으로 받는다 — 매 틱 원본 풀 디코드를 없앤다.
            # 확정 작업은 캐시를 쓰지 않고 원본 해상도로 새로 읽는다.
            is_preview = bool(params.get('preview'))
            # 축소본 픽셀 / 원본 픽셀 — 워터마크 글자 크기처럼 '원본 px' 로 받은 값을
            # 프리뷰에서도 확정 결과와 같은 비율로 보이게 할 때 곱한다.
            preview_scale = 1.0
            if is_preview:
                cache = getattr(self, '_editor_preview_cache', None)
                if cache is None:
                    cache = PreviewSourceCache(self._PREVIEW_MAX_EDGE)
                    self._editor_preview_cache = cache
                img, preview_scale = cache.get_scaled(clean_path)
            else:
                # IMREAD_UNCHANGED — 기본 IMREAD_COLOR는 알파를 버린다.
                # 그래서 '배경 제거' 후 아무 편집이나 하면 투명도가 죽고 배경이 검게 됐다.
                # 한글 경로에서도 읽히도록 np.fromfile + imdecode 로 읽는다.
                img = imread_unchanged(clean_path)
                if img is not None:
                    # 채널 정규화: 흑백 → BGR. BGRA는 그대로 두고 각 연산이 알파를 보존한다.
                    img = normalize_channels(img)
            if img is None:
                return json.dumps({'error': '이미지를 읽을 수 없습니다 (OpenCV)'})

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
                    from core import yolo_models
                    model_paths = yolo_models.load_model_paths()
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
                        if yolo_models.is_sam_file(mp):
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
                        # 이 블록의 로그는 예외를 내지 않는다 — stdout이 파이프·파일(cp949)이면 '✓'·'—'가
                        # UnicodeEncodeError를 내고, 이미 적용된 마스크에 'SAM 정밀화 실패' 거짓 토스트가 붙었다.
                        from core.safe_print import safe_print as _sam_log
                        try:
                            from core.sam_refiner import (
                                refine_boxes_with_sam, resolve_sam_model, notify_sam_unavailable,
                            )
                            models_dir = yolo_models.get_editor_models_dir()
                            # 파일만 있고 실행 패키지가 없으면(예: mobile_sam 미설치) 사용 불가로 판정 —
                            # 예전에는 bbox를 채운 마스크가 '✓ Refined'로 보고됐다.
                            sam_res = resolve_sam_model(models_dir, prefer_type=sam_choice)
                            sam_path, sam_type = sam_res.path, sam_res.sam_type
                            _sam_log(f"[SAM] choice={sam_choice}, models_dir={models_dir}, found={sam_path}, type={sam_type}, "
                                     f"has_seg={has_seg_mask}, missing={sam_res.missing_package}")

                            # 사용자 알림 콜백 — SAM3 exclude 안전장치, 패키지 없음(세션당 1회), 정밀화 실패
                            def _sam_notify(level, message):
                                try:
                                    self.showNotification.emit(level, message)
                                except Exception:
                                    pass

                            if has_seg_mask and (sam_type or sam_res.missing_type) != 'sam3':
                                # YOLO seg 마스크가 이미 있고 SAM3가 아니면 정밀화 생략
                                _sam_log("[SAM] YOLO seg mask available, skipping SAM")
                            elif sam_path:
                                # SAM3 전용: 마스크에서 빼고 싶은 영역의 텍스트 프롬프트
                                #   예: 'face' → 얼굴 영역을 검출해서 최종 마스크에서 빼기
                                excl_prompt = params.get('exclude_prompt') or params.get('excludePrompt')
                                excl_prompt = str(excl_prompt).strip() if excl_prompt else None
                                if excl_prompt and sam_type == 'sam3':
                                    _sam_log(f"[SAM3] exclude prompt requested: '{excl_prompt}'")
                                if detect_prompt and sam_type == 'sam3':
                                    _sam_log(f"[SAM3] detect prompt: '{detect_prompt}'")
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
                                    _sam_log(f"[SAM] ✓ Refined mask applied ({sam_type}, {len(yolo_boxes)} boxes → {pixel_count} pixels)")
                                else:
                                    _sam_log("[SAM] No mask generated, using YOLO bbox")
                            elif sam_res.missing_package:
                                # 모델 파일은 있지만 실행 패키지가 없다 — 세션당 1회 경고 후 YOLO 마스크.
                                # auto라도 다른 모델(특히 SAM3 3.45GB)로 몰래 넘어가지 않는다.
                                notify_sam_unavailable(_sam_notify, sam_res.missing_type,
                                                       sam_res.missing_package)
                            else:
                                if sam_choice != 'auto':
                                    _sam_log(f"[SAM] '{sam_choice}' 모델이 editor_models/에 없음 — bbox 사용")
                                else:
                                    _sam_log(f"[SAM] No SAM model in {models_dir}, using YOLO bbox")
                        except ImportError as ie:
                            _sam_log("[SAM] Import error:", ie)
                            try:
                                self.showNotification.emit('warning', f'SAM 라이브러리 미설치 — bbox 마스크 사용 ({type(ie).__name__})')
                            except Exception:
                                pass
                        except Exception as sam_e:
                            import traceback
                            _sam_log("[SAM] Error:", sam_e)
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

            elif operation in ('text_watermark', 'image_watermark'):
                # 렌더링은 core.editor_watermark — BGRA 는 BGRA 로 돌려준다(투명도 보존),
                # 글꼴 표시명 → 파일 매핑·요청 크기 폴백, 이미지 워터마크 크기는 '%',
                # 프리뷰는 축소 배율(preview_scale)만큼 글자·워터마크를 줄여 확정과 같은 비율로.
                from core.editor_watermark import (
                    WatermarkError, render_image_watermark, render_text_watermark,
                )
                render = render_text_watermark if operation == 'text_watermark' else render_image_watermark
                try:
                    img = render(img, params, pixel_scale=preview_scale)
                except WatermarkError as e:
                    return json.dumps({'error': str(e)})

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
                # rembg 세션은 유휴 캐시(core/model_cache.REMBG_CACHE)에서 빌린다 — 예전에는
                # session 없이 remove() 를 불러 클릭마다 u2net 세션을 새로 만들었다.
                # 품질 프리셋·BGRA/16비트 입력·'quality' 엣지 정제는 core/bg_removal.py.
                try:
                    from core.bg_removal import remove_background
                    img = remove_background(img, params.get('quality', 'balanced'))
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
                src_img = imread_unchanged(src_path)   # 한글 경로 안전
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
                # 알파가 없으면 JPEG(수십~수백 KB), 있으면 PNG — 예전엔 늘 PNG 1.3~2.8MB 였다.
                data_url = encode_preview(img)
                if not data_url:
                    return json.dumps({'error': '프리뷰 인코딩 실패'})
                return json.dumps({
                    'preview': True,
                    'image_base64': data_url,
                    'width': img.shape[1], 'height': img.shape[0],
                })

            # 결과 저장 — uuid4로 파일명 충돌 제거.
            # (예전 f"edited_{int(time.time())}_{randint(100,999)}"는 같은 초에 1/900 확률로
            #  충돌해 직전 편집본을 덮어썼다.)
            import uuid
            out_dir = self._editor_temp_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"edited_{uuid.uuid4().hex}.png")
            if not _imwrite_u(out_path, img):   # 앱 경로에 한글이 있어도 쓰이도록 imencode+tofile
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
        try:
            from core.config_migration import load_ui_prefs, read_legacy_gallery_folder
            from core.ui_prefs import ui_prefs_path
            prefs_path = ui_prefs_path()
            prefs = load_ui_prefs(prefs_path)
            saved = str(prefs.get('galleryFolder', '') or '').strip()
            if saved:
                return saved
            # 옛 txt 는 이 PC 에 실제로 있는 폴더일 때만 흡수한다 — 저장소에 추적된 다른 PC 경로를
            # 새 클론이 ui_prefs 에 영구 저장해 갤러리가 빈 폴더에 고정되던 문제(generator_settings 와 동일 규칙).
            legacy = read_legacy_gallery_folder(os.path.join(base, 'config', 'gallery_last_folder.txt'))
            if legacy:
                self._save_gallery_folder(legacy)
                return legacy
        except Exception as e:
            logger.warning("getLastGalleryFolder failed: %s", e)
        from config import OUTPUT_DIR
        return OUTPUT_DIR

    def _save_gallery_folder(self, folder: str):
        from core.config_migration import load_ui_prefs, save_ui_prefs
        from core.ui_prefs import ui_prefs_path
        prefs_path = ui_prefs_path()
        prefs = load_ui_prefs(prefs_path)
        prefs['galleryFolder'] = str(folder or '').strip()
        save_ui_prefs(prefs_path, prefs)

    def _gallery_images_payload(self, folder: str) -> str:
        """scandir의 stat 캐시를 이용해 날짜순 미디어 목록을 만든다.

        JSON ``{folder, files, versions}`` — ``versions`` 는 ``files`` 와 같은 순서의 원본 서명
        (카드 URL 버전, ``_scan_gallery_media_versions``).
        """
        import os
        from config import OUTPUT_DIR
        target = folder if folder else OUTPUT_DIR
        if not os.path.isdir(target):
            return json.dumps({'folder': folder, 'files': [], 'versions': []})
        try:
            creator_root = os.path.join(target, 'creator')
            recursive_roots = (
                (creator_root,)
                if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(OUTPUT_DIR))
                else ()
            )
            files, versions = _scan_gallery_media_versions(target, recursive_roots)
        except Exception as e:
            logger.warning("gallery scan failed (%s): %s", target, e)
            files, versions = [], []
        return json.dumps({'folder': folder, 'files': files, 'versions': versions})

    # 동기 getGalleryImages 슬롯은 없앴다 — 프론트는 늘 requestGalleryImages(워커 스레드 스캔 →
    # galleryImagesReady)를 먼저 써서 죽은 폴백이었고, 웹 facade 로는 GUI 스레드 디스크 스캔을 부를 수 있었다.

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
        """즐겨찾기 목록 반환 — 추가·삭제와 같은 reader(core.favorites)를 쓴다.

        원문을 그대로 돌려주면 깨진 파일은 프론트 JSON.parse 에서 조용히 빈 화면이
        됐다. 존재하지 않는 경로도 거르지 않는다(분리된 드라이브의 항목을 개별 삭제할 수 있게).
        """
        from core.favorites import load_favorites
        try:
            return json.dumps(load_favorites(), ensure_ascii=False)
        except Exception as e:
            logger.warning('favorites load failed: %s', e)
            return json.dumps([])

    # JSON {"width": 폭, "items": [{"path": 원본, "thumb": "file:///…jpg" 또는 ""}]} — 청크 단위 통지.
    # 캐시 적중은 청크당 1건으로 모아 보내고, 새로 만든 썸네일만 만들 때마다 보낸다(core.thumb_prefetch).
    thumbnailReady = pyqtSignal(str)

    @pyqtSlot(str, int)
    def generateThumbnails(self, paths_json: str, width: int = 256):
        """히스토리 스트립 썸네일을 백그라운드 작업자(최대 2개)가 만들고 thumbnailReady 로 알린다.

        예전엔 호출(40장 청크)마다 스레드를 새로 띄워 부팅 때 수십 개가 동시에 디코드했다.
        이제 프리페처 하나(core.thumb_prefetch)를 지연 생성해 재사용하고, 이미 대기·처리 중인
        (경로, 폭)은 다시 넣지 않는다(처리 중에 다시 요청된 것은 끝난 뒤 한 번 더 돈다).
        렌더·원자적 저장·원본 서명 무효화는 core.thumb_cache. 항목의 ``v`` 는 썸네일 파일 버전.
        캐시: config.THUMB_DIR(image_cache/thumbs_v2)/<sha1 앞 2자리>/<sha1(path@width)>.jpg
        """
        try:
            paths = json.loads(paths_json) if paths_json else []
        except Exception:
            return
        if not isinstance(paths, list) or not paths:
            return
        self._thumbnail_prefetcher().submit(paths, width)

    def _thumbnail_prefetcher(self):
        """썸네일 프리페처(지연 생성, 브리지당 1개). 슬롯은 GUI 스레드에서만 부른다."""
        prefetcher = getattr(self, '_thumb_prefetcher', None)
        if prefetcher is None:
            from config import LEGACY_THUMB_DIR, THUMB_DIR
            from core import thumb_prefetch

            def _emit(payload):
                self.thumbnailReady.emit(json.dumps(payload))

            prefetcher = thumb_prefetch.ThumbnailPrefetcher(
                THUMB_DIR,
                _emit,
                # 용량 정리(maintain_thumb_cache)는 첫 작업 전에 락 안에서 한 번만(생성과 겹치지 않게)
                maintenance=lambda directory: thumb_prefetch.maintain_thumb_cache(
                    directory, _THUMB_CACHE_MAX_BYTES),
                # aithumb:·웹 /thumbnail 과 같은 경로 관문 — 웹 클라이언트가 이 슬롯으로 시스템 폴더
                # 이미지의 축소본을 받아 가지 못한다(지운 히스토리 파일의 캐시는 그대로 보인다).
                # 관문이 돌려준 '검사한 정규화 경로'로 렌더한다 — 원문('%5C'·'..')을 다시 OS 에 넘기지 않는다.
                is_allowed=thumb_prefetch.default_thumb_source_allowed,
                # 옛 캐시 폴더(image_cache/thumbs)의 같은 키는 렌더 전에 옮겨 온다 — 배경 정리(시작 30초
                # 뒤)를 기다리지 않아 첫 실행에 히스토리를 다시 렌더하지 않고, 지운 원본의 썸네일도 남는다.
                legacy_dir=LEGACY_THUMB_DIR,
            )
            self._thumb_prefetcher = prefetcher
        return prefetcher

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
            from core.search_rows import SEARCH_RESULT_CAP

            # 결과 cap 비활성화 — 사용자가 "무제한" 모드 선택 시.
            # cap 은 워커 한 곳에서만 적용한다(무작위 표본, SEARCH_RESULT_CAP).
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
                result_cap=None if self._disable_result_cap else SEARCH_RESULT_CAP,
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

        워커는 results_ready(object) 로 이미 정규화된 NormalizedSearchRows 를 보낸다
        (행 정규화·cap 은 워커 스레드에서 끝남 — GUI 스레드는 직렬화·게시만).
        그 밖의 list[dict] 는 같은 규칙(core.search_rows)으로 제자리 정규화한다.
        """
        # 순서 역전 차단 — 현재 워커가 아닌(이전 검색의) 늦은 시그널은 무시
        sender = self.sender()
        if sender is not None and sender is not getattr(self, '_search_worker', None):
            print("[Search] stale worker result ignored")
            return
        try:
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
            from core.search_rows import (
                NormalizedSearchRows,
                normalize_search_rows_in_place,
            )

            if isinstance(results, NormalizedSearchRows):
                # 워커가 워커 스레드에서 정규화를 끝냈다 — 행 루프를 다시 돌지 않는다.
                # 하위 소비자(덱·디스크·export)는 평범한 list 를 받도록 참조만 옮긴다.
                out = list(results)
            elif isinstance(results, list):
                # 그 밖의 list[dict] — 이 시점 이후 다른 소비자가 없으므로 기존 dict 를
                # 제자리에서 정규화해 재사용한다(두 번째 list[dict] 피크 메모리 없음).
                out = normalize_search_rows_in_place(results)
            else:
                raise TypeError(
                    f'검색 결과 형식이 올바르지 않습니다: {type(results).__name__}'
                )

            if getattr(self, '_disable_result_cap', False):
                # cp949 콘솔에서 인코딩 오류로 게시가 중단되지 않게 ASCII 구두점만 쓴다
                print(f"[Search] cap DISABLED - emitting all {len(out):,} rows (UI 느려질 수 있음)")

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

            # Python 메인 윈도우의 filtered_results도 업데이트 (랜덤 프롬프트용).
            # 새 검색 결과 자체가 필터 base — Vue 는 이 배열의 인덱스로 필터를 보낸다.
            main_win = self.parent()
            if main_win and hasattr(main_win, 'filtered_results'):
                from core.search_session import publish_snapshot
                publish_snapshot(
                    main_win,
                    active=out,
                    base=out,
                    identity=worker_identity,
                    snapshot_id=snapshot_id,
                )
                # 새 검색 → 덱 새로 채움(등급 필터 → 셔플) + 진행도 초기화 저장(옛 진행도
                # 덮어쓰기) + 자동화 패널 덱 현황 갱신 — core.search_deck 단일 경로.
                from core.search_deck import refill_owner_deck
                refill_owner_deck(main_win)

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
        """마지막 검색 결과(active) 로드 (Vue가 onMounted에서 호출)
        Returns: JSON string of list[dict] (빈 경우 '[]')

        Python 이 이미 현재 데이터셋의 스냅숏을 들고 있으면(시작 복원·새 검색·가져오기
        뒤) 메모리의 filtered_results 를 그대로 직렬화한다 — 같은 20MB 캐시를 다시
        파싱하고 덱을 다시 만들지 않는다. 없을 때만 디스크 active 캐시로 런타임을 복원한다.
        """
        try:
            from core.search_result_store import SearchResultStore
            from core.search_session import (
                restore_runtime_from_disk,
                runtime_snapshot_is_current,
            )
            store = SearchResultStore()
            main_win = self.parent()
            if runtime_snapshot_is_current(main_win, store):
                rows = main_win.filtered_results
                identity = main_win._search_dataset_identity
                snapshot_id = main_win._search_snapshot_id
            else:
                rows = restore_runtime_from_disk(main_win, store)
                if store.last_error:
                    print(f"[Search] cache ignored: {store.last_error}")
                identity = store.last_dataset_identity
                snapshot_id = store.last_snapshot_id
                if snapshot_id is not None:
                    print(f"[Search] restored {len(rows):,} rows from disk → filtered_results")
            if snapshot_id is not None:
                self.searchResultLineage.emit(
                    self._search_result_lineage_json(identity, snapshot_id)
                )
            return json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            print(f"[Search] loadLastSearchResults failed: {e}")
        return '[]'

    @pyqtSlot(result=str)
    def loadFullResults(self) -> str:
        """필터 적용 '전' 전체 검색 셋 (Vue가 '필터 해제' 베이스로 사용).
        full cache 우선, 없으면 active 로 폴백.
        (loadLastSearchResults와 달리 덱/filtered_results 의 내용·순서는 바꾸지 않는다 —
        full 을 새로 읽었으면 같은 행을 base 행 객체로 합쳐 메모리 한 벌만 남길 뿐.)

        Python 이 현재 스냅숏을 들고 있으면 그 base(메모리, 없으면 full 캐시를 스냅숏 id
        로 한 번만 읽음 — active 재파싱 없음)를 돌려주고 기록한다. 폴백 응답도 Vue 가 그
        것으로 정할 base 를 기록한다(core.search_session.serve_runtime_base). Vue 는 이 배열의
        인덱스로 필터를 보내므로 Python base 와 Vue base 가 같은 배열이어야 한다."""
        try:
            from core.search_result_store import SearchResultStore
            from core.search_session import serve_runtime_base
            base = serve_runtime_base(self.parent(), SearchResultStore())
            return json.dumps(base, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            print(f"[Search] loadFullResults failed: {e}")
        return '[]'

    @pyqtSlot(result=str)
    def getUiPrefs(self) -> str:
        """ui_prefs.json 전체를 JSON 문자열로 반환 — 세션 중 '지금 값'이 필요한 뷰의 능동 조회용.

        uiPrefsLoaded 는 부팅 때 getInitialConfig 로 당겨 온 **시작 시점 스냅숏**이고(bridge.js
        sticky 재생), save_ui_prefs 는 다시 방송하지 않는다(모든 클라이언트의 watch 가 되먹임된다).
        그래서 늦게 마운트되는 뷰(Settings·Search 등)가 그 뒤에 바뀐 값을 보려면 이 getter 가
        필요하다. (예전 설명의 'PID 경로라 localStorage 가 비워진다'는 틀렸다 — QWebEngine
        프로필은 repo/web_profile 고정 경로라 localStorage 도 재시작 후 남는다. 다만 파일이 주인이고
        localStorage 는 첫 렌더 캐시다.)
        """
        return json.dumps(self._ui_prefs_dict(), ensure_ascii=False)

    @staticmethod
    def _ui_prefs_dict() -> dict:
        """ui_prefs.json(마이그레이션 적용) — 읽지 못하면 {}. getUiPrefs·getInitialConfig 공용.

        클라이언트에게 보내는 사본이라 서버 전용 키(캡션 저장 폴더 승인 목록)는 뺀다.
        """
        try:
            from core.caption_out_dir import APPROVED_PREFS_KEY
            from core.config_migration import load_ui_prefs
            from core.ui_prefs import ui_prefs_path
            prefs = load_ui_prefs(ui_prefs_path())
            if not isinstance(prefs, dict):
                return {}
            prefs.pop(APPROVED_PREFS_KEY, None)
            return prefs
        except Exception as e:
            print(f"[UIPrefs] ui_prefs 읽기 실패: {e}")
            return {}

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
    def saveChatSchemaDraft(self, text):
        """Acknowledge after atomic persistence, not after a fire-and-forget action."""
        try:
            from core.chat_schema_settings import save_schema_draft
            return json.dumps({'ok': True, 'schemaText': save_schema_draft(text)}, ensure_ascii=False)
        except (ValueError, TypeError) as exc:
            return json.dumps({'ok': False, 'error': str(exc)[:500]}, ensure_ascii=False)
        except Exception as exc:
            from core.error_handler import handle_error
            try:
                handle_error('E030', '구조화된 출력 자동 저장', exc, notify=False)
            except Exception:
                pass
            return json.dumps({'ok': False, 'error': '스키마를 저장하지 못했습니다. 파일과 쓰기 권한을 확인하세요.'}, ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def saveInstructionPreset(self, payload):
        return self._instruction_presets_request('save', payload)

    @pyqtSlot(str, result=str)
    def deleteInstructionPreset(self, payload):
        return self._instruction_presets_request('delete', payload)

    def _module_choice_inventory(self):
        """메인 VAE/TE 선택지용 통합 인벤토리 — 연결 시와 같은 활성 엔진 기준."""
        from core.model_inventory import get_model_inventory

        engine = None
        try:
            from backends import BackendType, get_backend_type
            engine = 'comfyui' if get_backend_type() == BackendType.COMFYUI else 'forge'
        except Exception:
            pass
        return get_model_inventory(active_engine=engine)

    def _refresh_forge_module_widgets(self) -> None:
        """Settings 모델 경로 저장·리셋·새로고침 뒤 메인 VAE/TE 선택지를 다시 만든다.

        연결 시(generator_webui.on_webui_info_loaded)와 같은 규칙(core.main_module_choices)·
        통합 인벤토리·마지막 API VAE 목록을 쓴다. 예전엔 Forge 설정 폴더 하나의 파일명 목록으로
        덮어써서 하위폴더·보조 루트 TE 선택과 API 이름 VAE 선택이 조용히 지워졌다(audit #156).
        DesktopNativeHost 가 이 이름으로 호출한다(ui/studio_qwebchannel.py).
        """
        try:
            from core.main_module_choices import (
                filter_te_selection, keep_vae_selection, main_module_choices,
            )

            api_vae = getattr(self.parent(), '_last_vae_api_items', None)
            try:
                inventory = self._module_choice_inventory()
            except Exception as exc:
                logger.warning("Model inventory unavailable for module refresh: %s", exc)
                inventory = None
            vae_items, te_items = main_module_choices(inventory, api_vae)

            vae_proxy = self._proxies.get('vae_main_combo')
            if vae_proxy is not None and hasattr(vae_proxy, 'addItems'):
                # 연결 오류로 비운 콤보면 currentText 는 '' — 비우기 직전의 실제 선택(ComboBoxProxy.preservedText)을 잇는다
                read_current = getattr(vae_proxy, 'preservedText', None) or getattr(vae_proxy, 'currentText', None)
                current = read_current() if callable(read_current) else ''
                vae_proxy.clear()
                vae_proxy.addItems(vae_items)
                selected = keep_vae_selection(current, vae_items)
                if hasattr(vae_proxy, 'setCurrentText'):
                    vae_proxy.setCurrentText(selected)
                # addItems()가 index 0을 자동 선택해도 값 signal은 보내지 않으므로
                # 제거된 이전 VAE가 Vue 상태에 남지 않게 선택값을 명시 동기화한다.
                self.pushWidgetValue('vae_main_combo', selected)

            self.pushWidgetProperty('te_main_input', 'items', te_items)
            te_proxy = self._proxies.get('te_main_input')
            if te_proxy is not None and hasattr(te_proxy, 'text') and hasattr(te_proxy, 'setText'):
                filtered = filter_te_selection(te_proxy.text() or '', te_items)
                if filtered is not None:
                    te_proxy.setText(filtered)

            # 다음 LoRA Manager 열기/새로고침 때 Forge API 목록을 다시 받는다.
            # (진행 중인 프리워밍이 무효화 전 목록을 되살리지 않도록 락 아래에서 비운다)
            from ui.lora_catalog_cache import invalidate as invalidate_lora_cache
            invalidate_lora_cache(self)
        except Exception as exc:
            logger.warning("Forge module widget refresh failed: %s", exc)

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
        """Vue 업스케일러 드롭다운 — 현재 백엔드가 실제로 받는 이름만 (core/upscale_settings).

        예전엔 Forge 의 /sdapi/v1/upscalers 만 물어서 ComfyUI 에선 늘 [] 였다. ComfyUI 는
        모델 없이 되는 내장 방식(Lanczos 등)과 UpscaleModelLoader 선택지를 합친다.
        """
        from core.upscale_settings import comfy_upscale_model_names, upscaler_choices
        is_comfy = False
        try:
            from backends import BackendType, get_backend, get_backend_type
            is_comfy = get_backend_type() == BackendType.COMFYUI
            backend = get_backend()
            if backend:
                import requests
                if is_comfy:
                    r = requests.get(f"{backend.api_url}/object_info/UpscaleModelLoader", timeout=5)
                    names = comfy_upscale_model_names(r.json()) if r.status_code == 200 else []
                    return json.dumps(upscaler_choices('comfyui', names))
                r = requests.get(f"{backend.api_url}/sdapi/v1/upscalers", timeout=5)
                if r.status_code == 200:
                    names = [u.get('name') for u in r.json() if isinstance(u, dict)]
                    return json.dumps(upscaler_choices('webui', names))
        except Exception:
            pass
        # ComfyUI 는 서버를 못 읽어도 내장 방식은 항상 쓸 수 있다.
        return json.dumps(upscaler_choices('comfyui', []) if is_comfy else [])

    @pyqtSlot()
    def requestUpscalers(self):
        """업스케일러 목록을 백그라운드에서 조회한다(결과: upscalersReady).

        HTTP(timeout 5초)를 GUI 스레드에서 돌리던 동기 getUpscalers 슬롯은 없앴다 — 웹 facade 로
        부르면 GUI 이벤트 루프와 모든 WebSocket 클라이언트가 그동안 멈췄다(Codex R3 #3).
        """
        self._run_async_lookup('upscalers', self._load_upscalers_json, self.upscalersReady)

    @pyqtSlot(str, str, str, result=str)
    def saveImagePrompt(self, filepath: str, prompt: str, negative: str) -> str:
        """Gallery 'EXIF 저장' — PNG parameters 의 프롬프트/네거티브만 바꾼다.

        재조립은 core.image_metadata 가 한다: 파라미터 꼬리(Template 줄·따옴표 값 포함)는
        원문 그대로, parameters 청크만 청크 단위로 바꾸고 나머지 청크(APNG 프레임·16비트
        픽셀·gAMA·eXIf·ICC·다른 텍스트 …)는 바이트 그대로, 임시 파일 → os.replace.
        읽기는 IDAT 뒤 텍스트 청크까지 본다 — 꼬리를 잃거나 Comfy 거부를 건너뛰지 않는다.
        ComfyUI 그래프가 든 이미지는 parameters 청크를 새로 만들면 소스 판정이 바뀌므로 거부한다.
        성공하면 새로 읽은 메타데이터(``info``)를 함께 돌려준다.
        """
        try:
            from core.image_metadata import (
                MetadataSource, extract_from_file, read_metadata_for_ui,
                replace_prompt_in_parameters, rewrite_png_parameters,
            )
            clean = _normalize_vue_path(filepath)
            if not clean:
                return json.dumps({'error': '파일을 찾을 수 없습니다'}, ensure_ascii=False)
            if not clean.lower().endswith('.png'):
                return json.dumps({'error': 'PNG 파일만 메타데이터 수정 가능'}, ensure_ascii=False)
            meta = extract_from_file(clean)
            if meta.source == MetadataSource.COMFYUI:
                return json.dumps({'error': 'ComfyUI 워크플로 메타데이터는 수정하지 않습니다'}, ensure_ascii=False)
            text = replace_prompt_in_parameters(meta.raw_parameters, prompt, negative)
            rewrite_png_parameters(clean, text)
            return json.dumps({'ok': True, 'info': read_metadata_for_ui(clean)}, ensure_ascii=False)
        except Exception as e:
            from core.error_handler import sanitize_for_ui
            return json.dumps({'error': sanitize_for_ui(e)}, ensure_ascii=False)

    @pyqtSlot(str, str, result=str)
    def renameFile(self, filepath: str, new_name: str) -> str:
        """파일 이름 변경 — 갤러리가 보여 주는 미디어(이미지·영상·오디오) 전부.

        성공하면 ``new_path`` 를 돌려준다. 프런트는 이 값으로 목록·확대 뷰의 경로를 바꾼다
        (입력한 원문이 아니라 정리·확장자 보정을 거친 실제 경로).
        """
        try:
            import os
            import re
            from core.file_naming import sanitize_filename
            clean = _normalize_vue_path(filepath, allowed_exts=_GALLERY_MEDIA_EXTS)
            if not clean:
                return json.dumps({'error': '파일을 찾을 수 없습니다'})
            dir_path = os.path.dirname(clean)
            ext = os.path.splitext(clean)[1]
            # 구분자 제거 — 새 이름은 같은 디렉토리 안의 단일 파일명만 허용.
            # 'char.v2' 같은 이름의 점은 살리되, 확장자처럼 보이는 꼬리도 영문·숫자만 허용한다
            # (':' 이 든 꼬리는 NTFS 대체 데이터 스트림 이름이 된다).
            stem, new_ext = os.path.splitext(new_name)
            if new_ext and not re.fullmatch(r'\.[A-Za-z0-9_-]{1,16}', new_ext):
                stem, new_ext = new_name, ''
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
        """Canny edge detection → base64 PNG (자석 올가미용).

        읽기는 core.cv_io(np.fromfile + imdecode) — 예전 cv2.imread 는 한글·일본어 경로에서
        None 을 돌려 자석 올가미가 조용히 죽었다.
        """
        try:
            from core.edge_map import edge_map_data_url
            clean = _normalize_vue_path(image_path)
            if not clean:
                return ''
            return edge_map_data_url(clean, canny_low, canny_high)
        except Exception as e:
            logger.warning("getEdgeMap failed: %s", e)
            return ''

    @pyqtSlot(str, str, str)
    def ollamaEnhance(self, tags: str, mode: str, extra_json: str):
        """Ollama로 태그 강화 (비동기). 슬롯은 워커만 띄우고 곧바로 돌아온다 —
        설치 모델 대조(/api/tags, core.ollama_client.resolve_model)도 워커 스레드에서 한다."""
        try:
            from workers.ollama_worker import OllamaWorker, detach_result_signals, release_when_done
            # 이전 요청의 결과 연결만 끊는다(HTTP 는 취소 불가). 끝나면 워커가 스스로 정리된다.
            detach_result_signals(getattr(self, '_ollama_worker', None))
            extra = json.loads(extra_json) if extra_json else {}
            url = str(extra.get('url') or '').strip() or DEFAULT_OLLAMA_URL
            model = str(extra.get('model') or '').strip()   # 비면 워커가 설치 모델로 정한다
            extra_prompt = extra.get('prompt', '')
            if mode == 'creative':
                # 창의 모드: 캐릭터의 실제 외견 태그를 DB에서 조회해 입력에 포함
                tags, extra_prompt = self._build_creative_input(tags, extra.get('character', ''))
            worker = OllamaWorker(url, model, tags, mode, extra_prompt, self, resolve_installed=True)
            worker.finished.connect(lambda r: self.ollamaResult.emit(r))
            worker.error.connect(lambda e: self.ollamaResult.emit(json.dumps({'error': e})))
            release_when_done(worker, self, '_ollama_worker')
            self._ollama_worker = worker
            worker.start()
        except Exception as e:
            self.ollamaResult.emit(json.dumps({'error': str(e)}))

    @pyqtSlot(str, str)
    def convertPromptToNl(self, text: str, extra_json: str):
        """생성 시 태그→자연어(nl_caption) 변환 — 전용 시그널 genNlResult로 결과 전달.
        PromptPanel의 ollamaResult 리스너와 충돌하지 않도록 별도 채널을 사용한다.
        모델 대조는 ollamaEnhance 와 같이 워커 스레드에서 한다."""
        try:
            from workers.ollama_worker import OllamaWorker, detach_result_signals, release_when_done
            detach_result_signals(getattr(self, '_gennl_worker', None))
            extra = json.loads(extra_json) if extra_json else {}
            url = str(extra.get('url') or '').strip() or DEFAULT_OLLAMA_URL
            model = str(extra.get('model') or '').strip()
            worker = OllamaWorker(
                url, model, text, 'nl_caption', '', self, instruction_feature='auto_nl',
                resolve_installed=True,
            )
            worker.finished.connect(lambda r: self.genNlResult.emit(r))
            worker.error.connect(lambda e: self.genNlResult.emit(json.dumps({'error': e})))
            release_when_done(worker, self, '_gennl_worker')
            self._gennl_worker = worker
            worker.start()
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
        웹 모드에도 공개된 쓰기 슬롯이라 확장자·내용·저장 위치를 서버에서 모두 정한다
        (core.clipboard_paste — mime 은 명시 맵으로만, 바이트는 시그니처 확인, mkstemp).
        """
        from core.clipboard_paste import ClipboardPasteError, save_clipboard_image
        try:
            saved = save_clipboard_image(b64_data, mime_type)
            return json.dumps({"path": saved.replace('\\', '/')}, ensure_ascii=False)
        except ClipboardPasteError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except Exception as e:
            logger.warning('clipboard paste save failed: %s', e)
            from core.error_handler import sanitize_for_ui
            return json.dumps({"error": sanitize_for_ui(e)}, ensure_ascii=False)

    @pyqtSlot(result=str)
    def editorPasteFromSystemClipboard(self) -> str:
        """데스크톱 전용 — 시스템 클립보드의 이미지를 Qt 로 직접 읽어 에디터로 연다.

        QtWebEngine 페이지에는 클립보드 읽기 권한(JavascriptCanPaste·permissionRequested)이
        없어 navigator.clipboard.read() 가 막히고, 막히지 않아도 1~3MB 스크린샷은 JS 쪽
        base64 변환과 브리지 왕복이 무겁다. 여기서는 Qt 가 읽은 그림을 PNG 로 바로 저장한다.
        탐색기에서 '복사'한 이미지 파일이면 그 원본을 그대로 연다(저장은 원본을 덮어쓰지 않는다).
        원격 웹 모드가 서버 PC 의 클립보드를 읽으면 안 되므로 _WEB_METHODS 에 넣지 않는다.

        반환: {"path"} | {"empty": true, "error"} | {"error"}
        """
        from core.clipboard_paste import (
            ClipboardPasteError, pick_clipboard_image_file, store_clipboard_bytes,
        )
        try:
            from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                return json.dumps({"error": "클립보드를 읽을 수 없습니다"}, ensure_ascii=False)
            clipboard = app.clipboard()
            mime = clipboard.mimeData()
            if mime is not None and mime.hasUrls():
                local = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
                picked = pick_clipboard_image_file(local)
                if picked:
                    return json.dumps({"path": picked.replace('\\', '/')}, ensure_ascii=False)
            image = clipboard.image()
            if image.isNull():
                return json.dumps({"empty": True, "error": "클립보드에 이미지가 없습니다"}, ensure_ascii=False)
            data = QByteArray()
            buffer = QBuffer(data)
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            ok = image.save(buffer, "PNG")
            buffer.close()
            if not ok:
                return json.dumps({"error": "클립보드 이미지를 PNG 로 바꾸지 못했습니다"}, ensure_ascii=False)
            saved = store_clipboard_bytes(bytes(data))
            return json.dumps({"path": saved.replace('\\', '/')}, ensure_ascii=False)
        except ClipboardPasteError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except Exception as e:
            logger.warning('system clipboard paste failed: %s', e)
            from core.error_handler import sanitize_for_ui
            return json.dumps({"error": sanitize_for_ui(e)}, ensure_ascii=False)

    def _editor_autosave_json(self, path: str, overlay_base64: str, overlay_opacity: float,
                              request_id: str, ticket: int) -> str:
        """크래시 복구본 한 건을 쓰고 결과 JSON 을 돌려준다(호출 스레드에서) — requestEditorAutoSave 의 본체.

        병합 안 한 드로잉 레이어(``overlay_base64``, 불투명도 0~100)가 오면 수동 저장처럼
        합성해서 쓴다 — 예전엔 확정 이미지 파일만 복사해 그린 것이 복구본에서 빠졌다.
        쓰기 순서·폐기와의 경합은 core.editor_autosave 가 맡는다(``ticket`` = 요청 시점 세대).
        """
        try:
            from pathlib import Path
            from core.editor_autosave import AUTOSAVE
            clean = _normalize_vue_path(path)
            if not clean:
                return json.dumps({'requestId': request_id, 'error': '자동저장할 이미지 경로가 올바르지 않습니다'},
                                  ensure_ascii=False)
            try:
                opacity = float(overlay_opacity) / 100.0   # editor_save_actions 와 같은 0~100 규약
            except (TypeError, ValueError):
                opacity = 1.0
            result = AUTOSAVE.write(
                str(Path(clean)), overlay_base64 if isinstance(overlay_base64, str) else None, opacity, ticket)
            return json.dumps({'requestId': request_id, **result}, ensure_ascii=False)
        except Exception as e:
            from core.error_handler import sanitize_for_ui
            logger.warning('editor autosave failed: %s', e)
            return json.dumps({'requestId': request_id, 'error': sanitize_for_ui(str(e), 200)},
                              ensure_ascii=False)

    @pyqtSlot(str, str, float, str)
    def requestEditorAutoSave(self, path: str, overlay_base64: str = '', overlay_opacity: float = 100.0,
                              request_id: str = ''):
        """현재 편집 중인 이미지를 크래시 복구본으로 — 워커 스레드에서 쓰고 editorAutoSaveReady 로 알린다.

        예전 동기 슬롯(editorAutoSave)은 레이어 합성(디코드·합성·PNG 인코딩·fsync)을 GUI 스레드에서
        해서, 5분 타이머가 돌 때마다 창(웹 모드는 모든 클라이언트)이 그리는 도중에도 멈췄다
        (CPU 기준 2048² 0.6초, 4K 1초 이상). 슬롯은 세대만 받아 두고 곧바로 돌아온다.
        """
        from core.editor_autosave import AUTOSAVE
        path = str(path or '')
        overlay = overlay_base64 if isinstance(overlay_base64, str) else ''
        request_id = str(request_id or '')
        # 요청 시점의 세대 — 이 뒤에 온 폐기(저장 성공 → editorClearAutoSave)가 이 쓰기를 무효로 한다
        ticket = AUTOSAVE.ticket()
        self._run_async_lookup(
            f'editor-autosave:{request_id}',
            lambda: self._editor_autosave_json(path, overlay, overlay_opacity, request_id, ticket),
            self.editorAutoSaveReady,
        )

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
        """복구본 폐기.

        자동저장은 워커에서 쓴다 — 폐기 전에 시작한 쓰기가 폐기 뒤에 끝나 이미 저장한 작업의
        복구본을 되살리지 않게 세대를 먼저 올린다(쓰기 잠금은 기다리지 않는다 — GUI 스레드).
        """
        try:
            from core.editor_autosave import AUTOSAVE, remove_autosave_files
            AUTOSAVE.invalidate()
            return json.dumps({"cleared": remove_autosave_files()})
        except Exception:
            return json.dumps({"cleared": False})

    @pyqtSlot(result=str)
    def editorRecoverAutoSave(self) -> str:
        """복구본을 에디터 작업 폴더로 복사해 **그 사본** 경로를 준다.

        복구본(_autosave_session.png)을 그대로 열면 그 파일이 undo 히스토리의 바닥이자
        편집 중인 이미지가 된다. 그러면 저장 뒤 복구본 정리(editorClearAutoSave)가 편집
        중인 그림을 지우고(다음 편집·저장이 '경로가 없다'로 실패), 5분마다 도는 자동저장은
        그 파일을 새 편집으로 덮어써 undo 바닥을 바꾼다. 작업 사본을 따로 둔다.
        """
        try:
            import tempfile
            from core.editor_save import snapshot_file
            src = os.path.join(tempfile.gettempdir(), "AIStudioPro_editor", "_autosave_session.png")
            if not os.path.isfile(src):
                return json.dumps({"error": "복구할 작업이 없습니다"}, ensure_ascii=False)
            copy = snapshot_file(src, self._editor_temp_dir, prefix="recovered")
            return json.dumps({"path": copy.replace('\\', '/')}, ensure_ascii=False)
        except Exception as e:
            logger.warning("editor autosave recovery failed: %s", e)
            return json.dumps({"error": f"복구 실패: {e}"}, ensure_ascii=False)

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
            from core.ollama_client import OllamaClient, clear_thinking_mode_cache
            url = base_url.strip() if base_url.strip() else DEFAULT_OLLAMA_URL
            client = OllamaClient(base_url=url)
            models = client.list_models()
            # 목록을 다시 읽는 때(Settings 연결 테스트·자동 로드, 부팅, 대화·캡션 탭) = 재연결·pull 뒤 —
            # 그 서버의 think 능력·거부 기억을 비워 다음 요청이 /api/show 를 새로 읽게 한다
            clear_thinking_mode_cache(url)
            return json.dumps(models)
        except Exception as e:
            print(f"[Ollama] 모델 목록 조회 오류: {e}")
            return json.dumps([])

    @pyqtSlot(str)
    def requestOllamaModels(self, base_url: str = ''):
        """Ollama 모델 목록을 백그라운드에서 조회한다(결과: ollamaModelsReady {url, models}).

        동기 ollamaListModels 슬롯은 없앴다 — 클라이언트가 준 주소로 HTTP(timeout 5초)를 GUI
        스레드에서 돌려 웹 facade 호출 한 번이 GUI 와 모든 WebSocket 클라이언트를 멈췄다(Codex R3 #3).
        """
        url = base_url.strip() if base_url.strip() else DEFAULT_OLLAMA_URL

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
        데스크톱 Qt 모드도 바인딩 직후 이것을 당겨 간다 — 예전엔 부팅 1초 타이머의 1회 emit 에
        기대다가 JS 가 connect 하기 전에 터지면 영구히 유실됐다(감사 #107). 레거시 마이그레이션은
        GeneratorMainUI.__init__ 가 동기로 먼저 끝낸다(_apply_saved_configs).
        """
        from core.storage_paths import config_file

        def _load(name, default):
            try:
                path = str(config_file(name))
                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning("initial config load failed (%s): %s", name, e)
            return default

        # tabDefaults 는 싣지 않는다 — 소비자(bridge.js _requestInitialConfig)는 uiPrefs·condRules·
        # globalWeights 만 읽고, 기본값이 필요한 화면은 getTabDefaults 를 직접 부른다(audit #140).
        return json.dumps({
            'uiPrefs': self._ui_prefs_dict(),
            'condRules': _load('cond_rules.json', {'positive': [], 'negative': []}),
            'globalWeights': _load('global_weights.json', []),
        }, ensure_ascii=False)

    @pyqtSlot(result=str)
    def getAutomationSettings(self) -> str:
        """현재 백엔드 모드의 자동화 설정(core.mode_aware_automation) — Vue 가 마운트 때 당겨 간다.

        Vue 는 이 값으로 자동화 패널을 채운 뒤에야 set_automation_settings 로 동기화한다. 예전엔
        하드코딩 기본값을 먼저 보내 automation_settings_<mode>.json 을 부팅마다 덮었다(감사 #42).
        클라이언트별 응답이라 다른 웹 클라이언트의 화면을 건드리지 않는다.
        """
        try:
            persistence = getattr(self.parent(), 'automation_persistence', None)
            if persistence is None:
                return '{}'
            return json.dumps(persistence.snapshot(), ensure_ascii=False)
        except Exception as e:
            logger.warning("getAutomationSettings failed: %s", e)
            return '{}'

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
        """wildcards/ 파일 목록 — [{name, file, tags, lines}]. 파일 관리 규칙(주석·이름)은
        utils.file_wildcard.FileWildcardManager 한 곳에 있다(생성 시 해석기와 같은 규칙).
        lines 는 주석(#)까지 담은 원문이라 Vue 에서 저장해도 주석이 지워지지 않는다."""
        try:
            from utils.file_wildcard import get_file_wildcard_manager
            return json.dumps(get_file_wildcard_manager().get_wildcard_tree())
        except Exception as e:
            logger.warning("getWildcardTree failed: %s", e)
            return json.dumps([])

    vramUpdated = pyqtSignal(str)  # JSON {used, total, pct}

    @pyqtSlot(result=str)
    def getPresetList(self) -> str:
        """생성 프리셋 이름 목록(presets/*.json)."""
        from core.generation_presets import list_presets
        return json.dumps(list_presets(), ensure_ascii=False)

    @pyqtSlot(str, result=str)
    def getPresetData(self, name: str) -> str:
        """프리셋 미리보기 — 불러오기가 실제로 적용하는 키(core.generation_presets.PRESET_KEYS)만.

        예전엔 파일 전체(백엔드 URL·단축키·테마 등 82키)를 보여 줘서 미리보기와 불러오기가 달랐다.
        읽기 경로도 저장/삭제와 같은 이름 정규화를 거친다(traversal 차단, 라운드트립 일치).
        """
        try:
            from core.generation_presets import read_preset
            return json.dumps(read_preset(name) or {}, ensure_ascii=False)
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

    @pyqtSlot(str, str)
    def requestCharacterTagsOnline(self, name: str, request_id: str = ''):
        """danbooru 표본에서 캐릭터의 실제 공통 general 태그를 집계한다(로컬 DB가 틀린/없는 캐릭터 보완).

        HTTPS 최대 2회라 워커 스레드에서 돌리고 characterTagsOnlineReady 로 알린다 — 예전 동기 슬롯
        (fetchCharacterTagsOnline)은 GUI 스레드에서 최대 24초 멈췄다. 결과에는 요청 id 와 이름을
        되돌려 줘서, 그사이 캐릭터를 바꾼 프론트가 옛 결과를 버릴 수 있게 한다. 실패도 항상 신호로
        온다(core/danbooru_character_tags.py 가 예외를 내지 않는다).
        """
        name = str(name or '')
        request_id = str(request_id or '')

        def _load() -> str:
            try:
                from core.danbooru_character_tags import fetch_character_tags_online
                result = fetch_character_tags_online(name)
            except Exception as exc:
                result = {"error": str(exc)}
            return json.dumps({**result, "requestId": request_id, "name": name}, ensure_ascii=False)

        # 요청마다 키가 달라 중복 제거로 버려지지 않는다(다른 캐릭터·다시 누르기 모두 응답을 받는다).
        self._run_async_lookup(f'danbooru-tags:{request_id}:{name}', _load, self.characterTagsOnlineReady)

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

    def getLoras(self, mode: str = '') -> str:
        """활성 API 메타데이터 + 메인/보조 디스크 LoRA 카탈로그 반환(호출 스레드에서, 동기).

        QWebChannel 슬롯이 아니다 — 캐시가 빗나가면 백엔드 HTTP(timeout 10초)와 디스크 병합을
        호출 스레드에서 하므로, GUI 스레드에서 부르는 프론트는 비동기 짝 :meth:`requestLoras`
        (→ ``lorasReady``)를 쓴다. 이 메서드는 파이썬 쪽 동기 진입점으로 남긴다(캐시 계약 테스트).

        본문은 requestLoras 워커와 같은 ``ui.lora_catalog_cache.load_catalog_json`` 이다 — 캐시
        알고리즘 사본을 따로 두면 테스트가 운영에서 돌지 않는 사본을 고정하고, 사본에는 무효화
        경합(백엔드 전환) 보호가 빠져 둘이 조용히 갈라졌다. raw 캐시(``lora_catalog_cache.raw_loras()``)에는
        백엔드의 raw 응답만 보관한다(병합 뷰 모델은 ``_merged_lora_cache``). ``self`` 가 None 이면 병합 캐시 없이 동작한다.
        """
        try:
            from ui.lora_catalog_cache import load_catalog_json

            return load_catalog_json(self, str(mode or ''))
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(str, str)
    def requestLoras(self, mode: str = '', request_id: str = ''):
        """getLoras 의 비동기 짝 — LoRA 매니저가 쓴다. 결과는 lorasReady.

        캐시가 빗나가면(첫 열기·force·백엔드 전환 뒤) 백엔드 HTTP(timeout 10초)와 디스크 카탈로그
        병합(os.walk·해시)을 하므로 워커 스레드에서 한다 — 동기 getLoras 는 GUI 스레드를 수 초 막았다.
        캐시 갱신은 ui.lora_catalog_cache.load_catalog_json 이 경합(백엔드 전환 무효화)에 안전하게 한다.
        요청마다 키가 달라 '다시 스캔'(force)이 앞선 요청에 합쳐져 사라지지 않는다.
        """
        mode = str(mode or '')
        request_id = str(request_id or '')
        head = json.dumps({"requestId": request_id, "mode": mode}, ensure_ascii=False)[:-1]

        def _load() -> str:
            try:
                from ui.lora_catalog_cache import load_catalog_json
                encoded = load_catalog_json(self, mode)
                # 카탈로그 JSON 을 다시 파싱하지 않고 봉투에 그대로 끼운다(수천 개 목록도 싸게).
                return f'{head}, "loras": {encoded}}}'
            except Exception as exc:
                logger.warning("requestLoras failed: %s", exc)
                return json.dumps({"requestId": request_id, "mode": mode, "error": str(exc)},
                                  ensure_ascii=False)

        self._run_async_lookup(f'loras:{request_id}:{mode}', _load, self.lorasReady)

    @pyqtSlot(result=str)
    def getStatusMessage(self) -> str:
        """마지막 show_status 한 줄(JSON {text, level, timeoutMs, at}, 없으면 {}).

        statusMessage 는 push 라 Vue 가 뜨기 전(설정 불러오기 실패 등)·웹 클라이언트가 붙기 전에 보낸
        문구는 사라진다 — StatusStrip 이 마운트될 때 한 번 읽어 남은 시간만큼 보인다(ui/status_line.py).
        """
        payload = getattr(self, '_last_status_payload', None)
        return json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else '{}'

    @pyqtSlot(str, result=str)
    def saveSession(self, payload_json: str) -> str:
        """세션 상태(탭/프롬프트 등)를 cache/session에 저장 (크래시 복구용).

        편집 중 백업은 늘 ``clean: false`` 로 쓰고, 정상 종료만 clean 을 세운다
        (core/session_backup.py — 다음 부팅의 복구 제안 판단)."""
        try:
            from core.session_backup import write_session_backup
            payload = json.loads(payload_json or '{}')
            write_session_backup(payload)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @pyqtSlot(result=str)
    def getSession(self) -> str:
        """저장된 세션 상태 반환 (없으면 {})."""
        try:
            from core.session_backup import read_session_backup
            return json.dumps(read_session_backup(), ensure_ascii=False)
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

    # 와일드카드 파일 관리 — FileWildcardManager 의 얇은 래퍼. 이름 규칙(sanitize·'.txt' 접미)과
    # 캐시 무효화가 생성 시 해석기와 한 곳에 있어야 저장한 파일을 해석기가 같은 이름으로 찾는다.
    # 응답에 실제 저장 이름(name)을 담는다 — sanitize 로 바뀐 이름을 Vue 가 그대로 쓰게.
    @pyqtSlot(str, str, result=str)
    def saveWildcard(self, filename: str, content: str) -> str:
        """와일드카드 파일 저장/수정 (주석·빈 줄 포함 원문 그대로)"""
        try:
            from utils.file_wildcard import get_file_wildcard_manager
            name = get_file_wildcard_manager().save_wildcard(filename, content)
            return json.dumps({'ok': True, 'name': name})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def createWildcard(self, name: str) -> str:
        """새 빈 와일드카드 파일 ('+ NEW') — 같은 파일이 이미 있으면(대소문자만 다름·이름 규칙으로
        같아짐 포함) 비우지 않고 created=False 와 그 파일의 이름을 돌려준다. {ok, name, created}"""
        try:
            from utils.file_wildcard import get_file_wildcard_manager
            result = get_file_wildcard_manager().create_wildcard(name)
            return json.dumps({'ok': True, **result})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def deleteWildcard(self, filename: str) -> str:
        """와일드카드 파일 삭제 — 목록의 이름 그대로인 파일을 지운다. 지우지 못하면 error(Vue 가 목록을 둔다)"""
        try:
            from utils.file_wildcard import get_file_wildcard_manager
            get_file_wildcard_manager().delete_wildcard(filename)
            return json.dumps({'ok': True})
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, str, result=str)
    def renameWildcard(self, old_name: str, new_name: str) -> str:
        """와일드카드 파일 이름 변경 — 같은 이름의 파일이 있거나 옛 파일이 없으면 바꾸지 않고 error"""
        try:
            from utils.file_wildcard import get_file_wildcard_manager
            name = get_file_wildcard_manager().rename_wildcard(old_name, new_name)
            return json.dumps({'ok': True, 'name': name})
        except Exception as e:
            return json.dumps({'error': str(e)})

    def _get_tag_classifier(self):
        """프로세스 공유 TagClassifier — 메인 창(GeneratorBase.tag_classifier)과 같은 객체.

        예전엔 브리지가 따로 하나 더 만들어 태그 그룹 parquet 재적재(약 0.5초 GUI 정지)와
        이름 사전 복사를 한 번 더 치렀다.
        """
        from core.tag_classifier import get_tag_classifier
        return get_tag_classifier()

    def _exclude_vocabulary(self) -> set:
        """제외 규칙 미리보기용 태그 사전(소문자·밑줄형). 처음 한 번만 만든다.

        지역 집합에 끝까지 채운 뒤 한 번에 공개한다 — 도중 예외가 나도 반쪽 사전이
        캐시로 남지 않는다.
        """
        vocabulary = getattr(self, '_all_tags_set', None)
        if vocabulary is not None:
            return vocabulary

        vocabulary = set()
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
                vocabulary.update(
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
            vocabulary.update(
                str(tag).strip().lower().replace(' ', '_')
                for tag in catalog['tag'].dropna()
                if str(tag).strip()
            )
        except Exception as e:
            logger.warning("Korean tag catalog scan failed: %s", e)
        # 통합 Wiki 그룹의 실제 tag 열만 수집
        try:
            vocabulary.update(
                tag.lower().replace(' ', '_')
                for tag in database.all_group_tags()
            )
        except Exception as e:
            logger.warning("tag group scan failed: %s", e)
        # TagClassifier의 tag_to_category + character/copyright/artist 사전 (공유 인스턴스)
        try:
            tc = self._get_tag_classifier()
            vocabulary.update(
                tag.lower().replace(' ', '_')
                for tag in getattr(tc, 'tag_to_category', {})
            )
            for names in (
                getattr(tc, 'characters', ()),
                getattr(tc, 'copyrights', ()),
                getattr(tc, 'artists', ()),
            ):
                vocabulary.update(t.lower().replace(' ', '_') for t in names)
        except Exception as e:
            logger.debug("TagClassifier vocabulary load failed: %s", e)
        print(f"[Exclude] Tag DB loaded: {len(vocabulary)} tags")
        self._all_tags_set = vocabulary
        return vocabulary

    # 규칙별 결과 캐시 상한 — 사용자가 입력한 규칙 수 정도면 충분하다
    _EXCLUDE_MATCH_CACHE_MAX = 256

    @pyqtSlot(str, result=str)
    def getExcludeMatches(self, rule: str) -> str:
        """제외 규칙에 매칭되는 태그 목록 반환 (tags_db 기반).

        GUI 스레드 슬롯이다. 태그 사전(약 77만 개)은 처음 한 번만 만들고, 규칙별 결과를
        기억해 같은 규칙을 다시 눌러도 사전을 다시 훑지 않는다. 완전 일치(*태그)는
        집합 조회 한 번으로 끝난다(core.exclude_rule_match). 규칙 종류는 프롬프트 적용과
        같은 파서(core.exclude_rules)가 가린다.
        """
        try:
            from core.exclude_rule_match import match_exclude_rule, normalize_exclude_rule

            # 캐시 키 = 파싱된 규칙 (비교 방식:키워드). 지우는 태그가 없는 규칙(빈 칸·~유지)은 ""
            key = normalize_exclude_rule(rule)
            if not key:
                return json.dumps([])

            cache = getattr(self, '_exclude_match_cache', None)
            if cache is None:
                cache = self._exclude_match_cache = {}
            cached = cache.get(key)
            if cached is not None:
                return cached

            result = json.dumps(match_exclude_rule(rule, self._exclude_vocabulary()))
            cache[key] = result
            while len(cache) > self._EXCLUDE_MATCH_CACHE_MAX:
                cache.pop(next(iter(cache)))
            return result
        except Exception as e:
            return json.dumps({'error': str(e)})

    @pyqtSlot(str, result=str)
    def deepCleanPrompt(self, prompt_json: str) -> str:
        """딥 프롬프트 클리너: 메인 태그의 중복 제거 + 충돌 감지 + 순서 재배치.

        페이로드 ``{prompt: 메인 태그, context?: [다른 칸 텍스트…]}``. context 에 이미 있는
        태그는 메인에서 빼고, 충돌 검사는 메인∪context 로 한다(core.prompt_deep_clean).
        """
        try:
            from core.prompt_deep_clean import deep_clean_request
            data = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
            return json.dumps(deep_clean_request(data), ensure_ascii=False)
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

            # TagClassifier 시도 (메인 창과 공유하는 인스턴스)
            tc = None
            try:
                tc = self._get_tag_classifier()
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

    def _compare_gif_json(self, before_path: str, after_path: str, duration: int, loops: int,
                          request_id: str = '') -> str:
        """비교 GIF 한 건을 만들어 결과 JSON 을 돌려준다(호출 스레드에서) — requestCompareGif 의 본체.

        결과 파일은 앱 폴더의 gif/ 에 새 이름으로만 쓴다(core/compare_gif.py). 실패도 JSON 으로
        돌려준다 — 신호가 안 오면 프론트의 'GIF 생성 중'이 풀리지 않는다.
        """
        try:
            clean_before = _normalize_vue_path(before_path)
            clean_after = _normalize_vue_path(after_path)
            if not clean_before or not clean_after:
                return json.dumps({'requestId': request_id, 'error': '비교 이미지 경로가 올바르지 않습니다'},
                                  ensure_ascii=False)
            from core.compare_gif import export_compare_gif
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gif')
            result = export_compare_gif(clean_before, clean_after, duration, loops, out_dir)
            return json.dumps({'requestId': request_id, **result}, ensure_ascii=False)
        except Exception as e:
            from core.error_handler import sanitize_for_ui
            logger.warning("compare GIF export failed: %s", e)
            return json.dumps({'requestId': request_id, 'error': sanitize_for_ui(str(e), 200)},
                              ensure_ascii=False)

    @pyqtSlot(str, str, int, int, str)
    def requestCompareGif(self, before_path: str, after_path: str, duration: int, loops: int,
                          request_id: str = ''):
        """Before/After 비교 GIF — 워커 스레드에서 만들고 compareGifReady 로 알린다.

        예전 동기 슬롯(exportCompareGif)은 풀해상도 LANCZOS·16프레임 blend·optimize 저장을 GUI
        스레드에서 해 창이 수 초 멈췄다. 이제 긴 변 1024px 로 줄여 RGB 로 맞춘 뒤 만든다.
        """
        before_path, after_path = str(before_path or ''), str(after_path or '')
        request_id = str(request_id or '')
        self._run_async_lookup(
            f'compare-gif:{request_id}',
            lambda: self._compare_gif_json(before_path, after_path, duration, loops, request_id),
            self.compareGifReady,
        )

    @pyqtSlot(result=str)
    def getTabDefaults(self) -> str:
        """tab_defaults.json(정규화된 알려진 키만) — Settings 기본값 패널·I2I·에디터가 읽는다."""
        try:
            from core.tab_defaults import load_tab_defaults
            return json.dumps(load_tab_defaults(), ensure_ascii=False)
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

    @pyqtSlot()
    def requestADetailerModels(self):
        """ADetailer 모델 목록을 백그라운드에서 조회한다(결과: adetailerModelsReady — 프록시 목록은
        __init__ 에서 이은 _apply_adetailer_models_json 이 GUI 스레드에서 채운다).

        HTTP(timeout 5초)를 GUI 스레드에서 돌리던 동기 getADetailerModels 슬롯은 없앴다(Codex R3 #3).
        """
        self._run_async_lookup(
            'adetailer-models',
            self._load_adetailer_models_json,
            self.adetailerModelsReady,
        )

    @pyqtSlot(result=str)
    def getYoloModelLabel(self) -> str:
        """YOLO 모델 라벨 반환 (Editor_models/ 자동 감지 − '초기화'로 비활성된 모델)"""
        try:
            from core import yolo_models
            return yolo_models.model_label()
        except Exception:
            return "No Model Loaded"

    @pyqtSlot(result=str)
    def refreshYoloModels(self) -> str:
        """editor_models/ 재스캔 후 라벨 반환"""
        label = self.getYoloModelLabel()
        self.yoloModelUpdated.emit(label)
        return label

    def warmTagSuggestions(self) -> None:
        """자동완성 데이터를 백그라운드에서 미리 적재 — 첫 키 입력의 2초 공백을 없앤다.

        TagCompleter(태그 DB 2.2초) + 한국어 카탈로그(0.5초). 앱 기동 후 메인이 한 번 부른다.
        (테스트에서는 부르지 않는다 — 전역 싱글톤을 백그라운드에서 건드리면 가짜 DB 패치와 엉킨다.)

        싱글턴 팩토리와 적재(_ensure)는 락으로 한 번만 돌고, 예열이 도는 동안 GUI 슬롯은
        락을 기다리지 않는다(_tag_warm_active) — 예열로 없애려던 UI 정지를 다시 만들지 않게.
        """
        def _warm():
            try:
                from utils.tag_completer import get_tag_completer
                get_tag_completer()
                from core.tag_korean_lookup import get_korean_tag_lookup
                get_korean_tag_lookup().label("solo")
            except Exception as e:
                print(f"[TagSuggest] warm-up skipped: {e}")

        thread = threading.Thread(target=_warm, name="tag-suggest-warm", daemon=True)
        self._tag_warm_thread = thread
        thread.start()

    def _tag_warm_active(self) -> bool:
        """자동완성 예열 스레드가 아직 적재 중인가."""
        thread = getattr(self, '_tag_warm_thread', None)
        return thread is not None and thread.is_alive()

    @pyqtSlot(str, result=str)
    def getTagSuggestionsRich(self, prefix: str) -> str:
        """자동완성 후보 + 한국어 라벨 — [{tag, ko, category, desc, count}].

        영문 접두사는 기존 완성기(TagCompleter) 결과에 한국어 라벨만 붙이고,
        한글이 섞인 질의는 한국어 키워드/설명 색인에서 찾는다("장발" → long hair).
        자동완성의 유일한 슬롯이다(문자열 목록만 주던 옛 getTagSuggestions 는 호출자가 없어 제거).

        GUI 스레드 슬롯이라 예열 스레드의 적재를 기다리지 않는다: 완성기가 적재 중이면
        빈 목록, 한국어 카탈로그만 적재 중이면 라벨 없이 영문 후보만 돌려준다(다음 키
        입력에서 정상 결과). 예열이 없을 때는 예전처럼 이 스레드에서 적재한다.
        """
        try:
            from core.tag_korean_lookup import get_korean_tag_lookup, has_hangul
            lookup = get_korean_tag_lookup()
            labels_pending = self._tag_warm_active() and not lookup.is_ready()
            if has_hangul(prefix):
                if labels_pending:
                    return json.dumps([])
                return json.dumps(lookup.search(prefix, limit=10), ensure_ascii=False)
            from utils.tag_completer import try_get_tag_completer
            completer = try_get_tag_completer()
            if completer is None:
                return json.dumps([])
            tags = completer.get_suggestions(prefix, max_count=10)
            if labels_pending and not lookup.is_ready():
                return json.dumps([
                    {"tag": tag, "ko": "", "category": "", "desc": "", "count": 0}
                    for tag in tags
                ], ensure_ascii=False)
            return json.dumps(lookup.labels(tags), ensure_ascii=False)
        except Exception as e:
            import traceback
            print(f"[getTagSuggestionsRich] 오류: {e}")
            traceback.print_exc()
            return json.dumps([])

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

    # ── 캡션 저장 폴더 승인 (core/caption_out_dir.py) ──
    # outDir 는 클라이언트가 보내는 값이다. 호스트의 폴더 대화상자가 돌려준 폴더만 승인하고
    # (generator_main 의 caption_pick_outdir → approve_caption_out_dir), 캡션 슬롯은 승인된
    # 폴더에만 .txt 를 읽고 쓴다. 아래는 슬롯이 아닌 파이썬 전용 메서드라 QWebChannel·웹
    # facade 에 나가지 않는다 — 웹 클라이언트가 스스로 폴더를 승인할 길이 없다.

    def approved_caption_out_dirs(self) -> list:
        """승인된 캡션 저장 폴더 목록(최근 것 먼저). 첫 호출 때 ui_prefs 의 서버 전용 키에서 읽는다."""
        approvals = getattr(self, '_caption_out_dir_approvals', None)
        if approvals is None:
            from core.caption_out_dir import APPROVED_PREFS_KEY, normalize_approved
            from core.ui_prefs import read_ui_prefs
            approvals = normalize_approved(read_ui_prefs().get(APPROVED_PREFS_KEY))
            self._caption_out_dir_approvals = approvals
        return list(approvals)

    def approve_caption_out_dir(self, folder) -> str:
        """호스트 대화상자가 돌려준 폴더를 승인하고 ui_prefs 에 남긴다. → 승인한 절대 경로.

        시스템 폴더·드라이브 루트·없는 폴더면 CaptionOutDirError(사용자에게 보여도 되는 메시지).
        빈 값은 승인할 것이 없어 ''.
        """
        from core.caption_out_dir import remember_approved, resolve_caption_out_dir
        resolved = resolve_caption_out_dir(folder)
        if not resolved:
            return ''
        approvals = remember_approved(self.approved_caption_out_dirs(), resolved)
        self._caption_out_dir_approvals = approvals
        self._persist_caption_out_dir_approvals(approvals)
        return resolved

    @staticmethod
    def _persist_caption_out_dir_approvals(approvals) -> None:
        try:
            from core.caption_out_dir import APPROVED_PREFS_KEY
            from core.config_migration import load_ui_prefs, save_ui_prefs
            from core.ui_prefs import ui_prefs_path
            prefs_path = ui_prefs_path()
            prefs = load_ui_prefs(prefs_path)
            prefs[APPROVED_PREFS_KEY] = list(approvals)
            save_ui_prefs(prefs_path, prefs)
        except Exception as exc:
            # 파일에 못 남겨도 이번 실행 동안은 메모리 승인으로 쓴다.
            logger.warning('caption output folder approval save failed: %s', exc)

    def seed_caption_out_dir_approval_from_prefs(self) -> bool:
        """(데스크톱 진입점 전용, 한 번) 이 규칙 전에 대화상자로 고른 captionOutDir 을 승인으로 옮긴다.

        승인 키가 이미 있으면(빈 목록 포함) 아무것도 하지 않는다 — 옮긴 뒤 키를 남겨 다시 하지 않는다.
        웹 모드에서는 하지 않는다. 이 규칙 뒤의 captionOutDir 은 save_ui_prefs 가 승인된 폴더만
        받으므로(filter_client_caption_prefs) 웹 클라이언트가 심어 둔 값이 여기서 승인되지 않는다.
        """
        if self._backend_runtime_is_web_mode():
            return False
        from core.caption_out_dir import APPROVED_PREFS_KEY, OUT_DIR_PREFS_KEY, CaptionOutDirError
        from core.ui_prefs import read_ui_prefs
        prefs = read_ui_prefs()
        if not prefs or APPROVED_PREFS_KEY in prefs:
            return False
        legacy = str(prefs.get(OUT_DIR_PREFS_KEY) or '').strip()
        if legacy:
            try:
                if self.approve_caption_out_dir(legacy):
                    return True
            except CaptionOutDirError as exc:
                # 지워졌거나 시스템 폴더 — 승인하지 않는다(쓰려 하면 다시 고르라는 오류가 난다).
                logger.info('legacy caption output folder not approved: %s', exc)
        # 빈 목록이라도 키를 남겨 다음 실행부터는 이식을 다시 하지 않는다.
        self._caption_out_dir_approvals = self.approved_caption_out_dirs()
        self._persist_caption_out_dir_approvals(self._caption_out_dir_approvals)
        return False

    def _resolve_caption_out_dir(self, raw) -> str:
        """캡션 슬롯 공용 — 이미 있고 호스트가 승인한 폴더만(core/caption_out_dir.py)."""
        from core.caption_out_dir import resolve_caption_out_dir
        return resolve_caption_out_dir(raw, approved=self.approved_caption_out_dirs())

    def _check_caption_target(self, txt_path: str) -> None:
        """웹 모드에서 앱 설치 폴더 안(생성 이미지 폴더 제외) .txt 이거나 설치 목록 이름
        (requirements*·*constraints*·CMakeLists) .txt 대상이면 CaptionOutDirError.

        이미지 옆 사이드카 경로로도 이 앱의 requirements.txt·config/*.txt(두 번째 벽)나 다른 파이썬 앱
        폴더의 requirements.txt(세 번째 벽 — renameFile 로 stem 을 고를 수 있다)를 읽거나 덮어쓰지
        못하게 한다. 데스크톱 페이지는 신뢰 경계 안이라 그대로 쓴다(core/caption_out_dir.py).
        """
        if not self._backend_runtime_is_web_mode():
            return
        from config import OUTPUT_DIR
        from core.caption_out_dir import (
            MANIFEST_TARGET_MESSAGE,
            PROTECTED_TARGET_MESSAGE,
            CaptionOutDirError,
            is_manifest_caption_target,
            is_protected_caption_target,
        )
        from core.storage_paths import PROJECT_ROOT
        if is_protected_caption_target(txt_path, app_root=PROJECT_ROOT, allowed_roots=(OUTPUT_DIR,)):
            raise CaptionOutDirError(PROTECTED_TARGET_MESSAGE)
        if is_manifest_caption_target(txt_path):
            raise CaptionOutDirError(MANIFEST_TARGET_MESSAGE)

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
        """중단 시 반쪽짜리 sidecar를 남기지 않는 동일 폴더 원자 저장.

        공용 구현(utils.atomic_json.atomic_write_text: fsync + 실패 시 tmp 정리)에 위임한다.
        newline='' — 캡션 본문의 개행을 그대로 쓴다(Windows CRLF 변환 없음).
        """
        from utils.atomic_json import atomic_write_text
        atomic_write_text(path, text, newline='')

    def _prepare_caption_payload(self, raw_payload: dict, *, batch: bool) -> tuple[dict, list[str]]:
        """외부 payload의 경로·모드·숫자를 검증하고 정규화한다."""
        import math
        import os
        import uuid
        from core.caption_out_dir import CaptionOutDirError

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
        payload['url'] = str(payload.get('url') or DEFAULT_OLLAMA_URL).strip().rstrip('/')
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
            # 수동 저장·불러오기와 같은 규칙 — 이미 있고 호스트 대화상자가 승인한 폴더만 받고
            # 만들지 않는다(core/caption_out_dir.py). 아니면 다시 고르라는 한국어 오류로 시작 전에 멈춘다.
            out_dir = self._resolve_caption_out_dir(out_dir)
        elif out_dir:
            # 저장하지 않으면 outDir 는 표시용 txtPath 에만 쓰인다. 검증 안 된 원문으로 경로를
            # 만들지 않게(폴더 존재 여부도 알려 주지 않게) 통과할 때만 남기고, 아니면 무시한다.
            try:
                out_dir = self._resolve_caption_out_dir(out_dir)
            except CaptionOutDirError:
                out_dir = ''
        payload['outDir'] = out_dir

        if payload['save']:
            targets: dict[str, str] = {}
            for image_path in files:
                target = self._caption_txt_path(image_path, out_dir)
                # 기존 .txt 를 읽는(건너뛰기) 경로와 쓰는 경로 모두 이 대상만 쓴다.
                self._check_caption_target(target)
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
            ollama_base_url=payload.get('url') or DEFAULT_OLLAMA_URL,
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
            ollama_base_url=payload.get('url') or DEFAULT_OLLAMA_URL,
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

        url = str(payload.get('url') or DEFAULT_OLLAMA_URL).rstrip('/')
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

    # 동기 captionImage 슬롯은 없앴다(Codex R3 재검토 #3) — 프론트 호출자가 없는 하위 호환 슬롯이었는데
    # 웹 facade 로 부르면 GUI 스레드에서 Ollama HTTP(클라이언트가 준 url·최대 900초)·CAFormer ONNX
    # 추론을 돌려 그동안 창과 모든 WebSocket 클라이언트가 멈췄다. 1장도 startCaptionBatch(워커 스레드).

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
                    # 진행 중인 '생성 후 언로드'(같은 리스를 쥔다)는 이 작업 스레드에서 기다린다
                    from core.post_generation import reserve_generation_lease
                    lease = (
                        reserve_generation_lease(
                            'caption', coordinator=get_generation_coordinator())
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
                                        p.get('url') or DEFAULT_OLLAMA_URL,
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
            try:
                payload = json.loads(payload_or_path)
            except (TypeError, json.JSONDecodeError):
                payload = {'path': payload_or_path}
            if not isinstance(payload, dict):
                payload = {'path': payload_or_path}
            path = _normalize_vue_path(str(payload.get('path') or ''))
            if not path:
                raise ValueError('허용되지 않은 이미지 경로입니다')
            # 수동 저장·일괄 처리와 같은 규칙(이미 있고 호스트가 승인한 폴더만,
            # 웹 모드면 앱 설치 폴더 안 .txt 금지 — core/caption_out_dir.py).
            out_dir = self._resolve_caption_out_dir(payload.get('outDir'))
            txt = self._caption_txt_path(path, out_dir)
            self._check_caption_target(txt)
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
            p = json.loads(payload_json) if payload_json else {}
            path = _normalize_vue_path(str(p.get('path') or ''))
            if not path:
                raise ValueError('허용되지 않은 이미지 경로입니다')
            save_locked = self._caption_job_lock.acquire(blocking=False)
            if not save_locked:
                raise RuntimeError('캡션 작업 중에는 수동 저장할 수 없습니다')
            # outDir 은 호스트 폴더 대화상자가 승인한 '이미 있는' 폴더다. 수동 저장도 일괄 처리
            # (_prepare_caption_payload)도 같은 규칙으로 폴더를 만들지 않고, 승인 안 된 폴더
            # (웹 클라이언트가 보낸 임의 경로)에는 쓰지 않는다. 웹 모드면 앱 설치 폴더 안 .txt 도
            # 막는다 — 없거나 승인 안 됐으면 다시 고르라는 한국어 오류(core/caption_out_dir.py).
            out_dir = self._resolve_caption_out_dir(p.get('outDir'))
            txt = self._caption_txt_path(path, out_dir)
            self._check_caption_target(txt)
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
            # 표시 그룹(params)도 core 가 파싱한 dict 에서 만든다 — 표시 문자열을 다시 파싱하지 않는다.
            info = read_metadata_for_ui(clean)
            return json.dumps(info, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'error': str(e), 'path': filepath})

    def _parse_params_line(self, params_line: str) -> dict:
        """(호환 래퍼) 파라미터 줄 → 표시 그룹. 따옴표를 아는 core 파서/그룹기를 쓴다.

        getImageExif 는 더 이상 부르지 않지만 tests/test_comfy_metadata_actions.py 하네스가
        클래스 본문에서 이 이름을 바인딩하므로 남겨 둔다.
        """
        from core.image_metadata import group_parameters, parse_parameters_text
        return group_parameters(parse_parameters_text(params_line or ''))

    imageSearchTextsReady = pyqtSignal(str)  # JSON {token, texts: {path: 소문자 검색 텍스트}, error?}

    @pyqtSlot(str)
    def requestImageSearchTexts(self, request_json: str):
        """Gallery·Favorites 'EXIF 검색' — 검색 텍스트만 백그라운드에서 모아 돌려준다.

        예전엔 이미지마다 동기 getImageExif(GUI 스레드, 전체 메타 dict)를 불렀다. 요청은
        ``{"token": str, "paths": [...]}`` (최대 200개). 실패해도 같은 token 으로 반드시 응답한다
        — 프런트가 기다리다 멈추지 않게. (경로, 수정 시각, 크기) 캐시라 'EXIF 저장' 뒤에도 새로 읽는다.
        """
        try:
            request = json.loads(request_json or '{}')
        except (TypeError, ValueError):
            request = {}
        token = str(request.get('token') or '') if isinstance(request, dict) else ''
        paths = request.get('paths') if isinstance(request, dict) else None
        if not isinstance(paths, list):
            paths = []

        def _work():
            from core.metadata_search import SearchTextCache, search_texts_for
            from core.thumb_cache import THUMB_SOURCE_EXTS
            cache = getattr(self, '_search_text_cache', None)
            if cache is None:
                cache = self._search_text_cache = SearchTextCache()
            payload = {'token': token, 'texts': {}}
            try:
                payload['texts'] = search_texts_for(
                    [str(p) for p in paths], cache=cache,
                    is_allowed=lambda p: _normalize_vue_path(p, allowed_exts=THUMB_SOURCE_EXTS))
            except Exception as e:
                logger.warning('image search text failed: %s', e)
                payload['error'] = str(e)
            try:
                self.imageSearchTextsReady.emit(json.dumps(payload, ensure_ascii=False))
            except RuntimeError:
                pass

        threading.Thread(target=_work, daemon=True, name='vue-image-search-texts').start()

