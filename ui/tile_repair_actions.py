"""Anima Tile & Repair — Vue ``tile_repair_*`` 액션의 Python 쪽. sam-extra ``/sam-extra/tile-repair`` 로 보낸다.

확장 패널(sam3ext/ui_anima.py Tile-Repair 모드)과 같은 파이프라인을 JSON 라우트로 부른다(확장
sam3ext/tile_repair_api.py). 기본값·범위는 원본 그대로 — core/tile_repair_request.py 참조.

액션(브리지 계약 — tests/test_bridge_contract.py 가 아래 ``action in (...)`` 의 리터럴을 AST 로 읽는다):
  tile_repair_options {requestId}                            → GET  /sam-extra/tile-repair/options
  tile_repair_run     {requestId, image_path, image, settings} → POST /sam-extra/tile-repair,
                      결과 PNG 는 ``<출력 폴더>/tile_repair`` 에 새 파일로만 쓴다(덮어쓰지 않는다)
  tile_repair_cancel  {requestId}(실행 중인 run 의 id)       → POST /sam-extra/tile-repair/stop
                      확장은 요청이 도착한 순간부터(Forge 큐 대기 포함) 멈춘다. 아직 가는 중이라 멈출 게
                      없다고 답하면(stopped=false) run 이 끝나지 않은 동안 잠깐씩 다시 묻는다.
                      stop 은 요청을 가리지 않고 그 순간의 route 요청을 전부 멈추므로, 다음 run 은 이전 run 의
                      취소가 stop 을 다 보낸 뒤에 Forge 로 보낸다(그 run 이 답한 뒤엔 더 묻지 않는다).
시그널(ui/vue_bridge.py): ``tileRepairResult`` — JSON {action, requestId, ok, ...}
  options → {options} · run → {path, width, height, seed, info, model} | {canceled} | {error}
  cancel → {stopped} | {error}  (stopped=false 여도 run 의 답은 온다 — 취소했으면 canceled)

로컬 앱 전용이다(웹 모드 거부 — 원본을 호스트 경로로 읽고 결과를 호스트 출력 폴더에 쓴다).
Forge(WebUI) 백엔드만 쓴다. 한 번에 하나만 돈다. 네트워크·파일 읽기는 워커 스레드에서 하고
GUI 스레드는 기다리지 않는다.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import datetime

from core.forge_tile_repair_client import ForgeTileRepairClient
from core.local_image_io import export_exclusive_png
from core.tile_repair_request import build_route_body

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
_INIT_LOCK = threading.Lock()
OUTPUT_SUBDIR = "tile_repair"
CANCELED_MESSAGE = "Tile & Repair 를 취소했습니다."
# 취소 직후엔 run 요청이 아직 Forge 에 닿기 전(본문 직렬화·연결)일 수 있다 — 그러면 확장은 멈출 게 없다고
# 답한다. run 이 아직 도는 동안 이 간격으로 몇 번 더 묻는다(요청이 닿으면 큐 대기 중이어도 멈춘다).
STOP_RETRIES = 6
STOP_RETRY_DELAY = 0.5


def export_tile_repair_png(data: bytes, output_root) -> str:
    # datetime/secrets 는 부를 때 읽는다 — 테스트가 이 모듈의 것을 바꾼다.
    return export_exclusive_png(
        data, output_root, subdir=OUTPUT_SUBDIR, prefix="tile_repair",
        now=lambda: datetime.now(), token=lambda: secrets.token_hex(4),
        outside_message="Tile & Repair 결과 폴더가 앱 출력 폴더 밖을 가리킵니다.",
        exhausted_message="새 결과 파일 이름을 만들지 못했습니다. 다시 실행하세요.",
    )


def _user_message(exc: Exception) -> str:
    from core.error_handler import sanitize_for_ui

    return sanitize_for_ui(str(exc) or type(exc).__name__, 600)


class TileRepairActionsMixin:
    def _handle_tile_repair_action(self, action, payload):
        if action in ("tile_repair_options", "tile_repair_run", "tile_repair_cancel"):
            self._tile_repair_dispatch(action, payload)
            return True
        return False

    # ── 상태 ──
    def _tile_repair_ensure(self):
        with _INIT_LOCK:
            if not hasattr(self, "_tile_repair_lock"):
                self._tile_repair_lock = threading.Lock()
                self._tile_repair_job = None
                # 아직 stop 을 보낼 수 있는 취소 워커의 job(워커마다 하나). 같은 락을 쓰는 조건 변수로
                # run 이 끝남(슬롯 비움)과 취소 워커가 끝남을 알린다.
                self._tile_repair_stopping = []
                self._tile_repair_changed = threading.Condition(self._tile_repair_lock)

    def _tile_repair_emit(self, event: dict) -> None:
        if getattr(self, "_tile_repair_closed", False):
            return
        signal = getattr(getattr(self, "vue_bridge", None), "tileRepairResult", None)
        if signal is None:
            return
        try:
            signal.emit(json.dumps(event, ensure_ascii=False))
        except RuntimeError:
            pass  # 종료 중 QObject 가 이미 사라졌다

    def _tile_repair_client(self):
        """지금 백엔드의 라우트 클라이언트. 쓸 수 없으면 ValueError(사용자 문구)."""
        from backends import BackendType, get_backend, get_backend_type

        if get_backend_type() != BackendType.WEBUI:
            raise ValueError("Tile & Repair 는 Forge(WebUI) 백엔드에서만 쓸 수 있습니다.")
        capabilities = getattr(self, "sam_extra_capabilities", None)
        if capabilities is not None and not capabilities.may_use("tile_repair_route"):
            raise ValueError("연결된 Forge 의 sam-extra 에 Tile & Repair 라우트가 없습니다 — 확장을 업데이트하세요.")
        factory = getattr(self, "_tile_repair_client_factory", None) or ForgeTileRepairClient
        return factory(getattr(get_backend(), "api_url", ""))

    def _tile_repair_output_root(self) -> str:
        import config

        return str(getattr(config, "OUTPUT_DIR", "") or "")

    def _tile_repair_start(self, work, name: str) -> None:
        threading.Thread(target=work, daemon=True, name=name).start()

    def _shutdown_tile_repair(self):
        """종료 — 결과를 더 보내지 않는다. 도는 요청은 Forge 쪽에서 끝까지 간다(취소는 사용자 몫)."""
        self._tile_repair_closed = True

    # ── 액션 ──
    def _tile_repair_dispatch(self, action, payload):
        if getattr(self, "_tile_repair_closed", False):
            return
        request = dict(payload) if isinstance(payload, dict) else {}
        request_id = request.get("requestId", "")
        event = {"action": action, "requestId": request_id if isinstance(request_id, str) else "", "ok": False}
        try:
            if getattr(self, "web_mode", False):
                raise ValueError("Tile & Repair 는 로컬 앱에서만 사용할 수 있습니다.")
            if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
                raise ValueError("작업 식별자가 올바르지 않습니다. 패널을 다시 열어 주세요.")
            self._tile_repair_ensure()
            if action == "tile_repair_cancel":
                self._tile_repair_cancel(event, request_id)
                return
            client = self._tile_repair_client()
            if action == "tile_repair_options":
                self._tile_repair_start(lambda: self._tile_repair_options_work(event, client),
                                        "tile-repair-options")
                return
            with self._tile_repair_lock:
                running = self._tile_repair_job
                if running is not None:
                    if running["requestId"] == request_id:
                        return  # 같은 요청의 재전송 — 두 번째 실행이 아니다
                    raise ValueError("Tile & Repair 작업 하나가 실행 중입니다. 끝난 뒤 다시 시도하세요.")
                job = {"requestId": request_id, "cancel": threading.Event(), "client": client}
                self._tile_repair_job = job
            try:
                self._tile_repair_start(lambda: self._tile_repair_run_work(event, job, request),
                                        "tile-repair-run")
            except Exception:
                with self._tile_repair_lock:
                    if self._tile_repair_job is job:
                        self._tile_repair_job = None
                raise
        except Exception as exc:
            self._tile_repair_emit({**event, "error": _user_message(exc)})

    def _tile_repair_cancel(self, event: dict, request_id: str) -> None:
        with self._tile_repair_lock:
            job = self._tile_repair_job
            if job is None or job["requestId"] != request_id:
                job = None
            else:
                job["cancel"].set()
                # 워커가 시작되기 전에 올린다 — 이 job 이 도는 동안이라 다음 run 은 아직 없다.
                self._tile_repair_stopping.append(job)
        if job is None:
            self._tile_repair_emit({**event, "ok": True, "stopped": False})
            return

        def work():
            try:
                stopped = False
                for attempt in range(1 + STOP_RETRIES):
                    with self._tile_repair_changed:
                        if attempt:     # 다시 묻기 전에 기다리되, run 이 답하면 곧바로 깬다
                            self._tile_repair_changed.wait_for(lambda: self._tile_repair_job is not job,
                                                               STOP_RETRY_DELAY)
                        if self._tile_repair_job is not job:
                            break   # run 이 이미 답했다(취소했으니 결과는 버린다) — 더 보내면 다음 run 을 멈춘다
                    stopped = job["client"].stop()
                    if stopped:
                        break
                self._tile_repair_emit({**event, "ok": True, "stopped": stopped})
            except Exception as exc:
                self._tile_repair_emit({**event, "error": _user_message(exc)})
            finally:
                self._tile_repair_stop_finished(job)

        try:
            self._tile_repair_start(work, "tile-repair-stop")
        except Exception:
            self._tile_repair_stop_finished(job)
            raise

    def _tile_repair_stop_finished(self, job: dict) -> None:
        with self._tile_repair_changed:
            self._tile_repair_stopping.remove(job)
            self._tile_repair_changed.notify_all()

    def _tile_repair_wait_for_earlier_stops(self, job: dict) -> None:
        """이전 run 의 취소 워커가 끝날 때까지 기다린다 — 그 stop 이 이 run 을 멈추지 않게.

        워커는 그 run 이 답하면 곧 끝난다(가는 중인 stop 하나만 기다린다). 이 run 자신의 취소는 기다리지 않는다.
        """
        with self._tile_repair_changed:
            self._tile_repair_changed.wait_for(
                lambda: all(other is job for other in self._tile_repair_stopping))

    def _tile_repair_options_work(self, event: dict, client) -> None:
        try:
            options = client.options()
            self._tile_repair_emit({**event, "ok": True, "options": options})
        except Exception as exc:
            self._tile_repair_emit({**event, "error": _user_message(exc)})

    def _tile_repair_run_work(self, event: dict, job: dict, request: dict) -> None:
        final = {**event, "error": CANCELED_MESSAGE, "canceled": True}
        try:
            body = build_route_body(request)          # 파일 읽기·검사도 워커에서
            self._tile_repair_wait_for_earlier_stops(job)
            if not job["cancel"].is_set():
                result = job["client"].run(body)
                if not result.interrupted and not job["cancel"].is_set():
                    path = export_tile_repair_png(result.png, self._tile_repair_output_root())
                    final = {**event, "ok": True, "path": path, "width": result.width,
                             "height": result.height, "seed": result.seed, "info": result.info,
                             "model": result.model}
        except Exception as exc:
            final = {**event, "error": _user_message(exc)}
        finally:
            # 결과를 보내기 전에 풀어 둔다 — 결과를 받은 화면이 곧바로 다시 실행할 수 있게.
            with self._tile_repair_changed:
                if self._tile_repair_job is job:
                    self._tile_repair_job = None
                self._tile_repair_changed.notify_all()     # 이 run 을 멈추려던 취소 워커를 깨운다
        self._tile_repair_emit(final)
