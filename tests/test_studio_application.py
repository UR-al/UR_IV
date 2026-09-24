from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from core.studio_application import (
    CallContext,
    StudioApplication,
    StudioApplicationError,
)


NATIVE = CallContext("desktop", "qwebchannel", frozenset({"native"}))
WEB = CallContext("browser", "websocket", frozenset())


def request(request_id: str, operation: str, values=None):
    return {
        "version": 1,
        "requestId": request_id,
        "operation": operation,
        "input": {} if values is None else values,
    }


class FakeRuntimeError(RuntimeError):
    def as_dict(self):
        return {
            "code": "OPERATION_BUSY",
            "message": "another operation is running",
            "retryable": True,
            "details": {
                "stage": "start",
                "apiUrl": "http://127.0.0.1:7860",
                "endpoint": "http://127.0.0.1:7860/private",
                "outputPath": "C:/private/runtime/error.json",
                "safeLabel": "runtime error",
            },
        }


class FakeGenerationError(RuntimeError):
    def as_dict(self):
        return {
            "code": "GENERATION_FAILED",
            "message": "generation failed",
            "retryable": False,
            "details": {
                "url": "http://127.0.0.1:17860/private",
                "clientSecret": "private-secret",
                "session": "private-session",
                "safeLabel": "generation error",
            },
        }


class FakeRuntimeManager:
    def __init__(self):
        self.calls = []
        self.fail = False

    def snapshot(self):
        return {
            "ok": True,
            "activeEngine": "forge_neo",
            "primaryModelEngine": "comfy",
            "runtimeRoot": "C:/private/managed",
            "engines": {
                "forge_neo": {
                    "engine": "forge_neo",
                    "name": "Forge Neo",
                    "running": False,
                    "installRoot": "C:/private/forge",
                    "existingRoot": "C:/private/forge",
                    "extensionDir": "C:/private/forge/extensions",
                    "apiUrl": "http://127.0.0.1:7860",
                    "pythonPath": "C:/private/forge/venv/python.exe",
                    "logPath": "C:/private/forge/run.log",
                    "modelPaths": {
                        "loras": ["C:/private/forge/models/Lora"],
                        "checkpoints": ["C:/private/forge/models/Stable-diffusion"],
                    },
                },
                "comfy": {
                    "engine": "comfy",
                    "name": "ComfyUI",
                    "running": False,
                    "installRoot": "C:/private/comfy",
                },
            },
        }

    def execute(self, engine, action, payload, on_progress=None):
        self.calls.append((engine, action, dict(payload)))
        if on_progress:
            on_progress({
                "stage": "prepare",
                "percent": 25,
                "apiUrl": "http://127.0.0.1:7860",
                "nested": {
                    "endpoint": "http://127.0.0.1:7860/private",
                    "runtimePath": "C:/private/runtime",
                    "safeLabel": "runtime update",
                },
            })
        if self.fail:
            raise FakeRuntimeError("busy")
        return {
            "ok": True,
            "engine": engine,
            "action": action,
            "message": "done",
            "activate": action in {"start", "use"},
            "apiUrl": "http://127.0.0.1:7860",
            "pid": 4321,
            "details": {
                "endpoint": "http://127.0.0.1:7860",
                "outputPath": "C:/private/runtime/result.json",
                "host": "127.0.0.1",
                "port": 7860,
                "launchArgs": ["--listen", "127.0.0.1"],
                "process": {"pid": 4321},
                "credentials": {"cookie": "private-session"},
                "safeLabel": "runtime result",
            },
            "snapshot": self.snapshot(),
        }


class FakeGenerationApiManager:
    def __init__(self):
        self.snapshot_args = []
        self.calls = []
        self.fail = False

    def snapshot(self, include_secret=False):
        self.snapshot_args.append(bool(include_secret))
        return {
            "running": False,
            "listenUrl": "http://127.0.0.1:17860",
            "config": {
                "token": "top-secret" if include_secret else "",
                "defaultTarget": "active",
                "targets": [{
                    "id": "remote-comfy",
                    "name": "Remote Comfy",
                    "engine": "comfyui",
                    "url": "http://192.0.2.10:8188",
                    "workflowPath": "C:/private/t2i.json",
                    "img2imgWorkflowPath": "C:/private/i2i.json",
                }],
            },
        }

    def execute(self, action, payload):
        self.calls.append((action, dict(payload)))
        if self.fail:
            raise FakeGenerationError("failed at C:/private/config.json")
        return {
            "ok": True,
            "action": action,
            "endpoint": "http://127.0.0.1:17860",
            "artifact": {
                "url": "http://127.0.0.1:17860/private/result.png",
                "outputPath": "C:/private/generation/result.png",
                "refreshToken": "private-refresh-token",
                "clientSecret": "private-client-secret",
                "apiKey": "private-api-key",
                "sessionId": "private-session",
                "safeLabel": "generation result",
            },
            "state": self.snapshot(include_secret=True),
        }


class BlockingGenerationApiManager(FakeGenerationApiManager):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, action, payload):
        self.calls.append((action, dict(payload)))
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError("test did not release generation operation")
        return {
            "ok": True,
            "action": action,
            "state": self.snapshot(include_secret=True),
        }


class FakeAppUpdateManager:
    def __init__(self):
        self.calls = []
        self.fail = False

    def snapshot(self):
        return {
            "ok": True,
            "repository": "UR-al/UR_IV",
            "currentVersion": "2.6.0",
            "currentDisplay": "v2.6.0 + 41개 변경 (abcdef123)",
            "currentRevision": "abcdef123",
            "developmentBuild": True,
            "mode": "git",
            "branch": "main",
            "pendingVersion": "2.7.0",
            "latestVersion": "2.7.0",
            "tagName": "v2.7.0",
            "releaseName": "v2.7.0",
            "releaseUrl": "https://github.com/UR-al/UR_IV/releases/tag/v2.7.0",
            "notes": "changes",
            "updateAvailable": True,
            "notificationAvailable": True,
            "canInstall": True,
        }

    def execute(self, action, payload, on_progress=None):
        self.calls.append((action, dict(payload)))
        if on_progress:
            on_progress({"stage": "checking", "message": "checking"})
        if self.fail:
            raise RuntimeError("update failed")
        return {
            "ok": True,
            "action": action,
            "message": "done",
            "restartRequired": action == "install",
            "helperPid": 98765,
            "targetVersion": "2.7.0",
        }


class FakeHost:
    def __init__(self):
        self.picks = []
        self.refresh_count = 0
        self.runtime_events = []
        self.restart_events = []
        self.next_path = "C:/selected"

    def pick_directory(self, kind, selector, current):
        self.picks.append((kind, selector, current))
        return self.next_path

    def refresh_model_widgets(self):
        self.refresh_count += 1

    def handle_runtime_event(self, payload):
        self.runtime_events.append(payload)

    def request_app_restart(self, payload):
        self.restart_events.append(payload)


class StudioApplicationContractTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntimeManager()
        self.generation = FakeGenerationApiManager()
        self.updates = FakeAppUpdateManager()
        self.host = FakeHost()
        self.app = StudioApplication(
            host=self.host,
            runtime_manager=self.runtime,
            generation_api_manager=self.generation,
            app_update_manager=self.updates,
        )

    def test_describe_advertises_one_version_and_native_availability(self):
        native = self.app.describe(NATIVE)
        web = self.app.describe(WEB)
        other = StudioApplication(
            host=self.host,
            runtime_manager=self.runtime,
            generation_api_manager=self.generation,
        ).describe(WEB)

        self.assertEqual(native["version"], 1)
        self.assertRegex(native["eventEpoch"], r"^[0-9a-f]{32}$")
        self.assertEqual(native["eventEpoch"], web["eventEpoch"])
        self.assertNotEqual(native["eventEpoch"], other["eventEpoch"])
        self.assertTrue(native["nativeOperations"])
        self.assertFalse(web["nativeOperations"])
        native_ops = {item["name"]: item for item in native["operations"]}
        web_ops = {item["name"]: item for item in web["operations"]}
        self.assertTrue(native_ops["runtime.execute"]["available"])
        self.assertFalse(web_ops["runtime.execute"]["available"])
        self.assertTrue(web_ops["runtime.snapshot"]["available"])
        self.assertTrue(web_ops["app_update.snapshot"]["available"])
        self.assertFalse(web_ops["app_update.execute"]["available"])

    def test_invoke_requires_the_exact_envelope(self):
        missing = self.app.invoke(WEB, {"version": 1})
        self.assertEqual(missing["status"], "error")
        self.assertEqual(missing["error"]["code"], "INVALID_ARGUMENT")
        self.assertIn("missing", missing["error"]["details"])

        unknown = request("bad-extra", "runtime.snapshot")
        unknown["extra"] = True
        reply = self.app.invoke(WEB, unknown)
        self.assertEqual(reply["status"], "error")
        self.assertEqual(reply["error"]["details"]["unknown"], ["extra"])

        wrong_version = request("bad-version", "runtime.snapshot")
        wrong_version["version"] = 2
        reply = self.app.invoke(WEB, wrong_version)
        self.assertEqual(reply["error"]["code"], "UNSUPPORTED_VERSION")

        wrong_input = request("bad-input", "runtime.snapshot")
        wrong_input["input"] = []
        reply = self.app.invoke(WEB, wrong_input)
        self.assertEqual(reply["error"]["code"], "INVALID_ARGUMENT")

    def test_web_is_read_only_and_snapshots_are_redacted(self):
        runtime = self.app.invoke(WEB, request("runtime", "runtime.snapshot"))
        self.assertEqual(runtime["status"], "ok")
        self.assertFalse(runtime["data"]["nativeOperations"])
        self.assertEqual(runtime["data"]["activeEngine"], "forge")
        self.assertEqual(runtime["data"]["primaryModelEngine"], "comfyui")
        self.assertNotIn("runtimeRoot", runtime["data"])
        forge = runtime["data"]["engines"]["forge"]
        for key in (
            "installRoot", "existingRoot", "extensionDir", "apiUrl",
            "pythonPath", "logPath", "modelPaths",
        ):
            self.assertNotIn(key, forge)
        self.assertEqual(forge["modelPathCounts"]["loras"], 1)

        generation = self.app.invoke(
            WEB, request("generation", "generation_api.snapshot")
        )
        config = generation["data"]["config"]
        self.assertNotIn("token", config)
        target = config["targets"][0]
        self.assertNotIn("url", target)
        self.assertNotIn("workflowPath", target)
        self.assertTrue(target["urlConfigured"])
        self.assertTrue(target["workflowConfigured"])

        blocked = self.app.invoke(
            WEB,
            request("blocked", "runtime.execute", {
                "engine": "forge",
                "action": "start",
                "payload": {},
            }),
        )
        self.assertEqual(blocked["status"], "error")
        self.assertEqual(blocked["error"]["code"], "FORBIDDEN")
        self.assertEqual(self.runtime.calls, [])

        update = self.app.invoke(WEB, request("update", "app_update.snapshot"))
        self.assertEqual(update["status"], "ok")
        self.assertFalse(update["data"]["nativeOperations"])
        self.assertFalse(update["data"]["canInstall"])
        self.assertNotIn("branch", update["data"])
        self.assertNotIn("currentRevision", update["data"])
        self.assertNotIn("mode", update["data"])
        self.assertNotIn("pendingVersion", update["data"])
        self.assertNotIn("abcdef123", update["data"]["currentDisplay"])

        projected = self.app._public_event({
            "topic": "app_update.operation",
            "data": {
                "action": "install",
                "result": {
                    "ok": True,
                    "message": "done",
                    "helperPid": 98765,
                    "targetVersion": "2.7.0",
                    "snapshot": self.updates.snapshot(),
                },
                "error": {
                    "code": "UPDATE_LOCAL_CHANGES",
                    "message": "conflict",
                    "details": {"paths": ["config/private.json"]},
                },
            },
        }, WEB)
        projected_text = repr(projected)
        self.assertNotIn("98765", projected_text)
        self.assertNotIn("abcdef123", projected_text)
        self.assertNotIn("config/private.json", projected_text)

    def test_bootstrap_returns_coherent_read_models(self):
        state = {
            "paths": {"lora_dir": "C:/private/Lora"},
            "defaults": {"lora_dir": "C:/default/Lora"},
            "entries": {"lora_dir": {"exists": True, "count": 3}},
            "environmentLocked": {"lora_dir": False},
        }
        with patch("core.forge_modules.get_forge_path_state", return_value=state):
            reply = self.app.invoke(WEB, request("boot", "sync.bootstrap"))

        self.assertEqual(reply["status"], "ok")
        self.assertIn("description", reply["data"])
        self.assertEqual(reply["data"]["eventEpoch"], self.app.event_epoch)
        self.assertEqual(
            reply["data"]["description"]["eventEpoch"],
            reply["data"]["eventEpoch"],
        )
        self.assertIn("runtime", reply["data"])
        self.assertIn("generationApi", reply["data"])
        self.assertIn("appUpdate", reply["data"])
        self.assertNotIn("paths", reply["data"]["modelPaths"])
        self.assertTrue(reply["data"]["modelPaths"]["configured"]["lora_dir"])

    def _run_runtime_job(self, payload, updates):
        snapshots = []
        original_snapshot = self.runtime.snapshot

        def counting_snapshot():
            snapshots.append(1)
            return original_snapshot()

        def execute(engine, action, job_payload, on_progress=None):
            self.runtime.calls.append((engine, action, dict(job_payload)))
            for update in updates:
                on_progress(update)
            return {"ok": True, "engine": engine, "action": action, "message": "done"}

        self.runtime.snapshot = counting_snapshot
        self.runtime.execute = execute
        events = []
        terminal = threading.Event()
        host_before = len(self.host.runtime_events)
        stop = self.app.subscribe(
            NATIVE,
            lambda event: (
                events.append(event),
                terminal.set() if event["type"] in {"completed", "error"} else None,
            ),
            after_seq=self.app.describe(NATIVE)["eventCursor"],
        )
        reply = self.app.invoke(NATIVE, request("rt-gate", "runtime.execute", {
            "engine": "forge", "action": "start", "payload": payload,
        }))
        self.assertTrue(terminal.wait(2))
        stop()
        # host 통보는 completed 발행 직후에 온다 — 도착까지 잠깐 기다린다.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not any(
            event.get("type") in {"completed", "error"}
            for event in self.host.runtime_events[host_before:]
        ):
            time.sleep(0.01)
        job = [event for event in events if event.get("jobId") == reply["data"]["jobId"]]
        return job, snapshots

    def test_runtime_progress_carries_a_snapshot_only_when_the_phase_changes(self):
        updates = [{"phase": "download", "message": f"line {i}"} for i in range(6)]
        updates += [{"phase": "health", "message": "waiting"}] * 3
        job, snapshots = self._run_runtime_job({}, updates)
        progress = [event for event in job if event["type"] == "progress"]
        self.assertEqual(9, len(progress))
        with_snapshot = [event for event in progress if "snapshot" in event["data"]]
        self.assertEqual(
            ["download", "health"],
            [event["data"]["update"]["phase"] for event in with_snapshot],
        )
        # started 1 + 단계 2 + completed 1 — 예전엔 progress 마다 1번씩 더 계산했다(=11).
        self.assertEqual(4, len(snapshots))
        self.assertIn("snapshot", job[-1]["data"])

    def test_host_receives_started_and_progress_only_for_startup_jobs(self):
        self._run_runtime_job({}, [{"phase": "health", "message": "waiting"}])
        self.assertEqual(["completed"], [event["type"] for event in self.host.runtime_events])
        self.host.runtime_events.clear()
        self._run_runtime_job({"startup": True}, [{"phase": "health", "message": "waiting"}])
        self.assertEqual(
            ["started", "progress", "completed"],
            [event["type"] for event in self.host.runtime_events],
        )
        progress = self.host.runtime_events[1]
        self.assertNotIn("snapshot", progress)
        self.assertEqual("waiting", progress["message"])

    def test_job_reporting_failure_still_publishes_a_minimal_error(self):
        original_publish = self.app._publish
        failed = {"started": False}

        def flaky_publish(**kwargs):
            if kwargs.get("event_type") == "started" and not failed["started"]:
                failed["started"] = True
                raise RuntimeError("journal unavailable")
            if kwargs.get("event_type") == "error" and "snapshot" in (kwargs.get("data") or {}):
                raise RuntimeError("snapshot publish broken")
            return original_publish(**kwargs)

        self.app._publish = flaky_publish
        events = []
        terminal = threading.Event()
        stop = self.app.subscribe(NATIVE, lambda event: (
            events.append(event),
            terminal.set() if event["type"] == "error" else None,
        ))
        reply = self.app.invoke(NATIVE, request("gen-flaky", "generation_api.execute", {
            "action": "start", "payload": {},
        }))
        self.assertTrue(terminal.wait(2))
        stop()
        error = [event for event in events if event["type"] == "error"][-1]
        self.assertEqual(reply["data"]["jobId"], error["jobId"])
        self.assertNotIn("snapshot", error["data"])
        # 작업 락도 풀려 다음 요청을 받을 수 있다.
        self.app._publish = original_publish
        again = self.app.invoke(NATIVE, request("gen-again", "generation_api.execute", {
            "action": "start", "payload": {},
        }))
        self.assertEqual("accepted", again["status"])

    def test_bootstrap_can_skip_the_git_backed_app_update_snapshot(self):
        state = {"paths": {}, "defaults": {}, "entries": {}, "environmentLocked": {}}
        calls = []
        original = self.updates.snapshot
        self.updates.snapshot = lambda: calls.append("snapshot") or original()
        with patch("core.forge_modules.get_forge_path_state", return_value=state):
            reply = self.app.invoke(
                WEB, request("boot-lite", "sync.bootstrap", {"includeAppUpdate": False})
            )
            self.assertEqual(reply["status"], "ok")
            self.assertNotIn("appUpdate", reply["data"])
            self.assertIn("runtime", reply["data"])
            self.assertEqual(calls, [])
            invalid = self.app.invoke(
                WEB, request("boot-bad", "sync.bootstrap", {"includeAppUpdate": "no"})
            )
        self.assertEqual(invalid["status"], "error")
        self.assertEqual(invalid["error"]["code"], "INVALID_ARGUMENT")

    def test_app_update_job_publishes_progress_and_requests_restart_after_completion(self):
        events = []
        terminal = threading.Event()
        stop = self.app.subscribe(
            NATIVE,
            lambda event: (
                events.append(event),
                terminal.set()
                if event["topic"] == "app_update.operation"
                and event["type"] in {"completed", "error"}
                else None,
            ),
        )

        reply = self.app.invoke(
            NATIVE,
            request("app-update", "app_update.execute", {
                "action": "install",
                "payload": {},
            }),
        )

        self.assertEqual(reply["status"], "accepted")
        self.assertEqual(reply["job"]["operation"], "app_update.execute")
        self.assertTrue(terminal.wait(2))
        stop()
        job_events = [event for event in events if event.get("jobId") == reply["job"]["id"]]
        self.assertEqual(job_events[0]["type"], "accepted")
        self.assertEqual(job_events[1]["type"], "started")
        self.assertIn("progress", [event["type"] for event in job_events])
        self.assertEqual(job_events[-1]["type"], "completed")
        self.assertEqual(self.updates.calls, [("install", {})])
        self.assertEqual(len(self.host.restart_events), 1)
        self.assertTrue(self.host.restart_events[0]["restartRequired"])

    def test_runtime_job_has_one_id_and_monotonic_generic_events(self):
        events = []
        terminal = threading.Event()

        def sink(event):
            events.append(event)
            if event["topic"] == "runtime.operation" and event["type"] in {"completed", "error"}:
                terminal.set()

        unsubscribe = self.app.subscribe(NATIVE, sink)
        reply = self.app.invoke(
            NATIVE,
            request("runtime-job", "runtime.execute", {
                "engine": "forge_neo",
                "action": "start",
                "payload": {"startup": True},
            }),
        )
        self.assertEqual(reply["status"], "accepted")
        self.assertEqual(reply["job"]["id"], reply["data"]["jobId"])
        self.assertEqual(reply["job"]["operation"], "runtime.execute")
        self.assertEqual(reply["job"]["state"], "queued")
        self.assertTrue(terminal.wait(2))
        unsubscribe()
        unsubscribe()

        job_id = reply["data"]["jobId"]
        job_events = [event for event in events if event.get("jobId") == job_id]
        self.assertEqual(job_events[0]["type"], "accepted")
        self.assertEqual(job_events[1]["type"], "started")
        self.assertIn("progress", [event["type"] for event in job_events])
        self.assertEqual(job_events[-1]["type"], "completed")
        self.assertTrue(all(event["topic"] == "runtime.operation" for event in job_events))
        self.assertTrue(all(event["eventEpoch"] == self.app.event_epoch for event in job_events))
        self.assertEqual(
            [event["seq"] for event in job_events],
            sorted(event["seq"] for event in job_events),
        )
        self.assertEqual(self.runtime.calls, [("forge", "start", {"startup": True})])
        self.assertEqual(job_events[-1]["data"]["snapshot"]["activeEngine"], "forge")

        host_events = [
            event for event in self.host.runtime_events
            if event.get("operationId") == job_id
        ]
        self.assertEqual(
            [event["type"] for event in host_events],
            ["started", "progress", "completed"],
        )
        completed = host_events[-1]
        self.assertTrue(completed["ok"])
        self.assertTrue(completed["activate"])
        self.assertTrue(completed["startup"])
        self.assertEqual(completed["result"]["apiUrl"], "http://127.0.0.1:7860")
        self.assertEqual(completed["state"]["engine"], "forge")

    def test_runtime_job_normalises_structured_manager_error(self):
        self.runtime.fail = True
        events = []
        terminal = threading.Event()
        stop = self.app.subscribe(
            NATIVE,
            lambda event: (
                events.append(event),
                terminal.set() if event["type"] == "error" else None,
            ),
        )
        reply = self.app.invoke(
            NATIVE,
            request("runtime-fail", "runtime.execute", {
                "engine": "comfy",
                "action": "update",
                "payload": {},
            }),
        )
        self.assertEqual(reply["status"], "accepted")
        self.assertEqual(reply["job"]["operation"], "runtime.execute")
        self.assertTrue(terminal.wait(2))
        stop()

        error_event = [event for event in events if event["type"] == "error"][-1]
        self.assertEqual(error_event["data"]["error"]["code"], "OPERATION_BUSY")
        self.assertTrue(error_event["data"]["error"]["retryable"])
        self.assertIn("snapshot", error_event["data"])
        host_error = self.host.runtime_events[-1]
        self.assertEqual(host_error["type"], "error")
        self.assertFalse(host_error["ok"])
        self.assertEqual(host_error["error"]["code"], "OPERATION_BUSY")

    def test_runtime_job_redacts_web_result_extensions_recursively(self):
        web_events = []
        terminal = threading.Event()

        def sink(event):
            web_events.append(event)
            if event["topic"] == "runtime.operation" and event["type"] == "completed":
                terminal.set()

        stop = self.app.subscribe(WEB, sink)
        reply = self.app.invoke(
            NATIVE,
            request("runtime-web-result", "runtime.execute", {
                "engine": "forge",
                "action": "start",
                "payload": {},
            }),
        )
        self.assertEqual(reply["status"], "accepted")
        self.assertTrue(terminal.wait(2))
        stop()

        completed = [event for event in web_events if event["type"] == "completed"][-1]
        result = completed["data"]["result"]
        self.assertNotIn("apiUrl", result)
        self.assertNotIn("pid", result)
        self.assertNotIn("endpoint", result["details"])
        self.assertNotIn("outputPath", result["details"])
        self.assertNotIn("host", result["details"])
        self.assertNotIn("port", result["details"])
        self.assertNotIn("launchArgs", result["details"])
        self.assertNotIn("process", result["details"])
        self.assertNotIn("credentials", result["details"])
        self.assertEqual(result["details"]["safeLabel"], "runtime result")
        self.assertNotIn("runtimeRoot", result["snapshot"])
        self.assertNotIn("installRoot", result["snapshot"]["engines"]["forge"])

        progress = [event for event in web_events if event["type"] == "progress"][-1]
        update = progress["data"]["update"]
        self.assertNotIn("apiUrl", update)
        self.assertNotIn("endpoint", update["nested"])
        self.assertNotIn("runtimePath", update["nested"])
        self.assertEqual(update["nested"]["safeLabel"], "runtime update")

    def test_web_error_details_are_fail_closed_for_runtime_and_generation(self):
        for manager, operation, values, expected_label in (
            (
                self.runtime,
                "runtime.execute",
                {"engine": "forge", "action": "start", "payload": {}},
                "runtime error",
            ),
            (
                self.generation,
                "generation_api.execute",
                {"action": "start", "payload": {}},
                "generation error",
            ),
        ):
            manager.fail = True
            events = []
            terminal = threading.Event()
            stop = self.app.subscribe(
                WEB,
                lambda event: (
                    events.append(event),
                    terminal.set() if event["type"] == "error" else None,
                ),
                after_seq=self.app.describe(WEB)["eventCursor"],
            )
            reply = self.app.invoke(
                NATIVE,
                request(f"web-{operation}", operation, values),
            )
            self.assertEqual(reply["status"], "accepted")
            self.assertTrue(terminal.wait(2))
            stop()
            manager.fail = False

            details = [event for event in events if event["type"] == "error"][-1][
                "data"
            ]["error"]["details"]
            for secret_key in (
                "apiUrl", "endpoint", "outputPath", "url", "clientSecret", "session",
            ):
                self.assertNotIn(secret_key, details)
            self.assertEqual(details["safeLabel"], expected_label)

    def test_runtime_manager_initialisation_failure_finishes_the_accepted_job(self):
        app = StudioApplication(
            host=self.host,
            generation_api_manager=self.generation,
        )
        events = []
        terminal = threading.Event()
        stop = app.subscribe(
            NATIVE,
            lambda event: (
                events.append(event),
                terminal.set() if event["type"] == "error" else None,
            ),
        )

        with patch(
            "core.backend_runtime.get_backend_runtime_manager",
            side_effect=RuntimeError("runtime manager init failed"),
        ):
            reply = app.invoke(
                NATIVE,
                request("runtime-init-fail", "runtime.execute", {
                    "engine": "forge",
                    "action": "start",
                    "payload": {},
                }),
            )
            self.assertEqual(reply["status"], "accepted")
            self.assertTrue(terminal.wait(2))
        stop()

        job_id = reply["job"]["id"]
        job_events = [event for event in events if event.get("jobId") == job_id]
        self.assertEqual([event["type"] for event in job_events], ["accepted", "error"])
        self.assertEqual(job_events[-1]["data"]["error"]["code"], "INTERNAL")
        self.assertIn("runtime manager init failed", job_events[-1]["data"]["error"]["message"])

    def test_worker_start_failure_does_not_publish_an_orphaned_job(self):
        events = []
        stop = self.app.subscribe(NATIVE, events.append)

        with patch.object(
            threading.Thread,
            "start",
            side_effect=RuntimeError("worker could not start"),
        ):
            reply = self.app.invoke(
                NATIVE,
                request("worker-start-fail", "runtime.execute", {
                    "engine": "forge",
                    "action": "start",
                    "payload": {},
                }),
            )
        stop()

        self.assertEqual(reply["status"], "error")
        self.assertEqual(reply["error"]["code"], "INTERNAL")
        self.assertTrue(reply["error"]["retryable"])
        self.assertEqual(events, [])

    def test_generation_job_reuses_manager_and_redacts_web_events(self):
        web_events = []
        terminal = threading.Event()

        def sink(event):
            web_events.append(event)
            if event["topic"] == "generation_api.operation" and event["type"] in {"completed", "error"}:
                terminal.set()

        stop = self.app.subscribe(WEB, sink)
        reply = self.app.invoke(
            NATIVE,
            request("generation-job", "generation_api.execute", {
                "action": "start",
                "payload": {"enabled": True},
            }),
        )
        self.assertEqual(reply["status"], "accepted")
        self.assertEqual(reply["job"]["operation"], "generation_api.execute")
        self.assertTrue(terminal.wait(2))
        stop()
        self.assertEqual(self.generation.calls, [("start", {"enabled": True})])

        completed = [event for event in web_events if event["type"] == "completed"][-1]
        snapshot = completed["data"]["snapshot"]
        self.assertFalse(snapshot["nativeOperations"])
        self.assertNotIn("token", snapshot["config"])
        target = snapshot["config"]["targets"][0]
        self.assertNotIn("url", target)
        self.assertNotIn("workflowPath", target)
        state = completed["data"]["result"]["state"]
        self.assertNotIn("token", state["config"])
        self.assertNotIn("listenUrl", state)
        result = completed["data"]["result"]
        self.assertNotIn("endpoint", result)
        self.assertNotIn("url", result["artifact"])
        self.assertNotIn("outputPath", result["artifact"])
        self.assertNotIn("refreshToken", result["artifact"])
        self.assertNotIn("clientSecret", result["artifact"])
        self.assertNotIn("apiKey", result["artifact"])
        self.assertNotIn("sessionId", result["artifact"])
        self.assertEqual(result["artifact"]["safeLabel"], "generation result")

    def test_generation_manager_initialisation_failure_finishes_the_accepted_job(self):
        app = StudioApplication(
            host=self.host,
            runtime_manager=self.runtime,
        )
        events = []
        terminal = threading.Event()
        stop = app.subscribe(
            NATIVE,
            lambda event: (
                events.append(event),
                terminal.set() if event["type"] == "error" else None,
            ),
        )

        with patch(
            "core.generation_api.get_generation_api_manager",
            side_effect=RuntimeError("generation manager init failed"),
        ):
            reply = app.invoke(
                NATIVE,
                request("generation-init-fail", "generation_api.execute", {
                    "action": "start",
                    "payload": {},
                }),
            )
            self.assertEqual(reply["status"], "accepted")
            self.assertTrue(terminal.wait(2))
        stop()

        job_id = reply["job"]["id"]
        job_events = [event for event in events if event.get("jobId") == job_id]
        self.assertEqual([event["type"] for event in job_events], ["accepted", "error"])
        self.assertEqual(job_events[-1]["data"]["error"]["code"], "INTERNAL")
        self.assertIn("generation manager init failed", job_events[-1]["data"]["error"]["message"])

    def test_generation_api_rejects_a_second_operation_while_one_is_running(self):
        generation = BlockingGenerationApiManager()
        app = StudioApplication(
            host=self.host,
            runtime_manager=self.runtime,
            generation_api_manager=generation,
        )
        terminal = threading.Event()
        stop = app.subscribe(
            NATIVE,
            lambda event: terminal.set()
            if event["topic"] == "generation_api.operation"
            and event["type"] in {"completed", "error"}
            else None,
        )

        first = app.invoke(
            NATIVE,
            request("generation-first", "generation_api.execute", {
                "action": "start",
                "payload": {},
            }),
        )
        self.assertEqual(first["status"], "accepted")
        self.assertTrue(generation.started.wait(2))

        second = app.invoke(
            NATIVE,
            request("generation-second", "generation_api.execute", {
                "action": "stop",
                "payload": {},
            }),
        )
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["error"]["code"], "OPERATION_BUSY")
        self.assertTrue(second["error"]["retryable"])
        self.assertEqual(second["error"]["details"]["activeJobId"], first["job"]["id"])
        self.assertEqual(generation.calls, [("start", {})])

        generation.release.set()
        self.assertTrue(terminal.wait(2))
        stop()

    def test_generation_worker_start_failure_releases_the_operation_gate(self):
        events = []
        terminal = threading.Event()
        stop = self.app.subscribe(
            NATIVE,
            lambda event: (
                events.append(event),
                terminal.set()
                if event["topic"] == "generation_api.operation"
                and event["type"] in {"completed", "error"}
                else None,
            ),
        )

        with patch.object(
            threading.Thread,
            "start",
            side_effect=RuntimeError("worker could not start"),
        ):
            failed = self.app.invoke(
                NATIVE,
                request("generation-worker-fail", "generation_api.execute", {
                    "action": "start",
                    "payload": {},
                }),
            )
        self.assertEqual(failed["status"], "error")
        self.assertEqual(events, [])

        retry = self.app.invoke(
            NATIVE,
            request("generation-worker-retry", "generation_api.execute", {
                "action": "start",
                "payload": {},
            }),
        )
        self.assertEqual(retry["status"], "accepted")
        self.assertTrue(terminal.wait(2))
        stop()

    def test_model_path_commands_publish_replayable_event_and_unsubscribe(self):
        state = {
            "paths": {"lora_dir": "C:/selected/Lora"},
            "defaults": {"lora_dir": "C:/default/Lora"},
            "entries": {"lora_dir": {"exists": True, "count": 2}},
            "environmentLocked": {"lora_dir": False},
        }
        with (
            patch("core.forge_modules.save_forge_paths") as save,
            patch("core.forge_modules.get_forge_path_state", return_value=state),
        ):
            reply = self.app.invoke(
                NATIVE,
                request("save-paths", "model_paths.save", {
                    "paths": {"lora_dir": "C:/selected/Lora"},
                }),
            )
        self.assertEqual(reply["status"], "ok")
        save.assert_called_once_with({"lora_dir": "C:/selected/Lora"})
        self.assertEqual(self.host.refresh_count, 1)

        replayed = []
        stop = self.app.subscribe(NATIVE, replayed.append, after_seq=0)
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0]["topic"], "model_paths.changed")
        self.assertEqual(replayed[0]["seq"], reply["seq"])
        stop()
        stop()

        with (
            patch("core.forge_modules.reset_forge_paths"),
            patch("core.forge_modules.get_forge_path_state", return_value=state),
        ):
            self.app.invoke(NATIVE, request("reset-paths", "model_paths.reset"))
        self.assertEqual(len(replayed), 1)

    def test_model_refresh_requires_native_host_method(self):
        app = StudioApplication(
            runtime_manager=self.runtime,
            generation_api_manager=self.generation,
        )
        reply = app.invoke(NATIVE, request("refresh", "model_paths.refresh"))
        self.assertEqual(reply["status"], "error")
        self.assertEqual(reply["error"]["code"], "UNAVAILABLE")

    def test_directory_picker_uses_exact_purpose_selector_and_current_contract(self):
        runtime_reply = self.app.invoke(
            NATIVE,
            request("pick-runtime", "native.pick_directory", {
                "purpose": "runtime_install",
                "engine": "forge_neo",
            }),
        )
        self.assertEqual(runtime_reply["status"], "ok")
        self.assertEqual(runtime_reply["data"]["engine"], "forge")
        self.assertEqual(
            self.host.picks[-1],
            ("runtime_install", "forge", "C:/private/forge"),
        )

        state = {
            "paths": {"lora_dir": "C:/private/Lora"},
            "defaults": {},
            "entries": {},
            "environmentLocked": {},
        }
        with patch("core.forge_modules.get_forge_path_state", return_value=state):
            model_reply = self.app.invoke(
                NATIVE,
                request("pick-model", "native.pick_directory", {
                    "purpose": "model_path",
                    "key": "lora_dir",
                }),
            )
        self.assertEqual(model_reply["data"]["key"], "lora_dir")
        self.assertEqual(
            self.host.picks[-1],
            ("model_path", "lora_dir", "C:/private/Lora"),
        )

        self.host.next_path = None
        cancelled = self.app.invoke(
            NATIVE,
            request("pick-cancel", "native.pick_directory", {
                "purpose": "runtime_extension",
                "engine": "comfyui",
            }),
        )
        self.assertTrue(cancelled["data"]["cancelled"])

    def test_subscription_rejects_bad_cursor_and_isolates_sink_errors(self):
        with self.assertRaises(ValueError):
            self.app.subscribe(NATIVE, lambda _event: None, after_seq=-1)
        with self.assertRaises(TypeError):
            self.app.subscribe(NATIVE, None)  # type: ignore[arg-type]

        delivered = []
        stop_bad = self.app.subscribe(NATIVE, lambda _event: (_ for _ in ()).throw(RuntimeError("boom")))
        stop_good = self.app.subscribe(NATIVE, delivered.append)
        state = {"paths": {}, "defaults": {}, "entries": {}, "environmentLocked": {}}
        with (
            patch("core.forge_modules.save_forge_paths"),
            patch("core.forge_modules.get_forge_path_state", return_value=state),
        ):
            reply = self.app.invoke(
                NATIVE, request("event", "model_paths.save", {"paths": {}})
            )
        stop_bad()
        stop_good()
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(len(delivered), 1)

    def test_subscription_rejects_cursor_older_than_retained_journal(self):
        journal_size = self.app._journal.maxlen
        self.assertIsNotNone(journal_size)
        for index in range(journal_size + 1):
            self.app._publish(
                topic="runtime.operation",
                event_type="progress",
                operation="runtime.execute",
                data={"index": index},
            )

        with self.assertRaises(StudioApplicationError) as raised:
            self.app.subscribe(WEB, lambda _event: None, after_seq=0)

        error = raised.exception
        self.assertEqual(error.code, "CURSOR_EXPIRED")
        self.assertTrue(error.retryable)
        self.assertEqual(error.details, {
            "earliestSeq": 2,
            "currentSeq": journal_size + 1,
        })
        self.assertEqual(self.app._subscribers, {})

        replayed = []
        stop = self.app.subscribe(WEB, replayed.append, after_seq=1)
        self.assertEqual(len(replayed), journal_size)
        self.assertEqual(replayed[0]["seq"], 2)
        stop()

    def test_subscription_rejects_cursor_ahead_after_server_epoch_reset(self):
        with self.assertRaises(StudioApplicationError) as raised:
            self.app.subscribe(WEB, lambda _event: None, after_seq=9)

        error = raised.exception
        self.assertEqual(error.code, "CURSOR_EXPIRED")
        self.assertTrue(error.retryable)
        self.assertEqual(error.details, {
            "earliestSeq": 1,
            "currentSeq": 0,
        })


if __name__ == "__main__":
    unittest.main()
