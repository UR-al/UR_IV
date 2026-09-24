from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backends import BackendType
from ui.generator_main import GeneratorMainUI, _same_api_endpoint


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _Button:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _Harness:
    _on_backend_runtime_event = GeneratorMainUI._on_backend_runtime_event

    def __init__(self):
        self._backend_connected = True
        self._managed_runtime_startup_inflight = False
        self._managed_runtime_startup_apply_done = True
        self.vue_bridge = SimpleNamespace(showNotification=_Signal())
        self.btn_generate = _Button()
        self.info_loads = 0

    def load_webui_info(self):
        self.info_loads += 1


def _completed_event(action: str, *, current_engine: str = 'forge') -> str:
    return json.dumps({
        'engine': current_engine,
        'type': 'completed',
        'action': action,
        'ok': True,
        'activate': action == 'start',
        'result': {
            'apiUrl': 'http://127.0.0.1:7861',
            'replacedEngine': 'comfyui',
            'stopped': action == 'stop',
            'owned': action == 'stop',
        },
        'state': {'apiUrl': 'http://127.0.0.1:7861'},
        'snapshot': {
            'engines': {
                'forge': {'apiUrl': 'http://127.0.0.1:7861'},
                'comfyui': {'apiUrl': 'http://127.0.0.1:8188'},
            },
        },
    })


class ManagedRuntimeUiIdentityTests(unittest.TestCase):
    def test_endpoint_comparison_ignores_only_case_and_trailing_slash(self):
        self.assertTrue(_same_api_endpoint('HTTP://127.0.0.1:7860/', 'http://127.0.0.1:7860'))
        self.assertFalse(_same_api_endpoint('', 'http://127.0.0.1:7860'))
        self.assertFalse(_same_api_endpoint('http://127.0.0.1:7860', 'http://127.0.0.1:7861'))

    def test_start_replacement_does_not_override_an_external_backend(self):
        harness = _Harness()
        with (
            patch('backends.get_backend', return_value=SimpleNamespace(api_url='http://127.0.0.1:9000')),
            patch('backends.set_backend') as set_backend,
        ):
            harness._on_backend_runtime_event(_completed_event('start'))

        set_backend.assert_not_called()
        self.assertTrue(harness._backend_connected)
        self.assertEqual(harness.info_loads, 0)

    def test_start_replacement_follows_the_managed_backend_it_stopped(self):
        harness = _Harness()
        with (
            patch('backends.get_backend', return_value=SimpleNamespace(api_url='http://127.0.0.1:8188/')),
            patch('backends.set_backend') as set_backend,
        ):
            harness._on_backend_runtime_event(_completed_event('start'))

        set_backend.assert_called_once_with(BackendType.WEBUI, 'http://127.0.0.1:7861')
        self.assertEqual(harness.info_loads, 1)

    def test_stop_only_disconnects_the_exact_managed_endpoint(self):
        external = _Harness()
        managed = _Harness()
        stop_event = _completed_event('stop')
        with (
            patch('backends.get_backend_type', return_value=BackendType.WEBUI),
            patch('backends.get_backend', return_value=SimpleNamespace(api_url='http://127.0.0.1:9000')),
        ):
            external._on_backend_runtime_event(stop_event)
        self.assertTrue(external._backend_connected)
        self.assertTrue(external.btn_generate.enabled)

        with (
            patch('backends.get_backend_type', return_value=BackendType.WEBUI),
            patch('backends.get_backend', return_value=SimpleNamespace(api_url='http://127.0.0.1:7861/')),
        ):
            managed._on_backend_runtime_event(stop_event)
        self.assertFalse(managed._backend_connected)
        self.assertFalse(managed.btn_generate.enabled)

    def test_stop_noop_never_disconnects_an_external_backend_on_the_same_port(self):
        harness = _Harness()
        stop_event = json.loads(_completed_event('stop'))
        stop_event['result']['stopped'] = False
        stop_event['result']['owned'] = False
        with (
            patch('backends.get_backend_type', return_value=BackendType.WEBUI),
            patch('backends.get_backend', return_value=SimpleNamespace(api_url='http://127.0.0.1:7861')),
        ):
            harness._on_backend_runtime_event(json.dumps(stop_event))

        self.assertTrue(harness._backend_connected)
        self.assertTrue(harness.btn_generate.enabled)


if __name__ == '__main__':
    unittest.main()
