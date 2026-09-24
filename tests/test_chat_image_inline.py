"""대화 경로 이미지 → base64 는 워커 스레드에서, 최근 턴만 (가짜 HTTP 만, Ollama 없음).

예전 _chat_send 는 GUI 스레드(onAction 동기 슬롯)에서 스레드 **전체**의 경로 이미지(최대 20MB)를
읽어 base64 로 만든 뒤에야 24턴으로 잘랐다.
"""
from __future__ import annotations

import base64
import builtins
import json
import sys
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PyQt6.QtCore import QCoreApplication, QObject, QThread

from core.chat_store import build_ollama_messages, inline_image_paths, looks_like_image_path
# ui.* 는 QtWebEngine 을 끌어오므로 QCoreApplication 보다 먼저 import 해야 한다
from ui.chat_actions import ChatActionsMixin
from ui.vue_bridge import VueBridge
from workers.chat_worker import ChatWorker


def _png_bytes(color='green'):
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color).save(buffer, format='PNG')
    return buffer.getvalue()


class _Stream:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_lines(self, **_):
        yield json.dumps({'message': {'content': 'ok'}, 'done': True})


class LooksLikePathTests(unittest.TestCase):
    def test_base64_bodies_are_never_paths(self):
        jpeg = base64.b64encode(b'\xff\xd8\xff\xe0' + bytes(range(256)) * 4).decode('ascii')
        self.assertTrue(jpeg.startswith('/9j/'))
        self.assertIn('/', jpeg[4:])
        for value in (jpeg, base64.b64encode(_png_bytes()).decode('ascii'), 'data:image/png;base64,iVBOR', '', None):
            with self.subTest(value=str(value)[:12]):
                self.assertFalse(looks_like_image_path(value))

    def test_windows_and_posix_paths_are_paths(self):
        for value in (r'C:\images\a.png', 'C:/images/a.png', '/home/user/a.png', r'\\server\share\a.webp',
                      'generated_images/2026/a.png'):
            with self.subTest(value=value):
                self.assertTrue(looks_like_image_path(value))

    def test_inlining_after_data_url_stripping_keeps_bare_base64(self):
        encoded = base64.b64encode(b'\xff\xd8\xff\xe0' + bytes(range(256))).decode('ascii')
        stripped = build_ollama_messages([{'role': 'user', 'content': 'x', 'images': ['data:image/jpeg;base64,' + encoded]}])
        self.assertEqual(inline_image_paths(stripped)[0]['images'], [encoded],
                         "'/' 가 든 맨 base64 를 경로로 오판해 버리면 안 된다")


class ChatSendThreadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def _host(self):
        class Host(QObject, ChatActionsMixin):
            def __init__(self):
                super().__init__()
                self.vue_bridge = VueBridge(self)
        return Host()

    def test_only_kept_turns_are_read_and_only_on_the_worker_thread(self):
        turns = []
        paths = []
        for index in range(30):   # 24턴 상한을 넘는 긴 대화
            path = self.root / f'turn{index}.png'
            path.write_bytes(_png_bytes())
            paths.append(str(path))
            turns.append({'role': 'user', 'content': f'q{index}', 'images': [str(path)]})
        host = self._host()
        started = []
        opened = []
        real_open = builtins.open

        def spy_open(file, *args, **kwargs):
            if str(file) in paths:
                opened.append((str(file), threading.current_thread() is threading.main_thread()))
            return real_open(file, *args, **kwargs)

        with patch.object(ChatWorker, 'start', new=lambda worker: started.append(worker)), \
                patch('builtins.open', side_effect=spy_open), \
                patch.object(host, '_chat_start_media'):
            host._handle_chat_action('chat_send', {
                'id': 'long-thread', 'provider': 'ollama', 'model': 'vision', 'messages': turns,
                'generation': {'mode': 'chat'},
            })
        self.assertEqual(opened, [], 'GUI 스레드(onAction)에서는 파일을 읽지 않는다')
        worker = started[0]
        self.assertEqual(len(worker.messages), 24)
        self.assertEqual(worker.messages[0]['images'], [paths[6]], '잘린 뒤의 메시지만 워커로 간다')

        posts = []

        def post(url, json=None, **_):
            posts.append(json)
            return _Stream()

        with patch('core.ollama_client.requests.post', side_effect=post), \
                patch('builtins.open', side_effect=spy_open):
            QThread.start(worker)   # 가려 둔 start 대신 실제 QThread 로 — 시그널이 정상 경로로 간다
            self.assertTrue(worker.wait(10000))
        for _ in range(3):   # 워커의 done/finished 큐 이벤트를 host 가 살아 있을 때 처리한다
            self.app.processEvents()
        self.assertEqual(sorted(p for p, _ in opened), sorted(paths[6:]), '잘려 나간 옛 턴의 파일은 읽지 않는다')
        self.assertTrue(all(not on_main for _p, on_main in opened), '파일 읽기는 워커 스레드에서')
        encoded = base64.b64encode(_png_bytes()).decode('ascii')
        self.assertEqual(posts[0]['messages'][0]['images'], [encoded])


if __name__ == '__main__':
    unittest.main()
