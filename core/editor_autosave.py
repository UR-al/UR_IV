"""에디터 크래시 복구본(자동저장) 쓰기 — Qt 비의존, 작업 스레드에서 부른다.

복구본은 OS 임시 폴더의 ``AIStudioPro_editor/_autosave_session.png`` 한 장과 메타 JSON 이다.
병합 안 한 드로잉 레이어가 있으면 수동 저장처럼 합성(레이어 PNG 디코드 → 합성 → PNG 인코딩 →
fsync)해서 쓰는데, CPU 로 1216×832 에 약 0.2초, 2048² 에 0.6초, 4K 에 1초가 넘는다(그냥 복사는
수 ms). 예전 동기 슬롯(``editorAutoSave``)은 이 일을 GUI 스레드에서 해서, 5분 타이머가 돌 때마다
창(웹 모드는 모든 클라이언트)이 그리는 도중에도 멈췄다. 지금은 ``VueBridge.requestEditorAutoSave``
가 워커에서 :meth:`AutosaveWriter.write` 를 부르고 ``editorAutoSaveReady`` 로 알린다.

워커로 옮기면서 생긴 두 경합을 여기서 막는다.

* **쓰기끼리** — 복구본은 한 장이라 두 쓰기가 겹치면 그림과 메타가 서로 다른 요청의 것이 될 수
  있다(웹 모드는 탭마다 타이머가 돈다). 쓰기는 한 줄로 세운다.
* **쓰기와 폐기** — 저장에 성공하면 프론트가 복구본을 지운다(``editorClearAutoSave``). 그 직전에
  시작한 자동저장이 폐기보다 늦게 끝나면, 이미 저장한 작업의 복구본이 되살아나 다음 시작 때
  '복구할까요?'를 묻는다. 폐기는 세대만 올리고(:meth:`AutosaveWriter.invalidate` — 쓰기 잠금을
  기다리지 않으니 GUI 스레드를 막지 않는다), 요청 때 받은 세대가 낡은 쓰기는 쓰지 않거나
  이미 쓴 것을 지운다.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Optional

AUTOSAVE_DIR_NAME = 'AIStudioPro_editor'
AUTOSAVE_IMAGE = '_autosave_session.png'
AUTOSAVE_META = '_autosave_session.meta.json'


def autosave_dir() -> str:
    """복구본 폴더. 부를 때마다 계산한다(``tempfile.gettempdir`` 를 따라간다)."""
    return os.path.join(tempfile.gettempdir(), AUTOSAVE_DIR_NAME)


def remove_autosave_files(folder: Optional[str] = None) -> bool:
    """복구본(그림·메타)을 지운다. 없는 파일은 넘어간다. 하나라도 못 지웠으면 False."""
    folder = folder or autosave_dir()
    ok = True
    for name in (AUTOSAVE_IMAGE, AUTOSAVE_META):
        try:
            os.remove(os.path.join(folder, name))
        except FileNotFoundError:
            pass
        except OSError:
            ok = False
    return ok


class AutosaveWriter:
    """복구본 쓰기를 한 줄로 세우고, 쓰는 사이 폐기가 오면 쓴 것을 버린다."""

    def __init__(self) -> None:
        self._write_lock = threading.Lock()   # 쓰기끼리(오래 잡힌다 — 워커만 잡는다)
        self._epoch_lock = threading.Lock()   # 세대 읽기·올리기(짧다 — GUI 스레드도 잡는다)
        self._epoch = 0

    def ticket(self) -> int:
        """요청 시점의 세대. 요청을 받는 스레드(GUI)에서 받아 :meth:`write` 에 넘긴다."""
        with self._epoch_lock:
            return self._epoch

    def invalidate(self) -> None:
        """복구본 폐기를 알린다 — 이 전에 받은 세대로 하는 쓰기는 결과를 남기지 않는다."""
        with self._epoch_lock:
            self._epoch += 1

    def is_current(self, ticket: int) -> bool:
        with self._epoch_lock:
            return ticket == self._epoch

    def write(self, src: str, overlay_base64: Optional[str], overlay_opacity: float, ticket: int,
              folder: Optional[str] = None) -> dict:
        """확정 이미지 ``src`` (+ 드로잉 레이어)를 복구본으로 쓴다. 작업 스레드에서 부른다.

        :param overlay_opacity: 0~1.
        :param ticket: 요청 때 받은 :meth:`ticket`.
        :returns: ``{'path', 'drawing'}`` — 썼다. ``{'discarded': True}`` — 쓰기 전·중에 폐기됐다
            (남은 복구본 없음).
        :raises EditorSaveError: 이미지·레이어를 읽거나 쓰지 못했을 때.
        :raises OSError: 폴더·메타를 만들지 못했을 때.
        """
        from core.editor_save import write_autosave_snapshot
        folder = folder or autosave_dir()
        with self._write_lock:
            if not self.is_current(ticket):
                return {'discarded': True}
            os.makedirs(folder, exist_ok=True)
            dst = os.path.join(folder, AUTOSAVE_IMAGE)
            drawing = write_autosave_snapshot(src, dst, overlay_base64, overlay_opacity)
            meta = {'original': src, 'saved_at': int(time.time()), 'drawing': drawing}
            with open(os.path.join(folder, AUTOSAVE_META), 'w', encoding='utf-8') as handle:
                json.dump(meta, handle, ensure_ascii=False)
            if not self.is_current(ticket):
                # 쓰는 사이 저장이 끝나 복구본을 폐기했다 — 방금 쓴 것은 이미 저장한 작업이다.
                # (폐기가 이 확인보다 늦으면 폐기 쪽이 지운다 — 어느 쪽이든 남지 않는다)
                remove_autosave_files(folder)
                return {'discarded': True}
            return {'path': dst.replace('\\', '/'), 'drawing': drawing}


#: 앱 전체에서 하나 — VueBridge 의 자동저장·폐기 슬롯이 같이 쓴다.
AUTOSAVE = AutosaveWriter()
