# ui/worker_helpers.py
"""메인 윈도우 보조 헬퍼 — 이미지 폴더 스캔 및 QThread 워커 수명 정리."""
from __future__ import annotations

import os


def scan_image_folder(folder: str, exts: frozenset[str]) -> list[str]:
    """단일 패스 os.scandir 로 폴더 내 이미지 파일 수집.

    glob.glob() 은 경로에 '[', ']', '?', '*' 가 있으면 패턴으로 해석해 빈 목록을 돌려준다.
    """
    imgs: list[str] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
                        imgs.append(entry.path.replace('\\', '/'))
                except OSError:
                    continue
    except OSError:
        return []
    imgs.sort()
    return imgs


def attach_worker_cleanup(host, attr: str, worker) -> None:
    """QThread 종료 시 호스트 참조 해제 + deleteLater() (단일 워커의 finished 시그널 가림 대응)."""
    from PyQt6.QtCore import QThread

    if not isinstance(worker, QThread):
        return
    from ui.event_search_actions import thread_finished_signal

    def _on_thread_done():
        if getattr(host, attr, None) is worker:
            setattr(host, attr, None)
        try:
            worker.deleteLater()
        except RuntimeError:
            pass

    thread_finished_signal(worker).connect(_on_thread_done)
