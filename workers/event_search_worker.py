# workers/event_search_worker.py
"""이벤트 검색 비동기 워커 — UI 블로킹 방지 + 진행도"""
import json
import threading
from PyQt6.QtCore import QThread, pyqtSignal


class EventSearchWorker(QThread):
    """EventDataLoader.search_by_prompt를 비동기로 실행

    결과는 `search_finished(str)` 로 보낸다. 예전 이름 `finished` 는 QThread 자체의
    finished() 를 가려, 스레드 종료에 deleteLater 를 걸 수 없었다 — 그래서 끝난 워커가
    메인 창의 Qt 자식으로 쌓이며 적재본(loader)을 계속 붙들었다.
    """
    search_finished = pyqtSignal(str)  # JSON results
    progress = pyqtSignal(int, int)    # (current, total)

    def __init__(self, loader, params: dict, parent=None):
        super().__init__(parent)
        # 검색 중에만 쓴다 — run() 이 끝나면 놓는다(워커가 살아 있어도 적재본은 풀리게)
        self._loader = loader
        self._params = params if isinstance(params, dict) else {}
        # QThread.requestInterruption() 은 스레드가 아직(또는 이미) 돌지 않으면 무시된다 —
        # start() 직후 취소가 사라지지 않도록 자체 플래그를 함께 둔다.
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """진행 중인 검색을 멈춘다 — 새 검색이 이 요청을 대체할 때 부른다."""
        self._cancel_event.set()
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    @staticmethod
    def _text(value) -> str:
        return value if isinstance(value, str) else ''

    @staticmethod
    def _int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _convert_tags(raw: str) -> str:
        """danbooru 태그 형식 → SD 프롬프트 형식
        순서: 공백→콤마 먼저 (black_eyes를 보존), 그 다음 _→공백
        """
        if not raw:
            return ''
        raw = str(raw).strip()
        if not raw:
            return ''
        # 이미 콤마 구분이면 _ → 공백만
        if ',' in raw:
            tags = [t.strip().replace('_', ' ') for t in raw.split(',') if t.strip()]
        else:
            # 공백 구분 (danbooru 형식): 공백 → 콤마, 그 다음 _ → 공백
            tags = [t.replace('_', ' ') for t in raw.split() if t.strip()]
        return ', '.join(tags)

    def run(self):
        from core.event_data_loader import EventSearchCancelled

        params = self._params
        try:
            results = self._loader.search_by_prompt(
                prompt=self._text(params.get('prompt', '')),
                exclude_tags=self._text(params.get('exclude_tags', '')),
                min_children=self._int(params.get('min_steps', 2), 2),
                max_children=self._int(params.get('max_steps', 20), 20),
                limit=100 if params.get('limit') else 5000,
                progress_callback=lambda cur, total: self.progress.emit(cur, total),
                # EventGenView 의 캐릭터·작품·작가 입력 — 예전엔 여기서 버려져 필터가 조용히 빠졌다
                character=self._text(params.get('character', '')),
                copyright=self._text(params.get('copyright', '')),
                artist=self._text(params.get('artist', '')),
                cancel_check=self.is_cancelled,
            )

            out = []
            for ev in results:
                parent = ev.get('parent', {})
                children = ev.get('children', [])
                # 스텝 생성
                steps = []
                parent_prompt = self._convert_tags(parent.get('tag_string_general', ''))
                parent_tags = set(t.strip() for t in parent_prompt.split(',') if t.strip())
                steps.append({
                    'prompt': parent_prompt,
                    'type': 'parent',
                })
                prev_tags = parent_tags.copy()
                for child in children:
                    child_prompt = self._convert_tags(child.get('tag_string_general', ''))
                    child_tags = set(t.strip() for t in child_prompt.split(',') if t.strip())
                    added = sorted(child_tags - prev_tags)
                    removed = sorted(prev_tags - child_tags)
                    steps.append({
                        'prompt': child_prompt,
                        'type': 'child',
                        'added': added[:20],
                        'removed': removed[:20],
                    })
                    prev_tags = child_tags

                out.append({
                    'parent_tags': parent_prompt,
                    'character': self._convert_tags(parent.get('tag_string_character', '')),
                    'copyright': self._convert_tags(parent.get('tag_string_copyright', '')),
                    'children_count': ev.get('child_count', len(children)),
                    'similarity': ev.get('similarity', 0),
                    'steps': steps,
                })

            if self.is_cancelled():
                self.search_finished.emit(json.dumps({'cancelled': True}))
                return
            self.search_finished.emit(json.dumps(out))
        except EventSearchCancelled:
            self.search_finished.emit(json.dumps({'cancelled': True}))
        except Exception as e:
            self.search_finished.emit(json.dumps({'error': str(e)}))
        finally:
            # 적재본은 수백 MB 다 — 결과를 보낸 뒤엔 워커가 붙들지 않는다. 워커 객체는
            # 스레드 종료 뒤 deleteLater 될 때까지 남을 수 있고, 등급을 바꾸면 믹스인은
            # 옛 적재본을 놓는데 여기서 붙들면 둘이 함께 상주했다.
            self._loader = None
            self._params = {}
