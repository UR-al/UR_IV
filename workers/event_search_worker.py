# workers/event_search_worker.py
"""이벤트 검색 비동기 워커 — UI 블로킹 방지 + 진행도"""
import json
from PyQt6.QtCore import QThread, pyqtSignal


class EventSearchWorker(QThread):
    """EventDataLoader.search_by_prompt를 비동기로 실행"""
    finished = pyqtSignal(str)       # JSON results
    progress = pyqtSignal(int, int)  # (current, total)

    def __init__(self, loader, params: dict, parent=None):
        super().__init__(parent)
        self._loader = loader
        self._params = params

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
        try:
            results = self._loader.search_by_prompt(
                prompt=self._params.get('prompt', ''),
                exclude_tags=self._params.get('exclude_tags', ''),
                min_children=self._params.get('min_steps', 2),
                max_children=self._params.get('max_steps', 20),
                limit=100 if self._params.get('limit') else 5000,
                progress_callback=lambda cur, total: self.progress.emit(cur, total),
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

            self.finished.emit(json.dumps(out))
        except Exception as e:
            self.finished.emit(json.dumps({'error': str(e)}))
