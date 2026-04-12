# core/gen_stats.py
"""생성 통계 추적 및 집계"""
import json
import os
import threading
from collections import Counter
from datetime import datetime, timedelta

_STATS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'gen_stats.json')
_MAX_RECORDS = 5000  # 최대 보관 레코드 수
_instance = None


class GenStats:
    """생성 통계 매니저 (스레드 안전)"""

    def __init__(self, path: str = _STATS_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._records = self._load()

    def _load(self) -> list:
        try:
            if os.path.exists(self._path):
                with open(self._path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[gen_stats] 로드 실패: {e}")
        return []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            # 원자적 쓰기: tmp 파일에 쓰고 rename
            tmp = self._path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._records, f, ensure_ascii=False)
            os.replace(tmp, self._path)
        except Exception as e:
            print(f"[gen_stats] 저장 실패: {e}")

    def record(self, entry: dict):
        """생성 레코드 추가 (스레드 안전)"""
        with self._lock:
            entry.setdefault('timestamp', datetime.now().isoformat(timespec='seconds'))
            self._records.append(entry)
            # 메모리 내 절단
            if len(self._records) > _MAX_RECORDS:
                self._records = self._records[-_MAX_RECORDS:]
            self._save()

    def get_summary(self) -> dict:
        """집계 통계 반환"""
        records = self._records
        total = len(records)
        if total == 0:
            return {
                'total': 0, 'success': 0, 'fail': 0,
                'success_rate': 0, 'avg_time': 0,
                'total_time': 0,
                'daily': [], 'daily_max': 0,
                'top_models': [], 'recent': [],
            }

        success = sum(1 for r in records if r.get('success'))
        fail = total - success
        times = [r.get('duration_sec', 0) for r in records if r.get('success') and r.get('duration_sec')]
        avg_time = round(sum(times) / len(times), 1) if times else 0
        total_time = round(sum(times), 1)

        # 일별 통계 (최근 30일)
        today = datetime.now().date()
        daily_counts = Counter()
        for r in records:
            try:
                d = datetime.fromisoformat(r['timestamp']).date()
                if (today - d).days <= 30:
                    daily_counts[d.isoformat()] += 1
            except Exception:
                pass
        daily = []
        for i in range(30, -1, -1):
            d = (today - timedelta(days=i)).isoformat()
            daily.append({'date': d, 'count': daily_counts.get(d, 0)})
        daily_max = max((d['count'] for d in daily), default=0)

        # 모델별 통계 (TOP 5)
        model_counts = Counter(r.get('model', 'unknown') for r in records)
        top_models = [{'name': m, 'count': c} for m, c in model_counts.most_common(5)]

        # 최근 10건
        recent = records[-10:][::-1]

        # 해상도별 통계
        res_counts = Counter(f"{r.get('width', '?')}x{r.get('height', '?')}" for r in records)
        top_res = [{'res': r, 'count': c} for r, c in res_counts.most_common(5)]

        return {
            'total': total,
            'success': success,
            'fail': fail,
            'success_rate': round(success / total * 100, 1) if total else 0,
            'avg_time': avg_time,
            'total_time': total_time,
            'daily': daily,
            'daily_max': daily_max,
            'top_models': top_models,
            'top_resolutions': top_res,
            'recent': recent,
        }


def get_gen_stats() -> GenStats:
    """싱글턴 인스턴스 반환"""
    global _instance
    if _instance is None:
        _instance = GenStats()
    return _instance
