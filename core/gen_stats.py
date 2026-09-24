# core/gen_stats.py
"""생성 통계 추적 및 집계"""
import json
import os
import threading
from collections import Counter
from datetime import datetime, timedelta

from core.storage_paths import user_data_file
from utils.atomic_json import atomic_write_json

_STATS_PATH = str(user_data_file(
    'stats/generation.json',
    legacy_paths='config/gen_stats.json',
))
_MAX_RECORDS = 5000  # 최대 보관 레코드 수
_instance = None


def _as_int(value):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number


def request_meta_from_payload(model, payload) -> dict:
    """start_generation 이 워커에 넘긴 **실제 요청**에서 통계용 메타를 뽑는다.

    UI 위젯(model_combo/width_input)은 고해상도 배율·Anima 해상도 가드·XYZ 축·Comfy 스냅샷
    큐가 바꾼 값을 모르고, 빈 칸이면 int('') 예외로 성공 레코드가 통째로 빠졌다(감사 #112).
    model 은 콤보 title 형식(선택/override 값) 그대로 — 기존 통계와 이어지게.
    단, Krea2 family 요청(payload['_generation_family']=='krea2')은 체크포인트를 쓰지 않는
    ComfyUI 워크플로라, Krea2 모드에서 숨겨진 Standard 체크포인트 title 대신 Krea2 라벨로 기록한다.
    (워커가 run() 에서 이 키를 pop 하므로 반드시 워커 시작 전에 호출할 것.)
    """
    payload = payload if isinstance(payload, dict) else {}
    if str(payload.get('_generation_family') or '').strip().lower() == 'krea2':
        from core.generation_family import KREA2_LABEL
        model = KREA2_LABEL
    return {
        'model': str(model or ''),
        'width': _as_int(payload.get('width')),
        'height': _as_int(payload.get('height')),
        'seed': _as_int(payload.get('seed')),
    }


def build_generation_record(*, success: bool, duration_sec: float,
                            request_meta=None, gen_info=None) -> dict:
    """통계 레코드 한 건. 실패 레코드는 설계상 model 만 남긴다(해상도·시드 없음).

    시드는 백엔드가 확정한 값(gen_info['seed'], -1 해석 후)을 우선하고, 없으면 요청값.
    """
    meta = request_meta if isinstance(request_meta, dict) else {}
    record = {
        'success': bool(success),
        'duration_sec': duration_sec,
        'model': str(meta.get('model') or ''),
    }
    if not success:
        return record
    info = gen_info if isinstance(gen_info, dict) else {}
    seed = _as_int(info.get('seed'))
    if seed is None:
        seed = meta.get('seed')
    record['seed'] = seed if seed is not None else 0
    record['width'] = meta.get('width') or 0
    record['height'] = meta.get('height') or 0
    return record


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
            # 원자적 쓰기는 공용 구현 한 벌(fsync + 실패 시 tmp 정리). 레코드가 많아 compact.
            atomic_write_json(self._path, self._records, indent=None)
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
