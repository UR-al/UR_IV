# tools/refresh_character_tags.py
# -*- coding: utf-8 -*-
"""Danbooru 라이브 태그로 정식 캐릭터 프로필의 외견(core_tags)을 교정·갱신.

동시성(ThreadPool) + 전역 레이트 리미터로 danbooru 10 req/s 천장 바로 아래를 채워
순차 방식(~14시간) 대비 약 10배 빠르게 처리. 중단(Ctrl+C)/재개 가능.

기본 fetch: related_tag.json (캐릭터당 1요청, 서버측 집계). frequency = P(tag|캐릭터)
라서 임계값(>=0.2) 그대로 적용. related 결과가 비면 posts.json 샘플링으로 자동 폴백.

사용 (앱·기존 refresh 종료 후 실행 권장 — 같은 IP가 10 req/s 한도를 공유):
  python tools/refresh_character_tags.py                      # 전체(34k, ~70분 @ 8rps)
  python tools/refresh_character_tags.py --min-count 100      # post_count 100+ 만
  python tools/refresh_character_tags.py --limit 500          # 인기 상위 500 (테스트)
  python tools/refresh_character_tags.py --rps 6              # 더 느리게/안전 (429 보이면)
  python tools/refresh_character_tags.py --workers 16 --rps 8 # 동시성/속도 조절
  python tools/refresh_character_tags.py --method posts       # 기존 posts.json 방식 강제
  python tools/refresh_character_tags.py --login NAME --api-key KEY   # 인증(권장)
  python tools/refresh_character_tags.py --dry-run            # 저장 안 함(테스트/측정용)

중단해도 진행도와 1회 백업은 ``user_data/tag_refresh/``에 저장되어 다시 실행하면
이어집니다. 런타임 부산물을 ``tags_db`` 안에 섞지 않습니다.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from core.tag_database import TagAsset, get_tag_database
from utils.atomic_json import atomic_write_json

CHAR_JSON = str(get_tag_database().path(TagAsset.CHARACTER_PROFILES))
STATE_DIR = os.path.join(BASE, "user_data", "tag_refresh")
PROGRESS = os.path.join(STATE_DIR, "progress.json")
BACKUP = os.path.join(STATE_DIR, "character_profiles.original.json")
HDR = {"User-Agent": "UR_IV/1.0 character tag refresh (personal)"}
SESSION = requests.Session()
SESSION.headers.update(HDR)

_COUNT_RE = re.compile(r"^\d+\+?(girl|boy|other)s?$")
_EXTRA_COUNT = {"solo", "solo focus", "multiple girls", "multiple boys",
                "multiple others", "no humans"}


def is_count_tag(n):
    return bool(_COUNT_RE.match(n)) or n in _EXTRA_COUNT


class RateLimiter:
    """전역 토큰 슬롯 리미터 — 워커 수와 무관하게 초당 rps 회로 제한."""

    def __init__(self, rps):
        self.min_interval = (1.0 / rps) if rps and rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def make_filter():
    """tag_intelligence 로 외견 태그 판별기 구성 (없으면 느슨한 폴백)."""
    try:
        from core.tag_intelligence import get_tag_intelligence
        ti = get_tag_intelligence(); ti._ensure()

        def keep(tn):
            if is_count_tag(tn):
                return False
            cat = ti.category_of(tn)
            top = cat.split(">")[0].strip() if cat else ""
            if top in ("신체", "패션"):
                return True
            if cat.startswith("인물 > 눈") or cat.startswith("인물 > 머리"):
                return True
            return ti.is_appearance(tn) or ti.is_clothing(tn)
        return keep
    except Exception as e:
        print(f"[warn] tag_intelligence 로드 실패({e}) — 느슨한 폴백 사용")
        return lambda tn: not is_count_tag(tn)


def _get(url, params, limiter, auth):
    """레이트 리미터 통과 + 인증 파라미터 병합 후 GET. 429/503 백오프."""
    p = dict(params)
    if auth:
        p.update(auth)
    for attempt in range(3):
        limiter.acquire()
        try:
            r = SESSION.get(url, params=p, timeout=20)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2.0)
    return None


def fetch_related(tag, keep, limiter, auth):
    """related_tag.json — 서버측 집계 general 태그(frequency>=0.2, 상위 28). 실패/빈 결과 시 None."""
    data = _get("https://danbooru.donmai.us/related_tag.json",
                {"query": tag, "category": 0}, limiter, auth)
    if not isinstance(data, dict):
        return None
    related = data.get("related_tags") or []
    if not related:
        return None
    query_name = tag
    q = data.get("query")
    if isinstance(q, str):
        query_name = q
    elif isinstance(data.get("tag"), dict):
        query_name = data["tag"].get("name") or tag
    scored = []
    for item in related:
        if not isinstance(item, dict):
            continue
        tobj = item.get("tag") or {}
        if tobj.get("category") != 0:
            continue
        name = tobj.get("name")
        if not name or name == query_name:
            continue
        freq = item.get("frequency")
        if freq is None:
            freq = item.get("overlap_coefficient") or item.get("cosine_similarity") or 0.0
        scored.append((name, float(freq)))
    scored.sort(key=lambda x: x[1], reverse=True)
    feats = []
    for name, freq in scored:
        if freq < 0.2:
            break
        tn = name.replace("_", " ")
        if keep(tn):
            feats.append(tn)
        if len(feats) >= 28:
            break
    return feats if len(feats) >= 3 else None


def fetch_posts(tag, keep, limiter, auth):
    """posts.json 100표본 집계 (frequency>=0.2, 상위 28). 검증된 폴백/대체 방식."""
    for q in (f"{tag} solo", tag):
        posts = _get("https://danbooru.donmai.us/posts.json",
                     {"tags": q, "limit": 100, "only": "tag_string_general"},
                     limiter, auth)
        if not isinstance(posts, list) or not posts:
            continue
        cnt, n = Counter(), 0
        for p in posts:
            g = p.get("tag_string_general") if isinstance(p, dict) else ""
            if not g:
                continue
            n += 1
            for t in g.split():
                cnt[t] += 1
        if not n:
            continue
        feats = []
        for t, c in cnt.most_common(70):
            if c / n < 0.2:
                break
            tn = t.replace("_", " ")
            if keep(tn):
                feats.append(tn)
        if len(feats) >= 3:
            return feats[:28]
    return None


def fetch_features(tag, keep, limiter, auth, method):
    """method='related'(기본): related_tag 우선, 비면 posts 폴백. 'posts': posts만."""
    if method == "related":
        f = fetch_related(tag, keep, limiter, auth)
        if f:
            return f
        return fetch_posts(tag, keep, limiter, auth)
    return fetch_posts(tag, keep, limiter, auth)


def _save(data, done, dry):
    if dry:
        return
    # 원자적 교체 (부분 쓰기 손상 방지) — 공용 구현 한 벌(fsync + 실패 시 tmp 정리).
    # 진행 파일도 같은 방식: 중단 시 절단된 progress.json 이 재개를 막지 않게.
    atomic_write_json(CHAR_JSON, data, indent=None, separators=(",", ":"))
    atomic_write_json(PROGRESS, sorted(done), indent=None, ensure_ascii=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=0, help="이 post_count 이상만 갱신")
    ap.add_argument("--limit", type=int, default=0, help="인기 상위 N개만 (0=전체)")
    ap.add_argument("--workers", type=int, default=16, help="동시 요청 워커 수")
    ap.add_argument("--rps", type=float, default=8.0, help="초당 최대 요청 (danbooru 한도 10)")
    ap.add_argument("--method", choices=["related", "posts"], default="related",
                    help="related_tag(기본) 또는 posts.json 샘플링")
    ap.add_argument("--login", default="", help="danbooru 계정명 (인증, 선택)")
    ap.add_argument("--api-key", default="", help="danbooru API key (인증, 선택)")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않음 (측정/테스트)")
    args = ap.parse_args()

    auth = {}
    if args.login and args.api_key:
        auth = {"login": args.login, "api_key": args.api_key}

    with open(CHAR_JSON, encoding="utf-8") as f:
        data = json.load(f)
    if not args.dry_run and not os.path.exists(BACKUP):
        import shutil
        os.makedirs(STATE_DIR, exist_ok=True)
        shutil.copy2(CHAR_JSON, BACKUP)
        print(f"백업: {BACKUP}")

    done = set()
    if os.path.exists(PROGRESS):
        try:
            done = set(json.load(open(PROGRESS, encoding="utf-8")))
        except Exception:
            pass

    targets = [e for e in data if e.get("tag") and e["tag"] not in done
               and (e.get("post_count") or 0) >= args.min_count]
    targets.sort(key=lambda e: -(e.get("post_count") or 0))
    if args.limit:
        targets = targets[:args.limit]

    keep = make_filter()
    limiter = RateLimiter(args.rps)
    total = len(targets)
    eta_h = total / args.rps / 3600 if args.rps else 0
    print(f"대상 {total:,}개 (완료 {len(done):,}) · {args.method} · "
          f"워커 {args.workers} · {args.rps} req/s · "
          f"인증 {'on' if auth else 'off'}{' · DRY-RUN' if args.dry_run else ''} · "
          f"예상 ~{eta_h:.1f}시간")

    updated = 0
    t0 = time.time()

    def work(e):
        return e, fetch_features(e["tag"].strip().lower(), keep, limiter, auth, args.method)

    ex = ThreadPoolExecutor(max_workers=args.workers)
    futures = [ex.submit(work, e) for e in targets]
    try:
        for i, fut in enumerate(as_completed(futures)):
            e, feats = fut.result()
            if feats:
                e["core_tags"] = feats
                updated += 1
            done.add(e["tag"])
            if (i + 1) % 100 == 0 or (i + 1) == total:
                _save(data, done, args.dry_run)
                el = time.time() - t0
                rate = (i + 1) / el if el else 0
                eta = (total - i - 1) / rate / 3600 if rate else 0
                print(f"  {i+1:,}/{total:,} · 갱신 {updated:,} · "
                      f"{rate:.1f} char/s · ETA {eta:.1f}h"
                      f"{' (미저장)' if args.dry_run else ' · 저장됨'}")
        ex.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\n중단됨 — 진행도 저장 중...")
        ex.shutdown(wait=False, cancel_futures=True)
        _save(data, done, args.dry_run)
        print("저장 완료. 다시 실행하면 이어서 진행됩니다.")
        return
    print(f"\n완료: {updated:,}개 갱신 / {total:,}개 처리 "
          f"({(time.time()-t0)/60:.1f}분)")
    if not args.dry_run:
        print("앱 재시작하면 갱신된 캐릭터 외견이 반영됩니다.")


if __name__ == "__main__":
    main()
