# core/event_data_loader.py
"""
이벤트 데이터 로더 - variant_set 기반 시퀀스 검색
Step 0 = Parent (베이스), Step 1+ = Children (변형)

개선사항:
- 유사도 기반 프롬프트 검색 (Jaccard similarity)
- Children ID순 정렬 (스토리 순서 보장)
- 이전 스텝 기준 diff (스토리 진행감)

pandas 는 쓰는 메서드 안에서 import 한다 — 숨은 레거시 EventGenTab 이 창 표시 전에 이
모듈을 import 하므로, 모듈 최상단 import 는 기동 임계 경로에 pandas(+pyarrow)를 얹었다.
"""
from pathlib import Path

from core.tag_matcher import contains_tag_text


class EventSearchCancelled(Exception):
    """search_by_prompt 가 cancel_check 로 중단됐다 (새 요청이 이전 검색을 대체)."""


class EventDataLoader:
    """이벤트 데이터 로더 - variant_set 검색용"""

    REQUIRED_COLUMNS = [
        'id', 'parent_id', 'has_children', 'has_visible_children',
        'tag_string_general', 'tag_string_character',
        'tag_string_copyright', 'tag_string_artist', 'tag_string_meta',
        'rating', 'score', 'fav_count',
        'image_width', 'image_height',
    ]

    def __init__(self, parquet_dir: str = None):
        self.parquet_dir = parquet_dir
        self.df = None
        self.parents_df = None
        self.children_df = None
        self.parent_child_map = {}
        # parent_id → children_df 행 위치 배열 (_build_parent_child_index 가 채운다)
        self._child_positions = {}
        # 불러온 shard 중 dataset_manifest 와 크기가 다른 것(상대 경로) — 옛 데이터를 조용히
        # 쓰지 않도록 호출자(UI)가 알린다. manifest 가 없으면 판단하지 않는다(빈 목록).
        self.stale_shards: list[str] = []

    def _find_stale_shards(self, loaded_files: list) -> list:
        """불러온 파일 중 manifest(parquet_dir 또는 그 부모)의 size_bytes 와 다른 것."""
        if not loaded_files or not self.parquet_dir:
            return []
        try:
            from core.dataset_artifacts import read_manifest, stale_among

            base = Path(self.parquet_dir)
            for root in (base, base.parent):
                manifest = read_manifest(root)
                if manifest is not None:
                    return stale_among(manifest, root, loaded_files, kind='event_graph')
        except Exception as exc:
            print(f"[EventData] manifest 확인 실패(무시): {exc}")
        return []

    def load_parquets_by_rating(self, ratings: list = None, progress_callback=None):
        """Rating별 parquet 파일 로드 (고속 버전)"""
        import pandas as pd

        if ratings is None:
            ratings = ['e']
        loaded_files = []

        rating_files = {
            'g': 'danbooru_g.parquet',
            's': 'danbooru_s.parquet',
            'q': 'danbooru_q.parquet',
            'e': 'danbooru_e.parquet',
        }

        dfs = []
        total_before = 0
        total_after = 0

        for i, rating in enumerate(ratings):
            filename = rating_files.get(rating)
            if not filename:
                continue

            filepath = Path(self.parquet_dir) / filename
            if not filepath.exists():
                print(f"⚠️ {filename} 파일 없음")
                continue

            try:
                # ★ 필요한 컬럼만 읽기 (I/O 대폭 감소)
                df = pd.read_parquet(filepath, columns=self.REQUIRED_COLUMNS)

                total_before += len(df)

                # parent_id가 있거나 has_children인 것만 필터링
                mask = df['parent_id'].notna()
                if 'has_children' in df.columns:
                    mask = mask | (df['has_children'] == True)
                df = df[mask]

                total_after += len(df)

                # ★ 벡터화된 parent_id 정수 변환 (apply 대신)
                if 'parent_id' in df.columns:
                    df['parent_id'] = pd.to_numeric(df['parent_id'], errors='coerce').astype('Int64')

                if 'id' in df.columns:
                    df['id'] = df['id'].astype(int)

                dfs.append(df)
                loaded_files.append(filepath)
                print(f"✅ {filename}: {len(df)}개 로드")

                if progress_callback:
                    progress_callback(i + 1, len(ratings), filename)

            except Exception as e:
                # columns 파라미터 실패 시 폴백
                try:
                    df = pd.read_parquet(filepath)
                    available_cols = [c for c in self.REQUIRED_COLUMNS if c in df.columns]
                    df = df[available_cols]

                    total_before += len(df)
                    mask = df['parent_id'].notna()
                    if 'has_children' in df.columns:
                        mask = mask | (df['has_children'] == True)
                    df = df[mask]
                    total_after += len(df)

                    if 'parent_id' in df.columns:
                        df['parent_id'] = pd.to_numeric(df['parent_id'], errors='coerce').astype('Int64')
                    if 'id' in df.columns:
                        df['id'] = df['id'].astype(int)

                    dfs.append(df)
                    loaded_files.append(filepath)
                    print(f"✅ {filename}: {len(df)}개 로드 (폴백)")
                    if progress_callback:
                        progress_callback(i + 1, len(ratings), filename)
                except Exception as e2:
                    print(f"⚠️ {filename} 로드 실패: {e2}")
                    import traceback
                    traceback.print_exc()

        self.stale_shards = self._find_stale_shards(loaded_files)
        if self.stale_shards:
            print(
                "[EventData] dataset_manifest 와 크기가 다른 shard(구버전일 수 있음): "
                + ", ".join(self.stale_shards)
            )

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)

            # 최신 Event 샤드는 단일 등급만 선택해도 그래프가
            # 완결되도록, child가 참조하는 다른 등급의 parent를 각 샤드에
            # 복제해 둔다. 여러 등급을 함께 로드하면 같은 parent가 중복될
            # 수 있으므로 id를 기준으로 한 번만 남겨 인덱스를 안정화한다.
            duplicate_count = int(self.df.duplicated(subset=['id']).sum())
            if duplicate_count:
                self.df = self.df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

            duplicate_note = f", 중복 parent {duplicate_count}개 제거" if duplicate_count else ""
            print(
                f"✅ 총 {len(self.df)}개 로드 "
                f"(원본 {total_before}개 중{duplicate_note})"
            )
            self._build_parent_child_index(
                progress_callback=lambda msg: progress_callback(0, 0, msg) if progress_callback else None
            )

        return self.df

    def _build_parent_child_index(self, progress_callback=None):
        """Parent-Child 인덱스 구축 (고속 버전)"""
        if self.df is None:
            return
        import pandas as pd

        if progress_callback:
            progress_callback('Children 필터링...')

        # Parent가 있는 이미지들 (Children)
        self.children_df = self.df[self.df['parent_id'].notna()].copy()

        if progress_callback:
            progress_callback(f'Children: {len(self.children_df)}개 발견')

        # Parent ID 목록
        parent_ids = self.children_df['parent_id'].dropna().unique()

        # Parents (Children을 가진 이미지들) - set으로 빠른 lookup
        parent_id_set = set(parent_ids.astype(int))
        self.parents_df = self.df[self.df['id'].isin(parent_id_set)].copy()

        if progress_callback:
            progress_callback(f'Parents: {len(self.parents_df)}개 발견')

        # ★ Parent -> Children 매핑 생성 (groupby 한 번으로)
        # _child_positions 는 children_df 의 행 위치(iloc) 배열이다. 검색이 후보 parent
        # 마다 children_df 전체를 isin 으로 다시 훑던 것(q 28만 행 × 후보 6만 개 ≈ 2분)을
        # O(자식 수) 조회로 바꾼다. 위치는 원래 행 순서 그대로라 isin 결과와 같은 행·순서다.
        if progress_callback:
            progress_callback('Parent-Child 매핑 구축...')
        child_ids = self.children_df['id'].to_numpy()
        self._child_positions = {
            int(k): positions
            for k, positions in self.children_df.groupby('parent_id').indices.items()
        }
        self.parent_child_map = {
            parent_id: child_ids[positions].tolist()
            for parent_id, positions in self._child_positions.items()
        }

        # ★ Parents에 미리 태그 세트를 캐싱 (유사도 검색 고속화)
        if progress_callback:
            progress_callback('태그 인덱스 캐싱...')
        if 'tag_string_general' in self.parents_df.columns:
            self.parents_df['_tag_set'] = self.parents_df['tag_string_general'].apply(
                lambda x: set(
                    t.strip().lower().replace('_', ' ')
                    for t in str(x).split() if t.strip()
                ) if pd.notna(x) else set()
            )

        if progress_callback:
            progress_callback(f'완료: {len(self.parent_child_map)}개 그룹')

    # ──────────────────────────────────────────────────────
    #  A. 유사도 기반 프롬프트 검색 (신규)
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_tags(text: str) -> set:
        """쉼표/공백 구분 태그 문자열을 정규화된 set으로 변환"""
        if not isinstance(text, str):
            # parquet 결측(NaN/None/pd.NA)은 빈 태그다 — `',' in nan` 이 TypeError 로 검색 전체를 죽였다
            import pandas as pd

            try:
                if text is None or pd.isna(text):
                    return set()
            except (TypeError, ValueError):
                pass
            text = str(text)
        if not text:
            return set()
        # 쉼표로 먼저 분리, 없으면 공백
        if ',' in text:
            parts = text.split(',')
        else:
            parts = text.split()
        return set(
            t.strip().lower().replace('_', ' ')
            for t in parts if t.strip()
        )

    # ── 유사도 단위 — 사전 필터(core.tag_matcher)와 같은 문법으로 질의를 읽는다 ──
    # 예전에는 유사도를 _parse_tags 토큰으로 셌다. 그 토큰엔 '[1girl|1boy]'·'*1girl' 처럼 연산자가
    # 그대로 남고, OR/AND 그룹은 쉼표·공백에서 쪼개져 어떤 태그와도 맞지 않았다 — 매처가 통과시킨
    # 부모가 '최소 1개 일치' 단계에서 모두 탈락해 [A|B]·*x·_x_ 만 쓴 검색은 늘 0건이었고, 일반
    # 태그와 섞으면 연산자 토큰이 '안 맞은 태그'로 세어져 순위가 틀어졌다.

    @staticmethod
    def _term_alternative(term: str):
        """매처 텀 하나 → (방식, 정규화 텍스트). 방식은 tag_matcher._apply_pattern 과 같은 순서로 정한다."""
        from core.tag_matcher import _normalize

        text = (term or '').strip()
        if not text:
            return None
        if text.startswith('*'):
            mode, body = 'exact', text[1:]
        elif text.startswith('_') and text.endswith('_') and len(text) > 2:
            mode, body = 'contains', text[1:-1]
        elif text.startswith('_') and not text.endswith('_'):
            mode, body = 'suffix', text[1:]
        elif text.endswith('_') and not text.startswith('_'):
            mode, body = 'prefix', text[:-1]
        else:
            mode, body = 'contains', text
        body = _normalize(body)
        return (mode, body) if body else None

    @classmethod
    def _query_units(cls, prompt) -> list:
        """일반 태그 질의 → 유사도 단위 목록. 단위 = (대안, …), 대안 = (방식, 텍스트).

        방식: exact(``*x``) · suffix(``_x``) · prefix(``x_``) · contains(``_x_``·일반 태그).
        ``[A|B]`` 는 대안이 여럿인 단위 하나, ``[A, B]`` 는 텀마다 단위, 빈 대안이 붙은 ``[A|B|]``
        (항상 통과)는 빼고 센다. 쉼표도 대괄호도 없고 연산자도 없으면 예전처럼 공백으로 나눈다
        (danbooru 식 ``1girl smile``). 연산자가 붙은 텀이 있으면(``*school uniform``) 매처처럼 통째로
        한 텀으로 읽는다 — 공백으로 나누면 정확 ``school`` + 부분 ``uniform`` 두 단위가 되어, 매처가
        ``school uniform`` 태그로 통과시킨 부모가 늘 1/2 로 세어졌다. 같은 단위는 한 번만 센다.
        """
        from core.tag_matcher import parse_query

        text = prompt if isinstance(prompt, str) else ''
        if not text.strip():
            return []
        units: list = []
        parts = text.split()
        # 연산자 판정은 매처(tag_matcher._eval_condition)와 같다 — 앞의 * · _, 뒤의 _
        has_operator = any(part[0] in '*_' or part[-1] == '_' for part in parts)
        if ',' not in text and '[' not in text and not has_operator:
            candidates = [(cls._term_alternative(part),) for part in parts]
        else:
            candidates = []
            for cond in parse_query(text):
                terms = cond.get('terms') or []
                if cond['type'] == 'or':
                    if cond.get('has_wildcard'):
                        continue
                    candidates.append(tuple(cls._term_alternative(t) for t in terms))
                else:
                    candidates.extend((cls._term_alternative(t),) for t in terms)
        for unit in candidates:
            unit = tuple(dict.fromkeys(alt for alt in unit if alt))
            if unit and unit not in units:
                units.append(unit)
        return units

    @staticmethod
    def _unit_match(unit, tags):
        """단위가 맞은 부모 태그(없으면 None). 같은 글자 태그가 있으면 그것을 먼저 고른다."""
        for mode, text in unit:
            if text in tags:
                return text
            if mode == 'exact':
                continue
            for tag in tags:
                if mode == 'suffix':
                    hit = tag.endswith(text)
                elif mode == 'prefix':
                    hit = tag.startswith(text)
                else:
                    hit = text in tag
                if hit:
                    return tag
        return None

    @classmethod
    def _unit_similarity(cls, units, tags) -> tuple:
        """(맞은 단위 수, 유사도). 유사도 = 0.6·overlap + 0.4·jaccard (소수 셋째 자리 반올림).

        overlap = 맞은 단위 / 단위 수. jaccard = |맞은 부모 태그| / (단위 수 + |부모 태그| − 글자가
        같은 태그로 맞은 단위 수) — 연산자 없는 일반 태그만 쓰면 예전 _overlap_ratio·_jaccard_fuzzy
        (부분 문자열 일치 포함)와 같은 값이다.
        """
        count = len(units)
        if not count:
            return 0, 0.0
        tags = tags or set()
        matched = 0
        shared = 0
        hit_tags = set()
        for unit in units:
            tag = cls._unit_match(unit, tags)
            if tag is None:
                continue
            matched += 1
            hit_tags.add(tag)
            if any(text == tag for _mode, text in unit):
                shared += 1
        union = count + len(tags) - shared
        jaccard = len(hit_tags) / union if tags and union > 0 else 0.0
        return matched, round(0.6 * (matched / count) + 0.4 * jaccard, 3)

    @staticmethod
    def _query_mask(frame, col: str, query: str):
        """통합 태그 매처(core.tag_matcher) 마스크. 매처를 못 쓰면 None."""
        try:
            from core.tag_matcher import filter_dataframe
            return filter_dataframe(frame, col, query)
        except Exception:
            return None

    @classmethod
    def _contains_all_mask(cls, frame, col: str, query: str):
        """매처 폴백 — 쉼표로 나눈 태그가 모두 (공백/밑줄 양쪽) 부분 일치해야 참."""
        import pandas as pd

        mask = pd.Series(True, index=frame.index)
        lowered = frame[col].fillna('').astype(str).str.lower()
        for tag in cls._parse_tags(query):
            tag_u = tag.replace(' ', '_')
            mask &= (
                lowered.str.contains(tag_u, na=False, regex=False)
                | lowered.str.contains(tag, na=False, regex=False)
            )
        return mask

    def _ensure_child_positions(self):
        """parent_id → children_df 행 위치 색인. 외부에서 df 를 직접 채운 경우 여기서 만든다."""
        if self._child_positions or self.children_df is None or not len(self.children_df):
            return self._child_positions
        self._child_positions = {
            int(k): positions
            for k, positions in self.children_df.groupby('parent_id').indices.items()
        }
        return self._child_positions

    def search_by_prompt(
        self,
        prompt: str,
        exclude_tags: str = "",
        min_children: int = 2,
        max_children: int = 20,
        limit: int = 100,
        progress_callback=None,
        character: str = "",
        copyright: str = "",
        artist: str = "",
        cancel_check=None,
    ) -> list:
        """
        ★ 프롬프트 기반 유사도 검색 (핵심 개선)

        입력 프롬프트의 태그와 Parent 태그의 유사도를 계산하여
        가장 근접한 이벤트를 랭킹하여 반환합니다.

        유사도 = 0.6 * overlap_ratio + 0.4 * jaccard
        (overlap_ratio: 내 태그가 얼마나 포함되었는지 중시)

        character / copyright / artist 는 각 tag_string_* 열에 거는 필수 필터다
        (문법은 일반 태그와 같다: 쉼표=AND, [A|B]=OR, *정확 …). 일반 태그 없이
        이 필터만 줘도 검색되며, 그때는 유사도 1.0 으로 보고 score 순으로 줄 세운다.

        cancel_check: 인자 없는 호출이 True 를 돌려주면 EventSearchCancelled 를 던진다.

        순서는 예전 구현(후보마다 자식 구성 → 전부 모은 뒤 (유사도, score) 안정 정렬 →
        상위 limit) 과 같다. 다만 자식 수·유사도만으로 먼저 후보를 줄 세운 뒤
        그 순서대로 상위 limit 개만 자식을 꺼낸다.

        (숨은 레거시 EventGenTab 만 쓰던 child_include/child_exclude·min_score·require_variant_set
        인자와 그 분기는 탭과 함께 지웠다 — Vue 이벤트 검색(EventSearchWorker)은 넘긴 적이 없다.)
        """
        if self.parents_df is None or len(self.parents_df) == 0:
            return []

        prompt = prompt if isinstance(prompt, str) else ''
        exclude_tags = exclude_tags if isinstance(exclude_tags, str) else ''
        name_filters = [
            (column, query.strip())
            for column, query in (
                ('tag_string_character', character),
                ('tag_string_copyright', copyright),
                ('tag_string_artist', artist),
            )
            if isinstance(query, str) and query.strip()
        ]

        # 유사도 단위 — 사전 필터와 같은 문법([A|B]·[A, B]·*x·_x·x_·_x_)으로 읽는다(_query_units).
        # 단위가 없는 고급 문법(예: 항상 통과하는 [A|B|])은 사전 필터만 거르고 유사도 1.0 으로 본다.
        query_units = self._query_units(prompt)
        exclude_set = self._parse_tags(exclude_tags)

        if not prompt.strip() and not name_filters:
            return []

        def _cancelled() -> bool:
            return bool(cancel_check is not None and cancel_check())

        # ── 사전 필터: 전체 복사 없이 조건마다 부분집합만 남긴다 ──
        filtered = self.parents_df

        # 캐릭터·작품·작가 필수 필터 (열이 없으면 확인할 수 없으므로 통과시키지 않는다)
        for column, query in name_filters:
            if column not in filtered.columns:
                filtered = filtered.iloc[0:0]
                break
            mask = self._query_mask(filtered, column, query)
            if mask is None:
                mask = self._contains_all_mask(filtered, column, query)
            filtered = filtered[mask]

        # 프롬프트 사전 필터 (고급 문법: [], |, *, _ 지원). 매처를 못 쓰면 유사도만으로 거른다.
        prompt_prefiltered = False
        if prompt.strip() and 'tag_string_general' in filtered.columns:
            inc_mask = self._query_mask(filtered, 'tag_string_general', prompt)
            if inc_mask is not None:
                filtered = filtered[inc_mask]
                prompt_prefiltered = True

        # 제외 태그 필터
        if exclude_tags.strip() and 'tag_string_general' in filtered.columns:
            exc_mask = self._query_mask(filtered, 'tag_string_general', exclude_tags)
            if exc_mask is not None:
                filtered = filtered[~exc_mask]
            elif exclude_set:
                # 폴백: 기존 방식 (공백/밑줄 양쪽 부분일치, 한 번의 스캔)
                for tag in exclude_set:
                    mask = ~contains_tag_text(
                        filtered['tag_string_general'].str.lower(), tag
                    )
                    filtered = filtered[mask]

        total_candidates = len(filtered)
        print(f"[EventSearch] 사전 필터링 후 Parent 후보: {total_candidates}개")

        # ── 1단계: 자식 수 + 유사도만으로 후보 줄 세우기 (DataFrame 행 객체를 만들지 않는다) ──
        child_positions = self._ensure_child_positions()
        count = len(filtered)
        ids = filtered['id'].tolist() if 'id' in filtered.columns else []
        tag_sets = (
            filtered['_tag_set'].tolist() if '_tag_set' in filtered.columns else [None] * count
        )
        generals = (
            filtered['tag_string_general'].tolist()
            if 'tag_string_general' in filtered.columns else [''] * count
        )
        scores = filtered['score'].tolist() if 'score' in filtered.columns else [0] * count
        progress_step = max(50, count // 100)
        n_query = len(query_units)

        candidates = []   # (similarity, score, row_pos, parent_id, matched_count)
        for row_pos, raw_id in enumerate(ids):
            if row_pos % progress_step == 0:
                if _cancelled():
                    raise EventSearchCancelled()
                if progress_callback:
                    progress_callback(row_pos, total_candidates)
            try:
                parent_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            positions = child_positions.get(parent_id)
            if positions is None:
                continue
            # 자식 id 는 유일하다(로드 시 id 중복 제거) — 위치 개수 = isin 으로 고른 자식 수
            if len(positions) < min_children or len(positions) > max_children:
                continue

            if n_query:
                parent_tag_set = tag_sets[row_pos]
                if not parent_tag_set:
                    parent_tag_set = self._parse_tags(generals[row_pos])
                # 유사도 계산 (단위마다 매처와 같은 방식: 정확·접미·접두·부분 문자열)
                matched_count, similarity = self._unit_similarity(query_units, parent_tag_set)
                # 사전 필터를 못 쓴 경우에만 '최소 1개 일치'로 거른다 — 매처가 통과시킨 부모는 이미
                # 질의를 만족하므로 유사도는 순위만 정한다(예전엔 여기서 [A|B]·*x 결과가 모두 빠졌다).
                if matched_count == 0 and not prompt_prefiltered:
                    continue
            else:
                # 캐릭터/작품/작가 필터만 준 검색, 또는 단위가 없는 고급 문법(사전 필터만) —
                # 필터를 모두 만족했으니 유사도 1.0
                matched_count = 0
                similarity = 1.0

            score = scores[row_pos]
            if score is None or score != score:   # NaN 은 비교 불가 → 0 으로 줄 세움
                score = 0
            candidates.append((similarity, score, row_pos, parent_id, matched_count))

        if progress_callback:
            progress_callback(total_candidates, total_candidates)

        # ★ 유사도 내림차순 정렬, 같으면 score 내림차순 (안정 정렬 — 동률은 원래 행 순서)
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        # ── 2단계: 줄 선 순서대로 자식 구성, limit 개에서 멈춘다 ──
        children_df = self.children_df
        max_results = len(candidates) if limit is None else limit
        accepted = []   # (row_pos, entry)
        for similarity, _score, row_pos, parent_id, matched_count in candidates:
            if len(accepted) >= max_results:
                break
            if _cancelled():
                raise EventSearchCancelled()
            children = children_df.iloc[child_positions[parent_id]]

            # ★ Children을 ID순 정렬 (스토리 순서)
            children = children.sort_values('id', ascending=True)

            accepted.append((row_pos, {
                'parent': None,   # 아래에서 한 번에 채운다
                'children': children.to_dict('records'),
                'child_count': len(children),
                'similarity': similarity,
                'matched_tags': matched_count,
                'total_query_tags': n_query,
            }))

        # parent 행 dict 는 채택된 행만 만든다 — iterrows 로 만들어 예전 결과와 값 형식이 같다
        if accepted:
            parent_rows = filtered.iloc[[row_pos for row_pos, _entry in accepted]]
            for (_label, parent), (_row_pos, entry) in zip(parent_rows.iterrows(), accepted):
                entry['parent'] = parent.to_dict()

        results = [entry for _row_pos, entry in accepted]
        print(f"[EventSearch] 유사도 검색 결과: 후보 {len(candidates)}개 중 {len(results)}개 반환 (limit {limit})")
        return results
