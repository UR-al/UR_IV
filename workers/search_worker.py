# workers/search_worker.py
import os
import re
import json
import hashlib
import threading
# pandas(+pyarrow)는 검색을 실제로 돌릴 때(_run_locked/_load_data) import 한다 — 이 모듈이
# 창 표시 전 경로에서 import 돼도 pandas 로드(수백 ms)를 기동에 얹지 않는다.
from PyQt6.QtCore import QThread, pyqtSignal

from core.search_rows import (
    SEARCH_RESULT_CAP,
    SEARCH_ROW_KEYS,
    NormalizedSearchRows,
    normalize_search_rows_in_place,
)

class PandasSearchWorker(QThread):
    """Pandas를 이용한 검색 워커"""
    # object 시그니처(PyQt_PyObject) — list 로 선언하면 C++ 시그니처가 QVariantList 가
    # 되어 emit 마다 수십만 dict 가 QVariant 로 왕복 복사되고 GUI 스레드가 수 초 멈춘다.
    # 값은 언제나 NormalizedSearchRows(워커에서 정규화 완료된 list[dict]).
    results_ready = pyqtSignal(object, int)
    status_update = pyqtSignal(str)

    # 결과 행 상한(무작위 표본). bridge 도 이 값 하나만 쓴다.
    DEFAULT_RESULT_CAP = SEARCH_RESULT_CAP

    # 클래스 전역 캐시(cached_df) 보호 — 이전 워커가 1초 내 종료되지
    # 않아 새 워커와 겹칠 때, 서로 다른 연도/등급 로드가 캐시를 동시 변이해 인덱스 불일치
    # 마스크/오데이터/RAM 중복이 나던 경합 방지. run() 전체를 직렬화한다(검색은 사실상
    # 사용자 직렬이라 동시성 손실 무해).
    _run_lock = threading.RLock()

    # 검색에 필요한 컬럼만 로드 (메모리 절약)
    # image_width/height 추가 — 자동(PARQUET) 해상도 + Search 결과 해상도 표기용.
    # 필수 5개 컬럼은 로드 전에 검증하고, 선택 컬럼은 교집합만 읽어 기본값을 보충한다.
    # 현재 Search 릴리스의 고정 런타임 계약. score는 검색/결과에 쓰지 않아 제외한다.
    REQUIRED_COLUMNS = [
        'rating', 'general', 'character', 'copyright', 'artist',
        'meta', 'image_width', 'image_height',
    ]
    LOAD_COLUMNS = REQUIRED_COLUMNS

    # 태그 컬럼은 빌드(tools/refresh_danbooru_data.py)가 소문자로 보장하므로 질의용
    # 소문자 사본을 따로 캐시하지 않는다(기본 g 에서만 RSS 약 1.4GB 절약).
    cached_df = None
    loaded_ratings = set()
    loaded_year = ''       # 현재 캐시에 실제 로드된 릴리스
    loaded_file_signature = ()  # ((path, size, mtime_ns, file_id), ...)
    # SHA-256 검증은 큰 Search shard에서 비싸므로, 같은 manifest 내용과 같은 파일
    # identity가 이미 검증된 경우에만 재사용한다. 파일 교체/manifest 변경 시 key가
    # 달라져 다시 검증된다.
    verified_artifact_signatures = set()

    # dataset_manifest.json이 단일 활성 릴리스를 정한다. manifest가 없거나 검증에
    # 실패하면 닫힌 상태로 종료하며, 파일명 추측/구형 parquet 폴백은 하지 않는다.
    DEFAULT_DATASET_LABEL = '2026_07'
    MANIFEST_FORMAT_VERSION = 1

    # 결과로 내보내는 컬럼 — Search 행 계약(core.search_rows)의 키만.
    # to_dict 전에 이 컬럼만 남겨 안 쓰는 컬럼(meta)의 복제를 제거.
    OUTPUT_COLUMNS = list(SEARCH_ROW_KEYS)

    def __init__(self, parquet_dir, selected_ratings, queries, exclude_queries=None,
                 combine_mode: str = 'and', result_cap: int = None):
        """
        :param queries: { col: 'pattern' } 포함 검색 — 필드 간 결합은 combine_mode로 제어
        :param exclude_queries: { col: 'pattern' } 제외 검색 — 모드와 무관하게
            언제나 AND-NOT (제외는 "이건 절대 안 됨"의 의미라 OR 모드여도 항상 제외 적용)
        :param combine_mode: 'and' (교집합) | 'or' (합집합) — 필드 간 결합 방식
        :param result_cap: 결과 행 수 상한 (무작위 샘플, 무편향). None = 무제한.
            워커 단계에서 자르면 '전체 결과 list[dict] 물질화'가 사라져 피크 RAM이 준다.
        """
        super().__init__()
        self.parquet_dir = parquet_dir
        self.selected_ratings = set(selected_ratings)
        self.queries = queries
        self.exclude_queries = exclude_queries or {}
        self.combine_mode = (combine_mode or 'and').lower()
        if self.combine_mode not in ('and', 'or'):
            self.combine_mode = 'and'
        self.dataset_year = self.DEFAULT_DATASET_LABEL
        # 검색 결과가 실제로 검증된 manifest의 불변 provenance. Bridge/저장소는
        # 검색 완료 뒤 파일을 다시 읽지 않고 이 값을 그대로 사용해야 한다.
        self.dataset_identity = None
        self._pending_dataset_identity = None
        self._manifest_validation_token = None
        self._resolved_dataset_paths = {}
        self._verified_dataset_file_identities = {}
        self.result_cap = int(result_cap) if result_cap else None
        self.is_running = True

    def run(self):
        """검색 실행 — 클래스 락으로 직렬화(공유 캐시 경합 방지). 대체된 워커는
        락을 기다리는 동안에도 is_running=False면 즉시 빠져나간다."""
        try:
            if not self.is_running:   # 이미 대체됨 — 락 잡기 전 빠른 탈출
                return
            with PandasSearchWorker._run_lock:
                self._run_locked()
        except Exception as e:
            self.status_update.emit(f"❌ 오류 발생: {str(e)}")

    def _run_locked(self):
        import pandas as pd

        try:
            if not self._load_data():
                return
            if not self.is_running:   # 새 검색으로 대체됨 — stale 결과 emit 금지
                return

            if self.cached_df is None or self.cached_df.empty:
                self.results_ready.emit(NormalizedSearchRows(), 0)
                return

            self.status_update.emit(
                f"🔍 데이터 검색 중 (모드: {self.combine_mode.upper()})..."
            )
            print(f"[Search] === START === year={self.dataset_year} ratings={sorted(self.selected_ratings)} mode={self.combine_mode}")

            df = self.cached_df
            print(f"[Search] cached_df shape: {df.shape}")

            # ── 포함 검색 — 필드 간 결합은 combine_mode에 따라 ──
            non_empty_fields = [
                (col, txt) for col, txt in self.queries.items()
                if txt and col in df.columns
            ]
            print(f"[Search] non_empty_fields: {[(c, t[:50]) for c, t in non_empty_fields]}")

            n_total = len(df)

            def _wc_note(n: int) -> str:
                """매칭 수가 전체와 같으면 wildcard(필드 면제) 발동 표시"""
                return ' [WILDCARD - 필드 면제]' if n == n_total else ''

            if not non_empty_fields:
                total_mask = pd.Series(True, index=df.index)
                print(f"[Search] no fields -> all rows pass ({n_total:,})")
            elif self.combine_mode == 'or':
                # OR: 빈 마스크에서 시작해 |= 누적
                total_mask = pd.Series(False, index=df.index)
                for col, search_text in non_empty_fields:
                    if not self.is_running:
                        return
                    cm = self._parse_condition(df, col, search_text)
                    n_match = int(cm.sum())
                    print(f"[Search] OR  | {col:>10s} '{search_text[:60]}...' -> {n_match:,} matches{_wc_note(n_match)}")
                    total_mask |= cm
            else:
                # AND: True에서 시작해 &= 누적
                total_mask = pd.Series(True, index=df.index)
                for col, search_text in non_empty_fields:
                    if not self.is_running:
                        return
                    cm = self._parse_condition(df, col, search_text)
                    n_match = int(cm.sum())
                    print(f"[Search] AND | {col:>10s} '{search_text[:60]}...' -> {n_match:,} matches{_wc_note(n_match)}")
                    total_mask &= cm
                    print(f"[Search]       cumulative AND mask: {int(total_mask.sum()):,}")

            # ── 제외 검색 — 모드와 무관하게 항상 AND-NOT ──
            for col, search_text in self.exclude_queries.items():
                if not search_text:
                    continue
                if col not in df.columns:
                    continue
                if not self.is_running:
                    return
                exclude_mask = self._parse_condition(df, col, search_text)
                n_excl = int(exclude_mask.sum())
                print(f"[Search] EXC | {col:>10s} '{search_text[:60]}...' -> {n_excl:,} excluded")
                total_mask &= ~exclude_mask

            # 결과 필터링
            filtered_df = df[total_mask]
            total_count = len(filtered_df)
            print(f"[Search] === DONE === final mask: {int(total_mask.sum()):,} -> {total_count:,} rows")

            # cap을 여기(워커)에서 적용 — 전체 결과의 dict 물질화 자체를 회피.
            # sample()은 무작위라 기존 '셔플 후 슬라이스'와 동등(무편향).
            if self.result_cap and total_count > self.result_cap:
                print(f"[Search] capping {total_count:,} -> {self.result_cap:,} (워커 단계, RAM 절약)")
                filtered_df = filtered_df.sample(n=self.result_cap)

            # 출력 컬럼만 — 검색용으로만 쓰는 meta 복제 제거
            out_cols = [c for c in self.OUTPUT_COLUMNS if c in filtered_df.columns]
            if out_cols:
                filtered_df = filtered_df[out_cols]

            # 행 정규화(결측→''/None, 해상도 int)는 GUI 스레드가 아니라 여기서 끝낸다.
            # 결측 처리는 정규화가 하므로 dtype 을 바꾸는 fillna 사본도 만들지 않는다.
            records = filtered_df.to_dict('records')
            if not self.is_running:
                return
            results = NormalizedSearchRows(normalize_search_rows_in_place(records))
            del records

            if not self.is_running:   # emit 직전 최종 체크
                return
            self.results_ready.emit(results, total_count)
            self.status_update.emit(
                f"✅ {self.combine_mode.upper()} 검색 완료: {total_count:,}건"
            )

        except Exception as e:
            self.status_update.emit(f"❌ 오류 발생: {str(e)}")

    def _parse_condition(self, df, col, query_text):
        """통합 태그 매칭 엔진 — 와일드카드 + 그룹 + OR/AND 지원
        combine_mode를 tag_matcher에 전달:
        - 'and': 콤마=AND (기존)
        - 'or':  콤마=OR (필드 내 콤마도 OR로 결합)
        명시적 [A|B], [A,B] 그룹은 모드와 무관하게 항상 OR/AND.

        성능: 태그 컬럼은 이미 소문자(빌드 보장)이고 로드 때 fillna('')까지 끝나
        있으므로 원본 Series 를 그대로 매칭에 넘긴다 — 소문자 사본(컬럼당 수백 MB)과
        컬럼별 첫 질의의 1.8~4초 변환이 없다. 질의 쪽은 tag_matcher 가 소문자로 만든다.
        """
        from core.tag_matcher import filter_dataframe
        return filter_dataframe(df, col, query_text,
                                default_combine=self.combine_mode,
                                col_lower=df[col])

    def _active_dataset_manifest(self):
        """Read and minimally validate the authoritative runtime manifest."""
        manifest_path = os.path.join(self.parquet_dir, 'dataset_manifest.json')
        if not os.path.isfile(manifest_path):
            self.status_update.emit(
                "❌ 검색 데이터 manifest 없음: dataset_manifest.json"
            )
            return None
        try:
            before = self._file_identity(manifest_path)
            with open(manifest_path, 'rb') as handle:
                raw_manifest = handle.read()
            after = self._file_identity(manifest_path)
            if after != before:
                raise ValueError('dataset manifest가 읽는 중 변경되었습니다')
            manifest = json.loads(raw_manifest.decode('utf-8-sig'))
            if not isinstance(manifest, dict):
                raise ValueError('dataset manifest 루트는 객체여야 합니다')
            if manifest.get('format_version') != self.MANIFEST_FORMAT_VERSION:
                raise ValueError(
                    '지원하지 않는 dataset manifest 형식 버전: '
                    f"{manifest.get('format_version')!r}"
                )
            label = manifest.get('dataset_label')
            if not isinstance(label, str):
                raise ValueError('dataset_label은 문자열이어야 합니다')
            label = label.strip()
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label):
                raise ValueError('dataset_label 형식이 올바르지 않습니다')
            artifacts = manifest.get('artifacts')
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError('manifest artifacts는 비어 있지 않은 배열이어야 합니다')
            if any(not isinstance(item, dict) for item in artifacts):
                raise ValueError('manifest artifact는 객체여야 합니다')
            digest = hashlib.sha256(raw_manifest).hexdigest()
            return manifest, label, digest, manifest_path, after
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            self.status_update.emit(f"❌ 검색 데이터 manifest 오류: {exc}")
            return None

    @staticmethod
    def _file_identity(path):
        stat = os.stat(path)
        return (
            os.path.abspath(path),
            stat.st_size,
            stat.st_mtime_ns,
            getattr(stat, 'st_ino', 0),
        )

    def _sha256(self, path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _assert_manifest_current(self, path, identity, digest):
        """Reject a manifest replaced while shard verification/loading ran."""
        try:
            before = self._file_identity(path)
            with open(path, 'rb') as handle:
                raw_manifest = handle.read()
            after = self._file_identity(path)
        except OSError as exc:
            raise ValueError(f'dataset manifest 재확인 실패: {exc}') from exc
        if before != identity or after != identity:
            raise ValueError('dataset manifest가 검색 데이터 검증 중 변경되었습니다')
        if hashlib.sha256(raw_manifest).hexdigest() != digest:
            raise ValueError('dataset manifest 내용이 검색 데이터 검증 중 변경되었습니다')

    @staticmethod
    def _schema_fields(schema_arrow):
        return [
            {
                'name': field.name,
                'type': str(field.type),
                'nullable': bool(field.nullable),
            }
            for field in schema_arrow
        ]

    def _validate_manifest_artifact(
        self, artifact, *, label, rating, manifest_digest
    ):
        """Validate one selected Search shard and return its resolved path."""
        expected_name = f"danbooru_{label}_{rating}.parquet"
        relative_path = artifact.get('path')
        if not isinstance(relative_path, str) or relative_path != expected_name:
            raise ValueError(
                f"'{rating}' artifact 경로 불일치: "
                f"{relative_path!r} (예상: {expected_name!r})"
            )
        if artifact.get('format') != 'parquet':
            raise ValueError(f"'{rating}' artifact 형식은 parquet이어야 합니다")

        expected_size = artifact.get('size_bytes')
        expected_rows = artifact.get('rows')
        expected_sha = artifact.get('sha256')
        if (isinstance(expected_size, bool) or
                not isinstance(expected_size, int) or expected_size < 0):
            raise ValueError(f"'{rating}' artifact size_bytes가 올바르지 않습니다")
        if (isinstance(expected_rows, bool) or
                not isinstance(expected_rows, int) or expected_rows < 0):
            raise ValueError(f"'{rating}' artifact rows가 올바르지 않습니다")
        if (not isinstance(expected_sha, str) or
                not re.fullmatch(r'[0-9a-f]{64}', expected_sha)):
            raise ValueError(f"'{rating}' artifact sha256이 올바르지 않습니다")
        expected_schema = artifact.get('schema')
        if not isinstance(expected_schema, list):
            raise ValueError(f"'{rating}' artifact schema가 올바르지 않습니다")

        root = os.path.realpath(self.parquet_dir)
        path = os.path.realpath(os.path.join(root, relative_path))
        try:
            if os.path.commonpath((root, path)) != root:
                raise ValueError(f"'{rating}' artifact가 데이터 폴더를 벗어납니다")
        except ValueError as exc:
            raise ValueError(f"'{rating}' artifact 경로가 올바르지 않습니다") from exc
        if not os.path.isfile(path):
            raise ValueError(f"'{rating}' 활성 릴리스 파일 없음: {relative_path}")

        before = self._file_identity(path)
        if before[1] != expected_size:
            raise ValueError(
                f"'{rating}' artifact 크기 불일치: "
                f"manifest={expected_size}, actual={before[1]}"
            )
        verification_key = (
            manifest_digest, before, expected_size, expected_rows, expected_sha
        )
        if verification_key in self.verified_artifact_signatures:
            return path, before

        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        actual_rows = parquet.metadata.num_rows
        if actual_rows != expected_rows:
            raise ValueError(
                f"'{rating}' artifact 행 수 불일치: "
                f"manifest={expected_rows}, actual={actual_rows}"
            )
        actual_schema = self._schema_fields(parquet.schema_arrow)
        if actual_schema != expected_schema:
            raise ValueError(f"'{rating}' artifact schema 불일치")
        missing = set(self.REQUIRED_COLUMNS) - set(parquet.schema.names)
        if missing:
            raise ValueError(
                f"{label} '{rating}' 필수 컬럼 누락: "
                + ", ".join(sorted(missing))
            )
        actual_sha = self._sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"'{rating}' artifact SHA-256 불일치: "
                f"manifest={expected_sha}, actual={actual_sha}"
            )
        after = self._file_identity(path)
        if after != before:
            raise ValueError(f"'{rating}' artifact가 검증 중 변경되었습니다")
        self.verified_artifact_signatures.add(verification_key)
        return path, before

    def _resolve_dataset_year(self):
        """Validate the one active release; never fall back to old snapshots."""
        self.dataset_identity = None
        self._pending_dataset_identity = None
        self._manifest_validation_token = None
        self._resolved_dataset_paths = {}
        if not self.selected_ratings:
            return None
        active = self._active_dataset_manifest()
        if not active:
            return None
        manifest, year, manifest_digest, manifest_path, manifest_file_identity = active
        artifacts = manifest['artifacts']
        resolved_paths = {}
        verified_identities = {}
        try:
            for rating in sorted(self.selected_ratings):
                matches = [
                    item for item in artifacts
                    if item.get('kind') == 'search'
                    and item.get('rating_shard') == rating
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"'{rating}' Search artifact는 정확히 1개여야 합니다 "
                        f"(현재 {len(matches)}개)"
                    )
                (
                    resolved_paths[rating],
                    verified_identities[rating],
                ) = self._validate_manifest_artifact(
                    matches[0],
                    label=year,
                    rating=rating,
                    manifest_digest=manifest_digest,
                )
            self._assert_manifest_current(
                manifest_path, manifest_file_identity, manifest_digest
            )
        except Exception as exc:
            self.status_update.emit(f"❌ 검색 데이터 manifest 검증 실패: {exc}")
            return None
        self._resolved_dataset_paths = resolved_paths
        self._verified_dataset_file_identities = verified_identities
        self._pending_dataset_identity = {
            'label': year,
            'fingerprint': manifest_digest,
        }
        self._manifest_validation_token = (
            manifest_path, manifest_file_identity, manifest_digest
        )
        return year

    def _dataset_signature(self, year):
        """Confirm every shard still has the identity that was hash-verified."""
        signature = []
        try:
            for rating in sorted(self.selected_ratings):
                path = self._resolved_dataset_paths[rating]
                current = self._file_identity(path)
                expected = self._verified_dataset_file_identities[rating]
                if current != expected:
                    raise ValueError(
                        f"'{rating}' artifact가 검증 후 변경되었습니다"
                    )
                signature.append(current)
        except (KeyError, OSError, ValueError) as exc:
            self.status_update.emit(f"❌ 검색 데이터 상태 확인 실패: {exc}")
            return None
        return tuple(signature)

    def _load_data(self):
        """선택된 등급의 Parquet 파일 로드 (년도 + rating set으로 캐시 키)"""
        import pandas as pd

        resolved_year = self._resolve_dataset_year()
        if resolved_year is None:
            self.status_update.emit("❌ 사용 가능한 검색 데이터가 없습니다.")
            return False
        self.dataset_year = resolved_year
        file_signature = self._dataset_signature(resolved_year)
        if file_signature is None:
            return False

        if (PandasSearchWorker.cached_df is not None and
            PandasSearchWorker.loaded_ratings == self.selected_ratings and
            PandasSearchWorker.loaded_year == self.dataset_year and
            PandasSearchWorker.loaded_file_signature == file_signature):
            self.dataset_identity = dict(self._pending_dataset_identity)
            return True

        # rating set 또는 년도가 바뀌면 캐시 무효화
        PandasSearchWorker.cached_df = None
        dfs = []

        load_failed = False
        for rating in sorted(self.selected_ratings):
            if not self.is_running:   # 새 검색으로 대체됨 — 비싼 parquet 로드 중단
                return False
            # manifest의 단일 활성 릴리스만 로드한다.
            path = self._resolved_dataset_paths[rating]

            if os.path.exists(path):
                self.status_update.emit(f"📂 '{rating}' 등급 데이터 로딩 중...")
                try:
                    df = pd.read_parquet(path, columns=self.LOAD_COLUMNS)
                    if self._file_identity(path) != (
                        self._verified_dataset_file_identities[rating]
                    ):
                        raise ValueError(
                            f"'{rating}' artifact가 로딩 중 변경되었습니다"
                        )
                    dfs.append(df)
                except Exception as e:
                    self.status_update.emit(f"⚠️ 파일 로드 실패 ({rating}): {e}")
                    load_failed = True
                    break
            else:
                self.status_update.emit(f"⚠️ 파일 없음: {path}")
                load_failed = True
                break

        if load_failed or len(dfs) != len(self.selected_ratings):
            self.status_update.emit("❌ 선택한 릴리스 전체를 로드하지 못했습니다.")
            return False

        # 검증 뒤 실제 parquet 로드가 끝날 때까지 manifest나 shard가 바뀌었다면
        # 서로 다른 snapshot을 섞은 결과가 될 수 있으므로 cache publish 전 폐기한다.
        try:
            current_signature = self._dataset_signature(resolved_year)
            if current_signature != file_signature:
                raise ValueError('검색 데이터 파일이 로딩 중 변경되었습니다')
            manifest_path, manifest_identity, manifest_digest = (
                self._manifest_validation_token
            )
            self._assert_manifest_current(
                manifest_path, manifest_identity, manifest_digest
            )
        except (OSError, TypeError, ValueError) as exc:
            self.status_update.emit(f"❌ 검색 데이터 검증 실패: {exc}")
            return False

        self.status_update.emit(f"📊 {self.dataset_year} 데이터 병합 중...")
        PandasSearchWorker.cached_df = pd.concat(dfs, ignore_index=True)
        PandasSearchWorker.loaded_ratings = self.selected_ratings
        PandasSearchWorker.loaded_year = self.dataset_year
        PandasSearchWorker.loaded_file_signature = file_signature
        
        # 문자열 컬럼 결측치 처리
        text_cols = ['copyright', 'character', 'artist', 'general', 'meta'] 
        for col in text_cols:
            if col in PandasSearchWorker.cached_df.columns:
                PandasSearchWorker.cached_df[col] = (
                    PandasSearchWorker.cached_df[col].fillna("")
                )

        self.dataset_identity = dict(self._pending_dataset_identity)
        return True

    def stop(self):
        self.is_running = False
