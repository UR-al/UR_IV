# Danbooru runtime datasets

이 폴더의 Parquet은 Search와 Event Gen이 실제로 검색하는 로컬 런타임 데이터입니다.
대용량 파일은 Git에 넣지 않으며, 현재 설치에는 2026-07-13까지의 게시물을 담은
`2026_07` 릴리스가 적용되어 있습니다. 정확한 행 수, 스키마, 크기와 SHA-256은
`dataset_manifest.json`에서 확인합니다.

## 파일 역할

- `danbooru_2026_07_{g,s,q,e}.parquet`: Search용 전체 게시물 태그
- `danbooru_sorted/danbooru_{g,s,q,e}.parquet`: Event Gen용 parent-child 그래프
- `dataset_manifest.json`: 활성 릴리스, 원본 revision과 모든 Search/Event 산출물의 검증 정보
- `legacy_search_tags_before_2026_07.csv`: 구형 Search 3세대에만 남은 태그 archive

Search는 manifest가 지정한 `2026_07` 단일 활성 릴리스의 G/S/Q/E 4개 shard만
불러옵니다. 과거 릴리스를 선택하거나 구형 파일로 fallback하지 않습니다.

`legacy_search_tags_before_2026_07.csv`는 삭제 전 `2025`, `2026`, `2026_06`
릴리스의 합집합에서 최신 `2026_07`에 없는 고유 태그만 추린 파일입니다.
열은 `tag`, `legacy_categories`, `legacy_releases`이며 Search 게시물 shard를
대체하지 않는 보존용 목록입니다. 현재 보관본은 27,108행, 968,296바이트이며
SHA-256은 `3274463b91c21559eaae915319c780ca54e89adc5abace9e08d7a2ef943e44ab`입니다.

Event shard에는 해당 등급 child와 그 child가 참조하는 모든 parent가 들어갑니다.
parent의 자체 등급이 달라도 포함되므로 단일 등급만 선택해도 그래프가 끊기지 않습니다.
여러 등급을 함께 불러올 때 생기는 parent 중복은 `EventDataLoader`가 ID 기준으로 제거합니다.

## 앱 시작 시 자동 받기

앱은 시작할 때 `core/fetch_data.py`가 `dataset_manifest.json`의 각 artifact `path`와
`size_bytes`를 디스크와 비교합니다. 파일이 없거나 크기가 다르면(저장소를 pull해
manifest가 새 릴리스를 가리키는데 parquet은 옛것인 경우 포함) **그 경로만**
Hugging Face 데이터셋 `UR-AR/UR_IV`에서 다시 받습니다. `tags_dictionary.parquet`이
없을 때도 함께 받습니다. 기동 시간 때문에 SHA-256은 계산하지 않으며, Search가 활성
릴리스를 불러올 때 SHA-256·행 수·스키마를 검증합니다.

받기에 실패해도 이미 쓰던 데이터가 있으면 앱은 뜨고 경고만 남깁니다. 이때 Search는
manifest 검증 오류를, Event Gen은 manifest와 크기가 다른 shard 경고를 표시합니다.
이 폴더에 parquet이 하나도 없는 첫 설치에서 받기에 실패하면 예전처럼 시작을 중단합니다.
현재 manifest의 경로가 하나도 없더라도 `tags_dictionary.parquet`이나 옛 릴리스 이름의
parquet이 있으면 쓰던 데이터가 있는 것으로 보고 경고만 남긴 채 앱을 띄웁니다.
수동으로 다시 받으려면 프로젝트 루트에서 `python -m core.fetch_data`를 실행합니다.

manifest 최상위에 선택 필드 `"hf_revision": "<커밋 해시>"`를 두면 그 Hugging Face
커밋에서 받습니다. parquet을 업로드한 뒤 그 커밋을 적어 두면, 저장소의 manifest와
받는 파일이 항상 같은 릴리스로 고정됩니다.

## 다시 생성하기

고정된 원본 revision을 받은 뒤 다음처럼 실행합니다.

```powershell
hf download nyanko-devs/danbooru2026 metadata/posts-snapshot.parquet `
  --repo-type dataset `
  --revision ebb02a630201c7b51487e45fb90b3fcf4cbedc20 `
  --local-dir .cache/danbooru2026

.\venv\Scripts\python.exe tools\refresh_danbooru_data.py posts `
  --source .cache/danbooru2026/metadata/posts-snapshot.parquet `
  --output-dir danbooru_optimized `
  --dataset-label 2026_07 `
  --source-url https://huggingface.co/datasets/nyanko-devs/danbooru2026 `
  --source-revision ebb02a630201c7b51487e45fb90b3fcf4cbedc20 `
  --snapshot-at 2026-07-13T16:25:53.082Z `
  --expected-source-sha256 5b6b2671dc0fa966de71af76dfd342f485f76581447cec9e26c313ba9fb1c2fd
```

구형 shard를 삭제하기 전에 legacy 태그 archive를 다시 만들려면:

```powershell
.\venv\Scripts\python.exe tools\refresh_danbooru_data.py archive-legacy-tags `
  --input-dir danbooru_optimized `
  --latest-label 2026_07 `
  --legacy-labels 2025 2026 2026_06 `
  --output danbooru_optimized/legacy_search_tags_before_2026_07.csv
```

생성 결과만 다시 검사하려면:

```powershell
.\venv\Scripts\python.exe tools\refresh_danbooru_data.py validate `
  --manifest danbooru_optimized/dataset_manifest.json `
  --root danbooru_optimized
```
