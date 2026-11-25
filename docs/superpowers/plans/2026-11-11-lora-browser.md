# 스튜디오 LoRA 선택 창 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스튜디오 패널의 "+ LoRA"·LoRA 줄·XYZ LoRA 줄에서 여는 검색·미리보기 LoRA 선택 창을 팩 1.19.0 으로 만든다.

**Architecture:** 경로 해석(짧은 이름·런·에폭)은 JS 와 Python 이 같은 golden 표로 묶인 두 구현을 가진다. Python 은 safetensors
머리말 요약·샘플/출력 그림 찾기·썸네일·즐겨찾기를 맡아 `/aistudio/studio/loras/*` 라우트로 내주고, 캐시는 ComfyUI user 폴더에만 쓴다.
JS 는 순수 상태(`lora_browser_state.js`)와 DOM(`lora_browser.js`)을 나누고, 패널·XYZ·갤러리는 주입된 `ctx.loraBrowser()` 하나만 부른다.

**Tech Stack:** ComfyUI 커스텀 노드 팩(Python 3, aiohttp 라우트, PIL), 바닐라 ES 모듈(`studio_dom.js` 의 `h()` 헬퍼), node:test, unittest.

**Spec:** `docs/superpowers/specs/2026-11-11-lora-browser-design.md` (10절 사용자 결정 포함)

## Global Constraints

- **GPL 코드 금지(clean-room):** ComfyUI-Lora-Manager 의 소스·CSS·아이콘을 열거나 옮기지 않는다. 참고는 스펙과 README 요약뿐.
- **저장값 불변:** 패널 `state.loras[] = {on, name, strength}` 의 `name` 은 지금처럼 `fileStem(entry)`; XYZ `rows[].lora` 도 같은 형식.
  기존 워크플로·`<lora:이름:세기>`·EXIF 재현이 그대로 동작해야 한다.
- **쓰기 위치:** 캐시·설정은 `folder_paths.get_user_directory()/default/ai_studio/` 아래에만. LoRA 폴더(D:\LoRA 등)·output 폴더에는 쓰지 않는다.
- **머리말 읽기:** `degrid_files.safetensors_header(path)`(상한 있음)만 쓴다. 텐서는 읽지 않는다.
- **기본 경로(설정으로 바꿀 수 있음):** `anima_root = C:\Users\URL\Desktop\UR\ANIMA`, `board_root = C:\Users\URL\Desktop\UR\board`.
- **버전:** `__init__.__version__` 와 `core/comfy_node_pack.PACK_VERSION` 을 `"1.19.0"` 으로 같이 올린다(버전 고정 테스트 3곳 갱신).
- **파일 크기:** 새 JS·Python 파일은 ~400줄 안. 큰 파일(`panel_view.js`, `studio_main.js`)에는 연결 몇 줄만.
- **검증 명령:** Python `venv\Scripts\python.exe run_tests.py`(저장소 루트, venv 필수), JS `cd frontend && npm run test:pack`.
- **커밋은 사용자가 요청할 때만**(저장소 CLAUDE.md). 각 태스크 끝의 "커밋" 단계는 커밋 메시지 초안만 적어 두고 실제 커밋은 `/ship` 때.
- **GPU 금지.** 화면 확인은 CPU 전용 임시 ComfyUI(포트 8197, `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`). 사용자 8188 은 건드리지 않는다.
- 화면 글자는 한국어, 짧게. 새 CSS 클래스 접두는 `ais-lb-`, 색은 `studio.css` 의 기존 회색 계열 값.

## Review Focus

1. **USB(D:) 가 빠졌거나 파일이 지워진 뒤** 캐시에 남은 항목: 색인에서 빠지고, detail/thumb 는 404 JSON — 예외로 창이 깨지지 않는다. (Task 2·6 테스트)
2. **이상한 머리말**(JSON 아님, 상한 초과, `__metadata__` 없음, `ss_epoch` 가 문자열 "None"): 요약 필드만 비고 색인은 만들어진다. (Task 2)
3. **한글·태국 문자·괄호·공백 이름**(`묵밥 (@Bjunbae)`, `໊ (@9an9ta)`, `_人間失联_ (@renjianshilian0)`): 짧은 이름·URL 인코딩·경로 비교가 맞다. (Task 1 golden, Task 6)
4. **thumb/detail 의 경로 탈출**(`name=..\..\x`, 절대 경로, 색인에 없는 이름): 색인에 있는 이름만 받는다. (Task 6)
5. **output 폴더에 그림 수천 장**: 두 번째부터는 새 파일만 읽고, 인포텍스트 없는 그림·깨진 PNG 는 건너뛴다. (Task 5)

---

### Task 1: 짧은 이름·경로 해석 (JS) + 공용 golden

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/lora_names.js`
- Create: `comfy_custom_nodes/ai_studio_forge_parity/js_tests/fixtures/lora_names_golden.json`
- Test: `comfy_custom_nodes/ai_studio_forge_parity/js_tests/lora_names.test.mjs`

**Interfaces:**
- Produces:
  - `parseLoraPath(entry: string) -> {algo, group, groupLabel, run, runShort, recipe: string[], epoch: number|null, final: boolean, downloaded: boolean, runKey: string}`
  - `shortLoraName(entry: string, meta?: {num_epochs?: number}) -> string` (예: `"묵밥 · @bjunbae fp16 · e14"`)
  - `middleEllipsis(text: string, max: number) -> string` (에폭 꼬리 ` · eNN`/` · 최종` 은 남김)
  - `matchesQuery(item: {entry, trigger?}, query: string) -> boolean`
  - `familyOfRecipe(recipe: string[]) -> "2.9b"|"3.8b"|null`
  - `familyVisible(itemFamily: string|null, current: string, showAll: boolean) -> boolean`
  - `appendTrigger(text: string, trigger: string) -> string`
- 규칙(스펙 5절): 구분자는 `\` 와 `/` 둘 다. `algo/group/run/file` 이 기본, 파일이 그룹 바로 아래면 run = stem 에서 `-\d{6}$` 뗀 것.
  stem 끝 `-0000NN` → `epoch = NN`; 런 폴더 안 `<run>.safetensors` → `final = true`. `_downloaded` 세그먼트가 있으면 `downloaded = true`,
  group = 그 다음 폴더. recipe 토큰(`_` 로 나눈 런 이름에서 제거): `^\d+(\.\d+)?b$`, `{lokr, loha, locon, lora, dora, tsm, full}`,
  `{came, adamw, adamw8bit, prodigy, adafactor}`, `^b\d+$`, `^\d+e\d+$`. 남는 토큰은 공백으로 이어 `runShort`.
  `groupLabel` = group 에서 끝의 `(@...)` 괄호를 뗀 것(비면 group). runShort 가 groupLabel(대소문자 무시)로 시작하면 표시에서 그룹 생략.
  `runKey` = 런 폴더 레이아웃이면 `algo/group/run`, 옛 레이아웃이면 `algo/group/run`(run 은 stem 기준) — 에폭 파일들이 같은 키를 갖는다.

- [ ] **Step 1: golden 작성** — D:\LoRA 실제 경로 30개 이상(모든 레이아웃·특수 문자 그룹 포함)을 `{entry, expect: {group, run, runShort, epoch, final, downloaded, label}}` 로. 반드시 들어갈 행:

| entry | label |
|---|---|
| `LoKr\묵밥 (@Bjunbae)\@bjunbae_2.9b_lokr_came_b10_2e06_fp16\@bjunbae_2.9b_lokr_came_b10_2e06_fp16-000014.safetensors` | `묵밥 · @bjunbae fp16 · e14` |
| `LoKr\GT_DunE\@gt_dune_2.9b_lokr_came_b10_2e06_fp16\@gt_dune_2.9b_lokr_came_b10_2e06_fp16.safetensors` | `GT_DunE · @gt_dune fp16 · 최종` |
| `LoKr\묵밥 (@Bjunbae)\@bjunbae_3.8b_anima-000001.safetensors` | `묵밥 · @bjunbae anima · e1` |
| `LoKr\@enaa97\@enaa97-000001.safetensors` | `@enaa97 · e1` |
| `LoKr\@enaa97\@enaa97_2.9b_lokr_came_b10_2e06_wsd_fp16\@enaa97_2.9b_lokr_came_b10_2e06_wsd_fp16-000010.safetensors` | `@enaa97 wsd fp16 · e10` |
| `LoRA\_downloaded\Komano\Komano_Civ.safetensors` | `Komano Civ` |
| `LoRA\HUSI\HUSI.safetensors` | `HUSI` |
| `LoKr\໊ (@9an9ta)\@na-eun_2.9b_lokr_came_b10_2e06_fp16\@na-eun_2.9b_lokr_came_b10_2e06_fp16-000020.safetensors` | `໊ · @na-eun fp16 · e20` |

- [ ] **Step 2: 실패하는 테스트** — `lora_names.test.mjs`:
  - `golden: every entry parses to its expected fields and label` (golden 전 행, `deepStrictEqual`)
  - `final label uses num_epochs when given`: `shortLoraName(<GT_DunE 최종>, {num_epochs: 15}) === "GT_DunE · @gt_dune fp16 · 최종 e15"`
  - `middleEllipsis keeps the epoch tail`: 60자 label 을 30으로 줄이면 `endsWith(" · e14")`, 길이 ≤ 30, `…` 포함
  - `matchesQuery`: `"bjun 14"` → 묵밥 e14 true, 묵밥 e15 false(에폭 토큰은 `e14`·`14` 둘 다, 숫자 토큰은 에폭과 정확히 일치), `"BJUN"` true, `"@gt dune"` 이 trigger `@gt dune` 인 항목 true, `"_"`·공백 무시
  - `familyOfRecipe(["2.9b","lokr"]) === "2.9b"`, `familyOfRecipe(["anima"]) === null`
  - `familyVisible(null, "2.9b", false) === true`, `familyVisible("base", "2.9b", false) === false`, `familyVisible("base", "2.9b", true) === true`
  - `appendTrigger("", "@gt dune") === "@gt dune"`, `appendTrigger("a, b", "@gt dune") === "a, b, @gt dune"`, `appendTrigger("a, @gt dune", "@gt dune") === "a, @gt dune"`
- [ ] **Step 3:** `cd frontend && npm run test:pack` → 새 테스트 FAIL(모듈 없음)
- [ ] **Step 4:** `lora_names.js` 구현(위 시그니처·규칙). 다른 팩 파일 import 없음(순수).
- [ ] **Step 5:** `npm run test:pack` → 전부 PASS
- [ ] **Step 6: 커밋 초안** `feat: LoRA 짧은 이름·경로 해석(lora_names.js)과 golden 표`

### Task 2: LoRA 색인 (Python)

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/lora_index.py`
- Test: `tests/test_lora_index.py`, `tests/test_lora_names_golden.py`

**Interfaces:**
- Consumes: golden 표(Task 1), `degrid_files.safetensors_header`, `studio_choices.unet_family`.
- Produces:
  - `parse_lora_path(entry: str) -> dict` — Task 1 `parseLoraPath` 와 같은 키(snake_case: `group_label`, `run_short`, `run_key`)
  - `header_summary(header: dict | None) -> dict` — `{has_meta, epoch, num_epochs, steps, max_steps, base_model, algo, network_args, precision, optimizer, lr, scheduler, date, dataset_dirs, top_tags, sha}` (없는 것은 None; `top_tags` = `ss_tag_frequency` 합산 상위 10개 `[tag, count]`)
  - `lora_family(summary: dict, parsed: dict) -> str | None` — `base_model` 이 있으면 `unet_family(base_model)`, 없으면 recipe 크기 토큰, 그래도 없으면 None
  - `trigger_of(summary: dict, board_run: dict | None) -> str` — ① `board_run["trigger"]` ② `top_tags` 중 `@` 로 시작하는 첫 태그 ③ `""`
  - `class LoraIndex(cache_path: Path)`: `build(names: list[str], full_path: Callable[[str], str | None], board: dict) -> list[dict]` — 항목 키: `name, stem, size, mtime, run_key, epoch, final, num_epochs, family, algo, precision, date, trigger, has_meta, downloaded, group, group_label, run, run_short, recipe`. 캐시 키 `(name, size, mtime_ns)`; 바뀐 파일만 머리말을 다시 읽고, 목록에 없는 캐시 항목은 버린다. 캐시는 원자적으로(`.tmp` → replace) 저장.
  - `load_board(board_root: Path) -> dict` — `board_db.json` 의 `{runs, marks}`, 파일이 없거나 깨졌으면 `{"runs": {}, "marks": {}}`

- [ ] **Step 1: 실패하는 테스트** — `test_lora_names_golden.py`: golden 전 행에서 `parse_lora_path` 의 `group, run, run_short, epoch, final,
  downloaded` 가 golden `expect` 의 `group, run, runShort, epoch, final, downloaded` 와 같다(`label` 은 JS 만 만든다). `test_lora_index.py` (임시 폴더에 가짜 safetensors: 8바이트 길이 + JSON 머리말 + 0바이트 본문):
  - `test_summary_reads_epoch_and_base_model`: `ss_epoch="14"`, `ss_num_epochs="15"`, `ss_sd_model_name="Anima-2.9B-preview-v1.safetensors"` → `epoch == 14`, `num_epochs == 15`, `lora_family(...) == "2.9b"`
  - `test_bad_headers_do_not_break_the_index` (Review Focus 2): JSON 아닌 머리말, 길이 0, `__metadata__` 없음, `ss_epoch="None"` 각각 → `has_meta`/`epoch` 가 False/None, 예외 없음
  - `test_trigger_prefers_board_then_at_tag`: board run trigger 있음 → 그 값; 없음 + tag_frequency `{"x": {"@gt dune": 70, "1boy": 90}}` → `"@gt dune"`
  - `test_cache_rereads_only_changed_files`: 두 번 build 하며 `safetensors_header` 호출 횟수를 mock 으로 셈 → 두 번째 0; 한 파일 mtime 변경 → 1
  - `test_missing_files_drop_out` (Review Focus 1): 캐시에 있던 이름이 `names` 에 없거나 `full_path` 가 None/없는 경로 → 결과에서 빠짐, 예외 없음
  - `test_load_board_tolerates_missing_and_broken_files`
- [ ] **Step 2:** `venv\Scripts\python.exe run_tests.py` → 새 테스트 FAIL
- [ ] **Step 3:** `lora_index.py` 구현(Qt·ComfyUI import 없음; `folder_paths` 는 호출자가 넘긴 함수로만).
- [ ] **Step 4:** `run_tests.py` → PASS
- [ ] **Step 5: 커밋 초안** `feat: LoRA 색인(머리말 요약·계열·트리거·mtime 캐시)`

### Task 3: 사용자 폴더 저장소 (설정·즐겨찾기·직접 지정 미리보기)

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/lora_store.py`
- Test: `tests/test_lora_store.py`

**Interfaces:**
- Produces:
  - `store_dir(user_dir: Path) -> Path` = `user_dir/"default"/"ai_studio"` (만들어 둠)
  - `load_settings(dir: Path) -> dict` — 기본 `{"anima_root": "C:\\Users\\URL\\Desktop\\UR\\ANIMA", "board_root": "C:\\Users\\URL\\Desktop\\UR\\board", "thumb_size": 384}`, `lora_browser.json` 값이 덮음(알 수 없는 키 무시)
  - `load_prefs(dir) -> dict` / `save_prefs(dir, prefs: dict) -> dict` — 키 `favorites: list[str run_key]`, `recent: list[str stem]`(최대 20, 최신 앞), `sort: "name"|"date"|"group"|"recent"`, `show_all: bool`, `tree_open: bool`, `close_after_add: bool`(기본 false). 저장 전 정규화(형식 틀린 값은 기본값).
  - `set_custom_preview(dir, run_key: str, image: Path, size: int) -> str` — 썸네일 webp 를 `previews/<sha1(run_key)>.webp` 로 만들고 `previews.json` 에 기록, 파일명 반환
  - `clear_custom_preview(dir, run_key) -> None`, `custom_preview(dir, run_key) -> Path | None`

- [ ] **Step 1: 실패하는 테스트**: `test_settings_defaults_and_override`, `test_prefs_normalise_bad_values`(recent 25개 → 20개, sort "x" → "name"), `test_custom_preview_round_trip`(8×8 PNG → webp 생김, 긴 변 ≤ size, clear 후 None), `test_store_never_writes_outside_dir`(쓰인 파일 경로가 모두 `store_dir` 아래)
- [ ] **Step 2:** FAIL 확인 → **Step 3:** 구현(PIL, 원자적 저장) → **Step 4:** PASS
- [ ] **Step 5: 커밋 초안** `feat: LoRA 선택 창 설정·즐겨찾기·직접 지정 미리보기 저장소`

### Task 4: 학습 샘플 찾기·썸네일

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/lora_samples.py`
- Test: `tests/test_lora_samples.py`

**Interfaces:**
- Consumes: Task 2 항목(`name`, `run`, `epoch`, `final`, `num_epochs`), Task 3 settings.
- Produces:
  - `run_dirs(full_path: Path, run: str, anima_root: Path, manifest: Path) -> list[Path]` — 순서(스펙 6절): 그 파일 폴더의 `move_log.jsonl` 첫 줄 `src` 의 부모 → `manifest` 에서 `run` 이 같은 마지막 줄의 `src` → `anima_root/run`; 존재하는 폴더만
  - `epoch_samples(run_dirs: list[Path], run: str, epoch: int) -> list[Path]` — `sample/<run>_e{epoch:06d}_{NN}_*.png` 를 번호순, 번호마다 가장 최근 1장, 최대 4장
  - `board_strip(board_root: Path, run: str, epoch: int) -> Path | None` — `site/s/<run 에서 앞 @ 뗀 것>/e{epoch:03d}.webp`
  - `file_epoch(item: dict) -> int | None` — `epoch`, final 이면 `num_epochs`
  - `make_thumb(src: Path, dst: Path, size: int, crop: tuple | None = None) -> Path` — 긴 변 `size` webp; 이미 있고 src 보다 새것이면 그대로
  - `strip_tiles(strip: Path) -> list[tuple]` — 보드 띠(4장 가로)를 4칸 crop 박스로 (세로 띠면 세로로)

- [ ] **Step 1: 실패하는 테스트**: `test_run_dir_from_move_log_then_manifest_then_root`(세 경우 각각), `test_epoch_samples_pick_latest_per_prompt`(같은 번호 두 장 → 최신 1장, e10 요청에 e1 파일 안 섞임), `test_board_strip_name_without_at`, `test_final_file_uses_num_epochs`, `test_thumb_is_cached`(두 번째 호출은 파일 mtime 불변)
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** PASS
- [ ] **Step 5: 커밋 초안** `feat: LoRA 에폭 학습 샘플 찾기와 썸네일`

### Task 5: 출력 그림에서 자동 미리보기

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/lora_previews.py`
- Test: `tests/test_lora_previews.py`

**Interfaces:**
- Produces:
  - `lora_names_in_infotext(text: str) -> list[str]` — `<lora:NAME:...>` 의 NAME(파일 stem, 폴더 앞부분 있으면 마지막 조각) 중복 없이
  - `class OutputScan(cache_path: Path)`: `update(output_dir: Path, limit_seconds: float = 20.0) -> dict` — PNG 의 `parameters` 텍스트를 읽어 `{stem: {"path": str, "mtime": float}}`(stem 별 가장 최근) 갱신; 캐시 `{path: [mtime, [stems]]}` 로 새·바뀐 파일만 읽음; 시간 상한 넘으면 다음 호출에 이어서. 반환 `{"latest": {...}, "pending": int}`
  - `latest_for(stems: list[str]) -> dict | None` — 한 런의 에폭 stem 들 중 가장 최근 그림

- [ ] **Step 1: 실패하는 테스트**: `test_names_from_infotext`(`"a <lora:x\\y-000010:0.8> b <lora:z:1>"` → `["y-000010", "z"]`), `test_scan_reads_only_new_files` (Review Focus 5: 두 번째 update 에서 PNG 열기 횟수 0, 새 파일 1장 추가 → 1), `test_scan_skips_broken_and_textless_png`, `test_latest_for_picks_newest_across_epochs`, `test_time_limit_leaves_pending`
- [ ] **Step 2:** FAIL → **Step 3:** 구현(PIL `Image.open(...).text`, `os.scandir` 재귀, 이미지 확장자 png 만) → **Step 4:** PASS
- [ ] **Step 5: 커밋 초안** `feat: 출력 그림 인포텍스트로 LoRA 자동 미리보기`

### Task 6: 라우트

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/lora_routes.py`
- Modify: `comfy_custom_nodes/ai_studio_forge_parity/__init__.py` (`_register_studio_routes()` 앞에서 `register_lora_handlers()` 호출)
- Test: `tests/test_lora_routes.py`

**Interfaces:**
- Consumes: Task 2~5, `studio_routes.add_handler`, `folder_paths`(`get_filename_list("loras")`, `get_full_path("loras", n)`, `get_user_directory()`, `get_output_directory()`).
- Produces (접두 `/aistudio/studio`): 순수 함수가 `(status, payload)` 를 돌려주고 얇은 aiohttp 래퍼가 감싼다.
  - `GET /loras/index` → `handle_index(query) -> (200, {"items": [...], "prefs": {...}, "board_stars": {run: [epochs]}, "prompt_labels": {run: [labels]}})`; `?fresh=1` 이면 캐시 무시. 머리말 읽기는 `asyncio.to_thread`.
    부를 때마다 `OutputScan.update(output_dir)` 를 백그라운드 스레드로 한 번 시작한다(이미 돌고 있으면 건너뜀) — 응답을 기다리게 하지 않는다.
  - `GET /loras/detail?name=` → `(200, {"item", "summary", "trigger", "epochs": [{"epoch", "name", "samples": n, "starred"}], "prompt_labels", "cover": "custom"|"sample"|"output"|None})`, 색인에 없는 이름 → `(404, {"error"})`
  - `GET /loras/thumb?name=&kind=cover|sample&epoch=&i=` → webp 바이트(`image/webp`, `Cache-Control: max-age=600`); 없으면 404 JSON.
    sample: C: 학습 샘플 i 번째(`epoch_samples`) → 없으면 보드 띠의 i 번째 칸(`board_strip` + `strip_tiles` crop) → 404.
    cover 순서: 직접 지정 → 마지막 에폭 sample 0번(위 순서) → 출력 그림(`OutputScan.latest_for(런의 에폭 stem 들)`) → 404
  - `GET /loras/prefs`, `POST /loras/prefs` → 정규화된 prefs
  - `POST /loras/preview` `{run_key, image: {filename, subfolder, type}}` → 직접 지정(이미지는 `studio_routes.resolve_image` 로 output/input 안에서만); `{run_key, clear: true}` → 해제
  - `register_lora_handlers() -> None`

- [ ] **Step 1: 실패하는 테스트**: `test_index_payload_shape`(가짜 folder_paths·임시 user dir), `test_sample_thumb_falls_back_to_board_strip_tile`(C: 샘플 없음 + 4칸 띠 → i=2 가 세 번째 칸 크기), `test_index_starts_one_background_scan`(연달아 두 번 불러도 스캔 스레드 1개), `test_detail_lists_epochs_of_the_run_with_stars`, `test_unknown_or_escaping_names_are_404` (Review Focus 4: `..\\..\\x`, `C:\\x`, 색인에 없는 이름), `test_unicode_names_round_trip` (Review Focus 3: `묵밥 (@Bjunbae)`·`໊ (@9an9ta)` 이름을 URL 인코딩해 detail 200), `test_thumb_falls_back_custom_sample_output`, `test_preview_rejects_images_outside_output_input`, `test_handlers_registered_under_prefix`
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** `run_tests.py` 전체 PASS
- [ ] **Step 5: 커밋 초안** `feat: LoRA 선택 창 라우트(index·detail·thumb·prefs·preview)`

### Task 7: 선택 창 상태 (JS, 순수)

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/lora_browser_state.js`
- Test: `comfy_custom_nodes/ai_studio_forge_parity/js_tests/lora_browser_state.test.mjs`

**Interfaces:**
- Consumes: Task 1 (`parseLoraPath`, `matchesQuery`, `familyVisible`), Task 6 index payload. Python 키는 snake_case(`run_key`,
  `num_epochs`, `group_label` …)로 온다 — `initialState` 에서 한 번 camelCase 로 바꾸고, 이 파일 밖으로는 camelCase 만 내보낸다
  (`POST /loras/prefs`·`/loras/preview` 로 보낼 때만 다시 snake_case).
- Produces:
  - `initialState({mode: "add"|"replace"|"xyz", current?: string, family: string, index, prefs}) -> State`
  - `reduce(state, action) -> State` — actions: `{type: "query", text}`, `{type: "folder", key}`(트리: `"*fav"`, `"*recent"`, `algo`, `algo/group`), `{type: "sort", by}`, `{type: "showAll", on}`, `{type: "move", dx, dy, cols}`, `{type: "select", runKey}`, `{type: "epoch", delta}`|`{type: "epoch", value}`, `{type: "toggleEpoch", value}`(xyz 모드 다중 체크), `{type: "favorite", runKey}`, `{type: "strength", value}`(0~2 로 자르고 소수 둘째 자리),
    `{type: "strengthStep", delta}`(±0.05), `{type: "closeAfter", on}`, `{type: "added", pick: {key, name, strength, label}}`, `{type: "unadd", key}`
  - `quickPick(state, runKey) -> {name, strength, label}` — 카드 버튼용: 그 런이 선택된 런이면 선택 에폭, 아니면 마지막 에폭(final), 세기는 현재 값
  - `state.added: [{key, name, strength, label}]` — 이번에 연 창에서 넣은 줄(아래 "이번에 추가" 칩); `key` = `name` + 줄 번호
  - `visibleRuns(state) -> Run[]` — `Run = {runKey, label, family, groupLabel, epochs: [{epoch, name, final}], favorite, latestDate}`; 계열 거르기(10절 결정 3), 검색, 폴더, 정렬
  - `picks(state) -> [{name: stem, strength: number|null}]` — add/replace: 선택 에폭 1개(기본 세기 1.0); xyz: 체크한 에폭들, strength null
- [ ] **Step 1: 실패하는 테스트**: `quickPick uses the selected epoch for the selected run and the final epoch otherwise`, `strengthStep clamps to 0..2 in 0.05 steps`, `added then unadd leaves the rest in order`, `closeAfter starts from prefs.close_after_add`, `groups epoch files into one run`, `replace mode starts on the current file's run and epoch`, `family filter hides base runs on 2.9b until showAll`(10절 결정 3), `unknown family stays visible`, `query narrows and keeps selection when still visible`, `arrow moves in a grid of cols`, `[ and ] step epochs and stop at the ends`, `xyz picks are the checked epochs in order`, `favorites and recent folders`
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** PASS
- [ ] **Step 5: 커밋 초안** `feat: LoRA 선택 창 상태(검색·거르기·에폭·다중 선택)`

### Task 8: 선택 창 화면 (DOM)

**Files:**
- Create: `comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/lora_browser.js`
- Modify: `comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/studio.css` (`ais-lb-*`)
- Test: `comfy_custom_nodes/ai_studio_forge_parity/js_tests/lora_browser.test.mjs` (DOM 없이 테스트할 부분만: 키 → action 표)

**Interfaces:**
- Consumes: Task 7, `studio_dom.js` `h()`/`pill()`, Task 6 라우트(주입된 `api`).
- Produces:
  - `keyAction(event: {key, shiftKey, ctrlKey}, focus: "search"|"grid") -> action | "add" | "addKeep" | "close" | "focusSearch" | null` — Enter→`"add"`, Shift+Enter→`"addKeep"`, Esc→`"close"`, Ctrl+F→`"focusSearch"`, 화살표→`move`, `[`/`]`→`epoch`, `f`(grid)→`favorite`; 검색칸에서 글자 키는 null
  - `openLoraBrowser({mode, current, family, api, onPick, onUnpick, onTrigger}) -> {close(): void}` — `api = {index(fresh), detail(name), thumbUrl(name, kind, epoch, i), savePrefs(prefs)}`;
    `onPick(pick) -> rowIndex`(누를 때마다 바로 패널에 한 줄; add 모드), replace 모드는 `onPick` 한 번 후 닫힘; `onUnpick(rowIndex)`(아래 칩 ✕); `onTrigger(trigger)`
- 화면(스펙 3·4절, 시안 https://claude.ai/artifact/8RBPWX6BmQnYGfZveAkvkG 의 Main·Xyz): 오버레이 `div.ais-lb`(`role=dialog`, `aria-modal`), 폭 `min(1200px, 95vw)`;
  헤더(검색·"모든 모델 · 숨김 N"·정렬·**"추가하면 닫기 켬/끔"**(add 모드만)·↻·✕); 트리 | 카드 격자(가상 스크롤: 보이는 줄 ±2 만 DOM, **카드마다 "+ eNN 추가" 버튼** = `quickPick`) |
  상세(에폭 칩 ★, 샘플 2×2 + 프롬프트 이름, 트리거 + "작가 칸에 넣기", 요약 정보, **세기 −/+·슬라이더·숫자**, "+ eNN 추가[하고 닫기]" / replace 모드 "이걸로 바꾸기", "미리보기 지정 해제");
  바닥줄(**"이번에 추가" 칩들(✕) + "완료 · N개 추가됨"/"닫기"**, 단축키는 작은 글자 title 로만). 모든 동작이 마우스로 되어야 한다(키보드는 보조).
  폭 < 900px 이면 트리 접기, 상세는 격자 아래.
  Esc: 열린 lightbox(`.ais-lightbox`)가 있으면 그쪽이 먼저 닫히도록 이 창의 keydown 은 lightbox 가 없을 때만 처리. 포커스 가두기, 닫으면 여는 버튼으로 포커스 복귀.
- [ ] **Step 1: 실패하는 테스트**: `keyAction` 표 전체(위 매핑, 검색칸에서 `f` 는 null)
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** `npm run test:pack` PASS
- [ ] **Step 5: 커밋 초안** `feat: LoRA 선택 창 화면`

### Task 9: 패널·자동완성 연결

**Files:**
- Modify: `web/ai_studio/panel_view.js:361-379` (`loraCard`), `web/ai_studio/studio_main.js` (ctx 에 `loraBrowser` 주입, 라우트 api), `web/ai_studio/studio_autocomplete.js:116`(힌트를 `shortLoraName`)
- Test: `js_tests/studio_autocomplete.test.mjs`(힌트), `tests/test_studio_frontend_contract.py`(필요하면 새 파일 목록)

**Interfaces:**
- Consumes: Task 1 (`shortLoraName`, `middleEllipsis`, `appendTrigger`), Task 8 (`openLoraBrowser`).
- Produces: `ctx.loraBrowser({mode, current, onPick}) -> void` (studio_main 이 api·family·onTrigger 를 채워 Task 8 을 부름). `onTrigger` 는 `state.prompt_artist = appendTrigger(state.prompt_artist, trigger)`.
- 변경: 줄의 `select` → `button.ais-lb-pick`(글자 `middleEllipsis(shortLoraName(entry), 40)`, `title` = 전체 경로, 이름이 목록에 없으면 기존처럼 빨간 표시) → `mode: "replace"`; "+ LoRA" → `mode: "add"`,
  `onPick` 이 누를 때마다 `{on: true, name, strength}` 한 줄을 push 하고 그 줄 번호를 돌려준다, `onUnpick(i)` 는 그 줄을 뺀다(사이에 사용자가 패널을 고쳤으면 name 이 같은 줄만 뺀다). 저장 형식 불변.
- 테스트 추가: `onPick then onUnpick restores the rows`, `onUnpick ignores a row the user already changed`.
- [ ] **Step 1: 실패하는 테스트**: 자동완성 LoRA 항목의 `hint` 가 짧은 이름; 패널 상태 왕복(`normalizeState`)이 `{on, name, strength}` 그대로(기존 golden 통과)
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** `npm run test:pack` + `run_tests.py` PASS
- [ ] **Step 5: 커밋 초안** `feat: 패널 LoRA 줄·+LoRA 를 선택 창으로, 짧은 이름 표시`

### Task 10: XYZ LoRA 줄 연결

**Files:**
- Modify: `web/ai_studio/xyz_lora_view.js:86-112` (`rowBlock`, `loraAxisBody`), 그 view 를 만드는 `panel_view.js:484` 부근
- Test: `js_tests/xyz_lora_view.test.mjs`

**Interfaces:**
- Consumes: `ctx.loraBrowser`(Task 9), Task 7 xyz picks.
- Produces: `addPickedRows(block, picks: [{name}]) -> block` (xyz_lora_view.js export) — 마지막 줄이 비었으면 그 줄을 첫 pick 으로 채우고 나머지는 `emptyRow(name)` 로 뒤에 추가. 기준 줄(index 0)은 지금처럼 프롬프트 태그 select 유지.
- 변경: 기준 아닌 줄의 select → 짧은 이름 버튼(`mode: "replace"`, 한 개), "+ 비교할 LoRA" → `mode: "xyz"`(여러 에폭 체크).
- [ ] **Step 1: 실패하는 테스트**: `addPickedRows fills the empty last row first`, `addPickedRows keeps base row and existing rows`, `picked names are stems (same as before)`
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** PASS
- [ ] **Step 5: 커밋 초안** `feat: XYZ LoRA 축도 선택 창으로(에폭 여러 개 한 번에)`

### Task 11: 갤러리 "LoRA 미리보기로"

**Files:**
- Modify: `web/ai_studio/result_view.js:12-20` (`GALLERY_BUTTONS` 에 `["lora_preview", "LoRA 미리보기로"]`, 아이콘), `web/ai_studio/studio_main.js:209` (`galleryAction`)
- Test: `js_tests/result_view.test.mjs`, `js_tests/run_controller.test.mjs` 해당 없으면 새 순수 함수 테스트

**Interfaces:**
- Consumes: `POST /loras/preview`(Task 6), `studio_pnginfo.js`(그림의 인포텍스트 읽기), Task 1 `parseLoraPath`.
- Produces: `previewTargets(infotext: string, entries: string[]) -> [{runKey, label}]` (lora_names.js 에 추가) — 인포텍스트의 `<lora:` 이름을 목록 entry 와 stem 으로 맞춰 런 단위로.
- 동작: 고른 그림 → 대상이 1개면 바로 지정, 여러 개면 작은 선택 목록, 0개면 알림 "이 그림에 LoRA 가 없음".
- [ ] **Step 1: 실패하는 테스트**: `previewTargets` (두 에폭이 같은 런이면 하나로, 목록에 없는 이름 제외), 버튼 표에 `lora_preview` 포함
- [ ] **Step 2:** FAIL → **Step 3:** 구현 → **Step 4:** PASS
- [ ] **Step 5: 커밋 초안** `feat: 갤러리 그림을 LoRA 미리보기로 지정`

### Task 12: 버전·문서·설치·화면 확인

**Files:**
- Modify: `comfy_custom_nodes/ai_studio_forge_parity/__init__.py:78`, `core/comfy_node_pack.py:116`, 버전 고정 테스트 3곳(1.18.0 → 1.19.0), 팩 `README.md`(1.19.0 변경 기록), `studio_forge_parity.md`(clean-room 메모: LoRA Manager 는 문서만 참고)
- 확인: `THIRD_PARTY_NOTICES.md` 변화 없음

- [ ] **Step 1:** 버전·문서 수정 → `run_tests.py` 전체 PASS, `npm run test:pack` 전체 PASS
- [ ] **Step 2:** `core.comfy_node_pack.install_bundled_node_pack` 로 설치(ComfyUI 8188 이 켜져 있으면 제자리 복사 후 사용자에게 재시작 안내 — 직접 재시작 안 함)
- [ ] **Step 3:** CPU 임시 ComfyUI(8197) 띄워 브라우저로 — **마우스만으로**(키보드 없이) 카드 "+ 추가" 두 번 → 패널 줄 2개·바닥 칩 2개, 칩 ✕ → 줄 1개, "완료" 로 닫기, "추가하면 닫기" 켜고 한 번 추가 → 바로 닫힘·다시 열어도 켜져 있음, 세기 −/+; 그다음: 창 열림, 871개 첫 색인 시간 기록(두 번째는 즉시), `bjun 14` 검색 → 추가 → 패널 줄 짧은 이름·저장값 stem, 줄 눌러 바꾸기, 2.9B 에서 base 런 숨김·"모든 모델" 로 보임, 샘플 있는 런(GT_DunE fp16) 에폭 칩·4장, XYZ 에폭 3개 체크 → 줄 3개, 트리거 넣기 → 작가 칸, Esc 순서(lightbox 먼저), 콘솔 오류 0. 끝나면 8197 의 python 만 종료.
- [ ] **Step 4:** 스크린샷을 사용자에게 보여 주고 커밋은 사용자가 요청할 때(`/ship`): `feat: 스튜디오 1.19.0 — LoRA 선택 창(검색·에폭 미리보기·XYZ)`
