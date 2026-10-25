# UR-al/sd-scripts `cloud/integration` — 병합 내역, CI, 로컬 GPU A/B 체크리스트

- 브랜치: UR-al/sd-scripts **`cloud/integration`** = `moss-image`(3a4f196b) + `cloud/review-fixes`(a1a8acb1) + `cloud/anima-speed-ports`(f738479b) + 리뷰 권고 A2·bf16 테스트 허용폭 커밋. `moss-image` 는 손대지 않았다 — 로컬 학습이 끝난 뒤 로컬 세션이 `moss-image` 를 이 브랜치로 **fast-forward** 하면 된다(3a4f196b 를 조상으로 두고 위에 쌓았으므로 FF 가능).
- 최종 SHA: **`34b842b414dfdb28d2c90ca8331be55ddb890e1e`** (푸시됨; `1436f253` 위에 §2.1 의 수정 커밋). CI 결과: §3.

## 1. 병합 내역

| 순서 | 커밋 | 내용 | 충돌 |
|---|---|---|---|
| 기준 | `3a4f196b` | moss-image (`--step_rows`, TSM 마스크드 단일 호출) | — |
| 1 | `813f8f9` merge `cloud/review-fixes` (`a1a8acb1`) | `--dataset_class` 행수 0 수정, 에폭 셔플을 첫 fetch 앞으로(`attach_epoch_value`/`sync_epoch`), `RowPacker` 조각 복사·부모 해제, `train_batch_size > step_rows`·`--dataset_class` 경고, 재개 비용 문서화, 테스트 6개 | 없음(자동 병합) |
| 2 | `d9f146b` merge `cloud/anima-speed-ports` (`f738479b`) | `--anima_block_compile` + 하위 옵션, `--anima_activation_memory_budget`, `library/anima_block_compile.py`, 가짜 5D 블록 루프, LyCORIS 재바인딩·TSM 호이스트, attention 루프, `after_network_loaded` 훅, `tests/test_anima_block_compile.py`(CI 목록), 문서, `THIRD_PARTY_NOTICES.md` | 없음(자동 병합) — 두 브랜치가 같은 파일을 만진 곳은 `train_network.py`(서로 다른 함수: `after_network_loaded` 훅/호출 vs DataLoader 생성 전 `attach_epoch_value`)와 `docs/anima_train_network.md`(다른 항목) 뿐 |
| 3 | `1436f253` 테스트 커밋 (§2) | A2 매개화 + bf16 허용폭 | — |
| 4 | `34b842b4` (§2.1) | 블록 컴파일: LyCORIS 모듈 지연 캐시를 첫 trace 전에 채움(`prime_module_caches`), 테스트 fixture 의 패치 순서를 운영 순서로, 콜드 스타트 테스트 | — |

### 블록 컴파일 경로와 review-fixes 변경의 정합성
- **`attach_epoch_value` / `sync_epoch`** 는 데이터셋 `__getitem__` 쪽(DataLoader 워커·메인 프로세스의 fetch) 에서만 동작하고, 블록 컴파일은 `after_network_loaded`(네트워크 적용·가중치 로드 뒤, `accelerator.prepare` 앞)에서 모델의 Block 만 감싼다. 두 변경이 공유하는 상태는 없다. 블록 컴파일이 데이터셋에서 읽는 것은 `token_counts_from_dataset_group`(버킷 해상도 → 토큰 수 밴드) 하나이고, 이것은 `bucket_manager.resos` 만 읽는다 — 에폭 셔플은 `buckets_indices` 순서만 바꾸고 해상도 집합은 바꾸지 않는다.
- **`RowPacker` 조각 처리**는 StepPack 의 서브배치(행 슬라이스) 를 만들고, 트레이너는 서브배치 → `split_batch_for_backward` 청크 → 청크마다 `set_timestep(그 청크의 timestep)` → forward 를 돈다. 블록 컴파일의 `skip_call = mixed_timestep_chunk()` 는 그 청크의 `_TSM_SAMPLE_T` 를 읽으므로 **청크 단위**로 "혼합이면 eager, 단일이면 컴파일" 이 결정된다 — 잘린 조각이든 통 배치든 같은 규칙. batch 1(또는 `--step_rows` + batch 1) 이면 모든 청크가 1행이라 TSM 도 항상 컴파일 경로.
- **`step_rows` 경고**(`train_batch_size > step_rows`, `--dataset_class`) 는 `step_rows_messages` 에서 나오고 `validate_block_compile_args` 와 독립(둘 다 `assert_extra_args` 단계).
- 검증: 병합 트리에서 `tests/test_anima_block_compile.py`(32) + `tests/test_micro_batch.py`(23) 를 포함한 CI 전체 실행(§3).

## 2. 테스트 커밋 (리뷰 권고 A2, bf16 허용폭)

- **A2**: `test_tsm_grouped_batch_equals_per_sample_calls`, `test_tsm_mixed_plan_and_switch`, `test_tsm_fused_both_lowrank_modes_match_split_fp64` 를 `tsm_n_scales ∈ {(8,1), (4,2,1)}` 로 매개화하고 부분 겹침 청크 `[100, 300, 600, 900]`(코어 전문가 4개 모두 다르고 컨텍스트 전문가를 공유: (4,2,1) 에서 |U| = 7) 을 추가. 플랜 테스트는 공유 블록이 **그 블록을 고른 행에서만** 켜지는지((1,0) → 행 0·1, (1,1) → 행 2·3, (2,0) → 전부) 확인. 결과: 90 케이스 전부 샘플별 / 그룹별 / 분리 경로 기준과 일치(fp64 rtol 1e-9; fp32 저랭크 모드는 2e-4).
- **`test_factored_forward_bf16_not_worse[ortholora]`**: 이 테스트의 5항 평균(출력 + 기울기 4개의 fp64 대비 상대 오차, 6 시드)은 반올림 위치가 다른 두 경로를 비교하므로 시드마다 몇 % 씩 양쪽으로 움직인다. 이 CPU(torch 2.7.1+cu126 휠, avx512_bf16 없음 → oneDNN 에뮬레이션 bf16 GEMM)에서 36 시드 실측: ortholora 시드별 flag/stock 비 **0.83 ~ 1.11**(sd 0.058 → 6-시드 평균의 sd ≈ 2.4%), 6-시드 창별 비 1.016 / 0.994 / 0.969 / 0.954 / 0.948 / 0.945 — 테스트의 시드 0~5 가 정확히 1.016 으로 실패(GitHub 러너의 `+cpu` 휠에선 통과). lora 는 0.84 ~ 0.87 로 안정. 출력 항 단독은 모든 시드에서 0.70 ~ 0.73(flag 가 30% 더 정확). → 평균에 **10% 허용폭**(≈ 4 sd) 을 두고 출력 항은 엄격 유지. "stock 보다 의미 있게 나쁘지 않다" 는 의도는 유지된다: 반올림 자리 하나의 가치가 ~30%(출력 항이 보여 줌)이므로 진짜 회귀(예: stock 이 fp32 로 두는 곱을 bf16 으로)는 10% 를 훌쩍 넘는다. 근거는 테스트 주석과 커밋 메시지에.

### 2.1 통합 CI 가 드러낸 결함과 수정 (블록 컴파일, LoKr+DoRA 콜드 스타트)

병합 트리의 21 파일을 **한 프로세스**로 돌리자 `test_lycoris_algo_fixes.py::test_lokr_fixed_path_still_runs_the_fast_rebuild`, `test_moss_pipeline.py::test_fast_lokr_forward_is_bit_identical_to_lycoris`, `::test_fast_lokr_with_bias_and_cached_static_check` 3개가 실패했다(두 브랜치를 따로 돌릴 땐 통과). 원인은 `tests/test_anima_block_compile.py` 가 그 앞에서 실행되며 fixture 가 **운영 순서의 반대로**(`apply_module_patches()` → `install_fast_lokr_forward()`) 패치해 `install_fast_lokr_forward` 가 수정 경로의 LoKr forward 를 "stock" 으로 기록한 것. fixture 를 운영 순서(`prepare_anima_runtime` 처럼 fast forward 먼저)로 고쳤다.

그러자 **블록 컴파일의 진짜 결함**이 드러났다: 트레이너는 가중치 로드 직후 컴파일하므로 **첫 trace 가 네트워크의 첫 forward** 인데, LoKr+DoRA 의 `lycoris_fast._compute_static_eligible` 가 모듈마다 한 번 `float(scalar)`(= `.item()`) 를 읽는다 → 컴파일 그래프 안에서 그래프 분단 → dynamo 가 그때 컴파일한 조각(Block 호출당 **21 그래프**)을 그대로 유지하고 해상도마다 +2 재특화. 이전 테스트는 모두 컴파일 전에 eager 스텝을 한 번 돌려 캐시가 따뜻했기에 못 봤다(콜드 explain: lokr_dora 19 breaks `Tensor.item`, 나머지 4 프리셋 0). 수정: `lycoris_algo_fixes.prime_module_caches(network)` 가 지연 캐시(LoKr `_static_eligible`/`_fast_lokr_scale`, LoCon/OrthoLoRA fast 플래그)를 eager 로 채우고, `after_network_loaded` 가 **가중치 로드 뒤·compile_blocks 앞**에 부른다(LoKr 판정은 로드된 scalar 에 의존). 새 테스트: 5 프리셋 모두 콜드 네트워크에서 prime 뒤 1 그래프·0 분단, lokr_dora 는 prime 없이는 `.item()` 분단; 해상도 스윕·재컴파일 수 테스트도 이제 트레이너처럼 콜드 네트워크를 컴파일한다. GPU A/B 때 **`TORCH_LOGS=recompiles` 에 `_fast_forward`/`_static_eligible` 프레임이 따로 보이면 이 결함의 재발** — §4.3.

## 3. CI (`.github/workflows/moss-image.yml` 의 21 파일 = 기존 20 + `tests/test_anima_block_compile.py`, `CUDA_VISIBLE_DEVICES=-1`, `PYTHONIOENCODING=utf-8`, torch 2.7.1 CPU)

- 1차 실행(`1436f253`, 한 프로세스, 8:34): **1024 passed, 53 skipped, 3 failed** — §2.1 의 LoKr 3개(테스트 순서 상호작용 → 수정 `34b842b4`). ortholora 는 허용폭으로 통과.
- 2차 실행(`34b842b4`, 한 프로세스, 8:24): **1032 passed, 53 skipped, 0 failed** (53 skip 은 GitHub CI 와 같은 환경 skip: 모델 파일·Forge·외부 체크아웃 없음). 가장 느린 테스트 35.6 s(`test_lowrank_bf16_error_not_worse_than_stock[dense-tlora]`).

## 4. 로컬 GPU A/B 체크리스트 (RTX 5090 32 GB, Windows 11, torch 2.7.1 + Triton)

목적: 블록 컴파일의 실제 이득(추정 −20~−35%/스텝, 512/1024 티어 중심)과 메모리를 **실측**하고, 이상(그래프 분단·재컴파일 폭주·OOM)을 걸러 낸다. 순서: **① LoKr+DoRA 16 스텝 → ② TSM 1단계 16 스텝**, 각각 A(끔) B(켬) A B 교차.

### 4.1 공통 준비
- 코드: `cloud/integration`(§3 의 SHA) 체크아웃. 기본값은 전부 끔 → 플래그 없이 돌리면 3a4f196b 와 같은 학습(review-fixes 의 에폭 셔플 정정만 포함: 에폭 첫 배치가 이전 순열이 아니게 됨).
- 데이터·설정: 실제 4-tier 세트, `batch_size 1 × gradient_accumulation_steps 8`(기존 기준선 구성) 또는 `--step_rows 8`(이때 `train_batch_size 1`). **TSM 은 batch 1 이어야 모든 청크가 컴파일 경로**(batch > 1 의 혼합 timestep 청크는 eager).
- 같은 `--seed`, `--max_train_steps 16`(`--step_rows` 면 24), 샘플 이미지·검증 끔(`--sample_every_n_steps 0`), `--save_every_n_steps 0`, 로그만.
- 컴파일 캐시: `TORCHINDUCTOR_CACHE_DIR=D:\cache\torchinductor`(원하는 위치; 없으면 `%TEMP%\torchinductor_<user>`). 블록 컴파일은 그 아래 **`anima-blocks-<해시16>`** 하위 폴더를 쓴다(밴드·백엔드·모드·torch 버전별 서명). 첫 실행이 채우고 두 번째 실행부터 재사용. 지우면 다음 실행이 다시 컴파일.

### 4.2 켜는 플래그와 함께 맞출 옵션
| 항목 | 값 | 이유 |
|---|---|---|
| `--anima_block_compile` | B 에서만 | 본 플래그 |
| `--anima_block_compile_backend` | 기본 `inductor` | `aot_eager`/`eager` 는 디버그용(속도 이득 없음) |
| `--anima_block_compile_dynamic` | 기본 `union` | 재컴파일이 많으면 `static`(버킷 토큰 수마다 1회 컴파일) 과 비교 |
| `--anima_block_compile_recompile_limit` | 기본 128 | TSM: 2 × 8(코어 전문가) × 3(GC 방식) = 48 안 |
| `--anima_activation_memory_budget` | 1.0(끔) | 체크포인팅 없는 구간이 생길 때(`--anima_no_checkpoint_max_tokens`) 0.7 로 별도 실험 |
| `--blocks_to_swap`, `--use_ramtorch*`, `--torch_compile`, `--lycoris_stock` | 끔 | 블록 컴파일과 함께면 시작 시 거부 |
| `--split_attn` | 끔(Anima 기본) | 켜면 attention 마다 행 단위 그래프 분단(경고) |
| `--anima_no_compile`, `--anima_no_lokr_compile` | 아무 값 | 블록 컴파일 아래선 elementwise/LoKr 재빌드 compile 이 자동으로 꺼짐 |
| `--anima_no_checkpoint_max_tokens`, `--anima_half_checkpoint_max_tokens`, `--anima_adaln_hoist` | A/B 동일 | GC 방식이 그래프 종류를 바꾸므로 A 와 B 에서 같은 값으로 |
| 로그 | `CUDA_MEMLOG=4` | 4 스텝마다 allocated/reserved/peak |

### 4.3 GPU 에서 그래프 분단·재컴파일 확인
- `set TORCH_LOGS=graph_breaks,recompiles`(PowerShell: `$env:TORCH_LOGS="graph_breaks,recompiles"`) 로 B 를 한 번 돌린다.
  - 기대: **graph break 0 건**(한 줄도 없어야 함). 뜨면 그 줄의 파일·이유를 기록(`library/attention.py`·`lycoris_algo_fixes.py` 어디인지).
  - 기대 recompile 수: 2(Block 0 bf16 입력 vs 나머지 fp32 잔차) × 전문가 선택 수(LoKr 1, TSM 1단계 n_core=8) × GC 방식 수(`--gradient_checkpointing` 만: 1~3 — 체크포인트 forward(no-grad)·재계산 forward·backward 를 inductor 가 어떻게 나누느냐에 따라) × 밴드 수(union 1). LoKr 에서 4 티어를 다 지난 뒤에도 recompile 줄이 계속 늘면 밴드가 작동하지 않는 것 → `--anima_block_compile_dynamic static` 과 비교하고 보고.
  - 시작 로그 `block compile: N Blocks compiled per call (backend inductor, ... union bands [(lo, hi)], token counts [...], K LyCORIS module caches primed)` 에서 밴드가 데이터의 토큰 수 범위(512² = 1024 … 2048² = 16384)를 덮는지, K 가 네트워크 모듈 수(LoKr: 640)인지 확인. `... is outside every compiled band` 경고가 뜨면 그 해상도는 정적 특화(한 번 더 컴파일).
  - §2.1 의 콜드 스타트 결함 재발 신호: `TORCH_LOGS=recompiles,graph_breaks` 에 `_fast_forward` / `_static_eligible` / `torch_dynamo_resume_in__fast_forward_*` 가 **별도 프레임**으로 보이거나 `Tensor.item` 분단이 뜨면 캐시가 trace 전에 안 채워진 것(prime 이 안 돌았거나 새 지연 캐시가 생긴 것) — 보고.
- 더 자세히: `TORCH_LOGS=+dynamo` 는 매우 길다; 필요하면 첫 10 스텝만.

### 4.4 예상 첫 컴파일 시간
- 그래프 하나(Block forward+backward, inductor+Triton) 당 수십 초. LoKr: 약 2~6 그래프 → **첫 스텝들에서 1~5분**. TSM 1단계: 전문가 선택이 바뀔 때마다(8 가지) 그만큼 더 — 첫 에폭 동안 점진적으로, 합계 수 분~십여 분. 두 번째 실행(캐시 워밍)은 수 초~수십 초.
- 그래서 16 스텝 비교는 **같은 구성을 두 번** 돌리고 두 번째(캐시 워밍) 실행의 **마지막 8 스텝** 으로 s/step 을 읽는다(또는 첫 실행의 마지막 8 스텝: 컴파일이 끝난 뒤의 스텝).

### 4.5 잴 지표
| 지표 | 어디서 | 비고 |
|---|---|---|
| s/step | tqdm 진행바(마지막 8 스텝 평균; 또는 로그의 스텝 시각 차) | A B A B 네 값, 쌍별 차이 |
| `max_allocated`, reserved | `CUDA_MEMLOG=4` 창 피크 | B 가 A 보다 커지면 양과 어느 티어에서 |
| 티어별 마이크로스텝 시간 | 트레이너에 티어별 타이머가 없음 → (a) 512 만 / 1024 만 / 2048 만 담은 하위 폴더로 각 8~16 스텝 단일 티어 실행(A/B), 또는 (b) 전체 세트의 s/step 만 | 예상: 512/1024 는 크게, 2048 은 거의 그대로 |
| 손실 곡선 | 같은 seed 의 A 와 B | bf16 반올림 수준(~1e-3)에서 겹쳐야 함 — 벌어지면 수치 문제 |
| 컴파일 시간 | 첫 실행의 첫 스텝들(진행바가 멈춘 시간) | 참고용 |

### 4.6 실행 순서
1. **LoKr+DoRA**(`algo=lokr factor=4 full_matrix dora_wd wd_on_output`, 기존 설정) — A 16 스텝 → B 16 스텝(첫 실행: 컴파일) → B 16 스텝(캐시 워밍) → A 16 스텝. s/step·메모리·손실 비교, `TORCH_LOGS` 로 분단 0·재컴파일 수 기록.
2. **TSM 1단계**(`algo=tsm tsm_n_scales=8,1 tsm_stage=1`, dim 32) — 같은 순서. 재컴파일 수가 LoKr 의 ~8배(전문가 선택)인지 확인; `step_rows`/batch > 1 을 쓰면 혼합 청크는 eager 라 이득이 줄어드는 것이 정상.
3. 결과(네 s/step, 피크, 분단/재컴파일 수, 첫 컴파일 시간)를 `reports/` 에 한 표로.

### 4.7 되돌리는 법
- 즉시: B 플래그 제거(`--anima_block_compile` 없으면 코드 경로가 전과 같음; 컴파일 캐시는 디스크만 차지 → `anima-blocks-*` 폴더 삭제).
- 코드: `moss-image` 는 3a4f196b 그대로이므로 FF 전엔 그냥 `moss-image` 로 돌아가면 됨; FF 뒤엔 로컬에서 `git checkout 3a4f196b`(또는 `git branch backup/pre-integration 3a4f196b` 를 FF 전에 만들어 두기). 원격 강제 푸시는 하지 않는다.
- 부분 되돌림: 블록 컴파일만 빼고 싸면 `git revert -m 1 d9f146b`(병합 커밋 되돌림) 를 로컬에서; review-fixes 의 에폭 셔플 정정만 빼려면 `git revert -m 1 813f8f9`.

## 5. 미확인
- GPU 실측은 이 보고서 작성 시점에 없다(이 환경에 GPU 없음). §4 가 그 절차.
- 블록 컴파일 + `--step_rows`(batch 1) 의 동시 실행은 CPU 테스트가 각각을 덮고 상호작용 지점이 없음을 코드로 확인했지만, 트레이너 전체를 한 번에 돌린 통합 스모크(실제 모델)는 GPU 쪽 1 단계에서 함께 확인된다.
