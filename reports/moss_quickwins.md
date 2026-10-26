# MOSS-Image 학습기 퀵윈 5건 (UR-al/sd-scripts `cloud/moss-quickwins`)

- 브랜치: UR-al/sd-scripts **`cloud/moss-quickwins`** = **`704b11b5589891349bc2b28be7b29a0253ee9997`** — 현재 `moss-image` **1f7f231**(cloud/block-compile-algos 병합) 위 1 커밋, 새 브랜치로 푸시. 작업은 0938398 에서 시작했고, 그사이 `moss-image` 가 1f7f231 로 움직여 푸시 전에 그 위로 rebase 했다(충돌 없음). 블록 컴파일 이식은 하지 않았다.
- 줄 번호는 이 커밋 기준. Anima 쪽 줄은 1f7f231 기준이라 요청에 적힌 번호보다 20줄 밀려 있다(OT :1416–1428 → **:1436–1448**, adv :1469–1470 → **:1489–1490**).
- 검증: GPU 없음 — 전부 CPU(`CUDA_VISIBLE_DEVICES=-1`, torch 2.7.1 CPU, LyCORIS c2cfd8ae) 테스트.
- **옵션을 끈 eager 결과는 moss-image 와 비트 동일**: 같은 스크립트로 만든 텐서 **5583개가 전부 `torch.equal`** — 이 브랜치 대 현재 moss-image 1f7f231, rebase 전 커밋 대 0938398 두 쌍 모두(§6.3). 의도적으로 달라지는 것은 TSM·T-LoRA·TSM-LoKr 망의 **샘플 이미지**(§1)와, 켰을 때의 새 옵션뿐이다.

| # | 요청 | 결과 |
|---|---|---|
| 1 | 샘플링 타임스텝 → TSM·TSM-LoKr·T-LoRA | 연결됨. 전에는 샘플 이미지 매 스텝이 "타임스텝 없음"(TSM 최고 노이즈 전문가, T-LoRA 전체 랭크)으로 돌았다 |
| 2 | 어텐션 마스크 경로 GPU→CPU 동기화 제거 | 마이크로스텝당 호스트 읽기 **1회**(전: 실제 모델 배치 1 에서 forward 92회, 체크포인팅 재계산까지 180회). strategy.py 의 배치당 `.item()` 도 제거 |
| 3 | `--flow_use_ot`·`adv_scale` 조용히 무시 | 둘 다 **이식**(거부 아님). adv 는 디스크 캐시 latents 의 격자 키(stride 8 가정)도 고쳐야 돌았다(전에는 디스크 캐시면 시작 시 죽고, 메모리 캐시면 INERT) |
| 4 | `--anima_tier_timing` 의 MOSS 판 | `--moss_tier_timing` / `--moss_tier_timing_every`. patch 1, 텍스트 토큰은 티어 옆에 따로. 끄면 속성 확인 1회 |
| 5 | TSM grouped accumulation 을 MOSS 에서 | 켜짐(`_tsm_grouped_supported`, `grouped_draw`). 타임스텝 전달 테스트 추가 |

## 1. 샘플링 타임스텝 (TSM · TSM-LoKr · T-LoRA)

**원인.** `train_network._lycoris_fix_wrap_sampling`(train_network.py:1461–1497)은 `_lycoris_fix_sampler_timestep_scale()` 이 값을 주면 DiT 에 forward pre-hook 을 걸어 `h_args[1] × scale` 을 그 스텝의 타임스텝으로 넣는다. 기본값은 `None`(train_network.py:1457, "이 모델은 스텝별 훅 없음")이고 MOSS 는 이를 물려받아 훅이 없었다 → 샘플 이미지 매 스텝에서 TSM 은 `no timestep set; using t = T (highest-noise experts)` 경고와 함께 최고 노이즈 전문가, T-LoRA 는 전체 랭크로 돌았다(테스트에서 재현: 아래 `inherited` 분기).

**수정.** `moss_train_network.py:327` `_lycoris_fix_sampler_timestep_scale()` → **1.0**. MOSS 샘플러(`library/moss/train_utils.py:365,367` `moss_sample`)는 이미 `dit(latents, t, …)` 에 `t = 1000·σ` 를 넘기므로 학습 마스크·전문가 구간과 같은 `[0, 1000]` 척도다(Anima 는 σ 를 넘기므로 1000.0, anima_train_network.py:1294). cond 와 skip-layer uncond 두 호출 모두 같은 t 를 받는다. TSM-LoKr 은 `lycoris_fast.cached_lokr_weights` 의 고정 캐시 대상(LokrModule)이 아니라 스텝마다 전문가를 고른다. 학습 경로는 바뀌지 않고, 이 알고리즘들의 **샘플 이미지만** 달라진다(의도한 변화 — 학습 때와 같은 타임스텝별 전문가·마스크로 그린다).

**테스트** (`tests/test_moss_pipeline.py`)
- `:748 test_sample_images_feed_every_sampler_step_timestep` — 프리셋 moss 의 TSM / T-LoRA / TSM-LoKr 망을 tiny DiT 에 걸고 `_lycoris_fix_wrap_sampling` 으로 감싼 `moss_sample`(4 스텝 × cond+uncond = 8 호출)을 돌린다. 각 DiT 호출의 t 가 `moss_inference_sigmas(4, 3.0)·1000`(1000/858/602/9)과 같고, 블록 안 어댑터 Linear 에서 읽은 상태(TSM 정수 타임스텝, T-LoRA 마스크)가 `lycoris_algo_fixes.set_timestep` 이 그 t 로 만드는 값과 같다. 스케일을 `None`(물려받던 값)으로 돌리면 8 호출 모두 "타임스텝 없음". 끝나면 상태가 지워진다.
- `:720 test_training_step_gives_tsm_tlora_tsmlokr_their_timesteps` — 학습 경로(`get_noise_pred_and_target` → `apply_tlora_mask`)에서 같은 세 알고리즘이 DiT 에 넘긴 샘플별 타임스텝을 받는지, **그래디언트 체크포인팅의 backward 재계산에서도** 같은 값을 보는지(어댑터 Linear 의 pre-hook 이 forward 1 + 재계산 1 = 2번), backward 뒤엔 지워지는지. 요청에서 짚은 "test_moss_pipeline.py 에 없던" 전달 테스트가 이것이다.

## 2. 어텐션 마스크 경로의 GPU→CPU 동기화

**원인.** `library/attention.py` 의 `attention()` 은 마스크가 있으면 호출마다 `torch.all(seqlens == seqlens[0])`(→ `bool`)와 균일할 때 `.item()` 을 했고, split 경로는 샘플마다 디바이스 원소를 슬라이스 끝으로 썼다(q·k·v 각각 동기화). MOSS 는 텍스트 마스크를 늘 넘기므로(`moss/models.py` 의 `create_attention_params_from_mask` 두 곳) **블록마다** 동기화가 생겼다. forward 앞부분에도 `torch.equal`, `(lengths == 0).any()`, `lengths.max()`, `lengths.tolist()` 4번.

**수정.**
- `library/attention.py:75–78` `AttentionParams.seqlens_host`(파이썬 int 리스트) 추가. `:93–130` `create_attention_params_from_mask(…, text_lengths=None)` — 호출자가 이미 호스트에 가진 길이를 주면 `seqlens_host = [n + img_len]` 를 함께 싣는다(xformers 의 `BlockDiagonalMask` 도 `.cpu().tolist()` 대신 이 리스트, `:123`).
- `:168–186` `attention()` 의 균일 길이 트림 판단과 `seqlen`, `:211–213` split 경로의 슬라이스 끝이 `seqlens_host` 를 쓴다. `seqlens_host` 가 없으면(Anima·HunyuanImage — `text_lengths` 를 안 넘김) **예전 경로 그대로**.
- `library/moss/models.py:356–370` `_right_align_text` — 길이와 "이미 정렬됨" 판정을 `torch.cat(...).tolist()` **한 번**으로 호스트에 가져오고 길이를 int 리스트로 돌려준다. `:392–412` forward 의 빈 캡션 검사·`max_len`·RoPE 행·두 `create_attention_params_from_mask(…, text_lengths=lengths)` 가 이 int 를 쓴다. `:290,397` `last_text_tokens`(§4 가 읽음, 이미 있는 int 대입 1회).
- `library/moss/strategy.py:172–196` — `drop_cached_text_encoder_outputs` 의 배치당 `torch.all(...).item()`(accelerate 가 배치를 GPU 로 옮긴 뒤라 매 스텝 동기화)을 없앴다. 같은 검사를 출력을 **만들거나 읽는 호스트 쪽**으로 옮김: `check_caption_dropout_rate` 를 `load_outputs_npz`(`:259`, DataLoader 워커) 와 메모리 캐시 `cache_batch_outputs`(`:286`)에서 부른다. CPU 에 남은 rates(리스트·CPU 텐서)는 여전히 여기서 검사. (`validate_moss_args` 가 시작 시 `caption_dropout_rate > 0` + 변형 없음 을 이미 거부하므로 이건 2차 방어선이다. 메모리 내 변형 재인코딩 경로는 MOSS 에서 `--cache_aug_refresh_epochs` 거부로 막혀 있다.)

**측정** — 호스트 읽기 카운터(`tests/_device_reads.py`: `aten._local_scalar_dense`·`aten.equal`·`aten.is_nonzero` 디스패치 + `Tensor.tolist/cpu/numpy`; GPU 라면 각각이 동기화)로 tiny 모델(본 블록 3 + 텍스트 리파이너 2 = 어텐션 5회) forward+backward 1회:

| 경로 | moss-image | 이 브랜치 |
|---|---|---|
| 배치, 길이 균일(배치 1 포함) | 14 (체크포인팅 24) | **1** (1) |
| 배치, 길이 제각각 | 9 (14) | **1** (1) |
| split, 배치 1 / 배치 2 | 19 / 34 (34 / 64) | **1** (1) |

실제 모델(본 42 + 리파이너 2 = 어텐션 44회)로 환산하면 배치 1 에서 forward 92회(4 + 2·44), 체크포인팅 재계산까지 **180회 → 1회**(`enable_gradient_checkpointing` 이 리파이너 블록 2개도 체크포인팅하고, `MossTextProjection` 이 그때 입력에 grad 를 켜 둬서 44 블록 모두 재계산된다). split/xformers 배치 B 는 forward 당 4 + 3B·44 회 → 1회. 이게 실제로 몇 % 빨라지는지는 GPU 가 없어 못 쟀다(§7).

남은 1회: forward 가 받을 때 텍스트 마스크는 이미 GPU 에 있어서(accelerate 의 DataLoader 가 배치를 옮김) 길이를 한 번은 가져와야 한다. collate 에서 CPU 길이를 배치에 실어 보내면 0회도 되지만 모든 학습기가 쓰는 `train_util` collate 를 바꿔야 해서 하지 않았다(배치당 1회로 요청 범위는 맞음).

**테스트**
- `tests/test_attention_host_seqlens.py`(신규, CI 목록 `.github/workflows/moss-image.yml:60` 에 추가) — `:51` 호스트 길이 경로 출력이 디바이스 경로와 **비트 동일**(배치/split × img_len 0/4 × fp32/bf16 × 균일·제각각·배치 1·꽉 찬 길이, 리스트 입력 형태 포함), `:66` 호스트 경로는 텐서 값 읽기 **0회**(디바이스 경로는 >0 — 카운터가 잡는다는 대조), `:81` 그래디언트도 비트 동일, `:39` 파라미터에 실리는 값.
- `tests/test_moss_models.py` — `:378` 실제 MOSS forward 를 "모든 어텐션이 디바이스 길이를 읽는" 예전 경로로 강제한 것과 출력·그래디언트 비트 동일(배치/split × 정렬/비정렬), `:404` forward 당 읽기 정확히 1회(`tolist`), 체크포인팅 backward 재계산도 0회, `:360` `_right_align_text` 가 예전 구현(moss-image 0938398 사본)과 같은 text/mask·길이, `:427` `last_text_tokens`.
- `tests/test_moss_pipeline.py:958` 캡션 드롭아웃 검사가 호스트에서(디스크 캐시는 읽을 때, 메모리 캐시는 만들 때 거부, meta 텐서 rates 는 읽지 않음).

## 3. `--flow_use_ot` 와 `adv_scale`

둘 다 MOSS `get_noise_pred_and_target` 에 **옮겼다**(시작 시 거부 대신).

- **OT** — `moss_train_network.py:210–211, 251–263` `_ot_reorder_noise`: Anima(anima_train_network.py:1436–1448)와 같이 `train_util.cosine_optimal_transport` 로 배치의 노이즈 행을 latents 에 재배정, **antithetic 페어링 전**, 첫 배치에서 한 번 로그. 차이 하나: MOSS 노이즈는 fp32 이고 캐시 latents 는 bf16 일 수 있어 `torch.mm` 비용을 fp32 로 계산(Anima 는 latents dtype). 배치 1 은 건드리지 않음.
- **adv** — `:232–233` `self._adv_noisy_latents = noisy.detach()`(Anima :1489–1490 과 같음). 전에는 저장을 안 해 `_apply_adv_term` 이 "INERT" 경고 한 번 뒤 손실을 그대로 돌려줬다(메모리 캐시 latents 의 경우 — 디스크 캐시는 아래 이유로 그 전에 시작 시 죽었다). MOSS 는 이미 `hf_prediction_mode="flow"`, `hf_timesteps_in_sigma=True` 라 adv 도 flow/σ 모드로 잡힌다.
- **adv 추가 수정(이식에 필요했음)** — `train_network.py:201–203` `NetworkTrainer.latent_stride = 8`, `:857–859, 946, 977` `_adv_cache_sources(…, latent_stride)`: 기준 캐시가 **디스크 캐시 latents** 의 격자 키를 `bucket_reso // 8` 로 만들어, MOSS(stride 16)에선 1024² 가 "128x128" 로 잡히고 실제 64×64 latents 와 어긋나 `build_adv_references` 가 `adv source grid hint was inconsistent` 로 거부했다 — `prepare_adv_reference` 의 except 블록 안에서 나는 ValueError 라 `--cache_latents_to_disk` + `adv_scale > 0` 은 **시작 시 죽었다**. MOSS 는 `moss_train_network.py:71` 에서 16. 다른 학습기는 기본 8 그대로 → 동작 동일.

**테스트** (`tests/test_moss_pipeline.py`)
- `:838` OT: 켜면 노이즈 = 같은 시드의 원 노이즈를 `cosine_optimal_transport` 순열로 재배열한 것, 그 순열이 4! 전수 탐색의 최소 비용과 같음, target 이 재배열된 노이즈를 씀; 끄거나 배치 1 이면 원 노이즈 그대로.
- `:875` adv: **실제 `MossLatentsCachingStrategy` 로 디스크에 쓴 npz**(48×32 px → 2×3 latents, 변형 2개)에서 기준을 만들면 키 "2x3" 으로 성공, 예전 stride-8 키 "4x6" 은 `inconsistent` 로 거부됨을 대조; 학습 스텝이 DiT 입력과 같은 noisy 를 저장하고 adv 항이 손실을 올림; 저장이 없으면(예전 MOSS) 손실 불변.

## 4. `--moss_tier_timing` (`--anima_tier_timing` 의 MOSS 판)

- 옵션: `library/moss/train_utils.py:78–87` `--moss_tier_timing`, `--moss_tier_timing_every`(기본 50), `:121–122` `validate_moss_args` 에서 켰을 때 every ≥ 1 검사.
- 훅: `moss_train_network.py:295–324`(Anima :1247–1271 과 같은 구조). 티어 = 배치의 **샘플당 이미지 토큰**, **patch 1**(MOSS latents 는 이미 2×2 패킹 — 잠재 1픽셀 = 1토큰): 캐시 latents 면 `H·W`, 없으면 픽셀 / **16**(`library/tier_timing.py:59` `sample_tokens(…, vae_downscale=)` 추가). 티어 의미는 Anima 와 같다(1024² px = 64² = 4096 토큰 = "1024" 티어).
- **텍스트 토큰 별도 표기**: `TierTimer.stop(text_tokens=)`(`tier_timing.py:114–126`)가 스팬마다 그 마이크로스텝 forward 의 텍스트 길이(`MossTransformer.last_text_tokens` = 결합 시퀀스에 실린 패딩 포함 텍스트 행 수, 이미 호스트에 있는 int)를 받아 티어별 평균을 낸다 → tracker `tier_time/<티어>/text_tokens`, 로그 줄 `1024 n=2 mean … ms median … ms text 200 tok`. `text_tokens` 없이 멈춘 스팬(Anima)은 아무것도 더하지 않아 **Anima 출력은 그대로**.
- **끄면 비용 0**: 훅 4개가 `args` 속성 확인 1회로 반환(배치·accelerator·모델을 건드리지 않음 — 테스트가 접근 시 실패하는 객체로 확인). DiT forward 의 `last_text_tokens` 는 이미 계산된 int 의 속성 대입 1회(텐서 연산·동기화 없음).
- CUDA 에선 Anima 와 같이 이벤트 쌍을 끝난 것만 읽는다(동기화 없음).

**테스트** (`tests/test_tier_timing.py`) — `:237` text_tokens 집계·스칼라·로그 줄·이벤트 쌍 보류 후 수확, `:264` patch 1 / stride 16 토큰 수, `:281` MOSS 훅(끔: 무접촉·타이머 없음 / 켬: 4 마이크로스텝, 2 스텝마다 tracker, 텍스트 토큰 12·7·20·30 → 티어별 평균, 에폭 줄 "1024 n=2 … text 16 tok", 캐시 없는 픽셀 배치 포함), `:330` 파서 기본값·검증.

## 5. TSM-LoKr grouped accumulation 을 MOSS 에서

- `moss_train_network.py:73` `_tsm_grouped_supported = True`(기본 학습기는 False 라 `attach_tsm_offload` 가 거부), `:214–224` 노이즈 추첨을 `draw()` 로 감싸 학습·비고정 타임스텝이고 계획이 grouped 이면 `tsm_plan.grouped_draw(draw)`(Anima :1470–1475 와 같음). 검증·고정 타임스텝은 그대로.
- 거부 문구·문서: `library/lycoris_algo_fixes.py:4737–4738` "anima_train_network.py and moss_train_network.py only / Anima·MOSS 학습기에서만", `docs/tsm_lokr.md:43` "(Anima and MOSS trainers only)".
- **테스트** `tests/test_moss_pipeline.py:788` — tiny DiT + TSM-LoKr(`tsm_n_scales 4,1`, resident 2, AdamW fused, 누적 2, 배치 2, uniform 샘플링) 6 옵티마이저 스텝: grouped 면 **스텝마다 4 샘플이 한 코어 구간**, 끄면 섞임. 기본 학습기 플래그(False)로는 거부, MOSS 플래그로는 붙음. 각 행의 noisy 입력이 그 행을 받아들인 추첨의 σ·노이즈와 일치.

## 6. 테스트 결과

### 6.1 이번에 추가·변경한 테스트
CPU, `CUDA_VISIBLE_DEVICES=-1`, 이 브랜치(704b11b)에서:

| 파일 | 결과 | 새 테스트 |
|---|---|---|
| `tests/test_attention_host_seqlens.py` (신규) | 12 passed | 12 |
| `tests/test_moss_models.py` | 13 passed, 1 skipped | 4 |
| `tests/test_moss_pipeline.py` | 29 passed, 1 skipped | 7 |
| `tests/test_tier_timing.py` | 12 passed | 4 |

- skip 2개는 기존 것 — 실제 Qwen3.5 `tokenizer.json` 이 없는 환경(`test_tokenizer_golden_ids`, `test_te_cache_roundtrip_with_the_real_tokenizer`).
- `test_moss_models.py`·`test_moss_pipeline.py` 는 pytest 없이 `python tests/<파일>` 로도 돈다(파일 끝의 러너): 13 PASS / 29 PASS, failed 0. 새 테스트는 그 형식을 지켜 픽스처를 쓰지 않는다.
- 공용 헬퍼 `tests/_device_reads.py`(pytest 의존 없음; `test_` 로 시작하지 않아 수집 대상 아님).

### 6.2 CI 세트 (`.github/workflows/moss-image.yml` 목록, 한 프로세스)
`.github/workflows/moss-image.yml` 의 파일 목록을 CI 처럼 한 프로세스로(`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=1`):

| 트리 | 파일 | passed | skipped | failed | 시간 |
|---|---|---|---|---|---|
| moss-image **1f7f231** (기준) | 22 | 1106 | 53 | **0** | 33:55 |
| **이 브랜치** 704b11b | 23 (+`test_attention_host_seqlens.py`) | **1133** | 53 | **0** | 35:24 |

+27 = 새 테스트. (CPU 4개를 두 실행이 나눠 쓴 시간 — 블록 컴파일 테스트의 inductor 컴파일이 대부분. GitHub 러너에선 이 워크플로가 `moss-image` 푸시/PR 에만 걸려 새 브랜치 푸시로는 돌지 않았다.)

### 6.3 비트 동일 (moss-image 대 이 브랜치)
같은 스크립트를 두 트리에서 실행(각 트리의 모듈을 import 하는지 경로 확인함)해 저장한 텐서 비교: **이 브랜치 대 moss-image 1f7f231, rebase 전 커밋 대 0938398 모두 5583개 전부 `torch.equal`**, 불일치 0.
- DiT forward 출력·입력 그래디언트·파라미터 그래디언트 4576개: 배치/split × fp32/bf16 × 길이 (5,5,5)/(5,9,2)/(4,)/(3,7) × 정렬/비정렬 × 체크포인팅 끔/켬 64 조합(skip_layers 포함).
- 학습 스텝 `get_noise_pred_and_target` 출력 5종과 파라미터 그래디언트, 모두 462개: moss / uniform / sigmoid 샘플링 × 체크포인팅 끔/켬, 기본 옵션(OT·adv·grouped·tier timing 끔).
- 어댑터를 단 학습 스텝 544개: `lora_moss`, LyCORIS LoKr(DoRA, full matrix), TSM, T-LoRA(고정 경로, 프리셋 moss) × 체크포인팅 끔/켬 — 출력 5종과 어댑터 그래디언트.
- `moss_sample` 출력 1개(어댑터 없음, 5 스텝, CFG 3, skip layer).
트리 안에서도 `tests/test_moss_pipeline.py:934`(기본 스텝을 예전 순서로 재연해 비트 동일)와 §2 의 테스트가 같은 성질을 지킨다.

### 6.4 전체 CPU 테스트 (`tests/` 전체, `--continue-on-collection-errors`)
`pytest tests/ -q --continue-on-collection-errors`, CPU 4개(같은 머신에서 두 트리를 동시에, `OMP_NUM_THREADS=2`):

| 트리 | failed | passed | skipped | errors | 시간 |
|---|---|---|---|---|---|
| moss-image **1f7f231** (기준) | 268 | 1820 | 123 | 49 | 36:59 |
| **이 브랜치** 704b11b | 268 | **1847** | 123 | 49 | 38:30 |
| (참고) moss-image 0938398 / rebase 전 커밋 | 268 / 268 | 1754 / 1781 | 123 / 123 | 49 / 49 | |

- **실패 집합(FAILED + ERROR 테스트 ID 317개)이 기준과 정확히 같다** — 1f7f231 대 이 브랜치, 0938398 대 rebase 전 커밋, 그리고 0938398 대 1f7f231 도 같은 317개. passed 의 +27 이 새 테스트 수 그대로.
- 317개는 전부 이 CPU 환경 문제로 기준에서도 실패하는 것: `LoraEasyCustomOptimizer` 미설치(수집 오류 2 파일), pytorch-wavelets 의 `pkg_resources` 없음(wavelet masking 계열), "CUDA must be available"(device 테스트), 그 밖의 업스트림 테스트(token mining antithetic, hf token loss, SNR weight 등). MOSS·어텐션·tier timing·TSM·LyCORIS 테스트는 0개이고, CI 목록 23 파일에 걸린 것도 0개.

## 7. 확인 못 한 것
- **GPU 실측 전부**: 동기화 제거의 실제 s/step 이득, CUDA 이벤트 tier timing(`tier_time/pending` 이 0 근처로 돌아오는지), TSM-LoKr grouped + 오프로드의 pinned 메모리·스트림, 5B 실모델 VRAM. 읽기 횟수는 CPU 디스패치 카운터로 센 "GPU 였다면 동기화가 났을 곳" 이다.
- **flash / xformers / sageattn 경로**: 이 환경에 없음. 호스트 길이는 xformers `BlockDiagonalMask` 리스트와 split·트림 판단에 들어가는데 실행 검증은 torch SDPA 경로만 했다.
- **adv 를 실제 MOSS 데이터셋으로**: 기준 캐시 빌드 시간·메모리, 손실 크기·품질 영향. 테스트는 작은 격자와 실제 MOSS npz 포맷으로 배선만 확인.
- **OT 의 `torch_linear_assignment`(CUDA) 경로**: 미설치라 SciPy 대체 경로만 실행.
- **실제 `moss_train_utils.sample_images`**(프롬프트 파일·VAE 디코드·저장)로 TSM 샘플링: 테스트는 같은 DiT 호출을 하는 `moss_sample` 을 래퍼로 감싸 확인했다.
- **HunyuanImage**: adv 디스크 키 — 그 VAE 의 stride 가 8 이 아니면 같은 문제가 있을 수 있으나 건드리지 않았다(기본 8 유지). 어텐션 — HunyuanImage 도 `create_attention_params_from_mask` 를 쓰므로 `text_lengths` 를 넘기면 같은 동기화 제거가 되지만 범위 밖이라 예전 경로 그대로.
- **마스크 없는 split 경로**(`attention()` 에서 `seqlens` 가 없을 때 만드는 디바이스 텐서 — Anima `--split_attn`): 호출마다 슬라이스 끝 동기화가 남아 있다. MOSS 범위 밖이고 Anima 컴파일 경로를 건드리게 되어 그대로 뒀다.
- 남아 있는 동기화(기존 설계): `lycoris_algo_fixes.set_timestep` 의 TSM `t.cpu().tolist()`(마이크로스텝당 1회), `grouped_draw` 의 추첨마다 `.tolist()`(Anima 와 같음).
- UR_IV GUI 에 `--moss_tier_timing` 노출은 안 했다.
