# anima_lora 속도 조사와 블록 단위 컴파일 적용성 검증

- 작성: 2026-10-25 (클라우드 세션, GPU 없음 — 코드·문서 읽기와 CPU 하네스 실험)
- 대상: [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) `c98b6ae`(2026-10-25) vs 우리 포크 [UR-al/sd-scripts](https://github.com/UR-al/sd-scripts) `moss-image` `3a4f196b`(2026-10-26)
- 우리 조건: Windows 11, RTX 5090 32 GB, torch 2.7.1, Python 3.11, Anima 2.9B(40블록), 4단 버킷(512·1024·1536·2048, `bucket_no_upscale`), 배치 1 × 누적 8(또는 `--step_rows 8`), GC(unsloth 오프로드; 512 단만 재계산 없음), FA2, AdamW fused, LyCORIS TSM(rank 32, 8+1 전문가) / LoKr factor 4 + DoRA
- 확인하지 못한 주장은 **미확인**으로 표시

## 0. 요약

- **anima_lora 가 빠른 이유는 거의 하나, 블록 단위 `torch.compile`** 입니다. 각 Block 의 `_forward`(체크포인트 래퍼 안쪽)를 컴파일해 LoRA 가 패치한 Linear 까지 한 그래프로 묶어, 모듈 단위 파이썬 호출·커널 런치·autograd 노드를 함께 줄입니다. 덧붙는 장치는 GC 끄기(2B·16 GB 기본값, 메모리는 `activation_memory_budget`), QKV/KV·adaLN-down 융합(어댑터 대상 블록당 10개), LLM 어댑터 출력 캐시, fused AdamW 입니다. **LoRA 전용 융합 커널은 없고**, 커스텀 autograd 도 2026-06 에 지웠습니다. OrthoLoRA 는 현재 트리에서 빠져 있습니다(`_archive`).
- **내세우는 수치의 조건은 2B(28블록)·1024 단·LoRA r32·배치 1·GC 끔**입니다(5060 Ti 1.1 s/장, 5070 Ti 0.59 s/장). 2.9B 는 16 GB 에서 GC+unsloth 로만 돌았고(32스텝 2.2 s/it), 2048 단은 지원하지 않습니다(최대 1536).
- **같은 단위로 맞추면** 5090·40블록·1024 단 일반 LoRA 는 대략 0.35–0.55 s/장(추정). 포크 TSM 의 1024·512 단은 호스트 바닥 1.1 s/장. 차이의 대부분은 호스트 오버헤드이고, 블록 컴파일이 바로 그 부분을 겨냥합니다.
- **CPU 하네스 검증 결과(§5)**: 포크의 실제 `Block` 코드와 LyCORIS 고친 경로를 torch 2.7.1 에서 anima 식으로 컴파일해 보니, 그대로는 모든 LyCORIS algo 가 어댑터 Linear 마다 끊기지만(블록당 21그래프), **포크 쪽 수정 2개**(어댑터 forward 클로저 재바인딩, `attention.py` 의 `del q, k, v` 제거)로 LoKr+DoRA·LoRA 계열 9종이 **블록당 그래프 1개·끊김 0**, **TSM·TSM-LoKr 는 수정 3개 추가**로 같은 결과가 됩니다. 수치는 eager 와 비트 동일(aot_eager), inductor 는 bf16 1 ulp 수준.
- **기대 이득(추정)**: 블록당 커널 런치 TSM 672→235(−65%), LoKr+DoRA 980→121(−88%)(GC 켠 상태), 모듈 호출 파이썬(포크 실측 ~0.5 s/마이크로스텝) 소멸. 1024·512 단 마이크로스텝 1.1 s → 0.3~0.7 s, 2048 단은 거의 변화 없음. 스텝 전체 −20~−35%(티어 분포 미확인). 첫 실행 컴파일 수 분, VRAM 변동 위험(2048 단 재측정 필요).

## 1. 범위와 방법

- anima_lora: `docs/structure/anima-optimizations.md`, `docs/optimizations/*`, `docs/findings/torch214_no_win_on_sm120.md`, `docs/findings/custom_autograd_removal_partitioner_oom.md`, `docs/methods/anima-2.9b.md`, `docs/structure/{lora,anima}.md`, `README.md`, `CLAUDE.md`, `configs/{base,presets}.toml`, `configs/methods/lora.toml`, `library/runtime/*`, `networks/attention_dispatch.py`, `networks/attn_fuse.py`, `networks/lora_modules/*`, `networks/lora_anima/*`, `library/anima/trainer/*`, `library/anima/{training,models}.py`, `library/training/{loop,optimizers}.py`, `library/datasets/{buckets,cache}.py`.
- 우리 포크: `anima_train_network.py`, `train_network.py`, `library/anima_models.py`, `library/attention.py`, `library/lycoris_algo_fixes.py`, `library/lycoris_fast.py`, `library/tsm_lokr.py`, `library/micro_batch.py`, `library/cuda_memory.py`, `library/train_util.py`, `docs/anima_train_network.md`, `docs/lycoris_fixed_paths.md`, `docs/tsm_lokr.md`.
- torch 2.7.1 의 dynamo/inductor 설정은 PyPI 휠 소스에서 직접 확인했습니다(`inline_inbuilt_nn_modules=True`, `mark_dynamic(min,max)`·`activation_memory_budget` 존재, 설정이 ContextVar 가 아님, `mix_order_reduction` 없음). flash-attn 은 v2.6.3/2.7.0/2.7.4/2.8.3 소스로 확인(2.7+ 만 custom op 등록 → 컴파일 그래프가 끊기지 않음).
- CPU 하네스(§5): torch 2.7.1·Python 3.11·LyCORIS 포크 고정 커밋 `c2cfd8ae`·포크 `library/anima_models.py` 를 그대로 import 한 2블록 토이 Anima(dim 64, adaLN-LoRA, `attn_mode=torch`). bf16 CPU autocast 로 포크의 GPU 빠른 경로가 선택되게 함. GPU·모델 로드 없음.

## 2. anima_lora 의 속도 기법과 우리 포크 상태 (3a4f196b)

| # | 기법·원리 | anima_lora 위치 | 문서가 주장하는 이득 (조건) | OS/GPU 제약 | 우리 포크 (근거) |
|---|---|---|---|---|---|
| 1 | **블록 단위 torch.compile** — `Block._forward` 를 블록마다 컴파일, LoRA 패치 Linear 포함, 어댑터 부착 뒤 컴파일 | `library/anima/models.py:1567-1680`, `library/runtime/harness.py:537-600`, `configs/base.toml:52` | 단독 수치 없음(**미확인**). README: 1.1 s/step·13.4 GB (5060 Ti, r32, 1 MP, 배치 1, GC 끔). "재컴파일 폭주로 eager 폴백 시 ~2× 회귀" | OS 무관(Windows 는 triton). torch 2.12–2.14 에서 검증 | **부분**: 원소별 체인(LN+AdaLN, q/k RMSNorm+RoPE, 잔차)과 LoKr+DoRA 재구성만 컴파일(`library/elementwise_compile.py`, `anima_models.py:204-266`, `lycoris_fast.py:94-99`). 어댑터·GEMM·어텐션은 eager. `--torch_compile` 은 accelerate 지역 컴파일이라 unsloth 래퍼(`@torch._disable_dynamo`)에 막힘(GPU 미확인) |
| 2 | **동적 seq + 가짜 5D 평탄화** — (B,T,H,W,D)→(B,1,L,1,D), seq 축만 `mark_dynamic(min,max)` 로 티어 밴드에 묶어 티어당 그래프 1개 | `models.py:100-169, 1988-1994`, `library/anima/trainer/data.py:341-367`, `library/datasets/buckets.py:84-108` | 재컴파일 폭주 방지 | 2.7.1 에 `mark_dynamic(min,max)` 있음 | **없음** |
| 3 | **activation_memory_budget** — AOT 분할기가 싼 중간값을 재계산(GC 대신) | `harness.py:433-462`, `base.toml:55`(0.99) | 0.85 로 첫 스텝 OOM 해결, 1.02 vs 1.01 s/it (5060 Ti, 4200 토큰, GC 끔) | 컴파일 필수, GC 와 함께면 무시 | **없음** (eager 의 토큰 수별 GC 해제·절반 GC 가 대신, `anima_train_network.py` `--anima_no_checkpoint_max_tokens`) |
| 4 | **GC 끔이 기본** | `base.toml:41`, `presets.toml` | GC 비용 ~18%(2026-06). 2.9B 는 16 GB 에서 low_vram 필수 | VRAM | **부분**: 512 단(≤1100 토큰)만 GC 끔(−3.8%, +0.5 GiB 실측). TSM 은 1024 단이 32 GB 초과 |
| 5 | **QKV / KV / adaLN-down 융합** — 로드 때 q·k·v→`qkv_proj`, 교차 k·v→`kv_proj`, adaLN down 3→1. 어댑터 대상 블록당 10개 | `models.py:323-329, 374-384, 1115-1121`, `library/anima/weights.py:112-160`, `networks/attn_fuse.py` | 정성적("GEMM·HBM 읽기 2–3배 감소", **미확인**) | 없음 | **없음**: q/k/v 분리, 어댑터 대상 블록당 16개 → 마이크로스텝 640 호출 |
| 6 | **LLM 어댑터 출력 캐시** — TE 캐시에 llm_adapter(6층·512토큰) 출력 저장 | `library/anima/training.py:126-130`, `library/anima/strategy.py:227-291`, `configs/methods/lora.toml:10` | 수치 없음(**미확인**) | 어댑터 미학습 시만 | **없음**: 매 forward 실행(`anima_models.py:1446-1455`) |
| 7 | **fused AdamW** | `library/training/optimizers.py:201-202` | 1.17→0.86 s/step, 상태 +180 MB (5060 Ti, static-pad 시절) | CUDA | **있음** (`optimizer_args fused=True`) |
| 8 | **FA2** (FA4 제거) | `networks/attention_dispatch.py:276-281`, `docs/optimizations/fa4.md` | FA4+compile 5:15 vs FA2+compile 5:01 /epoch (5060 Ti, r32, 배치 2) | Ampere+; flash-attn ≥2.7 이어야 컴파일 그래프 유지 | **있음** (`library/attention.py`, `--attn_mode flash`) |
| 9 | **컴파일 위생** — requires_grad 통일, 컴파일 중 RoPE dict 캐시 우회, `.item()` 없는 σ 조회, 불완전 배치 버림, 시그니처별 컴파일 캐시 분리, 한도·inductor 플래그 고정 | `models.py:1878, 551`, `library/runtime/noise.py:50`, `datasets/base.py:602`, `harness.py:356-430`, `library/runtime/dynamo.py` | 가드 안정(정성적) | ContextVar 우회·`mix_order_reduction` 끄기는 torch 2.12 대응 — 2.7.1 엔 불필요 | **부분**: RoPE 가드 있음(`anima_models.py:541, 594`). TSM 은 마이크로스텝당 `.cpu().tolist()` 1회(eager 무해) |
| 10 | **CUDA 그래프** (`--compile_inductor_mode reduce-overhead`, opt-in) | `library/config/cli_args.py:256-270` | 기본 아님, 수치 없음 | 블록 스왑 비호환 | **없음** |
| 11 | **expandable_segments** | `library/runtime/allocator.py` | 수치 없음 | **Linux 전용** | 해당 없음(Windows 빌드엔 없음, `library/cuda_memory.py:1-6`) |
| 12 | **σ-lowres** — σ 큰 스텝을 낮은 해상도 형제 latent 로 학습 | `library/anima/trainer/sigma_route.py`, `base.toml:65-72` | wall −14%(기본)~−18% (LoRA, 480스텝, GPU 미기재) | 기본 꺼짐, 1024 출발 경로만 인증 | **없음** |
| 13 | **블록 스왑 / unsloth 오프로드** | `library/runtime/offloading.py` | — | 동시 사용 불가 | **있음** (같은 계열 코드; 포크는 RNG 재생 추가) |
| 14 | **LoRA rank GEMM 을 bf16 로** | `networks/lora_modules/base.py:159-171` | fp32 경로와 비트 동일 → 대역폭 절약 | — | **부분** (TSM `tsm_lowrank_fp32` 는 stage 2 만, LoKr fast forward 는 fp32 복사 제거) |
| 15 | **해 보고 버린 것** — torch 2.14, max-autotune, combo_kernels, FA4, bnb 8bit Adam | `docs/findings/torch214_no_win_on_sm120.md` 외 | 2.14 −0.7%, max-autotune 평균 1.7× 나쁨, combo 1.5–2.4% 느림 (5070 Ti) | sm_120 엔 NVGEMM 커널 없음 | 사용자 실측(2.14 이득 없음)과 일치 |
| — | **포크에만 있는 것** (anima 에 없음) | — | — | — | 토큰 예산 마이크로배치(3effef4), **TSM 혼합 timestep 마스크드 단일 호출**(4a9c2637: 그룹 루프 11.05→10.35 s/step, −6.4%, 배치 8×누적 2+예산), **`--step_rows`**(3a4f196b: 8행 11.37 s/step = 1.42 s/장, 배치 1×누적 8 의 1.40 과 같음), 토큰 수별 GC 레짐, adaLN hoist(±0), CUDA 메모리 cap·로그 |

속도와 무관한 anima 기본 기능: T-LoRA 마스크(연산량 동일), 채널 스케일링·SVD-down 초기화·REPA(품질용), free-fit 버킷(재컴파일 방지·crop 손실 감소).

## 3. 이식 후보 순위 (TSM / LoKr+DoRA, Windows + torch 2.7.1 + 5090)

### 시간은 어디로 가나 (추정)

- 티어별 1장 FLOPs(2.9B·동결 기반·GC 기준 직접 계산): 512 ≈17, 1024 ≈86, 1536 ≈263, 2048 ≈640 TFLOP (비율 0.2 : 1 : 3.1 : 7.4).
- anima 5070 Ti 실효 ~74 TFLOPS → 5090 ~175 TFLOPS 추정 → TSM 1024 단 GPU 일 ~0.5 s, 2048 단 ~3–4 s. 사용자 프로파일("1024·512 호스트 한계, 2048 GPU 한계")과 일치.
- 평균 1.40 s/장이 바닥 1.1 s 에 가까우므로 스텝의 약 2/3 가 호스트 바닥 구간(티어 분포 미확인).

### 1순위 — 블록 단위 컴파일 (+동적 seq, 평탄화) — §5 에서 검증 완료

- 기대: 1024·512 단 1.1 → 0.4–0.7 s, 2048 단 0–5%. 스텝 −20~−35%.
- TSM·LoKr·LoRA 계열 모두 가능(§5.4). 필요한 포크 수정은 §5.3.

### 2순위 — (1 이후) GC 줄이기, LoKr 중심

- LoKr 할당 11.5 GiB → 1024 단 활성값 ~13 GB(추정)와 합쳐 들어갈 가능성(**미확인**). `--anima_no_checkpoint_max_tokens 4200` 으로 eager 에서도 시험 가능. TSM 은 상태 ~14 GB 라 여지 작음. anima: GC 비용 ~18%.

### 3순위 — LLM 어댑터 출력 캐시

- 이득 ~1%(런치 2만 중 수백), 위험 낮음, algo 무관. 어댑터 미학습 시만; 캡션 드롭아웃용 빈 캡션 출력도 캐시해야 함.

### 4순위 — σ-lowres (2048·1536 단 한정, 실험)

- 2048→1536 강등이 스텝의 ~40% 에서 걸리면 2048 장당 −20~−25%, 전체 −5~−9%(추정). 2048 출발 경로 미인증, TSM 은 고σ 전문가가 저해상도만 보게 됨.

### 5순위 — QKV/KV 융합: 사실상 일반 LoRA 전용

- LoKr: 융합 행렬의 kron 을 q/k/v 로 정확히 못 나눔(Forge 형식 깨짐). TSM: 전문가를 q/k/v 가 공유 → 의미 변경·체크포인트 비호환.

### 의미 없음

expandable_segments(Windows), FA4, torch 2.14, max-autotune, combo, free-fit, 품질용 기능, 불완전 배치 버림(배치 1), CUDA 그래프(anima 도 미사용).

| 기법 | 일반 LoRA | LoKr+DoRA | TSM |
|---|---|---|---|
| 블록 컴파일 | ✓ (anima 검증) | ✓ (§5 검증) | ✓ (§5 검증, 수정 3개 추가) |
| GC 축소 / activation budget | ✓ | ✓ (VRAM 여유) | △ (상태 ~14 GB) |
| QKV/KV 융합 | ✓ | ✗ | △ |
| LLM 어댑터 캐시 | ✓ | ✓ | ✓ |
| σ-lowres | ✓ (1024 경로) | △ | △~✗ |

부수 관찰: LoKr+DoRA 재구성 모드는 어댑터 Linear 마다 dense dW GEMM 이 하나 더 돌아 GPU 한계 구간 GEMM 일이 TSM 보다 ~1/3 많음. 토큰 예산 마이크로배치의 이득 0 은 `bucket_no_upscale` 로 버킷 대부분이 1장이라 묶을 행이 없기 때문(4a9c2637 이 TSM 의 그룹 루프 비용을 없앴지만 안전 예산 16384 에선 여전히 이득 없음을 포크가 실측).

## 4. 속도 비교 (장당 초)

환산: GPU 한계 가정, SM×클럭 비(5090/5060 Ti 4.4, /5070 Ti 2.4)와 대역폭 비(4.0, 2.0) 사이 × 블록 비 40/28. 큰 GPU 는 4k 토큰에서 활용률이 떨어지므로 실제는 범위 위쪽일 가능성.

| 출처 | GPU | 모델 | 해상도 | 어댑터 | GC | 문서 수치 | 장당 초 | 5090·40블록 환산(추정) |
|---|---|---|---|---|---|---|---|---|
| README | 5060 Ti 16GB | 2B | 1 MP | LoRA r32 + 기본 기능 | 끔 | 1.1 s/step, 13.4 GB | 1.10 | 0.36–0.39 |
| findings/custom_autograd | 5060 Ti | 2B | 4200 토큰 | LoRA | 끔 | 1.01 s/it, 15.2 GiB | 1.01 | 0.33–0.36 |
| findings/torch214 | 5070 Ti | 2B | 미기재 | LoRA 기본 | 끔 | 0.588 s/it | 0.59 | 0.35–0.42 |
| findings/paired_dw | 미기재 | 2B | 미기재 | LoRA | 끔 | 1.95 it/s | 0.51 | — |
| methods/anima-2.9b | 5070 Ti | **2.9B** | 미기재 | LoRA | **켬+unsloth** | 32스텝 4.6→2.2 s/it | 2.2(비정상상태) | ~0.9–1.1(불확실) |

포크(사용자 실측·커밋 메시지): TSM 4티어 평균 11.2 s/step = **1.40 s/장**; 같은 조건 TSM 12.64 → 1.58, LoKr 12.97 → 1.62, stock LyCORIS TSM 17.61 → 2.20 s/장; `--step_rows 8` 1.42 s/장; 마스크드 혼합 경로(배치 8×누적 2) 10.35 s/step; 1024·512 단 ≥1.1 s/장(호스트 바닥); 2048 단 GPU 한계(장당 실측 없음, FLOP 비 1024 의 ~7.4배).

조건 차이: 해상도(1024 단 하나 vs 4티어), 깊이(28 vs 40), 어댑터(LoRA r32 ≈50M vs TSM 892M·LoKr 175M, 블록당 10 vs 16 모듈), GC(끔 vs 켬+unsloth), 컴파일(블록 전체 vs 원소별 체인만), QKV 융합·LLM 캐시(anima 만), 소프트웨어(2.12–2.14·Py3.13·Linux 추정 vs 2.7.1·Py3.11·Windows), GPU(16 GB vs 32 GB).

결론: 1024 단·2.9B 에서 일반 LoRA 추정 0.35–0.55 s/장 vs TSM ≥1.1 s/장 — 2–3배 차이, 주로 호스트 오버헤드와 어댑터 복잡도. anima 의 유일한 2.9B 수치는 환산하면 우리와 비슷. "매우 빠르다"는 평판은 2B·16 GB·GC 끔 설정의 것.

## 5. 블록 단위 컴파일 적용성 검증 (CPU 하네스)

### 5.1 방법

- 토이 Anima: 포크 `Anima` 클래스(실제 생성자 인자 기준, `rope3d`, adaLN-LoRA, 2블록·dim 64). 네트워크는 트레이너와 같은 순서로 생성(`apply_module_patches` → `prepare_network_kwargs` → `lycoris.kohya.create_network(preset=anima-exp)` → `apply_to` → `after_network_created` → `set_timestep`). CPU bf16 autocast 로 `_lokr_forward`→`_fast_forward`, `_tsm_forward` fused 경로가 실제로 선택됨을 확인(블록당 16 모듈 = 실제 640 호출/40블록).
- 측정: `torch._dynamo.explain`(끊김 사유·위치), 블록별 `torch.compile(_forward, backend=aot_eager|inductor)` fwd+bwd 와 eager 의 최대 절대 차이, timestep 5개·해상도 4개에 대한 재컴파일 수, anima 식 가짜-5D + `mark_dynamic(seq)`, inductor 생성 wrapper 코드에서 런치 수 집계, unsloth GC + 컴파일 조합의 수치.

### 5.2 지금 상태에서 끊기는 이유 (원인 확정)

| 끊김 | 횟수/블록 | 위치 | 원인 |
|---|---|---|---|
| 어댑터 Linear | 16 | `lycoris_fast.py:106`, `lycoris_algo_fixes.py` `_tsm_fused_plan` 등 `self.org_module[0]` | LyCORIS 가 인스턴스 `forward` 에 묶은 bound method 의 `__self__`(LyCORIS 모듈)를 dynamo 2.7.1 이 source 없이 만들어, 파이썬 list 안의 nn.Linear 를 못 만듦("Unexpected type in sourceless builder nn.Linear"). **stock LyCORIS 도 동일**(`base.py:759`). 순수 torch 매트릭스: anima 식 `self.org_forward(x)` 호출은 모든 부착 방식에서 0 끊김, list 읽기는 인스턴스 패치에서만 끊김, 클로저/래퍼로 source 를 주면 0 |
| 어텐션 | 2 | `library/attention.py:210, 214, 226` `del q, k, v` | Python 3.11 에선 리스트 컴프리헨션이 q/k/v 를 셀 변수로 만들어 `DELETE_DEREF` 가 나오는데 dynamo 2.7.1 에 핸들러 없음. anima 의 `del` 은 일반 지역변수라 무해 |
| seq 특수화 | — | `_rope_tables` `assert cur_seq_len <= max_seq_len` | 끊긴 프레임에서 H 가 파이썬 int 로 굳음(`Eq(s0,16)`); 끊김 제거 후엔 rope 표(dim 0)도 함께 동적 표시해야 함(anima 도 그렇게 함) |

### 5.3 포크에 필요한 수정 (전부 포크 쪽, LyCORIS 설치본 무변경)

| # | 수정 | 크기 | 효과(검증) |
|---|---|---|---|
| 1 | `apply_to` 뒤 모든 어댑터 Linear 의 forward 재바인딩: `lin.forward = (lambda m: (lambda x,*a,**k: m.forward(x,*a,**k)))(lora)` | ~5줄 | 끊김 16→0 (모든 algo) |
| 2 | `library/attention.py` `del q, k, v` 3곳 제거(또는 컴프리헨션을 루프로) | 3줄 | 끊김 2→0 |
| 3 | `_tsm_scale`: compile 중엔 0-d `scalar` 버퍼를 그대로 `B * s` 에 넘김(`data_ptr()/_version` 캐시는 eager 전용) | ~8줄 | TSM 끊김 19→0 |
| 4 | TSM 전문가 선택을 `set_timestep` 에서 계산해 튜플로 저장; `_tsm_experts`/`_get_timestep` 은 compile 중 그것만 읽음; stage 2 는 `timestep_embed` 행을 스텝당 1회 만든 인덱스 텐서로 `index_select`. TSM-LoKr 는 `_timesteps`/`_expert_indices` 에 같은 처리. **4a9c2637 의 혼합 timestep 경로(`_tsm_mixed_plan` 의 파이썬 list·`torch.tensor(...)`·dict 캐시)도 compile 중엔 `set_timestep` 에서 미리 만든 plan 을 읽게 해야 함**(배치 1 에선 안 타는 경로; 미검증) | ~40줄 | 정수 timestep 마다 재컴파일(무한) → 전문가 선택당 1회(stage 1 기본 8개) |
| 5 | 블록 컴파일 배선(신규): 네트워크 적용·가중치 로드 뒤 `block._forward`(adaln hoist 시 `_forward_hoisted` 도) 컴파일, unsloth 래퍼는 eager 유지; `forward_mini_train_dit` 에서 (B,T,H,W,D)→(B,1,L,1,D) 평탄화; x 의 dim 2 와 rope 두 표의 dim 0 을 버킷 토큰 수 밴드로 `mark_dynamic`; `recompile_limit` 상향; 블록 컴파일 시 포크의 ElementwiseChains·LoKr 재구성 컴파일은 끔(중첩 컴파일) | ~80줄 | 해상도 재컴파일 0 |

### 5.4 algo 별 결과 (수정 1·2 적용, TSM 계열은 3·4 추가)

| algo | 그래프/끊김 | 수치(aot_eager vs eager) | timestep 재컴파일 | 해상도(동적) | 판정 |
|---|---|---|---|---|---|
| **lokr + DoRA**(생산) | 1 / 0 | 출력·기울기 0 | 0 | 0 | ✅ |
| lokr | 1 / 0 | 기울기 2.4e-4 | 0 | 0 | ✅ |
| **tsm stage 1**(생산) | 1 / 0 | 0 | 같은 구간 0, 전문가당 +1 | 0 | ✅ (수정 3·4) |
| tsm stage 2 | 1 / 0 | 0 | 같은 구간 0 | 0 | ✅ (수정 3·4) |
| tsm-lokr | 1 / 0 | 0 | 같은 구간 0 | 0 | ✅ (수정 4) |
| lora / locon | 1 / 0 | 기울기 1.9e-4 | 0 | 0 | ✅ |
| lora + DoRA | 1 / 0 | 0 | 0 | 0 | ✅ |
| glora, ia3, gora, ralora, lora2 | 1 / 0 | 0 | 0 | 0 | ✅ |
| ortholora | 1 / 0 | 기울기 2.4e-4 | 0 | 0 | ✅ |
| abba | 1 / 0 | 0 | 0 | 해상도마다 재컴파일 | △ 토큰 수에 따른 파이썬 경로 선택(`_hada_factored_cheaper`) 고정 필요 |
| loha (+DoRA) | 38/37 (50/49) | — | — | — | ✗ `torch.tensor(self.scale)` 를 forward 안에서 생성(미니 테스트로 확정) → 버퍼로; HadaWeight Function 추가 확인 |
| tlora | 20 / 19 | — | — | — | ✗ 마스크를 LyCORIS 전역 dict 에서 읽고 씀 → anima 처럼 스텝당 버퍼 |
| diag-oft / boft | 27 / 26 | — | — | — | ✗ `baddbmm(out=비연속 뷰)` → out= 제거 |
| full | 20 / 19 | — | — | — | ✗ `self.kw_dict \| {...}` dict 합집합 → `{**a, **b}` |
| dylora | 44 / 43, 컴파일 실패 | — | — | — | ✗ 호출마다 랜덤 rank(`_current_graph_task_id`) — 설계상 부적합 |

기울기 2e-4 차이는 aot_eager 분해의 bf16 반올림 순서 차이(값 범위 ~6e-2); DoRA 경로(손수 쓴 backward Function)는 0. unsloth GC + 컴파일: aot_eager 0, inductor(CPU) 출력 2.3e-2 @ 최대 4.3, 기울기 4.9e-4 @ 최대 5.9e-2(bf16 1 ulp 수준).

### 5.5 캐시 엔트리·컴파일 비용

- 블록들은 컴파일을 **공유**합니다(2블록 토이에서 컴파일 2회였던 유일한 이유는 블록 0 입력 bf16 vs 블록 1+ fp32 의 dtype 차이 — anima 의 "requires_grad 통일"과 같은 종류). 40블록이어도 2엔트리.
- 엔트리 수 ≈ 2(dtype) × 전문가 선택(TSM 8, LoKr 1) × 체크포인트 레짐(GC: no-grad fwd·grad fwd·bwd / 512 단 no-GC: fwd·bwd) × 밴드(union 1개 또는 티어별 4개). TSM 32~64개 → `recompile_limit` 그 이상(anima 는 2n+8).
- 첫 실행 컴파일: Windows Triton(ptxas) 기준 엔트리당 수십 초로 TSM 수 분~10여 분, LoKr 1~3분 예상(**미측정**). `TORCHINDUCTOR_CACHE_DIR` 고정으로 다음 실행은 캐시; anima 처럼 설정 시그니처별 디렉터리 분리 권장.

### 5.6 이득 추정 (GPU 미측정 — A/B 필요)

| 지표 | eager | 블록 컴파일 | 근거 |
|---|---|---|---|
| 런치/블록 fwd+bwd, TSM, GC | 672 (aten 1271) | 235 (3그래프: 82+67+86; GEMM 160) | inductor wrapper 집계 |
| 런치/블록, TSM, no-GC(512 단) | 475 | 153 | " |
| 런치/블록, LoKr+DoRA, GC | 980 | 121 | " |
| 런치/블록, LoKr+DoRA, no-GC | 736 | 77 | " |
| 모듈 호출 파이썬 | 640회/마이크로스텝 ≈ 0.5 s(포크 실측) | ≈0 (블록당 그래프 호출 2~3회) | 추적 결과 |
| autograd 노드 | 수백/블록 | 1/블록 | 구조 |

eager 런치 수는 사용자 프로파일(forward 2만 회 ≈ 500/블록)과 크기가 맞습니다.

- 1024·512 단 마이크로스텝: 호스트 바닥 1.1 s → 약 0.3~0.45 s(파이썬 0.5 s 제거 + 런치 −65%; unsloth 복사·동기화는 그대로). 1024 단은 GPU 일(~0.5 s 추정)이 드러나 **1.1 → 0.5~0.7 s**, 512 단 **1.1 → 0.3~0.45 s**.
- 1536·2048 단: GPU 한계라 −2~−10%(원소별 체인은 포크가 이미 컴파일).
- 스텝 전체: 티어 분포를 모르므로 −20~−35%(예: 11.2 → 7.5~9 s). anima 의 "eager 폴백 ~2× 회귀"와 방향 일치.
- LoKr+DoRA 는 런치 감소율이 커 호스트 쪽 이득이 TSM 보다 클 가능성.
- CPU 시간 비교는 프록시로 부적합(CPU 에선 커널 실행과 호스트가 같은 코어: LoKr 14→23 ms, TSM 22→16 ms 로 방향조차 엇갈림)해 버렸습니다.

### 5.7 위험·주의

- **VRAM**: 컴파일된 그래프의 저장 활성값 집합이 eager 와 다를 수 있음(anima 는 Function 하나 지우고 +0.8 GB). TSM 21.7 GiB 에 여유 ~10 GiB — 2048 단에서 먼저 재측정. `activation_memory_budget` 은 GC 와 함께면 못 씀.
- **수치**: inductor ~1 bf16 ulp.
- **혼합 timestep 배치(`--step_rows`, 배치>1)**: 4a9c2637 의 `_tsm_mixed_linear`/`_tsm_mixed_plan`(파이썬 list, `torch.tensor` 생성, dict 캐시)은 compile 중 끊김·sourceless 텐서 원인이 됨 → plan 을 `set_timestep` 에서 미리 만들어 읽게 해야 함(미검증). 배치 1 에선 안 타는 경로.
- **중첩 컴파일**: 블록 컴파일 시 포크의 체인·LoKr 재구성 compile 은 꺼야 함.
- 미검증: GPU 시간, 실제 크기(2048·8192 차원), CUDA inductor 융합(CPU 로 대리), Windows, RamTorch·fp8 경로(fused plan 이 split 경로로 떨어짐), 블록 스왑(미사용), `--anima_adaln_hoist`(`_forward_hoisted` 도 컴파일), 샘플 생성(eval 모드 그래프 엔트리 추가).

### 5.8 권장 순서

1. 수정 1·2 + 배선 5 를 **LoKr+DoRA** 로 먼저. 16스텝 A/B: s/step, `max_allocated`, 티어별 마이크로스텝 시간, 첫 컴파일 시간.
2. 통과하면 수정 3·4 로 **TSM stage 1**, 이어서 TSM-LoKr.
3. 512 단 no-GC·절반 GC 레짐은 그래프 엔트리만 늘 뿐 구조 동일.
4. 그 다음 후보: GC 축소(LoKr, VRAM 여유 시), LLM 어댑터 출력 캐시(~1%), σ-lowres(2048 단 한정, 실험).

## 6. 라이선스 (법률 자문 아님)

- anima_lora 도구 코드는 MIT(`LICENSE`, Copyright (c) 2026 Seunghyun Ji). kohya-ss/sd-scripts 유래 부분은 Apache-2.0 그대로(`NOTICE`, `LICENSE-APACHE`; §4(b) 변경 진술 있음). 파일 단위 경계 표시는 없음.
- 우리 포크는 Apache-2.0(`LICENSE.md`), 루트 NOTICE 없음.
- anima 고유 코드(`compile_blocks`, `_make_dynamic_seq_forward`, `isolate_compile_cache`, `cluster_token_bands`, `attn_fuse` 등)를 가져오면 저작권 줄과 MIT 허가 문구 전문을 함께 둠. sd-scripts 유래 부분(ModelOffloader, Unsloth 체크포인터 등)은 Apache §4(b)·(c)·(d) 고지 유지. 경계가 불분명하므로 파일 머리 주석 + 루트 `THIRD_PARTY_NOTICES.md`(MIT 전문·가져온 파일 목록) 둘 다 두는 것이 안전.

  ```
  # Portions adapted from sorryhyun/anima_lora (https://github.com/sorryhyun/anima_lora,
  # commit c98b6ae, library/anima/models.py), Copyright (c) 2026 Seunghyun Ji, MIT License
  # (full text: THIRD_PARTY_NOTICES.md). Modified for sd-scripts.
  ```
- 아이디어만 다시 구현(코드 복사 없음)하면 라이선스 의무 없음(출처 문서화 권장).
- Anima 계열 가중치는 CircleStone Labs 비상업 라이선스 — 어떤 트레이너로 학습해도 LoRA 는 파생물.

## 7. 확인하지 못한 것

- anima 측정 조건: OS, 0.588 s·1.95 it/s·2.9B 2.2 s/it 의 해상도, paired_dw 의 GPU, README 1.1 s 에 REPA/adaLN LoRA 포함 여부(원자료 `_archive` 는 gitignore).
- 블록 컴파일의 실제 이득: 포크·2.7.1·Windows 조합에서 GPU A/B 필요. §5.6 은 모두 추정.
- 사용자 환경: flash-attn 휠 버전(≥2.7 필요; `requirements.txt` 주석의 2.8.0.post2 면 충족), 데이터의 티어별 장수.

## 8. 부록 — 하네스 재현

하네스(세션 스크래치패드, 휘발)는 토이 Anima + LyCORIS 네트워크 생성(`toy.py`, `presets.py`), 끊김 프로브(`breaks.py`, `probe.py`), 후보 패치 참조 구현(`fixes.py`: 클로저 재바인딩, `tsm_compile_safe`, `tsm_hoist_experts`, `tsm_lokr_hoist`), 런치 집계(`launches.py`), 재컴파일 측정(`tsm_recompile.py`), GC 수치 비교(`gc_check.py`), 부착 방식 매트릭스(`patch_matrix.py`)로 구성됩니다. 이식 브랜치 `cloud/anima-speed-ports` 의 CPU 테스트가 같은 검사(블록당 그래프 1개·끊김 0·eager 와 수치 동일·해상도 재컴파일 0)를 담습니다.

## 9. 후속 (2026-10-25)

- **이식 브랜치** UR-al/sd-scripts `cloud/anima-speed-ports` (`f738479b1cdfb9bfbdfd14f702a250a6595c8a80`, 3a4f196b 기준): §3 상위 기법 중 TSM·LoKr(+DoRA) 에 통하고 RTX 5090 + torch 2.7.1 에서 이득이 기대되는 ① 블록 단위 `torch.compile`(`--anima_block_compile`, 동적 토큰 축 밴드 · 가짜 5D 블록 루프 · 컴파일 캐시 격리 · 샘플링/검증은 eager · 혼합 timestep TSM 청크는 eager)과 ② `--anima_activation_memory_budget`(체크포인팅 없는 구간용) 을 기본 꺼짐 옵션으로 넣었다. QKV 융합(LoKr 저장 형식·TSM 의미 변경), LLM 어댑터 캐시(~1%, 캐시 형식 변경), σ-저해상(TSM 외) 은 넣지 않았다. 테스트 `tests/test_anima_block_compile.py`(CI 목록에 추가): lokr+DoRA·lora·tsm 1/2단계·tsm-lokr 모두 블록당 그래프 1개·끊김 0, 컴파일 fwd+bwd 가 eager 와 동일(체크포인팅 유/무; 소 `lora` 의 adaLN up 기울기 6개만 bf16 1~2 ulp), TSM 같은 구간 timestep 재컴파일 0, 해상도 스윕 재컴파일 0. 기법별 기대 이득·GPU A/B 측정법은 `docs/anima_train_network.md` 의 `--anima_block_compile` 항목, anima_lora 유래 설계의 MIT 고지는 `THIRD_PARTY_NOTICES.md`. GPU 실측은 아직 없음(§7).
- **리뷰** 4a9c2637·3a4f196b: `reports/review_4a9c2637_3a4f196b.md`. 확정 결함 수정은 UR-al/sd-scripts `cloud/review-fixes` (`a1a8acb1d87c35b2de74ebf3d59c45dbd6d94d67`).
