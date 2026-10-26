# `--anima_block_compile` algo 확장 — loha(+DoRA)·full·tlora·abba(+DoRA)·diag-oft·boft, dylora 거부 (UR-al/sd-scripts `cloud/block-compile-algos`)

- 브랜치: UR-al/sd-scripts **`cloud/block-compile-algos`** (`cloud/integration` 34b842b4 기준). SHA·CI: §7.
- 수정은 전부 포크 쪽(`library/lycoris_algo_fixes.py`, `library/anima_block_compile.py`, `anima_train_network.py`, 테스트·문서). **LyCORIS site-packages 는 손대지 않았다**(검증 에이전트 3명이 각자 `importlib.metadata` RECORD 의 sha256 46 파일 일치 확인).
- 진행: ultracode 워크플로 — algo 묶음 3개(a: loha·full / b: tlora·abba / c: diag-oft·boft·dylora)를 에이전트 3명이 각자 worktree 에서 고치고, 독립 검증 에이전트 3명이 반박을 시도했다. 통합하면서 검증 지적을 반영하고 통합 중 새로 찾은 문제(LoHa 경로 규칙이 토큰 축 가드가 되는 것, §3.1 6번)를 고쳐 1차 커밋, 그 커밋을 **두 번째 독립 검증 3명**이 다시 반박 — 여기서 나온 차단 결함 3건(Block 마다 따로 컴파일, 34b842b4 부터 있던 TSM stage 2 행 0, 무의미한 LoHa 테스트)과 나머지를 2차 커밋에서 고쳤다(§3.2).

## 1. 결과 요약

콜드 네트워크(트레이너처럼 첫 forward 가 첫 trace) 기준, Block 0 `_forward` 의 `torch._dynamo.explain`, torch 2.7.1 CPU, 2-Block 토이 Anima, `preset=anima-exp`.

| algo (토이 프리셋) | 이전 `cloud/integration` 34b842b4: 그래프 / 끊김 | 이후: 그래프 / 끊김 | 끊김 사유(이전) |
|---|---|---|---|
| loha (dim 8) | 38 / 37 | **1 / 0** | `Unexpected type in sourceless builder torch.Tensor` — stock `HadaWeight` 의 기본값 텐서 |
| loha + DoRA | 50 / 49 | **1 / 0** | 같음 |
| full | 20 / 19 | **1 / 0** | `Failed to trace builtin operator` — `dict.__or__` |
| tlora (dim 8) | 20 / 19 | **1 / 0** | `sourceless builder ... Parameter` — bound `org_forward` (+ 연쇄) |
| abba (dim 8) | 1 / 0 | **1 / 0** | 끊김은 없었음 — 토큰 축 가드(아래 엄격 밴드 행) |
| abba + DoRA | 1 / 0 | **1 / 0** | 같음 |
| diag-oft (dim 4) | 27 / 26 | **1 / 0** | `out= op was called where output tensor was non-contiguous` |
| boft (dim 6, 블록 경로) | trace 실패 (`TorchRuntimeError ... lshift`) | **1 / 0** | 위 out= 끊김이 `1 << lo` 를 SymInt 로 만듦 |
| dylora (dim 8, block 2) | 43 / 42 | (거부) 41 / 40 | `_current_graph_task_id`, ParameterList `slice.indices` — 뽑기마다 수가 변함 |

엄격 `mark_dynamic` 밴드 16..4096 행에 걸친 Linear 하나(경로 규칙 전환점 포함), aot_eager:

| 모듈 | 이전 | 이후 |
|---|---|---|
| loha `mlp.layer1` (256x64, 64↔256 행에서 전환) | `ConstraintViolationError` | **그래프 1개**, 출력 = 계획 행 수가 고르는 eager 경로 |
| abba `self_attn.q_proj` (64x64, 256↔1024 행에서 전환) | `ConstraintViolationError` | **그래프 1개** |
| loha+DoRA `self_attn.q_proj` (전환 없음) | 7 그래프(`HadaWeight` 끊김) | **그래프 1개** |


- **eager 대비 수치**(`compile_blocks` aot_eager, forward + backward, 두 번째 스텝 포함, 0 초기화 인자는 시드): 출력과 **모든** 네트워크 기울기가 `torch.equal` 로 eager 와 같다 — 체크포인트 없음 / unsloth / torch.utils.checkpoint 모두. 예외는 하나, 토이 전용 영역인 `boft_rows`(토큰 행이 BOFT 의 few-rows 경로로 가는 256행 미만 설정)의 `oft_blocks` 기울기가 fp32 반올림 거리(5.8e-8 / 4.2e-2)라 fp32 허용폭(rtol 1e-5, atol 1e-7)으로 고정. 운영 행 수(≥ 880 토큰)가 타는 블록 경로(`boft`)는 정확히 같다.
- **그래프 공유**: 모든 Block 이 컴파일 엔트리 하나를 공유 — 4-Block 토이도 2 그래프(Block 0 의 bf16 입력, Block 1+ 의 fp32 잔차). Anima 2.9B(28 Block) 에서 방식당 그래프 수는 Block 수와 무관.
- **해상도 재컴파일**: union 밴드(16..64 토큰) 안 8x8 → 8x12 → 12x12 → 16x16 스윕에서 새 그래프 **0**(lokr_dora·loha·loha_dora·full·tlora·abba·diag_oft·boft). LoHa/ABBA 의 경로 규칙이 뒤집히는 엄격 밴드(16..4096 행): 이전 `ConstraintViolationError`(첫 그래프 전에 실패) → 이후 **그래프 1개**.
- **dylora**: `--anima_block_compile` 과 함께면 시작 시 거부(`validate_block_compile_args`, `ValueError`). 설계상 부적합(§2.6).
- 기존 eager 동작: 비트 동일 유지(§4).

## 2. algo 별 원인과 수정

### 2.1 loha (+DoRA)
- **원인**: 끊김은 `torch.tensor(self.scale)`(gamma)가 아니라 **stock `HadaWeight` Function 의 forward 시그니처 기본값** `scale=torch.tensor(1)` 이었다. dynamo 2.7 은 Function 을 trace 할 때 이 기본값 텐서를 source 없이 감싸야 해서 `.apply` 마다 끊는다(무엇을 넘기든). 토이 LoHa Block 38 그래프 / 37 끊김, LoHa+DoRA 50 / 49.
- **수정**: `_HadaWeight`(포크) — stock 과 같은 연산·순서·dtype, `scale` 은 기본값 없는 필수 인자. `_loha_weight` 가 이것을 쓴다. 요청하신 "forward 안의 `torch.tensor(self.scale)` → 버퍼" 는 `_loha_gamma` 로: eager 에선 모듈당 0-d 텐서 하나를 만들어 두고 dtype/device 가 바뀔 때만 다시 만든다(매 호출 `torch.tensor(..., device=cuda)` 의 블로킹 H2D 복사 제거). **등록 버퍼가 아니라 `__dict__`** 에 둔 이유: state_dict(`--resume`)·저장 파일에 새 키가 생기지 않고, 이후 `.to(bf16)` 가 fp32 텐서를 캐스팅하면 float 에서 바로 만든 bf16 과 이중 반올림으로 달라질 수 있어서. 컴파일 중에는 같은 float 로 만든 **그래프 상수**(dynamo 가 `torch.tensor(float, dtype, device)` 를 op 로 trace) — 트레이너가 `compile_blocks` 뒤에 `network.to(device/dtype)` 를 하므로 캐시 텐서를 그래프에 넣으면 낡는다. 그래서 priming 도 필요 없다.
- 두 고정 경로(분해형 `(x KA^T) KB^T` = `_HadaDelta`/`_HadaDoRAOut`, 밀집 Hadamard 델타) 모두 그래프 안. 경로 선택 규칙은 §3.1 6번.

### 2.2 full
- **원인**: `_full_forward` 의 `self.kw_dict | {...}` — dynamo 2.7 이 `dict.__or__` 를 trace 못 함("Failed to trace builtin operator"), adapted Linear 마다 끊김(20 / 19).
- **수정**: `{**self.kw_dict, "weight": ..., "bias": ...}` (같은 dict; Linear 에선 kw_dict 가 비어 있음).

### 2.3 tlora
- **원인 두 가지**: (1) 랭크 마스크가 LyCORIS 의 모듈 전역 dict 에 있고 `set_timestep_mask` 가 매 스텝 **새 텐서**를 넣는다 — 그래프 안에서 읽으면 dict 가드 + 매 스텝 다시 source 되는 입력, eager 경로의 device 이동은 그래프 안에서 dict 를 쓴다. (2) 실제 끊김 19개의 대부분은 마스크가 아니라 **bound `org_forward`**: dynamo 2.7.1 이 bound `Linear.forward` 의 모듈을 source 없이 만들어 Parameter 를 읽을 때마다 끊는다(Block 당 7 + 연쇄 12, 20 그래프).
- **수정**: `set_timestep` 이 블록 컴파일일 때 마스크를 모듈마다 `_hx_tmask`(float32 (1 또는 B, rank), 모듈 device)로 **호이스트**하고, 모양이 같으면 **제자리**로 다시 쓴다(rank·device 묶음당 `_foreach_copy_` 1번). `_tlora_mask` 는 컴파일 중 이것을 읽는다 → 타임스텝이 바뀌어 활성 랭크 수가 달라져도 값만 바뀌고 그래프는 그대로(재컴파일 0). 기저 matmul 은 컴파일 중 `org_module[0]` 의 weight/bias 로 `F.linear`(`_tlora_org_forward`; nn.Linear.forward 가 하는 바로 그 호출). `clear_timestep` 은 `_hx_tmask` 를 제자리에서 1 로 채운다(마스크 없을 때 eager 값) — §3.1 4번.

### 2.4 diag-oft / boft
- **원인**: `_BlockResidualMM` 이 `baddbmm(out=<비연속 [nb, rows, S] 뷰>)` 로 `z + z D` 를 쓴다. dynamo 2.7 은 비연속 out= 을 거부("out= op was called where output tensor was non-contiguous") → diag-oft Block 당 26 끊김(27 그래프). boft 는 그 끊김이 `_boft_chunks` 의 정수를 SymInt 로 바꿔 `1 << lo` 에서 **trace 자체가 실패**.
- **수정**: 컴파일 중에만 같은 `baddbmm`(같은 피연산자·커널, 출력 stride 만 다름)을 새 연속 텐서에 쓰고 permute/reshape 로 되돌리는 `_block_residual_mm`(forward·backward). eager 는 기존 strided 쓰기 그대로(복사 없음).
- **boft 의 dim**: Anima 에서 `boft` 는 블록 컴파일과 무관하게 **`network_dim >= 6`** 이 필요하다. LyCORIS `power2factorization` 은 out_dim / b 가 2 의 거듭제곱인 짝수 b <= dim 중 가장 큰 것을 고르는데, adaLN-LoRA 의 up Linear 가 3 x model_channels 폭(토이 192, Anima 2.9B 6144)이라 그 블록 크기는 3 x 2^k(6, 12, 24…)여야 하고, 그래서 dim 6 이상이 필요하다(dim 6..11 → b = 6, 12..23 → 12; dim 4 는 생성 불가). c 에이전트 문서의 "3 의 배수" 는 틀린 설명이라 검증 지적대로 고쳤다.
- 토이의 16..64 토큰 행은 BOFT 의 few-rows 경로(rows <= `_OFT_MAX_BLOCK` 256)로 가므로, 테스트 프리셋 `boft` 는 `_OFT_MAX_BLOCK` 을 8 로 낮춰 운영(≥ 880 토큰 행)이 타는 블록 경로를 태운다. `boft_rows`(기본 256) 는 few-rows 경로 확인용.

### 2.5 abba (+DoRA)
- **원인**: 콜드 trace 는 원래 1 / 0. 문제는 `_abba_forward` 의 **토큰 수 루프라인 규칙**(`_hada_factored_cheaper`, DoRA 는 `_lf_dora_dense` 도): 컴파일 그래프 안에서 행 수는 동적 토큰 축의 심볼이라 비교가 그 축의 가드가 된다 — 엄격 `mark_dynamic` 밴드가 규칙의 전환점을 포함하면 첫 그래프 전에 `ConstraintViolationError`, soft 마크면 전환점에서 재컴파일.
- **수정**: 그래프 안에서는 융합 밀집 DoRA 형(`_abba_dora_dense`)을 건너뛰고(인덕터 융합이 그 이점을 대신함), 경로 규칙은 **계획된 행 수**로 한 번 고정한다(§3.1 6번 — b 에이전트는 "컴파일 중 항상 분해형" 으로 고쳤고, 통합에서 LoHa 와 같은 방식으로 일반화). Anima 모양에서 ABBA 는 dim 16 까지 모든 토큰 수에서 분해형이라 eager 와 같은 경로다.

### 2.6 dylora — 시작 시 거부
- **설계상 부적합**: forward 마다 **파이썬 난수로 랭크(블록 수)를 뽑아** 그 스텝의 텐서 모양이 정해지고(dylora.py:114-116; 잘해야 랭크마다 그래프), 체크포인트 재계산에 같은 랭크를 주려고 `torch._C._current_graph_task_id()` 를 읽는다 — 둘 다 그래프가 담을 수 없다. 콜드 trace 40~43 그래프(뽑기에 따라 변동), 끊김 사유에 `_current_graph_task_id` 와 ParameterList `slice.indices`.
- **거부**: `validate_block_compile_args`(`assert_extra_args` 단계, 데이터셋·네트워크 생성 전)가 `--network_module lycoris.kohya` + `algo=dylora`(대소문자·공백 무시) 또는 `--network_module networks.dylora` 이면 `ValueError`: LyCORIS 쪽은 `"--anima_block_compile: not with LyCORIS DyLoRA (algo=dylora): it draws a random rank per forward, ... Use another algo or train eager"`, kohya 의 `networks.dylora` 는 자기 사유(forward 마다 무작위 랭크를 뽑아 그 슬라이스의 `requires_grad` 를 forward 안에서 바꿈 — 그래프 끊김, 뽑힌 랭크마다 재컴파일; 모양은 안 바뀜)로 따로 거부. `--anima_block_compile` 이 꺼져 있으면 아무것도 거부하지 않는다.


## 3. 통합 중 반영·발견한 것

### 3.1 1차 검증(algo 별 패치) 반영과 통합 중 발견

1. **autocast 상태를 그래프 인자로** (a·c 에이전트가 각자 발견, `compile_blocks`): torch 2.7.1 AOTAutograd 는 **첫 호출 시점의 autocast 상태로 backward 를 trace** 한다. 트레이너는 Block 을 `accelerator.autocast()` 안에서 부르고 eager 의 autograd 엔진은 backward 를 autocast 밖에서 돌리므로, trace 된 backward 의 **모든 fp32 matmul** — Block 의 fp32 영역(adaLN 변조: forward 에서 autocast 를 끈 구간)과 어댑터 자체의 fp32 계산(LoHa 의 밀집 Hadamard 델타, BOFT 의 회전) — 이 bf16 으로 돌았다(plain LoRA 의 adaLN-up `lora_up` 6개가 1~2 ulp — 기존 테스트가 "허용폭" 으로 두던 것; LoHa `hada_w2_a`·BOFT `oft_blocks` 는 attention/MLP 층까지, max|d|/max|g| ~3e-3..8e-3; full·ABBA(+DoRA)·T-LoRA·diag-OFT 는 영향 없음. 처음엔 "adaLN 어댑터만"·"LoHa ~0.8 상대"·"ABBA+DoRA 도" 로 잘못 적었다 — §3.2 7번). 수정: 컴파일 엔트리를 autocast 끈 상태로 부르고 호출자의 상태(enabled, dtype)를 인자로 넘겨 **그래프 안에서 다시 켠다** → forward 그래프는 그대로, backward 는 저장 텐서 dtype 으로 계산. 결과: lora 포함 **모든 프리셋이 비트 동일**(boft_rows 의 fp32 반올림 하나 제외), 테스트의 bf16 허용폭 목록(`_ULP_GRADS`)은 비었다(abba_dora 의 옛 허용폭은 경로 차이였고 계획된 행 수로 사라짐).
2. **`_HadaWeight.backward` 의 autocast 가드 제거** (검증 a, 차단 사항): a 에이전트가 backward 를 `autocast(enabled=False)` 로 감쌌는데, 이것은 eager 에서 항상 no-op 이 아니다 — 엔진은 호출자의 autocast 를 물려받고, 포크의 unsloth 오프로드 체크포인터(CUDA `@custom_bwd`)는 forward 의 autocast 를 복원해 backward 를 돈다. 그 경우 stock 은 bf16 으로 계산하는데 가드가 fp32 로 바꿔 eager 결과가 달라진다. 1번 수정으로 그래프 쪽엔 가드가 필요 없으므로 stock 본문 그대로 되돌렸다(검증 a 가 가드 없이도 컴파일 결과가 정확함을 확인).
3. **`_forward_hoisted` 는 컴파일하지 않음** (검증 a, 기존 결함): `--anima_adaln_hoist` + 블록 컴파일 + 체크포인트에서 첫 호출이 `AssertionError: Attempt to trace forbidden callable maybe_mark_dynamic` 로 죽었다(`cloud/integration` 에서도 동일). 컴파일된 `_forward_hoisted` 가 컴파일된 `_forward` 래퍼(그 안의 `mark_dynamic`)를 trace 했기 때문. `_forward_hoisted` 는 `self._forward(...)` 로 넘기기만 하므로 `_forward` 만 감싸면 eager `_forward_hoisted` → 컴파일 `_forward` 로 간다. 테스트 추가(unsloth / torch 체크포인트, eager 와 비트 동일).
4. **`clear_timestep` 이 호이스트를 되돌림** (검증 b): 트레이너는 매 backward 뒤 `clear_timestep` 을 부르는데 `_hx_tmask` 는 남아 있었다 → 마스크 없이 컴파일 forward 를 돌리면(트레이너 경로엔 없음) 직전 스텝 마스크를 쓰고 eager 는 전 랭크. 이제 `_hx_tmask` 는 **제자리에서 1 로**(`_foreach_zero_` + `_foreach_add_`, 할당 없음 — 새 텐서를 만들면 매 스텝 모듈마다 복사가 생김), TSM 의 `_hx_experts`/`_hx_idx`(파이썬 튜플)는 제거(stage 2 의 `_hx_tidx` 텐서는 제자리 재기록용으로 유지; `_hx_experts` 옆에서만 읽힘). 트레이너 순서(set → forward(+backward) → clear) 를 반복해도 **재컴파일 0** 인지 tlora·tsm·tsm_stage2·tsm_lokr 로 테스트.
5. **T-LoRA 테스트 강화** (검증 b): 초기값에선 T-LoRA 델타가 어떤 마스크든 0 이라 출력 비교로는 잘못된/낡은 마스크를 못 잡는다. 인자를 기저에서 흔든 뒤(마스크 의존 델타) 4 / 6 / 7 랭크에서 forward + backward 가 eager 와 비트 동일, 새 그래프 0, `clear_timestep` 뒤 컴파일 forward == 마스크 없는 eager.
6. **LoHa/ABBA 경로 규칙의 토큰 축 가드 → 계획된 행 수** (통합 중 발견): a 에이전트는 LoHa 규칙을 그대로 두었다("토이에선 밴드 안에서 안 뒤집힘"). Anima 실제 모양에서 포크 비용 모델의 전환점(분해형 → 밀집, 행 수)을 계산하면 4-티어 밴드(880..16384 토큰) 안에서 뒤집힌다 — LoHa 는 기본값(fp32 인자를 bf16 autocast 로: `fp32_grads=True`) 기준:

   | algo, dim (rank) | 2048x2048 (attn) | 8192x2048 (mlp1) | 2048x8192 (mlp2) |
   |---|---|---|---|
   | loha 8 (64) | 전부 분해형 | 전부 분해형 | 전부 분해형 |
   | loha 8 + DoRA | **8224** | 29504 | 분해형 |
   | loha 16 (256) | **3456** | **9936** | **8112** |
   | loha 16 + DoRA | **1520** | **4992** | **7584** |
   | loha 32 (1024) | **432** | **1040** | **928** |
   | abba 16 (64) | 전부 분해형 | 전부 분해형 | 전부 분해형 |
   | abba 32 (256) | **7472** | 분해형 | 27360 |
   | abba 32 + DoRA | **3680** | **11216** | 22000 |

   (굵게 = 밴드 안 전환. `--full_bf16` 이면 LoHa 인자가 bf16 이라 loha 16 attn 6576, +DoRA 2320. ABBA 는 항상 같은 값.) soft 마크(attn_mode torch)에선 전환점마다 그래프가 늘고, 엄격 마크(flash·xformers 등)에선 **첫 그래프 전에 `ConstraintViolationError` 로 학습이 죽는다**. dynamo 는 그래프 안에서 심볼 정수를 `int` 로 보이게 해서(`isinstance(n, torch.SymInt)` 가 False — 프로브로 확인) 규칙 안에서 "이 행 수가 심볼인가" 를 알 수 없다. 그래서 `plan_block_compile_rows(network, anima, rows)` 를 추가했다: `after_network_loaded` 가 prime 직후(첫 컴파일 호출 전) 부르고, Block 의 **토큰 행 Linear** 에는 밴드의 최대 토큰 수(**샘플당**; 배치 크기는 세지 않음 — GPU 律速 티어가 더 싼 경로를 받도록)를, **cross-attn key/value(텍스트 행)·adaLN(샘플당 1 행)** 에는 0(그래프 안에서도 행 수가 정적 → eager 처럼 실제 값으로 판단)을 `_bc_rows` 로 둔다. 그래프 안에서 규칙은 그 값으로 한 번 정해지고(가드 없음), 계획이 없으면 분해형. eager·`--torch_compile` 단독은 그대로. bands 모드도 값 하나(최대 티어 기준). 비용: 작은 티어가 큰 티어 기준 경로를 타는 층이 생긴다(예: LoHa dim 16 attn 은 1024 토큰에서도 밀집 — 비용 모델로 층당 ~0.27 ms, 작은 티어 마이크로스텝당 수십 ms). 대신 2048 티어(스텝 시간 대부분)는 최적 경로.
7. **boft 문서의 dim 설명** (검증 c): "3 의 배수" → "`network_dim >= 6`"(§2.4).
8. **`boft_rows` 허용폭을 fp32 수준으로** (검증 c): bf16 허용폭(rtol 2^-6)이면 autocast 회귀(4.9e-4)도 통과해 가드가 못 됐다 → fp32 쌍(rtol 1e-5, atol 1e-7, 실측 5.8e-8).
9. 통합 실수 한 건도 기록: c 패치의 `anima_train_network.py`·`lycoris_algo_fixes.py` hunk 가 첫 적용에서 index 불일치로 빠져 있던 것을 테스트 전에 발견해 다시 적용했다(dylora 거부, `_block_residual_mm`).


### 3.2 2차 검증(통합 커밋 fc00f668 뒤) — 독립 검증 3명(경로 계획 / 상태 수명·eager 동일성 / 테스트·문서 정확성)

계획된 행 수 설계 자체는 확인됐다(Anima Block 의 모든 Linear 역할 분류가 맞음 — 2.9B·3.8B 같은 Block 클래스, 텍스트는 512 토큰으로 패딩되어 정적; 다른 행 수 의존 결정 없음; eager·`--torch_compile` 단독 불변; `_bc_rows` 가드로 스텝당 재컴파일 없음; 엄격 밴드에서 그래프 1개, eager 와 비트 동일). 대신 아래를 찾아 두 번째 커밋에서 고쳤다.

1. **[차단] Block 마다 따로 컴파일되던 문제** (a 의 autocast 수정이 만든 회귀): 컴파일 엔트리가 각 Block 의 bound `_forward` 를 클로저로 잡아, dynamo 가 그 셀을 객체 id 로 가드(`___check_obj_id(orig, ...)`) → Block 마다 캐시 엔트리. Anima 2.9B 28 Block 이면 방식당 2 → 28 그래프, 기본 `recompile_limit` 128 을 TSM stage 1(n_core 8) × unsloth 에서 넘겨 **나머지 Block 이 조용히 eager** 로 돈다. 토이(2 Block)는 원래도 Block 당 dtype 이 달라 2 그래프라 테스트가 못 봤다. 수정: 엔트리를 Block **클래스 함수**에 대해 한 번만 만들고 Block 을 인자로 넘김(`shared_entry`) → 4-Block 토이도 2 그래프(34b842b4 와 같음). 테스트 `test_blocks_share_the_compiled_graphs`(lora·loha·tsm, 4 Block → 2 그래프).
2. **[차단, 34b842b4 부터 있던 결함] 컴파일된 TSM stage 2 가 매 타임스텝 임베딩 행 0 을 읽음**: 그래프 안에선 정수 타임스텝이 상수 0(호이스트)이라, 그것으로 행을 만드는 코드가 행 0 을 골랐다 — **TSM-LoKr stage 2 기본 경로**(`_gates`)와 TSMModule 의 `tsm_expert_fn=True`(`_TSMExpertLinear` 에 정수 행). 결과: 라우터 게이트가 틀린 임베딩을 쓰고 행 0 외엔 기울기가 안 감. 기존 테스트는 stage-2 전문가·라우터·임베딩이 초기값 0 이라 못 봤다. 수정: TSM-LoKr 도 `_hx_tidx`(제자리 갱신 인덱스 텐서)를 호이스트해 `index_select`, `tsm_expert_fn` 은 그래프 안 stage 2 에서 인자 경로로. 테스트 `test_tsm_stage2_compiled_reads_the_step_embedding_row`(tsm_stage2·tsm_stage2_fn·tsm_lokr_stage2, stage-2 가중치를 흔든 뒤 260/369 — 같은 전문가 구간·다른 행 — 에서 재컴파일 없이 eager 와 비트 동일, 임베딩 기울기 행 일치).
3. **[차단] 계획 행 수 LoHa 테스트가 경로를 구별 못 함**: LoHa `hada_w2_a` 가 0 초기화라 델타가 0 → 분해형·밀집이 같은 출력. 수정: `seed_zero_factors`(ABBA `lora_up2`·LoHa `hada_w2_a`·OFT/BOFT `oft_blocks`)로 시드 + "두 강제 경로의 eager 출력이 서로 다름" 을 먼저 단언. 같은 이유로 LoHa·OFT·BOFT 의 eager 대비 비교와 강제 경로 테스트도 시드.
4. **`--gradient_checkpointing` 없으면 블록 컴파일이 아무것도 안 하던 문제**(34b842b4 부터): 트레이너가 GC 없으면 DiT 를 학습 중에도 `eval()` 로 두는데 래퍼가 `block.training` 만 봤다 → 전부 eager, 문서에 "GC 없이" 쓰라고 적힌 `--anima_activation_memory_budget` 도 효과 0. 수정: train 모드 **또는** grad 활성이면 컴파일(검증은 no_grad·eval, 샘플링은 `pause()` → 여전히 eager). 테스트 `test_eval_mode_training_forward_runs_compiled`.
5. **TSM 의 작은 결함 둘**(34b842b4 부터): (a) `clear_timestep` 뒤 컴파일 forward(트레이너의 adaptive-timestep 패스가 no_grad 로 실제로 돈다)에서 stage 2 가 직전 스텝의 `_hx_tidx` 를 읽음 → `_hx_experts` 가 있을 때만 읽게. (b) batch ≥ 2 에 행 타임스텝이 같으면 `_TSM_SAMPLE_T` 를 그래프 안에서 읽어 **타임스텝 값마다 재컴파일** → 호이스트된 스텝(단일 타임스텝 보장)에선 읽지 않음.
6. **GoRA/RaLoRA/LoRA² + DoRA** 의 `_lf_dora_dense`(CUDA 에서 행 수로 밀집/분해형 선택)도 같은 가드 문제 → 그래프 안에선 분해형 고정(이 algo 들은 테스트 대상 밖이라고 문서에 명시).
7. **문서·주석 정정**: LoHa 전환점은 fp32 인자 기본값 기준(위 표) — 처음 적은 ~6.6k/~2.4k 는 `--full_bf16` 값이었다; 옛 autocast 문제는 "adaLN 어댑터만" 이 아니라 trace 된 backward 의 **모든 fp32 matmul**(LoHa·BOFT 의 attention/MLP 어댑터 포함; full·ABBA(+DoRA)·T-LoRA·diag-OFT 는 영향 없었음 — ABBA+DoRA 의 옛 허용폭은 분해형 vs 밀집 경로 차이였다); "~0.8 상대" 수치는 재현 안 됨(max|d|/max|g| ~3e-3..8e-3)이라 삭제; CUDA unsloth 주의는 **inductor 백엔드에만** 해당(aot_eager/eager 백엔드는 복원된 autocast 아래서 backward); BOFT 블록 크기는 "b = 6" 이 아니라 3 x 2^k(6, 12, 24…); `networks.dylora` 거부 사유를 kohya DyLoRA 의 실제 동작(forward 마다 무작위 랭크 슬라이스의 requires_grad 변경)으로; 계획 값은 샘플당(배치 미포함)·첫 컴파일 전에 설정·bands 모드도 값 하나; `is_compiling()` 분기는 "블록 컴파일 전용" 이 아니라 모든 torch.compile trace; BOFT 의 256행 경계가 밴드에 걸리면 특수화 2개; 항상 참인 단언 2개 정리.
8. **커밋 메시지**: fc00f668 의 메시지에 위 부정확한 문장("Every preset is exact", "~0.8 relative", "from dim 16 on")이 남아 있다. 강제 푸시 금지라 고치지 않고 두 번째 커밋 메시지와 이 보고서에서 바로잡았다.

## 4. eager 비트 동일성

- eager 에서 바뀐 코드와 근거:
  - `_HadaWeight`: stock `HadaWeight` 와 연산·순서·dtype 동일(기본값만 없음; backward 는 stock 본문 그대로 — §3.1 2번). `_loha_gamma`: stock 이 매 호출 만드는 것과 같은 float·dtype·device 의 0-d 텐서.
  - `set_timestep`: T-LoRA 마스크를 그룹마다 `.to()` 하던 것을 한 번만(같은 값; 저장된 마스크를 제자리 변경하는 곳 없음 — 검증 b 확인). `_hoist_tlora_step`·`_hx_*` 는 블록 컴파일일 때만.
  - `_BlockResidualMM`·`_tlora_mask`·`_tlora_org_forward`·`_loha_gamma`·`_hada_factored_cheaper`·`_abba_forward`: 새 분기는 전부 `torch.compiler.is_compiling()`(규칙·ABBA 는 `BLOCK_COMPILE` 도) 안 — eager 에선 기존 코드 그대로(검증: is_compiling 은 최상위·eager autograd Function 의 forward/backward 에서 False).
  - `_full_forward`: 같은 dict.
  - `clear_timestep`: eager 상태(LyCORIS 전역)는 그대로, 블록 컴파일 호이스트만 정리.
  - `_hx_*`·`_bc_rows`·`_laf_gamma` 는 `__dict__` 항목 — state_dict·저장 파일 불변(검증 b: 224 키 그대로).
- 기존 eager 테스트(변경 없음)로 확인: CI 21 파일 안의 `tests/test_lycoris_algo_fixes.py`·`tests/test_lycoris_stock_flag.py`·`tests/test_moss_pipeline.py` 등 — §7.


## 5. 테스트

`tests/test_anima_block_compile.py` (전체 103 케이스; 1차 커밋 92) — algo 마다 콜드 네트워크 기준:
- **그래프·끊김**: `test_block_forward_traces_to_one_graph`, `test_cold_start_traces_whole_after_priming` 에 loha·loha_dora·full·tlora·abba·abba_dora·diag_oft·boft 추가(1 그래프 / 0 끊김; `COLD_BREAK` 는 lokr_dora 의 `.item()` 하나뿐 — priming 뒤 1 / 0).
- **eager 대비 수치**: `test_compiled_blocks_match_eager` 에 loha(none·unsloth)·loha_dora(none·unsloth)·full(none·unsloth)·tlora(none·unsloth)·abba(none·unsloth)·abba_dora(none·unsloth)·diag_oft(none·unsloth)·boft(none·unsloth·torch)·boft_rows(none). 0 초기화 인자(ABBA `lora_up2`, LoHa `hada_w2_a`, OFT/BOFT `oft_blocks`)를 시드해 델타와 모든 인자 기울기를 비교(`seed_zero_factors`). 출력·모든 기울기 `torch.equal`(boft_rows 의 `oft_blocks` 만 fp32 허용), 두 번째 스텝도 동일. `test_forced_path_traces_and_matches_eager`: 비용 규칙과 무관하게 분해형·밀집 양쪽 — loha·loha_dora(둘 다)·abba·abba_dora(밀집).
- **해상도 재컴파일**: `test_resolution_sweep_inside_the_union_band_shares_the_graphs` 를 lokr_dora·loha·loha_dora·full·tlora·abba·diag_oft·boft 로 매개화(새 그래프 0, 밴드 밖 4x4 는 정적 특수화 + 경고 1번). `test_planned_rows_fix_the_path_choice_under_block_compile`: 규칙 전환점을 걸친 엄격 밴드에서 그래프 1개, 계획 64 → 분해형, 4096 → 밀집, 텍스트·adaLN Linear 는 0.
- **타임스텝/상태**: `test_tlora_timesteps_share_the_graphs_through_the_hoisted_mask`(인자 흔든 뒤 5 타임스텝 재컴파일 0·출력 동일, 배치 2 마스크), `test_tlora_mask_values_reach_the_compiled_gradients`(4/6/7 랭크 forward+backward 동일, clear 뒤 = 마스크 없는 eager), `test_clear_timestep_resets_the_hoists_without_a_recompile`(tlora·tsm·tsm_stage2·tsm_lokr), `test_abba_path_choice_is_fixed_under_compile`(계획 없음 → 분해형).
- **2차 검증에서 추가**: `test_blocks_share_the_compiled_graphs`(4 Block → 2 그래프; lora·loha·tsm), `test_tsm_stage2_compiled_reads_the_step_embedding_row`(tsm_stage2·tsm_stage2_fn·tsm_lokr_stage2), `test_eval_mode_training_forward_runs_compiled`(GC 없는 eval 모드 학습 forward 가 컴파일·eager 와 동일, no_grad 검증은 eager).
- **기타**: `test_adaln_hoist_checkpointed_blocks_run_the_compiled_forward`(unsloth·torch), `test_validate_block_compile_args`(dylora 거부 3 표기, lokr·networks.lora 의 algo=dylora 는 통과), `test_dylora_is_unfit_for_block_compile_by_design`(거부 메시지 + 콜드 trace 의 `_current_graph_task_id` 끊김), `test_after_network_loaded_compiles_from_the_bucket_token_counts`(행 계획 {0, 4096}).
- 허용폭: bf16 허용 목록 `_ULP_GRADS` 는 **비었다**(lora 는 autocast 수정 전, abba_dora 는 그래프의 분해형 vs eager 의 밀집 경로 차이 때문에 필요했었다). fp32 허용 `_FP32_GRADS = {boft_rows: oft_blocks}` 만 남음.


## 6. GPU 에서 확인할 것 / 주의

- **unsloth 오프로드 체크포인트(CUDA)**: `UnslothOffloadedGradientCheckpointer.backward` 는 `@torch.amp.custom_bwd(device_type="cuda")` 라 eager 의 재계산·backward 가 forward 의 autocast 아래서 돈다 → eager-unsloth 의 fp32 backward 계산(adaLN 영역, LoHa·BOFT 어댑터 계산)은 bf16 반올림된다. **inductor** 백엔드의 컴파일 backward 는 그 autocast 의 영향을 받지 않아 체크포인트 없는 eager 와 같은 수치가 되고(A/B 손실 곡선의 bf16 수준 차이는 eager 쪽 반올림), `aot_eager`/`eager` 백엔드는 eager 처럼 autocast 아래서 돈다(AOTAutograd 는 autocast 안에서 컴파일된 그래프에만 backward 의 autocast 를 끈다). CPU 에선 `custom_bwd` 가 no-op 이라 테스트 불가(검증자가 CPU 에뮬레이션으로 aot_eager 쪽을 확인).
- **ABBA+DoRA**: CUDA 의 eager 는 `_lf_dora_dense` 가 거의 모든 층에서 융합 밀집 DoRA(`_DenseDoRAOut`)를 고르고, 컴파일 그래프는 `_hada_*` 경로(계획 행 수 기준) — **같은 수식의 다른 반올림**이라 비트 동일이 아니다(CPU 토이에서 eager-밀집 vs eager-분해형과 같은 크기: 출력 1 bf16 ulp, fp64 대비 정확도는 동등 이상, `dora_scale` 만 약간 나쁨). ABBA(DoRA 없음) dim <= 16 은 Anima 모양에서 eager 와 같은 경로.
- **LoHa dim >= 16(DoRA 면 8) / ABBA dim 32**: 경로가 밴드 최대 토큰 수(샘플당) 기준으로 고정되므로 작은 티어 일부 층이 밀집 경로를 탄다(§3.1 6번 표). 티어별 시간은 `cloud/tier-timing` 의 `--anima_tier_timing` 으로 A/B 를 읽으면 보인다(512·1024 이 기대만큼 안 줄면 이 항목 의심).
- **boft 의 256행 경계**: `_boft_rotate` 의 `rows <= _OFT_MAX_BLOCK` 은 파이썬 비교라, 밴드가 256 토큰을 걸치면(예: 256² px jitter 코너, batch 1) 경계 양쪽 두 특수화(끊김이 아니라 재컴파일 1번). Anima 티어는 ≥ 880 토큰이라 한쪽.
- **엄격 마크 + LoHa/ABBA**: 계획 없이(트레이너 밖에서) 컴파일하면 분해형으로 고정된다 — 트레이너 경로는 항상 계획을 둔다.
- 실측은 아직 없음: 이 환경엔 GPU 가 없고 테스트는 aot_eager(같은 dynamo trace·AOT joint graph, 코드 생성 없음)로 돌았다. inductor 가 융합 원소별 체인을 ~1 bf16 ulp 다르게 반올림하는 것은 기존 문서대로.


## 7. 결과

- `cloud/block-compile-algos` = **`08133a77ab73db600e53b484c268b9aadac1f48e`** (`cloud/integration` 34b842b4 위 2 커밋, 푸시됨): 1차 `fc00f668` (algo 확장 + 통합 1차 검증 반영), 2차 `08133a77` (2차 검증 반영, §3.2).
- CI(`.github/workflows/moss-image.yml` 21 파일, `CUDA_VISIBLE_DEVICES=-1`, 한 프로세스):
  - 2차 커밋: **1098 passed, 53 skipped, 0 failed** (31:56). 커밋 직전 트리에서 돌렸고, 실행 중에 주석·docstring 만 두 곳(`library/anima_block_compile.py` 의 `wrapped` 주석, `library/lycoris_algo_fixes.py` 의 `_tlora_org_forward` docstring) 고쳐 그 부분은 py_compile 로 확인.
  - 1차 커밋: 1087 passed, 53 skipped, 0 failed (33:25). 기준 `cloud/integration` 은 1032 / 53 / 0.
- 콜드 trace 표(§1)는 같은 스크립트로 `cloud/integration` 트리와 이 브랜치 트리를 각각 잰 값.
- 검증: 1차(algo 별 패치 검증 3명) → 통합 → 2차(통합 커밋 검증 3명: 경로 계획 / 상태 수명·eager 동일성 / 테스트·문서 정확성). 2차에서 나온 차단 3건과 나머지는 §3.2 대로 모두 반영. 반영하지 않은 것: bands 모드에서 밴드마다 다른 계획 행 수(속도 최적화 제안, 정확성 무관), `_boft_rotate` 의 256행 경계(문서에 적음).
