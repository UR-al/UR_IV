# `--anima_tier_timing` — 티어별 마이크로스텝 시간 (UR-al/sd-scripts `cloud/tier-timing`)

- 브랜치: UR-al/sd-scripts **`cloud/tier-timing`** (`cloud/integration` 34b842b4 기준). SHA·CI: 아래 §4.
- 구현: `library/tier_timing.py`(순수 로직), `train_network.py` 의 no-op 훅 4개(`on_micro_step_start/end`, `on_optimizer_step_end`, `on_epoch_end`), `anima_train_network.py` 의 구현과 옵션 2개, `tests/test_tier_timing.py`(CI 목록 추가), `docs/anima_train_network.md` 영·일 항목.

## 1. 무엇을 재나

- **마이크로스텝** = 한 청크의 forward + backward. 청크 = 버킷 배치 1개, 또는 `--anima_micro_batch_max_tokens` / `--step_rows` 로 잘린 조각. `train_network.py` 의 청크 루프에서 `process_batch` 직전에 시작, `accelerator.backward` (+ LyCORIS after-backward 정리) 직후에 끝.
- **티어** = 그 청크의 **샘플당 DiT 토큰 수**(캐시 latents 의 `ceil(H/2)·ceil(W/2)`, 없으면 픽셀/8)에 **로그 척도로 가장 가까운** 512²=1024 / 1024²=4096 / 1536²=9216 / 2048²=16384. 경계는 기하 평균(2048·6144·12288), 동점은 아래 티어. 그래서 704×320(880 토큰)도 512 티어, 1536×640(3840)도 1024 티어.
- **GPU**: 학습 스트림에 CUDA 이벤트 쌍을 기록. 두 이벤트 사이의 GPU 타임라인 시간이라 **GPU 가 호스트를 기다린 공백도 포함**된다 — 작은 티어의 "호스트 바닥" 이 그대로 보인다. 쌍은 리스트에 쌓아 두고 **보고 시점에 `Event.query()` 가 끝났다고 하는 것만** 읽는다(`elapsed_time`). 동기화 호출 없음. 아직 안 끝난 쌍은 `tier_time/pending` 으로 보이고 다음 보고에 들어간다.
- **CPU**: `time.perf_counter`.
- **꺼짐(기본)**: 훅이 속성 확인 1번으로 반환. 이벤트 생성 없음.

## 2. 켜는 법

```
--anima_tier_timing                 # 켬
--anima_tier_timing_every 50        # 보고 간격(옵티마이저 스텝), 기본 50; 16 스텝 A/B 라면 4 또는 8
```
- tracker(tensorboard/wandb) 가 있으면 `tier_time/<티어>/count`, `tier_time/<티어>/mean_ms`, `tier_time/<티어>/median_ms`, `tier_time/pending` 스칼라가 간격마다 기록된다(**간격 창** 안의 값). tracker 가 없으면 같은 내용이 INFO 한 줄.
- 에폭 끝마다 INFO 한 줄: `tier timing (epoch N; fwd+bwd per micro-step by tokens-per-sample tier): 512 n=… mean … ms median … ms | 1024 … | 2048 … | pending k`.
- `--anima_tier_timing_every` 는 1 이상(시작 시 검사).

## 3. 블록 컴파일 A/B 에서 읽는 법

목표: `--anima_block_compile` 의 이득은 **512·1024 티어의 호스트 바닥**에서 나와야 하고 2048 티어는 거의 그대로여야 한다(reports/anima_lora_speed.md §5.6 의 추정: 512/1024 마이크로스텝 ~1.1 s → ~0.3–0.7 s).

1. A(끔)와 B(켬)을 **같은 데이터·seed·`--max_train_steps`** 로 돌리되 둘 다 `--anima_tier_timing --anima_tier_timing_every 8`(16 스텝 실행이면) 을 켠다. 타이밍 자체의 비용은 마이크로스텝당 이벤트 2개(µs 단위)라 A/B 양쪽에 같게 들어간다.
2. 각 실행에서 **첫 보고 창은 버린다** — B 의 첫 마이크로스텝들에는 컴파일 시간(수십 초~분)이 들어가 평균이 폭주한다(`median_ms` 는 덜 흔들리지만 창 안 샘플이 적으면 같이 흔들림). 두 번째 창 이후, 또는 캐시가 따뜻한 두 번째 B 실행의 값을 쓴다.
3. 비교 지표는 티어별 **`median_ms`**(이상치에 둔감). `count` 로 그 창에 티어가 몇 번 나왔는지 확인(4 티어 배치 1 × 누적 8 이면 한 창 8 스텝 = 64 마이크로스텝, 티어당 ~16).
4. 기대 패턴: 512·1024 `median_ms` 가 B 에서 크게 줄고(추정 −40~−70%), 1536 은 조금, 2048 은 ±수 %. **2048 이 늘었다면** 블록 컴파일이 GPU 바운드 구간에서 손해를 보는 것 → `--anima_block_compile_dynamic static` 과 비교하거나 보고. **512 가 안 줄었다면** 그래프 분단/재컴파일 의심 → `TORCH_LOGS=graph_breaks,recompiles` (reports/integration.md §4.3).
5. `tier_time/pending` 이 매 보고마다 커지면 이벤트가 끝나지 않는 것(스트림 정지·행) — 정상이면 0~수 개.
6. 에폭 끝 한 줄은 에폭 전체(첫 창 포함)라 A/B 비교엔 TB 창 값을, 전체 분포 확인엔 에폭 줄을 쓴다.
7. 전체 s/step(진행바)과 교차 확인: `Σ 티어별 (count × mean_ms)` ≈ 창의 스텝 수 × s/step × 1000 − 옵티마이저·로깅 시간. 크게 어긋나면 forward+backward 밖(옵티마이저, 샘플링, 로더)에 시간이 있다는 뜻.

## 4. 결과

- `cloud/tier-timing` = **`09383988b705a01f1cc611b9fb92f3759ce93602`** (34b842b4 위 1 커밋, 푸시됨).
- 테스트: `tests/test_tier_timing.py` 8개 통과(티어 매핑·배치 토큰 수·CPU 타이머·가짜 이벤트로 "끝난 쌍만 읽기"·스칼라/로그 줄·트레이너 배선·파서/검증).
- CI(`.github/workflows/moss-image.yml` 22 파일, `CUDA_VISIBLE_DEVICES=-1`, 한 프로세스): **1040 passed, 53 skipped, 0 failed** (38:23 — 다른 에이전트와 CPU 4개를 나눠 쓴 시간; GitHub 러너에선 수 분).

## 5. 미확인
- GPU 에서의 실측은 아직 없다(이 환경에 GPU 없음). 이벤트 경로는 가짜 이벤트 객체(`event_factory`)로 "끝난 쌍만 읽기" 논리를 테스트했고, 실제 `torch.cuda.Event` 의 `record/query/elapsed_time` 호출 형태만 따른다 — 첫 GPU 실행에서 `tier_time/pending` 이 0 근처로 돌아오는지 확인.
- 비용: 켠 상태에서 마이크로스텝당 CUDA 이벤트 2개 + 파이썬 호출 몇 개. 이벤트 생성 비용(µs)은 측정하지 않았다 — A/B 양쪽에 같이 켜면 비교에는 영향 없음.
