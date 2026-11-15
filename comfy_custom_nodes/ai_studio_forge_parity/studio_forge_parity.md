# 스튜디오 "Forge처럼" — Forge Neo와 ComfyUI 비교

스튜디오 제어판의 "Forge처럼" 스위치(`forge_like.master` · `noise` · `prompt_syntax` · `dora` · `attention`)가 무엇을 바꾸는지,
무엇은 같게 만들 수 없는지를 적은 조사 결과다. 아래 첫 표의 **가능/불가** 칸이 제어판 스위치를 켤 수 있게 둘지(가능),
회색으로 두고 마우스를 올리면 이유를 보일지(불가)를 정한다. `forge_like.master` 의 기본값은 1.13.0 부터 꺼짐이다.

조사 기준(2026-10-29, 읽기만 했고 GPU는 쓰지 않음):

- Forge Neo 2.31-9 (d880e8b8). 비교한 설정: Random Number Generator `CPU`(기본값),
  Eta noise seed delta `0`(기본값), Emphasis `Original`, 실행 옵션 `--use-pytorch-cross-attention`.
  샘플러 "EXP Heun 2 x0 SDE"는 `extensions/sd_forge_neo_extra_samplers`가 등록한다.
- ComfyUI v0.43.1, `--use-sage-attention`으로 실행.
- Forge 코드는 AGPL-3.0이라 팩에 옮기지 않았다. 동작만 따라 새로 썼다(`TestPackCarriesNoForgeCode`).

## 표

| 항목 | Forge Neo | ComfyUI | 가능/불가 | 방법·이유 |
|---|---|---|---|---|
| `prompt_syntax` 프롬프트 편집 `[a:b:w]` `[a::w]` `[:b:w]` `[a\|b]` | 프롬프트를 스텝 구간으로 나눠 구간마다 따로 인코딩하고, 샘플링 중 스텝 번호로 바꿔 끼운다 | 문법이 없다. 글자 그대로 인코딩된다 | 가능 | `AIStudioPromptEncode`(`forge_syntax`)가 같은 규칙으로 구간을 나누고(`studio_prompt_schedule.py`), 구간마다 한 번 인코딩해 `ConditioningSetTimestepRange`로 그 몫의 스텝에 둔다. 바꾸는 시점이 Forge와 몇 스텝 다를 수 있다(아래 "프롬프트 편집") |
| 가중치 `(x:1.2)` `((x))` `\(` | Anima 경로는 ComfyUI에서 가져온 토크나이저 코드로 읽는다 | 같은 코드 | 가능(이미 같음) | 할 일 없음. `AIStudioPromptEncode`도 손대지 않는다(아래 "가중치") |
| BREAK | Anima(Qwen3) 경로에는 BREAK 처리가 없다. 단어 "BREAK"로 인코딩된다 | 단어 "BREAK"로 인코딩된다 | 가능(이미 같음) | 할 일 없음. 지우거나 나누면 오히려 Forge와 달라진다 |
| AND | 프롬프트를 나눠 따로 인코딩하고 CFG에서 가중합한다(composable diffusion) | 단어 "AND"로 인코딩된다 | 불가 | 스위치 없음(안내만). ComfyUI의 `ConditioningCombine`은 합치는 수식이 달라 같은 결과가 나오지 않는다 |
| `noise` 초기 노이즈 | 그림마다 CPU 생성기(시드 = seed + i)에서 뽑는다 | `prepare_noise`가 배치 전체를 seed 하나의 CPU 스트림에서 뽑는다 | 가능 | 배치 1장은 원래 같다. 여러 장일 때는 `ForgeNeoKSamplerCNS` / `ForgeNeoHiresFix`의 `forge_noise`가 그림마다 seed + i로 뽑는다(`studio_forge_noise.py`) |
| `noise` SDE 단계 노이즈(EXP Heun 2 x0 SDE) | 전역 RNG에서 `torch.randn_like`. 전역 시드는 초기 노이즈를 뽑을 때 `(마지막 그림 시드 + 100000) % 65536`으로 정해진다 | 샘플러 시드로 시드한 생성기 | 가능 | `forge_noise`가 샘플러 시드를 그 값으로 넘긴다. GPU 위에서 두 앱의 뽑기가 같아진다(아래 "노이즈") |
| `noise` 그 밖의 단계 노이즈(Euler a, DPM++ SDE 등 Forge 내장 샘플러), ENSD | 그림마다 CPU 생성기에서 이어 뽑고(ENSD가 0이 아니면 seed + ENSD로 다시 시드), DPM++ SDE 계열은 그림 시드의 Brownian tree | 샘플러 시드 하나의 GPU 생성기, Brownian tree는 샘플러 시드 | 불가 | `KSampler` 경로에서는 `noise_sampler`를 샘플러 함수에 넘길 수 없다. 넘기려면 `SamplerCustom` 그래프로 바꿔야 해서 이번 버전 범위 밖이다. 사용자 샘플러(EXP Heun 2 x0 SDE)는 위 줄로 맞는다 |
| `dora` DoRA 합치기 | sam-extra "DoRA 추론 방식"을 켜 둔 LyCORIS(학습과 동일·fp32)가 txt2img 설정이다(`ui-config.json`) | 기본은 `weight_decompose`(원본 크기로 나눔) | 가능 | `forge_like.master`와 `.dora`가 켜져 있으면 `ForgeNeoDoraInferenceMode`의 모드를 `"LyCORIS fp32 (as trained)"`로 넣는다. 끄면 상태의 `dora_mode` |
| `attention` 어텐션 | `--use-pytorch-cross-attention`(PyTorch SDPA) | 실행 옵션으로 고른다 | 가능 | `AIStudioAttention`이 MODEL에 PyTorch 어텐션을 건다(아래 "어텐션") |
| i2i `inpaint_area` 인페인트 영역 | "Whole picture"(기본) / "Only masked": 마스크 둘레를 잘라 생성 크기로 키워 그리고 제자리에 붙인다 | `ForgeNeoLatentInput`은 언제나 그림 전체를 생성 크기로 다시 그린다 | 가능 | `AIStudioInpaintDetailer`가 두 영역을 모두 그린다(아래 "인페인트") |

## 프롬프트 편집

Forge 규칙(`modules/prompt_parser.py`의 동작, 테스트 `tests/test_studio_prompt_schedule.py`):

- `[a:b:w]`는 `w` 스텝까지 `a`, 그 뒤 `b`. `[a::w]`는 `w` 뒤에 `a`를 지우고, `[b:w]`·`[:b:w]`는 `w` 뒤에 `b`를 넣는다.
- `w`에 점이 있으면 스텝 비율(`0.5`, `.25`, `1.5` = 150% → 끝까지 `a`), 점이 없으면 스텝 수다. `steps`로 자르고,
  1보다 작으면 처음부터 두 번째 것이다. 브리프의 "w<1 비율, w≥1 스텝 수"와는 `1.5`·`2.0` 같은 값에서 다르며 Forge를 따랐다.
- `[a|b|...]`는 스텝마다 번갈아 쓰고, 빈 칸은 빈 글자다. 서로 겹쳐 쓸 수 있고 `( )` `[ ]` 강조 안에서도 된다(강조 괄호는
  글자로 남고 가중치는 토크나이저가 읽는다).
- `\[` 같은 이스케이프는 백슬래시째 글자로 남는다(Forge도 남긴다. 그 뒤 두 앱의 토크나이저가 똑같이 읽는다).
- 최상위의 짝 없는 괄호·콜론은 글자다. 교대 밖의 `|`나 맨 끝의 백슬래시 하나가 있으면 Forge는 프롬프트 전체를
  읽지 못하고 편집 없이 그대로 쓴다. 우리도 그렇게 한다.

Forge의 파서(lark Earley)와 다른 점: `[a:b:2]::`·`(x [a:b:2]))`처럼 편집 묶음 바로 뒤에 짝 없는 괄호·콜론이 두 개 넘게
붙은 경우, Forge는 파서 내부의 모호성 해소 순서에 따라 그 묶음을 글자로 두기도 한다. 우리는 언제나 편집으로 읽는다.
Forge 파서를 직접 돌린 무작위 비교(괄호를 많이 섞은 프롬프트 8만 개)에서 22개가 달랐고 모두 이 경우였다. 괄호 짝이 맞는
보통 프롬프트(`((x [a:b:0.5]:1.2))`, `[red|blue] hair, [cat:dog:0.4]` 등)는 모두 같았다.

바꾸는 시점 — ComfyUI 쪽 한계:

1. `ConditioningSetTimestepRange`의 범위는 스텝 번호가 아니라 **모델 시간 비율**이다. ComfyUI는 비율을 그 모델의
   sigma로 바꿔(`percent_to_sigma`, shift 3이면 `3t/(1+2t)`, `t = 1 - 비율`) 현재 sigma와 비교한다. 스케줄러의 sigma가
   그 곡선을 따르지 않으면 바뀌는 스텝이 달라진다. 기본값(Flow Cosmos Dynamic, 32스텝, shift 3)에서 첫 프롬프트를 쓰는 스텝
   수는 비율 0.25 / 0.5 / 0.75에 대해 **우리 12 / 17 / 20**, Forge(1차 샘플러) 9 / 17 / 25다.
2. Forge는 구간 `(끝, 글자)`를 0부터 센 호출 번호가 `끝` 이하인 동안 쓴다. 그래서 첫 구간이 한 스텝 더 길다(`[a:b:10]`,
   20스텝 → `a` 11스텝). 우리는 비율을 `끝 / steps`로 둔다(브리프: 0–0.5 / 0.5–1).
3. Forge는 스텝이 아니라 **디노이저 호출**을 센다. "EXP Heun 2 x0 SDE"는 스텝마다 모델을 두 번 부르지만 확장이
   `second_order`로 등록하지 않아, 32스텝에서 구간표는 32호출분인데 실제 호출은 63번이다. `[cat:dog:0.5]`는 17번째
   호출(약 8.5스텝)에서 `dog`로 바뀌고, 32번째 호출을 넘으면 Forge가 첫 구간으로 돌아가 남은 절반은 다시 `cat`이다.
   `[a|b]`도 호출마다 번갈아 한 스텝 안에서 바뀐다. 이 동작은 sigma 범위로는 만들 수 없고 의도된 규칙도 아니어서 따르지
   않았다. 이 샘플러로 Forge와 나란히 비교할 때 편집 문법이 든 프롬프트는 결과가 다르다.
4. Hires: Forge는 Hires 패스에서 비율 편집을 이미 끝난 것으로 보고 두 번째 것을 쓴다(정수는 `w - 기본 스텝`). ComfyUI의
   Hires 샘플러는 같은 조건을 받아 낮은 sigma 구간에서 시작하므로 비율이 약 0.6 이하면 두 번째 것을 써 같다. 그보다 늦게
   바뀌는 편집은 Hires 앞부분에서 첫 번째 것이 쓰인다.

나아질 수 있는 방법(이번 버전 아님): 조건에 스텝 비율을 표시해 두고 샘플러가 실제 sigma 목록으로 범위를 다시 계산하면
1번은 Forge와 스텝 단위로 맞출 수 있다.

`AIStudioPromptEncode`의 글 칸은 `dynamicPrompts`를 끄고 있어 ComfyUI 프런트엔드는 `{a|b}`를 바꾸지 않는다. 와일드카드는
생성을 시작할 때 스튜디오가 푼다(아래 "와일드카드").

## 가중치

Forge Neo의 Anima 엔진(`backend/text_processing/anima_engine.py`)은 ComfyUI v0.36.0의 `comfy/text_encoders/anima.py`를 옮긴 것이고,
가중치는 ComfyUI에서 가져온 `token_weights`(`backend/text_processing/_comfy.py`)로 읽는다. ComfyUI v0.43.1에서 확인한 것:

- `AnimaTokenizer.tokenize_with_weights`가 같은 `sd1_clip.token_weights` / `escape_important`(`\(` `\)`만 이스케이프)를 쓴다.
- Qwen3 쪽 가중치는 1.0으로 두고, T5 쪽 가중치(`t5xxl_weights`)를 LLM 어댑터 출력에 곱한 뒤 512토큰까지 0으로 채운다
  (`comfy/ldm/anima/model.py` `preprocess_text_embeds`) — Forge의 `_preprocess`와 같다.
- `[x]`는 두 앱 모두 강조가 아니라 글자다(A1111식 `[ ]` 약화는 Anima 경로에 없다).

Emphasis가 `None`/`Ignore`이면 Forge는 가중치를 끄거나 1로 두는데, 사용자 설정은 `Original`이라 같다.

## 노이즈

Forge Neo(`modules/rng.py`, `modules/processing.py`, `modules/sd_samplers_common.py`의 동작):

- 그림 `i`의 시드는 `seed + i`. 초기 노이즈는 그 시드만으로 시드한 CPU 생성기에서 `(채널, 높이/8, 너비/8)`을 뽑는다.
  이때 `torch.manual_seed((그림 시드 + 100000) % 65536)`로 전역 RNG(모든 장치)도 시드한다. 마지막 그림이 남는다.
- Forge 내장 k-diffusion 샘플러의 단계 노이즈는 `TorchHijack`이 `ImageRNG.next()`로 돌려, 그림마다 CPU 생성기에서
  이어 뽑는다(ENSD가 0이 아니면 seed + ENSD로 다시 시드). 사용자 샘플러는 확장의 `default_noise_sampler`가 확장 자신의
  `torch.randn_like`를 불러 이 우회를 거치지 않고 **전역 RNG**에서 뽑는다. 그래서 ENSD는 이 샘플러에 영향이 없다.

ComfyUI(`comfy/sample.py` `prepare_noise`, `comfy/k_diffusion/sampling.py` `default_noise_sampler`):

- 초기 노이즈: `torch.manual_seed(seed)` 생성기로 배치 `(B, 16, 1, 높이/8, 너비/8)`을 한 번에 CPU float32로 뽑는다.
  원소 수와 순서가 Forge의 한 장과 같아 **배치 1장은 원래 같다**. 둘째 장부터는 같은 스트림을 이어 뽑아 Forge(seed + 1 …)와 다르다.
- 단계 노이즈: 샘플러 시드로 시드한 `torch.Generator(x.device)`에서 `torch.randn(x.size())`(CPU 라텐트면 시드 + 1).

`forge_noise`(꺼짐이 기본, `ForgeNeoKSamplerCNS`·`ForgeNeoHiresFix`의 선택 입력)를 켜면:

1. 초기 노이즈를 그림마다 `seed + i`(라텐트에 `batch_index`가 있으면 `seed + 그 번호`)의 CPU 생성기로 뽑는다.
2. 샘플러 시드를 `(마지막 그림 시드 + 100000) % 65536`으로 넘긴다. GPU 라텐트에서 ComfyUI 생성기의 첫 상태가 Forge가 남긴
   전역 상태와 같아, 스텝마다 같은 순서로 같은 크기를 뽑는 샘플러면 단계 노이즈가 같다.

CPU 비교(`tests/test_studio_forge_noise.py`, 64×64 라텐트): 한 장의 초기 노이즈는 두 앱이 같고, 배치 둘째 장은 옵션을 켜야
Forge와 같으며, 전역 RNG를 Forge 값으로 시드한 `randn_like` 두 번과 그 값으로 시드한 생성기의 두 번이 같다.

같다고 말할 수 있는 조건: t2i·Hires(Forge도 Hires 노이즈를 같은 시드로 다시 뽑는다), 같은 GPU, float32 라텐트, 단계 노이즈를
위 방식으로 뽑는 샘플러(EXP Heun 2 x0 SDE, SEEDS·SA Solver 계열). 다른 경우:

- 인페인트: Forge는 마스크 섞기에서 호출마다 `torch.randn_like`로 전역 RNG를 써서 단계 노이즈 순서가 달라진다.
- SPEED를 켜면 샘플러가 시드를 직접 받아 이 옵션이 적용되지 않는다. CNS는 단계 노이즈에 색을 입혀 Forge의 CNS와 같은 경우만 같다.
- 시드 하나가 65536가지 단계 노이즈 중 하나로 접히는 것은 Forge의 성질이다(그대로 따랐다).
- 실제 GPU에서 Forge와 같은 그림이 나오는지는 Task 14의 나란히 비교에서 확인한다(이번 작업은 GPU를 쓰지 않았다).

## DoRA

sam-extra "DoRA 추론 방식"의 코드 기본값은 꺼짐이지만, 이 PC의 Forge는 `ui-config.json`으로 txt2img에서 켜 두고 LyCORIS(학습과 동일·fp32)를
쓴다. 그래서 "Forge처럼 DoRA"는 `"LyCORIS fp32 (as trained)"`를 넣는다.

`"Forge/Comfy (stock)"`은 쓰지 않는다. ComfyUI의 `weight_decompose`는 출력 축에서 합친 가중치가 아니라 원본 `||W0||`로 나눈다. LoRA가 거의 0에서
키운 행은 그만큼 과하게 커진다. `wd_on_output` LoKr 파일 하나(640층)를 CPU에서 재 보니 `adaln_modulation_cross_attn` 행이 최대 71배, adaLN 층의
오차 중앙값이 LoRA 변화량의 20~27%였다(어텐션·MLP는 2~3%). 사용자 비교(2026-10-30)에서 stock은 그림에 입자 노이즈가 끼고 LyCORIS는 깨끗했다.

## 어텐션

ComfyUI는 시작할 때 어텐션 함수를 하나 고른다(`comfy/ldm/modules/attention.py`의 `optimized_attention`; Sage > Flash >
xformers > PyTorch 순, 실행 옵션으로 정함). 위 비교의 ComfyUI는 Sage, Forge는 PyTorch SDPA다.

- 1.15.0부터 제어판의 어텐션 칸이 실행 중에 고른다. ComfyUI v0.43.1의 `ModelPatcher.set_model_optimized_attention`을 쓰며, Anima·
  MOSS·Krea2의 어텐션이 모두 그 경로를 지난다. 다시 켤 필요가 없고 X/Y/Z 축으로도 비교한다.
- "Forge처럼" 어텐션은 PyTorch다. 켜면 어텐션 칸이 잠긴다. `parameters`에는 고른 경우만 `Attention`을 적는다.
- 고르지 않으면(`launch`) 노드가 MODEL을 그대로 넘겨 1.14.0과 같다.

## 모델 계열과 외부 팩 (1.11.0)

- 계열은 체크포인트 이름으로 정한다: `moss` → MOSS-Image, `krea` → Krea2, 나머지 → Anima. Forge 대응이 없는 항목이다.
- MOSS fp32: ComfyUI-MOSS-Toolkit `MOSSStreamFP32`. 코어 로더로 읽은 모델에서만 된다.
- MOSS 레이어 스킵(1.20.0): 아래 "스텝 줄 스위치". 같은 노드(`AIStudioMossStream`)가 맡고 외부 팩은 필요 없다.
- Anima 레퍼런스: SnapRef 또는 ComfyUI-Anima-Ref-Adapter. 가중치는 Anima base(28블록) 기준이고, 2.9B·3.8B 는 일반 LoRA 와
  같은 블록 계보 표로 변환한 사본을 쓴다(실측 확인). 변환 없이 쓰면 SnapRef 는 참조 그림을 베끼고 Ref Adapter 는 노이즈가
  나온다. SnapRef 는 PAG 와 같이 못 쓴다. v2a 프리뷰는 v1a 와 같은 방식으로 쓰며 캐릭터만 모드가 나아졌고, 둘째 그림과 충실도는
  v2a 와 함께 나온 SnapRef 노드가 필요하다. 충실도는 SnapRef 가이던스를 끈 때만 듣는다.
- SnapRef v2a 정식판(1.18.0): 노드가 28블록 파일을 2.9B 40블록에 직접 맞추므로(끼운 12블록은 비움) 2.9B 에선 원본 파일을
  그대로 넘긴다. 3.8B 는 노드가 못 맞춰서 `__52b` 사본을 그대로 쓰고, Ref Adapter 는 두 사본 모두 그대로 쓴다. 셋째 그림
  (`reference_image_3`)과 여러 장 배치(`multi_layout`: 따로 / 나란히 붙이기)가 생겼다. 나란히 붙이면 한 장처럼 읽혀 그림별
  비중은 안 쓴다.
- Krea2 레퍼런스: comfyui-krea2edit 노드 + 편집 LoRA(identity, general, QuadView 등).

## 결과 쪽 도구 (1.12.0)

- 갤러리 자동 비우기: Forge 갤러리는 새 결과로 바뀐다. 켜면 같게, 끄면 앞에 쌓인다(ComfyUI 기본). Forge 처럼 생성이 끝나 이미지가
  나올 때 바뀌므로 정지·오류에는 지난 이미지가 남고, 업스케일·Hires·SAM3 버튼은 다른 이미지를 남긴다. 파일은 지우지 않는다.
- 걸린 시간: Forge 의 Time taken 대응. 생성을 누른 때부터 마지막 이미지까지, X/Y/Z 는 전체 시간이다. 대기열에서 기다린 시간도 든다.
- EXIF 패널: Forge 의 PNG Info 대응. 갤러리에서 고른 이미지를 따라가고, 파일을 끌어다 놓거나 열어도 된다. PNG 는 text chunk,
  JPEG·WebP 는 EXIF(Forge 는 UserComment 에 parameters 를 넣는다)를 읽는다. 파일은 올리지 않고 브라우저에서 읽는다.

## 후처리 세부 설정 (1.13.0)

Hires, 업스케일, SAM3 의 Forge 설정을 패널에 넣었다. 표는 `studio_post_schema.py` 하나다. Forge 와 다른 점:

- SAM3 스텝: 적은 수만큼 그대로 돈다. Forge 는 스텝 × 디노이즈를 돈다(`img2img_fix_steps` 꺼짐, 기본값). 32스텝·디노이즈 0.4 면
  Forge 는 12스텝, 스튜디오는 32스텝이다.
- SAM3 마스크 흐림: Forge 와 폭이 다르다(`mask_ops.gaussian_blur` 가 약 3분의 1). 같은 숫자에서 가장자리가 더 좁다.
- SAM3 시드: -1 은 메인 시드다. Forge 처럼 따로 랜덤을 뽑는 값은 없다. 디테일러에는 `forge_noise` 가 없어 같은 시드라도 노이즈가 다르다.
- SAM3·Hires 프롬프트: 일반 `CLIPTextEncode` 로 들어간다. `<lora:>` 와 `[a:b:0.5]` 는 읽지 않고, 3.8B 에서는 v2 커넥터를 거치지 않는다
  (스튜디오가 생성 때 알려 준다). 비워 두면 메인 프롬프트를 그대로 쓴다.
- Nearest: 이름대로 최근접으로 키운다. Forge 는 `Upscaler.scale` 이 1로 남아 실제로는 Lanczos 다.
- ControlNet: 이 PC 의 LLLite 파일 셋은 base(28블록)로 학습한 것이라 2.9B·3.8B 에서는 블록 수 오류로 멈춘다. Forge 도 같다.
- Mask only: Refine·타일도 돌지 않는다. 마스크만 만들어 저장하고 그림은 그대로다.
- 타일 기본값: 512 / 32(ComfyUI 값). 이 PC 의 Forge 는 256 / 16 이다. 이음매 처리가 달라 숫자를 맞춰도 픽셀은 같지 않다.
- 크기 단위: 기본값 64(Forge 기본값이자 이 PC 값). 832×1216 ×1.5 는 Forge 처럼 1280×1856 이 된다.

넣지 않은 것과 이유:

- Hires 체크포인트·TE 바꾸기: 새로 읽은 모델에는 LoRA, NegPiP, DoRA, 가이던스가 없어 조용히 다른 그림이 된다. Forge 도 기본으로 숨긴 항목이다.
- 얼굴 복원: 이 ComfyUI 에 CodeFormer·GFPGAN 노드가 없고, 노드의 대체 경로는 Impact Subpack 이 있어야 한다.
- Extra noise: 샘플링 안쪽에 넣어야 하고, Forge 설정도 꺼짐(0)이다.
- ControlNet 의 Resize Mode, Override External: 이 그래프에서는 효과가 없다(`control_image` 를 잇지 않고, 앞단 ControlNet 도 없다).
- SAM3 Refine 의 수동 마스크와 메인 프롬프트 물려받기, Anima Tile-Repair 패널, Extras 의 일괄 처리, 저장 폴더: 후처리 카드의 설정이 아니라 별도 기능이다.

## 인페인트 (1.15.0)

인페인트는 영역과 상관없이 `AIStudioInpaintDetailer`가 그리고 메인 샘플러는 돌지 않는다. 마스크 내용의 기본은 Forge처럼 원본이라 디노이즈가 1보다 작아도 원본에서 시작하고,
마스크 바깥 픽셀은 그대로 남는다. 1.14.0까지의 전체 인페인트는 `VAEEncodeForInpaint`가 마스크 칸을 회색으로 지우고 시작했다.

- 마스크만: Forge의 Only masked. `ForgeNeoSAM3Detailer`의 규칙대로 마스크 경계에 패딩을 더하고(`mask_bounds`), 생성 크기의 비율로 넓히고(`expand_crop_region`), 잘라서 그 크기로
  그린 뒤 원본에 붙인다. 그림 크기는 원본 그대로이고 `Size`는 Forge처럼 생성 크기다.
- 전체: Forge처럼 그림을 먼저 생성 크기로 바꾸고 그린다. 결과도 생성 크기다.

Forge와 다른 점:

- 스텝: 적은 수만큼 그대로 돈다. Forge는 스텝 × 디노이즈를 돈다(메인 i2i와 같다).
- 배치: 마스크 하나로 여러 장을 한 번에 그린다. 시드는 하나고, Forge처럼 시드 + i가 아니다(디테일러에 `forge_noise`가 없다). 저장 시드도 하나다.
- 마스크 흐림은 Forge처럼 시그마다. SAM3 마스크 흐림과 폭이 다르다. 생성 크기가 아니라 원본 크기에서 흐린다.
- 메인 패스에만 있는 것은 돌지 않는다: VAE 2x, PiD, CNS, SPEED(인페인트에는 원래 없다). Detail Daemon은 메인 패스 것을 쓴다. 전체 인페인트의 노이즈도 이제 Forge처럼 만들 수 없지만, Forge의
  인페인트 단계 노이즈는 원래 순서가 달라 같은 그림이 나오지 않았다.

## XYZ 확장기능 축 (1.15.0)

Forge에는 없다. 확장 표(`ext_schema.json`)의 설정마다 축이 하나 있고, 축 목록에서 확장별로 묶인다. 프롬프트, 경로, 파일 목록은 축이 없다.
값을 넣으면 그 설정이 듣는 데 필요한 스위치도 그 칸에서 켠다(`xyz_axes._SECTION_SWITCHES`). 스위치 자체의 축은 칸마다 켜고 끄며, 다른 축이 덮어쓰지 않는다.
글 설정(블록 목록 등)은 쉼표가 값의 일부라 한 줄에 한 값이다. 범위 밖 값은 패널처럼 잘린다. 고정 목록에 없는 값이나 이 모델 계열이 못 쓰는 확장은 격자를 만들지
않는다. 칸들이 같은 프롬프트로 만들어지는 축(스위치 꺼짐, 파일 미선택, 탭에서 안 도는 확장, 옛 워크플로)은 생성할 때 알려 준다.

## 설정 가져오기 (1.16.0)

EXIF 패널의 버튼. Forge의 PNG Info → Send to 에 해당하지만 스튜디오 이미지는 글이 아니라 `workflow` 안의 제어판·XYZ 상태를 그대로 되돌린다.
다른 워크플로에서 만든 그림이나 Forge 그림은 `parameters`만 읽는다: 프롬프트, 샘플링, 크기, Hires, 인페인트, 어텐션.
Forge와 다른 점: 시드는 가져오지 않는다(스튜디오 이미지는 시드 칸에 적혀 있던 값 그대로). 필요하면 ♻.

## 와일드카드 (1.15.0)

앱의 문법이다. `__이름__`·`~/이름/~`는 `이름.txt`를 읽고, 한 줄이 그룹이며 그룹마다 쉼표로 나뉜 항목 하나를 고른다. `__이름:2__`는 그룹 2개만 쓴다. 고른 항목의
`[a|b]`와 다른 와일드카드도 푼다. 여기에 Dynamic Prompts의 `{a|b}`와 `{2::a|b}`를 더했다. 프롬프트에 직접 쓴 `[a|b]`는 앱과 달리 그대로 둔다. Forge의 프롬프트 편집이다.

- 파일: `<ComfyUI>/wildcards`와 `extra_model_paths.yaml`의 `wildcards:` 항목. 이름은 경로가 될 수 없다.
- 시드: 그림의 시드로 칸마다 따로 뽑는다. 같은 시드는 같은 프롬프트다. X/Y/Z는 칸의 시드로 뽑아서 같은 시드끼리 같은 선택을 비교한다.
- 이름: 글자, 숫자, 공백, `.`, `:`, `-`, `_`만 이름으로 읽는다(앱보다 좁다). `o__o ^__^`나 `<lora:a__b__c:1>`는 와일드카드가 아니다.
- 장수: 와일드카드가 있으면 그림마다 따로 뽑으려고 한 장씩 시드 + i로 돈다.
- 저장: 풀린 프롬프트가 들어가고, 쓴 글은 Dynamic Prompts처럼 `Template`, `Negative Template`에 남는다.
- 대상: 메인 프롬프트와 네거티브, 켜져 있는 Hires·SAM3·Refine·타일 자체 프롬프트. 갤러리 Hires·SAM3와 같은 시드도 그 그림의 시드로 뽑는다. 파일이 없으면 글자
  그대로 두고 알려 준다.

## 프롬프트 칸 (1.14.0)

Forge 에는 없고 앱의 프롬프트 섹션과 같다. 선행 고정, 인물 수, 캐릭터, 작품, 작가, 프롬프트, 후행 고정 순으로 쉼표로 잇고, 빈 칸은 건너뛰며, 이어지는
끝의 쉼표는 지운다. 순서는 사용자가 정한 앱 설정과 같고 패널에서 바꾸지는 않는다. 프롬프트 칸만 쓰면 글자 하나 바뀌지 않는다. 메인 프롬프트만 대상이고
Hires·SAM3 의 자체 프롬프트에는 붙지 않는다. PNG `parameters` 에는 합친 글이 저장된다. XYZ 의 Prompt S/R, Prompt order, LoRA 축은 합친 글에서 찾는다.
선행 고정·후행 고정은 스텝 줄의 `선행`·`후행` 으로 끌 수 있다(1.20.0, 아래 "스텝 줄 스위치").

## XYZ LoRA 축 (1.14.0)

Forge 에는 없는 축이다. Forge 에서는 Prompt S/R 로 `<lora:...>` 글자를 바꾼다. 이 축은 같은 일을 드롭다운으로 하고 에폭 범위를 펼친다.

- LoRA: 기준 줄의 태그를 정확히 찾아 바꾼다. 프롬프트의 다른 LoRA 는 그대로다.
- 같은 LoRA 두 번: 칸이 열의 파일을 기준 LoRA 보다 더 많이 부르면 생성하지 않는다. 이름이 같으면 마지막 강도만 쓰이고, 표기가 다르면 파일이 두 번 올라가서 제목과 다른 그림이 된다. 다른 축이 넣은 태그도 센다.
- 찾을 말: 긍정 프롬프트에 있어야 한다. Prompt S/R 처럼 프롬프트와 네거티브를 다 바꾸지만 `<lora:...>` 태그 안은 건드리지 않는다. 트리거가 파일 이름의 앞부분인 경우가 많기 때문이다.
- 지우기(`-`): 찾을 말만 있는 자리는 쉼표까지 지운다(`(말:1.2)` 포함, `(말, 다른 말:1.1)` 은 묶음이 남는다). 긴 항목의 일부면 그 글자만 지우고 앞뒤 항목은 붙이지 않는다.
- 뒤에 넣기: 긍정 프롬프트의 첫 위치, 그 쉼표 항목의 끝에 넣는다. 항목 끝의 `<lora:...>` 는 뒤로 남는다. 찾을 말을 비우면 프롬프트 맨 앞에 넣는다.
- 트리거에 `<` 나 `:` 가 있어 프롬프트의 태그가 바뀌면 생성하지 않는다.
- 에폭: 번호 없는 파일이 마지막이다. 번호는 파일 헤더의 `ss_epoch` 를 읽고, 없으면 가장 큰 번호에 저장 간격을 더한 값으로 친다.
- 순서: 비용 0.5 라 스텝·CFG 보다 바깥에서 돈다. LoRA 를 바꾸면 모델을 다시 패치하기 때문이다.

## 프롬프트 이행 (1.17.0)

Forge 에는 없다. LoRA 를 걸었을 때 프롬프트 지시가 덜 지켜지는 것을 줄이려는 네 가지다. 모두 기본 꺼짐이고, 켜면 `parameters` 에
남고(`LoRA base first`, `LoRA cross-attn`, `CFG schedule`, `Word emphasis`), X/Y/Z 축으로 켠 것과 끈 것을 나란히 볼 수 있다.

- LoRA Base 먼저: 메인 샘플링의 처음 N% 스텝은 LoRA 없이, 나머지는 LoRA 로 그린다(100% 미만이면 마지막 스텝은 늘 LoRA). 구도는
  노이즈가 큰 초반에 정해지므로 Base 가 프롬프트대로 잡은 구도 위에 LoRA 가 화풍을 입힌다. ComfyUI 실행을 두 번(Base 모델, LoRA
  모델) 하지만 둘 다 전체 스텝 목록을 받고 가장 안쪽 샘플러 함수만 잘라서, Detail Daemon·DAVE·S²·CNS·Dy 샘플러·노이즈 흐름은 한 번
  실행과 같다. 다단계 샘플러는 경계에서 이력이 끊긴다. Hires·SAM3·후처리·인페인트와 LoRA 의 텍스트 인코더 부분은 LoRA 그대로다.
  SPEED·Spectrum·Restart·UniPC·DPM fast/adaptive 와는 같이 못 쓴다. 모델 패치가 두 번 바뀌어 그림마다 조금 느려진다.
- LoRA 교차 어텐션 제외: LoRA 가 Anima 의 글자→그림 경로(교차 어텐션과 LLM 어댑터 전체)는 건드리지 않는다. `+변조` 는 교차
  어텐션의 adaLN 변조까지 뺀다. 모든 패스에 걸린다. 트리거가 덜 먹을 수 있다. MOSS·Krea2 에서는 회색이다.
- CFG 스케줄: 처음 CFG 를 높게 두고 정한 % 까지 패널 CFG 로 내린다(코사인·선형·계단). 메인 패스만.
- 단어 주목: 적은 단어의 T5 토큰이 교차 어텐션에서 받는 비중을 배수만큼 키운다(로짓에 log 배수를 더함). `(단어:1.2)` 는 값의
  크기를 키우고(Anima 는 키를 정규화해서 비중은 그대로) 이것은 비중을 키운다. Anima 만, 메인 패스만. 3.8B 의 연결 경로는 토큰 배치가
  달라 건너뛴다.
- % 는 그 실행의 스텝 순서로 센다(Base 먼저의 두 실행도 전체 스텝 기준). 2차 샘플러의 끝점 호출은 다음 스텝으로 센다.
- 듣지 않는 경우(인페인트의 Base 먼저, Anima 가 아닌 모델의 교차 어텐션 제외, 3.8B 나 단어가 빈 단어 주목)는 회색이거나 알림이
  뜨고, `parameters` 에 남지 않으며, 그 X/Y/Z 축은 격자를 만들지 않는다.

## 스텝 줄 스위치 (1.20.0)

Forge 에는 없다. 모델 · 샘플링 카드의 스텝 칸 바로 왼쪽에 `스킵 22`(MOSS 만), `선행`, `후행` 이 있고 모두 기본 켬이다.

- 스킵 22: MOSS-Image 공식 샘플링(OpenMOSS ComfyUI 워크플로, 학습기 샘플러)의 레이어 스킵이다. 네거티브 패스에서만 22번 블록을
  건너뛰고 전 구간에 건다. 코어 `SkipLayerGuidanceDiTSimple(single_layers="22", 0~1)` 과 같은 결과다. MOSS 로 샘플링하는 모든
  패스(메인, Hires, 인페인트, SAM3, XYZ 칸, 갤러리 후처리)에 걸리고 CFG 1 에서는 네거티브가 안 돌아 차이가 없다.
- **1.19.0 까지와 달라지는 점:** 전에는 MOSS 에 레이어 스킵이 없었다. 스위치가 없던 저장 패널도 켬으로 읽으므로 MOSS 그림은 같은
  시드라도 달라진다. 일부러 공식 값에 맞춘 것이고, 끄면 전과 같다. Anima·Krea2 는 영향이 없다.
- 코어 노드 대신 22번 블록의 block replace 로 구현했다(`studio_model_nodes.moss_layer_skip`). 코어 노드는
  `sampler_calc_cond_batch_function` 을 혼자 쓰는데 가이던스 묶음의 ADG·PAG 도 그 자리를 쓴다. 한 배치에 포지티브와 네거티브가
  같이 돌면 블록을 돌린 뒤 네거티브 줄을 들어오기 전 값으로 되돌린다. 코어(GPL) 코드는 복사하지 않았다. 블록을 샘플 하나씩
  돌리는 DiT(`moss-image` 로더, MOSS-Toolkit 레퍼런스 패스)는 `sigmas` 로 배치 크기를 알아 네거티브 샘플에서만 건너뛰고, 줄을
  가릴 수 없는 호출은 스킵 없이 돌리는 대신 오류로 멈춘다.
- 같이 쓸 수 있는 것: MOSS NegPiP(DIFFUSION_MODEL 래퍼), fp32 stream(APPLY_MODEL 래퍼), CFG 스케줄(PREDICT_NOISE 래퍼), LoRA
  Base 먼저(Base 모델은 사본이라 스킵이 따라간다), MOSS Spectrum(가지별 예측이라 네거티브 가지는 스킵한 특징을 배운다). 같이 못
  쓰는 조합은 없다.
- 선행 / 후행: 선행 고정·후행 고정 칸을 합친 프롬프트에 넣을지. 끄면 칸의 글은 남고 흐려지며, 실행·XYZ 칸·와일드카드·LoRA 태그
  검사·`parameters` 의 프롬프트에서 빠진다. 둘 다 켜면 1.19.0 과 글자 하나 다르지 않다. 나머지 칸과 네거티브의 스위치는 프롬프트
  카드에 있다(아래 "프롬프트 카드 스위치와 색").
- `parameters`: MOSS 그림이고 스킵을 켰을 때만 `MOSS layer skip: 22`. 프롬프트 줄은 실제로 쓴 합친 글이다.
- 설정 가져오기: 스튜디오 그림은 세 스위치를 패널 상태에서 되돌린다. 패널 상태에 스위치가 없으면 워크플로를 열 때처럼 셋 다
  켬이다(1.20.0 전의 그림, 그리고 1.20.0 에서 패널을 건드리지 않고 만든 그림: 숨은 위젯은 첫 변경 때만 쓰인다). 1.19.0 까지의
  MOSS 그림도 켬으로 들어온다. `parameters` 만 있는 MOSS 그림은 `MOSS layer skip` 유무를 따른다.
- 스텝 줄 스위치는 `data-focus-key` 를 가져서 키보드로 눌러 패널이 다시 그려져도 포커스를 잃지 않는다.
- 그래프: 노드·링크는 그대로다. `AIStudioMossStream` 에 선택 위젯 `stream`·`layer_skip` 이 붙었을 뿐이라 저장한 워크플로를 다시
  만들지 않아도 된다. MOSS 노드가 없는 옛 워크플로에서는 `스킵 22` 가 회색이다.

## 프롬프트 카드 스위치와 색 (1.20.0)

Forge 에는 없다. 프롬프트 카드의 여덟 칸마다 이름 왼쪽에 켬/끔 스위치가 있고 모두 기본 켬이다.

- 키: `prompt_prefix_on`, `prompt_char_count_on`, `prompt_character_on`, `prompt_copyright_on`, `prompt_artist_on`, `prompt_on`,
  `prompt_suffix_on`, `negative_on` (`studio_state.BOX_SWITCHES`, 불리언만). `prompt_prefix_on`·`prompt_suffix_on` 은 스텝 줄의
  `선행`·`후행` 과 같은 값이다.
- 끈 칸: 글은 남고(고칠 수도 있다) 합친 프롬프트에서 빠진다(`assembled_prompt`). 프롬프트 칸만 켜져 있으면 전처럼 쓴 그대로이고,
  프롬프트 칸을 끄면 나머지 칸만 잇는다. 전부 끄면 빈 프롬프트다(LoRA 목록의 태그는 칸이 아니라서 그대로 붙는다).
- 네거티브 끔: `effective_negative` 가 빈 글을 준다. 패널은 그 글을 `lora_stack` 의 `negative` 에 쓰고, 메인·i2i·인페인트 디테일러·
  Hires·SAM3 는 모두 거기서 나온 네거티브 조건을 받는다. Hires·SAM3·Refine·타일의 자체 네거티브 칸은 이 스위치와 무관하다.
- XYZ: 칸 상태는 실제 쓰는 글로 만든다. 합친 프롬프트는 프롬프트 칸에 들어가고 그 칸은 켬이 되며(`single_prompt`), 꺼진 네거티브는
  빈 글에 켬으로 들어간다. Prompt S/R·Prompt order·LoRA 축은 그 글에서 찾으므로 꺼진 칸에만 있는 말은 "찾지 못함" 으로 격자를
  만들지 않는다. 꺼진 칸의 `<lora:...>` 는 파일 검사 대상이 아니다.
- 와일드카드: 꺼진 칸과 꺼진 네거티브는 뽑지 않고 글자 그대로 남는다. 뽑은 뒤의 프롬프트는 프롬프트 칸에 들어가고 그 칸은 켬이
  된다(끈 채로 두면 뽑은 글이 실행에서 빠진다).
- `parameters`: 프롬프트 줄과 `Negative prompt` 는 실제로 쓴 글이다. 네거티브를 끄면 `Negative prompt` 줄을 쓰지 않는다.
- 예전 것: 스위치가 없는 패널·워크플로·그림·갤러리 기록은 전부 켬이다. 프롬프트와 네거티브가 전과 바이트 단위로 같다.
- 설정 가져오기: 스튜디오 그림은 패널 상태의 스위치를 되돌리고 없으면 켬이다. `parameters` 만 있는 그림은 `prompt_on`·
  `negative_on` 을 켬으로 가져온다(네거티브 없이 만든 그림은 빈 네거티브에 켬). 비워지는 나머지 칸의 스위치는 패널 그대로 둔다.
- 화면: 스위치는 이름 왼쪽의 작은 토글(`button.ais-box-switch`, `role="switch"`, `data-switch="<키>"`)이다. 끄면 줄(`.ais-row.ais-off`)의
  칸이 흐려지고 이름이 회색 취소선이 된다. 스위치와 스텝 줄 pill 은 `data-focus-key` 로 다시 그린 뒤에도 포커스를 지킨다.
- 색: 포지티브 이름 `--ais-pos` `#7ddf9f`(카드 배경 `#2b2b2b` 대비 8.7:1), 네거티브 이름 `--ais-neg` `#ff9585`(6.7:1), 꺼진 이름
  `--ais-box-off` `#9a9a9a`(5.0:1). 칸 테두리는 같은 색을 옅게 쓰고 포커스 테두리(파랑)는 그대로다.
- 칸 이름은 스위치와 따로인 `.ais-box-label[data-section="<칸 키>"]` 다. `PanelView` 의 `ctx.boxLabel(key, anchorEl)` 이 있으면 버튼이
  되어 누를 때 그 함수를 부르고, 없으면 글자다. 프리셋 목록이 여기에 붙는다.
  - 이름 요소는 칸마다 하나를 패널이 살아 있는 동안 다시 쓴다(`PanelView.boxNames`, "+ LoRA" 버튼과 같은 방식). 프리셋 목록
    (`preset_quick.js`)은 열릴 때 받은 `anchorEl` 을 닫힐 때까지 들고 있으면서 자리 계산, "이름을 누른 것은 바깥 클릭이 아님",
    `aria-expanded`, 닫힐 때 포커스 돌려주기에 쓴다. 다시 그릴 때마다 요소가 바뀌면 프리셋을 넣은 뒤 포커스가 페이지로 빠지고,
    되돌리기 뒤 목록이 창 구석으로 가고, 이름이 토글 노릇을 못 한다. 그래서 요소를 바꾸지 않는다.
  - 프리셋을 고르면 포커스는 목록 안에 있으므로, 돌려주는 것은 목록의 `close()` 다(이름이 그대로 문서에 붙어 있어서 된다).
    `data-focus-key="box-label-<키>"` 는 이름 자체에 포커스가 있는 채로 패널이 다시 그려질 때(목록 로딩 등)를 맡는다.
  - 이름 요소에는 상태에 따라 바뀌는 속성을 두지 않는다. 꺼짐 표시(회색·취소선)는 줄의 `.ais-off` 가 한다.
- 레퍼런스 카드의 "트리거 넣기"(Krea2 QuadView)는 `studio_logic.addPromptTrigger` 를 쓴다. 실행에 들어가는 프롬프트
  (`assembledPrompt`, 꺼진 칸 제외)에 트리거가 없을 때만 프롬프트 칸 맨 앞에 넣고 `prompt_on` 을 켠다. 칸에 이미 있으면 다시 넣지
  않고 켜기만 한다. 다른 칸의 스위치는 건드리지 않는다. (프롬프트 칸에 글을 써 넣는 동작은 와일드카드·설정 가져오기처럼 그 칸을 켠다.)

## 프롬프트 프리셋 (1.21.0)

Forge 의 Styles 와 같은 자리에 있지만 다르게 만들었다(Forge 코드는 보지 않았다).

- Forge 의 스타일 하나는 프롬프트와 네거티브에 통째로 얹힌다. 스튜디오의 프리셋은 **칸마다** 따로다: 프롬프트 카드의 일곱 칸과
  네거티브가 저마다 목록을 가진다. 넣는 방법은 둘이다: 칸을 바꾸기, 칸 뒤에 `, ` 로 붙이기. `{prompt}` 같은 자리 표시는 없다.
- 칸의 이름을 누르면 이름 아래에 그 칸의 목록(`preset_quick.js`)이 뜨고, 거기의 `프리셋 관리…` 가 큰 창(`preset_window.js`)을 연다.
  이름 아래 작은 글이 지금 칸이 어떤 프리셋인지 말한다: 프리셋 이름 / `저장 안 됨` / (빈 칸은 없음). 같은지는 양 끝의
  쉼표·공백·줄바꿈을 뺀 글로 본다(`preset_logic.js` `matchPreset`, Python `studio_presets.match_preset`: 둘은 golden 표로 묶여 있다).
- 프리셋은 실행에 아무것도 더하지 않는다. 칸에 글을 넣을 뿐이라 `parameters`, XYZ, 와일드카드, 스위치는 칸의 글만 본다. 프리셋
  이름은 그림에 남지 않는다.
- 저장: ComfyUI `user/default/ai_studio/prompt_presets.json`(`studio_presets.py`). 워크플로 밖이라 워크플로를 바꿔도 같은 프리셋이
  보인다. 변경은 하나씩 서버로 가고(`POST /presets`) 서버가 파일을 읽어 그 변경만 적용해 통째로 바꿔 쓴다. 되돌리기도 목록 사본이
  아니라 그 변경의 반대다(더한 것 지우기, 지운 것 제자리에 넣기). 다른 탭이 그 사이 저장한 것을 덮지 않으려는 것이다.
- 패널 배선(`preset_panel.js`): 스토어는 페이지에 하나, 패널이 만들어질 때 한 번 읽는다. 읽기 전이나 읽기에 실패하면 이름 아래에
  아무것도 쓰지 않는다(전부 `저장 안 됨` 이라고 하면 거짓말이다). 스토어가 바뀌면 이름 아래 글만 다시 채우고 패널은 다시 그리지
  않는다(열려 있는 목록, 치고 있던 칸의 커서가 그대로다).
- 이름 요소는 1.20.0 그대로 칸마다 하나다(위 "프롬프트 카드 스위치와 색"). 아래 글은 그 옆의 다른 요소(`.ais-box-mark[data-for]`)라
  이름에는 여전히 상태에 따라 바뀌는 것이 없다.

## XYZ 값 블록 (1.21.0)

- 목록 축: Forge 도 목록에서 고르는 축은 값을 드롭다운으로 여러 개 고른다. 스튜디오는 고른 값을 블록으로 놓는다(`xyz_chips.js`).
  값 글은 전처럼 축에 남고(한 줄에 한 값) 플래너가 그 글을 읽으므로, Forge 에서 가져온 `X Values` 글도 그대로 듣는다.
- Prompt S/R·Prompt order: Forge 는 글 칸 하나에 CSV 로 쓴다. 쉼표가 든 값은 따옴표로 감싸야 한 값이 된다
  (`"blue hair, long hair", "red hair, short hair"`). 스튜디오의 플래너도 처음부터 같은 규칙으로 읽었다(`xyz_values.split_values`).
  다만 글 칸에서는 그 규칙이 보이지 않아 태그 하나씩만 바꿀 수 있는 것처럼 보였다. 1.21.0 은 화면을 바꿨다: 값마다 글 블록 하나
  (`xyz_blocks.js`), Prompt S/R 은 첫 블록이 `찾을 글`. 블록에 쓴 쉼표·따옴표·줄바꿈은 스튜디오가 따옴표를 붙여 값 글에 적는다.
- 값 글의 모양은 Forge 와 같다. 블록이 쓰는 글(한 줄에 한 값, 필요한 값만 따옴표)도, 손으로 쓴 Forge 식 한 줄도 같은 값으로 읽힌다.
  `직접 입력` 으로 값 글을 그대로 볼 수 있다. JS 와 Python 이 같은 값으로 나누는지는 `xyz_values_golden.json` 이 지킨다.
- 값의 양 끝 공백은 전처럼 버린다. 블록에 치는 동안에는 친 그대로 보이고, 값 글에는 버린 채로 적힌다.
- Forge 와 다른 점 하나: Prompt S/R 의 찾을 글이 비어 있으면 격자를 만들지 않는다
  (`Prompt S/R: the first value (the text to search for) is empty`). Forge 는 빈 글을 "찾아서" 모든 글자 사이에 바꿀 글을 넣는다.
  블록에서는 빈 `찾을 글` 이 생기기 쉬워서 막았다. 빈 `바꿀 글` 은 Forge 처럼 찾은 글을 지운다.
- 찾을 글은 실제로 쓰는 프롬프트에서 찾는다. 칸들은 `, ` 로 이어진 뒤라서 `@a, 1girl` 처럼 두 칸에 걸친 글도 찾는다. 꺼진 칸의
  글은 거기 없다(위 "프롬프트 카드 스위치와 색").
- 글 블록 목록은 300px 에서 안쪽이 스크롤되고, 손댄 블록(새로 연 것, 옮긴 것, 치는 동안 줄이 늘어난 것)이 가려지면 목록 안쪽만
  그 블록이 보일 만큼 움직인다. XYZ 열과 캔버스는 움직이지 않는다.
- LoRA 선택 창(1.19.0)의 스크롤도 같이 고쳤다. 카드를 눌러도 목록이 위로 밀리지 않고(스크롤 앵커링 끔), `최근 사용`(폴더와 정렬)은 창이
  열려 있는 동안 창을 열 때의 순서로 보여 준다(방금 추가한 것이 맨 앞으로 가면 누르던 카드가 자리를 옮긴다). 최근 목록 자체는
  바로 저장되고 창을 다시 열면 새 순서다.

## 샘플러·스케줄러 목록 순서 (1.21.0)

- Forge 는 샘플러와 스케줄러를 등록한 순서로 보여 준다. ComfyUI 도 그런데, 팩이 더한 이름이 전부 끝에 붙어서 비슷한 이름이
  떨어진다. 스튜디오는 이름의 첫 낱말이 같은 것을 한 줄로 모은다(`studio_choices_order.group_similar`).
- 값은 ComfyUI 이름 그대로다. 순서만 다르므로 워크플로, `parameters`, XYZ 값 글에는 영향이 없다.

## 패널 안의 캔버스 단축키 (1.21.0)

- Forge 에는 없는 문제다. 스튜디오의 패널은 ComfyUI 캔버스 위의 노드라서, 패널 안의 버튼·스위치·고르는 칸에 포커스가 있을 때 누른
  Delete·Backspace 를 ComfyUI 가 "선택한 노드 지우기" 로 듣는다(패널을 누르면 그 노드가 선택된다). P(고정), Alt+C(접기),
  Ctrl+B(우회), Ctrl+M(끄기) 등도 같다.
- 제어판, XYZ 패널, EXIF 패널의 맨 바깥 요소가 이 키들을 밖으로 내보내지 않는다(`studio_dom.js` `guardCanvasKeys`). 결과 노드는
  `result_view.js` 가 같은 일을 한다. 글 칸은 제 키를 그대로 가지고, 브라우저의 동작(고르는 칸에서 글자로 항목 찾기)과 Ctrl+Enter
  같은 다른 ComfyUI 단축키는 그대로다.
- 1.21.0 에서 프리셋 목록·창이 닫힐 때 포커스가 칸의 이름으로 돌아오고 XYZ 의 `직접 입력` 이 포커스를 가지게 되면서, 막지 않으면
  프리셋을 고른 직후의 ⌫ 한 번에 패널이 지워졌다.

## 갤러리 키보드 (1.21.0)

- Forge 의 갤러리는 확대해서 볼 때 ←/→ 로 넘긴다. 스튜디오는 갤러리 띠에서도 된다: 결과 노드 안에 포커스가 있으면 ←/→,
  Home/End 가 그림을 고르고, 고른 썸네일이 띠 가운데로 온다.
- ComfyUI 캔버스 위라서 생기는 일 하나를 막았다. 노드 안에서 누른 Delete·Backspace 등 캔버스 단축키는 노드 밖으로 나가지 않는다
  (그대로 두면 선택된 결과 노드가 지워진다). 글자를 치는 칸은 제 키를 그대로 가진다.
- 확대 창(그림을 눌러 여는 큰 보기)에서 넘긴 그림을 갤러리가 고른다(`ResultView.follow`). 창을 닫으면 보던 그림이 선택되어 있다.
  자리가 아니라 파일로 찾고, 그 사이 갤러리에서 빠진 그림이면 선택을 바꾸지 않는다.

## LoRA 선택 창 (1.19.0) clean-room 메모

ComfyUI-Lora-Manager (GPL-3.0) 는 기능 참고용으로 README/wiki 문서만 봤다. 코드·CSS·아이콘은 복사하지 않았고 구현은 이 팩에서 새로 썼다.
