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
- Anima 레퍼런스: SnapRef 또는 ComfyUI-Anima-Ref-Adapter. 가중치는 Anima base(28블록) 기준이고, 2.9B·3.8B 는 일반 LoRA 와
  같은 블록 계보 표로 변환한 사본을 쓴다(실측 확인). 변환 없이 쓰면 SnapRef 는 참조 그림을 베끼고 Ref Adapter 는 노이즈가
  나온다. SnapRef 는 PAG 와 같이 못 쓴다. v2a 프리뷰는 v1a 와 같은 방식으로 쓰며 캐릭터만 모드가 나아졌고, 둘째 그림과 충실도는
  v2a 와 함께 나온 SnapRef 노드가 필요하다. 충실도는 SnapRef 가이던스를 끈 때만 듣는다.
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
