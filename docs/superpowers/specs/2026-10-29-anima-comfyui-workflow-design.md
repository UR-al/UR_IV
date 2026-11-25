# Anima ComfyUI 워크플로 (WebUI식 제어판 + XYZ) 설계

작성 2026-10-29 · 화면 시안: https://claude.ai/artifact/3gfEnw1YFovVVwV8QLyccm (확정안은 **C4**, 탭·생성/정지 버튼은 시안에서 눌러 볼 수 있음)

## 1. 왜 만드나

- 사용자는 Anima 그림을 거의 Forge Neo(WebUI)에서 뽑아 왔고, 그 편한 사용감에 손이 익어 있다. 이번에 ComfyUI로 옮기려 한다.
- ComfyUI를 피해 온 가장 큰 이유는 **XYZ 플롯이 불편해서**다. LoRA 에폭별 비교, LoRA 세기, 프롬프트 바꿔 끼우기, 체크포인트·TE·VAE 비교 등
  WebUI XYZ에 있는 것은 거의 다 써 봤다.
- 예전에 만들던 Anima 워크플로는 없어져서 새로 만든다. 화면이 깔끔하고 한눈에 읽혀야 한다.
- 사용자가 좋아하는 순서: ① 설정이 한눈에 다 보이는 WebUI식 > ② 기능마다 그룹 주석이 있어 중간에 고치기 쉬운 것.
  **가장 싫은 것: 정해진 흐름 없이 왼쪽 갔다 오른쪽 갔다 하는 배선.**

## 2. 목표 · 비목표 · 실패 조건

**목표**
- Forge처럼 쓰는 Anima 생성 화면 하나(t2i·i2i·인페인트·Hires fix·업스케일·SAM3 디테일러·XYZ)를 ComfyUI 안에 만든다.
- XYZ는 WebUI만큼 쉬워야 한다: 축 고르고 값 적고 "생성" 한 번이면 격자가 나온다.
- Forge와 결과가 "비슷하게" 나오도록 맞추는 장치(토글)를 둔다. 픽셀 일치는 목표가 아니다.

**비목표 (첫 판)**
- Anima가 아닌 모델(MOSS, Krea2, SDXL 등) 지원.
- Forge와 픽셀 단위로 같은 결과.
- 앱(AI Studio Pro)의 기존 ComfyUI 컴파일러 경로(`core/comfy_workflow_compiler.py`)를 바꾸거나 대체하는 것.
- 와일드카드·다이내믹 프롬프트, ControlNet, 영상 (논의 안 됨 → 첫 판에서 뺌, *Claude 결정*).

**실패 조건 (돌아가도 실패)**
- XYZ가 WebUI보다 불편하다(축 설정이 여러 노드에 흩어지거나, 격자를 따로 합쳐야 함).
- 배선·그룹이 좌우로 왔다 갔다 한다.
- 결과 이미지가 패널 크기에 끌려 작아진다.
- 버튼·탭 글자가 길다 ("t2i에서 i2i로 전환" ✗ → "t2i <> i2i" ○).

## 3. 사용자가 정한 것

### 3.1 화면 배치 (시안 C4, 1920×1080 기준)
왼쪽 열(약 790px) · 가운데 결과(680px) · 오른쪽 XYZ 패널(나머지).

- **맨 위 막대**: 제목, **생성 버튼(Ctrl+Enter)**. 생성 중에는 같은 버튼이 **정지** 버튼으로 바뀐다. 생성 중 막대 바로 아래 얇은 **진행 바**.
- **공통 설정** (왼쪽 맨 위, 그룹 하나)
  - 1줄: **프롬프트**가 가로 한 줄을 다 쓴다(긍정 + 부정).
  - 2줄: **모델 · 샘플링** 한 칸 = 체크포인트, TE, VAE, 샘플러, 스케줄러, 스텝·CFG·Shift, 시드 + 제목 옆 "Forge처럼" 스위치 /
    **LoRA** 목록(켬/끔, 이름, 세기) / **후처리** = Hires fix · 업스케일 · SAM3 디테일러 토글(켜면 생성 때 바로 적용).
- **t2i | i2i 탭** (공통 설정 아래, WebUI처럼 같은 자리에서 바꿔 보기)
  - t2i: 크기, 장수, Hires fix 설정(업스케일러, 배율, 디노이즈, 스텝).
  - i2i: 입력 이미지, 마스크 그리기, i2i/인페인트 모드, 디노이즈, 인페인트 영역, 마스크 흐림.
  - t2i 그룹과 i2i 그룹은 **별개 그룹**이다.
- **엔진룸**: 실제 노드 그룹. 평소엔 접혀 있고, 펼치면 ①모델 → ②LoRA → ③프롬프트 → ④샘플링 → … → ⑧ 순서로 **위에서 아래로**(세로) 쌓인다.
  각 줄 안에서는 그 그룹의 노드가 나란히 놓인다. 배선은 한 방향으로만 흐른다.
- **결과 (가운데)**
  - 이미지는 칸 높이를 꽉 채우고 원래 비율을 지킨다(가로가 줄어도 작아지지 않게).
  - 미리보기 / XYZ 격자 탭. 생성 중에는 진행 표시(스텝 n/N, %, 남은 시간)와 중간 단계 미리보기.
  - 그 아래 **갤러리** 썸네일 줄(가운데 정렬, 고른 이미지에 테두리).
  - 그 아래 **아이콘 버튼**, Forge 순서: 폴더 → i2i → 인페인트 → 업스케일 → Hires → SAM3 → 같은 시드.
    마우스를 올리면 짧은 이름. 누르면 **고른 이미지 하나에** 그 작업을 바로 한다(Forge 갤러리 밑 버튼처럼).
- **XYZ 패널 (오른쪽)**: X·Y·Z 축마다 종류 + 값(여러 줄 칸, 에폭 파일 이름이 한 줄에 하나씩 다 보일 만큼), 목록형 축은 값 고르기 버튼,
  Forge XYZ 옵션(범례 그리기 · -1 시드 유지 · 개별 이미지 포함 · 하위 격자 포함 · 여백 · 축 바꾸기), 예상 장수. **격자 결과는 이 패널이 아니라 결과 칸의 "XYZ 격자" 탭**에서 본다.

### 3.2 기능
- **XYZ = Forge Neo의 XYZ plot이 그대로 동작하게** (사용자 결정, 2026-10-29). 기준은 `C:\sd-webui-forge-classic\scripts\xyz_grid.py`의 동작:
  - 축 목록: Forge Neo 축 중 이 워크플로에 해당하는 것(Seed, Steps, Size, CFG Scale, Shift, Prompt S/R, Prompt order, Sampler, Schedule type,
    Checkpoint name, VAE, Denoising, …) + sam-extra가 Forge XYZ에 더한 축 중 팩에 같은 기능이 있는 것(TE, [Detail Daemon] …, [SAM3] … 등).
    해당 기능이 없는 축(Face restore, Token merging, Refiner 등)은 목록에서 뺀다.
  - 값 문법(쉼표 목록, `1-5`, `1-5 (+2)`, `1-10 [5]` 같은 범위, Prompt S/R은 첫 값이 찾을 글), 격자·범례 그리는 모양, 옵션(범례 그리기,
    -1 시드 유지, 개별 이미지 포함, 하위 격자 포함, 여백, 축 바꾸기)을 Forge와 같게.
  - "생성" 한 번에 조합 수만큼 뽑고 격자 한 장으로 합친다.
- **LoRA 비교는 Forge처럼 Prompt S/R로**: 프롬프트에 `<lora:LoRA1:1>`이 있어야 하고(첫 값이 찾을 글 — 없으면 Forge처럼 생성 전에 알림),
  값마다 `<lora:LoRA2:1>`(다른 파일·에폭)이나 `<lora:LoRA1:0.9>`, `<lora:LoRA1:1.1>`(세기)로 바꿔 끼운다.
- **체크포인트 축은 같은 계열끼리만**: 2.9B·base끼리, 3.8B는 3.8B끼리. 2.9B와 3.8B를 한 격자에 섞을 일은 없다(사용자) → 섞으면 생성 전에 알린다.
- **LoRA**: 목록에서 고르는 게 기본. 프롬프트에 `<lora:이름:세기>`를 적어도 Forge처럼 걸린다(태그는 글에서 지운 뒤 적용).
- **Forge처럼 토글**: 커스텀 노드의 토글로 켜고 끌 수 있으면 된다. 픽셀까지 같을 필요 없다.
- **SAM3 / SAM3.1 디테일러**: 가끔 켠다. Forge에서는 sam3 확장(sam-extra)에서 **sam3.1을 골라 그냥 썼다** → 첫 판에 SAM3와 SAM3.1 둘 다 고를 수 있어야 한다.
- **i2i · 인페인트 · Hires fix · 업스케일**: 토글만 켜면 바로 되게.
- **새 노드는 앱 팩(`ai_studio_forge_parity`)에 넣는다.**

## 4. Claude가 정한 것 (사용자 확인 전 기본값)
- 같은 LoRA가 목록과 프롬프트 양쪽에 있으면 한 번만 걸고, 세기는 프롬프트 값을 따른다.
- 제어판 LoRA 목록에서 고른 LoRA도 XYZ의 Prompt S/R이 찾을 수 있게, S/R은 "목록 LoRA를 `<lora:이름:세기>` 태그로 붙인 프롬프트"에서 찾는다
  (목록만 쓰는 사람도 Forge처럼 LoRA를 바꿔 끼울 수 있게).
- "Forge처럼"은 큰 스위치 하나 + 엔진룸 항목별 스위치: 초기 노이즈 방식, 프롬프트 가중치·편집 문법(`(x:1.2)`, `[a:b:0.5]`, BREAK),
  DoRA 적용 방식(팩의 `ForgeNeoDoraInferenceMode`), 어텐션. 마지막 단계에서 같은 시드로 Forge와 나란히 놓고 눈으로 확인한다.
- 기본값 = 사용자의 실제 Forge 설정: EXP Heun 2 x0 SDE · Flow Cosmos Dynamic · 32스텝 · CFG 3 · Shift 3.
  Hires fix 기본 1.5배 · 디노이즈 0.4 · 16스텝 · 4x-UltraSharp(Forge 기본값과 비슷하게). XYZ는 모든 칸 같은 시드.
- 모델: Anima 2.9B 기본, base · 3.8B로 바꿔 끼울 수 있게(3.8B는 팩의 `ForgeNeoAnima38V2Loader`/`Prompt`).
- 저장: Forge 형식 생성 정보를 이미지에 넣는다(앱 이미지 뷰어 메타데이터 패널에서 그대로 읽히게) — 팩의 `ForgeNeoSaveImage` 우선 검토.
- t2i와 i2i는 한 번에 하나만 켜진다. 탭을 바꾸면 그쪽 그룹만 펼치고 켜며, 생성은 열린 탭 쪽으로만 흐른다.
- 마스크 그리기는 ComfyUI 기본 마스크 편집기를 쓴다.
- XYZ 격자에는 칸마다 축 값 라벨을, 격자 머리에는 LoRA 조합을 "meow (1.0) + enaa97 V2 (0.9)" 식으로 적는다(사용자의 기존 표기 선호).
- 갤러리 버튼: **→ i2i / → 인페인트는 이미지를 그 탭 입력으로 보내고 탭만 바꾼다**(Forge와 같음, 생성은 사용자가 누름 — 인페인트는 마스크가 필요).
  **업스케일 · Hires · SAM3 · 같은 시드는 고른 이미지에 바로 실행**하며, 그 이미지를 만든 설정(프롬프트·시드 등: 이번 세션 기록,
  없으면 이미지에 저장된 생성 정보)을 쓰고 후처리 설정만 지금 값을 쓴다.
- 대상 ComfyUI는 사용자가 쓰는 `C:\ComfyUI_windows_portable`(앱이 관리하는 ComfyUI에도 팩이 설치되면 그대로 동작).

## 5. 제약
- ComfyUI 0.36.0, 프런트엔드 패키지 1.52.7, 내장 Python 3.13 + torch cu130, Manager 4.2.2. Sage 어텐션으로 실행하는 배치 파일을 쓴다.
  노드 의존성 설치는 torch·numpy 고정 constraints로(그 ComfyUI 메모리 규칙).
- 앱 팩은 지금 1.9.0, 프런트 JS 확장(WEB_DIRECTORY)이 **없다** → 새로 만든다. 원본은 앱 저장소 `comfy_custom_nodes/ai_studio_forge_parity`,
  설치·업데이트는 앱 설치기(`core.comfy_node_pack.install_bundled_node_pack`, `.aistudio-owned.json`). 팩 버전·README·THIRD_PARTY_NOTICES 갱신.
- 이미 있는 팩 노드를 최대한 재사용한다: `ForgeNeoAnimaQwen35Loader/Prompt`, `ForgeNeoAnima38V2Loader/Prompt`, `ForgeNeoAnimaLoraLoader`,
  `ForgeNeoLatentInput`(t2i/i2i/inpaint), `ForgeNeoKSamplerCNS`, `ForgeNeoModelSamplingShift`, `ForgeNeoHiresFix`, `ForgeNeoExtraSchedulers`,
  `ForgeNeoSAM3Detailer`, `ForgeNeoADetailer`, `ForgeNeoDoraInferenceMode`, `ForgeNeoSaveImage`.
- 프롬프트 `<lora:…>` 규칙은 앱의 `core/comfy_workflow_compiler.py`의 `parse_lora_tags`(Forge 규칙)를 따른다. ComfyUI 안에서 돌아야 하므로
  순수 함수로 팩에 옮기거나 공유한다(앱 쪽 동작은 바꾸지 않는다).
- 저장소 규칙(CLAUDE.md): 작은 단일책임 파일, 순수 로직은 Qt·ComfyUI와 분리해 `tests/`에 unittest, Python 수정 뒤 venv `run_tests.py`.
  다른 팩 코드를 가져올 땐 라이선스 확인(팩은 MIT, GPL 코드 금지; sam-extra 자체 코드는 MIT 재사용 허락됨).
- XYZ 라이선스: Forge Neo `xyz_grid.py`는 AGPL-3.0이라 MIT인 팩에 코드를 복사하지 않는다. 동작(축·값 문법·격자 모양·옵션)을 보고 새로 짠다.
  sam-extra가 더한 축 코드는 sam-extra 자체 코드라 MIT 재사용 가능.
- SAM3.1: 앱 팩(`sam3_nodes.py`)과 `comfyui-easy-sam3`에는 아직 SAM3.1(`sam3.1_multiplex_fp16.safetensors`, Forge 모델 폴더에 있음)을
  읽는 코드가 없다. Forge sam3 확장(sam-extra, `C:\sd-webui-forge-classic\extensions\forge_sam3_extension`)이 실제로 3.1을 읽는 방식을 찾아
  팩으로 옮긴다. sam-extra 자체 코드는 팩에서 MIT로 재사용해도 된다(사용자 허락); 그 안의 제3자 코드는 라이선스를 따로 확인한다.
- GPU: 학습이 돌고 있는 경우가 많다. GPU를 쓰는 확인(생성 시험)은 먼저 사용자에게 묻고 VRAM 사용량을 밝힌다.
- 커밋·푸시는 사용자가 시킬 때만.

## 6. 빌더에게 맡기는 것 (정해지지 않은 '어떻게')
- 제어판·탭·갤러리·진행 표시·XYZ 패널을 ComfyUI 화면에 어떻게 그릴지(노드 위젯·DOM 위젯·사이드 패널 등). 위 배치와 "엔진룸은 접혀 있고
  위→아래 한 방향" 원칙만 지키면 된다.
- XYZ 실행 방식(한 번의 큐 안에서 리스트로 도는지, 프런트에서 여러 프롬프트를 큐에 넣는지)과 격자 합치기·축 라벨 그리기.
- 갤러리 버튼이 고른 이미지를 어떤 경로로 다시 돌리는지(ComfyUI 기록·업로드 API 등), 정지(인터럽트)와 진행률(웹소켓 진행 이벤트) 연결.
- "Forge처럼" 항목별로 실제로 어디까지 맞출 수 있는지 조사(초기 노이즈 RNG, 프롬프트 가중치가 Anima TE에 어떻게 먹는지 등) — 못 맞추는
  항목은 토글을 회색으로 두고 이유를 적는다.
- 업스케일 모델 목록·SAM3 모델 경로를 어디서 읽을지(Forge 폴더 공유 `extra_model_paths.yaml`).
- 워크플로 JSON을 손으로 만들지 생성기 스크립트로 만들지.

## 7. 완료 기준
- 사용자 ComfyUI에서 워크플로를 열면 C4 배치대로 보이고, 공통 설정만 만져서 t2i·i2i·인페인트·Hires·업스케일·SAM3가 된다.
- XYZ: Prompt S/R로 LoRA1 에폭 파일 4개 × Prompt S/R로 LoRA1 세기 3개를 "생성" 한 번으로 12장 격자(Forge와 같은 범례)까지 나온다.
  Forge Neo에서 같은 축·값을 넣었을 때와 같은 조합·같은 격자 배치가 나온다. 찾을 글이 프롬프트에 없거나 2.9B와 3.8B를 섞으면 생성 전에 알린다.
- SAM3 디테일러에서 SAM3와 SAM3.1을 모두 골라 쓸 수 있다.
- 프롬프트 `<lora:…>`가 걸린다. 생성/정지 버튼과 진행 표시가 동작한다. 갤러리 버튼 7개가 고른 이미지에 동작한다.
- 같은 시드·같은 설정에서 Forge와 나란히 놓으면 눈으로 같은 그림(구도·색감)으로 보인다("Forge처럼" 켬).
- 팩 테스트(순수 로직)와 저장소 전체 테스트가 통과한다.
