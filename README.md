# UR_IV / AI Studio Pro

Windows용 AI 이미지 생성 데스크톱 앱이다. PyQt6 창 안의 `QWebEngineView` 가 Vue 3 SPA 를 띄우고,
두 층은 `QWebChannel` 로 통신한다. 생성·검색·마스킹 같은 무거운 작업은 Python 이 맡고 화면은 Vue 가 그린다.
Stable Diffusion WebUI(A1111/Forge) 나 ComfyUI 중 **한 번에 하나**의 백엔드에 연결해 쓴다. 직접 띄운 백엔드에 연결해도 되고,
앱이 Forge Neo·ComfyUI 를 직접 설치하고 실행하게 할 수도 있다. 이 밖에 Danbooru 태그 데이터셋 검색, 편집기, 일괄 처리,
로컬 LLM 대화, 영상·만화 제작 기능이 한 앱에 들어 있다.

---

## ✨ 핵심 기능

- **생성**: T2I 에는 Hires.fix 와 LoRA 스택을 붙일 수 있다. ADetailer(2슬롯)·SAM3 Mask(ControlNet 포함)·ANIMA 가이던스는 I2I·Inpaint 에도 적용된다.
  생성 엔진은 Standard 와 Krea2(ComfyUI 전용) 중에서 고른다. 시드 탐색(3×3)과 XYZ Plot 도 있다.
- **백엔드**: WebUI(A1111/Forge) 또는 ComfyUI 중 하나를 쓴다. 앱이 설치·실행하는 관리형 런타임을 쓰거나 기존 설치를 연결할 수 있다.
  ComfyUI 에서는 동봉 노드 팩이 설치돼 있으면 일반 작업에 워크플로 JSON 이 필요 없다.
- **Danbooru 데이터**: `2026_07` 릴리스 검색(G/S/Q/E, AND/OR)과 이벤트 시퀀스 생성(Event Gen)을 지원한다. 데이터는 첫 실행 때 자동으로 받는다.
- **프롬프트 도구**: 블록/텍스트 모드, Undo/Redo, 토큰 수, 한국어 라벨·한글 검색 자동완성, 제외 규칙 9종, 와일드카드, 즉석 와일드카드,
  조건부 규칙, 프리셋, 워크플로우 프로파일, A/B 테스트, 생성 통계를 제공한다.
- **AI 어시스트(Ollama)**: 태그 확장·유사 태그·자연어→태그, 자연어 캡션·장면 묘사·번역·창의 생성, AI 네거티브, 생성 시 태그를 자연어로 바꾸는 기능이 있다.
- **대화**: Ollama 또는 LM Studio 로 대화한다. JSON Schema 구조화 출력을 지원하고, 대화 중에 이미지·영상 생성을 요청할 수도 있다.
- **편집·일괄 처리**: 15개 도구 편집기, YOLO+SAM 자동 검열, 배경 제거(rembg), 워터마크를 쓸 수 있다.
  일괄 처리에는 리사이즈·업스케일·ADetailer·SAM3·캡션(CAFormer/ToriiGate) 모드가 있다.
- **Creator**: MiniMax H3 영상, AI 스토리보드 만화(리빙 코믹), Krea2 아이덴티티 편집을 만든다.
- **대기열·자동화**: 일시정지·재개, 위/아래 이동, 항목 편집, 다중 삭제, ETA 를 지원한다. 자동 모드는 매수·시간·무제한으로 돌고 재시도와 프롬프트 덱을 쓴다.
- **운영**: 검증된 모델 다운로드, GitHub 릴리스 기반 앱 업데이트, 외부 프로그램용 Generation API, 브라우저에서 쓰는 웹 모드를 지원한다.

---

## 📦 요구 사항

설치하기 전에 준비한다.

- **Windows 10/11**: 런처가 `.bat` 이고, 일부 패키지(`triton-windows`)가 Windows 전용이다.
- **Python 3.11**(py 런처 포함): 검증된 venv 는 3.11.9 다. `requirements.txt` 는 하한만 두지만, 검증 venv 에 설치된
  numpy 2.3·pandas 3.0·onnxruntime 1.29·av 18·rembg 2.0.78 은 Python 3.11 이상 전용이라 3.10 조합은 검증되지 않았다.
  런처는 3.11 → 3.10 → `py -3` → PATH 의 `python` 순서로 찾으므로, 3.11 이 없으면 다른 버전으로 venv 가 만들어질 수 있다.
- **NVIDIA GPU 권장**: 이미지 생성은 백엔드가 하지만, 앱 안의 SAM3·YOLO·배경 제거 같은 로컬 처리가 GPU 를 쓴다.
  GPU 유무와 상관없이 앱은 CUDA 빌드 torch 를 요구한다. `core/check_requirements.py` 가 cu128 인덱스에서 설치하고,
  설치에 실패하거나 CPU 빌드만 남으면 앱이 시작되지 않는다. 다른 CUDA 버전이 필요하면 그 파일의 `_TORCH_CUDA_INDEX` 를 바꾼다.
- **디스크**: 첫 실행에 venv 약 6 GB(그중 torch 약 4 GB)와 Danbooru 데이터 약 1.25 GB 를 내려받는다. 시간이 꽤 걸린다.
- **Git**: `git clone`, 관리형 백엔드, 앱 업데이트에 필요하다.
- **백엔드**: Forge/A1111 WebUI(`--api`) 또는 ComfyUI. **첫 실행 전에 하나를 띄워 두어야 한다**([처음 실행](#처음-실행-데스크톱) 참고).
- **관리형 런타임을 쓸 때**: `uv`(Python 3.13 을 스스로 받는다) 또는 py 런처로 찾을 수 있는 Python 3.13. 앱 자체는 3.11 에 그대로 둔다.
- **선택 사항**:
  - Ollama: AI 어시스트, ToriiGate·비전 캡션, 대화에 쓴다(기본 `http://localhost:11434`).
  - LM Studio: 대화 탭에서만 쓴다(기본 `http://localhost:1234`).
  - `uv`: 있으면 앱 패키지 설치에도 우선 쓴다.
  - **Node.js** 20.19+ 또는 22.12+(Vite 8 요구치. vitest 는 Node 23 을 지원하지 않는다): 프론트엔드를 다시 빌드·테스트할 때만 필요하다.

### Python 패키지

목록의 단일 출처는 [`requirements.txt`](requirements.txt) 이고, 앱 시작 때 `core/check_requirements.py` 가 누락분을 설치한다.
torch/torchvision 은 별도 인덱스라 `requirements.txt` 에 없다. SAM3(`sam3`, `triton-windows`), YOLO(`ultralytics`), CAFormer 태거(`onnxruntime`),
영상(`av`), 배경 제거(`rembg`, `pymatting`), VRAM 표시(`nvidia-ml-py`)는 이 목록에 들어 있다.

아래 패키지는 자동으로 설치되지 않는다. PATH 의 다른 Python 이 아니라 앱 venv 에 설치해야 한다(`git+` 주소는 Git 이 필요하다).

```bat
:: MobileSAM (가볍고 빠른 bbox 기반 SAM)
venv\Scripts\python.exe -m pip install git+https://github.com/ChaoningZhang/MobileSAM.git
:: SAM v1
venv\Scripts\python.exe -m pip install segment-anything
```

---

## 🚀 설치 / 실행

### 처음 실행 (데스크톱)

먼저 백엔드를 띄운다. Forge/A1111 은 `--api` 로, ComfyUI 는 평소대로 실행한다.
시작 화면(백엔드 선택)은 연결에 성공해야 닫히고 '백엔드 없이 시작' 버튼이 없다. 그래서 백엔드가 없으면 앱 안으로 들어갈 수 없다.
관리형 런타임은 앱에 들어간 뒤 설정 → **런타임 · 엔진** 에서 설치한다.

```bat
git clone https://github.com/UR-al/UR_IV.git
cd UR_IV
.\new_run_main_ui.bat
```

`.\` 를 붙이면 cmd 와 PowerShell 에서 모두 실행된다. 탐색기에서 더블클릭해도 된다.
첫 실행은 창이 뜨기 전에 콘솔에서 패키지를 설치한다. 이 콘솔을 닫으면 앱도 꺼진다.

1. `new_run_main_ui.bat` 는 `venv\` 가 없으면 만든다(`py -3.11` → `py -3.10` → `py -3` → PATH 의 `python` 순서로 찾는다).
   pip 를 준비하고 이전 크래시 로그를 보존한 뒤 `venv\Scripts\python.exe new_main_ui.py` 를 실행한다.
   기존 venv 가 망가져 실행되지 않으면 `venv.incompatible-…` 로 옮겨 두고 새로 만든다.
2. 앱이 시작되면 `core/check_requirements.py` 가 누락된 패키지를 설치한다. 먼저 CUDA 빌드 torch/torchvision(cu128 인덱스)을,
   다음으로 `requirements.txt` 의 나머지를 설치한다. `uv` 가 PATH 에 있으면 우선 쓴다. 필요한 패키지가 끝내 없으면 시작을 멈춘다.
3. 이어서 `core.fetch_data` 가 Danbooru 데이터(약 1.25 GB)를 Hugging Face 에서 받는다. 없는 파일과 크기가 다른 파일만 받는다.
4. 스플래시 다음에 백엔드 선택 화면이 뜬다. 주소를 확인하고 WebUI 또는 ComfyUI 를 고른다.
   ComfyUI 의 워크플로 칸은 비워 둬도 된다. 비우면 앱이 그래프를 직접 만든다. API 형식 JSON 을 지정하는 것은 고급 옵션이다.

`git clone` 으로 받은 사본만 앱 안에서 업데이트를 설치할 수 있다. 압축으로 받은 사본은 업데이트 알림만 받는다.

### 첫 이미지 만들기

1. **T2I** 탭 왼쪽 프롬프트 열의 `Checkpoint` 에서 모델을 고른다(목록은 연결된 백엔드가 알려 준다).
2. 캐릭터·작품·메인 등 프롬프트 칸에 태그를 쓴다. 해상도·스텝 등은 레일의 `파라미터` 에서 바꾼다.
3. `GENERATE IMAGE`(또는 `Ctrl+G`)를 누른다. 결과는 가운데 미리보기와 오른쪽 히스토리에 나오고, 기본 출력 폴더는 `generated_images/` 다.

### 실행 파일

| 파일 | 용도 |
|---|---|
| `new_run_main_ui.bat` | **평소에도 이것을 쓴다.** venv 생성·복구, pip 준비, 크래시 로그 보존 후 데스크톱 창을 띄운다. 앱 업데이트 후 재시작도 이 파일로 한다 |
| `run_gui.bat` | 같은 데스크톱 창을 띄우는 가벼운 런처. 이미 잘 동작하는 venv 가 있어야 하고, venv 를 만들거나 고치지 않는다 |
| `run_WEB_gui.bat` | 웹 모드(`web_main_ui.py`). 이미 만들어진 venv 가 있어야 한다 |

세 런처 모두 앱 업데이트가 진행 중이면 최대 120초 기다린다. 그래도 끝나지 않으면 안내를 띄우고 종료한다.
직접 실행하려면 `venv\Scripts\python.exe new_main_ui.py` 를 쓴다. 이 경우에도 의존성 확인과 데이터 준비를 똑같이 거친다.

### 데이터 다시 받기

명령 프롬프트(cmd)에서 저장소 폴더로 이동한 뒤 실행한다.

```bat
venv\Scripts\activate.bat
python -m core.fetch_data
```

처음 설치할 때 데이터를 받지 못하면 앱이 시작되지 않는다. 기존 데이터가 있으면 경고만 하고 그대로 시작한다.
저장소 접근이 거부되면 같은 창에서 `hf auth login` 을 실행한 뒤 다시 시도한다.

### 백엔드 연결

| 방법 | 설명 |
|---|---|
| 직접 실행 | Forge/A1111 WebUI 는 `--api` 로 띄운다(기본 연결 주소 `http://127.0.0.1:7860`). ComfyUI 의 기본 연결 주소는 `http://127.0.0.1:8188` 이다. 다른 포트(ComfyUI 데스크톱 앱 등)로 띄웠다면 선택 화면에서 주소를 고친다 |
| 관리형 런타임 | 설정 → **런타임 · 엔진** 에서 Forge Neo(`Haoming02/sd-webui-forge-classic`, `neo` 브랜치)나 ComfyUI(`Comfy-Org/ComfyUI`)를 설치·업데이트·시작·중지한다 |
| 기존 설치 연결 | 같은 화면에서 설치 폴더를 지정한다. Forge 는 `launch.py` 와 `venv`/`.venv` 가 있어야 한다. ComfyUI 는 `main.py` 와 `venv`/`.venv` 가 있거나 Windows 포터블 구성(`python_embeded`)이어야 한다. 연결한 설치는 앱이 업데이트하지 않는다 |

- 시작한 뒤 백엔드를 바꾸려면 설정 → **일반**(또는 **네트워크**)의 `백엔드 관리` 를 누른다.
- 관리형 설치 위치는 `user_data/managed_backends/<engine>/` 이다(`AISTUDIO_MANAGED_BACKENDS_DIR` 로 바꿀 수 있다).
  엔진마다 venv 를 따로 만든다. `uv` 가 있으면 `uv venv --python 3.13`, 없으면 `py -3.13` 을 쓰고, 둘 다 없으면 앱의 Python 으로 만든다. Git 이 PATH 에 있어야 한다.
- 관리형 포트는 Forge 17860, ComfyUI 18188 부터 시작해, 50개 범위 안에서 비어 있는 첫 포트를 쓴다.
- 설치된 관리형 런타임에 자동 시작이 켜져 있으면 데스크톱 시작 때 백엔드 선택 화면을 건너뛰고 그 런타임을 띄운다.
- **ComfyUI 동봉 노드 팩**: 앱이 만드는 기본 그래프는 [`comfy_custom_nodes/ai_studio_forge_parity`](comfy_custom_nodes/ai_studio_forge_parity/README.md) 의 노드를 쓴다.
  이 팩이 있으면 일반 T2I/I2I/인페인트/업스케일/ADetailer/SAM3 작업에 워크플로 JSON 이 필요 없고, 없으면 일반 T2I 도 실패한다.
  - 앱에 설정된 로컬 ComfyUI 런타임에서는 처음 생성할 때(워크플로 컴파일 직전) `custom_nodes` 로 자동 복사된다.
    조건은 연결 주소가 loopback 이고 설정된 런타임과 포트가 같을 것(런타임 주소가 없으면 생략), 런타임의 확장 폴더에 쓰기가 승인돼 있을 것이다.
    앱이 띄운 ComfyUI 는 앱이 재시작하고, 외부에서 띄운 ComfyUI 는 직접 재시작하라고 안내한다.
  - 직접 띄운 ComfyUI 에는 자동으로 들어가지 않는다. 이 폴더를 `ComfyUI/custom_nodes/` 에 복사하고 ComfyUI 를 재시작하거나,
    그 설치를 설정 → **런타임 · 엔진** 에서 기존 설치로 연결하고 확장 폴더 쓰기를 승인한다.
- **확장 요구**: WebUI 경로의 ADetailer 는 Forge/A1111 쪽에 ADetailer 확장이 있어야 한다. ComfyUI 의 ADetailer 는 Impact Pack(FaceDetailer)과
  Impact Subpack(UltralyticsDetectorProvider)이 필요하다. 생성 중 SAM3 Mask 는 Forge 쪽에 `sam-extra`(forge_sam3_extension) 확장이 필요하고,
  ANIMA 가이던스는 그 확장 v0.21.2 이상이 필요하다.
- **Krea2**: ComfyUI 에서만 동작한다. I2I 아이덴티티 편집에는 Identity Edit·TextFusion LoRA 와 Krea2 Edit 커스텀 노드가 추가로 필요하다.
- **Generation API**: 외부 프로그램에서 생성 작업을 받거나 승인된 다른 Forge/ComfyUI 로 넘기려면 설정 → **네트워크** 에서 켠다.
  기본은 꺼져 있고, 켜면 `127.0.0.1:17990` 에서 Bearer 토큰으로 인증한다. A1111 API 일부와 호환된다. 규격과 예시는 [`docs/generation_api.md`](docs/generation_api.md) 를 참고한다.

### 모델 넣을 곳

- **직접 띄운 백엔드**는 그 백엔드 자신의 models 폴더를 쓴다.
- **관리형 런타임과 모델 다운로드**는 공용 모델 루트를 쓴다. 루트는 다음 순서로 정한다.
  1. 환경 변수 `FORGE_MODELS_ROOT`
  2. `C:\sd-webui-forge-neo\models` 또는 `C:\sd-webui-forge-classic\models` 가 있으면 그 폴더
  3. 둘 다 없으면 `user_data\models\` (앱이 `Stable-diffusion`, `diffusion_models`, `Lora`, `VAE`, `text_encoder`, `upscale_models` 하위 폴더를 만든다)
- 이 폴더들은 관리형 Forge 에는 `--ckpt-dirs`·`--lora-dirs` 같은 인자로, 관리형 ComfyUI 에는 추가 모델 경로 파일로 전달된다.
- 설정 → **Forge** 에서 체크포인트·LoRA·VAE·텍스트 인코더 폴더를 따로 지정할 수 있다(`config/forge_model_paths.json`).

**편집기·SAM 모델**

- **`Editor_models/`**(대문자 E, 프로젝트 루트): 편집기 자동 검열용 폴더다. 새 clone 에는 없고, 앱이 처음 필요할 때 만든다(직접 만들어도 된다).
  - YOLO 검출 모델(`.pt`/`.onnx`/`.safetensors`)은 직접 준비해 넣으면 자동으로 인식되고, `Editor_models/yolo_config.json` 으로 추가·비활성화할 수 있다.
  - SAM 체크포인트도 여기에 둔다. `sam3.pt` 는 Hugging Face `facebook/sam3`(접근 승인 필요), `mobile_sam.pt` 는 MobileSAM 저장소에서 받는다.
    SAM3 는 `.pt` 만 받는다(`.safetensors` 불가).
  - SAM 선택이 `AUTO` 면 MobileSAM → SAM3 → FastSAM → SAM v1 순서로 파일을 고른다.
    고른 파일의 패키지가 없으면 SAM3 로 넘어가지 않고 YOLO 박스 마스크를 쓴다. 이럴 때는 `SAM3` 를 직접 선택한다.
- **SAM3 어휘 파일**(`bpe_simple_vocab_16e6.txt.gz`)은 처음 필요할 때 `facebook/sam3` 에서 `image_cache/sam3_assets` 로 자동으로 받는다.
- **생성 중 SAM3 Mask**(Forge/ComfyUI)의 체크포인트 목록은 `Editor_models` 가 아니라 위 공용 모델 루트의 `sam3` 폴더에서 읽는다.
- **캡션**: CAFormer(`animetimm/caformer_s18.dbv4-full`)는 자동으로 받지 않는다. 로컬 HF 캐시에 있거나 직접 고른 폴더에 있어야 한다.
  ToriiGate 는 Ollama 모델 `hf.co/DraconicDragon/ToriiGate-0.5-GGUF:BF16` 로 쓴다.

### 웹 모드

`run_WEB_gui.bat` 는 창을 띄우지 않고 `frontend_dist` 를 HTTP 로 서비스한다.
이 런처는 venv 를 만들지 않으므로, 먼저 `new_run_main_ui.bat` 로 한 번 실행해 venv 를 만들고 백엔드·런타임 설정을 끝내 둔다.

- 기본 주소는 `http://127.0.0.1:7800`(WebSocket 7801)이다. 포트는 `AISTUDIO_HTTP_PORT` / `AISTUDIO_WS_PORT` 로 바꾼다.
- 실행할 때마다 새 세션 토큰이 든 주소(`…/?token=…`)를 콘솔에 출력하고 기본 브라우저로 연다(토큰은 그 실행 동안 유효하다).
  첫 접속 때 세션 쿠키가 설정되고, 쿠키가 없는 요청은 401 이 된다.
- 실행할 때마다 호스트 PC 화면에 '백엔드 선택' 창이 뜬다. 여기서 백엔드를 고른다('백엔드 없이 시작'도 있다).
  브라우저에서는 백엔드를 바꿀 수 없고, 관리형 런타임 자동 시작도 건너뛴다.
- LAN 에 공유하려면 바인드 주소를 지정한 **같은 콘솔 창에서** 런처를 실행한다. cmd 에서는 `set "AISTUDIO_BIND=0.0.0.0"`,
  PowerShell 에서는 `$env:AISTUDIO_BIND="0.0.0.0"` 을 입력한 뒤 `.\run_WEB_gui.bat` 를 실행한다(더블클릭하면 이 값이 전달되지 않는다).
  Windows 방화벽에서 HTTP·WebSocket 포트 두 개(기본 7800·7801)를 허용해야 한다.
- 모델 다운로드, 조명 편집·손 재구성, 런타임 설치·시작, 앱 업데이트 설치, Generation API·모델 경로 변경, 설정 복원·앱 재시작처럼
  호스트 전용인 기능은 웹 모드에서 막힌다. 파일·폴더 대화상자는 LAN 에 공유할 때만 막힌다.
- 레일의 **Web**·**Backend** 탭은 데스크톱 창 전용이라 웹 모드에서는 눌러도 브라우저에 아무것도 나오지 않는다.
- 크래시 로그는 `logs/last_crash_web.log` 에 남는다.

### 앱 업데이트

설정 → **앱 업데이트** 는 GitHub Releases(`UR-al/UR_IV`)의 정식 `vX.Y.Z` 릴리스를 확인한다.
자동 확인(기본 켜짐)은 12시간마다 한다. 설정 화면에서는 자동 확인 켜기/끄기와 버전 건너뛰기만 할 수 있다.
간격은 `config/app_update.json` 의 `intervalHours` 를 직접 고쳐야 바뀐다(1–168로 제한).

- 설치 조건: 원격이 `UR-al/UR_IV` 인 git 사본이어야 하고, 브랜치에 체크아웃돼 있어야 하며(detached HEAD 불가), 현재 버전의 기준 릴리스를 식별할 수 있어야 한다.
- 설치는 `git merge --ff-only` 로만 하며 reset 이나 stash 는 하지 않는다. 바뀐 파일과 겹치는 로컬 수정이 있거나 다른 앱 인스턴스가 실행 중이면 거부한다.
- 끝나면 `new_run_main_ui.bat` 로 다시 시작하고, 결과는 `logs/updates/last_result.json` 에 남는다.
  새로 필요해진 패키지는 재시작 때 `core/check_requirements.py` 가 설치한다. 현재 버전은 `VERSION` 파일에 있다.

### 모델 다운로드

설정 → **모델 다운로드** 는 `core/model_download_catalog.json` 에 등록된 팩을 받는다.
팩은 Krea 2(t2i/edit/hires), MiniMax H3(t2v/i2v/turbo/ref2va/audio), Anima(base/2.9B/3.8B + Semantic Connector v2) 11개다.
R-ESRGAN 4x+(`RealESRGAN_x4plus.pth`)는 Krea 2 Hires 팩에 함께 들어 있다.
모든 파일의 크기와 SHA-256 을 검증하고, 끊기면 이어 받는다. 파일은 주 엔진의 모델 폴더에 저장되고, 없으면 `user_data/models/…` 에 저장된다.
데스크톱 창에서만 쓸 수 있다.

### 프론트엔드를 고쳤을 때

`frontend_dist/` 가 커밋돼 있으므로 실행만 할 때는 Node 가 필요 없다. `frontend/src` 를 고쳤다면 다시 빌드한다.

```bat
cd frontend
npm install
npm run build
```

Vue 는 다시 빌드해야 반영된다. Python 코드를 고쳤다면 앱을 재시작한다. 자세한 명령은 [`frontend/README.md`](frontend/README.md) 에 있다.

---

## 🗂️ 탭 가이드

왼쪽 세로 레일(NavRail)에 16개 탭이 네 그룹으로 나뉘어 있다. 레일은 196px 와 52px 사이에서 접을 수 있다.
탭에 세부 모드가 있으면 활성 탭 아래 **서랍 항목**으로 나타난다. 예를 들어 T2I 의 `프롬프트 | 파라미터`, Batch 의 `일괄 | 업스케일 | …` 이 그렇다.

| 그룹 | 탭 | 설명 |
|---|---|---|
| 생성 | **T2I** | 텍스트→이미지. 레일의 `프롬프트`/`파라미터`로 왼쪽 열 내용을 바꾼다(파라미터 열은 `Esc`나 X로 닫는다). 가운데에는 실시간 미리보기, 진행률, 해상도·시드, `시드 탐색`, Positive/Negative/Parameters 바가 있다. 오른쪽 히스토리의 우클릭 메뉴에는 즐겨찾기, I2I/인페인트/에디터로 보내기, 비교, ADetailer, 프롬프트 당겨오기, 다음 큐에 추가, 경로 복사, 휴지통 이동이 있다. 하단에는 GENERATE, 취소, 자동, 무작위, `생성 시 태그를 자연어로 변환` 이 있다 |
| 생성 | **I2I** | `img2img` 모드: 원본 업로드·드래그, 프롬프트 덮어쓰기(비우면 T2I 프롬프트 사용), denoise, 크기 조정, 고급(steps/CFG/seed), 실험 · 조명 편집. 생성 엔진이 Krea2 면 아이덴티티 편집으로 바뀐다. `SAM3 정밀화` 모드: Target/Exclude/Replacement 를 지정해 SAM3 마스크로 img2img 하고, 연쇄 정밀화도 된다 |
| 생성 | **Inpaint** | 마스크 편집기. 사각 선택(M), 올가미(L, 자유/자석), 마스크 브러시(B), 지우개(E, 브러시/박스/올가미 모양)를 쓴다. 브러시 크기, denoise, 프롬프트 덮어쓰기, 마스크 blur·padding, 고급 설정, 마스크 Undo/Redo 가 있다. 기본으로 꺼진 [실험 · 손 형태 재구성](docs/experimental-hand-reconstruction.md) 패널도 여기 있다 |
| 생성 | **Event Gen** | 캐릭터·작품·일반 태그·작가·제외 태그·스텝 수·등급으로 이벤트를 찾는다. 결과는 이벤트 목록과 단계별 시퀀스로 보인다. 외모/의상/배경 유지, 단계별 T2I 보내기, 반복 횟수, 큐에 추가, 지금 생성, `.parquet` 가져오기/내보내기를 지원한다 |
| 생성 | **Search** | Danbooru 데이터셋을 캐릭터/작품/일반 태그/작가 필드로 검색한다(필드마다 제외 입력). 등급 칩, 50만 건 상한 또는 무제한, AND/OR, `.parquet` 가져오기/내보내기, 필터 관리, 정밀 검색이 있다. 단일 보기에서는 `프롬프트로 사용`/`대기열에 추가`를 쓰고, 목록 보기는 정렬과 페이지(50개)를 지원한다 |
| 생성 | **XYZ Plot** | 연결된 백엔드가 알려준 축을 쓴다(`축 새로고침`). 축 값은 선택, S/R, 범위(`20-40:5`)로 넣는다. 최대 256 조합이고 CSV 로 내보낸다. 앱이 실행하지 않는 확장 축은 따로 표시한다. Krea2 는 지원하지 않는다 |
| 생성 | **Creator** | `영상`: MiniMax H3 T2V/I2V/V2V, 24 FPS 고정, 오디오 유지·생성. `만화`: AI 스토리보드, 최대 6컷, Auto/Grid/Vertical/Strip/Hero 레이아웃, 말풍선을 쓰고 PNG/WebP 나 리빙 코믹(영상)으로 내보낸다. `Krea2`: 원본과 참조로 아이덴티티를 편집한다. 영상과 Krea2 는 ComfyUI 가 필요하다 |
| 생성 | **대화** | Ollama 또는 LM Studio 로컬 LLM 과 대화한다. 스레드(새 스레드 `Ctrl+N`), 지침 프리셋, JSON Schema 구조화 출력, 모델별 옵션, 이미지·텍스트 첨부, Markdown 내보내기를 지원한다. 요청 모드는 자동/대화만/이미지 생성/영상(H3) 이다 |
| 편집 | **Editor** | 생성 없이 이미지를 편집한다. 도구 15개(마스크 도구와 그리기 레이어 도구), `보정`(밝기·대비·필터, 히스토그램·레벨·커브), `효과`(모자이크·검은 띠·블러, AI 검열, 배경 제거, 워터마크), `변형`(회전·뒤집기, 자르기, 크기 변경, 원근 보정, 선택 영역 이동 후 인페인트로 보내기) 탭이 있다. 복구용 자동 저장은 5분마다 한다 |
| 편집 | **Batch / Upscale** | 왼쪽은 설정, 오른쪽은 파일 그리드다. 모드는 5가지다. `일괄`: 리사이즈·PNG/JPEG/WEBP 변환. `업스케일`. `ADetailer`: ETA·중지. `SAM3`: 인페인트 또는 마스크만. `캡션`: CAFormer 태그, ToriiGate 자연어, 둘의 조합, 다른 Ollama 비전 모델 중에서 골라 `.txt` 를 저장한다 |
| 라이브러리 | **Gallery** | 폴더 기반으로 이미지·영상·오디오를 보여 준다(스크롤할 때 이어서 불러온다). EXIF 검색, 날짜/이름 정렬, 썸네일 크기 조절이 된다. 큰 보기에는 이름 변경, EXIF 저장, I2I/인페인트/에디터, `프롬프트 사용` 이 있고, 메타 사이드바에는 `T2I에서 사용` 이 있다. 우클릭 메뉴는 즐겨찾기, 정보 보기, I2I/인페인트/에디터로 보내기, 비교, ADetailer, 휴지통 이동이다 |
| 라이브러리 | **Favorites** | 즐겨찾기 그리드. EXIF 검색, 정렬, 썸네일 크기를 지원한다. 뷰어에서는 항목별 복사(프롬프트/네거티브/원본/파라미터)와 T2I/I2I/Inpaint/Editor 전송 카드를 쓴다 |
| 라이브러리 | **PNG Info** | `PNG Info` 모드: Prompt/네거티브/Parameters/Raw 를 항목별로 복사하고 ComfyUI 메타데이터도 본다. 전송 카드(T2I 프롬프트, 즉시 생성, I2I, Inpaint, Editor, 즐겨찾기)가 있다. `메타 이식` 은 이 이미지의 프롬프트·파라미터·워크플로를 다른 이미지에 새 PNG 로 써 넣는다. `비교` 모드: 두 이미지를 슬라이더로 비교하고, 파라미터 차이·프롬프트 차이를 보며, GIF 로 내보낸다 |
| 시스템 | **Settings** | 14개 섹션(아래 표). 설정 화면에 포커스가 있을 때 `Ctrl+F` 로 검색하고 일치 항목을 강조한다 |
| 시스템 | **Web** | 내장 브라우저(PyQt 네이티브 화면, 데스크톱 창 전용). 뒤로·홈·주소창·`← AI Studio` 가 있다. 기본 홈은 `hijiribe.donmai.us` 이고 로그인 상태가 유지된다 |
| 시스템 | **Backend** | 연결된 Forge/ComfyUI 의 웹 UI 를 앱 안에 띄운다(PyQt 네이티브 화면, 데스크톱 창 전용). 새로고침과 외부 브라우저로 열기를 지원하며, 탭을 열 때 처음 불러온다 |

**공통 화면 요소**
- 데스크톱에서 시작하면 전체 화면 **백엔드 선택**(WebUI / ComfyUI, URL 확인, ComfyUI 워크플로 선택)이 먼저 뜬다. 연결에 성공해야 닫힌다. 관리형 런타임에 자동 시작이 켜져 있으면 이 단계를 건너뛴다.
- 하단 **상태 줄**에는 백엔드·VRAM·모델이 보인다. VRAM 을 누르면 모델 언로드를 요청하고, 위험 수준이면 먼저 확인을 묻는다.
- **대기열 서랍**: 시작, 일시정지/재개, 중지, 다중 선택 삭제, 전체 비우기, 위/아래 이동, 편집을 지원한다. 생성 중인 항목은 옮기거나 지울 수 없다.
- **탭 순서**: 설정 → 워크스페이스에서 드래그로 바꾸며, 순서는 그룹 안에서만 적용된다. 목록에 없는 Editor·Creator·대화·Web·Backend 는
  사용자 순서가 저장되면 각 그룹의 끝에 원래 순서대로 붙는다(예: Editor 가 Batch / Upscale 아래로 간다).

### 설정 화면 구성

| 섹션 | 내용 |
|---|---|
| 일반 | 시스템 상태·`백엔드 관리`, 미리보기 썸네일, VRAM(`생성 후 모델 언로드`, 기본 꺼짐), Comfy SAM3 RAM 유지, Forge 출력 저장 |
| 앱 업데이트 | 업데이트 확인·설치 후 재시작(위 [앱 업데이트](#앱-업데이트)) |
| 네트워크 | Generation API, Forge/WebUI·ComfyUI 연결, 외부 생성 작업 |
| 런타임 · 엔진 | 관리형/기존 설치 백엔드, 공유 모델 소스, 확장 폴더, 추가 인자, 설치·업데이트·시작·중지, 확장 저장소 설치 |
| 모델 다운로드 | 검증된 모델 팩 다운로드, H3 캐시, Spectrum 가속(실험), Comfy 호환성·워크플로 설정 |
| Forge | Forge Neo 모델 경로 |
| 로직 | 프롬프트 정리 옵션, 태그 블록 모드, 갤러리 메타 패널, 자동 작품 태그, 히스토리 깜빡임, 전역 저장 |
| 워크스페이스 | 탭 순서 |
| 테마 | 테마 선택 |
| 단축키 | 단축키 안내, 히스토리 점프 수식키(Shift/Ctrl/Alt) |
| 가드 | ANIMA 해상도 가드 |
| 기본값 | UI 배율, T2I 기본값, EDITOR 기본값(브러시·효과 세기·YOLO 신뢰도·스냅 반경), 첫 실행 확장 기본값(Hires.fix·ADetailer·SAM3) |
| 데이터 · 백업 | 설정·사용자 자료 백업 |
| AI 어시스트 | Ollama 서버·모델, 비전 모델 추천, `이미지 생성 시 LLM 언로드`, 기능별 지침([설명](docs/ai-assist-instructions.md)) |

---

## 🏗️ 아키텍처

```
new_main_ui.py / web_main_ui.py
  └─ core.app_startup.prepare_application()   의존성 확인 → 데이터 준비

QMainWindow  GeneratorMainUI (ui/generator_main.py + ui/generator_*.py·일부 ui/*_actions.py 믹스인)
└── QStackedWidget (_main_stack)
    ├── 0: QWebEngineView ─ Vue SPA (frontend_dist/index.html)
    ├── 1: BrowserTab   (Web,     tabs/browser_tab.py)
    └── 2: BackendUITab (Backend, tabs/backend_ui_tab.py)

QWebChannel 객체
├── backend : ui/vue_bridge.py (VueBridge — @pyqtSlot / pyqtSignal)
└── studio  : ui/studio_qwebchannel.py → core/studio_application.py
              (런타임 · Generation API · 앱 업데이트 · 모델 경로 작업)

Vue SPA (frontend/src)
├── App.vue             백엔드 선택 · NavRail · 왼쪽 열(PromptPanel ↔ 파라미터 카드)
│                       · 가운데 router-view · 오른쪽 히스토리 · 대기열 · 상태 줄 · 모달
├── router.js           T2I = components/ImageViewer.vue, 나머지 = views/*View.vue 지연 로드
├── components/params/  파라미터 카드 (기본 · Hires.fix · 프롬프트 필터 · ADetailer · SAM3 Mask · LoRA)
├── components/managers/ 매니저 모달 (프리셋 · 가중치 · 와일드카드 · 프로파일 · 순서 · 즉석 WC · 통계)
├── composables/        use*.ts/js — 기능별 상태
├── stores/             widgetStore.js (위젯 값 동기화 + requestAction), appUpdateStore.ts
├── studio/             studio 채널 클라이언트
└── bridge.js           연결 (Qt 내장 / 웹 모드 WebSocket / 개발용 mock)

Python
├── backends/  WebUIBackend (A1111/Forge, /sdapi/v1) · ComfyUIBackend (페이로드 → API 그래프 컴파일)
├── workers/   QThread 워커 (생성 · 검색 · 이벤트 · ADetailer · SAM3 · 정밀화 · 업스케일 · 일괄 · 대화 · Ollama)
└── core/      Qt 없는 순수 로직 (대기열, 프롬프트 파이프라인, 런타임 관리, 데이터셋, 저장 경로 …)
```

- **Vue → Python**: `requestAction(name, payload)` 가 `backend.onAction` 을 거쳐 `GeneratorMainUI._handle_vue_action` 으로 간다.
- **Python → Vue**: `VueBridge` 시그널을 Vue 에서 `onBackendEvent(name, cb)` 로 구독한다.
- **위젯 프록시**: Python 코드는 `ui/widget_proxies.py` 의 프록시로 Vue 상태를 읽고 쓴다.
  Vue 의 `widgets.<id>` 키와 프록시 `widget_id` 가 같다(예: `widgets.character_input` ↔ `LineEditProxy(b, 'character_input')`).
- **Web·Backend 탭**은 Vue 뷰가 아니라 PyQt 화면이다. 레일에서 누르면 `native_tab_switch` 로 스택을 바꾼다.
- **웹 모드**는 창 없이 같은 `GeneratorMainUI` 를 쓴다. 허용 목록만 노출하는 `WebBridgeFacade` 가 WebSocket 으로 연결되고, 호스트 전용 작업은 막힌다.
- 숨겨진 PyQt `Img2ImgTab` 은 I2I 생성 경로와 설정 저장에, `InpaintTab` 은 설정 저장에만 아직 쓰인다.
  `UpscaleTab` 은 만들어지기만 하고 쓰이지 않는다(업스케일은 `ui/upscale_actions.py`).

---

## 📊 데이터셋과 검색 문법

### 데이터 파일

```
danbooru_optimized/
├── README.md                                 # 데이터 생성·검증 방법 (git 추적)
├── dataset_manifest.json                     # 활성 릴리스 2026_07 정의 (git 추적)
├── legacy_search_tags_before_2026_07.csv     # 구형 릴리스에만 있던 태그 보존 (git 추적)
├── danbooru_2026_07_{g,s,q,e}.parquet        # Search 등급별 shard (약 965 MB)
├── danbooru_sorted/danbooru_{g,s,q,e}.parquet  # Event Gen 그래프 shard (약 273 MB)
└── tags_dictionary.parquet                   # 영어 자동완성 사전
```

- `*.parquet` 는 git 에 없다. 앱이 시작할 때 Hugging Face 데이터셋 `UR-AR/UR_IV` 에서 없는 파일과 크기가 다른 파일만 받는다.
- Search 는 manifest(`format_version` 1)가 지정한 파일 이름만 쓰고, 크기·행 수·스키마·SHA-256 을 검증한다. 구형 릴리스로 넘어가거나 파일 이름을 추측하지 않는다.
- Event Gen shard 크기가 manifest 와 다르면 경고만 띄우고 막지는 않는다.
- `legacy_search_tags_before_2026_07.csv` 는 구형 `2025`·`2026`·`2026_06` 릴리스에만 있던 태그를 보존한다(열: `tag`, `legacy_categories`, `legacy_releases`). Search shard 를 대신하지는 않는다.
- 데이터 재생성과 검증 방법은 [`danbooru_optimized/README.md`](danbooru_optimized/README.md) 에 있다.
- **`tags_db/`** 는 git 에 들어 있는 태그 DB 다(약 39 MB). 자동완성 카탈로그, 한국어 태그 카탈로그, 별칭·함의, 캐릭터 데이터, 어휘 목록을 담고, 코드는 `manifest.json` 을 거쳐서만 읽는다. 자세한 내용은 [`tags_db/README.md`](tags_db/README.md) 에 있다.

### 자동완성

태그 입력칸은 영어 2글자, 한글 1글자부터 후보를 보여 준다.
- 영어는 `tags_dictionary.parquet`(별칭은 `tags_db`)를 쓴다. 카테고리 순서(general → character → copyright → artist → meta)대로,
  카테고리 안에서는 사용 수 순으로 접두 일치를 찾고, 이어서 별칭 → 일반 태그 포함 일치 순으로 최대 10개를 보여 준다. 각 후보에 한국어 라벨을 붙인다.
- 한글은 한국어 카탈로그의 **키워드**로 찾는다. 예를 들어 `장발` 을 입력하면 `long_hair`, `very_long_hair` 가 나온다.

### Search 탭 검색 문법

검색 필드는 캐릭터·작품·일반 태그·작가 네 가지이고, 필드마다 제외 입력칸이 있다.
검색어는 소문자로 바꿔 비교하며, 공백과 밑줄(`_`)은 같은 것으로 취급한다.

| 문법 | 의미 | 예 |
|---|---|---|
| `word` | 포함 | `girl` → `1girl`, `multiple girls` |
| `*word` | 완전 일치 | `*1girl` → 정확히 `1girl` 만 |
| `_word` | 문자열 끝 일치 | `_hair` → `long hair`, `short hair` (문자열 기준이라 `armchair` 도 걸린다) |
| `word_` | 문자열 시작 일치 | `hair_` → `hair ornament`, 그리고 `hair` 자체 |
| `_word_` | 명시적 포함 | `word` 와 같다 |
| `[A\|B\|C]` | OR 그룹 (모드 무관) | `[blue_hair\|red_hair]` |
| `[A,B,C]` | AND 그룹 (모드 무관) | `[1girl,blue_hair]` |
| `[A\|B\|]` | **필드 와일드카드**: 끝이 빈 칸이면 이 필드는 항상 통과 | `[tots\|alc\|]` → tots, alc, 또는 아무 작품. 중간의 `\|\|` 나 맨 앞의 `\|` 는 무시한다. AND 모드용이다. OR 모드에서 쓰면 필드끼리 OR 로 묶여 모든 결과가 통과한다 |
| `,` | 모드에 따라 AND 또는 OR | AND: `a, b` = 둘 다 / OR: 둘 중 하나 |

- **AND/OR 모드**는 필드끼리의 결합과 한 필드 안의 콤마를 함께 정한다.
- **제외 입력**은 모드와 상관없이 결과에서 뺀다. 다만 제외칸 안의 콤마도 모드를 따르므로,
  AND 모드에서 `a, b` 를 넣으면 둘 다 가진 결과만 빠진다. 하나라도 가진 결과를 빼려면 `[a|b]` 로 쓴다.
- **등급 칩**은 GEN/SENS/QUES/EXPL 이다(기본은 GEN 만 켜짐).
- **상한**: 결과가 50만 건을 넘으면 무작위로 뽑는다. `무제한 ⚠` 으로 상한을 끌 수 있다.
- **정밀 검색**은 위 문법이 아니라 태그 단위 완전 일치로 동작한다. 포함 조건은 모두 만족해야 하고, 제외 조건은 하나만 걸려도 빠진다.

---

## ✍️ 프롬프트 문법

### 제외 규칙 (9종)

PromptPanel 의 `제외 (로컬)` 칸에 쓰고, `관리` 창에서 무엇이 걸리는지 미리 볼 수 있다.

| 제외 | 예외 (유지) |
|---|---|
| `단어` 포함 제외 | `~단어` 완전일치 유지 |
| `*단어` 완전일치 제외 | `~_단어` 접미 유지 |
| `_단어` 접미 제외 | `~단어_` 접두 유지 |
| `단어_` 접두 제외 | `~_단어_` 포함 유지 |
| `_단어_` 포함 제외 |  |

- 예외 규칙을 먼저 본다. 비교할 때 밑줄은 공백으로, 대소문자는 소문자로 맞춘다.
- 이 규칙은 **데이터 태그를 프롬프트 칸에 채울 때**만 적용된다. Search 결과 적용, 랜덤 덱·자동화, 대기열 추가, 프롬프트 당겨오기가 그런 경우다. 직접 입력한 프롬프트에는 생성할 때 적용하지 않는다.
- 캐릭터·작품·작가 칸에서는 `단어`, `_단어_` 가 태그 전체와 같을 때만 걸린다. 접두·접미 규칙은 그대로 적용된다.
- 규칙은 콤마로 나눈다. 콤마가 없으면 공백으로 나누고 밑줄을 공백으로 바꾸므로 `_단어`·`단어_`·`~_단어` 같은 접두/접미 규칙이 깨진다.
  규칙이 하나뿐이면 항상 끝에 콤마를 붙인다(예: `short_,`).
- `config/default_excludes.txt` 는 카테고리별로 정리한 **참고용** 목록이다. 앱이 읽거나 적용하지 않으니 필요한 줄을 제외 칸에 직접 붙여 넣는다.

### 와일드카드

스튜디오 도구 → **와일드카드** 에서 파일을 관리하고 프롬프트에 넣는다. 같은 창의 `생성 시 치환` 스위치로 켜고 끈다(기본 켜짐).

| 문법 | 의미 |
|---|---|
| `__name__` / `~/name/~` | 같은 의미. `wildcards/name.txt` 의 **모든 줄**에서 후보 하나씩을 골라 `, ` 로 잇는다 (한 줄 = 콤마로 나열한 후보) |
| `__name:n__` / `~/name:n/~` | 무작위 n 줄만 골라 쓴다 |
| `[A\|B\|C]` | 프롬프트 어디서든 하나를 무작위로 고른다 |
| `{A\|B\|C}` | 무작위 선택. 중첩할 수 있다(`{{red\|blue} hair\|ponytail}`) |
| `{A:3\|B:1}` | 가중치 선택 |
| `{1-10}` / `{1-10:2}` | 범위 안의 숫자 (`:2` 는 간격) |

- 빈 줄과 `#` 으로 시작하는 줄은 주석이라 건너뛴다. 와일드카드는 10단계까지 중첩되고, 파일이 없으면 토큰을 그대로 둔다.
- 앞뒤가 공백·밑줄이거나 콤마가 든 이름은 `__name__` 으로 쓸 수 없다. 이런 이름은 관리 창이 `~/name/~` 로 넣어 준다.
- 스위치가 켜져 있으면 생성할 때 긍정·부정 프롬프트 모두에서 파일 와일드카드를 먼저 풀고, 이어서 `{…}` 를 푼다.

### 즉석 와일드카드

스튜디오 도구 → **즉석 WC** 에서 이름별 후보 목록을 만들고 `$$name$$` 으로 쓴다(`user_data/instant_wildcards.json` 에 저장된다).
- 생성할 때마다 한 줄을 고른다. `{3}:tag` 처럼 줄 앞에 가중치를 붙일 수 있다.
- 8단계까지 중첩되고, 없는 이름은 그대로 둔다.
- `생성 시 치환` 스위치와 상관없이 항상 풀린다. 풀린 다음에는 중복 태그를 정리한다.

실시간 프롬프트 정리(밑줄→공백 등)는 `$$name$$`, `~/name/~`, `__name__` 토큰을 건드리지 않는다.

### 조건부 규칙

스튜디오 도구 → **조건부** 에서 IF→THEN 규칙을 관리한다(`config/cond_rules.json`, 전체 켜기/끄기).

| 항목 | 값 |
|---|---|
| 조건 | 태그 목록(콤마 = 모두 있어야 함), `있으면` / `없으면` |
| 대상 | 추가·제거·대체할 태그(콤마 목록) |
| 동작 | 긍정: 추가 / 제거 / 대체, 부정: 추가 / 제거 |
| 위치 | 본문 / 선행 / 후행 / 바로 뒤(긍정 규칙만) |

- 조건은 긍정 칸(캐릭터·작품·접두·메인·접미)에서만 확인한다. 태그는 소문자·밑줄→공백·괄호 이스케이프 해제로 맞춰 비교한다.
- 추가는 이미 있는 태그를 다시 넣지 않고, 제거는 모든 긍정 칸에서 뺀다. 대체는 조건 태그를 대상으로 바꾸며 `없으면` 규칙에는 쓰이지 않는다. 한 번 적용할 때 규칙끼리 연쇄되지 않는다.
- 규칙은 데이터 태그를 프롬프트 칸에 채울 때 적용되고(와일드카드를 켰으면 풀린 뒤에 한 번 더), 생성 직전의 최종 프롬프트에는 적용되지 않는다.
- 캐릭터 특징 프리셋에도 캐릭터별 규칙을 넣을 수 있다.

---

## ⌨️ 단축키

백엔드 선택 화면이 떠 있는 동안에는 전역 단축키가 모두 꺼진다.

| 키 | 동작 |
|---|---|
| `Ctrl+G` | 생성 |
| `Ctrl+S` | 설정 저장 |
| `F5` | 히스토리 새로고침 |
| `Ctrl+Tab` / `Ctrl+Shift+Tab` | 레일 순서대로 다음/이전 탭 |
| `Esc` | 맨 위 모달을 닫는다. 모달이 없으면 파라미터 열을 닫고 프롬프트로 돌아간다 |
| `↑` / `↓` | 이전/다음 히스토리 이미지 (입력칸에 포커스가 없고 모달이 없을 때) |
| `Shift+↑` / `Shift+↓` | 히스토리 처음/끝 (수식키는 설정 → 단축키에서 Shift/Ctrl/Alt 로 바꾼다) |

**프롬프트 Undo/Redo**: `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z`. 다음 경우에만 프롬프트 패널 기록을 되돌린다.
- 포커스가 프롬프트 입력칸이나 패널 버튼에 있을 때
- 아무것도 포커스되지 않았고, 패널이 보이며 모달에 가려지지 않았을 때

그 밖의 입력칸(대화, 설정, 검색, 모달 등)에서는 브라우저 기본 Undo 가 동작한다. 자동완성 목록은 `↑↓`, `Tab`, `Enter`, `Esc` 로 다룬다.

| 화면 | 키 |
|---|---|
| Settings | `Ctrl+F` 설정 검색 (설정 화면에 포커스가 있을 때) |
| Editor | `Ctrl+O` 열기, `Ctrl+V` 붙여넣기, `Ctrl+S` 저장(덮어쓰기. 전역 `설정 저장` 도 함께 실행된다), `Ctrl+Shift+S` 다른 이름으로 저장, `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z` 되돌리기·다시 하기(마스크가 우선), `Esc` 원근 보정 취소·선택 해제 |
| Editor 도구 | `M` 사각 선택 · `L` 올가미 · `B` 마스크 브러시 · `E` 지우개 · `S` 스탬프 · `P` 펜 · `N` 직선 · `R` 사각형 · `O` 원·타원 · `G` 채우기 · `I` 스포이트 · `Y` 클론 스탬프 · `T` 텍스트 · `D` 그라디언트 · `J` 복원 브러시 |
| Inpaint | `M` 사각 선택 · `L` 올가미 · `B` 마스크 브러시 · `E` 지우개 |
| 대화 | `Ctrl+N` 새 스레드 |

전체 안내는 설정 → **단축키** 에 있다.

---

## 🗄️ 설정·데이터 위치

| 위치 | 내용 |
|---|---|
| `config/` | 앱 설정과 경로. 대표 파일은 `ui_prefs.json`(UI 설정의 단일 출처), `prompt_settings.json`, `backend_runtime.json`, `forge_model_paths.json`, `cond_rules.json`, `global_weights.json`, `chat_threads.json`, `instruction_presets.json`, `app_update.json` 이다. 폴더는 `profiles/`, `workflows/` 가 있다 |
| `user_data/` | 사용자 자료(git 무시). 즐겨찾기, 프롬프트 기록, 캐릭터 프리셋, `instant_wildcards.json`, `stats/generation.json`, `creator/`, `generation_api.json`(토큰), `managed_backends/`, `models/`, Web 탭 프로필 |
| `cache/` | 런타임 자료. 마지막 검색 결과, 대기열·세션 복구, Web 탭 캐시. 지워도 앱은 돌지만, 남은 대기열과 세션 복구가 사라진다 |
| `logs/` | `last_crash.log`(데스크톱), `last_crash_web.log`(웹 모드), 이전 크래시 로그 `last_crash.previous-*.log`, `updates/` |
| `app.log` | 회전 로그(10 MB × 5). 저장소 루트에 생기며, 위치는 `AISTUDIO_LOG_FILE` 로 바꾼다 |
| `image_cache/` | 썸네일 캐시 `thumbs_v2/`, 편집기 임시 파일, SAM3 어휘 파일 |
| `web_profile/` · `backend_ui_cache/` | Vue 화면과 Backend 탭의 Chromium 프로필(localStorage 유지) |
| `generated_images/` | 기본 출력 폴더 |
| `wildcards/` | 와일드카드 파일. 예시(`hairstyle.txt`, `test.txt`)가 들어 있다 |
| `presets/` | 생성 프리셋. 새 clone 에는 없고 첫 프리셋을 저장할 때 생긴다 |
| `Editor_models/` | YOLO·SAM 체크포인트. git 에 없다(가중치 `*.pt` 등과 `yolo_config.json` 은 무시 대상). 앱이 처음 필요할 때 폴더를 만든다 |

- 예전 위치에 있던 설정 파일은 처음 읽을 때 새 위치로 자동으로 옮겨진다.
- 설정 → **데이터 · 백업** 은 `config/`·`user_data/` 의 설정 파일과 `presets/`, `wildcards/` 폴더를 함께 백업한다(대화 기록은 선택).

---

## 🩺 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 앱이 뜨지 않고 꺼진다 | 콘솔 메시지와 `logs/last_crash.log`(웹 모드는 `logs/last_crash_web.log`), `app.log` 를 본다 |
| 첫 실행이 패키지 설치에서 멈춘다 | CUDA torch 설치에 실패하면 앱이 시작되지 않는다. 네트워크를 확인하고, 필요하면 `core/check_requirements.py` 의 `_TORCH_CUDA_INDEX` 를 맞춘다 |
| 첫 실행에 데이터를 받지 못한다 | 오프라인이면 첫 설치는 시작되지 않는다. 연결한 뒤 다시 실행하거나 [데이터 다시 받기](#데이터-다시-받기)를 한다 |
| 백엔드 선택 화면이 '연결 실패'로 닫히지 않는다 | 백엔드가 떠 있는지, Forge 는 `--api` 로 띄웠는지, 주소와 포트가 맞는지 확인한다 |
| ComfyUI 에서 '필요한 노드가 없습니다' 오류 | [동봉 노드 팩](#백엔드-연결)을 설치하고 ComfyUI 를 재시작한다 |
| Hugging Face 에서 접근이 거부된다 | venv 를 활성화한 창에서 `hf auth login` 을 실행하고, 저장소 페이지(예: `facebook/sam3`)에서 접근을 승인한다 |

---

## 📁 디렉토리 구조

```
.
├── new_main_ui.py          # 데스크톱 진입점
├── web_main_ui.py          # 웹 모드 진입점
├── new_run_main_ui.bat     # 기본 런처 (venv 생성·복구, pip 준비)
├── run_gui.bat             # 데스크톱 실행 (기존 venv)
├── run_WEB_gui.bat         # 웹 모드 실행 (기존 venv)
├── config.py               # 경로 상수
├── requirements.txt · VERSION · run_tests.py
├── ui/                     # QMainWindow(generator_*.py 믹스인), VueBridge, 위젯 프록시, *_actions.py
├── backends/               # WebUIBackend(A1111/Forge) · ComfyUIBackend + 진행률/미리보기
├── core/                   # Qt 없는 순수 로직
├── workers/                # QThread 워커
├── utils/                  # 로거, 와일드카드, 조건부 규칙, 태그 자동완성, 테마 등
├── tabs/                   # Web·Backend 네이티브 화면 + 숨겨진 I2I/Inpaint/Upscale PyQt 탭
├── widgets/                # 대기열 상태(QObject)·QueueManager, 작은 PyQt 위젯
├── frontend/               # Vue 3 + Vite 소스 (src/, dev/)
├── frontend_dist/          # Vue 빌드 결과 (커밋됨)
├── comfy_custom_nodes/     # 동봉 ComfyUI 노드 팩 ai_studio_forge_parity
├── danbooru_optimized/     # Danbooru 데이터 (parquet 은 자동 다운로드)
├── tags_db/                # 태그 DB (git 추적)
├── wildcards/              # 와일드카드 파일
├── config/                 # 설정 (참고용 목록 등 일부만 추적)
├── docs/                   # 기능 문서
├── design/                 # 디자인 캔버스
├── tests/                  # 백엔드 테스트 (unittest)
├── tools/                  # 데이터 재생성, 테마 CSS 생성, 테스트 훅
├── scripts/                # CPU 스모크 스크립트
├── assets/                 # 아이콘
└── CLAUDE.md · AGENTS.md   # 개발 지침
```

---

## 🔧 개발

작업 지침은 [`CLAUDE.md`](CLAUDE.md) 에 있다. [`AGENTS.md`](AGENTS.md) 는 같은 필수 규칙에 핵심 파일 표를 더한 문서라서, 한쪽을 고치면 다른 쪽도 맞춘다.
git 과 빌드 명령은 모두 저장소 루트 기준으로 실행한다.

### 검증 명령

```bat
:: 백엔드 전체 (표준 unittest, pytest 불필요)
venv\Scripts\python.exe run_tests.py
:: 느린 통합 테스트·torch 테스트 제외
venv\Scripts\python.exe run_tests.py --quick
:: 모듈별 실행 시간·import 시간 측정
venv\Scripts\python.exe run_tests.py --durations 15
```

```bat
cd frontend
:: vitest (src/**/*.test.ts)
npm run test
:: node --test (src/studio/*.test.mjs)
npm run test:node
:: vue-tsc --noEmit, 0 errors 유지
npm run type-check
:: ../frontend_dist 로 빌드
npm run build
```

- 반드시 venv 의 Python 을 쓴다. PATH 의 다른 Python 에는 PyQt6·pandas 등이 없어서 가짜 ImportError 가 난다.
  `run_tests.py` 는 venv 밖에서 실행되면 venv 로 다시 실행한다(`URIV_TESTS_REEXEC=1` 로 끈다). `py_compile` 같은 다른 명령에는 이런 장치가 없다.
- `--include MODULE` 은 `--quick` 에서 뺀 느린 모듈만 다시 넣는다. 느린 모듈 목록은 `run_tests.py` 의 `SLOW_MODULES` 에 있다. `--with-torch` 는 `--quick` 에서도 torch 테스트를 돌린다.
- Vue 를 고쳤으면 `npm run build` 결과인 `frontend_dist/` 도 함께 커밋한다. `frontend/src/utils/` 의 순수 로직을 고쳤으면 `npm run test` 를 돌린다.

### 계약 테스트

| 테스트 | 지키는 것 |
|---|---|
| `tests/test_bridge_contract.py` | Vue 액션·이벤트 이름과 Python 핸들러·시그널이 양방향으로 일치하는지 확인한다. Python 전용 이름은 `PYTHON_INTERNAL` 에, 정리 예정 이름은 `PENDING`(사유 필수)에 적는다 |
| `tests/test_web_mode_security.py` | 웹 모드 인증, 파일 전송, 허용 목록 |
| `tests/test_component_emit_contract.py` | 컴포넌트가 선언한 emit 을 부모가 실제로 바인딩하는지 |
| `frontend/src/App.templateBindings.test.ts` | 템플릿이 쓰는 이름이 `<script setup>` 에 모두 정의돼 있는지 |
| `frontend/src/types/bridgePayloads.test-d.ts` | 브리지 페이로드 타입 강제가 살아 있는지(`type-check` 로 검사) |

### 규칙

- 새 기능은 거대 파일에 더하지 않는다. 프론트는 `frontend/src/composables/use*.ts`, 백엔드 순수 로직은 `core/`·`utils/` 에 따로 만들고 테스트를 같이 둔다.
- 커밋 메시지는 한국어 conventional 형식(`feat:` / `fix:` / `refactor:` / `test:` / `chore:` / `docs:`)이다.
- 런타임 데이터(`config/cond_rules.json`, `config/char_global_prefs.json`, `cache/session/session_backup.json`)와 API 키·토큰은 커밋하지 않는다.
- `.bat`/`.cmd` 는 CRLF 로 저장한다(`.gitattributes` 와 `tests/test_line_ending_policy.py` 로 강제).

---

## 라이선스

MIT

동봉 ComfyUI 노드 팩에 들어 있는 제3자 코드의 라이선스는 [`comfy_custom_nodes/ai_studio_forge_parity/LICENSES/`](comfy_custom_nodes/ai_studio_forge_parity/LICENSES/) 에 있다.
