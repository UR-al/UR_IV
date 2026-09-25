# sam-extra 동기화 — 확장이 업데이트되면

앱(UR_IV)은 Forge 확장 sam-extra(`forge_sam3_extension`)의 스크립트 제목, 위치 인자 순서, dict 키, 라우트를 그대로 믿고 요청을 만든다. 확장이 바뀌어도 Forge 는 대부분 오류를 내지 않는다. 인자가 한 칸 밀리거나 키 이름이 바뀌면 값이 조용히 버려지거나 엉뚱한 곳에 들어간다. 그래서 확장이 드러내는 항목을 하나도 빠짐없이 레지스트리에 적어 두고, 테스트가 설치된 확장과 양방향으로 비교한다.

## 구성

| 파일 | 역할 |
|---|---|
| `core/sam_extra_contract.py` | 레지스트리. 확장의 모든 항목을 mapped, ignored, deferred 중 하나로 분류한다. `EXT_VERSION_AUDITED` 가 맞춘 확장 버전이다. |
| `core/sam_extra_scan.py` | 확장 소스를 AST 로만 읽는 스캐너. 확장 코드를 실행하지 않으므로 GPU 와 무관하다. 설치 위치 찾기(`find_installed_extension`)도 여기 있다. |
| `core/sam_extra_diff.py` | script-info 와 앱 spec 을 비교하는 순수 함수. 인자 모양 해시도 만든다. |
| `tests/test_sam_extra_contract.py` | 계약 테스트. `run_tests.py --quick` 에 포함된다. |
| `tests/fixtures/sam_extra_script_info.json` | 라이브 `GET /sdapi/v1/script-info` 에서 sam-extra 항목만 저장한 픽스처. AST 로는 못 읽는 기본값, 범위, 선택지의 출처다. |
| `tools/refresh_sam_extra_fixture.py` | 픽스처를 갱신하는 도구. 실행 중인 Forge 에 읽기 전용 GET 두 번만 보낸다. |
| `tests/_sam_extra_ext.py` | 확장 위치 찾기와 skip 규칙을 가드 테스트끼리 공유한다 (`test_sam3_args`, `test_anima_guidance` 도 쓴다). |
| `core/sam_extra_notices.py` · `tests/test_sam_extra_notices.py` | 결과 infotext(`SAM3 Error`, `Anima38: off: …`, `Anima Perturbation Guidance` 누락)와 생성 전 조건(CFG≈1 의 SMC/APG/CWM, `sam3.pt` 자동 다운로드, 없는 스크립트), HTTP 422 본문을 사용자 알림으로 바꾼다(P4). 테스트의 `ExtensionSourceTests` 가 이 infotext 키와 가드 함수가 설치된 확장에 그대로 있는지 본다. |

레지스트리가 다루는 범위는 다음과 같다. always-on 스크립트 제목(10개), 스크립트별 `ui()` 반환 순서와 `ARG_NAMES`, API 경로가 `_arg(N)` 으로 읽는 인덱스, `Sam3Args` 필드·기본값·어노테이션 범위와 `process()` 가 state 에서 읽는 키, Forge 옵션 키(14개), FastAPI 라우트, XYZ 축 라벨(104개), Gradio 전용 이름 엔드포인트(앱이 기대면 안 되는 것), 모듈 파일 단위, `preload.py` 명령줄 플래그, 의미 핀(Detail Daemon 배율, 공유 메모 스키마·한도, LoRA Manager `_BRIDGE_JS` 메시지 모양 등), 앱 spec 과 라이브 값의 알려진 차이.

### 소스와 픽스처를 둘 다 보는 이유

픽스처 테스트(shape 해시, 앱 spec 비교, 의미 핀)는 저장된 script-info 만 본다. 그래서 확장 소스에서 슬라이더 범위나 기본값만 바꾸면(인자 수는 그대로) Forge 를 다시 시작해 픽스처를 갱신하기 전까지 통과한다. 이를 막으려고 계약 테스트는 설치된 소스의 `ui()` 컴포넌트에서 라벨, 기본값, 범위, step, 선택지를 AST 로 읽어 픽스처와 비교한다(`test_fixture_matches_installed_source`). 모듈 상수, `CONST[1:]` 슬라이스, 지역 변수, `with InputAccordion(...) as x` 도 따라간다. 실행 시점 목록(파일, 어댑터, CLIP-L)은 읽지 않는다. 픽스처에는 값이 있는데 AST 로 못 읽는 칸은 `UI_UNREAD` 에 사유와 함께 적는다. 새로 못 읽게 된 칸도, 이제 읽히는 칸도 실패한다.

### 아직 없는 검사

gap matrix 5-(a) 가운데 다음은 아직 없다. 이 검사가 있다고 가정하지 않는다.

- 5번 메시지 계약 중 `wire_tipo` 입력 순서, `MODES`/`LENGTHS`, `REFERENCE_ARG_KEYS` (P19·P20 이 Gradio 방식을 고를 때 추가). `_BRIDGE_JS` 의 `sam3-add-lora` 메시지 모양은 의미 핀 `lora_bridge_message` 가 본다.
- 6번 앱 내부 커버리지: 모든 `SAM3_KEYS` 가 `_build_sam3_settings` 에서 읽히고, 프록시가 있고, 저장·복원되고, `Sam3MaskCard.vue` 나 `Sam3ControlNetPanel.vue` 에 바인딩되는지. 위치 spec 키가 `components/guidance/*Section.vue`(P0-B 분할 뒤 칸이 있는 곳, `AnimaGuidancePanel.vue` 는 섹션을 놓는 틀) 에 모두 바인딩되는지. 지금 레지스트리는 섹션마다 켜기 키 하나만 참조로 본다.
- 7번 Comfy: mapped 제목마다 `comfy_workflow_compiler` 가 블록을 쓰거나 unsupported 로 선언하는지, `SEMANTIC_PINS` 가 Comfy 미러에도 반영됐는지. 지금 `comfy=` 참조는 파일과 심볼이 있는지만 본다.
- `UI_ONLY_FEATURES` 는 등록된 함수가 그 파일에 있는지만 보는 한 방향 검사다. 새 Gradio 이름 엔드포인트는 잡지 않는다.

### 분류 규칙

- **mapped**: 앱이 쓴다. `app` 에 `"파일:심볼"` 을 적는다. 테스트가 그 파일에 그 심볼이 실제로 있는지 확인한다. 앱 코드를 옮기면 여기도 함께 고친다. 알려진 빈틈은 `gaps` 에 작업 패키지 id 와 함께 적는다.
- **ignored**: 앱과 무관하다. Forge 화면 전용이거나 해당 없는 항목이다. `reason` 이 필요하다.
- **deferred**: 아직 앱에 없다. `package`(gap matrix 의 P1-P21, 보류 항목은 `HOLD`)와 `reason` 이 필요하다.
- `optional=True` 는 병렬로 들어오는 중인 항목에만 쓴다. 확장이나 앱 한쪽에 아직 없어도 통과한다. 양쪽이 자리 잡으면 표시를 지운다. 지금은 쓰는 항목이 없다(공유 메모 라우트와 파일은 양쪽이 자리 잡아 표시를 지웠다).
- mapped 의 앱 참조는 파이썬 파일이면 심볼이 코드 식별자로 있어야 한다. 주석이나 문자열에만 남은 이름은 인정하지 않는다.

## 체크리스트: 확장이 업데이트되면

1. **무엇이 바뀌었는지 본다.**
   - `git -C C:\sd-webui-forge-classic\extensions\forge_sam3_extension log --oneline 818b8fe..HEAD`
   - `git -C <확장> status` (작업 트리가 dirty 이면 커밋 해시가 실제 코드를 대표하지 못한다)
   - `CHANGELOG.md` 맨 위 항목과 `sam3ext/__version__.py`
   - 확장 저장소는 사용자 것이다. 앱 작업 중에는 확장 파일을 고치거나 git 상태를 바꾸지 않는다.

2. **픽스처를 갱신한다.** Forge 는 시작할 때 확장 코드를 읽는다. 확장을 업데이트한 뒤 Forge 가 다시 시작됐는지 먼저 확인한다(재시작은 사용자가 한다).
   ```powershell
   venv\Scripts\python.exe tools\refresh_sam_extra_fixture.py
   ```
   - 기본 주소는 `http://127.0.0.1:7860` 이다. 다르면 `--api-url` 로 준다.
   - `GET /sdapi/v1/script-info` 와 `GET /sdapi/v1/extensions` 만 보낸다. POST, 옵션 변경, 생성은 하지 않는다.
   - Forge 가 꺼져 있으면 예전에 저장한 응답으로 만들 수 있다: `--from-file <script-info.json> --extensions-file <extensions.json>`. 이 경우 Forge 가 다시 켜지면 한 번 더 받는다.
   - 응답이 목록이 아니거나(인증 오류 등) sam-extra 스크립트가 하나도 없으면(확장이 꺼진 Forge, 관리형 Forge 등) 저장하지 않고 종료 코드 3 으로 끝난다. 픽스처는 임시 파일에 쓴 뒤 바꿔 끼운다.
   - `git diff tests/fixtures/sam_extra_script_info.json` 으로 라벨, 기본값, 범위, 선택지 변화를 먼저 읽어 둔다.

3. **계약 테스트를 강제 모드로 돌린다.** 강제 모드에서는 확장을 못 찾으면 skip 하지 않고 실패한다.
   ```powershell
   $env:AISTUDIO_REQUIRE_FORGE_EXT = '1'
   venv\Scripts\python.exe -m unittest -v tests.test_sam_extra_contract tests.test_sam3_args tests.test_anima_guidance tests.test_sam_extra_notices
   Remove-Item Env:AISTUDIO_REQUIRE_FORGE_EXT
   ```
   확장이 다른 곳에 있으면 `$env:AISTUDIO_FORGE_EXTENSION_DIR = '<확장 루트>'` 로 지정한다. 지정하면 그 경로만 본다. 경로가 틀리면 다른 설치로 대신하지 않고 skip 한다(강제 모드면 실패). skip 이유에 지정한 경로가 나온다.

4. **실패 항목을 하나씩 처리한다.** 메시지의 `+` 는 새 항목, `-` 는 사라진 항목이다.
   - **새 항목**(스크립트, 인자, 라우트, 옵션, 축, 파일, 플래그): 레지스트리에 mapped, ignored(사유), deferred(패키지와 사유) 중 하나로 적는다. mapped 로 적는다면 앱 spec, 카드 UI, 페이로드, 보조 경로 전달(P7), Comfy 미러까지 함께 고친다.
   - **사라진 항목**: 앱 UI, 페이로드, 저장된 설정값에서 쓰는 곳을 정리한다. 옛 저장값을 어떻게 옮길지 정한 뒤 레지스트리에서 지운다.
   - **기본값, 범위, 선택지, 라벨이 바뀜**(`test_arg_shapes`, 스펙 비교 테스트): CHANGELOG 와 코드로 **의미**가 바뀌었는지 먼저 확인한다. 그다음 앱 spec 을 고치거나, 일부러 다르게 두는 것이면 `KNOWN_DIFFS` 에 사유와 함께 적는다. 확인이 끝나면 메시지가 알려 준 새 해시로 `shape` 를 바꾼다.
   - **`KNOWN_DIFFS` 에 있는데 더는 차이가 없음**: 작업 패키지가 고쳤다는 뜻이다. 그 항목을 지운다.
   - **설치된 확장 소스가 픽스처와 다름**(`test_fixture_matches_installed_source`): 소스의 라벨, 기본값, 범위, step, 선택지가 픽스처와 다르다. 픽스처가 낡았다는 뜻이다. Forge 를 다시 시작한 뒤(사용자가 한다) 2번으로 픽스처를 갱신하면, 픽스처 테스트가 앱에서 무엇을 고칠지 알려 준다.
   - **AST 로 못 읽는 칸이 바뀜**(`test_ui_components_are_read_statically`): 확장이 컴포넌트를 새 방식(도우미 함수, 계산식 등)으로 만든다. `core/sam_extra_scan.py` 를 보강하거나 `UI_UNREAD` 에 사유와 함께 적는다. 이제 읽히는 칸은 목록에서 지운다.
   - **Sam3Args 기본값·범위가 바뀜**(`test_sam3args_defaults_match_app_defaults`, `test_sam3args_constraints_match_clamp_table_and_accept_app_range`): 앱 `core/sam3_args.py` 의 기본값과 범위를 맞춘다. 어노테이션(`confloat(le=…)`)과 `_NUMERIC_BOUNDS` 가 서로 다르면 확장 쪽 버그일 수 있다. 앱 범위가 제약을 넘으면 그 값을 보낸 생성에서 SAM3 가 꺼진다.
   - **위치 읽기 인덱스가 바뀜**(`test_positional_reads`): API 가 어떤 인자를 새로 읽거나 더는 읽지 않는다는 뜻이다. Detail Daemon 의 arg1(preset)을 API 가 무시하게 된 변화가 이 경우다.
   - **의미 핀이 깨짐**(`SEMANTIC_PINS`): 인자 모양은 같은데 의미가 바뀐 경우다. 예: Detail Daemon arg2 의 범위(±5)와 `_SIGMA_SCALE`(0.1)은 원본 노드(Jonseed/ComfyUI-Detail-Daemon) 값이다. 앱은 노드 단위를 변환 없이 보내므로(`core/anima_guidance.py` `DETAIL_DAEMON_SPEC`), 확장이 원본과 다른 단위·범위를 쓰게 되면 원본 대조 테스트(`tests/test_detail_daemon_origin.py`)와 Comfy 미러(`DD_SIGMA_SCALE`, `tests/test_comfy_detail_daemon.py`)부터 확인한다. 핀에 `app` 이 붙은 항목은 앱 상수가 다르면 확장 없이도 실패한다.
   - **`process()` 키 테스트가 깨짐**: 확장이 SAM3 state 키 이름을 바꿨다. 앱이 보내는 값이 조용히 버려지고 있으므로 가장 먼저 고친다.
   - **알림 키 테스트가 깨짐**(`tests.test_sam_extra_notices.ExtensionSourceTests`, 핀 `anima38_status_key`·`sam3_hf_checkpoint_*`): 확장이 실패 흔적을 남기는 infotext 키나 CFG 1 가드·OOM 폴백 함수를 바꿨다. 그대로 두면 사용자 알림이 조용히 멈춘다. `core/sam_extra_notices.py` 의 `KEY_*` 상수와 원인 → 설정 힌트(`sam3_error_hint`, `anima38_off_hint`)를 새 문구에 맞춘다.
   - **정적으로 풀지 못한 항목**(`test_everything_was_resolved_statically`): 확장이 새 방식으로 상수를 만든다. `core/sam_extra_scan.py` 를 보강하거나, 그 항목을 레지스트리에 직접 분류한다.
   - **Gradio 전용 함수가 사라짐**: 앱은 이 엔드포인트를 부르지 않는다. 다만 뒤 패키지(P11, P12, P19, P20)가 참고하므로 새 이름으로 고쳐 둔다.

5. **버전을 올린다.** 모든 항목을 분류한 뒤 `EXT_VERSION_AUDITED` 를 설치된 버전으로 바꾸고, `EXT_COMMIT_AUDITED` 도 새 HEAD 로 바꾼다. 1번의 `git log` 범위도 새 커밋 기준이 된다.

6. **전체 검증을 돌린다.**
   ```powershell
   venv\Scripts\python.exe run_tests.py
   cd frontend; npm run test; npm run test:node; npm run type-check; npm run build
   ```

7. **결과 픽셀이 바뀌는 항목은 GPU 로 확인한다.** 기본값, 배율, 인자 의미가 바뀐 기능(가이던스, SAM3 인페인트, Anima 3.8B, DoRA 등)은 같은 시드로 Forge UI 결과와 앱 결과를 비교해야 한다. 이 PC 의 GPU 는 학습 작업으로 바쁜 경우가 많다. **실행하기 전에 사용자에게 묻고, 쓸 VRAM 을 먼저 밝힌다.** 실행 중인 학습 프로세스는 건드리지 않는다.

8. **커밋한다.** `/ship` 으로 한국어 conventional 메시지를 쓴다. 예: `feat: sam-extra 0.31.0 동기화 — …`. 픽스처, 레지스트리, 앱 수정을 같은 커밋에 넣어 기준 버전이 어긋나지 않게 한다.

## 참고

- 확장 위치는 다음 순서로 찾는다. 환경 변수 `AISTUDIO_FORGE_EXTENSION_DIR`, `config/backend_runtime.json` 의 Forge 확장 폴더, 알려진 Forge 설치(`C:\sd-webui-forge-classic`, `C:\sd-webui-forge-neo`, 관리형 `shared/extensions`). 폴더 이름은 `forge_sam3_extension` 과 `sam-extra` 둘 다 본다.
- 확장이 없는 PC 에서는 AST 테스트가 이유를 남기고 skip 한다. 레지스트리 형식 테스트와 픽스처 테스트는 확장이 없어도 돈다.
- 런타임에서 확장 상태를 확인하는 기능(설치 여부, 인자 수, 옵션 존재)은 P5 의 `core/sam_extra_capabilities.py` 가 맡는다. 이 문서의 절차는 개발할 때 쓰는 것이다.
- 근거 분석: gap matrix 5-(a), 5-(c), 나-3·나-4, 라-6.
