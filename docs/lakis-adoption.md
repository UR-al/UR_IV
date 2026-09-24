# LAKIS 참고 기능 적용 · 2026-09-06

LAKIS의 기능 구성을 참고해 AI Studio Pro의 기존 Vue / QWebChannel / Comfy 컴파일러에 연결했다. LAKIS 앱·설치기·카메라 UI·조명 소스를 통째로 복사하지 않았으며 새 프론트 애니메이션 라이브러리를 추가하지 않았다. 기존 Forge 경로와 공유 Comfy 서버의 작업 취소 범위는 유지한다.

## 사용 위치

| 기능 | 위치 | 동작 |
| --- | --- | --- |
| 생성 중 이미지 미리보기 | Comfy 생성 결과 영역 | 실행 중인 앱 작업의 WebSocket 이미지 프레임만 표시 |
| 구도 · 카메라 | 생성 프롬프트 패널 | 시점·거리·기울기·프레이밍을 조절하고 명시적으로 메인 태그에 추가 |
| 빠름 / 정밀 | Settings → 모델 다운로드 → ComfyUI 워크플로 도구 | 빠름은 Hires/ADetailer/SAM3 OFF, 정밀은 Hires 1.5× + 선택 영역 보정 |
| 얼굴 → 눈 | 정밀 보정 대상 | 얼굴 보정 결과를 다시 입력으로 삼는 실제 순차 2패스 |
| 워크플로 노드 설정 | 같은 패널 → 사용자 워크플로 노드별 상세 설정 / 백엔드 관리 고급 설정 | 설치된 서버의 `object_info` 기반 스칼라 입력 폼 |
| 생성 전 사전 검증 | 같은 패널 → 현재 기능 사전 검증 | 현재 T2I 설정을 컴파일하고 누락 노드·입력·모델 문제 표시. GPU 생성은 실행하지 않음 |
| 호환 조합 안내 | Settings → 모델 다운로드 | 서버 스키마·모델 목록·알 수 있는 버전과 사용자가 저장한 기준 비교 |
| Spectrum | Settings → 모델 다운로드 → Spectrum 가속 | 기본 OFF. 외부 `DiTSpectrumPatch`가 있는 Comfy 기본 샘플러에만 적용 |
| 조명 편집 | I2I → 실험 · 조명 편집 | 기본 OFF. CPU 미리보기 → I2I 적용 / 원본 복원 / 별도 PNG 저장 |
| Comfy 조명 노드 | 앱 번들 노드 `AIStudioRelight` | 이미지 + 선택 깊이/노멀/마스크 입력, 결과·조명·노멀·그림자 출력 |

설정 검색에서 `Spectrum`, `호환 조합`, `워크플로`로 해당 설정으로 이동할 수 있다. 프리셋은 모델·해상도·기본 스텝·업스케일러를 임의로 바꾸지 않는다. 고해상도 처리에 맞는 업스케일러와 SAM3 가중치는 사용자가 준비해야 한다.

## 상태와 파일 보호

- 카메라 태그는 미리보기 후 추가한다. 기존 태그를 삭제하지 않으며 중복 추가 방지, 충돌 안내, 마지막 적용 되돌리기를 제공한다.
- 사용자 워크플로 원본 JSON과 노드 연결은 변경하지 않는다. 기본 입력 폼은 BOOL/INT/FLOAT/문자열/선택값을 지원하며 링크와 앱이 관리하는 입력은 보호한다.
- 대기열은 등록 시 상세 입력값을 고정한다. 이후 설정을 바꿔도 등록된 작업은 바뀌지 않으며, 서버·워크플로가 달라진 경우 다른 조건으로 조용히 실행하지 않고 재등록을 안내한다.
- 단독 ADetailer/SAM3/Refine에는 이전 생성의 추가 눈 보정 패스를 가져오지 않는다.
- 추가 눈 패스는 프리셋을 적용한 창·서버에서만 활성화한다. SAM3 대상/처리 모드를 직접 바꾸거나 끄면 해제되며, 원래 값으로 돌려도 자동 재활성화되지 않는다. 앱 재시작 뒤에는 프리셋을 다시 적용해야 하고 기존 설정 복원본은 유지한다.
- `config/ui_prefs.json`의 `compositionControl`, `comfySpectrum`에 UI 옵션을 저장한다.
- `config/comfy_workflow_controls.json`은 서버·워크플로별 입력 설정, `config/comfy_quality_preset.json`은 프리셋 복원본이다.
- `config/comfy_compatibility_baselines.json`은 사용자가 기준 저장을 눌렀을 때만 생성/갱신한다. 해시와 버전 위주로 저장하며 모델 이름과 인증 URL을 복사하지 않는다.
- 조명 결과는 출력 폴더의 `relight/`에 새로운 PNG로 저장한다. 원본이나 기존 출력은 덮어쓰지 않는다.
- 새 로컬 파일/설정 작업은 네이티브 전용이다. 웹 브리지의 허용 목록을 확대하지 않았다.
- 전역 `/free`, 다른 클라이언트 작업 중단, 서버 재시작, 자동 다운로드/설치는 추가하지 않았다.

## 동작 제한과 수치

### 미리보기

레거시 바이너리와 메타데이터 포함 프레임을 분리해서 읽는다. 이미지 형식·크기·작업 식별자를 검사하고 250ms 제한 및 중복 프레임 해시로 과도한 브리지 전송을 막는다. 메타데이터가 없는 구형 프레임은 앱 전용 클라이언트 소켓과 현재 실행 구간을 기준으로 구분한다. 서버가 미리보기를 보내지 않으면 새 이미지를 임의로 만들지 않는다.

### Spectrum

외부 확장을 자동 설치하지 않는다. 캐시 간격 2, 간격 증가량 0.25, 준비 6단계, 마지막 실제 계산 3단계, 혼합 0.3, 차수 3, 안정화 0.1, 이력 100이 기본값이다. 준비+마지막 단계보다 전체 Steps가 커야 한다. SPEED 동시 사용은 막고 샘플러별 옵션 컨테이너를 분리한다.

이는 외부 Spectrum 노드 내부의 GPU deepcopy나 훅 오류를 수정한 것이 아니다. 외부 구현의 호환성과 실제 품질·속도는 별도 검증이 필요하다. Forge 및 Krea2 전용 생성에는 적용하지 않는다.

### 조명 편집

새 모델을 이용한 자동 Depth/Normal 추론이 아니라 독립적인 2.5D 후처리이다. 사용자가 올린 깊이 맵은 흰색이 가까움, 노멀 맵은 RGB=XYZ / +Y 위 / +Z 카메라 방향이다. 맵은 원본과 같은 해상도여야 한다. 맵 없이 실행할 경우 명암 기반 근사라고 표시하며 투영 그림자는 계산하지 않는다. 기존 그림자 완화도 밝기를 평탄화하는 근사이지 정확한 그림자 제거가 아니다.

로컬 패널은 정지 PNG/JPEG/WebP, 원본 최대 16MP, 한 번에 한 작업을 처리한다. Comfy 노드는 결과·진단 출력의 메모리를 고려해 배치 전체를 4,194,304픽셀 이하로 제한한다. 강도 0과 마스크 0 영역 및 알파는 보존한다. 직접 업로드한 데이터의 지원 메타데이터는 저장하지만, Gallery 이미지를 브라우저 canvas로 가져온 경우 원본 메타데이터는 유지되지 않을 수 있다.

### 호환 조합

LAKIS 커밋 `19ec1be13414ea8c029782184121ee43b3662bea`의 설치기 버전 고정값은 참고 자료이지 검증된 최적 조합이나 자동 다운그레이드 지시가 아니다. 노드 스키마가 존재한다는 사실만으로 GPU 생성 성공을 판정하지 않는다. 연결 불가·정보 없음은 확인 불가로 표시한다.

## 검증

- 최종 Python 전체 테스트: 1,375개 실행, 실패 0, 3개 건너뜀. Vue: 26개 파일의 188개 테스트 통과. 변경된 Python 83개 파일 문법 검사, `vue-tsc --noEmit`, Vite 배포 빌드 모두 통과했다.
- 생성 프리뷰 프레임 파싱/작업 격리, 스칼라 값 검증/드리프트, 순차 보정, 프리셋 복원, 조명 입력/취소/저장, Spectrum 범위/옵션 격리 회귀 테스트를 추가했다.
- 실제 앱 컴포넌트를 불러오는 `frontend/dev/theme-audit.html`에서 라이트·다크, 카메라 태그 적용/중복 차단, 설정 검색, 실험 기능 기본 OFF와 비활성 조건을 확인했다. 이 페이지의 브리지는 메모리 전용 모의 구현이므로 실제 파일 저장이나 Comfy 생성 증거로 보지 않는다.
- `venv/Scripts/python.exe scripts/smoke_relight.py`는 새 TEMP 폴더에 합성 구/깊이 맵의 좌우 조명 결과, 진단 PNG, `verification.json`을 저장한다. 10개 검증이 모두 통과했고 실제 결과 이미지를 확인했다. 이는 자연 이미지의 조명 품질 검증을 대신하지 않는다.
- 현재 설정된 Comfy 서버에 연결되지 않아 이번 변경의 GPU 생성, 실시간 미리보기 수신, 얼굴→눈 결과 품질, Spectrum 속도 측정은 아직 검증하지 못했다.

번들 노드 소스 버전은 1.2.0이다. 앱을 다시 실행해야 새 Python/Vue 코드가 반영된다. 기존 설치의 Comfy 노드 배포/서버 재시작은 기존 런타임 관리 흐름을 사용하며 이번 작업에서 임의로 실행하지 않았다.

## 주요 변경 파일

- 프리뷰: `backends/comfyui_preview.py`, `backends/comfyui_backend.py`, `backends/base.py`
- 구도 UI: `frontend/src/components/CompositionControl.vue`, `frontend/src/utils/compositionPrompt.ts`, `frontend/src/components/PromptPanel.vue`
- 상세 입력/프리셋: `core/comfy_workflow_controls.py`, `ui/comfy_workflow_actions.py`, `core/comfy_workflow_compiler.py`, `frontend/src/components/ComfyWorkflowControls.vue`, `frontend/src/utils/comfyWorkflowControls.ts`, `frontend/src/components/BackendGate.vue`
- 조합 비교: `core/comfy_compatibility.py`, `ui/comfy_compatibility_actions.py`, `frontend/src/components/ComfyCompatibilitySettings.vue`
- Spectrum: `core/spectrum_settings.py`, `frontend/src/components/SpectrumSettings.vue`, `comfy_custom_nodes/ai_studio_forge_parity/generation.py` (샘플러 호출마다 `ModelPatcher.clone()` 으로 model_options 를 격리 — 별도 `spectrum_isolation.py` 는 clone 과 중복이라 제거됨)
- 조명: `comfy_custom_nodes/ai_studio_forge_parity/relight.py`, `ui/relight_actions.py`, `frontend/src/components/RelightPanel.vue`, `frontend/src/views/I2IView.vue`, `scripts/smoke_relight.py`
- 공통 연결: `ui/vue_bridge.py`, `ui/generator_main.py`, `ui/generator_generation.py`, `ui/chat_actions.py`, `frontend/src/types/bridge.d.ts`, `frontend/src/views/SettingsView.vue`, `core/comfy_node_pack.py`, 번들 노드 `__init__.py`, `.gitignore`
- 대응 Python/Vue 회귀 테스트와 `frontend_dist` 빌드 산출물

Editor 및 Claude/GPT 아이콘 애니메이션 소스는 이번 LAKIS 기능 작업에서 수정하지 않았다. 기존 미커밋 변경과 개인 설정 파일도 되돌리거나 정리하지 않았다.
