# ComfyUI 이미지 메타데이터 읽기

Gallery와 PNG Info는 `core.image_metadata.read_metadata_for_ui()`의 같은 결과를 사용한다. 이미지 파일은 읽기만 하며 모델, 워크플로 노드, 서버를 실행하지 않는다.

## 읽는 정보

- PNG `prompt` API 그래프가 우선이고, 없으면 `workflow`의 링크와 알려진 위젯 배치를 사용한다.
- 저장/미리보기 노드로 이어진 샘플러의 positive/negative 연결을 추적한다. 노드 번호·배열 순서·제목으로 역할을 추측하지 않는다.
- KSampler 계열, SamplerCustom/Advanced의 CFGGuider·BasicGuider, CLIP/SDXL/Flux 인코더, 앱의 Anima semantic 인코더, 알려진 Conditioning·ControlNet 연결을 읽는다.
- Seed, Steps, CFG, Sampler, Scheduler, Denoise, Model, Latent 크기는 연결에서 확정할 수 있는 값만 제공한다.
- JPEG/WebP의 EXIF UserComment와 기존 A1111/Forge `parameters`도 유지한다. A1111 정보와 Comfy JSON이 함께 있으면 A1111 표시가 우선이고 Comfy 원문은 별도로 보존한다.

## 모호한 경우

여러 샘플러의 문장, SDXL/Flux의 서로 다른 인코더 문장, 혼합 conditioning은 합치거나 하나를 선택하지 않는다. 공통으로 확정된 문장·설정만 정규화하며, 전체 자동 적용이 안전하지 않으면 `can_apply=false`로 T2I 전송/즉시 생성을 차단한다. 후보별 원문을 따로 보고 복사할 수 있다.

알 수 없는 커스텀 노드, 동적 텍스트 생성, muted/bypass 의미, subgraph, 잘못된 링크는 경고와 원본 JSON을 남긴다. 이 기능은 기본 문장·설정의 재사용이지 ComfyUI 워크플로 전체 복원이 아니다.

JSON 8 MiB, 노드 4,096개, 연결 깊이 96, 후보 128개 및 탐색 예산을 제한한다. 그래프 문자열을 평가하거나 참조 파일/URL을 열지 않는다.

## UI 및 보존

PNG Info에 Prompt만/Negative만 복사 버튼을 항상 보이도록 제공한다. Qt 데스크톱과 웹 단말의 공통 클립보드 함수를 사용하고 실제 성공일 때만 성공 알림을 표시한다. 원본 `prompt`/`workflow` JSON은 각각 펼쳐 보기와 복사가 가능하다.

Gallery의 Comfy 메타데이터는 읽기 전용이다. A1111 'EXIF 저장'(`saveImagePrompt`)은 프롬프트/네거티브만 바꾸고 파라미터 꼬리(따옴표 값·Template 줄)를 원문 그대로 둔다(`replace_prompt_in_parameters` + `rewrite_png_parameters`). PNG는 다시 인코딩하지 않고 `parameters` 텍스트 청크만 청크 단위로 바꾼다(`core.png_chunks`) — APNG 프레임·시간, 16비트 깊이, gAMA·sRGB·iCCP·eXIf·pHYs·tRNS·다른 텍스트 청크가 바이트 그대로 남는다. Comfy 그래프가 든 PNG는 거부한다.

메타 읽기(Gallery·PNG Info·EXIF 검색·메타 이식·'EXIF 저장'·SAM3/ADetailer 배치)는 모두 `extract_from_pil` 하나를 쓴다. PNG의 IDAT 뒤(IEND 앞)에 붙은 텍스트 청크도 청크 머리만 훑어 읽고(픽셀 디코드 없음), `Description` 청크에만 든 A1111 infotext는 파라미터 줄이 있을 때만 쓴다.

PNG Info의 '메타 이식'(`pnginfo_transplant_meta`)은 보고 있는 이미지의 메타를 다른 이미지에 박아 새 PNG(기본 `<이름>_withmeta.png`)로 저장한다. 대상 픽셀·투명도·ICC·dpi는 유지하고 JPEG 회전 태그는 픽셀에 반영한다. 원본 Comfy 청크가 있으면 합성 `parameters`를 만들어 출처를 바꾸지 않으며, 'ComfyUI 워크플로 포함'을 끄면 A1111 형식 텍스트만 남긴다. 호스트 파일 대화상자를 쓰므로 원격 웹 모드에서는 막힌다.

## A1111 infotext 파서 (유일본)

`core.infotext`(`split_infotext` / `parse_parameters_text`, `core.image_metadata`에서도 같은 이름으로 다시 내보냄)가 앱 전체의 유일한 A1111 파서다. Gallery·Favorites 'T2I에서 사용', PNG Info 'T2I 전송/즉시 생성', 히스토리 '프롬프트 당겨오기/다음 큐', SAM3·ADetailer 배치 'EXIF 프롬프트 사용'이 모두 이 결과(`core.metadata_actions`)를 쓴다.

- 네거티브가 비어 줄이 없는 이미지, 빈 positive(`Negative prompt:`로 시작), 파라미터만 있는 텍스트(`Steps:`로 시작), CRLF를 읽는다.
- 파라미터 줄은 네거티브 줄보다 뒤의 마지막 `Steps: <정수>` 줄이다. 뒤 세그먼트(점 키 `Anima 3.8B adapter`, 따옴표 없는 JSON 값 `Hashes: {…}`·`Civitai resources: […]`·`Tiled Diffusion: {…}`)나 뒤따르는 줄(여러 줄 `Template:` 값, 빈 `Negative Template:`, 자유 텍스트)과 관계없이 그 줄부터가 파라미터 꼬리다.
- Steps 줄이 없을 때만 줄 전체가 `Key: value` 쌍(3쌍 이상 또는 알려진 키로 시작하는 2쌍)이고 뒤가 모두 한 줄 항목인 줄을 파라미터로 본다. `(masterpiece:1.2), …` 가중치 프롬프트나 `Size: huge, masterpiece` 같은 문장은 프롬프트로 남는다.
- 모든 판정은 줄 길이에 선형이다(세그먼트를 손으로 나눈다). 줄 전체를 중첩 정규식으로 fullmatch 하면 실패하는 줄에서 LoRA 해시마다 백트래킹이 두 배로 늘어 GUI가 멈췄다.
- 값의 따옴표는 A1111과 같은 JSON 규칙으로 풀고(`\"` 포함), 직렬화도 같은 규칙으로 따옴표친다. Dynamic Prompts의 `Template:` 꼬리 줄은 파라미터로 둔다.
- 표시용 `params_line`은 WebUI면 원문 꼬리 그대로, Comfy면 A1111 규칙 직렬화다. 표시 그룹 `params`와 PNG Info 비교는 `parameters` dict에서 만든다 — 표시 문자열을 다시 파싱하지 않는다.

`read_metadata_for_ui()`는 기존 `raw`, `prompt`, `negative`, `params_line`, `path`, `filename`, `size` 필드를 유지한다. 추가 필드는 `source`, `parameters`, `params`(표시 그룹, 파라미터가 없으면 null), `raw_prompt`, `raw_workflow`, `metadata_warnings`, `prompt_candidates`, `can_apply`, `metadata_ambiguous`이다.

즉시 생성과 히스토리 큐 추가는 소스와 관계없이 표시용 `params_line`을 다시 텍스트 파싱하지 않고 확정된 `parameters`를 사용한다(sampler·scheduler·steps·cfg·size). 즉시 생성은 원래 시드를, 큐 추가는 새 변형용 `-1`을 사용하고, 프롬프트는 prefix/suffix를 다시 붙이지 않은 완성본으로 적용한다. 현재 모델·LoRA는 유지하며 메타데이터의 모델 경로로 자동 변경하거나 다운로드하지 않는다. PNG Info·Gallery 전송 및 히스토리 당겨오기에서도 모호한 그래프의 부분 프롬프트를 자동 적용하지 않는다.

## 검증

`tests/test_image_metadata.py`는 작은 임시 PNG/JPEG/WebP를 생성하여 실제 추출 함수를 검증하고 원본 바이트가 변하지 않는지 확인한다. API/워크플로 그래프, 순서에 의존하지 않는 역할 추적, 다중 샘플러, SDXL, custom advanced guider, ControlNet, Anima, 순환/미지 노드, 손상 JSON, A1111 공존, EXIF, A1111 infotext 골든(네거티브 생략·빈 positive·파라미터만·따옴표·Template 줄·CRLF), 'EXIF 저장'의 eXIf/dpi/IDAT 뒤 청크 보존, 메타 이식을 포함한다. `tests/test_worker_exif_prompts.py`는 배치 워커의 EXIF 프롬프트를, `tests/test_metadata_action_parity.py`는 네거티브 없는 PNG에서 네 액션이 같은 결과를 내는지를 검증한다.

`frontend/src/components/ComfyMetadataDetails.test.ts`는 후보 표시와 원본 JSON의 HTML 이스케이프를 검증한다. 모델 다운로드나 GPU 생성 결과를 검증하는 테스트는 아니다.

`tests/test_comfy_metadata_actions.py`는 실제 브릿지 읽기와 Vue 액션을 작은 임시 PNG 및 가짜 화면·생성 경계에서 연결해 검증한다. 모호한 정보의 적용 차단, 원문 JSON과 파라미터가 프롬프트로 섞이지 않는지, 원래 모델·LoRA 보존, 공통 설정·시드 전달 및 기존 WebUI 동작을 확인한다. 실제 생성이나 서버 연결은 실행하지 않는다.
