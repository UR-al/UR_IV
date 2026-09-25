# Generation API Gateway

AI Studio Pro는 별도 머신용 HTTP 서버를 열어 외부 프로그램의 생성 요청을 받고,
현재 앱 백엔드 또는 설정에 등록한 Forge/WebUI·ComfyUI로 작업을 전달할 수 있습니다.
브라우저용 Web UI 서버와는 포트·인증·수명주기가 분리됩니다.

## 켜기

1. 설정 → **네트워크**의 '이 앱을 생성 API로 사용' 카드를 엽니다.
2. '바인드 호스트'와 '포트'를 확인합니다. 기본값은 `127.0.0.1:17990`입니다.
   (관리형 Forge `17860~`·ComfyUI `18188~` 자동 포트 범위와 겹치지 않게 골랐습니다. 이전 버전의
   기본값 `17860`을 그대로 둔 채 꺼져 있던 설정은 처음 실행할 때 `17990`으로 옮겨집니다.
   켜 둔 설정의 포트는 바꾸지 않으며, 그때는 관리형 Forge가 다음 빈 포트를 씁니다.)
3. 필요하면 'Forge/WebUI · ComfyUI 연결' 카드의 `+ ADD TARGET`으로 원격 target을 추가합니다.
4. '설정 적용'을 누른 뒤 '시작'을 누릅니다.
5. 'Bearer 토큰' 칸의 '복사'로 token을 복사합니다.

'앱 시작'을 켜고 '설정 적용'을 누르면 다음 앱 실행부터 자동으로 시작합니다. 서버는 기본적으로
꺼져 있으며, `127.0.0.1`은 같은 PC에서만 접속할 수 있습니다. `0.0.0.0`으로 바꾸면
LAN에 공개되므로 방화벽과 네트워크 접근 범위를 별도로 제한해야 합니다.
내장 서버는 TLS를 종료하지 않는 일반 HTTP 서버이므로 token, prompt, 입력 이미지가 평문으로
전송됩니다. 신뢰할 수 있는 LAN에서만 사용하거나 HTTPS reverse proxy/VPN 뒤에 두세요.
'시작'과 '중지'는 현재 앱 세션의 실행 상태만 바꾸며 '앱 시작' 값은 변경하지 않습니다.

## Target 규칙

- `active`: 요청을 제출한 시점의 현재 앱 백엔드를 사용합니다.
- 등록 target: 설정에서 승인한 ID의 Forge/WebUI 또는 ComfyUI를 사용합니다.
- target URL은 경로가 없는 서버 root(예: `http://192.168.0.20:7860`)여야 합니다.
- 원격 Forge/A1111은 `--api`로 API가 활성화되어 있어야 하며, ComfyUI도 해당 주소에서 API 요청을 받아야 합니다.
- 요청 본문에는 target URL, 로컬 경로 또는 Comfy workflow JSON을 넣을 수 없습니다.
- 일반 ComfyUI T2I/I2I target은 설정의 'T2I 워크플로 프로필 경로'·'I2I 워크플로 프로필 경로'에 각각 API-format workflow JSON 경로를 저장해야 합니다.
- `family: "krea2"`는 ComfyUI target에서만 동작하며 앱의 내장 Krea2 workflow builder를 사용합니다.
- Krea2 API 작업은 현재 요청당 이미지 1장만 지원하며 batch 값이 1보다 크면 `400`으로 거부합니다.
- 같은 로컬 GPU를 사용하는 작업은 앱의 생성 리소스 잠금을 공유합니다.

## 네이티브 작업 API

모든 경로는 `/api/v1/health`를 제외하고 다음 헤더가 필요합니다.

```text
Authorization: Bearer <설정에서 복사한 token>
```

| Method | Path | 용도 |
|---|---|---|
| `GET` | `/api/v1/health` | 서버 생존 확인 |
| `GET` | `/api/v1/targets` | 승인된 target 목록 |
| `POST` | `/api/v1/generations` | 작업 제출 |
| `GET` | `/api/v1/generations` | 최근 작업 목록 |
| `GET` | `/api/v1/generations/{jobId}` | 상태·진행률·결과 조회 |
| `GET` | `/api/v1/generations/{jobId}/artifacts/{index}` | 결과 파일 다운로드 |
| `DELETE` | `/api/v1/generations/{jobId}` | 대기/실행 작업 취소 |

제출은 기본적으로 즉시 `202 Accepted`를 반환합니다. `?wait=120`처럼 초를 지정하면 최대
600초까지 기다린 뒤 완료된 작업은 `200 OK`, 아직 실행 중이면 `202 Accepted`를 반환합니다.

### PowerShell T2I 예시

```powershell
$base = 'http://127.0.0.1:17990'
$token = '<설정에서 복사한 token>'
$headers = @{ Authorization = "Bearer $token" }
$body = @{
  target = 'active'
  mode = 'txt2img'
  family = 'standard'
  model = ''
  payload = @{
    prompt = 'a red fox in snow, detailed'
    negative_prompt = 'low quality, watermark'
    width = 1024
    height = 1024
    steps = 24
    cfg_scale = 5.5
    seed = -1
    sampler_name = 'Euler'
  }
} | ConvertTo-Json -Depth 8

$job = Invoke-RestMethod `
  -Uri "$base/api/v1/generations?wait=120" `
  -Method Post -Headers $headers -ContentType 'application/json' -Body $body
$job
```

비동기 제출 후 직접 기다리려면 다음처럼 조회합니다.

```powershell
do {
  Start-Sleep -Milliseconds 500
  $job = Invoke-RestMethod -Uri "$base/api/v1/generations/$($job.id)" -Headers $headers
} while ($job.state -in @('queued', 'running'))

if ($job.state -eq 'completed') {
  Invoke-WebRequest `
    -Uri "$base$($job.artifacts[0].url)" `
    -Headers $headers -OutFile '.\result.png'
}
```

### I2I 예시

`init_images`에는 로컬 경로나 URL이 아니라 base64 이미지 데이터만 넣습니다.

```powershell
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes('.\input.png'))
$body = @{
  target = 'remote-comfy'
  mode = 'img2img'
  family = 'standard'
  payload = @{
    prompt = 'watercolor illustration'
    negative_prompt = 'watermark'
    init_images = @($image)
    denoising_strength = 0.55
    width = 1024
    height = 1024
    steps = 20
    cfg_scale = 5
    seed = 42
  }
} | ConvertTo-Json -Depth 8

$job = Invoke-RestMethod `
  -Uri "$base/api/v1/generations" `
  -Method Post -Headers $headers -ContentType 'application/json' -Body $body
```

### Krea2 예시

```json
{
  "target": "remote-comfy",
  "mode": "txt2img",
  "family": "krea2",
  "payload": {
    "prompt": "cinematic portrait, city lights",
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "cfg_scale": 1,
    "seed": 42,
    "sampler_name": "euler"
  }
}
```

## A1111/Forge 생성 API 호환 subset

기존 A1111 클라이언트는 같은 Bearer 헤더와 다음 경로를 사용할 수 있습니다.
전체 A1111 관리 API가 아니라 생성·진행률·중단·모델 옵션에 필요한 subset입니다.

| Method | Path |
|---|---|
| `POST` | `/sdapi/v1/txt2img` |
| `POST` | `/sdapi/v1/img2img` |
| `GET` | `/sdapi/v1/progress` |
| `POST` | `/sdapi/v1/interrupt` |
| `GET` | `/sdapi/v1/options` |

대상을 생략하면 설정의 '기본 대상'을 사용합니다. 호출별로 바꾸려면
`X-AIStudio-Target: <targetId>` 헤더를 추가하거나 요청 JSON에
`ai_studio_target`을 넣습니다. `generation_family: "krea2"`도 호환 요청에서 사용할 수
있습니다. 생성 응답은 A1111 형식의 `images`, `parameters`, `info`를 반환합니다.

## 결과와 복구

- 결과 이미지는 `user_data/generation_api/results/<jobId>/` 아래에 별도로 저장됩니다.
- 결과 폴더는 자동 삭제하지 않습니다. 디스크 정리가 필요하면 앱을 종료한 뒤 보존할 결과를 확인하고 직접 정리합니다.
- API에는 로컬 파일 경로 대신 인증이 필요한 artifact URL만 반환됩니다.
- 앱이 비정상 종료되면 완료되지 않은 이전 작업은 다음 시작 시 `failed`로 복구됩니다.
- 취소는 실행 중인 해당 backend에만 best-effort interrupt를 전달합니다. ComfyUI는 gateway가
  제출한 prompt ID를 확인하지만, A1111/Forge의 interrupt는 서버 전역 API이므로 다른 클라이언트와
  공유하는 target에서는 그 클라이언트의 동시 작업도 중단될 수 있습니다. 취소 격리가 중요하면
  gateway 전용 Forge/WebUI 인스턴스를 사용하세요.
- WebUI가 여러 이미지를 반환하거나 ComfyUI가 여러 미디어를 반환하면 모두 artifact로 보존합니다.

## 보안 제한

- Bearer 인증은 끌 수 없습니다.
- CORS는 기본적으로 제공하지 않습니다.
- 요청 본문, 이미지 크기, 해상도, steps, batch 수와 작업 큐에 상한이 있습니다.
- 외부 요청으로 runtime 설치·업데이트·확장 설치·프로세스 실행은 할 수 없습니다.
- 요청에서 임의 URL, callback, 로컬 파일 경로, raw Comfy workflow를 전달할 수 없습니다.
- token과 target 설정은 Git에서 제외되는 `user_data/generation_api.json`에 저장됩니다.
