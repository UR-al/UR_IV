/**
 * 이미지/미디어 URL 헬퍼 — 실행 모드에 맞는 src 생성.
 *
 *  - Qt 임베드 모드(run_gui.bat): QWebEngineView 는 로컬 file:/// 를 직접 로드.
 *    → 기존과 100% 동일하게 `file:///<path>` 반환 (동작 보존).
 *  - 웹 모드(run_WEB_gui.bat): 원격 브라우저는 file:/// 를 못 읽으므로
 *    Python 정적 서버의 `/file?path=` 엔드포인트로 HTTP 서빙.
 *
 * 즉 컴포넌트는 경로(raw path)만 다루고, 화면 표시용 src 는 항상 이 헬퍼를 거친다.
 */

// 웹 모드 감지 — Python 정적 서버가 index.html 에 주입한 전역으로 판별.
// 호출 시점마다 window 를 확인한다(모듈 로드 시점 1회 캡처는 주입 타이밍에 취약).
function _detectWeb() {
  return typeof window !== 'undefined' &&
    !!(window.__AISTUDIO_WS_PORT__ || window.__AISTUDIO_WS_URL__)
}

/**
 * 입력을 '글자 그대로의 로컬 경로'와 'file URL' 로 가른다.
 * 퍼센트 인코딩은 file URL 에만 있다 — 원시 Windows 경로의 `%` 는 이름의 일부다(`image%20(1).png`).
 * (백엔드 core/path_safety.strip_file_url · utils/dropPaths.ts 와 같은 규칙)
 * @param {string} s @returns {{ isFileUrl: boolean, rest: string }} rest = 스킴을 뗀 나머지(인코딩은 그대로)
 */
function _splitFileUrl(s) {
  const isFileUrl = /^file:/i.test(s)
  return { isFileUrl, rest: isFileUrl ? s.replace(/^file:\/\/\/?/i, '') : s }
}

/** 퍼센트 디코딩 — 깨진 시퀀스(`100%.png` 를 그냥 붙인 file URL)는 원문 그대로 둔다. @param {string} s */
function _safeDecode(s) {
  try { return decodeURIComponent(s) } catch { return s }
}

/**
 * 백엔드에 넘길 글자 그대로의 로컬 경로 — file URL 이면 디코딩, 원시 경로는 그대로.
 * @param {string} s @returns {string}
 */
function _literalPath(s) {
  const { isFileUrl, rest } = _splitFileUrl(s)
  return isFileUrl ? _safeDecode(rest) : rest
}

/**
 * 로컬 파일 경로 → 화면 표시용 src.
 * @param {string} [path]  로컬 파일 경로(또는 이미 file:/// 가 붙은 경로)
 * @param {boolean} [bust]  true 면 캐시 무력화용 타임스탬프 쿼리 추가
 * @returns {string}
 */
export function mediaUrl(path, bust = false) {
  if (!path) return ''
  const s = String(path)
  // 이미 완성된 URL(생성결과 미리보기, data/blob 등)이면 그대로 둔다
  if (/^(https?:|data:|blob:)/i.test(s)) return s
  // file:// 접두사 제거 — file URL 이면 나머지는 인코딩된 채, 원시 경로면 글자 그대로
  const { isFileUrl, rest } = _splitFileUrl(s)
  let url
  if (_detectWeb()) {
    // 서버는 쿼리를 한 번만 디코딩해 글자 그대로의 경로로 쓴다 — file URL 은 여기서 먼저 푼다
    url = '/file?path=' + encodeURIComponent(isFileUrl ? _safeDecode(rest) : rest)
  } else {
    // 원시 경로의 `%` 는 이름의 일부다. 그대로 두면 QtWebEngine 이 `%XX` 를 디코딩해 다른 파일을 연다
    // (`C:/art%231/a.png` → `C:/art#1/a.png`). 그래서 `%` 를 먼저 `%25` 로 바꾼다 — 이미 인코딩된
    // file URL 은 이중 인코딩되지 않게 건드리지 않는다.
    // `#` 는 Windows 파일·폴더 이름에 쓸 수 있지만 URL 에선 프래그먼트 시작이다 — 그대로 두면
    // `C:/art#1/clip.mp4` 가 `C:/art` 로 잘려 로드에 실패하고, 뒤에 붙는 `?t=` 도 프래그먼트 안으로
    // 들어간다. 순서가 중요하다: `%` 다음에 `#`(먼저 `#`→`%23` 하면 그 `%` 가 다시 `%25` 가 된다).
    const escaped = isFileUrl ? rest : rest.replace(/%/g, '%25')
    url = 'file:///' + escaped.replace(/#/g, '%23')
  }
  if (bust) url += (url.includes('?') ? '&' : '?') + 't=' + Date.now()
  return url
}

/**
 * 갤러리·즐겨찾기 카드용 축소 URL.
 *  - 웹 모드: Python 정적 서버 `/thumbnail?path=&width=`
 *  - Qt 모드: `aithumb:thumb?path=&width=` 커스텀 스킴(ui/thumb_scheme.py) — 예전엔 원본
 *    file:/// 를 그대로 써서 200px 카드마다 1~2K PNG 를 풀해상도로 디코드했다.
 * 둘 다 core/thumb_cache.py 의 같은 캐시를 쓰고, 실패하면 카드가 원본으로 폴백한다
 * (utils/thumbFallback.ts). 애니메이션은 호출부가 원본(mediaUrl)을 쓴다.
 * `version`(utils/mediaVersions.ts mediaVersion)은 `v=` 로 붙는다 — URL 이 경로·폭만 담으면 원본을
 * 덮어써 백엔드가 썸네일을 다시 만들어도 브라우저가 같은 URL 의 옛 그림을 쓴다(Qt: 페이지 안
 * 이미지 캐시, 웹: Cache-Control max-age=3600). 백엔드는 v 를 읽지 않는다(캐시 키만 바꾼다).
 * @param {string} [path] @param {number} [width] @param {string} [version] @returns {string}
 */
export function thumbnailUrl(path, width = 384, version = '') {
  if (!path) return ''
  const s = String(path)
  if (/^(data:|blob:|https?:)/i.test(s)) return s
  // 백엔드(parse_thumb_url·/thumbnail)는 쿼리를 한 번만 디코딩한다 — 글자 그대로의 경로를 싣는다
  const clean = _literalPath(s)
  const query = 'path=' + encodeURIComponent(clean) + '&width=' + bucketThumbWidth(width)
  return withUrlVersion(_detectWeb() ? '/thumbnail?' + query : 'aithumb:thumb?' + query, version)
}

/**
 * URL 에 내용 버전(`v=`)을 붙인다 — 같은 경로의 파일이 바뀌면 URL 도 바뀌어 캐시가 새로 읽는다.
 * 버전이 없거나 data:/blob: URL 이면 그대로.
 * @param {string} url @param {string | number | null | undefined} [version] @returns {string}
 */
export function withUrlVersion(url, version) {
  if (!url || version === undefined || version === null || version === '') return url
  if (/^(data:|blob:)/i.test(url)) return url
  return url + (url.includes('?') ? '&' : '?') + 'v=' + encodeURIComponent(String(version))
}

/**
 * 갤러리 카드 썸네일 폭 버킷(px) — Python core/thumb_cache.py GALLERY_THUMB_BUCKETS 와 같아야 한다.
 * 슬라이더(100~380px) × DPR(최대 2) = 760px 까지 선명하게. 폭마다 캐시 파일이 생기므로
 * 요청 폭을 '같거나 큰' 가장 가까운 버킷으로 올린다(드래그 중 중간 폭이 캐시 변형을 쌓지 않게).
 */
export const GALLERY_THUMB_BUCKETS = [192, 256, 384, 512, 768]

/** @param {unknown} width @returns {number} */
export function bucketThumbWidth(width) {
  const n = Number(width)
  const value = Math.max(64, Math.min(1024, Math.round(Number.isFinite(n) && n > 0 ? n : 384)))
  return GALLERY_THUMB_BUCKETS.find(bucket => bucket >= value) ?? GALLERY_THUMB_BUCKETS[GALLERY_THUMB_BUCKETS.length - 1]
}

/** 웹 모드 여부 (컴포넌트에서 분기 필요 시) */
export function isWebMode() { return _detectWeb() }
