/**
 * file URL → 글자 그대로의 로컬 경로 (순수 로직).
 *
 * 백엔드 core/path_safety.strip_file_url 과 같은 규칙이다:
 *  - `file://`·`file:///`(대소문자 무관)로 시작하면 스킴을 떼고 퍼센트 인코딩을 푼다.
 *  - 원시 경로는 글자 그대로 둔다 — 원시 Windows 경로의 `%` 는 이름의 일부다
 *    (웹에서 받은 `image%20(1).png`). 디코드하면 다른 파일(`image (1).png`)을 가리킨다.
 *
 * 디코드는 Python `unquote` 처럼 관대하다: 깨진 시퀀스(`100%.png` 를 그냥 붙인 file URL)는
 * 그 자리만 글자 그대로 두고 나머지(`a%20b`)는 푼다. 올바른 UTF-8 이 아닌 `%XX` 묶음도 원문 유지.
 * (표시용 src 는 utils/media.js 가 따로 만든다 — 여기는 비교·백엔드 전달용 경로다)
 */
export function stripFileUrl(path: string | null | undefined): string {
  const text = String(path ?? '')
  const scheme = /^file:\/\/\/?/i.exec(text)
  if (!scheme) return text
  return decodePercentRuns(text.slice(scheme[0].length))
}

/** 이어진 `%XX` 묶음마다 따로 디코드 — 한 곳이 깨져도 나머지는 푼다(멀티바이트 UTF-8 은 묶음째). */
function decodePercentRuns(text: string): string {
  return text.replace(/(?:%[0-9a-f]{2})+/gi, run => {
    try { return decodeURIComponent(run) } catch { return run }
  })
}
