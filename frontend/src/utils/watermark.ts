/**
 * 워터마크 패널의 순수 규칙 — 글꼴 목록·파일 이름 표시·프리뷰 재요청 대상.
 *
 * 글꼴은 **표시명**으로 보낸다. 백엔드(core/editor_watermark.py FONT_FILES)가 표시명을
 * 글꼴 파일로 바꾼다 — 예전에는 표시명을 그대로 파일로 열려다 전부 실패해 늘 10px
 * 기본 글꼴이었다. 이 목록의 이름은 모두 FONT_FILES 키여야 한다(tests/test_editor_watermark.py).
 */
export const WATERMARK_FONTS: readonly string[] = [
  'Arial', 'Times New Roman', 'Courier New', 'Verdana', 'Georgia', '맑은 고딕',
]

export const NO_WATERMARK_IMAGE = '이미지 없음'

/** 워터마크 이미지 경로 → 패널에 보일 파일 이름. 없으면 '이미지 없음'. */
export function watermarkFileLabel(path: string | null | undefined, empty = NO_WATERMARK_IMAGE): string {
  const text = String(path ?? '').trim()
  if (!text) return empty
  const name = text.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || ''
  return name || empty
}

export type WatermarkKind = 'text' | 'image'

/**
 * 공용 옵션(영역 제한·글자 색)이 바뀌었을 때 다시 그릴 프리뷰.
 * 마지막으로 보여 준 종류를 다시 그린다 — 텍스트는 글자가 있을 때만, 이미지는 파일이 있을 때만.
 */
export function watermarkPreviewToRefresh(
  last: WatermarkKind | null, hasText: boolean, hasImage: boolean,
): WatermarkKind | null {
  if (last === 'text') return hasText ? 'text' : null
  if (last === 'image') return hasImage ? 'image' : null
  return null
}
