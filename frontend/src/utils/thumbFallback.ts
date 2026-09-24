/**
 * 카드 썸네일(Qt aithumb: / 웹 /thumbnail)을 못 받으면 원본 URL 로 한 번만 바꾼다.
 *
 * 썸네일 스킴이 등록되지 않은 실행 경로, Pillow 가 못 읽는 포맷, 캐시 폴더 쓰기 실패 같은
 * 경우에도 카드가 깨진 그림으로 남지 않게 한다. 원본까지 실패하면 더 바꾸지 않는다(무한 루프 방지).
 */
export interface ImgLike {
  getAttribute(name: string): string | null
  setAttribute(name: string, value: string): void
  dataset: Record<string, string | undefined>
}

export function fallbackToOriginal(target: EventTarget | ImgLike | null, originalUrl: string): boolean {
  const img = target as ImgLike | null
  if (!img || typeof img.getAttribute !== 'function' || !originalUrl) return false
  const failed = img.getAttribute('src') || ''
  if (!failed || failed === originalUrl || img.dataset.thumbFallbackFrom === failed) return false
  img.dataset.thumbFallbackFrom = failed
  img.setAttribute('src', originalUrl)
  return true
}
