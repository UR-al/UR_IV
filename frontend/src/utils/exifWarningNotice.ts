/**
 * SAM3/ADetailer 배치 'EXIF 프롬프트 사용' 경고 알림.
 *
 * 워커는 메타데이터를 못 읽었거나(ComfyUI 그래프가 모호함·JPEG 에 메타 없음 등) 결과 JSON 에
 * ``exif_warning`` 을 싣는다. 예전엔 빈 프롬프트로 조용히 인페인트가 돌았다. 배치에서 장마다
 * 토스트를 띄우면 도배가 되므로 실행마다 첫 경고만 알리고, 끝나면 몇 장이었는지 한 번 더 알린다.
 */
import { filenameOf } from './mediaKind'

export interface ExifWarningResult {
  exif_warning?: unknown
  before?: unknown
  path?: unknown
}

export function createExifWarningNotice(label: string) {
  let shown = false
  let count = 0
  return {
    /** 새 실행 시작 */
    reset() { shown = false; count = 0 },
    /** 결과 하나 — 이번 실행의 첫 경고면 토스트 문구, 아니면 null */
    note(result: ExifWarningResult | null | undefined): string | null {
      const warning = typeof result?.exif_warning === 'string' ? result.exif_warning.trim() : ''
      if (!warning) return null
      count++
      if (shown) return null
      shown = true
      const name = filenameOf(String(result?.before || result?.path || ''))
      return `${label}${name ? ` · ${name}` : ''}: ${warning}`
    },
    /** 실행 끝 — 경고가 둘 이상이었으면 요약 문구 */
    summary(): string | null {
      return count > 1 ? `${label}: EXIF 프롬프트를 적용하지 못한 이미지 ${count}장 (빈 프롬프트로 처리)` : null
    },
    get count() { return count },
  }
}
