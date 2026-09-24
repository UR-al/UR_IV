/**
 * 하단 EXIF 바(Positive / Negative / Parameters)가 쓰는 메타데이터 — getImageExif 응답에서
 * 필요한 칸만 꺼낸다. 히스토리 선택과 생성 직후 자동 로드가 같은 모양을 쓴다(App.vue 에서 추출).
 */
export interface ExifParams { generation?: string; core?: string; model?: string; hires?: string; extensions?: string; other?: string; [k: string]: any }
export interface ExifData { prompt: string; negative: string; raw: string; params?: ExifParams | null; params_line?: string; [k: string]: any }

/** getImageExif 가 돌려준 객체 → EXIF 바 데이터(빠진 칸은 빈 값). */
export function exifFromPayload(d: any): ExifData {
  return {
    prompt: d?.prompt || '',
    negative: d?.negative || '',
    raw: d?.raw || '',
    params: d?.params || null,
    params_line: d?.params_line || '',
  }
}

/** 탭별 본문 — Positive 는 비어 있으면 'No EXIF data'. (Parameters 탭의 구조화 표시는 템플릿이 따로 그린다) */
export function exifTabContent(exif: ExifData, tab: string): string {
  if (tab === 'positive') return exif.prompt || 'No EXIF data'
  if (tab === 'negative') return exif.negative || ''
  return exif.raw || ''
}
