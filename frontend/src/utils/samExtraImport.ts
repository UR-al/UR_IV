/**
 * sam-extra 가져오기 — 붙여 넣은 infotext 의 카드 기능 값(Colorcraft·Anima SPEED)을 카드로 되살리는 브리지 액션 하나
 * (sam_extra_import — ui/sam_extra_import.py, 계획 X3c). 읽기·적용·토스트는 파이썬이 한다. 여기는 보낼 글을 만들고
 * 이미지 매개변수에 가져올 값이 있는지 본다(PNG Info·갤러리의 단추를 보일지).
 */
import type { SamExtraImportFeature, SamExtraImportPayload } from '../types/bridge'
import { requestAction } from '../stores/widgetStore.js'

/** 기능 → 그 기능을 알아보는 infotext 키(파이썬 ui/sam_extra_import.FEATURE_KEYS 의 거울 — 미러 시험이 본다). */
export const IMPORT_KEYS: Readonly<Record<SamExtraImportFeature, readonly string[]>> = {
  colorcraft: ['SAM Extra Colorcraft', 'Colorcraft'],
  speed: ['Anima SPEED'],
}
export const IMPORT_LABELS: Readonly<Record<SamExtraImportFeature, string>> = {
  colorcraft: 'Colorcraft',
  speed: 'Anima SPEED',
}
export const IMPORT_TITLE = '이 이미지의 sam-extra 값(Colorcraft·Anima SPEED)을 파라미터 카드로 가져온다 — 이미지에 있는 기능만 바꾼다'

/** 이미지 매개변수에 들어 있는(가져올 수 있는) 기능 — 순서는 IMPORT_KEYS. */
export function importableFeatures(params: Record<string, unknown> | null | undefined): SamExtraImportFeature[] {
  if (!params) return []
  return (Object.keys(IMPORT_KEYS) as SamExtraImportFeature[])
    .filter(feature => IMPORT_KEYS[feature].some(key => params[key] !== undefined && params[key] !== null))
}

/** 매개변수에서 가져올 키만 ``Key: value`` 줄로(값은 JSON 따옴표 — 쉼표·세미콜론이 든 값도 한 줄로 읽힌다). */
export function importText(params: Record<string, unknown>, features = importableFeatures(params)): string {
  const lines: string[] = []
  for (const feature of features) {
    for (const key of IMPORT_KEYS[feature]) {
      const value = params[key]
      if (value === undefined || value === null) continue
      lines.push(`${key}: ${JSON.stringify(String(value))}`)
    }
  }
  return lines.join('\n')
}

type Send = (action: 'sam_extra_import', payload: SamExtraImportPayload) => void

/** 액션을 보낸다(only 가 있으면 그 기능만). 빈 글은 보내지 않는다(false). ``send`` 는 시험용 대역. */
export function importSamExtra(text: string, only?: SamExtraImportFeature, send?: Send): boolean {
  const trimmed = String(text ?? '').trim()
  if (!trimmed) return false
  const payload: SamExtraImportPayload = only ? { text: trimmed, only } : { text: trimmed }
  if (send) send('sam_extra_import', payload)
  else requestAction('sam_extra_import', payload)
  return true
}
