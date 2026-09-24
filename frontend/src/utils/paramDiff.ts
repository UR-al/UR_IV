/**
 * PNG Info '파라미터 차이' — getImageExif 의 ``parameters`` dict(core 가 따옴표를 풀어 파싱한 값)를
 * 키별로 비교한다.
 *
 * 예전엔 표시 문자열 ``params_line`` 을 따옴표를 모르는 정규식으로 다시 파싱해서
 * 'ADetailer prompt: "smile, blue eyes"' 가 'smile' 로 잘리고 'lora2' 같은 가짜 행이 생겼다.
 */
export interface ParamDiffRow {
  key: string
  before: string
  after: string
  changed: boolean
}

type Params = Record<string, unknown> | null | undefined

function display(value: unknown): string {
  if (value === null || value === undefined) return ''
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

/** 두 이미지의 파라미터 행 — 바뀐 것이 위로(같은 무리 안에서는 앞 이미지의 키 순서). */
export function diffParameters(before: Params, after: Params): ParamDiffRow[] {
  const b = before && typeof before === 'object' ? before : {}
  const a = after && typeof after === 'object' ? after : {}
  const keys = [...new Set([...Object.keys(b), ...Object.keys(a)])]
  const rows = keys.map(key => {
    const bv = display(b[key])
    const av = display(a[key])
    return { key, before: bv, after: av, changed: bv !== av }
  })
  return [...rows.filter(r => r.changed), ...rows.filter(r => !r.changed)]
}
