/**
 * 히스토그램 계산 — DOM 의존 없는 순수 로직.
 *
 * PyQt 판(`tabs/editor/histogram_widget.py`, 은퇴해 삭제됨)은 `cv2.calcHist` 로 R/G/B 를 센 뒤
 * **전체 최댓값**으로 정규화했다. 그 정규화가 문제였다 — 배경이 단색인 그림 한 장이면
 * 그 색 하나가 만든 스파이크가 천장을 다 먹고 나머지 형태는 바닥에 눕는다.
 * 여기서는 위쪽 몇 개 빈을 빼고 천장을 잡는다(`normalizationCeiling`).
 * 스파이크는 잘려 나가고 정작 봐야 할 분포가 남는다.
 */

export const BINS = 256

export type ChannelKey = 'r' | 'g' | 'b' | 'lum'

export interface Histograms {
  r: Uint32Array
  g: Uint32Array
  b: Uint32Array
  lum: Uint32Array
  /** 계산에 실제로 쓴 픽셀 수 (완전 투명 픽셀 제외) */
  count: number
}

/** 빈 히스토그램 — 이미지가 없을 때의 중립값. */
export function emptyHistograms(): Histograms {
  return {
    r: new Uint32Array(BINS),
    g: new Uint32Array(BINS),
    b: new Uint32Array(BINS),
    lum: new Uint32Array(BINS),
    count: 0,
  }
}

/**
 * RGBA 픽셀 배열에서 채널별 256빈 히스토그램을 센다.
 *
 * 완전 투명 픽셀은 건너뛴다. 투명 영역의 RGB 는 보통 0 이라, 세어버리면
 * 화면에 보이지도 않는 픽셀이 좌단에 가짜 봉우리를 세운다.
 */
export function computeHistograms(data: Uint8ClampedArray): Histograms {
  const h = emptyHistograms()
  let count = 0
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    h.r[r]++
    h.g[g]++
    h.b[b]++
    // Rec.601 휘도의 정수 근사 — 77+150+29 = 256 이라 >>8 로 딱 떨어진다.
    h.lum[(r * 77 + g * 150 + b * 29) >> 8]++
    count++
  }
  h.count = count
  return h
}

/**
 * 정규화 천장(=그래프 높이 100%에 해당하는 빈 값) — 표시할 채널들의 최댓값.
 *
 * 채널마다 따로 정규화하면 R/G/B 높이를 서로 비교할 수 없다. 천장은 하나여야 한다.
 *
 * 위쪽 몇 개 빈을 빼서 스파이크를 잘라내는 방법을 먼저 썼다가 버렸다. 단색 면이 넓은
 * 그림(AI 그림에 흔하다)은 큰 빈 몇 개 + 낮고 평평한 꼬리 형태라, 위를 빼면 천장이
 * 꼬리 높이까지 내려앉아 **꼬리 전체가 최대 높이로 차오른다**. 봉우리와 구분이 안 된다.
 * 스파이크가 나머지를 눌러버리는 문제는 세로축을 로그로 두는 쪽이 정직하다 (`barRatio`).
 */
export function normalizationCeiling(h: Histograms, channels: ChannelKey[]): number {
  let ceiling = 1
  for (const key of channels) {
    for (let i = 0; i < BINS; i++) {
      if (h[key][i] > ceiling) ceiling = h[key][i]
    }
  }
  return ceiling
}

/**
 * 빈 값 → 그래프 높이 비율(0~1).
 *
 * 선형은 정직하지만, 단색 배경 한 덩어리가 천장을 다 먹으면 나머지 분포가 바닥에 눕는다.
 * 로그는 그 경우에도 형태를 남긴다. 0 은 로그에서도 0 이고 천장은 1 이라 눈금은 그대로다.
 */
export function barRatio(value: number, ceiling: number, log: boolean): number {
  if (ceiling <= 0) return 0
  const ratio = log
    ? Math.log1p(value) / Math.log1p(ceiling)
    : value / ceiling
  return Math.max(0, Math.min(1, ratio))
}

/**
 * 계조가 뭉개진 비율 — 0(순검정)과 255(순백)에 몰린 픽셀의 비율.
 *
 * 채널 하나만 터져도 그 색은 복구가 안 되므로 채널별 최댓값을 쓴다.
 * 바로 아래 블랙/화이트 포인트 슬라이더가 이 값을 늘리는 조작이라, 같이 봐야 한다.
 */
export function clippingRatio(h: Histograms): { shadow: number; highlight: number } {
  if (!h.count) return { shadow: 0, highlight: 0 }
  const shadow = Math.max(h.r[0], h.g[0], h.b[0]) / h.count
  const highlight = Math.max(h.r[BINS - 1], h.g[BINS - 1], h.b[BINS - 1]) / h.count
  return { shadow, highlight }
}

/**
 * 원본을 다 읽지 않기 위한 축소 크기. 긴 변을 `maxEdge` 로 맞춘다.
 *
 * 축소해도 분포 모양은 사실상 같다. 단 **보간을 꺼야** 한다 —
 * 보간하면 없던 중간값이 생겨 스파이크(계조 뭉갬)가 눈에 안 띄게 뭉개진다.
 */
export function sampleSize(width: number, height: number, maxEdge = 512): { w: number; h: number } {
  const longest = Math.max(width, height)
  if (!longest) return { w: 0, h: 0 }
  const scale = Math.min(1, maxEdge / longest)
  return {
    w: Math.max(1, Math.round(width * scale)),
    h: Math.max(1, Math.round(height * scale)),
  }
}
