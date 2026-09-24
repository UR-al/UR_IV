/**
 * 에디터 실시간 프리뷰의 순수 계산 — 마스크 축소, 늦게 온 프리뷰 거르기.
 *
 * 백엔드(vue_bridge._editor_process_impl)는 프리뷰를 긴 변 PREVIEW_MAX_EDGE 로 줄여 처리하고
 * 마스크를 그 크기로 INTER_NEAREST 리사이즈한다. 그러니 프론트가 원본 해상도 마스크를
 * PNG 로 인코딩해 보낼 이유가 없다 — 같은 크기로 먼저 줄여 보낸다(4K 마스크 PNG 인코딩은
 * 슬라이더 틱마다 수십 ms 였다).
 */

/** vue_bridge.VueBridge._PREVIEW_MAX_EDGE 와 같은 값이어야 한다. */
export const PREVIEW_MAX_EDGE = 1024

/** 백엔드 프리뷰 축소와 같은 식(긴 변 기준 비율, 버림, 최소 1). */
export function previewDims(w: number, h: number, maxEdge = PREVIEW_MAX_EDGE): { w: number; h: number } {
  const long = Math.max(w, h)
  if (!(long > maxEdge)) return { w, h }
  const r = maxEdge / long
  return { w: Math.max(1, Math.floor(w * r)), h: Math.max(1, Math.floor(h * r)) }
}

/**
 * 최근접 이웃 축소. 픽셀 중심을 샘플링한다 — 백엔드 INTER_NEAREST 와 같은 성질이라
 * 프리뷰 마스크가 확정 결과와 한두 픽셀 이상 어긋나지 않는다.
 */
export function downscaleMaskNearest(
  mask: Uint8Array, w: number, h: number, tw: number, th: number,
): Uint8Array {
  const out = new Uint8Array(tw * th)
  if (!w || !h || !tw || !th) return out
  const sx = w / tw
  const sy = h / th
  const cols = new Uint32Array(tw)
  for (let x = 0; x < tw; x++) cols[x] = Math.min(w - 1, Math.floor((x + 0.5) * sx))
  for (let y = 0; y < th; y++) {
    const srcRow = Math.min(h - 1, Math.floor((y + 0.5) * sy)) * w
    const outRow = y * tw
    for (let x = 0; x < tw; x++) out[outRow + x] = mask[srcRow + cols[x]]
  }
  return out
}

/**
 * 프리뷰 세대 번호. 프리뷰를 걷거나(탭 전환·리셋·도구 변경) 확정 작업을 보내면 세대를 올리고,
 * 그 전에 보낸 프리뷰 요청의 결과는 도착해도 버린다.
 *
 * job_id 가드만으로는 못 잡는다 — '걷기'는 새 요청이 아니라서 job_id 가 오르지 않는다.
 * 그래서 요청에 `preview_token` 을 실어 보내고 백엔드가 결과에 그대로 돌려준다.
 */
export class PreviewGate {
  private epoch = 0

  /** 지금 보내는 프리뷰 요청에 실을 토큰 */
  current(): number {
    return this.epoch
  }

  /** 이전에 보낸 프리뷰는 모두 무효 */
  invalidate(): void {
    this.epoch++
  }

  /** 결과의 토큰이 지금 세대인지. 토큰이 없는 결과(옛 백엔드)는 막지 않는다. */
  accepts(token: unknown): boolean {
    if (typeof token !== 'number') return true
    return token === this.epoch
  }
}

/** 프리뷰를 걷는 순간 캔버스 상태 — `shouldDeferPreviewHide` 입력. */
export interface PreviewHideState {
  /** 프리뷰 레이어가 지금 화면에 보이는지 */
  previewShown: boolean
  /** 캔버스가 요청한 원본이 아직 디코드 중인지 */
  baseLoadPending: boolean
  /** 부모가 지금 넘긴 원본 src (props.imageSrc) */
  imageSrc: string
  /** 캔버스가 마지막으로 로드를 시작한 원본 src */
  requestedImageSrc: string
}

/**
 * 프리뷰를 바로 숨기지 말고 새 원본이 그려질 때까지 남겨 둘지.
 *
 * 확정 작업(모자이크 '적용' 등)의 결과가 오면 부모는 같은 tick 에 프리뷰를 걷고 새 원본을
 * 넘긴다. 프리뷰를 즉시 숨기면 base 캔버스에 아직 남아 있는 '적용 전' 원본이 새 이미지가
 * 디코드될 때까지(큰 PNG 50~300ms) 비친다 — 적용할 때마다 원본이 번쩍인다.
 * 원본이 바뀌는 중일 때만 미룬다. 탭 전환·리셋처럼 원본이 그대로면 바로 숨긴다.
 * (props 감시 순서는 보장되지 않는다 — 원본 감시가 아직 안 돌았어도 src 가 다르면 '바뀌는 중')
 */
export function shouldDeferPreviewHide(s: PreviewHideState): boolean {
  if (!s.previewShown) return false
  if (s.baseLoadPending) return true
  return !!s.imageSrc && s.imageSrc !== s.requestedImageSrc
}
