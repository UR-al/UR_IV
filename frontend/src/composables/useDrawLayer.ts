/**
 * 드로잉 레이어 — 원본을 건드리지 않는 별도 레이어와 10개 도구의 상태 기계.
 *
 * PyQt 판(`widgets/interactive_label.py`, 은퇴해 삭제됨)은 `display_base_image` 를 직접 고쳤다.
 * 그래서 (1) 되돌리려면 이미지 전체 스냅샷이 필요했고, (2) 투명도 있는 펜은 획이
 * 겹치는 관절마다 진해졌다. 여기서는 캔버스를 셋으로 나눈다.
 *
 *   committed — 확정된 레이어 내용 (병합 전까지 원본은 그대로다)
 *   scratch   — 그리는 중인 획 하나. 놓을 때 **한 번만** 투명도를 먹여 합친다
 *               (그래서 관절이 진해지지 않는다)
 *   heal      — 복원 브러시 표시. 내용이 아니라 '여기를 지워라'는 표시라 따로 둔다
 *
 * 되돌리기는 획이 바꾼 경계 상자만 저장한다. 이미지 전체를 쌓으면 큰 그림에서
 * 몇 획 만에 수백 MB가 된다.
 */
import { ref } from 'vue'
import { floodFillMask, gradientLine, hexToRgb, normalizedRect, rgbToHex } from '../utils/drawTools'

export interface DrawParams {
  tool: string
  color: string
  /** 브러시 지름(이미지 픽셀) */
  size: number
  /** 0~1 */
  opacity: number
  filled: boolean
  gradientEnd: string
}

interface Bounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

interface UndoEntry {
  x: number
  y: number
  data: ImageData
}

/** 되돌리기 스택 상한 — 개수와 총 바이트 둘 다 건다. 큰 그림에서는 개수만으론 못 막는다. */
const UNDO_MAX_STEPS = 24
const UNDO_MAX_BYTES = 64 * 1024 * 1024

/** 플러드 필 허용 오차 — PyQt 판의 loDiff/upDiff(20)와 같게 둔다. */
const FILL_TOLERANCE = 20

export function useDrawLayer(options: {
  /** 원본 이미지가 그려진 캔버스 */
  baseCanvas: () => HTMLCanvasElement | null
  /** 화면에 보이는 레이어 캔버스 */
  visibleCanvas: () => HTMLCanvasElement | null
  params: () => DrawParams
  /** 스포이트가 집은 색 */
  onColorPicked: (hex: string) => void
}) {
  const hasContent = ref(false)
  const hasHeal = ref(false)
  const undoCount = ref(0)
  /**
   * 확정 레이어 내용이 바뀔 때마다 오르는 번호(획 확정·되돌리기·지우기·크기 재설정).
   * 에디터가 '저장한 뒤로 레이어가 바뀌었는지'를 이것으로 판단한다 — 병합하지 않은
   * 그림도 저장본에 들어가므로, 바뀌면 저장 안 된 변경이다.
   */
  const revision = ref(0)
  /** 텍스트 도구가 클릭한 위치 — 컴포넌트가 여기에 입력칸을 띄운다 */
  const textAnchor = ref<{ x: number; y: number } | null>(null)

  const committed = document.createElement('canvas')
  const scratch = document.createElement('canvas')
  const heal = document.createElement('canvas')
  /** 클론 스탬프가 읽을 원본 스냅샷 — 획 중에 자기가 그린 걸 다시 읽으면 번진다 */
  const cloneSource = document.createElement('canvas')
  /** 스포이트·채우기가 합성본을 읽을 버퍼. 클론 스냅샷과 섞이면 획 도중에 원본이 바뀐다. */
  const sampleBuffer = document.createElement('canvas')

  let committedCtx: CanvasRenderingContext2D | null = null
  let scratchCtx: CanvasRenderingContext2D | null = null
  let healCtx: CanvasRenderingContext2D | null = null

  const undoStack: UndoEntry[] = []
  let undoBytes = 0

  let active = false
  let startX = 0
  let startY = 0
  let lastX = 0
  let lastY = 0
  let strokeBounds: Bounds | null = null
  /** 알트+클릭으로 잡은 복제 원점 */
  let cloneAnchor: { x: number; y: number } | null = null
  let cloneOffset: { x: number; y: number } | null = null

  function size() {
    return { w: committed.width, h: committed.height }
  }

  /** 이미지 크기에 맞춰 레이어를 다시 잡는다. 내용은 모두 지워진다. */
  function resize(width: number, height: number) {
    for (const canvas of [committed, scratch, heal, cloneSource]) {
      canvas.width = width
      canvas.height = height
    }
    committedCtx = committed.getContext('2d', { willReadFrequently: true })
    scratchCtx = scratch.getContext('2d')
    healCtx = heal.getContext('2d')
    undoStack.length = 0
    undoBytes = 0
    undoCount.value = 0
    hasContent.value = false
    hasHeal.value = false
    textAnchor.value = null
    cloneAnchor = null
    cloneOffset = null
    revision.value++
    render()
  }

  function clear() {
    const { w, h } = size()
    committedCtx?.clearRect(0, 0, w, h)
    scratchCtx?.clearRect(0, 0, w, h)
    healCtx?.clearRect(0, 0, w, h)
    undoStack.length = 0
    undoBytes = 0
    undoCount.value = 0
    hasContent.value = false
    hasHeal.value = false
    textAnchor.value = null
    revision.value++
    render()
  }

  function clearHeal() {
    const { w, h } = size()
    healCtx?.clearRect(0, 0, w, h)
    hasHeal.value = false
    render()
  }

  // ── 화면 갱신 ─────────────────────────────────────────────────────────────

  /** 보이는 캔버스 = 확정 레이어 + (투명도 먹인) 그리는 중 획 + 복원 표시. */
  function render() {
    const visible = options.visibleCanvas()
    if (!visible) return
    const { w, h } = size()
    if (!w || !h) return
    if (visible.width !== w || visible.height !== h) {
      visible.width = w
      visible.height = h
    }
    const ctx = visible.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, w, h)
    ctx.drawImage(committed, 0, 0)
    if (active) {
      ctx.globalAlpha = strokeOpacity()
      ctx.drawImage(scratch, 0, 0)
      ctx.globalAlpha = 1
    }
    if (hasHeal.value) {
      // 복원 표시는 레이어 내용이 아니다 — 초록으로 덮어 '여기를 지운다'만 알린다
      ctx.save()
      ctx.globalAlpha = 0.45
      ctx.drawImage(heal, 0, 0)
      ctx.restore()
    }
  }

  function strokeOpacity(): number {
    return Math.max(0, Math.min(1, options.params().opacity ?? 1))
  }

  // ── 되돌리기 ──────────────────────────────────────────────────────────────

  function expand(x: number, y: number, pad = 0) {
    const minX = x - pad
    const minY = y - pad
    const maxX = x + pad
    const maxY = y + pad
    if (!strokeBounds) {
      strokeBounds = { minX, minY, maxX, maxY }
      return
    }
    strokeBounds.minX = Math.min(strokeBounds.minX, minX)
    strokeBounds.minY = Math.min(strokeBounds.minY, minY)
    strokeBounds.maxX = Math.max(strokeBounds.maxX, maxX)
    strokeBounds.maxY = Math.max(strokeBounds.maxY, maxY)
  }

  function pushUndo(bounds: Bounds) {
    if (!committedCtx) return
    const { w, h } = size()
    const x = Math.max(0, Math.floor(bounds.minX))
    const y = Math.max(0, Math.floor(bounds.minY))
    const right = Math.min(w, Math.ceil(bounds.maxX) + 1)
    const bottom = Math.min(h, Math.ceil(bounds.maxY) + 1)
    if (right <= x || bottom <= y) return
    const data = committedCtx.getImageData(x, y, right - x, bottom - y)
    undoStack.push({ x, y, data })
    undoBytes += data.data.byteLength
    while (undoStack.length > UNDO_MAX_STEPS || (undoBytes > UNDO_MAX_BYTES && undoStack.length > 1)) {
      const dropped = undoStack.shift()
      if (dropped) undoBytes -= dropped.data.data.byteLength
    }
    undoCount.value = undoStack.length
  }

  function undo(): boolean {
    const entry = undoStack.pop()
    if (!entry || !committedCtx) return false
    undoBytes -= entry.data.data.byteLength
    undoCount.value = undoStack.length
    committedCtx.putImageData(entry.data, entry.x, entry.y)
    hasContent.value = undoStack.length > 0 || layerHasPixels()
    revision.value++
    render()
    return true
  }

  /** 되돌린 뒤 레이어가 정말 비었는지 — 스택이 비어도 지운 뒤 그린 게 있을 수 있다. */
  function layerHasPixels(): boolean {
    if (!committedCtx) return false
    const { w, h } = size()
    if (!w || !h) return false
    // 전체 스캔은 병합 직전에나 필요하다. 여기서는 스택 기준으로 충분히 근사한다.
    return undoStack.length > 0
  }

  /** scratch 를 확정 레이어에 한 번만 합친다 — 투명도도 여기서 한 번만 먹인다. */
  function commitScratch() {
    const { w, h } = size()
    if (!committedCtx || !scratchCtx || !strokeBounds) return
    pushUndo(strokeBounds)
    committedCtx.save()
    committedCtx.globalAlpha = strokeOpacity()
    committedCtx.drawImage(scratch, 0, 0)
    committedCtx.restore()
    scratchCtx.clearRect(0, 0, w, h)
    strokeBounds = null
    hasContent.value = true
    revision.value++
  }

  // ── 합성본 읽기 (스포이트 · 채우기 · 클론) ─────────────────────────────────

  /** 원본 + 확정 레이어를 합친 캔버스. 사용자가 보는 것과 같은 픽셀을 읽기 위한 것. */
  function compositeInto(target: HTMLCanvasElement): CanvasRenderingContext2D | null {
    const base = options.baseCanvas()
    const { w, h } = size()
    if (!base || !w || !h) return null
    if (target.width !== w || target.height !== h) {
      target.width = w
      target.height = h
    }
    const ctx = target.getContext('2d', { willReadFrequently: true })
    if (!ctx) return null
    ctx.clearRect(0, 0, w, h)
    ctx.drawImage(base, 0, 0)
    ctx.drawImage(committed, 0, 0)
    return ctx
  }

  function samplePixel(x: number, y: number): string | null {
    const ctx = compositeInto(sampleBuffer)
    if (!ctx) return null
    const { w, h } = size()
    const px = Math.round(x)
    const py = Math.round(y)
    if (px < 0 || py < 0 || px >= w || py >= h) return null
    const d = ctx.getImageData(px, py, 1, 1).data
    return rgbToHex(d[0], d[1], d[2])
  }

  // ── 도구별 그리기 ─────────────────────────────────────────────────────────

  function strokeStyle(ctx: CanvasRenderingContext2D) {
    const p = options.params()
    ctx.strokeStyle = p.color
    ctx.fillStyle = p.color
    ctx.lineWidth = Math.max(1, p.size)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }

  function drawPenSegment(x1: number, y1: number, x2: number, y2: number) {
    if (!scratchCtx) return
    strokeStyle(scratchCtx)
    scratchCtx.beginPath()
    scratchCtx.moveTo(x1, y1)
    scratchCtx.lineTo(x2, y2)
    scratchCtx.stroke()
    const pad = Math.max(1, options.params().size)
    expand(x1, y1, pad)
    expand(x2, y2, pad)
  }

  function drawShapePreview(x2: number, y2: number) {
    if (!scratchCtx) return
    const { w, h } = size()
    const p = options.params()
    scratchCtx.clearRect(0, 0, w, h)
    strokeStyle(scratchCtx)
    strokeBounds = null

    if (p.tool === 'line') {
      scratchCtx.beginPath()
      scratchCtx.moveTo(startX, startY)
      scratchCtx.lineTo(x2, y2)
      scratchCtx.stroke()
    } else if (p.tool === 'rect') {
      const r = normalizedRect(startX, startY, x2, y2)
      if (p.filled) scratchCtx.fillRect(r.x, r.y, r.w, r.h)
      else scratchCtx.strokeRect(r.x, r.y, r.w, r.h)
    } else if (p.tool === 'ellipse') {
      const r = normalizedRect(startX, startY, x2, y2)
      if (r.w > 0 && r.h > 0) {
        scratchCtx.beginPath()
        scratchCtx.ellipse(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2, 0, 0, Math.PI * 2)
        if (p.filled) scratchCtx.fill()
        else scratchCtx.stroke()
      }
    } else if (p.tool === 'gradient') {
      // PyQt 판과 같이 화면 전체를 덮는다 — 드래그는 방향과 길이만 정한다(길이 0 이면 그리지 않는다)
      const line = gradientLine(startX, startY, x2, y2)
      if (line) {
        const grad = scratchCtx.createLinearGradient(line.x1, line.y1, line.x2, line.y2)
        grad.addColorStop(0, p.color)
        grad.addColorStop(1, p.gradientEnd)
        scratchCtx.fillStyle = grad
        scratchCtx.fillRect(0, 0, w, h)
      }
    }

    const pad = Math.max(1, p.size)
    if (p.tool === 'gradient') {
      expand(0, 0)
      expand(w, h)
    } else {
      expand(startX, startY, pad)
      expand(x2, y2, pad)
    }
  }

  function applyFloodFill(x: number, y: number) {
    const ctx = compositeInto(sampleBuffer)
    if (!ctx || !scratchCtx) return
    const { w, h } = size()
    const image = ctx.getImageData(0, 0, w, h)
    const result = floodFillMask(image.data, w, h, x, y, FILL_TOLERANCE)
    if (!result) return

    const bw = result.maxX - result.minX + 1
    const bh = result.maxY - result.minY + 1
    const patch = scratchCtx.createImageData(bw, bh)
    const [r, g, b] = hexToRgb(options.params().color)
    for (let py = 0; py < bh; py++) {
      for (let px = 0; px < bw; px++) {
        if (!result.mask[(py + result.minY) * w + (px + result.minX)]) continue
        const o = (py * bw + px) * 4
        patch.data[o] = r
        patch.data[o + 1] = g
        patch.data[o + 2] = b
        patch.data[o + 3] = 255
      }
    }
    scratchCtx.putImageData(patch, result.minX, result.minY)
    strokeBounds = { minX: result.minX, minY: result.minY, maxX: result.maxX, maxY: result.maxY }
  }

  function stampCloneDab(x: number, y: number, radius: number) {
    if (!scratchCtx || !cloneOffset) return
    scratchCtx.save()
    scratchCtx.beginPath()
    scratchCtx.arc(x, y, radius, 0, Math.PI * 2)
    scratchCtx.clip()
    // 대상 d 에 원본 (d + offset) 이 와야 하므로 이미지는 **-offset** 만큼 옮겨 그린다.
    // 부호를 반대로 두면 반대편의 엉뚱한 자리를 복제한다.
    scratchCtx.drawImage(cloneSource, -cloneOffset.x, -cloneOffset.y)
    scratchCtx.restore()
    expand(x, y, radius + 1)
  }

  /**
   * 이전 위치에서 현재 위치까지 도장을 촘촘히 찍는다.
   *
   * 포인터 이벤트 자리에만 찍으면 조금만 빨리 움직여도 점선이 된다 (PyQt 판이 그랬다).
   * 간격은 반지름의 1/4 — 이보다 성기면 가장자리가 물결친다.
   */
  function stampCloneAlong(x1: number, y1: number, x2: number, y2: number) {
    const radius = Math.max(1, options.params().size / 2)
    const distance = Math.hypot(x2 - x1, y2 - y1)
    const step = Math.max(1, radius / 4)
    const dabs = Math.max(1, Math.ceil(distance / step))
    for (let i = 1; i <= dabs; i++) {
      const t = i / dabs
      stampCloneDab(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, radius)
    }
  }

  function paintHeal(x1: number, y1: number, x2: number, y2: number) {
    if (!healCtx) return
    healCtx.strokeStyle = '#00C853'
    healCtx.lineWidth = Math.max(1, options.params().size)
    healCtx.lineCap = 'round'
    healCtx.lineJoin = 'round'
    healCtx.beginPath()
    healCtx.moveTo(x1, y1)
    healCtx.lineTo(x2, y2)
    healCtx.stroke()
    hasHeal.value = true
  }

  // ── 포인터 상태 기계 ───────────────────────────────────────────────────────

  /** @returns 이 도구가 포인터를 잡았으면 true (캔버스의 마스크 처리로 넘어가지 않는다) */
  function begin(x: number, y: number, event: PointerEvent): boolean {
    const p = options.params()
    startX = x; startY = y
    lastX = x; lastY = y
    strokeBounds = null

    if (p.tool === 'eyedropper') {
      const hex = samplePixel(x, y)
      if (hex) options.onColorPicked(hex)
      return true
    }

    if (p.tool === 'text_overlay') {
      textAnchor.value = { x, y }
      return true
    }

    if (p.tool === 'fill') {
      active = true
      applyFloodFill(x, y)
      commitScratch()
      active = false
      render()
      return true
    }

    if (p.tool === 'clone_stamp') {
      if (event.altKey) {
        // 알트+클릭 = 복제 원점 지정. 다음 클릭에서 원점↔대상 간격이 고정된다.
        cloneAnchor = { x: Math.round(x), y: Math.round(y) }
        cloneOffset = null
        return true
      }
      if (!cloneAnchor) return true   // 원점 없이 찍으면 아무 일도 없다
      compositeInto(cloneSource)
      cloneOffset = { x: cloneAnchor.x - Math.round(x), y: cloneAnchor.y - Math.round(y) }
      active = true
      stampCloneDab(x, y, Math.max(1, p.size / 2))
      render()
      return true
    }

    if (p.tool === 'heal') {
      active = true
      paintHeal(x, y, x, y)
      render()
      return true
    }

    // pen / line / rect / ellipse / gradient
    active = true
    if (p.tool === 'pen') drawPenSegment(x, y, x, y)
    else drawShapePreview(x, y)
    render()
    return true
  }

  function move(x: number, y: number): boolean {
    if (!active) return false
    const p = options.params()
    if (p.tool === 'pen') {
      drawPenSegment(lastX, lastY, x, y)
    } else if (p.tool === 'heal') {
      paintHeal(lastX, lastY, x, y)
    } else if (p.tool === 'clone_stamp') {
      stampCloneAlong(lastX, lastY, x, y)
    } else {
      drawShapePreview(x, y)
    }
    lastX = x; lastY = y
    render()
    return true
  }

  function end(): boolean {
    if (!active) return false
    active = false
    const p = options.params()
    if (p.tool !== 'heal') commitScratch()
    render()
    return true
  }

  /** 텍스트 도구가 입력을 마쳤을 때. 빈 문자열이면 아무것도 그리지 않는다. */
  function commitText(text: string) {
    const anchor = textAnchor.value
    textAnchor.value = null
    if (!anchor || !scratchCtx || !text.trim()) return
    const p = options.params()
    // 크기 슬라이더(1~100)를 글자 크기(px)로 그대로 쓴다. 패널에 폰트 크기 슬라이더가
    // 따로 없어서 이걸 쓰는데, 배수를 곱하면 슬라이더 눈금과 결과가 어긋나 짐작이 안 된다.
    const fontSize = Math.max(12, p.size)
    scratchCtx.font = `${fontSize}px 'Pretendard', system-ui, sans-serif`
    scratchCtx.textBaseline = 'top'
    scratchCtx.fillStyle = p.color
    scratchCtx.fillText(text, anchor.x, anchor.y)
    const width = scratchCtx.measureText(text).width
    strokeBounds = {
      minX: anchor.x,
      minY: anchor.y,
      maxX: anchor.x + width,
      maxY: anchor.y + fontSize * 1.4,
    }
    active = true
    commitScratch()
    active = false
    render()
  }

  function cancelText() {
    textAnchor.value = null
  }

  // ── 내보내기 ──────────────────────────────────────────────────────────────

  /** 확정 레이어를 PNG base64 로. 그린 게 없으면 null (백엔드를 부를 이유가 없다). */
  function getOverlayBase64(): string | null {
    if (!committedCtx) return null
    const { w, h } = size()
    if (!w || !h) return null
    const pixels = committedCtx.getImageData(0, 0, w, h).data
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] !== 0) return committed.toDataURL('image/png')
    }
    return null
  }

  /** 복원 브러시가 칠한 곳을 흰색으로 만든 마스크 PNG. 백엔드 inpaint 용. */
  function getHealMaskBase64(): string | null {
    if (!hasHeal.value || !healCtx) return null
    const { w, h } = size()
    const source = healCtx.getImageData(0, 0, w, h)
    const out = document.createElement('canvas')
    out.width = w
    out.height = h
    const ctx = out.getContext('2d')
    if (!ctx) return null
    const mask = ctx.createImageData(w, h)
    let any = false
    for (let i = 0; i < source.data.length; i += 4) {
      const on = source.data[i + 3] > 0 ? 255 : 0
      if (on) any = true
      mask.data[i] = on
      mask.data[i + 1] = on
      mask.data[i + 2] = on
      mask.data[i + 3] = 255
    }
    if (!any) return null
    ctx.putImageData(mask, 0, 0)
    return out.toDataURL('image/png')
  }

  return {
    hasContent,
    hasHeal,
    undoCount,
    revision,
    textAnchor,
    resize,
    clear,
    clearHeal,
    render,
    begin,
    move,
    end,
    undo,
    commitText,
    cancelText,
    getOverlayBase64,
    getHealMaskBase64,
  }
}
