import { afterEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import source from './CompareSlider.vue?raw'
import { byClass, mountFake, type Mounted } from '../testing/fakeDomRenderer'
import { compileSfc } from '../testing/compileSfc'

let mounted: Mounted | null = null

afterEach(() => {
  mounted?.unmount()
  mounted = null
  vi.unstubAllGlobals()
})

describe('CompareSlider (U-08 / U-15)', () => {
  it('clips after image on the left so before stays left and after stays right, and supports keyboard + window pointerup release', async () => {
    const winListeners = new Map<string, Set<(...args: any[]) => void>>()
    vi.stubGlobal('window', {
      addEventListener: (type: string, fn: (...args: any[]) => void) => {
        if (!winListeners.has(type)) winListeners.set(type, new Set())
        winListeners.get(type)!.add(fn)
      },
      removeEventListener: (type: string, fn: (...args: any[]) => void) => {
        winListeners.get(type)?.delete(fn)
      },
    })

    const comp = compileSfc(source, 'compare-slider-test', {})
    mounted = mountFake(comp, { beforeSrc: 'before.png', afterSrc: 'after.png' })
    await nextTick()

    const container = byClass(mounted.root, 'compare-container')[0]
    const afterImg = byClass(mounted.root, 'after')[0]
    expect(container.props.role).toBe('slider')
    expect(afterImg.props.style).toEqual({ clipPath: 'inset(0 0 0 50%)' })

    // 키보드 좌/우/Home/End 조절
    container.props.onKeydown({ key: 'ArrowRight', shiftKey: true, preventDefault: vi.fn() })
    await nextTick()
    expect(byClass(mounted.root, 'after')[0].props.style).toEqual({ clipPath: 'inset(0 0 0 60%)' })

    container.props.onKeydown({ key: 'Home', shiftKey: false, preventDefault: vi.fn() })
    await nextTick()
    expect(byClass(mounted.root, 'after')[0].props.style).toEqual({ clipPath: 'inset(0 0 0 0%)' })

    container.props.onKeydown({ key: 'End', shiftKey: false, preventDefault: vi.fn() })
    await nextTick()
    expect(byClass(mounted.root, 'after')[0].props.style).toEqual({ clipPath: 'inset(0 0 0 100%)' })

    // 포인터 드래그 시작 → window pointerup 으로 외부 해제
    const setPointerCapture = vi.fn()
    ;(container as any).getBoundingClientRect = () => ({ left: 0, width: 100 })
    container.props.onPointerdown({
      pointerId: 7,
      clientX: 25,
      currentTarget: { setPointerCapture },
    })
    await nextTick()
    expect(setPointerCapture).toHaveBeenCalledWith(7)
    expect(byClass(mounted.root, 'after')[0].props.style).toEqual({ clipPath: 'inset(0 0 0 25%)' })
    expect(winListeners.get('pointerup')?.size).toBe(1)

    for (const fn of [...(winListeners.get('pointerup') || [])]) fn({ pointerId: 7 })
    expect(winListeners.get('pointerup')?.size).toBe(0)
  })
})
