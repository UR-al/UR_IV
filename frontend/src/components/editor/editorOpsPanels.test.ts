import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import ColorMatchPanel from './ColorMatchPanel.vue'
import TransformPanel from './TransformPanel.vue'

const IconStub = { props: ['name'], render: () => h('i') }

async function render(component: any, props: Record<string, unknown>) {
  const app = createSSRApp(component, props)
  app.component('Icon', IconStub)
  return renderToString(app)
}

describe('ColorMatchPanel', () => {
  it('starts in reference mode with the load button and the empty label', async () => {
    const html = await render(ColorMatchPanel, {})
    expect(html).toContain('참조 이미지 불러오기')
    expect(html).toContain('불러온 이미지 없음')
    expect(html).toContain('칠한 영역만')
    expect(html).toContain('색 맞추기 적용')
    expect(html).toContain('미리보기 해제')
    expect(html).not.toContain('주변 폭 px')
    // 참조 이미지 모드 기본값은 마스크가 필요 없다 — '먼저 칠하세요' 안내가 없어야 한다
    expect(html).not.toContain('먼저 브러시·올가미로 영역을 칠하세요')
  })

  it('shows the loaded file name, not the full path', async () => {
    const html = await render(ColorMatchPanel, { referencePath: 'C:/사진/참조.png' })
    expect(html).toContain('>참조.png<')
  })
})

describe('TransformPanel mesh section', () => {
  // PanelSection 이 열림 상태를 window.localStorage 에서 읽는다 — 빈 저장소로 둔다
  beforeEach(() => {
    vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } })
  })
  afterEach(() => { vi.unstubAllGlobals() })

  it('offers the start button and the pin checkbox (on by default)', async () => {
    const html = await render(TransformPanel, {})
    expect(html).toContain('격자 변형')
    expect(html).toContain('격자 점 맞추기 시작')
    expect(html).toMatch(/<input type="checkbox" checked[^>]*>\s*가장자리 고정/)
  })

  it('swaps to apply / reset / cancel while active', async () => {
    const html = await render(TransformPanel, { meshActive: true, meshPinEdges: false })
    expect(html).not.toContain('격자 점 맞추기 시작')
    expect(html).toContain('Ctrl+Z 는 마지막 점 이동 되돌리기')
    expect(html).toContain('초기화')
    expect(html).toMatch(/<input type="checkbox"(?![^>]*checked)[^>]*>\s*가장자리 고정/)
  })
})
