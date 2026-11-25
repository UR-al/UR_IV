import { describe, expect, it } from 'vitest'
import inpaintSource from './InpaintView.vue?raw'
import gallerySource from './GalleryView.vue?raw'
import favoritesSource from './FavoritesView.vue?raw'
import searchSource from './SearchView.vue?raw'
import settingsSource from './SettingsView.vue?raw'
import creatorSource from './CreatorStudioView.vue?raw'
import batchSource from './BatchView.vue?raw'
import editorSource from './EditorView.vue?raw'
import eventGenSource from './EventGenView.vue?raw'
import i2iSource from './I2IView.vue?raw'
import xyzSource from './XYZPlotView.vue?raw'
import chatSource from './ChatView.vue?raw'
import compareSliderSource from '../components/CompareSlider.vue?raw'
import editorCanvasSource from '../components/editor/EditorCanvas.vue?raw'
import promptPanelSource from '../components/PromptPanel.vue?raw'

function splitSfc(src: string) {
  const scriptStart = src.indexOf('<script')
  const scriptEnd = src.indexOf('</script>')
  const styleStart = src.indexOf('<style')
  const styleEnd = src.indexOf('</style>')
  return {
    template: src.slice(0, scriptStart),
    script: src.slice(scriptStart, scriptEnd),
    style: src.slice(styleStart, styleEnd),
  }
}

describe('B-25 / B-30 / U-06 / U-17 / U-19: InpaintView fixes', () => {
  const sfc = splitSfc(inpaintSource)

  it('guards onInpaintKeyDown against inactive keep-alive and open modals (B-25)', () => {
    expect(sfc.script).toContain("import { appModalStack } from '../utils/modalStack'")
    expect(sfc.script).toContain('if (!inpaintKeyActive || appModalStack.isAnyOpen()) return')
    expect(sfc.script).toContain('onActivated(bindInpaintKey)')
    expect(sfc.script).toContain('onDeactivated(unbindInpaintKey)')
    expect(sfc.script).toContain('onUnmounted(unbindInpaintKey)')
  })

  it('accepts Windows backslash paths on drop (B-30)', () => {
    expect(sfc.script).toContain("if (p && (p.includes('/') || p.includes('\\\\'))) loadFromPath(p)")
  })

  it('unsubscribes backend events on unmount (B-28)', () => {
    expect(sfc.script).toContain("unsubs.push(onBackendEvent('inpaintImageLoaded'")
    expect(sfc.script).toContain('unsubs.splice(0).forEach')
  })

  it('uses theme tokens on overlay info chip and min-width:0 on canvas-area (U-06, U-17, B-7)', () => {
    expect(sfc.style).not.toContain('color: #585858')
    expect(sfc.style).not.toContain('rgba(255,255,255,0.85)')
    expect(sfc.style).toMatch(/\.cv-info\s*\{[^}]*color:\s*var\(--text-secondary\)/)
    expect(sfc.style).toMatch(/\.cv-info\s*\{[^}]*background:\s*var\(--bg-secondary\)/)
    expect(sfc.style).toMatch(/\.canvas-area\s*\{[^}]*min-width:\s*0/)
  })

  it('unifies Korean spacing in prompts/messages (U-19)', () => {
    expect(sfc.script).not.toContain('직접 올려주세요.')
    expect(sfc.script).toContain('직접 올려 주세요.')
  })
})

describe('B-27 / B-32 / P-06 / P-07 / U-10 / U-13 / U-14 / U-15 / U-19: GalleryView & FavoritesView fixes', () => {
  const gallery = splitSfc(gallerySource)
  const favorites = splitSfc(favoritesSource)

  it('drops stale getImageExif callbacks via generation guard in GalleryView and FavoritesView (B-27)', () => {
    expect(gallery.script).toMatch(/exifViewGen\+\+\s*\n\s*const gen = exifViewGen/)
    expect(gallery.script).toContain('if (gen !== exifViewGen) return')
    expect(favorites.script).toContain('const gen = ++viewerGen')
    expect(favorites.script).toContain('if (gen !== viewerGen) return')
    expect(favorites.script).toMatch(/function closeViewer\(\)\s*\{\s*viewerGen\+\+/)
  })

  it('uses versionedMediaUrl instead of cache-busting Date.now() in GalleryView largeView (B-32)', () => {
    expect(gallery.template).not.toContain('mediaUrl(largeView.path, true)')
    expect(gallery.template).toContain(':src="versionedMediaUrl(largeView.path)"')
  })

  it('resets grid paging on deactivation and prevents duplicate initial keep-alive load (P-06, P-07)', () => {
    expect(gallery.script).toMatch(/onDeactivated\(\(\) => \{\s*paging\.reset\(\)/)
    expect(gallery.script).toContain('if (initialFolderLoaded) loadImages()')
    expect(favorites.script).toMatch(/onDeactivated\(\(\) => \{\s*paging\.reset\(\)/)
    expect(favorites.script).toContain('if (firstActivate)')
  })

  it('adds empty search state, Favorites loading state, modal layers, IME guard, and Korean labels (U-10, U-13, U-14, U-19)', () => {
    expect(gallery.template).toContain('exifFiltered && filteredImages.length === 0')
    expect(gallery.template).toContain('검색 결과가 없습니다')
    expect(gallery.template).toContain('불러오는 중...')
    expect(gallery.template).not.toContain('Loading...')
    expect(gallery.template).not.toContain('title="Refresh"')
    expect(gallery.template).toContain('role="dialog" aria-modal="true"')
    expect(gallery.script).toContain('useModalLayer({ isOpen: () => !!largeView.value, close: closeLargeView })')
    expect(gallery.script).toContain('isImeComposing(e)')

    expect(favorites.template).toContain('v-if="isLoading"')
    expect(favorites.template).toContain('exifFiltered && filteredImages.length === 0')
    expect(favorites.template).not.toContain('title="Refresh"')
    expect(favorites.template).toContain('role="dialog" aria-modal="true"')
    expect(favorites.script).toContain('useModalLayer({ isOpen: () => !!viewerData.value, close: closeViewer })')
    expect(favorites.script).toContain('isImeComposing(e)')
  })

  it('uses z-modal token, theme tokens on .vi-copy-float, and min-width:0 on .viewer-img in FavoritesView (U-01, U-04, U-17)', () => {
    expect(favorites.style).toContain('z-index: var(--z-modal, 2000)')
    expect(favorites.style).not.toContain('rgba(20,20,20,0.5)')
    expect(favorites.style).not.toContain('rgba(45,45,45,0.9)')
    expect(favorites.style).toContain('background: var(--bg-button)')
    expect(favorites.style).toContain('background: var(--bg-button-hover)')
    expect(favorites.style).toMatch(/\.viewer-img\s*\{[^}]*min-width:\s*0;\s*min-height:\s*0/)
  })
})

describe('SearchView fixes (condRules import, B-28, U-01, U-07, U-11, U-13, U-14, U-15, U-17, U-18, M-02, A-5)', () => {
  const search = splitSfc(searchSource)

  it('uses extensionless condRules import for TS migration compatibility', () => {
    expect(search.script).toContain("from '../composables/condRules'")
    expect(search.script).not.toContain("from '../composables/condRules.js'")
  })

  it('preserves the original 16-color tag/legend palette and design decision comment (A-5)', () => {
    expect(search.style).toContain('아래 16개 태그 분류색은 아직 토큰화하지 않았다')
    expect(search.style).toContain('.tag.count { color: #60a5fa;')
    expect(search.style).toContain('.legend-item.count { color: #60a5fa;')
  })

  it('fixes Next icon, 작품 copy, IME guard, modal layer, z-index, layout, error sanitization, unsubs, and export payload', () => {
    expect(search.template).not.toContain('<Icon name="play" />')
    expect(search.template).toContain('<Icon name="chevron-right" />')
    expect(search.template).not.toContain('<span class="ml">프로젝트</span>')
    expect(search.template).toContain('<span class="ml">작품</span>')
    expect(search.template).toContain('role="dialog" aria-modal="true"')
    expect(search.script).toContain('useModalLayer({ isOpen: () => showFilterManager.value')
    expect(search.script).toContain('isImeComposing(e)')
    expect(search.script).toContain("unsubs.push(onBackendEvent('searchResultsReady'")
    expect(search.script).not.toContain('검색 결과 JSON 파싱 실패: ${')
    expect(search.script).toContain("requestAction('export_search_results', { data: filteredResults.value })")
    expect(search.style).toContain('z-index: var(--z-modal, 2000)')
    expect(search.style).toContain('padding: 6px var(--notif-gutter) 6px 12px')
    expect(search.style).toContain('width: min(560px, calc(100vw - 32px))')
  })
})

describe('SettingsView fixes (P-02, U-06, U-11, U-13, U-16, U-18)', () => {
  const settings = splitSfc(settingsSource)

  it('debounces MutationObserver without characterData and pauses/resumes on keep-alive lifecycle (P-02, B-14)', () => {
    expect(settings.script).not.toContain('characterData: true')
    expect(settings.script).toContain('scheduleIndexSettingLabels')
    expect(settings.script).toContain('connectSettingsObserver()')
    expect(settings.script).toContain('disconnectSettingsObserver()')
    expect(settings.script).toMatch(/onActivated\(\(\) => \{\s*indexSettingLabels\(\)\s*connectSettingsObserver\(\)/)
  })

  it('fixes nav overflow, body min-width, search-clear alignment, tokens, busy states, IME guard, and header copy', () => {
    expect(settings.template).toContain('<div class="nav-header">설정</div>')
    expect(settings.template).not.toContain('<div class="nav-header">환경설정</div>')
    expect(settings.template).toContain(':disabled="ollamaTesting || ollamaLoadingModels"')
    expect(settings.script).toContain('isImeComposing(e)')
    expect(settings.style).toMatch(/\.settings-nav\s*\{[^}]*flex-shrink:\s*0;[^}]*min-height:\s*0;[^}]*overflow-y:\s*auto/)
    expect(settings.style).toMatch(/\.settings-body\s*\{[^}]*min-width:\s*0/)
    expect(settings.style).toMatch(/\.settings-search-wrap\s*\{[^}]*padding:\s*0 12px;\s*margin-bottom:\s*12px/)
    expect(settings.style).toContain('.glass-card { background: var(--bg-card);')
    expect(settings.style).toContain('.s-row:hover { background: var(--bg-button-hover); }')
  })
})

describe('CreatorStudio, Batch, Editor, EventGen, I2I, XYZPlot, Chat view fixes', () => {
  const creator = splitSfc(creatorSource)
  const batch = splitSfc(batchSource)
  const editor = splitSfc(editorSource)
  const eventGen = splitSfc(eventGenSource)
  const i2i = splitSfc(i2iSource)
  const xyz = splitSfc(xyzSource)
  const chat = splitSfc(chatSource)

  it('CreatorStudioView debounces recovery, uses shallow panel ID watch, disables export while busy, and localizes copy (P-05, U-11, U-19)', () => {
    expect(creator.script).toContain('scheduleComicRecovery(true)')
    expect(creator.script).toContain("window.addEventListener('pagehide', flushComicRecovery)")
    expect(creator.script).toContain('comicDoc.value.panels.map((p) => p.id)')
    expect(creator.template).toContain(':disabled="comicExporting"')
    expect(creator.template).toContain('title="실행 취소"')
    expect(creator.script).not.toContain('Comic export failed:')
    expect(creator.script).toContain("toast('error', '만화 내보내기 실패')")
  })

  it('BatchView unsubscribes all backend events, disables caption model refresh while busy, and fixes spacing/layout (B-28, U-11, U-17, U-19)', () => {
    expect(batch.script).not.toMatch(/^\s*onBackendEvent\(/m)
    expect(batch.template).toContain(':disabled="captionModelsLoading"')
    expect(batch.script).not.toContain('캡션이 비어있습니다')
    expect(batch.script).toContain('캡션이 비어 있습니다')
    expect(batch.style).toMatch(/\.ad-compare\s*\{[^}]*min-width:\s*0/)
  })

  it('EditorView unsubscribes backend events, sanitizes clipboard error, and guards shortcuts when modal is open (B-28, U-11, U-14)', () => {
    expect(editor.script).not.toMatch(/^\s*onBackendEvent\(/m)
    expect(editor.script).toContain('if (!_editorActive || appModalStack.isAnyOpen()) return')
    expect(editor.script).not.toContain('클립보드 접근 실패: ${')
  })

  it('EventGenView, I2IView, XYZPlotView, and ChatView apply remaining fixes (B-28, U-06, U-13, U-17, U-18, U-19)', () => {
    expect(eventGen.script).not.toMatch(/^\s*onBackendEvent\(/m)
    expect(eventGen.script).toContain('isImeComposing(e)')
    expect(eventGen.template).toContain('대기열에 추가')
    expect(eventGen.template).not.toContain('큐에 추가')

    expect(i2i.script).not.toMatch(/^\s*onBackendEvent\(/m)
    expect(i2i.template).toContain('비워 두면 T2I 프롬프트를 사용합니다...')
    expect(i2i.style).toMatch(/\.canvas-area\s*\{[^}]*min-width:\s*0/)

    expect(xyz.style).toContain('color: var(--state-alert-fg);')
    expect(xyz.style).not.toContain('--error')
    expect(xyz.script).toContain('대기열 등록')
    expect(xyz.script).not.toContain('큐 등록')

    expect(chat.template).toContain('@keydown.enter="onRenameEnter"')
    expect(chat.style).toContain('border-radius: var(--radius-card);')
    expect(chat.style).not.toContain('--radius-lg')
  })
})

describe('2nd-pass regressions (A-4, A-6, A-7, A-9, B-7, B-13)', () => {
  it('uses .self modifier on role=button containers that contain inner buttons (A-4)', () => {
    expect(batchSource).toContain('@keydown.enter.self.prevent="previewAdFile(i)"')
    expect(batchSource).toContain('@keydown.space.self.prevent="previewAdFile(i)"')
    expect(batchSource).toContain('@keydown.enter.self.prevent="previewSam3File(i)"')
    expect(batchSource).toContain('@keydown.space.self.prevent="previewSam3File(i)"')
    expect(eventGenSource).toContain('@keydown.enter.self.prevent="toggleStep(i)"')
    expect(eventGenSource).toContain('@keydown.space.self.prevent="toggleStep(i)"')
    expect(gallerySource).toContain('@keydown.enter.self.prevent="viewImage(img)"')
    expect(gallerySource).toContain('@keydown.space.self.prevent="viewImage(img)"')
    expect(favoritesSource).toContain('@keydown.enter.self.prevent="viewImage(img)"')
    expect(favoritesSource).toContain('@keydown.space.self.prevent="viewImage(img)"')
    expect(searchSource).toContain('@keydown.enter.self.prevent="onListRowClick(r)"')
    expect(searchSource).toContain('@keydown.space.self.prevent="onListRowClick(r)"')
  })

  it('orders dock drawers above notif bell and below modals, and keeps input:focus specificity low (A-6, A-9)', async () => {
    const fs = await import('node:fs' as string)
    const styleCssSource: string = fs.readFileSync(new URL('../style.css', import.meta.url), 'utf8')
    const zVal = (name: string) => Number(styleCssSource.match(new RegExp(`${name}:\\s*(\\d+)`))?.[1])
    expect(zVal('--z-dock-panel')).toBeGreaterThan(zVal('--z-notif'))
    expect(zVal('--z-dock-panel')).toBeLessThan(zVal('--z-modal'))
    expect(styleCssSource).toContain('input:where(:not([type="checkbox"]):not([type="radio"])):focus')
  })

  it('stops Escape propagation during PromptPanel Exclude Manager inline rule edit (A-7)', () => {
    expect(promptPanelSource).toContain('@keydown.escape.prevent.stop="editingExRule = -1"')
  })

  it('uses theme tokens on CompareSlider .compare-info and EditorCanvas .canvas-info (B-7)', () => {
    const compare = splitSfc(compareSliderSource)
    const canvas = splitSfc(editorCanvasSource)
    expect(compare.style).toMatch(/\.compare-info\s*\{[^}]*color:\s*var\(--text-secondary\)/)
    expect(compare.style).toMatch(/\.compare-info\s*\{[^}]*background:\s*var\(--bg-secondary\)/)
    expect(canvas.style).toMatch(/\.canvas-info\s*\{[^}]*color:\s*var\(--text-secondary\)/)
    expect(canvas.style).toMatch(/\.canvas-info\s*\{[^}]*background:\s*var\(--bg-secondary\)/)
  })

  it('blurs contenteditable EXIF editor on Escape instead of closing largeView (B-13)', () => {
    const gallery = splitSfc(gallerySource)
    expect(gallery.template).toContain('@keydown.escape="onExifEditEscape($event)"')
    expect(gallery.script).toMatch(/function onExifEditEscape\(e: KeyboardEvent\)[\s\S]*e\.preventDefault\(\)[\s\S]*e\.stopPropagation\(\)[\s\S]*el\.blur\(\)/)
  })
})
