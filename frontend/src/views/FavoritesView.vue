<template>
  <div class="gallery-workspace">
    <!-- Top Filter & Action Bar -->
    <header class="gallery-toolbar">
      <div class="folder-info no-click">
        <span class="icon"><Icon name="star" /></span>
        <span class="path">즐겨찾기</span>
      </div>

      <div class="spacer"></div>

      <!-- EXIF 검색 -->
      <div class="search-box">
        <input v-model="exifSearch" placeholder="EXIF 검색..." class="search-input"
          @keydown.enter="runExifSearch" />
        <button class="search-go" @click="runExifSearch" :disabled="exifSearching">{{ exifSearching ? '...' : 'GO' }}</button>
        <button class="search-clear" v-if="exifFiltered || exifSearching" :title="exifSearching ? '검색 취소' : '검색 해제'" @click="clearExifSearch"><Icon name="close" /></button>
      </div>

      <div class="control-group">
        <button class="icon-btn" @click="loadFavorites" title="Refresh"><Icon name="refresh" /></button>
        <div class="sep"></div>
        <div class="sort-chips">
          <button v-for="s in sortOptions" :key="s.val"
            class="mini-chip" :class="{ active: sortBy === s.val }"
            @click="sortBy = s.val"
          >{{ s.label }}</button>
        </div>
        <div class="sep"></div>
        <!-- 썸네일 크기 슬라이더 -->
        <div class="thumb-size-ctl" :title="`썸네일 크기: ${thumbSize}px`">
          <span class="thumb-icon-small">▫</span>
          <input type="range" v-model.number="thumbSize" min="100" max="380" step="20" class="thumb-slider" />
          <span class="thumb-icon-large">▪</span>
          <span class="thumb-size-val">{{ thumbSize }}px</span>
        </div>
        <span class="count-badge">{{ exifFiltered ? filteredImages.length + '/' : '' }}{{ images.length }}</span>
      </div>
    </header>

    <!-- Masonry-style Grid -->
    <section class="gallery-content" ref="galleryContentRef" @scroll="onGalleryScroll">
      <div class="masonry-grid" :style="{ 'columns': `auto ${thumbSize}px` }">
        <div v-for="img in displayImages" :key="img" class="gallery-card"
          @click="viewImage(img)"
          @contextmenu.prevent="showMenu($event, img)"
        >
          <video v-if="isVideo(img)" class="gallery-media" :src="mediaUrl(img)"
            muted preload="metadata" playsinline />
          <div v-else-if="isAudio(img)" class="audio-card">
            <span class="audio-icon"><Icon name="music" /></span>
            <span class="audio-name">{{ filenameOf(img) }}</span>
            <audio :src="mediaUrl(img)" controls preload="metadata" @click.stop />
          </div>
          <img v-else :src="cardImageUrl(img)" loading="lazy" decoding="async" @error="onCardImageError($event, img)" />
          <span v-if="mediaKind(img) !== 'image' || isAnimated(img)" class="media-kind-badge">
            {{ mediaLabel(img) }}
          </span>
          <div class="card-hover-actions">
            <button class="tiny-btn" @click.stop="removeFav(img)" title="즐겨찾기 제거"><Icon name="star" /></button>
            <button v-if="isImage(img)" class="tiny-btn" @click.stop="copyImageAndNotify(img)" title="복사"><Icon name="clipboard" /></button>
          </div>
        </div>
      </div>
      <div class="load-more-info" v-if="visibleCount < (exifFiltered ? filteredImages.length : images.length)">
        {{ visibleCount }} / {{ exifFiltered ? filteredImages.length : images.length }} — 스크롤하여 더 보기
      </div>

      <div v-if="images.length === 0" class="empty-placeholder">
        <div class="icon"><Icon name="star" /></div>
        <h2>즐겨찾기가 없습니다</h2>
        <p>아직 즐겨찾기한 이미지가 없습니다</p>
      </div>
    </section>

    <!-- 별도 뷰어 창 (이미지 확대 + EXIF + 전송 버튼) — 이전 Favorites 방식 복원 -->
    <transition name="fade">
      <div v-if="viewerData" class="viewer-overlay" @mousedown.self="viewerData = null">
        <div class="viewer-panel">
          <div class="viewer-header">
            <span>{{ viewerData.filename }}</span>
            <button class="viewer-close" @click="viewerData = null"><Icon name="close" /></button>
          </div>
          <div class="viewer-body">
            <div class="viewer-img">
              <video v-if="isVideo(viewerData.path)" :src="mediaUrl(viewerData.path)" controls autoplay playsinline />
              <audio v-else-if="isAudio(viewerData.path)" :src="mediaUrl(viewerData.path)" controls autoplay />
              <img v-else :src="versionedMediaUrl(viewerData.path)" />
            </div>
            <div class="viewer-info">
              <div class="vi-size">{{ viewerData.mediaType }} · {{ viewerData.size }}</div>
              <div v-if="viewerData.prompt" class="vi-section">
                <div class="vi-head"><label>프롬프트</label></div>
                <div class="vi-pre-wrap">
                  <button class="vi-copy-float" @click="copySection(viewerData.prompt, 'Prompt')" title="Prompt 복사"><Icon name="clipboard" /></button>
                  <pre>{{ viewerData.prompt }}</pre>
                </div>
              </div>
              <div v-if="viewerData.negative" class="vi-section">
                <div class="vi-head"><label class="neg">네거티브</label></div>
                <div class="vi-pre-wrap">
                  <button class="vi-copy-float" @click="copySection(viewerData.negative, 'Negative')" title="Negative 복사"><Icon name="clipboard" /></button>
                  <pre>{{ viewerData.negative }}</pre>
                </div>
              </div>
              <div v-if="viewerData.raw && !viewerData.prompt" class="vi-section">
                <div class="vi-head"><label>원본</label></div>
                <div class="vi-pre-wrap">
                  <button class="vi-copy-float" @click="copySection(viewerData.raw, 'Raw')" title="Raw 복사"><Icon name="clipboard" /></button>
                  <pre>{{ viewerData.raw }}</pre>
                </div>
              </div>
              <div v-if="viewerParams" class="vi-section">
                <div class="vi-head"><label>파라미터</label></div>
                <div class="vi-pre-wrap">
                  <button class="vi-copy-float" @click="copySection(viewerParams, 'Parameters')" title="Parameters 복사"><Icon name="clipboard" /></button>
                  <pre>{{ viewerParams }}</pre>
                </div>
              </div>
              <div v-if="isImage(viewerData.path)" class="vi-actions-section">
                <label class="vi-actions-label">보내기</label>
                <div class="vi-send-grid">
                  <button class="send-card primary" :disabled="viewerData.can_apply === false" @click="sendExifToT2I" title="이미지의 프롬프트를 T2I 탭에 전송">
                    <span class="send-ico"><Icon name="upload" /></span>
                    <span class="send-name">T2I</span>
                  </button>
                  <button class="send-card" @click="action('send_to_i2i', { path: viewerData.path })" title="I2I 탭으로">
                    <span class="send-ico"><Icon name="image" /></span>
                    <span class="send-name">I2I</span>
                  </button>
                  <button class="send-card" @click="action('send_to_inpaint', { path: viewerData.path })" title="Inpaint 탭으로">
                    <span class="send-ico"><Icon name="scissors" /></span>
                    <span class="send-name">Inpaint</span>
                  </button>
                  <button class="send-card" @click="action('send_to_editor', { path: viewerData.path })" title="Editor 탭으로">
                    <span class="send-ico"><Icon name="palette" /></span>
                    <span class="send-name">Editor</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>

    <!-- Context Menu -->
    <transition name="pop">
      <div v-if="ctxMenu.show" ref="ctxMenuEl" class="modern-ctx-menu" :style="ctxMenuStyle">
        <div class="ctx-item" @click="ctx('gallery_load_exif')"><Icon name="clipboard" /> {{ isImage(ctxMenu.path) ? 'EXIF 보기' : '정보 보기' }}</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctx('send_to_i2i')"><Icon name="image" /> I2I로 보내기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctx('send_to_inpaint')"><Icon name="palette" /> 인페인트로 보내기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctx('send_to_editor')"><Icon name="pencil" /> 에디터로 보내기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctxCopyImage"><Icon name="clipboard" /> 복사</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="sendToCompare('before')"><Icon name="search" /> 비교 (이전)</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="sendToCompare('after')"><Icon name="search" /> 비교 (이후)</div>
        <div class="ctx-separator"></div>
        <div class="ctx-item unfav" @click="ctxRemoveFav"><Icon name="star" /> 즐겨찾기 해제</div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, onActivated, onMounted, onUnmounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { mediaUrl, thumbnailUrl, withUrlVersion } from '../utils/media.js'
import { mediaVersion } from '../utils/mediaVersions'
import { copyTextToClipboard } from '../utils/clipboard'
// 이미지 복사 — 데스크톱은 호스트 Qt 클립보드, 웹은 이 브라우저 클립보드(호스트 PC 클립보드 금지)
import { copyImageAndNotify } from '../utils/clipboardImageCopy'
import { isSameImagePath } from '../utils/imageDeleteResult'
import { fallbackToOriginal } from '../utils/thumbFallback'
import {
  filenameOf, isAnimated, isAudio, isImage, isVideo, mediaKind, mediaLabel,
  sortMediaPaths, type MediaSortKey,
} from '../utils/mediaKind'
import { createSearchTextFetcher } from '../utils/imageSearchTexts'
import { useExifSearch } from '../composables/useExifSearch'
import { useGridPaging } from '../composables/useGridPaging'
import { useContextMenu } from '../composables/useContextMenu'
import type { ActionName, ActionPayload } from '../types/bridge'

interface ViewerData {
  filename?: string
  path: string
  mediaType?: string
  size?: string
  prompt?: string
  negative?: string
  raw?: string
  params_line?: string
  can_apply?: boolean
  [k: string]: any
}

const images = ref<string[]>([])

// 썸네일 크기 — localStorage 영속 (gallery와 공유)
const thumbSize = ref(parseInt(window.localStorage.getItem('gallery_thumb_size') || '200'))
const thumbPixelRatio = Math.min(2, Math.max(1, window.devicePixelRatio || 1))
watch(thumbSize, (v) => window.localStorage.setItem('gallery_thumb_size', String(v)))
/**
 * 카드 — 정지 이미지는 공용 썸네일 캐시(Qt aithumb: / 웹 /thumbnail). 폭 버킷이 380px × DPR 2 까지
 * 올라가(768px) 예전 '저화질 썸네일'(고정 384px) 문제 없이 원본 풀디코드를 피한다. 애니메이션은 원본.
 * 둘 다 내용 버전(utils/mediaVersions — 갤러리·히스토리 목록의 원본 서명, 앱이 덮어쓴 표시)을 붙인다.
 * 즐겨찾기 목록은 원본 서명을 싣지 않는다(getFavorites 는 파일을 stat 하지 않는 동기 슬롯이다).
 */
const versionedMediaUrl = (path: string) => withUrlVersion(mediaUrl(path), mediaVersion(path))
const cardImageUrl = (path: string) => isAnimated(path)
  ? versionedMediaUrl(path)
  : thumbnailUrl(path, thumbSize.value * thumbPixelRatio, mediaVersion(path))
const onCardImageError = (e: Event, path: string) => { fallbackToOriginal(e.target, versionedMediaUrl(path)) }

// 정렬 — 원본(추가 순서)을 바꾸지 않고 파생
const sortBy = ref<MediaSortKey>('date')
const sortOptions: { label: string; val: MediaSortKey }[] = [{ label: '날짜', val: 'date' }, { label: '이름', val: 'name' }]

// EXIF 검색 (composables/useExifSearch — Gallery 와 공용)
const searchTexts = createSearchTextFetcher({ getBackend, onBackendEvent })
const exifSearchState = useExifSearch({
  source: () => images.value,
  isImage,
  labelFor: (path) => `${filenameOf(path)} ${mediaLabel(path)}`,
  fetchTexts: searchTexts.fetch,
  onChange: () => paging.reset(),
})
const {
  query: exifSearch, searching: exifSearching, filtered: exifFiltered, results: filteredImages,
  run: runExifSearch, clear: clearExifSearch,
} = exifSearchState

const displaySource = computed(() => sortMediaPaths(exifFiltered.value ? filteredImages.value : images.value, sortBy.value))

// 카드 그리드 페이징 (composables/useGridPaging — Gallery 와 공용)
const galleryContentRef = ref<HTMLElement | null>(null)
const paging = useGridPaging({
  container: galleryContentRef,
  total: () => displaySource.value.length,
  cell: () => thumbSize.value,
  sources: [images, filteredImages, exifFiltered, thumbSize],
})
const { visibleCount, fillViewport, onScroll: onGalleryScroll } = paging
const displayImages = computed(() => displaySource.value.slice(0, visibleCount.value))

// 중앙 모달 뷰어 (이전 Favorites 방식)
const viewerData = ref<ViewerData | null>(null)
// 파라미터 표시 — core 가 만든 params_line(WebUI 는 원문 꼬리 그대로, 따옴표 보존)
const viewerParams = computed(() => viewerData.value?.params_line || '')

async function loadFavorites() {
  const backend: any = await getBackend()
  if (backend.getFavorites) {
    backend.getFavorites((json: string) => {
      try {
        const list = JSON.parse(json)
        images.value = Array.isArray(list) ? list : []
        paging.keep(images.value.length)
      } catch {}
    })
  }
}

const viewImage = async (path: string) => {
  const basic: ViewerData = { path, filename: filenameOf(path), mediaType: mediaLabel(path), size: '—' }
  // 영상·오디오는 Pillow 메타 슬롯으로 보내지 않는다
  if (!isImage(path)) { viewerData.value = basic; return }
  const backend: any = await getBackend()
  if (!backend.getImageExif) { viewerData.value = basic; return }
  backend.getImageExif(path, (json: string) => {
    try {
      const d = JSON.parse(json)
      viewerData.value = d?.error ? basic : { ...basic, ...d, path, filename: basic.filename, mediaType: basic.mediaType }
    } catch {
      viewerData.value = basic
    }
  })
}

// 우클릭 메뉴 — 화면 밖 보정(composables/useContextMenu, Gallery·히스토리와 같은 규칙)
const { menu: ctxMenu, menuEl: ctxMenuEl, style: ctxMenuStyle, open: showMenu, hide: hideMenu } =
  useContextMenu({ width: 220, height: 320 })

function ctx(actionName: ActionName | 'gallery_load_exif') {
  const path = ctxMenu.value.path
  if (actionName === 'gallery_load_exif') viewImage(path)
  else requestAction(actionName, { path })
  hideMenu()
}
function removeFav(path: string) {
  requestAction('remove_favorite', { path })
  images.value = images.value.filter(i => i !== path)   // EXIF 필터 결과도 현재 목록과의 교집합이라 함께 빠진다
  exifSearchState.forget(path)
  if (isSameImagePath(viewerData.value?.path, path)) viewerData.value = null
}
function ctxRemoveFav() { removeFav(ctxMenu.value.path); hideMenu() }
/** 복사는 클릭 처리 안에서 바로 시작한다(웹 브라우저 클립보드는 사용자 동작 안에서만 쓸 수 있다). */
function ctxCopyImage() { const path = ctxMenu.value.path; hideMenu(); void copyImageAndNotify(path) }

const sendToCompare =(slot: string) => { requestAction('send_to_compare', { path: ctxMenu.value.path, slot }); hideMenu() }
/** 뷰어가 이미 읽은 core 파싱 결과를 그대로 보낸다 — 백엔드가 raw 를 다시 쪼개지 않는다 */
const sendExifToT2I = () => {
  const data = viewerData.value
  if (data && isImage(data.path) && data.can_apply !== false) {
    requestAction('gallery_send_exif_to_t2i', { path: data.path, metadata: data })
  }
}
const action = <K extends ActionName>(name: K, payload?: ActionPayload<K>) => requestAction(name, payload)

// Qt 데스크톱·웹 단말 공통 클립보드(실제 성공일 때만 성공 알림)
async function copySection(text: string, label: string) {
  if (!text) return
  const ok = await copyTextToClipboard(text)
  requestAction('show_toast', { type: ok ? 'success' : 'error', msg: ok ? `${label} 복사됨` : '클립보드에 복사하지 못했습니다' })
}

onMounted(() => { loadFavorites() })

// 라우터가 <keep-alive> 로 감싸므로 탭을 다시 열어도 onMounted 는 안 돈다.
// 갤러리에서 즐겨찾기를 더하고 이 탭으로 오면 옛 목록이 그대로 보였다 —
// 목록을 밀어 주는 이벤트가 따로 없어서, 들어올 때마다 다시 읽는다.
// (GalleryView 가 같은 이유로 onActivated 에서 loadImages 를 부른다.)
onActivated(() => { loadFavorites(); fillViewport() })

onUnmounted(() => { searchTexts.dispose() })
</script>

<style scoped>

.folder-info { display: flex; align-items: center; gap: 10px; opacity: 0.85; }
.folder-info.no-click { cursor: default; }
.folder-info .icon { font-size: 15px; }
/* 경로는 있는 그대로 — 대문자로 밀면 실제와 다른 문자열이 된다 */
.folder-info .path { font-size: var(--fs-meta); color: var(--text-muted); }

/* 카드 안 이미지·영상·오디오·종류 배지는 galleryShared.css (갤러리와 공용) */

/* 이전 Favorites 중앙 모달 뷰어 복원 */
.viewer-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.85); z-index: 100; display: flex; align-items: center; justify-content: center; }
/* 떠 있는 모달이라 --bg-secondary 가 아니라 --bg-card — 안쪽 pre(--bg-input)와 면이 겹치지 않아야 한다 */
.viewer-panel { width: 85%; height: 85%; background: var(--bg-card); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; border: 1px solid var(--border); }
.viewer-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--rule); }
.viewer-header span { font-size: 12px; color: var(--text-muted); }
.viewer-close { background: none; border: none; color: var(--state-alert-fg); font-size: 18px; cursor: pointer; }
.viewer-body { flex: 1; display: flex; overflow: hidden; }
.viewer-img { flex: 1; display: flex; align-items: center; justify-content: center; background: var(--bg-primary); padding: 16px; }
.viewer-img img, .viewer-img video { max-width: 100%; max-height: 100%; object-fit: contain; }
.viewer-img audio { width: min(620px, 90%); }
.viewer-info { width: 460px; max-width: 46vw; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 12px; border-left: 1px solid var(--rule); }
.vi-size { color: var(--text-muted); font-size: 12px; }
.vi-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; min-height: 18px; }
.vi-head label { color: var(--accent); font-size: 11px; font-weight: var(--fw-bold); letter-spacing: 0; margin: 0; }
.vi-head label.neg { color: var(--state-alert-fg); }
.vi-pre-wrap { position: relative; }
.vi-copy-float { position: absolute; top: 7px; right: 7px; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; background: rgba(20,20,20,0.5); border: 1px solid rgba(255,255,255,0.12); border-radius: 7px; color: var(--text-primary); font-size: 14px; cursor: pointer; opacity: 0.5; backdrop-filter: blur(3px); transition: opacity .15s, background .15s, border-color .15s; z-index: 2; }
.vi-pre-wrap:hover .vi-copy-float { opacity: 0.85; }
.vi-copy-float:hover { opacity: 1 !important; background: rgba(45,45,45,0.9); border-color: var(--accent); color: var(--accent); }
.vi-section pre { color: var(--text-secondary); font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; background: var(--bg-input); padding: 12px 14px; border-radius: 6px; margin: 0; max-height: 360px; overflow-y: auto; }
.vi-actions-section { margin-top: auto; padding-top: 12px; }
.vi-actions-label { display: block; font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin-bottom: 8px; }
.vi-send-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }
.send-card { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 10px 6px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: all 0.15s; }
.send-card:hover { background: var(--bg-input); border-color: var(--text-muted); transform: translateY(-1px); }
.send-card.primary { background: var(--accent-dim); border-color: rgba(250,204,21,0.4); }
.send-card.primary:hover { background: rgba(250,204,21,0.15); border-color: var(--accent); box-shadow: 0 2px 8px rgba(250,204,21,0.2); }
.send-ico { font-size: 18px; line-height: 1; }
.send-name { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-secondary); letter-spacing: 0; }
.send-card.primary .send-name { color: var(--accent); }
.send-card:disabled { opacity: .45; cursor: not-allowed; transform: none; }

.ctx-item.unfav { color: var(--accent); }

.pop-enter-active { transition: all 0.12s; }
.pop-enter-from { opacity: 0; transform: scale(0.95); }
</style>
