<template>
  <div class="gallery-workspace">
    <!-- Top Filter & Action Bar -->
    <header class="gallery-toolbar">
      <div class="folder-info" v-host-dialog="'gallery_open_folder'" @click="openFolder">
        <span class="icon"><Icon name="folder" /></span>
        <span class="path">{{ currentFolder || '출력 폴더를 선택하세요' }}</span>
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
        <button class="icon-btn" @click="loadImages()" title="Refresh"><Icon name="refresh" /></button>
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
            <button class="tiny-btn" @click.stop="quickAction('add_favorite', img)"><Icon name="star" /></button>
            <button v-if="isImage(img)" class="tiny-btn" @click.stop="copyImageAndNotify(img)"><Icon name="clipboard" /></button>
          </div>
        </div>
      </div>
      <div class="load-more-info" v-if="visibleCount < (exifFiltered ? filteredImages.length : images.length)">
        {{ visibleCount }} / {{ exifFiltered ? filteredImages.length : images.length }} — 스크롤하여 더 보기
      </div>
      
      <div v-if="isLoading" class="empty-placeholder">
        <div class="spinner"></div>
        <p>Loading...</p>
      </div>
      <div v-else-if="images.length === 0" class="empty-placeholder">
        <div class="icon"><Icon name="video" /></div>
        <h2>갤러리가 비어 있습니다</h2>
        <p>이 폴더에는 볼 수 있는 이미지가 없습니다</p>
      </div>
    </section>

    <!-- 이미지 확대 뷰 (풀스크린 오버레이) -->
    <transition name="fade">
      <div v-if="largeView" class="large-view-overlay" @mousedown.self="closeLargeView">
        <div class="large-view-panel">
          <div class="large-view-header">
            <span class="large-filename">{{ largeView.filename }}</span>
            <div class="large-actions">
              <button class="lv-btn" @click="editFilename"><Icon name="pencil" /> 이름 변경</button>
              <button v-if="canSaveExif" class="lv-btn save" :disabled="exifSaving" @click="saveExif"><Icon name="save" /> EXIF 저장</button>
              <button v-if="isImage(largeView.path)" class="lv-btn" @click="action('send_to_i2i', { path: largeView.path })">I2I</button>
              <button v-if="isImage(largeView.path)" class="lv-btn" @click="action('send_to_inpaint', { path: largeView.path })">인페인트</button>
              <button v-if="isImage(largeView.path)" class="lv-btn" @click="action('send_to_editor', { path: largeView.path })">에디터</button>
              <button class="lv-btn" @click="quickAction('add_favorite', largeView.path)"><Icon name="star" /> 즐겨찾기</button>
              <button v-if="isImage(largeView.path)" class="lv-btn accent" :disabled="largeView.can_apply === false" @click="sendExifToT2I">프롬프트 사용</button>
              <button class="lv-close" @click="closeLargeView"><Icon name="close" /></button>
            </div>
          </div>
          <div class="large-view-body">
            <div class="large-img-area">
              <video v-if="isVideo(largeView.path)" :src="mediaUrl(largeView.path)" controls autoplay playsinline />
              <audio v-else-if="isAudio(largeView.path)" :src="mediaUrl(largeView.path)" controls autoplay />
              <img v-else :src="mediaUrl(largeView.path, true)" />
            </div>
            <div class="large-exif">
              <div class="meta-row"><span>종류</span><p>{{ largeView.mediaType || mediaLabel(largeView.path) }}</p></div>
              <div class="meta-row"><span>크기</span><p>{{ largeView.size }}</p></div>
              <div class="meta-row path-row"><span>경로</span><p>{{ largeView.path }}</p></div>
              <div v-if="largeView.prompt" class="meta-block">
                <label>프롬프트</label>
                <div class="code-box" :class="{ editable: largeView.source !== 'comfyui' }" :contenteditable="largeView.source !== 'comfyui' && !exifSaving" @blur="onExifEdit($event, 'prompt')">{{ largeView.prompt }}</div>
              </div>
              <div v-if="largeView.negative" class="meta-block mt-8">
                <label class="danger">네거티브</label>
                <div class="code-box" :class="{ editable: largeView.source !== 'comfyui' }" :contenteditable="largeView.source !== 'comfyui' && !exifSaving" @blur="onExifEdit($event, 'negative')">{{ largeView.negative }}</div>
              </div>
              <div v-if="largeView.raw && !largeView.prompt && !largeView.raw_prompt && !largeView.raw_workflow" class="meta-block">
                <label>원본</label>
                <div class="code-box">{{ largeView.raw }}</div>
              </div>
              <div v-if="largeViewParams" class="meta-block mt-8">
                <label>파라미터</label>
                <div class="code-box params">{{ largeViewParams }}</div>
              </div>
              <ComfyMetadataDetails :data="largeView" />
            </div>
          </div>
        </div>
      </div>
    </transition>

    <!-- Slide-out EXIF Panel (간단 사이드바 — 하위호환) -->
    <transition name="slide">
      <aside v-if="exifData && !largeView && showMetadata" class="exif-sidebar">
        <div class="exif-close" @click="exifData = null"><Icon name="arrow-right" /></div>
        <div class="exif-content">
          <div class="exif-preview" @click="largeView = exifData">
            <video v-if="isVideo(exifData.path)" :src="mediaUrl(exifData.path)" muted preload="metadata" playsinline />
            <div v-else-if="isAudio(exifData.path)" class="sidebar-audio-preview">
              <span><Icon name="music" /></span>
              <audio :src="mediaUrl(exifData.path)" controls preload="metadata" @click.stop />
            </div>
            <img v-else :src="versionedMediaUrl(exifData.path)" />
            <div class="click-hint">클릭하여 확대</div>
          </div>
          <div class="exif-meta">
            <h3>메타데이터</h3>
            <div class="meta-row"><span>파일</span><p>{{ exifData.filename }}</p></div>
            <div class="meta-row"><span>종류</span><p>{{ exifData.mediaType || mediaLabel(exifData.path) }}</p></div>
            <div class="meta-row"><span>크기</span><p>{{ exifData.size }}</p></div>

            <div v-if="exifData.prompt" class="meta-block">
              <label>프롬프트</label>
              <div class="code-box">{{ exifData.prompt }}</div>
            </div>
            <div v-if="exifData.negative" class="meta-block mt-12">
              <label class="danger">네거티브</label>
              <div class="code-box">{{ exifData.negative }}</div>
            </div>
            <div v-if="exifData.params" class="meta-block mt-12">
              <label>파라미터</label>
              <div class="params-grid">
                <div class="param-line" v-if="exifData.params.generation"><span class="pl">생성</span><span>{{ exifData.params.generation }}</span></div>
                <div class="param-line" v-if="exifData.params.core"><span class="pl">기본</span><span>{{ exifData.params.core }}</span></div>
                <div class="param-line" v-if="exifData.params.model"><span class="pl">모델</span><span>{{ exifData.params.model }}</span></div>
                <div class="param-line" v-if="exifData.params.hires"><span class="pl">고해상도</span><span>{{ exifData.params.hires }}</span></div>
                <div class="param-line" v-if="exifData.params.extensions"><span class="pl">확장</span><span>{{ exifData.params.extensions }}</span></div>
                <div class="param-line" v-if="exifData.params.other"><span class="pl">기타</span><span>{{ exifData.params.other }}</span></div>
              </div>
            </div>
            <div v-else-if="sidebarParams" class="meta-block mt-12">
              <label>파라미터</label>
              <div class="code-box params">{{ sidebarParams }}</div>
            </div>
            <ComfyMetadataDetails :data="exifData" />
          </div>
          <div v-if="isImage(exifData.path)" class="exif-footer">
            <button class="main-apply-btn" :disabled="exifData.can_apply === false" @click="sendExifToT2I">T2I에서 사용</button>
            <div class="grid-2 mt-8">
              <button class="mini-action" @click="action('send_to_i2i', { path: exifData.path })">I2I</button>
              <button class="mini-action" @click="action('send_to_inpaint', { path: exifData.path })">인페인트</button>
            </div>
          </div>
        </div>
      </aside>
    </transition>

    <!-- Context Menu -->
    <transition name="pop">
      <div v-if="ctxMenu.show" ref="ctxMenuEl" class="modern-ctx-menu" :style="ctxMenuStyle">
        <div class="ctx-item" @click="ctx('add_favorite')"><Icon name="star" /> 즐겨찾기 추가</div>
        <div class="ctx-item" @click="ctx('gallery_load_exif')"><Icon name="clipboard" /> 정보 보기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctx('send_to_i2i')"><Icon name="image" /> I2I로 보내기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctx('send_to_inpaint')"><Icon name="palette" /> 인페인트로 보내기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctx('send_to_editor')"><Icon name="pencil" /> 에디터로 보내기</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="sendToCompare('before')"><Icon name="search" /> 비교 (이전)</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="sendToCompare('after')"><Icon name="search" /> 비교 (이후)</div>
        <div v-if="isImage(ctxMenu.path)" class="ctx-item" @click="ctxAdetailer"><Icon name="target" /> ADetailer</div>
        <div class="ctx-separator"></div>
        <div class="ctx-item delete" @click="ctx('delete_image')"><Icon name="trash" /> 휴지통으로 이동</div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, onActivated, onMounted, onUnmounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
// 이미지 복사 — 데스크톱은 호스트 Qt 클립보드, 웹은 이 브라우저 클립보드(호스트 PC 클립보드 금지)
import { copyImageAndNotify } from '../utils/clipboardImageCopy'
import { vHostDialog } from '../utils/hostDialogs'
import { isSameImagePath, parseImageDeleteResult, withoutImagePath } from '../utils/imageDeleteResult'
import { mediaUrl, thumbnailUrl, withUrlVersion } from '../utils/media.js'
import { bumpMediaVersion, mediaVersion, recordListedMediaVersions } from '../utils/mediaVersions'
import { fallbackToOriginal } from '../utils/thumbFallback'
import {
  filenameOf, isAnimated, isAudio, isImage, isPng, isVideo, mediaKind, mediaLabel,
  replaceMediaPath, sortMediaPaths, type MediaSortKey,
} from '../utils/mediaKind'
import { createSearchTextFetcher } from '../utils/imageSearchTexts'
import { resolveExifSaveResponse, type ExifSaveRequest } from '../utils/exifSaveResponse'
import { useExifSearch } from '../composables/useExifSearch'
import { useGridPaging } from '../composables/useGridPaging'
import { useContextMenu } from '../composables/useContextMenu'
import { galleryShowMetadata } from '../composables/uiPrefs'
import type { ActionName, ActionPayload } from '../types/bridge'
import ComfyMetadataDetails from '../components/ComfyMetadataDetails.vue'

interface ExifParams {
  generation?: string
  core?: string
  model?: string
  hires?: string
  extensions?: string
  other?: string
  [k: string]: any
}
interface ExifData {
  source?: string
  path: string
  filename: string
  mediaType?: string
  size?: string
  prompt?: string
  negative?: string
  raw?: string
  params?: ExifParams | null
  params_line?: string
  [k: string]: any
}

const images = ref<string[]>([])
const currentFolder = ref('')
const isLoading = ref(false)
const toast = (type: 'success' | 'error' | 'info' | 'warning', msg: string) => requestAction('show_toast', { type, msg })

// 썸네일 크기 — localStorage 영속, 100~380px
const thumbSize = ref(parseInt(window.localStorage.getItem('gallery_thumb_size') || '200'))
const thumbPixelRatio = Math.min(2, Math.max(1, window.devicePixelRatio || 1))
watch(thumbSize, (v) => {
  window.localStorage.setItem('gallery_thumb_size', String(v))
})
/**
 * 카드 — 정지 이미지는 썸네일(Qt aithumb: / 웹 /thumbnail), 애니메이션은 원본(움직임 유지).
 * 둘 다 내용 버전(목록의 원본 서명·앱이 덮어쓴 표시, utils/mediaVersions)을 붙인다 — 덮어쓴 파일의
 * 카드가 같은 URL 의 옛 그림으로 남지 않게.
 */
const versionedMediaUrl = (path: string) => withUrlVersion(mediaUrl(path), mediaVersion(path))
const cardImageUrl = (path: string) => isAnimated(path)
  ? versionedMediaUrl(path)
  : thumbnailUrl(path, thumbSize.value * thumbPixelRatio, mediaVersion(path))
const onCardImageError = (e: Event, path: string) => { fallbackToOriginal(e.target, versionedMediaUrl(path)) }

// 정렬 — 원본(백엔드 날짜순)을 바꾸지 않고 파생한다. 새로고침·탭 재진입·폴더 변경 뒤에도 칩과 순서가 맞는다.
const sortBy = ref<MediaSortKey>('date')
const sortOptions: { label: string; val: MediaSortKey }[] = [{ label: '날짜', val: 'date' }, { label: '이름', val: 'name' }]

// EXIF 검색 (composables/useExifSearch — Favorites 와 공용)
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

// 카드 그리드 페이징 (composables/useGridPaging — Favorites 와 공용)
const galleryContentRef = ref<HTMLElement | null>(null)
const paging = useGridPaging({
  container: galleryContentRef,
  total: () => displaySource.value.length,
  cell: () => thumbSize.value,
  sources: [images, filteredImages, exifFiltered, thumbSize],
})
const { visibleCount, fillViewport, onScroll: onGalleryScroll } = paging
const displayImages = computed(() => displaySource.value.slice(0, visibleCount.value))

const exifData = ref<ExifData | null>(null)
const largeView = ref<ExifData | null>(null)
/** 확대 뷰에서 프롬프트/네거티브를 고쳤는가 — 안 고쳤으면 저장하지 않는다 */
const exifDirty = ref(false)
const exifSaving = ref(false)
// 확대 뷰 세대 — 이미지를 열거나 닫으면 오른다. 늦게 온 EXIF 저장 응답이 다른 이미지(또는 다시 연 뷰)의
// 편집 표시·텍스트를 덮지 않게 한다(utils/exifSaveResponse).
let exifViewGen = 0
// Settings 와 같은 모듈 전역 ref(composables/uiPrefs) — 바꾸면 곧바로 반영된다. 예전엔 keep-alive 로
// 첫 방문 뒤 멈추지 않는 500ms setInterval 로 localStorage 를 폴링했다(감사 #145).
const showMetadata = galleryShowMetadata

// 파라미터 표시 — core 가 만든 params_line(WebUI 는 원문 꼬리 그대로, 따옴표 보존)
const largeViewParams = computed(() => largeView.value?.params_line || '')
const sidebarParams = computed(() => exifData.value?.params_line || '')
/** 'EXIF 저장' — PNG 의 A1111 parameters 만. ComfyUI 그래프는 읽기 전용. */
const canSaveExif = computed(() => !!largeView.value && isPng(largeView.value.path) && largeView.value.source === 'webui')

/** 목록에 있는 표기 그대로의 경로(검색 캐시 키) — 확대 뷰 경로와 '/'·대소문자가 달라도 찾는다 */
const listPathOf = (path: string) => images.value.find(item => isSameImagePath(item, path)) || path

async function editFilename() {
  const view = largeView.value
  if (!view) return
  const newName = window.prompt('파일 이름 변경:', view.filename)
  if (!newName || newName === view.filename) return
  const backend: any = await getBackend()
  if (!backend.renameFile) return
  const oldPath = view.path
  backend.renameFile(oldPath, newName, (json: string) => {
    let r: any = null
    try { r = JSON.parse(json) } catch { r = null }
    if (!r?.ok || !r.new_path) { toast('error', r?.error || '이름 변경 실패'); return }
    applyRename(oldPath, String(r.new_path))
    toast('success', `이름 변경: ${filenameOf(String(r.new_path))}`)
  })
}

/**
 * 백엔드가 돌려준 실제 새 경로(정리·확장자 보정 후)로 목록·검색 캐시·확대 뷰를 옮긴다.
 * 예전엔 입력한 원문만 제목에 쓰고 옛 경로를 그대로 들고 있어서, 이어서 누른 EXIF 저장과
 * I2I/인페인트/에디터 보내기가 '파일 없음'으로 실패했다.
 */
function applyRename(oldPath: string, newPath: string) {
  const listed = listPathOf(oldPath)
  const next = replaceMediaPath(images.value, oldPath, newPath)
  if (next) images.value = next
  // 새 이름에는 목록 버전이 없다 — 예전에 같은 이름이던(지운) 파일의 캐시된 카드 URL 과 겹치지 않게
  bumpMediaVersion(newPath)
  exifSearchState.rename(listed, newPath)
  for (const view of [largeView.value, exifData.value]) {
    if (view && isSameImagePath(view.path, oldPath)) {
      view.path = newPath
      view.filename = filenameOf(newPath)
    }
  }
}

function onExifEdit(e: FocusEvent, field: 'prompt' | 'negative') {
  const view = largeView.value
  if (!view || view.source === 'comfyui') return
  const text = (e.target as HTMLElement).textContent || ''
  if (text !== (view[field] || '')) {
    view[field] = text
    exifDirty.value = true
  }
}

async function saveExif() {
  const view = largeView.value
  if (!view || !canSaveExif.value || exifSaving.value) return
  if (!exifDirty.value) { toast('info', '바뀐 내용이 없습니다'); return }
  const gen = exifViewGen
  const backend: any = await getBackend()
  if (!backend.saveImagePrompt || gen !== exifViewGen) return
  // 보낸 그대로를 기억한다 — 응답이 오기 전에 또 고친 것은 응답이 덮지 않는다
  const sent: ExifSaveRequest = {
    viewGen: gen, path: view.path, prompt: view.prompt || '', negative: view.negative || '',
  }
  exifSaving.value = true
  // 재조립은 백엔드(core.image_metadata)가 한다 — 파라미터 꼬리·Template 줄·eXIf·dpi 보존
  backend.saveImagePrompt(sent.path, sent.prompt, sent.negative, (json: string) => {
    // 다른 이미지를 열었거나 닫았으면 저장 중 표시는 그때 이미 풀었다 — 새 뷰의 저장 표시를 건드리지 않는다
    if (gen === exifViewGen) exifSaving.value = false
    let r: any = null
    try { r = JSON.parse(json) } catch { r = null }
    if (!r?.ok) { toast('error', r?.error || '저장 실패'); return }
    exifSearchState.invalidate(listPathOf(sent.path))
    const out = resolveExifSaveResponse({ sent, currentViewGen: exifViewGen, current: largeView.value, info: r.info })
    if (out.view) {
      largeView.value = out.view
      exifData.value = out.view
    }
    if (out.dirty !== null) exifDirty.value = out.dirty
    toast('success', out.editedSince
      ? 'EXIF 저장 완료 — 저장 뒤에 고친 내용은 아직 저장되지 않았습니다'
      : 'EXIF 저장 완료')
  })
}

async function loadImages() {
  isLoading.value = true
  const folder = currentFolder.value
  const backend: any = await getBackend()
  // 결과는 galleryImagesReady(워커 스레드 스캔). 목 모드(개발 서버)엔 슬롯이 없어 로딩만 푼다.
  if (backend.requestGalleryImages) {
    backend.requestGalleryImages(folder)
  } else {
    isLoading.value = false
  }
}

/** `versions` — 목록과 같은 순서의 원본 서명(ui/vue_bridge._gallery_images_payload). */
function applyGalleryImages(folder: string, list: unknown, versions?: unknown) {
  if (folder !== currentFolder.value || !Array.isArray(list)) return
  recordListedMediaVersions(list, versions)
  images.value = list as string[]
  paging.keep(images.value.length)
  isLoading.value = false
}

function closeLargeView() {
  exifViewGen++
  largeView.value = null
  exifDirty.value = false
  // 응답이 끝내 안 오면(웹 모드 연결 끊김) 다음 뷰의 저장 버튼이 잠긴 채 남았다
  exifSaving.value = false
  // metadata OFF면 사이드바도 닫기
  if (!showMetadata.value) exifData.value = null
}

const openFolder = () => requestAction('gallery_open_folder')
const viewImage = async (path: string) => {
  exifViewGen++
  exifDirty.value = false
  exifSaving.value = false
  const basic: ExifData = {
    path,
    filename: filenameOf(path),
    mediaType: mediaLabel(path),
    size: '—',
  }
  if (!isImage(path)) {
    largeView.value = basic
    exifData.value = basic
    return
  }
  const backend: any = await getBackend()
  if (!backend.getImageExif) {
    largeView.value = basic
    exifData.value = basic
    return
  }
  backend.getImageExif(path, (json: string) => {
    try {
      const d = JSON.parse(json)
      // 경로는 목록의 표기를 유지한다(삭제·이름 변경·검색 캐시가 같은 키를 쓴다)
      const data = d?.error ? basic : { ...basic, ...d, path, filename: basic.filename, mediaType: basic.mediaType }
      largeView.value = data  // 확대 뷰
      exifData.value = data   // 사이드바 데이터 (showMetadata로 표시 여부 제어)
    } catch {
      largeView.value = basic
      exifData.value = basic
    }
  })
}

// 우클릭 메뉴 — 화면 밖 보정(composables/useContextMenu, Favorites·히스토리와 같은 규칙)
const { menu: ctxMenu, menuEl: ctxMenuEl, style: ctxMenuStyle, open: showMenu, hide: hideMenu } =
  useContextMenu({ width: 220, height: 380 })

function ctx(actionName: ActionName | 'gallery_load_exif') {
  const path = ctxMenu.value.path
  if (actionName === 'gallery_load_exif') viewImage(path)
  else requestAction(actionName, { path })
  // 삭제는 여기서 목록을 건드리지 않는다 — 휴지통 이동이 실패해도 목록에서 먼저 사라지면
  // 파일은 남았는데 갤러리에서만 없어진다. imageDeleteResult 가 오면 뺀다.
  hideMenu()
}

/** 백엔드 삭제 결과 반영 — 실제로 파일이 없어졌을 때(removed)만 목록·검색·확대 뷰에서 뺀다. */
function applyImageDeleteResult(raw: unknown) {
  const result = parseImageDeleteResult(raw)
  if (!result || !result.removed) return
  const listed = listPathOf(result.path)
  const next = withoutImagePath(images.value, result.path)
  if (next) images.value = next   // EXIF 필터 결과는 현재 목록과의 교집합이라 함께 빠진다
  exifSearchState.forget(listed)
  if (isSameImagePath(largeView.value?.path, result.path)) largeView.value = null
  if (isSameImagePath(exifData.value?.path, result.path)) exifData.value = null
}
const quickAction = (name: ActionName, path: string) => requestAction(name, { path })
const sendToCompare = (slot: string) => { requestAction('send_to_compare', { path: ctxMenu.value.path, slot }); hideMenu() }
const ctxAdetailer = () => { requestAction('run_adetailer_single', { path: ctxMenu.value.path, settings: { ad_model: 'face_yolov8n.pt', ad_confidence: 0.3, ad_denoise: 0.4 } }); hideMenu() }
/** 확대 뷰(편집한 프롬프트 포함)의 core 파싱 결과를 그대로 보낸다 — 백엔드가 raw 를 다시 쪼개지 않는다 */
const sendExifToT2I = () => {
  const data = largeView.value || exifData.value
  if (data && data.can_apply !== false) requestAction('gallery_send_exif_to_t2i', { path: data.path, metadata: data })
}
const action = <K extends ActionName>(name: K, payload?: ActionPayload<K>) => requestAction(name, payload)

// onBackendEvent disconnect 핸들 — unmount 시 정리
let _galleryFolderUnsub: (() => void) | null = null
let _galleryImagesUnsub: (() => void) | null = null
let _imageDeleteUnsub: (() => void) | null = null

onMounted(async () => {
  _galleryImagesUnsub = onBackendEvent('galleryImagesReady', (json: string) => {
    try {
      const payload = JSON.parse(json)
      applyGalleryImages(payload.folder || '', payload.files, payload.versions)
    } catch {}
  })
  // 삭제 결과는 await 전에 구독한다 — 늦게 붙으면 먼저 온 결과를 놓친다.
  _imageDeleteUnsub = onBackendEvent('imageDeleteResult', applyImageDeleteResult)
  // 폴더가 바뀌면 옛 폴더의 검색 결과를 남기지 않는다
  _galleryFolderUnsub = onBackendEvent('galleryFolderLoaded', (f: string) => {
    currentFolder.value = f
    clearExifSearch()
    paging.reset()
    loadImages()
  })
  // 마지막 폴더 경로 로드 후 이미지 로드
  const bk: any = await getBackend()
  if (bk.getLastGalleryFolder) {
    bk.getLastGalleryFolder((f: string) => {
      if (f) currentFolder.value = f
      loadImages()  // 경로 설정 후 로드
    })
  } else {
    loadImages()
  }
})
// keep-alive 재진입마다 새로 읽는다 — 갤러리 폴더는 생성 이벤트로 무효화할 수 없다(93e2634b6)
onActivated(() => { loadImages(); fillViewport() })
onUnmounted(() => {
  if (_galleryFolderUnsub) _galleryFolderUnsub()
  if (_galleryImagesUnsub) _galleryImagesUnsub()
  if (_imageDeleteUnsub) _imageDeleteUnsub()
  searchTexts.dispose()
})
</script>

<style scoped>

.folder-info { display: flex; align-items: center; gap: 10px; cursor: pointer; opacity: 0.7; transition: var(--transition); }
.folder-info:hover { opacity: 1; }
/* 경로는 있는 그대로 — 대문자로 밀면 실제와 다른 문자열이 된다 */
.folder-info .path { font-size: var(--fs-meta); color: var(--text-muted); max-width: 400px; overflow: hidden; text-overflow: ellipsis; }

/* 카드 안 이미지·영상·오디오·종류 배지는 galleryShared.css (즐겨찾기와 공용) */

.exif-close { position: absolute; top: 20px; left: -20px; width: 40px; height: 40px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transform: rotate(0deg); }

.exif-preview { width: 100%; aspect-ratio: 1; overflow: hidden; }
.exif-preview img, .exif-preview video { width: 100%; height: 100%; object-fit: contain; background: var(--bg-primary); }
.sidebar-audio-preview { width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px; background: var(--bg-primary); color: var(--accent); font-size: 42px; }
.sidebar-audio-preview audio { width: 85%; }

.meta-row p { font-size: 12px; font-weight: var(--fw-bold); color: var(--text-primary); }
.meta-row.path-row { align-items: flex-start; }
.meta-row.path-row p { max-width: 250px; word-break: break-all; text-align: right; font-size: var(--fs-label); }

.meta-block label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); margin-bottom: 8px; }
.code-box { background: var(--bg-input); padding: 12px; border-radius: 8px; font-family: 'Consolas', monospace; font-size: 11px; line-height: 1.6; color: var(--text-secondary); word-break: break-all; }

.params-grid { background: var(--bg-input); border-radius: 8px; padding: 8px 12px; }
.param-line { display: flex; align-items: baseline; gap: 8px; padding: 3px 0; font-size: 11px; color: var(--text-secondary); border-bottom: 1px solid rgba(255,255,255,0.03); font-family: 'Consolas', monospace; }
.param-line:last-child { border-bottom: none; }
.pl { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); letter-spacing: 0; min-width: 45px; flex-shrink: 0; }

.mini-action { flex: 1; height: 36px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }

.ctx-item.delete { color: var(--state-alert-fg); }

/* 채움이라 --state-ok(글자용 -fg 아님). 그 위 글자는 흰색 고정 —
   상태 채움색 자체가 '흰 글자와 4.5:1' 기준으로 잡힌 값이고, --text-primary 는 라이트에서 검정이 된다. */
.lv-btn.save { background: var(--state-ok); color: #FFFFFF; border: none; }

.large-img-area img, .large-img-area video { max-width: 100%; max-height: 100%; object-fit: contain; }
.large-img-area audio { width: min(620px, 90%); }

.code-box.editable { cursor: text; border: 1px solid transparent; }
.code-box.editable:focus { border-color: var(--accent-dim); outline: none; }

.exif-preview { position: relative; cursor: pointer; }

.spinner { width: 32px; height: 32px; border: 3px solid var(--rule); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto 12px; }
@keyframes spin { to { transform: rotate(360deg); } }

</style>
