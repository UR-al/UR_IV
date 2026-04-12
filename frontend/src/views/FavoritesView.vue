<template>
  <div class="fav-view">
    <div class="fav-header">
      <h3>Favorites</h3>
      <div class="fav-search">
        <input v-model="exifSearch" placeholder="🔍 EXIF 검색..." class="fav-search-input" @keydown.enter="runExifSearch" />
        <button class="fav-search-btn" @click="runExifSearch" :disabled="exifSearching">{{ exifSearching ? '...' : 'GO' }}</button>
        <button class="fav-search-clr" v-if="exifFiltered" @click="clearExifSearch">✕</button>
      </div>
      <button class="btn" @click="loadFavorites">🔄</button>
      <span class="count">{{ exifFiltered ? displayFavs.length + '/' : '' }}{{ favorites.length }}장</span>
    </div>
    <div class="fav-grid">
      <div v-for="(img, i) in displayFavs" :key="img" class="fav-item"
        @click="openViewer(img)"
        @contextmenu.prevent="showMenu($event, img, i)"
      >
        <img :src="'file:///' + img" loading="lazy" />
      </div>
      <div v-if="favorites.length === 0" class="empty">즐겨찾기가 없습니다</div>
    </div>

    <!-- 우클릭 메뉴 (뷰어로 보기 제거) -->
    <div v-if="ctxMenu.show" class="ctx-menu" :style="ctxMenuStyle">
      <div class="ctx-item" @click="ctx('send_to_i2i')">🖼️ I2I로 보내기</div>
      <div class="ctx-item" @click="ctx('send_to_inpaint')">🎨 Inpaint</div>
      <div class="ctx-item" @click="ctx('send_to_editor')">✏️ Editor</div>
      <div class="ctx-item" @click="ctx('copy_to_clipboard')">📋 복사</div>
      <div class="ctx-item delete" @click="removeFav">⭐ 즐겨찾기 제거</div>
    </div>

    <!-- 별도 뷰어 창 (이미지 확대 + EXIF + 전송 버튼) -->
    <transition name="fade">
      <div v-if="viewerData" class="viewer-overlay" @click.self="viewerData = null">
        <div class="viewer-panel">
          <div class="viewer-header">
            <span>{{ viewerData.filename }}</span>
            <button class="viewer-close" @click="viewerData = null">✕</button>
          </div>
          <div class="viewer-body">
            <div class="viewer-img">
              <img :src="'file:///' + viewerData.path" />
            </div>
            <div class="viewer-info">
              <div class="vi-size">{{ viewerData.size }}</div>
              <div v-if="viewerData.prompt" class="vi-section">
                <label>PROMPT</label>
                <pre>{{ viewerData.prompt }}</pre>
              </div>
              <div v-if="viewerData.negative" class="vi-section">
                <label class="neg">NEGATIVE</label>
                <pre>{{ viewerData.negative }}</pre>
              </div>
              <div v-if="viewerData.raw && !viewerData.prompt" class="vi-section">
                <label>RAW</label>
                <pre>{{ viewerData.raw }}</pre>
              </div>
              <div v-if="viewerData.params" class="vi-section">
                <label>PARAMETERS</label>
                <div class="params-grid">
                  <div class="param-line" v-if="viewerData.params.generation"><span class="pl">GEN</span><span>{{ viewerData.params.generation }}</span></div>
                  <div class="param-line" v-if="viewerData.params.core"><span class="pl">CORE</span><span>{{ viewerData.params.core }}</span></div>
                  <div class="param-line" v-if="viewerData.params.model"><span class="pl">MODEL</span><span>{{ viewerData.params.model }}</span></div>
                  <div class="param-line" v-if="viewerData.params.hires"><span class="pl">HIRES</span><span>{{ viewerData.params.hires }}</span></div>
                  <div class="param-line" v-if="viewerData.params.extensions"><span class="pl">EXT</span><span>{{ viewerData.params.extensions }}</span></div>
                  <div class="param-line" v-if="viewerData.params.other"><span class="pl">ETC</span><span>{{ viewerData.params.other }}</span></div>
                </div>
              </div>
              <div v-else-if="viewerParams" class="vi-section">
                <label>PARAMETERS</label>
                <pre>{{ viewerParams }}</pre>
              </div>
              <div class="vi-actions">
                <button class="vi-btn accent" @click="action('gallery_send_exif_to_t2i', { exif: viewerData.raw, path: viewerData.path })">📤 T2I</button>
                <button class="vi-btn" @click="action('send_to_i2i', { path: viewerData.path })">I2I</button>
                <button class="vi-btn" @click="action('send_to_inpaint', { path: viewerData.path })">INPAINT</button>
                <button class="vi-btn" @click="action('send_to_editor', { path: viewerData.path })">EDITOR</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { getBackend } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'

const favorites = ref([])
const ctxMenu = ref({ show: false, x: 0, y: 0, path: '', index: -1 })
const viewerData = ref(null)

// EXIF 검색
const exifSearch = ref('')
const exifFiltered = ref(false)
const exifSearching = ref(false)
const filteredFavs = ref([])
const exifCache = ref({})

const displayFavs = computed(() => exifFiltered.value ? filteredFavs.value : favorites.value)
const viewerParams = computed(() => {
  if (!viewerData.value?.raw) return ''
  const m = viewerData.value.raw.match(/Steps:.*$/m)
  return m ? m[0] : ''
})

async function runExifSearch() {
  const q = exifSearch.value.trim().toLowerCase()
  if (!q) { clearExifSearch(); return }
  exifSearching.value = true
  const backend = await getBackend()
  for (const img of favorites.value) {
    if (img in exifCache.value) continue
    await new Promise(resolve => {
      if (backend.getImageExif) {
        backend.getImageExif(img, (json) => {
          try { const d = JSON.parse(json); exifCache.value[img] = `${d.prompt||''} ${d.negative||''} ${d.raw||''}`.toLowerCase() }
          catch { exifCache.value[img] = '' }
          resolve()
        })
      } else resolve()
    })
  }
  filteredFavs.value = favorites.value.filter(img => (exifCache.value[img] || '').includes(q))
  exifFiltered.value = true; exifSearching.value = false
}
function clearExifSearch() { exifSearch.value = ''; exifFiltered.value = false; filteredFavs.value = [] }

const ctxMenuStyle = computed(() => {
  const menuW = 200, menuH = 200
  let x = ctxMenu.value.x, y = ctxMenu.value.y
  if (x + menuW > window.innerWidth) x = window.innerWidth - menuW - 10
  if (y + menuH > window.innerHeight) y = window.innerHeight - menuH - 10
  return { top: y + 'px', left: x + 'px' }
})

async function loadFavorites() {
  const backend = await getBackend()
  if (backend.getFavorites) {
    backend.getFavorites((json) => {
      try { favorites.value = JSON.parse(json) } catch {}
    })
  }
}

async function openViewer(path) {
  const backend = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json) => {
      try { viewerData.value = JSON.parse(json) } catch {}
    })
  }
}

function showMenu(e, path, i) {
  let x = e.clientX, y = e.clientY
  if (x + 200 > window.innerWidth) x = window.innerWidth - 210
  if (y + 250 > window.innerHeight) y = window.innerHeight - 260
  ctxMenu.value = { show: true, x, y, path, index: i }
}

function ctx(actionName) {
  requestAction(actionName, { path: ctxMenu.value.path })
  ctxMenu.value.show = false
}

function removeFav() {
  const path = ctxMenu.value.path
  favorites.value = favorites.value.filter(f => f !== path)
  requestAction('remove_favorite', { path })
  ctxMenu.value.show = false
}

function action(name, payload = {}) { requestAction(name, payload) }

function hideMenu() { ctxMenu.value.show = false }
onMounted(() => { document.addEventListener('click', hideMenu); loadFavorites() })
onUnmounted(() => document.removeEventListener('click', hideMenu))
</script>

<style scoped>
.fav-view { width: 100%; height: 100%; display: flex; flex-direction: column; position: relative; }
.fav-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; flex-shrink: 0; }
.fav-header h3 { color: #E8E8E8; font-size: 14px; margin: 0; }
.fav-search { display: flex; align-items: center; gap: 4px; flex: 1; margin: 0 8px; }
.fav-search-input { flex: 1; padding: 4px 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-primary); font-size: 11px; }
.fav-search-input:focus { border-color: var(--accent); }
.fav-search-btn { padding: 4px 10px; background: var(--accent); border: none; border-radius: var(--radius-pill); color: #000; font-size: 9px; font-weight: 800; cursor: pointer; }
.fav-search-btn:disabled { opacity: 0.4; }
.fav-search-clr { background: none; border: none; color: #f87171; cursor: pointer; font-size: 13px; }
.count { color: #585858; font-size: 12px; }
.btn { padding: 5px 12px; background: #181818; border: none; border-radius: 4px; color: #787878; font-size: 11px; cursor: pointer; }
.btn:hover { background: #222; color: #E8E8E8; }
.fav-grid { flex: 1; display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px; padding: 8px; overflow-y: auto; align-content: start; }
.fav-item { height: 150px; border-radius: 4px; overflow: hidden; cursor: pointer; border: 2px solid transparent; transition: 0.15s; }
.fav-item:hover { border-color: #E2B340; }
.fav-item img { width: 100%; height: 100%; object-fit: cover; }
.empty { grid-column: 1 / -1; text-align: center; color: #484848; padding: 60px; }

.ctx-menu { position: fixed; background: #1A1A1A; border-radius: 6px; padding: 4px; z-index: 9999; min-width: 170px; box-shadow: 0 4px 16px rgba(0,0,0,0.7); }
.ctx-item { padding: 7px 14px; font-size: 12px; color: #B0B0B0; cursor: pointer; border-radius: 4px; }
.ctx-item:hover { background: #222; color: #E8E8E8; }
.ctx-item.delete { color: #E2B340; }

/* 뷰어 오버레이 */
.viewer-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.85); z-index: 100; display: flex; align-items: center; justify-content: center; }
.viewer-panel { width: 85%; height: 85%; background: #0D0D0D; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; border: 1px solid #222; }
.viewer-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-bottom: 1px solid #1A1A1A; }
.viewer-header span { font-size: 12px; color: #787878; }
.viewer-close { background: none; border: none; color: #f87171; font-size: 18px; cursor: pointer; }
.viewer-body { flex: 1; display: flex; overflow: hidden; }
.viewer-img { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; padding: 16px; }
.viewer-img img { max-width: 100%; max-height: 100%; object-fit: contain; }
.viewer-info { width: 300px; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; border-left: 1px solid #1A1A1A; }
.vi-size { color: #585858; font-size: 11px; }
.vi-section label { color: #E2B340; font-size: 10px; font-weight: 700; display: block; margin-bottom: 4px; }
.vi-section label.neg { color: #f87171; }
.vi-section pre { color: #B0B0B0; font-size: 11px; white-space: pre-wrap; word-break: break-all; background: #111; padding: 8px; border-radius: 4px; margin: 0; max-height: 200px; overflow-y: auto; }

.params-grid { background: #111; border-radius: 4px; padding: 6px 8px; }
.param-line { display: flex; align-items: baseline; gap: 8px; padding: 3px 0; font-size: 11px; color: #B0B0B0; border-bottom: 1px solid #1a1a1a; font-family: 'Consolas', monospace; }
.param-line:last-child { border-bottom: none; }
.pl { font-size: 9px; font-weight: 900; color: #E2B340; letter-spacing: 1px; min-width: 45px; flex-shrink: 0; }
.vi-actions { display: flex; flex-wrap: wrap; gap: 4px; margin-top: auto; }
.vi-btn { padding: 6px 12px; background: #181818; border: none; border-radius: 4px; color: #787878; font-size: 11px; cursor: pointer; }
.vi-btn:hover { background: #222; color: #E8E8E8; }
.vi-btn.accent { background: #E2B340; color: #000; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
