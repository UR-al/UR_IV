<template>
  <div class="ovr-overlay" @mousedown.self="close">
    <div class="ovr-modal">
      <div class="ovr-header">
        <div>
          <h3>특징 override</h3>
          <span class="ovr-sub">auto remove 시, 선택한 카테고리는 덱 태그를 제거하고 캐릭터 특징으로 강제 교체합니다</span>
        </div>
        <button class="ovr-close" @click="close"><Icon name="close" /></button>
      </div>

      <div class="ovr-body">
        <label class="ovr-row">
          <input type="checkbox" v-model="hairLength" />
          <div>
            <span class="ovr-name">머리 길이 override</span>
            <span class="ovr-desc">덱에 short hair 등이 있어도 캐릭터의 머리 길이로 교체</span>
          </div>
        </label>
        <label class="ovr-row">
          <input type="checkbox" v-model="eyeColor" />
          <div>
            <span class="ovr-name">눈 색 override</span>
            <span class="ovr-desc">덱에 closed eyes가 있어도 제거하고 캐릭터 눈 색을 추가</span>
          </div>
        </label>
      </div>

      <div class="ovr-footer">
        <div class="ovr-spacer"></div>
        <button class="ovr-save" @click="save">저장</button>
        <button class="ovr-cancel" @click="close">닫기</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { getBackend } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { useModalLayer } from '../composables/useModalLayer'

const emit = defineEmits<{ close: [] }>()
const hairLength = ref(false)
const eyeColor = ref(false)

function close() { emit('close') }
function onKey(e: KeyboardEvent) { if (e.key === 'Escape') { e.stopPropagation(); close() } }
// 열려 있는 동안 앱 모달 스택에 올라간다 — App 의 ↑/↓ 히스토리 이동이 이 모달 뒤에서 넘어가지 않게.
// ESC 는 위 onKey 가 직접 처리한다(window capture + stopPropagation, utils/modalStack).
useModalLayer()

async function save() {
  const bk: any = await getBackend()   // QWebChannel 백엔드 — 동적 타입
  if (bk && bk.setCharFeatureOverride) {
    bk.setCharFeatureOverride(JSON.stringify({ hair_length: hairLength.value, eye_color: eyeColor.value }), (_json: string) => { /* 응답 무시 가능 */ })
  }
  requestAction('show_toast', { type: 'success', msg: 'override 설정 저장' })
  close()
}

onMounted(async () => {
  window.addEventListener('keydown', onKey, true)
  const bk: any = await getBackend()
  if (bk && bk.getCharFeatureOverride) {
    bk.getCharFeatureOverride((json: string) => {
      try {
        const d = JSON.parse(json)
        hairLength.value = !!d.hair_length
        eyeColor.value = !!d.eye_color
      } catch {}
    })
  }
})
onUnmounted(() => window.removeEventListener('keydown', onKey, true))
</script>

<style scoped>
.ovr-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.72); z-index: 2300; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px); }
.ovr-modal { width: min(520px, 92vw); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-card); display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.6); }
.ovr-header { display: flex; align-items: flex-start; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); }
.ovr-header h3 { font-size: 16px; font-weight: var(--fw-bold); }
.ovr-sub { font-size: 11px; color: var(--text-muted); display: block; margin-top: 2px; max-width: 400px; }
.ovr-close { width: 30px; height: 30px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); cursor: pointer; }
.ovr-close:hover { color: var(--text-primary); border-color: var(--accent); }
.ovr-body { padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
.ovr-row { display: flex; align-items: flex-start; gap: 10px; cursor: pointer; padding: 10px 12px; border: 1px solid var(--border); border-radius: var(--radius-base); background: var(--bg-primary); }
.ovr-row:hover { border-color: var(--accent); }
.ovr-row input { margin-top: 3px; accent-color: var(--accent); }
.ovr-name { display: block; font-size: 13px; font-weight: var(--fw-bold); color: var(--text-primary); }
.ovr-desc { display: block; font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.ovr-footer { display: flex; gap: 8px; padding: 14px 20px; border-top: 1px solid var(--border); }
.ovr-spacer { flex: 1; }
/* 주 버튼: 글자를 얹는 면이라 --accent 가 아니라 --accent-fill + --on-accent */
.ovr-save { background: var(--accent-fill); color: var(--on-accent); border: none; border-radius: var(--radius-base); font-weight: var(--fw-bold); font-size: 12px; padding: 9px 18px; cursor: pointer; }
.ovr-cancel { background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-weight: var(--fw-bold); font-size: 12px; padding: 9px 16px; cursor: pointer; }
</style>
