<template>
  <div class="wm-overlay" @mousedown.self="closeWeightManager">
    <div class="wm-modal">
      <div class="wm-header">
        <h3>글로벌 태그 가중치</h3>
        <span class="wm-desc">생성 버튼(Ctrl+G)을 누를 때 메인 태그 칸의 같은 태그(쉼표 단위)에 지정 가중치를 붙여 넣습니다</span>
        <button class="close-btn" @click="closeWeightManager"><Icon name="close" /></button>
      </div>
      <div class="wm-body">
        <div v-for="(w, i) in globalWeights" :key="weightRowKey(w)" class="wm-row">
          <input v-model="w.tag" placeholder="태그명..." class="wm-tag-input" />
          <input type="range" min="50" max="200" v-model.number="w.weight" class="wm-slider" />
          <span class="wm-val">{{ (w.weight / 100).toFixed(2) }}</span>
          <button class="wm-rm" @click="removeWeightRow(i)"><Icon name="close" /></button>
        </div>
        <button class="wm-add" @click="addWeightRow">+ ADD TAG WEIGHT</button>
      </div>
      <div class="wm-footer">
        <button class="wm-save" @click="saveGlobalWeights"><Icon name="save" /> 저장</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 글로벌 태그 가중치 모달 — 표시만 한다. 목록은 composables/useGlobalWeights.ts(모듈 싱글턴):
 * 생성 경로가 모달이 닫혀 있어도 같은 목록을 적용한다. App.vue 가 `<transition name="fade">` 안에서 연다.
 */
import { useGlobalWeights } from '../../composables/useGlobalWeights'
import { useModalLayer } from '../../composables/useModalLayer'

const {
  showWeightManager, globalWeights, weightRowKey,
  saveGlobalWeights, addWeightRow, removeWeightRow, closeWeightManager,
} = useGlobalWeights()

useModalLayer({ isOpen: () => showWeightManager.value, close: closeWeightManager })
</script>

<style scoped>
.close-btn { width: 28px; height: 28px; background: var(--bg-button); border: 1px solid var(--border-strong); border-radius: 6px; color: var(--text-secondary); font-size: 16px; cursor: pointer; }
.close-btn:hover { border-color: var(--state-alert-fg); color: var(--state-alert-fg); background: rgba(248, 113, 113, 0.08); }

/* Weight Manager */
.wm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.wm-modal { width: min(680px, 92vw); max-height: 86vh; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.wm-header { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.wm-header h3 { font-size: 12px; letter-spacing: 0; color: var(--text-muted); }
.wm-desc { font-size: var(--fs-label); color: var(--text-muted); flex: 1; }
.wm-body { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 6px; }
.wm-row { display: flex; align-items: center; gap: 6px; }
.wm-tag-input { flex: 1; padding: 5px 8px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); }
.wm-slider { width: 100px; accent-color: var(--accent); }
.wm-val { font-size: 11px; color: var(--accent); min-width: 35px; text-align: right; font-family: monospace; }
.wm-rm { background: none; border: none; color: var(--state-alert-fg); cursor: pointer; font-size: 14px; }
.wm-add { width: 100%; padding: 6px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 4px; color: var(--text-muted); font-size: var(--fs-label); cursor: pointer; }
.wm-footer { padding: 10px 16px; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; }
.wm-save { padding: 7px 20px; background: var(--accent-fill); border: none; border-radius: 6px; color: var(--on-accent); font-size: 11px; font-weight: var(--fw-bold); cursor: pointer; }
</style>
