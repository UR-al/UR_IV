<template>
  <div class="pm-overlay" @mousedown.self="closePresetManager">
    <div class="pm-modal">
      <div class="pm-header">
        <h3>프리셋 관리</h3>
        <button class="close-btn" @click="closePresetManager"><Icon name="close" /></button>
      </div>
      <div class="pm-body">
        <div class="pm-list">
          <div v-for="p in presetList" :key="p" class="pm-item"
            :class="{ active: selectedPreset === p }" @click="selectPreset(p)">
            {{ p }}
          </div>
          <div v-if="presetList.length === 0" class="pm-empty">저장된 프리셋 없음</div>
        </div>
        <div class="pm-preview">
          <div v-if="presetPreview" class="pm-detail">
            <div class="pm-field" v-for="(val, key) in presetPreview" :key="key">
              <span class="pm-key">{{ key }}</span>
              <span class="pm-val">{{ typeof val === 'string' ? val.substring(0, 100) : val }}</span>
            </div>
          </div>
          <div v-else class="pm-empty">프리셋을 선택하세요</div>
        </div>
      </div>
      <div class="pm-footer">
        <button class="pm-btn" @click="loadSelectedPreset" :disabled="!selectedPreset"><Icon name="folder-open" /> 불러오기</button>
        <button class="pm-btn" @click="deleteSelectedPreset" :disabled="!selectedPreset"><Icon name="trash" /> 삭제</button>
        <div class="pm-spacer"></div>
        <button class="pm-btn accent" @click="saveNewPreset"><Icon name="save" /> 현재 상태 저장</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 프리셋 관리 모달 — 표시만 한다. 상태·동작은 composables/usePresetManager.ts.
 * App.vue 가 `<transition name="fade">` 안에서 v-if 로 연다.
 */
import { usePresetManager } from '../../composables/usePresetManager'
import { useModalLayer } from '../../composables/useModalLayer'

const {
  showPresetManager, presetList, selectedPreset, presetPreview,
  selectPreset, loadSelectedPreset, deleteSelectedPreset, saveNewPreset, closePresetManager,
} = usePresetManager()

useModalLayer({ isOpen: () => showPresetManager.value, close: closePresetManager })
</script>

<style scoped>
.close-btn { width: 28px; height: 28px; background: var(--bg-button); border: 1px solid var(--border-strong); border-radius: 6px; color: var(--text-secondary); font-size: 16px; cursor: pointer; }
.close-btn:hover { border-color: var(--state-alert-fg); color: var(--state-alert-fg); background: rgba(248, 113, 113, 0.08); }

/* Preset Manager */
.pm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.pm-modal { width: min(860px, 94vw); height: min(640px, 90vh); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.pm-header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.pm-header h3 { font-size: 12px; letter-spacing: 0; color: var(--text-muted); flex: 1; }
.pm-body { flex: 1; display: flex; overflow: hidden; }
.pm-list { width: 180px; overflow-y: auto; border-right: 1px solid var(--border); padding: 8px; }
.pm-item { padding: 7px 10px; font-size: 11px; color: var(--text-secondary); cursor: pointer; border-radius: 4px; margin-bottom: 2px; }
.pm-item:hover { background: var(--bg-input); }
.pm-item.active { background: var(--accent-dim); color: var(--accent); }
.pm-preview { flex: 1; overflow-y: auto; padding: 12px; }
.pm-detail { display: flex; flex-direction: column; gap: 4px; }
.pm-field { display: flex; gap: 8px; font-size: var(--fs-label); border-bottom: 1px solid rgba(255,255,255,0.03); padding: 3px 0; }
.pm-key { color: var(--accent); font-weight: var(--fw-bold); min-width: 100px; }
.pm-val { color: var(--text-secondary); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pm-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); }
.pm-footer { display: flex; align-items: center; gap: 6px; padding: 10px 16px; border-top: 1px solid var(--border); }
.pm-btn { padding: 6px 14px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.pm-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.pm-btn:disabled { opacity: 0.3; }
.pm-btn.accent { background: var(--accent-fill); color: var(--on-accent); border: none; }
.pm-spacer { flex: 1; }
</style>
