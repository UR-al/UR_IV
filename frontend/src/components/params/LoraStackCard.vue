<template>
  <div id="sec-lora" class="ext-card">
    <div class="ext-title">LoRA 스택
      <span v-if="loraStack.length" class="lora-tools">
        <button class="lora-tool-btn" @click="toggleAllLoras(!allLorasOn)" :title="allLorasOn ? '전체 끄기' : '전체 켜기'">{{ allLorasOn ? '전체 OFF' : '전체 ON' }}</button>
        <button class="lora-tool-btn" @click="insertAllTriggers" title="활성 LoRA 트리거 워드를 메인 프롬프트에 일괄 삽입">트리거 삽입</button>
      </span>
    </div>
    <div class="lora-empty" v-if="loraStack.length === 0">
      LoRA 매니저에서 추가하세요
    </div>
    <template v-for="(lora, i) in loraStack" :key="lora.name">
    <div class="lora-drop-marker" v-if="loraDragIdx >= 0 && loraDropIdx === i && i !== loraDragIdx && i !== loraDragIdx + 1"></div>
    <div class="lora-block" :class="{ 'lora-dragging': loraDragIdx === i }"
      @dragover.prevent="loraDragOver($event, i)" @drop.prevent="loraDrop">
      <span class="lora-grip" title="여기(⠿)를 잡아 순서 변경" draggable="true"
        @dragstart.stop="loraDragStart(i)" @dragend="loraDragEnd">⠿</span>
      <label class="lora-check"><input type="checkbox" v-model="lora.enabled" /></label>
      <div class="lora-info-col">
        <div class="lora-name">{{ lora.name }}</div>
        <div class="lora-triggers" v-if="lora.triggerWords && lora.triggerWords.length">
          <button v-for="tw in lora.triggerWords" :key="tw" class="trigger-chip"
            @click="insertTriggerWord(tw)" :title="'클릭하여 프롬프트에 삽입'">{{ tw }}</button>
        </div>
      </div>
      <input type="range" min="-100" max="200" v-model.number="lora.weight" class="lora-slider" />
      <input type="number" class="lora-weight-input" :value="(lora.weight / 100).toFixed(2)" step="0.05" min="-1" max="3"
        title="가중치 직접 입력" @change="lora.weight = Math.round((parseFloat(($event.target as HTMLInputElement).value) || 0) * 100)" @mousedown.stop />
      <button class="lora-remove" @click="loraStack.splice(i, 1)"><Icon name="close" /></button>
    </div>
    </template>
    <div class="lora-drop-marker" v-if="loraDragIdx >= 0 && loraDropIdx === loraStack.length && loraStack.length !== loraDragIdx + 1"></div>
    <button class="ext-add-btn" @click="showLoraModal = true">+ ADD LoRA</button>
    <!-- LoRA 세트 저장/불러오기 -->
    <div class="lora-sets">
      <input v-model="loraSetName" class="lora-set-input" placeholder="세트 이름" />
      <button class="lora-tool-btn" @click="saveLoraSet" :disabled="!loraStack.length || !loraSetName.trim()" title="현재 스택을 세트로 저장">저장</button>
      <select v-model="loraSetSel" class="lora-set-sel">
        <option value="">불러오기…</option>
        <option v-for="n in loraSetNames" :key="n" :value="n">{{ n }}</option>
      </select>
      <button class="lora-tool-btn" @click="loadLoraSet" :disabled="!loraSetSel" title="선택 세트를 스택에 적용">적용</button>
      <button class="lora-tool-btn" @click="deleteLoraSet" :disabled="!loraSetSel" title="세트 삭제"><Icon name="close" /></button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — LoRA 스택 카드(`#sec-lora`). App.vue 에서 추출(App.vue 분할 ④).
 * 스택 상태의 주인은 App(생성 직전 syncLoraStack · uiPrefsLoaded 복원 · LoRA 매니저 모달)이라
 * useLoraStack 결과를 prop 으로 받아 그린다. '+ ADD LoRA' 는 같은 객체의 showLoraModal 을 켠다.
 */
import type { PropType } from 'vue'
import type { useLoraStack } from '../../composables/useLoraStack.js'

const props = defineProps({
  lora: { type: Object as PropType<ReturnType<typeof useLoraStack>>, required: true },
})

const {
  loraStack,
  allLorasOn, toggleAllLoras, insertTriggerWord, insertAllTriggers,
  showLoraModal,
  loraSetName, loraSetSel, loraSetNames, saveLoraSet, loadLoraSet, deleteLoraSet,
  loraDragIdx, loraDropIdx, loraDragStart, loraDragOver, loraDragEnd, loraDrop,
} = props.lora
</script>

<style scoped>
/* LoRA block */
.lora-block { display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: var(--bg-button); border-radius: 6px; margin-bottom: 4px; }
.lora-info-col { flex: 1; min-width: 0; }
.lora-name { font-size: 11px; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lora-triggers { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 3px; }
.trigger-chip {
  padding: 1px 6px; font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent);
  background: var(--accent-dim); border: 1px solid rgba(250, 204, 21, 0.15);
  border-radius: 4px; cursor: pointer; transition: var(--transition);
}
.trigger-chip:hover { background: rgba(250, 204, 21, 0.2); border-color: var(--accent); }
.lora-slider { width: 60px; accent-color: var(--accent); }
.lora-grip { cursor: grab; color: var(--text-muted); font-size: 12px; user-select: none; flex-shrink: 0; }
.lora-grip:active { cursor: grabbing; }
.lora-grip:hover { color: var(--accent); }
.lora-dragging { opacity: 0.4; }
/* 드롭(삽입) 위치 — 블록 사이 가로선 (태그블록 마커와 동일 콘셉트) */
.lora-drop-marker { height: 3px; background: var(--accent); border-radius: 2px; margin: 1px 2px; pointer-events: none; box-shadow: 0 0 4px var(--accent); }
.lora-weight-input { width: 48px; flex-shrink: 0; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; padding: 2px 4px; color: var(--accent); font-size: var(--fs-label); text-align: center; font-family: monospace; }
.lora-weight-input:focus { outline: none; border-color: var(--accent); }
.lora-remove { background: none; border: none; color: var(--state-alert-fg); cursor: pointer; font-size: 12px; }
.lora-tools { float: right; display: inline-flex; gap: 4px; }
.lora-tool-btn { background: var(--bg-button); border: 1px solid var(--border); border-radius: 5px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 2px 7px; cursor: pointer; }
.lora-tool-btn:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
.lora-tool-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.lora-sets { display: flex; gap: 4px; margin-top: 6px; align-items: center; }
.lora-set-input { width: 70px; flex-shrink: 0; background: var(--bg-input); border: 1px solid var(--border); border-radius: 5px; padding: 4px 6px; color: var(--text-primary); font-size: var(--fs-label); }
.lora-set-sel { flex: 1; min-width: 0; background: var(--bg-input); border: 1px solid var(--border); border-radius: 5px; padding: 4px; color: var(--text-primary); font-size: var(--fs-label); }
.ext-add-btn { width: 100%; padding: 8px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 6px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; margin-top: 4px; }
.lora-check { flex-shrink: 0; }
.lora-check input { accent-color: var(--accent); }
.lora-empty { font-size: 11px; color: var(--text-muted); text-align: center; padding: 12px; }
</style>
