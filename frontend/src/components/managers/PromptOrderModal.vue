<template>
  <div class="wc-overlay" @mousedown.self="closeOrderManager">
    <div class="wc-modal" style="max-width: 680px;">
      <div class="wc-modal-header">
        <h3>프롬프트 섹션 순서</h3>
        <span class="wc-path">생성 시 합쳐지는 순서를 ↑/↓로 조정. 저장 시 즉시 반영</span>
        <button class="close-btn" @click="closeOrderManager"><Icon name="close" /></button>
      </div>
      <div class="order-modal-body">
        <div class="order-list">
          <div v-for="(sec, i) in promptOrderList" :key="sec.key" class="order-item">
            <span class="order-num">{{ i + 1 }}</span>
            <span class="order-label">{{ sec.label }}</span>
            <button class="order-btn" :disabled="i === 0" @click="moveOrderUp(i)" title="위로"><Icon name="arrow-up" /></button>
            <button class="order-btn" :disabled="i === promptOrderList.length - 1" @click="moveOrderDown(i)" title="아래로"><Icon name="arrow-down" /></button>
          </div>
        </div>
        <div class="order-preview">
          <span class="order-preview-label">미리보기:</span>
          <code class="order-preview-text">{{ promptOrderList.map(s => s.label).join(' → ') }}</code>
        </div>
        <div class="order-actions">
          <button class="order-reset" @click="resetPromptOrder"><Icon name="rotate-ccw" /> 기본값 복원</button>
          <div style="flex: 1;"></div>
          <button class="order-cancel" @click="closeOrderManager">취소</button>
          <button class="order-save" @click="saveOrderAndClose"><Icon name="save" /> 저장 & 적용</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 프롬프트 섹션 순서 모달 — 표시만 한다. 상태·동작은 composables/usePromptOrder.ts.
 * App.vue 가 `<transition name="fade">` 안에서 v-if 로 연다.
 */
import { usePromptOrder } from '../../composables/usePromptOrder'
import { useModalLayer } from '../../composables/useModalLayer'

const {
  showOrderManager, promptOrderList,
  moveOrderUp, moveOrderDown, saveOrderAndClose, resetPromptOrder, closeOrderManager,
} = usePromptOrder()

useModalLayer({ isOpen: () => showOrderManager.value, close: closeOrderManager })
</script>

<style scoped src="./managerModal.css"></style>
