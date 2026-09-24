<template>
  <Teleport to="body">
    <div class="toast-container" :class="{ 'toast-multi': toasts.length > 1 }">
      <button v-if="toasts.length > 1" class="toast-clear-all" @click="clearAllToasts" title="모두 닫기">
        모두 닫기 ({{ toasts.length }})
      </button>
      <transition-group name="toast" tag="div" class="toast-stack">
        <div v-for="t in toasts" :key="t.id" class="toast" :class="t.type">
          <span class="toast-icon"><Icon :name="t.type === 'error' ? 'alert' : t.type === 'success' ? 'check' : 'info'" /></span>
          <span class="toast-msg">{{ t.msg }}</span>
          <span v-if="t.count > 1" class="toast-count">×{{ t.count }}</span>
          <button class="toast-close" @click.stop="removeToast(t.id)" title="닫기"><Icon name="close" /></button>
        </div>
      </transition-group>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
/** 전역 토스트 스택 — 목록은 composables/useToasts.ts(모듈 싱글턴). App.vue 가 한 번 놓는다. */
import { useToasts } from '../composables/useToasts'

const { toasts, removeToast, clearAllToasts } = useToasts()
</script>

<style scoped>
/* Toast Notifications */
/* 토스트의 #fff/#000 은 토큰으로 못 바꾼다 — 면이 rgba 고정색(초록·빨강·파랑·노랑)
   이라 테마와 무관하다. --text-primary 를 쓰면 라이트에서 검은 글자가 빨간
   토스트에 얹혀 읽히지 않는다. .toast-clear-all 도 면이 rgba(0,0,0,.6) 고정이다. */
.toast-container {
  position: fixed; top: 70px; right: 20px; z-index: 99999;
  display: flex; flex-direction: column; gap: 6px; pointer-events: none;
  max-width: 400px;
}
.toast-stack { display: flex; flex-direction: column; gap: 6px; pointer-events: auto; }
.toast-clear-all {
  align-self: flex-end; padding: 3px 10px; font-size: var(--fs-label); font-weight: var(--fw-bold);
  background: rgba(0,0,0,0.6); color: #fff; border: 1px solid rgba(255,255,255,0.2);
  border-radius: 4px; cursor: pointer; pointer-events: auto; backdrop-filter: blur(8px);
}
.toast-clear-all:hover { background: rgba(248,113,113,0.5); border-color: var(--state-alert-fg); }
.toast {
  display: flex; align-items: center; gap: 8px;
  padding: 11px 13px 11px 15px; border-radius: 9px; font-size: 13px; font-weight: var(--fw-bold);
  color: #FFF; pointer-events: auto; min-width: 240px; max-width: 400px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4); backdrop-filter: blur(8px);
  word-break: break-word; line-height: 1.45;
}
.toast.success { background: rgba(74, 222, 128, 0.92); color: #000; }
.toast.error { background: rgba(248, 113, 113, 0.92); color: #FFF; }
.toast.info { background: rgba(96, 165, 250, 0.92); color: #FFF; }
.toast.warning { background: rgba(251, 191, 36, 0.92); color: #000; }
.toast-icon { font-size: 16px; flex-shrink: 0; }
.toast-msg { flex: 1; }
.toast-count {
  background: rgba(0,0,0,0.25); padding: 1px 6px; border-radius: 8px;
  font-size: var(--fs-label); flex-shrink: 0;
}
.toast-close {
  width: 18px; height: 18px; padding: 0; flex-shrink: 0;
  background: rgba(0,0,0,0.2); border: none; border-radius: 50%;
  color: inherit; font-size: var(--fs-label); cursor: pointer; opacity: 0.7;
}
.toast-close:hover { opacity: 1; background: rgba(0,0,0,0.4); }
.toast-enter-active { animation: slideIn 0.25s ease; }
.toast-leave-active { animation: slideOut 0.25s ease; }
@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes slideOut { from { opacity: 1; max-height: 60px; } to { transform: translateX(100%); opacity: 0; max-height: 0; padding: 0; margin: 0; } }
</style>
