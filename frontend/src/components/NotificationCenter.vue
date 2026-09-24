<template>
  <!-- 알림 벨도 z-index 2003 이라 게이트를 뚫고 뜬다. 게이트 동안엔 숨긴다(showBell) —
       고를 것이 하나뿐인 화면에 눌러도 뒤가 안 보이는 버튼이 떠 있으면 안 된다. -->
  <button v-if="showBell" class="notif-bell" @click.stop="toggleNotifPanel" title="알림 기록"><Icon name="bell" /><span v-if="unread > 0" class="notif-badge">{{ unreadBadgeText(unread) }}</span>
  </button>
  <div v-if="showNotifPanel" class="notif-overlay" @click="closePanel"></div>
  <transition name="fade">
    <div v-if="showNotifPanel" class="notif-panel" @click.stop>
      <div class="notif-head">
        <span>알림 기록</span>
        <button v-if="toastHistory.length" class="notif-clear" @click="clearNotifHistory">모두 지우기</button>
      </div>
      <div class="notif-list">
        <div v-for="n in toastHistory" :key="n.id" class="notif-item" :class="n.type">
          <span class="notif-ico"><Icon :name="n.type === 'error' || n.type === 'warning' ? 'alert' : n.type === 'success' ? 'check' : 'info'" /></span>
          <span class="notif-msg">{{ n.msg }}</span>
          <span class="notif-time">{{ relativeTimeKo(n.ts) }}</span>
        </div>
        <div v-if="toastHistory.length === 0" class="notif-empty">알림 없음</div>
      </div>
    </div>
  </transition>
</template>

<script setup lang="ts">
/** 알림 벨 + 알림 기록 패널 — 사라진 토스트를 다시 본다. 목록은 composables/useToasts.ts. */
import { useToasts } from '../composables/useToasts'
import { relativeTimeKo, unreadBadgeText } from '../utils/relativeTime'

withDefaults(defineProps<{ showBell?: boolean }>(), { showBell: true })

const { toastHistory, showNotifPanel, unread, toggleNotifPanel, clearNotifHistory } = useToasts()
function closePanel() { showNotifPanel.value = false }
</script>

<style scoped>
/* 알림 벨 + 히스토리 패널 */
.notif-bell { position: fixed; top: 14px; right: 18px; z-index: 2003; width: 34px; height: 34px; border-radius: 50%; background: var(--bg-button); border: 1px solid var(--border); color: var(--text-secondary); font-size: 15px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.notif-bell:hover { border-color: var(--accent); color: var(--accent); }
.notif-badge { position: absolute; top: -4px; right: -4px; min-width: 16px; height: 16px; padding: 0 4px; border-radius: 8px; background: var(--state-alert); color: #fff; font-size: var(--fs-label); font-weight: var(--fw-bold); display: flex; align-items: center; justify-content: center; }
.notif-overlay { position: fixed; inset: 0; z-index: 2002; }
.notif-panel { position: fixed; top: 54px; right: 18px; z-index: 2003; width: 330px; max-height: 60vh; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 12px 32px rgba(0,0,0,0.6); display: flex; flex-direction: column; overflow: hidden; }
.notif-head { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 12px; font-weight: var(--fw-bold); color: var(--text-secondary); letter-spacing: 0; }
.notif-clear { background: none; border: none; color: var(--state-alert-fg); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.notif-list { overflow-y: auto; padding: 6px; }
.notif-item { display: flex; align-items: flex-start; gap: 8px; padding: 8px 10px; border-radius: 6px; font-size: 12px; color: var(--text-primary); }
.notif-item:hover { background: var(--bg-button); }
.notif-ico { flex-shrink: 0; }
.notif-item.error .notif-ico { color: var(--state-alert-fg); }
.notif-item.success .notif-ico { color: var(--state-ok-fg); }
.notif-item.info .notif-ico { color: var(--state-info-fg); }
.notif-item.warning .notif-ico { color: var(--state-warn-fg); }
.notif-msg { flex: 1; word-break: break-word; line-height: 1.4; }
.notif-time { flex-shrink: 0; font-size: var(--fs-label); color: var(--text-muted); white-space: nowrap; }
.notif-empty { padding: 24px; text-align: center; color: var(--text-muted); font-size: 12px; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
