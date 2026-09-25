<template>
  <div ref="dockRef" class="quick-dock">
    <!-- 대기열 드로어는 예전 그대로 — 여닫기만 도크가 한다 -->
    <QueuePanel ref="queueRef" v-model:open="queueOpen" />

    <!-- 펼친 아이콘 줄 — 대기열 · 대화 · 메모장 -->
    <transition name="qdk-pop">
      <div v-if="expanded" id="quick-dock-menu" class="qdk-menu" role="menu" aria-label="빠른 도구">
        <button v-for="item in DOCK_ITEMS" :key="item.id" type="button" role="menuitem" class="qdk-item"
          :class="{ on: activePanel === item.id }" :title="item.hint" :aria-label="item.label"
          @click="togglePanel(item.id)">
          <span class="qdk-label">{{ item.label }}</span>
          <span class="qdk-icon">
            <Icon :name="item.icon" />
            <span v-if="item.id === 'queue' && queueCount" class="qdk-badge">{{ queueCount }}</span>
          </span>
        </button>
      </div>
    </transition>

    <!-- 항상 보이는 핀 — 접혀도 대기열 개수 · 실행 상태가 보인다. 누르면 아이콘 셋이 펼쳐진다. -->
    <button type="button" class="queue-pin"
      :class="{ running: queueRunning, paused: queuePaused, open: expanded || !!activePanel, bump: queueBump }"
      :aria-expanded="expanded" aria-haspopup="menu" aria-controls="quick-dock-menu"
      :title="queueRunning ? '대기열 실행 중 — 눌러서 대기열 · 대화 · 메모장' : '대기열 · 대화 · 메모장'"
      @click="toggleExpanded">
      <Icon name="list" class="qp-ico" />
      <span class="qp-label">대기열</span>
      <span v-if="queueCount" class="qp-count">{{ queueCount }}</span>
      <span v-if="queueRunning" class="qp-run" title="실행 중"><Icon name="play" /></span>
      <span v-else-if="queuePaused" class="qp-run paused" title="일시정지"><Icon name="pause" /></span>
      <Icon name="chevron-up" class="qp-caret" :class="{ flipped: expanded }" />
    </button>

    <!-- 한 번 열면 계속 마운트해 둔다 — 닫아도 흐르던 답 · 밀린 메모 저장이 끊기지 않게 -->
    <ChatMiniPanel v-if="chatMounted" :open="activePanel === 'chat'" @close="closePanel('chat')" />
    <MemoPanel v-if="memoMounted" :open="activePanel === 'memo'" @close="closePanel('memo')" />
  </div>
</template>

<script setup lang="ts">
/**
 * 우하단 빠른 도크 — 예전 '대기열' 핀 자리.
 *
 * 핀을 누르면 대기열 · 대화 · 메모장 세 아이콘이 펼쳐지고, 고른 것이 오른쪽에 열린다.
 * 대기열은 기존 드로어(QueuePanel) 그대로, 대화는 대화 탭과 같은 대화 목록을 쓰는 작은 패널,
 * 메모장은 Forge Notebook 메모와 공유되는 메모. 어느 것이 열렸는지는 composables/useQuickDock.
 * 펼친 아이콘 줄은 ESC · 바깥 클릭으로 접힌다(앱 모달 스택 — utils/modalStack).
 */
import { computed, defineAsyncComponent, onMounted, onUnmounted, ref, watch } from 'vue'
import QueuePanel from '../QueuePanel.vue'

// 대화 · 메모장 패널은 처음 열 때 불러온다 — 대화 로직(대화 탭과 공유)이 첫 화면 번들에 붙지 않게
const ChatMiniPanel = defineAsyncComponent(() => import('./ChatMiniPanel.vue'))
const MemoPanel = defineAsyncComponent(() => import('./MemoPanel.vue'))
import { useQuickDock, type DockPanel } from '../../composables/useQuickDock'
import { useModalLayer } from '../../composables/useModalLayer'

const DOCK_ITEMS: ReadonlyArray<{ id: DockPanel; label: string; icon: string; hint: string }> = [
  { id: 'queue', label: '대기열', icon: 'list', hint: '대기열 — 예약한 생성 관리' },
  { id: 'chat', label: '대화', icon: 'message', hint: '대화 — 대화 탭과 같은 대화를 작게' },
  { id: 'memo', label: '메모장', icon: 'note', hint: '메모장 — Forge Notebook 메모와 공유' },
]

const { expanded, activePanel, toggleExpanded, collapse, openPanel, togglePanel, closePanel } = useQuickDock()

// 대기열 드로어 열림 = 도크의 'queue' 패널. 드로어가 스스로 닫으면(ESC · 바깥 · X) 도크도 닫는다.
const queueOpen = computed({
  get: () => activePanel.value === 'queue',
  set: (open: boolean) => { if (open) openPanel('queue'); else closePanel('queue') },
})

// 핀에 보일 대기열 상태 — QueuePanel 이 defineExpose 로 내준다
const queueRef = ref<InstanceType<typeof QueuePanel> | null>(null)
const queueCount = computed(() => queueRef.value?.items.length ?? 0)
const queuePaused = computed(() => !!queueRef.value?.isPaused)
const queueRunning = computed(() => !!queueRef.value?.isRunning && !queuePaused.value)
const queueBump = computed(() => !!queueRef.value?.pinBump)

// 대화 · 메모장 패널은 처음 열 때 마운트한다(그 전엔 대화 목록 · 설정을 불러오지 않는다)
const chatMounted = ref(false)
const memoMounted = ref(false)
watch(activePanel, (panel) => {
  if (panel === 'chat') chatMounted.value = true
  if (panel === 'memo') memoMounted.value = true
}, { immediate: true })

// 펼친 아이콘 줄 — ESC 로 접는다(모달 스택 맨 위일 때). 떠 있는 동안 ↑/↓ 히스토리는 멈춘다.
useModalLayer({ isOpen: () => expanded.value, close: collapse })

// 바깥을 누르면 접는다
const dockRef = ref<HTMLElement | null>(null)
function onPointerDown(e: PointerEvent) {
  if (!expanded.value) return
  const root = dockRef.value
  if (root && e.target instanceof Node && root.contains(e.target)) return
  collapse()
}
onMounted(() => document.addEventListener('pointerdown', onPointerDown, true))
onUnmounted(() => document.removeEventListener('pointerdown', onPointerDown, true))
</script>

<style scoped>
.quick-dock { display: contents; }  /* 흐름을 차지하지 않는다 — 핀 · 아이콘 · 패널은 모두 fixed */

/* ── 핀 (항상 보임, 우하단 플로팅) — 예전 QueuePanel 의 .queue-pin ──
   클래스 이름은 그대로 둔다: 아이콘 모션(styles/iconMotion*.css)의 '.queue-pin.running' 재생 표시가 건다. */
.queue-pin {
  position: fixed; right: 18px; bottom: 32px;  /* 하단 VRAM 바(22px) 위로 띄움 */
  z-index: 2400; display: flex; align-items: center; gap: var(--sp-2);
  height: 34px; padding: 0 var(--sp-3) 0 var(--sp-4); border-radius: 18px;
  background: var(--bg-button); border: 1px solid var(--border); color: var(--text-muted);
  font-size: var(--fs-meta); font-weight: var(--fw-bold); letter-spacing: 0; cursor: pointer;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4); transition: 0.18s;
}
.queue-pin:hover { background: var(--bg-button-hover); color: var(--text-primary); border-color: var(--accent); }
.queue-pin.open { border-color: var(--accent); color: var(--accent); }
.queue-pin.running { border-color: var(--state-ok-fg); color: var(--state-ok-fg); }
.queue-pin.paused { border-color: var(--state-warn-fg); color: var(--state-warn-fg); }
.queue-pin.bump { animation: pin-bump 0.55s ease; }
@keyframes pin-bump { 0% { transform: scale(1); } 30% { transform: scale(1.12); } 100% { transform: scale(1); } }
.qp-ico { opacity: 0.9; }
.qp-label { letter-spacing: 0; }
/* 상태색으로 채운 배지의 글자는 '바탕색 뒤집기'(--bg-primary) — 상태색은 다크에서
   밝고 라이트에서 어두워 두 모드 모두 대비가 나오고, 사용자 강조색에 묶이지도 않는다 */
.qp-count { background: var(--accent-fill); color: var(--on-accent); min-width: 16px; text-align: center; padding: 1px 5px; border-radius: 9px; font-size: var(--fs-label); font-weight: var(--fw-bold); }
.queue-pin.running .qp-count { background: var(--state-ok-fg); color: var(--bg-primary); }
.queue-pin.paused .qp-count { background: var(--state-warn-fg); color: var(--bg-primary); }
.qp-run { font-size: var(--fs-label); }
.qp-run.paused { color: var(--state-warn-fg); }
.qp-caret { font-size: var(--fs-label); opacity: 0.8; transition: transform 0.18s; }
.qp-caret.flipped { transform: rotate(180deg); }

/* ── 펼친 아이콘 줄 — 핀 바로 위, 오른쪽 끝을 맞춘 세로 줄 ── */
.qdk-menu {
  position: fixed; right: 18px; bottom: 74px; z-index: 2410;
  display: flex; flex-direction: column; align-items: flex-end; gap: var(--sp-2);
}
.qdk-item {
  display: flex; align-items: center; gap: var(--sp-2);
  padding: 0; border: 0; background: transparent; cursor: pointer; color: var(--text-secondary);
}
.qdk-label {
  padding: var(--sp-1) var(--sp-2); border-radius: var(--radius-base);
  background: var(--bg-secondary); border: 1px solid var(--edge);
  font-size: var(--fs-meta); font-weight: var(--fw-medium); color: var(--text-primary); white-space: nowrap;
}
.qdk-icon {
  position: relative; display: inline-flex; align-items: center; justify-content: center;
  width: 40px; height: 40px; border-radius: 50%;
  background: var(--bg-button); border: 1px solid var(--edge);
  font-size: 17px; box-shadow: 0 4px 16px rgba(0,0,0,0.35); transition: 0.15s;
}
.qdk-item:hover .qdk-icon, .qdk-item:focus-visible .qdk-icon { color: var(--text-primary); border-color: var(--accent); background: var(--bg-button-hover); }
.qdk-item.on .qdk-icon { color: var(--accent); border-color: var(--accent); }
.qdk-badge {
  position: absolute; top: -4px; right: -4px; min-width: 16px; height: 16px; padding: 0 var(--sp-1);
  border-radius: 8px; background: var(--accent-fill); color: var(--on-accent);
  font-size: var(--fs-label); font-weight: var(--fw-bold); line-height: 16px; text-align: center;
}
.qdk-pop-enter-active, .qdk-pop-leave-active { transition: opacity 0.15s ease, transform 0.15s ease; }
.qdk-pop-enter-from, .qdk-pop-leave-to { opacity: 0; transform: translateY(var(--sp-2)); }
@media (prefers-reduced-motion: reduce) {
  .qdk-pop-enter-active, .qdk-pop-leave-active, .qp-caret { transition: none; }
}
</style>
