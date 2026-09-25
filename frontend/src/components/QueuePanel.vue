<template>
  <div class="queue-root">
    <!-- 항상 보이는 핀(카운트 · 실행 상태)은 우하단 도크의 버튼이 됐다(components/dock/QuickDock.vue).
         도크가 v-model:open 으로 이 드로어를 열고, 핀에 보일 상태는 defineExpose 로 읽어 간다. -->

    <!-- 우측 슬라이드 드로어 — 세로 리스트라 항목이 안 찌그러짐 -->
    <transition name="qd-fade">
      <div class="qd-backdrop" v-if="drawerOpen" @click="drawerOpen = false"></div>
    </transition>
    <transition name="qd-slide">
      <aside class="queue-drawer" v-if="drawerOpen">
        <div class="qd-head">
          <span class="title">
            대기열
            <span class="count-badge" v-if="items.length">{{ items.length }}</span>
            <span class="running-badge" v-if="isRunning && !isPaused"><Icon name="play" /> 실행 중</span>
            <span class="paused-badge" v-if="isPaused"><Icon name="pause" /> 일시정지</span>
            <span class="selected-badge" v-if="selectedIds.size > 0">{{ selectedIds.size }} 선택됨</span>
          </span>
          <button class="qd-x" @click="drawerOpen = false" title="닫기 (Esc)"><Icon name="close" /></button>
        </div>

        <div class="qd-actions">
          <button class="btn" @click="startQueue" v-if="items.length && !isRunning" :disabled="automationRunning"
            :title="automationRunning ? AUTOMATION_OWNS_QUEUE : '대기열 시작'"><Icon name="play" /> 시작</button>
          <button class="btn" @click="pauseQueue" v-if="isRunning && !isPaused" title="일시정지"><Icon name="pause" /> 일시정지</button>
          <button class="btn primary" @click="resumeQueue" v-if="isPaused" title="재개"><Icon name="play" /> 재개</button>
          <button class="btn danger" @click="stopQueue" v-if="isRunning"><Icon name="stop" /> 중지</button>
          <button class="btn danger" @click="removeSelected" v-if="selectedIds.size > 0" title="선택한 항목 삭제"><Icon name="trash" /> 선택삭제 ({{ selectedIds.size }})
          </button>
          <button class="btn" @click="clearAll" v-if="items.length && !isRunning && selectedIds.size === 0"><Icon name="trash" /> 전체</button>
        </div>

        <div class="queue-progress" v-if="isRunning">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: progressPct + '%' }"></div>
          </div>
          <span class="progress-text">
            {{ completedCount }} / {{ items.length }}
            <span v-if="etaText" class="eta">· ETA {{ etaText }}</span>
          </span>
        </div>

        <!-- 세로 스크롤 리스트 -->
        <div class="qd-list" v-if="items.length">
          <div v-for="(item, i) in items" :key="item.id || i"
            class="q-row"
            :class="{ active: (i === currentIdx && isRunning) || isItemRunning(i), done: item._done, sel: item.id && selectedIds.has(item.id) }"
            @click="onItemClick($event, item, i)"
            :title="(item.prompt || 'No prompt').toString().substring(0, 200)">
            <span class="q-row-st">
              <template v-if="isItemRunning(i) || (i === currentIdx && isRunning && !isPaused)"><Icon name="hourglass" /></template>
              <template v-else-if="i === currentIdx && isPaused"><Icon name="pause" /></template>
              <template v-else-if="item._done"><Icon name="check" /></template>
              <template v-else>{{ i + 1 }}</template>
            </span>
            <span class="q-row-body">
              <span class="q-row-prompt">{{ (item.prompt || '—').toString().slice(0, 90) || '—' }}</span>
              <span class="q-row-neg" v-if="item.negative_prompt">⊘ {{ item.negative_prompt.toString().slice(0, 60) }}</span>
            </span>
            <span class="q-row-tools" @click.stop>
              <button class="qr-btn" @click="moveItem(item, 'up')" :disabled="!canMoveUp(i)" title="위로"><Icon name="chevron-up" /></button>
              <button class="qr-btn" @click="moveItem(item, 'down')" :disabled="!canMoveDown(i)" title="아래로"><Icon name="chevron-down" /></button>
              <button class="qr-btn" @click="openEdit(item, i)" title="편집"><Icon name="pencil" /></button>
              <button class="qr-btn danger" @click="removeItem(item, i)" :disabled="isItemRunning(i)"
                :title="isItemRunning(i) ? '생성 중인 항목은 지울 수 없습니다' : '삭제'"><Icon name="trash" /></button>
            </span>
          </div>
        </div>
        <div class="queue-empty" v-else>
          대기열이 비어있습니다<br>
          <span class="qe-hint">생성 화면에서 프롬프트를 큐에 추가하세요</span>
        </div>
      </aside>
    </transition>

    <!-- 큐 항목 편집 모달 -->
    <div class="qe-overlay" v-if="editItem" @mousedown.self="closeEdit">
      <div class="qe-modal">
        <div class="qe-head">
          <h3>큐{{ editIdx + 1 }} 편집</h3>
          <button class="qe-x" @click="closeEdit"><Icon name="close" /></button>
        </div>
        <label class="qe-label">프롬프트</label>
        <textarea v-model="editPrompt" class="qe-text" rows="6" placeholder="프롬프트..."></textarea>
        <label class="qe-label">네거티브</label>
        <textarea v-model="editNeg" class="qe-text" rows="3" placeholder="네거티브..."></textarea>
        <div class="qe-foot">
          <button class="qe-btn" @click="moveEdit('up')" :disabled="!canMoveUp(editRowIdx)"><Icon name="chevron-up" /> 위로</button>
          <button class="qe-btn" @click="moveEdit('down')" :disabled="!canMoveDown(editRowIdx)"><Icon name="chevron-down" /> 아래로</button>
          <button class="qe-btn danger" @click="deleteEdit" :disabled="isItemRunning(editRowIdx)"
            :title="isItemRunning(editRowIdx) ? '생성 중인 항목은 지울 수 없습니다' : '삭제'"><Icon name="trash" /> 삭제</button>
          <div class="qe-sp"></div>
          <button class="qe-btn" @click="closeEdit">취소</button>
          <button class="qe-btn primary" @click="saveEdit"><Icon name="save" /> 저장</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { onBackendBound, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import {
  ONLY_RUNNING_ROW_NOTICE, canMoveDown as rowCanMoveDown, canMoveUp as rowCanMoveUp, clearConfirmMessage,
  isRowLocked, removableIds,
} from '../utils/queueLocks'

// 자동화가 돌면 자동화가 '큐 우선'으로 대기열을 먼저 처리한다 — 백엔드가 대기열 '시작'을 거절하므로
// 버튼을 미리 끈다(ui/queue_coordination.py). 예전엔 둘이 같은 항목을 보내 서로의 '생성 중' 표시를 풀었다.
const AUTOMATION_OWNS_QUEUE = '자동화가 대기열 항목을 먼저 처리하고 있습니다 — 자동화 중에는 따로 시작하지 않습니다'

interface QueueItem {
  id?: string
  prompt?: string
  negative_prompt?: string
  _done?: boolean
  [k: string]: any
}

const items = ref<QueueItem[]>([])
// 우측 드로어 열림 — 도크(components/dock/QuickDock.vue)가 v-model:open 으로 연다. 안 넘기면 제 상태로 산다.
const drawerOpen = defineModel<boolean>('open', { default: false })
const pinBump = ref(false)      // 항목 추가 시 핀(도크 버튼) 1회 강조 애니메이션
const isRunning = ref(false)
const isPaused = ref(false)
const currentIdx = ref(-1)
// 지금 실제로 생성 중인 항목(대기열 실행 · 자동화 '큐 우선' 모두) — 없으면 -1. 이 행은 지우거나 옮기지 못한다.
const processingIdx = ref(-1)
const completedCount = ref(0)
// 자동화가 도는가 — queueUpdated.automation(당겨 온 현재 상태) · automationStatus.running(이후 변화)
const automationRunning = ref(false)
const selectedIds = ref<Set<string>>(new Set())  // 다중 선택 (Shift+클릭으로 범위 선택)
const _lastClickedIdx = ref(-1)

// 큐 항목 편집 모달
const editItem = ref<QueueItem | null>(null)
const editIdx = ref(-1)
const editPrompt = ref('')
const editNeg = ref('')
function openEdit(item: QueueItem, i: number) {
  editItem.value = item
  editIdx.value = i
  editPrompt.value = (item.prompt || '').toString()
  editNeg.value = (item.negative_prompt || '').toString()
}
function closeEdit() { editItem.value = null; editIdx.value = -1 }
// 편집 중인 항목의 '지금' 위치 — 모달이 열린 사이 대기열이 줄거나 바뀌어도 id 로 다시 찾는다
const editRowIdx = computed(() => {
  const id = editItem.value?.id
  if (!id) return editIdx.value
  const i = items.value.findIndex(it => it.id === id)
  return i >= 0 ? i : -1
})
function saveEdit() {
  if (editItem.value && editItem.value.id) {
    requestAction('update_queue_item', {
      item_id: editItem.value.id,
      prompt: editPrompt.value,
      negative_prompt: editNeg.value,
    })
  }
  closeEdit()
}
function deleteEdit() {
  if (editItem.value && editItem.value.id && !isItemRunning(editRowIdx.value)) {
    requestAction('remove_queue_items', { item_ids: [editItem.value.id] })
  }
  closeEdit()
}
function moveEdit(dir: 'up' | 'down') {
  if (editItem.value && editItem.value.id) {
    requestAction('move_queue_item', { item_id: editItem.value.id, direction: dir })
  }
  closeEdit()
}

// ETA 추적: 처리된 항목당 평균 시간
const _startTime = ref(0)
const _completedAtStart = ref(0)

const progressPct = computed(() => {
  if (items.value.length === 0) return 0
  return Math.round(completedCount.value / items.value.length * 100)
})

const etaText = computed(() => {
  if (!isRunning.value || isPaused.value) return ''
  const done = completedCount.value - _completedAtStart.value
  const remaining = items.value.length - completedCount.value
  if (done < 1 || remaining <= 0 || _startTime.value === 0) return ''
  const elapsed = (Date.now() - _startTime.value) / 1000
  const avgPerItem = elapsed / done
  const sec = Math.round(avgPerItem * remaining)
  if (sec < 60) return `${sec}초`
  if (sec < 3600) return `${Math.floor(sec / 60)}분 ${sec % 60}초`
  return `${Math.floor(sec / 3600)}시 ${Math.floor((sec % 3600) / 60)}분`
})

// 생성 중인 항목 — 삭제·이동 금지(백엔드 QueueModel 도 같은 규칙으로 막는다)
function isItemRunning(i: number) {
  return isRowLocked(i, processingIdx.value)
}
function canMoveUp(i: number) {
  return rowCanMoveUp(i, processingIdx.value)
}
function canMoveDown(i: number) {
  return rowCanMoveDown(i, items.value.length, processingIdx.value)
}

function onItemClick(e: MouseEvent, item: QueueItem, i: number) {
  if (!item.id) return  // id 없으면 선택 불가
  if (e.shiftKey && _lastClickedIdx.value >= 0) {
    // 범위 선택
    const [lo, hi] = [Math.min(_lastClickedIdx.value, i), Math.max(_lastClickedIdx.value, i)]
    const next = new Set(selectedIds.value)
    for (let k = lo; k <= hi; k++) {
      const it = items.value[k]
      if (it?.id) next.add(it.id)
    }
    selectedIds.value = next
  } else if (e.ctrlKey || e.metaKey) {
    // 개별 토글
    const next = new Set(selectedIds.value)
    if (next.has(item.id)) next.delete(item.id)
    else next.add(item.id)
    selectedIds.value = next
    _lastClickedIdx.value = i
  } else {
    // 일반 클릭은 선택 해제 (현재는 아무 동작 없음)
    if (selectedIds.value.size > 0) selectedIds.value = new Set()
    _lastClickedIdx.value = i
  }
}

function moveItem(item: QueueItem, direction: 'up' | 'down') {
  if (!item.id) return
  requestAction('move_queue_item', { item_id: item.id, direction })
}

function removeItem(item: QueueItem, i: number) {
  if (isItemRunning(i)) return   // 생성 중 — 버튼도 꺼져 있다
  if (item.id) {
    requestAction('remove_queue_items', { item_ids: [item.id] })
  } else {
    // fallback: id 없으면 로컬에서만 제거
    items.value.splice(i, 1)
  }
}

function removeSelected() {
  if (selectedIds.value.size === 0) return
  // 생성 중인 항목은 빼고 보낸다(선택은 유지 — 끝나면 다시 지울 수 있다)
  const runningId = items.value[processingIdx.value]?.id
  const ids = removableIds(selectedIds.value, runningId)
  if (ids.length) requestAction('remove_queue_items', { item_ids: ids })
  selectedIds.value = runningId && selectedIds.value.has(runningId) ? new Set([runningId]) : new Set()
}

function clearAll() {
  if (!items.value.length) return
  // 생성 중인 항목은 남는다 — 확인 문구에서 빼고, 그것뿐이면 지울 것이 없다고 알린다
  const message = clearConfirmMessage(items.value.length, processingIdx.value)
  if (message === null) {
    requestAction('show_toast', { type: 'warning', msg: ONLY_RUNNING_ROW_NOTICE })
    return
  }
  if (!confirm(message)) return
  requestAction('clear_queue')
  selectedIds.value = new Set()
}
function startQueue() {
  if (automationRunning.value) return   // 버튼도 꺼져 있다 — 백엔드도 거절한다
  _startTime.value = Date.now()
  _completedAtStart.value = completedCount.value
  requestAction('start_queue')
}
function stopQueue() {
  requestAction('stop_queue')
  _startTime.value = 0
}
function pauseQueue() { requestAction('pause_queue') }
function resumeQueue() { requestAction('resume_queue') }

// 핀 강조 — 항목이 큐에 추가되면 1회 펄스 (드로어 안 열어도 변화 인지)
let _bumpTimer: ReturnType<typeof setTimeout> | null = null
function _bumpPin() {
  pinBump.value = true
  if (_bumpTimer) clearTimeout(_bumpTimer)
  _bumpTimer = setTimeout(() => { pinBump.value = false }, 600)
}
// queueItemAdded가 누락되는 경로(직접 queueUpdated만 오는 경우)도 커버
watch(() => items.value.length, (n, old) => { if (n > old) _bumpPin() })

// 이벤트 disconnect 핸들 — onUnmounted에서 정리 (메모리 누수 방지)
const _unsubs: Array<() => void> = []

// (자동 재시작 기능 제거됨 — 큐는 오직 '▶ 시작' 버튼으로만 시작. 사용자 요청.)

onMounted(() => {
  // Python → Vue: 대기열 상태 실시간 동기화
  _unsubs.push(onBackendEvent('queueUpdated', (json: string) => {
    try {
      const data: any = JSON.parse(json)
      if (Array.isArray(data.items)) {
        items.value = data.items
        // 삭제된 항목의 선택 정리
        if (selectedIds.value.size > 0) {
          const validIds = new Set<string>(data.items.map((it: QueueItem) => it.id).filter(Boolean))
          const next = new Set<string>()
          for (const id of selectedIds.value) if (validIds.has(id)) next.add(id)
          if (next.size !== selectedIds.value.size) selectedIds.value = next
        }
      }
      if (typeof data.running === 'boolean') isRunning.value = data.running
      if (typeof data.paused === 'boolean') isPaused.value = data.paused
      if (typeof data.current_index === 'number') currentIdx.value = data.current_index
      processingIdx.value = typeof data.processing_index === 'number' ? data.processing_index : -1
      if (typeof data.completed === 'number') completedCount.value = data.completed
      if (typeof data.automation === 'boolean') automationRunning.value = data.automation
      // 자동으로 드로어를 열지 않음 — 오버레이가 매번 튀어나오면 거슬림.
      // 대신 항상 보이는 핀의 카운트로 큐 변화를 인지(watch로 핀 강조).
      // 자동시작 없음 — 큐 추가/복원만으로는 절대 생성 시작 안 함. 오직 '▶ 시작' 버튼만.
    } catch {}
  }))

  // 아이템 추가 이벤트 — queueUpdated가 전체 목록(id 포함)을 권위 있게 보내므로
  // 여기선 핀만 1회 강조한다. (예전엔 여기서 push했는데, queueItemAdded payload에 id가 없어
  // 중복 체크를 통과하지 못하고 매번 push되어 '깡통 큐' 중복 항목이 생기던 버그였음.)
  _unsubs.push(onBackendEvent('queueItemAdded', () => { _bumpPin() }))

  // 완료 이벤트
  _unsubs.push(onBackendEvent('queueCompleted', (json: string) => {
    isRunning.value = false
    isPaused.value = false
    try {
      const data: any = JSON.parse(json)
      completedCount.value = data.total || items.value.length
    } catch {}
    _startTime.value = 0
  }))

  // 자동화가 시작·중지되면 '시작' 버튼을 따라 켜고 끈다(자동화 중엔 백엔드가 거절한다)
  _unsubs.push(onBackendEvent('automationStatus', (json: string) => {
    try {
      const data: any = JSON.parse(json)
      if (typeof data.running === 'boolean') automationRunning.value = data.running
    } catch {}
  }))

  // 현재 상태를 요청 — 시작 시 복구된 대기열은 이 패널이 뜨기 전에 알려져 그냥은 안 보인다.
  // 백엔드가 **붙은 뒤에** 보내야 한다: 이 onMounted 는 App 의 onMounted(await initBridge())보다
  // 먼저 돌아, 여기서 바로 보낸 requestAction 은 스토어에 백엔드가 없어 버려졌다. 웹 모드 재접속은
  // 패널을 다시 마운트하지 않으므로 붙을 때마다 다시 당겨 온다. 백엔드는 다음 턴에 queueUpdated 로 답한다.
  _unsubs.push(onBackendBound(() => requestAction('sync_queue_state')))
})

// Esc — 편집 모달 먼저, 없으면 드로어 닫기
function _onEditKey(e: KeyboardEvent) {
  if (e.key !== 'Escape') return
  if (editItem.value) { e.stopPropagation(); closeEdit() }
  else if (drawerOpen.value) { e.stopPropagation(); drawerOpen.value = false }
}
onMounted(() => {
  window.addEventListener('keydown', _onEditKey, true)
})
onUnmounted(() => {
  for (const off of _unsubs) { try { off() } catch {} }
  _unsubs.length = 0
  if (_bumpTimer) { clearTimeout(_bumpTimer); _bumpTimer = null }
  window.removeEventListener('keydown', _onEditKey, true)
})

// 도크 버튼이 핀처럼 보이려고 읽는 것 — 개수 · 실행/일시정지 · 추가 강조
defineExpose({ items, isRunning, isPaused, pinBump })
</script>

<style scoped>
.queue-root { display: contents; }  /* 레이아웃 흐름 차지 안 함 — 드로어는 fixed (핀은 dock/QuickDock.vue) */

/* ── 우측 드로어 ── */
.qd-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 2450; }
.queue-drawer {
  position: fixed; top: 0; right: 0; bottom: 0; width: min(440px, 94vw);
  z-index: 2500; background: var(--bg-secondary); border-left: 1px solid var(--border);
  display: flex; flex-direction: column; box-shadow: -10px 0 40px rgba(0,0,0,0.55);
}
.qd-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--border); }
.title { color: var(--text-secondary); font-size: 12px; font-weight: var(--fw-bold); letter-spacing: 0; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.count-badge { background: var(--accent-fill); color: var(--on-accent); padding: 1px 6px; border-radius: 8px; font-size: var(--fs-label); font-weight: var(--fw-bold); }
.running-badge { background: var(--state-ok-fg); color: var(--bg-primary); padding: 1px 6px; border-radius: 8px; font-size: var(--fs-label); font-weight: var(--fw-bold); animation: pulse 1.5s infinite; }
.paused-badge { background: var(--state-warn-fg); color: var(--bg-primary); padding: 1px 6px; border-radius: 8px; font-size: var(--fs-label); font-weight: var(--fw-bold); }
.selected-badge { background: var(--state-info-fg); color: var(--bg-primary); padding: 1px 6px; border-radius: 8px; font-size: var(--fs-label); font-weight: var(--fw-bold); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.qd-x { background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); width: 30px; height: 30px; cursor: pointer; font-size: 13px; }
.qd-x:hover { color: var(--text-primary); border-color: var(--accent); }

.qd-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; padding: 10px 16px; border-bottom: 1px solid var(--border); }
.btn { padding: 5px 12px; background: var(--bg-button); border: none; border-radius: 5px; color: var(--text-secondary); font-size: 11px; cursor: pointer; font-weight: var(--fw-bold); }
.btn:hover:not(:disabled) { background: var(--bg-button-hover); color: var(--text-primary); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn.danger { color: var(--state-alert-fg); }
.btn.primary { background: var(--accent-fill); color: var(--on-accent); }
.btn.primary:hover { background: var(--accent-fill-hover); }

/* ── 세로 리스트 (찌그러짐 없음) ── */
.qd-list { flex: 1; overflow-y: auto; padding: 6px 8px; }
.q-row {
  display: flex; align-items: center; gap: 10px; padding: 9px 10px; margin: 3px 0;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
  cursor: pointer; transition: 0.12s;
}
.q-row:hover { background: var(--bg-button); border-color: var(--border); }
.q-row.active { border-color: var(--accent); background: var(--accent-dim); }
.q-row.sel { border-color: var(--state-info-fg); background: rgba(96,165,250,0.08); }
.q-row.done { opacity: 0.45; }
.q-row-st { flex-shrink: 0; width: 22px; text-align: center; color: var(--accent); font-weight: var(--fw-bold); font-size: 12px; }
.q-row-body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.q-row-prompt { color: var(--text-primary); font-size: 12px; line-height: 1.4; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
/* 원래 8자리 hex(#f8717188 = 알파 53%)였다. 토큰에는 알파가 없어 색만 토큰으로 두고
   투명도는 color-mix 로 유지한다 */
.q-row-neg { color: color-mix(in srgb, var(--state-alert-fg) 53%, transparent); font-size: var(--fs-label); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.q-row-tools { flex-shrink: 0; display: flex; gap: 3px; opacity: 0; transition: 0.12s; }
.q-row:hover .q-row-tools { opacity: 1; }
.qr-btn { width: 24px; height: 24px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 5px; color: var(--text-muted); font-size: 11px; cursor: pointer; padding: 0; }
.qr-btn:hover:not(:disabled) { color: var(--text-primary); border-color: var(--accent); }
.qr-btn.danger:hover:not(:disabled) { color: var(--state-alert-fg); border-color: var(--state-alert-fg); }
.qr-btn:disabled { opacity: 0.3; cursor: not-allowed; }

/* 드로어 슬라이드/백드롭 트랜지션 */
.qd-slide-enter-active, .qd-slide-leave-active { transition: transform 0.22s ease; }
.qd-slide-enter-from, .qd-slide-leave-to { transform: translateX(100%); }
.qd-fade-enter-active, .qd-fade-leave-active { transition: opacity 0.22s ease; }
.qd-fade-enter-from, .qd-fade-leave-to { opacity: 0; }
.qe-hint { color: var(--text-muted); font-size: var(--fs-label); }

/* 큐 항목 편집 모달 */
.qe-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 3000; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(3px); }
.qe-modal { width: min(620px, 92vw); max-height: 86vh; overflow-y: auto; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.6); }
.qe-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.qe-head h3 { font-size: 16px; font-weight: var(--fw-bold); color: var(--text-primary); }
.qe-x { background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); width: 28px; height: 28px; cursor: pointer; }
.qe-label { display: block; font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin: 8px 0 4px; }
.qe-text { width: 100%; background: var(--bg-input); border: 1px solid var(--border); border-radius: 7px; padding: 9px 11px; color: var(--text-primary); font-size: 12px; line-height: 1.5; resize: vertical; }
.qe-text:focus { outline: none; border-color: var(--accent); }
.qe-foot { display: flex; align-items: center; gap: 6px; margin-top: 14px; }
.qe-sp { flex: 1; }
.qe-btn { background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: 11px; font-weight: var(--fw-bold); padding: 7px 12px; cursor: pointer; }
.qe-btn:hover:not(:disabled) { color: var(--text-primary); border-color: var(--accent); }
.qe-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.qe-btn.danger { color: var(--state-alert-fg); }
.qe-btn.primary { background: var(--accent-fill); color: var(--on-accent); border-color: var(--accent-fill); }

.queue-progress { padding: 4px 16px 8px; }
.progress-bar { width: 100%; height: 3px; background: var(--bg-input); border-radius: 2px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
.progress-text { font-size: var(--fs-label); color: var(--text-muted); text-align: right; margin-top: 2px; display: block; }
.progress-text .eta { color: var(--accent); font-weight: var(--fw-bold); margin-left: 4px; }

.queue-empty { padding: 12px; text-align: center; color: var(--text-muted); font-size: 11px; }
</style>
