<template>
  <div class="ap">
    <!-- ══════ 멈춰 있을 때 — 설정 ══════ -->
    <template v-if="!running">
      <!-- 얼마나 — 칩은 자주 쓰는 값의 지름길일 뿐, 아래 '직접' 과 같은 mode/limit
           한 벌을 가리킨다. 그래서 칩을 누르면 숫자가 따라오고, 숫자를 치면
           어느 칩과도 안 맞게 되어 선택이 저절로 풀린다(별도 상태가 없다). -->
      <div class="ap-row">
        <span class="ap-lab">얼마나</span>
        <button v-for="c in HOW_MANY" :key="c.key" type="button"
          class="ap-chip" :class="{ on: activeChip === c.key }"
          @click="pickChip(c)">{{ c.label }}</button>
      </div>

      <div class="ap-row">
        <span class="ap-lab" title="칩에 없는 값은 여기 직접 친다 — 단위는 장 · 분 · 시간">직접</span>
        <input class="ap-num" type="number" min="1" inputmode="numeric"
          :class="{ idle: settings.mode === 'unlimited' }"
          :title="settings.mode === 'unlimited' ? '무제한이라 지금은 안 쓰인다 — 숫자를 치면 무제한이 풀린다' : ''"
          :value="amountValue"
          @input="onAmount(($event.target as HTMLInputElement).value)" />
        <div class="ap-unit">
          <CustomSelect :modelValue="amountUnit" :options="UNITS" @update:modelValue="onUnit" />
        </div>
        <span class="ap-div"></span>
        <span class="ap-lab" title="한 장을 끝내고 다음 장을 시작하기까지 쉬는 시간">간격</span>
        <input class="ap-num narrow" type="number" min="0" step="0.5"
          :value="settings.delay"
          @input="patch({ delay: Math.max(0, toNum(($event.target as HTMLInputElement).value, 1)) })" />
        <span class="ap-suffix">초</span>
      </div>

      <!-- 덱은 한 번만 그린다. 예전엔 실행 전(.auto-deck-pre)·실행 중(.deck-status)이
           거의 같은 내용을 서로 다른 모양으로 두 번 그렸다. -->
      <div class="ap-deck" v-if="deckTotal > 0">
        <Icon name="cards" />
        <span v-if="deckAllowDup">덱 <b>{{ deckTotal }}</b>개 · 중복 허용</span>
        <span v-else>덱 <b>{{ deckRemaining }}</b> / {{ deckTotal }} 남음 · {{ deckUsed }}개 사용</span>
        <span class="ap-deck-tail" v-if="settings.autoResetDeck">비우면 다시 채움</span>
      </div>

      <!-- 고급 — 접힌 줄에 현재 값을 요약해 둔다. 열지 않아도 무엇이 걸려 있는지 보인다. -->
      <button type="button" class="ap-adv-head" :class="{ open: advOpen }"
        :aria-expanded="advOpen" @click="advOpen = !advOpen">
        <Icon :name="advOpen ? 'chevron-down' : 'chevron-right'" />
        <span>고급</span>
        <span class="ap-adv-sum">{{ advSummary }}</span>
      </button>

      <div class="ap-adv" v-if="advOpen">
        <!-- 설명을 title 밖으로 꺼낸다. 라벨만 보고는 뜻을 알 수 없는 항목들이라
             한 줄 설명을 항상 보이게 두고, 길어지는 나머지만 title 로 남긴다. -->
        <div class="ap-adv-item">
          <div class="ap-adv-line">
            <span class="ap-adv-lab">반복</span>
            <input class="ap-num narrow" type="number" min="1" max="100" :value="settings.repeat"
              @input="patch({ repeat: Math.max(1, toNum(($event.target as HTMLInputElement).value, 1)) })" />
            <span class="ap-suffix">장</span>
          </div>
          <p class="ap-why">한 프롬프트로 몇 장을 이어서 만들지. 프롬프트는 그대로, 시드만 바뀐다.</p>
        </div>

        <div class="ap-adv-item">
          <div class="ap-adv-line">
            <span class="ap-adv-lab">재시도</span>
            <input class="ap-num narrow" type="number" min="0" max="10" :value="settings.maxRetries"
              @input="patch({ maxRetries: Math.max(0, toNum(($event.target as HTMLInputElement).value, 2)) })" />
            <span class="ap-suffix">회</span>
          </div>
          <p class="ap-why" title="반복과 다르다 — 반복은 같은 프롬프트로 N장, 재시도는 실패 1회당 N번까지 다시. 2초 → 4초 → 8초(최대 30초)로 쉬어 가며 시도하고, 다 실패하면 그 장은 포기하고 다음으로 넘어간다.">
            서버 에러 · 타임아웃 · 메모리 부족으로 실패했을 때 몇 번까지 다시 해볼지.
          </p>
        </div>

        <div class="ap-adv-item">
          <div class="ap-adv-line">
            <span class="ap-adv-lab">대기열 VRAM 정리 주기</span>
            <input class="ap-num narrow" type="number" min="0" max="100" :value="settings.cleanupEveryN"
              @input="patch({ cleanupEveryN: Math.max(0, toNum(($event.target as HTMLInputElement).value, 0)) })" />
            <span class="ap-suffix">장</span>
          </div>
          <p class="ap-why" title="Forge 는 체크포인트 언로드(unload-checkpoint), ComfyUI 는 /free 로 VRAM 을 비운다. 다음 장이 모델을 다시 올리므로 정리할 때마다 모델 로딩 시간이 한 번 더 든다. 대기열의 마지막 장 뒤에는 정리하지 않는다 — 끝난 뒤 비우려면 설정의 '생성 후 모델 언로드'를 켠다.">
            대기열을 돌릴 때 몇 장마다 모델을 VRAM 에서 내렸다 다시 올릴지 — 긴 대기열의 메모리 부족을 막는다. 0 이면 끈다.
          </p>
        </div>

        <div class="ap-adv-item">
          <div class="ap-adv-line">
            <span class="ap-adv-lab">중복 허용</span>
            <ToggleSwitch :modelValue="settings.allowDupes" size="sm"
              @update:modelValue="patch({ allowDupes: $event })" />
          </div>
          <p class="ap-why">덱에서 이미 뽑은 프롬프트를 다시 뽑는다. 끄면 한 바퀴 안에서는 안 겹친다.</p>
        </div>

        <div class="ap-adv-item">
          <div class="ap-adv-line">
            <span class="ap-adv-lab">덱 소진 시 초기화</span>
            <ToggleSwitch :modelValue="settings.autoResetDeck" size="sm"
              @update:modelValue="patch({ autoResetDeck: $event })" />
          </div>
          <p class="ap-why">덱을 다 쓰면 가득 채우고 다시 섞어 계속한다 — 모두 한 번씩 뽑은 뒤라 공평하다.</p>
        </div>

        <div class="ap-adv-item" v-if="deckTotal > 0">
          <div class="ap-adv-line">
            <span class="ap-adv-lab">지금 덱</span>
            <button type="button" class="ap-mini" @click="emit('reset-deck')">
              <Icon name="rotate-cw" /> 덱 초기화
            </button>
          </div>
          <p class="ap-why">기다리지 않고 지금 바로 가득 채운다 (사용 0 으로).</p>
        </div>
      </div>
    </template>

    <!-- ══════ 돌고 있을 때 — 조종석 ══════ -->
    <template v-else>
      <div class="ap-head">
        <span class="ap-dot" :class="{ held: paused }"></span>
        <b class="ap-big">{{ count }}</b>
        <span class="ap-head-unit">{{ settings.mode === 'count' ? `/ ${settings.limit} 장` : '장' }}</span>
        <span class="ap-head-tail">{{ paused ? '멈춰 있음' : headTail }}</span>
      </div>

      <div class="ap-bar" v-if="showBar">
        <div class="ap-bar-fill" :style="{ width: progressPct + '%' }"></div>
      </div>

      <!-- ★ 다음에 나갈 프롬프트. 덱·와일드카드·조건식이 매번 바꾸는데도
           지금까지는 보이지 않았다 — 이 화면의 존재 이유다. -->
      <!-- promptIsNext=false: 보이는 건 지금 장(생성 중이거나 방금 생성한) 프롬프트이고 다음 장은
           새로 뽑는다 — 여기 건 편집은 뽑는 순간 버려지므로 받지 않는다. 멈추면 다음 프롬프트를
           뽑아 보여 준 뒤 서므로 그때 손본다(core/automation_prompt_state). -->
      <div class="ap-next" :class="{ locked: !promptIsNext }">
        <div class="ap-next-head">
          <span class="ap-next-title">{{ promptIsNext ? '다음 프롬프트' : '지금 장 프롬프트' }}</span>
          <!-- 보이는 프롬프트는 덱에서 이미 뽑은 deckUsed 번째다 -->
          <span class="ap-next-deck" v-if="deckTotal > 0 && deckUsed > 0 && !deckAllowDup">덱 #{{ deckUsed }}</span>
          <span class="ap-next-hint">{{ !promptIsNext ? '다음 장은 새로 뽑는다' : edits ? '이번 한 장에만' : '누르면 빠짐' }}</span>
        </div>

        <div class="ap-tags" v-if="baseTags.length || added.length">
          <button v-for="(t, i) in baseTags" :key="`b${i}`" type="button"
            class="ap-tag" :class="[`k-${tagKind(t)}`, { off: isRemoved(i) }]"
            :disabled="!promptIsNext"
            :title="!promptIsNext ? '지금 장 프롬프트는 고칠 수 없다' : isRemoved(i) ? '되돌린다' : '이번 한 장에서 뺀다'"
            @click="toggleTag(i)">{{ t }}</button>
          <button v-for="(t, i) in added" :key="`a${i}`" type="button"
            class="ap-tag added" title="내가 더한 태그 — 누르면 지운다"
            :disabled="!promptIsNext"
            @click="dropAdded(i)">{{ t }}<Icon name="close" size="0.85em" /></button>
        </div>
        <div class="ap-next-empty" v-else>백엔드가 아직 다음 프롬프트를 보내지 않았다</div>

        <div class="ap-add" v-if="promptIsNext">
          <Icon name="plus" />
          <input class="ap-add-input" v-model="draft" placeholder="태그를 치고 Enter"
            @keydown.enter="onDraftEnter" />
        </div>

        <p class="ap-once" v-if="missedEdit">편집을 보내기 전에 이 장 생성이 시작돼 <b>들어가지 않았다</b> — 끝나면 다음 프롬프트를 뽑는다</p>
        <p class="ap-once" v-else-if="!promptIsNext">이 장이 끝나면 다음 프롬프트를 뽑는다 — 그 전에 손보려면 <b>일시정지</b></p>
        <p class="ap-once" v-else-if="edits">이 편집은 <b>다음 한 장</b>에만 — 그 뒤 사라진다</p>
      </div>

      <div class="ap-deck" v-if="deckTotal > 0">
        <Icon name="cards" />
        <span v-if="deckAllowDup">덱 <b>{{ deckTotal }}</b>개 · 중복 허용</span>
        <span v-else>덱 <b>{{ deckRemaining }}</b> / {{ deckTotal }} 남음</span>
        <span class="ap-deck-tail" v-if="waiting && waitTotalMs > 0">다음까지 {{ waitSec }}초</span>
      </div>

      <div class="ap-ctl">
        <button type="button" class="ap-btn" @click="paused ? emit('resume') : emit('pause')">
          <Icon :name="paused ? 'play' : 'pause'" /> {{ paused ? '재개' : '일시정지' }}
        </button>
        <button type="button" class="ap-btn stop" @click="emit('stop')">
          <Icon name="stop" /> 멈추기
        </button>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * 자동화 패널 — 한 자리를 두 모드가 번갈아 쓴다.
 *
 * 멈춰 있을 땐 **설정**(얼마나 · 간격 · 덱 · 고급), 돌기 시작하면 같은 자리가 통째로
 * **조종석**(진행 · 다음 프롬프트 · 일시정지/멈추기)으로 바뀐다.
 *
 * 왜 별도 컴포넌트인가: App.vue 의 `.gen-footer` 안에서 설정·상태·덱이 서로 다른
 * 조건으로 겹쳐 그려지고 있었다(덱은 실행 전·중 두 번). 한 파일에 모아 두 모드를
 * 배타적으로 만들면 '무엇이 언제 보이는지' 가 마크업에서 바로 읽힌다.
 *
 * 편집 결과를 **전문으로** 보내는 이유: 추가/제거를 따로 추적해 백엔드에서 다시
 * 합치면 양쪽 규칙이 갈라진다. 화면이 만든 문자열을 그대로 넘기면 어긋날 곳이 없다.
 */
import { computed, onUnmounted, reactive, ref, watch } from 'vue'
import CustomSelect from './CustomSelect.vue'
import ToggleSwitch from './ToggleSwitch.vue'
import { getBackend } from '../bridge.js'
import { composeEditedPrompt, isOverrideEcho, splitPrompt } from '../utils/automationPromptEdit'
import { isImeComposing } from '../utils/imeComposition'
import type { AutomationSettings } from '../types/bridge'

const props = withDefaults(defineProps<{
  settings: AutomationSettings
  /** 자동화 루프가 도는 중인가 — 이 값 하나가 설정/조종석을 가른다. */
  running: boolean
  paused: boolean
  count: number
  waiting: boolean
  waitRemainingMs: number
  waitTotalMs: number
  deckTotal: number
  deckRemaining: number
  deckUsed: number
  deckAllowDup: boolean
  /** automationStatus.prompt — 다음 생성에 나갈 프롬프트 전문. 없으면 빈 문자열. */
  nextPrompt: string
  /** automationStatus.prompt_is_next — false 면 nextPrompt 는 지금 장 프롬프트라 편집을 받지 않는다. */
  promptIsNext?: boolean
}>(), { promptIsNext: true })

const emit = defineEmits<{
  /** 바뀐 항목만 담은 조각. 부모가 자기 reactive 에 합쳐 백엔드로 보낸다. */
  'update:settings': [patch: Partial<AutomationSettings>]
  'reset-deck': []
  pause: []
  resume: []
  stop: []
  /** 다음 한 장에만 쓸 프롬프트 전문. 빈 문자열이면 덮어쓰기 취소. */
  override: [prompt: string]
}>()

function patch(p: Partial<AutomationSettings>) { emit('update:settings', p) }
function toNum(v: string, fallback: number) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

// ── 얼마나 ────────────────────────────────────────────────────────────────
// 백엔드 계약(mode/limit)은 그대로다. 화면만 '장 · 분 · 시간' 으로 말한다.
type HowMany = { key: string; label: string; mode: AutomationSettings['mode']; limit: number }
const HOW_MANY: HowMany[] = [
  { key: 'c10', label: '10장', mode: 'count', limit: 10 },
  { key: 'c50', label: '50장', mode: 'count', limit: 50 },
  { key: 't60', label: '1시간', mode: 'timer', limit: 60 },
  { key: 'inf', label: '무제한', mode: 'unlimited', limit: 0 },
]
const UNITS = ['장', '분', '시간']

const activeChip = computed(() => {
  const s = props.settings
  if (s.mode === 'unlimited') return 'inf'
  if (s.mode === 'count' && s.limit === 10) return 'c10'
  if (s.mode === 'count' && s.limit === 50) return 'c50'
  if (s.mode === 'timer' && s.limit === 60) return 't60'
  return ''
})

function pickChip(c: HowMany) {
  // 무제한은 limit 을 안 건드린다 — 되돌아왔을 때 직전에 치던 숫자가 남아 있어야 한다.
  patch(c.mode === 'unlimited' ? { mode: 'unlimited' } : { mode: c.mode, limit: c.limit })
}

// 무제한일 땐 limit 이 뜻을 잃는다. 그때 단위를 '장'으로 되돌리면 무제한을 풀자마자
// '3시간'이 '180장'으로 바뀌어 버린다 — 직전 단위를 들고 있어야 한다.
const lastUnit = ref('장')
// timer 의 limit 은 분이다. 60의 배수면 '시간' 으로 읽어 준다(1시간 칩과 짝이 맞는다).
const amountUnit = computed(() => {
  const s = props.settings
  if (s.mode === 'unlimited') return lastUnit.value
  if (s.mode !== 'timer') return '장'
  return s.limit >= 60 && s.limit % 60 === 0 ? '시간' : '분'
})
watch(amountUnit, (u) => { if (props.settings.mode !== 'unlimited') lastUnit.value = u }, { immediate: true })
const amountValue = computed(() => {
  const s = props.settings
  return amountUnit.value === '시간' ? s.limit / 60 : s.limit
})

function applyAmount(n: number, unit: string) {
  const v = Math.max(1, Math.round(n))
  if (unit === '장') patch({ mode: 'count', limit: v })
  else patch({ mode: 'timer', limit: unit === '시간' ? v * 60 : v })
}
// 숫자를 치면 mode 가 count/timer 로 정해지므로 '무제한' 칩도 함께 풀린다.
function onAmount(raw: string) { applyAmount(toNum(raw, 1), amountUnit.value) }
function onUnit(u: unknown) { applyAmount(amountValue.value, String(u)) }

// ── 고급 ──────────────────────────────────────────────────────────────────
const advOpen = ref(false)
const advSummary = computed(() => {
  const s = props.settings
  const parts = [`반복 ${s.repeat}`, `재시도 ${s.maxRetries}`, `중복 ${s.allowDupes ? '켬' : '끔'}`]
  if (s.cleanupEveryN > 0) parts.push(`정리 ${s.cleanupEveryN}장마다`)
  return parts.join(' · ')
})

// ── 진행 · 남은 시간 ──────────────────────────────────────────────────────
// 백엔드는 경과 시간을 안 보낸다. 시작 시각을 여기서 잡아 재는 편이 payload 를
// 늘리는 것보다 싸고, 일시정지 동안 시계를 멈추는 것도 화면 쪽 사정이다.
const nowMs = ref(Date.now())
let ticker: ReturnType<typeof setInterval> | undefined
let startedAt = 0
let pausedAt = 0
let heldMs = 0

const elapsedMs = computed(() => {
  if (!startedAt) return 0
  const paused = props.paused && pausedAt ? nowMs.value - pausedAt : 0
  return Math.max(0, nowMs.value - startedAt - heldMs - paused)
})

const goal = computed(() => Math.max(1, Number(props.settings.limit) || 1))
const clampPct = (v: number) => Math.max(0, Math.min(100, v))
const deckPct = computed(() =>
  props.deckTotal > 0 && !props.deckAllowDup ? clampPct(props.deckUsed / props.deckTotal * 100) : 0)

const showBar = computed(() => props.settings.mode !== 'unlimited' || deckPct.value > 0)
const progressPct = computed(() => {
  const s = props.settings
  if (s.mode === 'count') return clampPct(props.count / goal.value * 100)
  if (s.mode === 'timer') return clampPct(elapsedMs.value / (goal.value * 60000) * 100)
  return deckPct.value
})

function fmtDur(ms: number) {
  const sec = Math.max(0, Math.round(ms / 1000))
  if (sec < 60) return `${sec}초`
  const min = Math.round(sec / 60)
  if (min < 60) return `${min}분`
  return `${Math.floor(min / 60)}시간 ${min % 60}분`
}

const headTail = computed(() => {
  const s = props.settings
  if (s.mode === 'timer') {
    const left = goal.value * 60000 - elapsedMs.value
    return left > 0 ? `${fmtDur(left)} 남음` : '곧 끝남'
  }
  // 장 수 모드의 남은 시간은 추정이다 — 지금까지의 장당 평균으로만 말할 수 있다.
  if (s.mode === 'count' && props.count > 0 && props.count < goal.value) {
    const per = elapsedMs.value / props.count
    return `약 ${fmtDur(per * (goal.value - props.count))} 남음`
  }
  return `${fmtDur(elapsedMs.value)} 경과`
})

const waitSec = computed(() => (props.waitRemainingMs / 1000).toFixed(1))

// ── 다음 프롬프트 편집 ────────────────────────────────────────────────────
// 편집의 기준(basePrompt)은 nextPrompt 를 그대로 따라가지 않는다. 덮어쓰기를 보내면 백엔드가
// 그 편집본을 nextPrompt 로 되돌려 주는데, 그걸 기준으로 삼으면 뺀 인덱스·더한 태그가 한 번 더
// 적용돼 고르지 않은 프롬프트가 나갔다('a, b, c' 에서 b 빼고 d 더하면 'a, d, d').
// 기준은 메아리가 아닌 새 프롬프트가 올 때(와 resetEdits)만 바꾼다 — utils/automationPromptEdit.
const basePrompt = ref(props.nextPrompt || '')
const baseTags = computed(() => splitPrompt(basePrompt.value))
const removed = ref<number[]>([])
const added = ref<string[]>([])
const draft = ref('')
// 보내기 전에(모으는 400ms 동안) 그 장 생성이 시작돼 버려진 편집이 있었는가 — 다음 프롬프트가 오면 지운다
const missedEdit = ref(false)
const edits = computed(() => removed.value.length + added.value.length > 0)
const isRemoved = (i: number) => removed.value.includes(i)

const finalPrompt = computed(() => composeEditedPrompt(baseTags.value, removed.value, added.value))

// 글자마다 보내면 백엔드가 매번 덮어쓰기를 다시 세운다 — 손이 멈춘 뒤 한 번만 보낸다.
let sendTimer: ReturnType<typeof setTimeout> | undefined
const lastSent = ref('')
function scheduleOverride() {
  if (sendTimer) clearTimeout(sendTimer)
  sendTimer = setTimeout(() => {
    sendTimer = undefined
    const text = edits.value ? finalPrompt.value : ''
    lastSent.value = text
    emit('override', text)
  }, 400)
}
function resetEdits() {
  if (sendTimer) { clearTimeout(sendTimer); sendTimer = undefined }
  removed.value = []
  added.value = []
  draft.value = ''
  lastSent.value = ''
  missedEdit.value = false
  // 편집이 사라지면 기준도 지금 백엔드가 보여 주는 프롬프트로 맞춘다(장 넘김·멈춤 경로 포함)
  basePrompt.value = props.nextPrompt || ''
}

const keptCount = computed(() =>
  baseTags.value.filter((_, i) => !isRemoved(i)).length + added.value.length)

// 지금 장 프롬프트(promptIsNext=false)는 고치지 않는다 — 다음 장은 새로 뽑아 그 편집이 버려진다.
function toggleTag(i: number) {
  if (!props.promptIsNext) return
  // 전부 빼면 보낼 문자열이 빈다. 빈 문자열은 계약상 '덮어쓰기 취소' 라, 취소선은
  // 그어졌는데 원래 프롬프트가 나가는 앞뒤 안 맞는 상태가 된다 — 마지막 하나는 남긴다.
  if (!isRemoved(i) && keptCount.value <= 1) return
  removed.value = isRemoved(i) ? removed.value.filter(x => x !== i) : [...removed.value, i]
  scheduleOverride()
}
/** 태그 입력 Enter — IME 조합 확정 Enter 는 추가가 아니다(마지막 음절이 잘린 태그가 들어간다) */
function onDraftEnter(e: KeyboardEvent) {
  if (isImeComposing(e)) return
  e.preventDefault()
  addTag()
}
function addTag() {
  if (!props.promptIsNext) return
  const t = draft.value.trim().replace(/,+$/, '').trim()
  if (!t) return
  added.value = [...added.value, t]
  draft.value = ''
  scheduleOverride()
}
function dropAdded(i: number) {
  if (!props.promptIsNext) return
  if (keptCount.value <= 1) return
  added.value = added.value.filter((_, k) => k !== i)
  scheduleOverride()
}

// ── 감시 ──────────────────────────────────────────────────────────────────
// watch 를 여기 모아 두는 이유: 이 콜백들이 resetEdits/sendTimer 를 건드리는데
// `{ immediate: true }` 는 setup 도중 바로 한 번 돈다. 선언보다 위에 두면
// TDZ 로 터진다(실제로 한 번 터뜨렸다).
watch(() => props.running, (on) => {
  if (on) {
    startedAt = Date.now()
    heldMs = 0
    pausedAt = props.paused ? startedAt : 0
    nowMs.value = startedAt
    ticker = setInterval(() => { nowMs.value = Date.now() }, 1000)
  } else {
    if (ticker) { clearInterval(ticker); ticker = undefined }
    // 백엔드는 덮어쓰기를 '쓰고 나서' 지운다. 안 쓰인 채 멈추면 그게 남아
    // 다음 실행의 첫 장을 몰래 바꾼다 — 여기서 명시적으로 취소한다.
    if (edits.value || lastSent.value) emit('override', '')
    resetEdits()
  }
}, { immediate: true })

watch(() => props.paused, (held) => {
  if (held) pausedAt = Date.now()
  else if (pausedAt) { heldMs += Date.now() - pausedAt; pausedAt = 0 }
})

// 다음 장이 나가면 편집은 사라진다 — 백엔드도 덮어쓰기를 한 번 쓰고 버린다.
// 단 반복(repeat≥2) 중 생성 도중에 건 편집은 아직 안 쓰였다: 장이 끝나 count 가 올라도 백엔드는
// 그 덮어쓰기를 그대로 되돌려 준다(메아리). 그때 초기화하면 취소선이 사라지고 기준이 편집본이
// 되어, 태그 하나를 껐다 켜면 '' 가 나가 덮어쓰기가 취소되고 뺀 태그가 몰래 돌아왔다.
// 덮어쓰기를 쓴 장이 끝나면 백엔드가 원래 프롬프트를 보내므로 그때 초기화된다.
watch(() => props.count, () => { if (!isOverrideEcho(props.nextPrompt || '', lastSent.value)) resetEdits() })
// 편집을 보내기 전에(400ms 모으는 중) 그 장 생성이 시작돼 보이는 프롬프트가 '지금 장'이 됐다 —
// 다음 장은 새로 뽑으므로 이 편집은 쓸 곳이 없다. 보내지 않고 버리되 그 사실을 패널에 남긴다.
watch(() => props.promptIsNext, (next) => {
  if (next || !sendTimer) return
  resetEdits()
  missedEdit.value = true
})
// 프롬프트가 바뀌면 편집의 기준이 달라졌다는 뜻이다. 단, 백엔드가 우리가 보낸
// 덮어쓰기를 그대로 되돌려준 경우는 **같은 장**이라 취소선도 기준도 그대로 둔다.
watch(() => props.nextPrompt, (v) => { if (!isOverrideEcho(v || '', lastSent.value)) resetEdits() })

onUnmounted(() => {
  if (ticker) clearInterval(ticker)
  if (sendTimer) clearTimeout(sendTimer)
})

// ── 태그 색 ───────────────────────────────────────────────────────────────
// 분류는 파이썬 TagClassifier 가 한다(tags_db 기반). 모르는 태그는 색을 지어내지
// 않고 중립으로 둔다 — 틀린 색은 없는 색보다 나쁘다.
const CATEGORY_TO_TAG: Record<string, string> = {
  sexual: 'nsfw',
  body_parts: 'person', character_trait: 'person', character: 'person', copyright: 'person',
  clothing: 'wear',
  pose: 'pose', expression: 'pose',
  background: 'scene', composition: 'scene',
  effect: 'fx', art_style: 'fx', color: 'fx',
}
/** 인물 수는 물어볼 것 없이 여기서 판단한다(SearchView 와 같은 규칙). */
const COUNT_RE = /^(\d+)?(girl|boy|other)s?$|^solo$|^multiple_/
const catCache = reactive<Record<string, string>>({})

function normalize(tag: string) { return tag.trim().toLowerCase().replace(/ /g, '_') }
function tagKind(tag: string) {
  const t = normalize(tag)
  if (!t || t.startsWith('<')) return 'neutral'   // <lora:...> 는 태그가 아니다
  if (COUNT_RE.test(t)) return 'person'
  return CATEGORY_TO_TAG[catCache[t]] || 'neutral'
}

async function classify(tags: string[]) {
  const want = [...new Set(tags.map(normalize))].filter(t => t && !t.startsWith('<') && !(t in catCache))
  if (!want.length) return
  const backend: any = await getBackend()
  if (!backend?.classifyTags) return
  backend.classifyTags(JSON.stringify(want), (json: string) => {
    try {
      const result = JSON.parse(json)
      if (!result.error) {
        for (const [k, v] of Object.entries(result)) catCache[k] = String(v)
      }
    } catch { /* 색이 없을 뿐이라 조용히 넘어간다 */ }
    // 답이 안 온 태그도 표시해 둔다 — 안 그러면 매 프레임 다시 묻는다.
    for (const t of want) if (!(t in catCache)) catCache[t] = 'general'
  })
}
watch(baseTags, (tags) => { if (props.running && tags.length) classify(tags) }, { immediate: true })
</script>

<style scoped>
.ap {
  display: flex; flex-direction: column; gap: var(--sp-1);
  padding: var(--sp-2);
  background: color-mix(in srgb, var(--accent) 3%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 12%, transparent);
  border-radius: var(--radius-base);
}

/* ── 공통 행 ── */
.ap-row { display: flex; align-items: center; gap: var(--sp-1); }
.ap-lab {
  flex-shrink: 0; font-size: var(--fs-label); font-weight: var(--fw-medium);
  color: var(--text-muted); letter-spacing: 0;
}
.ap-div { width: 1px; height: 16px; background: var(--rule); margin: 0 var(--sp-1); flex-shrink: 0; }
.ap-suffix { font-size: var(--fs-label); color: var(--text-muted); flex-shrink: 0; }

/* 최소 28px — 25px 짜리 칸은 손으로 집기 어렵고 이 저장소의 규칙에도 미달이었다. */
.ap-num {
  width: 56px; min-width: 0; height: 28px; padding: 0 var(--sp-1);
  font-size: var(--fs-meta); text-align: center;
  font-variant-numeric: tabular-nums;
}
.ap-num.narrow { width: 48px; }
.ap-num.idle { color: var(--text-muted); }
.ap-unit { width: 72px; flex-shrink: 0; }
.ap-unit :deep(.csel-display) { height: 28px; padding: 0 var(--sp-2); font-size: var(--fs-meta); }
.ap-unit :deep(.csel-option) { padding: var(--sp-1) var(--sp-2); font-size: var(--fs-meta); }

/* ── 얼마나 칩 ── */
.ap-chip {
  flex: 1; min-width: 0; height: 28px; padding: 0 var(--sp-1);
  background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-secondary);
  font-size: var(--fs-label); font-weight: var(--fw-medium);
  cursor: pointer; transition: var(--transition);
}
.ap-chip:hover { border-color: var(--edge); color: var(--text-primary); }
.ap-chip.on {
  background: var(--accent-dim); border-color: var(--accent);
  color: var(--accent); font-weight: var(--fw-bold);
}

/* ── 덱 (한 번만 그린다) ── */
.ap-deck {
  display: flex; align-items: center; gap: var(--sp-1);
  font-size: var(--fs-label); color: var(--text-muted);
  padding-top: var(--sp-1); border-top: 1px solid var(--rule);
}
.ap-deck b { color: var(--accent); font-weight: var(--fw-bold); font-variant-numeric: tabular-nums; }
.ap-deck-tail { margin-left: auto; color: var(--text-muted); }

/* ── 고급 ── */
.ap-adv-head {
  display: flex; align-items: center; gap: var(--sp-1);
  height: 28px; padding: 0; width: 100%;
  background: transparent; border: none; border-top: 1px solid var(--rule);
  color: var(--text-secondary); font-size: var(--fs-meta); cursor: pointer;
  text-align: left;
}
.ap-adv-head:hover { color: var(--text-primary); }
.ap-adv-sum {
  margin-left: auto; color: var(--text-muted); font-size: var(--fs-label);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
/* 여섯 항목을 다 펼치면 388px — 그만큼 프롬프트 영역이 사라진다.
   접힘이 기본이고 여는 건 잠깐이라, 자라는 대신 여기 안에서 스크롤시킨다. */
.ap-adv {
  display: flex; flex-direction: column; gap: var(--sp-2);
  max-height: 176px; overflow-y: auto;
  padding: var(--sp-1) var(--sp-1) var(--sp-1) 0;
}
.ap-adv-item { display: flex; flex-direction: column; gap: 2px; }
.ap-adv-line { display: flex; align-items: center; gap: var(--sp-2); }
.ap-adv-lab {
  flex: 1; min-width: 0; font-size: var(--fs-meta); color: var(--text-secondary);
  font-weight: var(--fw-medium);
}
/* 툴팁에 갇혀 있던 설명 — 255자·260자짜리 title 을 라벨 밑 한 줄로 꺼냈다. */
.ap-why { font-size: var(--fs-label); color: var(--text-muted); line-height: 1.5; }
.ap-mini {
  height: 28px; padding: 0 var(--sp-2); flex-shrink: 0;
  display: inline-flex; align-items: center; gap: var(--sp-1);
  background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-secondary); font-size: var(--fs-label); cursor: pointer;
  transition: var(--transition);
}
.ap-mini:hover { border-color: var(--accent); color: var(--accent); }

/* ── 조종석 ── */
.ap-head { display: flex; align-items: baseline; gap: var(--sp-2); }
.ap-dot {
  width: 8px; height: 8px; border-radius: 50%; align-self: center; flex-shrink: 0;
  background: var(--state-ok-fg); animation: apPulse 1.6s ease-in-out infinite;
}
.ap-dot.held { background: var(--text-muted); animation: none; }
@keyframes apPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.ap-big {
  font-size: 26px; font-weight: var(--fw-bold); color: var(--text-primary);
  line-height: 1; font-variant-numeric: tabular-nums;
}
.ap-head-unit { font-size: var(--fs-body); color: var(--text-secondary); }
.ap-head-tail {
  margin-left: auto; font-size: var(--fs-label); color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}
.ap-bar { height: 4px; background: var(--rule); border-radius: 2px; overflow: hidden; }
.ap-bar-fill { height: 100%; background: var(--accent); transition: width 0.3s linear; }

.ap-next {
  display: flex; flex-direction: column; gap: var(--sp-1);
  padding: var(--sp-2);
  background: var(--bg-primary); border: 1px solid var(--rule); border-radius: 6px;
}
.ap-next-head { display: flex; align-items: center; gap: var(--sp-1); }
.ap-next-title { font-size: var(--fs-meta); font-weight: var(--fw-medium); color: var(--text-primary); }
.ap-next-deck { font-size: var(--fs-label); color: var(--text-muted); font-variant-numeric: tabular-nums; }
.ap-next-hint { margin-left: auto; font-size: var(--fs-label); color: var(--text-muted); }
.ap-next-empty { font-size: var(--fs-label); color: var(--text-muted); padding: var(--sp-1) 0; }

/* 태그가 많아도 푸터가 자라면 안 된다 — 넘치면 여기 안에서만 스크롤한다.
   60px = 칩 두 줄(28+4+28). 실제 앱 창(높이 595px)에서 재 보니 88px 로는 조종석이
   275px 까지 자라 프롬프트 영역이 188px 로 떨어졌다 — 개선 전 199px 보다도 좁다.
   두 줄로 줄이면 216px 이 남는다. 잘린 태그는 이 안에서 스크롤되므로 접근성은 그대로. */
.ap-tags {
  display: flex; flex-wrap: wrap; gap: var(--sp-1);
  max-height: 60px; overflow-y: auto;
}
.ap-tag {
  --k: var(--tag-neutral);
  height: 28px; padding: 0 var(--sp-2); max-width: 100%;
  display: inline-flex; align-items: center; gap: var(--sp-1);
  background: color-mix(in srgb, var(--k) 16%, transparent);
  border: 1px solid transparent; border-radius: 5px;
  color: var(--k); font-size: var(--fs-label); cursor: pointer;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  transition: var(--transition);
}
.ap-tag:hover { border-color: var(--k); }
.ap-tag.k-person { --k: var(--tag-person); }
.ap-tag.k-scene { --k: var(--tag-scene); }
.ap-tag.k-pose { --k: var(--tag-pose); }
.ap-tag.k-wear { --k: var(--tag-wear); }
.ap-tag.k-fx { --k: var(--tag-fx); }
.ap-tag.k-nsfw { --k: var(--tag-nsfw); }
.ap-tag.off {
  --k: var(--text-muted);
  background: var(--bg-input); text-decoration: line-through;
}
.ap-tag.added { --k: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, transparent); }
/* 지금 장 프롬프트 — 읽기만 한다(편집해도 다음 장을 새로 뽑을 때 버려진다) */
.ap-next.locked .ap-tag { cursor: default; }
.ap-next.locked .ap-tag:hover { border-color: transparent; }
.ap-next.locked .ap-tag.added:hover { border-color: color-mix(in srgb, var(--accent) 40%, transparent); }

.ap-add { display: flex; align-items: center; gap: var(--sp-1); color: var(--text-muted); }
.ap-add-input {
  flex: 1; min-width: 0; height: 28px; padding: 0 var(--sp-2);
  background: var(--bg-input); font-size: var(--fs-label);
}
.ap-once { font-size: var(--fs-label); color: var(--text-muted); line-height: 1.5; }
.ap-once b { color: var(--accent); font-weight: var(--fw-medium); }

.ap-ctl { display: flex; gap: var(--sp-2); }
.ap-btn {
  flex: 1; height: 32px;
  display: inline-flex; align-items: center; justify-content: center; gap: var(--sp-1);
  background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-secondary); font-size: var(--fs-meta); cursor: pointer;
  transition: var(--transition);
}
.ap-btn:hover { border-color: var(--edge); color: var(--text-primary); }
.ap-btn.stop { border-color: var(--state-alert); color: var(--state-alert-fg); font-weight: var(--fw-bold); }
.ap-btn.stop:hover { background: color-mix(in srgb, var(--state-alert) 16%, transparent); color: var(--state-alert-fg); }
</style>
