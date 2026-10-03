<template>
  <div class="cc-spark">
    <svg v-if="values" class="cc-spark-svg" viewBox="0 0 100 24" preserveAspectRatio="none" role="img"
      :aria-label="summary">
      <line class="cc-spark-zero" x1="0" :y1="zeroY" x2="100" :y2="zeroY" />
      <polyline class="cc-spark-line" :points="points" />
    </svg>
    <span class="cc-spark-text">{{ values ? summary : NO_PREVIEW }}</span>
  </div>
</template>

<script setup lang="ts">
/**
 * 수정자 하나의 스텝별 강도 미리보기(보기 전용 — 보내는 값은 바꾸지 않는다). 값은 utils/colorcraft.scheduleValues
 * (업스트림 make_schedule 의 TS 판, 확장이 그 패스의 스텝 수로 만드는 배열)이다. 스텝 수는 카드가 지금 Steps 칸에서 준다.
 */
import { computed } from 'vue'
import { scheduleValues, type ColorcraftNode } from '../../utils/colorcraft'

const props = defineProps<{ modifier: ColorcraftNode; steps: number }>()

const NO_PREVIEW = '미리보기 없음 — 시작·끝·치우침이 0~1 밖이거나 스텝 수를 모릅니다'
const values = computed(() => scheduleValues(props.modifier, props.steps))
const scale = computed(() => Math.max(1, ...(values.value ?? [0]).map(v => Math.abs(v))))
const yOf = (v: number) => 12 - (v / scale.value) * 11
const zeroY = computed(() => yOf(0))
const points = computed(() => {
  const list = values.value ?? []
  const last = Math.max(1, list.length - 1)
  return list.map((v, i) => `${list.length === 1 ? 50 : ((i / last) * 100).toFixed(2)},${yOf(v).toFixed(2)}`).join(' ')
})
const summary = computed(() => {
  const list = values.value
  if (!list) return NO_PREVIEW
  let peak = 0
  list.forEach((v, i) => { if (Math.abs(v) > Math.abs(list[peak]!)) peak = i })
  const active = list.flatMap((v, i) => (v !== 0 ? [i + 1] : []))
  const span = active.length ? `스텝 ${active[0]}–${active[active.length - 1]}` : '모든 스텝 0'
  return `${list.length} 스텝 · 가장 큰 값 ${list[peak]!.toFixed(2)} (스텝 ${peak + 1}) · ${span}`
})
</script>

<style scoped>
.cc-spark { display: flex; align-items: center; gap: var(--sp-2); margin: 2px 0 6px; }
.cc-spark-svg { width: 120px; height: 24px; flex: none; }
.cc-spark-zero { stroke: var(--border); stroke-width: 0.5; }
.cc-spark-line { fill: none; stroke: var(--state-info-fg); stroke-width: 1.2; vector-effect: non-scaling-stroke; }
.cc-spark-text { color: var(--text-muted); font-size: var(--fs-meta); }
</style>
