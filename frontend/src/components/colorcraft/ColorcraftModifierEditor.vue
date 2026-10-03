<template>
  <div class="cc-editor" role="group" :aria-label="`수정자 ${tag}`">
    <div class="cc-editor-title">수정자 {{ tag }}</div>
    <label class="ext-check-row" :title="titleOf('active')">
      <ToggleSwitch :modelValue="Boolean(node.active)" size="sm" @update:modelValue="set('active', $event)" />
      <span>{{ labelOf('active') }}</span></label>
    <div class="ext-row">
      <div class="ext-field"><label :title="titleOf('kind')">{{ labelOf('kind') }}</label>
        <CustomSelect :modelValue="choiceLabel('kind', String(node.kind))" :options="optionLabels('kind')"
          @update:modelValue="pick('kind', $event)" /></div>
      <div class="ext-field"><label :title="titleOf('pass')">{{ labelOf('pass') }}</label>
        <CustomSelect :modelValue="choiceLabel('pass', String(node.pass))" :options="optionLabels('pass')"
          @update:modelValue="pick('pass', $event)" /></div>
    </div>
    <div v-if="showMask" class="ext-field"><label :title="titleOf('mask')">{{ labelOf('mask') }}</label>
      <CustomSelect :modelValue="choiceLabel('mask', String(node.mask))" :options="optionLabels('mask')"
        @update:modelValue="pick('mask', $event)" /></div>
    <p v-if="hiresIdle" class="ext-note cc-warn" role="note">{{ HIRES_OFF_NOTE }}</p>
    <p v-if="incomplete" class="ext-note cc-warn" role="note">{{ node.mask }}{{ INCOMPLETE }} — 마스크 없이 적용합니다</p>
    <div class="ext-row cc-three">
      <div v-for="field in scheduleFields" :key="field.name" class="ext-field">
        <label :title="field.title">{{ field.label }}</label>
        <input type="number" :value="node[field.name]" :min="field.min ?? undefined" :max="field.max ?? undefined"
          :step="field.step ?? undefined" :title="field.title" :class="{ 'cc-out': outOfRange(field, node[field.name]) }"
          @change="setText(field.name, $event)" />
      </div>
    </div>
    <slot name="schedule" :modifier="node" />
    <details class="ext-sub">
      <summary>{{ GROUP_LABELS.shaping }}</summary>
      <label class="ext-check-row" :title="titleOf('advanced')">
        <ToggleSwitch :modelValue="Boolean(node.advanced)" size="sm" @update:modelValue="set('advanced', $event)" />
        <span>{{ labelOf('advanced') }}</span></label>
      <div class="ext-row">
        <div v-for="field in shapingNumbers" :key="field.name" class="ext-field">
          <label :title="field.title">{{ field.label }}</label>
          <input type="number" :value="node[field.name]" :min="field.min ?? undefined" :max="field.max ?? undefined"
            :step="field.step ?? undefined" :disabled="!node.advanced" :title="field.title"
            :class="{ 'cc-out': outOfRange(field, node[field.name]) }" @change="setText(field.name, $event)" />
        </div>
      </div>
      <label class="ext-check-row" :title="titleOf('smooth')">
        <ToggleSwitch :modelValue="Boolean(node.smooth)" size="sm" @update:modelValue="set('smooth', $event)" />
        <span>{{ labelOf('smooth') }}</span></label>
    </details>
    <section v-for="group in groups" :key="group.group" class="cc-group" :data-group="group.group">
      <div class="cc-group-head">
        <span class="cc-group-title">{{ group.label }}</span>
        <label v-if="group.gate" class="ext-check-row cc-gate" :title="group.gate.title">
          <ToggleSwitch :modelValue="Boolean(node[group.gate.name])" size="sm"
            @update:modelValue="set(group.gate.name, $event)" /><span>{{ group.gate.label }}</span></label>
      </div>
      <div class="ext-row">
        <template v-for="field in group.fields" :key="field.name">
          <label v-if="field.kind === 'bool'" class="ext-check-row" :title="field.title">
            <ToggleSwitch :modelValue="Boolean(node[field.name])" size="sm" @update:modelValue="set(field.name, $event)" />
            <span>{{ field.label }}</span></label>
          <div v-else-if="field.kind === 'choice'" class="ext-field"><label :title="field.title">{{ field.label }}</label>
            <CustomSelect :modelValue="choiceLabel(field.name, String(node[field.name]))" :options="optionLabels(field.name)"
              @update:modelValue="pick(field.name, $event)" /></div>
          <div v-else class="ext-field"><label :title="field.title">{{ field.label }}</label>
            <input type="number" :value="node[field.name]" :min="field.min ?? undefined" :max="field.max ?? undefined"
              :step="field.step ?? undefined" :title="field.title" :disabled="lockedByOverride(field.name)"
              :class="{ 'cc-out': outOfRange(field, node[field.name]) }" @change="setText(field.name, $event)" /></div>
        </template>
      </div>
      <p v-if="gateOff(group)" class="ext-note">스위치가 꺼져 있어 이 묶음 값은 쓰이지 않습니다</p>
    </section>
    <p v-if="outside.length" class="ext-note cc-warn" role="note">{{ outside.join(' · ') }}: {{ OUT_OF_RANGE_NOTE }}</p>
    <div class="cc-actions">
      <button type="button" class="cc-reset" :title="`수정자 ${tag} 의 값을 모두 기본값으로`" @click="emit('reset')">이 수정자 초기화</button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Colorcraft 수정자 하나의 편집기 — 머리(켜기·종류·패스·마스크), 스케줄, 고급 스케줄, 그 종류가 보이는 묶음만
 * (utils/colorcraft.visibleGroups — 확장 KIND_GROUPS, Advanced 의 Chroma Plus·색 이동 스위치). 값은 바꾸지 않고
 * ``update(경로 이름, 값)`` 으로 알린다 — 카드가 상태 문자열에 쓴다(값 정규화는 setPath, 범위로 자르지 않는다).
 * 마스크 칸은 마스크 사용 + 마스크를 받는 종류(Basic 아님)일 때만. Pass=Hires 인데 Hires.fix 가 꺼져 있으면, 미완성 조합을
 * 골랐으면 안내한다. 확장 슬라이더 범위 밖 값은 그대로 두고 알린다.
 */
import { computed } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import {
  COMBO_TAGS, INCOMPLETE, MODIFIER_TAGS, OUT_OF_RANGE_NOTE, SCHEMA, choiceLabel, kindOf, outOfRange, resolvedCombos,
  takesMask, visibleGroups, type ColorcraftConfig, type FieldGroup,
} from '../../utils/colorcraft'

const props = withDefaults(defineProps<{
  config: ColorcraftConfig
  tag: string
  hiresOn?: boolean
}>(), { hiresOn: true })
const emit = defineEmits<{ update: [name: string, value: unknown]; reset: [] }>()

const HIRES_OFF_NOTE = 'Hires.fix 가 꺼져 있어 이 수정자(패스 Hires)는 돌지 않습니다'
const GROUP_LABELS = SCHEMA.groupLabels
const FIELDS = new Map(SCHEMA.modifierFields.map(f => [f.name, f] as const))
const node = computed(() => props.config.modifiers[Math.max(0, MODIFIER_TAGS.indexOf(props.tag))]!)
const kind = computed(() => kindOf(node.value))
const groups = computed<FieldGroup[]>(() => visibleGroups(kind.value))
const showMask = computed(() => props.config.masking && takesMask(kind.value))
const hiresIdle = computed(() => !props.hiresOn && node.value.active === true && node.value.pass === SCHEMA.passHires)
const incomplete = computed(() => showMask.value && COMBO_TAGS.includes(String(node.value.mask))
  && !resolvedCombos(props.config).has(String(node.value.mask)))
const scheduleFields = ['strength', 'start', 'end'].map(name => FIELDS.get(name)!)
const shapingNumbers = ['exponent', 'bias', 'start_off', 'end_off'].map(name => FIELDS.get(name)!)
const outside = computed(() => {
  const names = ['strength', 'start', 'end', ...(node.value.advanced ? ['exponent', 'bias', 'start_off', 'end_off'] : []),
    ...groups.value.flatMap(g => g.fields.map(f => f.name))]
  return names.filter(name => outOfRange(FIELDS.get(name)!, node.value[name])).map(name => FIELDS.get(name)!.label)
})

const labelOf = (name: string) => FIELDS.get(name)!.label
const titleOf = (name: string) => FIELDS.get(name)!.title
const optionLabels = (name: string) => FIELDS.get(name)!.choices.map(value => choiceLabel(name, value))

function set(name: string, value: unknown) {
  emit('update', name, value)
}

function setText(name: string, event: Event) {
  emit('update', name, (event.target as HTMLInputElement | null)?.value ?? '')
}

function pick(name: string, label: string | number) {
  const field = FIELDS.get(name)!
  const value = field.choices.find(choice => choiceLabel(name, choice) === String(label))
  if (value !== undefined) emit('update', name, value)
}

/** 개발자 묶음: '직접 지정' 이 꺼진 값 칸은 잠근다(확장도 그 값을 쓰지 않는다). */
function lockedByOverride(name: string): boolean {
  const override = `${name}_override`
  return FIELDS.has(override) && !node.value[override]
}

function gateOff(group: FieldGroup): boolean {
  return group.gate !== null && !node.value[group.gate.name]
}

</script>

<style scoped>
.cc-editor { border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin: 4px 0 8px; }
.cc-editor-title { font-weight: var(--fw-medium); font-size: var(--fs-label); margin-bottom: 4px; }
.cc-three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cc-group { margin-top: 6px; }
.cc-group-head { display: flex; align-items: center; justify-content: space-between; gap: var(--sp-2); }
.cc-group-title { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-medium); }
.cc-gate { margin: 0; }
.ext-note { margin: 3px 0 6px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.ext-note.cc-warn { color: var(--state-warn-fg); }
input.cc-out { border-color: var(--state-warn-fg); }
.cc-actions { display: flex; justify-content: flex-end; margin-top: var(--sp-2); }
.cc-reset {
  height: 26px; padding: 0 10px; font-size: var(--fs-meta); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.cc-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
</style>
