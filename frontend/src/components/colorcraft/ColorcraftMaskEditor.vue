<template>
  <div class="cc-editor" role="group" :aria-label="`마스크 ${tag}`">
    <div class="cc-editor-title">{{ isCombo ? `조합 ${tag}` : `마스크 ${tag}` }}
      <span v-if="isCombo && !resolved" class="cc-incomplete">{{ INCOMPLETE }} — A·B 를 모두 골라야 마스크가 됩니다</span>
    </div>
    <div class="ext-row">
      <template v-for="field in fields" :key="field.name">
        <label v-if="field.kind === 'bool'" class="ext-check-row" :title="field.title">
          <ToggleSwitch :modelValue="Boolean(node[field.name])" size="sm" @update:modelValue="emit('update', field.name, $event)" />
          <span>{{ field.label }}</span></label>
        <div v-else-if="field.kind === 'choice'" class="ext-field"><label :title="field.title">{{ field.label }}</label>
          <CustomSelect :modelValue="choiceLabel(labelKey(field.name), String(node[field.name]))"
            :options="optionsOf(field).map(value => choiceLabel(labelKey(field.name), value))"
            @update:modelValue="pick(field, $event)" /></div>
        <div v-else class="ext-field"><label :title="field.title">{{ field.label }}</label>
          <input type="number" :value="node[field.name]" :min="field.min ?? undefined" :max="field.max ?? undefined"
            :step="field.step ?? undefined" :title="field.title" :class="{ 'cc-out': outOfRange(field, node[field.name]) }"
            @change="emit('update', field.name, ($event.target as HTMLInputElement | null)?.value ?? '')" /></div>
      </template>
    </div>
    <p v-if="!isCombo" class="ext-note">흐림은 이미지 픽셀 기준입니다(latent 에서는 VAE 배율로 나눕니다).</p>
    <p v-if="isCombo && index === 0" class="ext-note">C1 은 앞선 조합이 없어 마스크 잎만 고를 수 있습니다.</p>
    <p v-if="outside.length" class="ext-note cc-warn" role="note">{{ outside.join(' · ') }}: {{ OUT_OF_RANGE_NOTE }}</p>
    <div class="cc-actions">
      <button type="button" class="cc-reset" :title="`${tag} 의 값을 모두 기본값으로`" @click="emit('reset')">이 마스크 초기화</button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Colorcraft 마스크 하나의 편집기 — 잎(M1 … M10, 10칸) 또는 조합(C1 … C5, 7칸). 조합의 Mask A/B 선택지는
 * comboRefChoices(조합 번호) — 잎과 앞선 조합만(확장과 같다). 값은 바꾸지 않고 ``update(칸 이름, 값)`` 으로 알린다.
 */
import { computed } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import {
  COMBO_TAGS, INCOMPLETE, MASK_TAGS, OUT_OF_RANGE_NOTE, SCHEMA, choiceLabel, comboRefChoices, outOfRange,
  resolvedCombos, type ColorcraftConfig, type ColorcraftField,
} from '../../utils/colorcraft'

const props = defineProps<{ config: ColorcraftConfig; tag: string }>()
const emit = defineEmits<{ update: [name: string, value: unknown]; reset: [] }>()

const isCombo = computed(() => COMBO_TAGS.includes(props.tag))
const index = computed(() => (isCombo.value ? COMBO_TAGS : MASK_TAGS).indexOf(props.tag))
const fields = computed(() => (isCombo.value ? SCHEMA.comboFields : SCHEMA.leafFields))
const node = computed(() => (isCombo.value ? props.config.combos : props.config.leaves)[Math.max(0, index.value)]!)
const resolved = computed(() => resolvedCombos(props.config).has(props.tag))
const outside = computed(() => fields.value.filter(f => outOfRange(f, node.value[f.name])).map(f => f.label))

/** Mask A/B 는 '없음' 라벨을 수정자 마스크 칸과 같이 쓴다. */
const labelKey = (name: string) => (name === 'mask_a' || name === 'mask_b' ? 'mask' : name)
const optionsOf = (field: ColorcraftField): readonly string[] =>
  isCombo.value && (field.name === 'mask_a' || field.name === 'mask_b') ? comboRefChoices(index.value) : field.choices

function pick(field: ColorcraftField, label: string | number) {
  const value = optionsOf(field).find(choice => choiceLabel(labelKey(field.name), choice) === String(label))
  if (value !== undefined) emit('update', field.name, value)
}
</script>

<style scoped>
.cc-editor { border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin: 4px 0 8px; }
.cc-editor-title { font-weight: var(--fw-medium); font-size: var(--fs-label); margin-bottom: 4px; }
.cc-incomplete { margin-left: var(--sp-2); color: var(--state-warn-fg); font-weight: normal; }
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
