<template>
  <div v-for="key in order" :key="key"><slot :name="key" /></div>
</template>

<script setup lang="ts">
/**
 * 프롬프트 입력 칸을 저장된 섹션 순서대로 놓는다. 칸은 부모의 슬롯이라 부모의 v-for 밖이고, 그래서
 * 부모의 템플릿 ref(`mainRef` 등)가 배열이 되지 않는다. 키가 있어 순서를 바꿔도 칸의 DOM 은 그대로 옮겨진다.
 */
import type { PromptSectionKey } from '../utils/promptSections'

defineProps<{ order: readonly PromptSectionKey[] }>()
</script>
