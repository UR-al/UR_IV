<template>
  <div class="wc-overlay" @mousedown.self="closeInstantWcManager">
    <div class="wc-modal">
      <div class="wc-modal-header">
        <h3>즉석 와일드카드 관리</h3>
        <span class="wc-path">user_data/instant_wildcards.json &nbsp;·&nbsp; 문법: <code>$$name$$</code></span>
        <button class="wc-new-btn" @click="createNewInstantWc">+ NEW</button>
        <button class="close-btn" @click="closeInstantWcManager"><Icon name="close" /></button>
      </div>
      <div class="wc-modal-body">
        <!-- 이름 목록 -->
        <div class="wc-sidebar">
          <div v-for="iw in instantWildcards" :key="iw.name" class="wc-file-item"
            :class="{ active: selectedInstantWc === iw.name }" @click="selectInstantWc(iw.name)">
            <span class="wc-fname">{{ iw.name }}</span>
            <span class="wc-file-count">{{ iw.lines.length }}</span>
            <button class="wc-del" @click.stop="deleteInstantWc(iw.name)"><Icon name="close" /></button>
          </div>
          <div v-if="instantWildcards.length === 0" class="wc-empty" style="padding: 12px; font-size: 11px;">
            와일드카드가 없습니다. + NEW로 추가하세요.
          </div>
        </div>
        <!-- 편집 영역 -->
        <div class="wc-content">
          <template v-if="selectedInstantWcData">
            <div class="wc-content-header">
              <h4>{{ selectedInstantWc }}</h4>
              <span class="wc-syntax">사용: <code>{{'$$' + selectedInstantWc + '$$'}}</code> ·
                가중치: <code>{100}:tag</code></span>
            </div>
            <div class="wc-blocks">
              <div v-for="(line, li) in iwEditLines" :key="li" class="wc-block-row">
                <span class="wc-block-idx">{{ li + 1 }}</span>
                <input v-model="iwEditLines[li]" class="wc-block-input"
                  :placeholder="li === 0 ? '예: happy 또는 {100}:happy (가중치)' : ''"
                  @keydown.enter="appendIwLine" />
                <button class="wc-block-rm" @click="removeIwLine(li)"><Icon name="close" /></button>
              </div>
              <button class="wc-add-line" @click="appendIwLine">+ 줄 추가</button>
            </div>
            <div class="wc-bottom-bar">
              <span style="font-size: 11px; color: var(--text-muted); margin-right: auto;"><Icon name="bulb" /><b>$$name$$</b>을 프롬프트에 넣으면 생성 시 라인 중 하나를 뽑아 치환합니다.
              </span>
              <button class="wc-save-btn" @click="saveCurrentInstantWc"><Icon name="save" /> 저장</button>
            </div>
          </template>
          <div v-else class="wc-empty">좌측에서 선택하거나 NEW를 클릭하세요</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 즉석 와일드카드(`$$name$$`) 관리 모달 — 표시만 한다. 상태·동작은 composables/useInstantWildcards.ts.
 * App.vue 가 `<transition name="fade">` 안에서 v-if 로 연다.
 */
import { useInstantWildcards } from '../../composables/useInstantWildcards'
import { useModalLayer } from '../../composables/useModalLayer'

const {
  showInstantWcManager, instantWildcards, selectedInstantWc, selectedInstantWcData, iwEditLines,
  selectInstantWc, createNewInstantWc, saveCurrentInstantWc, deleteInstantWc,
  appendIwLine, removeIwLine, closeInstantWcManager,
} = useInstantWildcards()

useModalLayer({ isOpen: () => showInstantWcManager.value, close: closeInstantWcManager })
</script>

<style scoped src="./managerModal.css"></style>
