<template>
  <div class="wc-overlay" @mousedown.self="closeWcManager">
    <div class="wc-modal">
      <div class="wc-modal-header">
        <h3>와일드카드 관리</h3>
        <span class="wc-path">wildcards/</span>
        <label class="wc-enable" title="끄면 __이름__ · ~/이름/~ · {A|B} 를 풀지 않고 그대로 보낸다">
          <ToggleSwitch v-model="wildcardEnabled" size="sm" /><span>생성 시 치환</span>
        </label>
        <button class="wc-new-btn" @click="createNewWildcard">+ NEW</button>
        <button class="close-btn" @click="closeWcManager"><Icon name="close" /></button>
      </div>
      <div class="wc-modal-body">
        <!-- 파일 목록 -->
        <div class="wc-sidebar">
          <div v-for="wc in wildcards" :key="wc.name" class="wc-file-item"
            :class="{ active: selectedWc === wc.name }" @click="selectWildcard(wc.name)">
            <span class="wc-fname" @dblclick.stop="renameWildcard(wc.name)">{{ wc.name }}</span>
            <span class="wc-file-count">{{ wc.tags.length }}</span>
            <button class="wc-del" @click.stop="deleteWildcard(wc.name)"><Icon name="close" /></button>
          </div>
        </div>
        <!-- 편집 영역 -->
        <div class="wc-content">
          <template v-if="selectedWcData">
            <div class="wc-content-header">
              <!-- 파일명 인라인 편집 -->
              <h4 v-if="!wcRenaming" @dblclick="beginRename">{{ selectedWc }}</h4>
              <input v-else v-model="wcNewName" class="wc-rename-input" @blur="finishWcRename" @keydown.enter="finishWcRename" ref="wcRenameRef" />
              <span class="wc-syntax" title="표기가 둘이면 같은 뜻 — 한 줄마다 쉼표 후보 중 하나를 뽑는다. # 으로 시작하는 줄은 주석">문법: <template v-for="(form, fi) in wildcardSyntaxForms(selectedWc)" :key="form"><template v-if="fi"> 또는 </template><code>{{ form }}</code></template></span>
            </div>
            <!-- 블록 편집 -->
            <div class="wc-blocks">
              <div v-for="(line, li) in wcEditLines" :key="li" class="wc-block-row" :class="{ comment: line.trim().startsWith('#') }">
                <span class="wc-block-idx">{{ li + 1 }}</span>
                <input v-model="wcEditLines[li]" class="wc-block-input" @keydown.enter="addWcLine(li)" />
                <button class="wc-block-rm" @click="removeWcLine(li)"><Icon name="close" /></button>
              </div>
              <button class="wc-add-line" @click="appendWcLine">+ 줄 추가</button>
            </div>
            <!-- 하단: 삽입 + 저장 -->
            <div class="wc-bottom-bar">
              <CustomSelect v-model="wcInsertTarget" :options="['main', 'prefix', 'suffix', 'clipboard']" placeholder="삽입 위치" class="wc-insert-sel" />
              <button class="wc-use-btn" @click="useWcSyntax">사용</button>
              <div class="wc-spacer"></div>
              <button class="wc-save-btn" @click="saveCurrentWildcard"><Icon name="save" /> 저장</button>
            </div>
          </template>
          <div v-else class="wc-empty">좌측에서 와일드카드를 선택하거나 NEW를 클릭하세요</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 와일드카드 관리 모달 — 표시만 한다. 상태·동작은 composables/useWildcardManager.ts(모듈 싱글턴):
 * PromptPanel 의 와일드카드 칩이 모달 밖에서 같은 상태를 연다. App.vue 가 `<transition name="fade">`
 * 안에서 v-if 로 연다.
 */
import { nextTick, ref } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWildcardManager } from '../../composables/useWildcardManager'
import { useModalLayer } from '../../composables/useModalLayer'
import { wildcardSyntaxForms } from '../../utils/wildcardFile'

const {
  wildcards, wildcardEnabled, showWcManager, selectedWc, selectedWcData, wcEditLines, wcInsertTarget,
  wcRenaming, wcNewName,
  selectWildcard, saveCurrentWildcard, createNewWildcard, deleteWildcard,
  renameWildcard, startWcRename, finishWcRename, useWcSyntax,
  addWcLine, appendWcLine, removeWcLine, closeWcManager,
} = useWildcardManager()

const wcRenameRef = ref<HTMLInputElement | null>(null)
function beginRename() {
  startWcRename()
  nextTick(() => { if (wcRenameRef.value) wcRenameRef.value.focus() })
}

// 전역 ESC(App.vue)가 닫고, 열려 있는 동안 ↑/↓ 히스토리 이동을 막는다
useModalLayer({ isOpen: () => showWcManager.value, close: closeWcManager })
</script>

<style scoped src="./managerModal.css"></style>
