<template>
  <div class="wc-overlay" @mousedown.self="closeProfileManager">
    <div class="wc-modal" style="max-width: 640px;">
      <div class="wc-modal-header">
        <h3>워크플로우 프로파일</h3>
        <span class="wc-path">config/profiles/*.json</span>
        <button class="close-btn" @click="closeProfileManager"><Icon name="close" /></button>
      </div>
      <div class="order-modal-body">
        <div v-if="workflowProfiles.length === 0" class="wc-empty" style="padding: 20px;">
          저장된 프로파일이 없습니다.<br/>
          <small style="color: var(--text-muted)">상단의 + 버튼으로 현재 세팅을 저장하세요.</small>
        </div>
        <div v-else class="order-list">
          <div v-for="prof in workflowProfiles" :key="prof.name" class="profile-item">
            <div class="profile-info">
              <div class="profile-name">{{ prof.name }}</div>
              <div class="profile-meta">{{ prof.model || '모델 미지정' }}{{ prof.vae ? ' · VAE: ' + prof.vae : '' }}</div>
            </div>
            <button class="order-btn" @click="loadWorkflowProfile(prof.name)" title="적용"><Icon name="play" /></button>
            <button class="order-btn" @click="renameWorkflowProfile(prof.name)" title="이름 변경"><Icon name="pencil" /></button>
            <button class="order-btn" @click="deleteWorkflowProfile(prof.name)" title="삭제" style="color: var(--state-alert-fg);"><Icon name="close" /></button>
          </div>
        </div>
        <div class="order-actions">
          <button class="order-save" @click="saveCurrentAsProfile">+ 현재 세팅 저장</button>
          <div style="flex: 1;"></div>
          <button class="order-cancel" @click="closeProfileManager">닫기</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 워크플로우 프로파일 관리 모달 — 표시만 한다. 목록·동작은 composables/useWorkflowProfiles.ts:
 * 왼쪽 열의 프로파일 드롭다운이 모달 밖에서 같은 목록을 쓴다. App.vue 가 v-if 로 연다.
 */
import { useWorkflowProfiles } from '../../composables/useWorkflowProfiles'
import { useModalLayer } from '../../composables/useModalLayer'

const {
  showProfileManager, workflowProfiles,
  loadWorkflowProfile, saveCurrentAsProfile, deleteWorkflowProfile, renameWorkflowProfile, closeProfileManager,
} = useWorkflowProfiles()

useModalLayer({ isOpen: () => showProfileManager.value, close: closeProfileManager })
</script>

<style scoped src="./managerModal.css"></style>
