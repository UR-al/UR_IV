<template>
  <section class="data-backup" aria-labelledby="data-backup-title">
    <div class="db-block">
      <h2 id="data-backup-title">설정 백업 · 복원</h2>
      <p class="db-note">
        앱 설정 · 기본값 · 조건식 · 가중치 · 프롬프트 순서 · 백엔드/모델 경로 · 지침 프리셋 · Comfy 컨트롤,
        생성 프리셋 · 캐릭터 프리셋 · 즐겨찾기 · 히스토리 · 와일드카드 · 워크플로 프로필을 ZIP 하나로 묶습니다.
      </p>
      <label class="db-check">
        <input v-model="includeChat" type="checkbox" />
        <span>대화 기록 포함 <em>이미지가 붙은 대화가 많으면 파일이 커집니다</em></span>
      </label>
      <div class="db-actions">
        <button type="button" v-host-dialog="'settings_export'" @click="exportSettings">
          <Icon name="download" size="14" /> 백업 내보내기 (ZIP)
        </button>
        <button type="button" :disabled="webHost" :title="webHost ? WEB_ONLY_HOST : ''" @click="importSettings">
          <Icon name="upload" size="14" /> 백업에서 복원…
        </button>
      </div>
      <p class="db-note">복원하면 현재 설정을 백업 내용으로 덮어쓰고 앱이 바로 재시작됩니다. 재시작 전까지는 설정을 저장하지 않습니다.</p>
    </div>

    <div class="db-block">
      <h2>앱 재시작</h2>
      <p class="db-note">현재 설정을 저장하고 앱을 다시 켭니다. 앱이 켠 Forge Neo / ComfyUI 는 함께 종료됩니다.</p>
      <div class="db-actions">
        <button type="button" :disabled="webHost || restarting" :title="webHost ? WEB_ONLY_HOST : ''" @click="restartApp">
          <Icon name="refresh" size="14" /> {{ restarting ? '재시작하는 중…' : '앱 재시작' }}
        </button>
      </div>
    </div>

    <div class="db-block">
      <h2>생성 프리셋 공유</h2>
      <p class="db-note">프리셋 관리의 생성 프리셋(프롬프트 · 모델 · 샘플링 · 해상도 · Hires · ADetailer · SAM3)을 JSON 파일 하나로 주고받습니다.</p>
      <label class="db-check">
        <input v-model="presetOverwrite" type="checkbox" />
        <span>같은 이름이 있으면 가져온 것으로 바꾸기 <em>끄면 기존 프리셋을 그대로 둡니다</em></span>
      </label>
      <div class="db-actions">
        <button type="button" v-host-dialog="'presets_export'" @click="send('presets_export')">
          <Icon name="download" size="14" /> 내보내기
        </button>
        <button type="button" v-host-dialog="'presets_import'" @click="send('presets_import', { overwrite: presetOverwrite })">
          <Icon name="upload" size="14" /> 가져오기
        </button>
      </div>
    </div>

    <div class="db-block">
      <h2>캐릭터 프리셋 공유</h2>
      <p class="db-note">캐릭터 특징 프리셋(추가 태그 · 조건부 규칙)을 JSON 파일로 주고받습니다.</p>
      <div class="db-mode" role="radiogroup" aria-label="캐릭터 프리셋 가져오기 방식">
        <label><input v-model="characterMode" type="radio" value="merge" /> 병합 <em>같은 캐릭터는 가져온 것으로 교체</em></label>
        <label><input v-model="characterMode" type="radio" value="replace" /> 전체 교체 <em>기존 캐릭터 프리셋을 모두 지움</em></label>
      </div>
      <div class="db-actions">
        <button type="button" v-host-dialog="'character_presets_export'" @click="send('character_presets_export')">
          <Icon name="download" size="14" /> 내보내기
        </button>
        <button type="button" v-host-dialog="'character_presets_import'" @click="importCharacters">
          <Icon name="upload" size="14" /> 가져오기
        </button>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { isWebHost, vHostDialog } from '../utils/hostDialogs'
import type { ActionName } from '../types/bridge'

/**
 * Settings '데이터 · 백업' — 설정 백업/복원, 앱 재시작, 생성/캐릭터 프리셋 공유(audit #179).
 * 파일 대화상자·실행은 호스트(ui/settings_data_actions.py)가 하고 결과는 토스트로 온다.
 * 복원·재시작은 웹 모드면 서버가 항상 거부하므로(core/web_action_policy) 버튼도 미리 끈다.
 */
const WEB_ONLY_HOST = '웹 모드에서는 호스트 PC 의 설정 복원·앱 재시작을 할 수 없습니다 — 호스트 PC 의 앱에서 하세요.'

const webHost = isWebHost()
const includeChat = ref(false)
const presetOverwrite = ref(false)
const characterMode = ref<'merge' | 'replace'>('merge')
/** 재시작을 보낸 뒤 잠시 버튼을 끈다 — 두 번 누르면 호스트가 거절하지만(restart_pending) 누를 수도 없게.
 *  재시작이 실패해 앱이 살아 있으면(오류 토스트) 시간이 지나 다시 켜진다. */
const RESTART_BUTTON_COOLDOWN_MS = 10000
const restarting = ref(false)

function send(name: ActionName, payload: Record<string, unknown> = {}) {
  requestAction(name, payload)
}

function exportSettings() {
  send('settings_export', { includeChat: includeChat.value })
}

function importSettings() {
  if (webHost) return
  if (!window.confirm('백업을 복원하면 현재 설정을 덮어쓰고 앱을 바로 재시작합니다. 계속할까요?')) return
  send('settings_import')
}

function restartApp() {
  if (webHost || restarting.value) return
  if (!window.confirm('현재 설정을 저장하고 앱을 재시작할까요?')) return
  restarting.value = true
  window.setTimeout(() => { restarting.value = false }, RESTART_BUTTON_COOLDOWN_MS)
  send('restart_app')
}

function importCharacters() {
  const replace = characterMode.value === 'replace'
  if (replace && !window.confirm('기존 캐릭터 프리셋을 모두 지우고 파일의 것으로 바꿉니다. 계속할까요?')) return
  send('character_presets_import', { replace })
}
</script>

<style scoped>
.data-backup { display: flex; flex-direction: column; gap: 16px; }
.db-block { padding: 20px; border: 1px solid var(--border); border-radius: 16px; background: var(--bg-card); color: var(--text-primary); }
h2 { font-size: 15px; margin: 0 0 8px; font-weight: var(--fw-bold); }
.db-note { margin: 8px 0; font-size: 12px; line-height: 1.6; color: var(--text-muted); }
.db-check, .db-mode label { display: flex; align-items: center; gap: 8px; font-size: 12px; cursor: pointer; }
.db-check em, .db-mode em { font-style: normal; color: var(--text-muted); margin-left: 4px; }
.db-mode { display: flex; flex-wrap: wrap; gap: 16px; margin: 8px 0; }
.db-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
button { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; background: var(--bg-button); color: var(--text-primary); cursor: pointer; font-size: 12px; font-weight: var(--fw-medium); }
button:hover:not(:disabled) { border-color: var(--accent); }
button:disabled { opacity: .5; cursor: not-allowed; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
