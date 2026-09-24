<template>
  <div class="stats-overlay" @mousedown.self="closeStatsModal">
    <div class="stats-modal">
      <div class="stats-header">
        <h3>생성 통계</h3>
        <button class="close-btn" @click="closeStatsModal">X</button>
      </div>
      <div class="stats-body" v-if="genStats.total > 0">
        <div class="stats-cards">
          <div class="stat-card">
            <div class="stat-val">{{ genStats.total }}</div>
            <div class="stat-label">Total</div>
          </div>
          <div class="stat-card accent">
            <div class="stat-val">{{ genStats.success_rate }}%</div>
            <div class="stat-label">Success</div>
          </div>
          <div class="stat-card">
            <div class="stat-val">{{ genStats.avg_time }}s</div>
            <div class="stat-label">Avg Time</div>
          </div>
          <div class="stat-card">
            <div class="stat-val">{{ formatDuration(genStats.total_time) }}</div>
            <div class="stat-label">총 시간</div>
          </div>
        </div>

        <div class="stats-section" v-if="genStats.daily && genStats.daily.length">
          <h4>일별 생성 (최근 30일)</h4>
          <div class="daily-chart">
            <div v-for="d in genStats.daily" :key="d.date" class="daily-bar-wrap"
              :title="d.date + ': ' + d.count + '장'">
              <div class="daily-bar" :style="{ height: genStats.daily_max ? (d.count / genStats.daily_max * 100) + '%' : '0%' }"></div>
            </div>
          </div>
          <div class="daily-labels">
            <span>30일 전</span><span>오늘</span>
          </div>
        </div>

        <div class="stats-two-col">
          <div class="stats-section" v-if="genStats.top_models && genStats.top_models.length">
            <h4>많이 쓴 모델</h4>
            <div v-for="m in genStats.top_models" :key="m.name" class="stats-bar-row">
              <span class="bar-name">{{ m.name }}</span>
              <div class="bar-track"><div class="bar-fill" :style="{ width: (m.count / genStats.total * 100) + '%' }"></div></div>
              <span class="bar-count">{{ m.count }}</span>
            </div>
          </div>
          <div class="stats-section" v-if="genStats.top_resolutions && genStats.top_resolutions.length">
            <h4>많이 쓴 해상도</h4>
            <div v-for="r in genStats.top_resolutions" :key="r.res" class="stats-bar-row">
              <span class="bar-name">{{ r.res }}</span>
              <div class="bar-track"><div class="bar-fill" :style="{ width: (r.count / genStats.total * 100) + '%' }"></div></div>
              <span class="bar-count">{{ r.count }}</span>
            </div>
          </div>
        </div>

        <div class="stats-section" v-if="genStats.recent && genStats.recent.length">
          <h4>최근 생성</h4>
          <div class="recent-table">
            <div v-for="r in genStats.recent" :key="r.timestamp" class="recent-row">
              <span class="r-time">{{ r.timestamp?.slice(5, 16).replace('T', ' ') }}</span>
              <span class="r-status" :class="r.success ? 'ok' : 'fail'">{{ r.success ? 'OK' : 'FAIL' }}</span>
              <span class="r-dur">{{ r.duration_sec }}s</span>
              <span class="r-res">{{ r.width }}x{{ r.height }}</span>
              <span class="r-model">{{ (r.model || '').split('/').pop()?.slice(0, 20) }}</span>
            </div>
          </div>
        </div>
      </div>
      <div class="stats-body stats-empty" v-else>
        <div class="stats-empty-msg">아직 생성 기록이 없습니다</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 생성 통계 모달 — 표시만 한다. 상태는 composables/useGenStats.ts.
 * App.vue 가 `<transition name="fade">` 안에서 v-if 로 연다.
 */
import { useGenStats } from '../../composables/useGenStats'
import { useModalLayer } from '../../composables/useModalLayer'
import { formatDuration } from '../../utils/durationFormat'

const { showStatsModal, genStats, closeStatsModal } = useGenStats()

useModalLayer({ isOpen: () => showStatsModal.value, close: closeStatsModal })
</script>

<style scoped>
.close-btn { width: 28px; height: 28px; background: var(--bg-button); border: 1px solid var(--border-strong); border-radius: 6px; color: var(--text-secondary); font-size: 16px; cursor: pointer; }
.close-btn:hover { border-color: var(--state-alert-fg); color: var(--state-alert-fg); background: rgba(248, 113, 113, 0.08); }

/* Generation Stats Modal */
.stats-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.stats-modal { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 16px; width: min(860px, 94vw); max-height: 88vh; overflow-y: auto; }
.stats-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 24px; border-bottom: 1px solid var(--border); }
.stats-header h3 { font-size: 13px; font-weight: var(--fw-bold); letter-spacing: 0; color: var(--text-primary); }
.stats-body { padding: 24px; display: flex; flex-direction: column; gap: 24px; }
.stats-empty { display: flex; align-items: center; justify-content: center; min-height: 200px; }
.stats-empty-msg { color: var(--text-muted); font-size: 14px; }

.stats-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat-card { background: var(--bg-input); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
.stat-card.accent { border-color: var(--accent-dim); }
.stat-val { font-size: 28px; font-weight: var(--fw-bold); color: var(--text-primary); font-family: monospace; }
.stat-card.accent .stat-val { color: var(--accent); }
.stat-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin-top: 4px; }

.stats-section h4 { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin-bottom: 12px; }
.stats-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

.daily-chart { display: flex; align-items: flex-end; gap: 2px; height: 80px; padding: 0 2px; }
.daily-bar-wrap { flex: 1; height: 100%; display: flex; align-items: flex-end; }
.daily-bar { width: 100%; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 1px; transition: height 0.3s; }
.daily-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: var(--fs-label); color: var(--text-muted); }

.stats-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.bar-name { font-size: 11px; color: var(--text-secondary); min-width: 80px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 6px; background: var(--bg-button); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
.bar-count { font-size: var(--fs-label); color: var(--text-muted); min-width: 30px; text-align: right; font-family: monospace; }

.recent-table { display: flex; flex-direction: column; gap: 4px; }
.recent-row { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: var(--bg-input); border-radius: 6px; font-size: 11px; }
.r-time { color: var(--text-muted); min-width: 80px; font-family: monospace; }
.r-status { font-weight: var(--fw-bold); font-size: var(--fs-label); min-width: 30px; }
.r-status.ok { color: var(--state-ok-fg); }
.r-status.fail { color: var(--state-alert-fg); }
.r-dur { color: var(--accent); min-width: 40px; font-family: monospace; }
.r-res { color: var(--text-secondary); min-width: 70px; }
.r-model { color: var(--text-muted); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
