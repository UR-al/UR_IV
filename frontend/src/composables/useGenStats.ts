import { reactive, ref } from 'vue'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

export interface DailyStat { date: string; count: number; [k: string]: any }
export interface ModelStat { name: string; count: number; [k: string]: any }
export interface ResStat { res: string; count: number; [k: string]: any }
export interface RecentStat { timestamp: string; success: boolean; duration_sec: number; width: number; height: number; model: string; [k: string]: any }
export interface GenStats {
  total: number; success: number; fail: number; success_rate: number; avg_time: number; total_time: number;
  daily: DailyStat[]; daily_max: number; top_models: ModelStat[]; top_resolutions: ResStat[]; recent: RecentStat[];
  [k: string]: any
}

/**
 * 생성 통계 모달 — App.vue 에서 추출(App.vue 분할 ④). 화면은 components/managers/GenStatsModal.vue.
 * 열 때마다 getGenStats 로 새로 받는다(core/gen_stats.py).
 */
export function createGenStats(deps: Pick<ManagerDeps, 'getBackend'>) {
  const showStatsModal = ref(false)
  const genStats = reactive<GenStats>({
    total: 0, success: 0, fail: 0, success_rate: 0, avg_time: 0, total_time: 0,
    daily: [], daily_max: 0, top_models: [], top_resolutions: [], recent: [],
  })

  async function loadGenStats() {
    const bk = await deps.getBackend()
    if (bk.getGenStats) {
      bk.getGenStats((json: string) => {
        try { Object.assign(genStats, JSON.parse(json)) } catch {}
      })
    }
  }
  /** 스튜디오 도구 '통계' 버튼 */
  function openStatsModal() {
    showStatsModal.value = true
    void loadGenStats()
  }
  function closeStatsModal() { showStatsModal.value = false }

  return { showStatsModal, genStats, loadGenStats, openStatsModal, closeStatsModal }
}

export type GenStatsState = ReturnType<typeof createGenStats>

let _app: GenStatsState | null = null
export function useGenStats(): GenStatsState {
  if (!_app) _app = createGenStats(appManagerDeps())
  return _app
}
