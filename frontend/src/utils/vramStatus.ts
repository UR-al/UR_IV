/**
 * 하단 계기 스트립의 VRAM 칸 — 사용률 단계와 툴팁 문구(App.vue 에서 추출).
 * source: 'nvml' | 'nvidia-smi' = GPU 전체(모든 프로세스, 작업 관리자와 같은 숫자) / 'backend' = 백엔드 자기 메모리만
 */
export interface VramInfo { used: number; total: number; pct: number; source: string }
export type VramLevel = 'ok' | 'warn' | 'critical'

/** 90% 초과 critical · 70% 초과 warn · 그 밖 ok */
export function vramLevel(pct: number): VramLevel {
  return pct > 90 ? 'critical' : pct > 70 ? 'warn' : 'ok'
}

/** 툴팁 — 전체 크기를 모르면(0) 빈 문자열. */
export function vramTooltipText(v: VramInfo): string {
  if (!v.total) return ''
  const level = vramLevel(v.pct)
  const free = (v.total - v.used).toFixed(1)
  let msg = `사용: ${v.used}GB / 전체: ${v.total}GB (여유: ${free}GB)`
  msg += v.source === 'backend' ? '\n백엔드가 잡은 메모리만 — Ollama 등 다른 프로세스는 빠짐' : '\nGPU 전체 (모든 프로세스 · 5초마다 갱신)'
  if (level === 'critical') msg += '\n⚠ VRAM 부족 — 해상도/배치 크기를 줄이거나 모델 unload 권장'
  else if (level === 'warn') msg += '\n▲ 70% 초과 — 추가 작업 시 OOM 가능성'
  msg += '\n\n클릭하여 백엔드 모델 unload 요청'
  return msg
}
