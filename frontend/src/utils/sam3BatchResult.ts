/**
 * SAM3 단독 실행 결과(`sam3Result`) 해석 — 단일 실행과 배치 항목을 가른다 (순수 로직).
 *
 * 워커(workers/sam3_worker.py)는 배치 항목마다 `{index, …}` 를 보낸다. 실패 항목은 `{error, path, index}` 이고,
 * 설정·설치 문제라 남은 이미지도 같은 이유로 실패할 때는 `batch_stopped: true`·`skipped` 를 싣고 배치를 멈춘다.
 * 예전 BatchView 는 배치 항목의 오류를 단일 실행 오류처럼 처리해 첫 실패에서 진행 표시를 끄고 시작 버튼을
 * 다시 열었다(워커는 남은 이미지를 계속 보냈다). 배치의 끝은 `sam3Progress` 가 total 에 닿을 때 또는 `batch_stopped`.
 * 완료 문구(성공·실패·건너뜀 수)는 Python 이 낸다.
 */
import { filenameOf } from './mediaKind'

export interface Sam3ResultEvent {
  before?: unknown
  after?: unknown
  output_path?: unknown
  path?: unknown
  index?: unknown
  error?: unknown
  batch_stopped?: unknown
  skipped?: unknown
  exif_warning?: unknown
}

export interface Sam3ResultOutcome {
  /** 배치 항목이면 그 인덱스, 단일 실행(또는 배치 시작 실패)이면 null */
  index: number | null
  ok: boolean
  error: string
  /** 이 이벤트로 실행이 끝났다 — 진행 표시를 끄고 시작 버튼을 연다 */
  endsRun: boolean
  /** 배치를 멈춰 건너뛴 이미지 수 (멈추지 않았으면 0) */
  skipped: number
  /** 실패한 파일 이름 (없으면 '') */
  file: string
}

function indexOf(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
}

export function sam3ResultOutcome(d: Sam3ResultEvent | null | undefined): Sam3ResultOutcome {
  const index = indexOf(d?.index)
  const error = d?.error ? String(d.error) : ''
  const stopped = index !== null && !!error && d?.batch_stopped === true
  const skipped = stopped && typeof d?.skipped === 'number' && d.skipped > 0 ? Math.floor(d.skipped) : 0
  return {
    index,
    ok: !error,
    error,
    // 단일 실행은 결과 하나로 끝난다. 배치 항목은 멈춘 경우에만 — 나머지는 sam3Progress 가 끝낸다.
    endsRun: index === null || stopped,
    skipped,
    file: error ? filenameOf(String(d?.path || d?.before || '')) : '',
  }
}

/** 오류 토스트 — 배치에서는 같은 원인을 실행마다 한 번만 띄운다(같은 설정 오류가 장마다 쌓이지 않게). */
export function createSam3ErrorToasts() {
  const seen = new Set<string>()
  return {
    /** 새 실행 시작 */
    reset() { seen.clear() },
    /** 실패 결과 하나 → 띄울 문구, 이미 띄운 원인이면 null */
    note(outcome: Sam3ResultOutcome): string | null {
      if (outcome.ok) return null
      if (outcome.index === null) return `SAM3 오류: ${outcome.error}`
      const tail = outcome.skipped ? ` — 같은 설정이면 남은 ${outcome.skipped}장도 실패해 배치를 멈췄습니다` : ''
      if (seen.has(outcome.error) && !tail) return null
      seen.add(outcome.error)
      return `SAM3 오류${outcome.file ? ` · ${outcome.file}` : ''}: ${outcome.error}${tail}`
    },
  }
}
