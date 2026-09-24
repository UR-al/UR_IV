/**
 * 에디터 '문서' 상태의 순수 계산 — 저장 여부(dirty), 저장 후 경로 치환, 이미지 크기 캐시.
 *
 * 예전에는 isDirty 가 여기저기서 직접 대입되는 ref 였다.
 *  - '저장'을 누르면 응답도 기다리지 않고 false 가 됐다(실제로는 파일을 안 썼는데도).
 *  - undo 로 원본까지 되돌아가도 ● 표시와 닫기 확인이 남았다.
 *  - 병합 안 한 드로잉 레이어는 아예 추적되지 않아 닫기 경고 없이 날아갔다.
 * 이제는 '마지막으로 저장된 상태'(SavedMarker) 와 현재 상태를 비교해서 계산한다.
 */

/** 마지막으로 디스크와 일치했던 상태. */
export interface SavedMarker {
  /** 그때의 확정 이미지 경로 (undo 스택의 한 항목) */
  imagePath: string
  /** 그때의 드로잉 레이어 리비전 */
  drawRevision: number
  /** 그때 레이어에 그린 것이 있었는지 — 저장본에 합성돼 들어갔는지 */
  drawHadContent: boolean
  /** 그때 레이어를 합성한 불투명도(0~100) — 저장본에 이 세기로 들어갔다 */
  drawOpacity: number
}

export interface EditorDocState {
  imagePath: string
  drawRevision: number
  drawHasContent: boolean
  /** 드로잉 레이어 불투명도(0~100) — 저장·병합 때 이 세기로 합성된다 */
  drawOpacity: number
}

/** 불투명도 비교 키 — 백엔드처럼 0~100 으로 자르고 정수로 맞춘다(슬라이더는 정수 단계). */
export function normalizeDrawOpacity(value: unknown): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return 100
  return Math.round(Math.min(100, Math.max(0, n)))
}

/**
 * 저장하지 않은 변경이 있는지.
 *
 * 레이어 리비전이 바뀌었어도, 지금도 비어 있고 저장 때도 비어 있었다면 변경이 아니다
 * (이미지를 새로 열 때 레이어를 비우는 것만으로 리비전이 오른다).
 *
 * 레이어 내용이 그대로여도 불투명도만 바꾸면 저장본(옛 불투명도로 합성)과 화면이 다르다 —
 * 저장본에 레이어가 들어갔을 때만 변경이다(빈 레이어의 불투명도는 결과에 영향이 없다).
 * 리비전을 올려서 표시하지 않는 이유: 슬라이더를 저장 때 값으로 되돌리면 다시 깨끗해야 한다.
 */
export function isEditorDirty(cur: EditorDocState, saved: SavedMarker | null): boolean {
  if (!cur.imagePath) return false
  if (!saved) return true
  if (cur.imagePath !== saved.imagePath) return true
  if (cur.drawRevision !== saved.drawRevision) return cur.drawHasContent || saved.drawHadContent
  if (cur.drawHasContent && saved.drawHadContent
    && normalizeDrawOpacity(cur.drawOpacity) !== normalizeDrawOpacity(saved.drawOpacity)) return true
  return false
}

/**
 * 문서 세대의 시작값 — 창(웹 모드의 여러 탭)마다 겹치지 않게 시각을 섞는다.
 * editorResult 는 연결된 모든 창에 방송된다. 세대가 0 부터 세는 작은 수면 다른 탭의
 * 편집 결과가 같은 세대 번호로 이 창의 문서에 들어온다.
 * (Date.now()*1000 ≈ 1.8e15 — Number.MAX_SAFE_INTEGER(9e15) 안이라 JSON 왕복에도 정확하다)
 */
export function initialDocGen(now: number = Date.now(), rand: number = Math.random()): number {
  return Math.floor(now) * 1000 + Math.floor(rand * 1000)
}

/**
 * editorProcess 결과가 지금 문서의 것인지. 요청마다 `doc_gen` 을 싣고 백엔드가 그대로 돌려준다.
 *
 * job_id 가드(더 새 작업이 시작되면 옛 결과를 버림)는 문서 전환을 못 잡는다 — A 에서 느린 작업
 * (배경 제거·자동 감지)을 시작하고 B 를 연 뒤 B 에서 아무 작업도 안 했으면 A 의 job 이 여전히
 * 최신이라, A 의 결과가 B 의 undo 히스토리로 들어가고(병합 안 한 드로잉도 지워짐) 닫은 에디터가
 * A 의 결과로 다시 열렸다. 세대가 다르면 이미지·마스크·오류·프리뷰 결과를 모두 버린다.
 * `doc_gen` 이 없는 결과(옛 백엔드)는 받아들인다.
 */
export function editorResultForDoc(
  result: { doc_gen?: unknown; [k: string]: unknown } | null | undefined, docGen: number,
): boolean {
  if (!result || typeof result !== 'object') return false
  const gen = result.doc_gen
  if (typeof gen !== 'number') return true
  return gen === docGen
}

/** 목록에서 `from` 을 `to` 로 바꾼 새 배열. 같은 항목이 여럿이면 모두 바꾼다. */
export function replacePath(list: readonly string[], from: string, to: string): string[] {
  return list.map((p) => (p === from ? to : p))
}

/** 백엔드 `editorSaveResult` 페이로드 (ui/editor_save_actions.py). */
export interface EditorSaveResult {
  request_id?: number | null
  mode?: 'save' | 'save_as'
  ok: boolean
  path?: string
  format?: string
  width?: number
  height?: number
  unchanged?: boolean
  /** 이 문서의 다음 '저장'이 `path` 를 덮어써도 되는지 — 연 원본이면 false (비파괴 저장) */
  owned?: boolean
  /** JPEG 원본인데 투명한 곳이 있어 PNG 로 저장했다 */
  alpha_png?: boolean
  cancelled?: boolean
  error?: string
  /** 덮어쓰기 전 내용의 사본 — 히스토리에서 `replaced_paths` 를 이걸로 바꾼다 */
  snapshot_path?: string
  replaced_paths?: string[]
}

/** 보낸 저장 요청 — 응답이 오면 그때의 문서 상태를 '저장됨'으로 기록한다. */
export interface PendingSave {
  id: number
  /** 요청할 때의 문서 세대 — 저장 중에 다른 이미지를 열면 달라진다 */
  docGen: number
  imagePath: string
  drawRevision: number
  drawHadContent: boolean
  /** 보낸 overlay_opacity — 응답이 오면 SavedMarker 로 옮긴다 */
  drawOpacity: number
  startedAt: number
}

/**
 * 저장 응답을 어떻게 다룰지.
 *  - `ignore`   : 이 창이 보낸 요청이 아니다(웹 모드의 다른 탭 등)
 *  - `cancelled`: 사용자가 대화상자를 닫았다
 *  - `failed`   : 저장 실패 — 문서가 바뀌었어도 사용자는 실패를 알아야 한다
 *  - `stale`    : 저장은 됐지만 그사이 다른 문서를 열거나 닫았다 — 알림만 띄우고
 *                 문서 상태(sourcePath·저장 표시·파일 정보·히스토리)는 건드리지 않는다.
 *                 예전에는 저장 중에 Ctrl+V 로 붙여 넣은 새 문서의 sourcePath 가 옛 파일이 되어,
 *                 다음 Ctrl+S 가 붙여 넣은 그림으로 옛 파일을 덮어썼다.
 *  - `apply`    : 지금 문서의 저장 결과 — 상태에 반영한다
 */
export type SaveResultAction = 'ignore' | 'cancelled' | 'failed' | 'stale' | 'apply'

export function saveResultAction(
  pending: PendingSave | null | undefined, result: EditorSaveResult, currentDocGen: number,
): SaveResultAction {
  if (!pending || result.request_id !== pending.id) return 'ignore'
  if (result.cancelled) return 'cancelled'
  if (!result.ok) return 'failed'
  if (pending.docGen !== currentDocGen) return 'stale'
  return 'apply'
}

/** 지금 문서의 저장이 아직 진행 중인지 — 같은 문서의 저장만 막는다(다른 문서는 따로 저장된다). */
export function hasPendingSaveFor(
  pending: Iterable<PendingSave>, docGen: number, now: number, staleMs: number,
): boolean {
  for (const p of pending) {
    if (p.docGen === docGen && now - p.startedAt < staleMs) return true
  }
  return false
}

/**
 * 경로 중 하나라도 크래시 복구 파일(%TEMP%/AIStudioPro_editor/_autosave_session.*)인지.
 * 편집 중인 문서가 그 파일을 참조하면 '저장했으니 복구본 정리'가 작업 중인 그림을 지운다.
 * (복구는 작업 사본으로 열지만, 사용자가 그 파일을 직접 열 수도 있다)
 */
export function referencesAutosaveFile(paths: Iterable<string>): boolean {
  for (const p of paths) {
    const norm = String(p || '').replace(/\\/g, '/').toLowerCase()
    if (/\/aistudiopro_editor\/_autosave_session\.[^/]*$/.test(norm)) return true
  }
  return false
}

/** 저장 성공 알림 문구. */
export function saveToastMessage(result: EditorSaveResult): string {
  const name = (result.path || '').replace(/\\/g, '/').split('/').pop() || ''
  if (result.unchanged) return `변경 사항 없음: ${name}`
  if (result.alpha_png) return `저장됨: ${name} (투명 영역을 지키려고 PNG 로 저장)`
  return `저장됨: ${name}`
}

export function parseSaveResult(json: string): EditorSaveResult | null {
  try {
    const value = JSON.parse(json)
    if (!value || typeof value !== 'object' || typeof value.ok !== 'boolean') return null
    return value as EditorSaveResult
  } catch {
    return null
  }
}

/**
 * 저장이 실제로 (새로 또는 덮어) 쓴 파일 경로 — 없으면 ''. 실패·취소·'변경 없음'(원본을 그대로
 * 돌려받음)은 쓰지 않았다. 어느 창이 보낸 요청이든 디스크의 파일은 바뀌었으므로 이 창의 갤러리·
 * 즐겨찾기 카드 캐시도 버려야 한다(utils/mediaVersions bumpMediaVersion).
 */
export function writtenPathFromSaveResult(result: EditorSaveResult | null | undefined): string {
  if (!result || !result.ok || result.cancelled || result.unchanged) return ''
  return typeof result.path === 'string' ? result.path : ''
}

/**
 * 저장이 덮어쓴 파일을 히스토리가 참조하고 있으면, 그 경로를 덮어쓰기 전 사본으로 바꿀
 * 치환 목록. (원본을 열고 바로 저장 → undo 스택 맨 아래가 '저장된 결과'를 가리키게 되는 것을 막는다)
 */
export function aliasesFromSaveResult(result: EditorSaveResult): Array<[string, string]> {
  const to = result.snapshot_path
  if (!result.ok || !to || !Array.isArray(result.replaced_paths)) return []
  return result.replaced_paths
    .filter((from): from is string => typeof from === 'string' && !!from && from !== to)
    .map((from) => [from, to])
}

export interface ImageSize { w: number; h: number }

/**
 * 경로별 이미지 크기. 편집 결과(result.width/height)와 첫 로드 때 채워 두고,
 * undo/redo 때 상단바·변형 패널의 크기를 새로 디코드하지 않고 되돌린다.
 * (예전에는 undo 뒤에도 옛 크기가 남아, 비율 유지 리사이즈가 뒤집힌 비율로 계산됐다)
 */
export class ImageSizeCache {
  private readonly map = new Map<string, ImageSize>()

  constructor(private readonly limit = 256) {}

  set(path: string, w: number, h: number): void {
    if (!path || !(w > 0) || !(h > 0)) return
    this.map.delete(path)
    this.map.set(path, { w: Math.round(w), h: Math.round(h) })
    while (this.map.size > this.limit) {
      const oldest = this.map.keys().next().value
      if (oldest === undefined) break
      this.map.delete(oldest)
    }
  }

  get(path: string): ImageSize | undefined {
    return path ? this.map.get(path) : undefined
  }

  /** 경로가 사본으로 바뀌었을 때 크기를 옮긴다. */
  rename(from: string, to: string): void {
    const size = this.map.get(from)
    if (size) this.set(to, size.w, size.h)
  }

  clear(): void {
    this.map.clear()
  }

  get size(): number {
    return this.map.size
  }
}
