/**
 * 모자이크 지우개의 '적용 전' 그림(pristine) 출처 — EditorCanvas 가 쓴다.
 *
 * 지우개는 화면에 pristine 픽셀을 칠하고, 커밋('restore')은 부모(EditorView)가 들고 있는
 * pristinePath 파일에서 같은 자리의 픽셀을 되가져온다. 둘은 같은 그림이어야 한다.
 *
 * 예전에는 캔버스가 스냅숏을 '로드한 이미지'에서 따로 떴다(효과 적용·복원 커밋 직전에만 '다음 로드
 * 한 번 유지'를 예약). 부모의 pristinePath 는 효과를 적용할 때마다 '그 효과 직전 그림'으로 앞으로
 * 가는데 캔버스 스냅숏은 첫 효과 전 원본에 머물렀다 — 효과를 두 번 적용한 뒤 첫 효과 자리를 지우면
 * 화면에선 지워졌다가 커밋 뒤 되살아났고, 효과→색 조정·효과×2→undo→redo 뒤에도 화면과 파일이 갈렸다.
 * 이제 캔버스는 스냅숏을 **pristinePath 그 파일에서만** 뜬다(prop `pristineSrc`) — 두 쪽이 한 출처다.
 *
 * `PristineSource` 는 디코드 순서만 판정한다: 마지막으로 요청한 출처의 디코드만 받아들이고,
 * 문서가 바뀌면(`reset`) 진행 중인 디코드를 버린다(옛 문서의 그림이 새 문서 위에 칠해지지 않게).
 */
export class PristineSource {
  private src = ''
  private token = 0
  private decodeFailed = false

  /**
   * 새 출처를 요청한다(빈 문자열 = 되돌릴 그림 없음). 반환한 토큰으로 디코드 결과를 `accepts` 에
   * 물어본다 — 그사이 더 새 요청이나 reset 이 있었으면 버린다.
   */
  request(src: string): number {
    this.src = src || ''
    this.decodeFailed = false
    return ++this.token
  }

  /** 이 토큰의 디코드를 스냅숏으로 받아들여도 되는가. */
  accepts(token: number): boolean {
    return token === this.token && this.src !== ''
  }

  /**
   * 이 토큰의 디코드가 실패했다(문서를 연 채 pristinePath 파일이 지워졌거나 잠겼다 등).
   * 지금 출처의 실패만 기록한다 — 옛 출처의 늦은 실패는 새 출처를 망가뜨리지 않는다.
   */
  fail(token: number): void {
    if (this.accepts(token)) this.decodeFailed = true
  }

  /** 문서가 바뀌었다 — 출처를 비우고 진행 중인 디코드를 버린다. */
  reset(): void {
    this.src = ''
    this.decodeFailed = false
    this.token++
  }

  /** 되돌릴 '적용 전' 그림이 지정돼 있는가(디코드가 끝났는지와는 별개). */
  get requested(): boolean {
    return this.src !== ''
  }

  /** 지금 출처의 디코드가 실패해 스냅숏이 영영 오지 않는다. */
  get failed(): boolean {
    return this.decodeFailed && this.src !== ''
  }

  /** 지금 출처 — 없으면 빈 문자열. */
  get current(): string {
    return this.src
  }
}

/**
 * 모자이크 지우개 한 스트로크를 어떻게 처리할지.
 *  - `'paint'`: 스냅숏(pristinePath 의 그림) 픽셀을 칠하고 복원 영역을 기록한다.
 *  - `'mark'`: 칠하지 않고 영역만 기록해 커밋을 부모(commitRestore)에 맡긴다.
 *    · 되돌릴 그림이 아예 없다(효과를 아직 적용하지 않았다) → 부모가 '되돌릴 이전 상태가 없습니다'를
 *      알리고 정리한다.
 *    · pristinePath 가 있는데 그 파일을 디코드하지 못했다(`decodeFailed` — 연 채 지워짐·잠김 등) →
 *      부모가 'restore' 를 보내 백엔드가 직접 읽는다. 없으면 '복원 원본 이미지를 찾을 수 없습니다'
 *      토스트, 읽히면 복원 결과가 화면에 온다. 예전엔 'skip' 이라 지우개가 아무 반응 없이 멈췄다.
 *  - `'skip'`: 스냅숏이 아직 디코드 중이거나 현재 이미지와 크기가 다르다(회전·자르기 뒤) — 아무것도
 *    하지 않는다. 백엔드도 크기가 다르면 복원을 거절한다.
 */
export type RestoreStrokeMode = 'paint' | 'mark' | 'skip'

export function restoreStrokeMode(
  requested: boolean,
  snapshot: { width: number; height: number } | null,
  image: { width: number; height: number },
  decodeFailed = false,
): RestoreStrokeMode {
  if (!requested) return 'mark'
  if (!snapshot) return decodeFailed ? 'mark' : 'skip'
  if (snapshot.width !== image.width || snapshot.height !== image.height) return 'skip'
  return 'paint'
}
