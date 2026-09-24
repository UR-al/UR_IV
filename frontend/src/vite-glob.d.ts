/**
 * Vite 의 `import.meta.glob` — tsconfig 가 `types: []` 라 vite/client 타입을 싣지 않는다.
 * 테스트가 SFC 원문을 한꺼번에 읽을 때(`{ query: '?raw', import: 'default', eager: true }`)만 쓰므로
 * 그 모양만 선언한다. 런타임 코드는 이걸 쓰지 않는다(Vite 가 빌드 때 정적으로 풀어 버린다).
 */
interface ImportMeta {
  glob<T = unknown>(
    pattern: string | string[],
    options?: { query?: string; import?: string; eager?: boolean },
  ): Record<string, T>
}
