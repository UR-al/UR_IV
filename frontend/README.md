# UR_IV 프론트엔드 (Vue 3 + Vite)

PyQt6 창 안의 QWebEngineView(데스크톱)와 `web_main_ui.py`(브라우저) 가 같은 빌드를 띄운다.
Python 쪽 계약·구조는 저장소 루트의 `CLAUDE.md` 를 본다.

모든 명령은 이 폴더(`frontend/`)에서 실행한다.

| 명령 | 하는 일 |
|---|---|
| `npm run build` | `../frontend_dist` 로 빌드한다(`emptyOutDir`). **Vue 를 고치면 반드시 다시 빌드** — 앱은 dist 만 띄운다. |
| `npm run test` | vitest — `src/**/*.test.ts` 만(순수 로직·SSR 렌더). |
| `npm run test:node` | `src/studio/*.test.mjs` (node:test 러너 — vitest 범위 밖). |
| `npm run type-check` | `vue-tsc --noEmit`. Vite 빌드는 타입을 검사하지 않으므로 따로 돌린다(베이스라인 0 errors). |
| `npm run dev` | Vite 개발 서버. 백엔드 transport 가 없으면 `src/bridge.js` 의 목(mock)으로 뜬다 — 화면 확인용. |

테마·화면 미리보기(가상 데이터 브리지)는 `dev/THEME-PREVIEW.md` 를 따른다.

정적 자산 폴더(`public/`)는 두지 않는다 — 아이콘은 `src/components/Icon.vue` 의 인라인 SVG 를 쓴다.
