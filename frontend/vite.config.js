import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue({
    // scoped CSS 의 data-v-<id> 를 '파일 경로'만으로 만든다. 기본값은 production 에서
    // hash(경로 + 소스)라, git 상 변경이 없어도 체크아웃 개행(CRLF/LF 혼합)이 다르면 id 와
    // dist 청크가 빌드마다 흔들려 frontend_dist 에 가짜 변경이 생겼다. 경로가 파일마다
    // 유일하므로 스타일 격리는 그대로다.
    features: { componentIdGenerator: 'filepath' },
  })],
  base: './',  // 상대 경로 (QWebEngineView 로컬 로드용)
  build: {
    outDir: '../frontend_dist',  // 프로젝트 루트/frontend_dist에 빌드
    emptyOutDir: true,
  },
})
