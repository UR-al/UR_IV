/**
 * 테스트 전용 — SFC 원문(`?raw`)을 브라우저용(비 SSR) 컴포넌트로 컴파일한다.
 *
 * vitest 의 environment 'node' 는 `.vue` import 를 SSR 모드로 변환해서, 그대로 마운트하면
 * `useSSRContext()` 가 비어 setup 이 죽는다. 그래서 기존 컴포넌트 테스트(AutomationPanel ·
 * CharacterPresetModal)처럼 원문을 compileScript(inlineTemplate) → CommonJS 로 옮겨 실행하고,
 * 컴포넌트가 import 하는 모듈은 `modules` 로 직접 넘긴다(목록에 없는 import 는 바로 실패 — 조용히
 * 진짜 브리지에 닿지 않게). 런타임 코드는 이 파일을 import 하지 않는다.
 */
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import ts from 'typescript'

export function compileSfc(source: string, id: string, modules: Record<string, unknown>, vue: unknown = Vue): any {
  const { descriptor, errors } = parse(source)
  if (errors.length) throw new Error(errors.map(e => String(e)).join('; '))
  const script = compileScript(descriptor, { id, inlineTemplate: true })
  const code = ts.transpileModule(script.content, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText
  const exports: { default?: any } = {}
  new Function('require', 'exports', code)((name: string) => {
    if (name === 'vue') return vue
    if (name in modules) return modules[name]
    throw Error(`Unexpected component dependency: ${name}`)
  }, exports)
  return exports.default
}

/** `.vue` 자식 컴포넌트 자리에 넣는 모듈 — 받은 속성을 그대로 드러내는 빈 요소(`tag`)를 그린다. */
export function stubComponentModule(tag: string) {
  return {
    __esModule: true,
    default: Vue.defineComponent({
      name: `Stub_${tag}`,
      inheritAttrs: false,
      setup: (_props, { attrs }) => () => Vue.h(tag, attrs),
    }),
  }
}
