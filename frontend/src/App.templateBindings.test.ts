import { describe, expect, it } from 'vitest'
import { compileScript, compileTemplate, parse } from '@vue/compiler-sfc'
import appSource from './App.vue?raw'

/**
 * 템플릿이 쓰는 이름이 `<script setup>` 에 전부 있는지 — 정적 가드.
 *
 * App.vue 를 composable·컴포넌트로 나눌 때 가장 흔한 사고가 "템플릿이 쓰는 이름을 setup 에서
 * 꺼내지(destructure) 않은 것"이다. Vite 빌드(esbuild)도 vue-tsc 도 이걸 못 잡는다 — 빌드는 통과하고
 * 런타임에 그 버튼·표시만 조용히 죽는다(`undefined` 경고 한 줄). 여기서는 실제 SFC 컴파일러로
 * 템플릿을 컴파일해, setup 바인딩(`$setup.x`)·props(`$props.x`)로 풀리지 않고 인스턴스 폴백
 * (`_ctx.x`)으로 떨어지는 식별자를 찾는다. `$event`·`$emit` 같은 `$` 이름과 Vue 가 템플릿에서
 * 허용하는 전역(Math·parseFloat …)은 컴파일러가 알아서 빼 준다.
 */
const SOURCES = import.meta.glob<string>('./**/*.vue', { query: '?raw', import: 'default', eager: true })

/**
 * main.js 가 전역으로 등록하는 컴포넌트(`app.component('Icon', …)`)와 vue-router 가 주는 것 — 이 이름만
 * setup 밖(resolveComponent)에서 찾아도 된다. 전역 지시자는 없다(v-* 는 전부 setup 에서 import).
 */
const GLOBAL_COMPONENTS = new Set(['Icon', 'router-view', 'RouterView', 'router-link', 'RouterLink'])

function compiledTemplate(source: string, filename: string): string | null {
  const { descriptor, errors } = parse(source, { filename })
  if (errors.length) throw new Error(`${filename}: ${errors.map(e => String(e)).join('; ')}`)
  if (!descriptor.template || !descriptor.scriptSetup) return null
  const script = compileScript(descriptor, { id: 'template-bindings', inlineTemplate: false })
  const template = compileTemplate({
    source: descriptor.template.content,
    filename,
    id: 'template-bindings',
    compilerOptions: { bindingMetadata: script.bindings, prefixIdentifiers: true },
  })
  if (template.errors.length) throw new Error(`${filename}: ${template.errors.map(e => String(e)).join('; ')}`)
  return template.code
}

/**
 * setup 에 없는 템플릿 이름 — `_ctx.X` 로 떨어진 식별자, 전역이 아닌데 resolveComponent 로 찾는
 * 컴포넌트(import 를 빠뜨리면 Vue 는 경고 한 줄 뒤 이름 그대로의 빈 요소를 그린다), resolveDirective 로
 * 찾는 지시자(`v-foo` 를 import 하지 않음).
 */
function unresolvedTemplateNames(source: string, filename: string): string[] {
  const code = compiledTemplate(source, filename)
  if (code === null) return []
  const names = new Set<string>()
  for (const match of code.matchAll(/\b_ctx\.([A-Za-z_$][\w$]*)/g)) {
    if (!match[1].startsWith('$')) names.add(match[1])
  }
  // 컴파일 결과는 헬퍼를 `_resolveComponent` 처럼 밑줄 별칭으로 부른다
  for (const match of code.matchAll(/(?<![\w$])_?resolveComponent\("([^"]+)"/g)) {
    if (!GLOBAL_COMPONENTS.has(match[1])) names.add(`<${match[1]}>`)
  }
  for (const match of code.matchAll(/(?<![\w$])_?resolveDirective\("([^"]+)"/g)) names.add(`v-${match[1]}`)
  return [...names].sort()
}

describe('template bindings guard', () => {
  it('catches a template name that setup never defined (the checker itself is not vacuous)', () => {
    const broken = [
      '<template><button @click="save">{{ label }} {{ missing }}</button></template>',
      '<script setup lang="ts">',
      "import { ref } from 'vue'",
      "const label = ref('x')",
      'function save() {}',
      '</script>',
    ].join('\n')
    expect(unresolvedTemplateNames(broken, 'Broken.vue')).toEqual(['missing'])
  })

  it('catches a child component or directive that was never imported (globals like <Icon> are fine)', () => {
    const broken = [
      '<template><div v-scroll-memory="\'a\'"><LoraStackCard /><Icon name="x" /><router-view />{{ n }}</div></template>',
      '<script setup lang="ts">',
      'const n = 1',
      '</script>',
    ].join('\n')
    expect(unresolvedTemplateNames(broken, 'Broken.vue')).toEqual(['<LoraStackCard>', 'v-scroll-memory'])
  })

  it('sees destructured composable results, props, imports and v-for aliases as defined', () => {
    const ok = [
      '<template><Child v-for="item in items" :key="item" :v="item + size + total" @x="bump" /></template>',
      '<script setup lang="ts">',
      "import { reactive } from 'vue'",
      "import Child from './Child.vue'",
      'const props = defineProps<{ size: number }>()',
      'const { items, total, bump } = reactive({ items: [1], total: 1, bump() {} })',
      'void props',
      '</script>',
    ].join('\n')
    expect(unresolvedTemplateNames(ok, 'Ok.vue')).toEqual([])
  })

  it('App.vue template names are all defined in its script setup', () => {
    expect(unresolvedTemplateNames(appSource, 'App.vue')).toEqual([])
  })

  it('every <script setup> component resolves all of its template names', () => {
    const offenders: Record<string, string[]> = {}
    for (const [path, source] of Object.entries(SOURCES)) {
      const missing = unresolvedTemplateNames(source, path)
      if (missing.length) offenders[path] = missing
    }
    expect(Object.keys(SOURCES).length).toBeGreaterThan(20)
    expect(offenders).toEqual({})
  })
})
