import { expect, it } from 'vitest'
import source from './SettingsView.vue?raw'

/**
 * 업데이트 패널이 실패로 굳지 않게(UR-02): 실패한 초기화는 캐시하지 않지만, 실패 상태에선 패널 버튼이
 * 모두 비활성이라 다시 시도할 계기가 화면 재진입뿐이다. keep-alive 화면이라 onMounted 는 첫 방문에만
 * 돌므로 onActivated 에서도 부른다(성공한 초기화는 캐시라 아무 일도 안 한다).
 */
const script = source.slice(source.indexOf('<script setup'), source.indexOf('</script>'))

it('다시 열릴 때(onActivated)도 업데이트 초기화를 다시 시도한다', () => {
  const hooks = [...script.matchAll(/onActivated\(\(\) => \{([\s\S]*?)\n\}\)/g)].map((m) => m[1])
  expect(hooks.some((body) => /\binitialiseAppUpdates\(false\)/.test(body))).toBe(true)
})
