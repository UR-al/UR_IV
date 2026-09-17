import { ref } from 'vue'
import { getBackend } from '../bridge.js'

/** One outstanding write; late acknowledgements cannot claim a newer draft saved. */
export function createSchemaAutosave() {
  const state = ref<'idle' | 'pending' | 'saving' | 'saved' | 'error'>('idle')
  const error = ref('')
  let pending: string | null = null
  let active = false
  let closed = false
  let serial = 0
  let debounce: ReturnType<typeof setTimeout> | undefined
  let timeout: ReturnType<typeof setTimeout> | undefined
  let backend: any = null

  function queue(text: string) {
    if (closed) return
    pending = text; error.value = ''; state.value = 'pending'
    clearTimeout(debounce)
    debounce = setTimeout(flush, 400)
  }
  function flush() {
    clearTimeout(debounce)
    if (active || pending === null) return
    const text = pending
    pending = null; active = true; state.value = 'saving'; error.value = ''
    const token = ++serial
    function finish(problem = '') {
      if (token !== serial) return
      ++serial; clearTimeout(timeout); active = false
      const newer = pending !== null && pending !== text
      if (problem) {
        if (pending === null) pending = text
        error.value = problem; state.value = 'error'
      } else {
        if (pending === text) pending = null
        state.value = pending === null ? 'saved' : 'pending'
      }
      if (newer) flush()
    }
    timeout = setTimeout(() => finish('자동 저장 응답이 없습니다. 입력은 유지됩니다. 연결 확인 후 다시 저장하세요.'), 12000)
    function send(host: any) {
      if (token !== serial) return
      try {
        if (!host?.saveChatSchemaDraft) throw Error('자동 저장 연결을 사용할 수 없습니다. 업데이트한 앱을 다시 시작하세요.')
        backend = host
        host.saveChatSchemaDraft(text, (raw: string) => {
          if (token !== serial) return
          try {
            const reply = JSON.parse(raw)
            if (!reply.ok || reply.schemaText !== text) throw Error(reply.error || '저장한 내용을 확인하지 못했습니다.')
            finish()
          } catch (problem) { finish(problem instanceof Error ? problem.message : '자동 저장 오류') }
        })
      } catch (problem) { finish(problem instanceof Error ? problem.message : '자동 저장 오류') }
    }
    if (backend) send(backend)
    else getBackend().then(send).catch(problem => finish(String(problem)))
  }
  function close() { closed = true; flush() }
  return { state, error, queue, flush, close }
}
