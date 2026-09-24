import { reactive } from 'vue'
import { getStudioClient, replyData, type StudioClient } from '../studio/client'
import { requestAction } from './widgetStore.js'
import { openExternalUrl } from '../utils/externalUrl'
import type { ShowToastPayload } from '../types/bridge'

export interface AppUpdateSnapshot {
  ok: boolean
  repository: string
  repositoryUrl: string
  releasesUrl: string
  currentVersion: string
  currentDisplay: string
  currentRevision: string
  developmentBuild: boolean
  mode: 'git' | 'manual'
  latestVersion: string
  tagName: string
  releaseName: string
  releaseUrl: string
  publishedAt: string
  notes: string
  updateAvailable: boolean
  notificationAvailable: boolean
  skipped: boolean
  autoCheck: boolean
  shouldAutoCheck: boolean
  lastCheckedAt: string
  busy: boolean
  busyAction: string
  canInstall: boolean
  installReason: string
  nativeOperations: boolean
  lastResult: {
    ok?: boolean
    message?: string
    tagName?: string
    finishedAt?: string
  }
  statusMessage: string
  available: boolean
}

export const appUpdateState = reactive<AppUpdateSnapshot>({
  ok: true,
  repository: 'UR-al/UR_IV',
  repositoryUrl: 'https://github.com/UR-al/UR_IV',
  releasesUrl: 'https://github.com/UR-al/UR_IV/releases',
  currentVersion: '',
  currentDisplay: '',
  currentRevision: '',
  developmentBuild: false,
  mode: 'manual',
  latestVersion: '',
  tagName: '',
  releaseName: '',
  releaseUrl: '',
  publishedAt: '',
  notes: '',
  updateAvailable: false,
  notificationAvailable: false,
  skipped: false,
  autoCheck: true,
  shouldAutoCheck: false,
  lastCheckedAt: '',
  busy: false,
  busyAction: '',
  canInstall: false,
  installReason: '',
  nativeOperations: false,
  lastResult: {},
  statusMessage: '',
  available: false,
})

let client: StudioClient | null = null
let initialisePromise: Promise<void> | null = null
let disconnect: (() => void) | null = null
let autoCheckScheduled = false
let activeSilent = false
let notifiedVersion = ''

function mapping(value: unknown): Record<string, any> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, any>
    : {}
}

function applySnapshot(value: unknown): void {
  const raw = mapping(value)
  if (!Object.keys(raw).length) return
  for (const key of Object.keys(appUpdateState) as Array<keyof AppUpdateSnapshot>) {
    if (key in raw && key !== 'statusMessage' && key !== 'available') {
      ;(appUpdateState as any)[key] = raw[key]
    }
  }
  appUpdateState.available = true
}

function toast(type: ShowToastPayload['type'], msg: string): void {
  requestAction('show_toast', { type, msg })
}

function maybeNotifyRelease(): void {
  if (!appUpdateState.notificationAvailable || !appUpdateState.latestVersion) return
  if (notifiedVersion === appUpdateState.latestVersion) return
  notifiedVersion = appUpdateState.latestVersion
  toast('info', `AI Studio Pro v${appUpdateState.latestVersion} 업데이트가 있습니다. 설정에서 패치노트를 확인하세요.`)
}

function maybeNotifyLastResult(): void {
  const result = appUpdateState.lastResult || {}
  const receiptId = String(result.finishedAt || '')
  if (!receiptId || window.localStorage.getItem('appUpdate.seenReceipt') === receiptId) return
  window.localStorage.setItem('appUpdate.seenReceipt', receiptId)
  toast(result.ok ? 'success' : 'error', String(result.message || (result.ok ? '앱 업데이트가 완료되었습니다.' : '앱 업데이트를 완료하지 못했습니다.')))
}

function eventMessage(data: Record<string, any>): string {
  const update = mapping(data.update)
  const result = mapping(data.result)
  const error = mapping(data.error)
  return String(update.message || result.message || data.message || error.message || '')
}

function consumeUpdateEvent(event: any): void {
  const data = mapping(event?.data)
  applySnapshot(data.snapshot)
  const action = String(data.action || '')
  const message = eventMessage(data)
  if (message) appUpdateState.statusMessage = message
  if (['accepted', 'started', 'progress'].includes(String(event?.type || ''))) {
    appUpdateState.busy = true
    appUpdateState.busyAction = action
    return
  }
  if (event?.type === 'completed') {
    appUpdateState.busy = false
    appUpdateState.busyAction = ''
    if (activeSilent && action === 'check') maybeNotifyRelease()
    else if (!activeSilent && message) toast('success', message)
    activeSilent = false
    return
  }
  if (event?.type === 'error') {
    appUpdateState.busy = false
    appUpdateState.busyAction = ''
    if (!activeSilent) toast('error', message || '업데이트 작업을 완료하지 못했습니다.')
    activeSilent = false
  }
}

async function performInitialise(scheduleAutomatic: boolean): Promise<void> {
  try {
    client = await getStudioClient()
    if (!client.supports('app_update.snapshot')) {
      appUpdateState.available = false
      return
    }
    if (!disconnect) disconnect = client.subscribe('app_update', consumeUpdateEvent)
    const reply = await client.invoke('app_update.snapshot', {})
    applySnapshot(replyData(reply))
    appUpdateState.nativeOperations = client.supports('app_update.execute')
    if (scheduleAutomatic) {
      maybeNotifyLastResult()
      maybeNotifyRelease()
    }
    if (
      scheduleAutomatic
      && !autoCheckScheduled
      && appUpdateState.shouldAutoCheck
      && client.supports('app_update.execute')
    ) {
      autoCheckScheduled = true
      window.setTimeout(() => {
        void runAppUpdateAction('check', {}, { silent: true }).catch(() => {})
      }, 3500)
    }
  } catch (error) {
    appUpdateState.available = false
    appUpdateState.statusMessage = error instanceof Error ? error.message : String(error)
  }
}

export function initialiseAppUpdates(scheduleAutomatic = true): Promise<void> {
  if (!initialisePromise) initialisePromise = performInitialise(scheduleAutomatic)
  else if (scheduleAutomatic) {
    void initialisePromise.then(() => {
      maybeNotifyLastResult()
      maybeNotifyRelease()
      if (!autoCheckScheduled && appUpdateState.shouldAutoCheck && appUpdateState.nativeOperations) {
        autoCheckScheduled = true
        window.setTimeout(() => {
          void runAppUpdateAction('check', {}, { silent: true }).catch(() => {})
        }, 3500)
      }
    })
  }
  return initialisePromise
}

export async function runAppUpdateAction(
  action: 'check' | 'configure' | 'skip' | 'install',
  payload: Record<string, unknown> = {},
  options: { silent?: boolean } = {},
): Promise<void> {
  await initialiseAppUpdates(false)
  if (!client?.supports('app_update.execute')) throw new Error('현재 환경에서는 앱 업데이트 작업을 사용할 수 없습니다.')
  activeSilent = Boolean(options.silent)
  appUpdateState.busy = true
  appUpdateState.busyAction = action
  appUpdateState.statusMessage = action === 'check'
    ? 'GitHub 릴리스를 확인하는 중입니다.'
    : action === 'install'
      ? '업데이트를 안전하게 준비하는 중입니다.'
      : '업데이트 설정을 저장하는 중입니다.'
  try {
    const reply = await client.invoke('app_update.execute', { action, payload })
    const data = mapping(replyData(reply))
    if (data.snapshot) applySnapshot(data.snapshot)
    if (reply.status === 'ok') {
      appUpdateState.busy = false
      appUpdateState.busyAction = ''
    }
  } catch (error) {
    appUpdateState.busy = false
    appUpdateState.busyAction = ''
    activeSilent = false
    appUpdateState.statusMessage = error instanceof Error ? error.message : String(error)
    if (!options.silent) toast('error', appUpdateState.statusMessage)
    throw error
  }
}

export function openAppRelease(): void {
  const url = appUpdateState.releaseUrl || appUpdateState.releasesUrl
  // 웹 모드면 이 브라우저에서 연다(호스트 PC 에서 열리면 원격 기기에선 안 보인다).
  if (url) openExternalUrl(url)
}
