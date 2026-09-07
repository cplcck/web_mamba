import { buildInspectorModel, renderInspector } from './inspector'
import { SchemaError, validateDocument, type CaptureDocument, type ExpectedIdentity } from './schema'
import { validatePublicBundle, type PublicBundle } from './public-data'
import type { ExplorerSelection } from './explorer'
import { entityLabel } from './entity-label'

type Scenario = 'prefill' | 'decode'

export class DataLoadError extends Error {
  readonly path: string
  constructor(path: string, message: string) {
    super(`${path}: ${message}`)
    this.name = 'DataLoadError'
    this.path = path
  }
}

async function sourceText(fetcher: typeof fetch, url: URL, path: string): Promise<string> {
  let response: Response
  try {
    response = await fetcher(url)
  } catch (error) {
    if (error instanceof Error) throw new DataLoadError(path, error.message)
    throw error
  }
  if (!response.ok) throw new DataLoadError(path, `request failed with HTTP ${response.status}`)
  try {
    return await response.text()
  } catch (error) {
    if (error instanceof Error) throw new DataLoadError(path, `response read failed (${error.message})`)
    throw error
  }
}

export function validateLoadedDocument(value: unknown, expected: ExpectedIdentity): CaptureDocument {
  try {
    return validateDocument(value, expected)
  } catch (error) {
    if (error instanceof Error) throw new DataLoadError('capture document', error.message)
    throw error
  }
}

export async function loadPublicData(fetcher: typeof fetch = globalThis.fetch, baseHref = document.baseURI): Promise<PublicBundle> {
  const load = (name: string): Promise<string> => sourceText(fetcher, new URL(`data/${name}.json`, baseHref), `data/${name}.json`)
  const [manifest, model, provenance, validation, prefill, decode] = await Promise.all(
    [load('manifest'), load('model'), load('provenance'), load('validation'), load('prefill'), load('decode')],
  )
  // Fixed paths never come from untrusted JSON. The boundary hashes original source bytes.
  try {
    return await validatePublicBundle({ manifest, model, provenance, validation, prefill, decode })
  } catch (error) {
    if (error instanceof SchemaError) throw new DataLoadError(error.path, error.detail)
    throw error
  }
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className !== undefined) node.className = className
  return node
}

function loadStylesheet(): void {
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href = new URL('./styles.css', import.meta.url).href
  document.head.append(link)
}

function loadingSkeleton(explorer: HTMLElement, inspector: HTMLElement): void {
  const placeholder = (className: string): HTMLElement => {
    const node = element('div', className)
    node.setAttribute('aria-hidden', 'true')
    return node
  }
  const breadcrumbs = placeholder('breadcrumbs loading-skeleton')
  breadcrumbs.append(placeholder('skeleton-bar'))
  const hierarchy = placeholder('hierarchy loading-skeleton skeleton-hierarchy')
  for (let row = 0; row < 4; row++) {
    const item = placeholder('skeleton-hierarchy__row')
    item.append(placeholder('skeleton-bar'))
    hierarchy.append(item)
  }
  const graph = placeholder('operator-graph loading-skeleton')
  const flow = placeholder('skeleton-flow')
  for (let row = 0; row < 3; row++) {
    const node = placeholder('skeleton-flow__node')
    node.append(placeholder('skeleton-bar'), placeholder('skeleton-bar'))
    flow.append(node)
  }
  graph.append(flow)
  explorer.append(breadcrumbs, hierarchy, graph)
  const tensors = placeholder('loading-skeleton skeleton-tensors')
  for (let section = 0; section < 3; section++) {
    const group = placeholder('tensor-section')
    group.append(placeholder('skeleton-bar'))
    const tensor = placeholder('tensor-row')
    for (let row = 0; row < 4; row++) {
      const fields = placeholder('skeleton-tensor-fields')
      fields.append(placeholder('skeleton-bar'), placeholder('skeleton-bar'))
      tensor.append(fields)
    }
    group.append(tensor)
    tensors.append(group)
  }
  inspector.append(tensors)
}

function shell(root: HTMLElement, loading = true) {
  root.replaceChildren()
  const skip = element('a', 'skip-link'); skip.href = '#workspace'; skip.textContent = '본문으로 건너뛰기'
  const app = element('div', 'app-shell')
  const topbar = element('header', 'topbar')
  const identity = element('div', 'topbar__identity')
  const eyebrow = element('p', 'topbar__eyebrow'); eyebrow.textContent = 'MAMBA 130M / TRACE INSTRUMENT'
  const title = element('h1'); title.textContent = 'Every-level inspector'
  identity.append(eyebrow, title)
  const controls = element('fieldset', 'scenario-switch')
  const legend = element('legend'); legend.textContent = 'scenario'
  controls.append(legend)
  const buttons = {} as Record<Scenario, HTMLButtonElement>
  for (const scenario of ['prefill', 'decode'] as const) {
    const button = element('button', 'scenario-switch__button'); button.type = 'button'; button.dataset.scenario = scenario
    button.disabled = true
    button.textContent = scenario.toUpperCase(); button.setAttribute('aria-pressed', String(scenario === 'prefill'))
    controls.append(button); buttons[scenario] = button
  }
  const status = element('p', 'topbar__status'); status.role = 'status'; status.textContent = '검증된 캡처 데이터를 불러오는 중입니다.'
  const toggle = element('button', 'action-button inspector-toggle'); toggle.type = 'button'; toggle.textContent = '선택 정보'; toggle.disabled = true
  toggle.setAttribute('aria-haspopup', 'dialog'); toggle.setAttribute('aria-controls', 'inspector-dialog'); toggle.setAttribute('aria-expanded', 'false')
  const actions = element('div', 'topbar__actions'); actions.append(controls, toggle)
  topbar.append(identity, actions, status)
  const workspace = element('main', 'workspace'); workspace.id = 'workspace'; workspace.tabIndex = -1
  workspace.setAttribute('aria-label', '모델 탐색과 tensor 검사')
  skip.addEventListener('click', event => { event.preventDefault(); workspace.focus() })
  const explorer = element('section', 'explorer'); explorer.setAttribute('aria-label', '모델 계층과 연산 흐름')
  const inspector = element('aside', 'inspector'); inspector.id = 'tensor-inspector'; inspector.tabIndex = -1; inspector.setAttribute('aria-label', '선택한 항목의 tensor inspector')
  const inspectorTitle = element('p', 'inspector__loading'); inspectorTitle.textContent = '선택 항목을 기다리는 중입니다.'; inspector.append(inspectorTitle)
  if (loading) loadingSkeleton(explorer, inspector)
  const pane = element('div', 'inspector-pane')
  const toolbar = element('div', 'inspector-toolbar')
  const label = element('p'); label.id = 'inspector-label'; label.textContent = '선택 정보'
  const close = element('button', 'action-button inspector-close'); close.type = 'button'; close.textContent = '닫기'; close.autofocus = true
  toolbar.append(label, close); pane.append(toolbar, inspector)
  const dialog = element('dialog', 'inspector-dialog'); dialog.id = 'inspector-dialog'; dialog.setAttribute('aria-labelledby', label.id)
  workspace.append(explorer, pane)
  app.append(topbar, workspace, dialog); root.append(skip, app)
  return { explorer, inspector, status, buttons, pane, dialog, close, toggle, workspace }
}

function connectInspectorPane(refs: ReturnType<typeof shell>): (activate: boolean) => void {
  const compact = matchMedia('(max-width: 1024px)')
  let returnFocus: HTMLElement | SVGElement | null = null
  const open = (activate: boolean): void => {
    const active = document.activeElement
    if (activate && (active instanceof HTMLElement || active instanceof SVGElement) && refs.explorer.contains(active)) returnFocus = active
    if (compact.matches) {
      if (!refs.dialog.open) { refs.dialog.showModal(); refs.toggle.setAttribute('aria-expanded', 'true') }
      if (activate) refs.inspector.scrollTop = 0
      refs.close.focus({ preventScroll: true })
    }
  }
  refs.close.addEventListener('click', () => refs.dialog.close())
  refs.dialog.addEventListener('close', () => {
    refs.toggle.setAttribute('aria-expanded', 'false')
    if (compact.matches) {
      const target = returnFocus?.isConnected ? returnFocus : refs.explorer.querySelector<HTMLElement>('.breadcrumbs [aria-current="page"]')
      target?.focus({ preventScroll: true })
    }
  })
  refs.toggle.addEventListener('click', () => open(false))
  const reflow = (): void => {
    const focused = refs.pane.contains(document.activeElement)
    if (compact.matches) { refs.dialog.append(refs.pane); if (focused) open(false) }
    else { if (refs.dialog.open) refs.dialog.close(); refs.workspace.append(refs.pane) }
  }
  compact.addEventListener('change', reflow); reflow()
  refs.toggle.disabled = false
  return open
}

function renderFailure(root: HTMLElement, error: DataLoadError, retry: () => void): void {
  const refs = shell(root, false)
  refs.status.textContent = '데이터 검증에 실패했습니다.'
  const alert = element('div', 'data-error'); alert.role = 'alert'
  const heading = element('h2'); heading.textContent = '데이터 검증에 실패했습니다.'
  const detail = element('p'); detail.textContent = error.message
  const action = element('button', 'action-button'); action.type = 'button'; action.textContent = '다시 불러오기'; action.addEventListener('click', retry)
  alert.append(heading, detail, action); refs.explorer.replaceChildren(alert)
}

async function start(root: HTMLElement): Promise<void> {
  const refs = shell(root)
  const data = await loadPublicData()
  refs.status.textContent = `${data.manifest.modelId} · capture identities validated`
  const { createExplorer } = await import('./explorer')
  const openInspector = connectInspectorPane(refs)
  const select = (selection: ExplorerSelection, activate: boolean): void => {
    refs.buttons.prefill.setAttribute('aria-pressed', String(selection.scenario === 'prefill'))
    refs.buttons.decode.setAttribute('aria-pressed', String(selection.scenario === 'decode'))
    const model = buildInspectorModel(data.documents[selection.scenario], selection.entityId)
    refs.status.textContent = `선택됨 · ${entityLabel(model.entity)} · ${selection.scenario}`
    refs.status.setAttribute('aria-description', selection.entityId)
    renderInspector(refs.inspector, model, data)
    if (activate) openInspector(true)
  }
  const handle = createExplorer(refs.explorer, data.documents, select)
  for (const scenario of ['prefill', 'decode'] as const) {
    refs.buttons[scenario].addEventListener('click', () => handle.setScenario(scenario))
    refs.buttons[scenario].disabled = false
  }
}

async function launch(root: HTMLElement): Promise<void> {
  try { await start(root) }
  catch (error) {
    if (error instanceof DataLoadError) renderFailure(root, error, () => void launch(root))
    else throw error
  }
}

if (typeof document !== 'undefined') {
  loadStylesheet()
  const root = document.querySelector<HTMLElement>('#app')
  if (root !== null) void launch(root)
}
