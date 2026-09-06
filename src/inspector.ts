// allow: SIZE_OK - task9 owns the complete inspector in this file; shared ownership forbids a new product module.
import { evalFormula, SchemaError, storageAccounting, type CaptureDocument, type Entity, type Tensor } from './schema'
import type { PublicBundle } from './public-data'

export const INSPECTOR_SECTION_KINDS = ['inputs', 'outputs', 'weights'] as const
export type InspectorSectionKind = (typeof INSPECTOR_SECTION_KINDS)[number]

type InspectorTensor = {
  readonly id: string
  readonly name: string
  readonly role: string
  readonly dtype: string
  readonly typeBlockSize: number
  readonly nativeShape: string
  readonly strides: string
  readonly logicalShape: string
  readonly numel: string
  readonly logicalBytes: string
  readonly logicalIEC: string
  readonly ggmlNbytes: string
  readonly requiredAllocBytes: string
  readonly bufferCapacity: string
  readonly backing: string
  readonly alias: string
  readonly producer: string
  readonly consumers: string
  readonly provenance: string
  readonly classification: string
  readonly captureId: string
  readonly observationEpoch: number
  readonly ggufPayload: Tensor['ggufPayload']
  readonly allocatorSlotBytes: number | null
  readonly allocatorSlotReason: string | undefined
  readonly allocationStatus: 'view' | 'unobserved' | 'observed'
}

export type InspectorSection = {
  readonly kind: InspectorSectionKind
  readonly label: string
  readonly count: number
  readonly empty: boolean
  readonly emptyMessage: string | null
  readonly tensors: readonly InspectorTensor[]
  readonly totalLogicalBytes: number
  readonly totalRequiredAllocBytes: number | null
  readonly uniqueGgufPayloadBytes: number
  readonly uniqueRuntimeRanges: number
}

export type InspectorModel = {
  readonly document: CaptureDocument
  readonly entity: Entity
  readonly sections: readonly InspectorSection[]
  readonly accounting: ReturnType<typeof storageAccounting>
}

export type ByteDisplay = {
  readonly exact: string
  readonly iec: string
}

const SECTION_LABELS = {
  inputs: 'Inputs',
  outputs: 'Outputs',
  weights: 'Weights',
} as const satisfies Record<InspectorSectionKind, string>

const SECTION_IDS = {
  inputs: 'inputTensorIds',
  outputs: 'outputTensorIds',
  weights: 'weightTensorIds',
} as const satisfies Record<InspectorSectionKind, keyof Entity>

const EMPTY_WEIGHT_MESSAGE = '이 항목은 직접 보유한 weight가 없습니다.'

export function formatIECBytes(bytes: number): ByteDisplay {
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new RangeError('bytes must be a non-negative safe integer')
  }
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'] as const
  let unitIndex = 0
  let value = bytes
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  const exact = `${bytes.toLocaleString('en-US')} B`
  const iec = unitIndex === 0 ? exact : `${Number(value.toFixed(2))} ${units[unitIndex]}`
  return { exact, iec }
}

function formatBytes(bytes: number): string {
  const { exact, iec } = formatIECBytes(bytes)
  return exact === iec ? exact : `${exact} / ${iec}`
}

function formatOptionalBytes(bytes: number | null): string {
  return bytes === null ? 'unknown — 관측되지 않음' : formatBytes(bytes)
}

function formatTuple(values: readonly number[]): string {
  return `[${values.join(', ')}]`
}

function formatLogicalShape(tensor: Tensor): string {
  return tensor.axisLabels.map((label, index) => `${label}=${tensor.logicalShape[index] ?? 'unknown'}`).join(' · ')
}

function formatRelations(ids: readonly string[]): string {
  return ids.length === 0 ? 'none' : ids.join(', ')
}

function formatTensor(tensor: Tensor, captureId: string): InspectorTensor {
  const storage = tensor.storage
  const backing = storage.bufferId === null
    ? 'unknown — runtime backing range not observed'
    : `${storage.bufferId} @ ${storage.bufferOffsetBytes ?? 0} B / ${formatIECBytes(storage.bufferBytes ?? 0).exact}`
  const alias = tensor.viewSourceId === null
    ? 'none'
    : `${tensor.viewSourceId} + ${formatIECBytes(tensor.viewOffsetBytes ?? 0).exact}`
  return {
    id: tensor.id,
    name: tensor.name,
    role: tensor.role,
    dtype: tensor.dtype,
    typeBlockSize: tensor.typeBlockSize,
    nativeShape: formatTuple(tensor.nativeShape),
    strides: formatTuple(tensor.strides),
    logicalShape: formatLogicalShape(tensor),
    numel: tensor.numel.toLocaleString('en-US'),
    logicalBytes: formatIECBytes(tensor.logicalBytes).exact,
    logicalIEC: formatIECBytes(tensor.logicalBytes).iec,
    ggmlNbytes: formatBytes(storage.ggmlNbytes),
    requiredAllocBytes: tensor.viewSourceId === null ? formatOptionalBytes(storage.requiredAllocBytes) : '해당 없음 — view는 독립 할당하지 않음',
    bufferCapacity: formatOptionalBytes(storage.bufferBytes),
    backing,
    alias,
    producer: formatRelations(tensor.producerIds),
    consumers: formatRelations(tensor.consumerIds),
    provenance: tensor.provenance,
    classification: tensor.classification,
    captureId: tensor.captureId ?? captureId,
    observationEpoch: storage.observationEpoch,
    ggufPayload: tensor.ggufPayload,
    allocatorSlotBytes: storage.allocatorSlotBytes,
    allocatorSlotReason: storage.allocatorSlotReason,
    allocationStatus: tensor.viewSourceId !== null ? 'view' : storage.requiredAllocBytes === null ? 'unobserved' : 'observed',
  }
}

function uniquePayloadBytes(tensors: readonly Tensor[]): number {
  const ranges = new Map<string, Array<readonly [number, number]>>()
  for (const tensor of tensors) {
    const payload = tensor.ggufPayload
    if (payload === undefined) continue
    const current = ranges.get(payload.fileId) ?? []
    current.push([payload.offsetBytes, payload.offsetBytes + payload.bytes])
    ranges.set(payload.fileId, current)
  }
  let total = 0
  for (const fileRanges of ranges.values()) {
    const sorted = [...fileRanges].sort(([a], [b]) => a - b)
    let end = 0
    for (const [start, finish] of sorted) {
      if (finish <= end) continue
      total += finish - Math.max(start, end)
      end = finish
    }
  }
  return total
}

function sectionFor(entity: Entity, kind: InspectorSectionKind, tensorMap: ReadonlyMap<string, Tensor>): InspectorSection {
  const ids = entity[SECTION_IDS[kind]]
  const tensors = ids.map(id => tensorMap.get(id)).filter((tensor): tensor is Tensor => tensor !== undefined)
  const totalRequiredAllocBytes = tensors.length === 0 || tensors.some(tensor => tensor.storage.requiredAllocBytes === null)
    ? null
    : tensors.reduce((total, tensor) => total + (tensor.storage.requiredAllocBytes ?? 0), 0)
  const empty = tensors.length === 0
  return {
    kind,
    label: SECTION_LABELS[kind],
    count: tensors.length,
    empty,
    emptyMessage: empty && kind === 'weights' ? EMPTY_WEIGHT_MESSAGE : empty ? '0 tensors' : null,
    tensors: [],
    totalLogicalBytes: tensors.reduce((total, tensor) => total + tensor.logicalBytes, 0),
    totalRequiredAllocBytes,
    uniqueGgufPayloadBytes: uniquePayloadBytes(tensors),
    uniqueRuntimeRanges: storageAccounting(tensors).uniqueRuntimeRanges,
  }
}

export function buildInspectorModel(document: CaptureDocument, entityId: string): InspectorModel {
  const entity = document.entities.find(candidate => candidate.id === entityId)
  if (entity === undefined) throw new Error(`Unknown inspector entity: ${entityId}`)
  const tensorMap = new Map(document.tensors.map(tensor => [tensor.id, tensor]))
  const sections = INSPECTOR_SECTION_KINDS.map(kind => ({
    ...sectionFor(entity, kind, tensorMap),
    tensors: entity[SECTION_IDS[kind]].map(id => {
      const tensor = tensorMap.get(id)
      if (tensor === undefined) throw new Error(`Unknown inspector tensor: ${id}`)
      return formatTensor(tensor, document.captureId)
    }),
  }))
  const referencedIds = new Set([...entity.inputTensorIds, ...entity.outputTensorIds, ...entity.weightTensorIds])
  const referencedTensors = document.tensors.filter(tensor => referencedIds.has(tensor.id))
  return { document, entity, sections, accounting: storageAccounting(referencedTensors) }
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className !== undefined) node.className = className
  return node
}

function renderTensor(section: HTMLElement, tensor: InspectorTensor): void {
  const article = element('article', 'tensor-row'); article.dataset.tensorId = tensor.id
  article.tabIndex = 0
  const table = element('table', 'tensor-table')
  const heading = element('caption', 'tensor-row__name')
  heading.textContent = tensor.name
  table.append(heading)
  const details = element('tbody')
  const payload = tensor.ggufPayload
  const entries: readonly [string, string][] = [
    ['id', tensor.id], ['role', tensor.role], ['dtype', tensor.dtype], ['typeBlockSize', String(tensor.typeBlockSize)], ['native ne[4]', tensor.nativeShape],
    ['native nb[4]', tensor.strides], ['logical shape', tensor.logicalShape], ['numel', tensor.numel],
    ['logical bytes', tensor.logicalBytes], ['IEC/logical', tensor.logicalIEC],
    ['ggmlNbytes', tensor.ggmlNbytes], ['requiredAllocBytes', tensor.requiredAllocBytes], ['buffer capacity', tensor.bufferCapacity],
    ['backing range', tensor.backing], ['view source / offset', tensor.alias], ['producer', tensor.producer],
    ['consumers', tensor.consumers], ['provenance', tensor.provenance], ['evidence', tensor.classification],
    ['captureId', tensor.captureId], ['observationEpoch', String(tensor.observationEpoch)],
    ['GGUF fileId', payload?.fileId ?? '해당 없음 — GGUF payload 참조 없음'],
    ['GGUF offsetBytes', payload === undefined ? '해당 없음' : formatBytes(payload.offsetBytes)],
    ['GGUF payload bytes', payload === undefined ? '해당 없음' : formatBytes(payload.bytes)],
    ['allocatorSlotBytes', 'unknown'], ['allocatorSlotReason', tensor.allocatorSlotReason ?? '정확한 allocator slot 용량은 노출되지 않음'],
  ]
  for (const [label, value] of entries) {
    const row = element('tr'); row.dataset.field = label
    const name = element('th'); name.scope = 'row'; name.textContent = label
    const content = element('td'); content.textContent = value
    if (label === 'requiredAllocBytes') content.dataset.status = tensor.allocationStatus
    row.append(name, content); details.append(row)
  }
  table.append(details); article.append(table)
  section.append(article)
}

export class EstimateError extends Error {
  readonly name = 'EstimateError'
  constructor(readonly code: 'unsupported-dimensions' | 'missing-formula' | 'observed-constant', message: string) { super(message) }
}

export function estimateTensor(tensor: Tensor, document: CaptureDocument, dimensions: Readonly<{ P: number; T: number; O: number }>) {
  const { P, T, O } = dimensions, Q = P * T
  if (![P, T, O].every(Number.isSafeInteger) || P !== 1 || T < 1 || T > 32 || (document.scenario.name === 'decode' && T !== 1) || O < 1 || O > Q) {
    throw new EstimateError('unsupported-dimensions', '지원 범위: P=1, prefill T=1..32 / decode T=1, O=1..Q, Q=P*T')
  }
  if (!tensor.shapeFormula || !tensor.logicalBytesFormula) throw new EstimateError('missing-formula', '이 tensor에는 검증된 symbolic formula가 없습니다.')
  if (tensor.formulaClassification !== 'symbolic-estimate' && Object.entries(dimensions).some(([key, value]) => document.scenario.dimensions[key] !== value)) {
    throw new EstimateError('observed-constant', '관측 상수만 있습니다. 변경된 차원의 shape를 추정하지 않습니다.')
  }
  const values = { P, T, Q, O }, shape = tensor.shapeFormula.map(formula => evalFormula(formula, values))
  const numel = shape.reduce((total, extent) => evalFormula({ op: 'mul', left: { op: 'const', value: total }, right: { op: 'const', value: extent } }, values), 1)
  return { shape, numel, logicalBytes: evalFormula(tensor.logicalBytesFormula, values), classification: 'derived' as const }
}

function panel(id: string, title: string): HTMLElement {
  const section = element('section', 'inspector-panel'); section.id = id
  const heading = element('h3'); heading.id = `${id}-heading`; heading.textContent = title
  section.setAttribute('aria-labelledby', heading.id); section.append(heading)
  return section
}

// Only typed, checked metadata reaches this text-only renderer. Arrays retain exact values.
function metadataTable(value: object): HTMLTableElement {
  const table = element('table', 'tensor-table'), body = element('tbody')
  function append(record: object, prefix: string): void {
    for (const [key, item] of Object.entries(record)) {
      const path = prefix ? `${prefix}.${key}` : key
      if (item !== null && typeof item === 'object' && !Array.isArray(item)) { append(item, path); continue }
      const row = element('tr'); row.dataset.field = path
      const label = element('th'); label.scope = 'row'; label.textContent = path
      const content = element('td'); content.textContent = item === null ? 'null — 해당 관측 없음' : typeof item === 'string' ? item : JSON.stringify(item)
      row.append(label, content); body.append(row)
    }
  }
  append(value, ''); table.append(body); return table
}

function renderCalculator(host: HTMLElement, model: InspectorModel): void {
  const section = panel('dimension-calculator', '차원 계산기 · derived symbolic estimate')
  const note = element('p'); note.textContent = '관측 실행이 아닌 추정입니다. shape / compact logical bytes만 계산하며, storage 할당·주소·수명·peak memory는 추정하지 않습니다.'
  section.append(note)
  const form = element('form', 'dimension-form'); form.noValidate = true
  const label = element('label'); label.textContent = 'tensor'
  const tensors = element('select'); tensors.name = 'tensor'; label.append(tensors); form.append(label)
  const ids = new Set(model.sections.flatMap(section => section.tensors.map(tensor => tensor.id)))
  const candidates = model.document.tensors.filter(tensor => ids.has(tensor.id))
  for (const tensor of candidates) { const option = element('option'); option.value = tensor.id; option.textContent = tensor.name; tensors.append(option) }
  const inputs = new Map<string, HTMLInputElement>()
  for (const key of ['P', 'T', 'O'] as const) {
    const label = element('label'); label.textContent = key
    const input = element('input'); input.type = 'number'; input.name = key; input.step = '1'; input.min = '1'
    input.max = String(key === 'P' || model.document.scenario.name === 'decode' ? 1 : 32)
    input.value = String(model.document.scenario.dimensions[key]); inputs.set(key, input); label.append(input); form.append(label)
  }
  const submit = element('button', 'action-button'); submit.type = 'submit'; submit.textContent = '추정 계산'; submit.disabled = candidates.length === 0
  const result = element('div', 'dimension-result'); result.role = 'status'; result.setAttribute('aria-live', 'polite')
  const calculate = (): void => {
    result.replaceChildren()
    const tensor = candidates.find(tensor => tensor.id === tensors.value)
    if (!tensor) { result.dataset.status = 'empty'; result.textContent = '선택 항목에 tensor가 없습니다.'; return }
    try {
      const estimate = estimateTensor(tensor, model.document, { P: Number(inputs.get('P')?.value), T: Number(inputs.get('T')?.value), O: Number(inputs.get('O')?.value) })
      result.dataset.status = 'derived'; result.dataset.tensorId = tensor.id
      result.append(metadataTable({ classification: estimate.classification, shape: estimate.shape, numel: estimate.numel,
        logicalBytes: formatBytes(estimate.logicalBytes), shapeFormula: tensor.shapeFormula, logicalBytesFormula: tensor.logicalBytesFormula,
        formulaClassification: tensor.formulaClassification, formulaSource: tensor.formulaSource }))
    } catch (error) {
      if (!(error instanceof EstimateError) && !(error instanceof SchemaError)) throw error
      result.dataset.status = 'unsupported'; result.textContent = error.message
    }
  }
  form.addEventListener('submit', event => { event.preventDefault(); calculate() })
  tensors.addEventListener('change', calculate)
  form.append(submit); section.append(form, result); host.append(section); calculate()
}

function renderState(host: HTMLElement, model: InspectorModel): void {
  const section = panel('recurrent-state', 'R / S · before / after')
  const state = model.document.recurrentState
  if (!state) { const note = element('p'); note.textContent = '이 문서에는 before/after state 관측이 없습니다.'; section.append(note); host.append(section); return }
  const layerLabel = element('label'); layerLabel.textContent = 'layer'
  const layerSelect = element('select'); layerSelect.name = 'state-layer'
  for (const layer of new Set(state.before.arrays.map(array => array.layer))) { const option = element('option'); option.value = String(layer); option.textContent = `layer ${layer}`; layerSelect.append(option) }
  let ancestor: Entity | undefined = model.entity
  while (ancestor && ancestor.kind !== 'block') ancestor = model.document.entities.find(entity => entity.id === ancestor?.parentId)
  const layer = model.document.entities.filter(entity => entity.kind === 'block').findIndex(entity => entity.id === ancestor?.id)
  if (layer >= 0) layerSelect.value = String(layer)
  layerLabel.append(layerSelect); section.append(layerLabel)
  const snapshots = element('div', 'state-snapshots')
  const update = (): void => {
    snapshots.replaceChildren()
    for (const [phase, snapshot] of [['before', state.before], ['after', state.after]] as const) {
      const group = element('section'); group.dataset.phase = phase
      const heading = element('h4'); heading.textContent = snapshot.label; group.append(heading)
      group.append(metadataTable({ id: snapshot.id, captureId: snapshot.captureId, position: snapshot.position, epoch: snapshot.epoch, mapping: snapshot.mapping }))
      for (const array of snapshot.arrays.filter(array => array.layer === Number(layerSelect.value))) {
        const detail = element('details'); detail.open = true; detail.dataset.family = array.family
        const summary = element('summary'); summary.textContent = `${array.family} · ${array.dtype} · ${formatTuple(array.shape)} · ${formatBytes(array.logicalBytes)}`
        detail.append(summary, metadataTable(array)); group.append(detail)
      }
      snapshots.append(group)
    }
  }
  layerSelect.addEventListener('change', update); update()
  section.append(snapshots, metadataTable({ nativeCaptureId: state.nativeCaptureId, sequenceId: state.sequenceId, continuity: state.continuity }))
  const note = element('p'); note.textContent = 'after-prefill → before-decode: 동일 native cache의 mapping / 48개 배열 byte 연속성입니다. epoch는 다를 수 있습니다. HF와 native cache의 byte 동일성을 뜻하지 않습니다.'
  section.append(note); host.append(section)
}

export function renderInspector(host: HTMLElement, model: InspectorModel, bundle?: PublicBundle): void {
  host.replaceChildren(); host.dataset.entityId = model.entity.id; host.dataset.scenario = model.document.scenario.name
  const heading = element('h2', 'inspector__title')
  heading.textContent = `${model.entity.kind} / ${model.entity.id}`
  const header = element('header', 'inspector__header')
  header.append(heading)
  const summary = element('p', 'inspector__summary')
  summary.textContent = `parent: ${model.entity.parentId ?? 'root'} · selection is validated metadata`
  header.append(summary); host.append(header)
  for (const sectionModel of model.sections) {
    const section = element('section', `tensor-section tensor-section--${sectionModel.kind}`)
    section.setAttribute('aria-labelledby', `${sectionModel.kind}-heading`)
    const title = element('h3')
    title.id = `${sectionModel.kind}-heading`
    title.textContent = `${sectionModel.label} — ${sectionModel.count} tensors`
    section.append(title)
    if (sectionModel.empty) {
      const empty = element('p', 'tensor-section__empty')
      empty.textContent = sectionModel.emptyMessage ?? '0 tensors'
      section.append(empty)
    } else {
      const totals = element('p', 'tensor-section__totals')
      totals.textContent = `logical ${formatIECBytes(sectionModel.totalLogicalBytes).exact} · unique GGUF ${formatIECBytes(sectionModel.uniqueGgufPayloadBytes).exact} · runtime ranges ${sectionModel.uniqueRuntimeRanges}`
      section.append(totals)
      for (const tensor of sectionModel.tensors) renderTensor(section, tensor)
    }
    host.append(section)
  }
  renderCalculator(host, model)
  renderState(host, model)
  const notes = panel('kernel-notes', 'Kernel 내부 / offline conversion')
  notes.dataset.classification = 'kernel-internal'
  if (model.document.kernelInternals) notes.append(metadataTable(model.document.kernelInternals))
  const explanation = element('p'); explanation.textContent = model.document.offlineConversion ?? '이 문서에는 kernel 설명이 없습니다.'
  notes.append(explanation)
  const boundary = element('p'); boundary.textContent = 'SSM_CONV / SSM_SCAN 내부 수식은 설명입니다. 별도 graph node, scalar 캡처 또는 메모리 할당이 아닙니다. A_log → -exp(A_log)는 offline 변환입니다.'
  notes.append(boundary); host.append(notes)
  if (bundle) {
    for (const [id, title, value] of [['public-model', '모델 구성', bundle.model], ['public-provenance', 'Source / runtime provenance', bundle.provenance], ['public-validation', '검증 보고서 / policy', bundle.validation]] as const) {
      const section = panel(id, title)
      const detail = element('details'), summary = element('summary'); summary.textContent = `${title} · ${bundle.manifest.captureId}`
      detail.append(summary, metadataTable(value)); section.append(detail); host.append(section)
    }
    const limitation = element('p'); limitation.textContent = '정적 보고서의 identity와 승인 상태를 확인했습니다. 브라우저가 수치 검증을 다시 실행한 결과는 아닙니다. logical bytes 합계는 peak memory가 아닙니다.'
    host.append(limitation)
  }
}
