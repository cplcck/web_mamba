// allow: SIZE_OK - task9 owns the complete inspector in this file; shared ownership forbids a new product module.
import { evalFormula, SchemaError, storageAccounting, type CaptureDocument, type Entity, type Tensor } from './schema'
import type { PublicBundle } from './public-data'
import { entityLabel } from './entity-label'

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

export function buildInspectorSummary(model: InspectorModel) {
  const sections = model.sections.map(section => {
    const activations = section.tensors.filter(tensor => tensor.role === 'activation')
    const tokens = section.tensors.filter(tensor => tensor.name === 'inp_tokens')
    const candidates = activations.length ? activations : tokens.length ? tokens : section.tensors.filter(tensor => tensor.role !== 'state')
    // A root preview represents global weights, not an arbitrary block's parameters.
    const preview = model.entity.kind === 'model' && section.kind === 'weights'
      ? candidates.filter(tensor => !tensor.name.startsWith('blk.')).slice(0, 2) : candidates.slice(0, 2)
    return { kind: section.kind, label: section.label, count: section.count, preview,
      stateCount: section.tensors.filter(tensor => tensor.role === 'state').length,
      indexCount: section.tensors.filter(tensor => tensor.role === 'index').length,
      uniqueGgufPayloadBytes: section.uniqueGgufPayloadBytes }
  })
  const states = [...new Map(model.sections.flatMap(section => section.tensors)
    .filter(tensor => tensor.role === 'state').map(tensor => [tensor.id, tensor])).values()]
  return { sections, state: { count: states.length, preview: model.entity.kind === 'model' ? [] : states.slice(0, 2) } }
}

const STAGE_ROLES: Readonly<Record<string, string>> = {
  embedding: '토큰 ID를 모델의 특징 벡터로 바꿉니다.',
  'final-normalization': '마지막 블록의 출력을 정규화합니다.',
  'final-projection': '특징 벡터를 어휘별 logits로 투영합니다.',
  normalization: '블록 입력의 크기를 정규화합니다.',
  'input-projection-split': '입력을 확장하고 상태 경로와 게이트 경로로 나눕니다.',
  'convolution-history': '이전 convolution 이력과 현재 입력을 결합합니다.',
  'dt-b-c-projection': '입력에서 시간 간격과 상태 갱신 매개변수를 계산합니다.',
  'selective-scan-state': '입력에 따라 순환 상태를 갱신하고 출력을 만듭니다.',
  'skip-gating': '직접 전달 경로와 상태 출력을 합쳐 게이트를 적용합니다.',
  'output-projection': '확장된 특징을 모델 차원으로 되돌립니다.',
  residual: '블록 입력을 더해 다음 블록으로 전달합니다.',
}

const OPERATOR_ROLES: Readonly<Record<string, string>> = {
  ADD: '입력 텐서들을 원소별로 더합니다.',
  MUL: '입력 텐서들을 원소별로 곱합니다.',
  MUL_MAT: '행렬 곱으로 입력 특징을 투영합니다.',
  RMS_NORM: '입력의 RMS 크기를 기준으로 정규화합니다.',
  GET_ROWS: '입력 인덱스에 해당하는 행을 가져옵니다.',
  RESHAPE: '값을 바꾸지 않고 텐서의 차원 구성을 바꿉니다.',
  VIEW: '기존 텐서의 일부를 새 모양으로 참조합니다.',
  TRANSPOSE: '텐서의 축 순서를 바꿉니다.',
  CONT: '텐서를 연속된 배치로 정리합니다.',
  CPY: '입력 값을 대상 텐서로 복사합니다.',
  CONCAT: '입력 텐서들을 지정된 축으로 이어 붙입니다.',
  SCALE: '각 원소에 스칼라 배율을 적용합니다.',
  UNARY: '각 원소에 단항 함수를 적용합니다.',
  GLU: '게이트로 특징의 전달량을 조절합니다.',
  SSM_CONV: '이전 이력과 현재 입력에 convolution을 적용합니다.',
  SSM_SCAN: '입력에 따라 순환 상태와 출력을 계산합니다.',
}

function roleExplanation(model: InspectorModel): string {
  switch (model.entity.kind) {
    case 'model': return '토큰을 임베딩하고 24개 Mamba 블록을 거쳐 어휘별 logits를 만듭니다.'
    case 'block': return '입력을 정규화한 뒤 convolution과 순환 상태를 갱신하고 잔차 출력을 만듭니다.'
    case 'stage': return STAGE_ROLES[model.entity.id.split('/').at(-1) ?? ''] ?? '선택한 단계의 입력을 처리해 다음 단계로 전달합니다.'
    case 'operator': {
      const op = model.document.tensors.find(tensor => model.entity.outputTensorIds.includes(tensor.id))?.op
      return `${op ?? 'GGML'} · ${OPERATOR_ROLES[op ?? ''] ?? '선택한 입력에서 출력 텐서를 만듭니다.'}`
    }
  }
}

function renderSummaryTensor(host: HTMLElement, tensor: InspectorTensor): void {
  const row = element('article', 'summary-tensor'); row.dataset.tensorId = tensor.id; row.dataset.role = tensor.role
  const name = element('h4', 'summary-tensor__name'); name.textContent = `tensor name : ${tensor.name}`
  const shape = element('p', 'summary-tensor__shape'); shape.dataset.field = 'native ne[4]'; shape.textContent = `native ne[4] ${tensor.nativeShape}`
  const meta = element('p', 'summary-tensor__meta')
  const dtype = element('span'); dtype.dataset.field = 'dtype'; dtype.textContent = tensor.dtype
  const bytes = element('span'); bytes.dataset.field = 'logicalIEC'; bytes.textContent = tensor.logicalIEC
  meta.append(dtype, ' / ', bytes, ` / ${tensor.role}`)
  row.append(name, shape, meta); host.append(row)
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
  host.scrollTop = 0
  const heading = element('h2', 'inspector__title')
  heading.id = 'inspector-selection'; heading.textContent = entityLabel(model.entity)
  heading.setAttribute('aria-description', model.entity.id)
  const header = element('header', 'inspector__header')
  const identity = element('p', 'inspector__identity'); identity.textContent = `${model.entity.kind} / ${model.document.scenario.name}`
  header.append(identity, heading)
  const summary = element('p', 'inspector__summary')
  summary.textContent = roleExplanation(model)
  header.append(summary); host.append(header)
  const compact = buildInspectorSummary(model)
  for (const section of compact.sections) {
    const group = element('section', `summary-section summary-section--${section.kind}`)
    group.dataset.section = section.kind; group.dataset.count = String(section.count)
    const title = element('h3'); title.id = `summary-${section.kind}`; title.textContent = `${section.label} / total tensors : ${section.count}`
    group.setAttribute('aria-labelledby', title.id); group.append(title)
    const note = element('p', 'summary-section__note')
    note.textContent = section.kind === 'weights'
      ? section.count ? `unique GGUF payload · ${formatIECBytes(section.uniqueGgufPayloadBytes).iec}` : EMPTY_WEIGHT_MESSAGE
      : `${section.count - section.stateCount - section.indexCount} activation / ${section.indexCount} index / ${section.stateCount} state`
    group.append(note)
    for (const tensor of section.preview) renderSummaryTensor(group, tensor)
    if (section.count > section.preview.length) {
      const hint = element('p', 'summary-section__hint'); hint.textContent = `${section.preview.length}개 미리보기 · 전체 목록은 고급 정보`
      group.append(hint)
    }
    host.append(group)
  }
  if (compact.state.count) {
    const group = element('section', 'summary-section summary-section--state'); group.dataset.section = 'state'; group.dataset.count = String(compact.state.count)
    const title = element('h3'); title.id = 'summary-state'; title.textContent = `State · ${compact.state.count} total`
    group.setAttribute('aria-labelledby', title.id)
    const note = element('p', 'summary-section__note'); note.textContent = '입출력 경계의 순환 상태 · activation과 별도'
    group.append(title, note)
    for (const tensor of compact.state.preview) renderSummaryTensor(group, tensor)
    host.append(group)
  }
  const advanced = element('details', 'inspector-advanced')
  const toggle = element('summary'); toggle.textContent = '고급 정보'
  const content = element('div', 'inspector-advanced__content'); content.tabIndex = 0
  content.setAttribute('role', 'region'); content.setAttribute('aria-label', '전체 tensor 필드와 캡처 근거')
  advanced.append(toggle, content)
  advanced.addEventListener('toggle', () => {
    if (advanced.open && !content.hasChildNodes()) renderFullInspector(content, model, bundle)
  })
  host.append(advanced)
}

function renderFullInspector(host: HTMLElement, model: InspectorModel, bundle?: PublicBundle): void {
  if (model.entity.kind === 'operator') {
    const output = model.document.tensors.find(tensor => model.entity.outputTensorIds.includes(tensor.id))
    if (output?.op !== undefined && output.opParamsI32 !== undefined && output.schedulerObserved !== undefined && output.arithmeticExecution !== undefined) {
      const evidence = panel('operator-evidence', '연산 캡처 근거')
      evidence.append(metadataTable({ id: model.entity.id, op: output.op, opParamsI32: output.opParamsI32, schedulerObserved: output.schedulerObserved, arithmeticExecution: output.arithmeticExecution }))
      host.append(evidence)
    }
  }
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
