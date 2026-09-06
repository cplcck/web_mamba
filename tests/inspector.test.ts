import { describe, expect, it } from 'vitest'
import { buildInspectorModel, estimateTensor, formatIECBytes } from '../src/inspector'
import { loadPublicData, validateLoadedDocument } from '../src/main'
import { fixtureFiles, fixtureSources } from './inspector-fixture'
import { validatePublicBundle } from '../src/public-data'
import { validateDocument, type CaptureDocument } from '../src/schema'

const tensor = (id: string, role: 'activation' | 'weight' = 'activation') => ({
  id,
  name: role === 'weight' ? 'dense.weight[untrusted]<b>' : id,
  role,
  dtype: role === 'weight' ? 'F16' : 'F32',
  nativeShape: [2, 1, 1, 1],
  strides: [role === 'weight' ? 2 : 4, 8, 8, 8],
  logicalShape: [2],
  axisLabels: ['features'],
  numel: 2,
  typeBlockSize: 1,
  logicalBytes: role === 'weight' ? 4 : 8,
  storage: {
    ggmlNbytes: role === 'weight' ? 4 : 8,
    requiredAllocBytes: role === 'weight' ? 4 : 8,
    bufferId: `buffer-${id}`,
    bufferOffsetBytes: 0,
    bufferBytes: 16,
    observationEpoch: 0,
    allocatorSlotBytes: null,
    allocatorSlotReason: 'not exposed',
  },
  viewSourceId: null,
  viewOffsetBytes: null,
  producerIds: id === 'output' ? ['scan'] : [],
  consumerIds: id === 'input' || role === 'weight' ? ['scan'] : [],
  provenance: 'synthetic-test-only',
  classification: 'observed' as const,
})

const fixture = (): CaptureDocument => validateDocument({
  schemaVersion: 1,
  captureId: 'capture-a',
  ggufSha256: 'a'.repeat(64),
  scenario: { name: 'prefill', dimensions: { P: 1, T: 16, Q: 16, O: 1 } },
  entities: [
    { id: 'model', kind: 'model', parentId: null, children: ['scan', 'empty'], inputTensorIds: ['input'], outputTensorIds: ['output'], weightTensorIds: ['weight'] },
    { id: 'scan', kind: 'operator', parentId: 'model', children: [], inputTensorIds: ['input'], outputTensorIds: ['output'], weightTensorIds: ['weight'] },
    { id: 'empty', kind: 'operator', parentId: 'model', children: [], inputTensorIds: [], outputTensorIds: [], weightTensorIds: [] },
  ],
  tensors: [tensor('input'), tensor('output'), tensor('weight', 'weight')],
})

describe('public loader and symbolic calculator', () => {
  it('loads all six original sources beneath the supplied project base', async () => {
    // Given
    const sources = fixtureSources(), requested: string[] = []
    const fetcher: typeof fetch = async input => {
      const url = new URL(String(input)); requested.push(url.pathname)
      const key = url.pathname.split('/').at(-1)?.replace('.json', '')
      return new Response(Object.entries(sources).find(([name]) => name === key)?.[1] ?? '', { status: 200 })
    }
    // When
    const bundle = await loadPublicData(fetcher, 'http://localhost/web_mamba/')
    // Then
    expect(requested.sort()).toEqual(['decode', 'manifest', 'model', 'prefill', 'provenance', 'validation'].map(name => `/web_mamba/data/${name}.json`))
    expect(bundle.provenance.revision).toBe('d'.repeat(40))
    expect(bundle.documents.decode.captureId).toBe('synthetic-decode')
  })

  it.each(['model', 'provenance', 'validation'] as const)('rejects stale %s identity through the actual loader', async key => {
    // Given
    const sources = fixtureSources(), files = fixtureFiles()
    const changed = { ...sources, [key]: JSON.stringify({ ...files[`${key}.json`], captureId: 'stale' }) }
    // When / Then
    const fetcher: typeof fetch = async input => new Response(Object.entries(changed).find(([name]) => String(input).endsWith(`/${name}.json`))?.[1] ?? '')
    await expect(loadPublicData(fetcher, 'http://localhost/')).rejects.toThrow(`${key}.identity`)
  })

  it('evaluates source-backed shape and bytes for supported dimensions', async () => {
    // Given
    const document = (await validatePublicBundle(fixtureSources())).documents.prefill
    const tensor = document.tensors[0]
    if (tensor === undefined) throw new Error('fixture input absent')
    // When
    const canonical = estimateTensor(tensor, document, { P: 1, T: 16, O: 1 })
    const estimate = estimateTensor(tensor, document, { P: 1, T: 32, O: 32 })
    // Then
    expect(canonical).toMatchObject({ shape: tensor.logicalShape, logicalBytes: tensor.logicalBytes })
    expect(estimate).toMatchObject({ shape: [1, 1, 1, 32], numel: 32, logicalBytes: 128, classification: 'derived' })
    expect(estimate).not.toHaveProperty('requiredAllocBytes')
  })

  it.each([{ P: 0, T: 16, O: 1 }, { P: 2, T: 16, O: 1 }, { P: 1, T: 33, O: 1 },
    { P: 1, T: 1.5, O: 1 }, { P: 1, T: 1, O: 2 }, { P: 1, T: NaN, O: 1 }, { P: 1, T: Infinity, O: 1 }])('rejects unsupported dimensions %j', async dimensions => {
    // Given
    const document = (await validatePublicBundle(fixtureSources())).documents.prefill
    const tensor = document.tensors[0]
    if (tensor === undefined) throw new Error('fixture input absent')
    // When / Then
    expect(() => estimateTensor(tensor, document, dimensions)).toThrow(expect.objectContaining({ code: 'unsupported-dimensions' }))
  })

  it('does not extrapolate observed constants or missing formulas', async () => {
    // Given
    const document = (await validatePublicBundle(fixtureSources())).documents.prefill
    const weight = document.tensors.find(tensor => tensor.role === 'weight')
    if (weight === undefined) throw new Error('fixture weight absent')
    // When / Then
    expect(() => estimateTensor(weight, document, { P: 1, T: 32, O: 1 })).toThrow(expect.objectContaining({ code: 'observed-constant' }))
    const minimal = fixture().tensors[0]
    if (minimal === undefined) throw new Error('fixture input absent')
    expect(() => estimateTensor(minimal, document, { P: 1, T: 16, O: 1 })).toThrow(expect.objectContaining({ code: 'missing-formula' }))
  })

  it('keeps unique GGUF payloads separate from duplicated runtime ranges', async () => {
    // Given
    const document = (await validatePublicBundle(fixtureSources())).documents.prefill
    // When
    const weights = buildInspectorModel(document, 'mamba-130m').sections[2]
    // Then
    expect(weights).toMatchObject({ count: 24, totalLogicalBytes: 192, uniqueGgufPayloadBytes: 8, uniqueRuntimeRanges: 24 })
  })

  it.each(['missing', 'malformed', 'http', 'network', 'unapproved', 'tampered'] as const)('fails closed for %s data', async failure => {
    // Given
    const sources = fixtureSources()
    const fetcher: typeof fetch = async input => {
      if (String(input).endsWith('/manifest.json')) {
        if (failure === 'network') throw new TypeError('synthetic network error')
        if (failure === 'http') return new Response('', { status: 503 })
        if (failure === 'missing') return new Response('', { status: 404 })
        if (failure === 'malformed') return new Response('{')
        if (failure === 'unapproved') return new Response(JSON.stringify({ ...fixtureFiles()['manifest.json'], numericalStatus: 'unvalidated' }))
      }
      const source = Object.entries(sources).find(([name]) => String(input).endsWith(`/${name}.json`))?.[1] ?? ''
      return new Response(failure === 'tampered' && String(input).endsWith('/prefill.json') ? source + ' ' : source)
    }
    // When / Then
    await expect(loadPublicData(fetcher, 'http://localhost/')).rejects.toBeInstanceOf(Error)
  })

  it('rejects decode T other than one', async () => {
    // Given
    const document = (await validatePublicBundle(fixtureSources())).documents.decode
    const tensor = document.tensors[0]
    if (tensor === undefined) throw new Error('fixture input absent')
    // When / Then
    expect(() => estimateTensor(tensor, document, { P: 1, T: 2, O: 1 })).toThrow(expect.objectContaining({ code: 'unsupported-dimensions' }))
  })
})

describe('every-level inspector model', () => {
  it('retains capture epoch, payload range and allocator-slot evidence per row', () => {
    // Given: a private synthetic payload range, distinct from runtime backing.
    const document = fixture()
    const weight = document.tensors.find(item => item.id === 'weight')
    if (weight === undefined) throw new Error('fixture weight missing')
    weight.ggufPayload = { fileId: 'synthetic.gguf', offsetBytes: 4096, bytes: 4 }
    weight.storage.observationEpoch = 7
    // When
    const row = buildInspectorModel(document, 'model').sections[2]?.tensors[0]
    // Then: machine fields retain identities, zero is not an unknown substitute.
    expect(row).toMatchObject({ captureId: 'capture-a', observationEpoch: 7,
      ggufPayload: weight.ggufPayload, allocatorSlotBytes: null, allocatorSlotReason: weight.storage.allocatorSlotReason })
  })

  it('distinguishes a view non-allocation from an unobserved backend requirement', () => {
    // Given
    const document = fixture()
    const input = document.tensors.find(item => item.id === 'input')
    const output = document.tensors.find(item => item.id === 'output')
    if (input === undefined || output === undefined) throw new Error('fixture boundary missing')
    input.storage.requiredAllocBytes = null
    output.viewSourceId = input.id; output.viewOffsetBytes = 0
    output.storage = { ...input.storage }
    const checked = validateDocument(document)
    // When
    const model = buildInspectorModel(checked, 'scan')
    // Then
    expect(model.sections[0]?.tensors[0]).toMatchObject({ allocationStatus: 'unobserved' })
    expect(model.sections[1]?.tensors[0]).toMatchObject({ allocationStatus: 'view' })
  })

  it('renders IEC beside exact storage sizes rather than only logical payload', () => {
    // Given
    const document = fixture()
    for (const item of document.tensors) item.storage.bufferBytes = 2048
    // When
    const row = buildInspectorModel(document, 'scan').sections[0]?.tensors[0]
    // Then
    expect(row?.bufferCapacity).toBe('2,048 B / 2 KiB')
  })
  it('rejects a stale capture before selection rendering', () => {
    // Given
    const document = fixture()
    // When / Then
    expect(() => validateLoadedDocument(document, { captureId: 'stale', ggufSha256: document.ggufSha256 })).toThrow(/capture document: identity/)
  })
  it('formats binary bytes with exact and IEC values', () => {
    // Given
    const bytes = 1024
    // When
    const formatted = formatIECBytes(bytes)
    // Then
    expect(formatted).toEqual({ exact: '1,024 B', iec: '1 KiB' })
  })

  it('preserves inputs, outputs, and weights as separate sections with tensor fields', () => {
    // Given
    const document = fixture()
    // When
    const model = buildInspectorModel(document, 'model')
    // Then
    expect(model.sections.map(section => section.kind)).toEqual(['inputs', 'outputs', 'weights'])
    expect(model.sections[0]?.tensors[0]).toMatchObject({
      name: 'input', dtype: 'F32', nativeShape: '[2, 1, 1, 1]', logicalShape: 'features=2',
      numel: '2', typeBlockSize: 1, logicalBytes: '8 B', ggmlNbytes: '8 B', classification: 'observed',
    })
    expect(model.sections[2]?.tensors[0]).toMatchObject({ name: 'dense.weight[untrusted]<b>', dtype: 'F16' })
  })

  it('retains an explicit empty weight section without inventing allocation bytes', () => {
    // Given
    const document = fixture()
    // When
    const model = buildInspectorModel(document, 'empty')
    // Then
    expect(model.sections[2]).toMatchObject({ kind: 'weights', count: 0, empty: true, totalRequiredAllocBytes: null })
  })
})
