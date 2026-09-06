// PRIVATE SYNTHETIC TEST BUNDLE. Its internally consistent 'pass' tokens exercise the
// validator; they are not a reviewed numerical PASS and must never be published.
import { createHash } from 'node:crypto'
import type { PublicBundleSources } from '../src/public-data'
const hash = 'a'.repeat(64), policy = 'b'.repeat(64), review = 'c'.repeat(64)
const native = 'private-synthetic-native', modelId = 'mamba-130m'
const constant = (value: number) => ({ op: 'const' as const, value })
const dim = (name: string) => ({ op: 'dim' as const, name })
const identity = { schemaVersion: 1 as const, captureId: 'private-synthetic-bundle', ggufSha256: hash }
const approval = { numericalStatus: 'pass' as const, policySha256: policy, reviewSha256: review }
function snapshot(scenario: 'prefill' | 'decode', after: boolean) {
  const label = scenario === 'prefill' ? (after ? 'after-prefill' : 'before-prefill') : (after ? 'after-decode' : 'before-decode')
  const initial = label === 'before-prefill', position = initial ? -1 : label === 'after-decode' ? 16 : 15
  return { id: `${native}/${label}`, captureId: `synthetic-${scenario}`, label, position,
    epoch: ['before-prefill', 'after-prefill', 'before-decode', 'after-decode'].indexOf(label),
    mapping: { head: 0, activeCell: initial ? null : 0, activeRow: initial ? null : 0, position,
      cells: [{ index: 0, pos: position, src: initial ? -1 : 0, src0: initial ? -1 : 0, tail: initial ? -1 : 0, sequenceIds: initial ? [] : [0] }] },
    arrays: Array.from({ length: 24 }, (_, layer) => (['R', 'S'] as const).map(family => {
      const extent = family === 'R' ? 3 : 16, numel = 1536 * extent
      return { layer, family, dtype: 'F32', shape: [1, 1536, extent], axisLabels: [initial ? 'cell' : 'sequence', 'inner', family === 'R' ? 'history' : 'state'],
        nativeShape: [numel, 1, 1, 1], nativeStrides: [4, numel * 4, numel * 4, numel * 4], numel, logicalBytes: numel * 4,
        sha256: hash, sourceRow: initial ? null : 0, fullInitialBuffer: initial,
        summary: { min: 0, max: 0, mean: 0, rms: 0, finite: true } }
    })).flat() }
}
function scenario(name: 'prefill' | 'decode') {
  const captureId = `synthetic-${name}`, T = name === 'prefill' ? 16 : 1
  const tensor = (id: string, role: 'index' | 'activation' | 'weight', producers: string[], consumers: string[]) => ({
    id, name: role === 'weight' ? `synthetic.${id}.<img src=x onerror=alert(1)>.${'긴텐서이름'.repeat(12)}` : `synthetic.${id}`,
    role, dtype: role === 'index' ? 'I32' : 'F32', nativeShape: [role === 'index' ? T : 2, 1, 1, 1],
    strides: [4, (role === 'index' ? T : 2) * 4, (role === 'index' ? T : 2) * 4, (role === 'index' ? T : 2) * 4],
    logicalShape: [1, 1, 1, role === 'index' ? T : 2], axisLabels: ['batch', 'rows', 'channels', role === 'index' ? 'tokens' : 'features'],
    numel: role === 'index' ? T : 2, typeBlockSize: 1, logicalBytes: (role === 'index' ? T : 2) * 4,
    storage: { ggmlNbytes: (role === 'index' ? T : 2) * 4, requiredAllocBytes: (role === 'index' ? T : 2) * 4,
      bufferId: `buffer-${id}`, bufferOffsetBytes: 0, bufferBytes: 2048, observationEpoch: name === 'prefill' ? 0 : 1,
      allocatorSlotBytes: null, allocatorSlotReason: 'private-synthetic-slot-not-exposed' },
    viewSourceId: null, viewOffsetBytes: null, producerIds: producers, consumerIds: consumers,
    provenance: 'private-synthetic-test-only', classification: 'observed', captureId,
    op: producers.length ? 'SSM_SCAN' : 'NONE', opParamsI32: Array<number>(16).fill(0),
    schedulerObserved: producers.length > 0, arithmeticExecution: producers.length > 0,
    shapeFormula: [constant(1), constant(1), constant(1), role === 'index' ? dim('Q') : constant(2)],
    logicalBytesFormula: role === 'index' ? { op: 'mul', left: dim('Q'), right: constant(4) } : constant(8),
    formulaClassification: role === 'index' ? 'symbolic-estimate' : 'observed-constant', formulaSource: 'private-synthetic-rule-not-model-evidence',
    ...(role === 'weight' ? { ggufPayload: { fileId: 'synthetic.gguf', offsetBytes: 4096, bytes: 8 } } : {}),
  })
  const blocks = Array.from({ length: 24 }, (_, layer) => {
    const id = `block-${layer}`, stage = `${id}/stage`, op = `${stage}/scan`, empty = `${stage}/empty`
    const boundary = { inputTensorIds: [`${id}/input`], outputTensorIds: [`${id}/output`], weightTensorIds: [`${id}/weight`] }
    const input = tensor(`${id}/input`, 'index', [], [op]), output = tensor(`${id}/output`, 'activation', [op], []), weight = tensor(`${id}/weight`, 'weight', [], [op])
    const alias = { ...output, op: 'VIEW', arithmeticExecution: false, viewSourceId: weight.id, viewOffsetBytes: 0,
      storage: { ...weight.storage, requiredAllocBytes: null } }
    const unobserved = { ...input, storage: { ...input.storage, requiredAllocBytes: null, bufferId: null, bufferOffsetBytes: null, bufferBytes: null } }
    return { boundary, entities: [
      { id, kind: 'block', parentId: modelId, children: [stage], ...boundary },
      { id: stage, kind: 'stage', parentId: id, children: [op, empty], ...boundary },
      { id: op, kind: 'operator', parentId: stage, children: [], ...boundary },
      { id: empty, kind: 'operator', parentId: stage, children: [], inputTensorIds: [], outputTensorIds: [], weightTensorIds: [] },
    ], tensors: [layer === 22 ? unobserved : input, layer === 23 ? alias : output, weight] }
  })
  return { schemaVersion: 1, captureId, ggufSha256: hash, numericalStatus: 'pass', policySha256: policy,
    scenario: { name, dimensions: { P: 1, T, Q: T, O: 1 } },
    entities: [{ id: modelId, kind: 'model', parentId: null, children: blocks.map((_, i) => `block-${i}`),
      inputTensorIds: blocks.flatMap(b => b.boundary.inputTensorIds), outputTensorIds: blocks.flatMap(b => b.boundary.outputTensorIds), weightTensorIds: blocks.flatMap(b => b.boundary.weightTensorIds) }, ...blocks.flatMap(b => b.entities)],
    tensors: blocks.flatMap(b => b.tensors), formulaClassification: 'symbolic-estimate',
    kernelInternals: { SSM_CONV: { classification: 'kernel-internal', formula: 'private synthetic convolution explanation' }, SSM_SCAN: { classification: 'kernel-internal', formula: 'private synthetic scan explanation' } },
    offlineConversion: 'private synthetic offline A_log -> -exp(A_log) note',
    recurrentState: { nativeCaptureId: native, sequenceId: 0, before: snapshot(name, false), after: snapshot(name, true),
      continuity: { fromSnapshotId: `${native}/after-prefill`, toSnapshotId: `${native}/before-decode`, mappingEqual: true, arraysByteEqual: true, comparedArrays: 48 } } }
}
export function fixtureFiles() {
  const prefill = scenario('prefill'), decode = scenario('decode')
  const capture = (document: ReturnType<typeof scenario>) => ({ captureId: document.captureId, path: `${document.scenario.name}.json`,
    sha256: createHash('sha256').update(JSON.stringify(document)).digest('hex'), tensorCount: document.tensors.length, operatorCount: document.entities.filter(e => e.kind === 'operator').length, emptyTensorCount: 0 })
  return {
    'manifest.json': { ...identity, ...approval, modelId, artifactAuditStatus: 'exact', captures: { prefill: capture(prefill), decode: capture(decode) } },
    'model.json': { ...identity, modelId, architecture: 'Mamba1', blockCount: 24, vocabularySize: 50280, tokenizerVocabularySize: 50277,
      configuration: { num_hidden_layers: 24, hidden_size: 768, intermediate_size: 1536, state_size: 16, conv_kernel: 4, time_step_rank: 48, vocab_size: 50280 } },
    'provenance.json': { ...identity, ...approval, repoId: 'state-spaces/mamba-130m-hf', revision: 'd'.repeat(40), manifestSha256: hash,
      preparationCaptureId: 'private-synthetic-preparation', artifactAuditSha256: hash, configurationHash: hash, canonicalNativeCaptureId: native,
      source: { commit: '8144f3192e5a3131cd043f284525e6ceebf82d0f', runtimeCommit: '8144f3192e5a3131cd043f284525e6ceebf82d0f', converterSha256: hash, mambaConverterSha256: hash, productPatchSha256: hash, worktreeDiffSha256: hash },
      environment: { cpu: 'PRIVATE SYNTHETIC CPU', python: 'synthetic-python', packages: { torch: 'synthetic-torch', transformers: 'synthetic-transformers' }, lockSha256: hash, requirementsSha256: hash, pythonBinarySha256: hash },
      compiler: { compilerId: 'synthetic-compiler', compilerVersion: 'synthetic-version', compilerSha256: hash }, buildFiles: { 'capture/main.cpp': hash }, cmakeFlagsSha256: hash,
      cmakeFlags: { GGML_CUDA: false }, configured: { n_threads: 1 }, effective: { n_threads: 1 }, nativeEnvironment: { LLAMA_GRAPH_REUSE_DISABLE: '1', GGML_CPU_DISABLE_FUSION: '1' }, systemInfo: 'private-synthetic-system' },
    'validation.json': { ...identity, status: 'pass', policySha256: policy, reviewSha256: review, configurationHash: hash,
      bindings: { manifestSha256: hash, ggufSha256: hash, auditSha256: hash, preparationCaptureId: 'private-synthetic-preparation',
        captures: [{ case: 'canonical', runtime: 'llama-split', captureId: native, receiptSha256: hash, archiveSha256: hash, runtimeHash: hash, fixtureSha256: hash, arrayCount: 48 }] },
      metrics: { 'private-synthetic-comparison': { maxAbs: 0, rmse: 0, nmse: 0, absoluteBound: 0, absolutePercentiles: [0, 0, 0], componentViolations: 0, referenceSha256: hash, candidateSha256: hash, shape: [2], envelopeViolations: [] } },
      failures: [], implementation: { validator: hash, ingestion: hash, graph: hash } },
    'prefill.json': prefill, 'decode.json': decode,
  }
}
export function fixtureSources(): PublicBundleSources {
  const files = fixtureFiles()
  return { manifest: JSON.stringify(files['manifest.json']), model: JSON.stringify(files['model.json']), provenance: JSON.stringify(files['provenance.json']),
    validation: JSON.stringify(files['validation.json']), prefill: JSON.stringify(files['prefill.json']), decode: JSON.stringify(files['decode.json']) }
}
