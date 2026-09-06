import { describe, expect, it } from 'vitest';
import { validatePublicBundle, type PublicBundleSources } from '../src/public-data';

const hash = 'a'.repeat(64), policy = 'b'.repeat(64), review = 'c'.repeat(64);
// Private structural data only. A coherent approval-shaped fixture is not numeric approval.
export function privateBundle() {
  const header = { schemaVersion: 1, captureId: 'private-web', ggufSha256: hash };
  const approval = { numericalStatus: 'pass', policySha256: policy, reviewSha256: review };
  const snapshot = (name: 'prefill' | 'decode', phase: 'before' | 'after') => {
    const initial = name === 'prefill' && phase === 'before', position = initial ? -1 : name === 'decode' && phase === 'after' ? 16 : 15;
    return { id: `private-native/${phase}-${name}`, captureId: `private-${name}`, label: `${phase}-${name}`, position,
      epoch: name === 'prefill' ? (phase === 'before' ? 1 : 2) : (phase === 'before' ? 3 : 4),
      mapping: { head: 0, activeCell: initial ? null : 0, activeRow: initial ? null : 0, position,
        cells: [{ index: 0, pos: position, src: initial ? -1 : 0, src0: initial ? -1 : 0, tail: initial ? -1 : 0, sequenceIds: initial ? [] : [0] }] },
      arrays: Array.from({ length: 48 }, (_, i) => {
        const width = i % 2 === 0 ? 3 : 16;
        return { layer: Math.floor(i / 2), family: i % 2 === 0 ? 'R' : 'S', dtype: 'F32', shape: [1, 1536, width],
          axisLabels: [initial ? 'cell' : 'sequence', 'inner', i % 2 === 0 ? 'history' : 'state'], nativeShape: [1536 * width, 1, 1, 1],
          nativeStrides: [4, ...Array<number>(3).fill(1536 * width * 4)], numel: 1536 * width, logicalBytes: 1536 * width * 4,
          sha256: hash, sourceRow: initial ? null : 0, fullInitialBuffer: initial, summary: { min: 0, max: 0, mean: 0, rms: 0, finite: true } };
      }) };
  };
  const graph = (name: 'prefill' | 'decode') => ({ ...header, ...approval, captureId: `private-${name}`,
    scenario: { name, dimensions: { P: 1, T: name === 'prefill' ? 16 : 1, Q: name === 'prefill' ? 16 : 1, O: 1 } },
    entities: [
      { id: 'mamba-130m', kind: 'model', parentId: null, children: ['op'], inputTensorIds: [], outputTensorIds: ['out'], weightTensorIds: [] },
      { id: 'op', kind: 'operator', parentId: 'mamba-130m', children: [], inputTensorIds: [], outputTensorIds: ['out'], weightTensorIds: [] } ],
    tensors: [{ id: 'out', name: 'private-output', role: 'activation', dtype: 'F32', nativeShape: [1, 1, 1, 1], strides: [4, 4, 4, 4], logicalShape: [1, 1, 1, 1], axisLabels: ['ne3', 'ne2', 'ne1', 'ne0'],
      numel: 1, typeBlockSize: 1, logicalBytes: 4, captureId: `private-${name}`,
      storage: { ggmlNbytes: 4, requiredAllocBytes: 4, bufferId: 'private-buffer', bufferOffsetBytes: 0, bufferBytes: 4, observationEpoch: 1, allocatorSlotBytes: null, allocatorSlotReason: 'not exposed' },
      viewSourceId: null, viewOffsetBytes: null, producerIds: ['op'], consumerIds: [], provenance: 'private-fixture', classification: 'observed',
      op: 'SSM_SCAN', opParamsI32: Array<number>(16).fill(0), schedulerObserved: true, arithmeticExecution: true,
      shapeFormula: Array.from({ length: 4 }, () => ({ op: 'const', value: 1 })), logicalBytesFormula: { op: 'const', value: 4 },
      formulaClassification: 'observed-constant', formulaSource: 'private observation-only reason' }],
    formulaClassification: 'private-root-classification',
    kernelInternals: { SSM_SCAN: { classification: 'kernel-internal', formula: 'private scan note' }, SSM_CONV: { classification: 'kernel-internal', formula: 'private conv note' } }, offlineConversion: 'private offline note',
    recurrentState: { nativeCaptureId: 'private-native', sequenceId: 0, before: snapshot(name, 'before'), after: snapshot(name, 'after'),
      continuity: { fromSnapshotId: 'private-native/after-prefill', toSnapshotId: 'private-native/before-decode', mappingEqual: true, arraysByteEqual: true, comparedArrays: 48 } } });
  const capture = (name: 'prefill' | 'decode') => ({ captureId: `private-${name}`, path: `${name}.json`, sha256: hash, tensorCount: 1, operatorCount: 1, emptyTensorCount: 0 });
  return {
    manifest: { ...header, ...approval, modelId: 'mamba-130m', artifactAuditStatus: 'exact', captures: { prefill: capture('prefill'), decode: capture('decode') } },
    model: { ...header, modelId: 'mamba-130m', architecture: 'Mamba1', blockCount: 24, vocabularySize: 50280, tokenizerVocabularySize: 50277,
      configuration: { num_hidden_layers: 24, hidden_size: 768, intermediate_size: 1536, state_size: 16, conv_kernel: 4, time_step_rank: 48, vocab_size: 50280 } },
    provenance: { ...header, ...approval, repoId: 'state-spaces/mamba-130m-hf', revision: 'd'.repeat(40), manifestSha256: hash, preparationCaptureId: 'private-prepare',
      artifactAuditSha256: hash, configurationHash: hash, canonicalNativeCaptureId: 'private-native',
      source: { commit: '8144f3192e5a3131cd043f284525e6ceebf82d0f', runtimeCommit: '8144f3192e5a3131cd043f284525e6ceebf82d0f', converterSha256: hash, mambaConverterSha256: hash, productPatchSha256: hash, worktreeDiffSha256: hash },
      environment: { cpu: 'private CPU', python: 'private version', packages: { torch: 'private', transformers: 'private' }, lockSha256: hash, requirementsSha256: hash, pythonBinarySha256: hash },
      compiler: { compilerId: 'private', compilerVersion: 'private', compilerSha256: hash }, buildFiles: { 'mamba-capture': hash }, cmakeFlagsSha256: hash,
      cmakeFlags: { GGML_CPU: 'ON' }, configured: { n_threads: 1 }, effective: { n_threads: 1, backends: ['CPU'] }, nativeEnvironment: { LLAMA_GRAPH_REUSE_DISABLE: '1' }, systemInfo: 'private' },
    validation: { ...header, status: 'pass', policySha256: policy, reviewSha256: review, configurationHash: hash,
      bindings: { manifestSha256: hash, ggufSha256: hash, auditSha256: hash, preparationCaptureId: 'private-prepare', captures: [
        { case: 'canonical', runtime: 'llama-split', captureId: 'private-native', receiptSha256: hash, archiveSha256: hash, runtimeHash: hash, fixtureSha256: hash, arrayCount: 48 } ] },
      metrics: { 'private/comparison': { maxAbs: 0, rmse: 0, nmse: 0, absoluteBound: 0, absolutePercentiles: [0, 0, 0], componentViolations: 0, referenceSha256: hash, candidateSha256: hash, shape: [1], envelopeViolations: [] } },
      failures: [], implementation: { validator: hash, ingestion: hash, graph: hash } }, prefill: graph('prefill'), decode: graph('decode'),
  };
}
async function digest(source: string) {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source)))].map(v => v.toString(16).padStart(2, '0')).join('');
}
export async function bundleSources(input = privateBundle()): Promise<PublicBundleSources> {
  const prefill = JSON.stringify(input.prefill), decode = JSON.stringify(input.decode);
  const manifest = { ...input.manifest, captures: {
    prefill: { ...input.manifest.captures.prefill, sha256: await digest(prefill) }, decode: { ...input.manifest.captures.decode, sha256: await digest(decode) } } };
  return { manifest: JSON.stringify(manifest), model: JSON.stringify(input.model), provenance: JSON.stringify(input.provenance), validation: JSON.stringify(input.validation), prefill, decode };
}

describe('public release identity boundary', () => {
  it('returns typed metadata and both populated documents for a consistent private fixture', async () => {
    // Given
    const sources = await bundleSources();
    // When
    const result = await validatePublicBundle(sources);
    // Then
    expect(result.documents.decode.tensors).toHaveLength(1);
    expect(result.documents.decode.recurrentState?.before.arrays).toHaveLength(48);
    expect(result.validation.policySha256).toBe(policy);
  });
  const mutations: readonly [string, keyof PublicBundleSources, object][] = [
    ['unvalidated manifest', 'manifest', { numericalStatus: 'unvalidated' }], ['failed manifest', 'manifest', { numericalStatus: 'failed' }],
    ['candidate manifest', 'manifest', { numericalStatus: 'candidate' }], ['unknown manifest version', 'manifest', { schemaVersion: 2 }],
    ['unknown model version', 'model', { schemaVersion: 999 }], ['unknown provenance version', 'provenance', { schemaVersion: 999 }],
    ['unknown validation version', 'validation', { schemaVersion: 999 }], ['unknown graph version', 'prefill', { schemaVersion: 999 }],
    ['stale model', 'model', { captureId: 'stale' }], ['stale provenance', 'provenance', { ggufSha256: review }],
    ['wrong model', 'model', { modelId: 'other' }], ['wrong architecture', 'model', { architecture: 'Mamba2' }],
    ['missing policy', 'manifest', { policySha256: null }], ['malformed hash', 'manifest', { reviewSha256: 'bad' }],
    ['stale policy', 'provenance', { policySha256: hash }], ['stale review', 'validation', { reviewSha256: hash }],
    ['unvalidated report', 'validation', { status: 'unvalidated' }], ['failed report', 'validation', { status: 'failed' }],
    ['stale configuration', 'validation', { configurationHash: review }], ['missing metrics', 'validation', { metrics: {} }],
    ['missing provenance', 'provenance', { source: undefined }], ['wrong native capture', 'provenance', { canonicalNativeCaptureId: 'stale' }],
    ['missing release state', 'prefill', { recurrentState: undefined }], ['missing kernel notes', 'prefill', { kernelInternals: undefined }],
    ['wrong scenario status', 'prefill', { numericalStatus: 'unvalidated' }], ['wrong scenario policy', 'prefill', { policySha256: hash }],
  ];
  it.each(mutations)('rejects %s even when scenario source hashes are fresh', async (_name, file, fields) => {
    // Given
    const sources = await bundleSources(), parsed: unknown = JSON.parse(sources[file]);
    if (typeof parsed !== 'object' || parsed === null) throw new Error('fixture object required');
    const changed = { ...sources, [file]: JSON.stringify({ ...parsed, ...fields }) };
    if (file === 'prefill' || file === 'decode') {
      const manifest = privateBundle().manifest;
      changed.manifest = JSON.stringify({ ...manifest, captures: { ...manifest.captures,
        prefill: { ...manifest.captures.prefill, sha256: await digest(changed.prefill) }, decode: { ...manifest.captures.decode, sha256: await digest(changed.decode) } } });
    }
    // When / Then
    await expect(validatePublicBundle(changed)).rejects.toThrow();
  });
  it('rejects scenario bytes changed only by whitespace without a refreshed manifest hash', async () => {
    // Given
    const sources = await bundleSources();
    // When / Then
    await expect(validatePublicBundle({ ...sources, prefill: sources.prefill + '\n' })).rejects.toMatchObject({ path: 'prefill.sha256' });
  });
  it('rejects a stale continuity hash despite matching local snapshot labels', async () => {
    // Given
    const input = privateBundle();
    input.decode.recurrentState.before.arrays = input.decode.recurrentState.before.arrays.map((a, i) => i === 0 ? { ...a, sha256: review } : a);
    // When / Then
    await expect(validatePublicBundle(await bundleSources(input))).rejects.toThrow();
  });
  it('rejects missing tensor presentation evidence in a public bundle', async () => {
    // Given
    const input = privateBundle();
    Reflect.deleteProperty(input.prefill.tensors[0] ?? {}, 'formulaSource');
    // When / Then
    await expect(validatePublicBundle(await bundleSources(input))).rejects.toThrow();
  });
  it('keeps graph root identity separate from the model identifier', async () => {
    // Given
    const input = privateBundle();
    for (const d of [input.prefill, input.decode]) d.entities = d.entities.map(e => e.parentId === null
      ? { ...e, id: 'private-graph-root' } : { ...e, parentId: 'private-graph-root' });
    // When
    const result = await validatePublicBundle(await bundleSources(input));
    // Then
    expect(result.documents.prefill.entities[0]?.id).toBe('private-graph-root');
  });
  it('reports malformed JSON at the file boundary', async () => {
    // Given
    const sources = await bundleSources();
    // When / Then
    await expect(validatePublicBundle({ ...sources, model: '{' })).rejects.toMatchObject({ path: 'model' });
  });
});
