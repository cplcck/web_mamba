import { expect, it } from 'vitest';
import { storageAccounting, validateDocument, type CaptureDocument, type Tensor } from '../src/schema';

// Tiny metadata fixtures, not model artifacts. Base layout/storage: split:prefill:t11.
// GGML 8144f319 ggml.c:1285-1319,1342-1346,1782-1787,1832-1836.
function empty(overrides: Partial<Tensor> = {}): Tensor {
  return {
    id: 'empty', name: 'node_7', role: 'activation', dtype: 'F32',
    nativeShape: [4608, 0, 1, 1], strides: [4, 18432, 0, 0],
    logicalShape: [1, 1, 0, 4608], axisLabels: ['ne3', 'ne2', 'ne1', 'ne0'],
    numel: 0, typeBlockSize: 1, logicalBytes: 0, viewSourceId: null, viewOffsetBytes: null,
    storage: { ggmlNbytes: 0, requiredAllocBytes: 0, bufferId: 'split:prefill:b1',
      bufferOffsetBytes: 0, bufferBytes: 1413312, observationEpoch: 17,
      allocatorSlotBytes: null, allocatorSlotReason: 'not exposed' },
    producerIds: [], consumerIds: ['op'], provenance: 'observed-layout-test-only',
    classification: 'observed', ...overrides,
  };
}
function document(tensors: readonly Tensor[]): CaptureDocument {
  const edges = { inputTensorIds: tensors.map(t => t.id), outputTensorIds: [], weightTensorIds: [] };
  return {
    schemaVersion: 1, captureId: 'empty-layout-test', ggufSha256: 'a'.repeat(64),
    scenario: { name: 'prefill', dimensions: { P: 1, T: 16, Q: 16, O: 1 } },
    entities: [
      { id: 'model', kind: 'model', parentId: null, children: ['op'], ...edges },
      { id: 'op', kind: 'operator', parentId: 'model', children: [], ...edges },
    ], tensors,
  };
}
function alias() {
  // Exact R storage/layout values: split:prefill:t4 and t12 (IDs remapped for isolation).
  const source = empty({ id: 'source', name: 'cache_r_l0', role: 'state',
    nativeShape: [4608, 1, 1, 1], logicalShape: [1, 1, 1, 4608], strides: [4, 18432, 18432, 18432],
    numel: 4608, logicalBytes: 18432 });
  source.storage = { ...source.storage, ggmlNbytes: 18432, requiredAllocBytes: 18432,
    bufferId: 'split:prefill:b2', bufferBytes: 2801664, observationEpoch: 5 };
  const view = empty({ name: 'cache_r_l0 (view)', role: 'state', viewSourceId: source.id, viewOffsetBytes: 18432,
    storage: { ...source.storage, ggmlNbytes: 0, requiredAllocBytes: null, bufferOffsetBytes: 18432, observationEpoch: 19 } });
  return { source, view };
}

it.each([4608, 24576])('preserves empty GET_ROWS metadata when row width is %i', width => {
  // Given: split prefill/decode t11/t49; nb1 is the observed F32 row stride.
  const t = empty({ nativeShape: [width, 0, 1, 1], logicalShape: [1, 1, 0, width], strides: [4, width * 4, 0, 0] });
  // When
  const checked = validateDocument(document([t]));
  // Then: exact equality retains zero allocation, full backing capacity, strides and node identity.
  expect(checked.tensors).toEqual([t]);
});
it('preserves a source-end empty view when R state has no extra rows', () => {
  // Given
  const { source, view } = alias();
  // When
  const checked = validateDocument(document([source, view]));
  // Then
  expect(checked.tensors).toEqual([source, view]);
});
it('preserves empty index metadata when ne0 is zero at the source end', () => {
  // Given: split:prefill:t8/t10.
  const { source, view } = alias();
  Object.assign(source, { role: 'index', dtype: 'I32', nativeShape: [1, 1, 1, 1], logicalShape: [1, 1, 1, 1], strides: [4, 4, 4, 4], numel: 1, logicalBytes: 4 });
  source.storage = { ...source.storage, ggmlNbytes: 4, requiredAllocBytes: 4, bufferId: 'split:prefill:b1', bufferOffsetBytes: 98432, bufferBytes: 1413312, observationEpoch: 11 };
  Object.assign(view, { role: 'index', dtype: 'I32', nativeShape: [0, 1, 1, 1], logicalShape: [1, 1, 1, 0], strides: [4, 0, 0, 0], viewOffsetBytes: 4 });
  view.storage = { ...source.storage, ggmlNbytes: 0, requiredAllocBytes: null, bufferOffsetBytes: 98436, observationEpoch: 15 };
  // When
  const checked = validateDocument(document([source, view]));
  // Then
  expect(checked.tensors).toEqual([source, view]);
});
it('preserves empty SCALE alias observations when decode has no state to zero', () => {
  // Given: split:decode:t4/t5/t6; SCALE is a root-backed alias, not an allocation.
  const { source, view } = alias();
  source.storage = { ...source.storage, bufferId: 'split:decode:b2', observationEpoch: 2329 };
  Object.assign(view, { nativeShape: [0, 1, 1, 1], logicalShape: [1, 1, 1, 0], strides: [4, 0, 0, 0], viewOffsetBytes: 0 });
  view.storage = { ...source.storage, ggmlNbytes: 0, requiredAllocBytes: null, observationEpoch: 2331 };
  const scale = { ...view, id: 'scale', storage: { ...view.storage, observationEpoch: 2333 } };
  const d = document([source, view, scale]); d.scenario = { name: 'decode', dimensions: { P: 1, T: 1, Q: 1, O: 1 } };
  // When
  const checked = validateDocument(d);
  // Then
  expect(checked.tensors).toEqual([source, view, scale]);
  expect(storageAccounting(checked.tensors)).toEqual({ logicalBytes: 18432, uniqueGgufBytes: 0, uniqueRuntimeRanges: 3 });
});
it.each([
  { dtype: 'F16', typeBlockSize: 1, nativeShape: [2, 3, 0, 1], strides: [2, 4, 12, 0] },
  { dtype: 'Q8_0', typeBlockSize: 32, nativeShape: [0, 1, 1, 1], strides: [34, 0, 0, 0] },
  { dtype: 'Q4_0', typeBlockSize: 32, nativeShape: [32, 2, 3, 0], strides: [18, 18, 36, 108] },
] as const)('preserves zero span when source-valid $dtype layout has a zero extent', layout => {
  // Given: additional constructor-valid layouts; quantized row alignment still applies.
  const t = empty({ ...layout, logicalShape: [...layout.nativeShape].reverse() });
  // When
  const checked = validateDocument(document([t]));
  // Then
  expect(checked.tensors).toEqual([t]);
});

const malformed: readonly [string, (t: Tensor) => unknown, RegExp][] = [
  ['negative native extent beside zero', t => ({ ...t, nativeShape: [-1, 0, 1, 1] }), /nativeShape\[0\]/],
  ['negative logical extent beside zero', t => ({ ...t, logicalShape: [0, -1, 1, 1] }), /logicalShape\[1\]/],
  ['unsafe unused extent', t => ({ ...t, nativeShape: [0, Number.MAX_SAFE_INTEGER + 1, 1, 1] }), /nativeShape\[1\]/],
  ['native intermediate overflow before zero', t => ({ ...t, nativeShape: [Number.MAX_SAFE_INTEGER, 2, 0, 1] }), /nativeShape: must be a safe integer/],
  ['logical intermediate overflow before zero', t => ({ ...t, logicalShape: [Number.MAX_SAFE_INTEGER, 2, 0, 1] }), /logicalShape: must be a safe integer/],
  ['missing native axis', t => ({ ...t, nativeShape: [4608, 0, 1] }), /must have four dimensions/],
  ['mismatched logical count', t => ({ ...t, logicalShape: [1, 1, 1, 4608] }), /element counts differ/],
  ['mismatched labels', t => ({ ...t, axisLabels: ['one'] }), /element counts differ/],
  ['nonzero numel', t => ({ ...t, numel: 1 }), /numel: does not match logical shape/],
  ['nonzero logical payload', t => ({ ...t, logicalBytes: 4 }), /logicalBytes: does not match dtype payload/],
  ['nonzero addressed span', t => ({ ...t, storage: { ...t.storage, ggmlNbytes: 4 } }), /ggmlNbytes: does not match strided addressed span/],
  ['negative allocation', t => ({ ...t, storage: { ...t.storage, requiredAllocBytes: -1 } }), /requiredAllocBytes/],
  ['missing dtype', t => ({ ...t, dtype: undefined }), /dtype: unsupported value/],
  ['unsupported dtype', t => ({ ...t, dtype: 'F64' }), /dtype: unsupported value/],
  ['wrong block size', t => ({ ...t, typeBlockSize: 0 }), /typeBlockSize: does not match dtype/],
  ['fractional unused stride', t => ({ ...t, strides: [4, 18432, 0.5, 0] }), /strides\[2\]/],
  ['negative unused stride', t => ({ ...t, strides: [4, 18432, -1, 0] }), /strides\[2\]/],
  ['unsafe unused stride', t => ({ ...t, strides: [4, 18432, Number.MAX_SAFE_INTEGER + 1, 0] }), /strides\[2\]/],
  ['missing stride', t => ({ ...t, strides: [4, 18432, 0] }), /must have four dimensions/],
  ['range starts past capacity', t => ({ ...t, storage: { ...t.storage, bufferOffsetBytes: 1413313 } }), /range exceeds backing buffer/],
  ['unsafe range offset', t => ({ ...t, storage: { ...t.storage, bufferOffsetBytes: Number.MAX_SAFE_INTEGER + 1 } }), /bufferOffsetBytes/],
  ['partially known range', t => ({ ...t, storage: { ...t.storage, bufferBytes: null } }), /runtime range fields must be all known or null/],
  ['fabricated slot size', t => ({ ...t, storage: { ...t.storage, allocatorSlotBytes: 0 } }), /allocatorSlotBytes: must remain null/],
];
it.each(malformed)('rejects malformed empty tensor when it has %s', (_name, mutate, error) => {
  // Given: each negative starts from an independently accepted observed positive control.
  const t = empty(), d = document([t]); validateDocument(d);
  // When / Then
  expect(() => validateDocument({ ...d, tensors: [mutate(t)] })).toThrow(error);
});
it.each(['Q8_0', 'Q4_0'] as const)('rejects a partial dtype block when empty %s has nonaligned ne0', dtype => {
  // Given: aligned positive empty row layout; only ne0 changes (numel/bytes remain zero).
  const size = dtype === 'Q8_0' ? 34 : 18;
  const t = empty({ dtype, typeBlockSize: 32, nativeShape: [32, 0, 1, 1], logicalShape: [1, 1, 0, 32], strides: [size, size, 0, 0] });
  const d = document([t]); validateDocument(d); t.nativeShape = [31, 0, 1, 1];
  // When / Then
  expect(() => validateDocument(d)).toThrow(/nativeShape\[0\]: must contain whole dtype blocks/);
});
const malformedViews: readonly [string, (pair: ReturnType<typeof alias>) => void, RegExp][] = [
  ['missing alias', ({ view }) => { view.viewSourceId = 'missing'; }, /dangling reference missing/],
  ['source-end overflow', ({ view }) => { view.viewOffsetBytes = 18433; view.storage.bufferOffsetBytes = 18433; }, /view exceeds backing source/],
  ['backing offset mismatch', ({ view }) => { view.storage.bufferOffsetBytes = 0; }, /does not agree with backing offset/],
  ['different backing buffer', ({ view }) => { view.storage.bufferId = 'other'; }, /view uses a different backing buffer/],
  ['inconsistent capacity', ({ view }) => { view.storage.bufferBytes = 2801665; }, /inconsistent for backing buffer/],
  ['view allocates zero independently', ({ view }) => { view.storage.requiredAllocBytes = 0; }, /views do not allocate independently/],
  ['missing view offset', ({ view }) => { view.viewOffsetBytes = null; }, /must be explicit for views/],
  ['self cycle', ({ view }) => { view.viewSourceId = view.id; }, /alias cycle/],
  ['pair cycle', ({ source, view }) => { source.viewSourceId = view.id; source.viewOffsetBytes = 0; source.storage.requiredAllocBytes = null; }, /alias cycle/],
];
it.each(malformedViews)('rejects malformed empty alias when it has %s', (_name, mutate, error) => {
  // Given
  const pair = alias(), d = document([pair.source, pair.view]); validateDocument(d); mutate(pair);
  // When / Then
  expect(() => validateDocument(d)).toThrow(error);
});
it.each(['P', 'O', 'T', 'Q'])('rejects zero scenario %s when its tensors are legitimately empty', key => {
  // Given
  const d = document([empty()]); validateDocument(d); d.scenario.dimensions[key] = 0;
  // When / Then
  expect(() => validateDocument(d)).toThrow(/scenario.dimensions/);
});
it('rejects missing crossing boundary when an empty tensor feeds an operator', () => {
  // Given
  const d = document([empty()]); validateDocument(d);
  d.entities = d.entities.map(e => e.id === 'model' ? { ...e, inputTensorIds: [] } : e);
  // When / Then
  expect(() => validateDocument(d)).toThrow(/entities.model.inputTensorIds: does not equal descendant crossing boundary/);
});
