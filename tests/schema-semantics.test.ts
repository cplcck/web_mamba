import { expect, it } from 'vitest';
import { validateDocument, storageAccounting, type CaptureDocument, type Tensor } from '../src/schema';

// Synthetic metadata only; layouts follow pinned GGML 8144f319 ggml_nbytes/view_2d.
function tensor(id: string, edges: Pick<Tensor, 'producerIds' | 'consumerIds'>): Tensor {
  return {
    id, name: id, role: 'activation', dtype: 'F32', nativeShape: [2, 1, 1, 1],
    strides: [4, 8, 8, 8], logicalShape: [2], axisLabels: ['features'], numel: 2,
    typeBlockSize: 1, logicalBytes: 8, viewSourceId: null, viewOffsetBytes: null,
    storage: { ggmlNbytes: 8, requiredAllocBytes: 8, bufferId: `buffer-${id}`,
      bufferOffsetBytes: 0, bufferBytes: 128, observationEpoch: 0,
      allocatorSlotBytes: null, allocatorSlotReason: 'not exposed' },
    provenance: 'synthetic-test-only', classification: 'observed', ...edges,
  };
}
function fixture(): CaptureDocument {
  const boundary = { inputTensorIds: ['x', 'state-before'], outputTensorIds: ['y', 'state-after'], weightTensorIds: ['w'] };
  return {
    schemaVersion: 1, captureId: 'synthetic-a', ggufSha256: 'a'.repeat(64),
    scenario: { name: 'prefill', dimensions: { P: 1, T: 16, Q: 16, O: 1 } },
    entities: [
      { id: 'model', kind: 'model', parentId: null, children: ['block'], ...boundary },
      { id: 'block', kind: 'block', parentId: 'model', children: ['stage'], ...boundary },
      { id: 'stage', kind: 'stage', parentId: 'block', children: ['op', 'empty-op'], ...boundary },
      { id: 'op', kind: 'operator', parentId: 'stage', children: [], ...boundary },
      { id: 'empty-op', kind: 'operator', parentId: 'stage', children: [], inputTensorIds: [], outputTensorIds: [], weightTensorIds: [] },
    ],
    tensors: [tensor('x', { producerIds: [], consumerIds: ['op'] }),
      tensor('y', { producerIds: ['op'], consumerIds: [] }),
      { ...tensor('w', { producerIds: [], consumerIds: ['op'] }), role: 'weight', ggufPayload: { fileId: 'gguf-a', offsetBytes: 100, bytes: 8 } },
      { ...tensor('state-before', { producerIds: [], consumerIds: ['op'] }), role: 'state' },
      { ...tensor('state-after', { producerIds: ['op'], consumerIds: [] }), role: 'state' }],
  };
}
function get(d: CaptureDocument, id: string): Tensor {
  const t = d.tensors.find(t => t.id === id);
  if (!t) throw new Error(`Missing test tensor ${id}`);
  return t;
}
function chain(): CaptureDocument {
  const d = fixture();
  get(d, 'y').consumerIds = ['op2'];
  d.tensors = [...d.tensors, tensor('z', { producerIds: ['op2'], consumerIds: [] })];
  d.entities = [...d.entities.map(e => e.kind === 'operator' ? e : {
    ...e, outputTensorIds: ['z', 'state-after'], children: e.id === 'stage' ? [...e.children, 'op2'] : e.children,
  }), { id: 'op2', kind: 'operator', parentId: 'stage', children: [], inputTensorIds: ['y'], outputTensorIds: ['z'], weightTensorIds: [] }];
  return d;
}
function view(): CaptureDocument {
  const d = fixture(), s = get(d, 'x'), v = get(d, 'y');
  Object.assign(s, { nativeShape: [4, 1, 1, 1], strides: [4, 16, 16, 16], logicalShape: [4], numel: 4, logicalBytes: 16 });
  s.storage.ggmlNbytes = 16; s.storage.requiredAllocBytes = 16; s.storage.bufferOffsetBytes = 8;
  v.viewSourceId = 'x'; v.viewOffsetBytes = 4;
  v.storage = { ...s.storage, ggmlNbytes: 8, requiredAllocBytes: null, bufferOffsetBytes: 12 };
  return d;
}

it('accepts crossing boundaries when a connected chain has internal y', () => {
  // Given
  const d = chain();
  // When
  const checked = validateDocument(d);
  // Then
  expect(checked.entities.filter(e => e.kind !== 'operator').map(e => [e.inputTensorIds, e.outputTensorIds, e.weightTensorIds]))
    .toEqual(Array.from({ length: 3 }, () => [['x', 'state-before'], ['z', 'state-after'], ['w']]));
});
it.each(['inputTensorIds', 'outputTensorIds'] as const)('rejects internal y when added to parent %s', field => {
  // Given: accepted connected graph, not a disjoint union fixture.
  const d = chain(); validateDocument(d);
  d.entities = d.entities.map(e => e.id === 'model' ? { ...e, [field]: [...e[field], 'y'] } : e);
  // When / Then
  expect(() => validateDocument(d)).toThrow(new RegExp(`entities.model.${field}`));
});
it('preserves residual crossings when stages sit directly under the model', () => {
  // Given: embedding -> block -> final stage, with x bypassing the block.
  const d = chain();
  d.entities = d.entities.map(e => e.id === 'model' ? { ...e, children: ['embedding', 'block', 'final'], inputTensorIds: ['tokens', 'state-before'], outputTensorIds: ['logits', 'state-after'] }
    : e.id === 'block' || e.id === 'stage' ? { ...e, outputTensorIds: ['z', 'state-after'] } : e);
  d.entities = [...d.entities,
    { id: 'embedding', kind: 'stage', parentId: 'model', children: ['embed'], inputTensorIds: ['tokens'], outputTensorIds: ['x'], weightTensorIds: [] },
    { id: 'embed', kind: 'operator', parentId: 'embedding', children: [], inputTensorIds: ['tokens'], outputTensorIds: ['x'], weightTensorIds: [] },
    { id: 'final', kind: 'stage', parentId: 'model', children: ['project'], inputTensorIds: ['x', 'z'], outputTensorIds: ['logits'], weightTensorIds: [] },
    { id: 'project', kind: 'operator', parentId: 'final', children: [], inputTensorIds: ['x', 'z'], outputTensorIds: ['logits'], weightTensorIds: [] }];
  get(d, 'x').producerIds = ['embed']; get(d, 'x').consumerIds = ['op', 'project']; get(d, 'z').consumerIds = ['project'];
  d.tensors = [...d.tensors, tensor('tokens', { producerIds: [], consumerIds: ['embed'] }), tensor('logits', { producerIds: ['project'], consumerIds: [] })];
  // When / Then
  expect(validateDocument(d).entities.find(e => e.id === 'final')?.inputTensorIds).toEqual(['x', 'z']);
});

const edgeCases: readonly [string, (d: CaptureDocument) => void][] = [
  ['extra producer', d => { get(d, 'y').producerIds = ['op', 'empty-op']; }],
  ['extra consumer', d => { get(d, 'x').consumerIds = ['op', 'empty-op']; }],
  ['weight missing consumer', d => { get(d, 'w').consumerIds = []; }],
  ['consumer missing weight reference', d => { get(d, 'w').consumerIds = ['op', 'empty-op']; }],
  ['input missing consumer', d => { get(d, 'x').consumerIds = []; }],
  ['output missing producer', d => { get(d, 'y').producerIds = []; }],
];
it.each(edgeCases)('rejects nonreciprocal edges when %s', (_name, mutate) => {
  // Given
  const d = fixture(); validateDocument(d); mutate(d);
  // When / Then
  expect(() => validateDocument(d)).toThrow(/reciprocal/);
});

const storageCases: readonly [string, () => CaptureDocument, (d: CaptureDocument) => void, RegExp][] = [
  ['view offset disagrees', view, d => { get(d, 'y').storage.bufferOffsetBytes = 8; }, /viewOffsetBytes/],
  ['view offset is null', view, d => { get(d, 'y').viewOffsetBytes = null; }, /viewOffsetBytes/],
  ['view allocates independently', view, d => { get(d, 'y').storage.requiredAllocBytes = 8; }, /requiredAllocBytes/],
  ['slot capacity is known', fixture, d => { get(d, 'x').storage.allocatorSlotBytes = 16; }, /allocatorSlotBytes/],
  ['slot reason is absent', fixture, d => { delete get(d, 'x').storage.allocatorSlotReason; }, /allocatorSlotReason/],
  ['span is allocation padding', fixture, d => { get(d, 'x').storage.ggmlNbytes = 16; get(d, 'x').storage.requiredAllocBytes = 16; }, /ggmlNbytes/],
  ['payload end overflows', fixture, d => { get(d, 'w').ggufPayload = { fileId: 'gguf-a', offsetBytes: Number.MAX_SAFE_INTEGER, bytes: 8 }; }, /ggufPayload/],
  ['non-view has view offset', fixture, d => { get(d, 'x').viewOffsetBytes = 0; }, /viewOffsetBytes/],
];
it.each(storageCases)('rejects storage metadata when %s', (_name, make, mutate, field) => {
  // Given: every negative starts from its own accepted positive control.
  const d = make(); validateDocument(d); mutate(d);
  // When / Then
  expect(() => validateDocument(d)).toThrow(field);
});
it.each(['self', 'pair'])('rejects alias cycles when the same-buffer relation is %s', kind => {
  // Given: same shapes, offsets, spans, buffer capacities; only alias topology changes.
  const d = fixture(), s = get(d, 'x'), v = get(d, 'y');
  v.storage = { ...s.storage, requiredAllocBytes: null }; v.viewSourceId = 'x'; v.viewOffsetBytes = 0;
  validateDocument(d);
  s.storage.requiredAllocBytes = null; s.viewSourceId = kind === 'self' ? 'x' : 'y'; s.viewOffsetBytes = 0;
  // When / Then
  expect(() => validateDocument(d)).toThrow(/cycle/);
});

const layouts = [
  { dtype: 'F32', nativeShape: [2, 3, 1, 1], strides: [4, 8, 24, 24], bytes: 24, block: 1 },
  { dtype: 'F32', nativeShape: [3, 2, 1, 1], strides: [8, 4, 24, 24], bytes: 24, block: 1 },
  { dtype: 'F16', nativeShape: [2, 3, 1, 1], strides: [2, 4, 12, 12], bytes: 12, block: 1 },
  { dtype: 'I32', nativeShape: [2, 3, 1, 1], strides: [4, 8, 24, 24], bytes: 24, block: 1 },
  { dtype: 'Q8_0', nativeShape: [32, 2, 1, 1], strides: [34, 34, 68, 68], bytes: 68, block: 32 },
  { dtype: 'Q4_0', nativeShape: [32, 2, 1, 1], strides: [18, 18, 36, 36], bytes: 36, block: 32 },
] as const;
it.each(layouts)('preserves GGML layout when dtype=$dtype strides=$strides', layout => {
  // Given
  const d = fixture(), t = get(d, 'x');
  Object.assign(t, layout, { logicalShape: layout.nativeShape, axisLabels: ['a', 'b', 'c', 'd'], numel: layout.nativeShape.reduce<number>((a, b) => a * b, 1), logicalBytes: layout.bytes, typeBlockSize: layout.block });
  t.storage.ggmlNbytes = layout.bytes; t.storage.requiredAllocBytes = layout.bytes;
  // When
  const checked = get(validateDocument(d), 'x');
  // Then
  expect([checked.logicalBytes, checked.storage.ggmlNbytes, checked.typeBlockSize, checked.strides]).toEqual([layout.bytes, layout.bytes, layout.block, layout.strides]);
});
it('preserves broadcast span when a source-valid GGML view has zero strides', () => {
  // Given
  const d = view(), v = get(d, 'y');
  Object.assign(v, { nativeShape: [2, 2, 1, 1], logicalShape: [2, 2], axisLabels: ['a', 'b'], strides: [4, 0, 0, 0], numel: 4, logicalBytes: 16, viewOffsetBytes: 0 });
  v.storage.bufferOffsetBytes = 8;
  // When
  const checked = get(validateDocument(d), 'y');
  // Then
  expect([checked.logicalBytes, checked.storage.ggmlNbytes, checked.strides]).toEqual([16, 8, [4, 0, 0, 0]]);
});
it('rejects byte-per-element quantization when Q8_0 metadata invents block size one', () => {
  // Given: accepted scalar layout; change only dtype to a type requiring real blocks.
  const d = fixture(), t = get(d, 'x');
  Object.assign(t, { dtype: 'I8', strides: [1, 2, 2, 2], logicalBytes: 2 });
  t.storage.ggmlNbytes = 2; t.storage.requiredAllocBytes = 2;
  validateDocument(d); t.dtype = 'Q8_0';
  // When / Then
  expect(() => validateDocument(d)).toThrow(/typeBlockSize/);
});

it.each(['prefill', 'decode'])('accepts canonical dimensions when scenario is %s', name => {
  // Given
  const d = fixture(), T = name === 'prefill' ? 16 : 1;
  d.scenario = { name, dimensions: { P: 1, T, Q: T, O: 1 } };
  // When / Then
  expect(validateDocument(d).scenario).toEqual(d.scenario);
});
const dimensionsCases = [{ P: 1, T: 16, Q: 16 }, { P: 1, T: 16, Q: 16, O: 2 }, { P: 0, T: 16, Q: 0, O: 1 }, { P: 2, T: 16, Q: 32, O: 1 }];
it.each(dimensionsCases)('rejects noncanonical captures when dimensions are %j', dimensions => {
  // Given
  const d = fixture(); validateDocument(d); d.scenario.dimensions = dimensions;
  // When / Then
  expect(() => validateDocument(d)).toThrow(/scenario.dimensions/);
});
it.each(['captureId', 'ggufSha256'] as const)('rejects stale identity when only %s changes', field => {
  // Given
  const d = fixture(), expected = { captureId: d.captureId, ggufSha256: d.ggufSha256 };
  validateDocument(d, expected); d[field] = field === 'captureId' ? 'stale' : 'b'.repeat(64);
  // When / Then
  expect(() => validateDocument(d, expected)).toThrow(/identity/);
});
it('preserves separate observations when recurrent state shares backing storage across epochs', () => {
  // Given
  const d = fixture(), before = get(d, 'state-before'), after = get(d, 'state-after');
  after.storage = { ...before.storage, observationEpoch: 1 };
  // When
  const checked = validateDocument(d);
  // Then
  expect(storageAccounting([get(checked, 'state-before'), get(checked, 'state-after')]).uniqueRuntimeRanges).toBe(2);
});
