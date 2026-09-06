import { describe, expect, it } from 'vitest';
import { evalFormula, storageAccounting, validateDocument, type CaptureDocument, type Tensor, type Formula } from '../src/schema';

const tensor = (id: string, role = 'activation', extra: Partial<Tensor> = {}) => ({
  id, name: id, role, nativeShape: [2, 1, 1, 1], strides: [4, 8, 8, 8], logicalShape: [2], axisLabels: ['features'],
  dtype: role === 'index' ? 'I32' : 'F32', numel: 2, typeBlockSize: 1, logicalBytes: 8,
  storage: { ggmlNbytes: 8, requiredAllocBytes: 8, bufferId: `b-${id}`, bufferOffsetBytes: 0, bufferBytes: 16,
    observationEpoch: 0, allocatorSlotBytes: null, allocatorSlotReason: 'not exposed' },
  viewSourceId: null, viewOffsetBytes: null, producerIds: id === 'output' ? ['op'] : [],
  consumerIds: id === 'output' ? [] : ['op'], provenance: 'synthetic-test-only', classification: 'observed', ...extra,
});
const fixture = (): CaptureDocument => validateDocument({ schemaVersion: 1, captureId: 'capture-a', ggufSha256: 'a'.repeat(64), scenario: { name: 'prefill', dimensions: { P: 1, T: 16, Q: 16, O: 1 } }, entities: [
  { id: 'model', kind: 'model', parentId: null, children: ['block'], inputTensorIds: ['input','index'], outputTensorIds: ['output'], weightTensorIds: ['weight'] },
  { id: 'block', kind: 'block', parentId: 'model', children: ['stage'], inputTensorIds: ['input','index'], outputTensorIds: ['output'], weightTensorIds: ['weight'] },
  { id: 'stage', kind: 'stage', parentId: 'block', children: ['op','empty-op'], inputTensorIds: ['input','index'], outputTensorIds: ['output'], weightTensorIds: ['weight'] },
  { id: 'op', kind: 'operator', parentId: 'stage', children: [], inputTensorIds: ['input','index'], outputTensorIds: ['output'], weightTensorIds: ['weight'] },
  { id: 'empty-op', kind: 'operator', parentId: 'stage', children: [], inputTensorIds: [], outputTensorIds: [], weightTensorIds: [] },
], tensors: [tensor('input'), tensor('index','index'), tensor('output'), tensor('weight', 'weight', { ggufPayload: { fileId: 'a'.repeat(64), offsetBytes: 0, bytes: 8 } })] });
function get(d: CaptureDocument, id: string): Tensor {
  const t = d.tensors.find(t => t.id === id);
  if (!t) throw new Error(`Missing fixture tensor ${id}`);
  return t;
}

const invalidDocuments: readonly [string, (d: CaptureDocument) => unknown, RegExp][] = [
  ['missing dtype', d => ({ ...d, tensors: d.tensors.map(({ dtype, ...t }) => t) }), /dtype/],
  ['dangling producer', d => ({ ...d, tensors: d.tensors.map(t => t.id === 'input' ? { ...t, producerIds: ['missing'] } : t) }), /producer/],
  ['wrong parent', d => ({ ...d, entities: d.entities.map(e => e.id === 'block' ? { ...e, parentId: 'other' } : e) }), /parent/],
  ['empty entity ID', d => ({ ...d, entities: d.entities.map(e => e.id === 'model' ? { ...e, id: '' } : e) }), /entities\[0\].id/],
  ['stale tensor capture', d => ({ ...d, tensors: d.tensors.map(t => t.id === 'input' ? { ...t, captureId: 'other' } : t) }), /captureId/],
  ['negative dimension', d => ({ ...d, scenario: { ...d.scenario, dimensions: { ...d.scenario.dimensions, P: -1 } } }), /scenario.dimensions.P/],
  ['wrong Q', d => ({ ...d, scenario: { ...d.scenario, dimensions: { ...d.scenario.dimensions, Q: 99 } } }), /scenario.dimensions.Q/],
  ['parent-child mismatch', d => ({ ...d, entities: d.entities.map(e => e.id === 'block' ? { ...e, parentId: 'op' } : e) }), /parent/],
];

describe('executable data contracts', () => {
  it('validates all hierarchy sections when tensors have mixed dtypes and empty weights', () => {
    // Given
    const d = fixture();
    // When
    const checked = validateDocument(d);
    // Then
    expect(checked.entities.map(e => e.kind)).toEqual(['model', 'block', 'stage', 'operator', 'operator']);
    expect(checked.entities.at(-1)?.weightTensorIds).toEqual([]);
    expect(new Set(checked.tensors.map(t => t.dtype))).toEqual(new Set(['F32', 'I32']));
  });
  it.each(invalidDocuments)('rejects a document when it has %s', (_name, mutate, field) => {
    // Given: fixture() validates the positive control before the isolated mutation.
    const invalid = mutate(fixture());
    // When / Then
    expect(() => validateDocument(invalid)).toThrow(field);
  });
  it('rejects arithmetic overflow when a safe constant is multiplied by two', () => {
    // Given
    const left: Formula = { op: 'const', value: Number.MAX_SAFE_INTEGER };
    expect(evalFormula(left, {})).toBe(Number.MAX_SAFE_INTEGER);
    // When / Then
    expect(() => evalFormula({ op: 'mul', left, right: { op: 'const', value: 2 } }, {})).toThrow(/formula/);
  });
  it('rejects an unsupported operation when a JavaScript caller passes div', () => {
    // Given: the typed API excludes div; exercise its runtime boundary without a type assertion.
    const valid: Formula = { op: 'add', left: { op: 'const', value: 1 }, right: { op: 'const', value: 1 } };
    expect(evalFormula(valid, {})).toBe(2);
    // When / Then
    expect(() => Reflect.apply(evalFormula, undefined, [{ ...valid, op: 'div' }, {}])).toThrow(/formula.op/);
  });
  it('computes general symbolic arithmetic when dimensions differ from observed captures', () => {
    // Given
    const f: Formula = { op: 'mul', left: { op: 'dim', name: 'P' }, right: { op: 'add', left: { op: 'dim', name: 'T' }, right: { op: 'const', value: 1 } } };
    // When / Then
    expect(evalFormula(f, { P: 2, T: 3 })).toBe(8);
  });
  it('rejects a negative result when subtraction exceeds its left operand', () => {
    // Given
    const f: Formula = { op: 'sub', left: { op: 'const', value: 1 }, right: { op: 'const', value: 1 } };
    expect(evalFormula(f, {})).toBe(0);
    // When / Then
    expect(() => evalFormula({ ...f, right: { op: 'const', value: 2 } }, {})).toThrow(/formula/);
  });
  it('excludes unknown runtime storage when allocation is unobserved', () => {
    // Given
    const d = fixture(), input = get(d, 'input'), index = get(d, 'index');
    const unknown = { ...input, storage: { ...input.storage, requiredAllocBytes: null, bufferId: null, bufferOffsetBytes: null, bufferBytes: null } };
    // When
    const checked = validateDocument({ ...d, tensors: d.tensors.map(t => t.id === 'input' ? unknown : t) });
    // Then
    expect(get(checked, 'input').storage.requiredAllocBytes).toBeNull();
    expect(storageAccounting([input, index, get(checked, 'input')])).toEqual({ logicalBytes: 24, uniqueRuntimeRanges: 2, uniqueGgufBytes: 0 });
  });
  it('preserves strided view shape and range alias without treating it as zero bytes', () => {
    // Given: a four-element source and ggml_view_2d(ne0=1, ne1=2, nb1=8, offset=4).
    const d = fixture(), input = get(d, 'input'), index = get(d, 'index');
    const source = { ...input, nativeShape: [4, 1, 1, 1], logicalShape: [4], numel: 4,
      logicalBytes: 16, strides: [4, 16, 16, 16], storage: { ...input.storage, ggmlNbytes: 16, requiredAllocBytes: 16 } };
    const view = { ...index, id: 'view', consumerIds: [], viewSourceId: 'input', viewOffsetBytes: 4,
      nativeShape: [1, 2, 1, 1], logicalShape: [1, 2], axisLabels: ['a', 'b'], strides: [4, 8, 16, 16],
      storage: { ...source.storage, ggmlNbytes: 12, bufferOffsetBytes: 4, requiredAllocBytes: null } };
    // When
    const checked = validateDocument({ ...d, tensors: [...d.tensors.map(t => t.id === 'input' ? source : t), view] });
    // Then
    expect(get(checked, 'view').logicalBytes).toBe(8);
    expect(get(checked, 'view').storage.ggmlNbytes).toBe(12);
    expect(storageAccounting(checked.tensors).uniqueGgufBytes).toBe(8);
  });
  it('rejects a document when the expected capture identity differs', () => {
    // Given
    const d = fixture(); validateDocument(d, { captureId: d.captureId, ggufSha256: d.ggufSha256 });
    // When / Then
    expect(() => validateDocument(d, { captureId: 'other', ggufSha256: d.ggufSha256 })).toThrow(/identity/);
  });
  it('deduplicates payloads when tied weights have independent runtime copies', () => {
    // Given
    const w = get(fixture(), 'weight'), copy = { ...w, id: 'weight-copy', storage: { ...w.storage, bufferId: 'runtime-copy' } };
    // When / Then
    expect(storageAccounting([w, copy])).toEqual({ logicalBytes: 16, uniqueGgufBytes: 8, uniqueRuntimeRanges: 2 });
  });
  it('keeps observations distinct when a backing range is reused at another epoch', () => {
    // Given
    const input = get(fixture(), 'input'), after = { ...input, id: 'epoch-1', storage: { ...input.storage, observationEpoch: 1 } };
    // When / Then
    expect(storageAccounting([input, after]).uniqueRuntimeRanges).toBe(2);
  });
});
