import { describe, expect, it } from 'vitest';
import { validateDocument } from '../src/schema';

// Private structural fixture; no numeric approval or real model observations.
export function metadataFixture(name: 'prefill' | 'decode' = 'prefill') {
  return {
    schemaVersion: 1, captureId: `private-${name}`, ggufSha256: 'a'.repeat(64),
    scenario: { name, dimensions: { P: 1, T: name === 'prefill' ? 16 : 1, Q: name === 'prefill' ? 16 : 1, O: 1 } },
    entities: [
      { id: 'mamba-130m', kind: 'model', parentId: null, children: ['op'], inputTensorIds: [], outputTensorIds: ['out'], weightTensorIds: [] },
      { id: 'op', kind: 'operator', parentId: 'mamba-130m', children: [], inputTensorIds: [], outputTensorIds: ['out'], weightTensorIds: [] },
    ],
    tensors: [{ id: 'out', name: 'private-output', role: 'activation', dtype: 'F32',
      nativeShape: [1, 1, 1, 1], strides: [4, 4, 4, 4], logicalShape: [1], axisLabels: ['P'],
      numel: 1, typeBlockSize: 1, logicalBytes: 4, captureId: `private-${name}`,
      storage: { ggmlNbytes: 4, requiredAllocBytes: 4, bufferId: 'private-buffer', bufferOffsetBytes: 0,
        bufferBytes: 4, observationEpoch: 1, allocatorSlotBytes: null, allocatorSlotReason: 'not exposed' },
      viewSourceId: null, viewOffsetBytes: null, producerIds: ['op'], consumerIds: [],
      provenance: 'private-fixture', classification: 'observed', op: 'SSM_SCAN', opParamsI32: Array<number>(16).fill(0),
      schedulerObserved: true, arithmeticExecution: true,
      shapeFormula: [{ op: 'dim', name: 'P' }],
      logicalBytesFormula: { op: 'mul', left: { op: 'dim', name: 'P' }, right: { op: 'const', value: 4 } },
    }],
    formulaClassification: 'observed-layout-constants; changed dimensions are unvalidated',
    kernelInternals: { SSM_SCAN: { classification: 'kernel-internal', formula: 'private scan explanation' },
      SSM_CONV: { classification: 'kernel-internal', formula: 'private conv explanation' } },
    offlineConversion: 'private offline explanation',
  };
}

describe('capture presentation metadata boundary', () => {
  it('preserves operator evidence and closed formulas when canonical values agree', () => {
    // Given
    const input = metadataFixture();
    // When
    const result = validateDocument(input);
    // Then: shipped-copy equality, not pinned explanatory prose.
    expect(result.tensors[0]).toMatchObject(input.tensors[0] ?? {});
    expect(result).toMatchObject({ kernelInternals: input.kernelInternals, offlineConversion: input.offlineConversion,
      formulaClassification: input.formulaClassification });
  });

  const badMetadata: readonly [string, object][] = [
    ['unknown opcode syntax', { op: '<script>' }],
    ['partial execution evidence', { schedulerObserved: undefined }],
    ['nonboolean execution', { arithmeticExecution: 'true' }],
    ['short op parameters', { opParamsI32: [0] }],
    ['overflow op parameters', { opParamsI32: Array<number>(16).fill(2 ** 31) }],
    ['metadata arithmetic claim', { op: 'VIEW', arithmeticExecution: true }],
    ['unknown formula operation', { shapeFormula: [{ op: 'eval', code: '1' }] }],
    ['extra formula key', { shapeFormula: [{ op: 'const', value: 1, code: '1' }] }],
    ['unknown formula dimension', { shapeFormula: [{ op: 'dim', name: 'vocab' }] }],
    ['wrong canonical shape', { shapeFormula: [{ op: 'const', value: 2 }] }],
    ['wrong canonical bytes', { logicalBytesFormula: { op: 'const', value: 5 } }],
    ['partial formulas', { logicalBytesFormula: undefined }],
    ['negative formula', { shapeFormula: [{ op: 'sub', left: { op: 'const', value: 0 }, right: { op: 'const', value: 1 } }] }],
    ['overflow formula', { shapeFormula: [{ op: 'mul', left: { op: 'const', value: Number.MAX_SAFE_INTEGER }, right: { op: 'const', value: 2 } }] }],
  ];
  it.each(badMetadata)('rejects %s rather than discarding it', (_name, fields) => {
    // Given
    const input = metadataFixture();
    const changed = { ...input, tensors: input.tensors.map(t => ({ ...t, ...fields })) };
    // When / Then
    expect(() => validateDocument(changed)).toThrow();
  });
  it('rejects kernel notes with invented kernel IDs', () => {
    // Given
    const input = { ...metadataFixture(), kernelInternals: { EXP: { classification: 'kernel-internal', formula: '1' } } };
    // When / Then
    expect(() => validateDocument(input)).toThrow();
  });
});

function recurrentFixture() {
  const snapshot = (initial: boolean) => ({
    id: `native/${initial ? 'before-prefill' : 'after-prefill'}`, captureId: 'private-prefill',
    label: initial ? 'before-prefill' : 'after-prefill', position: initial ? -1 : 15, epoch: initial ? 1 : 2,
    mapping: { head: 0, activeCell: initial ? null : 0, activeRow: initial ? null : 0, position: initial ? -1 : 15,
      cells: [{ index: 0, pos: initial ? -1 : 15, src: initial ? -1 : 0, src0: initial ? -1 : 0, tail: initial ? -1 : 0, sequenceIds: initial ? [] : [0] }] },
    arrays: Array.from({ length: 48 }, (_, i) => {
      const width = i % 2 === 0 ? 3 : 16;
      return { layer: Math.floor(i / 2), family: i % 2 === 0 ? 'R' : 'S', dtype: 'F32', shape: [1, 1536, width],
        axisLabels: [initial ? 'cell' : 'sequence', 'inner', i % 2 === 0 ? 'history' : 'state'],
        nativeShape: [1536 * width, 1, 1, 1], nativeStrides: [4, ...Array<number>(3).fill(1536 * width * 4)],
        numel: 1536 * width, logicalBytes: 1536 * width * 4, sha256: 'b'.repeat(64), sourceRow: initial ? null : 0,
        fullInitialBuffer: initial, summary: { min: 0, max: 0, mean: 0, rms: 0, finite: true } };
    }),
  });
  return { nativeCaptureId: 'native', sequenceId: 0, before: snapshot(true), after: snapshot(false),
    continuity: { fromSnapshotId: 'native/after-prefill', toSnapshotId: 'native/before-decode', mappingEqual: true, arraysByteEqual: true, comparedArrays: 48 } };
}

describe('capture-bound recurrent summaries', () => {
  it('preserves before/after state summaries and signed initial position', () => {
    // Given
    const recurrentState = recurrentFixture();
    // When
    const result = validateDocument({ ...metadataFixture(), recurrentState });
    // Then
    expect(result).toHaveProperty('recurrentState', recurrentState);
  });
  it.each([
    ['wrong sequence', { sequenceId: 1 }], ['wrong native identity', { nativeCaptureId: 'stale' }],
    ['missing summary', { after: undefined }], ['false continuity', { continuity: { ...recurrentFixture().continuity, arraysByteEqual: false } }],
    ['wrong snapshot capture', { before: { ...recurrentFixture().before, captureId: 'stale' } }],
    ['wrong label', { before: { ...recurrentFixture().before, label: 'before-decode' } }],
    ['wrong position', { after: { ...recurrentFixture().after, position: 16 } }],
    ['stale active row', { after: { ...recurrentFixture().after, mapping: { ...recurrentFixture().after.mapping, activeRow: null } } }],
    ['missing layer', { after: { ...recurrentFixture().after, arrays: recurrentFixture().after.arrays.slice(1) } }],
    ['nonfinite summary', { after: { ...recurrentFixture().after, arrays: recurrentFixture().after.arrays.map(a => ({ ...a, summary: { ...a.summary, rms: Infinity } })) } }],
    ['wrong state hash', { after: { ...recurrentFixture().after, arrays: recurrentFixture().after.arrays.map(a => ({ ...a, sha256: 'bad' })) } }],
  ])('rejects %s at the document boundary', (_name, changes) => {
    // Given
    const input = { ...metadataFixture(), recurrentState: { ...recurrentFixture(), ...changes } };
    // When / Then
    expect(() => validateDocument(input)).toThrow();
  });
  it('preserves per-tensor formula classification and source', () => {
    // Given
    const input = metadataFixture(), fields = { formulaClassification: 'symbolic-estimate', formulaSource: 'private source reference' };
    // When
    const result = validateDocument({ ...input, tensors: input.tensors.map(t => ({ ...t, ...fields })) });
    // Then: source copy equality only.
    expect(result.tensors[0]).toMatchObject(fields);
  });
  it('rejects unknown formula classification rather than offering arbitrary estimates', () => {
    // Given
    const input = metadataFixture();
    // When / Then
    expect(() => validateDocument({ ...input, tensors: input.tensors.map(t => ({ ...t, formulaClassification: 'executable', formulaSource: 'private' })) })).toThrow();
  });
});
