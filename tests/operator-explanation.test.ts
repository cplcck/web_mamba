import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { validateDocument, type CaptureDocument, type Entity, type Tensor } from '../src/schema';
import { buildOperatorExplanation, operatorSummary } from '../src/operator-explanation';

const documents = ['prefill', 'decode'].map(name => validateDocument(JSON.parse(readFileSync(new URL(`../public/data/${name}.json`, import.meta.url), 'utf8'))));
function captured(op: string, document = documents[0]) {
  if (!document) throw new Error('Missing capture fixture');
  const output = document.tensors.find(t => t.op === op);
  const entity = document.entities.find(e => e.kind === 'operator' && e.outputTensorIds.includes(output?.id ?? ''));
  if (!output || !entity) throw new Error(`Missing ${op} fixture`);
  return { document, entity, output };
}
function withOutput(fixture: ReturnType<typeof captured>, output: Tensor): CaptureDocument {
  return { ...fixture.document, tensors: fixture.document.tensors.map(t => t.id === output.id ? output : t) };
}
function floatBits(value: number): number {
  const bytes = new DataView(new ArrayBuffer(4));
  bytes.setFloat32(0, value, true);
  return bytes.getInt32(0, true);
}
function facts(document: CaptureDocument, entity: Entity) {
  const explanation = buildOperatorExplanation(document, entity);
  expect(explanation).not.toBeNull();
  return Object.fromEntries(explanation?.facts.map(f => [f.key, f.value]) ?? []);
}

describe('operator explanations from captured GGML evidence', () => {
  it('covers all 2320 operators, all 16 kinds, and exact ordered operand references', () => {
    // Given / When
    const results = documents.flatMap(document => document.entities.filter(e => e.kind === 'operator').map(entity => ({
      document, entity, explanation: buildOperatorExplanation(document, entity),
    })));
    // Then
    expect(results).toHaveLength(2320);
    expect([...new Set(results.map(r => r.explanation?.op))].sort()).toEqual([
      'ADD', 'CONCAT', 'CONT', 'CPY', 'GET_ROWS', 'GLU', 'MUL', 'MUL_MAT', 'RESHAPE', 'RMS_NORM', 'SCALE', 'SSM_CONV', 'SSM_SCAN', 'TRANSPOSE', 'UNARY', 'VIEW',
    ]);
    for (const { document, entity, explanation } of results) {
      expect(explanation).not.toBeNull();
      if (!explanation) throw new Error(`Missing explanation: ${entity.id}`);
      const inputIds = [...new Set([...entity.inputTensorIds, ...entity.weightTensorIds])];
      expect(explanation.inputs.map(o => o.tensor.id)).toEqual(inputIds);
      expect(explanation.outputs.map(o => o.tensor.id)).toEqual(entity.outputTensorIds);
      for (const operand of [...explanation.inputs, ...explanation.outputs]) {
        expect(operand.tensor).toBe(document.tensors.find(t => t.id === operand.tensor.id));
        expect(operand.label.length).toBeGreaterThan(0);
      }
      expect(explanation.summary.length).toBeGreaterThan(0);
      expect(explanation.operation.length).toBeGreaterThan(0);
      expect(explanation.notes.length).toBeGreaterThan(0);
      expect(explanation.facts.length).toBeGreaterThan(0);
      expect(new Set(explanation.facts.map(f => f.key)).size).toBe(explanation.facts.length);
      expect(explanation.facts.find(f => f.key === 'effect')?.value).not.toBe('unavailable');
      expect(explanation.facts.find(f => f.key === 'elementValues')?.value).toBe('unavailable');
      expect(explanation.summary).toBe(operatorSummary(explanation.outputs[0]?.tensor));
    }
  });
  it('decodes captured RMS epsilon from float bits, not the integer parameter', () => {
    // Given
    const { document, entity } = captured('RMS_NORM');
    // When
    const result = facts(document, entity);
    // Then
    expect(result.epsilon).toBe(Math.fround(1e-5));
    expect(result.normalizationAxis).toBe(0);
    expect(result.normalizationWidth).toBe(768);
  });
  it.each([[0, 0], [-2.5, 0.75], [1, -4]])('decodes SCALE floats %s and %s including sign bits and zero', (scale, bias) => {
    // Given
    const fixture = captured('SCALE');
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [floatBits(scale), floatBits(bias)] });
    // When
    const result = facts(document, fixture.entity);
    // Then
    expect(result).toMatchObject({ scale, bias, effect: 'compute', shapeRelation: 'equal' });
  });
  it.each([[0, 'ABS'], [6, 'RELU'], [10, 'SILU'], [15, 'SOFTPLUS'], [21, 'TRUNC'], [999, 'unknown']])('decodes UNARY subtype %s', (id, subtype) => {
    // Given
    const fixture = captured('UNARY');
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [id] });
    // When / Then
    expect(facts(document, fixture.entity)).toMatchObject({ subtypeId: id, subtype, effect: 'compute' });
  });
  it.each([[0, 'REGLU'], [1, 'GEGLU'], [2, 'SWIGLU'], [3, 'SWIGLU_OAI'], [4, 'GEGLU_ERF'], [5, 'GEGLU_QUICK'], [999, 'unknown']])('decodes GLU subtype %s without reversing its gate', (id, subtype) => {
    // Given
    const fixture = captured('GLU');
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [id, 1, floatBits(1.702), floatBits(7)] });
    // When / Then
    expect(facts(document, fixture.entity)).toMatchObject({ subtypeId: id, subtype, gateTensorId: fixture.entity.inputTensorIds[0],
      valueTensorId: fixture.entity.inputTensorIds[1], gateOffsetElements: 0, valueOffsetElements: 0, gluLayout: 'separate' });
  });
  it.each([0, 1])('uses swapped=%s only for packed GLU halves', swapped => {
    // Given
    const fixture = captured('GLU');
    const sourceId = fixture.entity.inputTensorIds[0];
    if (!sourceId) throw new Error('Missing gate');
    const entity = { ...fixture.entity, inputTensorIds: [sourceId] };
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [2, swapped] });
    // When / Then
    expect(facts(document, entity)).toMatchObject({ gluLayout: 'packed', gateTensorId: sourceId, valueTensorId: sourceId,
      gateOffsetElements: swapped ? 768 : 0, valueOffsetElements: swapped ? 0 : 768 });
  });
  it('decodes the additional OAI GLU float parameters', () => {
    // Given
    const fixture = captured('GLU');
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [3, 0, floatBits(1.702), floatBits(7)] });
    // When / Then
    expect(facts(document, fixture.entity)).toMatchObject({ alpha: Math.fround(1.702), limit: 7 });
  });
  it('reports GET_ROWS index identity and count without inventing index values', () => {
    // Given
    const { document, entity } = captured('GET_ROWS');
    // When / Then
    expect(facts(document, entity)).toMatchObject({ dataTensorId: 'split:prefill:t1', indexTensorId: 'split:prefill:t2',
      indexCount: 16, indexValues: 'unavailable', rowWidth: 768, availableRows: 50280, rowAxis: 1 });
  });
  it('reports equal padded RESHAPE shapes as metadata, without inventing a logical rank', () => {
    // Given
    const { document, entity } = captured('RESHAPE');
    // When / Then
    expect(facts(document, entity)).toMatchObject({ effect: 'metadata-only', shapeRelation: 'equal',
      inputShape: [4608, 1, 1, 1], outputShape: [4608, 1, 1, 1], logicalRank: 'unavailable' });
  });
  it('keeps a VIEW immediate source offset separate from its accumulated storage offset', () => {
    // Given
    const fixture = captured('VIEW');
    const document = withOutput(fixture, { ...fixture.output, viewOffsetBytes: 4096, opParamsI32: [128, 0] });
    // When / Then
    expect(facts(document, fixture.entity)).toMatchObject({ sourceTensorId: 'split:prefill:t3', viewSourceId: 'split:prefill:t4',
      viewOffsetBytes: 4096, sourceOffsetBytes: 128, outputStridesBytes: [4, 18432, 18432, 18432], effect: 'metadata-only' });
  });
  it('decodes 64-bit little-endian VIEW offsets instead of truncating to signed I32', () => {
    // Given
    const fixture = captured('VIEW');
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [-1, 1] });
    // When / Then
    expect(facts(document, fixture.entity).sourceOffsetBytes).toBe(8589934591);
  });
  it('identifies CPY destination storage from output view evidence, including empty copies', () => {
    // Given
    const { document, entity } = captured('CPY');
    // When / Then
    expect(facts(document, entity)).toMatchObject({ effect: 'copy', sourceTensorId: 'split:prefill:t11', destinationTensorId: 'split:prefill:t12',
      viewSourceId: 'split:prefill:t4', viewOffsetBytes: 18432, outputNumel: 0 });
  });
  it('reports TRANSPOSE axis and stride swapping without claiming a copy', () => {
    // Given
    const { document, entity } = captured('TRANSPOSE');
    // When / Then
    expect(facts(document, entity)).toMatchObject({ effect: 'metadata-only', axisPermutation: [1, 0, 2, 3],
      inputShape: [1536, 16, 1, 1], outputShape: [16, 1536, 1, 1],
      inputStridesBytes: [4, 12288, 196608, 196608], outputStridesBytes: [12288, 4, 196608, 196608] });
  });
  it('reports CONT copying with actual before and after strides', () => {
    // Given
    const { document, entity } = captured('CONT');
    // When / Then
    expect(facts(document, entity)).toMatchObject({ effect: 'copy', shapeRelation: 'equal',
      inputStridesBytes: [4, 12288, 196608, 196608], outputStridesBytes: [4, 6144, 98304, 98304] });
  });
  it.each([0, 1, 2, 3])('decodes CONCAT ne axis %s from params, not shapes', axis => {
    // Given
    const fixture = captured('CONCAT');
    const document = withOutput(fixture, { ...fixture.output, opParamsI32: [axis] });
    // When / Then
    expect(facts(document, fixture.entity).concatAxis).toBe(axis);
  });
  it('reports GGML MUL_MAT reduction and native output axes with weight input identity', () => {
    // Given
    const { document, entity } = captured('MUL_MAT');
    // When / Then
    expect(facts(document, entity)).toMatchObject({ matrixATensorId: 'split:prefill:t20', matrixBTensorId: 'split:prefill:t18',
      reductionAxis: 0, reductionSize: 768, outputAxis0Size: 3072, outputAxis1Size: 16, outputBatchShape: [1, 1] });
  });
  it.each(documents)('reports SSM_CONV and SSM_SCAN segment dimensions for $scenario.name', document => {
    // Given
    const conv = captured('SSM_CONV', document), scan = captured('SSM_SCAN', document);
    const tokens = document.scenario.name === 'prefill' ? 16 : 1;
    // When
    const convFacts = facts(document, conv.entity), scanFacts = facts(document, scan.entity);
    // Then
    expect(convFacts).toMatchObject({ kernelWidth: 4, historyWidth: 3, channels: 1536, tokenCount: tokens, windowStep: 1 });
    expect(scanFacts).toMatchObject({ stateWidth: 16, headDimension: 1, headCount: 1536, tokenCount: tokens, sequenceCount: 1,
      stateSlots: 1, activationNumel: tokens * 1536, stateNumel: 24576, stateOffsetBytes: tokens * 1536 * 4,
      activationShape: [1, 1536, tokens, 1], stateShape: [16, 1, 1536, 1], stateIndexTensorId: `split:${document.scenario.name}:t7` });
    expect(Number(scanFacts.activationNumel) + Number(scanFacts.stateNumel)).toBe(scan.output.numel);
  });
  it.each(['MUL', 'ADD'])('distinguishes same-shaped %s computation from unchanged values', op => {
    // Given
    const { document, entity } = captured(op);
    // When / Then
    expect(facts(document, entity)).toMatchObject({ effect: 'compute', shapeRelation: 'equal', broadcastAxes: [1], elementValues: 'unavailable' });
  });
  it('de-duplicates overlapping inputs and weights while preserving all output identities', () => {
    // Given
    const fixture = captured('MUL');
    const entity = { ...fixture.entity, inputTensorIds: [...fixture.entity.inputTensorIds, ...fixture.entity.weightTensorIds],
      outputTensorIds: [fixture.output.id, ...fixture.entity.inputTensorIds] };
    // When
    const result = buildOperatorExplanation(fixture.document, entity);
    // Then
    expect(result?.inputs.map(o => o.tensor.id)).toEqual(entity.inputTensorIds);
    expect(result?.outputs.map(o => o.tensor.id)).toEqual(entity.outputTensorIds);
  });
  it.each(['UNARY', 'GLU', 'SCALE', 'RMS_NORM', 'CONCAT'])('keeps %s explanations when optional params are unavailable', op => {
    // Given
    const fixture = captured(op);
    const { opParamsI32: _params, ...output } = fixture.output;
    // When
    const result = buildOperatorExplanation(withOutput(fixture, output), fixture.entity);
    // Then
    expect(result?.op).toBe(op);
    expect(result?.facts.find(f => f.key === 'parameterEvidence')?.value).toBe('unavailable');
  });
  it('keeps unknown opcode evidence without mislabeling it as a supported computation', () => {
    // Given
    const fixture = captured('MUL');
    const document = withOutput(fixture, { ...fixture.output, op: 'FUTURE_OP' });
    // When / Then
    expect(facts(document, fixture.entity)).toMatchObject({ effect: 'unavailable' });
    expect(buildOperatorExplanation(document, fixture.entity)?.op).toBe('FUTURE_OP');
  });
  it('returns null for a non-operator or absent opcode evidence', () => {
    // Given
    const fixture = captured('MUL');
    const { op: _op, opParamsI32: _params, ...output } = fixture.output;
    // When / Then
    expect(buildOperatorExplanation(fixture.document, { ...fixture.entity, kind: 'stage' })).toBeNull();
    expect(buildOperatorExplanation(withOutput(fixture, output), fixture.entity)).toBeNull();
    expect(buildOperatorExplanation(fixture.document, { ...fixture.entity, outputTensorIds: [] })).toBeNull();
  });
});
