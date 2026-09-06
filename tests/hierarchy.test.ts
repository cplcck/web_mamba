import { describe, expect, it } from 'vitest';
import { buildGraphModel, type GraphEdgeRole } from '../src/graph';
import {
  decodeSelectionHash,
  encodeSelectionHash,
  findSearchResults,
  resolveScenarioSelection,
  type ExplorerSelection,
} from '../src/explorer';
import { validateDocument, type CaptureDocument } from '../src/schema';

const digest = 'a'.repeat(64);

function tensor(
  id: string,
  role: 'activation' | 'weight' | 'state',
  name: string,
  producerIds: readonly string[],
  consumerIds: readonly string[],
) {
  return {
    id,
    name,
    role,
    dtype: role === 'weight' ? 'F16' : 'F32',
    nativeShape: [2, 1, 1, 1],
    strides: role === 'weight' ? [2, 4, 4, 4] : [4, 8, 8, 8],
    logicalShape: [2],
    axisLabels: ['features'],
    numel: 2,
    typeBlockSize: 1,
    logicalBytes: role === 'weight' ? 4 : 8,
    storage: {
      ggmlNbytes: role === 'weight' ? 4 : 8,
      requiredAllocBytes: role === 'weight' ? 4 : null,
      bufferId: `buffer-${id}`,
      bufferOffsetBytes: 0,
      bufferBytes: 16,
      observationEpoch: 0,
      allocatorSlotBytes: null,
      allocatorSlotReason: 'not exposed',
    },
    viewSourceId: null,
    viewOffsetBytes: null,
    producerIds,
    consumerIds,
    provenance: 'synthetic-test-only',
    classification: 'observed',
  };
}

function makeDocument(name: 'prefill' | 'decode', includeMissingOperator: boolean): CaptureDocument {
  const operatorIds = includeMissingOperator ? ['op-main', 'missing-op'] : ['op-main'];
  const entities = [
    { id: 'model', kind: 'model', parentId: null, children: ['block-00'], inputTensorIds: ['input', 'state-before'], outputTensorIds: ['output', 'state-after'], weightTensorIds: ['weight'] },
    { id: 'block-00', kind: 'block', parentId: 'model', children: ['stage-scan'], inputTensorIds: ['input', 'state-before'], outputTensorIds: ['output', 'state-after'], weightTensorIds: ['weight'] },
    { id: 'stage-scan', kind: 'stage', parentId: 'block-00', children: operatorIds, inputTensorIds: ['input', 'state-before'], outputTensorIds: ['output', 'state-after'], weightTensorIds: ['weight'] },
    { id: 'op-main', kind: 'operator', parentId: 'stage-scan', children: [], inputTensorIds: ['input', 'state-before'], outputTensorIds: ['output', 'state-after'], weightTensorIds: ['weight'] },
    ...(includeMissingOperator ? [{ id: 'missing-op', kind: 'operator', parentId: 'stage-scan', children: [], inputTensorIds: [], outputTensorIds: [], weightTensorIds: [] }] : []),
  ];
  const raw = {
    schemaVersion: 1,
    captureId: `capture-${name}`,
    ggufSha256: digest,
    scenario: { name, dimensions: { P: 1, T: name === 'prefill' ? 16 : 1, Q: name === 'prefill' ? 16 : 1, O: 1 } },
    entities,
    tensors: [
      tensor('input', 'activation', 'tokens/input', [], ['op-main']),
      tensor('state-before', 'state', 'recurrent/state-before', [], ['op-main']),
      tensor('weight', 'weight', 'mamba.weight.original', [], ['op-main']),
      tensor('output', 'activation', 'logits/output', ['op-main'], []),
      tensor('state-after', 'state', 'recurrent/state-after', ['op-main'], []),
    ],
  };
  return validateDocument(raw);
}

const documents = {
  prefill: makeDocument('prefill', true),
  decode: makeDocument('decode', false),
};

describe('hierarchy navigation contract', () => {
  it('round-trips scenario and entity IDs through a stable deep link', () => {
    // Given
    const selection: ExplorerSelection = { scenario: 'decode', entityId: 'op/main' };
    // When
    const hash = encodeSelectionHash(selection);
    // Then
    expect(decodeSelectionHash(hash)).toEqual(selection);
  });

  it('falls back to the nearest semantic ancestor when a scenario lacks the selected entity', () => {
    // Given
    const selected: ExplorerSelection = { scenario: 'prefill', entityId: 'missing-op' };
    // When
    const next = resolveScenarioSelection(selected, 'decode', documents);
    // Then
    expect(next).toEqual({ scenario: 'decode', entityId: 'stage-scan' });
  });

  it('resolves a scenario-missing deep link through the other captured hierarchy', () => {
    // Given: a decode deep link names an operator only present in prefill.
    const selected: ExplorerSelection = { scenario: 'decode', entityId: 'missing-op' };
    // When / Then
    expect(resolveScenarioSelection(selected, 'decode', documents)).toEqual({ scenario: 'decode', entityId: 'stage-scan' });
  });

  it('treats unknown and malformed deep links as non-selections', () => {
    // Given
    const unknown = '#scenario=decode&entity=not-in-document';
    // When / Then
    expect(decodeSelectionHash(unknown)).toEqual({ scenario: 'decode', entityId: 'not-in-document' });
    expect(decodeSelectionHash('#decode/%E0%A4%A')).toBeNull();
    expect(decodeSelectionHash('#scenario=other&entity=op-main')).toBeNull();
  });

  it('searches original operator and tensor names without changing hierarchy IDs', () => {
    // Given
    const query = 'mamba.weight.original';
    // When
    const results = findSearchResults(documents.prefill, query);
    // Then
    expect(results.map(result => result.entityId)).toEqual(['op-main']);
    expect(results[0]?.matchText).toBe(query);
  });

  it('deduplicates a tensor name repeated across an operator input and output', () => {
    // Given: a reciprocal self edge references the same original name twice.
    const document = documents.prefill;
    const output = document.tensors.find(t => t.id === 'output');
    if (!output) throw new Error('fixture output absent');
    const repeated = { ...document, tensors: document.tensors.map(t => t.id === 'output' ? { ...t, consumerIds: ['op-main'] } : t), entities: document.entities.map(e => e.id === 'op-main' ? { ...e, inputTensorIds: [...e.inputTensorIds, 'output'] } : e) };
    // When / Then
    expect(findSearchResults(repeated, 'logits/output')).toHaveLength(1);
  });

  it('returns one entity with every unique original match and normalizes whitespace', () => {
    // Given
    const doc = { ...documents.prefill, tensors: documents.prefill.tensors.map(tensor => ({ ...tensor, name: `original  shared ${tensor.id}` })) };
    // When
    const results = findSearchResults(doc, '  ORIGINAL\t shared ');
    // Then
    expect(results.map(result => result.entityId)).toEqual(['op-main']);
    expect(results[0]?.matches.map(match => match.matchText)).toEqual(['original  shared input', 'original  shared state-before', 'original  shared output', 'original  shared state-after', 'original  shared weight']);
    expect(findSearchResults(doc, ' \t ')).toEqual([]);
  });

  it('preserves a shared tensor fan-out as distinct endpoint pairs', () => {
    // Given: two actual operator consumers share one input.
    const doc = validateDocument({ ...documents.prefill,
      entities: documents.prefill.entities.map(e => e.id === 'missing-op' ? { ...e, inputTensorIds: ['input'] } : e),
      tensors: documents.prefill.tensors.map(t => t.id === 'input' ? { ...t, consumerIds: ['op-main', 'missing-op'] } : t),
    });
    // When / Then
    expect(buildGraphModel(doc, 'stage-scan').edges.filter(edge => edge.tensorId === 'input').map(edge => [edge.sourceId, edge.targetId, edge.role])).toEqual([
      ['boundary:input:input', 'op-main', 'input'], ['boundary:input:input', 'missing-op', 'input'],
    ]);
  });

  it('maps model flow to actual immediate child entities without substituting operators', () => {
    // Given / When
    const graph = buildGraphModel(documents.prefill, 'model');
    // Then: context plus individually selectable block, with remapped actual boundary endpoints.
    expect(graph.nodes.map(node => node.entityId)).toEqual(['model', 'block-00']);
    expect(graph.edges.some(edge => edge.targetId === 'block-00')).toBe(true);
    expect(graph.edges.every(edge => !edge.sourceId.startsWith('op-') && !edge.targetId.startsWith('op-'))).toBe(true);
  });

  it.each([['SSM_SCAN', true], ['VIEW', false]] as const)('exposes %s evidence from the validated single output tensor', (op, arithmeticExecution) => {
    // Given: opcode is unrelated to the entity ID, and evidence is tensor-level.
    const evidence = { op, opParamsI32: Array(16).fill(0), schedulerObserved: true, arithmeticExecution };
    const document = validateDocument({ ...documents.decode,
      entities: documents.decode.entities.map(entity => ({ ...entity, outputTensorIds: entity.outputTensorIds.filter(id => id !== 'state-after') })),
      tensors: documents.decode.tensors.filter(t => t.id !== 'state-after').map(t => t.id === 'output' ? { ...t, ...evidence } : t),
    });
    // When / Then
    expect(buildGraphModel(document, 'op-main').nodes[0]).toMatchObject({ evidence });
    expect(findSearchResults(document, op).map(result => result.entityId)).toEqual(['op-main']);
  });

  it('derives graph list edges from the same actual tensors as the SVG model', () => {
    // Given
    const graph = buildGraphModel(documents.prefill, 'stage-scan');
    const roles = new Set<GraphEdgeRole>(graph.edges.map(edge => edge.role));
    // When
    const svgKeys = graph.edges.map(edge => edge.key);
    const listKeys = graph.listEdges.map(edge => edge.key);
    // Then
    expect(graph.nodes.map(node => node.entityId)).toEqual(['op-main', 'missing-op']);
    expect(roles).toEqual(new Set(['input', 'output', 'state', 'weight']));
    expect(listKeys).toEqual(svgKeys);
    expect(graph.edges.every(edge => edge.tensorId.length > 0)).toBe(true);
  });
});
