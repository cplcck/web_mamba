import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { buildBlockFlowModel } from '../src/block-flow';
import { validateDocument } from '../src/schema';

const document = validateDocument(JSON.parse(readFileSync('public/data/prefill.json', 'utf8')));

describe('representative block context', () => {
  it.each(['block.0', 'block.23'])('counts all captured operators beneath %s, not just its stages', id => {
    // Given / When: the actual block contains eight stages and repeated operation names.
    const context = buildBlockFlowModel(document, id);
    // Then: each captured operator counts once, including metadata-only operators.
    expect(context.blockOperatorCount).toBe(50);
  });

  it.each([
    ['model/embedding', 1],
    ['model/final-normalization', 2],
    ['model/final-projection', 1],
  ] as const)('counts the operators beneath %s', (id, count) => {
    // Given / When
    const context = buildBlockFlowModel(document, id);
    // Then
    expect(context.stageOperatorCount).toBe(count);
  });

  it('keeps model root free of a chosen block and block detail', () => {
    // Given / When
    const context = buildBlockFlowModel(document, 'mamba-130m');
    // Then
    expect(context.block).toBeUndefined();
    expect(context.stage).toBeUndefined();
    expect(context.stages).toEqual([]);
    expect(context.operators).toEqual([]);
    expect(context.architecture.map(node => node.kind === 'blocks' ? 'blocks' : node.entity.id))
      .toEqual(['model/embedding', 'blocks', 'model/final-normalization', 'model/final-projection']);
  });
  it('uses the actual block and ordered stage records when selecting block.23', () => {
    // Given / When
    const context = buildBlockFlowModel(document, 'block.23');
    // Then: presentation preserves reference identity and never substitutes block.0 data.
    const block = document.entities.find(entity => entity.id === 'block.23');
    expect(context.block).toBe(block);
    expect(context.blocks).toHaveLength(24);
    expect(context.stages.map(stage => stage.id)).toEqual(block?.children);
    expect(context.stages).toHaveLength(8);
    for (const stage of context.stages) expect(stage).toBe(document.entities.find(entity => entity.id === stage.id));
    expect(context.operators).toEqual([]);
  });

  it('retains the real block and stage when an operator is selected', () => {
    // Given
    const stage = document.entities.find(entity => entity.id === 'block.7/input-projection-split');
    const operator = stage?.children[0];
    if (!operator) throw new Error('actual stage operator missing');
    // When
    const context = buildBlockFlowModel(document, operator);
    // Then
    expect(context.block?.id).toBe('block.7');
    expect(context.stage).toBe(stage);
    expect(context.operators.map(entity => entity.id)).toEqual(stage?.children);
    expect(context.stages).toHaveLength(8);
  });

  it('keeps model-wide stage selection distinct from the representative block', () => {
    // Given / When
    const context = buildBlockFlowModel(document, 'model/embedding');
    // Then: model data is not relabeled as block.0.
    expect(context.block).toBeUndefined();
    expect(context.stages).toEqual([]);
    expect(context.stage?.id).toBe('model/embedding');
    expect(context.selectedId).toBe('model/embedding');
    expect(context.operators.map(entity => entity.parentId)).toEqual(['model/embedding']);
    expect(context.root?.id).toBe('mamba-130m');
  });
});
