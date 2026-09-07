import { describe, expect, it } from 'vitest';
import { entityLabel } from '../src/entity-label';

describe('entity display name', () => {
  it.each([
    ['model/final-normalization/rms_norm.1449', 'rms_norm'],
    ['model/final-normalization/mul.1450', 'mul'],
    ['block.7/convolution-history/ssm_conv.451', 'ssm_conv'],
    ['block.7/convolution-history/conv_1d.42', 'conv_1d'],
    ['op-main', 'op-main'],
  ])('displays operator %s without its path and capture index', (id, expected) => {
    // Given / When / Then
    expect(entityLabel({ id, kind: 'operator', parentId: 'stage-scan' })).toBe(expected);
  });

  it.each(['model', 'block', 'stage'] as const)('preserves the ID for %s entities', kind => {
    // Given / When / Then
    expect(entityLabel({ id: `${kind}-id`, kind, parentId: null })).toBe(`${kind}-id`);
  });
});
