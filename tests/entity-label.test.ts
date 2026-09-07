import { describe, expect, it } from 'vitest';
import { entityLabel } from '../src/entity-label';

describe('entity display name', () => {
  it('uses the actual parent even when an operator ID has no path suffix', () => {
    // Given / When / Then
    expect(entityLabel({ id: 'op-main', kind: 'operator', parentId: 'stage-scan' })).toBe('stage-scan');
  });

  it.each(['model', 'block', 'stage'] as const)('preserves the ID for %s entities', kind => {
    // Given / When / Then
    expect(entityLabel({ id: `${kind}-id`, kind, parentId: null })).toBe(`${kind}-id`);
  });
});
