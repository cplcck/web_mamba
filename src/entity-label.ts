import type { Entity } from './schema';

export function entityLabel(entity: Pick<Entity, 'id' | 'kind' | 'parentId'>): string {
  return entity.kind === 'operator'
    ? entity.id.slice(entity.id.lastIndexOf('/') + 1).replace(/\.\d+$/, '')
    : entity.id;
}
