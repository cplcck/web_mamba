import type { Entity } from './schema';

export function entityLabel(entity: Pick<Entity, 'id' | 'kind' | 'parentId'>): string {
  return entity.kind === 'operator' ? entity.parentId ?? entity.id : entity.id;
}
