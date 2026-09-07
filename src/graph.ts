import type { CaptureDocument, Entity, OperatorEvidence, Tensor } from './schema';
import { entityLabel } from './entity-label';

export type GraphEdgeRole = 'input' | 'output' | 'weight' | 'state';
type Marker = 'circle' | 'diamond' | 'square' | 'triangle';
export type GraphNode = Readonly<{ entityId: string; kind: Entity['kind']; label: string; selected: boolean; evidence: OperatorEvidence | null }>;
export type GraphEdge = Readonly<{
  key: string; tensorId: string; tensorName: string; role: GraphEdgeRole;
  sourceId: string; targetId: string; label: string; marker: Marker;
}>;
export type GraphModel = Readonly<{ nodes: readonly GraphNode[]; edges: readonly GraphEdge[]; listEdges: readonly GraphEdge[] }>;
type SelectEntity = (entityId: string) => void;

const markers: Readonly<Record<GraphEdgeRole, Marker>> = {
  input: 'circle', output: 'triangle', weight: 'diamond', state: 'square',
};
const entitiesById = (document: CaptureDocument): ReadonlyMap<string, Entity> =>
  new Map(document.entities.map(entity => [entity.id, entity]));
const tensorsById = (document: CaptureDocument): ReadonlyMap<string, Tensor> =>
  new Map(document.tensors.map(tensor => [tensor.id, tensor]));

function descendants(document: CaptureDocument, entity: Entity): Set<string> {
  const map = entitiesById(document), result = new Set<string>(), pending = [...entity.children];
  while (pending.length > 0) {
    const id = pending.shift();
    if (id === undefined || result.has(id)) continue;
    result.add(id);
    const child = map.get(id);
    if (child) pending.push(...child.children);
  }
  return result;
}
function roleFor(tensor: Tensor, boundary: GraphEdgeRole): GraphEdgeRole {
  return tensor.role === 'state' ? 'state' : tensor.role === 'weight' ? 'weight' : boundary;
}
function addEdge(edges: GraphEdge[], seen: Set<string>, tensor: Tensor, role: GraphEdgeRole, sourceId: string, targetId: string): void {
  const key = `${role}:${tensor.id}:${sourceId}->${targetId}`;
  if (seen.has(key)) return;
  seen.add(key);
  edges.push({ key, tensorId: tensor.id, tensorName: tensor.name, role, sourceId, targetId, label: `${role} · ${tensor.name} · ${sourceId} → ${targetId}`, marker: markers[role] });
}

export function buildGraphModel(document: CaptureDocument, entityId: string): GraphModel {
  const entityMap = entitiesById(document), tensorMap = tensorsById(document), entity = entityMap.get(entityId);
  if (!entity) return { nodes: [], edges: [], listEdges: [] };
  const operatorIds = new Set<string>(entity.kind === 'operator' ? [entity.id] : [...descendants(document, entity)].filter(id => entityMap.get(id)?.kind === 'operator'));
  // Model/block collapse is presentation of actual children, never cloned operator records.
  const grouped = entity.kind === 'model' || entity.kind === 'block';
  const visibleIds = grouped ? new Set([entity.id, ...entity.children]) : operatorIds;
  const owners = new Map<string, string>();
  for (const id of operatorIds) {
    let owner = entityMap.get(id);
    while (owner && !visibleIds.has(owner.id)) owner = owner.parentId === null ? undefined : entityMap.get(owner.parentId);
    if (owner) owners.set(id, owner.id);
  }
  const nodes = document.entities.filter(item => visibleIds.has(item.id)).map(item => {
    const outputId = item.outputTensorIds.length === 1 ? item.outputTensorIds[0] : undefined;
    const output = item.kind === 'operator' && outputId ? tensorMap.get(outputId) : undefined;
    const evidence = output?.op !== undefined && output.opParamsI32 !== undefined && output.schedulerObserved !== undefined && output.arithmeticExecution !== undefined
      ? { op: output.op, opParamsI32: output.opParamsI32, schedulerObserved: output.schedulerObserved, arithmeticExecution: output.arithmeticExecution } : null;
    return { entityId: item.id, kind: item.kind, label: entityLabel(item), selected: item.id === entityId, evidence };
  });
  const edges: GraphEdge[] = [], seen = new Set<string>();
  for (const tensor of document.tensors) {
    const producers = tensor.producerIds.filter(id => operatorIds.has(id));
    const consumers = tensor.consumerIds.filter(id => operatorIds.has(id));
    for (const producer of producers) for (const consumer of consumers) {
      const source = owners.get(producer), target = owners.get(consumer);
      if (source && target && (!grouped || source !== target)) addEdge(edges, seen, tensor, roleFor(tensor, 'output'), source, target);
    }
  }
  const addBoundary = (ids: readonly string[], boundary: GraphEdgeRole): void => {
    for (const id of ids) {
      const tensor = tensorMap.get(id);
      if (!tensor) continue;
      const role = roleFor(tensor, boundary);
      if (boundary === 'input' || boundary === 'weight') {
        for (const consumer of tensor.consumerIds.filter(consumerId => operatorIds.has(consumerId))) { const target = owners.get(consumer); if (target) addEdge(edges, seen, tensor, role, `boundary:${boundary}:${tensor.id}`, target); }
      } else {
        for (const producer of tensor.producerIds.filter(producerId => operatorIds.has(producerId))) { const source = owners.get(producer); if (source) addEdge(edges, seen, tensor, role, source, `boundary:output:${tensor.id}`); }
      }
    }
  };
  addBoundary(entity.inputTensorIds, 'input');
  addBoundary(entity.outputTensorIds, 'output');
  addBoundary(entity.weightTensorIds, 'weight');
  const endpointLabel = (id: string): string => { const endpoint = entityMap.get(id); return endpoint ? entityLabel(endpoint) : id; };
  const displayEdges = edges.map(edge => ({ ...edge, label: `${edge.role} · ${edge.tensorName} · ${endpointLabel(edge.sourceId)} → ${endpointLabel(edge.targetId)}` }));
  return { nodes, edges: displayEdges, listEdges: displayEdges };
}

function nodeLabel(node: GraphNode): string {
  return node.kind === 'block' || node.kind === 'operator' ? node.label : `${node.kind} · ${node.label}`;
}
function edgeAttributes(element: Element, edge: GraphEdge): void {
  element.setAttribute('data-edge-key', edge.key); element.setAttribute('data-edge-role', edge.role);
  element.setAttribute('data-source-id', edge.sourceId); element.setAttribute('data-target-id', edge.targetId);
}
function markerShape(document: Document, marker: Marker, x: number, y: number): SVGElement {
  const tag = marker === 'circle' ? 'circle' : marker === 'square' ? 'rect' : marker === 'diamond' ? 'path' : 'path';
  const shape = document.createElementNS('http://www.w3.org/2000/svg', tag);
  if (tag === 'circle') { shape.setAttribute('cx', String(x)); shape.setAttribute('cy', String(y)); shape.setAttribute('r', '5'); }
  if (tag === 'rect') { shape.setAttribute('x', String(x - 5)); shape.setAttribute('y', String(y - 5)); shape.setAttribute('width', '10'); shape.setAttribute('height', '10'); }
  if (tag === 'path') { shape.setAttribute('d', marker === 'diamond' ? `M ${x} ${y - 6} L ${x + 6} ${y} L ${x} ${y + 6} L ${x - 6} ${y} Z` : `M ${x} ${y - 6} L ${x + 6} ${y + 6} L ${x - 6} ${y + 6} Z`); }
  shape.setAttribute('class', `edge-marker edge-marker--${marker}`);
  return shape;
}
function interactive(element: SVGElement, entityId: string, select: SelectEntity, cleanup: Array<() => void>): void {
  const activate = (event: Event): void => { event.preventDefault(); select(entityId); };
  const keydown = (event: KeyboardEvent): void => { if (event.key === 'Enter' || event.key === ' ') activate(event); };
  element.addEventListener('click', activate); element.addEventListener('keydown', keydown);
  cleanup.push(() => { element.removeEventListener('click', activate); element.removeEventListener('keydown', keydown); });
}

export function renderGraph(host: HTMLElement, document: Document, model: GraphModel, select: SelectEntity): () => void {
  const cleanup: Array<() => void> = [], svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'graph-svg'); svg.setAttribute('role', 'group'); svg.setAttribute('aria-label', 'Entity and tensor flow');
  const title = document.createElementNS('http://www.w3.org/2000/svg', 'title'); title.textContent = 'Entity and tensor flow'; svg.append(title);
  const viewport = document.createElement('div'); viewport.className = 'graph-viewport'; viewport.append(svg);
  const toolbar = document.createElement('div'); toolbar.className = 'graph-toolbar';
  const fit = document.createElement('button'); fit.type = 'button'; fit.className = 'graph-toolbar__fit'; fit.textContent = '그래프 맞춤';
  const reset = document.createElement('button'); reset.type = 'button'; reset.className = 'graph-toolbar__reset'; reset.textContent = '그래프 초기화';
  const legend = document.createElement('figcaption'); legend.className = 'graph-legend'; legend.textContent = 'input ○ ┄ · output △ ─ · weight ◇ ┈ · state □ ┅';
  const list = document.createElement('ol'); list.className = 'graph-list'; list.setAttribute('aria-label', 'Entity and tensor edge list');
  // Wrapped HTML in SVG keeps exact identifiers readable instead of clipping fixed text rows.
  const textBox = (text: string, className: string): { box: SVGForeignObjectElement; content: HTMLDivElement } => {
    const box = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject'), content = document.createElement('div');
    content.className = className; content.textContent = text; box.append(content); return { box, content };
  };
  const edgeViews = model.edges.map(edge => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g'), line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    edgeAttributes(line, edge); line.setAttribute('class', `graph-edge graph-edge--${edge.role}`); line.setAttribute('aria-label', edge.label);
    group.append(line); svg.append(group);
    const caption = textBox(edge.label, `graph-edge__caption graph-edge__caption--${edge.role}`); svg.append(caption.box);
    return { edge, group, line, ...caption };
  });
  const nodeViews = model.nodes.map(node => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g'), rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    group.setAttribute('class', `graph-node${node.selected ? ' graph-node--selected' : ''}${node.evidence?.arithmeticExecution === false ? ' graph-node--metadata' : ''}`);
    group.setAttribute('data-entity-id', node.entityId); group.setAttribute('data-focus-key', `graph-node:${node.entityId}`);
    group.setAttribute('tabindex', '0'); group.setAttribute('role', 'button'); group.setAttribute('aria-pressed', String(node.selected)); group.setAttribute('aria-label', nodeLabel(node));
    group.setAttribute('aria-description', `${node.kind} ${node.entityId}`);
    if (node.evidence) { group.dataset.op = node.evidence.op; group.dataset.schedulerObserved = String(node.evidence.schedulerObserved); group.dataset.arithmeticExecution = String(node.evidence.arithmeticExecution); }
    rect.setAttribute('class', 'graph-node__shape'); const text = textBox(nodeLabel(node), 'graph-node__content');
    group.append(rect, text.box); svg.append(group); interactive(group, node.entityId, select, cleanup);
    const item = document.createElement('li'), button = document.createElement('button'); button.type = 'button';
    button.className = `graph-list__node${node.selected ? ' graph-list__node--selected' : ''}`; button.dataset.entityId = node.entityId; button.dataset.focusKey = `graph-list:${node.entityId}`;
    button.setAttribute('aria-pressed', String(node.selected)); button.textContent = nodeLabel(node);
    button.setAttribute('aria-description', `${node.kind} ${node.entityId}`);
    if (node.evidence) { button.dataset.op = node.evidence.op; button.dataset.schedulerObserved = String(node.evidence.schedulerObserved); button.dataset.arithmeticExecution = String(node.evidence.arithmeticExecution); }
    const onClick = (): void => select(node.entityId); button.addEventListener('click', onClick); cleanup.push(() => button.removeEventListener('click', onClick)); item.append(button); list.append(item);
    return { node, group, rect, ...text };
  });
  for (const edge of model.listEdges) { const item = document.createElement('li'); item.className = `graph-list__edge graph-list__edge--${edge.role}`; edgeAttributes(item, edge); item.textContent = edge.label; list.append(item); }
  toolbar.append(fit, reset); host.replaceChildren(toolbar, legend, viewport, list);
  let fitted = false, lastWidth = -1;
  const layout = (): void => {
    const available = Math.max(44, viewport.clientWidth), narrow = available < 800;
    const width = fitted ? available : narrow ? Math.min(384, available) : 960;
    const columns = narrow ? 1 : Math.max(1, Math.min(3, Math.floor(width / 280)));
    const gap = 24, inset = 12, nodeWidth = (width - inset * 2 - gap * (columns - 1)) / columns;
    host.dataset.layout = columns === 1 ? 'vertical' : 'grid'; host.dataset.view = fitted ? 'fit' : 'natural';
    svg.setAttribute('width', String(width)); svg.style.inlineSize = `${width}px`;
    let y = 24;
    const positions = new Map<string, { x: number; y: number; height: number }>();
    for (let start = 0; start < nodeViews.length; start += columns) {
      const row = nodeViews.slice(start, start + columns);
      for (const view of row) view.box.setAttribute('width', String(nodeWidth - inset * 2));
      const height = Math.max(44, ...row.map(view => view.content.scrollHeight + inset * 2));
      for (const [column, view] of row.entries()) {
        const x = inset + column * (nodeWidth + gap); positions.set(view.node.entityId, { x, y, height });
        view.rect.setAttribute('x', String(x)); view.rect.setAttribute('y', String(y)); view.rect.setAttribute('width', String(nodeWidth)); view.rect.setAttribute('height', String(height));
        view.box.setAttribute('x', String(x + inset)); view.box.setAttribute('y', String(y + inset)); view.box.setAttribute('height', String(height - inset * 2));
      }
      y += height + 48;
    }
    for (const view of edgeViews) {
      const source = positions.get(view.edge.sourceId), target = positions.get(view.edge.targetId);
      const sx = source ? source.x + nodeWidth / 2 : 4, sy = source ? source.y + source.height : target?.y ?? 12;
      const tx = target ? target.x + nodeWidth / 2 : width - 4, ty = target?.y ?? sy + 24;
      view.line.setAttribute('d', `M ${sx} ${sy} C ${sx} ${(sy + ty) / 2}, ${tx} ${(sy + ty) / 2}, ${tx} ${ty}`);
      view.group.querySelector('.edge-marker')?.remove(); const marker = markerShape(document, view.edge.marker, tx, ty); marker.classList.add(`graph-edge--${view.edge.role}`); view.group.append(marker);
      view.box.setAttribute('width', String(width - inset * 2)); view.box.setAttribute('x', String(inset)); view.box.setAttribute('y', String(y));
      const height = Math.max(44, view.content.scrollHeight); view.box.setAttribute('height', String(height)); y += height + inset;
    }
    svg.setAttribute('height', String(y)); svg.setAttribute('viewBox', `0 0 ${width} ${y}`); lastWidth = available;
    host.dispatchEvent(new CustomEvent('graph-layout', { bubbles: true }));
  };
  const onFit = (): void => { fitted = true; layout(); viewport.scrollLeft = 0; viewport.scrollTop = 0; };
  const onReset = (): void => { fitted = false; layout(); viewport.scrollLeft = 0; viewport.scrollTop = 0; };
  fit.addEventListener('click', onFit); reset.addEventListener('click', onReset);
  let frame = 0;
  const observer = new ResizeObserver(() => { if (viewport.clientWidth !== lastWidth && frame === 0) frame = requestAnimationFrame(() => { frame = 0; layout(); }); }); observer.observe(viewport);
  layout();
  return () => { observer.disconnect(); cancelAnimationFrame(frame); fit.removeEventListener('click', onFit); reset.removeEventListener('click', onReset); for (const remove of cleanup) remove(); host.replaceChildren(); };
}
