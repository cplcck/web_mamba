import type { CaptureDocument, Entity, OperatorEvidence, Tensor } from './schema';
import { entityLabel } from './entity-label';

export type GraphEdgeRole = 'input' | 'output' | 'weight' | 'state';
export type GraphNode = Readonly<{ entityId: string; kind: Entity['kind']; label: string; selected: boolean; evidence: OperatorEvidence | null }>;
export type GraphEdge = Readonly<{
  key: string; tensorId: string; tensorName: string; role: GraphEdgeRole;
  sourceId: string; targetId: string; label: string;
}>;
export type GraphModel = Readonly<{ nodes: readonly GraphNode[]; edges: readonly GraphEdge[]; listEdges: readonly GraphEdge[] }>;
type SelectEntity = (entityId: string) => void;
type BoxPosition = Readonly<{ x: number; y: number; width: number; height: number }>;
type TextBox = Readonly<{ box: SVGForeignObjectElement; content: HTMLDivElement }>;

const graphGeometry = { inset: 16, cardInset: 12, sideGap: 24, rowGap: 32, portGap: 16, arrowOffset: 8, minCardHeight: 72, maxCardWidth: 320, naturalWidth: 960 } as const;
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
  edges.push({ key, tensorId: tensor.id, tensorName: tensor.name, role, sourceId, targetId, label: `${role} · ${tensor.name} · ${sourceId} → ${targetId}` });
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
        for (const consumer of tensor.consumerIds.filter(consumerId => operatorIds.has(consumerId))) {
          const target = owners.get(consumer);
          if (target) addEdge(edges, seen, tensor, role, `boundary:${boundary}:${tensor.id}`, target);
        }
      } else {
        for (const producer of tensor.producerIds.filter(producerId => operatorIds.has(producerId))) {
          const source = owners.get(producer);
          if (source) addEdge(edges, seen, tensor, role, source, `boundary:output:${tensor.id}`);
        }
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
function interactive(element: SVGElement, entityId: string, select: SelectEntity, cleanup: Array<() => void>): void {
  const activate = (event: Event): void => { event.preventDefault(); select(entityId); };
  const keydown = (event: KeyboardEvent): void => { if (event.key === 'Enter' || event.key === ' ') activate(event); };
  element.addEventListener('click', activate); element.addEventListener('keydown', keydown);
  cleanup.push(() => { element.removeEventListener('click', activate); element.removeEventListener('keydown', keydown); });
}

function endpointRole(endpoint: string): 'input' | 'output' | 'weight' | null {
  for (const role of ['input', 'output', 'weight'] as const) {
    if (endpoint.startsWith(`boundary:${role}:`)) return role;
  }
  return null;
}
function setBox(box: SVGForeignObjectElement | SVGRectElement, position: BoxPosition): void {
  box.setAttribute('x', String(position.x)); box.setAttribute('y', String(position.y));
  box.setAttribute('width', String(position.width)); box.setAttribute('height', String(position.height));
}
export function renderGraph(host: HTMLElement, document: Document, model: GraphModel, select: SelectEntity): () => void {
  const cleanup: Array<() => void> = [], svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'graph-svg'); svg.setAttribute('role', 'group'); svg.setAttribute('aria-label', 'Entity and tensor flow');
  const title = document.createElementNS('http://www.w3.org/2000/svg', 'title'); title.textContent = 'Entity and tensor flow'; svg.append(title);
  const viewport = document.createElement('div'); viewport.className = 'graph-viewport'; viewport.append(svg);
  const toolbar = document.createElement('div'); toolbar.className = 'graph-toolbar';
  const fit = document.createElement('button'); fit.type = 'button'; fit.className = 'graph-toolbar__fit'; fit.textContent = '흐름도 너비 맞추기';
  const reset = document.createElement('button'); reset.type = 'button'; reset.className = 'graph-toolbar__reset'; reset.textContent = '기본 보기로 돌아가기';
  const list = document.createElement('ol'); list.className = 'graph-list'; list.setAttribute('aria-label', 'Entity and tensor edge list');
  const textBox = (text: string, className: string): TextBox => {
    const box = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject'), content = document.createElement('div');
    content.className = className; content.textContent = text; box.append(content); return { box, content };
  };
  const edgeViews = model.edges.map(edge => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g'), line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    edgeAttributes(line, edge); line.setAttribute('class', `graph-edge graph-edge--${edge.role}`); line.setAttribute('aria-label', edge.label);
    const edgeTitle = document.createElementNS('http://www.w3.org/2000/svg', 'title'); edgeTitle.classList.add('graph-edge__caption'); edgeTitle.textContent = edge.label;
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    arrow.setAttribute('class', `graph-arrow graph-edge--${edge.role}`); arrow.setAttribute('d', 'M -5 -5 L 0 0 L 5 -5'); arrow.setAttribute('aria-hidden', 'true');
    line.append(edgeTitle); group.append(line, arrow); svg.append(group);
    return { edge, group, line, arrow };
  });
  const nodeViews = model.nodes.map(node => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g'), rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    group.setAttribute('class', `graph-node${node.selected ? ' graph-node--selected' : ''}${node.evidence?.arithmeticExecution === false ? ' graph-node--metadata' : ''}`);
    group.setAttribute('data-entity-id', node.entityId); group.setAttribute('data-focus-key', `graph-node:${node.entityId}`);
    group.setAttribute('tabindex', '0'); group.setAttribute('role', 'button'); group.setAttribute('aria-pressed', String(node.selected)); group.setAttribute('aria-label', nodeLabel(node));
    group.setAttribute('aria-description', `${node.kind} ${node.entityId}`);
    if (node.evidence) { group.dataset.op = node.evidence.op; group.dataset.schedulerObserved = String(node.evidence.schedulerObserved); group.dataset.arithmeticExecution = String(node.evidence.arithmeticExecution); }
    rect.setAttribute('class', 'graph-node__shape');
    const kindLabels = { model: '모델 / model', block: '블록 / block', stage: '단계 / stage', operator: '연산 / operator' } as const;
    const kind = textBox(kindLabels[node.kind], 'graph-node__kind'), text = textBox(node.label, 'graph-node__content');
    group.append(rect, kind.box, text.box); svg.append(group); interactive(group, node.entityId, select, cleanup);
    const item = document.createElement('li'), button = document.createElement('button'); button.type = 'button';
    button.className = `graph-list__node${node.selected ? ' graph-list__node--selected' : ''}`; button.dataset.entityId = node.entityId; button.dataset.focusKey = `graph-list:${node.entityId}`;
    button.setAttribute('aria-pressed', String(node.selected)); button.textContent = nodeLabel(node);
    button.setAttribute('aria-description', `${node.kind} ${node.entityId}`);
    if (node.evidence) { button.dataset.op = node.evidence.op; button.dataset.schedulerObserved = String(node.evidence.schedulerObserved); button.dataset.arithmeticExecution = String(node.evidence.arithmeticExecution); }
    const onClick = (): void => select(node.entityId); button.addEventListener('click', onClick); cleanup.push(() => button.removeEventListener('click', onClick)); item.append(button); list.append(item);
    return { node, group, rect, kind, text };
  });
  const endpointEdges = new Map<string, { readonly edge: GraphEdge; readonly boundaryRole: 'input' | 'output' | 'weight' }>();
  for (const edge of model.edges) {
    for (const endpoint of [edge.sourceId, edge.targetId]) {
      const boundaryRole = endpointRole(endpoint);
      if (boundaryRole && !endpointEdges.has(endpoint)) endpointEdges.set(endpoint, { edge, boundaryRole });
    }
  }
  const tensorViews = [...endpointEdges.entries()].map(([endpoint, { edge, boundaryRole }]) => {
    const isWeight = edge.role === 'weight' && boundaryRole !== 'output';
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `graph-tensor graph-tensor--${boundaryRole}${isWeight ? ' graph-tensor--weight' : ''}${edge.role === 'state' ? ' graph-tensor--state' : ''}`); group.dataset.endpointId = endpoint; group.dataset.tensorId = edge.tensorId; group.dataset.boundaryRole = boundaryRole;
    group.setAttribute('aria-label', `tensor ${edge.tensorName} / ${boundaryRole} / ${edge.role}`);
    const shape = document.createElementNS('http://www.w3.org/2000/svg', 'rect'); shape.setAttribute('class', 'graph-tensor__shape');
    const kindLabels = { input: '입력 텐서 / input', output: '출력 텐서 / output', weight: '가중치 텐서 / weight' } as const;
    const kind = textBox(`${kindLabels[isWeight ? 'weight' : boundaryRole]}${edge.role === 'state' ? ' / state' : ''}`, 'graph-tensor__kind');
    const name = textBox('tensor name : ', 'graph-tensor__content');
    const value = document.createElement('span'); value.className = 'graph-tensor__name'; value.textContent = edge.tensorName; name.content.append(value);
    group.append(shape, kind.box, name.box); svg.append(group);
    return { endpoint, boundaryRole, isWeight, consumerId: edge.targetId, group, shape, kind, name };
  });
  for (const edge of model.listEdges) {
    const item = document.createElement('li'); item.className = `graph-list__edge graph-list__edge--${edge.role}`; edgeAttributes(item, edge); item.textContent = edge.label; list.append(item);
  }
  toolbar.append(fit, reset); host.replaceChildren(toolbar, viewport, list);
  let fitted = false, lastWidth = -1;
  const incoming = new Map<string, GraphEdge[]>(), outgoing = new Map<string, GraphEdge[]>();
  for (const edge of model.edges) {
    const inputs = incoming.get(edge.targetId) ?? []; inputs.push(edge); incoming.set(edge.targetId, inputs);
    const outputs = outgoing.get(edge.sourceId) ?? []; outputs.push(edge); outgoing.set(edge.sourceId, outputs);
  }
  const cards = [
    ...tensorViews.filter(view => !view.isWeight && view.boundaryRole !== 'output').map(view => ({ id: view.endpoint, shape: view.shape, fields: [view.kind, view.name] })),
    ...nodeViews.map(view => ({ id: view.node.entityId, shape: view.rect, fields: [view.kind, view.text] })),
    ...tensorViews.filter(view => !view.isWeight && view.boundaryRole === 'output').map(view => ({ id: view.endpoint, shape: view.shape, fields: [view.kind, view.name] })),
  ];
  const weightCards = tensorViews.filter(view => view.isWeight).map(view => ({ id: view.endpoint, ownerId: view.consumerId, shape: view.shape, fields: [view.kind, view.name] }));
  const weightsByOwner = new Map<string, typeof weightCards>();
  for (const card of weightCards) {
    const weights = weightsByOwner.get(card.ownerId) ?? []; weights.push(card); weightsByOwner.set(card.ownerId, weights);
  }
  const weightOwners = new Map(weightCards.map(card => [card.id, card.ownerId]));
  const primaryOrder = new Map(cards.map((card, index) => [card.id, index]));
  const layout = (): void => {
    const available = Math.max(44, viewport.clientWidth);
    const width = fitted ? available : Math.min(graphGeometry.naturalWidth, available);
    const { inset, cardInset, sideGap, rowGap, portGap, arrowOffset, minCardHeight, maxCardWidth } = graphGeometry;
    const withWeights = weightCards.length > 0;
    const cardWidth = Math.max(24, Math.min(maxCardWidth, withWeights ? (width - inset * 2 - sideGap) / 2 : width - inset * 2));
    const mainX = (width - (withWeights ? cardWidth * 2 + sideGap : cardWidth)) / 2;
    const sideX = mainX + cardWidth + sideGap;
    host.dataset.layout = 'vertical'; host.dataset.view = fitted ? 'fit' : 'natural';
    svg.setAttribute('width', String(width)); svg.style.inlineSize = String(width) + 'px';
    const positions = new Map<string, BoxPosition>();
    const measure = (card: typeof cards[number]) => {
      const contentWidth = Math.max(1, cardWidth - cardInset * 2);
      for (const field of card.fields) field.box.setAttribute('width', String(contentWidth));
      const heights = card.fields.map(field => field.content.scrollHeight);
      const sidePorts = (incoming.get(card.id) ?? []).filter(edge => weightOwners.has(edge.sourceId)).length;
      const height = Math.max(minCardHeight, heights.reduce((sum, value) => sum + value, 0) + cardInset * 2, (sidePorts + 1) * portGap);
      return { card, heights, contentWidth, height };
    };
    const place = (measured: ReturnType<typeof measure>, x: number, y: number, height: number): void => {
      const position = { x, y, width: cardWidth, height };
      positions.set(measured.card.id, position); setBox(measured.card.shape, position);
      let textY = y + cardInset;
      for (const [index, field] of measured.card.fields.entries()) {
        const textHeight = measured.heights[index] ?? 0;
        setBox(field.box, { x: x + cardInset, y: textY, width: measured.contentWidth, height: textHeight });
        textY += textHeight;
      }
    };
    let nextY: number = inset;
    for (const card of cards) {
      const measured = measure(card), weights = (weightsByOwner.get(card.id) ?? []).map(measure);
      const weightHeight = weights.reduce((total, weight) => total + weight.height, 0) + Math.max(0, weights.length - 1) * rowGap;
      const height = Math.max(measured.height, weightHeight);
      place(measured, mainX, nextY, height);
      let weightY = nextY + (height - weightHeight) / 2;
      for (const weight of weights) { place(weight, sideX, weightY, weight.height); weightY += weight.height + rowGap; }
      nextY += height + rowGap;
    }
    for (const view of edgeViews) {
      const source = positions.get(view.edge.sourceId), target = positions.get(view.edge.targetId);
      if (!source || !target) throw new Error('Graph edge endpoint missing: ' + view.edge.key);
      if (weightOwners.has(view.edge.sourceId)) {
        const peers = (incoming.get(view.edge.targetId) ?? []).filter(edge => weightOwners.has(edge.sourceId));
        const ty = target.y + target.height * (peers.indexOf(view.edge) + 1) / (peers.length + 1);
        const tx = target.x + target.width, rail = tx + sideGap / 2;
        if (weightOwners.get(view.edge.sourceId) === view.edge.targetId) {
          const sy = source.y + source.height / 2;
          view.line.setAttribute('d', 'M ' + source.x + ' ' + sy + ' H ' + rail + ' V ' + ty + ' H ' + tx);
        } else {
          const bottom = source.y + source.height;
          view.line.setAttribute('d', 'M ' + (source.x + source.width / 2) + ' ' + bottom + ' V ' + (bottom + rowGap / 2) + ' H ' + rail + ' V ' + ty + ' H ' + tx);
        }
        view.arrow.dataset.direction = 'left';
        view.arrow.setAttribute('transform', 'translate(' + (tx + arrowOffset) + ' ' + ty + ') rotate(90)');
      } else {
        const sourcePeers = outgoing.get(view.edge.sourceId) ?? [];
        const targetPeers = (incoming.get(view.edge.targetId) ?? []).filter(edge => !weightOwners.has(edge.sourceId));
        const sx = source.x + source.width * (sourcePeers.indexOf(view.edge) + 1) / (sourcePeers.length + 1);
        const tx = target.x + target.width * (targetPeers.indexOf(view.edge) + 1) / (targetPeers.length + 1);
        const bottom = source.y + source.height;
        if ((primaryOrder.get(view.edge.sourceId) ?? -2) + 1 === primaryOrder.get(view.edge.targetId)) {
          const midY = (bottom + target.y) / 2;
          view.line.setAttribute('d', 'M ' + sx + ' ' + bottom + ' V ' + midY + ' H ' + tx + ' V ' + target.y);
        } else {
          const rail = mainX - inset / 2;
          view.line.setAttribute('d', 'M ' + sx + ' ' + bottom + ' V ' + (bottom + rowGap / 2) + ' H ' + rail + ' V ' + (target.y - rowGap / 2) + ' H ' + tx + ' V ' + target.y);
        }
        view.arrow.dataset.direction = 'down';
        view.arrow.setAttribute('transform', 'translate(' + tx + ' ' + (target.y - arrowOffset) + ')');
      }
    }
    const height = Math.max(inset * 2, nextY - rowGap + inset);
    svg.setAttribute('height', String(height)); svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    lastWidth = available;
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
