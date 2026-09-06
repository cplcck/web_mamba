import { buildGraphModel, renderGraph } from './graph';
import type { CaptureDocument, Entity } from './schema';

export type ExplorerScenario = 'prefill' | 'decode';
export type ExplorerSelection = Readonly<{ scenario: ExplorerScenario; entityId: string }>;
type SearchMatch = Readonly<{ matchText: string; matchKind: 'operator' | 'tensor' }>;
export type SearchResult = SearchMatch & Readonly<{ entityId: string; matches: readonly SearchMatch[] }>;
export interface ExplorerHandle { setScenario(scenario: ExplorerScenario): void; selectEntity(entityId: string): void; destroy(): void; }
type Documents = Record<ExplorerScenario, CaptureDocument>;
type SelectionListener = (selection: ExplorerSelection) => void;
const isScenario = (value: string): value is ExplorerScenario => value === 'prefill' || value === 'decode';
const mapEntities = (document: CaptureDocument): ReadonlyMap<string, Entity> => new Map(document.entities.map(entity => [entity.id, entity]));
function rootOf(document: CaptureDocument): Entity {
  const root = document.entities.find(entity => entity.parentId === null);
  if (!root) throw new Error('Capture document has no model root');
  return root;
}

export function encodeSelectionHash(selection: ExplorerSelection): string {
  return `#scenario=${selection.scenario}&entity=${encodeURIComponent(selection.entityId)}`;
}
export function decodeSelectionHash(hash: string): ExplorerSelection | null {
  const value = hash.startsWith('#') ? hash.slice(1) : hash;
  const params = new URLSearchParams(value);
  const scenario = params.get('scenario'), entityId = params.get('entity');
  if (scenario && entityId && isScenario(scenario) && entityId.length > 0) return { scenario, entityId };
  const [legacyScenario, ...legacyEntity] = value.split('/');
  if (legacyScenario && legacyEntity.length > 0 && isScenario(legacyScenario)) {
    try {
      const entityId = decodeURIComponent(legacyEntity.join('/'));
      if (entityId.length > 0) return { scenario: legacyScenario, entityId };
    } catch (error) {
      if (!(error instanceof URIError)) throw error;
    }
  }
  return null;
}
function ancestorFallback(source: CaptureDocument, target: CaptureDocument, entityId: string): string {
  const sourceMap = mapEntities(source), targetMap = mapEntities(target);
  let current = sourceMap.get(entityId);
  while (current) { if (targetMap.has(current.id)) return current.id; current = current.parentId === null ? undefined : sourceMap.get(current.parentId); }
  return rootOf(target).id;
}
export function resolveScenarioSelection(selection: ExplorerSelection, scenario: ExplorerScenario, documents: Documents): ExplorerSelection {
  const target = documents[scenario], targetMap = mapEntities(target);
  if (targetMap.has(selection.entityId)) return { scenario, entityId: selection.entityId };
  const source = documents[selection.scenario].entities.some(entity => entity.id === selection.entityId) ? documents[selection.scenario] : documents[selection.scenario === 'prefill' ? 'decode' : 'prefill'];
  return { scenario, entityId: ancestorFallback(source, target, selection.entityId) };
}

export function findSearchResults(document: CaptureDocument, query: string): readonly SearchResult[] {
  const normalize = (value: string): string => value.trim().replace(/\s+/g, ' ').toLocaleLowerCase();
  const needle = normalize(query);
  if (needle.length === 0) return [];
  const tensorMap = new Map(document.tensors.map(tensor => [tensor.id, tensor]));
  const operators = document.entities.filter(entity => entity.kind === 'operator');
  const results: SearchResult[] = [];
  for (const entity of operators) {
    const matches = new Map<string, SearchMatch>();
    for (const name of [entity.id, ...entity.outputTensorIds.flatMap(id => tensorMap.get(id)?.op ?? [])]) {
      if (normalize(name).includes(needle)) matches.set(`operator:${name}`, { matchText: name, matchKind: 'operator' });
    }
    for (const tensorId of new Set([...entity.inputTensorIds, ...entity.outputTensorIds, ...entity.weightTensorIds])) {
      const tensor = tensorMap.get(tensorId);
      if (tensor && normalize(tensor.name).includes(needle)) matches.set(`tensor:${tensor.name}`, { matchText: tensor.name, matchKind: 'tensor' });
    }
    const first = matches.values().next().value;
    if (first) results.push({ entityId: entity.id, ...first, matches: [...matches.values()] });
  }
  return results;
}
function element(document: Document, tag: string, className?: string): HTMLElement {
  const node = document.createElement(tag); if (className) node.className = className; return node;
}
function button(document: Document, className: string): HTMLButtonElement {
  const node = document.createElement('button'); node.type = 'button'; node.className = className; return node;
}
function searchInput(document: Document, className: string): HTMLInputElement {
  const node = document.createElement('input'); node.type = 'search'; node.className = className; return node;
}
function renderBreadcrumbs(root: HTMLElement, document: CaptureDocument, entityId: string, select: (id: string) => void): void {
  const map = mapEntities(document), path: Entity[] = []; let current = map.get(entityId);
  while (current) { path.unshift(current); current = current.parentId === null ? undefined : map.get(current.parentId); }
  const nav = root.querySelector('.breadcrumbs'); if (!(nav instanceof HTMLElement)) return; nav.replaceChildren();
  const list = element(nav.ownerDocument, 'ol', 'breadcrumbs__list');
  for (const entity of path) { const item = element(nav.ownerDocument, 'li', 'breadcrumbs__item'), link = button(nav.ownerDocument, 'breadcrumbs__link'); link.dataset.focusKey = `breadcrumb:${entity.id}`; link.dataset.entityId = entity.id; link.setAttribute('aria-label', `${entity.kind} ${entity.id}`); link.textContent = entity.id; link.addEventListener('click', () => select(entity.id)); item.append(link); list.append(item); }
  nav.append(list);
}
function renderHierarchy(root: HTMLElement, document: CaptureDocument, selectedId: string, query: string, select: (id: string) => void): void {
  const nav = root.querySelector('.hierarchy'); if (!(nav instanceof HTMLElement)) return;
  const previous = nav.querySelector('.hierarchy__list'); previous?.remove();
  const list = element(nav.ownerDocument, 'ol', 'hierarchy__list'), map = mapEntities(document), results = findSearchResults(document, query);
  const searching = query.trim().length > 0, narrow = nav.ownerDocument.defaultView?.matchMedia('(max-width: 680px)').matches ?? false;
  const path: Entity[] = []; let current = map.get(selectedId);
  while (current) { path.unshift(current); current = current.parentId === null ? undefined : map.get(current.parentId); }
  const ancestors = new Set(path.slice(0, -1).map(entity => entity.id));
  nav.dataset.layout = narrow ? 'drilldown' : 'tree';
  const add = (entity: Entity, parent: HTMLElement, matches: readonly SearchMatch[] = []): void => {
    const item = element(nav.ownerDocument, 'li', 'hierarchy__item'), row = button(nav.ownerDocument, 'hierarchy__row');
    row.dataset.entityId = entity.id; row.dataset.focusKey = `hierarchy:${entity.id}`;
    row.setAttribute('aria-label', `${entity.kind} ${entity.id}`); row.setAttribute('aria-current', entity.id === selectedId ? 'page' : 'false');
    row.classList.toggle('hierarchy__row--selected', entity.id === selectedId); row.classList.toggle('hierarchy__row--ancestor', ancestors.has(entity.id));
    row.textContent = `${entity.kind} · ${entity.id}`; row.addEventListener('click', () => select(entity.id)); item.append(row);
    for (const match of matches) { const text = element(nav.ownerDocument, 'span', 'hierarchy__match'); text.dataset.matchKind = match.matchKind; text.textContent = match.matchText; row.append(text); }
    if (!searching && !narrow && entity.children.length > 0) { const children = element(nav.ownerDocument, 'ol', 'hierarchy__children'); for (const childId of entity.children) { const child = map.get(childId); if (child) add(child, children); } item.append(children); }
    parent.append(item);
  };
  if (searching) {
    for (const result of results) { const entity = map.get(result.entityId); if (entity) add(entity, list, result.matches); }
    if (results.length === 0) { const item = element(nav.ownerDocument, 'li', 'hierarchy__empty'); item.textContent = '일치하는 operator 또는 tensor가 없습니다.'; list.append(item); }
  } else if (narrow) {
    for (const entity of path) add(entity, list);
    const selected = map.get(selectedId), parent = selected?.parentId ? map.get(selected.parentId) : undefined;
    for (const id of selected?.kind === 'operator' ? parent?.children ?? [] : selected?.children ?? []) { const child = map.get(id); if (child && id !== selectedId) add(child, list); }
  } else add(rootOf(document), list);
  const status = nav.querySelector<HTMLElement>('.hierarchy__search-status');
  if (status) { const count = results.reduce((sum, result) => sum + result.matches.length, 0); status.dataset.resultCount = String(results.length); status.dataset.matchCount = String(count); status.textContent = searching ? `${results.length}개 operator · ${count}개 이름 일치` : ''; }
  nav.append(list);
}
type SelectionReason = 'selected' | 'invalid-route' | 'ancestor-fallback';
function renderStatus(root: HTMLElement, selection: ExplorerSelection, reason: SelectionReason): void {
  const status = root.querySelector<HTMLElement>('.explorer__status');
  const messages = { selected: '선택됨', 'invalid-route': '유효하지 않은 선택 경로입니다. 표시 가능한 항목으로 이동했습니다.', 'ancestor-fallback': '이 scenario에 이전 항목이 없어 가장 가까운 상위 항목으로 이동했습니다.' };
  if (status) { status.dataset.reason = reason; status.textContent = `${messages[reason]} · ${selection.entityId} · ${selection.scenario}`; }
}
const isContentAnchor = (hash: string): boolean => hash.length > 1 && !/(?:^#|&)(?:scenario|entity)=|^#(?:prefill|decode)\//.test(hash);

export function createExplorer(host: HTMLElement, documents: Documents, onSelection: SelectionListener): ExplorerHandle {
  const document = host.ownerDocument, view = document.defaultView;
  const requested = decodeSelectionHash(view?.location.hash ?? ''), initial: ExplorerSelection = requested ? resolveScenarioSelection(requested, requested.scenario, documents) : { scenario: 'prefill', entityId: rootOf(documents.prefill).id };
  let current = initial, query = '', graphCleanup = (): void => undefined, destroyed = false;
  let reason: SelectionReason = (requested && requested.entityId !== initial.entityId) || (!requested && view?.location.hash && !isContentAnchor(view.location.hash)) ? 'invalid-route' : 'selected';
  const narrow = view?.matchMedia('(max-width: 680px)');
  host.classList.add('explorer');
  const breadcrumbs = element(document, 'nav', 'breadcrumbs'); breadcrumbs.setAttribute('aria-label', 'Breadcrumb');
  const hierarchy = element(document, 'nav', 'hierarchy'); hierarchy.setAttribute('aria-label', 'Model hierarchy');
  const searchLabel = element(document, 'label', 'hierarchy__search-label'); searchLabel.textContent = 'operator / tensor 검색'; const search = searchInput(document, 'hierarchy__search'); search.id = 'explorer-search'; search.placeholder = '이름으로 찾기'; searchLabel.append(search);
  const clear = button(document, 'hierarchy__clear'); clear.textContent = '검색 지우기'; clear.addEventListener('click', () => { search.value = ''; query = ''; renderHierarchy(host, documents[current.scenario], current.entityId, query, selectEntity); search.focus(); });
  const searchStatus = element(document, 'p', 'hierarchy__search-status'); searchStatus.setAttribute('role', 'status');
  hierarchy.append(searchLabel, clear, searchStatus);
  const graphHost = element(document, 'figure', 'operator-graph'), status = element(document, 'p', 'explorer__status'); status.setAttribute('role', 'status');
  host.replaceChildren(breadcrumbs, hierarchy, graphHost, status);
  const render = (): void => {
    if (destroyed) return;
    const active = document.activeElement, focusKey = active && host.contains(active) ? active.getAttribute('data-focus-key') : null;
    host.dataset.scenario = current.scenario; host.dataset.entityId = current.entityId;
    renderBreadcrumbs(host, documents[current.scenario], current.entityId, selectEntity); renderHierarchy(host, documents[current.scenario], current.entityId, query, selectEntity);
    graphCleanup(); graphCleanup = renderGraph(graphHost, document, buildGraphModel(documents[current.scenario], current.entityId), selectEntity); renderStatus(host, current, reason);
    const selected = mapEntities(documents[current.scenario]).get(current.entityId)!;
    const boundary = element(document, 'p', 'explorer__boundary');
    boundary.dataset.entityId = selected.id; boundary.dataset.kind = selected.kind; boundary.dataset.scenario = current.scenario;
    const kinds = { model: '모델', block: '블록', stage: '단계', operator: '연산자' };
    for (const [role, text] of [
      ['scope', `선택한 ${kinds[selected.kind]}의 경계에서`],
      ['input', '입력은 들어오는 텐서,'],
      ['output', '출력은 나가는 텐서,'],
      ['weight', '가중치는 참조하는 매개변수입니다.'],
    ] as const) {
      const clause = element(document, 'span', 'explorer__boundary-clause');
      clause.dataset.boundaryClause = role; clause.textContent = text;
      if (boundary.childNodes.length > 0) boundary.append(' ');
      boundary.append(clause);
    }
    graphHost.insertBefore(boundary, graphHost.querySelector('.graph-viewport'));
    if (focusKey) {
      const actions = [...host.querySelectorAll<HTMLElement | SVGElement>('[data-focus-key]')];
      const replacement = actions.find(node => node.getAttribute('data-focus-key') === focusKey) ?? actions.find(node => node.getAttribute('data-entity-id') === current.entityId);
      replacement?.focus({ preventScroll: true });
    }
  };
  let handledHash = view?.location.hash ?? '';
  const canonicalize = (): void => { if (view && view.location.hash !== encodeSelectionHash(current)) view.history.replaceState({ ...current }, '', encodeSelectionHash(current)); handledHash = view?.location.hash ?? ''; };
  const apply = (next: ExplorerSelection, push: boolean, nextReason: SelectionReason): void => {
    if (destroyed) return;
    reason = nextReason;
    if (next.scenario === current.scenario && next.entityId === current.entityId) { canonicalize(); renderStatus(host, current, reason); return; }
    current = next;
    if (push && view) view.history.pushState({ ...next }, '', encodeSelectionHash(next)); else canonicalize();
    handledHash = view?.location.hash ?? '';
    render(); onSelection(current);
  };
  function selectEntity(entityId: string): void { const next = resolveScenarioSelection({ scenario: current.scenario, entityId }, current.scenario, documents); apply(next, true, next.entityId === entityId ? 'selected' : 'invalid-route'); }
  const setScenario = (scenario: ExplorerScenario): void => { const next = resolveScenarioSelection(current, scenario, documents); apply(next, true, next.entityId === current.entityId ? 'selected' : 'ancestor-fallback'); };
  const navigate = (): void => {
    const hash = view?.location.hash ?? ''; if (hash === handledHash) return;
    handledHash = hash; if (isContentAnchor(hash)) return;
    const requested = decodeSelectionHash(hash), next = requested ? resolveScenarioSelection(requested, requested.scenario, documents) : { scenario: 'prefill' as const, entityId: rootOf(documents.prefill).id };
    apply(next, false, requested && requested.entityId === next.entityId ? 'selected' : 'invalid-route');
  };
  const onSearch = (): void => { query = search.value; renderHierarchy(host, documents[current.scenario], current.entityId, query, selectEntity); };
  const onKey = (event: KeyboardEvent): void => { if (event.key === 'Escape' && document.activeElement === search) { search.value = ''; query = ''; onSearch(); } };
  search.addEventListener('input', onSearch); search.addEventListener('keydown', onKey); view?.addEventListener('popstate', navigate); view?.addEventListener('hashchange', navigate);
  narrow?.addEventListener('change', render);
  if (!isContentAnchor(view?.location.hash ?? '')) canonicalize();
  render(); onSelection(current);
  return { setScenario, selectEntity, destroy: () => { if (destroyed) return; destroyed = true; graphCleanup(); search.removeEventListener('input', onSearch); search.removeEventListener('keydown', onKey); view?.removeEventListener('popstate', navigate); view?.removeEventListener('hashchange', navigate); narrow?.removeEventListener('change', render); host.replaceChildren(); host.classList.remove('explorer'); } };
}
