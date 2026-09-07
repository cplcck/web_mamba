import type { CaptureDocument, Entity } from './schema';
import { entityLabel } from './entity-label';

export function buildBlockFlowModel(document: CaptureDocument, selectedId: string) {
  const entities = new Map(document.entities.map(entity => [entity.id, entity]));
  const path: Entity[] = [];
  let ancestor = entities.get(selectedId);
  while (ancestor) { path.push(ancestor); ancestor = ancestor.parentId === null ? undefined : entities.get(ancestor.parentId); }
  const blocks = document.entities.filter(entity => entity.kind === 'block');
  const block = path.find(entity => entity.kind === 'block');
  const stage = path.find(entity => entity.kind === 'stage');
  const children = (entity: Entity | undefined): readonly Entity[] => (entity?.children ?? []).flatMap(id => {
    const child = entities.get(id); return child ? [child] : [];
  });
  const root = document.entities.find(entity => entity.parentId === null);
  const architecture = children(root).filter(entity => entity.kind !== 'block' || entity === blocks[0])
    .map(entity => entity.kind === 'block' ? { kind: 'blocks' as const } : { kind: 'entity' as const, entity });
  return { selectedId, root, architecture, blocks, block, stage, stages: children(block), operators: children(stage) };
}

type BlockFlowModel = ReturnType<typeof buildBlockFlowModel>;

export function renderBlockFlow(host: HTMLElement, model: BlockFlowModel, select: (id: string) => void): void {
  const document = host.ownerDocument;
  const expanded = host.querySelector('.representative-block__toggle')?.getAttribute('aria-expanded') === 'true';
  const scrollLeft = host.querySelector('.stage-reel')?.scrollLeft ?? 0;
  host.replaceChildren();
  const element = <K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ''): HTMLElementTagNameMap[K] => {
    const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
  };
  const action = (entity: Entity, className: string): HTMLButtonElement => {
    const node = element('button', className); node.type = 'button'; node.dataset.entityId = entity.id;
    node.dataset.focusKey = `${className}:${entity.id}`;
    node.setAttribute('aria-description', `${entity.kind} ${entity.id}`);
    node.setAttribute('aria-current', entity.id === model.selectedId ? 'page' : 'false');
    node.textContent = entityLabel(entity); node.addEventListener('click', () => select(entity.id)); return node;
  };
  const architecture = element('section', 'model-architecture'); architecture.setAttribute('aria-label', 'Model architecture');
  const header = element('header', 'model-architecture__header');
  header.append(element('h2', 'model-architecture__heading', 'Model architecture'));
  if (model.root) { const back = action(model.root, 'model-overview'); back.textContent = '모델 전체로 돌아가기'; header.append(back); }
  const flow = element('ol', 'model-flow'); flow.setAttribute('aria-label', 'Model architecture order');
  const toggle = element('button', 'representative-block__toggle'); toggle.type = 'button';
  toggle.dataset.focusKey = 'representative-block'; toggle.setAttribute('aria-controls', 'block-choices');
  toggle.setAttribute('aria-current', model.block ? 'step' : 'false');
  const toggleState = element('span', 'model-card__detail');
  toggle.append(element('strong', 'model-card__name', `Mamba block × ${model.blocks.length}`), toggleState);
  const labels: Readonly<Record<string, string>> = { 'model/embedding': 'Embedding', 'model/final-normalization': 'Final Norm', 'model/final-projection': 'Output' };
  for (const [index, node] of model.architecture.entries()) {
    const item = element('li', 'model-flow__item');
    const card = node.kind === 'blocks' ? toggle : action(node.entity, 'model-card');
    card.classList.add('model-card'); card.dataset.architectureNode = node.kind === 'blocks' ? 'blocks' : node.entity.id;
    if (node.kind === 'entity') {
      card.replaceChildren(element('strong', 'model-card__name', labels[node.entity.id] ?? node.entity.id));
      if (node.entity.id === model.stage?.id && node.entity.id !== model.selectedId) card.setAttribute('aria-current', 'step');
    }
    item.append(card);
    if (index < model.architecture.length - 1) { const arrow = element('span', 'model-flow__arrow', '→'); arrow.setAttribute('aria-hidden', 'true'); item.append(arrow); }
    flow.append(item);
  }
  const hint = element('p', 'model-architecture__hint', '모델 구조 순서 · tensor edge가 아닙니다. 블록을 선택하면 실제 단계와 데이터를 표시합니다.');
  const choices = element('div', 'block-choices'); choices.id = 'block-choices'; choices.setAttribute('role', 'group'); choices.setAttribute('aria-label', '실제 데이터 블록 선택');
  const setExpanded = (open: boolean): void => {
    toggle.setAttribute('aria-expanded', String(open)); choices.hidden = !open;
    toggleState.textContent = `${open ? '▾' : '▸'} ${open ? '블록 선택' : '블록 펼치기'}`;
  };
  setExpanded(expanded); toggle.addEventListener('click', () => setExpanded(choices.hidden === true));
  for (const block of model.blocks) {
    const choice = element('button', 'block-choice', block.id); choice.type = 'button'; choice.dataset.entityId = block.id;
    choice.setAttribute('aria-pressed', String(block.id === model.block?.id));
    choice.addEventListener('click', () => {
      setExpanded(false); toggle.focus({ preventScroll: true });
      select(block.id);
    });
    choices.append(choice);
  }
  architecture.append(header, flow, hint, choices); host.append(architecture);
  if (model.block) {
    const outline = element('section', 'block-detail'); outline.setAttribute('aria-label', `실제 데이터 · ${model.block.id}`);
    const flowHeading = element('div', 'stage-flow__heading');
    const overview = action(model.block, 'block-overview');
    overview.textContent = `${model.block.id} · 실제 데이터 / 전체 경계`;
    const hint = element('p', 'stage-flow__hint', `${model.stages.length} stages → 가로 스크롤 · 방향키로 이동 · 단계 선택`);
    hint.id = 'stage-flow-hint';
    flowHeading.append(overview, hint);
    const reel = element('div', 'stage-reel'); reel.tabIndex = 0;
    reel.setAttribute('role', 'region'); reel.setAttribute('aria-label', 'Mamba block stages'); reel.setAttribute('aria-describedby', hint.id);
    const list = element('ol', 'stage-flow');
    const reveal = (card: HTMLButtonElement): void => {
      const inset = parseFloat(getComputedStyle(reel).paddingInlineStart);
      const left = card.getBoundingClientRect().left - reel.getBoundingClientRect().left + reel.scrollLeft;
      if (left - inset < reel.scrollLeft) reel.scrollLeft = left - inset;
      else if (left + card.offsetWidth + inset > reel.scrollLeft + reel.clientWidth) reel.scrollLeft = left + card.offsetWidth + inset - reel.clientWidth;
    };
    const cards = model.stages.map((stage, index) => {
      const item = element('li', 'stage-flow__item');
      const card = action(stage, 'stage-card'); card.replaceChildren();
      if (stage.id === model.stage?.id && stage.id !== model.selectedId) card.setAttribute('aria-current', 'step');
      const name = stage.id.slice(stage.id.lastIndexOf('/') + 1).replaceAll('-', ' ');
      card.append(element('span', 'stage-card__number', `${String(index + 1).padStart(2, '0')} →`),
        element('strong', 'stage-card__name', name), element('span', 'stage-card__id', stage.id),
        element('span', 'stage-card__count', `${stage.children.length} operators`));
      card.addEventListener('focus', () => reveal(card));
      item.append(card); list.append(item); return card;
    });
    reel.addEventListener('keydown', event => {
      const index = cards.findIndex(card => card === document.activeElement);
      let next: number;
      switch (event.key) {
        case 'ArrowRight': next = Math.min(cards.length - 1, index + 1); break;
        case 'ArrowLeft': next = Math.max(0, index - 1); break;
        case 'Home': next = 0; break;
        case 'End': next = cards.length - 1; break;
        default: return;
      }
      event.preventDefault(); cards[next]?.focus({ preventScroll: true });
    });
    reel.append(list); outline.append(flowHeading, reel); host.append(outline); reel.scrollLeft = scrollLeft;
    const currentStage = cards.find(card => card.dataset.entityId === model.stage?.id);
    if (currentStage) reveal(currentStage);
  }
  if (model.stage) {
    const detail = element('section', 'stage-detail'); detail.setAttribute('aria-label', `Operators · ${model.stage.id}`);
    detail.append(element('h3', 'stage-detail__title', `${model.stage.id} · ${model.operators.length} operators`));
    const operators = element('ol', 'operator-choices');
    for (const operator of model.operators) { const item = element('li', 'operator-choices__item'); item.append(action(operator, 'operator-choice')); operators.append(item); }
    detail.append(operators); host.append(detail);
  }
}
