import type { CaptureDocument, Entity } from './schema';

export function buildBlockFlowModel(document: CaptureDocument, selectedId: string) {
  const entities = new Map(document.entities.map(entity => [entity.id, entity]));
  const path: Entity[] = [];
  let ancestor = entities.get(selectedId);
  while (ancestor) { path.push(ancestor); ancestor = ancestor.parentId === null ? undefined : entities.get(ancestor.parentId); }
  const blocks = document.entities.filter(entity => entity.kind === 'block');
  const block = path.find(entity => entity.kind === 'block') ?? blocks[0];
  const stage = path.find(entity => entity.kind === 'stage');
  const children = (entity: Entity | undefined): readonly Entity[] => (entity?.children ?? []).flatMap(id => {
    const child = entities.get(id); return child ? [child] : [];
  });
  const root = document.entities.find(entity => entity.parentId === null);
  return { selectedId, blocks, block, stage, stages: children(block), operators: children(stage),
    modelEntities: root ? [root, ...children(root).filter(entity => entity.kind !== 'block')] : [] };
}

type BlockFlowModel = ReturnType<typeof buildBlockFlowModel>;

export function renderBlockFlow(host: HTMLElement, model: BlockFlowModel, select: (id: string) => void): void {
  const document = host.ownerDocument;
  const expanded = host.querySelector('.representative-block__toggle')?.getAttribute('aria-expanded') === 'true';
  const scrollLeft = host.querySelector('.stage-reel')?.scrollLeft ?? 0;
  const modelOpen = host.querySelector<HTMLDetailsElement>('.model-context')?.open ?? false;
  host.replaceChildren();
  const element = <K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ''): HTMLElementTagNameMap[K] => {
    const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
  };
  const action = (entity: Entity, className: string): HTMLButtonElement => {
    const node = element('button', className); node.type = 'button'; node.dataset.entityId = entity.id;
    node.dataset.focusKey = `${className}:${entity.id}`;
    node.setAttribute('aria-label', `${entity.kind} ${entity.id}`);
    node.setAttribute('aria-current', entity.id === model.selectedId ? 'page' : 'false');
    node.textContent = entity.id; node.addEventListener('click', () => select(entity.id)); return node;
  };
  const outline = element('section', 'representative-block'); outline.setAttribute('aria-label', '공통 Mamba block 구조');
  const header = element('header', 'representative-block__header');
  const heading = element('h2', 'representative-block__heading');
  const toggle = element('button', 'representative-block__toggle'); toggle.type = 'button';
  toggle.dataset.focusKey = 'representative-block'; toggle.setAttribute('aria-controls', 'block-choices');
  heading.append(toggle);
  const active = element('p', 'representative-block__data', `실제 데이터 · ${model.block?.id ?? '모델'} · weights / state는 블록마다 다릅니다.`);
  header.append(heading, active);
  const choices = element('div', 'block-choices'); choices.id = 'block-choices'; choices.setAttribute('role', 'group'); choices.setAttribute('aria-label', '실제 데이터 블록 선택');
  const setExpanded = (open: boolean): void => {
    toggle.setAttribute('aria-expanded', String(open)); choices.hidden = !open;
    toggle.textContent = `Mamba block × ${model.blocks.length} ${open ? '▾' : '▸'}`;
  };
  setExpanded(expanded); toggle.addEventListener('click', () => setExpanded(choices.hidden === true));
  for (const block of model.blocks) {
    const choice = element('button', 'block-choice', block.id); choice.type = 'button'; choice.dataset.entityId = block.id;
    choice.setAttribute('aria-pressed', String(block.id === model.block?.id));
    choice.addEventListener('click', () => {
      setExpanded(false); select(block.id);
      host.querySelector<HTMLButtonElement>('.representative-block__toggle')?.focus({ preventScroll: true });
    });
    choices.append(choice);
  }
  const flowHeading = element('div', 'stage-flow__heading');
  const overview = model.block ? action(model.block, 'block-overview') : element('span', 'block-overview');
  overview.textContent = `${model.block?.id ?? '모델'} · 전체 경계`;
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
  reel.append(list); outline.append(header, choices, flowHeading, reel); host.append(outline); reel.scrollLeft = scrollLeft;
  const currentStage = cards.find(card => card.dataset.entityId === model.stage?.id);
  if (currentStage) reveal(currentStage);
  if (model.stage) {
    const detail = element('section', 'stage-detail'); detail.setAttribute('aria-label', `Operators · ${model.stage.id}`);
    detail.append(element('h3', 'stage-detail__title', `${model.stage.id} · ${model.operators.length} operators`));
    const operators = element('ol', 'operator-choices');
    for (const operator of model.operators) { const item = element('li', 'operator-choices__item'); item.append(action(operator, 'operator-choice')); operators.append(item); }
    detail.append(operators); host.append(detail);
  }
  const modelContext = element('details', 'model-context'); modelContext.open = modelOpen;
  modelContext.append(element('summary', 'model-context__summary', '모델 경계 · embedding / final layers'));
  const modelChoices = element('div', 'model-context__choices');
  for (const entity of model.modelEntities) modelChoices.append(action(entity, 'model-choice'));
  modelContext.append(modelChoices); host.append(modelContext);
}
