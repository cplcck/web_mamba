import { readFileSync } from 'node:fs';
import { test, expect } from '@playwright/test';
import { validateDocument } from '../src/schema';
import { arm, changed, closeInspector, installReadiness, mounted, openAdvanced } from './browser-helpers';

const capture = validateDocument(JSON.parse(readFileSync('public/data/prefill.json', 'utf8')));
const architectureIds = ['model/embedding', 'blocks', 'model/final-normalization', 'model/final-projection'];
test.beforeEach(async ({ page }) => { await installReadiness(page); });

test('operators display opcodes without capture indices and retain independent selections', async ({ page }) => {
  // Given: two actual operators belonging to the same model stage.
  const stage = capture.entities.find(entity => entity.id === 'model/final-normalization');
  if (!stage) throw new Error('Final normalization fixture missing');
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`./#scenario=prefill&entity=${encodeURIComponent(stage.id)}`); await mounted(page);
  expect(await page.locator('.operator-choice').allTextContents()).toEqual(['rms_norm', 'mul']);
  // When / Then: shortened names do not change individual operator actions.
  for (const id of stage.children) {
    const entity = capture.entities.find(entity => entity.id === id);
    if (!entity) throw new Error('Operator fixture missing');
    const output = capture.tensors.find(tensor => entity.outputTensorIds.includes(tensor.id));
    if (!output?.op) throw new Error('Operator output fixture missing');
    const label = output.op.toLowerCase();
    await arm(page, `.inspector[data-entity-id="${id}"]`);
    await page.locator(`.operator-choice[data-entity-id="${id}"]`).click(); await changed(page);
    expect(new URL(page.url()).hash).toBe(`#scenario=prefill&entity=${encodeURIComponent(id)}`);
    expect(await page.locator('.inspector__title').textContent()).toBe(label);
    expect(await page.locator('.breadcrumbs [aria-current=page]').textContent()).toBe(label);
    expect(await page.locator('.summary-section--weights').getAttribute('data-count')).toBe(String(entity.weightTensorIds.length));
    expect(await page.locator(`.operator-choice[data-entity-id="${id}"]`).getAttribute('aria-description')).toContain(id);
    expect(await page.locator('.graph-node__content').allTextContents()).toEqual([label]);
    expect(await page.locator('.graph-list__node').allTextContents()).toEqual([label]);
    expect(await page.locator('.graph-list__edge').allTextContents()).toEqual(expect.not.arrayContaining([expect.stringContaining(id)]));
    await openAdvanced(page);
    expect(await page.locator('#operator-evidence').count()).toBe(1);
    for (const field of ['opParamsI32', 'schedulerObserved', 'arithmeticExecution'] as const) {
      expect(await page.locator(`#operator-evidence [data-field="${field}"] td`).textContent()).toBe(JSON.stringify(output[field]));
    }
  }
  // Original IDs remain searchable while the result name shows only the opcode.
  await arm(page, '.hierarchy__search-status', 'data-result-count', '1');
  await page.locator('.hierarchy__search').fill(stage.children[0] ?? ''); await changed(page);
  expect(await page.locator('.hierarchy__row').allTextContents()).toEqual(['rms_norm']);
});

test('navigation accessible names include their visible labels', async ({ page }) => {
  // Given: model and block navigation use the real rendered labels.
  for (const entityId of ['mamba-130m', 'block.23']) {
    await page.goto(`./#scenario=prefill&entity=${entityId}`); await mounted(page); await closeInspector(page);
    // When / Then: voice-control names identify the same visible action.
    for (const action of await page.locator('.model-card[data-entity-id], .model-overview, .block-overview').all()) {
      const label = (await action.innerText()).replace(/\s+/g, ' ').trim();
      expect(await action.and(page.getByRole('button', { name: label })).count()).toBe(1);
    }
  }
});

test('a mobile deep link reveals its selected stage inside the reel', async ({ page }) => {
  // Given / When: a stage beyond the first mobile reel viewport is addressed directly.
  await page.goto('./#scenario=decode&entity=block.23%2Fselective-scan-state'); await mounted(page); await closeInspector(page);
  // Then: context names the actual block and the selected card is visible without document overflow.
  expect(await page.locator('.block-overview').getAttribute('data-entity-id')).toBe('block.23');
  expect(await page.locator('.model-card').count()).toBe(4);
  const bounds = await page.locator('.stage-card[aria-current=page]').evaluate(node => {
    const card = node.getBoundingClientRect(), reel = node.closest('.stage-reel')?.getBoundingClientRect();
    return { left: card.left, right: card.right, reelLeft: reel?.left, reelRight: reel?.right };
  });
  expect(bounds.left).toBeGreaterThanOrEqual(bounds.reelLeft ?? Infinity);
  expect(bounds.right).toBeLessThanOrEqual(bounds.reelRight ?? -Infinity);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  test(`model root drills through representative to actual block ${viewport.width}`, async ({ page }, info) => {
    // Given: both public landing routes, not a fixture or a block deep link.
    await page.setViewportSize(viewport);
    for (const url of ['./', './#scenario=prefill&entity=mamba-130m']) {
      await page.goto(url); await mounted(page); await closeInspector(page);
      // Then: one semantic four-node architecture, no implicit selected block or stages.
      expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe('mamba-130m');
      expect(new URL(page.url()).hash).toBe('#scenario=prefill&entity=mamba-130m');
      expect(await page.locator('.model-card').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-architecture-node')))).toEqual(architectureIds);
      expect(await page.locator('.representative-block__toggle').count()).toBe(1);
      expect(await page.locator('.representative-block__toggle').getAttribute('data-entity-id')).toBeNull();
      expect(await page.locator('.stage-card, .block-overview, .operator-choice').count()).toBe(0);
      expect(await page.locator('.block-choice[aria-pressed=true]').count()).toBe(0);
      expect(await page.locator('.block-choice:visible').count()).toBe(0);
      expect(await page.locator('.graph-evidence').getAttribute('open')).toBeNull();
    }
    const representative = page.locator('.representative-block__toggle');
    expect(await representative.getAttribute('aria-expanded')).toBe('false');
    const architecture = await page.locator('.model-card').evaluateAll(nodes => nodes.map(node => {
      const rect = node.getBoundingClientRect();
      return { x: rect.x, y: rect.y, right: rect.right, bottom: rect.bottom, width: rect.width, height: rect.height };
    }));
    expect(new Set(architecture.map(rect => rect.y)).size).toBe(1);
    architecture.forEach((rect, index) => {
      expect(rect.x).toBeGreaterThanOrEqual(0); expect(rect.right).toBeLessThanOrEqual(viewport.width);
      expect(rect.bottom).toBeLessThan(viewport.height); expect(rect.height).toBeGreaterThanOrEqual(112);
      if (index > 0) expect(rect.x).toBeGreaterThan(architecture[index - 1]?.right ?? Infinity);
    });
    expect(await page.locator('.model-flow__arrow').count()).toBe(3);
    expect(await page.locator('.hierarchy').evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThanOrEqual(await page.locator('.breadcrumbs').evaluate(node => node.getBoundingClientRect().top));
    expect(await page.locator('.hierarchy__search').evaluate(node => node.getBoundingClientRect().bottom)).toBeLessThan(await page.locator('.model-flow').evaluate(node => node.getBoundingClientRect().top));
    const blockLabels = await page.locator('.graph-list__node[data-entity-id^="block."]').evaluateAll(nodes => nodes.map(node => ({ id: node.getAttribute('data-entity-id'), label: node.textContent })));
    expect(blockLabels).toHaveLength(24);
    for (const node of blockLabels) expect(node.label).toBe(node.id);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBe(viewport.height);
    await info.attach('architecture-positions.json', { body: JSON.stringify(architecture, null, 2), contentType: 'application/json' });
    await page.screenshot({ path: info.outputPath(`overview-${viewport.width}.png`) });

    // When: keyboard activation exposes all actual choices without selecting a block.
    await representative.focus();
    await arm(page, '.representative-block__toggle', 'aria-expanded', 'true');
    await page.keyboard.press('Enter'); await changed(page);
    expect(await page.locator('dialog').getAttribute('open')).toBeNull();
    expect(await page.locator('.explorer').getAttribute('data-view')).toBe('blocks');
    expect(await page.locator('.block-picker-summary').count()).toBe(1);
    expect(await page.locator('.inspector .summary-section, .graph-node').count()).toBe(0);
    expect(await page.locator('.block-choice:visible').count()).toBe(24);
    expect(await page.locator('.block-choice').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-entity-id'))))
      .toEqual(Array.from({ length: 24 }, (_, index) => `block.${index}`));
    expect(await page.locator('.block-choice[aria-pressed=true], .stage-card, .block-overview').count()).toBe(0);
    expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe('mamba-130m');
    await page.screenshot({ path: info.outputPath(`choices-${viewport.width}.png`) });

    // When: select the last real block.
    await arm(page, '.inspector[data-entity-id="block.23"]');
    await page.locator('.block-choice[data-entity-id="block.23"]').click(); await changed(page); await closeInspector(page);
    // Then: eight source-ordered horizontal stages and the inspector use only block.23.
    expect(await representative.getAttribute('aria-expanded')).toBe('false');
    expect(await page.locator('.summary-section--state .summary-tensor__name').allTextContents()).toEqual(['tensor name : cache_r_l23', 'tensor name : cache_s_l23']);
    expect(await page.locator('.stage-card').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-entity-id'))))
      .toEqual(capture.entities.find(entity => entity.id === 'block.23')?.children);
    expect(await page.locator('.model-card').count()).toBe(4);
    expect(new URL(page.url()).hash).toBe('#scenario=prefill&entity=block.23');
    expect(await representative.evaluate(node => node === document.activeElement)).toBe(true);
    const geometry = await page.locator('.stage-card').evaluateAll(nodes => nodes.map(node => {
      const rect = node.getBoundingClientRect(); return { x: rect.x, y: rect.y, width: rect.width };
    }));
    expect(geometry).toHaveLength(8); expect(new Set(geometry.map(rect => rect.y)).size).toBe(1);
    geometry.forEach((rect, index) => {
      expect(rect.width).toBeGreaterThanOrEqual(160);
      if (index > 0) expect(rect.x).toBeGreaterThan(geometry[index - 1]?.x ?? Infinity);
    });
    await info.attach('stage-positions.json', { body: JSON.stringify(geometry, null, 2), contentType: 'application/json' });
    await page.screenshot({ path: info.outputPath(`block-${viewport.width}.png`) });

    // When: select a stage with the native keyboard reel navigation.
    await page.locator('.stage-card').first().focus(); await page.keyboard.press('End');
    expect(await page.locator('.stage-card').last().evaluate(node => node === document.activeElement)).toBe(true);
    await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
    const stage = page.locator('.stage-card').nth(1), stageId = await stage.getAttribute('data-entity-id');
    await arm(page, `.inspector[data-entity-id="${stageId}"]`);
    await page.keyboard.press('Enter'); await changed(page); await closeInspector(page);
    expect(await page.locator('.stage-card').count()).toBe(8);
    expect(await stage.getAttribute('aria-current')).toBe('page');
    expect(await page.locator('.operator-choice').count()).toBe(5);
    await page.screenshot({ path: info.outputPath(`stage-${viewport.width}.png`) });
    const operator = page.locator('.operator-choice').first(), operatorId = await operator.getAttribute('data-entity-id');
    await arm(page, `.inspector[data-entity-id="${operatorId}"]`);
    await operator.click(); await changed(page); await closeInspector(page);
    expect(await page.locator('.stage-card').count()).toBe(8);
    expect(await stage.getAttribute('aria-current')).toBe('step');
    expect(await page.locator('.operator-choice[aria-current=page]').getAttribute('data-entity-id')).toBe(operatorId);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`operator-${viewport.width}.png`) });

    // When: navigate directly from block detail to each model stage, then back to model.
    for (const entityId of architectureIds.filter(id => id !== 'blocks')) {
      await arm(page, `.inspector[data-entity-id="${entityId}"]`);
      await page.locator(`.model-card[data-entity-id="${entityId}"]`).click(); await changed(page); await closeInspector(page);
      expect(await page.locator('.stage-card, .block-overview, .block-choice[aria-pressed=true]').count()).toBe(0);
      expect(await page.locator('.operator-choice').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-entity-id'))))
        .toEqual(capture.entities.find(entity => entity.id === entityId)?.children);
      expect(await page.locator('.model-card[aria-current=page]').getAttribute('data-entity-id')).toBe(entityId);
      const actualOperator = page.locator('.operator-choice').first(), actualId = await actualOperator.getAttribute('data-entity-id');
      await arm(page, `.inspector[data-entity-id="${actualId}"]`);
      await actualOperator.click(); await changed(page); await closeInspector(page);
      expect(await page.locator('.stage-card, .block-overview').count()).toBe(0);
      expect(await page.locator('.model-card[aria-current=step]').getAttribute('data-entity-id')).toBe(entityId);
      await arm(page, '.inspector[data-entity-id="mamba-130m"]');
      await page.locator('.model-overview').click(); await changed(page); await closeInspector(page);
      expect(await page.locator('.stage-card, .block-overview, .operator-choice, .block-choice[aria-pressed=true]').count()).toBe(0);
      // Restore a real block before the next model-stage navigation to detect leaked prior detail.
      await representative.click();
      await arm(page, '.inspector[data-entity-id="block.23"]');
      await page.locator('.block-choice[data-entity-id="block.23"]').click(); await changed(page); await closeInspector(page);
    }
    await arm(page, '.inspector[data-entity-id="mamba-130m"]');
    await page.locator('.model-overview').click(); await changed(page); await closeInspector(page);
    expect(await page.locator('.stage-card, .block-overview, .operator-choice, .block-choice[aria-pressed=true]').count()).toBe(0);
    // History restores the real block, then root, without sticky block detail.
    await arm(page, '.inspector[data-entity-id="block.23"]');
    await page.evaluate(() => history.back()); await changed(page); await closeInspector(page);
    expect(await page.locator('.stage-card').count()).toBe(8);
    await arm(page, '.inspector[data-entity-id="mamba-130m"]');
    await page.evaluate(() => history.forward()); await changed(page); await closeInspector(page);
    expect(await page.locator('.stage-card, .block-overview').count()).toBe(0);
  });
}
