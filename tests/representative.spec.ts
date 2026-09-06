import { test, expect } from '@playwright/test';
import { arm, changed, installReadiness, mounted } from './browser-helpers';

test.beforeEach(async ({ page }) => { await installReadiness(page); });

test('a mobile deep link reveals its selected stage inside the reel', async ({ page }) => {
  // Given / When: a stage beyond the first mobile reel viewport is addressed directly.
  await page.goto('./#scenario=decode&entity=block.23%2Fselective-scan-state'); await mounted(page);
  // Then: context names the actual block and the selected card is visible without document overflow.
  expect(await page.locator('.block-overview').getAttribute('data-entity-id')).toBe('block.23');
  const bounds = await page.locator('.stage-card[aria-current=page]').evaluate(node => {
    const card = node.getBoundingClientRect(), reel = node.closest('.stage-reel')?.getBoundingClientRect();
    return { left: card.left, right: card.right, reelLeft: reel?.left, reelRight: reel?.right };
  });
  expect(bounds.left).toBeGreaterThanOrEqual(bounds.reelLeft ?? Infinity);
  expect(bounds.right).toBeLessThanOrEqual(bounds.reelRight ?? -Infinity);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  test(`representative block expands real choices and retains horizontal stages ${viewport.width}`, async ({ page }, info) => {
    // Given: the production no-hash landing, not a test fixture.
    await page.setViewportSize(viewport);
    await page.goto('./'); await mounted(page);
    // Then: one representative and one real block's stages, with evidence disclosed on demand.
    expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe('block.0');
    const representative = page.locator('.representative-block__toggle');
    expect(await representative.count()).toBe(1);
    expect(await representative.getAttribute('aria-expanded')).toBe('false');
    expect(await page.locator('.block-choice:visible').count()).toBe(0);
    expect(await page.locator('.stage-card').count()).toBe(8);
    expect(await page.locator('.graph-evidence').getAttribute('open')).toBeNull();
    const geometry = await page.locator('.stage-card').evaluateAll(nodes => nodes.map(node => {
      const rect = node.getBoundingClientRect();
      return { id: node.getAttribute('data-entity-id'), x: rect.x, y: rect.y, width: rect.width, height: rect.height };
    }));
    expect(new Set(geometry.map(rect => rect.y)).size).toBe(1);
    geometry.forEach((rect, index) => {
      expect(rect.width).toBeGreaterThanOrEqual(160);
      if (index > 0) expect(rect.x).toBeGreaterThan(geometry[index - 1]?.x ?? Infinity);
    });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(await page.locator('.inspector').evaluate(node => node.getBoundingClientRect().top + scrollY)).toBeLessThan(1250);
    await info.attach('stage-positions.json', { body: JSON.stringify(geometry, null, 2), contentType: 'application/json' });
    await page.screenshot({ path: info.outputPath(`overview-${viewport.width}.png`) });

    // When: keyboard activation exposes the 24 actual blocks.
    await representative.focus();
    await arm(page, '.representative-block__toggle', 'aria-expanded', 'true');
    await page.keyboard.press('Enter'); await changed(page);
    // Then: choices use distinct original identities without cloned stage trees.
    expect(await page.locator('.block-choice:visible').count()).toBe(24);
    expect(await page.locator('.block-choice').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-entity-id'))))
      .toEqual(Array.from({ length: 24 }, (_, index) => `block.${index}`));
    expect(await page.locator('.stage-card').count()).toBe(8);
    await page.screenshot({ path: info.outputPath(`choices-${viewport.width}.png`) });

    // When: select the last real block.
    await arm(page, '.inspector[data-entity-id="block.23"]');
    await page.locator('.block-choice[data-entity-id="block.23"]').click(); await changed(page);
    // Then: the stages, inspector state and canonical route all refer to block.23.
    expect(await representative.getAttribute('aria-expanded')).toBe('false');
    expect(await page.locator('[name=state-layer]').inputValue()).toBe('23');
    expect(await page.locator('.stage-card').evaluateAll(nodes => nodes.every(node => node.getAttribute('data-entity-id')?.startsWith('block.23/')))).toBe(true);
    expect(new URL(page.url()).hash).toBe('#scenario=prefill&entity=block.23');
    expect(await representative.evaluate(node => node === document.activeElement)).toBe(true);

    // When: select a stage with the native keyboard reel navigation.
    await page.locator('.stage-card').first().focus();
    await page.keyboard.press('End');
    expect(await page.locator('.stage-card').last().evaluate(node => node === document.activeElement)).toBe(true);
    await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
    const stage = page.locator('.stage-card').nth(1), stageId = await stage.getAttribute('data-entity-id');
    await arm(page, `.inspector[data-entity-id="${stageId}"]`);
    await page.keyboard.press('Enter'); await changed(page);
    // Then: the full block remains, operators expand below it.
    expect(await page.locator('.stage-card').count()).toBe(8);
    expect(await stage.getAttribute('aria-current')).toBe('page');
    expect(await page.locator('.operator-choice').count()).toBe(5);
    await page.screenshot({ path: info.outputPath(`stage-${viewport.width}.png`) });
    const operator = page.locator('.operator-choice').first(), operatorId = await operator.getAttribute('data-entity-id');
    await arm(page, `.inspector[data-entity-id="${operatorId}"]`);
    await operator.click(); await changed(page);
    expect(await page.locator('.stage-card').count()).toBe(8);
    expect(await stage.getAttribute('aria-current')).toBe('step');
    expect(await page.locator('.operator-choice[aria-current=page]').getAttribute('data-entity-id')).toBe(operatorId);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`operator-${viewport.width}.png`) });
  });
}
