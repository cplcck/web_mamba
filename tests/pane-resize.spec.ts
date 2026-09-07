import { test, expect, type Page } from '@playwright/test';
import { arm, changed, closeInspector, installReadiness, mounted } from './browser-helpers';

test.beforeEach(async ({ page }) => {
  await installReadiness(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('./'); await mounted(page);
});

async function drag(page: Page, delta: number): Promise<number> {
  const handle = page.locator('.pane-splitter');
  const box = await handle.boundingBox();
  if (!box) throw new Error('Visible divider required');
  const current = Number(await handle.getAttribute('aria-valuenow'));
  const minimum = Number(await handle.getAttribute('aria-valuemin'));
  const maximum = Number(await handle.getAttribute('aria-valuemax'));
  const expected = Math.max(minimum, Math.min(maximum, current - delta));
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await arm(page, '.pane-splitter', 'aria-valuenow', String(expected));
  await page.mouse.move(box.x + box.width / 2 + delta, box.y + box.height / 2, { steps: 3 });
  await changed(page);
  await page.mouse.up();
  return expected;
}

test('mouse divider widens and narrows the inspector without changing selection', async ({ page }) => {
  // Given: the real desktop boundary, not an unrelated CSS resize handle.
  expect(await page.getByRole('separator').count()).toBe(1);
  const before = await page.locator('.inspector-pane').evaluate(node => node.getBoundingClientRect().width);
  const initialBox = await page.locator('.pane-splitter').boundingBox();
  if (!initialBox) throw new Error('Visible divider required');
  await page.mouse.move(initialBox.x + initialBox.width / 2, initialBox.y + initialBox.height / 2);
  await page.mouse.down();
  await arm(page, '.pane-splitter', 'aria-valuenow', String(before));
  await page.mouse.move(initialBox.x + initialBox.width / 2, initialBox.y + initialBox.height / 2 - 100);
  await changed(page); await page.mouse.up();
  expect(await page.locator('.inspector-pane').evaluate(node => node.getBoundingClientRect().width)).toBe(before);
  // When / Then: left grows the right pane, right shrinks it.
  expect(await drag(page, -180)).toBe(before + 180);
  expect(await page.locator('.inspector-pane').evaluate(node => node.getBoundingClientRect().width)).toBe(before + 180);
  const width = await drag(page, 100);
  expect(width).toBe(before + 80);
  expect(await page.locator('html').getAttribute('class') ?? '').not.toContain('pane-resizing');
  await page.mouse.move(50, 100);
  expect(await page.locator('.inspector-pane').evaluate(node => node.getBoundingClientRect().width)).toBe(width);
  await arm(page, '.inspector[data-entity-id="model/embedding"]');
  await page.getByRole('button', { name: 'Embedding', exact: true }).click(); await changed(page);
  await arm(page, '.inspector[data-scenario="decode"]');
  await page.locator('.scenario-switch__button[data-scenario="decode"]').click(); await changed(page);
  expect(await page.locator('.inspector-pane').evaluate(node => node.getBoundingClientRect().width)).toBe(width);
  expect(await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight])).toEqual([1440, 900]);
});

test('keyboard divider exposes bounds and reclamps when the desktop narrows', async ({ page }) => {
  // Given
  const handle = page.getByRole('separator');
  expect(await handle.getAttribute('aria-orientation')).toBe('vertical');
  expect(await handle.getAttribute('aria-controls')).toBe('tensor-inspector');
  await handle.focus();
  const original = Number(await handle.getAttribute('aria-valuenow'));
  // When / Then: keyboard and pointer limits share the same measured range.
  await arm(page, '.pane-splitter', 'aria-valuenow', String(original + 16));
  await handle.press('ArrowLeft'); await changed(page);
  await arm(page, '.pane-splitter', 'aria-valuenow', String(original));
  await handle.press('ArrowRight'); await changed(page);
  await arm(page, '.pane-splitter', 'aria-valuenow', '320');
  await handle.press('Home'); await changed(page);
  expect(await drag(page, 500)).toBe(320);
  const maximum = Number(await handle.getAttribute('aria-valuemax'));
  await arm(page, '.pane-splitter', 'aria-valuenow', String(maximum));
  await handle.press('End'); await changed(page);
  expect(await drag(page, -500)).toBe(maximum);
  await arm(page, '.pane-splitter', 'aria-valuemax', '616');
  await page.setViewportSize({ width: 1100, height: 900 }); await changed(page);
  expect(await handle.getAttribute('aria-valuenow')).toBe('616');
  expect(await page.locator('.explorer').evaluate(node => node.getBoundingClientRect().width)).toBeGreaterThanOrEqual(400);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(1100);
  await arm(page, '.pane-splitter', 'aria-valuemax', String(maximum));
  await page.setViewportSize({ width: 1440, height: 900 }); await changed(page);
  expect(await handle.getAttribute('aria-valuenow')).toBe(String(maximum));
});

test('cancel and compact transition end dragging and preserve the desktop width', async ({ page }) => {
  // Given: a genuine active pointer with its browser-assigned ID.
  const handle = page.locator('.pane-splitter');
  const width = await drag(page, -120);
  await page.evaluate(() => document.querySelector<HTMLElement>('.pane-splitter')?.addEventListener('pointerdown', event => {
    document.documentElement.dataset.testPointerId = String(event.pointerId);
  }, { once: true }));
  const box = await handle.boundingBox();
  if (!box) throw new Error('Visible divider required');
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  expect(await page.locator('html').getAttribute('class')).toContain('pane-resizing');
  // When / Then: cancellation clears capture styling and ignores later moves.
  const pointerId = Number(await page.locator('html').getAttribute('data-test-pointer-id'));
  await handle.dispatchEvent('pointercancel', { pointerId });
  expect(await page.locator('html').getAttribute('class') ?? '').not.toContain('pane-resizing');
  await page.mouse.move(300, 100); await page.mouse.up();
  expect(await handle.getAttribute('aria-valuenow')).toBe(String(width));
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await arm(page, '.pane-splitter', 'hidden', '');
  await page.setViewportSize({ width: 768, height: 1024 }); await changed(page);
  await page.mouse.up();
  expect(await page.getByRole('separator').count()).toBe(0);
  expect(await page.locator('html').getAttribute('class') ?? '').not.toContain('pane-resizing');
  await arm(page, 'dialog[open]');
  await page.getByRole('button', { name: 'Embedding', exact: true }).click(); await changed(page);
  await closeInspector(page, true);
  await arm(page, '.workspace > .inspector-pane');
  await page.setViewportSize({ width: 1440, height: 900 }); await changed(page);
  expect(await page.locator('.inspector-pane').evaluate(node => node.getBoundingClientRect().width)).toBe(width);
  expect(await page.getByRole('separator').count()).toBe(1);
});
