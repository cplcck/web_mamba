import { readFileSync } from 'node:fs';
import { test, expect } from '@playwright/test';
import { validateDocument } from '../src/schema';
import { arm, changed, installReadiness, mounted, route } from './browser-helpers';

const captures = {
  prefill: validateDocument(JSON.parse(readFileSync('public/data/prefill.json', 'utf8'))),
  decode: validateDocument(JSON.parse(readFileSync('public/data/decode.json', 'utf8'))),
};
const kinds = ['ADD', 'MUL', 'MUL_MAT', 'RMS_NORM', 'GET_ROWS', 'RESHAPE', 'VIEW', 'TRANSPOSE',
  'CONT', 'CPY', 'CONCAT', 'SCALE', 'UNARY', 'GLU', 'SSM_CONV', 'SSM_SCAN'];

test.beforeEach(async ({ page }) => { await installReadiness(page); });

test('operator choices and search results retain names alongside short descriptions', async ({ page }) => {
  // Given: the two real Final Norm operators.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('./#scenario=prefill&entity=model%2Ffinal-normalization'); await mounted(page);
  // When / Then: names remain machine identities; descriptions are separate readable content.
  expect(await page.locator('.operator-choice .operator-name').allTextContents()).toEqual(['rms_norm', 'mul']);
  expect(await page.locator('.operator-choice .operator-description').count()).toBe(2);
  for (const text of await page.locator('.operator-choice .operator-description').allTextContents()) expect(text.trim().length).toBeGreaterThan(0);
  const resultReady = page.locator('.hierarchy__row[data-entity-id="model/final-normalization/rms_norm.1449"]');
  await arm(page, '.hierarchy__search-status', 'data-result-count', '1');
  await page.locator('.hierarchy__search').fill('model/final-normalization/rms_norm.1449'); await changed(page);
  expect(await resultReady.locator('.operator-name').textContent()).toBe('rms_norm');
  expect((await resultReady.locator('.operator-description').textContent())?.trim().length).toBeGreaterThan(0);
});

for (const width of [1440, 390]) {
  test(`all operator kinds show real before and after metadata at ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('./'); await mounted(page);
    for (const scenario of ['prefill', 'decode'] as const) {
      const capture = captures[scenario], tensors = new Map(capture.tensors.map(tensor => [tensor.id, tensor]));
      for (const op of kinds) {
        // Given: prefer nonempty examples, but decode SCALE only has empty captured results.
        const candidates = capture.entities.filter(entity => entity.kind === 'operator' &&
          entity.outputTensorIds.some(id => tensors.get(id)?.op === op));
        const entity = candidates.find(entity => entity.outputTensorIds.some(id => (tensors.get(id)?.numel ?? 0) > 0)) ?? candidates[0];
        if (!entity) throw new Error(`Missing captured ${scenario}/${op}`);
        // When
        await route(page, scenario, entity.id);
        // Then: both sides are data-bound and every actual operand remains visible.
        const panel = page.locator('.operator-explanation');
        expect(await panel.getAttribute('data-op')).toBe(op);
        for (const [section, ids] of [
          ['inputs', [...new Set([...entity.inputTensorIds, ...entity.weightTensorIds])]],
          ['outputs', entity.outputTensorIds],
        ] as const) {
          const rows = panel.locator(`.operator-transition__${section} .operator-operand`);
          expect(await rows.evaluateAll(nodes => nodes.map(node => node.getAttribute('data-tensor-id')))).toEqual(ids);
          for (const [index, id] of ids.entries()) {
            const tensor = tensors.get(id);
            if (!tensor) throw new Error(`Missing captured tensor ${id}`);
            const row = rows.nth(index);
            expect(await row.locator('[data-field=name]').textContent()).toBe(tensor.name);
            expect(await row.locator('[data-field=shape]').textContent()).toBe(`[${tensor.nativeShape.join(', ')}]`);
            expect(await row.locator('[data-field=numel]').textContent()).toBe(tensor.numel.toLocaleString('en-US'));
            expect(await row.locator('[data-field=dtype]').textContent()).toBe(tensor.dtype);
          }
        }
        expect((await panel.locator('.operator-explanation__operation').textContent())?.trim().length).toBeGreaterThan(0);
        const clipping = await panel.evaluate(panel => ({
          pageOverflow: document.documentElement.scrollWidth > innerWidth || document.documentElement.scrollHeight > innerHeight,
          clipped: [...panel.querySelectorAll<HTMLElement>('p, h3, h4, dt, dd, li')].filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent),
        }));
        expect(clipping).toEqual({ pageOverflow: false, clipped: [] });
        if (scenario === 'prefill') {
          await panel.scrollIntoViewIfNeeded();
          await page.screenshot({ path: info.outputPath(`${op.toLowerCase()}-${width}.png`) });
          await panel.locator('.operator-transition__outputs').scrollIntoViewIfNeeded();
          await page.screenshot({ path: info.outputPath(`${op.toLowerCase()}-${width}-after.png`) });
        }
      }
    }
  });
}
