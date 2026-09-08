import { test, expect } from '@playwright/test';
import { arm, changed, closeInspector, installReadiness, mounted } from './browser-helpers';

test.beforeEach(async ({ page }) => { await installReadiness(page); });

for (const width of [1440, 390]) {
  test(`Mamba selection replaces every previous architecture detail at ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    for (const scenario of ['prefill', 'decode']) {
      await page.goto(`./#scenario=${scenario}&entity=mamba-130m`); await mounted(page); await closeInspector(page);
      for (const id of ['model/embedding', 'model/final-normalization', 'model/final-projection']) {
        // Given: an actual model stage with its operators and tensor details.
        await arm(page, '.inspector', 'data-entity-id', id);
        await page.locator(`.model-card[data-entity-id="${id}"]`).click(); await changed(page);
        await closeInspector(page);
        expect(await page.locator('.stage-detail').count()).toBe(1);
        // When: switch to the representative block selection.
        await page.locator('.representative-block__toggle').click();
        // Then: no old stage, block, graph, or inspector data survives this switch.
        expect(await page.locator('.stage-detail, .block-detail').count()).toBe(0);
        expect(await page.locator('.graph-evidence').getAttribute('hidden')).not.toBeNull();
        expect(await page.locator('.inspector .summary-section, .inspector-advanced').count()).toBe(0);
        expect(await page.locator('.block-picker-summary').count()).toBe(1);
        expect(await page.locator('.block-choice:visible').count()).toBe(24);
        expect(await page.locator('.model-card[aria-current=page]').getAttribute('data-architecture-node')).toBe('blocks');
        expect(await page.locator('.model-overview').getAttribute('aria-current')).toBe('false');
        expect(await page.locator('dialog').getAttribute('open')).toBeNull();
        const chooserHash = new URL(page.url()).hash;
        // Closing/reopening the chooser must never restore the previous stage.
        await arm(page, '.representative-block__toggle', 'aria-expanded', 'false');
        await page.locator('.representative-block__toggle').click(); await changed(page);
        expect(await page.locator('.stage-detail, .block-detail, .inspector .summary-section').count()).toBe(0);
        await arm(page, '.representative-block__toggle', 'aria-expanded', 'true');
        await page.locator('.representative-block__toggle').press('Enter'); await changed(page);
        expect(new URL(page.url()).hash).toBe(chooserHash);
        if (scenario === 'prefill') await page.screenshot({ path: info.outputPath(`chooser-from-${id.replaceAll('/', '-')}-${width}.png`) });
        // A real block replaces the chooser and retains its real operator total.
        await arm(page, '.inspector', 'data-entity-id', 'block.0');
        await page.locator('.block-choice[data-entity-id="block.0"]').click(); await changed(page);
        await closeInspector(page);
        expect(await page.locator('.block-detail').count()).toBe(1);
        expect(await page.locator('.stage-detail, .block-picker-summary').count()).toBe(0);
        expect(await page.locator('.block-choice:visible').count()).toBe(0);
        expect(await page.locator('.block-overview .operator-total__value').textContent()).toBe('50');
        expect(await page.locator('.graph-evidence').getAttribute('hidden')).toBeNull();
      }
    }
  });

  test(`block selection mode survives reload, scenario and history at ${width}`, async ({ page }) => {
    // Given: the example transition from Final Norm to Mamba.
    await page.setViewportSize({ width, height: 900 });
    await page.goto('./#scenario=prefill&entity=model%2Ffinal-normalization'); await mounted(page); await closeInspector(page);
    await page.locator('.representative-block__toggle').click();
    expect(new URLSearchParams(new URL(page.url()).hash.slice(1)).get('view')).toBe('blocks');
    // When / Then: no previous Final Norm content returns through reload or scenario switch.
    await page.reload(); await mounted(page);
    expect(await page.locator('.block-choice:visible').count()).toBe(24);
    expect(await page.locator('.stage-detail, .inspector .summary-section').count()).toBe(0);
    await arm(page, '.inspector', 'data-scenario', 'decode');
    await page.locator('.scenario-switch__button[data-scenario=decode]').click(); await changed(page);
    expect(await page.locator('.explorer').getAttribute('data-view')).toBe('blocks');
    expect(await page.locator('.block-choice:visible').count()).toBe(0);
    expect(await page.locator('.block-picker-summary').count()).toBe(1);
    await arm(page, '.inspector', 'data-entity-id', 'model/embedding');
    await page.locator('.model-card[data-entity-id="model/embedding"]').click(); await changed(page); await closeInspector(page);
    await arm(page, '.explorer', 'data-view', 'blocks');
    await page.evaluate(() => history.back()); await changed(page);
    expect(await page.locator('.block-choice:visible').count()).toBe(24);
    expect(await page.locator('.stage-detail, .inspector .summary-section').count()).toBe(0);
    await arm(page, '.inspector', 'data-entity-id', 'model/embedding');
    await page.evaluate(() => history.forward()); await changed(page); await closeInspector(page);
    expect(await page.locator('.block-picker-summary').count()).toBe(0);
    expect(await page.locator('.stage-detail__title .operator-total__value').textContent()).toBe('1');
  });
}
