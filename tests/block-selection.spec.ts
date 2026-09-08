import { test, expect, type Page } from '@playwright/test';
import { arm, changed, closeInspector, installReadiness, mounted } from './browser-helpers';

test.beforeEach(async ({ page }) => { await installReadiness(page); });

async function openChoices(page: Page): Promise<void> {
  await arm(page, '.representative-block__toggle', 'aria-expanded', 'true');
  await page.locator('.representative-block__toggle').click();
  await changed(page);
  expect(await page.locator('.block-choice:visible').count()).toBe(24);
}

async function choicesAreClosed(page: Page): Promise<void> {
  expect(await page.locator('.representative-block__toggle').getAttribute('aria-expanded')).toBe('false');
  expect(await page.locator('.block-choices').getAttribute('hidden')).not.toBeNull();
  expect(await page.locator('.block-choice:visible').count()).toBe(0);
}

for (const width of [1440, 390]) {
  test(`architecture selection closes block choices, including reselection, at ${width}`, async ({ page }) => {
    // Given: the model overview with an expanded block selector.
    await page.setViewportSize({ width, height: 900 });
    await page.goto('./'); await mounted(page);
    for (const id of ['model/embedding', 'model/final-normalization', 'model/final-projection']) {
      await openChoices(page);
      // When: another architecture card is selected.
      await arm(page, `.inspector[data-entity-id="${id}"]`);
      await page.locator(`.model-card[data-entity-id="${id}"]`).click(); await changed(page);
      // Then: disclosure state agrees with its visible choices.
      await choicesAreClosed(page);
      await closeInspector(page);
      // Returning to this architecture after Mamba replaces the chooser view.
      await openChoices(page);
      await arm(page, '.representative-block__toggle', 'aria-expanded', 'false');
      await page.locator(`.model-card[data-entity-id="${id}"]`).press('Enter'); await changed(page);
      await choicesAreClosed(page);
      expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe(id);
      await closeInspector(page);
    }
    // The representative itself still toggles open/closed normally.
    await openChoices(page);
    await arm(page, '.representative-block__toggle', 'aria-expanded', 'false');
    await page.locator('.representative-block__toggle').click(); await changed(page);
    await choicesAreClosed(page);
  });

  test(`selected block and model stages show total operator counts at ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    for (const scenario of ['prefill', 'decode']) {
      // Given: counts must describe the current capture, not a hard-coded stage count.
      await page.goto(`./#scenario=${scenario}&entity=mamba-130m`); await mounted(page); await closeInspector(page);
      for (const [id, count] of [
        ['block.0', 50], ['block.23', 50], ['model/embedding', 1],
        ['model/final-normalization', 2], ['model/final-projection', 1],
      ] as const) {
        const block = id.startsWith('block.');
        if (block) await openChoices(page);
        // When: use the real selection buttons.
        await arm(page, `.inspector[data-entity-id="${id}"]`);
        await page.locator(`${block ? '.block-choice' : '.model-card'}[data-entity-id="${id}"]`).click(); await changed(page);
        // Then: the displayed numeric field is the total, including nested operators.
        const heading = page.locator(block ? '.block-overview' : '.stage-detail__title');
        expect(await heading.locator('.operator-total__value').allTextContents()).toEqual([String(count)]);
        await closeInspector(page);
        await heading.scrollIntoViewIfNeeded();
        if (scenario === 'prefill') await page.screenshot({ path: info.outputPath(`count-${id.replaceAll('/', '-')}-${width}.png`) });
      }
    }
  });

  test(`graph keyboard actions remain usable after an exclusive chooser switch at ${width}`, async ({ page }) => {
    // Given: a previously selected SVG graph must disappear in chooser mode.
    const id = 'model/embedding/get_rows.0';
    await page.setViewportSize({ width, height: 900 });
    await page.goto(`./#scenario=prefill&entity=${encodeURIComponent(id)}`); await mounted(page); await closeInspector(page);
    await arm(page, '.graph-evidence', 'open', '');
    await page.locator('.graph-evidence > summary').click(); await changed(page);
    for (const key of ['Enter', 'Space']) {
      await openChoices(page);
      expect(await page.locator('.graph-node').count()).toBe(0);
      expect(await page.locator('.graph-evidence').getAttribute('hidden')).not.toBeNull();
      expect(await page.locator('.inspector .summary-section').count()).toBe(0);
      // When: return through the real architecture/operator actions, then activate SVG.
      await arm(page, '.inspector', 'data-entity-id', 'model/embedding');
      await page.locator('.model-card[data-entity-id="model/embedding"]').click(); await changed(page); await closeInspector(page);
      await arm(page, '.inspector', 'data-entity-id', id);
      await page.locator(`.operator-choice[data-entity-id="${id}"]`).click(); await changed(page); await closeInspector(page);
      await arm(page, '.graph-evidence', 'open', '');
      await page.locator('.graph-evidence > summary').click(); await changed(page);
      await page.locator(`.graph-node[data-entity-id="${id}"]`).press(key);
      // Then: concrete graph keyboard behavior and chooser dismissal still hold.
      await choicesAreClosed(page);
      expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe(id);
      await closeInspector(page);
    }
  });
}

test('non-selection controls close choices without restoring the previous entity view', async ({ page }) => {
  // Given: controls both inside and outside the explorer.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('./#scenario=prefill&entity=block.0'); await mounted(page);
  for (const selector of ['.hierarchy__clear', '.scenario-switch__button[data-scenario=prefill]']) {
    await openChoices(page);
    // When
    await arm(page, '.representative-block__toggle', 'aria-expanded', 'false');
    await page.locator(selector).click(); await changed(page);
    // Then
    await choicesAreClosed(page);
    expect(await page.locator('.inspector').getAttribute('data-view')).toBe('blocks');
    expect(await page.locator('.inspector .summary-section').count()).toBe(0);
  }
  // Model overview is a separate view even though both modes use the real root ID.
  await arm(page, '.inspector', 'data-view', 'entity');
  await page.locator('.model-overview').click(); await changed(page);
  expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe('mamba-130m');
  expect(await page.locator('.inspector .summary-section').count()).toBeGreaterThan(0);
  expect(await page.locator('.graph-evidence').getAttribute('hidden')).toBeNull();
});
