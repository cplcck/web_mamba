import { test, expect } from '@playwright/test'
import { arm, changed, closeInspector, installReadiness, mounted, openAdvanced } from './browser-helpers'

test.beforeEach(async ({ page }) => { await installReadiness(page) })

test('a new compact selection starts at its identity after closing a scrolled summary', async ({ page }) => {
  // Given: a model summary scrolled to its state section before closing.
  await page.goto('./'); await mounted(page)
  await arm(page, 'dialog[open]')
  await page.locator('.inspector-toggle').click(); await changed(page)
  await page.locator('.summary-section--state').scrollIntoViewIfNeeded()
  expect(await page.locator('.inspector').evaluate(node => node.scrollTop)).toBeGreaterThan(0)
  await closeInspector(page)
  await arm(page, '.representative-block__toggle', 'aria-expanded', 'true')
  await page.locator('.representative-block__toggle').click(); await changed(page)
  // When: choosing another layer reopens the compact drawer.
  await arm(page, '.inspector[data-entity-id="block.23"]')
  await page.locator('.block-choice[data-entity-id="block.23"]').click(); await changed(page)
  // Then: the new identity and role are not above an inherited scroll position.
  expect(await page.locator('dialog').evaluate(node => node.matches(':modal'))).toBe(true)
  expect(await page.locator('.inspector').evaluate(node => node.scrollTop)).toBe(0)
})

test('desktop default inspector is concise and right of a viewport-bounded explorer', async ({ page }) => {
  // Given / When: actual root data at the desktop target.
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('./'); await mounted(page)
  // Then: the inspector cannot lengthen the document, even for the full model.
  const bounds = await page.evaluate(() => ({
    height: document.documentElement.scrollHeight,
    explorer: document.querySelector('.explorer')?.getBoundingClientRect().toJSON(),
    inspector: document.querySelector('.inspector')?.getBoundingClientRect().toJSON(),
  }))
  expect(bounds.height).toBe(900)
  expect(bounds.inspector?.x).toBeGreaterThan(bounds.explorer?.right ?? Infinity)
  expect(await page.locator('.summary-section').count()).toBe(4)
  expect(await page.locator('.summary-tensor').count()).toBeLessThanOrEqual(8)
  expect(await page.locator('.inspector-advanced').getAttribute('open')).toBeNull()
  expect(await page.locator('.tensor-row:visible').count()).toBe(0)
})

for (const viewport of [{ width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  test(`selection automatically opens a native drawer with close, reopen and Escape ${viewport.width}`, async ({ page }) => {
    // Given: compact root leaves the full model navigation available.
    await page.setViewportSize(viewport)
    await page.goto('./'); await mounted(page)
    expect(await page.locator('.inspector-dialog').getAttribute('open')).toBeNull()
    const selected = page.locator('.model-card[data-entity-id="model/embedding"]')
    // When: a native keyboard selection updates and opens the summary.
    await selected.focus()
    await arm(page, '.inspector-dialog', 'open', '')
    await selected.press('Enter'); await changed(page)
    // Then: native modality, useful width, and initial close focus.
    expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe('model/embedding')
    expect(await page.getByRole('dialog').count()).toBe(1)
    expect(await page.locator('.inspector-dialog').evaluate(node => node.matches(':modal'))).toBe(true)
    expect(await page.locator('.inspector-close').evaluate(node => node === document.activeElement)).toBe(true)
    expect(await page.locator('.inspector').evaluate(node => node.getBoundingClientRect().width)).toBeGreaterThanOrEqual(320)
    await page.keyboard.press('Shift+Tab')
    // Native Chrome may visit browser chrome, but never the inert underlying app.
    expect(await page.locator('dialog').evaluate(node => node.matches(':modal') && (node.contains(document.activeElement) || !document.hasFocus()))).toBe(true)
    await page.keyboard.press('Tab')
    expect(await page.locator('.inspector-close').evaluate(node => node === document.activeElement)).toBe(true)
    await closeInspector(page)
    expect(await selected.evaluate(node => node === document.activeElement)).toBe(true)
    await arm(page, '.inspector-dialog', 'open', '')
    await page.locator('.inspector-toggle').click(); await changed(page)
    await closeInspector(page, true)
    expect(await selected.evaluate(node => node === document.activeElement)).toBe(true)
    // Reselecting the current layer also opens the summary.
    await arm(page, '.inspector-dialog', 'open', '')
    await selected.press('Enter'); await changed(page)
    expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBe(viewport.height)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width)
  })
}


test('full model evidence stays bounded when advanced is opened', async ({ page }) => {
  // Given: the model with hundreds of actual tensor references.
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('./'); await mounted(page)
  // When
  await openAdvanced(page)
  // Then: every full row is retained inside its local evidence scroll region.
  expect(await page.locator('.tensor-row').count()).toBe(438)
  expect(await page.locator('.inspector-advanced__content').evaluate(node => node.clientHeight)).toBeLessThanOrEqual(320)
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBe(900)
})

test('current and ancestor rails differ from hover and retain independent keyboard focus', async ({ page }) => {
  // Given: a selected actual operator within a block stage.
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('./#scenario=prefill&entity=block.23%2Finput-projection-split%2Freshape.1398'); await mounted(page)
  const ancestor = page.locator('.stage-card[aria-current=step]')
  // When
  await page.locator('.stage-card').first().hover()
  // Then: hover cannot impersonate persistent selection.
  expect(await ancestor.evaluate(node => getComputedStyle(node).borderTopWidth)).toBe('4px')
  expect(await page.locator('.stage-card').first().evaluate(node => getComputedStyle(node).borderTopWidth)).toBe('1px')
  expect(await page.locator('.operator-choice[aria-current=page]').evaluate(node => getComputedStyle(node).borderTopWidth)).toBe('4px')
  await page.locator('.stage-card').first().focus(); await page.keyboard.press('ArrowRight')
  expect(await ancestor.evaluate(node => node.matches(':focus-visible') && getComputedStyle(node).outlineWidth === '2px')).toBe(true)
  for (const action of await page.locator('.model-card, .model-overview, .block-overview, .stage-card, .operator-choice').all()) {
    const label = (await action.innerText()).replace(/\s+/g, ' ').trim()
    expect(await action.and(page.getByRole('button', { name: label, exact: true })).count()).toBe(1)
  }
})

test('resizing an open drawer releases modality without duplicating the inspector', async ({ page }) => {
  // Given: the same selected summary in a compact dialog.
  await page.goto('./'); await mounted(page)
  await arm(page, 'dialog[open]')
  await page.locator('.inspector-toggle').click(); await changed(page)
  // When
  await arm(page, '.workspace > .inspector-pane')
  await page.setViewportSize({ width: 1440, height: 900 }); await changed(page)
  // Then
  expect(await page.locator('dialog').evaluate(node => node.matches(':modal'))).toBe(false)
  expect(await page.locator('.inspector').count()).toBe(1)
  expect(await page.locator('.inspector').isVisible()).toBe(true)
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBe(900)
})


test('a selected layer deep link and reload open its compact summary without another click', async ({ page }) => {
  // Given / When: an actual selected layer is addressed rather than the root landing.
  await page.goto('./#scenario=decode&entity=block.23'); await mounted(page)
  // Then: the root-only initial closed exception cannot hide selected layer information.
  expect(await page.locator('dialog').evaluate(node => node.matches(':modal'))).toBe(true)
  expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe('block.23')
  await page.reload(); await mounted(page)
  expect(await page.locator('dialog').evaluate(node => node.matches(':modal'))).toBe(true)
  expect(await page.locator('.inspector').getAttribute('data-scenario')).toBe('decode')
  await closeInspector(page, true)
  expect(await page.locator('.breadcrumbs [aria-current=page]').evaluate(node => node === document.activeElement)).toBe(true)
})


test('history selection keeps focus inside an already-open drawer after replacing advanced content', async ({ page }) => {
  // Given: focus in a selected layer's advanced evidence.
  await page.goto('./'); await mounted(page)
  await arm(page, 'dialog[open]')
  await page.locator('.model-card[data-entity-id="model/embedding"]').click(); await changed(page)
  await openAdvanced(page)
  // When: browser history selects the root and replaces that focused content.
  await arm(page, '.inspector[data-entity-id="mamba-130m"]')
  await page.evaluate(() => history.back()); await changed(page)
  // Then: focus remains in the live modal, rather than falling into the inert page.
  expect(await page.locator('.inspector-close').evaluate(node => node === document.activeElement)).toBe(true)
  expect(await page.locator('.inspector-advanced').getAttribute('open')).toBeNull()
})
