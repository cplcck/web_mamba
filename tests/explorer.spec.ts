import { test as base, expect, type Page } from '@playwright/test';
import { allEntities, arm, changed, closeInspector, installReadiness, mounted, openAdvanced, route } from './browser-helpers';

const test = base.extend<{ audited: void }>({
  audited: [async ({ page, baseURL }, use, info) => {
    const requests: string[] = [], consoleErrors: { text: string; url: string }[] = [];
    const exceptions: string[] = [], failures: { url: string; error: string | null }[] = [];
    const badResponses: { url: string; status: number }[] = [];
    page.on('request', request => requests.push(request.url()));
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push({ text: message.text(), url: message.location().url }); });
    page.on('pageerror', error => exceptions.push(error.message));
    page.on('requestfailed', request => failures.push({ url: request.url(), error: request.failure()?.errorText ?? null }));
    page.on('response', response => { if (response.status() >= 400) badResponses.push({ url: response.url(), status: response.status() }); });
    const favicon = new URL('/favicon.ico', baseURL).href;
    const cdp = await page.context().newCDPSession(page);
    const browserRequests: { id: string; url: string; initiator: string; type: string | undefined }[] = [];
    const faviconResponses: { id: string; status: number }[] = [];
    cdp.on('Network.requestWillBeSent', event => browserRequests.push({ id: event.requestId, url: event.request.url, initiator: event.initiator.type, type: event.type }));
    cdp.on('Network.responseReceived', event => { if (event.response.url === favicon) faviconResponses.push({ id: event.requestId, status: event.response.status }); });
    await cdp.send('Network.enable');
    await installReadiness(page);
    await use();
    const missing = info.title.endsWith('[missing]');
    const manifest = `${baseURL}data/manifest.json`;
    const verifiedFavicons = browserRequests.filter(request => request.url === favicon && request.initiator === 'other' && request.type === 'Other' && faviconResponses.some(response => response.id === request.id && response.status === 404));
    await info.attach('network-console.json', { body: JSON.stringify({ requests, consoleErrors, exceptions, failures, badResponses, browserRequests, faviconResponses, verifiedFavicons }, null, 2), contentType: 'application/json' });
    expect(browserRequests.filter(request => !request.url.startsWith(baseURL!) && !verifiedFavicons.includes(request))).toEqual([]);
    await cdp.detach();
    expect(exceptions).toEqual([]);
    expect(requests.length).toBeGreaterThan(0);
    expect(requests.filter(url => !url.startsWith(baseURL!))).toEqual([]);
    expect(requests.filter(url => /\.(gguf|safetensors|bin|npz|pt|pth)(?:$|[?#])/i.test(url))).toEqual([]);
    expect(badResponses).toEqual(missing ? [{ url: manifest, status: 404 }] : []);
    // Only the deliberate manifest 404 and the CDP-proven browser favicon 404 are expected.
    // Application request assertions above remain restricted to the project base.
    expect(consoleErrors.filter(error => !((missing && error.url === manifest && /404/.test(error.text)) || (verifiedFavicons.length > 0 && error.url === favicon && /404/.test(error.text))))).toEqual([]);
    expect(failures.filter(failure => !(missing && failure.url === manifest && failure.error === 'net::ERR_ABORTED'))).toEqual([]);
  }, { auto: true }],
});

const selection = (page: Page) => page.locator('.inspector');

// The real public documents determine traversal; there is no synthetic positive fixture.
test('all actual entities, tensor fields, aliases, tied payloads, state and canonical estimates', async ({ page }, info) => {
  test.setTimeout(600_000);
  await page.goto('./#scenario=prefill&entity=mamba-130m');
  await mounted(page);
  const result = await allEntities(page);
  expect(result.coverage).toHaveLength(2760);
  expect(new Set(result.coverage.map(row => `${row.scenario}/${row.entityId}`)).size).toBe(2760);
  expect(result.coverage.reduce((n, row) => n + row.tensorRows, 0)).toBe(10112);
  expect(result.coverage.reduce((n, row) => n + row.fieldChecks, 0)).toBe(262912);
  expect(result.coverage.reduce((n, row) => n + row.stateArrays, 0)).toBe(11040);
  expect(result.dtypes).toEqual(expect.arrayContaining(['F32', 'I32']));
  expect(result.aliases).toBeGreaterThan(0);
  expect(result.tiedReferences).toBe(4);
  for (const scenario of ['prefill', 'decode']) {
    const rows = result.coverage.filter(row => row.scenario === scenario);
    expect(rows.filter(row => row.kind === 'block')).toHaveLength(24);
    expect(rows.filter(row => row.emptyWeights)).toHaveLength(941);
    expect(rows.find(row => row.entityId === 'block.23')?.layer).toBe(23);
  }
  await info.attach('all-entity-coverage.json', { body: JSON.stringify(result), contentType: 'application/json' });
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  test(`production interaction, accessibility and metrics ${viewport.width}`, async ({ page, context }, info) => {
    await page.setViewportSize(viewport);
    const cdp = await context.newCDPSession(page);
    await cdp.send('Performance.enable');
    await page.goto('./#scenario=prefill&entity=block.23');
    await mounted(page);
    expect(await selection(page).getAttribute('data-entity-id')).toBe('block.23');
    await page.reload();
    await mounted(page);
    expect(await selection(page).getAttribute('data-entity-id')).toBe('block.23');

    // Dismiss an auto-opened deep-link summary, then use the native keyboard path.
    await closeInspector(page);
    await page.locator('.scenario-switch__button[data-scenario=prefill]').focus();
    await page.keyboard.press('Shift+Tab');
    expect(await page.locator('.skip-link').evaluate(node => node === document.activeElement)).toBe(true);
    const focus = await page.locator('.skip-link').evaluate(node => ({ outline: getComputedStyle(node).outlineStyle, width: getComputedStyle(node).outlineWidth, visible: node.matches(':focus-visible') }));
    expect(focus.visible).toBe(true); expect(focus.outline).not.toBe('none'); expect(parseFloat(focus.width)).toBeGreaterThanOrEqual(2);
    await page.keyboard.press('Tab');
    expect(await page.locator('.scenario-switch__button[data-scenario=prefill]').evaluate(node => node === document.activeElement)).toBe(true);
    await page.keyboard.press('Tab');
    expect(await page.locator('.scenario-switch__button[data-scenario=decode]').evaluate(node => node === document.activeElement)).toBe(true);
    await arm(page, '.inspector[data-scenario=decode][data-entity-id="block.23"]');
    await page.keyboard.press('Enter'); await changed(page);
    expect(await page.locator('.scenario-switch__button[data-scenario=decode]').getAttribute('aria-pressed')).toBe('true');
    await arm(page, '.inspector[data-scenario=prefill][data-entity-id="block.23"]');
    await page.evaluate(() => history.back()); await changed(page); await closeInspector(page);
    await arm(page, '.inspector[data-scenario=decode][data-entity-id="block.23"]');
    await page.evaluate(() => history.forward()); await changed(page); await closeInspector(page);

    const search = page.locator('.hierarchy__search');
    await arm(page, '.hierarchy__empty'); await search.fill('__no_actual_tensor_matches__'); await changed(page);
    expect(await page.locator('.hierarchy__search-status').getAttribute('data-result-count')).toBe('0');
    expect(await selection(page).getAttribute('data-entity-id')).toBe('block.23');
    await arm(page, '.hierarchy__list', 'hidden', ''); await search.press('Escape'); await changed(page);
    expect(await search.inputValue()).toBe('');
    await arm(page, '.hierarchy__search-status', 'data-result-count', '24'); await search.fill('  SSM_SCAN  '); await changed(page);
    expect(Number(await page.locator('.hierarchy__search-status').getAttribute('data-result-count'))).toBe(24);
    const found = page.locator('.hierarchy__row').first();
    const operatorId = (await found.getAttribute('data-entity-id'))!;
    await found.focus();
    await arm(page, `.inspector[data-entity-id="${operatorId}"]`); await page.keyboard.press('Enter'); await changed(page); await closeInspector(page);
    expect(await page.evaluate(() => document.activeElement?.getAttribute('data-entity-id'))).toBe(operatorId);
    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => document.activeElement?.tagName)).not.toBe('BODY');
    await arm(page, '.hierarchy__search-status', 'data-result-count', '0'); await search.fill(''); await changed(page);
    await route(page, 'prefill', 'block.23');

    // Full-field regression intentionally opens the advanced disclosure.
    await openAdvanced(page);
    // Calculator rejection and recovery use real selected tensors.
    await page.locator('[name=P]').fill('2');
    await arm(page, '.dimension-result', 'data-status', 'unsupported');
    await page.locator('.dimension-form button').click(); await changed(page);
    expect(await page.locator('.dimension-result table').count()).toBe(0);
    await page.locator('[name=P]').fill('1');
    await arm(page, '.dimension-result', 'data-status', 'derived');
    await page.locator('.dimension-form button').click(); await changed(page);
    await arm(page, '[data-phase=before] [data-family=R] tr[data-field=layer] td');
    await page.locator('[name=state-layer]').selectOption('7'); await changed(page);
    expect(await page.locator('[data-phase=before] [data-family=R] tr[data-field=layer] td').textContent()).toBe('7');
    await route(page, 'prefill', 'block.0');
    await openAdvanced(page);
    expect(await page.locator('[name=state-layer]').inputValue()).toBe('0');

    const accessibility = await page.evaluate(() => {
      const must = (ok: unknown, label: string): void => { if (!ok) throw new Error(label); };
      const visible = (node: Element): boolean => node.getClientRects().length > 0;
      for (const input of document.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')) must(input.labels?.length || input.getAttribute('aria-label'), 'unlabelled input');
      for (const button of document.querySelectorAll('button')) must(button.getAttribute('aria-label') || button.textContent?.trim(), 'unnamed action');
      for (const section of document.querySelectorAll('.tensor-section')) must(document.getElementById(section.getAttribute('aria-labelledby')!), 'unnamed tensor section');
      for (const table of document.querySelectorAll('.tensor-row table')) must(table.querySelector('caption'), 'unnamed tensor table');
      for (const header of document.querySelectorAll('.tensor-table th')) must(header.getAttribute('scope') === 'row', 'header association');
      const svg = document.querySelector('.graph-svg')!;
      must(svg.getAttribute('aria-label') || svg.getAttribute('aria-labelledby'), 'unnamed SVG');
      must(document.querySelector('.graph-list'), 'missing semantic graph');
      must(matchMedia('(prefers-reduced-motion: reduce)').matches, 'reduced motion');
      for (const node of document.querySelectorAll('*')) {
        if (!visible(node)) continue;
        const style = getComputedStyle(node);
        must(style.animationName === 'none', 'unexpected animation');
        must(style.transitionDuration.split(',').every(value => parseFloat(value) <= 0.001), 'non-reduced transition');
      }
      const luminance = (color: string): number => {
        const rgb = color.match(/[\d.]+/g)!.slice(0, 3).map(Number).map(n => { const s = n / 255; return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; });
        return rgb[0]! * 0.2126 + rgb[1]! * 0.7152 + rgb[2]! * 0.0722;
      };
      let minContrast = Infinity, checked = 0;
      // Required rendered HTML text, including actual per-row machine fields.
      for (const node of document.querySelectorAll<HTMLElement>('button, label, th, td, caption, h1, h2, h3, p, summary')) {
        if (!visible(node) || !node.textContent?.trim()) continue;
        const style = getComputedStyle(node);
        let ancestor: Element | null = node, background = '';
        while (ancestor) {
          const color = getComputedStyle(ancestor).backgroundColor;
          if (color !== 'rgba(0, 0, 0, 0)' && color !== 'transparent') { background = color; break; }
          ancestor = ancestor.parentElement;
        }
        must(background, 'missing opaque background');
        const a = luminance(style.color), b = luminance(background), contrast = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
        const large = parseFloat(style.fontSize) >= 24 || (parseFloat(style.fontSize) >= 18.66 && Number(style.fontWeight) >= 700);
        must(contrast >= (large ? 3 : 4.5), `contrast ${node.tagName}: ${contrast}`);
        minContrast = Math.min(minContrast, contrast); checked++;
      }
      must(document.documentElement.scrollWidth <= innerWidth, 'viewport overflow');
      return { checkedTextElements: checked, minContrast, reducedMotion: true, viewport: { width: innerWidth, height: innerHeight } };
    });
    await info.attach('accessibility.json', { body: JSON.stringify(accessibility), contentType: 'application/json' });
    // Representative current production screenshots, not new pixel-equivalence claims.
    await page.locator('.inspector__header').scrollIntoViewIfNeeded();
    await page.evaluate(() => document.fonts.ready.then(() => undefined));
    await page.screenshot({ path: info.outputPath(`production-${viewport.width}.png`) });
    const cdpMetrics = await cdp.send('Performance.getMetrics');
    const navigation = await page.evaluate(() => performance.getEntriesByType('navigation').map(entry => entry.toJSON()));
    await info.attach('chromium-performance.json', { body: JSON.stringify({ metrics: cdpMetrics.metrics, navigation, score: null, scope: 'Measured Chromium metrics after production interactions; no synthetic performance score.' }), contentType: 'application/json' });
    await cdp.detach();
  });
}

for (const failure of ['malformed', 'missing'] as const) {
  test(`fail closed, disabled loading/error and native retry [${failure}]`, async ({ page }) => {
    let release!: () => void, entered!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const requested = new Promise<void>(resolve => { entered = resolve; });
    await page.route('**/data/manifest.json', async intercepted => {
      entered(); await gate;
      await intercepted.fulfill({ status: failure === 'missing' ? 404 : 200, contentType: 'application/json', body: failure === 'missing' ? '' : '{' });
    }, { times: 1 });
    // Drain all six exact request lifecycles before retry, including the injected failure.
    const drained = new Promise<void>((resolve, reject) => {
      const completed = new Set<string>();
      const done = (request: import('@playwright/test').Request): void => {
        if (/\/data\/(manifest|model|provenance|validation|prefill|decode)\.json$/.test(request.url())) completed.add(request.url());
        if (completed.size === 6) { clearTimeout(timer); page.off('requestfinished', done); page.off('requestfailed', done); resolve(); }
      };
      const timer = setTimeout(() => { page.off('requestfinished', done); page.off('requestfailed', done); reject(new Error('six data requests did not complete')); }, 15_000);
      page.on('requestfinished', done); page.on('requestfailed', done);
    });
    await page.goto('./#scenario=decode&entity=block.23', { waitUntil: 'domcontentloaded' });
    await requested;
    expect(await page.locator('.loading-skeleton').count()).toBeGreaterThan(0);
    expect(await page.locator('.scenario-switch__button:disabled').count()).toBe(2);
    expect(await page.locator('.tensor-row[data-tensor-id]').count()).toBe(0);
    release();
    expect(await page.evaluate(() => window.browserReady)).toBe('error');
    await drained;
    expect(await page.locator('[role=alert]').count()).toBe(1);
    expect(await page.locator('.scenario-switch__button:disabled').count()).toBe(2);
    expect(await page.locator('.graph-node, .tensor-row[data-tensor-id]').count()).toBe(0);
    await page.locator('[role=alert] button').focus();
    await arm(page, '.scenario-switch__button:not(:disabled)');
    await page.keyboard.press('Enter'); await changed(page);
    expect(await selection(page).getAttribute('data-entity-id')).toBe('block.23');
    expect(await selection(page).getAttribute('data-scenario')).toBe('decode');
    expect(await page.locator('[role=alert]').count()).toBe(0);
  });
}
