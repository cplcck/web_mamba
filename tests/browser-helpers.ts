import { expect, type Page } from '@playwright/test';
import type { CaptureDocument } from '../src/schema';

declare global {
  interface Window {
    browserReady: Promise<'ready' | 'error'>;
    browserChange: Promise<void>;
    browserDocuments?: CaptureDocument[];
  }
}

// Install before navigation: no readiness polling and no dependency on load timing.
export async function installReadiness(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.browserReady = new Promise(resolve => {
      const observer = new MutationObserver(() => {
        const ready = document.querySelector('.inspector[data-entity-id]') &&
          document.querySelector('.scenario-switch__button:not(:disabled)');
        const error = document.querySelector('.data-error[role=alert]');
        if (ready || error) { observer.disconnect(); resolve(error ? 'error' : 'ready'); }
      });
      observer.observe(document, { childList: true, subtree: true, attributes: true });
    });
  });
}

export async function mounted(page: Page): Promise<void> {
  expect(await page.evaluate(() => window.browserReady)).toBe('ready');
}

// Arm the exact resulting DOM state before a native action. Test timeout bounds navigation;
// this timer is failure-only, never a delay used to make an assertion pass.
export async function arm(page: Page, selector: string, attribute?: string, value?: string): Promise<void> {
  await page.evaluate(({ selector, attribute, value }) => {
    window.browserChange = new Promise<void>((resolve, reject) => {
      const observer = new MutationObserver(() => {
        const node = document.querySelector(selector);
        if (node && (!attribute || node.getAttribute(attribute) === value)) {
          observer.disconnect(); clearTimeout(timer); resolve();
        }
      });
      const timer = setTimeout(() => { observer.disconnect(); reject(new Error(`DOM transition timed out: ${selector}`)); }, 15_000);
      observer.observe(document, { childList: true, subtree: true, attributes: true });
    });
  }, { selector, attribute, value });
}
export async function changed(page: Page): Promise<void> {
  await page.evaluate(() => window.browserChange);
}
export async function route(page: Page, scenario: string, entity: string): Promise<void> {
  await arm(page, `.inspector[data-entity-id="${entity}"][data-scenario="${scenario}"]`);
  await page.evaluate(({ scenario, entity }) => {
    location.hash = `#scenario=${scenario}&entity=${encodeURIComponent(entity)}`;
  }, { scenario, entity });
  await changed(page);
}

export async function openInspector(page: Page): Promise<void> {
  if (await page.locator('.inspector-toggle').isVisible() && await page.locator('.inspector-dialog').getAttribute('open') === null) {
    await arm(page, '.inspector-dialog', 'open', '');
    await page.locator('.inspector-toggle').click(); await changed(page);
  }
}

export async function closeInspector(page: Page, escape = false): Promise<void> {
  if (await page.locator('.inspector-dialog').getAttribute('open') === null) return;
  await page.evaluate(() => {
    window.browserChange = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('drawer close event timed out')), 15_000);
      document.querySelector('dialog')?.addEventListener('close', () => { clearTimeout(timer); resolve(); }, { once: true });
    });
  });
  if (escape) await page.keyboard.press('Escape'); else await page.locator('.inspector-close').click();
  await changed(page);
}

export async function openAdvanced(page: Page): Promise<void> {
  await openInspector(page);
  if (await page.locator('.inspector-advanced').getAttribute('open') !== null) return;
  await arm(page, '.inspector-advanced__content > .tensor-section');
  await page.locator('.inspector-advanced > summary').click(); await changed(page);
}

// Independent DOM oracle adapted from the approved Integration browser method.
// Thirty actual selections per browser call let Chromium drain navigation work;
// there is no RPC per field, polling delay, or repeated document parsing.
export async function allEntities(page: Page) {
  await openInspector(page);
  const batches = [];
  let total = Infinity;
  for (let start = 0; start < total; start += 30) {
    const batch = await page.evaluate(async start => {
    const documents = window.browserDocuments ??= await Promise.all(['prefill', 'decode'].map(async scenario => {
      const response = await fetch(new URL(`data/${scenario}.json`, document.baseURI));
      if (!response.ok) throw new Error(`oracle HTTP ${response.status}`);
      return await response.json() as CaptureDocument;
    }));
    const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) :
      value !== null && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, canonical(v)])) : value;
    const eq = (actual: unknown, expected: unknown, label: string): void => {
      if (JSON.stringify(canonical(actual)) !== JSON.stringify(canonical(expected))) throw new Error(`${label}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
    };
    const must = (value: unknown, label: string): void => { if (!value) throw new Error(label); };
    const fields = (root: Element): Record<string, HTMLTableCellElement> => Object.fromEntries(
      [...root.querySelectorAll<HTMLTableRowElement>('tr[data-field]')].map(row => {
        eq(row.querySelector('th')?.scope, 'row', 'associated header');
        return [row.dataset.field!, row.querySelector('td')!];
      }));
    const metadata = (root: Element, record: object): void => {
      const cells = fields(root);
      const walk = (record: object, prefix = ''): void => {
        for (const [key, value] of Object.entries(record)) {
          const path = prefix ? `${prefix}.${key}` : key;
          if (value !== null && typeof value === 'object' && !Array.isArray(value)) { walk(value, path); continue; }
          must(cells[path], `missing metadata ${path}`);
          const text = cells[path]!.textContent!;
          eq(value === null ? text.split(' ')[0] : typeof value === 'string' ? text : JSON.parse(text), value === null ? 'null' : value, path);
        }
      };
      walk(record);
    };
    const bytes = (n: number): string => `${n.toLocaleString('en-US')} B`;
    const integer = (text: string): number => Number(text.split(' B')[0]!.replaceAll(',', ''));
    const tuple = (values: readonly number[]): string => `[${values.join(', ')}]`;
    const coverage = [];
    const dtypes = new Set<string>();
    let aliases = 0, tiedReferences = 0;
    let index = 0;
    for (const doc of documents) {
      const scenario = doc.scenario.name;
      const tensors = new Map(doc.tensors.map(t => [t.id, t]));
      const entities = new Map(doc.entities.map(e => [e.id, e]));
      const tied = doc.tensors.find(t => t.name === 'token_embd.weight')!;
      must(tied.ggufPayload, 'tied embedding payload');
      eq(tied.consumerIds.map(id => entities.get(id)!.parentId).sort(), ['model/embedding', 'model/final-projection'], 'tied input/output consumers');
      tiedReferences += tied.consumerIds.length;
      for (const entity of doc.entities) {
        if (index++ < start || index > start + 30) continue;
        const inspector = document.querySelector<HTMLElement>('.inspector')!;
        if (inspector.dataset.entityId !== entity.id || inspector.dataset.scenario !== scenario) {
          await new Promise<void>((resolve, reject) => {
            const observer = new MutationObserver(() => {
              if (inspector.dataset.entityId === entity.id && inspector.dataset.scenario === scenario) {
                observer.disconnect(); clearTimeout(timer); resolve();
              }
            });
            const timer = setTimeout(() => { observer.disconnect(); reject(new Error(`selection ${scenario}/${entity.id}`)); }, 15_000);
            observer.observe(document.querySelector('#app')!, { childList: true, subtree: true, attributes: true });
            location.hash = `#scenario=${scenario}&entity=${encodeURIComponent(entity.id)}`;
          });
        }
        const label = `${scenario}/${entity.id}`;
        eq(location.hash, `#scenario=${scenario}&entity=${encodeURIComponent(entity.id)}`, label);
        eq(document.querySelector<HTMLElement>('.explorer')!.dataset.entityId, entity.id, 'explorer');
        eq(document.querySelector<HTMLElement>('.explorer')!.dataset.scenario, scenario, 'scenario');
        eq(document.querySelector<HTMLElement>('.breadcrumbs [aria-current=page]')?.dataset.entityId, entity.id, 'selected hierarchy');
        const ancestors: string[] = [];
        for (let ancestor: typeof entity | undefined = entity; ancestor; ancestor = entities.get(ancestor.parentId!)) ancestors.unshift(ancestor.id);
        eq([...document.querySelectorAll<HTMLElement>('.breadcrumbs__link')].map(n => n.dataset.entityId), ancestors, 'breadcrumbs');
        const advanced = inspector.querySelector<HTMLDetailsElement>('.inspector-advanced')!;
        must(!advanced.open, 'advanced starts closed on every selection');
        eq(inspector.querySelectorAll('.tensor-row').length, 0, 'no full-field dump in default view');
        for (const kind of ['inputs', 'outputs', 'weights'] as const) {
          const key = kind === 'inputs' ? 'inputTensorIds' : kind === 'outputs' ? 'outputTensorIds' : 'weightTensorIds';
          const referenced = entity[key].map(id => tensors.get(id)!);
          const activations = referenced.filter(t => t.role === 'activation');
          const tokens = referenced.filter(t => t.name === 'inp_tokens');
          const candidates = activations.length ? activations : tokens.length ? tokens : referenced.filter(t => t.role !== 'state');
          const expected = (entity.kind === 'model' && kind === 'weights' ? candidates.filter(t => !t.name.startsWith('blk.')) : candidates).slice(0, 2);
          const section = inspector.querySelector<HTMLElement>(`.summary-section[data-section=${kind}]`)!;
          eq(Number(section.dataset.count), referenced.length, 'summary actual total');
          const rows = [...section.querySelectorAll<HTMLElement>('.summary-tensor')];
          eq(rows.map(row => row.dataset.tensorId), expected.map(t => t.id), 'meaningful summary preview');
          for (const [i, row] of rows.entries()) {
            const t = expected[i]!;
            eq(row.querySelector('h4')?.textContent, `tensor name : ${t.name}`, 'summary labeled original name');
            eq(row.dataset.role, t.role, 'summary role');
            eq(row.querySelector('[data-field="native ne[4]"]')?.textContent, `native ne[4] ${tuple(t.nativeShape)}`, 'summary native shape');
            eq(row.querySelector('[data-field=dtype]')?.textContent, t.dtype, 'summary dtype');
            let value = t.logicalBytes, unit = 0;
            while (value >= 1024 && unit < 4) { value /= 1024; unit++; }
            eq(row.querySelector('[data-field=logicalIEC]')?.textContent, unit ? `${Number(value.toFixed(2))} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][unit]}` : bytes(t.logicalBytes), 'summary compact bytes');
          }
        }
        const stateIds = [...new Set([...entity.inputTensorIds, ...entity.outputTensorIds].filter(id => tensors.get(id)!.role === 'state'))];
        eq(Number(inspector.querySelector<HTMLElement>('[data-section=state]')?.dataset.count ?? 0), stateIds.length, 'summary distinct state count');
        must(inspector.querySelectorAll('.summary-tensor').length <= 8, 'bounded preview rows');
        must(document.documentElement.scrollHeight <= innerHeight, 'bounded document');
        // Intentionally use the native disclosure, then inspect displayed full detail.
        await new Promise<void>((resolve, reject) => {
          const observer = new MutationObserver(() => {
            if (advanced.querySelector('.tensor-section')) { observer.disconnect(); clearTimeout(timer); resolve(); }
          });
          const timer = setTimeout(() => { observer.disconnect(); reject(new Error('advanced render timed out')); }, 15_000);
          observer.observe(advanced, { childList: true, subtree: true });
          advanced.querySelector('summary')!.click();
        });
        advanced.scrollIntoView({ block: 'nearest' });
        must(advanced.open && advanced.querySelector('.tensor-section')!.getClientRects().length > 0, 'full evidence is displayed');
        const sections = [...inspector.querySelectorAll('.inspector-advanced__content > .tensor-section')];
        eq(sections.length, 3, 'three sections');
        let tensorRows = 0;
        for (const [index, kind, key] of [[0, 'inputs', 'inputTensorIds'], [1, 'outputs', 'outputTensorIds'], [2, 'weights', 'weightTensorIds']] as const) {
          const section = sections[index]!;
          must(section.classList.contains(`tensor-section--${kind}`), 'section order');
          must(getComputedStyle(section).display !== 'none', 'section displayed');
          const rows = [...section.querySelectorAll<HTMLElement>(':scope > .tensor-row')];
          eq(rows.map(r => r.dataset.tensorId), entity[key], 'ordered tensor references');
          eq(section.querySelectorAll('.tensor-section__empty').length, rows.length ? 0 : 1, 'empty section');
          for (const row of rows) {
            const t = tensors.get(row.dataset.tensorId!)!, f = fields(row);
            const text = (key: string): string => f[key]!.textContent!;
            eq(Object.keys(f).length, 26, 'field inventory');
            eq(row.querySelector('caption')!.textContent, t.name, 'tensor name');
            eq(row.tabIndex, 0, 'focusable tensor');
            const scalar = { id: t.id, role: t.role, dtype: t.dtype, typeBlockSize: String(t.typeBlockSize),
              'native ne[4]': tuple(t.nativeShape), 'native nb[4]': tuple(t.strides),
              'logical shape': t.axisLabels.map((axis, i) => `${axis}=${t.logicalShape[i]}`).join(' · '),
              numel: t.numel.toLocaleString('en-US'), 'logical bytes': bytes(t.logicalBytes),
              producer: t.producerIds.join(', ') || 'none', consumers: t.consumerIds.join(', ') || 'none',
              provenance: t.provenance, evidence: t.classification, captureId: t.captureId ?? doc.captureId,
              observationEpoch: String(t.storage.observationEpoch), allocatorSlotBytes: 'unknown',
              allocatorSlotReason: t.storage.allocatorSlotReason };
            for (const [key, value] of Object.entries(scalar)) eq(text(key), value, `${label}/${t.id}/${key}`);
            let size = t.logicalBytes, unit = 0;
            const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
            while (size >= 1024 && unit < 4) { size /= 1024; unit++; }
            eq(text('IEC/logical'), unit ? `${Number(size.toFixed(2))} ${units[unit]}` : bytes(t.logicalBytes), 'IEC');
            eq(integer(text('ggmlNbytes')), t.storage.ggmlNbytes, 'addressed span');
            const allocation = t.viewSourceId !== null ? 'view' : t.storage.requiredAllocBytes === null ? 'unobserved' : 'observed';
            eq(f.requiredAllocBytes!.dataset.status, allocation, 'allocation status');
            if (allocation === 'observed') eq(integer(text('requiredAllocBytes')), t.storage.requiredAllocBytes, 'allocation');
            if (t.storage.bufferBytes === null) eq(text('buffer capacity').split(' ')[0], 'unknown', 'unknown capacity');
            else eq(integer(text('buffer capacity')), t.storage.bufferBytes, 'buffer capacity');
            if (t.storage.bufferId === null) eq(text('backing range').split(' ')[0], 'unknown', 'unknown backing');
            else eq(text('backing range'), `${t.storage.bufferId} @ ${t.storage.bufferOffsetBytes} B / ${bytes(t.storage.bufferBytes!)}`, 'backing');
            eq(text('view source / offset'), t.viewSourceId === null ? 'none' : `${t.viewSourceId} + ${bytes(t.viewOffsetBytes!)}`, 'alias');
            if (t.ggufPayload) {
              eq(text('GGUF fileId'), t.ggufPayload.fileId, 'payload file');
              eq(integer(text('GGUF offsetBytes')), t.ggufPayload.offsetBytes, 'payload offset');
              eq(integer(text('GGUF payload bytes')), t.ggufPayload.bytes, 'payload bytes');
            } else {
              // Absence is not numeric zero or an invented payload identifier.
              must(!Number.isFinite(integer(text('GGUF payload bytes'))), 'absent GGUF bytes');
              must(!Number.isFinite(integer(text('GGUF offsetBytes'))), 'absent GGUF offset');
              must(text('GGUF fileId').length > 0, 'explicit absent GGUF');
            }
            dtypes.add(t.dtype); if (t.viewSourceId !== null) aliases++;
            tensorRows++;
          }
          if (rows.length) {
            const payloadRanges = entity[key].flatMap(id => {
              const p = tensors.get(id)!.ggufPayload;
              return p ? [p] : [];
            });
            let unique = 0;
            for (const file of new Set(payloadRanges.map(p => p.fileId))) {
              let end = 0;
              for (const p of payloadRanges.filter(p => p.fileId === file).sort((a, b) => a.offsetBytes - b.offsetBytes)) {
                const finish = p.offsetBytes + p.bytes;
                unique += Math.max(0, finish - Math.max(end, p.offsetBytes)); end = Math.max(end, finish);
              }
            }
            const totals = section.querySelector('.tensor-section__totals')!.textContent!;
            const match = totals.match(/unique GGUF ([\d,]+) B/);
            eq(Number(match?.[1]?.replaceAll(',', '')), unique, 'deduplicated GGUF total');
          }
        }
        const ids = new Set([...entity.inputTensorIds, ...entity.outputTensorIds, ...entity.weightTensorIds]);
        const candidates = doc.tensors.filter(t => ids.has(t.id));
        const select = document.querySelector<HTMLSelectElement>('[name=tensor]')!;
        eq([...select.options].map(o => o.value), candidates.map(t => t.id), 'calculator inventory');
        const result = document.querySelector<HTMLElement>('.dimension-result')!;
        if (candidates.length) {
          eq(result.dataset.status, 'derived', 'canonical estimate');
          eq(result.dataset.tensorId, candidates[0]!.id, 'calculator reset');
          const f = fields(result);
          eq(JSON.parse(f.shape!.textContent!), candidates[0]!.logicalShape, 'canonical shape');
          eq(JSON.parse(f.numel!.textContent!), candidates[0]!.numel, 'canonical numel');
          eq(integer(f.logicalBytes!.textContent!), candidates[0]!.logicalBytes, 'canonical bytes');
        } else eq(result.dataset.status, 'empty', 'empty calculator');
        for (const dim of ['P', 'T', 'O']) eq(Number(document.querySelector<HTMLInputElement>(`[name=${dim}]`)!.value), doc.scenario.dimensions[dim], 'dimension');
        let ancestor: typeof entity | undefined = entity;
        while (ancestor && ancestor.kind !== 'block') ancestor = entities.get(ancestor.parentId!);
        const layer = Math.max(0, doc.entities.filter(e => e.kind === 'block').findIndex(e => e.id === ancestor?.id));
        const layerSelect = document.querySelector<HTMLSelectElement>('[name=state-layer]')!;
        eq(Number(layerSelect.value), layer, 'layer reset'); eq(layerSelect.options.length, 24, 'all layers');
        for (const phase of ['before', 'after'] as const) {
          const snapshot = doc.recurrentState![phase], group = document.querySelector(`[data-phase=${phase}]`)!;
          metadata(group.querySelector(':scope > table')!, { id: snapshot.id, captureId: snapshot.captureId, position: snapshot.position, epoch: snapshot.epoch, mapping: snapshot.mapping });
          const arrays = snapshot.arrays.filter(a => a.layer === layer);
          eq([...group.querySelectorAll<HTMLElement>('details')].map(n => n.dataset.family), arrays.map(a => a.family), 'state families');
          for (const array of arrays) metadata(group.querySelector(`[data-family=${array.family}]`)!, array);
        }
        eq([...document.querySelectorAll<HTMLElement>('.graph-node')].map(n => n.dataset.entityId), [...document.querySelectorAll<HTMLElement>('.graph-list__node')].map(n => n.dataset.entityId), 'SVG/list equivalence');
        const edgeIdentity = (selector: string) => [...document.querySelectorAll(selector)].map(node =>
          ['data-edge-key', 'data-edge-role', 'data-source-id', 'data-target-id'].map(attribute => node.getAttribute(attribute)));
        eq(edgeIdentity('.graph-edge'), edgeIdentity('.graph-list__edge'), 'SVG/list tensor edge identity');
        coverage.push({ scenario, entityId: entity.id, kind: entity.kind, tensorRows, fieldChecks: tensorRows * 26, stateArrays: 4, layer, emptyWeights: entity.weightTensorIds.length === 0 });
      }
    }
    return { coverage, dtypes: [...dtypes], aliases, tiedReferences, total: documents.reduce((n, doc) => n + doc.entities.length, 0) };
    }, start);
    batches.push(batch); total = batch.total;
  }
  return { coverage: batches.flatMap(batch => batch.coverage), dtypes: [...new Set(batches.flatMap(batch => batch.dtypes))],
    aliases: batches.reduce((n, batch) => n + batch.aliases, 0), tiedReferences: batches[0]!.tiedReferences,
    bulkAutomationFlag: '--disable-ipc-flooding-protection' };
}
