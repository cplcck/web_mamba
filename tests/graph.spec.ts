import { readFileSync } from 'node:fs';
import { test, expect, type Page } from '@playwright/test';
import { buildGraphModel } from '../src/graph';
import { validateDocument } from '../src/schema';
import { arm, changed, closeInspector, installReadiness, mounted } from './browser-helpers';

const captures = {
  prefill: validateDocument(JSON.parse(readFileSync('public/data/prefill.json', 'utf8'))),
  decode: validateDocument(JSON.parse(readFileSync('public/data/decode.json', 'utf8'))),
};
const rmsNorm = 'model/final-normalization/rms_norm.1449';

test.beforeEach(async ({ page }) => { await installReadiness(page); });

async function openFlow(page: Page, scenario: 'prefill' | 'decode', entityId: string): Promise<void> {
  await page.goto(`./#scenario=${scenario}&entity=${encodeURIComponent(entityId)}`);
  await mounted(page);
  await closeInspector(page);
  if (await page.locator('.graph-evidence').getAttribute('open') === null) {
    await arm(page, '.graph-evidence', 'open', '');
    await page.locator('.graph-evidence > summary').click();
    await changed(page);
  }
  await arm(page, '.operator-graph', 'data-view', 'fit');
  await page.locator('.graph-toolbar__fit').click();
  await changed(page);
}

async function checkFlow(page: Page, scenario: 'prefill' | 'decode', entityId: string): Promise<void> {
  const capture = captures[scenario];
  const model = buildGraphModel(capture, entityId);
  const boundaries = new Map<string, string>();
  for (const edge of model.edges) {
    for (const endpoint of [edge.sourceId, edge.targetId]) {
      if (endpoint.startsWith('boundary:')) boundaries.set(endpoint, edge.tensorId);
    }
  }
  const actual = await page.evaluate(() => {
    const svg = document.querySelector<SVGSVGElement>('.graph-svg');
    const viewport = document.querySelector<HTMLElement>('.graph-viewport');
    if (!svg || !viewport) throw new Error('Flow surface missing');
    const svgBounds = svg.getBoundingClientRect();
    const cards = [...svg.querySelectorAll('.graph-node__shape, .graph-tensor__shape')].map(card => card.getBoundingClientRect());
    const cardOverlaps = cards.flatMap((card, index) => cards.slice(index + 1).flatMap((other, offset) =>
      Math.min(card.right, other.right) - Math.max(card.left, other.left) > 0.5 &&
      Math.min(card.bottom, other.bottom) - Math.max(card.top, other.top) > 0.5 ? [[index, index + offset + 1]] : []));
    const clippedText = [...svg.querySelectorAll('foreignObject')].flatMap(box => {
      const content = box.firstElementChild;
      if (!content) throw new Error('Flow text missing');
      const bounds = box.getBoundingClientRect();
      return content.scrollHeight > bounds.height + 1 || content.scrollWidth > bounds.width + 1
        ? [content.textContent] : [];
    });
    const arrowFailures = [...svg.querySelectorAll('.graph-arrow')].flatMap(arrow => {
      const box = arrow.getBoundingClientRect();
      const edgeKey = arrow.parentElement?.querySelector('.graph-edge')?.getAttribute('data-edge-key');
      const clipped = box.left < svgBounds.left - 0.5 || box.right > svgBounds.right + 0.5 ||
        box.top < svgBounds.top - 0.5 || box.bottom > svgBounds.bottom + 0.5;
      const covered = cards.some(card => Math.min(box.right, card.right) - Math.max(box.left, card.left) > 0.5 &&
        Math.min(box.bottom, card.bottom) - Math.max(box.top, card.top) > 0.5);
      return clipped || covered ? [{ edgeKey, clipped, covered }] : [];
    });
    return {
      tensors: [...svg.querySelectorAll<SVGElement>('.graph-tensor')].map(tensor => ({
        endpoint: tensor.dataset.endpointId,
        tensorId: tensor.dataset.tensorId,
        name: tensor.querySelector('.graph-tensor__name')?.textContent,
        interactive: tensor.getAttribute('role') === 'button' || tensor.hasAttribute('tabindex'),
        box: tensor.querySelector('.graph-tensor__shape')?.getBoundingClientRect().toJSON(),
      })),
      nodeIds: [...svg.querySelectorAll<SVGElement>('.graph-node')].map(node => node.dataset.entityId),
      nodeBoxes: [...svg.querySelectorAll<SVGElement>('.graph-node')].map(node => ({
        id: node.dataset.entityId, box: node.querySelector('.graph-node__shape')?.getBoundingClientRect().toJSON(),
      })),
      listNodeIds: [...document.querySelectorAll<HTMLElement>('.graph-list__node')].map(node => node.dataset.entityId),
      edgeKeys: [...svg.querySelectorAll<SVGElement>('.graph-edge')].map(edge => edge.dataset.edgeKey),
      listEdgeKeys: [...document.querySelectorAll<HTMLElement>('.graph-list__edge')].map(edge => edge.dataset.edgeKey),
      obsoleteSymbols: document.querySelectorAll('.graph-legend, .edge-marker').length,
      arrowCount: svg.querySelectorAll('.graph-arrow').length,
      arrowFailures,
      cardOverlaps,
      clippedText,
      width: svgBounds.width,
      available: viewport.clientWidth,
      documentOverflow: document.documentElement.scrollWidth > innerWidth || document.documentElement.scrollHeight > innerHeight,
    };
  });
  expect(actual.tensors.map(tensor => tensor.endpoint).sort()).toEqual([...boundaries.keys()].sort());
  for (const tensor of actual.tensors) {
    expect(tensor.tensorId).toBe(boundaries.get(tensor.endpoint ?? ''));
    expect(tensor.name).toBe(capture.tensors.find(item => item.id === tensor.tensorId)?.name);
    expect(tensor.interactive).toBe(false);
    const firstConsumer = model.edges.find(edge => edge.sourceId === tensor.endpoint);
    if (firstConsumer?.role === 'weight') {
      const consumer = actual.nodeBoxes.find(node => node.id === firstConsumer.targetId)?.box;
      expect(consumer).toBeDefined();
      expect(tensor.box?.left).toBeGreaterThan(consumer?.right ?? Infinity);
      const center = (tensor.box?.top ?? 0) + (tensor.box?.height ?? 0) / 2;
      expect(center).toBeGreaterThanOrEqual(consumer?.top ?? Infinity);
      expect(center).toBeLessThanOrEqual(consumer?.bottom ?? -Infinity);
    }
  }
  expect(actual.nodeIds).toEqual(model.nodes.map(node => node.entityId));
  expect(actual.listNodeIds).toEqual(actual.nodeIds);
  expect(actual.edgeKeys).toEqual(model.edges.map(edge => edge.key));
  expect(actual.listEdgeKeys).toEqual(actual.edgeKeys);
  expect(actual.obsoleteSymbols).toBe(0);
  expect(actual.arrowCount).toBe(model.edges.length);
  expect(actual.arrowFailures).toEqual([]);
  expect(actual.cardOverlaps).toEqual([]);
  expect(actual.clippedText).toEqual([]);
  expect(actual.width).toBeLessThanOrEqual(actual.available + 0.5);
  expect(actual.documentOverflow).toBe(false);
}

async function checkPrimaryAxis(page: Page): Promise<void> {
  const axis = await page.evaluate(() => {
    const shape = (selector: string): DOMRect => {
      const node = document.querySelector(selector);
      if (!node) throw new Error(`Flow card missing: ${selector}`);
      return node.getBoundingClientRect();
    };
    const input = shape('.graph-tensor[data-boundary-role=input] .graph-tensor__shape');
    const operator = shape('.graph-node__shape');
    const output = shape('.graph-tensor[data-boundary-role=output] .graph-tensor__shape');
    return {
      inputBottom: input.bottom, operatorTop: operator.top, operatorBottom: operator.bottom, outputTop: output.top,
      centers: [input, operator, output].map(box => box.x + box.width / 2),
      paths: [...document.querySelectorAll<SVGPathElement>('.graph-edge')].filter(path => path.dataset.edgeRole !== 'weight').map(path => {
        const length = path.getTotalLength();
        return [0, length / 2, length].map(at => path.getPointAtLength(at).x);
      }),
    };
  });
  expect(axis.inputBottom).toBeLessThan(axis.operatorTop);
  expect(axis.operatorBottom).toBeLessThan(axis.outputTop);
  expect(Math.max(...axis.centers) - Math.min(...axis.centers)).toBeLessThan(1);
  for (const xs of axis.paths) expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(1);
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
  test(`RMSNorm shows a straight vertical flow without role symbols at ${viewport.width}`, async ({ page }, info) => {
    // Given: the user's exact selected operator and actual capture.
    await page.setViewportSize(viewport);
    // When: the flow is opened and fitted.
    await openFlow(page, 'prefill', rmsNorm);
    // Then: input/operator/output share a straight vertical axis without a shape legend.
    await checkFlow(page, 'prefill', rmsNorm);
    await checkPrimaryAxis(page);
    expect(await page.locator('.graph-node__content').allTextContents()).toEqual(['rms_norm']);
    const fittedIds = await page.locator('.graph-tensor').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-endpoint-id')));
    await page.locator('.graph-viewport').scrollIntoViewIfNeeded();
    await page.screenshot({ path: info.outputPath(`rms-flow-${viewport.width}.png`) });
    await arm(page, '.operator-graph', 'data-view', 'natural');
    await page.locator('.graph-toolbar__reset').click(); await changed(page);
    expect(await page.locator('.inspector').getAttribute('data-entity-id')).toBe(rmsNorm);
    expect(await page.locator('.graph-tensor').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-endpoint-id')))).toEqual(fittedIds);
    expect(await page.locator('.graph-viewport').evaluate(node => [node.scrollLeft, node.scrollTop])).toEqual([0, 0]);
    await checkFlow(page, 'prefill', rmsNorm);
    await checkPrimaryAxis(page);
  });

  test(`weight joins its operator from the side at ${viewport.width}`, async ({ page }, info) => {
    // Given: the actual final normalization multiply has one activation and one weight.
    const entityId = 'model/final-normalization/mul.1450';
    await page.setViewportSize(viewport);
    // When
    await openFlow(page, 'prefill', entityId);
    // Then: weight is beside the operator, never another step above it.
    await checkFlow(page, 'prefill', entityId);
    await checkPrimaryAxis(page);
    const side = await page.evaluate(() => {
      const operator = document.querySelector('.graph-node__shape')?.getBoundingClientRect();
      const weight = document.querySelector('.graph-tensor[data-boundary-role=weight] .graph-tensor__shape')?.getBoundingClientRect();
      if (!operator || !weight) throw new Error('Operator/weight pair missing');
      const edge = document.querySelector('.graph-edge[data-edge-role=weight]');
      return {
        operatorRight: operator.right, operatorTop: operator.top, operatorBottom: operator.bottom,
        weightLeft: weight.left, weightCenter: weight.y + weight.height / 2,
        direction: edge?.parentElement?.querySelector('.graph-arrow')?.getAttribute('data-direction'),
      };
    });
    expect(side.weightLeft).toBeGreaterThan(side.operatorRight);
    expect(side.weightCenter).toBeGreaterThanOrEqual(side.operatorTop);
    expect(side.weightCenter).toBeLessThanOrEqual(side.operatorBottom);
    expect(side.direction).toBe('left');
    await page.locator('.graph-viewport').scrollIntoViewIfNeeded();
    await page.screenshot({ path: info.outputPath(`weight-side-${viewport.width}.png`) });
  });
}

for (const width of [1440, 390]) {
  test(`complex flows preserve boundaries, side consumers and arrows at ${width}`, async ({ page }) => {
    // Given: shared inputs, weights, state and internal connections in both scenarios.
    await page.setViewportSize({ width, height: 900 });
    for (const scenario of ['prefill', 'decode'] as const) {
      for (const entityId of ['mamba-130m', 'block.23', 'block.23/selective-scan-state', 'model/final-normalization']) {
        // When / Then: all identities survive, with weight anchored to its real consumer.
        await openFlow(page, scenario, entityId);
        await checkFlow(page, scenario, entityId);
      }
    }
  });
}

test('fitting a roomy flow changes width and reset restores its natural size', async ({ page }) => {
  // Given: enough space to distinguish natural width from fitted width.
  await page.setViewportSize({ width: 1920, height: 1080 });
  await openFlow(page, 'prefill', rmsNorm);
  const fittedWidth = await page.locator('.graph-svg').evaluate(node => node.getBoundingClientRect().width);
  // When
  await arm(page, '.operator-graph', 'data-view', 'natural');
  await page.locator('.graph-toolbar__reset').click(); await changed(page);
  // Then: reset is not a cosmetic flag change.
  expect(await page.locator('.graph-svg').evaluate(node => node.getBoundingClientRect().width)).toBeLessThan(fittedWidth);
  await checkFlow(page, 'prefill', rmsNorm);
});
