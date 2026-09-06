// allow: SIZE_OK - task2 owns this single schema module; keep its boundary checks readable without an unauthorized source split.
export type Kind = 'model' | 'block' | 'stage' | 'operator';
export type Role = 'activation' | 'weight' | 'state' | 'index';
export type DType = 'F32' | 'F16' | 'BF16' | 'I32' | 'I16' | 'I8' | 'Q8_0' | 'Q4_0';
export type Formula = { op: 'const'; value: number } | { op: 'dim'; name: string } | { op: 'add' | 'mul' | 'sub'; left: Formula; right: Formula };
export interface GgufPayload { fileId: string; offsetBytes: number; bytes: number; }
export interface Storage {
  ggmlNbytes: number; requiredAllocBytes: number | null;
  bufferId: string | null; bufferOffsetBytes: number | null; bufferBytes: number | null;
  observationEpoch: number; allocatorSlotBytes: number | null; allocatorSlotReason?: string;
}
export interface Tensor {
  id: string; name: string; role: Role; dtype: DType;
  nativeShape: readonly number[]; strides: readonly number[];
  logicalShape: readonly number[]; axisLabels: readonly string[];
  numel: number; typeBlockSize: number; logicalBytes: number;
  storage: Storage; ggufPayload?: GgufPayload; captureId?: string;
  viewSourceId: string | null; viewOffsetBytes: number | null;
  producerIds: readonly string[]; consumerIds: readonly string[];
  provenance: string; classification: 'observed' | 'derived';
}
export interface Entity {
  id: string; kind: Kind; parentId: string | null; children: readonly string[];
  inputTensorIds: readonly string[]; outputTensorIds: readonly string[]; weightTensorIds: readonly string[];
}
export interface CaptureDocument {
  schemaVersion: 1; captureId: string; ggufSha256: string;
  scenario: { name: string; dimensions: Record<string, number> };
  entities: readonly Entity[]; tensors: readonly Tensor[];
}
export interface ExpectedIdentity { captureId: string; ggufSha256: string; }

// Pinned GGML 8144f319: ggml-common.h block_q8_0/block_q4_0 include a two-byte scale.
const sizes = { F32: 4, F16: 2, BF16: 2, I32: 4, I16: 2, I8: 1, Q8_0: 34, Q4_0: 18 } as const;
const blocks = { F32: 1, F16: 1, BF16: 1, I32: 1, I16: 1, I8: 1, Q8_0: 32, Q4_0: 32 } as const;
class SchemaError extends Error {
  readonly path: string;
  readonly detail: string;
  constructor(path: string, detail: string) {
    super(`${path}: ${detail}`);
    this.path = path;
    this.detail = detail;
  }
}
function fail(p: string, detail: string): never { throw new SchemaError(p, detail); }
function obj(v: unknown, p: string): Record<string, unknown> {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) return fail(p, 'must be an object');
  return Object.fromEntries(Object.entries(v));
}
function str(v: unknown, p: string): string {
  return typeof v === 'string' && v.length > 0 ? v : fail(p, 'must be a non-empty string');
}
function integer(v: unknown, p: string, positive = false): number {
  return typeof v === 'number' && Number.isSafeInteger(v) && (positive ? v > 0 : v >= 0) ? v : fail(p, 'must be a safe integer');
}
function list(v: unknown, p: string): unknown[] {
  return Array.isArray(v) ? v : fail(p, 'must be an array');
}
function ids(v: unknown, p: string): string[] {
  const a = list(v, p).map((x, i) => str(x, `${p}[${i}]`));
  if (new Set(a).size !== a.length) fail(p, 'contains duplicate IDs');
  return a;
}
function choice<T extends string>(v: unknown, allowed: readonly T[], p: string): T {
  return allowed.find(item => item === v) ?? fail(p, 'unsupported value');
}
function checkedAdd(a: number, b: number, p: string): number { return integer(a + b, p); }
function reference<T>(map: ReadonlyMap<string, T>, id: string, p: string): T {
  return map.get(id) ?? fail(p, `dangling reference ${id}`);
}

function tensor(v: unknown, i: number): Tensor {
  const p = `tensors[${i}]`, x = obj(v, p);
  const native = list(x.nativeShape, p + '.nativeShape').map((n, j) => integer(n, `${p}.nativeShape[${j}]`, true));
  const logical = list(x.logicalShape, p + '.logicalShape').map((n, j) => integer(n, `${p}.logicalShape[${j}]`, true));
  const strides = list(x.strides, p + '.strides').map((n, j) => integer(n, `${p}.strides[${j}]`));
  const labels = list(x.axisLabels, p + '.axisLabels').map((n, j) => str(n, `${p}.axisLabels[${j}]`));
  if (native.length !== 4 || strides.length !== 4) fail(p, 'nativeShape and strides must have four dimensions');
  const numel = logical.reduce((a, b) => integer(a * b, p + '.logicalShape'), 1);
  if (logical.length !== labels.length || native.reduce((a, b) => integer(a * b, p + '.nativeShape'), 1) !== numel) fail(p, 'native and logical element counts differ');
  const role = choice(x.role, ['activation', 'weight', 'state', 'index'] as const, p + '.role');
  const dtype = choice(x.dtype, ['F32', 'F16', 'BF16', 'I32', 'I16', 'I8', 'Q8_0', 'Q4_0'] as const, p + '.dtype');
  const classification = choice(x.classification, ['observed', 'derived'] as const, p + '.classification');
  const blockSize = blocks[dtype];
  if (integer(x.numel, p + '.numel') !== numel) fail(p + '.numel', 'does not match logical shape');
  if (integer(x.typeBlockSize, p + '.typeBlockSize') !== blockSize) fail(p + '.typeBlockSize', 'does not match dtype');
  const ne0 = integer(native[0], p + '.nativeShape[0]', true), nb0 = integer(strides[0], p + '.strides[0]');
  if (ne0 % blockSize !== 0) fail(p + '.nativeShape[0]', 'must contain whole dtype blocks');
  const logicalBytes = integer(numel / blockSize * sizes[dtype], p + '.logicalBytes');
  if (integer(x.logicalBytes, p + '.logicalBytes') !== logicalBytes) fail(p + '.logicalBytes', 'does not match dtype payload');
  // ggml.c:1297-1319: exact addressed span, not ggml_nbytes_pad or backend allocation.
  let span = blockSize === 1 ? sizes[dtype] : integer(ne0 / blockSize * nb0, p + '.strides');
  for (const [axis, extent] of native.entries()) {
    if (blockSize !== 1 && axis === 0) continue;
    span = checkedAdd(span, integer((extent - 1) * integer(strides[axis], p + '.strides'), p + '.strides'), p + '.strides');
  }
  const ss = obj(x.storage, p + '.storage');
  const storage: Storage = {
    ggmlNbytes: integer(ss.ggmlNbytes, p + '.storage.ggmlNbytes'),
    requiredAllocBytes: ss.requiredAllocBytes === null ? null : integer(ss.requiredAllocBytes, p + '.storage.requiredAllocBytes'),
    bufferId: ss.bufferId === null ? null : str(ss.bufferId, p + '.storage.bufferId'),
    bufferOffsetBytes: ss.bufferOffsetBytes === null ? null : integer(ss.bufferOffsetBytes, p + '.storage.bufferOffsetBytes'),
    bufferBytes: ss.bufferBytes === null ? null : integer(ss.bufferBytes, p + '.storage.bufferBytes'),
    observationEpoch: integer(ss.observationEpoch, p + '.storage.observationEpoch'),
    allocatorSlotBytes: ss.allocatorSlotBytes === null ? null : fail(p + '.storage.allocatorSlotBytes', 'must remain null'),
    allocatorSlotReason: str(ss.allocatorSlotReason, p + '.storage.allocatorSlotReason'),
  };
  if (storage.ggmlNbytes !== span) fail(p + '.storage.ggmlNbytes', 'does not match strided addressed span');
  if (storage.requiredAllocBytes !== null && storage.requiredAllocBytes < span) fail(p + '.storage.requiredAllocBytes', 'smaller than addressed span');
  if (storage.bufferId === null && (storage.bufferOffsetBytes !== null || storage.bufferBytes !== null)) fail(p + '.storage', 'runtime range fields must be all known or null');
  if (storage.bufferId !== null) {
    if (storage.bufferOffsetBytes === null || storage.bufferBytes === null) return fail(p + '.storage', 'runtime range fields must be all known or null');
    if (checkedAdd(storage.bufferOffsetBytes, span, p + '.storage') > storage.bufferBytes) fail(p + '.storage', 'range exceeds backing buffer');
  }
  const viewSourceId = x.viewSourceId === null ? null : str(x.viewSourceId, p + '.viewSourceId');
  const viewOffsetBytes = x.viewOffsetBytes === null ? null : integer(x.viewOffsetBytes, p + '.viewOffsetBytes');
  if ((viewSourceId === null) !== (viewOffsetBytes === null)) fail(p + '.viewOffsetBytes', 'must be explicit for views and null for non-views');
  if (viewSourceId !== null && storage.requiredAllocBytes !== null) fail(p + '.storage.requiredAllocBytes', 'views do not allocate independently');
  let payload: GgufPayload | undefined;
  if (x.ggufPayload !== undefined) {
    const g = obj(x.ggufPayload, p + '.ggufPayload');
    payload = { fileId: str(g.fileId, p + '.ggufPayload.fileId'), offsetBytes: integer(g.offsetBytes, p + '.ggufPayload.offsetBytes'), bytes: integer(g.bytes, p + '.ggufPayload.bytes', true) };
    checkedAdd(payload.offsetBytes, payload.bytes, p + '.ggufPayload');
  }
  return {
    id: str(x.id, p + '.id'), name: str(x.name, p + '.name'), role, dtype, classification,
    nativeShape: native, strides, logicalShape: logical, axisLabels: labels, numel, typeBlockSize: blockSize, logicalBytes, storage,
    ...(payload ? { ggufPayload: payload } : {}), ...(x.captureId === undefined ? {} : { captureId: str(x.captureId, p + '.captureId') }),
    viewSourceId, viewOffsetBytes, producerIds: ids(x.producerIds, p + '.producerIds'), consumerIds: ids(x.consumerIds, p + '.consumerIds'), provenance: str(x.provenance, p + '.provenance'),
  };
}

export function validateDocument(value: unknown, expected?: ExpectedIdentity): CaptureDocument {
  const x = obj(value, 'document');
  if (x.schemaVersion !== 1) fail('schemaVersion', 'must equal 1');
  const captureId = str(x.captureId, 'captureId'), hash = str(x.ggufSha256, 'ggufSha256');
  if (!/^[a-f\d]{64}$/i.test(hash)) fail('ggufSha256', 'must be a SHA-256 hex digest');
  if (expected && (captureId !== expected.captureId || hash !== expected.ggufSha256)) fail('identity', 'does not match expected capture');
  const sc = obj(x.scenario, 'scenario'), name = choice(sc.name, ['prefill', 'decode'] as const, 'scenario.name');
  const dimensions: Record<string, number> = {};
  for (const [k, v] of Object.entries(obj(sc.dimensions, 'scenario.dimensions'))) dimensions[k] = integer(v, `scenario.dimensions.${k}`);
  const { P, T, Q, O } = dimensions;
  if (P !== 1 || O !== 1) fail('scenario.dimensions', 'canonical captures require P=1 and O=1');
  if (T !== (name === 'prefill' ? 16 : 1)) fail('scenario.dimensions.T', 'does not match scenario');
  if (Q !== P * T) fail('scenario.dimensions.Q', 'must equal P*T');
  const ts = list(x.tensors, 'tensors').map(tensor);
  if (new Set(ts.map(t => t.id)).size !== ts.length) fail('tensors', 'contains duplicate IDs');
  const tm = new Map(ts.map(t => [t.id, t]));
  const capacities = new Map<string, number | null>();
  for (const t of ts) {
    const p = `tensors.${t.id}`, s = t.storage;
    if (t.captureId !== undefined && t.captureId !== captureId) fail(p + '.captureId', 'does not match document');
    if (s.bufferId !== null) {
      if (capacities.has(s.bufferId) && capacities.get(s.bufferId) !== s.bufferBytes) fail(p + '.storage.bufferBytes', 'inconsistent for backing buffer');
      capacities.set(s.bufferId, s.bufferBytes);
    }
    const aliases = new Set([t.id]);
    let current = t;
    while (current.viewSourceId !== null) {
      if (aliases.has(current.viewSourceId)) fail(p + '.viewSourceId', 'alias cycle');
      aliases.add(current.viewSourceId);
      current = reference(tm, current.viewSourceId, p + '.viewSourceId');
    }
    if (t.viewSourceId !== null) {
      const source = reference(tm, t.viewSourceId, p + '.viewSourceId'), offset = integer(t.viewOffsetBytes, p + '.viewOffsetBytes');
      // ggml_new_tensor_impl checks compact size before ggml_view sets its strides.
      if (checkedAdd(offset, Math.max(t.logicalBytes, s.ggmlNbytes), p + '.viewOffsetBytes') > source.storage.ggmlNbytes) fail(p + '.viewOffsetBytes', 'view exceeds backing source');
      if (s.bufferId !== source.storage.bufferId) fail(p + '.viewSourceId', 'view uses a different backing buffer');
      if (source.storage.bufferOffsetBytes !== null && s.bufferOffsetBytes !== checkedAdd(source.storage.bufferOffsetBytes, offset, p + '.viewOffsetBytes')) fail(p + '.viewOffsetBytes', 'does not agree with backing offset');
    }
  }
  const es: Entity[] = list(x.entities, 'entities').map((v, i) => {
    const p = `entities[${i}]`, e = obj(v, p);
    return {
      id: str(e.id, p + '.id'), kind: choice(e.kind, ['model', 'block', 'stage', 'operator'] as const, p + '.kind'),
      parentId: e.parentId === null ? null : str(e.parentId, p + '.parentId'), children: ids(e.children, p + '.children'),
      inputTensorIds: ids(e.inputTensorIds, p + '.inputTensorIds'), outputTensorIds: ids(e.outputTensorIds, p + '.outputTensorIds'), weightTensorIds: ids(e.weightTensorIds, p + '.weightTensorIds'),
    };
  });
  if (new Set(es.map(e => e.id)).size !== es.length) fail('entities', 'contains duplicate IDs');
  const em = new Map(es.map(e => [e.id, e])), roots = es.filter(e => e.parentId === null), root = roots[0];
  if (roots.length !== 1 || root?.kind !== 'model') return fail('entities', 'requires one model root');
  for (const e of es) {
    const p = `entities.${e.id}`;
    if (e.parentId !== null && !reference(em, e.parentId, p + '.parentId').children.includes(e.id)) fail(p, 'parent-child mismatch');
    for (const id of e.children) if (reference(em, id, p + '.children').parentId !== e.id) fail(p + '.children', 'parent-child mismatch');
    for (const id of [...e.inputTensorIds, ...e.outputTensorIds, ...e.weightTensorIds]) reference(tm, id, p);
    for (const id of e.weightTensorIds) if (reference(tm, id, p).role !== 'weight') fail(p + '.weightTensorIds', 'must reference weights');
    if (e.kind === 'operator') {
      for (const id of [...e.inputTensorIds, ...e.weightTensorIds]) if (!reference(tm, id, p).consumerIds.includes(e.id)) fail(p, 'input/weight edge is not reciprocal');
      for (const id of e.outputTensorIds) if (!reference(tm, id, p).producerIds.includes(e.id)) fail(p, 'output edge is not reciprocal');
    }
  }
  for (const t of ts) {
    const p = `tensors.${t.id}`;
    for (const id of t.producerIds) {
      const e = reference(em, id, p + '.producerIds');
      if (e.kind !== 'operator' || !e.outputTensorIds.includes(t.id)) fail(p + '.producerIds', 'producer edge is not reciprocal');
    }
    for (const id of t.consumerIds) {
      const e = reference(em, id, p + '.consumerIds');
      if (e.kind !== 'operator' || ![...e.inputTensorIds, ...e.weightTensorIds].includes(t.id)) fail(p + '.consumerIds', 'consumer edge is not reciprocal');
    }
  }
  const seen = new Set<string>();
  function walk(e: Entity): void {
    if (seen.has(e.id)) fail('entities', 'hierarchy cycle');
    seen.add(e.id);
    for (const id of e.children) walk(reference(em, id, 'entities'));
  }
  walk(root);
  if (seen.size !== es.length) fail('entities', 'unreachable entity');
  for (const e of es.filter(e => e.kind !== 'operator')) {
    const descendants = new Set<string>(), pending = [...e.children];
    for (const id of pending) {
      descendants.add(id);
      pending.push(...reference(em, id, 'entities').children);
    }
    const operators = es.filter(n => descendants.has(n.id) && n.kind === 'operator');
    const inputs = new Set(operators.flatMap(n => n.inputTensorIds));
    const outputs = new Set(operators.flatMap(n => n.outputTensorIds));
    // A tensor crossing any outside edge stays visible, including residual/state fan-out.
    const crosses = (ids: readonly string[]) => ids.length === 0 || ids.some(id => !descendants.has(id));
    const boundary = {
      inputTensorIds: new Set([...inputs].filter(id => crosses(reference(tm, id, 'tensors').producerIds))),
      outputTensorIds: new Set([...outputs].filter(id => crosses(reference(tm, id, 'tensors').consumerIds))),
      weightTensorIds: new Set(operators.flatMap(n => n.weightTensorIds)),
    };
    for (const key of ['inputTensorIds', 'outputTensorIds', 'weightTensorIds'] as const) {
      if (e[key].length !== boundary[key].size || e[key].some(id => !boundary[key].has(id))) fail(`entities.${e.id}.${key}`, 'does not equal descendant crossing boundary');
    }
  }
  return { schemaVersion: 1, captureId, ggufSha256: hash, scenario: { name, dimensions }, entities: es, tensors: ts };
}

export function evalFormula(formula: Formula, dimensions: Record<string, number>): number {
  switch (formula.op) {
    case 'const': return integer(formula.value, 'formula.const');
    case 'dim': return integer(dimensions[formula.name], `dimension ${formula.name}`);
    case 'add': return checkedAdd(evalFormula(formula.left, dimensions), evalFormula(formula.right, dimensions), 'formula');
    case 'mul': return integer(evalFormula(formula.left, dimensions) * evalFormula(formula.right, dimensions), 'formula');
    case 'sub': return integer(evalFormula(formula.left, dimensions) - evalFormula(formula.right, dimensions), 'formula');
    default: {
      const exhaustive: never = formula;
      return fail('formula.op', `unsupported operation: ${String(exhaustive)}`);
    }
  }
}
export function storageAccounting(tensors: readonly Tensor[]) {
  let logicalBytes = 0, uniqueGgufBytes = 0, runtimeTotal = 0;
  const payloadRanges = new Map<string, Array<[number, number]>>(), runtime = new Set<string>();
  for (const t of tensors) {
    logicalBytes = checkedAdd(logicalBytes, t.logicalBytes, 'storage.logicalBytes');
    if (t.storage.bufferId !== null) runtimeTotal = checkedAdd(runtimeTotal, t.storage.ggmlNbytes, 'storage.runtime span');
    if (t.ggufPayload) {
      const p = t.ggufPayload, ranges = payloadRanges.get(p.fileId) ?? [];
      ranges.push([p.offsetBytes, checkedAdd(p.offsetBytes, p.bytes, 'gguf range')]);
      payloadRanges.set(p.fileId, ranges);
    }
    const s = t.storage;
    if (s.bufferId !== null) runtime.add(JSON.stringify([s.bufferId, s.bufferOffsetBytes, s.ggmlNbytes, s.observationEpoch]));
  }
  for (const ranges of payloadRanges.values()) {
    ranges.sort((a, b) => a[0] - b[0]);
    let end = 0;
    for (const [r, e] of ranges) {
      if (e > end) {
        uniqueGgufBytes = checkedAdd(uniqueGgufBytes, e - Math.max(r, end), 'storage.uniqueGgufBytes');
        end = e;
      }
    }
  }
  return { logicalBytes, uniqueGgufBytes, uniqueRuntimeRanges: runtime.size };
}
