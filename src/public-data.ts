import { SchemaError, validateDocument, type CaptureDocument } from './schema';

type Scenario = 'prefill' | 'decode';
export type PublicBundleSources = Readonly<Record<'manifest' | 'model' | 'provenance' | 'validation' | Scenario, string>>;
interface Header { readonly schemaVersion: 1; readonly captureId: string; readonly ggufSha256: string; }
interface Approval { readonly numericalStatus: 'pass'; readonly policySha256: string; readonly reviewSha256: string; }
export interface ManifestCapture {
  readonly captureId: string; readonly path: string; readonly sha256: string;
  readonly tensorCount: number; readonly operatorCount: number; readonly emptyTensorCount: number;
}
export interface PublicManifest extends Header, Approval {
  readonly modelId: string; readonly artifactAuditStatus: 'exact';
  readonly captures: Readonly<Record<Scenario, ManifestCapture>>;
}
export interface PublicModel extends Header {
  readonly modelId: string; readonly architecture: 'Mamba1'; readonly blockCount: number;
  readonly vocabularySize: number; readonly tokenizerVocabularySize: number;
  readonly configuration: Readonly<Record<'num_hidden_layers' | 'hidden_size' | 'intermediate_size' | 'state_size' | 'conv_kernel' | 'time_step_rank' | 'vocab_size', number>>;
}
type Setting = string | number | boolean | null | readonly string[];
export interface PublicProvenance extends Header, Approval {
  readonly repoId: string; readonly revision: string; readonly manifestSha256: string;
  readonly preparationCaptureId: string; readonly artifactAuditSha256: string; readonly configurationHash: string;
  readonly canonicalNativeCaptureId: string;
  readonly source: Readonly<Record<'commit' | 'runtimeCommit' | 'converterSha256' | 'mambaConverterSha256' | 'productPatchSha256' | 'worktreeDiffSha256', string>>;
  readonly environment: { readonly cpu: string; readonly python: string; readonly packages: Readonly<Record<string, string>>; readonly lockSha256: string; readonly requirementsSha256: string; readonly pythonBinarySha256: string };
  readonly compiler: { readonly compilerId: string; readonly compilerVersion: string; readonly compilerSha256: string };
  readonly buildFiles: Readonly<Record<string, string>>; readonly cmakeFlagsSha256: string;
  readonly cmakeFlags: Readonly<Record<string, Setting>>; readonly configured: Readonly<Record<string, Setting>>;
  readonly effective: Readonly<Record<string, Setting>>; readonly nativeEnvironment: Readonly<Record<string, string>>; readonly systemInfo: string;
}
export interface ValidationMetric {
  readonly maxAbs: number; readonly rmse: number; readonly nmse: number; readonly absoluteBound: number;
  readonly absolutePercentiles: readonly number[]; readonly componentViolations: 0;
  readonly referenceSha256: string; readonly candidateSha256: string; readonly shape: readonly number[];
  readonly envelopeViolations: readonly string[];
}
interface CaptureBinding {
  readonly case: string; readonly runtime: 'hf' | 'llama-fresh' | 'llama-split'; readonly captureId: string;
  readonly receiptSha256: string; readonly archiveSha256: string; readonly runtimeHash: string; readonly fixtureSha256: string; readonly arrayCount: number;
}
export interface PublicValidation extends Header {
  readonly status: 'pass'; readonly policySha256: string; readonly reviewSha256: string; readonly configurationHash: string;
  readonly bindings: { readonly manifestSha256: string; readonly ggufSha256: string; readonly auditSha256: string;
    readonly preparationCaptureId: string; readonly captures: readonly CaptureBinding[] };
  readonly metrics: Readonly<Record<string, ValidationMetric>>; readonly failures: readonly string[];
  readonly implementation: Readonly<Record<'validator' | 'ingestion' | 'graph', string>>;
}
export interface PublicBundle {
  readonly manifest: PublicManifest; readonly model: PublicModel; readonly provenance: PublicProvenance;
  readonly validation: PublicValidation; readonly documents: Readonly<Record<Scenario, CaptureDocument>>;
}

function requireValue(condition: boolean, path: string, detail = 'does not match release contract'): asserts condition {
  if (!condition) throw new SchemaError(path, detail);
}
function object(value: unknown, path: string): Record<string, unknown> {
  requireValue(typeof value === 'object' && value !== null && !Array.isArray(value), path, 'must be an object');
  return Object.fromEntries(Object.entries(value));
}
function text(value: unknown, path: string): string {
  requireValue(typeof value === 'string' && value.length > 0, path, 'must be nonempty text'); return value;
}
function digest(value: unknown, path: string): string {
  const result = text(value, path); requireValue(/^[a-f0-9]{64}$/.test(result), path, 'must be a lowercase SHA-256'); return result;
}
function number(value: unknown, path: string): number {
  requireValue(typeof value === 'number' && Number.isFinite(value) && value >= 0, path, 'must be finite and nonnegative'); return value;
}
function integer(value: unknown, path: string): number {
  const result = number(value, path); requireValue(Number.isSafeInteger(result), path, 'must be a safe integer'); return result;
}
function array(value: unknown, path: string): unknown[] {
  requireValue(Array.isArray(value), path, 'must be an array'); return value;
}
function fields<K extends string, T>(value: unknown, keys: readonly K[], parse: (v: unknown, p: string) => T, path: string): Readonly<Record<K, T>>;
function fields(value: unknown, keys: readonly string[], parse: (v: unknown, p: string) => unknown, path: string): object {
  const x = object(value, path); return Object.fromEntries(keys.map(k => [k, parse(x[k], `${path}.${k}`)]));
}
function record<T>(value: unknown, parse: (v: unknown, p: string) => T, path: string): Readonly<Record<string, T>> {
  return Object.fromEntries(Object.entries(object(value, path)).map(([k, v]) => [k, parse(v, `${path}.${k}`)]));
}
function setting(value: unknown, path: string): Setting {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value;
  if (typeof value === 'number') { requireValue(Number.isFinite(value), path); return value; }
  return array(value, path).map((v, i) => text(v, `${path}[${i}]`));
}
function header(x: Record<string, unknown>, path: string): Header {
  requireValue(x.schemaVersion === 1, path + '.schemaVersion');
  return { schemaVersion: 1, captureId: text(x.captureId, path + '.captureId'), ggufSha256: digest(x.ggufSha256, path + '.ggufSha256') };
}
function approval(x: Record<string, unknown>, path: string): Approval {
  requireValue(x.numericalStatus === 'pass', path + '.numericalStatus', 'release requires independently reviewed pass');
  return { numericalStatus: 'pass', policySha256: digest(x.policySha256, path + '.policySha256'), reviewSha256: digest(x.reviewSha256, path + '.reviewSha256') };
}
function manifest(x: Record<string, unknown>): PublicManifest {
  const captures = object(x.captures, 'manifest.captures');
  const capture = (name: Scenario): ManifestCapture => {
    const p = `manifest.captures.${name}`, c = object(captures[name], p);
    requireValue(c.path === name + '.json', p + '.path');
    return { captureId: text(c.captureId, p + '.captureId'), path: c.path, sha256: digest(c.sha256, p + '.sha256'),
      ...fields(c, ['tensorCount', 'operatorCount', 'emptyTensorCount'] as const, integer, p) };
  };
  requireValue(x.artifactAuditStatus === 'exact', 'manifest.artifactAuditStatus');
  return { ...header(x, 'manifest'), ...approval(x, 'manifest'), artifactAuditStatus: 'exact', modelId: text(x.modelId, 'manifest.modelId'), captures: { prefill: capture('prefill'), decode: capture('decode') } };
}
function model(x: Record<string, unknown>): PublicModel {
  requireValue(x.architecture === 'Mamba1' && x.blockCount === 24 && x.vocabularySize === 50280 && x.tokenizerVocabularySize === 50277, 'model.architecture');
  const configuration = fields(x.configuration, ['num_hidden_layers', 'hidden_size', 'intermediate_size', 'state_size', 'conv_kernel', 'time_step_rank', 'vocab_size'] as const, integer, 'model.configuration');
  requireValue(configuration.num_hidden_layers === 24 && configuration.hidden_size === 768 && configuration.intermediate_size === 1536 && configuration.state_size === 16 && configuration.conv_kernel === 4 && configuration.time_step_rank === 48 && configuration.vocab_size === 50280, 'model.configuration');
  return { ...header(x, 'model'), modelId: text(x.modelId, 'model.modelId'), architecture: 'Mamba1', blockCount: 24, vocabularySize: 50280, tokenizerVocabularySize: 50277, configuration };
}
function provenance(x: Record<string, unknown>): PublicProvenance {
  const p = 'provenance', environment = object(x.environment, p + '.environment'), compiler = object(x.compiler, p + '.compiler');
  const source = fields(x.source, ['commit', 'runtimeCommit', 'converterSha256', 'mambaConverterSha256', 'productPatchSha256', 'worktreeDiffSha256'] as const, text, p + '.source');
  requireValue(source.commit === '8144f3192e5a3131cd043f284525e6ceebf82d0f' && source.runtimeCommit === source.commit, p + '.source.commit');
  for (const key of ['converterSha256', 'mambaConverterSha256', 'productPatchSha256', 'worktreeDiffSha256'] as const) digest(source[key], `${p}.source.${key}`);
  requireValue(x.repoId === 'state-spaces/mamba-130m-hf' && typeof x.revision === 'string' && /^[a-f0-9]{40}$/.test(x.revision), p + '.revision');
  return { ...header(x, p), ...approval(x, p), repoId: x.repoId, revision: x.revision, source,
    ...fields(x, ['manifestSha256', 'artifactAuditSha256', 'configurationHash', 'cmakeFlagsSha256'] as const, digest, p),
    ...fields(x, ['preparationCaptureId', 'canonicalNativeCaptureId', 'systemInfo'] as const, text, p),
    environment: { ...fields(environment, ['cpu', 'python'] as const, text, p + '.environment'), packages: record(environment.packages, text, p + '.environment.packages'),
      ...fields(environment, ['lockSha256', 'requirementsSha256', 'pythonBinarySha256'] as const, digest, p + '.environment') },
    compiler: { ...fields(compiler, ['compilerId', 'compilerVersion'] as const, text, p + '.compiler'), compilerSha256: digest(compiler.compilerSha256, p + '.compiler.compilerSha256') },
    buildFiles: record(x.buildFiles, digest, p + '.buildFiles'), cmakeFlags: record(x.cmakeFlags, setting, p + '.cmakeFlags'),
    configured: record(x.configured, setting, p + '.configured'), effective: record(x.effective, setting, p + '.effective'),
    nativeEnvironment: record(x.nativeEnvironment, text, p + '.nativeEnvironment') };
}
function metric(value: unknown, p: string): ValidationMetric {
  const x = object(value, p), measured = fields(x, ['maxAbs', 'rmse', 'nmse', 'absoluteBound'] as const, number, p);
  requireValue(x.componentViolations === 0 && measured.maxAbs <= measured.absoluteBound, p + '.componentViolations');
  const absolutePercentiles = array(x.absolutePercentiles, p + '.absolutePercentiles').map((v, i) => number(v, `${p}.absolutePercentiles[${i}]`));
  requireValue(absolutePercentiles.length === 3 && absolutePercentiles.every((v, i) => v <= measured.maxAbs && (i === 0 || v >= (absolutePercentiles[i - 1] ?? 0))), p + '.absolutePercentiles');
  const envelopeViolations = array(x.envelopeViolations, p + '.envelopeViolations');
  requireValue(envelopeViolations.length === 0, p + '.envelopeViolations');
  const shape = array(x.shape, p + '.shape').map((v, i) => integer(v, `${p}.shape[${i}]`));
  requireValue(shape.length > 0 && shape.every(v => v > 0), p + '.shape');
  return { ...measured, ...fields(x, ['referenceSha256', 'candidateSha256'] as const, digest, p), shape, absolutePercentiles, componentViolations: 0, envelopeViolations: [] };
}
function validation(x: Record<string, unknown>): PublicValidation {
  const p = 'validation', b = object(x.bindings, p + '.bindings');
  requireValue(x.status === 'pass', p + '.status');
  requireValue(array(x.failures, p + '.failures').length === 0, p + '.failures');
  const captures = array(b.captures, p + '.bindings.captures').map((v, i): CaptureBinding => {
    const path = `${p}.bindings.captures[${i}]`, c = object(v, path);
    requireValue(c.runtime === 'hf' || c.runtime === 'llama-fresh' || c.runtime === 'llama-split', path + '.runtime');
    return { ...fields(c, ['case', 'captureId'] as const, text, path), runtime: c.runtime,
      ...fields(c, ['receiptSha256', 'archiveSha256', 'runtimeHash', 'fixtureSha256'] as const, digest, path), arrayCount: integer(c.arrayCount, path + '.arrayCount') };
  });
  const metrics = record(x.metrics, metric, p + '.metrics'); requireValue(Object.keys(metrics).length > 0, p + '.metrics');
  return { ...header(x, p), status: 'pass', ...fields(x, ['policySha256', 'reviewSha256', 'configurationHash'] as const, digest, p),
    bindings: { ...fields(b, ['manifestSha256', 'ggufSha256', 'auditSha256'] as const, digest, p + '.bindings'), preparationCaptureId: text(b.preparationCaptureId, p + '.bindings.preparationCaptureId'), captures },
    metrics, failures: [], implementation: fields(x.implementation, ['validator', 'ingestion', 'graph'] as const, digest, p + '.implementation') };
}
async function sha256(source: string): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source)))].map(v => v.toString(16).padStart(2, '0')).join('');
}

/** Identity/format validation only. The independent numeric review remains an offline gate. */
export async function validatePublicBundle(sources: PublicBundleSources): Promise<PublicBundle> {
  const parse = (key: keyof PublicBundleSources): Record<string, unknown> => {
    try { const value: unknown = JSON.parse(sources[key]); return object(value, key); }
    catch (error) { if (error instanceof SyntaxError) throw new SchemaError(key, 'invalid JSON'); throw error; }
  };
  const m = manifest(parse('manifest')), modelData = model(parse('model')), p = provenance(parse('provenance')), v = validation(parse('validation'));
  for (const [name, item] of [['model', modelData], ['provenance', p], ['validation', v]] as const) {
    requireValue(item.captureId === m.captureId && item.ggufSha256 === m.ggufSha256, name + '.identity');
  }
  requireValue(m.modelId === 'mamba-130m' && modelData.modelId === m.modelId, 'model.modelId');
  for (const [name, item] of [['provenance', p], ['validation', v]] as const) {
    requireValue(item.policySha256 === m.policySha256 && item.reviewSha256 === m.reviewSha256, name + '.approval');
  }
  requireValue(p.configurationHash === v.configurationHash, 'validation.configurationHash');
  requireValue(v.bindings.manifestSha256 === p.manifestSha256 && v.bindings.ggufSha256 === m.ggufSha256 && v.bindings.auditSha256 === p.artifactAuditSha256 && v.bindings.preparationCaptureId === p.preparationCaptureId, 'validation.bindings');
  requireValue(v.bindings.captures.some(c => c.captureId === p.canonicalNativeCaptureId && c.runtime === 'llama-split' && c.case === 'canonical'), 'provenance.canonicalNativeCaptureId');
  const capture = async (name: Scenario): Promise<CaptureDocument> => {
    const expected = m.captures[name]; requireValue(await sha256(sources[name]) === expected.sha256, name + '.sha256');
    const d = validateDocument(parse(name), { captureId: expected.captureId, ggufSha256: m.ggufSha256 });
    requireValue(d.scenario.name === name, name + '.scenario');
    requireValue(d.numericalStatus === 'pass' && d.policySha256 === m.policySha256, name + '.approval');
    requireValue(d.tensors.length === expected.tensorCount && d.entities.filter(e => e.kind === 'operator').length === expected.operatorCount && d.tensors.filter(t => t.numel === 0).length === expected.emptyTensorCount, name + '.inventory');
    requireValue(d.kernelInternals !== undefined && d.offlineConversion !== undefined && d.recurrentState !== undefined, name + '.metadata');
    requireValue(d.recurrentState.nativeCaptureId === p.canonicalNativeCaptureId, name + '.recurrentState.nativeCaptureId');
    for (const t of d.tensors) requireValue(t.captureId === d.captureId && t.op !== undefined && t.shapeFormula?.length === 4 && t.logicalBytesFormula !== undefined && t.formulaClassification !== undefined && t.formulaSource !== undefined, `${name}.tensors.${t.id}.metadata`);
    return d;
  };
  const [prefill, decode] = await Promise.all([capture('prefill'), capture('decode')]);
  const after = prefill.recurrentState?.after, before = decode.recurrentState?.before;
  requireValue(after !== undefined && before !== undefined, 'recurrentState');
  requireValue(before.epoch > after.epoch && JSON.stringify(after.mapping) === JSON.stringify(before.mapping), 'recurrentState.continuity.mapping');
  requireValue(after.arrays.every((a, i) => a.sha256 === before.arrays[i]?.sha256 && JSON.stringify(a.summary) === JSON.stringify(before.arrays[i]?.summary)), 'recurrentState.continuity.arrays');
  return { manifest: m, model: modelData, provenance: p, validation: v, documents: { prefill, decode } };
}
