import { lstat, readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { validatePublicBundle, type PublicBundleSources } from '../src/public-data';

const names = ['manifest', 'model', 'provenance', 'validation', 'prefill', 'decode'] as const;
const dataPaths = new Set(names.map(name => `data/${name}.json`));
const privateContent = /\/(?:home|Users|tmp|root|var\/(?:tmp|lib)|mnt)\/|(?:^|[^A-Za-z0-9_])[A-Za-z]:[\\/]|(?:\.artifacts|\.ssh|\.aws|\.omo)[\\/]|-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})\b|["'](?:api[_-]?key|access[_-]?token|client[_-]?secret|password)["']\s*:\s*["'][^"']+["']/i;

export class PublicationError extends Error {
  constructor(readonly path: string, readonly reason: string) {
    super(`${path}: ${reason}`);
    this.name = 'PublicationError';
  }
}

/** Checkout inputs are the release authority; candidate output cannot approve itself. */
export async function checkPublish(output: string, sourceData: string): Promise<void> {
  const seen = new Set<string>();
  const visit = async (relative: string): Promise<void> => {
    const absolute = join(output, relative);
    const stat = await lstat(absolute);
    if (stat.isSymbolicLink()) throw new PublicationError(relative, 'symlink');
    if (stat.isDirectory()) {
      if (relative !== '' && relative !== 'assets' && relative !== 'data') {
        throw new PublicationError(relative, 'forbidden directory');
      }
      for (const entry of await readdir(absolute)) {
        await visit(relative === '' ? entry : `${relative}/${entry}`);
      }
      return;
    }
    if (!stat.isFile() || !(relative === 'index.html' ||
      /^assets\/[A-Za-z0-9_-]+\.(?:js|css)$/.test(relative) || dataPaths.has(relative))) {
      throw new PublicationError(relative, 'forbidden artifact');
    }
    const content = await readFile(absolute, 'utf8');
    // Inspect quoted strings embedded in JS as well as direct JSON properties.
    const unescaped = content.replace(/\\(["'\\/])/g, '$1');
    if (content.includes('\0') || privateContent.test(unescaped)) {
      throw new PublicationError(relative, 'private or binary content');
    }
    seen.add(relative);
  };
  await visit('');
  for (const required of ['index.html', ...dataPaths]) {
    if (!seen.has(required)) throw new PublicationError(required, 'missing artifact');
  }
  const load = async (name: keyof PublicBundleSources): Promise<string> => {
    const candidate = await readFile(join(output, 'data', `${name}.json`));
    const trusted = await readFile(join(sourceData, `${name}.json`));
    if (!candidate.equals(trusted)) throw new PublicationError(`data/${name}.json`, 'stale source bytes');
    return candidate.toString('utf8');
  };
  const [manifest, model, provenance, validation, prefill, decode] = await Promise.all(names.map(load));
  if (manifest === undefined || model === undefined || provenance === undefined ||
    validation === undefined || prefill === undefined || decode === undefined) {
    throw new PublicationError('data', 'incomplete bundle');
  }
  await validatePublicBundle({ manifest, model, provenance, validation, prefill, decode });
}
