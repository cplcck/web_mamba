import { cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { checkPublish } from '../tools/check_publish';

describe('publication boundary', () => {
  let root: string;
  let output: string;
  let trusted: string;
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), 'mamba-publication-'));
    output = join(root, 'dist');
    trusted = join(root, 'trusted');
    await cp(resolve('public/data'), trusted, { recursive: true });
    await cp(trusted, join(output, 'data'), { recursive: true });
    await mkdir(join(output, 'assets'));
    await writeFile(join(output, 'index.html'), '<script type="module" src="/web_mamba/assets/app.js"></script>');
    await writeFile(join(output, 'assets/app.js'), 'export {};');
  });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });

  it('accepts the actual six-document release with static assets', async () => {
    // Given the isolated actual-data output, when checked, then it is publishable.
    await expect(checkPublish(output, trusted)).resolves.toBeUndefined();
  });

  it('accepts SVG namespaces and ordinary HTTP URLs in JavaScript assets', async () => {
    // Given executable JS containing the SVG namespace and ordinary web URLs.
    await writeFile(join(output, 'assets/app.js'), [
      'document.createElementNS("http://www.w3.org/2000/svg", "svg");',
      'export const urls = ["https://example.org/assets/app.js", "http://localhost:4173/web_mamba/"];',
    ].join('\n'));
    // When checked, then URL scheme suffixes are not mistaken for drive paths.
    await expect(checkPublish(output, trusted)).resolves.toBeUndefined();
  });

  it.each([
    'model.gguf', 'model.safetensors', 'weights.bin', 'capture.npz',
    'logits.npy', 'cache.pt', '.env', 'assets/app.js.map',
    'data/raw.json', '.artifacts/capture.json', '.git/config',
  ])('rejects forbidden artifact %s', async path => {
    // Given a forbidden artifact in an otherwise valid temporary output.
    const absolute = join(output, path);
    await mkdir(join(absolute, '..'), { recursive: true });
    await writeFile(absolute, 'private');
    // When checked, then publication fails.
    await expect(checkPublish(output, trusted)).rejects.toThrow();
  });

  it.each([
    '/home/private/model', '/tmp/capture', 'C:\\private\\model', 'C:/private/model',
    '.artifacts/captures', '-----BEGIN PRIVATE KEY-----',
    `ghp_${'x'.repeat(36)}`, '"access_token":"private-value"',
  ])('rejects private content in an allowed asset: %s', async content => {
    // Given private content concealed in an allowed filename.
    await writeFile(join(output, 'assets/app.js'), JSON.stringify(content));
    // When checked, then publication fails.
    await expect(checkPublish(output, trusted)).rejects.toMatchObject({
      path: 'assets/app.js', reason: 'private or binary content',
    });
  });

  it('rejects a direct secret property in executable JavaScript', async () => {
    // Given an actual object property rather than a quoted JSON string.
    await writeFile(join(output, 'assets/app.js'),
      'export const config = {"access_token":"private-value"};');
    // When checked, then the secret-content guard rejects the allowed asset.
    await expect(checkPublish(output, trusted)).rejects.toMatchObject({
      path: 'assets/app.js', reason: 'private or binary content',
    });
  });

  it.each(['manifest', 'model', 'provenance', 'validation', 'prefill', 'decode'])(
    'rejects changed %s bytes against trusted source', async name => {
      // Given valid JSON with changed bytes in one published document.
      const file = join(output, 'data', `${name}.json`);
      await writeFile(file, `${await readFile(file, 'utf8')}\n`);
      // When checked, then even whitespace-only drift is rejected.
      await expect(checkPublish(output, trusted)).rejects.toMatchObject({
        path: `data/${name}.json`, reason: 'stale source bytes',
      });
    },
  );

  it('rejects unapproved status even when candidate and source bytes agree', async () => {
    // Given mutually matching documents whose release status is not approved.
    const original = await readFile(join(trusted, 'validation.json'), 'utf8');
    const changed = original.replace('"status":"pass"', '"status":"unvalidated"');
    expect(changed).not.toBe(original);
    await writeFile(join(trusted, 'validation.json'), changed);
    await writeFile(join(output, 'data/validation.json'), changed);
    // When checked, then the shared bundle validator rejects the status.
    await expect(checkPublish(output, trusted)).rejects.toMatchObject({ path: 'validation.status' });
  });

  it('rejects absent required JSON', async () => {
    // Given a missing scenario document.
    await rm(join(output, 'data/decode.json'));
    // When checked, then no incomplete bundle can publish.
    await expect(checkPublish(output, trusted)).rejects.toMatchObject({
      path: 'data/decode.json', reason: 'missing artifact',
    });
  });

  it('rejects symlinks rather than following private targets', async () => {
    // Given an allowed filename pointing outside the publication tree.
    await symlink(join(trusted, 'model.json'), join(output, 'assets/private.js'));
    // When checked, then the link is rejected.
    await expect(checkPublish(output, trusted)).rejects.toMatchObject({ reason: 'symlink' });
  });
});
