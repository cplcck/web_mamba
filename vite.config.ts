import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { checkPublish } from './tools/check_publish';

export default defineConfig({
  base: '/',
  build: { emptyOutDir: true },
  plugins: [{
    name: 'publication-boundary',
    apply: 'build',
    async writeBundle(options) {
      await checkPublish(resolve(options.dir ?? 'dist'), resolve('public/data'));
    },
  }],
});
