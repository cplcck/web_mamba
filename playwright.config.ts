/// <reference types="node" />
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 240_000,
  expect: { timeout: 15_000 },
  outputDir: 'test-results/e2e',
  reporter: [['list'], ['json', { outputFile: 'test-results/e2e-report.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173/web_mamba/',
    channel: 'chrome',
    viewport: { width: 390, height: 844 },
    reducedMotion: 'reduce',
    launchOptions: { args: ['--disable-ipc-flooding-protection'] },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'npm run preview -- --host 127.0.0.1 --port 4173 --strictPort --base=/web_mamba/',
    url: 'http://127.0.0.1:4173/web_mamba/',
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
