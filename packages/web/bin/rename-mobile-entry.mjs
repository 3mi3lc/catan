// Capacitor expects `index.html` at the webDir root — Vite's multi-page
// build names the output HTML file after its input filename, and there's no
// built-in way to rename just that at build time, so this runs as a small
// postbuild step for `pnpm build:mobile` instead. Idempotent: safe to run
// against a dist-mobile that's already been renamed (no-ops if mobile.html
// isn't there).
import { existsSync, renameSync } from 'node:fs';
import { join } from 'node:path';

const dir = join(import.meta.dirname, '..', 'dist-mobile');
const from = join(dir, 'mobile.html');
const to = join(dir, 'index.html');

if (existsSync(from)) {
    renameSync(from, to);
    console.log('dist-mobile/mobile.html -> index.html');
} else if (!existsSync(to)) {
    throw new Error(`Expected ${from} to exist after the mobile build — did the vite.mobile.config.ts input change?`);
}
