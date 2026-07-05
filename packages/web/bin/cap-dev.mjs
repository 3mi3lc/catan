// Live-reload dev loop: builds+installs the debug APK once (via `cap run
// android`), then points its WebView at the *running* Vite dev server
// (localhost:5173, reached over USB via `adb reverse` or over LAN) instead
// of the bundled dist-mobile files — so React/CSS edits hot-reload on-device
// with no further rebuild/reinstall. See capacitor.config.ts's
// CAP_LIVE_RELOAD_URL handling.
import { spawnSync } from 'node:child_process';

const port = process.env.VITE_PORT ?? '5173';
// CAP_LIVE_RELOAD_URL (if set) is just the origin — e.g.
// "http://192.168.1.20:5173" — not a full page URL. Vite's dev server
// serves `index.html` (the board viewer) for the bare origin, which isn't
// what the mobile shell should show; always append the actual mobile entry
// point explicitly rather than relying on whoever sets the env var to
// remember the path.
const origin = (process.env.CAP_LIVE_RELOAD_URL ?? `http://localhost:${port}`).replace(/\/+$/, '');
const url = `${origin}/online.html`;

const result = spawnSync('npx', ['cap', 'run', 'android'], {
    stdio: 'inherit',
    shell: true,
    env: { ...process.env, CAP_LIVE_RELOAD_URL: url },
});
process.exit(result.status ?? 1);
