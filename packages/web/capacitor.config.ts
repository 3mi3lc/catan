import type { CapacitorConfig } from '@capacitor/cli';

// appId/appName are placeholders — change these before actually publishing
// anywhere (appId in particular can't be changed after a first Play
// Store/App Store submission without shipping as a new app).
//
// CAP_LIVE_RELOAD_URL (set by `pnpm cap:dev`) points the WebView straight at
// the running Vite dev server instead of the bundled dist-mobile files, so
// React/CSS edits hot-reload on-device with no rebuild/reinstall. Only take
// this branch when it's explicitly set — a normal `cap sync`/release build
// must never end up pointed at a dev server.
const liveReloadUrl = process.env.CAP_LIVE_RELOAD_URL;

const config: CapacitorConfig = {
    appId: 'com.catan.online',
    appName: 'Catan Online',
    // The bundled build the native shell actually ships — see
    // vite.mobile.config.ts + bin/rename-mobile-entry.mjs (`pnpm build:mobile`
    // produces this). Not the regular multi-page `dist/` from `pnpm build`,
    // which still needs index.html to be the board-viewer for the web deploy.
    webDir: 'dist-mobile',
    // Android blocks plain-HTTP (cleartext) network calls by default from
    // API 28+. The tunnel-based "play with friends" workflow (see
    // packages/server/README.md) is already HTTPS, so this doesn't affect
    // that — it only matters if you point VITE_SERVER_URL at a bare local
    // dev server (http://<lan-ip>:8080) while iterating; Socket.io/fetch
    // calls to that will be silently blocked until you either put a TLS
    // tunnel in front of it too, or explicitly opt into cleartext for that
    // one case (android/app/src/main/res/xml/network_security_config.xml —
    // not set up here, since loosening it project-wide wasn't asked for).
    server: liveReloadUrl
        ? { url: liveReloadUrl, cleartext: true }
        : { androidScheme: 'https' },
};

export default config;
