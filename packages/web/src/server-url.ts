// The API/Socket.io server's origin — one definition shared by online.ts,
// auth-client.ts, and replay/ReplayApp.tsx (all three used to duplicate this
// independently; consolidated since the fallback logic below is no longer a
// one-liner).
//
// In dev (`vite dev`), the server runs separately on :8080 while Vite serves
// the web app on :5173 — different origins, so that's the default there.
// In a production build, the server serves the built web app itself (see
// packages/server/src/static.ts) — same origin as the page — unless
// VITE_SERVER_URL was set at build time to point somewhere else (e.g. a
// split web/server cloud deploy, web on one host and the server on another).
const env = (import.meta as { env?: Record<string, string> & { DEV?: boolean } }).env;
export const SERVER_URL = env?.VITE_SERVER_URL ?? (env?.DEV ? 'http://localhost:8080' : window.location.origin);
