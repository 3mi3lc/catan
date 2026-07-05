// Serves the built web app (packages/web/dist by default) directly from this
// process, alongside the Socket.io/API routes in index.ts. Exists for the
// local/"tunnel it and share a link with friends" use case: one process, one
// port, one URL — no second static host, no cross-origin cookies/CORS to
// juggle for the pages themselves (the /api/auth and /games routes still go
// through index.ts's applyCors, unchanged, since Socket.io's own CORS check
// and a browser's Origin header still apply regardless of same-origin
// serving of the HTML/JS).

import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { IncomingMessage, ServerResponse } from 'node:http';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIST = process.env.WEB_DIST ?? path.resolve(__dirname, '../../web/dist');

const MIME: Record<string, string> = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.mp3': 'audio/mpeg',
    '.onnx': 'application/octet-stream',
    '.json': 'application/json',
    '.woff2': 'font/woff2',
    '.ico': 'image/x-icon',
};

// The app is several standalone HTML pages, not one SPA — there's no single
// natural "index" for `/`. Default to the multiplayer client: that's what a
// friend opening a shared link actually wants.
const DEFAULT_PATH = '/online.html';

/** Always fully handles GET/HEAD requests (serves a file, or sends a
 *  definitive 403/404/500) — returns false only for other HTTP methods, so
 *  index.ts's own fallback applies to those instead. */
export async function serveStatic(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
    if (req.method !== 'GET' && req.method !== 'HEAD') return false;

    const urlPath = (req.url ?? '/').split('?')[0];
    const relative = urlPath === '/' ? DEFAULT_PATH : urlPath;

    let decoded: string;
    try { decoded = decodeURIComponent(relative); } catch { res.writeHead(400); res.end(); return true; }

    // Resolve-then-prefix-check is the standard defense against `..`-based
    // path traversal escaping WEB_DIST (decoding first so an encoded
    // "%2e%2e" can't slip past a naive string check).
    const resolved = path.resolve(WEB_DIST, '.' + decoded);
    if (!resolved.startsWith(WEB_DIST)) { res.writeHead(403); res.end(); return true; }

    let size: number;
    try {
        const stats = await stat(resolved);
        if (!stats.isFile()) throw new Error('not a file');
        size = stats.size;
    } catch {
        const distExists = await stat(WEB_DIST).then(() => true).catch(() => false);
        res.writeHead(distExists ? 404 : 500, { 'content-type': 'text/plain' });
        res.end(distExists ? 'Not found' : `No web build found at ${WEB_DIST} — run "pnpm --filter @catan/web build" first.`);
        return true;
    }

    res.writeHead(200, { 'content-type': MIME[path.extname(resolved).toLowerCase()] ?? 'application/octet-stream', 'content-length': size });
    if (req.method === 'HEAD') { res.end(); return true; }
    createReadStream(resolved).pipe(res);
    return true;
}
