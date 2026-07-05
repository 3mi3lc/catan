// Lazily loads/caches a ServerBot per model key, mirroring hotseat.ts's
// loadBotFor fallback logic (2-player and 4-player Catan get separate nets;
// 3-player reuses the 4-player net; fall back to the 4p net if no dedicated
// 2p net has been trained/deployed yet).

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ServerBot } from '@catan/ai/net-bot';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Same model files hotseat.ts's browser bot already serves from
// packages/web/public/models — no duplication. Overridable via env var for
// deployments where the models live elsewhere on disk.
const MODEL_DIR = process.env.MODEL_DIR ?? path.resolve(__dirname, '../../web/public/models');

type ModelKey = '2p' | '4p';
const modelKeyFor = (playerCount: number): ModelKey => (playerCount === 2 ? '2p' : '4p');

const cache = new Map<ModelKey, Promise<ServerBot>>();

export function getServerBot(playerCount: number): Promise<ServerBot> {
    const want = modelKeyFor(playerCount);
    let pending = cache.get(want);
    if (!pending) {
        const primary = path.join(MODEL_DIR, `catan_net_${want}.onnx`);
        pending = ServerBot.load(primary).catch((err) => {
            if (want === '2p') return ServerBot.load(path.join(MODEL_DIR, 'catan_net_4p.onnx'));
            throw err;
        });
        cache.set(want, pending);
    }
    return pending;
}
