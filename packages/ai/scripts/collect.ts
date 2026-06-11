/// <reference types="node" />
/**
 * collect.ts — self-play data collection for RL/BC training.
 *
 * Runs greedy vs. greedy games and writes binary shard files that Python
 * can read directly as numpy arrays.
 *
 * Usage:
 *   pnpm --filter @catan/ai collect [games] [out_dir] [seed_base]
 *   pnpm --filter @catan/ai collect 10000 data/greedy 1
 *
 * Binary shard format (.bin):
 *   bytes  0-3   : uint32 magic = 0x4E415443  ('CTAN' little-endian)
 *   bytes  4-7   : uint32 num_samples
 *   bytes  8-11  : uint32 obs_size  (1328)
 *   bytes 12-15  : uint32 act_size  (300)
 *   bytes 16 onward, three contiguous blocks:
 *     float32[num_samples × obs_size]  — observations
 *     uint16[num_samples]              — action indices
 *     uint8[num_samples]               — outcomes (1 = winner, 0 = loser)
 *
 * Python reads these with numpy.frombuffer; see packages/training/dataset.py.
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import {
    generateBoard, initialGameState, applyMove, rngStep,
    type GameState, type PlayerId, type PlayerColor,
} from '@catan/core';
import {
    buildBoardIndex, encodeObservation, actionToIndex,
    OBS_SIZE, ACT_SIZE, type BoardIndex,
} from '../src/encoding';
import { discardAction } from '../src/heuristics';
import { greedyPolicy } from '../src/greedy-policy';
import type { Policy } from '../src/policy';

// ── CLI args ──────────────────────────────────────────────────────────────────
const GAMES      = Number(process.argv[2] ?? 5_000);
const OUT_DIR    = process.argv[3] ?? 'data/greedy';
const SEED_BASE  = Number(process.argv[4] ?? 1);
const SHARD_SIZE = 10_000; // samples per file (~53 MB each)

// ── Constants ─────────────────────────────────────────────────────────────────
const COLORS: PlayerColor[] = ['red', 'blue', 'white', 'orange'];
const MAGIC = 0x4E415443;

function mkRng(seed: number) {
    let s = (seed >>> 0) || 1;
    return () => { const r = rngStep(s); s = r.next; return r.value; };
}

// ── Per-game collector ────────────────────────────────────────────────────────

interface RawStep { obs: Float32Array; action: number; player: PlayerId; }

/**
 * Runs one complete game and returns (rawSteps, winner).
 * Discard steps are handled by heuristic and never captured — the net
 * doesn't learn to discard; it uses discardAction at inference time.
 */
function collectGame(
    seed: number,
    policies: Policy[],
    bi: BoardIndex,
): { steps: RawStep[]; winner: PlayerId | null } {
    const n = policies.length;
    const board = generateBoard(mkRng(seed));
    const seats = COLORS.slice(0, n).map((c, i) => ({ id: `p${i}` as PlayerId, name: c, color: c }));
    let state = initialGameState(board, seats, seed);

    const steps: RawStep[] = [];

    for (let t = 0; t < 6000 && !state.winner; t++) {
        const actor: PlayerId = state.phase === 'discard'
            ? (Object.keys(state.pendingDiscards)[0] as PlayerId)
            : state.currentPlayer;

        let action;

        if (state.phase === 'discard') {
            // Heuristic only — never capture
            action = discardAction(state, actor);
        } else {
            const obs     = encodeObservation(state, actor, bi);
            const seat    = state.turnOrder.indexOf(actor);
            action        = policies[seat % n].decide(state, actor);
            const actIdx  = actionToIndex(action, state, actor, bi);
            steps.push({ obs, action: actIdx, player: actor });
        }

        const res = applyMove(state, { player: actor, action });
        if (!res.ok) break;
        state = res.state;
    }

    return { steps, winner: state.winner };
}

// ── Shard writer ──────────────────────────────────────────────────────────────

/**
 * Writes N samples to a binary shard file.
 * Layout: 16-byte header, then obs block, actions block, outcomes block.
 */
function writeShard(
    path: string,
    obsBlock: Float32Array,
    actBlock: Uint16Array,
    outBlock: Uint8Array,
    n: number,
): void {
    const header = Buffer.alloc(16);
    header.writeUInt32LE(MAGIC,    0);
    header.writeUInt32LE(n,        4);
    header.writeUInt32LE(OBS_SIZE, 8);
    header.writeUInt32LE(ACT_SIZE, 12);

    writeFileSync(path, Buffer.concat([
        header,
        Buffer.from(obsBlock.buffer, 0, n * OBS_SIZE * 4),
        Buffer.from(actBlock.buffer, 0, n * 2),
        Buffer.from(outBlock.buffer, 0, n),
    ]));
}

// ── Main ──────────────────────────────────────────────────────────────────────

mkdirSync(OUT_DIR, { recursive: true });

// Build the board index once — the radius-2 hex topology (54 vertices, 72 edges,
// 19 tiles) is identical across all boards of the same size. Terrain and number
// tokens vary per seed, but the position-derived IDs do not.
const firstBoard = generateBoard(mkRng(SEED_BASE));
const BI         = buildBoardIndex(firstBoard);

const policies: Policy[] = [greedyPolicy(), greedyPolicy()];

// Pre-allocate shard buffers (reused across flushes).
const shardObs = new Float32Array(SHARD_SIZE * OBS_SIZE);
const shardAct = new Uint16Array(SHARD_SIZE);
const shardOut = new Uint8Array(SHARD_SIZE);

let shardCount  = 0;  // samples in current shard
let shardIdx    = 0;  // shard file index
let totalSamples = 0;
let totalGames   = 0;
let skipped      = 0; // games that hit the step cap
let p0wins = 0, p1wins = 0;

function flushShard() {
    const path = join(OUT_DIR, `shard_${String(shardIdx).padStart(5, '0')}.bin`);
    writeShard(path, shardObs, shardAct, shardOut, shardCount);
    console.log(`  shard ${shardIdx}: ${shardCount.toLocaleString()} samples → ${path}`);
    shardIdx++;
    shardCount = 0;
}

const t0 = Date.now();

for (let g = 0; g < GAMES; g++) {
    const seed = SEED_BASE + g;
    const { steps, winner } = collectGame(seed, policies, BI);

    if (!winner) { skipped++; continue; }
    if (winner === 'p0') p0wins++; else p1wins++;

    // Pack steps into shard buffers, flushing when full.
    for (const s of steps) {
        const outcome: 0 | 1 = s.player === winner ? 1 : 0;
        const i = shardCount;

        shardObs.set(s.obs, i * OBS_SIZE);
        shardAct[i] = s.action;
        shardOut[i] = outcome;
        shardCount++;

        if (shardCount >= SHARD_SIZE) flushShard();
    }

    totalSamples += steps.length;
    totalGames++;

    if ((g + 1) % 1000 === 0 || g === GAMES - 1) {
        const sec = ((Date.now() - t0) / 1000).toFixed(1);
        const bias = totalGames > 0 ? ((p0wins / totalGames) * 100).toFixed(1) : '?';
        console.log(
            `[${(g + 1).toLocaleString()}/${GAMES.toLocaleString()}] ` +
            `games=${totalGames.toLocaleString()} samples=${totalSamples.toLocaleString()} ` +
            `skip=${skipped} p0_win%=${bias} ${sec}s`,
        );
    }
}

// Flush final partial shard.
if (shardCount > 0) flushShard();

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
const samplesPerSec = (totalSamples / Number(elapsed)).toFixed(0);

console.log([
    '',
    '── Collection complete ──────────────────────────────',
    `  games:        ${totalGames.toLocaleString()} (${skipped} skipped)`,
    `  samples:      ${totalSamples.toLocaleString()} (${samplesPerSec}/s)`,
    `  shards:       ${shardIdx}  (${SHARD_SIZE.toLocaleString()} samples each, ~${(SHARD_SIZE * (OBS_SIZE * 4 + 3) / 1e6).toFixed(0)} MB)`,
    `  obs_size:     ${OBS_SIZE}`,
    `  act_size:     ${ACT_SIZE}`,
    `  seat balance: p0=${p0wins} p1=${p1wins} (should be ~50/50)`,
    `  output:       ${OUT_DIR}/`,
    '',
    '  Next step:',
    `    cd packages/training`,
    `    pip install -r requirements.txt`,
    `    python train_bc.py --data-dir ../../${OUT_DIR}`,
    '────────────────────────────────────────────────────',
].join('\n'));