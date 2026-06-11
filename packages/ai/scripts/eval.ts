/// <reference types="node" />
/**
 * eval.ts — head-to-head: net policy vs greedy.
 *
 * Usage:
 *   pnpm --filter @catan/ai eval [games] [model_path] [mode]
 *   pnpm --filter @catan/ai eval 200 ../../packages/training/models/bc_v1/catan_net.onnx argmax
 */

import {
    generateBoard, rngStep,
} from '@catan/core';
import { playMatchAsync, greedyPolicy } from '../src';
import { loadNetPolicy } from '../src/net-policy';
import { buildBoardIndex } from '../src/encoding';

const games = Number(process.argv[2] ?? 200);
const model = process.argv[3] ?? '../../packages/training/models/bc_v1/catan_net.onnx';
const mode  = (process.argv[4] ?? 'argmax') as 'argmax' | 'sample';

// generateBoard requires an rng; any seed works since we only need the topology.
function mkRng(seed: number) {
    let s = (seed >>> 0) || 1;
    return () => { const r = rngStep(s); s = r.next; return r.value; };
}

const bi  = buildBoardIndex(generateBoard(mkRng(1)));
const net = await loadNetPolicy(model, bi, { mode, name: 'net' });

let netWins = 0, greedyWins = 0, unfinished = 0, totalTurns = 0;
const t0 = Date.now();

for (let g = 0; g < games; g++) {
    const netSeat = g % 2;
    const policies = netSeat === 0
        ? [net, greedyPolicy()]
        : [greedyPolicy(), net];

    // Net inference is async (onnxruntime-node has no sync run API).
    const r = await playMatchAsync({ seed: 9000 + g, policies, maxSteps: 6000 });

    totalTurns += r.turns;
    if (r.winner === null) { unfinished++; continue; }
    if (r.winnerSeat === netSeat) netWins++; else greedyWins++;

    const elapsed    = (Date.now() - t0) / 1000;
    const secPerGame = elapsed / (g + 1);
    const remaining  = secPerGame * (games - g - 1);
    const decided    = netWins + greedyWins;
    const winRate    = decided ? (netWins / decided * 100).toFixed(1) : '-';
    console.log(`[${g + 1}/${games}] ${remaining.toFixed(0)}s left — ${secPerGame.toFixed(2)}s/game — net: ${winRate}%`);
}

const secs    = ((Date.now() - t0) / 1000).toFixed(1);
const decided = netWins + greedyWins;
console.log(JSON.stringify({
    model, mode, games,
    netWins, greedyWins, unfinished,
    netWinRate: decided ? +(netWins / decided).toFixed(3) : 0,
    avgTurns: +(totalTurns / games).toFixed(1),
    seconds: +secs,
    secPerGame: +(Number(secs) / games).toFixed(2),
}, null, 2));