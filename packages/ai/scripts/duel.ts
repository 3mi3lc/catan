import { writeFileSync, mkdirSync } from 'fs';
import { playMatch, greedyPolicy, searchPolicy } from '../src';
import type { GameState } from '@catan/core';

// args: games K cap replays
const games = Number(process.argv[2] ?? 16);
const K = Number(process.argv[3] ?? 10);
const cap = Number(process.argv[4] ?? 1200);
const replayCount = Number(process.argv[5] ?? 3); // how many games to save

mkdirSync('replays', { recursive: true });

let searchWins = 0, greedyWins = 0, unfinished = 0, turns = 0;
const t0 = Date.now();

for (let g = 0; g < games; g++) {
    const searchSeat = g % 2;
    const policies = searchSeat === 0
        ? [searchPolicy({ rollouts: K, rolloutCap: cap, seed: 1000 + g }), greedyPolicy()]
        : [greedyPolicy(), searchPolicy({ rollouts: K, rolloutCap: cap, seed: 1000 + g })];

    const saveReplay = g < replayCount;
    const history: GameState[] = [];

    const r = playMatch({
        seed: 7000 + g, policies, maxSteps: 6000,
        onStep: saveReplay ? (state) => history.push(state) : undefined,
    });

    if (saveReplay && r.winner !== null) {
        const label = r.winnerSeat === searchSeat ? 'search-win' : 'greedy-win';
        writeFileSync(`replays/game-${g}-${label}.json`, JSON.stringify(history));
        console.log(`Saved replays/game-${g}-${label}.json (${history.length} steps)`);
    }

    turns += r.turns;
    if (r.winner === null) { unfinished++; continue; }
    if (r.winnerSeat === searchSeat) searchWins++; else greedyWins++;

    const elapsed = (Date.now() - t0) / 1000;
    const secPerGame = elapsed / (g + 1);
    const remaining = secPerGame * (games - g - 1);
    const decided = searchWins + greedyWins;
    const winRate = decided ? (searchWins / decided * 100).toFixed(1) : '-';
    console.log(`[${g + 1}/${games}] ${remaining.toFixed(0)}s remaining — ${secPerGame.toFixed(1)}s/game — search win rate so far: ${winRate}%`);
}

const secs = ((Date.now() - t0) / 1000).toFixed(1);
const decided = searchWins + greedyWins;
console.log(JSON.stringify({
    games, K, cap,
    searchWins, greedyWins, unfinished,
    searchWinRate: decided ? +(searchWins / decided).toFixed(3) : 0,
    avgTurns: +(turns / games).toFixed(1),
    seconds: +secs,
    secPerGame: +(Number(secs) / games).toFixed(2),
}, null, 2));