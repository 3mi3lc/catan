import { writeFileSync, mkdirSync } from 'fs';
import { playMatch, greedyPolicy, searchPolicy } from '../src';
import type { Policy } from '../src';
import type { GameArchive, MoveRecord } from '../src';

const games       = Number(process.argv[2] ?? 16);
const K           = Number(process.argv[3] ?? 10);
const cap         = Number(process.argv[4] ?? 1200);
const replayCount = Number(process.argv[5] ?? 3);

mkdirSync('replays', { recursive: true });

function intercepting(policy: Policy, moves: MoveRecord[]): Policy {
    return {
        name: policy.name,
        decide(state, player) {
            const action = policy.decide(state, player);
            moves.push({ player, phase: state.phase, action });
            return action;
        },
    };
}

let searchWins = 0, greedyWins = 0, unfinished = 0, totalTurns = 0;
const t0 = Date.now();

for (let g = 0; g < games; g++) {
    const searchSeat = g % 2;
    const base: Policy[] = searchSeat === 0
        ? [searchPolicy({ rollouts: K, rolloutCap: cap, seed: 1000 + g }), greedyPolicy()]
        : [greedyPolicy(), searchPolicy({ rollouts: K, rolloutCap: cap, seed: 1000 + g })];

    const moves: MoveRecord[] = [];
    const policies = g < replayCount
        ? base.map(p => intercepting(p, moves))
        : base;

    const r = playMatch({ seed: 7000 + g, policies, maxSteps: 6000 });

    totalTurns += r.turns;
    if (r.winner === null) { unfinished++; continue; }
    if (r.winnerSeat === searchSeat) searchWins++; else greedyWins++;

    if (g < replayCount) {
        const label = r.winnerSeat === searchSeat ? 'search-win' : 'greedy-win';
        const archive: GameArchive = {
            seed: 7000 + g,
            policyNames: base.map(p => p.name),
            winnerSeat: r.winnerSeat,
            turns: r.turns,
            moves,
        };
        const path = `replays/game-${g}-${label}.json`;
        writeFileSync(path, JSON.stringify(archive));
        console.log(`Saved ${path} (${moves.length} moves, ~${(JSON.stringify(archive).length / 1024).toFixed(1)} KB)`);
    }

    const elapsed    = (Date.now() - t0) / 1000;
    const secPerGame = elapsed / (g + 1);
    const remaining  = secPerGame * (games - g - 1);
    const decided    = searchWins + greedyWins;
    const winRate    = decided ? (searchWins / decided * 100).toFixed(1) : '-';
    console.log(`[${g + 1}/${games}] ${remaining.toFixed(0)}s left — ${secPerGame.toFixed(1)}s/game — search: ${winRate}%`);
}

const secs    = ((Date.now() - t0) / 1000).toFixed(1);
const decided = searchWins + greedyWins;
console.log(JSON.stringify({
    games, K, cap,
    searchWins, greedyWins, unfinished,
    searchWinRate: decided ? +(searchWins / decided).toFixed(3) : 0,
    avgTurns: +(totalTurns / games).toFixed(1),
    seconds: +secs,
    secPerGame: +(Number(secs) / games).toFixed(2),
}, null, 2));