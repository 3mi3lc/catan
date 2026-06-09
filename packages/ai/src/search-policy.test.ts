import { describe, it, expect } from 'vitest';
import { victoryPoints } from '@catan/core';
import { searchPolicy } from './search-policy';
import { greedyPolicy } from './greedy-policy';
import { playMatch } from './runner';

// These stay deliberately tiny — strength is measured separately via
// scripts/duel.ts (search beats greedy ~78% head-to-head). Here we only
// guarantee the policy is *sound* (plays legal games to a real win) and
// *reproducible* (a fresh seeded policy replays identically).

describe('searchPolicy — soundness', () => {
    it('plays a legal 2p game to a legitimate win', () => {
        const r = playMatch({
            seed: 4242,
            policies: [searchPolicy({ rollouts: 4, rolloutCap: 1000, seed: 1 }), greedyPolicy()],
            maxSteps: 6000,
        });
        expect(r.winner).not.toBeNull();
        expect(victoryPoints(r.finalState, r.winner!)).toBeGreaterThanOrEqual(10);
    }, 60_000);
});

describe('searchPolicy — determinism', () => {
    it('fresh seeded policies reproduce the same game', () => {
        const run = () =>
            playMatch({
                seed: 4242,
                policies: [searchPolicy({ rollouts: 4, rolloutCap: 1000, seed: 1 }), greedyPolicy()],
                maxSteps: 6000,
            });
        const a = run();
        const b = run();
        expect(a.winner).toBe(b.winner);
        expect(a.winnerSeat).toBe(b.winnerSeat);
        expect(a.turns).toBe(b.turns);
        expect(a.steps).toBe(b.steps);
    }, 90_000);
});