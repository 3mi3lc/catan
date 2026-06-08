/// <reference lib="dom" />
import { describe, it, expect } from 'vitest';
import { asPlayerId, victoryPoints } from '@catan/core';
import { greedyPolicy } from './greedy-policy';
import { randomPolicy } from './random-policy';
import { playMatch, playTournament } from './runner';
import { Policy } from './policy';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

interface MatchupStats {
  games: number;
  hitCap: number;
  trackedWins: number;
  trackedWinRate: number;
  winsBySeat: number[];
  winRateBySeat: number[];
  avgTurns: number;
  avgSteps: number;
  stepsPerTurn: number;
  avgWinnerVp: number;
  avgVpMargin: number;
  vpSourcesPerGame: { buildings: number; devCards: number; longestRoad: number; largestArmy: number };
  longestRoadWinRate: number;
  largestArmyWinRate: number;
  avgUnplayedDevCards: number;
  avgEndHandSize: number;
  turnDistribution: Record<string, number>;
  longGameSeeds: number[];
}

function runMatchup(
    games: number,
    makePolicies: (g: number) => Policy[],
    baseSeed = 10000,
    trackedSeat: ((g: number) => number) | null = null,
): MatchupStats {
  const MAX_SEATS = 4;
  const wins = new Array<number>(MAX_SEATS).fill(0);
  let totalTurns = 0;
  let totalSteps = 0;
  let totalVpMargin = 0;
  let totalWinnerVp = 0;
  let hitCap = 0;
  let trackedWins = 0;

  const vpSources = { buildings: 0, devCards: 0, longestRoad: 0, largestArmy: 0 };
  let longestRoadGames = 0;
  let largestArmyGames = 0;
  let totalUnplayedDevCards = 0;
  let totalEndHandSize = 0;

  const turnBuckets: Record<string, number> = {
    under100: 0, u150: 0, u200: 0, u300: 0, over300: 0,
  };
  const longGameSeeds: number[] = [];

  for (let g = 0; g < games; g++) {
    const ps = makePolicies(g);
    const n = ps.length;
    const seed = baseSeed + g;

    const r = playMatch({ seed, policies: ps });

    totalTurns += r.turns;
    totalSteps += r.steps;
    if (r.turns > 300) longGameSeeds.push(seed);

    const t = r.turns;
    if      (t < 100) turnBuckets.under100++;
    else if (t < 150) turnBuckets.u150++;
    else if (t < 200) turnBuckets.u200++;
    else if (t < 300) turnBuckets.u300++;
    else              turnBuckets.over300++;

    if (r.winner === null) {
      hitCap++;
      continue;
    }

    wins[r.winnerSeat!]++;

    // Track whether the designated seat won this game
    if (trackedSeat !== null && r.winnerSeat === trackedSeat(g)) {
      trackedWins++;
    }

    const winnerPid = asPlayerId(`p${r.winnerSeat}`);
    const winnerVp  = victoryPoints(r.finalState, winnerPid);
    totalWinnerVp  += winnerVp;

    // VP margin — winner vs best loser (works for any player count)
    let bestLoserVp = 0;
    for (let s = 0; s < n; s++) {
      if (s === r.winnerSeat) continue;
      bestLoserVp = Math.max(bestLoserVp, victoryPoints(r.finalState, asPlayerId(`p${s}`)));
    }
    totalVpMargin += winnerVp - bestLoserVp;

    // VP source breakdown
    let buildingVp = 0;
    for (const b of Object.values(r.finalState.buildings)) {
      if (b.owner !== winnerPid) continue;
      buildingVp += b.kind === 'city' ? 2 : 1;
    }
    const pl    = r.finalState.players[winnerPid];
    const devVp = [...pl.devCards, ...pl.pendingDevCards].filter(c => c === 'victoryPoint').length;
    const lrVp  = r.finalState.longestRoad?.player === winnerPid ? 2 : 0;
    const laVp  = r.finalState.largestArmy?.player === winnerPid ? 2 : 0;
    vpSources.buildings   += buildingVp;
    vpSources.devCards    += devVp;
    vpSources.longestRoad += lrVp;
    vpSources.largestArmy += laVp;
    if (lrVp) longestRoadGames++;
    if (laVp) largestArmyGames++;

    // Tracked-seat diagnostics
    const seat    = trackedSeat?.(g) ?? 0;
    const tracked = r.finalState.players[asPlayerId(`p${seat}`)];
    totalUnplayedDevCards += tracked.devCards.length;
    totalEndHandSize      += Object.values(tracked.resources).reduce((a, b) => a + b, 0);
  }

  const completed = games - hitCap;
  const nSeats    = makePolicies(0).length;

  return {
    games,
    hitCap,
    trackedWins,
    trackedWinRate: trackedWins / games,
    winsBySeat:     wins.slice(0, nSeats),
    winRateBySeat:  wins.slice(0, nSeats).map(w => w / games),
    avgTurns:       totalTurns / games,
    avgSteps:       totalSteps / games,
    stepsPerTurn:   totalSteps / totalTurns,
    avgWinnerVp:    totalWinnerVp / completed,
    avgVpMargin:    totalVpMargin / completed,
    vpSourcesPerGame: {
      buildings:   vpSources.buildings   / completed,
      devCards:    vpSources.devCards    / completed,
      longestRoad: vpSources.longestRoad / completed,
      largestArmy: vpSources.largestArmy / completed,
    },
    longestRoadWinRate:  longestRoadGames / completed,
    largestArmyWinRate:  largestArmyGames / completed,
    avgUnplayedDevCards: totalUnplayedDevCards / games,
    avgEndHandSize:      totalEndHandSize / games,
    turnDistribution:    turnBuckets,
    longGameSeeds:       longGameSeeds.slice(0, 5),
  };
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

describe('runner — games complete', () => {
  it('four greedy bots finish with a legitimate winner', () => {
    for (const seed of [1, 7, 42, 99, 2024]) {
      const r = playMatch({
        seed,
        policies: [greedyPolicy(), greedyPolicy(), greedyPolicy(), greedyPolicy()],
      });
      expect(r.winner, `seed ${seed} did not finish`).not.toBeNull();
      expect(victoryPoints(r.finalState, r.winner!)).toBeGreaterThanOrEqual(10);
    }
  });

  it('random bots also finish (with a generous cap)', () => {
    const r = playMatch({
      seed: 5,
      policies: [randomPolicy(1), randomPolicy(2), randomPolicy(3)],
      maxSteps: 8000,
    });
    expect(r.winner).not.toBeNull();
  });
});

describe('runner — determinism', () => {
  it('same seed and policies → identical result', () => {
    const run = () =>
        playMatch({
          seed: 42,
          policies: [greedyPolicy(), greedyPolicy(), greedyPolicy(), greedyPolicy()],
        });
    const a = run(), b = run();
    expect(a.winner).toBe(b.winner);
    expect(a.turns).toBe(b.turns);
    expect(a.steps).toBe(b.steps);
  });
});

describe('greedy vs random', () => {
  it('greedy beats random handily in 2-player games (both seat orders)', () => {
    let greedyWins = 0;
    const games = 30;
    for (let g = 0; g < games; g++) {
      const policies =
          g % 2 === 0
              ? [greedyPolicy(), randomPolicy(g + 1)]
              : [randomPolicy(g + 1), greedyPolicy()];
      const r = playMatch({ seed: 1000 + g, policies });
      if (r.winner === null) continue;
      const greedySeat = g % 2 === 0 ? 0 : 1;
      if (r.winnerSeat === greedySeat) greedyWins++;
    }
    expect(greedyWins).toBeGreaterThanOrEqual(Math.ceil(games * 0.75));
  });

  it('greedy outperforms random in a rotating 4-player tournament', () => {
    const result = playTournament(
        [greedyPolicy(), randomPolicy(11), randomPolicy(22), randomPolicy(33)],
        24,
    );
    expect(result.unfinished).toBe(0);
    expect(result.winsByName.greedy).toBeGreaterThan((result.winsByName.random ?? 0) / 3);
  });
});

describe('matchup benchmarks', () => {
  it('1 greedy vs 1 random (2p) — full metrics', () => {
    // Greedy alternates between seat 0 (even games) and seat 1 (odd games).
    // trackedSeat follows it so trackedWinRate = true greedy win rate.
    const stats = runMatchup(
        1000,
        g => g % 2 === 0
            ? [greedyPolicy(), randomPolicy(g)]
            : [randomPolicy(g), greedyPolicy()],
        10000,
        g => g % 2 === 0 ? 0 : 1,
    );

    console.log(JSON.stringify({ matchup: '1g_1r', ...stats }, null, 2));

    expect(stats.trackedWinRate).toBeGreaterThan(0.9);
    expect(stats.hitCap).toBeLessThan(20);
    expect(stats.avgWinnerVp).toBeLessThan(12);
    expect(stats.avgEndHandSize).toBeLessThan(8);
    expect(stats.vpSourcesPerGame.buildings).toBeGreaterThan(6);
  });

  it('greedy vs greedy (2p) — seat-bias check', () => {
    const stats = runMatchup(
        1000,
        () => [greedyPolicy(), greedyPolicy()],
        10000,
        () => 0,
    );
    const p0WinRate = stats.winsBySeat[0] / stats.games;
    const seatBias  = Math.abs(p0WinRate - 0.5);

    const { winRateBySeat, ...rest } = stats;
    console.log(JSON.stringify({
      matchup: '2g',
      winRateBySeat,
      p0WinRate,
      seatBias,
      ...rest,
    }, null, 2));

    expect(stats.hitCap).toBeLessThan(10);
    expect(stats.avgWinnerVp).toBeLessThan(12);
    // Neither seat should win more than 60% — large deviation = first-player bias
    expect(p0WinRate).toBeGreaterThan(0.4);
    expect(p0WinRate).toBeLessThan(0.6);
    expect(stats.avgTurns).toBeLessThan(150);
  });

  it('1 greedy vs 3 random (4p) — greedy dominance', () => {
    // Greedy rotates through all 4 seats; trackedSeat follows it so
    // trackedWinRate is the true greedy win rate across all seat positions.
    const stats = runMatchup(
        1000,
        g => {
          const greedySeat = g % 4;
          return Array.from({ length: 4 }, (_, i) =>
              i === greedySeat ? greedyPolicy() : randomPolicy(g * 4 + i),
          );
        },
        20000,
        g => g % 4,
    );

    const { winRateBySeat, ...rest } = stats;
    console.log(JSON.stringify({ matchup: '1g_3r', winRateBySeat, ...rest }, null, 2));

    // Random baseline is 25%; greedy should clearly beat that
    expect(stats.trackedWinRate).toBeGreaterThan(0.35);
    expect(stats.hitCap).toBeLessThan(10);
    expect(stats.avgWinnerVp).toBeLessThan(12);
  });

  it('greedy vs greedy vs greedy vs greedy (4p) — balanced seats', () => {
    const stats = runMatchup(
        1000,
        () => [greedyPolicy(), greedyPolicy(), greedyPolicy(), greedyPolicy()],
        30000,
        () => 0,
    );

    const maxWinRate = Math.max(...stats.winRateBySeat);
    const minWinRate = Math.min(...stats.winRateBySeat);
    const seatSpread = maxWinRate - minWinRate;

    const { winRateBySeat, ...rest } = stats;
    console.log(JSON.stringify({
      matchup: '4g',
      winRateBySeat,
      seatSpread,
      ...rest,
    }, null, 2));

    expect(stats.hitCap).toBeLessThan(10);
    expect(stats.avgWinnerVp).toBeLessThan(12);
    // No seat should dominate — spread under 15pp is reasonable for 4-player
    expect(seatSpread).toBeLessThan(0.15);
  });
});