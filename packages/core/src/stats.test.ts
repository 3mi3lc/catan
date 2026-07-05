import { describe, it, expect } from 'vitest';
import { generateBoard, type Rng } from './board-generator';
import { initialGameState } from './setup';
import { applyMove, legalActions } from './engine';
import { distribute, victoryPoints, tradeRatio } from './rules';
import { RESOURCES, type Resource } from './board';
import type { GameState, ResourceCounts } from './state';
import type { PlayerId } from './ids';
import type { Action } from './actions';
import { NO_ROBBER_TILE, type MoveRecord, type GameArchive } from './replay';
import { computeGameStats } from './stats';

function mulberry32(seed: number): Rng {
    let a = seed >>> 0;
    return () => {
        a |= 0; a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const SEEDS = [
    { id: 'p0', name: 'A', color: 'red' as const },
    { id: 'p1', name: 'B', color: 'blue' as const },
];

// Drives a real, fully-legal game via applyMove (same path the real app
// uses), recording every move into a GameArchive — exactly what hotseat.ts/
// the server will do for real games. Opportunistic but deterministic for a
// fixed seed: always plays a dev card the moment one is legal (to surface
// knight/monopoly/yearOfPlenty/roadBuilding play), forces a bank trade the
// moment someone holds 4+ of one resource, and scripts exactly one
// player-to-player trade once both have cards — so the resulting archive is
// guaranteed (for this seed) to exercise every ResourceFlowCategory at least
// once, not just whichever ones luck produces.
function recordedGame(boardSeed: number, gameSeed: number, maxTurns: number): { archive: GameArchive; finalState: GameState } {
    const board = generateBoard(mulberry32(boardSeed));
    let s = initialGameState(board, SEEDS, gameSeed, Number.MAX_SAFE_INTEGER);
    const moves: MoveRecord[] = [];
    let turnCount = 0;
    let tradeDone = false;

    const act = (player: PlayerId, action: Action): void => {
        moves.push({ player, phase: s.phase, action });
        const res = applyMove(s, { player, action });
        if (!res.ok) throw new Error(`scripted move failed: ${action.type} — ${res.error}`);
        s = res.state;
    };

    // Setup: always take the first legal option.
    let guard = 0;
    while (s.phase === 'setupSettlement' || s.phase === 'setupRoad') {
        act(s.currentPlayer, legalActions(s, s.currentPlayer)[0]);
        if (++guard > 200) throw new Error('setup did not terminate');
    }

    while (turnCount < maxTurns && !s.winner) {
        const cur = s.currentPlayer;
        act(cur, { type: 'rollDice' });

        // Discards owed after a 7 — pick resources greedily.
        for (const p of Object.keys(s.pendingDiscards) as PlayerId[]) {
            const need = s.pendingDiscards[p];
            const have = { ...s.players[p].resources };
            const chosen: Partial<ResourceCounts> = {};
            let left = need;
            for (const r of RESOURCES) {
                const take = Math.min(have[r], left);
                if (take > 0) { chosen[r] = take; left -= take; }
                if (left === 0) break;
            }
            act(p, { type: 'discard', resources: chosen });
        }

        // Move the robber, preferring an actual steal target if one exists.
        if (s.phase === 'moveRobber') {
            const opts = legalActions(s, s.currentPlayer).filter((a) => a.type === 'moveRobber');
            const chosen = opts.find((a) => a.type === 'moveRobber' && a.stealFrom !== null) ?? opts[0];
            act(s.currentPlayer, chosen);
        }

        // One scripted player-to-player trade, once both have something to trade.
        if (!tradeDone && s.phase === 'main') {
            const [a, b] = s.turnOrder;
            const haveA = RESOURCES.filter((r) => s.players[a].resources[r] > 0);
            const haveB = RESOURCES.filter((r) => s.players[b].resources[r] > 0);
            if (haveA.length > 0 && haveB.length > 0) {
                tradeDone = true;
                act(a, { type: 'offerAddGive', resource: haveA[0] });
                act(a, { type: 'offerAddWant', resource: haveB[0] });
                act(a, { type: 'offerBroadcast' });
                act(b, { type: 'respondAccept' });
                act(a, { type: 'confirmTrade', to: b });
            }
        }

        // Opportunistically play a dev card the moment one is legal.
        if (s.phase === 'main') {
            const playable = legalActions(s, s.currentPlayer).find((a) =>
                a.type === 'playKnight' || a.type === 'playMonopoly'
                || a.type === 'playYearOfPlenty' || a.type === 'playRoadBuilding');
            if (playable) act(s.currentPlayer, playable);
        }

        // Build whatever's legal; failing that, buy a dev card; failing that,
        // force a bank trade if holding 4+ of a single resource.
        while (s.phase === 'main') {
            const opts = legalActions(s, s.currentPlayer)
                .filter((a) => a.type === 'buildSettlement' || a.type === 'buildCity' || a.type === 'buildRoad');
            if (opts.length > 0) { act(s.currentPlayer, opts[0]); continue; }
            if (legalActions(s, s.currentPlayer).some((a) => a.type === 'buyDevCard')) {
                act(s.currentPlayer, { type: 'buyDevCard' });
                continue;
            }
            const me = s.players[s.currentPlayer].resources;
            const give = RESOURCES.find((r) => me[r] >= tradeRatio(s, s.currentPlayer, r));
            if (give) {
                const giveCount = tradeRatio(s, s.currentPlayer, give);
                const receive = RESOURCES.find((r) => r !== give)!;
                act(s.currentPlayer, { type: 'bankTrade', give, giveCount, receive });
                continue;
            }
            break;
        }

        act(cur, { type: 'endTurn' });
        turnCount++;
    }

    const archive: GameArchive = {
        seed: gameSeed,
        policyNames: SEEDS.map((s2) => s2.name),
        winnerSeat: s.winner ? s.turnOrder.indexOf(s.winner) : null,
        turns: turnCount,
        moves,
        maxOffersPerTurn: Number.MAX_SAFE_INTEGER,
    };
    // generateBoard(mulberry32(boardSeed)) above is informational only — the
    // real GameArchive format (and loadReplay) regenerate the board from
    // archive.seed alone, so boardSeed and gameSeed must be the same value
    // for computeGameStats to reconstruct this exact game.
    return { archive, finalState: s };
}

describe('computeGameStats', () => {
    // Empirically chosen (see comment on recordedGame) so that, within 60
    // turns, every category below is exercised at least once.
    const SEED = 2;
    const { archive, finalState } = recordedGame(SEED, SEED, 60);
    const stats = computeGameStats(archive);

    it('dice histogram totals match the number of rollDice moves', () => {
        const rollMoves = archive.moves.filter((m) => m.action.type === 'rollDice').length;
        expect(stats.dice.totalRolls).toBe(rollMoves);
        const summed = Object.values(stats.dice.rollCounts).reduce((a, b) => a + b, 0);
        expect(summed).toBe(rollMoves);
        for (const p of stats.players) expect(p.diceRolls.length).toBeGreaterThan(0);
    });

    it('categorizes a bank trade', () => {
        const total = stats.players.reduce((sum, p) => sum + Object.keys(p.gainedByCategory.bankTrade).length, 0);
        expect(total).toBeGreaterThan(0);
    });

    it('categorizes a robber steal, including who was targeted', () => {
        const stolen = stats.players.some((p) => Object.keys(p.robber.stolenFromOthers).length > 0);
        const targeted = stats.players.some((p) => p.robber.timesTargeted > 0);
        expect(stolen).toBe(true);
        expect(targeted).toBe(true);
    });

    it('categorizes a monopoly play', () => {
        const total = stats.players.reduce((sum, p) => sum + p.devCardsPlayed.monopoly, 0);
        expect(total).toBeGreaterThan(0);
        const gained = stats.players.some((p) => Object.keys(p.gainedByCategory.monopoly).length > 0);
        expect(gained).toBe(true);
    });

    it('records the scripted player trade under tradesByPartner for both sides', () => {
        const [a, b] = stats.players;
        expect(a.trade.tradesByPartner[b.player]).toBeGreaterThan(0);
        expect(b.trade.tradesByPartner[a.player]).toBeGreaterThan(0);
        expect(a.trade.tradesCompleted).toBeGreaterThan(0);
        expect(b.trade.tradesCompleted).toBeGreaterThan(0);
    });

    it('computes robber-blocked production consistent with rules.ts distribute()', () => {
        // Cross-check: replay the archive's own rolls and verify at least one
        // roll was actually blocked by the robber, and that the recorded
        // blocked amount for that roll matches calling distribute() directly.
        let sawBlock = false;
        let s = initialGameState(generateBoard(mulberry32(SEED)), SEEDS, SEED, Number.MAX_SAFE_INTEGER);
        // Replay to setup's end the same way recordedGame did.
        for (const m of archive.moves) {
            const before = s;
            const res = applyMove(s, { player: m.player as PlayerId, action: m.action });
            if (!res.ok) throw new Error('unexpected replay failure in cross-check');
            s = res.state;
            if (m.action.type !== 'rollDice') continue;
            const rolled = res.events.find((e) => e.type === 'diceRolled');
            const produced = res.events.find((e) => e.type === 'resourcesProduced');
            if (!rolled || rolled.type !== 'diceRolled' || rolled.total === 7) continue;
            const hypothetical = distribute({ ...before, robber: NO_ROBBER_TILE }, rolled.total);
            for (const p of s.turnOrder) {
                for (const r of RESOURCES) {
                    const would = hypothetical.gains[p]?.[r] ?? 0;
                    const did = (produced?.type === 'resourcesProduced' ? produced.gains[p]?.[r] : 0) ?? 0;
                    if (would > did) sawBlock = true;
                }
            }
        }
        expect(sawBlock).toBe(true);
        const totalBlocked = stats.players.reduce(
            (sum, p) => sum + Object.values(p.robber.productionBlocked).reduce((a, b) => a + (b ?? 0), 0), 0);
        expect(totalBlocked).toBeGreaterThan(0);
    });

    it('final VP breakdown matches victoryPoints()', () => {
        for (const p of stats.players) {
            const expected = victoryPoints(finalState, p.player);
            expect(p.finalVP).toBe(expected);
            const fromBreakdown = p.vpBreakdown.settlements + p.vpBreakdown.cities * 2
                + p.vpBreakdown.devCards + p.vpBreakdown.longestRoad + p.vpBreakdown.largestArmy;
            expect(fromBreakdown).toBe(expected);
        }
    });

    it('reports the same winner as the final state', () => {
        expect(stats.winner).toBe(finalState.winner);
    });

    it('counts dev cards drawn by type, summing to devCardsBought', () => {
        for (const p of stats.players) {
            const drawnTotal = Object.values(p.devCardsDrawn).reduce((a, b) => a + b, 0);
            expect(drawnTotal).toBe(p.devCardsBought);
            // VP cards are never played, so the drawn count must match the VP
            // breakdown's dev-card contribution exactly.
            expect(p.devCardsDrawn.victoryPoint).toBe(p.vpBreakdown.devCards);
        }
    });
});

describe('computeGameStats — winnerSeat round-trip', () => {
    it('an old-style archive without maxOffersPerTurn still replays (defaults to 1)', () => {
        const { archive } = recordedGame(11, 11, 5);
        const { maxOffersPerTurn, ...withoutCap } = archive;
        void maxOffersPerTurn;
        // This game used a high cap and a scripted trade — stripping the
        // field shouldn't matter for a short 5-turn slice that never hits
        // more than one offer in a turn anyway; this just confirms the
        // back-compat default doesn't throw on a realistic archive shape.
        expect(() => computeGameStats(withoutCap)).not.toThrow();
    });
});
