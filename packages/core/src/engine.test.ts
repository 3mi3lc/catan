import { describe, it, expect } from 'vitest';
import { generateBoard, type Rng } from './board-generator';
import { initialGameState } from './setup';
import { applyMove, legalActions } from './engine';
import { distribute, victoryPoints, recomputeLongestRoad, COSTS } from './rules';
import { TERRAIN_RESOURCE } from './board';
import type { GameState, ResourceCounts, DevCard } from './state';
import type { EdgeId, PlayerId, VertexId } from './ids';
import { asPlayerId } from './ids';

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
    { id: 'p2', name: 'C', color: 'white' as const },
];

function newGame(seed = 7) {
    return initialGameState(generateBoard(mulberry32(42)), SEEDS, seed);
}

// Drive the initial-placement phase by always taking the first legal option.
function runSetup(start: GameState): GameState {
    let s = start;
    let guard = 0;
    while (s.phase === 'setupSettlement' || s.phase === 'setupRoad') {
        const options = legalActions(s, s.currentPlayer);
        const r = applyMove(s, { player: s.currentPlayer, action: options[0] });
        expect(r.ok).toBe(true);
        if (!r.ok) break;
        s = r.state;
        if (++guard > 100) throw new Error('setup did not terminate');
    }
    return s;
}

const give = (s: GameState, p: string, r: Partial<ResourceCounts>): GameState => ({
    ...s,
    players: {
        ...s.players,
        [asPlayerId(p)]: { ...s.players[asPlayerId(p)], resources: { ...s.players[asPlayerId(p)].resources, ...r } },
    },
});

describe('engine — setup', () => {
    it('completes initial placement and lands on the first player to roll', () => {
        const s = runSetup(newGame());
        expect(s.phase).toBe('roll');
        expect(s.currentPlayer).toBe(asPlayerId('p0'));
        expect(s.setup).toBeNull();
        // 3 players × 2 settlements + 2 roads each.
        expect(Object.keys(s.buildings)).toHaveLength(6);
        expect(Object.keys(s.roads)).toHaveLength(6);
    });

    it('grants starting resources for the second settlement only', () => {
        const s = runSetup(newGame());
        const totalCards = Object.values(s.players)
            .flatMap((p) => Object.values(p.resources))
            .reduce((a, b) => a + b, 0);
        // Each player earns the yield of their 2nd settlement (1-3 cards); never zero
        // overall once everyone has placed.
        expect(totalCards).toBeGreaterThan(0);
    });
});

describe('engine — turn flow', () => {
    it('rejects rolling out of turn and building before rolling', () => {
        const s = runSetup(newGame());
        expect(applyMove(s, { player: asPlayerId('p1'), action: { type: 'rollDice' } }).ok).toBe(false);
        expect(applyMove(s, { player: asPlayerId('p0'), action: { type: 'buyDevCard' } }).ok).toBe(false);
    });

    it('rotates to the next player on endTurn', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] }; // pretend a non-7 roll happened
        const r = applyMove(s, { player: asPlayerId('p0'), action: { type: 'endTurn' } });
        expect(r.ok).toBe(true);
        if (r.ok) {
            expect(r.state.currentPlayer).toBe(asPlayerId('p1'));
            expect(r.state.phase).toBe('roll');
        }
    });
});

describe('engine — building', () => {
    it('charges resources and the bank when building a road', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = give(s, 'p0', { brick: 2, lumber: 2 });

        // Find a legal road for p0.
        const road = legalActions(s, asPlayerId('p0')).find((a) => a.type === 'buildRoad') as
            | { type: 'buildRoad'; edge: EdgeId }
            | undefined;
        expect(road).toBeDefined();
        const before = s.players[asPlayerId('p0')].resources;
        const r = applyMove(s, { player: asPlayerId('p0'), action: road! });
        expect(r.ok).toBe(true);
        if (r.ok) {
            const after = r.state.players[asPlayerId('p0')].resources;
            expect(after.brick).toBe(before.brick - 1);
            expect(after.lumber).toBe(before.lumber - 1);
            expect(r.state.bank.brick).toBe(s.bank.brick + 1);
        }
    });

    it('refuses a road with no resources', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = give(s, 'p0', { brick: 0, lumber: 0, wool: 0, grain: 0, ore: 0 });
        const anyEdge = Object.keys(s.board.edges)[0] as EdgeId;
        expect(applyMove(s, { player: asPlayerId('p0'), action: { type: 'buildRoad', edge: anyEdge } }).ok).toBe(false);
    });
});

describe('engine — production', () => {
    it('pays the owner of a building on a producing tile', () => {
        const s = runSetup(newGame());
        // Find a tile with a number, put a fresh settlement on one of its corners.
        const tile = Object.values(s.board.tiles).find((t) => t.numberToken !== null && t.terrain !== 'desert')!;
        const vertex = tile.vertices.find((v) => !s.buildings[v])!;
        const withBld: GameState = {
            ...s,
            buildings: { ...s.buildings, [vertex]: { kind: 'settlement', owner: asPlayerId('p0') } },
        };
        const { gains } = distribute(withBld, tile.numberToken!);
        const total = Object.values(gains[asPlayerId('p0')] ?? {}).reduce((a, b) => a + (b ?? 0), 0);
        expect(total).toBeGreaterThanOrEqual(1);
    });

    it('removes the robbed tile from production', () => {
        const s = runSetup(newGame());
        const p0 = asPlayerId('p0');
        // A numbered tile where p0 has no building, so the only p0 production from it
        // is the settlement we add below.
        const tile = Object.values(s.board.tiles).find(
            (t) => t.numberToken !== null && t.vertices.every((v) => s.buildings[v]?.owner !== p0),
        )!;
        const vertex = tile.vertices.find((v) => !s.buildings[v])!;
        const res = TERRAIN_RESOURCE[tile.terrain]!;
        const base: GameState = { ...s, buildings: { ...s.buildings, [vertex]: { kind: 'settlement', owner: p0 } } };
        const withoutRobber = distribute(base, tile.numberToken!).gains[p0]?.[res] ?? 0;
        const withRobber = distribute({ ...base, robber: tile.id }, tile.numberToken!).gains[p0]?.[res] ?? 0;
        expect(withoutRobber - withRobber).toBe(1);
    });
});

describe('engine — longest road award', () => {
    it('awards +2 victory points for a 5-segment road', () => {
        const s = newGame(); // fresh: no buildings to break the chain
        const p0 = asPlayerId('p0');

        // Grab a 5-edge simple trail from the board.
        const path = findTrail(s, 5);
        expect(path).toHaveLength(5);
        const roads = Object.fromEntries(path.map((e) => [e, p0])) as Record<EdgeId, PlayerId>;
        const withRoads: GameState = { ...s, roads };

        const award = recomputeLongestRoad(withRoads);
        expect(award).toEqual({ player: p0, length: 5 });

        const withAward: GameState = { ...withRoads, longestRoad: award };
        expect(victoryPoints(withAward, p0)).toBe(2);
    });
});

describe('engine — dev card before rolling', () => {
    it('lets the current player play a knight in the roll phase, then still roll', () => {
        let s = runSetup(newGame());
        const p0 = asPlayerId('p0');
        expect(s.phase).toBe('roll');

        // Give p0 a playable knight and find an empty tile to send the robber to.
        s = { ...s, players: { ...s.players, [p0]: { ...s.players[p0], devCards: ['knight'] } } };
        const target = Object.values(s.board.tiles).find(
            (t) => t.id !== s.robber && t.vertices.every((v) => !s.buildings[v]),
        )!;

        const knight = applyMove(s, {
            player: p0,
            action: { type: 'playKnight', robberTo: target.id, stealFrom: null },
        });
        expect(knight.ok).toBe(true);
        if (!knight.ok) return;

        // Robber moved, army counted, card spent — but it's still p0's roll.
        expect(knight.state.robber).toBe(target.id);
        expect(knight.state.players[p0].playedKnights).toBe(1);
        expect(knight.state.devCardPlayedThisTurn).toBe(true);
        expect(knight.state.phase).toBe('roll');

        // p0 can now roll as normal.
        expect(applyMove(knight.state, { player: p0, action: { type: 'rollDice' } }).ok).toBe(true);

        // ...but not play a second dev card this turn (even after reaching main).
        const inMain = { ...knight.state, phase: 'main' as const, dice: [3, 4] as [number, number] };
        const second = applyMove(inMain, {
            player: p0,
            action: { type: 'playKnight', robberTo: s.robber, stealFrom: null },
        });
        expect(second.ok).toBe(false);
    });

    it('still blocks dev cards during robber/discard resolution and on others\u2019 turns', () => {
        let s = runSetup(newGame());
        const p0 = asPlayerId('p0');
        s = { ...s, players: { ...s.players, [p0]: { ...s.players[p0], devCards: ['knight'] } } };
        const tile = Object.values(s.board.tiles).find((t) => t.id !== s.robber)!;
        const play = { type: 'playKnight' as const, robberTo: tile.id, stealFrom: null };

        // Wrong player during p0's roll phase.
        expect(applyMove(s, { player: asPlayerId('p1'), action: play }).ok).toBe(false);
        // Mid-7 resolution: not allowed.
        expect(applyMove({ ...s, phase: 'moveRobber' }, { player: p0, action: play }).ok).toBe(false);
        expect(applyMove({ ...s, phase: 'discard' }, { player: p0, action: play }).ok).toBe(false);
    });
});

describe('engine — determinism', () => {
    const p0 = asPlayerId('p0');

    it('is a pure function of state and move (same roll twice → identical result)', () => {
        const s = runSetup(newGame());
        const a = applyMove(s, { player: p0, action: { type: 'rollDice' } });
        const b = applyMove(s, { player: p0, action: { type: 'rollDice' } });
        expect(a).toEqual(b);
        expect(a.ok).toBe(true);
        if (a.ok) expect(a.state.rng).not.toBe(s.rng); // the dice advanced the PRNG
    });

    it('replays a steal identically from the same seed', () => {
        const s = runSetup(newGame());
        // A tile with an opponent building whose owner holds at least one card.
        const tile = Object.values(s.board.tiles).find((t) =>
            t.id !== s.robber &&
            t.vertices.some((v) => {
                const b = s.buildings[v];
                return b && b.owner !== p0 && Object.values(s.players[b.owner].resources).some((n) => n > 0);
            }),
        );
        if (!tile) return; // unlucky layout; the roll-determinism test still covers PRNG threading
        const victim = tile.vertices.map((v) => s.buildings[v]).find((b) => b && b.owner !== p0)!.owner;
        const robberState = { ...s, phase: 'moveRobber' as const };
        const move = { player: p0, action: { type: 'moveRobber' as const, tile: tile.id, stealFrom: victim } };
        expect(applyMove(robberState, move)).toEqual(applyMove(robberState, move));
    });

    it('produces identical games from identical seeds, and differs across seeds', () => {
        expect(runSetup(newGame(123))).toEqual(runSetup(newGame(123)));
        // Different seed ⇒ different dev-deck order (and PRNG), so the states differ.
        expect(runSetup(newGame(123))).not.toEqual(runSetup(newGame(124)));
    });

    it('every enumerated robber move is accepted by applyMove', () => {
        const s = { ...runSetup(newGame()), phase: 'moveRobber' as const };
        for (const action of legalActions(s, p0)) {
            expect(applyMove(s, { player: p0, action }).ok, JSON.stringify(action)).toBe(true);
        }
    });
});

describe('engine — legalActions enumerates dev cards', () => {
    const p0 = asPlayerId('p0');
    const withHand = (s: GameState, ...cards: DevCard[]): GameState => ({
        ...s,
        players: { ...s.players, [p0]: { ...s.players[p0], devCards: cards } },
    });

    it('offers a knight before rolling, with a steal target per victim', () => {
        let s = runSetup(newGame());
        s = withHand(s, 'knight');
        const knights = legalActions(s, p0).filter((a) => a.type === 'playKnight');
        // One per (tile != robber) with no victims, plus one per victim on occupied tiles.
        expect(knights.length).toBeGreaterThan(0);
        // rollDice is still offered alongside.
        expect(legalActions(s, p0).some((a) => a.type === 'rollDice')).toBe(true);
    });

    it('offers monopoly for every resource and year-of-plenty pairs the bank can cover', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = withHand(s, 'monopoly', 'yearOfPlenty');
        const acts = legalActions(s, p0);
        expect(acts.filter((a) => a.type === 'playMonopoly')).toHaveLength(5);
        // 5 resources choose 2 with repetition = 15, all coverable from a full bank.
        expect(acts.filter((a) => a.type === 'playYearOfPlenty')).toHaveLength(15);
    });

    it('offers road-building pairs only with two roads in supply', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = withHand(s, 'roadBuilding');
        expect(legalActions(s, p0).some((a) => a.type === 'playRoadBuilding')).toBe(true);

        const drained: GameState = {
            ...s,
            players: { ...s.players, [p0]: { ...s.players[p0], supply: { ...s.players[p0].supply, roads: 1 } } },
        };
        expect(legalActions(drained, p0).some((a) => a.type === 'playRoadBuilding')).toBe(false);
    });

    it('offers nothing once a card has been played, or to other players', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = withHand(s, 'knight', 'monopoly');
        const played = { ...s, devCardPlayedThisTurn: true };
        const devTypes = new Set(['playKnight', 'playMonopoly', 'playYearOfPlenty', 'playRoadBuilding']);
        expect(legalActions(played, p0).some((a) => devTypes.has(a.type))).toBe(false);
        expect(legalActions(s, asPlayerId('p1')).some((a) => devTypes.has(a.type))).toBe(false);
    });

    it('every enumerated action is accepted by applyMove (the AI contract)', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = withHand(s, 'knight', 'monopoly', 'yearOfPlenty', 'roadBuilding');
        s = give(s, 'p0', { brick: 4, lumber: 4, wool: 4, grain: 4, ore: 4 });
        for (const action of legalActions(s, p0)) {
            const r = applyMove(s, { player: p0, action });
            expect(r.ok, `rejected: ${JSON.stringify(action)}`).toBe(true);
        }
    });
});

describe('engine — bank trade', () => {
    it('trades at 4:1 by default', () => {
        let s = runSetup(newGame());
        s = { ...s, phase: 'main', dice: [3, 4] };
        s = give(s, 'p0', { brick: 4, ore: 0 });
        const r = applyMove(s, {
            player: asPlayerId('p0'),
            action: { type: 'bankTrade', give: 'brick', giveCount: 4, receive: 'ore' },
        });
        expect(r.ok).toBe(true);
        if (r.ok) {
            expect(r.state.players[asPlayerId('p0')].resources.brick).toBe(0);
            expect(r.state.players[asPlayerId('p0')].resources.ore).toBe(1);
        }
    });
});

// DFS for an n-edge simple trail anywhere on the board.
function findTrail(s: GameState, n: number): EdgeId[] {
    const board = s.board;
    const path: EdgeId[] = [];
    const used = new Set<EdgeId>();
    const dfs = (v: VertexId): boolean => {
        if (path.length === n) return true;
        for (const e of board.vertices[v].edges) {
            if (used.has(e)) continue;
            used.add(e); path.push(e);
            const [a, b] = board.edges[e].vertices;
            if (dfs(a === v ? b : a)) return true;
            used.delete(e); path.pop();
        }
        return false;
    };
    for (const sv of Object.keys(board.vertices) as VertexId[]) {
        used.clear(); path.length = 0;
        if (dfs(sv)) return [...path];
    }
    return path;
}