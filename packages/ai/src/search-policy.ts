import {
    legalActions, applyMove, victoryPoints,
    type GameState, type Action, type PlayerId, type Resource,
} from '@catan/core';
import type { Policy } from './policy';
import { ofType } from './policy';
import {
    type Weights, DEFAULT_WEIGHTS, vertexScore, vertexProduction, robberScore, argmax,
    discardAction, roadExpansionValue, bestYopTake,
} from './heuristics';
import { greedyPolicy } from './greedy-policy';

export interface SearchOptions {
    rollouts?: number;        // playouts per candidate (the strength/speed knob)
    rolloutCap?: number;      // hard ply limit per playout (games finish well under)
    rolloutPolicy?: Policy;   // default policy used inside playouts (default: greedy)
    weights?: Weights;        // heuristic weights for candidate proposal
    seed?: number;
    name?: string;
}

// Flat Monte Carlo search ("MCTS-lite"): for each candidate action, simulate it
// then play the game out several times with the rollout policy, varying the
// dice each time, and keep the action with the best average outcome. Strength
// scales with `rollouts`; the engine's purity + seeded RNG make this cheap and
// reproducible. The rollout policy can later be swapped for a learned net.
export function searchPolicy(opts: SearchOptions = {}): Policy {
    const rollouts = opts.rollouts ?? 24;
    const cap = opts.rolloutCap ?? 1500;
    const roll = opts.rolloutPolicy ?? greedyPolicy();
    const w = opts.weights ?? DEFAULT_WEIGHTS;
    const baseSeed = opts.seed ?? 0x5eed;
    let tick = 0; // advances per decision so playouts vary; reproducible per game

    function evaluateCandidate(state: GameState, player: PlayerId, action: Action, ci: number): number {
        let sum = 0;
        for (let k = 0; k < rollouts; k++) {
            const seeded: GameState = { ...state, rng: mix(baseSeed, tick, ci, k) };
            const applied = applyMove(seeded, { player, action });
            if (!applied.ok) return -Infinity; // illegal candidate; never pick it
            sum += leafScore(rollout(applied.state, roll, cap), player);
        }
        return sum / rollouts;
    }

    return {
        name: opts.name ?? 'search',
        decide(state, player): Action {
            // Forced / cheap decisions: don't waste playouts.
            if (state.phase === 'roll') return { type: 'rollDice' };
            if (state.phase === 'discard') return discardAction(state, player);

            // Robber placement: greedy heuristic is already strong here and
            // running full playouts per tile is expensive — short-circuit it.
            if (state.phase === 'moveRobber') {
                const acts = legalActions(state, player);
                const best = argmax(
                    ofType(acts, 'moveRobber'),
                    (a) => robberScore(state, player, a.tile, a.stealFrom, w),
                );
                return best ?? { type: 'endTurn' };
            }

            const cands = candidates(state, player, legalActions(state, player), w);
            tick++;
            if (cands.length <= 1) return cands[0] ?? { type: 'endTurn' };

            let best = cands[0];
            let bestScore = -Infinity;
            cands.forEach((a, ci) => {
                const s = evaluateCandidate(state, player, a, ci);
                if (s > bestScore) { bestScore = s; best = a; }
            });
            return best;
        },
    };
}

// Play a state forward with one policy until someone wins or the cap is hit.
function rollout(state: GameState, policy: Policy, cap: number): GameState {
    let s = state;
    for (let i = 0; i < cap && !s.winner; i++) {
        const actor = s.phase === 'discard'
            ? (Object.keys(s.pendingDiscards)[0] as PlayerId)
            : s.currentPlayer;
        const res = applyMove(s, { player: actor, action: policy.decide(s, actor) });
        if (!res.ok) break;
        s = res.state;
    }
    return s;
}

// Outcome of a finished (or capped) playout from `player`'s view, in [0, 1].
function leafScore(state: GameState, player: PlayerId): number {
    if (state.winner === player) return 1;
    if (state.winner !== null) return 0;
    // Unfinished: estimate from the VP gap, kept away from 0/1 to admit doubt.
    const mine = victoryPoints(state, player);
    let oppMax = 0;
    for (const p of state.turnOrder) if (p !== player) oppMax = Math.max(oppMax, victoryPoints(state, p));
    return clamp(0.5 + 0.08 * (mine - oppMax), 0.05, 0.95);
}

// A small, curated candidate set per phase — the search's branching factor.
// Pruning to the heuristically-promising moves keeps playouts affordable while
// still letting search correct the greedy's mistakes.
function candidates(state: GameState, player: PlayerId, acts: Action[], w: Weights): Action[] {
    switch (state.phase) {
        case 'setupSettlement':
            return topN(ofType(acts, 'buildSettlement'), (a) => vertexScore(state, a.vertex, w, player), 5);
        case 'setupRoad': {
            const last = state.setup?.lastSettlement;
            return topN(ofType(acts, 'buildRoad'), (a) => {
                const [x, y] = state.board.edges[a.edge].vertices;
                return vertexScore(state, x === last ? y : x, w, player);
            }, 3);
        }
        case 'moveRobber':
            return topN(ofType(acts, 'moveRobber'), (a) => robberScore(state, player, a.tile, a.stealFrom, w), 4);
        case 'main':
            return mainCandidates(state, player, acts, w);
        default:
            return acts.length ? [acts[0]] : [{ type: 'endTurn' }];
    }
}

function mainCandidates(state: GameState, player: PlayerId, acts: Action[], w: Weights): Action[] {
    const out: Action[] = [];
    const add = (a: Action | undefined) => { if (a) out.push(a); };

    add(argmax(ofType(acts, 'buildCity'), (a) => vertexProduction(state, a.vertex, w)));
    add(argmax(ofType(acts, 'buildSettlement'), (a) => vertexScore(state, a.vertex, w, player)));
    add(argmax(ofType(acts, 'playKnight'), (a) => robberScore(state, player, a.robberTo, a.stealFrom, w)));

    // BFS road evaluation: scores intermediate chain roads correctly.
    // Skip paid roads when we have a road building card to play for free.
    const hasRoadBuilding = state.players[player].devCards.includes('roadBuilding');
    const road = !hasRoadBuilding
        ? argmax(ofType(acts, 'buildRoad'), (a) => roadExpansionValue(state, player, a.edge, w))
        : undefined;
    if (road && roadExpansionValue(state, player, road.edge, w) > 0) add(road);

    // Road Building card — free roads, always worth evaluating as a candidate.
    const roadBuildingAct = argmax(
        ofType(acts, 'playRoadBuilding'),
        (a) => roadExpansionValue(state, player, a.edges[0], w) +
            roadExpansionValue(state, player, a.edges[1], w),
    );
    if (roadBuildingAct) add(roadBuildingAct);

    // Don't buy more dev cards when we already have unplayed action cards in hand.
    const hasUnplayedActionCard = state.players[player].devCards.some(
        (c) => c === 'roadBuilding' || c === 'yearOfPlenty' || c === 'monopoly' || c === 'knight',
    );
    if (ofType(acts, 'buyDevCard').length && !hasUnplayedActionCard) add({ type: 'buyDevCard' });

    // Monopoly: only worth evaluating when opponents actually hold the resource.
    const mono = argmax(ofType(acts, 'playMonopoly'), (a) => oppTotal(state, player, a.resource));
    if (mono && oppTotal(state, player, (mono as any).resource) >= 2) add(mono);

    // Smart YoP: compute the 2 most useful resources instead of taking yop[0]
    // which was always [brick,brick] due to legalActions enumeration order.
    if (ofType(acts, 'playYearOfPlenty').length) {
        const take = bestYopTake(state, player);
        if (take) add({ type: 'playYearOfPlenty', take });
    }

    // Goal-directed trades: toward the nearest city or settlement rather than
    // blindly dumping the most abundant resource.
    const R = state.players[player].resources;
    const cityGap  = Math.max(0, 2 - R.grain) + Math.max(0, 3 - R.ore);
    const settGap  = (['brick', 'lumber', 'wool', 'grain'] as Resource[]).filter(r => R[r] < 1).length;
    const tradeActs = ofType(acts, 'bankTrade');
    if (cityGap <= settGap && cityGap <= 2) {
        if (R.grain < 2) add(argmax(tradeActs.filter(t => t.receive === 'grain'), t => R[t.give]));
        if (R.ore   < 3) add(argmax(tradeActs.filter(t => t.receive === 'ore'),   t => R[t.give]));
    } else if (settGap <= 2) {
        const need = (['brick', 'lumber', 'wool', 'grain'] as Resource[]).find(r => R[r] < 1);
        if (need) add(argmax(tradeActs.filter(t => t.receive === need), t => R[t.give]));
    }
    // Hand pressure fallback: dump surplus before risking a robber discard.
    if (handTotal(R) >= 8) {
        add(argmax(tradeActs, (t) => R[t.give] * (1 / w.resource[t.give])));
    }

    add({ type: 'endTurn' });

    // Dedup (some "best of type" picks may coincide).
    const seen = new Set<string>();
    return out.filter((a) => { const k = JSON.stringify(a); return seen.has(k) ? false : (seen.add(k), true); });
}

const oppTotal = (state: GameState, player: PlayerId, r: Resource): number =>
    state.turnOrder.filter((p) => p !== player).reduce((s, p) => s + state.players[p].resources[r], 0);

const handTotal = (r: Record<Resource, number>): number =>
    r.brick + r.lumber + r.wool + r.grain + r.ore;

function topN<T>(items: T[], score: (t: T) => number, n: number): T[] {
    return [...items].sort((a, b) => score(b) - score(a)).slice(0, n);
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

function mix(...xs: number[]): number {
    let h = 2166136261 >>> 0;
    for (const x of xs) { h ^= x >>> 0; h = Math.imul(h, 16777619); }
    return (h >>> 0) || 1;
}