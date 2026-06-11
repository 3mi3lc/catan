/**
 * mcts.ts — PUCT Monte Carlo Tree Search guided by a policy/value network
 * (AlphaZero-style, adapted for Catan's stochasticity).
 *
 * Differences from search-policy.ts (flat Monte Carlo with heuristic rollouts):
 *   • No rollouts: leaves are evaluated by the net's value head.
 *   • Candidate priors come from the net's policy head (PUCT exploration).
 *   • Chance (dice, dev-card draws, robber steals) is handled open-loop: the
 *     engine RNG is reseeded from the search RNG before every applyMove, so
 *     repeated visits to the same edge sample different outcomes and Q
 *     averages over chance. Forced moves (e.g. rollDice) and the discard
 *     phase are auto-resolved inside the tree, so dice chains are absorbed
 *     between decision nodes.
 *
 * Two-player only (values are P(win) for the node's actor; the opponent's
 * perspective is 1 − v).
 *
 * The evaluator is async (ONNX inference); `runMcts` is therefore async.
 */

import { applyMove, victoryPoints, type GameState, type PlayerId } from '@catan/core';
import {
    encodeObservation, legalMask, indexToAction, ACT_SIZE,
    type BoardIndex,
} from './encoding';
import { discardAction } from './heuristics';
import { greedyPolicy } from './greedy-policy';
import type { Policy } from './policy';

// ── Public interfaces ─────────────────────────────────────────────────────────

export interface NetEvaluator {
    /**
     * Evaluate `state` from `player`'s perspective.
     * Returns masked-normalised priors over ACT_SIZE (zero on illegal slots)
     * and P(player wins) ∈ (0, 1).
     */
    evaluate(state: GameState, player: PlayerId, mask: Uint8Array):
        Promise<{ priors: Float32Array; winProb: number }>;
}

export interface MctsOptions {
    sims: number;            // simulations per decision (strength knob)
    cPuct?: number;          // exploration constant (default 1.5)
    rand?: () => number;     // uniform [0,1) RNG (seeded for reproducibility)
    /** Dirichlet root noise for self-play exploration; omit to disable. */
    rootNoise?: { alpha: number; eps: number };
    maxDepth?: number;       // tree-path cap per simulation (default 200)
    /**
     * Leaf evaluation: with probability `valueMix` the leaf is scored by the
     * net's value head alone; otherwise by a truncated greedy rollout.
     * Expectation matches a deterministic blend (Q averages over visits), but
     * rollout COST scales with (1 − valueMix), so annealing the mix toward 1
     * directly buys speed as the value head matures.
     * 1 = pure net value (AlphaZero); 0 = pure rollout (AlphaGo-style).
     * While the value head is young/biased, rollout grounding keeps search an
     * IMPROVEMENT operator — measured: pure net-value MCTS scored below the
     * raw policy (35% vs 45% against greedy) at generation 3.
     * Default 1 (no rollouts).
     */
    valueMix?: number;
    /** Ply cap for leaf rollouts (default 600 — deep enough that most
     *  rollouts from the mid-game reach a real winner; short caps leave the
     *  crude VP-gap heuristic doing the evaluating). */
    rolloutCap?: number;
}

export interface MctsResult {
    /** Visit distribution over ACT_SIZE (sums to 1) — the training target. */
    visits: Float32Array;
    /** Root Q of the most-visited action (P(actor wins)). */
    rootValue: number;
}

// ── Tree structures ───────────────────────────────────────────────────────────

interface Edge {
    actionIdx: number;
    prior: number;
    visits: number;
    valueSum: number;        // from the parent node actor's perspective
    child: Node | null;
}

interface Node {
    state: GameState;        // a decision state: actor has > 1 legal action
    actor: PlayerId;
    mask: Uint8Array;
    edges: Edge[];
    expanded: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Reseed the engine RNG from the search RNG so chance resamples per visit. */
function withFreshRng(state: GameState, rand: () => number): GameState {
    return { ...state, rng: (rand() * 0xffffffff) >>> 0 };
}

/**
 * Advance `state` until it is terminal or some player faces a real decision:
 * auto-resolves the discard phase (heuristic) and forced single-legal moves
 * (rollDice etc.). Chance inside those moves uses the search RNG.
 */
function advance(
    state: GameState, bi: BoardIndex, rand: () => number,
): { state: GameState; actor: PlayerId; mask: Uint8Array } | { terminal: GameState } {
    let s = state;
    for (let guard = 0; guard < 64; guard++) {
        if (s.winner) return { terminal: s };

        const actor: PlayerId = s.phase === 'discard'
            ? (Object.keys(s.pendingDiscards)[0] as PlayerId)
            : s.currentPlayer;

        if (s.phase === 'discard') {
            const res = applyMove(withFreshRng(s, rand),
                { player: actor, action: discardAction(s, actor) });
            if (!res.ok) return { terminal: s };
            s = res.state;
            continue;
        }

        const mask = legalMask(s, actor, bi);
        let legalCount = 0, onlyIdx = -1;
        for (let i = 0; i < mask.length; i++) {
            if (mask[i] === 1) { legalCount++; onlyIdx = i; }
        }

        if (legalCount === 0) {
            const res = applyMove(withFreshRng(s, rand),
                { player: actor, action: { type: 'endTurn' } });
            if (!res.ok) return { terminal: s };
            s = res.state;
            continue;
        }
        if (legalCount === 1) {
            const res = applyMove(withFreshRng(s, rand),
                { player: actor, action: indexToAction(onlyIdx, s, actor, bi) });
            if (!res.ok) return { terminal: s };
            s = res.state;
            continue;
        }
        return { state: s, actor, mask };
    }
    return { terminal: s }; // forced-move chain too long; treat as leaf
}

/**
 * Truncated greedy rollout from `state`, scored for `player` in [0, 1].
 * Same idea as search-policy.ts's playouts: if no one wins within the cap,
 * estimate from the victory-point gap, kept away from 0/1 to admit doubt.
 */
function rolloutScore(
    state: GameState,
    player: PlayerId,
    policy: Policy,
    cap: number,
    rand: () => number,
): number {
    let s = state;
    for (let i = 0; i < cap && !s.winner; i++) {
        const actor: PlayerId = s.phase === 'discard'
            ? (Object.keys(s.pendingDiscards)[0] as PlayerId)
            : s.currentPlayer;
        const res = applyMove(withFreshRng(s, rand),
            { player: actor, action: policy.decide(s, actor) });
        if (!res.ok) break;
        s = res.state;
    }
    if (s.winner === player) return 1;
    if (s.winner !== null) return 0;
    const mine = victoryPoints(s, player);
    let oppMax = 0;
    for (const p of s.turnOrder) {
        if (p !== player) oppMax = Math.max(oppMax, victoryPoints(s, p));
    }
    return Math.min(0.95, Math.max(0.05, 0.5 + 0.08 * (mine - oppMax)));
}

/** Gamma(alpha) sample via Marsaglia–Tsang (alpha < 1 boost included). */
function sampleGamma(alpha: number, rand: () => number): number {
    if (alpha < 1) {
        return sampleGamma(alpha + 1, rand) * Math.pow(rand() || 1e-12, 1 / alpha);
    }
    const d = alpha - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
        // Box–Muller standard normal
        const u1 = rand() || 1e-12, u2 = rand();
        const x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const v = Math.pow(1 + c * x, 3);
        if (v <= 0) continue;
        const u = rand();
        if (Math.log(u || 1e-12) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
            return d * v;
        }
    }
}

// ── Core search ───────────────────────────────────────────────────────────────

export async function runMcts(
    rootState: GameState,
    rootActor: PlayerId,
    rootMask: Uint8Array,
    bi: BoardIndex,
    evaluator: NetEvaluator,
    opts: MctsOptions,
): Promise<MctsResult> {
    const ctx: Ctx = {
        evaluator,
        cPuct: opts.cPuct ?? 1.5,
        rand: opts.rand ?? Math.random,
        maxDepth: opts.maxDepth ?? 200,
        valueMix: opts.valueMix ?? 1.0,
        rolloutCap: opts.rolloutCap ?? 600,
        rolloutPolicy: greedyPolicy(),
    };

    const root: Node = {
        state: rootState, actor: rootActor, mask: rootMask,
        edges: [], expanded: false,
    };
    await expand(root, ctx);

    // Dirichlet noise on root priors (self-play exploration).
    if (opts.rootNoise) {
        const { alpha, eps } = opts.rootNoise;
        const noise = root.edges.map(() => sampleGamma(alpha, ctx.rand));
        const sum = noise.reduce((a, b) => a + b, 0) || 1;
        root.edges.forEach((e, i) => {
            e.prior = (1 - eps) * e.prior + eps * (noise[i] / sum);
        });
    }

    for (let sim = 0; sim < opts.sims; sim++) {
        await simulate(root, bi, ctx);
    }

    const visits = new Float32Array(ACT_SIZE);
    let total = 0;
    for (const e of root.edges) { visits[e.actionIdx] = e.visits; total += e.visits; }
    if (total > 0) for (let i = 0; i < ACT_SIZE; i++) visits[i] /= total;

    let best: Edge | null = null;
    for (const e of root.edges) if (!best || e.visits > best.visits) best = e;
    const rootValue = best && best.visits > 0 ? best.valueSum / best.visits : 0.5;

    return { visits, rootValue };
}

interface Ctx {
    evaluator: NetEvaluator;
    cPuct: number;
    rand: () => number;
    maxDepth: number;
    valueMix: number;
    rolloutCap: number;
    rolloutPolicy: Policy;
}

/** Expand a node and return its blended leaf value (P(node.actor wins)). */
async function expand(node: Node, ctx: Ctx): Promise<number> {
    const { priors, winProb } =
        await ctx.evaluator.evaluate(node.state, node.actor, node.mask);
    node.edges = [];
    for (let i = 0; i < ACT_SIZE; i++) {
        if (node.mask[i] === 1) {
            node.edges.push({
                actionIdx: i, prior: priors[i], visits: 0, valueSum: 0, child: null,
            });
        }
    }
    node.expanded = true;

    if (ctx.valueMix >= 1 || ctx.rand() < ctx.valueMix) return winProb;
    return rolloutScore(
        node.state, node.actor, ctx.rolloutPolicy, ctx.rolloutCap, ctx.rand);
}

async function simulate(
    root: Node,
    bi: BoardIndex,
    ctx: Ctx,
): Promise<void> {
    const { rand, cPuct, maxDepth } = ctx;
    // Walk down with PUCT until an unexpanded child or terminal.
    const path: { node: Node; edge: Edge }[] = [];
    let node = root;
    let leafValue = 0.5;             // P(leafActor wins)
    let leafActor: PlayerId = root.actor;

    for (let depth = 0; depth < maxDepth; depth++) {
        // Select edge maximising PUCT.
        const sqrtN = Math.sqrt(
            1 + node.edges.reduce((a, e) => a + e.visits, 0));
        let bestEdge = node.edges[0];
        let bestScore = -Infinity;
        for (const e of node.edges) {
            const q = e.visits > 0 ? e.valueSum / e.visits : 0.5; // FPU 0.5
            const u = cPuct * e.prior * sqrtN / (1 + e.visits);
            const s = q + u;
            if (s > bestScore) { bestScore = s; bestEdge = e; }
        }
        path.push({ node, edge: bestEdge });

        if (bestEdge.child === null) {
            // Apply the action with fresh chance, advance to the next decision.
            const action = indexToAction(
                bestEdge.actionIdx, node.state, node.actor, bi);
            const res = applyMove(withFreshRng(node.state, rand),
                { player: node.actor, action });
            if (!res.ok) {
                // Encoding bug guard: poison this edge so it is never re-picked.
                bestEdge.prior = 0;
                bestEdge.visits = 1e9;
                return;
            }
            const adv = advance(res.state, bi, rand);
            if ('terminal' in adv) {
                const w = adv.terminal.winner;
                leafActor = node.actor;
                leafValue = w === null ? 0.5 : (w === node.actor ? 1 : 0);
            } else {
                const child: Node = {
                    state: adv.state, actor: adv.actor, mask: adv.mask,
                    edges: [], expanded: false,
                };
                bestEdge.child = child;
                leafValue = await expand(child, ctx);
                leafActor = child.actor;
            }
            break;
        }

        node = bestEdge.child;
        if (node.state.winner) {
            leafActor = node.actor;
            leafValue = node.state.winner === node.actor ? 1 : 0;
            break;
        }
        if (!node.expanded) {
            leafValue = await expand(node, ctx);
            leafActor = node.actor;
            break;
        }
    }

    // Backup (2-player: opponent perspective is 1 − v).
    for (const { node: n, edge } of path) {
        const v = n.actor === leafActor ? leafValue : 1 - leafValue;
        edge.visits += 1;
        edge.valueSum += v;
    }
}

// ── Action selection from a visit distribution ────────────────────────────────

/**
 * Pick an action index from the visit distribution: proportional sampling for
 * the first `tempMoves` decisions of a game (exploration / data diversity),
 * argmax afterwards.
 */
export function pickFromVisits(
    visits: Float32Array,
    decisionIdx: number,
    tempMoves: number,
    rand: () => number,
): number {
    let best = 0, bestV = -1;
    for (let i = 0; i < visits.length; i++) {
        if (visits[i] > bestV) { bestV = visits[i]; best = i; }
    }
    if (decisionIdx >= tempMoves) return best;

    let r = rand();
    for (let i = 0; i < visits.length; i++) {
        r -= visits[i];
        if (visits[i] > 0 && r <= 0) return i;
    }
    return best;
}
