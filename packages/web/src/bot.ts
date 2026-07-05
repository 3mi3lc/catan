/**
 * bot.ts — in-browser AI opponent for the hotseat game.
 *
 * The model (public/models/catan_net_4p.onnx) is the strong NO-TRADE 4-player
 * net: training it to trade regressed it to baseline, so instead the net plays
 * the game and a hand-built heuristic owns every trade decision. Concretely:
 *   • heuristicTradeAction handles all trade steps (propose / reply / arbitrate);
 *   • the net runs on the trade-free observation (encodeObservationNoTrade, 1603)
 *     and chooses among non-trade actions only (legalMaskNoTrade, 376);
 *   • runMcts auto-resolves trades inside the search via the same heuristic.
 *
 * Levels:
 *   'net'  — raw policy argmax, ~instant (one forward pass per move).
 *   'mcts' — PUCT search (value-head-dominated leaf eval); trades resolved by
 *            heuristic inside the tree. Stronger, slower.
 */

import * as ort from 'onnxruntime-web';
import {
    buildBoardIndex, encodeObservationNoTrade, legalMaskNoTrade, indexToAction,
    ACT_SIZE, ACT_SIZE_NOTRADE, MAX_SEATS,
    runMcts, discardAction, heuristicTradeAction,
    type BoardIndex, type NetEvaluator,
} from '@catan/ai';
import type { GameState, PlayerId, Action } from '@catan/core';

export type BotLevel = 'net' | 'mcts';

function argmaxMasked(logits: Float32Array, mask: Uint8Array): number {
    let best = -1, bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (mask[i] === 1 && logits[i] > bestVal) { bestVal = logits[i]; best = i; }
    }
    return best;
}

export class Bot {
    private constructor(
        private readonly session: ort.InferenceSession,
    ) {}

    private bi: BoardIndex | null = null;

    static async load(modelUrl: string): Promise<Bot> {
        // Single-threaded WASM: avoids the SharedArrayBuffer/COOP-COEP
        // requirements, and one thread is plenty for a 240k-param net.
        ort.env.wasm.numThreads = 1;
        // Load the WASM runtime from the CDN, version-locked to the installed
        // package. Local serving doesn't work under vite dev: its middleware
        // rewrites dynamic import() of public-dir .mjs files into asset-URL
        // shims ("?import"), which breaks ort's loader.
        ort.env.wasm.wasmPaths =
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
        const session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
        });
        return new Bot(session);
    }

    private async logits(state: GameState, player: PlayerId): Promise<{
        logits: Float32Array; value: Float32Array;
    }> {
        const obs = encodeObservationNoTrade(state, player, this.bi!);   // 1603, no trade block
        const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
        const out = await this.session.run({ obs: tensor });
        return {
            logits: out['policy_logits'].data as Float32Array,   // 376 non-trade actions
            value: out['value'].data as Float32Array,            // length MAX_SEATS (seat-relative logits)
        };
    }

    private evaluator(): NetEvaluator {
        return {
            evaluate: async (state, player, mask) => {
                const { logits, value } = await this.logits(state, player);
                // Net covers only the non-trade actions (0..375); priors are
                // padded to ACT_SIZE with zeros on the (unsearched) trade slots.
                let max = -Infinity;
                for (let i = 0; i < ACT_SIZE_NOTRADE; i++) {
                    if (mask[i] === 1 && logits[i] > max) max = logits[i];
                }
                const priors = new Float32Array(ACT_SIZE);
                let sum = 0;
                for (let i = 0; i < ACT_SIZE_NOTRADE; i++) {
                    if (mask[i] === 1) { priors[i] = Math.exp(logits[i] - max); sum += priors[i]; }
                }
                if (sum > 0) for (let i = 0; i < ACT_SIZE_NOTRADE; i++) priors[i] /= sum;

                // Softmax the per-seat value logits → win-prob vector (relative to player).
                const winProbs = new Float32Array(MAX_SEATS);
                let vmax = -Infinity;
                for (let i = 0; i < MAX_SEATS; i++) if (value[i] > vmax) vmax = value[i];
                let vsum = 0;
                for (let i = 0; i < MAX_SEATS; i++) { winProbs[i] = Math.exp(value[i] - vmax); vsum += winProbs[i]; }
                if (vsum > 0) for (let i = 0; i < MAX_SEATS; i++) winProbs[i] /= vsum;
                return { priors, winProbs };
            },
        };
    }

    async decide(
        state: GameState,
        player: PlayerId,
        level: BotLevel,
        sims = 96,
    ): Promise<Action> {
        this.bi ??= buildBoardIndex(state.board);

        if (state.phase === 'discard') return discardAction(state, player);

        // Every trade decision (propose / reply / arbitrate) is handled by the
        // heuristic; the net never sees trade actions.
        const trade = heuristicTradeAction(state, player);
        if (trade) return trade;

        const mask = legalMaskNoTrade(state, player, this.bi);   // non-trade actions only
        let legalCount = 0, onlyIdx = -1;
        for (let i = 0; i < mask.length; i++) {
            if (mask[i] === 1) { legalCount++; onlyIdx = i; }
        }
        if (legalCount === 0) return { type: 'endTurn' };
        if (legalCount === 1) return indexToAction(onlyIdx, state, player, this.bi);

        if (level === 'net') {
            const { logits } = await this.logits(state, player);
            return indexToAction(argmaxMasked(logits, mask), state, player, this.bi);
        }

        // 'mcts': trades auto-resolve inside the search via the heuristic.
        const { visits } = await runMcts(state, player, mask, this.bi, this.evaluator(), {
            sims,
            valueMix: 0.85,
            rolloutCap: 150,
        });
        let best = 0, bestV = -1;
        for (let i = 0; i < visits.length; i++) {
            if (visits[i] > bestV) { bestV = visits[i]; best = i; }
        }
        return indexToAction(best, state, player, this.bi);
    }
}
