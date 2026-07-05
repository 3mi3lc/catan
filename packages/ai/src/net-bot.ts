/**
 * net-bot.ts — server-side AI opponent (Node/onnxruntime-node port of
 * packages/web/src/bot.ts's in-browser `Bot`).
 *
 * Same decision flow as the browser bot, so AI seats play identically
 * whether a game runs locally in a browser tab (hotseat.ts) or server-side
 * for a networked room (packages/server): heuristicTradeAction owns every
 * trade decision, the strong NO-TRADE net (encodeObservationNoTrade /
 * legalMaskNoTrade) drives non-trade actions, and the 'mcts' level wraps it
 * in a PUCT search. Kept as its own file (not shared with bot.ts) because the
 * two platforms' ONNX runtimes (onnxruntime-web vs onnxruntime-node) have
 * separate session/tensor types — there are only two call sites, so a shared
 * abstraction isn't worth it.
 *
 * Deliberately excluded from index.ts's barrel export: onnxruntime-node is a
 * native Node module and must never end up in @catan/web's browser bundle.
 * Reachable via the "./net-bot" subpath in package.json's `exports` map.
 */

import * as ort from 'onnxruntime-node';
import {
    buildBoardIndex, encodeObservationNoTrade, legalMaskNoTrade, indexToAction,
    ACT_SIZE, ACT_SIZE_NOTRADE, MAX_SEATS,
    type BoardIndex,
} from './encoding';
import { runMcts, type NetEvaluator } from './mcts';
import { discardAction } from './heuristics';
import { heuristicTradeAction } from './trade-heuristic';
import type { GameState, PlayerId, Action } from '@catan/core';

export type BotLevel = 'net' | 'mcts';

function argmaxMasked(logits: Float32Array, mask: Uint8Array): number {
    let best = -1, bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (mask[i] === 1 && logits[i] > bestVal) { bestVal = logits[i]; best = i; }
    }
    return best;
}

export class ServerBot {
    private constructor(
        private readonly session: ort.InferenceSession,
    ) {}

    private bi: BoardIndex | null = null;

    static async load(modelPath: string): Promise<ServerBot> {
        const session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['cpu'],
        });
        return new ServerBot(session);
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
