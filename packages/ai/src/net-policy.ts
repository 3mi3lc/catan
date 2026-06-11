/**
 * net-policy.ts — ONNX-backed CatanNet policy for inference.
 *
 * Loads catan_net.onnx (produced by train_bc.py / train_ppo.py) and wraps it
 * as an AsyncPolicy. onnxruntime-node only exposes the Promise-based
 * `session.run()` — there is no synchronous inference API — so `decide`
 * returns a Promise and games must be driven with `playMatchAsync`.
 *
 * Usage:
 *   import { loadNetPolicy } from '../src/net-policy';
 *   const policy = await loadNetPolicy('models/bc_v1/catan_net.onnx', bi);
 *   const action = await policy.decide(state, player);
 */

import * as ort from 'onnxruntime-node';
import {
    encodeObservation, legalMask, indexToAction,
    type BoardIndex,
} from './encoding';
import { discardAction } from './heuristics';
import type { AsyncPolicy } from './policy';
import type { GameState, PlayerId } from '@catan/core';

// ── Public API ────────────────────────────────────────────────────────────────

export interface NetPolicyOptions {
    /**
     * 'argmax' — always pick the highest-probability legal action (deterministic).
     * 'sample' — sample from the softmax distribution (adds exploration, useful
     *            during self-play data collection).
     * Default: 'argmax'
     */
    mode?: 'argmax' | 'sample';

    /**
     * Temperature applied before softmax when mode = 'sample'.
     * < 1 sharpens (more deterministic), > 1 flattens (more random).
     * Default: 1.0
     */
    temperature?: number;

    /** Display name for this policy instance. Default: 'net' */
    name?: string;
}

/**
 * Loads the ONNX model and returns an AsyncPolicy (decide() awaits inference).
 */
export async function loadNetPolicy(
    modelPath: string,
    bi: BoardIndex,
    options: NetPolicyOptions = {},
): Promise<AsyncPolicy> {
    const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
    });

    const {
        mode        = 'argmax',
        temperature = 1.0,
        name        = 'net',
    } = options;

    return {
        name,
        async decide(state: GameState, player: PlayerId) {
            // Discard phase: net was not trained on this; use heuristic.
            if (state.phase === 'discard') return discardAction(state, player);

            const mask = legalMask(state, player, bi);
            if (!mask.some(v => v === 1)) return { type: 'endTurn' } as const;

            // Build input tensor and run the forward pass (Promise-based;
            // typically < 1 ms for this network size).
            const obs    = encodeObservation(state, player, bi);
            const tensor = new ort.Tensor('float32', obs, [1, obs.length]);

            const results = await session.run({ obs: tensor });
            const logits  = results['policy_logits'].data as Float32Array;

            // Mask illegal actions to −∞ so they get zero probability.
            const masked = new Float32Array(logits.length);
            for (let i = 0; i < logits.length; i++) {
                masked[i] = mask[i] === 1 ? logits[i] : -Infinity;
            }

            const actionIdx = mode === 'sample'
                ? sampleFromLogits(masked, temperature)
                : argmaxFinite(masked);

            return indexToAction(actionIdx, state, player, bi);
        },
    };
}

// ── Sampling helpers ──────────────────────────────────────────────────────────

function argmaxFinite(logits: Float32Array): number {
    let best = -1, bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
    }
    return best;
}

function sampleFromLogits(logits: Float32Array, temperature: number): number {
    // Numerically stable softmax with temperature.
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (isFinite(logits[i]) && logits[i] > max) max = logits[i];
    }

    const probs = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        if (isFinite(logits[i])) {
            probs[i] = Math.exp((logits[i] - max) / temperature);
            sum += probs[i];
        }
    }

    let r = Math.random() * sum;
    for (let i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (r <= 0) return i;
    }
    return argmaxFinite(logits);
}