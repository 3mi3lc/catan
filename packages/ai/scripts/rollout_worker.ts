/// <reference types="node" />
/**
 * rollout_worker.ts — persistent self-play worker for PPO training.
 *
 * Spawned (in a pool) by packages/training/train_ppo.py and reused for every
 * game. Each worker loads the current policy as ONNX and plays games fully
 * in-process — no per-step IPC. The only traffic is one "start" command per
 * game and one trajectory message back, so throughput is bound by engine +
 * inference speed, not pipe round-trips.
 *
 * Protocol (Python → worker), line-delimited JSON on stdin:
 *   {"cmd":"model","path":"<abs path to .onnx>"}
 *       (Re)load the policy. Sent once per training iteration. Worker replies
 *       {"t":"model_ok"}.
 *   {"cmd":"start","seed":123,"opponent":"self"|"greedy","netSeat":0,"mode":"sample"|"argmax"}
 *       Play one game. With "self" both seats are net-controlled; with
 *       "greedy" only `netSeat` is, the other seat plays the frozen greedy
 *       heuristic. "sample" draws from the masked softmax (training);
 *       "argmax" is deterministic (eval).
 *   {"cmd":"stop"}
 *       Exit cleanly.
 *
 * Protocol (worker → Python), one message per finished game:
 *   {"t":"end","winner":0|1|null,"turns":T,"steps":S,"n":K,
 *    "obs":"<b64 f32[K*1328]>","mask":"<b64 u8[K*300]>",
 *    "actions":"<b64 u16[K]>","seats":"<b64 u8[K]>"}
 *       K = number of recorded net decisions. Python recomputes log-probs and
 *       values for these in one batched torch pass (same weights → identical
 *       up to float noise), so neither needs to cross the pipe.
 *
 * Decisions that are not recorded (and produce no training samples): the
 * discard phase (heuristic, same as inference) and forced moves where exactly
 * one action is legal (no choice → no gradient signal).
 */

import * as readline from 'readline';
import * as ort from 'onnxruntime-node';
import {
    generateBoard, initialGameState, applyMove, rngStep,
    type GameState, type Action, type PlayerId, type PlayerColor,
} from '@catan/core';
import {
    buildBoardIndex, encodeObservation, legalMask, indexToAction, ACT_SIZE,
    type BoardIndex,
} from '../src/encoding';
import { discardAction } from '../src/heuristics';
import { greedyPolicy } from '../src/greedy-policy';
import { searchPolicy } from '../src/search-policy';
import { runMcts, pickFromVisits, type NetEvaluator } from '../src/mcts';
import type { Policy } from '../src/policy';
import type { GameState as GS, PlayerId as PID } from '@catan/core';

const COLORS: PlayerColor[] = ['red', 'blue'];
const MAX_STEPS = 6000;

function mkRng(seed: number) {
    let s = (seed >>> 0) || 1;
    return () => { const r = rngStep(s); s = r.next; return r.value; };
}

// Seeded uniform RNG for action sampling, independent of the engine RNG so a
// game replays identically given (model, seed, mode).
function mulberry32(seed: number) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ── stdin line reader (await-able) ────────────────────────────────────────────

const rl = readline.createInterface({ input: process.stdin, terminal: false });
const pending: string[] = [];
const waiters: ((line: string) => void)[] = [];
rl.on('line', (line) => {
    const w = waiters.shift();
    if (w) w(line); else pending.push(line);
});
rl.on('close', () => process.exit(0));

function readLine(): Promise<string> {
    const line = pending.shift();
    if (line !== undefined) return Promise.resolve(line);
    return new Promise((resolve) => waiters.push(resolve));
}

function send(msg: unknown): void {
    process.stdout.write(JSON.stringify(msg) + '\n');
}

function b64(view: { buffer: ArrayBufferLike; byteOffset: number; byteLength: number }): string {
    return Buffer.from(view.buffer, view.byteOffset, view.byteLength).toString('base64');
}

// ── Action selection ──────────────────────────────────────────────────────────

function argmaxMasked(logits: Float32Array, mask: Uint8Array): number {
    let best = -1, bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (mask[i] === 1 && logits[i] > bestVal) { bestVal = logits[i]; best = i; }
    }
    return best;
}

function sampleMasked(
    logits: Float32Array, mask: Uint8Array, rand: () => number, temp: number,
): number {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (mask[i] === 1 && logits[i] > max) max = logits[i];
    }
    const probs = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        if (mask[i] === 1) { probs[i] = Math.exp((logits[i] - max) / temp); sum += probs[i]; }
    }
    let r = rand() * sum;
    for (let i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (probs[i] > 0 && r <= 0) return i;
    }
    return argmaxMasked(logits, mask);
}

// ── One game ──────────────────────────────────────────────────────────────────

interface StartCmd {
    cmd: 'start';
    seed: number;
    /**
     * 'self'   — both seats net-controlled
     * 'greedy' — frozen greedy heuristic on the non-net seat
     * 'search' — hand-built flat-MC searchPolicy (eval benchmark opponent)
     * 'net2'   — a second ONNX model (loaded via {"cmd":"model2"}), played
     *            raw argmax — for checkpoint-ladder evals
     */
    opponent: 'self' | 'greedy' | 'search' | 'net2';
    netSeat?: number;            // which seat the net controls vs an opponent
    searchK?: number;            // 'search': rollouts per candidate, default 10
    searchCap?: number;          // 'search': rollout ply cap, default 1200
    mode?: 'sample' | 'argmax' | 'mcts';  // default 'sample'
    temp?: number;               // sampling temperature ('sample' mode), default 1.0
    sims?: number;               // MCTS simulations per decision ('mcts'), default 96
    tempMoves?: number;          // 'mcts': sample ∝ visits for first N decisions
                                 // per seat, argmax after; default 12
    noise?: boolean;             // 'mcts': Dirichlet root noise (self-play), default true
    valueMix?: number;           // 'mcts': P(net-value leaf) vs rollout leaf, default 0.25
    rolloutCap?: number;         // 'mcts': leaf rollout ply cap, default 600
    /**
     * Playout-cap randomization (KataGo): probability a decision gets the
     * full `sims` search AND is recorded as a training sample. Other
     * decisions use a cheap `fastSims` search just to keep the game
     * on-policy, and are not recorded. ~3x faster collection at near-equal
     * policy-target quality. Default 1 (every decision full + recorded).
     */
    fullProb?: number;
    fastSims?: number;           // sims for non-recorded decisions, default 12
    noiseEps?: number;           // Dirichlet root-noise weight, default 0.1
    /**
     * Policy-target pruning (KataGo): actions with fewer than
     * targetPrune × max_visits are zeroed in the RECORDED target (and the
     * target renormalised). At low sim counts the visit tail is mostly
     * Dirichlet noise + prior fuzz; training on it teaches the net to be
     * flat, degrading its argmax play even as the loss falls. Default 0.1.
     */
    targetPrune?: number;
}

/** Zero the sub-threshold tail of a visit distribution and renormalise. */
function pruneTarget(visits: Float32Array, minFrac: number): Float32Array {
    let max = 0;
    for (let i = 0; i < visits.length; i++) if (visits[i] > max) max = visits[i];
    const out = new Float32Array(visits.length);
    let sum = 0;
    const thresh = max * minFrac;
    for (let i = 0; i < visits.length; i++) {
        if (visits[i] >= thresh) { out[i] = visits[i]; sum += visits[i]; }
    }
    if (sum > 0) for (let i = 0; i < out.length; i++) out[i] /= sum;
    return out;
}

// ── ONNX evaluator for MCTS ───────────────────────────────────────────────────

function makeEvaluator(session: ort.InferenceSession, bi: BoardIndex): NetEvaluator {
    return {
        async evaluate(state: GS, player: PID, mask: Uint8Array) {
            const obs    = encodeObservation(state, player, bi);
            const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
            const out    = await session.run({ obs: tensor });
            const logits = out['policy_logits'].data as Float32Array;
            const rawVal = (out['value'].data as Float32Array)[0];

            // Masked softmax → priors.
            let max = -Infinity;
            for (let i = 0; i < ACT_SIZE; i++) {
                if (mask[i] === 1 && logits[i] > max) max = logits[i];
            }
            const priors = new Float32Array(ACT_SIZE);
            let sum = 0;
            for (let i = 0; i < ACT_SIZE; i++) {
                if (mask[i] === 1) { priors[i] = Math.exp(logits[i] - max); sum += priors[i]; }
            }
            if (sum > 0) for (let i = 0; i < ACT_SIZE; i++) priors[i] /= sum;

            return { priors, winProb: 1 / (1 + Math.exp(-rawVal)) };
        },
    };
}

async function playGame(
    cmd: StartCmd,
    bi: BoardIndex,
    session: ort.InferenceSession,
    session2: ort.InferenceSession | null,
): Promise<void> {
    const board = generateBoard(mkRng(cmd.seed));
    const seats = COLORS.map((c, i) => ({ id: `p${i}` as PlayerId, name: c, color: c }));
    let state = initialGameState(board, seats, cmd.seed);

    // Opponent for the non-net seat (sync policies; 'net2' handled inline).
    const opponentPolicy: Policy | null =
        cmd.opponent === 'greedy' ? greedyPolicy()
        : cmd.opponent === 'search' ? searchPolicy({
            rollouts:   cmd.searchK ?? 10,
            rolloutCap: cmd.searchCap ?? 1200,
            seed:       cmd.seed ^ 0x5EEDFACE,
        })
        : null;
    if (cmd.opponent === 'net2' && !session2) {
        throw new Error('opponent "net2" requires a prior {"cmd":"model2"}');
    }
    const rand   = mulberry32(cmd.seed ^ 0x9E3779B9);
    const mode   = cmd.mode ?? 'sample';
    const temp   = cmd.temp ?? 1.0;
    const sims   = cmd.sims ?? 96;
    const tempMoves = cmd.tempMoves ?? 12;
    const evaluator = mode === 'mcts' ? makeEvaluator(session, bi) : null;
    const isNetSeat = (seat: number) =>
        cmd.opponent === 'self' || seat === (cmd.netSeat ?? 0);

    // Recorded trajectory (net decisions with > 1 legal action only).
    const recObs:  Float32Array[] = [];
    const recMask: Uint8Array[]   = [];
    const recAct:  number[]       = [];
    const recSeat: number[]       = [];
    const recPol:  Float32Array[] = [];     // mcts visit distributions
    const decisionCount = [0, 0];           // per-seat, for temperature schedule

    let steps = 0;
    let turns = 0;

    for (; steps < MAX_STEPS && !state.winner; steps++) {
        const actor: PlayerId = state.phase === 'discard'
            ? (Object.keys(state.pendingDiscards)[0] as PlayerId)
            : state.currentPlayer;
        const seat = state.turnOrder.indexOf(actor);

        let action: Action;

        if (state.phase === 'discard') {
            action = discardAction(state, actor);          // heuristic, never trained
        } else if (!isNetSeat(seat)) {
            if (opponentPolicy) {
                action = opponentPolicy.decide(state, actor);   // greedy / search
            } else {
                // 'net2': second model, raw argmax.
                const mask2 = legalMask(state, actor, bi);
                let any = false;
                for (let i = 0; i < mask2.length; i++) {
                    if (mask2[i] === 1) { any = true; break; }
                }
                if (!any) {
                    action = { type: 'endTurn' };
                } else {
                    const obs2    = encodeObservation(state, actor, bi);
                    const tensor2 = new ort.Tensor('float32', obs2, [1, obs2.length]);
                    const out2    = await session2!.run({ obs: tensor2 });
                    const logits2 = out2['policy_logits'].data as Float32Array;
                    action = indexToAction(argmaxMasked(logits2, mask2), state, actor, bi);
                }
            }
        } else {
            const mask = legalMask(state, actor, bi);
            let legalCount = 0, onlyIdx = -1;
            for (let i = 0; i < mask.length; i++) {
                if (mask[i] === 1) { legalCount++; onlyIdx = i; }
            }

            if (legalCount === 0) {
                action = { type: 'endTurn' };
            } else if (legalCount === 1) {
                // Forced move (e.g. rollDice): no decision to learn from.
                action = indexToAction(onlyIdx, state, actor, bi);
            } else if (mode === 'mcts') {
                // Playout-cap randomization: full search + record on a random
                // subset of decisions; cheap search (no noise, no sample) on
                // the rest to keep the game on-policy.
                const full = rand() < (cmd.fullProb ?? 1);
                const { visits } = await runMcts(state, actor, mask, bi, evaluator!, {
                    sims: full ? sims : (cmd.fastSims ?? 12),
                    rand,
                    rootNoise: (full && (cmd.noise ?? true))
                        ? { alpha: 0.3, eps: cmd.noiseEps ?? 0.1 } : undefined,
                    valueMix:   cmd.valueMix ?? 0.25,
                    rolloutCap: cmd.rolloutCap ?? 600,
                });
                const idx = pickFromVisits(
                    visits, decisionCount[seat]++, tempMoves, rand);

                if (full) {
                    recObs.push(encodeObservation(state, actor, bi));
                    recMask.push(mask);
                    recAct.push(idx);
                    recSeat.push(seat);
                    recPol.push(pruneTarget(visits, cmd.targetPrune ?? 0.1));
                }
                action = indexToAction(idx, state, actor, bi);
            } else {
                const obs    = encodeObservation(state, actor, bi);
                const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
                const out    = await session.run({ obs: tensor });
                const logits = out['policy_logits'].data as Float32Array;

                const idx = mode === 'sample'
                    ? sampleMasked(logits, mask, rand, temp)
                    : argmaxMasked(logits, mask);

                recObs.push(obs);
                recMask.push(mask);
                recAct.push(idx);
                recSeat.push(seat);
                action = indexToAction(idx, state, actor, bi);
            }
        }

        if (action.type === 'endTurn') turns++;
        const res = applyMove(state, { player: actor, action });
        if (!res.ok) {
            // Mask guarantees legality, so this indicates an encoding bug —
            // surface it loudly rather than training on a corrupt game.
            process.stderr.write(
                `illegal move (${res.error}) seed=${cmd.seed} step=${steps}: ${JSON.stringify(action)}\n`,
            );
            send({ t: 'end', winner: null, turns, steps, n: 0 });
            return;
        }
        state = res.state;
    }

    const winnerSeat = state.winner ? state.turnOrder.indexOf(state.winner) : null;

    // Pack the trajectory into contiguous binary blocks.
    const n = recAct.length;
    const obsAll  = new Float32Array(n * (recObs[0]?.length ?? 0));
    const maskAll = new Uint8Array(n * (recMask[0]?.length ?? 0));
    for (let i = 0; i < n; i++) {
        obsAll.set(recObs[i],  i * recObs[i].length);
        maskAll.set(recMask[i], i * recMask[i].length);
    }

    // MCTS mode additionally ships the visit distributions (training targets).
    let polB64: string | undefined;
    if (recPol.length === n && n > 0) {
        const polAll = new Float32Array(n * ACT_SIZE);
        for (let i = 0; i < n; i++) polAll.set(recPol[i], i * ACT_SIZE);
        polB64 = b64(polAll);
    }

    send({
        t: 'end', winner: winnerSeat, turns, steps, n,
        obs:     b64(obsAll),
        mask:    b64(maskAll),
        actions: b64(Uint16Array.from(recAct)),
        seats:   b64(Uint8Array.from(recSeat)),
        ...(polB64 !== undefined ? { policy: polB64 } : {}),
    });
}

// ── Main loop ─────────────────────────────────────────────────────────────────

type Cmd = StartCmd
    | { cmd: 'model'; path: string }
    | { cmd: 'model2'; path: string }
    | { cmd: 'stop' };

// One thread per session: the nets are small and many workers run in
// parallel — the default (one pool per session sized to ALL cores)
// oversubscribes the CPU catastrophically.
const SESSION_OPTS = {
    executionProviders: ['cpu'],
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
} as const;

async function main(): Promise<void> {
    // Topology is identical for all radius-2 boards; build the index once.
    const bi = buildBoardIndex(generateBoard(mkRng(1)));
    let session: ort.InferenceSession | null = null;
    let session2: ort.InferenceSession | null = null;   // 'net2' opponent
    send({ t: 'ready' });

    for (;;) {
        const msg = JSON.parse(await readLine()) as Cmd;
        if (msg.cmd === 'stop') break;
        if (msg.cmd === 'model') {
            session = await ort.InferenceSession.create(msg.path, SESSION_OPTS);
            send({ t: 'model_ok' });
        } else if (msg.cmd === 'model2') {
            session2 = await ort.InferenceSession.create(msg.path, SESSION_OPTS);
            send({ t: 'model2_ok' });
        } else if (msg.cmd === 'start') {
            if (!session) throw new Error('received "start" before "model"');
            await playGame(msg, bi, session, session2);
        }
    }
    process.exit(0);
}

main().catch((err) => {
    process.stderr.write(`rollout_worker fatal: ${err?.stack ?? err}\n`);
    process.exit(1);
});
