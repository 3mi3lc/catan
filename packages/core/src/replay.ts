import { generateBoard } from './board-generator';
import { initialGameState } from './setup';
import { applyMove } from './engine';
import { rngStep } from './rules';
import { asTileId, PlayerId } from './ids';
import { GameState } from './state';
import { Action, GameEvent } from './actions';

export interface MoveRecord {
    player: string;
    phase: string;
    action: Action;
}

export interface GameArchive {
    seed: number;
    policyNames: string[];
    winnerSeat: number | null;
    turns: number;
    moves: MoveRecord[];
    // Defaults to 1 (bounded self-play search) if absent, for back-compat
    // with archives recorded before this field existed. Interactive games
    // (hotseat/online) record a much larger cap here — see GameState.maxOffersPerTurn.
    maxOffersPerTurn?: number;
}

const rngFromSeed = (seed: number) => {
    let s = seed >>> 0;
    return () => { const r = rngStep(s); s = r.next; return r.value; };
};

const COLORS = ['red', 'blue', 'white', 'orange'] as const;

export interface Replay {
    archive: GameArchive;
    length: number;
    at(i: number): GameState;
    // Events produced by move i (the transition from state i to state i+1).
    // Out-of-range i (including the synthetic "before any move" state 0) returns [].
    eventsAt(i: number): GameEvent[];
}

export function loadReplay(archive: GameArchive): Replay {
    const n     = archive.policyNames.length;
    const board = generateBoard(rngFromSeed(archive.seed));
    const seats = COLORS.slice(0, n).map((c, i) => ({ id: `p${i}`, name: archive.policyNames[i], color: c }));

    const states: GameState[] = [];
    const events: GameEvent[][] = [];
    let state = initialGameState(board, seats, archive.seed, archive.maxOffersPerTurn ?? 1);
    states.push(state);

    for (const { player, action } of archive.moves) {
        const res = applyMove(state, { player: player as PlayerId, action });
        if (!res.ok) throw new Error(`Replay error at move ${states.length}: ${res.error}`);
        state = res.state;
        states.push(state);
        events.push(res.events);
    }

    return {
        archive,
        length: states.length,
        at: (i) => states[Math.max(0, Math.min(i, states.length - 1))],
        eventsAt: (i) => events[i] ?? [],
    };
}

// A tile id that can never match a real board tile — used to compute "what
// would this roll have produced with no robber blocking anything" (see
// computeGameStats's robber-blocked-production stat in stats.ts).
export const NO_ROBBER_TILE = asTileId('__none__');
