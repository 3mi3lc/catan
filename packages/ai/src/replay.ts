import { generateBoard, initialGameState, applyMove, rngStep, type GameState, type PlayerId, type Action } from '@catan/core';

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
}

export function loadReplay(archive: GameArchive): Replay {
    const n     = archive.policyNames.length;
    const board = generateBoard(rngFromSeed(archive.seed));
    const seats = COLORS.slice(0, n).map((c, i) => ({ id: `p${i}`, name: c, color: c }));

    const states: GameState[] = [];
    let state = initialGameState(board, seats, archive.seed);
    states.push(state);

    for (const { player, action } of archive.moves) {
        const res = applyMove(state, { player: player as PlayerId, action });
        if (!res.ok) throw new Error(`Replay error at move ${states.length}: ${res.error}`);
        state = res.state;
        states.push(state);
    }

    return {
        archive,
        length: states.length,
        at: (i) => states[Math.max(0, Math.min(i, states.length - 1))],
    };
}