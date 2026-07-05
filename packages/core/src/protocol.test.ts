import { describe, it, expect } from 'vitest';
import { generateBoard, type Rng } from './board-generator';
import { initialGameState } from './setup';
import { toClientView } from './protocol';
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
];

const P0 = asPlayerId('p0');
const P1 = asPlayerId('p1');

function newGame() {
    const state = initialGameState(generateBoard(mulberry32(42)), SEEDS, 7);
    return {
        ...state,
        players: {
            ...state.players,
            [P0]: { ...state.players[P0], resources: { brick: 2, lumber: 1, wool: 0, grain: 3, ore: 0 } },
            [P1]: { ...state.players[P1], resources: { brick: 0, lumber: 0, wool: 1, grain: 0, ore: 4 } },
        },
    };
}

describe('toClientView', () => {
    it('keeps the viewer\'s own hand exact and collapses everyone else to a count', () => {
        const state = newGame();
        const view = toClientView(state, P0);

        expect(view.players[P0].resources).toEqual({ brick: 2, lumber: 1, wool: 0, grain: 3, ore: 0 });
        expect(view.players[P0].handSize).toBe(6);

        expect(view.players[P1].resources).toBeNull();
        expect(view.players[P1].handSize).toBe(5);
    });

    it('redacts every seat for a spectator (viewerSeat: null)', () => {
        const state = newGame();
        const view = toClientView(state, null);

        expect(view.players[P0].resources).toBeNull();
        expect(view.players[P1].resources).toBeNull();
        expect(view.players[P0].handSize).toBe(6);
        expect(view.players[P1].handSize).toBe(5);
    });

    it('drops the raw rng seed and dev-deck order, keeping only a count', () => {
        const state = newGame();
        const view = toClientView(state, P0) as Record<string, unknown>;

        expect(view.rng).toBeUndefined();
        expect(view.devDeck).toBeUndefined();
        expect((view as unknown as { devDeckCount: number }).devDeckCount).toBe(state.devDeck.length);
    });

    it('reveals every hand once the game has a winner', () => {
        const state = { ...newGame(), winner: P0 };
        const view = toClientView(state, P1); // viewer is the loser, not the winner

        expect(view.players[P0].resources).toEqual({ brick: 2, lumber: 1, wool: 0, grain: 3, ore: 0 });
        expect(view.players[P1].resources).toEqual({ brick: 0, lumber: 0, wool: 1, grain: 0, ore: 4 });
    });
});
