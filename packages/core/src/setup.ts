import { Board } from './board';
import { PlayerId, asPlayerId } from './ids';
import { GameState, Player, PlayerColor, DevCard, ResourceCounts } from './state';
import { rngStep, zeroResources } from './rules';

export interface PlayerSeed {
    id: string;
    name: string;
    color: PlayerColor;
}

// The official development-card deck: 14 knights, 5 victory points, and two
// each of the three progress cards (25 total).
function fullDevDeck(): DevCard[] {
    return [
        ...Array<DevCard>(14).fill('knight'),
        ...Array<DevCard>(5).fill('victoryPoint'),
        ...Array<DevCard>(2).fill('roadBuilding'),
        ...Array<DevCard>(2).fill('yearOfPlenty'),
        ...Array<DevCard>(2).fill('monopoly'),
    ];
}

// A new game ready for initial placement. `seed` drives everything random from
// here on: it shuffles the dev deck and seeds the in-game PRNG as one
// continuous stream, so the same seed (with the same board) replays exactly.
// Pass a fixed integer for reproducible games / self-play; omit it for a random
// one. (Seed the board separately via generateBoard for a fully fixed setup.)
export function initialGameState(
    board: Board,
    seeds: PlayerSeed[],
    seed: number = (Math.random() * 0x1_0000_0000) >>> 0,
    maxOffersPerTurn: number = 1,
): GameState {
    if (seeds.length < 2) throw new Error('Need at least two players');

    const players: Record<PlayerId, Player> = {};
    const turnOrder: PlayerId[] = [];
    for (const s of seeds) {
        const id = asPlayerId(s.id);
        turnOrder.push(id);
        players[id] = {
            id,
            name: s.name,
            color: s.color,
            resources: zeroResources(),
            devCards: [],
            pendingDevCards: [],
            playedKnights: 0,
            supply: { settlements: 5, cities: 4, roads: 15 },
        };
    }

    const desert = Object.values(board.tiles).find((t) => t.terrain === 'desert');
    if (!desert) throw new Error('Board has no desert for the robber');

    // Draw from a moving cursor so the deck shuffle and the in-game stream are one
    // deterministic sequence; the advanced cursor is stored as the game's rng.
    let cursor = seed >>> 0;
    const draw = (): number => {
        const { value, next } = rngStep(cursor);
        cursor = next;
        return value;
    };
    const devDeck = fullDevDeck();
    for (let i = devDeck.length - 1; i > 0; i--) {
        const j = Math.floor(draw() * (i + 1));
        [devDeck[i], devDeck[j]] = [devDeck[j], devDeck[i]];
    }

    const bank: ResourceCounts = { brick: 19, lumber: 19, wool: 19, grain: 19, ore: 19 };

    return {
        board,
        players,
        turnOrder,
        currentPlayer: turnOrder[0],
        phase: 'setupSettlement',
        dice: null,
        robber: desert.id,
        buildings: {},
        roads: {},
        bank,
        devDeck,
        longestRoad: null,
        largestArmy: null,
        winner: null,
        rng: cursor,
        setup: { placed: 0, lastSettlement: null },
        devCardPlayedThisTurn: false,
        pendingDiscards: {},
        negotiation: null,
        draftOffer: null,
        tradesThisTurn: 0,
        maxOffersPerTurn,
    };
}