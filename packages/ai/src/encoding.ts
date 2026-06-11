/**
 * encoding.ts — Fixed-size observation vector and action index for the RL agent.
 *
 * Design decisions:
 *  • Fair-information: own hand is fully visible; opponent hand size visible
 *    but not composition; dev deck order hidden. Required for online play.
 *  • Flat action space (300 slots for 2-player) with a legal mask derived
 *    from legalActions(). The net outputs 300 logits; illegal slots get −∞
 *    before softmax.
 *  • Discard and Road Building use heuristics (never passed to the net);
 *    the calling policy handles those phases before consulting the net.
 *  • BoardIndex maps vertex/edge/tile IDs → stable integer indices. The
 *    hex topology is fixed for all radius-2 boards (same 54 vertex positions
 *    every game), so the index computed for one board applies to all games.
 */

import {
    legalActions, victoryPoints, RESOURCES,
    type Board, type GameState, type PlayerId, type Action,
    type VertexId, type EdgeId, type TileId, type Resource, type DevCard, roadConnected,
} from '@catan/core';

// ── Shared constants ──────────────────────────────────────────────────────────

const DEV_TYPES: readonly DevCard[] =
    ['knight', 'victoryPoint', 'roadBuilding', 'yearOfPlenty', 'monopoly'];

const TERRAINS = ['hills', 'forest', 'pasture', 'fields', 'mountains', 'desert'] as const;

/** Token values in canonical one-hot order. Index 0 = null (desert). */
const TOKEN_ORDER: readonly (number | null)[] = [null, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12];

/** All 15 unordered resource pairs for Year of Plenty. */
export const YOP_PAIRS: readonly [Resource, Resource][] = (() => {
    const pairs: [Resource, Resource][] = [];
    for (let i = 0; i < 5; i++)
        for (let j = i; j < 5; j++)
            pairs.push([RESOURCES[i], RESOURCES[j]]);
    return pairs;
})();

/** All 20 give→receive pairs for bank trading (give ≠ receive). */
export const TRADE_PAIRS: readonly [Resource, Resource][] = (() => {
    const pairs: [Resource, Resource][] = [];
    for (const g of RESOURCES)
        for (const r of RESOURCES)
            if (g !== r) pairs.push([g, r]);
    return pairs;
})();

// ── Action index layout (2-player, 300 slots) ─────────────────────────────────
//
//   0       rollDice
//   1       endTurn
//   2       buyDevCard
//   3       playRoadBuilding  (heuristic picks road locations)
//   4-8     playMonopoly      × 5 resources
//   9-23    playYearOfPlenty  × 15 pairs
//   24-43   bankTrade         × 20 (give, receive)  giveCount from state
//   44-97   buildSettlement   × 54 vertices
//   98-151  buildCity         × 54 vertices
//   152-223 buildRoad         × 72 edges
//   224-242 moveRobber(tile, stealFrom=null)      × 19 tiles
//   243-261 moveRobber(tile, stealFrom=opponent)  × 19 tiles
//   262-280 playKnight(tile, stealFrom=null)      × 19 tiles
//   281-299 playKnight(tile, stealFrom=opponent)  × 19 tiles

export const ACT_SIZE = 300;

// ── Observation layout (1328 floats per step) ─────────────────────────────────
//
//   0-21    Own player (22)
//     0-4   resources[5] / 18
//     5-9   devCards by type[5] / 4
//    10-14  pendingDevCards[5] / 4
//    15     playedKnights / 14
//    16     supply.settlements / 5
//    17     supply.cities / 4
//    18     supply.roads / 15
//    19     victoryPoints / 10  (includes own hidden VP cards)
//    20     hasLongestRoad
//    21     hasLargestArmy
//
//   22-29   Opponent fair-info (8)
//    22     handSize / 18       (card count known; composition hidden)
//    23     playedKnights / 14
//    24     supply.settlements / 5
//    25     supply.cities / 4
//    26     supply.roads / 15
//    27     publicVP / 10       (buildings + LR + LA; NOT hidden dev-card VPs)
//    28     hasLongestRoad
//    29     hasLargestArmy
//
//   30-48   robberTile one-hot (19)
//   49-55   phase one-hot (7)
//   56-60   bank resources / 19 (5)
//   61      devDeck remaining / 25
//   62-64   longestRoad holder: none/me/opp (3)
//   65-67   largestArmy holder: none/me/opp (3)
//
//   68-769  Per-vertex × 54 × 13 = 702
//            [0-2]  owner one-hot: none/me/opp
//            [3-5]  kind one-hot:  empty/settlement/city
//            [6-12] port one-hot:  none, 3:1, 2:1-brick/lumber/wool/grain/ore
//
//   770-1111 Per-tile × 19 × 18 = 342
//            [0-5]  terrain one-hot (6)
//            [6-16] numberToken one-hot (11; index 0 = null/desert)
//            [17]   isRobber
//
//   1112-1327 Per-edge × 72 × 3 = 216
//            [0-2]  owner one-hot: none/me/opp

export const OBS_SIZE = 1328;

// ── Board index ────────────────────────────────────────────────────────────────

/**
 * Stable integer mapping for all board IDs. Build once at game start;
 * reuse for every step of that game. The sorted lexicographic order of
 * position-derived IDs is identical across all standard 19-tile boards.
 */
export interface BoardIndex {
    /** Sorted vertex IDs (length 54). */
    vertices: VertexId[];
    /** Sorted edge IDs (length 72). */
    edges: EdgeId[];
    /** Sorted tile IDs (length 19). */
    tiles: TileId[];
    vIdx: Map<VertexId, number>;
    eIdx: Map<EdgeId, number>;
    tIdx: Map<TileId, number>;
}

export function buildBoardIndex(board: Board): BoardIndex {
    const vertices = (Object.keys(board.vertices) as VertexId[]).slice().sort();
    const edges    = (Object.keys(board.edges)    as EdgeId[]).slice().sort();
    const tiles    = (Object.keys(board.tiles)    as TileId[]).slice().sort();
    return {
        vertices, edges, tiles,
        vIdx: new Map(vertices.map((id, i) => [id, i])),
        eIdx: new Map(edges.map((id, i)    => [id, i])),
        tIdx: new Map(tiles.map((id, i)    => [id, i])),
    };
}

// ── Observation encoding ───────────────────────────────────────────────────────

/**
 * Encode `state` from `player`'s perspective into a flat Float32Array.
 * Always call with the acting player (which may differ from state.currentPlayer
 * during the discard phase when multiple players owe cards).
 */
export function encodeObservation(
    state: GameState,
    player: PlayerId,
    bi: BoardIndex,
): Float32Array {
    const obs = new Float32Array(OBS_SIZE);
    let o = 0;

    const put  = (v: number)            => { obs[o++] = v; };
    const hot  = (idx: number, n: number) => { obs[o + idx] = 1; o += n; };

    const pl  = state.players[player];
    const opp = state.turnOrder.find(p => p !== player)!;
    const op  = state.players[opp];

    // ── Own player ──────────────────────────────────────────────────────────────
    for (const r of RESOURCES) put(pl.resources[r] / 18);          // 0-4
    for (const d of DEV_TYPES) put(pl.devCards.filter(c => c === d).length / 4);        // 5-9
    for (const d of DEV_TYPES) put(pl.pendingDevCards.filter(c => c === d).length / 4); // 10-14
    put(pl.playedKnights / 14);        // 15
    put(pl.supply.settlements / 5);    // 16
    put(pl.supply.cities / 4);         // 17
    put(pl.supply.roads / 15);         // 18
    put(victoryPoints(state, player) / 10); // 19  (includes own VP dev cards)
    put(state.longestRoad?.player === player ? 1 : 0); // 20
    put(state.largestArmy?.player === player ? 1 : 0); // 21

    // ── Opponent fair-info ─────────────────────────────────────────────────────
    const oppHand = RESOURCES.reduce((s, r) => s + op.resources[r], 0);
    put(oppHand / 18);                 // 22  hand SIZE visible; not composition
    put(op.playedKnights / 14);        // 23
    put(op.supply.settlements / 5);    // 24
    put(op.supply.cities / 4);         // 25
    put(op.supply.roads / 15);         // 26
    const oppPublicVP =
        Object.values(state.buildings)
            .filter(b => b.owner === opp)
            .reduce((s, b) => s + (b.kind === 'city' ? 2 : 1), 0)
        + (state.longestRoad?.player === opp ? 2 : 0)
        + (state.largestArmy?.player === opp ? 2 : 0);
    put(oppPublicVP / 10);             // 27  no hidden dev-card VPs
    put(state.longestRoad?.player === opp ? 1 : 0); // 28
    put(state.largestArmy?.player === opp ? 1 : 0); // 29

    // ── Shared board state ─────────────────────────────────────────────────────
    hot(bi.tIdx.get(state.robber)!, 19);           // 30-48  robber tile

    const PHASE_IDX: Record<string, number> = {
        setupSettlement: 0, setupRoad: 1, roll: 2,
        discard: 3, moveRobber: 4, main: 5, gameOver: 6,
    };
    hot(PHASE_IDX[state.phase] ?? 6, 7);           // 49-55  phase

    for (const r of RESOURCES) put(state.bank[r] / 19); // 56-60  bank
    put(state.devDeck.length / 25);                // 61     dev deck remaining

    const lrp = state.longestRoad?.player;
    hot(lrp == null ? 0 : lrp === player ? 1 : 2, 3); // 62-64  LR holder
    const lap = state.largestArmy?.player;
    hot(lap == null ? 0 : lap === player ? 1 : 2, 3); // 65-67  LA holder

    // ── Per-vertex (54 × 13 = 702) ─────────────────────────────────────────────
    for (const vid of bi.vertices) {
        const b = state.buildings[vid];
        hot(!b ? 0 : b.owner === player ? 1 : 2, 3);           // owner
        hot(!b ? 0 : b.kind === 'settlement' ? 1 : 2, 3);      // kind
        const port = state.board.vertices[vid].port;
        let portIdx = 0;
        if (port) portIdx = port.kind === '3:1' ? 1 : 2 + RESOURCES.indexOf(port.resource);
        hot(portIdx, 7);                                         // port type
    }

    // ── Per-tile (19 × 18 = 342) ───────────────────────────────────────────────
    for (const tid of bi.tiles) {
        const t = state.board.tiles[tid];
        hot(TERRAINS.indexOf(t.terrain), 6);                              // terrain
        hot(TOKEN_ORDER.indexOf(t.numberToken as number | null), 11);     // number token
        put(tid === state.robber ? 1 : 0);                                // is robber
    }

    // ── Per-edge (72 × 3 = 216) ────────────────────────────────────────────────
    for (const eid of bi.edges) {
        const ow = state.roads[eid];
        hot(!ow ? 0 : ow === player ? 1 : 2, 3);
    }

    return obs; // length 1328
}

// ── Action encoding ────────────────────────────────────────────────────────────

/** Converts a concrete Action to its integer index in [0, ACT_SIZE). */
export function actionToIndex(
    action: Action,
    state: GameState,
    player: PlayerId,
    bi: BoardIndex,
): number {
    const opp = state.turnOrder.find(p => p !== player)!;

    switch (action.type) {
        case 'rollDice':         return 0;
        case 'endTurn':          return 1;
        case 'buyDevCard':       return 2;
        case 'playRoadBuilding': return 3;

        case 'playMonopoly':
            return 4 + RESOURCES.indexOf(action.resource);

        case 'playYearOfPlenty': {
            // Normalise pair order to match YOP_PAIRS construction (RESOURCES index order).
            const sorted = [...action.take].sort(
                (a, b) => RESOURCES.indexOf(a) - RESOURCES.indexOf(b),
            ) as [Resource, Resource];
            const i = YOP_PAIRS.findIndex(([x, y]) => x === sorted[0] && y === sorted[1]);
            return 9 + i;
        }

        case 'bankTrade': {
            const i = TRADE_PAIRS.findIndex(
                ([g, r]) => g === action.give && r === action.receive,
            );
            return 24 + i;
        }

        case 'buildSettlement': return 44  + bi.vIdx.get(action.vertex)!;
        case 'buildCity':       return 98  + bi.vIdx.get(action.vertex)!;
        case 'buildRoad':       return 152 + bi.eIdx.get(action.edge)!;

        case 'moveRobber':
            return (action.stealFrom == null ? 224 : 243) + bi.tIdx.get(action.tile)!;

        case 'playKnight':
            return (action.stealFrom == null ? 262 : 281) + bi.tIdx.get(action.robberTo)!;

        default: return 1; // endTurn as safe fallback
    }
}

/**
 * Reconstructs an Action from its integer index.
 * For bankTrade the giveCount is resolved from the current game state
 * (the net doesn't need to predict the count — it's determined by ports).
 */
export function indexToAction(
    idx: number,
    state: GameState,
    player: PlayerId,
    bi: BoardIndex,
): Action {
    const opp = state.turnOrder.find(p => p !== player)!;

    if (idx === 0) return { type: 'rollDice' };
    if (idx === 1) return { type: 'endTurn' };
    if (idx === 2) return { type: 'buyDevCard' };
    if (idx === 3) {
        // Road Building: take a concrete legal edge pair from legalActions —
        // it enumerates only valid pairs, including the single-road and
        // low-supply edge cases. (Hand-building the pair here used to emit a
        // degenerate [e, e] duplicate when only one extension existed, which
        // the engine rejects.)
        const match = legalActions(state, player).find(
            (a): a is Extract<Action, { type: 'playRoadBuilding' }> =>
                a.type === 'playRoadBuilding',
        );
        return match ?? { type: 'endTurn' };
    }
    if (idx >= 4  && idx <= 8)  return { type: 'playMonopoly', resource: RESOURCES[idx - 4] };
    if (idx >= 9  && idx <= 23) return { type: 'playYearOfPlenty', take: YOP_PAIRS[idx - 9] };
    if (idx >= 24 && idx <= 43) {
        const [give, receive] = TRADE_PAIRS[idx - 24];
        // Resolve giveCount from legal actions (2, 3, or 4 depending on ports).
        const match = legalActions(state, player).find(
            (a): a is Extract<Action, { type: 'bankTrade' }> =>
                a.type === 'bankTrade' && a.give === give && a.receive === receive,
        );
        return match ?? { type: 'bankTrade', give, receive, giveCount: 4 };
    }
    if (idx >= 44  && idx <= 97)  return { type: 'buildSettlement', vertex: bi.vertices[idx - 44] };
    if (idx >= 98  && idx <= 151) return { type: 'buildCity',       vertex: bi.vertices[idx - 98] };
    if (idx >= 152 && idx <= 223) return { type: 'buildRoad',       edge:   bi.edges[idx - 152]   };
    if (idx >= 224 && idx <= 242) return { type: 'moveRobber', tile: bi.tiles[idx - 224], stealFrom: null };
    if (idx >= 243 && idx <= 261) return { type: 'moveRobber', tile: bi.tiles[idx - 243], stealFrom: opp  };
    if (idx >= 262 && idx <= 280) return { type: 'playKnight', robberTo: bi.tiles[idx - 262], stealFrom: null };
    if (idx >= 281 && idx <= 299) return { type: 'playKnight', robberTo: bi.tiles[idx - 281], stealFrom: opp  };

    return { type: 'endTurn' };
}

// ── Legal mask ────────────────────────────────────────────────────────────────

/**
 * Returns a Uint8Array of length ACT_SIZE where 1 = legal, 0 = illegal.
 * During discard phase the mask will be all-zero; the calling policy should
 * handle that phase with the heuristic `discardAction` instead.
 */
export function legalMask(
    state: GameState,
    player: PlayerId,
    bi: BoardIndex,
): Uint8Array {
    const mask = new Uint8Array(ACT_SIZE);
    if (state.phase === 'discard') return mask; // use heuristic; never consult net

    for (const action of legalActions(state, player)) {
        const idx = actionToIndex(action, state, player, bi);
        if (idx >= 0 && idx < ACT_SIZE) mask[idx] = 1;
    }
    return mask;
}

// ── Helpers exported for training data generation ─────────────────────────────

/** One training sample: what the agent saw, what it did, and who won. */
export interface TrainingSample {
    /** Flattened Float32Array as regular number[]. JSON-serialisable. */
    obs: number[];
    /** Integer action index. */
    action: number;
    /** 1 = this player won, 0 = this player lost. */
    outcome: 0 | 1;
    /** Which player was acting (for bookkeeping). */
    player: PlayerId;
    /** Game step index within the episode. */
    step: number;
}

/**
 * Collects (obs, action, outcome) tuples for every non-discard decision
 * in a completed game by replaying the move list.
 *
 * Usage:
 *   const samples = collectEpisode(archive.moves, finalState, winner, bi);
 *   // write samples to JSON, feed to Python trainer
 */
export function collectEpisode(
    moves: ReadonlyArray<{ player: PlayerId; action: Action }>,
    finalState: GameState,
    winner: PlayerId | null,
    bi: BoardIndex,
): TrainingSample[] {
    // Replay from initial state is not done here — callers should run the engine
    // and capture (state_before, action) pairs step by step.
    // This stub documents the interface; the actual collector lives in the
    // benchmark/self-play script where the engine is already running.
    void moves; void finalState; void winner; void bi;
    return [];
}