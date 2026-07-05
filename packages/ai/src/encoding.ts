/**
 * encoding.ts — Fixed-size observation vector and action index for the RL agent.
 *
 * Design decisions:
 *  • Fair-information: own hand is fully visible; opponent hand sizes visible
 *    but not composition; dev deck order hidden. Required for online play.
 *  • Seat-relative & padded to MAX_SEATS (4): the observation is encoded from
 *    the acting player's perspective, and opponents are ordered by how soon
 *    they act after the acting player (opp1 = next to move, …). Unused seats
 *    (2- and 3-player games) are zero-padded with a per-opponent "present"
 *    bit, so a single net serves 2/3/4-player games.
 *  • Flat action space (ACT_SIZE slots) with a legal mask derived from
 *    legalActions(). The net outputs ACT_SIZE logits; illegal slots get −∞
 *    before softmax. Robber/Knight steal targets select a *relative* opponent.
 *  • Discard and Road Building use heuristics (never passed to the net);
 *    the calling policy handles those phases before consulting the net.
 *  • BoardIndex maps vertex/edge/tile IDs → stable integer indices. The
 *    hex topology is fixed for all radius-2 boards (same 54 vertex positions
 *    every game), so the index computed for one board applies to all games.
 */

import {
    legalActions, victoryPoints, RESOURCES,
    type Board, type GameState, type PlayerId, type Action,
    type VertexId, type EdgeId, type TileId, type Resource, type DevCard,
    roadConnected,
} from '@catan/core';

// ── Shared constants ──────────────────────────────────────────────────────────

/** Maximum seats the representation supports. Opponents beyond the actual
 *  player count are zero-padded (present-bit = 0). */
export const MAX_SEATS = 4;
/** Opponent slots = MAX_SEATS − 1 (self is always slot "me"). */
const MAX_OPP = MAX_SEATS - 1; // 3

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

// ── Action index layout (4-seat, 376 slots) ───────────────────────────────────
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
//   224-299 moveRobber  × 19 tiles × {none, opp1, opp2, opp3}   (group g = relative opp; 0 = no steal)
//   300-375 playKnight  × 19 tiles × {none, opp1, opp2, opp3}
//   ── Trade negotiation (376-395) ──
//   376-380 offerAddGive × 5 resources   (add 1 of resource to the give side)
//   381-385 offerAddWant × 5 resources   (add 1 of resource to the want side)
//   386     offerBroadcast               (send the composed offer to all)
//   387     offerCancel
//   388     respondAccept                (opponent accepts the broadcast)
//   389     respondReject
//   390     counterStart                 (opponent begins a counter-offer)
//   391     submitCounter                (opponent submits the counter)
//   392-394 confirmTrade × 3 rel opp     (proposer executes with that opponent)
//   395     declineAll                   (proposer ends the negotiation)
//
// The active player composes a bundle (add give/want) and broadcasts it;
// opponents accept/reject/counter (a counter is composed with the same add
// actions then submitted); the proposer confirms with one opponent or declines.
// Several offers per turn. Confirm targets are RELATIVE opponents (1..3).

const N_TILES = 19;
const A_SETTLE = 44;
const A_CITY = 98;
const A_ROAD = 152;
const A_ROBBER = 224;                       // 224 + g*19 + tile,  g ∈ 0..3
const A_KNIGHT = A_ROBBER + MAX_SEATS * N_TILES; // 300 + g*19 + tile
const A_ADDGIVE   = A_KNIGHT + MAX_SEATS * N_TILES; // 376  + RESOURCES index (5)
const A_ADDWANT   = A_ADDGIVE + 5;          // 381  + RESOURCES index (5)
const A_BROADCAST = A_ADDWANT + 5;          // 386
const A_CANCEL    = A_BROADCAST + 1;        // 387
const A_ACCEPT    = A_CANCEL + 1;           // 388
const A_REJECT    = A_ACCEPT + 1;           // 389
const A_COUNTER   = A_REJECT + 1;           // 390  counterStart
const A_SUBMIT    = A_COUNTER + 1;          // 391  submitCounter
const A_CONFIRM   = A_SUBMIT + 1;           // 392  + (relOpp − 1)  (3)
const A_DECLINE   = A_CONFIRM + MAX_OPP;    // 395
export const ACT_SIZE = A_DECLINE + 1;      // 396

// ── Observation layout (1603 floats per step) ─────────────────────────────────
//
//   ── Global block (91) ──
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
//   22-48   Opponents fair-info: 3 blocks × 9, ordered by turn proximity
//           (opp1 = next to act). Absent seats are all-zero (present = 0).
//     +0    handSize / 18       (card count known; composition hidden)
//     +1    playedKnights / 14
//     +2    supply.settlements / 5
//     +3    supply.cities / 4
//     +4    supply.roads / 15
//     +5    publicVP / 10       (buildings + LR + LA; NOT hidden dev-card VPs)
//     +6    hasLongestRoad
//     +7    hasLargestArmy
//     +8    present (1 if this opponent seat exists)
//
//   49-67   robberTile one-hot (19)
//   68-74   phase one-hot (7)
//   75-79   bank resources / 19 (5)
//   80      devDeck remaining / 25
//   81-85   longestRoad holder one-hot: none/me/opp1/opp2/opp3 (5)
//   86-90   largestArmy holder one-hot: none/me/opp1/opp2/opp3 (5)
//
//   91-146  Trade negotiation (56):
//     91-93  stage: composing(me) / responding / arbitrating (3)
//     94-97  proposer one-hot relative to me: me/opp1/opp2/opp3 (4)
//     98-102 primary give bundle / 4 (5)  (my draft, or the broadcast terms)
//    103-107 primary want bundle / 4 (5)
//    108-116 per rel-opp reply one-hot {accept,counter,reject} × 3 opp (9)
//    117-146 per rel-opp counter terms (give5 + want5) × 3 opp (30)
//
//   ── Per-vertex × 54 × 15 = 810 ──  (offset 147)
//            [0-4]  owner one-hot: none/me/opp1/opp2/opp3 (5)
//            [5-7]  kind one-hot:  empty/settlement/city (3)
//            [8-14] port one-hot:  none, 3:1, 2:1-brick/lumber/wool/grain/ore (7)
//
//   ── Per-tile × 19 × 18 = 342 ──  (offset 957)
//            [0-5]  terrain one-hot (6)
//            [6-16] numberToken one-hot (11; index 0 = null/desert)
//            [17]   isRobber
//
//   ── Per-edge × 72 × 5 = 360 ──  (offset 1299)
//            [0-4]  owner one-hot: none/me/opp1/opp2/opp3 (5)

export const OBS_SIZE = 1659;

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

// ── Seat-relative helpers ──────────────────────────────────────────────────────

/**
 * The up-to-3 opponents ordered by turn proximity to `player` (opp1 = next to
 * act). Returns a length-3 array; entries are null for absent seats in 2/3-player
 * games. The same ordering drives the observation owner one-hots and the steal
 * action groups, so they stay consistent.
 */
export function seatRelativeOpponents(
    state: GameState, player: PlayerId,
): (PlayerId | null)[] {
    const order = state.turnOrder;
    const n = order.length;
    const me = order.indexOf(player);
    const opps: (PlayerId | null)[] = [];
    for (let k = 1; k <= MAX_OPP; k++) {
        opps.push(k < n ? order[(me + k) % n] : null);
    }
    return opps;
}

/** Relative turn offset of `q` from `player` in 1..n−1 (q must differ). */
function relOffset(state: GameState, player: PlayerId, q: PlayerId): number {
    const order = state.turnOrder;
    const n = order.length;
    return (order.indexOf(q) - order.indexOf(player) + n) % n;
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

    const put = (v: number)              => { obs[o++] = v; };
    const hot = (idx: number, n: number) => { obs[o + idx] = 1; o += n; };

    const opps = seatRelativeOpponents(state, player);

    /** Owner one-hot slot: 0 none, 1 me, 2..4 opp by relative turn offset. */
    const slotOf = (owner: PlayerId | null | undefined): number => {
        if (owner == null) return 0;
        if (owner === player) return 1;
        return 1 + relOffset(state, player, owner); // 2..4
    };

    const pl = state.players[player];

    // ── Own player (22) ──────────────────────────────────────────────────────
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

    // ── Opponents fair-info (3 × 9 = 27) ─────────────────────────────────────
    for (let k = 0; k < MAX_OPP; k++) {
        const oppId = opps[k];
        if (oppId == null) { o += 9; continue; }   // absent seat → all zeros
        const op = state.players[oppId];
        const oppHand = RESOURCES.reduce((s, r) => s + op.resources[r], 0);
        put(oppHand / 18);                 // hand SIZE visible; not composition
        put(op.playedKnights / 14);
        put(op.supply.settlements / 5);
        put(op.supply.cities / 4);
        put(op.supply.roads / 15);
        const oppPublicVP =
            Object.values(state.buildings)
                .filter(b => b.owner === oppId)
                .reduce((s, b) => s + (b.kind === 'city' ? 2 : 1), 0)
            + (state.longestRoad?.player === oppId ? 2 : 0)
            + (state.largestArmy?.player === oppId ? 2 : 0);
        put(oppPublicVP / 10);             // no hidden dev-card VPs
        put(state.longestRoad?.player === oppId ? 1 : 0);
        put(state.largestArmy?.player === oppId ? 1 : 0);
        put(1);                            // present
    }

    // ── Shared board state ───────────────────────────────────────────────────
    hot(bi.tIdx.get(state.robber)!, 19);           // robber tile

    const PHASE_IDX: Record<string, number> = {
        setupSettlement: 0, setupRoad: 1, roll: 2,
        discard: 3, moveRobber: 4, main: 5, gameOver: 6,
    };
    hot(PHASE_IDX[state.phase] ?? 6, 7);           // phase

    for (const r of RESOURCES) put(state.bank[r] / 19); // bank
    put(state.devDeck.length / 25);                // dev deck remaining

    hot(slotOf(state.longestRoad?.player), 5);     // LR holder none/me/opp1-3
    hot(slotOf(state.largestArmy?.player), 5);     // LA holder none/me/opp1-3

    // ── Trade negotiation (56) ───────────────────────────────────────────────
    const neg = state.negotiation;
    const draft = state.draftOffer;
    const composing = !!draft && draft.by === player;
    put(composing ? 1 : 0);                                       // I am composing
    put(neg && neg.stage === 'responding' ? 1 : 0);
    put(neg && neg.stage === 'arbitrating' ? 1 : 0);
    const proposer = neg ? neg.proposer : (composing ? player : null);
    if (proposer) hot(relOffset(state, player, proposer), 4); else o += 4;   // who initiated
    const primary = composing ? draft! : neg ? { give: neg.give, want: neg.want } : null;
    for (const r of RESOURCES) put(primary ? (primary.give[r] ?? 0) / 4 : 0); // terms on the table
    for (const r of RESOURCES) put(primary ? (primary.want[r] ?? 0) / 4 : 0);
    for (let k = 0; k < MAX_OPP; k++) {                          // each rel-opp's reply
        const oppId = opps[k];
        const rep = neg && oppId ? neg.responses[oppId] : undefined;
        put(rep === 'accept' ? 1 : 0);
        put(rep === 'counter' ? 1 : 0);
        put(rep === 'reject' ? 1 : 0);
    }
    for (let k = 0; k < MAX_OPP; k++) {                          // each rel-opp's counter terms
        const oppId = opps[k];
        const c = neg && oppId ? neg.counters[oppId] : undefined;
        for (const r of RESOURCES) put(c ? (c.give[r] ?? 0) / 4 : 0);
        for (const r of RESOURCES) put(c ? (c.want[r] ?? 0) / 4 : 0);
    }

    // ── Per-vertex (54 × 15 = 810) ───────────────────────────────────────────
    for (const vid of bi.vertices) {
        const b = state.buildings[vid];
        hot(slotOf(b?.owner), 5);                              // owner
        hot(!b ? 0 : b.kind === 'settlement' ? 1 : 2, 3);     // kind
        const port = state.board.vertices[vid].port;
        let portIdx = 0;
        if (port) portIdx = port.kind === '3:1' ? 1 : 2 + RESOURCES.indexOf(port.resource);
        hot(portIdx, 7);                                       // port type
    }

    // ── Per-tile (19 × 18 = 342) ─────────────────────────────────────────────
    for (const tid of bi.tiles) {
        const t = state.board.tiles[tid];
        hot(TERRAINS.indexOf(t.terrain), 6);                              // terrain
        hot(TOKEN_ORDER.indexOf(t.numberToken as number | null), 11);     // number token
        put(tid === state.robber ? 1 : 0);                                // is robber
    }

    // ── Per-edge (72 × 5 = 360) ──────────────────────────────────────────────
    for (const eid of bi.edges) {
        hot(slotOf(state.roads[eid]), 5);
    }

    return obs; // length 1603
}

// ── Action encoding ────────────────────────────────────────────────────────────

/** Converts a concrete Action to its integer index in [0, ACT_SIZE). */
export function actionToIndex(
    action: Action,
    state: GameState,
    player: PlayerId,
    bi: BoardIndex,
): number {
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

        case 'buildSettlement': return A_SETTLE + bi.vIdx.get(action.vertex)!;
        case 'buildCity':       return A_CITY   + bi.vIdx.get(action.vertex)!;
        case 'buildRoad':       return A_ROAD   + bi.eIdx.get(action.edge)!;

        case 'moveRobber': {
            const g = action.stealFrom == null ? 0 : relOffset(state, player, action.stealFrom);
            return A_ROBBER + g * N_TILES + bi.tIdx.get(action.tile)!;
        }

        case 'playKnight': {
            const g = action.stealFrom == null ? 0 : relOffset(state, player, action.stealFrom);
            return A_KNIGHT + g * N_TILES + bi.tIdx.get(action.robberTo)!;
        }

        case 'offerAddGive':  return A_ADDGIVE + RESOURCES.indexOf(action.resource);
        case 'offerAddWant':  return A_ADDWANT + RESOURCES.indexOf(action.resource);
        case 'offerBroadcast': return A_BROADCAST;
        case 'offerCancel':   return A_CANCEL;
        case 'respondAccept': return A_ACCEPT;
        case 'respondReject': return A_REJECT;
        case 'counterStart':  return A_COUNTER;
        case 'submitCounter': return A_SUBMIT;
        case 'confirmTrade': {
            const g = relOffset(state, player, action.to);       // 1..3
            return g < 1 ? 1 : A_CONFIRM + (g - 1);
        }
        case 'declineAll':    return A_DECLINE;

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
    const order = state.turnOrder;
    const n = order.length;
    const me = order.indexOf(player);
    /** Absolute player at relative turn offset g (1..n−1). */
    const oppAt = (g: number): PlayerId => order[(me + g) % n];

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
    if (idx >= A_SETTLE && idx < A_SETTLE + 54) return { type: 'buildSettlement', vertex: bi.vertices[idx - A_SETTLE] };
    if (idx >= A_CITY   && idx < A_CITY + 54)   return { type: 'buildCity',       vertex: bi.vertices[idx - A_CITY]   };
    if (idx >= A_ROAD   && idx < A_ROAD + 72)   return { type: 'buildRoad',       edge:   bi.edges[idx - A_ROAD]      };

    if (idx >= A_ROBBER && idx < A_ROBBER + MAX_SEATS * N_TILES) {
        const local = idx - A_ROBBER;
        const g = Math.floor(local / N_TILES);
        const t = local % N_TILES;
        return { type: 'moveRobber', tile: bi.tiles[t], stealFrom: g === 0 ? null : oppAt(g) };
    }
    if (idx >= A_KNIGHT && idx < A_KNIGHT + MAX_SEATS * N_TILES) {
        const local = idx - A_KNIGHT;
        const g = Math.floor(local / N_TILES);
        const t = local % N_TILES;
        return { type: 'playKnight', robberTo: bi.tiles[t], stealFrom: g === 0 ? null : oppAt(g) };
    }

    if (idx >= A_ADDGIVE && idx < A_ADDGIVE + 5)
        return { type: 'offerAddGive', resource: RESOURCES[idx - A_ADDGIVE] };
    if (idx >= A_ADDWANT && idx < A_ADDWANT + 5)
        return { type: 'offerAddWant', resource: RESOURCES[idx - A_ADDWANT] };
    if (idx === A_BROADCAST) return { type: 'offerBroadcast' };
    if (idx === A_CANCEL)    return { type: 'offerCancel' };
    if (idx === A_ACCEPT)    return { type: 'respondAccept' };
    if (idx === A_REJECT)    return { type: 'respondReject' };
    if (idx === A_COUNTER)   return { type: 'counterStart' };
    if (idx === A_SUBMIT)    return { type: 'submitCounter' };
    if (idx >= A_CONFIRM && idx < A_CONFIRM + MAX_OPP)
        return { type: 'confirmTrade', to: oppAt((idx - A_CONFIRM) + 1) };
    if (idx === A_DECLINE)   return { type: 'declineAll' };

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

// ── No-trade view (for a strong net that doesn't model trades) ────────────────
// The trade work appended the trade actions (376+) and inserted a 56-float
// trade block at obs[91:147]. Dropping both recovers EXACTLY the pre-trade
// 4-seat representation, so the no-trade champion runs unchanged on an engine
// that has trading — a heuristic handles the trade decisions instead.

const TRADE_OBS_START = 91;
const TRADE_OBS_LEN = 56;
export const OBS_SIZE_NOTRADE = OBS_SIZE - TRADE_OBS_LEN;  // 1603
export const ACT_SIZE_NOTRADE = A_ADDGIVE;                // 376 (trade actions start here)

/** Observation with the trade block spliced out (1603) — matches the no-trade net. */
export function encodeObservationNoTrade(
    state: GameState, player: PlayerId, bi: BoardIndex,
): Float32Array {
    const full = encodeObservation(state, player, bi);
    const out = new Float32Array(OBS_SIZE_NOTRADE);
    out.set(full.subarray(0, TRADE_OBS_START), 0);
    out.set(full.subarray(TRADE_OBS_START + TRADE_OBS_LEN), TRADE_OBS_START);
    return out;
}

/** Legal mask over only the non-trade actions (376). */
export function legalMaskNoTrade(
    state: GameState, player: PlayerId, bi: BoardIndex,
): Uint8Array {
    return legalMask(state, player, bi).slice(0, ACT_SIZE_NOTRADE);
}

// ── Helpers exported for training data generation ─────────────────────────────

/** One training sample: what the agent saw, what it did, and who won. */
export interface TrainingSample {
    /** Flattened Float32Array as regular number[]. JSON-serialisable. */
    obs: number[];
    /** Integer action index. */
    action: number;
    /** Relative seat offset of the winner from the acting player (0 = self won). */
    outcome: number;
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
