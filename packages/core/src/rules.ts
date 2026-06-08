import { Board, Resource, RESOURCES, TERRAIN_RESOURCE } from './board';
import { TileId, VertexId, EdgeId, PlayerId } from './ids';
import { GameState, ResourceCounts } from './state';
import { ResourceBundle } from './actions';

// --- resources & costs ----------------------------------------------------

export const COSTS: Record<'road' | 'settlement' | 'city' | 'devCard', ResourceBundle> = {
    road: { brick: 1, lumber: 1 },
    settlement: { brick: 1, lumber: 1, wool: 1, grain: 1 },
    city: { grain: 2, ore: 3 },
    devCard: { wool: 1, grain: 1, ore: 1 },
};

export function zeroResources(): ResourceCounts {
    return { brick: 0, lumber: 0, wool: 0, grain: 0, ore: 0 };
}

// Pure mulberry32 step: returns a float in [0, 1) plus the next generator state.
// Storing this single integer in GameState makes every game a deterministic
// function of its seed — the foundation for replay and reproducible self-play.
export function rngStep(state: number): { value: number; next: number } {
    const a = ((state | 0) + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, next: a };
}

const bundleEntries = (b: ResourceBundle): [Resource, number][] =>
    Object.entries(b).map(([r, n]) => [r as Resource, n ?? 0]);

export function canAfford(have: ResourceCounts, cost: ResourceBundle): boolean {
    return bundleEntries(cost).every(([r, n]) => have[r] >= n);
}

// Add (sign +1) or subtract (sign -1) a bundle from a resource hand, immutably.
export function applyBundle(have: ResourceCounts, bundle: ResourceBundle, sign: 1 | -1): ResourceCounts {
    const next = { ...have };
    for (const [r, n] of bundleEntries(bundle)) next[r] += sign * n;
    return next;
}

export function handSize(have: ResourceCounts): number {
    return RESOURCES.reduce((s, r) => s + have[r], 0);
}

export function bundleTotal(b: ResourceBundle): number {
    return bundleEntries(b).reduce((s, [, n]) => s + n, 0);
}

// --- board queries --------------------------------------------------------

export function vertexNeighbors(board: Board, v: VertexId): VertexId[] {
    return board.vertices[v].edges.map((e) => {
        const [a, b] = board.edges[e].vertices;
        return a === v ? b : a;
    });
}

// A settlement is legal at v if v is empty and no adjacent vertex is built on
// (the "distance rule").
export function canPlaceSettlement(state: GameState, v: VertexId): boolean {
    if (state.buildings[v]) return false;
    return vertexNeighbors(state.board, v).every((n) => !state.buildings[n]);
}

// A new road at `edge` connects to the player's network if one of its endpoints
// holds their building, or meets another of their roads (and isn't blocked by an
// opponent's building).
export function roadConnected(state: GameState, player: PlayerId, edge: EdgeId): boolean {
    return state.board.edges[edge].vertices.some((v) => {
        const building = state.buildings[v];
        if (building) return building.owner === player;
        return state.board.vertices[v].edges.some((e) => e !== edge && state.roads[e] === player);
    });
}

// --- longest road ---------------------------------------------------------
// Longest continuous road for a player: the longest trail (no edge reused) in
// their road subgraph, where an opponent's building blocks passage *through* a
// vertex. The networks are tiny, so a brute-force DFS is plenty.

export function longestRoadLength(state: GameState, player: PlayerId): number {
    const board = state.board;
    const mine = new Set<EdgeId>(
        (Object.keys(state.roads) as EdgeId[]).filter((e) => state.roads[e] === player),
    );
    if (mine.size === 0) return 0;

    const blockedThrough = (v: VertexId): boolean => {
        const b = state.buildings[v];
        return !!b && b.owner !== player;
    };

    const endpoints = new Set<VertexId>();
    for (const e of mine) for (const v of board.edges[e].vertices) endpoints.add(v);

    let best = 0;
    const used = new Set<EdgeId>();

    const walk = (v: VertexId, arrivedVia: EdgeId | null, len: number): void => {
        best = Math.max(best, len);
        if (arrivedVia !== null && blockedThrough(v)) return; // can't pass through
        for (const e of board.vertices[v].edges) {
            if (!mine.has(e) || used.has(e)) continue;
            const [a, b] = board.edges[e].vertices;
            used.add(e);
            walk(a === v ? b : a, e, len + 1);
            used.delete(e);
        }
    };

    for (const v of endpoints) walk(v, null, 0);
    return best;
}

// --- awards (longest road, largest army) ----------------------------------
// Generic incumbent-keeps-on-tie resolution. The award goes to a strict leader
// at/above the threshold; an existing holder keeps it on a tie or until beaten,
// and it is set aside if nobody is at/above the threshold.

function resolveAward<H extends { player: PlayerId }>(
    current: H | null,
    scores: [PlayerId, number][],
    threshold: number,
    make: (player: PlayerId, score: number) => H,
): H | null {
    const max = Math.max(0, ...scores.map(([, s]) => s));
    if (max < threshold) return null;
    const leaders = scores.filter(([, s]) => s === max).map(([p]) => p);
    if (current && leaders.includes(current.player)) return make(current.player, max);
    if (leaders.length === 1) return make(leaders[0], max);
    if (current) {
        const score = scores.find(([p]) => p === current.player)?.[1] ?? 0;
        if (score >= threshold) return make(current.player, score);
    }
    return null;
}

export function recomputeLongestRoad(state: GameState): GameState['longestRoad'] {
    const scores = state.turnOrder.map((p) => [p, longestRoadLength(state, p)] as [PlayerId, number]);
    return resolveAward(state.longestRoad, scores, 5, (player, length) => ({ player, length }));
}

export function recomputeLargestArmy(state: GameState): GameState['largestArmy'] {
    const scores = state.turnOrder.map((p) => [p, state.players[p].playedKnights] as [PlayerId, number]);
    return resolveAward(state.largestArmy, scores, 3, (player, size) => ({ player, size }));
}

// --- victory points -------------------------------------------------------

export function victoryPoints(state: GameState, player: PlayerId): number {
    let vp = 0;
    for (const b of Object.values(state.buildings)) if (b.owner === player) vp += b.kind === 'city' ? 2 : 1;
    const pl = state.players[player];
    vp += [...pl.devCards, ...pl.pendingDevCards].filter((c) => c === 'victoryPoint').length;
    if (state.longestRoad?.player === player) vp += 2;
    if (state.largestArmy?.player === player) vp += 2;
    return vp;
}

// --- ports & bank trade ratio ---------------------------------------------

export function tradeRatio(state: GameState, player: PlayerId, give: Resource): number {
    let ratio = 4;
    for (const v of Object.keys(state.buildings) as VertexId[]) {
        if (state.buildings[v].owner !== player) continue;
        const port = state.board.vertices[v].port;
        if (!port) continue;
        if (port.kind === '3:1') ratio = Math.min(ratio, 3);
        else if (port.kind === '2:1' && port.resource === give) ratio = Math.min(ratio, 2);
    }
    return ratio;
}

// --- resource production --------------------------------------------------
// Distribute resources for a dice total, honouring the bank's finite supply:
// if demand for a resource exceeds the bank and more than one player wants it,
// none of them receive any; if only one player wants it, they take what's left.

export function distribute(
    state: GameState,
    total: number,
): { players: GameState['players']; bank: ResourceCounts; gains: Record<PlayerId, ResourceBundle> } {
    const demand = Object.fromEntries(RESOURCES.map((r) => [r, new Map<PlayerId, number>()])) as Record<
        Resource,
        Map<PlayerId, number>
    >;

    for (const tile of Object.values(state.board.tiles)) {
        if (tile.numberToken !== total || tile.id === state.robber) continue;
        const res = TERRAIN_RESOURCE[tile.terrain];
        if (!res) continue;
        for (const v of tile.vertices) {
            const b = state.buildings[v];
            if (!b) continue;
            const m = demand[res];
            m.set(b.owner, (m.get(b.owner) ?? 0) + (b.kind === 'city' ? 2 : 1));
        }
    }

    const bank = { ...state.bank };
    const players = { ...state.players };
    const gains: Record<PlayerId, ResourceBundle> = {};

    const grant = (p: PlayerId, res: Resource, amt: number): void => {
        if (amt <= 0) return;
        bank[res] -= amt;
        const pl = players[p];
        players[p] = { ...pl, resources: { ...pl.resources, [res]: pl.resources[res] + amt } };
        gains[p] = { ...(gains[p] ?? {}), [res]: (gains[p]?.[res] ?? 0) + amt };
    };

    for (const res of RESOURCES) {
        const m = demand[res];
        const want = [...m.values()].reduce((a, b) => a + b, 0);
        if (want === 0) continue;
        if (want <= bank[res]) {
            for (const [p, amt] of m) grant(p, res, amt);
        } else if (m.size === 1) {
            const [p, amt] = [...m][0];
            grant(p, res, Math.min(amt, bank[res]));
        }
        // else: multiple claimants, not enough — nobody receives this resource.
    }

    return { players, bank, gains };
}

// Resources earned from a single vertex (used for the second setup settlement).
export function vertexYield(state: GameState, v: VertexId): ResourceBundle {
    const out: ResourceBundle = {};
    for (const tileId of state.board.vertices[v].tiles as TileId[]) {
        const res = TERRAIN_RESOURCE[state.board.tiles[tileId].terrain];
        if (res) out[res] = (out[res] ?? 0) + 1;
    }
    return out;
}