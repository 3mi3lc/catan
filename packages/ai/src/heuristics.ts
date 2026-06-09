import {
  RESOURCES, TERRAIN_RESOURCE, handSize, victoryPoints,
  type GameState, type PlayerId, type VertexId, type EdgeId, type TileId, type Resource, type Action,
} from '@catan/core';

export const pips = (n: number | null): number => (n === null ? 0 : 6 - Math.abs(7 - n));

export interface Weights {
  resource: Record<Resource, number>;
  diversity: number;
  port2: number;
  port3: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  resource: { brick: 1.0, lumber: 1.0, wool: 0.9, grain: 1.1, ore: 1.1 },
  diversity: 1.5,
  // Raised from 1.2/0.6 — a matching 2:1 port is a major economic asset and was
  // being drastically undervalued vs production.  3:1 is always useful.
  port2: 4.0,
  port3: 1.8,
};

export function vertexProduction(state: GameState, v: VertexId, w: Weights = DEFAULT_WEIGHTS): number {
  let score = 0;
  for (const tid of state.board.vertices[v].tiles as TileId[]) {
    const t = state.board.tiles[tid];
    const res = TERRAIN_RESOURCE[t.terrain];
    if (res) score += pips(t.numberToken) * w.resource[res];
  }
  return score;
}

// Optional `player` param enables a smarter 2:1 port bonus: only grant the full
// bonus when this vertex (or the player's existing network) produces the matching
// resource.  A 2:1 ore port is worthless if you never generate ore.
export function vertexScore(
    state: GameState, v: VertexId, w: Weights = DEFAULT_WEIGHTS, player?: PlayerId,
): number {
  let score = vertexProduction(state, v, w);
  const seen = new Set<Resource>();
  for (const tid of state.board.vertices[v].tiles as TileId[]) {
    const r = TERRAIN_RESOURCE[state.board.tiles[tid].terrain];
    if (r) seen.add(r);
  }
  score += w.diversity * seen.size;

  const port = state.board.vertices[v].port;
  if (port) {
    if (port.kind === '3:1') {
      score += w.port3; // 3:1 is always useful regardless of what you produce
    } else {
      const portRes = port.resource;
      // Full bonus only if this vertex produces the resource OR the player already does.
      const localProd = seen.has(portRes);
      const networkProd = player != null && Object.entries(state.buildings).some(([vid, b]) => {
        if (b.owner !== player) return false;
        return (state.board.vertices[vid as VertexId].tiles as TileId[]).some(
            tid => TERRAIN_RESOURCE[state.board.tiles[tid].terrain] === portRes,
        );
      });
      score += (localProd || networkProd) ? w.port2 : w.port2 * 0.1;
    }
  }
  return score;
}

export function argmax<T>(items: readonly T[], score: (t: T) => number): T | undefined {
  let best: T | undefined;
  let bestScore = -Infinity;
  for (const it of items) {
    const s = score(it);
    if (s > bestScore) { bestScore = s; best = it; }
  }
  return best;
}

// Robber scoring with a scarcity bonus: blocking a tile is more valuable when
// the opponent has few alternative production sources for that resource,
// because it forces expensive 4:1 trades.
export function robberScore(
    state: GameState, player: PlayerId, tile: TileId, stealFrom: PlayerId | null,
    w: Weights = DEFAULT_WEIGHTS,
): number {
  const t = state.board.tiles[tile];
  const res = TERRAIN_RESOURCE[t.terrain];
  const perBuilding = pips(t.numberToken) * (res ? w.resource[res] : 1);
  let score = 0;

  for (const v of t.vertices) {
    const b = state.buildings[v];
    if (!b) continue;
    const mult = b.kind === 'city' ? 2 : 1;
    const yieldv = perBuilding * mult;

    if (b.owner === player) {
      score -= 0.8 * yieldv;
    } else {
      score += yieldv;
      // Scarcity bonus: how many pips of `res` does this opponent get elsewhere?
      // The fewer alternatives, the more it hurts to block this tile.
      if (res) {
        let altPips = 0;
        for (const [vid2, b2] of Object.entries(state.buildings)) {
          if (b2.owner !== b.owner) continue;
          for (const tid2 of (state.board.vertices[vid2 as VertexId].tiles as TileId[])) {
            if (tid2 === tile) continue;
            if (TERRAIN_RESOURCE[state.board.tiles[tid2].terrain] === res)
              altPips += pips(state.board.tiles[tid2].numberToken) * (b2.kind === 'city' ? 2 : 1);
          }
        }
        // 0 alternatives → scarcityMult ≈ 1 → +50% on top of base; many alternatives → ~0
        score += yieldv * 0.5 / (1 + altPips * 0.3);
      }
    }
  }

  if (stealFrom) {
    score += 0.4 * handSize(state.players[stealFrom].resources);
    score += 0.6 * victoryPoints(state, stealFrom);
  }
  return score;
}

// BFS-based road value: score a road by the best settlement spot reachable within
// `maxExtraRoads` additional roads from the extended network.  Existing roads of
// the player are traversed for FREE so the bot correctly values intermediate roads
// in a chain — the original 1-hop version returned 0 for them, causing random
// direction changes when building a road chain.
export function roadExpansionValue(
    state: GameState, player: PlayerId, edge: EdgeId,
    w: Weights = DEFAULT_WEIGHTS, maxExtraRoads = 3,
): number {
  // Temporarily add the proposed road to our network.
  const tempOwner = new Map<string, string>(Object.entries(state.roads));
  tempOwner.set(edge, player);

  const dist = new Map<string, number>();
  const queue: Array<[VertexId, number]> = [];

  const enqueue = (v: VertexId, d: number) => {
    const key = v as string;
    if ((dist.get(key) ?? Infinity) > d) {
      dist.set(key, d);
      queue.push([v, d]);
    }
  };

  for (const startV of state.board.edges[edge].vertices as VertexId[]) enqueue(startV, 0);

  let qi = 0;
  while (qi < queue.length) {
    const [v, d] = queue[qi++];
    if ((dist.get(v as string) ?? Infinity) < d) continue; // stale

    for (const adjEdge of state.board.vertices[v].edges as EdgeId[]) {
      const owner = tempOwner.get(adjEdge);
      if (owner && owner !== player) continue; // blocked by opponent road
      const [a, b] = state.board.edges[adjEdge].vertices;
      const nextV = (a === v ? b : a) as VertexId;
      const newD = d + (owner === player ? 0 : 1); // our road = free to traverse
      if (newD <= maxExtraRoads) enqueue(nextV, newD);
    }
  }

  let best = 0;
  for (const [vKey, d] of dist) {
    const v = vKey as VertexId;
    if (state.buildings[v]) continue;
    const blocked = state.board.vertices[v].edges.some((e: EdgeId) => {
      const [a, b] = state.board.edges[e].vertices;
      return !!state.buildings[a === v ? b : a];
    });
    if (!blocked) {
      // Discount by distance: each extra road needed reduces the value.
      const score = vertexScore(state, v, w, player) * Math.pow(0.72, d);
      if (score > best) best = score;
    }
  }
  return best;
}

// Choose the two resources to take with Year of Plenty based on build priorities.
// Without this, the first legal YoP pair (always [brick,brick] in legalActions
// enumeration order) gets picked — completely ignoring what would actually help.
export function bestYopTake(state: GameState, player: PlayerId): [Resource, Resource] | null {
  const R = state.players[player].resources;
  const pl = state.players[player];
  const bk = state.bank;
  const ownsSettlement = Object.values(state.buildings)
      .some(b => b.owner === player && b.kind === 'settlement');

  // Score each resource by urgency toward our current best build goal.
  const urgency: Record<Resource, number> = { brick: 0, lumber: 0, wool: 0, grain: 0, ore: 0 };

  // City needs 2 grain + 3 ore; urgent if we have a settlement to upgrade.
  if (ownsSettlement && pl.supply.cities > 0) {
    const ng = Math.max(0, 2 - R.grain);
    const no = Math.max(0, 3 - R.ore);
    if (ng > 0) urgency.grain += 4 / ng;   // closer to completion = higher urgency
    if (no > 0) urgency.ore   += 3 / no;
  }

  // Settlement needs 1 each of brick/lumber/wool/grain.
  if (pl.supply.settlements > 0) {
    if (R.grain  < 1) urgency.grain  += 2;
    if (R.lumber < 1) urgency.lumber += 2;
    if (R.brick  < 1) urgency.brick  += 2;
    if (R.wool   < 1) urgency.wool   += 2;
  }

  // Pick the 2 most-urgent resources available in bank (tracking take[0] reduces bank).
  const pick: Resource[] = [];
  const tempBank = { ...bk };
  for (let i = 0; i < 2; i++) {
    // Try to pick a resource with urgency > 0 first, then fall back to least-held.
    const sorted = ([...RESOURCES] as Resource[])
        .filter(r => tempBank[r] > 0)
        .sort((a, b) => (urgency[b] - urgency[a]) || (R[a] - R[b]));
    const choice = sorted[0];
    if (!choice) break;
    pick.push(choice);
    tempBank[choice]--;
  }

  if (pick.length < 2) return null;
  return [pick[0], pick[1]];
}

export function discardAction(state: GameState, player: PlayerId): Action {
  const owed = state.pendingDiscards[player] ?? 0;
  const have = { ...state.players[player].resources };
  const dropWeight: Record<Resource, number> = { wool: 1.0, brick: 0.9, lumber: 0.9, grain: 0.6, ore: 0.6 };
  const resources: Partial<Record<Resource, number>> = {};
  for (let i = 0; i < owed; i++) {
    let pick: Resource | null = null;
    let best = -1;
    for (const r of RESOURCES) {
      if (have[r] <= 0) continue;
      const v = have[r] * dropWeight[r];
      if (v > best) { best = v; pick = r; }
    }
    if (!pick) break;
    have[pick]--;
    resources[pick] = (resources[pick] ?? 0) + 1;
  }
  return { type: 'discard', resources };
}