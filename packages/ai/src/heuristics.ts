import {
  RESOURCES, TERRAIN_RESOURCE, handSize, victoryPoints,
  type GameState, type PlayerId, type VertexId, type TileId, type Resource, type Action,
} from '@catan/core';

// Probability "pips" for a number token: how many dots it has (2/12→1 … 6/8→5).
export const pips = (n: number | null): number => (n === null ? 0 : 6 - Math.abs(7 - n));

// Tunable weights for the greedy bot. Resource weights nudge it toward
// ore/grain (cities); diversity and ports get small bonuses.
export interface Weights {
  resource: Record<Resource, number>;
  diversity: number;
  port2: number;
  port3: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  resource: { brick: 1.0, lumber: 1.0, wool: 0.9, grain: 1.1, ore: 1.1 },
  diversity: 1.5,
  port2: 1.2,
  port3: 0.6,
};

// Expected production value of a corner: Σ pip-weight × resource-weight over its
// adjacent tiles. Robber is ignored (placement looks at the long run).
export function vertexProduction(state: GameState, v: VertexId, w: Weights = DEFAULT_WEIGHTS): number {
  let score = 0;
  for (const tid of state.board.vertices[v].tiles as TileId[]) {
    const t = state.board.tiles[tid];
    const res = TERRAIN_RESOURCE[t.terrain];
    if (res) score += pips(t.numberToken) * w.resource[res];
  }
  return score;
}

// Placement desirability: production + diversity + a port bonus.
export function vertexScore(state: GameState, v: VertexId, w: Weights = DEFAULT_WEIGHTS): number {
  let score = vertexProduction(state, v, w);
  const seen = new Set<Resource>();
  for (const tid of state.board.vertices[v].tiles as TileId[]) {
    const r = TERRAIN_RESOURCE[state.board.tiles[tid].terrain];
    if (r) seen.add(r);
  }
  score += w.diversity * seen.size;
  const port = state.board.vertices[v].port;
  if (port) score += port.kind === '2:1' ? w.port2 : w.port3;
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

// How much placing the robber on `tile` and stealing from `stealFrom` helps us:
// reward blocking opponents' production (weighted by their VP lead) and stealing
// from card-rich leaders; penalise blocking our own tiles.
export function robberScore(
  state: GameState, player: PlayerId, tile: TileId, stealFrom: PlayerId | null, w: Weights = DEFAULT_WEIGHTS,
): number {
  const t = state.board.tiles[tile];
  const res = TERRAIN_RESOURCE[t.terrain];
  const perBuilding = pips(t.numberToken) * (res ? w.resource[res] : 1);
  let score = 0;
  for (const v of t.vertices) {
    const b = state.buildings[v];
    if (!b) continue;
    const yieldv = perBuilding * (b.kind === 'city' ? 2 : 1);
    score += b.owner === player ? -0.8 * yieldv : yieldv;
  }
  if (stealFrom) {
    score += 0.4 * handSize(state.players[stealFrom].resources);
    score += 0.6 * victoryPoints(state, stealFrom);
  }
  return score;
}

// Choose which cards to drop for a forced discard: shed abundant, low-value
// cards first (keep ore/grain for cities).
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
