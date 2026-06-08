import {
  legalActions, COSTS, tradeRatio,
  type GameState, type Action, type PlayerId, type Resource, type EdgeId, type VertexId,
} from '@catan/core';
import type { Policy } from './policy';
import { ofType } from './policy';
import {
  type Weights, DEFAULT_WEIGHTS, vertexScore, vertexProduction, argmax, robberScore, discardAction,
} from './heuristics';

const STRONG_ROBBER = 3;

export function greedyPolicy(weights: Weights = DEFAULT_WEIGHTS, name = 'greedy'): Policy {
  return { name, decide: (state, player) => decide(state, player, weights) };
}

function decide(state: GameState, player: PlayerId, w: Weights): Action {
  const acts = legalActions(state, player);
  switch (state.phase) {
    case 'setupSettlement': {
      const best = argmax(ofType(acts, 'buildSettlement'), (a) => vertexScore(state, a.vertex, w));
      return best ?? acts[0];
    }
    case 'setupRoad': {
      const last = state.setup?.lastSettlement;
      const best = argmax(ofType(acts, 'buildRoad'), (a) => {
        const [x, y] = state.board.edges[a.edge].vertices;
        const far = x === last ? y : x;
        return vertexScore(state, far, w);
      });
      return best ?? acts[0];
    }
    case 'roll':
      return { type: 'rollDice' };
    case 'discard':
      return discardAction(state, player);
    case 'moveRobber': {
      const best = argmax(ofType(acts, 'moveRobber'), (a) => robberScore(state, player, a.tile, a.stealFrom, w));
      return best ?? acts[0];
    }
    case 'main':
      return mainDecision(state, player, w, acts);
    default:
      return { type: 'endTurn' };
  }
}

function mainDecision(state: GameState, player: PlayerId, w: Weights, acts: Action[]): Action {
  const R = state.players[player].resources;
  const hand = handTotal(R);

  // 1. Knight — robber relief, largest-army grab, or a strong target.
  const knights = ofType(acts, 'playKnight');
  if (knights.length) {
    const robberOnUs = state.board.tiles[state.robber].vertices.some(
        (v) => state.buildings[v]?.owner === player,
    );
    const best = argmax(knights, (a) => robberScore(state, player, a.robberTo, a.stealFrom, w))!;
    if (
        robberOnUs ||
        gainsLargestArmy(state, player) ||
        robberScore(state, player, best.robberTo, best.stealFrom, w) >= STRONG_ROBBER
    ) {
      return best;
    }
  }

  // 2. City — most VP-efficient build.
  const cities = ofType(acts, 'buildCity');
  if (cities.length) return argmax(cities, (a) => vertexProduction(state, a.vertex, w))!;

  // 3. Settlement — +1 VP and a new income source.
  const setts = ofType(acts, 'buildSettlement');
  if (setts.length) return argmax(setts, (a) => vertexScore(state, a.vertex, w))!;

  const ownsSettlement = Object.values(state.buildings).some(
      (b) => b.owner === player && b.kind === 'settlement',
  );
  const cityShort = ownsSettlement && state.players[player].supply.cities > 0;
  const settlementAvailable = state.players[player].supply.settlements > 0;

  // 4. Year of Plenty — complete a city or settlement this turn.
  const yop = ofType(acts, 'playYearOfPlenty');
  if (yop.length) {
    if (cityShort) {
      const need: Resource[] = [];
      for (let i = R.grain; i < 2; i++) need.push('grain');
      for (let i = R.ore; i < 3; i++) need.push('ore');
      if (
          need.length >= 1 &&
          need.length <= 2 &&
          need.every((r) => state.bank[r] >= need.filter((x) => x === r).length)
      ) {
        while (need.length < 2) need.push('ore');
        return { type: 'playYearOfPlenty', take: [need[0], need[1]] };
      }
    }
    if (settlementAvailable) {
      const missing = missingForSettlement(R);
      if (missing.length >= 1 && missing.length <= 2) {
        const take: [Resource, Resource] = [missing[0], missing[1] ?? missing[0]];
        if (take.every((r) => state.bank[r] > 0)) {
          return { type: 'playYearOfPlenty', take };
        }
      }
    }
  }

  // 5. Bank trade to complete a city (one resource short).
  if (cityShort) {
    const trade = tradeForCity(state, player, ofType(acts, 'bankTrade'));
    if (trade) return trade;
  }

  // 6. Bank trade to complete a settlement (one resource short).
  if (settlementAvailable) {
    const trade = tradeForSettlement(state, player, ofType(acts, 'bankTrade'));
    if (trade) return trade;
  }

  // 7. Hand pressure — trade surplus before it accumulates and attracts the robber.
  //    Bias away from brick/lumber when already road-rich.
  if (hand >= 6) {
    const flush = ofType(acts, 'bankTrade');
    const roadRich = (15 - state.players[player].supply.roads) >= 8;
    const best = argmax(flush, (t) => {
      const base = R[t.give] * (1 / w.resource[t.give]);
      const penalty = roadRich && (t.give === 'brick' || t.give === 'lumber') ? 0.5 : 1;
      return base * penalty;
    });
    if (best) return best;
  }

  // 8. Monopoly — grab a resource pile worth taking.
  const monos = ofType(acts, 'playMonopoly');
  if (monos.length) {
    const totals = Object.fromEntries(
        (['brick', 'lumber', 'wool', 'grain', 'ore'] as Resource[]).map((r) => [
          r,
          state.turnOrder
              .filter((p) => p !== player)
              .reduce((s, p) => s + state.players[p].resources[r], 0),
        ]),
    ) as Record<Resource, number>;
    const best = argmax(monos, (a) => totals[a.resource])!;
    if (totals[best.resource] >= 2) return best;
  }

  // 9. Buy a development card whenever affordable — clears hand, builds toward
  //    largest army, and may draw a VP card.
  if (ofType(acts, 'buyDevCard').length) return { type: 'buyDevCard' };

  // 10. Expansion road toward the best reachable new settlement spot.
  const roads = ofType(acts, 'buildRoad');
  if (roads.length) {
    if (settlementAvailable) {
      const best = argmax(roads, (a) => roadExpansionValue(state, a.edge, w));
      if (best && roadExpansionValue(state, best.edge, w) > 0) return best;
    }

    // Chase longest road if within striking distance.
    const lrHolder = state.longestRoad?.player;
    const ourRoads = 15 - state.players[player].supply.roads;
    const lrThreshold = lrHolder ? 15 - state.players[lrHolder].supply.roads : 4;
    if (lrHolder !== player && ourRoads >= lrThreshold - 2) {
      const best = argmax(roads, (a) => roadExpansionValue(state, a.edge, w) + 0.1);
      if (best) return best;
    }
  }

  // 11. End turn.
  return { type: 'endTurn' };
}

// Resources still needed to build a settlement (brick + lumber + wool + grain).
function missingForSettlement(R: Record<Resource, number>): Resource[] {
  const missing: Resource[] = [];
  if (R.brick < 1)  missing.push('brick');
  if (R.lumber < 1) missing.push('lumber');
  if (R.wool < 1)   missing.push('wool');
  if (R.grain < 1)  missing.push('grain');
  return missing;
}

// Trade for a settlement when exactly one resource short, without spending
// resources already allocated toward it.
function tradeForSettlement(
    state: GameState,
    player: PlayerId,
    trades: Extract<Action, { type: 'bankTrade' }>[],
): Action | null {
  const R = state.players[player].resources;
  const missing = missingForSettlement(R);
  if (missing.length !== 1) return null;
  const need = missing[0];
  // Don't give away anything the settlement itself needs.
  const protected_: Resource[] = ['brick', 'lumber', 'wool', 'grain'];
  const candidates = trades.filter(
      (t) => t.receive === need && !protected_.includes(t.give),
  );
  return argmax(candidates, (t) => R[t.give]) ?? null;
}

// Trade for a city when exactly one resource short, without spending what the
// city needs.
function tradeForCity(
    state: GameState,
    player: PlayerId,
    trades: Extract<Action, { type: 'bankTrade' }>[],
): Action | null {
  const R = state.players[player].resources;
  const needGrain = Math.max(0, 2 - R.grain);
  const needOre = Math.max(0, 3 - R.ore);
  if (needGrain + needOre !== 1) return null;
  const missing: Resource = needGrain > 0 ? 'grain' : 'ore';
  const candidates = trades.filter(
      (t) => t.receive === missing && t.give !== 'grain' && t.give !== 'ore',
  );
  return argmax(candidates, (t) => R[t.give]) ?? null;
}

function roadExpansionValue(state: GameState, edge: EdgeId, w: Weights): number {
  let best = 0;
  for (const v of state.board.edges[edge].vertices as VertexId[]) {
    if (state.buildings[v]) continue;
    const blocked = state.board.vertices[v].edges.some((e) => {
      const [a, b] = state.board.edges[e].vertices;
      const nb = a === v ? b : a;
      return !!state.buildings[nb];
    });
    if (!blocked) best = Math.max(best, vertexScore(state, v, w));
  }
  return best;
}

function gainsLargestArmy(state: GameState, player: PlayerId): boolean {
  const k = state.players[player].playedKnights + 1;
  if (k < 3) return false;
  const la = state.largestArmy;
  if (!la) return true;
  if (la.player === player) return false;
  return k > state.players[la.player].playedKnights;
}

const handTotal = (r: Record<Resource, number>): number =>
    r.brick + r.lumber + r.wool + r.grain + r.ore;

export { DEFAULT_WEIGHTS };
export type { Weights };