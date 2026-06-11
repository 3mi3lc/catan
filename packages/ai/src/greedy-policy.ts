import {
  legalActions, COSTS, tradeRatio,
  type GameState, type Action, type PlayerId, type Resource, type EdgeId, type VertexId,
} from '@catan/core';
import type { Policy } from './policy';
import { ofType } from './policy';
import {
  type Weights, DEFAULT_WEIGHTS, vertexScore, vertexProduction, argmax,
  robberScore, discardAction, roadExpansionValue, bestYopTake,
} from './heuristics';

const STRONG_ROBBER = 6; // play a knight aggressively only if the target is juicy

// A competent value-greedy bot: best-production placement in setup, and a
// priority ladder in the main phase (cities → settlements → enabling trades/cards
// → expansion roads → end). Deterministic given the state, so greedy-vs-greedy
// games replay exactly from the seed. `weights` makes its taste tunable.
export function greedyPolicy(weights: Weights = DEFAULT_WEIGHTS, name = 'greedy'): Policy {
  return { name, decide: (state, player) => decide(state, player, weights) };
}

function decide(state: GameState, player: PlayerId, w: Weights): Action {
  const acts = legalActions(state, player);
  switch (state.phase) {
    case 'setupSettlement': {
      const best = argmax(ofType(acts, 'buildSettlement'), (a) => vertexScore(state, a.vertex, w, player));
      return best ?? acts[0];
    }
    case 'setupRoad': {
      const last = state.setup?.lastSettlement;
      const best = argmax(ofType(acts, 'buildRoad'), (a) => {
        const [x, y] = state.board.edges[a.edge].vertices;
        const far = x === last ? y : x;
        return vertexScore(state, far, w, player);
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

  // 1. Knight — robber relief, a clear largest-army grab, or a strong target.
  const knights = ofType(acts, 'playKnight');
  if (knights.length) {
    const robberOnUs = state.board.tiles[state.robber].vertices.some((v) => state.buildings[v]?.owner === player);
    const best = argmax(knights, (a) => robberScore(state, player, a.robberTo, a.stealFrom, w))!;
    if (robberOnUs || gainsLargestArmy(state, player) || robberScore(state, player, best.robberTo, best.stealFrom, w) >= STRONG_ROBBER) {
      return best;
    }
  }

  // 2. City — the most VP-efficient build; double the best producer.
  const cities = ofType(acts, 'buildCity');
  if (cities.length) return argmax(cities, (a) => vertexProduction(state, a.vertex, w))!;

  // 3. Settlement — +1 VP and a new income source.
  const setts = ofType(acts, 'buildSettlement');
  if (setts.length) return argmax(setts, (a) => vertexScore(state, a.vertex, w, player))!;

  const ownsSettlement = Object.values(state.buildings).some((b) => b.owner === player && b.kind === 'settlement');
  const cityShort = ownsSettlement && state.players[player].supply.cities > 0;

  // 4. Year of Plenty — pick the 2 most useful resources toward city/settlement.
  const yop = ofType(acts, 'playYearOfPlenty');
  if (yop.length) {
    const take = bestYopTake(state, player);
    if (take) return { type: 'playYearOfPlenty', take };
  }

  // 5. Bank trade to complete a city we're one resource short of.
  if (cityShort) {
    const trade = tradeForCity(state, player, ofType(acts, 'bankTrade'));
    if (trade) return trade;
  }

  // 6. Monopoly when an opponent stash is worth grabbing.
  const monos = ofType(acts, 'playMonopoly');
  if (monos.length) {
    const totals = Object.fromEntries((['brick', 'lumber', 'wool', 'grain', 'ore'] as Resource[]).map((r) => [
      r, state.turnOrder.filter((p) => p !== player).reduce((s, p) => s + state.players[p].resources[r], 0),
    ])) as Record<Resource, number>;
    const best = argmax(monos, (a) => totals[a.resource])!;
    if (totals[best.resource] >= 4) return best;
  }

  // 7. Buy a development card when flush (knights → army, plus VP cards).
  if (ofType(acts, 'buyDevCard').length && handTotal(R) >= 5) return { type: 'buyDevCard' };

  // 8. Hand pressure — when hand ≥ 8 (one card below the robber discard threshold),
  //    trade surplus resources toward a concrete build goal rather than sitting on
  //    them until a 7 forces a costly discard.
  if (handTotal(R) >= 8) {
    const trades = ofType(acts, 'bankTrade');
    // a) City: rung 5 handles 1-short; here also fire when 2-short.
    if (cityShort) {
      const needGrain = Math.max(0, 2 - R.grain);
      const needOre   = Math.max(0, 3 - R.ore);
      if (needGrain + needOre <= 2) {
        const target: Resource = needGrain > 0 ? 'grain' : 'ore';
        const t = argmax(
            trades.filter(tr => tr.receive === target && tr.give !== 'grain' && tr.give !== 'ore'),
            tr => R[tr.give],
        );
        if (t) return t;
      }
    }
    // b) Settlement: trade toward missing resource, protecting what settlement needs.
    if (state.players[player].supply.settlements > 0) {
      const missing: Resource[] = [];
      if (R.brick  < 1) missing.push('brick');
      if (R.lumber < 1) missing.push('lumber');
      if (R.wool   < 1) missing.push('wool');
      if (R.grain  < 1) missing.push('grain');
      if (missing.length >= 1 && missing.length <= 2) {
        const target = missing[0];
        const safe: Resource[] = ['brick', 'lumber', 'wool', 'grain'];
        const t = argmax(
            trades.filter(tr => tr.receive === target && !safe.includes(tr.give)),
            tr => R[tr.give],
        );
        if (t) return t;
      }
    }
    // c) No specific goal reachable: dump most-abundant low-priority resource.
    //    Protect grain/ore when saving toward a city.
    const giveOrder: Resource[] = ['wool', 'lumber', 'brick', 'grain', 'ore'];
    const t = giveOrder
        .flatMap(give => trades.filter(tr => tr.give === give && R[give] >= 4))
        .find(tr => !(cityShort && (tr.give === 'grain' || tr.give === 'ore')));
    if (t) return t;
  }

  // 9. Expansion road — BFS lookahead so intermediate roads in a chain toward a
  //    good spot score positively (the old 1-hop version returned 0 for them,
  //    causing random direction changes when building road chains).
  const roads = ofType(acts, 'buildRoad');
  if (roads.length && state.players[player].supply.settlements > 0) {
    const best = argmax(roads, (a) => roadExpansionValue(state, player, a.edge, w));
    if (best && roadExpansionValue(state, player, best.edge, w) > 0) return best;
  }

  // 10. Nothing worthwhile — end the turn.
  return { type: 'endTurn' };
}

// If we own a settlement to upgrade and are exactly one resource short of a
// city, find a legal bank trade that fills the gap without spending what the
// city needs.
function tradeForCity(
    state: GameState, player: PlayerId, trades: Extract<Action, { type: 'bankTrade' }>[],
): Action | null {
  const R = state.players[player].resources;
  const needGrain = Math.max(0, 2 - R.grain);
  const needOre = Math.max(0, 3 - R.ore);
  if (needGrain + needOre !== 1) return null;
  const missing: Resource = needGrain > 0 ? 'grain' : 'ore';
  const candidates = trades.filter((t) => t.receive === missing && t.give !== 'grain' && t.give !== 'ore');
  // Prefer giving away the resource we hold the most of.
  return argmax(candidates, (t) => R[t.give]) ?? null;
}

function gainsLargestArmy(state: GameState, player: PlayerId): boolean {
  const k = state.players[player].playedKnights + 1;
  if (k < 3) return false;
  const la = state.largestArmy;
  if (!la) return true;
  if (la.player === player) return false;
  return k > state.players[la.player].playedKnights;
}

const handTotal = (r: Record<Resource, number>): number => r.brick + r.lumber + r.wool + r.grain + r.ore;

export { DEFAULT_WEIGHTS };
export type { Weights };