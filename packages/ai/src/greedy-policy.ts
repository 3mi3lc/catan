import {
  legalActions,
  type GameState, type Action, type PlayerId, type Resource
} from '@catan/core';
import type { Policy } from './policy';
import { ofType } from './policy';
import {
  type Weights, DEFAULT_WEIGHTS, vertexScore, vertexProduction, argmax,
  robberScore, discardAction, roadExpansionValue, bestYopTake,
} from './heuristics';

const STRONG_ROBBER = 3;

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
  if (setts.length) return argmax(setts, (a) => vertexScore(state, a.vertex, w, player))!;

  const ownsSettlement = Object.values(state.buildings).some(
      (b) => b.owner === player && b.kind === 'settlement',
  );
  const cityShort        = ownsSettlement && state.players[player].supply.cities > 0;
  const settlementAvail  = state.players[player].supply.settlements > 0;

  const cityGap  = Math.max(0, 2 - R.grain) + Math.max(0, 3 - R.ore);
  const settGap  = ([R.brick, R.lumber, R.wool, R.grain] as number[]).filter(x => x < 1).length;
  const buildGoal: 'city' | 'sett' | null =
      cityShort && settlementAvail ? (cityGap <= settGap ? 'city' : 'sett') :
          cityShort                   ? 'city' :
              settlementAvail             ? 'sett' :
                  null;

  // 4. Year of Plenty — toward nearest build goal.
  const yop = ofType(acts, 'playYearOfPlenty');
  if (yop.length) {
    const take = bestYopTake(state, player);
    if (take) return { type: 'playYearOfPlenty', take };
  }

  // 5. Bank trade to complete a city (exactly 1 short).
  if (cityShort) {
    const trade = tradeForCity(state, player, ofType(acts, 'bankTrade'));
    if (trade) return trade;
  }

  // 6. Bank trade to complete a settlement (exactly 1 short).
  if (settlementAvail) {
    const trade = tradeForSettlement(state, player, ofType(acts, 'bankTrade'));
    if (trade) return trade;
  }

  // 7. Monopoly — grab a pile worth taking.
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

  // 8. Hand pressure — fire before the discard threshold (≥8).
  //    Only trade toward a concrete build goal to avoid cycling.
  if (hand >= 8) {
    const trades = ofType(acts, 'bankTrade');

    // a) Drain toward city — also fires when 2-short (not just 1-short as in rung 5)
    if (cityShort && cityGap <= 2) {
      const target: Resource = Math.max(0, 2 - R.grain) > 0 ? 'grain' : 'ore';
      const keep: Record<Resource, number> = { grain: 2, ore: 3, brick: 0, lumber: 0, wool: 0 };
      const t = argmax(
          trades.filter(tr => tr.receive === target && R[tr.give] - tr.giveCount >= keep[tr.give]),
          tr => R[tr.give],
      );
      if (t) return t;
    }

    // b) Drain toward settlement — also fires when 2-short
    if (settlementAvail && settGap <= 2) {
      const missing = (['brick', 'lumber', 'wool', 'grain'] as Resource[]).find(r => R[r] < 1);
      if (missing) {
        const keep: Record<Resource, number> = { brick: 1, lumber: 1, wool: 1, grain: 1, ore: 0 };
        const t = argmax(
            trades.filter(tr => tr.receive === missing && R[tr.give] - tr.giveCount >= keep[tr.give]),
            tr => R[tr.give],
        );
        if (t) return t;
      }
    }

    // c) No goal reachable within 2 trades — dump the most-abundant
    //    low-priority resource. Protect city inputs when saving for one.
    const giveOrder: Resource[] = ['wool', 'lumber', 'brick', 'grain', 'ore'];
    const dump = giveOrder
        .flatMap(give => trades.filter(tr => tr.give === give && R[give] >= 4))
        .find(tr => !(cityShort && (tr.give === 'grain' || tr.give === 'ore')));
    if (dump) return dump;
  }

  // 9. Buy a development card whenever affordable — clears hand, builds
  //    toward largest army, may draw a VP card.
  if (ofType(acts, 'buyDevCard').length) return { type: 'buyDevCard' };

  // 10. Expansion road toward the best reachable new settlement spot.
  const roads = ofType(acts, 'buildRoad');
  if (roads.length) {
    if (settlementAvail) {
      const best = argmax(roads, (a) => roadExpansionValue(state, player, a.edge, w));
      if (best && roadExpansionValue(state, player, best.edge, w) > 0) return best;
    }

    // Chase longest road if within striking distance.
    const lrHolder    = state.longestRoad?.player;
    const lrThreshold = lrHolder ? 15 - state.players[lrHolder].supply.roads : 4;
    const ourRoads    = 15 - state.players[player].supply.roads;
    if (lrHolder !== player && ourRoads >= lrThreshold - 2) {
      const best = argmax(roads, (a) => roadExpansionValue(state, player, a.edge, w) + 0.1);
      if (best) return best;
    }
  }

  // 11. Road Building — enumerate the two best edges from the legal pairs
  //    and play the card when it's worth it.
  const roadBuildingActs = ofType(acts, 'playRoadBuilding');
  if (roadBuildingActs.length) {
    const lrHolder    = state.longestRoad?.player;
    const lrThreshold = lrHolder ? 15 - state.players[lrHolder].supply.roads : 4;
    const ourRoads    = 15 - state.players[player].supply.roads;
    const chasingLR   = lrHolder !== player && ourRoads >= lrThreshold - 3;
    const best = argmax(roadBuildingActs, (a) =>
        roadExpansionValue(state, player, a.edges[0], w) +
        roadExpansionValue(state, player, a.edges[1], w) +
        (chasingLR ? 0.2 : 0),
    );
    const score = best
        ? roadExpansionValue(state, player, best.edges[0], w) +
        roadExpansionValue(state, player, best.edges[1], w)
        : 0;

    if (best && (chasingLR || score > 0)) return best;
  }


  // 12. End turn.
  return { type: 'endTurn' };
}

// Trade to complete a city — only fires when exactly 1 resource short.
// giveCount already reflects port ratios so port trades are handled automatically.
function tradeForCity(
    state: GameState,
    player: PlayerId,
    trades: Extract<Action, { type: 'bankTrade' }>[],
): Action | null {
  const R = state.players[player].resources;
  const needGrain = Math.max(0, 2 - R.grain);
  const needOre   = Math.max(0, 3 - R.ore);
  if (needGrain + needOre !== 1) return null;

  const missing: Resource = needGrain > 0 ? 'grain' : 'ore';

  const keep: Record<Resource, number> = {
    grain: 2, ore: 3, brick: 0, lumber: 0, wool: 0,
  };

  const candidates = trades.filter(
      (t) => t.receive === missing && R[t.give] - t.giveCount >= keep[t.give],
  );

  return argmax(candidates, (t) => R[t.give]) ?? null;
}

// Trade to complete a settlement — only fires when exactly 1 resource short.
function tradeForSettlement(
    state: GameState,
    player: PlayerId,
    trades: Extract<Action, { type: 'bankTrade' }>[],
): Action | null {
  const R = state.players[player].resources;
  const needBrick  = Math.max(0, 1 - R.brick);
  const needLumber = Math.max(0, 1 - R.lumber);
  const needWool   = Math.max(0, 1 - R.wool);
  const needGrain  = Math.max(0, 1 - R.grain);

  if (needBrick + needLumber + needWool + needGrain !== 1) return null;

  const missing: Resource =
      needBrick  > 0 ? 'brick'  :
          needLumber > 0 ? 'lumber' :
              needWool   > 0 ? 'wool'   : 'grain';

  const keep: Record<Resource, number> = {
    brick: 1, lumber: 1, wool: 1, grain: 1, ore: 0,
  };

  const candidates = trades.filter(
      (t) => t.receive === missing && R[t.give] - t.giveCount >= keep[t.give],
  );

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

const handTotal = (r: Record<Resource, number>): number =>
    r.brick + r.lumber + r.wool + r.grain + r.ore;

export { DEFAULT_WEIGHTS };
export type { Weights };