// End-of-game statistics, computed by replaying a GameArchive once. See
// replay.ts for the replay mechanics this builds on.

import { loadReplay, GameArchive, NO_ROBBER_TILE } from './replay';
import { distribute } from './rules';
import { RESOURCES, Resource } from './board';
import { PlayerId } from './ids';
import { DevCard } from './state';
import { ResourceBundle, GameEvent, Action } from './actions';

export type ResourceFlowCategory =
  | 'production' | 'bankTrade' | 'playerTrade' | 'robberSteal'
  | 'discarded' | 'monopoly' | 'yearOfPlenty' | 'devCardCost';

const CATEGORIES: ResourceFlowCategory[] = [
  'production', 'bankTrade', 'playerTrade', 'robberSteal',
  'discarded', 'monopoly', 'yearOfPlenty', 'devCardCost',
];

// Each move causes resource flow from exactly one of these causes (or none,
// for moves like buildSettlement that move VP/pieces but no cards) — so a
// move's action type alone disambiguates how to bucket its resource diff.
function categoryFor(actionType: Action['type']): ResourceFlowCategory | null {
  switch (actionType) {
    case 'rollDice': return 'production';
    case 'bankTrade': return 'bankTrade';
    case 'confirmTrade': return 'playerTrade';
    case 'playKnight': case 'moveRobber': return 'robberSteal';
    case 'discard': return 'discarded';
    case 'playMonopoly': return 'monopoly';
    case 'playYearOfPlenty': return 'yearOfPlenty';
    case 'buyDevCard': return 'devCardCost';
    default: return null;
  }
}

export interface RobberStats {
  timesMoved: number;
  stolenFromOthers: ResourceBundle;
  lostToOthers: ResourceBundle;
  timesTargeted: number;
  timesNoVictim: number;
  // Production this player would have received on a roll, but didn't,
  // because the robber sat on the matching tile — see computeGameStats.
  productionBlocked: ResourceBundle;
}

export interface TradeStats {
  bankTradesCount: number;
  offersBroadcast: number;
  counterOffersMade: number;
  tradesCompleted: number;
  tradesByPartner: Partial<Record<PlayerId, number>>;
  offersDeclined: number;
}

export interface PlayerStats {
  player: PlayerId;
  name: string;
  finalVP: number;
  vpBreakdown: { settlements: number; cities: number; longestRoad: number; largestArmy: number; devCards: number };
  diceRolls: number[];
  resourcesGained: ResourceBundle;
  resourcesLost: ResourceBundle;
  gainedByCategory: Record<ResourceFlowCategory, ResourceBundle>;
  lostByCategory: Record<ResourceFlowCategory, ResourceBundle>;
  devCardsBought: number;
  devCardsPlayed: { knight: number; roadBuilding: number; yearOfPlenty: number; monopoly: number };
  // Every dev card this player drew, by type (played + still in hand) — the
  // sum equals devCardsBought. Mirrors Colonist's "Development Cards Drawn".
  devCardsDrawn: Record<DevCard, number>;
  robber: RobberStats;
  trade: TradeStats;
}

export interface GameStats {
  players: PlayerStats[];
  dice: { rollCounts: Record<number, number>; totalRolls: number };
  turns: number;
  totalMoves: number;
  winner: PlayerId | null;
}

function emptyBundle(): ResourceBundle { return {}; }
function addInto(bundle: ResourceBundle, resource: Resource, amount: number): void {
  if (amount === 0) return;
  bundle[resource] = (bundle[resource] ?? 0) + amount;
}
function freshPlayerStats(player: PlayerId, name: string): PlayerStats {
  return {
    player, name, finalVP: 0,
    vpBreakdown: { settlements: 0, cities: 0, longestRoad: 0, largestArmy: 0, devCards: 0 },
    diceRolls: [],
    resourcesGained: emptyBundle(),
    resourcesLost: emptyBundle(),
    gainedByCategory: Object.fromEntries(CATEGORIES.map((c) => [c, emptyBundle()])) as Record<ResourceFlowCategory, ResourceBundle>,
    lostByCategory: Object.fromEntries(CATEGORIES.map((c) => [c, emptyBundle()])) as Record<ResourceFlowCategory, ResourceBundle>,
    devCardsBought: 0,
    devCardsPlayed: { knight: 0, roadBuilding: 0, yearOfPlenty: 0, monopoly: 0 },
    devCardsDrawn: { knight: 0, victoryPoint: 0, roadBuilding: 0, yearOfPlenty: 0, monopoly: 0 },
    robber: {
      timesMoved: 0, stolenFromOthers: emptyBundle(), lostToOthers: emptyBundle(),
      timesTargeted: 0, timesNoVictim: 0, productionBlocked: emptyBundle(),
    },
    trade: { bankTradesCount: 0, offersBroadcast: 0, counterOffersMade: 0, tradesCompleted: 0, tradesByPartner: {}, offersDeclined: 0 },
  };
}

function eventOfType<T extends GameEvent['type']>(events: GameEvent[], type: T): Extract<GameEvent, { type: T }> | undefined {
  return events.find((e) => e.type === type) as Extract<GameEvent, { type: T }> | undefined;
}

export function computeGameStats(archive: GameArchive): GameStats {
  const replay = loadReplay(archive);
  const final = replay.at(replay.length - 1);

  const players: Record<PlayerId, PlayerStats> = {};
  for (const p of final.turnOrder) players[p] = freshPlayerStats(p, final.players[p].name);

  const rollCounts: Record<number, number> = {};
  let totalRolls = 0;

  for (let i = 0; i < archive.moves.length; i++) {
    const move = archive.moves[i];
    const mover = move.player as PlayerId;
    const before = replay.at(i);
    const after = replay.at(i + 1);
    const events = replay.eventsAt(i);
    const category = categoryFor(move.action.type);

    if (category) {
      for (const p of final.turnOrder) {
        for (const r of RESOURCES) {
          const delta = after.players[p].resources[r] - before.players[p].resources[r];
          if (delta > 0) {
            addInto(players[p].resourcesGained, r, delta);
            addInto(players[p].gainedByCategory[category], r, delta);
          } else if (delta < 0) {
            addInto(players[p].resourcesLost, r, -delta);
            addInto(players[p].lostByCategory[category], r, -delta);
          }
        }
      }
    }

    for (const ev of events) {
      if (ev.type === 'diceRolled') {
        rollCounts[ev.total] = (rollCounts[ev.total] ?? 0) + 1;
        totalRolls++;
        players[mover].diceRolls.push(ev.total);
      } else if (ev.type === 'devCardBought') {
        players[ev.player].devCardsBought++;
      } else if (ev.type === 'robberMoved') {
        players[mover].robber.timesMoved++;
        if (ev.stolen && ev.from) {
          addInto(players[mover].robber.stolenFromOthers, ev.stolen, 1);
          addInto(players[ev.from].robber.lostToOthers, ev.stolen, 1);
          players[ev.from].robber.timesTargeted++;
        } else {
          players[mover].robber.timesNoVictim++;
        }
      } else if (ev.type === 'tradeExecuted') {
        const [proposer, partner] = ev.between;
        players[proposer].trade.tradesCompleted++;
        players[partner].trade.tradesCompleted++;
        players[proposer].trade.tradesByPartner[partner] = (players[proposer].trade.tradesByPartner[partner] ?? 0) + 1;
        players[partner].trade.tradesByPartner[proposer] = (players[partner].trade.tradesByPartner[proposer] ?? 0) + 1;
      }
    }

    switch (move.action.type) {
      case 'playKnight': players[mover].devCardsPlayed.knight++; break;
      case 'playRoadBuilding': players[mover].devCardsPlayed.roadBuilding++; break;
      case 'playYearOfPlenty': players[mover].devCardsPlayed.yearOfPlenty++; break;
      case 'playMonopoly': players[mover].devCardsPlayed.monopoly++; break;
      case 'bankTrade': players[mover].trade.bankTradesCount++; break;
      case 'offerBroadcast': players[mover].trade.offersBroadcast++; break;
      case 'submitCounter': players[mover].trade.counterOffersMade++; break;
      case 'respondReject': case 'declineAll': players[mover].trade.offersDeclined++; break;
      default: break;
    }

    // Robber-blocked production: re-run the same production rule
    // (rules.ts's distribute) as if the robber were nowhere, and the
    // difference from what was actually produced is exactly what the
    // robber's placement cost each player on this roll — including the
    // bank-supply-limited edge case, since it's the same function.
    if (move.action.type === 'rollDice') {
      const rolled = eventOfType(events, 'diceRolled');
      if (rolled && rolled.total !== 7) {
        const hypothetical = distribute({ ...before, robber: NO_ROBBER_TILE }, rolled.total);
        const actual = eventOfType(events, 'resourcesProduced');
        for (const p of final.turnOrder) {
          for (const r of RESOURCES) {
            const would = hypothetical.gains[p]?.[r] ?? 0;
            const did = actual?.gains[p]?.[r] ?? 0;
            if (would > did) addInto(players[p].robber.productionBlocked, r, would - did);
          }
        }
      }
    }
  }

  for (const p of final.turnOrder) {
    let settlements = 0, cities = 0;
    for (const b of Object.values(final.buildings)) {
      if (b.owner !== p) continue;
      if (b.kind === 'city') cities++; else settlements++;
    }
    // Dev cards drawn by type: cards still in hand (incl. this-turn pending)
    // plus the ones already spent (played cards are removed from the hand,
    // so the held count alone would undercount them). VP cards are never
    // played, so their held count is their drawn count.
    const held = players[p].devCardsDrawn;
    for (const c of [...final.players[p].devCards, ...final.players[p].pendingDevCards]) held[c]++;
    held.knight += players[p].devCardsPlayed.knight;
    held.roadBuilding += players[p].devCardsPlayed.roadBuilding;
    held.yearOfPlenty += players[p].devCardsPlayed.yearOfPlenty;
    held.monopoly += players[p].devCardsPlayed.monopoly;

    const devCards = held.victoryPoint;
    const longestRoad = final.longestRoad?.player === p ? 2 : 0;
    const largestArmy = final.largestArmy?.player === p ? 2 : 0;
    players[p].vpBreakdown = { settlements, cities, longestRoad, largestArmy, devCards };
    players[p].finalVP = settlements + cities * 2 + devCards + longestRoad + largestArmy;
  }

  return {
    players: final.turnOrder.map((p) => players[p]),
    dice: { rollCounts, totalRolls },
    turns: archive.turns,
    totalMoves: archive.moves.length,
    winner: final.winner,
  };
}
