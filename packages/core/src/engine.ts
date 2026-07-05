import { GameState, Player, Negotiation, TradeReply, VICTORY_POINTS_TO_WIN } from './state';
import { PlayerId, TileId, VertexId, EdgeId } from './ids';
import { Resource, RESOURCES } from './board';
import { Action, Move, GameEvent, ResourceBundle } from './actions';
import {
  COSTS, canAfford, applyBundle, handSize, bundleTotal,
  canPlaceSettlement, roadConnected, recomputeLongestRoad, recomputeLargestArmy,
  victoryPoints, tradeRatio, distribute, vertexYield, rngStep,
} from './rules';

// The result of attempting a move. The engine NEVER throws on an illegal move —
// it returns { ok: false }, so the UI (gray out the button) and the server
// (reject the message) can handle failure the same way.
export type ApplyResult =
    | { ok: true; state: GameState; events: GameEvent[] }
    | { ok: false; error: string };

const ok = (state: GameState, events: GameEvent[] = []): ApplyResult => ({ ok: true, state, events });
const err = (error: string): ApplyResult => ({ ok: false, error });

// --- small immutable helpers ---------------------------------------------

function withPlayer(state: GameState, id: PlayerId, patch: Partial<Player>): GameState['players'] {
  return { ...state.players, [id]: { ...state.players[id], ...patch } };
}

function removeOne<T>(arr: T[], item: T): T[] {
  const i = arr.indexOf(item);
  return i < 0 ? arr.slice() : [...arr.slice(0, i), ...arr.slice(i + 1)];
}

// Whoever sits at setup placement index k (snake order: forward, then reverse).
function setupPlayerAt(order: PlayerId[], k: number): PlayerId {
  const n = order.length;
  return k < n ? order[k] : order[2 * n - 1 - k];
}

// The current player wins the moment they reach the target on their own turn.
function checkWin(state: GameState): GameState {
  if (state.winner) return state;
  if (victoryPoints(state, state.currentPlayer) >= VICTORY_POINTS_TO_WIN) {
    return { ...state, winner: state.currentPlayer, phase: 'gameOver' };
  }
  return state;
}

// Players the robber could steal from on a tile: anyone but `player` with a
// building there and at least one card. Shared by the handler and legalActions.
function robberVictims(state: GameState, player: PlayerId, tile: TileId): PlayerId[] {
  const victims = new Set<PlayerId>();
  for (const v of state.board.tiles[tile].vertices) {
    const b = state.buildings[v];
    if (b && b.owner !== player && handSize(state.players[b.owner].resources) > 0) victims.add(b.owner);
  }
  return [...victims];
}

// Move the robber and (optionally) steal one random resource from a victim with
// a building on the target tile. Shared by playKnight and moveRobber.
function applyRobber(
    state: GameState,
    player: PlayerId,
    tile: TileId,
    stealFrom: PlayerId | null,
): { ok: true; state: GameState; stolen: Resource | null } | { ok: false; error: string } {
  const victims = robberVictims(state, player, tile);

  if (stealFrom === null) {
    if (victims.length > 0) return { ok: false, error: 'You must steal from a player on that tile' };
    return { ok: true, state: { ...state, robber: tile }, stolen: null };
  }
  if (!victims.includes(stealFrom)) return { ok: false, error: 'Cannot steal from that player' };

  // Pick a uniformly random resource card from the victim's hand, advancing the
  // game's PRNG so the steal is reproducible.
  const hand: Resource[] = [];
  const vr = state.players[stealFrom].resources;
  (Object.keys(vr) as Resource[]).forEach((r) => { for (let i = 0; i < vr[r]; i++) hand.push(r); });
  const { value, next } = rngStep(state.rng);
  const stolen = hand[Math.floor(value * hand.length)];

  const players = {
    ...state.players,
    [stealFrom]: { ...state.players[stealFrom], resources: { ...vr, [stolen]: vr[stolen] - 1 } },
    [player]: {
      ...state.players[player],
      resources: { ...state.players[player].resources, [stolen]: state.players[player].resources[stolen] + 1 },
    },
  };
  return { ok: true, state: { ...state, robber: tile, players, rng: next }, stolen };
}

// Gate shared by all dev-card plays.
function devCardGate(state: GameState, player: PlayerId, card: Player['devCards'][number]): string | null {
  if (state.currentPlayer !== player) return 'Not your turn';
  if (state.phase !== 'main' && state.phase !== 'roll') return 'You cannot play a development card right now';
  if (state.devCardPlayedThisTurn) return 'You already played a development card this turn';
  if (!state.players[player].devCards.includes(card)) return 'You do not have that card to play';
  return null;
}

// =========================================================================
// Entry point
// =========================================================================

export function applyMove(state: GameState, move: Move): ApplyResult {
  if (state.phase === 'gameOver') return err('The game is over');
  const { action } = move;
  switch (action.type) {
    case 'rollDice':         return rollDice(state, move.player);
    case 'buildSettlement':  return buildSettlement(state, move.player, action.vertex);
    case 'buildCity':        return buildCity(state, move.player, action.vertex);
    case 'buildRoad':        return buildRoad(state, move.player, action.edge);
    case 'buyDevCard':       return buyDevCard(state, move.player);
    case 'playKnight':       return playKnight(state, move.player, action);
    case 'playRoadBuilding': return playRoadBuilding(state, move.player, action);
    case 'playYearOfPlenty': return playYearOfPlenty(state, move.player, action);
    case 'playMonopoly':     return playMonopoly(state, move.player, action);
    case 'moveRobber':       return moveRobber(state, move.player, action);
    case 'discard':          return discard(state, move.player, action);
    case 'bankTrade':        return bankTrade(state, move.player, action);
    case 'offerAddGive':     return offerAdd(state, move.player, 'give', action.resource);
    case 'offerAddWant':     return offerAdd(state, move.player, 'want', action.resource);
    case 'offerBroadcast':   return offerBroadcast(state, move.player);
    case 'offerCancel':      return offerCancel(state, move.player);
    case 'respondAccept':    return respond(state, move.player, 'accept');
    case 'respondReject':    return respond(state, move.player, 'reject');
    case 'counterStart':     return counterStart(state, move.player);
    case 'submitCounter':    return submitCounter(state, move.player);
    case 'confirmTrade':     return confirmTrade(state, move.player, action.to);
    case 'declineAll':       return declineAll(state, move.player);
    case 'endTurn':          return endTurn(state, move.player);
    default:                 return assertNever(action);
  }
}

// =========================================================================
// Handlers
// =========================================================================

function rollDice(state: GameState, player: PlayerId): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  if (state.phase !== 'roll') return err('You cannot roll right now');

  // Two dice draws from the game PRNG; the advanced state rides along.
  const d1 = rngStep(state.rng);
  const d2 = rngStep(d1.next);
  const dice: [number, number] = [1 + Math.floor(d1.value * 6), 1 + Math.floor(d2.value * 6)];
  const rng = d2.next;
  const total = dice[0] + dice[1];
  const rolled: GameEvent = { type: 'diceRolled', dice, total };

  if (total === 7) {
    const pendingDiscards: Record<PlayerId, number> = {};
    for (const p of state.turnOrder) {
      const n = handSize(state.players[p].resources);
      if (n > 7) pendingDiscards[p] = Math.floor(n / 2);
    }
    if (Object.keys(pendingDiscards).length > 0) {
      return ok({ ...state, dice, rng, phase: 'discard', pendingDiscards }, [rolled]);
    }
    return ok({ ...state, dice, rng, phase: 'moveRobber' }, [rolled]);
  }

  const { players, bank, gains } = distribute(state, total);
  return ok({ ...state, dice, rng, players, bank, phase: 'main' }, [rolled, { type: 'resourcesProduced', gains }]);
}

function buildSettlement(state: GameState, player: PlayerId, vertex: VertexId): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  const inSetup = state.phase === 'setupSettlement';
  if (!inSetup && state.phase !== 'main') return err('You cannot build a settlement now');
  if (!canPlaceSettlement(state, vertex)) return err('Too close to another building, or occupied');

  const me = state.players[player];

  if (inSetup) {
    const index = state.setup!.placed;
    let next: GameState = {
      ...state,
      players: withPlayer(state, player, { supply: { ...me.supply, settlements: me.supply.settlements - 1 } }),
      buildings: { ...state.buildings, [vertex]: { kind: 'settlement', owner: player } },
      setup: { placed: index, lastSettlement: vertex },
      phase: 'setupRoad',
    };
    // The second settlement (placed in the reverse round) yields starting cards.
    if (index >= state.turnOrder.length) {
      const yield_ = vertexYield(next, vertex);
      next = {
        ...next,
        bank: applyBundle(next.bank, yield_, -1),
        players: withPlayer(next, player, { resources: applyBundle(next.players[player].resources, yield_, +1) }),
      };
    }
    return ok(next, [{ type: 'built', player, what: 'settlement' }]);
  }

  // Normal build: must touch your own road, pay the cost, respect supply.
  if (!state.board.vertices[vertex].edges.some((e) => state.roads[e] === player))
    return err('A settlement must connect to one of your roads');
  if (me.supply.settlements <= 0) return err('No settlements left');
  if (!canAfford(me.resources, COSTS.settlement)) return err('Not enough resources');

  let next: GameState = {
    ...state,
    players: withPlayer(state, player, {
      resources: applyBundle(me.resources, COSTS.settlement, -1),
      supply: { ...me.supply, settlements: me.supply.settlements - 1 },
    }),
    bank: applyBundle(state.bank, COSTS.settlement, +1),
    buildings: { ...state.buildings, [vertex]: { kind: 'settlement', owner: player } },
  };
  next = { ...next, longestRoad: recomputeLongestRoad(next) }; // may split an opponent's road
  return ok(checkWin(next), [{ type: 'built', player, what: 'settlement' }]);
}

function buildRoad(state: GameState, player: PlayerId, edge: EdgeId): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  const inSetup = state.phase === 'setupRoad';
  if (!inSetup && state.phase !== 'main') return err('You cannot build a road now');
  if (state.roads[edge]) return err('That edge already has a road');

  const me = state.players[player];

  if (inSetup) {
    const ends = state.board.edges[edge].vertices;
    if (!ends.includes(state.setup!.lastSettlement as VertexId))
      return err('Your setup road must touch the settlement you just placed');

    const players = withPlayer(state, player, { supply: { ...me.supply, roads: me.supply.roads - 1 } });
    const roads = { ...state.roads, [edge]: player };
    const placed = state.setup!.placed + 1;

    if (placed >= 2 * state.turnOrder.length) {
      // Setup complete — first player begins the first real turn.
      return ok({
        ...state, players, roads, setup: null,
        phase: 'roll', currentPlayer: state.turnOrder[0],
        devCardPlayedThisTurn: false, tradesThisTurn: 0,
      }, [{ type: 'built', player, what: 'road' }]);
    }
    return ok({
      ...state, players, roads,
      setup: { placed, lastSettlement: null },
      phase: 'setupSettlement', currentPlayer: setupPlayerAt(state.turnOrder, placed),
    }, [{ type: 'built', player, what: 'road' }]);
  }

  if (!roadConnected(state, player, edge)) return err('A road must connect to your network');
  if (me.supply.roads <= 0) return err('No roads left');
  if (!canAfford(me.resources, COSTS.road)) return err('Not enough resources');

  let next: GameState = {
    ...state,
    players: withPlayer(state, player, {
      resources: applyBundle(me.resources, COSTS.road, -1),
      supply: { ...me.supply, roads: me.supply.roads - 1 },
    }),
    bank: applyBundle(state.bank, COSTS.road, +1),
    roads: { ...state.roads, [edge]: player },
  };
  next = { ...next, longestRoad: recomputeLongestRoad(next) };
  return ok(checkWin(next), [{ type: 'built', player, what: 'road' }]);
}

function buildCity(state: GameState, player: PlayerId, vertex: VertexId): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  if (state.phase !== 'main') return err('You cannot build a city now');
  const b = state.buildings[vertex];
  if (!b || b.owner !== player || b.kind !== 'settlement') return err('You need your own settlement here');
  const me = state.players[player];
  if (me.supply.cities <= 0) return err('No cities left');
  if (!canAfford(me.resources, COSTS.city)) return err('Not enough resources');

  const next: GameState = {
    ...state,
    players: withPlayer(state, player, {
      resources: applyBundle(me.resources, COSTS.city, -1),
      // The upgraded settlement returns to the supply.
      supply: { ...me.supply, cities: me.supply.cities - 1, settlements: me.supply.settlements + 1 },
    }),
    bank: applyBundle(state.bank, COSTS.city, +1),
    buildings: { ...state.buildings, [vertex]: { kind: 'city', owner: player } },
  };
  return ok(checkWin(next), [{ type: 'built', player, what: 'city' }]);
}

function buyDevCard(state: GameState, player: PlayerId): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  if (state.phase !== 'main') return err('You cannot buy a card now');
  if (state.devDeck.length === 0) return err('No development cards left');
  const me = state.players[player];
  if (!canAfford(me.resources, COSTS.devCard)) return err('Not enough resources');

  const [card, ...rest] = state.devDeck;
  const next: GameState = {
    ...state,
    players: withPlayer(state, player, {
      resources: applyBundle(me.resources, COSTS.devCard, -1),
      pendingDevCards: [...me.pendingDevCards, card], // playable next turn
    }),
    bank: applyBundle(state.bank, COSTS.devCard, +1),
    devDeck: rest,
  };
  return ok(checkWin(next), [{ type: 'devCardBought', player }]); // a VP card can win immediately
}

function playKnight(state: GameState, player: PlayerId, a: Extract<Action, { type: 'playKnight' }>): ApplyResult {
  const gate = devCardGate(state, player, 'knight');
  if (gate) return err(gate);
  if (a.robberTo === state.robber) return err('The robber must move to a new tile');

  const moved = applyRobber(state, player, a.robberTo, a.stealFrom);
  if (!moved.ok) return err(moved.error);

  let next: GameState = {
    ...moved.state,
    players: withPlayer(moved.state, player, {
      devCards: removeOne(moved.state.players[player].devCards, 'knight'),
      playedKnights: moved.state.players[player].playedKnights + 1,
    }),
    devCardPlayedThisTurn: true,
  };
  next = { ...next, largestArmy: recomputeLargestArmy(next) };
  return ok(checkWin(next), [{ type: 'robberMoved', tile: a.robberTo, from: a.stealFrom, stolen: moved.stolen }]);
}

function playRoadBuilding(
    state: GameState, player: PlayerId, a: Extract<Action, { type: 'playRoadBuilding' }>,
): ApplyResult {
  const gate = devCardGate(state, player, 'roadBuilding');
  if (gate) return err(gate);

  let s = state;
  for (const edge of a.edges) {
    if (s.roads[edge]) return err('That edge already has a road');
    if (!roadConnected(s, player, edge)) return err('A road must connect to your network');
    if (s.players[player].supply.roads <= 0) return err('No roads left');
    s = {
      ...s,
      players: withPlayer(s, player, {
        supply: { ...s.players[player].supply, roads: s.players[player].supply.roads - 1 },
      }),
      roads: { ...s.roads, [edge]: player },
    };
  }
  s = {
    ...s,
    players: withPlayer(s, player, { devCards: removeOne(s.players[player].devCards, 'roadBuilding') }),
    devCardPlayedThisTurn: true,
  };
  s = { ...s, longestRoad: recomputeLongestRoad(s) };
  return ok(checkWin(s), a.edges.map(() => ({ type: 'built', player, what: 'road' as const })));
}

function playYearOfPlenty(
    state: GameState, player: PlayerId, a: Extract<Action, { type: 'playYearOfPlenty' }>,
): ApplyResult {
  const gate = devCardGate(state, player, 'yearOfPlenty');
  if (gate) return err(gate);

  const want: ResourceBundle = {};
  for (const r of a.take) want[r] = (want[r] ?? 0) + 1;
  if (!canAfford(state.bank, want)) return err('The bank cannot supply those resources');

  const me = state.players[player];
  const next: GameState = {
    ...state,
    bank: applyBundle(state.bank, want, -1),
    players: withPlayer(state, player, {
      resources: applyBundle(me.resources, want, +1),
      devCards: removeOne(me.devCards, 'yearOfPlenty'),
    }),
    devCardPlayedThisTurn: true,
  };
  return ok(next);
}

function playMonopoly(
    state: GameState, player: PlayerId, a: Extract<Action, { type: 'playMonopoly' }>,
): ApplyResult {
  const gate = devCardGate(state, player, 'monopoly');
  if (gate) return err(gate);

  const res = a.resource;
  let taken = 0;
  const players = { ...state.players };
  for (const p of state.turnOrder) {
    if (p === player) continue;
    const amt = players[p].resources[res];
    if (amt > 0) {
      taken += amt;
      players[p] = { ...players[p], resources: { ...players[p].resources, [res]: 0 } };
    }
  }
  const me = players[player];
  players[player] = {
    ...me,
    resources: { ...me.resources, [res]: me.resources[res] + taken },
    devCards: removeOne(me.devCards, 'monopoly'),
  };
  return ok({ ...state, players, devCardPlayedThisTurn: true });
}

function moveRobber(state: GameState, player: PlayerId, a: Extract<Action, { type: 'moveRobber' }>): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  if (state.phase !== 'moveRobber') return err('It is not time to move the robber');
  if (a.tile === state.robber) return err('The robber must move to a new tile');

  const moved = applyRobber(state, player, a.tile, a.stealFrom);
  if (!moved.ok) return err(moved.error);
  return ok({ ...moved.state, phase: 'main' }, [
    { type: 'robberMoved', tile: a.tile, from: a.stealFrom, stolen: moved.stolen },
  ]);
}

function discard(state: GameState, player: PlayerId, a: Extract<Action, { type: 'discard' }>): ApplyResult {
  if (state.phase !== 'discard') return err('No discards are required right now');
  const owed = state.pendingDiscards[player] ?? 0;
  if (owed <= 0) return err('You do not need to discard');
  if (bundleTotal(a.resources) !== owed) return err(`You must discard exactly ${owed} cards`);
  if (!canAfford(state.players[player].resources, a.resources)) return err('You do not have those cards');

  const pendingDiscards = { ...state.pendingDiscards };
  delete pendingDiscards[player];

  let next: GameState = {
    ...state,
    players: withPlayer(state, player, {
      resources: applyBundle(state.players[player].resources, a.resources, -1),
    }),
    bank: applyBundle(state.bank, a.resources, +1),
    pendingDiscards,
  };
  // Once everyone has discarded, the roller relocates the robber.
  if (Object.keys(pendingDiscards).length === 0) next = { ...next, phase: 'moveRobber' };
  return ok(next);
}

function bankTrade(state: GameState, player: PlayerId, a: Extract<Action, { type: 'bankTrade' }>): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  if (state.phase !== 'main') return err('You cannot trade now');
  if (a.give === a.receive) return err('Trade must be for a different resource');
  const ratio = tradeRatio(state, player, a.give);
  if (a.giveCount !== ratio) return err(`This trade needs ${ratio} ${a.give} (your best rate)`);
  if (state.players[player].resources[a.give] < ratio) return err('Not enough to trade');
  if (state.bank[a.receive] < 1) return err('The bank is out of that resource');

  const me = state.players[player];
  const resources = { ...me.resources, [a.give]: me.resources[a.give] - ratio, [a.receive]: me.resources[a.receive] + 1 };
  const bank = { ...state.bank, [a.give]: state.bank[a.give] + ratio, [a.receive]: state.bank[a.receive] - 1 };
  return ok({ ...state, players: withPlayer(state, player, { resources }), bank });
}

// ── Player-to-player trade negotiation ──────────────────────────────────────
// The active player composes an offer (offerAdd…) and broadcasts it; each
// opponent accepts / rejects / counters; the proposer then confirms with one of
// them or declines. Counters are single-level. See Negotiation in state.ts.

const GIVE_CAP = 5;              // max TOTAL cards offered in one trade (give side)
const WANT_CAP = 3;             // max TOTAL cards requested in one trade (want side)
                                // — total caps bound how many compose steps an
                                //   offer can take, which keeps games short.
                                // (Per-turn offer count is bounded by
                                //  state.maxOffersPerTurn — see GameState.)

// Add one card to the give/want side of the active draft (the proposer's initial
// offer when no negotiation is open, otherwise a responding opponent's counter).
function offerAdd(
  state: GameState, player: PlayerId, side: 'give' | 'want', resource: Resource,
): ApplyResult {
  if (state.phase !== 'main') return err('You cannot trade now');
  const draft = state.draftOffer;

  if (!draft) {
    // Start the proposer's initial offer (opponents counter via counterStart).
    if (state.negotiation) return err('Make a counter-offer instead');
    if (state.currentPlayer !== player) return err('Not your turn');
    if (side !== 'give') return err('Start an offer with a card to give');
    if (state.tradesThisTurn >= state.maxOffersPerTurn) return err('No more offers this turn');
    if ((state.players[player].resources[resource] ?? 0) < 1) return err('You have none of that');
    return ok({ ...state, draftOffer: { by: player, give: { [resource]: 1 }, want: {} } });
  }

  if (draft.by !== player) return err('Not your draft');
  const cur = draft[side];
  const next = (cur[resource] ?? 0) + 1;
  if (side === 'give') {
    if (next > (state.players[player].resources[resource] ?? 0)) return err('You do not have that many to give');
    if (bundleTotal(draft.give) >= GIVE_CAP) return err('That offer is already large enough');
  }
  if (side === 'want' && bundleTotal(draft.want) >= WANT_CAP) return err('That is more than you can request');
  return ok({ ...state, draftOffer: { ...draft, [side]: { ...cur, [resource]: next } } });
}

// Finalise the proposer's draft and broadcast it to every opponent.
function offerBroadcast(state: GameState, player: PlayerId): ApplyResult {
  const draft = state.draftOffer;
  if (!draft || draft.by !== player) return err('No offer to broadcast');
  if (state.currentPlayer !== player || state.negotiation) return err('Cannot broadcast now');
  if (bundleTotal(draft.give) === 0 || bundleTotal(draft.want) === 0)
    return err('An offer needs something on both sides');
  const responses: Record<PlayerId, TradeReply> = {};
  for (const p of state.turnOrder) if (p !== player) responses[p] = 'pending';
  return ok({
    ...state, draftOffer: null, tradesThisTurn: state.tradesThisTurn + 1,
    negotiation: { proposer: player, give: draft.give, want: draft.want, responses, counters: {}, stage: 'responding' },
  });
}

// Abandon the current draft. A counterer reverts to still-pending.
function offerCancel(state: GameState, player: PlayerId): ApplyResult {
  const draft = state.draftOffer;
  if (!draft || draft.by !== player) return err('No offer to cancel');
  return ok({ ...state, draftOffer: null });
}

// Move to arbitration once every opponent has replied to the broadcast.
function advanceNegotiation(state: GameState, neg: Negotiation): GameState {
  const allIn = state.turnOrder.every((p) => p === neg.proposer || neg.responses[p] !== 'pending');
  return { ...state, negotiation: { ...neg, stage: allIn ? 'arbitrating' : 'responding' } };
}

// An opponent accepts or rejects the broadcast offer.
function respond(state: GameState, player: PlayerId, reply: 'accept' | 'reject'): ApplyResult {
  const neg = state.negotiation;
  if (!neg || neg.stage !== 'responding') return err('No offer to respond to');
  if (state.draftOffer) return err('Finish your counter first');
  if (neg.responses[player] !== 'pending') return err('That offer is not awaiting you');
  if (reply === 'accept' && !canAfford(state.players[player].resources, neg.want))
    return err('You cannot cover the requested resources');
  return ok(advanceNegotiation(state, { ...neg, responses: { ...neg.responses, [player]: reply } }));
}

// An opponent begins composing a counter-offer (built up via offerAdd).
function counterStart(state: GameState, player: PlayerId): ApplyResult {
  const neg = state.negotiation;
  if (!neg || neg.stage !== 'responding') return err('No offer to counter');
  if (state.draftOffer) return err('Already composing');
  if (neg.responses[player] !== 'pending') return err('That offer is not awaiting you');
  return ok({ ...state, draftOffer: { by: player, give: {}, want: {} } });
}

// An opponent submits their composed counter terms.
function submitCounter(state: GameState, player: PlayerId): ApplyResult {
  const neg = state.negotiation;
  const draft = state.draftOffer;
  if (!neg || neg.stage !== 'responding' || !draft || draft.by !== player)
    return err('No counter to submit');
  if (bundleTotal(draft.give) === 0 || bundleTotal(draft.want) === 0)
    return err('A counter needs something on both sides');
  const responses = { ...neg.responses, [player]: 'counter' as TradeReply };
  const counters = { ...neg.counters, [player]: { give: draft.give, want: draft.want } };
  return ok(advanceNegotiation({ ...state, draftOffer: null }, { ...neg, responses, counters }));
}

// The proposer executes the trade with one accepter or counterer.
function confirmTrade(state: GameState, player: PlayerId, to: PlayerId): ApplyResult {
  const neg = state.negotiation;
  if (!neg || neg.stage !== 'arbitrating' || player !== neg.proposer) return err('Not arbitrating');
  const reply = neg.responses[to];
  if (reply !== 'accept' && reply !== 'counter') return err('That player is not available');

  // Accept → the broadcast terms apply. Counter → that opponent's terms apply
  // (they give `give`, want `want`), so the proposer gives `want`, gets `give`.
  const [pGives, pGets] = reply === 'accept'
    ? [neg.give, neg.want]
    : [neg.counters[to].want, neg.counters[to].give];
  const prop = state.players[player];
  const other = state.players[to];
  if (!canAfford(prop.resources, pGives)) return err('You can no longer cover that');
  if (!canAfford(other.resources, pGets)) return err('They can no longer cover that');

  const players = {
    ...state.players,
    [player]: { ...prop, resources: applyBundle(applyBundle(prop.resources, pGives, -1), pGets, +1) },
    [to]: { ...other, resources: applyBundle(applyBundle(other.resources, pGets, -1), pGives, +1) },
  };
  return ok({ ...state, players, negotiation: null, draftOffer: null },
    [{ type: 'tradeExecuted', between: [player, to], proposerGives: pGives, proposerGets: pGets }]);
}

// The proposer ends the negotiation without trading.
function declineAll(state: GameState, player: PlayerId): ApplyResult {
  const neg = state.negotiation;
  if (!neg || neg.stage !== 'arbitrating' || player !== neg.proposer) return err('Not arbitrating');
  return ok({ ...state, negotiation: null });
}

function endTurn(state: GameState, player: PlayerId): ApplyResult {
  if (state.currentPlayer !== player) return err('Not your turn');
  if (state.phase !== 'main') return err('Finish the current phase first');

  const order = state.turnOrder;
  const nextPlayer = order[(order.indexOf(player) + 1) % order.length];
  const me = state.players[player];

  return ok({
    ...state,
    players: withPlayer(state, player, { devCards: [...me.devCards, ...me.pendingDevCards], pendingDevCards: [] }),
    currentPlayer: nextPlayer,
    phase: 'roll',
    dice: null,
    devCardPlayedThisTurn: false,
    tradesThisTurn: 0,
    negotiation: null,
    draftOffer: null,
  });
}

// Every development-card play available to `player` right now, fully
// parameterised so an AI can branch over them directly. Playable in both the
// 'roll' and 'main' phases, one card per turn.
function devCardActions(state: GameState, player: PlayerId): Action[] {
  const acts: Action[] = [];
  if (state.currentPlayer !== player) return acts;
  if (state.phase !== 'main' && state.phase !== 'roll') return acts;
  if (state.devCardPlayedThisTurn) return acts;
  const hand = state.players[player].devCards;

  // Knight: relocate the robber to any other tile and steal from a victim there
  // (or from nobody if the tile is unoccupied).
  if (hand.includes('knight')) {
    for (const t of Object.keys(state.board.tiles) as TileId[]) {
      if (t === state.robber) continue;
      const victims = robberVictims(state, player, t);
      if (victims.length === 0) acts.push({ type: 'playKnight', robberTo: t, stealFrom: null });
      else for (const v of victims) acts.push({ type: 'playKnight', robberTo: t, stealFrom: v });
    }
  }

  // Monopoly: name any one resource.
  if (hand.includes('monopoly')) {
    for (const r of RESOURCES) acts.push({ type: 'playMonopoly', resource: r });
  }

  // Year of Plenty: take any two resources the bank can actually supply
  // (unordered, repeats allowed).
  if (hand.includes('yearOfPlenty')) {
    for (let i = 0; i < RESOURCES.length; i++)
      for (let j = i; j < RESOURCES.length; j++) {
        const take: [Resource, Resource] = [RESOURCES[i], RESOURCES[j]];
        const need: ResourceBundle = {};
        for (const r of take) need[r] = (need[r] ?? 0) + 1;
        if (canAfford(state.bank, need)) acts.push({ type: 'playYearOfPlenty', take });
      }
  }

  // Road Building: two free roads. Enumerate distinct unordered pairs where the
  // second road is legal once the first is placed (the in-game sequencing).
  if (hand.includes('roadBuilding') && state.players[player].supply.roads >= 2) {
    const edges = Object.keys(state.board.edges) as EdgeId[];
    const legalFirst = edges.filter((e) => !state.roads[e] && roadConnected(state, player, e));
    const seen = new Set<string>();
    for (const e1 of legalFirst) {
      const after: GameState = { ...state, roads: { ...state.roads, [e1]: player } };
      for (const e2 of edges) {
        if (after.roads[e2] || !roadConnected(after, player, e2)) continue;
        const key = [e1, e2].sort().join('|');
        if (seen.has(key)) continue;
        seen.add(key);
        acts.push({ type: 'playRoadBuilding', edges: [e1, e2] as [EdgeId, EdgeId] });
      }
    }
  }

  return acts;
}

// =========================================================================
// legalActions — enumeration for UI buttons and simple/learned AIs. Player-to-
// player trades are a full negotiation: the active player composes a bundle
// (offerAddGive/Want) and broadcasts it; opponents accept/reject/counter; the
// proposer confirms with one or declines. These steps can belong to a player
// other than currentPlayer (off-turn), so they precede the phase enumeration.
// The discard step is UI-driven.
// =========================================================================

export function legalActions(state: GameState, player: PlayerId): Action[] {
  const acts: Action[] = [];
  const board = state.board;

  if (state.phase === 'gameOver' || state.phase === 'discard') return acts;

  // Composing an offer/counter: only the composer may act, building it up then
  // broadcasting (initial) or submitting (counter), or cancelling.
  if (state.draftOffer) {
    const draft = state.draftOffer;
    if (draft.by !== player) return acts;
    const have = state.players[player].resources;
    const giveFull = bundleTotal(draft.give) >= GIVE_CAP;
    const wantFull = bundleTotal(draft.want) >= WANT_CAP;
    for (const r of Object.keys(have) as Resource[]) {
      if (!giveFull && (draft.give[r] ?? 0) < (have[r] ?? 0)) acts.push({ type: 'offerAddGive', resource: r });
      if (!wantFull) acts.push({ type: 'offerAddWant', resource: r });
    }
    if (bundleTotal(draft.give) > 0 && bundleTotal(draft.want) > 0)
      acts.push(state.negotiation ? { type: 'submitCounter' } : { type: 'offerBroadcast' });
    acts.push({ type: 'offerCancel' });
    return acts;
  }

  // An offer is on the table: opponents reply, then the proposer arbitrates.
  if (state.negotiation) {
    const neg = state.negotiation;
    if (neg.stage === 'responding') {
      if (neg.responses[player] !== 'pending') return acts;     // already replied / not involved
      acts.push({ type: 'respondReject' });
      if (canAfford(state.players[player].resources, neg.want)) acts.push({ type: 'respondAccept' });
      acts.push({ type: 'counterStart' });
      return acts;
    }
    if (player !== neg.proposer) return acts;                   // arbitrating: proposer only
    for (const to of state.turnOrder) {
      const r = neg.responses[to];
      if (r !== 'accept' && r !== 'counter') continue;
      const pGives = r === 'accept' ? neg.give : neg.counters[to].want;
      const pGets  = r === 'accept' ? neg.want : neg.counters[to].give;
      if (canAfford(state.players[player].resources, pGives) && canAfford(state.players[to].resources, pGets))
        acts.push({ type: 'confirmTrade', to });
    }
    acts.push({ type: 'declineAll' });
    return acts;
  }

  if (state.phase === 'setupSettlement') {
    if (state.currentPlayer !== player) return acts;
    for (const v of Object.keys(board.vertices) as VertexId[])
      if (canPlaceSettlement(state, v)) acts.push({ type: 'buildSettlement', vertex: v });
    return acts;
  }

  if (state.phase === 'setupRoad') {
    if (state.currentPlayer !== player) return acts;
    const last = state.setup?.lastSettlement;
    if (last) for (const e of board.vertices[last].edges) if (!state.roads[e]) acts.push({ type: 'buildRoad', edge: e });
    return acts;
  }

  if (state.phase === 'moveRobber') {
    if (state.currentPlayer !== player) return acts;
    for (const t of Object.keys(board.tiles) as TileId[]) {
      if (t === state.robber) continue;
      const victims = robberVictims(state, player, t);
      if (victims.length === 0) acts.push({ type: 'moveRobber', tile: t, stealFrom: null });
      else for (const v of victims) acts.push({ type: 'moveRobber', tile: t, stealFrom: v });
    }
    return acts;
  }

  if (state.phase === 'roll') {
    if (state.currentPlayer === player) {
      acts.push({ type: 'rollDice' });
      acts.push(...devCardActions(state, player)); // e.g. a knight before rolling
    }
    return acts;
  }

  // main (draft/negotiation already handled at the top of legalActions)
  if (state.currentPlayer !== player) return acts;
  const me = state.players[player];
  const resources = Object.keys(me.resources) as Resource[];

  if (canAfford(me.resources, COSTS.road) && me.supply.roads > 0)
    for (const e of Object.keys(board.edges) as EdgeId[])
      if (!state.roads[e] && roadConnected(state, player, e)) acts.push({ type: 'buildRoad', edge: e });

  if (canAfford(me.resources, COSTS.settlement) && me.supply.settlements > 0)
    for (const v of Object.keys(board.vertices) as VertexId[])
      if (canPlaceSettlement(state, v) && board.vertices[v].edges.some((e) => state.roads[e] === player))
        acts.push({ type: 'buildSettlement', vertex: v });

  if (canAfford(me.resources, COSTS.city) && me.supply.cities > 0)
    for (const v of Object.keys(state.buildings) as VertexId[]) {
      const b = state.buildings[v];
      if (b.owner === player && b.kind === 'settlement') acts.push({ type: 'buildCity', vertex: v });
    }

  if (canAfford(me.resources, COSTS.devCard) && state.devDeck.length > 0) acts.push({ type: 'buyDevCard' });

  for (const give of resources) {
    const ratio = tradeRatio(state, player, give);
    if (me.resources[give] >= ratio)
      for (const receive of resources)
        if (receive !== give && state.bank[receive] > 0) acts.push({ type: 'bankTrade', give, giveCount: ratio, receive });
  }

  // Start a player-to-player offer by adding the first card you will give; the
  // draft is then built up incrementally and broadcast. Up to a few per turn.
  if (state.tradesThisTurn < state.maxOffersPerTurn)
    for (const r of resources)
      if (me.resources[r] >= 1) acts.push({ type: 'offerAddGive', resource: r });

  acts.push(...devCardActions(state, player));
  acts.push({ type: 'endTurn' });
  return acts;
}

function assertNever(x: never): never {
  throw new Error(`Unhandled action: ${JSON.stringify(x)}`);
}