/**
 * trade-heuristic.ts — a hand-built player-trade policy, used as a layer on top
 * of a net that does NOT itself trade.
 *
 * Self-play training the net to trade regressed it to baseline (it spent its
 * capacity flailing in the negotiation space). So instead the strong no-trade
 * net plays the game, and this heuristic owns every trade decision: it proposes
 * a 1:1 swap to complete a build, accepts offers that advance a near-complete
 * build (or, lacking one, are clearly lopsided in its favor by card count), and
 * arbitrates/cancels. `heuristicTradeAction` returns the trade move to make
 * right now, or null when it's an ordinary (non-trade) decision the net should
 * handle.
 */

import {
    legalActions, COSTS, RESOURCES,
    type GameState, type Action, type PlayerId, type Resource,
} from '@catan/core';

/** The build we're saving toward (highest priority reachable with 1-2 more
 *  cards) and the resources it consumes. Null when no VP build is that close. */
interface Goal { missing: Resource[]; uses: Set<Resource>; }
function buildGoal(state: GameState, player: PlayerId): Goal | null {
    const R = state.players[player].resources;
    const sup = state.players[player].supply;
    const owns = Object.values(state.buildings).some((b) => b.owner === player && b.kind === 'settlement');
    const goals: Partial<Record<Resource, number>>[] = [];
    if (owns && sup.cities > 0) goals.push(COSTS.city);
    if (sup.settlements > 0)    goals.push(COSTS.settlement);
    for (const cost of goals) {
        const missing: Resource[] = [];
        for (const r of RESOURCES)
            for (let i = 0, short = (cost[r] ?? 0) - R[r]; i < short; i++) missing.push(r);
        if (missing.length >= 1 && missing.length <= 2)
            return { missing, uses: new Set(RESOURCES.filter((r) => (cost[r] ?? 0) > 0)) };
    }
    return null;
}

/** The 1:1 offer to make: give a surplus resource (held ≥2, not needed for the
 *  goal) for the one card that COMPLETES a build. Null if none. */
export function plannedOffer(state: GameState, player: PlayerId): { give: Resource; want: Resource } | null {
    const goal = buildGoal(state, player);
    if (!goal || goal.missing.length !== 1) return null;
    const R = state.players[player].resources;
    let give: Resource | null = null;
    for (const r of RESOURCES)
        if (R[r] >= 2 && !goal.uses.has(r) && !goal.missing.includes(r) && (give === null || R[r] > R[give]))
            give = r;
    return give ? { give, want: goal.missing[0] } : null;
}

/** Accept offers that hand us a resource our near-build needs without costing
 *  us a resource that same build consumes — or, lacking such a goal to check
 *  against, offers that are clearly lopsided in our favor by raw card count
 *  (e.g. 5 cards for 1), which any reasonable player would take. */
function goodOffer(state: GameState, player: PlayerId, neg: NonNullable<GameState['negotiation']>): boolean {
    const goal = buildGoal(state, player);
    if (goal
        && goal.missing.some((r) => (neg.give[r] ?? 0) > 0)
        && !RESOURCES.some((r) => (neg.want[r] ?? 0) > 0 && goal.uses.has(r)))
        return true;

    const giveTotal = RESOURCES.reduce((s, r) => s + (neg.give[r] ?? 0), 0);
    const wantTotal = RESOURCES.reduce((s, r) => s + (neg.want[r] ?? 0), 0);
    return giveTotal >= wantTotal + 2;
}

/**
 * The trade move for `player` while a trade is ALREADY in flight — continuing/
 * sending their draft, replying to an offer, or arbitrating. Null if there's no
 * such pending trade decision for them. (Does not initiate new offers.)
 */
export function tradeFlowAction(state: GameState, player: PlayerId): Action | null {
    if (state.draftOffer?.by === player) {
        const plan = plannedOffer(state, player);
        if (!plan) return { type: 'offerCancel' };
        const d = state.draftOffer;
        if ((d.give[plan.give] ?? 0) < 1) return { type: 'offerAddGive', resource: plan.give };
        if ((d.want[plan.want] ?? 0) < 1) return { type: 'offerAddWant', resource: plan.want };
        return { type: 'offerBroadcast' };
    }
    if (state.negotiation) {
        const neg = state.negotiation;
        if (neg.stage === 'responding' && neg.responses[player] === 'pending') {
            const canAccept = legalActions(state, player).some((a) => a.type === 'respondAccept');
            return canAccept && goodOffer(state, player, neg)
                ? { type: 'respondAccept' } : { type: 'respondReject' };
        }
        if (neg.stage === 'arbitrating' && player === neg.proposer) {
            const accepter = state.turnOrder.find((p) =>
                neg.responses[p] === 'accept'
                && legalActions(state, player).some((a) => a.type === 'confirmTrade' && a.to === p));
            return accepter ? { type: 'confirmTrade', to: accepter } : { type: 'declineAll' };
        }
    }
    return null;
}

/** Open ONE offer per turn if it would complete a build. Null otherwise. */
export function proposeStart(state: GameState, player: PlayerId): Action | null {
    if (state.phase !== 'main' || state.currentPlayer !== player || state.tradesThisTurn !== 0) return null;
    const plan = plannedOffer(state, player);
    if (plan && legalActions(state, player).some((a) => a.type === 'offerAddGive' && a.resource === plan.give))
        return { type: 'offerAddGive', resource: plan.give };
    return null;
}

/**
 * The trade move `player` should make right now, or null if there is no trade
 * decision for them (i.e. the net should pick an ordinary action). Combines the
 * in-flight flow with opening a fresh offer.
 */
export function heuristicTradeAction(state: GameState, player: PlayerId): Action | null {
    return tradeFlowAction(state, player) ?? proposeStart(state, player);
}
