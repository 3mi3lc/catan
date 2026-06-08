import { PlayerId, TileId, VertexId, EdgeId } from './ids';
import { Board, Resource } from './board';
import { ResourceBundle } from './actions';

export type PlayerColor = 'red' | 'blue' | 'white' | 'orange';

export type DevCard =
    | 'knight'
    | 'victoryPoint'
    | 'roadBuilding'
    | 'yearOfPlenty'
    | 'monopoly';

// A full hand of resources. Every resource key is always present (often 0),
// which keeps arithmetic simple — no undefined checks.
export type ResourceCounts = Record<Resource, number>;

export interface Player {
  id: PlayerId;
  name: string;
  color: PlayerColor;
  resources: ResourceCounts;
  devCards: DevCard[];        // playable cards in hand
  pendingDevCards: DevCard[]; // bought this turn — not playable until next turn
  playedKnights: number;      // feeds the Largest Army award
  // Unbuilt pieces remaining in this player's personal supply.
  supply: { settlements: number; cities: number; roads: number };
}

export type BuildingKind = 'settlement' | 'city';
export interface Building {
  kind: BuildingKind;
  owner: PlayerId;
}

// An explicit turn state machine. Illegal sequencing (building before rolling,
// ending a turn mid-discard) becomes impossible to express.
export type Phase =
    | 'setupSettlement' // initial placement: place a free settlement
    | 'setupRoad'       // initial placement: place its adjoining road
    | 'roll'            // current player must roll the dice
    | 'discard'         // a 7 was rolled; players holding > 7 cards discard half
    | 'moveRobber'      // the roller relocates the robber and may steal a card
    | 'main'            // build, trade, play dev cards, then end turn
    | 'gameOver';

// A single outstanding player-to-player trade offer.
export interface PendingTrade {
  id: string;
  from: PlayerId;
  to: PlayerId | 'all';
  give: ResourceBundle;
  want: ResourceBundle;
}

export interface GameState {
  board: Board;

  players: Record<PlayerId, Player>;
  turnOrder: PlayerId[];
  currentPlayer: PlayerId;
  phase: Phase;
  dice: readonly [number, number] | null;

  robber: TileId;
  buildings: Record<VertexId, Building>; // occupied corners
  roads: Record<EdgeId, PlayerId>;       // occupied edges

  bank: ResourceCounts; // the bank's finite supply — you really can run out
  devDeck: DevCard[];   // remaining undrawn development cards

  longestRoad: { player: PlayerId; length: number } | null;
  largestArmy: { player: PlayerId; size: number } | null;
  winner: PlayerId | null;

  // Deterministic PRNG state (mulberry32). Every in-game random draw — dice and
  // robber steals — reads and advances this, so a game replays exactly from its
  // seed. It lives in the state (not an external closure) to keep applyMove a
  // pure, serializable reducer.
  rng: number;

  // --- transient bookkeeping ---
  // Initial-placement progress; null once normal play begins.
  setup: { placed: number; lastSettlement: VertexId | null } | null;
  // At most one development card may be played per turn.
  devCardPlayedThisTurn: boolean;
  // After a 7, how many cards each over-the-limit player still owes.
  pendingDiscards: Record<PlayerId, number>;
  // The current open trade offer, if any.
  pendingTrade: PendingTrade | null;
}

export const VICTORY_POINTS_TO_WIN = 10;