// Shared client/server contract for networked play. Pure types + one pure
// redaction function — no runtime behavior beyond that. Both packages/server
// and packages/web depend on @catan/core already, so this is the natural
// home for the wire format; no separate "protocol" package is needed.

import { RESOURCES } from './board';
import { PlayerId } from './ids';
import { Move, GameEvent, Action } from './actions';
import { GameState, Player, DevCard, ResourceCounts } from './state';
import { GameArchive } from './replay';

// ── Redacted, per-viewer game state ─────────────────────────────────────────
// The server never puts a full GameState on the wire: every other player's
// exact hand (and the dev-card draw order) would leak hidden information that
// a real board game keeps private. `toClientView` is the one place that
// decides what a given seat (or a spectator, `viewerSeat: null`) gets to see.

export interface ClientPlayer {
  id: PlayerId;
  name: string;
  color: Player['color'];
  // Exact only for the viewer's own seat; null for everyone else, who only
  // see the count (handSize) — real hidden information, not client-side UI
  // hiding of data that was already on the wire.
  resources: ResourceCounts | null;
  handSize: number;
  devCards: DevCard[] | null;
  devCardCount: number;
  pendingDevCardCount: number;
  playedKnights: number;
  supply: Player['supply'];
}

export type ClientGameState = Omit<GameState, 'players' | 'devDeck' | 'rng'> & {
  players: Record<PlayerId, ClientPlayer>;
  devDeckCount: number;
};

export function toClientView(state: GameState, viewerSeat: PlayerId | null): ClientGameState {
  const players: Record<PlayerId, ClientPlayer> = {};
  // Once the game is over, every hand is revealed — there's nothing left to
  // protect, and it lets a winner's true (possibly hidden-VP-card-boosted)
  // total be shown to everyone, not just themselves.
  const revealAll = state.winner !== null;
  for (const id of state.turnOrder) {
    const p = state.players[id];
    const isViewer = id === viewerSeat || revealAll;
    const handSize = RESOURCES.reduce((s, r) => s + p.resources[r], 0);
    players[id] = {
      id: p.id,
      name: p.name,
      color: p.color,
      resources: isViewer ? p.resources : null,
      handSize,
      devCards: isViewer ? p.devCards : null,
      devCardCount: p.devCards.length,
      pendingDevCardCount: p.pendingDevCards.length,
      playedKnights: p.playedKnights,
      supply: p.supply,
    };
  }
  const { players: _players, devDeck, rng: _rng, ...rest } = state;
  return { ...rest, players, devDeckCount: devDeck.length };
}

// ── Lobby / room types ──────────────────────────────────────────────────────

export type AiLevel = 'net' | 'mcts';

export type SeatConfig =
  | { type: 'open' }
  | { type: 'ai'; level: AiLevel }
  | { type: 'taken'; name: string; connected: boolean };

export interface RoomSummary {
  roomId: string;
  seats: SeatConfig[];
  hostSeat: number;
  started: boolean;
  // Which seat (if any) the recipient of this summary currently occupies —
  // computed per-recipient, like the redacted game state.
  yourSeat: number | null;
}

// ── Socket.io event contract ────────────────────────────────────────────────

export interface CreateRoomRequest {
  name: string;
  playerToken: string;
  // Length sets the player count (2-4). The host's own seat is whichever
  // 'open' entry they occupy automatically; the rest are pre-configured as
  // open (for other humans to claim) or AI.
  seats: ('open' | { ai: AiLevel })[];
}

export interface JoinRoomRequest {
  roomId: string;
  playerToken: string;
  name: string;
}

export interface ClaimSeatRequest {
  seatIndex: number;
}

export interface MoveRequest {
  move: Move;
}

export type RoomResult = { roomId: string } | { error: string };
export type JoinResult = RoomSummary | { error: string };

export interface ClientToServerEvents {
  createRoom: (req: CreateRoomRequest, cb: (res: RoomResult) => void) => void;
  joinRoom: (req: JoinRoomRequest, cb: (res: JoinResult) => void) => void;
  claimSeat: (req: ClaimSeatRequest) => void;
  startGame: () => void;
  move: (req: MoveRequest) => void;
  leaveRoom: () => void;
}

export interface ServerToClientEvents {
  roomUpdate: (summary: RoomSummary) => void;
  // `legalActions` is the recipient's OWN legal-action list (from
  // @catan/core's legalActions(state, viewerSeat) — empty when it's not
  // their turn/decision). The redacted ClientGameState is deliberately
  // missing fields (other players' resources, the real devDeck) that
  // legalActions needs, so clients can't compute this themselves — the
  // server, which still holds the real GameState, sends it instead.
  gameState: (state: ClientGameState, legalActionsForMe: Action[], events: GameEvent[]) => void;
  // Fired once, right when the game ends — carries the full move-by-move
  // archive (replayable via @catan/core's loadReplay/computeGameStats) for
  // a post-game stats & replay screen. Not redacted: by the time this fires
  // the game is over and toClientView already reveals every hand, so the
  // same archive is correct for every recipient.
  gameOver: (archive: GameArchive) => void;
  actionError: (message: string) => void;
}
