// In-memory room/seat model. Deliberately has no Socket.io dependency — it's
// pure game/lobby logic over @catan/core, independently testable, and the
// transport layer (index.ts) decides who gets told what.

import {
    generateBoard, initialGameState, applyMove, rngStep,
    type GameState, type GameEvent, type Move, type PlayerColor,
    type GameArchive, type MoveRecord,
} from '@catan/core';
import type { AiLevel, CreateRoomRequest, SeatConfig } from '@catan/core';
import { getServerBot } from './bots';

const COLORS: PlayerColor[] = ['red', 'blue', 'white', 'orange'];
// hotseat.ts names every seat after its color, never the literal "AI" —
// matters here because a room can have 2+ AI seats, and "AI"/"AI" gives a
// human nothing to distinguish them by when choosing who to rob.
const COLOR_NAME: Record<PlayerColor, string> = { red: 'Red', blue: 'Blue', white: 'White', orange: 'Orange' };
const ROOM_ID_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no 0/O/1/I — easier to read aloud

function rngFromSeed(seed: number) {
    let s = seed >>> 0;
    return () => { const r = rngStep(s); s = r.next; return r.value; };
}

function randomRoomId(): string {
    let id = '';
    for (let i = 0; i < 5; i++) id += ROOM_ID_CHARS[Math.floor(Math.random() * ROOM_ID_CHARS.length)];
    return id;
}

export type Seat =
    | { type: 'open' }
    | { type: 'ai'; level: AiLevel }
    // userId links this seat to a better-auth account, when the claiming
    // socket was authenticated at claim/reattach time (see index.ts's
    // io.use() handshake middleware) — null for guest play, which keeps
    // working exactly as before. This is what lets a finished game show up
    // in that account's "my games" list later (see db/games.ts).
    | { type: 'human'; token: string; name: string; socketId: string | null; userId: string | null };

export interface Room {
    id: string;
    seats: Seat[];
    hostToken: string;
    state: GameState | null;
    started: boolean;
    // Recorded once the game starts, for the post-game stats/replay screen
    // (see buildArchive below) — seed is whatever startGame picked; moves
    // accumulate every applied move, human or AI, in order. startedAt feeds
    // the games table's started_at column.
    seed: number | null;
    startedAt: Date | null;
    moves: MoveRecord[];
    archiveSent: boolean;
}

const rooms = new Map<string, Room>();

export function getRoom(id: string): Room | undefined {
    return rooms.get(id);
}

export function toSeatConfig(seat: Seat): SeatConfig {
    if (seat.type === 'open') return { type: 'open' };
    if (seat.type === 'ai') return { type: 'ai', level: seat.level };
    return { type: 'taken', name: seat.name, connected: seat.socketId !== null };
}

export function createRoom(
    req: CreateRoomRequest, socketId: string, hostUserId: string | null = null,
): Room | { error: string } {
    if (req.seats.length < 2 || req.seats.length > 4) return { error: 'Need 2-4 seats' };
    const seats: Seat[] = req.seats.map((s) => (s === 'open' ? { type: 'open' } : { type: 'ai', level: s.ai }));
    const hostIdx = seats.findIndex((s) => s.type === 'open');
    if (hostIdx === -1) return { error: 'At least one seat must be open for you to host it' };
    seats[hostIdx] = { type: 'human', token: req.playerToken, name: req.name, socketId, userId: hostUserId };

    let id = randomRoomId();
    while (rooms.has(id)) id = randomRoomId();
    const room: Room = {
        id, seats, hostToken: req.playerToken, state: null, started: false,
        seed: null, startedAt: null, moves: [], archiveSent: false,
    };
    rooms.set(id, room);
    return room;
}

/** Reconnect a token to whichever seat it already occupies, if any. A
 *  non-null userId updates the seat's linked account (so logging in mid-game
 *  still gets the finished game attributed to you); a null userId — a
 *  guest/logged-out reconnect — leaves any existing link untouched rather
 *  than clobbering it, since the absence of a session on THIS reconnect
 *  doesn't mean the account link should be forgotten. */
export function reattach(room: Room, token: string, socketId: string, userId: string | null = null): number | null {
    const idx = room.seats.findIndex((s) => s.type === 'human' && s.token === token);
    if (idx === -1) return null;
    const seat = room.seats[idx] as Extract<Seat, { type: 'human' }>;
    seat.socketId = socketId;
    if (userId) seat.userId = userId;
    return idx;
}

export function markDisconnected(room: Room, socketId: string): void {
    const seat = room.seats.find((s) => s.type === 'human' && s.socketId === socketId);
    if (seat && seat.type === 'human') seat.socketId = null;
}

export function claimSeat(
    room: Room, seatIndex: number, token: string, name: string, socketId: string, userId: string | null = null,
): { ok: true } | { ok: false; error: string } {
    if (room.started) return { ok: false, error: 'Game already started' };
    const seat = room.seats[seatIndex];
    if (!seat) return { ok: false, error: 'No such seat' };
    if (seat.type === 'human' && seat.token === token) {
        seat.socketId = socketId; // reclaiming your own seat
        if (userId) seat.userId = userId;
        return { ok: true };
    }
    if (seat.type !== 'open') return { ok: false, error: 'That seat is not open' };
    if (room.seats.some((s) => s.type === 'human' && s.token === token))
        return { ok: false, error: 'You already hold a seat in this room' };
    room.seats[seatIndex] = { type: 'human', token, name, socketId, userId };
    return { ok: true };
}

export function startGame(room: Room, callerToken: string): { ok: true } | { ok: false; error: string } {
    if (room.started) return { ok: false, error: 'Already started' };
    if (callerToken !== room.hostToken) return { ok: false, error: 'Only the host can start the game' };
    if (room.seats.some((s) => s.type === 'open')) return { ok: false, error: 'Every seat must be filled first' };

    // The server picks the seed — a client must never be trusted with this,
    // or it could predict/replay dice rolls and robber steals.
    const seed = (Math.random() * 0x1_0000_0000) >>> 0;
    const board = generateBoard(rngFromSeed(seed));
    const seats = room.seats.map((s, i) => ({
        id: `p${i}`,
        name: s.type === 'human' ? s.name : `AI (${COLOR_NAME[COLORS[i]]})`,
        color: COLORS[i],
    }));
    // Unlimited player-to-player offers per turn — this is interactive play,
    // not bounded self-play search (see GameState.maxOffersPerTurn). Use a
    // large finite cap rather than Infinity: GameState crosses the wire as
    // JSON for online play (unlike hotseat.ts, which never serializes it),
    // and JSON.stringify(Infinity) is `null` — `tradesThisTurn >= null`
    // coerces to `>= 0`, which is true immediately and hides the offer UI
    // from turn one.
    room.state = initialGameState(board, seats, seed, Number.MAX_SAFE_INTEGER);
    room.seed = seed;
    room.startedAt = new Date();
    room.started = true;
    return { ok: true };
}

export function applyClientMove(
    room: Room, callerToken: string, move: Move,
): { ok: true; events: GameEvent[] } | { ok: false; error: string } {
    if (!room.state) return { ok: false, error: 'Game has not started' };
    const seatIdx = room.state.turnOrder.indexOf(move.player);
    const seat = room.seats[seatIdx];
    if (!seat || seat.type !== 'human' || seat.token !== callerToken)
        return { ok: false, error: 'Not your seat' };
    const phase = room.state.phase;
    const res = applyMove(room.state, move);
    if (!res.ok) return { ok: false, error: res.error };
    room.state = res.state;
    room.moves.push({ player: move.player, phase, action: move.action });
    return { ok: true, events: res.events };
}

/** Who must act next, regardless of seat type — a build/trade/discard actor
 *  derivation mirroring hotseat.ts's negotiationActor()/botActor(), just
 *  without the "is it the human's turn" framing (that's the caller's job). */
export function nextActor(state: GameState): import('@catan/core').PlayerId | null {
    if (state.winner) return null;
    if (state.draftOffer) return state.draftOffer.by;
    const neg = state.negotiation;
    if (neg) {
        if (neg.stage === 'responding')
            return state.turnOrder.find((p) => p !== neg.proposer && neg.responses[p] === 'pending') ?? null;
        return neg.proposer;
    }
    if (state.phase === 'discard') {
        const owed = Object.keys(state.pendingDiscards);
        return owed.length ? (owed[0] as import('@catan/core').PlayerId) : null;
    }
    return state.currentPlayer;
}

export function seatTypeOf(room: Room, player: import('@catan/core').PlayerId): Seat | null {
    if (!room.state) return null;
    const idx = room.state.turnOrder.indexOf(player);
    return room.seats[idx] ?? null;
}

/** Drives every AI seat's turn (and off-turn negotiation/discard decisions)
 *  until a human seat is up or the game ends — fully server-side, no client
 *  involvement. Streams each individual move's events to `onMove` as it
 *  happens (awaited before continuing), rather than only handing back the
 *  full list once every AI seat is done — a room with several AI seats (e.g.
 *  the whole setup-placement round before the first human turn) would
 *  otherwise leave every viewer's screen frozen for as long as all of that
 *  computation takes, then dump the whole batch at once. `onMove` is where
 *  index.ts broadcasts + paces each move; still returns every event in
 *  order for callers that just want the final tally (tests, non-streaming
 *  use). */
export async function advanceAiTurns(
    room: Room, onMove?: (events: GameEvent[]) => void | Promise<void>,
): Promise<GameEvent[]> {
    const events: GameEvent[] = [];
    while (room.state && !room.state.winner) {
        const actor = nextActor(room.state);
        if (!actor) break;
        const idx = room.state.turnOrder.indexOf(actor);
        const seat = room.seats[idx];
        if (!seat || seat.type !== 'ai') break;

        const bot = await getServerBot(room.state.turnOrder.length);
        const action = await bot.decide(room.state, actor, seat.level);
        const phase = room.state.phase;
        const res = applyMove(room.state, { player: actor, action });
        if (!res.ok) {
            // The bot only ever proposes legal moves in practice; bail rather
            // than spin forever if that ever stops being true.
            console.error(`AI seat ${actor} proposed an illegal move:`, action, res.error);
            break;
        }
        room.state = res.state;
        room.moves.push({ player: actor, phase, action });
        events.push(...res.events);
        if (onMove) await onMove(res.events);
    }
    return events;
}

/** A GameArchive for the post-game stats/replay screen — call once the game
 *  has a winner (see index.ts's settle(), guarded by room.archiveSent). */
export function buildArchive(room: Room): GameArchive {
    const state = room.state!;
    return {
        seed: room.seed!,
        policyNames: room.seats.map((s, i) => (s.type === 'human' ? s.name : `AI (${COLOR_NAME[COLORS[i]]})`)),
        winnerSeat: state.winner ? state.turnOrder.indexOf(state.winner) : null,
        turns: room.moves.filter((m) => m.action.type === 'endTurn').length,
        moves: room.moves,
        maxOffersPerTurn: state.maxOffersPerTurn,
    };
}

/** Per-seat metadata for the games/game_players persistence layer (see
 *  db/games.ts) — display name, colour, and linked account (null for guests
 *  and AI seats) for every seat in turn order. Separate from buildArchive
 *  since the archive itself has no notion of accounts. */
export function seatPlayers(room: Room): { seatIndex: number; userId: string | null; displayName: string; color: PlayerColor }[] {
    return room.seats.map((s, i) => ({
        seatIndex: i,
        userId: s.type === 'human' ? s.userId : null,
        displayName: s.type === 'human' ? s.name : `AI (${COLOR_NAME[COLORS[i]]})`,
        color: COLORS[i],
    }));
}
