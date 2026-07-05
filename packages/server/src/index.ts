import { createServer } from 'node:http';
import { Server, type Socket } from 'socket.io';
import {
    toClientView, legalActions,
    type ClientToServerEvents, type ServerToClientEvents,
    type RoomSummary, type GameEvent, type PlayerId,
} from '@catan/core';
import {
    getRoom, createRoom, reattach, claimSeat, startGame, applyClientMove,
    markDisconnected, advanceAiTurns, toSeatConfig, buildArchive, type Room,
} from './room';
import { saveGame, listGamesForUser, getGameForUser } from './db/games';
import { auth } from './auth';
import { toNodeHandler } from 'better-auth/node';
import { serveStatic } from './static';

const PORT = Number(process.env.PORT ?? 8080);
const CORS_ORIGIN = process.env.CORS_ORIGIN ?? '*';

const authHandler = toNodeHandler(auth);

// Manual CORS handling — this server predates any framework and stays on
// raw http.createServer deliberately (see README), so credentialed
// cross-origin requests to /api/auth/* need their headers set by hand. A
// wildcard origin is rejected at auth.ts's load time (browsers refuse
// `Access-Control-Allow-Origin: *` combined with credentials anyway), so
// CORS_ORIGIN is guaranteed to be a concrete origin here once auth is live.
function applyCors(req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse): void {
    const origin = req.headers.origin;
    if (origin && (CORS_ORIGIN === '*' || origin === CORS_ORIGIN)) {
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Access-Control-Allow-Credentials', 'true');
        res.setHeader('Vary', 'Origin');
    }
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization');
}

// Shared by the socket handshake and the /games/* REST routes below — both
// need a Fetch-API Headers object to hand to auth.api.getSession, but start
// from differently-shaped Node header bags (IncomingHttpHeaders vs
// Socket.io's handshake.headers).
function toFetchHeaders(raw: Record<string, string | string[] | undefined>): Headers {
    const headers = new Headers();
    for (const [key, value] of Object.entries(raw)) {
        if (typeof value === 'string') headers.set(key, value);
    }
    return headers;
}

async function sessionForRequest(req: import('node:http').IncomingMessage) {
    return auth.api.getSession({ headers: toFetchHeaders(req.headers) }).catch(() => null);
}

function sendJson(res: import('node:http').ServerResponse, status: number, body: unknown): void {
    res.writeHead(status, { 'content-type': 'application/json' });
    res.end(JSON.stringify(body));
}

const httpServer = createServer((req, res) => {
    if (req.url === '/health') { res.writeHead(200, { 'content-type': 'text/plain' }); res.end('ok'); return; }

    if (req.url?.startsWith('/api/auth')) {
        applyCors(req, res);
        if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
        void authHandler(req, res);
        return;
    }

    if (req.url === '/games/mine' || req.url?.startsWith('/games/')) {
        applyCors(req, res);
        if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
        if (req.method !== 'GET') { sendJson(res, 405, { error: 'Method not allowed' }); return; }
        void (async () => {
            const session = await sessionForRequest(req);
            if (!session) { sendJson(res, 401, { error: 'Not signed in' }); return; }
            if (req.url === '/games/mine') {
                sendJson(res, 200, await listGamesForUser(session.user.id));
                return;
            }
            const gameId = req.url!.slice('/games/'.length);
            const game = await getGameForUser(gameId, session.user.id);
            // 404, not 403, for "exists but you weren't a participant" — same
            // reasoning as auth's no-enumeration login errors: don't let a
            // response distinguish "wrong game id" from "private game".
            if (!game) { sendJson(res, 404, { error: 'No such game' }); return; }
            sendJson(res, 200, game);
        })();
        return;
    }

    // Last resort: the built web app (see static.ts). Matters for the
    // "tunnel this and share a link with friends" workflow — see README —
    // where the server is the only thing exposed, so it needs to serve the
    // HTML/JS itself rather than assuming a separate static host.
    void (async () => {
        if (await serveStatic(req, res)) return;
        res.writeHead(404); res.end();
    })();
});

const io = new Server<ClientToServerEvents, ServerToClientEvents>(httpServer, {
    cors: { origin: CORS_ORIGIN, credentials: true },
});

// Resolve the better-auth session (if any) for a Socket.io connection by
// reading the same session cookie the browser already sends — ties a socket
// back to a userId with no client-side change needed. Session-less
// (logged-out/guest) connections are allowed through; `userId` is just null
// for them, same as today.
io.use(async (socket, next) => {
    const result = await auth.api.getSession({ headers: toFetchHeaders(socket.handshake.headers) }).catch(() => null);
    (socket.data as { userId: string | null }).userId = result?.user.id ?? null;
    next();
});

// Lobby membership (everyone currently watching a room, seated or not) is a
// transport concern, not game logic — kept here, not in room.ts.
const roomSockets = new Map<string, Set<string>>();
const addToRoom = (roomId: string, socketId: string) =>
    (roomSockets.get(roomId) ?? roomSockets.set(roomId, new Set()).get(roomId)!).add(socketId);
const removeFromRoom = (roomId: string, socketId: string) => roomSockets.get(roomId)?.delete(socketId);

function seatIndexFor(room: Room, socketId: string): number {
    return room.seats.findIndex((s) => s.type === 'human' && s.socketId === socketId);
}

function summaryFor(room: Room, socketId: string): RoomSummary {
    return {
        roomId: room.id,
        seats: room.seats.map(toSeatConfig),
        hostSeat: room.seats.findIndex((s) => s.type === 'human' && s.token === room.hostToken),
        started: room.started,
        yourSeat: ((idx) => (idx === -1 ? null : idx))(seatIndexFor(room, socketId)),
    };
}

function broadcastRoom(room: Room): void {
    for (const socketId of roomSockets.get(room.id) ?? []) {
        io.to(socketId).emit('roomUpdate', summaryFor(room, socketId));
    }
}

function broadcastState(room: Room, events: GameEvent[]): void {
    if (!room.state) return;
    for (const socketId of roomSockets.get(room.id) ?? []) {
        const seatIdx = seatIndexFor(room, socketId);
        const viewerSeat: PlayerId | null = seatIdx === -1 ? null : room.state.turnOrder[seatIdx];
        // legalActions already returns [] when it isn't this seat's turn or
        // decision to make, so spectators (viewerSeat: null) just get [].
        const mine = viewerSeat ? legalActions(room.state, viewerSeat) : [];
        io.to(socketId).emit('gameState', toClientView(room.state, viewerSeat), mine, events);
    }
}

function broadcastGameOver(room: Room): void {
    if (room.archiveSent || !room.state?.winner) return;
    room.archiveSent = true;
    const archive = buildArchive(room);
    for (const socketId of roomSockets.get(room.id) ?? []) io.to(socketId).emit('gameOver', archive);
    // Best-effort: the archive has already reached every live player above,
    // so a DB outage here must not be allowed to affect them — log and move
    // on rather than throwing back into settle()'s caller.
    saveGame(room, archive).catch((err) => console.error(`Failed to persist game for room ${room.id}:`, err));
}

// Pacing for streamed AI moves (see settle() below). Long enough that a room
// full of AI seats — e.g. the whole setup-placement round before the first
// human turn — reads as a sequence of individual turns rather than a burst
// dumped on screen all at once; the dice-roll bump specifically gives the
// client's ~1.6s roll animation (DiceOverlay) room to actually be seen
// before the next move lands and moves the log on.
const AI_MOVE_DELAY_MS = 600;
const AI_DICE_ROLL_EXTRA_MS = 1200;
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** After any move (human or AI), broadcast it immediately, then let every AI
 *  seat play out its turn(s) — streaming each individual AI move to the room
 *  as it happens (paced) rather than batching the whole chain into one
 *  broadcast at the end, which would otherwise freeze every viewer's screen
 *  for as long as all of it takes to compute. */
async function settle(room: Room, events: GameEvent[]): Promise<void> {
    broadcastState(room, events);
    await advanceAiTurns(room, async (moveEvents) => {
        await sleep(AI_MOVE_DELAY_MS + (moveEvents.some((e) => e.type === 'diceRolled') ? AI_DICE_ROLL_EXTRA_MS : 0));
        broadcastState(room, moveEvents);
    });
    broadcastGameOver(room);
}

io.on('connection', (socket: Socket<ClientToServerEvents, ServerToClientEvents>) => {
    // Per-connection identity, fixed at createRoom/joinRoom time — claimSeat
    // and startGame need this before the socket necessarily occupies a seat.
    let joinedRoomId: string | null = null;
    let playerToken: string | null = null;
    let playerName: string | null = null;

    const userId = (socket.data as { userId: string | null }).userId;

    socket.on('createRoom', (req, cb) => {
        const room = createRoom(req, socket.id, userId);
        if ('error' in room) { cb({ error: room.error }); return; }
        joinedRoomId = room.id;
        playerToken = req.playerToken;
        playerName = req.name;
        addToRoom(room.id, socket.id);
        cb({ roomId: room.id });
        broadcastRoom(room);
    });

    socket.on('joinRoom', (req, cb) => {
        const room = getRoom(req.roomId);
        if (!room) { cb({ error: 'No such room' }); return; }
        joinedRoomId = room.id;
        playerToken = req.playerToken;
        playerName = req.name;
        addToRoom(room.id, socket.id);
        reattach(room, req.playerToken, socket.id, userId); // no-op if this token holds no seat here
        cb(summaryFor(room, socket.id));
        broadcastRoom(room);
        if (room.started && room.state) broadcastState(room, []);
    });

    socket.on('claimSeat', ({ seatIndex }) => {
        if (!joinedRoomId || !playerToken || !playerName) { socket.emit('actionError', 'Join a room first'); return; }
        const room = getRoom(joinedRoomId);
        if (!room) return;
        const res = claimSeat(room, seatIndex, playerToken, playerName, socket.id, userId);
        if (!res.ok) { socket.emit('actionError', res.error); return; }
        broadcastRoom(room);
    });

    socket.on('startGame', () => {
        if (!joinedRoomId || !playerToken) { socket.emit('actionError', 'Join a room first'); return; }
        const room = getRoom(joinedRoomId);
        if (!room) return;
        const res = startGame(room, playerToken);
        if (!res.ok) { socket.emit('actionError', res.error); return; }
        broadcastRoom(room);
        void settle(room, []);
    });

    socket.on('move', ({ move }) => {
        if (!joinedRoomId || !playerToken) { socket.emit('actionError', 'Join a room first'); return; }
        const room = getRoom(joinedRoomId);
        if (!room) return;
        const res = applyClientMove(room, playerToken, move);
        if (!res.ok) { socket.emit('actionError', res.error); return; }
        void settle(room, res.events);
    });

    socket.on('leaveRoom', () => {
        if (!joinedRoomId) return;
        const room = getRoom(joinedRoomId);
        removeFromRoom(joinedRoomId, socket.id);
        if (room) { markDisconnected(room, socket.id); broadcastRoom(room); }
        joinedRoomId = null;
    });

    socket.on('disconnect', () => {
        if (!joinedRoomId) return;
        const room = getRoom(joinedRoomId);
        removeFromRoom(joinedRoomId, socket.id);
        if (room) { markDisconnected(room, socket.id); broadcastRoom(room); }
    });
});

httpServer.listen(PORT, () => {
    console.log(`Catan server listening on :${PORT}`);
});
