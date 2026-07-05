import { io as ioClient, type Socket } from 'socket.io-client';
import { legalActions } from '@catan/core';
import type { ClientToServerEvents, ServerToClientEvents, RoomSummary, ClientGameState, GameEvent } from '@catan/core';

const URL = process.env.URL ?? 'http://localhost:8080';
const ROOM = process.argv[2];
if (!ROOM) throw new Error('usage: _verify_online_guest.ts <ROOM_CODE>');

type ClientSocket = Socket<ServerToClientEvents, ClientToServerEvents>;
function connect(): ClientSocket { return ioClient(URL, { transports: ['websocket'] }); }
function waitFor<T extends unknown[]>(s: ClientSocket, e: string): Promise<T> {
    return new Promise((resolve) => s.once(e as any, (...a: T) => resolve(a)));
}
function waitUntil<T extends unknown[]>(s: ClientSocket, e: string, pred: (...a: T) => boolean): Promise<T> {
    return new Promise((resolve) => {
        const h = (...a: T) => { if (pred(...a)) { s.off(e as any, h as any); resolve(a); } };
        s.on(e as any, h as any);
    });
}

async function main() {
    const guest = connect();
    await waitFor(guest, 'connect');

    // Attach listeners BEFORE the join ack resolves: if we're reconnecting to
    // an already-started game, the server fires roomUpdate/gameState from
    // inside the very same joinRoom handler, and a listener attached after
    // `await` could miss them.
    const gameStatePromise = waitFor<[ClientGameState, ReturnType<typeof legalActions>, GameEvent[]]>(guest, 'gameState');

    const joined = await new Promise<RoomSummary | { error: string }>((resolve) =>
        guest.emit('joinRoom', { roomId: ROOM, playerToken: 'guest-script-token', name: 'GuestBot' }, resolve));
    if ('error' in joined) throw new Error(`joinRoom failed: ${joined.error}`);
    console.log('joined room, seats:', joined.seats, 'yourSeat:', joined.yourSeat);

    if (joined.yourSeat === null) {
        const claimed = waitUntil<[RoomSummary]>(guest, 'roomUpdate', (s) => s.yourSeat === 1);
        guest.emit('claimSeat', { seatIndex: 1 });
        const [afterClaim] = await claimed;
        console.log('claimed seat 1. yourSeat =', afterClaim.yourSeat);
    } else {
        console.log('already held seat', joined.yourSeat, '(reconnect) — skipping claimSeat');
    }

    const [gState] = await gameStatePromise;
    console.log('game started! phase:', gState.phase, 'currentPlayer:', gState.currentPlayer);
    console.log('my (p1) resources:', gState.players.p1?.resources);
    console.log('host (p0) resources (should be null):', gState.players.p0?.resources);
    console.log('board tiles:', Object.keys(gState.board.tiles).length,
        'vertices:', Object.keys(gState.board.vertices).length,
        'edges:', Object.keys(gState.board.edges).length);
    const firstTile = Object.values(gState.board.tiles)[0] as any;
    console.log('first tile sample:', JSON.stringify(firstTile));
    const firstVertexId = firstTile.vertices[0];
    console.log('that vertex exists in board.vertices?', firstVertexId in gState.board.vertices, firstVertexId);

    // Play a couple of setup moves whenever it's seat p1's turn, to exercise the move path from a second client.
    let state = gState;
    for (let i = 0; i < 20 && state.currentPlayer !== undefined && !state.winner; i++) {
        if (state.currentPlayer !== 'p1') break;
        const acts = legalActions(state as any, 'p1' as any);
        if (!acts.length) break;
        const next = waitFor<[ClientGameState, ReturnType<typeof legalActions>, GameEvent[]]>(guest, 'gameState');
        guest.emit('move', { move: { player: 'p1' as any, action: acts[0] } });
        [state] = await next;
        console.log(`guest played ${acts[0].type}, phase now ${state.phase}, current ${state.currentPlayer}`);
    }

    console.log('OK — guest script joined the real browser-hosted room and played through the server');
    guest.close();
    process.exit(0);
}

main().catch((err) => { console.error(err); process.exit(1); });
