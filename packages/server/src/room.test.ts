import { describe, it, expect } from 'vitest';
import { asPlayerId, loadReplay } from '@catan/core';
import {
    createRoom, claimSeat, startGame, applyClientMove, nextActor, advanceAiTurns, buildArchive,
    type Room,
} from './room';

function freshRoom(seats: ('open' | { ai: 'net' | 'mcts' })[] = ['open', 'open']) {
    const room = createRoom({ name: 'Host', playerToken: 'host-token', seats }, 'host-socket');
    if ('error' in room) throw new Error(room.error);
    return room;
}

describe('createRoom', () => {
    it('rejects seat counts outside 2-4', () => {
        expect(createRoom({ name: 'H', playerToken: 't', seats: ['open'] }, 's')).toEqual({ error: 'Need 2-4 seats' });
        expect(createRoom({ name: 'H', playerToken: 't', seats: ['open', 'open', 'open', 'open', 'open'] }, 's'))
            .toEqual({ error: 'Need 2-4 seats' });
    });

    it('rejects an all-AI config (host needs a seat to occupy)', () => {
        const res = createRoom({ name: 'H', playerToken: 't', seats: [{ ai: 'net' }, { ai: 'net' }] }, 's');
        expect('error' in res).toBe(true);
    });

    it('seats the host in the first open slot', () => {
        const room = freshRoom(['open', { ai: 'net' }, 'open']);
        expect(room.seats[0]).toMatchObject({ type: 'human', token: 'host-token', socketId: 'host-socket' });
        expect(room.seats[1]).toEqual({ type: 'ai', level: 'net' });
        expect(room.seats[2]).toEqual({ type: 'open' });
    });
});

describe('claimSeat', () => {
    it('lets a new token claim an open seat', () => {
        const room = freshRoom();
        const res = claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        expect(res.ok).toBe(true);
        expect(room.seats[1]).toMatchObject({ type: 'human', token: 'guest-token' });
    });

    it('rejects claiming a seat someone else already holds', () => {
        const room = freshRoom();
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        const res = claimSeat(room, 1, 'someone-else', 'Other', 'other-socket');
        expect(res).toEqual({ ok: false, error: 'That seat is not open' });
    });

    it('lets the same token reclaim its own seat (reconnect)', () => {
        const room = freshRoom();
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        const res = claimSeat(room, 1, 'guest-token', 'Guest', 'new-socket-id');
        expect(res.ok).toBe(true);
        expect((room.seats[1] as { socketId: string | null }).socketId).toBe('new-socket-id');
    });

    it('rejects a token claiming a second seat once it already holds one', () => {
        const room = freshRoom(['open', 'open', 'open']);
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        const res = claimSeat(room, 2, 'guest-token', 'Guest', 'guest-socket-2');
        expect(res).toEqual({ ok: false, error: 'You already hold a seat in this room' });
        expect(room.seats[2]).toEqual({ type: 'open' });
    });

    it('rejects claiming once the game has started', () => {
        const room = freshRoom();
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        startGame(room, 'host-token');
        const res = claimSeat(room, 1, 'someone-else', 'Other', 'other-socket');
        expect(res).toEqual({ ok: false, error: 'Game already started' });
    });
});

describe('startGame', () => {
    it('rejects starting before every seat is filled', () => {
        const room = freshRoom();
        expect(startGame(room, 'host-token')).toEqual({ ok: false, error: 'Every seat must be filled first' });
    });

    it('rejects a non-host caller', () => {
        const room = freshRoom(['open', { ai: 'net' }]);
        expect(startGame(room, 'not-the-host')).toEqual({ ok: false, error: 'Only the host can start the game' });
    });

    it('creates a GameState sized to the seat count once every seat is filled', () => {
        const room = freshRoom(['open', { ai: 'net' }, { ai: 'net' }]);
        const res = startGame(room, 'host-token');
        expect(res.ok).toBe(true);
        expect(room.started).toBe(true);
        expect(room.state?.turnOrder.length).toBe(3);
        expect(room.state?.phase).toBe('setupSettlement');
        // Interactive games allow effectively-unlimited player offers per
        // turn (unlike bounded self-play search) — see GameState.maxOffersPerTurn.
        // A large finite number, not Infinity: this GameState is sent to
        // clients as JSON, and JSON.stringify(Infinity) is `null`.
        expect(room.state?.maxOffersPerTurn).toBe(Number.MAX_SAFE_INTEGER);
    });
});

describe('applyClientMove', () => {
    function startedRoom(): Room {
        const room = freshRoom(['open', 'open']);
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        startGame(room, 'host-token');
        return room;
    }

    it('rejects moves before the game has started', () => {
        const room = freshRoom();
        const res = applyClientMove(room, 'host-token', { player: asPlayerId('p0'), action: { type: 'rollDice' } });
        expect(res).toEqual({ ok: false, error: 'Game has not started' });
    });

    it('rejects a token submitting a move for a seat it does not own', () => {
        const room = startedRoom();
        const vertex = Object.keys(room.state!.board.vertices)[0];
        const res = applyClientMove(room, 'guest-token', {
            player: asPlayerId('p0'),
            action: { type: 'buildSettlement', vertex: vertex as any },
        });
        expect(res).toEqual({ ok: false, error: 'Not your seat' });
    });

    it('applies a legal move from the seat that actually owns it', () => {
        const room = startedRoom();
        const vertex = Object.keys(room.state!.board.vertices)[0];
        const res = applyClientMove(room, 'host-token', {
            player: asPlayerId('p0'),
            action: { type: 'buildSettlement', vertex: vertex as any },
        });
        expect(res.ok).toBe(true);
        expect(room.state?.phase).toBe('setupRoad');
    });
});

describe('nextActor', () => {
    it('is the current player in a normal phase', () => {
        const room = freshRoom(['open', 'open']);
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        startGame(room, 'host-token');
        expect(nextActor(room.state!)).toBe(room.state!.currentPlayer);
    });

    it('is null once the game has a winner', () => {
        const room = freshRoom(['open', 'open']);
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        startGame(room, 'host-token');
        room.state = { ...room.state!, winner: asPlayerId('p0') };
        expect(nextActor(room.state!)).toBeNull();
    });

    it('is the draft composer when an offer is being built', () => {
        const room = freshRoom(['open', 'open']);
        claimSeat(room, 1, 'guest-token', 'Guest', 'guest-socket');
        startGame(room, 'host-token');
        room.state = { ...room.state!, draftOffer: { by: asPlayerId('p1'), give: {}, want: {} } };
        expect(nextActor(room.state!)).toBe('p1');
    });
});

describe('advanceAiTurns', () => {
    it('drives every AI seat\'s setup placement until a human seat is up', async () => {
        const room = freshRoom(['open', { ai: 'net' }, { ai: 'net' }]);
        startGame(room, 'host-token');
        expect(room.state!.currentPlayer).toBe('p0'); // human seat goes first; nothing to drive yet

        // Place the host's first settlement+road, then the two AI seats (p1,
        // p2) should play their entire turn on their own.
        const v1 = Object.keys(room.state!.board.vertices)[0];
        applyClientMove(room, 'host-token', { player: asPlayerId('p0'), action: { type: 'buildSettlement', vertex: v1 as any } });
        const edge = room.state!.board.vertices[v1 as any].edges[0];
        applyClientMove(room, 'host-token', { player: asPlayerId('p0'), action: { type: 'buildRoad', edge: edge as any } });

        const events = await advanceAiTurns(room);
        expect(events.some((e) => e.type === 'built')).toBe(true);
        // Snake setup order for 3 players is p0,p1,p2,p2,p1,p0 — after p0's
        // first turn, both AI seats (p1,p2) place before it's p0's turn again.
        expect(room.state!.currentPlayer).toBe('p0');
        expect(room.state!.phase).toBe('setupSettlement');
    }, 20000);
});

describe('buildArchive', () => {
    it('records every applied move (human and AI) into a replayable archive', async () => {
        const room = freshRoom(['open', { ai: 'net' }]);
        startGame(room, 'host-token');
        const v1 = Object.keys(room.state!.board.vertices)[0];
        applyClientMove(room, 'host-token', { player: asPlayerId('p0'), action: { type: 'buildSettlement', vertex: v1 as any } });
        const edge = room.state!.board.vertices[v1 as any].edges[0];
        applyClientMove(room, 'host-token', { player: asPlayerId('p0'), action: { type: 'buildRoad', edge: edge as any } });
        await advanceAiTurns(room);

        expect(room.moves.length).toBeGreaterThan(0);
        const archive = buildArchive(room);
        expect(archive.seed).toBe(room.seed);
        expect(archive.policyNames).toEqual(['Host', 'AI (Blue)']);
        expect(archive.maxOffersPerTurn).toBe(room.state!.maxOffersPerTurn);
        expect(archive.moves.length).toBe(room.moves.length);

        // The whole point: a server-built archive must actually replay to the
        // same state the room is really in — that's what the post-game stats
        // and replay screen will run on.
        const replay = loadReplay(archive);
        expect(replay.at(replay.length - 1)).toEqual(room.state);
    });
});
