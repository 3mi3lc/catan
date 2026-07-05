// Manual verification for the games/game_players persistence layer
// (src/db/games.ts) — exercises saveGame, listGamesForUser, and the
// participant-only privacy gate on getGameForUser against a real Postgres
// instance, without needing to play a full game through to a legitimate win
// (the persistence code doesn't care how the win happened, just the Room/
// GameArchive shape — see room.ts's buildArchive/seatPlayers).
//
// Requires DATABASE_URL pointed at a real (ideally disposable/local)
// Postgres with migrations applied. Run with:
//   DATABASE_URL=... npx tsx scripts/_verify_persistence.ts

import { randomUUID } from 'node:crypto';
import { db } from '../src/db/client';
import { user as userTable } from '../src/db/schema';
import { createRoom, claimSeat, startGame, buildArchive } from '../src/room';
import { saveGame, listGamesForUser, getGameForUser } from '../src/db/games';

function assert(cond: unknown, msg: string): asserts cond {
    if (!cond) throw new Error(`FAILED: ${msg}`);
}

async function main() {
    const hostId = randomUUID();
    const outsiderId = randomUUID();
    await db.insert(userTable).values([
        { id: hostId, name: 'Persistence Test Host', email: `host-${hostId}@example.test` },
        { id: outsiderId, name: 'Persistence Test Outsider', email: `outsider-${outsiderId}@example.test` },
    ]);
    console.log('inserted test users:', hostId, outsiderId);

    const room = createRoom({ name: 'Host', playerToken: 'persist-test-host-token', seats: ['open', 'open'] }, 'fake-socket-0', hostId);
    assert(!('error' in room), 'createRoom failed');
    if ('error' in room) throw new Error(room.error);

    const claimed = claimSeat(room, 1, 'persist-test-guest-token', 'Guest', 'fake-socket-1', null);
    assert(claimed.ok, 'claimSeat failed');

    const started = startGame(room, 'persist-test-host-token');
    assert(started.ok, 'startGame failed');
    assert(room.startedAt instanceof Date, 'startedAt should be set after startGame');

    // Fake a finish without playing a real game — only the persistence path
    // is under test here, not game rules.
    room.state!.winner = room.state!.turnOrder[0];
    room.moves.push({ player: room.state!.turnOrder[0], phase: 'main', action: { type: 'endTurn' } });

    const archive = buildArchive(room);
    await saveGame(room, archive);
    console.log('saved game for room', room.id);

    const mine = await listGamesForUser(hostId);
    assert(mine.length === 1, `expected 1 game for host, got ${mine.length}`);
    assert(mine[0].players.length === 2, `expected 2 players recorded, got ${mine[0].players.length}`);
    assert(mine[0].players.some((p) => p.userId === hostId), 'host userId not recorded on a seat');
    assert(mine[0].players.some((p) => p.userId === null), 'guest seat should have null userId');
    console.log('listGamesForUser(host) OK:', mine[0].id, mine[0].players);

    const fetched = await getGameForUser(mine[0].id, hostId);
    assert(fetched !== null, 'host should be able to fetch their own game');
    assert(fetched!.archive.seed === archive.seed, 'fetched archive seed mismatch');
    console.log('getGameForUser(host) OK — archive round-trips, seed', fetched!.archive.seed);

    const denied = await getGameForUser(mine[0].id, outsiderId);
    assert(denied === null, 'a non-participant must NOT be able to fetch this game (privacy gate)');
    console.log('getGameForUser(outsider) correctly denied — private-by-default confirmed');

    console.log('\nALL PERSISTENCE CHECKS PASSED');
    process.exit(0);
}

main().catch((err) => { console.error(err); process.exit(1); });
