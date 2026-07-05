// Persists a finished game (the GameArchive already built for the in-memory
// gameOver broadcast — see room.ts's buildArchive) plus a row per seat, so
// "every game user X played" is a normal indexed query rather than a scan
// over the archive JSON. Called once, fire-and-forget from index.ts's
// broadcastGameOver, after the archive has already gone out over the socket
// — a DB hiccup here must never block or fail the live game finishing for
// the players actually at the table.

import { eq, and, desc, inArray } from 'drizzle-orm';
import type { GameArchive } from '@catan/core';
import { db } from './client';
import { games, gamePlayers } from './schema';
import { seatPlayers, type Room } from '../room';

export async function saveGame(room: Room, archive: GameArchive): Promise<void> {
    const startedAt = room.startedAt ?? new Date();
    await db.transaction(async (tx) => {
        const [row] = await tx
            .insert(games)
            .values({
                roomId: room.id,
                startedAt,
                winnerSeat: archive.winnerSeat,
                turns: archive.turns,
                archive,
            })
            .returning({ id: games.id });

        const players = seatPlayers(room);
        if (players.length) {
            await tx.insert(gamePlayers).values(
                players.map((p) => ({
                    gameId: row.id,
                    seatIndex: p.seatIndex,
                    userId: p.userId,
                    displayName: p.displayName,
                    color: p.color,
                })),
            );
        }
    });
}

export interface GameSummary {
    id: string;
    startedAt: Date;
    finishedAt: Date;
    winnerSeat: number | null;
    turns: number;
    players: { seatIndex: number; userId: string | null; displayName: string; color: string }[];
}

/** Every finished game a given account played in, most recent first. */
export async function listGamesForUser(userId: string): Promise<GameSummary[]> {
    const rows = await db
        .select({ game: games, player: gamePlayers })
        .from(gamePlayers)
        .innerJoin(games, eq(gamePlayers.gameId, games.id))
        .where(eq(gamePlayers.userId, userId))
        .orderBy(desc(games.finishedAt));

    const byId = new Map<string, GameSummary>();
    for (const { game } of rows) {
        if (!byId.has(game.id)) {
            byId.set(game.id, {
                id: game.id, startedAt: game.startedAt, finishedAt: game.finishedAt,
                winnerSeat: game.winnerSeat, turns: game.turns, players: [],
            });
        }
    }
    // Second pass to attach every seat (not just this user's own) to each
    // summary, for an opponent list in the UI — one extra query keeps the
    // first one's WHERE simple (only games this user actually played in).
    const ids = [...byId.keys()];
    if (ids.length) {
        const allPlayers = await db.select().from(gamePlayers).where(inArray(gamePlayers.gameId, ids));
        for (const p of allPlayers) {
            const summary = byId.get(p.gameId);
            if (summary) summary.players.push({ seatIndex: p.seatIndex, userId: p.userId, displayName: p.displayName, color: p.color });
        }
        for (const summary of byId.values()) summary.players.sort((a, b) => a.seatIndex - b.seatIndex);
    }
    return ids.map((id) => byId.get(id)!);
}

/** A single finished game's full archive, gated on the requesting user
 *  actually having been a participant — replays are private by default. */
export async function getGameForUser(gameId: string, userId: string): Promise<{ archive: GameArchive; players: GameSummary['players'] } | null> {
    const participates = await db
        .select({ id: gamePlayers.id })
        .from(gamePlayers)
        .where(and(eq(gamePlayers.gameId, gameId), eq(gamePlayers.userId, userId)))
        .limit(1);
    if (!participates.length) return null;

    const [game] = await db.select().from(games).where(eq(games.id, gameId)).limit(1);
    if (!game) return null;
    const players = await db.select().from(gamePlayers).where(eq(gamePlayers.gameId, gameId));
    return {
        archive: game.archive as GameArchive,
        players: players
            .map((p) => ({ seatIndex: p.seatIndex, userId: p.userId, displayName: p.displayName, color: p.color }))
            .sort((a, b) => a.seatIndex - b.seatIndex),
    };
}
