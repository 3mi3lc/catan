// Drizzle schema for the server's Postgres database.
//
// `user`/`session`/`account`/`verification` are better-auth's standard core
// tables (https://better-auth.com) — shape and naming follow its documented
// schema exactly, since the drizzle adapter reads/writes these directly.
// `account` doubles as the credential store for email+password: better-auth
// keeps the password hash in `account.password` on a row with
// providerId: 'credential', rather than a separate column on `user`.
//
// `games`/`gamePlayers` are this app's own tables, added in the same schema
// file so one `drizzle-kit generate` covers everything.

import { pgTable, text, timestamp, boolean, integer, jsonb } from 'drizzle-orm/pg-core';

export const user = pgTable('user', {
    id: text('id').primaryKey(),
    name: text('name').notNull(),
    email: text('email').notNull().unique(),
    emailVerified: boolean('email_verified').notNull().default(false),
    image: text('image'),
    createdAt: timestamp('created_at').notNull().defaultNow(),
    updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

export const session = pgTable('session', {
    id: text('id').primaryKey(),
    expiresAt: timestamp('expires_at').notNull(),
    token: text('token').notNull().unique(),
    createdAt: timestamp('created_at').notNull().defaultNow(),
    updatedAt: timestamp('updated_at').notNull().defaultNow(),
    ipAddress: text('ip_address'),
    userAgent: text('user_agent'),
    userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
});

export const account = pgTable('account', {
    id: text('id').primaryKey(),
    accountId: text('account_id').notNull(),
    providerId: text('provider_id').notNull(),
    userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
    accessToken: text('access_token'),
    refreshToken: text('refresh_token'),
    idToken: text('id_token'),
    accessTokenExpiresAt: timestamp('access_token_expires_at'),
    refreshTokenExpiresAt: timestamp('refresh_token_expires_at'),
    scope: text('scope'),
    password: text('password'),
    createdAt: timestamp('created_at').notNull().defaultNow(),
    updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

export const verification = pgTable('verification', {
    id: text('id').primaryKey(),
    identifier: text('identifier').notNull(),
    value: text('value').notNull(),
    expiresAt: timestamp('expires_at').notNull(),
    createdAt: timestamp('created_at').defaultNow(),
    updatedAt: timestamp('updated_at').defaultNow(),
});

// One row per finished game. `archive` is the existing GameArchive shape
// (@catan/core's replay.ts) stored whole — it's already a tiny, deterministic,
// self-contained replay unit (seed + ordered move list), so there is nothing
// to gain from normalizing moves into their own table.
export const games = pgTable('games', {
    id: text('id').primaryKey().$defaultFn(() => crypto.randomUUID()),
    roomId: text('room_id').notNull(),
    startedAt: timestamp('started_at').notNull(),
    finishedAt: timestamp('finished_at').notNull().defaultNow(),
    winnerSeat: integer('winner_seat'),
    turns: integer('turns').notNull(),
    archive: jsonb('archive').notNull(),
});

// One row per seat in a finished game — exists so "every game user X played"
// is a plain indexed query, rather than scanning the `archive` JSON blob.
// `userId` is nullable: guest and AI seats are recorded too (for the move
// log / opponent names), just with no account attached.
export const gamePlayers = pgTable('game_players', {
    id: text('id').primaryKey().$defaultFn(() => crypto.randomUUID()),
    gameId: text('game_id').notNull().references(() => games.id, { onDelete: 'cascade' }),
    seatIndex: integer('seat_index').notNull(),
    userId: text('user_id').references(() => user.id, { onDelete: 'set null' }),
    displayName: text('display_name').notNull(),
    color: text('color').notNull(),
});
