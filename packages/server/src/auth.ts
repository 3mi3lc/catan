// Auth configuration (better-auth) — kept as one focused module so every
// session/cookie/rate-limit decision lives in one place rather than spread
// across index.ts.
//
// Deliberately NOT hand-rolled: session-token generation (and the fact that
// only a hash of the token is ever persisted), and rate limiting are all
// handled by better-auth itself rather than reimplemented here — that
// surface is exactly where DIY auth code tends to grow subtle holes. See
// packages/server/README.md for the full security rationale.

import { betterAuth } from 'better-auth';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { haveIBeenPwned } from 'better-auth/plugins';
import { hash, verify, Algorithm } from '@node-rs/argon2';
import { db } from './db/client';
import * as schema from './db/schema';

// better-auth's own default hasher is scrypt (via node:crypto, no native
// deps) — a perfectly OWASP-acceptable choice, but Argon2id is the current
// first recommendation, so it's swapped in explicitly here rather than left
// on the library default. Params are OWASP's first recommended Argon2id
// profile (m=19 MiB, t=2, p=1).
const ARGON2_OPTS = { algorithm: Algorithm.Argon2id, memoryCost: 19456, timeCost: 2, parallelism: 1 };

const CORS_ORIGIN = process.env.CORS_ORIGIN ?? 'http://localhost:5173';
const BETTER_AUTH_SECRET = process.env.BETTER_AUTH_SECRET;
if (!BETTER_AUTH_SECRET) throw new Error('BETTER_AUTH_SECRET is required (see .env.example)');
if (CORS_ORIGIN === '*')
    throw new Error('CORS_ORIGIN must be a concrete origin once auth is enabled — browsers reject "*" combined with credentialed requests');

export const auth = betterAuth({
    secret: BETTER_AUTH_SECRET,
    baseURL: process.env.BETTER_AUTH_URL ?? 'http://localhost:8080',
    trustedOrigins: [CORS_ORIGIN],
    database: drizzleAdapter(db, { provider: 'pg', schema }),
    emailAndPassword: {
        enabled: true,
        minPasswordLength: 10,
        // Generic success/failure messaging on top of this (no "no such
        // email" vs "wrong password" distinction) is enforced client-side by
        // never branching the UI on better-auth's specific error code.
        autoSignIn: true,
        password: {
            hash: (password) => hash(password, ARGON2_OPTS),
            verify: ({ hash: stored, password }) => verify(stored, password, ARGON2_OPTS),
        },
    },
    // Rejects signup/password-change/reset with a password found in the
    // Have I Been Pwned breach corpus — checked via k-anonymity (only a
    // 5-char SHA-1 prefix leaves this server), so the real password is never
    // sent anywhere.
    plugins: [haveIBeenPwned()],
    session: {
        // Sessions are DB-backed (the `session` table), not bare JWTs, so a
        // password change or explicit logout can actually revoke them.
        expiresIn: 60 * 60 * 24 * 30, // 30 days
        updateAge: 60 * 60 * 24, // refresh the expiry at most once a day
    },
    advanced: {
        useSecureCookies: process.env.NODE_ENV === 'production',
        cookiePrefix: 'catan',
        defaultCookieAttributes: {
            httpOnly: true,
            sameSite: 'lax',
        },
    },
    rateLimit: {
        enabled: true,
        window: 60,
        max: 20, // generous for normal use, low enough to blunt credential-stuffing
    },
});
