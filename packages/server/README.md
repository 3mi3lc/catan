# @catan/server

Authoritative Socket.io game server for `online.html`. Holds the only real
`GameState` per room, validates every client move through `@catan/core`'s
`applyMove`, and drives AI seats itself via `@catan/ai`'s `ServerBot`
(`onnxruntime-node`) — an AI-only room plays itself with zero browsers
connected.

## Running locally

```sh
pnpm --filter @catan/server dev    # tsx watch, restarts on src/ changes
pnpm --filter @catan/server start  # same, without the watcher
```

Both automatically run `docker compose up -d postgres` first (`predev`/`prestart` in `package.json`), so local Postgres doesn't need to be started by hand before every session — it's part of the startup sequence, not a separate step you have to remember. **Docker Desktop itself still has to be running** — if it isn't, the `predev`/`prestart` hook fails fast with a clear "can't connect to the Docker daemon" error and the server never starts, rather than booting and only failing later on the first request that touches the database.

Both run TypeScript directly via `tsx` — there is no compiled `dist/`.
`@catan/core` and `@catan/ai` are consumed as workspace packages whose `main`
points straight at `.ts` source (see their `package.json`), so a plain
`tsc && node dist/index.js` build does not work: Node has no loader for the
`.ts` files those imports resolve to. Don't reintroduce a `build` script that
shells out to `tsc` for emit unless that source-as-main setup changes too.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | HTTP/Socket.io listen port. |
| `CORS_ORIGIN` | `*` | Socket.io + REST CORS origin. **Must be a concrete origin (not `*`) once auth is in use** — `auth.ts` refuses to start otherwise, since browsers reject a wildcard `Access-Control-Allow-Origin` on credentialed requests. Set to your deployed web app's origin in production. |
| `MODEL_DIR` | `../../web/public/models` (relative to `src/`) | Where `.onnx` model files for AI seats live. See below. |
| `DATABASE_URL` | — | Postgres connection string. Required — accounts, sessions, and finished-game history all live here. |
| `BETTER_AUTH_SECRET` | — | Required. Signs/encrypts session material — generate with `node -e "console.log(crypto.randomBytes(32).toString('hex'))"` and never commit it. |
| `BETTER_AUTH_URL` | `http://localhost:8080` | This server's own public URL (used for cookie/callback construction). |
| `WEB_DIST` | `../../web/dist` (relative to `src/`) | Where the built web app lives, for `static.ts`'s single-origin serving. See "Playing with friends locally" below. |

Copy `.env.example` to `.env` and fill these in for local dev. Both `dev` and `start` pass Node's native `--env-file-if-exists=.env` flag (Node 20.6+) to `tsx` — it loads `.env` when present and silently continues without it otherwise, so a production deploy that injects env vars directly (e.g. `docker run -e ...`, no `.env` file in the image) still works unchanged. `tsx`/Node do **not** load `.env` on their own without that flag.

## Accounts, sessions, and game history

Auth is [better-auth](https://better-auth.com) (email + password, `/api/auth/*`, mounted directly on the raw `http.createServer` in `index.ts` — see `applyCors`/`authHandler` there) with Postgres via Drizzle (`src/db/`). Deliberately not hand-rolled: session-token generation and rate limiting are better-auth's responsibility, not this codebase's. Specific hardening in `src/auth.ts`:

- **Password hashing**: Argon2id (`@node-rs/argon2`, OWASP's first-choice params: m=19 MiB, t=2, p=1) — explicitly swapped in over better-auth's own default (`scrypt` via `node:crypto`, also OWASP-acceptable, but Argon2id is the stronger current recommendation).
- **Breach checking**: the `haveIBeenPwned` plugin rejects any password found in the HIBP corpus at signup/reset, checked via k-anonymity (only a 5-character hash prefix ever leaves this server).
- **Sessions**: DB-backed and revocable (not bare JWTs), `httpOnly`/`SameSite=Lax` cookies.
- **No user enumeration**: sign-in returns an identical 401 for "no such email" and "wrong password" — verified manually; don't add a branch that distinguishes them in any client UI built on top of this.
- Rate-limited auth endpoints.

Socket.io connections resolve their better-auth session on handshake (same cookie the browser already sends) so a seat claimed while logged in can be linked to a `user_id` — see `io.use(...)` in `index.ts`. Guest play (no account) keeps working exactly as before; `userId` is just `null` for those connections.

When a room finishes (`broadcastGameOver` in `index.ts`), the same `GameArchive` already broadcast to live players is also persisted via `saveGame` (`src/db/games.ts`) — one `games` row plus one `game_players` row per seat (`userId` null for guest/AI seats). This is fire-and-forget: a DB outage logs an error but never blocks or fails the live game for the players at the table.

Two REST endpoints read it back, both requiring a session and both **participant-only** (a finished game is private to the people who played it — fetching someone else's game by id returns `404`, not `403`, so a non-participant can't even confirm the id is valid):

- `GET /games/mine` — every finished game the signed-in account played, most recent first.
- `GET /games/<id>` — that game's full `GameArchive` (feed straight into `@catan/core`'s `loadReplay`) plus its seat list.

`scripts/_verify_persistence.ts` exercises `saveGame`/`listGamesForUser`/`getGameForUser` directly against a real Postgres instance (including the privacy gate) without needing to play a full game to a legitimate win — run with `DATABASE_URL=... npx tsx scripts/_verify_persistence.ts` from this directory.

### Local Postgres

```sh
pnpm --filter @catan/server db:generate   # after changing src/db/schema.ts — no DB connection needed
pnpm --filter @catan/server db:migrate    # apply pending migrations
```

`dev`/`start`/`db:migrate` all auto-start the `postgres` container from the root `docker-compose.yml` first (see "Running locally" above) — Docker Desktop just needs to already be running. `db:generate` doesn't touch the database at all (it only reads `src/db/schema.ts`), so it has no such hook.

`docker-compose.yml` is local-dev-only; point `DATABASE_URL` at a managed Postgres instance (Fly/Railway/Render all offer one) for any real deployment.

## Model files

AI seats (`packages/server/src/bots.ts`) load `catan_net_2p.onnx` /
`catan_net_4p.onnx` from `MODEL_DIR`, falling back to the 4p net if no 2p net
is found. These are the same files `hotseat.ts` already serves to the
browser from `packages/web/public/models` — **that directory is listed in
the repo's `.gitignore` (`models/`)**, so a fresh `git clone` will not have
them. Either:

- make sure they exist on disk at `packages/web/public/models` before
  `docker build` (the root `Dockerfile` copies them in if present), or
- set `MODEL_DIR` to wherever you've put them on the deploy target (e.g. a
  mounted volume), independent of the web app's directory.

## Playing with friends locally (no cloud hosting)

For "run this on my own machine and share a link when I want to play" — no Railway/Fly/Vercel account needed. The server can serve the built web app itself (`src/static.ts`), so there's exactly one process, one port, and one URL to share — no CORS/cross-origin cookie juggling between a separate web host and this server.

```sh
pnpm --filter @catan/web build          # rebuild after any web code change
pnpm --filter @catan/server dev         # (or start) — now serves the web app too, at /
```

Visiting `http://localhost:8080/` serves `online.html` by default (the multiplayer client — what a friend clicking a shared link actually wants); `/game.html`, `/replay.html`, `/index.html` are still reachable directly.

To let friends outside your network in, expose the port with a tunnel — no router config, no account:

```sh
cloudflared tunnel --url http://localhost:8080
```

This prints a random `https://<something>.trycloudflare.com` URL, live only while the tunnel runs. Two things to set to match it before starting the server:

- `CORS_ORIGIN` — Socket.io validates the browser's `Origin` header against this on every connection, tunnel or not (same-origin serving doesn't exempt it)
- `BETTER_AUTH_URL` — should match the same public URL

Both change every time you restart the tunnel (free/quick tunnels don't give you a stable URL) — update `.env` and restart the server each session. `VITE_SERVER_URL` does **not** need setting for this workflow: the web app defaults to `window.location.origin` in a production build (see `packages/web/src/server-url.ts`), so it automatically targets whatever origin it was actually loaded from — including a fresh tunnel URL — with no rebuild required between sessions, only when the web code itself changes.

## Docker

Build from the **repo root**, not this directory — the image needs the
whole pnpm workspace:

```sh
docker build -t catan-server .
docker run -p 8080:8080 -e CORS_ORIGIN=https://your-web-app catan-server
```

The image is Debian-based (`node:24-bookworm-slim`), not Alpine: `onnxruntime-node`
only ships prebuilt native bindings for glibc, not musl.

## Deploying

This is a long-lived process holding in-memory WebSocket connections, not a
request/response handler — it fits hosts built for always-on containers
(Fly.io, Railway, Render, a plain VM) rather than serverless function
platforms (e.g. Vercel/Netlify functions), which kill idle processes and
don't keep a persistent socket open. Whatever host you pick, point it at the
`Dockerfile` and set `CORS_ORIGIN` to your deployed web app's URL.

**Known v1 limitation**: room/game *state* is kept in memory only
(`src/room.ts`'s `Map<roomId, Room>`) — a server restart or redeploy still
drops every **in-progress** game. *Finished* games no longer have this
problem: each one is persisted to Postgres the moment it ends (see
"Accounts, sessions, and game history" above), so game history and replays
survive restarts even though live rooms don't. Solving in-progress
durability too (so a redeploy mid-game doesn't drop active players) would
mean moving room state itself into Postgres/Redis, not just the archive —
a deliberately deferred, separate problem.

No actual cloud deployment has been done from this repo yet — the above is
what's needed to do it, not a confirmation it's live anywhere.
