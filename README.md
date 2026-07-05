# 🔶 Catan

A TypeScript monorepo implementation of Catan — a game engine, AI players
(heuristic + neural-net/MCTS), a Socket.io multiplayer server with accounts
and game history, and a React web client (desktop, and a Capacitor Android
wrapper for mobile).

---

## Packages

| Package    | Description |
|------------|-------------|
| `core`     | Game engine, rules, board generation, protocol, stats, and replay |
| `ai`       | AI policies (greedy, MCTS, neural-net bots, trade heuristics) and simulation runner |
| `server`   | Socket.io multiplayer game server — accounts/sessions (better-auth), Postgres persistence of finished games, AI seats. See [packages/server/README.md](packages/server/README.md) |
| `web`      | React/Vite frontend: board viewer, local hotseat, online multiplayer client, replay viewer, and a Capacitor Android wrapper. See [packages/web/README.md](packages/web/README.md) |
| `training` | Python training/eval pipeline for the neural-net AI (behavioral cloning, PPO, expert iteration, tournaments) |

---

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) + [pnpm](https://pnpm.io/)
- [Docker](https://www.docker.com/) (for local Postgres — see below)
- Python 3 (only needed for `packages/training`)

### Install dependencies
```bash
pnpm install
```

### Run the web app only (board viewer, no server)
```bash
pnpm --filter web dev
```
Serves `index.html` (viewer), `hotseat.html`/`game.html` (local hotseat vs.
AI — no server needed), and `replay.html` at `http://localhost:5173`.

### Run the full multiplayer stack (server + web)
```bash
pnpm --filter @catan/server dev   # auto-starts local Postgres via Docker Compose
pnpm --filter web dev             # separate terminal
```
Then open `online.html` for the multiplayer client. See
[packages/server/README.md](packages/server/README.md) for environment
variables, accounts/auth, and "playing with friends" over a tunnel without
any cloud hosting.

### Run tests
```bash
pnpm --filter core test
pnpm --filter ai test
pnpm --filter @catan/server test
```

### Typecheck everything
```bash
pnpm typecheck
```

---

## Mobile (Android)

`packages/web` ships a Capacitor wrapper (`android/`) with a live-reload dev
workflow for testing on a real device. See
[packages/web/README.md](packages/web/README.md#mobile-capacitorandroid).

---

## Project Structure

```
packages/
├── core/      # Game engine, rules, protocol, stats, replay
├── ai/        # AI players, MCTS, neural-net bots, runner
├── server/    # Multiplayer Socket.io server, auth, Postgres persistence
├── web/       # React frontend (viewer/hotseat/online/replay) + Android shell
└── training/  # Python training/eval pipeline for the neural-net AI
```
