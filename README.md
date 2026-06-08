# 🔶 Catan

A TypeScript monorepo implementation of Catan, featuring a game engine, AI players, and a web interface.

---

## Packages

| Package | Description |
|--------|-------------|
| `core` | Game engine, rules, board generation, and state management |
| `ai`   | AI policies (greedy, random) and simulation runner |
| `web`  | Browser-based UI built with React and Vite |

---

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/)
- [pnpm](https://pnpm.io/)

### Install dependencies
```bash
pnpm install
```

### Run the web app
```bash
pnpm --filter web dev
```

### Run tests
```bash
pnpm --filter core test
pnpm --filter ai test
```

---

## Project Structure

```
packages/
├── core/   # Game logic and state
├── ai/     # AI players and runner
└── web/    # Frontend
```