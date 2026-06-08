import {
  generateBoard, initialGameState, applyMove, rngStep,
  type GameState, type PlayerId, type PlayerColor,
} from '@catan/core';
import type { Policy } from './policy';

const COLORS: PlayerColor[] = ['red', 'blue', 'white', 'orange'];

const rngFromSeed = (seed: number) => {
  let s = seed >>> 0;
  return () => { const r = rngStep(s); s = r.next; return r.value; };
};

export interface MatchOptions {
  seed: number;
  policies: Policy[];      // one per seat; length sets the player count (2–4)
  maxSteps?: number;
  onStep?: (state: GameState) => void;
}

export interface MatchResult {
  winner: PlayerId | null;
  winnerSeat: number | null;
  turns: number;
  steps: number;
  finalState: GameState;
}

// Play one game to completion (or the step cap). Fully deterministic given the
// seed and deterministic policies, so any result replays exactly.
export function playMatch({ seed, policies, maxSteps = 5000, onStep }: MatchOptions): MatchResult {
  const n = policies.length;
  const board = generateBoard(rngFromSeed(seed));
  const seats = COLORS.slice(0, n).map((c, i) => ({ id: `p${i}`, name: c, color: c }));
  let state = initialGameState(board, seats, seed);
  const seatOf = (p: PlayerId) => state.turnOrder.indexOf(p);

  let steps = 0;
  let turns = 0;
  for (; steps < maxSteps && !state.winner; steps++) {
    // In the discard phase any over-the-limit player acts (not the current one).
    const actor = state.phase === 'discard'
      ? (Object.keys(state.pendingDiscards)[0] as PlayerId)
      : state.currentPlayer;
    const action = policies[seatOf(actor)].decide(state, actor);
    if (action.type === 'endTurn') turns++;
    const res = applyMove(state, { player: actor, action });
    if (!res.ok) {
      throw new Error(`${policies[seatOf(actor)].name} proposed an illegal move (${res.error}): ${JSON.stringify(action)}`);
    }
    state = res.state;
    onStep?.(state);
  }

  return {
    winner: state.winner,
    winnerSeat: state.winner ? seatOf(state.winner) : null,
    turns,
    steps,
    finalState: state,
  };
}

export interface TournamentResult {
  games: number;
  unfinished: number;
  winsByName: Record<string, number>;
  winsBySeat: number[];
}

// Play `games` matches, rotating seat assignments each game so every policy
// occupies every seat equally — this cancels out first-player advantage, making
// `winsByName` a fair head-to-head comparison.
export function playTournament(
  policies: Policy[], games: number, baseSeed = 1, maxSteps = 5000,
): TournamentResult {
  const n = policies.length;
  const winsByName: Record<string, number> = {};
  const winsBySeat = new Array<number>(n).fill(0);
  let unfinished = 0;
  for (const p of policies) winsByName[p.name] ??= 0;

  for (let g = 0; g < games; g++) {
    const rotated = Array.from({ length: n }, (_, i) => policies[(i + g) % n]);
    const r = playMatch({ seed: baseSeed + g, policies: rotated, maxSteps });
    if (r.winner === null) { unfinished++; continue; }
    winsBySeat[r.winnerSeat!]++;
    const winnerName = rotated[r.winnerSeat!].name;
    winsByName[winnerName] = (winsByName[winnerName] ?? 0) + 1;
  }
  return { games, unfinished, winsByName, winsBySeat };
}
