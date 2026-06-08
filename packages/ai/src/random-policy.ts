import { legalActions, rngStep, type Action } from '@catan/core';
import type { Policy } from './policy';
import { discardAction } from './heuristics';

// Uniform-random legal play. The baseline every smarter policy must beat. Seeded
// for reproducibility; create a fresh instance per game for deterministic runs.
export function randomPolicy(seed = 1): Policy {
  let s = seed >>> 0;
  const rand = (): number => { const r = rngStep(s); s = r.next; return r.value; };
  return {
    name: 'random',
    decide(state, player): Action {
      if (state.phase === 'discard') return discardAction(state, player);
      const acts = legalActions(state, player);
      if (acts.length === 0) return { type: 'endTurn' }; // safety; shouldn't happen
      return acts[Math.floor(rand() * acts.length)];
    },
  };
}
