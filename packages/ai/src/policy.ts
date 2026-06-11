import type { GameState, Action, PlayerId } from '@catan/core';

// A Policy decides a single Action for `player` in the given state. This is the
// universal seat-driver: the same interface backs the hotseat opponent, the
// self-play inner loop, and (later) a networked bot behind the server. It must
// handle every phase the player can be asked to act in — including `discard`,
// which `legalActions` does not enumerate (the policy synthesises a discard).
export interface Policy {
  readonly name: string;
  decide(state: GameState, player: PlayerId): Action ;
}

// A policy whose decision may be asynchronous (e.g. ONNX inference, a remote
// bot). onnxruntime-node only exposes the async `run()`, so net-backed policies
// cannot satisfy the sync Policy interface. Every sync Policy is assignable to
// AsyncPolicy, so mixed seats work with `playMatchAsync`.
export interface AsyncPolicy {
  readonly name: string;
  decide(state: GameState, player: PlayerId): Action | Promise<Action>;
}

// Narrow a legal-action list to a single action variant, with types intact.
export function ofType<T extends Action['type']>(
  actions: Action[],
  type: T,
): Extract<Action, { type: T }>[] {
  return actions.filter((a): a is Extract<Action, { type: T }> => a.type === type);
}
