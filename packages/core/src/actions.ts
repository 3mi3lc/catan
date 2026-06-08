import { PlayerId, TileId, VertexId, EdgeId } from './ids';
import { Resource } from './board';

// A partial bundle of resources — used for trades and discards, where not every
// resource is involved.
export type ResourceBundle = Partial<Record<Resource, number>>;

// Every legal thing a player can attempt, as data. Discriminated on `type`, so
// the reducer can `switch` over it and the compiler will flag any case you
// forget to handle. This single union is also exactly what an AI enumerates.
export type Action =
  | { type: 'rollDice' }
  | { type: 'buildSettlement'; vertex: VertexId }
  | { type: 'buildCity'; vertex: VertexId }
  | { type: 'buildRoad'; edge: EdgeId }
  | { type: 'buyDevCard' }
  | { type: 'playKnight'; robberTo: TileId; stealFrom: PlayerId | null }
  | { type: 'playRoadBuilding'; edges: [EdgeId, EdgeId] }
  | { type: 'playYearOfPlenty'; take: [Resource, Resource] }
  | { type: 'playMonopoly'; resource: Resource }
  | { type: 'moveRobber'; tile: TileId; stealFrom: PlayerId | null }
  | { type: 'discard'; resources: ResourceBundle }
  | { type: 'bankTrade'; give: Resource; giveCount: number; receive: Resource }
  | { type: 'proposeTrade'; to: PlayerId | 'all'; give: ResourceBundle; want: ResourceBundle }
  | { type: 'respondToTrade'; tradeId: string; accept: boolean }
  | { type: 'endTurn' };

export type ActionType = Action['type'];

// An action attributed to a player. The server resolves the player from the
// socket connection; storing it on the move keeps replays self-contained.
export interface Move {
  player: PlayerId;
  action: Action;
}

// Things that happened as a result of a move. The UI animates these, and the
// stream of them doubles as a game log / replay.
export type GameEvent =
  | { type: 'diceRolled'; dice: [number, number]; total: number }
  | { type: 'resourcesProduced'; gains: Record<PlayerId, ResourceBundle> }
  | { type: 'built'; player: PlayerId; what: BuildingKindOrRoad }
  | { type: 'robberMoved'; tile: TileId; from: PlayerId | null; stolen: Resource | null }
  | { type: 'devCardBought'; player: PlayerId }
  | { type: 'tradeExecuted'; between: [PlayerId, PlayerId] }
  | { type: 'awardMoved'; award: 'longestRoad' | 'largestArmy'; to: PlayerId }
  | { type: 'gameWon'; player: PlayerId };

type BuildingKindOrRoad = 'settlement' | 'city' | 'road';
