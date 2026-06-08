// Branded ID types. These are just strings at runtime, but the brand stops you
// from ever passing a VertexId where a TileId is expected — a very common bug
// when everything is "just a string".

export type TileId = string & { readonly __brand: 'TileId' };
export type VertexId = string & { readonly __brand: 'VertexId' };
export type EdgeId = string & { readonly __brand: 'EdgeId' };
export type PlayerId = string & { readonly __brand: 'PlayerId' };

export const asTileId = (s: string): TileId => s as TileId;
export const asVertexId = (s: string): VertexId => s as VertexId;
export const asEdgeId = (s: string): EdgeId => s as EdgeId;
export const asPlayerId = (s: string): PlayerId => s as PlayerId;
