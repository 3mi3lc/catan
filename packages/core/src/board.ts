import { TileId, VertexId, EdgeId } from './ids';

// The five collectible resources.
export const RESOURCES = ['brick', 'lumber', 'wool', 'grain', 'ore'] as const;
export type Resource = (typeof RESOURCES)[number];

// Terrain types. Each produces one resource, except desert, which produces none.
export type Terrain =
  | 'hills'      // brick
  | 'forest'     // lumber
  | 'pasture'    // wool
  | 'fields'     // grain
  | 'mountains'  // ore
  | 'desert';    // nothing

export const TERRAIN_RESOURCE: Record<Terrain, Resource | null> = {
  hills: 'brick',
  forest: 'lumber',
  pasture: 'wool',
  fields: 'grain',
  mountains: 'ore',
  desert: null,
};

// Harbours: a 3:1 generic port, or a 2:1 port for one specific resource.
export type Port = { kind: '3:1' } | { kind: '2:1'; resource: Resource };

// A placed harbour and the two vertices that access it. Recorded explicitly
// (rather than inferred by grouping vertices that share a Port object) so the
// pairing survives a JSON round-trip — e.g. a Board sent to a client over the
// network deserializes each vertex's `port` into its own distinct object,
// even though on the server the two vertices originally pointed at the same one.
export interface PortGroup {
  port: Port;
  vertices: [VertexId, VertexId];
}

// Axial hex coordinates. Neighbour math is clean in this system:
// a tile's six neighbours are (q±1, r), (q, r±1), (q+1, r-1), (q-1, r+1).
export interface AxialCoord {
  q: number;
  r: number;
}

// A hex tile. `vertices` and `edges` reference SHARED objects — adjacent tiles
// point at the same VertexId/EdgeId, which is what makes the placement rules
// trivial later.
export interface Tile {
  id: TileId;
  coord: AxialCoord;
  terrain: Terrain;
  numberToken: number | null; // 2..12 excluding 7; null only for the desert
  vertices: VertexId[];       // exactly 6, ordered clockwise from the top corner
  edges: EdgeId[];            // exactly 6, ordered clockwise from the top edge
}

// A corner where settlements and cities are built. Up to three tiles meet here.
export interface Vertex {
  id: VertexId;
  tiles: TileId[]; // 1..3 tiles touch this corner — used for resource payout
  edges: EdgeId[]; // 2..3 edges connect here — used for road/longest-road logic
  port: Port | null;
}

// A side where roads are built. Borders one or two tiles.
export interface Edge {
  id: EdgeId;
  vertices: [VertexId, VertexId];
  tiles: TileId[]; // 1..2
}

// The fixed topology of the board. Generated once at game start, then never
// mutated — all the things that change during play live in GameState.
export interface Board {
  tiles: Record<TileId, Tile>;
  vertices: Record<VertexId, Vertex>;
  edges: Record<EdgeId, Edge>;
  ports: PortGroup[];
}
