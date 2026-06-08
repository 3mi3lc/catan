import { Board, Tile, Vertex, Edge, Terrain, AxialCoord, Port, Resource, RESOURCES } from './board';
import {
  TileId, VertexId, EdgeId,
  asTileId, asVertexId, asEdgeId,
} from './ids';

// A random source returning a float in [0, 1). Inject a seeded RNG here (instead
// of the default Math.random) when you want reproducible, replayable games.
export type Rng = () => number;

export interface BoardOptions {
  // Official rule: the high-frequency red tokens (6 and 8) may not be placed on
  // adjacent tiles. On by default. Turn off only for a "wild" unbalanced board.
  forbidAdjacentRed?: boolean;
  // Popular house rule (NOT official): no two identical numbers on adjacent
  // tiles. Off by default to match the official setup.
  forbidAdjacentSameNumber?: boolean;
  // Place the 9 harbours around the coast. On by default. Turn off for a
  // bare island (handy in tests that only care about land).
  includePorts?: boolean;
}

// The standard base-game board is a hexagon of radius 2 in axial coordinates:
// 19 tiles arranged in rows of 3-4-5-4-3.
const RADIUS = 2;
const SQRT3 = Math.sqrt(3);

// The high-frequency "red" number tokens, which are dots-heavy on a real board.
const RED = new Set<number>([6, 8]);

// The six axial directions to a tile's neighbours.
const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1],
];

// --- geometry -------------------------------------------------------------
// We place each tile in a flat 2D plane, then identify every corner by its
// rounded (x, y) position. Two neighbouring tiles compute the SAME coordinate
// for a shared corner, so they map to the same VertexId automatically. This is
// the whole trick: sharing falls out of geometry instead of hand-written
// adjacency tables that are easy to get subtly wrong.

function hexCenter(q: number, r: number): { x: number; y: number } {
  // Pointy-top axial -> pixel layout (unit size).
  return { x: SQRT3 * q + (SQRT3 / 2) * r, y: 1.5 * r };
}

function corner(cx: number, cy: number, i: number): { x: number; y: number } {
  // Corner 0 is the top point; the rest go clockwise.
  const angle = (Math.PI / 180) * (60 * i - 90);
  return { x: cx + Math.cos(angle), y: cy + Math.sin(angle) };
}

// Round hard enough to absorb floating-point noise, but far finer than the
// ~0.5-unit gap between genuinely distinct corners.
const round = (n: number): number => Math.round(n * 1000) / 1000;
const posKey = (x: number, y: number): string => `${round(x)},${round(y)}`;

// --- resource and token bags ---------------------------------------------

function terrainBag(): Terrain[] {
  return [
    ...Array<Terrain>(4).fill('fields'),    // grain
    ...Array<Terrain>(4).fill('forest'),    // lumber
    ...Array<Terrain>(4).fill('pasture'),   // wool
    ...Array<Terrain>(3).fill('hills'),     // brick
    ...Array<Terrain>(3).fill('mountains'), // ore
    'desert',
  ]; // 4+4+4+3+3+1 = 19
}

// 18 tokens for the 18 non-desert tiles (no 7; one each of 2 and 12).
function tokenBag(): number[] {
  return [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];
}

function shuffle<T>(arr: readonly T[], rng: Rng): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// --- tile adjacency -------------------------------------------------------
// Two tiles are neighbours when their axial coordinates differ by one of the
// six directions. Exported because it's broadly useful (token rules, the
// robber, board analysis) and lets tests verify placement.

export function tileAdjacency(board: Board): Map<TileId, TileId[]> {
  const byCoord = new Map<string, TileId>();
  for (const t of Object.values(board.tiles)) byCoord.set(`${t.coord.q},${t.coord.r}`, t.id);

  const adj = new Map<TileId, TileId[]>();
  for (const t of Object.values(board.tiles)) {
    const neighbours: TileId[] = [];
    for (const [dq, dr] of DIRECTIONS) {
      const id = byCoord.get(`${t.coord.q + dq},${t.coord.r + dr}`);
      if (id) neighbours.push(id);
    }
    adj.set(t.id, neighbours);
  }
  return adj;
}

// Does this number-token assignment satisfy the requested adjacency rules?
function placementValid(
    numberByTile: Map<TileId, number>,
    adjacency: Map<TileId, TileId[]>,
    forbidRed: boolean,
    forbidSame: boolean,
): boolean {
  for (const [id, n] of numberByTile) {
    for (const neighbour of adjacency.get(id) ?? []) {
      const m = numberByTile.get(neighbour);
      if (m === undefined) continue; // neighbour is the desert
      if (forbidRed && RED.has(n) && RED.has(m)) return false;
      if (forbidSame && n === m) return false;
    }
  }
  return true;
}

// --- generation -----------------------------------------------------------

export function generateBoard(rng: Rng = Math.random, options: BoardOptions = {}): Board {
  const forbidAdjacentRed = options.forbidAdjacentRed ?? true;
  const forbidAdjacentSameNumber = options.forbidAdjacentSameNumber ?? false;

  // 1. The set of axial coordinates inside a radius-2 hexagon.
  const coords: AxialCoord[] = [];
  for (let q = -RADIUS; q <= RADIUS; q++) {
    for (let r = -RADIUS; r <= RADIUS; r++) {
      if (Math.abs(q + r) <= RADIUS) coords.push({ q, r });
    }
  }

  // 2. Terrain is placed at random — the official variable setup has no terrain
  //    constraints. Number tokens come later, once the topology exists.
  const terrains = shuffle(terrainBag(), rng);

  const tiles: Record<TileId, Tile> = {};
  const vertices: Record<VertexId, Vertex> = {};
  const edges: Record<EdgeId, Edge> = {};

  const vertexByPos = new Map<string, VertexId>();
  const edgeByPair = new Map<string, EdgeId>();
  let nextV = 0, nextE = 0;

  const vertexAt = (x: number, y: number): VertexId => {
    const k = posKey(x, y);
    let id = vertexByPos.get(k);
    if (!id) {
      id = asVertexId(`v${nextV++}`);
      vertexByPos.set(k, id);
      vertices[id] = { id, tiles: [], edges: [], port: null };
    }
    return id;
  };

  const edgeBetween = (a: VertexId, b: VertexId): EdgeId => {
    const k = [a, b].sort().join('|');
    let id = edgeByPair.get(k);
    if (!id) {
      id = asEdgeId(`e${nextE++}`);
      edgeByPair.set(k, id);
      edges[id] = { id, vertices: [a, b], tiles: [] };
    }
    return id;
  };

  // 3. Build the topology. Tokens start null and are assigned in step 4.
  coords.forEach((coord, i) => {
    const tileId = asTileId(`t${i}`);
    const { x: cx, y: cy } = hexCenter(coord.q, coord.r);

    const vertexIds = Array.from({ length: 6 }, (_, c) => {
      const p = corner(cx, cy, c);
      return vertexAt(p.x, p.y);
    });
    const edgeIds = vertexIds.map((v, c) => edgeBetween(v, vertexIds[(c + 1) % 6]));

    tiles[tileId] = {
      id: tileId, coord, terrain: terrains[i], numberToken: null,
      vertices: vertexIds, edges: edgeIds,
    };

    vertexIds.forEach((v, c) => {
      pushUnique(vertices[v].tiles, tileId);
      const e = edgeIds[c];
      const next = vertexIds[(c + 1) % 6];
      pushUnique(edges[e].tiles, tileId);
      pushUnique(vertices[v].edges, e);
      pushUnique(vertices[next].edges, e);
    });
  });

  // 4. Assign number tokens to the 18 non-desert tiles, honouring the red rule.
  const board: Board = { tiles, vertices, edges };
  const adjacency = tileAdjacency(board);
  const nonDesert = Object.values(tiles).filter((t) => t.terrain !== 'desert').map((t) => t.id);

  let numberByTile = new Map<TileId, number>();
  for (let attempt = 1; ; attempt++) {
    const tokens = shuffle(tokenBag(), rng);
    numberByTile = new Map(nonDesert.map((id, i) => [id, tokens[i]]));
    if (placementValid(numberByTile, adjacency, forbidAdjacentRed, forbidAdjacentSameNumber)) break;
    // Rejection sampling: valid layouts are common, so this converges quickly.
    // The cap only guards against a logic error or an impossible constraint set.
    if (attempt >= 10000) throw new Error('Failed to place number tokens within the adjacency rules');
  }
  for (const [id, n] of numberByTile) tiles[id].numberToken = n;

  // 5. Place the 9 harbours around the coast.
  if (options.includePorts ?? true) placePorts(board, rng);

  return board;
}

function pushUnique<T>(arr: T[], item: T): void {
  if (!arr.includes(item)) arr.push(item);
}

// --- ports ----------------------------------------------------------------

// Walk the coast as an ordered cycle of edge ids. A coastal edge borders just
// one tile (edge.tiles.length === 1); each coastal vertex sits on exactly two
// of them, so the perimeter is a single loop we can follow edge to edge.
export function coastline(board: Board): EdgeId[] {
  const coastal = Object.values(board.edges).filter((e) => e.tiles.length === 1);
  const edgesAtVertex = new Map<VertexId, EdgeId[]>();
  for (const e of coastal) {
    for (const v of e.vertices) {
      const list = edgesAtVertex.get(v) ?? [];
      list.push(e.id);
      edgesAtVertex.set(v, list);
    }
  }

  const start = coastal[0];
  const ring: EdgeId[] = [start.id];
  let current = start;
  let from = start.vertices[0];

  while (true) {
    const to = current.vertices[0] === from ? current.vertices[1] : current.vertices[0];
    const next = (edgesAtVertex.get(to) ?? []).find((id) => id !== current.id)!;
    if (next === start.id) break;
    ring.push(next);
    from = to;
    current = board.edges[next];
  }
  return ring;
}

// Place 9 harbours: five 2:1 (one per resource) and four 3:1, alternating
// specific/generic and spaced evenly around the coast so none share a vertex.
function placePorts(board: Board, rng: Rng): void {
  const ring = coastline(board);
  const COUNT = 9;

  // S G S G S G S G S — resources shuffled for variety, generics between them.
  const resources = shuffle(RESOURCES, rng);
  const ports: Port[] = [];
  for (let i = 0, r = 0; i < COUNT; i++) {
    ports.push(i % 2 === 0 ? { kind: '2:1', resource: resources[r++] } : { kind: '3:1' });
  }

  const offset = Math.floor(rng() * ring.length);
  ports.forEach((port, i) => {
    const pos = (offset + Math.round((i * ring.length) / COUNT)) % ring.length;
    const edge = board.edges[ring[pos]];
    board.vertices[edge.vertices[0]].port = port;
    board.vertices[edge.vertices[1]].port = port;
  });
}

// Enumerate the placed harbours, each with the two vertices that access it.
// Ports are stored on vertices (a settlement on either vertex uses the port);
// the same Port object is shared by its two vertices, so identity groups them.
export function boardPorts(board: Board): { port: Port; vertices: VertexId[] }[] {
  const byPort = new Map<Port, VertexId[]>();
  for (const v of Object.values(board.vertices)) {
    if (!v.port) continue;
    const list = byPort.get(v.port) ?? [];
    list.push(v.id);
    byPort.set(v.port, list);
  }
  return [...byPort].map(([port, vertices]) => ({ port, vertices }));
}