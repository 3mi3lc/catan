import {
  rngStep, RESOURCES,
  type Board, type Terrain, type Resource, type PlayerColor,
  type VertexId, type GameState, type PlayerId,
} from '@catan/core';

// --- geometry (rendering only; topology comes from core) -------------------
export const SQRT3 = Math.sqrt(3);
export const SIZE = 56;
export const hexCenter = (q: number, r: number) => ({ x: SQRT3 * q + (SQRT3 / 2) * r, y: 1.5 * r });
export const corner = (cx: number, cy: number, i: number) => {
  const a = (Math.PI / 180) * (60 * i - 90);
  return { x: cx + Math.cos(a), y: cy + Math.sin(a) };
};
export const pips = (n: number) => 6 - Math.abs(7 - n);

// One integer seed reproduces the whole game (board + dice + steals).
export const rngFromSeed = (seed: number) => {
  let s = seed >>> 0;
  return () => { const { value, next } = rngStep(s); s = next; return value; };
};

// --- palette ---------------------------------------------------------------
export const TERRAIN: Record<Terrain, { fill: string; stroke: string; label: string }> = {
  fields:    { fill: '#e3b23c', stroke: '#a87b18', label: 'Fields · grain' },
  forest:    { fill: '#2f6b3c', stroke: '#1d4527', label: 'Forest · lumber' },
  pasture:   { fill: '#9dbe5a', stroke: '#6c8a34', label: 'Pasture · wool' },
  hills:     { fill: '#c5673a', stroke: '#8c4422', label: 'Hills · brick' },
  mountains: { fill: '#9298a0', stroke: '#666c74', label: 'Mountains · ore' },
  desert:    { fill: '#ddcca4', stroke: '#b09c6e', label: 'Desert' },
};

export const RES_COLOR: Record<Resource, string> = {
  brick: '#c5673a', lumber: '#2f6b3c', wool: '#9dbe5a', grain: '#e3b23c', ore: '#9298a0',
};
export const RES_ABBR: Record<Resource, string> = {
  brick: 'Brick', lumber: 'Lumber', wool: 'Wool', grain: 'Grain', ore: 'Ore',
};

export const PLAYER: Record<PlayerColor, { fill: string; ink: string; name: string }> = {
  red:    { fill: '#c0392b', ink: '#fff', name: 'Red' },
  blue:   { fill: '#2f6aa8', ink: '#fff', name: 'Blue' },
  white:  { fill: '#e9e5d8', ink: '#2c2620', name: 'White' },
  orange: { fill: '#d98324', ink: '#fff', name: 'Orange' },
};

export const SEAT_DEFS: { id: string; color: PlayerColor }[] = [
  { id: 'p0', color: 'red' }, { id: 'p1', color: 'blue' },
  { id: 'p2', color: 'white' }, { id: 'p3', color: 'orange' },
];

// --- board-derived helpers -------------------------------------------------
export interface Layout {
  pos: Map<VertexId, { x: number; y: number }>;
  ox: number; oy: number; W: number; H: number;
}

export function layoutBoard(board: Board): Layout {
  const unit = new Map<VertexId, { x: number; y: number }>();
  for (const t of Object.values(board.tiles)) {
    const c = hexCenter(t.coord.q, t.coord.r);
    t.vertices.forEach((v, i) => { if (!unit.has(v)) unit.set(v, corner(c.x, c.y, i)); });
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of unit.values()) {
    minX = Math.min(minX, p.x * SIZE); minY = Math.min(minY, p.y * SIZE);
    maxX = Math.max(maxX, p.x * SIZE); maxY = Math.max(maxY, p.y * SIZE);
  }
  const pad = 60, ox = pad - minX, oy = pad - minY;
  const pos = new Map<VertexId, { x: number; y: number }>();
  for (const [v, p] of unit) pos.set(v, { x: p.x * SIZE + ox, y: p.y * SIZE + oy });
  return { pos, ox, oy, W: maxX - minX + pad * 2, H: maxY - minY + pad * 2 };
}

export const nameOf = (g: GameState, p: PlayerId) => PLAYER[g.players[p].color].name;
export const colorOf = (g: GameState, p: PlayerId) => PLAYER[g.players[p].color].fill;
export const handSize = (g: GameState, p: PlayerId) =>
  RESOURCES.reduce((s, r) => s + g.players[p].resources[r], 0);
