import { describe, it, expect } from 'vitest';
import { generateBoard, tileAdjacency, boardPorts, coastline, type Rng } from './board-generator';

// A tiny seeded RNG so we can test that generation is reproducible.
function mulberry32(seed: number): Rng {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('generateBoard', () => {
  const board = generateBoard(mulberry32(42));
  const tiles = Object.values(board.tiles);
  const vertices = Object.values(board.vertices);
  const edges = Object.values(board.edges);

  it('has the canonical Catan topology counts', () => {
    // These three numbers are the headline test: if any corner or side were
    // duplicated instead of shared, the counts would come out too high.
    expect(tiles).toHaveLength(19);
    expect(vertices).toHaveLength(54);
    expect(edges).toHaveLength(72);
  });

  it('gives every tile exactly 6 vertices and 6 edges', () => {
    for (const t of tiles) {
      expect(t.vertices).toHaveLength(6);
      expect(t.edges).toHaveLength(6);
    }
  });

  it('keeps vertex and edge tile-counts within legal bounds', () => {
    // A corner is touched by 1-3 tiles; a side borders 1-2 tiles.
    for (const v of vertices) {
      expect(v.tiles.length).toBeGreaterThanOrEqual(1);
      expect(v.tiles.length).toBeLessThanOrEqual(3);
    }
    for (const e of edges) {
      expect(e.tiles.length).toBeGreaterThanOrEqual(1);
      expect(e.tiles.length).toBeLessThanOrEqual(2);
    }
  });

  it('shares all 6 corners of the center tile with neighbours', () => {
    const center = tiles.find((t) => t.coord.q === 0 && t.coord.r === 0)!;
    const interior = center.vertices.filter((v) => board.vertices[v].tiles.length === 3);
    expect(interior).toHaveLength(6);
  });

  it('has consistent cross-references (no dangling ids)', () => {
    // Every id a tile points at must exist, and references must be mutual.
    for (const t of tiles) {
      for (const v of t.vertices) expect(board.vertices[v].tiles).toContain(t.id);
      for (const e of t.edges) expect(board.edges[e].tiles).toContain(t.id);
    }
    for (const e of edges) {
      for (const v of e.vertices) expect(board.vertices[v].edges).toContain(e.id);
    }
  });

  it('places exactly one desert with no number token', () => {
    const deserts = tiles.filter((t) => t.terrain === 'desert');
    expect(deserts).toHaveLength(1);
    expect(deserts[0].numberToken).toBeNull();
  });

  it('puts a number token on all 18 non-desert tiles', () => {
    const tokened = tiles.filter((t) => t.numberToken !== null);
    expect(tokened).toHaveLength(18);
  });

  it('never places two red tokens (6 or 8) on adjacent tiles', () => {
    // Stress the official rule across many independent boards.
    const RED = new Set([6, 8]);
    for (let seed = 0; seed < 200; seed++) {
      const b = generateBoard(mulberry32(seed));
      const adj = tileAdjacency(b);
      for (const t of Object.values(b.tiles)) {
        if (t.numberToken === null || !RED.has(t.numberToken)) continue;
        for (const n of adj.get(t.id)!) {
          const m = b.tiles[n].numberToken;
          expect(m === null || !RED.has(m)).toBe(true);
        }
      }
    }
  });

  it('can be told to allow adjacent reds (unbalanced board)', () => {
    // With the rule off, generation still succeeds and produces valid counts.
    const b = generateBoard(mulberry32(7), { forbidAdjacentRed: false });
    expect(Object.values(b.tiles).filter((t) => t.numberToken !== null)).toHaveLength(18);
  });

  it('can also forbid identical adjacent numbers when asked (house rule)', () => {
    for (let seed = 0; seed < 200; seed++) {
      const b = generateBoard(mulberry32(seed), { forbidAdjacentSameNumber: true });
      const adj = tileAdjacency(b);
      for (const t of Object.values(b.tiles)) {
        if (t.numberToken === null) continue;
        for (const n of adj.get(t.id)!) {
          const m = b.tiles[n].numberToken;
          expect(m === null || m !== t.numberToken).toBe(true);
        }
      }
    }
  });

  it('walks a 30-edge coastline loop', () => {
    const ring = coastline(board);
    expect(ring).toHaveLength(30);
    expect(new Set(ring).size).toBe(30);
    for (const id of ring) expect(board.edges[id].tiles).toHaveLength(1);
  });

  it('places nine harbours with the official type distribution', () => {
    const ports = boardPorts(board);
    expect(ports).toHaveLength(9);
    expect(ports.filter((p) => p.port.kind === '3:1')).toHaveLength(4);
    const specific = ports.filter((p) => p.port.kind === '2:1');
    expect(specific).toHaveLength(5);
    const resources = specific
        .map((p) => (p.port.kind === '2:1' ? p.port.resource : ''))
        .sort();
    expect(resources).toEqual(['brick', 'grain', 'lumber', 'ore', 'wool']);
  });

  it('puts each harbour on a coastal edge, with no two harbours sharing a vertex', () => {
    const coastalPairs = new Set(
        Object.values(board.edges)
            .filter((e) => e.tiles.length === 1)
            .map((e) => [...e.vertices].sort().join('|')),
    );
    const seen = new Set<string>();
    for (const { vertices } of boardPorts(board)) {
      expect(vertices).toHaveLength(2);
      for (const v of vertices) {
        expect(seen.has(v)).toBe(false); // no sharing
        seen.add(v);
      }
      expect(coastalPairs.has([...vertices].sort().join('|'))).toBe(true);
    }
  });

  it('can omit ports', () => {
    const bare = generateBoard(mulberry32(1), { includePorts: false });
    expect(boardPorts(bare)).toHaveLength(0);
  });

  it('is reproducible from the same seed', () => {
    const a = generateBoard(mulberry32(123));
    const b = generateBoard(mulberry32(123));
    const layout = (brd: ReturnType<typeof generateBoard>) =>
        Object.values(brd.tiles).map((t) => `${t.terrain}:${t.numberToken}`);
    expect(layout(a)).toEqual(layout(b));
  });
});