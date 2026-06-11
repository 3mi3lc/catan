/// <reference types="node" />
/**
 * dump_topology.ts — exports the standard board's graph structure for the
 * Python graph network (packages/training/model_gnn.py).
 *
 * The radius-2 hex topology is identical for every board (same 54 vertices,
 * 72 edges, 19 tiles, same adjacency); only terrain/tokens/ports vary, and
 * those live in the observation. Indices match encoding.ts's BoardIndex
 * (sorted lexicographic IDs), so token order here equals the per-vertex /
 * per-tile / per-edge blocks of the observation vector.
 *
 * Usage:  pnpm --filter @catan/ai exec tsx scripts/dump_topology.ts
 * Writes: packages/training/topology.json
 */

import { writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
import { generateBoard, rngStep } from '@catan/core';
import { buildBoardIndex } from '../src/encoding';

function mkRng(seed: number) {
    let s = (seed >>> 0) || 1;
    return () => { const r = rngStep(s); s = r.next; return r.value; };
}

const board = generateBoard(mkRng(1));
const bi = buildBoardIndex(board);

// edge index → [vertex index, vertex index]
const edgeVerts = bi.edges.map((eid) =>
    board.edges[eid].vertices.map((v) => bi.vIdx.get(v)!));

// tile index → [6 vertex indices]
const tileVerts = bi.tiles.map((tid) =>
    board.tiles[tid].vertices.map((v) => bi.vIdx.get(v)!));

const out = {
    nVertices: bi.vertices.length,   // 54
    nEdges: bi.edges.length,         // 72
    nTiles: bi.tiles.length,         // 19
    edgeVerts,
    tileVerts,
};

const path = join(__dirname, '..', '..', 'training', 'topology.json');
writeFileSync(path, JSON.stringify(out));
console.log(`Wrote ${path}: ${out.nVertices} vertices, ${out.nEdges} edges, ${out.nTiles} tiles`);
