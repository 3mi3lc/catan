// Board-viewer debug page (index.html) — React port of the vanilla main.ts.
// Renders boards from the real generateBoard() in @catan/core; the SVG is
// still composed as a markup string (same rendering code as before) and
// injected, since it's a static picture with no interactivity.

import { useMemo, useState } from 'react';
import { generateBoard, boardPorts, type Rng, type Terrain, type Board } from '@catan/core';
import './viewer.css';

// --- layout geometry (rendering only; the real topology comes from core) ---
const SQRT3 = Math.sqrt(3);
const SIZE = 54;
const hexCenter = (q: number, r: number) => ({ x: SQRT3 * q + (SQRT3 / 2) * r, y: 1.5 * r });
const corner = (cx: number, cy: number, i: number) => {
    const a = (Math.PI / 180) * (60 * i - 90);
    return { x: cx + Math.cos(a), y: cy + Math.sin(a) };
};
const pips = (n: number) => 6 - Math.abs(7 - n);

// Seeded RNG so a given seed always reproduces the same board.
function mulberry32(seed: number): Rng {
    let a = seed >>> 0;
    return () => {
        a |= 0; a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const TERRAIN: Record<Terrain, { fill: string; stroke: string; label: string }> = {
    fields:    { fill: '#e3b23c', stroke: '#a87b18', label: 'Fields → grain' },
    forest:    { fill: '#2f6b3c', stroke: '#1d4527', label: 'Forest → lumber' },
    pasture:   { fill: '#9dbe5a', stroke: '#6c8a34', label: 'Pasture → wool' },
    hills:     { fill: '#c5673a', stroke: '#8c4422', label: 'Hills → brick' },
    mountains: { fill: '#9298a0', stroke: '#666c74', label: 'Mountains → ore' },
    desert:    { fill: '#dccca4', stroke: '#b09c6e', label: 'Desert → robber' },
};

function renderSvg(board: Board, showGrid: boolean): { svg: string; counts: [number, number, number] } {
    const tiles = Object.values(board.tiles);

    // Recover each vertex's pixel position from the order tiles list their corners
    // (tile.vertices[i] is corner i) — no need to store positions in core.
    const vpos = new Map<string, { x: number; y: number }>();
    for (const t of tiles) {
        const c = hexCenter(t.coord.q, t.coord.r);
        t.vertices.forEach((v, i) => { if (!vpos.has(v)) vpos.set(v, corner(c.x, c.y, i)); });
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of vpos.values()) {
        minX = Math.min(minX, p.x * SIZE); minY = Math.min(minY, p.y * SIZE);
        maxX = Math.max(maxX, p.x * SIZE); maxY = Math.max(maxY, p.y * SIZE);
    }
    const pad = 58, ox = pad - minX, oy = pad - minY;
    const W = maxX - minX + pad * 2, H = maxY - minY + pad * 2;

    const out: string[] = [`<svg viewBox="0 0 ${W.toFixed(1)} ${H.toFixed(1)}" xmlns="http://www.w3.org/2000/svg">`];

    for (const t of tiles) {
        const T = TERRAIN[t.terrain];
        const c = hexCenter(t.coord.q, t.coord.r);
        const cx = c.x * SIZE + ox, cy = c.y * SIZE + oy;
        const pts = Array.from({ length: 6 }, (_, i) => {
            const p = corner(c.x, c.y, i);
            return `${(p.x * SIZE + ox).toFixed(1)},${(p.y * SIZE + oy).toFixed(1)}`;
        }).join(' ');

        out.push(`<polygon points="${pts}" fill="${T.fill}" stroke="${T.stroke}" stroke-width="2" stroke-linejoin="round"/>`);

        if (t.numberToken !== null) {
            const red = t.numberToken === 6 || t.numberToken === 8;
            out.push(`<circle cx="${cx}" cy="${cy}" r="${SIZE * 0.34}" fill="#f3e9cf" stroke="#bda673" stroke-width="1.5"/>`);
            out.push(`<text x="${cx}" y="${cy - 2}" text-anchor="middle" dominant-baseline="central" font-family="Fraunces,serif" font-weight="600" font-size="${red ? 22 : 19}" fill="${red ? '#b23a2e' : '#34302a'}">${t.numberToken}</text>`);
            const n = pips(t.numberToken), gap = 5, total = (n - 1) * gap;
            for (let i = 0; i < n; i++) {
                out.push(`<circle cx="${cx - total / 2 + i * gap}" cy="${cy + SIZE * 0.2}" r="1.7" fill="${red ? '#b23a2e' : '#6b6358'}"/>`);
            }
        } else {
            out.push(`<circle cx="${cx}" cy="${cy}" r="${SIZE * 0.26}" fill="#3a342c" stroke="#221e18" stroke-width="2"/>`);
        }
    }

    if (showGrid) {
        for (const e of Object.values(board.edges)) {
            const a = vpos.get(e.vertices[0])!, b = vpos.get(e.vertices[1])!;
            out.push(`<line x1="${(a.x * SIZE + ox).toFixed(1)}" y1="${(a.y * SIZE + oy).toFixed(1)}" x2="${(b.x * SIZE + ox).toFixed(1)}" y2="${(b.y * SIZE + oy).toFixed(1)}" stroke="#1f1c17" stroke-width="1" stroke-opacity="0.35"/>`);
        }
        for (const p of vpos.values()) {
            out.push(`<circle cx="${(p.x * SIZE + ox).toFixed(1)}" cy="${(p.y * SIZE + oy).toFixed(1)}" r="3.2" fill="#fff" stroke="#1f1c17" stroke-width="1.2"/>`);
        }
    }

    // Harbours: a labelled badge pushed out into the sea, with dock lines to the
    // two intersections that can access it. Specific ports carry a resource dot.
    const RESOURCE_COLOR: Record<string, string> = {
        brick: TERRAIN.hills.fill, lumber: TERRAIN.forest.fill, wool: TERRAIN.pasture.fill,
        grain: TERRAIN.fields.fill, ore: TERRAIN.mountains.fill,
    };
    for (const { port, vertices } of boardPorts(board)) {
        const a = vpos.get(vertices[0])!, b = vpos.get(vertices[1])!;
        const ax = a.x * SIZE + ox, ay = a.y * SIZE + oy, bx = b.x * SIZE + ox, by = b.y * SIZE + oy;
        const mx = (ax + bx) / 2, my = (ay + by) / 2;
        let dx = mx - ox, dy = my - oy; const len = Math.hypot(dx, dy) || 1; dx /= len; dy /= len;
        const px = mx + dx * 22, py = my + dy * 22;
        out.push(`<line x1="${ax.toFixed(1)}" y1="${ay.toFixed(1)}" x2="${px.toFixed(1)}" y2="${py.toFixed(1)}" stroke="#9b8f78" stroke-width="1.2" stroke-dasharray="2 2"/>`);
        out.push(`<line x1="${bx.toFixed(1)}" y1="${by.toFixed(1)}" x2="${px.toFixed(1)}" y2="${py.toFixed(1)}" stroke="#9b8f78" stroke-width="1.2" stroke-dasharray="2 2"/>`);
        out.push(`<rect x="${(px - 17).toFixed(1)}" y="${(py - 10).toFixed(1)}" width="34" height="20" rx="6" fill="#f6efe0" stroke="#2c2620" stroke-width="1.2"/>`);
        out.push(`<text x="${px.toFixed(1)}" y="${py.toFixed(1)}" text-anchor="middle" dominant-baseline="central" font-family="Spline Sans,sans-serif" font-weight="600" font-size="11" fill="#2c2620">${port.kind}</text>`);
        if (port.kind === '2:1') {
            out.push(`<circle cx="${(px + 14).toFixed(1)}" cy="${(py - 9).toFixed(1)}" r="4.5" fill="${RESOURCE_COLOR[port.resource]}" stroke="#2c2620" stroke-width="1"/>`);
        }
    }

    out.push('</svg>');
    return {
        svg: out.join(''),
        counts: [tiles.length, Object.keys(board.vertices).length, Object.keys(board.edges).length],
    };
}

export default function ViewerApp() {
    const [seedInput, setSeedInput] = useState('42');
    // The seed actually rendered — updated only by "Render seed"/"Random
    // board", so typing in the box doesn't redraw (same as vanilla).
    const [seed, setSeed] = useState(42);
    const [showGrid, setShowGrid] = useState(false);
    const [noSameNumber, setNoSameNumber] = useState(false);

    const { svg, counts } = useMemo(() => {
        const board = generateBoard(mulberry32(seed), { forbidAdjacentSameNumber: noSameNumber });
        return renderSvg(board, showGrid);
    }, [seed, showGrid, noSameNumber]);

    const [t, v, e] = counts;
    const ok = t === 19 && v === 54 && e === 72;

    return (
        <div className="wrap">
            <h1>Catan board viewer</h1>
            <div className="sub">Rendering boards from the real <code>generateBoard()</code> in @catan/core.</div>
            <div className="controls">
                <label className="field">seed <input id="seed" type="number" value={seedInput} onChange={(e2) => setSeedInput(e2.target.value)} /></label>
                <button id="apply" onClick={() => setSeed(Number(seedInput) || 0)}>Render seed</button>
                <button id="random" className="ghost" onClick={() => {
                    const s = Math.floor(Math.random() * 1e6);
                    setSeedInput(String(s));
                    setSeed(s);
                }}>Random board</button>
                {/* Toggling a checkbox also re-reads the seed box (vanilla's
                    handlers called draw(Number(seedEl.value) || 0)), so a
                    typed-but-not-applied seed takes effect here too. */}
                <label className="toggle"><input id="grid" type="checkbox" checked={showGrid} onChange={(e2) => { setShowGrid(e2.target.checked); setSeed(Number(seedInput) || 0); }} /> show vertices &amp; edges</label>
                <label className="toggle"><input id="nosame" type="checkbox" checked={noSameNumber} onChange={(e2) => { setNoSameNumber(e2.target.checked); setSeed(Number(seedInput) || 0); }} /> no identical neighbours (house rule)</label>
            </div>
            <div className="panel"><div id="board" dangerouslySetInnerHTML={{ __html: svg }} /></div>
            <div className="legend">
                {Object.values(TERRAIN).map((T) => (
                    <span key={T.label}><i style={{ background: T.fill }} />{T.label}</span>
                ))}
            </div>
            <div className="stats" id="stats">
                <b>{t}</b> tiles · <b>{v}</b> vertices · <b>{e}</b> edges{' '}
                <span className={`mark ${ok ? 'ok' : 'bad'}`}>{ok ? '✓ topology matches 19 / 54 / 72' : '✗ unexpected counts'}</span>
            </div>
        </div>
    );
}
