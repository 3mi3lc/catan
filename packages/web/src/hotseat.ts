import {
    generateBoard, initialGameState, applyMove, legalActions, boardPorts,
    roadConnected, tradeRatio, victoryPoints, rngStep, RESOURCES,
    type GameState, type Move, type Action, type GameEvent,
    type PlayerId, type VertexId, type EdgeId, type TileId,
    type Resource, type Terrain, type PlayerColor, type Board,
} from '@catan/core';

// ---------------------------------------------------------------------------
// Rendering geometry (mirrors the viewer; the real topology lives in core).
// ---------------------------------------------------------------------------
const SQRT3 = Math.sqrt(3);
const SIZE = 56;
const hexCenter = (q: number, r: number) => ({ x: SQRT3 * q + (SQRT3 / 2) * r, y: 1.5 * r });
const corner = (cx: number, cy: number, i: number) => {
    const a = (Math.PI / 180) * (60 * i - 90);
    return { x: cx + Math.cos(a), y: cy + Math.sin(a) };
};
const pips = (n: number) => 6 - Math.abs(7 - n);

// An Rng built on the core PRNG, so one integer seed reproduces the whole game.
const rngFromSeed = (seed: number) => {
    let s = seed >>> 0;
    return () => { const { value, next } = rngStep(s); s = next; return value; };
};

const TERRAIN: Record<Terrain, { fill: string; stroke: string }> = {
    fields:    { fill: '#e3b23c', stroke: '#a87b18' },
    forest:    { fill: '#2f6b3c', stroke: '#1d4527' },
    pasture:   { fill: '#9dbe5a', stroke: '#6c8a34' },
    hills:     { fill: '#c5673a', stroke: '#8c4422' },
    mountains: { fill: '#9298a0', stroke: '#666c74' },
    desert:    { fill: '#dccca4', stroke: '#b09c6e' },
};

const RES_COLOR: Record<Resource, string> = {
    brick: '#c5673a', lumber: '#2f6b3c', wool: '#9dbe5a', grain: '#e3b23c', ore: '#9298a0',
};
const RES_ABBR: Record<Resource, string> = { brick: 'Br', lumber: 'Lu', wool: 'Wo', grain: 'Gr', ore: 'Or' };

const PLAYER: Record<PlayerColor, { fill: string; ink: string; name: string }> = {
    red:    { fill: '#c0392b', ink: '#fff', name: 'Red' },
    blue:   { fill: '#2f6aa8', ink: '#fff', name: 'Blue' },
    white:  { fill: '#eceadf', ink: '#2c2620', name: 'White' },
    orange: { fill: '#d98324', ink: '#fff', name: 'Orange' },
};
const SEAT_DEFS = [
    { id: 'p0', color: 'red' as const },
    { id: 'p1', color: 'blue' as const },
    { id: 'p2', color: 'white' as const },
    { id: 'p3', color: 'orange' as const },
];

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------
type Mode =
    | { kind: 'normal' }
    | { kind: 'robber'; via: 'seven' | 'knight' }
    | { kind: 'roadbuilding'; edges: EdgeId[] }
    | { kind: 'yearofplenty'; pick: Resource[] }
    | { kind: 'monopoly' };

let game: GameState;
let mode: Mode = { kind: 'normal' };
let victimChoice: { tile: TileId; via: 'seven' | 'knight'; victims: PlayerId[] } | null = null;
let tradeGive: Resource | null = null;
let discardSel: Record<string, Partial<Record<Resource, number>>> = {};
let log: string[] = [];
let moveCount = 0;   // successful engine moves this game
let turnCount = 1;   // 1-based turn number (increments on endTurn)

// AI opponent (seat p1, 2-player games only). Loaded lazily so the WASM
// runtime + model only download when an AI mode is selected.
import type { Bot, BotLevel } from './bot';
let botLevel: BotLevel | 'human' = 'human';
let bot: Bot | null = null;
let botBusy = false;
const BOT_SEAT: PlayerId = 'p1' as PlayerId;

// Click callbacks for board targets, rebuilt each render.
let vClicks = new Map<VertexId, () => void>();
let eClicks = new Map<EdgeId, () => void>();
let tClicks = new Map<TileId, () => void>();

const handSize = (p: PlayerId) => RESOURCES.reduce((s, r) => s + game.players[p].resources[r], 0);
const nameOf = (p: PlayerId) => PLAYER[game.players[p].color].name;
const colorOf = (p: PlayerId) => PLAYER[game.players[p].color].fill;

// ---------------------------------------------------------------------------
// Applying moves — every interaction funnels through here, so the engine is the
// single source of truth and any illegal move surfaces as a toast.
// ---------------------------------------------------------------------------
function act(move: Move): void {
    const res = applyMove(game, move);
    if (!res.ok) { toast(res.error); return; }
    game = res.state;
    moveCount++;
    if (move.action.type === 'endTurn') turnCount++;
    for (const ev of res.events) log.unshift(describe(ev));
    if (game.winner) log.unshift(`${nameOf(game.winner)} wins in ${turnCount} turns (${moveCount} moves)!`);
    log = log.slice(0, 40);
    mode = { kind: 'normal' };
    victimChoice = null;
    tradeGive = null;
    render();
}

function describe(ev: GameEvent): string {
    switch (ev.type) {
        case 'diceRolled': return `${nameOf(game.currentPlayer)} rolled ${ev.dice[0]} + ${ev.dice[1]} = ${ev.total}`;
        case 'resourcesProduced': {
            const parts = Object.entries(ev.gains).map(([p, g]) => {
                const items = RESOURCES.filter((r) => g[r]).map((r) => `+${g[r]} ${r}`).join(', ');
                return items ? `${nameOf(p as PlayerId)}: ${items}` : '';
            }).filter(Boolean);
            return parts.length ? `Production — ${parts.join('; ')}` : 'Production — nobody collected';
        }
        case 'built': return `${nameOf(ev.player)} built a ${ev.what}`;
        case 'robberMoved': return `${nameOf(game.currentPlayer)} moved the robber${ev.stolen ? ' and stole a card' : ''}`;
        case 'devCardBought': return `${nameOf(ev.player)} bought a development card`;
        case 'tradeExecuted': return `Trade: ${nameOf(ev.between[0])} ↔ ${nameOf(ev.between[1])}`;
        case 'awardMoved': return `${ev.award} → ${nameOf(ev.to)}`;
        case 'gameWon': return `${nameOf(ev.player)} wins!`;
    }
}

// ---------------------------------------------------------------------------
// Board interactions per phase/mode
// ---------------------------------------------------------------------------
function robberVictims(tile: TileId): PlayerId[] {
    const set = new Set<PlayerId>();
    for (const v of game.board.tiles[tile].vertices) {
        const b = game.buildings[v];
        if (b && b.owner !== game.currentPlayer && handSize(b.owner) > 0) set.add(b.owner);
    }
    return [...set];
}

function resolveRobber(tile: TileId, via: 'seven' | 'knight'): void {
    const victims = robberVictims(tile);
    const build = (steal: PlayerId | null): Action =>
        via === 'knight'
            ? { type: 'playKnight', robberTo: tile, stealFrom: steal }
            : { type: 'moveRobber', tile, stealFrom: steal };
    if (victims.length <= 1) act({ player: game.currentPlayer, action: build(victims[0] ?? null) });
    else { victimChoice = { tile, via, victims }; render(); }
}

function computeInteractions(): void {
    vClicks = new Map(); eClicks = new Map(); tClicks = new Map();
    const cur = game.currentPlayer;

    if (mode.kind === 'robber') {
        for (const t of Object.keys(game.board.tiles) as TileId[])
            if (t !== game.robber) tClicks.set(t, () => resolveRobber(t, (mode as { via: 'seven' | 'knight' }).via));
        return;
    }

    if (mode.kind === 'roadbuilding') {
        const placed = mode.edges;
        const temp: GameState = { ...game, roads: { ...game.roads } };
        for (const e of placed) temp.roads[e] = cur;
        for (const e of Object.keys(game.board.edges) as EdgeId[]) {
            if (temp.roads[e] || !roadConnected(temp, cur, e)) continue;
            eClicks.set(e, () => {
                const edges = [...placed, e];
                if (edges.length === 2) act({ player: cur, action: { type: 'playRoadBuilding', edges: [edges[0], edges[1]] } });
                else { mode = { kind: 'roadbuilding', edges }; render(); }
            });
        }
        return;
    }

    // Normal phases: derive directly from the authoritative legal-action list.
    for (const a of legalActions(game, cur)) {
        if (a.type === 'buildSettlement' || a.type === 'buildCity') vClicks.set(a.vertex, () => act({ player: cur, action: a }));
        else if (a.type === 'buildRoad') eClicks.set(a.edge, () => act({ player: cur, action: a }));
        else if (a.type === 'moveRobber' && !tClicks.has(a.tile)) tClicks.set(a.tile, () => resolveRobber(a.tile, 'seven'));
    }
}

// ---------------------------------------------------------------------------
// SVG board
// ---------------------------------------------------------------------------
function vertexPositions(board: Board): Map<VertexId, { x: number; y: number }> {
    const vpos = new Map<VertexId, { x: number; y: number }>();
    for (const t of Object.values(board.tiles)) {
        const c = hexCenter(t.coord.q, t.coord.r);
        t.vertices.forEach((v, i) => { if (!vpos.has(v)) vpos.set(v, corner(c.x, c.y, i)); });
    }
    return vpos;
}

function houseGlyph(cx: number, cy: number, kind: 'settlement' | 'city', fill: string): string {
    const s = kind === 'city' ? 11 : 8;
    const body = `M ${cx - s} ${cy + s} L ${cx - s} ${cy - s * 0.2} L ${cx} ${cy - s} L ${cx + s} ${cy - s * 0.2} L ${cx + s} ${cy + s} Z`;
    const out = [`<path d="${body}" fill="${fill}" stroke="#1c1813" stroke-width="1.6" stroke-linejoin="round"/>`];
    if (kind === 'city') out.push(`<rect x="${cx + 1}" y="${cy - s * 0.2}" width="${s}" height="${s * 1.2}" fill="${fill}" stroke="#1c1813" stroke-width="1.6"/>`);
    return out.join('');
}

function renderBoard(): string {
    const board = game.board;
    const vpos = vertexPositions(board);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of vpos.values()) {
        minX = Math.min(minX, p.x * SIZE); minY = Math.min(minY, p.y * SIZE);
        maxX = Math.max(maxX, p.x * SIZE); maxY = Math.max(maxY, p.y * SIZE);
    }
    const pad = 60, ox = pad - minX, oy = pad - minY;
    const W = maxX - minX + pad * 2, H = maxY - minY + pad * 2;
    const px = (v: VertexId) => ({ x: vpos.get(v)!.x * SIZE + ox, y: vpos.get(v)!.y * SIZE + oy });

    const out: string[] = [`<svg viewBox="0 0 ${W.toFixed(1)} ${H.toFixed(1)}" xmlns="http://www.w3.org/2000/svg">`];

    // Sea backdrop + a soft shadow so the island reads as floating on water.
    out.push(
        `<defs>` +
        `<radialGradient id="sea" cx="50%" cy="42%" r="75%">` +
        `<stop offset="0%" stop-color="#cfe6e8"/><stop offset="70%" stop-color="#a9cdd4"/><stop offset="100%" stop-color="#8fb6bf"/>` +
        `</radialGradient>` +
        `<filter id="tileshadow" x="-20%" y="-20%" width="140%" height="140%">` +
        `<feDropShadow dx="0" dy="2.5" stdDeviation="3" flood-color="#1c2e33" flood-opacity="0.28"/>` +
        `</filter>` +
        `</defs>`,
        `<rect x="0" y="0" width="${W.toFixed(1)}" height="${H.toFixed(1)}" rx="20" fill="url(#sea)"/>`,
        `<g filter="url(#tileshadow)">`,
    );
    // Tiles get the shadow as a group; close the group after the tile loop.

    // Tiles + tokens + robber.
    for (const t of Object.values(board.tiles)) {
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
            out.push(`<circle cx="${cx}" cy="${cy}" r="${SIZE * 0.32}" fill="#f3e9cf" stroke="#bda673" stroke-width="1.5"/>`);
            out.push(`<text x="${cx}" y="${cy - 2}" text-anchor="middle" dominant-baseline="central" font-family="Fraunces,serif" font-weight="600" font-size="${red ? 21 : 18}" fill="${red ? '#b23a2e' : '#34302a'}">${t.numberToken}</text>`);
            const n = pips(t.numberToken), gap = 5, total = (n - 1) * gap;
            for (let i = 0; i < n; i++)
                out.push(`<circle cx="${cx - total / 2 + i * gap}" cy="${cy + SIZE * 0.19}" r="1.6" fill="${red ? '#b23a2e' : '#6b6358'}"/>`);
        }
        if (t.id === game.robber) {
            out.push(`<circle cx="${cx}" cy="${cy - 6}" r="6" fill="#2a2620"/>`);
            out.push(`<path d="M ${cx - 9} ${cy + 12} Q ${cx} ${cy - 2} ${cx + 9} ${cy + 12} Z" fill="#2a2620"/>`);
        }
    }
    out.push('</g>'); // end tile shadow group

    // Harbours: a badge floated into the sea with dock lines to the two
    // intersections that can use it; 2:1 ports carry a resource dot.
    for (const { port, vertices } of boardPorts(board)) {
        const a = px(vertices[0]), b = px(vertices[1]);
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        let dx = mx - ox, dy = my - oy; const len = Math.hypot(dx, dy) || 1; dx /= len; dy /= len;
        const bx = mx + dx * 20, by = my + dy * 20;
        out.push(`<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="#5f7c82" stroke-width="1.4" stroke-dasharray="2 2"/>`);
        out.push(`<line x1="${b.x.toFixed(1)}" y1="${b.y.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="#5f7c82" stroke-width="1.4" stroke-dasharray="2 2"/>`);
        out.push(`<rect x="${(bx - 17).toFixed(1)}" y="${(by - 10).toFixed(1)}" width="34" height="20" rx="6" fill="#f6efe0" stroke="#2c2620" stroke-width="1.2"/>`);
        out.push(`<text x="${bx.toFixed(1)}" y="${by.toFixed(1)}" text-anchor="middle" dominant-baseline="central" font-family="Spline Sans,sans-serif" font-weight="600" font-size="11" fill="#2c2620">${port.kind}</text>`);
        if (port.kind === '2:1')
            out.push(`<circle cx="${(bx + 14).toFixed(1)}" cy="${(by - 9).toFixed(1)}" r="4.5" fill="${RES_COLOR[port.resource]}" stroke="#2c2620" stroke-width="1"/>`);
    }

    // Roads.
    for (const [edge, owner] of Object.entries(game.roads) as [EdgeId, PlayerId][]) {
        const [a, b] = board.edges[edge].vertices;
        const pa = px(a), pb = px(b);
        out.push(`<line x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}" x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}" stroke="${colorOf(owner)}" stroke-width="7" stroke-linecap="round"/>`);
        out.push(`<line x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}" x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}" stroke="#00000022" stroke-width="7" stroke-linecap="round"/>`);
    }

    // Buildings.
    for (const [v, b] of Object.entries(game.buildings) as [VertexId, { kind: 'settlement' | 'city'; owner: PlayerId }][]) {
        const p = px(v);
        out.push(houseGlyph(p.x, p.y, b.kind, colorOf(b.owner)));
    }

    // Interactive highlights (legal targets for the current phase/mode).
    for (const e of eClicks.keys()) {
        const [a, b] = board.edges[e].vertices;
        const pa = px(a), pb = px(b);
        out.push(`<line class="hit" data-kind="edge" data-id="${e}" x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}" x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}" stroke="${colorOf(game.currentPlayer)}" stroke-width="7" stroke-linecap="round" stroke-dasharray="2 6" opacity="0.9"/>`);
        out.push(`<line class="hit" data-kind="edge" data-id="${e}" x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}" x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}" stroke="transparent" stroke-width="16" stroke-linecap="round"/>`);
    }
    for (const v of vClicks.keys()) {
        const p = px(v);
        out.push(`<circle class="hit pulse" data-kind="vertex" data-id="${v}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="9" fill="#ffffffcc" stroke="${colorOf(game.currentPlayer)}" stroke-width="2.5"/>`);
    }
    for (const t of tClicks.keys()) {
        const c = hexCenter(board.tiles[t].coord.q, board.tiles[t].coord.r);
        const cx = c.x * SIZE + ox, cy = c.y * SIZE + oy;
        const pts = Array.from({ length: 6 }, (_, i) => {
            const pp = corner(c.x, c.y, i);
            return `${(pp.x * SIZE + ox).toFixed(1)},${(pp.y * SIZE + oy).toFixed(1)}`;
        }).join(' ');
        out.push(`<polygon class="hit" data-kind="tile" data-id="${t}" points="${pts}" fill="#1c181333" stroke="#1c1813" stroke-width="3" stroke-dasharray="4 4"/>`);
    }

    out.push('</svg>');
    return out.join('');
}

// ---------------------------------------------------------------------------
// Side panels
// ---------------------------------------------------------------------------
function chip(r: Resource, n: number): string {
    return `<span class="res" style="--c:${RES_COLOR[r]}" title="${r}">${RES_ABBR[r]}<b>${n}</b></span>`;
}

function renderPlayers(): string {
    return game.turnOrder.map((p) => {
        const pl = game.players[p];
        const isCur = p === game.currentPlayer;
        const vp = victoryPoints(game, p);
        const devCount = pl.devCards.length + pl.pendingDevCards.length;
        const badges = [
            game.longestRoad?.player === p ? '<span class="badge">Longest road</span>' : '',
            game.largestArmy?.player === p ? '<span class="badge">Largest army</span>' : '',
        ].join('');
        return `
      <div class="pcard${isCur ? ' cur' : ''}">
        <div class="phead">
          <span class="dot" style="background:${PLAYER[pl.color].fill}"></span>
          <span class="pname">${PLAYER[pl.color].name}</span>
          <span class="vp" title="victory points">${vp} VP</span>
        </div>
        <div class="hand">${RESOURCES.map((r) => chip(r, pl.resources[r])).join('')}</div>
        <div class="meta">Dev: ${devCount} · Knights: ${pl.playedKnights} · Pieces: ${pl.supply.settlements}s/${pl.supply.cities}c/${pl.supply.roads}r ${badges}</div>
      </div>`;
    }).join('');
}

function renderActions(): string {
    if (game.winner) return `<div class="won">${nameOf(game.winner)} wins the game! 🏆</div>`;

    const cur = game.currentPlayer;
    const acts = legalActions(game, cur);
    const has = (t: Action['type']) => acts.some((a) => a.type === t);
    const turn = `<div class="turnline"><span class="dot" style="background:${colorOf(cur)}"></span> <b>${nameOf(cur)}</b>`;

    if (victimChoice) {
        const btns = victimChoice.victims.map((vp) =>
            `<button data-steal="${vp}">${nameOf(vp)} (${handSize(vp)})</button>`).join('');
        return `${turn} — steal from whom?</div><div class="row">${btns}</div>`;
    }

    if (game.phase === 'discard') return renderDiscard();

    if (mode.kind === 'robber') return `${turn} — click a tile to move the robber.</div>`;
    if (game.phase === 'moveRobber') return `${turn} — move the robber (click a tile).</div>`;
    if (mode.kind === 'roadbuilding') return `${turn} — Road Building: pick ${2 - mode.edges.length} more road(s).</div><div class="row"><button class="ghost" data-cancel>Cancel</button></div>`;
    if (mode.kind === 'monopoly') return `${turn} — Monopoly: choose a resource.</div><div class="row">${RESOURCES.map((r) => `<button data-monopoly="${r}" style="--c:${RES_COLOR[r]}">${r}</button>`).join('')}<button class="ghost" data-cancel>Cancel</button></div>`;
    if (mode.kind === 'yearofplenty') return `${turn} — Year of Plenty: choose 2 (picked ${mode.pick.length}).</div><div class="row">${RESOURCES.map((r) => `<button data-yop="${r}" style="--c:${RES_COLOR[r]}">${r}</button>`).join('')}<button class="ghost" data-cancel>Cancel</button></div>`;

    const buttons: string[] = [];
    if (game.phase === 'setupSettlement') buttons.push('<span class="hint">Place your starting settlement (click a highlighted spot).</span>');
    if (game.phase === 'setupRoad') buttons.push('<span class="hint">Place a road next to it.</span>');
    if (has('rollDice')) buttons.push('<button data-roll>Roll dice</button>');
    if (game.phase === 'main') {
        buttons.push('<span class="hint">Build by clicking highlighted spots.</span>');
        if (has('buyDevCard')) buttons.push('<button data-buydev>Buy dev card</button>');
    }
    // Dev cards are playable in BOTH the roll and main phases (official rule:
    // e.g. a knight before rolling). legalActions already enumerates them.
    if (game.phase === 'main' || game.phase === 'roll') {
        if (has('playKnight')) buttons.push('<button data-knight>Play Knight</button>');
        if (has('playRoadBuilding')) buttons.push('<button data-roadbuild>Road Building</button>');
        if (has('playYearOfPlenty')) buttons.push('<button data-yopstart>Year of Plenty</button>');
        if (has('playMonopoly')) buttons.push('<button data-monopolystart>Monopoly</button>');
    }
    if (game.phase === 'main' && has('endTurn')) buttons.push('<button data-endturn>End turn</button>');

    return `${turn} — ${phaseLabel()}</div><div class="row">${buttons.join('')}</div>${game.phase === 'main' ? renderTrade() : ''}`;
}

function phaseLabel(): string {
    return {
        setupSettlement: 'initial settlement', setupRoad: 'initial road', roll: 'roll the dice',
        discard: 'discard', moveRobber: 'move robber', main: 'build & trade', gameOver: 'game over',
    }[game.phase];
}

function renderTrade(): string {
    const cur = game.currentPlayer;
    const gives = RESOURCES.map((r) => {
        const ratio = tradeRatio(game, cur, r);
        const can = game.players[cur].resources[r] >= ratio;
        const sel = tradeGive === r ? ' sel' : '';
        return `<button class="trade${sel}" data-give="${r}" ${can ? '' : 'disabled'} style="--c:${RES_COLOR[r]}">${ratio} ${RES_ABBR[r]}</button>`;
    }).join('');
    const receives = tradeGive
        ? RESOURCES.filter((r) => r !== tradeGive && game.bank[r] > 0)
            .map((r) => `<button class="trade" data-receive="${r}" style="--c:${RES_COLOR[r]}">${RES_ABBR[r]}</button>`).join('')
        : '<span class="hint">pick what to give</span>';
    return `<div class="trade-wrap"><div class="trade-lab">Bank trade — give</div><div class="row">${gives}</div><div class="trade-lab">receive</div><div class="row">${receives}</div></div>`;
}

function renderDiscard(): string {
    const owe = Object.keys(game.pendingDiscards) as PlayerId[];
    const cards = owe.map((p) => {
        const sel = discardSel[p] ?? {};
        const chosen = RESOURCES.reduce((s, r) => s + (sel[r] ?? 0), 0);
        const need = game.pendingDiscards[p];
        const chips = RESOURCES.map((r) => {
            const have = game.players[p].resources[r];
            const picked = sel[r] ?? 0;
            return `<span class="dres" style="--c:${RES_COLOR[r]}">${RES_ABBR[r]} ${picked}/${have}
        <button data-dminus="${p}:${r}" ${picked <= 0 ? 'disabled' : ''}>−</button>
        <button data-dplus="${p}:${r}" ${picked >= have || chosen >= need ? 'disabled' : ''}>+</button></span>`;
        }).join('');
        return `<div class="dcard"><div class="phead"><span class="dot" style="background:${colorOf(p)}"></span><b>${nameOf(p)}</b> must discard ${need} (${chosen}/${need})</div>
      <div class="hand">${chips}</div>
      <button data-discard="${p}" ${chosen === need ? '' : 'disabled'}>Confirm discard</button></div>`;
    }).join('');
    return `<div class="hint">A 7 was rolled — players over 7 cards discard half.</div>${cards}`;
}

// ---------------------------------------------------------------------------
// Render + event wiring
// ---------------------------------------------------------------------------
function render(): void {
    computeInteractions();
    document.getElementById('board')!.innerHTML = renderBoard();
    document.getElementById('players')!.innerHTML = renderPlayers();
    document.getElementById('actions')!.innerHTML = renderActions();
    document.getElementById('log')!.innerHTML = log.map((l) => `<li>${l}</li>`).join('');
    const stats = document.getElementById('logstats');
    if (stats) stats.textContent = `turn ${turnCount} · ${moveCount} moves`;

    // When it's the AI's turn (outside the discard phase, where the human may
    // also owe cards), replace the action panel so the human can't act for it.
    if (botTurnPending() && game.phase !== 'discard') {
        document.getElementById('actions')!.innerHTML =
            `<div class="hint">🤖 ${nameOf(BOT_SEAT)} is thinking${botLevel === 'mcts' ? ' (searching)…' : '…'}</div>`;
    }
    wire();
    maybeBotMove();
}

let toastTimer: number | undefined;
function toast(msg: string): void {
    const el = document.getElementById('toast')!;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => el.classList.remove('show'), 2200);
}

function wire(): void {
    document.querySelectorAll<SVGElement>('.hit').forEach((el) => {
        el.addEventListener('click', () => {
            const kind = el.dataset.kind, id = el.dataset.id!;
            if (kind === 'vertex') vClicks.get(id as VertexId)?.();
            else if (kind === 'edge') eClicks.get(id as EdgeId)?.();
            else if (kind === 'tile') tClicks.get(id as TileId)?.();
        });
    });

    const cur = game.currentPlayer;
    const click = (sel: string, fn: (el: HTMLElement) => void) =>
        document.querySelectorAll<HTMLElement>(sel).forEach((el) => el.addEventListener('click', () => fn(el)));

    click('[data-roll]', () => act({ player: cur, action: { type: 'rollDice' } }));
    click('[data-endturn]', () => act({ player: cur, action: { type: 'endTurn' } }));
    click('[data-buydev]', () => act({ player: cur, action: { type: 'buyDevCard' } }));
    click('[data-knight]', () => { mode = { kind: 'robber', via: 'knight' }; render(); });
    click('[data-roadbuild]', () => { mode = { kind: 'roadbuilding', edges: [] }; render(); });
    click('[data-yopstart]', () => { mode = { kind: 'yearofplenty', pick: [] }; render(); });
    click('[data-monopolystart]', () => { mode = { kind: 'monopoly' }; render(); });
    click('[data-cancel]', () => { mode = { kind: 'normal' }; render(); });

    click('[data-monopoly]', (el) => act({ player: cur, action: { type: 'playMonopoly', resource: el.dataset.monopoly as Resource } }));
    click('[data-yop]', (el) => {
        if (mode.kind !== 'yearofplenty') return;
        const pick = [...mode.pick, el.dataset.yop as Resource];
        if (pick.length === 2) act({ player: cur, action: { type: 'playYearOfPlenty', take: [pick[0], pick[1]] } });
        else { mode = { kind: 'yearofplenty', pick }; render(); }
    });

    click('[data-steal]', (el) => {
        if (!victimChoice) return;
        const steal = el.dataset.steal as PlayerId;
        const a: Action = victimChoice.via === 'knight'
            ? { type: 'playKnight', robberTo: victimChoice.tile, stealFrom: steal }
            : { type: 'moveRobber', tile: victimChoice.tile, stealFrom: steal };
        act({ player: cur, action: a });
    });

    click('[data-give]', (el) => { tradeGive = el.dataset.give as Resource; render(); });
    click('[data-receive]', (el) => {
        if (!tradeGive) return;
        const give = tradeGive;
        act({ player: cur, action: { type: 'bankTrade', give, giveCount: tradeRatio(game, cur, give), receive: el.dataset.receive as Resource } });
    });

    // Discard controls.
    const adjust = (key: string, delta: number) => {
        const [p, r] = key.split(':') as [PlayerId, Resource];
        const sel = { ...(discardSel[p] ?? {}) };
        sel[r] = Math.max(0, (sel[r] ?? 0) + delta);
        discardSel = { ...discardSel, [p]: sel };
        render();
    };
    click('[data-dplus]', (el) => adjust(el.dataset.dplus!, +1));
    click('[data-dminus]', (el) => adjust(el.dataset.dminus!, -1));
    click('[data-discard]', (el) => {
        const p = el.dataset.discard as PlayerId;
        const resources = discardSel[p] ?? {};
        const r = applyMove(game, { player: p, action: { type: 'discard', resources } });
        if (!r.ok) { toast(r.error); return; }
        game = r.state;
        discardSel = { ...discardSel, [p]: {} };
        render();
    });
}

// ---------------------------------------------------------------------------
// AI opponent turn loop
// ---------------------------------------------------------------------------
function botTurnPending(): boolean {
    if (botLevel === 'human' || !bot || game.winner) return false;
    if (game.phase === 'discard') {
        return BOT_SEAT in game.pendingDiscards;
    }
    return game.currentPlayer === BOT_SEAT;
}

function maybeBotMove(): void {
    if (!botTurnPending() || botBusy) return;
    botBusy = true;
    // Small delay so the human can follow the log between bot moves.
    setTimeout(async () => {
        try {
            const action = await bot!.decide(game, BOT_SEAT, botLevel as BotLevel);
            botBusy = false;
            act({ player: BOT_SEAT, action });   // render() → maybeBotMove() chains
        } catch (err) {
            botBusy = false;
            toast(`AI error: ${err instanceof Error ? err.message : err}`);
        }
    }, 220);
}

// ---------------------------------------------------------------------------
// New game + page shell
// ---------------------------------------------------------------------------
function startGame(seed: number, players: number, noSameNumbers = false): void {
    const board = generateBoard(rngFromSeed(seed), { forbidAdjacentSameNumber: noSameNumbers });
    const seats = SEAT_DEFS.slice(0, players).map((s) => ({ id: s.id, name: PLAYER[s.color].name, color: s.color }));
    game = initialGameState(board, seats, seed);
    mode = { kind: 'normal' };
    victimChoice = null; tradeGive = null; discardSel = {};
    log = [`New game · seed ${seed} · ${players} players`];
    moveCount = 0;
    turnCount = 1;
    render();
}

const app = document.getElementById('app')!;
app.innerHTML = `
  <style>
    :root{--bg:#ece2cb;--panel:#f6efe0;--ink:#2c2620;--muted:#7d7361;--line:#d8cbac;--accent:#b5552f}
    *{box-sizing:border-box}
    body{margin:0;min-height:100vh;background:radial-gradient(1200px 700px at 50% -10%,#f1e8d2,transparent 60%),var(--bg);color:var(--ink);font-family:"Spline Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
    .wrap{max-width:1180px;margin:0 auto;padding:28px 22px 60px}
    h1{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:32px;letter-spacing:-.5px;margin:0}
    .sub{color:var(--muted);font-size:14px;margin:2px 0 18px}
    .topbar{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:18px}
    .topbar label{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--muted)}
    input[type=number]{width:90px;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);font:inherit}
    select{padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);font:inherit}
    .layout{display:grid;grid-template-columns:1fr 320px;gap:22px;align-items:start}
    @media(max-width:880px){.layout{grid-template-columns:1fr}}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 18px 40px -30px rgba(60,45,20,.5)}
    svg{display:block;width:100%;height:auto}
    .hit{cursor:pointer}
    .pulse{animation:pulse 1.4s ease-in-out infinite}
    @keyframes pulse{0%,100%{opacity:.55}50%{opacity:1}}
    button{font:inherit;font-weight:600;font-size:13px;cursor:pointer;border:none;border-radius:9px;padding:8px 13px;background:var(--accent);color:#fff;transition:transform .08s,filter .15s}
    button:hover{filter:brightness(1.07)} button:active{transform:translateY(1px)} button:disabled{opacity:.4;cursor:not-allowed}
    button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
    .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
    .turnline{font-size:15px;display:flex;align-items:center;gap:8px}
    .hint{color:var(--muted);font-size:13px}
    .dot{width:13px;height:13px;border-radius:50%;display:inline-block;border:1px solid #0003}
    .pcard{border:1px solid var(--line);border-radius:13px;padding:11px 12px;margin-bottom:10px;background:#fffdf7}
    .pcard.cur{border-color:var(--accent);box-shadow:0 0 0 2px #b5552f33}
    .phead{display:flex;align-items:center;gap:8px;font-size:14px}
    .pname{font-weight:600}.vp{margin-left:auto;font-family:"Fraunces",serif;font-weight:600}
    .hand{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
    .res{display:inline-flex;align-items:center;gap:4px;font-size:12px;color:#3a342c;background:#fff;border:1px solid var(--line);border-left:4px solid var(--c);border-radius:7px;padding:3px 7px}
    .res b{font-weight:700}
    .meta{font-size:11.5px;color:var(--muted)}
    .badge{display:inline-block;background:#2c2620;color:#f6efe0;border-radius:6px;padding:1px 6px;font-size:10.5px;margin-left:5px}
    [data-give],[data-receive],[data-monopoly],[data-yop]{background:#fff;color:var(--ink);border:1px solid var(--line);border-left:4px solid var(--c)}
    .trade.sel{outline:2px solid var(--accent)}
    .trade-wrap{margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
    .trade-lab{font-size:12px;color:var(--muted);margin-top:6px}
    .dcard{border:1px solid var(--line);border-radius:11px;padding:10px;margin-top:8px;background:#fffdf7}
    .dres{display:inline-flex;align-items:center;gap:5px;font-size:12px;background:#fff;border:1px solid var(--line);border-left:4px solid var(--c);border-radius:7px;padding:2px 5px}
    .dres button{padding:1px 7px;font-size:13px;background:var(--line);color:var(--ink)}
    .won{font-family:"Fraunces",serif;font-size:20px;font-weight:600}
    h2{font-family:"Fraunces",serif;font-size:16px;margin:0 0 8px}
    #log{list-style:none;margin:0;padding:0;max-height:280px;overflow:auto;font-size:12.5px;color:#4a4339}
    #log li{padding:5px 0;border-bottom:1px solid var(--line)}
    #toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(20px);background:#2c2620;color:#f6efe0;padding:10px 16px;border-radius:10px;font-size:13px;opacity:0;pointer-events:none;transition:.2s}
    #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
  </style>
  <div class="wrap">
    <h1>Catan — hotseat</h1>
    <div class="sub">Pass-and-play on one screen. Every click is validated by the real <code>@catan/core</code> engine.</div>
    <div class="topbar">
      <label>seed <input id="seed" type="number" value="42" /></label>
      <label>players
        <select id="playercount"><option>2</option><option>3</option><option selected>4</option></select>
      </label>
      <label>opponent
        <select id="opponent">
          <option value="human" selected>humans (pass &amp; play)</option>
          <option value="net">AI — fast (raw net)</option>
          <option value="mcts">AI — strong (net + search)</option>
        </select>
      </label>
      <label title="House rule: identical number tokens never on adjacent tiles">
        <input id="nosame" type="checkbox" checked /> no same numbers adjacent
      </label>
      <button id="newgame">New game</button>
      <a href="/" style="margin-left:auto;font-size:13px;color:var(--muted)">board viewer →</a>
    </div>
    <div class="layout">
      <div class="panel"><div id="board"></div><div id="actions" style="margin-top:14px"></div></div>
      <div>
        <div class="panel" style="margin-bottom:18px"><h2>Players</h2><div id="players"></div></div>
        <div class="panel"><h2>Log <small id="logstats" style="font-weight:normal;color:#8a7f6e"></small></h2><ul id="log"></ul></div>
      </div>
    </div>
  </div>
  <div id="toast"></div>
`;

const seedInput = document.getElementById('seed') as HTMLInputElement;
const playerSelect = document.getElementById('playercount') as HTMLSelectElement;
const opponentSelect = document.getElementById('opponent') as HTMLSelectElement;

opponentSelect.onchange = () => {
    if (opponentSelect.value !== 'human') {
        playerSelect.value = '2';            // AI games are 1v1 (it plays seat 2)
        playerSelect.disabled = true;
    } else {
        playerSelect.disabled = false;
    }
};

(document.getElementById('newgame') as HTMLButtonElement).onclick = async () => {
    botLevel = opponentSelect.value as BotLevel | 'human';
    if (botLevel !== 'human' && !bot) {
        toast('Loading AI model…');
        try {
            const { Bot } = await import('./bot');
            bot = await Bot.load('/models/catan_net.onnx');
        } catch (err) {
            toast(`Could not load AI: ${err instanceof Error ? err.message : err}`);
            botLevel = 'human';
        }
    }
    const players = botLevel !== 'human' ? 2 : Number(playerSelect.value);
    const noSame = (document.getElementById('nosame') as HTMLInputElement).checked;
    startGame(Number(seedInput.value) || 0, players, noSame);
};

startGame(42, 4);