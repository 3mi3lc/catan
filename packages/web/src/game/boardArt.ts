// Single source of truth for the board's art — both the STATIC layer (sea,
// terrain tiles, number tokens, harbour badges) and the shared geometry/style
// recipe for the DYNAMIC pieces (settlements, cities, roads, the robber).
// Pure SVG-string functions (no React, no DOM) so every renderer shares one
// definition: the vanilla string-SVG clients (online.ts, hotseat.ts) splice
// these in directly, and the React Board.tsx injects the static layer as one
// <g dangerouslySetInnerHTML>, and individual pieces the same way inside their
// own <motion.g> wrappers (so framer-motion entrance animation still applies
// per piece — dangerouslySetInnerHTML on a leaf node with no JSX children is
// fine for that).
//
// Phase 2 (illustrated tiles) swaps `tileBody` from a flat fill to a clipped
// <image>; callers don't change. Until art assets exist it renders the
// procedural lacquered look below, so the board is never blank.
//
// Phase 3 (this file's piece/token/port functions) adds a shared depth recipe
// — a small drop-shadow filter + a diagonal glass-sheen gradient overlay —
// applied identically to every piece regardless of owner colour, so light
// reads consistently across the whole board without needing a gradient per
// player colour.

import { boardPorts, type Board, type Tile, type TileId } from '@catan/core';
import { SIZE, hexCenter, corner, pips, type Layout } from './theme';

// Lacquered terrain palette for the procedural fallback (and the base tint the
// illustrated tiles sit on). Richer/darker than the debug viewer's flat fills.
const TILE_FILL: Record<Tile['terrain'], { fill: string; stroke: string }> = {
    fields:    { fill: '#d9a437', stroke: '#9a6f15' },
    forest:    { fill: '#2e6038', stroke: '#1a3d22' },
    pasture:   { fill: '#92b050', stroke: '#5f7c2c' },
    hills:     { fill: '#b9552e', stroke: '#7e3a1c' },
    mountains: { fill: '#7e858e', stroke: '#525861' },
    desert:    { fill: '#cdb988', stroke: '#998354' },
};

const PORT_DOT: Record<string, string> = {
    brick: '#c5673a', lumber: '#3d7a4a', wool: '#9dbe5a', grain: '#e3b23c', ore: '#98a0aa',
};

// How many illustrated variants exist for each terrain, under
// `public/tiles/<terrain>-<n>.webp` (1-based). 0 = no art yet → that terrain
// falls back to the procedural fill above. Bump a number here once the matching
// WebP files are dropped in (see public/tiles/PROMPTS.md). Defaults to 0 so the
// board is unchanged until real art lands — and a declared-but-missing file
// still degrades gracefully, because the procedural polygon is always drawn
// underneath the image.
const TILE_VARIANTS: Record<Tile['terrain'], number> = {
    fields: 4, forest: 4, pasture: 4, hills: 4, mountains: 4, desert: 4,
};

// Deterministic-but-game-varying variant pick. Tile ids are positional
// ("t0".."t18") and identical across every game, so hashing on id alone would
// pick the SAME variant for the SAME board position in every game ever
// played — not actually random. Terrain and number-token assignment, by
// contrast, ARE re-shuffled per game (see board-generator.ts), so hashing on
// id+terrain+numberToken instead gives a pick that varies game to game while
// staying stable for the lifetime of one game (no flicker on re-render, since
// neither terrain nor numberToken change after the board is generated).
function variantFor(t: Tile, count: number): number {
    if (count <= 1) return 1;
    const key = `${t.id}:${t.terrain}:${t.numberToken}`;
    let h = 0;
    for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
    return (h % count) + 1;
}

// The illustrated-tile URL for a tile, or null when that terrain has no art.
function tileAsset(t: Tile): string | null {
    const count = TILE_VARIANTS[t.terrain];
    if (!count) return null;
    return `/tiles/${t.terrain}-${variantFor(t, count)}.webp`;
}

// A square illustrated tile clipped to the hex. The art is authored square and
// hex-cropped on the fly (slice = cover), so the source needs no hex alpha.
function clippedImage(t: Tile, L: Layout, points: string, src: string): string {
    const { cx, cy } = tileCenter(t, L);
    const clipId = `ba-clip-${t.id}`;
    const side = (SIZE * 2).toFixed(1);
    return (
        `<clipPath id="${clipId}"><polygon points="${points}"/></clipPath>` +
        `<image href="${src}" x="${(cx - SIZE).toFixed(1)}" y="${(cy - SIZE).toFixed(1)}" ` +
        `width="${side}" height="${side}" preserveAspectRatio="xMidYMid slice" clip-path="url(#${clipId})"/>`
    );
}

// Pixel-space centre of a tile under the given layout.
function tileCenter(t: Tile, L: Layout): { cx: number; cy: number } {
    const c = hexCenter(t.coord.q, t.coord.r);
    return { cx: c.x * SIZE + L.ox, cy: c.y * SIZE + L.oy };
}

// Pixel-space hex polygon points string for a tile.
function hexPoints(t: Tile, L: Layout): string {
    const c = hexCenter(t.coord.q, t.coord.r);
    return Array.from({ length: 6 }, (_, i) => {
        const p = corner(c.x, c.y, i);
        return `${(p.x * SIZE + L.ox).toFixed(1)},${(p.y * SIZE + L.oy).toFixed(1)}`;
    }).join(' ');
}

/** Reusable gradients/filters. Ids are namespaced (`ba-*`) so this can be
 *  injected alongside a renderer's own <defs> without colliding. */
export function boardDefs(): string {
    return (
        `<defs>` +
        `<radialGradient id="ba-sea" cx="50%" cy="40%" r="80%">` +
        `<stop offset="0%" stop-color="#27414a"/><stop offset="55%" stop-color="#1b2e35"/><stop offset="100%" stop-color="#101c21"/>` +
        `</radialGradient>` +
        `<radialGradient id="ba-lamp" cx="50%" cy="36%" r="70%">` +
        `<stop offset="0%" stop-color="#ffdf9e" stop-opacity="0.13"/><stop offset="60%" stop-color="#ffdf9e" stop-opacity="0.03"/><stop offset="100%" stop-color="#ffdf9e" stop-opacity="0"/>` +
        `</radialGradient>` +
        `<filter id="ba-tileshadow" x="-20%" y="-20%" width="140%" height="140%">` +
        `<feDropShadow dx="0" dy="3.5" stdDeviation="4" flood-color="#000000" flood-opacity="0.55"/>` +
        `</filter>` +
        `<filter id="ba-pieceshadow" x="-60%" y="-60%" width="220%" height="220%">` +
        `<feDropShadow dx="0" dy="1.4" stdDeviation="1.1" flood-color="#000000" flood-opacity="0.5"/>` +
        `</filter>` +
        // A diagonal glass-sheen overlay (bright top-left fading to a faint
        // dark shade bottom-right) — drawn a second time over any flat-filled
        // piece shape to read as a lit, beveled surface. Works for ANY owner
        // colour since it's additive, not a per-colour gradient.
        `<linearGradient id="ba-bevel" x1="0%" y1="0%" x2="100%" y2="100%">` +
        `<stop offset="0%" stop-color="#ffffff" stop-opacity="0.55"/>` +
        `<stop offset="55%" stop-color="#ffffff" stop-opacity="0.06"/>` +
        `<stop offset="100%" stop-color="#000000" stop-opacity="0.2"/>` +
        `</linearGradient>` +
        // Plays once per dice roll on the tiles that just produced (see
        // tileProducePulse()) — a quick scale bounce in, a hold, then a fade
        // out. `both` (not `infinite`) so it settles at its 100% keyframe
        // (invisible) once done rather than snapping back to frame 0.
        // Embedded as a <style> tag in the SVG itself, not one of the three
        // separate app stylesheets, so it travels wherever this markup is
        // injected (hotseat/online splice these defs in via
        // dangerouslySetInnerHTML — neither shares a CSS file with the other).
        `<style>` +
        `.ba-produce-pop{transform-box:fill-box;transform-origin:center;animation:ba-produce-pop-kf 1.6s cubic-bezier(.2,.8,.3,1) both}` +
        `@keyframes ba-produce-pop-kf{0%{transform:scale(1);opacity:0}15%{transform:scale(1.08);opacity:1}30%{transform:scale(1);opacity:1}75%{opacity:1}100%{transform:scale(1);opacity:0}}` +
        `</style>` +
        `</defs>`
    );
}

/** The night-sea backdrop with a warm lamp glow over the island. */
export function seaLayer(L: Layout): string {
    const w = L.W.toFixed(1), h = L.H.toFixed(1);
    return (
        `<rect x="0" y="0" width="${w}" height="${h}" rx="22" fill="url(#ba-sea)"/>` +
        `<rect x="0" y="0" width="${w}" height="${h}" rx="22" fill="url(#ba-lamp)"/>`
    );
}

// The terrain face of one tile, in three layers, bottom to top:
//   1. procedural fill — the floor, always present (so a missing illustration
//      shows the flat terrain colour rather than a hole);
//   2. the illustrated tile, hex-clipped, when this terrain has art;
//   3. a crisp rim stroke that reads on both the flat fill and the illustration.
function tileBody(t: Tile, L: Layout): string {
    const T = TILE_FILL[t.terrain];
    const points = hexPoints(t, L);
    const base = `<polygon points="${points}" fill="${T.fill}" stroke="none"/>`;
    const src = tileAsset(t);
    const art = src ? clippedImage(t, L, points, src) : '';
    const rim = `<polygon points="${points}" fill="none" stroke="${T.stroke}" stroke-width="2" stroke-linejoin="round"/>`;
    return base + art + rim;
}

// The clay number-token chip (red for the 6/8 hot numbers) with its pip row —
// drop-shadowed and bevel-sheened to read as a raised ceramic disc.
function numberToken(t: Tile, L: Layout): string {
    if (t.numberToken === null) return '';
    const { cx, cy } = tileCenter(t, L);
    const red = t.numberToken === 6 || t.numberToken === 8;
    const r = SIZE * 0.33;
    const out = [
        `<g filter="url(#ba-pieceshadow)">`,
        `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#f0e4c4" stroke="#8a6f3e" stroke-width="1.6"/>`,
        `<circle cx="${cx}" cy="${cy}" r="${SIZE * 0.27}" fill="none" stroke="#c8b389" stroke-width="0.8"/>`,
        `<text x="${cx}" y="${cy - 2}" text-anchor="middle" dominant-baseline="central" font-family="Marcellus,serif" font-size="${red ? 21 : 18}" fill="${red ? '#a8321f' : '#3b342a'}">${t.numberToken}</text>`,
    ];
    const n = pips(t.numberToken), gap = 5, total = (n - 1) * gap;
    for (let i = 0; i < n; i++)
        out.push(`<circle cx="${cx - total / 2 + i * gap}" cy="${cy + SIZE * 0.19}" r="1.6" fill="${red ? '#a8321f' : '#6f6450'}"/>`);
    out.push(`<circle cx="${cx}" cy="${cy}" r="${r}" fill="url(#ba-bevel)"/>`, `</g>`);
    return out.join('');
}

/** Terrain tiles + number tokens, as one drop-shadowed group. */
export function tilesLayer(board: Board, L: Layout): string {
    const parts = ['<g filter="url(#ba-tileshadow)">'];
    for (const t of Object.values(board.tiles)) parts.push(tileBody(t, L), numberToken(t, L));
    parts.push('</g>');
    return parts.join('');
}

/** Every tile that just produced on a given roll — every tile whose number
 *  token matches the total, except whichever one the robber sits on (that
 *  one is blocked regardless of a number match — the real game rule, not
 *  just a visual nicety). Callers re-derive this fresh each roll from the
 *  event's `total`, not from stored state, so it's always in sync with what
 *  the server/engine actually resolved. */
export function producingTilesForRoll(board: Board, robber: TileId, total: number): TileId[] {
    return Object.values(board.tiles)
        .filter((t) => t.numberToken === total && t.id !== robber)
        .map((t) => t.id);
}

/** A brief golden pop/glow on each tile in `tiles` — the "these hexes just
 *  produced" flash for a dice roll (see boardDefs()'s ba-produce-pop rules).
 *  Draw this ABOVE tilesLayer() so the glow reads over the terrain art. */
export function tileProducePulse(board: Board, L: Layout, tiles: readonly TileId[]): string {
    if (!tiles.length) return '';
    const out: string[] = [];
    for (const id of tiles) {
        const t = board.tiles[id];
        if (!t) continue;
        const points = hexPoints(t, L);
        out.push(
            `<g class="ba-produce-pop">` +
            `<polygon points="${points}" fill="#ffd98c" fill-opacity="0.24" stroke="#ffd98c" stroke-width="3.5" stroke-linejoin="round"/>` +
            `</g>`,
        );
    }
    return out.join('');
}

// Tiny abstract resource icon (drawn in a dark ink over the coloured badge
// dot) — kept very simple since it's rendered at ~3px, just enough silhouette
// to distinguish resources at a glance beyond colour alone.
function resourceGlyph(resource: string, cx: number, cy: number): string {
    const ink = '#17130eb0';
    switch (resource) {
        case 'lumber': // little fir tree
            return `<path d="M ${cx} ${cy - 3} L ${cx + 2.6} ${cy + 1.4} L ${cx - 2.6} ${cy + 1.4} Z M ${cx} ${cy - 1} L ${cx + 2.2} ${cy + 3} L ${cx - 2.2} ${cy + 3} Z" fill="${ink}"/>`;
        case 'brick': // brick block
            return `<rect x="${cx - 3}" y="${cy - 2}" width="6" height="4" rx="0.6" fill="none" stroke="${ink}" stroke-width="1"/>`;
        case 'wool': // sheep fluff — three overlapping puffs
            return `<circle cx="${cx - 1.6}" cy="${cy}" r="1.7" fill="${ink}"/><circle cx="${cx + 1.6}" cy="${cy}" r="1.7" fill="${ink}"/><circle cx="${cx}" cy="${cy - 1.4}" r="1.7" fill="${ink}"/>`;
        case 'grain': // wheat sheaf — three fanned blades
            return `<path d="M ${cx} ${cy + 3} L ${cx} ${cy - 3} M ${cx} ${cy + 3} L ${cx - 2.4} ${cy - 2} M ${cx} ${cy + 3} L ${cx + 2.4} ${cy - 2}" stroke="${ink}" stroke-width="1" stroke-linecap="round" fill="none"/>`;
        case 'ore': // gem facet
            return `<path d="M ${cx} ${cy - 3.2} L ${cx + 3} ${cy} L ${cx} ${cy + 3.2} L ${cx - 3} ${cy} Z" fill="${ink}"/>`;
        default:
            return '';
    }
}

/** Harbour badges floated into the sea with dock lines to their two vertices —
 *  drop-shadowed and bevel-sheened to match the rest of the piece set.
 *
 *  The badge is positioned perpendicular to the actual edge (a→b) and pushed
 *  outward — away from the board's pixel centre (W/2, H/2). Using L.ox/L.oy
 *  as the "origin" was wrong (those are coordinate offsets, not the centre),
 *  causing badges to clip into adjacent tiles and dock lines to run at wrong
 *  angles. */
export function portsLayer(board: Board, L: Layout): string {
    const px = (v: string) => L.pos.get(v as never)!;
    const cx = L.W / 2, cy = L.H / 2; // true board centre in pixel space
    const out: string[] = [];
    for (const { port, vertices } of boardPorts(board)) {
        const a = px(vertices[0]), b = px(vertices[1]);
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;

        // Edge-perpendicular direction — rotated 90° from the a→b vector.
        // Pick the perpendicular that points AWAY from the board centre so
        // the badge always lands in the sea, never inside a land tile.
        const eLen = Math.hypot(b.x - a.x, b.y - a.y) || 1;
        let nx = -(b.y - a.y) / eLen, ny = (b.x - a.x) / eLen;
        if (nx * (mx - cx) + ny * (my - cy) < 0) { nx = -nx; ny = -ny; }

        const bx = mx + nx * 22, by = my + ny * 22;
        out.push(`<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="#5b7178" stroke-width="1.4" stroke-dasharray="2 3"/>`);
        out.push(`<line x1="${b.x.toFixed(1)}" y1="${b.y.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="#5b7178" stroke-width="1.4" stroke-dasharray="2 3"/>`);
        out.push(`<g filter="url(#ba-pieceshadow)">`);
        out.push(`<rect x="${(bx - 17).toFixed(1)}" y="${(by - 10).toFixed(1)}" width="34" height="20" rx="7" fill="#211c15" stroke="#c9a45c" stroke-width="1.1"/>`);
        out.push(`<text x="${bx.toFixed(1)}" y="${by.toFixed(1)}" text-anchor="middle" dominant-baseline="central" font-family="Alegreya Sans,sans-serif" font-weight="700" font-size="11" fill="#e8dcc0">${port.kind}</text>`);
        if (port.kind === '2:1') {
            out.push(`<circle cx="${(bx + 14).toFixed(1)}" cy="${(by - 9).toFixed(1)}" r="4.5" fill="${PORT_DOT[port.resource]}" stroke="#17130e" stroke-width="1"/>`);
            out.push(resourceGlyph(port.resource, bx + 14, by - 9));
        }
        out.push(`<rect x="${(bx - 17).toFixed(1)}" y="${(by - 10).toFixed(1)}" width="34" height="20" rx="7" fill="url(#ba-bevel)"/>`);
        out.push(`</g>`);
    }
    return out.join('');
}

/** Everything static, in draw order: defs + sea + tiles + ports. Convenient for
 *  the React board, which injects it as a single innerHTML chunk beneath its
 *  interactive JSX. Vanilla renderers can call the pieces individually instead. */
export function staticBoard(board: Board, L: Layout): string {
    return boardDefs() + seaLayer(L) + tilesLayer(board, L) + portsLayer(board, L);
}

// ─────────────────────────────────────────────────────────────────────────
// Dynamic pieces — settlements, cities, roads, the robber. These move/appear
// during play, so each renderer still owns its own click handling and entry
// animation (CSS `.born` for the vanilla string-SVG clients, framer-motion
// for Board.tsx) — but the actual SHAPE and depth recipe (shadow + bevel
// sheen) is defined exactly once here, requires `boardDefs()` to already be
// present (for `ba-pieceshadow`/`ba-bevel`), and is shared by all three.
// ─────────────────────────────────────────────────────────────────────────

/** A settlement (small pentagon "house") or city (house + an attached cube),
 *  filled in the owner's colour with a drop shadow and glass-sheen bevel. */
export function housePieceMarkup(cx: number, cy: number, kind: 'settlement' | 'city', fill: string): string {
    const s = kind === 'city' ? 11 : 8;
    const body = `M ${cx - s} ${cy + s} L ${cx - s} ${cy - s * 0.2} L ${cx} ${cy - s} L ${cx + s} ${cy - s * 0.2} L ${cx + s} ${cy + s} Z`;
    const cap = kind === 'city' ? { x: cx + 1, y: cy - s * 0.2, w: s, h: s * 1.2 } : null;
    const parts = [`<g filter="url(#ba-pieceshadow)">`];
    parts.push(`<path d="${body}" fill="${fill}" stroke="#1c1813" stroke-width="1.6" stroke-linejoin="round"/>`);
    if (cap) parts.push(`<rect x="${cap.x}" y="${cap.y}" width="${cap.w}" height="${cap.h}" fill="${fill}" stroke="#1c1813" stroke-width="1.6"/>`);
    parts.push(`<path d="${body}" fill="url(#ba-bevel)"/>`);
    if (cap) parts.push(`<rect x="${cap.x}" y="${cap.y}" width="${cap.w}" height="${cap.h}" fill="url(#ba-bevel)"/>`);
    parts.push(`</g>`);
    return parts.join('');
}

/** A road segment between two pixel points: a coloured base plank, a soft
 *  shading pass, and a thin centred highlight line for a "raised wood" feel.
 *
 *  Deliberately NOT wrapped in the shared `ba-pieceshadow` filter (unlike the
 *  other pieces): that filter's region is percentage-based (`objectBoundingBox`,
 *  the SVG default), and a perfectly vertical or horizontal `<line>` has a
 *  ZERO-width or zero-height geometric bounding box — collapsing the filter
 *  region to nothing and silently dropping the whole filtered element in
 *  Chromium. Hex edges genuinely include perfectly vertical ones (two of the
 *  three edge orientations in a pointy-top hex grid), so this isn't an edge
 *  case — it's a third of all roads. The dark overlay line below already
 *  gives an adequate shading pass without needing a real filter. */
export function roadMarkup(pa: { x: number; y: number }, pb: { x: number; y: number }, fill: string): string {
    const x1 = pa.x.toFixed(1), y1 = pa.y.toFixed(1), x2 = pb.x.toFixed(1), y2 = pb.y.toFixed(1);
    return (
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${fill}" stroke-width="7" stroke-linecap="round"/>` +
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#00000026" stroke-width="7" stroke-linecap="round"/>` +
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#ffffff2e" stroke-width="2.2" stroke-linecap="round"/>`
    );
}

/** The robber: an obsidian pawn with its own ground-contact shadow (so it
 *  doesn't get the generic piece drop-shadow — it already grounds itself),
 *  a bevel sheen, and a small glint for "nicer" per the design pass.
 *
 *  The pawn is drawn offset right by SIZE/2 from the logical (cx, cy) so it
 *  doesn't overlap the number-token chip (radius SIZE*0.33) when both sit on
 *  the same tile. Callers always pass the tile centre — no caller changes needed. */
export function robberPawnMarkup(cx: number, cy: number): string {
    const ox = cx + SIZE * 0.5, oy = cy;
    const head = `<circle cx="${ox}" cy="${oy - 7}" r="6.4" fill="#17130e" stroke="#c9a45c77" stroke-width="1.1"/>`;
    const body = `<path d="M ${ox - 9.5} ${oy + 12} Q ${ox} ${oy - 3} ${ox + 9.5} ${oy + 12} Z" fill="#17130e" stroke="#c9a45c77" stroke-width="1.1"/>`;
    return (
        `<ellipse cx="${ox}" cy="${oy + 13}" rx="11" ry="3" fill="#00000055"/>` +
        body + head +
        `<path d="M ${ox - 9.5} ${oy + 12} Q ${ox} ${oy - 3} ${ox + 9.5} ${oy + 12} Z" fill="url(#ba-bevel)"/>` +
        `<circle cx="${ox}" cy="${oy - 7}" r="6.4" fill="url(#ba-bevel)"/>` +
        `<circle cx="${(ox - 2).toFixed(1)}" cy="${(oy - 9.5).toFixed(1)}" r="1.3" fill="#fff8e0aa"/>`
    );
}
