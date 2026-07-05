// Shared interactive board renderer for the hotseat and online React apps.
// The SVG itself is still composed as one markup string from the boardArt
// layer (single source of truth for how the board looks, shared with
// game/Board.tsx and the replay viewer) and injected via
// dangerouslySetInnerHTML; interactivity is one delegated click handler on
// the container that resolves the nearest `.hit` element's data attributes —
// the exact contract the vanilla clients used, now behind a React component.

import { useMemo, useRef } from 'react';
import type { Board, VertexId, EdgeId, TileId, PlayerId } from '@catan/core';
import { SIZE, hexCenter, corner, layoutBoard } from '../game/theme';
import {
    boardDefs, seaLayer, tilesLayer, portsLayer, housePieceMarkup, roadMarkup, robberPawnMarkup, tileProducePulse,
} from '../game/boardArt';

export interface BoardSvgProps {
    board: Board;
    robber: TileId;
    roads: Record<EdgeId, PlayerId>;
    buildings: Record<VertexId, { kind: 'settlement' | 'city'; owner: PlayerId }>;
    colorOf: (p: PlayerId) => string;
    /** Colour used for the legal-target highlights (the acting player's). */
    highlightColor: string;
    vTargets: readonly VertexId[];
    eTargets: readonly EdgeId[];
    tTargets: readonly TileId[];
    /** Tiles to flash "just produced" on, for the most recent dice roll —
     *  see game/boardArt.ts's producingTilesForRoll(). Pass a stable (empty
     *  by default) array reference when there's nothing to flash; a fresh
     *  `[]` literal here would defeat this component's own memoization the
     *  same way an inline vTargets/eTargets array did before (see
     *  HotseatApp's/OnlineApp's EMPTY_TILES constant). */
    producingTiles?: readonly TileId[];
    onVertex: (v: VertexId) => void;
    onEdge: (e: EdgeId) => void;
    onTile: (t: TileId) => void;
}

export default function BoardSvg(props: BoardSvgProps) {
    const { board, robber, roads, buildings, colorOf, highlightColor, vTargets, eTargets, tTargets, producingTiles } = props;

    // Piece keys present at the previous committed render — a piece pops in
    // (`.born`) exactly when it wasn't there last time. Mutated in place
    // during the memo (mirroring the vanilla renderBoard's synchronous
    // seenPieces swap) so that a same-commit recompute can't double-pop.
    const seenPieces = useRef(new Set<string>());

    const html = useMemo(() => {
        const L = layoutBoard(board);
        const ox = L.ox, oy = L.oy;
        const px = (v: VertexId) => L.pos.get(v)!;

        const out: string[] = [`<svg viewBox="0 0 ${L.W.toFixed(1)} ${L.H.toFixed(1)}" xmlns="http://www.w3.org/2000/svg">`];
        out.push(boardDefs(), seaLayer(L), tilesLayer(board, L));

        if (producingTiles?.length) out.push(tileProducePulse(board, L, producingTiles));

        // Robber: obsidian pawn on its tile, above the terrain.
        const rt = board.tiles[robber];
        const rc = hexCenter(rt.coord.q, rt.coord.r);
        out.push(robberPawnMarkup(rc.x * SIZE + ox, rc.y * SIZE + oy));

        out.push(portsLayer(board, L));

        // Roads + buildings, with a "born" pop for pieces new since last render
        // (a city upgrade changes the key, so it re-pops too).
        const currentPieces = new Set<string>();
        for (const [edge, owner] of Object.entries(roads) as [EdgeId, PlayerId][]) {
            const key = `r:${edge}`;
            currentPieces.add(key);
            const born = !seenPieces.current.has(key) ? ' class="born"' : '';
            const [a, b] = board.edges[edge].vertices;
            out.push(`<g${born}>${roadMarkup(px(a), px(b), colorOf(owner))}</g>`);
        }
        for (const [v, b] of Object.entries(buildings) as [VertexId, { kind: 'settlement' | 'city'; owner: PlayerId }][]) {
            const key = `b:${v}:${b.kind}`;
            currentPieces.add(key);
            const born = !seenPieces.current.has(key) ? ' class="born"' : '';
            const p = px(v);
            out.push(`<g${born}>${housePieceMarkup(p.x, p.y, b.kind, colorOf(b.owner))}</g>`);
        }
        seenPieces.current = currentPieces;

        // Interactive highlights (legal targets for the current phase/mode).
        for (const e of eTargets) {
            const [a, b] = board.edges[e].vertices;
            const pa = px(a), pb = px(b);
            out.push(`<g class="hit" data-kind="edge" data-id="${e}">` +
                `<line class="marching" x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}" x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}" stroke="${highlightColor}" stroke-width="7" stroke-linecap="round" stroke-dasharray="2 6" opacity="0.9"/>` +
                `<line x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}" x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}" stroke="transparent" stroke-width="16" stroke-linecap="round"/>` +
                `</g>`);
        }
        for (const v of vTargets) {
            const p = px(v);
            out.push(`<g class="hit" data-kind="vertex" data-id="${v}">` +
                // Invisible, larger-than-visible tap target — the r=9 pulse
                // circles below are the right VISUAL size, but a bare 18px
                // hit area is under most touch-target guidelines (~44px).
                // Matches the edges' visible-thin-line + fat-invisible-line
                // pattern just above.
                `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="18" fill="transparent"/>` +
                `<circle class="pulse-ring" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="9" fill="none" stroke="${highlightColor}" stroke-width="2"/>` +
                `<circle class="pulse" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="9" fill="#ffd98ccc" stroke="${highlightColor}" stroke-width="2.5"/>` +
                `</g>`);
        }
        for (const t of tTargets) {
            const c = hexCenter(board.tiles[t].coord.q, board.tiles[t].coord.r);
            const pts = Array.from({ length: 6 }, (_, i) => {
                const pp = corner(c.x, c.y, i);
                return `${(pp.x * SIZE + ox).toFixed(1)},${(pp.y * SIZE + oy).toFixed(1)}`;
            }).join(' ');
            out.push(`<polygon class="hit pulse-march" data-kind="tile" data-id="${t}" points="${pts}" fill="#ffd98c22" stroke="#ffd98c" stroke-width="2.5" stroke-dasharray="5 5"/>`);
        }

        out.push('</svg>');
        return out.join('');
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [board, robber, roads, buildings, colorOf, highlightColor, vTargets, eTargets, tTargets, producingTiles]);

    const onClick = (ev: React.MouseEvent) => {
        const hit = (ev.target as Element).closest?.('.hit') as SVGElement | null;
        if (!hit) return;
        const kind = hit.dataset.kind, id = hit.dataset.id!;
        if (kind === 'vertex') props.onVertex(id as VertexId);
        else if (kind === 'edge') props.onEdge(id as EdgeId);
        else if (kind === 'tile') props.onTile(id as TileId);
    };

    return <div id="board" onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />;
}
