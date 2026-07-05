import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { type GameState, type VertexId, type EdgeId, type TileId, type PlayerId } from '@catan/core';
import { SIZE, hexCenter, corner, layoutBoard, colorOf } from './theme';
import { staticBoard, housePieceMarkup, roadMarkup, robberPawnMarkup } from './boardArt';

interface Props {
    game: GameState;
    legalVertices: Set<VertexId>;
    legalEdges: Set<EdgeId>;
    legalTiles: Set<TileId>;
    onVertex: (v: VertexId) => void;
    onEdge: (e: EdgeId) => void;
    onTile: (t: TileId) => void;
    /** Override piece colour per player (e.g. colour by policy in replays). Defaults to seat colour. */
    colorFor?: (id: PlayerId) => string;
}

const hexPoints = (q: number, r: number, ox: number, oy: number): string => {
    const c = hexCenter(q, r);
    return Array.from({ length: 6 }, (_, i) => {
        const p = corner(c.x, c.y, i);
        return `${(p.x * SIZE + ox).toFixed(1)},${(p.y * SIZE + oy).toFixed(1)}`;
    }).join(' ');
};

function House({ x, y, kind, fill }: { x: number; y: number; kind: 'settlement' | 'city'; fill: string }) {
    return (
        <motion.g
            initial={{ opacity: 0, y: -7 }} animate={{ opacity: 1, y: 0 }} transition={{ type: 'spring', stiffness: 360, damping: 22 }}
            dangerouslySetInnerHTML={{ __html: housePieceMarkup(x, y, kind, fill) }}
        />
    );
}

export default function Board({ game, legalVertices, legalEdges, legalTiles, onVertex, onEdge, onTile, colorFor }: Props) {
    const board = game.board;
    const layout = useMemo(() => layoutBoard(board), [board]);
    const { pos, ox, oy, W, H } = layout;
    const px = (v: VertexId) => pos.get(v)!;
    const colorFn = colorFor ?? ((id: PlayerId) => colorOf(game, id));
    const cur = colorFn(game.currentPlayer);
    const robberTile = board.tiles[game.robber];
    const robberC = hexCenter(robberTile.coord.q, robberTile.coord.r);
    const rx = robberC.x * SIZE + ox, ry = robberC.y * SIZE + oy;

    return (
        <svg viewBox={`0 0 ${W.toFixed(1)} ${H.toFixed(1)}`} className="board-svg" xmlns="http://www.w3.org/2000/svg">
            {/* Static art (sea + terrain + number tokens + harbours) comes from
                the shared board-art layer — one definition across every renderer.
                Injected as a single SVG chunk beneath the interactive JSX below. */}
            <g dangerouslySetInnerHTML={{ __html: staticBoard(board, layout) }} />


            {/* Roads */}
            {(Object.entries(game.roads) as [EdgeId, PlayerId][]).map(([edge, owner]) => {
                const [a, b] = board.edges[edge].vertices;
                const pa = px(a), pb = px(b);
                return (
                    <motion.g key={edge} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}
                              dangerouslySetInnerHTML={{ __html: roadMarkup(pa, pb, colorFn(owner)) }} />
                );
            })}

            {/* Buildings */}
            {(Object.entries(game.buildings) as [VertexId, { kind: 'settlement' | 'city'; owner: PlayerId }][]).map(([v, b]) => {
                const p = px(v);
                return <House key={v} x={p.x} y={p.y} kind={b.kind} fill={colorFn(b.owner)} />;
            })}

            {/* Robber — snaps to desert on mount, springs to new tile when moved */}
            <motion.g
                initial={false}
                animate={{ x: rx, y: ry }}
                transition={{ type: 'spring', stiffness: 200, damping: 22 }}
                dangerouslySetInnerHTML={{ __html: robberPawnMarkup(0, 0) }}
            />

            {/* Legal-target highlights */}
            {[...legalEdges].map((e) => {
                const [a, b] = board.edges[e].vertices;
                const pa = px(a), pb = px(b);
                return (
                    <g key={`he-${e}`} className="hit" onClick={() => onEdge(e)}>
                        <line className="marching" x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke={cur} strokeWidth={7} strokeLinecap="round" strokeDasharray="2 6" opacity={0.9} />
                        <line x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke="transparent" strokeWidth={16} strokeLinecap="round" />
                    </g>
                );
            })}
            {[...legalVertices].map((v) => {
                const p = px(v);
                return (
                    <g key={`hv-${v}`} className="hit" onClick={() => onVertex(v)}>
                        <circle className="pulse-ring" cx={p.x} cy={p.y} r={9} fill="none" stroke={cur} strokeWidth={2} />
                        <circle className="pulse" cx={p.x} cy={p.y} r={9} fill="#ffffffcc" stroke={cur} strokeWidth={2.5} />
                    </g>
                );
            })}
            {[...legalTiles].map((t) => (
                <polygon key={`ht-${t}`} className="hit pulse" onClick={() => onTile(t)}
                         points={hexPoints(board.tiles[t].coord.q, board.tiles[t].coord.r, ox, oy)}
                         fill="#1c181333" stroke="#1c1813" strokeWidth={3} strokeDasharray="4 4" />
            ))}
        </svg>
    );
}