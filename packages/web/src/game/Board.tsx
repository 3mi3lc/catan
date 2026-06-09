import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { boardPorts, type GameState, type VertexId, type EdgeId, type TileId, type PlayerId } from '@catan/core';
import { SIZE, hexCenter, corner, pips, layoutBoard, TERRAIN, RES_COLOR, colorOf } from './theme';

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
    const s = kind === 'city' ? 11 : 8;
    const body = `M ${x - s} ${y + s} L ${x - s} ${y - s * 0.2} L ${x} ${y - s} L ${x + s} ${y - s * 0.2} L ${x + s} ${y + s} Z`;
    return (
        <motion.g initial={{ opacity: 0, y: -7 }} animate={{ opacity: 1, y: 0 }} transition={{ type: 'spring', stiffness: 360, damping: 22 }}>
            <path d={body} fill={fill} stroke="#1c1813" strokeWidth={1.6} strokeLinejoin="round" />
            {kind === 'city' && <rect x={x + 1} y={y - s * 0.2} width={s} height={s * 1.2} fill={fill} stroke="#1c1813" strokeWidth={1.6} />}
        </motion.g>
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
            <defs>
                <radialGradient id="sea" cx="50%" cy="42%" r="75%">
                    <stop offset="0%" stopColor="#d4e9ea" />
                    <stop offset="70%" stopColor="#a9cdd4" />
                    <stop offset="100%" stopColor="#84b0ba" />
                </radialGradient>
                <filter id="tileshadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="0" dy="2.5" stdDeviation="3" floodColor="#1c2e33" floodOpacity="0.3" />
                </filter>
            </defs>

            <rect x="0" y="0" width={W} height={H} rx={22} fill="url(#sea)" />

            {/* Tiles, number tokens */}
            <g filter="url(#tileshadow)">
                {Object.values(board.tiles).map((t) => {
                    const c = hexCenter(t.coord.q, t.coord.r);
                    const cx = c.x * SIZE + ox, cy = c.y * SIZE + oy;
                    const T = TERRAIN[t.terrain];
                    const red = t.numberToken === 6 || t.numberToken === 8;
                    const n = t.numberToken ? pips(t.numberToken) : 0;
                    const gap = 5, total = (n - 1) * gap;
                    return (
                        <g key={t.id}>
                            <polygon points={hexPoints(t.coord.q, t.coord.r, ox, oy)} fill={T.fill} stroke={T.stroke} strokeWidth={2} strokeLinejoin="round" />
                            {t.numberToken !== null && (
                                <>
                                    <circle cx={cx} cy={cy} r={SIZE * 0.32} fill="#f4ecd6" stroke="#bda673" strokeWidth={1.5} />
                                    <text x={cx} y={cy - 2} textAnchor="middle" dominantBaseline="central" fontFamily="Fraunces, serif" fontWeight={600} fontSize={red ? 21 : 18} fill={red ? '#b23a2e' : '#34302a'}>{t.numberToken}</text>
                                    {Array.from({ length: n }, (_, i) => (
                                        <circle key={i} cx={cx - total / 2 + i * gap} cy={cy + SIZE * 0.19} r={1.6} fill={red ? '#b23a2e' : '#6b6358'} />
                                    ))}
                                </>
                            )}
                        </g>
                    );
                })}
            </g>

            {/* Harbours */}
            {boardPorts(board).map(({ port, vertices }, i) => {
                const a = px(vertices[0]), b = px(vertices[1]);
                const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
                let dx = mx - ox, dy = my - oy; const len = Math.hypot(dx, dy) || 1; dx /= len; dy /= len;
                const bx = mx + dx * 20, by = my + dy * 20;
                return (
                    <g key={i}>
                        <line x1={a.x} y1={a.y} x2={bx} y2={by} stroke="#5f7c82" strokeWidth={1.4} strokeDasharray="2 2" />
                        <line x1={b.x} y1={b.y} x2={bx} y2={by} stroke="#5f7c82" strokeWidth={1.4} strokeDasharray="2 2" />
                        <rect x={bx - 17} y={by - 10} width={34} height={20} rx={6} fill="#f6efe0" stroke="#2c2620" strokeWidth={1.2} />
                        <text x={bx} y={by} textAnchor="middle" dominantBaseline="central" fontFamily="Spline Sans, sans-serif" fontWeight={600} fontSize={11} fill="#2c2620">{port.kind}</text>
                        {port.kind === '2:1' && <circle cx={bx + 14} cy={by - 9} r={4.5} fill={RES_COLOR[port.resource]} stroke="#2c2620" strokeWidth={1} />}
                    </g>
                );
            })}

            {/* Roads */}
            {(Object.entries(game.roads) as [EdgeId, PlayerId][]).map(([edge, owner]) => {
                const [a, b] = board.edges[edge].vertices;
                const pa = px(a), pb = px(b);
                return (
                    <motion.line key={edge} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}
                                 x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke={colorFn(owner)} strokeWidth={7} strokeLinecap="round" />
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
            >
                <circle cx={0} cy={-6} r={6} fill="#2a2620" />
                <path d="M -9 12 Q 0 -2 9 12 Z" fill="#2a2620" />
            </motion.g>

            {/* Legal-target highlights */}
            {[...legalEdges].map((e) => {
                const [a, b] = board.edges[e].vertices;
                const pa = px(a), pb = px(b);
                return (
                    <g key={`he-${e}`} className="hit" onClick={() => onEdge(e)}>
                        <line x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke={cur} strokeWidth={7} strokeLinecap="round" strokeDasharray="2 6" opacity={0.9} />
                        <line x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke="transparent" strokeWidth={16} strokeLinecap="round" />
                    </g>
                );
            })}
            {[...legalVertices].map((v) => {
                const p = px(v);
                return <circle key={`hv-${v}`} className="hit pulse" onClick={() => onVertex(v)} cx={p.x} cy={p.y} r={9} fill="#ffffffcc" stroke={cur} strokeWidth={2.5} />;
            })}
            {[...legalTiles].map((t) => (
                <polygon key={`ht-${t}`} className="hit" onClick={() => onTile(t)}
                         points={hexPoints(board.tiles[t].coord.q, board.tiles[t].coord.r, ox, oy)}
                         fill="#1c181333" stroke="#1c1813" strokeWidth={3} strokeDasharray="4 4" />
            ))}
        </svg>
    );
}