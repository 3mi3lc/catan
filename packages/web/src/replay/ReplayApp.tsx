import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import Board from '../game/Board';
import { PLAYER, RES_COLOR, RES_ABBR, colorOf } from '../game/theme';
import '../game/styles.css';   // base layout, cards, buttons, h2, pcard, hand, etc.
import './replay.css';          // replay-only additions
import { loadReplay, type GameArchive, type Replay } from '@catan/ai';
import {
    RESOURCES, victoryPoints, longestRoadLength,
    type GameState, type PlayerId, type Resource, type Action,
} from '@catan/core';

const POLICY_PALETTE = [PLAYER.red.fill, PLAYER.blue.fill, PLAYER.orange.fill, PLAYER.white.fill];

const DEV_ABBR: Record<string, string> = {
    knight: '⚔ Knight', victoryPoint: '★ VP', roadBuilding: '🛣 Road Bldg',
    yearOfPlenty: '🌾 Year+', monopoly: '💰 Monopoly',
};

const shortId = (id: unknown) => String(id).replace(/^[A-Za-z]+:/, '').slice(0, 12);

function diceTotal(d: unknown): number | null {
    if (d == null) return null;
    if (Array.isArray(d)) return d.reduce((a, b) => a + Number(b), 0);
    if (typeof d === 'object') {
        const o = d as Record<string, number>;
        if ('a' in o && 'b' in o) return o.a + o.b;
        if ('total' in o) return o.total;
    }
    return typeof d === 'number' ? d : null;
}

function bundleStr(res: Partial<Record<Resource, number>>): string {
    const parts = Object.entries(res).filter(([, n]) => (n ?? 0) > 0).map(([r, n]) => `${n} ${RES_ABBR[r as Resource]}`);
    return parts.length ? parts.join(', ') : 'nothing';
}

function describeAction(a: Action, resultState: GameState | null): string {
    switch (a.type) {
        case 'rollDice': { const t = resultState ? diceTotal(resultState.dice) : null; return t != null ? `rolled ${t}` : 'rolled the dice'; }
        case 'buildSettlement': return 'built a settlement';
        case 'buildCity':       return 'upgraded to a city';
        case 'buildRoad':       return 'built a road';
        case 'buyDevCard':      return 'bought a development card';
        case 'playKnight':      return `played Knight${a.stealFrom ? `, robbing ${a.stealFrom}` : ''}`;
        case 'playRoadBuilding':return 'played Road Building';
        case 'playYearOfPlenty':return `played Year of Plenty (${a.take.map(r => RES_ABBR[r]).join(' + ')})`;
        case 'playMonopoly':    return `played Monopoly on ${RES_ABBR[a.resource]}`;
        case 'moveRobber':      return `moved the robber${a.stealFrom ? `, robbing ${a.stealFrom}` : ''}`;
        case 'discard':         return `discarded ${bundleStr(a.resources)}`;
        case 'bankTrade':       return `bank trade — ${a.giveCount}× ${RES_ABBR[a.give]} → ${RES_ABBR[a.receive]}`;
        case 'proposeTrade':    return `proposed trade to ${a.to}`;
        case 'respondToTrade':  return `${a.accept ? 'accepted' : 'declined'} trade`;
        case 'endTurn':         return 'ended turn';
        default:                return (a as { type: string }).type;
    }
}

function actionLocation(a: Action): string | null {
    switch (a.type) {
        case 'buildSettlement': case 'buildCity': return shortId(a.vertex);
        case 'buildRoad':        return shortId(a.edge);
        case 'moveRobber':       return shortId(a.tile);
        case 'playKnight':       return shortId(a.robberTo);
        default:                 return null;
    }
}

function devCounts(devCards: string[], pending: string[]): Record<string, number> {
    const c: Record<string, number> = {};
    for (const d of [...devCards, ...pending]) c[d] = (c[d] ?? 0) + 1;
    return c;
}

function computeGains(before: GameState, after: GameState, players: readonly PlayerId[]): Array<{ pid: PlayerId; gains: Partial<Record<Resource, number>> }> {
    return players.map(pid => {
        const gains: Partial<Record<Resource, number>> = {};
        for (const r of RESOURCES) {
            const delta = (after.players[pid]?.resources[r] ?? 0) - (before.players[pid]?.resources[r] ?? 0);
            if (delta > 0) gains[r] = delta;
        }
        return { pid, gains };
    }).filter(({ gains }) => Object.keys(gains).length > 0);
}

export default function ReplayApp() {
    const [archive, setArchive] = useState<GameArchive | null>(null);
    const [idx, setIdx]         = useState(0);
    const [error, setError]     = useState<string | null>(null);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed]     = useState(300);

    const replay: Replay | null = useMemo(() => {
        if (!archive) return null;
        try { return loadReplay(archive); }
        catch (e) { setError(String(e)); return null; }
    }, [archive]);

    // Stable policy-name → colour mapping (alphabetical so it's the same across all archives).
    const policyColor = useMemo<((id: PlayerId) => string) | undefined>(() => {
        if (!archive) return undefined;
        const names = archive.policyNames;
        const unique = [...new Set(names)];
        if (unique.length !== names.length) return undefined;
        const byName = new Map(unique.slice().sort().map((n, i) => [n, POLICY_PALETTE[i % POLICY_PALETTE.length]]));
        const bySeat = new Map(names.map((n, seat) => [`p${seat}`, byName.get(n)!]));
        return (id: PlayerId) => bySeat.get(id) ?? POLICY_PALETTE[Number(String(id).replace(/^p/, '')) % POLICY_PALETTE.length];
    }, [archive]);

    function loadFile(e: React.ChangeEvent<HTMLInputElement>) {
        setError(null); setPlaying(false);
        const file = e.target.files?.[0];
        if (!file) return;
        file.text().then(t => {
            try {
                const parsed = JSON.parse(t) as GameArchive;
                if (Array.isArray(parsed)) { setError('Old snapshot format — re-run benchmark to get a new archive.'); return; }
                setArchive(parsed); setIdx(0);
            } catch { setError('Failed to parse JSON.'); }
        });
    }

    useEffect(() => {
        if (!playing || !replay) return;
        if (idx >= replay.length - 1) { setPlaying(false); return; }
        const t = setTimeout(() => setIdx(i => i + 1), speed);
        return () => clearTimeout(t);
    }, [playing, idx, replay, speed]);

    const step = useCallback((delta: number) => {
        setPlaying(false);
        setIdx(i => replay ? Math.max(0, Math.min(i + delta, replay.length - 1)) : 0);
    }, [replay]);

    useEffect(() => {
        const h = (e: KeyboardEvent) => {
            if (!replay) return;
            if      (e.key === 'ArrowRight') step(1);
            else if (e.key === 'ArrowLeft')  step(-1);
            else if (e.key === ' ')          { e.preventDefault(); setPlaying(p => !p); }
        };
        window.addEventListener('keydown', h);
        return () => window.removeEventListener('keydown', h);
    }, [replay, step]);

    const activeRowRef = useRef<HTMLButtonElement | null>(null);
    useEffect(() => { activeRowRef.current?.scrollIntoView({ block: 'nearest' }); }, [idx]);

    const state         = replay?.at(idx) ?? null;
    const currentMoveIdx = idx - 1;

    function colorOfPid(s: GameState, pid: string): string {
        return policyColor?.(pid as PlayerId) ?? colorOf(s, pid as PlayerId);
    }
    function pidLabel(pid: string): string {
        if (!archive) return pid;
        const seat = Number(pid.replace(/^p/, ''));
        return archive.policyNames[seat] ?? pid;
    }
    function safeLR(s: GameState, pid: PlayerId): number {
        try { return longestRoadLength(s, pid); } catch { return 0; }
    }

    return (
        <div className="page">
            {/* ── Header ───────────────────────────────────────────── */}
            <header className="topbar">
                <div className="brand">
                    <span className="logo">⬡</span>
                    <div><h1>Catan</h1><div className="tagline">replay viewer</div></div>
                </div>

                {archive && (
                    <div className="rp-legend">
                        {archive.policyNames.map((n, seat) => (
                            <span key={seat}>
                                <span className="rp-swatch" style={{ background: policyColor ? policyColor(`p${seat}` as PlayerId) : POLICY_PALETTE[seat] }} />
                                {n} <span style={{ color: 'var(--muted)', fontSize: 11 }}>(seat {seat})</span>
                            </span>
                        ))}
                    </div>
                )}

                <div className="newgame" style={{ gap: 10 }}>
                    {archive && <a className="link" href="/game.html">game →</a>}
                    <input type="file" accept=".json" onChange={loadFile} />
                </div>
            </header>

            {error && <div className="rp-error">{error}</div>}

            {/* ── Three-column layout (same grid as game.html) ─────── */}
            <div className="layout">

                {/* LEFT — playback controls + current state + bank */}
                <section className="controls card">
                    {!archive ? (
                        <div>
                            <h2>Replay viewer</h2>
                            <p className="hint">Load a <code>replays/game-*.json</code> file using the button above.</p>
                            <p className="hint small">Arrow keys step · space toggles play.</p>
                        </div>
                    ) : (<>
                        <h2>Playback</h2>

                        {/* Transport buttons */}
                        <div className="row" style={{ marginTop: 0 }}>
                            <button className="ghost rp-btn" title="Start" onClick={() => { setPlaying(false); setIdx(0); }}>⏮</button>
                            <button className="ghost rp-btn" title="Back"  disabled={idx === 0} onClick={() => step(-1)}>◀</button>
                            <button title={playing ? 'Pause' : 'Play'} onClick={() => setPlaying(p => !p)}>
                                {playing ? '⏸' : '▶'} {playing ? 'Pause' : 'Play'}
                            </button>
                            <button className="ghost rp-btn" title="Forward" disabled={!replay || idx >= replay.length - 1} onClick={() => step(1)}>▶</button>
                            <button className="ghost rp-btn" title="End" onClick={() => { setPlaying(false); if (replay) setIdx(replay.length - 1); }}>⏭</button>
                        </div>

                        {/* Scrubber */}
                        <input className="rp-scrub" type="range" min={0} max={replay ? replay.length - 1 : 0} value={idx}
                               onChange={e => { setPlaying(false); setIdx(Number(e.target.value)); }} />

                        {/* Speed */}
                        <div className="rp-speed">
                            <span className="hint">Speed</span>
                            <input type="range" min={50} max={800} step={50}
                                   value={850 - speed} onChange={e => setSpeed(850 - Number(e.target.value))} />
                            <span className="hint">{speed}ms</span>
                        </div>

                        {state && replay && archive && (<>
                            <hr className="rp-sep" />

                            {/* Step + phase + player + dice */}
                            <div className="hint small">Step {idx} / {replay.length - 1}</div>
                            <div className="turnline" style={{ marginTop: 6 }}>
                                <span className="dot" style={{ background: colorOfPid(state, state.currentPlayer) }} />
                                <b>{pidLabel(state.currentPlayer)}</b>
                                <span className="rp-phase">{state.phase}</span>
                                {diceTotal(state.dice) != null && <span className="dice">🎲 {diceTotal(state.dice)}</span>}
                            </div>

                            {/* Last action */}
                            <div className="rp-action">
                                {currentMoveIdx >= 0
                                    ? describeAction(archive.moves[currentMoveIdx].action, state)
                                    : 'initial board'}
                            </div>
                            <div className="hint small" style={{ marginTop: 4 }}>
                                robber · {shortId(state.robber)}
                                {state.longestRoad?.player && <> · LR: {pidLabel(state.longestRoad.player)}</>}
                                {state.largestArmy?.player && <> · LA: {pidLabel(state.largestArmy.player)}</>}
                            </div>
                            {state.winner && <div className="rp-winner">👑 {pidLabel(state.winner)} wins!</div>}

                            <hr className="rp-sep" />

                            {/* Bank */}
                            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 7 }}>Bank</div>
                            <div className="hand">
                                {RESOURCES.map(r => (
                                    <span key={r} className="res"
                                          style={{ ['--c' as string]: RES_COLOR[r] }}>
                                        {RES_ABBR[r]}<b>{state.bank[r]}</b>
                                    </span>
                                ))}
                            </div>
                            <div className="hint small" style={{ marginTop: 5 }}>dev deck: {state.devDeck.length} left</div>

                            <hr className="rp-sep" />

                            {/* Archive metadata */}
                            <div className="hint small">seed {archive.seed} · {archive.turns} turns · {archive.moves.length} moves</div>
                        </>)}
                    </>)}
                </section>

                {/* CENTER — board fills this panel exactly like the game */}
                <section className="board-panel">
                    {state && replay ? (
                        <div className="board-wrap">
                            <Board
                                game={state}
                                legalVertices={new Set()} legalEdges={new Set()} legalTiles={new Set()}
                                onVertex={noop} onEdge={noop} onTile={noop}
                                colorFor={policyColor}
                            />
                        </div>
                    ) : (
                        <div className="rp-empty-board">
                            <span style={{ fontSize: 42 }}>⬡</span>
                            <p>Load a replay to see the board</p>
                        </div>
                    )}
                </section>

                {/* RIGHT — players + log */}
                <aside className="side">
                    {state && archive && replay ? (<>
                        {/* Players */}
                        <div className="card">
                            <h2>Players</h2>
                            {(state.turnOrder as PlayerId[]).map(pid => {
                                const pl       = state.players[pid];
                                const seat     = Number(String(pid).replace(/^p/, ''));
                                const color    = colorOfPid(state, pid);
                                const vp       = victoryPoints(state, pid);
                                const isCur    = state.currentPlayer === pid;
                                const isWinner = state.winner === pid;
                                const isLR     = state.longestRoad?.player === pid;
                                const isLA     = state.largestArmy?.player === pid;
                                let setts = 0, cities = 0;
                                for (const b of Object.values(state.buildings)) {
                                    if (b.owner !== pid) continue;
                                    if (b.kind === 'city') cities++; else setts++;
                                }
                                const roads = Object.values(state.roads).filter(o => o === pid).length;
                                const lrLen = safeLR(state, pid);
                                const devs  = devCounts(pl.devCards as unknown as string[], pl.pendingDevCards as unknown as string[]);

                                return (
                                    <div key={pid} className={`pcard${isCur ? ' cur' : ''}`}
                                         style={{ ['--pc' as string]: color }}>
                                        <div className="phead">
                                            <span className="dot" style={{ background: color }} />
                                            <span className="pname">{archive!.policyNames[seat]}</span>
                                            {isCur    && <span className="turntag">turn</span>}
                                            {isWinner && <span className="badge">👑</span>}
                                            {isLR     && <span className="badge">LR</span>}
                                            {isLA     && <span className="badge">LA</span>}
                                            <span className="vp">{vp} <small>VP</small></span>
                                        </div>
                                        <div className="hand">
                                            {RESOURCES.map(r => (
                                                <span key={r} className="res" style={{ ['--c' as string]: RES_COLOR[r] }}>
                                                    {RES_ABBR[r]}<b>{pl.resources[r]}</b>
                                                </span>
                                            ))}
                                        </div>
                                        <div className="meta">
                                            🏠 {setts} · 🏙 {cities} · 🛣 {roads} (len {lrLen}) · ⚔ {pl.playedKnights}
                                        </div>
                                        {Object.keys(devs).length > 0 && (
                                            <div className="meta">
                                                {Object.entries(devs).map(([k, n]) => (
                                                    <span key={k} className="badge">{DEV_ABBR[k] ?? k} ×{n}</span>
                                                ))}
                                                {pl.pendingDevCards.length > 0 && <span>(+{pl.pendingDevCards.length} pending)</span>}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Action log */}
                        <div className="card log-card">
                            <h2>Action log <span className="rp-log-meta">({archive.moves.length} moves)</span></h2>
                            <div className="movelog">
                                {archive.moves.map((m, j) => {
                                    const prev   = replay.at(j);
                                    const result = replay.at(j + 1) ?? null;
                                    const isActive = j === currentMoveIdx;
                                    const newTurn  = j === 0 || archive.moves[j - 1].player !== m.player;
                                    const loc  = actionLocation(m.action);
                                    const gains = m.action.type === 'rollDice' && prev && result
                                        ? computeGains(prev, result, state.turnOrder as PlayerId[])
                                        : null;
                                    return (
                                        <button key={j}
                                                ref={isActive ? activeRowRef : undefined}
                                                className={`move${isActive ? ' active' : ''}${newTurn ? ' newturn' : ''}`}
                                                onClick={() => { setPlaying(false); setIdx(j + 1); }}>
                                            <span className="idx">{j + 1}</span>
                                            <span className="pdot" style={{ background: colorOfPid(result ?? state, m.player) }} />
                                            <span className="ptag">{m.phase}</span>
                                            <span className="desc">
                                                {describeAction(m.action, result)}
                                                {gains && gains.length > 0 && (
                                                    <span className="dice-gains">
                                                        {gains.map(({ pid, gains: g }) => (
                                                            <span key={pid} className="gain-entry">
                                                                <span className="tiny-dot" style={{ background: colorOfPid(result ?? state, pid) }} />
                                                                {RESOURCES.filter(r => g[r]).map(r => (
                                                                    <span key={r} className="gain-chip"
                                                                          style={{ background: RES_COLOR[r] + '28', borderColor: RES_COLOR[r] }}>
                                                                        +{g[r]}{RES_ABBR[r]}
                                                                    </span>
                                                                ))}
                                                            </span>
                                                        ))}
                                                    </span>
                                                )}
                                                {gains && gains.length === 0 && <span style={{ color: 'var(--muted)' }}> — no production</span>}
                                            </span>
                                            {loc && <span className="loc">{loc}</span>}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    </>) : null}
                </aside>
            </div>
        </div>
    );
}

const noop = () => {};