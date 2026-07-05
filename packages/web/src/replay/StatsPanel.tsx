import { useMemo, useState, type ReactNode } from 'react';
import { RES_COLOR } from '../game/theme';
import {
    computeGameStats, RESOURCES,
    type GameArchive, type PlayerId, type Resource, type ResourceBundle, type ResourceFlowCategory,
    type PlayerStats,
} from '@catan/core';

// Compact 2-letter labels for the dense matrices (theme's RES_ABBR is full
// words like "Lumber", too long for table headers).
const SHORT: Record<Resource, string> = { brick: 'Br', lumber: 'Lu', wool: 'Wo', grain: 'Gr', ore: 'Or' };

const CATEGORY_LABEL: Partial<Record<ResourceFlowCategory, string>> = {
    production: 'Production', bankTrade: 'Bank trade', playerTrade: 'Player trade',
    robberSteal: 'Robber', monopoly: 'Monopoly', yearOfPlenty: 'Year of Plenty', discarded: 'Discarded',
};
const CATEGORY_ORDER: ResourceFlowCategory[] = [
    'production', 'bankTrade', 'playerTrade', 'robberSteal', 'monopoly', 'yearOfPlenty', 'discarded',
];

const DEV_TYPES = [
    { key: 'knight', label: '⚔ Knight' },
    { key: 'victoryPoint', label: '★ VP' },
    { key: 'roadBuilding', label: '🛣 Road' },
    { key: 'yearOfPlenty', label: '🌾 Year+' },
    { key: 'monopoly', label: '💰 Mono' },
] as const;

const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'resources', label: 'Resources' },
    { id: 'dice', label: 'Dice' },
    { id: 'cards', label: 'Cards' },
    { id: 'activity', label: 'Activity' },
] as const;
type TabId = (typeof TABS)[number]['id'];

function bundleTotal(b: ResourceBundle): number {
    return Object.values(b).reduce((s, n) => s + (n ?? 0), 0);
}
function netOf(gained: ResourceBundle, lost: ResourceBundle, r: Resource): number {
    return (gained[r] ?? 0) - (lost[r] ?? 0);
}
function signed(n: number): string {
    return n > 0 ? `+${n}` : n < 0 ? `−${-n}` : '·';
}
function cellClass(n: number): string {
    return n > 0 ? 'pos' : n < 0 ? 'neg' : 'zero';
}

function Chips({ bundle }: { bundle: ResourceBundle }) {
    const entries = (Object.keys(bundle) as Resource[]).filter((r) => (bundle[r] ?? 0) > 0);
    if (entries.length === 0) return <span className="muted">—</span>;
    return (
        <span className="chips">
            {entries.map((r) => (
                <span key={r} className="chip" style={{ borderColor: RES_COLOR[r], color: RES_COLOR[r] }}>
                    <span className="cdot" style={{ background: RES_COLOR[r] }} />{bundle[r]} {SHORT[r]}
                </span>
            ))}
        </span>
    );
}

// ── Generic comparison table: one row per player, leader per column marked ──
interface Col {
    label: string; title?: string;
    value: (p: PlayerStats) => number;
    render?: (p: PlayerStats) => ReactNode;
    highlight?: boolean;
}
function StatTable({ players, colorFor, cols }: { players: PlayerStats[]; colorFor: (id: PlayerId) => string; cols: Col[] }) {
    const maxes = cols.map((c) => (c.highlight ? Math.max(0, ...players.map(c.value)) : -Infinity));
    return (
        <table className="stat-table">
            <thead>
                <tr>
                    <th className="st-player" />
                    {cols.map((c) => <th key={c.label} title={c.title}>{c.label}</th>)}
                </tr>
            </thead>
            <tbody>
                {players.map((p) => (
                    <tr key={p.player}>
                        <td className="st-player">
                            <span className="dot" style={{ background: colorFor(p.player) }} />{p.name}
                        </td>
                        {cols.map((c) => {
                            const v = c.value(p);
                            const lead = c.highlight && v === maxes[cols.indexOf(c)] && v > 0;
                            return (
                                <td key={c.label} className={lead ? 'lead' : ''}>
                                    {c.render ? c.render(p) : v}
                                </td>
                            );
                        })}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

interface Props {
    archive: GameArchive;
    policyColor?: (id: PlayerId) => string;
    onWatchReplay: () => void;
}

const FALLBACK_PALETTE = ['#c0392b', '#2f6aa8', '#e9e5d8', '#d98324'];

export default function StatsPanel({ archive, policyColor, onWatchReplay }: Props) {
    const stats = useMemo(() => computeGameStats(archive), [archive]);
    const [tab, setTab] = useState<TabId>('overview');

    const nameByPid = useMemo(
        () => Object.fromEntries(stats.players.map((p) => [p.player, p.name])) as Record<PlayerId, string>,
        [stats],
    );
    const colorFor = (pid: PlayerId): string =>
        policyColor?.(pid) ?? FALLBACK_PALETTE[Number(String(pid).replace(/^p/, '')) % FALLBACK_PALETTE.length];

    const ranked = useMemo(() => [...stats.players].sort((a, b) => b.finalVP - a.finalVP), [stats]);
    const maxRollCount = Math.max(1, ...Object.values(stats.dice.rollCounts));

    return (
        <div className="stats-layout">
            {/* ── Summary band (always visible) ──────────────────────── */}
            <div className="card stats-summary">
                <div className="stats-stat"><b>{stats.turns}</b><span>turns</span></div>
                <div className="stats-stat"><b>{stats.totalMoves}</b><span>moves</span></div>
                <div className="stats-stat"><b>{stats.dice.totalRolls}</b><span>dice rolls</span></div>
                {stats.winner && <div className="winner-banner">👑 {nameByPid[stats.winner]} wins</div>}
                <button className="ghost stats-watch" onClick={onWatchReplay}>Watch replay →</button>
            </div>

            {/* ── Tab bar ────────────────────────────────────────────── */}
            <div className="stats-tabs">
                {TABS.map((t) => (
                    <button key={t.id} className={`stats-tab${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
                        {t.label}
                    </button>
                ))}
            </div>

            {/* ── Overview ───────────────────────────────────────────── */}
            {tab === 'overview' && (
                <div className="card">
                    <h3>Final standings</h3>
                    <StatTable players={ranked} colorFor={colorFor} cols={[
                        { label: 'VP', value: (p) => p.finalVP, highlight: true },
                        { label: '🏠', title: 'Settlements', value: (p) => p.vpBreakdown.settlements },
                        { label: '🏙', title: 'Cities', value: (p) => p.vpBreakdown.cities },
                        { label: '⚔', title: 'Knights played', value: (p) => p.devCardsPlayed.knight, highlight: true },
                        { label: '★', title: 'VP cards', value: (p) => p.vpBreakdown.devCards },
                        { label: 'LR', title: 'Longest road', value: (p) => p.vpBreakdown.longestRoad, render: (p) => p.vpBreakdown.longestRoad > 0 ? '✓' : '—' },
                        { label: 'LA', title: 'Largest army', value: (p) => p.vpBreakdown.largestArmy, render: (p) => p.vpBreakdown.largestArmy > 0 ? '✓' : '—' },
                        { label: 'Gained', title: 'Resource cards gained', value: (p) => bundleTotal(p.resourcesGained), highlight: true },
                        { label: 'Lost', title: 'Resource cards lost', value: (p) => bundleTotal(p.resourcesLost) },
                    ]} />
                </div>
            )}

            {/* ── Resources: income/loss by source ───────────────────── */}
            {tab === 'resources' && (
                <div className="stats-players">
                    {stats.players.map((p) => {
                        const catRows = CATEGORY_ORDER.filter(
                            (c) => bundleTotal(p.gainedByCategory[c]) + bundleTotal(p.lostByCategory[c]) > 0,
                        );
                        const colNet = (r: Resource) => (p.resourcesGained[r] ?? 0) - (p.resourcesLost[r] ?? 0);
                        const grandNet = RESOURCES.reduce((s, r) => s + colNet(r), 0);
                        return (
                            <div key={p.player} className="card stats-player" style={{ ['--pc' as string]: colorFor(p.player) }}>
                                <div className="phead">
                                    <span className="dot" style={{ background: colorFor(p.player) }} />
                                    <span className="pname">{p.name}</span>
                                    <span className="vp">{p.finalVP} <small>VP</small></span>
                                </div>
                                <table className="rmatrix">
                                    <thead>
                                        <tr>
                                            <th />
                                            {RESOURCES.map((r) => <th key={r} style={{ color: RES_COLOR[r] }}>{SHORT[r]}</th>)}
                                            <th className="rmatrix-netcol">Net</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {catRows.map((c) => {
                                            const rowNet = RESOURCES.reduce((s, r) => s + netOf(p.gainedByCategory[c], p.lostByCategory[c], r), 0);
                                            return (
                                                <tr key={c}>
                                                    <td className="rmatrix-cat">{CATEGORY_LABEL[c]}</td>
                                                    {RESOURCES.map((r) => {
                                                        const v = netOf(p.gainedByCategory[c], p.lostByCategory[c], r);
                                                        return <td key={r} className={cellClass(v)}>{signed(v)}</td>;
                                                    })}
                                                    <td className={`rmatrix-netcol ${cellClass(rowNet)}`}>{signed(rowNet)}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                    <tfoot>
                                        <tr>
                                            <td className="rmatrix-cat">Total</td>
                                            {RESOURCES.map((r) => { const v = colNet(r); return <td key={r} className={cellClass(v)}>{signed(v)}</td>; })}
                                            <td className={`rmatrix-netcol ${cellClass(grandNet)}`}>{signed(grandNet)}</td>
                                        </tr>
                                    </tfoot>
                                </table>
                                <div className="rmatrix-gl">
                                    <span><span className="pos">▲</span> gained Σ{bundleTotal(p.resourcesGained)}</span>
                                    <span><span className="neg">▼</span> lost Σ{bundleTotal(p.resourcesLost)}</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* ── Dice histogram ─────────────────────────────────────── */}
            {tab === 'dice' && (
                <div className="card dice-card">
                    <h3>Dice rolls <span className="muted">· {stats.dice.totalRolls} total</span></h3>
                    <div className="dice-hist">
                        {Array.from({ length: 11 }, (_, i) => i + 2).map((n) => {
                            const count = stats.dice.rollCounts[n] ?? 0;
                            const h = Math.round((count / maxRollCount) * 150);
                            const kind = n === 7 ? ' seven' : (n === 6 || n === 8) ? ' hot' : '';
                            return (
                                <div key={n} className="dice-bar-col">
                                    <div className="dice-bar-count">{count || ''}</div>
                                    <div className="dice-bar-track">
                                        <div className={`dice-bar${kind}`} style={{ height: `${Math.max(h, count > 0 ? 4 : 0)}px` }} />
                                    </div>
                                    <div className={`dice-bar-label${kind}`}>{n}</div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* ── Cards drawn (resource + dev) ────────────────────────── */}
            {tab === 'cards' && (
                <>
                    <div className="card">
                        <h3>Resource cards collected</h3>
                        <StatTable players={stats.players} colorFor={colorFor} cols={[
                            ...RESOURCES.map((r) => ({
                                label: SHORT[r], title: r,
                                value: (p: PlayerStats) => p.resourcesGained[r] ?? 0, highlight: true,
                                render: (p: PlayerStats) => <span style={{ color: RES_COLOR[r] }}>{p.resourcesGained[r] ?? 0}</span>,
                            })),
                            { label: 'Σ', title: 'Total drawn', value: (p) => bundleTotal(p.resourcesGained), highlight: true },
                        ]} />
                    </div>
                    <div className="card">
                        <h3>Development cards drawn</h3>
                        <StatTable players={stats.players} colorFor={colorFor} cols={[
                            ...DEV_TYPES.map((d) => ({
                                label: d.label, value: (p: PlayerStats) => p.devCardsDrawn[d.key], highlight: true,
                            })),
                            { label: 'Σ', title: 'Total bought', value: (p) => p.devCardsBought, highlight: true },
                        ]} />
                    </div>
                </>
            )}

            {/* ── Activity: trades + robber ──────────────────────────── */}
            {tab === 'activity' && (
                <>
                    <div className="card">
                        <h3>Trading</h3>
                        <StatTable players={stats.players} colorFor={colorFor} cols={[
                            { label: 'Offered', title: 'Player offers proposed', value: (p) => p.trade.offersBroadcast },
                            { label: 'Countered', value: (p) => p.trade.counterOffersMade },
                            { label: 'Completed', title: 'Player trades completed', value: (p) => p.trade.tradesCompleted, highlight: true },
                            { label: 'Declined', value: (p) => p.trade.offersDeclined },
                            { label: 'Bank', title: 'Bank / port trades', value: (p) => p.trade.bankTradesCount, highlight: true },
                        ]} />
                    </div>
                    <div className="card">
                        <h3>Robber &amp; dev cards</h3>
                        <StatTable players={stats.players} colorFor={colorFor} cols={[
                            { label: 'Moved', title: 'Robber/knight moves', value: (p) => p.robber.timesMoved, highlight: true },
                            { label: 'Empty', title: 'Moves with no one to rob', value: (p) => p.robber.timesNoVictim },
                            { label: 'Stole', title: 'Cards stolen from others', value: (p) => bundleTotal(p.robber.stolenFromOthers), highlight: true },
                            { label: 'Robbed', title: 'Times robbed by others', value: (p) => p.robber.timesTargeted },
                            { label: 'Blocked', title: 'Production blocked by robber', value: (p) => bundleTotal(p.robber.productionBlocked),
                              render: (p) => <Chips bundle={p.robber.productionBlocked} /> },
                            { label: 'Dev used', title: 'Dev cards played', value: (p) => p.devCardsPlayed.knight + p.devCardsPlayed.roadBuilding + p.devCardsPlayed.yearOfPlenty + p.devCardsPlayed.monopoly },
                        ]} />
                    </div>
                </>
            )}
        </div>
    );
}
