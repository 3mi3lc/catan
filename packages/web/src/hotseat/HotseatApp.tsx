// Hotseat pass-&-play client (hotseat.html) — React port of the vanilla
// hotseat.ts. All game logic still funnels through @catan/core's applyMove
// (the engine stays the single source of truth; illegal moves surface as
// toasts); what changed is only the rendering/wiring layer: module-level
// state + innerHTML templates became one per-game `Session` object in React
// state, and the data-attribute click wiring became ordinary props/handlers.
//
// The per-game state deliberately lives in ONE Session object (not a dozen
// useStates): the vanilla act() updated game/log/counters/mode atomically
// before a single render() call, and a single setSession preserves exactly
// that all-at-once semantics. A ref mirror (sessionRef) gives event handlers
// and the bot's setTimeout callback the CURRENT session (the vanilla code
// read live module variables at fire time — closures over render-time state
// would reintroduce staleness the vanilla code never had).

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    generateBoard, initialGameState, applyMove, legalActions,
    roadConnected, tradeRatio, victoryPoints, rngStep, RESOURCES,
    type GameState, type Move, type Action, type GameEvent,
    type PlayerId, type EdgeId, type TileId, type VertexId,
    type Resource, type PlayerColor,
    type MoveRecord, type GameArchive,
} from '@catan/core';
import type { Bot, BotLevel } from '../bot';
import BoardSvg from '../shared/BoardSvg';
import { producingTilesForRoll } from '../game/boardArt';
import DiceOverlay, { type DiceRoll } from '../shared/DiceOverlay';
import SfxControl from '../shared/SfxControl';
import { useToast } from '../shared/useToast';
import { preloadSfx, installClickSfx, playForEvents, playForAction } from '../audio/sound';
import { hapticsForEvents, hapticsForAction } from '../haptics';
import './hotseat.css';

// ---------------------------------------------------------------------------
// Palette (dark tabletop theme — deliberately hotseat-local, not theme.ts's).
// ---------------------------------------------------------------------------
const RES_COLOR: Record<Resource, string> = {
    brick: '#c5673a', lumber: '#3d7a4a', wool: '#9dbe5a', grain: '#e3b23c', ore: '#98a0aa',
};
const RES_ABBR: Record<Resource, string> = { brick: 'Br', lumber: 'Lu', wool: 'Wo', grain: 'Gr', ore: 'Or' };
const PLAYER: Record<PlayerColor, { fill: string; ink: string; name: string }> = {
    red:    { fill: '#d24a36', ink: '#fff', name: 'Red' },
    blue:   { fill: '#3f7cc4', ink: '#fff', name: 'Blue' },
    white:  { fill: '#e9e4d6', ink: '#2c2620', name: 'White' },
    orange: { fill: '#e08c2e', ink: '#fff', name: 'Orange' },
};
const SEAT_DEFS = [
    { id: 'p0', color: 'red' as const },
    { id: 'p1', color: 'blue' as const },
    { id: 'p2', color: 'white' as const },
    { id: 'p3', color: 'orange' as const },
];

// An Rng built on the core PRNG, so one integer seed reproduces the whole game.
const rngFromSeed = (seed: number) => {
    let s = seed >>> 0;
    return () => { const { value, next } = rngStep(s); s = next; return value; };
};
const randSeed = () => (Math.random() * 0x1_0000_0000) >>> 0;

// Pacing between consecutive bot moves — mirrors packages/server's identical
// constants (see index.ts's settle()) so AI turns feel the same whether the
// bots are running here or server-side in online play.
const AI_MOVE_DELAY_MS = 600;
const AI_DICE_ROLL_EXTRA_MS = 1200;

// How long the "just produced" tile pop stays up before clearing — matches
// boardArt.ts's ba-produce-pop-kf animation duration (1.6s), plus a small
// margin so the CSS fade genuinely finishes before the DOM nodes go away.
const PRODUCE_POP_MS = 1700;
// Stable empty-array reference for BoardSvg's producingTiles prop when
// there's nothing to flash — a fresh `[]` literal on every render would
// defeat BoardSvg's own memoization (see its prop doc comment).
const EMPTY_TILES: readonly TileId[] = [];

// ---------------------------------------------------------------------------
// Per-game session state
// ---------------------------------------------------------------------------
type Mode =
    | { kind: 'normal' }
    | { kind: 'robber'; via: 'seven' | 'knight' }
    | { kind: 'roadbuilding'; edges: EdgeId[] }
    | { kind: 'yearofplenty'; pick: Resource[] }
    | { kind: 'monopoly' };

interface Session {
    game: GameState;
    mode: Mode;
    victimChoice: { tile: TileId; via: 'seven' | 'knight'; victims: PlayerId[] } | null;
    tradeGive: Resource | null;
    discardSel: Record<string, Partial<Record<Resource, number>>>;
    log: string[];
    moveCount: number;   // successful engine moves this game
    turnCount: number;   // 1-based turn number (increments on endTurn)
    // Recorded for the post-game stats/replay screen (see storeArchiveForReplay).
    moves: MoveRecord[];
    gameSeed: number;
    policyNames: string[];
    // AI opponent config for this game. In an AI game the bot fills every seat
    // except the human's, whose seat is randomised so they don't always play
    // first. In pass-&-play this stays p0 and is unused.
    botLevel: BotLevel | 'human';
    humanSeat: PlayerId;
}

function buildSession(seed: number, players: number, noSameNumbers: boolean, botLevel: BotLevel | 'human'): Session {
    const board = generateBoard(rngFromSeed(seed), { forbidAdjacentSameNumber: noSameNumbers });
    const seats = SEAT_DEFS.slice(0, players).map((s) => ({ id: s.id, name: PLAYER[s.color].name, color: s.color }));
    // Interactive play isn't bounded by search depth like self-play training is,
    // so let people negotiate as many offers per turn as the real board game does.
    // A large finite cap, not Infinity: this game's archive is JSON-serialized
    // to sessionStorage for the post-game replay screen, and
    // JSON.stringify(Infinity) is `null` (see the matching fix in room.ts).
    const game = initialGameState(board, seats, seed, Number.MAX_SAFE_INTEGER);
    const humanSeat = botLevel !== 'human'
        ? (`p${Math.floor(Math.random() * players)}` as PlayerId)
        : ('p0' as PlayerId);
    return {
        game,
        mode: { kind: 'normal' },
        victimChoice: null,
        tradeGive: null,
        discardSel: {},
        log: [`New game · seed ${seed} · ${players} players`
            + (botLevel !== 'human' ? ` · you are ${PLAYER[game.players[humanSeat].color].name}` : '')],
        moveCount: 0,
        turnCount: 1,
        moves: [],
        gameSeed: seed,
        policyNames: seats.map((s) =>
            botLevel !== 'human' && s.id !== humanSeat ? `AI (${s.name})` : s.name),
        botLevel,
        humanSeat,
    };
}

// ---------------------------------------------------------------------------
// Pure helpers over a session/game (no React)
// ---------------------------------------------------------------------------
const nameOf = (g: GameState, p: PlayerId) => PLAYER[g.players[p].color].name;
const colorOf = (g: GameState, p: PlayerId) => PLAYER[g.players[p].color].fill;
const handSize = (g: GameState, p: PlayerId) => RESOURCES.reduce((s, r) => s + g.players[p].resources[r], 0);

function bundleTotalOf(b: Partial<Record<Resource, number>>): number {
    return RESOURCES.reduce((s, r) => s + (b[r] ?? 0), 0);
}
/** Compact "1 Wo + 2 Or" rendering of a resource bundle. */
function bundleStr(b: Partial<Record<Resource, number>>): string {
    return (Object.keys(b) as Resource[])
        .filter((r) => (b[r] ?? 0) > 0)
        .map((r) => `${b[r]} ${RES_ABBR[r]}`).join(' + ') || '—';
}

function describe(g: GameState, ev: GameEvent): string {
    switch (ev.type) {
        case 'diceRolled': return `${nameOf(g, g.currentPlayer)} rolled ${ev.dice[0]} + ${ev.dice[1]} = ${ev.total}`;
        case 'resourcesProduced': {
            const parts = Object.entries(ev.gains).map(([p, gains]) => {
                const items = RESOURCES.filter((r) => gains[r]).map((r) => `+${gains[r]} ${r}`).join(', ');
                return items ? `${nameOf(g, p as PlayerId)}: ${items}` : '';
            }).filter(Boolean);
            return parts.length ? `Production — ${parts.join('; ')}` : 'Production — nobody collected';
        }
        case 'built': return `${nameOf(g, ev.player)} built a ${ev.what}`;
        case 'robberMoved': return `${nameOf(g, g.currentPlayer)} moved the robber${ev.stolen ? ' and stole a card' : ''}`;
        case 'devCardBought': return `${nameOf(g, ev.player)} bought a development card`;
        case 'tradeExecuted':
            return `Trade: ${nameOf(g, ev.between[0])} gives ${bundleStr(ev.proposerGives)} to ${nameOf(g, ev.between[1])} for ${bundleStr(ev.proposerGets)}`;
        case 'awardMoved': return `${ev.award} → ${nameOf(g, ev.to)}`;
        case 'gameWon': return `${nameOf(g, ev.player)} wins!`;
    }
}

/** Negotiation steps (offer/respond/counter/decline) carry no resource change
 *  and so emit no GameEvent — log them from the action itself instead. */
function describeTradeStep(g: GameState, move: Move): string | null {
    const a = move.action;
    switch (a.type) {
        case 'offerBroadcast': {
            const neg = g.negotiation;
            return neg ? `${nameOf(g, neg.proposer)} offers ${bundleStr(neg.give)} for ${bundleStr(neg.want)}` : null;
        }
        case 'respondAccept': return `${nameOf(g, move.player)} accepts the offer`;
        case 'respondReject': return `${nameOf(g, move.player)} declines the offer`;
        case 'submitCounter': {
            const c = g.negotiation?.counters[move.player];
            return c ? `${nameOf(g, move.player)} counters with ${bundleStr(c.give)} for ${bundleStr(c.want)}` : null;
        }
        case 'declineAll': return `${nameOf(g, move.player)} ends the negotiation without a trade`;
        default: return null;
    }
}

/** The seat that must act on the trade negotiation right now, or null if none:
 *  a draft composer, then each replying opponent, then the arbitrating proposer. */
function negotiationActor(g: GameState): PlayerId | null {
    if (g.draftOffer) return g.draftOffer.by;
    const neg = g.negotiation;
    if (!neg) return null;
    if (neg.stage === 'responding')
        return g.turnOrder.find((p) => p !== neg.proposer && neg.responses[p] === 'pending') ?? null;
    return neg.proposer;
}

function robberVictims(g: GameState, tile: TileId): PlayerId[] {
    const set = new Set<PlayerId>();
    for (const v of g.board.tiles[tile].vertices) {
        const b = g.buildings[v];
        if (b && b.owner !== g.currentPlayer && handSize(g, b.owner) > 0) set.add(b.owner);
    }
    return [...set];
}

function phaseLabel(g: GameState): string {
    return {
        setupSettlement: 'initial settlement', setupRoad: 'initial road', roll: 'roll the dice',
        discard: 'discard', moveRobber: 'move robber', main: 'build & trade', gameOver: 'game over',
    }[g.phase];
}

/** The seat the bot should act for right now, or null (human's turn / no AI). */
function botActor(s: Session, bot: Bot | null): PlayerId | null {
    if (s.botLevel === 'human' || !bot || s.game.winner) return null;
    // Trade negotiation steps are off-turn decisions for whoever must act.
    const trader = negotiationActor(s.game);
    if (trader) return trader !== s.humanSeat ? trader : null;
    if (s.game.phase === 'discard') {
        // Resolve each bot seat that owes cards; the human handles their own.
        return (Object.keys(s.game.pendingDiscards) as PlayerId[])
            .find((p) => p !== s.humanSeat) ?? null;
    }
    return s.game.currentPlayer !== s.humanSeat ? s.game.currentPlayer : null;
}

/** Hands the finished game to the stats/replay screen (replay.html) via
 *  sessionStorage — mirrors what the online client does with the server-built
 *  archive from the `gameOver` socket event, just built client-side since
 *  there's no server in pass-&-play. */
function storeArchiveForReplay(s: Session): void {
    const archive: GameArchive = {
        seed: s.gameSeed,
        policyNames: s.policyNames,
        winnerSeat: s.game.winner ? s.game.turnOrder.indexOf(s.game.winner) : null,
        turns: s.turnCount,
        moves: s.moves,
        maxOffersPerTurn: s.game.maxOffersPerTurn,
    };
    sessionStorage.setItem('catan-last-archive', JSON.stringify(archive));
}

const cvar = (c: string) => ({ '--c': c } as React.CSSProperties);

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function HotseatApp() {
    // The vanilla page randomised the seed box and immediately auto-started a
    // 4-player pass-&-play game on load — note the auto-start does NOT apply
    // the house rule even though its checkbox defaults checked (startGame's
    // noSameNumbers parameter was simply omitted). Preserved as-is.
    const [initial] = useState(() => {
        const seed = randSeed();
        return { seed, session: buildSession(seed, 4, false, 'human') };
    });
    const [seedText, setSeedText] = useState(String(initial.seed));
    const seedDirty = useRef(false);
    const [playerCount, setPlayerCount] = useState('4');
    const [opponent, setOpponent] = useState<BotLevel | 'human'>('human');
    const [noSame, setNoSame] = useState(true);

    const [session, setSessionState] = useState<Session>(initial.session);
    const sessionRef = useRef(session);
    const commit = (next: Session) => { sessionRef.current = next; setSessionState(next); };

    const [dice, setDice] = useState<DiceRoll | null>(null);
    const diceNonce = useRef(0);
    const showDice = (a: number, b: number) => setDice({ a, b, nonce: ++diceNonce.current });

    // Tiles to flash "just produced" on for the most recent roll (see
    // BoardSvg's producingTiles prop) — cleared after PRODUCE_POP_MS, guarded
    // by nonce so a stale timeout from an earlier roll can't clear a newer
    // one's highlight if rolls somehow land less than that far apart.
    const [producing, setProducing] = useState<{ tiles: readonly TileId[]; nonce: number } | null>(null);
    const producingNonce = useRef(0);
    const flashProducingTiles = (game: GameState, total: number) => {
        const tiles = producingTilesForRoll(game.board, game.robber, total);
        if (!tiles.length) return;
        const nonce = ++producingNonce.current;
        setProducing({ tiles, nonce });
        setTimeout(() => setProducing((p) => (p?.nonce === nonce ? null : p)), PRODUCE_POP_MS);
    };

    const [victoryDismissed, setVictoryDismissed] = useState(false);
    const { toastEl, toast } = useToast();

    const botRef = useRef<Bot | null>(null);
    const loadedModel = useRef<string | null>(null);
    const botBusy = useRef(false);
    // What the most recently committed move produced — read by the bot-turn
    // effect below to decide how long to pace before its NEXT decision, so a
    // dice roll actually gets to finish its ~1.6s animation (DiceOverlay)
    // before the following move overwrites the log/board.
    const lastEventsRef = useRef<GameEvent[]>([]);

    useEffect(() => { preloadSfx(); installClickSfx(); }, []);

    // -----------------------------------------------------------------------
    // Applying moves — every interaction funnels through here.
    // -----------------------------------------------------------------------
    const act = (move: Move): void => {
        const s = sessionRef.current;
        const phase = s.game.phase;
        const res = applyMove(s.game, move);
        if (!res.ok) { toast(res.error); return; }
        const game = res.state;
        playForEvents(res.events);
        playForAction(move.action);
        hapticsForEvents(res.events);
        hapticsForAction(move.action);
        let log = s.log;
        for (const ev of res.events) {
            log = [describe(game, ev), ...log];
            if (ev.type === 'diceRolled') { showDice(ev.dice[0], ev.dice[1]); flashProducingTiles(game, ev.total); }
        }
        const tradeMsg = describeTradeStep(game, move);
        if (tradeMsg) log = [tradeMsg, ...log];
        const turnCount = s.turnCount + (move.action.type === 'endTurn' ? 1 : 0);
        const moveCount = s.moveCount + 1;
        const next: Session = {
            ...s,
            game,
            moves: [...s.moves, { player: move.player, phase, action: move.action }],
            moveCount,
            turnCount,
            log,
            mode: { kind: 'normal' },
            victimChoice: null,
            tradeGive: null,
        };
        if (game.winner) {
            next.log = [`${nameOf(game, game.winner)} wins in ${turnCount} turns (${moveCount} moves)!`, ...next.log];
            storeArchiveForReplay(next);
        }
        next.log = next.log.slice(0, 40);
        lastEventsRef.current = res.events;
        commit(next);
    };

    const resolveRobber = (tile: TileId, via: 'seven' | 'knight'): void => {
        const s = sessionRef.current;
        const victims = robberVictims(s.game, tile);
        const build = (steal: PlayerId | null): Action =>
            via === 'knight'
                ? { type: 'playKnight', robberTo: tile, stealFrom: steal }
                : { type: 'moveRobber', tile, stealFrom: steal };
        if (victims.length <= 1) act({ player: s.game.currentPlayer, action: build(victims[0] ?? null) });
        else commit({ ...s, victimChoice: { tile, via, victims } });
    };

    // -----------------------------------------------------------------------
    // Board interactions per phase/mode (rebuilt each render, like vanilla).
    // -----------------------------------------------------------------------
    const g = session.game;
    const interactions = useMemo(() => {
        const vClicks = new Map<VertexId, () => void>();
        const eClicks = new Map<EdgeId, () => void>();
        const tClicks = new Map<TileId, () => void>();
        // While the bot is acting (its turn, or its move in a trade negotiation),
        // the board stays inert — otherwise a click here would place a piece on
        // the bot's behalf before it gets to decide.
        if (botActor(session, botRef.current) !== null) return { vClicks, eClicks, tClicks };
        const cur = g.currentPlayer;
        const mode = session.mode;

        if (mode.kind === 'robber') {
            for (const t of Object.keys(g.board.tiles) as TileId[])
                if (t !== g.robber) tClicks.set(t, () => resolveRobber(t, mode.via));
            return { vClicks, eClicks, tClicks };
        }

        if (mode.kind === 'roadbuilding') {
            const placed = mode.edges;
            const temp: GameState = { ...g, roads: { ...g.roads } };
            for (const e of placed) temp.roads[e] = cur;
            for (const e of Object.keys(g.board.edges) as EdgeId[]) {
                if (temp.roads[e] || !roadConnected(temp, cur, e)) continue;
                eClicks.set(e, () => {
                    const edges = [...placed, e];
                    if (edges.length === 2) act({ player: cur, action: { type: 'playRoadBuilding', edges: [edges[0], edges[1]] } });
                    else commit({ ...sessionRef.current, mode: { kind: 'roadbuilding', edges } });
                });
            }
            return { vClicks, eClicks, tClicks };
        }

        // Normal phases: derive directly from the authoritative legal-action list.
        for (const a of legalActions(g, cur)) {
            if (a.type === 'buildSettlement' || a.type === 'buildCity') vClicks.set(a.vertex, () => act({ player: cur, action: a }));
            else if (a.type === 'buildRoad') eClicks.set(a.edge, () => act({ player: cur, action: a }));
            else if (a.type === 'moveRobber' && !tClicks.has(a.tile)) tClicks.set(a.tile, () => resolveRobber(a.tile, 'seven'));
        }
        return { vClicks, eClicks, tClicks };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session]);

    // BoardSvg's own useMemo is keyed on these by reference — an inline arrow
    // or a `[...map.keys()]` spread written directly in JSX is a brand-new
    // object on every render regardless of whether the board actually
    // changed, which defeated that memo entirely (forcing a full SVG string
    // rebuild, and the browser to redecode every tile image, on every
    // unrelated re-render — e.g. the dice-overlay or toast timers firing).
    // Stabilizing the references here means BoardSvg only redoes that work
    // when the game state or legal-target set genuinely changed.
    const boardColorOf = useCallback((p: PlayerId) => colorOf(g, p), [g]);
    const vTargetsArr = useMemo(() => [...interactions.vClicks.keys()], [interactions]);
    const eTargetsArr = useMemo(() => [...interactions.eClicks.keys()], [interactions]);
    const tTargetsArr = useMemo(() => [...interactions.tClicks.keys()], [interactions]);

    // -----------------------------------------------------------------------
    // AI opponent turn loop — fires after every committed session, mirroring
    // the vanilla render() → maybeBotMove() chain.
    // -----------------------------------------------------------------------
    useEffect(() => {
        const seat = botActor(session, botRef.current);
        if (!seat || botBusy.current) return;
        botBusy.current = true;
        // Paced so consecutive bot turns (several AI seats' worth of setup
        // placements, a full AI turn's roll→build→end sequence, …) read as a
        // series of individual moves rather than flashing by instantly — and
        // long enough after a dice roll specifically for DiceOverlay's ~1.6s
        // animation to actually finish before the next move overwrites it.
        // The game state is re-read from the ref at fire time (the vanilla
        // version read the live module variable), so a discard the human
        // confirmed during the delay is already reflected.
        const delay = AI_MOVE_DELAY_MS + (lastEventsRef.current.some((e) => e.type === 'diceRolled') ? AI_DICE_ROLL_EXTRA_MS : 0);
        setTimeout(async () => {
            try {
                const cur = sessionRef.current;
                const action = await botRef.current!.decide(cur.game, seat, cur.botLevel as BotLevel);
                botBusy.current = false;
                act({ player: seat, action });   // commit → effect runs again → chains
            } catch (err) {
                botBusy.current = false;
                toast(`AI error: ${err instanceof Error ? err.message : err}`);
            }
        }, delay);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session]);

    /**
     * Load the bot net appropriate to the table size. 2-player and 4-player
     * Catan reward different strategies, so they get separate nets; 3-player
     * reuses the 4-player net. Falls back to the 4-player net if a dedicated
     * 2-player net hasn't been trained/deployed yet.
     */
    async function loadBotFor(players: number): Promise<void> {
        const want = players === 2 ? '2p' : '4p';
        if (botRef.current && loadedModel.current === want) return;
        const { Bot } = await import('../bot');
        const primary = `/models/catan_net_${want}.onnx`;
        try {
            botRef.current = await Bot.load(primary);
        } catch {
            if (want === '2p') botRef.current = await Bot.load('/models/catan_net_4p.onnx');
            else throw new Error(`missing model ${primary}`);
        }
        loadedModel.current = want;
    }

    const newGame = async (): Promise<void> => {
        let botLevel = opponent;
        const players = Number(playerCount);
        if (botLevel !== 'human') {
            toast('Loading AI model…');
            try {
                await loadBotFor(players);
            } catch (err) {
                toast(`Could not load AI: ${err instanceof Error ? err.message : err}`);
                botLevel = 'human';
            }
        }
        // The board is randomised on every "New game" press. Typing into the
        // seed box opts that one game out (for sharing/reproducing a specific
        // board); once used, the field auto-randomises again next press.
        const seed = seedDirty.current ? Number(seedText) || 0 : randSeed();
        setSeedText(String(seed));
        seedDirty.current = false;
        setVictoryDismissed(false);
        commit(buildSession(seed, players, noSame, botLevel));
    };

    // -----------------------------------------------------------------------
    // Panels
    // -----------------------------------------------------------------------
    const trader = negotiationActor(g);
    const botThinksTrade = trader !== null && !(session.botLevel === 'human' || trader === session.humanSeat);

    function ActionsPanel() {
        if (g.winner) return <div className="won">{nameOf(g, g.winner)} wins the game! 🏆</div>;

        // A trade negotiation takes over the panel for whoever must act.
        if (trader) {
            const turnline = (label: string) => (
                <div className="turnline"><span className="dot" style={{ background: colorOf(g, trader!) }} /> <b>{nameOf(g, trader!)}</b> — {label}</div>
            );
            if (botThinksTrade)
                return <>{turnline('trading')}<div className="ctx"><div className="hint">🤖 {nameOf(g, trader)} is negotiating…</div></div></>;
            if (g.draftOffer && g.draftOffer.by === trader)
                return <>{turnline(g.negotiation ? 'counter-offer' : 'compose offer')}<div className="ctx">{DraftBuilder()}</div></>;
            const neg = g.negotiation!;
            if (neg.stage === 'responding')
                return <>{turnline('respond to offer')}<div className="ctx">{Respond({ responder: trader })}</div></>;
            return <>{turnline('choose a trade')}<div className="ctx">{Arbitrate({ proposer: trader })}</div></>;
        }

        const cur = g.currentPlayer;
        const thinking = botActor(session, botRef.current) !== null && g.phase !== 'discard';
        const acts = legalActions(g, cur);
        const has = (t: Action['type']) => acts.some((a) => a.type === t);

        // A transient sub-flow (robber, dev-card picker, discard) or the bot's
        // turn "owns" the panel — core buttons stay locked so the layout never
        // shifts and the human can't stack two actions at once.
        const busy = thinking || !!session.victimChoice
            || session.mode.kind === 'robber' || session.mode.kind === 'roadbuilding'
            || session.mode.kind === 'monopoly' || session.mode.kind === 'yearofplenty'
            || g.phase === 'moveRobber' || g.phase === 'discard'
            || g.phase === 'setupSettlement' || g.phase === 'setupRoad';

        const btn = (legal: boolean, onClick: () => void, label: string, extra?: Record<string, unknown>) => (
            <button disabled={!(legal && !busy)} onClick={onClick} {...extra}>{label}</button>
        );

        return (
            <>
                <div className="turnline"><span className="dot" style={{ background: colorOf(g, cur) }} /> <b>{nameOf(g, cur)}</b> — {phaseLabel(g)}</div>
                <div className="row">
                    {btn(has('rollDice'), () => act({ player: cur, action: { type: 'rollDice' } }), 'Roll dice')}
                    {btn(has('buyDevCard'), () => act({ player: cur, action: { type: 'buyDevCard' } }), 'Buy dev card')}
                    {btn(has('endTurn'), () => act({ player: cur, action: { type: 'endTurn' } }), 'End turn')}
                </div>
                <div className="row">
                    {btn(has('playKnight'), () => commit({ ...sessionRef.current, mode: { kind: 'robber', via: 'knight' } }), 'Knight')}
                    {btn(has('playRoadBuilding'), () => commit({ ...sessionRef.current, mode: { kind: 'roadbuilding', edges: [] } }), 'Road Building')}
                    {btn(has('playYearOfPlenty'), () => commit({ ...sessionRef.current, mode: { kind: 'yearofplenty', pick: [] } }), 'Year of Plenty')}
                    {btn(has('playMonopoly'), () => commit({ ...sessionRef.current, mode: { kind: 'monopoly' } }), 'Monopoly')}
                </div>
                <div className="ctx">{Context({ thinking })}</div>
            </>
        );
    }

    // The single region that legitimately changes shape: prompts and pickers
    // for whatever the current step needs.
    function Context({ thinking }: { thinking: boolean }) {
        const cur = g.currentPlayer;
        const cancelBtn = <button className="ghost" onClick={() => commit({ ...sessionRef.current, mode: { kind: 'normal' } })}>Cancel</button>;
        if (thinking)
            return <div className="hint">🤖 {nameOf(g, cur)} is thinking{session.botLevel === 'mcts' ? ' (searching)…' : '…'}</div>;
        if (session.victimChoice) {
            return (
                <>
                    <div className="hint">Steal from whom?</div>
                    <div className="row">
                        {session.victimChoice.victims.map((vp) => (
                            <button key={vp} onClick={() => {
                                const vc = sessionRef.current.victimChoice;
                                if (!vc) return;
                                const a: Action = vc.via === 'knight'
                                    ? { type: 'playKnight', robberTo: vc.tile, stealFrom: vp }
                                    : { type: 'moveRobber', tile: vc.tile, stealFrom: vp };
                                act({ player: cur, action: a });
                            }}>{nameOf(g, vp)} ({handSize(g, vp)})</button>
                        ))}
                    </div>
                </>
            );
        }
        if (g.phase === 'discard') return Discard();
        if (session.mode.kind === 'robber') return <div className="hint">Click a tile to move the robber.</div>;
        if (g.phase === 'moveRobber') return <div className="hint">Move the robber — click a tile.</div>;
        if (session.mode.kind === 'roadbuilding')
            return <><div className="hint">Road Building — pick {2 - session.mode.edges.length} more road(s).</div><div className="row">{cancelBtn}</div></>;
        if (session.mode.kind === 'monopoly')
            return (
                <>
                    <div className="hint">Monopoly — choose a resource.</div>
                    <div className="row">
                        {RESOURCES.map((r) => (
                            <button key={r} data-monopoly={r} style={cvar(RES_COLOR[r])}
                                onClick={() => act({ player: cur, action: { type: 'playMonopoly', resource: r } })}>{r}</button>
                        ))}
                        {cancelBtn}
                    </div>
                </>
            );
        if (session.mode.kind === 'yearofplenty') {
            const picked = session.mode.pick;
            return (
                <>
                    <div className="hint">Year of Plenty — choose 2 (picked {picked.length}).</div>
                    <div className="row">
                        {RESOURCES.map((r) => (
                            <button key={r} data-yop={r} style={cvar(RES_COLOR[r])} onClick={() => {
                                const m = sessionRef.current.mode;
                                if (m.kind !== 'yearofplenty') return;
                                const pick = [...m.pick, r];
                                if (pick.length === 2) act({ player: cur, action: { type: 'playYearOfPlenty', take: [pick[0], pick[1]] } });
                                else commit({ ...sessionRef.current, mode: { kind: 'yearofplenty', pick } });
                            }}>{r}</button>
                        ))}
                        {cancelBtn}
                    </div>
                </>
            );
        }
        if (g.phase === 'setupSettlement') return <div className="hint">Place your starting settlement — click a highlighted spot.</div>;
        if (g.phase === 'setupRoad') return <div className="hint">Place a road next to it — click a highlighted edge.</div>;
        if (g.phase === 'roll') return <div className="hint">Roll the dice to begin your turn.</div>;
        // Draft/negotiation panels are handled at the top of ActionsPanel.
        if (g.phase === 'main')
            return <><div className="hint">Build by clicking highlighted spots on the board.</div>{TradePanel()}</>;
        return null;
    }

    function TradePanel() {
        const cur = g.currentPlayer;
        return (
            <>
                <div className="trade-wrap">
                    <div className="trade-lab">Bank trade — give</div>
                    <div className="row">
                        {RESOURCES.map((r) => {
                            const ratio = tradeRatio(g, cur, r);
                            const can = g.players[cur].resources[r] >= ratio;
                            return (
                                <button key={r} className={`trade${session.tradeGive === r ? ' sel' : ''}`} data-give={r}
                                    disabled={!can} style={cvar(RES_COLOR[r])}
                                    onClick={() => commit({ ...sessionRef.current, tradeGive: r })}>{ratio} {RES_ABBR[r]}</button>
                            );
                        })}
                    </div>
                    <div className="trade-lab">receive</div>
                    <div className="row">
                        {session.tradeGive
                            ? RESOURCES.filter((r) => r !== session.tradeGive && g.bank[r] > 0).map((r) => (
                                <button key={r} className="trade" data-receive={r} style={cvar(RES_COLOR[r])} onClick={() => {
                                    const give = sessionRef.current.tradeGive;
                                    if (!give) return;
                                    act({ player: cur, action: { type: 'bankTrade', give, giveCount: tradeRatio(sessionRef.current.game, cur, give), receive: r } });
                                }}>{RES_ABBR[r]}</button>
                            ))
                            : <span className="hint">pick what to give</span>}
                    </div>
                </div>
                {StartOffer()}
            </>
        );
    }

    /** No draft yet: a row of "give" cards that each start an offer. */
    function StartOffer() {
        const cur = g.currentPlayer;
        if (g.tradesThisTurn >= g.maxOffersPerTurn)
            return <div className="trade-wrap"><div className="trade-lab">Player offer — already made this turn</div></div>;
        const me = g.players[cur].resources;
        return (
            <div className="trade-wrap">
                <div className="trade-lab">Offer a player — start with a card to give</div>
                <div className="row">
                    {RESOURCES.map((r) => (
                        <button key={r} className="trade" data-offer-addgive={r} disabled={me[r] < 1} style={cvar(RES_COLOR[r])}
                            onClick={() => act({ player: composer(), action: { type: 'offerAddGive', resource: r } })}>{RES_ABBR[r]}</button>
                    ))}
                </div>
            </div>
        );
    }

    // The composer is the draft's owner once one exists, else the active player
    // who is starting the offer (the first offerAddGive creates the draft).
    const composer = () => sessionRef.current.game.draftOffer?.by ?? sessionRef.current.game.currentPlayer;
    const traderNow = () => negotiationActor(sessionRef.current.game)!;

    /** A draft is open: the running bundle + build-up, then broadcast (initial)
     *  or send-counter (a responder's counter), or cancel. */
    function DraftBuilder() {
        const d = g.draftOffer!;
        const me = g.players[d.by].resources;
        const giveFull = bundleTotalOf(d.give) >= 5;   // GIVE_CAP
        const wantFull = bundleTotalOf(d.want) >= 3;   // WANT_CAP
        const ready = bundleTotalOf(d.give) > 0 && bundleTotalOf(d.want) > 0;
        return (
            <div className="trade-wrap">
                <div className="trade-lab">You give</div>
                <div className="hint">{bundleStr(d.give)}</div>
                <div className="row">
                    {RESOURCES.map((r) => (
                        <button key={r} className="trade" data-offer-addgive={r}
                            disabled={!(!giveFull && (d.give[r] ?? 0) < (me[r] ?? 0))} style={cvar(RES_COLOR[r])}
                            onClick={() => act({ player: composer(), action: { type: 'offerAddGive', resource: r } })}>+{RES_ABBR[r]}</button>
                    ))}
                </div>
                <div className="trade-lab">You want</div>
                <div className="hint">{bundleStr(d.want)}</div>
                <div className="row">
                    {RESOURCES.map((r) => (
                        <button key={r} className="trade" data-offer-addwant={r} disabled={wantFull} style={cvar(RES_COLOR[r])}
                            onClick={() => act({ player: composer(), action: { type: 'offerAddWant', resource: r } })}>+{RES_ABBR[r]}</button>
                    ))}
                </div>
                <div className="row">
                    {g.negotiation
                        ? <button disabled={!ready} onClick={() => act({ player: composer(), action: { type: 'submitCounter' } })}>Send counter</button>
                        : <button disabled={!ready} onClick={() => act({ player: composer(), action: { type: 'offerBroadcast' } })}>Offer to all</button>}
                    <button className="ghost" onClick={() => act({ player: composer(), action: { type: 'offerCancel' } })}>Cancel</button>
                </div>
            </div>
        );
    }

    /** A responder's choices for the broadcast offer: accept / counter / decline. */
    function Respond({ responder }: { responder: PlayerId }) {
        const neg = g.negotiation!;
        const canAccept = legalActions(g, responder).some((a) => a.type === 'respondAccept');
        return (
            <>
                <div className="hint">{nameOf(g, neg.proposer)} offers you {bundleStr(neg.give)} for your {bundleStr(neg.want)}.</div>
                <div className="row">
                    <button disabled={!canAccept} onClick={() => act({ player: traderNow(), action: { type: 'respondAccept' } })}>Accept</button>
                    <button onClick={() => act({ player: traderNow(), action: { type: 'counterStart' } })}>Counter</button>
                    <button className="ghost" onClick={() => act({ player: traderNow(), action: { type: 'respondReject' } })}>Decline</button>
                </div>
            </>
        );
    }

    /** The proposer's arbitration: every opponent's reply + a trade button. */
    function Arbitrate({ proposer }: { proposer: PlayerId }) {
        const neg = g.negotiation!;
        const can = (p: PlayerId) => legalActions(g, proposer).some((a) => a.type === 'confirmTrade' && a.to === p);
        return (
            <>
                <div className="trade-lab">Responses</div>
                {g.turnOrder.filter((p) => p !== proposer).map((p) => {
                    const r = neg.responses[p];
                    if (r === 'accept')
                        return (
                            <div className="row" key={p}>
                                <span className="hint">{nameOf(g, p)} accepts — you give {bundleStr(neg.give)}, get {bundleStr(neg.want)}</span>
                                <button disabled={!can(p)} onClick={() => act({ player: traderNow(), action: { type: 'confirmTrade', to: p } })}>Trade</button>
                            </div>
                        );
                    if (r === 'counter') {
                        const c = neg.counters[p];
                        return (
                            <div className="row" key={p}>
                                <span className="hint">{nameOf(g, p)} counters — you give {bundleStr(c.want)}, get {bundleStr(c.give)}</span>
                                <button disabled={!can(p)} onClick={() => act({ player: traderNow(), action: { type: 'confirmTrade', to: p } })}>Trade</button>
                            </div>
                        );
                    }
                    return <div className="row" key={p}><span className="hint">{nameOf(g, p)} declined</span></div>;
                })}
                <div className="row"><button className="ghost" onClick={() => act({ player: traderNow(), action: { type: 'declineAll' } })}>Decline all</button></div>
            </>
        );
    }

    function Discard() {
        const owe = Object.keys(g.pendingDiscards) as PlayerId[];
        const adjust = (p: PlayerId, r: Resource, delta: number) => {
            const s = sessionRef.current;
            const sel = { ...(s.discardSel[p] ?? {}) };
            sel[r] = Math.max(0, (sel[r] ?? 0) + delta);
            commit({ ...s, discardSel: { ...s.discardSel, [p]: sel } });
        };
        const confirm = (p: PlayerId) => {
            const s = sessionRef.current;
            const resources = s.discardSel[p] ?? {};
            const phase = s.game.phase;
            const r = applyMove(s.game, { player: p, action: { type: 'discard', resources } });
            if (!r.ok) { toast(r.error); return; }
            playForAction({ type: 'discard', resources });
            hapticsForAction({ type: 'discard', resources });
            commit({
                ...s,
                game: r.state,
                moves: [...s.moves, { player: p, phase, action: { type: 'discard', resources } }],
                discardSel: { ...s.discardSel, [p]: {} },
            });
        };
        return (
            <>
                <div className="hint">A 7 was rolled — players over 7 cards discard half.</div>
                {owe.map((p) => {
                    const sel = session.discardSel[p] ?? {};
                    const chosen = RESOURCES.reduce((sum, r) => sum + (sel[r] ?? 0), 0);
                    const need = g.pendingDiscards[p];
                    return (
                        <div className="dcard" key={p}>
                            <div className="phead"><span className="dot" style={{ background: colorOf(g, p) }} /><b>{nameOf(g, p)}</b> must discard {need} ({chosen}/{need})</div>
                            <div className="hand">
                                {RESOURCES.map((r) => {
                                    const have = g.players[p].resources[r];
                                    const picked = sel[r] ?? 0;
                                    return (
                                        <span className="dres" key={r} style={cvar(RES_COLOR[r])}>
                                            {RES_ABBR[r]} {picked}/{have}
                                            <button disabled={picked <= 0} onClick={() => adjust(p, r, -1)}>−</button>
                                            <button disabled={picked >= have || chosen >= need} onClick={() => adjust(p, r, +1)}>+</button>
                                        </span>
                                    );
                                })}
                            </div>
                            <button disabled={chosen !== need} onClick={() => confirm(p)}>Confirm discard</button>
                        </div>
                    );
                })}
            </>
        );
    }

    function PlayersPanel() {
        return (
            <>
                {g.turnOrder.map((p) => {
                    const pl = g.players[p];
                    const isCur = p === g.currentPlayer;
                    const isYou = session.botLevel !== 'human' && p === session.humanSeat;
                    const vp = victoryPoints(g, p);
                    const devCount = pl.devCards.length + pl.pendingDevCards.length;
                    return (
                        <div className={`pcard${isCur ? ' cur' : ''}`} key={p}>
                            <div className="phead">
                                <span className="dot" style={{ background: PLAYER[pl.color].fill }} />
                                <span className="pname">{PLAYER[pl.color].name}{isYou && <span className="badge">you</span>}</span>
                                <span className="vp" title="victory points">{vp}<small> VP</small></span>
                            </div>
                            <div className="hand">
                                {RESOURCES.map((r) => (
                                    <span className="res" key={r} style={cvar(RES_COLOR[r])} title={r}>{RES_ABBR[r]}<b>{pl.resources[r]}</b></span>
                                ))}
                            </div>
                            <div className="meta">
                                Dev: {devCount} · Knights: {pl.playedKnights} · Pieces: {pl.supply.settlements}s/{pl.supply.cities}c/{pl.supply.roads}r{' '}
                                {g.longestRoad?.player === p && <span className="badge">Longest road</span>}
                                {g.largestArmy?.player === p && <span className="badge">Largest army</span>}
                            </div>
                        </div>
                    );
                })}
            </>
        );
    }

    // Ember sparks scattered around the victory card — regenerated per game.
    const sparks = useMemo(
        () => Array.from({ length: 9 }, () => ({
            left: `${8 + Math.random() * 84}%`,
            top: `${55 + Math.random() * 40}%`,
            animationDelay: `${Math.random() * 2.4}s`,
        })),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [g.winner, session.gameSeed],
    );

    return (
        <>
            <div className="wrap">
                <div className="topbar">
                    <h1>Catan<small>the table</small></h1>
                    <label>seed <input id="seed" type="number" value={seedText}
                        onChange={(e) => { setSeedText(e.target.value); seedDirty.current = true; }} /></label>
                    <label>players
                        <select id="playercount" value={playerCount} onChange={(e) => setPlayerCount(e.target.value)}>
                            <option>2</option><option>3</option><option>4</option>
                        </select>
                    </label>
                    <label>opponent
                        <select id="opponent" value={opponent} onChange={(e) => {
                            const v = e.target.value as BotLevel | 'human';
                            setOpponent(v);
                            // AI games: nudge a lone 1v1 up to a full 4-player table by default.
                            if (v !== 'human' && playerCount === '2') setPlayerCount('4');
                        }}>
                            <option value="human">humans (pass &amp; play)</option>
                            <option value="mcts">AI — strong (net + search)</option>
                            <option value="net">AI — fast (raw net)</option>
                        </select>
                    </label>
                    <label title="House rule: identical number tokens never on adjacent tiles">
                        <input id="nosame" type="checkbox" checked={noSame} onChange={(e) => setNoSame(e.target.checked)} /> no same numbers adjacent
                    </label>
                    <button id="newgame" onClick={() => void newGame()}>New game</button>
                    <SfxControl />
                    <a className="viewer-link" href="/">board viewer →</a>
                </div>
                <div className="layout">
                    <div className="panel boardpanel">
                        <BoardSvg
                            board={g.board}
                            robber={g.robber}
                            roads={g.roads}
                            buildings={g.buildings}
                            colorOf={boardColorOf}
                            highlightColor={colorOf(g, g.currentPlayer)}
                            vTargets={vTargetsArr}
                            eTargets={eTargetsArr}
                            tTargets={tTargetsArr}
                            producingTiles={producing?.tiles ?? EMPTY_TILES}
                            onVertex={(v) => interactions.vClicks.get(v)?.()}
                            onEdge={(e) => interactions.eClicks.get(e)?.()}
                            onTile={(t) => interactions.tClicks.get(t)?.()}
                        />
                    </div>
                    <div className="actioncol">
                        <div className="panel actionpanel"><h2>Actions</h2><div id="actions">{ActionsPanel()}</div></div>
                    </div>
                    <div className="sidebar">
                        <div className="panel"><h2>Players</h2><div id="players">{PlayersPanel()}</div></div>
                        <div className="panel chronicle">
                            <h2>Chronicle <small id="logstats">turn {session.turnCount} · {session.moveCount} moves</small></h2>
                            <ul id="log">{session.log.map((l, i) => <li key={`${session.log.length - i}`}>{l}</li>)}</ul>
                        </div>
                    </div>
                </div>
            </div>
            <DiceOverlay roll={dice} />
            <div id="victory" className={g.winner && !victoryDismissed ? 'show' : ''}>
                {g.winner && (
                    <div className="vcard">
                        <div className="vlaurel">🏆</div>
                        <div className="vname" id="vname" style={{ color: PLAYER[g.players[g.winner].color].fill }}>{PLAYER[g.players[g.winner].color].name}</div>
                        <div className="vsub">takes the island</div>
                        <div className="vstats">
                            <div className="vstat"><span id="vvp">{victoryPoints(g, g.winner)}</span><small>victory points</small></div>
                            <div className="vstat"><span id="vturns">{session.turnCount}</span><small>turns</small></div>
                            <div className="vstat"><span id="vmoves">{session.moveCount}</span><small>moves</small></div>
                        </div>
                        <button id="vagain" onClick={() => { setVictoryDismissed(true); void newGame(); }}>Play again</button>
                        <a className="vlink" href="replay.html">View stats &amp; replay →</a>
                        {sparks.map((s, i) => <span className="spark" key={i} style={s} />)}
                    </div>
                )}
            </div>
            {toastEl}
        </>
    );
}
