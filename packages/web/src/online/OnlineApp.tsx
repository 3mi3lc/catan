// Online multiplayer client (online.html) — React port of the vanilla
// online.ts. The server stays fully authoritative: this client renders the
// redacted ClientGameState it receives, and every game action is just a
// socket `move` emit — legality is the server's problem (`myLegalActions`
// only drives what's clickable). What changed in the port is purely the
// rendering/wiring layer: module-level state + innerHTML templates became
// React state and components.
//
// Everything the `gameState` socket event updates atomically (game, legal
// actions, transient UI modes, log, turn count) lives in ONE `Gs` object so
// a single commit preserves the vanilla handler's all-at-once semantics.
// Socket handlers and async callbacks read current values through refs —
// the vanilla code read live module variables, and closures over
// render-time state would be stale in exactly those places.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { io, type Socket } from 'socket.io-client';
import {
    RESOURCES,
    type ClientGameState, type RoomSummary, type ClientToServerEvents, type ServerToClientEvents,
    type AiLevel, type Move, type Action, type GameEvent,
    type PlayerId, type VertexId, type EdgeId, type TileId,
    type Resource, type PlayerColor, type Board,
} from '@catan/core';
import BoardSvg from '../shared/BoardSvg';
import { producingTilesForRoll } from '../game/boardArt';
import DiceOverlay, { type DiceRoll } from '../shared/DiceOverlay';
import SfxControl from '../shared/SfxControl';
import { useToast } from '../shared/useToast';
import { preloadSfx, installClickSfx, playForEvents, playForAction, play } from '../audio/sound';
import { hapticsForEvents, hapticsForAction } from '../haptics';
import { authClient, getCurrentUser, type AuthUser } from '../auth-client';
import { SERVER_URL } from '../server-url';
import './online.css';

// ---------------------------------------------------------------------------
// Constants + persistent identity (module scope, same as vanilla)
// ---------------------------------------------------------------------------
const RES_COLOR: Record<Resource, string> = {
    brick: '#c5673a', lumber: '#3d7a4a', wool: '#9dbe5a', grain: '#e3b23c', ore: '#98a0aa',
};
const RES_ABBR: Record<Resource, string> = { brick: 'Br', lumber: 'Lu', wool: 'Wo', grain: 'Gr', ore: 'Or' };
const PLAYER: Record<PlayerColor, { fill: string; ink: string; name: string }> = {
    red: { fill: '#d24a36', ink: '#fff', name: 'Red' },
    blue: { fill: '#3f7cc4', ink: '#fff', name: 'Blue' },
    white: { fill: '#e9e4d6', ink: '#2c2620', name: 'White' },
    orange: { fill: '#e08c2e', ink: '#fff', name: 'Orange' },
};
const SEAT_COLORS: PlayerColor[] = ['red', 'blue', 'white', 'orange'];

const TOKEN_KEY = 'catan-online-token';
const ROOM_KEY = 'catan-online-room';
const NAME_KEY = 'catan-online-name';

// crypto.randomUUID() only exists in a secure context (HTTPS, or
// http://localhost specifically) — a bare LAN IP over plain HTTP doesn't
// qualify (see packages/web/README.md's mobile live-reload workflow, which
// hits exactly this). This is just a per-browser identity key in
// localStorage, not a security credential (real auth is better-auth session
// cookies), so a non-cryptographic fallback is fine.
function makeClientToken(): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

const playerToken = localStorage.getItem(TOKEN_KEY) ?? makeClientToken();
localStorage.setItem(TOKEN_KEY, playerToken);

const cvar = (c: string) => ({ '--c': c } as React.CSSProperties);

// How long the "just produced" tile pop stays up before clearing — matches
// boardArt.ts's ba-produce-pop-kf animation duration (1.6s), plus a small
// margin so the CSS fade genuinely finishes before the DOM nodes go away.
const PRODUCE_POP_MS = 1700;
// Stable empty-array reference for BoardSvg's producingTiles prop when
// there's nothing to flash — a fresh `[]` literal on every render would
// defeat BoardSvg's own memoization (see its prop doc comment).
const EMPTY_TILES: readonly TileId[] = [];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Mode =
    | { kind: 'normal' }
    | { kind: 'robber'; via: 'seven' | 'knight' }
    | { kind: 'roadbuilding'; edges: EdgeId[] }
    | { kind: 'yearofplenty'; pick: Resource[] }
    | { kind: 'monopoly' };

type Screen = 'boot' | 'landing' | 'lobby' | 'game' | 'mygames' | 'login';

interface GameSummary {
    id: string;
    startedAt: string;
    finishedAt: string;
    winnerSeat: number | null;
    turns: number;
    players: { seatIndex: number; userId: string | null; displayName: string; color: PlayerColor }[];
}

/** Everything the gameState socket event updates atomically. */
interface Gs {
    game: ClientGameState;
    myLegalActions: Action[];
    mode: Mode;
    victimChoice: { tile: TileId; via: 'seven' | 'knight'; victims: PlayerId[] } | null;
    tradeGive: Resource | null;
    discardSel: Partial<Record<Resource, number>>;
    log: string[];
    turnCount: number;
}

// ---------------------------------------------------------------------------
// Pure helpers (ports of the vanilla module functions)
// ---------------------------------------------------------------------------
function mySeat(room: RoomSummary | null, game: ClientGameState | null): PlayerId | null {
    if (!room || !game || room.yourSeat === null) return null;
    return game.turnOrder[room.yourSeat] ?? null;
}
const nameOf = (g: ClientGameState, p: PlayerId) => g.players[p].name;
const colorOf = (g: ClientGameState, p: PlayerId) => PLAYER[g.players[p].color].fill;
const handSizeOf = (g: ClientGameState, p: PlayerId) => g.players[p].handSize;

/** Public-data ports of two @catan/core rules (roadConnected, tradeRatio):
 *  reimplemented here, not imported, because their real signatures take a
 *  full GameState — which the redacted ClientGameState deliberately isn't.
 *  Both only ever read board/buildings/roads, all fully public. */
function roadConnected(board: Board, buildings: ClientGameState['buildings'], roads: Record<EdgeId, PlayerId>, player: PlayerId, edge: EdgeId): boolean {
    return board.edges[edge].vertices.some((v) => {
        const b = buildings[v];
        if (b) return b.owner === player;
        return board.vertices[v].edges.some((e) => e !== edge && roads[e] === player);
    });
}
function tradeRatioFor(g: ClientGameState, player: PlayerId, give: Resource): number {
    let ratio = 4;
    for (const v of Object.keys(g.buildings) as VertexId[]) {
        if (g.buildings[v].owner !== player) continue;
        const port = g.board.vertices[v].port;
        if (!port) continue;
        if (port.kind === '3:1') ratio = Math.min(ratio, 3);
        else if (port.kind === '2:1' && port.resource === give) ratio = Math.min(ratio, 2);
    }
    return ratio;
}
/** Visible VP only: opponents' hidden VP dev-cards aren't counted (real
 *  hidden information). The server's actual win check uses full info. */
function clientVisibleVP(g: ClientGameState, player: PlayerId): number {
    let vp = 0;
    for (const b of Object.values(g.buildings)) if (b.owner === player) vp += b.kind === 'city' ? 2 : 1;
    const pl = g.players[player];
    if (pl.devCards) vp += [...pl.devCards].filter((c) => c === 'victoryPoint').length;
    if (g.longestRoad?.player === player) vp += 2;
    if (g.largestArmy?.player === player) vp += 2;
    return vp;
}

function bundleTotalOf(b: Partial<Record<Resource, number>>): number {
    return RESOURCES.reduce((s, r) => s + (b[r] ?? 0), 0);
}
function bundleStr(b: Partial<Record<Resource, number>>): string {
    return (Object.keys(b) as Resource[])
        .filter((r) => (b[r] ?? 0) > 0)
        .map((r) => `${b[r]} ${RES_ABBR[r]}`).join(' + ') || '—';
}

function describe(g: ClientGameState, ev: GameEvent): string {
    switch (ev.type) {
        case 'diceRolled': return `${nameOf(g, g.currentPlayer)} rolled ${ev.dice[0]} + ${ev.dice[1]} = ${ev.total}`;
        case 'resourcesProduced': {
            const parts = Object.entries(ev.gains).map(([p, gn]) => {
                const items = RESOURCES.filter((r) => gn[r]).map((r) => `+${gn[r]} ${r}`).join(', ');
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

function phaseLabel(g: ClientGameState): string {
    return {
        setupSettlement: 'initial settlement', setupRoad: 'initial road', roll: 'roll the dice',
        discard: 'discard', moveRobber: 'move robber', main: 'build & trade', gameOver: 'game over',
    }[g.phase];
}

function robberVictims(g: ClientGameState, mine: PlayerId, tile: TileId): PlayerId[] {
    const set = new Set<PlayerId>();
    for (const v of g.board.tiles[tile].vertices) {
        const b = g.buildings[v];
        if (b && b.owner !== mine && handSizeOf(g, b.owner) > 0) set.add(b.owner);
    }
    return [...set];
}

/** Which step (if any) of the player-trade flow is mine to act on right now —
 *  derived purely from the (fully public) draftOffer/negotiation fields. */
function myTradeStep(g: ClientGameState, mine: PlayerId | null): 'compose' | 'respond' | 'arbitrate' | null {
    if (!mine) return null;
    if (g.draftOffer?.by === mine) return 'compose';
    const neg = g.negotiation;
    if (!neg) return null;
    if (neg.stage === 'responding' && neg.responses[mine] === 'pending') return 'respond';
    if (neg.stage === 'arbitrating' && neg.proposer === mine) return 'arbitrate';
    return null;
}

function tradeStatusLine(g: ClientGameState): string | null {
    if (g.draftOffer) return `${nameOf(g, g.draftOffer.by)} is composing an offer…`;
    const neg = g.negotiation;
    if (!neg) return null;
    if (neg.stage === 'responding') {
        const pending = g.turnOrder.filter((p) => p !== neg.proposer && neg.responses[p] === 'pending');
        return pending.length ? `Waiting on ${pending.map((p) => nameOf(g, p)).join(', ')} to reply to ${nameOf(g, neg.proposer)}'s offer…` : null;
    }
    return `${nameOf(g, neg.proposer)} is deciding who to trade with…`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function OnlineApp() {
    const [screen, setScreenState] = useState<Screen>('boot');
    const screenRef = useRef(screen);
    const setScreen = (s: Screen) => { screenRef.current = s; setScreenState(s); };

    const [room, setRoomState] = useState<RoomSummary | null>(null);
    const roomRef = useRef(room);
    const setRoom = (r: RoomSummary | null) => { roomRef.current = r; setRoomState(r); };

    const [gs, setGsState] = useState<Gs | null>(null);
    const gsRef = useRef(gs);
    const commitGs = (next: Gs | null) => { gsRef.current = next; setGsState(next); };

    const socketRef = useRef<Socket<ServerToClientEvents, ClientToServerEvents> | null>(null);
    const wasMyTurn = useRef(false);
    const sawFirstState = useRef(false);

    const [dice, setDice] = useState<DiceRoll | null>(null);
    const diceNonce = useRef(0);

    // Tiles to flash "just produced" on for the most recent roll (see
    // BoardSvg's producingTiles prop) — cleared after PRODUCE_POP_MS, guarded
    // by nonce so a stale timeout from an earlier roll can't clear a newer
    // one's highlight if rolls somehow land less than that far apart.
    const [producing, setProducing] = useState<{ tiles: readonly TileId[]; nonce: number } | null>(null);
    const producingNonce = useRef(0);
    const flashProducingTiles = (board: Board, robber: TileId, total: number) => {
        const tiles = producingTilesForRoll(board, robber, total);
        if (!tiles.length) return;
        const nonce = ++producingNonce.current;
        setProducing({ tiles, nonce });
        setTimeout(() => setProducing((p) => (p?.nonce === nonce ? null : p)), PRODUCE_POP_MS);
    };

    const { toastEl, toast } = useToast();
    const toastRef = useRef(toast);
    toastRef.current = toast;

    // Account state
    const [authUser, setAuthUser] = useState<AuthUser | null>(null);
    const authUserRef = useRef(authUser);
    const setAuthUserB = (u: AuthUser | null) => { authUserRef.current = u; setAuthUser(u); };
    const [authScreenMode, setAuthScreenMode] = useState<'login' | 'signup'>('login');
    const [authError, setAuthError] = useState<string | null>(null);
    const [authBusy, setAuthBusy] = useState(false);
    // Where to land after a successful sign-in/sign-up — 'mygames' when the
    // screen was reached by clicking "My games" while logged out.
    const authRedirect = useRef<'landing' | 'mygames'>('landing');
    const [myGames, setMyGames] = useState<GameSummary[] | null>(null);
    const [myGamesError, setMyGamesError] = useState<string | null>(null);

    // Landing form state (both name inputs prefill from the remembered name).
    const storedName = useRef(localStorage.getItem(NAME_KEY) ?? '');
    const [hostName, setHostName] = useState(storedName.current);
    const [joinName, setJoinName] = useState(storedName.current);
    const [joinCode, setJoinCode] = useState('');
    const [hostCount, setHostCount] = useState('4');
    // Seat configs for seats 1..count-1 (seat 0 is always the host). Resetting
    // on count change matches the vanilla re-render of the seat rows.
    const [seatTypes, setSeatTypes] = useState<string[]>(['open', 'open', 'open']);

    // -----------------------------------------------------------------------
    // Socket lifecycle (once)
    // -----------------------------------------------------------------------
    useEffect(() => {
        preloadSfx();
        installClickSfx();

        // withCredentials so the better-auth session cookie rides along on the
        // handshake — the server resolves it back to a userId and links any
        // seat this socket claims to that account.
        const socket: Socket<ServerToClientEvents, ClientToServerEvents> =
            io(SERVER_URL, { transports: ['websocket'], withCredentials: true });
        socketRef.current = socket;

        socket.on('connect_error', (err) => toastRef.current(`Connection error: ${err.message}`));

        socket.on('roomUpdate', (summary) => setRoom(summary));

        socket.on('gameState', (state, legal, events) => {
            playForEvents(events);
            hapticsForEvents(events);
            // "Your turn" cue — only on a genuine hand-off (not the very first
            // state we ever see, which would otherwise ding on page load if you
            // happen to already be up).
            const myId = mySeat(roomRef.current, state);
            const isMyTurn = myId === state.currentPlayer;
            if (sawFirstState.current && isMyTurn && !wasMyTurn.current) play('yourTurn');
            wasMyTurn.current = isMyTurn;
            sawFirstState.current = true;

            const prev = gsRef.current;
            let log = prev?.log ?? [];
            let turnCount = prev?.turnCount ?? 0;
            for (const ev of events) {
                log = [describe(state, ev), ...log];
                if (ev.type === 'diceRolled') {
                    setDice({ a: ev.dice[0], b: ev.dice[1], nonce: ++diceNonce.current });
                    flashProducingTiles(state.board, state.robber, ev.total);
                    turnCount++;
                }
            }
            log = log.slice(0, 60);
            const entering = screenRef.current !== 'game';
            if (entering) log = [`Game started · ${state.turnOrder.length} players`];
            commitGs({
                game: state,
                myLegalActions: legal,
                mode: { kind: 'normal' },
                victimChoice: null,
                tradeGive: null,
                discardSel: {},
                log,
                turnCount,
            });
            if (entering) setScreen('game');
        });

        // The server builds and broadcasts the archive once, right when the
        // game ends — stash it for the post-game stats/replay screen
        // (replay.html), mirroring hotseat's client-built equivalent.
        socket.on('gameOver', (archive) => {
            sessionStorage.setItem('catan-last-archive', JSON.stringify(archive));
        });

        socket.on('actionError', (msg) => toastRef.current(msg));

        // A reload (or a dropped connection coming back) should drop you back
        // into your game, not an empty landing page — the playerToken already
        // lets the server reattach you to whatever seat you held.
        const roomId = new URLSearchParams(location.search).get('room') ?? localStorage.getItem(ROOM_KEY);
        if (!roomId) {
            setScreen('landing');
        } else {
            socket.emit('joinRoom', { roomId, playerToken, name: storedName.current || 'Player' }, (res) => {
                if ('error' in res) { localStorage.removeItem(ROOM_KEY); setScreen('landing'); return; }
                setRoom(res);
                localStorage.setItem(ROOM_KEY, roomId);
                history.replaceState(null, '', `?room=${roomId}`);
                setScreen('lobby');
            });
        }

        void (async () => { setAuthUserB(await getCurrentUser()); })();

        return () => { socket.close(); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const act = (move: Move): void => { socketRef.current?.emit('move', { move }); };

    // -----------------------------------------------------------------------
    // Account flows
    // -----------------------------------------------------------------------
    const showAuth = (mode: 'login' | 'signup', redirect: 'landing' | 'mygames' = 'landing') => {
        setAuthScreenMode(mode);
        setAuthError(null);
        authRedirect.current = redirect;
        setScreen('login');
    };

    const showMyGames = () => {
        if (!authUserRef.current) { showAuth('login', 'mygames'); return; }
        setScreen('mygames');
    };

    const submitAuth = async (mode: 'login' | 'signup', email: string, password: string, name?: string) => {
        setAuthBusy(true); setAuthError(null);
        const res = mode === 'signup'
            ? await authClient.signUp.email({ email, password, name: name || 'Player' })
            : await authClient.signIn.email({ email, password });
        setAuthBusy(false);
        if (res.error) {
            // Deliberately the same generic message regardless of better-auth's
            // specific error code (wrong password vs unknown email vs rate
            // limited) — branching the UI on that distinction is exactly the
            // user-enumeration leak the server already avoids.
            setAuthError(mode === 'login' ? 'Could not log in — check your email and password.' : (res.error.message ?? 'Could not create that account.'));
            return;
        }
        setAuthUserB(res.data.user);
        toast(mode === 'signup' ? `Welcome, ${res.data.user.name}!` : `Welcome back, ${res.data.user.name}!`);
        if (authRedirect.current === 'mygames') setScreen('mygames');
        else setScreen('landing');
    };

    const logout = async () => {
        await authClient.signOut();
        setAuthUserB(null);
        if (screenRef.current === 'mygames') setScreen('landing');
        toast('Signed out');
    };

    // My games — fetched on entering the screen (server-side participant
    // check; see GET /games/mine in packages/server).
    useEffect(() => {
        if (screen !== 'mygames') return;
        let alive = true;
        setMyGames(null); setMyGamesError(null);
        void (async () => {
            try {
                const res = await fetch(`${SERVER_URL}/games/mine`, { credentials: 'include' });
                if (!res.ok) throw new Error(res.status === 401 ? 'Please log in to see your games.' : 'Could not load your games.');
                const data = (await res.json()) as GameSummary[];
                if (alive) setMyGames(data);
            } catch (err) {
                if (alive) setMyGamesError(err instanceof Error ? err.message : 'Could not load your games.');
            }
        })();
        return () => { alive = false; };
    }, [screen]);

    // -----------------------------------------------------------------------
    // In-game interactions
    // -----------------------------------------------------------------------
    const game = gs?.game ?? null;
    const mine = mySeat(room, game);

    const resolveRobber = (tile: TileId, via: 'seven' | 'knight'): void => {
        const cur = gsRef.current;
        const myId = mySeat(roomRef.current, cur?.game ?? null);
        if (!cur || !myId) return;
        const victims = robberVictims(cur.game, myId, tile);
        const build = (steal: PlayerId | null): Action =>
            via === 'knight' ? { type: 'playKnight', robberTo: tile, stealFrom: steal } : { type: 'moveRobber', tile, stealFrom: steal };
        if (victims.length <= 1) act({ player: myId, action: build(victims[0] ?? null) });
        else commitGs({ ...cur, victimChoice: { tile, via, victims } });
    };

    const interactions = useMemo(() => {
        const vClicks = new Map<VertexId, () => void>();
        const eClicks = new Map<EdgeId, () => void>();
        const tClicks = new Map<TileId, () => void>();
        if (!gs || !game || !mine) return { vClicks, eClicks, tClicks };
        const cur = mine;
        const mode = gs.mode;

        if (mode.kind === 'robber') {
            for (const t of Object.keys(game.board.tiles) as TileId[])
                if (t !== game.robber) tClicks.set(t, () => resolveRobber(t, mode.via));
            return { vClicks, eClicks, tClicks };
        }

        if (mode.kind === 'roadbuilding') {
            const placed = mode.edges;
            const tempRoads: Record<EdgeId, PlayerId> = { ...game.roads };
            for (const e of placed) tempRoads[e] = cur;
            for (const e of Object.keys(game.board.edges) as EdgeId[]) {
                if (tempRoads[e] || !roadConnected(game.board, game.buildings, tempRoads, cur, e)) continue;
                eClicks.set(e, () => {
                    const edges = [...placed, e];
                    if (edges.length === 2) act({ player: cur, action: { type: 'playRoadBuilding', edges: [edges[0], edges[1]] } });
                    else { const s = gsRef.current!; commitGs({ ...s, mode: { kind: 'roadbuilding', edges } }); }
                });
            }
            return { vClicks, eClicks, tClicks };
        }

        // Normal phases: the server already tells us exactly what we may do.
        for (const a of gs.myLegalActions) {
            if (a.type === 'buildSettlement' || a.type === 'buildCity') vClicks.set(a.vertex, () => act({ player: cur, action: a }));
            else if (a.type === 'buildRoad') eClicks.set(a.edge, () => act({ player: cur, action: a }));
            else if (a.type === 'moveRobber' && !tClicks.has(a.tile)) tClicks.set(a.tile, () => resolveRobber(a.tile, 'seven'));
        }
        return { vClicks, eClicks, tClicks };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [gs, room]);

    const setMode = (mode: Mode) => { const s = gsRef.current; if (s) commitGs({ ...s, mode }); };

    // BoardSvg's own useMemo is keyed on these by reference — an inline arrow
    // or a `[...map.keys()]` spread written directly in JSX is a brand-new
    // object on every render regardless of whether the board actually
    // changed, which defeated that memo entirely (forcing a full SVG
    // string rebuild, and the browser to redecode every tile image, on every
    // unrelated re-render — e.g. the toast timer clearing). Stabilizing the
    // references here means BoardSvg only redoes that work when the game
    // state or legal-target set genuinely changed.
    const boardColorOf = useCallback((p: PlayerId) => (game ? colorOf(game, p) : ''), [game]);
    const vTargetsArr = useMemo(() => [...interactions.vClicks.keys()], [interactions]);
    const eTargetsArr = useMemo(() => [...interactions.eClicks.keys()], [interactions]);
    const tTargetsArr = useMemo(() => [...interactions.tClicks.keys()], [interactions]);

    // -----------------------------------------------------------------------
    // Screens
    // -----------------------------------------------------------------------
    function Landing() {
        const seatCount = Number(hostCount);
        return (
            <div className="landing">
                <div className="panel landing-card">
                    <h2>Host a game</h2>
                    <label>Your name <input id="host-name" type="text" value={hostName} placeholder="Your name" onChange={(e) => setHostName(e.target.value)} /></label>
                    <label>Players
                        <select id="host-count" value={hostCount} onChange={(e) => { setHostCount(e.target.value); setSeatTypes(Array.from({ length: Number(e.target.value) - 1 }, () => 'open')); }}>
                            <option>2</option><option>3</option><option>4</option>
                        </select>
                    </label>
                    <div id="host-seats">
                        {Array.from({ length: seatCount }, (_, i) => (
                            i === 0
                                ? <div className="seatrow" key={i}><span className="dot" style={{ background: PLAYER[SEAT_COLORS[i]].fill }} /> Seat {i + 1} — you</div>
                                : (
                                    <div className="seatrow" key={i}>
                                        <span className="dot" style={{ background: PLAYER[SEAT_COLORS[i]].fill }} /> Seat {i + 1}
                                        <select value={seatTypes[i - 1] ?? 'open'} onChange={(e) => {
                                            setSeatTypes((prev) => { const next = [...prev]; next[i - 1] = e.target.value; return next; });
                                        }}>
                                            <option value="open">Open — invite a player</option>
                                            <option value="ai-net">AI — fast</option>
                                            <option value="ai-mcts">AI — strong</option>
                                        </select>
                                    </div>
                                )
                        ))}
                    </div>
                    <button id="host-create" onClick={() => {
                        const name = hostName.trim() || 'Player';
                        storedName.current = name;
                        localStorage.setItem(NAME_KEY, name);
                        const seats: ('open' | { ai: AiLevel })[] = Array.from({ length: seatCount }, (_, i) => {
                            if (i === 0) return 'open';
                            const sel = seatTypes[i - 1] ?? 'open';
                            return sel === 'open' ? 'open' : { ai: (sel === 'ai-mcts' ? 'mcts' : 'net') as AiLevel };
                        });
                        socketRef.current!.emit('createRoom', { name, playerToken, seats }, (res) => {
                            if ('error' in res) { toastRef.current(res.error); return; }
                            localStorage.setItem(ROOM_KEY, res.roomId);
                            history.replaceState(null, '', `?room=${res.roomId}`);
                            setScreen('lobby');
                        });
                    }}>Create room</button>
                </div>
                <div className="panel landing-card">
                    <h2>Join a game</h2>
                    <label>Your name <input id="join-name" type="text" value={joinName} placeholder="Your name" onChange={(e) => setJoinName(e.target.value)} /></label>
                    <label>Room code <input id="join-code" type="text" maxLength={6} style={{ textTransform: 'uppercase' }} value={joinCode} onChange={(e) => setJoinCode(e.target.value)} /></label>
                    <button id="join-room" onClick={() => {
                        const name = joinName.trim() || 'Player';
                        storedName.current = name;
                        localStorage.setItem(NAME_KEY, name);
                        const roomId = joinCode.trim().toUpperCase();
                        if (!roomId) { toastRef.current('Enter a room code'); return; }
                        socketRef.current!.emit('joinRoom', { roomId, playerToken, name }, (res) => {
                            if ('error' in res) { toastRef.current(res.error); return; }
                            setRoom(res);
                            localStorage.setItem(ROOM_KEY, roomId);
                            history.replaceState(null, '', `?room=${roomId}`);
                            setScreen('lobby');
                        });
                    }}>Join room</button>
                </div>
            </div>
        );
    }

    function Lobby() {
        if (!room) return null;
        const allFilled = room.seats.every((s) => s.type !== 'open');
        const amHost = room.yourSeat !== null && room.yourSeat === room.hostSeat;
        const invite = `${location.origin}${location.pathname}?room=${room.roomId}`;
        return (
            <div className="panel lobby-card">
                <h2>Room <span className="roomcode" id="roomcode">{room.roomId}</span></h2>
                <div className="hint">Share this link — <code id="invite-link">{invite}</code>{' '}
                    <button id="copy-link" className="ghost" onClick={() => {
                        navigator.clipboard?.writeText(invite);
                        toast('Link copied');
                    }}>Copy</button>
                </div>
                <div id="lobby-seats">
                    {room.seats.map((s, i) => {
                        const dot = <span className="dot" style={{ background: PLAYER[SEAT_COLORS[i]].fill }} />;
                        const isYou = room.yourSeat === i;
                        const isHost = room.hostSeat === i;
                        if (s.type === 'open') {
                            // Only offer a claim button if you don't already hold a
                            // seat — the server also rejects a second claim, but
                            // hiding the button means you never see an error for
                            // clicking something that was never going to work.
                            return (
                                <div className="seatrow" key={i}>{dot} Seat {i + 1} — open{' '}
                                    {room.yourSeat === null && <button onClick={() => socketRef.current!.emit('claimSeat', { seatIndex: i })}>Claim seat</button>}
                                </div>
                            );
                        }
                        if (s.type === 'ai') return <div className="seatrow" key={i}>{dot} Seat {i + 1} — AI ({s.level === 'mcts' ? 'strong' : 'fast'})</div>;
                        const tag = `${isYou ? ' (you)' : ''}${isHost ? ' — host' : ''}${s.connected ? '' : ' (disconnected)'}`;
                        return <div className="seatrow" key={i}>{dot} Seat {i + 1} — {s.name}{tag}</div>;
                    })}
                </div>
                <div className="row">
                    {amHost && <button id="lobby-start" disabled={!allFilled} onClick={() => socketRef.current!.emit('startGame')}>Start game</button>}
                    <span className="hint" id="lobby-wait">
                        {amHost ? (allFilled ? '' : 'Waiting for every seat to be filled…') : 'Waiting for the host to start the game…'}
                    </span>
                </div>
            </div>
        );
    }

    function AuthScreen() {
        const isSignup = authScreenMode === 'signup';
        return (
            <div className="panel landing-card auth-card">
                <h2>{isSignup ? 'Create account' : 'Log in'}</h2>
                {authError && <div className="auth-error">{authError}</div>}
                <form id="auth-form" onSubmit={(e) => {
                    e.preventDefault();
                    const form = e.currentTarget;
                    const email = (form.querySelector('#auth-email') as HTMLInputElement).value.trim();
                    const password = (form.querySelector('#auth-password') as HTMLInputElement).value;
                    const name = isSignup ? (form.querySelector('#auth-name') as HTMLInputElement).value.trim() : undefined;
                    void submitAuth(isSignup ? 'signup' : 'login', email, password, name);
                }}>
                    <label>Email <input id="auth-email" type="email" autoComplete="email" required /></label>
                    {isSignup && <label>Display name <input id="auth-name" type="text" autoComplete="name" required /></label>}
                    <label>Password <input id="auth-password" type="password" minLength={10} autoComplete={isSignup ? 'new-password' : 'current-password'} required /></label>
                    <div className="row">
                        <button type="submit" disabled={authBusy}>{authBusy ? 'Please wait…' : (isSignup ? 'Sign up' : 'Log in')}</button>
                        <button type="button" className="ghost" id="auth-back" onClick={() => setScreen('landing')}>← Back</button>
                    </div>
                </form>
                <div className="hint small">
                    {isSignup ? 'Already have an account? ' : 'Need an account? '}
                    <a href="#" id="auth-switch" onClick={(e) => { e.preventDefault(); showAuth(isSignup ? 'login' : 'signup', authRedirect.current); }}>
                        {isSignup ? 'Log in' : 'Sign up'}
                    </a>
                </div>
            </div>
        );
    }

    function MyGamesScreen() {
        const back = <button className="ghost" id="mygames-back" onClick={() => setScreen('landing')}>← Back</button>;
        if (myGamesError) return <div className="panel landing-card"><h2>My games</h2><p className="hint">{myGamesError}</p>{back}</div>;
        if (!myGames) return <div className="panel landing-card"><h2>My games</h2><p className="hint">Loading…</p></div>;
        if (!myGames.length) return <div className="panel landing-card"><h2>My games</h2><p className="hint">No finished games yet — they'll show up here once you complete an online game.</p>{back}</div>;
        return (
            <div className="panel landing-card mygames-card">
                <h2>My games</h2>
                <div id="mygames-list">
                    {myGames.map((gm) => {
                        const winner = gm.winnerSeat !== null ? gm.players.find((p) => p.seatIndex === gm.winnerSeat) : null;
                        const opponents = gm.players.map((p) => p.displayName).join(', ');
                        const when = new Date(gm.finishedAt).toLocaleString();
                        return (
                            <div className="seatrow mygame-row" key={gm.id} onClick={() => { location.href = `/replay.html?game=${gm.id}`; }}>
                                <span className="dot" style={{ background: winner ? PLAYER[winner.color].fill : '#888' }} />
                                <div>
                                    <div>{opponents}</div>
                                    <div className="hint small">{when} · {gm.turns} turns{winner ? ` · ${winner.displayName} won` : ''}</div>
                                </div>
                            </div>
                        );
                    })}
                </div>
                {back}
            </div>
        );
    }

    // -----------------------------------------------------------------------
    // Game screen panels
    // -----------------------------------------------------------------------
    function ActionsPanel() {
        const g = game!;
        if (!mine) return <div className="hint">Spectating.</div>;
        if (g.winner) return <div className="won">{nameOf(g, g.winner)} wins the game! 🏆</div>;

        const owe = g.pendingDiscards[mine] ?? 0;
        if (owe > 0) return <><div className="turnline"><b>Discard</b></div><div className="ctx">{Discard()}</div></>;
        if (g.phase === 'discard') {
            const owing = Object.keys(g.pendingDiscards) as PlayerId[];
            return <div className="turnline"><span className="hint">Waiting on {owing.map((p) => nameOf(g, p)).join(', ')} to discard…</span></div>;
        }

        const step = myTradeStep(g, mine);
        if (step) {
            const turnline = (label: string) => (
                <div className="turnline"><span className="dot" style={{ background: colorOf(g, mine!) }} /> <b>You</b> — {label}</div>
            );
            if (step === 'compose') return <>{turnline(g.negotiation ? 'counter-offer' : 'compose offer')}<div className="ctx">{DraftBuilder()}</div></>;
            if (step === 'respond') return <>{turnline('respond to offer')}<div className="ctx">{Respond()}</div></>;
            return <>{turnline('choose a trade')}<div className="ctx">{Arbitrate()}</div></>;
        }
        const status = tradeStatusLine(g);
        if (status) return <div className="turnline"><span className="hint">{status}</span></div>;

        const cur = g.currentPlayer;
        const isMyTurn = cur === mine;
        const has = (t: Action['type']) => gs!.myLegalActions.some((a) => a.type === t);
        const busy = !!gs!.victimChoice || gs!.mode.kind === 'robber' || gs!.mode.kind === 'roadbuilding'
            || gs!.mode.kind === 'monopoly' || gs!.mode.kind === 'yearofplenty' || g.phase === 'moveRobber' || !isMyTurn;

        const btn = (legal: boolean, onClick: () => void, label: string) => (
            <button disabled={!(legal && !busy)} onClick={onClick}>{label}</button>
        );
        const core = (
            <>
                <div className="row">
                    {btn(has('rollDice'), () => act({ player: cur, action: { type: 'rollDice' } }), 'Roll dice')}
                    {btn(has('buyDevCard'), () => act({ player: cur, action: { type: 'buyDevCard' } }), 'Buy dev card')}
                    {btn(has('endTurn'), () => act({ player: cur, action: { type: 'endTurn' } }), 'End turn')}
                </div>
                <div className="row">
                    {btn(has('playKnight'), () => setMode({ kind: 'robber', via: 'knight' }), 'Knight')}
                    {btn(has('playRoadBuilding'), () => setMode({ kind: 'roadbuilding', edges: [] }), 'Road Building')}
                    {btn(has('playYearOfPlenty'), () => setMode({ kind: 'yearofplenty', pick: [] }), 'Year of Plenty')}
                    {btn(has('playMonopoly'), () => setMode({ kind: 'monopoly' }), 'Monopoly')}
                </div>
            </>
        );
        const turnline = (
            <div className="turnline"><span className="dot" style={{ background: colorOf(g, cur) }} /> <b>{nameOf(g, cur)}</b> — {phaseLabel(g)}</div>
        );

        if (!isMyTurn) {
            const seatIdx = g.turnOrder.indexOf(cur);
            const isAiTurn = room?.seats[seatIdx]?.type === 'ai';
            return <>{turnline}{core}<div className="ctx"><div className="hint">{isAiTurn ? `🤖 ${nameOf(g, cur)} is thinking…` : `Waiting for ${nameOf(g, cur)}…`}</div></div></>;
        }
        return <>{turnline}{core}<div className="ctx">{Context()}</div></>;
    }

    function Context() {
        const g = game!;
        const cur = mine!;
        const cancelBtn = <button className="ghost" onClick={() => setMode({ kind: 'normal' })}>Cancel</button>;
        if (gs!.victimChoice) {
            return (
                <>
                    <div className="hint">Steal from whom?</div>
                    <div className="row">
                        {gs!.victimChoice.victims.map((vp) => (
                            <button key={vp} onClick={() => {
                                const vc = gsRef.current?.victimChoice;
                                if (!vc) return;
                                const a: Action = vc.via === 'knight'
                                    ? { type: 'playKnight', robberTo: vc.tile, stealFrom: vp }
                                    : { type: 'moveRobber', tile: vc.tile, stealFrom: vp };
                                act({ player: cur, action: a });
                            }}>{nameOf(g, vp)} ({handSizeOf(g, vp)})</button>
                        ))}
                    </div>
                </>
            );
        }
        if (gs!.mode.kind === 'robber') return <div className="hint">Click a tile to move the robber.</div>;
        if (g.phase === 'moveRobber') return <div className="hint">Move the robber — click a tile.</div>;
        if (gs!.mode.kind === 'roadbuilding')
            return <><div className="hint">Road Building — pick {2 - gs!.mode.edges.length} more road(s).</div><div className="row">{cancelBtn}</div></>;
        if (gs!.mode.kind === 'monopoly')
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
        if (gs!.mode.kind === 'yearofplenty') {
            const picked = gs!.mode.pick;
            return (
                <>
                    <div className="hint">Year of Plenty — choose 2 (picked {picked.length}).</div>
                    <div className="row">
                        {RESOURCES.map((r) => (
                            <button key={r} data-yop={r} style={cvar(RES_COLOR[r])} onClick={() => {
                                const s = gsRef.current!;
                                if (s.mode.kind !== 'yearofplenty') return;
                                const pick = [...s.mode.pick, r];
                                if (pick.length === 2) act({ player: cur, action: { type: 'playYearOfPlenty', take: [pick[0], pick[1]] } });
                                else commitGs({ ...s, mode: { kind: 'yearofplenty', pick } });
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
        if (g.phase === 'main') return <><div className="hint">Build by clicking highlighted spots on the board.</div>{TradePanel()}</>;
        return null;
    }

    function TradePanel() {
        const g = game!;
        const cur = mine!;
        const me = g.players[cur];
        return (
            <>
                <div className="trade-wrap">
                    <div className="trade-lab">Bank trade — give</div>
                    <div className="row">
                        {RESOURCES.map((r) => {
                            const ratio = tradeRatioFor(g, cur, r);
                            const can = (me.resources?.[r] ?? 0) >= ratio;
                            return (
                                <button key={r} className={`trade${gs!.tradeGive === r ? ' sel' : ''}`} data-give={r}
                                    disabled={!can} style={cvar(RES_COLOR[r])}
                                    onClick={() => { const s = gsRef.current!; commitGs({ ...s, tradeGive: r }); }}>{ratio} {RES_ABBR[r]}</button>
                            );
                        })}
                    </div>
                    <div className="trade-lab">receive</div>
                    <div className="row">
                        {gs!.tradeGive
                            ? RESOURCES.filter((r) => r !== gs!.tradeGive && g.bank[r] > 0).map((r) => (
                                <button key={r} className="trade" data-receive={r} style={cvar(RES_COLOR[r])} onClick={() => {
                                    const s = gsRef.current!;
                                    if (!s.tradeGive) return;
                                    const give = s.tradeGive;
                                    const action: Action = { type: 'bankTrade', give, giveCount: tradeRatioFor(s.game, cur, give), receive: r };
                                    act({ player: cur, action });
                                    playForAction(action); // optimistic — bankTrade emits no dedicated GameEvent to react to later
                                    hapticsForAction(action);
                                }}>{RES_ABBR[r]}</button>
                            ))
                            : <span className="hint">pick what to give</span>}
                    </div>
                </div>
                {StartOffer()}
            </>
        );
    }

    function StartOffer() {
        const g = game!;
        const cur = mine!;
        if (g.tradesThisTurn >= g.maxOffersPerTurn)
            return <div className="trade-wrap"><div className="trade-lab">Player offer — already made this turn</div></div>;
        const me = g.players[cur].resources!;
        return (
            <div className="trade-wrap">
                <div className="trade-lab">Offer a player — start with a card to give</div>
                <div className="row">
                    {RESOURCES.map((r) => (
                        <button key={r} className="trade" data-offer-addgive={r} disabled={me[r] < 1} style={cvar(RES_COLOR[r])}
                            onClick={() => act({ player: cur, action: { type: 'offerAddGive', resource: r } })}>{RES_ABBR[r]}</button>
                    ))}
                </div>
            </div>
        );
    }

    function DraftBuilder() {
        const g = game!;
        const cur = mine!;
        const d = g.draftOffer!;
        const me = g.players[d.by].resources!;
        const giveFull = bundleTotalOf(d.give) >= 5;
        const wantFull = bundleTotalOf(d.want) >= 3;
        const ready = bundleTotalOf(d.give) > 0 && bundleTotalOf(d.want) > 0;
        return (
            <div className="trade-wrap">
                <div className="trade-lab">You give</div>
                <div className="hint">{bundleStr(d.give)}</div>
                <div className="row">
                    {RESOURCES.map((r) => (
                        <button key={r} className="trade" data-offer-addgive={r}
                            disabled={!(!giveFull && (d.give[r] ?? 0) < (me[r] ?? 0))} style={cvar(RES_COLOR[r])}
                            onClick={() => act({ player: cur, action: { type: 'offerAddGive', resource: r } })}>+{RES_ABBR[r]}</button>
                    ))}
                </div>
                <div className="trade-lab">You want</div>
                <div className="hint">{bundleStr(d.want)}</div>
                <div className="row">
                    {RESOURCES.map((r) => (
                        <button key={r} className="trade" data-offer-addwant={r} disabled={wantFull} style={cvar(RES_COLOR[r])}
                            onClick={() => act({ player: cur, action: { type: 'offerAddWant', resource: r } })}>+{RES_ABBR[r]}</button>
                    ))}
                </div>
                <div className="row">
                    {g.negotiation
                        ? <button disabled={!ready} onClick={() => act({ player: cur, action: { type: 'submitCounter' } })}>Send counter</button>
                        : <button disabled={!ready} onClick={() => act({ player: cur, action: { type: 'offerBroadcast' } })}>Offer to all</button>}
                    <button className="ghost" onClick={() => act({ player: cur, action: { type: 'offerCancel' } })}>Cancel</button>
                </div>
            </div>
        );
    }

    function Respond() {
        const g = game!;
        const cur = mine!;
        const neg = g.negotiation!;
        const canAccept = gs!.myLegalActions.some((a) => a.type === 'respondAccept');
        return (
            <>
                <div className="hint">{nameOf(g, neg.proposer)} offers you {bundleStr(neg.give)} for your {bundleStr(neg.want)}.</div>
                <div className="row">
                    <button disabled={!canAccept} onClick={() => act({ player: cur, action: { type: 'respondAccept' } })}>Accept</button>
                    <button onClick={() => act({ player: cur, action: { type: 'counterStart' } })}>Counter</button>
                    <button className="ghost" onClick={() => act({ player: cur, action: { type: 'respondReject' } })}>Decline</button>
                </div>
            </>
        );
    }

    function Arbitrate() {
        const g = game!;
        const cur = mine!;
        const neg = g.negotiation!;
        const can = (p: PlayerId) => gs!.myLegalActions.some((a) => a.type === 'confirmTrade' && a.to === p);
        return (
            <>
                <div className="trade-lab">Responses</div>
                {g.turnOrder.filter((p) => p !== neg.proposer).map((p) => {
                    const r = neg.responses[p];
                    if (r === 'accept')
                        return (
                            <div className="row" key={p}>
                                <span className="hint">{nameOf(g, p)} accepts — you give {bundleStr(neg.give)}, get {bundleStr(neg.want)}</span>
                                <button disabled={!can(p)} onClick={() => act({ player: cur, action: { type: 'confirmTrade', to: p } })}>Trade</button>
                            </div>
                        );
                    if (r === 'counter') {
                        const c = neg.counters[p];
                        return (
                            <div className="row" key={p}>
                                <span className="hint">{nameOf(g, p)} counters — you give {bundleStr(c.want)}, get {bundleStr(c.give)}</span>
                                <button disabled={!can(p)} onClick={() => act({ player: cur, action: { type: 'confirmTrade', to: p } })}>Trade</button>
                            </div>
                        );
                    }
                    return <div className="row" key={p}><span className="hint">{nameOf(g, p)} declined</span></div>;
                })}
                <div className="row"><button className="ghost" onClick={() => act({ player: cur, action: { type: 'declineAll' } })}>Decline all</button></div>
            </>
        );
    }

    function Discard() {
        const g = game!;
        const cur = mine!;
        const sel = gs!.discardSel;
        const chosen = RESOURCES.reduce((s, r) => s + (sel[r] ?? 0), 0);
        const need = g.pendingDiscards[cur];
        const have = g.players[cur].resources!;
        const adjust = (r: Resource, delta: number) => {
            const s = gsRef.current!;
            commitGs({ ...s, discardSel: { ...s.discardSel, [r]: Math.max(0, (s.discardSel[r] ?? 0) + delta) } });
        };
        return (
            <>
                <div className="hint">A 7 was rolled — you must discard {need} ({chosen}/{need}).</div>
                <div className="hand">
                    {RESOURCES.map((r) => {
                        const picked = sel[r] ?? 0;
                        return (
                            <span className="dres" key={r} style={cvar(RES_COLOR[r])}>
                                {RES_ABBR[r]} {picked}/{have[r]}
                                <button disabled={picked <= 0} onClick={() => adjust(r, -1)}>−</button>
                                <button disabled={picked >= have[r] || chosen >= need} onClick={() => adjust(r, +1)}>+</button>
                            </span>
                        );
                    })}
                </div>
                <button disabled={chosen !== need} onClick={() => {
                    const s = gsRef.current!;
                    const action: Action = { type: 'discard', resources: s.discardSel };
                    act({ player: cur, action });
                    playForAction(action); // optimistic — discard emits no dedicated GameEvent
                    hapticsForAction(action);
                }}>Confirm discard</button>
            </>
        );
    }

    function PlayersPanel() {
        const g = game!;
        return (
            <>
                {g.turnOrder.map((p, idx) => {
                    const pl = g.players[p];
                    const isCur = p === g.currentPlayer;
                    const isYou = p === mine;
                    const isAi = room?.seats[idx]?.type === 'ai';
                    const vp = clientVisibleVP(g, p);
                    const devCount = pl.devCardCount + pl.pendingDevCardCount;
                    return (
                        <div className={`pcard${isCur ? ' cur' : ''}`} key={p}>
                            <div className="phead">
                                <span className="dot" style={{ background: PLAYER[pl.color].fill }} />
                                <span className="pname">{pl.name}{isYou && <span className="badge">you</span>}</span>
                                <span className="vp" title="victory points">{vp}<small> VP</small></span>
                            </div>
                            <div className="hand">
                                {pl.resources
                                    ? RESOURCES.map((r) => (
                                        <span className="res" key={r} style={cvar(RES_COLOR[r])} title={r}>{RES_ABBR[r]}<b>{pl.resources![r]}</b></span>
                                    ))
                                    : <span className="cardback" title="cards in hand">🂠 <b>{pl.handSize}</b></span>}
                            </div>
                            <div className="meta">
                                Dev: {devCount} · Knights: {pl.playedKnights} · Pieces: {pl.supply.settlements}s/{pl.supply.cities}c/{pl.supply.roads}r{' '}
                                {g.longestRoad?.player === p && <span className="badge">Longest road</span>}
                                {g.largestArmy?.player === p && <span className="badge">Largest army</span>}
                                {isAi && <span className="badge">AI</span>}
                            </div>
                        </div>
                    );
                })}
            </>
        );
    }

    const sparks = useMemo(
        () => Array.from({ length: 9 }, () => ({
            left: `${8 + Math.random() * 84}%`,
            top: `${55 + Math.random() * 40}%`,
            animationDelay: `${Math.random() * 2.4}s`,
        })),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [game?.winner],
    );

    // -----------------------------------------------------------------------
    // Shell
    // -----------------------------------------------------------------------
    return (
        <>
            <div className="wrap">
                <div className="topbar">
                    <h1>Catan<small>online</small></h1>
                    <div id="topbar-right">{room && <>Room <b>{room.roomId}</b></>}</div>
                    <SfxControl />
                    <div id="topbar-auth" className="topbar-auth">
                        {authUser
                            ? (
                                <>
                                    <a className="link" id="mygames-link" href="#" onClick={(e) => { e.preventDefault(); showMyGames(); }}>My games</a>
                                    <span className="hint">{authUser.name}</span>
                                    <button className="ghost" id="logout-btn" onClick={() => void logout()}>Log out</button>
                                </>
                            )
                            : (
                                <>
                                    <button className="ghost" id="login-btn" onClick={() => showAuth('login')}>Log in</button>
                                    <button id="signup-btn" onClick={() => showAuth('signup')}>Sign up</button>
                                </>
                            )}
                    </div>
                </div>
                <div id="main">
                    {screen === 'landing' && Landing()}
                    {screen === 'lobby' && Lobby()}
                    {screen === 'login' && AuthScreen()}
                    {screen === 'mygames' && MyGamesScreen()}
                    {screen === 'game' && gs && game && (
                        <div className="layout">
                            <div className="panel boardpanel">
                                <BoardSvg
                                    board={game.board}
                                    robber={game.robber}
                                    roads={game.roads}
                                    buildings={game.buildings}
                                    colorOf={boardColorOf}
                                    highlightColor={colorOf(game, game.currentPlayer)}
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
                                <div className="panel chronicle"><h2>Chronicle</h2><ul id="log">{gs.log.map((l, i) => <li key={`${gs.log.length - i}`}>{l}</li>)}</ul></div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <DiceOverlay roll={dice} />
            <div id="victory" className={game?.winner ? 'show' : ''}>
                {game?.winner && (
                    <div className="vcard">
                        <div className="vlaurel">🏆</div>
                        <div className="vname" id="vname" style={{ color: PLAYER[game.players[game.winner].color].fill }}>{game.players[game.winner].name}</div>
                        <div className="vsub">takes the island</div>
                        <div className="vstats">
                            <div className="vstat"><span id="vvp">{clientVisibleVP(game, game.winner)}</span><small>victory points</small></div>
                            <div className="vstat"><span id="vturns">{gs?.turnCount ?? 0}</span><small>turns</small></div>
                        </div>
                        <a className="vlink" href="replay.html">View stats &amp; replay →</a>
                        {sparks.map((s, i) => <span className="spark" key={i} style={s} />)}
                    </div>
                )}
            </div>
            {toastEl}
        </>
    );
}
