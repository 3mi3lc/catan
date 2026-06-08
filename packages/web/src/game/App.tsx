import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  generateBoard, initialGameState, applyMove, legalActions,
  roadConnected, tradeRatio, victoryPoints, RESOURCES,
  type GameState, type Move, type Action, type GameEvent,
  type PlayerId, type VertexId, type EdgeId, type TileId, type Resource,
} from '@catan/core';
import Board from './Board';
import {
  RES_COLOR, RES_ABBR, PLAYER, SEAT_DEFS, TERRAIN,
  rngFromSeed, nameOf, colorOf, handSize,
} from './theme';

type Mode =
    | { kind: 'normal' }
    | { kind: 'robber'; via: 'seven' | 'knight' }
    | { kind: 'roadbuilding'; edges: EdgeId[] }
    | { kind: 'yearofplenty'; pick: Resource[] }
    | { kind: 'monopoly' };

type VictimChoice = { tile: TileId; via: 'seven' | 'knight'; victims: PlayerId[] };

function newGameState(seed: number, players: number): GameState {
  const board = generateBoard(rngFromSeed(seed));
  const seats = SEAT_DEFS.slice(0, players).map((s) => ({ id: s.id, name: PLAYER[s.color].name, color: s.color }));
  return initialGameState(board, seats, seed);
}

function describe(ev: GameEvent, g: GameState): string {
  switch (ev.type) {
    case 'diceRolled': return `${nameOf(g, g.currentPlayer)} rolled ${ev.dice[0]} + ${ev.dice[1]} = ${ev.total}`;
    case 'resourcesProduced': {
      const parts = Object.entries(ev.gains).map(([p, gain]) => {
        const items = RESOURCES.filter((r) => gain[r]).map((r) => `+${gain[r]} ${r}`).join(', ');
        return items ? `${nameOf(g, p as PlayerId)}: ${items}` : '';
      }).filter(Boolean);
      return parts.length ? `Production — ${parts.join('; ')}` : 'Production — nobody collected';
    }
    case 'built': return `${nameOf(g, ev.player)} built a ${ev.what}`;
    case 'robberMoved': return `${nameOf(g, g.currentPlayer)} moved the robber${ev.stolen ? ' and stole a card' : ''}`;
    case 'devCardBought': return `${nameOf(g, ev.player)} bought a development card`;
    case 'tradeExecuted': return `Trade: ${nameOf(g, ev.between[0])} ↔ ${nameOf(g, ev.between[1])}`;
    case 'awardMoved': return `${ev.award} → ${nameOf(g, ev.to)}`;
    case 'gameWon': return `${nameOf(g, ev.player)} wins!`;
  }
}

export default function App() {
  const [seed, setSeed] = useState(42);
  const [playerCount, setPlayerCount] = useState(4);
  const [game, setGame] = useState<GameState>(() => newGameState(42, 4));
  const [mode, setMode] = useState<Mode>({ kind: 'normal' });
  const [victim, setVictim] = useState<VictimChoice | null>(null);
  const [tradeGive, setTradeGive] = useState<Resource | null>(null);
  const [discardSel, setDiscardSel] = useState<Record<string, Partial<Record<Resource, number>>>>({});
  const [log, setLog] = useState<string[]>(['New game · seed 42 · 4 players']);
  const [toast, setToast] = useState<string | null>(null);

  const cur = game.currentPlayer;

  const reset = () => { setMode({ kind: 'normal' }); setVictim(null); setTradeGive(null); };

  function apply(move: Move): void {
    const res = applyMove(game, move);
    if (!res.ok) { flash(res.error); return; }
    const events = res.events.map((e) => describe(e, res.state));
    if (res.state.winner) events.push(`${nameOf(res.state, res.state.winner)} wins!`);
    setGame(res.state);
    setLog((prev) => [...events.reverse(), ...prev].slice(0, 50));
    reset();
  }

  let toastTimer: ReturnType<typeof setTimeout>;
  function flash(msg: string): void {
    setToast(msg);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => setToast(null), 2200);
  }

  function startNew(): void {
    setGame(newGameState(seed, playerCount));
    setDiscardSel({});
    setLog([`New game · seed ${seed} · ${playerCount} players`]);
    reset();
  }

  // --- interaction targets for the current phase / mode ---
  const interaction = useMemo(() => {
    const vAction = new Map<VertexId, Action>();
    const eAction = new Map<EdgeId, Action>();
    const tiles = new Set<TileId>();

    if (mode.kind === 'robber') {
      for (const t of Object.keys(game.board.tiles) as TileId[]) if (t !== game.robber) tiles.add(t);
    } else if (mode.kind === 'roadbuilding') {
      const temp: GameState = { ...game, roads: { ...game.roads } };
      for (const e of mode.edges) temp.roads[e] = cur;
      for (const e of Object.keys(game.board.edges) as EdgeId[])
        if (!temp.roads[e] && roadConnected(temp, cur, e)) eAction.set(e, { type: 'buildRoad', edge: e });
    } else {
      for (const a of legalActions(game, cur)) {
        if (a.type === 'buildSettlement' || a.type === 'buildCity') vAction.set(a.vertex, a);
        else if (a.type === 'buildRoad') eAction.set(a.edge, a);
        else if (a.type === 'moveRobber') tiles.add(a.tile);
      }
    }
    return { vAction, eAction, tiles };
  }, [game, mode, cur]);

  function resolveRobber(tile: TileId, via: 'seven' | 'knight'): void {
    const victims = [...new Set(
        game.board.tiles[tile].vertices
            .map((v) => game.buildings[v])
            .filter((b) => b && b.owner !== cur && handSize(game, b.owner) > 0)
            .map((b) => b!.owner),
    )];
    const build = (steal: PlayerId | null): Action =>
        via === 'knight' ? { type: 'playKnight', robberTo: tile, stealFrom: steal } : { type: 'moveRobber', tile, stealFrom: steal };
    if (victims.length <= 1) apply({ player: cur, action: build(victims[0] ?? null) });
    else setVictim({ tile, via, victims });
  }

  const onVertex = (v: VertexId) => { const a = interaction.vAction.get(v); if (a) apply({ player: cur, action: a }); };
  const onTile = (t: TileId) => {
    if (mode.kind === 'robber') resolveRobber(t, mode.via);
    else if (game.phase === 'moveRobber') resolveRobber(t, 'seven');
  };
  const onEdge = (e: EdgeId) => {
    if (mode.kind === 'roadbuilding') {
      const edges = [...mode.edges, e];
      if (edges.length === 2) apply({ player: cur, action: { type: 'playRoadBuilding', edges: [edges[0], edges[1]] } });
      else setMode({ kind: 'roadbuilding', edges });
    } else {
      const a = interaction.eAction.get(e);
      if (a) apply({ player: cur, action: a });
    }
  };

  // --- discard handling ---
  function adjustDiscard(p: PlayerId, r: Resource, delta: number): void {
    setDiscardSel((s) => {
      const sel = { ...(s[p] ?? {}) };
      sel[r] = Math.max(0, (sel[r] ?? 0) + delta);
      return { ...s, [p]: sel };
    });
  }
  function confirmDiscard(p: PlayerId): void {
    const owed = game.pendingDiscards[p];
    const resources = discardSel[p] ?? {};
    const res = applyMove(game, { player: p, action: { type: 'discard', resources } });
    if (!res.ok) { flash(res.error); return; }
    setGame(res.state);
    setLog((prev) => [`${nameOf(game, p)} discarded ${owed}`, ...prev].slice(0, 50));
    setDiscardSel((s) => ({ ...s, [p]: {} }));
  }

  const acts = legalActions(game, cur);
  const has = (t: Action['type']) => acts.some((a) => a.type === t);

  return (
      <div className="page">
        <header className="topbar">
          <div className="brand">
            <span className="logo">⬡</span>
            <div>
              <h1>Catan</h1>
              <div className="tagline">hotseat · pass &amp; play</div>
            </div>
          </div>
          <div className="newgame">
            <label>seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} /></label>
            <label>players
              <select value={playerCount} onChange={(e) => setPlayerCount(Number(e.target.value))}>
                <option value={2}>2</option><option value={3}>3</option><option value={4}>4</option>
              </select>
            </label>
            <button onClick={startNew}>New game</button>
            <a className="link" href="/">board viewer →</a>
          </div>
        </header>

        <div className="layout">
          <section className="controls card">
            <h2>Actions</h2>
            <ActionBar />
          </section>

          <section className="board-panel">
            <AnimatePresence>
              {game.winner && (
                  <motion.div className="victory" initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                    <span className="dot" style={{ background: colorOf(game, game.winner) }} /> {nameOf(game, game.winner)} wins the game
                  </motion.div>
              )}
            </AnimatePresence>
            <div className="board-wrap">
              <Board
                  game={game}
                  legalVertices={new Set(interaction.vAction.keys())}
                  legalEdges={new Set(interaction.eAction.keys())}
                  legalTiles={interaction.tiles}
                  onVertex={onVertex} onEdge={onEdge} onTile={onTile}
              />
            </div>
          </section>

          <aside className="side">
            <Players />
            <div className="card log-card">
              <h2>Log</h2>
              <ul className="log">
                {log.map((l, i) => <li key={`${i}-${l}`}>{l}</li>)}
              </ul>
            </div>
          </aside>
        </div>

        <AnimatePresence>
          {toast && (
              <motion.div className="toast" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }}>
                {toast}
              </motion.div>
          )}
        </AnimatePresence>
      </div>
  );

  // ---- inline panels (close over state) ----
  function Players() {
    return (
        <div className="card">
          <h2>Players</h2>
          {game.turnOrder.map((p) => {
            const pl = game.players[p];
            const isCur = p === cur;
            const dev = pl.devCards.length + pl.pendingDevCards.length;
            return (
                <div key={p} className={`pcard${isCur ? ' cur' : ''}`} style={{ ['--pc' as string]: PLAYER[pl.color].fill }}>
                  <div className="phead">
                    <span className="dot" style={{ background: PLAYER[pl.color].fill }} />
                    <span className="pname">{PLAYER[pl.color].name}</span>
                    {isCur && <span className="turntag">turn</span>}
                    <span className="vp">{victoryPoints(game, p)} <small>VP</small></span>
                  </div>
                  <div className="hand">
                    {RESOURCES.map((r) => (
                        <span key={r} className="res" style={{ ['--c' as string]: RES_COLOR[r] }} title={r}>
                    {RES_ABBR[r]}<b>{pl.resources[r]}</b>
                  </span>
                    ))}
                  </div>
                  <div className="meta">
                    Dev {dev} · Knights {pl.playedKnights} · {pl.supply.settlements}s/{pl.supply.cities}c/{pl.supply.roads}r
                    {game.longestRoad?.player === p && <span className="badge">Longest road</span>}
                    {game.largestArmy?.player === p && <span className="badge">Largest army</span>}
                  </div>
                </div>
            );
          })}
        </div>
    );
  }

  function ActionBar() {
    const phaseText: Record<GameState['phase'], string> = {
      setupSettlement: 'place your starting settlement', setupRoad: 'place a road beside it',
      roll: 'roll the dice (or play a development card first)', discard: 'discard down to the limit',
      moveRobber: 'move the robber', main: 'build, trade, play cards', gameOver: 'game over',
    };
    const turn = (
        <div className="turnline">
          <span className="dot" style={{ background: colorOf(game, cur) }} />
          <b>{nameOf(game, cur)}</b>
          <span className="phase">— {phaseText[game.phase]}</span>
          {game.dice && <span className="dice">🎲 {game.dice[0]} + {game.dice[1]}</span>}
        </div>
    );

    if (game.winner) return <div className="actionbar"><div className="turnline"><b>{nameOf(game, game.winner)} wins!</b> Start a new game above.</div></div>;

    let body: React.ReactNode;
    if (victim) {
      body = (
          <div className="row">
            <span className="hint">Steal from:</span>
            {victim.victims.map((vp) => (
                <button key={vp} onClick={() => apply({ player: cur, action: victim.via === 'knight'
                      ? { type: 'playKnight', robberTo: victim.tile, stealFrom: vp }
                      : { type: 'moveRobber', tile: victim.tile, stealFrom: vp } })}>
                  {nameOf(game, vp)} ({handSize(game, vp)})
                </button>
            ))}
          </div>
      );
    } else if (game.phase === 'discard') {
      body = <Discard />;
    } else if (mode.kind === 'robber') {
      body = <div className="hint">Click a tile to move the robber.</div>;
    } else if (mode.kind === 'roadbuilding') {
      body = <div className="row"><span className="hint">Road Building — pick {2 - mode.edges.length} more road(s).</span><button className="ghost" onClick={reset}>Cancel</button></div>;
    } else if (mode.kind === 'monopoly') {
      body = <div className="row"><span className="hint">Monopoly — choose a resource:</span>{RESOURCES.map((r) => (
          <button key={r} className="respick" style={{ ['--c' as string]: RES_COLOR[r] }} onClick={() => apply({ player: cur, action: { type: 'playMonopoly', resource: r } })}>{r}</button>
      ))}<button className="ghost" onClick={reset}>Cancel</button></div>;
    } else if (mode.kind === 'yearofplenty') {
      body = <div className="row"><span className="hint">Year of Plenty — pick 2 ({mode.pick.length} chosen):</span>{RESOURCES.map((r) => (
          <button key={r} className="respick" style={{ ['--c' as string]: RES_COLOR[r] }} onClick={() => {
            const pick = [...mode.pick, r];
            if (pick.length === 2) apply({ player: cur, action: { type: 'playYearOfPlenty', take: [pick[0], pick[1]] } });
            else setMode({ kind: 'yearofplenty', pick });
          }}>{r}</button>
      ))}<button className="ghost" onClick={reset}>Cancel</button></div>;
    } else {
      body = (
          <>
            <div className="row">
              {has('rollDice') && <button onClick={() => apply({ player: cur, action: { type: 'rollDice' } })}>Roll dice</button>}
              {has('buyDevCard') && <button className="ghost" onClick={() => apply({ player: cur, action: { type: 'buyDevCard' } })}>Buy dev card</button>}
              {has('playKnight') && <button className="ghost" onClick={() => setMode({ kind: 'robber', via: 'knight' })}>Play Knight</button>}
              {has('playRoadBuilding') && <button className="ghost" onClick={() => setMode({ kind: 'roadbuilding', edges: [] })}>Road Building</button>}
              {has('playYearOfPlenty') && <button className="ghost" onClick={() => setMode({ kind: 'yearofplenty', pick: [] })}>Year of Plenty</button>}
              {has('playMonopoly') && <button className="ghost" onClick={() => setMode({ kind: 'monopoly' })}>Monopoly</button>}
              {has('endTurn') && <button className="end" onClick={() => apply({ player: cur, action: { type: 'endTurn' } })}>End turn</button>}
            </div>
            {(game.phase === 'setupSettlement' || game.phase === 'setupRoad' || game.phase === 'main') &&
                <div className="hint small">Highlighted spots on the board are clickable.</div>}
            {game.phase === 'main' && <Trade />}
          </>
      );
    }

    return <div className="actionbar">{turn}{body}</div>;
  }

  function Trade() {
    return (
        <div className="trade">
          <div className="trade-lab">Bank trade — give</div>
          <div className="row">
            {RESOURCES.map((r) => {
              const ratio = tradeRatio(game, cur, r);
              const can = game.players[cur].resources[r] >= ratio;
              return (
                  <button key={r} className={`respick${tradeGive === r ? ' sel' : ''}`} style={{ ['--c' as string]: RES_COLOR[r] }}
                          disabled={!can} onClick={() => setTradeGive(r)}>{ratio} {RES_ABBR[r]}</button>
              );
            })}
          </div>
          <div className="trade-lab">receive</div>
          <div className="row">
            {tradeGive
                ? RESOURCES.filter((r) => r !== tradeGive && game.bank[r] > 0).map((r) => (
                    <button key={r} className="respick" style={{ ['--c' as string]: RES_COLOR[r] }}
                            onClick={() => apply({ player: cur, action: { type: 'bankTrade', give: tradeGive, giveCount: tradeRatio(game, cur, tradeGive), receive: r } })}>
                      {RES_ABBR[r]}
                    </button>))
                : <span className="hint">pick what to give first</span>}
          </div>
        </div>
    );
  }

  function Discard() {
    const owe = Object.keys(game.pendingDiscards) as PlayerId[];
    return (
        <>
          <div className="hint">A 7 was rolled — players over 7 cards discard half.</div>
          {owe.map((p) => {
            const sel = discardSel[p] ?? {};
            const chosen = RESOURCES.reduce((s, r) => s + (sel[r] ?? 0), 0);
            const need = game.pendingDiscards[p];
            return (
                <div key={p} className="dcard">
                  <div className="phead"><span className="dot" style={{ background: colorOf(game, p) }} /><b>{nameOf(game, p)}</b> discard {chosen}/{need}</div>
                  <div className="hand">
                    {RESOURCES.map((r) => {
                      const have = game.players[p].resources[r];
                      const picked = sel[r] ?? 0;
                      return (
                          <span key={r} className="dres" style={{ ['--c' as string]: RES_COLOR[r] }}>
                      {RES_ABBR[r]} {picked}/{have}
                            <button disabled={picked <= 0} onClick={() => adjustDiscard(p, r, -1)}>−</button>
                      <button disabled={picked >= have || chosen >= need} onClick={() => adjustDiscard(p, r, +1)}>+</button>
                    </span>
                      );
                    })}
                  </div>
                  <button disabled={chosen !== need} onClick={() => confirmDiscard(p)}>Confirm discard</button>
                </div>
            );
          })}
        </>
    );
  }
}