// Tiny dependency-free sound-effect manager. Maps the engine's GameEvent
// stream (see @catan/core's actions.ts) to short SFX cues, shared by every
// surface (hotseat.ts, online.ts, game/App.tsx) so "what makes a sound" is
// defined exactly once.
//
// No SFX files ship in this repo (see public/sfx/SOURCES.md for what to drop
// in and where to get them) — every play() call degrades silently (no error,
// no exception, no console noise) when a file is missing or playback is
// rejected, so the game is fully playable, just silent, until real audio
// assets land. Same pattern as boardArt.ts's tile-art fallback in Phase 2.

import type { GameEvent, Action } from '@catan/core';

export type SfxName =
    | 'diceRoll'
    | 'buildSettlement' | 'buildCity' | 'buildRoad'
    | 'robberMove' | 'steal'
    | 'devCardBuy'
    | 'tradeExecuted' | 'tradeBank' | 'discard'
    | 'award' | 'victory'
    | 'click' | 'yourTurn';

const SFX_FILES: Record<SfxName, string> = {
    diceRoll: '/sfx/dice-roll.mp3',
    buildSettlement: '/sfx/build-settlement.mp3',
    buildCity: '/sfx/build-city.mp3',
    buildRoad: '/sfx/build-road.mp3',
    robberMove: '/sfx/robber-move.mp3',
    steal: '/sfx/steal.mp3',
    devCardBuy: '/sfx/card-buy.mp3',
    tradeExecuted: '/sfx/trade-executed.mp3',
    tradeBank: '/sfx/trade-bank.mp3',
    discard: '/sfx/discard.mp3',
    award: '/sfx/award.mp3',
    victory: '/sfx/victory.mp3',
    click: '/sfx/click.mp3',
    yourTurn: '/sfx/your-turn.mp3',
};

const MUTE_KEY = 'catan-sfx-muted';
const VOLUME_KEY = 'catan-sfx-volume';

let muted = localStorage.getItem(MUTE_KEY) === '1';
let volume = (() => {
    const raw = localStorage.getItem(VOLUME_KEY);
    // Number(null) is 0, not NaN — an explicit null check is required here,
    // or a never-before-set volume silently defaults to 0 (silent) instead
    // of the intended 0.6.
    if (raw === null) return 0.6;
    const v = Number(raw);
    return Number.isFinite(v) && v >= 0 && v <= 1 ? v : 0.6;
})();

// One base <audio> per sound — doubles as both "does this file actually
// exist" detection (via its error event) and the clone source for play(), so
// repeated/overlapping triggers of the same cue don't re-fetch.
const base = new Map<SfxName, HTMLAudioElement>();
const broken = new Set<SfxName>();

function ensureLoaded(name: SfxName): HTMLAudioElement | null {
    if (broken.has(name)) return null;
    let el = base.get(name);
    if (!el) {
        el = new Audio(SFX_FILES[name]);
        el.preload = 'auto';
        el.addEventListener('error', () => { broken.add(name); base.delete(name); }, { once: true });
        base.set(name, el);
    }
    return el;
}

let unlocked = false;
function armUnlock(): void {
    if (unlocked) return;
    const unlock = () => {
        if (unlocked) return;
        unlocked = true;
        // Most browsers' autoplay policy gates ANY audio playback on the page
        // behind the first user gesture — Safari/WKWebView (i.e. the eventual
        // Capacitor mobile shell) enforce this strictly. One real play()
        // inside a direct gesture handler unlocks the page for every
        // subsequent programmatic play() call — including ones triggered by
        // a socket.io message, which is never itself a user gesture (e.g. an
        // opponent's move arriving over the network).
        for (const el of base.values()) {
            el.play().then(() => { el.pause(); el.currentTime = 0; }).catch(() => {});
        }
        window.removeEventListener('pointerdown', unlock);
        window.removeEventListener('keydown', unlock);
    };
    window.addEventListener('pointerdown', unlock, { once: true });
    window.addEventListener('keydown', unlock, { once: true });
}

/** Call once at app start — kicks off loading every clip (cheap no-op for
 *  missing files) and arms the autoplay-policy unlock for the first tap. */
export function preloadSfx(): void {
    (Object.keys(SFX_FILES) as SfxName[]).forEach(ensureLoaded);
    armUnlock();
}

export function setMuted(next: boolean): void {
    muted = next;
    localStorage.setItem(MUTE_KEY, next ? '1' : '0');
}
export function isMuted(): boolean { return muted; }
export function toggleMuted(): boolean { setMuted(!muted); return muted; }

export function setVolume(next: number): void {
    volume = Math.max(0, Math.min(1, next));
    localStorage.setItem(VOLUME_KEY, String(volume));
}
export function getVolume(): number { return volume; }

/** Play a named cue. Silently does nothing if muted, the file is missing, or
 *  playback is rejected (e.g. still pre-gesture on a strict browser). */
export function play(name: SfxName): void {
    if (muted) return;
    const el = ensureLoaded(name);
    if (!el) return;
    const node = el.cloneNode(true) as HTMLAudioElement;
    node.volume = volume;
    node.play().catch(() => {});
}

function buildSfx(what: 'settlement' | 'city' | 'road'): SfxName {
    return what === 'settlement' ? 'buildSettlement' : what === 'city' ? 'buildCity' : 'buildRoad';
}

/** Map a batch of engine events (as returned by applyMove, or the server's
 *  `gameState` push) to SFX cues — the single source of "what sound plays
 *  when," shared by hotseat.ts, online.ts and game/App.tsx. */
export function playForEvents(events: readonly GameEvent[]): void {
    for (const ev of events) {
        switch (ev.type) {
            case 'diceRolled': play('diceRoll'); break;
            case 'built': play(buildSfx(ev.what)); break;
            case 'robberMoved': play(ev.stolen ? 'steal' : 'robberMove'); break;
            case 'devCardBought': play('devCardBuy'); break;
            case 'tradeExecuted': play('tradeExecuted'); break;
            case 'awardMoved': play('award'); break;
            case 'gameWon': play('victory'); break;
            case 'resourcesProduced': break; // covered by the diceRolled cue for the same roll
        }
    }
}

/** A couple of actions are audible but emit no dedicated GameEvent (bank
 *  trades, discards) — call alongside playForEvents right after a move
 *  succeeds, passing the action that was just applied. */
export function playForAction(action: Action): void {
    if (action.type === 'bankTrade') play('tradeBank');
    else if (action.type === 'discard') play('discard');
}

/** Delegated click cue for every button on the page — call once per surface.
 *  Cheaper than threading play('click') through every individual handler. */
export function installClickSfx(): void {
    document.addEventListener('click', (e) => {
        if ((e.target as HTMLElement | null)?.closest?.('button')) play('click');
    });
}
