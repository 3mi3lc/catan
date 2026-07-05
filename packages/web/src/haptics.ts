// Haptic feedback for the same GameEvent/Action stream audio/sound.ts already
// turns into SFX cues — a parallel, physical feedback channel, mapped at the
// same call sites (playForEvents/playForAction) rather than a separate wiring
// pass through the game code.
//
// Safe to call unconditionally on plain web too, no Capacitor.isNativePlatform()
// guard needed: @capacitor/haptics ships a "web" implementation that either
// falls back to navigator.vibrate() where the browser supports it, or
// silently no-ops — it doesn't throw just because there's no native shell.
// The try/catch below is extra insurance on top of that, matching
// audio/sound.ts's "degrade silently, never break the game" philosophy.

import { Haptics, ImpactStyle, NotificationType } from '@capacitor/haptics';
import type { GameEvent, Action } from '@catan/core';

function impact(style: ImpactStyle): void {
    Haptics.impact({ style }).catch(() => {});
}

/** Mirrors audio/sound.ts's playForEvents — same call sites, same events. */
export function hapticsForEvents(events: readonly GameEvent[]): void {
    for (const ev of events) {
        switch (ev.type) {
            case 'built': impact(ev.what === 'city' ? ImpactStyle.Heavy : ImpactStyle.Medium); break;
            case 'robberMoved': impact(ev.stolen ? ImpactStyle.Heavy : ImpactStyle.Light); break;
            case 'devCardBought': impact(ImpactStyle.Light); break;
            case 'tradeExecuted': impact(ImpactStyle.Medium); break;
            case 'awardMoved': impact(ImpactStyle.Medium); break;
            case 'gameWon': Haptics.notification({ type: NotificationType.Success }).catch(() => {}); break;
            // Every roll buzzing would be more annoying than informative,
            // and resourcesProduced always accompanies diceRolled anyway.
            case 'diceRolled': case 'resourcesProduced': break;
        }
    }
}

/** Mirrors audio/sound.ts's playForAction — actions with no dedicated
 *  GameEvent (bank trades, discards). */
export function hapticsForAction(action: Action): void {
    if (action.type === 'bankTrade' || action.type === 'discard') impact(ImpactStyle.Light);
}
