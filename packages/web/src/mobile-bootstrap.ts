// Native-shell-only startup (mobile.html only — the web entry points never
// load this). Guarded on Capacitor.isNativePlatform() since StatusBar calls
// do throw/reject when there's no native bridge underneath them, unlike
// @capacitor/haptics's graceful web fallback (see ../haptics.ts).
//
// Deliberately NOT wiring a hardware-back-button listener here: Capacitor's
// own default (history.back() when there's a web history entry, otherwise
// backgrounding the app) already matches this app's actual navigation model
// — online.ts/OnlineApp.tsx sync the room URL via history.replaceState, not
// pushState, so there's essentially no in-app history stack to walk anyway.
import { Capacitor } from '@capacitor/core';
import { StatusBar, Style } from '@capacitor/status-bar';

if (Capacitor.isNativePlatform()) {
    void StatusBar.setStyle({ style: Style.Dark }); // light status-bar icons/text over the dark felt background
    void StatusBar.setBackgroundColor({ color: '#14110d' }).catch(() => {}); // Android only; iOS ignores this call
}
