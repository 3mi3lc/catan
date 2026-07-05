# @catan/web

React/Vite frontend — `index.html` (board viewer), `game.html`/`hotseat.html`
(local hotseat), `online.html` (multiplayer client for `@catan/server`), and
`replay.html`. See the root README for the general `pnpm --filter web dev`
workflow.

## Mobile (Capacitor/Android)

The native Android wrapper lives in `android/` (scaffolded via `npx cap add
android`, checked into git except Gradle build output/`local.properties` —
see `android/.gitignore`). iOS isn't buildable from Windows (`android/ios`
platform packages are installed but there's no Xcode).

### Prerequisites

- **JDK 21+** — `android/app/capacitor.build.gradle` requires it (Capacitor
  8's Gradle setup). Check both `[Environment]::GetEnvironmentVariable("JAVA_HOME","User")`
  and `...,"Machine")` in PowerShell if a build fails with `invalid source
  release: 21` despite `java -version` looking right — a stale **User**-scope
  `JAVA_HOME` silently overrides a correct Machine-scope one, and neither
  takes effect in a terminal window that was already open before you set it.
- Android SDK (`ANDROID_HOME`) with a recent platform + build-tools installed.

### One-off static build + install

```sh
pnpm --filter @catan/web cap:sync   # builds dist-mobile/, copies it into android/, syncs plugins
```
Then open `android/` in Android Studio and run, or from `android/`:
```sh
JAVA_HOME="<path-to-jdk-21+>" ./gradlew assembleDebug
```
Installs `android/app/build/outputs/apk/debug/app-debug.apk`.

### Live-reload dev loop

`pnpm --filter @catan/web cap:dev` builds+installs the app once, but points
its WebView at your **running** Vite dev server instead of the bundled
files — edit React/CSS and it hot-reloads on-device, no rebuild/reinstall.
Wired through `CAP_LIVE_RELOAD_URL` (`capacitor.config.ts`) and
`bin/cap-dev.mjs`.

```sh
pnpm --filter @catan/web dev   # separate terminal, leave running (needs server.host: true — see vite.config.ts)
$env:CAP_LIVE_RELOAD_URL = "http://<your-PC-LAN-IP>:5173"
pnpm --filter @catan/web cap:dev
```

Phone and PC must be on the **same Wi-Fi network**, and Windows Firewall
needs an inbound allow rule for the port (one-time, admin PowerShell):
```powershell
New-NetFirewallRule -DisplayName "Vite Dev Server" -Direction Inbound -Protocol TCP -LocalPort 5173 -Action Allow
```

**Why LAN and not `adb reverse tcp:5173 tcp:5173` over USB**: that's the more
common Capacitor live-reload setup and *should* work, and does on most
devices/emulators. It reproducibly failed on the physical test device used
to build this feature — every request through the tunnel (from the app *and*
from a plain Chrome tab, ruling out anything Capacitor/manifest-specific)
came back `net::ERR_EMPTY_RESPONSE`, despite the tunnel registering fine
(`adb reverse --list`) and the dev server responding fine to `curl` from the
PC itself. Root cause was never pinned down (suspected: this particular
OEM Android build enforcing some network policy against the adb loopback
path) — LAN Wi-Fi sidesteps it entirely and is the recommended default here.
If USB/`adb reverse` works fine on your device, it's a reasonable alternative
to avoid needing a shared Wi-Fi network.

Also present, unrelated to the above: `vite.config.ts` excludes `android/`
from the dev server's file watcher (`server.watch.ignored`) — without it,
every Gradle build under `android/build`/`android/app/build` fires spurious
HMR page-reload events, since that directory otherwise falls inside Vite's
watched project root.

### Native config notes

- `capacitor.config.ts`'s `server` block only sets `androidScheme: 'https'`
  for a normal build. Contrary to older Capacitor docs, `server.cleartext:
  true` (used automatically when `CAP_LIVE_RELOAD_URL` is set) does **not**
  get Capacitor 8 to patch `AndroidManifest.xml`'s
  `usesCleartextTraffic`/`networkSecurityConfig` for you — confirmed by
  grepping the installed `@capacitor/cli`/`@capacitor/android` packages for
  "cleartext" (no matches). `android/app/src/main/res/xml/network_security_config.xml`
  does that by hand instead, scoped to `localhost`/`127.0.0.1` only — a real
  release build never reaches that code path (always `androidScheme:
  'https'`), so this doesn't loosen anything for a shipped app.
