import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['@catan/core'] },
  // android/ is the native Capacitor project living inside this package (see
  // capacitor.config.ts) — Gradle rewrites thousands of files under
  // android/build, android/app/build, and android/app/src/main/assets/public
  // (the copied web bundle) on every `cap sync`/`cap run`. Without this, the
  // dev server's fs watcher treats all of that churn as reasons to fire page
  // reloads, which was making it hang/misbehave right when a live-reload
  // Android build finished and the app tried to load.
  server: {
    // Listen on the LAN, not just localhost — needed for the Capacitor
    // live-reload workflow (`pnpm cap:dev`) when pointing a phone at this
    // dev server over Wi-Fi instead of `adb reverse` (some devices/ROMs
    // don't reliably deliver traffic through the adb loopback tunnel).
    host: true,
    watch: { ignored: ['**/android/**'] },
  },
  build: {
    rollupOptions: {
      input: { viewer: 'index.html', game: 'game.html', replay: 'replay.html', online: 'online.html' },
    },
  },
});