// Separate build config for the Capacitor wrapper — vite.config.ts's regular
// `build` outputs a multi-page site (index/game/online/replay.html); a
// native app needs exactly one `index.html` at the webDir root, and it
// should be the online (multiplayer) client, not the debug board viewer.
// Building mobile.html on its own (bin/rename-mobile-entry.mjs renames it to
// index.html afterward) keeps that one native entry point out of the
// regular multi-page bundle instead of trying to alias index.html itself,
// which the web deploy still needs to be the board viewer.
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    optimizeDeps: { exclude: ['@catan/core'] },
    build: {
        outDir: 'dist-mobile',
        rollupOptions: {
            input: { mobile: 'mobile.html' },
        },
    },
});
