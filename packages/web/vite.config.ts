import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['@catan/core'] },
  build: {
    rollupOptions: {
      input: { viewer: 'index.html', game: 'game.html' },
    },
  },
});