import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev: `npm run dev` serves the UI on :5173 and proxies /api to Flask on :5000.
// Production: `npm run build` writes dist/, which Flask serves from the same origin.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 900 },
  server: { port: 5173, proxy: { '/api': { target: 'http://localhost:5000', changeOrigin: true } } },
});
