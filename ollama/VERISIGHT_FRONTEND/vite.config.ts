import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
      plugins: [react()],
      server: {
            port: 5173,
            open: true,
            proxy: {
                  '/api': 'http://localhost:8001',
                  '/health': 'http://localhost:8001',
            },
      },
      build: {
            outDir: 'dist',
            sourcemap: false,
      },
});
