import { defineConfig } from 'vite';

// Relative base so the built site can be served from a subdirectory (GitHub
// Pages, an archive mirror) without rewriting asset URLs.
export default defineConfig({
  base: './',
  build: {
    // Safari 15 is the oldest engine we care about; it has everything the
    // decoder needs (Web Audio, OffscreenCanvas is feature-detected).
    target: ['es2020', 'safari15'],
    assetsInlineLimit: 0, // never inline the FLAC slices
    chunkSizeWarningLimit: 1200,
  },
  server: {
    // The frame slices are large; keep the dev server from choking on them.
    fs: { strict: false },
  },
  worker: { format: 'es' },
});
