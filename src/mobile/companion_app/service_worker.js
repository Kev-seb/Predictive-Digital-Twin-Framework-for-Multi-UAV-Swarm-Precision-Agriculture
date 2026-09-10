/**
 * service_worker.js
 * ------------------
 * PWA Service Worker for Live Field Companion.
 *
 * Provides:
 *   - Offline app shell caching (install all static assets)
 *   - Network-first strategy for API calls
 *   - Cache-first strategy for app assets
 *   - Background sync placeholder for future use
 */

'use strict';

const CACHE_NAME = 'live-field-v9';
const APP_SHELL = [
  '/mobile',
  '/mobile/app.js',
  '/mobile/sensor_manager.js',
  '/mobile/offline_manager.js',
  '/mobile/webrtc_client.js',
  '/mobile/calibration.js',
  '/mobile/manifest.json',
];

// ── Install: cache app shell ───────────────────────────────────

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(APP_SHELL);
    }).catch((e) => {
      console.warn('[SW] Cache install partial failure:', e);
    })
  );
  self.skipWaiting();
});

// ── Activate: clean old caches ─────────────────────────────────

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      );
    })
  );
  self.clients.claim();
});

// ── Fetch: cache-first for shell, network-first for API ────────

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API calls: always try network first, no cache
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
    return; // Let browser handle directly
  }

  // App shell: cache-first
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
        }
        return response;
      }).catch(() => {
        // Offline fallback for HTML
        if (event.request.destination === 'document') {
          return caches.match('/mobile');
        }
      });
    })
  );
});
