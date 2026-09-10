/**
 * offline_manager.js
 * ------------------
 * Handles offline recording and reconnect upload.
 *
 * When the connection drops:
 *   - Frames are cached in IndexedDB (JPEG base64 + telemetry + timestamp)
 *   - GPS track continues to be recorded
 *
 * When connection is restored:
 *   - Cached frames are uploaded via POST /api/frames/bulk
 *   - Cache is cleared after successful upload
 */

'use strict';

const OfflineManager = (() => {
  const DB_NAME    = 'live_field_offline';
  const STORE_NAME = 'cached_frames';
  const MAX_CACHE  = 300; // max frames to cache (RAM protection)
  let _db = null;
  let _cache_count = 0;

  // ── IndexedDB init ─────────────────────────────────────────────
  async function init() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { autoIncrement: true });
        }
      };
      req.onsuccess = (e) => {
        _db = e.target.result;
        _countCached().then(n => { _cache_count = n; });
        resolve();
      };
      req.onerror = (e) => {
        console.warn('[OfflineManager] IndexedDB failed:', e);
        _db = null;
        resolve(); // degrade gracefully
      };
    });
  }

  // ── Cache a frame ──────────────────────────────────────────────
  async function cacheFrame(jpegBlob, telemetry) {
    if (!_db || _cache_count >= MAX_CACHE) return;
    try {
      const b64 = await _blobToBase64(jpegBlob);
      const record = {
        jpeg_b64:  b64,
        telemetry: telemetry,
        timestamp: Date.now() / 1000,
      };
      await _dbPut(record);
      _cache_count++;
    } catch (e) {
      console.warn('[OfflineManager] Cache failed:', e);
    }
  }

  // ── Upload cached frames on reconnect ─────────────────────────
  async function uploadCached(serverUrl, fetchFn) {
    if (!_db) return 0;
    const frames = await _dbGetAll();
    if (frames.length === 0) return 0;

    // Use provided fetchFn (e.g. tunnel-bypass wrapper) or fall back to native fetch
    const _fetch = typeof fetchFn === 'function' ? fetchFn : fetch;

    try {
      const resp = await _fetch(`${serverUrl}/api/frames/bulk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(frames),
      });
      if (resp.ok) {
        await _dbClear();
        _cache_count = 0;
        console.log(`[OfflineManager] Uploaded ${frames.length} cached frames`);
        return frames.length;
      }
    } catch (e) {
      console.warn('[OfflineManager] Upload failed:', e);
    }
    return 0;
  }

  // ── Helpers ───────────────────────────────────────────────────
  function _blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function _dbPut(record) {
    return new Promise((resolve, reject) => {
      const tx = _db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).add(record);
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  function _dbGetAll() {
    return new Promise((resolve, reject) => {
      if (!_db) { resolve([]); return; }
      const tx = _db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).getAll();
      req.onsuccess = () => resolve(req.result);
      req.onerror = reject;
    });
  }

  function _dbClear() {
    return new Promise((resolve, reject) => {
      if (!_db) { resolve(); return; }
      const tx = _db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  function _countCached() {
    return new Promise((resolve) => {
      if (!_db) { resolve(0); return; }
      const tx = _db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).count();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(0);
    });
  }

  function getCacheCount() { return _cache_count; }

  return { init, cacheFrame, uploadCached, getCacheCount };
})();
