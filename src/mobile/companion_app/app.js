/**
 * app.js
 * -------
 * Main application logic for the Live Field Companion PWA.
 *
 * Manages:
 *   - Camera stream (getUserMedia)
 *   - WebSocket telemetry connection
 *   - WebRTC video streaming (primary)
 *   - JPEG frame upload (fallback at ~5 FPS)
 *   - Offline caching
 *   - UI state and updates
 */

'use strict';

// Global error logger to capture any runtime JS failures on the phone
// and print them directly to the phone UI Status Log.
window.addEventListener('error', (event) => {
  const logEl = document.getElementById('log');
  if (logEl) {
    const line = document.createElement('div');
    line.style.color = '#f87171'; // light red
    line.style.fontWeight = 'bold';
    line.textContent = `[JS ERROR] ${event.message} at ${event.filename.split('/').pop()}:${event.lineno}:${event.colno}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }
});

window.addEventListener('unhandledrejection', (event) => {
  const logEl = document.getElementById('log');
  if (logEl) {
    const line = document.createElement('div');
    line.style.color = '#f87171';
    line.style.fontWeight = 'bold';
    line.textContent = `[PROMISE ERROR] ${event.reason}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }
});

const app = (() => {
  // ── State ──────────────────────────────────────────────────────
  let _serverUrl = '';
  let _wsUrl = '';
  let _ws = null;
  let _localStream = null;
  let _missionActive = false;
  let _framesTotal = 0;
  let _fpsMeter = { count: 0, ts: Date.now() };
  let _fpsDisplay = 0;
  let _uploading = false;
  let _useWebRTC = false;
  let _captureInterval = null;
  let _telemetryInterval = null;
  let _uiUpdateInterval = null;

  const JPEG_FPS = 5;          // JPEG fallback frame rate
  const TELEMETRY_HZ = 10;     // Telemetry packets per second

  // ── Tunnel-bypass fetch wrapper ────────────────────────────────
  // Adds headers that tell localtunnel and ngrok to skip their
  // browser interstitial pages, which otherwise block all JS requests.
  async function _apiFetch(url, options = {}) {
    const headers = Object.assign(
      {
        'bypass-tunnel-reminder': '1',      // localtunnel bypass
        'ngrok-skip-browser-warning': '1',  // ngrok bypass
      },
      options.headers || {}
    );
    return fetch(url, { ...options, headers });
  }

  // ── Connect ───────────────────────────────────────────────────
  async function connect() {
    const urlInput = document.getElementById('server-url-input');
    _serverUrl = (urlInput.value || '').trim().replace(/\/$/, '');

    if (!_serverUrl) {
      _log('⚠ Enter the Ground Station IP address first');
      return;
    }

    _setBadge('connecting');
    _log(`Connecting to ${_serverUrl}...`);

    try {
      // Test HTTP connectivity — also validates the response is JSON,
      // not a localtunnel/ngrok interstitial HTML page.
      const resp = await _apiFetch(`${_serverUrl}/`);
      if (!resp.ok) throw new Error('Server returned ' + resp.status);
      const ct = (resp.headers && typeof resp.headers.get === 'function') ? (resp.headers.get('content-type') || '') : '';
      if (!ct.includes('application/json')) {
        throw new Error(
          'Tunnel interstitial detected. Open ' + _serverUrl +
          ' in your browser first, click through the warning page, then tap Connect here.'
        );
      }

      // Build WebSocket URL — http→ws, https→wss
      _wsUrl = _serverUrl.replace(/^https/, 'wss').replace(/^http(?!s)/, 'ws') + '/ws/telemetry';

      // Try WebSocket telemetry — non-fatal if tunnel doesn't support WS upgrades
      try {
        await _openTelemetryWS();
      } catch (wsErr) {
        _log('⚠ Telemetry WS unavailable (' + wsErr.message + ') — using HTTP fallback');
        _ws = null;
      }

      await _startCamera();

      // Try WebRTC first
      const rtcOk = await WebRTCClient.connect(_serverUrl, _localStream, _onWebRTCState);
      if (rtcOk) {
        _useWebRTC = true;
        _setStreamModeBadge('⚡ WebRTC');
      } else {
        _useWebRTC = false;
        _setStreamModeBadge('📷 JPEG');
      }

      // Start JPEG fallback loop (always running; WebRTC takes priority)
      _startJpegLoop();
      _startTelemetryLoop();
      _startUIUpdateLoop();

      // Upload any cached frames from previous offline period
      const uploaded = await OfflineManager.uploadCached(_serverUrl, _apiFetch);
      if (uploaded > 0) _log(`☁ Uploaded ${uploaded} cached frames from offline period`);

      _setBadge('connected');
      document.getElementById('btn-start').disabled = false;
      document.getElementById('btn-calibrate').disabled = false;
      document.getElementById('btn-connect').textContent = '✓ Connected';

      // Auto-activate phone streaming if a mission is already running on the server
      // (e.g. started from the desktop Streamlit dashboard)
      try {
        const statusResp = await _apiFetch(`${_serverUrl}/api/mission/status`);
        if (statusResp.ok) {
          const status = await statusResp.json();
          if (status && status.active) {
            _missionActive = true;
            _framesTotal = 0;
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = false;
            _log('🔄 Mission already active on server — phone stream activated automatically');
          }
        }
      } catch (_) {}

    } catch (e) {
      _log('❌ Connection failed: ' + e.message);
      _setBadge('disconnected');
    }
  }

  // ── Camera ────────────────────────────────────────────────────
  async function _startCamera() {
    const constraints = {
      video: {
        facingMode: 'environment',  // Rear camera
        width:  { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
      audio: false,
    };

    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
      throw new Error(
        'Camera API (getUserMedia) not supported in this browser. Please open this link directly in the standalone Google Chrome app, not inside a QR scanner app, messaging app webview, or private/incognito tab.'
      );
    }

    // Stop and release previous stream if it exists to avoid device conflict or play interruptions
    if (_localStream) {
      try {
        _localStream.getTracks().forEach(track => track.stop());
      } catch (_) {}
      _localStream = null;
      const videoEl = document.getElementById('camera-video');
      if (videoEl) videoEl.srcObject = null;
    }

    _localStream = await navigator.mediaDevices.getUserMedia(constraints);
    const video = document.getElementById('camera-video');
    video.srcObject = _localStream;
    
    try {
      await video.play();
    } catch (playErr) {
      // Non-fatal: Chrome sometimes interrupts play() if we load a new stream immediately
      console.log('[Camera] video.play() interrupted/delayed:', playErr.message);
    }

    let resolution = 'unknown';
    try {
      if (_localStream && typeof _localStream.getVideoTracks === 'function') {
        const tracks = _localStream.getVideoTracks();
        if (tracks && tracks.length > 0) {
          const track = tracks[0];
          const settings = (track && typeof track.getSettings === 'function') ? track.getSettings() : null;
          if (settings && settings.width && settings.height) {
            resolution = settings.width + 'x' + settings.height;
          }
        }
      }
    } catch (_) {}

    _log('📷 Camera started (' + resolution + ')');
  }

  // ── Telemetry WebSocket ────────────────────────────────────────
  async function _openTelemetryWS() {
    return new Promise((resolve, reject) => {
      _ws = new WebSocket(_wsUrl);
      // Increase timeout slightly to give tunnels a fair chance
      const timeout = setTimeout(() => reject(new Error('WS timeout')), 6000);

      _ws.onopen = () => {
        clearTimeout(timeout);
        _log('📡 Telemetry channel open (WebSocket)');
        resolve();
      };

      _ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('WebSocket error'));
      };

      _ws.onclose = () => {
        // Only mark disconnected if WS was our primary connected path
        if (_ws && _missionActive) _setBadge('disconnected');
        // Attempt silent reconnect
        setTimeout(() => {
          if (_missionActive) _openTelemetryWS().catch(() => {});
        }, 5000);
      };

      _ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (!msg.mission_active && _missionActive) {
            _log('ℹ Mission ended by ground station');
            _missionActive = false;
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
          }
        } catch (_) {}
      };
    });
  }

  // ── Telemetry sending loop (WS + HTTP fallback) ────────────────
  function _startTelemetryLoop() {
    if (_telemetryInterval) clearInterval(_telemetryInterval);
    _telemetryInterval = setInterval(async () => {
      if (!_missionActive) return;
      // Only send via WS if it's open; otherwise silently skip
      // (GPS is still embedded in JPEG frame metadata when _apiFetch is used)
      if (_ws && _ws.readyState === WebSocket.OPEN) {
        const packet = SensorManager.getTelemetryPacket();
        _ws.send(JSON.stringify(packet));
      }
    }, 1000 / TELEMETRY_HZ);
  }

  // ── JPEG frame capture + upload (fallback) ─────────────────────
  function _startJpegLoop() {
    if (_captureInterval) clearInterval(_captureInterval);
    _captureInterval = setInterval(async () => {
      if (!_missionActive || _uploading) return;
      if (_useWebRTC && WebRTCClient.isConnected()) return; // WebRTC handling it

      const video = document.getElementById('camera-video');
      if (!video.srcObject || video.readyState < 2) return;

      const canvas = document.createElement('canvas');
      const maxW = 640;
      const nativeW = video.videoWidth || 640;
      const nativeH = video.videoHeight || 480;
      const scale = Math.min(1.0, maxW / nativeW);
      
      canvas.width  = Math.round(nativeW * scale);
      canvas.height = Math.round(nativeH * scale);
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob) return;
        _uploading = true;
        try {
          const formData = new FormData();
          formData.append('file', blob, 'frame.jpg');

          // Append telemetry packet to form data for HTTP fallback mode
          const packet = SensorManager.getTelemetryPacket();
          formData.append('telemetry', JSON.stringify(packet));

          const resp = await _apiFetch(`${_serverUrl}/api/frame`, {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(2000),
          });

          if (resp.ok) {
            _framesTotal++;
            _fpsMeter.count++;
            _setBadge('connected');
          } else {
            // Cache for later
            const tele = SensorManager.getTelemetryPacket();
            await OfflineManager.cacheFrame(blob, tele);
          }
        } catch (e) {
          // Offline: cache frame
          try {
            const tele = SensorManager.getTelemetryPacket();
            await OfflineManager.cacheFrame(blob, tele);
          } catch (_) {}
          _setBadge('disconnected');
        } finally {
          _uploading = false;
        }
      }, 'image/jpeg', 0.8);

    }, 1000 / JPEG_FPS);
  }

  // ── Mission status polling (HTTP fallback when WS is down) ─────
  async function _pollMissionStatus() {
    if (!_serverUrl) return;
    try {
      const resp = await _apiFetch(`${_serverUrl}/api/mission/status`);
      if (resp.ok) {
        const status = await resp.json();
        if (status) {
          const serverActive = !!status.active;
          if (serverActive !== _missionActive) {
            _missionActive = serverActive;
            if (_missionActive) {
              _framesTotal = 0;
              document.getElementById('btn-start').disabled = true;
              document.getElementById('btn-stop').disabled = false;
              _log('🔄 Mission started on server — phone stream activated');
            } else {
              document.getElementById('btn-start').disabled = false;
              document.getElementById('btn-stop').disabled = true;
              _log('⏹ Mission stopped on server — phone stream stopped');
            }
          }
        }
      }
    } catch (_) {}
  }

  // ── UI update loop ─────────────────────────────────────────────
  function _startUIUpdateLoop() {
    if (_uiUpdateInterval) clearInterval(_uiUpdateInterval);
    let pollTick = 0;
    _uiUpdateInterval = setInterval(() => {
      _updateTelemetryUI();
      _updateFPS();

      // Poll server status every 2 seconds if WebSocket is not active
      pollTick++;
      if (pollTick >= 8) {
        pollTick = 0;
        if (!_ws || _ws.readyState !== WebSocket.OPEN) {
          _pollMissionStatus();
        }
      }
    }, 250);
  }

  function _updateTelemetryUI() {
    const gps = SensorManager.getGPS();
    const bat = SensorManager.getBattery();
    const compass = SensorManager.getCompass();

    _setText('m-lat', gps.lat ? gps.lat.toFixed(5) : '—');
    _setText('m-lon', gps.lon ? gps.lon.toFixed(5) : '—');
    _setText('m-acc', gps.acc ? gps.acc.toFixed(1) : '—');
    _setText('m-heading', compass ? compass.toFixed(0) + '°' : '—');
    _setText('m-speed', SensorManager.getSpeed().toFixed(2));
    _setText('m-frames', _framesTotal);
    _setText('m-bat', bat + '%');

    const batBar = document.getElementById('bat-bar');
    if (batBar) {
      batBar.style.width = bat + '%';
      batBar.style.background = bat > 50 ? 'var(--green)' :
                                bat > 20 ? 'var(--yellow)' : 'var(--red)';
    }

    // Offline cache indicator
    const cached = OfflineManager.getCacheCount();
    if (cached > 0) {
      _log(`📦 ${cached} frames queued offline`);
    }
  }

  function _updateFPS() {
    const now = Date.now();
    const elapsed = (now - _fpsMeter.ts) / 1000;
    if (elapsed >= 1.0) {
      _fpsDisplay = Math.round(_fpsMeter.count / elapsed);
      _fpsMeter = { count: 0, ts: now };
      const badge = document.getElementById('fps-badge');
      if (badge) badge.textContent = _fpsDisplay + ' FPS';
    }
  }

  // ── Mission control ────────────────────────────────────────────
  async function startMission() {
    const gps = SensorManager.getGPS();
    try {
      // Fire the start request, but do NOT gate streaming on the response.
      // The desktop may have already started the mission; the server may
      // return 4xx/5xx in that case — we still want the phone to stream.
      await _apiFetch(
        `${_serverUrl}/api/mission/start?lat=${gps.lat}&lon=${gps.lon}&name=Field+Mission`
      ).catch(() => {});
    } catch (_) {}

    // Always activate phone-side streaming when the button is tapped
    _missionActive = true;
    _framesTotal = 0;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    _log('✅ Phone stream activated — sending frames to ground station');
  }

  async function stopMission() {
    try {
      await _apiFetch(`${_serverUrl}/api/mission/stop`);
    } catch (_) {}
    _missionActive = false;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    _log('⏹ Mission stopped. ' + _framesTotal + ' frames captured.');
  }

  // ── Calibration ───────────────────────────────────────────────
  function startCalibration() {
    const video = document.getElementById('camera-video');
    CalibrationManager.start(video, _serverUrl, 'grass_patch');
  }

  function cancelCalibration() {
    CalibrationManager.cancel();
  }

  // ── WebRTC state callback ─────────────────────────────────────
  function _onWebRTCState(state) {
    if (state === 'connected') {
      _useWebRTC = true;
      _setStreamModeBadge('⚡ WebRTC');
    } else if (state === 'fallback') {
      _useWebRTC = false;
      _setStreamModeBadge('📷 JPEG');
    }
  }

  // ── Helpers ───────────────────────────────────────────────────
  function _setBadge(state) {
    const badge = document.getElementById('conn-badge');
    if (!badge) return;
    badge.className = 'conn-badge ' + state;
    const labels = {
      connected:    'ONLINE',
      connecting:   'LINKING',
      disconnected: 'OFFLINE',
    };
    badge.textContent = labels[state] || state.toUpperCase();
  }

  function _setStreamModeBadge(text) {
    const el = document.getElementById('stream-mode');
    if (el) el.textContent = text;
  }

  function _setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function _log(msg) {
    const el = document.getElementById('log');
    if (!el) return;
    const line = document.createElement('div');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
    console.log('[App]', msg);
  }

  // Safe LocalStorage helpers to prevent incognito/webview storage crashes
  function _getStorage(key) {
    try { return localStorage.getItem(key); } catch (_) { return null; }
  }
  function _setStorage(key, val) {
    try { localStorage.setItem(key, val); } catch (_) {}
  }

  // ── Init ──────────────────────────────────────────────────────
  async function init() {
    await OfflineManager.init();
    await SensorManager.startAll();

    // Restore last server URL or default to current origin
    const input = document.getElementById('server-url-input');
    if (input) {
      const saved = _getStorage('lastServerUrl');
      if (saved) {
        input.value = saved;
      } else {
        // Automatically default to the domain the PWA was loaded from
        input.value = window.location.origin;
      }
    }

    document.getElementById('server-url-input')?.addEventListener('input', (e) => {
      _setStorage('lastServerUrl', e.target.value);
    });

    _log('Live Field Companion ready. Enter Ground Station IP to connect.');
  }

  // Auto-init
  window.addEventListener('load', init);

  return {
    connect,
    startMission,
    stopMission,
    startCalibration,
    cancelCalibration,
  };
})();
