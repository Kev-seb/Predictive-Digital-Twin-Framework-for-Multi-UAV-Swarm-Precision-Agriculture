/**
 * sensor_manager.js
 * ------------------
 * Manages all phone sensors: GPS, IMU (accelerometer + gyroscope),
 * compass, battery, and network signal estimation.
 *
 * Provides a unified SensorManager object that is polled by app.js
 * and the offline/WebRTC modules.
 */

'use strict';

const SensorManager = (() => {
  // ── Internal state ─────────────────────────────────────────────
  let _gps = { lat: 0, lon: 0, alt: 0, acc: 99, ts: 0 };
  let _imu = { ax: 0, ay: 0, az: 9.81, gx: 0, gy: 0, gz: 0 };
  let _compass = 0;
  let _battery = { level: 1.0, charging: false };
  let _speed_mps = 0;
  let _last_gps_pos = null;
  let _last_gps_time = 0;
  let _gps_watcher = null;
  let _motion_active = false;
  let _orientation_active = false;

  // ── GPS ────────────────────────────────────────────────────────
  function startGPS() {
    if (!navigator.geolocation) {
      _log('GPS not available');
      return;
    }
    _gps_watcher = navigator.geolocation.watchPosition(
      (pos) => {
        const now = Date.now();
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;

        // Compute walking speed from GPS if we have a previous position
        if (_last_gps_pos && (now - _last_gps_time) > 500) {
          const d = _haversine(lat, lon, _last_gps_pos[0], _last_gps_pos[1]);
          const dt = (now - _last_gps_time) / 1000.0;
          _speed_mps = 0.5 * _speed_mps + 0.5 * (d / dt);
        }
        _last_gps_pos = [lat, lon];
        _last_gps_time = now;

        _gps = {
          lat: lat,
          lon: lon,
          alt: pos.coords.altitude || 0,
          acc: pos.coords.accuracy || 99,
          ts:  now / 1000,
        };
      },
      (err) => { _log('GPS error: ' + err.message); },
      {
        enableHighAccuracy: true,
        maximumAge: 1000,
        timeout: 5000,
      }
    );
  }

  function stopGPS() {
    if (_gps_watcher !== null) {
      navigator.geolocation.clearWatch(_gps_watcher);
      _gps_watcher = null;
    }
  }

  // ── IMU (Accelerometer + Gyroscope) ───────────────────────────
  async function startIMU() {
    // Requires permission on iOS 13+
    if (typeof DeviceMotionEvent !== 'undefined' &&
        typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const perm = await DeviceMotionEvent.requestPermission();
        if (perm !== 'granted') {
          _log('Motion permission denied');
          return;
        }
      } catch (e) {
        _log('Motion permission error: ' + e);
        return;
      }
    }

    window.addEventListener('devicemotion', (e) => {
      const a = e.accelerationIncludingGravity || {};
      const g = e.rotationRate || {};
      const DEGTORAD = Math.PI / 180;

      _imu = {
        ax: a.x || 0,
        ay: a.y || 0,
        az: a.z || 9.81,
        gx: (g.alpha || 0) * DEGTORAD,
        gy: (g.beta  || 0) * DEGTORAD,
        gz: (g.gamma || 0) * DEGTORAD,
      };
      _motion_active = true;
    }, true);
  }

  // ── Compass (DeviceOrientationEvent) ──────────────────────────
  async function startCompass() {
    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        const perm = await DeviceOrientationEvent.requestPermission();
        if (perm !== 'granted') {
          _log('Orientation permission denied');
          return;
        }
      } catch (e) {}
    }

    window.addEventListener('deviceorientationabsolute', (e) => {
      // alpha = compass heading (0 = North, clockwise)
      if (e.alpha !== null) {
        _compass = (360 - e.alpha) % 360; // Convert to 0=North clockwise
        _orientation_active = true;
      }
    }, true);

    // Fallback non-absolute
    window.addEventListener('deviceorientation', (e) => {
      if (!_orientation_active && e.alpha !== null) {
        _compass = (360 - e.alpha) % 360;
      }
    }, true);
  }

  // ── Battery ───────────────────────────────────────────────────
  async function startBattery() {
    try {
      if (navigator.getBattery) {
        const bat = await navigator.getBattery();
        _battery = { level: bat.level, charging: bat.charging };
        bat.addEventListener('levelchange', () => {
          _battery = { level: bat.level, charging: bat.charging };
        });
        bat.addEventListener('chargingchange', () => {
          _battery = { level: bat.level, charging: bat.charging };
        });
      }
    } catch (e) {
      _log('Battery API not available');
    }
  }

  // ── Public API ────────────────────────────────────────────────
  async function startAll() {
    startGPS();
    await startIMU();
    await startCompass();
    await startBattery();
    _log('Sensors initialised');
  }

  function stopAll() {
    stopGPS();
  }

  function getTelemetryPacket() {
    return {
      type: 'telemetry',
      ts: Date.now() / 1000,
      gps: { ..._gps },
      imu: { ..._imu },
      compass: _compass,
      battery: Math.round(_battery.level * 100),
      signal: _estimateSignal(),
    };
  }

  function getGPS()     { return { ..._gps }; }
  function getCompass() { return _compass; }
  function getBattery() { return Math.round(_battery.level * 100); }
  function getSpeed()   { return _speed_mps; }

  // ── Helpers ───────────────────────────────────────────────────
  function _estimateSignal() {
    if (!navigator.connection) return 2;
    const type = navigator.connection.effectiveType;
    if (type === '4g') return 4;
    if (type === '3g') return 3;
    if (type === '2g') return 2;
    return 1;
  }

  function _haversine(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2)**2 +
              Math.cos(lat1*Math.PI/180) * Math.cos(lat2*Math.PI/180) *
              Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  }

  function _log(msg) {
    const el = document.getElementById('log');
    if (el) {
      const line = document.createElement('div');
      line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      el.appendChild(line);
      el.scrollTop = el.scrollHeight;
    }
    console.log('[SensorManager]', msg);
  }

  return {
    startAll,
    stopAll,
    getTelemetryPacket,
    getGPS,
    getCompass,
    getBattery,
    getSpeed,
  };
})();
