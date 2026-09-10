/**
 * calibration.js
 * ---------------
 * Camera calibration UI and logic.
 *
 * Captures N frames of a reference scene (grey card / grass patch),
 * sends them to the server for calibration profile computation.
 */

'use strict';

const CalibrationManager = (() => {
  let _active = false;
  let _frames = [];
  const TARGET_FRAMES = 15;
  let _videoEl = null;
  let _serverUrl = '';
  let _method = 'grass_patch';

  function start(videoEl, serverUrl, method = 'grass_patch') {
    _videoEl = videoEl;
    _serverUrl = serverUrl;
    _method = method;
    _frames = [];
    _active = true;

    document.getElementById('calib-overlay').classList.add('show');
    _updateInstruction(method);
    _captureLoop();
  }

  function cancel() {
    _active = false;
    _frames = [];
    document.getElementById('calib-overlay').classList.remove('show');
  }

  async function _captureLoop() {
    while (_active && _frames.length < TARGET_FRAMES) {
      const blob = await _captureFrame();
      if (blob) {
        _frames.push(blob);
        const pct = (_frames.length / TARGET_FRAMES) * 100;
        document.getElementById('calib-fill').style.width = pct + '%';
        document.getElementById('calib-count').textContent =
          `${_frames.length} / ${TARGET_FRAMES} frames`;
      }
      await _sleep(200);
    }

    if (!_active) return;

    // Upload to server
    await _uploadCalibration();
    document.getElementById('calib-overlay').classList.remove('show');
    _active = false;
  }

  async function _captureFrame() {
    if (!_videoEl || !_videoEl.srcObject) return null;
    const canvas = document.createElement('canvas');
    canvas.width  = _videoEl.videoWidth  || 320;
    canvas.height = _videoEl.videoHeight || 240;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(_videoEl, 0, 0);
    return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
  }

  async function _uploadCalibration() {
    const formData = new FormData();
    _frames.forEach((blob, i) => {
      formData.append('frames', blob, `calib_${i}.jpg`);
    });
    formData.append('method', _method);

    try {
      await fetch(`${_serverUrl}/api/calibrate`, {
        method: 'POST',
        body: formData,
      });
    } catch (e) {
      console.warn('[Calibration] Upload failed:', e);
    }
  }

  function _updateInstruction(method) {
    const el = document.getElementById('calib-instruction');
    if (!el) return;
    const msgs = {
      grass_patch: 'Point camera at healthy green crop or grass.<br>Hold steady for 3 seconds.',
      grey_card:   'Hold an 18% grey card in front of camera.<br>Fill the entire frame.',
      auto_white:  'Point camera at a grey or white surface.<br>Avoid direct sunlight.',
    };
    el.innerHTML = msgs[method] || msgs['auto_white'];
  }

  function _sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  return { start, cancel };
})();
