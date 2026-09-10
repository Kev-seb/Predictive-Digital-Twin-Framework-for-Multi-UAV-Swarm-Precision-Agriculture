/**
 * webrtc_client.js
 * -----------------
 * WebRTC peer connection client.
 *
 * Connects to the server's /ws/signal WebSocket for signaling,
 * then establishes a peer-to-peer WebRTC video connection.
 *
 * Falls back to JPEG upload mode if:
 *   - WebRTC is not supported
 *   - SDP negotiation fails
 *   - ICE connection fails within 8 seconds
 */

'use strict';

const WebRTCClient = (() => {
  let _pc = null;
  let _signalingWs = null;
  let _localStream = null;
  let _connected = false;
  let _fallback = false;
  let _serverUrl = '';
  let _onStateChange = null;

  const ICE_TIMEOUT_MS = 8000;

  // ── Public API ─────────────────────────────────────────────────

  async function connect(serverUrl, localStream, onStateChange) {
    _serverUrl = serverUrl;
    _localStream = localStream;
    _onStateChange = onStateChange;
    _fallback = false;

    // Check WebRTC support
    if (typeof RTCPeerConnection === 'undefined') {
      _log('WebRTC not supported — using JPEG fallback');
      _fallback = true;
      _onStateChange?.('fallback');
      return false;
    }

    try {
      await _openSignaling(serverUrl);
      return true;
    } catch (e) {
      _log('WebRTC signaling failed: ' + e.message + ' — using JPEG fallback');
      _fallback = true;
      _onStateChange?.('fallback');
      return false;
    }
  }

  function disconnect() {
    _pc?.close();
    _signalingWs?.close();
    _pc = null;
    _signalingWs = null;
    _connected = false;
  }

  function isConnected()   { return _connected; }
  function isFallback()    { return _fallback; }

  // ── Signaling WebSocket ────────────────────────────────────────

  async function _openSignaling(serverUrl) {
    const wsUrl = serverUrl.replace(/^http/, 'ws') + '/ws/signal';
    _signalingWs = new WebSocket(wsUrl);

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Signaling timeout')), 5000);
      _signalingWs.onopen = () => { clearTimeout(timeout); resolve(); };
      _signalingWs.onerror = (e) => { clearTimeout(timeout); reject(new Error('WS error')); };
    });

    _signalingWs.onmessage = async (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'answer') {
        await _pc?.setRemoteDescription(new RTCSessionDescription(msg));
      } else if (msg.type === 'ice_candidate' && msg.candidate) {
        await _pc?.addIceCandidate(new RTCIceCandidate(msg.candidate));
      }
    };

    _signalingWs.onclose = () => {
      _connected = false;
      _onStateChange?.('disconnected');
    };

    await _createPeerConnection();
  }

  // ── Peer Connection ───────────────────────────────────────────

  async function _createPeerConnection() {
    // LAN-only: no STUN/TURN needed
    _pc = new RTCPeerConnection({
      iceServers: [],
      iceTransportPolicy: 'all',
    });

    // Add video track from camera
    if (_localStream) {
      _localStream.getTracks().forEach(track => {
        _pc.addTrack(track, _localStream);
      });
    }

    // ICE candidate → send to server
    _pc.onicecandidate = (e) => {
      if (e.candidate && _signalingWs?.readyState === WebSocket.OPEN) {
        _signalingWs.send(JSON.stringify({
          type: 'ice_candidate',
          candidate: e.candidate.toJSON(),
        }));
      }
    };

    _pc.onconnectionstatechange = () => {
      const state = _pc.connectionState;
      _log('WebRTC state: ' + state);
      if (state === 'connected') {
        _connected = true;
        _fallback = false;
        _onStateChange?.('connected');
      } else if (state === 'failed' || state === 'disconnected') {
        _connected = false;
        _fallback = true;
        _onStateChange?.('fallback');
      }
    };

    // ICE connection timeout → fallback
    const iceTimeout = setTimeout(() => {
      if (!_connected) {
        _log('ICE timeout — switching to JPEG fallback');
        _fallback = true;
        _onStateChange?.('fallback');
      }
    }, ICE_TIMEOUT_MS);

    _pc.onconnectionstatechange = () => {
      if (_pc.connectionState === 'connected') {
        clearTimeout(iceTimeout);
        _connected = true;
        _onStateChange?.('connected');
      }
    };

    // Create offer
    const offer = await _pc.createOffer({ offerToReceiveVideo: false });
    await _pc.setLocalDescription(offer);

    // Send offer to server via signaling
    _signalingWs.send(JSON.stringify({
      type: 'offer',
      sdp: _pc.localDescription.sdp,
    }));
  }

  function _log(msg) {
    console.log('[WebRTCClient]', msg);
    const el = document.getElementById('log');
    if (el) {
      const line = document.createElement('div');
      line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      el.appendChild(line);
      el.scrollTop = el.scrollHeight;
    }
  }

  return { connect, disconnect, isConnected, isFallback };
})();
