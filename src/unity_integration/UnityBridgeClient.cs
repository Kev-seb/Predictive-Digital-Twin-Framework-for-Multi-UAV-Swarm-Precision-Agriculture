using System;
using System.Collections;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace DigitalTwin.UnityIntegration
{
    [RequireComponent(typeof(Rigidbody))]
    public class UnityBridgeClient : MonoBehaviour
    {
        [Header("Server Settings")]
        [SerializeField] private string apiHost = "127.0.0.1";
        [SerializeField] private int apiPort = 8080;
        [SerializeField] private string droneId = "drone_01";

        [Header("Perception (Stereo Cameras)")]
        [SerializeField] private Camera leftCamera;
        [SerializeField] private Camera rightCamera;
        [SerializeField] private float sendFrameInterval = 2.0f; // Seconds
        [SerializeField] private float focalLength = 1200f;      // Pixel representation
        [SerializeField] private float baseline = 0.5f;          // Distance between cameras

        [Header("Control Parameters")]
        [SerializeField] private float sendTelemetryInterval = 0.05f; // 20 Hz
        [SerializeField] private float commandTimeoutMs = 500f;       // Fail-safe limit

        // Internal State
        private Rigidbody rb;
        private ClientWebSocket ws;
        private Uri wsUri;
        private string httpUrl;
        
        private Vector3 commandedVelocity = Vector3.zero;
        private float lastCommandTime = 0f;
        private int telemetrySeq = 0;
        private int frameSeq = 100;
        private bool isConnecting = false;

        // JSON Contracts
        [Serializable]
        private class TelemetryPayload
        {
            public string type = "telemetry";
            public string drone_id;
            public int telemetry_seq;
            public float[] position;
            public float[] velocity;
            public float[] orientation;
            public float battery_pct = 95f;
            public double timestamp;
        }

        [Serializable]
        private class CommandPayload
        {
            public string type;
            public string drone_id;
            public int telemetry_seq;
            public float[] velocity;
            public float[] acceleration;
            public string control_mode;
            public double timestamp;
        }

        [Serializable]
        private class ImageMetadata
        {
            public string drone_id;
            public int frame_id;
            public double timestamp;
            public float focal_length;
            public float baseline;
            public float[] position;
            public float[] orientation;
        }

        private void Start()
        {
            rb = GetComponent<Rigidbody>();
            rb.useGravity = false; // Controlled externally by potential field forces

            wsUri = new Uri($"ws://{apiHost}:{apiPort}/ws/telemetry");
            httpUrl = $"http://{apiHost}:{apiPort}/api/frame/stereo";

            StartCoroutine(ReconnectWebSocketLoop());
            StartCoroutine(TelemetryLoop());
            StartCoroutine(FrameCaptureLoop());
        }

        private void Update()
        {
            // Fail-safe timeout validation
            if (Time.time - lastCommandTime > (commandTimeoutMs / 1000f))
            {
                // Safety hover / Stop moving
                commandedVelocity = Vector3.zero;
            }
        }

        private void FixedUpdate()
        {
            // Execute the commanded potential-field velocity safely on the Rigidbody
            if (rb != null)
            {
                rb.velocity = commandedVelocity;
            }
        }

        private void OnDestroy()
        {
            DisconnectWebSocket();
        }

        #region WebSocket Telemetry Channel

        private IEnumerator ReconnectWebSocketLoop()
        {
            while (true)
            {
                if (ws == null || ws.State != WebSocketState.Open)
                {
                    if (!isConnecting)
                    {
                        StartCoroutine(ConnectWebSocket());
                    }
                }
                yield return new WaitForSeconds(5.0f); // Check connection every 5s
            }
        }

        private IEnumerator ConnectWebSocket()
        {
            isConnecting = true;
            Debug.Log($"[UnityBridge] Connecting to Telemetry WebSocket at {wsUri}...");
            ws = new ClientWebSocket();
            
            Task connectTask = ws.ConnectAsync(wsUri, CancellationToken.None);
            yield return new WaitUntil(() => connectTask.IsCompleted);

            if (connectTask.IsFaulted || ws.State != WebSocketState.Open)
            {
                Debug.LogWarning($"[UnityBridge] WebSocket Connection failed: {connectTask.Exception?.Message}");
                isConnecting = false;
                yield break;
            }

            Debug.Log("[UnityBridge] Telemetry WebSocket connection established.");
            isConnecting = false;
            
            // Spawn WebSocket receiver loop
            Task.Run(ReceiveCommandsLoop);
        }

        private async Task ReceiveCommandsLoop()
        {
            byte[] buffer = new byte[4096];
            while (ws != null && ws.State == WebSocketState.Open)
            {
                try
                {
                    WebSocketReceiveResult result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                        break;
                    }

                    string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    ProcessCommandMessage(message);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[UnityBridge] Receive error: {e.Message}");
                    break;
                }
            }
        }

        private void ProcessCommandMessage(string json)
        {
            try
            {
                CommandPayload cmd = JsonUtility.FromJson<CommandPayload>(json);
                if (cmd != null && cmd.type == "velocity_command")
                {
                    // Map 3D coordinate vector from Python back to Unity Space
                    // Python: [x_local, y_local, z_altitude] -> Unity: [x, z, y] or match axes directly
                    // By default, we keep coordinates 1:1 mapped to avoid axis confusion
                    Vector3 targetVel = new Vector3(cmd.velocity[0], cmd.velocity[1], cmd.velocity[2]);
                    
                    // Thread-safe update
                    lock (this)
                    {
                        commandedVelocity = targetVel;
                        lastCommandTime = Time.time;
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[UnityBridge] JSON deserialization failed: {e.Message}");
            }
        }

        private IEnumerator TelemetryLoop()
        {
            while (true)
            {
                if (ws != null && ws.State == WebSocketState.Open)
                {
                    telemetrySeq++;
                    TelemetryPayload packet = new TelemetryPayload
                    {
                        drone_id = droneId,
                        telemetry_seq = telemetrySeq,
                        position = new float[] { transform.position.x, transform.position.y, transform.position.z },
                        velocity = new float[] { rb.velocity.x, rb.velocity.y, rb.velocity.z },
                        orientation = new float[] { transform.eulerAngles.x, transform.eulerAngles.y, transform.eulerAngles.z },
                        battery_pct = 94.0f,
                        timestamp = DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds
                    };

                    string json = JsonUtility.ToJson(packet);
                    byte[] bytes = Encoding.UTF8.GetBytes(json);
                    
                    Task sendTask = ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None);
                    yield return new WaitUntil(() => sendTask.IsCompleted);
                }

                yield return new WaitForSeconds(sendTelemetryInterval);
            }
        }

        private void DisconnectWebSocket()
        {
            if (ws != null)
            {
                ws.Dispose();
                ws = null;
            }
        }

        #endregion

        #region HTTP Perception Channel

        private IEnumerator FrameCaptureLoop()
        {
            while (true)
            {
                yield return new WaitForSeconds(sendFrameInterval);
                
                if (leftCamera == null || rightCamera == null) continue;

                frameSeq++;
                yield return StartCoroutine(CaptureAndSendStereoFrames(frameSeq));
            }
        }

        private IEnumerator CaptureAndSendStereoFrames(int id)
        {
            // Capture Left Viewport
            byte[] leftBytes = CaptureCameraBytes(leftCamera);
            yield return null;

            // Capture Right Viewport
            byte[] rightBytes = CaptureCameraBytes(rightCamera);
            yield return null;

            if (leftBytes == null || rightBytes == null) yield break;

            // Build Metadata
            ImageMetadata meta = new ImageMetadata
            {
                drone_id = droneId,
                frame_id = id,
                timestamp = DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds,
                focal_length = focalLength,
                baseline = baseline,
                position = new float[] { transform.position.x, transform.position.y, transform.position.z },
                orientation = new float[] { transform.eulerAngles.x, transform.eulerAngles.y, transform.eulerAngles.z }
            };

            string metaJson = JsonUtility.ToJson(meta);

            // Construct multipart form data request
            WWWForm form = new WWWForm();
            form.AddBinaryData("left_file", leftBytes, "left.jpg", "image/jpeg");
            form.AddBinaryData("right_file", rightBytes, "right.jpg", "image/jpeg");
            form.AddField("metadata", metaJson);

            using (UnityWebRequest www = UnityWebRequest.Post(httpUrl, form))
            {
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[UnityBridge] HTTP Perception POST failed: {www.error}");
                }
                else
                {
                    Debug.Log($"[UnityBridge] Stereo Frame {id} Ingested OK: {www.downloadHandler.text}");
                }
            }
        }

        private byte[] CaptureCameraBytes(Camera cam)
        {
            RenderTexture activeRT = RenderTexture.active;
            RenderTexture camRT = cam.targetTexture;

            if (camRT == null)
            {
                // Create temporary RenderTexture if none assigned to camera
                camRT = RenderTexture.GetTemporary(640, 480, 24);
                cam.targetTexture = camRT;
            }

            bool originallyEnabled = cam.enabled;
            cam.enabled = true;
            cam.Render();
            cam.enabled = originallyEnabled;
            RenderTexture.active = camRT;

            Texture2D tex = new Texture2D(camRT.width, camRT.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, camRT.width, camRT.height), 0, 0);
            tex.Apply();

            byte[] bytes = tex.EncodeToJPG(85);

            // Cleanup
            RenderTexture.active = activeRT;
            Destroy(tex);

            if (cam.targetTexture != camRT)
            {
                cam.targetTexture = null;
                RenderTexture.ReleaseTemporary(camRT);
            }

            return bytes;
        }

        #endregion
    }
}
