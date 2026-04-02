import { useEffect, useMemo, useState } from "react";
import "../../styles/cameraFeed.css";
import { useAuth } from "../../services/authService";

const API_BASE = import.meta.env.VITE_CAMERA_API_BASE as string | undefined;
const DEVICE_ID = (import.meta.env.VITE_DEVICE_ID as string | undefined) || "jetson-001";

type CameraMode = "raw" | "detection";

export default function CameraFeedSecure() {
  const { token } = useAuth();

  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<CameraMode>("raw");
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState("Idle");
  const [streamNonce, setStreamNonce] = useState(0);

  const base = (API_BASE || "").replace(/\/+$/, "");

  const authHeaders = () => ({
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  });

  const streamSrc = useMemo(() => {
    if (!base) return "";
    return `${base}/camera/stream?device_id=${encodeURIComponent(
      DEVICE_ID
    )}&mode=${encodeURIComponent(mode)}&n=${streamNonce}`;
  }, [base, mode, streamNonce]);

  const activateCamera = async (newMode: CameraMode) => {
    if (!base) return;

    setBusy(true);
    setErr(null);
    setStatusText(`Starting ${newMode}...`);

    try {
      const res = await fetch(
        `${base}/camera/control/activate?device_id=${encodeURIComponent(
          DEVICE_ID
        )}&mode=${encodeURIComponent(newMode)}`,
        {
          method: "POST",
          credentials: "include",
          headers: authHeaders(),
        }
      );

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        throw new Error(data?.detail || `Activate failed (${res.status})`);
      }

      setMode(newMode);
      setOk(false);
      setErr(null);
      setStatusText(newMode === "raw" ? "Raw mode active" : "Detection mode active");
      setStreamNonce((n) => n + 1);
    } catch (e: any) {
      setErr(e?.message || "Failed to activate camera");
      setStatusText("Camera start failed");
      setOk(false);
    } finally {
      setBusy(false);
    }
  };

  const setCameraMode = async (newMode: CameraMode) => {
    if (busy) return;
    if (mode === newMode) return;
    await activateCamera(newMode);
  };

  const deactivateCamera = async () => {
    if (!base) return;

    try {
      await fetch(
        `${base}/camera/control/deactivate?device_id=${encodeURIComponent(DEVICE_ID)}`,
        {
          method: "POST",
          credentials: "include",
          headers: authHeaders(),
        }
      );
    } catch {
      // ignore
    } finally {
      setStatusText("Camera off");
      setOk(false);
      setStreamNonce((n) => n + 1);
    }
  };

  useEffect(() => {
    activateCamera("raw");

    return () => {
      deactivateCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!API_BASE) {
    return (
      <section className="camera-feed-card">
        <div className="camera-feed-header">
          <h2>Live Camera Feed</h2>
          <div className="camera-status-badge off">● Missing VITE_CAMERA_API_BASE</div>
        </div>
        <div className="camera-feed-body">
          <p>Set VITE_CAMERA_API_BASE in your frontend env.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="camera-feed-card">
      <div className="camera-feed-header">
        <div>
          <h2>Live Camera Feed</h2>
          <div className={`camera-status-badge ${ok ? "on" : "off"}`}>
            ● {ok ? "Connected" : "Disconnected"}
          </div>
        </div>

        <div className="camera-controls-inline" style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button
            onClick={() => setCameraMode("raw")}
            disabled={busy || mode === "raw"}
            style={{
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: mode === "raw" ? "var(--accent, #dbeafe)" : "var(--card-bg)",
              fontWeight: 900,
              cursor: busy || mode === "raw" ? "not-allowed" : "pointer",
              opacity: busy || mode === "raw" ? 0.7 : 1,
            }}
          >
            Raw
          </button>

          <button
            onClick={() => setCameraMode("detection")}
            disabled={busy || mode === "detection"}
            style={{
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: mode === "detection" ? "var(--accent, #dbeafe)" : "var(--card-bg)",
              fontWeight: 900,
              cursor: busy || mode === "detection" ? "not-allowed" : "pointer",
              opacity: busy || mode === "detection" ? 0.7 : 1,
            }}
          >
            Detection
          </button>

          <button
            onClick={() => setStreamNonce((n) => n + 1)}
            disabled={busy}
            style={{
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: "var(--card-bg)",
              fontWeight: 900,
              cursor: busy ? "not-allowed" : "pointer",
              opacity: busy ? 0.7 : 1,
            }}
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="camera-feed-body">
        <div style={{ marginBottom: 10, fontWeight: 700 }}>{statusText}</div>

        <img
          key={`${mode}-${streamNonce}`}
          src={streamSrc}
          alt="AURA live camera"
          className="camera-feed-image"
          onLoad={() => {
            setOk(true);
            setErr(null);
          }}
          onError={() => {
            setOk(false);
            setErr("Stream unavailable");
          }}
        />

        {err && (
          <div style={{ marginTop: 12, color: "#dc2626", fontWeight: 700 }}>
            {err}
          </div>
        )}
      </div>
    </section>
  );
}