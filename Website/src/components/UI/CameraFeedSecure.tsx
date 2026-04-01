import { useEffect, useRef, useState } from "react";
import "../../styles/cameraFeed.css";
import { useAuth } from "../../services/authService";

const API_BASE = import.meta.env.VITE_CAMERA_API_BASE as string | undefined;
const DEVICE_ID = (import.meta.env.VITE_DEVICE_ID as string | undefined) || "jetson-001";

type CameraMode = "raw" | "detection";

export default function CameraFeedSecure() {
  const { token } = useAuth();

  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [src, setSrc] = useState("");
  const [mode, setMode] = useState<CameraMode>("raw");
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState("Idle");

  const mountedRef = useRef(false);
  const refreshTimerRef = useRef<number | null>(null);
  const metaTimerRef = useRef<number | null>(null);

  const base = (API_BASE || "").replace(/\/+$/, "");

  const authHeaders = () => ({
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  });

  const buildFrameUrl = () => {
    if (!base) return "";
    return `${base}/camera/latest?device_id=${encodeURIComponent(DEVICE_ID)}&t=${Date.now()}&r=${Math.random()}`;
  };

  const hardClose = () => {
    setSrc("");
    setOk(false);
  };

  const pollMeta = async () => {
    if (!base) return;

    try {
      const res = await fetch(
        `${base}/camera/latest/meta?device_id=${encodeURIComponent(DEVICE_ID)}`,
        {
          method: "GET",
          credentials: "include",
          headers: authHeaders(),
        }
      );

      const data = await res.json().catch(() => null);

      if (!res.ok || !data?.available || !data?.fresh) {
        setOk(false);
        setStatusText("Disconnected");
        return;
      }

      setOk(true);
      setStatusText(
        data.mode === "detection" ? "Detection mode active" : "Raw mode active"
      );
    } catch {
      setOk(false);
      setStatusText("Disconnected");
    }
  };

  const startPolling = () => {
    if (refreshTimerRef.current) {
      window.clearInterval(refreshTimerRef.current);
    }
    if (metaTimerRef.current) {
      window.clearInterval(metaTimerRef.current);
    }

    refreshTimerRef.current = window.setInterval(() => {
      if (!mountedRef.current) return;
      setSrc(buildFrameUrl());
    }, 350);

    metaTimerRef.current = window.setInterval(() => {
      if (!mountedRef.current) return;
      void pollMeta();
    }, 1000);
  };

  const stopPolling = () => {
    if (refreshTimerRef.current) {
      window.clearInterval(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }
    if (metaTimerRef.current) {
      window.clearInterval(metaTimerRef.current);
      metaTimerRef.current = null;
    }
  };

  const activateCamera = async (newMode: CameraMode) => {
    if (!base) return;

    setBusy(true);
    setErr(null);
    setStatusText(`Starting ${newMode}...`);

    try {
      const res = await fetch(
        `${base}/camera/control/activate?device_id=${encodeURIComponent(DEVICE_ID)}&mode=${encodeURIComponent(newMode)}`,
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
      setStatusText(newMode === "raw" ? "Raw mode active" : "Detection mode active");
      setSrc(buildFrameUrl());
      startPolling();
      void pollMeta();
    } catch (e: any) {
      setErr(e?.message || "Failed to activate camera");
      setStatusText("Camera start failed");
      hardClose();
    } finally {
      setBusy(false);
    }
  };

  const setCameraMode = async (newMode: CameraMode) => {
    if (!base) return;
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
      // ignore on unmount
    } finally {
      stopPolling();
      setStatusText("Camera off");
      hardClose();
    }
  };

  useEffect(() => {
    mountedRef.current = true;

    void activateCamera("raw");

    const onFocus = () => {
      if (!mountedRef.current) return;
      setSrc(buildFrameUrl());
      void pollMeta();
    };

    const onVis = () => {
      if (!mountedRef.current) return;
      if (document.visibilityState === "visible") {
        setSrc(buildFrameUrl());
        void pollMeta();
      }
    };

    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVis);

    return () => {
      mountedRef.current = false;
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVis);
      stopPolling();
      void deactivateCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!API_BASE) {
    return (
      <section className="camera-feed-card">
        <div className="camera-feed-header">
          <h2>Live Camera Feed</h2>
          <div className="camera-status-badge disconnected">● Missing VITE_CAMERA_API_BASE</div>
        </div>
        <p>Add `VITE_CAMERA_API_BASE` and `VITE_DEVICE_ID` to your frontend env.</p>
      </section>
    );
  }

  return (
    <section className="camera-feed-card">
      <div className="camera-feed-header">
        <h2>Live Camera Feed</h2>
        <div className={`camera-status-badge ${ok ? "connected" : "disconnected"}`}>
          ● {ok ? "Connected" : "Disconnected"}
        </div>
      </div>

      <div style={{ marginBottom: 12, fontWeight: 700 }}>{statusText}</div>

      <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <button
          onClick={() => void setCameraMode("raw")}
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
          onClick={() => void setCameraMode("detection")}
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
          onClick={() => void activateCamera(mode)}
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

      {src ? (
        <img
          className="camera-frame"
          src={src}
          alt="AURA live feed"
          onLoad={() => {
            setOk(true);
            setErr(null);
          }}
          onError={() => {
            setOk(false);
            setErr("Frame unavailable");
            setStatusText("Disconnected");
          }}
        />
      ) : (
        <div className="camera-placeholder">Connecting...</div>
      )}

      {err && <div style={{ marginTop: 12, color: "crimson", fontWeight: 700 }}>{err}</div>}
    </section>
  );
}