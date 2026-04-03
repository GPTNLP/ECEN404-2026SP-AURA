import { useEffect, useMemo, useState } from "react";
import "../../styles/cameraFeed.css";
import { useAuth } from "../../services/authService";

const API_BASE = import.meta.env.VITE_CAMERA_API_BASE as string | undefined;
const DEVICE_ID =
  (import.meta.env.VITE_DEVICE_ID as string | undefined) || "jetson-001";

type CameraMode = "raw" | "detection";

export default function CameraFeedSecure() {
  const { token } = useAuth();

  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<CameraMode>("raw");
  const [busy, setBusy] = useState(false);
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
      setStreamNonce((n) => n + 1);
    } catch (e: any) {
      setErr(e?.message || "Failed to activate camera");
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
        `${base}/camera/control/deactivate?device_id=${encodeURIComponent(
          DEVICE_ID
        )}`,
        {
          method: "POST",
          credentials: "include",
          headers: authHeaders(),
        }
      );
    } catch {
      // ignore
    } finally {
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
      <div className="cam-card">
        <div className="cam-card-header">
          <div className="cam-title">Live Camera Feed</div>
          <div className="cam-status bad">● Missing VITE_CAMERA_API_BASE</div>
        </div>
        <div className="cam-help">Set VITE_CAMERA_API_BASE in your frontend env.</div>
      </div>
    );
  }

  return (
    <div className="cam-card">
      <div className="cam-card-header">
        <div className="cam-title">Live Camera Feed</div>

        <div className="cam-toolbar">
          <div className={`cam-status ${ok ? "good" : "bad"}`}>
            ● {ok ? "Connected" : "Disconnected"}
          </div>

          <button
            onClick={() => setCameraMode("raw")}
            disabled={busy || mode === "raw"}
            className={`cam-btn ${mode === "raw" ? "active" : ""}`}
          >
            Raw
          </button>

          <button
            onClick={() => setCameraMode("detection")}
            disabled={busy || mode === "detection"}
            className={`cam-btn ${mode === "detection" ? "active" : ""}`}
          >
            Detection
          </button>

          <button
            onClick={() => setStreamNonce((n) => n + 1)}
            disabled={busy}
            className="cam-btn"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="cam-frame">
        {ok || !err ? (
          <img
            key={`${mode}-${streamNonce}`}
            className="cam-img"
            src={streamSrc}
            alt="Stream"
            onLoad={() => {
              setOk(true);
              setErr(null);
            }}
            onError={() => {
              setOk(false);
              setErr("Stream unavailable");
            }}
          />
        ) : (
          <div className="cam-placeholder">
            <div className="cam-placeholder-title">Camera stream unavailable</div>
            <div className="cam-placeholder-subtitle">
              Check the Jetson camera service, then press Refresh.
            </div>
          </div>
        )}
      </div>

      {err && <div className="cam-error">{err}</div>}
    </div>
  );
}