import { useEffect, useMemo, useRef, useState } from "react";
import "../../styles/cameraFeed.css";
import { useAuth } from "../../services/authService";

const API_BASE = import.meta.env.VITE_CAMERA_API_BASE as string | undefined;
const DEVICE_ID =
  (import.meta.env.VITE_DEVICE_ID as string | undefined) || "jetson-001";

type CameraMode = "raw" | "detection" | "colorcode" | "face";

type CameraMeta = {
  ok?: boolean;
  device_id?: string;
  available?: boolean;
  mode?: CameraMode;
  updated_at?: number;
  bytes?: number;
};

const modeLabel = (mode: CameraMode) => {
  switch (mode) {
    case "raw":
      return "Raw";
    case "detection":
      return "Detection";
    case "colorcode":
      return "Color Code";
    case "face":
      return "Face";
    default:
      return mode;
  }
};

export default function CameraFeedSecure() {
  const { token } = useAuth();

  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<CameraMode>("raw");
  const [busy, setBusy] = useState(false);
  const [streamNonce, setStreamNonce] = useState(0);
  const [statusText, setStatusText] = useState("Starting camera...");

  const mountedRef = useRef(true);
  const metaTimerRef = useRef<number | null>(null);

  const base = (API_BASE || "").replace(/\/+$/, "");

  const authHeaders = () => ({
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  });

  const streamSrc = useMemo(() => {
    if (!base) return "";
    return `${base}/camera/stream?device_id=${encodeURIComponent(
      DEVICE_ID
    )}&mode=${encodeURIComponent(mode)}&t=${streamNonce}`;
  }, [base, mode, streamNonce]);

  const metaUrl = useMemo(() => {
    if (!base) return "";
    return `${base}/camera/latest/meta?device_id=${encodeURIComponent(
      DEVICE_ID
    )}`;
  }, [base]);

  const activateCamera = async (newMode: CameraMode) => {
    if (!base) return;

    setBusy(true);
    setErr(null);
    setStatusText(`Starting ${modeLabel(newMode)}...`);

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

      if (!mountedRef.current) return;

      setMode(newMode);
      setOk(false);
      setErr(null);
      setStatusText(`${modeLabel(newMode)} active`);
      setStreamNonce((n) => n + 1);
    } catch (e: any) {
      if (!mountedRef.current) return;
      setErr(e?.message || "Failed to activate camera");
      setStatusText("Camera start failed");
      setOk(false);
    } finally {
      if (!mountedRef.current) return;
      setBusy(false);
    }
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
          keepalive: true,
        }
      );
    } catch {
      // ignore
    } finally {
      if (!mountedRef.current) return;
      setOk(false);
      setStatusText("Camera off");
    }
  };

  const setCameraMode = async (newMode: CameraMode) => {
    if (busy) return;
    if (mode === newMode) return;
    await activateCamera(newMode);
  };

  useEffect(() => {
    mountedRef.current = true;

    const start = async () => {
      await activateCamera("raw");
    };

    start();

    const onPageHide = () => {
      fetch(
        `${base}/camera/control/deactivate?device_id=${encodeURIComponent(
          DEVICE_ID
        )}`,
        {
          method: "POST",
          credentials: "include",
          headers: authHeaders(),
          keepalive: true,
        }
      ).catch(() => {});
    };

    window.addEventListener("pagehide", onPageHide);

    return () => {
      mountedRef.current = false;
      if (metaTimerRef.current) window.clearInterval(metaTimerRef.current);
      window.removeEventListener("pagehide", onPageHide);
      deactivateCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!base || !metaUrl) return;

    const pollMeta = async () => {
      try {
        const res = await fetch(metaUrl, {
          credentials: "include",
          headers: authHeaders(),
          cache: "no-store",
        });

        const data = (await res.json()) as CameraMeta;

        if (!mountedRef.current) return;

        if (!data?.available) {
          setOk(false);
          setStatusText("Waiting for camera frame...");
          return;
        }

        const isFresh =
          typeof data.updated_at === "number"
            ? Date.now() / 1000 - data.updated_at < 2
            : false;

        if (typeof data.mode === "string" && data.mode !== mode) {
          setMode(data.mode);
          setStreamNonce((n) => n + 1);
        }

        setStatusText(
          isFresh ? `${modeLabel((data.mode || mode) as CameraMode)} active` : "Camera paused"
        );
        setOk(isFresh && !!data.available);
        setErr(null);
      } catch {
        if (!mountedRef.current) return;
        setOk(false);
        setStatusText("Camera disconnected");
      }
    };

    pollMeta();
    metaTimerRef.current = window.setInterval(pollMeta, 500);

    return () => {
      if (metaTimerRef.current) window.clearInterval(metaTimerRef.current);
    };
  }, [base, metaUrl, mode, token]);

  if (!API_BASE) {
    return (
      <div className="cam-card">
        <div className="cam-header">
          <h2>Live Camera Feed</h2>
        </div>
        <div className="cam-status error">Missing VITE_CAMERA_API_BASE</div>
        <p>Set VITE_CAMERA_API_BASE in your frontend env.</p>
      </div>
    );
  }

  return (
    <div className="cam-card">
      <div className="cam-header">
        <h2>Live Camera Feed</h2>
        <div className={`cam-status ${ok ? "ok" : "error"}`}>
          ● {ok ? "Connected" : "Disconnected"}
        </div>
      </div>

      <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 10 }}>
        {statusText}
      </div>

      <div
        className="cam-toolbar"
        style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}
      >
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
          onClick={() => setCameraMode("colorcode")}
          disabled={busy || mode === "colorcode"}
          className={`cam-btn ${mode === "colorcode" ? "active" : ""}`}
        >
          Color Code
        </button>

        <button
          onClick={() => setCameraMode("face")}
          disabled={busy || mode === "face"}
          className={`cam-btn ${mode === "face" ? "active" : ""}`}
        >
          Face
        </button>

        <button
          onClick={() => setStreamNonce((n) => n + 1)}
          disabled={busy}
          className="cam-btn"
        >
          Refresh
        </button>
      </div>

      <div className="cam-frame" style={{ position: "relative" }}>
        {streamSrc ? (
          <img
            key={`${mode}-${streamNonce}`}
            className="cam-img"
            src={streamSrc}
            alt="AURA camera stream"
            onLoad={() => {
              if (!mountedRef.current) return;
              setOk(true);
              setErr(null);
            }}
            onError={() => {
              if (!mountedRef.current) return;
              setOk(false);
              setErr("Stream unavailable");
            }}
          />
        ) : (
          <div className="cam-placeholder">
            <div>Waiting for camera frame...</div>
            <div style={{ marginTop: 8, fontSize: 13, opacity: 0.8 }}>
              The Jetson camera is starting up.
            </div>
          </div>
        )}
      </div>

      {err && (
        <div className="cam-error" style={{ marginTop: 12 }}>
          {err}
        </div>
      )}
    </div>
  );
}