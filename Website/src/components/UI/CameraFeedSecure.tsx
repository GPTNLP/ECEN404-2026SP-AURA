import { useEffect, useMemo, useRef, useState } from "react";
import "../../styles/cameraFeed.css";
import { useAuth } from "../../services/authService";

const API_BASE = import.meta.env.VITE_CAMERA_API_BASE as string | undefined;
const DEVICE_ID =
  (import.meta.env.VITE_DEVICE_ID as string | undefined) || "jetson-001";

type CameraMeta = {
  ok?: boolean;
  device_id?: string;
  available?: boolean;
  mode?: string;
  updated_at?: number;
  bytes?: number;
};

export default function CameraFeedSecure() {
  const { token } = useAuth();

  const [ok, setOk] = useState(false);
  const [err, setErr] = useState<string | null>(null);
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
    )}&t=${streamNonce}`;
  }, [base, streamNonce]);

  const metaUrl = useMemo(() => {
    if (!base) return "";
    return `${base}/camera/latest/meta?device_id=${encodeURIComponent(
      DEVICE_ID
    )}`;
  }, [base]);

  const activateCamera = async () => {
    if (!base) return;

    setBusy(true);
    setErr(null);
    setStatusText("Starting raw camera...");

    try {
      const res = await fetch(
        `${base}/camera/control/activate?device_id=${encodeURIComponent(
          DEVICE_ID
        )}`,
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

      setOk(false);
      setErr(null);
      setStatusText("Raw camera active");
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

  useEffect(() => {
    mountedRef.current = true;

    const start = async () => {
      await activateCamera();
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

        setStatusText(isFresh ? "Raw camera active" : "Camera paused");
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
  }, [base, metaUrl, token]);

  if (!API_BASE) {
    return (
      <div className="cam-shell">
        <div className="cam-card">
          <div className="cam-topbar">
            <div className="cam-left">
              <h2 className="cam-title">Live Camera Feed</h2>
            </div>
          </div>
          <div className="cam-error-box">Missing VITE_CAMERA_API_BASE</div>
        </div>
      </div>
    );
  }

  return (
    <div className="cam-shell">
      <div className="cam-card">
        <div className="cam-topbar">
          <div className="cam-left">
            <h2 className="cam-title">Live Camera Feed</h2>
            <div className="cam-substatus">{statusText}</div>
          </div>

          <div className="cam-right">
            <div className={`cam-connection ${ok ? "ok" : "error"}`}>
              <span className="cam-connection-dot" />
              {ok ? "Connected" : "Disconnected"}
            </div>
          </div>
        </div>

        <div className="cam-frame">
          {streamSrc ? (
            <img
              key={streamNonce}
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
              <div className="cam-placeholder-sub">
                The Jetson camera is starting up.
              </div>
            </div>
          )}

          {!ok && !busy && (
            <div className="cam-overlay-message">
              <div className="cam-overlay-title">Camera offline</div>
              <div className="cam-overlay-sub">Waiting for Jetson stream...</div>
            </div>
          )}
        </div>

        {err && <div className="cam-error-box">{err}</div>}
      </div>
    </div>
  );
}