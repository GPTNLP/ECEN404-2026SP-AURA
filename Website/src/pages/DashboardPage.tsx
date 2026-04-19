// Website/src/pages/DashboardPage.tsx
import { useEffect, useMemo, useState } from "react";
import "../styles/dashboard.css";
import robotImage from "../assets/robot.png";
import { useAuth } from "../services/authService";

type HealthStatus = "OK" | "WARN" | "BAD";

type DeviceRecord = {
  device_id: string;
  device_name?: string;
  online?: boolean;
  last_seen_at?: number;
  status?: {
    battery_percent?: number | null;
    battery_voltage?: number | null;
    charging?: boolean | null;
    cpu_percent?: number | null;
    ram_percent?: number | null;
    disk_percent?: number | null;
    temperature_c?: number | null;
    camera_ready?: boolean | null;
    mic_ready?: boolean | null;
    speaker_ready?: boolean | null;
    ollama_ready?: boolean | null;
    vector_db_ready?: boolean | null;
    current_mode?: string | null;
    current_task?: string | null;
    updated_at?: number | null;
    extra?: {
      hostname?: string;
      local_ip?: string;
      uptime_seconds?: number;
      gpu_percent?: number;
      db_name?: string;
      esp32?: {
        port?: string;
        port_exists?: boolean;
        connected?: boolean;
        last_connect_ok?: boolean;
        last_error?: string;
      };
    };
  };
};

type AdminListResponse = {
  ok: boolean;
  count: number;
  items: DeviceRecord[];
};

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined)?.trim() ||
  "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";

const DEVICE_ID =
  (import.meta.env.VITE_DEVICE_ID as string | undefined)?.trim() || "jetson-001";

const LS_TOKEN = "aura-auth-token";
const STALE_AFTER_SECONDS = 15;

function thermalStatus(tempC?: number | null): HealthStatus {
  if (tempC == null) return "WARN";
  if (tempC < 70) return "OK";
  if (tempC < 82) return "WARN";
  return "BAD";
}

function formatUptime(sec?: number) {
  if (sec == null) return "Unknown";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return `${h}h ${m}m ${s}s`;
}

function formatUpdated(unixSeconds?: number) {
  if (!unixSeconds) return "";
  return new Date(unixSeconds * 1000).toLocaleTimeString();
}

function fmt(value?: number | null, suffix = "", digits = 1) {
  if (value == null || Number.isNaN(value)) return "Unknown";
  return `${value.toFixed(digits)}${suffix}`;
}

function staleHealth(): HealthStatus {
  return "WARN";
}

function statusBadgeText(status: HealthStatus) {
  if (status === "OK") return "Good";
  if (status === "WARN") return "Watch";
  return "Alert";
}

function statusBadgeClass(status: HealthStatus) {
  if (status === "OK") return "ok";
  if (status === "WARN") return "warn";
  return "bad";
}

export default function DashboardPage() {
  const { token, user } = useAuth();
  const canFlush = user?.role === "admin" || user?.role === "ta";

  const [device, setDevice] = useState<DeviceRecord | null>(null);
  const [error, setError] = useState("");
  const [nowMs, setNowMs] = useState(Date.now());

  const [flushState, setFlushState] = useState<"idle" | "pending" | "ok" | "err">("idle");
  const [flushNote, setFlushNote] = useState("");

  const [reloadState, setReloadState] = useState<"idle" | "pending" | "ok" | "err">("idle");
  const [reloadNote, setReloadNote] = useState("");

  useEffect(() => {
    let alive = true;

    async function load() {
      const storedToken = localStorage.getItem(LS_TOKEN);

      try {
        const res = await fetch(`${API_BASE}/device/admin/list`, {
          method: "GET",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            ...(storedToken ? { Authorization: `Bearer ${storedToken}` } : {}),
          },
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`${res.status} ${text}`);
        }

        const data: AdminListResponse = await res.json();
        const found =
          data.items.find((item) => item.device_id === DEVICE_ID) ??
          data.items[0] ??
          null;

        if (alive) {
          setDevice(found);
          setError("");
          setNowMs(Date.now());
        }
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard");
          setNowMs(Date.now());
        }
      }
    }

    load();
    const id = setInterval(load, 2000);
    const clock = setInterval(() => setNowMs(Date.now()), 1000);

    return () => {
      alive = false;
      clearInterval(id);
      clearInterval(clock);
    };
  }, []);

  const s = device?.status;
  const extra = s?.extra;

  const freshestTimestampSeconds = useMemo(() => {
    const candidates = [device?.last_seen_at, s?.updated_at].filter(
      (v): v is number => typeof v === "number" && v > 0
    );
    if (!candidates.length) return 0;
    return Math.max(...candidates);
  }, [device?.last_seen_at, s?.updated_at]);

  const ageSeconds = freshestTimestampSeconds
    ? Math.max(0, Math.floor(nowMs / 1000 - freshestTimestampSeconds))
    : Number.POSITIVE_INFINITY;

  const isFresh = freshestTimestampSeconds > 0 && ageSeconds <= STALE_AFTER_SECONDS;
  const isOnline = Boolean(device?.online) && isFresh;
  const status = isOnline ? "ONLINE" : "OFFLINE";

  const updatedLabel = useMemo(() => {
    if (!isFresh || !device?.last_seen_at) return "";
    return formatUpdated(device.last_seen_at);
  }, [device?.last_seen_at, isFresh]);

  const ramValue = isOnline ? fmt(s?.ram_percent, "%", 1) : "Unknown";
  const cpuValue = isOnline ? fmt(s?.cpu_percent, "%", 1) : "Unknown";
  const gpuValue = isOnline ? fmt(extra?.gpu_percent, "%", 1) : "Unknown";
  const uptimeValue = isOnline ? formatUptime(extra?.uptime_seconds) : "Unknown";

  const flushModels = async () => {
    if (!token || !device) return;

    setFlushState("pending");
    setFlushNote("");

    try {
      const res = await fetch(`${API_BASE}/device/admin/flush_models`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          device_id: device.device_id,
          command: "flush_models",
        }),
      });

      if (!res.ok) throw new Error(await res.text());

      setFlushState("ok");
      setFlushNote("Flush queued - Jetson will unload idle models on next poll.");
    } catch (err) {
      setFlushState("err");
      setFlushNote(err instanceof Error ? err.message : "Flush failed");
    }
  };

  const reloadLlm = async () => {
    if (!token || !device) return;

    setReloadState("pending");
    setReloadNote("");

    try {
      const res = await fetch(`${API_BASE}/device/admin/reload_llm`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          device_id: device.device_id,
          command: "reload_llm",
        }),
      });

      if (!res.ok) throw new Error(await res.text());

      setReloadState("ok");
      setReloadNote("Reload queued - Jetson will unload then reload the LLM to GPU.");
    } catch (err) {
      setReloadState("err");
      setReloadNote(err instanceof Error ? err.message : "Reload failed");
    }
  };

  const esp32 = extra?.esp32;
  const esp32Value = !isOnline
    ? "Unknown"
    : esp32?.connected
    ? "Connected"
    : esp32?.port_exists
    ? "Detected"
    : "Disconnected";

  const esp32Health: HealthStatus = !isOnline
    ? "WARN"
    : esp32?.connected
    ? "OK"
    : esp32?.port_exists
    ? "WARN"
    : "BAD";

  const thermalsValue = isOnline
    ? s?.temperature_c != null
      ? fmt(s.temperature_c, "°C", 1)
      : "Unknown"
    : "Unknown";

  const thermalsHealth = isOnline ? thermalStatus(s?.temperature_c) : staleHealth();

  return (
    <div className="dashboard-page">
      <section className="aura-header">
        <div className="aura-panel">
          <div className="aura-panel-left">
            <img src={robotImage} alt="AURA robot" className="aura-img" />

            <div className="aura-text">
              <h1 className="aura-title">{device?.device_name || "AURA Jetson"}</h1>

              <div className="aura-sub">
                Status:{" "}
                <span className={isOnline ? "status-online" : "status-offline"}>
                  {status}
                </span>
                {isOnline && updatedLabel ? (
                  <span> • Updated {updatedLabel}</span>
                ) : null}
                {!isOnline && freshestTimestampSeconds > 0 ? (
                  <span> • No fresh Jetson data</span>
                ) : null}
              </div>
            </div>
          </div>

          {canFlush ? (
            <div className="aura-panel-actions">
              <button
                className="dashboard-action-btn"
                onClick={() => void flushModels()}
                disabled={flushState === "pending" || !device}
                type="button"
                title="Free VRAM/RAM by unloading idle models."
              >
                {flushState === "pending" ? "Queuing..." : "Flush Models"}
              </button>

              <button
                className="dashboard-action-btn"
                onClick={() => void reloadLlm()}
                disabled={reloadState === "pending" || !device}
                type="button"
                title="Unload then force-reload the LLM onto the Jetson GPU."
              >
                {reloadState === "pending" ? "Queuing..." : "Reload LLM to GPU"}
              </button>
            </div>
          ) : null}
        </div>
      </section>

      {error ? (
        <div className="dash-error">Dashboard load failed: {error}</div>
      ) : null}

      {(flushNote || reloadNote) && (
        <section className="dashboard-section dashboard-maintenance-notes">
          {flushNote ? <div className="dashboard-note">{flushNote}</div> : null}
          {reloadNote ? <div className="dashboard-note">{reloadNote}</div> : null}
        </section>
      )}

      <section className="dashboard-section">
        <div className="dash-title-row">
          <div>
            <h2 className="dash-title">System Overview</h2>
          </div>
          <div className="dash-subtitle">
            {isOnline ? "Live Jetson values" : "Waiting for fresh Jetson data"}
          </div>
        </div>

        <div className="filo-grid">
          <FiloCard
            label="RAM Usage"
            value={ramValue}
            sub="Memory load"
            status={isOnline ? "BAD" : "WARN"}
          />
          <FiloCard
            label="CPU Usage"
            value={cpuValue}
            sub="Processor load"
            status={isOnline ? "BAD" : "WARN"}
          />
          <FiloCard
            label="GPU Usage"
            value={gpuValue}
            sub="GPU load"
            status={isOnline ? "BAD" : "WARN"}
          />
          <FiloCard
            label="Uptime"
            value={uptimeValue}
            sub="Since boot"
            status={isOnline ? "BAD" : "WARN"}
          />
          <FiloCard
            label="ESP32 Connection"
            value={esp32Value}
            sub="Controller link"
            status={esp32Health}
          />
          <FiloCard
            label="Thermals"
            value={thermalsValue}
            sub="Overall thermal condition"
            status={thermalsHealth}
          />
        </div>
      </section>
    </div>
  );
}

function FiloCard({
  label,
  value,
  sub,
  status,
}: {
  label: string;
  value: string;
  sub: string;
  status?: HealthStatus;
}) {
  const resolvedStatus = status ?? "WARN";
  const badgeText = statusBadgeText(resolvedStatus);
  const badgeClass = statusBadgeClass(resolvedStatus);

  return (
    <div className="filo-item">
      <div className="filo-top">
        <div className="filo-label">{label}</div>
        <div className={`filo-status-badge ${badgeClass}`}>{badgeText}</div>
      </div>

      <div className="filo-value">{value}</div>
      <div className="filo-sub">{sub}</div>
    </div>
  );
}