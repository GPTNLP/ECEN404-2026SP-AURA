import { useEffect, useMemo, useState } from "react";
import "../styles/dashboard.css";
import robotImage from "../assets/robot.png";

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
  (import.meta.env.VITE_DEVICE_ID as string | undefined)?.trim() ||
  "jetson-001";

const LS_TOKEN = "aura-auth-token";
const STALE_AFTER_SECONDS = 15;

function dotColor(status: HealthStatus) {
  if (status === "OK") return "var(--status-good)";
  if (status === "WARN") return "var(--status-warn)";
  return "var(--status-bad)";
}

function healthFromBool(value?: boolean | null): HealthStatus {
  if (value === true) return "OK";
  if (value === false) return "BAD";
  return "WARN";
}

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

function statusTextFromBool(value?: boolean | null) {
  if (value === true) return "Ready";
  if (value === false) return "Not Ready";
  return "Unknown";
}

function staleHealth(): HealthStatus {
  return "WARN";
}

export default function DashboardPage() {
  const [device, setDevice] = useState<DeviceRecord | null>(null);
  const [error, setError] = useState("");
  const [nowMs, setNowMs] = useState(Date.now());

  useEffect(() => {
    let alive = true;

    async function load() {
      const token = localStorage.getItem(LS_TOKEN);

      try {
        const res = await fetch(`${API_BASE}/device/admin/list`, {
          method: "GET",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
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
    const candidates = [
      device?.last_seen_at,
      s?.updated_at,
    ].filter((v): v is number => typeof v === "number" && v > 0);

    if (!candidates.length) return 0;
    return Math.max(...candidates);
  }, [device?.last_seen_at, s?.updated_at]);

  const ageSeconds = freshestTimestampSeconds
    ? Math.max(0, Math.floor(nowMs / 1000 - freshestTimestampSeconds))
    : Number.POSITIVE_INFINITY;

  const isFresh = freshestTimestampSeconds > 0 && ageSeconds <= STALE_AFTER_SECONDS;
  const isOnline = Boolean(device?.online) && isFresh;

  const updatedLabel = useMemo(() => {
    if (!isFresh || !device?.last_seen_at) return "";
    return formatUpdated(device.last_seen_at);
  }, [device?.last_seen_at, isFresh]);

  const status = isOnline ? "ONLINE" : "OFFLINE";

  const ramValue = isOnline ? fmt(s?.ram_percent, "%", 1) : "Unknown";
  const cpuValue = isOnline ? fmt(s?.cpu_percent, "%", 1) : "Unknown";
  const gpuValue = isOnline ? fmt(extra?.gpu_percent, "%", 1) : "Unknown";
  const uptimeValue = isOnline ? formatUptime(extra?.uptime_seconds) : "Unknown";
  const dbValue = isOnline ? extra?.db_name || "None" : "Unknown";

  const cameraValue = isOnline ? statusTextFromBool(s?.camera_ready) : "Unknown";
  const micValue = isOnline ? statusTextFromBool(s?.mic_ready) : "Unknown";
  const speakerValue = isOnline ? statusTextFromBool(s?.speaker_ready) : "Unknown";

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

  const thermalsValue = isOnline ? (s?.temperature_c != null ? fmt(s.temperature_c, "°C", 1) : "Unknown") : "Unknown";
  const thermalsHealth = isOnline ? thermalStatus(s?.temperature_c) : staleHealth();

  return (
    <div className="dashboard-page">
      <div className="aura-header">
        <div className="aura-panel">
          <div
            className="aura-img-wrap"
            style={{
              width: 88,
              height: 88,
              borderRadius: 20,
              overflow: "hidden",
              flexShrink: 0,
              background: "rgba(0,0,0,0.06)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <img
              src={robotImage}
              alt="AURA"
              className="aura-img"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                objectPosition: "center top",
                display: "block",
              }}
            />
          </div>

          <div className="aura-text">
            <h1 className="aura-title">{device?.device_name || "AURA"}</h1>
            <div className="aura-sub">
              Status: <b>{status}</b>
              {isOnline && updatedLabel ? <> • Updated {updatedLabel}</> : null}
              {!isOnline && freshestTimestampSeconds > 0 ? <> • No fresh Jetson data</> : null}
            </div>
          </div>
        </div>
      </div>

      {error ? <div className="dash-error">Dashboard load failed: {error}</div> : null}

      <section className="dashboard-section">
        <div className="dash-title-row">
          <h2 className="dash-title">System Overview</h2>
          <div className="dash-subtitle">
            {isOnline ? "Live Jetson values" : "Waiting for fresh Jetson data"}
          </div>
        </div>

        <div className="filo-grid">
          <FiloCard
            label="RAM Usage"
            value={ramValue}
            sub={isOnline ? "Memory load" : "No recent data"}
            status={isOnline ? "BAD" : "WARN"}
            autoStatus={false}
          />

          <FiloCard
            label="CPU Usage"
            value={cpuValue}
            sub={isOnline ? "Processor load" : "No recent data"}
            status={isOnline ? "BAD" : "WARN"}
            autoStatus={false}
          />

          <FiloCard
            label="GPU Usage"
            value={gpuValue}
            sub={isOnline ? "GPU load" : "No recent data"}
            status={isOnline ? "BAD" : "WARN"}
            autoStatus={false}
          />

          <FiloCard
            label="Uptime"
            value={uptimeValue}
            sub={isOnline ? "Since boot" : "No recent data"}
            status={isOnline ? "BAD" : "WARN"}
            autoStatus={false}
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
  autoStatus = true,
}: {
  label: string;
  value: string;
  sub: string;
  status?: HealthStatus;
  autoStatus?: boolean;
}) {
  let resolvedStatus: HealthStatus;

  if (!autoStatus) {
    resolvedStatus = status ?? "WARN";
  } else {
    resolvedStatus = status ?? "WARN";
  }

  return (
    <div className="filo-item">
      <div className="filo-top">
        <div className="filo-label">{label}</div>
        <div className="filo-dot" style={{ background: dotColor(resolvedStatus) }} />
      </div>
      <div className="filo-value">{value}</div>
      <div className="filo-sub">{sub}</div>
    </div>
  );
}