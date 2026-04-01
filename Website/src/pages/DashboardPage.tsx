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
    };
  };
};

type AdminListResponse = {
  ok: boolean;
  count: number;
  items: DeviceRecord[];
  offline_after_seconds?: number;
  server_time?: number;
};

const API_BASE = "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";
const DEVICE_ID = "jetson-001";
const LS_TOKEN = "aura-auth-token";
const OFFLINE_AFTER_SECONDS = 10;

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
  if (sec == null) return "—";
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
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

export default function DashboardPage() {
  const [device, setDevice] = useState<DeviceRecord | null>(null);
  const [error, setError] = useState("");

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
        const found = data.items.find((item) => item.device_id === DEVICE_ID) ?? data.items[0] ?? null;

        if (alive) {
          setDevice(found);
          setError("");
        }
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard");
        }
      }
    }

    void load();
    const id = setInterval(() => {
      void load();
    }, 2000);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const updatedLabel = useMemo(() => {
    if (!device?.last_seen_at) return "";
    return formatUpdated(device.last_seen_at);
  }, [device?.last_seen_at]);

  const computedOnline =
    !!device?.last_seen_at &&
    Date.now() / 1000 - device.last_seen_at <= OFFLINE_AFTER_SECONDS;

  const status = computedOnline ? "ONLINE" : "OFFLINE";
  const s = device?.status;
  const extra = s?.extra;

  const motorsHealth: HealthStatus = "OK";
  const sensorsHealth: HealthStatus =
    healthFromBool(s?.camera_ready) === "BAD" ||
    healthFromBool(s?.mic_ready) === "BAD" ||
    healthFromBool(s?.speaker_ready) === "BAD"
      ? "BAD"
      : "OK";
  const thermalsHealth = thermalStatus(s?.temperature_c);

  return (
    <div className="dashboard-shell">
      <header className="dashboard-hero">
        <div className="dashboard-hero-copy">
          <p className="dashboard-eyebrow">Robot Overview</p>
          <h1>{device?.device_name || "AURA"}</h1>
          <p className="dashboard-subtitle">
            Status: <strong>{status}</strong>
            {updatedLabel ? <> • Updated {updatedLabel}</> : null}
          </p>
          {error ? <p className="dashboard-error">Dashboard load failed: {error}</p> : null}
        </div>

        <div className="dashboard-hero-art">
          <img src={robotImage} alt="AURA robot" />
        </div>
      </header>

      <section className="dashboard-section">
        <div className="section-heading">
          <h2>Filometrics</h2>
          <p>Live Jetson values</p>
        </div>

        <div className="filo-grid">
          <FiloCard label="Battery" value={fmt(s?.battery_percent, "%")} sub={fmt(s?.battery_voltage, " V", 2)} />
          <FiloCard label="Charging" value={s?.charging == null ? "—" : s.charging ? "Yes" : "No"} sub="Power state" />
          <FiloCard label="CPU" value={fmt(s?.cpu_percent, "%")} sub="Processor usage" />
          <FiloCard label="RAM" value={fmt(s?.ram_percent, "%")} sub="Memory usage" />
          <FiloCard label="Disk" value={fmt(s?.disk_percent, "%")} sub="Storage usage" />
          <FiloCard label="Temp" value={fmt(s?.temperature_c, " °C")} sub="Thermals" />
          <FiloCard label="Mode" value={s?.current_mode || "—"} sub={s?.current_task || "No active task"} />
          <FiloCard label="Uptime" value={formatUptime(extra?.uptime_seconds)} sub={extra?.hostname || "Unknown host"} />
        </div>
      </section>

      <section className="dashboard-section">
        <div className="section-heading">
          <h2>System Health</h2>
          <p>Quick status checks</p>
        </div>

        <div className="health-grid">
          <HealthCard label="Motors" status={motorsHealth} />
          <HealthCard label="Sensors" status={sensorsHealth} />
          <HealthCard label="Thermals" status={thermalsHealth} />
          <HealthCard label="Camera" status={healthFromBool(s?.camera_ready)} />
          <HealthCard label="Microphone" status={healthFromBool(s?.mic_ready)} />
          <HealthCard label="Speaker" status={healthFromBool(s?.speaker_ready)} />
          <HealthCard label="Ollama" status={healthFromBool(s?.ollama_ready)} />
          <HealthCard label="Vector DB" status={healthFromBool(s?.vector_db_ready)} />
        </div>
      </section>
    </div>
  );
}

function FiloCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <article className="filo-card">
      <p className="filo-label">{label}</p>
      <h3 className="filo-value">{value}</h3>
      <p className="filo-sub">{sub}</p>
    </article>
  );
}

function HealthCard({ label, status }: { label: string; status?: HealthStatus }) {
  const s = status ?? "WARN";

  return (
    <article className="health-card">
      <div className="health-card-top">
        <span className="health-dot" style={{ background: dotColor(s) }} />
        <p>{label}</p>
      </div>
      <h3>{s}</h3>
      <p>Overall condition</p>
    </article>
  );
}