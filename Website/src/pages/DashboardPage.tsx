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
};

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined)?.trim() ||
  "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";

const DEVICE_ID =
  (import.meta.env.VITE_DEVICE_ID as string | undefined)?.trim() ||
  "jetson-001";

const LS_TOKEN = "aura-auth-token";

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

function statusTextFromBool(value?: boolean | null) {
  if (value === true) return "Ready";
  if (value === false) return "Not Ready";
  return "Unknown";
}

function healthFromReadyText(value: string): HealthStatus {
  if (value === "Ready") return "OK";
  if (value === "Not Ready") return "BAD";
  return "WARN";
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
        const found =
          data.items.find((item) => item.device_id === DEVICE_ID) ??
          data.items[0] ??
          null;

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

    load();
    const id = setInterval(load, 2000);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const updatedLabel = useMemo(() => {
    if (!device?.last_seen_at) return "";
    return formatUpdated(device.last_seen_at);
  }, [device?.last_seen_at]);

  const status = device?.online ? "ONLINE" : "OFFLINE";
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

  const aiReadyText = statusTextFromBool(s?.ollama_ready);
  const vectorReadyText = statusTextFromBool(s?.vector_db_ready);

  return (
    <div className="dashboard-page">
      <div className="aura-header">
        <div className="aura-panel">
          <img src={robotImage} alt="AURA" className="aura-img" />
          <div className="aura-text">
            <h1 className="aura-title">{device?.device_name || "AURA"}</h1>
            <div className="aura-sub">
              Status: <b>{status}</b>
              {updatedLabel ? <> • Updated {updatedLabel}</> : null}
            </div>
          </div>
        </div>
      </div>

      {error ? <div className="dash-error">Dashboard load failed: {error}</div> : null}

      <section className="dashboard-section">
        <div className="dash-title-row">
          <h2 className="dash-title">Filometrics</h2>
          <div className="dash-subtitle">Live Jetson values</div>
        </div>

        <div className="filo-grid">
          <FiloCard
            label="RAM Usage"
            value={fmt(s?.ram_percent, "%", 1)}
            sub="Memory load"
          />
          <FiloCard
            label="CPU Usage"
            value={fmt(s?.cpu_percent, "%", 1)}
            sub="Processor load"
          />
          <FiloCard
            label="GPU Usage"
            value={fmt(extra?.gpu_percent, "%", 1)}
            sub="GPU load"
          />
          <FiloCard
            label="Uptime"
            value={formatUptime(extra?.uptime_seconds)}
            sub="Since boot"
          />
          <FiloCard
            label="Active DB"
            value={extra?.db_name || "None"}
            sub="Loaded on Jetson"
          />
          <FiloCard
            label="AI (LLM)"
            value={aiReadyText}
            sub="Ollama status"
            status={healthFromReadyText(aiReadyText)}
          />
          <FiloCard
            label="Vector DB"
            value={vectorReadyText}
            sub="RAG system"
            status={healthFromReadyText(vectorReadyText)}
          />
        </div>
      </section>

      <section className="dashboard-section">
        <div className="dash-title-row">
          <h2 className="dash-title">System Health</h2>
          <div className="dash-subtitle">Quick status checks</div>
        </div>

        <div className="filo-grid">
          <HealthCard label="Motors" status={motorsHealth} />
          <HealthCard label="Sensors" status={sensorsHealth} />
          <HealthCard label="Thermals" status={thermalsHealth} />
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
  const dot =
    status != null ? dotColor(status) : "var(--accent)";

  return (
    <div className="filo-item">
      <div className="filo-top">
        <div className="filo-label">{label}</div>
        <div className="filo-dot" style={{ background: dot }} />
      </div>
      <div className="filo-value">{value}</div>
      <div className="filo-sub">{sub}</div>
    </div>
  );
}

function HealthCard({ label, status }: { label: string; status?: HealthStatus }) {
  const s = status ?? "WARN";
  return (
    <div className="filo-item">
      <div className="filo-top">
        <div className="filo-label">{label}</div>
        <div className="filo-dot" style={{ background: dotColor(s) }} />
      </div>
      <div className="filo-value">{s}</div>
      <div className="filo-sub">Overall condition</div>
    </div>
  );
}