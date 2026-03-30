import { useEffect, useMemo, useState } from "react";
import { useAuth } from "../services/authService";
import "../styles/page-ui.css";

type LogItem = {
  ts: number;
  event?: string;
  user_email?: string;
  user_role?: string;
  prompt?: string;
  response_preview?: string;
  model?: string;
  latency_ms?: number;
  meta?: Record<string, any>;
};

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined) ||
  (import.meta.env.VITE_CAMERA_API_BASE as string | undefined) ||
  "http://127.0.0.1:9000";

function fmtTime(ts: number) {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

export default function ChatLogsPage() {
  const { token, user } = useAuth();
  const isAdmin = user?.role === "admin";

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [items, setItems] = useState<LogItem[]>([]);
  const [matched, setMatched] = useState<number>(0);

  const [q, setQ] = useState("");
  const [role, setRole] = useState("");
  const [event, setEvent] = useState("");

  const [limit, setLimit] = useState(200);
  const [offset, setOffset] = useState(0);

  const canFetch = !!token && isAdmin;

  const queryString = useMemo(() => {
    const p = new URLSearchParams();
    p.set("limit", String(limit));
    p.set("offset", String(offset));
    if (q.trim()) p.set("q", q.trim());
    if (role) p.set("role", role);
    if (event) p.set("event", event);
    return p.toString();
  }, [limit, offset, q, role, event]);

  const fetchLogs = async () => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/logs/list?${queryString}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      setItems((data.items || []) as LogItem[]);
      setMatched(Number(data.total_matched || 0));
    } catch (e: any) {
      setError(e?.message || "Failed to load logs");
      setItems([]);
      setMatched(0);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!canFetch) return;
    fetchLogs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, queryString]);

  if (!token) {
    return (
      <div className="page-wrap">
        <h2 className="page-title">Chat Logs</h2>
        <div className="muted">Please login.</div>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="page-wrap">
        <h2 className="page-title">Chat Logs</h2>
        <div className="muted">Admin only.</div>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div className="page-wrap">
        <div className="page-header">
          <h2 className="page-title">Chat Logs</h2>

          <button onClick={fetchLogs} disabled={loading} className="btn">
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        <div className="card card-pad" style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <input
              value={q}
              onChange={(e) => {
                setOffset(0);
                setQ(e.target.value);
              }}
              placeholder="Search (email / prompt / meta...)"
              className="input"
              style={{ flex: "1 1 320px" }}
            />

            <select
              value={role}
              onChange={(e) => {
                setOffset(0);
                setRole(e.target.value);
              }}
              className="select"
            >
              <option value="">All Roles</option>
              <option value="admin">Admin</option>
              <option value="student">Student</option>
            </select>

            <select
              value={event}
              onChange={(e) => {
                setOffset(0);
                setEvent(e.target.value);
              }}
              className="select"
            >
              <option value="">All Events</option>
              <option value="chat">chat</option>
              <option value="bot">bot</option>
              <option value="upload">upload</option>
              <option value="login">login</option>
            </select>

            <select
              value={String(limit)}
              onChange={(e) => {
                setOffset(0);
                setLimit(parseInt(e.target.value, 10));
              }}
              className="select"
            >
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
              <option value="500">500</option>
            </select>

            <button onClick={fetchLogs} disabled={loading} className="btn btn-primary">
              Search
            </button>

            <button
              onClick={() => {
                setQ("");
                setRole("");
                setEvent("");
                setOffset(0);
              }}
              disabled={loading}
              className="btn"
            >
              Clear
            </button>
          </div>

          <div style={{ marginTop: 10, fontSize: 12 }} className="muted">
            Matched: <b>{matched}</b>
          </div>

          {error && (
            <div style={{ marginTop: 10, color: "var(--status-bad)", fontSize: 13, whiteSpace: "pre-wrap" }}>
              {error}
            </div>
          )}
        </div>

        <div className="card" style={{ overflow: "hidden" }}>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th>Time</th>
                  <th>User</th>
                  <th>Role</th>
                  <th>Event</th>
                  <th style={{ minWidth: 420 }}>Prompt</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {items.length === 0 && !loading && (
                  <tr>
                    <td colSpan={6} style={{ padding: 14 }} className="muted">
                      No logs found.
                    </td>
                  </tr>
                )}

                {items.map((it, idx) => (
                  <tr key={idx}>
                    <td style={{ whiteSpace: "nowrap" }} className="muted">
                      {fmtTime(it.ts)}
                    </td>
                    <td style={{ whiteSpace: "nowrap" }}>{it.user_email || "-"}</td>
                    <td style={{ whiteSpace: "nowrap" }}>{it.user_role || "-"}</td>
                    <td style={{ whiteSpace: "nowrap" }}>{it.event || "-"}</td>
                    <td style={{ minWidth: 420 }}>
                      <div style={{ whiteSpace: "pre-wrap" }}>
                        {it.prompt ? it.prompt.slice(0, 500) : "-"}
                        {it.prompt && it.prompt.length > 500 ? "…" : ""}
                      </div>
                    </td>
                    <td style={{ whiteSpace: "nowrap" }} className="muted">
                      {typeof it.latency_ms === "number" ? `${it.latency_ms}ms` : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: 12,
              borderTop: "1px solid var(--card-border)",
            }}
          >
            <div className="muted" style={{ fontSize: 12 }}>
              Showing {items.length} / {matched} (offset {offset})
            </div>

            <div style={{ display: "flex", gap: 10 }}>
              <button
                onClick={() => setOffset((v) => Math.max(0, v - limit))}
                disabled={loading || offset === 0}
                className="btn"
              >
                ← Prev
              </button>

              <button
                onClick={() => setOffset((v) => v + limit)}
                disabled={loading || offset + limit >= matched}
                className="btn"
              >
                Next →
              </button>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 10, fontSize: 12 }} className="muted">
          Endpoint: <span className="mono">{API_BASE}/logs/list</span>
        </div>
      </div>
    </div>
  );
}