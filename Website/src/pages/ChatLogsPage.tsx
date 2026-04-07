import { useEffect, useMemo, useState } from "react";
import { useAuth } from "../services/authService";
import "../styles/page-ui.css";

/* ─── Types ─────────────────────────────────────────────────── */
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

type SessionMeta = {
  session_id: string;
  device_id?: string;
  message_count: number;
  updated_ts?: number;
};

type Message = {
  role: string;
  content: string;
  ts?: number;
};

type SessionDetail = {
  session_id: string;
  device_id?: string;
  history: Message[];
  updated_ts?: number;
};

/* ─── Helpers ────────────────────────────────────────────────── */
const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined) ||
  (import.meta.env.VITE_CAMERA_API_BASE as string | undefined) ||
  "http://127.0.0.1:9000";

const JETSON_BASE =
  (import.meta.env.VITE_JETSON_API_BASE as string | undefined) ||
  "http://127.0.0.1:8000";

function fmtTime(ts?: number) {
  if (!ts) return "-";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

/* ─── Component ─────────────────────────────────────────────── */
export default function ChatLogsPage() {
  const { token, user } = useAuth();
  const isAdmin = user?.role === "admin";

  const [tab, setTab] = useState<"events" | "sessions">("sessions");

  /* ── Event log state ──────────────────────────────────────── */
  const [evLoading, setEvLoading] = useState(false);
  const [evError, setEvError] = useState<string | null>(null);
  const [items, setItems] = useState<LogItem[]>([]);
  const [matched, setMatched] = useState(0);
  const [q, setQ] = useState("");
  const [role, setRole] = useState("");
  const [event, setEvent] = useState("");
  const [limit, setLimit] = useState(200);
  const [offset, setOffset] = useState(0);

  /* ── Session state ────────────────────────────────────────── */
  const [sessLoading, setSessLoading] = useState(false);
  const [sessError, setSessError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionDetail | null>(null);
  const [sessDetailLoading, setSessDetailLoading] = useState(false);
  const [loadToJetsonStatus, setLoadToJetsonStatus] = useState<string>("");

  const canFetch = !!token && isAdmin;

  /* ── Event log fetch ────────────────────────────────────────── */
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
    setEvLoading(true);
    setEvError(null);
    try {
      const res = await fetch(`${API_BASE}/logs/list?${queryString}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setItems((data.items || []) as LogItem[]);
      setMatched(Number(data.total_matched || 0));
    } catch (e: any) {
      setEvError(e?.message || "Failed to load logs");
      setItems([]);
    } finally {
      setEvLoading(false);
    }
  };

  /* ── Session list fetch ─────────────────────────────────────── */
  const fetchSessions = async () => {
    if (!token) return;
    setSessLoading(true);
    setSessError(null);
    try {
      const res = await fetch(`${API_BASE}/logs/sessions/list`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSessions((data.sessions || []) as SessionMeta[]);
    } catch (e: any) {
      setSessError(e?.message || "Failed to load sessions");
      setSessions([]);
    } finally {
      setSessLoading(false);
    }
  };

  /* ── Session detail fetch ───────────────────────────────────── */
  const openSession = async (session_id: string) => {
    if (!token) return;
    setSessDetailLoading(true);
    setSelectedSession(null);
    setLoadToJetsonStatus("");
    try {
      const res = await fetch(`${API_BASE}/logs/sessions/${encodeURIComponent(session_id)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSelectedSession(data as SessionDetail);
    } catch (e: any) {
      setSessError(e?.message || "Failed to load session");
    } finally {
      setSessDetailLoading(false);
    }
  };

  /* ── Load session → Jetson ──────────────────────────────────── */
  const loadSessionToJetson = async (session_id: string) => {
    setLoadToJetsonStatus("Loading...");
    try {
      const res = await fetch(
        `${JETSON_BASE}/sessions/load/${encodeURIComponent(session_id)}`,
        { method: "POST" }
      );
      if (!res.ok) throw new Error(await res.text());
      setLoadToJetsonStatus("Loaded to Jetson");
    } catch (e: any) {
      setLoadToJetsonStatus(`Failed: ${e?.message || String(e)}`);
    }
  };

  useEffect(() => {
    if (!canFetch) return;
    if (tab === "events") fetchLogs();
    else fetchSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, tab, queryString]);

  if (!token || !isAdmin) {
    return (
      <div className="page-wrap">
        <h2 className="page-title">Chat Logs</h2>
        <div className="muted">{!token ? "Please login." : "Admin only."}</div>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div className="page-wrap">
        <div className="page-header">
          <h2 className="page-title">Chat Logs</h2>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              className={`btn ${tab === "sessions" ? "btn-primary" : ""}`}
              onClick={() => setTab("sessions")}
            >
              Sessions
            </button>
            <button
              className={`btn ${tab === "events" ? "btn-primary" : ""}`}
              onClick={() => setTab("events")}
            >
              Event Log
            </button>
          </div>
        </div>

        {/* ── Sessions Tab ─────────────────────────────────────── */}
        {tab === "sessions" && (
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
            {/* Session list */}
            <div className="card card-pad" style={{ flex: "0 0 300px", minWidth: 240 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <span style={{ fontWeight: 600 }}>Sessions</span>
                <button onClick={fetchSessions} disabled={sessLoading} className="btn btn-sm">
                  {sessLoading ? "..." : "Refresh"}
                </button>
              </div>

              {sessError && (
                <div style={{ color: "var(--status-bad)", fontSize: 12, marginBottom: 8 }}>
                  {sessError}
                </div>
              )}

              {sessions.length === 0 && !sessLoading && (
                <div className="muted" style={{ fontSize: 13 }}>
                  No sessions found.
                </div>
              )}

              {sessions.map((s) => (
                <div
                  key={s.session_id}
                  onClick={() => openSession(s.session_id)}
                  className={`card card-pad ${selectedSession?.session_id === s.session_id ? "selected" : ""}`}
                  style={{
                    marginBottom: 6,
                    cursor: "pointer",
                    background:
                      selectedSession?.session_id === s.session_id
                        ? "var(--accent-soft, rgba(99,102,241,.12))"
                        : undefined,
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{s.session_id}</div>
                  <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>
                    {s.message_count} messages · {s.device_id || "unknown device"}
                  </div>
                  <div className="muted" style={{ fontSize: 11 }}>{fmtTime(s.updated_ts)}</div>
                </div>
              ))}
            </div>

            {/* Session detail */}
            <div className="card card-pad" style={{ flex: 1 }}>
              {!selectedSession && !sessDetailLoading && (
                <div className="muted" style={{ padding: 20 }}>
                  Select a session to view the conversation.
                </div>
              )}

              {sessDetailLoading && (
                <div className="muted" style={{ padding: 20 }}>Loading...</div>
              )}

              {selectedSession && !sessDetailLoading && (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                    <div>
                      <span style={{ fontWeight: 600 }}>{selectedSession.session_id}</span>
                      <span className="muted" style={{ fontSize: 12, marginLeft: 10 }}>
                        {selectedSession.device_id} · {fmtTime(selectedSession.updated_ts)}
                      </span>
                    </div>
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => loadSessionToJetson(selectedSession.session_id)}
                      title="Load this session into the active Jetson instance"
                    >
                      Load to Jetson
                    </button>
                  </div>

                  {loadToJetsonStatus && (
                    <div
                      style={{
                        fontSize: 12,
                        marginBottom: 10,
                        color: loadToJetsonStatus.startsWith("Failed")
                          ? "var(--status-bad)"
                          : "var(--status-good)",
                      }}
                    >
                      {loadToJetsonStatus}
                    </div>
                  )}

                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 10,
                      maxHeight: 600,
                      overflowY: "auto",
                    }}
                  >
                    {selectedSession.history.map((msg, i) => (
                      <div
                        key={i}
                        style={{
                          padding: "10px 14px",
                          borderRadius: 8,
                          background:
                            msg.role === "user"
                              ? "var(--card-bg, #1e1e2e)"
                              : "var(--accent-soft, rgba(99,102,241,.10))",
                          alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
                          maxWidth: "80%",
                          border: "1px solid var(--card-border)",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            fontWeight: 600,
                            marginBottom: 4,
                            textTransform: "uppercase",
                            color: "var(--muted)",
                          }}
                        >
                          {msg.role}
                          {msg.ts && (
                            <span style={{ marginLeft: 8, fontWeight: 400 }}>
                              {fmtTime(msg.ts)}
                            </span>
                          )}
                        </div>
                        <div style={{ whiteSpace: "pre-wrap", fontSize: 14 }}>{msg.content}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* ── Event Log Tab ─────────────────────────────────────── */}
        {tab === "events" && (
          <>
            <div className="card card-pad" style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                <input
                  value={q}
                  onChange={(e) => { setOffset(0); setQ(e.target.value); }}
                  placeholder="Search (email / prompt / meta...)"
                  className="input"
                  style={{ flex: "1 1 320px" }}
                />
                <select value={role} onChange={(e) => { setOffset(0); setRole(e.target.value); }} className="select">
                  <option value="">All Roles</option>
                  <option value="admin">Admin</option>
                  <option value="student">Student</option>
                </select>
                <select value={event} onChange={(e) => { setOffset(0); setEvent(e.target.value); }} className="select">
                  <option value="">All Events</option>
                  <option value="chat">chat</option>
                  <option value="bot">bot</option>
                  <option value="upload">upload</option>
                  <option value="login">login</option>
                </select>
                <select
                  value={String(limit)}
                  onChange={(e) => { setOffset(0); setLimit(parseInt(e.target.value, 10)); }}
                  className="select"
                >
                  <option value="50">50</option>
                  <option value="100">100</option>
                  <option value="200">200</option>
                  <option value="500">500</option>
                </select>
                <button onClick={fetchLogs} disabled={evLoading} className="btn btn-primary">Search</button>
                <button onClick={() => { setQ(""); setRole(""); setEvent(""); setOffset(0); }} disabled={evLoading} className="btn">Clear</button>
              </div>
              <div style={{ marginTop: 10, fontSize: 12 }} className="muted">Matched: <b>{matched}</b></div>
              {evError && (
                <div style={{ marginTop: 10, color: "var(--status-bad)", fontSize: 13, whiteSpace: "pre-wrap" }}>
                  {evError}
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
                    {items.length === 0 && !evLoading && (
                      <tr><td colSpan={6} style={{ padding: 14 }} className="muted">No logs found.</td></tr>
                    )}
                    {items.map((it, idx) => (
                      <tr key={idx}>
                        <td style={{ whiteSpace: "nowrap" }} className="muted">{fmtTime(it.ts)}</td>
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
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: 12, borderTop: "1px solid var(--card-border)" }}>
                <div className="muted" style={{ fontSize: 12 }}>
                  Showing {items.length} / {matched} (offset {offset})
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button onClick={() => setOffset((v) => Math.max(0, v - limit))} disabled={evLoading || offset === 0} className="btn">← Prev</button>
                  <button onClick={() => setOffset((v) => v + limit)} disabled={evLoading || offset + limit >= matched} className="btn">Next →</button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
