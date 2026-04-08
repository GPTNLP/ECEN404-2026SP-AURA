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
  meta?: Record<string, unknown>;
};

type SessionMeta = {
  session_id: string;
  title?: string;
  owner_email?: string;
  owner_role?: string;
  device_id?: string;
  db_name?: string;
  message_count: number;
  created_ts?: number;
  updated_ts?: number;
  source?: string;
};

type Message = {
  role: "user" | "assistant" | "error";
  content: string;
  ts?: number;
};

type SessionDetail = SessionMeta & {
  history: Message[];
};

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined) ||
  (import.meta.env.VITE_CAMERA_API_BASE as string | undefined) ||
  "http://127.0.0.1:9000";

function fmtTime(ts?: number) {
  if (!ts) return "-";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

function normalizeHistory(history: any[]): Message[] {
  if (!Array.isArray(history)) return [];

  return history.map((msg) => ({
    role:
      msg.role === "user"
        ? "user"
        : msg.role === "error"
          ? "error"
          : "assistant",
    content: String(msg.content ?? ""),
    ts: typeof msg.ts === "number" ? msg.ts : undefined,
  }));
}

export default function ChatLogsPage() {
  const { token, user } = useAuth();
  const isAdmin = user?.role === "admin";

  const [tab, setTab] = useState<"sessions" | "events">("sessions");

  const [evLoading, setEvLoading] = useState(false);
  const [evError, setEvError] = useState<string | null>(null);
  const [items, setItems] = useState<LogItem[]>([]);
  const [matched, setMatched] = useState(0);
  const [q, setQ] = useState("");
  const [role, setRole] = useState("");
  const [event, setEvent] = useState("");
  const [limit, setLimit] = useState(200);
  const [offset, setOffset] = useState(0);

  const [sessLoading, setSessLoading] = useState(false);
  const [sessError, setSessError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionDetail | null>(null);
  const [sessDetailLoading, setSessDetailLoading] = useState(false);

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

    setEvLoading(true);
    setEvError(null);

    try {
      const res = await fetch(`${API_BASE}/logs/list?${queryString}`, {
        headers: { Authorization: `Bearer ${token}` },
        credentials: "include",
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

  const fetchSessions = async () => {
    if (!token) return;

    setSessLoading(true);
    setSessError(null);

    try {
      const res = await fetch(`${API_BASE}/logs/sessions/list`, {
        headers: { Authorization: `Bearer ${token}` },
        credentials: "include",
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

  const openSession = async (sessionId: string) => {
    if (!token) return;

    setSessDetailLoading(true);
    setSelectedSession(null);
    setSessError(null);

    try {
      const res = await fetch(`${API_BASE}/logs/sessions/${encodeURIComponent(sessionId)}`, {
        headers: { Authorization: `Bearer ${token}` },
        credentials: "include",
      });
      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      const session = (data.session ?? data) as SessionDetail;

      setSelectedSession({
        ...session,
        history: normalizeHistory(session.history),
      });
    } catch (e: any) {
      setSessError(e?.message || "Failed to load session");
    } finally {
      setSessDetailLoading(false);
    }
  };

  const openInSimulator = (sessionId: string) => {
    localStorage.setItem("aura_active_session_id", sessionId);
    window.location.href = `/simulator?session_id=${encodeURIComponent(sessionId)}`;
  };

  useEffect(() => {
    if (!canFetch) return;

    if (tab === "events") {
      void fetchLogs();
    } else {
      void fetchSessions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canFetch, tab, queryString]);

  if (!token || !isAdmin) {
    return (
      <div className="page-shell">
        <h1 className="page-title">Chat Logs</h1>
        <div className="card card-pad">{!token ? "Please login." : "Admin only."}</div>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          marginBottom: 16,
        }}
      >
        <h1 className="page-title" style={{ margin: 0 }}>
          Chat Logs
        </h1>

        <div style={{ display: "flex", gap: 10 }}>
          <button
            className={tab === "sessions" ? "btn btn-primary" : "btn"}
            onClick={() => setTab("sessions")}
          >
            Sessions
          </button>
          <button
            className={tab === "events" ? "btn btn-primary" : "btn"}
            onClick={() => setTab("events")}
          >
            Event Log
          </button>
        </div>
      </div>

      {tab === "sessions" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "380px minmax(0, 1fr)",
            gap: 18,
            alignItems: "start",
          }}
        >
          <div className="card card-pad">
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 10,
                marginBottom: 12,
              }}
            >
              <div>
                <div style={{ fontWeight: 800, fontSize: 18 }}>Sessions</div>
                <div style={{ fontSize: 13, opacity: 0.75 }}>
                  {sessLoading ? "Loading..." : `${sessions.length} saved`}
                </div>
              </div>

              <button className="btn" onClick={() => void fetchSessions()}>
                Refresh
              </button>
            </div>

            {sessError && (
              <div className="card" style={{ padding: 10, marginBottom: 10, color: "#b42318" }}>
                {sessError}
              </div>
            )}

            {sessions.length === 0 && !sessLoading && <div>No sessions found.</div>}

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {sessions.map((s) => {
                const selected = selectedSession?.session_id === s.session_id;
                return (
                  <button
                    key={s.session_id}
                    type="button"
                    onClick={() => void openSession(s.session_id)}
                    className="card card-pad"
                    style={{
                      textAlign: "left",
                      cursor: "pointer",
                      border: selected
                        ? "1px solid var(--accent, #7c3aed)"
                        : "1px solid var(--card-border, #ddd)",
                      background: selected
                        ? "var(--accent-soft, rgba(124,58,237,.10))"
                        : "var(--card-bg, white)",
                    }}
                  >
                    <div style={{ fontWeight: 800, marginBottom: 4 }}>
                      {s.title || s.session_id}
                    </div>
                    <div style={{ fontSize: 13, opacity: 0.78 }}>
                      {s.owner_email || "unknown owner"}
                    </div>
                    <div style={{ fontSize: 12, opacity: 0.72, marginTop: 6 }}>
                      {s.message_count} messages
                      {s.db_name ? ` • ${s.db_name}` : ""}
                      {s.source ? ` • ${s.source}` : ""}
                    </div>
                    <div style={{ fontSize: 12, opacity: 0.65, marginTop: 4 }}>
                      {fmtTime(s.updated_ts)}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="card card-pad" style={{ minHeight: 560 }}>
            {!selectedSession && !sessDetailLoading && (
              <div style={{ opacity: 0.8 }}>Select a session to view the conversation.</div>
            )}

            {sessDetailLoading && <div>Loading...</div>}

            {selectedSession && !sessDetailLoading && (
              <>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 12,
                    alignItems: "flex-start",
                    marginBottom: 16,
                    flexWrap: "wrap",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 22, fontWeight: 900 }}>
                      {selectedSession.title || selectedSession.session_id}
                    </div>
                    <div style={{ fontSize: 13, opacity: 0.78, marginTop: 4 }}>
                      {selectedSession.owner_email || "unknown owner"}
                      {selectedSession.owner_role ? ` • ${selectedSession.owner_role}` : ""}
                      {selectedSession.db_name ? ` • ${selectedSession.db_name}` : ""}
                      {selectedSession.device_id ? ` • ${selectedSession.device_id}` : ""}
                    </div>
                    <div style={{ fontSize: 12, opacity: 0.68, marginTop: 4 }}>
                      Created: {fmtTime(selectedSession.created_ts)} • Updated:{" "}
                      {fmtTime(selectedSession.updated_ts)}
                    </div>
                  </div>

                  <button
                    className="btn btn-primary"
                    onClick={() => openInSimulator(selectedSession.session_id)}
                  >
                    Open in Simulator
                  </button>
                </div>

                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 12,
                    maxHeight: 470,
                    overflowY: "auto",
                    paddingRight: 4,
                  }}
                >
                  {selectedSession.history.map((msg, i) => {
                    const isUser = msg.role === "user";
                    const isError = msg.role === "error";

                    return (
                      <div
                        key={`${msg.role}-${i}-${msg.ts || 0}`}
                        style={{
                          display: "flex",
                          justifyContent: isUser ? "flex-end" : "flex-start",
                        }}
                      >
                        <div
                          style={{
                            maxWidth: "78%",
                            padding: "12px 14px",
                            borderRadius: 16,
                            border: "1px solid var(--card-border, #ddd)",
                            background: isUser
                              ? "var(--accent, #7c3aed)"
                              : isError
                                ? "rgba(220, 38, 38, 0.08)"
                                : "var(--card-bg, white)",
                            color: isUser ? "white" : "inherit",
                            whiteSpace: "pre-wrap",
                            lineHeight: 1.45,
                            textAlign: "left",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 12,
                              fontWeight: 800,
                              marginBottom: 6,
                              opacity: 0.8,
                              textTransform: "capitalize",
                            }}
                          >
                            {msg.role === "assistant" ? "AURA" : msg.role}{" "}
                            {msg.ts ? `• ${fmtTime(msg.ts)}` : ""}
                          </div>
                          <div>{msg.content}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {tab === "events" && (
        <>
          <div
            className="card card-pad"
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 10,
              alignItems: "center",
              marginBottom: 14,
            }}
          >
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
              <option value="ta">TA</option>
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
              <option value="chat_error">chat_error</option>
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

            <button className="btn btn-primary" onClick={() => void fetchLogs()}>
              Search
            </button>

            <button
              className="btn"
              onClick={() => {
                setQ("");
                setRole("");
                setEvent("");
                setOffset(0);
              }}
              disabled={evLoading}
            >
              Clear
            </button>
          </div>

          <div style={{ marginBottom: 10, fontSize: 13, opacity: 0.78 }}>
            Matched: {matched}
          </div>

          {evError && (
            <div className="card card-pad" style={{ color: "#b42318", marginBottom: 10 }}>
              {evError}
            </div>
          )}

          <div className="card card-pad" style={{ overflowX: "auto" }}>
            {items.length === 0 && !evLoading ? (
              <div>No logs found.</div>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "10px 8px" }}>Time</th>
                    <th style={{ textAlign: "left", padding: "10px 8px" }}>User</th>
                    <th style={{ textAlign: "left", padding: "10px 8px" }}>Role</th>
                    <th style={{ textAlign: "left", padding: "10px 8px" }}>Event</th>
                    <th style={{ textAlign: "left", padding: "10px 8px" }}>Prompt</th>
                    <th style={{ textAlign: "left", padding: "10px 8px" }}>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((it, idx) => (
                    <tr key={`${it.ts}-${idx}`} style={{ borderTop: "1px solid var(--card-border, #eee)" }}>
                      <td style={{ padding: "10px 8px", whiteSpace: "nowrap" }}>{fmtTime(it.ts)}</td>
                      <td style={{ padding: "10px 8px" }}>{it.user_email || "-"}</td>
                      <td style={{ padding: "10px 8px" }}>{it.user_role || "-"}</td>
                      <td style={{ padding: "10px 8px" }}>{it.event || "-"}</td>
                      <td style={{ padding: "10px 8px", maxWidth: 540 }}>
                        {it.prompt ? it.prompt.slice(0, 500) : "-"}
                        {it.prompt && it.prompt.length > 500 ? "…" : ""}
                      </td>
                      <td style={{ padding: "10px 8px" }}>
                        {typeof it.latency_ms === "number" ? `${it.latency_ms}ms` : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 10,
                marginTop: 14,
                flexWrap: "wrap",
              }}
            >
              <div style={{ fontSize: 13, opacity: 0.75 }}>
                Showing {items.length} / {matched} (offset {offset})
              </div>

              <div style={{ display: "flex", gap: 8 }}>
                <button
                  className="btn"
                  onClick={() => setOffset((v) => Math.max(0, v - limit))}
                  disabled={evLoading || offset === 0}
                >
                  ← Prev
                </button>
                <button
                  className="btn"
                  onClick={() => setOffset((v) => v + limit)}
                  disabled={evLoading || offset + limit >= matched}
                >
                  Next →
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}