import { useEffect, useMemo, useState } from "react";
import { useAuth } from "../services/authService";
import "../styles/chatlogs.css";

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

function fmtTimeShort(ts?: number) {
  if (!ts) return "";
  try {
    return new Date(ts * 1000).toLocaleString([], {
      month: "numeric",
      day: "numeric",
      year: "2-digit",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return String(ts);
  }
}

function normalizeHistory(history: unknown[]): Message[] {
  if (!Array.isArray(history)) return [];
  return history.map((msg: any) => ({
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

function initialsFromTitle(title?: string, fallback = "S") {
  const raw = (title || "").trim();
  if (!raw) return fallback;
  const parts = raw.split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || "").join("") || fallback;
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
  const [sessionSearch, setSessionSearch] = useState("");
  const [deletingSession, setDeletingSession] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);

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

  const filteredSessions = useMemo(() => {
    const needle = sessionSearch.trim().toLowerCase();
    if (!needle) return sessions;

    return sessions.filter((s) => {
      const haystack = [
        s.title,
        s.session_id,
        s.owner_email,
        s.owner_role,
        s.db_name,
        s.device_id,
        s.source,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return haystack.includes(needle);
    });
  }, [sessions, sessionSearch]);

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
      const nextSessions = (data.sessions || []) as SessionMeta[];
      setSessions(nextSessions);

      if (!selectedSession && nextSessions.length > 0) {
        void openSession(nextSessions[0].session_id);
      }
    } catch (e: any) {
      setSessError(e?.message || "Failed to load sessions");
      setSessions([]);
    } finally {
      setSessLoading(false);
    }
  };

  const openSession = async (sessionId: string) => {
    if (!token) return;
    setDeleteConfirm(false);
    setSessDetailLoading(true);
    setSessError(null);

    try {
      const res = await fetch(
        `${API_BASE}/logs/sessions/${encodeURIComponent(sessionId)}`,
        {
          headers: { Authorization: `Bearer ${token}` },
          credentials: "include",
        }
      );

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      const session = (data.session ?? data) as SessionDetail;

      setSelectedSession({
        ...session,
        history: normalizeHistory(session.history || []),
      });
    } catch (e: any) {
      setSessError(e?.message || "Failed to load session");
      setSelectedSession(null);
    } finally {
      setSessDetailLoading(false);
    }
  };

  const openInSimulator = (sessionId: string) => {
    localStorage.setItem("aura_active_session_id", sessionId);
    window.location.href = `/simulator?session_id=${encodeURIComponent(sessionId)}`;
  };

  const deleteSession = async (sessionId: string) => {
    if (!token) return;
    setDeletingSession(true);
    setSessError(null);

    try {
      const res = await fetch(
        `${API_BASE}/logs/sessions/${encodeURIComponent(sessionId)}`,
        {
          method: "DELETE",
          headers: { Authorization: `Bearer ${token}` },
          credentials: "include",
        }
      );

      if (!res.ok) throw new Error(await res.text());
      setSelectedSession(null);
      setDeleteConfirm(false);
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    } catch (e: any) {
      setSessError(e?.message || "Failed to delete session");
    } finally {
      setDeletingSession(false);
    }
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
        <div className="page-wrap">
          <div className="panel">
            {!token ? "Please login." : "Admin only."}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div className="page-wrap chatlogs-page">
        <div className="page-header chatlogs-header">
          <div>
            <h1 className="page-title">Chat Logs</h1>
            <div className="page-subtitle">
              Review saved Jetson conversations and system events
            </div>
          </div>

          <div className="chatlogs-tabs">
            <button
              className={`chatlogs-tab ${tab === "sessions" ? "active" : ""}`}
              onClick={() => setTab("sessions")}
              type="button"
            >
              Sessions
            </button>
            <button
              className={`chatlogs-tab ${tab === "events" ? "active" : ""}`}
              onClick={() => setTab("events")}
              type="button"
            >
              Event Log
            </button>
          </div>
        </div>

        {tab === "sessions" && (
          <div className="chatlogs-layout">
            <aside className="card chatlogs-sidebar">
              <div className="chatlogs-sidebar-top">
                <div>
                  <div className="chatlogs-section-title">Sessions</div>
                  <div className="chatlogs-section-subtitle">
                    {sessLoading ? "Loading..." : `${sessions.length} saved`}
                  </div>
                </div>

                <button
                  className="btn chatlogs-action-btn"
                  onClick={() => void fetchSessions()}
                  type="button"
                >
                  Refresh
                </button>
              </div>

              <input
                className="input chatlogs-session-search"
                value={sessionSearch}
                onChange={(e) => setSessionSearch(e.target.value)}
                placeholder="Search sessions..."
              />

              {sessError && <div className="panel">{sessError}</div>}

              <div className="chatlogs-session-list">
                {!sessLoading && filteredSessions.length === 0 ? (
                  <div className="chatlogs-empty-state">No sessions found.</div>
                ) : (
                  filteredSessions.map((s) => {
                    const selected = selectedSession?.session_id === s.session_id;

                    return (
                      <button
                        key={s.session_id}
                        type="button"
                        className={`chatlogs-session-card ${selected ? "selected" : ""}`}
                        onClick={() => void openSession(s.session_id)}
                      >
                        <div className="chatlogs-session-avatar">
                          {initialsFromTitle(s.title, "J")}
                        </div>

                        <div className="chatlogs-session-content">
                          <div className="chatlogs-session-row">
                            <div className="chatlogs-session-title">
                              {s.title || s.session_id}
                            </div>
                            <div className="chatlogs-session-time">
                              {fmtTimeShort(s.updated_ts)}
                            </div>
                          </div>

                          <div className="chatlogs-session-owner">
                            {s.owner_email || "unknown owner"}
                          </div>

                          <div className="chatlogs-session-meta">
                            <span>{s.message_count} messages</span>
                            {s.db_name && <span>{s.db_name}</span>}
                            {s.device_id && <span>{s.device_id}</span>}
                            {s.source && <span>{s.source}</span>}
                          </div>
                        </div>
                      </button>
                    );
                  })
                )}
              </div>
            </aside>

            <section className="card chatlogs-thread-panel">
              {!selectedSession && !sessDetailLoading && (
                <div className="chatlogs-thread-empty">
                  Select a session to view the conversation.
                </div>
              )}

              {sessDetailLoading && (
                <div className="chatlogs-thread-empty">Loading session...</div>
              )}

              {selectedSession && !sessDetailLoading && (
                <>
                  <div className="chatlogs-thread-header">
                    <div>
                      <h2 className="chatlogs-thread-title">
                        {selectedSession.title || selectedSession.session_id}
                      </h2>

                      <div className="chatlogs-thread-meta">
                        {selectedSession.owner_email || "unknown owner"}
                        {selectedSession.owner_role
                          ? ` • ${selectedSession.owner_role}`
                          : ""}
                        {selectedSession.db_name ? ` • ${selectedSession.db_name}` : ""}
                        {selectedSession.device_id
                          ? ` • ${selectedSession.device_id}`
                          : ""}
                      </div>

                      <div className="chatlogs-thread-dates">
                        Created: {fmtTime(selectedSession.created_ts)} • Updated:{" "}
                        {fmtTime(selectedSession.updated_ts)}
                      </div>
                    </div>

                    <div className="chatlogs-thread-actions">
                      <button
                        className="btn chatlogs-action-btn"
                        onClick={() => openInSimulator(selectedSession.session_id)}
                        type="button"
                      >
                        Open in Simulator
                      </button>

                      {!deleteConfirm ? (
                        <button
                          className="btn chatlogs-action-btn"
                          onClick={() => setDeleteConfirm(true)}
                          type="button"
                        >
                          Delete
                        </button>
                      ) : (
                        <>
                          <button
                            className="btn chatlogs-action-btn"
                            disabled={deletingSession}
                            onClick={() => void deleteSession(selectedSession.session_id)}
                            type="button"
                          >
                            {deletingSession ? "Deleting..." : "Confirm Delete"}
                          </button>
                          <button
                            className="btn chatlogs-action-btn chatlogs-action-btn-secondary"
                            onClick={() => setDeleteConfirm(false)}
                            disabled={deletingSession}
                            type="button"
                          >
                            Cancel
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="chatlogs-thread-body">
                    {selectedSession.history.length === 0 ? (
                      <div className="chatlogs-thread-empty">
                        This session has no messages.
                      </div>
                    ) : (
                      selectedSession.history.map((msg, i) => {
                        const isUser = msg.role === "user";
                        const isError = msg.role === "error";

                        return (
                          <div
                            key={`${selectedSession.session_id}-${i}-${msg.ts ?? i}`}
                            className={`chatlogs-message-row ${
                              isUser ? "user" : "assistant"
                            }`}
                          >
                            <div
                              className={`chatlogs-message ${
                                isUser
                                  ? "user"
                                  : isError
                                  ? "error"
                                  : "assistant"
                              }`}
                            >
                              <div className="chatlogs-message-top">
                                <span className="chatlogs-message-role">
                                  {msg.role === "assistant"
                                    ? "AURA"
                                    : msg.role === "user"
                                    ? "User"
                                    : "Error"}
                                </span>
                                {msg.ts && (
                                  <span className="chatlogs-message-time">
                                    {fmtTime(msg.ts)}
                                  </span>
                                )}
                              </div>

                              <div className="chatlogs-message-content">
                                {msg.content}
                              </div>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </>
              )}
            </section>
          </div>
        )}

        {tab === "events" && (
          <div className="card card-pad chatlogs-events-panel">
            <div className="chatlogs-events-controls">
              <input
                value={q}
                onChange={(e) => {
                  setOffset(0);
                  setQ(e.target.value);
                }}
                placeholder="Search email, prompt, metadata..."
                className="input"
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

              <button
                className="btn chatlogs-action-btn"
                onClick={() => void fetchLogs()}
                type="button"
              >
                Search
              </button>

              <button
                className="btn chatlogs-action-btn chatlogs-action-btn-secondary"
                onClick={() => {
                  setQ("");
                  setRole("");
                  setEvent("");
                  setOffset(0);
                }}
                disabled={evLoading}
                type="button"
              >
                Clear
              </button>
            </div>

            <div className="chatlogs-events-summary">
              <span className="badge">Matched: {matched}</span>
            </div>

            {evError && <div className="panel">{evError}</div>}

            <div className="chatlogs-events-table-wrap">
              {items.length === 0 && !evLoading ? (
                <div className="chatlogs-empty-state">No logs found.</div>
              ) : (
                <table className="chatlogs-events-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>User</th>
                      <th>Role</th>
                      <th>Event</th>
                      <th>Prompt</th>
                      <th>Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((it, idx) => (
                      <tr key={`${it.ts}-${idx}`}>
                        <td>{fmtTime(it.ts)}</td>
                        <td className="chatlogs-event-user">{it.user_email || "-"}</td>
                        <td className="chatlogs-event-role">{it.user_role || "-"}</td>
                        <td
                          className={`chatlogs-event-type ${
                            it.event === "chat_error" ? "error" : ""
                          }`}
                        >
                          {it.event || "-"}
                        </td>
                        <td className="chatlogs-event-prompt">
                          {it.prompt ? (
                            <>
                              {it.prompt.slice(0, 500)}
                              {it.prompt.length > 500 ? "…" : ""}
                            </>
                          ) : (
                            "-"
                          )}
                        </td>
                        <td className="chatlogs-event-latency">
                          {typeof it.latency_ms === "number"
                            ? `${it.latency_ms}ms`
                            : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            <div className="chatlogs-pagination">
              <div className="muted">
                Showing {items.length} / {matched} (offset {offset})
              </div>

              <div className="chatlogs-pagination-actions">
                <button
                  className="btn chatlogs-action-btn chatlogs-action-btn-secondary"
                  onClick={() => setOffset((v) => Math.max(0, v - limit))}
                  disabled={evLoading || offset === 0}
                  type="button"
                >
                  ← Prev
                </button>
                <button
                  className="btn chatlogs-action-btn"
                  onClick={() => setOffset((v) => v + limit)}
                  disabled={evLoading || offset + limit >= matched}
                  type="button"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
