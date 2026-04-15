import { useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "../services/authService";
import "../styles/page-ui.css";
import "../styles/simulator.css";

type ChatMsg = {
  role: "user" | "assistant" | "error";
  content: string;
  ts?: number;
};

type ChatResponse = {
  ok?: boolean;
  answer?: string;
  status?: string;
  detail?: string;
  message?: string;
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

type SessionDetail = SessionMeta & {
  history: Array<{
    role: string;
    content: string;
    ts?: number;
  }>;
};

function fmtTime(ts?: number) {
  if (!ts) return "-";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

function fmtShortTime(ts?: number) {
  if (!ts) return "";
  try {
    return new Date(ts * 1000).toLocaleString([], {
      month: "numeric",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return String(ts);
  }
}

function buildSessionTitle(query: string) {
  const trimmed = query.trim().replace(/\s+/g, " ");
  if (!trimmed) return "New chat";
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed;
}

function normalizeHistory(history: SessionDetail["history"] | undefined | null): ChatMsg[] {
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

function initialsFromTitle(title?: string) {
  const raw = (title || "Chat").trim();
  const parts = raw.split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || "").join("") || "C";
}

export default function SimulatorPage() {
  const { token } = useAuth();

  const [query, setQuery] = useState("");
  const [history, setHistory] = useState<ChatMsg[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [statusText, setStatusText] = useState("");
  const [loadedDb, setLoadedDb] = useState(() => localStorage.getItem("aura_loaded_db") || "");

  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activeSessionTitle, setActiveSessionTitle] = useState("New chat");
  const [sessionSearch, setSessionSearch] = useState("");

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const didAutoLoadRef = useRef(false);

  const API_URL = useMemo(() => {
    return (
      (import.meta.env.VITE_AUTH_API_BASE as string | undefined)?.trim() ||
      (import.meta.env.VITE_API_URL as string | undefined)?.trim() ||
      "http://127.0.0.1:9000"
    );
  }, []);

  const DEVICE_ID = useMemo(() => {
    return (import.meta.env.VITE_DEVICE_ID as string | undefined)?.trim() || "jetson-001";
  }, []);

  const authHeaders = useMemo(() => {
    const h = new Headers();
    if (token) h.set("Authorization", `Bearer ${token}`);
    return h;
  }, [token]);

  const filteredSessions = useMemo(() => {
    const needle = sessionSearch.trim().toLowerCase();
    if (!needle) return sessions;

    return sessions.filter((s) => {
      const haystack = [
        s.title,
        s.session_id,
        s.db_name,
        s.device_id,
        s.owner_email,
        s.owner_role,
        s.source,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return haystack.includes(needle);
    });
  }, [sessions, sessionSearch]);

  const refreshLoadedDb = () => {
    setLoadedDb(localStorage.getItem("aura_loaded_db") || "");
  };

  const writeLog = async (payload: {
    event: string;
    prompt?: string;
    response_preview?: string;
    latency_ms?: number;
    meta?: Record<string, unknown>;
  }) => {
    if (!token) return;
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");
      await fetch(`${API_URL}/logs/write`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify({
          event: payload.event,
          prompt: payload.prompt,
          response_preview: payload.response_preview,
          model: "jetson-chat",
          latency_ms: payload.latency_ms,
          meta: payload.meta || {},
        }),
      });
    } catch {
      // ignore log write failures
    }
  };

  const fetchMySessions = async (): Promise<SessionMeta[]> => {
    if (!token) return [];
    setSessionsLoading(true);
    try {
      const res = await fetch(`${API_URL}/logs/my-sessions`, {
        headers: authHeaders,
        credentials: "include",
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const nextSessions = (data.sessions || []) as SessionMeta[];
      setSessions(nextSessions);
      return nextSessions;
    } catch (err: any) {
      console.error("Failed to load sessions", err?.message || err);
      setSessions([]);
      return [];
    } finally {
      setSessionsLoading(false);
    }
  };

  const loadSession = async (sessionId: string) => {
    if (!token) return;

    try {
      const res = await fetch(`${API_URL}/logs/my-sessions/${encodeURIComponent(sessionId)}`, {
        headers: authHeaders,
        credentials: "include",
      });
      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      const session = data.session as SessionDetail;

      setActiveSessionId(session.session_id);
      setActiveSessionTitle(session.title || "New chat");
      setHistory(normalizeHistory(session.history));
      localStorage.setItem("aura_active_session_id", session.session_id);

      if (session.db_name) {
        localStorage.setItem("aura_loaded_db", session.db_name);
        window.dispatchEvent(new Event("aura:loaded-db"));
      }

      const nextUrl = new URL(window.location.href);
      nextUrl.searchParams.set("session_id", session.session_id);
      window.history.replaceState({}, "", nextUrl.toString());
    } catch (err: any) {
      setStatusText(`Failed to load session: ${err?.message || String(err)}`);
    }
  };

  const createSession = async (firstTitle: string, nextHistory: ChatMsg[]) => {
    const headers = new Headers(authHeaders);
    headers.set("Content-Type", "application/json");

    const res = await fetch(`${API_URL}/logs/my-sessions/start`, {
      method: "POST",
      headers,
      credentials: "include",
      body: JSON.stringify({
        title: firstTitle,
        db_name: loadedDb || null,
        device_id: DEVICE_ID,
        history: nextHistory,
      }),
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    const data = await res.json();
    const session = data.session as SessionDetail;

    setActiveSessionId(session.session_id);
    setActiveSessionTitle(session.title || firstTitle);
    localStorage.setItem("aura_active_session_id", session.session_id);

    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("session_id", session.session_id);
    window.history.replaceState({}, "", nextUrl.toString());

    await fetchMySessions();
    return session.session_id;
  };

  const saveSession = async (sessionId: string, nextHistory: ChatMsg[], nextTitle?: string) => {
    const headers = new Headers(authHeaders);
    headers.set("Content-Type", "application/json");

    const res = await fetch(`${API_URL}/logs/my-sessions/${encodeURIComponent(sessionId)}`, {
      method: "POST",
      headers,
      credentials: "include",
      body: JSON.stringify({
        title: nextTitle || activeSessionTitle,
        db_name: loadedDb || null,
        device_id: DEVICE_ID,
        history: nextHistory,
      }),
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    await fetchMySessions();
  };

  const startNewChat = () => {
    setActiveSessionId(null);
    setActiveSessionTitle("New chat");
    setHistory([]);
    setStatusText("");
    setQuery("");
    localStorage.removeItem("aura_active_session_id");

    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.delete("session_id");
    window.history.replaceState({}, "", nextUrl.toString());
  };

  const handleAsk = async () => {
    const q = query.trim();
    if (!q || loading) return;

    if (!loadedDb) {
      setStatusText("No database is loaded on Jetson. Go to Database page and push one first.");
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatusText("");
    setQuery("");
    setLoading(true);

    const userMsg: ChatMsg = { role: "user", content: q, ts: Math.floor(Date.now() / 1000) };
    const optimisticHistory = [...history, userMsg];
    setHistory(optimisticHistory);

    const t0 = performance.now();
    let sessionId = activeSessionId;
    const inferredTitle = activeSessionId ? activeSessionTitle : buildSessionTitle(q);

    try {
      if (!sessionId) {
        sessionId = await createSession(inferredTitle, optimisticHistory);
      } else {
        await saveSession(sessionId, optimisticHistory, inferredTitle);
      }

      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_URL}/device/admin/chat`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify({
          device_id: DEVICE_ID,
          command: "chat_prompt",
          payload: {
            db_name: loadedDb,
            query: q,
            session_id: sessionId,
          },
        }),
        signal: controller.signal,
      });

      let data: ChatResponse | null = null;
      try {
        data = (await res.json()) as ChatResponse;
      } catch {
        data = null;
      }

      if (!res.ok) {
        const msg = data?.detail || data?.message || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      const answer =
        typeof data?.answer === "string" && data.answer.trim()
          ? data.answer
          : "(No answer returned)";

      const aiMsg: ChatMsg = {
        role: "assistant",
        content: answer,
        ts: Math.floor(Date.now() / 1000),
      };

      const finalHistory = [...optimisticHistory, aiMsg];
      setHistory(finalHistory);
      setActiveSessionTitle(inferredTitle);

      if (sessionId) {
        await saveSession(sessionId, finalHistory, inferredTitle);
      }

      const latency = Math.round(performance.now() - t0);
      await writeLog({
        event: "chat",
        prompt: q,
        response_preview: answer.slice(0, 600),
        latency_ms: latency,
        meta: {
          db: loadedDb,
          device_id: DEVICE_ID,
          session_id: sessionId,
          command_status: data?.status || "unknown",
        },
      });
    } catch (err: any) {
      if (err?.name === "AbortError") return;

      const msg = `Simulation Error: ${err?.message || String(err)}`;
      const errorMsg: ChatMsg = {
        role: "error",
        content: msg,
        ts: Math.floor(Date.now() / 1000),
      };
      const erroredHistory = [...optimisticHistory, errorMsg];
      setHistory(erroredHistory);

      if (sessionId) {
        try {
          await saveSession(sessionId, erroredHistory, inferredTitle);
        } catch {
          // ignore secondary save failure
        }
      }

      const latency = Math.round(performance.now() - t0);
      await writeLog({
        event: "chat_error",
        prompt: q,
        response_preview: msg.slice(0, 600),
        latency_ms: latency,
        meta: {
          db: loadedDb,
          device_id: DEVICE_ID,
          session_id: sessionId,
        },
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [history, loading]);

  useEffect(() => {
    window.addEventListener("storage", refreshLoadedDb);
    window.addEventListener("aura:loaded-db", refreshLoadedDb as EventListener);

    return () => {
      window.removeEventListener("storage", refreshLoadedDb);
      window.removeEventListener("aura:loaded-db", refreshLoadedDb as EventListener);
    };
  }, []);

  useEffect(() => {
    let timer: number | null = null;
    let cancelled = false;

    const ping = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, { method: "GET" });
        if (!cancelled) setApiOnline(res.ok);
      } catch {
        if (!cancelled) setApiOnline(false);
      }
    };

    const start = () => {
      if (timer == null) timer = window.setInterval(ping, 15000);
    };

    const stop = () => {
      if (timer != null) {
        window.clearInterval(timer);
        timer = null;
      }
    };

    const onVis = () => {
      if (document.hidden) stop();
      else start();
    };

    void ping();
    start();
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelled = true;
      stop();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [API_URL]);

  useEffect(() => {
    if (!token) return;
    void fetchMySessions();
  }, [token]);

  useEffect(() => {
    if (!token || didAutoLoadRef.current) return;

    didAutoLoadRef.current = true;

    void (async () => {
      const url = new URL(window.location.href);
      const fromUrl = url.searchParams.get("session_id");
      const fromStorage = localStorage.getItem("aura_active_session_id");

      if (fromUrl) {
        await loadSession(fromUrl);
        return;
      }

      if (fromStorage) {
        await loadSession(fromStorage);
        return;
      }

      const currentSessions = sessions.length > 0 ? sessions : await fetchMySessions();
      if (currentSessions.length > 0) {
        await loadSession(currentSessions[0].session_id);
      }
    })();
  }, [token, sessions]);

  return (
    <div className="page-shell">
      <div className="page-wrap simulator-page">
        <div className="page-header simulator-header">
          <div>
            <h1 className="page-title">Simulator</h1>
            <div className="page-subtitle">
              Jetson chat, saved sessions, and database-backed Q&A
            </div>
          </div>

          <div className="simulator-header-actions">
            <div className={`simulator-status-pill ${apiOnline ? "online" : "offline"}`}>
              <span className="simulator-status-dot" />
              {apiOnline === null ? "Checking..." : apiOnline ? "Backend online" : "Backend offline"}
            </div>

            <button
              className="btn simulator-action-btn"
              type="button"
              onClick={() => void fetchMySessions()}
            >
              Refresh
            </button>
          </div>
        </div>

        <div className="simulator-layout">
          <aside className="card simulator-sidebar">
            <div className="simulator-sidebar-top">
              <div>
                <div className="simulator-section-title">Chats</div>
                <div className="simulator-section-subtitle">
                  {sessionsLoading ? "Loading..." : `${sessions.length} saved`}
                </div>
              </div>

              <button className="btn" type="button" onClick={() => void fetchMySessions()}>
                Refresh
              </button>
            </div>

            <div className="simulator-sidebar-info">
              <div className="simulator-sidebar-info-row">
                <span>DB</span>
                <strong>{loadedDb || "(none loaded)"}</strong>
              </div>
              <div className="simulator-sidebar-info-row">
                <span>Device</span>
                <strong>{DEVICE_ID}</strong>
              </div>
            </div>

            <input
              className="input simulator-session-search"
              value={sessionSearch}
              onChange={(e) => setSessionSearch(e.target.value)}
              placeholder="Search chats..."
            />

            <div className="simulator-session-list">
              {filteredSessions.map((s) => {
                const selected = s.session_id === activeSessionId;

                return (
                  <button
                    key={s.session_id}
                    type="button"
                    onClick={() => {
                      localStorage.setItem("aura_active_session_id", s.session_id);
                      void loadSession(s.session_id);
                    }}
                    className={`simulator-session-card ${selected ? "selected" : ""}`}
                  >
                    <div className="simulator-session-avatar">
                      {initialsFromTitle(s.title)}
                    </div>

                    <div className="simulator-session-content">
                      <div className="simulator-session-row">
                        <div className="simulator-session-title">
                          {s.title || "Untitled chat"}
                        </div>
                        <div className="simulator-session-time">
                          {fmtShortTime(s.updated_ts)}
                        </div>
                      </div>

                      <div className="simulator-session-meta">
                        <span>{s.message_count} messages</span>
                        {s.db_name && <span>{s.db_name}</span>}
                      </div>

                      <div className="simulator-session-updated">
                        {fmtTime(s.updated_ts)}
                      </div>
                    </div>
                  </button>
                );
              })}

              {!sessionsLoading && filteredSessions.length === 0 && (
                <div className="simulator-empty-sessions">No saved chats yet.</div>
              )}
            </div>
          </aside>

          <section className="card simulator-main">
            <div className="simulator-main-top">
              <div>
                <h2 className="simulator-thread-title">{activeSessionTitle}</h2>
                <div className="simulator-thread-subtitle">
                  {activeSessionId ? `Session: ${activeSessionId}` : "Unsaved draft chat"}
                </div>
              </div>

              <div className="simulator-thread-badges">
                <span className="badge">{loadedDb ? `DB: ${loadedDb}` : "No DB loaded"}</span>
                <span className="badge">Device: {DEVICE_ID}</span>
              </div>
            </div>

            <div className="simulator-thread-note">
              Ask questions using the database currently loaded from the Database page.
            </div>

            {statusText && <div className="simulator-alert">{statusText}</div>}

            <div ref={scrollRef} className="simulator-thread-body">
              {history.length === 0 && !loading && (
                <div className="simulator-empty-chat">
                  <div className="simulator-empty-icon">AURA</div>
                  <div className="simulator-empty-title">Ready.</div>
                  <div className="simulator-empty-text">
                    {loadedDb
                      ? `Using ${loadedDb}. Ask something like: "Explain Ohm's law with units."`
                      : "Go to Database page and push a DB to Jetson first."}
                  </div>
                </div>
              )}

              {history.map((msg, i) => {
                const isUser = msg.role === "user";
                const isError = msg.role === "error";

                return (
                  <div
                    key={`${msg.role}-${i}-${msg.ts || 0}`}
                    className={`simulator-message-row ${isUser ? "user" : "assistant"}`}
                  >
                    <div
                      className={`simulator-message ${
                        isUser ? "user" : isError ? "error" : "assistant"
                      }`}
                    >
                      <div className="simulator-message-top">
                        <span className="simulator-message-role">
                          {isUser ? "User" : isError ? "Error" : "AURA"}
                        </span>
                        {msg.ts && (
                          <span className="simulator-message-time">{fmtTime(msg.ts)}</span>
                        )}
                      </div>

                      <div className="simulator-message-content">{msg.content}</div>
                    </div>
                  </div>
                );
              })}

              {loading && (
                <div className="simulator-message-row assistant">
                  <div className="simulator-message assistant typing">
                    <div className="simulator-message-top">
                      <span className="simulator-message-role">AURA</span>
                    </div>
                    <div className="simulator-typing">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="simulator-composer">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={loadedDb ? `Ask ${loadedDb}...` : "Push a database from Database page first..."}
                disabled={loading || !loadedDb}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleAsk();
                  }
                }}
                className="simulator-input"
                rows={1}
              />

              <button
                onClick={() => void handleAsk()}
                disabled={loading || !query.trim() || !loadedDb}
                className="btn btn-primary simulator-send-btn"
                type="button"
              >
                Ask
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}