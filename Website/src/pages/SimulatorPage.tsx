import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useAuth } from "../services/authService";

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
    <div style={{ display: "grid", gridTemplateColumns: "320px minmax(0, 1fr)", gap: 18 }}>
      <aside
        className="card card-pad"
        style={{
          minHeight: "78vh",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
          <div>
            <h2 style={{ margin: 0 }}>Chats</h2>
            <div style={{ fontSize: 13, opacity: 0.75 }}>
              {sessionsLoading ? "Loading..." : `${sessions.length} saved`}
            </div>
          </div>

          <button className="btn" onClick={startNewChat}>
            New chat
          </button>
        </div>

        <div
          style={{
            fontSize: 12,
            padding: 10,
            borderRadius: 12,
            background: "var(--panel-2, rgba(0,0,0,.03))",
            border: "1px solid var(--card-border)",
            lineHeight: 1.45,
          }}
        >
          <div><strong>DB:</strong> {loadedDb || "(none loaded)"}</div>
          <div><strong>Device:</strong> {DEVICE_ID}</div>
        </div>

        <div style={{ overflowY: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
          {sessions.map((s) => {
            const selected = s.session_id === activeSessionId;
            return (
              <button
                key={s.session_id}
                type="button"
                onClick={() => {
                  localStorage.setItem("aura_active_session_id", s.session_id);
                  void loadSession(s.session_id);
                }}
                style={{
                  textAlign: "left",
                  width: "100%",
                  padding: 12,
                  borderRadius: 14,
                  border: selected
                    ? "1px solid var(--accent)"
                    : "1px solid var(--card-border)",
                  background: selected
                    ? "color-mix(in srgb, var(--accent) 10%, var(--card-bg))"
                    : "var(--card-bg)",
                  color: "var(--text)",
                  cursor: "pointer",
                }}
              >
                <div style={{ fontWeight: 800, marginBottom: 4 }}>
                  {s.title || "Untitled chat"}
                </div>
                <div style={{ fontSize: 12, opacity: 0.75 }}>
                  {s.message_count} messages
                  {s.db_name ? ` • ${s.db_name}` : ""}
                </div>
                <div style={{ fontSize: 12, opacity: 0.65, marginTop: 4 }}>
                  {fmtTime(s.updated_ts)}
                </div>
              </button>
            );
          })}

          {!sessionsLoading && sessions.length === 0 && (
            <div
              style={{
                fontSize: 14,
                opacity: 0.7,
                padding: 12,
                borderRadius: 12,
                border: "1px dashed var(--card-border)",
              }}
            >
              No saved chats yet.
            </div>
          )}
        </div>
      </aside>

      <section
        className="card card-pad"
        style={{
          minHeight: "78vh",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
          <div>
            <h2 style={{ margin: 0 }}>{activeSessionTitle}</h2>
            <div style={{ fontSize: 13, opacity: 0.75 }}>
              {activeSessionId ? `Session: ${activeSessionId}` : "Unsaved draft chat"}
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 999,
                display: "inline-block",
                background:
                  apiOnline === null
                    ? "#cbd5e1"
                    : apiOnline
                      ? "#10b981"
                      : "#ef4444",
              }}
            />
            {apiOnline === null ? "Checking…" : apiOnline ? "Backend online" : "Backend offline"}
          </div>
        </div>

        <div style={{ fontSize: 13, opacity: 0.82 }}>
          Ask questions using the database currently loaded from the Database page.
        </div>

        {statusText && (
          <div
            style={{
              padding: "10px 12px",
              borderRadius: 12,
              background: "color-mix(in srgb, var(--accent) 8%, var(--card-bg))",
              border: "1px solid var(--card-border)",
            }}
          >
            {statusText}
          </div>
        )}

        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: "auto",
            borderRadius: 16,
            border: "1px solid var(--card-border)",
            background: "var(--panel-2, rgba(0,0,0,.02))",
            padding: 14,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {history.length === 0 && !loading && (
            <div style={{ opacity: 0.8, lineHeight: 1.6 }}>
              <div style={{ fontWeight: 800, marginBottom: 4 }}>Ready.</div>
              <div>
                {loadedDb
                  ? `Using ${loadedDb}. Ask something like: "Explain Ohm's law with units."`
                  : "Go to Database page and push a DB to Jetson first."}
              </div>
            </div>
          )}

          {history.map((msg, i) => {
            const isUser = msg.role === "user";
            const isError = msg.role === "error";

            const bubbleStyle: CSSProperties = {
              maxWidth: "78%",
              padding: "12px 14px",
              borderRadius: 16,
              whiteSpace: "pre-wrap",
              lineHeight: 1.45,
              fontSize: 14,
              boxShadow: "var(--shadow)",
              border: "1px solid var(--card-border)",
              background: "var(--card-bg)",
              color: "var(--text)",
            };

            if (isUser) {
              bubbleStyle.background = "var(--accent)";
              bubbleStyle.color = "white";
              bubbleStyle.border = "1px solid rgba(0,0,0,0)";
              bubbleStyle.borderBottomRightRadius = 6;
            } else if (isError) {
              bubbleStyle.background = "color-mix(in srgb, var(--status-bad) 12%, var(--card-bg))";
              bubbleStyle.color = "color-mix(in srgb, var(--status-bad) 80%, var(--text))";
              bubbleStyle.border =
                "1px solid color-mix(in srgb, var(--status-bad) 35%, var(--card-border))";
              bubbleStyle.borderBottomLeftRadius = 6;
            } else {
              bubbleStyle.borderBottomLeftRadius = 6;
            }

            return (
              <div
                key={`${msg.role}-${i}-${msg.ts || 0}`}
                style={{
                  display: "flex",
                  justifyContent: isUser ? "flex-end" : "flex-start",
                }}
              >
                <div style={bubbleStyle}>{msg.content}</div>
              </div>
            );
          })}

          {loading && (
            <div style={{ display: "flex", justifyContent: "flex-start" }}>
              <div
                style={{
                  maxWidth: "78%",
                  padding: "12px 14px",
                  borderRadius: 16,
                  borderBottomLeftRadius: 6,
                  border: "1px solid var(--card-border)",
                  background: "var(--card-bg)",
                }}
              >
                AURA is thinking…
              </div>
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={loadedDb ? `Ask ${loadedDb}…` : "Push a database from Database page first…"}
            disabled={loading || !loadedDb}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void handleAsk();
              }
            }}
            style={{
              flex: 1,
              minHeight: 56,
              resize: "vertical",
              padding: "12px 12px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: "var(--card-bg)",
              color: "var(--text)",
              outline: "none",
              fontFamily: "inherit",
            }}
          />

          <button
            onClick={() => void handleAsk()}
            disabled={loading || !query.trim() || !loadedDb}
            style={{
              padding: "12px 16px",
              borderRadius: 12,
              border: "1px solid rgba(0,0,0,0)",
              background: "var(--accent)",
              color: "white",
              fontWeight: 900,
              cursor: loading || !query.trim() || !loadedDb ? "not-allowed" : "pointer",
              opacity: loading || !query.trim() || !loadedDb ? 0.6 : 1,
              boxShadow: "var(--shadow)",
              alignSelf: "stretch",
            }}
          >
            Ask
          </button>
        </div>
      </section>
    </div>
  );
}