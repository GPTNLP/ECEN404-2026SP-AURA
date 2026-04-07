import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useAuth } from "../services/authService";
import { loadPrefs } from "../services/prefs";

type ChatMsg = {
  role: "user" | "ai" | "error";
  content: string;
  sources?: string[];
};

type ChatResponse = {
  answer?: string;
  sources?: string[];
  detail?: string;
  message?: string;
};

export default function SimulatorPage() {
  const { token, user } = useAuth();

  const [query, setQuery] = useState("");
  const [history, setHistory] = useState<ChatMsg[]>([]);
  const [loading, setLoading] = useState(false);

  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [statusText, setStatusText] = useState<string>("");

  const [databases, setDatabases] = useState<string[]>([]);
  const [activeDb, setActiveDb] = useState<string>("");

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Voice state (mic capture)
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // prefs (loaded from localStorage)
  const [prefs, setPrefs] = useState(() => loadPrefs());

  const voiceInputEnabled = prefs.voiceInputEnabled;
  const speakAiEnabled = prefs.speakAiEnabled;

  const API_URL = useMemo(() => {
    return (
      (import.meta.env.VITE_AUTH_API_BASE as string | undefined)?.trim() ||
      (import.meta.env.VITE_API_URL as string | undefined)?.trim() ||
      "http://127.0.0.1:9000"
    );
  }, []);

  const authHeaders = useMemo(() => {
    const h = new Headers();
    if (token) h.set("Authorization", `Bearer ${token}`);
    return h;
  }, [token]);

  async function readError(res: Response, fallback: string) {
    try {
      const j = await res.json();
      return j?.detail || j?.message || fallback;
    } catch {
      try {
        const t = await res.text();
        return t || fallback;
      } catch {
        return fallback;
      }
    }
  }

  const writeLog = async (payload: {
    event: string;
    prompt?: string;
    response_preview?: string;
    latency_ms?: number;
    meta?: Record<string, any>;
  }) => {
    if (!token) return;
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      await fetch(`${API_URL}/logs/write`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          event: payload.event,
          user_email: user?.email,
          user_role: user?.role,
          prompt: payload.prompt,
          response_preview: payload.response_preview,
          model: "rag-chat",
          latency_ms: payload.latency_ms,
          meta: payload.meta || {},
        }),
      });
    } catch {
      // swallow
    }
  };

  const speakText = async (text: string) => {
    if (!speakAiEnabled || !token) return;
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      await fetch(`${API_URL}/api/tts/speak`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify({ text, interrupt: true }),
      });
    } catch {
      // swallow
    }
  };

  // cleanup
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  // auto-scroll
  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, loading]);

  // refresh prefs when Settings changes them
  useEffect(() => {
    const refresh = () => setPrefs(loadPrefs());

    const onStorage = () => {
      refresh();
    };

    const onCustom = () => refresh();

    window.addEventListener("storage", onStorage);
    window.addEventListener("aura:prefs", onCustom as any);

    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("aura:prefs", onCustom as any);
    };
  }, []);

  // backend ping
  useEffect(() => {
    let timer: any = null;
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
      if (!timer) timer = setInterval(ping, 15000);
    };

    const stop = () => {
      if (timer) clearInterval(timer);
      timer = null;
    };

    const onVis = () => (document.hidden ? stop() : start());

    ping();
    start();
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelled = true;
      stop();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [API_URL]);

  // DB list
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await fetch(`${API_URL}/api/databases`, { headers: authHeaders });
        const data = await res.json().catch(() => null);
        const list = Array.isArray(data?.databases) ? (data.databases as string[]) : [];
        if (cancelled) return;

        setDatabases(list);
        setActiveDb((prev) => {
          if (prev && list.includes(prev)) return prev;
          return list[0] || "";
        });
      } catch {
        if (!cancelled) {
          setDatabases([]);
          setActiveDb("");
        }
      }
    };

    load();
    const t = setInterval(load, 20000);

    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [API_URL, authHeaders]);

  const handleSearch = async () => {
    const q = query.trim();
    if (!q || loading) return;

    if (!activeDb) {
      setStatusText("❌ No database selected. Create/build one on the Database page first.");
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatusText("");
    setHistory((prev) => [...prev, { role: "user", content: q }]);
    setQuery("");
    setLoading(true);

    const t0 = performance.now();

    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_URL}/api/jetson/chat`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify({
          db_name: activeDb,
          query: q,
          session_id: `${user?.email || "anon"}::${activeDb}`,
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

      const sources = Array.isArray(data?.sources) ? data.sources : [];
      setHistory((prev) => [...prev, { role: "ai", content: answer, sources }]);

      void speakText(answer);

      const latency = Math.round(performance.now() - t0);

      await writeLog({
        event: "chat",
        prompt: q,
        response_preview: answer.slice(0, 600),
        latency_ms: latency,
        meta: { db: activeDb, engine: "jetson", sources_count: sources.length },
      });
    } catch (err: any) {
      if (err?.name === "AbortError") return;

      const msg = `Simulation Error: ${err?.message || String(err)}`;
      setHistory((prev) => [...prev, { role: "error", content: msg }]);

      const latency = Math.round(performance.now() - t0);

      await writeLog({
        event: "chat_error",
        prompt: q,
        response_preview: msg.slice(0, 600),
        latency_ms: latency,
        meta: { db: activeDb, engine: "jetson" },
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleRecord = async () => {
    if (!voiceInputEnabled) return;

    if (recording) {
      mediaRecorderRef.current?.stop();
      setRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        try {
          streamRef.current?.getTracks().forEach((t) => t.stop());
          streamRef.current = null;

          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          chunksRef.current = [];

          const fd = new FormData();
          fd.append("file", blob, "speech.webm");

          const res = await fetch(`${API_URL}/api/stt/transcribe`, {
            method: "POST",
            headers: token ? { Authorization: `Bearer ${token}` } : undefined,
            credentials: "include",
            body: fd,
          });

          if (!res.ok) throw new Error(await readError(res, "STT failed"));
          const data = await res.json().catch(() => null);
          const text = (data?.text || "").trim();
          if (text) setQuery((prev) => (prev ? `${prev} ${text}` : text));
        } catch (e: any) {
          setStatusText(`🎤 STT error: ${e?.message || "Failed"}`);
        }
      };

      mr.start();
      setRecording(true);
    } catch (e: any) {
      setStatusText(`🎤 Mic error: ${e?.message || "Permission denied"}`);
    }
  };

  const statusDotClass =
    apiOnline === null ? "bg-slate-300" : apiOnline ? "bg-emerald-500" : "bg-red-500";

  return (
    <div style={{ padding: 18 }}>
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          background: "var(--card-bg)",
          border: "1px solid var(--card-border)",
          borderRadius: "var(--card-radius)",
          overflow: "hidden",
          boxShadow: "var(--shadow)",
        }}
      >
        <div
          style={{
            padding: 18,
            background: "color-mix(in srgb, var(--card-bg) 80%, var(--accent-soft))",
            borderBottom: "1px solid var(--card-border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
              <h1 style={{ margin: 0, fontSize: 22, fontWeight: 900, color: "var(--text)" }}>
                RAG Chat
              </h1>

              <span
                style={{
                  fontSize: 12,
                  padding: "4px 10px",
                  borderRadius: 999,
                  border: "1px solid var(--card-border)",
                  background: "color-mix(in srgb, var(--card-bg) 85%, var(--accent-soft))",
                  color: "var(--muted-text)",
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                }}
              >
                API: {API_URL}
              </span>
            </div>

            <p style={{ margin: "6px 0 0", color: "var(--muted-text)", fontSize: 13 }}>
              Pick a database and ask questions. Voice input + TTS can be enabled in Settings.
            </p>
          </div>

          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "6px 10px",
              borderRadius: 999,
              border: "1px solid var(--card-border)",
              background: "color-mix(in srgb, var(--card-bg) 85%, var(--accent-soft))",
              color: "var(--muted-text)",
              fontSize: 12,
              fontWeight: 800,
            }}
            title="Backend status"
          >
            <span className={`inline-block w-2.5 h-2.5 rounded-full ${statusDotClass}`} />
            {apiOnline === null ? "Checking…" : apiOnline ? "Backend online" : "Backend offline"}
          </span>
        </div>

        <div
          style={{
            padding: 14,
            borderBottom: "1px solid var(--card-border)",
            display: "flex",
            gap: 10,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ fontSize: 12, color: "var(--muted-text)", fontWeight: 900 }}>Database</div>

            <select
              value={activeDb}
              onChange={(e) => setActiveDb(e.target.value)}
              style={{
                minWidth: 260,
                padding: "10px 12px",
                borderRadius: 12,
                border: "1px solid var(--card-border)",
                background: "var(--card-bg)",
                color: "var(--text)",
                fontWeight: 800,
              }}
              disabled={databases.length === 0}
            >
              {databases.length === 0 ? (
                <option value="">No databases found</option>
              ) : (
                databases.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))
              )}
            </select>

            <span style={{ fontSize: 12, color: "var(--muted-text)" }}>
              {databases.length ? `${databases.length} available` : "Create one in Database page"}
            </span>
          </div>

          {statusText && (
            <div
              style={{
                marginLeft: "auto",
                padding: "10px 12px",
                borderRadius: 12,
                border: "1px solid var(--card-border)",
                background: "color-mix(in srgb, var(--card-bg) 85%, var(--accent-soft))",
                color: "var(--muted-text)",
                fontSize: 13,
              }}
            >
              {statusText}
            </div>
          )}
        </div>

        <div
          ref={scrollRef}
          style={{
            height: 560,
            overflowY: "auto",
            padding: 16,
            background: "color-mix(in srgb, var(--bg) 85%, var(--accent-soft))",
          }}
        >
          {history.length === 0 && !loading && (
            <div style={{ textAlign: "center", color: "var(--muted-text)", padding: "70px 16px" }}>
              <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 8 }}>Ready.</div>
              <div style={{ fontSize: 13, maxWidth: 680, margin: "0 auto" }}>
                Choose a database above, then ask something like:
                <span style={{ fontFamily: "monospace" }}> “Explain Ohm’s law with units.”</span>
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
                key={i}
                style={{
                  display: "flex",
                  justifyContent: isUser ? "flex-end" : "flex-start",
                  marginBottom: 12,
                }}
              >
                <div style={bubbleStyle}>
                  {msg.content}
                  {msg.sources && msg.sources.length > 0 && (
                    <div
                      style={{
                        marginTop: 10,
                        paddingTop: 10,
                        borderTop: "1px solid var(--card-border)",
                        fontSize: 12,
                        opacity: 0.9,
                        fontFamily:
                          "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                      }}
                    >
                      <strong>Sources:</strong> {msg.sources.join(", ")}
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {loading && (
            <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 12 }}>
              <div
                style={{
                  maxWidth: "78%",
                  padding: "12px 14px",
                  borderRadius: 16,
                  borderBottomLeftRadius: 6,
                  border: "1px solid var(--card-border)",
                  background: "var(--card-bg)",
                  color: "var(--muted-text)",
                  fontSize: 14,
                  boxShadow: "var(--shadow)",
                }}
              >
                AURA is thinking…
              </div>
            </div>
          )}
        </div>

        <div
          style={{
            padding: 14,
            borderTop: "1px solid var(--card-border)",
            background: "var(--card-bg)",
            display: "flex",
            gap: 10,
            alignItems: "center",
          }}
        >
          <button
            onClick={toggleRecord}
            disabled={!voiceInputEnabled || loading}
            title={
              !voiceInputEnabled
                ? "Enable Voice Input in Settings"
                : recording
                ? "Stop recording"
                : "Record voice"
            }
            style={{
              width: 44,
              height: 44,
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: recording
                ? "color-mix(in srgb, var(--accent) 35%, var(--card-bg))"
                : "var(--card-bg)",
              fontWeight: 900,
              cursor: !voiceInputEnabled || loading ? "not-allowed" : "pointer",
              opacity: !voiceInputEnabled || loading ? 0.6 : 1,
            }}
          >
            {recording ? "■" : "🎤"}
          </button>

          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={activeDb ? `Ask ${activeDb}…` : "Create a database first…"}
            disabled={loading}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSearch();
              }
            }}
            style={{
              flex: 1,
              padding: "12px 12px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: "var(--card-bg)",
              color: "var(--text)",
              outline: "none",
            }}
          />

          <button
            onClick={handleSearch}
            disabled={loading || !query.trim() || !activeDb}
            style={{
              padding: "12px 16px",
              borderRadius: 12,
              border: "1px solid rgba(0,0,0,0)",
              background: "var(--accent)",
              color: "white",
              fontWeight: 900,
              cursor: loading || !query.trim() || !activeDb ? "not-allowed" : "pointer",
              opacity: loading || !query.trim() || !activeDb ? 0.6 : 1,
              boxShadow: "var(--shadow)",
            }}
          >
            Ask
          </button>
        </div>

        <div
          style={{
            padding: "10px 14px",
            borderTop: "1px solid var(--card-border)",
            fontSize: 12,
            color: "var(--muted-text)",
          }}
        >
          Voice Input: {voiceInputEnabled ? "On" : "Off"} • Speak AI: {speakAiEnabled ? "On" : "Off"}{" "}
          (toggle in Settings)
        </div>
      </div>
    </div>
  );
}