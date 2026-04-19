import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useAuth } from "../services/authService";
import "../styles/tamanagepage.css";

type TaItem = {
  email: string;
  added_by?: string;
  addedBy?: string;
  created_at?: string;
  createdAt?: string;
  added_at?: string;
  addedAt?: string;
  added_ts?: number | string;
  addedTs?: number | string;
  ts?: number | string;
  timestamp?: number | string;
};

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined)?.trim() ||
  "http://127.0.0.1:9000";

function normalizeEmail(value: string) {
  return value.trim().toLowerCase();
}

function isTamuEmail(value: string) {
  return /^[a-z0-9._%+-]+@tamu\.edu$/i.test(normalizeEmail(value));
}

function toMillis(value?: string | number) {
  if (value === undefined || value === null || value === "") return 0;

  if (typeof value === "number") {
    return value < 10_000_000_000 ? value * 1000 : value;
  }

  const raw = String(value).trim();
  if (!raw) return 0;

  if (/^\d+$/.test(raw)) {
    const num = Number(raw);
    return num < 10_000_000_000 ? num * 1000 : num;
  }

  const parsed = Date.parse(raw);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function formatDateTime(value?: string | number) {
  const ms = toMillis(value);
  if (!ms) return "-";
  return new Date(ms).toLocaleString();
}

function getAddedValue(ta: TaItem) {
  return (
    ta.added_ts ??
    ta.addedTs ??
    ta.created_at ??
    ta.createdAt ??
    ta.added_at ??
    ta.addedAt ??
    ta.ts ??
    ta.timestamp
  );
}

function getAddedByValue(ta: TaItem, currentUserEmail?: string | null) {
  const addedBy = normalizeEmail(ta.added_by || ta.addedBy || "");
  const me = normalizeEmail(currentUserEmail || "");

  if (!addedBy) return "-";
  if (me && addedBy === me) return "You";
  return addedBy;
}

export default function TaManagerPage() {
  const { token, user } = useAuth();
  const isAdmin = user?.role === "admin";

  const [email, setEmail] = useState("");
  const [tas, setTas] = useState<TaItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [adding, setAdding] = useState(false);
  const [removingEmail, setRemovingEmail] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  const cleanEmail = useMemo(() => normalizeEmail(email), [email]);
  const emailIsValid = useMemo(() => !cleanEmail || isTamuEmail(cleanEmail), [cleanEmail]);

  const authHeaders = useMemo(() => {
    const h = new Headers();
    if (token) h.set("Authorization", `Bearer ${token}`);
    h.set("Content-Type", "application/json");
    return h;
  }, [token]);

  const loadTAs = async () => {
    if (!token) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/admin/ta/list`, {
        headers: authHeaders,
        credentials: "include",
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Failed to load TAs");

      const rawItems = Array.isArray(data?.tas)
        ? data.tas
        : Array.isArray(data?.items)
          ? data.items
          : [];

      const next = [...(rawItems as TaItem[])].sort((a, b) => {
        return toMillis(getAddedValue(b)) - toMillis(getAddedValue(a));
      });

      setTas(next);
    } catch (e: any) {
      setError(e?.message || "Failed to load TAs");
      setTas([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token && isAdmin) void loadTAs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, isAdmin]);

  const handleAdd = async (e?: FormEvent) => {
    e?.preventDefault();

    const nextEmail = normalizeEmail(email);

    if (!nextEmail) {
      setError("Enter a TA email first.");
      setStatus("");
      return;
    }

    if (!isTamuEmail(nextEmail)) {
      setError("Only @tamu.edu email addresses are allowed.");
      setStatus("");
      return;
    }

    if (!token) return;

    setAdding(true);
    setError("");
    setStatus("");

    try {
      const res = await fetch(`${API_BASE}/admin/ta/add`, {
        method: "POST",
        headers: authHeaders,
        credentials: "include",
        body: JSON.stringify({ email: nextEmail }),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Failed to add TA");

      setEmail("");
      setStatus(`Added ${nextEmail} as a TA.`);
      await loadTAs();
    } catch (e: any) {
      setError(e?.message || "Failed to add TA");
    } finally {
      setAdding(false);
    }
  };

  const handleRemove = async (taEmail: string) => {
    if (!token) return;

    const nextEmail = normalizeEmail(taEmail);
    setRemovingEmail(nextEmail);
    setError("");
    setStatus("");

    try {
      const res = await fetch(`${API_BASE}/admin/ta/remove`, {
        method: "POST",
        headers: authHeaders,
        credentials: "include",
        body: JSON.stringify({ email: nextEmail }),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Failed to remove TA");

      setStatus(`Removed ${nextEmail}.`);
      await loadTAs();
    } catch (e: any) {
      setError(e?.message || "Failed to remove TA");
    } finally {
      setRemovingEmail("");
    }
  };

  if (!token || !isAdmin) {
    return (
      <div className="page-shell">
        <div className="page-wrap">
          <div className="panel">{!token ? "Please login." : "Admin only."}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div className="page-wrap tamanage-page">
        <div className="page-header tamanage-header">
          <div>
            <h1 className="page-title">TA Manager</h1>
            <div className="page-subtitle">
              Add or remove TA access by email. Only @tamu.edu emails are allowed.
            </div>
          </div>
        </div>

        <div className="card card-pad tamanage-card">
          <form className="tamanage-add-row" onSubmit={handleAdd}>
            <div className="tamanage-input-wrap">
              <input
                className={`input tamanage-input ${!emailIsValid ? "is-invalid" : ""}`}
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError("");
                  setStatus("");
                }}
                placeholder="someone@tamu.edu"
                autoComplete="off"
              />
              {!emailIsValid && (
                <div className="tamanage-warning">
                  Only @tamu.edu email addresses are allowed.
                </div>
              )}
            </div>

            <button
              className="btn tamanage-action-btn"
              type="submit"
              disabled={adding || !cleanEmail || !emailIsValid}
            >
              {adding ? "Adding..." : "Add"}
            </button>
          </form>

          {(error || status) && (
            <div className={`tamanage-message ${error ? "is-error" : "is-success"}`}>
              {error || status}
            </div>
          )}
        </div>

        <div className="card tamanage-table-card">
          <div className="tamanage-table-head">
            <div>
              <div className="tamanage-section-title">Current TAs</div>
              <div className="tamanage-section-subtitle">
                {loading ? "Loading..." : `${tas.length} total`}
              </div>
            </div>

            <button className="btn tamanage-action-btn" onClick={() => void loadTAs()} type="button">
              Refresh
            </button>
          </div>

          <div className="tamanage-table-wrap">
            <table className="tamanage-table">
              <thead>
                <tr>
                  <th>Email</th>
                  <th>Added By</th>
                  <th>Added</th>
                  <th className="tamanage-actions-col">Action</th>
                </tr>
              </thead>
              <tbody>
                {!loading && tas.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="tamanage-empty">
                      No TAs yet.
                    </td>
                  </tr>
                ) : (
                  tas.map((ta) => {
                    const taEmail = normalizeEmail(ta.email);
                    const addedValue = getAddedValue(ta);
                    const addedByLabel = getAddedByValue(ta, user?.email || null);

                    return (
                      <tr key={taEmail}>
                        <td className="tamanage-email">{ta.email}</td>
                        <td>{addedByLabel}</td>
                        <td>{formatDateTime(addedValue)}</td>
                        <td className="tamanage-actions-col">
                          <button
                            className="btn tamanage-action-btn tamanage-action-btn-secondary"
                            type="button"
                            disabled={removingEmail === taEmail}
                            onClick={() => void handleRemove(taEmail)}
                          >
                            {removingEmail === taEmail ? "Removing..." : "Remove"}
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}