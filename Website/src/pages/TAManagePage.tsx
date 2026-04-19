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
};

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined) ||
  "http://127.0.0.1:9000";

function normalizeEmail(value: string) {
  return value.trim().toLowerCase();
}

function isTamuEmail(value: string) {
  return /^[a-z0-9._%+-]+@tamu\.edu$/i.test(normalizeEmail(value));
}

function formatDate(value?: string) {
  if (!value) return "-";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
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

      const next = Array.isArray(data?.tas)
        ? data.tas
        : Array.isArray(data?.items)
        ? data.items
        : [];
      setTas(next as TaItem[]);
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
                    return (
                      <tr key={taEmail}>
                        <td className="tamanage-email">{ta.email}</td>
                        <td>{ta.added_by || ta.addedBy || "-"}</td>
                        <td>{formatDate(ta.created_at || ta.createdAt)}</td>
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
