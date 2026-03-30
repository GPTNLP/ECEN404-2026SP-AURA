import { useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "../services/authService";
import "../styles/page-ui.css";
import "../styles/database.css";

const API_BASE =
  (import.meta.env.VITE_AUTH_API_BASE as string | undefined) ||
  "http://127.0.0.1:9000";

type TreeNode = {
  name: string;
  type: "dir" | "file";
  children?: TreeNode[];
};

type TreeResponse = {
  tree?: TreeNode;
};

type SelectedItem =
  | { kind: "dir"; path: string }
  | { kind: "file"; path: string }
  | null;

type CtxTarget =
  | { kind: "dir"; path: string; name: string }
  | { kind: "file"; path: string; name: string };

type ContextMenuState =
  | { open: false; x: number; y: number; target: null }
  | { open: true; x: number; y: number; target: CtxTarget };

function joinPath(parent: string, name: string) {
  if (!parent) return name;
  return `${parent}/${name}`.replaceAll("//", "/");
}

function dirname(path: string) {
  const p = (path || "").replaceAll("\\", "/").replace(/\/+$/, "");
  const i = p.lastIndexOf("/");
  if (i <= 0) return "";
  return p.slice(0, i);
}

function basename(path: string) {
  const p = (path || "").replaceAll("\\", "/").replace(/\/+$/, "");
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(i + 1) : p;
}

function humanCount(n: number) {
  if (!Number.isFinite(n)) return "-";
  return `${n}`;
}

function splitPath(path: string) {
  const p = (path || "")
    .replaceAll("\\", "/")
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");
  if (!p) return [];
  return p.split("/").filter(Boolean);
}

function containsText(hay: string, needle: string) {
  if (!needle.trim()) return true;
  return hay.toLowerCase().includes(needle.trim().toLowerCase());
}

export default function DatabasePage() {
  const { token, user } = useAuth();
  const isAdmin = user?.role === "admin";
  const isTA = user?.role === "ta";

  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState<
    | ""
    | "tree"
    | "upload"
    | "mkdir"
    | "move"
    | "delete"
    | "db-create"
    | "db-build"
    | "db-stats"
  >("");

  // Documents
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ "": true });

  // Selected item (file OR dir)
  const [selected, setSelected] = useState<SelectedItem>({ kind: "dir", path: "" });

  // UI state
  const [newFolderName, setNewFolderName] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const [moveTargetDir, setMoveTargetDir] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);

  // Upload
  const [files, setFiles] = useState<FileList | null>(null);
  const selectedFiles = useMemo(() => (files ? Array.from(files) : []), [files]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // DBs
  const [dbList, setDbList] = useState<string[]>([]);
  const [activeDb, setActiveDb] = useState<string>("");
  const [newDbName, setNewDbName] = useState<string>("");

  // Folder selection for DB build
  const [folderChecks, setFolderChecks] = useState<Record<string, boolean>>({});
  const [dbStats, setDbStats] = useState<any>(null);

  // Search
  const [treeSearch, setTreeSearch] = useState("");
  const [contentsSearch, setContentsSearch] = useState("");

  // Context menu
  const [ctx, setCtx] = useState<ContextMenuState>({
    open: false,
    x: 0,
    y: 0,
    target: null,
  });

  const renameRef = useRef<HTMLInputElement | null>(null);
  const moveRef = useRef<HTMLSelectElement | null>(null);

  // ✅ always return a real Headers object
  const authHeaders = useMemo(() => {
    const h = new Headers();
    if (token) h.set("Authorization", `Bearer ${token}`);
    return h;
  }, [token]);

  const selectedPath = selected?.path ?? "";
  const selectedKind = selected?.kind ?? "dir";

  const selectedDir = selectedKind === "dir" ? selectedPath : dirname(selectedPath);

  // Flatten folders for dropdowns
  const allFolders = useMemo(() => {
    const out: string[] = [];
    const walk = (node: TreeNode | null, parentPath: string) => {
      if (!node || node.type !== "dir") return;
      const kids = node.children || [];
      for (const ch of kids) {
        const p = joinPath(parentPath, ch.name);
        if (ch.type === "dir") {
          out.push(p);
          walk(ch, p);
        }
      }
    };
    out.push(""); // Documents (root)
    walk(tree, "");
    return out;
  }, [tree]);

  const refreshTree = async () => {
    setBusy("tree");
    try {
      const res = await fetch(`${API_BASE}/api/documents/tree`, {
        headers: authHeaders,
      });
      const data = (await res.json().catch(() => null)) as TreeResponse | null;
      if (!res.ok)
        throw new Error(
          (data as any)?.detail ? (data as any).detail : "Failed to load documents tree"
        );
      setTree((data?.tree as TreeNode) || null);
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  const refreshDatabases = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/databases`, { headers: authHeaders });
      const data = await res.json().catch(() => null);
      const list = Array.isArray(data?.databases) ? (data.databases as string[]) : [];
      setDbList(list);
      if (!activeDb && list.length) setActiveDb(list[0]);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    refreshTree();
    refreshDatabases();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // keep renameValue synced when you click a new item
  useEffect(() => {
    setRenameValue(selected ? basename(selected.path) : "");
    setMoveTargetDir("");
  }, [selected?.kind, selected?.path]);

  // Close context menu on click / escape / scroll
  useEffect(() => {
    if (!ctx.open) return;

    const onDown = () => setCtx({ open: false, x: 0, y: 0, target: null });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setCtx({ open: false, x: 0, y: 0, target: null });
    };

    window.addEventListener("mousedown", onDown);
    window.addEventListener("scroll", onDown, true);
    window.addEventListener("keydown", onKey);

    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("scroll", onDown, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [ctx.open]);

  // ---------- Helpers: get node + folder contents ----------
  const getDirNode = (dirPath: string) => {
    if (!tree || tree.type !== "dir") return null;
    const parts = splitPath(dirPath);
    let cur: TreeNode = tree;

    for (const part of parts) {
      const next = (cur.children || []).find((x) => x.type === "dir" && x.name === part);
      if (!next) return null;
      cur = next;
    }
    return cur;
  };

  const dirNode = useMemo(() => getDirNode(selectedDir), [tree, selectedDir]);

  const dirContents = useMemo(() => {
    const kids = dirNode?.children || [];
    const sorted = [...kids].sort((a, b) => {
      if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    return sorted.filter((n) => {
      const p = joinPath(selectedDir, n.name);
      return containsText(n.name, contentsSearch) || containsText(p, contentsSearch);
    });
  }, [dirNode, selectedDir, contentsSearch]);

  // ---------- Tree rendering ----------
  const toggle = (path: string) => {
    setExpanded((p) => ({ ...p, [path]: !p[path] }));
  };
  const isExpanded = (path: string) => expanded[path] ?? false;

  const openCtxMenu = (
    e: React.MouseEvent,
    target: CtxTarget
  ) => {
    e.preventDefault();
    setCtx({
      open: true,
      x: e.clientX,
      y: e.clientY,
      target,
    });
  };

  const setSelectedFromTarget = (t: CtxTarget) => {
    setSelected({ kind: t.kind, path: t.path });
  };

  const toggleInclude = (path: string) => {
    setFolderChecks((p) => ({ ...p, [path]: !p[path] }));
  };

  const renderTree = (node: TreeNode, parentPath: string) => {
    if (node.type !== "dir") return null;

    const folders = (node.children || []).filter((c) => c.type === "dir");

    return (
      <div className="db-tree">
        {folders.map((ch) => {
          const chPath = joinPath(parentPath, ch.name);
          const open = isExpanded(chPath);

          const selfMatch = containsText(ch.name, treeSearch) || containsText(chPath, treeSearch);
          const childMatch =
            !treeSearch.trim()
              ? true
              : (ch.children || []).some((x) => containsText(x.name, treeSearch));
          const visible = selfMatch || childMatch;

          if (!visible) return null;

          const isSel = selected?.kind === "dir" && selected?.path === chPath;
          const checked = !!folderChecks[chPath];

          return (
            <div key={chPath} className="db-tree-rowwrap">
              <div
                className={`db-tree-row ${isSel ? "is-selected" : ""}`}
                onClick={() => setSelected({ kind: "dir", path: chPath })}
                onContextMenu={(e) =>
                  openCtxMenu(e, { kind: "dir", path: chPath, name: ch.name })
                }
                title={chPath}
              >
                <button
                  className="db-tree-toggle"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggle(chPath);
                  }}
                  title={open ? "Collapse" : "Expand"}
                >
                  {open ? "▾" : "▸"}
                </button>

                <div className="db-tree-main">
                  <div className="db-tree-name">{ch.name}</div>
                  <div className="db-tree-path">{chPath}</div>
                </div>

                <label
                  className="db-tree-include"
                  onClick={(e) => e.stopPropagation()}
                  title="Include this folder when building the active database"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      setFolderChecks((p) => ({ ...p, [chPath]: e.target.checked }));
                    }}
                  />
                  Include
                </label>
              </div>

              {open && (
                <div className="db-tree-children">
                  {renderTree(ch, chPath)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // ---------- Breadcrumbs ----------
  const crumbs = useMemo(() => {
    const parts = splitPath(selectedDir);
    const out: { label: string; path: string }[] = [{ label: "Documents", path: "" }];
    let cur = "";
    for (const p of parts) {
      cur = cur ? `${cur}/${p}` : p;
      out.push({ label: p, path: cur });
    }
    return out;
  }, [selectedDir]);

  const includeCount = useMemo(
    () => Object.values(folderChecks).filter(Boolean).length,
    [folderChecks]
  );

  // ---------- Documents actions ----------
  const doMkdir = async () => {
    const name = newFolderName.trim();
    if (!name) return;

    const base = selectedDir;
    const path = base ? `${base}/${name}` : name;

    setBusy("mkdir");
    setStatus("Creating folder…");
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_BASE}/api/documents/mkdir`, {
        method: "POST",
        headers,
        body: JSON.stringify({ path }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "mkdir failed");

      setStatus(`✅ Created folder: ${path}`);
      setNewFolderName("");
      await refreshTree();

      setSelected({ kind: "dir", path });
      setExpanded((p) => ({ ...p, [base]: true, [path]: true }));
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  const doUpload = async () => {
    if (!selectedFiles.length) return;

    setBusy("upload");
    setStatus("Uploading…");

    try {
      const fd = new FormData();
      for (const f of selectedFiles) fd.append("files", f);

      const url = new URL(`${API_BASE}/api/documents/upload`);
      if (selectedDir) url.searchParams.set("path", selectedDir);

      const res = await fetch(url.toString(), {
        method: "POST",
        headers: authHeaders,
        body: fd,
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Upload failed");

      setStatus(`✅ Uploaded ${selectedFiles.length} file(s) into "${selectedDir || "Documents"}"`);
      setFiles(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refreshTree();
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  const doRename = async () => {
    if (!selected) return;
    const newName = renameValue.trim();
    if (!newName) return;

    const src = selected.path;
    const parent = dirname(src);
    const dst = parent ? `${parent}/${newName}` : newName;

    if (dst === src) return;

    setBusy("move");
    setStatus("Renaming…");
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_BASE}/api/documents/move`, {
        method: "POST",
        headers,
        body: JSON.stringify({ src, dst }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Rename failed");

      setStatus(`✅ Renamed: ${src} → ${dst}`);
      await refreshTree();
      setSelected({ kind: selected.kind, path: dst });
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  const doMoveToFolder = async () => {
    if (!selected) return;
    const target = moveTargetDir; // "" Documents ok
    const src = selected.path;
    const dst = target ? `${target}/${basename(src)}` : basename(src);
    if (dst === src) return;

    setBusy("move");
    setStatus("Moving…");
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_BASE}/api/documents/move`, {
        method: "POST",
        headers,
        body: JSON.stringify({ src, dst }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Move failed");

      setStatus(`✅ Moved: ${src} → ${dst}`);
      await refreshTree();
      setSelected({ kind: selected.kind, path: dst });
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  const doDelete = async () => {
    if (!selected) return;

    setBusy("delete");
    setStatus("Deleting…");
    try {
      const url = new URL(`${API_BASE}/api/documents/delete`);
      url.searchParams.set("path", selected.path);

      const res = await fetch(url.toString(), {
        method: "DELETE",
        headers: authHeaders,
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Delete failed");

      setStatus(`✅ Deleted: ${selected.path}`);
      setDeleteOpen(false);

      const parent = dirname(selected.path);
      setSelected({ kind: "dir", path: parent });
      await refreshTree();
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  // ---------- DB actions ----------
  const loadDbStats = async (name: string) => {
    if (!name) return;
    setBusy("db-stats");
    try {
      const res = await fetch(
        `${API_BASE}/api/databases/${encodeURIComponent(name)}/stats`,
        { headers: authHeaders }
      );
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Stats failed");
      setDbStats(data);
    } catch {
      setDbStats(null);
    } finally {
      setBusy("");
    }
  };

  useEffect(() => {
    if (activeDb) loadDbStats(activeDb);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDb]);

  const doCreateDb = async () => {
    const name = newDbName.trim();
    if (!name) {
      setStatus("❌ Enter a database name first.");
      return;
    }

    const folders = Object.entries(folderChecks)
      .filter(([, v]) => v)
      .map(([k]) => k);

    setBusy("db-create");
    setStatus("Creating database…");
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_BASE}/api/databases/create`, {
        method: "POST",
        headers,
        body: JSON.stringify({ name, folders }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Create DB failed");

      setStatus(`✅ Created DB: ${name}`);
      setActiveDb(name);
      setNewDbName("");
      await refreshDatabases();
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  const doBuildDb = async () => {
    if (!activeDb) {
      setStatus("❌ Choose a database first.");
      return;
    }

    const folders = Object.entries(folderChecks)
      .filter(([, v]) => v)
      .map(([k]) => k);

    if (!folders.length) {
      setStatus("❌ Select at least one folder to build from.");
      return;
    }

    setBusy("db-build");
    setStatus("Building database (can take a bit)…");
    try {
      const headers = new Headers(authHeaders);
      headers.set("Content-Type", "application/json");

      const res = await fetch(`${API_BASE}/api/databases/build`, {
        method: "POST",
        headers,
        body: JSON.stringify({ name: activeDb, folders, force: true }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail || "Build failed");

      setStatus(
        `✅ Built "${activeDb}". Files: ${data?.files_found ?? "?"}, chunks: ${data?.inserted_chunks ?? "?"}, skipped: ${data?.skipped_files ?? "?"}`
      );
      await loadDbStats(activeDb);
    } catch (e: any) {
      setStatus(`❌ ${e?.message || String(e)}`);
    } finally {
      setBusy("");
    }
  };

  // ---------- Upload dropzone ----------
  const onDropFiles = (fileList: FileList) => {
    const dt = new DataTransfer();
    for (const f of selectedFiles) dt.items.add(f);
    for (const f of Array.from(fileList)) dt.items.add(f);
    setFiles(dt.files);
  };

  const selectedLabel = selected
    ? selected.path
      ? `${selected.kind.toUpperCase()}: ${selected.path}`
      : `${selected.kind.toUpperCase()}: (Documents)`
    : "None";

  // ---------- Context menu actions ----------
  const ctxTarget = ctx.open ? ctx.target : null;

  const ctxOpenFolder = () => {
    if (!ctxTarget || ctxTarget.kind !== "dir") return;
    setSelected({ kind: "dir", path: ctxTarget.path });
    setExpanded((p) => ({ ...p, [ctxTarget.path]: true }));
    setCtx({ open: false, x: 0, y: 0, target: null });
  };

  const ctxRename = () => {
    if (!ctxTarget) return;
    setSelectedFromTarget(ctxTarget);
    setRenameValue(ctxTarget.name);
    setCtx({ open: false, x: 0, y: 0, target: null });

    setTimeout(() => {
      renameRef.current?.focus();
      renameRef.current?.select();
    }, 50);
  };

  const ctxMove = () => {
    if (!ctxTarget) return;
    setSelectedFromTarget(ctxTarget);
    setCtx({ open: false, x: 0, y: 0, target: null });

    setTimeout(() => {
      moveRef.current?.focus();
    }, 50);
  };

  const ctxDelete = () => {
    if (!ctxTarget) return;
    setSelectedFromTarget(ctxTarget);
    setCtx({ open: false, x: 0, y: 0, target: null });
    setTimeout(() => setDeleteOpen(true), 0);
  };

  const ctxToggleInclude = () => {
    if (!ctxTarget || ctxTarget.kind !== "dir") return;
    toggleInclude(ctxTarget.path);
    setCtx({ open: false, x: 0, y: 0, target: null });
  };

  return (
    <div className="page-shell">
      <div className="page-wrap">
        <div className="page-header">
          <div>
            <h2 className="page-title">Database</h2>
            <div className="page-subtitle">
              Manage documents → select folders → build named databases
            </div>
          </div>

          <div className="badge" title="Role">
            Role:
            <span className="db-role-pill">
              {isAdmin ? "ADMIN" : isTA ? "TA" : "STUDENT"}
            </span>
          </div>
        </div>

        <div className="db-grid">
          {/* LEFT: Folder Tree */}
          <div className="card card-pad db-panel">
            <div className="db-panel-head">
              <div>
                <div className="db-panel-title">Folders</div>
                <div className="db-panel-sub">Right-click folders for actions</div>
              </div>

              <button className="btn" disabled={busy !== ""} onClick={refreshTree}>
                {busy === "tree" ? "Refreshing…" : "Refresh"}
              </button>
            </div>

            <div className="db-search">
              <input
                value={treeSearch}
                onChange={(e) => setTreeSearch(e.target.value)}
                placeholder="Search folders…"
                className="db-input"
              />
              <div className="db-mini">
                Included: <b>{includeCount}</b>
              </div>
            </div>

            <div className="db-scroll">
              {/* Documents root */}
              <div
                className={`db-tree-row ${
                  selected?.kind === "dir" && selected?.path === "" ? "is-selected" : ""
                }`}
                onClick={() => setSelected({ kind: "dir", path: "" })}
                onContextMenu={(e) =>
                  openCtxMenu(e, { kind: "dir", path: "", name: "Documents" })
                }
                title="Documents"
              >
                <button
                  className="db-tree-toggle"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggle("");
                  }}
                  title={isExpanded("") ? "Collapse" : "Expand"}
                >
                  {isExpanded("") ? "▾" : "▸"}
                </button>

                <div className="db-tree-main">
                  <div className="db-tree-name">Documents</div>
                  <div className="db-tree-path">(top level)</div>
                </div>
              </div>

              {tree ? (isExpanded("") ? renderTree(tree, "") : null) : <div className="muted">No documents yet.</div>}
            </div>
          </div>

          {/* MIDDLE: Contents + Actions */}
          <div className="card card-pad db-panel">
            <div className="db-panel-head">
              <div>
                <div className="db-panel-title">Files</div>
                <div className="db-panel-sub">
                  Selected: <span className="db-mono">{selectedLabel}</span>
                </div>
              </div>

              <div className="db-actions">
                <button
                  className="btn"
                  onClick={() => setDeleteOpen(true)}
                  disabled={busy !== "" || !selected || selected.path === ""}
                  title="Delete selected item"
                >
                  Delete
                </button>
              </div>
            </div>

            {/* Breadcrumbs + quick create folder */}
            <div className="db-breadcrumb-row">
              <div className="db-breadcrumbs" aria-label="Breadcrumb">
                {crumbs.map((c, idx) => (
                  <button
                    key={c.path || "Documents"}
                    className="db-crumb"
                    onClick={() => setSelected({ kind: "dir", path: c.path })}
                    title={c.path || "Documents"}
                  >
                    {c.label}
                    {idx < crumbs.length - 1 ? <span className="db-crumb-sep">/</span> : null}
                  </button>
                ))}
              </div>

              <div className="db-inline">
                <input
                  value={newFolderName}
                  onChange={(e) => setNewFolderName(e.target.value)}
                  placeholder="New folder…"
                  className="db-input"
                  disabled={busy !== "" || !selected}
                />
                <button
                  className="btn btn-primary"
                  onClick={doMkdir}
                  disabled={busy !== "" || !newFolderName.trim()}
                >
                  {busy === "mkdir" ? "Creating…" : "Create"}
                </button>
              </div>
            </div>

            {/* Upload */}
            <div
              className={`db-dropzone ${busy === "upload" ? "is-busy" : ""}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files?.length) onDropFiles(e.dataTransfer.files);
              }}
            >
              <div className="db-dropzone-top">
                <div className="db-drop-left">
                  <div className="db-drop-title">
                    Upload to <span className="db-mono">{selectedDir || "Documents"}</span>
                  </div>
                  <div className="db-drop-sub">Drag & drop files here, or browse.</div>
                </div>

                <div className="db-drop-actions">
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={(e) => setFiles(e.target.files)}
                    className="db-hidden-input"
                  />
                  <button className="btn" onClick={() => fileInputRef.current?.click()} disabled={busy !== ""}>
                    Browse
                  </button>
                  <button className="btn btn-primary" onClick={doUpload} disabled={busy !== "" || selectedFiles.length === 0}>
                    {busy === "upload" ? "Uploading…" : `Upload (${selectedFiles.length || 0})`}
                  </button>
                </div>
              </div>

              {selectedFiles.length > 0 && (
                <div className="db-filechips">
                  {selectedFiles.slice(0, 8).map((f) => (
                    <div key={f.name + f.size} className="db-chip" title={`${f.name} (${f.size} bytes)`}>
                      {f.name}
                    </div>
                  ))}
                  {selectedFiles.length > 8 && (
                    <div className="db-chip db-chip-muted">+{selectedFiles.length - 8} more</div>
                  )}
                  <button
                    className="db-link"
                    onClick={() => {
                      setFiles(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>

            {/* Rename / Move */}
            <div className="db-two">
              <div className="db-box">
                <div className="db-box-title">Rename</div>
                <div className="db-row">
                  <input
                    ref={renameRef}
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    placeholder="New name…"
                    className="db-input"
                    disabled={busy !== "" || !selected || selected.path === ""}
                  />
                  <button
                    className="btn"
                    onClick={doRename}
                    disabled={
                      busy !== "" ||
                      !selected ||
                      selected.path === "" ||
                      !renameValue.trim() ||
                      renameValue.trim() === basename(selected.path)
                    }
                  >
                    {busy === "move" ? "Saving…" : "Save"}
                  </button>
                </div>
              </div>

              <div className="db-box">
                <div className="db-box-title">Move to folder</div>
                <div className="db-row">
                  <select
                    ref={moveRef}
                    value={moveTargetDir}
                    onChange={(e) => setMoveTargetDir(e.target.value)}
                    className="db-input"
                    disabled={busy !== "" || !selected || selected.path === ""}
                  >
                    <option value="">Documents</option>
                    {allFolders
                      .filter((p) => p !== "")
                      .map((p) => (
                        <option key={p} value={p}>
                          {p}
                        </option>
                      ))}
                  </select>
                  <button className="btn" onClick={doMoveToFolder} disabled={busy !== "" || !selected || selected.path === ""}>
                    {busy === "move" ? "Moving…" : "Move"}
                  </button>
                </div>
              </div>
            </div>

            {/* Folder contents */}
            <div className="db-contents-head">
              <input
                value={contentsSearch}
                onChange={(e) => setContentsSearch(e.target.value)}
                placeholder="Search this folder…"
                className="db-input"
              />
              <div className="db-mini">
                In folder: <b>{dirContents.length}</b>
              </div>
            </div>

            <div className="db-scroll db-contents">
              {dirContents.length === 0 ? (
                <div className="muted">Nothing here.</div>
              ) : (
                dirContents.map((n) => {
                  const p = joinPath(selectedDir, n.name);
                  const isSel = selected?.path === p && selected?.kind === n.type;

                  return (
                    <div
                      key={p}
                      className={`db-item ${isSel ? "is-selected" : ""}`}
                      onClick={() =>
                        setSelected({
                          kind: n.type === "dir" ? "dir" : "file",
                          path: p,
                        })
                      }
                      onContextMenu={(e) =>
                        openCtxMenu(e, {
                          kind: n.type === "dir" ? "dir" : "file",
                          path: p,
                          name: n.name,
                        })
                      }
                      title={p}
                    >
                      <div className="db-item-ic">{n.type === "dir" ? "📁" : "📄"}</div>
                      <div className="db-item-main">
                        <div className="db-item-name">{n.name}</div>
                        <div className="db-item-path">{p}</div>
                      </div>

                      {n.type === "dir" ? (
                        <button
                          className="db-link"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelected({ kind: "dir", path: p });
                            setExpanded((st) => ({ ...st, [p]: true }));
                          }}
                        >
                          Open
                        </button>
                      ) : null}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* RIGHT: Databases */}
          <div className="card card-pad db-panel">
            <div className="db-panel-head">
              <div>
                <div className="db-panel-title">Databases</div>
                <div className="db-panel-sub">Create/build from included folders</div>
              </div>

              <button className="btn" disabled={busy !== ""} onClick={refreshDatabases}>
                Refresh
              </button>
            </div>

            <div className="db-box">
              <div className="db-box-title">Active database</div>
              <div className="db-row">
                <select
                  value={activeDb}
                  onChange={(e) => setActiveDb(e.target.value)}
                  className="db-input"
                  disabled={dbList.length === 0}
                >
                  {dbList.length === 0 ? (
                    <option value="">No databases yet</option>
                  ) : (
                    dbList.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div className="db-mini">
                Included folders: <b>{includeCount}</b>
              </div>
            </div>

            <div className="db-box">
              <div className="db-box-title">Create new DB</div>
              <div className="db-row">
                <input
                  value={newDbName}
                  onChange={(e) => setNewDbName(e.target.value)}
                  placeholder="db name (e.g., ecen214)"
                  className="db-input"
                />
                <button className="btn btn-primary" disabled={busy !== ""} onClick={doCreateDb}>
                  {busy === "db-create" ? "Creating…" : "Create"}
                </button>
              </div>
            </div>

            <div className="db-box">
              <div className="db-box-title">Build / Rebuild</div>
              <button className="btn btn-primary" disabled={busy !== "" || !activeDb} onClick={doBuildDb} style={{ width: "100%" }}>
                {busy === "db-build" ? "Building…" : `Build "${activeDb || "DB"}"`}
              </button>
              <div className="db-mini">Builds from included folders (force </div>
              <div className="db-mini">rebuild).</div>
            </div>

            {dbStats && (
              <div className="db-box">
                <div className="db-box-title">Stats</div>
                <div className="db-stat">chunks: <b>{humanCount(dbStats?.stats?.chunk_count ?? 0)}</b></div>
                <div className="db-stat">
                  file: <span className="db-mono db-wrap">{dbStats?.stats?.vdb_path || "-"}</span>
                </div>
                <div className="db-stat">
                  model: <b>{dbStats?.config?.llm_model || "-"}</b> · embed: <b>{dbStats?.config?.embed_model || "-"}</b>
                </div>
              </div>
            )}

            <div className="status-box mono" style={{ fontSize: 13, opacity: 0.95, marginTop: 14 }}>
              {status ? `> ${status}` : "> Idle"}
            </div>
          </div>
        </div>

        {/* Context Menu */}
        {ctx.open && ctx.target && (
          <div
            className="db-ctx"
            style={{ left: ctx.x, top: ctx.y }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="db-ctx-title" title={ctx.target.path || "Documents"}>
              {ctx.target.name}
            </div>

            {ctx.target.kind === "dir" && (
              <button className="db-ctx-item" onClick={ctxOpenFolder}>
                Open
              </button>
            )}

            {ctx.target.kind === "dir" && ctx.target.path !== "" && (
              <button className="db-ctx-item" onClick={ctxToggleInclude}>
                {folderChecks[ctx.target.path] ? "Uninclude from DB build" : "Include in DB build"}
              </button>
            )}

            {ctx.target.path !== "" && (
              <>
                <button className="db-ctx-item" onClick={ctxRename}>
                  Rename…
                </button>
                <button className="db-ctx-item" onClick={ctxMove}>
                  Move…
                </button>
                <div className="db-ctx-sep" />
                <button className="db-ctx-item danger" onClick={ctxDelete}>
                  Delete…
                </button>
              </>
            )}
          </div>
        )}

        {/* Delete Modal */}
        {deleteOpen && selected && selected.path !== "" && (
          <div className="db-modal-overlay" onClick={() => setDeleteOpen(false)}>
            <div className="card db-modal" onClick={(e) => e.stopPropagation()}>
              <div className="db-modal-title">Delete</div>
              <div className="muted" style={{ fontSize: 13 }}>
                Are you sure you want to delete:
              </div>

              <div className="db-modal-path">{selected.path}</div>

              <div className="db-modal-actions">
                <button className="btn" disabled={busy !== ""} onClick={() => setDeleteOpen(false)}>
                  Cancel
                </button>
                <button className="btn btn-primary" disabled={busy !== ""} onClick={doDelete}>
                  {busy === "delete" ? "Deleting…" : "Delete"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}