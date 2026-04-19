import { createContext, useContext, useState, useEffect } from "react";
import type { ReactNode } from "react";

const STORAGE_KEY = "aura_loaded_db";

interface LoadedDbContextType {
  loadedDb: string;
  setLoadedDb: (db: string) => void;
}

const LoadedDbContext = createContext<LoadedDbContextType>({
  loadedDb: "",
  setLoadedDb: () => {},
});

export function LoadedDbProvider({ children }: { children: ReactNode }) {
  const [loadedDb, _setLoadedDb] = useState(() => localStorage.getItem(STORAGE_KEY) || "");

  const setLoadedDb = (db: string) => {
    if (db) {
      localStorage.setItem(STORAGE_KEY, db);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
    _setLoadedDb(db);
  };

  // Sync across browser tabs via the native storage event
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        _setLoadedDb(e.newValue || "");
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return (
    <LoadedDbContext.Provider value={{ loadedDb, setLoadedDb }}>
      {children}
    </LoadedDbContext.Provider>
  );
}

export function useLoadedDb() {
  return useContext(LoadedDbContext);
}
