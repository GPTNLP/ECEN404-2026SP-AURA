import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import AppRouter from "./router/AppRouter";
import { AuthProvider } from "./services/authService";
import { LoadedDbProvider } from "./context/LoadedDbContext";
import "./styles/index.css";
import "./styles/chatlogs.css";
import { loadTheme, applyTheme } from "./services/themeStore";

applyTheme(loadTheme());

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <LoadedDbProvider>
          <AppRouter />
        </LoadedDbProvider>
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>
);