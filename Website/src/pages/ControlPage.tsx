import { useCallback } from "react";
import "../styles/controlPage.css";

type MoveCmd =
  | "forward"
  | "backward"
  | "left"
  | "right"
  | "stop"
  | "left90"
  | "right90"
  | "left180"
  | "right180"
  | "left360"
  | "right360";

const API_BASE =
  "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";
const DEVICE_ID = "jetson-001";
const LS_TOKEN = "aura-auth-token";

const PRESET_TURNS: Array<{
  command: MoveCmd;
  title: string;
  meta: string;
}> = [
  { command: "left90", title: "Left 90", meta: "Quarter turn" },
  { command: "left180", title: "Left 180", meta: "Half turn" },
  { command: "left360", title: "Left 360", meta: "Full turn" },
  { command: "right90", title: "Right 90", meta: "Quarter turn" },
  { command: "right180", title: "Right 180", meta: "Half turn" },
  { command: "right360", title: "Right 360", meta: "Full turn" },
];

async function sendMove(command: MoveCmd) {
  const token = localStorage.getItem(LS_TOKEN);

  const bodyData = {
    device_id: DEVICE_ID,
    command,
  };

  try {
    const res = await fetch(`${API_BASE}/device/admin/command`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(bodyData),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Command failed: ${res.status} ${text}`);
    }

    const data = await res.json();
    console.log("Command queued:", data);
  } catch (err) {
    console.error("Failed to send command:", err);
  }
}

export default function ControlPage() {
  const startMove = useCallback((cmd: Exclude<MoveCmd, "stop">) => {
    void sendMove(cmd);
  }, []);

  const handleHome = useCallback(() => {
    void sendMove("stop");
  }, []);

  const bindMoveButton = (cmd: Exclude<MoveCmd, "stop">) => ({
    onPointerDown: (e: React.PointerEvent<HTMLButtonElement>) => {
      e.preventDefault();
      startMove(cmd);
    },
    onContextMenu: (e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
    },
  });

  return (
    <div className="page">
      <div className="control-header">
        <h1>Robot Control</h1>
      </div>

      <div className="control-grid">
        <section className="control-card movement-card">
          <div className="control-section-head">
            <h2>Movement</h2>
          </div>
          <div className="control-divider" />

          <div className="movement-body">
            <div className="dpad-wrap">
              <div className="dpad">
                <button
                  type="button"
                  className="dpad-btn up"
                  {...bindMoveButton("forward")}
                  aria-label="Move forward"
                >
                  <span>▲</span>
                </button>

                <button
                  type="button"
                  className="dpad-btn left"
                  {...bindMoveButton("left")}
                  aria-label="Move left"
                >
                  <span>◀</span>
                </button>

                <button
                  type="button"
                  className="stop-btn"
                  onClick={handleHome}
                  aria-label="Send home override"
                >
                  HOME
                </button>

                <button
                  type="button"
                  className="dpad-btn right"
                  {...bindMoveButton("right")}
                  aria-label="Move right"
                >
                  <span>▶</span>
                </button>

                <button
                  type="button"
                  className="dpad-btn down"
                  {...bindMoveButton("backward")}
                  aria-label="Move backward"
                >
                  <span>▼</span>
                </button>
              </div>
            </div>
          </div>
        </section>

        <section className="control-card preset-card">
          <div className="control-section-head">
            <h2>Preset Turns</h2>
          </div>
          <div className="control-divider" />

          <div className="preset-grid">
            {PRESET_TURNS.map((preset) => (
              <button
                key={preset.command}
                type="button"
                className="preset-btn"
                onClick={() => {
                  void sendMove(preset.command);
                }}
                aria-label={preset.title}
              >
                <span className="preset-title">{preset.title}</span>
                <span className="preset-meta">{preset.meta}</span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}