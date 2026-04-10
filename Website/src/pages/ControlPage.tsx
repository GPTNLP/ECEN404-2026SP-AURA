import { useCallback, useRef, useState } from "react";
import "../styles/controlPage.css";

type MoveCmd =
  | "forward"
  | "backward"
  | "left"
  | "right"
  | "stop"
  | "pitch"
  | "yaw";

const API_BASE =
  "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";
const DEVICE_ID = "jetson-001";
const LS_TOKEN = "aura-auth-token";

async function sendMove(command: MoveCmd, value?: number) {
  const token = localStorage.getItem(LS_TOKEN);

  const bodyData: any = {
    device_id: DEVICE_ID,
    command,
  };

  if (value !== undefined) {
    bodyData.payload = { value };
  }

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
  const [pitch, setPitch] = useState<number>(0);
  const [yaw, setYaw] = useState<number>(0);
  const activeMoveRef = useRef<MoveCmd | null>(null);

  const startMove = useCallback((cmd: MoveCmd) => {
    if (cmd === "stop" || cmd === "pitch" || cmd === "yaw") return;
    activeMoveRef.current = cmd;
    sendMove(cmd);
  }, []);

  const handleStopAndReset = useCallback(() => {
    activeMoveRef.current = null;
    setPitch(0);
    setYaw(0);
    sendMove("stop");
  }, []);

  const bindMoveButton = (cmd: MoveCmd) => ({
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
        <p className="control-subtitle">Command the robot&apos;s movement and stance.</p>
      </div>

      <div className="control-grid" style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>
        <section className="control-card" style={{ flex: "1 1 300px" }}>
          <h2>Movement</h2>
          <div className="control-divider" />

          <div className="dpad-wrap">
            <div className="dpad">
              <button
                className="dpad-btn up"
                {...bindMoveButton("forward")}
                aria-label="Move forward"
              >
                <span>▲</span>
              </button>

              <button
                className="dpad-btn left"
                {...bindMoveButton("left")}
                aria-label="Move left"
              >
                <span>◀</span>
              </button>

              <button
                className="stop-btn"
                onClick={handleStopAndReset}
                aria-label="Stop all"
              >
                STOP
              </button>

              <button
                className="dpad-btn right"
                {...bindMoveButton("right")}
                aria-label="Move right"
              >
                <span>▶</span>
              </button>

              <button
                className="dpad-btn down"
                {...bindMoveButton("backward")}
                aria-label="Move backward"
              >
                <span>▼</span>
              </button>
            </div>
          </div>
        </section>

        <section className="control-card" style={{ flex: "1 1 300px" }}>
          <h2>Stance Control</h2>
          <div className="control-divider" />

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "1.5rem",
              marginTop: "1rem",
            }}
          >
            <div className="slider-group">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "0.5rem",
                }}
              >
                <label style={{ fontWeight: "bold" }}>Pitch (Tilt)</label>
                <span>{pitch}°</span>
              </div>

              <input
                type="range"
                min="-45"
                max="45"
                value={pitch}
                onChange={(e) => setPitch(parseInt(e.target.value, 10))}
                style={{ width: "100%", marginBottom: "0.5rem" }}
              />

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.8rem",
                  color: "#666",
                }}
              >
                <span>Backward</span>
                <span>Forward</span>
              </div>

              <button
                onClick={() => sendMove("pitch", pitch)}
                style={{
                  marginTop: "0.5rem",
                  width: "100%",
                  padding: "0.5rem",
                  cursor: "pointer",
                }}
              >
                Apply Pitch
              </button>
            </div>

            <div className="slider-group">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "0.5rem",
                }}
              >
                <label style={{ fontWeight: "bold" }}>Yaw (Pivot)</label>
                <span>{yaw}°</span>
              </div>

              <input
                type="range"
                min="-45"
                max="45"
                value={yaw}
                onChange={(e) => setYaw(parseInt(e.target.value, 10))}
                style={{ width: "100%", marginBottom: "0.5rem" }}
              />

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.8rem",
                  color: "#666",
                }}
              >
                <span>Left</span>
                <span>Right</span>
              </div>

              <button
                onClick={() => sendMove("yaw", yaw)}
                style={{
                  marginTop: "0.5rem",
                  width: "100%",
                  padding: "0.5rem",
                  cursor: "pointer",
                }}
              >
                Apply Yaw
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}