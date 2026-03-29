import "../styles/controlPage.css";

type MoveCmd = "forward" | "backward" | "left" | "right" | "stop";

const API_BASE = "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";
const DEVICE_ID = "jetson-001";
const LS_TOKEN = "aura-auth-token";

async function sendMove(command: MoveCmd) {
  const token = localStorage.getItem(LS_TOKEN);

  try {
    const res = await fetch(`${API_BASE}/device/admin/command`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        device_id: DEVICE_ID,
        command,
      }),
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
  return (
    <div className="page">
      <div className="control-header">
        <h1>Robot Control</h1>
        <p className="control-subtitle">Use the D-pad to command the robot.</p>
      </div>

      <div className="control-grid">
        <section className="control-card">
          <h2>Movement</h2>
          <div className="control-divider" />

          <div className="dpad-wrap">
            <div className="dpad">
              <button
                className="dpad-btn up"
                onClick={() => sendMove("forward")}
                aria-label="Move forward"
              >
                <span>▲</span>
              </button>

              <button
                className="dpad-btn left"
                onClick={() => sendMove("left")}
                aria-label="Move left"
              >
                <span>◀</span>
              </button>

              <button
                className="stop-btn"
                onClick={() => sendMove("stop")}
                aria-label="Stop all"
              >
                STOP
              </button>

              <button
                className="dpad-btn right"
                onClick={() => sendMove("right")}
                aria-label="Move right"
              >
                <span>▶</span>
              </button>

              <button
                className="dpad-btn down"
                onClick={() => sendMove("backward")}
                aria-label="Move backward"
              >
                <span>▼</span>
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}