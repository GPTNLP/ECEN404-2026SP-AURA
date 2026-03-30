import { useState } from "react";
import "../styles/controlPage.css";

// Added pitch and yaw to the allowed commands
type MoveCmd = "forward" | "backward" | "left" | "right" | "stop" | "pitch" | "yaw";

const API_BASE = "https://aura-backend-fmfyemepbybgebcs.eastus-01.azurewebsites.net";
const DEVICE_ID = "jetson-001";
const LS_TOKEN = "aura-auth-token";

async function sendMove(command: MoveCmd, value?: number) {
  const token = localStorage.getItem(LS_TOKEN);

  // Construct the body. If a value is provided (for pitch/yaw), wrap it in the payload object.
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
  // State for the sliders
  const [pitch, setPitch] = useState<number>(0);
  const [yaw, setYaw] = useState<number>(0);

  // Helper to completely reset the robot's stance and zero out the sliders
  const handleStopAndReset = () => {
    setPitch(0);
    setYaw(0);
    sendMove("stop"); // The ESP32 code resets to BASE_ANGLE on 'stop'
  };

  return (
    <div className="page">
      <div className="control-header">
        <h1>Robot Control</h1>
        <p className="control-subtitle">Command the robot's movement and stance.</p>
      </div>

      <div className="control-grid" style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
        
        {/* Locomotion D-Pad (Existing) */}
        <section className="control-card" style={{ flex: '1 1 300px' }}>
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
                onClick={handleStopAndReset}
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

        {/* New Stance Control Section */}
        <section className="control-card" style={{ flex: '1 1 300px' }}>
          <h2>Stance Control</h2>
          <div className="control-divider" />

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1rem' }}>
            
            {/* Pitch Control */}
            <div className="slider-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <label style={{ fontWeight: 'bold' }}>Pitch (Tilt)</label>
                <span>{pitch}°</span>
              </div>
              <input 
                type="range" 
                min="-45" 
                max="45" 
                value={pitch} 
                onChange={(e) => setPitch(parseInt(e.target.value))}
                style={{ width: '100%', marginBottom: '0.5rem' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#666' }}>
                <span>Backward</span>
                <span>Forward</span>
              </div>
              <button 
                onClick={() => sendMove("pitch", pitch)}
                style={{ marginTop: '0.5rem', width: '100%', padding: '0.5rem', cursor: 'pointer' }}
              >
                Apply Pitch
              </button>
            </div>

            {/* Yaw Control */}
            <div className="slider-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <label style={{ fontWeight: 'bold' }}>Yaw (Pivot)</label>
                <span>{yaw}°</span>
              </div>
              <input 
                type="range" 
                min="-45" 
                max="45" 
                value={yaw} 
                onChange={(e) => setYaw(parseInt(e.target.value))}
                style={{ width: '100%', marginBottom: '0.5rem' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#666' }}>
                <span>Left</span>
                <span>Right</span>
              </div>
              <button 
                onClick={() => sendMove("yaw", yaw)}
                style={{ marginTop: '0.5rem', width: '100%', padding: '0.5rem', cursor: 'pointer' }}
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