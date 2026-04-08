import { useEffect, useState } from "react";
import ThemeSelector from "../components/UI/ThemeSelector";
import { loadPrefs, savePrefs } from "../services/prefs";
import "../styles/settings.css";

export default function SettingsPage() {
  const [voiceInputEnabled, setVoiceInputEnabled] = useState(true);
  const [speakAiEnabled, setSpeakAiEnabled] = useState(false);

  useEffect(() => {
    const p = loadPrefs();
    setVoiceInputEnabled(p.voiceInputEnabled);
    setSpeakAiEnabled(p.speakAiEnabled);
  }, []);

  useEffect(() => {
    savePrefs({ voiceInputEnabled, speakAiEnabled });
  }, [voiceInputEnabled, speakAiEnabled]);

  return (
    <div className="settings-container">
      <h2>Settings</h2>

      <div className="settings-section">
        <h3>Appearance</h3>
        <ThemeSelector />
      </div>

      <div className="settings-section">
        <h3>System Preferences</h3>

        <div
          style={{
            display: "grid",
            gap: 12,
            maxWidth: 560,
          }}
        >
          <label
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 14,
              padding: "12px 14px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: "var(--card-bg)",
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 900 }}>Voice Input</div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>
                Use the mic button in Simulator to transcribe speech into the prompt box.
              </div>
            </div>

            <input
              type="checkbox"
              checked={voiceInputEnabled}
              onChange={(e) => setVoiceInputEnabled(e.target.checked)}
            />
          </label>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 14,
              padding: "12px 14px",
              borderRadius: 12,
              border: "1px solid var(--card-border)",
              background: "var(--card-bg)",
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 900 }}>Speak AI Responses</div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>
                When enabled, the Jetson will speak the AI response out loud.
              </div>
            </div>

            <input
              type="checkbox"
              checked={speakAiEnabled}
              onChange={(e) => setSpeakAiEnabled(e.target.checked)}
            />
          </label>

          <div style={{ fontSize: 12, opacity: 0.75 }}>
            Note: Voice features require backend <code>/api/stt/transcribe</code> and{" "}
            <code>/api/tts/speak</code>.
          </div>
        </div>
      </div>
    </div>
  );
}