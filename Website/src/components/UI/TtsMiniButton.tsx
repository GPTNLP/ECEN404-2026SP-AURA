import { useEffect, useMemo, useState } from "react";

export default function TtsMiniButton({
  textToSpeak,
  size = 30,
}: {
  textToSpeak: string;
  size?: number;
}) {
  const supported = useMemo(() => "speechSynthesis" in window, []);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    if (!supported) return;
    return () => {
      window.speechSynthesis?.cancel();
    };
  }, [supported]);

  const speak = () => {
    if (!supported) return;
    const t = (textToSpeak || "").trim();
    if (!t) return;

    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(t);
    u.rate = 1.0;
    u.pitch = 1.0;
    u.volume = 1.0;
    window.speechSynthesis.speak(u);
  };

  return (
    <button
      type="button"
      title={supported ? "Text to Speech" : "TTS not supported"}
      onClick={() => {
        if (!supported) return;
        const next = !enabled;
        setEnabled(next);
        if (next) speak();
        else window.speechSynthesis.cancel();
      }}
      style={{
        width: size,
        height: size,
        borderRadius: 10,
        border: "1px solid rgba(0,0,0,0.15)",
        background: enabled ? "rgba(0,0,0,0.10)" : "rgba(0,0,0,0.04)",
        cursor: supported ? "pointer" : "not-allowed",
        display: "grid",
        placeItems: "center",
        userSelect: "none",
      }}
    >
      {enabled ? "🔊" : "🗣️"}
    </button>
  );
}