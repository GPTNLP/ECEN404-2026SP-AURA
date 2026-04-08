// Website/src/services/prefs.ts
export type AuraPrefs = {
  voiceInputEnabled: boolean;
  speakAiEnabled: boolean;
};

const KEY = "aura-prefs";

export function loadPrefs(): AuraPrefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) {
      return { voiceInputEnabled: true, speakAiEnabled: false };
    }
    const j = JSON.parse(raw);
    return {
      voiceInputEnabled: !!j.voiceInputEnabled,
      speakAiEnabled: !!j.speakAiEnabled,
    };
  } catch {
    return { voiceInputEnabled: true, speakAiEnabled: false };
  }
}

export function savePrefs(p: AuraPrefs) {
  localStorage.setItem(KEY, JSON.stringify(p));
}