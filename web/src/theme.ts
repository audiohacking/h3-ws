/** Theme preference: Auto follows OS; Light/Dark force a palette. */

export type ThemePreference = "auto" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "h3-ws.theme";

export function getStoredTheme(): ThemePreference {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === "light" || raw === "dark" || raw === "auto") return raw;
  } catch {
    /* private mode */
  }
  return "auto";
}

export function systemPrefersDark(): boolean {
  return typeof window !== "undefined"
    && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function resolveTheme(pref: ThemePreference): ResolvedTheme {
  if (pref === "light") return "light";
  if (pref === "dark") return "dark";
  return systemPrefersDark() ? "dark" : "light";
}

export function applyTheme(pref: ThemePreference): ResolvedTheme {
  const effective = resolveTheme(pref);
  const root = document.documentElement;
  root.dataset.theme = effective;
  root.style.colorScheme = effective;
  return effective;
}

export function setThemePreference(pref: ThemePreference): ResolvedTheme {
  try {
    localStorage.setItem(STORAGE_KEY, pref);
  } catch {
    /* private mode */
  }
  return applyTheme(pref);
}

export function cycleTheme(pref: ThemePreference): ThemePreference {
  if (pref === "auto") return "light";
  if (pref === "light") return "dark";
  return "auto";
}

export function themeLabel(pref: ThemePreference): string {
  if (pref === "auto") return "Auto";
  if (pref === "light") return "Light";
  return "Dark";
}

/** Re-apply when OS preference changes (only matters for Auto). */
export function subscribeSystemTheme(onChange: () => void): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const handler = () => onChange();
  mq.addEventListener("change", handler);
  return () => mq.removeEventListener("change", handler);
}

// Avoid a light flash before React mounts.
applyTheme(getStoredTheme());
