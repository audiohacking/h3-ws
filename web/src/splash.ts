/** Inline #app-splash in index.html — dismiss once App boot data is ready. */

const SPLASH_MIN_MS = 1000;
const SPLASH_FADE_MS = 450;

const splashStarted = performance.now();
let dismissed = false;

export function dismissAppSplash(): void {
  if (dismissed) return;
  dismissed = true;
  const el = document.getElementById("app-splash");
  if (!el) return;
  const reduce =
    typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = Math.max(0, SPLASH_MIN_MS - (performance.now() - splashStarted));
  window.setTimeout(() => {
    el.classList.add("is-out");
    window.setTimeout(() => el.remove(), reduce ? 0 : SPLASH_FADE_MS);
  }, wait);
}
