import { useEffect, useMemo, useRef, useState } from "react";
import { useDismissable } from "../../hooks/useDismissable";
import { faceCanvasForVideo } from "../../h3Canvas";
import type {
  Clip,
  Config,
  FacesConfigPublic,
  LongTakeConfigPublic,
  RedetailConfigPublic,
  UpframeConfigPublic,
} from "../../types";

export type LibraryPassKind = "redetail" | "upframe" | "longtake" | "faces";

export type LibraryPassParams = {
  multiplier?: number;
  denoise?: number;
  keyframes?: string;
  seed?: number;
  canvas?: number;
  interpolation?: string;
};

/** Composer defaults so the library panel matches the last pill settings. */
export type LibraryPassDefaults = {
  multiplier?: number;
  denoise?: number;
  keyframes?: string;
  seed?: number;
  interpolation?: string;
};

export type LibraryPassPanelProps = {
  clip: Clip;
  kind: LibraryPassKind;
  config: Config | null;
  busy?: boolean;
  /** Live status while the pass runs (player overlay mirrors this). */
  statusMessage?: string | null;
  statusPct?: number | null;
  defaults?: LibraryPassDefaults;
  onClose: () => void;
  onRun: (params: LibraryPassParams) => void;
};

const TITLES: Record<LibraryPassKind, string> = {
  upframe: "Upframe",
  redetail: "ReDetail",
  longtake: "Long take",
  faces: "Faces",
};

function shortEdgeOf(clip: Clip): number {
  const w = Number(clip.width) || 0;
  const h = Number(clip.height) || 0;
  if (w > 0 && h > 0) return Math.min(w, h);
  return 512;
}

function approxRedetailSize(clip: Clip): string {
  const w = Number(clip.width) || 0;
  const h = Number(clip.height) || 0;
  if (w <= 0 || h <= 0) return "×2 canvas";
  const tw = Math.max(32, Math.round((w * 2) / 32) * 32);
  const th = Math.max(32, Math.round((h * 2) / 32) * 32);
  return `${tw}×${th}`;
}

export function LibraryPassPanel({
  clip,
  kind,
  config,
  busy,
  statusMessage,
  statusPct,
  defaults,
  onClose,
  onRun,
}: LibraryPassPanelProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const upframeCfg = config?.upframe as UpframeConfigPublic | undefined;
  const redetailCfg = config?.redetail as RedetailConfigPublic | undefined;
  const longtakeCfg = config?.longtake as LongTakeConfigPublic | undefined;
  const facesCfg = config?.faces as FacesConfigPublic | undefined;

  const [multiplier, setMultiplier] = useState(
    defaults?.multiplier ?? upframeCfg?.multiplier ?? 2,
  );
  const [denoise, setDenoise] = useState(() => {
    if (defaults?.denoise != null) return defaults.denoise;
    if (kind === "faces") return facesCfg?.denoise ?? 0.45;
    if (kind === "longtake") return longtakeCfg?.denoise ?? 0.55;
    return redetailCfg?.denoise ?? 0.45;
  });
  const [keyframes, setKeyframes] = useState(
    defaults?.keyframes ?? longtakeCfg?.keyframes ?? "0:0, 22:0, 44:1",
  );
  const [seed, setSeed] = useState(
    () => defaults?.seed ?? (Number(clip.seed) || 42),
  );
  const [interpolation, setInterpolation] = useState(
    defaults?.interpolation === "smooth"
      ? "smooth"
      : longtakeCfg?.interpolation === "smooth"
        ? "smooth"
        : "linear",
  );

  const faceCanvas = useMemo(
    () => faceCanvasForVideo(shortEdgeOf(clip)),
    [clip.width, clip.height],
  );

  useDismissable(
    true,
    () => {
      if (!busy) onClose();
    },
    rootRef,
  );

  useEffect(() => {
    if (kind === "upframe" && upframeCfg?.multiplier) setMultiplier(upframeCfg.multiplier);
  }, [kind, upframeCfg?.multiplier]);

  const framesHint = useMemo(() => {
    const n = Number(clip.num_frames) || 0;
    if (kind !== "upframe" || n <= 0) return null;
    const out = (n - 1) * multiplier + 1;
    return `${n} → ~${out} frames @ 24 fps`;
  }, [clip.num_frames, kind, multiplier]);

  function submit() {
    if (busy) return;
    if (kind === "upframe") {
      onRun({ multiplier: Math.max(2, Math.round(multiplier)) });
      return;
    }
    if (kind === "redetail") {
      onRun({ denoise, seed });
      return;
    }
    if (kind === "longtake") {
      onRun({ denoise, keyframes, seed, interpolation });
      return;
    }
    onRun({ denoise, seed, canvas: faceCanvas });
  }

  const denoiseMin =
    kind === "faces"
      ? (facesCfg?.denoise_min ?? 0.1)
      : kind === "longtake"
        ? (longtakeCfg?.denoise_min ?? 0.1)
        : (redetailCfg?.denoise_min ?? 0.1);
  const denoiseMax =
    kind === "faces"
      ? (facesCfg?.denoise_max ?? 0.9)
      : kind === "longtake"
        ? (longtakeCfg?.denoise_max ?? 0.9)
        : (redetailCfg?.denoise_max ?? 0.9);

  return (
    <div
      ref={rootRef}
      className="library-pass-panel"
      role="dialog"
      aria-label={`${TITLES[kind]} settings`}
    >
      <div className="library-pass-panel__head">
        <div>
          <strong>{TITLES[kind]}</strong>
          <p className="library-pass-panel__sub">
            New library clip · original kept
          </p>
        </div>
        <button
          type="button"
          className="library-pass-panel__close"
          disabled={busy}
          onClick={onClose}
          aria-label="Close"
        >
          ×
        </button>
      </div>

      <div className="library-pass-panel__fields">
        {kind === "upframe" && (
          <>
            <label>
              Frame multiplier
              <select
                value={multiplier}
                disabled={busy}
                onChange={(e) => setMultiplier(Number(e.target.value))}
              >
                <option value={2}>×2 (insert mid-frames)</option>
                <option value={3}>×3</option>
                <option value={4}>×4</option>
              </select>
            </label>
            {framesHint ? (
              <p className="library-pass-panel__hint">{framesHint}</p>
            ) : (
              <p className="library-pass-panel__hint">
                FILM temporal interpolate · needs film_net_fp16
              </p>
            )}
          </>
        )}

        {kind === "redetail" && (
          <>
            <div className="library-pass-panel__row">
              <span>Output</span>
              <span className="library-pass-panel__value">{approxRedetailSize(clip)}</span>
            </div>
            <label>
              Denoise
              <input
                type="number"
                min={denoiseMin}
                max={denoiseMax}
                step={0.05}
                value={denoise}
                disabled={busy}
                onChange={(e) => setDenoise(Number(e.target.value))}
              />
            </label>
            <label>
              Seed
              <input
                type="number"
                min={0}
                value={seed}
                disabled={busy}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <p className="library-pass-panel__hint">
              ×2 H3 init-video refine. Heavy — uses the GPU.
            </p>
          </>
        )}

        {kind === "longtake" && (
          <>
            <label>
              Denoise
              <input
                type="number"
                min={denoiseMin}
                max={denoiseMax}
                step={0.05}
                value={denoise}
                disabled={busy}
                onChange={(e) => setDenoise(Number(e.target.value))}
              />
            </label>
            <label>
              Keyframes
              <input
                type="text"
                value={keyframes}
                disabled={busy}
                onChange={(e) => setKeyframes(e.target.value)}
                spellCheck={false}
              />
            </label>
            <label>
              Seed
              <input
                type="number"
                min={0}
                value={seed}
                disabled={busy}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <label>
              Interpolation
              <select
                value={interpolation}
                disabled={busy}
                onChange={(e) => setInterpolation(e.target.value)}
              >
                <option value="linear">linear</option>
                <option value="smooth">smooth</option>
              </select>
            </label>
            <p className="library-pass-panel__hint">
              Init-video re-denoise + temporal blend · new clip, original kept
            </p>
          </>
        )}

        {kind === "faces" && (
          <>
            <div className="library-pass-panel__row" title="Face crop follows the clip short edge.">
              <span>Canvas</span>
              <span className="library-pass-panel__value">matches shot · {faceCanvas}p</span>
            </div>
            <label>
              Denoise
              <input
                type="number"
                min={denoiseMin}
                max={denoiseMax}
                step={0.05}
                value={denoise}
                disabled={busy}
                onChange={(e) => setDenoise(Number(e.target.value))}
              />
            </label>
            <label>
              Seed
              <input
                type="number"
                min={0}
                value={seed}
                disabled={busy}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <p className="library-pass-panel__hint">
              SAM 3.1 downloads on first Faces run if needed.
            </p>
          </>
        )}
      </div>

      {busy && (
        <div className="library-pass-panel__status" aria-live="polite">
          <div className="library-pass-panel__status-bar">
            {statusPct != null ? (
              <div
                className="library-pass-panel__status-fill"
                style={{ width: `${Math.min(100, statusPct)}%` }}
              />
            ) : (
              <div className="library-pass-panel__status-pulse" />
            )}
          </div>
          <p className="library-pass-panel__status-msg">
            {statusMessage || `${TITLES[kind]} in progress…`}
          </p>
        </div>
      )}

      <div className="library-pass-panel__actions">
        <button type="button" className="btn-secondary" disabled={busy} onClick={onClose}>
          Cancel
        </button>
        <button type="button" className="btn-primary" disabled={busy} onClick={submit}>
          {busy ? `${TITLES[kind]}…` : `Run ${TITLES[kind]}`}
        </button>
      </div>
    </div>
  );
}
