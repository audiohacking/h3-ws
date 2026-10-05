import { useState } from "react";
import type { FacesConfigPublic } from "../../types";

export type FacesPillProps = {
  disabled?: boolean;
  config: FacesConfigPublic | null | undefined;
  enabled: boolean;
  canvas: number;
  denoise: number;
  seed: number;
  onEnabled: (v: boolean) => void;
  onCanvas: (v: number) => void;
  onDenoise: (v: number) => void;
  onSeed: (v: number) => void;
};

export function FacesPill({
  disabled,
  config,
  enabled,
  canvas,
  denoise,
  seed,
  onEnabled,
  onCanvas,
  onDenoise,
  onSeed,
}: FacesPillProps) {
  const [open, setOpen] = useState(false);
  const samReady = Boolean(config?.sam3_ready);
  // Continuity: SAM is essential — Faces stays off until MLX SAM 3.1 is ready.
  const canEnable = Boolean(config?.available) && samReady;
  const title = !config
    ? "Faces"
    : !samReady
      ? "Download SAM 3.1 in Models first (required face detector)"
      : "Continuity face pass — SAM detect, refine with your Ref2VA refs";

  return (
    <div className={`faces-pill${enabled ? " is-on" : ""}${open ? " is-open" : ""}`}>
      <button
        type="button"
        className="faces-pill__main"
        disabled={disabled || !canEnable}
        title={title}
        aria-pressed={enabled}
        onClick={() => {
          if (!canEnable) return;
          onEnabled(!enabled);
        }}
      >
        Faces
      </button>
      <button
        type="button"
        className="faces-pill__gear"
        disabled={disabled || !canEnable}
        title="Faces settings"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        ▾
      </button>
      {open && canEnable && (
        <div className="faces-pill__pop" role="dialog" aria-label="Faces settings">
          <label>
            Canvas
            <input
              type="number"
              min={config?.canvas_min ?? 384}
              max={config?.canvas_max ?? 768}
              step={32}
              value={canvas}
              disabled={disabled}
              onChange={(e) => onCanvas(Number(e.target.value))}
            />
          </label>
          <label>
            Denoise
            <input
              type="number"
              min={config?.denoise_min ?? 0.1}
              max={config?.denoise_max ?? 0.9}
              step={0.05}
              value={denoise}
              disabled={disabled}
              onChange={(e) => onDenoise(Number(e.target.value))}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              min={0}
              value={seed}
              disabled={disabled}
              onChange={(e) => onSeed(Number(e.target.value))}
            />
          </label>
          <p className="faces-pill__hint">
            {config?.note ||
              "End-of-run face refine. Reuses your Ref2VA picture/video/audio refs. Audio remuxed."}
          </p>
        </div>
      )}
    </div>
  );
}
