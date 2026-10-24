import { useEffect, useRef, useState } from "react";
import { useDismissable } from "../../hooks/useDismissable";
import type { FacesConfigPublic } from "../../types";

export type FacesPillProps = {
  disabled?: boolean;
  config: FacesConfigPublic | null | undefined;
  enabled: boolean;
  canvas: number;
  denoise: number;
  seed: number;
  onEnabled: (v: boolean) => void;
  /** Optional — canvas follows the shot; kept for API compatibility. */
  onCanvas?: (v: number) => void;
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
  onDenoise,
  onSeed,
}: FacesPillProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const canEnable = Boolean(config?.available);
  const samReady = Boolean(config?.sam3_ready);
  const title = !config
    ? "Faces"
    : !samReady
      ? "Faces — SAM 3.1 downloads automatically when you generate"
      : "Face refine — SAM detect, re-draw with your Ref2VA refs";

  useDismissable(open, () => setOpen(false), rootRef);
  useEffect(() => {
    if (disabled || !canEnable) setOpen(false);
  }, [disabled, canEnable]);

  return (
    <div
      ref={rootRef}
      className={`faces-pill${enabled ? " is-on" : ""}${open ? " is-open" : ""}`}
    >
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
        {!samReady && enabled ? <span className="faces-pill__badge">SAM</span> : null}
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
          <div className="faces-pill__row" title="Face crop canvas follows the shot short edge (384–768).">
            <span>Canvas</span>
            <span className="faces-pill__match">
              matches shot · {canvas}p
            </span>
          </div>
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
              "Face crop canvas matches the shot short edge. SAM downloads on first Faces run if needed."}
          </p>
        </div>
      )}
    </div>
  );
}
