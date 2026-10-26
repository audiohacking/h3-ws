import { useEffect, useRef, useState } from "react";
import { useDismissable } from "../../hooks/useDismissable";
import type { UpframeConfigPublic } from "../../types";

export type UpframePillProps = {
  disabled?: boolean;
  config: UpframeConfigPublic | null | undefined;
  enabled: boolean;
  multiplier: number;
  onEnabled: (v: boolean) => void;
  onMultiplier: (v: number) => void;
  /** Open Models when FILM weights are missing. */
  onNeedWeights?: () => void;
};

export function UpframePill({
  disabled,
  config,
  enabled,
  multiplier,
  onEnabled,
  onMultiplier,
  onNeedWeights,
}: UpframePillProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const weightsReady = config?.available === true;
  const title = !weightsReady
    ? "Upframe — needs film_net_fp16 (~69 MB). Click to open Models."
    : "Upframe — FILM temporal interpolate → new library clip (original kept)";

  useDismissable(open, () => setOpen(false), rootRef);
  useEffect(() => {
    if (disabled || !weightsReady) setOpen(false);
  }, [disabled, weightsReady]);

  return (
    <div
      ref={rootRef}
      className={`faces-pill upframe-pill${enabled ? " is-on" : ""}${open ? " is-open" : ""}${!weightsReady ? " needs-setup" : ""}`}
    >
      <button
        type="button"
        className="faces-pill__main"
        disabled={disabled}
        title={title}
        aria-pressed={enabled}
        onClick={() => {
          if (!weightsReady) {
            onNeedWeights?.();
            return;
          }
          onEnabled(!enabled);
        }}
      >
        Upframe
        {enabled && weightsReady ? (
          <span className="faces-pill__badge">×{multiplier}</span>
        ) : null}
      </button>
      <button
        type="button"
        className="faces-pill__gear"
        disabled={disabled || !weightsReady}
        title="Upframe settings"
        aria-expanded={open}
        onClick={() => {
          if (!weightsReady) {
            onNeedWeights?.();
            return;
          }
          setOpen((v) => !v);
        }}
      >
        ▾
      </button>
      {open && weightsReady && (
        <div className="faces-pill__pop" role="dialog" aria-label="Upframe settings">
          <label>
            Frame multiplier
            <select
              value={multiplier}
              disabled={disabled}
              onChange={(e) => onMultiplier(Math.max(2, Math.round(Number(e.target.value))))}
            >
              <option value={2}>×2 (insert mid-frames)</option>
              <option value={3}>×3</option>
              <option value={4}>×4</option>
            </select>
          </label>
          <p className="faces-pill__hint">
            {config?.note ||
              "Off by default. When on, adds a new library clip; the original is kept."}
          </p>
        </div>
      )}
    </div>
  );
}
