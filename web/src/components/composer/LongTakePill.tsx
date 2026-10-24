import { useEffect, useRef, useState } from "react";
import { useDismissable } from "../../hooks/useDismissable";
import type { LongTakeConfigPublic } from "../../types";

export type LongTakePillProps = {
  disabled?: boolean;
  config: LongTakeConfigPublic | null | undefined;
  enabled: boolean;
  denoise: number;
  keyframes: string;
  seed: number;
  onEnabled: (v: boolean) => void;
  onDenoise: (v: number) => void;
  onKeyframes: (v: string) => void;
  onSeed: (v: number) => void;
};

export function LongTakePill({
  disabled,
  config,
  enabled,
  denoise,
  keyframes,
  seed,
  onEnabled,
  onDenoise,
  onKeyframes,
  onSeed,
}: LongTakePillProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const canEnable = Boolean(config?.available);
  const title = !config
    ? "Long take"
    : "Long take refine — init-video re-denoise + temporal blend (A/B twin)";

  useDismissable(open, () => setOpen(false), rootRef);
  useEffect(() => {
    if (disabled || !canEnable) setOpen(false);
  }, [disabled, canEnable]);

  return (
    <div
      ref={rootRef}
      className={`faces-pill longtake-pill${enabled ? " is-on" : ""}${open ? " is-open" : ""}`}
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
        Long take
      </button>
      <button
        type="button"
        className="faces-pill__gear"
        disabled={disabled || !canEnable}
        title="Long take settings"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        ▾
      </button>
      {open && canEnable && (
        <div className="faces-pill__pop" role="dialog" aria-label="Long take settings">
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
            Keyframes
            <input
              type="text"
              value={keyframes}
              disabled={disabled}
              onChange={(e) => onKeyframes(e.target.value)}
              spellCheck={false}
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
              "Off by default. When on, writes a *_longtake.mp4 twin beside the baseline."}
          </p>
        </div>
      )}
    </div>
  );
}
