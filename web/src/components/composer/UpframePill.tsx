import type { UpframeConfigPublic } from "../../types";

export type UpframePillProps = {
  disabled?: boolean;
  config: UpframeConfigPublic | null | undefined;
  enabled: boolean;
  onEnabled: (v: boolean) => void;
};

export function UpframePill({ disabled, config, enabled, onEnabled }: UpframePillProps) {
  const canEnable = Boolean(config?.available);
  const title = !canEnable
    ? "Upframe — download film_net_fp16 in Models first"
    : "Upframe — FILM temporal interpolate (A/B twin, independent of ReDetail)";

  return (
    <div className={`faces-pill upframe-pill${enabled ? " is-on" : ""}`}>
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
        Upframe
      </button>
    </div>
  );
}
