import { useMemo } from "react";
import type { PresetOption } from "../../types";

export type ResolutionPassMode = "one_pass" | "redetail";

type Props = {
  disabled?: boolean;
  resolutionId: string;
  resolutionPresets: PresetOption[];
  onResolutionId: (id: string) => void;
  passMode: ResolutionPassMode;
  onPassMode: (mode: ResolutionPassMode) => void;
  redetailAvailable?: boolean;
};

function shortSide(p: PresetOption | undefined): number {
  if (!p?.width || !p?.height) return 0;
  return Math.min(p.width, p.height);
}

export function ResolutionPill({
  disabled,
  resolutionId,
  resolutionPresets,
  onResolutionId,
  passMode,
  onPassMode,
  redetailAvailable = true,
}: Props) {
  const resolution = resolutionPresets.find((r) => r.id === resolutionId);
  const aspect = resolution?.aspect || "1:1";
  const inAspect = useMemo(
    () =>
      resolutionPresets.filter(
        (p) => (p.aspect || "1:1") === aspect && !p.render_width,
      ),
    [resolutionPresets, aspect],
  );
  const idx = Math.max(
    0,
    inAspect.findIndex((p) => p.id === resolutionId),
  );
  const sample = inAspect[idx] ?? resolution;
  const short = shortSide(sample);
  const outW = sample?.width ?? 0;
  const outH = sample?.height ?? 0;
  const rdW = outW * 2;
  const rdH = outH * 2;
  const isNative = short >= 768;

  return (
    <div className={`resolution-pill${passMode === "redetail" ? " is-redetail" : ""}`}>
      <div className="resolution-pill__head">
        <span className="resolution-pill__short">{short || "—"} px</span>
        <span className="resolution-pill__size">
          {outW && outH ? `${outW} × ${outH}` : resolutionId}
        </span>
      </div>

      <div className="stepper-pill resolution-pill__stepper" title="Sampled canvas (one-pass size)">
        <button
          type="button"
          disabled={disabled || idx <= 0}
          onClick={() => onResolutionId(inAspect[idx - 1]!.id)}
          aria-label="Lower sample size"
        >
          −
        </button>
        <span className="stepper-pill__value">
          {isNative ? "native" : `${short} px`}
        </span>
        <button
          type="button"
          disabled={disabled || idx >= inAspect.length - 1}
          onClick={() => onResolutionId(inAspect[idx + 1]!.id)}
          aria-label="Raise sample size"
        >
          +
        </button>
      </div>
      <p className="resolution-pill__caption">
        {isNative
          ? "Native. What the open weights were trained at."
          : "Sampled below native — faster, softer detail."}
      </p>

      <div className="resolution-pill__modes" role="radiogroup" aria-label="Generation pass">
        <button
          type="button"
          role="radio"
          aria-checked={passMode === "one_pass"}
          className={`resolution-pill__mode${passMode === "one_pass" ? " is-on" : ""}`}
          disabled={disabled}
          onClick={() => onPassMode("one_pass")}
        >
          <span className="resolution-pill__mode-label">one pass</span>
          <span className="resolution-pill__mode-meta">
            {outW} × {outH}, as sampled
          </span>
          <span className="resolution-pill__check" aria-hidden>
            {passMode === "one_pass" ? "✓" : ""}
          </span>
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={passMode === "redetail"}
          className={`resolution-pill__mode${passMode === "redetail" ? " is-on" : ""}`}
          disabled={disabled || !redetailAvailable}
          title={
            redetailAvailable
              ? "×2 re-render refine twin (not a sharpen)"
              : "ReDetail unavailable"
          }
          onClick={() => onPassMode("redetail")}
        >
          <span className="resolution-pill__mode-label">ReDetail</span>
          <span className="resolution-pill__mode-meta">
            ×2 to {rdW} × {rdH} — re-rendered, not sharpened
          </span>
          <span className="resolution-pill__check" aria-hidden>
            {passMode === "redetail" ? "✓" : ""}
          </span>
        </button>
      </div>

      <div className="resolution-pill__sampled">
        <span>sampled at</span>
        <div className="stepper-pill">
          <button
            type="button"
            disabled={disabled || idx <= 0}
            onClick={() => onResolutionId(inAspect[idx - 1]!.id)}
          >
            −
          </button>
          <span className="stepper-pill__value">{short} px</span>
          <button
            type="button"
            disabled={disabled || idx >= inAspect.length - 1}
            onClick={() => onResolutionId(inAspect[idx + 1]!.id)}
          >
            +
          </button>
        </div>
      </div>
    </div>
  );
}
