/**
 * Continuity resolution popover — port of ``openResolutionPopover`` /
 * ``edgeSlider`` from ComfyUI-Continuity ``web/creator/pills.js``.
 *
 * Settings match Continuity H3 ``declare.RULES`` and ``state.js`` defaults:
 * short edge 384–2048 (native 768), modes two_pass / direct / redetail,
 * sampled-at / drawn-up-by / refine / steps when two-pass.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  BICUBIC_REFINE,
  CANVAS_MULTIPLE,
  DEFAULT_REFINE_DENOISE,
  DEFAULT_REFINE_STEPS,
  isTwoPass,
  MAX_REFINE_DENOISE,
  MAX_REFINE_STEPS,
  MAX_SHORT_EDGE,
  MIN_REFINE_DENOISE,
  MIN_SHORT_EDGE,
  NATIVE_SHORT_EDGE,
  ratioOfAspect,
  redetailTarget,
  resolveCanvas,
  sampleEdgeOf,
  snapShortEdge,
  TRAINED_REFINE,
  type RefineUpscaler,
  type UpscaleMode,
} from "../../h3Canvas";

export type ResolutionPassMode = "one_pass" | "redetail";

export type ResolutionCanvasState = {
  aspect: string;
  shortEdge: number;
  sampleEdge: number;
  upscale: UpscaleMode;
  refineDenoise: number;
  refineUpscaler: RefineUpscaler;
  refineSteps: number;
  /** Latent upscaler file picked under Models — enables "trained". */
  trainedUpscalerReady?: boolean;
};

type Props = {
  disabled?: boolean;
  open: boolean;
  anchor: HTMLElement | null;
  canvas: ResolutionCanvasState;
  onCanvas: (next: Partial<ResolutionCanvasState>) => void;
  redetailAvailable?: boolean;
  onClose: () => void;
};

function StepperPill({
  value,
  min,
  max,
  step,
  disabled,
  format,
  title,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  format: (n: number) => string;
  title?: string;
  onChange: (n: number) => void;
}) {
  const snap = (n: number) => {
    const clamped = Math.min(max, Math.max(min, n));
    return Math.min(max, Math.max(min, Math.round((clamped - min) / step) * step + min));
  };
  return (
    <span className="mmc-stepper" title={title}>
      <button
        type="button"
        className="mmc-step"
        disabled={disabled || value <= min}
        aria-label="Decrease"
        onClick={() => onChange(snap(value - step))}
      >
        −
      </button>
      <span className="mmc-stepper__val">{format(value)}</span>
      <button
        type="button"
        className="mmc-step"
        disabled={disabled || value >= max}
        aria-label="Increase"
        onClick={() => onChange(snap(value + step))}
      >
        +
      </button>
    </span>
  );
}

export function ResolutionPill({
  disabled,
  open,
  anchor,
  canvas,
  onCanvas,
  redetailAvailable = true,
  onClose,
}: Props) {
  const popRef = useRef<HTMLDivElement>(null);
  const {
    aspect,
    shortEdge,
    sampleEdge,
    upscale,
    refineDenoise,
    refineUpscaler,
    refineSteps,
    trainedUpscalerReady = false,
  } = canvas;

  const ratio = ratioOfAspect(aspect);
  const [draft, setDraft] = useState(shortEdge);

  useEffect(() => {
    setDraft(shortEdge);
  }, [shortEdge, open]);

  const edge = snapShortEdge(draft);
  const [outW, outH] = resolveCanvas(ratio, edge);
  const sampled = sampleEdgeOf(sampleEdge, edge);
  const two = isTwoPass(sampleEdge, edge, upscale);
  const over = edge > NATIVE_SHORT_EDGE;
  const cap = Math.min(NATIVE_SHORT_EDGE, edge);
  const finished =
    upscale === "redetail" && redetailAvailable
      ? redetailTarget(ratio, sampleEdge, edge)
      : null;

  function commitEdge(raw: number) {
    const next = snapShortEdge(raw);
    setDraft(next);
    const patch: Partial<ResolutionCanvasState> = { shortEdge: next };
    if (upscale === "redetail" && next > NATIVE_SHORT_EDGE) {
      patch.shortEdge = NATIVE_SHORT_EDGE;
      setDraft(NATIVE_SHORT_EDGE);
    }
    // Keep sample_edge from floating above the new slider.
    if (sampleEdge > (patch.shortEdge ?? next)) {
      patch.sampleEdge = patch.shortEdge ?? next;
    }
    onCanvas(patch);
  }

  function nudge(delta: number) {
    commitEdge(draft + delta * CANVAS_MULTIPLE);
  }

  function setUpscale(mode: UpscaleMode) {
    const patch: Partial<ResolutionCanvasState> = { upscale: mode };
    if (mode === "redetail" && draft > NATIVE_SHORT_EDGE) {
      patch.shortEdge = NATIVE_SHORT_EDGE;
      setDraft(NATIVE_SHORT_EDGE);
    }
    onCanvas(patch);
  }

  useLayoutEffect(() => {
    if (!open || !anchor || !popRef.current) return;
    const pop = popRef.current;
    const place = () => {
      const rect = anchor.getBoundingClientRect();
      const box = pop.getBoundingClientRect();
      const left = Math.max(8, Math.min(rect.left, window.innerWidth - box.width - 8));
      const above = rect.top - box.height - 8 > 8;
      const top = above
        ? rect.top - box.height - 8
        : Math.min(rect.bottom + 8, window.innerHeight - box.height - 8);
      pop.style.left = `${left}px`;
      pop.style.top = `${Math.max(8, top)}px`;
    };
    place();
    const ro = new ResizeObserver(place);
    ro.observe(pop);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, anchor, draft, upscale, sampleEdge, refineDenoise, refineUpscaler, refineSteps]);

  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => {
      const t = event.target as Node | null;
      if (!t) return;
      if (popRef.current?.contains(t)) return;
      if (anchor?.contains(t)) return;
      onCloseRef.current();
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onCloseRef.current();
      }
    };
    const timer = window.setTimeout(() => {
      document.addEventListener("pointerdown", away, true);
      document.addEventListener("keydown", key, true);
    }, 0);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", key, true);
    };
  }, [open, anchor]);

  if (!open) return null;

  const nativeMarkPos =
    MAX_SHORT_EDGE > MIN_SHORT_EDGE
      ? (NATIVE_SHORT_EDGE - MIN_SHORT_EDGE) / (MAX_SHORT_EDGE - MIN_SHORT_EDGE)
      : 0.5;

  let note: string;
  let warn = false;
  let sizeLabel: string;
  if (finished) {
    sizeLabel = `${finished.width} × ${finished.height}`;
    note =
      "Fine detail is invented rather than recovered — faces and logos come back changed.";
  } else if (two) {
    sizeLabel = `${outW} × ${outH}`;
    note =
      refineUpscaler === "trained"
        ? `Sampled at ${sampled} px, drawn up by the trained upscaler, then refined at this size.`
        : `Sampled at ${sampled} px, then a second pass refines up to this size.`;
  } else {
    sizeLabel = `${outW} × ${outH}`;
    warn = over;
    note = over
      ? `Above the trained ${NATIVE_SHORT_EDGE} px short edge — off-distribution, not just slower.`
      : edge === NATIVE_SHORT_EDGE
        ? "Native. What the open weights were trained at."
        : `${(NATIVE_SHORT_EDGE / edge).toFixed(1)}× smaller short edge than native — faster, softer.`;
  }

  const showSampledAt = cap > MIN_SHORT_EDGE && (!over || upscale !== "direct");

  return createPortal(
    <div
      ref={popRef}
      className="mmc-pop mmc-slider res-pop"
      role="dialog"
      aria-label="Resolution"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="mmc-slider-body res-pop__slider">
        <div className="mmc-slider-read res-pop__read">
          <span>
            <span className="mmc-edge res-pop__edge">{edge}</span>
            <span className="mmc-edge-unit res-pop__edge-unit">px</span>
          </span>
          <span className="res-pop__size">{sizeLabel}</span>
        </div>

        <div className="mmc-slider-row res-pop__row">
          <button
            type="button"
            className="mmc-step res-pop__step"
            disabled={disabled || edge <= MIN_SHORT_EDGE}
            title={`Down ${CANVAS_MULTIPLE} px`}
            aria-label="Smaller short edge"
            onClick={() => nudge(-1)}
          >
            −
          </button>
          <div className="mmc-slider-track res-pop__track">
            <input
              type="range"
              min={MIN_SHORT_EDGE}
              max={MAX_SHORT_EDGE}
              step={CANVAS_MULTIPLE}
              value={edge}
              disabled={disabled}
              aria-label="Short edge in pixels"
              onPointerDown={(e) => e.stopPropagation()}
              onInput={(e) => {
                setDraft(Number((e.target as HTMLInputElement).value));
              }}
              onChange={(e) => {
                commitEdge(Number(e.target.value));
              }}
            />
            <button
              type="button"
              className={`mmc-slider-mark res-pop__mark${edge === NATIVE_SHORT_EDGE ? " on is-on" : ""}`}
              style={{ ["--p" as string]: String(nativeMarkPos) }}
              title={`native — ${NATIVE_SHORT_EDGE} px`}
              disabled={disabled}
              onClick={() => commitEdge(NATIVE_SHORT_EDGE)}
            >
              <span>native</span>
            </button>
          </div>
          <button
            type="button"
            className="mmc-step res-pop__step"
            disabled={disabled || edge >= MAX_SHORT_EDGE}
            title={`Up ${CANVAS_MULTIPLE} px`}
            aria-label="Larger short edge"
            onClick={() => nudge(1)}
          >
            +
          </button>
        </div>

        <p className={`mmc-native res-pop__note${warn ? " over is-over" : ""}`}>{note}</p>
      </div>

      <div className="mmc-twopass res-pop__modes">
        {over ? (
          <>
            <button
              type="button"
              className="mmc-opt res-pop__opt"
              role="radio"
              aria-checked={upscale === "two_pass"}
              disabled={disabled}
              onClick={() => setUpscale("two_pass")}
            >
              <span className="mmc-opt-label mmc-opt-col res-pop__opt-text">
                <span className="res-pop__opt-label">two passes</span>
                <span className="mmc-opt-sub res-pop__opt-sub">
                  {sampled} px first, refined up to {outW} × {outH}
                </span>
              </span>
              <span className="mmc-radio res-pop__radio" aria-hidden />
            </button>
            <button
              type="button"
              className="mmc-opt res-pop__opt"
              role="radio"
              aria-checked={upscale === "direct"}
              disabled={disabled}
              onClick={() => setUpscale("direct")}
            >
              <span className="mmc-opt-label mmc-opt-col res-pop__opt-text">
                <span className="res-pop__opt-label">direct</span>
                <span className="mmc-opt-sub res-pop__opt-sub">
                  one pass at {outW} × {outH} — off-distribution
                </span>
              </span>
              <span className="mmc-radio res-pop__radio" aria-hidden />
            </button>
          </>
        ) : redetailAvailable ? (
          <button
            type="button"
            className="mmc-opt res-pop__opt"
            role="radio"
            aria-checked={upscale !== "redetail"}
            disabled={disabled}
            onClick={() => {
              if (upscale === "redetail") setUpscale("two_pass");
            }}
          >
            <span className="mmc-opt-label mmc-opt-col res-pop__opt-text">
              <span className="res-pop__opt-label">{two ? "two passes" : "one pass"}</span>
              <span className="mmc-opt-sub res-pop__opt-sub">
                {two
                  ? `${sampled} px first, refined up to ${outW} × ${outH}`
                  : `${outW} × ${outH}, as sampled`}
              </span>
            </span>
            <span className="mmc-radio res-pop__radio" aria-hidden />
          </button>
        ) : null}

        {redetailAvailable ? (
          <button
            type="button"
            className="mmc-opt res-pop__opt"
            role="radio"
            aria-checked={upscale === "redetail"}
            disabled={disabled}
            onClick={() => setUpscale("redetail")}
          >
            <span className="mmc-opt-label mmc-opt-col res-pop__opt-text">
              <span className="res-pop__opt-label">ReDetail</span>
              <span className="mmc-opt-sub res-pop__opt-sub">
                {(() => {
                  const fin = redetailTarget(ratio, sampleEdge, Math.min(edge, NATIVE_SHORT_EDGE));
                  return `×2 to ${fin.width} × ${fin.height} — re-rendered, not sharpened`;
                })()}
              </span>
            </span>
            <span className="mmc-radio res-pop__radio" aria-hidden />
          </button>
        ) : null}

        {showSampledAt ? (
          <div className="mmc-refine-row">
            <span className="mmc-refine-label">sampled at</span>
            <StepperPill
              value={sampled}
              min={MIN_SHORT_EDGE}
              max={cap}
              step={CANVAS_MULTIPLE}
              disabled={disabled}
              format={(n) => `${n} px`}
              title="The short edge the first pass samples at. At the slider's size it is the only pass; under it, a second pass refines up to the slider."
              onChange={(next) => {
                onCanvas({
                  sampleEdge: next,
                  ...(!over ? { upscale: "two_pass" as UpscaleMode } : {}),
                });
              }}
            />
          </div>
        ) : null}

        {two ? (
          <>
            <div className="mmc-refine-row">
              <span className="mmc-refine-label">drawn up by</span>
              <div className="mmc-aspect-flip mmc-refine-flip">
                <button
                  type="button"
                  className="mmc-flip-opt"
                  aria-pressed={refineUpscaler !== "trained"}
                  disabled={disabled}
                  title="Interpolation. The second pass invents every detail from a blurred picture, so it needs a deep re-noise."
                  onClick={() =>
                    onCanvas({
                      refineUpscaler: "bicubic",
                      refineDenoise: BICUBIC_REFINE.refineDenoise,
                      refineSteps: BICUBIC_REFINE.refineSteps,
                    })
                  }
                >
                  bicubic
                </button>
                <button
                  type="button"
                  className="mmc-flip-opt"
                  aria-pressed={refineUpscaler === "trained"}
                  disabled={disabled || !trainedUpscalerReady}
                  title={
                    trainedUpscalerReady
                      ? "The latent upscaler picked under weights draws the picture up. Holds faces at a light re-noise and needs few steps — measured at 0.30 and 3."
                      : "Pick a latent upscaler under weights first."
                  }
                  onClick={() =>
                    onCanvas({
                      refineUpscaler: "trained",
                      refineDenoise: TRAINED_REFINE.refineDenoise,
                      refineSteps: TRAINED_REFINE.refineSteps,
                    })
                  }
                >
                  trained
                </button>
              </div>
            </div>
            {!trainedUpscalerReady ? (
              <div className="mmc-twopass-hint">no latent upscaler under weights</div>
            ) : null}
            <div className="mmc-refine-row">
              <span className="mmc-refine-label">refine</span>
              <StepperPill
                value={refineDenoise}
                min={MIN_REFINE_DENOISE}
                max={MAX_REFINE_DENOISE}
                step={0.05}
                disabled={disabled}
                format={(n) => n.toFixed(2)}
                title="How much of the schedule the second pass re-runs. Lower keeps more of the first pass; higher resolves more detail and drifts further from it."
                onChange={(next) => onCanvas({ refineDenoise: next })}
              />
            </div>
            <div className="mmc-refine-row">
              <span className="mmc-refine-label">steps</span>
              <StepperPill
                value={refineSteps}
                min={0}
                max={MAX_REFINE_STEPS}
                step={1}
                disabled={disabled}
                format={(n) => (n === 0 ? "all" : String(n))}
                title="Steps the second pass runs at the target. 'all' is the sampler's count again; a short tail is faster and, through the trained upscaler, just as clean."
                onChange={(next) => onCanvas({ refineSteps: next })}
              />
            </div>
          </>
        ) : null}
      </div>
    </div>,
    document.body,
  );
}

/** @deprecated Prefer Continuity ``UpscaleMode`` via canvas state. */
export function passModeFromUpscale(upscale: UpscaleMode): ResolutionPassMode {
  return upscale === "redetail" ? "redetail" : "one_pass";
}

export function defaultResolutionCanvas(): ResolutionCanvasState {
  return {
    aspect: "16:9",
    shortEdge: NATIVE_SHORT_EDGE,
    sampleEdge: NATIVE_SHORT_EDGE,
    upscale: "two_pass",
    refineDenoise: DEFAULT_REFINE_DENOISE,
    refineUpscaler: "bicubic",
    refineSteps: DEFAULT_REFINE_STEPS,
    trainedUpscalerReady: false,
  };
}
