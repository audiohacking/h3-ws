import { useMemo, useRef, useState } from "react";
import { activeRefs, handleForRef, nearestDurationId } from "../../compile";
import { ratioOfAspect, resolutionChipLabel } from "../../h3Canvas";
import { useDismissable } from "../../hooks/useDismissable";
import type { PresetOption, ReferenceItem, RoutingMode } from "../../types";
import { AspectPill } from "./AspectPill";
import { MediaLibraryModal, type LibraryPickPurpose } from "./MediaLibraryModal";
import {
  type ResolutionCanvasState,
  ResolutionPill,
} from "./ResolutionPill";

type Props = {
  disabled?: boolean;
  routing: RoutingMode;
  onRoutingChange: (next: RoutingMode) => void;
  durationId: string;
  durationPresets: PresetOption[];
  onDurationId: (id: string) => void;
  canvas: ResolutionCanvasState;
  onCanvas: (next: Partial<ResolutionCanvasState>) => void;
  redetailAvailable?: boolean;
  refs: ReferenceItem[];
  imageName: string | null;
  endImageName: string | null;
  onUpload: (file: File, kind: "image" | "video" | "audio") => Promise<{
    path: string;
    durationS?: number;
    filename?: string;
  }>;
  onPickStartImage: (item: { path: string; name: string }) => void;
  onPickEndImage: (item: { path: string; name: string }) => void;
  onClearStart: () => void;
  onClearEnd: () => void;
  engineNote?: string;
};

const ROUTING_LABEL: Record<RoutingMode, string> = {
  auto: "AUTO",
  fl2va: "FL2VA",
  ref2va: "REF2VA",
};

function cycleRouting(current: RoutingMode): RoutingMode {
  if (current === "auto") return "fl2va";
  if (current === "fl2va") return "ref2va";
  return "auto";
}

export function ConfigRow({
  disabled,
  routing,
  onRoutingChange,
  durationId,
  durationPresets,
  onDurationId,
  canvas,
  onCanvas,
  redetailAvailable = true,
  refs,
  imageName,
  endImageName,
  onUpload,
  onPickStartImage,
  onPickEndImage,
  onClearStart,
  onClearEnd,
  engineNote,
}: Props) {
  const aspectAnchor = useRef<HTMLDivElement>(null);
  const [aspectChipEl, setAspectChipEl] = useState<HTMLButtonElement | null>(null);
  const [resChipEl, setResChipEl] = useState<HTMLButtonElement | null>(null);
  const durAnchor = useRef<HTMLDivElement>(null);
  const [framePicker, setFramePicker] = useState<LibraryPickPurpose | null>(null);
  const [aspectOpen, setAspectOpen] = useState(false);
  const [resOpen, setResOpen] = useState(false);
  const [durOpen, setDurOpen] = useState(false);
  useDismissable(durOpen, () => setDurOpen(false), durAnchor);

  const durIndex = Math.max(0, durationPresets.findIndex((d) => d.id === durationId));
  const durLabel = durationPresets[durIndex]?.id ?? durationId;

  const matchRef = activeRefs(refs).find((r) => r.durationS && r.durationS > 0);
  const matchIndex = matchRef ? refs.indexOf(matchRef) : -1;
  const matchHandle = matchRef && matchIndex >= 0 ? handleForRef(refs, matchIndex) : null;

  const chip = useMemo(
    () =>
      resolutionChipLabel({
        aspect: canvas.aspect,
        shortEdge: canvas.shortEdge,
        sampleEdge: canvas.sampleEdge,
        upscale: canvas.upscale,
      }),
    [canvas.aspect, canvas.shortEdge, canvas.sampleEdge, canvas.upscale],
  );
  const resTitle = `${chip.title} ${chip.sub}`;

  return (
    <>
      <div className="config-card__row">
        <button
          type="button"
          className="chip-btn chip-btn--routing"
          disabled={disabled}
          onClick={() => onRoutingChange(cycleRouting(routing))}
          title="Cycle AUTO / always FL2VA / always Ref2VA"
        >
          {ROUTING_LABEL[routing]}
        </button>
        <div className="frame-chip-group">
          <button
            type="button"
            className={`chip-btn${imageName ? " is-on" : ""}`}
            disabled={disabled}
            title={imageName ?? "Choose start frame from library or upload"}
            onClick={() => setFramePicker("start_frame")}
          >
            {imageName ? `Start · ${imageName}` : "Start frame"}
          </button>
          {imageName && (
            <button
              type="button"
              className="chip-btn chip-btn--clear"
              disabled={disabled}
              title="Clear start frame"
              aria-label="Clear start frame"
              onClick={onClearStart}
            >
              ×
            </button>
          )}
        </div>
        <div className="frame-chip-group">
          <button
            type="button"
            className={`chip-btn${endImageName ? " is-on" : ""}`}
            disabled={disabled}
            title={endImageName ?? "Choose end frame from library or upload"}
            onClick={() => setFramePicker("end_frame")}
          >
            {endImageName ? `End · ${endImageName}` : "End frame"}
          </button>
          {endImageName && (
            <button
              type="button"
              className="chip-btn chip-btn--clear"
              disabled={disabled}
              title="Clear end frame"
              aria-label="Clear end frame"
              onClick={onClearEnd}
            >
              ×
            </button>
          )}
        </div>
        <div className="popover-anchor" ref={durAnchor}>
          <div className="stepper-pill" title="Duration 1–15s (H3 snaps to 5+17n @ 24fps)">
            <button
              type="button"
              disabled={disabled || durIndex <= 0}
              onClick={() => onDurationId(durationPresets[durIndex - 1].id)}
            >
              −
            </button>
            <button
              type="button"
              className="stepper-pill__value stepper-pill__value--dur stepper-pill__value--btn"
              disabled={disabled}
              aria-expanded={durOpen}
              title="Pick any length 1–15s"
              onClick={() => setDurOpen((v) => !v)}
            >
              {durLabel}
            </button>
            <button
              type="button"
              disabled={disabled || durIndex >= durationPresets.length - 1}
              onClick={() => onDurationId(durationPresets[durIndex + 1].id)}
            >
              +
            </button>
          </div>
          {durOpen && (
            <div className="aspect-menu aspect-menu--duration" role="listbox">
              {durationPresets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  role="option"
                  aria-selected={p.id === durationId}
                  className={p.id === durationId ? "is-active" : ""}
                  onClick={() => {
                    onDurationId(p.id);
                    setDurOpen(false);
                  }}
                >
                  {p.id}
                </button>
              ))}
            </div>
          )}
        </div>
        {matchRef && matchHandle && matchRef.durationS != null && (
          <button
            type="button"
            className="chip-btn chip-btn--dashed"
            disabled={disabled}
            title={`Snap duration to ${matchHandle} (${matchRef.durationS.toFixed(2)} s)`}
            onClick={() => {
              const id = nearestDurationId(matchRef.durationS ?? 0, durationPresets);
              if (id) onDurationId(id);
            }}
          >
            match {matchHandle}
          </button>
        )}
        <div className="popover-anchor" ref={aspectAnchor}>
          <button
            ref={setAspectChipEl}
            type="button"
            className="chip-btn chip-btn--aspect"
            disabled={disabled}
            aria-expanded={aspectOpen}
            onClick={() => setAspectOpen((v) => !v)}
          >
            {canvas.aspect}
          </button>
          <AspectPill
            disabled={disabled}
            open={aspectOpen}
            anchor={aspectChipEl}
            aspect={canvas.aspect}
            onAspect={(label) => onCanvas({ aspect: label })}
            aspectSource="pill"
            refs={refs}
            autoRatio={ratioOfAspect(canvas.aspect)}
            autoSub={canvas.aspect}
            onClose={() => setAspectOpen(false)}
          />
        </div>
        <div className="popover-anchor">
          <button
            ref={setResChipEl}
            type="button"
            className="chip-btn"
            disabled={disabled}
            title={resTitle}
            aria-expanded={resOpen}
            onClick={() => setResOpen((v) => !v)}
          >
            {chip.sub.includes("→") ? `${chip.title} ${chip.sub}` : `${chip.title} ${chip.sub}`}
          </button>
          <ResolutionPill
            disabled={disabled}
            open={resOpen}
            anchor={resChipEl}
            canvas={canvas}
            onCanvas={onCanvas}
            redetailAvailable={redetailAvailable}
            onClose={() => setResOpen(false)}
          />
        </div>
      </div>
      {engineNote && <p className="engine-note">{engineNote}</p>}

      <MediaLibraryModal
        open={framePicker != null}
        purpose={framePicker ?? "start_frame"}
        initialTab="image"
        refs={refs}
        disabled={disabled}
        onClose={() => setFramePicker(null)}
        onUpload={onUpload}
        onAdd={() => {
          /* frame pick uses onPickImage */
        }}
        onPickImage={(item) => {
          if (framePicker === "end_frame") onPickEndImage(item);
          else onPickStartImage(item);
          setFramePicker(null);
        }}
      />
    </>
  );
}
