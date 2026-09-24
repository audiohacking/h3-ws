import { useEffect, useMemo, useRef, useState } from "react";
import { activeRefs, handleForRef, nearestDurationId } from "../../compile";
import type { PresetOption, ReferenceItem, RoutingMode } from "../../types";
import { MediaLibraryModal, type LibraryPickPurpose } from "./MediaLibraryModal";

type Props = {
  disabled?: boolean;
  routing: RoutingMode;
  onRoutingChange: (next: RoutingMode) => void;
  durationId: string;
  durationPresets: PresetOption[];
  onDurationId: (id: string) => void;
  resolutionId: string;
  resolutionPresets: PresetOption[];
  onResolutionId: (id: string) => void;
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
  resolutionId,
  resolutionPresets,
  onResolutionId,
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
  const resAnchor = useRef<HTMLDivElement>(null);
  const durAnchor = useRef<HTMLDivElement>(null);
  const [framePicker, setFramePicker] = useState<LibraryPickPurpose | null>(null);
  const [aspectOpen, setAspectOpen] = useState(false);
  const [resOpen, setResOpen] = useState(false);
  const [durOpen, setDurOpen] = useState(false);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (aspectAnchor.current && !aspectAnchor.current.contains(t)) setAspectOpen(false);
      if (resAnchor.current && !resAnchor.current.contains(t)) setResOpen(false);
      if (durAnchor.current && !durAnchor.current.contains(t)) setDurOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const resolution = resolutionPresets.find((r) => r.id === resolutionId);
  const aspect = resolution?.aspect || "1:1";
  const aspects = useMemo(() => {
    const seen = new Set<string>();
    const order: string[] = [];
    for (const p of resolutionPresets) {
      const a = p.aspect || "1:1";
      if (!seen.has(a)) {
        seen.add(a);
        order.push(a);
      }
    }
    return order;
  }, [resolutionPresets]);
  const inAspect = resolutionPresets.filter((p) => (p.aspect || "1:1") === aspect);

  const durIndex = Math.max(0, durationPresets.findIndex((d) => d.id === durationId));
  const durLabel = durationPresets[durIndex]?.id ?? durationId;

  const matchRef = activeRefs(refs).find((r) => r.durationS && r.durationS > 0);
  const matchIndex = matchRef ? refs.indexOf(matchRef) : -1;
  const matchHandle = matchRef && matchIndex >= 0 ? handleForRef(refs, matchIndex) : null;

  const resLabel = resolution
    ? resolution.render_width
      ? `${resolution.render_width}→${resolution.width}×${resolution.height}`
      : `${resolution.width}×${resolution.height}`
    : resolutionId;

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
            type="button"
            className="chip-btn chip-btn--aspect"
            disabled={disabled}
            onClick={() => setAspectOpen((v) => !v)}
          >
            {aspect}
          </button>
          {aspectOpen && (
            <div className="aspect-menu">
              {aspects.map((a) => (
                <button
                  key={a}
                  type="button"
                  className={a === aspect ? "is-active" : ""}
                  onClick={() => {
                    const first =
                      resolutionPresets.find((p) => (p.aspect || "1:1") === a && !p.render_width) ??
                      resolutionPresets.find((p) => (p.aspect || "1:1") === a);
                    if (first) onResolutionId(first.id);
                    setAspectOpen(false);
                  }}
                >
                  {a}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="popover-anchor" ref={resAnchor}>
          <button
            type="button"
            className="chip-btn"
            disabled={disabled}
            title={resolution?.label ?? resLabel}
            onClick={() => setResOpen((v) => !v)}
          >
            {resLabel}
          </button>
          {resOpen && (
            <div className="aspect-menu">
              {inAspect.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={p.id === resolutionId ? "is-active" : ""}
                  onClick={() => {
                    onResolutionId(p.id);
                    setResOpen(false);
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
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
