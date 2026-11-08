import { useMemo, useState } from "react";
import { MediaLibraryModal } from "../composer/MediaLibraryModal";
import type { Clip, WorkspaceManifest } from "../../types";

type Props = {
  manifest?: WorkspaceManifest | null;
  clips: Clip[];
  selectedClipId: string | null;
  onSelectClip: (id: string | null) => void;
  upframe: boolean;
  onUpframe: (on: boolean) => void;
  upframeMultiplier: number;
  onUpframeMultiplier: (n: number) => void;
  redetail: boolean;
  onRedetail: (on: boolean) => void;
  redetailDenoise: number;
  onRedetailDenoise: (n: number) => void;
  faces: boolean;
  onFaces: (on: boolean) => void;
  facesDenoise: number;
  onFacesDenoise: (n: number) => void;
  facesSeed: number;
  onFacesSeed: (n: number) => void;
  facesCanvas: number;
  busy: boolean;
  canRun: boolean;
  onRun: () => void;
  status?: string | null;
  onUpload: (
    file: File,
    kind: "image" | "video" | "audio",
  ) => Promise<{ path: string; filename?: string; durationS?: number; hasAudio?: boolean }>;
};

function clipLabel(c: Clip): string {
  const raw = (c.prompt || c.id || "clip").trim();
  return raw.length > 48 ? `${raw.slice(0, 45)}…` : raw;
}

export function EnhancePanel({
  manifest,
  clips,
  selectedClipId,
  onSelectClip,
  upframe,
  onUpframe,
  upframeMultiplier,
  onUpframeMultiplier,
  redetail,
  onRedetail,
  redetailDenoise,
  onRedetailDenoise,
  faces,
  onFaces,
  facesDenoise,
  onFacesDenoise,
  facesSeed,
  onFacesSeed,
  facesCanvas,
  busy,
  canRun,
  onRun,
  status,
  onUpload,
}: Props) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const selected = useMemo(
    () => clips.find((c) => c.id === selectedClipId) ?? null,
    [clips, selectedClipId],
  );
  const label = selected ? clipLabel(selected) : null;
  const title = manifest?.label || "Enhance";
  const sub =
    manifest?.hint ||
    "Faces, ReDetail, and Upframe on a finished library clip — each writes a new clip.";
  const anyPass = faces || redetail || upframe;

  return (
    <div className="focus-panel enhance-panel">
      <div className="focus-panel__head">
        <h2 className="focus-panel__title">{title}</h2>
        <p className="focus-panel__sub">{sub}</p>
      </div>

      <div className="focus-field">
        <div className="focus-field__label">Library clip</div>
        <div className="focus-field__row">
          <div className="frame-chip-group">
            <button
              type="button"
              className={`chip-btn${selected ? " is-on" : ""}`}
              disabled={busy}
              title={label ?? "Choose a finished render"}
              onClick={() => setPickerOpen(true)}
            >
              {label ? `Clip · ${label}` : "Choose clip…"}
            </button>
            {selected ? (
              <button
                type="button"
                className="chip-btn chip-btn--clear"
                disabled={busy}
                title="Clear clip"
                aria-label="Clear clip"
                onClick={() => onSelectClip(null)}
              >
                ×
              </button>
            ) : null}
          </div>
          {selected?.video_url ? (
            <video
              className="focus-thumb"
              src={`${selected.video_url}#t=0.1`}
              muted
              playsInline
              preload="metadata"
            />
          ) : null}
        </div>
      </div>

      <div className="focus-pass-grid">
        <div className={`focus-pass${faces ? " is-on" : ""}`}>
          <label className="focus-check">
            <input
              type="checkbox"
              checked={faces}
              disabled={busy}
              onChange={(e) => onFaces(e.target.checked)}
            />
            <span>
              <strong>Faces</strong>
              <em>SAM soft-hold refine · {facesCanvas}p</em>
            </span>
          </label>
          {faces ? (
            <div className="focus-pass__controls">
              <label className="focus-inline">
                denoise
                <input
                  type="number"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={facesDenoise}
                  disabled={busy}
                  onChange={(e) => onFacesDenoise(Number(e.target.value) || 0.45)}
                />
              </label>
              <label className="focus-inline">
                seed
                <input
                  type="number"
                  min={0}
                  value={facesSeed}
                  disabled={busy}
                  onChange={(e) => onFacesSeed(Math.max(0, Math.round(Number(e.target.value) || 0)))}
                />
              </label>
            </div>
          ) : null}
        </div>

        <div className={`focus-pass${redetail ? " is-on" : ""}`}>
          <label className="focus-check">
            <input
              type="checkbox"
              checked={redetail}
              disabled={busy}
              onChange={(e) => onRedetail(e.target.checked)}
            />
            <span>
              <strong>ReDetail</strong>
              <em>×2 H3 refine pass</em>
            </span>
          </label>
          {redetail ? (
            <label className="focus-inline">
              denoise
              <input
                type="number"
                min={0.05}
                max={0.95}
                step={0.05}
                value={redetailDenoise}
                disabled={busy}
                onChange={(e) => onRedetailDenoise(Number(e.target.value) || 0.45)}
              />
            </label>
          ) : null}
        </div>

        <div className={`focus-pass${upframe ? " is-on" : ""}`}>
          <label className="focus-check">
            <input
              type="checkbox"
              checked={upframe}
              disabled={busy}
              onChange={(e) => onUpframe(e.target.checked)}
            />
            <span>
              <strong>Upframe</strong>
              <em>FILM temporal interpolate</em>
            </span>
          </label>
          {upframe ? (
            <label className="focus-inline">
              ×
              <input
                type="number"
                min={2}
                max={4}
                step={1}
                value={upframeMultiplier}
                disabled={busy}
                onChange={(e) =>
                  onUpframeMultiplier(Math.max(2, Math.round(Number(e.target.value) || 2)))
                }
              />
            </label>
          ) : null}
        </div>
      </div>

      {status ? <p className="focus-panel__status">{status}</p> : null}
      {!selected ? (
        <p className="focus-panel__status">Pick a finished clip from the library.</p>
      ) : !anyPass ? (
        <p className="focus-panel__status">Enable Faces, ReDetail, and/or Upframe to run.</p>
      ) : (
        <p className="focus-panel__status">
          Runs in order: Faces → ReDetail → Upframe. Progress shows on the player; originals stay.
        </p>
      )}

      <div className="focus-actions">
        <button type="button" className="btn-primary" disabled={!canRun || busy} onClick={onRun}>
          {busy ? "Working…" : "Run enhance"}
        </button>
      </div>

      <MediaLibraryModal
        open={pickerOpen}
        purpose="library_clip"
        initialTab="renders"
        refs={[]}
        disabled={busy}
        onClose={() => setPickerOpen(false)}
        onUpload={onUpload}
        onAdd={() => undefined}
        onPickVideo={(item) => {
          if (item.clipId) onSelectClip(item.clipId);
          setPickerOpen(false);
        }}
      />
    </div>
  );
}
