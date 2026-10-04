/**
 * Continuity-style asset capsules (card / pool chip face).
 *
 * Face of an ordinary image:  [thumb] @img-1  whole  max  ⊘  ✕
 * Face of a clip with sound:  [▶]    @vid-1  whole  ⊞  ⊘  ✕
 *
 * Thumb / glyph opens the segment door (picture or trim editor) so the
 * reference can always be previewed and locked after add — same editors as
 * Continuity, not a third lightbox. The door label always shows ("whole",
 * "0:02–0:07", "cropped · ↻ 90°").
 */
import { useState, type ReactNode } from "react";
import { handleForRef } from "../../compile";
import { cropLabel, isFramed, mediaSrc, trimLabel } from "../../crop";
import type { ImageCrop, MediaTrim, ReferenceItem, RefKind, RefSize } from "../../types";
import { PictureEditor } from "../media/PictureEditor";
import { TrimEditor, type TrackChoice } from "../media/TrimEditor";

type Props = {
  refs: ReferenceItem[];
  disabled?: boolean;
  onChange: (refs: ReferenceItem[]) => void;
  shotAspect?: { ratio: number; label: string } | null;
  cardSeconds?: number | null;
};

type Editor =
  | { mode: "picture"; index: number }
  | { mode: "trim"; index: number }
  | null;

function isTimed(kind: RefKind): boolean {
  return kind === "audio" || kind === "silent_video" || kind === "video" || kind === "video_audio";
}

function isCroppable(kind: RefKind): boolean {
  return kind === "image" || kind === "silent_video" || kind === "video" || kind === "video_audio";
}

function isSizeable(kind: RefKind): boolean {
  return kind === "image";
}

/** Library may stash a poster in previewUrl; ignore raw video media URLs for <img>. */
function isPosterUrl(url: string): boolean {
  if (/\.(mp4|webm|mov|m4v)(\?|$)/i.test(url)) return false;
  if (url.includes("/api/media?")) return false;
  return true;
}

function trackForKind(kind: RefKind): TrackChoice | undefined {
  if (kind === "silent_video") return "picture";
  if (kind === "audio") return "sound";
  if (kind === "video" || kind === "video_audio") return "picture+sound";
  return undefined;
}

function CropIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M6 2v14a2 2 0 0 0 2 2h14" />
      <path d="M2 6h14a2 2 0 0 1 2 2v14" />
    </svg>
  );
}

function MuteIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      <circle cx="12" cy="12" r="9" />
      <path d="M5.6 5.6l12.8 12.8" />
    </svg>
  );
}

function Mark({
  on,
  disabled,
  title,
  onClick,
  children,
  className = "",
}: {
  on?: boolean;
  disabled?: boolean;
  title: string;
  onClick: () => void;
  children: ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={`asset-capsule__mark${on ? " asset-capsule__mark--on" : ""}${className ? ` ${className}` : ""}`}
      disabled={disabled}
      title={title}
      aria-pressed={on ? "true" : "false"}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function AssetCapsules({ refs, disabled, onChange, shotAspect, cardSeconds }: Props) {
  const [editor, setEditor] = useState<Editor>(null);

  if (refs.length === 0 && !editor) return null;

  function patch(index: number, next: Partial<ReferenceItem>) {
    onChange(refs.map((r, i) => (i === index ? { ...r, ...next } : r)));
  }

  function openSegment(index: number) {
    const ref = refs[index];
    if (ref.kind === "image") setEditor({ mode: "picture", index });
    else if (isTimed(ref.kind)) setEditor({ mode: "trim", index });
  }

  const editing = editor ? refs[editor.index] : null;

  return (
    <>
      {refs.length > 0 && (
        <div className="asset-row">
          {refs.map((ref, i) => {
            const handle = handleForRef(refs, i);
            const muted = ref.enabled === false;
            const size: RefSize = ref.refSize ?? "max";
            const framed = isFramed(ref.crop);
            const isImage = ref.kind === "image";
            const timed = isTimed(ref.kind);
            const tag = i % 6;
            // Images: file itself. Video: poster only when previewUrl is not the media file.
            const thumbSrc = isImage
              ? ref.previewUrl || mediaSrc(ref.path)
              : ref.kind !== "audio" && ref.previewUrl && isPosterUrl(ref.previewUrl)
                ? ref.previewUrl
                : null;

            const doorLabel = isImage
              ? cropLabel(ref.crop) || "whole"
              : timed
                ? trimLabel(ref.trim)
                : "";
            const doorOn = isImage ? framed : Boolean(ref.trim);
            const doorTitle = isImage
              ? framed
                ? `${cropLabel(ref.crop)} — press to change the framing`
                : "Used whole — press to preview, crop, turn or mirror"
              : ref.trim
                ? `${trimLabel(ref.trim)} — press to change the segment`
                : "Whole clip — press to preview and set the in/out range";
            const thumbTitle = isImage
              ? "Preview and frame this picture"
              : timed
                ? "Preview and set the in/out range"
                : ref.name;

            return (
              <div
                key={ref.id}
                className={`asset-capsule asset-capsule--tag-${tag}${muted ? " asset-capsule--muted" : ""}`}
                title={ref.name}
              >
                <button
                  type="button"
                  className="asset-capsule__thumb-btn"
                  disabled={disabled}
                  title={thumbTitle}
                  aria-label={thumbTitle}
                  onClick={() => openSegment(i)}
                >
                  {thumbSrc ? (
                    <img className="asset-capsule__thumb" src={thumbSrc} alt="" />
                  ) : (
                    <span className="asset-capsule__thumb asset-capsule__thumb--glyph" aria-hidden>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                        {ref.kind === "audio" ? (
                          <path d="M9 18V6l12-2v12M6 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm12-2a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" />
                        ) : (
                          <path d="M8 5.5l11 6.5-11 6.5z" />
                        )}
                      </svg>
                    </span>
                  )}
                </button>

                <span className="asset-capsule__handle">{handle}</span>

                {(isImage || timed) && (
                  <button
                    type="button"
                    className={`asset-capsule__door${doorOn ? " asset-capsule__door--on" : ""}`}
                    disabled={disabled}
                    title={doorTitle}
                    onClick={() => openSegment(i)}
                  >
                    {doorLabel}
                  </button>
                )}

                {isCroppable(ref.kind) && !isImage && (
                  <Mark
                    on={framed}
                    disabled={disabled}
                    title={
                      framed
                        ? `${cropLabel(ref.crop)} — press to change the framing`
                        : "Crop, turn or mirror the picture"
                    }
                    onClick={() => setEditor({ mode: "picture", index: i })}
                  >
                    <CropIcon />
                  </Mark>
                )}

                {framed && !isImage && (
                  <span className="asset-capsule__said">{cropLabel(ref.crop)}</span>
                )}

                {isSizeable(ref.kind) && (
                  <button
                    type="button"
                    className="asset-capsule__ghost"
                    disabled={disabled}
                    title={
                      size === "max"
                        ? "max: encode at original size — better identity, slower. Click for match."
                        : "match: scale to the generation's pixel area. Click for max."
                    }
                    onClick={() => patch(i, { refSize: size === "max" ? "match" : "max" })}
                  >
                    {size}
                  </button>
                )}

                <Mark
                  on={muted}
                  disabled={disabled}
                  className="asset-capsule__mute"
                  title={
                    muted
                      ? `${handle} is muted — out of the run, kept as attached. Click to bring it back.`
                      : `Mute ${handle}: out of the run, but the file and framing stay.`
                  }
                  onClick={() => patch(i, { enabled: muted })}
                >
                  <MuteIcon />
                </Mark>

                <button
                  type="button"
                  className="asset-capsule__x"
                  disabled={disabled}
                  aria-label={`Remove ${handle}`}
                  title={`Remove ${handle}`}
                  onClick={() => onChange(refs.filter((_, j) => j !== i))}
                >
                  ✕
                </button>
              </div>
            );
          })}
        </div>
      )}

      {editor?.mode === "picture" && editing && isCroppable(editing.kind) && (
        <PictureEditor
          path={editing.path}
          name={editing.name}
          kind={editing.kind === "audio" ? "image" : editing.kind}
          crop={editing.crop}
          trim={editing.trim}
          aspect={shotAspect}
          previewUrl={editing.previewUrl}
          onCancel={() => setEditor(null)}
          onUse={(crop: ImageCrop | null) => {
            patch(editor.index, { crop });
            setEditor(null);
          }}
        />
      )}

      {editor?.mode === "trim" && editing && isTimed(editing.kind) && (
        <TrimEditor
          path={editing.path}
          name={editing.name}
          kind={editing.kind as "audio" | "video" | "silent_video" | "video_audio"}
          trim={editing.trim}
          showTrack={editing.kind !== "audio"}
          track={trackForKind(editing.kind)}
          cardSeconds={cardSeconds}
          previewUrl={editing.previewUrl}
          durationHint={editing.durationS}
          onCancel={() => setEditor(null)}
          onUse={({ trim, kind: nextKind }: { trim: MediaTrim | null; kind?: RefKind }) => {
            const next: Partial<ReferenceItem> = { trim };
            if (nextKind && nextKind !== editing.kind) next.kind = nextKind;
            patch(editor.index, next);
            setEditor(null);
          }}
        />
      )}
    </>
  );
}
