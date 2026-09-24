/**
 * Continuity-style asset capsules (card / pool chip face).
 *
 * Face of an ordinary image:  [thumb] @img-1  ✎  max  ⊘  ✕
 * Face of a clip with sound:  [▶]    @vid-1  sound on  ⊘  ✕
 *
 * No filename on the chip (it lives in the title). No word "whole" —
 * framing is the edit icon, lit only when something is set. Mute is the
 * circle-slash glyph beside ✕, same as Continuity.
 */
import { useState, type ReactNode } from "react";
import { handleForRef, kindHandlePrefix } from "../../compile";
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

function trackForKind(kind: RefKind): TrackChoice | undefined {
  if (kind === "silent_video") return "picture";
  if (kind === "audio") return "sound";
  if (kind === "video" || kind === "video_audio") return "picture+sound";
  return undefined;
}

/** Continuity referenceSummary — empty for an ordinary file. */
function saidLabel(ref: ReferenceItem): string {
  const parts: string[] = [];
  if (isTimed(ref.kind) && ref.trim) parts.push(trimLabel(ref.trim));
  if (isCroppable(ref.kind)) {
    const framed = cropLabel(ref.crop);
    if (framed) parts.push(framed);
  }
  // Track only when not the default "sound on" for video+audio kinds.
  if (ref.kind === "silent_video") parts.push("silent");
  else if (ref.kind === "audio") parts.push("sound only");
  // video / video_audio default is sound on — say nothing (Continuity rule).
  return parts.join(" · ");
}

function EditIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z" />
    </svg>
  );
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
            const said = saidLabel(ref);
            const framed = isFramed(ref.crop);
            const isImage = ref.kind === "image";
            const tag = i % 6;
            const thumbSrc =
              kindHandlePrefix(ref.kind) === "img"
                ? ref.previewUrl || mediaSrc(ref.path)
                : null;

            const editTitle = isImage
              ? framed
                ? `${cropLabel(ref.crop)} — press to change the framing`
                : "Used whole — press to crop, turn or mirror"
              : said
                ? `${said} — press to change the segment`
                : "Whole clip — press to set the in/out range";

            return (
              <div
                key={ref.id}
                className={`asset-capsule asset-capsule--tag-${tag}${muted ? " asset-capsule--muted" : ""}`}
                title={ref.name}
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

                <span className="asset-capsule__handle">{handle}</span>

                {isCroppable(ref.kind) && isImage && (
                  <Mark on={framed} disabled={disabled} title={editTitle} onClick={() => openSegment(i)}>
                    <EditIcon />
                  </Mark>
                )}

                {isTimed(ref.kind) && (
                  <Mark
                    on={Boolean(ref.trim)}
                    disabled={disabled}
                    title={editTitle}
                    onClick={() => openSegment(i)}
                  >
                    <EditIcon />
                  </Mark>
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

                {said ? <span className="asset-capsule__said">{said}</span> : null}

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
