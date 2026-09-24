import { useState } from "react";
import { MediaLibraryModal, type LibraryTab } from "./MediaLibraryModal";
import type { RefKind, ReferenceItem } from "../../types";

type VideoKind = "video" | "silent_video" | "video_audio";

type Props = {
  disabled?: boolean;
  loraCount: number;
  castSlot: React.ReactNode;
  librarySlot?: React.ReactNode;
  presetSlot: React.ReactNode;
  refineEnabled?: boolean;
  onRefine?: () => void;
  onOpenFeatures?: () => void;
  refs: ReferenceItem[];
  onUpload: (file: File, kind: "image" | "video" | "audio") => Promise<{
    path: string;
    durationS?: number;
    filename?: string;
  }>;
  onAddFromLibrary: (items: Array<{
    kind: RefKind;
    path: string;
    name: string;
    durationS?: number;
    previewUrl?: string;
  }>) => void;
  /** Kept for video+replacement-audio two-file flow from the menu. */
  onAddVideoAudio: (video: File, audio: File) => void;
  onOpenLora: () => void;
  onClear: () => void;
};

export function ComposerRail({
  disabled,
  loraCount,
  castSlot,
  librarySlot,
  presetSlot,
  refineEnabled,
  onRefine,
  onOpenFeatures,
  refs,
  onUpload,
  onAddFromLibrary,
  onAddVideoAudio,
  onOpenLora,
  onClear,
}: Props) {
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryTab, setLibraryTab] = useState<LibraryTab>("image");
  const [videoAttachKind, setVideoAttachKind] = useState<"video" | "silent_video">("video");
  const [videoMenu, setVideoMenu] = useState(false);

  function openLibrary(tab: LibraryTab, attach: "video" | "silent_video" = "video") {
    setLibraryTab(tab);
    setVideoAttachKind(attach);
    setLibraryOpen(true);
    setVideoMenu(false);
  }

  return (
    <div className="composer-rail">
      <div className="composer-rail__group">
        <button type="button" className="rail-btn" disabled={disabled} onClick={() => openLibrary("image")}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <circle cx="8.5" cy="10" r="1.5" />
            <path d="M21 16l-5-5-11 8" />
          </svg>
          Add image
        </button>
        <div className="popover-anchor">
          <button
            type="button"
            className="rail-btn rail-btn--menu"
            disabled={disabled}
            onClick={() => setVideoMenu((v) => !v)}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <rect x="3" y="6" width="14" height="12" rx="2" />
              <path d="M17 10l5-3v10l-5-3" />
            </svg>
            Add video
          </button>
          {videoMenu && (
            <div className="rail-menu">
              <button type="button" onClick={() => openLibrary("video", "video")}>
                Video (keep audio)
              </button>
              <button type="button" onClick={() => openLibrary("video", "silent_video")}>
                Silent video / from library
              </button>
              <button type="button" onClick={() => openLibrary("renders", "silent_video")}>
                From renders
              </button>
              <button
                type="button"
                onClick={() => {
                  setVideoMenu(false);
                  // Two-file pick stays as native inputs via parent helper.
                  const v = document.createElement("input");
                  v.type = "file";
                  v.accept = "video/*";
                  v.onchange = () => {
                    const video = v.files?.[0];
                    if (!video) return;
                    const a = document.createElement("input");
                    a.type = "file";
                    a.accept = "audio/*";
                    a.onchange = () => {
                      const audio = a.files?.[0];
                      if (audio) onAddVideoAudio(video, audio);
                    };
                    a.click();
                  };
                  v.click();
                }}
              >
                Video + replacement audio
              </button>
            </div>
          )}
        </div>
        <button type="button" className="rail-btn" disabled={disabled} onClick={() => openLibrary("audio")}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M9 18V6l12-2v12" />
            <circle cx="6" cy="18" r="2.5" />
            <circle cx="18" cy="16" r="2.5" />
          </svg>
          Add audio
        </button>
        <button type="button" className="rail-btn" disabled={disabled} onClick={onOpenLora}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M12 3l1.8 5.4H19l-4.4 3.2 1.7 5.4L12 14.8 7.7 17l1.7-5.4L5 8.4h5.2z" />
          </svg>
          Add LoRA
          {loraCount > 0 && <span className="rail-btn__badge">{loraCount}</span>}
        </button>
        {onRefine && (
          <button
            type="button"
            className="rail-btn"
            disabled={disabled || !refineEnabled}
            onClick={onRefine}
            title={refineEnabled ? "Rewrite prompt via remote Refine" : "Enable Refine in Features"}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M4 19h16M7 15l3-9h4l3 9M9 11h6" />
            </svg>
            Refine
          </button>
        )}
        {onOpenFeatures && (
          <button type="button" className="rail-btn" disabled={disabled} onClick={onOpenFeatures}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
            Features
          </button>
        )}
        {castSlot}
        {librarySlot}
        <button type="button" className="rail-btn" disabled={disabled} onClick={onClear}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M4 7h16M9 7V5h6v2M8 7l1 12h6l1-12" />
          </svg>
          Clear
        </button>
      </div>
      <div className="composer-rail__group">{presetSlot}</div>

      <MediaLibraryModal
        open={libraryOpen}
        initialTab={libraryTab}
        videoAttachKind={videoAttachKind}
        refs={refs}
        disabled={disabled}
        onClose={() => setLibraryOpen(false)}
        onUpload={onUpload}
        onAdd={onAddFromLibrary}
      />
    </div>
  );
}

export type { VideoKind };
