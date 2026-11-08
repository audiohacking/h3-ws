/**
 * Continuity-style media library picker.
 * Tabs: Image / Video / Audio / Renders — recycle prior uploads or upload new.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { REF_LIMITS } from "../../config";
import type { RefKind, ReferenceItem } from "../../types";

export type LibraryTab = "image" | "video" | "audio" | "renders";

export type LibraryAsset = {
  id: string;
  path: string;
  name: string;
  kind: "image" | "video" | "audio";
  source: string;
  created_at?: string;
  duration_s?: number | null;
  has_audio?: boolean | null;
  thumb_url?: string | null;
  video_url?: string | null;
  media_url?: string | null;
  clip_id?: string;
};

type Capacity = { used: number; max: number; label: string };

export type LibraryPickPurpose =
  | "refs"
  | "start_frame"
  | "end_frame"
  | "source_video"
  | "paint_images"
  | "library_clip";

export type LibraryPickItem = {
  kind: RefKind;
  path: string;
  name: string;
  durationS?: number;
  previewUrl?: string;
};

export type LibraryPickedMedia = {
  path: string;
  name: string;
  previewUrl?: string;
  clipId?: string;
  durationS?: number;
};

type Props = {
  open: boolean;
  initialTab?: LibraryTab;
  /** When attaching from Video/Renders, prefer keep-audio vs silent. */
  videoAttachKind?: "video" | "silent_video";
  /**
   * ``refs`` (default) = multi-select for Ref2VA.
   * ``start_frame`` / ``end_frame`` = single image pick for FL2VA anchors.
   * ``source_video`` = single video (uploads + renders).
   * ``paint_images`` = multi image pick for recipe workspaces.
   * ``library_clip`` = single finished render for Enhance / library passes.
   */
  purpose?: LibraryPickPurpose;
  /** Current refs — drives the slot counter (ignored for single picks). */
  refs: ReferenceItem[];
  /** Cap for purpose=paint_images (default 4). */
  paintMax?: number;
  disabled?: boolean;
  onClose: () => void;
  onUpload: (file: File, kind: "image" | "video" | "audio") => Promise<{
    path: string;
    durationS?: number;
    filename?: string;
    hasAudio?: boolean;
  }>;
  /** Attach selected library assets as new references (purpose=refs|paint_images). */
  onAdd: (items: LibraryPickItem[]) => void;
  /** Single image chosen as start/end frame (purpose=start_frame|end_frame). */
  onPickImage?: (item: LibraryPickedMedia) => void;
  /** Single video / render chosen (purpose=source_video|library_clip). */
  onPickVideo?: (item: LibraryPickedMedia) => void;
};

const TABS: { id: LibraryTab; label: string }[] = [
  { id: "image", label: "Image" },
  { id: "video", label: "Video" },
  { id: "audio", label: "Audio" },
  { id: "renders", label: "Renders" },
];

function countSlots(refs: ReferenceItem[]) {
  let images = 0;
  let videos = 0;
  let audios = 0;
  let files = 0;
  for (const r of refs) {
    if (r.enabled === false) continue;
    if (r.kind === "image") images += 1;
    else if (r.kind === "audio") audios += 1;
    else videos += 1;
    files += r.kind === "video_audio" ? 2 : 1;
  }
  return { images, videos, audios, files };
}

function capacityForTab(tab: LibraryTab, refs: ReferenceItem[]): Capacity {
  const c = countSlots(refs);
  if (tab === "image") {
    return {
      used: c.images,
      max: Math.min(REF_LIMITS.MAX_IMAGES, REF_LIMITS.MAX_TOTAL_FILES - c.files + c.images),
      label: "image",
    };
  }
  if (tab === "audio") {
    return {
      used: c.audios,
      max: Math.min(REF_LIMITS.MAX_AUDIOS, REF_LIMITS.MAX_TOTAL_FILES - c.files + c.audios),
      label: "audio",
    };
  }
  // video + renders attach as silent/video refs
  return {
    used: c.videos,
    max: Math.min(REF_LIMITS.MAX_VIDEOS, REF_LIMITS.MAX_TOTAL_FILES - c.files + c.videos),
    label: "video",
  };
}

function acceptForTab(tab: LibraryTab): string {
  if (tab === "image") return "image/*";
  if (tab === "audio") return "audio/*";
  return "video/*";
}

function uploadKindForTab(tab: LibraryTab): "image" | "video" | "audio" {
  if (tab === "image") return "image";
  if (tab === "audio") return "audio";
  return "video";
}

function tabsForPurpose(purpose: LibraryPickPurpose): LibraryTab[] {
  if (purpose === "start_frame" || purpose === "end_frame" || purpose === "paint_images") {
    return ["image"];
  }
  if (purpose === "source_video") return ["video", "renders"];
  if (purpose === "library_clip") return ["renders"];
  return TABS.map((t) => t.id);
}

function bootTabForPurpose(purpose: LibraryPickPurpose, initialTab: LibraryTab): LibraryTab {
  const allowed = tabsForPurpose(purpose);
  if (allowed.includes(initialTab)) return initialTab;
  return allowed[0] ?? "image";
}

export function MediaLibraryModal({
  open,
  initialTab = "image",
  videoAttachKind = "video",
  purpose = "refs",
  refs,
  paintMax = 4,
  disabled,
  onClose,
  onUpload,
  onAdd,
  onPickImage,
  onPickVideo,
}: Props) {
  const framePick = purpose === "start_frame" || purpose === "end_frame";
  const paintPick = purpose === "paint_images";
  const videoPick = purpose === "source_video" || purpose === "library_clip";
  const singlePick = framePick || videoPick;

  const [tab, setTab] = useState<LibraryTab>(() => bootTabForPurpose(purpose, initialTab));
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<LibraryAsset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<LibraryAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const visibleTabs = useMemo(() => {
    const allowed = new Set(tabsForPurpose(purpose));
    return TABS.filter((t) => allowed.has(t.id));
  }, [purpose]);

  const capacity = useMemo((): Capacity => {
    if (framePick) {
      return { used: 0, max: 1, label: purpose === "end_frame" ? "end frame" : "start frame" };
    }
    if (purpose === "source_video") {
      return { used: 0, max: 1, label: "source video" };
    }
    if (purpose === "library_clip") {
      return { used: 0, max: 1, label: "library clip" };
    }
    if (paintPick) {
      const used = countSlots(refs).images;
      return { used, max: Math.max(used, paintMax), label: "painted frame" };
    }
    return capacityForTab(tab, refs);
  }, [framePick, paintPick, purpose, tab, refs, paintMax]);
  const slotsLeft = Math.max(0, capacity.max - capacity.used);
  const filled = capacity.used + selected.length;

  const load = useCallback(async (kind: LibraryTab, q: string) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ kind, q });
      const r = await fetch(`/api/library?${params}`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setItems((data.items ?? []) as LibraryAsset[]);
    } catch (e) {
      setError(String(e));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const bootTab = bootTabForPurpose(purpose, initialTab);
    setTab(bootTab);
    setQuery("");
    setSelected([]);
    setError(null);
    void load(bootTab, "");
    const t = window.setTimeout(() => searchRef.current?.focus(), 50);
    return () => window.clearTimeout(t);
  }, [open, initialTab, purpose, load]);

  useEffect(() => {
    if (!open) return;
    const handle = window.setTimeout(() => void load(tab, query), 180);
    return () => window.clearTimeout(handle);
  }, [open, tab, query, load]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  function assetMatchesPurpose(asset: LibraryAsset): boolean {
    if (framePick || paintPick) return asset.kind === "image";
    if (videoPick) return asset.kind === "video";
    return true;
  }

  function toPickedMedia(a: LibraryAsset): LibraryPickedMedia {
    return {
      path: a.path,
      name: a.name,
      previewUrl: a.thumb_url || a.video_url || a.media_url || undefined,
      clipId: a.clip_id,
      durationS: a.duration_s ?? undefined,
    };
  }

  function toggle(asset: LibraryAsset) {
    if (!assetMatchesPurpose(asset)) return;
    setSelected((prev) => {
      const exists = prev.some((a) => a.id === asset.id);
      if (exists) return prev.filter((a) => a.id !== asset.id);
      if (singlePick) return [asset];
      if (prev.length >= slotsLeft) return prev;
      return [...prev, asset];
    });
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);
    try {
      const kind = uploadKindForTab(tab === "renders" ? "video" : tab);
      const up = await onUpload(file, kind);
      const asset: LibraryAsset = {
        id: up.path,
        path: up.path,
        name: up.filename || file.name,
        kind,
        source: "upload",
        duration_s: up.durationS,
        has_audio: kind === "video" ? up.hasAudio ?? null : undefined,
        thumb_url: kind === "image" ? `/api/media?path=${encodeURIComponent(up.path)}` : null,
        video_url: kind === "video" ? `/api/media?path=${encodeURIComponent(up.path)}` : null,
        media_url: `/api/media?path=${encodeURIComponent(up.path)}`,
      };
      setItems((prev) => [asset, ...prev.filter((a) => a.path !== asset.path)]);
      setSelected((prev) => {
        if (singlePick) return [asset];
        if (prev.length >= slotsLeft) return prev;
        if (prev.some((a) => a.id === asset.id)) return prev;
        return [...prev, asset];
      });
      if (tab === "renders") setTab("video");
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  }

  function confirmAdd() {
    if (selected.length === 0) return;
    if (framePick) {
      onPickImage?.(toPickedMedia(selected[0]));
      onClose();
      return;
    }
    if (videoPick) {
      onPickVideo?.(toPickedMedia(selected[0]));
      onClose();
      return;
    }
    onAdd(
      selected.map((a) => {
        let kind: RefKind = a.kind as RefKind;
        if (a.kind === "video") {
          // Video-only files have no soundtrack for H3 — attach as silent_video.
          const silent =
            tab === "renders" ||
            videoAttachKind === "silent_video" ||
            a.has_audio === false;
          kind = silent ? "silent_video" : "video";
        }
        return {
          kind,
          path: a.path,
          name: a.name,
          durationS: a.duration_s ?? undefined,
          previewUrl: a.thumb_url || a.video_url || a.media_url || undefined,
        };
      }),
    );
    onClose();
  }

  const uploadLabel =
    tab === "image" ? "Upload image" : tab === "audio" ? "Upload audio" : "Upload video";
  const confirmLabel = (() => {
    if (purpose === "end_frame") return "Use as end frame";
    if (purpose === "start_frame") return "Use as start frame";
    if (purpose === "source_video") return "Use as source";
    if (purpose === "library_clip") return "Use this clip";
    if (paintPick) return `Add${selected.length > 0 ? ` (${selected.length})` : ""}`;
    return `Add${selected.length > 0 ? ` (${selected.length})` : ""}`;
  })();
  const dialogLabel = (() => {
    if (purpose === "end_frame") return "Choose end frame";
    if (purpose === "start_frame") return "Choose start frame";
    if (purpose === "source_video") return "Choose source video";
    if (purpose === "library_clip") return "Choose library clip";
    if (paintPick) return "Choose painted frames";
    return "Media library";
  })();
  const slotsHint = (() => {
    if (framePick) {
      return selected.length === 0
        ? "Pick an image (uploads + captured frames)"
        : `Selected · ${selected[0]?.name ?? "image"}`;
    }
    if (purpose === "source_video") {
      return selected.length === 0
        ? "Pick a video from uploads or renders"
        : `Selected · ${selected[0]?.name ?? "video"}`;
    }
    if (purpose === "library_clip") {
      return selected.length === 0
        ? "Pick a finished render"
        : `Selected · ${selected[0]?.name ?? "clip"}`;
    }
    if (paintPick) {
      return `${filled} / ${capacity.max} painted frames`;
    }
    return `${filled} / ${capacity.max} slots filled`;
  })();

  const allowUpload = tab !== "renders" && purpose !== "library_clip";

  return (
    <div className="media-lib-overlay" role="presentation" onClick={onClose}>
      <div
        className="media-lib"
        role="dialog"
        aria-modal="true"
        aria-label={dialogLabel}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="media-lib__head">
          <nav className="media-lib__tabs" aria-label="Media kind">
            {visibleTabs.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`media-lib__tab${tab === t.id ? " is-on" : ""}`}
                aria-selected={tab === t.id}
                onClick={() => {
                  setTab(t.id);
                  setSelected([]);
                }}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <button type="button" className="media-lib__close" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </header>

        <div className="media-lib__bar">
          <input
            ref={searchRef}
            className="media-lib__search"
            type="search"
            placeholder={
              purpose === "library_clip"
                ? "Search finished renders…"
                : "Search uploads and captured frames…"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {allowUpload && (
            <button
              type="button"
              className="media-lib__upload"
              disabled={disabled || uploading}
              onClick={() => fileRef.current?.click()}
            >
              + {uploading ? "Uploading…" : uploadLabel}
            </button>
          )}
          <input
            ref={fileRef}
            type="file"
            accept={acceptForTab(tab === "renders" ? "video" : tab)}
            hidden
            multiple={!singlePick}
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              e.target.value = "";
              void (async () => {
                for (const f of files) await handleUpload(f);
              })();
            }}
          />
        </div>

        <div className="media-lib__grid">
          {loading && items.length === 0 && <p className="media-lib__empty">Loading…</p>}
          {!loading && items.length === 0 && (
            <p className="media-lib__empty">
              {error
                ? error
                : tab === "renders"
                  ? "No finished renders yet."
                  : `No ${tab} files yet — upload one.`}
            </p>
          )}
          {items.map((asset) => {
            const on = selected.some((a) => a.id === asset.id);
            const blocked = !on && selected.length >= slotsLeft;
            return (
              <button
                key={asset.id}
                type="button"
                className={`media-lib__cell${on ? " is-on" : ""}${blocked ? " is-blocked" : ""}`}
                disabled={blocked}
                title={asset.name}
                onClick={() => toggle(asset)}
                onDoubleClick={() => {
                  if (!assetMatchesPurpose(asset)) return;
                  if (framePick) {
                    onPickImage?.(toPickedMedia(asset));
                    onClose();
                    return;
                  }
                  if (videoPick) {
                    onPickVideo?.(toPickedMedia(asset));
                    onClose();
                  }
                }}
              >
                {asset.kind === "image" && asset.thumb_url ? (
                  <img className="media-lib__thumb" src={asset.thumb_url} alt="" loading="lazy" />
                ) : asset.thumb_url ? (
                  <img className="media-lib__thumb" src={asset.thumb_url} alt="" loading="lazy" />
                ) : asset.kind === "video" && (asset.video_url || asset.media_url) ? (
                  <video
                    className="media-lib__thumb"
                    src={`${asset.video_url || asset.media_url || ""}#t=0.1`}
                    muted
                    playsInline
                    preload="metadata"
                  />
                ) : (
                  <span className="media-lib__thumb media-lib__thumb--glyph" aria-hidden>
                    {asset.kind === "audio" ? "♪" : "▶"}
                  </span>
                )}
                <span className="media-lib__name">{asset.name}</span>
              </button>
            );
          })}
        </div>

        <footer className="media-lib__foot">
          <span className="media-lib__slots">{slotsHint}</span>
          <button type="button" className="media-lib__ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="media-lib__add"
            disabled={selected.length === 0 || disabled}
            onClick={confirmAdd}
          >
            {confirmLabel}
          </button>
        </footer>
      </div>
    </div>
  );
}
