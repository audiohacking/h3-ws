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

export type LibraryPickPurpose = "refs" | "start_frame" | "end_frame";

export type LibraryPickItem = {
  kind: RefKind;
  path: string;
  name: string;
  durationS?: number;
  previewUrl?: string;
};

type Props = {
  open: boolean;
  initialTab?: LibraryTab;
  /** When attaching from Video/Renders, prefer keep-audio vs silent. */
  videoAttachKind?: "video" | "silent_video";
  /**
   * ``refs`` (default) = multi-select for Ref2VA.
   * ``start_frame`` / ``end_frame`` = single image pick for FL2VA anchors
   * (uploads + captured frames from the Image tab).
   */
  purpose?: LibraryPickPurpose;
  /** Current refs — drives the slot counter (ignored for frame picks). */
  refs: ReferenceItem[];
  disabled?: boolean;
  onClose: () => void;
  onUpload: (file: File, kind: "image" | "video" | "audio") => Promise<{
    path: string;
    durationS?: number;
    filename?: string;
    hasAudio?: boolean;
  }>;
  /** Attach selected library assets as new references (purpose=refs). */
  onAdd: (items: LibraryPickItem[]) => void;
  /** Single image chosen as start/end frame (purpose=start_frame|end_frame). */
  onPickImage?: (item: { path: string; name: string; previewUrl?: string }) => void;
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

export function MediaLibraryModal({
  open,
  initialTab = "image",
  videoAttachKind = "video",
  purpose = "refs",
  refs,
  disabled,
  onClose,
  onUpload,
  onAdd,
  onPickImage,
}: Props) {
  const framePick = purpose === "start_frame" || purpose === "end_frame";
  const [tab, setTab] = useState<LibraryTab>(framePick ? "image" : initialTab);
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<LibraryAsset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<LibraryAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const visibleTabs = useMemo(
    () => (framePick ? TABS.filter((t) => t.id === "image") : TABS),
    [framePick],
  );

  const capacity = useMemo((): Capacity => {
    if (framePick) {
      return { used: 0, max: 1, label: purpose === "end_frame" ? "end frame" : "start frame" };
    }
    return capacityForTab(tab, refs);
  }, [framePick, purpose, tab, refs]);
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
    const bootTab: LibraryTab = framePick ? "image" : initialTab;
    setTab(bootTab);
    setQuery("");
    setSelected([]);
    setError(null);
    void load(bootTab, "");
    const t = window.setTimeout(() => searchRef.current?.focus(), 50);
    return () => window.clearTimeout(t);
  }, [open, initialTab, framePick, load]);

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

  function toggle(asset: LibraryAsset) {
    if (framePick && asset.kind !== "image") return;
    setSelected((prev) => {
      const exists = prev.some((a) => a.id === asset.id);
      if (exists) return prev.filter((a) => a.id !== asset.id);
      if (framePick) return [asset];
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
        if (framePick) return [asset];
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
      const a = selected[0];
      onPickImage?.({
        path: a.path,
        name: a.name,
        previewUrl: a.thumb_url || a.media_url || undefined,
      });
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
  const confirmLabel = framePick
    ? purpose === "end_frame"
      ? "Use as end frame"
      : "Use as start frame"
    : `Add${selected.length > 0 ? ` (${selected.length})` : ""}`;
  const dialogLabel = framePick
    ? purpose === "end_frame"
      ? "Choose end frame"
      : "Choose start frame"
    : "Media library";

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
            placeholder="Search uploads and captured frames…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {tab !== "renders" && (
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
            multiple={!framePick}
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
                  if (!framePick || asset.kind !== "image") return;
                  onPickImage?.({
                    path: asset.path,
                    name: asset.name,
                    previewUrl: asset.thumb_url || asset.media_url || undefined,
                  });
                  onClose();
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
          <span className="media-lib__slots">
            {framePick
              ? selected.length === 0
                ? "Pick an image (uploads + captured frames)"
                : `Selected · ${selected[0]?.name ?? "image"}`
              : `${filled} / ${capacity.max} slots filled`}
          </span>
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
