/**
 * Generic host for ``panel: "recipe"`` workspace manifests.
 * Specialty modes ship as JSON + LoRA recipes; this renders their fields.
 */
import { useMemo, useState } from "react";
import {
  MediaLibraryModal,
  type LibraryPickPurpose,
} from "../composer/MediaLibraryModal";
import type { ReferenceItem, WorkspaceField, WorkspaceManifest } from "../../types";

export type RecipeDraft = Record<string, unknown>;

type Props = {
  manifest: WorkspaceManifest;
  draft: RecipeDraft;
  onChange: (patch: RecipeDraft) => void;
  busy: boolean;
  canSubmit: boolean;
  onGenerate: () => void;
  onUpload: (
    file: File,
    kind: "image" | "video" | "audio",
  ) => Promise<{ path: string; filename?: string; durationS?: number; hasAudio?: boolean }>;
  status?: string | null;
};

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asBool(v: unknown, fallback = false): boolean {
  return typeof v === "boolean" ? v : fallback;
}

type MediaVal = { path: string; name: string; previewUrl?: string; clipId?: string };

function asMedia(v: unknown): MediaVal | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  if (typeof o.path !== "string" || !o.path) return null;
  return {
    path: o.path,
    name: typeof o.name === "string" ? o.name : o.path,
    previewUrl: typeof o.previewUrl === "string" ? o.previewUrl : undefined,
    clipId: typeof o.clipId === "string" ? o.clipId : undefined,
  };
}

function asMediaList(v: unknown): MediaVal[] {
  return Array.isArray(v) ? (v.map(asMedia).filter(Boolean) as MediaVal[]) : [];
}

export function RecipePanel({
  manifest,
  draft,
  onChange,
  busy,
  canSubmit,
  onGenerate,
  onUpload,
  status,
}: Props) {
  const [picker, setPicker] = useState<{
    purpose: LibraryPickPurpose;
    fieldId: string;
  } | null>(null);

  const fields = useMemo(() => manifest.fields ?? [], [manifest.fields]);

  const paintRefs: ReferenceItem[] = useMemo(() => {
    if (!picker) return [];
    const field = fields.find((f) => f.id === picker.fieldId);
    if (field?.type !== "library_images") return [];
    return asMediaList(draft[picker.fieldId]).map((m, i) => ({
      id: `recipe-${picker.fieldId}-${i}`,
      kind: "image" as const,
      path: m.path,
      name: m.name,
      previewUrl: m.previewUrl,
    }));
  }, [picker, fields, draft]);

  function renderField(field: WorkspaceField) {
    const label = field.label || field.id;
    if (field.type === "segment" && field.options?.length) {
      const cur = asString(draft[field.id]) || field.options[0];
      return (
        <div key={field.id} className="focus-field">
          <div className="focus-field__label">{label}</div>
          <div className="focus-seg" role="group">
            {field.options.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`focus-seg__btn${cur === opt ? " is-on" : ""}`}
                disabled={busy}
                onClick={() => onChange({ [field.id]: opt })}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>
      );
    }
    if (field.type === "textarea") {
      return (
        <div key={field.id} className="focus-field">
          <div className="focus-field__label">{label}</div>
          <textarea
            className="focus-prompt"
            rows={8}
            value={asString(draft[field.id])}
            disabled={busy}
            placeholder={field.placeholder || ""}
            onChange={(e) => onChange({ [field.id]: e.target.value })}
          />
        </div>
      );
    }
    if (field.type === "toggle") {
      return (
        <label key={field.id} className="focus-check">
          <input
            type="checkbox"
            checked={asBool(draft[field.id], Boolean(field.default))}
            disabled={busy}
            onChange={(e) => onChange({ [field.id]: e.target.checked })}
          />
          {label}
        </label>
      );
    }
    if (field.type === "library_video" || field.type === "library_clip") {
      const media = asMedia(draft[field.id]);
      const purpose: LibraryPickPurpose =
        field.type === "library_clip" ? "library_clip" : "source_video";
      return (
        <div key={field.id} className="focus-field">
          <div className="focus-field__label">{label}</div>
          <div className="focus-field__row">
            <div className="frame-chip-group">
              <button
                type="button"
                className={`chip-btn${media ? " is-on" : ""}`}
                disabled={busy}
                onClick={() => setPicker({ purpose, fieldId: field.id })}
              >
                {media ? media.name : `Choose ${label.toLowerCase()}…`}
              </button>
              {media ? (
                <button
                  type="button"
                  className="chip-btn chip-btn--clear"
                  disabled={busy}
                  aria-label={`Clear ${label}`}
                  onClick={() => onChange({ [field.id]: null })}
                >
                  ×
                </button>
              ) : null}
            </div>
            {media?.previewUrl ? (
              <video
                className="focus-thumb"
                src={`${media.previewUrl}#t=0.1`}
                muted
                playsInline
                preload="metadata"
              />
            ) : null}
          </div>
        </div>
      );
    }
    if (field.type === "library_images") {
      const items = asMediaList(draft[field.id]);
      const max = field.max ?? 4;
      return (
        <div key={field.id} className="focus-field">
          <div className="focus-field__label">{label}</div>
          <ul className="focus-list">
            {items.map((item, index) => (
              <li key={`${item.path}-${index}`} className="focus-list__row">
                {item.previewUrl ? (
                  <img className="focus-thumb focus-thumb--sm" src={item.previewUrl} alt="" />
                ) : (
                  <span className="focus-thumb focus-thumb--sm focus-thumb--empty" />
                )}
                <span className="focus-list__name">{item.name}</span>
                <button
                  type="button"
                  className="chip-btn chip-btn--clear"
                  disabled={busy}
                  aria-label={`Remove ${item.name}`}
                  onClick={() =>
                    onChange({
                      [field.id]: items.filter((_, i) => i !== index),
                    })
                  }
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          {items.length < max ? (
            <button
              type="button"
              className="chip-btn"
              disabled={busy}
              onClick={() => setPicker({ purpose: "paint_images", fieldId: field.id })}
            >
              Add image…
            </button>
          ) : null}
        </div>
      );
    }
    return null;
  }

  return (
    <div className="focus-panel recipe-panel">
      <div className="focus-panel__head">
        <h2 className="focus-panel__title">{manifest.label}</h2>
        {manifest.hint ? <p className="focus-panel__sub">{manifest.hint}</p> : null}
      </div>
      {fields.map(renderField)}
      {status ? <p className="focus-panel__status">{status}</p> : null}
      <div className="focus-actions">
        <button
          type="button"
          className="btn-primary"
          disabled={!canSubmit || busy}
          onClick={onGenerate}
        >
          {busy ? "Working…" : "Generate"}
        </button>
      </div>
      <MediaLibraryModal
        open={picker != null}
        purpose={picker?.purpose ?? "source_video"}
        initialTab={picker?.purpose === "paint_images" ? "image" : "video"}
        refs={paintRefs}
        paintMax={
          fields.find((f) => f.id === picker?.fieldId)?.max ?? 4
        }
        disabled={busy}
        onClose={() => setPicker(null)}
        onUpload={onUpload}
        onAdd={(items) => {
          if (!picker || picker.purpose !== "paint_images") return;
          const prev = asMediaList(draft[picker.fieldId]);
          const max = fields.find((f) => f.id === picker.fieldId)?.max ?? 4;
          onChange({
            [picker.fieldId]: [
              ...prev,
              ...items.map((it) => ({
                path: it.path,
                name: it.name,
                previewUrl: it.previewUrl,
              })),
            ].slice(0, max),
          });
          setPicker(null);
        }}
        onPickVideo={(item) => {
          if (!picker) return;
          onChange({
            [picker.fieldId]: {
              path: item.path,
              name: item.name,
              previewUrl: item.previewUrl,
              clipId: item.clipId,
            },
          });
          setPicker(null);
        }}
      />
    </div>
  );
}
