import { useMemo, useRef } from "react";
import type { LoraPreset, ReferenceItem } from "../../types";

export type RemakeMode = "swap" | "edit";

export type RemakeDraft = {
  mode: RemakeMode;
  prompt: string;
  sourcePath: string | null;
  sourceName: string | null;
  painted: Array<{ path: string; name: string; pin: string }>;
};

type Props = {
  draft: RemakeDraft;
  onChange: (next: Partial<RemakeDraft>) => void;
  busy: boolean;
  canSubmit: boolean;
  onGenerate: () => void;
  onUpload: (file: File, kind: "image" | "video") => Promise<{ path: string; filename?: string }>;
  swapPreset: LoraPreset | null;
  editPreset: LoraPreset | null;
  adaptersReady: boolean;
  onDownloadAdapters?: () => void;
};

export function RemakePanel({
  draft,
  onChange,
  busy,
  canSubmit,
  onGenerate,
  onUpload,
  swapPreset,
  editPreset,
  adaptersReady,
  onDownloadAdapters,
}: Props) {
  const videoRef = useRef<HTMLInputElement>(null);
  const paintRef = useRef<HTMLInputElement>(null);
  const active = draft.mode === "swap" ? swapPreset : editPreset;

  const status = useMemo(() => {
    const need = draft.mode === "swap" ? swapPreset : editPreset;
    if (!need) return "Remake recipes missing from catalog — restart the server.";
    if (!need.cached && !adaptersReady) {
      return `Download “${need.label}” in Models (or both Swap+Edit, ~7 GB).`;
    }
    if (!draft.sourcePath) return "Add a source video.";
    if (draft.painted.length < 1) return "Add at least one painted frame.";
    return "Prototype: runs Ref2VA + Remake LoRA + 6-step schedule. Keyframe pins / clean-audio engine still TODO.";
  }, [adaptersReady, draft.sourcePath, draft.painted.length, draft.mode, swapPreset, editPreset]);

  async function addSource(file: File) {
    const up = await onUpload(file, "video");
    onChange({ sourcePath: up.path, sourceName: up.filename || file.name });
  }

  async function addPainted(file: File) {
    const up = await onUpload(file, "image");
    onChange({
      painted: [
        ...draft.painted,
        { path: up.path, name: up.filename || file.name, pin: "" },
      ].slice(0, 4),
    });
  }

  return (
    <div className="focus-panel remake-panel">
      <div className="focus-panel__head">
        <h2 className="focus-panel__title">Remake</h2>
        <p className="focus-panel__sub">
          Powered by{" "}
          <a href="https://huggingface.co/Viggle/Viggle-Remake" target="_blank" rel="noreferrer">
            Viggle Remake
          </a>
          . Locked 6-step schedule — no Turbo or other LoRAs.
        </p>
      </div>

      <div className="focus-seg" role="group" aria-label="Remake mode">
        {(["swap", "edit"] as RemakeMode[]).map((mode) => (
          <button
            key={mode}
            type="button"
            className={`focus-seg__btn${draft.mode === mode ? " is-on" : ""}`}
            disabled={busy}
            onClick={() => onChange({ mode })}
          >
            {mode === "swap" ? "Swap" : "Edit"}
          </button>
        ))}
      </div>

      <p className="focus-panel__hint">
        {draft.mode === "swap"
          ? "Replace people / characters. Paint identity onto source frames."
          : "Change objects, weather, light, or setting. Paint the target look."}
        {active?.schedule ? ` · schedule ${active.schedule}` : ""}
      </p>

      {!adaptersReady && (
        <div className="focus-banner">
          <span>Remake adapters not on disk.</span>
          {onDownloadAdapters ? (
            <button type="button" className="btn-secondary" onClick={onDownloadAdapters} disabled={busy}>
              Open Models
            </button>
          ) : null}
        </div>
      )}

      <div className="focus-field">
        <div className="focus-field__label">Source video</div>
        <div className="focus-field__row">
          <button
            type="button"
            className="btn-secondary"
            disabled={busy}
            onClick={() => videoRef.current?.click()}
          >
            {draft.sourceName || "Choose video…"}
          </button>
          {draft.sourcePath ? (
            <button
              type="button"
              className="btn-ghost"
              disabled={busy}
              onClick={() => onChange({ sourcePath: null, sourceName: null })}
            >
              Clear
            </button>
          ) : null}
          <input
            ref={videoRef}
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void addSource(f).catch(() => undefined);
            }}
          />
        </div>
      </div>

      <div className="focus-field">
        <div className="focus-field__label">Painted frames (1–4)</div>
        <ul className="focus-list">
          {draft.painted.map((item, index) => (
            <li key={`${item.path}-${index}`} className="focus-list__row">
              <span className="focus-list__name" title={item.path}>
                Picture {index + 1}: {item.name}
              </span>
              <label className="focus-pin">
                Pin
                <input
                  type="text"
                  inputMode="numeric"
                  placeholder="frame #"
                  value={item.pin}
                  disabled={busy}
                  onChange={(e) => {
                    const painted = draft.painted.slice();
                    painted[index] = { ...item, pin: e.target.value.replace(/[^\d]/g, "") };
                    onChange({ painted });
                  }}
                />
              </label>
              <button
                type="button"
                className="btn-ghost"
                disabled={busy}
                onClick={() =>
                  onChange({ painted: draft.painted.filter((_, i) => i !== index) })
                }
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
        {draft.painted.length < 4 ? (
          <button
            type="button"
            className="btn-secondary"
            disabled={busy}
            onClick={() => paintRef.current?.click()}
          >
            Add painted frame…
          </button>
        ) : null}
        <input
          ref={paintRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void addPainted(f).catch(() => undefined);
          }}
        />
      </div>

      <div className="focus-field">
        <div className="focus-field__label">Six-section prompt</div>
        <textarea
          className="focus-prompt"
          rows={10}
          value={draft.prompt}
          disabled={busy}
          placeholder={
            "subject_definitions\nsummary\nretention_analysis\ndetailed_description\noverall_soundscape\nnon_diegetic_music\n\nDescribe the target video — not instructions. See Viggle PROMPTING.md."
          }
          onChange={(e) => onChange({ prompt: e.target.value })}
        />
      </div>

      <p className="focus-panel__status">{status}</p>

      <div className="focus-actions">
        <button
          type="button"
          className="btn-primary"
          disabled={!canSubmit || busy}
          onClick={onGenerate}
        >
          {busy ? "Working…" : "Generate Remake"}
        </button>
      </div>
    </div>
  );
}

/** Build Ref2VA refs for experimental Remake submits (engine path later). */
export function remakeRefsFromDraft(draft: RemakeDraft): ReferenceItem[] {
  const out: ReferenceItem[] = [];
  if (draft.sourcePath) {
    out.push({
      id: `remake-src-${draft.sourcePath}`,
      kind: "video",
      path: draft.sourcePath,
      name: draft.sourceName || "source",
    });
  }
  draft.painted.forEach((p, i) => {
    out.push({
      id: `remake-paint-${i}-${p.path}`,
      kind: "image",
      path: p.path,
      name: p.name || `Picture ${i + 1}`,
    });
  });
  return out;
}
