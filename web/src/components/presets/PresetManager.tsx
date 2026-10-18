import { useEffect, useState } from "react";
import type { GenerationPreset } from "../../types";
import type { ApplyStyleOptions, StyleEntry } from "../../styleAtlas";
import { StyleAtlas } from "../media/StyleAtlas";

interface PresetManagerProps {
  presets: GenerationPreset[];
  onSave: (name: string, description?: string) => void;
  onLoad: (preset: GenerationPreset) => void;
  onDelete: (id: string) => void;
  disabled?: boolean;
  activeStyleId?: string | null;
  onApplyStyle: (style: StyleEntry, options?: ApplyStyleOptions) => void | Promise<void>;
}

export function PresetManager({
  presets,
  onSave,
  onLoad,
  onDelete,
  disabled,
  activeStyleId,
  onApplyStyle,
}: PresetManagerProps) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"looks" | "saved">("looks");
  const [showSaveForm, setShowSaveForm] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saveDescription, setSaveDescription] = useState("");

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open]);

  useEffect(() => {
    if (!open) {
      setShowSaveForm(false);
      setSaveName("");
      setSaveDescription("");
    }
  }, [open]);

  function handleSave() {
    if (!saveName.trim()) return;
    onSave(saveName.trim(), saveDescription.trim() || undefined);
    setSaveName("");
    setSaveDescription("");
    setShowSaveForm(false);
  }

  function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    if (confirm("Delete this preset?")) {
      onDelete(id);
    }
  }

  function handleLoad(preset: GenerationPreset) {
    onLoad(preset);
    setOpen(false);
  }

  function handleApplyStyle(style: StyleEntry, options?: ApplyStyleOptions) {
    void Promise.resolve(onApplyStyle(style, options)).finally(() => setOpen(false));
  }

  return (
    <div className="preset-manager">
      <button
        type="button"
        className="preset-manager__toggle"
        onClick={() => setOpen(true)}
        disabled={disabled}
      >
        <span>Presets</span>
        {presets.length > 0 && (
          <span className="preset-manager__count">{presets.length}</span>
        )}
      </button>

      {open && (
        <div className="modal-backdrop" onClick={() => setOpen(false)}>
          <div
            className="modal modal--fullscreen presets-modal"
            role="dialog"
            aria-labelledby="presets-modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="modal__header">
              <h2 id="presets-modal-title" className="modal__title">
                Presets
              </h2>
              <button
                type="button"
                className="modal__close"
                onClick={() => setOpen(false)}
                aria-label="Close modal"
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden
                >
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </header>

            <div className="presets-modal__tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={tab === "looks"}
                className={`presets-modal__tab${tab === "looks" ? " is-on" : ""}`}
                onClick={() => setTab("looks")}
              >
                Looks
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "saved"}
                className={`presets-modal__tab${tab === "saved" ? " is-on" : ""}`}
                onClick={() => setTab("saved")}
              >
                Saved{presets.length > 0 ? ` (${presets.length})` : ""}
              </button>
            </div>

            <div className="modal__body presets-modal__body">
              {tab === "looks" && (
                <StyleAtlas
                  disabled={disabled}
                  activeId={activeStyleId}
                  onApply={handleApplyStyle}
                />
              )}

              {tab === "saved" && presets.length === 0 && !showSaveForm && (
                <p className="preset-manager__empty">
                  No saved presets. Save your current settings to reuse them later.
                </p>
              )}

              {tab === "saved" && presets.length > 0 && (
                <div className="presets-modal__list">
                  {presets.map((preset) => (
                    <div
                      key={preset.id}
                      className="preset-card"
                      onClick={() => handleLoad(preset)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === "Enter" && handleLoad(preset)}
                    >
                      <div className="preset-card__content">
                        <span className="preset-card__name">{preset.name}</span>
                        {preset.description && (
                          <span className="preset-card__description">
                            {preset.description}
                          </span>
                        )}
                        <span className="preset-card__meta">
                          {preset.mode} · {preset.resolutionId} · {preset.numSteps} steps
                          {preset.turboEnabled && " · Turbo"}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="preset-card__delete"
                        onClick={(e) => handleDelete(e, preset.id)}
                        title="Delete preset"
                      >
                        &times;
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {tab === "saved" &&
                (showSaveForm ? (
                  <div className="preset-save-form">
                    <input
                      type="text"
                      className="preset-save-form__input"
                      placeholder="Preset name"
                      value={saveName}
                      onChange={(e) => setSaveName(e.target.value)}
                      autoFocus
                    />
                    <input
                      type="text"
                      className="preset-save-form__input"
                      placeholder="Description (optional)"
                      value={saveDescription}
                      onChange={(e) => setSaveDescription(e.target.value)}
                    />
                    <div className="preset-save-form__actions">
                      <button
                        type="button"
                        className="preset-save-form__btn preset-save-form__btn--save"
                        onClick={handleSave}
                        disabled={!saveName.trim()}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="preset-save-form__btn preset-save-form__btn--cancel"
                        onClick={() => {
                          setShowSaveForm(false);
                          setSaveName("");
                          setSaveDescription("");
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="preset-manager__save-btn"
                    onClick={() => setShowSaveForm(true)}
                    disabled={disabled}
                  >
                    + Save Current Settings
                  </button>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
