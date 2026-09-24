import { useEffect, useState } from "react";
import type { RefineSettingsPublic } from "../../types";

type Props = {
  open: boolean;
  onClose: () => void;
  api: string;
  initial: RefineSettingsPublic;
  onSaved: (next: RefineSettingsPublic) => void;
};

export function FeaturesPopup({ open, onClose, api, initial, onSaved }: Props) {
  const [enabled, setEnabled] = useState(initial.enabled);
  const [baseUrl, setBaseUrl] = useState(initial.base_url);
  const [model, setModel] = useState(initial.model);
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setEnabled(initial.enabled);
    setBaseUrl(initial.base_url);
    setModel(initial.model);
    setApiKey("");
    setClearKey(false);
    setError(null);
  }, [open, initial]);

  if (!open) return null;

  async function save() {
    setSaving(true);
    setError(null);
    const body: Record<string, unknown> = {
      enabled,
      base_url: baseUrl,
      model,
    };
    if (clearKey) body.api_key = "";
    else if (apiKey.trim()) body.api_key = apiKey.trim();
    try {
      const r = await fetch(`${api}/api/settings/refine`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "Save failed");
      }
      const next = (await r.json()) as RefineSettingsPublic;
      onSaved(next);
      onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal features-modal"
        role="dialog"
        aria-labelledby="features-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-header">
          <h2 id="features-title">Features</h2>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="modal-body">
          <p className="features-lead">
            Optional remote Refine rewrites prompts over an OpenAI-compatible API before you
            generate. Live TAEH3 previews use a local decoder and need no settings here.
          </p>
          <label className="features-toggle">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            Enable Refine
          </label>
          <label className="field-label">
            Base URL
            <input
              type="url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:1234/v1"
              disabled={!enabled}
            />
          </label>
          <label className="field-label">
            Model
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="model id"
              disabled={!enabled}
            />
          </label>
          <label className="field-label">
            API key {initial.key_set ? "(set — leave blank to keep)" : "(optional)"}
            <input
              type="password"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setClearKey(false);
              }}
              placeholder={initial.key_set ? "••••••••" : ""}
              disabled={!enabled}
              autoComplete="off"
            />
          </label>
          {initial.key_set && (
            <label className="features-toggle">
              <input
                type="checkbox"
                checked={clearKey}
                onChange={(e) => {
                  setClearKey(e.target.checked);
                  if (e.target.checked) setApiKey("");
                }}
                disabled={!enabled}
              />
              Clear stored API key
            </label>
          )}
          {error && <p className="form-error">{error}</p>}
        </div>
        <footer className="modal-footer">
          <button type="button" className="btn-secondary" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={() => void save()} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </footer>
      </div>
    </div>
  );
}
