type Props = {
  open: boolean;
  draft: string;
  original: string;
  busy?: boolean;
  error?: string | null;
  onDraftChange: (v: string) => void;
  onUse: () => void;
  onRevert: () => void;
  onKeep: () => void;
  onClose: () => void;
};

export function RefinePanel({
  open,
  draft,
  original,
  busy,
  error,
  onDraftChange,
  onUse,
  onRevert,
  onKeep,
  onClose,
}: Props) {
  if (!open) return null;
  return (
    <div className="refine-panel" role="region" aria-label="Refine draft">
      <div className="refine-panel__head">
        <strong>Refine</strong>
        <button type="button" className="refine-panel__close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      {busy ? (
        <p className="refine-panel__status">Rewriting…</p>
      ) : (
        <textarea
          className="refine-panel__draft"
          value={draft}
          onChange={(e) => onDraftChange(e.target.value)}
          rows={8}
        />
      )}
      {error && <p className="form-error">{error}</p>}
      {!busy && (
        <div className="refine-panel__actions">
          <button type="button" className="btn-primary" onClick={onUse} disabled={!draft.trim()}>
            Use
          </button>
          <button type="button" className="btn-secondary" onClick={onRevert} disabled={!original}>
            Revert
          </button>
          <button type="button" className="btn-secondary" onClick={onKeep}>
            Keep draft
          </button>
        </div>
      )}
    </div>
  );
}
