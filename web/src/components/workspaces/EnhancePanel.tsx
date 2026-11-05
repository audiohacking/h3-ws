import type { Clip } from "../../types";

type Props = {
  clips: Clip[];
  selectedClipId: string | null;
  onSelectClip: (id: string | null) => void;
  upframe: boolean;
  onUpframe: (on: boolean) => void;
  upframeMultiplier: number;
  onUpframeMultiplier: (n: number) => void;
  redetail: boolean;
  onRedetail: (on: boolean) => void;
  redetailDenoise: number;
  onRedetailDenoise: (n: number) => void;
  busy: boolean;
  canRun: boolean;
  onRun: () => void;
  status?: string | null;
};

function clipLabel(c: Clip): string {
  const raw = (c.prompt || c.id || "clip").trim();
  return raw.length > 48 ? `${raw.slice(0, 45)}…` : raw;
}

export function EnhancePanel({
  clips,
  selectedClipId,
  onSelectClip,
  upframe,
  onUpframe,
  upframeMultiplier,
  onUpframeMultiplier,
  redetail,
  onRedetail,
  redetailDenoise,
  onRedetailDenoise,
  busy,
  canRun,
  onRun,
  status,
}: Props) {
  const done = clips.filter((c) => c.status === "done" && c.video_url);
  return (
    <div className="focus-panel enhance-panel">
      <div className="focus-panel__head">
        <h2 className="focus-panel__title">Enhance</h2>
        <p className="focus-panel__sub">
          Upframe and ReDetail on a finished library clip. No LoRAs, Turbo, or Remake mixing.
        </p>
      </div>

      <div className="focus-field">
        <div className="focus-field__label">Library clip</div>
        <select
          className="focus-select"
          value={selectedClipId || ""}
          disabled={busy}
          onChange={(e) => onSelectClip(e.target.value || null)}
        >
          <option value="">Select a clip…</option>
          {done.map((c) => (
            <option key={c.id} value={c.id}>
              {clipLabel(c)}
            </option>
          ))}
        </select>
      </div>

      <div className="focus-checks">
        <label className="focus-check">
          <input
            type="checkbox"
            checked={upframe}
            disabled={busy}
            onChange={(e) => onUpframe(e.target.checked)}
          />
          Upframe (FILM)
        </label>
        {upframe ? (
          <label className="focus-inline">
            ×
            <input
              type="number"
              min={2}
              max={4}
              step={1}
              value={upframeMultiplier}
              disabled={busy}
              onChange={(e) => onUpframeMultiplier(Math.max(2, Math.round(Number(e.target.value) || 2)))}
            />
          </label>
        ) : null}
      </div>

      <div className="focus-checks">
        <label className="focus-check">
          <input
            type="checkbox"
            checked={redetail}
            disabled={busy}
            onChange={(e) => onRedetail(e.target.checked)}
          />
          ReDetail (×2 H3 refine)
        </label>
        {redetail ? (
          <label className="focus-inline">
            denoise
            <input
              type="number"
              min={0.05}
              max={0.95}
              step={0.05}
              value={redetailDenoise}
              disabled={busy}
              onChange={(e) => onRedetailDenoise(Number(e.target.value) || 0.45)}
            />
          </label>
        ) : null}
      </div>

      {status ? <p className="focus-panel__status">{status}</p> : null}

      <div className="focus-actions">
        <button type="button" className="btn-primary" disabled={!canRun || busy} onClick={onRun}>
          {busy ? "Working…" : "Run enhance"}
        </button>
      </div>
    </div>
  );
}
