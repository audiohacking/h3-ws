import { useEffect, useRef, useState } from "react";

interface ModelComponent {
  id: string;
  label: string;
  present: boolean;
  path: string;
  size_gib: number;
  note: string;
}

interface ModelsStatus {
  ok: boolean;
  model_dir: string;
  repo_root?: string;
  lora_dir?: string;
  lora_count?: number;
  lora_presets?: import("../../types").LoraPreset[];
  components: ModelComponent[];
  candidates?: string[];
  looks_valid?: boolean;
}

interface DownloadProgress {
  percent: number;
  downloaded_gb: number;
  expected_gb: number;
  speed: string;
  active: boolean;
}

interface DownloadStatusResponse {
  active: boolean;
  component?: string | null;
  error?: string | null;
  progress?: Pick<DownloadProgress, "percent" | "downloaded_gb" | "expected_gb">;
}

interface ModelsManagerProps {
  api: string;
  onClose?: () => void;
  /** Called whenever a server-side download becomes active/inactive (App uses this to guard close). */
  onDownloadStateChange?: (active: boolean) => void;
  /** Fired after a model-folder change so the app can refresh LoRA / path state. */
  onPathApplied?: (status: ModelsStatus) => void;
}

/**
 * Dedicated Models page. Shows what is already on disk and lets the user
 * OPTIONALLY download a missing component. Downloads are never automatic:
 * the user must click "Download". Supports resume via huggingface_hub.
 *
 * The download task lives server-side and survives client disconnects, so the
 * modal is never hard-locked: closing it just detaches the SSE stream, and
 * progress re-attaches on reopen/reload.
 */
export function ModelsManager({ api, onClose, onDownloadStateChange, onPathApplied }: ModelsManagerProps) {
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [pathDraft, setPathDraft] = useState("");
  const [pathBusy, setPathBusy] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${api}/api/models`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as ModelsStatus;
      setStatus(data);
      setPathDraft(data.model_dir || "");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function applyModelPath(path: string) {
    const trimmed = path.trim();
    if (!trimmed) return;
    setPathBusy(true);
    setError(null);
    try {
      const r = await fetch(`${api}/api/models/path`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: trimmed }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as ModelsStatus;
      setStatus(data);
      setPathDraft(data.model_dir || trimmed);
      onPathApplied?.(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPathBusy(false);
    }
  }

  async function browseModelPath() {
    setPathBusy(true);
    setError(null);
    try {
      const r = await fetch(`${api}/api/models/browse`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      if (data.cancelled) return;
      setStatus(data as ModelsStatus);
      setPathDraft(String(data.model_dir || ""));
      onPathApplied?.(data as ModelsStatus);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPathBusy(false);
    }
  }

  /** Open (or re-attach to) an SSE stream for the given component. */
  function openProgressStream(component: string) {
    eventSourceRef.current?.close();
    setBusyId(component);
    setError(null);

    const es = new EventSource(`${api}/api/models/download/stream?component=${component}`);
    eventSourceRef.current = es;

    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      es.close();
      if (eventSourceRef.current === es) eventSourceRef.current = null;
    };

    es.addEventListener("progress", (e) => {
      try {
        setProgress(JSON.parse(e.data));
      } catch {
        // ignore malformed payloads
      }
    });

    es.addEventListener("complete", (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.components) {
          setStatus((prev) => (prev ? { ...prev, components: data.components } : prev));
        }
      } catch {
        // ignore
      }
      finish();
      setProgress(null);
      setBusyId(null);
      onDownloadStateChange?.(false);
      void refresh();
    });

    es.addEventListener("error", (e) => {
      // ONLY the server-sent "error" event (a MessageEvent carrying JSON) is a
      // terminal failure. Transport-level errors arrive as ErrorEvent here but
      // have no `data`, so we leave those to EventSource to retry — we must NOT
      // close the stream on the first network blip, or progress dies mid-download.
      if (e instanceof MessageEvent && e.data) {
        try {
          const data = JSON.parse(e.data);
          setError(data.error || "Download failed");
        } catch {
          setError("Download failed");
        }
        finish();
        setProgress(null);
        setBusyId(null);
        onDownloadStateChange?.(false);
      }
    });

    // No es.onerror handler — EventSource auto-reconnects on transient drops.
    onDownloadStateChange?.(true);
  }

  // On mount: load status, then re-attach to any in-flight server download.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      await refresh();
      if (cancelled) return;
      try {
        const r = await fetch(`${api}/api/models/download/status`);
        if (!r.ok) return;
        const data = (await r.json()) as DownloadStatusResponse;
        if (data.active && data.component && !cancelled) {
          if (data.progress) {
            setProgress({ ...data.progress, speed: "resuming…", active: true });
          }
          openProgressStream(data.component);
        }
      } catch {
        // ignore — modal stays usable on transient status failures
      }
    })();
    return () => {
      cancelled = true;
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
      // The download lives server-side; releasing this flag lets App re-open the
      // modal at any time (the flag must never stick "true" on unmount).
      onDownloadStateChange?.(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  function startDownload(component: ModelComponent) {
    if (busyId !== null) return;
    if (!component.present && !window.confirm(
      `Download ${component.label}?\n\n` +
      `This can be large (${component.id === "fl2va" ? "~134" : "~62"} GB).\n` +
      `Downloads resume automatically if interrupted.\n\nProceed?`
    )) {
      return;
    }
    setError(null);
    setProgress({ percent: 0, downloaded_gb: 0, expected_gb: 0, speed: "starting...", active: true });
    openProgressStream(component.id);
  }

  /** Close is always enabled; a running download continues in the background. */
  function handleClose() {
    if (busyId !== null) {
      if (!window.confirm(
        "Download is still running. It will continue in the background.\n\nClose this panel?"
      )) {
        return;
      }
    }
    onClose?.();
  }

  return (
    <div className="models-page">
      <div className="models-page__head">
        <div>
          <h2 className="models-page__title">Models</h2>
          <p className="models-page__hint">
            Point at an existing MiniMax-H3 folder (or h3-ws clone root) to reuse weights —
            LoRAs under models/loras are picked up automatically.
          </p>
        </div>
        <div className="models-page__actions">
          <button type="button" className="btn-ghost" onClick={() => void refresh()} disabled={loading}>
            Refresh
          </button>
          {onClose && (
            <button type="button" className="btn-ghost" onClick={handleClose}>
              Close
            </button>
          )}
        </div>
      </div>

      {status && (status.lora_dir || status.lora_count != null) && (
        <p className="models-page__hint models-page__lora-hint">
          LoRA folder: <code>{status.lora_dir || "—"}</code>
          {status.lora_count != null ? ` · ${status.lora_count} on disk` : ""}
        </p>
      )}

      <div className="models-page__path">
        <label className="models-page__path-label" htmlFor="models-path-input">
          Model folder
        </label>
        <div className="models-page__path-row">
          <input
            id="models-path-input"
            className="models-page__path-input"
            type="text"
            value={pathDraft}
            disabled={pathBusy || busyId !== null}
            onChange={(e) => setPathDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void applyModelPath(pathDraft);
            }}
            spellCheck={false}
          />
          <button
            type="button"
            className="btn-ghost"
            disabled={pathBusy || busyId !== null}
            onClick={() => void browseModelPath()}
          >
            Browse…
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={pathBusy || busyId !== null || !pathDraft.trim()}
            onClick={() => void applyModelPath(pathDraft)}
          >
            {pathBusy ? "Saving…" : "Apply"}
          </button>
        </div>
        {status?.candidates && status.candidates.length > 0 && (
          <div className="models-page__candidates">
            <span className="models-page__candidates-label">Detected:</span>
            {status.candidates.map((c) => (
              <button
                key={c}
                type="button"
                className={`models-page__candidate${c === status.model_dir ? " is-on" : ""}`}
                disabled={pathBusy || busyId !== null}
                title={c}
                onClick={() => void applyModelPath(c)}
              >
                {c}
              </button>
            ))}
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {loading && !status && <p className="models-page__hint">Checking local model layout…</p>}

      {status && (
        <div className="models-page__list">
          {status.components.map((c) => (
            <div key={c.id} className={`model-card ${c.present ? "model-card--present" : "model-card--missing"}`}>
              <div className="model-card__body">
                <div className="model-card__row">
                  <span className="model-card__label">{c.label}</span>
                  <span className={`model-card__badge ${c.present ? "model-card__badge--ok" : "model-card__badge--missing"}`}>
                    {c.present ? "Present" : "Missing"}
                  </span>
                </div>
                {c.size_gib > 0 && <p className="model-card__size">{c.size_gib.toFixed(1)} GB on disk</p>}
                <p className="model-card__note">{c.note}</p>
                <p className="model-card__path">{c.path}</p>

                {/* Download progress bar */}
                {busyId === c.id && progress && (
                  <div className="download-progress">
                    <div className="download-progress__bar-container">
                      <div
                        className="download-progress__bar"
                        style={{ width: `${progress.percent}%` }}
                      />
                    </div>
                    <div className="download-progress__text">
                      <span>
                        {progress.downloaded_gb > 0 && progress.speed.startsWith("resuming") ? (
                          <span className="download-progress__percent">Resuming {progress.downloaded_gb.toFixed(1)} GB…</span>
                        ) : (
                          <>
                            <span className="download-progress__percent">{progress.percent}%</span>
                            {" · "}
                            {progress.downloaded_gb.toFixed(1)} / {progress.expected_gb.toFixed(0)} GB
                          </>
                        )}
                      </span>
                      <span className="download-progress__speed">{progress.speed}</span>
                    </div>
                  </div>
                )}
              </div>
              <div className="model-card__actions">
                {c.present ? (
                  <span className="model-card__ok">Ready</span>
                ) : (
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => startDownload(c)}
                    disabled={busyId !== null}
                  >
                    {busyId === c.id ? "Downloading…" : "Download"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && !status && !error && (
        <p className="models-page__hint">No model status available.</p>
      )}

      {busyId && (
        <p className="models-page__hint" style={{ marginTop: 8 }}>
          💡 Downloads resume automatically if interrupted. You can close this window.
        </p>
      )}
    </div>
  );
}
