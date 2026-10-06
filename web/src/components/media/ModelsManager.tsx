import { useEffect, useRef, useState } from "react";

interface ModelComponent {
  id: string;
  label: string;
  present: boolean;
  path: string;
  size_gib: number;
  note: string;
  essential?: boolean;
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

const SIZE_HINT: Record<string, string> = {
  fl2va: "~134 GB",
  ref2va: "~62 GB",
  taeh3: "~22 MB",
  taomate: "~2.4 GB",
  sam3: "~3.3 GB",
};

/**
 * Compact Models dialog. Fixed chrome (path + actions); only the component
 * list scrolls. Downloads are optional except where a workflow auto-fetches
 * (TAEH3 on launch, SAM when Faces runs).
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
  const listRef = useRef<HTMLDivElement | null>(null);

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

    onDownloadStateChange?.(true);
  }

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
        // ignore
      }
    })();
    return () => {
      cancelled = true;
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
      onDownloadStateChange?.(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  function startDownload(component: ModelComponent) {
    if (busyId !== null) return;
    const sizeHint = SIZE_HINT[component.id] || "~? GB";
    const needsConfirm = component.id === "fl2va" || component.id === "ref2va" || component.id === "taomate" || component.id === "sam3";
    if (
      !component.present &&
      needsConfirm &&
      !window.confirm(
        `Download ${component.label}?\n\n` +
          `Size: ${sizeHint}.\n` +
          (component.id === "sam3" ? "May need Hugging Face login + SAM license.\n" : "") +
          `Downloads resume if interrupted.\n\nProceed?`,
      )
    ) {
      return;
    }
    setError(null);
    setProgress({ percent: 0, downloaded_gb: 0, expected_gb: 0, speed: "starting...", active: true });
    openProgressStream(component.id);
  }

  function handleClose() {
    if (busyId !== null) {
      if (
        !window.confirm(
          "Download is still running. It will continue in the background.\n\nClose this panel?",
        )
      ) {
        return;
      }
    }
    onClose?.();
  }

  const components = status?.components ?? [];

  return (
    <div className="models-page">
      <div className="models-page__head">
        <h2 className="models-page__title">Models</h2>
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
            <span className="models-page__candidates-label">Nearby:</span>
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

      {(status?.lora_dir || status?.lora_count != null) && (
        <p className="models-page__meta">
          <span>
            LoRAs: <code>{status.lora_dir || "—"}</code>
            {status.lora_count != null ? ` · ${status.lora_count}` : ""}
          </span>
          {busyId ? <span>Download continues if you close</span> : null}
        </p>
      )}

      {error && <div className="error-banner">{error}</div>}
      {loading && !status && <p className="models-page__hint">Checking local layout…</p>}

      {status && (
        <div
          className="models-page__scroll"
          ref={listRef}
          onWheel={(e) => e.stopPropagation()}
        >
          <div className="models-page__list">
            {components.map((c) => (
              <div
                key={c.id}
                className={`model-card ${c.present ? "model-card--present" : "model-card--missing"}${
                  c.essential ? " model-card--required" : ""
                }`}
              >
                <div className="model-card__body">
                  <div className="model-card__row">
                    <span className="model-card__label" title={c.path}>
                      {c.label}
                    </span>
                    <span
                      className={`model-card__tag ${
                        c.essential ? "model-card__tag--required" : "model-card__tag--optional"
                      }`}
                    >
                      {c.essential ? "required" : "optional"}
                    </span>
                    <span
                      className={`model-card__badge ${
                        c.present ? "model-card__badge--ok" : "model-card__badge--missing"
                      }`}
                    >
                      {c.present ? "Ready" : "Missing"}
                    </span>
                  </div>
                  <p className="model-card__note" title={c.note}>
                    {c.note}
                    {c.size_gib > 0
                      ? ` · ${
                          c.size_gib >= 1
                            ? `${c.size_gib.toFixed(1)} GB`
                            : `${Math.max(1, Math.round(c.size_gib * 1024))} MB`
                        } on disk`
                      : !c.present && SIZE_HINT[c.id]
                        ? ` · ${SIZE_HINT[c.id]}`
                        : ""}
                  </p>
                </div>
                <div className="model-card__actions">
                  {c.present ? (
                    <span className="model-card__ok">✓</span>
                  ) : (
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={() => startDownload(c)}
                      disabled={busyId !== null}
                    >
                      {busyId === c.id ? "…" : "Download"}
                    </button>
                  )}
                </div>
                {busyId === c.id && progress && (
                  <div className="model-card__progress download-progress">
                    <div className="download-progress__bar-container">
                      <div
                        className="download-progress__bar"
                        style={{ width: `${progress.percent}%` }}
                      />
                    </div>
                    <div className="download-progress__text">
                      <span>
                        {progress.downloaded_gb > 0 && progress.speed.startsWith("resuming") ? (
                          <span className="download-progress__percent">
                            Resuming {progress.downloaded_gb.toFixed(1)} GB…
                          </span>
                        ) : progress.expected_gb > 0 && progress.expected_gb < 0.1 ? (
                          <>
                            <span className="download-progress__percent">{progress.percent}%</span>
                            {" · "}~{(progress.expected_gb * 1024).toFixed(0)} MB
                          </>
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
            ))}
          </div>
        </div>
      )}

      {!loading && !status && !error && (
        <p className="models-page__hint">No model status available.</p>
      )}
    </div>
  );
}
