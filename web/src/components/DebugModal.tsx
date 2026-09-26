/**
 * Debug panel: live backend console + built-in self-test, both exportable as
 * one zip (device facts, test results, generation timings, console, logs).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Props = {
  open: boolean;
  api: string;
  onClose: () => void;
};

type Tab = "console" | "selftest";

type Step = {
  name: string;
  status: string;
  seconds?: number;
  summary?: string;
  output?: string[];
};

type BenchRow = { op: string; shape?: string; seconds?: number; tflops?: number; error?: string };

type DitProfile = {
  mode?: string;
  sequence_rows?: number;
  rows?: number;
  sequence_parts?: string;
  seconds_per_block?: number;
  ops?: { op: string; ms_per_block: number; percent: number }[];
};

type SelfTestStatus = {
  state: string;
  mode?: string;
  seconds?: number;
  error?: string;
  steps?: Step[];
  bench?: BenchRow[];
  dit_profile?: DitProfile | null;
};

type Saved = { path: string; name: string; bytes: number };

const MAX_LINES = 5000;

const STEP_LABELS: Record<string, string> = {
  h3_tests: "Core engine checks",
  h3_gqa_tests: "Qwen causal attention (vs double oracle)",
  h3_conv3d_tests: "Video VAE Conv3d (vs double oracle)",
  h3_sdpa_split_tests: "DiT attention, odd lengths (vs double oracle)",
  h3_cache_invalidate_tests: "Warm-session cache",
  h3_bench: "GPU throughput bench",
  dit_op_profile: "Real-weights DiT op profile",
};

function useSavedReport(api: string) {
  const [saved, setSaved] = useState<Saved | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = useCallback(
    async (kind: string) => {
      setBusy(true);
      setError(null);
      try {
        const r = await fetch(`${api}/api/debug/report/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setSaved((await r.json()) as Saved);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api],
  );

  const reveal = useCallback(async () => {
    if (!saved) return;
    await fetch(`${api}/api/debug/reveal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: saved.path }),
    });
  }, [api, saved]);

  return { saved, busy, error, save, reveal };
}

function SavedNotice({ saved, onReveal, downloadHref }: {
  saved: Saved | null;
  onReveal: () => void;
  downloadHref: string;
}) {
  if (!saved) {
    return (
      <a className="debug-modal__link" href={downloadHref}>
        or download in the browser
      </a>
    );
  }
  return (
    <span className="debug-modal__saved">
      Saved <code>{saved.name}</code> ({Math.max(1, Math.round(saved.bytes / 1024))} KiB) to Downloads ·{" "}
      <button type="button" className="debug-modal__linkbtn" onClick={onReveal}>
        Show in Finder
      </button>
    </span>
  );
}

function ConsoleTab({ api, active }: { api: string; active: boolean }) {
  const [lines, setLines] = useState<string[]>([]);
  const [live, setLive] = useState(false);
  const [follow, setFollow] = useState(true);
  const [filter, setFilter] = useState("");
  const [copied, setCopied] = useState(false);
  const cursor = useRef(0);
  const preRef = useRef<HTMLPreElement>(null);
  const report = useSavedReport(api);

  useEffect(() => {
    if (!active) return;
    let source: EventSource | null = null;
    let retry: number | undefined;
    let closed = false;
    const connect = () => {
      // Reconnect from our own cursor: EventSource's automatic retry would
      // replay the original URL and duplicate lines.
      source = new EventSource(`${api}/api/console/stream?after=${cursor.current}`);
      source.addEventListener("open", () => setLive(true));
      source.addEventListener("lines", (event) => {
        const chunk = JSON.parse((event as MessageEvent).data) as {
          lines: string[];
          seq: number;
          dropped: number;
        };
        cursor.current = chunk.seq;
        setLines((prev) => {
          const gap = chunk.dropped ? [`… ${chunk.dropped} earlier lines not shown …`] : [];
          const next = prev.concat(gap, chunk.lines);
          return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
        });
      });
      source.onerror = () => {
        setLive(false);
        source?.close();
        if (!closed) retry = window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      source?.close();
      setLive(false);
    };
  }, [api, active]);

  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return needle ? lines.filter((line) => line.toLowerCase().includes(needle)) : lines;
  }, [lines, filter]);

  useEffect(() => {
    if (follow && preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [shown, follow]);

  async function copyAll() {
    try {
      await navigator.clipboard.writeText(shown.join("\n") || "(empty)");
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      if (!preRef.current) return;
      const range = document.createRange();
      range.selectNodeContents(preRef.current);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
  }

  async function clearAll() {
    await fetch(`${api}/api/console/clear`, { method: "POST" });
    setLines([]);
  }

  return (
    <>
      <div className="debug-modal__toolbar">
        <span className={live ? "debug-modal__live is-on" : "debug-modal__live"}>
          {live ? "● live" : "○ reconnecting…"}
        </span>
        <input
          className="debug-modal__filter"
          placeholder="Filter lines…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <label className="debug-modal__follow">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          Follow
        </label>
      </div>
      <pre
        ref={preRef}
        className="console-modal__pre"
        onScroll={(e) => {
          const el = e.currentTarget;
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
          if (!atBottom && follow) setFollow(false);
        }}
      >
        {shown.join("\n") || "(no log lines yet — start a generation or run a self-test)"}
      </pre>
      {report.error && <p className="form-error">{report.error}</p>}
      <div className="console-modal__actions">
        <SavedNotice
          saved={report.saved}
          onReveal={() => void report.reveal()}
          downloadHref={`${api}/api/debug/report?kind=snapshot`}
        />
        <button type="button" className="btn-secondary" onClick={() => void clearAll()}>
          Clear
        </button>
        <button type="button" className="btn-secondary" onClick={() => void copyAll()}>
          {copied ? "Copied" : "Copy"}
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={report.busy}
          onClick={() => void report.save("snapshot")}
        >
          {report.busy ? "Saving…" : "Save snapshot"}
        </button>
      </div>
    </>
  );
}

function SelfTestTab({ api, active }: { api: string; active: boolean }) {
  const [status, setStatus] = useState<SelfTestStatus>({ state: "idle" });
  const [error, setError] = useState<string | null>(null);
  const report = useSavedReport(api);
  const running = status.state === "running";

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${api}/api/debug/selftest`);
      if (r.ok) setStatus((await r.json()) as SelfTestStatus);
    } catch {
      /* next poll retries */
    }
  }, [api]);

  useEffect(() => {
    if (!active) return;
    void refresh();
    const id = window.setInterval(() => void refresh(), running ? 1000 : 4000);
    return () => window.clearInterval(id);
  }, [active, running, refresh]);

  async function start(mode: "quick" | "deep") {
    setError(null);
    const r = await fetch(`${api}/api/debug/selftest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!r.ok) {
      const body = (await r.json().catch(() => ({}))) as { detail?: string };
      setError(body.detail || `HTTP ${r.status}`);
      return;
    }
    setStatus((await r.json()) as SelfTestStatus);
  }

  async function cancel() {
    await fetch(`${api}/api/debug/selftest/cancel`, { method: "POST" });
    void refresh();
  }

  const profile = status.dit_profile;
  const kind = status.mode ? `selftest-${status.mode}` : "snapshot";

  return (
    <>
      <p className="console-modal__lead">
        Quick (~10 s–2 min) checks the Metal kernels against a double-precision reference and measures GPU
        throughput. Deep (~5 min) also profiles a real DiT forward with your installed weights; it
        unloads the warm model first. Generations wait while a test runs.
      </p>
      <div className="debug-modal__toolbar">
        <button type="button" className="btn-primary" disabled={running} onClick={() => void start("quick")}>
          Run quick test
        </button>
        <button type="button" className="btn-secondary" disabled={running} onClick={() => void start("deep")}>
          Run deep test
        </button>
        {running && (
          <button type="button" className="btn-secondary" onClick={() => void cancel()}>
            Cancel
          </button>
        )}
        <span className={`debug-modal__state is-${status.state}`}>
          {status.state === "idle"
            ? "Not run yet"
            : `${status.mode ?? ""} ${status.state}${status.seconds ? ` in ${status.seconds}s` : ""}`}
        </span>
      </div>
      {(error || status.error) && <p className="form-error">{error || status.error}</p>}
      <div className="debug-modal__results">
        {(status.steps ?? []).map((step) => (
          <div key={step.name} className={`debug-modal__step is-${step.status}`}>
            <span className="debug-modal__badge">{step.status}</span>
            <span className="debug-modal__stepname">{STEP_LABELS[step.name] ?? step.name}</span>
            <span className="debug-modal__stepmeta">
              {step.seconds !== undefined ? `${step.seconds}s` : ""}
            </span>
            {step.summary && <span className="debug-modal__summary">{step.summary}</span>}
          </div>
        ))}
        {(status.bench ?? []).length > 0 && (
          <table className="debug-modal__table">
            <thead>
              <tr><th>GPU op</th><th>Shape</th><th>TFLOPS</th></tr>
            </thead>
            <tbody>
              {(status.bench ?? []).map((row) => (
                <tr key={`${row.op}-${row.shape}`}>
                  <td>{row.op}</td>
                  <td>{row.shape ?? ""}</td>
                  <td>{row.error ? row.error : row.tflops}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {profile?.ops && profile.ops.length > 0 && (
          <table className="debug-modal__table">
            <thead>
              <tr>
                <th colSpan={3}>
                  DiT block at {profile.rows ?? profile.sequence_rows} rows · {profile.seconds_per_block} s/block
                </th>
              </tr>
            </thead>
            <tbody>
              {profile.ops.map((op) => (
                <tr key={op.op}>
                  <td>{op.op}</td>
                  <td>{op.ms_per_block.toFixed(0)} ms</td>
                  <td>{op.percent.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {report.error && <p className="form-error">{report.error}</p>}
      <div className="console-modal__actions">
        <SavedNotice
          saved={report.saved}
          onReveal={() => void report.reveal()}
          downloadHref={`${api}/api/debug/report?kind=${kind}`}
        />
        <button
          type="button"
          className="btn-primary"
          disabled={report.busy || running}
          onClick={() => void report.save(kind)}
        >
          {report.busy ? "Saving…" : "Save report"}
        </button>
      </div>
    </>
  );
}

export function DebugModal({ open, api, onClose }: Props) {
  const [tab, setTab] = useState<Tab>("console");

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-overlay" role="presentation" onClick={onClose}>
      <div
        className="modal console-modal"
        role="dialog"
        aria-labelledby="debug-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal__header">
          <h2 id="debug-title" className="modal__title">
            Debug
          </h2>
          <div className="debug-modal__tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "console"}
              className={tab === "console" ? "debug-modal__tab is-on" : "debug-modal__tab"}
              onClick={() => setTab("console")}
            >
              Console
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "selftest"}
              className={tab === "selftest" ? "debug-modal__tab is-on" : "debug-modal__tab"}
              onClick={() => setTab("selftest")}
            >
              Self-test
            </button>
          </div>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="modal__body console-modal__body">
          {tab === "console" ? (
            <ConsoleTab api={api} active={open} />
          ) : (
            <SelfTestTab api={api} active={open} />
          )}
          <p className="debug-modal__privacy">
            Reports contain device info, test results, generation timings and settings, the console and
            recent logs. No prompts, images or videos; prompt-based file names are masked and your home
            folder shows as ~.
          </p>
        </div>
      </div>
    </div>
  );
}
