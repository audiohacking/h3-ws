/**
 * Backend console viewer — live ring buffer + on-disk log tails for bug reports.
 */
import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  open: boolean;
  api: string;
  onClose: () => void;
};

type ConsolePayload = {
  ok: boolean;
  lines: string[];
  count: number;
  log_dir?: string;
  log_paths?: Record<string, string>;
};

export function ConsoleModal({ open, api, onClose }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [logDir, setLogDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const preRef = useRef<HTMLPreElement>(null);
  const stickBottom = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${api}/api/console?limit=1000`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as ConsolePayload;
      setLines(data.lines ?? []);
      setLogDir(data.log_dir ?? "");
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    if (!open) return;
    void refresh();
    const id = window.setInterval(() => void refresh(), 1500);
    return () => window.clearInterval(id);
  }, [open, refresh]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!stickBottom.current || !preRef.current) return;
    preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [lines]);

  if (!open) return null;

  const text = lines.join("\n");

  async function copyAll() {
    try {
      await navigator.clipboard.writeText(text || "(empty)");
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Fallback: select the pre so the user can ⌘C
      const range = document.createRange();
      if (preRef.current) {
        range.selectNodeContents(preRef.current);
        const sel = window.getSelection();
        sel?.removeAllRanges();
        sel?.addRange(range);
      }
    }
  }

  return (
    <div className="modal-overlay" role="presentation" onClick={onClose}>
      <div
        className="modal console-modal"
        role="dialog"
        aria-labelledby="console-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal__header">
          <h2 id="console-title" className="modal__title">
            Console
          </h2>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="modal__body console-modal__body">
          <p className="console-modal__lead">
            Backend + h3 output for bug reports. Select any text, or Copy all.
            {logDir ? (
              <>
                {" "}
                Files also under <code>{logDir}</code>.
              </>
            ) : null}
          </p>
          {error && <p className="form-error">{error}</p>}
          <pre
            ref={preRef}
            className="console-modal__pre"
            onScroll={(e) => {
              const el = e.currentTarget;
              stickBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
            }}
          >
            {text || "(no log lines yet — start a generation or open Models)"}
          </pre>
          <div className="console-modal__actions">
            <button type="button" className="btn-secondary" onClick={() => void refresh()}>
              Refresh
            </button>
            <button type="button" className="btn-primary" onClick={() => void copyAll()}>
              {copied ? "Copied" : "Copy all"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
