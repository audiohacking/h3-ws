import { useEffect, useRef, useState } from "react";
import type { CompileToken } from "../../compile";
import { useDismissable } from "../../hooks/useDismissable";

interface WhatTheModelReadsProps {
  compiledPrompt: string;
  tokens: CompileToken[];
  warnings?: string[];
  disabled?: boolean;
}

export function WhatTheModelReads({
  compiledPrompt,
  tokens,
  warnings,
  disabled,
}: WhatTheModelReadsProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useDismissable(open, () => setOpen(false), rootRef);
  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  return (
    <div
      ref={rootRef}
      className={`wtmr${disabled ? " wtmr--disabled" : ""}`}
    >
      <button
        type="button"
        className={`wtmr__toggle${open ? " is-open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-expanded={open}
      >
        <span className="wtmr__chevron" aria-hidden>
          ▸
        </span>
        What the model reads
        {tokens.length > 0 && <span className="wtmr__count">{tokens.length}</span>}
      </button>
      {open && (
        <div className="wtmr__body">
          {tokens.length === 0 ? (
            <p className="wtmr__none">No references — prompt only.</p>
          ) : (
            <div className="wtmr__tokens">
              {tokens.map((t) => (
                <span key={t.handle} className={`wtmr__token wtmr__token--${t.token.split(" ")[0].toLowerCase()}`}>
                  {t.token}
                  <span className="wtmr__token-name">{t.name}</span>
                </span>
              ))}
            </div>
          )}
          {warnings && warnings.length > 0 && (
            <section>
              <h4>warnings</h4>
              {warnings.map((w) => (
                <p key={w}>{w}</p>
              ))}
            </section>
          )}
          <section>
            <h4>compiled prompt</h4>
            <pre className="wtmr__pre">{compiledPrompt.trim() || "(empty)"}</pre>
          </section>
        </div>
      )}
    </div>
  );
}
