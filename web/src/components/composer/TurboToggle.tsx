import { useEffect, useMemo, useRef, useState } from "react";
import { TURBO_CONFIG, type TurboTier } from "../../config";
import type { LoraPreset } from "../../types";

export function isTurboPreset(p: LoraPreset): boolean {
  if (p.turbo) return true;
  const hay = `${p.label} ${p.spec}`.toLowerCase();
  return TURBO_CONFIG.PREFERRED.some((n) => hay.includes(n));
}

export function rankTurboPreset(p: LoraPreset): number {
  const hay = `${p.label} ${p.spec}`.toLowerCase();
  const rank = TURBO_CONFIG.PREFERRED.findIndex((n) => hay.includes(n));
  return rank < 0 ? 99 : rank;
}

export function pickDefaultTurboId(presets: LoraPreset[]): string | null {
  const ranked = presets.filter(isTurboPreset).sort((a, b) => rankTurboPreset(a) - rankTurboPreset(b));
  return ranked[0]?.id ?? null;
}

interface TurboToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  tier: TurboTier;
  onTierChange: (tier: TurboTier) => void;
  /** Available turbo / distill LoRAs (on-disk + hub). */
  options: LoraPreset[];
  selectedId: string | null;
  onSelectId: (id: string) => void;
  disabled?: boolean;
  loading?: boolean;
}

function shortLabel(preset: LoraPreset): string {
  const raw = preset.label.replace(/\s*·\s*HF hub\s*$/i, "").trim();
  if (raw.length <= 36) return raw;
  return raw.slice(0, 33) + "…";
}

export function TurboToggle({
  enabled,
  onChange,
  tier,
  onTierChange,
  options,
  selectedId,
  onSelectId,
  disabled,
  loading,
}: TurboToggleProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const tiers = Object.entries(TURBO_CONFIG.TIERS) as [TurboTier, (typeof TURBO_CONFIG.TIERS)[TurboTier]][];

  const turboOptions = useMemo(
    () => [...options].filter(isTurboPreset).sort((a, b) => rankTurboPreset(a) - rankTurboPreset(b)),
    [options],
  );

  const selected = turboOptions.find((p) => p.id === selectedId) ?? turboOptions[0] ?? null;
  const fixedSteps = selected?.steps != null && selected.steps <= 4;

  const locked = Boolean(disabled || loading);

  useEffect(() => {
    if (locked) setMenuOpen(false);
  }, [locked]);

  useEffect(() => {
    if (!menuOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  return (
    <div className="turbo-group" ref={rootRef}>
      <div className={`turbo-pill${enabled ? " is-on" : ""}${loading ? " is-busy" : ""}`}>
        <button
          type="button"
          className="turbo-pill__main"
          disabled={locked}
          onClick={() => {
            if (locked) return;
            onChange(!enabled);
          }}
          title={
            locked
              ? "Settings are locked while a generation is running"
              : selected
                ? `${TURBO_CONFIG.GUIDANCE}\n\nCurrent: ${selected.label}`
                : TURBO_CONFIG.GUIDANCE
          }
          aria-pressed={enabled}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
          </svg>
          <span>{loading ? "Loading…" : enabled ? "turbo" : "turbo off"}</span>
        </button>
        {turboOptions.length > 0 && (
          <button
            type="button"
            className="turbo-pill__pick"
            disabled={locked}
            title={
              locked
                ? "Settings are locked while a generation is running"
                : selected
                  ? `Pick a different turbo LoRA — now ${selected.label}`
                  : "Pick which turbo LoRA to load"
            }
            aria-haspopup="listbox"
            aria-expanded={menuOpen}
            onClick={(e) => {
              e.stopPropagation();
              if (locked) return;
              setMenuOpen((v) => !v);
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
        )}
      </div>

      {menuOpen && !locked && (
        <div className="turbo-menu" role="listbox" aria-label="Turbo LoRA">
          <div className="turbo-menu__title">Turbo LoRA</div>
          {turboOptions.map((opt) => {
            const on = opt.id === (selected?.id ?? "");
            return (
              <button
                key={opt.id}
                type="button"
                role="option"
                aria-selected={on}
                className={`turbo-menu__item${on ? " is-on" : ""}`}
                title={opt.spec}
                onClick={() => {
                  if (locked) return;
                  onSelectId(opt.id);
                  setMenuOpen(false);
                  if (!enabled) onChange(true);
                }}
              >
                <span className="turbo-menu__name">{shortLabel(opt)}</span>
                <span className="turbo-menu__meta">
                  {opt.steps ? `${opt.steps} step` : "turbo"}
                  {opt.scale != null ? ` · ×${opt.scale}` : ""}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {enabled && !loading && !fixedSteps && (
        <div className="seg" role="radiogroup" aria-label="Turbo quality">
          {tiers.map(([key, config]) => (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={tier === key}
              className={`seg__btn${tier === key ? " is-on" : ""}`}
              disabled={locked}
              title={locked ? "Settings are locked while a generation is running" : config.description}
              onClick={() => {
                if (locked) return;
                onTierChange(key);
              }}
            >
              {config.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
