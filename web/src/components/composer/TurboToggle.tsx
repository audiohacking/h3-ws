import { useEffect, useMemo, useRef, useState } from "react";
import { TURBO_CONFIG, type TurboTier } from "../../config";
import type { LoraPreset } from "../../types";

/**
 * Distill / turbo adapters only. Prefer the backend `turbo` flag; fall back to
 * tight filename needles — never bare "minimax_h3" (that matches every H3 LoRA).
 */
export function isTurboPreset(p: LoraPreset): boolean {
  if (p.turbo === true || p.category === "turbo") return true;
  const hay = `${p.id} ${p.label} ${p.spec}`.toLowerCase();
  if (/\b(taomate|dmad|fasth3)\b/.test(hay)) return true;
  if (/tutu|20to8[_-]?nfe/.test(hay)) return true;
  // H3/MiniMax distill packs (LightX2V, *turbo*, *8step*, …) — require h3|minimax.
  if (
    /(?:lightx2v|[_-]turbo[_-]|\bdmd\b|8step|3step|4step).*(?:h3|minimax)|(?:h3|minimax).*(?:lightx2v|[_-]turbo[_-]|dmd|8step|3step|4step|nfe)/.test(
      hay,
    )
  ) {
    return true;
  }
  return false;
}

export function rankTurboPreset(p: LoraPreset): number {
  const hay = `${p.id} ${p.label} ${p.spec}`.toLowerCase();
  const rank = TURBO_CONFIG.PREFERRED.findIndex((n) => hay.includes(n));
  return rank < 0 ? 99 : rank;
}

export function pickDefaultTurboId(presets: LoraPreset[]): string | null {
  const ranked = presets.filter(isTurboPreset).sort((a, b) => rankTurboPreset(a) - rankTurboPreset(b));
  return ranked[0]?.id ?? null;
}

/**
 * Published recipe strength for a turbo adapter. Prefer `default_scale` from the
 * API; fall back to Continuity / HF card defaults so a stale catalog (e.g. old
 * LightX2V 0.6) cannot ship the wrong fuse weight.
 */
export function recipeScaleForTurbo(p: LoraPreset): number {
  if (typeof p.default_scale === "number" && Number.isFinite(p.default_scale)) {
    return p.default_scale;
  }
  const hay = `${p.id} ${p.label} ${p.spec}`.toLowerCase();
  if (/\bdmad\b/.test(hay) && !/8step/.test(hay)) return 1.0;
  if (
    /(?:lightx2v|ref2v_turbo|fl2v_turbo|[_-]turbo[_-].*8step|8step.*turbo)/.test(hay) &&
    /(?:h3|minimax)/.test(hay)
  ) {
    return 1.0;
  }
  if (/\b(taomate|tutu|20to8|fasth3)\b/.test(hay)) return 0.8;
  return Number.isFinite(p.scale) ? p.scale : TURBO_CONFIG.SCALE;
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
  /** Change strength for the active turbo LoRA (defaults come from the recipe). */
  onScaleChange?: (id: string, scale: number) => void;
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
  onScaleChange,
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
                  // Select alone enables turbo (applyTurboPreset). Do NOT also call
                  // onChange(true) — that raced with a stale turboLoraId and re-applied TaoMate.
                  onSelectId(opt.id);
                  setMenuOpen(false);
                }}
              >
                <span className="turbo-menu__name">{shortLabel(opt)}</span>
                <span className="turbo-menu__meta">
                  {opt.steps ? `${opt.steps} step` : "turbo"}
                  {` · ×${recipeScaleForTurbo(opt)}`}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {enabled && !loading && selected && onScaleChange && (
        <label
          className="turbo-scale"
          title="LoRA strength / intensity. Recipe default is a starting point — edit freely."
        >
          <span className="turbo-scale__label">strength</span>
          <input
            type="number"
            className="turbo-scale__input"
            min={0}
            max={2}
            step={0.05}
            value={
              Number.isFinite(selected.scale)
                ? selected.scale
                : recipeScaleForTurbo(selected)
            }
            disabled={locked}
            onChange={(e) => {
              if (locked) return;
              const next = Number(e.target.value);
              if (!Number.isFinite(next)) return;
              onScaleChange(selected.id, Math.min(2, Math.max(0, next)));
            }}
          />
        </label>
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
