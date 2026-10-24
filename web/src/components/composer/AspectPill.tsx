/**
 * Continuity aspect popover — ``openAspectPopover`` / ``aspectGrid`` from
 * ComfyUI-Continuity ``web/creator/pills.js``.
 *
 * Wide/Tall switch + reciprocal shape grid. Optional Auto / @ref donors when
 * attached media reports a ratio.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  aspectShapes,
  H3_ASPECTS,
  ratioOfAspect,
  type AspectShape,
} from "../../h3Canvas";
import type { ReferenceItem } from "../../types";
import { activeRefs, handleForRef } from "../../compile";

export type AspectDonor = {
  value: string;
  label: string;
  tag: string;
  ratio: number;
  sub: string;
};

type Props = {
  disabled?: boolean;
  open: boolean;
  anchor: HTMLElement | null;
  aspect: string;
  onAspect: (label: string) => void;
  /** Continuity ``aspect_source``: auto | pill | donor value. */
  aspectSource?: string;
  onAspectSource?: (source: string | undefined) => void;
  refs?: ReferenceItem[];
  autoRatio?: number | null;
  autoSub?: string | null;
  onClose: () => void;
};

function AspectGlyph({ ratio, long = 18 }: { ratio: number; long?: number }) {
  const width = ratio >= 1 ? long : long * ratio;
  const height = ratio >= 1 ? long / ratio : long;
  return (
    <span className="mmc-aspect-glyph" style={{ width: long, height: long }}>
      <span style={{ width, height }} />
    </span>
  );
}

function donorsFromRefs(refs: ReferenceItem[]): AspectDonor[] {
  const out: AspectDonor[] = [];
  const live = activeRefs(refs);
  for (let i = 0; i < refs.length; i++) {
    const r = refs[i]!;
    if (!live.includes(r)) continue;
    if (r.kind !== "image" && r.kind !== "video" && r.kind !== "silent_video" && r.kind !== "video_audio") {
      continue;
    }
    // Without probed pixels we still offer the capsule — glyph falls back to 1:1.
    const ratio = 1;
    const handle = handleForRef(refs, i);
    out.push({
      value: r.id,
      label: handle,
      tag: r.kind.startsWith("video") ? "vid" : "img",
      ratio,
      sub: r.kind.includes("video") ? "reference video" : "reference image",
    });
  }
  return out;
}

export function AspectPill({
  disabled,
  open,
  anchor,
  aspect,
  onAspect,
  aspectSource = "pill",
  onAspectSource,
  refs = [],
  autoRatio = null,
  autoSub = null,
  onClose,
}: Props) {
  const popRef = useRef<HTMLDivElement>(null);
  const shapes = useMemo(() => aspectShapes(H3_ASPECTS), []);
  const donors = useMemo(() => donorsFromRefs(refs), [refs]);
  const showDonors = donors.length > 0;
  const [tall, setTall] = useState(() => ratioOfAspect(aspect) < 1);

  useEffect(() => {
    if (open) setTall(ratioOfAspect(aspect) < 1);
  }, [open, aspect]);

  const checked =
    !showDonors || aspectSource === "pill" || aspectSource === "auto"
      ? aspect
      : null;

  useLayoutEffect(() => {
    if (!open || !anchor || !popRef.current) return;
    const pop = popRef.current;
    const place = () => {
      const rect = anchor.getBoundingClientRect();
      const box = pop.getBoundingClientRect();
      const left = Math.max(8, Math.min(rect.left, window.innerWidth - box.width - 8));
      const above = rect.top - box.height - 8 > 8;
      const top = above
        ? rect.top - box.height - 8
        : Math.min(rect.bottom + 8, window.innerHeight - box.height - 8);
      pop.style.left = `${left}px`;
      pop.style.top = `${Math.max(8, top)}px`;
    };
    place();
    const ro = new ResizeObserver(place);
    ro.observe(pop);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, anchor, tall, aspect]);

  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => {
      const t = event.target as Node | null;
      if (!t) return;
      if (popRef.current?.contains(t)) return;
      if (anchor?.contains(t)) return;
      onCloseRef.current();
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onCloseRef.current();
      }
    };
    const timer = window.setTimeout(() => {
      document.addEventListener("pointerdown", away, true);
      document.addEventListener("keydown", key, true);
    }, 0);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", key, true);
    };
  }, [open, anchor]);

  function pickPreset(label: string, shut: boolean) {
    onAspect(label);
    onAspectSource?.(showDonors ? "pill" : undefined);
    if (shut) onClose();
  }

  function flip(wantTall: boolean) {
    if (wantTall === tall) return;
    setTall(wantTall);
    if (!checked) return;
    const shape = shapes.find(
      (s) => s.wide?.[0] === checked || s.tall?.[0] === checked,
    );
    if (!shape) return;
    const here = (wantTall ? shape.tall : shape.wide) ?? shape.tall ?? shape.wide;
    if (!here || here[0] === checked) return;
    pickPreset(here[0], false);
  }

  function showing(shape: AspectShape): [readonly [string, number], boolean] {
    const here = tall ? shape.tall : shape.wide;
    const entry = (here ?? shape.tall ?? shape.wide)!;
    return [entry, here == null];
  }

  if (!open) return null;

  return createPortal(
    <div
      ref={popRef}
      className="mmc-pop aspect-pop"
      role="dialog"
      aria-label="Aspect Ratio"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="mmc-pop-title">Aspect Ratio</div>

      {showDonors ? (
        <>
          <button
            type="button"
            className="mmc-opt"
            aria-checked={aspectSource === "auto"}
            disabled={disabled}
            title="Start frame decides, then footage, then the preset."
            onClick={() => {
              onAspectSource?.(undefined);
              onClose();
            }}
          >
            <span className="mmc-opt-label">
              <AspectGlyph ratio={autoRatio ?? ratioOfAspect(aspect)} />
              <span>Auto</span>
              {autoSub ? <span className="mmc-opt-sub">{autoSub}</span> : null}
            </span>
            <span className="mmc-radio" aria-hidden />
          </button>
          {donors.map((d) => (
            <button
              key={d.value}
              type="button"
              className="mmc-opt"
              aria-checked={aspectSource === d.value}
              disabled={disabled}
              onClick={() => {
                onAspectSource?.(d.value);
                onClose();
              }}
            >
              <span className="mmc-opt-label">
                <AspectGlyph ratio={d.ratio} />
                <span className={`mmc-ref mmc-tag-${d.tag}`}>{d.label}</span>
                <span className="mmc-opt-sub">{d.sub}</span>
              </span>
              <span className="mmc-radio" aria-hidden />
            </button>
          ))}
        </>
      ) : null}

      <div className="mmc-aspect-picker">
        <div className="mmc-aspect-head">
          {showDonors ? (
            <span className="mmc-pop-title mmc-aspect-title">Preset</span>
          ) : null}
          <div className="mmc-aspect-flip">
            <button
              type="button"
              className="mmc-flip-opt"
              aria-pressed={!tall}
              disabled={disabled}
              onClick={() => flip(false)}
            >
              Wide
            </button>
            <button
              type="button"
              className="mmc-flip-opt"
              aria-pressed={tall}
              disabled={disabled}
              onClick={() => flip(true)}
            >
              Tall
            </button>
          </div>
        </div>
        <div className="mmc-aspect-grid" role="radiogroup" aria-label="Aspect presets">
          {shapes.map((shape) => {
            const [[label, ratio], missing] = showing(shape);
            return (
              <button
                key={shape.key}
                type="button"
                className="mmc-aspect-tile"
                role="radio"
                aria-checked={label === checked}
                disabled={disabled || missing}
                title={
                  missing
                    ? `${label.split(":").reverse().join(":")} is outside this family's aspect range.`
                    : undefined
                }
                onClick={() => pickPreset(label, true)}
              >
                <AspectGlyph ratio={ratio} long={26} />
                <span className="mmc-aspect-num">{label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>,
    document.body,
  );
}
