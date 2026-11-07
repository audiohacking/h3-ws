/**
 * h3.c audio-reference caps (mirrors ``h3_media.MIN/MAX_REF_AUDIO_*``).
 * Long clips are auto-segmented instead of failing generate.
 */
import type { MediaTrim, ReferenceItem, RefKind } from "./types";

export const MIN_REF_AUDIO_S = 2;
export const MAX_REF_AUDIO_S = 15;
export const MAX_REF_AUDIO_TOTAL_S = 15;
/** Bake under h3.c's hard 15.000 s sample wall (resampler flush headroom). */
export const REF_AUDIO_BAKE_CAP_S = 14.9;

const SLACK = 0.05;

function carriesAudio(kind: RefKind): boolean {
  return kind === "audio" || kind === "video" || kind === "video_audio";
}

export function clampAudioSegment(
  durationS: number,
  start: number | null | undefined,
  end: number | null | undefined,
  maxSpanS: number,
): { start: number; end: number } {
  const dur = Math.max(0, durationS);
  let s = start == null ? 0 : start;
  let e = end == null ? dur : end;
  s = Math.max(0, Math.min(s, dur));
  e = Math.max(s, Math.min(e, dur));
  const maxSpan = Math.max(0, maxSpanS);
  if (e - s <= maxSpan + SLACK) return { start: s, end: e };
  const end2 = Math.min(dur, s + maxSpan);
  const start2 = Math.max(0, end2 - maxSpan);
  return { start: start2, end: end2 };
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

/**
 * Ensure every audio-bearing ref has a Continuity trim within the 15 s caps.
 * Returns the same array reference when nothing changed.
 */
export function fitRefAudioTrims(refs: ReferenceItem[]): {
  refs: ReferenceItem[];
  changed: boolean;
  note: string | null;
} {
  const audioIdx: number[] = [];
  for (let i = 0; i < refs.length; i += 1) {
    const r = refs[i];
    if (!r || r.enabled === false) continue;
    if (!carriesAudio(r.kind)) continue;
    if (r.durationS == null || !(r.durationS > 0)) continue;
    audioIdx.push(i);
  }
  if (audioIdx.length === 0) {
    return { refs, changed: false, note: null };
  }

  const rawSpans = audioIdx.map((i) => {
    const r = refs[i]!;
    const { start, end } = clampAudioSegment(
      r.durationS!,
      r.trim?.start,
      r.trim?.end,
      r.durationS!,
    );
    return end - start;
  });

  let remaining = REF_AUDIO_BAKE_CAP_S;
  const budgets = rawSpans.map((span) => {
    const allow = Math.min(span, REF_AUDIO_BAKE_CAP_S, remaining);
    remaining = Math.max(0, remaining - allow);
    return allow;
  });

  let changed = false;
  let note: string | null = null;
  const next = refs.map((r) => ({ ...r }));

  audioIdx.forEach((refIndex, j) => {
    const r = next[refIndex]!;
    const dur = r.durationS!;
    const budget = budgets[j]!;
    const { start, end } = clampAudioSegment(dur, r.trim?.start, r.trim?.end, budget);
    const span = end - start;
    const needsTrim = span + SLACK < dur || dur - SLACK > REF_AUDIO_BAKE_CAP_S;
    if (!needsTrim) return;
    const trim: MediaTrim = { start: round3(start), end: round3(end) };
    const prev = r.trim;
    if (
      !prev
      || Math.abs(prev.start - trim.start) > 0.01
      || Math.abs(prev.end - trim.end) > 0.01
    ) {
      next[refIndex] = { ...r, trim };
      changed = true;
      if (!note) {
        note =
          audioIdx.length > 1
            ? `Audio over ${MAX_REF_AUDIO_S}s total — segments capped (edit via the capsule door).`
            : `Audio is ${dur.toFixed(1)}s — using first ${span.toFixed(1)}s (edit via the capsule door).`;
      }
    }
  });

  return { refs: changed ? next : refs, changed, note };
}
