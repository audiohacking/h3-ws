import type { ModelProgress, ProgressState } from "./types";

/** tqdm-style clock: 54 → "00:54", 125 → "02:05" */
export function formatMmSs(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export function progressFromModel(
  mp?: ModelProgress | null,
  elapsed_s?: number,
): Partial<ProgressState> {
  if (!mp) {
    return elapsed_s != null ? { elapsed_s } : {};
  }
  return {
    stage: mp.stage,
    step: mp.step,
    total: mp.total,
    pct: mp.pct,
    eta_s: mp.eta_s,
    elapsed_s: mp.elapsed_s ?? elapsed_s,
  };
}

/** Match CLI tqdm: ``2/8 [00:18<00:54, 9.11s/it]`` — lead with remaining time. */
export function formatProgressMessage(
  mp?: ModelProgress | null,
  _elapsed_s?: number,
): string {
  if (!mp?.stage && mp?.step == null && !mp?.label) {
    return "Generating…";
  }
  const stage = (mp?.stage || "generating").toLowerCase();
  if (stage === "starting" || stage === "loading") {
    const label = (mp?.label || "").trim();
    if (label) return label;
    return "Loading model weights…";
  }

  const parts: string[] = [];
  parts.push(stage.charAt(0).toUpperCase() + stage.slice(1));

  if (mp?.step != null && mp?.total != null) {
    parts.push(`${mp.step}/${mp.total}`);
  }

  if (mp?.eta_s != null) {
    parts.push(`${formatMmSs(mp.eta_s)} remaining`);
  } else if (mp?.pct != null) {
    parts.push(`${mp.pct}%`);
  }

  if (mp?.avg_step_s != null) {
    parts.push(`${mp.avg_step_s}s/it`);
  }

  return parts.join(" · ");
}

/** True for twin/post-pass phases that wrap a nested H3 denoise. */
export function isPostPassPhase(phase: string | undefined | null): boolean {
  if (!phase) return false;
  return (
    phase.startsWith("longtake_")
    || phase.startsWith("redetail_")
    || phase.startsWith("upframe_")
    || phase.startsWith("faces_")
  );
}

/**
 * Keep the post-pass window label and fold in live nested denoise (stage/step/eta).
 * Continuity-style: "Faces window 1/2 · … · Video vae decode · 2/8 · 00:54 remaining".
 */
export function composePostPassMessage(
  explicit: string | null | undefined,
  mp?: ModelProgress | null,
  elapsed_s?: number,
): string {
  const label = (explicit || "").trim();
  const nested = formatProgressMessage(mp, elapsed_s);
  const hasNested = Boolean(mp?.stage || mp?.step != null || mp?.label);
  if (!label) return nested;
  if (!hasNested || nested === "Generating…") return label;
  // Server may already have composed (longtake / faces) — don't double-append.
  const stage = (mp?.stage || "").trim().toLowerCase();
  if (stage && label.toLowerCase().includes(stage)) return label;
  if (mp?.step != null && mp?.total != null && label.includes(`${mp.step}/${mp.total}`)) {
    return label;
  }
  return `${label} · ${nested}`;
}

export function applyProgressEvent(
  prev: ProgressState | null,
  msg: Record<string, unknown>,
): ProgressState {
  const mp = (msg.model_progress as ModelProgress | undefined) ?? undefined;
  const wall_elapsed =
    typeof msg.elapsed_s === "number" ? msg.elapsed_s : prev?.elapsed_s;
  const phase =
    typeof msg.phase === "string"
      ? msg.phase
      : mp?.stage ?? prev?.phase ?? "generating";
  const hasStepData = Boolean(mp?.stage || mp?.step != null || mp?.label);
  const explicit =
    typeof msg.message === "string" && msg.message.trim() ? msg.message.trim() : null;
  // If a keepalive/generic tick arrives mid post-pass without the faces_/…
  // phase, keep the post-pass phase so the overlay does not snap back to a
  // bare nested stage (e.g. "video vae decode").
  const phaseOut =
    !isPostPassPhase(phase) && isPostPassPhase(prev?.phase) && !explicit
      ? (prev!.phase)
      : phase;
  const postPass = isPostPassPhase(phaseOut);
  // Post-pass: never replace the window label with a bare nested stage
  // ("video vae decode") — that made Faces look like the baseline pass.
  const message = postPass
    ? composePostPassMessage(
        explicit
          ?? (isPostPassPhase(prev?.phase) ? stripNestedProgress(prev?.message) : null),
        mp,
        wall_elapsed,
      )
    : hasStepData
      ? formatProgressMessage(mp, wall_elapsed)
      : explicit ?? prev?.message ?? formatProgressMessage(mp, wall_elapsed);
  return {
    phase: phaseOut,
    message,
    ...progressFromModel(mp, wall_elapsed),
  };
}

/** Drop a previously appended ``· Stage · n/m · …`` tail so we can recompose. */
function stripNestedProgress(message: string | undefined | null): string | null {
  if (!message) return null;
  // Window labels use " · " too (ceiling · video_t · mode). Only strip from the
  // first token that looks like a denoise stage / step counter.
  const parts = message.split(" · ");
  if (parts.length < 2) return message;
  const stagey = /^(video|audio|text|dit|vae|decode|encode|denoise|loading|starting)\b/i;
  const stepCounter = /^\d+\/\d+$/;
  let cut = parts.length;
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i]!.trim();
    if (stagey.test(p) || stepCounter.test(p) || /remaining$/i.test(p) || /s\/it$/i.test(p)) {
      cut = i;
      break;
    }
  }
  return parts.slice(0, cut).join(" · ").trim() || message;
}
