import type {
  Clip,
  Config,
  ReferenceItem,
  RoutingMode,
  SceneQueueItem,
} from "./types";
import { mediaSrc } from "./crop";

export function clipDisplayPrompt(prompt: string): string {
  return prompt.replace(/\s*\(×\d+ merged\)\s*$/i, "").trim();
}

const PASS_LABELS = ["UPFRAME", "REDETAIL", "LONGTAKE", "FACES", "BASE"] as const;

function generationPassEnabled(gen: Record<string, unknown> | null | undefined, key: string): boolean {
  if (!gen) return false;
  const v = gen[key];
  if (v === true) return true;
  if (v && typeof v === "object" && (v as { enabled?: boolean }).enabled) return true;
  return false;
}

/** Library badge for processed clips (Upframe / ReDetail / …), including older CURRENT-labeled ones. */
export function clipDisplayLabel(clip: Clip): string {
  const raw = (clip.label || "").trim();
  const upper = raw.toUpperCase().replace(/\s+/g, "");
  if ((PASS_LABELS as readonly string[]).includes(upper)) return upper;
  if (upper === "LONGTAKE" || raw.toUpperCase() === "LONG TAKE") return "LONGTAKE";

  const g = clip.generation;
  if (g && typeof g === "object") {
    const pass = typeof g.pass === "string" ? g.pass.trim().toUpperCase() : "";
    if ((PASS_LABELS as readonly string[]).includes(pass)) return pass;
    if (generationPassEnabled(g, "upframe")) return "UPFRAME";
    if (generationPassEnabled(g, "redetail")) return "REDETAIL";
    if (generationPassEnabled(g, "longtake")) return "LONGTAKE";
    if (generationPassEnabled(g, "faces")) return "FACES";
  }

  // Filename suffix from prior builds (stem_upframe.mp4).
  const name = (clip.filename || "").toLowerCase();
  if (name.includes("_upframe")) return "UPFRAME";
  if (name.includes("_redetail")) return "REDETAIL";
  if (name.includes("_longtake")) return "LONGTAKE";
  if (name.includes("_faces")) return "FACES";

  return raw || "CURRENT";
}

export function resolutionIdForClip(clip: Clip, config: Config | null): string {
  if (!clip.width || !clip.height) return "512x512";
  const presets = config?.resolution_presets ?? [];
  const native = presets.find(
    (r) => r.width === clip.width && r.height === clip.height && !r.render_width,
  );
  if (native) return native.id;
  const match = presets.find((r) => r.width === clip.width && r.height === clip.height);
  return match?.id ?? `${clip.width}x${clip.height}`;
}

export function durationIdForClip(clip: Clip, config: Config | null): string {
  if (clip.num_frames && config?.duration_presets) {
    const match = config.duration_presets.find((d) => d.num_frames === clip.num_frames);
    if (match) return match.id;
  }
  if (clip.duration_seconds != null) {
    const match = config?.duration_presets.find((d) => d.seconds === clip.duration_seconds);
    if (match) return match.id;
  }
  return config?.duration_presets[0]?.id ?? "1s";
}

/** Everything needed to put a clip back into the composer and re-generate. */
export interface GenerationSnapshot {
  prompt: string;
  mode: string;
  routing?: RoutingMode;
  selectedCastIds?: string[];
  quality: string;
  resolutionId: string;
  durationId: string;
  numSteps: number;
  layers: number;
  reuse: number;
  seed: string;
  loraPresetIds: string[];
  turboEnabled?: boolean;
  turboTier?: string;
  refs: ReferenceItem[];
  imagePath?: string | null;
  endImagePath?: string | null;
  clipMultiplier: number;
  tokenReduction?: boolean;
  ssdStreaming?: boolean;
  upscale?: boolean;
  /** Opt-in ReDetail → new library clip (independent of upframe). */
  redetail?: boolean;
  redetailDenoise?: number;
  redetailSeed?: number;
  /** Opt-in Upframe → new library clip (independent of redetail). */
  upframe?: boolean;
  upframeMultiplier?: number;
  /** Opt-in long-take refine → new library clip. */
  longtake?: boolean;
  longtakeDenoise?: number;
  longtakeKeyframes?: string;
  longtakeSeed?: number;
  longtakeInterpolation?: string;
  /** Opt-in Faces refine → new standalone library clip (source kept). */
  faces?: boolean;
  facesDenoise?: number;
  facesSeed?: number;
  facesCanvas?: number;
}

function passObj(raw: unknown): Record<string, unknown> | null {
  if (raw && typeof raw === "object") return raw as Record<string, unknown>;
  return null;
}

function passEnabledFlag(raw: unknown): boolean {
  if (raw === true) return true;
  const o = passObj(raw);
  return Boolean(o && o.enabled);
}

export interface ClipEditorSnapshot {
  prompt: string;
  mode: string;
  resolutionId: string;
  durationId: string;
  clipMultiplier: number;
  numSteps: number;
  layers: number;
  reuse: number;
  seed: string;
  quality: string;
}

export function snapshotFromClip(
  clip: Clip,
  config: Config | null,
  defaults: { numSteps: number; layers: number; reuse: number; quality: string },
): ClipEditorSnapshot {
  return {
    prompt: clipDisplayPrompt(clip.prompt),
    mode: clip.mode || "t2va",
    resolutionId: resolutionIdForClip(clip, config),
    durationId: durationIdForClip(clip, config),
    clipMultiplier: clip.clip_count ?? 1,
    numSteps: clip.num_steps ?? defaults.numSteps,
    layers: clip.layers ?? defaults.layers,
    reuse: clip.reuse ?? defaults.reuse,
    seed: clip.seed != null ? String(clip.seed) : "",
    quality: clip.quality ?? defaults.quality,
  };
}

function reviveRef(raw: unknown): ReferenceItem | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const path = String(r.path ?? "").trim();
  const kind = String(r.kind ?? "").trim();
  if (!path || !kind) return null;
  const id = String(r.id ?? `ref_${Math.random().toString(36).slice(2, 10)}`);
  const name = String(r.name ?? path.split("/").pop() ?? "ref");
  const trimRaw = r.trim;
  const cropRaw = r.crop;
  const item: ReferenceItem = {
    id,
    kind: kind as ReferenceItem["kind"],
    path,
    name,
    enabled: r.enabled === false ? false : true,
    previewUrl: mediaSrc(path, typeof r.previewUrl === "string" ? r.previewUrl : undefined),
    source: (r.source as ReferenceItem["source"]) || "upload",
  };
  if (typeof r.audioPath === "string" && r.audioPath) item.audioPath = r.audioPath;
  else if (typeof r.audio_path === "string" && r.audio_path) item.audioPath = r.audio_path;
  if (typeof r.refSize === "string") item.refSize = r.refSize as ReferenceItem["refSize"];
  else if (typeof r.ref_size === "string") item.refSize = r.ref_size as ReferenceItem["refSize"];
  if (typeof r.castId === "string") item.castId = r.castId;
  if (trimRaw && typeof trimRaw === "object") {
    const t = trimRaw as Record<string, unknown>;
    const start = Number(t.start ?? t.start_s);
    const end = Number(t.end ?? t.end_s);
    if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
      item.trim = { start, end };
    }
  }
  if (cropRaw && typeof cropRaw === "object") {
    item.crop = cropRaw as ReferenceItem["crop"];
  }
  return item;
}

/** Parse clip.generation (new) or fall back to flat clip fields (legacy). */
export function generationFromClip(
  clip: Clip,
  config: Config | null,
  defaults: { numSteps: number; layers: number; reuse: number; quality: string },
): GenerationSnapshot {
  const raw = clip.generation;
  if (raw && typeof raw === "object") {
    const g = raw as Record<string, unknown>;
    const refsIn = Array.isArray(g.refs) ? g.refs : [];
    const refs = refsIn.map(reviveRef).filter((r): r is ReferenceItem => r != null);
    const mode = String(g.mode ?? clip.mode ?? "t2va");
    const routing = (g.routing as RoutingMode | undefined)
      ?? (mode === "ref2va" ? "ref2va" : mode === "t2va" ? "auto" : "fl2va");
    return {
      prompt: String(g.prompt ?? clipDisplayPrompt(clip.prompt)),
      mode,
      routing,
      selectedCastIds: Array.isArray(g.selectedCastIds)
        ? g.selectedCastIds.map(String)
        : Array.isArray(g.selected_cast_ids)
          ? (g.selected_cast_ids as unknown[]).map(String)
          : [],
      quality: String(g.quality ?? clip.quality ?? defaults.quality),
      resolutionId: String(g.resolutionId ?? g.resolution_id ?? resolutionIdForClip(clip, config)),
      durationId: String(g.durationId ?? g.duration_id ?? durationIdForClip(clip, config)),
      numSteps: Number(g.numSteps ?? g.num_steps ?? clip.num_steps ?? defaults.numSteps),
      layers: Number(g.layers ?? clip.layers ?? defaults.layers),
      reuse: Number(g.reuse ?? clip.reuse ?? defaults.reuse),
      seed: g.seed != null && String(g.seed) !== "" ? String(g.seed) : (clip.seed != null ? String(clip.seed) : ""),
      loraPresetIds: Array.isArray(g.loraPresetIds)
        ? g.loraPresetIds.map(String)
        : Array.isArray(g.lora_preset_ids)
          ? (g.lora_preset_ids as unknown[]).map(String)
          : (clip.loras ?? []).map((l) => String((l as { id?: string }).id ?? "")).filter(Boolean),
      turboEnabled: Boolean(g.turboEnabled ?? g.turbo_enabled),
      turboTier: g.turboTier != null ? String(g.turboTier) : g.turbo_tier != null ? String(g.turbo_tier) : undefined,
      refs,
      imagePath: (g.imagePath ?? g.image_path ?? null) as string | null,
      endImagePath: (g.endImagePath ?? g.end_image_path ?? null) as string | null,
      clipMultiplier: Number(g.clipMultiplier ?? g.clip_multiplier ?? clip.clip_count ?? 1),
      tokenReduction: Boolean(g.tokenReduction ?? g.token_reduction),
      ssdStreaming: Boolean(g.ssdStreaming ?? g.ssd_streaming),
      upscale: Boolean(g.upscale),
      redetail: passEnabledFlag(g.redetail),
      redetailDenoise: (() => {
        const o = passObj(g.redetail);
        const n = Number(o?.denoise ?? g.redetailDenoise ?? g.redetail_denoise);
        return Number.isFinite(n) ? n : undefined;
      })(),
      redetailSeed: (() => {
        const o = passObj(g.redetail);
        const n = Number(o?.seed ?? g.redetailSeed ?? g.redetail_seed);
        return Number.isFinite(n) ? n : undefined;
      })(),
      upframe: passEnabledFlag(g.upframe),
      upframeMultiplier: (() => {
        const o = passObj(g.upframe);
        const n = Number(o?.multiplier ?? g.upframeMultiplier ?? g.upframe_multiplier);
        return Number.isFinite(n) && n >= 2 ? Math.round(n) : undefined;
      })(),
      longtake: passEnabledFlag(g.longtake),
      longtakeDenoise: (() => {
        const o = passObj(g.longtake);
        const n = Number(o?.denoise ?? g.longtakeDenoise ?? g.longtake_denoise);
        return Number.isFinite(n) ? n : undefined;
      })(),
      longtakeKeyframes: (() => {
        const o = passObj(g.longtake);
        const s = o?.keyframes ?? g.longtakeKeyframes ?? g.longtake_keyframes;
        return typeof s === "string" && s.trim() ? s : undefined;
      })(),
      longtakeSeed: (() => {
        const o = passObj(g.longtake);
        const n = Number(o?.seed ?? g.longtakeSeed ?? g.longtake_seed);
        return Number.isFinite(n) ? n : undefined;
      })(),
      longtakeInterpolation: (() => {
        const o = passObj(g.longtake);
        const s = o?.interpolation ?? g.longtakeInterpolation ?? g.longtake_interpolation;
        return typeof s === "string" && s.trim() ? s : undefined;
      })(),
      faces: passEnabledFlag(g.faces),
      facesDenoise: (() => {
        const o = passObj(g.faces);
        const n = Number(o?.denoise ?? g.facesDenoise ?? g.faces_denoise);
        return Number.isFinite(n) ? n : undefined;
      })(),
      facesSeed: (() => {
        const o = passObj(g.faces);
        const n = Number(o?.seed ?? g.facesSeed ?? g.faces_seed);
        return Number.isFinite(n) ? n : undefined;
      })(),
      facesCanvas: (() => {
        const o = passObj(g.faces);
        const n = Number(o?.canvas ?? g.facesCanvas ?? g.faces_canvas);
        return Number.isFinite(n) ? n : undefined;
      })(),
    };
  }

  const legacy = snapshotFromClip(clip, config, defaults);
  return {
    ...legacy,
    routing: legacy.mode === "ref2va" ? "ref2va" : legacy.mode === "t2va" ? "auto" : "fl2va",
    selectedCastIds: [],
    loraPresetIds: (clip.loras ?? []).map((l) => String((l as { id?: string }).id ?? "")).filter(Boolean),
    refs: [],
    imagePath: null,
    endImagePath: null,
    tokenReduction: false,
    ssdStreaming: false,
    redetail: false,
    upframe: false,
    longtake: false,
    faces: false,
  };
}

export function composerIsEmpty(state: {
  prompt: string;
  refs: ReferenceItem[];
  imagePath: string | null;
  endImagePath: string | null;
}): boolean {
  return (
    !state.prompt.trim()
    && state.refs.length === 0
    && !state.imagePath
    && !state.endImagePath
  );
}

/** Build the blob we persist on the clip at generate time. */
export function buildGenerationSnapshot(scene: Pick<
  SceneQueueItem,
  | "prompt"
  | "mode"
  | "routing"
  | "selectedCastIds"
  | "quality"
  | "resolutionId"
  | "durationId"
  | "numSteps"
  | "layers"
  | "reuse"
  | "seed"
  | "loraPresetIds"
  | "turboEnabled"
  | "turboTier"
  | "refs"
  | "imagePath"
  | "endImagePath"
  | "clipMultiplier"
  | "tokenReduction"
  | "ssdStreaming"
  | "longtake"
  | "longtakeDenoise"
  | "longtakeKeyframes"
  | "longtakeSeed"
  | "longtakeInterpolation"
  | "redetail"
  | "redetailDenoise"
  | "redetailSeed"
  | "upframe"
  | "upframeMultiplier"
  | "faces"
  | "facesDenoise"
  | "facesSeed"
  | "facesCanvas"
> & { upscale?: boolean }): GenerationSnapshot {
  return {
    prompt: scene.prompt,
    mode: scene.mode,
    routing: scene.routing,
    selectedCastIds: [...(scene.selectedCastIds ?? [])],
    quality: scene.quality,
    resolutionId: scene.resolutionId,
    durationId: scene.durationId,
    numSteps: scene.numSteps,
    layers: scene.layers,
    reuse: scene.reuse,
    seed: scene.seed,
    loraPresetIds: [...scene.loraPresetIds],
    turboEnabled: scene.turboEnabled,
    turboTier: scene.turboTier,
    refs: scene.refs.map((r) => ({
      ...r,
      // Drop blob: URLs — they die with the session; path is enough to revive.
      previewUrl: undefined,
    })),
    imagePath: scene.imagePath ?? null,
    endImagePath: scene.endImagePath ?? null,
    clipMultiplier: scene.clipMultiplier,
    tokenReduction: scene.tokenReduction,
    ssdStreaming: scene.ssdStreaming,
    upscale: scene.upscale,
    redetail: scene.redetail,
    redetailDenoise: scene.redetailDenoise,
    redetailSeed: scene.redetailSeed,
    upframe: scene.upframe,
    upframeMultiplier: scene.upframeMultiplier,
    longtake: scene.longtake,
    longtakeDenoise: scene.longtakeDenoise,
    longtakeKeyframes: scene.longtakeKeyframes,
    longtakeSeed: scene.longtakeSeed,
    longtakeInterpolation: scene.longtakeInterpolation,
    faces: scene.faces,
    facesDenoise: scene.facesDenoise,
    facesSeed: scene.facesSeed,
    facesCanvas: scene.facesCanvas,
  };
}
