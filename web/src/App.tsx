import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildGenerationSnapshot, clipDisplayPrompt, composerIsEmpty, generationFromClip } from "./clipEditor";
import { applyProgressEvent } from "./progress";
import { captureVideoFrame, formatVideoTime } from "./frameCapture";
import { refsAreValid } from "./components/composer/RefListEnhanced";
import { generateId } from "./utils";
import { FEATURES, TURBO_CONFIG, type TurboTier } from "./config";
import { LoraModal } from "./components/lora/LoraModal";
import { TimelineStrip } from "./components/timeline/TimelineStrip";
import { ModelsManager } from "./components/media/ModelsManager";
import { ComposerPanel } from "./components/composer/ComposerPanel";
import { isTurboPreset, pickDefaultTurboId } from "./components/composer/TurboToggle";
import { FeaturesPopup } from "./components/composer/FeaturesPopup";
import { DebugModal } from "./components/DebugModal";
import { RefinePanel } from "./components/composer/RefinePanel";
import { ProjectSwitcher, type Project } from "./components/ProjectSwitcher";
import { compilePrompt } from "./compile";
import { leadWithStyle } from "./styleAtlas";
import {
  applyTheme,
  cycleTheme,
  getStoredTheme,
  resolveTheme,
  setThemePreference,
  subscribeSystemTheme,
  themeLabel,
  type ThemePreference,
} from "./theme";
import type { CastMediaType, CastMember, CastMedia, Clip, Config, GenerationPreset, LibraryFrame, LoraPreset, NetworkSettingsPublic, PillOption, ProgressState, QualityPreset, ReferenceItem, RefineSettingsPublic, RefKind, RoutingMode, SceneQueueItem, UpdateCheckPublic } from "./types";

const H3_DEFAULT_STEPS = 20;
const H3_DEFAULT_LAYERS = 50;
const H3_DEFAULT_REUSE = 1;

/** Composer base strip — distill / aggressive stay API-only. */
const BASE_QUALITY_IDS = new Set(["fast", "balanced", "close"]);

function coerceBaseQuality(id: string | undefined | null): string {
  const raw = (id || "fast").trim().toLowerCase();
  return BASE_QUALITY_IDS.has(raw) ? raw : "balanced";
}

function ThemeIcon({ pref }: { pref: ThemePreference }) {
  if (pref === "light") {
    return (
      <svg className="theme-toggle__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
    );
  }
  if (pref === "dark") {
    return (
      <svg className="theme-toggle__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden>
        <path d="M21 14.5A8.5 8.5 0 0 1 9.5 3 7 7 0 1 0 21 14.5z" />
      </svg>
    );
  }
  return (
    <svg className="theme-toggle__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v18" />
      <path d="M12 3a9 9 0 0 0 0 18" fill="currentColor" fillOpacity="0.22" stroke="none" />
    </svg>
  );
}

function fieldsFromPreset(preset: QualityPreset | undefined) {
  const steps = preset?.steps ?? H3_DEFAULT_STEPS;
  return {
    steps,
    layers: preset?.layers ?? H3_DEFAULT_LAYERS,
    reuse: steps <= 7 ? 1 : (preset?.reuse ?? H3_DEFAULT_REUSE),
    tokenReduction: Boolean(preset?.token_reduction),
  };
}

const API = "";
const BLOB_VIDEO_PREFIX = "blob:";
const NOTIFY_READY_KEY = "h3-ws-notify-on-ready";
const NOTIFY_ASKED_KEY = "h3-ws-notify-permission-asked";

/** localhost / 127.0.0.1 count as secure; https-only checks broke desktop notifications. */
function isNotifyCapableContext(): boolean {
  try {
    return typeof window !== "undefined" && Boolean(window.isSecureContext);
  } catch {
    return false;
  }
}

function persistNotifyDecision(enabled: boolean) {
  try {
    localStorage.setItem(NOTIFY_ASKED_KEY, "1");
    localStorage.setItem(NOTIFY_READY_KEY, enabled ? "1" : "0");
  } catch {
    /* ignore */
  }
}

function notificationsWanted(): boolean {
  try {
    return localStorage.getItem(NOTIFY_READY_KEY) === "1";
  } catch {
    return false;
  }
}

/** Ask once, preferably from a click (Generate). Safe no-op if unsupported. */
async function ensureNotifyPermission(): Promise<void> {
  if (!isNotifyCapableContext()) return;
  if (typeof Notification === "undefined") return;
  try {
    if (localStorage.getItem(NOTIFY_ASKED_KEY) === "1") return;
  } catch {
    return;
  }
  if (Notification.permission === "granted") {
    persistNotifyDecision(true);
    return;
  }
  if (Notification.permission === "denied") {
    persistNotifyDecision(false);
    return;
  }
  try {
    const permission = await Notification.requestPermission();
    persistNotifyDecision(permission === "granted");
  } catch {
    persistNotifyDecision(false);
  }
}

function notifyApp(body: string, tag: string) {
  if (!isNotifyCapableContext()) return;
  if (!notificationsWanted()) return;
  if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
  try {
    new Notification("H3-WS", { body, tag });
  } catch {
    /* pywebview / older WebKit may lack Notification */
  }
}

function notifyGenerationReady(body = "Your video is ready to play.") {
  notifyApp(body, "h3-ws-generation-ready");
}

function notifyGenerationError(body: string) {
  notifyApp(body || "Generation failed", "h3-ws-generation-error");
}

function revokeClipBlob(clip: Clip) {
  if (clip.video_url?.startsWith(BLOB_VIDEO_PREFIX)) URL.revokeObjectURL(clip.video_url);
}

function revokeBlobVideoUrls(clips: Clip[]) {
  for (const clip of clips) revokeClipBlob(clip);
}

function preserveBlobVideoUrls(prev: Clip[], incoming: Clip[]): Clip[] {
  const blobById = new Map(
    prev
      .filter((c) => c.video_url?.startsWith(BLOB_VIDEO_PREFIX))
      .map((c) => [c.id, c.video_url] as const),
  );
  return incoming.map((c) => {
    const blob = blobById.get(c.id);
    return blob ? { ...c, video_url: blob } : c;
  });
}

function mergeClips(prev: Clip[], incoming: Clip[]): Clip[] {
  const byId = new Map(prev.map((c) => [c.id, c]));
  for (const c of incoming) byId.set(c.id, c);
  return Array.from(byId.values());
}

function replaceChainClips(prev: Clip[], chainId: string, chainClips: Clip[]): Clip[] {
  const rest = prev.filter((c) => c.chain_id !== chainId);
  return [...rest, ...preserveBlobVideoUrls(prev, chainClips)];
}

const ROUNDED_DURATIONS: [number, number][] = [
  [22, 1],
  [56, 2],
  [107, 5],
  [243, 10],
  [362, 15],
];

function formatDuration(frames?: number, fps = 24) {
  if (!frames) return "";
  const hit = ROUNDED_DURATIONS.find(([nf]) => nf === frames);
  if (hit) return `${hit[1]}s`;
  return `${Math.round(frames / fps)}s`;
}

function pickPlaybackClip(clips: Clip[], chainId: string): string | null {
  const chain = clips.filter((c) => c.chain_id === chainId && c.status === "done" && c.video_url);
  const merged = chain.find((c) => c.label === "MERGED");
  const current = chain.find((c) => c.label === "CURRENT");
  const latest = [...chain].sort((a, b) => b.clip_index - a.clip_index)[0];
  return merged?.id ?? current?.id ?? latest?.id ?? null;
}

async function fetchConfig(): Promise<Config> {
  const r = await fetch(`${API}/api/config`);
  if (!r.ok) throw new Error("Failed to load config");
  return r.json();
}

async function fetchClips(chainId?: string): Promise<Clip[]> {
  const q = chainId ? `?chain_id=${encodeURIComponent(chainId)}` : "";
  const r = await fetch(`${API}/api/clips${q}`);
  if (!r.ok) throw new Error("Failed to load clips");
  const data = await r.json();
  return data.clips as Clip[];
}

/** Deep-link: `?id=<clip_id>` opens that clip. Bare reload stays a clean canvas. */
function clipIdFromUrl(): string | null {
  try {
    const id = new URLSearchParams(window.location.search).get("id");
    return id?.trim() || null;
  } catch {
    return null;
  }
}

function writeClipIdToUrl(clipId: string | null) {
  try {
    const url = new URL(window.location.href);
    if (clipId) url.searchParams.set("id", clipId);
    else url.searchParams.delete("id");
    const next = `${url.pathname}${url.search}${url.hash}`;
    const cur = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (next !== cur) window.history.replaceState(null, "", next);
  } catch {
    /* ignore */
  }
}

async function fetchProjects(): Promise<{ projects: Project[]; active_project_id: string | null }> {
  const r = await fetch(`${API}/api/projects`);
  if (!r.ok) throw new Error("Failed to load projects");
  const data = await r.json();
  return {
    projects: (data.projects ?? []) as Project[],
    active_project_id: (data.active_project_id as string | null) ?? null,
  };
}

async function fetchFrames(): Promise<LibraryFrame[]> {
  const r = await fetch(`${API}/api/frames`);
  if (!r.ok) throw new Error("Failed to load frames");
  const data = await r.json();
  return (data.frames ?? []) as LibraryFrame[];
}

async function fetchCastMembers(): Promise<CastMember[]> {
  const r = await fetch(`${API}/api/cast`);
  if (!r.ok) throw new Error("Failed to load cast");
  const data = await r.json();
  return (data.cast ?? []) as CastMember[];
}

async function fetchBackendPresets(): Promise<GenerationPreset[]> {
  const r = await fetch(`${API}/api/presets`);
  if (!r.ok) throw new Error("Failed to load presets");
  const data = await r.json();
  return (data.presets ?? []) as GenerationPreset[];
}

async function createPreset(preset: GenerationPreset): Promise<void> {
  await fetch(`${API}/api/presets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preset),
  });
}

async function uploadFile(file: File, kind: string): Promise<{ path: string; durationS?: number; filename?: string; hasAudio?: boolean }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/api/upload?kind=${encodeURIComponent(kind)}`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error("Upload failed");
  const data = await r.json();
  return {
    path: data.path as string,
    durationS: typeof data.duration_s === "number" ? data.duration_s : undefined,
    filename: data.filename as string | undefined,
    hasAudio: typeof data.has_audio === "boolean" ? data.has_audio : undefined,
  };
}

const IMAGE_MODES = new Set(["first_frame", "fl2va"]);

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [frameLibrary, setFrameLibrary] = useState<LibraryFrame[]>([]);
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState("ref2va");
  const [routing, setRouting] = useState<RoutingMode>("ref2va");
  const [quality, setQuality] = useState("fast");
  const [resolutionId, setResolutionId] = useState("512x512");
  const [durationId, setDurationId] = useState("1s");
  const [clipMultiplier, setClipMultiplier] = useState(1);
  const [numSteps, setNumSteps] = useState(H3_DEFAULT_STEPS);
  const [layers, setLayers] = useState(H3_DEFAULT_LAYERS);
  const [reuse, setReuse] = useState(H3_DEFAULT_REUSE);
  const [seed, setSeed] = useState("");
  const [ssdStreaming, setSsdStreaming] = useState(false);
  const [tokenReduction, setTokenReduction] = useState(true);
  // Lanczos post-upscale retired — wait for latent upscale. Keep false so old
  // snapshots / generate bodies never re-arm the timeline-dupe path.
  const upscale = false;
  const [loraPresetIds, setLoraPresetIds] = useState<string[]>([]);
  const [loraPresets, setLoraPresets] = useState<LoraPreset[]>([]);
  const [addingCustomLora, setAddingCustomLora] = useState(false);
  const [loraBusy, setLoraBusy] = useState(false);
  const [loraActivity, setLoraActivity] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  /** Player shows live preview / progress for the followed run (vs a finished clip). */
  const [watchLive, setWatchLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [chainId, setChainId] = useState<string | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(() => clipIdFromUrl());
  const urlClipHydratedRef = useRef(false);
  const [clipsReady, setClipsReady] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const activeRunIdRef = useRef<string | null>(null);
  activeRunIdRef.current = activeRunId;
  const watchLiveRef = useRef(false);
  watchLiveRef.current = watchLive;
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [endImagePath, setEndImagePath] = useState<string | null>(null);
  const [endImageName, setEndImageName] = useState<string | null>(null);
  const [refs, setRefs] = useState<ReferenceItem[]>([]);
  const [savingFrame, setSavingFrame] = useState(false);
  const [turboEnabled, setTurboEnabled] = useState(false);
  const [turboTier, setTurboTier] = useState<TurboTier>(TURBO_CONFIG.DEFAULT_TIER);
  const [turboLoading, setTurboLoading] = useState(false);
  const [turboLoraId, setTurboLoraId] = useState<string | null>(null);
  const [facesEnabled, setFacesEnabled] = useState(false);
  const [facesCanvas, setFacesCanvas] = useState(512);
  const [facesDenoise, setFacesDenoise] = useState(0.45);
  const [facesSeed, setFacesSeed] = useState(42);
  const [loraModalOpen, setLoraModalOpen] = useState(false);
  const [sceneQueue, setSceneQueue] = useState<SceneQueueItem[]>([]);
  const [queueRunning, setQueueRunning] = useState(false);
  const [generationPresets, setGenerationPresets] = useState<GenerationPreset[]>([]);
  const [castMembers, setCastMembers] = useState<CastMember[]>([]);
  const [selectedCastIds, setSelectedCastIds] = useState<string[]>([]);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [modelsDownloadActive, setModelsDownloadActive] = useState(false);
  const [modelsCloseWarning, setModelsCloseWarning] = useState(false);
  const [lockedClipIds, setLockedClipIds] = useState<Set<string>>(new Set());
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [framesOpen, setFramesOpen] = useState(true);
  const [activeStyleId, setActiveStyleId] = useState<string | null>(null);
  const [livePreviewUrl, setLivePreviewUrl] = useState<string | null>(null);
  const [livePreviewMime, setLivePreviewMime] = useState<string>("image/webp");
  const [refineSettings, setRefineSettings] = useState<RefineSettingsPublic>({
    enabled: false,
    base_url: "",
    model: "",
    key_set: false,
  });
  const [networkSettings, setNetworkSettings] = useState<NetworkSettingsPublic | null>(null);
  const [updateInfo, setUpdateInfo] = useState<UpdateCheckPublic | null>(null);
  const [updateDismissed, setUpdateDismissed] = useState(false);
  const [featuresOpen, setFeaturesOpen] = useState(false);
  const [themePref, setThemePref] = useState<ThemePreference>(() => getStoredTheme());
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [refineOpen, setRefineOpen] = useState(false);
  const [refineBusy, setRefineBusy] = useState(false);
  const [refineError, setRefineError] = useState<string | null>(null);
  const [refineDraft, setRefineDraft] = useState("");
  const [refineOriginal, setRefineOriginal] = useState("");
  const playerVideoRef = useRef<HTMLVideoElement>(null);
  const runEventSourceRef = useRef<EventSource | null>(null);
  const clipsRef = useRef<Clip[]>([]);
  clipsRef.current = clips;

  const inProgressClips = useMemo(
    () =>
      clips
        .filter((c) => c.status === "queued" || c.status === "running")
        .slice()
        .sort((a, b) => (a.created_at || "").localeCompare(b.created_at || "")),
    [clips],
  );
  const pipelineActive = busy || inProgressClips.length > 0;
  const libraryClips = useMemo(
    () =>
      clips
        .filter((c) => c.status === "done" && c.video_url)
        .slice()
        .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")),
    [clips],
  );
  const framesNewest = useMemo(
    () => [...frameLibrary].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")),
    [frameLibrary],
  );
  const activeClip = useMemo(() => {
    if (!selectedClipId) return null;
    return clips.find((c) => c.id === selectedClipId) ?? null;
  }, [clips, selectedClipId]);
  const chainParts = useMemo(
    () => (chainId ? clips.filter((c) => c.chain_id === chainId) : []),
    [clips, chainId],
  );
  const showChainPicker = chainParts.filter((c) => c.video_url).length > 1;
  const compiled = useMemo(
    () =>
      compilePrompt({
        prompt,
        refs,
        castMembers,
        selectedCastIds,
        routing,
        imagePath,
        endImagePath,
      }),
    [prompt, refs, castMembers, selectedCastIds, routing, imagePath, endImagePath],
  );
  const effectiveMode = compiled.modeHint;
  const isRef2va = effectiveMode === "ref2va";
  const needsFirst = IMAGE_MODES.has(effectiveMode) && !isRef2va;
  const previewCanvas = resolutionId === "256x256";
  const aggressiveInternal = resolutionId === "512x512-aggressive";
  const tokenReductionLocked =
    previewCanvas ||
    aggressiveInternal ||
    quality === "aggressive" ||
    (layers === 40 && reuse === 3) ||
    loraPresetIds.length > 0;
  const qualityOptions: PillOption[] = useMemo(
    () =>
      (config?.quality_presets ?? [])
        .filter((p) => p.ui !== false && BASE_QUALITY_IDS.has(p.id))
        .map((p) => ({
          id: p.id,
          label: p.label,
          shortLabel: ({
            fast: "fast",
            balanced: "bal",
            close: "close",
          } as Record<string, string>)[p.id],
          description: p.guidance ?? undefined,
        })),
    [config?.quality_presets],
  );

  useEffect(() => {
    applyTheme(themePref);
    return subscribeSystemTheme(() => {
      if (getStoredTheme() === "auto") applyTheme("auto");
    });
  }, [themePref]);

  useEffect(() => {
    // Best-effort warm ask; Generate also requests (needs a user gesture in some WebKits).
    void ensureNotifyPermission();
    fetchConfig()
      .then((cfg) => {
        setConfig(cfg);
        const qid = coerceBaseQuality(cfg.defaults.quality ?? "fast");
        setQuality(qid);
        const preset = cfg.quality_presets.find((p) => p.id === qid);
        const fields = fieldsFromPreset(preset);
        setNumSteps(fields.steps);
        setLayers(fields.layers);
        setReuse(fields.reuse);
        setTokenReduction(fields.tokenReduction);
        setSsdStreaming(false);
        setLoraPresets(cfg.lora_presets ?? []);
        setTurboLoraId((prev) => prev ?? pickDefaultTurboId(cfg.lora_presets ?? []));
        if (cfg.refine) setRefineSettings(cfg.refine);
        if (cfg.faces) {
          setFacesCanvas(cfg.faces.canvas);
          setFacesDenoise(cfg.faces.denoise);
        }
        if (cfg.network) setNetworkSettings(cfg.network);
        const defRes =
          cfg.resolution_presets.find((r) => r.id === "512x512") ??
          cfg.resolution_presets.find(
            (r) =>
              r.width === cfg.defaults.width &&
              r.height === cfg.defaults.height &&
              !r.render_width,
          );
        if (defRes) setResolutionId(defRes.id);
        const defDur = cfg.duration_presets.find((d) => d.num_frames === cfg.defaults.num_frames);
        if (defDur) setDurationId(defDur.id);
      })
      .catch((e) => setError(String(e)));
    fetch(`${API}/api/update`)
      .then(async (r) => {
        if (!r.ok) return;
        const data = (await r.json()) as UpdateCheckPublic;
        setUpdateInfo(data);
      })
      .catch(() => {
        /* offline / rate-limit — ignore */
      });
    fetchClips()
      .then((c) => {
        setClips(c);
        setClipsReady(true);
      })
      .catch(() => setClipsReady(true));
    fetchProjects()
      .then((data) => {
        setProjects(data.projects);
        setActiveProjectId(data.active_project_id);
      })
      .catch(() => undefined);
    fetchFrames().then(setFrameLibrary).catch(() => undefined);
    fetchCastMembers().then(setCastMembers).catch(() => undefined);
    fetchBackendPresets().then((presets) => {
      if (presets.length > 0) setGenerationPresets(presets);
    }).catch(() => undefined);
    return () => {
      runEventSourceRef.current?.close();
      revokeBlobVideoUrls(clipsRef.current);
    };
  }, []);

  const selectClipId = useCallback((clipId: string | null) => {
    setSelectedClipId(clipId);
    writeClipIdToUrl(clipId);
  }, []);

  const applyClipSelection = useCallback(
    (clip: Clip) => {
      // Leave the run subscribed in the background; library "In progress" brings it back.
      if (clip.status === "done") setWatchLive(false);
      if (!config) {
        selectClipId(clip.id);
        setChainId(clip.chain_id);
        return;
      }
      const occupied = !composerIsEmpty({ prompt, refs, imagePath, endImagePath });
      if (occupied) {
        const ok = window.confirm(
          "Replace the current composer with this clip’s settings (prompt, references, mode, sampler)?",
        );
        if (!ok) {
          // Still select for playback without stomping the composer.
          selectClipId(clip.id);
          setChainId(clip.chain_id);
          return;
        }
      }
      selectClipId(clip.id);
      setChainId(clip.chain_id);
      const snap = generationFromClip(clip, config, {
        numSteps: config.defaults.num_steps,
        layers: config.defaults.layers ?? H3_DEFAULT_LAYERS,
        reuse: config.defaults.reuse ?? H3_DEFAULT_REUSE,
        quality: config.defaults.quality ?? "fast",
      });
      setPrompt(snap.prompt);
      setMode(snap.mode);
      setRouting(
        snap.routing
          ?? (snap.mode === "ref2va" ? "ref2va" : snap.mode === "t2va" ? "auto" : "fl2va"),
      );
      setSelectedCastIds(snap.selectedCastIds ?? []);
      setResolutionId(snap.resolutionId);
      setDurationId(snap.durationId);
      setClipMultiplier(snap.clipMultiplier);
      setNumSteps(snap.numSteps);
      setLayers(snap.layers);
      setReuse(snap.reuse);
      setSeed(snap.seed);
      setQuality(coerceBaseQuality(snap.quality));
      setLoraPresetIds(snap.loraPresetIds ?? []);
      setTurboEnabled(Boolean(snap.turboEnabled));
      if (snap.turboTier) setTurboTier(snap.turboTier as TurboTier);
      setRefs(snap.refs ?? []);
      setImagePath(snap.imagePath ?? null);
      setImageName(snap.imagePath ? snap.imagePath.split("/").pop() ?? "start" : null);
      setEndImagePath(snap.endImagePath ?? null);
      setEndImageName(snap.endImagePath ? snap.endImagePath.split("/").pop() ?? "end" : null);
      setTokenReduction(Boolean(snap.tokenReduction));
      setSsdStreaming(Boolean(snap.ssdStreaming));
    },
    [config, selectClipId, prompt, refs, imagePath, endImagePath],
  );

  // Hydrate composer from `?id=` once clips+config are ready; ignore missing ids.
  useEffect(() => {
    if (urlClipHydratedRef.current || !config || !clipsReady) return;
    const fromUrl = clipIdFromUrl();
    urlClipHydratedRef.current = true;
    if (!fromUrl) return;

    async function openFromUrl(clipId: string) {
      let clip = clips.find((c) => c.id === clipId);
      if (!clip) {
        // May live in another project — search the full library once.
        const r = await fetch(`${API}/api/clips?all_projects=true`);
        if (r.ok) {
          const data = await r.json();
          const all = (data.clips ?? []) as Clip[];
          clip = all.find((c) => c.id === clipId);
          if (clip?.project_id && clip.project_id !== activeProjectId) {
            await fetch(`${API}/api/projects/${clip.project_id}/activate`, { method: "POST" });
            setActiveProjectId(clip.project_id);
            setProjects((prev) =>
              prev.map((p) => ({ ...p, active: p.id === clip!.project_id })),
            );
            setClips(all.filter((c) => (c.project_id || clip!.project_id) === clip!.project_id));
          }
        }
      }
      if (clip?.video_url) applyClipSelection(clip);
      else selectClipId(null);
    }

    void openFromUrl(fromUrl);
  }, [config, clips, clipsReady, applyClipSelection, selectClipId, activeProjectId]);

  async function loadProjectClips() {
    const next = await fetchClips();
    setClips(next);
    return next;
  }

  async function switchProject(projectId: string) {
    const r = await fetch(`${API}/api/projects/${projectId}/activate`, { method: "POST" });
    if (!r.ok) throw new Error("Failed to switch project");
    const data = await r.json();
    setProjects((data.projects ?? []) as Project[]);
    setActiveProjectId(data.active_project_id ?? projectId);
    selectClipId(null);
    setChainId(null);
    setBusy(false);
    setProgress(null);
    setError(null);
    await loadProjectClips();
  }

  async function createProject() {
    const r = await fetch(`${API}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!r.ok) throw new Error("Failed to create project");
    const data = await r.json();
    setProjects((data.projects ?? []) as Project[]);
    setActiveProjectId(data.active_project_id ?? data.project?.id ?? null);
    selectClipId(null);
    setChainId(null);
    setPrompt("");
    setClipMultiplier(1);
    setBusy(false);
    setProgress(null);
    setError(null);
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    setRefs([]);
    setSelectedCastIds([]);
    setClips([]);
  }

  async function renameProject(projectId: string, name: string) {
    const r = await fetch(`${API}/api/projects/${projectId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) throw new Error("Failed to rename project");
    const data = await r.json();
    setProjects((data.projects ?? []) as Project[]);
  }

  async function deleteProject(projectId: string) {
    const r = await fetch(`${API}/api/projects/${projectId}`, { method: "DELETE" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(typeof err.detail === "string" ? err.detail : "Failed to delete project");
    }
    const data = await r.json();
    setProjects((data.projects ?? []) as Project[]);
    setActiveProjectId(data.active_project_id ?? null);
    selectClipId(null);
    setChainId(null);
    await loadProjectClips();
  }

  function applyLoraHints(preset: LoraPreset) {
    if (preset.steps) setNumSteps(preset.steps);
    if (preset.layers) setLayers(preset.layers);
    if (preset.reuse) setReuse(preset.reuse);
    if (preset.steps && preset.steps <= 7) setReuse(1);
    setTokenReduction(false);
    setSsdStreaming(false);
  }

  async function applyTurboPreset(chosen: LoraPreset) {
    if (busy) return;
    if (chosen.cached === false) {
      await ensureLoraSpec(chosen.spec, chosen.label);
    }
    setTurboLoraId(chosen.id);
    setLoraPresetIds((prev) => {
      const withoutTurbo = prev.filter((id) => {
        const p = loraPresets.find((x) => x.id === id);
        return !p || !isTurboPreset(p);
      });
      return withoutTurbo.includes(chosen.id) ? withoutTurbo : [...withoutTurbo, chosen.id];
    });
    const fixedSteps = chosen.steps;
    const tierConfig = TURBO_CONFIG.TIERS[turboTier];
    setNumSteps(fixedSteps && fixedSteps <= 4 ? fixedSteps : tierConfig.steps);
    setLayers(chosen.layers ?? TURBO_CONFIG.LAYERS);
    setReuse(chosen.reuse ?? TURBO_CONFIG.REUSE);
    setTokenReduction(false);
    setSsdStreaming(false);
    setTurboEnabled(true);
    setLoraActivity(`${chosen.label} enabled`);
  }

  async function handleTurboLoraSelect(id: string) {
    if (busy) return;
    const chosen = loraPresets.find((p) => p.id === id);
    if (!chosen) return;
    setTurboLoading(true);
    try {
      await applyTurboPreset(chosen);
    } catch (e) {
      setError(String(e));
    } finally {
      setTurboLoading(false);
    }
  }

  async function handleTurboToggle(enabled: boolean) {
    if (!FEATURES.TURBO_MODE || busy) return;

    if (enabled) {
      setTurboLoading(true);
      try {
        let chosen: LoraPreset | undefined =
          (turboLoraId ? loraPresets.find((p) => p.id === turboLoraId) : undefined) ??
          [...loraPresets]
            .filter(isTurboPreset)
            .sort((a, b) => {
              const rank = (p: LoraPreset) => {
                const hay = `${p.label} ${p.spec}`.toLowerCase();
                const i = TURBO_CONFIG.PREFERRED.findIndex((n) => hay.includes(n));
                return i < 0 ? 99 : i;
              };
              return rank(a) - rank(b);
            })[0];

        if (!chosen) {
          await ensureLoraSpec(TURBO_CONFIG.LORA_SPEC, TURBO_CONFIG.LABEL);
          chosen = loraPresets.find((p) => p.spec === TURBO_CONFIG.LORA_SPEC);
          if (!chosen) {
            const r = await fetch(`${API}/api/loras/custom`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                spec: TURBO_CONFIG.LORA_SPEC,
                label: TURBO_CONFIG.LABEL,
                scale: TURBO_CONFIG.SCALE,
              }),
            });
            if (r.ok) {
              const data = (await r.json()) as { id: string; lora_presets?: LoraPreset[] };
              if (data.lora_presets) setLoraPresets(data.lora_presets);
              if (data.id) {
                chosen = (data.lora_presets ?? []).find((p) => p.id === data.id);
              }
            }
          }
        }

        if (!chosen) throw new Error("No turbo LoRA available — add one under models/loras or HF hub");
        await applyTurboPreset(chosen);
      } catch (e) {
        setError(String(e));
        setTurboEnabled(false);
      } finally {
        setTurboLoading(false);
      }
    } else {
      const turboIds = new Set(loraPresets.filter(isTurboPreset).map((p) => p.id));
      setLoraPresetIds((prev) => prev.filter((id) => !turboIds.has(id)));
      setTurboEnabled(false);
      setLoraActivity(null);

      const qid = coerceBaseQuality(quality);
      if (qid !== quality) setQuality(qid);
      const preset = config?.quality_presets.find((p) => p.id === qid);
      if (preset) {
        const fields = fieldsFromPreset(preset);
        setNumSteps(fields.steps);
        setLayers(fields.layers);
        setReuse(fields.reuse);
        setTokenReduction(fields.tokenReduction);
      }
    }
  }

  function handleTurboTierChange(tier: TurboTier) {
    if (busy) return;
    setTurboTier(tier);
    if (turboEnabled) {
      const active = loraPresets.find((p) => p.id === turboLoraId);
      if (active?.steps && active.steps <= 4) {
        setNumSteps(active.steps);
      } else {
        setNumSteps(TURBO_CONFIG.TIERS[tier].steps);
      }
    }
  }

  function addToSceneQueue() {
    const scene: SceneQueueItem = {
      id: generateId(),
      prompt,
      mode: compiled.modeHint,
      routing,
      selectedCastIds: [...selectedCastIds],
      quality,
      resolutionId,
      durationId,
      numSteps,
      layers,
      reuse,
      seed,
      loraPresetIds,
      turboEnabled,
      turboTier,
      refs: [...refs],
      imagePath,
      endImagePath,
      clipMultiplier,
      autocontinue: clipMultiplier > 1,
      autoconcat: clipMultiplier > 1,
      tokenReduction,
      ssdStreaming,
      status: "pending",
      createdAt: new Date().toISOString(),
    };
    setSceneQueue((prev) => [...prev, scene]);
  }

  function removeFromSceneQueue(id: string) {
    setSceneQueue((prev) => prev.filter((s) => s.id !== id));
  }

  function reorderSceneQueue(scenes: SceneQueueItem[]) {
    setSceneQueue(scenes);
  }

  function loadSceneToEditor(scene: SceneQueueItem) {
    setPrompt(scene.prompt);
    setMode(scene.mode);
    setRouting(scene.routing ?? (scene.mode === "ref2va" ? "ref2va" : scene.mode === "t2va" ? "auto" : "fl2va"));
    setSelectedCastIds(scene.selectedCastIds ?? []);
    setQuality(coerceBaseQuality(scene.quality));
    setResolutionId(scene.resolutionId);
    setDurationId(scene.durationId);
    setNumSteps(scene.numSteps);
    setLayers(scene.layers);
    setReuse(scene.reuse);
    setSeed(scene.seed);
    setLoraPresetIds(scene.loraPresetIds);
    setTurboEnabled(scene.turboEnabled);
    if (scene.turboTier) setTurboTier(scene.turboTier as TurboTier);
    setRefs(scene.refs);
    setImagePath(scene.imagePath ?? null);
    setEndImagePath(scene.endImagePath ?? null);
    setClipMultiplier(scene.clipMultiplier);
    // autocontinue/autoconcat are derived from clipMultiplier
    setTokenReduction(scene.tokenReduction);
    setSsdStreaming(scene.ssdStreaming);
  }

  function clearSceneQueue() {
    setSceneQueue([]);
  }

  async function runSceneQueue() {
    const pending = sceneQueue.filter((s) => s.status === "pending");
    if (pending.length === 0 || queueRunning || busy) return;

    setQueueRunning(true);
    for (const scene of pending) {
      loadSceneToEditor(scene);
      setSceneQueue((prev) =>
        prev.map((s) => (s.id === scene.id ? { ...s, status: "generating" as const } : s)),
      );
      try {
        await submitScene(scene);
        setSceneQueue((prev) =>
          prev.map((s) => (s.id === scene.id ? { ...s, status: "done" as const } : s)),
        );
      } catch (e) {
        setSceneQueue((prev) =>
          prev.map((s) =>
            s.id === scene.id ? { ...s, status: "failed" as const, error: String(e) } : s,
          ),
        );
      }
    }
    setQueueRunning(false);
  }

  function savePreset(name: string, description?: string) {
    const preset: GenerationPreset = {
      id: generateId(),
      name,
      description,
      mode,
      quality,
      resolutionId,
      durationId,
      numSteps,
      layers,
      reuse,
      loraIds: loraPresetIds,
      turboEnabled,
      tokenReduction,
      ssdStreaming,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setGenerationPresets((prev) => [...prev, preset]);
    // Persist to backend (survives restarts); localStorage as local fallback
    const updated = [...generationPresets, preset];
    localStorage.setItem("h3ws-presets", JSON.stringify(updated));
    void createPreset(preset);
  }

  function loadPreset(preset: GenerationPreset) {
    setMode(preset.mode);
    setRouting(preset.mode === "ref2va" ? "ref2va" : preset.mode === "t2va" ? "auto" : "fl2va");
    setQuality(coerceBaseQuality(preset.quality));
    setResolutionId(preset.resolutionId);
    setDurationId(preset.durationId);
    setNumSteps(preset.numSteps);
    setLayers(preset.layers);
    setReuse(preset.reuse);
    setLoraPresetIds(preset.loraIds);
    setTurboEnabled(preset.turboEnabled);
    setTokenReduction(preset.tokenReduction);
    setSsdStreaming(preset.ssdStreaming);
  }

  function deletePreset(id: string) {
    setGenerationPresets((prev) => prev.filter((p) => p.id !== id));
    const updated = generationPresets.filter((p) => p.id !== id);
    localStorage.setItem("h3ws-presets", JSON.stringify(updated));
    void fetch(`${API}/api/presets/${id}`, { method: "DELETE" }).catch(() => undefined);
  }

  function toggleClipLock(clipId: string) {
    setLockedClipIds((prev) => {
      const next = new Set(prev);
      if (next.has(clipId)) {
        next.delete(clipId);
      } else {
        next.add(clipId);
      }
      return next;
    });
  }

  async function createCastMember(name: string, description?: string) {
    try {
      const r = await fetch(`${API}/api/cast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => null);
        throw new Error((body && body.detail) || "Failed to create cast member");
      }
      const data = await r.json();
      setCastMembers((prev) => [...prev, data.cast as CastMember]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function deleteCastMember(id: string) {
    try {
      const r = await fetch(`${API}/api/cast/${id}`, { method: "DELETE" });
      if (!r.ok) {
        const body = await r.json().catch(() => null);
        throw new Error((body && body.detail) || "Failed to delete cast member");
      }
      setCastMembers((prev) => prev.filter((m) => m.id !== id));
      setSelectedCastIds((prev) => prev.filter((cid) => cid !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function toggleCast(id: string) {
    setSelectedCastIds((prev) =>
      prev.includes(id) ? prev.filter((cid) => cid !== id) : [...prev, id]
    );
  }

  async function persistCast(member: CastMember) {
    const r = await fetch(`${API}/api/cast/${member.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(member),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => null);
      throw new Error((body && body.detail) || "Failed to update cast member");
    }
    const data = await r.json();
    const next = data.cast as CastMember;
    setCastMembers((prev) => prev.map((m) => (m.id === next.id ? next : m)));
  }

  async function attachCastMedia(id: string, file: File, type: CastMediaType) {
    const member = castMembers.find((m) => m.id === id);
    if (!member) return;
    const up = await uploadFile(file, type);
    const media: CastMedia = {
      id: generateId(),
      type,
      path: up.path,
      label: file.name,
      thumbnailUrl: type === "image" ? URL.createObjectURL(file) : undefined,
      durationS: up.durationS,
    };
    await persistCast({ ...member, media: [...(member.media ?? []), media] });
  }

  async function removeCastMedia(id: string, mediaId: string) {
    const member = castMembers.find((m) => m.id === id);
    if (!member) return;
    await persistCast({ ...member, media: member.media.filter((m) => m.id !== mediaId) });
  }

  // Header "Models" button: always opens. Never blocks re-opening.
  function openModels() {
    setModelsOpen(true);
  }

  // Lock page scroll while the Models dialog is open (list scrolls inside).
  useEffect(() => {
    if (!modelsOpen) return;
    document.body.classList.add("modal-open");
    return () => document.body.classList.remove("modal-open");
  }, [modelsOpen]);

  // Backdrop click / close request: if a download is running, surface a
  // confirmation (the server task survives), otherwise close.
  function closeModels() {
    if (modelsDownloadActive) {
      setModelsCloseWarning(true);
      return;
    }
    setModelsOpen(false);
  }

  function handleModelsDownloadStateChange(active: boolean) {
    setModelsDownloadActive(active);
    if (!active) setModelsCloseWarning(false);
  }

  function confirmModelsClose() {
    setModelsCloseWarning(false);
    setModelsOpen(false);
  }

  // Load presets from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem("h3ws-presets");
      if (stored) {
        setGenerationPresets(JSON.parse(stored) as GenerationPreset[]);
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  async function ensureLoraSpec(spec: string, label: string) {
    setLoraBusy(true);
    setLoraActivity(`Downloading ${label}…`);
    try {
      const r = await fetch(`${API}/api/loras/ensure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof (err as { detail?: string }).detail === "string"
          ? (err as { detail: string }).detail
          : `LoRA download failed: ${label}`);
      }
      setLoraActivity(`${label} ready`);
    } finally {
      setLoraBusy(false);
    }
  }

  async function toggleLoraPreset(id: string, checked: boolean) {
    if (busy) return;
    const preset = loraPresets.find((p) => p.id === id);
    setLoraPresetIds((prev) => {
      const next = checked ? [...prev.filter((x) => x !== id), id] : prev.filter((x) => x !== id);
      return next;
    });
    if (checked && preset) {
      applyLoraHints(preset);
      try {
        await ensureLoraSpec(preset.spec, preset.label);
      } catch (e) {
        setError(String(e));
        setLoraActivity(String(e));
      }
    }
  }

  function setLoraScale(id: string, scale: number) {
    if (busy) return;
    const next = Math.min(2, Math.max(0, Number(scale)));
    if (!Number.isFinite(next)) return;
    setLoraPresets((prev) =>
      prev.map((p) => (p.id === id ? { ...p, scale: next } : p)),
    );
  }

  async function addCustomLora(spec: string, label: string, scale: number) {
    if (busy || !spec || addingCustomLora) return;
    setAddingCustomLora(true);
    setLoraActivity(`Downloading ${label || "LoRA"}…`);
    try {
      const r = await fetch(`${API}/api/loras/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          label: label || undefined,
          scale: scale || 1,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof (err as { detail?: string }).detail === "string"
          ? (err as { detail: string }).detail
          : "Could not add custom LoRA");
      }
      const data = (await r.json()) as { id: string; preset?: LoraPreset; lora_presets?: LoraPreset[] };
      if (data.lora_presets) setLoraPresets(data.lora_presets);
      if (data.id) {
        setLoraPresetIds((prev) => (prev.includes(data.id) ? prev : [...prev, data.id]));
        if (data.preset) applyLoraHints(data.preset);
      }
      const ready = `LoRA ready: ${data.preset?.label ?? (label || spec)}`;
      setLoraActivity(ready);
    } catch (e) {
      setLoraActivity(null);
      setError(String(e));
      throw e;
    } finally {
      setAddingCustomLora(false);
    }
  }

  async function removeLoraPreset(preset: LoraPreset) {
    if (busy || !preset.custom) return;
    if (!confirm(`Remove custom LoRA "${preset.label}"?`)) return;
    const r = await fetch(`${API}/api/loras/custom/${encodeURIComponent(preset.id)}`, {
      method: "DELETE",
    });
    if (!r.ok) return;
    const data = (await r.json()) as { lora_presets?: LoraPreset[] };
    if (data.lora_presets) setLoraPresets(data.lora_presets);
    setLoraPresetIds((prev) => prev.filter((id) => id !== preset.id));
  }

  async function startNewProject() {
    try {
      await createProject();
    } catch (err) {
      setError(String(err));
    }
  }

  async function deleteClip(clip: Clip) {
    await fetch(`${API}/api/clips/${clip.id}`, { method: "DELETE" });
    setClips((prev) => {
      revokeClipBlob(clip);
      return prev.filter((c) => c.id !== clip.id);
    });
    if (selectedClipId === clip.id) selectClipId(null);
    setRefs((prev) =>
      prev.filter((r) => r.path !== clip.path && r.path !== clip.filename && !r.path.endsWith(`/${clip.filename}`)),
    );
  }

  async function saveCurrentFrame() {
    const video = playerVideoRef.current;
    if (!video || !activeClip) return;
    setSavingFrame(true);
    try {
      const blob = await captureVideoFrame(video);
      const fd = new FormData();
      fd.append("file", blob, "frame.png");
      fd.append("source_clip_id", activeClip.id);
      fd.append("time_s", String(video.currentTime));
      fd.append("label", `Frame @ ${formatVideoTime(video.currentTime)}`);
      const r = await fetch(`${API}/api/frames`, { method: "POST", body: fd });
      if (!r.ok) throw new Error("Could not save frame");
      setFrameLibrary(await fetchFrames());
    } catch (e) {
      setError(String(e));
    } finally {
      setSavingFrame(false);
    }
  }

  function handleRefsChange(next: ReferenceItem[]) {
    setRefs(next);
    if (next.length === 0) return;
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    setClipMultiplier(1);
  }

  async function addFromLibrary(
    items: Array<{
      kind: RefKind;
      path: string;
      name: string;
      durationS?: number;
      previewUrl?: string;
    }>,
  ) {
    if (items.length === 0) return;
    handleRefsChange([
      ...refs,
      ...items.map((item) => ({
        id: generateId(),
        kind: item.kind,
        path: item.path,
        name: item.name,
        enabled: true as const,
        durationS: item.durationS,
        previewUrl: item.previewUrl,
        refSize: "max" as const,
        source: "library" as const,
      })),
    ]);
  }

  async function addVideoAudioRef(video: File, audio: File) {
    const v = await uploadFile(video, "video");
    const a = await uploadFile(audio, "audio");
    handleRefsChange([
      ...refs,
      {
        id: generateId(),
        kind: "video_audio",
        path: v.path,
        name: video.name,
        audioPath: a.path,
        audioName: audio.name,
        enabled: true,
        durationS: a.durationS ?? v.durationS,
        refSize: "max",
      },
    ]);
  }

  function applyFrameAsInput(frame: LibraryFrame, which: "start" | "end") {
    if (mode === "ref2va") {
      if (refs.some((r) => r.path === frame.path)) return;
      handleRefsChange([
        ...refs,
        { id: generateId(), kind: "image", path: frame.path, name: frame.label },
      ]);
      return;
    }
    setRefs([]);
    if (which === "end") {
      setEndImagePath(frame.path);
      setEndImageName(frame.label);
      if (mode === "t2va" || mode === "first_frame") setMode("fl2va");
      return;
    }
    setImagePath(frame.path);
    setImageName(frame.label);
    if (mode === "t2va") setMode("first_frame");
  }

  async function deleteFrame(frame: LibraryFrame) {
    const r = await fetch(`${API}/api/frames/${frame.id}`, { method: "DELETE" });
    if (!r.ok) return;
    const data = await r.json();
    setFrameLibrary((data.frames ?? []) as LibraryFrame[]);
    if (imagePath === frame.path) {
      setImagePath(null);
      setImageName(null);
    }
    if (endImagePath === frame.path) {
      setEndImagePath(null);
      setEndImageName(null);
    }
    setRefs((prev) => prev.filter((r) => r.path !== frame.path));
  }

  async function cancelRun(runId: string | null | undefined) {
    if (!runId) return;
    await fetch(`${API}/api/runs/${runId}/cancel`, { method: "POST" });
  }

  async function cancelActiveRun() {
    await cancelRun(activeRunId);
  }

  async function refreshProjectClips() {
    const next = await fetchClips();
    setClips((prev) => mergeClips(prev, next));
    return next;
  }

  function subscribeRun(runId: string, runChainId: string): Promise<void> {
    runEventSourceRef.current?.close();
    setActiveRunId(runId);
    setBusy(true);
    const es = new EventSource(`${API}/api/runs/${runId}/events`);
    runEventSourceRef.current = es;
    return new Promise((resolve, reject) => {
      let closed = false;
      const finishRun = (err?: string) => {
        if (closed) return;
        closed = true;
        es.close();
        if (runEventSourceRef.current === es) runEventSourceRef.current = null;

        void (async () => {
          let nextJob: Clip | undefined;
          try {
            const refreshed = await fetchClips();
            setClips((prev) => mergeClips(prev, refreshed));
            nextJob = refreshed
              .filter(
                (c) =>
                  (c.status === "queued" || c.status === "running")
                  && c.run_id
                  && c.run_id !== runId,
              )
              .sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""))[0];
          } catch {
            /* ignore refresh errors */
          }

          if (nextJob?.run_id) {
            setProgress({ phase: "queued", message: "Starting next in queue…" });
            setLivePreviewUrl(null);
            setLivePreviewMime("image/webp");
            if (watchLiveRef.current) {
              selectClipId(nextJob.id);
              setChainId(nextJob.chain_id);
            }
            // Resolve this run's waiter before attaching the next job so a
            // Cancelled reject cannot clear the new subscription.
            if (err && err !== "Cancelled") {
              reject(new Error(err));
            } else {
              if (!err) notifyGenerationReady();
              resolve();
            }
            void subscribeRun(nextJob.run_id, nextJob.chain_id);
            return;
          }

          setActiveRunId(null);
          setBusy(false);
          setProgress(null);
          setLivePreviewUrl(null);
          setLivePreviewMime("image/webp");
          setWatchLive(false);
          if (err && err !== "Cancelled") {
            reject(new Error(err));
          } else {
            if (!err) notifyGenerationReady();
            resolve();
          }
        })();
      };
      es.onmessage = (ev) => {
        const msg = JSON.parse(ev.data) as Record<string, unknown>;
        if (msg.type === "ping") return;
        if (msg.type === "progress" || msg.type === "generation_keepalive") {
          setProgress((prev) => applyProgressEvent(prev, msg));
          return;
        }
        if (msg.type === "preview") {
          const url = String(msg.url ?? "");
          if (url) {
            setLivePreviewUrl(url);
            setLivePreviewMime(String(msg.mime ?? "image/webp"));
          }
          const step = msg.step != null ? Number(msg.step) : null;
          const total = msg.total != null ? Number(msg.total) : null;
          if (step != null && total != null && total > 0) {
            setProgress((prev) => ({
              phase: prev?.phase ?? "generating",
              message: `Preview ${step}/${total}`,
              pct: Math.round((100 * step) / total),
              step,
              total,
            }));
          }
          return;
        }
        if (msg.type === "clip_started") {
          const clipId = String(msg.clip_id ?? "");
          if (clipId) {
            setClips((prev) =>
              prev.map((c) => (c.id === clipId ? { ...c, status: "running" } : c)),
            );
            if (watchLiveRef.current) selectClipId(clipId);
          }
          setLivePreviewUrl(null);
          setLivePreviewMime("image/webp");
          setProgress({
            phase: "generating",
            message: `Clip ${Number(msg.index ?? 0) + 1}/${Number(msg.total ?? 1)}`,
          });
        }
        if (msg.type === "clip_done" || msg.type === "merged") {
          setLivePreviewUrl(null);
          setLivePreviewMime("image/webp");
          const clipId = String(msg.clip_id ?? "");
          fetchClips(runChainId).then((chainClips) => {
            setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
            if (watchLiveRef.current) {
              selectClipId(pickPlaybackClip(chainClips, runChainId) ?? clipId ?? null);
            }
          });
          void fetchClips().then((all) => setClips((prev) => mergeClips(prev, all)));
        }
        if (msg.type === "run_cancelled") {
          setProgress({ phase: "cancelled", message: String(msg.message || "Cancelled") });
          finishRun("Cancelled");
        } else if (msg.type === "run_complete" || msg.type === "run_done") {
          finishRun();
        } else if (msg.type === "error" || msg.type === "clip_failed") {
          const message = String(msg.error || msg.message || "Failed");
          setError(message);
          notifyGenerationError(message);
          finishRun(message);
        }
      };
      es.onerror = () => {
        const message = "Lost connection to server while waiting for progress.";
        setError((prev) => prev ?? message);
        notifyGenerationError(message);
        finishRun(message);
      };
    });
  }

  function resumeLiveTracking(clip: Clip) {
    selectClipId(clip.id);
    setChainId(clip.chain_id);
    setError(null);
    // Queued cards are selectors only — keep watching the active run's preview.
    if (clip.status === "queued" && activeRunIdRef.current && clip.run_id !== activeRunIdRef.current) {
      return;
    }
    setWatchLive(true);
    if (clip.run_id && clip.run_id !== activeRunIdRef.current) {
      void subscribeRun(clip.run_id, clip.chain_id);
    }
  }

  async function editQueuedJob(clip: Clip) {
    applyClipSelection(clip);
    setWatchLive(false);
    if (clip.status === "queued" && clip.run_id) {
      await cancelRun(clip.run_id);
      await refreshProjectClips();
    }
  }

  async function killJob(clip: Clip) {
    if (!clip.run_id) return;
    await cancelRun(clip.run_id);
    if (clip.run_id !== activeRunIdRef.current) {
      await refreshProjectClips();
    }
  }

  async function postGenerate(
    body: Record<string, unknown>,
    opts?: { follow?: boolean },
  ): Promise<void> {
    const follow = opts?.follow ?? !activeRunIdRef.current;
    setError(null);
    setLoraModalOpen(false);
    if (follow) {
      setBusy(true);
      setWatchLive(true);
      setLivePreviewUrl(null);
      setLivePreviewMime("image/webp");
      setProgress({ phase: "starting", message: "Submitting…" });
      selectClipId(null);
    }
    const r = await fetch(`${API}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const detail = (err as { detail?: unknown }).detail;
      throw new Error(typeof detail === "string" ? detail : "Generate failed");
    }
    const data = await r.json();
    const projectClips = await fetchClips();
    setClips((prev) => mergeClips(prev, projectClips));
    if (follow) {
      setChainId(data.chain_id);
      setProgress(
        data.started_immediately
          ? { phase: "starting", message: "Starting…" }
          : { phase: "queued", message: "Queued — waiting for current job…" },
      );
      await subscribeRun(data.run_id, data.chain_id);
    }
  }

  function generateBody(opts: {
    compiledPrompt: string;
    fallbackPrompt: string;
    modeHint: string;
    refs: ReferenceItem[];
    quality: string;
    resolutionId: string;
    durationId: string;
    numSteps: number;
    layers: number;
    reuse: number;
    seed: string;
    clipMultiplier: number;
    tokenReduction: boolean;
    ssdStreaming: boolean;
    loraPresetIds: string[];
    imagePath: string | null;
    endImagePath: string | null;
    upscale?: boolean;
    /** User-facing composer state for library restore (kept separate from compiled prompt). */
    generation?: ReturnType<typeof buildGenerationSnapshot>;
  }): Record<string, unknown> {
    const res = config?.resolution_presets.find((p) => p.id === opts.resolutionId);
    const dur = config?.duration_presets.find((d) => d.id === opts.durationId);
    const ref2va = opts.modeHint === "ref2va";
    const multi = opts.clipMultiplier > 1 && !ref2va;
    const tokenLocked =
      opts.resolutionId === "256x256" ||
      opts.resolutionId === "512x512-aggressive" ||
      opts.quality === "aggressive" ||
      (opts.layers === 40 && opts.reuse === 3) ||
      opts.loraPresetIds.length > 0;
    const body: Record<string, unknown> = {
      prompt: opts.compiledPrompt.trim() || opts.fallbackPrompt.trim(),
      mode: opts.modeHint,
      quality: opts.quality,
      width: res?.width ?? 512,
      height: res?.height ?? 512,
      duration_seconds: dur?.seconds,
      num_frames: dur?.num_frames,
      clip_count: ref2va ? 1 : opts.clipMultiplier,
      num_steps: opts.numSteps,
      layers: opts.layers,
      reuse: opts.reuse,
      autocontinue: multi,
      autoconcat: multi,
      ssd_streaming: opts.loraPresetIds.length ? false : opts.ssdStreaming,
      token_reduction: tokenLocked ? false : opts.tokenReduction,
      loras: loraPresets
        .filter((p) => opts.loraPresetIds.includes(p.id))
        .map((p) => ({ id: p.id, spec: p.spec, scale: p.scale })),
      project_id: activeProjectId || undefined,
    };
    if (opts.generation) body.generation = opts.generation;
    if (res?.render_width) body.render_width = res.render_width;
    if (res?.render_height) body.render_height = res.render_height;
    if (opts.seed.trim() !== "") body.seed = Number(opts.seed);
    if (facesEnabled) {
      body.faces = {
        enabled: true,
        canvas: facesCanvas,
        denoise: facesDenoise,
        seed: facesSeed,
        abstain: false,
      };
    }
    if (ref2va) {
      body.refs = opts.refs.map((r) => ({
        kind: r.kind,
        path: r.path,
        name: r.name,
        audio_path: r.audioPath || undefined,
        ref_size: r.refSize ?? "max",
        ...(r.trim ? { trim: { start: r.trim.start, end: r.trim.end } } : {}),
        ...(r.crop ? { crop: r.crop } : {}),
      }));
    } else {
      if (IMAGE_MODES.has(opts.modeHint) && opts.imagePath) body.image_path = opts.imagePath;
      if (opts.modeHint === "last_frame" && (opts.imagePath || opts.endImagePath)) {
        body.end_image_path = opts.imagePath || opts.endImagePath;
      }
      if (opts.modeHint === "fl2va" && opts.endImagePath) body.end_image_path = opts.endImagePath;
    }
    return body;
  }

  async function submitScene(scene: SceneQueueItem): Promise<void> {
    const sceneCompiled = compilePrompt({
      prompt: scene.prompt,
      refs: scene.refs,
      castMembers,
      selectedCastIds: scene.selectedCastIds ?? [],
      routing: scene.routing ?? (scene.mode === "ref2va" ? "ref2va" : scene.mode === "t2va" ? "auto" : "fl2va"),
      imagePath: scene.imagePath,
      endImagePath: scene.endImagePath,
    });
    if (!scene.prompt.trim()) throw new Error("Empty prompt");
    if (sceneCompiled.modeHint === "ref2va" && !refsAreValid(sceneCompiled.refs).ok) {
      throw new Error("Scene needs a valid image or video reference");
    }
    await postGenerate(
      generateBody({
        compiledPrompt: sceneCompiled.compiledPrompt,
        fallbackPrompt: scene.prompt,
        modeHint: sceneCompiled.modeHint,
        refs: sceneCompiled.refs,
        quality: scene.quality,
        resolutionId: scene.resolutionId,
        durationId: scene.durationId,
        numSteps: scene.numSteps,
        layers: scene.layers,
        reuse: scene.reuse,
        seed: scene.seed,
        clipMultiplier: scene.clipMultiplier,
        tokenReduction: scene.tokenReduction,
        ssdStreaming: scene.ssdStreaming,
        loraPresetIds: scene.loraPresetIds,
        imagePath: scene.imagePath ?? null,
        endImagePath: scene.endImagePath ?? null,
        upscale,
        generation: buildGenerationSnapshot({ ...scene, upscale }),
      }),
      { follow: true },
    );
  }

  async function handleGenerate() {
    if (!canSubmit || !prompt.trim()) return;
    await ensureNotifyPermission();
    const follow = !activeRunIdRef.current;
    try {
      await postGenerate(
        generateBody({
          compiledPrompt: compiled.compiledPrompt,
          fallbackPrompt: prompt,
          modeHint: compiled.modeHint,
          refs: compiled.refs,
          quality,
          resolutionId,
          durationId,
          numSteps,
          layers,
          reuse,
          seed,
          clipMultiplier,
          tokenReduction,
          ssdStreaming,
          loraPresetIds,
          imagePath,
          endImagePath,
          upscale,
          generation: buildGenerationSnapshot({
            prompt,
            mode: compiled.modeHint,
            routing,
            selectedCastIds,
            quality,
            resolutionId,
            durationId,
            numSteps,
            layers,
            reuse,
            seed,
            loraPresetIds,
            turboEnabled,
            turboTier,
            refs,
            imagePath,
            endImagePath,
            clipMultiplier,
            tokenReduction,
            ssdStreaming,
            upscale,
          }),
        }),
        { follow },
      );
    } catch (e) {
      setError(String(e));
      if (follow) {
        setBusy(false);
        setProgress(null);
        setLivePreviewUrl(null);
        setLivePreviewMime("image/webp");
        setWatchLive(false);
      }
    }
  }

  async function runRefine() {
    if (!refineSettings.enabled || !refineSettings.base_url.trim()) return;
    const original = prompt;
    setRefineOriginal(original);
    setRefineDraft("");
    setRefineError(null);
    setRefineOpen(true);
    setRefineBusy(true);
    try {
      const r = await fetch(`${API}/api/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: original,
          mode: effectiveMode,
          refs: compiled.refs.map((ref) => ({
            kind: ref.kind,
            name: ref.name || ref.path,
          })),
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "Refine failed");
      }
      const data = (await r.json()) as { refined?: string };
      setRefineDraft(String(data.refined || ""));
    } catch (exc) {
      setRefineError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setRefineBusy(false);
    }
  }

  const serverOk = config?.server_connected;
  const canSubmit = useMemo(() => {
    if (!prompt.trim() || !serverOk) return false;
    if (isRef2va) return refsAreValid(compiled.refs).ok;
    if (needsFirst && !imagePath) return false;
    if (effectiveMode === "last_frame" && !imagePath && !endImagePath) return false;
    if (effectiveMode === "fl2va" && (!imagePath || !endImagePath)) return false;
    return true;
  }, [prompt, serverOk, isRef2va, compiled.refs, needsFirst, imagePath, endImagePath, effectiveMode]);

  const endpointLabel = useMemo(() => {
    if (typeof window === "undefined") return config?.server_url ?? "";
    return `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`;
  }, [config?.server_url]);

  return (
    <div className="app">
      {updateInfo?.update_available && !updateDismissed && (
        <div className="update-banner" role="status">
          <span>
            Update available: <strong>v{updateInfo.latest}</strong> (you have v{updateInfo.installed}).
          </span>
          <a href={updateInfo.html_url || updateInfo.releases_url} target="_blank" rel="noreferrer">
            Download
          </a>
          <button type="button" className="update-banner__dismiss" onClick={() => setUpdateDismissed(true)}>
            Dismiss
          </button>
        </div>
      )}
      <header className="header">
        <div className="brand">
          <span className="brand-mark">H3-WS</span>
          <span className="brand-sub">MiniMax-H3</span>
        </div>
        <div className="header-status">
          <button
            type="button"
            className="btn-secondary theme-toggle"
            title={
              themePref === "auto"
                ? `Theme: Auto (using ${resolveTheme("auto")})`
                : `Theme: ${themeLabel(themePref)}`
            }
            aria-label={
              themePref === "auto"
                ? `Theme Auto, currently ${resolveTheme("auto")}`
                : `Theme ${themeLabel(themePref)}`
            }
            onClick={() => {
              const next = cycleTheme(themePref);
              setThemePreference(next);
              setThemePref(next);
            }}
          >
            <ThemeIcon pref={themePref} />
          </button>
          {FEATURES.MODELS_PAGE && (
            <button type="button" className="btn-secondary" onClick={openModels}>
              Models
            </button>
          )}
          <button type="button" className="btn-secondary" onClick={() => setConsoleOpen(true)} title="Live console, self-test and bug-report export">
            Debug
          </button>
          <ProjectSwitcher
            projects={projects}
            activeProjectId={activeProjectId}
            disabled={busy}
            onSelect={(id) => void switchProject(id).catch((e) => setError(String(e)))}
            onCreate={() => void startNewProject()}
            onRename={(id, name) => void renameProject(id, name).catch((e) => setError(String(e)))}
            onDelete={(id) => void deleteProject(id).catch((e) => setError(String(e)))}
          />
          <span className={`status-dot ${serverOk ? "ok" : "off"}`} title={endpointLabel} />
          {serverOk ? "Server connected" : "Server offline"}
          {networkSettings?.listen_lan && networkSettings.lan_urls[0] ? (
            <span className="status-lan" title="LAN listen is on — open this URL from other devices">
              LAN {networkSettings.lan_urls[0].replace(/^https?:\/\//, "")}
            </span>
          ) : null}
        </div>
      </header>

      <div className="app-body">
        <div className="app-main">
          <section className="player-section">
            <div className="player-wrap">
              {watchLive && busy && livePreviewUrl ? (
                livePreviewMime.startsWith("video/") ? (
                  <video
                    key={livePreviewUrl}
                    className="player player--preview"
                    src={livePreviewUrl}
                    autoPlay
                    muted
                    loop
                    playsInline
                    preload="metadata"
                  />
                ) : (
                  /* Continuity-on-Mac default: animated WebP in <img> (GIF-like loop). */
                  <img
                    key={livePreviewUrl}
                    className="player player--preview"
                    src={livePreviewUrl}
                    alt="Denoise preview"
                  />
                )
              ) : activeClip?.video_url && !watchLive ? (
                <video
                  key={activeClip.id}
                  ref={playerVideoRef}
                  className="player"
                  src={activeClip.video_url}
                  poster={activeClip.thumb_url || undefined}
                  controls
                  loop
                  playsInline
                  preload="auto"
                />
              ) : (
                <div className="player placeholder">
                  {watchLive || busy
                    ? progress?.message ?? "Generating…"
                    : "Your video will appear here"}
                </div>
              )}
              {watchLive && busy && (
                <div className="progress-overlay">
                  <div className="progress-bar">
                    {progress?.pct != null ? (
                      <div className="progress-fill" style={{ width: `${Math.min(100, progress.pct)}%` }} />
                    ) : (
                      <div className="progress-pulse" />
                    )}
                  </div>
                  <div className="progress-overlay-row">
                    <span>{progress?.message ?? "Working…"}</span>
                    {activeRunId && progress?.phase !== "cancelled" && (
                      <button type="button" className="btn-cancel" onClick={() => void cancelActiveRun()}>
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
              )}
              {activeClip?.video_url && !watchLive && !busy && (
                <button
                  type="button"
                  className="player-capture-btn"
                  disabled={savingFrame}
                  onClick={() => void saveCurrentFrame()}
                  title="Save frame to library"
                >
                  {savingFrame ? (
                    <span className="player-capture-spinner" aria-hidden />
                  ) : (
                    <svg className="player-capture-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden>
                      <path d="M4 7h2l2-3h8l2 3h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z" />
                      <circle cx="12" cy="13" r="4" />
                    </svg>
                  )}
                </button>
              )}
            </div>
            {error && (
              <div className="error-banner" role="alert">
                <pre className="error-banner__text">{error}</pre>
                <button
                  type="button"
                  className="error-banner__copy"
                  onClick={() => void navigator.clipboard.writeText(error).catch(() => undefined)}
                  title="Copy error"
                >
                  Copy
                </button>
              </div>
            )}
            {showChainPicker && (
              <div className="player-context">
                <div className="player-context-body">
                  <label className="player-context-label">
                    Chain clip
                    <select
                      className="chain-picker-select"
                      value={selectedClipId ?? ""}
                      onChange={(e) => {
                        const c = chainParts.find((x) => x.id === e.target.value);
                        if (c) applyClipSelection(c);
                      }}
                    >
                      {chainParts.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.label}
                          {c.num_frames ? ` · ${formatDuration(c.num_frames, config?.defaults.fps ?? 24)}` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
            )}
          </section>

          {config && (
            <ComposerPanel
              busy={busy}
              pipelineActive={pipelineActive}
              canSubmit={canSubmit}
              prompt={prompt}
              onPromptChange={setPrompt}
              onGenerate={() => void handleGenerate()}
              compiled={compiled}
              refs={routing === "fl2va" ? [] : refs}
              onRefsChange={handleRefsChange}
              onUpload={async (file, kind) => uploadFile(file, kind)}
              onAddFromLibrary={(items) => void addFromLibrary(items)}
              onAddVideoAudio={(video, audio) => void addVideoAudioRef(video, audio)}
              onClearComposer={() => {
                setPrompt("");
                setRefs([]);
                setImagePath(null);
                setImageName(null);
                setEndImagePath(null);
                setEndImageName(null);
                setSelectedCastIds([]);
              }}
              onOpenLora={() => setLoraModalOpen(true)}
              onRefine={() => void runRefine()}
              refineEnabled={
                refineSettings.enabled && Boolean(refineSettings.base_url.trim())
              }
              onOpenFeatures={() => setFeaturesOpen(true)}
              loraCount={loraPresetIds.length}
              loraActivity={loraActivity}
              castMembers={castMembers}
              selectedCastIds={selectedCastIds}
              onToggleCast={toggleCast}
              onCreateCast={createCastMember}
              onDeleteCast={deleteCastMember}
              onAttachCastMedia={attachCastMedia}
              onRemoveCastMedia={removeCastMedia}
              presets={generationPresets}
              onSavePreset={savePreset}
              onLoadPreset={loadPreset}
              onDeletePreset={deletePreset}
              activeStyleId={activeStyleId}
              onApplyStyle={(style) => {
                setPrompt((prev) => leadWithStyle(prev, style.text));
                setActiveStyleId(style.id);
              }}
              routing={routing}
              onRoutingChange={setRouting}
              durationId={durationId}
              onDurationId={setDurationId}
              resolutionId={resolutionId}
              onResolutionId={(id) => {
                setResolutionId(id);
                if (id === "256x256" || id === "512x512-aggressive") setTokenReduction(false);
              }}
              config={config}
              imageName={imageName}
              endImageName={endImageName}
              onPickStartImage={(item) => {
                setRefs([]);
                setImagePath(item.path);
                setImageName(item.name);
                setRouting("fl2va");
                if (!endImagePath) setMode("first_frame");
                else setMode("fl2va");
              }}
              onPickEndImage={(item) => {
                setRefs([]);
                setEndImagePath(item.path);
                setEndImageName(item.name);
                setRouting("fl2va");
                if (!imagePath) setMode("last_frame");
                else setMode("fl2va");
              }}
              onClearStart={() => { setImagePath(null); setImageName(null); }}
              onClearEnd={() => { setEndImagePath(null); setEndImageName(null); }}
              frames={framesNewest}
              clips={libraryClips}
              onAddFrameRef={(frame) => applyFrameAsInput(frame, "start")}
              onAddClipRef={(clip) => {
                const path = clip.path || clip.filename;
                if (!path) return;
                handleRefsChange([
                  ...refs,
                  { id: generateId(), kind: "silent_video", path, name: clip.label || clip.filename, enabled: true, source: "library" },
                ]);
              }}
              onAddShot={addToSceneQueue}
              seed={seed}
              onSeed={setSeed}
              numSteps={numSteps}
              onNumSteps={setNumSteps}
              layers={layers}
              onLayers={setLayers}
              reuse={reuse}
              onReuse={setReuse}
              quality={quality}
              qualityOptions={qualityOptions}
              onQuality={(id) => {
                const qid = coerceBaseQuality(id);
                const preset = config.quality_presets.find((p) => p.id === qid);
                const fields = fieldsFromPreset(preset);
                setQuality(qid);
                setNumSteps(fields.steps);
                setLayers(fields.layers);
                setReuse(fields.reuse);
                setTokenReduction(fields.tokenReduction);
                if (turboEnabled) setTurboEnabled(false);
              }}
              turboEnabled={turboEnabled}
              turboTier={turboTier}
              turboLoading={turboLoading}
              turboOptions={loraPresets}
              turboLoraId={turboLoraId}
              onTurboLoraId={(id) => void handleTurboLoraSelect(id)}
              onTurboScale={setLoraScale}
              loraBusy={loraBusy}
              onTurbo={(enabled) => void handleTurboToggle(enabled)}
              onTurboTier={handleTurboTierChange}
              tokenReduction={tokenReduction}
              tokenReductionLocked={tokenReductionLocked}
              onTokenReduction={setTokenReduction}
              ssdStreaming={ssdStreaming}
              ssdLocked={loraPresetIds.length > 0}
              onSsdStreaming={setSsdStreaming}
              clipMultiplier={clipMultiplier}
              onClipMultiplier={setClipMultiplier}
              facesConfig={config.faces}
              facesEnabled={facesEnabled}
              facesCanvas={facesCanvas}
              facesDenoise={facesDenoise}
              facesSeed={facesSeed}
              onFacesEnabled={(v) => {
                setFacesEnabled(v);
                // Prefetch SAM when Faces is turned on (survives if EventSource closes).
                if (v && config?.faces && !config.faces.sam3_ready) {
                  const es = new EventSource(`${API}/api/models/download/stream?component=sam3`);
                  const stop = () => {
                    es.close();
                    void fetchConfig().then((cfg) => {
                      setConfig(cfg);
                    });
                  };
                  es.addEventListener("complete", stop);
                  es.addEventListener("error", (ev) => {
                    if (ev instanceof MessageEvent) stop();
                  });
                  // Detach UI listener; server download keeps running.
                  window.setTimeout(() => {
                    try {
                      es.close();
                    } catch {
                      /* ignore */
                    }
                  }, 1500);
                }
              }}
              onFacesCanvas={setFacesCanvas}
              onFacesDenoise={setFacesDenoise}
              onFacesSeed={setFacesSeed}
              sceneQueue={sceneQueue}
              onRemoveScene={removeFromSceneQueue}
              onReorderScenes={reorderSceneQueue}
              onEditScene={loadSceneToEditor}
              onRunQueue={() => void runSceneQueue()}
              onClearQueue={clearSceneQueue}
              queueRunning={queueRunning}
            />
          )}
        </div>

        <aside className="library">
          {/* Timeline view for current chain */}
          {FEATURES.TIMELINE_VIEW && chainId && chainParts.length > 1 && (
            <div style={{ marginBottom: "12px" }}>
              <TimelineStrip
                clips={chainParts}
                selectedClipId={selectedClipId}
                onSelectClip={applyClipSelection}
                lockedClipIds={lockedClipIds}
                onLockToggle={FEATURES.LOCKED_TAKES ? toggleClipLock : undefined}
                disabled={busy}
              />
            </div>
          )}

          {inProgressClips.length > 0 && (
            <div className="library-queue">
              <div className="library-queue__header">
                <span className="library-title">In progress</span>
                <span className="library-count">{inProgressClips.length}</span>
              </div>
              <div className="library-grid">
                {inProgressClips.map((clip) => {
                  const isRunning = clip.status === "running";
                  const isFollowed = Boolean(clip.run_id && clip.run_id === activeRunId);
                  const showPreview = isFollowed && watchLive && isRunning && livePreviewUrl;
                  const active =
                    selectedClipId === clip.id
                    || (watchLive && isFollowed);
                  return (
                    <div
                      key={clip.id}
                      className={`library-card-wrap library-card-wrap--job${active ? " active" : ""}`}
                    >
                      <button
                        type="button"
                        className="library-card"
                        onClick={() => resumeLiveTracking(clip)}
                        title={isRunning ? "Watch live preview" : "Queued — waiting"}
                      >
                        {showPreview ? (
                          livePreviewMime.startsWith("video/") ? (
                            <video
                              className="library-thumb"
                              src={livePreviewUrl}
                              muted
                              playsInline
                              autoPlay
                              loop
                            />
                          ) : (
                            <img className="library-thumb" src={livePreviewUrl!} alt="" />
                          )
                        ) : (
                          <div className={`library-thumb library-thumb--job${isRunning ? " is-running" : ""}`}>
                            {isRunning ? (isFollowed && progress?.message ? progress.message : "Generating") : "Queued"}
                          </div>
                        )}
                        <span className={`library-label ${isRunning ? "current" : "queue"}`}>
                          {isRunning ? "NOW" : "QUEUE"}
                        </span>
                        <span className="library-prompt">{clipDisplayPrompt(clip.prompt)}</span>
                      </button>
                      <div className="library-job-actions">
                        <button
                          type="button"
                          className="library-job-btn"
                          title={clip.status === "queued" ? "Edit (pull into composer & remove from queue)" : "Edit (load into composer)"}
                          onClick={(e) => {
                            e.stopPropagation();
                            void editQueuedJob(clip);
                          }}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="library-job-btn library-job-btn--kill"
                          title="Cancel"
                          onClick={(e) => {
                            e.stopPropagation();
                            void killJob(clip);
                          }}
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <button
            type="button"
            className="library-header library-header--toggle"
            onClick={() => setLibraryOpen((v) => !v)}
            aria-expanded={libraryOpen}
          >
            <span className="library-title">Library</span>
            <span className="library-header__meta">
              <span className="library-count">{libraryClips.length}</span>
              <span className={`library-chevron${libraryOpen ? " is-open" : ""}`} aria-hidden>
                ▾
              </span>
            </span>
          </button>
          {libraryOpen && (
          <div className="library-grid">
            {libraryClips.map((clip) => (
              <div
                key={clip.id}
                className={`library-card-wrap ${activeClip?.id === clip.id && !watchLive ? "active" : ""}`}
              >
                <button
                  type="button"
                  className="library-card"
                  onClick={() => applyClipSelection(clip)}
                  title={clip.prompt}
                >
                  {clip.thumb_url ? (
                    <img
                      className="library-thumb"
                      src={clip.thumb_url}
                      alt=""
                      loading="lazy"
                    />
                  ) : clip.video_url ? (
                    <video
                      className="library-thumb"
                      src={`${clip.video_url}#t=0.1`}
                      muted
                      playsInline
                      preload="metadata"
                    />
                  ) : null}
                  <span className={`library-label ${clip.label.toLowerCase()}`}>{clip.label}</span>
                  <span className="library-prompt">{clipDisplayPrompt(clip.prompt)}</span>
                </button>
                <div className="library-job-actions library-job-actions--done">
                  <button
                    type="button"
                    className="library-job-btn"
                    title="Inspect faces (detect + overlay boxes)"
                    disabled={busy}
                    onClick={(e) => {
                      e.stopPropagation();
                      void (async () => {
                        try {
                          const r = await fetch(`${API}/api/clips/${clip.id}/faces/inspect`);
                          const data = await r.json();
                          if (!r.ok) throw new Error(data.detail || "Inspect failed");
                          const msg = data.found_count
                            ? `Found faces on ${data.found_count} sample frame(s) · backend ${data.backend}${data.abstain ? " · would abstain (close-up)" : ""}`
                            : `No faces detected (${data.backend})`;
                          window.alert(msg);
                          window.open(
                            `${API}/api/clips/${clip.id}/faces/overlay.png?frame=0`,
                            "_blank",
                            "noopener,noreferrer",
                          );
                        } catch (err) {
                          window.alert(err instanceof Error ? err.message : String(err));
                        }
                      })();
                    }}
                  >
                    Inspect
                  </button>
                  <button
                    type="button"
                    className="library-job-btn"
                    title={
                      config?.faces?.sam3_ready
                        ? "Run Faces repair (SAM detect + Ref2VA refs)"
                        : "Run Faces — SAM 3.1 downloads automatically if needed"
                    }
                    disabled={busy}
                    onClick={(e) => {
                      e.stopPropagation();
                      void (async () => {
                        if (!window.confirm("Run Faces pass on this clip? This uses the GPU.")) return;
                        try {
                          const r = await fetch(`${API}/api/clips/${clip.id}/faces`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                              canvas: facesCanvas,
                              denoise: facesDenoise,
                              seed: facesSeed,
                              confirm_boxes: true,
                            }),
                          });
                          const data = await r.json();
                          if (!r.ok) throw new Error(data.detail || "Faces failed");
                          if (data.needs_confirm) {
                            const ok = window.confirm(
                              `Detected ${data.overlays?.length ?? 0} face box(es)` +
                                (data.abstain ? " (close-up — may abstain)" : "") +
                                ". Continue with repair?",
                            );
                            if (!ok) return;
                            const boxes = (data.overlays || []).reduce(
                              (acc: Array<{ x: number; y: number; w: number; h: number }>, o: { frame: number; box: { x: number; y: number; w: number; h: number } }) => {
                                acc[o.frame] = o.box;
                                return acc;
                              },
                              [] as Array<{ x: number; y: number; w: number; h: number }>,
                            );
                            // Fill missing frames with zeros for confirm path length check — server re-detects if incomplete.
                            const r2 = await fetch(`${API}/api/clips/${clip.id}/faces`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({
                                canvas: facesCanvas,
                                denoise: facesDenoise,
                                seed: facesSeed,
                                confirm_boxes: false,
                              }),
                            });
                            const data2 = await r2.json();
                            if (!r2.ok) throw new Error(data2.detail || "Faces failed");
                            await refreshProjectClips();
                            window.alert("Faces pass complete.");
                            void boxes;
                            return;
                          }
                          await refreshProjectClips();
                          window.alert("Faces pass complete.");
                        } catch (err) {
                          window.alert(err instanceof Error ? err.message : String(err));
                        }
                      })();
                    }}
                  >
                    Faces
                  </button>
                </div>
                <button
                  type="button"
                  className="library-delete"
                  title="Delete"
                  disabled={busy && watchLive}
                  onClick={(e) => {
                    e.stopPropagation();
                    void deleteClip(clip);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          )}

          <div className="library-section">
            <button
              type="button"
              className="library-header library-header--toggle"
              onClick={() => setFramesOpen((v) => !v)}
              aria-expanded={framesOpen}
            >
              <span className="library-title">Frames</span>
              <span className="library-header__meta">
                <span className="library-count">{framesNewest.length}</span>
                <span className={`library-chevron${framesOpen ? " is-open" : ""}`} aria-hidden>
                  ▾
                </span>
              </span>
            </button>
            {framesOpen && (framesNewest.length === 0 ? (
              <p className="library-empty-hint">
                Pause a video and tap the camera icon to capture stills. Use them as a
                first or last frame, or as a reference image when that mode is selected.
              </p>
            ) : (
              <div className="frame-library-grid">
                {framesNewest.map((frame) => (
                  <div
                    key={frame.id}
                    className={`frame-card-wrap ${
                      imagePath === frame.path || endImagePath === frame.path ? "active" : ""
                    }`}
                  >
                    <button
                      type="button"
                      className="frame-card"
                      title={`Use as start image: ${frame.label}`}
                      onClick={() => applyFrameAsInput(frame, "start")}
                    >
                      <img className="frame-thumb" src={frame.image_url} alt={frame.label} loading="lazy" />
                      <span className="frame-label">{frame.label}</span>
                    </button>
                    {mode === "fl2va" && (
                      <button
                        type="button"
                        className="frame-use-end"
                        title="Use as end image"
                        disabled={busy}
                        onClick={(e) => {
                          e.stopPropagation();
                          applyFrameAsInput(frame, "end");
                        }}
                      >
                        End
                      </button>
                    )}
                    <button
                      type="button"
                      className="library-delete"
                      title="Delete frame"
                      disabled={busy}
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteFrame(frame);
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </aside>
      </div>

      {/* Models panel */}
      {FEATURES.MODELS_PAGE && modelsOpen && (
        <div className="modal-backdrop" onClick={closeModels}>
          <div
            className="modal modal--fullscreen models-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <ModelsManager
              api={API}
              onClose={closeModels}
              onDownloadStateChange={handleModelsDownloadStateChange}
              onPathApplied={(status) => {
                if (status.lora_presets?.length) setLoraPresets(status.lora_presets);
                void fetchConfig().then((cfg) => {
                  setConfig(cfg);
                  if (cfg.lora_presets) setLoraPresets(cfg.lora_presets);
                });
              }}
            />
          </div>
        </div>
      )}

      {/* Models close guard — download keeps running in the background */}
      {FEATURES.MODELS_PAGE && modelsCloseWarning && (
        <div className="models-close-toast">
          <span>Download in progress — it will continue in the background.</span>
          <button type="button" className="btn-ghost" onClick={confirmModelsClose}>
            Close anyway
          </button>
          <button type="button" className="btn-ghost" onClick={() => setModelsCloseWarning(false)}>
            Keep open
          </button>
        </div>
      )}

      {/* LoRA Modal */}
      {FEATURES.LORA_MODAL && (
        <LoraModal
          open={loraModalOpen}
          onClose={() => setLoraModalOpen(false)}
          presets={loraPresets}
          selectedIds={loraPresetIds}
          onToggle={(id, checked) => void toggleLoraPreset(id, checked)}
          onScaleChange={setLoraScale}
          onRemove={(preset) => void removeLoraPreset(preset)}
          onAddCustom={addCustomLora}
          addingCustom={addingCustomLora}
          activity={loraActivity}
          disabled={busy || loraBusy}
        />
      )}

      <FeaturesPopup
        open={featuresOpen}
        onClose={() => setFeaturesOpen(false)}
        api={API}
        initial={refineSettings}
        onSaved={setRefineSettings}
        networkInitial={networkSettings}
        onNetworkSaved={setNetworkSettings}
      />

      <DebugModal open={consoleOpen} api={API} onClose={() => setConsoleOpen(false)} />

      <RefinePanel
        open={refineOpen}
        draft={refineDraft}
        original={refineOriginal}
        busy={refineBusy}
        error={refineError}
        onDraftChange={setRefineDraft}
        onUse={() => {
          if (refineDraft.trim()) setPrompt(refineDraft.trim());
          setRefineOpen(false);
        }}
        onRevert={() => {
          setRefineDraft(refineOriginal);
          setPrompt(refineOriginal);
          setRefineOpen(false);
        }}
        onKeep={() => setRefineOpen(false)}
        onClose={() => setRefineOpen(false)}
      />
    </div>
  );
}
