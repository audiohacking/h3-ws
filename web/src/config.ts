/**
 * Feature flags for gradual rollout of new UI components.
 * Toggle these to enable/disable specific features during development.
 */
export const FEATURES = {
  /** Phase 1: Pill-style grouped controls */
  PILLS_UI: true,

  /** Phase 1: One-click turbo mode with distillation LoRA */
  TURBO_MODE: true,

  /** Phase 2: @ reference chips in prompt field */
  REFERENCE_CHIPS: true,

  /** Phase 2: Full-screen LoRA browser modal */
  LORA_MODAL: true,

  /** Phase 3: Timeline strip for multi-shot composition */
  TIMELINE_VIEW: true,

  /** Phase 4: Scene queue for multi-scene scheduling */
  SCENE_QUEUE: true,

  /** Phase 5: Per-reference scope controls (face/object/scene/style) */
  SCOPE_CONTROLS: false,

  /** Phase 5: Named character cast system */
  CAST_SYSTEM: true,

  /** A dedicated Models page for inspecting/downloading model components */
  MODELS_PAGE: true,

  /** Continuity-style "what the model reads" reference elaboration panel */
  WHAT_MODEL_READS: true,

  /** Phase 5: Save/load generation presets */
  PRESETS: true,

  /** Phase 5: Lock clips to exclude from batch regeneration */
  LOCKED_TAKES: true,
} as const;

/**
 * Turbo mode configuration - distillation LoRA for fast generation
 */
export const TURBO_CONFIG = {
  /**
   * Fallback HF recipe when no on-disk turbo LoRA is found.
   * Prefer selecting from catalog entries marked turbo (TaoMate, Tutu, LightX2V, …).
   */
  LORA_SPEC:
    "https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/" +
    "resolve/main/comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors",

  /** Display label for the turbo LoRA */
  LABEL: "Tutu 8-NFE Turbo",

  /** Preferred on-disk filename needles (first match wins). Continuity default: TaoMate. */
  PREFERRED: ["taomate", "tutu", "minimax_h3", "minimax-h3", "h3_turbo", "h3-turbo", "fasth3"] as const,

  /** Recommended layers for turbo mode */
  LAYERS: 50,

  /** Reuse must be 1 for low-step generation */
  REUSE: 1,

  /** LoRA strength for turbo mode */
  SCALE: 0.8,

  /** Quality tiers with step counts */
  TIERS: {
    draft: { steps: 4, label: "Draft", description: "Fastest, softer details" },
    medium: { steps: 6, label: "Medium", description: "Balanced quality/speed" },
    good: { steps: 8, label: "Good", description: "Near-native quality" },
  } as const,

  /** Default tier */
  DEFAULT_TIER: "good" as const,

  /** Guidance text */
  GUIDANCE:
    "Uses a turbo/distill LoRA from your models/loras folder (TaoMate, Tutu, LightX2V, …). " +
    "Pick another in the chevron menu before you generate — settings lock while a run is in progress.",
} as const;

export type TurboTier = keyof typeof TURBO_CONFIG.TIERS;

/**
 * Timeline configuration
 */
export const TIMELINE_CONFIG = {
  /** Minimum clip width in pixels */
  MIN_CLIP_WIDTH: 100,

  /** Maximum clips in a single chain */
  MAX_CHAIN_LENGTH: 10,

  /** Seam transition duration options (seconds) */
  SEAM_DURATIONS: [0, 0.5, 1, 2] as const,
} as const;

/**
 * Reference limits (matching backend validation)
 */
export const REF_LIMITS = {
  MAX_IMAGES: 9,
  MAX_VIDEOS: 3,
  MAX_AUDIOS: 3,
  MAX_TOTAL_FILES: 12,
} as const;
