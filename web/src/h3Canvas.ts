/**
 * Continuity H3 canvas math — port of ``creator/canvas.py`` /
 * ``web/creator/canvas.js`` + ``families/h3/declare.RULES``.
 *
 * Do not invent values: these match Continuity's H3 family.
 */

export const CANVAS_MULTIPLE = 32;
/** Continuity ``native_short_edge``. */
export const NATIVE_SHORT_EDGE = 768;
/** Continuity ``min_short_edge``. */
export const MIN_SHORT_EDGE = 384;
/** Continuity ``max_short_edge``. */
export const MAX_SHORT_EDGE = 2048;
/** Continuity ``native_max_pixels`` = 768 × 1344. */
export const NATIVE_MAX_PIXELS = 768 * 1344;
export const MIN_RATIO = 9 / 21;
export const MAX_RATIO = 21 / 9;

/** Continuity H3 ``aspects`` — label → ratio, popover order. */
export const H3_ASPECTS: ReadonlyArray<readonly [string, number]> = [
  ["16:9", 16 / 9],
  ["21:9", 21 / 9],
  ["3:2", 3 / 2],
  ["4:3", 4 / 3],
  ["5:4", 5 / 4],
  ["1:1", 1.0],
  ["4:5", 4 / 5],
  ["3:4", 3 / 4],
  ["2:3", 2 / 3],
  ["9:16", 9 / 16],
  ["9:21", 9 / 21],
];

/** Continuity ``UPSCALE_MODES``. */
export type UpscaleMode = "two_pass" | "direct" | "redetail";
export const UPSCALE_MODES: UpscaleMode[] = ["two_pass", "direct", "redetail"];

export const DEFAULT_REFINE_DENOISE = 0.5;
export const MIN_REFINE_DENOISE = 0.1;
export const MAX_REFINE_DENOISE = 0.9;
export type RefineUpscaler = "bicubic" | "trained";
export const DEFAULT_REFINE_UPSCALER: RefineUpscaler = "bicubic";
export const TRAINED_REFINE = { refineDenoise: 0.3, refineSteps: 3 } as const;
export const BICUBIC_REFINE = {
  refineDenoise: DEFAULT_REFINE_DENOISE,
  refineSteps: 0,
} as const;
export const DEFAULT_REFINE_STEPS = 0;
export const MAX_REFINE_STEPS = 200;

/** Continuity empty-piece canvas defaults. */
export const DEFAULT_ASPECT = "16:9";
export const DEFAULT_SHORT_EDGE = NATIVE_SHORT_EDGE;
export const DEFAULT_SAMPLE_EDGE = NATIVE_SHORT_EDGE;
export const DEFAULT_UPSCALE: UpscaleMode = "two_pass";

export function ratioOfAspect(label: string): number {
  return H3_ASPECTS.find(([a]) => a === label)?.[1] ?? 16 / 9;
}

export function clampRatio(ratio: number): [number, boolean] {
  if (ratio < MIN_RATIO) return [MIN_RATIO, true];
  if (ratio > MAX_RATIO) return [MAX_RATIO, true];
  return [ratio, false];
}

function snap(value: number): number {
  return Math.max(CANVAS_MULTIPLE, Math.round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE);
}

export function resolveCanvas(ratio: number, shortEdge: number): [number, number] {
  const [clamped] = clampRatio(ratio);
  const edge = Math.max(
    MIN_SHORT_EDGE,
    Math.min(MAX_SHORT_EDGE, Math.round(shortEdge)),
  );
  const maxPixels = NATIVE_MAX_PIXELS * (edge / NATIVE_SHORT_EDGE) ** 2;

  let width: number;
  let height: number;
  if (clamped >= 1) {
    width = edge * clamped;
    height = edge;
  } else {
    width = edge;
    height = edge / clamped;
  }

  if (width * height > maxPixels) {
    const scale = Math.sqrt(maxPixels / (width * height));
    width *= scale;
    height *= scale;
  }

  width = snap(width);
  height = snap(height);

  while (width * height > maxPixels && Math.max(width, height) > CANVAS_MULTIPLE) {
    if (width >= height) width -= CANVAS_MULTIPLE;
    else height -= CANVAS_MULTIPLE;
  }
  return [width, height];
}

export function snapShortEdge(n: number): number {
  const clamped = Math.min(MAX_SHORT_EDGE, Math.max(MIN_SHORT_EDGE, n));
  return Math.min(
    MAX_SHORT_EDGE,
    Math.max(
      MIN_SHORT_EDGE,
      Math.round((clamped - MIN_SHORT_EDGE) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
        + MIN_SHORT_EDGE,
    ),
  );
}

/** Continuity face-crop canvas — follows shot short edge, clamped 384–768. */
export const MIN_FACE_CANVAS = MIN_SHORT_EDGE;
export const MAX_FACE_CANVAS = NATIVE_SHORT_EDGE;

export function faceCanvasForVideo(shortEdge: number): number {
  const clamped = Math.min(
    MAX_FACE_CANVAS,
    Math.max(MIN_FACE_CANVAS, Math.round(shortEdge)),
  );
  return Math.max(
    MIN_FACE_CANVAS,
    Math.round(clamped / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
  );
}

/** Continuity ``clampSampleEdge`` — sample edge never above native. */
export function clampSampleEdge(value: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return NATIVE_SHORT_EDGE;
  const snapped = Math.round(n / CANVAS_MULTIPLE) * CANVAS_MULTIPLE;
  return Math.min(NATIVE_SHORT_EDGE, Math.max(MIN_SHORT_EDGE, snapped));
}

/** Continuity ``sampleEdge`` — stored sample, capped by the slider. */
export function sampleEdgeOf(sampleEdge: number, shortEdge: number): number {
  return Math.min(clampSampleEdge(sampleEdge), shortEdge);
}

/** Continuity ``twoPass``. */
export function isTwoPass(
  sampleEdge: number,
  shortEdge: number,
  upscale: UpscaleMode,
): boolean {
  return sampleEdgeOf(sampleEdge, shortEdge) < shortEdge
    && upscale !== "direct"
    && upscale !== "redetail";
}

/** Continuity ``redetailTarget`` — ×2 of the sampled canvas. */
export function redetailTarget(
  ratio: number,
  sampleEdge: number,
  shortEdge: number,
): { width: number; height: number; scale: number } {
  const edge = sampleEdgeOf(sampleEdge, shortEdge);
  const [w, h] = resolveCanvas(ratio, edge);
  return { width: w * 2, height: h * 2, scale: 2 };
}

export function describeAspect(ratio: number): string {
  let best = H3_ASPECTS[0]!;
  for (const preset of H3_ASPECTS) {
    if (Math.abs(preset[1] - ratio) < Math.abs(best[1] - ratio)) best = preset;
  }
  return best[0];
}

export type AspectShape = {
  wide: readonly [string, number] | null;
  tall: readonly [string, number] | null;
  key: number;
};

export function aspectShapes(
  presets: ReadonlyArray<readonly [string, number]> = H3_ASPECTS,
): AspectShape[] {
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-6;
  const shapes: AspectShape[] = [];
  const placed = new Set<string>();
  for (const entry of presets) {
    const [label, ratio] = entry;
    if (placed.has(label)) continue;
    placed.add(label);
    if (near(ratio, 1)) {
      shapes.push({ wide: entry, tall: entry, key: 1 });
      continue;
    }
    const twin = presets.find(([other, r]) => other !== label && near(r * ratio, 1));
    if (twin) placed.add(twin[0]);
    const [wide, tall] = ratio > 1 ? [entry, twin ?? null] : [twin ?? null, entry];
    shapes.push({ wide, tall, key: Math.max(ratio, 1 / ratio) });
  }
  return shapes.sort((a, b) => b.key - a.key);
}

export function canvasId(width: number, height: number): string {
  return `${width}x${height}`;
}

export function shortEdgeOf(width: number, height: number): number {
  return Math.min(width, height);
}

/** Continuity ``resolutionPillText`` chip sub-line. */
export function resolutionChipLabel(opts: {
  aspect: string;
  shortEdge: number;
  sampleEdge: number;
  upscale: UpscaleMode;
}): { title: string; sub: string } {
  const ratio = ratioOfAspect(opts.aspect);
  const [width, height] = resolveCanvas(ratio, opts.shortEdge);
  const sampled = sampleEdgeOf(opts.sampleEdge, opts.shortEdge);
  if (opts.upscale === "redetail") {
    const fin = redetailTarget(ratio, opts.sampleEdge, opts.shortEdge);
    return {
      title: `${opts.shortEdge}p`,
      sub: `${sampled} → ${fin.width} × ${fin.height}`,
    };
  }
  if (isTwoPass(opts.sampleEdge, opts.shortEdge, opts.upscale)) {
    return {
      title: `${opts.shortEdge}p`,
      sub: `${sampled} → ${width} × ${height}`,
    };
  }
  return {
    title: `${opts.shortEdge}p`,
    sub: `${width} × ${height}`,
  };
}

/** Recover Continuity geometry from a legacy ``resolutionId``. */
export function geometryFromResolutionId(
  resolutionId: string,
  presets: Array<{ id: string; width?: number; height?: number; aspect?: string }>,
): { aspect: string; shortEdge: number; width: number; height: number } {
  const preset = presets.find((p) => p.id === resolutionId && !("render_width" in p && (p as { render_width?: number }).render_width));
  if (preset?.width && preset?.height) {
    const aspect = preset.aspect || describeAspect(preset.width / preset.height);
    return {
      aspect: H3_ASPECTS.some(([a]) => a === aspect) ? aspect : describeAspect(preset.width / preset.height),
      shortEdge: Math.max(MIN_SHORT_EDGE, shortEdgeOf(preset.width, preset.height)),
      width: preset.width,
      height: preset.height,
    };
  }
  const m = /^(\d+)x(\d+)/.exec(resolutionId);
  if (m) {
    const width = Number(m[1]);
    const height = Number(m[2]);
    return {
      aspect: describeAspect(width / height),
      shortEdge: Math.max(MIN_SHORT_EDGE, shortEdgeOf(width, height)),
      width,
      height,
    };
  }
  const [width, height] = resolveCanvas(ratioOfAspect(DEFAULT_ASPECT), DEFAULT_SHORT_EDGE);
  return { aspect: DEFAULT_ASPECT, shortEdge: DEFAULT_SHORT_EDGE, width, height };
}
