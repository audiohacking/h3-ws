/** Continuity framing blob — fractions of the turned, mirrored picture. */

export type CropMirror = "" | "h" | "v" | "hv";

export interface ImageCrop {
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  /** Clockwise degrees: 0 | 90 | 180 | 270 */
  turn?: number;
  mirror?: CropMirror | string;
}

const WHOLE_SLACK = 0.004;
const MIN_FRACTION = 0.01;

export function isWindowed(crop: ImageCrop | null | undefined): boolean {
  if (!crop) return false;
  const x = crop.x ?? 0;
  const y = crop.y ?? 0;
  const w = crop.w ?? 1;
  const h = crop.h ?? 1;
  return !(x <= WHOLE_SLACK && y <= WHOLE_SLACK && w >= 1 - WHOLE_SLACK && h >= 1 - WHOLE_SLACK);
}

export function isFramed(crop: ImageCrop | null | undefined): boolean {
  if (!crop) return false;
  return isWindowed(crop) || Boolean(crop.turn) || Boolean(crop.mirror);
}

export function cropLabel(crop: ImageCrop | null | undefined): string {
  if (!isFramed(crop)) return "";
  const said: string[] = [];
  if (isWindowed(crop)) said.push("cropped");
  if (crop?.turn) said.push(`↻ ${crop.turn}°`);
  if (crop?.mirror) said.push("mirrored");
  return said.join(" · ");
}

export function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

export { WHOLE_SLACK, MIN_FRACTION };

export function ratioLabel(ratio: number): string {
  const common: [number, number][] = [
    [1, 1],
    [4, 3],
    [3, 2],
    [16, 9],
    [21, 9],
    [3, 4],
    [2, 3],
    [9, 16],
    [4, 5],
    [5, 4],
  ];
  for (const [a, b] of common) {
    if (Math.abs(ratio - a / b) < 0.015) return `${a}:${b}`;
  }
  return ratio >= 1 ? `${ratio.toFixed(2)}:1` : `1:${(1 / ratio).toFixed(2)}`;
}

export function mediaSrc(path: string, previewUrl?: string): string {
  if (previewUrl) return previewUrl;
  return `/api/media?path=${encodeURIComponent(path)}`;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "–";
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest < 10 ? "0" : ""}${rest.toFixed(1)}`;
}

/** Continuity chip label for a temporal segment. */
export function trimLabel(trim: { start: number; end: number } | null | undefined): string {
  return trim ? `${formatTime(trim.start)}–${formatTime(trim.end)}` : "whole";
}

/** What the capsule's segment door shows — crop and/or trim, Continuity-style. */
export function segmentLabel(ref: {
  kind: string;
  crop?: ImageCrop | null;
  trim?: { start: number; end: number } | null;
}): string {
  const parts: string[] = [];
  if (ref.kind === "image" || ref.kind === "silent_video" || ref.kind === "video" || ref.kind === "video_audio") {
    const framed = cropLabel(ref.crop);
    if (framed) parts.push(framed);
  }
  if (ref.kind === "audio" || ref.kind === "silent_video" || ref.kind === "video" || ref.kind === "video_audio") {
    if (ref.trim) parts.push(trimLabel(ref.trim));
  }
  if (parts.length === 0) return "whole";
  return parts.join(" · ");
}
