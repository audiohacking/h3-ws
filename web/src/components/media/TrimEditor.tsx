/**
 * Continuity segment editor for video/audio — in/out range, waveform, track switch.
 * Port of ComfyUI-Continuity-Mac `web/creator/trim.js`.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { clamp, formatTime, mediaSrc } from "../../crop";
import type { MediaTrim, RefKind } from "../../types";

const MIN_SEGMENT = 0.25;
const WAVE_COLOUR = "rgba(24,24,27,.34)";

export type TrackChoice = "picture+sound" | "picture" | "sound";

const TRACK_ORDER: TrackChoice[] = ["picture+sound", "picture", "sound"];
const TRACK_LABEL: Record<TrackChoice, string> = {
  "picture+sound": "Picture + sound",
  picture: "Picture only",
  sound: "Sound only",
};

type Props = {
  path: string;
  name?: string;
  kind: "audio" | "video" | "silent_video" | "video_audio";
  trim?: MediaTrim | null;
  /** Offer Continuity track switch (video kinds only). */
  showTrack?: boolean;
  track?: TrackChoice;
  /** Shot length in seconds — drawn on the track + "Take Ns" button. */
  cardSeconds?: number | null;
  previewUrl?: string;
  durationHint?: number;
  onCancel: () => void;
  onUse: (result: { trim: MediaTrim | null; track?: TrackChoice; kind?: RefKind }) => void;
};

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function Icon({ d, size = 16 }: { d: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

function drawPeaks(canvas: HTMLCanvasElement, data: Float32Array | null, colour: string) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  if (!data?.length) return;
  ctx.fillStyle = colour;
  const middle = height / 2;
  for (let x = 0; x < width; x += 1) {
    const from = Math.floor((x / width) * data.length);
    const to = Math.max(from + 1, Math.floor(((x + 1) / width) * data.length));
    let peak = 0;
    for (let at = from; at < to && at < data.length; at += 1) {
      if (data[at] > peak) peak = data[at];
    }
    const bar = Math.max(1, peak * (height - 4));
    ctx.fillRect(x, middle - bar / 2, 1, bar);
  }
}

function trackToKind(track: TrackChoice, base: RefKind): RefKind {
  if (track === "picture") return "silent_video";
  if (track === "sound") return "audio";
  if (base === "video_audio") return "video_audio";
  return "video";
}

export function TrimEditor({
  path,
  name,
  kind,
  trim: initial,
  showTrack = kind !== "audio",
  track: initialTrack,
  cardSeconds,
  previewUrl,
  durationHint,
  onCancel,
  onUse,
}: Props) {
  const isVideo = kind !== "audio";
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const stageRef = useRef<HTMLCanvasElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<HTMLCanvasElement>(null);
  const peaksRef = useRef<Float32Array | null>(null);

  const [duration, setDuration] = useState(durationHint ?? 0);
  const [start, setStart] = useState(initial?.start ?? 0);
  const [end, setEnd] = useState(initial?.end ?? Infinity);
  const [track, setTrack] = useState<TrackChoice | undefined>(initialTrack);
  const [hasAudio, setHasAudio] = useState<boolean | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playhead, setPlayhead] = useState(0);
  const [status, setStatus] = useState("Reading the clip…");
  const card = Number(cardSeconds) > 0 ? Number(cardSeconds) : null;

  const startRef = useRef(start);
  const endRef = useRef(end);
  startRef.current = start;
  endRef.current = end;

  const setDurationFit = useCallback((seconds: number) => {
    setDuration(seconds);
    setEnd((prev: number) => {
      const nextEnd = Math.min(Number.isFinite(prev) ? prev : seconds, seconds);
      setStart((prevStart: number) => {
        const s = Math.max(0, Math.min(prevStart, Math.max(0, nextEnd - MIN_SEGMENT)));
        if (nextEnd <= s) return 0;
        return s;
      });
      if (nextEnd <= 0) return seconds;
      return nextEnd;
    });
  }, []);

  const drawFrame = useCallback(() => {
    const stage = stageRef.current;
    const media = mediaRef.current as HTMLVideoElement | null;
    if (!stage || !media || !media.videoWidth) return;
    const ratio = window.devicePixelRatio || 1;
    const w = media.videoWidth;
    const h = media.videoHeight;
    stage.width = Math.round(w * ratio);
    stage.height = Math.round(h * ratio);
    const ctx = stage.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.drawImage(media, 0, 0, w, h);
  }, []);

  useEffect(() => {
    const src = mediaSrc(path, previewUrl);
    const media = document.createElement(isVideo ? "video" : "audio") as HTMLVideoElement | HTMLAudioElement;
    media.preload = isVideo ? "auto" : "metadata";
    if (isVideo) (media as HTMLVideoElement).playsInline = true;
    media.src = src;
    mediaRef.current = media;

    const onMeta = () => {
      const dur = Number.isFinite(media.duration) ? media.duration : 0;
      if (dur) setDurationFit(dur);
      setStatus("");
    };
    const onData = () => drawFrame();
    const onSeek = () => drawFrame();
    const onTime = () => {
      const s = startRef.current;
      const e = endRef.current;
      if (media.currentTime >= e - 0.02 && !media.paused) {
        media.pause();
        media.currentTime = s;
      }
      setPlayhead(media.currentTime || 0);
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onErr = () =>
      setStatus("This browser cannot play this file — the segment can still be set by time.");

    media.addEventListener("loadedmetadata", onMeta);
    if (isVideo) {
      media.addEventListener("loadeddata", onData);
      media.addEventListener("seeked", onSeek);
    }
    media.addEventListener("timeupdate", onTime);
    media.addEventListener("play", onPlay);
    media.addEventListener("pause", onPause);
    media.addEventListener("error", onErr);

    void fetch(`/api/media/peaks?path=${encodeURIComponent(path)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.peaks?.length) {
          peaksRef.current = Float32Array.from(data.peaks);
          if (waveRef.current) drawPeaks(waveRef.current, peaksRef.current, WAVE_COLOUR);
        }
        if (typeof data?.has_audio === "boolean") {
          setHasAudio(data.has_audio);
          if (!data.has_audio) setTrack("picture");
          else if (track === undefined && showTrack) setTrack("picture+sound");
        }
        if (!duration && data?.duration) setDurationFit(data.duration);
      })
      .catch(() => {});

    return () => {
      media.pause();
      media.removeAttribute("src");
      media.load();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, previewUrl, isVideo]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const follow = () => {
      drawFrame();
      raf = requestAnimationFrame(follow);
    };
    raf = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(raf);
  }, [playing, drawFrame]);

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      if (waveRef.current) drawPeaks(waveRef.current, peaksRef.current, WAVE_COLOUR);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const fraction = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect?.width) return 0;
    return clamp((clientX - rect.left) / rect.width, 0, 1);
  };

  const seek = (time: number, pause = false) => {
    const media = mediaRef.current;
    if (!media || !duration) return;
    if (pause) media.pause();
    media.currentTime = clamp(time, 0, duration);
    setPlayhead(media.currentTime);
    if (isVideo) drawFrame();
  };

  const snapCard = (value: number) => {
    if (!card) return value;
    const target = start + card;
    return Math.abs(value - target) <= 2 / 24 ? target : value;
  };

  const setEdge = (edge: "start" | "end", value: number) => {
    if (duration <= MIN_SEGMENT) return;
    if (edge === "start") {
      const next = Math.min(Math.max(0, value), end - MIN_SEGMENT);
      setStart(next);
      seek(next, true);
    } else {
      const next = Math.max(Math.min(duration, snapCard(value)), start + MIN_SEGMENT);
      setEnd(next);
      seek(Math.max(start, next - 1 / 24), true);
    }
  };

  const slide = (nextStart: number, length: number) => {
    if (!duration) return;
    const s = Math.min(Math.max(0, nextStart), Math.max(0, duration - length));
    setStart(s);
    setEnd(s + length);
    seek(s, true);
  };

  const beginDrag = (
    event: React.PointerEvent,
    begin: (from: number) => (at: number) => void,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    if (!duration) return;
    const node = event.currentTarget as HTMLElement;
    node.setPointerCapture(event.pointerId);
    const from = fraction(event.clientX);
    const step = begin(from);
    let moved = false;
    const move = (e: PointerEvent) => {
      const at = fraction(e.clientX);
      if (Math.abs(at - from) * duration > 0.02) moved = true;
      step(at);
    };
    const up = (e: PointerEvent) => {
      node.removeEventListener("pointermove", move);
      node.removeEventListener("pointerup", up);
      node.removeEventListener("pointercancel", up);
      if (!moved) seek(fraction(e.clientX) * duration);
    };
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", up);
    node.addEventListener("pointercancel", up);
  };

  const resolvedEnd = duration ? Math.min(end, duration) : 0;
  const isWhole = !duration || (start <= 0.001 && resolvedEnd >= duration - 0.001);
  const fits =
    card && duration
      ? Math.abs(resolvedEnd - start - Math.min(card, duration)) <= 1 / 24
      : false;
  const shownTrack = track ?? "picture+sound";

  const percent = (time: number) => (duration ? (time / duration) * 100 : 0);

  return (
    <div
      className="mmc-overlay"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="mmc-trim" role="dialog" aria-label="Segment">
        <div className="mmc-trim-head-row">
          <span className="mmc-trim-name">{name || path.split(/[/\\]/).pop()}</span>
          <button type="button" className="mmc-close" onClick={onCancel} aria-label="Close">
            ✕
          </button>
        </div>

        {isVideo && <canvas ref={stageRef} className="mmc-trim-media" />}

        <div className="mmc-trim-bar">
          <button
            type="button"
            className="mmc-trim-play"
            title="Play the segment"
            onClick={() => {
              const media = mediaRef.current;
              if (!media || !duration) return;
              if (media.paused) {
                if (media.currentTime < start || media.currentTime >= resolvedEnd - 0.02) {
                  media.currentTime = start;
                }
                void media.play().catch(() => {});
              } else {
                media.pause();
              }
            }}
          >
            <Icon d={playing ? "M6 5h4v14H6zM14 5h4v14h-4z" : "M8 5v14l11-7z"} />
          </button>
          <div
            ref={trackRef}
            className={`mmc-trim-track${isVideo ? "" : " mmc-trim-track-tall"}`}
            onPointerDown={(e) => {
              if (e.target !== trackRef.current && e.target !== waveRef.current) return;
              seek(fraction(e.clientX) * duration);
            }}
          >
            <canvas ref={waveRef} className="mmc-trim-wave" />
            {card && !fits && (
              <div
                className="mmc-trim-card"
                style={{ left: `${percent(start)}%`, width: `${percent(Math.min(card, duration || card))}%` }}
              >
                <span>card · {card.toFixed(2)} s</span>
              </div>
            )}
            <div
              className="mmc-trim-sel"
              tabIndex={0}
              title="Drag to slide the segment"
              style={{ left: `${percent(start)}%`, width: `${Math.max(0, percent(resolvedEnd) - percent(start))}%` }}
              onPointerDown={(e) =>
                beginDrag(e, (from) => {
                  const origin = start;
                  const length = resolvedEnd - start;
                  return (at) => slide(origin + (at - from) * duration, length);
                })
              }
              onKeyDown={(e) => {
                const step = e.shiftKey ? 1 : 1 / 24;
                if (e.key === "ArrowLeft") {
                  e.preventDefault();
                  slide(start - step, resolvedEnd - start);
                } else if (e.key === "ArrowRight") {
                  e.preventDefault();
                  slide(start + step, resolvedEnd - start);
                }
              }}
            />
            <div className="mmc-trim-head" style={{ left: `${percent(playhead)}%` }} />
            <div
              className="mmc-trim-handle mmc-trim-start"
              role="slider"
              tabIndex={0}
              title="Segment start"
              style={{ left: `${percent(start)}%` }}
              onPointerDown={(e) => beginDrag(e, () => (at) => setEdge("start", at * duration))}
            />
            <div
              className="mmc-trim-handle mmc-trim-end"
              role="slider"
              tabIndex={0}
              title="Segment end"
              style={{ left: `${percent(resolvedEnd)}%` }}
              onPointerDown={(e) => beginDrag(e, () => (at) => setEdge("end", at * duration))}
            />
          </div>
        </div>

        <div className="mmc-trim-read">
          {!duration ? (
            <span>{status}</span>
          ) : (
            <>
              <span>
                {isWhole
                  ? `Whole clip · ${formatTime(duration)}`
                  : `${formatTime(start)} – ${formatTime(resolvedEnd)}`}
              </span>
              <span className="mmc-trim-len">{(resolvedEnd - start).toFixed(1)} s</span>
              {status ? <span className="mmc-trim-note">{status}</span> : null}
            </>
          )}
        </div>

        <div className="mmc-trim-foot">
          <button
            type="button"
            className="mmc-ghost"
            disabled={isWhole || !duration}
            title="Reset to the whole file"
            onClick={() => {
              setStart(0);
              setEnd(duration);
              seek(0, true);
            }}
          >
            Whole clip
          </button>
          {card ? (
            <button
              type="button"
              className="mmc-ghost"
              disabled={!duration || fits}
              title="Cut the segment to the length of the shot referencing it, from the in point."
              onClick={() => {
                const length = Math.min(card, duration);
                const s = Math.min(start, duration - length);
                setStart(s);
                setEnd(s + length);
                seek(s, true);
              }}
            >
              Take {card.toFixed(2)} s
            </button>
          ) : null}
          {showTrack && (
            <div className="mmc-seg" role="group" aria-label="What to reference from this clip">
              {TRACK_ORDER.map((choice) => {
                const unavailable = hasAudio === false && choice !== "picture";
                return (
                  <button
                    key={choice}
                    type="button"
                    className="mmc-seg-opt"
                    aria-pressed={choice === shownTrack}
                    disabled={unavailable}
                    title={unavailable ? "This clip has no audio track." : TRACK_LABEL[choice]}
                    onClick={() => setTrack(choice)}
                  >
                    {TRACK_LABEL[choice]}
                  </button>
                );
              })}
            </div>
          )}
          <span className="mmc-trim-spacer" />
          <button type="button" className="mmc-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="mmc-add"
            disabled={!duration}
            onClick={() => {
              const trim = isWhole
                ? null
                : { start: round3(start), end: round3(resolvedEnd) };
              const chosen = hasAudio === false ? "picture" : shownTrack;
              onUse({
                trim,
                track: showTrack ? chosen : undefined,
                kind: showTrack ? trackToKind(chosen, kind) : undefined,
              });
            }}
          >
            Use
          </button>
        </div>
      </div>
    </div>
  );
}
