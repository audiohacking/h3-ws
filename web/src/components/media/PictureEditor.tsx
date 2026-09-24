/**
 * Continuity picture editor — framing window, turn, mirror, aspect lock.
 * Port of ComfyUI-Continuity-Mac `web/creator/picture.js` (sans SAM cutout).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  MIN_FRACTION,
  WHOLE_SLACK,
  clamp,
  mediaSrc,
  ratioLabel,
  round4,
  type ImageCrop,
} from "../../crop";

const STAGE_EDGE = 1600;

type AspectEntry = { key: string; label: string; ratio: number | null; shot?: boolean };

const BASE_ASPECTS: AspectEntry[] = [
  { key: "free", label: "free", ratio: null },
  { key: "square", label: "1:1", ratio: 1 },
  { key: "wide", label: "16:9", ratio: 16 / 9 },
  { key: "tall", label: "9:16", ratio: 9 / 16 },
];

type Box = { x: number; y: number; w: number; h: number };

type Props = {
  path: string;
  name?: string;
  kind: "image" | "video" | "silent_video" | "video_audio";
  crop?: ImageCrop | null;
  trim?: { start: number; end: number } | null;
  /** Shot canvas ratio so "shot" can lock the window to it. */
  aspect?: { ratio: number; label: string } | null;
  previewUrl?: string;
  onCancel: () => void;
  onUse: (crop: ImageCrop | null) => void;
};

function Icon({ d, size = 15 }: { d: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

const ICONS = {
  turnLeft: "M4.5 12a7.5 7.5 0 1 1 2.2 5.3M4 7v5h5",
  turnRight: "M19.5 12a7.5 7.5 0 1 0-2.2 5.3M20 7v5h-5",
  mirrorH: "M12 3v18M8 7L3 12l5 5M16 7l5 5-5 5",
  mirrorV: "M3 12h18M7 8l5-5 5 5M7 16l5 5 5-5",
  play: "M8 5v14l11-7z",
  pause: "M6 5h4v14H6zM14 5h4v14h-4z",
};

export function PictureEditor({
  path,
  name,
  kind,
  crop: initial,
  trim,
  aspect: shotAspect,
  previewUrl,
  onCancel,
  onUse,
}: Props) {
  const isVideo = kind !== "image";
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const windowRef = useRef<HTMLDivElement>(null);
  const sourceRef = useRef<HTMLImageElement | HTMLVideoElement | null>(null);
  const sourceSize = useRef<[number, number] | null>(null);
  const stageScale = useRef(1);
  const boxRef = useRef<Box | null>(null);
  const [pw, setPw] = useState(0);
  const [ph, setPh] = useState(0);
  const [turn, setTurn] = useState(Number(initial?.turn) || 0);
  const [mirror, setMirror] = useState(String(initial?.mirror ?? ""));
  const [box, setBoxState] = useState<Box | null>(null);
  const [aspectKey, setAspectKey] = useState("free");
  const [aspect, setAspect] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState("Reading the picture…");
  const [ready, setReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [segment, setSegment] = useState<[number, number]>([0, 0]);
  const [playing, setPlaying] = useState(false);
  const [headAt, setHeadAt] = useState(0);
  const pending = useRef({
    x: initial?.x ?? 0,
    y: initial?.y ?? 0,
    w: initial?.w ?? 1,
    h: initial?.h ?? 1,
  });
  const turnRef = useRef(turn);
  const mirrorRef = useRef(mirror);
  turnRef.current = turn;
  mirrorRef.current = mirror;
  boxRef.current = box;

  const aspects: AspectEntry[] = (() => {
    const list = [...BASE_ASPECTS];
    if (shotAspect?.ratio) {
      list.splice(1, 0, {
        key: "shot",
        label: shotAspect.label || "shot",
        ratio: shotAspect.ratio,
        shot: true,
      });
    }
    return list;
  })();

  const setBox = useCallback((x: number, y: number, w: number, h: number, stageW: number, stageH: number) => {
    const minW = Math.max(1, stageW * MIN_FRACTION);
    const minH = Math.max(1, stageH * MIN_FRACTION);
    w = clamp(w, minW, stageW);
    h = clamp(h, minH, stageH);
    x = clamp(x, 0, stageW - w);
    y = clamp(y, 0, stageH - h);
    const next = { x, y, w, h };
    boxRef.current = next;
    setBoxState(next);
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const source = sourceRef.current;
    const size = sourceSize.current;
    if (!canvas || !source || !size) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const [w, h] = size;
    const t = turnRef.current;
    const m = mirrorRef.current;
    const sw = Math.round(w * stageScale.current);
    const sh = Math.round(h * stageScale.current);
    const [cw, ch] = t % 180 ? [sh, sw] : [sw, sh];
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (m.includes("h")) {
      ctx.translate(cw, 0);
      ctx.scale(-1, 1);
    }
    if (m.includes("v")) {
      ctx.translate(0, ch);
      ctx.scale(1, -1);
    }
    if (t === 90) {
      ctx.translate(cw, 0);
      ctx.rotate(Math.PI / 2);
    } else if (t === 180) {
      ctx.translate(cw, ch);
      ctx.rotate(Math.PI);
    } else if (t === 270) {
      ctx.translate(0, ch);
      ctx.rotate(-Math.PI / 2);
    }
    ctx.drawImage(source, 0, 0, sw, sh);
  }, []);

  const layout = useCallback(
    (t: number) => {
      const size = sourceSize.current;
      if (!size) return { pw: 0, ph: 0 };
      const [w, h] = size;
      const sw = Math.round(w * stageScale.current);
      const sh = Math.round(h * stageScale.current);
      const next = t % 180 ? { pw: sh, ph: sw } : { pw: sw, ph: sh };
      setPw(next.pw);
      setPh(next.ph);
      draw();
      return next;
    },
    [draw],
  );

  const arrived = useCallback(
    (source: HTMLImageElement | HTMLVideoElement, size: [number, number]) => {
      if (sourceSize.current) {
        draw();
        return;
      }
      if (!size[0] || !size[1]) {
        setStatus("This browser cannot show this file, so there is nothing to draw the window on.");
        return;
      }
      sourceRef.current = source;
      sourceSize.current = size;
      stageScale.current = Math.min(1, STAGE_EDGE / Math.max(...size));
      const { pw: stageW, ph: stageH } = layout(turnRef.current);
      const p = pending.current;
      setBox(p.x * stageW, p.y * stageH, p.w * stageW, p.h * stageH, stageW, stageH);
      setReady(true);
      setStatus("");
      draw();
    },
    [draw, layout, setBox],
  );

  useEffect(() => {
    const src = mediaSrc(path, previewUrl);
    if (isVideo) {
      const video = document.createElement("video");
      video.preload = "auto";
      video.muted = true;
      video.playsInline = true;
      video.src = src;
      const onMeta = () => {
        const dur = Number.isFinite(video.duration) ? video.duration : 0;
        setDuration(dur);
        const seg: [number, number] = trim
          ? [clamp(trim.start, 0, dur), clamp(trim.end, 0, dur)]
          : [0, dur];
        setSegment(seg);
        video.currentTime = seg[0];
      };
      const onData = () => arrived(video, [video.videoWidth, video.videoHeight]);
      const onSeek = () => draw();
      const onTime = () => {
        if (duration && video.currentTime >= segment[1] - 0.02 && !video.paused) {
          video.pause();
          video.currentTime = segment[0];
        }
        const span = segment[1] - segment[0] || 1;
        setHeadAt(clamp((video.currentTime - segment[0]) / span, 0, 1));
      };
      const onPlay = () => setPlaying(true);
      const onPause = () => setPlaying(false);
      const onErr = () =>
        setStatus("This browser cannot show this file, so there is nothing to draw the window on.");
      video.addEventListener("loadedmetadata", onMeta);
      video.addEventListener("loadeddata", onData);
      video.addEventListener("seeked", onSeek);
      video.addEventListener("timeupdate", onTime);
      video.addEventListener("play", onPlay);
      video.addEventListener("pause", onPause);
      video.addEventListener("error", onErr);
      sourceRef.current = video;
      return () => {
        video.pause();
        video.removeAttribute("src");
        video.load();
      };
    }
    const image = new Image();
    image.onload = () => arrived(image, [image.naturalWidth, image.naturalHeight]);
    image.onerror = () =>
      setStatus("This browser cannot show this file, so there is nothing to draw the window on.");
    image.src = src;
    sourceRef.current = image;
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once per path
  }, [path, previewUrl, isVideo]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const follow = () => {
      draw();
      raf = requestAnimationFrame(follow);
    };
    raf = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(raf);
  }, [playing, draw]);

  const point = (event: { clientX: number; clientY: number }) => {
    const canvas = canvasRef.current;
    if (!canvas || !pw) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return { x: 0, y: 0 };
    return {
      x: clamp(((event.clientX - rect.left) / rect.width) * pw, 0, pw),
      y: clamp(((event.clientY - rect.top) / rect.height) * ph, 0, ph),
    };
  };

  const resize = (where: string, from: Box, dx: number, dy: number, locked: number | null) => {
    let { x, y, w, h } = from;
    const right = from.x + from.w;
    const bottom = from.y + from.h;
    if (where.includes("e")) w = from.w + dx;
    if (where.includes("w")) {
      x = from.x + dx;
      w = right - x;
    }
    if (where.includes("s")) h = from.h + dy;
    if (where.includes("n")) {
      y = from.y + dy;
      h = bottom - y;
    }
    if (locked) {
      const edge = where.length === 1;
      if (edge && (where === "n" || where === "s")) w = h * locked;
      else if (edge) h = w / locked;
      else if (Math.abs(dx) >= Math.abs(dy)) h = w / locked;
      else w = h * locked;
      const maxW = where.includes("w") ? right : where.includes("e") ? pw - from.x : pw;
      const maxH = where.includes("n") ? bottom : where.includes("s") ? ph - from.y : ph;
      if (w > maxW) {
        w = maxW;
        h = w / locked;
      }
      if (h > maxH) {
        h = maxH;
        w = h * locked;
      }
      if (edge && (where === "n" || where === "s")) x = from.x + (from.w - w) / 2;
      else if (edge) y = from.y + (from.h - h) / 2;
      if (where.includes("w")) x = right - w;
      if (where.includes("n")) y = bottom - h;
    }
    const minW = Math.max(1, pw * MIN_FRACTION);
    const minH = Math.max(1, ph * MIN_FRACTION);
    if (w < minW) {
      w = minW;
      if (where.includes("w")) x = right - w;
    }
    if (h < minH) {
      h = minH;
      if (where.includes("n")) y = bottom - h;
    }
    setBox(x, y, w, h, pw, ph);
  };

  const beginHandleDrag = (where: string, event: React.PointerEvent) => {
    if (!box) return;
    event.preventDefault();
    event.stopPropagation();
    const node = event.currentTarget as HTMLElement;
    node.setPointerCapture(event.pointerId);
    const origin = point(event);
    const from = { ...box };
    const locked = aspect;
    let moved = false;
    setDragging(true);
    const move = (e: PointerEvent) => {
      const at = point(e);
      if (!moved && Math.abs(at.x - origin.x) < 2 && Math.abs(at.y - origin.y) < 2) return;
      moved = true;
      resize(where, from, at.x - origin.x, at.y - origin.y, locked);
    };
    const up = () => {
      node.removeEventListener("pointermove", move);
      node.removeEventListener("pointerup", up);
      node.removeEventListener("pointercancel", up);
      setDragging(false);
    };
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", up);
    node.addEventListener("pointercancel", up);
  };

  const beginWindowDrag = (event: React.PointerEvent) => {
    if (!box) return;
    if ((event.target as HTMLElement).classList.contains("mmc-crop-handle")) return;
    event.preventDefault();
    event.stopPropagation();
    const node = windowRef.current!;
    node.setPointerCapture(event.pointerId);
    const origin = point(event);
    const from = { ...box };
    let moved = false;
    setDragging(true);
    const move = (e: PointerEvent) => {
      const at = point(e);
      if (!moved && Math.abs(at.x - origin.x) < 2 && Math.abs(at.y - origin.y) < 2) return;
      moved = true;
      setBox(from.x + (at.x - origin.x), from.y + (at.y - origin.y), from.w, from.h, pw, ph);
    };
    const up = () => {
      node.removeEventListener("pointermove", move);
      node.removeEventListener("pointerup", up);
      node.removeEventListener("pointercancel", up);
      setDragging(false);
    };
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", up);
    node.addEventListener("pointercancel", up);
  };

  const beginDraw = (event: React.PointerEvent) => {
    if (!box || event.target !== canvasRef.current) return;
    event.preventDefault();
    const origin = point(event);
    const stage = stageRef.current!;
    stage.setPointerCapture(event.pointerId);
    let drew = false;
    const locked = aspect;
    setDragging(true);
    const move = (e: PointerEvent) => {
      const at = point(e);
      let w = Math.abs(at.x - origin.x);
      let h = Math.abs(at.y - origin.y);
      if (w < 2 && h < 2) return;
      drew = true;
      if (locked) {
        if (w / locked >= h) h = w / locked;
        else w = h * locked;
      }
      const x = at.x >= origin.x ? origin.x : origin.x - w;
      const y = at.y >= origin.y ? origin.y : origin.y - h;
      setBox(x, y, w, h, pw, ph);
    };
    const up = () => {
      stage.removeEventListener("pointermove", move);
      stage.removeEventListener("pointerup", up);
      stage.removeEventListener("pointercancel", up);
      setDragging(false);
      if (!drew && boxRef.current) {
        setBox(origin.x - boxRef.current.w / 2, origin.y - boxRef.current.h / 2, boxRef.current.w, boxRef.current.h, pw, ph);
      }
    };
    stage.addEventListener("pointermove", move);
    stage.addEventListener("pointerup", up);
    stage.addEventListener("pointercancel", up);
  };

  const turnBy = (degrees: number) => {
    if (!box || !pw || !ph) return;
    const { x, y, w, h } = box;
    const [oldPw, oldPh] = [pw, ph];
    const nextTurn = (turn + degrees) % 360;
    let nextMirror = mirror;
    if (mirror) {
      nextMirror = [...mirror]
        .map((c) => (c === "h" ? "v" : "h"))
        .sort()
        .join("");
    }
    setTurn(nextTurn);
    setMirror(nextMirror);
    turnRef.current = nextTurn;
    mirrorRef.current = nextMirror;
    const dims = layout(nextTurn);
    const cw = degrees === 90;
    if (cw) setBox(oldPh - (y + h), x, h, w, dims.pw, dims.ph);
    else setBox(y, oldPw - (x + w), h, w, dims.pw, dims.ph);
    if (aspect) setAspect(1 / aspect);
    draw();
  };

  const flip = (axis: "h" | "v") => {
    if (!box) return;
    const { x, y, w, h } = box;
    const next = mirror.includes(axis)
      ? mirror.replace(axis, "")
      : axis === "h"
        ? "h" + mirror
        : mirror + "v";
    setMirror(next);
    mirrorRef.current = next;
    draw();
    if (axis === "h") setBox(pw - (x + w), y, w, h, pw, ph);
    else setBox(x, ph - (y + h), w, h, pw, ph);
  };

  const lockAspect = (entry: AspectEntry) => {
    setAspectKey(entry.key);
    setAspect(entry.ratio);
    if (entry.ratio && box) {
      const { x, y, w, h } = box;
      let nw = w;
      let nh = w / entry.ratio;
      if (nh > h) {
        nh = h;
        nw = h * entry.ratio;
      }
      setBox(x + (w - nw) / 2, y + (h - nh) / 2, nw, nh, pw, ph);
    }
  };

  const isWhole = () => {
    if (!box || !pw || !ph) return true;
    const f = { x: box.x / pw, y: box.y / ph, w: box.w / pw, h: box.h / ph };
    return f.x <= WHOLE_SLACK && f.y <= WHOLE_SLACK && f.w >= 1 - WHOLE_SLACK && f.h >= 1 - WHOLE_SLACK;
  };

  const commit = () => {
    const crop: ImageCrop = {};
    if (box && pw && ph && !isWhole()) {
      crop.x = round4(box.x / pw);
      crop.y = round4(box.y / ph);
      crop.w = round4(box.w / pw);
      crop.h = round4(box.h / ph);
    }
    if (turn) crop.turn = turn;
    if (mirror) crop.mirror = mirror;
    onUse(Object.keys(crop).length ? crop : null);
  };

  const pixels = (): [number, number, number, number] => {
    const size = sourceSize.current;
    if (!size || !box || !pw || !ph) return [0, 0, 0, 0];
    const [w, h] = size;
    const [tw, th] = turn % 180 ? [h, w] : [w, h];
    const f = { x: box.x / pw, y: box.y / ph, w: box.w / pw, h: box.h / ph };
    return [
      Math.max(1, Math.round(f.w * tw)),
      Math.max(1, Math.round(f.h * th)),
      tw,
      th,
    ];
  };

  const [pwOut, phOut, tw, th] = pixels();
  const said: string[] = [];
  if (turn) said.push(`turned ${turn}°`);
  if (mirror) said.push("mirrored");

  const windowStyle = box && pw && ph
    ? {
        left: `${(box.x / pw) * 100}%`,
        top: `${(box.y / ph) * 100}%`,
        width: `${(box.w / pw) * 100}%`,
        height: `${(box.h / ph) * 100}%`,
      }
    : undefined;

  return (
    <div
      className="mmc-overlay"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="mmc-trim mmc-crop" role="dialog" aria-label="Crop picture">
        <div className="mmc-trim-head-row">
          <span className="mmc-trim-name">{name || path.split(/[/\\]/).pop()}</span>
          <button type="button" className="mmc-close" onClick={onCancel} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="mmc-crop-frame">
          <div
            ref={stageRef}
            className="mmc-crop-stage"
            onPointerDown={beginDraw}
          >
            <canvas ref={canvasRef} className="mmc-crop-canvas" />
            {ready && box && (
              <div
                ref={windowRef}
                className={`mmc-crop-window${dragging ? " dragging" : ""}`}
                tabIndex={0}
                title="Drag to move the window; arrow keys nudge it"
                style={windowStyle}
                onPointerDown={beginWindowDrag}
                onKeyDown={(e) => {
                  const steps: Record<string, [number, number]> = {
                    ArrowLeft: [-1, 0],
                    ArrowRight: [1, 0],
                    ArrowUp: [0, -1],
                    ArrowDown: [0, 1],
                  };
                  const step = steps[e.key];
                  if (!step || !box) return;
                  e.preventDefault();
                  const by = (e.shiftKey ? 10 : 1) * stageScale.current;
                  setBox(box.x + step[0] * by, box.y + step[1] * by, box.w, box.h, pw, ph);
                }}
              >
                {(["nw", "ne", "sw", "se"] as const).map((corner) => (
                  <div
                    key={corner}
                    className={`mmc-crop-handle mmc-crop-${corner} mmc-crop-corner`}
                    tabIndex={0}
                    role="slider"
                    title="Drag a corner to resize"
                    onPointerDown={(e) => beginHandleDrag(corner, e)}
                  />
                ))}
                {(["n", "s", "w", "e"] as const).map((edge) => (
                  <div
                    key={edge}
                    className={`mmc-crop-handle mmc-crop-${edge}`}
                    tabIndex={0}
                    role="slider"
                    title="Drag an edge to resize"
                    onPointerDown={(e) => beginHandleDrag(edge, e)}
                  />
                ))}
                <div className="mmc-crop-thirds" />
              </div>
            )}
          </div>
        </div>

        {isVideo && duration > 0 && (
          <div className="mmc-trim-bar">
            <button
              type="button"
              className="mmc-trim-play"
              title="Play the segment"
              onClick={() => {
                const video = sourceRef.current as HTMLVideoElement | null;
                if (!video) return;
                if (video.paused) {
                  if (video.currentTime < segment[0] || video.currentTime >= segment[1] - 0.02) {
                    video.currentTime = segment[0];
                  }
                  void video.play().catch(() => {});
                } else {
                  video.pause();
                }
              }}
            >
              <Icon d={playing ? ICONS.pause : ICONS.play} size={16} />
            </button>
            <div
              className="mmc-crop-scrub"
              onPointerDown={(e) => {
                e.preventDefault();
                const scrub = e.currentTarget;
                scrub.setPointerCapture(e.pointerId);
                const seek = (ev: PointerEvent) => {
                  const video = sourceRef.current as HTMLVideoElement | null;
                  if (!video) return;
                  const rect = scrub.getBoundingClientRect();
                  const fraction = clamp((ev.clientX - rect.left) / (rect.width || 1), 0, 1);
                  video.pause();
                  video.currentTime = segment[0] + fraction * (segment[1] - segment[0]);
                  setHeadAt(fraction);
                };
                const up = () => {
                  scrub.removeEventListener("pointermove", seek);
                  scrub.removeEventListener("pointerup", up);
                  scrub.removeEventListener("pointercancel", up);
                };
                seek(e.nativeEvent);
                scrub.addEventListener("pointermove", seek);
                scrub.addEventListener("pointerup", up);
                scrub.addEventListener("pointercancel", up);
              }}
            >
              <div className="mmc-trim-head" style={{ left: `${headAt * 100}%` }} />
            </div>
          </div>
        )}

        <div className="mmc-trim-read">
          {!ready ? (
            <span>{status}</span>
          ) : (
            <>
              <span>
                {isWhole()
                  ? `Whole picture · ${tw} × ${th}`
                  : `${pwOut} × ${phOut} of ${tw} × ${th}`}
              </span>
              <span className="mmc-trim-len">
                {[ratioLabel(pwOut / (phOut || 1)), ...said].filter(Boolean).join(" · ")}
              </span>
            </>
          )}
        </div>

        <div className="mmc-trim-foot">
          <div className="mmc-seg" role="group" aria-label="Turn or mirror">
            <button type="button" className="mmc-seg-opt mmc-crop-tool" title="Turn a quarter anticlockwise" onClick={() => turnBy(270)}>
              <Icon d={ICONS.turnLeft} />
            </button>
            <button type="button" className="mmc-seg-opt mmc-crop-tool" title="Turn a quarter clockwise" onClick={() => turnBy(90)}>
              <Icon d={ICONS.turnRight} />
            </button>
            <button type="button" className="mmc-seg-opt mmc-crop-tool" title="Mirror left to right" onClick={() => flip("h")}>
              <Icon d={ICONS.mirrorH} />
            </button>
            <button type="button" className="mmc-seg-opt mmc-crop-tool" title="Mirror top to bottom" onClick={() => flip("v")}>
              <Icon d={ICONS.mirrorV} />
            </button>
          </div>
          <div className="mmc-seg" role="group" aria-label="Window shape">
            {aspects.map((entry) => (
              <button
                key={entry.key}
                type="button"
                className="mmc-seg-opt"
                aria-pressed={aspectKey === entry.key}
                title={
                  entry.shot
                    ? "Lock the window to the shape of the shot it is going to"
                    : entry.ratio
                      ? `Lock the window to ${entry.label}`
                      : "Any shape"
                }
                onClick={() => lockAspect(entry)}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="mmc-ghost"
            disabled={!box || isWhole()}
            title="Take the whole picture again — the turn and the mirror stay"
            onClick={() => setBox(0, 0, pw, ph, pw, ph)}
          >
            Whole picture
          </button>
          <span className="mmc-trim-spacer" />
          <button type="button" className="mmc-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="mmc-add" onClick={commit} disabled={!ready}>
            Use
          </button>
        </div>
      </div>
    </div>
  );
}
