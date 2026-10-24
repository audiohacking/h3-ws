import {
  applyProgressEvent,
  composePostPassMessage,
  formatProgressMessage,
  isPostPassPhase,
} from "./progress";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

assert(isPostPassPhase("faces_window"), "faces_window is post-pass");
assert(isPostPassPhase("faces_detect"), "faces_detect is post-pass");
assert(isPostPassPhase("longtake_refine"), "longtake is post-pass");
assert(!isPostPassPhase("generating"), "generating is not post-pass");

const composed = composePostPassMessage(
  "Faces window 1/2 · ceiling 0.45 · video_t=4 · soft_hold",
  {
    stage: "video vae decode",
    step: 2,
    total: 8,
    eta_s: 54,
  },
);
assert(composed.startsWith("Faces window 1/2"), `label preserved: ${composed}`);
assert(/video vae decode/i.test(composed), `nested stage: ${composed}`);
assert(composed.includes("2/8"), `step: ${composed}`);
assert(/remaining/.test(composed), `eta: ${composed}`);

const once = composePostPassMessage(
  "Faces window 1/2 · video vae decode · 2/8",
  { stage: "video vae decode", step: 2, total: 8 },
);
assert(
  once === "Faces window 1/2 · video vae decode · 2/8",
  `no double-append: ${once}`,
);

const applied = applyProgressEvent(null, {
  type: "progress",
  phase: "faces_window",
  message: "Faces window 1/1 · ceiling 0.45 · video_t=4 · soft_hold",
  model_progress: {
    stage: "video vae decode",
    step: 1,
    total: 8,
    pct: 12,
  },
});
assert(applied.phase === "faces_window", "phase kept");
assert(applied.message.startsWith("Faces window"), `msg: ${applied.message}`);
assert(/video vae decode/i.test(applied.message), `nested in msg: ${applied.message}`);
assert(applied.step === 1 && applied.total === 8, "bar steps");

const bare = formatProgressMessage({ stage: "video vae decode", step: 3, total: 8 });
assert(/Video vae decode/i.test(bare), "bare format");
assert(!bare.startsWith("Faces"), "bare is not faces");

// Mid-pass keepalive with generic phase must not wipe faces.
const kept = applyProgressEvent(applied, {
  type: "generation_keepalive",
  phase: "generating",
  model_progress: { stage: "video vae decode", step: 4, total: 8 },
});
assert(kept.phase === "faces_window", `keepalive phase: ${kept.phase}`);
assert(kept.message.startsWith("Faces window"), `keepalive msg: ${kept.message}`);
assert(kept.message.includes("4/8"), `keepalive step: ${kept.message}`);

console.log("progress.test.ts: ok");
