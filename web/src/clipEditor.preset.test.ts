import {
  generationPresetFields,
  suggestedPresetName,
  type GenerationSnapshot,
} from "./clipEditor";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function snap(partial: Partial<GenerationSnapshot> = {}): GenerationSnapshot {
  return {
    prompt: "A fisherman mends a net on the pier.",
    mode: "ref2va",
    routing: "ref2va",
    selectedCastIds: [],
    quality: "fast",
    resolutionId: "768x768",
    durationId: "5s",
    numSteps: 3,
    layers: 50,
    reuse: 1,
    seed: "7",
    loraPresetIds: ["taomate_h3_3step"],
    turboEnabled: true,
    turboTier: "balanced",
    refs: [
      {
        id: "r1",
        kind: "image",
        path: "/tmp/ref.jpg",
        name: "ref.jpg",
        previewUrl: "blob:dead",
      },
    ],
    imagePath: null,
    endImagePath: null,
    clipMultiplier: 1,
    tokenReduction: false,
    ssdStreaming: false,
    faces: true,
    facesDenoise: 0.45,
    ...partial,
  };
}

assert(suggestedPresetName("Hello pier\nSecond line") === "Hello pier", "first line");
const long = "x".repeat(60);
assert(suggestedPresetName(long).endsWith("…"), "truncates with ellipsis");
assert(suggestedPresetName(long).length <= 48, "truncation length");

const fields = generationPresetFields(snap());
assert(fields.prompt.includes("fisherman"), "stores prompt");
assert(fields.loraIds[0] === "taomate_h3_3step", "lora ids");
assert(fields.turboEnabled === true, "turbo flag");
const refs = fields.generation.refs as { previewUrl?: string }[];
assert(refs[0]?.previewUrl === undefined, "strips blob preview");
assert(fields.generation.faces === true, "faces flag in generation");

console.log("clipEditor.preset.test.ts: ok");
