import { fitRefAudioTrims, REF_AUDIO_BAKE_CAP_S } from "./refAudioLimits";
import type { ReferenceItem } from "./types";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function audio(partial: Partial<ReferenceItem> & Pick<ReferenceItem, "id" | "path">): ReferenceItem {
  return {
    kind: "audio",
    name: partial.name ?? partial.path,
    enabled: true,
    ...partial,
  };
}

const long = fitRefAudioTrims([
  audio({ id: "a", path: "/tmp/a.wav", durationS: 18.6 }),
]);
assert(long.changed, "long audio should auto-trim");
assert(long.refs[0]?.trim != null, "trim set");
assert(
  Math.abs((long.refs[0]!.trim!.end - long.refs[0]!.trim!.start) - REF_AUDIO_BAKE_CAP_S) < 0.02,
  `bake window ${REF_AUDIO_BAKE_CAP_S}s`,
);

const ok = fitRefAudioTrims([
  audio({ id: "b", path: "/tmp/b.wav", durationS: 8 }),
]);
assert(!ok.changed, "short audio unchanged");

const two = fitRefAudioTrims([
  audio({ id: "c", path: "/tmp/c.wav", durationS: 10 }),
  audio({ id: "d", path: "/tmp/d.wav", durationS: 10 }),
]);
assert(two.changed, "total over bake cap");
assert(two.refs[0]!.trim == null, "first 10s stays whole");
assert(two.refs[1]!.trim != null, "second needs a trim");
const secondSpan = two.refs[1]!.trim!.end - two.refs[1]!.trim!.start;
const expectSecond = REF_AUDIO_BAKE_CAP_S - 10;
assert(Math.abs(secondSpan - expectSecond) < 0.05, `second capped to ${expectSecond}s, got ${secondSpan}`);

console.log("refAudioLimits.test.ts: ok");
