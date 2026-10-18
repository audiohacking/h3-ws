import {
  extendWithStyleOf,
  filterStyles,
  leadWithStyle,
  setStyleVocabulary,
  STYLE_ATLAS,
  styleOfPhrase,
  stripStyleOfPhrases,
} from "./styleAtlas";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

setStyleVocabulary([
  "Claymation with visible fingerprint texture and gently stuttering stop-motion movement",
  "Live-action, modern cinematic film with anamorphic flares and rain-slick night textures",
]);

const clay = "Claymation with visible fingerprint texture and gently stuttering stop-motion movement";

assert(leadWithStyle("", clay) === clay, "empty prompt");
assert(
  leadWithStyle("A dog runs.", clay) === `${clay}, a dog runs.`,
  "article lowercased",
);
assert(
  leadWithStyle("Marcus waits at the gate", clay) === `${clay}, Marcus waits at the gate`,
  "name preserved",
);

const noir = "Live-action, modern cinematic film with anamorphic flares and rain-slick night textures";
const swapped = leadWithStyle(leadWithStyle("A dog runs.", clay), noir);
assert(swapped === `${noir}, a dog runs.`, `swap not stack: ${swapped}`);

assert(STYLE_ATLAS.styles.length === 941, `count ${STYLE_ATLAS.styles.length}`);
assert(STYLE_ATLAS.categories.length === 8, "categories");
assert(filterStyles("claymation", "").length > 0, "search claymation");
assert(filterStyles("", "2D Animation").every((s) => s.category === "2D Animation"), "shelf");

assert(styleOfPhrase("@img-1") === "(using the style of @img-1)", "style-of cue");
assert(
  extendWithStyleOf("A dog runs.", "@img-2") === "A dog runs. (using the style of @img-2)",
  "extend style-of",
);
assert(
  stripStyleOfPhrases("Noir look (using the style of @img-1) more") === "Noir look more",
  "strip style-of",
);
assert(
  extendWithStyleOf("Noir (using the style of @img-1)", "@img-3") ===
    "Noir (using the style of @img-3)",
  "replace style-of",
);

console.log("style atlas tests ok");
