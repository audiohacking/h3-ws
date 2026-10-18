import type { LoraPreset } from "../../types";
import { isTurboPreset, pickDefaultTurboId, rankTurboPreset, recipeScaleForTurbo } from "./TurboToggle";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function preset(partial: Partial<LoraPreset> & Pick<LoraPreset, "id" | "label" | "spec">): LoraPreset {
  return { scale: 0.8, ...partial };
}

export function runTurboToggleSelfTest() {
  assert(
    isTurboPreset(preset({ id: "a", label: "X", spec: "/x.safetensors", turbo: true })),
    "turbo flag",
  );
  assert(
    isTurboPreset(
      preset({
        id: "local_ref2v",
        label: "minimax h3 ref2v turbo 8step",
        spec: "/loras/lightx2v__Minimax-h3-Turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        turbo: true,
      }),
    ),
    "lightx2v ref2v turbo",
  );
  assert(
    isTurboPreset(
      preset({
        id: "taomate_h3_3step",
        label: "TaoMate H3 3-step",
        spec: "https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI/resolve/main/taomate_h3_3step_comfy.safetensors",
      }),
    ),
    "taomate by name",
  );
  assert(
    !isTurboPreset(
      preset({
        id: "character_swap",
        label: "Character Swap (Ref2VA)",
        spec: "https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/resolve/main/h3_character_swap.safetensors",
      }),
    ),
    "exclude character swap",
  );
  assert(
    !isTurboPreset(
      preset({
        id: "wan_lx",
        label: "wan2.2 i2v lightx2v 4steps",
        spec: "/ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
      }),
    ),
    "exclude wan lightx2v",
  );

  const presets = [
    preset({
      id: "ref2v",
      label: "ref2v turbo",
      spec: "/minimax_h3_ref2v_turbo_8step.safetensors",
      turbo: true,
    }),
    preset({
      id: "taomate_h3_3step",
      label: "TaoMate H3 3-step",
      spec: "/taomate_h3_3step_comfy.safetensors",
      turbo: true,
    }),
  ];
  assert(pickDefaultTurboId(presets) === "taomate_h3_3step", "default taoMate");
  assert(rankTurboPreset(presets[1]!) < rankTurboPreset(presets[0]!), "taomate ranks first");

  assert(
    recipeScaleForTurbo(
      preset({
        id: "local_ref2v",
        label: "minimax h3 ref2v turbo 8step",
        spec: "/lightx2v__Minimax-h3-Turbo/minimax_h3_ref2v_turbo_8step.safetensors",
        scale: 0.6, // stale catalog must not win
      }),
    ) === 1.0,
    "lightx2v recipe scale 1.0",
  );
  assert(
    recipeScaleForTurbo(
      preset({
        id: "taomate_h3_3step",
        label: "TaoMate H3 3-step",
        spec: "/taomate_h3_3step_comfy.safetensors",
        scale: 0.8,
        default_scale: 0.8,
      }),
    ) === 0.8,
    "taomate recipe scale 0.8",
  );
}

runTurboToggleSelfTest();
console.log("TurboToggle tests ok");
