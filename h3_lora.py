"""H3 DiT LoRA catalog, Hugging Face download, and h3.c --lora wiring."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from h3_paths import hf_hub_cache, hf_snapshot, load_desktop_config, writable_root

log = logging.getLogger("h3")

TAOMATE_REPO = "Robert1212star/TaoMate-H3-3Step-ComfyUI"
TAOMATE_FILE = "taomate_h3_3step_comfy.safetensors"
TAOMATE_GUIDANCE = (
    "TaoMate 3-step distill LoRA for MiniMax-H3 (ComfyUI package). "
    "Quality tiers collapse to 3 steps at strength ~0.8. Prefer this over "
    "older 8-step turbo recipes when you want the fastest turnaround."
)

DMAD_REPO = "ZhengmingYu/DMAD"
DMAD_FILE = "minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors"
DMAD_GUIDANCE = (
    "DMAD 4-step distill LoRA for MiniMax-H3 T2VA/FL2VA (paper EMA student). "
    "Best at 4 steps, strength 1.0; text-to-audio-video. Diffusers weights are "
    "converted to native qkv_proj on download for h3.c fuse. Not a Ref2VA "
    "adapter — use TaoMate/Tutu when references dominate."
)

TUTU_REPO = "tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA"
TUTU_GUIDANCE = (
    "Tutu 20→8 NFE LoRA for FL2VA. Trained for 8 Euler steps at strength 0.8. "
    "h3.c uses its own 8-step shifted schedule (not ComfyUI ManualSigmas). "
    "SSD streaming is off while a LoRA is enabled. Rebuild h3 after pull "
    "(scripts/build_h3.sh) so --lora is fused at DiT load."
)


def _hf_card(repo: str) -> str:
    return f"https://huggingface.co/{repo}"


def _hf_resolve(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _recipe(
    *,
    id: str,
    label: str,
    repo: str,
    filename: str,
    scale: float,
    guidance: str,
    category: str,
    turbo: bool = False,
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
) -> dict[str, Any]:
    """One downloadable LoRA preset with HF model-card link for prompting notes."""
    entry: dict[str, Any] = {
        "id": id,
        "label": label,
        "spec": _hf_resolve(repo, filename),
        "scale": scale,
        "guidance": guidance,
        "category": category,
        "card_url": _hf_card(repo),
        "turbo": turbo,
    }
    if steps is not None:
        entry["steps"] = steps
    if layers is not None:
        entry["layers"] = layers
    if reuse is not None:
        entry["reuse"] = reuse
    return entry


# Continuity-style filename hints — scale / steps when scanning on-disk LoRAs.
# Only hints; any other .safetensors under models/loras still appears in the catalog.
_DISK_LORA_HINTS: tuple[dict[str, Any], ...] = (
    {
        "match": re.compile(r"taomate", re.I),
        "scale": 0.8,
        "steps": 3,
        "guidance": "TaoMate 3-step distill. Quality tiers collapse to 3 steps.",
        "turbo": True,
        "category": "turbo",
        "card_url": _hf_card(TAOMATE_REPO),
    },
    {
        "match": re.compile(r"\bdmad\b|dmad_minimax", re.I),
        "scale": 1.0,
        "steps": 4,
        "guidance": DMAD_GUIDANCE,
        "turbo": True,
        "category": "turbo",
        "card_url": _hf_card(DMAD_REPO),
    },
    {
        "match": re.compile(r"tutu|20to8[_-]?nfe", re.I),
        "scale": 0.8,
        "steps": 8,
        "guidance": TUTU_GUIDANCE,
        "turbo": True,
        "category": "turbo",
        "card_url": _hf_card(TUTU_REPO),
    },
    {
        "match": re.compile(r"(?:lightx2v|[_-]turbo[_-]|\bdmd\b|8step).*(?:h3|minimax)|(?:h3|minimax).*(?:lightx2v|turbo|dmd|8step)", re.I),
        "scale": 0.6,
        "steps": 8,
        "guidance": "Turbo / distill adapter — try 6–8 steps at the suggested strength.",
        "turbo": True,
        "category": "turbo",
    },
    {
        "match": re.compile(r"(?:fasth3|3step|3[_-]?nfe).*(?:h3|minimax)|(?:h3|minimax).*(?:fasth3|3step|3[_-]?nfe)", re.I),
        "scale": 0.8,
        "steps": 3,
        "guidance": "Fast / low-NFE adapter — try 3–4 steps.",
        "turbo": True,
        "category": "turbo",
    },
    {
        "match": re.compile(r"equi360|equirect360|360-equirect", re.I),
        "scale": 1.0,
        "guidance": "Trigger `equirect360`. Generate 21:9 / 768p; unsqueeze to 2:1 + spherical metadata.",
        "category": "immersion",
        "card_url": _hf_card("shamanic/minimax-h3-equi360-lora"),
    },
    {
        "match": re.compile(r"vr180|sbs-lora", re.I),
        "scale": 1.0,
        "guidance": "Trigger `vr180sbs`. Side-by-side VR180; 21:9 / 768p then package as stereo spherical.",
        "category": "immersion",
        "card_url": _hf_card("rehan-fal/minimax-h3-vr180-sbs-lora"),
    },
    {
        "match": re.compile(r"realism-people|r34l1sm", re.I),
        "scale": 1.0,
        "guidance": "Trigger `r34l1sm`. Realistic people / faces; strength 1.0 (0.6–0.8 lighter).",
        "category": "style",
        "card_url": _hf_card("fal/MiniMax-H3-Realism-People-LoRA"),
    },
    {
        "match": re.compile(r"facial[-_]?realism|facial.realism.closeup", re.I),
        "scale": 0.8,
        "guidance": "Trigger `Facial Realism`. Close-up portrait detail; see HF card for settings.",
        "category": "style",
        "card_url": _hf_card("prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp"),
    },
    {
        "match": re.compile(r"vh5tape", re.I),
        "scale": 1.0,
        "guidance": "Trigger `vh5tape`. Worn VHS / broadcast look; 0.7–0.9 softens the wear.",
        "category": "style",
        "card_url": _hf_card("KennethFal/vh5tape-vhs-lora-minimax-h3"),
    },
    {
        "match": re.compile(r"studio1939|studio-1939|gulliv3r", re.I),
        "scale": 1.0,
        "guidance": "Trigger `gulliv3r`. 1930s hand-painted animation; 0.4–0.8 to blend under modern style.",
        "category": "style",
        "card_url": _hf_card("lovis93/studio-1939-old-animation-lora-minimax-h3"),
    },
    {
        "match": re.compile(r"16bit[-_]?pixel", re.I),
        "scale": 1.0,
        "guidance": "SNES-era pixel animation look. Prompting notes on the HF card (fal gallery tuned).",
        "category": "style",
        "card_url": _hf_card("KennethFal/16bit-pixel-lora-minimax-h3"),
    },
    {
        "match": re.compile(r"looping.?sketch|sketch.?anime", re.I),
        "scale": 1.0,
        "guidance": "Looping hand-drawn anime sketch. Strength ~0.75–1.25; see HF card.",
        "category": "style",
        "card_url": _hf_card("Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime"),
    },
    {
        "match": re.compile(r"camera.?motion", re.I),
        "scale": 0.9,
        "guidance": "Start prompt with `camera motion`, then push/pull/pan/tilt/orbit/handheld + scene.",
        "category": "motion",
        "card_url": _hf_card("Jojocodex/minimax-h3-Camera-Motion-lora"),
    },
    {
        "match": re.compile(r"spatial.?physics|wushu_spatial", re.I),
        "scale": 0.5,
        "guidance": "Object collisions / stacking / falling. Start ~0.3–0.5; see HF card.",
        "category": "motion",
        "card_url": _hf_card("Jojocodex/minimax-h3-spatial-physics-lora"),
    },
    {
        "match": re.compile(r"wushu|kung.?fu", re.I),
        "scale": 0.9,
        "guidance": "Trigger `wushu_action`. Martial-arts motion; strength 0.8–1.0; ~3.75 s clips work well.",
        "category": "motion",
        "card_url": _hf_card("Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora"),
    },
    {
        "match": re.compile(r"motion.?repair|continuity.?repair|bunny_crisp", re.I),
        "scale": 0.7,
        "guidance": "Optional `bunny_crisp_motion`. Improves continuity on sports / dance / combat.",
        "category": "motion",
        "card_url": _hf_card("JOKER141/MiniMax-H3-General-Motion-Continuity-Repair"),
    },
    {
        "match": re.compile(r"better.?motion|human.?motion", re.I),
        "scale": 0.6,
        "guidance": "Human motion / temporal consistency. Weight 0.4–0.8; ~15–30 steps; vertical 720×1280.",
        "category": "motion",
        "card_url": _hf_card("vpakarinen/better-human-motion-h3-lora"),
    },
    {
        "match": re.compile(r"weapon.?combat|bunny_weapon|Bunny_weapon", re.I),
        "scale": 0.45,
        "guidance": "Trigger `BUNNY`. Weapon combat trajectories; recommended weight ~0.45.",
        "category": "motion",
        "card_url": _hf_card("JOKER141/MiniMax-H3-Weapon-Combat-LoRA"),
    },
    {
        "match": re.compile(r"character.?swap", re.I),
        "scale": 1.0,
        "guidance": "Ref2VA character replace. Prompt Picture N carefully; short 4–5 s continuous shots.",
        "category": "utility",
        "card_url": _hf_card("akatz-ai/MiniMax-H3-Character-Swap-LoRA"),
    },
    {
        "match": re.compile(r"360.?orbit|flf2v_lora", re.I),
        "scale": 1.0,
        "guidance": "First–last frame 360° orbit that lands on the start frame. FL2VA; strength 1.0.",
        "category": "immersion",
        "card_url": _hf_card("pablodawson/MiniMax-H3-360-Orbit-LoRA"),
    },
    {
        "match": re.compile(r"crossview|cross.?view.?warp", re.I),
        "scale": 0.8,
        "guidance": "Trigger `crossview`. Ref2VA cross-view warp; strength 0.8–1.0.",
        "category": "utility",
        "card_url": _hf_card("Cseti/MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1"),
    },
    {
        "match": re.compile(r"anime.?motion", re.I),
        "scale": 0.8,
        "guidance": "Trigger `Anime-Motion`. I2V looping anime motion; see HF card for prompts.",
        "category": "style",
        "card_url": _hf_card("prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA"),
    },
)

# Turbo distill recipes first (Models / Turbo toggle), then creative library
# curated from community H3 LoRAs (non-turbo). Each entry has card_url for
# prompting / settings — most require special trigger words.
BUILTIN_LORAS: list[dict[str, Any]] = [
    _recipe(
        id="taomate_h3_3step",
        label="TaoMate H3 3-step",
        repo=TAOMATE_REPO,
        filename=TAOMATE_FILE,
        scale=0.8,
        steps=3,
        layers=50,
        reuse=1,
        guidance=TAOMATE_GUIDANCE,
        category="turbo",
        turbo=True,
    ),
    _recipe(
        id="dmad_h3_4step",
        label="DMAD H3 4-step",
        repo=DMAD_REPO,
        filename=DMAD_FILE,
        scale=1.0,
        steps=4,
        layers=50,
        reuse=1,
        guidance=DMAD_GUIDANCE,
        category="turbo",
        turbo=True,
    ),
    _recipe(
        id="tutu_20to8_nfe_step100",
        label="Tutu 20→8 NFE (step 100)",
        repo=TUTU_REPO,
        filename=(
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors"
        ),
        scale=0.8,
        steps=8,
        layers=50,
        reuse=1,
        guidance=TUTU_GUIDANCE,
        category="turbo",
        turbo=True,
    ),
    _recipe(
        id="tutu_20to8_nfe_step200",
        label="Tutu 20→8 NFE (step 200)",
        repo=TUTU_REPO,
        filename=(
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors"
        ),
        scale=0.8,
        steps=8,
        layers=50,
        reuse=1,
        guidance=TUTU_GUIDANCE,
        category="turbo",
        turbo=True,
    ),
    _recipe(
        id="tutu_20to8_nfe_step300",
        label="Tutu 20→8 NFE (step 300)",
        repo=TUTU_REPO,
        filename=(
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors"
        ),
        scale=0.8,
        steps=8,
        layers=50,
        reuse=1,
        guidance=TUTU_GUIDANCE,
        category="turbo",
        turbo=True,
    ),
    # —— Creative / effect library (non-turbo; HF cards have prompt recipes) ——
    _recipe(
        id="equi360_reviewed_v2",
        label="Equirectangular 360°",
        repo="shamanic/minimax-h3-equi360-lora",
        filename="h3-equi360-reviewed-v2-step2500.safetensors",
        scale=1.0,
        guidance=(
            "Trigger `equirect360` then describe the environment + sound. "
            "Generate 21:9 @ 768p; unsqueeze to 2:1 and add equirectangular metadata. "
            "Add “static camera, locked tripod” if the horizon warps."
        ),
        category="immersion",
    ),
    _recipe(
        id="vr180_sbs_v2",
        label="VR180 stereoscopic SBS",
        repo="rehan-fal/minimax-h3-vr180-sbs-lora",
        filename="h3-vr180-sbs-lora-v2.safetensors",
        scale=1.0,
        guidance=(
            "Trigger `vr180sbs`. Side-by-side VR180 (L|R hemispheres). "
            "Use 21:9 @ 768p; package as stereo spherical. Strength 1.0."
        ),
        category="immersion",
    ),
    _recipe(
        id="orbit_360_fl2va",
        label="360° Orbit (first–last)",
        repo="pablodawson/MiniMax-H3-360-Orbit-LoRA",
        filename="minimax_h3_flf2v_lora_v1.safetensors",
        scale=1.0,
        guidance=(
            "FL2VA orbit: same photo as first and last frame for a frozen-time "
            "360° camera move that lands on the start. Strength 1.0 — see HF card."
        ),
        category="immersion",
    ),
    _recipe(
        id="realism_people",
        label="Realism People",
        repo="fal/MiniMax-H3-Realism-People-LoRA",
        filename="h3-realism-people-t2v-i2v-r2v.safetensors",
        scale=1.0,
        guidance=(
            "Trigger `r34l1sm` then the scene. Faces, skin, handheld documentary feel. "
            "Strength 1.0 (0.6–0.8 lighter). Works T2V / I2V / R2V."
        ),
        category="style",
    ),
    _recipe(
        id="facial_realism_closeup",
        label="Facial Realism Close-Up",
        repo="prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp",
        filename="minimax-h3-facial-realism-closeup-cp2000.safetensors",
        scale=0.8,
        guidance=(
            "Trigger `Facial Realism`. Portrait / close-up micro-expressions. "
            "Read the HF card for recommended strength and framing."
        ),
        category="style",
    ),
    _recipe(
        id="vh5tape_vhs",
        label="VH5Tape VHS",
        repo="KennethFal/vh5tape-vhs-lora-minimax-h3",
        filename="vh5tape-comfyui.safetensors",
        scale=1.0,
        guidance=(
            "Trigger `vh5tape`. Soft smear, chroma bleed, tracking noise. "
            "Strength 1.0 full look; 0.7–0.9 softens wear."
        ),
        category="style",
    ),
    _recipe(
        id="studio_1939",
        label="STUDIO 1939 Animation",
        repo="lovis93/studio-1939-old-animation-lora-minimax-h3",
        filename="studio1939-light.safetensors",
        scale=1.0,
        guidance=(
            "Trigger `gulliv3r`. Golden-age hand-painted animation. "
            "Light pack default; use studio1939-strong from the card for heavier ink."
        ),
        category="style",
    ),
    _recipe(
        id="pixel_16bit",
        label="16Bit Pixel",
        repo="KennethFal/16bit-pixel-lora-minimax-h3",
        filename="16bit-pixel.safetensors",
        scale=1.0,
        guidance=(
            "SNES-era pixel clusters and held sprite poses. "
            "Prompt phrasing lives on the HF card / fal gallery recipe."
        ),
        category="style",
    ),
    _recipe(
        id="looping_sketch_anime",
        label="Looping Sketch Anime",
        repo="Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime",
        filename="minimax_h3_looping_sketch_anime_v1.safetensors",
        scale=1.0,
        guidance=(
            "Seamless looping hand-drawn anime sketches. "
            "Strength ~0.75–1.25; examples and prompts on the HF card."
        ),
        category="style",
    ),
    _recipe(
        id="anime_motion_i2v",
        label="I2V Anime Motion",
        repo="prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA",
        filename="MiniMax-H3-I2V-Anime-Motion-LoRA-1400.safetensors",
        scale=0.8,
        guidance=(
            "Trigger `Anime-Motion`. Image-to-video looping anime motion. "
            "Pair with a reference image; see HF card for sample prompts."
        ),
        category="style",
    ),
    _recipe(
        id="camera_motion_v1",
        label="Camera Motion",
        repo="Jojocodex/minimax-h3-Camera-Motion-lora",
        filename="camera_motion_h3_lora_v1_3000_pruned.safetensors",
        scale=0.9,
        guidance=(
            "Start with `camera motion`, then the move (push / pull / pan / tilt / "
            "orbit / handheld) and scene. Strength 0.8–1.0. Prompt library on the card."
        ),
        category="motion",
    ),
    _recipe(
        id="spatial_physics",
        label="Spatial Physics",
        repo="Jojocodex/minimax-h3-spatial-physics-lora",
        filename="wushu_spatial_physics_clean_3000_pruned.safetensors",
        scale=0.5,
        guidance=(
            "Object collisions, stacking, falling, occlusion. "
            "Start around 0.3–0.5; raise carefully — see HF card."
        ),
        category="motion",
    ),
    _recipe(
        id="wushu_action_v8",
        label="Wushu Action",
        repo="Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora",
        filename="wushu_h3_v8_2000step.safetensors",
        scale=0.9,
        guidance=(
            "Trigger `wushu_action`. Martial-arts techniques in natural language. "
            "Strength 0.8–1.0; ~90-frame (3.75 s) clips match training length."
        ),
        category="motion",
    ),
    _recipe(
        id="motion_continuity_repair",
        label="Motion Continuity Repair",
        repo="JOKER141/MiniMax-H3-General-Motion-Continuity-Repair",
        filename="Motion_Repair_V2.safetensors",
        scale=0.7,
        guidance=(
            "Optional trigger `bunny_crisp_motion`. Helps running / sports / dance / "
            "combat continuity. Moderate weight; too high can flatten intensity."
        ),
        category="motion",
    ),
    _recipe(
        id="better_human_motion",
        label="Better Human Motion",
        repo="vpakarinen/better-human-motion-h3-lora",
        filename="better_motion_h3_lora_v1_500.safetensors",
        scale=0.6,
        guidance=(
            "Smoother human movement (T2V/I2V). Weight 0.4–0.8; ~15–30 steps; "
            "720×1280 recommended on the card."
        ),
        category="motion",
    ),
    _recipe(
        id="weapon_combat_bunny",
        label="Weapon Combat",
        repo="JOKER141/MiniMax-H3-Weapon-Combat-LoRA",
        filename="Bunny_weapon_combatV1.safetensors",
        scale=0.45,
        guidance=(
            "Trigger `BUNNY`. Weapon trajectories and combat interaction. "
            "Recommended weight ~0.45 — see HF card demos."
        ),
        category="motion",
    ),
    _recipe(
        id="character_swap",
        label="Character Swap (Ref2VA)",
        repo="akatz-ai/MiniMax-H3-Character-Swap-LoRA",
        filename="h3_character_swap_pro4500_1000.safetensors",
        scale=1.0,
        guidance=(
            "Ref2VA character replacement. Reference the sheet as Picture N and "
            "write an explicit replace-only prompt (card has a template). "
            "Prefer short continuous 4–5 s shots."
        ),
        category="utility",
    ),
    _recipe(
        id="crossview_warp",
        label="CrossView Warp (Ref2VA)",
        repo="Cseti/MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1",
        filename="MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1_3500.safetensors",
        scale=0.8,
        guidance=(
            "Trigger `crossview`. Ref2VA cross-view / warp control. "
            "Strength 0.8–1.0 (0.8 recommended)."
        ),
        category="utility",
    ),
]


def lora_cache_dir() -> Path:
    env = os.environ.get("H3_LORA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_desktop_config()
    configured = str(cfg.get("lora_dir") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # Prefer the clone's models/loras when the user pointed the app at a git checkout.
    repo = str(cfg.get("repo_root") or "").strip()
    if repo:
        candidate = Path(repo).expanduser() / "models" / "loras"
        if candidate.is_dir():
            return candidate.resolve()
    model = str(cfg.get("model_dir") or "").strip()
    if model:
        mp = Path(model).expanduser()
        if mp.name == "MiniMax-H3" and mp.parent.name == "models":
            candidate = mp.parent / "loras"
            if candidate.is_dir():
                return candidate.resolve()
    return writable_root() / "models" / "loras"


# Hub repo names that look like LoRA / turbo adapters (not the base MiniMax tree).
_HF_LORA_REPO_RE = re.compile(
    r"lora|turbo|tutu|taomate|dmad|nfe|distill|fasth3|lightx|20to8|3step|4step|8step|"
    r"equi360|vr180|wushu|camera.?motion|spatial|vh5tape|studio.?1939|realism|"
    r"character.?swap|crossview|anime.?motion|pixel|sketch",
    re.I,
)
_HF_BASE_REPO_RE = re.compile(
    r"MiniMaxAI--MiniMax-H3$|Comfy-Org--MiniMax-H3$",
    re.I,
)
# TaoMate is ~2.4 GB; keep a little headroom, skip giant base DiT shards.
_MAX_LORA_BYTES = 3_600_000_000
_MIN_LORA_BYTES = 1_000_000


def _lora_search_roots() -> list[Path]:
    """Directories that may already hold downloaded LoRA files."""
    roots: list[Path] = []
    seen: set[str] = set()

    def push(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        roots.append(resolved)

    push(lora_cache_dir())
    push(writable_root() / "models" / "loras")
    cfg = load_desktop_config()
    repo = str(cfg.get("repo_root") or "").strip()
    if repo:
        push(Path(repo) / "models" / "loras")
    model = str(cfg.get("model_dir") or "").strip()
    if model:
        mp = Path(model)
        if mp.name == "MiniMax-H3" and mp.parent.name == "models":
            push(mp.parent / "loras")
    home = Path.home()
    for base in (
        home / "Documents" / "git" / "h3-ws" / "models" / "loras",
        home / "git" / "h3-ws" / "models" / "loras",
        Path("/Users") / home.name / "Documents" / "git" / "h3-ws" / "models" / "loras",
        home / "Documents" / "ComfyUI" / "models" / "loras",
        home / "ComfyUI-Shared" / "models" / "loras",
    ):
        push(base)

    # Hugging Face hub snapshots that expose a loras/ folder, plus LoRA-named repos.
    hub = hf_hub_cache()
    if hub.is_dir():
        try:
            for child in hub.iterdir():
                if not child.name.startswith("models--"):
                    continue
                snap = hf_snapshot(child.name, hub=hub)
                if snap is None:
                    continue
                push(snap / "loras")
                push(snap / "models" / "loras")
                low = child.name.lower()
                if _HF_LORA_REPO_RE.search(low) and not _HF_BASE_REPO_RE.search(child.name):
                    push(snap)
        except OSError:
            pass
    return roots


def normalize_lora_spec(spec: str) -> str:
    raw = (spec or "").strip().strip("'\"")
    if not raw:
        return ""
    raw = raw.replace("/blob/", "/resolve/", 1)
    if raw.startswith("hf.co/"):
        raw = "https://huggingface.co/" + raw[len("hf.co/") :]
    if raw.startswith("huggingface.co/"):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.netloc in {"hf.co", "www.hf.co"}:
        raw = f"https://huggingface.co{parsed.path}"
        if parsed.query:
            raw += f"?{parsed.query}"
    if re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        raw = f"https://huggingface.co/{raw}/resolve/main"
    return raw


def _parse_hf_resolve(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) >= 5 and parts[2] == "resolve":
        repo = f"{parts[0]}/{parts[1]}"
        revision = parts[3]
        filename = "/".join(parts[4:])
        return repo, revision, filename
    # Handle /repo/name/resolve/revision without filename
    if len(parts) == 4 and parts[2] == "resolve":
        return f"{parts[0]}/{parts[1]}", parts[3], ""
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}", "main", ""
    return None


def _label_for_spec(spec: str) -> str:
    parsed = _parse_hf_resolve(spec)
    if parsed and parsed[2]:
        return Path(parsed[2]).name
    path = Path(spec)
    if path.suffix:
        return path.name
    return spec[-48:] if len(spec) > 48 else spec


def _is_usable(path: Path | None) -> bool:
    return bool(path and path.is_file() and path.stat().st_size > 64)


def _default_filename(repo: str) -> str:
    """When a repo-only Turbo/LoRA URL has no file path, pick the catalog default."""
    if repo == TUTU_REPO:
        return (
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-"
            "bf16-comfyui.safetensors"
        )
    if repo == DMAD_REPO:
        return DMAD_FILE
    # Fall back to the first builtin's leaf if somehow empty.
    parsed = _parse_hf_resolve(normalize_lora_spec(str(BUILTIN_LORAS[0]["spec"])))
    if parsed and parsed[2]:
        return parsed[2]
    return Path(BUILTIN_LORAS[0]["spec"]).name


def _native_comfy_candidates(repo_dir: Path, filename: str) -> list[Path]:
    """Possible converted native paths next to a Diffusers HF file."""
    leaf = Path(filename).stem + "_comfy.safetensors"
    parent = Path(filename).parent
    out = [repo_dir / "native" / leaf]
    if str(parent) not in {"", "."}:
        out.insert(0, repo_dir / parent / "native" / leaf)
    return out


def find_cached_lora(repo: str, filename: str) -> Path | None:
    """Return an on-disk LoRA matching repo/filename without downloading.

    Hugging Face layouts keep files under a nested path (e.g. ``comfyui/…``).
    Older flat-leaf copies and App Support caches are also accepted.
    Diffusers packs (DMAD) prefer the converted ``native/*_comfy.safetensors``.
    """
    if not filename:
        filename = _default_filename(repo)
    repo_key = repo.replace("/", "__")
    leaf = Path(filename).name
    for root in _lora_search_roots():
        package = root / repo_key
        # Converted native layout first (DMAD / other Diffusers adapters).
        for native in _native_comfy_candidates(package, filename):
            if _is_usable(native):
                return native
        for candidate in (
            package / filename,  # nested: …/comfyui/foo.safetensors
            package / leaf,  # flat: …/foo.safetensors
        ):
            if _is_usable(candidate):
                from h3_lora_convert import ensure_native_h3_lora

                try:
                    return ensure_native_h3_lora(candidate)
                except Exception as exc:  # noqa: BLE001 — fall through to raw file
                    log.warning("LoRA native convert skipped for %s: %s", candidate, exc)
                    return candidate
        if package.is_dir():
            for hit in package.rglob(leaf):
                if ".cache" in hit.parts or "native" in hit.parts:
                    continue
                if _is_usable(hit):
                    from h3_lora_convert import ensure_native_h3_lora

                    try:
                        return ensure_native_h3_lora(hit)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("LoRA native convert skipped for %s: %s", hit, exc)
                        return hit
    return None


def resolve_lora_path(spec: str) -> Path:
    """Local path or Hugging Face download into models/loras/."""
    from h3_lora_convert import ensure_native_h3_lora

    spec = normalize_lora_spec(spec)
    if not spec:
        raise ValueError("LoRA spec is empty")
    local = Path(spec).expanduser()
    if local.is_file():
        return ensure_native_h3_lora(local.resolve())
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        if spec.startswith(("http://", "https://")):
            raise ValueError(f"only Hugging Face LoRA URLs are supported: {spec}")
        raise FileNotFoundError(f"LoRA file not found: {spec}")
    repo, revision, filename = parsed
    if not filename:
        filename = _default_filename(repo)
    hit = find_cached_lora(repo, filename)
    if hit is not None:
        log.info("Reusing cached LoRA %s", hit)
        return hit
    dest_dir = lora_cache_dir() / repo.replace("/", "__")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Keep the HF relative path so future lookups match the clone layout.
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_usable(dest):
        return ensure_native_h3_lora(dest)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download LoRAs") from exc
    log.info("Downloading LoRA %s (%s) …", repo, filename)
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        local_dir=str(dest_dir),
    )
    path = Path(local_path).resolve()
    if path != dest and _is_usable(path) and not _is_usable(dest):
        try:
            dest.unlink(missing_ok=True)
            dest.symlink_to(path)
        except OSError:
            return ensure_native_h3_lora(path)
    if not _is_usable(path):
        raise RuntimeError(f"downloaded LoRA is empty: {path}")
    return ensure_native_h3_lora(path if _is_usable(path) else dest)


def lora_cached_path(spec: str) -> Path | None:
    spec = normalize_lora_spec(spec)
    local = Path(spec).expanduser()
    if local.is_file():
        from h3_lora_convert import ensure_native_h3_lora

        try:
            return ensure_native_h3_lora(local.resolve())
        except Exception:  # noqa: BLE001
            return local.resolve()
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        return None
    repo, _rev, filename = parsed
    return find_cached_lora(repo, filename or _default_filename(repo))


def ensure_lora(spec: str) -> dict[str, Any]:
    normalized = normalize_lora_spec(spec)
    cached = lora_cached_path(normalized)
    if cached is not None:
        return {"ok": True, "spec": normalized, "path": str(cached), "cached": True}
    path = resolve_lora_path(normalized)
    return {"ok": True, "spec": normalized, "path": str(path), "cached": False}


def read_custom_loras(output_dir: Path) -> list[dict[str, Any]]:
    from web_ui import read_web_settings

    raw = read_web_settings(output_dir).get("custom_loras")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        spec = normalize_lora_spec(str(item.get("spec") or ""))
        lid = str(item.get("id") or "").strip()
        if not spec or not lid:
            continue
        try:
            scale = float(item.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        out.append(
            {
                "id": lid,
                "label": str(item.get("label") or "").strip() or _label_for_spec(spec),
                "spec": spec,
                "scale": scale,
                "custom": True,
            }
        )
    return out


def write_custom_loras(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    from web_ui import read_web_settings, write_web_settings

    data = read_web_settings(output_dir)
    data["custom_loras"] = [
        {
            "id": e["id"],
            "label": e.get("label") or _label_for_spec(str(e.get("spec") or "")),
            "spec": normalize_lora_spec(str(e.get("spec") or "")),
            "scale": float(e.get("scale") or 1.0),
        }
        for e in entries
        if e.get("id") and e.get("spec")
    ]
    write_web_settings(output_dir, data)


def _hint_for_lora_name(name: str) -> dict[str, Any]:
    for hint in _DISK_LORA_HINTS:
        if hint["match"].search(name):
            return hint
    return {}


def _hint_for_lora_path(path: Path) -> dict[str, Any]:
    """Match hints against filename and parent package (HF / clone layouts)."""
    hay = f"{path.name} {path.parent.name} {path.as_posix()}"
    return _hint_for_lora_name(hay)


def _human_lora_label(path: Path, root: Path) -> str:
    leaf = path.stem.replace("_", " ").replace("-", " ")
    try:
        rel = path.relative_to(root)
        package = rel.parts[0] if len(rel.parts) > 1 else ""
    except ValueError:
        package = path.parent.name
    package = package.replace("__", "/").replace("_", " ")
    if package and package.lower() not in leaf.lower():
        short = package.split("/")[-1] if "/" in package else package
        if len(short) > 28:
            short = short[:25] + "…"
        return f"{leaf} · {short}"
    return leaf


def _local_lora_id(path: Path) -> str:
    # Stable id from absolute path so selection survives restarts.
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()[:40]
    return f"local_{stem}_{digest}"


def _card_url_from_spec(spec: str) -> str | None:
    parsed = _parse_hf_resolve(normalize_lora_spec(spec))
    if parsed and parsed[0]:
        return _hf_card(parsed[0])
    return None


def _builtin_match_for_path(path: Path) -> dict[str, Any] | None:
    """Prefer exact safetensors leaf, else HF package folder (Owner__Repo)."""
    leaf = path.name.lower()
    by_leaf: dict[str, dict[str, Any]] = {}
    by_repo: dict[str, dict[str, Any]] = {}
    for item in BUILTIN_LORAS:
        parsed = _parse_hf_resolve(normalize_lora_spec(str(item["spec"])))
        if not parsed or not parsed[2]:
            continue
        by_leaf[Path(parsed[2]).name.lower()] = item
        by_repo.setdefault(parsed[0].replace("/", "__").lower(), item)
    hit = by_leaf.get(leaf)
    if hit:
        return hit
    for part in path.parts:
        key = part.lower()
        if key in by_repo:
            return by_repo[key]
    return None


def _entry_for_lora_path(path: Path, *, root: Path | None = None, source: str = "disk") -> dict[str, Any]:
    hint = _hint_for_lora_path(path)
    builtin = _builtin_match_for_path(path)
    label_root = root if root is not None else path.parent
    entry: dict[str, Any] = {
        "id": _local_lora_id(path),
        "label": _human_lora_label(path, label_root),
        "spec": str(path.resolve()),
        "scale": float(
            hint.get("scale")
            or (builtin or {}).get("scale")
            or 0.8
        ),
        "local": True,
        "cached": True,
        "path": str(path.resolve()),
        "source": source,
    }
    if hint.get("steps") is not None:
        entry["steps"] = int(hint["steps"])
    guidance = hint.get("guidance") or (builtin or {}).get("guidance")
    if guidance:
        entry["guidance"] = guidance
    card_url = hint.get("card_url") or (builtin or {}).get("card_url")
    if card_url:
        entry["card_url"] = card_url
    category = hint.get("category") or (builtin or {}).get("category")
    if category:
        entry["category"] = category
    if hint.get("turbo") or (builtin or {}).get("turbo"):
        entry["turbo"] = True
        entry["layers"] = int((builtin or {}).get("layers") or 50)
        entry["reuse"] = int((builtin or {}).get("reuse") or 1)
        if hint.get("steps") is None and builtin and builtin.get("steps") is not None:
            entry["steps"] = int(builtin["steps"])
    return entry


def _collect_loras_under(root: Path, *, source: str, min_bytes: int | None = None) -> dict[str, Path]:
    """Deduped leaf→path map for one tree (prefer nested comfyui/)."""
    chosen: dict[str, Path] = {}
    if not root.is_dir():
        return chosen
    floor = _MIN_LORA_BYTES if min_bytes is None and source == "hf" else (min_bytes if min_bytes is not None else 64)
    try:
        paths = sorted(root.rglob("*.safetensors"))
    except OSError:
        return chosen
    for path in paths:
        if ".cache" in path.parts or any(p.startswith(".") for p in path.parts):
            continue
        if not _is_usable(path):
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        if sz < floor or sz > _MAX_LORA_BYTES:
            continue
        # Hide Diffusers source when a converted native sibling exists.
        native_twin = path.parent / "native" / f"{path.stem}_comfy.safetensors"
        if native_twin.is_file() and "native" not in path.parts:
            continue
        try:
            rel = path.relative_to(root)
            package = rel.parts[0] if len(rel.parts) > 1 else "_"
        except ValueError:
            package = path.parent.name
        key = f"{package}/{path.name}"
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = path
            continue
        if "comfyui" in path.parts and "comfyui" not in prev.parts:
            chosen[key] = path
        elif path.stat().st_mtime > prev.stat().st_mtime and "comfyui" not in prev.parts:
            chosen[key] = path
    return chosen


def iter_hf_hub_lora_files() -> list[tuple[Path, Path]]:
    """``(file, repo_snap_root)`` for LoRA-sized weights in the HF hub cache."""
    hub = hf_hub_cache()
    if not hub.is_dir():
        return []
    out: list[tuple[Path, Path]] = []
    try:
        children = list(hub.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.name.startswith("models--"):
            continue
        if _HF_BASE_REPO_RE.search(child.name):
            continue
        low = child.name.lower()
        # Prefer repos whose name says LoRA/turbo; also accept *minimax*h3* adapters.
        if not (_HF_LORA_REPO_RE.search(low) or ("minimax" in low and "h3" in low and "lora" in low)):
            # Still pick up a bare loras/ folder under any snapshot.
            snap = hf_snapshot(child.name, hub=hub)
            if snap is None:
                continue
            for sub in (snap / "loras", snap / "models" / "loras"):
                for path in _collect_loras_under(sub, source="hf").values():
                    out.append((path, sub))
            continue
        snap = hf_snapshot(child.name, hub=hub)
        if snap is None:
            continue
        for path in _collect_loras_under(snap, source="hf").values():
            # Skip obvious non-LoRA shard names inside adapter repos.
            leaf = path.name.lower()
            if leaf.startswith("model-") and "of-" in leaf:
                continue
            if leaf in {"model.safetensors", "diffusion_pytorch_model.safetensors"}:
                # Single-file adapters sometimes use this — keep if small enough already.
                pass
            out.append((path, snap))
    return out


def scan_disk_loras() -> list[dict[str, Any]]:
    """Every usable LoRA under models/loras **and** the Hugging Face hub cache.

    Mirrors Continuity: do not pin a fixed list — whatever the user already
    downloaded (checkout or ``~/.cache/huggingface/hub``) shows up.
    """
    seen_paths: set[str] = set()
    out: list[dict[str, Any]] = []

    for root in _lora_search_roots():
        # Skip whole HF hub snaps here — handled below with better labeling.
        if "huggingface" in str(root).lower() and "hub" in str(root).lower():
            if root.name != "loras":
                continue
        for path in _collect_loras_under(root, source="disk").values():
            key = str(path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            out.append(_entry_for_lora_path(path, root=root, source="disk"))

    for path, snap in iter_hf_hub_lora_files():
        key = str(path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        entry = _entry_for_lora_path(path, root=snap, source="hf")
        # Clarify hub origin in the label when not already obvious.
        if "hf" not in entry["label"].lower() and "hub" not in entry["label"].lower():
            entry["label"] = f"{entry['label']} · HF hub"
        out.append(entry)

    out.sort(key=lambda e: str(e.get("label") or "").lower())
    return out


def lora_catalog(output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Curated download recipes (always) + other on-disk LoRAs + customs.

    Built-ins stay in the library even when already cached so Turbo/Creative
    sections remain visible; matching disk files are not duplicated under
    "On disk".
    """
    presets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    # 1) Curated HF recipes — always listed (cached flag drives the ✓ badge).
    for item in BUILTIN_LORAS:
        preset = dict(item)
        cached = lora_cached_path(str(preset["spec"]))
        preset["cached"] = cached is not None
        if cached is not None:
            seen_paths.add(str(cached.resolve()))
            preset["path"] = str(cached.resolve())
        presets.append(preset)
        seen_ids.add(str(preset["id"]))

    # 2) Other files under models/loras / HF hub (skip curated duplicates).
    for entry in scan_disk_loras():
        key = str(Path(entry["spec"]).resolve())
        if key in seen_paths:
            continue
        if entry["id"] in seen_ids:
            continue
        presets.append(entry)
        seen_ids.add(str(entry["id"]))
        seen_paths.add(key)

    # 3) User-added custom specs (settings.json).
    if output_dir is not None:
        for entry in read_custom_loras(output_dir):
            if entry["id"] in seen_ids:
                continue
            cached = lora_cached_path(entry["spec"])
            if cached is not None and str(cached.resolve()) in seen_paths:
                continue
            entry["cached"] = cached is not None
            presets.append(entry)
            seen_ids.add(entry["id"])

    for preset in presets:
        if "cached" not in preset:
            preset["cached"] = lora_cached_path(str(preset["spec"])) is not None
        if not preset.get("card_url"):
            card = _card_url_from_spec(str(preset.get("spec") or ""))
            if card:
                preset["card_url"] = card
    return presets


def materialize_loras(specs: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Download each spec and return ``{spec, path, scale}`` dicts."""
    out: list[dict[str, Any]] = []
    for spec, scale in specs:
        path = resolve_lora_path(spec)
        out.append({"spec": spec, "path": str(path), "scale": float(scale)})
    return out


def parse_lora_specs(raw: Any, catalog: list[dict[str, Any]] | None = None) -> list[tuple[str, float]]:
    """Body `loras` / `lora_specs`: [{id, spec, scale}] or [[spec, scale], ...]."""
    if raw is None:
        return []
    items = raw
    if isinstance(raw, dict):
        items = [raw]
    if not isinstance(items, list):
        raise ValueError("loras must be a list")
    catalog = catalog or []
    by_id = {str(p.get("id")): p for p in catalog}
    out: list[tuple[str, float]] = []
    for item in items:
        spec = ""
        scale = 1.0
        if isinstance(item, dict):
            lid = str(item.get("id") or "").strip()
            if lid and lid in by_id:
                spec = str(by_id[lid].get("spec") or "")
                scale = float(item.get("scale", by_id[lid].get("scale", 1.0)))
            else:
                spec = str(item.get("spec") or item.get("path") or "")
                scale = float(item.get("scale", 1.0))
        elif isinstance(item, (list, tuple)) and item:
            spec = str(item[0])
            scale = float(item[1]) if len(item) > 1 else 1.0
        elif isinstance(item, str):
            spec = item
        spec = normalize_lora_spec(spec)
        if not spec:
            continue
        out.append((spec, max(0.0, float(scale))))
    return out
