# AGENTS.md

Canonical guide for AI agents using **h3-ws** to generate video on Apple Silicon.

**Read [`ROADMAP.md`](ROADMAP.md)** for architecture and phase status. Prompting for H3 should follow a Context-IR skeleton (scene, action, camera, look, audio) — a full DIRECTOR.md lands in P7.

## Stack

| Piece | Role |
|-------|------|
| `server.py` | Local MiniMax-H3 inference via native **h3.c**. One job at a time. Embeds Web UI. |
| `h3_backend.py` | Spawns `./h3`. Warm interactive session for FL2VA **and** Ref2VA (`h3_session.py`); one-shot fallback if the session dies. |
| `web_ui.py` + `web/` | Browser library, quality presets, SSE progress. |
| Weights | `models/MiniMax-H3/{FL2VA,Ref2VA}` from `MiniMaxAI/MiniMax-H3` |

**Default Web UI:** http://127.0.0.1:8765/
**WebSocket:** ws://127.0.0.1:8765/ws

The app opens in **Ref2VA** mode by default (add an image/video/audio reference to enable Generate; switch modes as needed).

This stack does **not** run cloud prompt expansion. What you send is what H3 sees. h3.c media I/O is the PyAV shim (`scripts/h3-av`); no system ffmpeg. `third_party/h3.c` is a **vendored local fork** of antirez/h3.c (in-tree sources, not a git submodule) — managed inside this repo, never pushed upstream. See [`DEV.md`](DEV.md) for the current handoff state.

## Vendored h3.c fork (agents)

Rebuild after any fork edit: `./scripts/build_h3.sh`. Package the desktop app with PyInstaller (`H3WS.spec`) so `dist/H3-WS.app` embeds the new binary. Never push Continuity/`main` to `fasth3-live` or patches to upstream antirez/h3.c.

Already in this fork (do not re-port):

| Topic | Notes |
|-------|--------|
| VAE tiles | Locked to **256** px (`configured_tile_pixels`); Metal 320 auto-pick caused 16px quilt seams ([antirez/h3.c#1](https://github.com/antirez/h3.c/pull/1) lesson). Keep 256. |
| LoRA | Runtime `--lora PATH:SCALE` fuse in `h3_lora.c` (up to 8). Offline Turbo fold PRs are redundant. |
| Preview latents | `--preview-latent` / `preview_estimate` already present. |
| GQA precision | Causal GQA keeps Q×scale in F32; barrier before reusing reduction scratch (#4, #63). |
| Metal TG align | GQA `score_bytes` rounded to 16 bytes (#44) — required under `MTL_DEBUG_LAYER`. |
| VAE RGB | Unpack clamps with `fminf(fmaxf(...))` so NaNs become 0 (#9). |
| Warm VAE cache | `h3_cache_invalidate_inputs()` drops conditioning/DiT but keeps Video VAE when latent WxH unchanged (#68). CLI `!ref-*` / first-last use it; `!cache clear` still full-clears. |
| Long SDPA | Non-causal MPSGraph SDPA splits when `sequence > 12288` (block **4096**). Kill-switch: `H3_SDPA_MAX_QUERY_ROWS=0` or `--sdpa-query-block 0`. Needed on M2/M3 Ultra (#64 / Apple FB24605554). Do not go back to 2048: at 97k Ref2VA tokens it runs 29.2 s/attention vs 15.1 s (MLX fused ceiling 15.2 s). Test: `./h3_sdpa_split_tests [seq] [heads]`. |
| SDPA alignment | MPSGraph SDPA is ~2× slower when the sequence isn't a multiple of 8 (105377-row Ref2VA attention: 39.0 s vs 18.4 s). Non-causal graphs pad Q/K/V to a multiple of 32 (`H3_SDPA_PAD_ROWS`, 0 disables), mask padded keys with −∞, and slice rows back. Covered by `h3_sdpa_split_tests` (odd default length, BF16 + F32). |
| DiT op profile | `H3_DIT_OP_PROFILE=N` waits after each op of the first N blocks and prints ms/block per op; h3 also logs the DiT sequence composition at load. Use it before guessing where denoise time goes. |
| Qwen causal GQA | Prompts of **≥1024 tokens** run MPSGraph SDPA with an in-graph **F32** cast (0.7 → ~10 TFLOPS; 12k-token Ref2VA text encoder 195 → 38 s), split into causal 4096-row query blocks above 12288 rows (each block only sees keys up to its end). The BF16 graph (`H3_MPS_GQA_BF16`) fails the GQA oracle — don't use it. Shorter prompts keep the Metal kernel bit-for-bit; past ~7.9k tokens the kernel uses `h3_gqa_causal_tiled_bf16` (online softmax). Switches: `H3_GQA_KERNEL=1`, `H3_MPS_GQA=1`, `H3_GQA_MPS_MIN_ROWS`, `H3_GQA_FORCE_TILED` / `H3_GQA_TILE_KEYS`. Embedding drift vs the kernel equals the kernel's own tile-order noise floor (rel RMS 0.0072 over 50 BF16 layers). |
| Conv3d speed | MPSGraph native Conv3d runs ~6 TFLOPS F32 on M3 Ultra; `h3_gpu_conv3d_graph` emits one Conv2d per temporal tap (~29 TFLOPS, same F32 math). Video VAE encoder recycles activation buffers (fresh multi-GB buffers cost ~3 s/tile in page zero-fill). Ref-video encode 10.4 → 2.7 min. Kill-switch: `H3_CONV3D_NATIVE=1`. Test: `./h3_conv3d_tests`. |
| Safetensors | Headers must be padded so `(8 + header_size) % 8 == 0` or `h3_st_read_header` refuses (GPU mmap). Unit fixtures must pad with spaces. |

**Deferred:** [antirez/h3.c#55](https://github.com/antirez/h3.c/pull/55) `H3_ATTENTION_CACHE` int8 stream — later product/Models phase for lower-RAM Macs; not urgent on 128–512 GB Studio.

**Tests (in `third_party/h3.c`):** `./h3_tests`, `./h3_gqa_tests`, `./h3_cache_invalidate_tests`. Skip CUDA / GUI / packaging upstream PRs.

## Weights (mandatory)

**Never download MiniMax-H3 weights on the development workstation.** Fetch them only on the Apple Silicon test host.

Only the official MiniMax-H3 **native** checkpoint trees. Not LTX, not Diffusers root shards, not community quants.

The Hugging Face repo is ~464 GB. Unfiltered `hf download MiniMaxAI/MiniMax-H3` pulls everything: native `FL2VA/` (~134 GB), native `Ref2VA/` (~134 GB, of which ~72 GB duplicates FL2VA encoder/VAE), and a Diffusers copy at repo root (~196 GB) that h3.c never opens.

On the test host:

```
python scripts/download_model.py                 # FL2VA only, ~134 GB; resumes, never deletes
python scripts/download_model.py --with-ref2va   # + Ref2VA transformer ~62 GB
python scripts/download_model.py --status        # what is already on disk
```

Build the engine: `./scripts/build_h3.sh` (needs `third_party/h3.c` sources in-tree).

## Frame math

H3 snaps **up** to `5 + 17n` at 24 fps (22, 39, 56, 107, 243, 362, …). Width and height must each be multiples of 32, at least 32, and their product must not exceed 768×1344. The UI offers tagged canvases: 1:1 (256 through 768), exact 16:9 `1024×576` / 9:16 `576×1024`, largest near-16:9 `1248×704` / `704×1248`, 4:5 `768×960` and 5:4 `960×768`, 4:3 / 3:4, and the 7:4 / 4:7 pixel-cap extremes `1344×768` / `768×1344`. H3-Base is a 768p model; 512×512 is the default development size.

## Modes (v1)

- `t2va` — text to video+audio (FL2VA)
- `first_frame` / `last_frame` / `fl2va` — image anchors
- `ref2va` — ordered image / silent video / video / video+audio / audio references (Ref2VA)

Do not mix first/last-frame anchors with Ref2VA references. Prompt Ref2VA with `Picture N` / `Video N` / `Audio N` in list order. Standalone audio must accompany an image or video. Limits: ≤9 images, ≤3 videos, ≤3 audio, mixed files ≤12.

## Quality presets

`four_step` · `aggressive` · `fast` (default) · `balanced` · `close`

h3.c defaults are `--steps 20 --layers 50 --reuse 1`. The UI always lets you edit steps, layers, and reuse (a preset fills them). Close keeps `--steps 50` explicit: 50 complete 50-block denoiser forwards — the oracle when a fast mode changes subject, anatomy, motion, or composition.

`--reuse` and `--core-reuse` are mutually exclusive. Do not combine token-reduction with `--layers 40 --reuse 3`. `--ssd-streaming` saves RAM and makes denoise much slower — leave it off unless the process is killed for memory. On M5, `--use-int8-row-fc2` is on automatically.

## LoRA

The Web UI LoRA menu can download and enable FL2VA DiT adapters. First builtin: [Tutu MiniMax-H3 Audio-Video 20→8 NFE](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA) (step 100 at strength 0.8, 8 steps). h3.c fuses `W += scale * B @ A` at DiT load (`--lora PATH:SCALE`). Not compatible with `--ssd-streaming`. Rebuild `./h3` after pull so the fuse patch is in the binary.
