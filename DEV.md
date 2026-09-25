# DEV.md — h3-ws development state & notes

Handoff doc for the next agent. Captures the current working state, what was just
shipped, known constraints, and open items. Read [`ROADMAP.md`](ROADMAP.md) for
architecture/phases and [`AGENTS.md`](AGENTS.md) for operating guidance.

**Read this first, then ROADMAP.md.** Keep this file close to the walking state of
the tree when you land meaningful engine or Continuity UX work.

---

## Current running state

- **Product remote:** `origin` = `https://github.com/audiohacking/h3-ws` (Continuity /
  `main`). `fasth3-live` is a separate experiment remote — never push Continuity
  work there. See `.cursor/rules/git-remote-fasth3-live.mdc`.
- **Engine:** vendored **local fork** at `third_party/h3.c` (in-tree sources, **not**
  a git submodule). Never push patches to upstream `antirez/h3.c`.
- **Host:** Apple Silicon Metal. Studio under test has been M3 Ultra / 512 GiB;
  patches stay portable (GQA align, SDPA split for Ultra, 256 VAE tiles).
- **Server / app:** Web UI on `:8765`; packaged app is `dist/H3-WS.app`
  (PyInstaller `H3WS.spec` + `./build/macos/codesign.sh`). Rebuild `./h3` then the
  `.app` after fork changes before user validation.
- **Models** (`models/MiniMax-H3`): FL2VA + Ref2VA native trees. Download **only** on
  the Apple Silicon test host (`scripts/download_model.py`), never on a remote
  workstation.

---

## Just landed (local fork Metal / session backports)

Upstream triage plan (antirez/h3.c PRs) → ported into `third_party/h3.c`:

| Upstream | What we took |
|----------|----------------|
| #4 + #63 | Causal GQA: keep Q×scale in F32; barrier before reusing `reductions[]` |
| #44 | Round GQA threadgroup `score_bytes` to 16 bytes (`MTL_DEBUG_LAYER`) |
| #9 | Finite VAE RGB clamp (`fminf(fmaxf)`) |
| #68 | `h3_cache_invalidate_inputs()` — warm Ref2VA keeps ~9 GiB Video VAE across ref/anchor edits |
| #64 | Split long non-causal SDPA queries (`H3_SDPA_MAX_QUERY_ROWS` / `--sdpa-query-block`) |
| (test) | `tests/test_h3.c` safetensors fixture padded to 8-byte header alignment |

**Skipped / already ours:** #1 (256 tiles — locked in `beb9d93`), #14 (runtime LoRA fuse), #34 (preview estimate). **Deferred:** #55 `H3_ATTENTION_CACHE` → Models/product phase.

**Verify after rebuild:**

```
./scripts/build_h3.sh
cd third_party/h3.c && make h3_tests h3_gqa_tests h3_cache_invalidate_tests
./h3_tests && ./h3_gqa_tests && ./h3_cache_invalidate_tests
```

GQA expected ballpark: max abs ~0.0039, BF16 mismatch ≪1%. Then rebuild
`dist/H3-WS.app` if the user is testing the packaged app.

---

## Earlier Continuity commits still relevant

- **`beb9d93`** — Metal VAE tile seams fixed by locking tiles to 256 (320 Metal path
  quilted at 16px); silent-video ingest + PyAV `adapt_audio_for_h3`; Lanczos
  “upscale” retired from UI/API (latent density later).
- Warm interactive session in `h3_backend.py` / `h3_session.py` for FL2VA **and**
  Ref2VA (one-shot fallback if the session dies).
- PyAV shim (`scripts/h3-av` / `h3_av.py`) is the only media I/O path; Homebrew
  ffmpeg is skipped when dyld-broken.
- Ref2VA is the default UI mode; Continuity-style segment editors for trim/crop
  (see `.cursor/rules/continuity-segment.mdc`).
- TAEH3 / TaoMate / Refine / Console / LAN listen — see recent `main` history;
  Models page can fetch TAEH3.

---

## Known constraints / gotchas

- **VAE tiles stay 256.** Do not reintroduce 256–320 auto-search from older
  upstream; Metal seams reappear at 320.
- **Safetensors alignment:** fork refuses headers where `(8 + header_size) % 8 != 0`
  (GPU mmap). Pad JSON with spaces in fixtures and any hand-written `.safetensors`.
- **SDPA Ultra:** long sequences can silently corrupt without #64. Defaults split
  only when `sequence > 12288`; short 512² jobs stay on the fast path. Set
  `H3_SDPA_MAX_QUERY_ROWS=0` to disable.
- **Warm cache:** input edits should call `h3_cache_invalidate_inputs`, not
  `h3_cache_clear`, or the Video VAE reloads every ref change.
- **No system ffmpeg** when dyld-broken; rely on `H3_AV` / PyAV shim.
- **Standalone audio ref invalid** unless paired with image or video.
- **`--ssd-streaming`** vs LoRA fuse: incompatible; leave streaming off unless OOM.
- **Weights policy:** never `hf download` MiniMax-H3 on the development workstation.

---

## Open / next (agents)

1. **#55 attention cache** — productize for lower-RAM Macs (build cache once, wire
   env via `h3_backend.py` / Models page). Not needed on 512 GiB Studio.
2. Continuity polish / ROADMAP P7+ (DIRECTOR.md, CLI/MCP) as scheduled.
3. Optional: validate large-canvas × long-clip jobs exercise the SDPA split path
   on Ultra without regressing short jobs.

Community map for checkpoints/LoRAs/Metal ideas (ideas only; engine stays h3.c):
https://github.com/wildminder/awesome-minimax-H3 — fetch live README + performance
guide before adopting adapters. See `.cursor/rules/awesome-minimax-h3.mdc`.
