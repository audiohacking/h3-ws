# h3-ws MCP agent rules

You drive **local MiniMax-H3** video+audio generation through the `h3-ws` MCP tools.
The MCP is a thin client of a **running** h3-ws server (`H3WS_URL`, default `http://127.0.0.1:8765`).
Start the app or `python server.py` before calling tools.

## Modes

| Mode | When | Inputs |
|------|------|--------|
| `t2va` | Text only | `prompt` |
| `first_frame` / `last_frame` / `fl2va` | Image anchors | `image_path` and/or `end_image_path` — **not** Ref2VA refs |
| `ref2va` | Ordered references | `refs[]` with `kind` + `path` — **not** first/last frames |

Never mix Ref2VA references with FL2VA start/end frames in one call.

Ref kinds: `image`, `silent_video`, `video`, `video_audio`, `audio`.
Standalone audio must accompany an image or video. Limits: ≤9 images, ≤3 videos, ≤3 audio, ≤12 files mixed.
Prompt Ref2VA with `Picture N` / `Video N` / `Audio N` in list order.

## Prompting (Context-IR)

Describe the shot — do not write UI instructions. Prefer:

```
Scene: …
Action: …
Camera: …
Look: …
Audio: …
```

Use `h3_improve_prompt` when Refine is enabled in the app Features panel.

## Frame math

H3 snaps frames **up** to `5 + 17n` at 24 fps. Width and height multiples of 32; one-pass canvases ≤ 768×1344.

## Workspaces

`h3_list_workspaces` returns declarative tabs (Create = full composer, Enhance = Upframe/ReDetail).
Generate through `h3_generate`; post-process finished clips with `h3_enhance`.

## Workflow

1. `h3_health` — confirm server + weights.
2. `h3_upload` / `h3_list_library` — stage media; use absolute paths the server can read, or upload first.
3. `h3_generate` — queue the job; keep `run_id`.
4. `h3_wait` — block until done (raise client `MCP_TOOL_TIMEOUT` for long runs).
5. Inspect returned `video_url` / paths before declaring success.

Do not invent media paths. If a tool returns an error about missing refs, fix inputs — do not fall back to text-only scenery when the user asked for references.
