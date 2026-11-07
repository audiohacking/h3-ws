# Workspace plugins

Tabs in the Web UI are declarative JSON manifests under [`workspaces/`](../workspaces/).
The server merges shipped files with optional overrides in the app data `workspaces/` directory
(same `id` replaces). Exposed as `workspaces` on `GET /api/config`.

## Ship a new specialty mode (no App.tsx change)

1. Add a LoRA recipe in [`h3_lora.py`](../h3_lora.py) (category, steps, optional `schedule`).
2. Add `workspaces/<id>.json` with `"panel": "recipe"` and a `fields` list the Recipe host understands:
   - `segment`, `textarea`, `toggle`
   - `library_video`, `library_images`, `library_clip` (MediaLibraryModal)
3. Restart the server (or reload config). The tab appears automatically.

## Built-in hosts

| `panel` | Role |
|---------|------|
| `composer` | Full Continuity Create surface |
| `enhance` | Upframe / ReDetail on a library clip |
| `recipe` | Generic form for specialty LoRA modes |

## Example skeleton

```json
{
  "id": "my_mode",
  "label": "My mode",
  "hint": "One-line description",
  "order": 30,
  "enabled": true,
  "panel": "recipe",
  "isolation": { "lora_categories": ["style"], "forbid": ["turbo"] },
  "engine": { "mode": "ref2va", "locked_steps": 20 },
  "fields": [
    { "id": "prompt", "type": "textarea", "template": "context_ir" }
  ],
  "submit": { "prompt": "prompt" }
}
```
