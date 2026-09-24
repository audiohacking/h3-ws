import { useEffect, useState } from "react";
import type { NetworkSettingsPublic, RefineSettingsPublic } from "../../types";

type DiscoverEndpoint = {
  name: string;
  base_url: string;
  models: string[];
  reachable?: boolean;
};

type DiscoverHint = {
  kind: string;
  name: string;
  path?: string;
  note: string;
};

type Props = {
  open: boolean;
  onClose: () => void;
  api: string;
  initial: RefineSettingsPublic;
  onSaved: (next: RefineSettingsPublic) => void;
  networkInitial?: NetworkSettingsPublic | null;
  onNetworkSaved?: (next: NetworkSettingsPublic) => void;
};

const EMPTY_NETWORK: NetworkSettingsPublic = {
  listen_lan: false,
  bind_host: "127.0.0.1",
  port: 8765,
  lan_urls: [],
  desktop_supervised: false,
};

export function FeaturesPopup({
  open,
  onClose,
  api,
  initial,
  onSaved,
  networkInitial,
  onNetworkSaved,
}: Props) {
  const [enabled, setEnabled] = useState(initial.enabled);
  const [baseUrl, setBaseUrl] = useState(initial.base_url);
  const [model, setModel] = useState(initial.model);
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [listenLan, setListenLan] = useState(networkInitial?.listen_lan ?? false);
  const [networkInfo, setNetworkInfo] = useState<NetworkSettingsPublic>(
    networkInitial ?? EMPTY_NETWORK,
  );
  const [saving, setSaving] = useState(false);
  const [discovering, setDiscovering] = useState(false);
  const [endpoints, setEndpoints] = useState<DiscoverEndpoint[]>([]);
  const [cliHints, setCliHints] = useState<DiscoverHint[]>([]);
  const [discoverNote, setDiscoverNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setEnabled(initial.enabled);
    setBaseUrl(initial.base_url);
    setModel(initial.model);
    setApiKey("");
    setClearKey(false);
    setListenLan(networkInitial?.listen_lan ?? false);
    setNetworkInfo(networkInitial ?? EMPTY_NETWORK);
    setError(null);
    setNote(null);
    setEndpoints([]);
    setCliHints([]);
    setDiscoverNote(null);
  }, [open, initial, networkInitial]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void (async () => {
      try {
        const r = await fetch(`${api}/api/settings/network`);
        if (!r.ok || cancelled) return;
        const net = (await r.json()) as NetworkSettingsPublic;
        if (cancelled) return;
        setNetworkInfo(net);
        setListenLan(net.listen_lan);
        onNetworkSaved?.(net);
      } catch {
        /* keep initial */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, api]); // eslint-disable-line react-hooks/exhaustive-deps -- refresh once per open

  if (!open) return null;

  async function discover() {
    setDiscovering(true);
    setError(null);
    setDiscoverNote(null);
    try {
      const r = await fetch(`${api}/api/settings/refine/discover`, { method: "POST" });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "Discover failed");
      }
      const data = (await r.json()) as {
        endpoints?: DiscoverEndpoint[];
        cli_hints?: DiscoverHint[];
        note?: string;
      };
      setEndpoints(data.endpoints ?? []);
      setCliHints(data.cli_hints ?? []);
      setDiscoverNote(data.note ?? null);
      if (!(data.endpoints ?? []).length && !(data.cli_hints ?? []).length) {
        setDiscoverNote(
          "No local OpenAI-compatible servers found. Start LM Studio, Ollama, llama.cpp, or a Codex/Agent proxy, then try again.",
        );
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setDiscovering(false);
    }
  }

  function applyEndpoint(ep: DiscoverEndpoint) {
    setBaseUrl(ep.base_url);
    setEnabled(true);
    if (ep.models[0]) setModel(ep.models[0]);
    setNote(`Using ${ep.name} at ${ep.base_url}`);
  }

  async function save() {
    setSaving(true);
    setError(null);
    setNote(null);
    const body: Record<string, unknown> = {
      enabled,
      base_url: baseUrl,
      model,
    };
    if (clearKey) body.api_key = "";
    else if (apiKey.trim()) body.api_key = apiKey.trim();
    try {
      const r = await fetch(`${api}/api/settings/refine`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "Save failed");
      }
      const next = (await r.json()) as RefineSettingsPublic;
      onSaved(next);

      const nr = await fetch(`${api}/api/settings/network`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ listen_lan: listenLan }),
      });
      if (!nr.ok) {
        const err = await nr.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "Network save failed");
      }
      const net = (await nr.json()) as NetworkSettingsPublic & {
        rebinding?: boolean;
        restart_required?: boolean;
      };
      setNetworkInfo(net);
      onNetworkSaved?.(net);
      if (net.rebinding) {
        setNote(
          "Rebinding the server to " +
            (net.listen_lan ? "all interfaces (0.0.0.0)" : "localhost only") +
            "… the UI may flicker for a moment.",
        );
        setTimeout(() => onClose(), 1200);
        return;
      }
      if (net.restart_required) {
        setNote("Saved. Quit and reopen H3-WS (or restart python server.py) to apply the new bind.");
        return;
      }
      onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal features-modal"
        role="dialog"
        aria-labelledby="features-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal__header">
          <h2 id="features-title" className="modal__title">
            Settings
          </h2>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </header>

        <div className="modal__body">
          <h3 className="features-section-title">Network</h3>
          <p className="features-lead">
            By default the app only accepts connections from this Mac. Turn on LAN access to
            reach the UI from phones or other machines on your private network.
          </p>
          <label className="features-toggle">
            <input
              type="checkbox"
              checked={listenLan}
              onChange={(e) => setListenLan(e.target.checked)}
            />
            Listen on all interfaces (0.0.0.0)
          </label>
          {listenLan && (
            <div className="features-lan-urls">
              <div className="field-label">Open from other devices</div>
              {(networkInfo.lan_urls.length > 0
                ? networkInfo.lan_urls
                : [`http://<this-mac-ip>:${networkInfo.port}/`]
              ).map((url) => (
                <code key={url} className="features-lan-url">
                  {url}
                </code>
              ))}
            </div>
          )}

          <h3 className="features-section-title">Refine</h3>
          <p className="features-lead">
            Optional remote Refine rewrites prompts over an OpenAI-compatible API before you
            generate. Use Discover to find LM Studio, Ollama, llama.cpp, or a local Codex/Agent
            proxy already running on this Mac.
          </p>
          <label className="features-toggle">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            Enable Refine
          </label>

          <div className="features-discover-row">
            <button
              type="button"
              className="btn-secondary btn-compact"
              onClick={() => void discover()}
              disabled={discovering}
            >
              {discovering ? "Scanning…" : "Discover local APIs"}
            </button>
          </div>

          {endpoints.length > 0 && (
            <ul className="features-discover-list">
              {endpoints.map((ep) => (
                <li key={ep.base_url}>
                  <button type="button" className="features-discover-item" onClick={() => applyEndpoint(ep)}>
                    <strong>{ep.name}</strong>
                    <span>{ep.base_url}</span>
                    {ep.models[0] ? <em>{ep.models.slice(0, 3).join(", ")}</em> : null}
                  </button>
                </li>
              ))}
            </ul>
          )}
          {cliHints.length > 0 && (
            <ul className="features-discover-hints">
              {cliHints.map((h) => (
                <li key={h.name}>
                  <strong>{h.name}</strong> — {h.note}
                </li>
              ))}
            </ul>
          )}
          {discoverNote && <p className="features-note">{discoverNote}</p>}

          <label className="field-label">
            Base URL
            <input
              type="url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:1234/v1"
              disabled={!enabled}
            />
          </label>
          <label className="field-label">
            Model
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="model id"
              disabled={!enabled}
            />
          </label>
          <label className="field-label">
            API key {initial.key_set ? "(set — leave blank to keep)" : "(optional)"}
            <input
              type="password"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setClearKey(false);
              }}
              placeholder={initial.key_set ? "••••••••" : ""}
              disabled={!enabled}
              autoComplete="off"
            />
          </label>
          {initial.key_set && (
            <label className="features-toggle">
              <input
                type="checkbox"
                checked={clearKey}
                onChange={(e) => {
                  setClearKey(e.target.checked);
                  if (e.target.checked) setApiKey("");
                }}
                disabled={!enabled}
              />
              Clear stored API key
            </label>
          )}
          {note && <p className="features-note">{note}</p>}
          {error && <p className="form-error">{error}</p>}
        </div>

        <footer className="modal__footer">
          <button type="button" className="btn-secondary" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={() => void save()} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </footer>
      </div>
    </div>
  );
}
