import type { WorkspaceManifest } from "../../types";

type Props = {
  workspaces: WorkspaceManifest[];
  value: string;
  onChange: (next: string) => void;
  disabled?: boolean;
};

const FALLBACK: WorkspaceManifest[] = [
  {
    id: "full",
    label: "Create",
    hint: "Text, references, and first/last frames",
    order: 10,
    enabled: true,
    panel: "composer",
  },
  {
    id: "enhance",
    label: "Enhance",
    hint: "Upframe and ReDetail on a finished clip",
    order: 20,
    enabled: true,
    panel: "enhance",
  },
];

export function WorkspaceTabs({ workspaces, value, onChange, disabled }: Props) {
  const tabs = (workspaces.length > 0 ? workspaces : FALLBACK).filter((w) => w.enabled !== false);

  if (tabs.length <= 1) return null;

  return (
    <div className="workspace-tabs" role="tablist" aria-label="Mode">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={value === tab.id}
          className={`workspace-tabs__tab${value === tab.id ? " is-on" : ""}`}
          title={tab.hint || tab.label}
          disabled={disabled}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export function workspacePanel(
  workspaces: WorkspaceManifest[],
  id: string,
): WorkspaceManifest["panel"] {
  const hit = workspaces.find((w) => w.id === id);
  if (hit?.panel) return hit.panel;
  if (id === "enhance") return "enhance";
  if (id === "recipe") return "recipe";
  return "composer";
}
