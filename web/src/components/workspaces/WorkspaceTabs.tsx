export type WorkspaceId = "full" | "remake" | "enhance";

const TABS: { id: WorkspaceId; label: string; hint: string }[] = [
  { id: "full", label: "Full", hint: "Open Continuity composer" },
  { id: "remake", label: "Remake", hint: "Viggle swap / edit — locked presets" },
  { id: "enhance", label: "Enhance", hint: "Upframe / ReDetail on a library clip" },
];

export function WorkspaceTabs({
  value,
  onChange,
  disabled,
}: {
  value: WorkspaceId;
  onChange: (next: WorkspaceId) => void;
  disabled?: boolean;
}) {
  return (
    <div className="workspace-tabs" role="tablist" aria-label="Workspace">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={value === tab.id}
          className={`workspace-tabs__tab${value === tab.id ? " is-on" : ""}`}
          title={tab.hint}
          disabled={disabled}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
