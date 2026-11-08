import { useEffect, useRef } from "react";

export type LibraryCardMenuAction =
  | "enhance"
  | "redetail"
  | "upframe"
  | "faces"
  | "longtake"
  | "save"
  | "zip"
  | "delete";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  disabled?: boolean;
  hasVideo: boolean;
  upframeNeedsSetup?: boolean;
  redetailUnavailable?: boolean;
  onAction: (action: LibraryCardMenuAction) => void;
};

/**
 * Compact overflow for library cards — keeps the foot to one short row.
 * Primary Enhance lives outside; this holds pass shortcuts + save/zip/delete.
 */
export function LibraryCardMenu({
  open,
  onOpenChange,
  disabled,
  hasVideo,
  upframeNeedsSetup,
  redetailUnavailable,
  onAction,
}: Props) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) onOpenChange(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onOpenChange(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onOpenChange]);

  function run(action: LibraryCardMenuAction) {
    onOpenChange(false);
    onAction(action);
  }

  return (
    <div className="library-menu" ref={rootRef}>
      <button
        type="button"
        className={`library-tool library-tool--more${open ? " is-on" : ""}`}
        title="More actions"
        aria-label="More actions"
        aria-expanded={open}
        aria-haspopup="menu"
        disabled={disabled}
        onClick={(e) => {
          e.stopPropagation();
          onOpenChange(!open);
        }}
      >
        ···
      </button>
      {open ? (
        <div className="library-menu__panel" role="menu" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            role="menuitem"
            disabled={!hasVideo || redetailUnavailable}
            onClick={() => run("redetail")}
          >
            ReDetail ×2
          </button>
          <button
            type="button"
            role="menuitem"
            disabled={!hasVideo}
            className={upframeNeedsSetup ? "is-needs-setup" : undefined}
            onClick={() => run("upframe")}
          >
            Upframe 2ƒ{upframeNeedsSetup ? " · setup" : ""}
          </button>
          <button
            type="button"
            role="menuitem"
            disabled={!hasVideo}
            onClick={() => run("faces")}
          >
            Faces
          </button>
          <button
            type="button"
            role="menuitem"
            disabled={!hasVideo}
            onClick={() => run("longtake")}
          >
            Long take
          </button>
          <hr className="library-menu__rule" />
          <button type="button" role="menuitem" onClick={() => run("save")}>
            Save preset
          </button>
          <button type="button" role="menuitem" onClick={() => run("zip")}>
            Export zip
          </button>
          <button
            type="button"
            role="menuitem"
            className="is-danger"
            onClick={() => run("delete")}
          >
            Delete
          </button>
        </div>
      ) : null}
    </div>
  );
}
