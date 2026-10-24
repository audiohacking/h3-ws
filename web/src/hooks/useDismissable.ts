import { useEffect, useRef, type RefObject } from "react";

/**
 * Continuity-style ``dismissable``: close on outside pointerdown (capture) or
 * Escape. Deferred one tick so the opening click does not immediately shut it.
 */
export function useDismissable(
  open: boolean,
  onClose: () => void,
  rootRef: RefObject<HTMLElement | null>,
): void {
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => {
      const t = event.target as Node | null;
      if (!t) return;
      if (rootRef.current?.contains(t)) return;
      onCloseRef.current();
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onCloseRef.current();
      }
    };
    const timer = window.setTimeout(() => {
      document.addEventListener("pointerdown", away, true);
      document.addEventListener("keydown", key, true);
    }, 0);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", key, true);
    };
  }, [open, rootRef]);
}
