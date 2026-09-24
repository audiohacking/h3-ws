"""Continuity framing blob — turn, mirror, and a window on the picture.

Port of ComfyUI-Continuity-Mac ``creator/crop.py``. One blob shape::

    {"x": 0.25, "y": 0.1, "w": 0.4, "h": 0.6, "turn": 90, "mirror": "h"}

Fractions of the picture *as shown* after turn and mirror, in that order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TURNS = (0, 90, 180, 270)
MIRRORS = ("", "h", "v", "hv")
MIN_FRACTION = 0.01


class CropError(ValueError):
    """A framing blob that does not describe a window on a picture."""


@dataclass(frozen=True)
class Crop:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    turn: int = 0
    mirror: str = ""

    @property
    def windowed(self) -> bool:
        return not (self.x <= 0 and self.y <= 0 and self.w >= 1 and self.h >= 1)

    @property
    def turned(self) -> bool:
        return self.turn != 0 or bool(self.mirror)


def parse(raw: Any, what: str = "picture") -> Crop | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CropError(f"{what}: crop must be an object")
    try:
        x = float(raw.get("x", 0.0))
        y = float(raw.get("y", 0.0))
        w = float(raw.get("w", 1.0))
        h = float(raw.get("h", 1.0))
    except (TypeError, ValueError) as exc:
        raise CropError(f"{what}: crop needs numeric x, y, w, h fractions") from exc
    try:
        turn = int(raw.get("turn", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise CropError(f"{what}: crop turn must be 0, 90, 180 or 270") from exc
    if turn not in TURNS:
        raise CropError(f"{what}: crop turn must be 0, 90, 180 or 270 (got {turn})")
    mirror = str(raw.get("mirror") or "")
    if mirror not in MIRRORS:
        raise CropError(f"{what}: crop mirror must be one of h, v, hv (got {mirror!r})")
    if not (
        0 <= x < 1
        and 0 <= y < 1
        and w > 0
        and h > 0
        and x + w <= 1 + 1e-6
        and y + h <= 1 + 1e-6
    ):
        raise CropError(
            f"{what}: the crop window must lie inside the picture "
            f"(got x {x:.3f} y {y:.3f} w {w:.3f} h {h:.3f})"
        )
    if w < MIN_FRACTION or h < MIN_FRACTION:
        raise CropError(f"{what}: the crop window is too small to be a picture")
    crop = Crop(x=x, y=y, w=min(w, 1 - x), h=min(h, 1 - y), turn=turn, mirror=mirror)
    return crop if (crop.windowed or crop.turned) else None


def to_dict(crop: Crop) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if crop.windowed:
        out.update(
            x=round(crop.x, 4),
            y=round(crop.y, 4),
            w=round(crop.w, 4),
            h=round(crop.h, 4),
        )
    if crop.turn:
        out["turn"] = crop.turn
    if crop.mirror:
        out["mirror"] = crop.mirror
    return out


def turned_size(size: tuple[int, int], crop: Crop | None) -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    return (height, width) if crop is not None and crop.turn % 180 else (width, height)


def box(size: tuple[int, int], crop: Crop | None) -> tuple[int, int, int, int]:
    width, height = turned_size(size, crop)
    if crop is None or not crop.windowed:
        return 0, 0, width, height
    left = min(width - 1, int(round(crop.x * width)))
    top = min(height - 1, int(round(crop.y * height)))
    right = max(left + 1, min(width, int(round((crop.x + crop.w) * width))))
    bottom = max(top + 1, min(height, int(round((crop.y + crop.h) * height))))
    return left, top, right, bottom


def pil(image: Any, crop: Crop | None) -> Any:
    """A PIL image, framed. Returns the input untouched when there is nothing to do."""
    if crop is None:
        return image
    from PIL import Image

    turn = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }.get(crop.turn)
    if turn is not None:
        image = image.transpose(turn)
    if "h" in crop.mirror:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if "v" in crop.mirror:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if crop.windowed:
        image = image.crop(
            box(
                (image.width, image.height),
                Crop(crop.x, crop.y, crop.w, crop.h),
            )
        )
    return image


def apply_still(src: Path | str, dest: Path | str, crop: Crop) -> Path:
    """Write a framed PNG of ``src`` to ``dest``."""
    from PIL import Image

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        framed = pil(im.convert("RGB"), crop)
        framed.save(target, format="PNG")
    return target
