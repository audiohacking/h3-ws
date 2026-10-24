"""Crop/composite must use Continuity affine_grid math — not W-1 perspective.

Continuity facepass._crop: quantising the box (OpenCV corners at W-1) is the
largest source of frame-to-frame jitter after trajectory smoothing. These
tests lock the sub-pixel round-trip under a moving crop.
"""

from __future__ import annotations

import unittest

import numpy as np

from h3_facepass import composite_window, warp_crop_window


def _legacy_w1_crop(frames, crops, span, canvas_w, canvas_h):
    """The pre-fix OpenCV W-1 perspective crop (jitter source)."""
    import cv2

    start, end = span
    out = np.zeros((end - start, canvas_h, canvas_w, 3), dtype=np.uint8)
    for i, index in enumerate(range(start, end)):
        x, y, bw, bh = crops[index]
        src = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
        dst = np.float32(
            [[0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        out[i] = cv2.warpPerspective(
            frames[index],
            M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out


def _legacy_w1_paste(base, refined, crops, face_rects, weights):
    import cv2
    from h3_facepass import _paste_mask, feather_in_canvas

    count = base.shape[0]
    src_h, src_w = int(base.shape[1]), int(base.shape[2])
    canvas_h, canvas_w = int(refined.shape[1]), int(refined.shape[2])
    out = base.astype(np.float32) / 255.0
    patch = refined.astype(np.float32) / 255.0
    for i in range(count):
        w = float(weights[i])
        if w <= 0:
            continue
        x, y, bw, bh = crops[i]
        feather = feather_in_canvas(bh, canvas_h)
        mask_c = _paste_mask(face_rects[i], canvas_w, canvas_h, feather)
        src = np.float32(
            [[0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]]
        )
        dst = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            patch[i], M, (src_w, src_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mask = cv2.warpPerspective(
            mask_c, M, (src_w, src_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mask = np.clip(mask, 0.0, 1.0)[..., None]
        out[i] = np.clip((1.0 - mask * w) * out[i] + (mask * w) * warped, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _face_centroid(rgb: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(rgb[:, :, 0] > 180)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


class ContinuityAffineRegistrationTests(unittest.TestCase):
    def test_affine_roundtrip_stabler_than_w1_under_moving_crop(self) -> None:
        """Paste centre must not crawl as crop size changes (Continuity claim)."""
        h, w = 480, 640
        n = 12
        frames = np.zeros((n, h, w, 3), dtype=np.uint8)
        # Bright face blob at a fixed location — only the crop box moves/grows.
        frames[:, 120:200, 260:340] = 220
        canvas = 256
        # Smoothed-like trajectory: position drifts slowly, size changes every frame
        # (the Continuity size-jitter case).
        crops = []
        face_rects = []
        for i in range(n):
            bw = 120.0 + i * 3.5
            bh = 160.0 + i * 4.0
            x = 220.0 - i * 0.4
            y = 80.0 - i * 0.3
            crops.append((x, y, bw, bh))
            # Face rect roughly centred in canvas (matches crop_boxes geometry).
            face_rects.append((80.0, 70.0, 90.0, 110.0))

        span = (0, n)
        weights = [1.0] * n

        cont_crop = warp_crop_window(frames, crops, span, canvas, canvas)
        cont_back = composite_window(
            frames.copy(), cont_crop, crops, face_rects, weights
        )
        legacy_crop = _legacy_w1_crop(frames, crops, span, canvas, canvas)
        legacy_back = _legacy_w1_paste(
            frames.copy(), legacy_crop, crops, face_rects, weights
        )

        cont_centers = [_face_centroid(cont_back[i]) for i in range(n)]
        legacy_centers = [_face_centroid(legacy_back[i]) for i in range(n)]
        true = _face_centroid(frames[0])

        def drift(centers):
            return max(
                abs(c[0] - true[0]) + abs(c[1] - true[1]) for c in centers
            )

        cont_drift = drift(cont_centers)
        legacy_drift = drift(legacy_centers)
        # Continuity affine must keep the pasted face nearer the source blob
        # across the moving-crop sequence than the old W-1 path.
        self.assertLess(
            cont_drift,
            legacy_drift * 0.85 + 0.5,
            f"affine drift {cont_drift:.2f} should beat W-1 drift {legacy_drift:.2f}",
        )
        # Absolute: pasted centroid should stay within a couple of pixels of source.
        self.assertLess(cont_drift, 4.0, f"affine paste crawled {cont_drift:.2f}px")

    def test_warp_crop_is_uint8_shaped(self) -> None:
        frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
        frames[:, 16:48, 16:48] = 200
        crops = [(10.0, 10.0, 40.0, 40.0)] * 5
        out = warp_crop_window(frames, crops, (0, 5), 32, 32)
        self.assertEqual(out.shape, (5, 32, 32, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.mean()), 50)


if __name__ == "__main__":
    unittest.main()
