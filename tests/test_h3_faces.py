"""Geometry tests for the Continuity-ported face pass planner."""

from __future__ import annotations

import unittest

import h3_faces as faces


LEGAL = set(faces.legal_frame_counts())


class TestFaceWindows(unittest.TestCase):
    def test_legal_counts_include_grid(self) -> None:
        self.assertIn(5, LEGAL)
        self.assertIn(22, LEGAL)
        self.assertIn(362, LEGAL)
        self.assertNotIn(23, LEGAL)

    def test_exact_legal_is_one_window(self) -> None:
        for count in (5, 22, 39, 124, 226, 362):
            self.assertEqual(faces.windows(count), [(0, count)])

    def test_off_grid_covers_without_padding(self) -> None:
        for count in (23, 40, 100, 225, 363, 500, 1445):
            spans = faces.windows(count)
            lengths = {end - start for start, end in spans}
            self.assertTrue(lengths <= LEGAL, msg=f"{count}: {lengths}")
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], count)
            self.assertTrue(all(0 <= start and end <= count for start, end in spans))
            covered: set[int] = set()
            for start, end in spans:
                covered.update(range(start, end))
            self.assertEqual(len(covered), count)
            self.assertTrue(
                all(spans[i][0] < spans[i + 1][0] for i in range(len(spans) - 1))
            )

    def test_long_pass_capped(self) -> None:
        spans = faces.windows(1445)
        self.assertLessEqual(
            max(end - start for start, end in spans),
            faces.WINDOW_CAP,
        )

    def test_too_short_raises(self) -> None:
        with self.assertRaisesRegex(faces.FaceError, "nothing here to refine"):
            faces.windows(3)


class TestFaceDetectTrack(unittest.TestCase):
    def test_detect_frames_includes_last(self) -> None:
        self.assertEqual(faces.detect_frames(10), [0, 4, 8, 9])

    def test_pick_stays_on_track(self) -> None:
        prev = (10.0, 10.0, 40.0, 50.0)
        other = (200.0, 200.0, 80.0, 90.0)
        near = (12.0, 11.0, 42.0, 48.0)
        self.assertEqual(faces.pick([other, near], prev), near)

    def test_pick_largest_when_no_overlap(self) -> None:
        a = (0.0, 0.0, 10.0, 10.0)
        b = (100.0, 100.0, 50.0, 60.0)
        # Previous track is far from both — fall back to largest area.
        self.assertEqual(faces.pick([a, b], (400.0, 400.0, 5.0, 5.0)), b)


class TestFaceStrengths(unittest.TestCase):
    def test_small_face_gets_stronger_denoise(self) -> None:
        s = faces.strengths([30.0, 120.0])
        self.assertGreater(s[0], s[1])

    def test_abstain_on_large_faces(self) -> None:
        self.assertTrue(faces.should_abstain([130.0, 140.0]))
        self.assertFalse(faces.should_abstain([40.0, 140.0]))

    def test_window_denoise_scales(self) -> None:
        d = faces.window_denoise(0.45, [0.8, 0.8])
        self.assertAlmostEqual(d, 0.45 * 0.8, places=5)


class TestCropBoxes(unittest.TestCase):
    def test_crop_aspect_matches_canvas(self) -> None:
        boxes = [(100.0, 80.0, 40.0, 50.0)] * 5
        found = [True] * 5
        crops, face_rects = faces.crop_boxes(boxes, found, 512, 512)
        self.assertEqual(len(crops), 5)
        for crop in crops:
            self.assertAlmostEqual(crop[2] / crop[3], 1.0, places=5)
        self.assertEqual(len(face_rects), 5)


if __name__ == "__main__":
    unittest.main()
