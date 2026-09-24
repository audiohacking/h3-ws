"""Prompt length / prompt-file helpers for Continuity-class Context-IR text."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from h3_backend import GenerateRequest, build_h3_argv
from h3_session import (
    MAX_PROMPT_CHARS,
    PROMPT_FILE_THRESHOLD,
    session_prompt_line,
    write_session_prompt_file,
)


class PromptFileTests(unittest.TestCase):
    def test_write_preserves_newlines(self) -> None:
        text = "Scene:\nA person walks.\n\nAudio:\nFootsteps."
        with tempfile.TemporaryDirectory() as tmp:
            path = write_session_prompt_file(Path(tmp), text)
            self.assertEqual(path.read_text(encoding="utf-8"), text)
            self.assertTrue(path.is_file())

    def test_rejects_over_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "max supported"):
            write_session_prompt_file(None, "x" * (MAX_PROMPT_CHARS + 1))

    def test_session_prompt_line_collapses_whitespace(self) -> None:
        self.assertEqual(session_prompt_line("  a\n\tb  "), "a b")

    def test_oneshot_uses_prompt_file_for_long_text(self) -> None:
        long = "word " * (PROMPT_FILE_THRESHOLD // 2 + 10)
        req = GenerateRequest(
            prompt=long,
            output_path=Path("/tmp/out.mp4"),
            width=512,
            height=512,
            num_frames=22,
        )
        argv = build_h3_argv(
            h3_bin=Path("/usr/bin/true"),
            model_dir=Path("/tmp/models"),
            req=req,
        )
        self.assertIn("--prompt-file", argv)
        self.assertNotIn("-p", argv)
        path = Path(argv[argv.index("--prompt-file") + 1])
        self.assertTrue(path.is_file())
        path.unlink(missing_ok=True)

    def test_oneshot_keeps_short_prompt_on_argv(self) -> None:
        req = GenerateRequest(
            prompt="short scene",
            output_path=Path("/tmp/out.mp4"),
            width=512,
            height=512,
            num_frames=22,
        )
        argv = build_h3_argv(
            h3_bin=Path("/usr/bin/true"),
            model_dir=Path("/tmp/models"),
            req=req,
        )
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "short scene")
        self.assertNotIn("--prompt-file", argv)


if __name__ == "__main__":
    unittest.main()
