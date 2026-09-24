"""Unit tests for OpenAI-compatible Refine helpers."""

from __future__ import annotations

import unittest

from h3_refine import merge_refine_settings, normalize_base_url, refine_settings_public


class TestRefineHelpers(unittest.TestCase):
    def test_normalize_strips_chat_completions(self) -> None:
        self.assertEqual(
            normalize_base_url("http://127.0.0.1:1234/v1/chat/completions"),
            "http://127.0.0.1:1234/v1",
        )

    def test_normalize_rejects_non_http(self) -> None:
        with self.assertRaises(ValueError):
            normalize_base_url("file:///tmp")

    def test_public_hides_api_key(self) -> None:
        pub = refine_settings_public(
            {"enabled": True, "base_url": "http://x", "model": "m", "api_key": "secret"}
        )
        self.assertTrue(pub["key_set"])
        self.assertNotIn("api_key", pub)

    def test_merge_keeps_key_when_omitted(self) -> None:
        cur = {"enabled": False, "base_url": "http://a", "model": "m", "api_key": "keep"}
        nxt = merge_refine_settings(cur, {"enabled": True, "model": "n"})
        self.assertEqual(nxt["api_key"], "keep")
        self.assertTrue(nxt["enabled"])
        self.assertEqual(nxt["model"], "n")


if __name__ == "__main__":
    unittest.main()
