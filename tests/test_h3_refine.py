"""Unit tests for OpenAI-compatible Refine helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from h3_refine import (
    discover_refine_endpoints,
    merge_refine_settings,
    normalize_base_url,
    refine_settings_public,
)
from h3_update import is_newer, parse_version


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

    def test_discover_skips_closed_ports(self) -> None:
        with mock.patch("h3_refine._port_open", return_value=False):
            with mock.patch("h3_refine._agent_cli_hints", return_value=[]):
                out = discover_refine_endpoints()
        self.assertTrue(out["ok"])
        self.assertEqual(out["endpoints"], [])


class TestUpdateHelpers(unittest.TestCase):
    def test_parse_and_compare(self) -> None:
        self.assertEqual(parse_version("v0.0.1"), (0, 0, 1))
        self.assertEqual(parse_version("0.1.0-macos"), (0, 1, 0))
        self.assertTrue(is_newer("0.0.2", "0.0.1"))
        self.assertFalse(is_newer("0.0.1", "0.0.1"))
        self.assertFalse(is_newer("0.0.1", "0.1.0"))


if __name__ == "__main__":
    unittest.main()
