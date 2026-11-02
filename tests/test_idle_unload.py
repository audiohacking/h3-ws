"""Warm h3 idle unload: activity clock + unload_warm_if_idle gates."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from h3_backend import (
    DEFAULT_IDLE_UNLOAD_S,
    H3Engine,
    resolve_idle_unload_s,
)


class _FakeSession:
    def __init__(self) -> None:
        self.alive = True
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self.alive = False


def _engine() -> H3Engine:
    with patch.object(H3Engine, "__init__", lambda self, *a, **k: None):
        eng = H3Engine()  # type: ignore[call-arg]
    eng.h3_bin = Path("/tmp/fake-h3")
    eng.model_dir = Path("/tmp/fake-models")
    eng._session = None
    eng._session_boot_key = None
    eng._preview_latent_dir = None
    eng._proc = None
    eng._gen_depth = 0
    eng._last_activity_mono = time.monotonic()
    return eng


def test_resolve_idle_unload_s_defaults_and_env() -> None:
    assert resolve_idle_unload_s(None, environ={}) == DEFAULT_IDLE_UNLOAD_S
    assert resolve_idle_unload_s(120, environ={"H3_IDLE_UNLOAD_S": "999"}) == 120.0
    assert resolve_idle_unload_s(None, environ={"H3_IDLE_UNLOAD_S": "90"}) == 90.0
    assert resolve_idle_unload_s(None, environ={"H3_IDLE_UNLOAD_S": "0"}) == 0.0
    assert resolve_idle_unload_s(-5) == 0.0
    assert resolve_idle_unload_s(None, environ={"H3_IDLE_UNLOAD_S": "nope"}) == DEFAULT_IDLE_UNLOAD_S


def test_unload_skips_when_disabled_or_no_session() -> None:
    eng = _engine()
    assert eng.unload_warm_if_idle(0) is False
    assert eng.unload_warm_if_idle(1) is False


def test_unload_skips_when_busy() -> None:
    eng = _engine()
    session = _FakeSession()
    eng._session = session
    eng._gen_depth = 1
    eng._last_activity_mono = time.monotonic() - 10_000
    assert eng.unload_warm_if_idle(1) is False
    assert session.stop_calls == 0
    assert session.alive is True


def test_unload_skips_before_threshold() -> None:
    eng = _engine()
    session = _FakeSession()
    eng._session = session
    eng._last_activity_mono = time.monotonic()
    assert eng.unload_warm_if_idle(60) is False
    assert session.stop_calls == 0


def test_unload_stops_warm_session_when_idle() -> None:
    eng = _engine()
    session = _FakeSession()
    eng._session = session
    eng._session_boot_key = ("boot",)
    eng._last_activity_mono = time.monotonic() - 100
    destroyed: list[bool] = []

    def _destroy() -> None:
        destroyed.append(True)

    eng._destroy_preview_latent_dir = _destroy  # type: ignore[method-assign]
    assert eng.unload_warm_if_idle(10) is True
    assert session.stop_calls == 1
    assert session.alive is False
    assert eng._session is None
    assert eng._session_boot_key is None
    assert destroyed == [True]
    # Second call: already cold.
    assert eng.unload_warm_if_idle(10) is False


def test_touch_activity_resets_idle_clock() -> None:
    eng = _engine()
    eng._last_activity_mono = time.monotonic() - 100
    assert eng.idle_for_s() >= 99
    eng.touch_activity()
    assert eng.idle_for_s() < 1.0
