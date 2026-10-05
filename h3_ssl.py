"""TLS CA bundle for model downloads and HF hub (macOS / frozen apps).

python.org and some Homebrew interpreters ship without a usable system CA store,
which surfaces as ``SSL: CERTIFICATE_VERIFY_FAILED`` / ``unable to get local
issuer certificate`` on Models downloads. Prefer ``certifi``'s Mozilla bundle
and point OpenSSL / requests / huggingface_hub at it.
"""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

log = logging.getLogger("h3-ssl")

_ENV_KEYS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_applied: str | None = None


def ca_bundle_path() -> Path | None:
    """Return a readable CA bundle path, or None to keep interpreter defaults."""
    custom = os.environ.get("H3_SSL_CA_FILE", "").strip()
    if custom:
        path = Path(custom).expanduser()
        return path if path.is_file() else None
    try:
        import certifi

        where = certifi.where()
        if where and Path(where).is_file():
            return Path(where)
    except Exception:  # noqa: BLE001 — optional runtime dep on some hosts
        pass
    for key in _ENV_KEYS:
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path
    return None


def ensure_ssl_certs() -> Path | None:
    """Point process TLS clients at a known-good CA bundle.

    Safe to call repeatedly. Override with ``H3_SSL_CA_FILE`` when a custom
    corporate / MITM store is required.
    """
    global _applied
    path = ca_bundle_path()
    if path is None:
        return None
    resolved = str(path.resolve())
    if _applied == resolved:
        return path
    for key in _ENV_KEYS:
        os.environ[key] = resolved
    _applied = resolved
    log.debug("TLS CA bundle → %s", resolved)
    return path


def ssl_context() -> ssl.SSLContext:
    """SSLContext that verifies with the certifi (or override) CA bundle."""
    ensure_ssl_certs()
    ctx = ssl.create_default_context()
    ca = ca_bundle_path()
    if ca is not None:
        ctx.load_verify_locations(cafile=str(ca))
    return ctx
