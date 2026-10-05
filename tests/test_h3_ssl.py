"""TLS CA helpers for Models downloads."""

from __future__ import annotations

import os
import ssl
import tempfile
import unittest
from pathlib import Path

import h3_ssl


class TestSslCerts(unittest.TestCase):
    def tearDown(self) -> None:
        h3_ssl._applied = None
        for key in h3_ssl._ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.pop("H3_SSL_CA_FILE", None)

    def test_ensure_ssl_certs_sets_env_from_certifi(self) -> None:
        import certifi

        expected = Path(certifi.where())
        self.assertTrue(expected.is_file())
        got = h3_ssl.ensure_ssl_certs()
        self.assertEqual(got, expected)
        resolved = str(expected.resolve())
        self.assertEqual(os.environ["SSL_CERT_FILE"], resolved)
        self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], resolved)
        self.assertEqual(os.environ["CURL_CA_BUNDLE"], resolved)

    def test_h3_ssl_ca_file_override(self) -> None:
        import certifi

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "custom-ca.pem"
            fake.write_bytes(Path(certifi.where()).read_bytes())
            os.environ["H3_SSL_CA_FILE"] = str(fake)
            got = h3_ssl.ensure_ssl_certs()
            self.assertEqual(got.resolve(), fake.resolve())
            self.assertEqual(os.environ["SSL_CERT_FILE"], str(fake.resolve()))

    def test_ssl_context_loads_bundle(self) -> None:
        ctx = h3_ssl.ssl_context()
        self.assertIsInstance(ctx, ssl.SSLContext)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
