"""NAF trust-boundary regressions; synthetic bytes never leave temporary roots."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import assets, multiview, naf_checkpoint, worldsculpt
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path


class NafIntegrityTests(unittest.TestCase):
    def test_corrupt_download_is_not_promoted_and_existing_file_is_preserved(self):
        for payload, message in ((b"truncated", "size mismatch"),
                                 (b"x" * multiview.NAF_SIZE, "SHA256 mismatch")):
            for existing in (False, True):
                with self.subTest(message=message, existing=existing), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    final = resolve_storage_path(resolve_modly_layout(root), assets.AUXILIARY_ASSETS["naf"].sentinel_paths[0])
                    if existing:
                        final.parent.mkdir(parents=True)
                        final.write_bytes(b"preserved manual checkpoint")
                    def downloader(**kwargs):
                        Path(kwargs["destination"]).write_bytes(payload)
                    result = assets.bootstrap_auxiliary_assets(root, downloader=downloader, force=existing)
                    self.assertEqual(result["status"], "failed", result)
                    self.assertFalse(result["generation_allowed"])
                    self.assertIn(message, result["error"])
                    self.assertEqual(final.read_bytes() if final.exists() else None,
                                     b"preserved manual checkpoint" if existing else None)
                    self.assertFalse(list(final.parent.parent.glob(".bootstrap-*.tmp")))

    def test_verified_download_and_manual_checkpoint_remain_supported(self):
        payload = b"valid fixture checkpoint"
        with patch.object(naf_checkpoint, "NAF_SIZE", len(payload)), patch.object(
            naf_checkpoint, "NAF_SHA256", hashlib.sha256(payload).hexdigest()
        ):
            for manual in (False, True):
                with self.subTest(manual=manual), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    final = resolve_storage_path(resolve_modly_layout(root), assets.AUXILIARY_ASSETS["naf"].sentinel_paths[0])
                    if manual:
                        final.parent.mkdir(parents=True)
                        final.write_bytes(payload)
                    calls = []
                    def downloader(**kwargs):
                        calls.append(kwargs)
                        Path(kwargs["destination"]).write_bytes(payload)
                    result = assets.bootstrap_auxiliary_assets(root, downloader=downloader)
                    self.assertEqual(result["status"], "ready", result)
                    self.assertEqual(result["downloads_started"], not manual)
                    self.assertEqual(len(calls), 0 if manual else 1)
                    self.assertEqual(final.read_bytes(), payload)
                    self.assertFalse(list(final.parent.parent.glob(".bootstrap-*.tmp")))

    def test_existing_corrupt_manual_checkpoint_fails_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = resolve_storage_path(resolve_modly_layout(root), assets.AUXILIARY_ASSETS["naf"].sentinel_paths[0])
            final.parent.mkdir(parents=True)
            final.write_bytes(b"corrupt manual checkpoint")
            def forbidden(**kwargs):
                self.fail("existing manual checkpoint must not trigger a hidden download")
            result = assets.bootstrap_auxiliary_assets(root, downloader=forbidden)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["downloads_started"])
            self.assertFalse(result["generation_allowed"])
            self.assertEqual(final.read_bytes(), b"corrupt manual checkpoint")

    def test_worldsculpt_rejects_corrupt_checkpoint_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            models = {str(i): f"ckpts/{name}" for i, name in enumerate(worldsculpt._BASE_MODELS)}
            (base / "pipeline.json").write_text(json.dumps({"name": "Pixal3DImageTo3DPipeline", "args": {"models": models}}))
            for name in worldsculpt._BASE_MODELS:
                for suffix in ("json", "safetensors"):
                    file = base / "ckpts" / f"{name}.{suffix}"
                    file.parent.mkdir(exist_ok=True)
                    file.write_bytes(b"base fixture")
            for auxiliary, names in worldsculpt._AUXILIARY_FILES.items():
                for name in names:
                    file = base / "auxiliary" / auxiliary / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(b"auxiliary fixture")
            naf = base / "naf_release.pth"
            for payload, message in ((b"truncated", "size mismatch"),
                                     (b"x" * multiview.NAF_SIZE, "SHA256 mismatch")):
                with self.subTest(message=message):
                    naf.write_bytes(payload)
                    with self.assertRaisesRegex(RuntimeError, message):
                        worldsculpt.validate_base(base, naf)


if __name__ == "__main__":
    unittest.main()
