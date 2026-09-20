import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import multiview, naf_checkpoint, runtime


class SingleViewLocalNafTests(unittest.TestCase):
    def test_official_checkpoint_size_and_hash_are_enforced(self):
        self.assertEqual(multiview.NAF_SIZE, 2664431)
        self.assertEqual(multiview.NAF_SHA256, "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "naf_release.pth"
            checkpoint.write_bytes(b"verified-local-checkpoint")
            with patch.object(naf_checkpoint, "NAF_SIZE", checkpoint.stat().st_size), patch.object(
                naf_checkpoint, "NAF_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ):
                multiview._verify_naf_checkpoint(checkpoint)
                checkpoint.write_bytes(b"corrupted-local-checkpoint")
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    multiview._verify_naf_checkpoint(checkpoint)
                checkpoint.write_bytes(b"x" * len(b"verified-local-checkpoint"))
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    multiview._verify_naf_checkpoint(checkpoint)

    def test_host_local_naf_override_covers_low_vram_and_standard_and_restores_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "naf_release.pth"
            checkpoint.write_bytes(b"verified-local-checkpoint")
            calls = []

            class Extractor:
                def __init__(self):
                    self.naf_model = None
                    self.model = types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(device="cuda")]))

                def _load_naf(self):
                    calls.append("remote torch.hub.load")
                    raise AssertionError("remote NAF must not run")

            original = lambda config: Extractor()
            inference = types.SimpleNamespace(build_image_cond_model=original)

            class LocalNaf:
                def eval(self):
                    return self

                def requires_grad_(self, enabled):
                    return self

            source = {"sources": {"naf": {"kind": "local", "value": str(checkpoint)}}}
            with patch.object(naf_checkpoint, "NAF_SIZE", checkpoint.stat().st_size), patch.object(
                naf_checkpoint, "NAF_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ), patch.object(multiview, "_load_local_naf", side_effect=lambda path, device: calls.append((path, device)) or LocalNaf()):
                for low_vram in (True, False):
                    with runtime._scoped_single_view_naf_extractors(inference, source):
                        extractors = [inference.build_image_cond_model({}) for _ in range(4)]
                        for extractor in extractors:
                            extractor._load_naf()
                            self.assertIsInstance(extractor.naf_model, LocalNaf)
                    self.assertIs(inference.build_image_cond_model, original)
            self.assertEqual(len(calls), 8)
            self.assertTrue(all(path == checkpoint and device == "cuda" for path, device in calls))

    def test_run_job_uses_local_extractor_loader_in_both_host_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"image")
            checkpoint = root / "naf_release.pth"
            checkpoint.write_bytes(b"verified-local-checkpoint")
            output = root / "output"
            output.mkdir()
            calls = []

            class Extractor:
                naf_model = None
                model = types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(device="cuda")]))

                def _load_naf(self):
                    calls.append("remote torch.hub.load")
                    raise AssertionError("remote NAF must not run")

            inference = types.ModuleType("inference")
            original = lambda config: Extractor()
            inference.build_image_cond_model = original
            hubconf = types.ModuleType("hubconf")
            original_hubconf_naf = lambda *args, **kwargs: None
            hubconf.naf = original_hubconf_naf

            def run_inference(*, output_path, low_vram, **kwargs):
                del kwargs
                calls.append(low_vram)
                extractor = inference.build_image_cond_model({})
                extractor._load_naf()
                Path(output_path).write_bytes(b"glb")

            inference.run_inference = run_inference
            source = {"sources": {"naf": {"kind": "local", "value": str(checkpoint)}}}
            with patch.dict(sys.modules, {"inference": inference, "hubconf": hubconf}), \
                 patch.object(runtime, "_preflight_runtime", return_value=(None, source)), \
                 patch.object(runtime, "_prepare_runtime_compat"), \
                 patch.object(runtime, "_install_windows_native_module_aliases"), \
                 patch.object(runtime, "_install_natten_fallback"), \
                 patch.object(runtime, "_silence_flex_gemm_autotuners"), \
                 patch.object(naf_checkpoint, "NAF_SIZE", checkpoint.stat().st_size), \
                 patch.object(naf_checkpoint, "NAF_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()), \
                 patch.object(multiview, "_load_local_naf", side_effect=lambda path, device: calls.append((path, device)) or types.SimpleNamespace(eval=lambda: None, requires_grad_=lambda enabled: None)):
                for mode in ("low_vram", "standard"):
                    result = runtime.run_job({
                        "input_image": str(image), "output_dir": str(output),
                        "readiness": {"generation_allowed": True},
                        "params": {"low_vram": mode, "seed": 1},
                    })
                    self.assertEqual(result["status"], "completed", result)
                    self.assertIs(inference.build_image_cond_model, original)
                    self.assertIs(hubconf.naf, original_hubconf_naf)
            self.assertEqual(calls[0], True)
            self.assertEqual(calls[2], False)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(value == (checkpoint, "cuda") for value in (calls[1], calls[3])))


if __name__ == "__main__":
    unittest.main()
