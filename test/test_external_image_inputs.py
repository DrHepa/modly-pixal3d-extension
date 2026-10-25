"""CPU-only coverage of the stock Modly bytes + native image-path contract."""

import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from generator import Pixal3DGenerator
from pixal3d_extension.multiview_images import run_multiview_from_images, validate_ordered_images
from pixal3d_extension.scene_prepare import run_scene_from_images


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(stream, format="PNG")
    return stream.getvalue()


class ExternalImageInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.output = self.workspace / "outputs"
        self.external = self.root / "selected images ñ"
        self.external.mkdir()
        self.source = self.external / "outside image.png"
        self.source.write_bytes(png("green"))
        self.original = self.source.read_bytes()

    def test_primary_bytes_and_external_auxiliary_are_admitted_before_assets(self):
        generator = Pixal3DGenerator(self.root / "models")
        generator.workspace_dir = self.workspace
        generator.MODEL_NODE_ID = "generate-mv"
        generator.shared_model_dirs = {
            "pixal3d-base": str(self.root / "base"),
            "pixal3d-mv": str(self.root / "mv"),
            "da3-base": str(self.root / "da3"),
        }
        expected = self.output / "result.glb"
        with patch.object(generator, "_prepare_generation_assets", return_value=self.root / "naf"), \
             patch("pixal3d_extension.multiview_images.run_multiview_from_images", return_value=expected) as run:
            self.assertEqual(generator.generate(png("red"), {"extra_image_paths": [str(self.source)]}), expected)
        self.assertEqual(run.call_args.kwargs["extra_image_paths"], [str(self.source)])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse(self.output.exists())

    def test_mixed_absolute_external_and_workspace_relative_images_keep_order(self):
        inside = self.workspace / "inside.png"
        inside.write_bytes(png("blue"))
        images = validate_ordered_images(
            png("red"), [str(self.source), None, "inside.png"], self.workspace,
        )
        self.assertEqual([image.data for image in images], [png("red"), self.original, inside.read_bytes()])

    def test_windows_open_error_keeps_regular_file_rejection_and_original_cause(self):
        """Exercise error translation on any host, not a simulated Win32 read."""
        from pixal3d_extension import multiview_images

        denied = PermissionError("directory cannot be opened as an image file")
        with patch.object(multiview_images, "os", SimpleNamespace(name="nt")), \
             patch.object(multiview_images, "_windows_stable_read", side_effect=denied) as reader, \
             self.assertRaisesRegex(ValueError, "regular.*file") as raised:
            multiview_images._workspace_file(str(self.workspace), self.workspace, "view")
        self.assertIs(raised.exception.__cause__, denied)
        reader.assert_called_once_with(self.workspace, self.workspace, "view")

    def test_unsafe_inputs_remain_rejected_without_outputs(self):
        bad = self.external / "not-image.png"
        bad.write_bytes(b"not an image")
        truncated = self.external / "truncated.png"
        truncated.write_bytes(self.original[:40])
        relative_escape = os.path.relpath(self.source, self.workspace)
        cases = [str(self.external), relative_escape, str(bad), str(truncated),
                 str(self.source) + "\x00", str(self.external / "missing.png")]
        if os.name != "nt":
            cases.extend([r"C:\outside\view.png", r"\\server\share\view.png", "/dev/null"])
        for path in cases:
            with self.subTest(path=path), self.assertRaises((ValueError, TypeError)):
                validate_ordered_images(png("red"), [path], self.workspace)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ordered_images(png("red"), [str(self.source), str(self.source)], self.workspace)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ordered_images(self.original, [str(self.source)], self.workspace)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    @unittest.skipIf(os.name == "nt", "Windows reparse-point coverage uses real junctions")
    def test_external_symlinks_directory_aliases_and_fifo_are_rejected(self):
        direct = self.external / "alias.png"
        direct.symlink_to(self.source)
        directory = self.root / "directory alias"
        directory.symlink_to(self.external, target_is_directory=True)
        fifo = self.external / "pipe.png"
        os.mkfifo(fifo)
        for path in (direct, directory / self.source.name, fifo):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_ordered_images(png("red"), [str(path)], self.workspace)

    @unittest.skipIf(os.name == "nt", "Windows uses locked-handle race coverage")
    def test_symlink_swap_before_open_cannot_change_selected_bytes(self):
        original_open = os.open
        replacement = self.external / "replacement.png"
        replacement.write_bytes(png("blue"))
        saved = self.external / "saved.png"

        def swapped_open(path, flags, *args, **kwargs):
            if path == self.source.name:
                self.source.rename(saved)
                self.source.symlink_to(replacement)
            return original_open(path, flags, *args, **kwargs)

        with patch("pixal3d_extension.multiview_images.os.open", side_effect=swapped_open), \
             self.assertRaisesRegex(ValueError, "non-symlink"):
            validate_ordered_images(png("red"), [str(self.source)], self.workspace)
        self.assertEqual(saved.read_bytes(), self.original)
        self.assertFalse(self.output.exists())

    @unittest.skipIf(os.name == "nt", "Windows file handles deny concurrent writers")
    def test_in_place_mutation_during_snapshot_is_rejected(self):
        original_read = os.read
        mutated = False

        def changed_read(descriptor, count):
            nonlocal mutated
            data = original_read(descriptor, count)
            if data and not mutated:
                mutated = True
                self.source.write_bytes(png("blue") + b"changed")
            return data

        with patch("pixal3d_extension.multiview_images.os.read", side_effect=changed_read), \
             self.assertRaisesRegex(ValueError, "changed during image snapshot"):
            validate_ordered_images(png("red"), [str(self.source)], self.workspace)
        self.assertFalse(self.output.exists())

    def test_external_files_still_obey_format_dimensions_and_byte_bounds(self):
        for size, format_name, message in (((17, 12), "PNG", "matching dimensions"), ((16, 12), "GIF", "supported image")):
            Image.new("RGB", size, "blue").save(self.source, format=format_name)
            with self.subTest(format=format_name, size=size), self.assertRaisesRegex(ValueError, message):
                validate_ordered_images(png("red"), [str(self.source)], self.workspace)
        self.source.write_bytes(self.original)
        for bound, value in (("MAX_IMAGE_BYTES", len(self.original) - 1), ("MAX_IMAGE_PIXELS", 191)):
            with patch("pixal3d_extension.multiview_images." + bound, value), self.assertRaises(ValueError):
                validate_ordered_images(png("red"), [str(self.source)], self.workspace)
        self.assertFalse(self.output.exists())

    def test_corrupt_jpeg_is_rejected_before_generation_assets(self):
        stream = io.BytesIO()
        Image.new("RGB", (16, 12), "green").save(stream, format="JPEG")
        self.source.write_bytes(stream.getvalue()[:-15])
        generator = Pixal3DGenerator(self.root / "models")
        generator.workspace_dir = self.workspace
        generator.MODEL_NODE_ID = "generate-mv"
        with patch.object(generator, "_prepare_generation_assets") as assets, self.assertRaisesRegex(ValueError, "supported image"):
            generator.generate(png("red"), {"extra_image_paths": [str(self.source)]})
        assets.assert_not_called()

    def test_pre_cancel_leaves_external_original_and_workspace_untouched(self):
        event = threading.Event()
        event.set()
        for adapter in ("mv", "scene"):
            with self.subTest(adapter=adapter), self.assertRaisesRegex(RuntimeError, "cancelled"):
                self._run_adapter(adapter, event, lambda **_kwargs: self.fail("cancelled job reached worker"))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def _run_adapter(self, adapter, event, worker):
        if adapter == "mv":
            with patch("pixal3d_extension.multiview.run_multiview", side_effect=worker):
                return run_multiview_from_images(
                    primary_image_bytes=png("red"), extra_image_paths=[None, str(self.source)],
                    workspace_dir=self.workspace, output_dir=self.output,
                    da3_root=None, mv_root=self.root, base_root=self.root, naf_path=self.root / "naf",
                    params={"view_layout": "declared_roles", "image_role": "front", "image_3_role": "back"},
                    cancel_event=event,
                )
        with patch("pixal3d_extension.scene_prepare._run_scene_from_capture", side_effect=worker):
            return run_scene_from_images(
                primary_image_bytes=png("red"), extra_image_paths=[None, str(self.source)],
                workspace_dir=self.workspace, output_dir=self.output,
                sam_root=self.root, da3_root=self.root, params={}, cancel_event=event,
            )

    def test_both_adapters_stage_private_pngs_and_clean_on_success_failure_and_cancel(self):
        for adapter in ("mv", "scene"):
            for outcome in ("success", "failure", "cancel"):
                with self.subTest(adapter=adapter, outcome=outcome):
                    event = threading.Event()
                    observed = {}

                    def worker(**kwargs):
                        manifest = Path(kwargs.get("scene_manifest_path") or kwargs["capture_manifest"])
                        observed["staging"] = manifest.parent
                        self.assertTrue(manifest.parent.is_relative_to(self.workspace))
                        self.assertFalse(manifest.parent.is_relative_to(self.output))
                        with Image.open(manifest.parent / "0001.png") as image:
                            self.assertEqual(image.getpixel((0, 0)), (0, 128, 0))
                        if adapter == "scene":
                            self.assertEqual(json.loads(manifest.read_text())["provenance"]["sourcePorts"], [1, 3])
                        if outcome != "success":
                            if outcome == "cancel":
                                event.set()
                            raise RuntimeError("cancelled" if outcome == "cancel" else "worker failed")
                        result = self.output / ("result.glb" if adapter == "mv" else "scene-manifest.json")
                        result.write_bytes(b"final artifact")
                        return result

                    if outcome == "success":
                        result = self._run_adapter(adapter, event, worker)
                        self.assertEqual(set(self.output.iterdir()), {result})
                        result.unlink()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "cancelled|worker failed"):
                            self._run_adapter(adapter, event, worker)
                        self.assertEqual(list(self.output.iterdir()), [])
                    self.assertFalse(observed["staging"].exists())
                    self.assertEqual(self.source.read_bytes(), self.original)
                    self.assertEqual(set(self.external.iterdir()), {self.source})

    def test_external_inputs_do_not_authorize_external_outputs(self):
        with self.assertRaisesRegex(ValueError, "workspace"):
            run_multiview_from_images(
                primary_image_bytes=png("red"), extra_image_paths=[str(self.source)],
                workspace_dir=self.workspace, output_dir=self.external / "output",
                da3_root=None, mv_root=self.root, base_root=self.root, naf_path=self.root / "naf",
                params={"view_layout": "declared_roles", "image_role": "front", "image_2_role": "back"},
            )
        self.assertFalse((self.external / "output").exists())


if __name__ == "__main__":
    unittest.main()
