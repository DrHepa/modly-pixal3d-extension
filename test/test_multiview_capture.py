import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from generator import Pixal3DGenerator


FRONT = [[1, 0, 0, 0], [0, 0, -1, -3], [0, 1, 0, 0], [0, 0, 0, 1]]


def capture_fixture(root: Path, count: int = 2) -> tuple[Path, dict]:
    root.mkdir(parents=True)
    frames = []
    for index in range(count):
        image = root / f"image-{count - index}.png"
        Image.new("RGBA", (32, 24), (index * 50, 20, 30, 127)).save(image)
        frames.append({"index": index, "path": image.name, "byteSize": image.stat().st_size,
                       "width": 32, "height": 24})
    data = {
        "schema": "modly.capture-manifest.v1", "kind": "frames", "captureRoot": ".",
        "provenance": {"source": "test", "ordering": "manifest-index"}, "frames": frames,
        "multiview": {"cameraConvention": "blender-c2w", "meshScale": 1.0,
                      "cameras": [{"index": index, "transformMatrix": copy.deepcopy(FRONT),
                                   "cameraAngleX": 0.5} for index in range(count)]},
    }
    manifest = root / "capture-manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest, data


class MultiviewCaptureTests(unittest.TestCase):
    def test_node_exposes_ordered_images_while_internal_capture_adapter_remains_available(self):
        manifest = json.loads((Path(__file__).parents[1] / "manifest.json").read_text())
        nodes = {node["id"]: node for node in manifest["nodes"]}
        self.assertEqual((nodes["generate-mv"]["input"], nodes["generate-mv"]["output"]), ("image", "mesh"))
        self.assertEqual(nodes["generate-mv"]["inputs"], ["image"] * 4)
        self.assertEqual(nodes["generate-mv"]["weight_groups"], ["pixal3d-base", "pixal3d-mv", "da3-base"])
        self.assertEqual(nodes["worldsculpt"]["input"], "scene")

    def test_generator_rejects_non_image_and_incomplete_multi_image_inputs(self):
        gen = Pixal3DGenerator("/tmp/modly/models/pixal3d/generate-mv", "/tmp/modly/workspace")
        gen.MODEL_NODE_ID = "generate-mv"
        for incoming, params in (
            (b"", {}),
            (SimpleNamespace(kind="scene", path=Path("/tmp/scene.json")), {}),
            (b"not-an-image", {}),
        ):
            with self.subTest(incoming=incoming), self.assertRaisesRegex(ValueError, "primary image|extra_image_paths"):
                gen.generate(incoming, params)

    def test_ordered_images_are_staged_privately_with_alpha_and_real_cameras(self):
        from pixal3d_extension.multiview_capture import prepare_capture_views
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            manifest, data = capture_fixture(workspace / "capture")
            originals = {path: path.read_bytes() for path in manifest.parent.iterdir()}
            with prepare_capture_views(manifest, workspace, workspace / "out", 2) as staged:
                self.assertNotEqual(staged, manifest.parent)
                transforms = json.loads((staged / "transforms.json").read_text())
                self.assertEqual(transforms["mesh_scale"], 1.0)
                self.assertEqual([frame["file_path"] for frame in transforms["frames"]], ["0000.png", "0001.png"])
                self.assertEqual(transforms["frames"][0]["transform_matrix"], FRONT)
                with Image.open(staged / "0000.png") as first, Image.open(staged / "0001.png") as second:
                    self.assertEqual(first.getpixel((0, 0)), (0, 20, 30, 127))
                    self.assertEqual(second.getpixel((0, 0)), (50, 20, 30, 127))
            self.assertFalse(staged.exists())
            self.assertEqual({path: path.read_bytes() for path in manifest.parent.iterdir()}, originals)

    def test_uncalibrated_or_invalid_capture_is_rejected_before_output_creation(self):
        from pixal3d_extension.multiview_capture import prepare_capture_views
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            manifest, original = capture_fixture(workspace / "capture")
            mutations = [
                lambda data: data.pop("multiview"),
                lambda data: data["multiview"]["cameras"].pop(),
                lambda data: data["multiview"]["cameras"][1].update(cameraAngleX=float("nan")),
                lambda data: data["multiview"]["cameras"][0]["transformMatrix"][0].__setitem__(0, 2),
                lambda data: data["frames"][1].update(path="../outside.png"),
            ]
            for mutate in mutations:
                data = copy.deepcopy(original)
                mutate(data)
                manifest.write_text(json.dumps(data))
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    with prepare_capture_views(manifest, workspace, workspace / "out", 2):
                        self.fail("invalid capture was accepted")
                self.assertFalse((workspace / "out").exists())

    def test_grayscale_alpha_is_preserved_in_staged_views(self):
        from pixal3d_extension.multiview_capture import prepare_capture_views
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            manifest, data = capture_fixture(workspace / "capture", 1)
            source = manifest.parent / data["frames"][0]["path"]
            Image.new("LA", (32, 24), (80, 127)).save(source)
            data["frames"][0]["byteSize"] = source.stat().st_size
            manifest.write_text(json.dumps(data))
            with prepare_capture_views(manifest, workspace, workspace / "out", 1) as staged:
                with Image.open(staged / "0000.png") as image:
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.getpixel((0, 0)), (80, 80, 80, 127))

    def test_video_decode_order_and_metadata_are_checked_without_pose_estimation(self):
        # Other worker tests install lightweight cv2 stubs during discovery. Decode
        # with the real installed codec in an isolated interpreter, not that stub.
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'test'); "
             "from test_multiview_capture import MultiviewCaptureTests; "
             "MultiviewCaptureTests()._assert_video_decode_order_and_metadata()"],
            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _assert_video_decode_order_and_metadata(self):
        import cv2
        import numpy as np
        from pixal3d_extension.multiview_capture import prepare_capture_views
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            manifest, data = capture_fixture(workspace / "capture", 3)
            video = manifest.parent / "capture.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for color in (20, 90, 160):
                    writer.write(np.full((24, 32, 3), color, dtype=np.uint8))
            finally:
                writer.release()
            data.update(kind="video", video={"path": video.name, "byteSize": video.stat().st_size,
                                           "width": 32, "height": 24, "frameCount": 3})
            data.pop("frames")
            data["provenance"]["ordering"] = "decode-index"
            manifest.write_text(json.dumps(data))
            with prepare_capture_views(manifest, workspace, workspace / "out", 2) as staged:
                with Image.open(staged / "0000.png") as first, Image.open(staged / "0001.png") as second:
                    self.assertLess(first.getpixel((0, 0))[0], second.getpixel((0, 0))[0])
                self.assertEqual(len(json.loads((staged / "transforms.json").read_text())["frames"]), 2)
            data["video"]["frameCount"] = 4
            data["multiview"]["cameras"].append({**data["multiview"]["cameras"][0], "index": 3})
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "frame count"):
                with prepare_capture_views(manifest, workspace, workspace / "out", 2):
                    self.fail("incorrect frame count accepted")

    def test_capture_runtime_calls_mv_with_private_views_and_cleans_up_on_failure(self):
        from pixal3d_extension.multiview import run_multiview
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            manifest, _ = capture_fixture(workspace / "capture")
            calls = []

            def configuration(_mv, _base, target):
                target.write_text("{}")
                return target

            def inference(**kwargs):
                calls.append(kwargs)
                self.assertTrue((Path(kwargs["views_dir"]) / "transforms.json").is_file())
                Path(kwargs["output_path"]).write_bytes(b"mocked-glb")

            kwargs = dict(capture_manifest_path=manifest, workspace_dir=workspace,
                          mv_root=workspace / "unused-mv", base_root=workspace / "unused-base",
                          naf_path=workspace / "unused-naf", output_dir=workspace / "out",
                          params={"num_views": 2})
            with patch("pixal3d_extension.multiview.missing_mv_assets", return_value=[]), \
                 patch("pixal3d_extension.multiview.prepare_mv_pipeline_config", side_effect=configuration):
                result = run_multiview(**kwargs, inference_runner=inference)
                self.assertTrue(result.is_file())
                self.assertEqual(calls[0]["num_views"], 2)
                self.assertFalse(Path(calls[0]["views_dir"]).exists())

                def failure(**values):
                    calls.append(values)
                    raise RuntimeError("deliberate inference failure")

                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    run_multiview(**kwargs, inference_runner=failure)
                self.assertFalse(Path(calls[-1]["views_dir"]).exists())

    def test_mv_dynamic_modules_use_workspace_cache_when_home_cache_is_read_only(self):
        code = r'''
import os
import sys
from pathlib import Path

home = Path(sys.argv[1])
workspace = Path(sys.argv[2])
blocked = home / ".cache" / "huggingface" / "modules"
blocked.parent.mkdir(parents=True)
blocked.write_text("not a directory", encoding="utf-8")
blocked.parent.chmod(0o555)
os.environ["HOME"] = str(home)
for key in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"):
    os.environ.pop(key, None)

from pixal3d_extension.multiview import configure_mv_hf_cache
cache = configure_mv_hf_cache(workspace)
from transformers.dynamic_module_utils import HF_MODULES_CACHE, init_hf_modules
init_hf_modules()
assert Path(HF_MODULES_CACHE).resolve() == (cache / "modules").resolve()
assert Path(os.environ["HF_HOME"]).resolve() == cache.resolve()
assert Path(os.environ["HF_HUB_CACHE"]).resolve() == (cache / "hub").resolve()
assert Path(os.environ["TRANSFORMERS_CACHE"]).resolve() == (cache / "transformers").resolve()
assert Path(HF_MODULES_CACHE).is_dir()
assert blocked.is_file()
assert blocked.parent.stat().st_mode & 0o222 == 0
blocked.parent.chmod(0o755)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [sys.executable, "-c", code, str(root / "home"), str(root / "workspace")],
                cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_single_view_initializes_mv_cache_before_transformers_dynamic_modules(self):
        code = r'''
import os
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
for key in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"):
    os.environ.pop(key, None)

from generator import Pixal3DGenerator
generator = Pixal3DGenerator(workspace_dir=workspace)
generator.MODEL_NODE_ID = "generate"
import pixal3d.pipelines
from transformers.dynamic_module_utils import HF_MODULES_CACHE
from pixal3d_extension.multiview import configure_mv_hf_cache
cache = configure_mv_hf_cache(workspace)
assert Path(HF_MODULES_CACHE).resolve() == (cache / "modules").resolve()
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", code, str(Path(directory) / "workspace")],
                cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mv_cache_rejects_every_child_symlink_before_any_mutation(self):
        from pixal3d_extension.multiview import configure_mv_hf_cache

        environment = {
            "HF_HOME": "sentinel-home",
            "HF_HUB_CACHE": "sentinel-hub",
            "HF_MODULES_CACHE": "sentinel-modules",
            "TRANSFORMERS_CACHE": "sentinel-transformers",
            "XDG_CACHE_HOME": "sentinel-xdg",
        }
        for child in ("hub", "modules", "transformers", "xdg"):
            with self.subTest(child=child), tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, environment, clear=False
            ):
                root = Path(directory)
                workspace = root / "workspace"
                cache = workspace / ".pixal3d-runtime" / "huggingface"
                outside = root / "outside"
                cache.mkdir(parents=True)
                outside.mkdir()
                (cache / child).symlink_to(outside, target_is_directory=True)
                before = {path.relative_to(workspace) for path in workspace.rglob("*")}

                with self.assertRaisesRegex(ValueError, "symlink"):
                    configure_mv_hf_cache(workspace)

                after = {path.relative_to(workspace) for path in workspace.rglob("*")}
                self.assertEqual(after, before)
                self.assertEqual(list(outside.iterdir()), [])
                self.assertEqual({key: os.environ[key] for key in environment}, environment)

    def test_mv_cache_rejects_transformers_modules_descendant_symlink_before_mutation(self):
        from pixal3d_extension.multiview import configure_mv_hf_cache

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            modules = workspace / ".pixal3d-runtime" / "huggingface" / "modules"
            outside = root / "outside"
            modules.mkdir(parents=True)
            outside.mkdir()
            (modules / "transformers_modules").symlink_to(outside, target_is_directory=True)
            before = {path.relative_to(workspace) for path in workspace.rglob("*")}

            with self.assertRaisesRegex(ValueError, "descendant symlink"):
                configure_mv_hf_cache(workspace)

            self.assertEqual(
                {path.relative_to(workspace) for path in workspace.rglob("*")}, before
            )
            self.assertEqual(list(outside.iterdir()), [])

    def test_mv_cache_recursively_rejects_nested_symlinks_and_directory_path_files(self):
        from pixal3d_extension.multiview import configure_mv_hf_cache

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            cache = workspace / ".pixal3d-runtime" / "huggingface"
            outside = root / "outside"
            nested = cache / "hub" / "models--owner--model" / "snapshots"
            nested.mkdir(parents=True)
            outside.mkdir()
            (nested / "revision").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "descendant symlink"):
                configure_mv_hf_cache(workspace)
            self.assertEqual(list(outside.iterdir()), [])

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            modules = workspace / ".pixal3d-runtime" / "huggingface" / "modules"
            modules.mkdir(parents=True)
            (modules / "transformers_modules").write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a directory"):
                configure_mv_hf_cache(workspace)

    def test_mv_cache_custody_and_pre_cancel_do_not_escape_or_mutate_workspace(self):
        from pixal3d_extension.multiview import configure_mv_hf_cache, run_multiview
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            (workspace / ".pixal3d-runtime").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                configure_mv_hf_cache(workspace)
            self.assertEqual(list(outside.iterdir()), [])

            (workspace / ".pixal3d-runtime").unlink()
            cancel = threading.Event()
            cancel.set()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                run_multiview(
                    scene_manifest_path=workspace / "missing.json", workspace_dir=workspace,
                    mv_root=workspace / "mv", base_root=workspace / "base",
                    naf_path=workspace / "naf.pth", output_dir=workspace / "out",
                    params={}, cancel_event=cancel,
                )
            self.assertFalse((workspace / ".pixal3d-runtime").exists())
            self.assertFalse((workspace / "out").exists())

    def test_capture_path_and_output_custody_and_cancellation_are_fail_closed(self):
        from pixal3d_extension.multiview_capture import prepare_capture_views
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            workspace = parent / "workspace"
            manifest, data = capture_fixture(workspace / "capture")
            with self.assertRaises(ValueError):
                with prepare_capture_views(manifest, workspace, parent / "outside-output", 2):
                    self.fail("output escaped workspace")
            self.assertFalse((parent / "outside-output").exists())
            cancel = threading.Event()
            cancel.set()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with prepare_capture_views(manifest, workspace, workspace / "out", 2, cancel_event=cancel):
                    self.fail("cancelled staging ran")
            self.assertFalse((workspace / "out").exists())
            outside = parent / "outside.png"
            source = manifest.parent / data["frames"][1]["path"]
            outside.write_bytes(source.read_bytes())
            source.unlink()
            source.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                with prepare_capture_views(manifest, workspace, workspace / "out", 1):
                    self.fail("unselected escaped frame accepted")
            self.assertFalse((workspace / "out").exists())


if __name__ == "__main__":
    unittest.main()
