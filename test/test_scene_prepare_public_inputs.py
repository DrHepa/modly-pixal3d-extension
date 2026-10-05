import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from generator import Pixal3DGenerator, SCENE_IMAGES_NODE, SCENE_VIDEO_NODE
from pixal3d_extension.scene_prepare import parse_video_input, run_scene_from_images, run_scene_from_video
from pixal3d_extension.scene_video_input import stage_video_input


def _png(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG")
    return stream.getvalue()


class ScenePreparePublicInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.output = self.workspace / "Workflows"
        self.views = self.workspace / "views"
        self.views.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_image_input_preserves_order_and_null_holes_without_leaking_staging(self):
        second = self.views / "second.png"
        fourth = self.views / "fourth.png"
        second.write_bytes(_png("green"))
        fourth.write_bytes(_png("blue"))
        observed = {}

        def fake_capture(**kwargs):
            manifest = kwargs["capture_manifest"]
            observed["manifest"] = manifest.read_text(encoding="utf-8")
            observed["names"] = sorted(path.name for path in manifest.parent.glob("*.png"))
            result = self.output / "scene.json"
            result.write_text("{}", encoding="utf-8")
            return result

        with patch("pixal3d_extension.scene_prepare._run_scene_from_capture", side_effect=fake_capture):
            result = run_scene_from_images(
                primary_image_bytes=_png("red"),
                extra_image_paths=[str(second), None, str(fourth), None, None, None, None],
                workspace_dir=self.workspace,
                output_dir=self.output,
                sam_root=self.root,
                da3_root=self.root,
                params={},
            )
        self.assertEqual(result, self.output / "scene.json")
        self.assertEqual(observed["names"], ["0000.png", "0001.png", "0002.png"])
        self.assertEqual(json.loads(observed["manifest"])["provenance"]["sourcePorts"], [1, 2, 4])
        self.assertEqual(list(self.output.glob(".scene-images-*")), [])

    def test_images_require_two_views_and_accept_at_most_eight_ports(self):
        with self.assertRaisesRegex(ValueError, "at least two connected images"):
            run_scene_from_images(
                primary_image_bytes=_png("red"), extra_image_paths=[None] * 7,
                workspace_dir=self.workspace, output_dir=self.output,
                sam_root=self.root, da3_root=self.root, params={},
            )
        paths = []
        for index in range(8):
            path = self.views / f"{index}.png"
            path.write_bytes(_png((index, index, index)))
            paths.append(str(path))
        with self.assertRaisesRegex(ValueError, "at most eight"):
            run_scene_from_images(
                primary_image_bytes=_png("red"), extra_image_paths=paths,
                workspace_dir=self.workspace, output_dir=self.output,
                sam_root=self.root, da3_root=self.root, params={},
            )

    def test_video_transport_parser_isolated_and_typed(self):
        self.assertEqual(parse_video_input(SimpleNamespace(kind="video", path="clips/a.mp4")), Path("clips/a.mp4"))
        self.assertEqual(parse_video_input({"kind": "video", "path": "clips/a.mp4"}), Path("clips/a.mp4"))
        with self.assertRaisesRegex(ValueError, "kind=video"):
            parse_video_input({"kind": "image", "path": "clips/a.mp4"})
        with self.assertRaisesRegex(TypeError, "typed video envelope"):
            parse_video_input("clips/a.mp4")

    def test_video_custody_checks_workspace_extension_signature_and_cleanup(self):
        clips = self.workspace / "clips"
        clips.mkdir()
        source = clips / "input.mp4"
        source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"video")
        staging = self.output / "stage"
        staging.mkdir(parents=True)
        snapshot = stage_video_input({"kind": "video", "path": "clips/input.mp4"}, self.workspace, staging)
        self.assertEqual(snapshot.read_bytes(), source.read_bytes())
        bad = clips / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaisesRegex(ValueError, "content"):
            stage_video_input({"kind": "video", "path": str(bad)}, self.workspace, staging / "bad")
        outside = self.root / "outside.mp4"
        outside.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "workspace"):
            stage_video_input({"kind": "video", "path": str(outside)}, self.workspace, staging)

    def test_video_adapter_builds_private_decode_order_capture_and_cleans_it(self):
        clips = self.workspace / "clips"
        clips.mkdir()
        source = clips / "input.mp4"
        source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"video")
        observed = {}

        def fake_capture(**kwargs):
            observed.update(json.loads(kwargs["capture_manifest"].read_text(encoding="utf-8")))
            result = self.output / "video.scene"
            result.write_text("{}", encoding="utf-8")
            return result

        with patch("pixal3d_extension.scene_prepare.validate_scene_prepare_weights"), \
             patch("pixal3d_extension.scene_prepare.validate_runtime"), \
             patch("pixal3d_extension.scene_prepare._probe_video", return_value={"width": 640, "height": 360, "frameCount": 42}), \
             patch("pixal3d_extension.scene_prepare._run_scene_from_capture", side_effect=fake_capture):
            result = run_scene_from_video(
                video_input={"kind": "video", "path": "clips/input.mp4"},
                workspace_dir=self.workspace, output_dir=self.output,
                sam_root=self.root, da3_root=self.root,
                params={"frame_stride": 3, "max_frames": 8},
            )
        self.assertEqual(result, self.output / "video.scene")
        self.assertEqual(observed["kind"], "video")
        self.assertEqual(observed["provenance"]["ordering"], "decode-index")
        self.assertEqual(observed["video"]["frameCount"], 42)
        self.assertEqual(list(self.output.glob(".scene-video-*")), [])

    def test_generator_dispatches_native_inputs_and_rejects_reserved_transport(self):
        generator = Pixal3DGenerator(self.root)
        generator.workspace_dir = self.workspace
        generator.shared_model_dirs = {"sam3": str(self.root), "da3-base": str(self.root)}
        generator.outputs_dir = self.output
        generator.MODEL_NODE_ID = SCENE_IMAGES_NODE
        with patch("pixal3d_extension.scene_prepare.run_scene_from_images", return_value=self.output / "images.scene") as images:
            result = generator.generate(_png("red"), {"extra_image_paths": [str(self.views / "second.png")]})
        self.assertEqual(result, self.output / "images.scene")
        self.assertEqual(images.call_args.kwargs["extra_image_paths"], [str(self.views / "second.png")])
        generator.MODEL_NODE_ID = SCENE_VIDEO_NODE
        envelope = {"kind": "video", "path": "clips/a.mp4"}
        with patch("pixal3d_extension.scene_prepare.run_scene_from_video", return_value=self.output / "video.scene") as video:
            result = generator.generate(envelope, {})
        self.assertEqual(result, self.output / "video.scene")
        self.assertIs(video.call_args.kwargs["video_input"], envelope)
        with self.assertRaisesRegex(ValueError, "reserved transport"):
            generator.generate(envelope, {"extra_image_paths": []})


if __name__ == "__main__":
    unittest.main()
