import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generator import Pixal3DGenerator, SCENE_NORMALIZE_NODE, SCENE_VIDEO_NODE


class GenerateArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.output = self.workspace / "Workflows"
        self.video = self.workspace / "clips" / "input.mp4"
        self.scene_manifest = self.workspace / "scenes" / "scene.json"
        self.scene_dir = self.workspace / "scenes" / "scene"
        for path in (self.output, self.video.parent, self.scene_manifest.parent, self.scene_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.video.write_bytes(b"video")
        self.scene_manifest.write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _generator(self, node_id):
        with patch("pixal3d_extension.multiview.configure_mv_hf_cache"):
            generator = Pixal3DGenerator(self.root / "models" / "pixal3d" / node_id, self.workspace)
        generator.MODEL_NODE_ID = node_id
        generator.outputs_dir = self.output
        generator.shared_model_dirs = {
            "sam3": self.root,
            "da3-base": self.root,
            "pixal3d-base": self.root,
            "worldsculpt-adapters": self.root,
        }
        return generator

    def test_upstream_video_artifact_dispatches_to_scene_video_with_canonical_envelope(self):
        generator = self._generator(SCENE_VIDEO_NODE)
        progress = object()
        cancel = object()
        expected = self.output / "video.scene"

        with patch("pixal3d_extension.scene_prepare.run_scene_from_video", return_value=expected) as run:
            result = generator.generate_artifact(
                "video",
                self.workspace / "clips" / ".." / "clips" / "input.mp4",
                {"max_frames": 4},
                progress_cb=progress,
                cancel_event=cancel,
            )

        self.assertEqual(result, expected)
        self.assertEqual(run.call_args.kwargs["video_input"], {"kind": "video", "path": str(self.video.resolve())})
        self.assertEqual(run.call_args.kwargs["params"], {"max_frames": 4})
        self.assertIs(run.call_args.kwargs["progress_cb"], progress)
        self.assertIs(run.call_args.kwargs["cancel_event"], cancel)

    def test_upstream_scene_artifact_preserves_normalize_path_dispatch(self):
        generator = self._generator(SCENE_NORMALIZE_NODE)
        expected = self.output / "normalized.scene"
        injected = self.workspace / "scenes" / ".." / "scenes" / "scene.json"

        with patch("pixal3d_extension.scene_prepare_contract.normalize_annotated_scene", return_value=expected) as normalize:
            result = generator.generate_artifact(
                "scene",
                self.scene_manifest,
                {"scene_manifest_path": str(injected), "deduplicate_instances": "true"},
            )

        self.assertEqual(result, expected)
        normalize.assert_called_once_with(self.scene_manifest.resolve(), self.workspace, self.output)

    def test_upstream_scene_artifact_preserves_worldsculpt_scene_manifest_dispatch(self):
        generator = self._generator("worldsculpt")
        expected = self.output / "world.glb"
        injected = self.workspace / "scenes" / ".." / "scenes" / "scene.json"

        with patch.object(generator, "_prepare_generation_assets", return_value=self.root / "naf.pth"), \
             patch("pixal3d_extension.worldsculpt.resolve_scene_manifest", return_value=self.scene_dir) as resolve, \
             patch("pixal3d_extension.worldsculpt.run_worldsculpt", return_value=expected) as run:
            result = generator.generate_artifact(
                "scene",
                self.scene_manifest,
                {"scene_manifest_path": str(injected), "face_budget": 123},
            )

        self.assertEqual(result, expected)
        resolve.assert_any_call(str(self.scene_manifest.resolve()), self.workspace)
        self.assertEqual(run.call_args.kwargs["face_budget"], 123)

    def test_scene_artifact_rejects_mismatched_injected_scene_manifest_path(self):
        generator = self._generator("worldsculpt")
        forged = self.workspace / "scenes" / "forged.json"
        forged.write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "scene_manifest_path.*does not match"):
            generator.generate_artifact("scene", self.scene_manifest, {"scene_manifest_path": str(forged)})

    def test_generate_artifact_rejects_unsupported_kind_bad_path_and_reserved_params(self):
        generator = self._generator(SCENE_VIDEO_NODE)
        with self.assertRaisesRegex(ValueError, "Unsupported artifact kind"):
            generator.generate_artifact("mesh", self.video, {})
        with self.assertRaisesRegex(TypeError, "artifact path"):
            generator.generate_artifact("video", b"not-a-path", {})
        with self.assertRaisesRegex(ValueError, "reserved transport"):
            generator.generate_artifact("video", self.video, {"video_path": str(self.video)})

    def test_legacy_generate_scene_video_contract_remains_unchanged(self):
        generator = self._generator(SCENE_VIDEO_NODE)
        envelope = {"kind": "video", "path": "clips/input.mp4"}
        expected = self.output / "video.scene"
        with patch("pixal3d_extension.scene_prepare.run_scene_from_video", return_value=expected) as run:
            result = generator.generate(envelope, {})
        self.assertEqual(result, expected)
        self.assertIs(run.call_args.kwargs["video_input"], envelope)


if __name__ == "__main__":
    unittest.main()
