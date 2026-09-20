import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pixal3d_extension.scene_prepare_contract import normalize_annotated_scene


class NormalizeAnnotatedSceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name) / "workspace"
        self.source = self.workspace / "Scenes" / "source"
        self.source.mkdir(parents=True)
        Image.new("RGB", (64, 64), "white").save(self.source / "000.png")
        mask = self.source / "masks" / "obj01" / "0000.png"
        mask.parent.mkdir(parents=True)
        Image.new("L", (64, 64), 255).save(mask)
        transforms = {
            "fl_x": 70, "fl_y": 70, "cx": 32, "cy": 32, "w": 64, "h": 64,
            "frames": [{"file_path": "000.png", "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]}],
            "instances": [{"pass_index": 1, "label": "chair", "aabb_world": [[-.5, -.5, -.5], [.5, .5, .5]]}],
            "provenance": {"scale": {"mode": "unknown"}},
        }
        (self.source / "transforms.json").write_text(json.dumps(transforms))
        self.manifest = self.source / "scene-manifest.json"
        self.manifest.write_text(json.dumps({"schema": "modly.scene-manifest.v1", "sceneRoot": ".", "assets": [{"path": "transforms.json"}]}))

    def tearDown(self):
        self.temp.cleanup()

    def test_normalizes_into_fresh_workspace_scene_without_changing_source(self):
        before = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        output = normalize_annotated_scene(self.manifest, self.workspace, self.workspace / "Workflows")
        self.assertTrue(output.is_file())
        self.assertTrue(output.is_relative_to(self.workspace))
        normalized = json.loads((output.parent / "transforms.json").read_text())
        self.assertEqual(normalized["provenance"]["scale"]["mode"], "unknown")
        self.assertEqual(normalized["provenance"]["normalizer"], "pixal3d-clean-room-v1")
        self.assertEqual(before, {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()})

    def test_rejects_symlinked_scene_content(self):
        outside = Path(self.temp.name) / "outside.png"
        outside.write_bytes((self.source / "000.png").read_bytes())
        (self.source / "000.png").unlink()
        (self.source / "000.png").symlink_to(outside)
        with self.assertRaises(ValueError):
            normalize_annotated_scene(self.manifest, self.workspace, self.workspace / "Workflows")

    def test_rejects_unsafe_output_parents_before_filesystem_mutation(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = self.workspace / "linked-output"
        link.symlink_to(outside, target_is_directory=True)
        cases = (
            self.workspace,
            Path("/"),
            outside / "new-output",
            link / "new-output",
            self.workspace / "Workflows" / ".." / ".." / "escape",
            self.source / "nested-output",
        )
        before = {path.relative_to(self.workspace) for path in self.workspace.rglob("*")}
        for output in cases:
            with self.subTest(output=output), self.assertRaises(ValueError):
                normalize_annotated_scene(self.manifest, self.workspace, output)
        after = {path.relative_to(self.workspace) for path in self.workspace.rglob("*")}
        self.assertEqual(before, after)
        self.assertFalse((outside / "new-output").exists())


if __name__ == "__main__":
    unittest.main()
