import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from PIL import Image

from pixal3d_extension.worldsculpt_contract import (
    prepare_case_root, validate_adapters, validate_crops, validate_output, validate_scene,
)


class WorldSculptContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scene = self.root / "scene"
        self.scene.mkdir()
        Image.new("RGB", (64, 64), "white").save(self.scene / "000.png")
        self.mask = self.scene / "masks" / "obj01" / "0000.png"
        self.mask.parent.mkdir(parents=True)
        Image.new("L", (64, 64), 255).save(self.mask)
        self.meta = {
            "fl_x": 70, "fl_y": 70, "cx": 32, "cy": 32, "w": 64, "h": 64,
            "frames": [{"file_path": "000.png", "transform_matrix":
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]}],
            "instances": [{"pass_index": 1, "aabb_world": [[-.5, -.5, -.5], [.5, .5, .5]]}],
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_scene(self):
        (self.scene / "transforms.json").write_text(json.dumps(self.meta))

    def test_eligible_metric_instance(self):
        self.write_scene()
        self.assertEqual(validate_scene(self.scene), ("obj01",))

    def test_rejects_under_500_mask(self):
        image = Image.new("L", (64, 64), 0)
        image.paste(255, (0, 0, 20, 20))
        image.save(self.mask)
        self.write_scene()
        with self.assertRaisesRegex(ValueError, "no crop-eligible"):
            validate_scene(self.scene)

    def test_rejects_unprojectable_crop(self):
        self.meta["frames"][0]["transform_matrix"][2][3] = -3
        self.write_scene()
        with self.assertRaisesRegex(ValueError, "no crop-eligible"):
            validate_scene(self.scene)

    def test_rejects_unsafe_scene_inputs(self):
        self.write_scene()
        outside = self.root / "outside.png"
        Image.new("L", (64, 64), 255).save(outside)
        self.mask.unlink()
        self.mask.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "local regular file"):
            validate_scene(self.scene)

    def test_rejects_later_mask_escape_after_an_eligible_first_frame(self):
        Image.new("RGB", (64, 64), "white").save(self.scene / "001.png")
        self.meta["frames"].append({
            "file_path": "001.png",
            "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]],
        })
        outside = self.root / "outside-mask.png"
        Image.new("L", (64, 64), 255).save(outside)
        later = self.mask.parent / "0001.png"
        later.symlink_to(outside)
        self.write_scene()
        with self.assertRaisesRegex(ValueError, "local regular file"):
            validate_scene(self.scene)
        self.mask.unlink()
        Image.new("L", (64, 64), 255).save(self.mask)
        metadata = self.scene / "transforms.json"
        metadata.rename(self.root / "outside.json")
        metadata.symlink_to(self.root / "outside.json")
        with self.assertRaisesRegex(ValueError, "local regular file"):
            validate_scene(self.scene)

    def test_rejects_singular_and_scaled_camera(self):
        self.meta["frames"][0]["transform_matrix"][0][0] = 0
        self.write_scene()
        with self.assertRaisesRegex(ValueError, "singular"):
            validate_scene(self.scene)
        self.meta["frames"][0]["transform_matrix"][0][0] = 2
        self.write_scene()
        with self.assertRaisesRegex(ValueError, "rigid"):
            validate_scene(self.scene)

    def test_adapters_must_be_distinct_exact_steps(self):
        root = self.root / "adapters"
        names = ("ss_ft64_mv_lora_ibr_texverse", "shape_ft1024_mv_lora_ibr_texverse_fixedmem05")
        for name in names:
            stage = root / name
            (stage / "ckpts").mkdir(parents=True)
            (stage / "config.json").write_text('{"models": {}, "trainer": {}}')
            (stage / "ckpts" / "denoiser_step0015000.pt").write_bytes(b"weights")
            (stage / "ckpts" / "mv_aggregator_step0015000.pt").write_bytes(b"weights")
        validate_adapters(root)
        stage = root / names[1]
        for child in (stage / "ckpts").iterdir():
            child.unlink()
        (stage / "ckpts").rmdir()
        (stage / "config.json").unlink()
        stage.rmdir()
        stage.symlink_to(root / names[0], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "stage"):
            validate_adapters(root)

    def make_output(self, names=("obj01",)):
        case = prepare_case_root(self.root / "output")
        scene = trimesh.Scene()
        for name in names:
            mesh = trimesh.creation.box()
            scene.add_geometry(mesh, geom_name=name)
            folder = case / "_recon" / name
            folder.mkdir(parents=True)
            torch.save({"vertices": torch.tensor(mesh.vertices, dtype=torch.float32),
                        "faces": torch.tensor(mesh.faces, dtype=torch.int64),
                        "T_canon_to_metric": np.eye(4, dtype=np.float64),
                        "mode": "offcenter_mv_geom"}, folder / "mesh.pt")
        (case / "_scene").mkdir()
        scene.export(case / "_scene" / "scene.glb")
        return case

    def test_valid_real_output_and_omitted_instance(self):
        case = self.make_output()
        self.assertEqual(validate_output(case, ("obj01",)), case / "_scene" / "scene.glb")
        second = case / "_recon" / "obj02"
        second.mkdir()
        (second / "mesh.pt").write_bytes((case / "_recon" / "obj01" / "mesh.pt").read_bytes())
        with self.assertRaisesRegex(ValueError, "reconstruction set"):
            validate_output(case, ("obj01",))
        with self.assertRaisesRegex(ValueError, "omits obj02"):
            validate_output(case, ("obj01", "obj02"))

    def test_rejects_unexpected_scene_geometry(self):
        case = self.make_output()
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(), geom_name="obj01")
        scene.add_geometry(trimesh.creation.box(), geom_name="obj99")
        scene.export(case / "_scene" / "scene.glb")
        with self.assertRaisesRegex(ValueError, "unexpected geometry"):
            validate_output(case, ("obj01",))

    def test_rejects_invalid_mesh_and_glb_symlink(self):
        case = self.make_output()
        mesh = case / "_recon" / "obj01" / "mesh.pt"
        mesh.write_bytes(b"not torch")
        with self.assertRaisesRegex(ValueError, "invalid obj01 mesh.pt"):
            validate_output(case, ("obj01",))
        mesh.unlink()
        outside = self.root / "outside.pt"
        outside.write_bytes(b"not torch")
        mesh.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "local regular file"):
            validate_output(case, ("obj01",))
        mesh.unlink()
        torch.save({"vertices": torch.tensor(trimesh.creation.box().vertices, dtype=torch.float32),
                    "faces": torch.tensor(trimesh.creation.box().faces, dtype=torch.int64),
                    "T_canon_to_metric": np.eye(4, dtype=np.float64), "mode": "offcenter_mv_geom"}, mesh)
        glb = case / "_scene" / "scene.glb"
        glb.unlink()
        glb.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "local regular file"):
            validate_output(case, ("obj01",))

    def test_actual_crop_gate(self):
        case = prepare_case_root(self.root / "output")
        folder = case / "_crops" / "obj01"
        folder.mkdir(parents=True)
        image = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        image.save(folder / "0000.png")
        identity = np.eye(4).tolist()
        crop_meta = {"pass_index": 1, "R_box": np.eye(3).tolist(),
                     "anchor_full_idx": 0, "frames": [{
                         "file_path": "0000.png", "full_frame_idx": 0,
                         "crop_size": 32, "image_size_px": 32, "is_anchor": True,
                         "transform_matrix": identity,
                         "K_image_pix": [[70, 0, 16], [0, 70, 16], [0, 0, 1]],
                     }]}
        (folder / "transforms.json").write_text(json.dumps(crop_meta))
        validate_crops(case, ("obj01",))
        (case / "_crops" / "obj02").mkdir()
        with self.assertRaisesRegex(ValueError, "crop set"):
            validate_crops(case, ("obj01",))
        (case / "_crops" / "obj02").rmdir()
        crop_meta["frames"] = []
        (folder / "transforms.json").write_text(json.dumps(crop_meta))
        with self.assertRaisesRegex(ValueError, "no usable crop"):
            validate_crops(case, ("obj01",))

    def test_rejects_wrong_crop_size_and_duplicate_indices(self):
        case = prepare_case_root(self.root / "output")
        folder = case / "_crops" / "obj01"
        folder.mkdir(parents=True)
        Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(folder / "0000.png")
        frame = {"file_path": "0000.png", "full_frame_idx": 0, "crop_size": 32,
                 "image_size_px": 64, "is_anchor": True,
                 "transform_matrix": np.eye(4).tolist(), "K_image_pix": np.eye(3).tolist()}
        metadata = {"pass_index": 1, "R_box": np.eye(3).tolist(),
                    "anchor_full_idx": 0, "frames": [frame]}
        path = folder / "transforms.json"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "empty RGBA crop"):
            validate_crops(case, ("obj01",))
        frame["image_size_px"] = 32
        metadata["frames"].append(dict(frame, is_anchor=False))
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "duplicate crop frame index"):
            validate_crops(case, ("obj01",))

    def test_rejects_unexpected_pickle_globals_and_empty_named_geometry(self):
        case = self.make_output()
        mesh = case / "_recon" / "obj01" / "mesh.pt"
        torch.save({"mode": "offcenter_mv_geom", "T_canon_to_metric": np.eye(4),
                    "surprise": Path("unexpected")}, mesh)
        with self.assertRaisesRegex(ValueError, "invalid obj01 mesh.pt"):
            validate_output(case, ("obj01",))
        valid = trimesh.creation.box()
        torch.save({"vertices": torch.tensor(valid.vertices, dtype=torch.float32),
                    "faces": torch.tensor(valid.faces, dtype=torch.int64),
                    "T_canon_to_metric": np.eye(4, dtype=np.float64),
                    "mode": "offcenter_mv_geom"}, mesh)
        empty = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[], process=False)
        with patch("trimesh.load", return_value=type("SceneStub", (), {"geometry": {"obj01": empty}})()):
            with self.assertRaisesRegex(ValueError, "invalid geometry"):
                validate_output(case, ("obj01",))
        nonfinite = trimesh.creation.box()
        nonfinite.vertices[0, 0] = np.nan
        with patch("trimesh.load", return_value=type("SceneStub", (), {"geometry": {"obj01": nonfinite}})()):
            with self.assertRaisesRegex(ValueError, "invalid geometry"):
                validate_output(case, ("obj01",))

    def test_requires_private_fresh_root(self):
        case = self.root / "unmarked"
        case.mkdir()
        with self.assertRaisesRegex(ValueError, "marker"):
            validate_output(case, ("obj01",))
        with self.assertRaises(FileExistsError):
            prepare_case_root(case)


if __name__ == "__main__":
    unittest.main()
