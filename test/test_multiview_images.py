import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from generator import Pixal3DGenerator


def png_bytes(color=(30, 40, 50, 255)) -> bytes:
    import io

    stream = io.BytesIO()
    Image.new("RGBA", (32, 24), color).save(stream, format="PNG")
    return stream.getvalue()


class MultiviewImageContractTests(unittest.TestCase):
    def test_windows_extended_paths_and_identity_containment_are_structural(self):
        import pixal3d_extension.multiview_images as multiview_images

        self.assertEqual(
            multiview_images._windows_extended_path(r"C:\Users\runneradmin\workspace\view.png"),
            r"\\?\C:\Users\runneradmin\workspace\view.png",
        )
        self.assertEqual(
            multiview_images._windows_extended_path(r"\\server\share\workspace\view.png"),
            r"\\?\UNC\server\share\workspace\view.png",
        )
        self.assertEqual(
            multiview_images._windows_extended_path(r"\\?\C:\deep\view.png"),
            r"\\?\C:\deep\view.png",
        )
        self.assertEqual(
            multiview_images._windows_extended_path(r"\\?\UNC\server\share\view.png"),
            r"\\?\UNC\server\share\view.png",
        )
        for unsafe in (r"relative\view.png", r"\\.\PhysicalDrive0", r"\\?\GLOBALROOT\Device\HarddiskVolume1"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                multiview_images._windows_extended_path(unsafe)

        identity = multiview_images._WindowsFileIdentity
        workspace = identity(10, bytes.fromhex("01" * 16))
        candidate = identity(10, bytes.fromhex("02" * 16))
        case_only_distinct = identity(10, bytes.fromhex("03" * 16))
        other_volume_or_share = identity(11, bytes.fromhex("02" * 16))
        unavailable = identity(10, bytes(16))
        no_unique_id = identity(10, bytes([0xFF]) * 16)
        self.assertTrue(multiview_images._windows_file_identity_is_valid(workspace))
        self.assertFalse(multiview_images._windows_file_identity_is_valid(unavailable))
        self.assertFalse(multiview_images._windows_file_identity_is_valid(no_unique_id))
        self.assertTrue(multiview_images._windows_identity_chain_contains(
            workspace, candidate, [case_only_distinct, workspace],
        ))
        self.assertFalse(multiview_images._windows_identity_chain_contains(
            workspace, candidate, [case_only_distinct],
        ))
        self.assertFalse(multiview_images._windows_identity_chain_contains(
            workspace, other_volume_or_share, [workspace],
        ))
        self.assertFalse(multiview_images._windows_identity_chain_contains(
            unavailable, candidate, [unavailable],
        ))
        self.assertFalse(multiview_images._windows_identity_chain_contains(
            no_unique_id, candidate, [no_unique_id],
        ))
        self.assertFalse(multiview_images._windows_identity_chain_contains(
            workspace, unavailable, [workspace],
        ))
        self.assertFalse(multiview_images._windows_identity_chain_contains(
            workspace, no_unique_id, [workspace],
        ))

    def test_manifest_uses_upstream_ordered_image_ports_and_da3(self):
        manifest = json.loads((Path(__file__).parents[1] / "manifest.json").read_text())
        node = next(item for item in manifest["nodes"] if item["id"] == "generate-mv")
        self.assertEqual(node["input"], "image")
        self.assertEqual(node["inputs"], ["image", "image", "image", "image"])
        self.assertEqual(
            node["input_contract"],
            [
                {
                    "name": "image" if index == 1 else f"image_{index}",
                    "label": "Primary view" if index == 1 else f"View {index}",
                    "type": "image",
                    "required": index == 1,
                }
                for index in range(1, 5)
            ],
        )
        self.assertNotIn("input_labels", node)
        self.assertEqual(node["output"], "mesh")
        self.assertEqual(node["weight_groups"], ["pixal3d-base", "pixal3d-mv", "da3-base"])
        params = {item["id"]: item for item in node["params_schema"]}
        self.assertNotIn("num_views", params)
        self.assertEqual(params["texture_size"]["default"], 1024)
        self.assertEqual([item["value"] for item in params["texture_size"]["options"]], [1024, 2048])
        self.assertEqual(params["view_layout"]["default"], "auto")
        self.assertEqual(params["view_fov"]["default"], 20.0)
        self.assertEqual(params["view_fov"]["show_if"], {"view_layout": "declared_roles"})
        for role_id in ("image_role", "image_2_role", "image_3_role", "image_4_role"):
            self.assertEqual(params[role_id]["show_if"], {"view_layout": "declared_roles"})

    def test_generator_preserves_host_connection_order_and_invokes_da3_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            extras = workspace / "Workflows" / "inputs"
            extras.mkdir(parents=True)
            second = extras / "second.png"
            third = extras / "third.webp"
            Image.new("RGB", (32, 24), (2, 0, 0)).save(second)
            Image.new("RGB", (32, 24), (3, 0, 0)).save(third)
            gen = Pixal3DGenerator(workspace / "models")
            gen.workspace_dir = workspace
            gen.MODEL_NODE_ID = "generate-mv"
            gen.shared_model_dirs = {
                "pixal3d-base": str(workspace / "base"),
                "pixal3d-mv": str(workspace / "mv"),
                "da3-base": str(workspace / "da3"),
            }
            output = workspace / "Workflows" / "result.glb"
            with patch.object(gen, "_prepare_generation_assets", return_value=workspace / "naf.pth"), \
                 patch("pixal3d_extension.multiview_images.run_multiview_from_images", return_value=output) as run:
                result = gen.generate(
                    png_bytes((1, 0, 0, 255)),
                    {"extra_image_paths": [str(second), str(third)], "resolution": 1024},
                )
            self.assertEqual(result, output)
            self.assertEqual(run.call_args.kwargs["extra_image_paths"], [str(second), str(third)])
            self.assertEqual(run.call_args.kwargs["da3_root"], workspace / "da3")
            self.assertNotIn("extra_image_paths", run.call_args.kwargs["params"])

    def test_positional_gaps_are_ignored_without_reordering_connected_views(self):
        from pixal3d_extension.multiview_images import validate_ordered_images

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            second = inputs / "port-2.png"
            third = inputs / "port-3.png"
            fourth = inputs / "port-4.png"
            Image.new("RGBA", (32, 24), (2, 0, 0, 255)).save(second)
            Image.new("RGBA", (32, 24), (3, 0, 0, 255)).save(third)
            Image.new("RGBA", (32, 24), (4, 0, 0, 255)).save(fourth)

            port_three_only = validate_ordered_images(
                png_bytes((1, 0, 0, 255)), [None, str(third)], workspace,
            )
            self.assertEqual([image.data for image in port_three_only], [
                png_bytes((1, 0, 0, 255)), third.read_bytes(),
            ])

            multiple_holes = validate_ordered_images(
                png_bytes((1, 0, 0, 255)), [None, str(second), None], workspace,
            )
            self.assertEqual([image.data for image in multiple_holes], [
                png_bytes((1, 0, 0, 255)), second.read_bytes(),
            ])

            with self.assertRaisesRegex(ValueError, "at least two"):
                validate_ordered_images(png_bytes(), [None, None, None], workspace)
            with self.assertRaisesRegex(ValueError, "at most four"):
                validate_ordered_images(
                    png_bytes(), [str(second), None, str(third), str(fourth), str(second)], workspace,
                )
            for invalid in (False, 7, {}, []):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "None or a workspace image path"):
                    validate_ordered_images(png_bytes(), [invalid], workspace)

    def test_validation_rejects_fewer_than_two_duplicates_escape_symlinks_and_spoofing_without_mutation(self):
        from pixal3d_extension.multiview_images import validate_ordered_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            image = inputs / "view.png"
            image.write_bytes(png_bytes())
            outside = root / "outside.png"
            outside.write_bytes(png_bytes())
            link = inputs / "link.png"
            link.symlink_to(outside)
            output = workspace / "Workflows" / "outputs"
            cases = [
                ([], "at least two"),
                ([str(image), None, str(image)], "duplicate"),
                ([str(outside)], "workspace"),
                ([str(link)], "symlink"),
            ]
            for paths, message in cases:
                with self.subTest(paths=paths), self.assertRaisesRegex((ValueError, TypeError), message):
                    validate_ordered_images(png_bytes(), paths, workspace)
                self.assertFalse(output.exists())

            gen = Pixal3DGenerator(workspace / "models")
            gen.workspace_dir = workspace
            gen.MODEL_NODE_ID = "generate-mv"
            with self.assertRaisesRegex(ValueError, "reserved"):
                gen.generate(png_bytes(), {"extra_image_paths": [str(image)], "capture_manifest_path": "forged"})

    def test_validation_rejects_bad_content_and_more_than_four(self):
        from pixal3d_extension.multiview_images import validate_ordered_images

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            bad = inputs / "bad.png"
            bad.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "supported image"):
                validate_ordered_images(png_bytes(), [str(bad)], workspace)
            paths = []
            for index in range(4):
                path = inputs / f"{index}.png"
                path.write_bytes(png_bytes())
                paths.append(str(path))
            with self.assertRaisesRegex(ValueError, "at most four"):
                validate_ordered_images(png_bytes(), paths, workspace)

    def test_pre_cancel_has_no_filesystem_or_asset_side_effects(self):
        from pixal3d_extension.multiview_images import run_multiview_from_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            second = inputs / "second.png"
            second.write_bytes(png_bytes())
            before = {path.relative_to(workspace) for path in workspace.rglob("*")}
            cancel = threading.Event()
            cancel.set()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                run_multiview_from_images(
                    primary_image_bytes=png_bytes(), extra_image_paths=[str(second)],
                    workspace_dir=workspace, output_dir=workspace / "Workflows" / "outputs",
                    da3_root=root / "da3", mv_root=root / "mv", base_root=root / "base",
                    naf_path=root / "naf.pth", params={}, cancel_event=cancel,
                )
            self.assertEqual({path.relative_to(workspace) for path in workspace.rglob("*")}, before)

    def test_runtime_invokes_da3_then_calibrated_runner_and_cleans_private_staging(self):
        from pixal3d_extension.multiview_images import run_multiview_from_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            second = inputs / "second.png"
            second.write_bytes(png_bytes((2, 0, 0, 255)))
            da3 = root / "da3"
            da3.mkdir()
            for name in ("config.json", "model.safetensors"):
                (da3 / name).write_bytes(b"local")
            observed = {}

            def calibrate(frame_paths, da3_root, staging, extension_root, **_kwargs):
                observed["order"] = [Image.open(path).getpixel((0, 0))[0] for path in frame_paths]
                observed["da3"] = da3_root
                transforms = staging / "transforms.json"
                transforms.write_text(json.dumps({
                    "mesh_scale": 1.0,
                    "frames": [
                        {"file_path": f"{index:04d}.png", "camera_angle_x": 0.5,
                         "transform_matrix": [[1, 0, 0, 0], [0, 0, -1, -3], [0, 1, 0, index * 0.1], [0, 0, 0, 1]]}
                        for index in range(2)
                    ],
                }))
                return transforms

            output = workspace / "Workflows" / "result.glb"
            with patch("pixal3d_extension.multiview_images.validate_da3_runtime"), \
                 patch("pixal3d_extension.multiview_images._estimate_cameras", side_effect=calibrate) as estimator, \
                 patch("pixal3d_extension.multiview.run_multiview", return_value=output) as calibrated:
                result = run_multiview_from_images(
                    primary_image_bytes=png_bytes((1, 0, 0, 255)), extra_image_paths=[str(second)],
                    workspace_dir=workspace, output_dir=workspace / "Workflows", da3_root=da3,
                    mv_root=root / "mv", base_root=root / "base", naf_path=root / "naf.pth",
                    params={"resolution": 1024},
                )
            self.assertEqual(result, output)
            estimator.assert_called_once()
            self.assertEqual(observed, {"order": [1, 2], "da3": da3})
            self.assertEqual(calibrated.call_args.kwargs["params"]["num_views"], 2)
            staged_manifest = Path(calibrated.call_args.kwargs["scene_manifest_path"])
            self.assertFalse(staged_manifest.parent.exists())

    def test_declared_roles_reorder_connected_slots_and_build_fixed_rig_without_da3(self):
        from pixal3d_extension.multiview_images import run_multiview_from_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            third = inputs / "third.png"
            fourth = inputs / "fourth.png"
            third.write_bytes(png_bytes((3, 0, 0, 255)))
            fourth.write_bytes(png_bytes((4, 0, 0, 255)))
            observed = {}

            def calibrated(**kwargs):
                observed["transforms"] = json.loads(
                    (Path(kwargs["scene_manifest_path"]).parent / "transforms.json").read_text()
                )
                observed["manifest"] = json.loads(Path(kwargs["scene_manifest_path"]).read_text())
                observed["order"] = [
                    Image.open(Path(kwargs["scene_manifest_path"]).parent / frame["file_path"])
                    .getpixel((0, 0))[0]
                    for frame in observed["transforms"]["frames"]
                ]
                return workspace / "Workflows" / "result.glb"

            with patch("pixal3d_extension.multiview_images.validate_da3_runtime"), \
                 patch("pixal3d_extension.multiview_images._estimate_cameras") as estimator, \
                 patch("pixal3d_extension.multiview.run_multiview", side_effect=calibrated):
                run_multiview_from_images(
                    primary_image_bytes=png_bytes((1, 0, 0, 255)),
                    extra_image_paths=[None, str(third), str(fourth)],
                    workspace_dir=workspace, output_dir=workspace / "Workflows", da3_root=root / "missing-da3",
                    mv_root=root / "mv", base_root=root / "base", naf_path=root / "naf.pth",
                    params={
                        "view_layout": "declared_roles",
                        "image_role": "back",
                        "image_3_role": "front",
                        "image_4_role": "left",
                    },
                )

            estimator.assert_not_called()
            self.assertEqual(observed["order"], [3, 1, 4])
            expected_roles = [
                {"slot": 3, "handle": "image_3", "role": "front", "azimuthDegrees": 0},
                {"slot": 1, "handle": "image", "role": "back", "azimuthDegrees": 180},
                {"slot": 4, "handle": "image_4", "role": "left", "azimuthDegrees": 270},
            ]
            self.assertEqual(observed["transforms"]["provenance"]["viewRoles"], expected_roles)
            self.assertEqual(observed["manifest"]["provenance"]["viewRoles"], expected_roles)
            frames = observed["transforms"]["frames"]
            expected_fov = math.radians(20.0)
            expected_distance = 0.55 / math.tan(expected_fov / 2.0)
            self.assertEqual([frame["camera_angle_x"] for frame in frames], [expected_fov] * 3)
            expected_positions = [
                (0, -expected_distance, 0),
                (0, expected_distance, 0),
                (-expected_distance, 0, 0),
            ]
            expected_matrices = [
                [[1, 0, 0, 0], [0, 0, -1, -expected_distance], [0, 1, 0, 0], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 1, expected_distance], [0, 1, 0, 0], [0, 0, 0, 1]],
                [[0, 0, -1, -expected_distance], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            ]
            for frame, expected_position, expected_matrix in zip(
                frames, expected_positions, expected_matrices, strict=True,
            ):
                actual = tuple(frame["transform_matrix"][axis][3] for axis in range(3))
                for value, wanted in zip(actual, expected_position, strict=True):
                    self.assertAlmostEqual(value, wanted, places=7)
                for actual_row, expected_row in zip(frame["transform_matrix"], expected_matrix, strict=True):
                    for value, wanted in zip(actual_row, expected_row, strict=True):
                        self.assertAlmostEqual(value, wanted, places=7)
            self.assertEqual(
                observed["transforms"]["provenance"]["cameraPosePolicy"],
                "declared-role-canonical-orbit",
            )
            self.assertEqual(observed["transforms"]["provenance"]["cameraEstimator"], "declared fixed orbit")

    def test_declared_view_fov_rejects_non_numeric_non_finite_and_out_of_range_values(self):
        from pixal3d_extension.multiview_images import _write_declared_role_transforms

        roles = [{"slot": 1, "handle": "image", "role": "front", "azimuthDegrees": 0}]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "transforms.json"
            for value in (True, "invalid", float("nan"), 0, 171):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "between 1 and 170"):
                    _write_declared_role_transforms(target, roles, value)
                self.assertFalse(target.exists())

    def test_auto_layout_rejects_collapsed_da3_camera_coverage(self):
        from pixal3d_extension.multiview_images import run_multiview_from_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            inputs = workspace / "Workflows" / "inputs"
            inputs.mkdir(parents=True)
            extras = []
            for index in range(1, 4):
                path = inputs / f"view-{index}.png"
                path.write_bytes(png_bytes((index + 1, 0, 0, 255)))
                extras.append(str(path))
            da3 = root / "da3"
            da3.mkdir()
            for name in ("config.json", "model.safetensors"):
                (da3 / name).write_bytes(b"local")

            def calibrate(_frame_paths, _da3_root, staging, _extension_root, **_kwargs):
                transforms = staging / "transforms.json"
                transforms.write_text(json.dumps({
                    "mesh_scale": 1.0,
                    "frames": [
                        {
                            "file_path": f"{index:04d}.png",
                            "camera_angle_x": 0.5,
                            "transform_matrix": [
                                [1, 0, 0, offset],
                                [0, 0, -1, -3],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1],
                            ],
                        }
                        for index, offset in enumerate((0.0, 0.02, 0.001, -0.02))
                    ],
                }))
                return transforms

            with patch("pixal3d_extension.multiview_images.validate_da3_runtime"), \
                 patch("pixal3d_extension.multiview_images._estimate_cameras", side_effect=calibrate), \
                 patch("pixal3d_extension.multiview.run_multiview") as inference:
                with self.assertRaisesRegex(ValueError, "collapsed.*Declared roles"):
                    run_multiview_from_images(
                        primary_image_bytes=png_bytes((1, 0, 0, 255)),
                        extra_image_paths=extras,
                        workspace_dir=workspace, output_dir=workspace / "Workflows", da3_root=da3,
                        mv_root=root / "mv", base_root=root / "base", naf_path=root / "naf.pth",
                        params={"view_layout": "auto"},
                    )
            inference.assert_not_called()

    def test_declared_roles_reject_duplicate_roles_and_missing_front_for_connected_slots(self):
        from pixal3d_extension.multiview_images import _apply_declared_view_roles, _decode

        images = [_decode(png_bytes((index, 0, 0, 255)), f"view {index}") for index in (1, 2)]
        with self.assertRaisesRegex(ValueError, "unique"):
            _apply_declared_view_roles(
                images, ["connected"],
                {"view_layout": "declared_roles", "image_role": "front", "image_2_role": "front"},
            )
        with self.assertRaisesRegex(ValueError, "exactly one front"):
            _apply_declared_view_roles(
                images, ["connected"],
                {"view_layout": "declared_roles", "image_role": "left", "image_2_role": "right"},
            )

    def test_auto_layout_keeps_connected_order_and_has_no_role_provenance(self):
        from pixal3d_extension.multiview_images import _apply_declared_view_roles, _decode

        images = [_decode(png_bytes((index, 0, 0, 255)), f"view {index}") for index in (1, 3)]
        ordered, provenance = _apply_declared_view_roles(
            images, [None, "connected"], {"view_layout": "auto"},
        )
        self.assertIs(ordered, images)
        self.assertIsNone(provenance)

    def test_calibration_worker_produces_ordered_valid_transforms_from_da3(self):
        import numpy as np
        from pixal3d_extension.multiview_camera_worker import estimate_transforms

        intrinsics = np.asarray([
            [[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]],
            [[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]],
        ])
        w2c = np.asarray([
            np.eye(4),
            [[1.0, 0.0, 0.0, -0.3], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ])
        depth = np.full((2, 24, 32), 2.0)
        result = estimate_transforms(intrinsics, w2c, depth, source_hw=(24, 32), processed_hw=(24, 32))
        self.assertEqual([item["file_path"] for item in result["frames"]], ["0000.png", "0001.png"])
        self.assertEqual(len(result["frames"]), 2)
        self.assertGreater(result["frames"][0]["camera_angle_x"], 0)
        self.assertTrue(np.allclose(
            np.asarray(result["frames"][0]["transform_matrix"]),
            np.asarray([[1, 0, 0, 0], [0, 0, -1, -3], [0, 1, 0, 0], [0, 0, 0, 1]]),
            atol=1e-6,
        ))
        self.assertGreater(abs(result["frames"][1]["transform_matrix"][0][3]), 0.1)
        for frame in result["frames"]:
            matrix = np.asarray(frame["transform_matrix"])
            self.assertEqual(matrix.shape, (4, 4))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0, places=5)

    def test_da3_camera_estimator_loads_only_the_ui_managed_local_checkpoint(self):
        import numpy as np
        try:
            from pixal3d_extension.scene_prepare_worker import _run_da3
        except ModuleNotFoundError as exc:
            self.skipTest(f"isolated runtime dependency unavailable: {exc}")

        observed = {}

        class Prediction:
            depth = np.ones((2, 8, 8), dtype=np.float32)
            conf = None
            intrinsics = np.asarray([np.eye(3), np.eye(3)], dtype=np.float64)
            extrinsics = np.asarray([np.eye(4), np.eye(4)], dtype=np.float64)

        class Model:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                observed["path"] = path
                observed["kwargs"] = kwargs
                return cls()

            def to(self, device):
                observed["device"] = device
                return self

            def inference(self, **kwargs):
                observed["inference"] = kwargs
                return Prediction()

        with patch("pixal3d_extension.da3_official_adapter.load_depth_anything3", return_value=Model):
            _run_da3([Path("one.png"), Path("two.png")], Path("/ui-managed-da3"), {"process_resolution": 504})
        self.assertEqual(observed["path"], "/ui-managed-da3")
        self.assertEqual(observed["kwargs"], {"local_files_only": True})
        self.assertEqual(observed["device"], "cuda")
        self.assertEqual(observed["inference"]["ref_view_strategy"], "saddle_balanced")
        self.assertNotEqual(observed["inference"]["ref_view_strategy"], "middle")

    def test_windows_reader_uses_locked_handle_identity_and_final_path_custody(self):
        source = (Path(__file__).parents[1] / "pixal3d_extension" / "multiview_images.py").read_text()
        self.assertIn("CreateFileW", source)
        self.assertIn("GetFileInformationByHandle", source)
        self.assertIn("GetFinalPathNameByHandleW", source)
        self.assertIn("GetFileInformationByHandleEx", source)
        self.assertIn("FileIdInfo", source)
        self.assertIn("ReadFile", source)
        self.assertIn("FILE_ATTRIBUTE_REPARSE_POINT", source)
        self.assertIn("FILE_SHARE_READ", source)
        self.assertIn("FILE_FLAG_BACKUP_SEMANTICS", source)
        self.assertIn("actual_ancestor_identities", source)
        self.assertIn("lexical_ancestor_identities", source)
        self.assertNotIn("GetLongPathNameW", source)
        self.assertNotIn("normcase", source)
        self.assertNotIn("commonpath", source)
        workspace_file = source.split("def _workspace_file", 1)[1].split("def validate_ordered_images", 1)[0]
        self.assertLess(workspace_file.index('if os.name == "nt":'), workspace_file.index("resolve(strict=True)"))
        windows_branch = source.split('if os.name == "nt":', 1)[1].split("# Open every component", 1)[0]
        self.assertNotIn("read_bytes", windows_branch)


if __name__ == "__main__":
    unittest.main()
