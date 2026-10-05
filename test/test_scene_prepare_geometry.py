import unittest

import numpy as np

from pixal3d_extension.scene_geometry import (
    aabb_iou,
    deduplicate_labeled_aabbs,
    percentile_aabb,
    reprojection_coverage,
    scale_intrinsics,
    unproject_masked_points,
    w2c_opencv_to_c2w_blender,
)


class SceneGeometryTests(unittest.TestCase):
    def test_intrinsics_scale_from_declared_source_and_processing_resolution(self):
        intrinsic = np.array([[800.0, 0.0, 300.0], [0.0, 900.0, 200.0], [0.0, 0.0, 1.0]])
        scaled = scale_intrinsics(intrinsic, source_hw=(400, 600), processed_hw=(200, 150))
        np.testing.assert_allclose(scaled, [[200.0, 0.0, 75.0], [0.0, 450.0, 100.0], [0.0, 0.0, 1.0]])

    def test_w2c_opencv_to_c2w_blender_has_tested_axis_convention(self):
        w2c = np.eye(4)
        w2c[2, 3] = 2.0
        c2w = w2c_opencv_to_c2w_blender(w2c)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        np.testing.assert_allclose(np.linalg.inv(c2w @ flip), w2c)
        np.testing.assert_allclose(c2w[:3, 3], [0.0, 0.0, -2.0])

    def test_eroded_mask_confidence_and_depth_trimming_reject_boundaries_and_outliers(self):
        depth = np.ones((7, 7), dtype=np.float64)
        depth[3, 3] = 100.0
        confidence = np.ones((7, 7), dtype=np.float64)
        confidence[2, 2] = 0.0
        mask = np.ones((7, 7), dtype=bool)
        points = unproject_masked_points(
            depth=depth,
            confidence=confidence,
            mask=mask,
            intrinsics=np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]]),
            w2c=np.eye(4),
            source_hw=(7, 7),
            processed_hw=(7, 7),
            erosion_radius=1,
            confidence_percentile=10.0,
            depth_percentiles=(5.0, 95.0),
        )
        self.assertGreater(len(points), 0)
        self.assertLess(float(points[:, 2].max()), 2.0)
        self.assertTrue(np.all(np.abs(points[:, :2]) <= 0.5))

    def test_percentile_aabb_is_recomputed_only_after_frames_are_accepted(self):
        accepted = [np.array([[0, 0, 1], [1, 1, 2]], dtype=float), np.array([[0.2, 0.2, 1.2]], dtype=float)]
        rejected = np.array([[100, 100, 100]], dtype=float)
        box = percentile_aabb([accepted[0], rejected, accepted[1]], accepted_indices=[0, 2], percentiles=(0, 100))
        np.testing.assert_allclose(box, [[0, 0, 1], [1, 1, 2]])

    def test_reprojection_coverage_counts_mask_support(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:7, 4:7] = True
        points = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]], dtype=float)
        intrinsic = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=float)
        coverage = reprojection_coverage(points, mask, intrinsic, np.eye(4))
        self.assertGreater(coverage, 0.25)
        self.assertLessEqual(coverage, 1.0)

    def test_label_aware_3d_iou_dedup(self):
        a = [[0, 0, 0], [1, 1, 1]]
        b = [[0.05, 0.05, 0.05], [1.05, 1.05, 1.05]]
        self.assertGreater(aabb_iou(a, b), 0.7)
        instances = [
            {"label": "chair", "score": 0.9, "aabb_world": a},
            {"label": "chair", "score": 0.8, "aabb_world": b},
            {"label": "table", "score": 0.7, "aabb_world": b},
        ]
        kept = deduplicate_labeled_aabbs(instances, iou_threshold=0.5)
        self.assertEqual([(item["label"], item["score"]) for item in kept], [("chair", 0.9), ("table", 0.7)])


if __name__ == "__main__":
    unittest.main()
