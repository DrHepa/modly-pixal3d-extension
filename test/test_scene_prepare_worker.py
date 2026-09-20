import unittest
import sys
import types

import numpy as np

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
from pixal3d_extension.scene_prepare_worker import _consistent_cameras, _parameters


class ScenePrepareWorkerTests(unittest.TestCase):
    def test_parameters_are_bounded_to_ui_processing_resolutions(self):
        self.assertEqual(_parameters({})["process_resolution"], 504)
        self.assertEqual(_parameters({"minimum_geometry_points": 256})["minimum_geometry_points"], 256)
        with self.assertRaisesRegex(ValueError, "one of"):
            _parameters({"process_resolution": 500})

    def test_camera_consistency_recomputes_k_after_rejecting_outlier(self):
        def intrinsic(focal):
            return np.array([[focal, 0, 50], [0, focal, 40], [0, 0, 1]], dtype=float)

        accepted, median = _consistent_cameras(
            [intrinsic(100), intrinsic(102), intrinsic(1000)],
            np.repeat(np.eye(4)[None], 3, axis=0),
            0.15,
        )
        self.assertEqual(accepted, [0, 1])
        np.testing.assert_allclose(median, [101, 101, 50, 40])


if __name__ == "__main__":
    unittest.main()
