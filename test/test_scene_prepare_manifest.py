import json
import unittest
from pathlib import Path


class ScenePrepareManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((Path(__file__).parents[1] / "manifest.json").read_text())

    def test_public_scene_prep_nodes_use_native_image_and_video_contracts(self):
        nodes = {node["id"]: node for node in self.manifest["nodes"]}
        self.assertTrue({"generate", "generate-mv", "worldsculpt"}.issubset(nodes))
        self.assertNotIn("scene-from-estimates", nodes)
        images = nodes["scene-from-images"]
        self.assertEqual((images["input"], images["output"]), ("image", "scene"))
        self.assertEqual(images["inputs"], ["image"] * 8)
        self.assertEqual(images["input_labels"], ["Primary view", *[f"View {index}" for index in range(2, 9)]])
        self.assertEqual(images["weight_groups"], ["sam3", "da3-base"])
        video = nodes["scene-from-video"]
        self.assertEqual((video["input"], video["output"]), ("video", "scene"))
        self.assertEqual(video["weight_groups"], ["sam3", "da3-base"])
        self.assertEqual((nodes["normalize-annotated-scene"]["input"], nodes["normalize-annotated-scene"]["output"]), ("scene", "scene"))
        self.assertNotIn("weight_groups", nodes["normalize-annotated-scene"])
        params = {param["id"]: param for param in images["params_schema"]}
        self.assertNotIn("max_frames", params)
        self.assertNotIn("frame_stride", params)
        video_params = {param["id"]: param for param in video["params_schema"]}
        self.assertEqual(
            params,
            {key: value for key, value in video_params.items() if key not in {"max_frames", "frame_stride"}},
        )
        self.assertEqual(video_params["max_frames"]["default"], 16)
        self.assertEqual(video_params["frame_stride"]["default"], 1)
        self.assertEqual(params["minimum_geometry_points"]["default"], 128)

    def test_official_weight_groups_are_immutable_and_exact(self):
        groups = {group["id"]: group for group in self.manifest["weight_groups"]}
        sam = groups["sam3"]["model_sources"][0]
        da3 = groups["da3-base"]["model_sources"][0]
        self.assertEqual(sam["repo_id"], "facebook/sam3")
        self.assertEqual(sam["revision"], "3c879f39826c281e95690f02c7821c4de09afae7")
        self.assertEqual(sam["checks"], ["config.json", "sam3.pt", "LICENSE"])
        self.assertEqual(da3["repo_id"], "depth-anything/DA3-BASE")
        self.assertEqual(da3["revision"], "f4a6c9b3c95e41c82048423d3493a81ec3fa810e")
        self.assertEqual(da3["checks"], ["config.json", "model.safetensors"])


if __name__ == "__main__":
    unittest.main()
