"""Weights-free native smoke for pinned SAM3 APIs, not model inference."""
from __future__ import annotations


def _require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def run_native_probe() -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from scipy.ndimage import distance_transform_edt
    from torchvision.ops import nms
    from sam3.model.edt import edt_triton
    from sam3.perflib.triton.connected_components import connected_components_triton
    from sam3.perflib.triton.nms import nms_triton

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Windows Scene Prep native probe requires a real SM120 CUDA device")
    with torch.inference_mode():
        mask = torch.zeros((1, 16, 16), dtype=torch.bool, device="cuda")
        mask[:, 4:8, 4:8] = True
        mask[:, 11:13, 11:13] = True
        distance = edt_triton(mask)
        reference = distance_transform_edt(mask.cpu().numpy()[0])
        torch.cuda.synchronize()
        np.testing.assert_allclose(distance.cpu().numpy()[0], reference, rtol=1e-5, atol=1e-5)

        labels, sizes = connected_components_triton(mask.contiguous())
        torch.cuda.synchronize()
        _require(int(labels[0, 4, 4]) != int(labels[0, 11, 11]), "SAM3 connected components merged disjoint objects")
        _require(bool((sizes[0, 4:8, 4:8] == 16).all()), "SAM3 connected component size is incorrect")
        _require(bool((sizes[0, 11:13, 11:13] == 4).all()), "SAM3 connected component size is incorrect")
        _require(bool((labels[~mask] == 0).all()), "SAM3 connected component background is incorrect")

        ious = torch.tensor([[1, .9, .1], [.9, 1, .1], [.1, .1, 1]], device="cuda")
        scores = torch.tensor([.9, .8, .7], device="cuda")
        _require(nms_triton(ious, scores, .5).cpu().tolist() == [0, 2], "SAM3 Triton NMS returned incorrect indices")
        boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 5, 5]], dtype=torch.float32, device="cuda")
        _require(nms(boxes, scores, .5).cpu().tolist() == [0, 2], "TorchVision CUDA NMS returned incorrect indices")
        query = torch.ones((1, 1, 4, 8), dtype=torch.float16, device="cuda")
        attention = functional.scaled_dot_product_attention(query, query, query)
        torch.cuda.synchronize()
        _require(torch.allclose(attention, query, atol=1e-3, rtol=1e-3), "Torch SDPA returned incorrect values")
    return {name: "PASS" for name in ("sam3_triton_edt", "sam3_triton_connected_components", "sam3_triton_nms", "torchvision_cuda_nms", "torch_cuda_sdpa")}
