import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from o_voxel import _C


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


class VoxelRenderer:
    def __init__(self, rendering_options={}):
        self.rendering_options = edict(
            {
                "resolution": None,
                "near": 0.1,
                "far": 10.0,
                "ssaa": 1,
            }
        )
        self.rendering_options.update(rendering_options)

    def render(
        self,
        position: torch.Tensor,
        attrs: torch.Tensor,
        voxel_size: float,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> edict:
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        args = (
            position,
            attrs,
            voxel_size,
            view.T.contiguous(),
            (perspective @ view).T.contiguous(),
            camera,
            0.5 / focalx,
            0.5 / focaly,
            resolution * ssaa,
            resolution * ssaa,
        )
        color, depth, alpha = _C.rasterize_voxels_cuda(*args)

        if ssaa > 1:
            color = F.interpolate(color[None], size=(resolution, resolution), mode="bilinear", align_corners=False, antialias=True).squeeze()
            depth = F.interpolate(depth[None, None], size=(resolution, resolution), mode="bilinear", align_corners=False, antialias=True).squeeze()
            alpha = F.interpolate(alpha[None, None], size=(resolution, resolution), mode="bilinear", align_corners=False, antialias=True).squeeze()

        return edict({"attr": color, "depth": depth, "alpha": alpha})
