# Blackwell / RTX 50-series investigation

This document tracks an experimental investigation for NVIDIA Blackwell GPUs such as the GeForce RTX 5090. It is **not** a supported Pixal3D wheelhouse lane.

## Current finding

NVIDIA lists GeForce RTX 5090 as compute capability `12.0`. The current published Pixal3D Windows lane is:

```text
windows-x64-cp311-cuda124
torch==2.6.0+cu124
CUDA toolkit 12.4.1 for native builds
```

CUDA 12.4.1 `nvcc` documentation lists supported generated-code targets through `compute_90` / `sm_90`. It does not list `compute_120` / `sm_120`. Current CUDA documentation does list `compute_120` / `sm_120`.

Therefore Blackwell support should be treated as a new exact-stack lane, not as a small edit to the existing CUDA 12.4 lane.

## Why adding one arch flag is insufficient

The Windows cp311/cu124 wheelhouse does not build every native package in this repository. It packages exact-stack external wheels for:

- `flex_gemm_ap`
- `cumesh_vb`
- `o_voxel_vb_ap`
- `drtk`
- `flash_attn`
- `nvdiffrast`
- `nvdiffrec_render`

The repository-built NATTEN candidate workflow controls `TORCH_CUDA_ARCH_LIST` / `NATTEN_CUDA_ARCH`, but rebuilding only NATTEN for `sm_120` would not prove that the full Pixal3D runtime supports RTX 50-series GPUs. Any one native CUDA wheel without Blackwell code can still raise:

```text
CUDA error: no kernel image is available for execution on the device
```

## Probe workflow

`.github/workflows/blackwell-windows-x64-cp311-cuda128-probe.yml` is a manual probe. It only checks prerequisites:

- Python `3.11` compatibility for Modly packaged-app ABI.
- CUDA toolkit candidate support for `compute_120` / `sm_120`.
- PyTorch cu128 wheel availability for the selected torch/torchvision versions.

The probe uploads `BLACKWELL-SM120-PROBE.json` as a GitHub Actions artifact. It must not update `wheelhouse.manifest.json` and must not publish a release asset.

## Publish criteria for a future Blackwell lane

A future Blackwell lane requires all of the following before it can be declared supported:

1. Exact Python ABI, PyTorch version, CUDA minor, and platform tag selected.
2. `nvcc --list-gpu-arch` and `nvcc --list-gpu-code` include `compute_120` / `sm_120`.
3. All required native CUDA wheels are rebuilt or sourced for the same exact stack.
4. NATTEN imports with `HAS_LIBNATTEN == True` on a real NVIDIA Windows machine.
5. Full Pixal3D Low VRAM generation succeeds on RTX 50-series hardware and exports a valid GLB.
6. Release asset is checksum-pinned in `wheelhouse.manifest.json` only after validation.

Until those conditions are met, RTX 5090 / Blackwell remains experimental and unsupported by the published wheelhouse.
