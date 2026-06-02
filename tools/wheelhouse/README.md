# Pixal3D wheelhouse release recipe

This directory documents the maintainer workflow for publishing release-backed wheelhouse assets. End users should run `python3 setup.py --prepare --json`; they should not build wheels locally during normal setup.

## Local build recipe

1. Build wheels in a clean Linux `aarch64`, Python `cp312`, CUDA `12.4` environment.
2. Collect only the packages declared in `wheelhouse.manifest.json`.
3. Place wheels under a top-level `wheelhouse/` directory inside the archive.
4. Create a `.zip` archive so setup can extract it with Python standard-library tooling.

## Asset naming

Use the manifest lane in the filename:

```text
pixal3d-wheelhouse-v<version>-<os>-<arch>-<python-tag>-<runtime>.zip
```

Published lanes:

```text
pixal3d-wheelhouse-v0.1.0-linux-aarch64-cp312-cuda124.zip
pixal3d-wheelhouse-v0.1.0-linux-x64-cp312-cuda124.zip
pixal3d-wheelhouse-v0.1.0-windows-x64-cp311-cuda124.zip
pixal3d-wheelhouse-v0.1.0-windows-x64-cp312-cuda124.zip
```

Windows x64 lane:

```text
windows-x64-cp312-cuda124
```

The Windows lane is declared as supported only after the exact-stack candidate archive was uploaded to the pinned release and checksum-pinned in `wheelhouse.manifest.json`. Use `.github/workflows/wheelhouse-windows-x64-cp312-cuda124.yml` and `build-windows-x64-cp312-cuda124.ps1` to rebuild or audit exact-stack Windows wheels. Candidate artifacts include a `WINDOWS-CANDIDATE.json` metadata file recording downloaded wheel URLs, checksums, exact-stack tags, and optional exclusions.

Use the GitHub Actions recipe in `.github/workflows/wheelhouse-linux-x64-cp312-cuda124.yml` to validate Linux x64 rebuilds on hosted Linux. The workflow uses `ubuntu-22.04`, installs CUDA 12.4 `nvcc`/runtime headers via the CUDA toolkit action, then installs cuRAND/cuSPARSE/cuBLAS/cuSOLVER/NVRTC development headers and link libraries with apt (`libcurand-dev-12-4`, `libcusparse-dev-12-4`, `libcublas-dev-12-4`, `libcusolver-dev-12-4`, `cuda-nvrtc-dev-12-4`) before running the build recipe. It also exports `LIBRARY_PATH` and `LD_LIBRARY_PATH` with `${CUDA_HOME}/lib64` plus `${CUDA_HOME}/lib64/stubs` so native extensions can link against `libnvrtc.so` and the hosted-runner `libcuda.so` stub. It must fail clearly when the CUDA 12.4 toolchain or native build prerequisites are unavailable, and it must not publish placeholder archives.

Native source refs for the linux-x64 probe are documented in `native-sources.linux-x64-cp312-cuda124.env.example`. The file uses immutable SHAs and includes confidence notes for each package. The workflow loads this file explicitly; source fetching still requires `WHEELHOUSE_ALLOW_SOURCE_FETCH=1` from the env file, so local runs remain opt-in. `O_VOXEL_SOURCE_SUBDIR=o-voxel` is required because `o_voxel` lives inside the TRELLIS.2 repository rather than in its own top-level repo.

The candidate recipe in `build-linux-x64-cp312-cuda124.sh` is staged deliberately:

1. Create isolated `work/`, `wheelhouse/`, and `dist/wheelhouse/` directories.
2. Install Python build tooling and PyTorch from the CUDA 12.4 wheel index.
3. Copy the existing pure Python wheels from `wheels/`: `pixal3d_core`, `pipeline`, `moge`, `naf`, and `utils3d`.
4. Build base native wheels only from explicit source directories or explicit source URLs/refs. The script fails with `missing native source directories` instead of creating fake wheels.

The Linux x64 base wheelhouse deliberately does **not** block on NATTEN/libnatten. NATTEN is only needed for strict NAF; setup probes `natten.HAS_LIBNATTEN` and must keep/fall back to non-strict NAF unless that value is `True`. Maintainers can opt into a strict NATTEN attempt with `WHEELHOUSE_BUILD_STRICT_NATTEN=1`, but a failed optional NATTEN build does not invalidate the base Pixal3D wheelhouse. Hosted Linux x64 builds intentionally narrow NATTEN's optional default arch list to `WHEELHOUSE_NATTEN_CUDA_ARCH=8.9` and wrap each native package build with `WHEELHOUSE_NATIVE_BUILD_TIMEOUT=60m`. NATTEN can otherwise spend the full GitHub Actions job timeout compiling kernels for every CUDA architecture before the remaining native packages are attempted.

Windows wheelhouses must follow the exact-stack policy used by Pixal3D-ComfyUI references: Python ABI, PyTorch minor, CUDA minor, platform tag, and GPU architecture/SM coverage must all match. Modly's packaged app currently embeds python-build-standalone `3.11.9`, so `windows-x64-cp311-cuda124` is the primary GitHub-install lane; `windows-x64-cp312-cuda124` remains available for development environments running Python 3.12. The base lanes use exact-stack Pozzetti wheels for `flex_gemm_ap`, `cumesh_vb`, `o_voxel_vb_ap`, `drtk`, `flash_attn`, `nvdiffrast`, and `nvdiffrec_render`. Do not publish or auto-install generic Windows NATTEN wheels; use curated exact-stack artifacts only, and keep fallback NAF first-class when `HAS_LIBNATTEN` is unavailable.

## Windows NATTEN candidate workflow

`.github/workflows/natten-windows-x64-cp311-cuda124-candidate.yml` is a manual GitHub Actions probe for building a native Windows `win_amd64` NATTEN wheel from `SHI-Labs/NATTEN` tag `v0.17.5` against Python `3.11`, torch `2.6.0+cu124`, and CUDA `12.4`.

This workflow is candidate-only:

- It uploads a GitHub Actions artifact, not a release asset.
- It must not update `wheelhouse.manifest.json`.
- It accepts `workflow_dispatch` input `cuda_arch_list` with default `7.5;8.6;8.9` so maintainers can narrow or expand SM coverage deliberately.
- It writes `NATTEN-CANDIDATE.json` and `verification.json` alongside the built wheel.
- It must verify `natten.HAS_LIBNATTEN == True` before any future wheelhouse integration or setup-policy change.

Treat a successful build with `HAS_LIBNATTEN=False` as a failed integration candidate. The artifact may still be useful for compiler-log inspection, but it is NOT publishable and must not be wired into release-backed wheelhouse lanes.

Native source directories can be supplied with:

```bash
NATTEN_SOURCE_DIR=/path/to/natten \
O_VOXEL_SOURCE_DIR=/path/to/o_voxel \
CUMESH_SOURCE_DIR=/path/to/cumesh \
FLEX_GEMM_SOURCE_DIR=/path/to/flex_gemm \
NVDIFFRAST_SOURCE_DIR=/path/to/nvdiffrast \
NVDIFFREC_RENDER_SOURCE_DIR=/path/to/nvdiffrec_render \
tools/wheelhouse/build-linux-x64-cp312-cuda124.sh
```

If a maintainer chooses source fetching in CI, they must set both `*_SOURCE_URL` and immutable `*_SOURCE_REF` for each package plus `WHEELHOUSE_ALLOW_SOURCE_FETCH=1`. The script does not invent or fetch sources silently. Fetched repositories are cloned with `--recurse-submodules`, then checked out to the immutable ref and followed by `git submodule update --init --recursive`; this is required for packages such as CuMesh that vendor CUDA sources under submodules like `third_party/cubvh`.

## SHA256SUMS

Generate checksums after the archive is final:

```bash
sha256sum pixal3d-wheelhouse-v0.1.0-linux-aarch64-cp312-cuda124.zip > SHA256SUMS
```

Copy the archive hash and size into `wheelhouse.manifest.json`. Do not use `latest` tags or unchecked assets.

## Release checklist

- Publish assets under the pinned release tag declared in `wheelhouse.manifest.json`.
- Keep lane additions additive; do not replace an existing asset under the same tag without an explicit reissue and manifest checksum update.
- Verify setup with a preseeded cache, a mocked release download, checksum failure, unsupported lane, and vendored fallback.
- Do not commit generated wheelhouse archives or new binary wheels as part of the release recipe.

## Test policy

The wheelhouse contract is tested through `node --test` spawning small Python snippets against `modly_wheelhouse.py` and `setup.py`. This keeps the extension aligned with the repository's active strict-TDD runner while still exercising the Python helper behavior directly. Separate `pytest` parity tests are deferred unless the helper grows beyond what these contract tests can cover clearly.

## GitHub rate limits and auth failures

Setup derives direct release asset URLs from `wheelhouse.manifest.json`; it does not list releases or enumerate assets during normal setup. This intentionally reduces GitHub API rate-limit exposure for large wheelhouse downloads. HTTP 401/403 responses are treated as `auth_required`/retryable release-access failures so setup can use the verified vendored fallback during migration, but checksum or structural failures still fail closed.
