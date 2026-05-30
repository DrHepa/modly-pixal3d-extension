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

Current MVP lane:

```text
pixal3d-wheelhouse-v0.1.0-linux-aarch64-cp312-cuda124.zip
```

Candidate Linux x64 lane:

```text
linux-x64-cp312-cuda124
```

This lane is intentionally **not declared as supported** in `wheelhouse.manifest.json` until a real, checksum-pinned release asset exists. Use the GitHub Actions recipe in `.github/workflows/wheelhouse-linux-x64-cp312-cuda124.yml` to validate build prerequisites on hosted Linux x64. The workflow uses `ubuntu-22.04`, installs CUDA 12.4 `nvcc`/runtime headers via the CUDA toolkit action, then installs cuRAND/cuSPARSE/cuBLAS/cuSOLVER/NVRTC development headers and link libraries with apt (`libcurand-dev-12-4`, `libcusparse-dev-12-4`, `libcublas-dev-12-4`, `libcusolver-dev-12-4`, `libnvrtc-dev-12-4`) before running the build recipe. It also exports `LIBRARY_PATH` and `LD_LIBRARY_PATH` with `${CUDA_HOME}/lib64` plus `${CUDA_HOME}/lib64/stubs` so native extensions can link against `libnvrtc.so` and the hosted-runner `libcuda.so` stub. It must fail clearly when the CUDA 12.4 toolchain or native build prerequisites are unavailable, and it must not publish placeholder archives.

Native source refs for the linux-x64 probe are documented in `native-sources.linux-x64-cp312-cuda124.env.example`. The file uses immutable SHAs and includes confidence notes for each package. The workflow loads this file explicitly; source fetching still requires `WHEELHOUSE_ALLOW_SOURCE_FETCH=1` from the env file, so local runs remain opt-in. `O_VOXEL_SOURCE_SUBDIR=o-voxel` is required because `o_voxel` lives inside the TRELLIS.2 repository rather than in its own top-level repo.

The candidate recipe in `build-linux-x64-cp312-cuda124.sh` is staged deliberately:

1. Create isolated `work/`, `wheelhouse/`, and `dist/wheelhouse/` directories.
2. Install Python build tooling and PyTorch from the CUDA 12.4 wheel index.
3. Copy the existing pure Python wheels from `wheels/`: `pixal3d_core`, `pipeline`, `moge`, `naf`, and `utils3d`.
4. Build native wheels only from explicit source directories or explicit source URLs/refs. The script fails with `missing native source directories` instead of creating fake wheels.

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

If a maintainer chooses source fetching in CI, they must set both `*_SOURCE_URL` and immutable `*_SOURCE_REF` for each package plus `WHEELHOUSE_ALLOW_SOURCE_FETCH=1`. The script does not invent or fetch sources silently.

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
