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

This lane is intentionally **not declared as supported** in `wheelhouse.manifest.json` until a real, checksum-pinned release asset exists. Use the GitHub Actions recipe in `.github/workflows/wheelhouse-linux-x64-cp312-cuda124.yml` to validate build prerequisites on hosted Linux x64. The recipe must fail clearly when the CUDA 12.4 toolchain or native build prerequisites are unavailable, and it must not publish placeholder archives.

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
