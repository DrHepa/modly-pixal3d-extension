# Pixal3D Modly Extension

Pixal3D image-to-3D model extension for Modly.

This repository contains only the extension runtime, setup entrypoint, and a release-backed wheelhouse contract needed to prepare Pixal3D dependencies from a GitHub install. Model weights are not included; Modly downloads model assets through its UI into the normal Modly model storage.

## What setup does

```bash
python3 setup.py --prepare --json
```

- creates `venv/` inside the extension
- installs `requirements.txt`
- prepares the selected release-backed wheelhouse from `wheelhouse.manifest.json`
- installs native packages with `pip install --no-index --find-links <verified-wheelhouse>`
- runs `pip check`
- creates logical Modly model-storage folders under the configured Modly models root

It does **not** download model weights and does **not** run generation.

## Release-backed wheelhouse

`wheelhouse.manifest.json` pins the release tag, selected platform lane, archive filename, checksum, and fallback policy. Setup verifies the selected archive before extraction and installs native packages only from a verified local path using `--no-index --find-links`.

The vendored `wheels/` fallback is intentionally retained for migration/rollback. It is used only after retryable release access failures such as network/auth errors and only when every wheel is lane-compatible and hash-verified. Setup must not silently fall back to PyPI for native packages.

Current wheelhouse targets Linux `aarch64` / Python `cp312` / `cuda124` for native wheels.

Included packaged dependencies:

- `pixal3d-core==0.1.0+modly`
- `moge==2.0.0+modly`
- `naf==0.1.0+modly`
- `utils3d==1.3+modly.headless`
- `pipeline==1.0.0+modly`
- `natten==0.21.0`
- `o-voxel==0.0.1`
- `cumesh==0.0.1`
- `flex-gemm==1.0.0`
- `nvdiffrast==0.4.0`
- `nvdiffrec-render==0.0.0`

## Modly contract

- `manifest.json` declares the model extension.
- `setup.py` prepares the extension environment.
- `generator.py` exposes `Pixal3DGenerator`.
- Model assets must live under Modly's model storage, not inside this repository.

## View-aligned generation behavior

Pixal3D generates meshes aligned to the input image projection rather than always canonicalizing the object to a universal upright/front pose. Upstream's projected render path is designed so the first rendered frame matches the projected input view.

Practical implications:

- front/straight input images should usually produce upright meshes;
- angled or isometric input images can produce angled meshes;
- that input-dependent tilt is expected Pixal3D behavior, not a Modly orientation bug;
- do not apply a fixed post-export pitch correction to all Pixal3D outputs, because it can break already-upright generations.

The extension preserves Pixal3D's upstream GLB export orientation. It must not add an extra hardcoded yaw rotation after export; doing so can flip the generated mesh away from Modly's default camera.

## Remaining runtime requirement

After setup succeeds, use Modly UI to download Pixal3D/DINO/RMBG/NAF/MoGe model assets. Real generation should be validated only after those assets are present.
