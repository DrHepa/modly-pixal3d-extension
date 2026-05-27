# Pixal3D Modly Extension

Pixal3D image-to-3D model extension for Modly.

This repository contains only the extension runtime, setup entrypoint, and a local wheelhouse needed to prepare Pixal3D dependencies from a GitHub install. Model weights are not included; Modly downloads model assets through its UI into the normal Modly model storage.

## What setup does

```bash
python3 setup.py --prepare --json
```

- creates `venv/` inside the extension
- installs `requirements.txt`
- installs packaged wheels from `wheels/`
- runs `pip check`
- creates logical Modly model-storage folders under the configured Modly models root

It does **not** download model weights and does **not** run generation.

## Included wheelhouse

The `wheels/` folder is intentionally versioned so GitHub installs can prepare dependencies without cloning/building extra repositories at user setup time.

Current wheelhouse targets Linux `aarch64` / Python `cp312` for native wheels.

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

## Remaining runtime requirement

After setup succeeds, use Modly UI to download Pixal3D/DINO/RMBG/NAF/MoGe model assets. Real generation should be validated only after those assets are present.
