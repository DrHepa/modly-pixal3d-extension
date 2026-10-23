# Third-Party Notices

This repository's Modly integration and scene-preparation geometry are
licensed under the repository `LICENSE`. Third-party code, model weights, and
datasets retain their own terms.

## Pixal3D

- Upstream: `TencentARC/Pixal3D`
- Code and published Pixal3D model materials: MIT
- The preserved license is at `licenses/Pixal3D-MIT.txt`.

## AlayaLab WorldSculpt

- Upstream source snapshot: `AlayaLab/WorldSculpt` at
  `373f09f0ecddc94607b569bea0316ff6a2286501`
- Original WorldSculpt code and the official LoRA/aggregator weights: Apache-2.0
- The bundled source retains its upstream `LICENSE`, `NOTICE`, and
  `THIRD_PARTY_LICENSES.md`; the Apache-2.0 text is also at
  `licenses/Apache-2.0.txt`.
- Upstream training/evaluation datasets identified by WorldSculpt are CC BY 4.0.
  Those datasets are not bundled by this extension and must be obtained under
  their own attribution requirements.

## Meta SAM 3

- Official source: `facebookresearch/sam3` at
  `2345a4ad109ac29c569da749c91d84f10dc08c40`
- Official gated checkpoint: `facebook/sam3` at
  `3c879f39826c281e95690f02c7821c4de09afae7`
- License: custom SAM License, not MIT or Apache-2.0. The exact license from the
  pinned source revision is preserved at `licenses/SAM3-LICENSE.txt`.
- Access to the checkpoint requires accepting Meta's terms and authenticating
  in Modly's Hugging Face download flow. Repacked checkpoints do not change the
  SAM License.

## Depth Anything 3 Base

- Official source: `ByteDance-Seed/Depth-Anything-3` at
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Official checkpoint: `depth-anything/DA3-BASE` at
  `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`
- Code and model: Apache-2.0; see `licenses/Apache-2.0.txt`.
- DA3 Base produces relative-scale depth and official camera extrinsics are
  world-to-camera. This extension does not describe those outputs as metric.

## Community inspiration

Warm thanks to [jtydhr88/ComfyUI-WorldSculpt](https://github.com/jtydhr88/ComfyUI-WorldSculpt)
for the SAM3 + DA3 scene-preparation idea and workflow that inspired this Modly
integration. Official upstream code and model licenses remain as listed above.

## Experimental Windows advanced runtime distributions

- CPython 3.12.10 is automatically provisioned from the official Python NuGet
  package, SHA256 `0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8`.
  It retains Python Software Foundation and historical/component licences;
  the package's exact licence is also preserved at
  `licenses/CPython-3.12.10-LICENSE.txt`. This is runtime code, not model weights.
- The separate downstream SAM3 and DA3 `+modly.inference.1` wheels preserve all
  packaged upstream runtime bytes and upstream licences. Their distribution
  metadata describes only the extension's ordered-image inference path, not
  training, demo apps, direct video decoders or DA3 export extras. Their
  `.dist-info/provenance.json` records exact commits and source-file hashes;
  it is not presented as a VCS installation record.
- ANTLR4 Python runtime 4.9.3 retains its BSD-3-Clause terms. The upstream PyPI
  source distribution omits the licence file referenced by its runtime headers;
  the exact official `antlr/antlr4` tag `4.9.3` licence is therefore included at
  `licenses/ANTLR4-4.9.3-LICENSE.txt` alongside the faithfully built wheel.
- Windows runtime dependencies retain their own packaged licences. The immutable
  Windows lock records genuine wheel/source hashes. No inclusion changes the
  gated SAM3 checkpoint's custom terms or redistributes model weights.
- Triton Windows 3.3.1.post21, including its compiler binaries, is **not vendored
  or rehosted** by this extension. Setup obtains the genuine checksum-pinned
  upstream PyPI wheel directly. The wheel's compiler licence/notice completeness
  has not been established here; no compiler redistribution clearance is claimed.
  See the [upstream project](https://github.com/woct0rdho/triton-windows).
