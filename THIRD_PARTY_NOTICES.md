# Third-Party Notices

This repository's original Modly integration and clean-room scene geometry are
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

## Clean-room boundary

No code or structural adaptation was copied from the unlicensed community
`jtydhr88/ComfyUI-WorldSculpt` repository, including `nodes_realinput.py`.
The scene-preparation geometry was independently implemented from documented
input/output behavior and the official projects listed above.
