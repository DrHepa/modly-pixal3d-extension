# WorldSculpt integration status

This directory stages the 118 UTF-8 source, license, and notice blobs from
[`AlayaLab/WorldSculpt` commit `373f09f0ecddc94607b569bea0316ff6a2286501`](https://github.com/AlayaLab/WorldSculpt/tree/373f09f0ecddc94607b569bea0316ff6a2286501).
`EXPECTED_GIT_BLOBS` records each original Git blob SHA and can be checked with
`git hash-object`. Images and non-inference assets are intentionally not staged.
`LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, and `pixal3d/LICENSE` are retained;
the original AlayaLab code is Apache-2.0 while `pixal3d/` is MIT. Model weights
are not included and have separate terms.

The branch candidate declares a `worldsculpt` scene-to-mesh node in
`manifest.json`, with UI-managed shared `worldsculpt-adapters` weights pinned to
`AlayaLab/WorldSculpt` revision `8cb81056d803c61371dd84ef18a14142a738610e`
and a dependency on the shared `pixal3d-base` group. The runner in
`pixal3d_extension/worldsculpt.py` has not passed an actual GPU crop →
reconstruct → compose inference run or installed Modly UI E2E. The declaration
is a download/dispatch candidate, not a release or functional-compatibility
claim. Its companion
`pixal3d_extension/worldsculpt_contract.py` validates metric scenes, the two
distinct local LoRA adapter directories, crops, and generated mesh/GLB artifacts.
In particular,
`reconstruct_batch.py` catches per-instance exceptions and `compose_scene.py`
catches GLB export errors; process exit 0 alone is not proof of a scene.

The candidate runner creates a private source/config overlay, points
`pretrained/Pixal3D` at the supplied shared base, localizes DINOv3 and NAF,
forces local-only model loaders, and passes an eligible-instance allowlist to
reconstruction and composition. The crop and reconstruction gates reject extra
instance directories. The branch has an isolated, hash-locked WorldSculpt
dependency lane and the pinned shared adapter declaration, but still needs
installed Modly UI E2E and a real crop → reconstruct → compose job with a
loadable `scene.glb` before release or functional compatibility can be claimed. The existing
Pixal3D posed-view node is distinct and is not a WorldSculpt substitute.

The local contract now has two explicit execution gates: `validate_scene` checks
actual PNG decoding, the published 500-pixel / 0.001 mask-ratio threshold, rigid
camera matrices, and a projected crop candidate; `validate_crops` must run after
`prepare_crops_scene.py` and before reconstruction to prove saved RGBA crops and
camera metadata. `prepare_case_root` creates an exclusive private output root;
`validate_output` rejects stale/symlinked files, invalid `mesh.pt` tensors, and
GLBs that do not contain geometry for every eligible instance. The preflight
projection currently covers only the world-AABB path; OBB metadata is rejected
rather than guessed. These contracts are wired to the advertised branch-candidate
node, but their GPU path is not yet validated end to end.

The staged pinned source manifest does not contain
`pixal3d/modules/sparse/serialize.py`, although `pixal3d/modules/sparse/__init__.py`
lazily imports it for `encode_seq` and `decode_seq`. The missing file's pinned
upstream existence/content and runtime import closure remain **UNTESTED**; do
not promote the advertised candidate to a release until this is resolved from
a verified source and the real GPU path is validated.
