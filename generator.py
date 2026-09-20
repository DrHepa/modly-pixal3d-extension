from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from pixal3d_extension.assets import bootstrap_auxiliary_assets
from pixal3d_extension.naf_checkpoint import verify_naf_checkpoint
from pixal3d_extension.paths import derive_modly_home, shared_base_root


PIXAL3D_SOURCE = "TencentARC/Pixal3D"
_DINO_SOURCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_DINO_REPLACEMENT = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
_RMBG_SOURCE = "briaai/RMBG-2.0"
_RMBG_REPLACEMENT = "camenduru/RMBG-2.0"
SHARED_BASE_GROUP = "pixal3d-base"
SHARED_MV_GROUP = "pixal3d-mv"
SAM3_GROUP = "sam3"
DA3_GROUP = "da3-base"
SCENE_ESTIMATE_NODE = "scene-from-estimates"
SCENE_NORMALIZE_NODE = "normalize-annotated-scene"


def _patch_pipeline_json(model_dir: Path | None) -> None:
    if model_dir is None:
        return
    pipeline_path = model_dir / "pipeline.json"
    if not pipeline_path.is_file():
        return
    text = pipeline_path.read_text(encoding="utf-8")
    patched = text.replace(_DINO_SOURCE, _DINO_REPLACEMENT).replace(_RMBG_SOURCE, _RMBG_REPLACEMENT)
    if patched == text:
        return
    backup_path = model_dir / "pipeline.json.modly-original"
    if not backup_path.exists():
        backup_path.write_text(text, encoding="utf-8")
    pipeline_path.write_text(patched, encoding="utf-8")


class Pixal3DGenerator:
    """Root Modly model generator contract for Pixal3D.

    The class is intentionally defined in root ``generator.py`` because local
    Modly model extensions resolve ``manifest.json`` ``generator_class`` from
    this file. Heavy Pixal3D/HF/CUDA imports remain behind ``generate``.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        workspace_dir: str | Path | None = None,
        *,
        pipeline_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self.pipeline_factory = pipeline_factory
        self._loaded = False
        if self.workspace_dir is not None:
            # Configure process-global Hugging Face code-cache authority before
            # any single-view or MV path can import Transformers constants.
            from pixal3d_extension.multiview import configure_mv_hf_cache

            configure_mv_hf_cache(self.workspace_dir)

    def _effective_node_id(self) -> str | None:
        return getattr(self, "MODEL_NODE_ID", None) or getattr(self, "node_id", None)

    def _model_source(self) -> Path | None:
        shared_dirs = getattr(self, "shared_model_dirs", None)
        node_id = self._effective_node_id()
        if node_id == "worldsculpt":
            if isinstance(shared_dirs, dict) and "worldsculpt-adapters" in shared_dirs:
                return Path(shared_dirs["worldsculpt-adapters"])
            raise RuntimeError("Modly shared weight groups are required (worldsculpt-adapters); update Modly before using WorldSculpt")
        group = SHARED_MV_GROUP if node_id == "generate-mv" else SHARED_BASE_GROUP
        if isinstance(shared_dirs, dict) and group in shared_dirs:
            return Path(shared_dirs[group])
        # The extension's multi-source manifest requires PR #348. Do not
        # silently use stale private weights when the host has no shared root.
        if node_id in {"generate", "generate-mv"}:
            raise RuntimeError(f"Modly shared weight groups are required ({group}); update Modly before using this extension")
        return self.model_dir

    def _base_source(self) -> Path:
        shared_dirs = getattr(self, "shared_model_dirs", None)
        if not isinstance(shared_dirs, dict) or SHARED_BASE_GROUP not in shared_dirs:
            raise RuntimeError("Modly shared weight groups are required (pixal3d-base)")
        return Path(shared_dirs[SHARED_BASE_GROUP])

    def _scene_prep_sources(self) -> tuple[Path, Path]:
        shared_dirs = getattr(self, "shared_model_dirs", None)
        if not isinstance(shared_dirs, dict) or SAM3_GROUP not in shared_dirs or DA3_GROUP not in shared_dirs:
            raise RuntimeError("Scene preparation requires Modly shared groups sam3 and da3-base")
        return Path(shared_dirs[SAM3_GROUP]), Path(shared_dirs[DA3_GROUP])

    def _single_view_compatibility_error(self) -> dict | None:
        from pixal3d_extension.readiness import single_view_transformers_compatibility

        return single_view_transformers_compatibility()

    def _modly_home(self) -> Path:
        modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
        if modly_home is None:
            raise RuntimeError("Modly home could not be derived for the local NAF checkpoint")
        return modly_home

    def _naf_checkpoint_path(self) -> Path:
        modly_home = self._modly_home()
        return modly_home / "models/pixal3d/auxiliary/naf/naf_release.pth"

    def _preflight_non_naf_shared_assets(self) -> None:
        """Validate every host-managed shared asset before any NAF download."""

        node_id = self._effective_node_id()
        if node_id == "generate-mv":
            from pixal3d_extension.multiview import validate_mv_pipeline_config

            try:
                validate_mv_pipeline_config(self._model_source(), self._base_source())
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "mv_assets_missing: download or repair the Pixal3D base and MV shared groups in Modly Models UI: "
                    f"{exc}"
                ) from exc
            return
        if node_id == "worldsculpt":
            from pixal3d_extension.worldsculpt import validate_base
            from pixal3d_extension.worldsculpt_contract import validate_adapters

            try:
                validate_adapters(self._model_source())
                validate_base(self._base_source(), None)
            except (OSError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "worldsculpt_assets_missing: download or repair the Pixal3D base and WorldSculpt adapter "
                    f"shared groups in Modly Models UI: {exc}"
                ) from exc

    def _raise_if_generation_cancelled(self, cancel_evt: Any | None) -> None:
        if cancel_evt is not None and cancel_evt.is_set():
            label = "WorldSculpt" if self._effective_node_id() == "worldsculpt" else "Pixal3D MV generation"
            raise RuntimeError(f"{label} cancelled")

    def _prepare_generation_assets(self, cancel_evt: Any | None = None) -> Path:
        """Preflight shared assets, then bootstrap only the remaining NAF deficiency."""

        self._raise_if_generation_cancelled(cancel_evt)
        self._preflight_non_naf_shared_assets()
        self._raise_if_generation_cancelled(cancel_evt)
        checkpoint = self._ensure_naf_checkpoint()
        self._raise_if_generation_cancelled(cancel_evt)
        return checkpoint

    def _ensure_naf_checkpoint(self) -> Path:
        """Return the verified local NAF checkpoint, bootstrapping only when absent."""

        extension_dir = Path(__file__).resolve().parent
        manual = (
            "python3 setup.py --bootstrap-auxiliary-assets --force-auxiliary-assets "
            f"--workspace-root {extension_dir} --json"
        )
        try:
            modly_home = self._modly_home()
        except RuntimeError as exc:
            raise RuntimeError(
                "naf_bootstrap_failed: Modly home could not be derived for the local NAF checkpoint; "
                f"run {manual}"
            ) from exc
        checkpoint = modly_home / "models/pixal3d/auxiliary/naf/naf_release.pth"
        if checkpoint.is_file():
            try:
                verify_naf_checkpoint(checkpoint)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "naf_bootstrap_failed: the local NAF checkpoint is corrupt; automatic replacement is disabled; "
                    f"run {manual} ({type(exc).__name__}: {exc})"
                ) from exc
            return checkpoint

        try:
            result = bootstrap_auxiliary_assets(modly_home)
        except Exception as exc:
            raise RuntimeError(
                "naf_bootstrap_failed: NAF checkpoint download failed; "
                f"run {manual} ({type(exc).__name__}: {exc})"
            ) from exc
        if result.get("status") != "ready":
            detail = result.get("error") or result.get("message") or result.get("code") or "unknown bootstrap failure"
            raise RuntimeError(
                f"naf_bootstrap_failed: NAF checkpoint download failed: {detail}; run {manual}"
            )
        try:
            verify_naf_checkpoint(checkpoint)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "naf_bootstrap_failed: NAF bootstrap returned without a valid checkpoint; "
                f"run {manual} ({type(exc).__name__}: {exc})"
            ) from exc
        return checkpoint

    def params_schema(self) -> list[dict[str, Any]]:
        if self._effective_node_id() in {SCENE_ESTIMATE_NODE, SCENE_NORMALIZE_NODE}:
            manifest = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8"))
            return next(node["params_schema"] for node in manifest["nodes"] if node["id"] == self._effective_node_id())
        if self._effective_node_id() == "worldsculpt":
            return [{"id": "face_budget", "label": "Faces per Instance", "type": "int",
                     "default": 1000000, "min": 1000, "max": 3000000,
                     "tooltip": "Maximum faces per composed instance; geometry-only GLB."}]
        schema = [
            {
                "id": "resolution",
                "label": "Resolution",
                "type": "select",
                "default": 1024,
                "options": [
                    {"value": 1024, "label": "1024"},
                    {"value": 1536, "label": "1536"},
                ],
                "tooltip": "Generation resolution. Higher is slower and requires more VRAM.",
            },
            {
                "id": "low_vram",
                "label": "Low VRAM",
                "type": "select",
                "default": "low_vram",
                "options": [
                    {"value": "low_vram", "label": "Low VRAM"},
                    {"value": "standard", "label": "Standard"},
                ],
                "tooltip": "Prefer low-VRAM mode for safer Pixal3D generation; Standard loads all models on GPU.",
            },
            {
                "id": "manual_fov",
                "label": "Manual FOV",
                "type": "select",
                "default": "-1",
                "options": [
                    {"value": "-1", "label": "Auto (MoGe)"},
                    {"value": "0.2", "label": "0.2 rad"},
                    {"value": "0.35", "label": "0.35 rad"},
                    {"value": "0.5", "label": "0.5 rad"},
                ],
                "tooltip": "Auto uses MoGe camera estimation. Manual values skip MoGe and can help problematic inputs, but may cause perspective changes.",
            },
            {
                "id": "texture_size",
                "label": "Texture Size",
                "type": "select",
                "default": 1024,
                "options": [
                    {"value": 1024, "label": "1024"},
                    {"value": 2048, "label": "2048"},
                ],
                "tooltip": "Final GLB texture atlas size. 1024 reduces VRAM during final texturing; 2048 is higher quality and uses higher VRAM.",
            },
            {
                "id": "seed",
                "label": "Seed",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 4294967295,
                "tooltip": "Seed for reproducibility. -1 uses a random seed.",
            },
        ]
        if self._effective_node_id() == "generate-mv":
            schema = [item for item in schema if item["id"] not in {"manual_fov", "texture_size"}]
            schema.append({"id": "num_views", "label": "Views to Use", "type": "int", "default": 4, "min": 1, "max": 16,
                           "tooltip": "Use the first N ordered capture frames with calibrated cameras; frame 0 is the canonical front view."})
        return schema

    def readiness_status(self) -> dict:
        if self._effective_node_id() == SCENE_NORMALIZE_NODE:
            return {"ok": True, "machine_code": "ready", "reason": "Annotated-scene normalization requires no model weights."}
        if self._effective_node_id() == SCENE_ESTIMATE_NODE:
            from pixal3d_extension.scene_prepare import validate_scene_prepare_weights
            from pixal3d_extension.scene_prepare_lane import capability, validate_runtime

            support = capability()
            if not support["supported"]:
                return {"ok": False, "machine_code": "scene_prep_unsupported_platform", "reason": support["reason"]}
            try:
                sam_root, da3_root = self._scene_prep_sources()
                validate_scene_prepare_weights(sam_root, da3_root)
                validate_runtime(Path(__file__).resolve().parent)
            except (OSError, ValueError, RuntimeError) as exc:
                return {"ok": False, "machine_code": "scene_prep_not_ready", "reason": str(exc)}
            return {"ok": True, "machine_code": "ready", "reason": "Pinned local SAM3/DA3 assets and isolated Python 3.12 CUDA runtime are present; live inference remains untested."}
        if self._effective_node_id() == "worldsculpt":
            from pixal3d_extension.worldsculpt import missing_runtime, validate_base
            from pixal3d_extension.worldsculpt_contract import validate_adapters

            missing = missing_runtime()
            try:
                base = self._base_source()
                adapter = self._model_source()
                modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
                if modly_home is None:
                    raise RuntimeError("Modly home is required for the local NAF checkpoint")
                validate_adapters(adapter)
                validate_base(base, modly_home / "models/pixal3d/auxiliary/naf/naf_release.pth")
            except (OSError, ValueError, RuntimeError) as exc:
                return {"ok": False, "machine_code": "worldsculpt_assets_missing", "reason": str(exc), "missing_runtime": missing}
            if missing:
                return {"ok": False, "machine_code": "worldsculpt_runtime_missing",
                        "reason": "WorldSculpt requires exact pinned runtime dependencies and native kernels.", "missing_runtime": missing}
            return {"ok": True, "machine_code": "ready", "reason": "Local assets and imports present; GPU inference untested."}
        if self._effective_node_id() == "generate-mv":
            from pixal3d_extension.multiview import missing_mv_assets, mv_runtime_available

            try:
                model_root = self._model_source()
                base_root = self._base_source()
            except RuntimeError as exc:
                return {"ok": False, "machine_code": "mv_shared_groups_unavailable", "reason": str(exc)}
            modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
            naf_path = (modly_home / "models/pixal3d/auxiliary/naf/naf_release.pth") if modly_home else Path("__missing_naf__")
            missing = missing_mv_assets(model_root, base_root, naf_path)
            runtime_ready = mv_runtime_available()
            return {"ok": not missing and runtime_ready,
                    "machine_code": "mv_assets_missing" if missing else "mv_runtime_missing" if not runtime_ready else "ready",
                    "reason": "Download the Pixal3D MV group and provision NAF before generation." if missing else
                    "An exact-stack Pixal3D MV Python wheel is required; the published wheelhouse is single-view only." if not runtime_ready else
                    "Posed-view assets and Python module found; live GPU inference remains to be validated.",
                    "missing": missing}
        compatibility_error = self._single_view_compatibility_error()
        if compatibility_error is not None:
            return {"ok": False, "machine_code": compatibility_error["code"], "reason": compatibility_error["message"]}
        ready = self.is_downloaded()
        return {
            "ok": ready,
            "machine_code": "ready" if ready else "weights_missing_or_unvalidated",
            "reason": "Pixal3D model assets and runtime validation are required before generation.",
        }

    def is_downloaded(self, root: str | Path = ".") -> bool:
        if self._effective_node_id() == SCENE_NORMALIZE_NODE:
            return True
        if self._effective_node_id() == SCENE_ESTIMATE_NODE:
            try:
                from pixal3d_extension.scene_prepare import validate_scene_prepare_weights

                validate_scene_prepare_weights(*self._scene_prep_sources())
                return True
            except (OSError, ValueError, RuntimeError):
                return False
        model_dir = self._model_source() or Path(root)
        if self._effective_node_id() == "worldsculpt":
            return self.readiness_status()["ok"]
        if self._effective_node_id() == "generate-mv":
            from pixal3d_extension.multiview import MV_WEIGHT_FILES

            return all((Path(model_dir) / relative).is_file() for relative in MV_WEIGHT_FILES)
        return (Path(model_dir) / "pipeline.json").is_file()

    def load(self) -> "Pixal3DGenerator":
        if self._effective_node_id() in {SCENE_ESTIMATE_NODE, SCENE_NORMALIZE_NODE}:
            readiness = self.readiness_status()
            if not readiness["ok"]:
                raise RuntimeError(f"{readiness['machine_code']}: {readiness['reason']}")
            self._loaded = True
            return self
        if self._effective_node_id() == "worldsculpt":
            self._prepare_generation_assets()
            readiness = self.readiness_status()
            if not readiness["ok"]:
                raise RuntimeError(f"{readiness['machine_code']}: {readiness['reason']}")
            self._loaded = True
            return self
        model_source = self._model_source()
        if self._effective_node_id() == "generate-mv":
            self._prepare_generation_assets()
            readiness = self.readiness_status()
            if not readiness["ok"]:
                raise RuntimeError(f"{readiness['machine_code']}: {readiness['reason']}")
            self._loaded = True
            return self
        compatibility_error = self._single_view_compatibility_error()
        if compatibility_error is not None:
            raise RuntimeError(f"{compatibility_error['code']}: {compatibility_error['message']}")
        with shared_base_root(model_source if getattr(self, "shared_model_dirs", None) else None):
            modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
            if modly_home is not None or self.workspace_dir is not None:
                from pixal3d_extension.pipeline_patch import patch_pipeline

                patch_result = patch_pipeline(modly_home or self.workspace_dir, auxiliary_mode="default", network_available=True)
                if isinstance(patch_result, dict) and patch_result.get("status") != "patched":
                    raise RuntimeError(patch_result.get("message") or patch_result.get("code", "Pixal3D model assets are missing"))
            else:
                _patch_pipeline_json(model_source)
        self._loaded = True
        return self

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, image_or_job: Any, params: dict | None = None, progress_cb: Any | None = None, cancel_evt: Any | None = None) -> Path:
        if self._effective_node_id() == SCENE_ESTIMATE_NODE:
            from pixal3d_extension.scene_prepare import run_scene_from_estimates

            if self.workspace_dir is None:
                raise RuntimeError("Scene preparation requires the Modly workspace directory")
            sam_root, da3_root = self._scene_prep_sources()
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            return run_scene_from_estimates(
                capture_input=image_or_job,
                workspace_dir=self.workspace_dir,
                output_dir=output_dir,
                sam_root=sam_root,
                da3_root=da3_root,
                params=params or {},
                progress_cb=progress_cb,
                cancel_event=cancel_evt,
            )
        if self._effective_node_id() == SCENE_NORMALIZE_NODE:
            from pixal3d_extension.scene_prepare_contract import normalize_annotated_scene

            if self.workspace_dir is None:
                raise RuntimeError("Scene normalization requires the Modly workspace directory")
            if isinstance(image_or_job, (bytes, bytearray)):
                raise ValueError("Scene normalization requires a scene manifest, not image bytes")
            manifest_path = (params or {}).get("scene_manifest_path")
            if not isinstance(manifest_path, str):
                manifest_path = str(getattr(image_or_job, "path", image_or_job))
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            if progress_cb:
                progress_cb(10, "Validating annotated scene")
            result = normalize_annotated_scene(Path(manifest_path), self.workspace_dir, output_dir)
            if progress_cb:
                progress_cb(100, "Annotated scene normalized")
            return result
        if self._effective_node_id() == "worldsculpt":
            from pixal3d_extension.worldsculpt import resolve_scene_manifest, run_worldsculpt

            if isinstance(image_or_job, (bytes, bytearray)) and image_or_job:
                raise ValueError("WorldSculpt requires a scene manifest, not image bytes")
            if not isinstance(params, dict) or not params.get("scene_manifest_path"):
                raise ValueError("WorldSculpt requires scene_manifest_path from Modly /from-scene")
            if self.workspace_dir is None:
                raise RuntimeError("WorldSculpt requires the Modly workspace directory")
            self._raise_if_generation_cancelled(cancel_evt)
            # Reject an untrusted scene envelope before shared-asset checks or
            # first-run NAF bootstrap can mutate local state. The later resolve
            # remains intentional defense in depth at the runner boundary.
            resolve_scene_manifest(params["scene_manifest_path"], self.workspace_dir)
            adapter_root = self._model_source()
            base_root = self._base_source()
            naf_path = self._prepare_generation_assets(cancel_evt)
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            return run_worldsculpt(
                scene_dir=resolve_scene_manifest(params["scene_manifest_path"], self.workspace_dir),
                adapter_root=adapter_root, base_root=base_root,
                naf_path=naf_path,
                output_dir=output_dir, workspace_dir=self.workspace_dir,
                face_budget=params.get("face_budget", 1000000),
                progress_cb=progress_cb, cancel_event=cancel_evt)
        if self._effective_node_id() == "generate-mv":
            from pixal3d_extension.multiview import run_multiview

            if isinstance(image_or_job, (bytes, bytearray)):
                raise ValueError("Pixal3D MV requires a typed capture manifest, not image bytes; use Modly /from-artifact")
            if getattr(image_or_job, "kind", "capture") != "capture":
                raise ValueError("Pixal3D MV requires capture input, not a scene manifest")
            capture_path = getattr(image_or_job, "path", image_or_job)
            if not isinstance(capture_path, (str, Path)):
                raise ValueError("Pixal3D MV requires capture_manifest_path as its typed input")
            capture_path = Path(capture_path)
            if capture_path.name != "capture-manifest.json":
                raise ValueError("Pixal3D MV requires capture-manifest.json; migrate legacy posed scenes to calibrated captures")
            if self.workspace_dir is None:
                raise RuntimeError("Modly workspace directory is required for capture input")
            self._raise_if_generation_cancelled(cancel_evt)
            # Match run_multiview's parameter normalization, then validate the
            # complete capture without creating outputs before any bootstrap.
            from pixal3d_extension.multiview_capture import validate_mv_capture

            num_views = int((params or {}).get("num_views", 4))
            validate_mv_capture(capture_path, self.workspace_dir, num_views)
            mv_root = self._model_source()
            base_root = self._base_source()
            naf_path = self._prepare_generation_assets(cancel_evt)
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            return run_multiview(capture_manifest_path=capture_path, workspace_dir=self.workspace_dir,
                                 mv_root=mv_root, base_root=base_root,
                                 naf_path=naf_path,
                                 output_dir=output_dir, params=params or {},
                                 progress_cb=progress_cb, cancel_event=cancel_evt)
        compatibility_error = self._single_view_compatibility_error()
        if compatibility_error is not None:
            raise RuntimeError(f"{compatibility_error['code']}: {compatibility_error['message']}")
        from pixal3d_extension.runtime import run_job

        model_source = self._model_source()

        if isinstance(image_or_job, dict):
            job = dict(image_or_job)
            if model_source is not None:
                job["model_source"] = str(model_source)
            modly_home = derive_modly_home(
                model_dir=job.get("model_source") or self.model_dir,
                workspace_dir=job.get("workspace_root") or job.get("output_dir") or self.workspace_dir,
            )
            if modly_home is not None:
                job["workspace_root"] = str(modly_home)
        else:
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir
            if output_dir is None:
                raise RuntimeError("Pixal3D output directory is not configured")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_input = tempfile.NamedTemporaryFile(prefix="pixal3d-input-", suffix=".png", dir=output_dir, delete=False)
            input_path = Path(temp_input.name)
            temp_input.write(image_or_job)
            temp_input.close()
            job = {
                "input_image": str(input_path),
                "output_dir": str(output_dir),
                "model_source": str(model_source or PIXAL3D_SOURCE),
                "params": params or {},
                "readiness": {"generation_allowed": self.is_downloaded(), "code": "ready" if self.is_downloaded() else "weights_missing_or_unvalidated"},
            }
            modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir or output_dir)
            if modly_home is not None:
                job["workspace_root"] = str(modly_home)

        try:
            with shared_base_root(model_source if getattr(self, "shared_model_dirs", None) else None):
                result = run_job(job, pipeline_factory=self.pipeline_factory)
            if result.get("status") != "completed":
                raise RuntimeError(json.dumps(result, sort_keys=True))
            return Path(result["output"]["glb_path"])
        finally:
            if not isinstance(image_or_job, dict):
                try:
                    input_path.unlink(missing_ok=True)
                except Exception:
                    pass


def generate(job: dict, *, pipeline_factory: Callable[[str], Any] | None = None) -> dict:
    """Compatibility helper; the public Modly contract is Pixal3DGenerator."""

    from pixal3d_extension.runtime import run_job

    return run_job(job, pipeline_factory=pipeline_factory)


def main() -> None:
    print(
        json.dumps(
            {
                "status": "blocked",
                "code": "job_payload_required",
                "message": "Call generate(job) from Modly with an explicit job payload; CLI generation is not run by default.",
                "generation_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
