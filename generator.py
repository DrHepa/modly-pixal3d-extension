from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


_EXTENSION_ROOT = Path(__file__).resolve().parent
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except ModuleNotFoundError as exc:
    if exc.name not in {"services", "services.generators", "services.generators.base"}:
        raise

    class GenerationCancelled(Exception):
        """Standalone equivalent used only when Modly's generator API is absent."""

    class BaseGenerator:
        """Minimal standalone lifecycle contract for extension-local tooling."""

        def __init__(self, model_dir: Path | None, outputs_dir: Path | None) -> None:
            self.model_dir = model_dir
            self.outputs_dir = outputs_dir
            self._model: Any | None = None

        def is_loaded(self) -> bool:
            return self._model is not None

        def unload(self) -> None:
            self._model = None

        def _check_cancelled(self, cancel_event: Any | None) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled()


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
SCENE_IMAGES_NODE = "scene-from-images"
SCENE_VIDEO_NODE = "scene-from-video"
SCENE_NORMALIZE_NODE = "normalize-annotated-scene"
SCENE_PREP_NODES = {SCENE_ESTIMATE_NODE, SCENE_IMAGES_NODE, SCENE_VIDEO_NODE}


_LOADED_STATE = object()
_UI_MANAGED_ASSETS_MESSAGE = (
    "Pixal3D Hugging Face runtime downloads are disabled. Open Modly Models to download "
    "the Pixal3D weights, and use Repair on the Pixal3D extension if its "
    "setup or auxiliary assets are incomplete."
)
_UI_MANAGED_ASSET_FAILURE_CODES = {
    "missing_assets",
    "missing_auxiliary_assets",
    "missing_primary_assets",
    "weights_missing_or_unvalidated",
}


def _prepare_ui_managed_job(job: dict, *, model_dir: Path | None = None) -> dict:
    prepared = dict(job)
    params = dict(prepared.get("params") or {})
    for key in ("auxiliary_bootstrap_downloader", "auxiliary_mode", "network_available", "offline"):
        prepared.pop(key, None)
        params.pop(key, None)

    model_source = model_dir or prepared.get("model_source")
    if model_source is None:
        raise RuntimeError(_UI_MANAGED_ASSETS_MESSAGE)
    local_model_dir = Path(model_source).expanduser().resolve()

    prepared["model_source"] = str(local_model_dir)
    prepared["params"] = params
    prepared["auxiliary_mode"] = "local"
    prepared["network_available"] = False

    if not (local_model_dir / "pipeline.json").is_file():
        prepared["readiness"] = {
            "generation_allowed": False,
            "code": "weights_missing_or_unvalidated",
            "message": _UI_MANAGED_ASSETS_MESSAGE,
        }
    elif not isinstance(prepared.get("readiness"), dict):
        prepared["readiness"] = {"generation_allowed": True, "code": "ready"}
    return prepared


def _with_ui_managed_asset_guidance(result: dict) -> dict:
    if result.get("code") not in _UI_MANAGED_ASSET_FAILURE_CODES:
        return result
    detail = result.get("message")
    message = _UI_MANAGED_ASSETS_MESSAGE
    if detail:
        message = f"{message} Runtime preflight: {detail}"
    return {**result, "message": message}


class Pixal3DGenerator(BaseGenerator):
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
        resolved_model_dir = Path(model_dir) if model_dir is not None else None
        resolved_outputs_dir = Path(workspace_dir) if workspace_dir is not None else None
        super().__init__(resolved_model_dir, resolved_outputs_dir)
        self.workspace_dir = resolved_outputs_dir
        self.pipeline_factory = pipeline_factory
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
        # The extension's multi-source manifest requires Modly 0.4.3. Do not
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

    def _da3_source(self) -> Path:
        shared_dirs = getattr(self, "shared_model_dirs", None)
        if not isinstance(shared_dirs, dict) or DA3_GROUP not in shared_dirs:
            raise RuntimeError("Pixal3D MV automatic camera calibration requires the Modly shared group da3-base")
        return Path(shared_dirs[DA3_GROUP])

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
        if node_id in {None, "generate"}:
            from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET

            try:
                base = self._model_source()
                if base is None:
                    raise RuntimeError("the local Pixal3D base group is missing")
                required = [*PRIMARY_ASSET.sentinels]
                for key, asset in AUXILIARY_ASSETS.items():
                    if key != "naf":
                        required.extend(
                            str(Path(asset.local_root).relative_to(PRIMARY_ASSET.local_root) / filename)
                            for filename in asset.sentinels
                        )
                missing = [relative for relative in required if not (base / relative).is_file()]
                if missing:
                    raise RuntimeError(f"missing local files: {', '.join(missing)}")
                pipeline = json.loads((base / "pipeline.json").read_text(encoding="utf-8"))
                if not isinstance(pipeline, dict):
                    raise ValueError("pipeline.json must contain a JSON object")
                for key in ("image_cond_model", "rembg_model"):
                    reference = pipeline
                    for part in ("args", key, "args", "model_name"):
                        reference = reference.get(part) if isinstance(reference, dict) else None
                    if not isinstance(reference, str):
                        raise ValueError(f"pipeline.json is missing the {key} model reference")
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"missing_assets: {_UI_MANAGED_ASSETS_MESSAGE} {exc}") from exc
            return
        if node_id == "generate-mv":
            from pixal3d_extension.multiview import validate_mv_pipeline_config
            from pixal3d_extension.scene_prepare import validate_da3_weights
            from pixal3d_extension.scene_prepare_lane import validate_da3_runtime

            try:
                validate_mv_pipeline_config(self._model_source(), self._base_source())
                validate_da3_weights(self._da3_source())
                validate_da3_runtime(Path(__file__).resolve().parent)
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "mv_assets_missing: download or repair the Pixal3D base, MV, and DA3 Base shared groups in Modly Models UI: "
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
        self._check_cancelled(cancel_evt)

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
        if self._effective_node_id() in {*SCENE_PREP_NODES, SCENE_NORMALIZE_NODE, "generate-mv"}:
            manifest = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8"))
            node_id = self._effective_node_id()
            if node_id in {SCENE_ESTIMATE_NODE, SCENE_VIDEO_NODE}:
                # Keep the implemented video lane private until Modly releases
                # typed video model-input transport.
                schema = next(node["params_schema"] for node in manifest["nodes"] if node["id"] == SCENE_IMAGES_NODE)
                return [*schema,
                        {"id": "max_frames", "label": "Maximum Frames", "type": "int", "default": 16,
                         "min": 2, "max": 64, "tooltip": "Maximum ordered frames sent to SAM3 and DA3 Base."},
                        {"id": "frame_stride", "label": "Frame Stride", "type": "int", "default": 1,
                         "min": 1, "max": 120, "tooltip": "Deterministically keep every Nth frame before applying Maximum Frames."}]
            return next(node["params_schema"] for node in manifest["nodes"] if node["id"] == node_id)
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
        return schema

    def readiness_status(self) -> dict:
        if self._effective_node_id() == SCENE_NORMALIZE_NODE:
            return {"ok": True, "machine_code": "ready", "reason": "Annotated-scene normalization requires no model weights."}
        if self._effective_node_id() in SCENE_PREP_NODES:
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
            from pixal3d_extension.scene_prepare import validate_da3_weights
            from pixal3d_extension.scene_prepare_lane import da3_capability, validate_da3_runtime

            try:
                model_root = self._model_source()
                base_root = self._base_source()
                da3_root = self._da3_source()
            except RuntimeError as exc:
                return {"ok": False, "machine_code": "mv_shared_groups_unavailable", "reason": str(exc)}
            modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
            naf_path = (modly_home / "models/pixal3d/auxiliary/naf/naf_release.pth") if modly_home else Path("__missing_naf__")
            missing = missing_mv_assets(model_root, base_root, naf_path)
            support = da3_capability()
            if not support["supported"]:
                return {"ok": False, "machine_code": "mv_calibration_unsupported_platform", "reason": support["reason"]}
            try:
                validate_da3_weights(da3_root)
                validate_da3_runtime(Path(__file__).resolve().parent)
            except (OSError, ValueError, RuntimeError) as exc:
                missing.append(f"DA3 camera calibration: {exc}")
            runtime_ready = mv_runtime_available()
            return {"ok": not missing and runtime_ready,
                    "machine_code": "mv_assets_missing" if missing else "mv_runtime_missing" if not runtime_ready else "ready",
                    "reason": "Download the Pixal3D MV group and provision NAF before generation." if missing else
                    "An exact-stack Pixal3D MV Python wheel is required; the published wheelhouse is single-view only." if not runtime_ready else
                    "Multi-image assets, DA3 camera calibration, and Python module found; live GPU inference remains to be validated.",
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
        if self._effective_node_id() in SCENE_PREP_NODES:
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

    def _auto_download(self) -> None:
        raise RuntimeError(_UI_MANAGED_ASSETS_MESSAGE)

    def load(self) -> "Pixal3DGenerator":
        if self._effective_node_id() in {*SCENE_PREP_NODES, SCENE_NORMALIZE_NODE}:
            readiness = self.readiness_status()
            if not readiness["ok"]:
                raise RuntimeError(f"{readiness['machine_code']}: {readiness['reason']}")
            self._model = _LOADED_STATE
            return self
        if self._effective_node_id() == "worldsculpt":
            self._prepare_generation_assets()
            readiness = self.readiness_status()
            if not readiness["ok"]:
                raise RuntimeError(f"{readiness['machine_code']}: {readiness['reason']}")
            self._model = _LOADED_STATE
            return self
        model_source = self._model_source()
        if self._effective_node_id() == "generate-mv":
            self._prepare_generation_assets()
            readiness = self.readiness_status()
            if not readiness["ok"]:
                raise RuntimeError(f"{readiness['machine_code']}: {readiness['reason']}")
            self._model = _LOADED_STATE
            return self
        compatibility_error = self._single_view_compatibility_error()
        if compatibility_error is not None:
            raise RuntimeError(f"{compatibility_error['code']}: {compatibility_error['message']}")
        self._prepare_generation_assets()
        with shared_base_root(model_source if getattr(self, "shared_model_dirs", None) else None):
            modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
            if modly_home is not None or self.workspace_dir is not None:
                from pixal3d_extension.pipeline_patch import patch_pipeline

                patch_result = patch_pipeline(modly_home or self.workspace_dir, auxiliary_mode="local", network_available=False)
                if isinstance(patch_result, dict) and patch_result.get("status") != "patched":
                    raise RuntimeError(patch_result.get("message") or patch_result.get("code", "Pixal3D model assets are missing"))
        self._model = _LOADED_STATE
        return self

    def unload(self) -> None:
        super().unload()

    def _artifact_transport_path(self, kind: str, path: Any) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError(f"{kind} artifact path must be a str or pathlib.Path")
        artifact_path = Path(path).expanduser()
        if not str(artifact_path):
            raise ValueError(f"{kind} artifact path must not be empty")
        return artifact_path.resolve(strict=False)

    def _artifact_params(self, params: dict | None, reserved: set[str]) -> dict:
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise TypeError("artifact params must be a dict when provided")
        values = dict(params)
        unexpected = sorted(reserved.intersection(values))
        if unexpected:
            raise ValueError(f"Artifact reserved transport parameter is not allowed: {unexpected[0]}")
        return values

    def _scene_artifact_params(self, params: dict | None, artifact_path: Path) -> dict:
        values = self._artifact_params(
            params,
            {"extra_image_paths", "capture_manifest_path", "video_path", "input_image", "kind", "path"},
        )
        injected = values.get("scene_manifest_path")
        if injected is not None:
            injected_path = self._artifact_transport_path("scene_manifest_path", injected)
            if injected_path != artifact_path:
                raise ValueError("Artifact scene_manifest_path does not match the scene artifact path")
        values["scene_manifest_path"] = str(artifact_path)
        return values

    def generate_artifact(
        self,
        kind: str,
        path: str | Path,
        params: dict | None = None,
        progress_cb: Any | None = None,
        cancel_event: Any | None = None,
        **kwargs: Any,
    ) -> Path:
        """Dispatch upstream typed file artifacts into the existing node paths."""

        if "cancel_evt" in kwargs:
            if cancel_event is not None:
                raise TypeError("generate_artifact received both cancel_event and cancel_evt")
            cancel_event = kwargs.pop("cancel_evt")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"generate_artifact got unexpected keyword argument(s): {unexpected}")
        if not isinstance(kind, str):
            raise TypeError("artifact kind must be a string")
        artifact_kind = kind.strip().lower()
        if artifact_kind == "video":
            artifact_path = self._artifact_transport_path(artifact_kind, path)
            values = self._artifact_params(
                params,
                {"extra_image_paths", "capture_manifest_path", "scene_manifest_path", "video_path", "input_image", "kind", "path"},
            )
            return self.generate(
                {"kind": "video", "path": str(artifact_path)},
                values,
                progress_cb=progress_cb,
                cancel_evt=cancel_event,
            )
        if artifact_kind == "scene":
            artifact_path = self._artifact_transport_path(artifact_kind, path)
            values = self._scene_artifact_params(params, artifact_path)
            return self.generate(
                artifact_path,
                values,
                progress_cb=progress_cb,
                cancel_evt=cancel_event,
            )
        raise ValueError(f"Unsupported artifact kind for Pixal3D: {kind}")

    def generate(
        self,
        image_bytes: Any,
        params: dict | None = None,
        progress_cb: Any | None = None,
        cancel_event: Any | None = None,
        *,
        cancel_evt: Any | None = None,
    ) -> Path:
        if cancel_event is not None and cancel_evt is not None:
            raise TypeError("Pass either cancel_event or cancel_evt, not both")
        cancel_event = cancel_event if cancel_event is not None else cancel_evt
        self._check_cancelled(cancel_event)

        image_or_job = image_bytes
        cancel_evt = cancel_event
        if self._effective_node_id() == SCENE_IMAGES_NODE:
            from pixal3d_extension.scene_prepare import run_scene_from_images

            if self.workspace_dir is None:
                raise RuntimeError("Scene preparation requires the Modly workspace directory")
            if not isinstance(image_or_job, (bytes, bytearray)) or not image_or_job:
                raise ValueError("Scene from images requires a primary image and at least one additional connected image")
            values = dict(params or {})
            for reserved in ("capture_manifest_path", "scene_manifest_path", "video_path", "input_image", "primary_image_path", "num_views"):
                if reserved in values:
                    raise ValueError(f"Scene from images reserved transport parameter is not allowed: {reserved}")
            extra_image_paths = values.pop("extra_image_paths", None)
            if not isinstance(extra_image_paths, list):
                raise ValueError("Scene from images requires extra_image_paths from Modly's ordered multiple-image ports")
            sam_root, da3_root = self._scene_prep_sources()
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            return run_scene_from_images(
                primary_image_bytes=bytes(image_or_job), extra_image_paths=extra_image_paths,
                workspace_dir=self.workspace_dir, output_dir=output_dir,
                sam_root=sam_root, da3_root=da3_root, params=values,
                progress_cb=progress_cb, cancel_event=cancel_evt,
            )
        if self._effective_node_id() == SCENE_VIDEO_NODE:
            from pixal3d_extension.scene_prepare import run_scene_from_video

            if self.workspace_dir is None:
                raise RuntimeError("Scene preparation requires the Modly workspace directory")
            values = dict(params or {})
            reserved = {"extra_image_paths", "capture_manifest_path", "scene_manifest_path", "video_path", "input_image"}
            unexpected = sorted(reserved.intersection(values))
            if unexpected:
                raise ValueError(f"Scene from video reserved transport parameter is not allowed: {unexpected[0]}")
            sam_root, da3_root = self._scene_prep_sources()
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            return run_scene_from_video(
                video_input=image_or_job, workspace_dir=self.workspace_dir, output_dir=output_dir,
                sam_root=sam_root, da3_root=da3_root, params=values,
                progress_cb=progress_cb, cancel_event=cancel_evt,
            )
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
            from pixal3d_extension.multiview_images import run_multiview_from_images, validate_ordered_images

            if not isinstance(image_or_job, (bytes, bytearray)) or not image_or_job:
                raise ValueError("Pixal3D MV requires a primary image and at least one additional connected image")
            if self.workspace_dir is None:
                raise RuntimeError("Modly workspace directory is required for multi-image input")
            values = dict(params or {})
            for reserved in ("capture_manifest_path", "scene_manifest_path", "input_image", "primary_image_path", "num_views"):
                if reserved in values:
                    raise ValueError(f"Pixal3D MV reserved transport parameter is not allowed: {reserved}")
            extra_image_paths = values.pop("extra_image_paths", None)
            if not isinstance(extra_image_paths, list):
                raise ValueError("Pixal3D MV requires extra_image_paths from Modly's ordered multiple-image ports")
            self._raise_if_generation_cancelled(cancel_evt)
            validate_ordered_images(bytes(image_or_job), extra_image_paths, self.workspace_dir)
            mv_root = self._model_source()
            base_root = self._base_source()
            da3_root = self._da3_source()
            naf_path = self._prepare_generation_assets(cancel_evt)
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir / "Workflows"
            return run_multiview_from_images(
                primary_image_bytes=bytes(image_or_job), extra_image_paths=extra_image_paths,
                workspace_dir=self.workspace_dir, output_dir=output_dir, da3_root=da3_root,
                mv_root=mv_root, base_root=base_root, naf_path=naf_path, params=values,
                progress_cb=progress_cb, cancel_event=cancel_evt,
            )
        compatibility_error = self._single_view_compatibility_error()
        if compatibility_error is not None:
            raise RuntimeError(f"{compatibility_error['code']}: {compatibility_error['message']}")
        from pixal3d_extension.runtime import run_job

        self._prepare_generation_assets(cancel_event)
        model_source = self._model_source()
        input_path: Path | None = None

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
            temp_input = tempfile.NamedTemporaryFile(
                prefix=".pixal3d-input-", suffix=".png", dir=output_dir.parent, delete=False
            )
            input_path = Path(temp_input.name)
            temp_input.write(image_or_job)
            temp_input.close()
            job = {
                "input_image": str(input_path),
                "output_dir": str(output_dir),
                "model_source": str(model_source) if model_source is not None else None,
                "params": params or {},
                "readiness": {"generation_allowed": self.is_downloaded(), "code": "ready" if self.is_downloaded() else "weights_missing_or_unvalidated"},
            }

        try:
            job = _prepare_ui_managed_job(job, model_dir=model_source)
            modly_home = derive_modly_home(
                model_dir=job.get("model_source"),
                workspace_dir=job.get("workspace_root") or job.get("output_dir") or self.workspace_dir,
            )
            if modly_home is not None:
                job["workspace_root"] = str(modly_home)

            with shared_base_root(model_source if getattr(self, "shared_model_dirs", None) else None):
                result = run_job(job, pipeline_factory=self.pipeline_factory, cancel_event=cancel_event)
            result = _with_ui_managed_asset_guidance(result)
            if result.get("status") != "completed":
                raise RuntimeError(json.dumps(result, sort_keys=True))
            return Path(result["output"]["glb_path"])
        finally:
            if input_path is not None:
                try:
                    input_path.unlink(missing_ok=True)
                except Exception:
                    pass


def generate(job: dict, *, pipeline_factory: Callable[[str], Any] | None = None) -> dict:
    """Compatibility helper; the public Modly contract is Pixal3DGenerator."""

    from pixal3d_extension.runtime import run_job

    prepared = _prepare_ui_managed_job(job)
    return _with_ui_managed_asset_guidance(run_job(prepared, pipeline_factory=pipeline_factory))


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
