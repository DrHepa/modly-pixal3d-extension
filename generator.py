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

from pixal3d_extension.paths import derive_modly_home


_LOADED_STATE = object()
_UI_MANAGED_ASSETS_MESSAGE = (
    "Pixal3D runtime downloads are disabled. Open Modly Models to download "
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

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return [
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

    def readiness_status(self) -> dict:
        ready = self.is_downloaded()
        return {
            "ok": ready,
            "machine_code": "ready" if ready else "weights_missing_or_unvalidated",
            "reason": "Pixal3D model assets and runtime validation are required before generation.",
        }

    def is_downloaded(self, root: str | Path = ".") -> bool:
        model_dir = self.model_dir or Path(root)
        return (Path(model_dir) / "pipeline.json").is_file()

    def _auto_download(self) -> None:
        raise RuntimeError(_UI_MANAGED_ASSETS_MESSAGE)

    def load(self) -> "Pixal3DGenerator":
        modly_home = derive_modly_home(model_dir=self.model_dir, workspace_dir=self.workspace_dir)
        workspace_root = modly_home or self.workspace_dir
        if workspace_root is not None:
            from pixal3d_extension.pipeline_patch import patch_pipeline

            patch_pipeline(workspace_root, auxiliary_mode="local", network_available=False)
        self._model = _LOADED_STATE
        return self

    def unload(self) -> None:
        super().unload()

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

        from pixal3d_extension.runtime import run_job

        image_or_job = image_bytes
        input_path: Path | None = None
        if isinstance(image_or_job, dict):
            job = dict(image_or_job)
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
                "model_source": str(self.model_dir) if self.model_dir is not None else None,
                "params": params or {},
                "readiness": {"generation_allowed": self.is_downloaded(), "code": "ready" if self.is_downloaded() else "weights_missing_or_unvalidated"},
            }

        try:
            job = _prepare_ui_managed_job(job, model_dir=self.model_dir)
            modly_home = derive_modly_home(
                model_dir=job.get("model_source"),
                workspace_dir=job.get("workspace_root") or job.get("output_dir") or self.workspace_dir,
            )
            if modly_home is not None:
                job["workspace_root"] = str(modly_home)

            result = run_job(job, pipeline_factory=self.pipeline_factory)
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
