from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable


PIXAL3D_SOURCE = "TencentARC/Pixal3D"
_DINO_SOURCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_DINO_REPLACEMENT = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
_RMBG_SOURCE = "briaai/RMBG-2.0"
_RMBG_REPLACEMENT = "camenduru/RMBG-2.0"


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

    def load(self) -> "Pixal3DGenerator":
        _patch_pipeline_json(self.model_dir)
        self._loaded = True
        return self

    def unload(self) -> None:
        self._loaded = False

    def generate(self, image_or_job: Any, params: dict | None = None, progress_cb: Any | None = None, cancel_evt: Any | None = None) -> Path:
        from pixal3d_extension.runtime import run_job

        if isinstance(image_or_job, dict):
            job = image_or_job
        else:
            output_dir = getattr(self, "outputs_dir", None) or self.workspace_dir
            if output_dir is None:
                raise RuntimeError("Pixal3D output directory is not configured")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_input = tempfile.NamedTemporaryFile(prefix="pixal3d-input-", suffix=".png", delete=False)
            input_path = Path(temp_input.name)
            temp_input.write(image_or_job)
            temp_input.close()
            job = {
                "input_image": str(input_path),
                "output_dir": str(output_dir),
                "model_source": str(self.model_dir or PIXAL3D_SOURCE),
                "params": params or {},
                "readiness": {"generation_allowed": self.is_downloaded(), "code": "ready" if self.is_downloaded() else "weights_missing_or_unvalidated"},
            }

        try:
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
