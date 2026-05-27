from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


PIXAL3D_SOURCE = "TencentARC/Pixal3D"


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
    def params_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "seed": {"type": ["integer", "null"], "default": None},
                "resolution": {"type": "integer", "enum": [1024, 1536], "default": 1024},
                "low_vram": {"type": "boolean", "default": False},
                "manual_fov": {"type": ["number", "null"], "default": None},
            },
            "additionalProperties": False,
        }

    def readiness_status(self) -> dict:
        root = self.model_dir or "."
        ready = self.is_downloaded(root)
        return {
            "ok": ready,
            "machine_code": "ready" if ready else "weights_missing_or_unvalidated",
            "label_hint": "Ready" if ready else "Waiting for model assets",
            "reason": "Pixal3D model assets and runtime validation are required before generation.",
        }

    def is_downloaded(self, root: str | Path = ".") -> bool:
        from pixal3d_extension.readiness import SUPPORTED_RUNTIME_LANE, check_readiness

        result = check_readiness(root, runtime_lane=SUPPORTED_RUNTIME_LANE, runtime_validated=True)
        return result.get("generation_allowed") is True

    def load(self) -> "Pixal3DGenerator":
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
            input_path = output_dir / "input.png"
            input_path.write_bytes(image_or_job)
            job = {
                "input_image": str(input_path),
                "output_dir": str(output_dir),
                "params": params or {},
                "readiness": {"generation_allowed": False, "code": "weights_missing_or_unvalidated"},
            }

        result = run_job(job, pipeline_factory=self.pipeline_factory)
        if result.get("status") != "completed":
            raise RuntimeError(json.dumps(result, sort_keys=True))
        return Path(result["output"]["glb_path"])


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
