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

    def __init__(self, *, pipeline_factory: Callable[[str], Any] | None = None) -> None:
        self.pipeline_factory = pipeline_factory
        self._loaded = False

    def is_downloaded(self, root: str | Path = ".") -> bool:
        from pixal3d_extension.readiness import SUPPORTED_RUNTIME_LANE, check_readiness

        result = check_readiness(root, runtime_lane=SUPPORTED_RUNTIME_LANE, runtime_validated=True)
        return result.get("generation_allowed") is True

    def load(self) -> "Pixal3DGenerator":
        self._loaded = True
        return self

    def unload(self) -> None:
        self._loaded = False

    def generate(self, job: dict) -> Path:
        from pixal3d_extension.runtime import run_job

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
