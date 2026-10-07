"""Bounded, weights-free Blackwell base checks; CUDA imports happen only in the venv child."""

import importlib
import json
import sys
from importlib.metadata import version

from pixal3d_extension.readiness import TRANSFORMERS_VERSION

CHECK_NAMES = ("torch_stack", "native_imports", "natten_lib", "device", "torch_cuda", "natten_sm120")


def run_blackwell_probe(policy):
    payload = {"ok": False, "runtime_prepared": False, "inference_validated": False,
               "checks": {name: {"status": "NOT_RUN"} for name in CHECK_NAMES},
               "imports": [], "upstream_imports": []}
    current = "torch_stack"

    def begin(name):
        nonlocal current
        current = name
        payload["checks"][name]["status"] = "RUNNING"
        # Preserve completed checks even if a later native import/kernel hangs or crashes.
        print(json.dumps(payload, sort_keys=True), flush=True)

    def passed():
        payload["checks"][current]["status"] = "PASS"

    try:
        begin("torch_stack")
        import torch
        payload.update(torch_version=torch.__version__, torchvision_version=version("torchvision"),
                       torch_cuda_version=torch.version.cuda, transformers_version=version("transformers"),
                       torch_cuda_available=bool(torch.cuda.is_available()))
        for key, expected in (("torch_version", policy["torch"]), ("torchvision_version", policy["torchvision"]),
                              ("torch_cuda_version", policy["expected_torch_cuda"]), ("transformers_version", TRANSFORMERS_VERSION)):
            if payload[key] != expected:
                raise RuntimeError(f"{key} must be {expected}; found {payload[key]!r}")
        passed()
        begin("native_imports")
        for name in policy["required_imports"]:
            original_compile = torch.compile
            try:
                if name == "natten":
                    torch.compile = lambda function=None, *args, **kwargs: function if function is not None else lambda inner: inner
                importlib.import_module(name)
                payload["imports"].append(name)
            finally:
                torch.compile = original_compile
        for upstream, native in (("cumesh", "cumesh_vb"), ("flex_gemm", "flex_gemm_ap"), ("o_voxel", "o_voxel_vb_ap")):
            sys.modules.setdefault(upstream, importlib.import_module(native))
        for name in policy["required_upstream_imports"]:
            importlib.import_module(name)
            payload["upstream_imports"].append(name)
        passed()
        begin("natten_lib")
        natten = sys.modules["natten"]
        payload.update(natten_version=version("natten"), natten_has_libnatten=bool(natten.HAS_LIBNATTEN))
        if payload["natten_version"] != policy["natten"] or not payload["natten_has_libnatten"]:
            raise RuntimeError("Exact native NATTEN/libnatten is required")
        passed()
        begin("device")
        if not payload["torch_cuda_available"]:
            raise RuntimeError("CUDA GPU/driver unavailable; hardware checks BLOCKED")
        major, minor = torch.cuda.get_device_capability()
        payload.update(gpu_sm=f"{major}{minor}", gpu_name=torch.cuda.get_device_name())
        if payload["gpu_sm"] != policy["required_gpu_sm"]:
            raise RuntimeError(f"Expected SM{policy['required_gpu_sm']}; found SM{payload['gpu_sm']}")
        passed()
        begin("torch_cuda")
        tensor = torch.ones((8,), device="cuda", dtype=torch.float32)
        output = tensor + tensor
        torch.cuda.synchronize()
        if not torch.equal(output.cpu(), torch.full((8,), 2.0)):
            raise RuntimeError("Tiny synchronized CUDA operation returned incorrect data")
        passed()
        begin("natten_sm120")
        # Genuine r3 NAF uses this public backend, not blackwell-fna (CC100/103 only).
        q = torch.zeros((1, 8, 8, 1, 32), device="cuda", dtype=torch.float16)
        v = torch.ones_like(q)
        output = natten.na2d(q, q, v, kernel_size=3, stride=1, backend="cutlass-fna")
        torch.cuda.synchronize()
        if output.shape != q.shape or not torch.allclose(output.cpu(), torch.ones_like(output.cpu()), atol=0.002, rtol=0.002):
            raise RuntimeError("NATTEN SM120 kernel returned incorrect data")
        payload["checks"][current]["backend"] = "cutlass-fna"
        passed()
        payload.update(ok=True, runtime_prepared=True)
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["checks"][current].update(status="FAIL", error=payload["error"])
    return payload
