import base64
import csv
import hashlib
import importlib.util
import io
import unittest
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement

import setup

from tools.validation.windows_dependency_install import classify_probe


ROOT = Path(__file__).resolve().parents[1]
METADATA = "pixal3d_core-0.1.0+modly.dist-info/METADATA"
RECORD = "pixal3d_core-0.1.0+modly.dist-info/RECORD"
WINDOWS = {"o-voxel": "o-voxel-vb-ap==0.0.1", "cumesh": "cumesh-vb==1.0",
           "flex-gemm": "flex-gemm-ap==1.0.0", "nvdiffrec-render": "nvdiffrec-render==0.0.1"}


def requirements(data):
    return BytesParser().parsebytes(data).get_all("Requires-Dist")


def active_requirements(data, system, python):
    environment = {**default_environment(), "platform_system": system,
                   "sys_platform": "win32" if system == "Windows" else "linux",
                   "python_version": python, "python_full_version": python + ".9"}
    active = []
    for line in requirements(data):
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            active.append(requirement.name + str(requirement.specifier))
    return sorted(active)


class MvWheelMetadataTests(unittest.TestCase):
    def setUp(self):
        with ZipFile(ROOT / "wheels/pixal3d_core-0.1.0+modly-py3-none-any.whl") as archive:
            self.base = archive.read(METADATA)
        with ZipFile(ROOT / setup.MV_CORE_WHEEL) as archive:
            self.overlay = archive.read(METADATA)

    def test_platform_markers_preserve_linux_and_select_real_windows_distributions(self):
        linux = active_requirements(self.base, "Linux", "3.11")
        windows = sorted(WINDOWS.get(Requirement(line).name, line) for line in requirements(self.base))
        for python in ("3.11", "3.12"):
            with self.subTest(python=python):
                self.assertEqual(active_requirements(self.overlay, "Linux", python), linux)
                self.assertEqual(active_requirements(self.overlay, "Windows", python), windows)
                self.assertIn("natten==0.21.0", active_requirements(self.overlay, "Windows", python))
        untouched = [line for line in requirements(self.base) if Requirement(line).name not in WINDOWS]
        self.assertEqual([line for line in requirements(self.overlay)
                          if Requirement(line).name not in {*WINDOWS, *[Requirement(value).name for value in WINDOWS.values()]}], untouched)

    def test_builder_produces_the_bundled_metadata_without_other_changes(self):
        spec = importlib.util.spec_from_file_location("mv_builder", ROOT / "tools/wheelhouse/build-pixal3d-mv-python-wheel.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.assertEqual(builder.platform_metadata(self.base), self.overlay)
        for line in self.base.splitlines():
            if not any(line.startswith(("Requires-Dist: " + name + "==").encode()) for name in WINDOWS):
                self.assertIn(line, self.overlay.splitlines())
        with self.assertRaisesRegex(ValueError, "dependency"):
            builder.platform_metadata(self.base.replace(b"cumesh==0.0.1", b"cumesh==9.9"))

    def test_wheel_digest_and_every_record_member(self):
        wheel = ROOT / setup.MV_CORE_WHEEL
        self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), setup.MV_CORE_WHEEL_SHA256)
        with ZipFile(wheel) as archive:
            rows = list(csv.reader(io.StringIO(archive.read(RECORD).decode())))
            self.assertEqual(len(archive.namelist()), 105)
            self.assertEqual(len(rows), 105)
            self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
            for name, encoded, size in rows:
                if name == RECORD:
                    self.assertEqual((encoded, size), ("", ""))
                    continue
                data = archive.read(name)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                self.assertEqual(encoded, "sha256=" + digest, name)
                self.assertEqual(size, str(len(data)), name)

    def test_cp312_without_natten_remains_an_unqualified_graph(self):
        manifest = setup.load_manifest(ROOT / "wheelhouse.manifest.json")
        asset = next(item for item in manifest["assets"] if item["id"] == "windows-x64-cp312-cuda124")
        self.assertNotIn("natten", asset["packages"])
        self.assertIn("natten==0.21.0", active_requirements(self.overlay, "Windows", "3.12"))

    def test_hosted_probe_does_not_turn_hardware_blockers_into_import_passes(self):
        probe = {"torch_version": "2.6.0+cu124", "torchvision_version": "0.21.0+cu124",
                 "torch_cuda_version": "12.4", "torch_cuda_available": False,
                 "error": "RuntimeError: Found no NVIDIA driver on your system"}
        self.assertEqual(classify_probe(probe), "BLOCKED")
        for changed in ({"error": "ImportError: DLL load failed"}, {"torch_cuda_version": "12.8"},
                        {"torch_cuda_available": True}, {"error": None}):
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                classify_probe({**probe, **changed})
        imported = {**probe, "error": None, "imports": ["cumesh_vb", "flex_gemm_ap", "o_voxel_vb_ap",
                    "nvdiffrast", "nvdiffrec_render", "natten"], "upstream_imports": ["cumesh", "flex_gemm", "o_voxel"],
                    "natten_version": "0.21.0", "natten_has_libnatten": True}
        self.assertEqual(classify_probe(imported), "PASS")
        with self.assertRaises(AssertionError):
            classify_probe({**imported, "natten_has_libnatten": False})


if __name__ == "__main__":
    unittest.main()
