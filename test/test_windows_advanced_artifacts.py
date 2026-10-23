"""Artifact custody, not an assertion of native Windows execution."""
import base64
import csv
import hashlib
import io
import json
import unittest
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version

from pixal3d_extension.windows_scene_runtime import load_lock, EXPECTED

ROOT = Path(__file__).resolve().parents[1]


class WindowsAdvancedArtifactTests(unittest.TestCase):
    def test_source_wheels_are_complete_hashed_licensed_and_have_true_provenance(self):
        lock = load_lock(ROOT)
        commits = {'sam3':'2345a4ad109ac29c569da749c91d84f10dc08c40', 'depth-anything-3':'3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'}
        for item in lock['source_wheels']:
            path = ROOT/'wheels/scene-prep'/item['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])
            with ZipFile(path) as wheel:
                self.assertFalse(any(name.endswith('/direct_url.json') for name in wheel.namelist()))
                record = next(name for name in wheel.namelist() if name.endswith('.dist-info/RECORD'))
                for name, digest, size in csv.reader(io.StringIO(wheel.read(record).decode())):
                    if name == record:
                        self.assertEqual((digest, size), ('','')); continue
                    data = wheel.read(name)
                    self.assertEqual(int(size), len(data))
                    self.assertEqual(digest, 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode())
                provenance = json.loads(wheel.read(next(name for name in wheel.namelist() if name.endswith('/provenance.json'))))
                self.assertEqual(provenance['schema'], 'modly.inference-source-wheel.v1')
                self.assertEqual(provenance['source']['commit_id'], commits[item['name']])
                self.assertFalse(provenance['build']['runtime_source_modified'])
                self.assertTrue(any('/licenses/LICENSE' in name for name in wheel.namelist()))
                for source in provenance['source_files']:
                    data = wheel.read(source['wheel_path'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), source['sha256'])
                    self.assertEqual(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(), source['git_blob_sha1'])

    def test_windows_locked_metadata_dependency_closure(self):
        lock = load_lock(ROOT)
        pins = {item['name'].lower().replace('_','-'):item['version'] for item in lock['dependencies']+lock['source_wheels']}
        environment = {**default_environment(),'sys_platform':'win32','platform_system':'Windows','platform_machine':'AMD64',
                       'python_version':'3.12','python_full_version':'3.12.10','extra':''}
        for item in lock['dependencies']+lock['source_wheels']:
            for line in item.get('requires_dist', []):
                requirement = Requirement(line)
                if requirement.marker is not None and not requirement.marker.evaluate(environment):
                    continue
                name = requirement.name.lower().replace('_','-')
                self.assertIn(name, pins, (item['name'], line))
                self.assertIn(Version(pins[name]), requirement.specifier, (item['name'], line))
        self.assertEqual(pins['torch'], EXPECTED['torch'])
        self.assertEqual(pins['torchvision'], EXPECTED['torchvision'])
        self.assertEqual(pins['pywin32'], '311')
        self.assertNotIn('open3d', pins)
        self.assertNotIn('gsplat', pins)
        self.assertNotIn('xformers', pins)

    def test_python_runtime_lock_includes_headers_library_and_full_integrity_map(self):
        lock = load_lock(ROOT)
        runtime = lock['python_runtime']
        self.assertEqual(runtime['version'], [3,12,10])
        self.assertEqual(runtime['sha256'], '0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8')
        self.assertGreater(len(runtime['installed_files']), 1300)
        for name in ('include/Python.h','libs/python312.lib','python.exe','Lib/venv/__init__.py'):
            self.assertIn(name, runtime['installed_files'])
        self.assertEqual(lock['modly_host_runtime']['version'], [3,11,9])


if __name__ == '__main__':
    unittest.main()
