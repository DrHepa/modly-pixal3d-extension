import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import worldsculpt_lane as lane


class WorldSculptLaneTests(unittest.TestCase):
    def test_locked_wheels_and_tamper(self):
        self.assertEqual(len(lane.verify_wheels()), 9)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = json.loads(lane.MANIFEST.read_text())
            first = manifest['wheels'][0]
            (folder / first['file']).write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                lane.verify_wheels(lane.MANIFEST, folder)

    def test_platform_gate_rejects_abi_drift(self):
        valid = {'machine': 'aarch64', 'python': [3, 12], 'torch': '2.12.0+cu130', 'cuda': '13.0'}
        with patch.object(lane.sys, 'platform', 'linux'), patch.object(lane.platform, 'machine', return_value='aarch64'):
            lane._platform_gate(valid)
            for key, bad in [('python', [3, 11]), ('torch', '2.12.0'), ('cuda', '12.4')]:
                with self.assertRaisesRegex(RuntimeError, 'supported only'):
                    lane._platform_gate({**valid, key: bad})
            with patch.object(lane.platform, 'machine', return_value='x86_64'):
                with self.assertRaises(RuntimeError):
                    lane._platform_gate(valid)

    def test_interpreter_selection_is_separate(self):
        self.assertEqual(lane.python_path(), lane.ROOT / 'venv-worldsculpt/bin/python')
        self.assertNotEqual(lane.python_path(), lane.ROOT / 'venv/bin/python')

    def test_rejects_foreign_interpreter_and_site_symlink_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'venv-worldsculpt'
            site = target / 'lib/python3.12/site-packages'
            (target / 'bin').mkdir(parents=True)
            site.mkdir(parents=True)
            (target / 'pyvenv.cfg').write_text('home = /usr/bin\n')
            (target / 'bin/python').write_bytes(b'foreign')
            with patch.object(lane, '_probe', return_value={'machine': 'aarch64', 'python': [3, 12], 'torch': '2.12.0+cu130', 'cuda': '13.0', 'site': str(root / 'venv/lib/python3.12/site-packages')}), patch.object(lane, '_platform_gate'), patch.object(lane, 'verify_wheels', return_value=[]):
                (root / 'venv/bin').mkdir(parents=True)
                (root / 'venv/bin/python').write_bytes(b'primary')
                (root / 'venv/lib/python3.12/site-packages').mkdir(parents=True)
                with self.assertRaises(Exception):
                    lane.repair(root)
                self.assertFalse((site / 'primary-pixal3d.pth').exists())
            (target / 'bin/python').unlink()
            (target / 'bin/python').symlink_to(root / 'venv/bin/python')
            with self.assertRaisesRegex(RuntimeError, 'provenance/path conflict'):
                lane._validate_target(target)

    def test_installed_overlay_rejects_inherited_distributions(self):
        with self.assertRaisesRegex(RuntimeError, 'overlay verification failed'):
            lane._verify_overlay(lane.ROOT / 'venv-worldsculpt/bin/python', lane.verify_wheels(), Path('/tmp/foreign-worldsculpt-site'))

    def test_skip_install_skips_worldsculpt_repair_and_reports_it(self):
        import setup
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'venv/bin').mkdir(parents=True)
            (root / 'venv/bin/python').write_bytes(b'primary')
            (root / 'worldsculpt-wheelhouse.manifest.json').write_text('{}')
            with patch.object(setup, '_create_prepare_paths', return_value=([], [])), patch.object(setup, 'check_setup_readiness', return_value={}), patch.object(setup, '_wheelhouse_manifest_observation', return_value={}), patch.object(lane, '_probe', return_value={}), patch.object(lane, '_platform_gate'), patch.object(lane, 'repair', return_value={'status': 'installed'}) as repair:
                for args in (['--prepare', '--skip-install', '--workspace-root', str(root)], ['--workspace-root', str(root), '--skip-install', '{"ext_dir": "' + str(root) + '"}'], [sys.executable, str(root), '120', '--skip-install']):
                    result = setup.run_setup(args)
                    self.assertEqual(result['worldsculpt_lane']['status'], 'skipped')
                    self.assertFalse(result['installs_started'])
                repair.assert_not_called()


if __name__ == '__main__':
    unittest.main()
