import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pixal3d_extension.multiview_images import _windows_stable_read, _workspace_file


@unittest.skipUnless(os.name == "nt", "real Windows filesystem contract")
class WindowsMultiviewCustodyTests(unittest.TestCase):
    def test_regular_file_is_read_from_stable_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            source = workspace / "view.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
            resolved, data = _workspace_file(str(source), workspace.resolve(), "view")
            self.assertEqual(resolved, source.resolve())
            self.assertEqual(data, source.read_bytes())

    def test_directory_and_junction_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            Image.new("RGB", (8, 8), (1, 2, 3)).save(outside / "outside.png")
            with self.assertRaisesRegex(ValueError, "regular file"):
                _workspace_file(str(workspace), workspace.resolve(), "view")
            junction = workspace / "junction"
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
                capture_output=True, text=True,
            )
            self.assertEqual(created.returncode, 0, created.stdout + created.stderr)
            try:
                with self.assertRaisesRegex(ValueError, "reparse|workspace|custody|symlink"):
                    _workspace_file(str(junction / "outside.png"), workspace.resolve(), "view")
            finally:
                os.rmdir(junction)

    def test_open_handle_blocks_path_swap_until_read_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            source = workspace / "view.png"
            replacement = workspace / "replacement.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
            Image.new("RGB", (8, 8), (9, 8, 7)).save(replacement)
            original = source.read_bytes()

            def attempt_swap(_path):
                with self.assertRaises(PermissionError):
                    os.replace(replacement, source)

            resolved, data = _windows_stable_read(
                source, workspace.resolve(), "view", _opened_hook=attempt_swap,
            )
            self.assertEqual(resolved, source.resolve())
            self.assertEqual(data, original)
            self.assertTrue(replacement.is_file())


if __name__ == "__main__":
    unittest.main()
