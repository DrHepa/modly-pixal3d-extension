import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pixal3d_extension.multiview_images import _windows_extended_path, _windows_stable_read, _workspace_file


@unittest.skipUnless(os.name == "nt", "real Windows filesystem contract")
class WindowsMultiviewCustodyTests(unittest.TestCase):
    @staticmethod
    def _short_path(path: Path) -> Path:
        import ctypes
        from ctypes import wintypes

        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short_path.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_short_path(str(path), buffer, len(buffer))
        if not length or length >= len(buffer):
            raise OSError(ctypes.get_last_error(), f"Cannot obtain short path for {path}")
        return Path(buffer.value)

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
            with self.assertRaisesRegex(ValueError, "workspace"):
                _workspace_file(str(outside / "outside.png"), workspace.resolve(), "view")
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

    def test_equivalent_short_and_long_workspace_aliases_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace alias regression"
            workspace.mkdir()
            source = workspace / "view.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
            short_workspace = self._short_path(workspace)
            if os.path.normcase(str(short_workspace)) == os.path.normcase(str(workspace)):
                self.skipTest("8.3 short names are unavailable on this Windows volume")

            resolved, data = _workspace_file(str(source), short_workspace, "view")

            self.assertTrue(resolved.is_file())
            self.assertEqual(data, source.read_bytes())

    def test_extended_local_path_beyond_max_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / ("long-segment-" + "a" * 40)
            workspace = first
            while len(str(workspace)) <= 300:
                workspace /= "nested-segment-" + "b" * 40
            source = workspace / "view.png"
            try:
                os.makedirs(_windows_extended_path(str(workspace)))
                image = Image.new("RGB", (8, 8), (1, 2, 3))
                image.save(_windows_extended_path(str(source)), format="PNG")

                resolved, data = _workspace_file(str(source), workspace, "view")

                self.assertGreater(len(str(source)), 260)
                self.assertTrue(data)
                self.assertTrue(str(resolved).lower().endswith("view.png"))
            finally:
                shutil.rmtree(_windows_extended_path(str(first)), ignore_errors=True)

    def test_case_sensitive_case_only_distinct_directory_is_not_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                enabled = subprocess.run(
                    ["fsutil.exe", "file", "SetCaseSensitiveInfo", str(root), "enable"],
                    capture_output=True, text=True,
                )
            except OSError as exc:
                self.skipTest(f"Windows cannot invoke per-directory case sensitivity: {exc}")
            if enabled.returncode != 0:
                self.skipTest(
                    "Windows cannot enable per-directory case sensitivity: "
                    + (enabled.stderr or enabled.stdout).strip()
                )
            workspace = root / "Workspace"
            case_only_distinct = root / "workspace"
            workspace.mkdir()
            case_only_distinct.mkdir()
            inside = workspace / "inside.png"
            outside = case_only_distinct / "outside.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(inside)
            Image.new("RGB", (8, 8), (4, 5, 6)).save(outside)

            resolved, data = _workspace_file(str(inside), workspace, "inside")
            self.assertTrue(resolved.is_file())
            self.assertEqual(data, inside.read_bytes())
            with self.assertRaisesRegex(ValueError, "workspace"):
                _workspace_file(str(outside), workspace, "outside")
            junction = workspace / "case-only-target"
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(case_only_distinct)],
                capture_output=True, text=True,
            )
            self.assertEqual(created.returncode, 0, created.stdout + created.stderr)
            try:
                with self.assertRaisesRegex(ValueError, "reparse|workspace"):
                    _workspace_file(str(junction / "outside.png"), workspace, "junction")
            finally:
                os.rmdir(junction)

    def test_different_volume_is_rejected_when_writable_volume_exists(self):
        import ctypes

        current_drive = Path(tempfile.gettempdir()).drive.upper()
        mask = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalDrives()
        candidates = [
            f"{chr(ord('A') + index)}:\\"
            for index in range(26)
            if mask & (1 << index) and f"{chr(ord('A') + index)}:" != current_drive
        ]
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            for drive in candidates:
                try:
                    with tempfile.TemporaryDirectory(dir=drive) as other_directory:
                        outside = Path(other_directory) / "outside.png"
                        Image.new("RGB", (8, 8), (1, 2, 3)).save(outside)
                        with self.assertRaisesRegex(ValueError, "workspace|volume"):
                            _workspace_file(str(outside), workspace, "view")
                        return
                except (OSError, PermissionError):
                    continue
        self.skipTest("No second writable Windows volume is available")

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
