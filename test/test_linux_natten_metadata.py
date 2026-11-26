import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup


class LinuxNattenMetadataTests(unittest.TestCase):
    lane = "linux-x64-cp312-cuda124"
    naf = "naf 0.1.0+modly requires natten, which is not installed.\n"
    core = "pixal3d-core 0.1.0+modly requires natten, which is not installed.\n"

    def test_only_exact_missing_optional_dependency_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp)
            check = {
                "returncode": 1,
                "output_truncated": False,
                "stderr_tail": "",
                "stdout_tail": self.naf + self.core,
            }
            validate = setup._is_known_linux_x64_missing_natten
            self.assertTrue(validate(check, wheelhouse, self.lane))
            for change in (
                {"stdout_tail": self.core + "unrelated 0.1 requires natten, which is not installed.\n"},
                {"stdout_tail": self.core.replace("0.1.0", "0.1.1")},
                {"stdout_tail": ""},
                {"stderr_tail": "WARNING: failed to parse package metadata"},
                {"returncode": 2},
                {"output_truncated": True},
                {"output_truncated": None},
            ):
                with self.subTest(change=change):
                    self.assertFalse(validate(check | change, wheelhouse, self.lane))
            self.assertFalse(validate(check, wheelhouse, "linux-aarch64-cp312-cuda124"))
            self.assertFalse(validate(check, wheelhouse, "windows-x64-cp312-cuda124"))
            (wheelhouse / "natten-test.whl").touch()
            self.assertFalse(validate(check, wheelhouse, self.lane))

    def test_hidden_earlier_failure_in_truncated_pip_output_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            command = ["python", "-m", "pip", "check"]
            output = "unseen failure\n" + self.naf * 100
            with patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess(command, 1, output, "")):
                result = setup._run_setup_command(command, cwd=path)
            self.assertTrue(result["output_truncated"])
            self.assertNotIn("unseen failure", result["stdout_tail"])
            self.assertFalse(setup._is_known_linux_x64_missing_natten(result, path, self.lane))


if __name__ == "__main__":
    unittest.main()
