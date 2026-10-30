import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import setup
from generator import _stage_external_images

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class OptionalNattenPipCheckTests(unittest.TestCase):
    def test_accepts_only_missing_natten_when_wheelhouse_omits_it(self):
        lines = (
            "naf 0.1.0+modly requires natten, which is not installed.\n"
            "pixal3d-core 0.1.0+modly requires natten, which is not installed.\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse = Path(temp)
            self.assertTrue(setup._is_optional_natten_absent_check({"stdout_tail": lines}, wheelhouse))
            self.assertFalse(setup._is_optional_natten_absent_check(
                {"stdout_tail": lines + "foo 1.0 requires bar, which is not installed.\n"}, wheelhouse
            ))
            self.assertFalse(setup._is_optional_natten_absent_check({"stdout_tail": ""}, wheelhouse))
            (wheelhouse / "natten-0.21.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"")
            self.assertFalse(setup._is_optional_natten_absent_check({"stdout_tail": lines}, wheelhouse))


class StageExternalImagesTests(unittest.TestCase):
    def test_copies_outside_images_into_workspace_and_keeps_workspace_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            outside = Path(temp) / "Desktop" / "view.webp"
            outside.parent.mkdir()
            outside.write_bytes(b"image bytes")
            inside = workspace / "already.png"
            inside.write_bytes(b"png")

            staged = _stage_external_images([str(outside), str(inside), None], workspace)

            copied = Path(staged[0])
            self.assertEqual(copied.parent, workspace.resolve() / "Inputs" / "pixal3d-mv")
            self.assertEqual(copied.suffix, ".webp")
            self.assertEqual(copied.read_bytes(), b"image bytes")
            self.assertEqual(staged[1:], [str(inside), None])
            self.assertEqual(_stage_external_images([str(outside)], workspace), staged[:1])


@unittest.skipUnless(HAS_TORCH, "torch is required for the natten fallback")
class NattenFallbackChunkingTests(unittest.TestCase):
    def _fallback(self):
        import torch  # noqa: F401
        from pixal3d_extension import runtime

        saved = sys.modules.pop("natten", None)

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name == "natten" or name.startswith("natten."):
                    raise ModuleNotFoundError("blocked", name="natten")

        blocker = Block()
        sys.meta_path.insert(0, blocker)
        try:
            runtime._install_natten_fallback()
            return sys.modules.pop("natten").na2d
        finally:
            sys.meta_path.remove(blocker)
            if saved is not None:
                sys.modules["natten"] = saved

    @staticmethod
    def _reference(q, k, v, kernel_size, dilation):
        import torch

        b, h, w, n, d = q.shape
        kh, kw = kernel_size
        dh, dw = dilation
        out = torch.zeros(b, h, w, n, v.shape[-1], dtype=q.dtype)
        for y in range(h):
            for x in range(w):
                ys = [y + (i - kh // 2) * dh for i in range(kh)]
                xs = [x + (j - kw // 2) * dw for j in range(kw)]
                keys = [(yy, xx) for yy in ys for xx in xs if 0 <= yy < h and 0 <= xx < w]
                kk = torch.stack([k[:, yy, xx] for yy, xx in keys], dim=2)
                vv = torch.stack([v[:, yy, xx] for yy, xx in keys], dim=2)
                scores = (q[:, y, x].unsqueeze(2) * kk).sum(-1) / math.sqrt(d)
                out[:, y, x] = (torch.softmax(scores, dim=-1).unsqueeze(-1) * vv).sum(2)
        return out

    def test_chunked_fallback_matches_reference_including_borders(self):
        import torch

        na2d = self._fallback()
        torch.manual_seed(0)
        for h, w, kernel, dilation, d_v in ((7, 9, (3, 3), (1, 1), 4), (11, 8, (3, 5), (2, 1), 6)):
            with self.subTest(h=h, w=w, kernel=kernel, dilation=dilation):
                q = torch.randn(2, h, w, 3, 4)
                k = torch.randn(2, h, w, 3, 4)
                v = torch.randn(2, h, w, 3, d_v)
                expected = self._reference(q, k, v, kernel, dilation)
                actual = na2d(q, k, v, kernel_size=kernel, dilation=dilation)
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertLess((actual - expected).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
