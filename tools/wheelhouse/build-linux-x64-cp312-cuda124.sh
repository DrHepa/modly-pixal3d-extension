#!/usr/bin/env bash
set -euo pipefail

lane="linux-x64-cp312-cuda124"
dist_dir="dist/wheelhouse"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA 12.4 toolchain is unavailable; cannot build ${lane}." >&2
  echo "No placeholder wheelhouse archive will be created." >&2
  exit 1
fi

if [[ "$(python3 - <<'PY'
import sys
print(f'cp{sys.version_info.major}{sys.version_info.minor}')
PY
)" != "cp312" ]]; then
  echo "Python cp312 is required for ${lane}; no placeholder wheelhouse archive will be created." >&2
  exit 1
fi

mkdir -p "${dist_dir}"
echo "TODO: wire native Pixal3D wheel build commands for ${lane} once CUDA 12.4 hosted build prerequisites are proven." >&2
echo "Fail clearly instead of publishing incomplete or placeholder assets." >&2
exit 1
