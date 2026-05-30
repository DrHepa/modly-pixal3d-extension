#!/usr/bin/env bash
set -euo pipefail

lane="linux-x64-cp312-cuda124"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${WHEELHOUSE_BUILD_ROOT:-${repo_root}/build/wheelhouse/${lane}}"
work_dir="${build_root}/work"
wheelhouse_dir="${build_root}/wheelhouse"
dist_dir="${build_root}/dist/wheelhouse"
archive_path="${dist_dir}/pixal3d-wheelhouse-v0.1.0-${lane}.zip"

pure_wheels=(
  "pixal3d_core-0.1.0+modly-py3-none-any.whl"
  "pipeline-1.0.0+modly-py3-none-any.whl"
  "moge-2.0.0+modly-py3-none-any.whl"
  "naf-0.1.0+modly-py3-none-any.whl"
  "utils3d-1.3+modly.headless-py3-none-any.whl"
)

native_packages=(
  "natten"
  "o_voxel"
  "cumesh"
  "flex_gemm"
  "nvdiffrast"
  "nvdiffrec_render"
)

declare -A source_dir_vars=(
  ["natten"]="NATTEN_SOURCE_DIR"
  ["o_voxel"]="O_VOXEL_SOURCE_DIR"
  ["cumesh"]="CUMESH_SOURCE_DIR"
  ["flex_gemm"]="FLEX_GEMM_SOURCE_DIR"
  ["nvdiffrast"]="NVDIFFRAST_SOURCE_DIR"
  ["nvdiffrec_render"]="NVDIFFREC_RENDER_SOURCE_DIR"
)

declare -A default_source_dirs=(
  ["natten"]="${repo_root}/sources/natten"
  ["o_voxel"]="${repo_root}/sources/o_voxel"
  ["cumesh"]="${repo_root}/sources/cumesh"
  ["flex_gemm"]="${repo_root}/sources/flex_gemm"
  ["nvdiffrast"]="${repo_root}/sources/nvdiffrast"
  ["nvdiffrec_render"]="${repo_root}/sources/nvdiffrec_render"
)

declare -A source_ref_vars=(
  ["natten"]="NATTEN_SOURCE_REF"
  ["o_voxel"]="O_VOXEL_SOURCE_REF"
  ["cumesh"]="CUMESH_SOURCE_REF"
  ["flex_gemm"]="FLEX_GEMM_SOURCE_REF"
  ["nvdiffrast"]="NVDIFFRAST_SOURCE_REF"
  ["nvdiffrec_render"]="NVDIFFREC_RENDER_SOURCE_REF"
)

declare -A source_url_vars=(
  ["natten"]="NATTEN_SOURCE_URL"
  ["o_voxel"]="O_VOXEL_SOURCE_URL"
  ["cumesh"]="CUMESH_SOURCE_URL"
  ["flex_gemm"]="FLEX_GEMM_SOURCE_URL"
  ["nvdiffrast"]="NVDIFFRAST_SOURCE_URL"
  ["nvdiffrec_render"]="NVDIFFREC_RENDER_SOURCE_URL"
)

declare -A wheel_globs=(
  ["natten"]="natten-*-cp312-cp312-linux_x86_64.whl"
  ["o_voxel"]="o_voxel-*-cp312-cp312-linux_x86_64.whl"
  ["cumesh"]="cumesh-*-cp312-cp312-linux_x86_64.whl"
  ["flex_gemm"]="flex_gemm-*-cp312-cp312-linux_x86_64.whl"
  ["nvdiffrast"]="nvdiffrast-*-cp312-cp312-linux_x86_64.whl"
  ["nvdiffrec_render"]="nvdiffrec_render-*-cp312-cp312-linux_x86_64.whl"
)

fail_no_placeholder() {
  echo "$1" >&2
  echo "No placeholder wheelhouse archive will be created." >&2
  exit 1
}

setup_directories() {
  rm -rf "${work_dir}" "${wheelhouse_dir}" "${dist_dir}"
  mkdir -p "${work_dir}" "${wheelhouse_dir}" "${dist_dir}"
}

verify_build_prerequisites() {
  if ! command -v nvcc >/dev/null 2>&1; then
    fail_no_placeholder "CUDA 12.4 toolchain is unavailable; cannot build ${lane}."
  fi

  if ! nvcc --version | grep -Eq 'release 12\.4|V12\.4'; then
    fail_no_placeholder "CUDA 12.4 toolchain is required for ${lane}; detected nvcc is not 12.4."
  fi

  local python_tag
  python_tag="$(python3 - <<'PY'
import sys
print(f'cp{sys.version_info.major}{sys.version_info.minor}')
PY
)"
  if [[ "${python_tag}" != "cp312" ]]; then
    fail_no_placeholder "Python cp312 is required for ${lane}; detected ${python_tag}."
  fi
}

install_build_prerequisites() {
  if [[ "${WHEELHOUSE_SKIP_PREREQ_INSTALL:-0}" == "1" ]]; then
    echo "Skipping Python build prerequisite installation because WHEELHOUSE_SKIP_PREREQ_INSTALL=1." >&2
    return 0
  fi

  python3 -m pip install --upgrade pip setuptools wheel build ninja cmake packaging
  python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 "${WHEELHOUSE_TORCH_SPEC:-torch==2.6.0+cu124}"
}

copy_pure_wheels() {
  local wheel
  for wheel in "${pure_wheels[@]}"; do
    local source="${repo_root}/wheels/${wheel}"
    if [[ ! -f "${source}" ]]; then
      fail_no_placeholder "Required pure wheel is missing: wheels/${wheel}"
    fi
    cp "${source}" "${wheelhouse_dir}/${wheel}"
  done
}

env_value() {
  local name="$1"
  printf '%s' "${!name:-}"
}

source_dir_for_package() {
  local package="$1"
  local var_name="${source_dir_vars[${package}]}"
  local configured
  configured="$(env_value "${var_name}")"
  if [[ -n "${configured}" ]]; then
    printf '%s' "${configured}"
  else
    printf '%s' "${default_source_dirs[${package}]}"
  fi
}

maybe_fetch_source_ref() {
  local package="$1"
  local source_dir="$2"
  local ref_var="${source_ref_vars[${package}]}"
  local url_var="${source_url_vars[${package}]}"
  local source_ref source_url
  source_ref="$(env_value "${ref_var}")"
  source_url="$(env_value "${url_var}")"

  if [[ -d "${source_dir}" ]]; then
    return 0
  fi

  if [[ -n "${source_ref}" || -n "${source_url}" ]]; then
    if [[ "${WHEELHOUSE_ALLOW_SOURCE_FETCH:-0}" != "1" ]]; then
      echo "${package}: source ref/url was provided but ${source_dir} is absent; set WHEELHOUSE_ALLOW_SOURCE_FETCH=1 to fetch explicitly." >&2
      return 1
    fi
    if [[ -z "${source_url}" || -z "${source_ref}" ]]; then
      echo "${package}: both ${url_var} and ${ref_var} are required for explicit source fetching." >&2
      return 1
    fi
    git clone "${source_url}" "${source_dir}"
    git -C "${source_dir}" checkout "${source_ref}"
    return 0
  fi

  return 1
}

build_native_wheel() {
  local package="$1"
  local source_dir
  source_dir="$(source_dir_for_package "${package}")"

  if ! maybe_fetch_source_ref "${package}" "${source_dir}"; then
    return 1
  fi

  if [[ ! -f "${source_dir}/pyproject.toml" && ! -f "${source_dir}/setup.py" ]]; then
    echo "${package}: ${source_dir} does not contain pyproject.toml or setup.py." >&2
    return 1
  fi

  echo "Building ${package} from ${source_dir}" >&2
  python3 -m pip wheel "${source_dir}" --no-deps --wheel-dir "${wheelhouse_dir}"

  local glob="${wheel_globs[${package}]}"
  shopt -s nullglob
  local matches=("${wheelhouse_dir}"/${glob})
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    echo "${package}: build completed but no expected cp312 linux_x86_64 wheel matched ${glob}." >&2
    return 1
  fi
}

build_native_wheels() {
  local missing=()
  local package
  for package in "${native_packages[@]}"; do
    if ! build_native_wheel "${package}"; then
      missing+=("${package}=$(source_dir_for_package "${package}")")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "Missing native source directories for ${lane}:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "Provide package source directories via *_SOURCE_DIR or explicit *_SOURCE_URL + *_SOURCE_REF with WHEELHOUSE_ALLOW_SOURCE_FETCH=1." >&2
    echo "Native wheels are required for: ${native_packages[*]}." >&2
    echo "Pure wheels were copied from wheels/: ${pure_wheels[*]}." >&2
    echo "No placeholder wheelhouse archive will be created." >&2
    exit 1
  fi
}

finalize_wheelhouse_archive() {
  local package
  for package in "${native_packages[@]}"; do
    shopt -s nullglob
    local matches=("${wheelhouse_dir}"/${wheel_globs[${package}]})
    shopt -u nullglob
    if (( ${#matches[@]} == 0 )); then
      fail_no_placeholder "Cannot finalize ${lane}; missing native wheel for ${package}."
    fi
  done

  (cd "${build_root}" && python3 - <<PY
from pathlib import Path
import zipfile
archive = Path(${archive_path@Q})
wheelhouse = Path(${wheelhouse_dir@Q})
archive.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for wheel in sorted(wheelhouse.glob('*.whl')):
        zf.write(wheel, Path('wheelhouse') / wheel.name)
PY
  )
  sha256sum "${archive_path}" > "${archive_path}.sha256"
}

main() {
  setup_directories
  verify_build_prerequisites
  install_build_prerequisites
  copy_pure_wheels
  build_native_wheels
  finalize_wheelhouse_archive
  echo "Built ${archive_path}"
}

main "$@"
