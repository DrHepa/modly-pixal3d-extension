# Pixal3D Windows x64 cp312 CUDA 12.4 candidate wheelhouse probe.
# This script intentionally prepares candidate artifacts only. Do not add the
# Windows lane to wheelhouse.manifest.json until every required wheel is real,
# checksum-pinned, and the release asset has been uploaded.

[CmdletBinding()]
param(
    [string]$Lane = "windows-x64-cp312-cuda124",
    [string]$WheelhouseVersion = "0.1.0",
    [string]$BuildRoot = "",
    [switch]$AllowIncomplete
)

$ErrorActionPreference = "Stop"

if (($Lane -ne "windows-x64-cp312-cuda124") -and ($Lane -ne "windows-x64-cp311-cuda124")) {
    throw "Unsupported Windows candidate lane: $Lane"
}

$PythonTag = if ($Lane -match "cp311") { "cp311" } else { "cp312" }

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $BuildRoot = Join-Path $repoRoot "build\wheelhouse\$Lane"
}

$allowIncompleteEnv = $env:WHEELHOUSE_WINDOWS_ALLOW_INCOMPLETE -eq "1"
$allowIncompleteBuild = $AllowIncomplete.IsPresent -or $allowIncompleteEnv

$workDir = Join-Path $BuildRoot "work"
$wheelhouseDir = Join-Path $BuildRoot "wheelhouse"
$distDir = Join-Path $BuildRoot "dist\wheelhouse"
$archiveName = "pixal3d-wheelhouse-v$WheelhouseVersion-$Lane.zip"
$archivePath = Join-Path $distDir $archiveName

Remove-Item -Recurse -Force $workDir, $wheelhouseDir, $distDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $workDir, $wheelhouseDir, $distDir | Out-Null

$exactStackPattern = "cu124torch2.6-$PythonTag-$PythonTag-win_amd64"
$externalWheels = @(
    @{
        Name = "flex_gemm_ap"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "flex_gemm_ap-1.0.0+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    },
    @{
        Name = "cumesh_vb"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "cumesh_vb-1.0+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    },
    @{
        Name = "o_voxel_vb_ap"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "o_voxel_vb_ap-0.0.1+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    },
    @{
        Name = "drtk"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "drtk-0.1.0+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    },
    @{
        Name = "flash_attn"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "flash_attn-2.8.3+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    },
    @{
        Name = "nvdiffrast"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "nvdiffrast-0.4.0+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    },
    @{
        Name = "nvdiffrec_render"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrec_render-latest/nvdiffrec_render-0.0.1%2Bcu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
        Filename = "nvdiffrec_render-0.0.1+cu124torch2.6-$PythonTag-$PythonTag-win_amd64.whl"
    }
)

$pureWheelPatterns = @(
    "pixal3d_core-*.whl",
    "pipeline-*.whl",
    "moge-*.whl",
    "naf-*.whl",
    "utils3d-*.whl"
)

$copiedPure = @()
foreach ($pattern in $pureWheelPatterns) {
    $matches = @(Get-ChildItem -Path (Join-Path $repoRoot "wheels") -Filter $pattern -File)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one pure wheel matching $pattern, found $($matches.Count)."
    }
    Copy-Item $matches[0].FullName -Destination $wheelhouseDir
    $copiedPure += $matches[0].Name
}

$repairScript = @'
import base64
import csv
import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path


def wheel_by_prefix(wheelhouse: Path, prefix: str) -> Path:
    matches = sorted(wheelhouse.glob(f"{prefix}-*.whl"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one wheel matching {prefix}-*.whl, found {len(matches)}")
    return matches[0]


def hash_record(data: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", str(len(data))


def rewrite_record(record_text: str, replacements: dict[str, bytes]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for row in csv.reader(io.StringIO(record_text)):
        if not row:
            continue
        path = row[0]
        if path in replacements:
            digest, size = hash_record(replacements[path])
            writer.writerow([path, digest, size])
        elif path.endswith(".dist-info/RECORD"):
            writer.writerow([path, "", ""])
        else:
            writer.writerow(row)
    return output.getvalue()


def patch_wheel_metadata(wheel_path: Path, replace_requires: dict[str, list[str]]) -> None:
    with zipfile.ZipFile(wheel_path, "r") as src:
        names = src.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        metadata = src.read(metadata_name).decode("utf-8")
        new_lines = []
        for line in metadata.splitlines():
            if line.startswith("Requires-Dist: "):
                requirement = line.removeprefix("Requires-Dist: ").split(";", 1)[0].strip()
                normalized = requirement.split()[0].split("==", 1)[0].lower().replace("_", "-")
                if normalized in replace_requires:
                    new_lines.extend(replace_requires[normalized])
                    continue
            new_lines.append(line)
        metadata_bytes = ("\n".join(new_lines) + "\n").encode("utf-8")
        replacements = {metadata_name: metadata_bytes}
        record_bytes = rewrite_record(src.read(record_name).decode("utf-8"), replacements).encode("utf-8")
        replacements[record_name] = record_bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".whl", dir=wheel_path.parent) as tmp:
            tmp_path = Path(tmp.name)
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = replacements.get(item.filename, src.read(item.filename))
                dst.writestr(item, data)
    tmp_path.replace(wheel_path)


wheelhouse = Path(sys.argv[1])
patch_wheel_metadata(
    wheel_by_prefix(wheelhouse, "pixal3d_core"),
    {
        "natten": ['Requires-Dist: natten==0.21.0; platform_system != "Windows"'],
        "o-voxel": [
            'Requires-Dist: o-voxel==0.0.1; platform_system != "Windows"',
            'Requires-Dist: o-voxel-vb-ap==0.0.1; platform_system == "Windows"',
        ],
        "cumesh": [
            'Requires-Dist: cumesh==0.0.1; platform_system != "Windows"',
            'Requires-Dist: cumesh-vb==1.0; platform_system == "Windows"',
        ],
        "flex-gemm": [
            'Requires-Dist: flex-gemm==1.0.0; platform_system != "Windows"',
            'Requires-Dist: flex-gemm-ap==1.0.0; platform_system == "Windows"',
        ],
        "nvdiffrec-render": [
            'Requires-Dist: nvdiffrec-render==0.0.0; platform_system != "Windows"',
            'Requires-Dist: nvdiffrec-render==0.0.1; platform_system == "Windows"',
        ],
    },
)
patch_wheel_metadata(
    wheel_by_prefix(wheelhouse, "naf"),
    {"natten": ['Requires-Dist: natten; platform_system != "Windows"']},
)
'@

Write-Host "Repairing Windows pure-wheel dependency metadata for exact-stack native aliases."
$repairScript | python - $wheelhouseDir
if ($LASTEXITCODE -ne 0) {
    throw "Windows pure-wheel dependency metadata repair failed with exit code $LASTEXITCODE."
}

$downloaded = @()
foreach ($wheel in $externalWheels) {
    if ($wheel.Filename -notmatch [regex]::Escape($exactStackPattern)) {
        throw "Windows wheel $($wheel.Filename) does not match exact stack $exactStackPattern."
    }
    $destination = Join-Path $wheelhouseDir $wheel.Filename
    Write-Host "Downloading $($wheel.Name): $($wheel.Filename)"
    Invoke-WebRequest -Uri $wheel.Url -OutFile $destination -UseBasicParsing
    $downloaded += [ordered]@{
        name = $wheel.Name
        filename = $wheel.Filename
        url = $wheel.Url
        sha256 = (Get-FileHash -Algorithm SHA256 $destination).Hash.ToLowerInvariant()
        size_bytes = (Get-Item $destination).Length
    }
}

$externalRepairScript = @'
import tempfile
import zipfile
from pathlib import Path
import csv
import io
import sys

def wheel_by_prefix(wheelhouse: Path, prefix: str) -> Path:
    matches = sorted(wheelhouse.glob(f"{prefix}-*.whl"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one wheel matching {prefix}-*.whl, found {len(matches)}")
    return matches[0]

def rename_dist_info_prefix(wheel_path: Path, old_prefix: str, new_prefix: str) -> None:
    with zipfile.ZipFile(wheel_path, "r") as src:
        names = src.namelist()
        old_dirs = sorted({name.split("/", 1)[0] for name in names if name.startswith(old_prefix) and ".dist-info/" in name})
        if not old_dirs:
            if any(name.startswith(new_prefix) and ".dist-info/" in name for name in names):
                return
            raise SystemExit(f"No {old_prefix}*.dist-info directory found in {wheel_path.name}")
        if len(old_dirs) != 1:
            raise SystemExit(f"Expected one {old_prefix}*.dist-info directory in {wheel_path.name}, found {old_dirs}")
        old_dir = old_dirs[0]
        new_dir = old_dir.replace(old_prefix, new_prefix, 1)
        old_record = f"{old_dir}/RECORD"
        new_record = f"{new_dir}/RECORD"
        record_text = src.read(old_record).decode("utf-8")
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        for row in csv.reader(io.StringIO(record_text)):
            if not row:
                continue
            row[0] = row[0].replace(f"{old_dir}/", f"{new_dir}/", 1)
            if row[0] == new_record:
                row = [row[0], "", ""]
            writer.writerow(row)
        record_bytes = output.getvalue().encode("utf-8")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".whl", dir=wheel_path.parent) as tmp:
            tmp_path = Path(tmp.name)
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                data = record_bytes if item.filename == old_record else src.read(item.filename)
                filename = item.filename.replace(f"{old_dir}/", f"{new_dir}/", 1)
                dst.writestr(filename, data)
    tmp_path.replace(wheel_path)

wheelhouse = Path(sys.argv[1])
rename_dist_info_prefix(wheel_by_prefix(wheelhouse, "nvdiffrec_render"), "nvdiffrec-render-", "nvdiffrec_render-")
'@

Write-Host "Repairing Windows external wheel dist-info directory names."
$externalRepairScript | python - $wheelhouseDir
if ($LASTEXITCODE -ne 0) {
    throw "Windows external wheel dist-info repair failed with exit code $LASTEXITCODE."
}

$downloaded = @()
foreach ($wheel in $externalWheels) {
    $destination = Join-Path $wheelhouseDir $wheel.Filename
    $downloaded += [ordered]@{
        name = $wheel.Name
        filename = $wheel.Filename
        url = $wheel.Url
        sha256 = (Get-FileHash -Algorithm SHA256 $destination).Hash.ToLowerInvariant()
        size_bytes = (Get-Item $destination).Length
    }
}

$unresolvedRequired = @()

$metadata = [ordered]@{
    lane = $Lane
    wheelhouse_version = $WheelhouseVersion
    status = $(if ($unresolvedRequired.Count -eq 0) { "candidate_complete" } else { "candidate_incomplete" })
    exact_stack = [ordered]@{
        python_tag = $PythonTag
        torch = "2.6"
        cuda = "cu124"
        platform_tag = "win_amd64"
    }
    copied_pure_wheels = $copiedPure
    downloaded_external_wheels = $downloaded
    unresolved_required = $unresolvedRequired
    optional_excluded = @(
        [ordered]@{
            name = "natten"
            reason = "strict NAF requires natten.HAS_LIBNATTEN == True; no verified torch2.6/cu124/cp312/win_amd64 libnatten wheel is available"
        }
    )
    publish_policy = "Do not add to wheelhouse.manifest.json until this candidate archive is uploaded to the pinned release and checksum-pinned."
}

$metadataPath = Join-Path $wheelhouseDir "WINDOWS-CANDIDATE.json"
$metadata | ConvertTo-Json -Depth 8 | Set-Content -Path $metadataPath -Encoding UTF8

if (($unresolvedRequired.Count -gt 0) -and (-not $allowIncompleteBuild)) {
    throw "Windows candidate is incomplete: $($unresolvedRequired -join '; '). Set WHEELHOUSE_WINDOWS_ALLOW_INCOMPLETE=1 only to upload an inspection artifact."
}

Compress-Archive -Path $wheelhouseDir -DestinationPath $archivePath -Force
$archiveHash = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLowerInvariant()
$archiveSize = (Get-Item $archivePath).Length
"$archiveHash  $archiveName" | Set-Content -Path "$archivePath.sha256" -Encoding ASCII

Write-Host "Built Windows candidate archive: $archivePath"
Write-Host "size_bytes=$archiveSize"
Write-Host "sha256=$archiveHash"
if ($unresolvedRequired.Count -gt 0) {
    Write-Warning "Candidate is not publishable: $($unresolvedRequired -join '; ')"
}
