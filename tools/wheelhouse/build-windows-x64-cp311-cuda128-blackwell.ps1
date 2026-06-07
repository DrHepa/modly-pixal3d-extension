# Pixal3D Windows x64 cp311 CUDA 12.8 Blackwell candidate wheelhouse probe.
# Candidate-only: do not add this lane to wheelhouse.manifest.json until every
# native wheel is exact-stack, checksum-pinned, and validated on RTX 50-series.

[CmdletBinding()]
param(
    [string]$Lane = "windows-x64-cp311-cuda128-blackwell",
    [string]$WheelhouseVersion = "0.1.0",
    [string]$BuildRoot = "",
    [string]$NattenVersion = "0.21.6",
    [string]$NattenWheelPath = "",
    [switch]$AllowIncomplete
)

$ErrorActionPreference = "Stop"

if ($Lane -ne "windows-x64-cp311-cuda128-blackwell") {
    throw "Unsupported Blackwell Windows candidate lane: $Lane"
}

$PythonTag = "cp311"
$TorchMinor = "2.7"
$CudaTag = "cu128"
$exactStackPattern = "${CudaTag}torch${TorchMinor}-${PythonTag}-${PythonTag}-win_amd64"

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

$externalWheels = @(
    @{ Name = "flex_gemm_ap"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "flex_gemm_ap-1.0.0+cu128torch2.7-cp311-cp311-win_amd64.whl" },
    @{ Name = "cumesh_vb"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "cumesh_vb-1.0+cu128torch2.7-cp311-cp311-win_amd64.whl" },
    @{ Name = "o_voxel_vb_ap"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "o_voxel_vb_ap-0.0.1+cu128torch2.7-cp311-cp311-win_amd64.whl" },
    @{ Name = "drtk"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "drtk-0.1.0+cu128torch2.7-cp311-cp311-win_amd64.whl" },
    @{ Name = "flash_attn"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "flash_attn-2.8.3+cu128torch2.7-cp311-cp311-win_amd64.whl" },
    @{ Name = "nvdiffrast"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "nvdiffrast-0.4.0+cu128torch2.7-cp311-cp311-win_amd64.whl" },
    @{ Name = "nvdiffrec_render"; Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrec_render-latest/nvdiffrec_render-0.0.1%2Bcu128torch2.7-cp311-cp311-win_amd64.whl"; Filename = "nvdiffrec_render-0.0.1+cu128torch2.7-cp311-cp311-win_amd64.whl" }
)

$pureWheelPatterns = @("pixal3d_core-*.whl", "pipeline-*.whl", "moge-*.whl", "naf-*.whl", "utils3d-*.whl")
$copiedPure = @()
foreach ($pattern in $pureWheelPatterns) {
    $matches = @(Get-ChildItem -Path (Join-Path $repoRoot "wheels") -Filter $pattern -File)
    if ($matches.Count -ne 1) { throw "Expected exactly one pure wheel matching $pattern, found $($matches.Count)." }
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
                dst.writestr(item, replacements.get(item.filename, src.read(item.filename)))
    tmp_path.replace(wheel_path)

wheelhouse = Path(sys.argv[1])
patch_wheel_metadata(
    wheel_by_prefix(wheelhouse, "pixal3d_core"),
    {
        "natten": [f"Requires-Dist: natten=={sys.argv[2]}"],
        "o-voxel": ["Requires-Dist: o-voxel-vb-ap==0.0.1"],
        "cumesh": ["Requires-Dist: cumesh-vb==1.0"],
        "flex-gemm": ["Requires-Dist: flex-gemm-ap==1.0.0"],
        "nvdiffrec-render": ["Requires-Dist: nvdiffrec-render==0.0.1"],
    },
)
patch_wheel_metadata(wheel_by_prefix(wheelhouse, "naf"), {"natten": [f"Requires-Dist: natten=={sys.argv[2]}"]})
'@

Write-Host "Repairing Windows Blackwell pure-wheel dependency metadata."
$repairScript | python - $wheelhouseDir $NattenVersion
if ($LASTEXITCODE -ne 0) { throw "Windows Blackwell metadata repair failed with exit code $LASTEXITCODE." }

$downloaded = @()
foreach ($wheel in $externalWheels) {
    if ($wheel.Filename -notmatch [regex]::Escape($exactStackPattern)) { throw "Windows Blackwell wheel $($wheel.Filename) does not match exact stack $exactStackPattern." }
    $destination = Join-Path $wheelhouseDir $wheel.Filename
    Write-Host "Downloading $($wheel.Name): $($wheel.Filename)"
    Invoke-WebRequest -Uri $wheel.Url -OutFile $destination -UseBasicParsing
    $downloaded += [ordered]@{ name = $wheel.Name; filename = $wheel.Filename; url = $wheel.Url; sha256 = (Get-FileHash -Algorithm SHA256 $destination).Hash.ToLowerInvariant(); size_bytes = (Get-Item $destination).Length }
}

$includedNatten = $null
$unresolvedRequired = @()
if ([string]::IsNullOrWhiteSpace($NattenWheelPath)) {
    $unresolvedRequired += "missing NATTEN wheel path; Blackwell runtime candidate requires natten==$NattenVersion with HAS_LIBNATTEN=True for cp311/cu128/sm120"
} else {
    $resolvedNatten = Resolve-Path $NattenWheelPath
    $nattenName = Split-Path $resolvedNatten -Leaf
    $escapedNattenVersion = [regex]::Escape($NattenVersion)
    if ($nattenName -notmatch "^natten-$escapedNattenVersion-.*-win_amd64\.whl$") { throw "Unexpected Blackwell NATTEN wheel filename for natten ${NattenVersion}: $nattenName" }
    Copy-Item $resolvedNatten -Destination (Join-Path $wheelhouseDir $nattenName)
    $includedNatten = [ordered]@{ name = "natten"; filename = $nattenName; sha256 = (Get-FileHash -Algorithm SHA256 (Join-Path $wheelhouseDir $nattenName)).Hash.ToLowerInvariant(); size_bytes = (Get-Item (Join-Path $wheelhouseDir $nattenName)).Length; required_verification = "HAS_LIBNATTEN == True on real RTX 50-series hardware" }
}

$metadata = [ordered]@{
    lane = $Lane
    wheelhouse_version = $WheelhouseVersion
    status = $(if ($unresolvedRequired.Count -eq 0) { "candidate_complete_unvalidated" } else { "candidate_incomplete" })
    exact_stack = [ordered]@{ python_tag = $PythonTag; torch = $TorchMinor; cuda = $CudaTag; cuda_arch = "sm120"; platform_tag = "win_amd64" }
    copied_pure_wheels = $copiedPure
    downloaded_external_wheels = $downloaded
    included_natten = $includedNatten
    unresolved_required = $unresolvedRequired
    validation_required = @("Verify imports and pip check on a Windows NVIDIA machine", "Verify natten.HAS_LIBNATTEN == True", "Run Pixal3D Low VRAM generation on RTX 50-series and validate final GLB")
    publish_policy = "Candidate-only. Do not add to wheelhouse.manifest.json or publish as supported until validated on RTX 50-series hardware and checksum-pinned."
}

$metadataPath = Join-Path $wheelhouseDir "WINDOWS-BLACKWELL-CANDIDATE.json"
$metadata | ConvertTo-Json -Depth 8 | Set-Content -Path $metadataPath -Encoding UTF8

if (($unresolvedRequired.Count -gt 0) -and (-not $allowIncompleteBuild)) { throw "Windows Blackwell candidate is incomplete: $($unresolvedRequired -join '; ')." }

Compress-Archive -Path $wheelhouseDir -DestinationPath $archivePath -Force
$archiveHash = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLowerInvariant()
$archiveSize = (Get-Item $archivePath).Length
"$archiveHash  $archiveName" | Set-Content -Path "$archivePath.sha256" -Encoding ASCII

Write-Host "Built Windows Blackwell candidate archive: $archivePath"
Write-Host "size_bytes=$archiveSize"
Write-Host "sha256=$archiveHash"
