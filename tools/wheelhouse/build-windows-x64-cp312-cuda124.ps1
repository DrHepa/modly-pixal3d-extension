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

if ($Lane -ne "windows-x64-cp312-cuda124") {
    throw "Unsupported Windows candidate lane: $Lane"
}

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

$exactStackPattern = "cu124torch2.6-cp312-cp312-win_amd64"
$externalWheels = @(
    @{
        Name = "flex_gemm_ap"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu124torch2.6-cp312-cp312-win_amd64.whl"
        Filename = "flex_gemm_ap-1.0.0+cu124torch2.6-cp312-cp312-win_amd64.whl"
    },
    @{
        Name = "cumesh_vb"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu124torch2.6-cp312-cp312-win_amd64.whl"
        Filename = "cumesh_vb-1.0+cu124torch2.6-cp312-cp312-win_amd64.whl"
    },
    @{
        Name = "o_voxel_vb_ap"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu124torch2.6-cp312-cp312-win_amd64.whl"
        Filename = "o_voxel_vb_ap-0.0.1+cu124torch2.6-cp312-cp312-win_amd64.whl"
    },
    @{
        Name = "drtk"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu124torch2.6-cp312-cp312-win_amd64.whl"
        Filename = "drtk-0.1.0+cu124torch2.6-cp312-cp312-win_amd64.whl"
    },
    @{
        Name = "flash_attn"
        Url = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu124torch2.6-cp312-cp312-win_amd64.whl"
        Filename = "flash_attn-2.8.3+cu124torch2.6-cp312-cp312-win_amd64.whl"
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

$unresolvedRequired = @(
    "nvdiffrast win_amd64 cp312 torch2.6 cu124 exact-stack wheel",
    "nvdiffrec_render win_amd64 cp312 torch2.6 cu124 exact-stack wheel"
)

$metadata = [ordered]@{
    lane = $Lane
    wheelhouse_version = $WheelhouseVersion
    status = $(if ($unresolvedRequired.Count -eq 0) { "candidate_complete" } else { "candidate_incomplete" })
    exact_stack = [ordered]@{
        python_tag = "cp312"
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
    publish_policy = "Do not upload to release or add to wheelhouse.manifest.json while status is candidate_incomplete."
}

$metadataPath = Join-Path $wheelhouseDir "WINDOWS-CANDIDATE-NOT-PUBLISHABLE.json"
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
