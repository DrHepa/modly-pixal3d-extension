[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$CandidateArtifactPath,

  [Parameter(Mandatory=$true)]
  [ValidatePattern('^[0-9a-fA-F]{64}$')]
  [string]$ExpectedArtifactSha256,

  [Parameter(Mandatory=$true)]
  [long]$ExpectedArtifactSizeBytes,

  [Parameter(Mandatory=$true)]
  [string]$ModlyWeightsPath,

  [Parameter(Mandatory=$true)]
  [string]$FixtureImagePath,

  [Parameter(Mandatory=$true)]
  [string]$EvidenceDir,

  [string]$PythonExe = "python",
  [string]$SourceCommit = "",
  [string]$GitHubRunId = "",
  [string]$GitHubRunUrl = "",
  [switch]$KeepWorkDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:EvidenceRoot = $null
$script:WorkRoot = $null
$script:TempExtension = $null

function New-Gate {
  param([string]$Name, [string]$Status, [System.Collections.IDictionary]$Fields = [ordered]@{})
  $gate = [ordered]@{ status = $Status }
  foreach ($key in $Fields.Keys) { $gate[$key] = $Fields[$key] }
  return $gate
}

function Get-FileSha256 {
  param([Parameter(Mandatory=$true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-FileContract {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$ExpectedSha256,
    [Parameter(Mandatory=$true)][long]$ExpectedSizeBytes
  )
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "file_missing: $Path"
  }
  if ($ExpectedSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw "invalid_expected_sha256: expected a 64-character SHA256 hex digest"
  }
  $item = Get-Item -LiteralPath $Path
  if ($item.Length -ne $ExpectedSizeBytes) {
    throw "size_mismatch: expected $ExpectedSizeBytes bytes but found $($item.Length) at $Path"
  }
  $actual = Get-FileSha256 -Path $Path
  if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "sha256_mismatch: expected $ExpectedSha256 but found $actual at $Path"
  }
  return [ordered]@{ path = $Path; size_bytes = $item.Length; sha256 = $actual }
}

function Assert-PathContained {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$Root
  )
  $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
  $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
  $relative = [System.IO.Path]::GetRelativePath($resolvedRoot, $resolvedPath)
  if ($relative.StartsWith('..') -or [System.IO.Path]::IsPathRooted($relative)) {
    throw "path_escape: $resolvedPath escapes $resolvedRoot"
  }
  return [ordered]@{ path = $resolvedPath; root = $resolvedRoot }
}

function Copy-ExtensionTree {
  param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination
  )
  New-Item -ItemType Directory -Force -Path $Destination | Out-Null
  $excludedDirs = @('.git', '.modly', 'venv', '__pycache__', 'build', 'dist', 'node_modules')
  Get-ChildItem -LiteralPath $Source -Force | Where-Object { $excludedDirs -notcontains $_.Name } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Recurse -Force
  }
}


function Assert-BlackwellAuxiliaryAssets {
  param([Parameter(Mandatory=$true)][string]$WeightsPath)
  $resolved = (Resolve-Path -LiteralPath $WeightsPath).Path
  $basePipeline = Join-Path $resolved 'pipeline.json'
  $homePipeline = Join-Path $resolved 'models\pixal3d\_shared\pixal3d-base\pipeline.json'
  if ((Test-Path -LiteralPath $basePipeline -PathType Leaf) -or (Test-Path -LiteralPath $homePipeline -PathType Leaf)) {
    return [ordered]@{ path = $resolved; requirement = 'primary assets, DINO, RMBG, MoGe, and valid NAF sentinel are checked by the Python helper before generator load' }
  }
  throw "blackwell_weights_root_invalid: ModlyWeightsPath must be the Modly home or pixal3d-base directory containing pipeline.json"
}

function Assert-NetworkDeniedBoundary {
  return [ordered]@{
    status = 'configured'
    enforcement = 'Python helper sets HF/Transformers offline envs and installs a NetworkDenied monkeypatch for socket, urllib, and requests during generation'
  }
}

function Invoke-JsonCommand {
  param(
    [Parameter(Mandatory=$true)][string[]]$Command,
    [Parameter(Mandatory=$true)][string]$WorkingDirectory,
    [Parameter(Mandatory=$true)][string]$LogPath
  )
  $stdout = "$LogPath.stdout.txt"
  $stderr = "$LogPath.stderr.txt"
  $previousLocation = (Get-Location).Path
  try {
    Set-Location -LiteralPath $WorkingDirectory
    $arguments = @()
    if ($Command.Length -gt 1) { $arguments = $Command[1..($Command.Length - 1)] }
    & $Command[0] @arguments > $stdout 2> $stderr
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
  } finally {
    Set-Location -LiteralPath $previousLocation
  }
  $stdoutText = if (Test-Path -LiteralPath $stdout) { Get-Content -Raw -LiteralPath $stdout } else { '' }
  $stderrText = if (Test-Path -LiteralPath $stderr) { Get-Content -Raw -LiteralPath $stderr } else { '' }
  return [ordered]@{ returncode = $exitCode; stdout = $stdoutText; stderr = $stderrText; stdout_path = $stdout; stderr_path = $stderr }
}

function Invoke-SetupRepair {
  param(
    [Parameter(Mandatory=$true)][int]$Attempt,
    [Parameter(Mandatory=$true)][string]$TempExtension,
    [Parameter(Mandatory=$true)][string]$PayloadJson,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$EvidenceRoot
  )
  $setupLog = Join-Path $EvidenceRoot "setup-repair-$Attempt"
  $result = Invoke-JsonCommand -Command @($PythonExe, 'setup.py', '--json', '--payload-json', $PayloadJson) -WorkingDirectory $TempExtension -LogPath $setupLog
  $jsonLine = ($result.stdout -split "`r?`n" | Where-Object { $_.Trim().StartsWith('{') } | Select-Object -Last 1)
  if (-not $jsonLine) {
    throw "setup_no_json: setup.py attempt $Attempt did not emit JSON; see $($result.stdout_path) and $($result.stderr_path)"
  }
  $payload = $jsonLine | ConvertFrom-Json -Depth 100
  $payload | ConvertTo-Json -Depth 100 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $EvidenceRoot "setup-repair-$Attempt.json")
  if ($result.returncode -ne 0 -or $payload.status -ne 'prepared') {
    throw "setup_failed: setup.py attempt $Attempt returned $($result.returncode) status $($payload.status); see $($result.stdout_path) and $($result.stderr_path)"
  }
  return $payload
}

function Get-VenvPython {
  param([Parameter(Mandatory=$true)][string]$TempExtension)
  $candidates = @(
    (Join-Path $TempExtension 'venv\Scripts\python.exe'),
    (Join-Path $TempExtension 'venv\bin\python')
  )
  foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
  }
  throw "venv_python_missing: setup.py did not create a Python executable under $TempExtension\venv"
}

function Resolve-CandidateWheelhouseArchive {
  param(
    [Parameter(Mandatory=$true)][string]$CandidateArtifact,
    [Parameter(Mandatory=$true)][string]$WorkRoot
  )
  $extractRoot = Join-Path $WorkRoot 'candidate-artifact'
  New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
  Expand-Archive -LiteralPath $CandidateArtifact -DestinationPath $extractRoot -Force
  $inner = Get-ChildItem -LiteralPath $extractRoot -Recurse -File -Filter '*windows-x64-cp311-cuda128-blackwell*.zip' | Select-Object -First 1
  if ($null -ne $inner) { return $inner.FullName }
  return $CandidateArtifact
}

function Add-TemporaryBlackwellManifestAsset {
  param(
    [Parameter(Mandatory=$true)][string]$TempExtension,
    [Parameter(Mandatory=$true)][string]$WheelhouseArchive,
    [Parameter(Mandatory=$true)][string]$WheelhouseSha256,
    [Parameter(Mandatory=$true)][long]$WheelhouseSizeBytes
  )
  $manifestPath = Join-Path $TempExtension 'wheelhouse.manifest.json'
  $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json -Depth 100
  $assetId = 'windows-x64-cp311-cuda128-blackwell'
  $asset = [ordered]@{
    id = $assetId
    filename = (Split-Path -Leaf $WheelhouseArchive)
    size_bytes = $WheelhouseSizeBytes
    sha256 = $WheelhouseSha256
    compression = 'zip'
    packages = @('utils3d','pipeline','moge','naf','o-voxel-vb-ap','cumesh-vb','flex-gemm-ap','drtk','flash-attn','nvdiffrast','nvdiffrec-render','pixal3d-core','natten')
    selectors = [ordered]@{ os = 'windows'; arch = 'x64'; python_tag = 'cp311'; accelerator_lane = 'cuda128-blackwell' }
  }
  $manifest.assets += $asset
  $manifest | ConvertTo-Json -Depth 100 | Set-Content -Encoding UTF8 -LiteralPath $manifestPath

  $cacheRoot = Join-Path $TempExtension '.modly\cache\wheelhouse\pixal3d\0.1.0\windows-x64-cp311-cuda128-blackwell'
  New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
  Copy-Item -LiteralPath $WheelhouseArchive -Destination (Join-Path $cacheRoot 'archive.zip') -Force
  return [ordered]@{ manifest_path = $manifestPath; asset_id = $assetId; cache_archive = (Join-Path $cacheRoot 'archive.zip') }
}

$gates = [ordered]@{}
$evidence = [ordered]@{
  schema_version = 'pixal3d.blackwell.orchestrator/v1'
  status = 'failed'
  candidate = [ordered]@{}
  run = [ordered]@{ source_commit = $SourceCommit; github_run_id = $GitHubRunId; github_run_url = $GitHubRunUrl }
  host = [ordered]@{ computer_name = $env:COMPUTERNAME; runner_name = $env:RUNNER_NAME; os = [System.Environment]::OSVersion.VersionString }
  gates = $gates
  cancellation = [ordered]@{ status = 'not_supported_by_harness'; reason = 'The manual harness reports cancellation as untested and does not fake a Modly cancel event.' }
}

try {
  $script:EvidenceRoot = (New-Item -ItemType Directory -Force -Path $EvidenceDir).FullName
  $candidateArtifact = (Resolve-Path -LiteralPath $CandidateArtifactPath).Path
  $weightsPath = (Resolve-Path -LiteralPath $ModlyWeightsPath).Path
  $fixtureImage = (Resolve-Path -LiteralPath $FixtureImagePath).Path
  $gates['artifact_contract'] = New-Gate 'artifact_contract' 'passed' (Assert-FileContract -Path $candidateArtifact -ExpectedSha256 $ExpectedArtifactSha256 -ExpectedSizeBytes $ExpectedArtifactSizeBytes)
  $gates['weights_path'] = New-Gate 'weights_path' 'passed' @{ path = $weightsPath; note = 'pre-provisioned Modly weights only; no model downloads or secrets are accepted by this harness' }
  $gates['blackwell_auxiliary_assets'] = New-Gate 'blackwell_auxiliary_assets' 'deferred' (Assert-BlackwellAuxiliaryAssets -WeightsPath $weightsPath)
  $gates['network_denied_boundary'] = New-Gate 'network_denied_boundary' 'configured' (Assert-NetworkDeniedBoundary)
  $gates['fixture_image'] = New-Gate 'fixture_image' 'passed' @{ path = $fixtureImage; size_bytes = (Get-Item -LiteralPath $fixtureImage).Length; sha256 = (Get-FileSha256 -Path $fixtureImage) }

  $script:WorkRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pixal3d-blackwell-validation-" + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Force -Path $script:WorkRoot | Out-Null
  $script:TempExtension = Join-Path $script:WorkRoot 'extension'
  Copy-ExtensionTree -Source $script:RepoRoot -Destination $script:TempExtension
  Assert-PathContained -Path $script:TempExtension -Root $script:WorkRoot | Out-Null

  $candidateWheelhouse = Resolve-CandidateWheelhouseArchive -CandidateArtifact $candidateArtifact -WorkRoot $script:WorkRoot
  $tempExtension = $script:TempExtension
  $wheelhouseSha256 = Get-FileSha256 -Path $candidateWheelhouse
  $wheelhouseSize = (Get-Item -LiteralPath $candidateWheelhouse).Length
  $manifestPatch = Add-TemporaryBlackwellManifestAsset -TempExtension $script:TempExtension -WheelhouseArchive $candidateWheelhouse -WheelhouseSha256 $wheelhouseSha256 -WheelhouseSizeBytes $wheelhouseSize
  $gates['candidate_wheelhouse'] = New-Gate 'candidate_wheelhouse' 'passed' @{ path = $candidateWheelhouse; sha256 = $wheelhouseSha256; size_bytes = $wheelhouseSize; temp_manifest_asset = $manifestPatch }

  $setupPayload = [ordered]@{
    ext_dir = $tempExtension
    cuda_version = 128
    gpu_sm = 120
  } | ConvertTo-Json -Compress
  $setup1 = Invoke-SetupRepair -Attempt 1 -TempExtension $script:TempExtension -PayloadJson $setupPayload -PythonExe $PythonExe -EvidenceRoot $script:EvidenceRoot
  $setup2 = Invoke-SetupRepair -Attempt 2 -TempExtension $script:TempExtension -PayloadJson $setupPayload -PythonExe $PythonExe -EvidenceRoot $script:EvidenceRoot
  $gates['setup_repair_idempotent'] = New-Gate 'setup_repair_idempotent' 'passed' @{ attempts = @($setup1, $setup2); payload = ($setupPayload | ConvertFrom-Json) }

  $venvPython = Get-VenvPython -TempExtension $script:TempExtension
  # Exact metadata validation command: -m pip check
  $pipCheck = Invoke-JsonCommand -Command @($venvPython, '-m', 'pip', 'check') -WorkingDirectory $script:TempExtension -LogPath (Join-Path $script:EvidenceRoot 'pip-check')
  if ($pipCheck.returncode -ne 0) { throw "pip_check_failed: see $($pipCheck.stdout_path) and $($pipCheck.stderr_path)" }
  $gates['pip_check'] = New-Gate 'pip_check' 'passed' @{ stdout_path = $pipCheck.stdout_path; stderr_path = $pipCheck.stderr_path }

  $workspaceDir = Join-Path $script:WorkRoot 'workspace'
  $outputDir = Join-Path $workspaceDir 'Workflows'
  New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
  $helper = Join-Path $script:TempExtension 'tools\validation\blackwell_real_generation.py'
  $helperArgs = @(
    $helper, 'validate-real',
    '--candidate-artifact', $candidateArtifact,
    '--expected-artifact-sha256', $ExpectedArtifactSha256.ToLowerInvariant(),
    '--expected-artifact-size-bytes', [string]$ExpectedArtifactSizeBytes,
    '--candidate-wheelhouse-archive', $candidateWheelhouse,
    '--candidate-wheelhouse-sha256', $wheelhouseSha256,
    '--candidate-wheelhouse-size-bytes', [string]$wheelhouseSize,
    '--extension-dir', $script:TempExtension,
    '--model-dir', $weightsPath,
    '--workspace-dir', $workspaceDir,
    '--fixture-image', $fixtureImage,
    '--output-dir', $outputDir,
    '--evidence-dir', $script:EvidenceRoot,
    '--work-root', $script:WorkRoot,
    '--venv-python', $venvPython,
    '--source-commit', $SourceCommit,
    '--github-run-id', $GitHubRunId,
    '--github-run-url', $GitHubRunUrl
  )
  $helperCommand = @($venvPython) + $helperArgs
  $helperResult = Invoke-JsonCommand -Command $helperCommand -WorkingDirectory $script:TempExtension -LogPath (Join-Path $script:EvidenceRoot 'real-generation')
  if ($helperResult.returncode -ne 0) { throw "real_generation_failed: helper returned $($helperResult.returncode); see $($helperResult.stdout_path), $($helperResult.stderr_path), and blackwell-validation.json" }
  $gates['real_generation_helper'] = New-Gate 'real_generation_helper' 'passed' @{ stdout_path = $helperResult.stdout_path; stderr_path = $helperResult.stderr_path }
  $evidence.status = 'passed'
} catch {
  $evidence.status = 'failed'
  $evidence.failure_code = 'orchestrator_failed'
  $evidence.message = $_.Exception.Message
  throw
} finally {
  if ($null -ne $script:EvidenceRoot) {
    $orchestratorPath = Join-Path $script:EvidenceRoot 'blackwell-orchestrator.json'
    $evidence | ConvertTo-Json -Depth 100 | Set-Content -Encoding UTF8 -LiteralPath $orchestratorPath
  }
  if (-not $KeepWorkDir -and $null -ne $script:WorkRoot -and (Test-Path -LiteralPath $script:WorkRoot)) {
    $workRoot = $script:WorkRoot
    Remove-Item -LiteralPath $workRoot -Recurse -Force
  }
}
