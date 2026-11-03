# Launcher for Windows: finds uv or fetches it into .\.bench\uv, then runs bench.py.
# No admin rights, no PATH changes. Everything it sets (environment variables, current
# directory) is put back before it ends, also after an error or Ctrl+C, so the user's
# shell stays as it was: variables that did not exist before are removed again.
$ErrorActionPreference = 'Stop'

# Snapshot of the variables this script or the uv installer may touch.
$watched = @('BENCH_CWD', 'PYTHONUNBUFFERED') + @(Get-ChildItem Env: | Where-Object { $_.Name -like 'UV_*' } |
    ForEach-Object { $_.Name }) + @('UV_INSTALL_DIR', 'UV_NO_MODIFY_PATH')
$saved = @{}
foreach ($name in ($watched | Select-Object -Unique)) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

# `--engines ollama,llamacpp` typed without quotes arrives as an array; join it again.
$forward = @(foreach ($argument in $args) {
    if ($argument -is [array]) { $argument -join ',' } else { $argument }
})

# Relative paths the user gives (--output, --mirror, ...) mean the directory bench was
# started in, not the repository: hand it over before changing into the repository.
$exitCode = 1
$startDirectory = (Get-Location -PSProvider FileSystem).ProviderPath
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:BENCH_CWD = $startDirectory
    $env:PYTHONUNBUFFERED = '1'  # progress lines appear live in logs
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    $localUv = Join-Path $PSScriptRoot '.bench\uv\uv.exe'

    if ($uvCommand) {
        $uv = $uvCommand.Source
    } elseif (Test-Path -LiteralPath $localUv) {
        $uv = $localUv
    } else {
        Write-Host 'uv not found, installing it into .\.bench\uv (official installer from astral.sh)'
        New-Item -ItemType Directory -Force -Path (Split-Path $localUv) | Out-Null
        $env:UV_INSTALL_DIR = Split-Path $localUv
        $env:UV_NO_MODIFY_PATH = '1'
        # Windows PowerShell 5.1 turns every stderr line of a native program into an error
        # record when the output is redirected; with 'Stop' that would end the script.
        $ErrorActionPreference = 'Continue'
        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        } finally {
            $ErrorActionPreference = 'Stop'
        }
        $uv = $localUv
    }

    # Same reason as above: uv prints progress to stderr. Success is the exit code only.
    $ErrorActionPreference = 'Continue'
    try {
        & $uv run --no-project --python 3.12 bench.py @forward
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
    }
} finally {
    # Put back what was there, remove what was not (also anything new named UV_*).
    foreach ($item in @(Get-ChildItem Env: | Where-Object { $_.Name -like 'UV_*' })) {
        if (-not $saved.ContainsKey($item.Name)) { $saved[$item.Name] = $null }
    }
    foreach ($name in $saved.Keys) {
        if ($null -eq $saved[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item -LiteralPath "Env:$name" -Value $saved[$name]
        }
    }
    Pop-Location
}
exit $exitCode
