# Launcher for Windows: finds uv or fetches it into .\.bench\uv, then runs bench.py.
# No admin rights, no PATH changes. Everything it sets (environment variables, current
# directory) is put back before it ends, so the user's shell stays as it was.
$ErrorActionPreference = 'Stop'

# Relative paths the user gives (--output, --mirror, ...) mean the directory bench was
# started in, not the repository: hand it over before changing into the repository.
$saved = @{}
foreach ($name in 'BENCH_CWD', 'UV_INSTALL_DIR', 'UV_NO_MODIFY_PATH') {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$exitCode = 1
$startDirectory = (Get-Location -PSProvider FileSystem).ProviderPath
Push-Location -LiteralPath $PSScriptRoot
$env:BENCH_CWD = $startDirectory
try {
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
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $uv = $localUv
    }

    & $uv run --no-project --python 3.12 bench.py @args
    $exitCode = $LASTEXITCODE
} finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
    Pop-Location
}
exit $exitCode
