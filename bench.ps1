# Launcher for Windows: finds uv or fetches it into .\.bench\uv, then runs bench.py.
# No admin rights, no PATH changes.
$ErrorActionPreference = 'Stop'

Set-Location -LiteralPath $PSScriptRoot

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
exit $LASTEXITCODE
