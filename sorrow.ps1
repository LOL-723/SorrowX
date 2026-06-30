$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Project virtual environment not found: $ProjectRoot\.venv"
}

$env:PYTHONPATH = (
    (Join-Path $ProjectRoot "src") +
    [System.IO.Path]::PathSeparator +
    (Join-Path $ProjectRoot "src\core") +
    [System.IO.Path]::PathSeparator +
    $env:PYTHONPATH
)
& $PythonExe -m cli.main @args
exit $LASTEXITCODE
