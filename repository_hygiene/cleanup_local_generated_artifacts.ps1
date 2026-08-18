param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspacePrefix = $workspace.TrimEnd('\') + '\'
$manifest = Join-Path $PSScriptRoot "local_cleanup_manifest.csv"
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Cleanup manifest not found: $manifest"
}

$rows = Import-Csv $manifest | Where-Object { $_.action -eq "delete_local" }
if (-not $rows) { throw "Cleanup manifest contains no delete_local entries." }

foreach ($row in $rows) {
    $target = [System.IO.Path]::GetFullPath($row.absolute_path)
    if (-not $target.StartsWith(
            $workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing target outside workspace: $target"
    }
    $relative = $target.Substring($workspacePrefix.Length)
    if ($target -eq $workspace -or $target -eq (Join-Path $workspace ".git")) {
        throw "Refusing broad/destructive target: $target"
    }
    if (Test-Path -LiteralPath $target) {
        if ($Execute) {
            Remove-Item -LiteralPath $target -Recurse -Force
            Write-Output "REMOVED $relative"
        } else {
            Write-Output "DRY-RUN $relative ($($row.bytes) bytes)"
        }
    }
}

if (-not $Execute) {
    Write-Output "Dry run only. Re-run with -Execute after reviewing the CSV manifest."
}
