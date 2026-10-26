param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')]
    [string]$RunId
)

# Downloads results only. It neither launches nor cancels an experiment.
$ErrorActionPreference = 'Stop'
$hasboostVaultRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$hasboostRunOutput = Join-Path $hasboostVaultRoot "results/a100/runs/$RunId"
New-Item -ItemType Directory -Path $hasboostRunOutput -Force | Out-Null
$hasboostRemoteOutput = "hasboost-a100:/root/.orx/runs/$RunId/repo/results/compactness"
& scp -r $hasboostRemoteOutput $hasboostRunOutput
if ($LASTEXITCODE -ne 0) { throw 'Failed to download server results' }
& orx logs $RunId --bytes 1000000 | Out-File -LiteralPath (Join-Path $hasboostRunOutput 'run.log') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'Failed to download OpenResearch log' }
Write-Output $hasboostRunOutput
