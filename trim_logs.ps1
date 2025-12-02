$logDir = "runs_swin\logs"

# Get all .log files in the directory
Get-ChildItem -Path $logDir -Filter *.log | ForEach-Object {
    $file = $_.FullName
    Write-Host "Processing $file..."

    # Read all lines
    $lines = Get-Content $file

    # Skip first 19 lines
    $trimmed = $lines | Select-Object -Skip 19

    # Overwrite the file with trimmed content
    Set-Content -Path $file -Value $trimmed
}